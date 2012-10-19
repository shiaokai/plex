function [model,xmin,xmax,dtE]=trainRescore(dtw,dt,gtw,nFold,pNms,thr)
% Train SVM to score words
% 
% USAGE
%  [model,xmin,xmax,dtE]=trainRescore( dtw, dt, gtw, nFold, pNms, thr )
%
% INPUTS
%  dtw            - detected word objects
%  dt             - detected bounding boxes
%  gtw            - ground truth word objects
%  nFold          - Number of folds for cross validation
%  pNms           - additional word-level nms params
%   .type         - ['none'] NMS type (currently just 'none' or 'maxg')
%   .thr          - [-inf] word threshold
%  thr            - overlap threshold
%
% OUTPUTS
%  model          - trained SVM object
%  xmin           - min values of every feature
%  xmax           - max values of every feature
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

RandStream.setDefaultStream(RandStream('mrg32k3a', 'Seed', sum('iccv11')));
n=length(gtw); fld=randint2(n,1,[1 nFold]); x=cell(n,1); y=x;
 
for k=1:n, y{k}=zeros(length(dtw{k}),1);
  for j=1:length(dtw{k}), 
    y{k}(j)=2*dt{k}(j,6)-1;
    if(j==1), x{k}=repmat(computeWordFeatures(dtw{k}(1)),length(dtw{k}),1);
      else x{k}(j,:)=computeWordFeatures(dtw{k}(j)); end
  end
  if(isempty(dtw{k})),x{k}=zeros(0,17); end
end
xmin=min(cat(1,x{:}));
x=cellfun(@(x1)bsxfun(@minus,x1,xmin),x,'UniformOutput',0); 
xmax=max(cat(1,x{:})); x=cellfun(@(x1)bsxfun(@times,x1,1./xmax),x,'UniformOutput',0); 

Cs=[5e-1 1e0 5e0 1e1 5e1 1e2 5e2 1e3 5e3]; 
fs=zeros(length(Cs),nFold);
for c=1:length(Cs)
  for f=1:nFold, 
    xtr=cat(1,x{fld~=f}); ytr=cat(1,y{fld~=f});
    fprintf('C=%i, fold=%i, numpos=%i\n',Cs(c),f,sum(ytr==1));
    prm=sprintf('-c %i -t 1 -d 2 -w1 %.3f -w-1 %.3f',Cs(c),1,1);
    model=svmtrain(ytr,xtr,prm);
    fs(c,f)=evalF(dtw(fld==f),gtw(fld==f),model,xmin,xmax,pNms,thr);
    fprintf(1,sprintf('PRMS: %s\nF=%.3f\n',prm,fs(c,f)));
  end
end
disp(fs);
fs=mean(fs,2); [~,i]=max(fs);
xtr=cat(1,x{:}); ytr=cat(1,y{:});
prm=sprintf('-c %i -t 1 -d 2 -w1 %.3f -w-1 %.3f',Cs(i),1,1);
fprintf(1,['TRAINING WITH FINAL PRMS:\n' prm '\n']); 
model=svmtrain(ytr,xtr,prm);
[f,dtE]=evalF(dtw,gtw,model,xmin,xmax,pNms,thr);
fprintf(1,'training fscore=%.3f\n',f);
end

function [f,dt1]=evalF(dt,gt,model,xmin,xmax,pNms,thr)
% run nms on all detections and then compute f-score
pNms.clf={xmin,xmax,model};
dt1=dt; for k=1:length(dt), dt1{k}=wordNms(dt{k},pNms); end
gtE=gt; dtE=dt;
for k=1:length(gt)
  [gtE{k},dtE{k},~,dt1{k}]=evalReading1(gt{k},dt1{k},thr,0);
  for j=1:length(dt1{k}), dt1{k}(j).cor=dtE{k}(j,6); end
end
[xs,ys]=bbGt('compRoc', gtE, dtE, 0); f=Fscore(xs,ys);
end

function [gt, dt, gt0, dt0] = evalReading1( gt0, dt0, thr, mul )
% check inputs
if(nargin<3 || isempty(thr)), thr=.5; end
if(nargin<4 || isempty(mul)), mul=0; end
nd=length(dt0); ng=length(gt0);
for g=1:ng, gt0(g).det=0; gt0(g).read=0; end
if(ng==0), gt=zeros(0,5); else 
  [~,ord]=sort([gt0.ign],'ascend'); gt0=gt0(ord,:);
  gt=reshape([gt0.bb],4,[])'; gt(:,5)=-[gt0.ign];
end
if(nd==0), dt=zeros(0,6); else  
  wbbs=reshape([dt0.bb],5,[])'; [~,ord]=sort(wbbs(:,5),'descend');
  dt0=dt0(ord);
  dt=reshape([dt0.bb],5,[])'; dt(:,6)=0;
end
if(nd==0||ng==0), return; end

% Attempt to match each (sorted) dt to each (sorted) gt
for d=1:nd
  bstOa=thr; bstg=0; bstm=0; % info about best match so far
  for g=1:ng
    % if this gt already matched, continue to next gt
    m=gt(g,5); if( m==1 && ~mul ), continue; end
    % if dt already matched, and on ignore gt, nothing more to do
    if( bstm~=0 && m==-1 ), break; end
    % compute overlap area, continue to next gt unless better match made
    oa=bbGt('compOa',dt0(d).bb(1:4),gt0(g).bb(1:4),m==-1); 
    if(oa<bstOa), continue; end; gt0(g).det=1;
    % word must match, unless the gt is an ignore region
    if(m>=0 && ~strcmpi(gt0(g).lbl,dt0(d).word)), continue; end
    gt0(g).read=1;
    % match successful and best so far, store appropriately
    bstOa=oa; bstg=g; if(m==0), bstm=1; else bstm=-1; end
  end
  % store type of match for both dt and gt
  if(bstm~=0), gt(bstg,5)=bstm; dt(d,6)=bstm; end
end
end
