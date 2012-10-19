function [gt,dt,gtw,dtw,files1] = evalReading(gtDir, dtDir, varargin)
% Process output
%
% USAGE
%  [gt,dt,gtw,dtw,files1] = evalReading( gtDir, dtDir, kVals, varargin )
%
% INPUTS
%  gtDir           - directory for groundtruth annotations
%  dtDir           - directory of precomputed outputs
%  kVals           - vector of distractor values
%  varargin        - additional parameters
%   .thr           - [.5] overlap requirement for match
%   .mul           - [0] can match multiple groundtruths
%   .resize        - {} resize factor for detected bounding boxes
%   .f0            - [1] start offset
%   .f1            - [inf] end index
%   .imDir         - [''] directory for images
%   .lexDir        - [''] directory for lexicon files
%   .pNms          - additional word-level NMS params
%     .type        - ['none'] NMS type (currently just 'none' or 'maxg')
%     .thr         - [-inf] word threshold
%   .ocr         - [0] correct OCR output
%
% OUTPUTS
%  gt              - ground truth bounding boxes and if found match
%  dt              - detected bounding boxes if found match
%  gtw             - ground truth word objects
%  dtw             - detected word objects
%  files1          - paths of the image files
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dfs={'thr',.5,'mul',0,'resize',{},'f0',1,'f1',inf,'imDir','',...
  'lexDir','','pNms',struct('type','none','thr',-inf),'ocr',0};
[thr,mul,rs,f0,f1,imDir,lexDir,pNms,ocr]=getPrmDflt(varargin,dfs,1);
if(isempty(imDir)), imDir=gtDir; end

% collect all strings in ground truth
files=dir(fullfile(gtDir,'*.txt')); files={files.name};
files=files(f0:min(f1,end)); n=length(files); assert(n>0);

% get files in ground truth directory
gt=cell(1,n); dt=cell(1,n); dtw=dt; gtw=gt;
ticId=ticStatus('evaluating');

% loop over images
for i=1:n
  gtNm=fullfile(gtDir,files{i});
  gt1=bbGt('bbLoad',gtNm);
  [gtWords,gtInds]=filterValidGt(gt1); gt1=gt1(gtInds);
  dtNm=fullfile(dtDir,[files{i}(1:end-8),'.mat']);
  if(~exist(dtNm,'file')), dta=[]; else res=load(dtNm); dta=res.words; end
  if(isempty(lexDir)), error('Lexicon directory is empty.'); end
  
  % load lexicon
  fid=fopen(fullfile(lexDir,files{i}),'r');
  lexS=textscan(fid,'%s'); lexS=unique(lexS{1}');
  fclose(fid);

  % filter/spell-check detections
  if(isempty(dta)), dt1=[]; else
    if(ocr), 
      dt1=spellCheck(dta,lexS);
      for j=1:length(dt1)
        dt1(j).bb=bbApply('resize',dt1(j).bb,.75,.75); 
      end
    else
      dt1=dta(ismember(upper({dta.word}),upper(lexS)));
    end
  end
  
  % flip signs of word scores
  for j=1:length(dt1), dt1(j).bb(:,5)=-dt1(j).bb(:,5); end
  
  % word nonmax suppr
  dt1=wordNms(dt1,pNms);
  files1{i}=fullfile(imDir,files{i}(1:end-4));

  % evaluate detections
  [gt2,dt2,gtw2,dtw2] = evalReading1(gt1,dt1,thr,mul);
  gt{1,i}=gt2; dt{1,i}=dt2; dtw{1,i}=dtw2; gtw{1,i}=gtw2;
  
  tocStatus(ticId,i/n);
end

end

function [gt, dt, gt0, dt0] = evalReading1( gt0, dt0, thr, mul )
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





