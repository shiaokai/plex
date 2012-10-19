function outWords=wordNms(words,varargin)
% Word-level non maximal suppression
%
% USAGE
%  [outWords] = wordNms( words, varargin )
%
% INPUTS
%  words          - word structure before suppression
%  varargin       - additional params
%   .type         - ['none'] NMS type (currently just 'none' or 'maxg')
%   .thr          - [-inf] word threshold
%   .overlap      - [.5] overlap threshold
%   .overDnm      - ['min'] area of overlap denominator ('union' or 'min')
%   .clf          - [] SVM classifier
%
% OUTPUTS
%  outWords    - rescored and thresholded words
%
% NOTES
%  incoming scores: small=good, big=bad
%  outgoing scores: small=bad, big=good
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dfs={'type','none','thr',-inf,'overlap',.5,'ovrDnm','min','clf',[]};
[type,thr,overlap,ovrDnm,clf]=getPrmDflt(varargin,dfs,1);
if(isempty(words)), outWords=[]; return; end

% SVM rescore
if(~isempty(clf) && isfield(words(1),'bbs')) 
  x=zeros(length(words),size(clf{3}.SVs,2)); 
  for j=1:length(words), x(j,:)=computeWordFeatures(words(j)); end
  x=bsxfun(@minus,x,clf{1}); x=bsxfun(@times,x,1./clf{2});
  [~,~,py]=svmpredict(zeros(size(x,1),1),x,clf{3});
  py=py*clf{3}.Label(1);
  for j=1:length(words), words(j).bb(:,5)=py(j); end
end

if(~strcmp(type,'none'))
  % word NMS
  wbb=reshape([words.bb],5,[])'; 
  assert(all(isfinite(wbb(:,5))));
  kp=nms1(wbb,overlap,ovrDnm,strcmp(type,'maxg'));
  words=words(kp);
  wbb1=reshape([words.bb],5,[])';
  outWords=words(wbb1(:,5)>thr);
else
  wbb1=reshape([words.bb],5,[])';
  outWords=words(wbb1(:,5)>thr);
end
end

function [kp,rat]=nms1(bbs,overlap,ovrDnm,greedy)
[~,ord]=sort(bbs(:,5),'descend'); bbs=bbs(ord,:); n=size(bbs,1);
O=(compOas(bbs(:,1:4),bbs(:,1:4),strcmp(ovrDnm,'union'))>overlap)-eye(n);

kp=true(n,1); rat=cell(n,1); [rat{:}]=deal(inf);
for i=1:n
  if(~kp(i) && greedy), continue; end
  nbrs=O(i,:)>0; if(sum(nbrs)==0), continue; end
  v=max(bbs(nbrs,5)); 
  if(~isempty(v)), rat{i}=bbs(i,5)-v; end
  kp(nbrs & (1:n)>i)=false;
end
kp(ord)=kp; rat(ord)=rat;
end

function oa = compOas(dt,gt,ovrDnm)
m=size(dt,1); n=size(gt,1); oa=zeros(m,n);
de=dt(:,[1 2])+dt(:,[3 4]); da=dt(:,3).*dt(:,4);
ge=gt(:,[1 2])+gt(:,[3 4]); ga=gt(:,3).*gt(:,4);
for i=1:m
  for j=1:n
    w=min(de(i,1),ge(j,1))-max(dt(i,1),gt(j,1)); if(w<=0), continue; end
    h=min(de(i,2),ge(j,2))-max(dt(i,2),gt(j,2)); if(h<=0), continue; end
    t=w*h; if(ovrDnm), u=da(i)+ga(j)-t; else u=min(da(i),ga(j)); end
    oa(i,j)=t/u;
  end
end
end