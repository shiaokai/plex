function bbs=charDet(I,fModel,varargin)
% Multi-scale sliding window character detection using Ferns
%
% USAGE
%  bbs = charDet( I, fModel, varargin)
%
% INPUTS
%  I           - image
%  fModel      - fern object
%  varargin    - additional params
%   .ss        - [2^(1/4)] scale step
%   .minH      - [.04] min sliding window size ratio
%   .maxH      - [1] max sliding window size ratio
%   .thr       - [-75] character detection threshold
%
% OUTPUTS
%  bbs         - matrix of bounding box output: location, scale, score
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dfs={'ss',2^(1/4),'minH',.04,'maxH',1,'thr',-75};
[ss,minH,maxH,thr] = getPrmDflt(varargin,dfs,1);
ferns=fModel.ferns;
sBin=fModel.sBin; oBin=fModel.oBin; sz=[fModel.chH,fModel.chH];
hImg=size(I,1); wImg=size(I,2); k=0;

bbs=zeros(1e6,6); 
minHP=minH*min(size(I,1),size(I,2)); maxHP=maxH*min(size(I,1),size(I,2));
sStart=ceil(max(log(sz(1)/maxHP),log(sz(2)/wImg))/log(ss));
sEnd=floor(log(sz(1)/minHP)/log(ss));
for s=sStart:sEnd, r=ss^s;
  if(s==0), I1=I; else I1=imResample(I,[round(hImg*r),round(wImg*r)]); end
  bbs1=detect1(I1,ferns,sz,sBin,oBin,thr); 
  if(isempty(bbs1)), continue; end
  bbs1(:,1:4)=bbs1(:,1:4)/r; 
  k1=size(bbs1,1); bbs(k+1:k+k1,:)=bbs1; k=k+k1;
end
bbs=bbs(1:k,:);
end

function bbs=detect1(I,ferns,sz,sBin,oBin,thr)
x=5*hogOld(single(I),sBin,oBin); xs=size(x); 
%sz1=sz/sBin; % uncomment to use the updated HOG
sz1=sz/sBin-2;
x=fevalArrays(x,@im2col,sz1); if(ndims(x)==2),x=permute(x,[1 3 2]); end
x=permute(x,[1 3 2]); 
x=reshape(x,[],size(x,3))';
xinds=fernsInds(double(x),ferns.fids,ferns.thrs);
[~,ph]=fernsClfApply([],ferns,xinds);
ph=reshape(ph,xs(1)-sz1(1)+1,xs(2)-sz1(2)+1,[]);
bbs=zeros(numel(ph),6);
if(size(ph,3)==63), ph=bsxfun(@minus,ph(:,:,1:62),ph(:,:,63)); end

k=0; bbw=sz1(1)+2; bbh=sz1(2)+2;
for j=1:size(ph,3), M=ph(:,:,j);
  ind=find(M>thr); sub=ind2sub2(size(M),ind); 
  if(isempty(sub)); continue; end
  bbs1=[fliplr(sub) sub]; bbs1(:,5)=M(ind);
  bbs1(:,1)=bbs1(:,1)-floor(bbw/2); bbs1(:,3)=bbw;
	bbs1(:,2)=bbs1(:,2)-floor(bbh/2); bbs1(:,4)=bbh;
  
  k1=size(bbs1,1); bbs(k+1:k+k1,1:5)=bbs1; bbs(k+1:k+k1,6)=j;
  k=k+k1;
end
bbs=bbs(1:k,:); if(k<1), return; end
bbs(:,1:2)=bbs(:,1:2)+2; 
bbs(:,1:4)=bbs(:,1:4)*sBin; bbs(:,1:2)=bbs(:,1:2)+0.5;
end