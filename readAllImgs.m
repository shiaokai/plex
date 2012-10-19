function [I,yC]=readAllImgs(d,lbls,maxn,bgd,maxbg)
% Helper function to read all images from a directory into a matrix
%
% USAGE
%  [I,yC] = readAllImgs( d, lbls, maxn, bgd, maxbg )
%
% INPUTS
%  d       - base directory
%  lbls    - cell array of class labels
%  maxn    - max number of images per class to be read
%  bgd     - background class directory
%  maxbg   - max number of images for background class to be read
%
% OUTPUTS
%  I       - array of read images
%  yC      - labels for images
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

if(nargin<3), maxn=inf; end
if(nargin<4), bgd=[]; end
if(nargin<5), maxbg=maxn; end
I=[]; yC=zeros(1e5,1); k0=1;
ticId=ticStatus('reading images',1,30,1);
for k=1:length(lbls), lbl=lbls{k};    
    if(upper(lbl)~=lbl), lbl=['-',lbl]; end %#ok<AGROW>
    if(lbl=='_'), lbl='charBg'; dd=fullfile(bgd,lbl); maxi=maxbg;
    else dd=fullfile(d,lbl); maxi=maxn; end    
    if(~exist(dd,'dir') || size(dir(fullfile(dd,'*.png')),1) == 0)
      fprintf(1,'Warning: directory %s empty/non-existant',dd); continue; 
    end 
    I1=imwrite2([],1,0,dd); n1=size(I1,4);

    if(n1>maxi) 
      rids=randSample(n1,maxi); n1=length(rids); I1=I1(:,:,:,rids);
    end
    if(k==1), I=zeros([size(I1,1),size(I1,2),3,5e4],'uint8'); end
    I(:,:,:,k0:k0+n1-1)=I1; yC(k0:k0+n1-1)=k;
    k0=k0+n1; tocStatus(ticId,k/length(lbls));
end
I=I(:,:,:,1:k0-1); yC=yC(1:k0-1);
fprintf('\n');
end