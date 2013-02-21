function writeAllImgs(I,y,lbls,d,ctr)
% Helper function to write all images in a matrix by class label
%
% USAGE
%  writeAllImgs( I, y, lbls, d, ctr )
%
% INPUTS
%  I       - image matrix
%  y       - class labels for matrix
%  lbls    - mapping from class id number to label
%  d       - base directory
%  ctr     - [0] write index offset
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

if nargin<5, ctr=0; end
ticId=ticStatus('writing images',1,30,1);
for k=1:length(lbls)
  lbl=lbls{k}; if(lbl>=97&&lbl<=122), lbl=['-' lbl]; end %#ok<AGROW>
  if(lbl=='_'), continue; end
  dd=fullfile(d,lbl); if(~exist(dd,'dir')), mkdir(dd); end
  I1=I(:,:,:,y==k); 
  if(size(I1,4)>1), imwrite2(uint8(I1),1,ctr,dd); 
  elseif(size(I1,4)==1), imwrite2(uint8(I1),0,ctr,dd);
  end
  tocStatus(ticId,k/length(lbls));
end
end