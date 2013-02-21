function [gtOut,inds]=filterValidGt(gt)
% For simplicity, filter out words that are fewer than three
% characters and 
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

inds=false(1,length(gt));
for i=1:length(gt), inds(i)=checkValidGt(gt(i).lbl)>0; end
gtOut=gt(inds);
end