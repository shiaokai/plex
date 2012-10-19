function H = hogOld( I, binSize, nOrients)
% A wrapper for Piotr Dollar's HOG that clips the cells at the edges, as
% was done by his toolbox prior to version 3.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

if( nargin<2 ), binSize=8; end
if( nargin<3 ), nOrients=9; end
H = hog(I, binSize, nOrients);
H = H(2:end-1, 2:end-1, :);