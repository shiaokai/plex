function preprocessData
% Pre-process all the data from raw sources. Directions for how to download
% things are in the README
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

prepIcdar; % NOTE: An error is expected on image I00797. There is a missing
           % character level bounding box in the word.

prepSvt;
prepMsrc;

genLexIcdar;