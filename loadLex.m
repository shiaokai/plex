function [allwords,lex]=loadLex(tstDir)
% Read in all ground truth words from a test directory
%
% USAGE
%  [allwords,lex] = loadLex( tstDir )
%
% INPUTS
%  tstDir          - directory path
%
% OUTPUTS
%  allwords        - cell array of all words
%  lex             - trie structure
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

allwords=cell(0);
for k=0:length(dir(fullfile(tstDir,'wordAnn','*.txt')))-1;
  gt=bbGt('bbLoad',fullfile(tstDir,'wordAnn',...
    sprintf('I%05i.jpg.txt',k)));
  for j=1:length(gt)
    if(~checkValidGt(gt(j).lbl)), continue; end
    allwords{end+1}=gt(j).lbl;
  end
end
if nargout==2, lex=wordDet('build',allwords); end
end
