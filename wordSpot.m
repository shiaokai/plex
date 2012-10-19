function [words,t1,t2,bbs]=wordSpot(I,lexS,fModel,wordSvm,nmsPrms,frnPrms,plxPrms)
% Function for End-to-end word spotting function
%
% Full description can be found in: 
%   "End-to-end Scene Text Recognition," 
%    K. Wang, B. Babenko, and S. Belongie. ICCV 2011
%
% USAGE
%  [words1,words] = wordSpot( I, lexS )
%
% INPUTS
%   I        - input image
%   lexS     - input lexicon, comma-separated string
%   fModel   - trained Fern character classifier
%   wordSvm  - trained Svm word classifier
%   nmsPrms  - character-level non max suppression parameters (see bbNms.m)
%   frnPrms  - fern parameters (see charDet.m)
%   plxPrms  - plex parameters (see wordDet.m)
%
% OUTPUTS
%   words      - array of word objects with no threshold
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

[dPath,ch,ch1,chC,chClfNm,dfNP]=globals;

if nargin<3, error('not enough params'); end
if ~exist('wordSvm','var'), wordSvm={}; end
if (~exist('nmsPrms','var') || isempty(nmsPrms)), nmsPrms=dfNP; end
if ~exist('frnPrms','var'), frnPrms={}; end
if ~exist('plxPrms','var'), plxPrms={}; end

% construct trie
lex=wordDet('build',lexS);
% run character detector (Ferns)
t1S=tic; bbs=charDet(I,fModel,frnPrms); t1=toc(t1S);
% upper and lower case are equivalent
bbs(:,6)=equivClass(bbs(:,6),ch);
% character NMS
bbs=bbNms(bbs,nmsPrms);

% run word detector (PLEX)
t2S=tic; words=wordDet('plexApply',bbs,ch1,lex,plxPrms); t2=toc(t2S);
if ~isempty(wordSvm)
  % if available, score using SVM
  words=wordNms(words,wordSvm);
end

