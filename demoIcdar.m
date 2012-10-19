function demoIcdar(idx)
% Demo of PLEX running on an image from the ICDAR dataset.
%
% USAGE
%  demoIcdar( idx )
%
% INPUTS
%   idx        - filenumber to test: 1-249
%
% EXAMPLE
%   demoIcdar(23); litter,colchester,borough [rt=~20s]
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dPath=globals;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load image and request lexicon
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I=imread(fullfile(dPath,'icdar','test','images',sprintf('I%05i.jpg',idx)));
im(I); lexIn=input('Enter comma-separated strings for lexicon:','s');
lexS=textscan(lexIn,'%s','Delimiter',',')'; lexS=lexS{1}';
lexS=strtrim(lexS);
if isempty(lexS), error('must enter lexicon\n'); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load classifiers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% word threshold
wordThr=-.75;
% character fern
clfPath=fullfile('data','fern_synth.mat');
if(~exist(clfPath,'file')), error('FERN DOES NOT EXIST?!\n'); end
fModel=load(clfPath);
% word svm
svmPath=fullfile('data','svm_icdar.mat');
if(~exist(svmPath,'file')), error('SVM MODEL DOES NOT EXIST?!\n'); end
sModel=load(svmPath); wordSvm=sModel.pNms1; wordSvm.thr=wordThr;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run word recognition (PLEX)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic; words=wordSpot(I,lexS,fModel,wordSvm); toc
wordDetDraw( words, 0, 0, 1, [0 1 0] );
