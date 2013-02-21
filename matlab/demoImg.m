function demoImg
% Demo of PLEX running on an image in the data folder
%
% USAGE
%  demoImg
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

I=imread(fullfile('data','demo.jpg'));
lexS={'michaels','world','market','fitness'};

im(I); drawnow;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load classifiers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% word threshold
% character fern
%%%% clfPath=fullfile('data','fern_synth.mat');
clfPath='/data/text/plex/synth1000/clfs/fern_inlineS6M256trnSetsynth1000trnTcharbgDirmsrcnBg5000nTrnInf.mat';

if(~exist(clfPath,'file')), error('FERN DOES NOT EXIST?!\n'); end
dat=load(clfPath); fModel=dat.fModel;

% word svm
%%%%%svmPath=fullfile('data','svm_svt.mat');
svmPath='/data/text/plex/synth1000/clfs/fern_inlineS6M256trnSetsynth1000trnTcharbgDirmsrcnBg5000nTrnInf_svt_wclf.mat';
if(~exist(svmPath,'file')), error('SVM MODEL DOES NOT EXIST?!\n'); end
dat=load(svmPath); wdClf=dat.wdClf; alpha=dat.alpha; wdClf.thr=-1;
%wordSvm=model.pNms1; wordSvm.thr=wordThr;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run word recognition (PLEX)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic; words=wordSpot(I,lexS,fModel,wdClf,{},{'minH',.04},{'alpha',alpha}); toc

wordDetDraw( words, 1, 1, 1, [0 1 0] );

