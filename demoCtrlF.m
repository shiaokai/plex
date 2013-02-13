function demoCtrlF
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

%I=imread(fullfile('data','demo.jpg'));
%I=imread(fullfile('data','bug2.JPG'));
%I=imread(fullfile('data','bug5.jpg'));
I=imread(fullfile('data','01_09.jpg'));
% crop bounding box
im(I); drawnow;
[X,Y]=ginput(2); X=round(X); Y=round(Y);
I=I(min(Y):max(Y),min(X):max(X),:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load classifiers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% character fern
clfPath='/data/text/plex/synth1000/clfs/fern_inlineS6M256trnSetsynth1000trnTcharbgDirmsrcnBg5000nTrnInf.mat';
if(~exist(clfPath,'file')), error('FERN DOES NOT EXIST?!\n'); end
dat=load(clfPath); fModel=dat.fModel;
% word svm
svmPath='/data/text/plex/synth1000/clfs/fern_inlineS6M256trnSetsynth1000trnTcharbgDirmsrcnBg5000nTrnInf_svt_wclf.mat';
if(~exist(svmPath,'file')), error('SVM MODEL DOES NOT EXIST?!\n'); end
dat=load(svmPath); wdClf=dat.wdClf; alpha=dat.alpha; wdClf.thr=-Inf;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run word recognition (PLEX)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Initialization...');
cfg=globals;
% run character detector (Ferns)
t1S=tic; bbs=charDet(I,fModel,{'minH',.05}); tInit=toc(t1S);
% upper and lower case are equivalent
bbs(:,6)=equivClass(bbs(:,6),cfg.ch);
% character NMS
bbs=bbNms(bbs,cfg.dfNP);
fprintf('Done. (took %1.1f s.)\n',tInit);
accepted_words=[]; finished=0;
while 1
  im(I); drawnow;
  % enter string
  lexS=input('Enter a word to find or (Q) to exit:','s');
  if strcmpi(lexS,'Q'), break; end
  if ~isempty(setdiff(lexS,cfg.ch))
    fprintf('Entered an invalid word. Try again.\n');
    continue;
  end
  lex=wordDet('build',{lexS});
  t2S=tic;
  words=wordDet('plexApply',bbs,cfg.ch1,lex,{'alpha',alpha,'mpw',3});
  tSpot=toc(t2S);
  fprintf('Wordspot took %1.1f s.\n',tSpot);
  words=wordNms(words,wdClf);

  counter=0;
  while 1
    im(I); drawnow;
    % initially show top word with a border; all others just a light bg
    for i=1:length(words)
      bb=words(i).bb;
      minY=bb(2); maxY=bb(2)+bb(4);
      minX=bb(1); maxX=bb(1)+bb(3);
      patch([minX,minX,maxX,maxX],[minY,maxY,maxY,minY],...
        'b','FaceAlpha',.1,'EdgeAlpha',.075);
    end
    
    idx=mod(counter,length(words))+1;
    wordDetDraw( words(idx), 1, 1, 1, [0 1 0] );
    cmd=input('Next(n), Accept(a), Skip(s)','s');
    switch cmd
      case 'n'
        counter=counter+1;
      case 'a'
        accepted_words=[accepted_words,words(idx)];
        break;
      case 's'
        break;
    end
  end
end

im(I); drawnow;
wordDetDraw( accepted_words, 1, 1, 1, [0 1 0] );
title('Final annotation.');

end


% 
% 
% 
% tic; words=wordSpot(I,lexS,fModel,wdClf,{},{'minH',.04},{'alpha',alpha}); toc
% 
% wordDetDraw( words, 1, 1, 1, [0 1 0] );

