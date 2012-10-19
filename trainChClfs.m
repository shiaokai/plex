function trainChClfs
% Train character classifiers (FERNS)
%
% This function first trains FERNS using images for each character class
% and their provided background class. It also does another round of
% training after bootstrapping more negative examples.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

[dPath,ch,ch1,chC,chClfNm]=globals;
% parameters that pretty much won't change
sBin=8; oBin=8; chH=48;
S=6; M=256; thrr=[0 1]; nTrn=Inf;
cHogFtr=@(I)reshape((5*hogOld(imResample(single(I),[chH,chH]),sBin,oBin)),[],1);
cFtr=cHogFtr;

% Train character detectors specified in the param list
% paramSet={train dataset,with/without neighboring chars,
%           bg dataset,# bg images,bootstrap}
paramSets={{'synth','charHard','msrc',5000,1},...
           {'icdar','charHard','icdar',5000,1}};
           

fprintf('Training character classifiers.\n');
% Loop over param sets
for p=1:length(paramSets)
  paramSet=paramSets{p};
  trnD=paramSet{1}; trnT=paramSet{2}; trnBg=paramSet{3}; 
  nBg=paramSet{4}; bs=paramSet{5};
  cDir=fullfile(dPath,trnD,'clfs');
  clfPrms={'S',S,'M',M,'trnT',trnT,'bgDir',trnBg,'nBg',...
    nBg,'nTrn',nTrn};
  cNm=chClfNm(clfPrms{:}); clfPath=fullfile(cDir,[cNm,'.mat']);
  newBg=[trnBg,'Bt'];

  fprintf('Working on: ');
  disp(paramSet);
  
  % load training images
  RandStream.getDefaultStream.reset();
  [I,y]=readAllImgs(fullfile(dPath,trnD,'train',trnT),chC,nTrn,...
    fullfile(dPath,trnBg,'train'),nBg);
  x=fevalArrays(I,cFtr)';
  % train char classifier
  RandStream.getDefaultStream.reset();
  [ferns,yh]=fernsClfTrain(double(x),y,struct('S',S,'M',M,'thrr',thrr,'bayes',1));
  fprintf('training error=%f\n',mean(y~=yh));
  if(~exist(cDir,'dir')),mkdir(cDir); end
  save(clfPath,'ferns','sBin','oBin','chH');
  fModel=[]; fModel.ferns=ferns; fModel.sBin=sBin; fModel.oBin=oBin; 
  fModel.chH=chH;

  % bootstrap classifier if flag is on
  if ~bs, continue; end
  
  % copy base bg folder to new bootstrap folder
  fullBgD=fullfile(dPath,trnBg,'train','charBg');
  fullNewBgD=fullfile(dPath,newBg,'train','charBg');
  if(exist(fullNewBgD,'dir')), 
    fprintf('Clearing out old hardnegative folder');
    rmdir(fullNewBgD,'s'); 
  end  
  mkdir(fullNewBgD);
  copyfile(fullBgD,fullNewBgD);

  maxn=100; w=length(dir(fullfile(fullNewBgD,'*png'))); %<- starting index
  files=dir(fullfile(dPath,trnBg,'train','images','*.jpg')); files={files.name};
  filesAnn=dir(fullfile(dPath,trnBg,'train','wordAnn','*.txt')); filesAnn={filesAnn.name};
  % bootstrap
  ticId=ticStatus('Mining hard negatives',1,30,1);
  for f=1:length(files), 
    I=imread(fullfile(dPath,trnBg,'train','images',files{f}));
    if ~isempty(filesAnn)
      gtBbs=bbGt('bbLoad',fullfile(dPath,trnBg,'train','wordAnn',filesAnn{f}));
      gtBbs1=reshape([gtBbs.bb],4,[])';
      gtBbs1=[gtBbs1, zeros(size(gtBbs1,1),1)];
    else
      gtBbs1=[];
    end

    bbs=charDet(I,fModel,{'thr',0,'minH',.1});
    bbs(:,6)=equivClass(bbs(:,6),ch);
    bbs=bbNms(bbs,'thr',0,'separate',0,'type','maxg',...
      'resize',{1,1},'ovrDnm','union','overlap',.2,'maxn',inf);

    if(isempty(bbs)), continue; end
    P=bbGt('sampleWins',I,{'bbs',bbs,'n',maxn,'dims',[100 100],'ibbs',...
      gtBbs1,'thr',.01});
    if(isempty(P)), continue; end
    P=cell2array(P);
    imwrite2(P,size(P,4)>1,w,fullNewBgD);
    w=w+size(P,4);
    tocStatus(ticId,f/length(files));
  end

  % re-train again
  nBtBg=2*nBg;
  clfPrms={'S',S,'M',M,'trnT',trnT,'bgDir',newBg,...
    'nBg',nBtBg,'nTrn',nTrn};
  cNm=chClfNm(clfPrms{:});
  RandStream.getDefaultStream.reset();
  [I,y]=readAllImgs(fullfile(dPath,trnD,'train',trnT),chC,nTrn,...
    fullfile(dPath,newBg,'train'),nBtBg);
  x=fevalArrays(I,cFtr)';
  % train char classifier
  RandStream.getDefaultStream.reset();
  [ferns,yh]=fernsClfTrain(double(x),y,struct('S',S,'M',M,'thrr',thrr,'bayes',1));
  fprintf('training error=%f\n',mean(y~=yh));
  if(~exist(cDir,'dir')),mkdir(cDir); end
  save(fullfile(cDir,cNm),'ferns','sBin','oBin','chH');
end

end