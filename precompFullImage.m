function precompFullImage
% Run end-to-end PLEX pipeline on the ICDAR and SVT datasets
%
% One MAT file is created for each image to record the results. After all
% the precomp*.m files are complete, run genPrCurves.m to display results.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

[dPath,ch,ch1,chC,chClfNm]=globals;

% fern parameters
S=6; M=256; nTrn=Inf;

% paramSet={train dataset,with/without neighboring chars,
%           bg dataset,# bg images, test dataset, test split}
paramSets={{'synth','charHard','msrcBt',10000,'icdar','test'},...
           {'icdar','charHard','icdarBt',10000,'icdar','test'},...
           {'synth','charHard','msrcBt',10000,'svt','test'},...
           {'icdar','charHard','icdarBt',10000,'svt','test'},...
           {'synth','charHard','msrcBt',10000,'icdar','train'},...
           {'icdar','charHard','icdarBt',10000,'icdar','train'},...
           {'synth','charHard','msrcBt',10000,'svt','train'},...
           {'icdar','charHard','icdarBt',10000,'svt','train'}};
         
for p=1:length(paramSets)
  RandStream.getDefaultStream.reset();
  paramSet=paramSets{p};
  trnD=paramSet{1}; trnT=paramSet{2}; trnBg=paramSet{3}; nBg=paramSet{4}; 
  tstD=paramSet{5}; tstSpl=paramSet{6}; tstDir=fullfile(dPath,tstD,tstSpl);

  fprintf('Working on: ');
  disp(paramSet);
  
  allwords=loadLex(tstDir);
  % set up classifiers
  cDir=fullfile(dPath,trnD,'clfs');
  clfPrms={'S',S,'M',M,'trnT',trnT,'bgDir',trnBg,'nBg',...
    nBg,'nTrn',nTrn};
  cNm=chClfNm(clfPrms{:}); clfPath=fullfile(cDir,[cNm,'.mat']);
  
  % set up output locations
  d1=fullfile(tstDir,['res-' trnD],cNm,'images');
  
  %if(~exist(d1,'dir')), mkdir(d1); end
  if(exist(d1,'dir')), 
    rmdir(d1,'s');
  end
  mkdir(d1);
  save(fullfile(d1,'workspace')); % save all variables up to now
  saveRes=@(f,words,t1,t2,t3)save(f,'words','t1','t2','t3');
  
  % load clfs
  if(~exist(clfPath,'file')), error('FERN DOES NOT EXIST?!\n'); end
  fModel=load(clfPath);
  nImg=length(dir(fullfile(tstDir,'wordAnn','*.txt')));
  
  ticId=ticStatus('Running PLEX on full images',1,30,1);
  for f=0:nImg-1
    sF=fullfile(d1,sprintf('I%05d.mat',f));
    I=imread(fullfile(tstDir,sprintf('images/I%05i.jpg',f)));
    lexF=fullfile(tstDir,sprintf('lex/I%05i.jpg.txt',f));
    if(exist(lexF,'file'))
      fid=fopen(lexF,'r');
      lexS=textscan(fid,'%s'); lexS=lexS{1}';
      fclose(fid);
    else
      lexS=allwords;
    end    
    t3S=tic; [words,t1,t2]=wordSpot(I,lexS,fModel); t3=toc(t3S);    
    saveRes(sF,words,t1,t2,t3);
    tocStatus(ticId,f/nImg);
  end
end

end
