function createTable1
% Generate the results for Table 1, from our paper, for character
% classification.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

[dPath,ch,ch1,chC,chClfNm]=globals;

% paramSet={training dataset,with/without neighboring chars}
paramSets={{'icdar','charEasy'},...
           {'icdar','charHard'},...
           {'synth','charHard'}};
% tstSet={testing dataset, with/without neighboring chars}
tstSets={{'icdar','charEasy'},{'icdar','charHard'},{'synth','charHard'}};

RandStream.getDefaultStream.reset();
sBin=8; oBin=8; chH=48;
S=6; M=256; thrr=[0 1]; nTrn=Inf;
cHogFtr=@(I)reshape((5*hogOld(single(imResample(I,[chH,chH])),sBin,oBin)),[],1);
cFtr=cHogFtr;
dbgFileNm=sprintf('table1_%i_%i_%i_%i_%i_%1.2f.txt',clock);
fid=fopen(dbgFileNm,'w');
for p=1:length(paramSets)
  clear I y; 
  paramSet=paramSets{p};
  trnD=paramSet{1}; trnT=paramSet{2};
  
  cDir=fullfile(dPath,trnD,'clfs');
  clfPrms={'S',S,'M',M,'trnT',trnT,'bgDir','none','nBg',0,'nTrn',nTrn};
  cNm=chClfNm(clfPrms{:});
  clfPath=fullfile(cDir,[cNm,'.mat']);
  
  % train fern if doesn't already exist
  if(~exist(clfPath,'file'))
    [I,y]=readAllImgs(fullfile(dPath,trnD,'train',trnT),chC,nTrn);
    % extract features
    x=fevalArrays(I,cFtr)';
    % train ferns
    [ferns,~]=fernsClfTrain(double(x),y,struct('S',S,'M',M,'thrr',thrr,'bayes',1));
    if(~exist(cDir,'dir')),mkdir(cDir); end
    save(clfPath,'ferns');
  else
    load(clfPath);
  end
  fprintf(fid,'CLF:%s\n',clfPath);

  % run classifier on all the test sets
  for j=1:length(tstSets)
    tstSet=tstSets{j}; tstD=tstSet{1}; tstT=tstSet{2};
    clear I y;
    [I,y]=readAllImgs(fullfile(dPath,tstD,'test',tstT),chC,Inf);
    % extract features
    x=fevalArrays(I,cFtr)';
    % run ferns: yh are the class ids, and ph are the scores
    [yh,ph]=fernsClfApply(double(x),ferns); [~,yha]=sort(ph,2,'descend');
    [y1,~]=equivClass(y,ch); yh1=equivClass(yh,ch); yha1=equivClass(yha,ch);
    m=findRanks(y,yha); m1=findRanks(y1,yha1);
    fprintf(fid,'TRAIN:%s-%s TEST:%s-%s: top1 error = %f, top3 error = %f\n',...
      trnD,trnT,tstD,tstT,mean(y~=yh), mean(m>3));
    fprintf(fid,'EQ:TRAIN:%s-%s TEST:%s-%s: top1 error = %f, top3 error = %f\n',...
      trnD,trnT,tstD,tstT,mean(y1~=yh1), mean(m1>3));
  end
end
fclose(fid);
end