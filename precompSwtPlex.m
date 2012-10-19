function precompSwtPlex
% Run PLEX on regions returned by Stroke Width Transform on the ICDAR
% dataset
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

type='swtPad';
% fern parameters
S=6; M=256; nTrn=Inf;
% only consider words that span at least half the image width
widthThr=.5;
frnPrms={'ss',2^(1/5),'minH',.6};
nmsPrms={'thr',-75,'separate',1,'type','maxg','resize',{3/4,1/2},...
  'ovrDnm','union','overlap',.3,'maxn',inf};

% paramSet={train dataset,with/without neighboring chars,
%           bg dataset,# background images,test split}
paramSets={{'synth','charHard','msrcBt',10000,'test'},...
           {'icdar','charHard','icdarBt',10000,'test'},...
           {'synth','charHard','msrcBt',10000,'train'},...
           {'icdar','charHard','icdarBt',10000,'train'}};

for p=1:length(paramSets)
  RandStream.getDefaultStream.reset();
  paramSet=paramSets{p};
  trnD=paramSet{1}; trnT=paramSet{2}; trnBg=paramSet{3}; nBg=paramSet{4};
  tstSpl=paramSet{5}; tstDir=fullfile(dPath,'icdar',tstSpl);

  lexS=loadLex(tstDir);
  % set up classifiers
  cDir=fullfile(dPath,trnD,'clfs');
  clfPrms={'S',S,'M',M,'trnT',trnT,'bgDir',trnBg,'nBg',nBg,'nTrn',nTrn};
  cNm=chClfNm(clfPrms{:});
  clfPath=fullfile(cDir,[cNm,'.mat']);
  
  % set up output locations
  d1=fullfile(tstDir,['res-', type],['res-' trnD],cNm,'images');
  if(~exist(d1,'dir')), mkdir(d1); end
  save(fullfile(d1,'workspace')); % save all variables up to now
  saveRes=@(f,words,t1,t2,t3)save(f,'words','t1','t2','t3');
  
  if(~exist(clfPath,'file')), error('FERN DOES NOT EXIST?!\n'); end
  fModel=load(clfPath);
  
  imDir=fullfile(tstDir,'images');
  filesJpg=dir(fullfile(imDir,'*jpg'));
  tot1=[]; tot2=[]; tot3=[];
  
  for i=1:length(filesJpg)
    subSwtDir=fullfile(imDir,[filesJpg(i).name,'_',type]);
    if(~exist(subSwtDir,'dir')), continue; end
    filesSwtJpg=dir(fullfile(subSwtDir,'*jpg'));
    imId=filesJpg(i).name; didx=find(imId=='.'); imId=imId(1:didx(end)-1);
    sF=fullfile(d1,[imId,'.mat']); words=[];
    for j=1:length(filesSwtJpg)
      I=imread(fullfile(subSwtDir,filesSwtJpg(j).name));
      
      t3S=tic;
      [words1,t1,t2]=wordSpot(I,lexS,fModel,{},nmsPrms,frnPrms);
      t3=toc(t3S);
      
      tot1=[tot1,t1]; tot2=[tot2,t2]; tot3=[tot3,t3];
      
      if(isempty(words1)), continue; end
      % width threshold
      wbb=reshape([words1.bb],5,[])'; inds=wbb(:,3)>(size(I,2)*widthThr);
      words1=words1(inds); if(isempty(words1)), continue; end
      
      % fix bb offset
      swtBb=parse_bb(filesSwtJpg(j).name);
      for k=1:length(words1)
        bbOffset=zeros(1,size(words1(k).bb,2));
        bbOffset(1:2)=swtBb(1:2);
        words1(k).bb=words1(k).bb+bbOffset;
        bbsOffset=zeros(1,size(words1(k).bbs,2));
        bbsOffset(1:2)=swtBb(1:2);
        words1(k).bbs=words1(k).bbs+repmat(bbsOffset,size(words1(k).bbs,1),1);
        words1(k).bid=j;
      end
      words=[words,words1];
    end
    saveRes(sF,words,t1,t2,t3);
  end
end

end

function bb=parse_bb(str)
uscore=find(str=='_'); dotind=find(str=='.');
xval=str2double(str(1:uscore(1)-1));
yval=str2double(str(uscore(1)+1:uscore(2)-1));
wval=str2double(str(uscore(2)+1:uscore(3)-1));
hval=str2double(str(uscore(3)+1:dotind(1)-1));
bb=[xval,yval,wval,hval];
end