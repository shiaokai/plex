function runPipeline(params)
% Run everything from scratch
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

RandStream.getGlobalStream.reset();

cfg=globals(params);

if(exist(cfg.getClfPath(),'file'))
  res=load(cfg.getClfPath()); fModel=res.fModel;
else
  fModel=trainClassifier(cfg);
end

if 1
  evalCharClassifier(cfg,fModel);
end

if 1
  % tune classifier by discovering max and operating point
  evalCharDetector(cfg,fModel);
end

if 1
  % cross validate on training data for word detection parameters
  alpha=crossValWordDP(cfg);
else
  res=load(cfg.getWdClfPath()); alpha=res.alpha;  
end
  
if 1
  % train word classifier using parameters
  wdClf=trainWordClassifier(cfg,fModel,alpha);
else
  res=load(cfg.getWdClfPath()); wdClf=res.wdClf; alpha=res.alpha;
end

if 1
  % evaluate everything on test
  evalWordSpot(cfg,fModel,wdClf,alpha);
end

if 1
  % produce PR curve
  genFigures(cfg);
end

end

% train character classifier from config
function fModel=trainClassifier(cfg)

% parameters that pretty much won't change
sBin=cfg.sBin; oBin=cfg.oBin; chH=cfg.chH;
S=cfg.S; M=cfg.M; thrr=[0 1]; nTrn=cfg.n_train;
cFtr=cfg.cFtr;

fprintf('Training classifiers.\n');
% Loop over param sets

trnD=cfg.train; trnT=cfg.train_type; trnBg=cfg.train_bg;
nBg=cfg.n_bg; bs=cfg.bootstrap;

clfPath=cfg.getClfPath();
cDir=fileparts(clfPath);
newBg=[cfg.train,cfg.train_type,trnBg,'Bt'];

% load training images
[I,y]=readAllImgs(fullfile(cfg.dPath,trnD,'train',trnT),cfg.chC,nTrn,...
  fullfile(cfg.dPath,trnBg,'train'),nBg);
x=fevalArrays(I,cFtr)';
% train char classifier
[ferns,yh]=fernsClfTrain(double(x),y,struct('S',S,'M',M,'thrr',thrr,'bayes',1));
msg1=sprintf('training error=%f\n',mean(y~=yh));
fModel=[]; fModel.ferns=ferns; fModel.sBin=sBin; fModel.oBin=oBin;
fModel.chH=chH;

% bootstrap classifier if flag is on
if ~bs, return; end
% copy base bg folder to new bootstrap folder
fullBgD=fullfile(cfg.dPath,trnBg,'train','charBg');
fullNewBgD=fullfile(cfg.dPath,newBg,'train','charBg');
if(exist(fullNewBgD,'dir')),
  fprintf('Clearing out old hardnegative folder\n');
  rmdir(fullNewBgD,'s');
end
mkdir(fullNewBgD);
copyfile(fullBgD,fullNewBgD);

maxn=100; n_start=length(dir(fullfile(fullNewBgD,'*png'))); %<- starting index
files=dir(fullfile(cfg.dPath,trnBg,'train','images','*.jpg')); files={files.name};
filesAnn=dir(fullfile(cfg.dPath,trnBg,'train','wordAnn','*.txt')); filesAnn={filesAnn.name};
files=files(1:min(length(files),cfg.max_bs));
if ~isempty(filesAnn), filesAnn=filesAnn(1:min(length(filesAnn),cfg.max_bs)); end

% jump into extended for loop
has_par=cfg.has_par;
progress_file=[cfg.progress_prefix(),cfg.train,'_',cfg.test];
if exist(progress_file,'file'); delete(progress_file); end
system(['touch ', progress_file]);
fprintf('Progress file here: %s\n',progress_file);
  
if has_par,
  if matlabpool('size')>0, matlabpool close; end
  matlabpool open
end

parfor f=1:length(files),
  I=imread(fullfile(cfg.dPath,trnBg,'train','images',files{f}));
  if ~isempty(filesAnn)
    gtBbs=bbGt('bbLoad',fullfile(cfg.dPath,trnBg,'train','wordAnn',filesAnn{f}));
    gtBbs1=reshape([gtBbs.bb],4,[])';
    gtBbs1=[gtBbs1, zeros(size(gtBbs1,1),1)];
  else
    gtBbs1=[];
  end
  
  t1S=tic;
  bbs=charDet(I,fModel,{'thr',0,'minH',.1});
  bbs(:,6)=equivClass(bbs(:,6),cfg.ch);
  bbs=bbNms(bbs,'thr',0,'separate',0,'type','maxg',...
    'resize',{1,1},'ovrDnm','union','overlap',.2,'maxn',inf);
  
  if(isempty(bbs)), continue; end
  P=bbGt('sampleWins',I,{'bbs',bbs,'n',maxn,'dims',[100 100],'ibbs',...
    gtBbs1,'thr',.01});
  if(isempty(P)), continue; end
  P=cell2array(P);
  
  imwrite2(P,size(P,4)>1,n_start+maxn*(f-1),fullNewBgD);
  t1=toc(t1S);
  if has_par,t = getCurrentTask(); tStr=num2str(t.ID); else tStr=''; end
  dt = datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF AM');
  system(['echo ''' tStr ' : ' dt ' : ' num2str(t1) ' : ' ...
    files{f} ''' >> ' progress_file]);
end

if has_par, matlabpool close; end

% squeeze image IDs
files=dir(fullfile(fullNewBgD,'*png')); files = {files.name};
for dest_idx=1:length(files), src_nm=files{dest_idx};
  src_path=fullfile(fullNewBgD,src_nm);
  dst_path=fullfile(fullNewBgD,sprintf('I%05d.png',dest_idx-1));
  if ~strcmp(src_path,dst_path)
    movefile(src_path,dst_path);
  end
end

% re-train again
nBtBg=2*nBg;
RandStream.getGlobalStream.reset();
[I,y]=readAllImgs(fullfile(cfg.dPath,trnD,'train',trnT),cfg.chC,nTrn,...
  fullfile(cfg.dPath,newBg,'train'),nBtBg);
x=fevalArrays(I,cFtr)';
% train char classifier
RandStream.getGlobalStream.reset();
[ferns,yh]=fernsClfTrain(double(x),y,struct('S',S,'M',M,'thrr',thrr,'bayes',1));
msg2=sprintf('after mining negatives training error=%f\n',mean(y~=yh));
if(~exist(cDir,'dir')),mkdir(cDir); end
fModel=[]; fModel.ferns=ferns; fModel.sBin=sBin; fModel.oBin=oBin;
fModel.chH=chH;

% save stuff
if ~exist(fileparts(cfg.getClfPath()),'dir'),
  mkdir(fileparts(cfg.getClfPath()));
end
save(cfg.getClfPath(),'fModel','msg1','msg2');

end


% train character classifier from config
function fModel=evalCharClassifier(cfg,fModel)

% check if this test set has any characters to benchmark
if(isempty(cfg.test_type)),
  y=[]; yh=[]; y1=[]; yh1=[];
  msg3='no char classification to compute';
  msg4='no char classification to compute';
  save(cfg.resCharClf(),'y','yh','y1','yh1','msg3','msg4');
  return;
end

cFtr=cfg.cFtr;

fprintf('Testing classifiers.\n');

% load testing images
[I,y]=readAllImgs(fullfile(cfg.dPath,cfg.test,'test',cfg.test_type),...
  cfg.chC,Inf);
x=fevalArrays(I,cFtr)';

[yh,ph]=fernsClfApply(double(x),fModel.ferns); [~,yha]=sort(ph,2,'descend');
[y1,~]=equivClass(y,cfg.ch); yh1=equivClass(yh,cfg.ch);
yha1=equivClass(yha,cfg.ch);
m=findRanks(y,yha); m1=findRanks(y1,yha1);
msg3=sprintf('TRAIN:%s-%s TEST:%s-%s: top1 error = %f, top3 error = %f\n',...
  cfg.train,cfg.train_type,cfg.test,cfg.test_type,mean(y~=yh), mean(m>3));
msg4=sprintf('EQ:TRAIN:%s-%s TEST:%s-%s: top1 error = %f, top3 error = %f\n',...
  cfg.train,cfg.train_type,cfg.test,cfg.test_type,mean(y1~=yh1), mean(m1>3));

save(cfg.resCharClf(),'y','yh','y1','yh1','msg3','msg4');

end

% evaluate character detector
function fModel=evalCharDetector(cfg,fModel)

fprintf('Eval character detector.\n');
trnD=cfg.train;
cNm=cfg.getName();

% directory for eval images
evalDir=fullfile(cfg.dPath,cfg.test,'train');

% set up output locations
d1=fullfile(evalDir,['res-' trnD],cNm,'images-ch');

nImg=length(dir(fullfile(evalDir,'wordAnn','*.txt')));
nImg=min(nImg,cfg.max_tune_img);

if(exist(d1,'dir')), rmdir(d1,'s'); end
mkdir(d1);
saveRes=@(f,bbs,t1)save(f,'bbs','t1');

% jump into extended for loop
progress_file=[cfg.progress_prefix(),cfg.train,'_',cfg.test];
if exist(progress_file,'file'); delete(progress_file); end
system(['touch ', progress_file]);
fprintf('Progress file here: %s\n',progress_file);
  
has_par=cfg.has_par;
if has_par
  if matlabpool('size')>0, matlabpool close; end
  matlabpool open
end

parfor f=0:nImg-1
  sF=fullfile(d1,sprintf('I%05d.mat',f));
  I=imread(fullfile(evalDir,'images',sprintf('I%05i.jpg',f)));
  
  t1S=tic; bbs=charDet(I,fModel,{}); t1=toc(t1S);  % character detection
  bbs(:,6)=equivClass(bbs(:,6),cfg.ch);  % upper and lower case are equivalent
  bbs=bbNms(bbs,cfg.dfNP);
  if size(bbs,2)==5, bbs=zeros(0,6); end
  saveRes(sF,bbs,t1);

  if has_par,t = getCurrentTask(); tStr=num2str(t.ID); else tStr=''; end
  dt = datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF AM');
  system(['echo '''  tStr ' : ' dt ' : ' num2str(t1) ' : ' sF ''' >> ' progress_file]);
end

if has_par, matlabpool close; end

% PR curves
outDir=fullfile(cfg.dPath,cfg.test,'train',['res-',cfg.train],cNm,'images-ch');
gtDir=fullfile(cfg.dPath,cfg.test,'train','charAnn');
if ~exist(gtDir,'dir'), return; end

[gt,dt]=evalCharDet(gtDir,outDir,'f1',cfg.max_tune_img);

% compute threshold for each class
thrs=zeros(size(gt,2),1);
ranges=zeros(size(gt,2),2);
fsc50=zeros(size(gt,2),1);
fsc75=zeros(size(gt,2),1);
for i=1:size(gt,2)
  [xs,ys,sc]=bbGt('compRoc',gt(:,i),dt(:,i),0);
  [f,x,y,idx]=Fscore(xs,ys,.75);
  fsc75(i)=f;
  [f,x,y,idx]=Fscore(xs,ys,.5);
  fsc50(i)=f;
  dt1=vertcat(dt{:,i});
  tpMean=mean(dt1(dt1(:,6)==1,5));
  fprintf('Char: %s, Fscore: %.03f: P:%.03f R:%.03f tpmean:%1.03f %d\n',...
    cfg.ch(i),f,x,y,tpMean,sc(idx));
  thrs(i)=sc(idx);
  ranges(i,:)=[min(sc),max(sc)];
end

save(cfg.resCharDet(),'fsc50','fsc75');

end

% sweep alpha value for word detection, keep alpha that yields weighted
% fscore
function alpha=crossValWordDP(cfg)

fprintf('Eval word detector\n');

% directory for eval images
evalDir=fullfile(cfg.dPath,cfg.test,'train');

cachedCharDir=fullfile(evalDir,['res-',cfg.train],cfg.getName(),'images-ch');

% set up output locations
d1=fullfile(evalDir,['tune-' cfg.train],cfg.getName(),'images');
if(exist(d1,'dir')), rmdir(d1,'s'); end
mkdir(d1);
saveRes=@(f,words)save(f,'words');

nImg=length(dir(fullfile(evalDir,'wordAnn','*.txt')));
gtDir=fullfile(evalDir,'wordAnn');
lexDir=fullfile(evalDir,cfg.lex0);

has_par=cfg.has_par;

if has_par
  if matlabpool('size')>0, matlabpool close; end
  matlabpool open
end

% sweep over alpha
sweep=linspace(1/500,1/10,25);
scores=zeros(length(sweep),1);
for i=1:length(sweep), cur_alpha=sweep(i);

  progress_file=[cfg.progress_prefix(),cfg.train,'_',cfg.test];
  if exist(progress_file,'file'); delete(progress_file); end
  system(['touch ', progress_file]);
  fprintf('Progress file here: %s\n',progress_file);
    
  parfor f=0:nImg-1
    sF=fullfile(d1,sprintf('I%05d.mat',f));
    
    lexF=fullfile(lexDir,sprintf('I%05i.jpg.txt',f));
    if(exist(lexF,'file'))
      fid=fopen(lexF,'r');
      temp=textscan(fid,'%s'); lex0=temp{1}';
      fclose(fid);
    else
      error('feels bad man.');
    end
    
    % load character detection results
    charCache=load(fullfile(cachedCharDir,sprintf('I%05d.mat',f)));
    bbs=charCache.bbs;
    
    lex=wordDet('build',lex0);
    t1S=tic; words=wordDet('plexApply',bbs,cfg.ch1,lex,{'alpha',cur_alpha}); t1=toc(t1S);
    % store result
    saveRes(sF,words);
    
    if has_par,t=getCurrentTask(); tStr=num2str(t.ID); else tStr=''; end
    dt = datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF AM');
    system(['echo ''' tStr ' : ' dt ' : ' num2str(t1) ' : ' sF ''' >> ' progress_file]);

  end
  
  iDir=fullfile(evalDir,'images');
  pNms=struct('thr',-inf,'ovrDnm','min','overlap',.5); pNms.type='none';
  evalPrm={'thr',.5,'imDir',iDir,'f0',1,'f1',inf,'lexDir',lexDir,...
    'pNms',pNms};
  [gt,dt,gtW,dtW] = evalReading(gtDir,d1,evalPrm{:});
  
  % TP and FP margin
  tpSum=0; fpSum=0;
  for j=1:size(dt,2)
    dt1=dt{j};
    tpSum = tpSum + sum(dt1(dt1(:,6)==1,5));
    fpSum = fpSum + sum(dt1(dt1(:,6)==0,5));
  end
  
  fprintf('Margin diff [higher is better]= %f,\n',tpSum-fpSum);  
  
  % FSCORE
  pNms=struct('thr',-inf,'ovrDnm','min','overlap',.5); pNms.type='max';
  evalPrm={'thr',.5,'imDir',iDir,'f0',1,'f1',inf,'lexDir',lexDir,...
    'pNms',pNms};
  [gt,dt,gtW,dtW] = evalReading(gtDir,d1,evalPrm{:});
  
  [xs,ys,sc]=bbGt('compRoc',gt,dt,0);
  [f,x,y,idx]=Fscore(xs,ys,.75); % .75 for recall
  fprintf('alpha = %f, Fscore= %f,\n',cur_alpha, f);  
  
  scores(i)=f;
end

if has_par, matlabpool close; end

[~,idx]=max(scores); alpha=sweep(idx);
save(cfg.getWdClfPath(),'alpha','sweep','scores');

end

% sweep alpha value for word detection, keep alpha that yields weighted
% fscore
function wdClf=trainWordClassifier(cfg,fModel,alpha)

fprintf('Eval word detector\n');
trnD=cfg.train;
cNm=cfg.getName();

% directory for eval images
evalDir=fullfile(cfg.dPath,cfg.test,'train');
cachedCharDir=fullfile(evalDir,['res-',cfg.train],cfg.getName(),'images-ch');

gtDir=fullfile(evalDir,'wordAnn');
lexDir=fullfile(evalDir,cfg.lex);

% set up output locations
d1=fullfile(evalDir,['res-' trnD],cNm,'images');
  
if(exist(d1,'dir')), rmdir(d1,'s'); end
mkdir(d1);
saveRes=@(f,words,t1)save(f,'words','t1');

nImg=length(dir(fullfile(evalDir,'wordAnn','*.txt')));
nImg=min(nImg,cfg.max_tune_img);

% jump into extended for loop
progress_file=[cfg.progress_prefix(),cfg.train,'_',cfg.test];
if exist(progress_file,'file'); delete(progress_file); end
system(['touch ', progress_file]);
fprintf('Progress file here: %s\n',progress_file);
  
has_par=cfg.has_par;
if has_par
  if matlabpool('size')>0, matlabpool close; end
  matlabpool open
end

parfor f=0:nImg-1
  sF=fullfile(d1,sprintf('I%05d.mat',f));
  lexF=fullfile(lexDir,sprintf('I%05i.jpg.txt',f));
  if(exist(lexF,'file'))
    fid=fopen(lexF,'r');
    lexS=textscan(fid,'%s'); lexS=lexS{1}';
    fclose(fid);
    lex=wordDet('build',lexS);
  else
    error('feels bad man.');
  end
  
  charCache=load(fullfile(cachedCharDir,sprintf('I%05d.mat',f)));
  bbs=charCache.bbs;
    
  t1S=tic; 
  words=wordDet('plexApply',bbs,cfg.ch1,lex,{'alpha',alpha});
  t1=toc(t1S);
  saveRes(sF,words,t1);
  
  if has_par,t=getCurrentTask(); tStr=num2str(t.ID); else tStr=''; end
  dt = datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF AM');
  system(['echo ''' tStr ' : ' dt ' : ' num2str(t1) ' : ' sF ''' >> ' progress_file]);
end

if has_par, matlabpool close; end

% train word svm  
iDir=fullfile(evalDir,'images');
pNms=struct('thr',-inf,'ovrDnm','min','overlap',.5); pNms.type='none';
evalPrm={'thr',.5,'imDir',iDir,'f0',1,'f1',inf,'lexDir',lexDir,...
  'pNms',pNms};

[gtT,dtT,gtwT,dtwT]=evalReading(gtDir,d1,evalPrm);

[xs,ys]=bbGt('compRoc', gtT, dtT, 0);
Fscore(xs,ys)
% train
pNms1=pNms; pNms1.type='max';
[model,xmin,xmax]=trainRescore(dtwT,dtT,gtwT,5,pNms1,.5);
pNms1.clf={xmin,xmax,model}; pNms1.type='none';

wdClf=pNms1;

save(cfg.getWdClfPath(),'wdClf','-append');
  
end

% evaluate pipeline from all configs
function evalWordSpot(cfg,fModel,wdClf,alpha)

% run pipeline on test set

fprintf('Eval word detector\n');
trnD=cfg.train;
cNm=cfg.getName();

% directory for eval images
evalDir=fullfile(cfg.dPath,cfg.test,'test');
lexDir=fullfile(evalDir,cfg.lex);

% set up output locations
d1=fullfile(evalDir,['res-' trnD '-svm'],cNm,'images');

if(exist(d1,'dir')), rmdir(d1,'s'); end
mkdir(d1);
saveRes=@(f,words,t1,t2,t3)save(f,'words','t1','t2','t3');

nImg=length(dir(fullfile(evalDir,'wordAnn','*.txt')));
nImg=min(nImg,cfg.max_tune_img);

% jump into extended for loop
progress_file=[cfg.progress_prefix(),cfg.train,'_',cfg.test];
if exist(progress_file,'file'); delete(progress_file); end
system(['touch ', progress_file]);
fprintf('Progress file here: %s\n',progress_file);
  
has_par=cfg.has_par;
if has_par
  if matlabpool('size')>0, matlabpool close; end
  matlabpool open
end

parfor f=0:nImg-1
  sF=fullfile(d1,sprintf('I%05d.mat',f));
  I=imread(fullfile(evalDir,'images',sprintf('I%05i.jpg',f)));
  lexF=fullfile(lexDir,sprintf('I%05i.jpg.txt',f));
  if(exist(lexF,'file'))
    fid=fopen(lexF,'r');
    res=textscan(fid,'%s');
    lexS=vertcat(res{:})';
    fclose(fid);
  else
    error('feels bad man.');
  end

  t3S=tic; 
  [words,t1,t2]=wordSpot(I,lexS,fModel,wdClf,{},{},{'alpha',alpha}); 
  t3=toc(t3S);
  saveRes(sF,words,t1,t2,t3);  
  
  if has_par,t=getCurrentTask(); tStr=num2str(t.ID); else tStr=''; end
  dt = datestr(now,'mmmm dd, yyyy HH:MM:SS.FFF AM');
  system(['echo ''' tStr ' : ' dt ' : ' num2str(t1) ' : ' sF ''' >> ' progress_file]);
end

if has_par, matlabpool close; end

end

function genFigures(cfg)

evalDir=fullfile(cfg.dPath,cfg.test,'test');
resD=fullfile(['res-' cfg.train '-svm'],cfg.getName());
gtDir=fullfile(evalDir,'wordAnn');
dtDir=fullfile(evalDir,resD,'images');
lexDir=fullfile(evalDir,cfg.lex);
iDir=fullfile(evalDir,'images');
pNms=struct('thr',-inf,'ovrDnm','min','overlap',.5); pNms.type='max';
evalPrm={'thr',.5,'imDir',iDir,'f0',1,'f1',inf,'lexDir',lexDir,...
  'pNms',pNms};
[gt,dt,gtW,dtW] = evalReading(gtDir,dtDir,evalPrm{:});

[xs,ys,sc]=bbGt('compRoc', gt, dt, 0);
[fs,~,~,idx]=Fscore(xs,ys);
figure(1); clf;
plot(xs,ys,'Color',rand(3,1),'LineWidth',3);
lgs={sprintf('[%1.3f] thr=%1.3f',fs,sc(idx))};
legend(lgs,'Location','SouthWest','FontSize',14);
savefig(cfg.resWordspot(),'pdf');
save(cfg.resWordspot(),'xs','ys');
end
