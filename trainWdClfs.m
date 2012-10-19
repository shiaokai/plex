function trainWdClfs
% Train word-level classifiers (SVM); re-score words
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

[dPath,ch,ch1,chC,chClfNm]=globals;
RandStream.getDefaultStream.reset();

S=6; M=256; nTrn=Inf; 
trnT='charHard'; trnBg='msrcBt'; nBg=10000;
clfPrms={'S',S,'M',M,'trnT',trnT,'bgDir',trnBg,'nBg',nBg,'nTrn',nTrn};
cNm=chClfNm(clfPrms{:});

% -- paramSet={dataset, test split, lexicon dir, results dir}
paramSets={{'svt','test','lex',fullfile('res-synth')},...
           {'icdar','test','lex50',fullfile('res-swtPad','res-synth')},...
           {'icdar','test','lex50',fullfile('res-synth')}};
         
nFold=5; evalThr=.5;
pNms=struct('thr',-inf,'ovrDnm','min','overlap',.5);
saveRes=@(f,words)save(f,'words');

for p=1:length(paramSets)
  paramSet=paramSets{p};
  tstD=paramSet{1}; tstSpl=paramSet{2}; 
  lexD=paramSet{3}; resDir=paramSet{4};

  iDir=fullfile(dPath,tstD,tstSpl,'images');
  gtDir=fullfile(dPath,tstD,tstSpl,'wordAnn');
  dtDir=fullfile(dPath,tstD,tstSpl,resDir,cNm,'images');

  pNms1=pNms; pNms1.type='none';
  % training directories
  gtDirTr=fullfile(dPath,tstD,'train','wordAnn');
  dtDirTr=fullfile(dPath,tstD,'train',resDir,cNm,'images');
  lexDirTr=fullfile(dPath,tstD,'train',lexD);
  evalPrmTr={'thr',evalThr,'imDir',iDir,'f0',1,'f1',inf,...
    'lexDir',lexDirTr,'pNms',pNms1};
  
  % directory to save to after re-scoring
  outDir=fullfile(dPath,tstD,tstSpl,[resDir,'-svm'],cNm,'images');
  if(~exist(outDir,'dir')), mkdir(outDir); end
  
  % eval on training set
  [gtT,dtT,gtwT,dtwT]=evalReading(gtDirTr,dtDirTr,evalPrmTr);

  [xs,ys]=bbGt('compRoc', gtT, dtT, 0);
  Fscore(xs,ys)
  % train
  pNms1=pNms; pNms1.type='max';
  [model,xmin,xmax]=trainRescore(dtwT,dtT,gtwT,nFold,pNms1,evalThr);
  pNms1.clf={xmin,xmax,model}; pNms1.type='none';

  % apply svm to re-score test set
  files=dir(fullfile(gtDir,'*.txt')); files={files.name};
  for i=1:length(files), fname=[files{i}(1:end-8),'.mat'];
    dtNm=fullfile(dtDir,fname);
    if(~exist(dtNm,'file')), dta=[]; else res=load(dtNm); dta=res.words; end
    % TODO: fix issue with signs of word scores (very confusing)
    for j=1:length(dta), dta(j).bb(:,5)=-dta(j).bb(:,5); end
    dta=wordNms(dta,pNms1);
    for j=1:length(dta), dta(j).bb(:,5)=-dta(j).bb(:,5); end
    saveRes(fullfile(outDir,fname),dta);
  end

  % save SVM
  save(fullfile(outDir,'..','wordSvm'),'pNms1');
end

