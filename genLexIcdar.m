function genLexIcdar
% Create a synthetic lexicon for the icdar images
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dPath=globals;
RandStream.setDefaultStream(RandStream('mrg32k3a', 'Seed', sum('iccv11')));

% paramSet={dataset, test split, k distractors}
paramSets={{'icdar','test',5},...
           {'icdar','test',20},...
           {'icdar','test',50},...
           {'icdar','train',5},...
           {'icdar','train',20},...
           {'icdar','train',50}};
         
for p=1:length(paramSets)
  paramSet=paramSets{p};
  tstD=paramSet{1}; tstSpl=paramSet{2}; kVal=paramSet{3};
         
  gtDir=fullfile(dPath,tstD,tstSpl,'wordAnn');
  lexDir=fullfile(dPath,tstD,tstSpl,sprintf('lex%i',kVal));
  if(~exist(lexDir,'dir')), mkdir(lexDir); end
  
  files=dir([gtDir '/*.txt']); files={files.name};
  allGtS=[];
  % collect all ground truth words
  for i=1:length(files)
    % load ground truth and prepare for evaluation
    gtNm=[gtDir '/' files{i}];
    gt1=bbGt('bbLoad',gtNm);
    gt1=filterValidGt(gt1);
    for j=1:length(gt1), allGtS{end+1}=gt1(j).lbl; end
  end
  allGtS=unique(upper(allGtS)); numAll=length(allGtS);

  % create lexicons for each file
  for i=1:length(files)
    gtNm=[gtDir '/' files{i}];
    gt1=bbGt('bbLoad',gtNm);
    gt1=filterValidGt(gt1);
    lexS=unique({gt1.lbl});
    numGt=length(lexS);
    while(length(lexS)<(kVal+numGt))
      lexS=[lexS,allGtS(randSample(numAll,kVal+numGt-length(lexS)))];
      lexS=unique(lexS);
    end
    lexP=fullfile(lexDir,files{i});
    fid=fopen(lexP,'w');
    for j=1:length(lexS); fprintf(fid,'%s\n',lexS{j}); end
    fclose(fid);
  end
end