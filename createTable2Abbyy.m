function createTable2Abbyy
% Generate the ABBYY results for Table 2 in 'End-to-End Scene Text Recognition.'
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

fprintf('Results for ABBYY on SVT\n');
evalSvt;
fprintf('Results for ABBYY on ICDAR\n');
evalIcdar(-1); % no 'spell check'
evalIcdar(Inf);
evalIcdar(50);

end

function evalIcdar(kVal)
% Evaluate ABBYY output on ICDAR data

dPath=globals;
RandStream.getDefaultStream.reset();
tstDir=fullfile(dPath,'icdar','test');
nTot=0; nCor=0; labNm='wordCharAnnPad';
datadir=fullfile(dPath,'icdar','test','abbyy','wordsPad');
allGtStrs=[];
for j=0:length(dir(fullfile(tstDir,labNm,'*.txt')))-1
  objs=bbGt('bbLoad',fullfile(tstDir,labNm,sprintf('I%05i.jpg.txt',j)));
  gt=upper([objs.lbl]);
  if(~checkValidGt(gt)), continue; end
  allGtStrs{end+1}=gt;
end
allGtStrs=unique(upper(allGtStrs));

ticId=ticStatus('evaluating');
n=length(dir(fullfile(tstDir,labNm,'*.txt')));
for f=0:n-1
  objs=bbGt('bbLoad',fullfile(tstDir,labNm,sprintf('I%05i.jpg.txt',f)));
  gt=upper([objs.lbl]);
  if(~checkValidGt(gt)), continue; end
  abbyyRes=procAbbyy(fullfile(datadir,sprintf('I%05i.txt',f)));
  tmpWord=[]; tmpWord.word=abbyyRes; tmpWord.bb=[];
  
  if(isinf(kVal)), activeWords=allGtStrs;
  else
    % add K random distractors
    activeWords=unique(upper({gt})); numGt=length(activeWords);
    while(length(activeWords)<(kVal+numGt))
      activeWords=[activeWords,...
        allGtStrs(randSample(length(allGtStrs),kVal+numGt-length(activeWords)))]; %#ok<*AGROW>
      activeWords=unique(activeWords);
    end
  end
  
  if(kVal==-1)
    if(~isempty(tmpWord) && strcmpi(tmpWord.word,gt)), nCor=nCor+1; end
  else tmpWord=spellCheck(tmpWord,activeWords);
    if(~isempty(tmpWord) && strcmpi(tmpWord.word,gt)), nCor=nCor+1; end
  end

  nTot=nTot+1; tocStatus(ticId,f/n);
end
fprintf('kVal=%f\n',kVal);
fprintf('%i/%i, %f correct\n',nCor,nTot,nCor/nTot);
end

function evalSvt
% Evaluate ABBYY output on SVT data

dPath=globals;
tstSet=fullfile('svt','test');
nDet=0; nTot=0; nCor=0;
labNm='wordCharAnnPad'; lexNm='wordLexPad';
datadir=fullfile(dPath,'svt','test','abbyy','wordsPad');
n=length(dir(fullfile(dPath,tstSet,labNm,'*.txt')));
for f=0:n-1
  objs=bbGt('bbLoad',fullfile(dPath,tstSet,labNm,sprintf('I%05i.jpg.txt',f)));
  lfile=fullfile(dPath,tstSet,lexNm,sprintf('I%05i.jpg.txt',f));
  
  fid=fopen(lfile,'r');
  lexS=textscan(fid,'%s'); lexS=lexS{1}';
  fclose(fid);
  
  gt=upper([objs.lbl]);
  if(~checkValidGt(gt)), continue; end
  abbyyRes=procAbbyy(fullfile(datadir,sprintf('I%05i.txt',f)));
  tmpWord=[]; tmpWord.word=abbyyRes; tmpWord.bb=[];
  tmpWord1=spellCheck(tmpWord,lexS);
  if(~isempty(tmpWord1)), nDet=nDet+1;
    if(strcmpi(tmpWord1.word,gt)), nCor=nCor+1; end
  end
  nTot=nTot+1;
end
fprintf('%i/%i, %f correct\n',nCor,nTot,nCor/nTot);
end

