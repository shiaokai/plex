function createTable2
% Generate the PLEX results for Table 2 in 'End-to-End Scene Text
% Recognition.'
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

[dPath,ch,ch1,chC,chClfNm]=globals;

S=6; M=256; nTrn=Inf; minH=.6; topK=100;
frnPrms={'ss',2^(1/5),'minH',minH};
nmsPrms={'thr',-75,'separate',1,'type','maxg','resize',{1,1/2},...
  'ovrDnm','union','overlap',.3,'maxn',inf};
% only consider words that span at least half the image width
widthThr=.5;
    
% paramSet={train dataset,with/without neighboring chars, bg dataset,# bg images}
paramSets={{'synth','charHard','msrcBt',10000},...
           {'icdar','charHard','icdarBt',10000}};
% tstSet={test dataset, number of distractors}
tstSets={{'svt',Inf},{'icdar',50},{'icdar',Inf}};

RandStream.getDefaultStream.reset();
dbgFileNm=sprintf('table2_%i_%i_%i_%i_%i_%1.2f.txt',clock);
fid=fopen(dbgFileNm,'w'); fprintf('LOG:%s\n',dbgFileNm);
labNm='wordCharAnnPad'; datNm='wordsPad';
for p=1:length(paramSets)
  clear I y; paramSet=paramSets{p};
  trnD=paramSet{1}; trnT=paramSet{2}; trnBg=paramSet{3}; nBg=paramSet{4};
  
  cDir=fullfile(dPath,trnD,'clfs');
  clfPrms={'S',S,'M',M,'trnT',trnT,'bgDir',trnBg,...
    'nBg',nBg,'nTrn',nTrn};
  cNm=chClfNm(clfPrms{:}); clfPath=fullfile(cDir,[cNm,'.mat']);
  if(~exist(clfPath,'file')), error('FERN DOES NOT EXIST?!\n'); end
  fModel=load(clfPath);
  fprintf(fid,'CLF:%s\n',clfPath);
  
  % loop over test sets
  for i=1:length(tstSets)
    tstSet=tstSets{i}; tstD=tstSet{1}; kVal=tstSet{2};
    tstDir=fullfile(dPath,tstD,'test');
    fprintf(fid,'TEST DIR:%s\n',tstDir); fprintf(fid,'KVAL:%i\n',kVal);
    
    allGtStrs=[];
    % collect all ground truth words
    for j=0:length(dir(fullfile(tstDir,labNm,'*.txt')))-1
      objs=bbGt('bbLoad',fullfile(tstDir,labNm,sprintf('I%05i.jpg.txt',j)));
      gt=upper([objs.lbl]);
      if(~checkValidGt(gt)), continue; end
      allGtStrs{end+1}=gt;
    end
    allGtStrs=unique(upper(allGtStrs));    

    strMatchPos=[]; tot1=[]; tot2=[]; tot3=[];
    fclose(fid);
    
    % loop over images
    for f=0:length(dir(fullfile(tstDir,labNm,'*.txt')))-1
      fid1=fopen(dbgFileNm,'a'); fprintf(fid1,'%i,',f); fclose(fid1);
      objs=bbGt('bbLoad',fullfile(tstDir,sprintf('%s/I%05i.jpg.txt',labNm,f)));
      gt=upper([objs.lbl]); if(~checkValidGt(gt)), continue; end
      I=imread(fullfile(tstDir,sprintf('%s/I%05i.jpg',datNm,f)));
      
      if(~strcmp(tstSet,'svt'))
        if(isinf(kVal)),
          lexS=allGtStrs;
        else
          % add K random distractors
          lexS=unique(upper({gt})); numGt=length(lexS);
          while(length(lexS)<(kVal+numGt))
            lexS=[lexS,allGtStrs(randSample(length(allGtStrs),...
              kVal+numGt-length(lexS)))];
            lexS=unique(lexS);
          end
        end
      else
        lfile=fullfile(tstDir,'wordLexPad',sprintf('I%05i.jpg.txt',f));
        fid1=fopen(lfile);
        lexS=textscan(fid1,'%s'); lexS=lexS{1}';
        fclose(fid1);
      end
      
      t3S=tic;
      [words,t1,t2]=wordSpot(I,lexS,fModel,{},nmsPrms,frnPrms);
      t3=toc(t3S);
      
      tot1=[tot1,t1]; tot2=[tot2,t2]; tot3=[tot3,t3];
      
      words1=words(1:min(length(words),topK));
      % 1=miss, >1=match ind + 1
      [strMatch,words1]=getWordMatchInd(objs,words1,...
        size(I,2)*widthThr);
      strMatchPos=[strMatchPos,strMatch];
    end
    
    fid=fopen(dbgFileNm,'a');
    fprintf(fid,'$%s\n',clfPath);
    fprintf(fid,'(total examples):%i\n',length(strMatchPos));
    
    fprintf(fid,'(string match results):');
    u1=unique(strMatchPos);
    for j=1:length(u1)
      fprintf(fid,'%i:%i,',u1(j)-1,sum(strMatchPos==u1(j)));
    end
    fprintf(fid,'\n');
    fprintf(fid,'top1:%1.4f\n',sum(strMatchPos==2)/length(strMatchPos));
  end
end
fclose(fid);
end

% check if word is recalled in the result list
% ID of 1 is a miss; subtract 1 from all other positions (2=>1,3=>2...)
function [strMatch,words1]=getWordMatchInd(gtObj,words,widthThr)
if(nargin==2), widthThr=0; end
if((widthThr>0) && ~isempty(words))     
  wbb=reshape([words.bb],5,[])';
  inds=wbb(:,3)>widthThr; words1=words(inds); wbb=wbb(inds,:);
else
  words1=words;
end
strMatch=1;
% scan through list and track when we find a match
for i=1:length(words1)
  if((strMatch==1) && strcmpi([gtObj.lbl],words1(i).word))
    strMatch=i+1;
  end
end
end
