function genPrCurvesEZ
% Generate precision/recall curves. Given properly formatted output, this
% function will do the evaluation with nonmax suppression
%
% This code is to be run separately for ICDAR and SVT, and separately for
% various lexicon sizes (for ICDAR). The 'paramSets' variable controls what
% gets run and plotted.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dPath=globals;
RandStream.getDefaultStream.reset();

% testing conditions (ICDAR)
% -- paramSet={result directory, with/without spell check (for OCR)}

% tstD='icdar'; tstSpl='test'; lexD='lex20'; % lex5, lex50
% swtPlexD=fullfile(dPath,tstD,tstSpl,'EZ','swt+plex+r','images');
% plexD=fullfile(dPath,tstD,tstSpl,'EZ','plex+r','images');
% paramSets={{swtPlexD,0},...
%            {plexD,0}};

% -- other settings (SVT)
tstD='svt'; tstSpl='test'; lexD='lex';         
plexD=fullfile(dPath,tstD,tstSpl,'EZ','plex+r','images');
paramSets={{plexD,0}};

% lexicon folder
lexDir=fullfile(dPath,tstD,tstSpl,lexD);
         
pNms=struct('thr',-inf,'ovrDnm','min','overlap',.5); pNms.type='max';

% figure setup
hs=[]; lgs=cell(0); figure(1); clf; axis normal;
if(~strcmp(tstD,'svt')), axis([0 .8 .5 1]); else axis([0 .5 0 1]); end
set(gcf,'Position',[50 50 600 300]); hold on;

xlabel('Recall','FontSize',16); ylabel('Precision','FontSize',16);

% eval params
iDir=fullfile(dPath,tstD,tstSpl,'images');
evalPrm={'thr',.5,'imDir',iDir,'f0',1,'f1',inf,'lexDir',lexDir,...
  'pNms',pNms};

gtDir=fullfile(dPath,tstD,tstSpl,'wordAnn');

% loop over each paramset and plot it on the same figure
for p=1:length(paramSets)
  paramSet=paramSets{p};
  dtDir=paramSet{1}; isOcr=paramSet{2};
  
  [gt,dt] = evalReading(gtDir,dtDir,'ocr',isOcr,evalPrm{:});
  [xs,ys,sc]=bbGt('compRoc', gt, dt, 0);
  [fs,~,~,idx]=Fscore(xs,ys);
  hs(end+1)=plot(xs,ys,'Color',rand(3,1),'LineWidth',3);
  lgs{end+1}=sprintf('%i [%1.3f] thr=%1.3f',p,fs,sc(idx));
  legend(hs,lgs,'Location','SouthWest','FontSize',14);    
end

