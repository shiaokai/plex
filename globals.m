function cfg=globals(params)
% Global variables
% 
% USAGE
%  cfg=globals
%
% OUTPUTS
%  cfg
%   .dPath      - base directory for data (modify this before running!)
%   .ch         - list of character classes
%   .ch1        - list of collapsed character classes (upper == lower)
%   .chC        - character classes in a column cell
%   .chClfNm    - function handle to return a crazy classifier name from vars
%   .dfNP       - default character NMS params
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

%persistent v;
persistent has_par;
persistent train;
persistent train_bg;
persistent train_type;

persistent test;
persistent lex;
persistent lex0;
persistent test_type

cfg = struct();

% only set these values if param file is provided
if nargin==1
  P=params();
  has_par=P.has_par;
  train=P.train;
  train_bg=P.train_bg;
  train_type=P.train_type;  
  test=P.test;
  lex=P.lex;
  lex0=P.lex0;
  test_type=P.test_type;
end

cfg.train=train;
cfg.train_bg=train_bg;
cfg.train_type=train_type;

cfg.test=test;
cfg.lex=lex;
cfg.lex0=lex0;
cfg.test_type=test_type;

cfg.has_par=has_par;

cfg.bootstrap=1;
cfg.n_train=Inf;
cfg.n_bg=5000;
cfg.max_bs=Inf;
cfg.max_tune_img=Inf;

[~,hostname] = system('hostname');
hostname = strtrim(hostname);

switch hostname
  case 'ballotscan'
    [~,uname]=system('whoami'); uname=strtrim(uname);
    switch uname
      case 'shiaokai'
        cfg.dPath='/data/text/plex/';
        cfg.dBox='/home/shiaokai/Dropbox/res';
      case 'kai'
        cfg.dPath='/users/u1/kai/sharedata/plex/';
    end
  case 'symmetry'
    cfg.dPath='/home/shiaokai/data/';
    cfg.dBox='/home/shiaokai/Dropbox/res';
  otherwise
    error('Need to fill this in on new machines!');
end

cfg.ch='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_';
cfg.chC=mat2cell(cfg.ch',ones(length(cfg.ch),1));
[~,ch1]=equivClass(1:length(cfg.ch),cfg.ch);
cfg.ch1=ch1;
cfg.chClfNm=@(varargin)chClfNm1(varargin{:});

% default character nms params
cfg.dfNP={'thr',-75,'separate',1,'type','maxg','resize',{3/4,1/2},...
    'ovrDnm','union','overlap',.2,'maxn',inf};

% fern parameters
cfg.S=6; cfg.M=256;

cfg.sBin=8;
cfg.oBin=8;
cfg.chH=48;
  
cfg.cFtr=@(I)reshape((5*hogOld(imResample(single(I),[cfg.chH,cfg.chH]),...
  cfg.sBin,cfg.oBin)),[],1);

cfg.progress_prefix=@create_progress_name;
cfg.getName=@()getName(cfg);
cfg.getClfPath=@()getClfPath(cfg);
cfg.getWdClfPath=@()getWdClfPath(cfg);
cfg.resCharClf=@()resCharClf(cfg);
cfg.resCharDet=@()resCharDet(cfg);
cfg.resWordspot=@()resWordspot(cfg);
end

function t=chClfNm1(varargin)
dfs={'S',6,'M',256,'trnSet','train','trnT','charHard','bgDir','none',...
  'nBg',5000,'nTrn',Inf};
[S,M,trnSet,trnT,bgDir,nBg,nTrn]=getPrmDflt(varargin,dfs,1);

% a naming convention for the fern based on its parameters
t=sprintf('fern_S%01iM%03itrnSet%strnT%sbgDir%snBg%inTrn%i',S,M,trnSet,...
  trnT,bgDir,nBg,nTrn);
end

function fname=create_progress_name
cur_pos=dbstack(1); % unroll once
fname=['progress_',cur_pos.name,'_'];
end

function fname=getName(cfg)
% dfs={'S',6,'M',256,'trnSet','train','trnT','charHard','bgDir','none',...
%   'nBg',5000,'nTrn',Inf};
% [S,M,trnSet,trnT,bgDir,nBg,nTrn]=getPrmDflt(varargin,dfs,1);

S=cfg.S; M=cfg.M; 
trnD=cfg.train; trnT=cfg.train_type;
bgD=cfg.train_bg; nBg=cfg.n_bg; nTrn=cfg.n_train;
% a naming convention for the fern based on its parameters
fname=sprintf('fern_inlineS%01iM%03itrnSet%strnT%sbgDir%snBg%inTrn%i',S,M,trnD,...
  trnT,bgD,nBg,nTrn);
end

function clfPath=getWdClfPath(cfg)
cNm=cfg.getName();
clfPath=fullfile(cfg.dPath,cfg.train,'clfs',[cNm,'_',cfg.test,'_wclf.mat']);
end

function clfPath=getClfPath(cfg)
cNm=cfg.getName();
clfPath=fullfile(cfg.dPath,cfg.train,'clfs',[cNm,'.mat']);
end

function clfPath=resWordspot(cfg)
%cNm=cfg.getName();
clfPath=fullfile(cfg.dBox,[cfg.train,'_',cfg.test,'_wspot']);
end

function clfPath=resCharClf(cfg)
%cNm=cfg.getName();
clfPath=fullfile(cfg.dBox,[cfg.train,'_',cfg.test,'_charclf']);
end

function clfPath=resCharDet(cfg)
%cNm=cfg.getName();
clfPath=fullfile(cfg.dBox,[cfg.train,'_',cfg.test,'_chardet']);
end