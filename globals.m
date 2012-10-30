function cfg=globals
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

persistent v;
persistent has_par;

cfg = struct();

% parfor check
if isempty(v)
    v=ver;
    has_par=any(strcmp({v.Name},'Parallel Computing Toolbox'));
end

% run parameters
cfg.train='synth';
cfg.train_bg='msrc';
cfg.bootstrap=1;
cfg.test='icdar'; cfg.lex='lex50'; cfg.lex0='lex0';
%cfg.test='svt'; cfg.lex='lex'; cfg.lex0='lex';

if 1
  cfg.train_type='charHard';
  cfg.n_train=Inf;
  cfg.n_bg=5000;
  cfg.max_bs=Inf;
  cfg.max_tune_img=Inf;
else % if debugging
  cfg.train_type='charHard_shrunk';
  cfg.n_train=100;
  cfg.n_bg=100;
  cfg.max_bs=5;
  cfg.max_tune_img=10;
end


[~,hostname] = system('hostname');
hostname = strtrim(hostname);

switch hostname
  case 'ballotscan'
    cfg.dPath='/users/u1/kai/sharedata/plex/';
  case 'symmetry'
    cfg.dPath='/users/u1/kai/sharedata/plex/';
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

cfg.has_par=has_par;
cfg.progress_prefix=@create_progress_name;
cfg.getName=@()getName(cfg);
cfg.getClfPath=@()getClfPath(cfg);
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

function clfPath=getClfPath(cfg)
cNm=cfg.getName();
clfPath=fullfile(cfg.dPath,cfg.train,[cNm,'.mat']);
end