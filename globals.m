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

[~,hostname] = system('hostname');
hostname = strtrim(hostname);

switch hostname
    case 'ballotscan'
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

% parfor check
if isempty(v)
    v=ver;
    has_par=any(strcmp({v.Name},'Parallel Computing Toolbox'));
end
cfg.has_par=has_par;
cfg.progress_prefix=@create_progress_name;

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