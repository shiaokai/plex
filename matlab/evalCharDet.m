function [gt,dt] = evalCharDet(gtDir,dtDir,varargin)
dfs={'thr',.5,'mul',0,'resize',{},'f0',1,'f1',inf};
[thr,mul,rs,f0,f1]=getPrmDflt(varargin,dfs,1);
cfg=globals;
RandStream.setGlobalStream(RandStream('mrg32k3a', 'Seed', sum('iccv11')));

% collect all strings in ground truth
%gtCharDir=[gtDir '/../charAnn'];
files=dir([gtDir '/*.txt']); files={files.name};
files=files(f0:min(f1,end)); n=length(files); assert(n>0);
chFiles=dir([gtDir '/*.txt']); chFiles={chFiles.name};
chFiles=chFiles(f0:min(f1,end));

% get files in ground truth directory
ticId=ticStatus('evaluating char det');
dt=cell(n,length(unique(cfg.ch1))-1); gt=dt;
for i=1:n
  % detected characters in image
  dtNm=[dtDir '/' files{i}(1:end-8) '.mat'];
  res=load(dtNm); bbs=res.bbs; 

  % ground truth characters in image
  gtChNm=[gtDir '/' chFiles{i}]; gtCh=bbGt('bbLoad',gtChNm);
  gtBbs=zeros(length(gtCh),5); k0=1;
  % load up all ground truth characters into matrix
  for j=1:length(cfg.ch1)
    [~,gtBbs1]=bbGt('bbLoad',gtChNm,'lbls',cfg.ch1(j)); 
    n1=size(gtBbs1,1); 
    gtBbs(k0:n1+k0-1,:)=[gtBbs1(:,1:4) ch2id(cfg.ch1(j),cfg)*ones(n1,1)];
    k0=k0+size(gtBbs1,1);
    % upper
    if (upper(cfg.ch1(j))~=lower(cfg.ch1(j)))
      [~,gtBbs1]=bbGt('bbLoad',gtChNm,'lbls',lower(cfg.ch1(j)));
      n1=size(gtBbs1,1);
      gtBbs(k0:n1+k0-1,:)=[gtBbs1(:,1:4) ch2id(cfg.ch1(j),cfg)*ones(n1,1)];
      k0=k0+size(gtBbs1,1);
    end
  end
  
  for j=1:max([gtBbs(:,5);bbs(:,6)])
    gt0=gtBbs(gtBbs(:,5)==j,:); gt0(:,5)=0;
    dt0=bbs(bbs(:,6)==j,1:5);
    if(~isempty(rs)), dt0=bbApply('resize',dt0,rs{:}); end
    [gt{i,j} dt{i,j}]=bbGt('evalRes',gt0,dt0,thr,mul);
  end
  tocStatus(ticId,i/n);
end

end

function id=ch2id(c,cfg)
id=find(cfg.ch1==c);
end