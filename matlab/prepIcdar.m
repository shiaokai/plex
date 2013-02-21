function prepIcdar
% Process the raw files downloaded from ICDAR ROBUST READING into a common
% format. Download site,
%   http://algoval.essex.ac.uk/icdar/Datasets.html#RobustReading.html
%
% Move downloaded files here,
%  [dPath]/icdar/raw/
% After moving, the folder should look like,
%  [dPath]/icdar/raw/SceneTrialTest/.
%  [dPath]/icdar/raw/SceneTrialTrain/.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

cfg=globals;
% common character dimensions and # background samples
chSz=100; nBg=5000;
RandStream.getGlobalStream.reset();

trainrel = fullfile('icdar','raw', 'SceneTrialTrain');
testrel = fullfile('icdar','raw', 'SceneTrialTest');

repackage(cfg.dPath, trainrel, fullfile('icdar','train'));
repackage(cfg.dPath, testrel, fullfile('icdar','test'));

% create cropped character data with and without padding
cropChars('train',chSz,nBg,0);
cropChars('train',chSz,nBg,1);
cropChars('test',chSz,nBg,0);
cropChars('test',chSz,nBg,1);

% create cropped words data with and without padding
cropWords('train',0);
cropWords('train',1);
cropWords('test',0);
cropWords('test',1);

end

function cropChars(d,sz,nBg,easy)
% easy=1 means tightly cropped characters.
% easy=0 means cropped with possible neighbors

cfg=globals; d1=fullfile(cfg.dPath,'icdar',d);
files=dir(fullfile(d1,'images','*.jpg')); n=length(files);
I=zeros([sz,sz,3,1e4],'uint8'); y=zeros(1,1e4); k0=1;
B=zeros([sz,sz,3,nBg],'uint8'); b0=1; nBg1=ceil(nBg/n);
for k=1:n
  I1=imread(fullfile(d1,'images',files(k).name));
  fName=fullfile(d1,'charAnn',[files(k).name,'.txt']);
  %[objs,bbs]=bbGt('bbLoad',fullfile(d1,'charAnn',[files(k).name,'.txt']));
  for j=1:length(cfg.chC)
    %bbs=bbGt('toGt',objs,{'lbls',chC(j)});
    [~,bbs]=bbGt('bbLoad',fName,'lbls',cfg.chC(j));
    a=bbApply('area',bbs); bbs=bbs(a>0,:); % remove weird bbs
    if(easy)
      P=bbApply('crop',I1,bbs,'replicate');
      for z=1:length(P)
        bb=bbApply('squarify',[1 1 size(P{z},2) size(P{z},1)],3,1);
        P{z}=bbApply('crop',P{z},bb,'replicate',[sz sz]); P{z}=P{z}{1};
      end
    else
      bbs=bbApply('squarify',bbs,0,1);
      P=bbApply('crop',I1,bbs,'replicate',[sz sz]);
    end
    if(isempty(P)), continue; end
    I(:,:,:,k0:k0+length(P)-1)=cell2array(P);
    y(k0:k0+length(P)-1)=j; k0=k0+length(P);
  end
  %bbs=bbGt('toGt',objs,struct([])); bbs(:,5)=1;
  [~,bbs]=bbGt('bbLoad',fName); bbs(:,5)=1;
  %bbBg=bbApply('random',size(I1,2),size(I1,1),...
  %  [50 size(I1,2)],[50 size(I1,1)],nBg1*5);
  bbBg=bbApply('random','dims',[size(I1,1),size(I1,2)],...
    'wRng',[50 size(I1,2)],'hRng',[50 size(I1,1)],'n',nBg1*5);
  bbBg=bbApply('squarify',bbBg,1);
  B1 = bbGt( 'sampleWins', I1,...
    {'bbs',bbBg,'ibbs',bbs,'dims',[sz sz],'thr',.01} );
  B1=B1(1:min(nBg1,length(B1))); if(isempty(B1)), continue; end
  B(:,:,:,b0:b0+length(B1)-1)=cell2array(B1); b0=b0+length(B1);
end
I=I(:,:,:,1:k0-1); y=y(1:k0-1); B=B(:,:,:,1:b0-1);
bgD=fullfile(cfg.dPath,'icdar',d,'charBg');
if(~exist(bgD,'dir')), mkdir(bgD); end; imwrite2(B,1,0,bgD);
if(easy), writeAllImgs(I,y,cfg.chC,fullfile(cfg.dPath,'icdar',d,'charEasy'));
else writeAllImgs(I,y,cfg.chC,fullfile(cfg.dPath,'icdar',d,'charHard')); end
end

function cropWords(d,usePad)
cfg=globals; d1=fullfile(cfg.dPath,'icdar',d);

if usePad,
  wdir='wordsPad'; adir='wordCharAnnPad'; percpad = .25;
else
  wdir='words'; adir='wordCharAnn'; percpad = 0;
end

mkdir(fullfile(d1,wdir));
mkdir(fullfile(d1,adir));

files=dir(fullfile(d1,'images','*.jpg')); n=length(files);
wctr = 1;
for k=1:n
  I=imread(fullfile(d1,'images',files(k).name));
  cobjs=bbGt('bbLoad',fullfile(d1,'charAnn',[files(k).name,'.txt']));
  wobjs=bbGt('bbLoad',fullfile(d1,'wordAnn',[files(k).name,'.txt']));
  
  % loop through wobjs
  for j=1:length(wobjs)
    wobj=wobjs(j); wbb=bbGt('get', wobj, 'bb');
    [savecobjs,err]=getWordCobjs(wobj,cobjs);
    
    % crop and save word image
    xpad=round(wbb(3)*percpad); ypad=round(wbb(4)*percpad);
    pwbb=[wbb(1)-xpad,wbb(2)-ypad,wbb(3)+2*xpad,wbb(4)+2*ypad];
    Iw=bbApply('crop',I,pwbb,'circular');
    newimgbase=sprintf('I%05d',wctr); wctr=wctr+1;
    %newimgbase=sprintf('%04d',wctr); wctr=wctr+1;
    imwrite(Iw{1},fullfile(d1,wdir,[newimgbase,'.jpg']),'jpg');
    
    % check sizes
    if(err)
      bbstr=cell2mat(bbGt('get', savecobjs, 'lbl')');
      fprintf('ERROR: IMG ID=%s: BBs=%s, WORD=%s\n', newimgbase, bbstr, wobj.lbl);
      % save empty bbGt file
      dummycobjs=bbGt('create', 0);
      labdest=fullfile(d1, adir, [newimgbase, '.jpg.txt']);
      bbGt('bbSave', dummycobjs, labdest);
      continue;
    end
    
    % re-compute offsets
    origbbs=bbGt('get', savecobjs, 'bb');
    adjbbs=[origbbs(:,1)-pwbb(1), origbbs(:,2)-pwbb(2), origbbs(:,3:4)];
    savecobjs=bbGt('set', savecobjs, 'bb', adjbbs);
    
    % save label file
    labdest=fullfile(d1, adir, [newimgbase, '.jpg.txt']);
    bbGt('bbSave', savecobjs, labdest);
  end
end

end

function repackage(basedir, datarel, outrel)
% 1. Move all the images into a single folder
% 2. Create a BB file for word labels output into single folder
% 3. Create a BB file for char labels output into single folder

percpad = .25;

mkdir(fullfile(basedir,outrel,'images'));
mkdir(fullfile(basedir,outrel,'charAnn'));
mkdir(fullfile(basedir,outrel,'wordAnn'));

swtPath=fullfile(basedir,datarel,'swt.txt'); swtBbs=[];
if(exist(swtPath,'file')), swtBbs=readSwt(swtPath); end

tree=xmlread(fullfile(basedir, datarel, 'segmentation.xml'));
img_elms=tree.getElementsByTagName('image');

for i=0:img_elms.getLength-1
  img_item=img_elms.item(i);
  img_path=char(img_item.getElementsByTagName('imageName').item(0).getFirstChild.getData);
  wbb_list=img_item.getElementsByTagName('taggedRectangles').item(0).getElementsByTagName('taggedRectangle');
  
  wlbls=[]; wbbs=[];
  clbls=[]; cbbs=[]; cbb_ctr=1;
  
  for j=0:wbb_list.getLength-1
    wbb=wbb_list.item(j);
    tag=char(wbb.getElementsByTagName('tag').item(0).getFirstChild.getData);
    
    wordx=str2double(wbb.getAttribute('x'));
    wordy=str2double(wbb.getAttribute('y'));
    wordw=str2double(wbb.getAttribute('width'));
    wordh=str2double(wbb.getAttribute('height'));
    wlbls{j+1}=tag; wbbs(j+1,:)=[wordx,wordy,wordw,wordh];
    
    cbb_list=wbb.getElementsByTagName('segmentation').item(0).getElementsByTagName('xOff');
    lbkpt=1;
    clbls=[clbls,num2cell(tag)];
    for k=0:cbb_list.getLength-1
      cbb=cbb_list.item(k);
      rbkpt=str2double(cbb.getFirstChild.getData);
      charx=wordx+lbkpt; chary=wordy;
      charw=rbkpt-lbkpt; charh=wordh;
      lbkpt=rbkpt;
      cbbs(cbb_ctr,:)=[charx,chary,charw,charh]; cbb_ctr=cbb_ctr+1;
    end
    
    charx=wordx+lbkpt;
    chary=wordy;
    charw=wordw-lbkpt;
    charh=wordh;
    cbbs(cbb_ctr,:)=[charx,chary,charw,charh]; cbb_ctr=cbb_ctr+1;
  end
  
  I=imread(fullfile(basedir, datarel, img_path));
  newimgbase=sprintf('I%05d',i);
  
  % save image to new location
  imgdest=fullfile(outrel,'images',[newimgbase,'.jpg']);
  imwrite(I,fullfile(basedir,imgdest),'jpg');
  
  % save word bbs
  wobjs=bbGt('create',wbb_list.getLength);
  wobjs=bbGt('set',wobjs,'lbl',wlbls);
  wobjs=bbGt('set',wobjs,'bb',wbbs);
  
  labdest=fullfile(outrel,'wordAnn',[newimgbase,'.jpg.txt']);
  bbGt('bbSave',wobjs,fullfile(basedir,labdest));
  
  % save char bbs
  cobjs=bbGt('create',size(cbbs,1));
  cobjs=bbGt('set',cobjs,'lbl',clbls);
  cobjs=bbGt('set',cobjs,'bb',cbbs);
  
  labdest=fullfile(outrel,'charAnn',[newimgbase,'.jpg.txt']);
  bbGt('bbSave',cobjs,fullfile(basedir,labdest));
  
  % check if Stroke Width Text output file exists
  if(~isempty(swtBbs))
    cropDir=fullfile(basedir,[imgdest,'_swt']);
    padcropDir=fullfile(basedir,[imgdest,'_swtPad']);
    if(~exist(cropDir,'dir')), mkdir(cropDir); end
    if(~exist(padcropDir,'dir')), mkdir(padcropDir); end
    bbs=swtBbs{find(strcmp(swtBbs(:,1),img_path)),2};
    
    % crop and save word image
    for j=1:size(bbs,1)
      wbb=bbs(j,:);
      xpad=wbb(3)*percpad; ypad=wbb(4)*percpad;
      pwbb=round([wbb(1)-xpad,wbb(2)-ypad,wbb(3)+2*xpad,wbb(4)+2*ypad]);
      Iswt=bbApply('crop',I,wbb,'replicate');
      IswtPad=bbApply('crop',I,pwbb,'replicate');
      p1=fullfile(cropDir,sprintf('%i_%i_%i_%i.jpg',wbb(1),wbb(2),wbb(3),wbb(4)));
      p2=fullfile(padcropDir,sprintf('%i_%i_%i_%i.jpg',pwbb(1),pwbb(2),pwbb(3),pwbb(4)));
      imwrite(Iswt{1},p1,'jpg'); imwrite(IswtPad{1},p2,'jpg');
    end
    
    % alternative
    cropDir=fullfile(basedir,outrel,'wordsSWT');
    padcropDir=fullfile(basedir,outrel,'wordsSWTpad');
    if(~exist(cropDir,'dir')), mkdir(cropDir); end
    if(~exist(padcropDir,'dir')), mkdir(padcropDir); end
    bbs=swtBbs{find(strcmp(swtBbs(:,1),img_path)),2};
    % crop and save word image
    for j=1:size(bbs,1)
      wbb=bbs(j,:);
      xpad=wbb(3)*percpad; ypad=wbb(4)*percpad;
      pwbb=round([wbb(1)-xpad, wbb(2)-ypad, wbb(3)+2*xpad, wbb(4)+2*ypad]);
      Iswt=bbApply('crop',I,wbb,'replicate');
      IswtPad=bbApply('crop',I,pwbb,'replicate');
      p1=fullfile(cropDir,sprintf('%s_%i_%i_%i_%i.jpg',newimgbase,...
        wbb(1),wbb(2),wbb(3),wbb(4)));
      p2=fullfile(padcropDir,sprintf('%s_%i_%i_%i_%i.jpg',newimgbase,...
        pwbb(1),pwbb(2),pwbb(3),pwbb(4)));
      imwrite(Iswt{1},p1,'jpg'); imwrite(IswtPad{1},p2,'jpg');
    end
  end
end

end


function [outObjs,err]=getWordCobjs(wobj,cobjs)

outObjs=[]; err=0;
wbb = bbGt('get', wobj, 'bb');
for m=1:length(cobjs)
  cobj = cobjs(m);
  cbb = bbGt('get', cobj, 'bb');
  carea = bbApply('area', cbb);
  iarea = bbApply('area', bbApply('intersect', wbb, cbb));
  % intersect cobjs w/ wobj, if there is enough overlap, then associate it
  %   with the wobj
  if (iarea == carea) && (wobj.bb(4) == cbb(4))
    outObjs = [outObjs; cobj];
  end
end

if(length(outObjs)~=length(wobj.lbl)), err=1; end

end
