function prepSvt
% Process the raw files downloaded from Street View Text into a common
% format. Download site,
%   http://vision.ucsd.edu/~kai/svt
%
% Move the img folder and xml files here,
%  [dPath]/svt/raw/
% After moving, the folder should look like,
%  [dPath]/svt/raw/img/.
%  [dPath]/svt/raw/img/test.xml
%  [dPath]/svt/raw/img/train.xml
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dPath=globals;
RandStream.getDefaultStream.reset();

datarel=fullfile('svt','raw');
repackage(dPath, datarel, 'train.xml', fullfile('svt','train'));
repackage(dPath, datarel, 'test.xml', fullfile('svt','test'));

cropWords('train',0);
cropWords('train',1);
cropWords('test',0);
cropWords('test',1);

end

function cropWords(d,usePad)
dPath=globals; d1=fullfile(dPath,'svt',d);

if usePad
  wdir='wordsPad'; adir='wordCharAnnPad'; ldir='wordLexPad'; percpad=.2;
else
  wdir='words'; adir='wordCharAnn'; ldir='wordLex'; percpad=0;
end

d2=fullfile(d1,wdir); if(~exist(d2,'dir')), mkdir(d2); end
d2=fullfile(d1,adir); if(~exist(d2,'dir')), mkdir(d2); end
d2=fullfile(d1,ldir); if(~exist(d2,'dir')), mkdir(d2); end

files=dir(fullfile(d1,'images','*.jpg')); n=length(files);
wctr = 0;
for k=1:n
  I=imread(fullfile(d1,'images',files(k).name));
  wobjs=bbGt('bbLoad',fullfile(d1,'wordAnn',[files(k).name,'.txt']));
  fid=fopen(fullfile(d1,'lex',[files(k).name,'.txt']));
  lexS=textscan(fid,'%s\n'); lexS=lexS{1}';
  fclose(fid);
  
  % loop through wobjs
  for j=1:length(wobjs)
    wobj=wobjs(j); wbb=bbGt('get',wobj,'bb');
    
    % crop and save word image
    xpad=wbb(3)*percpad;
    ypad=wbb(4)*percpad;
    pwbb=[wbb(1)-xpad,wbb(2)-ypad,wbb(3)+2*xpad,wbb(4)+2*ypad];
    Iw=bbApply('crop',I,pwbb,'replicate');
    newimgbase=sprintf('I%05d',wctr); wctr=wctr+1;
    imwrite(Iw{1},fullfile(d1,wdir,[newimgbase,'.jpg']),'jpg');
    
    savecobjs=bbGt('create',length(wobj.lbl));
    savecobjs=bbGt('set',savecobjs,'lbl',...
      mat2cell(wobj.lbl',ones(length(wobj.lbl),1))');
    labdest=fullfile(d1,adir,[newimgbase,'.jpg.txt']);
    bbGt('bbSave',savecobjs,labdest);
    lexdest=fullfile(d1,ldir,[newimgbase,'.jpg.txt']);
    %fid=fopen(lexdest,'w'); fprintf(fid,'%s',lex{1}); fclose(fid);

    % save lexicon
    fid=fopen(lexdest,'w');
    for jj=1:length(lexS), fprintf(fid,'%s\n',lexS{jj}); end
    fclose(fid);
  end
end

end


% 1. Move all the images into a single folder
% 2. Create a BB file for word labels output into single folder
% 3. Create a BB file for char labels output into single folder
function repackage(basedir, datarel, labfile, outrel)
d2=fullfile(basedir,outrel,'images'); if(~exist(d2,'dir')), mkdir(d2); end
d2=fullfile(basedir,outrel,'wordAnn'); if(~exist(d2,'dir')), mkdir(d2); end
d2=fullfile(basedir,outrel,'lex'); if(~exist(d2,'dir')), mkdir(d2); end

tree=xmlread(fullfile(basedir, datarel, labfile));
img_elms=tree.getElementsByTagName('image');

for i = 0:img_elms.getLength-1
  img_item=img_elms.item(i);
  img_path=char(img_item.getElementsByTagName('imageName').item(0).getFirstChild.getData);
  img_lex=char(img_item.getElementsByTagName('lex').item(0).getFirstChild.getData);
  wbb_list=img_item.getElementsByTagName('taggedRectangles').item(0).getElementsByTagName('taggedRectangle');
  
  wlbls=[]; wbbs=[];
  
  for j=0:wbb_list.getLength-1
    wbb=wbb_list.item(j);
    tag=char(wbb.getElementsByTagName('tag').item(0).getFirstChild.getData);
    
    wordx=str2double(wbb.getAttribute('x'));
    wordy=str2double(wbb.getAttribute('y'));
    wordw=str2double(wbb.getAttribute('width'));
    wordh=str2double(wbb.getAttribute('height'));
    wlbls{j+1}=tag; wbbs(j+1,:)=[wordx, wordy, wordw, wordh];
  end
  
  I=imread(fullfile(basedir,datarel,img_path));
  newimgbase = sprintf('I%05d',i);
  
  % save image to new location
  imgdest=fullfile(outrel,'images',[newimgbase,'.jpg']);
  imwrite(I,fullfile(basedir,imgdest),'jpg');
  
  % save word bbs
  wobjs=bbGt('create',wbb_list.getLength);
  wobjs=bbGt('set',wobjs,'lbl',wlbls);
  wobjs=bbGt('set',wobjs,'bb',wbbs);
  
  labdest=fullfile(outrel,'wordAnn',[newimgbase,'.jpg.txt']);
  bbGt('bbSave',wobjs,fullfile(basedir,labdest));
  
  % save lexicon
  lexdest=fullfile(basedir,outrel,'lex',[newimgbase,'.jpg.txt']);
  lexS=textscan(img_lex,'%s','Delimiter',','); lexS=lexS{1};
  fid=fopen(lexdest,'w'); 
  for j=1:length(lexS), fprintf(fid,'%s\n',lexS{j}); end
  fclose(fid);
end

end
