function prepMsrc
% Process the raw files downloaded from MSRC into a common format
% Download site,
%  http://research.microsoft.com/en-us/downloads/b94de342-60dc-45d0-830b-9f6eff91b301/default.aspx
%
% Move the scenes,buildings, and miscellaneous folders here,
%  [dPath]/msrc/raw/
% After moving, the folder should look like,
%  [dPath]/msrc/raw/scenes/.
%  [dPath]/msrc/raw/scenes/countryside/.
%  [dPath]/msrc/raw/scenes/office/.
%  [dPath]/msrc/raw/scenes/urban/.
%  [dPath]/msrc/raw/buildings
%  [dPath]/msrc/raw/miscellaneous
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dPath=globals;
% common character dimensions and # background samples
sz=100; nBg=5000;
RandStream.getDefaultStream.reset();

% --This block needs to be run before the crop functions can be called
subdirs={fullfile('scenes','countryside'),...
         fullfile('scenes','office'),...
         fullfile('scenes','urban'),...
         'buildings','miscellaneous'};
repackage(dPath,fullfile('msrc','raw'),fullfile('msrc','train'),...
  fullfile('msrc','test'),subdirs);

cropChars('train',sz,nBg);
cropChars('test',sz,nBg);

end

% This function needs to be called with easy = {0,1} to produce all the
function cropChars(d,sz,nBg)
[dPath]=globals; d1=fullfile(dPath,'msrc',d);
files=dir(fullfile(d1,'images','*.jpg')); n=length(files);
B=zeros([sz,sz,3,nBg],'uint8'); b0=1; nBg1=ceil(nBg/n);
for k=1:n
  I1=imread(fullfile(d1,'images',files(k).name));
  %bbBg=bbApply('random',size(I1,2),size(I1,1),...
  %  [50 size(I1,2)],[50 size(I1,1)],nBg1*5);
  bbBg=bbApply('random','dims',[size(I1,1),size(I1,2)],...
    'wRng',[50 size(I1,2)],'hRng',[50 size(I1,1)],'n',nBg1*5);
  bbBg=bbApply('squarify',bbBg,1);
  B1 = bbGt( 'sampleWins', I1,...
    {'bbs',bbBg,'dims',[sz sz],'thr',.1} );
  B1=B1(1:min(nBg1,length(B1))); if(isempty(B1)), continue; end
  B(:,:,:,b0:b0+length(B1)-1)=cell2array(B1); b0=b0+length(B1);
end
B=B(:,:,:,1:b0-1);
bgD=fullfile(dPath,'msrc',d,'charBg');
if(~exist(bgD,'dir')), mkdir(bgD); end; imwrite2(B,1,0,bgD);
end

% 1. Move MSRC images into train and test folder
%    Place every other image into train/test folder
function repackage(basedir, datarel, outtrainrel, outtestrel, subdirs)

[dPath]=globals;
dtrain=fullfile(basedir,outtrainrel,'images');
dtest=fullfile(basedir,outtestrel,'images');
if(~exist(dtrain,'dir')), mkdir(dtrain); end
if(~exist(dtest,'dir')), mkdir(dtest); end
ctr=0;
for i=1:length(subdirs)
  sd=subdirs{i};
  d1=fullfile(dPath,datarel,sd)
  files=dir(fullfile(d1,'*.JPG')); n=length(files);
  for k=1:2:n-1
    I1=imread(fullfile(d1,files(k).name));
    I2=imread(fullfile(d1,files(k+1).name));
    
    newimgbase = sprintf('I%05d', ctr);
    
    imgdest = fullfile(outtrainrel,'images',[newimgbase, '.jpg']);
    imwrite(I1, fullfile(basedir, imgdest), 'jpg');

    imgdest = fullfile(outtestrel,'images',[newimgbase, '.jpg']);
    imwrite(I2, fullfile(basedir, imgdest), 'jpg');

    ctr=ctr+1;
  end
end


end