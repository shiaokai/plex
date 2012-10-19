function precompSwtAbbyy
% Run ABBYY on regions returned by Stroke Width Transform on the ICDAR
% dataset
%
% One MAT file is created for each image to record the results. After all
% the precomp*.m files are complete, run genPrCurves.m to display results.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dPath=globals;

tstDir=fullfile(dPath,'icdar','test');
abbyyDir=fullfile(tstDir,'abbyy','wordsSWTpad');
d1=fullfile(tstDir,'res-swtPad','abbyy','images');
if(~exist(d1,'dir')), mkdir(d1); end
% read abby SWT results into common structure

bbList=[];
dir1=dir(fullfile(abbyyDir,'*txt'));
for i=1:length(dir1)
  fname=dir1(i).name;
  [imId,bb]=parse_fname(fname);
  wd=procAbbyy(fullfile(abbyyDir,fname));
  bbList{end+1,1}=imId; bbList{end,2}=bb; bbList{end,3}=wd;
end
[B,I,J]=unique(bbList(:,1));
for i=1:length(B)
  idx=find(J==i);
  imId=B{i};
  words=[];
  for j=1:length(idx)
    bb1=bbList{idx(j),2};
    word1=bbList{idx(j),3};
    words(end+1).word=word1;    
    words(end).bb=[bb1, 0];
  end
  save(fullfile(d1,[imId,'.mat']),'words');
end

end

function [imId,bb]=parse_fname(str)
uscore=find(str=='_'); dotind=find(str=='.');
imId=str(1:uscore(1)-1);
xval=str2double(str(uscore(1)+1:uscore(2)-1));
yval=str2double(str(uscore(2)+1:uscore(3)-1));
wval=str2double(str(uscore(3)+1:uscore(4)-1));
hval=str2double(str(uscore(4)+1:dotind(1)-1));
bb=[xval,yval,wval,hval];
end
