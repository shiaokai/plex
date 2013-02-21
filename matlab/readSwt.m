function bbsByFile=readSwt(fpath)
% Process output files created by Stroke Width Transform
%
% USAGE
%  bbsByFile = readSwt( fpath )
%
% INPUTS
%  fpath     - path of SWT output file
%
% OUTPUTS
%  bbsByFile - bounding boxes grouped by image file
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

bbsByFile=[]; state=0; fname=''; bbs=[];
fid = fopen(fpath);
tline = fgets(fid);
while ischar(tline)
  tline1=strtrim(tline); tline=fgets(fid);
  if(isempty(tline1) && ~isempty(fname)), bbsByFile{end+1,1}=fname; 
    bbsByFile{end,2}=bbs; state=0; fname=''; bbs=[]; continue; end
  if(isempty(tline1)), state=0; continue; end
  if(state==0), fname=tline1; state=1; continue; end
  if(state==1), 
    spinds=find(tline1==' ');
    xval=str2double(tline1(spinds(1)+1:spinds(2)-1));
    yval=str2double(tline1(spinds(2)+1:spinds(3)-1));
    wval=str2double(tline1(spinds(3)+1:spinds(4)-1));
    hval=str2double(tline1(spinds(4)+1:end));
    bbs(end+1,:)=[xval, yval, wval, hval];
  end
end
bbsByFile{end+1,1}=fname; bbsByFile{end,2}=bbs;
fclose(fid);


end