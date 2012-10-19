function wd=procAbbyy(fpath)
% Collect cleaned ABBYY output from the filename
%
% USAGE
%  wd = procAbbyy( fpath )
%
% INPUTS
%  fpath    - filename of abbyy output
%
% OUTPUTS
%  wd       - cleaned string
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

wd='';
fid = fopen(fpath);
tline = fgets(fid);
while ischar(tline)
  tline1=strtrim(tline); 
  tline1=tline1(isstrprop(tline1,'alphanum'));
  tline = fgets(fid);
  wd=[wd, tline1];
end
fclose(fid);
end
