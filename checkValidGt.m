function valid=checkValidGt(str)
% Check if ground truth string is greater than three characters and doesn't
% contain non alphanumeric symbols.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

[dPath,ch,ch1]=globals;
valid=1;
if(length(str)<3), valid=0; return; end
for j=1:length(str), 
  if(size(find(ch1==upper(str(j))),2)==0), valid=0; return; end; 
end
end