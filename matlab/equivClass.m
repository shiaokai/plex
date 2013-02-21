function [y1,ch2]=equivClass(y,ch)
% Helper function to specify equivalence of upper and lower case characters 
%
% USAGE
%  [y1,ch2] = equivClass( y, ch )
%
% INPUTS
%  y       - class IDs
%  ch      - string of all character classes
%  
% OUTPUTS
%  y1      - new class IDs
%  ch2     - new (equivalent) character classes
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

ch1=upper(ch); ch2=unique(ch1); y1=y;
  for k=1:length(ch2)
    y1(ch1(y)==ch2(k))=k;
  end
end