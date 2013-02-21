function m=findRanks(y,yh)
% Return the rank of the correct result in the output
%
% USAGE
%  [m] = findRanks( y, yh )
%
% INPUTS
%  y        - vector of class IDs
%  yh       - matrix of results
%
% OUTPUTS
%  m        - rank of each output
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

m=zeros(length(y),1);
for k=1:length(y)
  t=find(y(k)==yh(k,:),1); if(isempty(t)), t=inf; end
   m(k)=t;
end
