function dtOut=spellCheck(dt,gtStrs)
% Correct OCR output to closest ground truth string
%
% USAGE
%  dtOut = spellCheck( dt, gtStrs )
%
% INPUTS
%  dt      - detected word objects
%  gtStrs  - possible ground truth strings
%
% OUTPUTS
%  dtOut   - detected word objects after spell check
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

dtOut=[]; if(isempty(gtStrs)), return; end
for i=1:length(dt)
  dtStr=dt(i).word;
  if(isempty(dtStr)), continue; end
  dvec=[];
  for j=1:length(gtStrs)
    dvec(j)=EditDist(upper(dtStr),upper(gtStrs{j}));
  end
  [val,ind]=min(dvec);
  dtOut(end+1).word=upper(gtStrs{ind});
  dtOut(end).bb=dt(i).bb;
  dtOut(end).bb(5)=val;
end

end