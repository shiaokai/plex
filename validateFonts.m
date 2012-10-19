function validateFonts
% Generate a list for valid fonts on the machine
%
% A figure pops up for each font, to verify if it renders properly on your
% machine.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

allfonts=listfonts;
validFonts=[];
for i=1:length(allfonts)
    fontname = allfonts{i};
    figure(1); clf;
    text(0,.5,fontname,'fontsize', 100,'fontname',fontname);
    in=input('Render properly y/n? ','s');
    if in=='y', validFonts{end+1} = fontname; end
end
save('validFonts', 'validFonts');
end