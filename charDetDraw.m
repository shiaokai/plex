function hs = charDetDraw( bb, ch, lw, ls )
% Draw character bounding boxes
%
% USAGE
%  charDetDraw( bb, ch, lw, ls )
%
% INPUTS
%  bb         - character bounding boxes
%  ch         - mapping classid to character
%  lw         - [2] line width
%  ls         - [-] line style
%
% OUTPUTS
%  hs         - handle
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

[n,m]=size(bb); if(n==0), hs=[]; return; end
if(nargin<3 || isempty(lw)), lw=2; end
if(nargin<4 || isempty(ls)), ls='-'; end
% prepare display properties
prop={'LineWidth' lw 'LineStyle' ls 'EdgeColor'};
tProp={'FontSize',10,'color','k','FontWeight','bold',...
  'VerticalAlignment','bottom','BackgroundColor'};

hs=zeros(1,n); clrs=hsv(length(ch));
for b=1:n, hs(b)=rectangle('Position',bb(b,1:4),prop{:},clrs(bb(b,6),:)); end
hs=[hs zeros(1,n)];
for b=1:n, hs(b+n)=text(bb(b,1),bb(b,2)+3,ch(bb(b,6)),tProp{:},clrs(bb(b,6),:)); end
end