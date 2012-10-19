function wordDetDraw( words, showRank, showBbs, showText, col, ls, lw )
% Draw word bounding boxes
%
% USAGE
%  wordDetDraw( words, showRank, showBbs, showText, col, ls, lw )
%
% INPUTS
%  words      - array of word objects
%  showRank   - [1] display the rank of detection
%  showBbs    - [1] display the character bounding boxes
%  showText   - [1] display the strings with bounding boxes
%  col        - [0 1 0] vector for the color of bounding boxes
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

if(nargin<2), showRank=1; end
if(nargin<3), showBbs=1; end
if(nargin<4), showText=1; end
if(nargin<5), col=[0 1 0]; end
if(nargin<6), ls='-'; end
if(nargin<7), lw=2; end
n=length(words);

for i=1:n
  if(size(words(1).bb,2)==5)
    wbbs=reshape([words.bb],5,[])';
    [~,ord]=sort(wbbs(:,5),'descend');
    words=words(ord);
  end
  
  prop={'LineWidth' lw 'LineStyle' ls 'EdgeColor'};
  tProp={'FontSize',8,'color','k'...
    'VerticalAlignment','bottom','BackgroundColor'};  
  if(showBbs)
    for b=1:length(words), if(~isfield(words(b),'bbs')), continue; end
      bbs=words(b).bbs; alt=ones(1,size(bbs,1)); alt(1,2:2:length(alt))=2;
      bbApply('draw',bbs(:,1:4),[1-col; 1-col],2,'-',[],alt);
    end
  end
  for b=1:length(words), bb=words(b).bb;
    rectangle('Position',bb(1:4)+[-bb(3)*.05 0 bb(3)*.1 0],prop{:},col);
  end
  if(showText)
    for b=1:length(words), bb=words(b).bb;
      if(isfield(words(b),'word')), w=words(b).word; else 
        w=words(b).lbl; end
      if(showRank && (size(words(b).bb,2)==5)) 
        text(bb(1)-bb(3)*.05,bb(2)+3,...
          sprintf('%i:%s (%.2f)',b,w,words(b).bb(5)),tProp{:},col); 
      else 
        text(bb(1)-bb(3)*.05,bb(2)+3,w,tProp{:},col); 
      end
    end
  end
  
end