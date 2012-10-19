function y=computeWordFeatures(word)
% Compute word-level features for SVM
%
% USAGE
%  y = computeWordFeatures( word )
%
% INPUTS
%  word     - word object
%
% OUTPUTS
%  y        - feature vector
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

y=[];
%%% unary features %%%
% - mean detection score
% - stdv detection score
y=[y,word.bb(5)]; % 1
y=[y,median(word.bbs(:,5))]; % 2
y=[y,mean(word.bbs(:,5))]; % 3
y=[y,min(word.bbs(:,5))]; % 4
y=[y,std(word.bbs(:,5))]; % 5

%%% pairwise features %%%
y=[y,pairwise(word.bbs)]; % 6,7,8

%%% global features %%%

% - horizontal std
% - min(hspace)/max(hspace)
% - vertical std
% - min(vspace)/max(vspace)
% - scale std
% - min(height)/max(height)
hgaps=abs((word.bbs(2:end,1)+word.bbs(2:end,3)/2)...
         - (word.bbs(1:end-1,1)+word.bbs(1:end-1,3)/2))...
         ./min([word.bbs(2:end,3),word.bbs(1:end-1,3)],[],2);
vgaps=abs((word.bbs(2:end,2)+word.bbs(2:end,4)/2)...
         - (word.bbs(1:end-1,2)+word.bbs(1:end-1,4)/2))...
         ./min([word.bbs(2:end,4),word.bbs(1:end-1,4)],[],2);       
y=[y,std(hgaps)]; % 9 10
if(min(hgaps)==0), y=[y,max(hgaps)/.01]; else y=[y,max(hgaps)/min(hgaps)]; end
y=[y,std(vgaps)]; % 11 12
if(min(vgaps)==0), y=[y,max(vgaps)/.01]; else y=[y,max(vgaps)/min(vgaps)]; end
y=[y,std(word.bbs(:,3))]; % 13
y=[y,max(word.bbs(:,3))/min(word.bbs(:,3))]; % 14

y=[y,size(word.bbs,1)]; % 15
y=[y,sum(word.bbs(:,5))]; % 16
y=[y,min(word.bbs(:,3))/word.bb(:,3)]; % 17

end

function y=pairwise(bbs)

y=[];
xdistSum=0; ydistSum=0; sdistSum=0;
for i=1:size(bbs,1)-1;  
  parbb=bbs(i,:);
  chibb=bbs(i+1,:);
  xcent=parbb(1,1)+parbb(1,3);
  ycent=parbb(1,2)+.5*parbb(1,4);
  xdist=abs(chibb(1,1)-xcent)/min(chibb(1,3),parbb(1,3));
  if(chibb(1,1)<xcent),xdist=2*xdist; end
  ydist=abs((chibb(1,2)+.5*chibb(1,4))-ycent)/min(chibb(1,4),parbb(1,4));
  sdist=1-min(chibb(1,1),parbb(1,1))/max(chibb(1,1),parbb(1,1));
  xdistSum=xdistSum+xdist; ydistSum=ydistSum+ydist; sdistSum=sdistSum+sdist;
end
y=[xdistSum,ydistSum,sdistSum]./(size(bbs,1)-1);
end


