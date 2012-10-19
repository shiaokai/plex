function varargout = wordDet( action, varargin )
% Word detection function using trie
%
% %% (1) Construct trie structure (lexicon)
% USAGE
%  lex = wordDet( 'build', wordList)
%
% INPUTS
%   wordList  - cell array of strings specifying a lexicon
%
% OUTPUTS
%   lex       - trie structure
%
% %% (2) Run word detection
% USAGE
%  lex = wordDet( 'plexApply', bbs, ch, lex, varargin)
%
% INPUTS
%  bbs        - bounding boxes from character detector
%  ch         - character identities 
%  lex        - trie structure
%  varargin   - additional params
%   .alpha    - [.8] unitary to pairwise cost parameter
%   .radx     - [5] range of x-values to consider
%   .rady     - [2] range of y-values to consider
%   .rads     - [2.5] range of scale to consider
%   .cap      - [500] normalization for fern scores
%   .mpw      - [1] max detections per word
%   .wthr     - [Inf] word score threshold
%   .wolap    - [.5] overlap threshold when returning >1 detections/word
%
% OUTPUTS
%   lex       - trie structure
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

varargout = cell(1,max(1,nargout));
[varargout{:}] = feval(action,varargin{:});
end

function lex=build(wordList)
char='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
lex=struct('kids',-1*ones(1,length(char)),'parent',-1,'isWord',0);
lex=repmat(lex,1,1e5); n=1;
for k=1:length(wordList), w=fliplr(wordList{k}); i=1;
  for j=1:length(w)
    wi=find(char==upper(w(j)));
    if(lex(i).kids(wi)>0), i=lex(i).kids(wi); else
      n=n+1; lex(i).kids(wi)=n; lex(n).parent=i; i=n;
    end
    if(j==length(w)), lex(i).isWord=1; end
  end
end
lex=lex(1:n);
end

function [words, pictres] = plexApply(bbs, ch, lex, varargin)
dfs={'alpha',.8,'radx',3,'rady',1,'rads',2,'cap',500,'mpw',5,...
  'wthr',Inf,'wolap',.5};
[prm.alpha,prm.radx,prm.rady,prm.rads,prm.cap,prm.mpw,prm.wthr,prm.wolap] = getPrmDflt(varargin,dfs,1);

% ASSUMPTIONS
% - words are at least 2 letters
% Configure each word onto the image and return the best match
pictres=[];
words=[];
if(size(bbs,1)==0), return; end

% consider each child of root node
top=lex(1);
for leafi=1:length(top.kids), leaf_t=top.kids(leafi);
  chibbs=bbs(bbs(:,6)==leafi,:);
  if(leaf_t==-1), continue; end
  if(isempty(chibbs)), continue; end  
  leaf=lex(leaf_t);
  
  % 1. PROCESS LEAVES
  % B_j(l_i)=min_{l_j}(m_j(l_j)+d_{ij}(l_i,l_j))
  for pari=1:length(leaf.kids), par_t=leaf.kids(pari);
    parbbs=bbs(bbs(:,6)==pari,:);
    if(par_t==-1), continue; end
    if(isempty(parbbs)), continue; end
        
    Bval1=Inf*ones(size(parbbs,1),1);
    Barg1=-1*ones(size(parbbs,1),1);                
    for ip=1:size(parbbs,1)
      parbb=parbbs(ip,:);
      xRight=parbb(1,1)+prm.radx*parbb(1,3);
      xLeft=parbb(1,1)+parbb(1,3)/prm.radx;
      yCent=parbb(1,2)+.5*parbb(1,4);
      yRange=prm.rady*parbb(1,4);
      sRangeMax=prm.rads*parbb(1,4);
      sRangeMin=parbb(1,4)/prm.rads;
      isectIdx=find((chibbs(:,1)<=xRight) & ...
        (chibbs(:,1)>xLeft) & ...
        (abs((chibbs(:,2)+.5*chibbs(:,4))-yCent)<=yRange) & ...
        (chibbs(:,4)<=sRangeMax) & ...
        (chibbs(:,4)>=sRangeMin));
      minb=Inf; mini=1;
      for j=1:length(isectIdx), ic=isectIdx(j);
        cost1=cost(chibbs(ic,:),parbb,prm.alpha,prm.cap);
        if(cost1<minb), minb=cost1; mini=ic; end
      end
      Bval1(ip,1)=minb; Barg1(ip,1)=mini;
    end    
    words=findW(ch(leafi),ch,ch(pari),par_t,lex,bbs,words,...
      {Bval1},{Barg1},prm);            
  end    
end

% sort results
if(isempty(words)), return; end

wbbs=reshape([words.bb],5,[])';
[~,idx]=sort(wbbs(:,5),'ascend');
words=words(idx);

end

function words=findW(endLetter,ch,pathCh,chi_t,lex,bbs,words,Bval,Barg,prm)

if(lex(chi_t).isWord),
  lword=[fliplr(pathCh),endLetter];  
  % 3. PROCESS ROOT NODES
  % l_r^*=argmin_{l_r}(m_r(l_r)+\sum_{v_c\inC_r}B_c(l_j)
  rootbbs=bbs(bbs(:,6)==find(ch==pathCh(end)),:);
  BvalC=Bval{end};
  
  % find best root location
  rootlocs=zeros(size(rootbbs,1),1);
  for i=1:size(rootbbs,1)
    rootbb=rootbbs(i,:); [~,u]=cost(rootbb,rootbb,0,prm.cap);
    rootlocs(i)=prm.alpha*u+BvalC(i);
  end
  
  % 4. TRAVERSE BACKWARDS FOR SOLUTION
  [vals,inds]=sort(rootlocs,'ascend');
  BargLR=fliplr(Barg);
  
  outwbbs=[]; nwords=0; i=1;
  while (nwords<prm.mpw && i<=size(rootbbs,1))
    optpath=[]; cbbs=[];
    optind=inds(i); optval=vals(i); optpath(1)=optind; i=i+1;
    for j=1:length(lword)-1
      Barg1=BargLR{j}; optind=Barg1(optind); optpath(j+1)=optind;
    end
    
    % 5. GATHER BBs AND SCORE
    for j=1:length(lword)
      curbbs=bbs(bbs(:,6)==find(ch==lword(j)),:);
      cbbs(j,:)=curbbs(optpath(j),:);
    end
    wbb=unionAll(cbbs);
    
    if(size(outwbbs,1)>0)
      oa=bbGt('compOas',wbb,outwbbs);
      if(sum(oa>prm.wolap)>0), continue; end
    end
    
    pscore=optval/length(lword);
    if(pscore>prm.wthr), break; end    
    if(isinf(pscore)), continue; end

    words(end+1).word=lword; 
    words(end).bb=wbb;
    words(end).bb(5)=pscore;
    words(end).bbs=cbbs;

    outwbbs=[outwbbs;wbb]; nwords=nwords+1;
  end    
end

for pari=1:length(lex(chi_t).kids), par_t=lex(chi_t).kids(pari);
  if(par_t==-1), continue; end
  parbbs=bbs(bbs(:,6)==pari,:);
  if(isempty(parbbs)), continue; end
  chibbs=bbs(bbs(:,6)==find(ch==pathCh(end)),:);
  
  % 2. PROCESS INTERIOR NODES
  % B_j(l_i)=min_{l_j}(m_j(l_j)+d_{ij}(l_i,l_j)+\Sum_{v_c\inC_j}B_c(l_j))
  BvalC=Bval{end};
  Bval1=Inf*ones(size(parbbs,1),1);
  Barg1=-1*ones(size(parbbs,1),1);
    
  % fix parent, find best child location
  for ip=1:size(parbbs,1)
    parbb=parbbs(ip,:);
    xRight=parbb(1,1)+prm.radx*parbb(1,3);
    xLeft=parbb(1,1)+parbb(1,3)/prm.radx;
    yCent=parbb(1,2)+.5*parbb(1,4);
    yRange=prm.rady*parbb(1,4);
    sRangeMax=prm.rads*parbb(1,4);
    sRangeMin=parbb(1,4)/prm.rads;
    isectIdx=find((chibbs(:,1)<=xRight) & ...
      (chibbs(:,1)>xLeft) & ...
      (abs((chibbs(:,2)+.5*chibbs(:,4))-yCent)<=yRange) & ...
      (chibbs(:,4)<=sRangeMax) & ...
      (chibbs(:,4)>=sRangeMin));
    minb=Inf; mini=1;
    for j=1:length(isectIdx), ic=isectIdx(j);
      if(BvalC(ic)>minb), continue; end
      cost1=cost(chibbs(ic,:),parbb,prm.alpha,prm.cap)+BvalC(ic);
      if(cost1<minb), minb=cost1; mini=ic; end        
    end
    Bval1(ip,1)=minb; Barg1(ip,1)=mini;
  end
  words=findW(endLetter,ch,[pathCh ch(pari)],par_t,lex,bbs,words,...
    [Bval,Bval1],[Barg,Barg1],prm);  
end

end

function bb=unionAll(bbs)
if(isempty(bbs)),bb=[]; return; end
bb=[min(bbs(:,1)) min(bbs(:,2))...
  max(bbs(:,1)+bbs(:,3)) max(bbs(:,2)+bbs(:,4)) ];
bb=[bb(1:2) bb(3:4)-bb(1:2)];
end

function [c,u,p]=cost(chibb,parbb,alpha,cap)

u = 1-chibb(5)/cap;

% Costs
% - horizontal distance: based on parent's BB
% - vertical distance: same y-center is good
% - size difference: similar size is good
% - overlap between parent and child: overlap is bad

xcent=parbb(1,1)+parbb(1,3);
ycent=parbb(1,2)+.5*parbb(1,4);
xdist=abs(chibb(1,1)-xcent)/min(chibb(1,3),parbb(1,3));
if(chibb(1,1)<xcent),xdist=2*xdist; end
ydist=abs((chibb(1,2)+.5*chibb(1,4))-ycent)/min(chibb(1,4),parbb(1,4));
%sdist=1-min(chibb(1,1),parbb(1,1))/max(chibb(1,1),parbb(1,1));
sdist=1-min(chibb(1,3),parbb(1,3))/max(chibb(1,3),parbb(1,3));
p=xdist+2*ydist+sdist;
c=alpha*u+(1-alpha)*p;
end