function prepSynthEasy(type)
% Generate synthetic training data for character classifiers
%
% USAGE
%  prepSynthEasy( type )
%
% INPUTS
%  type     - ['train'] should be either 'test' or 'train'
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

if(nargin==0), 
  type = 'train'; 
else
  if(~(strcmp(type,'train') || strcmp(type,'test'))), 
    error('Must be either train or test');
  end
end

cfg=globals;
RandStream.setGlobalStream(RandStream('mrg32k3a', 'Seed', sum(type)));
% render 500 instances per class at size 100 pixels
n=500; sz=100;
for k=1:length(cfg.ch)
  I=zeros(sz,sz,3,n,'uint8'); k0=1;
  for i=1:n % n synthetic examples per character
    if strcmp(type,'train')    
      I(:,:,:,k0)=genChar(cfg.ch(k),sz, ' ', [1 1]); k0=k0+1;
    else
      I(:,:,:,k0)=genChar(cfg.ch(k),sz, ' ', [500 1]); k0=k0+1;
    end
  end
  y=k*ones(1,n); y=y(:);
  writeAllImgs(I,y,cfg.chC,fullfile(cfg.dPath,'synth_easy',type,'char'));
  clear I;
end
end