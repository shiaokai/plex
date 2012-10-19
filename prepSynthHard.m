function prepSynthHard(type)
% Generate synthetic training data for character classifiers
%
% USAGE
%  prepSynthHard( type )
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

[dPath,ch,ch1,chC]=globals;
RandStream.setDefaultStream(RandStream('mrg32k3a', 'Seed', sum(type)));
% render 500 instances per class at size 100 pixels
n=500; sz=100;
for k=1:length(ch)
  if(ch(k)=='_'), continue; end
  I=zeros(sz,sz,3,n,'uint8'); k0=1;
  fInfo=zeros(n,3);
  for i=1:n % n synthetic examples per character
    if strcmp(type,'train')    
      [I(:,:,:,k0),fInfo(k0,1),fInfo(k0,2),fInfo(k0,3)]=genChar(ch(k),sz, [ch, ' '], [1 500]); k0=k0+1;
    else
      [I(:,:,:,k0),fInfo(k0,1),fInfo(k0,2),fInfo(k0,3)]=genChar(ch(k),sz, [ch, ' '], [500 500]); k0=k0+1;        
    end
  end
  y=k*ones(1,n); y=y(:);
  writeAllImgs(I,y,chC,fullfile(dPath,'synth_w_fonts',type,'charHard'));
  lbl=ch(k);
  if(lbl>=97&&lbl<=122), lbl=['-' lbl]; end
  dlmwrite(fullfile(dPath,'synth_w_fonts',type,'charHard',lbl,'fids.dat'),fInfo);
  clear I;
end
end