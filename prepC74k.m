function prepC74k
% Process the raw files downloaded from C74k into a common format
% Download site,
%   http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
%
% Move the English and Lists folder here,
%  [dPath]/c74k/raw/
% After moving, the folder should look like,
%  [dPath]/c74k/raw/English/.
%  [dPath]/c74k/raw/Lists/.
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

[dPath,ch,ch1,chC]=globals;
iPath = fullfile(dPath, 'c74k/raw/English/Img');
load(fullfile(iPath, 'lists.mat'));

% the last column contains the indices of the 15/class training
traininds = list.TRNind(:,16);
trainlabs = list.ALLlabels(traininds);
trainnames = list.ALLnames(traininds,:);

% loop through testing, and extract features and find nearest
testinds = list.TSTind(:,16);
testlabs = list.ALLlabels(testinds);
testnames = list.ALLnames(testinds,:);

% assume data is in a folder one level below
iPath = fullfile(dPath, 'c74k/raw/English/Img');
sz=100; I=zeros(sz,sz,3,length(traininds));

for i = 1:size(traininds,1)
    I1 = imread(fullfile(iPath, [trainnames(i,:), '.png']));
    bb=[1 1 size(I1,2) size(I1,1)];
    bb=bbApply('squarify',bb,3,1);
    P=bbApply('crop',I1,bb,'replicate',[sz sz]); P=P{1};
    if(size(P,3)==1), P=cat(3,P,P,P); end
    I(:,:,:,i)=P;
end

figure(1); montage2(uint8(I(:,:,:,:)),struct('hasChn',1));
writeAllImgs(I,trainlabs,chC,fullfile(dPath,'c74k','train','char'));

I=zeros(sz,sz,3,length(traininds));
for i = 1:size(traininds,1)
    I1 = imread(fullfile(iPath, [testnames(i,:), '.png']));
    bb=[1 1 size(I1,2) size(I1,1)];
    bb=bbApply('squarify',bb,3,1);
    P=bbApply('crop',I1,bb,'replicate',[sz sz]); P=P{1};
    if(size(P,3)==1), P=cat(3,P,P,P); end
    I(:,:,:,i)=P;
end

figure(2); montage2(uint8(I(:,:,:,:)),struct('hasChn',1));
writeAllImgs(I,testlabs,chC,fullfile(dPath,'c74k','test','char'));

