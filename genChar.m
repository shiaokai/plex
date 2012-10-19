function [P fontid rweight rangle]=genChar(c,sz,alphabet,pos)
% Generate a character with noise
%
% For this to work, you must first run validateFonts.m to get a list of
% fonts that your machine can properly display.
%
% USAGE
%  [P,fontid,rweight,rangle] = genChar(c, sz, alphabet,pos)
%
% INPUTS
%  c          - specifies the character to generate
%  sz         - specifies the height and width
%  alphabet   - specifies random characters to generate to left and right
%               if none is specified, then no other chars are rendered.
%               This can/should contain space characters.
%  pos        - [1,1] location to open the figure for rendering
%
% OUTPUTS
%  P          - image of rendered character
%  fontid     - id for the font during render
%  rweight    - weight during render
%  rangle     - angle during render
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

if nargin < 3, error('Not enough input arguments'); end
if nargin < 4, pos = [1,1]; end

if ~exist('validfonts.mat', 'file')
    error('You must first run validateFonts.m to filter font list');
end

weights={'normal','bold'}; fangles={'normal','italic'};
r=@(mn,mx) (mx-mn)*rand+mn;
bgH=rand; bgS=r(.5,1); bgV=[r(.1,.3) r(.7,.9)]; bgV=bgV((rand>0.5)+1); 
fgS=r(.5,1); fgH=rand; fgV=r(.1,.9);

% make sure there's enough contrast
while(abs(bgV-fgV)<.3 || min(abs(bgH-fgH),1-abs(bgH-fgH))<.4 )
  fgH=rand; if(bgV<.5), fgV=r(.5,1); else fgV=r(0,.5); end
end

% select left and right chars
lch = ' '; rch = ' ';
if nargin >= 3
    lind = randi([1,length(alphabet)]);
    rind = randi([1,length(alphabet)]);
    lch = alphabet(lind);
    rch = alphabet(rind);
    while(lch=='_'), lind=randi([1,length(alphabet)]); lch=alphabet(lind); end
    while(rch=='_'), rind=randi([1,length(alphabet)]); rch=alphabet(rind); end    
end
renderstr = [lch, c, rch];

bg=hsv2rgb([bgH bgS bgV]); fg=hsv2rgb([fgH fgS fgV]);
P=repmat(permute(bg,[1 3 2]),[sz,sz,1]);

% choose a random font that will actually render
load('validfonts', 'validfonts');
fontid = randi([1,length(validfonts)]);
fontname = validfonts{fontid};

% render left character alone for alignment
rweight = (rand>0.5)+1;
rangle = (rand>0.5)+1;
rsz = sz/r(1,1.5);

% get dimensions of left character
hf=figure('Visible','off'); clf; im(P,[],0); 
truesize; 
hold on;
ht=text(0,0,renderstr(1),'fontsize', rsz,'color',fg,...
  'fontweight',weights{rweight},....
  'fontangle',fangles{rangle},...
  'fontname',fontname,...
  'horizontalalignment','center','units','pixels');
left_e=get(ht,'Extent');
close(hf);

% get dimensions of center character
hf=figure('Visible','off'); clf; im(P,[],0); truesize; hold on;
ht=text(0,0,renderstr(2),'fontsize',rsz,'color',fg,...
  'fontweight',weights{rweight},....
  'fontangle',fangles{rangle},...
  'fontname',fontname,...
  'horizontalalignment','center','units','pixels');
mid_e=get(ht,'Extent');
close(hf);

hf=figure('Visible','off'); clf; im(P,[],0); 
truesize; 
hold on;
ht=text(0,0,renderstr,'fontsize',rsz,'color',fg,...
  'fontweight',weights{rweight},....
  'fontangle',fangles{rangle},...
  'fontname',fontname,...
  'horizontalalignment','center','units','pixels');
all_e=get(ht,'Extent'); 
c=[all_e(1)+left_e(3)+mid_e(3)/2,all_e(2)+all_e(4)/2];
set(ht,'Position',[sz/2-c(1),sz/2-c(2)]);
curp=get(hf,'Position'); 
set(hf,'Position',[pos, curp(3:4)]);

tim=getframe(gca);
%tim2 = im2frame(zbuffer_cdata(hf));
close(hf); 
P=tim.cdata;

P=P(2:end-1,2:end-1,:); P=padarray(P,[sz sz],'replicate');
th=r(-5,5); P=fevalArrays(P,@(I)imtransform2(I,th,'linear','crop'));
P=double(P)+randn(size(P))*25*rand; 
b=r(.5,2); P=uint8(gaussSmooth(P,[b b 0],'same'));
P=uint8(P(sz:end-sz,sz:end-sz,:));

end

% HAVEN'T GOTTEN THIS TO WORK 100%
% Has weird alignment issues somewhat randomly
function cdata = zbuffer_cdata(hfig)
% Get CDATA from hardcopy using zbuffer

% Need to have PaperPositionMode be auto 
orig_mode = get(hfig, 'PaperPositionMode');
set(hfig, 'PaperPositionMode', 'auto');

cdata = hardcopy(hfig, '-Dzbuffer', '-r0');

% Restore figure to original state
set(hfig, 'PaperPositionMode', orig_mode); % end
end
