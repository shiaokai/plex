function str=prm2str(params)
% Simple function to strip weird characters out of parameter cells
% so that they can be more easily displayed as a string for debugging

str='';
for i=1:length(params)
  switch class(params{i})
    case 'char'
      str=sprintf('%s%s',str,params{i});
    case 'double'
      str=sprintf('%s%d',str,params{i});
    otherwise
      error('prm2str: unsupported type');
  end
end


% str=in;
% % filter whitespace
% str=str(str~=' ');
% % filter braces
% str=str(str~='[');
% str=str(str~=']');
% % filter apostrophe
% str=str(str~='''');
% % trim just in case
% str=strtrim(str);