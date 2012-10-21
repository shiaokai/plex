function str=filterDescription(in)
% Simple function to strip weird characters out of parameter cells
% so that they can be more easily displayed as a string for debugging

str=in;
% filter whitespace
str=str(str~=' ');
% filter braces
str=str(str~='[');
str=str(str~=']');
% filter apostrophe
str=str(str~='''');
% trim just in case
str=strtrim(str);