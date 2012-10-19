function [f,x,y,i]=Fscore(xs,ys)
% Compute F-score
%
% USAGE
%  [f,x,y,i]=Fscore( xs, ys )
%
% INPUTS
%  xs     - precision
%  ys     - recall
%
% OUTPUTS
%  f      - fscore
%  x      - precision at best fscore
%  y      - recall at best fscore
%  i      - index of best fscore
%
% CREDITS
%  Written and maintained by Kai Wang and Boris Babenko
%  Copyright notice: license.txt
%  Changelog: changelog.txt
%  Please email kaw006@cs.ucsd.edu if you have questions.

fs=1./(.5./xs+.5./ys);
[f,i]=max(fs); x=xs(i); y=ys(i);