function cfg=TRAINsynth_TESTsvt_cfg

cfg=struct();
% training set
cfg.train='synth';
% background dataset
cfg.train_bg='msrc';
% name of the character folder
cfg.train_type='char';

% test set
cfg.test='svt';
% lexicon test set
cfg.lex='lex';
% lexicon tune set
cfg.lex0='lex';
% name of the character folder
cfg.test_type='';

% use parfor or not
cfg.has_par=1;
