function cfg=TRAINsynth_TESTicdar_cfg

cfg=struct();
% training set
cfg.train='synth';
% background dataset
cfg.train_bg='msrc';
% name of the character folder
cfg.train_type='charHard';

% test set
cfg.test='icdar';
% lexicon test set
cfg.lex='lex50';
% lexicon tune set
cfg.lex0='lex0';
% name of the character folder
cfg.test_type='charHard';

% use parfor or not
cfg.has_par=1;