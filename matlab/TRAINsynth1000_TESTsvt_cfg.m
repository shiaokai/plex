function cfg=TRAINsynth1000_TESTsvt_cfg

cfg=struct();
cfg.train='synth1000';
cfg.train_bg='msrc';
cfg.train_type='char';

cfg.test='svt';
cfg.lex='lex';
cfg.lex0='lex';
cfg.test_type='';

% use parfor or not
cfg.has_par=1;
