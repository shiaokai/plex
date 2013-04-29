function cfg=TRAINsynth1000_TESTicdar_cfg

cfg=struct();
cfg.train='synth1000';
cfg.train_bg='msrc';
cfg.train_type='char';

cfg.test='icdar';
cfg.lex='lex50';
cfg.lex0='lex0';
cfg.test_type='charHard';

% use parfor or not
cfg.has_par=1;