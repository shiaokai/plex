function cfg=TRAINsynthbg8x_TESTicdar_cfg

cfg=struct();
cfg.train='synth8x';
cfg.train_bg='msrc';
cfg.train_type='char-bg';

cfg.test='icdar';
cfg.lex='lex50';
cfg.lex0='lex0';
cfg.test_type='charHard';

cfg.has_par=0;