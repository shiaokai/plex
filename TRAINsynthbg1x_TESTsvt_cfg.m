function cfg=TRAINsynthbg1x_TESTsvt_cfg

cfg=struct();
cfg.train='synth1x';
cfg.train_bg='msrc';
cfg.train_type='char-bg';

cfg.test='svt';
cfg.lex='lex';
cfg.lex0='lex';
cfg.test_type='';

cfg.has_par=0;
