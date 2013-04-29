function cfg=TRAINicdar_TESTsvt_cfg

cfg=struct();
cfg.train='icdar';
cfg.train_bg='icdar';
cfg.train_type='charHard';

cfg.test='svt';
cfg.lex='lex';
cfg.lex0='lex';
cfg.test_type='';

% use parfor or not
cfg.has_par=0;
