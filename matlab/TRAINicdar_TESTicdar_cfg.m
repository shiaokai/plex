function cfg=TRAINicdar_TESTicdar_cfg

cfg=struct();
cfg.train='icdar';
cfg.train_bg='icdar';
cfg.train_type='charHard';

cfg.test='icdar';
cfg.lex='lex50';
cfg.lex0='lex0';
cfg.test_type='charHard';

% use parfor or not
cfg.has_par=0;