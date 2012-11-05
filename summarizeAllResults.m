function summarizeAllResults

colors={'b','g','r','c','m','y','k','w'};

test='icdar'; % icdar or svt

switch test
  case 'icdar'
    cfgSynth2x=globals(TRAINsynth2x_TESTicdar_cfg);
    cfgSynth4x=globals(TRAINsynth4x_TESTicdar_cfg);
    cfgSynth8x=globals(TRAINsynth8x_TESTicdar_cfg);
    cfgSynth16x=globals(TRAINsynth16x_TESTicdar_cfg);
    
    % plot PR curves together
    figure(1); clf; hold on; title('ICDAR wordspotting performance');
    res=load(cfgSynth2x.resWordspot()); fs2x=Fscore(res.xs,res.ys);
    plot(res.xs,res.ys,colors{1});
    res=load(cfgSynth4x.resWordspot()); fs4x=Fscore(res.xs,res.ys);
    plot(res.xs,res.ys,colors{2});
    res=load(cfgSynth8x.resWordspot()); fs8x=Fscore(res.xs,res.ys);
    plot(res.xs,res.ys,colors{3});
    res=load(cfgSynth16x.resWordspot()); fs16x=Fscore(res.xs,res.ys);
    plot(res.xs,res.ys,colors{4});
    legend({sprintf('2x-%.3f',fs2x),sprintf('4x-%.3f',fs4x),...
      sprintf('8x-%.3f',fs8x),sprintf('16x-%.3f',fs16x)});
    
    % char classification
    res=load(cfgSynth2x.resCharClf()); fprintf('%s\n',res.msg3);
    res=load(cfgSynth4x.resCharClf()); fprintf('%s\n',res.msg3);
    res=load(cfgSynth8x.resCharClf()); fprintf('%s\n',res.msg3);
    res=load(cfgSynth16x.resCharClf()); fprintf('%s\n',res.msg3);
    
  case 'svt'
    cfgSynth2x=globals(TRAINsynth2x_TESTsvt_cfg);
    cfgSynth4x=globals(TRAINsynth4x_TESTsvt_cfg);
    cfgSynth8x=globals(TRAINsynth8x_TESTsvt_cfg);
    cfgSynth16x=globals(TRAINsynth16x_TESTsvt_cfg);
    
    % plot PR curves together
    figure(1); clf; hold on; title('SVT wordspotting performance');
    res=load(cfgSynth2x.resWordspot()); fs2x=Fscore(res.xs,res.ys); 
    plot(res.xs,res.ys,colors{1});
    res=load(cfgSynth4x.resWordspot()); fs4x=Fscore(res.xs,res.ys);
    plot(res.xs,res.ys,colors{2});
    res=load(cfgSynth8x.resWordspot()); fs8x=Fscore(res.xs,res.ys);
    plot(res.xs,res.ys,colors{3});
    res=load(cfgSynth16x.resWordspot()); fs16x=Fscore(res.xs,res.ys);
    plot(res.xs,res.ys,colors{4});
    legend({sprintf('2x-%.3f',fs2x),sprintf('4x-%.3f',fs4x),...
      sprintf('8x-%.3f',fs8x),sprintf('16x-%.3f',fs16x)});
    
end
  
end
