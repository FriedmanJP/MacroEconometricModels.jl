% test/oracle/octave/ref_matrictint.m — reference NIW log-integral on a fixed case.
% Run from repo root:  octave --no-gui test/oracle/octave/ref_matrictint.m
source('test/oracle/octave/_setup.m');
S   = [2.0 0.3 0.1; 0.3 1.5 0.2; 0.1 0.2 1.8];   % 3x3 SPD
XXi = diag([1.0 2.0 0.5 1.5 0.8]);               % 5x5 SPD
df  = 25;
w = matrictint(S, df, XXi);
csvwrite(fullfile(OUTDIR, 'matrictint_w.csv'),   w);
csvwrite(fullfile(OUTDIR, 'matrictint_S.csv'),   S);
csvwrite(fullfile(OUTDIR, 'matrictint_XXi.csv'), XXi);
csvwrite(fullfile(OUTDIR, 'matrictint_df.csv'),  df);
printf('ref_matrictint OK: w=%.10f\n', w);
