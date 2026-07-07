% test/oracle/octave/ref_filters.m — reference HP and Hamilton filters on fixture column 1.
% Run from repo root:  octave --no-gui test/oracle/octave/ref_filters.m
source('test/oracle/octave/_setup.m');
y = csvread(fullfile(DATADIR, 'synthetic_var.csv'));
x = y(:, 1);
% HP filter (trend), lambda = 1600
s = Hpfilter(x, 1600);
csvwrite(fullfile(OUTDIR, 'hp_trend_ref.csv'), s);
% Hamilton regression filter: h=8, d=4, constant=1, no fig
[dyc, dyt] = hamfilter(x, 8, 4, 1, 0);
csvwrite(fullfile(OUTDIR, 'ham_cycle_ref.csv'), dyc(8+4:end));   % valid sample = d+h:end
csvwrite(fullfile(OUTDIR, 'ham_trend_ref.csv'), dyt(8+4:end));
printf('ref_filters OK\n');
