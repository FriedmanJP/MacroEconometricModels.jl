% test/oracle/octave/ref_forecast.m — reference unconditional (no-shock) VAR forecast.
% Run from repo root:  octave --no-gui test/oracle/octave/ref_forecast.m
source('test/oracle/octave/_setup.m');
y = csvread(fullfile(DATADIR, 'synthetic_var.csv'));
[T, n] = size(y); p = 2;
v = rfvar3(y, p, ones(T,1), [], 0, 0);
Sigma = (v.u' * v.u) / size(v.u,1);
fhor = 12;
fd.initval = y(end-p+1:end, :);     % last p obs, chronological (lags x ny)
fd.xdata   = ones(fhor, 1);         % constant path
[nf, ~] = forecasts(fd, v.B, Sigma, fhor, p);   % nf: fhor x ny  (no-shock)
csvwrite(fullfile(OUTDIR, 'fcast_noshock_ref.csv'), nf);
printf('ref_forecast OK: fhor=%d\n', fhor);
