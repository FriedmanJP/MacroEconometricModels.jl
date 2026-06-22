% test/oracle/octave/ref_irf_fevd.m — reference Cholesky IRF + FEVD on the fixture VAR.
% Feeds the SAME (alpha, Sigma) our Julia model uses (ML Sigma = u'u/N). Run from repo root.
source('test/oracle/octave/_setup.m');
y = csvread(fullfile(DATADIR, 'synthetic_var.csv'));
[T, n] = size(y); p = 2;
v = rfvar3(y, p, ones(T,1), [], 0, 0);
u = v.u; N = size(u,1);
Sigma = (u' * u) / N;            % ML covariance == our model.Sigma (F-01 form)
alpha = v.B(1:n*p, :);           % AR part (np x n) = [lag1; lag2], == our B(2:end,:)
hor = 24;
ir = iresponse(alpha, Sigma, hor, eye(n));   % n x hor x n  (var, horizon, shock); 1-std
FE = fevd(hor, alpha, Sigma, eye(n));        % n x n FEVD shares*100 at horizon `hor`
ir_lr = iresponse_longrun(alpha, Sigma, hor, p);   % Blanchard-Quah long-run IRF
csvwrite(fullfile(OUTDIR, 'irf_lr_ref.csv'), ir_lr(:));
csvwrite(fullfile(OUTDIR, 'irf_ref.csv'),   ir(:));     % column-major flatten of (n,hor,n)
csvwrite(fullfile(OUTDIR, 'irf_dims.csv'),  [n hor n]);
csvwrite(fullfile(OUTDIR, 'fevd_ref.csv'),  FE);
csvwrite(fullfile(OUTDIR, 'irf_alpha.csv'), alpha);
csvwrite(fullfile(OUTDIR, 'irf_Sigma.csv'), Sigma);
printf('ref_irf_fevd OK: hor=%d n=%d\n', hor, n);
