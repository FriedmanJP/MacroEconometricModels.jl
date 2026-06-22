% test/oracle/octave/ref_var.m — reference reduced-form VAR via rfvar3.
% Run from repo root:  octave --no-gui test/oracle/octave/ref_var.m
source('test/oracle/octave/_setup.m');
y = csvread(fullfile(DATADIR, 'synthetic_var.csv'));
[T, n] = size(y); p = 2;
xdata = ones(T, 1);                       % intercept (rfvar3 adds no constant)
v = rfvar3(y, p, xdata, [], 0, 0);        % no breaks, no persistence dummies
u = v.u;
N = size(u, 1); K = size(v.B, 1);         % N = T-p effective obs, K = n*p+1 regressors
Serror_dof = (u' * u) / (N - K);          % ols_reg.m default: dof-adjusted (N-K)
Serror_ml  = (u' * u) / N;                % ML denominator
% companion eigenvalues (ref B rows = [lag1; ...; lagp; const]; drop const row)
A = v.B(1:n*p, :)';                       % n x (n*p) = [A1 A2 ...]
F = [A; eye(n*(p-1)), zeros(n*(p-1), n)];
ev = sort(abs(eig(F)));
csvwrite(fullfile(OUTDIR, 'var_B.csv'),          v.B);
csvwrite(fullfile(OUTDIR, 'var_u.csv'),          u);
csvwrite(fullfile(OUTDIR, 'var_Serror_dof.csv'), Serror_dof);
csvwrite(fullfile(OUTDIR, 'var_Serror_ml.csv'),  Serror_ml);
csvwrite(fullfile(OUTDIR, 'var_eig.csv'),        ev);
printf('ref_var OK: N=%d K=%d\n', N, K);
