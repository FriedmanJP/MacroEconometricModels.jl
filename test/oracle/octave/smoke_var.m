% test/oracle/octave/smoke_var.m
% Smoke test: prove the BVAR_ reference toolbox loads in Octave and its core
% reduced-form VAR estimator (rfvar3) runs end to end. Run from repo root:
%   octave --no-gui test/oracle/octave/smoke_var.m
source('test/oracle/octave/_setup.m');
randn('seed', 0);
T = 200; n = 2; p = 2;
y = cumsum(randn(T, n));                 % smoke data
xdata = ones(T, 1);                      % constant goes in xdata (rfvar3 adds no const)
v = rfvar3(y, p, xdata, [], 0, 0);       % no breaks, no persistence dummies
csvwrite(fullfile(OUTDIR, 'smoke_var_B.csv'), v.B);
printf('smoke_var OK: B is %dx%d\n', size(v.B,1), size(v.B,2));
