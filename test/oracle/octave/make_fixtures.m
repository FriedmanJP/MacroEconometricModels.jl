% test/oracle/octave/make_fixtures.m
% Generate one deterministic stationary VAR(2) dataset shared by both stacks.
% Run from repo root:  octave --no-gui test/oracle/octave/make_fixtures.m
source('test/oracle/octave/_setup.m');
randn('seed', 42);
T = 300; n = 3; p = 2;
A1 = [0.5 0.1 0.0; 0.0 0.4 0.1; 0.1 0.0 0.3];
A2 = [-0.1 0.0 0.0; 0.0 -0.1 0.0; 0.0 0.0 -0.1];
c   = [0.2; 0.0; -0.1];
Sig = [1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0];
L = chol(Sig, 'lower');
y = zeros(T, n);
y(1:2, :) = randn(2, n);
for t = 3:T
  y(t, :) = (c + A1*y(t-1,:)' + A2*y(t-2,:)' + L*randn(n,1))';
end
csvwrite(fullfile(DATADIR, 'synthetic_var.csv'), y);
printf('fixtures OK: synthetic_var is %dx%d\n', size(y,1), size(y,2));
