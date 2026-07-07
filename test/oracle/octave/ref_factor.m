% test/oracle/octave/ref_factor.m — reference Bai-Ng (baing.m) + PCA common component.
% Standardizes ONCE here and saves it, so both stacks use identical (already-standardized) data
% and the comparison does not depend on either side's standardization routine.
% Run from repo root:  octave --no-gui test/oracle/octave/ref_factor.m
source('test/oracle/octave/_setup.m');
randn('seed', 7);
T = 200; N = 50; rtrue = 3;
F   = randn(T, rtrue);
Lam = randn(N, rtrue);
X   = F * Lam' + randn(T, N);        % approximate factor model
Xs  = (X - mean(X)) ./ std(X);       % standardize once (no NaNs ⇒ avoids nanstd dependency)
csvwrite(fullfile(DATADIR, 'factor_panel.csv'), Xs);

kmax = 8;
IC = zeros(kmax, 3);                 % columns = jj=1,2,3 ; rows = r=1..kmax
sel = zeros(1, 3);
for jj = 1:3
  res = baing(Xs, kmax, jj, 0);      % DEMEAN=0: data already standardized
  IC(:, jj) = res.IC1(1:kmax)';
  sel(jj)   = res.ic1;
end
csvwrite(fullfile(OUTDIR, 'factor_IC.csv'),  IC);
csvwrite(fullfile(OUTDIR, 'factor_sel.csv'), sel);

% PCA common component (projection) for r = 3
[U, ~, ~] = svd(Xs * Xs');
fhat   = U(:, 1:rtrue) * sqrt(T);
lambda = Xs' * fhat / T;
chat3  = fhat * lambda';             % = U_r U_r' Xs  (true projection)
Sigma3 = mean(sum((Xs - chat3).^2 / T));
csvwrite(fullfile(OUTDIR, 'factor_chat3.csv'),  chat3);
csvwrite(fullfile(OUTDIR, 'factor_Sigma3.csv'), Sigma3);
printf('ref_factor OK: selected (IC1,IC2,IC3) = %d %d %d\n', sel(1), sel(2), sel(3));
