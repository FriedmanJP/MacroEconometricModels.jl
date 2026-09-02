% ref_identification.m — BVAR_ proxy / long-run / sign-set references.
% Run from repo root:  octave --no-gui test/oracle/octave/ref_identification.m
% Writes CSVs to test/oracle/identification_ref/ (committed, bit-reproducible).
source('test/oracle/octave/_setup.m');
REFDIR = fullfile(pwd, 'test', 'oracle', 'identification_ref');
if ~exist(REFDIR, 'dir'); mkdir(REFDIR); end

y = csvread(fullfile(REFDIR, 'Y.csv'));
zfull = csvread(fullfile(REFDIR, 'z.csv'));
[Tfull, n] = size(y);
p = 1;
z = zfull((p+1):end, :);                 % align with residuals (T_eff x 1)

xdata = ones(Tfull, 1);
v = rfvar3(y, p, xdata, [], 0, 0);
u = v.u;
N = size(u, 1);
Sigma = (u' * u) / N;                    % ML covariance == MEMs model.Sigma
alpha = v.B(1:n*p, :);                   % [lag1; ...; lagp] == our B(2:end,:)
hor = 8;

% --- Proxy (Mertens-Ravn / iresponse_proxy) ---
in = struct();
in.vars = y;
in.p = p;
in.proxies = z;
in.T_m_end = 0;
in.irhor = hor;
in.res = u;
in.Phi = v.B;
in.Sigma = Sigma;
in.compute_F_stat = 1;
pr = iresponse_proxy(in);
csvwrite(fullfile(REFDIR, 'proxy_b1.csv'), pr.b1);
csvwrite(fullfile(REFDIR, 'proxy_F.csv'), pr.F_m);

% --- Long-run Blanchard-Quah ---
[ir_lr, Q_lr] = iresponse_longrun(alpha, Sigma, hor, p);
csvwrite(fullfile(REFDIR, 'long_run_Q.csv'), Q_lr);
csvwrite(fullfile(REFDIR, 'long_run_irf.csv'), ir_lr(:));

% --- Sign restrictions: many Haar draws, store impact of shock 1 ---
% Restriction: shock 1 raises variables 1 and 2 on impact (Rubio-Ramirez syntax).
signrestriction = cell(1, 2);
signrestriction{1} = 'y(1,1,1)>0;';
signrestriction{2} = 'y(2,1,1)>0;';
n_acc = 0;
n_draw = 2000;
impacts = nan(n, n_draw);
randn('seed', 75503);
for d = 1:n_draw
    [ir, Omeg] = iresponse_sign(alpha, Sigma, hor, signrestriction);
    if any(isnan(Omeg(:)))
        continue
    end
    n_acc = n_acc + 1;
    impacts(:, n_acc) = ir(:, 1, 1);     % impact of shock 1
end
impacts = impacts(:, 1:n_acc);
lo = quantile(impacts, 0.16, 2);
hi = quantile(impacts, 0.84, 2);
csvwrite(fullfile(REFDIR, 'sign_impact_p16.csv'), lo);
csvwrite(fullfile(REFDIR, 'sign_impact_p84.csv'), hi);
csvwrite(fullfile(REFDIR, 'sign_n_accepted.csv'), n_acc);

printf('ref_identification OK: proxy b1(1)=%.6f  BQ Q(1,1)=%.6f  sign acc=%d/%d\n', ...
       pr.b1(1), Q_lr(1,1), n_acc, n_draw);
