# [Simulation (DGPs)](@id simulation_page)

**MacroEconometricModels.jl** ships public data-generating processes (DGPs) for Monte Carlo experiments and estimator recovery checks. Every simulator draws from an explicit `rng`, discards a burn-in, and returns a NamedTuple that pairs the simulated sample with the population truth — coefficients, covariances, shocks, and latent paths — so a test or example asserts against known values instead of probed constants.

- **VAR**: stationary `dgp_var` plus population IRF, FEVD, historical decomposition, and stationary covariance helpers
- **SVAR identification**: non-Gaussian and heteroskedastic VARs with the shock features statistical identification needs
- **Univariate**: seasonal ARIMA, trend-cycle, spectral-pair, state-space, and hypothesis-test-pair designs
- **Cointegration**: VECM, cointegrating regression, panel VAR, ARDL, NARDL, and panel PMG designs
- **Volatility**: nine-kind GARCH family, stochastic volatility, multivariate GARCH, and MIDAS designs
- **Factors**: dynamic factor model and mixed-frequency (Mariano–Murasawa) ragged-edge panels
- **Local projections**: LP-IV, state-dependent VAR, propensity-score, and HAC-regression designs
- **Micro**: fifteen-kind cross-section, Mundlak panel, and staggered difference-in-differences designs
- **Regimes**: Markov-switching, SETAR, LSTAR, and ESTR designs with the true state path
- **GMM and measurement**: heteroskedastic GMM, PCE band draws, and DSGE observation-error designs
- **Analytic truth**: closed-form spectra, temporal aggregation, and marginal-effect helpers

```@setup simulation
using MacroEconometricModels, Random, LinearAlgebra, Statistics
```

## Quick Start

**Recipe 1: Simulate a VAR and check the covariance recovery**

```@example simulation
sim11 = dgp_var(MersenneTwister(11); T=200)
fit11 = estimate_var(sim11.Y, 1)
round(maximum(abs.(fit11.Sigma - sim11.Sigma)); digits=3)
```

**Recipe 2: Compare estimated and population IRFs**

```@example simulation
est_irf = compute_irf(fit11, Matrix{Float64}(I, 3, 3), 11)
round(maximum(abs.(est_irf - var_irf(sim11.A, sim11.B0, 10))); digits=3)
```

**Recipe 3: Heavy-tailed volatility in one line**

```@example simulation
g = dgp_garch_family(MersenneTwister(2); kind=:garch, innov=:t, T=500)
(size(g.y), round(mean(g.h); digits=3))
```

**Recipe 4: Staggered policy adoption**

```@example simulation
d = dgp_staggered_did(MersenneTwister(3))
(size(d.df), round(d.overall_att; digits=3))
```

**Recipe 5: A factor panel with known loadings**

```@example simulation
f = dgp_dynamic_factors(MersenneTwister(4); N=20, T=200)
(size(f.X), size(f.F))  # 20 series driven by 2 factors
```

---

## The Simulator Contract

Every simulator in this guide obeys one contract:

- **Explicit random numbers.** The first positional argument is `rng::AbstractRNG`, and no simulator touches the global RNG. Fix the seed and the draw reproduces bit-for-bit. (The analytic truth helpers `arma_spectrum`, `mm_aggregate`, `logit_ame`, and `probit_ame` are deterministic functions of their inputs and take no `rng`.)
- **Burn-in for dynamics.** Dynamic simulators discard a `burn` keyword of pre-sample draws (default 200 for most; 50 for panel designs, 100 for ARDL and regime-switching, 500 for the GARCH family). Static designs — cross-section, panel, LP-IV, propensity, GMM — draw independently and take no burn-in.
- **NamedTuple truth.** Every `dgp_*` simulator returns a NamedTuple with the sample plus the population parameters a recovery assertion needs: coefficients, covariances, shocks, and latent paths. Field names below match the code exactly.

---

## VAR Simulation and Population Moments

The stationary VAR(p) (Sims 1980; Lütkepohl 2005) is the workhorse Monte Carlo design:

```math
Y_t = c + A_1 Y_{t-1} + \ldots + A_p Y_{t-p} + u_t, \quad u_t = B_0 \varepsilon_t, \quad \varepsilon_t \sim N(0, I)
```

where:

- ``Y_t`` is the ``n \times 1`` vector of endogenous variables
- ``A_i`` are ``n \times n`` coefficient matrices and ``c`` is the ``n \times 1`` intercept
- ``B_0`` is the ``n \times n`` impact matrix with ``\Sigma = B_0 B_0'``
- ``\varepsilon_t`` are the structural shocks returned for exact shock-recovery asserts

```@example simulation
using MacroEconometricModels, Random
sim = dgp_var(MersenneTwister(11); T=100)
size(sim.Y)  # (100, 3)
```

The sample has 100 rows and 3 columns. `dgp_var` returns `(Y, eps, A, Sigma, B0, c)`; pass exactly one of `B0` and `Sigma` (`Sigma` alone takes the lower Cholesky factor). The default design has non-diagonal dynamics and a non-identity impact matrix, so shock orderings and rotations matter. The companion helpers `var_irf`, `var_fevd`, `var_hd`, and `lyapunov_gamma0` compute the population moments estimators target:

```@example simulation
big = dgp_var(MersenneTwister(12); T=500)
m = estimate_var(big.Y, 1)
Q0 = Matrix{Float64}(I, 3, 3)  # recursive ordering matches the triangular B0
irf_err = maximum(abs.(compute_irf(m, Q0, 11) - var_irf(big.A, big.B0, 10)))
round(irf_err; digits=3)  # estimation error, shrinks with T
```

OLS estimation followed by a recursive identification recovers the population IRF up to estimation error. `var_irf(A, B0, H)` returns the `(H+1)×n×n` structural moving-average array, `var_fevd(A, B0, H)` the variance-decomposition shares (rows sum to 1), `var_hd(A, B0, eps; c)` the `T×n×n` historical shock contributions, and `lyapunov_gamma0(A, Sigma)` the stationary covariance:

```@example simulation
G0 = lyapunov_gamma0(big.A[1], big.Sigma)
fevd = var_fevd(big.A, big.B0, 10)
hd = var_hd(big.A, big.B0, big.eps; c=big.c)
(size(G0), size(fevd), size(hd))
```

!!! note "Technical Note"
    `var_hd` sums exactly to the demeaned sample only from a zero initial deviation. With burn-in the early observations additionally carry the initial-condition term, so recovery asserts should skip the first rows or set `burn = 0` with `c = 0`.

See [VAR](@ref var_page) for estimation, [Impulse Responses](@ref ia_irf_page), [Variance Decomposition](@ref ia_fevd_page), and [Historical Decomposition](@ref ia_hd_page) for the estimators these moments validate, and [Bayesian VAR](@ref bvar_page) for posterior recovery checks.

---

## Non-Gaussian and Heteroskedastic SVARs

Gaussian white noise leaves every non-Gaussian or heteroskedasticity-based identification scheme unidentified in population. These two simulators give each method data with the feature it needs (Lanne, Lütkepohl & Maciejowska 2010; Lütkepohl & Netsunajev 2017):

```math
Y_t = A Y_{t-1} + B_0 \Lambda_t^{1/2} \varepsilon_t
```

where:

- ``Y_t`` is the ``n \times 1`` vector of endogenous variables
- ``B_0`` is the ``n \times n`` impact matrix shared across regimes
- ``\Lambda_t`` is the diagonal variance path (identity under non-Gaussianity, time-varying under heteroskedasticity)
- ``\varepsilon_t`` are independent shocks, Gaussian or non-Gaussian

```@example simulation
ng = dgp_nongaussian_var(MersenneTwister(21); dist=:t, T=500)
hs = dgp_heteroskedastic_var(MersenneTwister(22); kind=:markov, T=500)
(size(ng.Y), ng.dist, size(hs.scales))  # ((500, 3), :t, (500, 3))
```

`dgp_nongaussian_var` draws independent shocks from `dist ∈ :gauss|:t|:laplace|:mixture|:skew` (default `:t` with 5 degrees of freedom) and returns `(Y, eps, A, Sigma, B0, dist)` with `Sigma = B0*B0'`. `dgp_heteroskedastic_var` scales Gaussian shocks by a variance path `kind ∈ :markov|:garch|:smooth|:external` and returns `(Y, eps, scales, B0, Sigma_full, Lambda, path, kind)`, where `scales` is the `T×n` variance path and `path` the regime or transition descriptor. The default `Lambda` has genuinely distinct eigenvalue ratios, so the heteroskedasticity carries identifying information.

See [Non-Gaussian Methods](@ref id_nongaussian_page), [Heteroskedasticity-Based Identification](@ref id_heteroskedastic_page), and [Statistical Identification](@ref nongaussian_page) for the estimators that consume these designs.

---

## Univariate Time Series

Six simulators cover ARIMA, trend-cycle, spectral, state-space, and hypothesis-test designs (Hamilton 1994; Harvey 1989):

```math
y_t = c + \phi_1 y_{t-1} + \ldots + \phi_p y_{t-p} + e_t + \theta_1 e_{t-1} + \ldots + \theta_q e_{t-q}, \quad e_t \sim N(0, \sigma^2)
```

where:

- ``y_t`` is the scalar observation at time ``t``
- ``\phi`` and ``\theta`` are the AR and MA lag polynomials
- ``d`` and the seasonal pair ``(s, \Phi, \Theta)`` add integration and seasonal dynamics
- ``\sigma`` scales the Gaussian innovation

```@example simulation
ar = dgp_arima(MersenneTwister(31); phi=[0.7], theta=[0.2], T=300)
tc = dgp_trend_cycle(MersenneTwister(32); T=200)
ss = dgp_state_space(MersenneTwister(33); T=200)
(length(ar.y), length(tc.y), size(ss.y))  # (300, 200, (200, 1))
```

`dgp_arima` returns `(y, phi, theta, d, Phi, Theta, s, c, sigma)`. `dgp_trend_cycle` splits the sample into `(y, trend, cycle, phi)` with a random-walk-with-drift or linear trend plus an AR(2) cycle with complex roots. `dgp_state_space` simulates the linear Gaussian system `x_{t+1} = b + F x_t + w_t`, `y_t = d + H x_t + v_t` and returns `(y, x, F, H, Q, R, b, d)` with the true state path. The remaining three serve specialised checks: `dgp_ar2_peak` returns `(y, phi, freqs, spectrum)` with the analytic spectrum on a 256 grid, `dgp_lagged_pair` returns `(x, y, d, gain)` with known phase and gain, and `dgp_unit_root_pair` returns `(h0, h1, truth)` null/alternative pairs for thirteen test families (`:adf`, `:kpss`, `:trend`, `:break_level`, `:break_trend`, `:seasonal`, `:fourier`, `:explosive`, `:cointegrated_pair`, `:granger`, `:panel_ur`, `:nongaussian`, `:heteroskedastic_groups`).

See [ARIMA Models](@ref arima_page), [State-Space Models](@ref statespace_page), [Spectral Analysis](@ref spectral_page), and [Unit Root & Cointegration](@ref tests_unitroot_page) for the matching estimators and tests.

---

## Cointegration, ARDL, and Panel VAR

Six simulators cover error-correction, single-equation, and panel designs (Engle & Granger 1987; Johansen 1991; Pesaran, Shin & Smith 2001):

```math
\Delta Y_t = \alpha \beta' Y_{t-1} + \Gamma_1 \Delta Y_{t-1} + \ldots + \mu + \varepsilon_t, \quad \varepsilon_t \sim N(0, \Sigma)
```

where:

- ``Y_t`` is the ``n \times 1`` vector of I(1) variables
- ``\alpha`` (``n \times r``) and ``\beta`` (``n \times r``) are the adjustment and cointegrating matrices at rank ``r``
- ``\Gamma_i`` capture short-run dynamics and ``\mu`` the deterministic term

```@example simulation
v = dgp_vecm(MersenneTwister(41); T=300)
cr = dgp_cointreg(MersenneTwister(42); T=300)
(size(v.Y), v.beta[:], round.(cr.beta; digits=2))
```

`dgp_vecm` defaults to a 3-variable rank-1 design with distinct dynamics and returns `(Y, alpha, beta, Gamma, mu, Sigma, eps)`. `dgp_cointreg` builds `y = β′x + u` with random-walk regressors and endogenous errors (the FMOLS/DOLS bias-reduction target; `spurious = true` gives independent random walks) and returns `(y, X, beta, u)`. `dgp_panel_var` simulates a panel VAR(1) with random effects and returns `(Y, id, time, A1, mu, Sigma)` with `Y` stacked `NT×m`. `dgp_ardl` returns `(y, x, phi, beta, theta)` with the long-run multiplier `θ = (β₀+β₁)/(1−φ)` for bounds-test asserts; `dgp_nardl` adds the partial-sum decomposition and returns `(y, x, xp, xn, phi, theta_pos, theta_neg)`; `dgp_pmg` builds a panel ARDL with common long-run `θ` and heterogeneous short-run dynamics (`homogeneous = false` gives the Hausman-test power arm) and returns `(Y, X, id, theta, theta_i, phi_i)`.

See [Vector Error Correction Models](@ref vecm_page), [Cointegrating Regression](@ref cointreg_page), [ARDL & Bounds Testing](@ref ardl_page), and [Panel VAR](@ref pvar_page) for the matching estimators.

---

## Volatility

Four simulators cover univariate, latent-volatility, multivariate, and mixed-frequency variance designs (Engle 1982; Bollerslev 1986; Engle 2002):

```math
y_t = \mu + \sqrt{h_t} z_t, \quad h_t = \omega + \alpha (y_{t-1} - \mu)^2 + \beta h_{t-1}
```

where:

- ``y_t`` is the return and ``h_t`` the conditional variance at time ``t``
- ``\omega``, ``\alpha``, ``\beta`` are the GARCH intercept, ARCH, and GARCH parameters
- ``z_t`` is the standardised innovation, Gaussian or heavy-tailed

```@example simulation
sv = dgp_sv(MersenneTwister(51); T=500)
mg = dgp_mgarch(MersenneTwister(52); kind=:dcc, T=300)
md = dgp_midas(MersenneTwister(53); T_lf=100)
(length(sv.y), size(mg.H), length(md.y))  # (500, (300, 2, 2), 100)
```

`dgp_garch_family` simulates nine kinds (`:arch`, `:garch`, `:egarch`, `:gjr`, `:aparch`, `:igarch`, `:cgarch`, `:figarch`, `:fiegarch`) with `innov ∈ :gauss|:t|:laplace` and returns `(y, h, eps)` — returns, conditional variances, standardised shocks. `dgp_sv` simulates stochastic volatility `y_t = exp(h_t/2) ε_t` with AR(1) log-variance, leverage, and optional t shocks, and returns `(y, h)`. `dgp_mgarch` simulates `:ccc`, `:dcc`, or `:bekk` designs and returns `(Y, H, R)` with the `T×n×n` true covariance path. `dgp_midas` aggregates a high-frequency AR(1) into low-frequency `y` with normalised `:expalmon`, `:beta2`, or `:almon` weights and returns `(y, x_hf, w_true, beta)`.

!!! warning "EGARCH intercept convention"
    For `kind = :egarch`, `omega` is a log-variance intercept (often negative) and `alpha + beta` routinely exceeds 1. Do not impose the stationary-kind restriction `α + β < 1` on EGARCH draws.

See [Volatility Models](@ref volatility_page) for the matching estimators and [MIDAS Regression](@ref midas_page) for mixed-frequency estimation.

---

## Factor Models and Mixed Frequencies

Two simulators cover exact factor recovery and ragged-edge nowcasting designs (Stock & Watson 2002; Bai & Ng 2002; Mariano & Murasawa 2003):

```math
F_t = A_1 F_{t-1} + \ldots + A_p F_{t-p} + \eta_t, \quad X = F \Lambda' + e
```

where:

- ``F_t`` is the ``r \times 1`` vector of latent VAR(p) factors — never iid, so tests have factor dynamics to recover
- ``\Lambda`` is the ``N \times r`` loading matrix (optionally block-restricted)
- ``e`` is idiosyncratic noise, iid or AR(1), scaled to a `signal_share ≈ 0.7` common component by default

```@example simulation
mf = dgp_mixed_frequency_panel(MersenneTwister(62); T=120)
agg = mm_aggregate(mf.F, mf.Lambda_Q)
(size(mf.Y), size(agg))  # ((120, 12), (120, 2))
```

`dgp_dynamic_factors` returns `(X, F, Lambda, A, Sigma_F, idio_var, eps, r, p)` with the true factors, loadings, and standardised factor innovations. `dgp_mixed_frequency_panel` builds monthly series plus quarterly Mariano–Murasawa aggregates observed every third month (`NaN` elsewhere, with an optional ragged edge) and returns `(Y, is_quarterly, F, Lambda_M, Lambda_Q, A, agg_weights, withheld)`. The example cross-checks the quarterly signal against the analytic `mm_aggregate` helper.

See [Factor Models](@ref factor_page), [Factor-Augmented VAR](@ref favar_page), and [Nowcasting](@ref nowcast_page) for the matching estimators.

---

## Local Projections and Treatment Designs

Four simulators cover LP-IV, nonlinearity, selection, and serial-correlation designs (Jordà 2005; Dube et al. 2025; Newey & West 1987):

```math
y_{t+h} = \theta_h s_t + \gamma_h' w_t + u_{t+h}^{(h)}
```

where:

- ``y_{t+h}`` is the outcome at horizon ``h``
- ``s_t`` is the endogenous shock (instrumented in the LP-IV design)
- ``w_t`` holds lags and controls, and ``\theta_h`` traces the impulse response

```@example simulation
liv = dgp_lp_iv(MersenneTwister(71); T=300)
ps = dgp_propensity(MersenneTwister(72); n=500)
(size(liv.Y), size(liv.Z), round(ps.att; digits=2))
```

`dgp_lp_iv` builds the instrument `z`, the endogenous shock `s = π₁z + v`, and outcomes loading on `s` with impact response `theta`, and returns `(Y, Z, pi1, theta)` with `Y = [s y x2]`. `dgp_state_dependent_var` blends expansion and recession VARs through a logistic transition (returning `(Y, G, z, A_exp, A_rec, B0, irf_exp, irf_rec)` with each regime's true IRF at `H = 12`). `dgp_propensity` builds confounded selection `D ~ Bernoulli(logistic(Xβ))` with `Y = Y₀ + τD` and returns `(Y, D, X, tau, beta_ps, att, ps)`. `dgp_hac` builds AR(1)-error regressions with iid regressors and returns `(X, u, rho, lrv)` with the population long-run variance `1/(1−ρ)²`.

See [Local Projections](@ref lp_page) and [Event Study LP](@ref event_study_page) for LP estimation, [Difference-in-Differences](@ref did_page) for treatment designs, and [Linear Regression](@ref regression_page) for HAC covariance estimation.

---

## Cross-Section, Panel, and Staggered DiD

Three simulators cover fifteen cross-section kinds, linear and nonlinear panels, and staggered adoption (Hansen 1982; Pesaran, Shin & Smith 1999):

```math
y_{it} = \alpha_i + x_{it}' \beta + e_{it}, \quad \alpha_i = \bar{x}_i' \xi + u_i
```

where:

- ``y_{it}`` is the outcome for unit ``i`` at time ``t``
- ``\alpha_i`` is the unit effect with Mundlak correlated effects (``\xi`` scales the correlation; 0 gives random effects)
- ``x_{it}`` holds the regressors and ``\beta`` the slopes

```@example simulation
cs = dgp_cross_section(MersenneTwister(81); kind=:ols, n=500)
pn = dgp_panel(MersenneTwister(82); N=50, T=10)
(size(cs.y), size(pn.df))  # ((500,), (500, 5))
```

`dgp_cross_section` simulates `kind ∈ :ols|:hc|:cluster|:iv|:logit|:probit|:ordered|:mlogit|:poisson|:nb|:tobit|:truncreg|:heckman|:qreg|:rdd` and returns `(y, X, beta)` plus kind-specific truth (`:cluster` adds `clust` ids and `rho`; `:iv` adds `Z` instruments; `:ordered` adds `cutpoints`; `:nb` adds `dispersion`; `:heckman` adds `select_rho`; `:rdd` adds `cutoff`, `tau`, and the running variable `r`; discrete choice adds closed-form `ame`). `dgp_panel` builds `kind ∈ :linear|:logit|:probit` panels with AR(1) errors, common shocks, heteroskedasticity, and an optional lagged dependent variable (Nickell-bias design), and returns `(df, beta, sigma_u, sigma_e, alpha, mundlak, rho_ar, dynamic_rho)` with a long DataFrame. `dgp_staggered_did` assigns cohorts with heterogeneous effects `tau(g, e)` and returns `(df, att_by_event_time, att_by_cohort, overall_att, cohort_of)`; `violate_pt > 0` adds a pre-trend slope for parallel-trends power checks.

See [Linear Regression](@ref regression_page), [Binary Choice Models](@ref binary_choice_page), [Ordered & Multinomial Models](@ref ordered_multinomial_page), [Panel Regression](@ref panel_reg_page), and [Difference-in-Differences](@ref did_page) for the matching estimators.

---

## Regime Switching

One simulator covers four nonlinear time-series regimes (Hamilton 1989; Tong 1990; Luukkonen, Saikkonen & Teräsvirta 1988):

```math
y_t = \phi_{s_t} y_{t-1} + \sigma e_t, \quad s_t \in \{1, 2\}
```

where:

- ``y_t`` is the scalar observation at time ``t``
- ``s_t`` is the latent regime following a Markov chain (`:ms`), a threshold rule (`:setar`), or a smooth transition (`:lstar`, `:estr`)
- ``\phi_{s_t}`` is the regime-specific autoregressive coefficient

```@example simulation
rs = dgp_regime_switching(MersenneTwister(91); kind=:ms, T=300)
st = dgp_regime_switching(MersenneTwister(92); kind=:setar, T=300)
(length(rs.y), extrema(rs.s), length(st.y))  # (300, (1, 2), 300)
```

`kind = :ms` simulates a mean-switching MS-AR(1) and returns `(y, s, mu, phi, P)` with the true state path. `kind = :setar` thresholds on `y_{t-d}` and returns `(y, regime, phi_lo, phi_hi, c)`. `kind ∈ :lstar|:estr` blend `(φ_lo, c_lo)` and `(φ_hi, c_hi)` through a logistic or exponential transition and return `(y, G, phi_lo, phi_hi, gamma, c)` with the transition path.

See [Nonlinear Time Series](@ref nonlinear_page) for the matching estimators.

---

## GMM, Policy Bands, and DSGE Measurement

Three simulators cover moment-based estimation, policy-menu uncertainty, and observation error (Hansen 1982):

```math
y = X \beta + u, \quad E[Z' u] = 0
```

where:

- ``y`` is the ``n \times 1`` outcome vector and ``X`` the regressors (possibly endogenous)
- ``Z`` holds the instruments with first-stage strength `pi1`
- ``u`` is heteroskedastic, so two-step weighting matters

```@example simulation
gm = dgp_gmm(MersenneTwister(101); kind=:iv, n=500)
pce = dgp_pce_draws(MersenneTwister(102), [1.0, 2.0, 3.0, 4.0]; n_draws=100)
rng103 = MersenneTwister(103)
obs = dgp_dsge_observed(rng103, randn(rng103, 100, 2); H=[0.1, 0.1])
(size(gm.Z), size(pce.draws), size(obs.y_obs))  # ((500, 3), (100, 4), (100, 2))
```

`dgp_gmm` simulates `kind ∈ :ols|:iv` designs with heteroskedastic errors, `overid_k` instruments, and `invalid_k` of them correlated with the error (the J-test power arm), and returns `(y, X, Z, beta, pi1)`. `dgp_pce_draws` perturbs a point policy menu with Gaussian noise of known `sd` and AR(1) `corr` along the horizon and returns `(draws, point, sd)` for band width-scaling and coverage asserts. `dgp_dsge_observed` adds trends and measurement error `y_obs = y_clean + trends + sqrt(H)⋅η` to a clean simulation and returns `(y_obs, H)`, keeping the estimated measurement-error variances matched to the DGP.

!!! note "Technical Note"
    In `dgp_gmm`, the invalid instruments are the last `invalid_k` (the overidentifying ones). Contaminating the relevant first instrument instead lets the slope absorb the violation — the slope turns biased while the J-test stays silent — so that variant belongs in-test, not here.

See [Generalized & Simulated Method of Moments](@ref gmm_page) for GMM estimation and [DSGE Estimation](@ref dsge_estimation) for measurement-error designs.

---

## Analytic Truth Helpers

Four deterministic helpers compute the closed-form moments a recovery assertion compares against. They take no `rng`:

```math
S(\omega) = \frac{\sigma^2}{2\pi} \cdot \frac{|1 + \sum_j \theta_j e^{-i\omega j}|^2}{|1 - \sum_j \phi_j e^{-i\omega j}|^2}
```

where:

- ``S(\omega)`` is the ARMA spectral density at frequency ``\omega``
- ``\phi`` and ``\theta`` are the AR and MA coefficients
- ``\sigma`` scales the innovation variance

```@example simulation
w = collect(range(0, pi; length=64))
sp = arma_spectrum([0.7], Float64[], 1.0, w)
cs2 = dgp_cross_section(MersenneTwister(111); kind=:logit, n=500)
ame_err = maximum(abs.(logit_ame(cs2.X, cs2.beta) - cs2.ame))
(length(sp), round(ame_err; digits=12))  # analytic spectrum + exact AME match
```

`arma_spectrum(phi, theta, sigma, freqs)` evaluates the spectral density on any grid. `mm_aggregate(F, Lambda_Q; weights)` aggregates a monthly factor path into the quarterly Mariano–Murasawa signal. `logit_ame(X, beta)` and `probit_ame(X, beta)` return closed-form average marginal effects; the example asserts the helper reproduces the `:logit` design's planted `ame` exactly.

See [Spectral Analysis](@ref spectral_page) for spectrum estimation, [Nowcasting](@ref nowcast_page) for temporal aggregation, and [Binary Choice Models](@ref binary_choice_page) for marginal effects.

---

## Summary of Exported Simulators

| Name | Family (file) | Key truth fields |
|------|---------------|------------------|
| `dgp_var` | VAR (`dgp_var.jl`) | `Y, eps, A, Sigma, B0, c` |
| `lyapunov_gamma0` | VAR | stationary ``\Gamma_0`` matrix |
| `var_irf` | VAR | `(H+1)×n×n` structural moving average |
| `var_fevd` | VAR | `(H+1)×n×n` variance shares |
| `var_hd` | VAR | `T×n×n` shock contributions |
| `dgp_nongaussian_var` | SVAR non-Gaussian (`dgp_svar_nongauss.jl`) | `Y, eps, A, Sigma, B0, dist` |
| `dgp_heteroskedastic_var` | SVAR heteroskedastic | `Y, eps, scales, B0, Sigma_full, Lambda, path, kind` |
| `dgp_arima` | Univariate (`dgp_univariate.jl`) | `y, phi, theta, d, Phi, Theta, s, c, sigma` |
| `dgp_trend_cycle` | Univariate | `y, trend, cycle, phi` |
| `dgp_ar2_peak` | Univariate | `y, phi, freqs, spectrum` |
| `dgp_lagged_pair` | Univariate | `x, y, d, gain` |
| `dgp_state_space` | Univariate | `y, x, F, H, Q, R, b, d` |
| `dgp_unit_root_pair` | Univariate | `h0, h1, truth` |
| `dgp_vecm` | Cointegration (`dgp_cointegration.jl`) | `Y, alpha, beta, Gamma, mu, Sigma, eps` |
| `dgp_cointreg` | Cointegration | `y, X, beta, u` |
| `dgp_panel_var` | Cointegration | `Y, id, time, A1, mu, Sigma` |
| `dgp_ardl` | Cointegration | `y, x, phi, beta, theta` |
| `dgp_nardl` | Cointegration | `y, x, xp, xn, phi, theta_pos, theta_neg` |
| `dgp_pmg` | Cointegration | `Y, X, id, theta, theta_i, phi_i` |
| `dgp_garch_family` | Volatility (`dgp_volatility.jl`) | `y, h, eps` |
| `dgp_sv` | Volatility | `y, h` |
| `dgp_mgarch` | Volatility | `Y, H, R` |
| `dgp_midas` | Volatility | `y, x_hf, w_true, beta` |
| `dgp_dynamic_factors` | Factors (`dgp_factors.jl`) | `X, F, Lambda, A, Sigma_F, idio_var, eps, r, p` |
| `dgp_mixed_frequency_panel` | Factors | `Y, is_quarterly, F, Lambda_M, Lambda_Q, A, agg_weights, withheld` |
| `dgp_lp_iv` | LP (`dgp_lp.jl`) | `Y, Z, pi1, theta` |
| `dgp_state_dependent_var` | LP | `Y, G, z, A_exp, A_rec, B0, irf_exp, irf_rec` |
| `dgp_propensity` | LP | `Y, D, X, tau, beta_ps, att, ps` |
| `dgp_hac` | LP | `X, u, rho, lrv` |
| `dgp_cross_section` | Micro (`dgp_micro.jl`) | `y, X, beta` + kind-specific extras |
| `dgp_panel` | Micro | `df, beta, sigma_u, sigma_e, alpha, mundlak, rho_ar, dynamic_rho` |
| `dgp_staggered_did` | Micro | `df, att_by_event_time, att_by_cohort, overall_att, cohort_of` |
| `dgp_regime_switching` | Regime (`dgp_regime.jl`) | `y` + `s` / `regime` / `G` by kind |
| `dgp_gmm` | GMM (`dgp_gmm.jl`) | `y, X, Z, beta, pi1` |
| `dgp_pce_draws` | GMM | `draws, point, sd` |
| `dgp_dsge_observed` | GMM | `y_obs, H` |
| `arma_spectrum` | Truth (`dgp_truth.jl`) | spectral density vector |
| `mm_aggregate` | Truth | quarterly aggregate matrix |
| `logit_ame` | Truth | average marginal effects vector |
| `probit_ame` | Truth | average marginal effects vector |

---

## Complete Example

A five-replication Monte Carlo over the VAR design: simulate, estimate by OLS, identify recursively, and record the worst IRF recovery error per replication.

```@example simulation
mc = [begin
    s = dgp_var(MersenneTwister(1000 + r); T=200)
    e = estimate_var(s.Y, 1)
    maximum(abs.(compute_irf(e, Matrix{Float64}(I, 3, 3), 6) - var_irf(s.A, s.B0, 5)))
end for r in 1:5]
round.(mc; digits=3)  # five IRF recovery errors, all small
```

Each replication draws a fresh 200-observation sample from the same population design, fits a VAR(1), and compares the recursively identified IRF against the population `var_irf`. The errors cluster near zero and shrink with `T`, which is exactly the recovery behaviour a Monte Carlo study reports at scale.

---

## Common Pitfalls

1. **Setting `burn = 0` and keeping every draw.** The pre-sample history starts at zero, so the first observations carry an initialization transient. Keep the default burn-in unless the check needs the transient (as in the `var_hd` zero-history identity).
2. **Passing both `B0` and `Sigma` to `dgp_var`.** The simulator throws `ArgumentError`: pass exactly one. `Sigma` alone takes the lower Cholesky factor as the impact matrix.
3. **Comparing OLS slopes directly to `sim.A`.** `estimate_var` stores transposed coefficient blocks in `m.B` (intercept first); compare `m.B[2:4, :]'` against `sim.A[1]`, or compare IRFs as above.
4. **Reading `var_hd` levels as the sample.** `var_hd` returns demeaned shock contributions: they sum to `Y_t − μ`, not `Y_t`. Add the mean back before comparing to the simulated sample.
5. **Imposing stationarity on EGARCH parameters.** For `dgp_garch_family` with `kind = :egarch`, `omega` is a log-variance intercept and `alpha + beta` routinely exceeds 1 by design.
6. **Treating `:truncreg` draws as a balanced sample.** `dgp_cross_section` with `kind = :truncreg` returns only the selected rows, so `y` and `X` are shorter than `n`. The `:tobit` kind instead censors at `censor` and keeps all `n` rows.

---

## References

- Bai, J., & Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models. *Econometrica*, 70(1), 191--221. [DOI](https://doi.org/10.1111/1468-0262.00273)
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307--327. [DOI](https://doi.org/10.1016/0304-4076(86)90063-1)
- Dube, A., Girardi, D., Jorda, O., & Taylor, A. M. (2025). A Local Projections Approach to Difference-in-Differences. *Journal of Applied Econometrics*, 40(7), 741--758. [DOI](https://doi.org/10.1002/jae.70000)
- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987--1007. [DOI](https://doi.org/10.2307/1912773)
- Engle, R. F. (2002). Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models. *Journal of Business & Economic Statistics*, 20(3), 339--350. [DOI](https://doi.org/10.1198/073500102288618487)
- Engle, R. F., & Granger, C. W. J. (1987). Co-Integration and Error Correction: Representation, Estimation, and Testing. *Econometrica*, 55(2), 251--276. [DOI](https://doi.org/10.2307/1913236)
- Ghysels, E., Sinko, A., & Valkanov, R. (2007). MIDAS Regressions: Further Results and New Directions. *Econometric Reviews*, 26(1), 53--90. [DOI](https://doi.org/10.1080/07474930600972467)
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357--384. [DOI](https://doi.org/10.2307/1912559)
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton, NJ: Princeton University Press. ISBN 978-0-691-04289-3.
- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029--1054. [DOI](https://doi.org/10.2307/1912775)
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press. [DOI](https://doi.org/10.1017/CBO9781107049994)
- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models. *Econometrica*, 59(6), 1551--1580. [DOI](https://doi.org/10.2307/2938278)
- Jorda, O. (2005). Estimation and Inference of Impulse Responses by Local Projections. *American Economic Review*, 95(1), 161--182. [DOI](https://doi.org/10.1257/0002828053828518)
- Lanne, M., Lutkepohl, H., & Maciejowska, K. (2010). Structural Vector Autoregressions with Markov Switching. *Journal of Economic Dynamics and Control*, 34(2), 121--131. [DOI](https://doi.org/10.1016/j.jedc.2009.08.002)
- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8. [DOI](https://doi.org/10.1007/978-3-540-27752-1)
- Lutkepohl, H., & Netsunajev, A. (2017). Structural Vector Autoregressions with Smooth Transition in Variances. *Journal of Economic Dynamics and Control*, 84, 43--57. [DOI](https://doi.org/10.1016/j.jedc.2017.09.001)
- Luukkonen, R., Saikkonen, P., & Terasvirta, T. (1988). Testing linearity against smooth transition autoregressive models. *Biometrika*, 75(3), 491--499. [DOI](https://doi.org/10.1093/biomet/75.3.491)
- Mariano, R. S., & Murasawa, Y. (2003). A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series. *Journal of Applied Econometrics*, 18(4), 427--443. [DOI](https://doi.org/10.1002/jae.695)
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703--708. [DOI](https://doi.org/10.2307/1913610)
- Pesaran, M. H., Shin, Y., & Smith, R. P. (1999). Pooled Mean Group Estimation of Dynamic Heterogeneous Panels. *Journal of the American Statistical Association*, 94(446), 621--634. [DOI](https://doi.org/10.1080/01621459.1999.10474156)
- Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds Testing Approaches to the Analysis of Level Relationships. *Journal of Applied Econometrics*, 16(3), 289--326. [DOI](https://doi.org/10.1002/jae.616)
- Sims, C. A. (1980). Macroeconomics and Reality. *Econometrica*, 48(1), 1--48. [DOI](https://doi.org/10.2307/1912017)
- Stock, J. H., & Watson, M. W. (2002). Forecasting Using Principal Components from a Large Number of Predictors. *Journal of the American Statistical Association*, 97(460), 1167--1179. [DOI](https://doi.org/10.1198/016214502388618960)
- Tong, H. (1990). *Non-linear Time Series: A Dynamical System Approach.* Oxford University Press. ISBN 978-0-19-852300-6.

