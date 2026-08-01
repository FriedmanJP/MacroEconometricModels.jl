# [Local Projections](@id lp_page)

**MacroEconometricModels.jl** provides a complete toolkit for estimating impulse response functions via Local Projections (Jordà 2005), an alternative to VAR-based methods that offers greater robustness to dynamic misspecification. The package implements five LP variants, structural identification, LP-based FEVD, and direct multi-step forecasting.

- **Standard LP**: Horizon-by-horizon OLS regressions with Newey-West HAC standard errors
- **LP-IV**: Two-stage least squares with external instruments for endogenous shocks (Stock & Watson 2018)
- **Smooth LP**: B-spline basis functions with roughness penalty for noise reduction (Barnichon & Brownlees 2019)
- **State-Dependent LP**: Logistic smooth-transition models for regime-varying responses (Auerbach & Gorodnichenko 2012)
- **Propensity Score LP**: Inverse propensity weighting and doubly robust estimation for discrete treatments (Angrist, Jordà & Kuersteiner 2018)
- **Structural LP**: VAR-based identification (Cholesky, sign restrictions, ICA, etc.) with LP estimation of dynamic responses (Plagborg-Møller & Wolf 2021)
- **LP-FEVD**: R²-based forecast error variance decomposition with bias correction (Gorodnichenko & Lee 2019)
- **LP Forecasting**: Direct multi-step forecasts with analytical or bootstrap confidence intervals

All results integrate with `report()` for publication-quality output and `plot_result()` for interactive D3.js visualization.

Local Projections and VARs target the same population impulse responses (Plagborg-Møller & Wolf 2021); the [VAR](@ref var_page) page documents the system-based route and the [LP vs. VAR](@ref) section below sets out the trade-off. The variance decomposition on this page is the LP-specific estimator of Gorodnichenko & Lee (2019) — the VMA-based FEVD and the concept itself live on [Variance Decomposition](@ref ia_fevd_page).

```@setup lp
using MacroEconometricModels, Random, Statistics
Random.seed!(42)
fred = load_example(:fred_md)
Y = to_matrix(apply_tcode(fred[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
vnames = ["INDPRO", "CPIAUCSL", "FEDFUNDS"]
```

The examples use three FRED-MD series transformed by their published `tcode`: industrial production and the federal funds rate in first differences of logs and levels respectively, and `CPIAUCSL` under `tcode=6`, the **second log difference** — the change in inflation, not inflation itself.

## Quick Start

**Recipe 1: Standard LP with HAC standard errors**

```@example lp
# LP-IRF of a federal funds rate shock up to horizon 20
lp = estimate_lp(Y, 3, 20; lags=4, cov_type=:newey_west, varnames=vnames)
result = lp_irf(lp; conf_level=0.95)
report(result)
```

**Recipe 2: LP-IV with external instruments**

```@example lp
# A proxy external instrument: the FFR change observed with measurement error. An
# instrument that is a deterministic function of the controls has an infinite first
# stage and identifies nothing.
Z = reshape([0.0; diff(Y[:, 3])] .+ 1.5 * std(diff(Y[:, 3])) *
            randn(Random.MersenneTwister(7), size(Y, 1)), :, 1)
lpiv = estimate_lp_iv(Y, 3, Z, 20; lags=4, cov_type=:newey_west, varnames=vnames)
report(lpiv)
```

**Recipe 3: Smooth LP with B-splines**

```@example lp
smooth_lp = estimate_smooth_lp(Y, 3, 20; lambda=1.0, n_knots=4, lags=4, varnames=vnames)
report(smooth_lp)
```

**Recipe 4: Structural LP with Cholesky identification**

```@example lp
# Cholesky ordering: output -> prices -> monetary policy
slp = structural_lp(Y, 20; method=:cholesky, lags=4, varnames=vnames)
report(slp)
```

```julia
plot_result(slp)
```

```@raw html
<iframe src="../assets/plots/irf_structural_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 5: State-dependent LP (recession vs. expansion)**

```@example lp
# 7-month MA of IP growth as state variable, standardized
ip_growth = Y[:, 1]
state_var = [mean(ip_growth[max(1, t-6):t]) for t in 1:length(ip_growth)]
state_var = Float64.((state_var .- mean(state_var)) ./ std(state_var))

slm = estimate_state_lp(Y, 3, state_var, 20; gamma=1.5, threshold=0.0, lags=4,
                        varnames=vnames)
report(slm)
```

**Recipe 6: LP-FEVD with bias correction**

```@example lp
lfevd = lp_fevd(slp, 20; method=:r2, bias_correct=true, n_boot=50,
                rng=MersenneTwister(20260801))
report(lfevd)
```

```julia
plot_result(lfevd)
```

```@raw html
<iframe src="../assets/plots/fevd_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

---

## Standard Local Projections

**Local Projections** (Jordà 2005) estimate impulse responses by running a separate predictive regression at each forecast horizon. Unlike VARs, which derive IRFs from a single dynamic system, LPs directly estimate the response at each horizon ``h`` without imposing autoregressive restrictions.

For each horizon ``h = 0, 1, \ldots, H``, the LP regression is:

```math
y_{i,t+h} = \alpha_{i,h} + \beta_{i,h} \, x_t + \gamma_{i,h}' \, w_t + \varepsilon_{i,t+h}
```

where:
- ``y_{i,t+h}`` is the response variable ``i`` at time ``t+h``
- ``x_t`` is the shock variable at time ``t``
- ``w_t = (y_{t-1}', y_{t-2}', \ldots, y_{t-p}')' `` is the vector of lagged controls
- ``\beta_{i,h}`` is the impulse response of variable ``i`` to shock ``x`` at horizon ``h``
- ``\varepsilon_{i,t+h}`` is the regression error

OLS at each horizon yields:

```math
\hat{\beta}_h = (X'X)^{-1} X' Y_h
```

where:
- ``X`` is the ``T_{\text{eff}} \times k`` regressor matrix (intercept, shock, controls)
- ``Y_h`` is the ``T_{\text{eff}} \times 1`` response vector at horizon ``h``
- ``k = 2 + np`` (intercept + shock + ``p`` lags of ``n`` variables)

### HAC Standard Errors

LP residuals ``\varepsilon_{t+h}`` are serially correlated --- at least MA(``h-1``) under the null --- because overlapping forecast horizons create mechanical dependence. Newey-West HAC standard errors are therefore essential:

```math
\hat{V}_{\text{NW}} = (X'X)^{-1} \, \hat{S} \, (X'X)^{-1}
```

where:
- ``\hat{V}_{\text{NW}}`` is the HAC variance-covariance matrix of ``\hat{\beta}_h``
- ``\hat{S} = \hat{\Gamma}_0 + \sum_{j=1}^{m} w_j (\hat{\Gamma}_j + \hat{\Gamma}_j')`` is the long-run covariance estimator
- ``w_j`` are Bartlett kernel weights and ``m`` is the bandwidth

!!! note "Automatic Bandwidth Selection"
    When `bandwidth=0` (the default), the effective bandwidth at each horizon ``h`` is `max(m̂_NW, h+1)` where `m̂_NW` is the Newey-West (1994) data-driven selection. This ensures the bandwidth always accounts for the MA(``h-1``) serial correlation structure induced by the overlapping projection.

```@example lp
# Estimate LP-IRF of a federal funds rate shock up to horizon 20
lp_model = estimate_lp(Y, 3, 20;       # shock_var=3 (FEDFUNDS)
    lags = 4,                           # Control lags
    cov_type = :newey_west,             # HAC standard errors
    bandwidth = 0,                      # 0 = automatic bandwidth
    varnames = vnames
)

# Extract IRF with confidence intervals
irf_result = lp_irf(lp_model; conf_level=0.95)
report(irf_result)
```

```julia
plot_result(irf_result)
```

```@raw html
<iframe src="../assets/plots/irf_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `irf_result.values` matrix has dimension ``(H+1) \times n_{\text{resp}}``, where each row gives the response at a particular horizon. At ``h = 0`` the shock variable moves by ``1.000`` by construction, and the same-period response of industrial production is ``0.0201`` — significant at 5%, which is the recursive-ordering artefact of projecting on the *reduced-form* federal funds rate rather than on an identified monetary policy shock (the [Structural Local Projections](@ref) section fixes this). The funds rate response decays from ``0.92`` at ``h=1`` to ``0.26`` by ``h=8`` and is indistinguishable from zero thereafter. Standard errors in `irf_result.se` widen from ``0.0081`` at ``h=0`` to roughly ``0.006``–``0.010`` across horizons for `INDPRO` and much more sharply for the funds rate itself, because longer-horizon LP residuals are more strongly serially correlated and the effective sample shrinks by one observation per horizon — from 56 usable observations at ``h=0`` to 36 at ``h=20``.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `Int` | `4` | Number of control lags ``p`` |
| `response_vars` | `Vector{Int}` | all columns | Indices of response variables |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator (`:newey_west`, `:white`, `:ols`) |
| `bandwidth` | `Int` | `0` | HAC bandwidth (`0` = automatic) |
| `conf_level` | `Real` | `0.95` | Confidence level carried into the IRF |
| `varnames` | `Vector{String}` | `["y1", …]` | Variable labels used in `report` and plots |

### `lp_irf` Confidence Intervals

The bootstrap is **fixed-design**: at each horizon the regressor matrix is held fixed and only the errors are resampled (``y^* = X_h\hat{\beta}_h + u^*``, refit by OLS). That is the right form for LP, whose regressors are predetermined but whose errors are MA(``h``)-correlated by construction. `:wild` is the default here --- unlike the VAR, where `:iid` is --- because LP residuals are serially correlated *and* frequently heteroskedastic. Only the bands change: the reported responses and standard errors remain the analytical ones, so switching `ci_type` never moves the point estimate.

```@example lp
# Bootstrap bands instead of HAC ones: the design is held fixed and only the errors
# are resampled, which is what makes it valid for LP's MA(h)-correlated residuals.
boot_result = lp_irf(lp_model; ci_type=:bootstrap, bootstrap=:wild, reps=500, seed=1)
(analytical_lower = round.(irf_result.ci_lower[1:4, 1], digits=4),
 bootstrap_lower  = round.(boot_result.ci_lower[1:4, 1], digits=4),
 values_identical = irf_result.values == boot_result.values)
```

The two lower bands for `INDPRO` track each other closely over the first four horizons — ``0.0042`` against ``0.0051`` at ``h=0``, ``-0.0363`` against ``-0.0342`` at ``h=1`` — so at this sample size the normal approximation underlying the HAC bands is adequate. The wild bootstrap is slightly tighter here, which is typical when the residuals are less heteroskedastic than the analytical formula's implicit worst case. The point estimates are bit-for-bit identical, confirming that `ci_type` changes only the interval construction.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `conf_level` | `Real` | `0.95` | Confidence level for CIs |
| `ci_type` | `Symbol` | `:analytical` | `:analytical` (HAC) or `:bootstrap` (percentile bands) |
| `bootstrap` | `Symbol` | `:wild` | Resampling scheme: `:wild`, `:block`, `:iid` |
| `block_length` | `Int` | `0` | Moving-block length; `0` selects ``\lceil T^{1/3} \rceil`` |
| `wild_dist` | `Symbol` | `:rademacher` | `:rademacher` or `:mammen` |
| `reps` | `Int` | `500` | Bootstrap replications |
| `seed` | `Union{Integer,Nothing}` | `nothing` | Fixes the bands for reproducibility |

### Return Values (`LPModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `shock_var` | `Int` | Index of the shock variable |
| `response_vars` | `Vector{Int}` | Indices of response variables |
| `horizon` | `Int` | Maximum horizon ``H`` |
| `lags` | `Int` | Number of control lags |
| `B` | `Vector{Matrix{T}}` | Coefficient matrices (one per horizon) |
| `residuals` | `Vector{Matrix{T}}` | Residuals at each horizon |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices (HAC) |
| `T_eff` | `Vector{Int}` | Effective sample size at each horizon |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |
| `varnames` | `Vector{String}` | Variable labels |

### Return Values (`LPImpulseResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `values` | `Matrix{T}` | ``(H+1) \times n_{\text{resp}}`` IRF point estimates |
| `ci_lower` | `Matrix{T}` | Lower confidence bounds |
| `ci_upper` | `Matrix{T}` | Upper confidence bounds |
| `se` | `Matrix{T}` | Standard errors at each horizon |
| `horizon` | `Int` | Maximum horizon |
| `response_vars` | `Vector{String}` | Response variable names |
| `shock_var` | `String` | Shock variable name |
| `cov_type` | `Symbol` | Covariance estimator type |
| `conf_level` | `T` | Confidence level |

Two convenience wrappers extend the single-shock estimator: [`estimate_lp_multi`](@ref) fits one `LPModel` per shock variable in a supplied index vector, and [`estimate_lp_cholesky`](@ref) orthogonalizes the reduced-form residuals via a Cholesky factorization before projecting, returning one recursively-identified `LPModel` per structural shock.

---

## LP with Instrumental Variables

When the shock variable ``x_t`` is endogenous or measured with error, external instruments provide identification. Stock & Watson (2018) develop the **LP-IV** methodology using two-stage least squares at each horizon.

**First stage** --- regress the endogenous shock on instruments and controls:

```math
x_t = \pi_0 + \pi_1' z_t + \pi_2' w_t + v_t
```

**Second stage** --- use fitted values in the LP regression:

```math
y_{i,t+h} = \alpha_{i,h} + \beta_{i,h} \, \hat{x}_t + \gamma_{i,h}' \, w_t + \varepsilon_{i,t+h}
```

where:
- ``z_t`` is the vector of external instruments
- ``\hat{x}_t`` is the first-stage fitted value
- ``\beta_{i,h}`` is the instrumented impulse response at horizon ``h``

### Instrument Relevance

The first-stage F-statistic tests whether instruments predict the shock:

```math
F = \frac{\hat{\pi}_1' \, \hat{V}_{\pi}^{-1} \, \hat{\pi}_1}{q}
```

where:
- ``\hat{\pi}_1`` is the vector of first-stage coefficients on the instruments
- ``\hat{V}_{\pi}`` is the HAC variance-covariance of ``\hat{\pi}_1``
- ``q`` is the number of instruments

A rule of thumb requires ``F > 10`` for strong instruments (Stock & Yogo 2005).

!!! note "HAC-Robust F-Statistic"
    The first-stage F-statistic uses Newey-West HAC standard errors, consistent with the second-stage inference. This accounts for the MA(``h-1``) serial correlation in LP residuals. The HAC bandwidth follows the same automatic selection as the second stage.

An external instrument is a **noisy measure** of the structural shock — a narrative series or a high-frequency surprise — not a deterministic function of the model's own controls. FRED-MD ships no such series, so the example builds a proxy: the federal-funds-rate change plus measurement noise.

```@example lp
# LP-IV with the proxy instrument built in the Quick Start
lpiv_model = estimate_lp_iv(Y, 3, Z, 20;    # shock_var=3 (FEDFUNDS)
    lags = 4,
    cov_type = :newey_west,
    varnames = vnames
)

# Check first-stage strength
weak_test = weak_instrument_test(lpiv_model; threshold=10.0)
(min_F = round(weak_test.min_F, digits=2), weak_horizons = weak_test.weak_horizons,
 passes = weak_test.passes_threshold)
```

The `weak_test.min_F` reports the minimum first-stage F-statistic across all horizons: ``60.82`` here, six times the Stock & Yogo (2005) rule-of-thumb threshold of 10, with `weak_horizons` empty so the instrument is strong at every one of the 21 horizons. Because the proxy is the funds-rate change plus mean-zero noise, its relevance does not decay with the horizon — the first-stage F drifts *up* from ``67.30`` at ``h=0`` to a maximum of ``122.51``. With a genuine narrative or high-frequency instrument the opposite pattern is common, and the minimum across horizons, not the ``h=0`` value, is the number to check.

```@example lp
report(lpiv_model)
```

The instrumented responses differ noticeably from the reduced-form LP above: the impact response of `INDPRO` falls from ``0.0201`` to ``0.0130`` and loses significance, and the funds-rate path turns negative at long horizons (``-0.76`` at ``h=12``) where the OLS projection stayed positive. Instrumenting removes the component of the funds-rate change that is a systematic response to output and prices, so what remains is closer to an exogenous policy movement. The wider bands are the price: 2SLS discards the variation the instrument cannot explain, so LP-IV standard errors always exceed their OLS counterparts at the same horizon.

### Weak-Instrument-Robust Inference

External macro instruments — narrative series, high-frequency surprises — are routinely weak, and the horizon-wise 2SLS bands above are then unreliable at every horizon. Two tools address this.

**The Montiel Olea & Pflueger (2013) effective F** is the correct relevance diagnostic when errors are heteroskedastic or serially correlated, which macro data always are. It replaces the homoskedastic scale in the classical first-stage F with a HAR one:

```math
F_{\text{eff}} = \frac{\tilde{x}' P_{\tilde{Z}} \tilde{x}}
                        {\operatorname{tr}\!\big(\hat{V}_{\hat{\pi}}\,\tilde{Z}'\tilde{Z}\big)}
```

where:
- ``\tilde{Z} = M_W Z`` and ``\tilde{x} = M_W x`` are the instruments and the shock with the LP controls partialled out
- ``\hat{V}_{\hat{\pi}}`` is the Newey-West covariance of the first-stage coefficients
- ``q`` is the number of excluded instruments

Substituting the homoskedastic covariance ``\hat\sigma_v^2(\tilde{Z}'\tilde{Z})^{-1}`` makes the denominator exactly ``q\hat\sigma_v^2``, so ``F_{\text{eff}}`` collapses to the classical first-stage F — the reduction Montiel Olea and Pflueger establish.

```@example lp
mop = montiel_olea_pflueger_f(lpiv_model)
report(mop)
```

The effective F of ``67.30`` clears the 10%-worst-case-bias critical value of ``23.11`` comfortably, so the 2SLS bands reported above are trustworthy. The critical values are Montiel Olea and Pflueger's **simplified**, nuisance-parameter-free bounds by worst-case relative bias of 2SLS (`tau` = 0.05, 0.10, 0.20, 0.30 → 37.42, 23.11, 15.06, 12.04), as tabulated in Andrews, Stock & Sun (2019). They are conservative: the exact MOP critical values depend on the estimated covariance structure and are weakly smaller. Note how much higher the bar is than the familiar `F > 10` — an instrument with ``F = 15`` passes Stock–Yogo but fails the MOP 10% bias threshold.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `tau` | `Real` | `0.10` | Worst-case relative bias target (`0.05`, `0.10`, `0.20`, `0.30`) |
| `bandwidth` | `Int` | `0` | Fixed HAC lag length for the first-stage covariance; `0` uses the data-driven rule |

**Anderson-Rubin bands** give correct coverage at any instrument strength. At each horizon `h`, the AR test of ``H_0: \theta_h = \theta_0`` regresses ``y_{t+h} - \theta_0 x_t`` on the controls and instruments and tests the instruments' joint irrelevance — a restriction that holds under ``H_0`` regardless of how weak the first stage is. Inverting the test over ``\theta_0`` gives the band:

```@example lp
ar_band = lp_iv_ar_band(lpiv_model; responses=[1], n_grid=101)
report(ar_band)
```

The covariance uses **Newey-West with a lag length that scales with the horizon** — `max(data-driven, h+1)`, the rule `estimate_lp_iv` applies to its own standard errors, because the horizon-`h` LP residual is MA(`h`) by construction. The bandwidth actually used at each horizon and response is returned in `bandwidths`.

An AR set is not forced to be an interval: where the instrument is too weak to bound the response the set is **unbounded** and reported as `±∞`, and the corresponding Wald band there is over-confident. Here the instrument is strong at every horizon, so all 21 cells are bounded (`Unbounded cells 0 / 21`) and the AR and Wald bands nearly coincide — at ``h = 0`` the AR interval is ``[-0.0069, 0.0307]`` against a Wald interval of ``[-0.0052, 0.0313]``. That agreement is itself the diagnostic: with a weak instrument the AR set would widen or open up while the Wald band stayed narrow, and only the AR band would retain correct coverage.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `level` | `Real` | `0.95` | Nominal coverage |
| `n_grid` | `Int` | `401` | Grid points per horizon/response cell |
| `span` | `Real` | `20` | Half-width of the search range, in 2SLS standard errors |
| `bandwidth` | `Int` | `0` | Fixed HAC lag length; `0` uses `max(auto, h+1)` |
| `responses` | `Union{Nothing,Vector{Int}}` | `nothing` (all) | Subset of response positions to compute |

`LPIVARBand{T}` return value:

| Field | Type | Description |
|-------|------|-------------|
| `lower` / `upper` | `Matrix{T}` | ``(H+1) \times n_{resp}`` envelope; `±Inf` where unbounded, `NaN` where empty |
| `sets` | `Matrix{Vector{Tuple{T,T}}}` | Connected components of each set |
| `bounded` / `is_empty` | `Matrix{Bool}` | Set-shape flags |
| `wald_lower` / `wald_upper` | `Matrix{T}` | 2SLS Wald band, for comparison |
| `point` | `Matrix{T}` | LP-IV point estimates |
| `bandwidths` | `Matrix{Int}` | HAC lag length used at each horizon/response |
| `critical_value` | `T` | AR critical value at `level` |

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `lags` | `Int` | `4` | Number of control lags |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator |
| `bandwidth` | `Int` | `0` | HAC bandwidth (0 = automatic) |
| `response_vars` | `Vector{Int}` | all | Response variable indices |

### Return Values (`LPIVModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `shock_var` | `Int` | Index of the endogenous shock variable |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `instruments` | `Matrix{T}` | External instrument matrix |
| `horizon` | `Int` | Maximum horizon |
| `lags` | `Int` | Number of control lags |
| `B` | `Vector{Matrix{T}}` | Second-stage coefficient matrices |
| `residuals` | `Vector{Matrix{T}}` | Residuals at each horizon |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices |
| `first_stage_F` | `Vector{T}` | First-stage F-statistics by horizon |
| `first_stage_coef` | `Vector{Vector{T}}` | First-stage instrument coefficients |
| `T_eff` | `Vector{Int}` | Effective sample sizes |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |

---

## Smooth Local Projections

Standard LPs produce noisy impulse responses because each horizon is estimated independently. Barnichon & Brownlees (2019) propose **Smooth Local Projections** that parameterize the IRF as a smooth function of the horizon using B-spline basis functions, trading some bias for substantial variance reduction.

The impulse response is modeled as:

```math
\beta(h) = \sum_{j=1}^{J} \theta_j \, B_j(h)
```

where:
- ``\theta_j`` are spline coefficients
- ``B_j(h)`` are cubic B-spline basis functions evaluated at horizon ``h``
- ``J`` is the number of basis functions (determined by the knot count and degree)

### Penalized Estimation

Estimation proceeds in two steps — a "smooth-the-point-IRF" approximation to the one-step Barnichon & Brownlees penalized regression. First, standard LP produces ``\hat{\beta}_h`` at each horizon together with the **full cross-horizon covariance** ``\text{Cov}(\hat{\beta}_h, \hat{\beta}_{h'})`` (overlapping LP horizons share future windows and are strongly correlated). Second, a weighted penalized spline fit imposes smoothness:

```math
\hat{\theta} = \left( B' W B + \lambda R \right)^{-1} B' W \hat{\beta}
```

where:
- ``B`` is the ``(H+1) \times J`` basis matrix
- ``W = \text{diag}(1/\text{Var}(\hat{\beta}_h))`` is the precision-weight matrix
- ``R`` is the ``J \times J`` roughness penalty matrix with ``R_{ij} = \int B_i''(x) \, B_j''(x) \, dx``
- ``\lambda \geq 0`` is the smoothing parameter (``\lambda = 0`` gives unpenalized fit)

Reported confidence bands propagate the full cross-horizon covariance of ``\hat{\beta}`` through the spline map, ``\text{Var}(\hat{\theta}) = (B'WB+\lambda R)^{-1} B'W\,\text{Cov}(\hat{\beta})\,WB(B'WB+\lambda R)^{-1}``, rather than a per-horizon diagonal — otherwise the strong correlation between overlapping horizons would make the bands systematically too narrow.

The smoothing parameter ``\lambda`` controls the bias-variance trade-off. Larger values impose more smoothness, shrinking the IRF toward a low-frequency polynomial. The default ``\lambda = 0`` gives the *unpenalized* spline fit — smoothing then comes only from the finite basis, not from the roughness penalty — so a positive `lambda` must be supplied (or selected) to obtain a genuinely penalized estimator.

```@example lp
# Smooth LP with cubic splines
smooth_model = estimate_smooth_lp(Y, 3, 20;   # shock_var=3 (FEDFUNDS)
    degree = 3,           # Cubic splines
    n_knots = 4,          # Interior knots
    lambda = 1.0,         # Smoothing penalty (source default is 0.0 = unpenalized)
    lags = 4,
    varnames = vnames
)
report(smooth_model)
```

The smoothed impact response of `INDPRO` is ``0.0137`` against the standard LP's ``0.0201``: the spline pulls the noisy horizon-by-horizon estimates toward a smooth curve, damping the impact spike. The funds-rate path is visibly cleaner — a monotone decay from ``1.000`` to ``0.21`` by ``h=8`` rather than the standard LP's jagged ``0.92, 0.65, 0.95, 0.82`` sequence over the first four horizons — and its standard errors fall from around ``0.28`` to ``0.17`` in the middle horizons. The cubic basis with four interior knots forces the fitted IRF to zero at the right edge, which is why ``h=20`` reports exactly ``0.0000``; that endpoint behaviour is a property of the basis, not evidence that the effect has died out.

```@example lp
# Automatic lambda selection via cross-validation
optimal_lambda = cross_validate_lambda(Y, 3, 20;
    lambda_grid = 10.0 .^ (-4:0.5:2),
    k_folds = 5
)

# Compare smooth vs standard LP: the ratio of mean squared standard errors
comparison = compare_smooth_lp(Y, 3, 20; lambda=optimal_lambda)
(lambda = optimal_lambda,
 variance_reduction = round(comparison.variance_reduction, digits=4))
```

`comparison.variance_reduction` is the **ratio** ``\text{mean}(\text{se}^2_{\text{smooth}}) / \text{mean}(\text{se}^2_{\text{standard}})``, so values below 1 mean the smooth IRF is more precise; comparing it to 1, not to 0, is the correct reading. Here it is ``0.628``, a 37% reduction in average sampling variance — the favourable trade-off that motivates smooth LP in moderate samples where standard LP bands are wide. Five-fold cross-validation selects ``\lambda = 10^{-4}``, the smallest value on the grid, so on this 60-observation sample nearly all of the variance reduction comes from the spline basis itself rather than from the roughness penalty.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `degree` | `Int` | `3` | B-spline degree (3 = cubic) |
| `n_knots` | `Int` | `4` | Number of interior knots |
| `lambda` | `Real` | `0.0` | Roughness penalty; `0` gives the unpenalized spline fit |
| `lags` | `Int` | `4` | Number of control lags |
| `response_vars` | `Vector{Int}` | all columns | Indices of response variables |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator |
| `bandwidth` | `Int` | `0` | HAC bandwidth (`0` = automatic) |

`cross_validate_lambda` searches `lambda_grid` (default ``10^{-4}, 10^{-3.5}, \ldots, 10^{2}``) with `k_folds=5`; `compare_smooth_lp` takes `lambda` (default `1.0`) and forwards the rest to both estimators.

### Return Values (`SmoothLPModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `shock_var` | `Int` | Shock variable index |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `horizon` | `Int` | Maximum horizon |
| `lags` | `Int` | Number of control lags |
| `spline_basis` | `BSplineBasis{T}` | B-spline basis (knots, degree, basis matrix) |
| `theta` | `Matrix{T}` | Spline coefficients |
| `vcov_theta` | `Matrix{T}` | Variance-covariance of spline coefficients |
| `lambda` | `T` | Smoothing penalty parameter |
| `irf_values` | `Matrix{T}` | Smoothed IRF point estimates |
| `irf_se` | `Matrix{T}` | Standard errors of smoothed IRF |
| `residuals` | `Matrix{T}` | Regression residuals stacked over horizons |
| `T_eff` | `Int` | Observations pooled across all horizons, ``\sum_h (T - h - p)`` — not a per-horizon count |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |

---

## State-Dependent Local Projections

Economic responses may differ across states of the economy. Auerbach & Gorodnichenko (2012, 2013) develop **state-dependent LPs** using smooth transition functions to estimate regime-varying impulse responses --- for example, whether fiscal multipliers differ between recessions and expansions (Ramey & Zubairy 2018).

The state-dependent model is:

```math
y_{t+h} = F(z_t) \left[ \alpha_E + \beta_E \, x_t + \gamma_E' \, w_t \right] + (1 - F(z_t)) \left[ \alpha_R + \beta_R \, x_t + \gamma_R' \, w_t \right] + \varepsilon_{t+h}
```

where:
- ``F(z_t)`` is the logistic smooth transition function
- ``z_t`` is the state variable (e.g., moving average of GDP growth)
- ``\beta_E`` is the expansion regime impulse response (``F \to 1``)
- ``\beta_R`` is the recession regime impulse response (``F \to 0``)

### Logistic Transition Function

```math
F(z_t) = \frac{1}{1 + \exp(-\gamma(z_t - c))}
```

where:
- ``\gamma > 0`` controls the transition speed (higher = sharper regime switching)
- ``c`` is the threshold parameter (often 0 for standardized ``z_t``)

The function satisfies ``F(z) \to 0`` as ``z \to -\infty`` (deep recession), ``F(z) \to 1`` as ``z \to +\infty`` (strong expansion), and ``F(c) = 0.5`` (neutral state).

### Regime Difference Test

Testing whether responses differ across regimes uses a Wald-type test at each horizon:

```math
t = \frac{\hat{\beta}_E - \hat{\beta}_R}{\sqrt{\text{Var}(\hat{\beta}_E) + \text{Var}(\hat{\beta}_R) - 2\text{Cov}(\hat{\beta}_E, \hat{\beta}_R)}}
```

where:
- ``\hat{\beta}_E, \hat{\beta}_R`` are the regime-specific impulse responses
- The variance-covariance terms use HAC standard errors

!!! note "Optimization of Transition Parameters"
    When `gamma=:estimate` and `threshold=:estimate` (both defaults), the transition parameters ``(\gamma, c)`` are jointly optimized using Nelder-Mead over the nonlinear least squares objective. The threshold ``c`` is box-constrained within the data's interquartile range, and ``\gamma > 0`` is enforced.

!!! warning "Each regime needs its own sample"
    The model fits ``2(2 + np)`` coefficients per horizon, so a short sample with a lopsided transition function leaves one regime nearly unidentified and the HAC covariance near-singular. Check the reported `% in expansion`: values far from 50% on a small sample are the warning sign. Calibrating ``\gamma`` and setting ``c = 0`` on a standardized state variable, as Auerbach & Gorodnichenko (2012) and Ramey & Zubairy (2018) do, keeps both regimes populated.

```@example lp
# Construct state variable: 7-month MA of industrial production growth
ip_growth = Y[:, 1]
state_var = [mean(ip_growth[max(1, t-6):t]) for t in 1:length(ip_growth)]
state_var = Float64.((state_var .- mean(state_var)) ./ std(state_var))

# Estimate state-dependent LP: FFR shock with IP growth as state.
# gamma and threshold are calibrated rather than estimated — with 60 observations
# the optimizer drives the transition into a corner and starves one regime.
state_model = estimate_state_lp(Y, 3, state_var, 20;
    gamma = 1.5,
    threshold = 0.0,
    lags = 4,
    varnames = vnames
)
report(state_model)
```

The transition function puts 47.3% of the sample in the expansion regime, so both branches are estimated on comparable numbers of effective observations. The funds-rate shock propagates very differently across regimes: in recessions the rate itself stays elevated (``1.01`` at ``h=1``, ``2.00`` at ``h=4``, still ``0.67`` at ``h=20``), while in expansions the path reverses sign by ``h=4`` and ends at ``-1.02``. The output responses are correspondingly asymmetric — `INDPRO` rises ``0.050`` on impact in recessions against ``0.007`` in expansions — which is the qualitative pattern Auerbach & Gorodnichenko report for fiscal shocks and Ramey & Zubairy revisit critically.

```@example lp
# Extract regime-specific IRFs and test H0: β_E = β_R at each horizon
irf_expansion = state_irf(state_model; regime=:expansion)
irf_recession = state_irf(state_model; regime=:recession)
diff_test = test_regime_difference(state_model)

(t_stats_h0 = round.(diff_test.t_stats[1, :], digits=3),
 p_values_h0 = round.(diff_test.p_values[1, :], digits=3),
 joint = (avg_t = round(diff_test.joint_test.avg_t_stat, digits=3),
          p = round(diff_test.joint_test.p_value, digits=4)))
```

`test_regime_difference` computes a Wald-type ``t`` on ``\hat\beta_E - \hat\beta_R`` at each horizon using the HAC covariance of the difference, plus a joint test averaging across horizons. At ``h = 0`` only `CPIAUCSL` separates the regimes (``t = 2.445``, ``p = 0.014``); `INDPRO` and the funds rate do not. The joint statistic of ``2.573`` with ``p = 0.010`` rejects regime equality overall, so the state dependence visible in the two IRF tables is not attributable to sampling noise alone. Horizon-by-horizon rejections should be read with care: 21 horizons times three responses is 63 tests, and no multiplicity correction is applied.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `gamma` | `Real` or `Symbol` | `:estimate` | Transition speed (`:estimate` for optimization) |
| `threshold` | `Real` or `Symbol` | `:estimate` | Threshold parameter (`:estimate` for optimization) |
| `lags` | `Int` | `4` | Number of control lags |
| `response_vars` | `Vector{Int}` | all columns | Indices of response variables |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator |
| `bandwidth` | `Int` | `0` | HAC bandwidth (`0` = automatic) |

```julia
plot_result(state_model)
```

```@raw html
<iframe src="../assets/plots/state_lp.html" width="100%" height="460" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Return Values (`StateLPModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `shock_var` | `Int` | Shock variable index |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `horizon` | `Int` | Maximum horizon |
| `lags` | `Int` | Number of control lags |
| `state` | `StateTransition{T}` | State transition function (``\gamma``, threshold, ``F(z_t)`` values) |
| `B_expansion` | `Vector{Matrix{T}}` | Expansion regime coefficients |
| `B_recession` | `Vector{Matrix{T}}` | Recession regime coefficients |
| `residuals` | `Vector{Matrix{T}}` | Residuals at each horizon |
| `vcov_expansion` | `Vector{Matrix{T}}` | Expansion regime variance-covariance |
| `vcov_recession` | `Vector{Matrix{T}}` | Recession regime variance-covariance |
| `vcov_diff` | `Vector{Matrix{T}}` | Variance-covariance of regime difference |
| `T_eff` | `Vector{Int}` | Effective sample sizes |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |

---

## Propensity Score Local Projections

When the shock is a discrete treatment (e.g., a policy intervention), selection bias may confound causal inference. Angrist, Jordà & Kuersteiner (2018) develop **LP with inverse propensity weighting (IPW)** to address treatment selection. Weighting by the estimated rather than the known propensity score is efficient (Hirano, Imbens & Ridder 2003). The package provides two estimators: IPW and doubly robust (AIPW). The underlying propensity-score fit is also available standalone via [`estimate_propensity_score`](@ref), which returns the estimated ``\hat{p}(X_t)`` from a logit or probit model of treatment on covariates.

### IPW Estimator

Let ``D_t \in \{0, 1\}`` be a binary treatment indicator. The propensity score is:

```math
p(X_t) = P(D_t = 1 \mid X_t) = \frac{1}{1 + \exp(-X_t' \beta)}
```

where:
- ``D_t`` is the treatment indicator
- ``X_t`` is the vector of covariates
- ``p(X_t)`` is the estimated probability of treatment

The IPW-LP estimator reweights observations via weighted least squares:

```math
y_{t+h} = \alpha_h + \beta_h \, D_t + \gamma_h' \, W_t + \varepsilon_{t+h}
```

where:
- ``W_t`` includes lagged outcomes and covariates
- Weights are ``w_t = 1/\hat{p}(X_t)`` for treated and ``w_t = 1/(1 - \hat{p}(X_t))`` for control observations
- ``\beta_h`` is the Average Treatment Effect (ATE) at horizon ``h``

### Doubly Robust Estimator

The **doubly robust (DR)** estimator combines IPW with separate outcome regressions for treated and control groups (Robins, Rotnitzky & Zhao 1994). It computes the ATE from the influence function:

```math
\hat{\text{ATE}}_h^{\text{DR}} = \frac{1}{n} \sum_{t=1}^{n} \hat{\psi}_t
```

where:
- ``\hat{\psi}_t`` is the doubly robust influence function combining IPW and outcome model predictions
- ``\hat{\mu}_1(X_t) = E[y_{t+h} \mid D_t = 1, X_t]`` and ``\hat{\mu}_0(X_t) = E[y_{t+h} \mid D_t = 0, X_t]`` are outcome regressions

The DR estimator is consistent if **either** the propensity score model **or** the outcome regression model is correctly specified, providing insurance against single-model misspecification.

| Feature | `estimate_propensity_lp` (IPW) | `doubly_robust_lp` (DR/AIPW) |
|---------|-------------------------------|------------------------------|
| **Method** | WLS with inverse propensity weights | Influence function combining IPW + outcome regression |
| **Consistency requires** | Correct propensity model | Correct propensity **or** outcome model |
| **Best when** | Propensity model well-specified | Uncertainty about either model |

!!! note "Recommendation"
    Use `doubly_robust_lp` as the default. It is never worse than IPW asymptotically, and can be substantially better when the propensity model is misspecified. Use `estimate_propensity_lp` when you have strong confidence in the propensity score specification or want direct WLS coefficients.

```@example lp
# Construct binary treatment: large absolute FFR changes (top quartile)
ffr_changes = abs.(diff(Y[:, 3]))
treatment = Bool.(ffr_changes .> quantile(ffr_changes, 0.75))
Y_trim = Y[2:end, :]
covariates = Y_trim[:, 1:2]

# IPW estimation
ipw_model = estimate_propensity_lp(Y_trim, treatment, covariates, 20;
    ps_method = :logit,
    trimming = (0.01, 0.99),
    lags = 4,
    varnames = vnames
)

# Doubly robust estimation
dr_model = doubly_robust_lp(Y_trim, treatment, covariates, 20;
    ps_method = :logit,
    trimming = (0.01, 0.99),
    lags = 4,
    varnames = vnames
)
report(dr_model)
```

The treatment splits 59 observations into 15 treated and 44 control periods, and the doubly robust ATE of a large funds-rate move on the funds rate itself is ``0.085`` on impact, decaying to near zero by ``h=4``. The effects on `INDPRO` and `CPIAUCSL` are an order of magnitude smaller and mostly insignificant, which is the expected result: a top-quartile *absolute* rate change mixes tightenings and easings, so the average treatment effect on output nets out.

```@example lp
# ATE paths from both estimators, for the funds-rate response
ate_irf = propensity_irf(ipw_model)
dr_irf  = propensity_irf(dr_model)
(ipw = round.(ate_irf.values[1:4, 3], digits=4),
 dr  = round.(dr_irf.values[1:4, 3], digits=4))
```

The two estimators separate most at impact — IPW gives ``0.0075`` where the doubly robust estimator gives ``0.0851`` — and converge from ``h=2`` onward. That gap is informative rather than alarming: the two agree asymptotically only when the propensity model is correctly specified, so a divergence signals that the outcome regression in the DR influence function is carrying weight. Following the recommendation above, the DR path is the one to report.

```@example lp
# Diagnostics: overlap and covariate balance
diagnostics = propensity_diagnostics(ipw_model)
(common_support = round.(diagnostics.overlap.common_support, digits=4),
 treated_in_support = round(diagnostics.overlap.treated_in_support, digits=4),
 control_in_support = round(diagnostics.overlap.control_in_support, digits=4))
```

Common support spans propensity scores in ``[0.165, 0.369]``: every treated period lies inside it, and 90.9% of control periods do. Narrow but non-degenerate overlap like this is the well-behaved case — no score approaches 0 or 1, so the inverse weights stay bounded and the `trimming=(0.01, 0.99)` cap never binds. The 9% of control observations outside the support contribute no comparable treated match and are the observations that drive any remaining bias.

### Keyword Arguments

Both `estimate_propensity_lp` and `doubly_robust_lp` take the same keywords.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `ps_method` | `Symbol` | `:logit` | Propensity score model (`:logit`, `:probit`) |
| `trimming` | `Tuple{T,T}` | `(0.01, 0.99)` | Propensity score trimming bounds |
| `lags` | `Int` | `4` | Number of control lags |
| `response_vars` | `Vector{Int}` | all columns | Indices of response variables |
| `cov_type` | `Symbol` | `:newey_west` | Covariance estimator |
| `bandwidth` | `Int` | `0` | HAC bandwidth (`0` = automatic) |

### Return Values (`PropensityLPModel`)

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data matrix |
| `treatment` | `Vector{Bool}` | Binary treatment indicator |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `covariates` | `Matrix{T}` | Selection-relevant covariates |
| `horizon` | `Int` | Maximum horizon |
| `propensity_scores` | `Vector{T}` | Estimated propensity scores ``\hat{p}(X_t)`` |
| `ipw_weights` | `Vector{T}` | Inverse propensity weights |
| `B` | `Vector{Matrix{T}}` | Weighted regression coefficients |
| `residuals` | `Vector{Matrix{T}}` | Weighted residuals |
| `vcov` | `Vector{Matrix{T}}` | Variance-covariance matrices |
| `ate` | `Matrix{T}` | Average treatment effect estimates |
| `ate_se` | `Matrix{T}` | Standard errors of ATE |
| `config` | `PropensityScoreConfig{T}` | Configuration (method, trimming, normalize) |
| `T_eff` | `Vector{Int}` | Effective sample sizes |
| `cov_estimator` | `AbstractCovarianceEstimator` | Covariance estimator used |

---

## Structural Local Projections

**Structural Local Projections** combine VAR-based identification with LP estimation of dynamic responses. Plagborg-Møller & Wolf (2021) show that under correct specification, LP and VAR estimate the same impulse responses. Structural LP leverages this equivalence by using the VAR only for shock identification (computing the rotation matrix ``Q``), then estimating dynamics via LP regressions --- gaining LP's robustness while retaining SVAR's structural interpretability.

The procedure proceeds in four steps:

1. **Estimate VAR(p)**: Fit a VAR on ``Y`` to obtain the residual covariance ``\hat{\Sigma}`` and reduced-form residuals ``\hat{u}_t``
2. **Identify structural shocks**: Compute the rotation matrix ``Q`` via the chosen identification method
3. **Recover structural shocks**: Compute ``\hat{\varepsilon}_t = Q' L^{-1} \hat{u}_t`` where ``L = \text{chol}(\hat{\Sigma})``
4. **Run LP regressions**: For each structural shock ``j``, estimate:

```math
y_{i,t+h} = \alpha_{i,h}^{(j)} + \beta_{i,h}^{(j)} \, \hat{\varepsilon}_{j,t} + \gamma_{i,h}^{(j)\prime} \, w_t + u_{i,t+h}^{(j)}
```

where:
- ``\hat{\varepsilon}_{j,t}`` is the identified structural shock ``j``
- ``\beta_{i,h}^{(j)}`` is the structural impulse response of variable ``i`` to shock ``j`` at horizon ``h``
- ``w_t`` contains lagged values of ``Y`` as controls

The 3D IRF array stores ``\Theta[h, i, j] = \hat{\beta}_{i,h}^{(j)}`` for ``h = 1, \ldots, H``.

### Identification Methods

| Method | Keyword | Description |
|--------|---------|-------------|
| Cholesky | `:cholesky` | Recursive ordering (lower triangular ``B_0``) |
| Sign restrictions | `:sign` | Constrain signs of responses (Uhlig 2005) |
| Long-run | `:long_run` | Blanchard-Quah (1989) zero long-run effect |
| Narrative | `:narrative` | Historical events + sign restrictions (Antolín-Díaz & Rubio-Ramírez 2018) |
| FastICA | `:fastica` | Non-Gaussian ICA (Hyvärinen 1999) |
| JADE | `:jade` | Joint Approximate Diagonalization of Eigenmatrices |
| SOBI | `:sobi` | Second-Order Blind Identification |
| dCov | `:dcov` | Distance covariance independence criterion |
| HSIC | `:hsic` | Hilbert-Schmidt independence criterion |
| Student-t ML | `:student_t` | Maximum likelihood with Student-t errors |
| Mixture-normal ML | `:mixture_normal` | Gaussian mixture ML |
| PML | `:pml` | Pseudo maximum likelihood |

The statistical (non-Gaussian) schemes — FastICA, JADE, SOBI, dCov, HSIC, and the maximum-likelihood variants — are documented in full on the [Statistical Identification](@ref nongaussian_page) hub and its [Non-Gaussian Methods](@ref id_nongaussian_page) child.

For a moment-based alternative to the OLS projection, [`estimate_lp_gmm`](@ref) estimates the horizon-``h`` local projections by GMM (returning one `GMMModel` per horizon), which admits overidentifying instruments and efficient two-step weighting; see the [GMM & SMM](@ref gmm_page) page for the underlying estimator.

```@example lp
# Structural LP with Cholesky identification
slp = structural_lp(Y, 20; method=:cholesky, lags=4, varnames=vnames)
report(slp)
```

```julia
plot_result(slp)
```

```@raw html
<iframe src="../assets/plots/irf_structural_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `slp.irf.values` array has shape ``H \times n \times n``, where `values[h, i, j]` gives the response of variable ``i`` to structural shock ``j`` at horizon ``h``. Under Cholesky identification with ordering [INDPRO, CPIAUCSL, FEDFUNDS], the monetary policy shock (shock 3) affects all variables contemporaneously, but the federal funds rate does not respond to output or price shocks within the period. The identified policy shock raises the funds rate by ``0.077`` at ``h=1`` and leaves output essentially untouched (``-0.0006``, against the ``0.020`` the reduced-form projection reported) — removing the endogenous policy response is precisely what the identification buys. Standard errors in `slp.se` come from the HAC-corrected LP regressions and tend to be wider than VAR-based IRF bands, reflecting the efficiency cost of LP's robustness.

```@example lp
# With bootstrap confidence intervals
slp_ci = structural_lp(Y, 20; method=:cholesky, ci_type=:bootstrap, reps=50,
                       varnames=vnames, rng=MersenneTwister(1))

# With sign restrictions: positive supply shock raises output and lowers prices
check_fn(irf) = irf[1, 1, 1] > 0 && irf[1, 2, 1] < 0
slp_sign = structural_lp(Y, 20; method=:sign, check_func=check_fn, varnames=vnames,
                         rng=MersenneTwister(2))

(cholesky_bootstrap_reps = slp_ci.n_effective,
 cholesky_ci = slp_ci.irf.ci_type,
 sign_identified = slp_sign.method)
```

All 50 bootstrap draws were usable (`n_failed` is zero), so the percentile bands on `slp_ci` rest on the full requested sample of rotations; a shortfall here would mean draws were discarded by the recoverable-error catch and the bands are thinner than requested. The sign-restricted fit searches rotations until one satisfies `check_fn`, so its shock ordering and scale are not comparable to the Cholesky fit's — sign restrictions identify a *set*, and `slp_sign` reports one admissible member of it.

Once identified, a `StructuralLP` dispatches to the same downstream tools as an SVAR:

```@example lp
decomp = fevd(slp, 20)
hd = historical_decomposition(slp)
(fevd_type = typeof(decomp).name.name, hd_type = typeof(hd).name.name)
```

`fevd(slp, H)` routes to the LP-specific estimator documented in [LP-Based FEVD](@ref) below and returns an `LPFEVD`, not the VMA-based `FEVD` of [Variance Decomposition](@ref ia_fevd_page); `historical_decomposition` returns the ordinary `HistoricalDecomposition` computed from the identified shocks.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:cholesky` | Identification method (see table above) |
| `lags` | `Int` | `4` | Number of LP control lags |
| `var_lags` | `Union{Nothing,Int}` | `nothing` (uses `lags`) | Lag order of the identification VAR, if different |
| `cov_type` | `Symbol` | `:newey_west` | HAC estimator type |
| `conf_level` | `Real` | `0.95` | Confidence level for the reported bands |
| `ci_type` | `Symbol` | `:none` | CI method (`:none`, `:bootstrap`) |
| `reps` | `Int` | `200` | Bootstrap replications |
| `check_func` | `Function` | `nothing` | Sign restriction check function |
| `narrative_check` | `Function` | `nothing` | Narrative restriction check function |
| `max_draws` | `Int` | `1000` | Maximum rotation draws for set-identified methods |
| `varnames` | `Vector{String}` | `["y1", …]` | Variable labels |
| `shock_names` | `Union{Nothing,Vector{String}}` | `nothing` (uses `varnames`) | Structural shock labels |

### Return Values (`StructuralLP`)

| Field | Type | Description |
|-------|------|-------------|
| `irf` | `ImpulseResponse{T}` | 3D IRF result (``H \times n \times n``) with optional bootstrap CIs |
| `structural_shocks` | `Matrix{T}` | ``T_{\text{eff}} \times n`` recovered structural shocks |
| `var_model` | `VARModel{T}` | Underlying VAR model used for identification |
| `Q` | `Matrix{T}` | ``n \times n`` rotation/identification matrix |
| `method` | `Symbol` | Identification method used |
| `lags` | `Int` | Number of LP control lags |
| `cov_type` | `Symbol` | HAC estimator type |
| `se` | `Array{T,3}` | ``H \times n \times n`` standard errors |
| `lp_models` | `Vector{LPModel{T}}` | Individual LP model per shock, fitted on ``[\hat\varepsilon_j\ \ Y]`` |
| `n_requested` / `n_effective` / `n_failed` | `Int` | Bootstrap draws attempted, usable, and dropped; all zero unless `ci_type=:bootstrap` |

---

## LP Forecasting

LP-based forecasts use horizon-specific regression coefficients directly --- no VAR recursion required. For each horizon ``h = 1, \ldots, H``, the direct multi-step forecast is:

```math
\hat{y}_{T+h} = \hat{\alpha}_h + \hat{\beta}_h \cdot s_h + \hat{\Gamma}_h \, w_T
```

where:
- ``\hat{y}_{T+h}`` is the ``h``-step-ahead point forecast
- ``s_h`` is the assumed shock path value at horizon ``h``
- ``\hat{\Gamma}_h`` is the coefficient vector on controls ``w_T`` (last ``p`` observations)
- ``\hat{\alpha}_h`` is the horizon-specific intercept

This direct approach avoids compounding misspecification errors across horizons, unlike iterated VAR forecasts.

### Confidence Intervals

| Method | Description |
|--------|-------------|
| `:analytical` | HAC standard errors + normal quantiles: ``\hat{y}_{T+h} \pm z_{\alpha/2} \cdot \hat{\sigma}_h`` |
| `:bootstrap` | Residual resampling with percentile CIs |
| `:none` | Point forecasts only |

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `ci_method` | `Symbol` | `:analytical` | CI construction (`:analytical`, `:bootstrap`, `:none`) |
| `conf_level` | `Real` | `0.95` | Confidence level |
| `n_boot` | `Int` | `500` | Bootstrap replications when `ci_method=:bootstrap` |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Draw source for the bootstrap |

`shock_path` must have length ``H``, matching the LP model's horizon.

```@example lp
# Forecast with a unit shock path sustained over the forecast window
shock_path = ones(20)
fc = forecast(lp_model, shock_path; ci_method=:analytical, conf_level=0.95)
report(fc)
```

```julia
plot_result(fc)
```

```@raw html
<iframe src="../assets/plots/forecast_lp.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `fc.forecast` matrix has shape ``H \times n_{\text{resp}}``, where each row gives the point forecast at a given horizon. A sustained unit funds-rate shock produces a funds-rate path that starts at ``0.958`` and decays to roughly zero by ``h = 13``, while the output and price forecasts stay within ``\pm 0.02`` — the scale difference reflects the `tcode` transformations, since `FEDFUNDS` is in levels while the other two are differenced logs. Analytical CIs widen with the horizon because LP residual variance grows and the effective sample shrinks: the funds-rate standard error rises from ``0.125`` at ``h=1`` to ``0.306`` at ``h=2`` and stays in that range. Bootstrap CIs are more reliable at this sample size because they do not rely on the normal approximation.

```@example lp
# Structural LP forecast driven by the identified monetary policy shock
fc_struct = forecast(slp, 3, shock_path;  # shock_idx=3 (monetary policy)
                     ci_method=:bootstrap, n_boot=50, rng=MersenneTwister(3))
report(fc_struct)
```

!!! note "Response labels in structural LP forecasts"
    `structural_lp` fits each per-shock LP on the augmented matrix ``[\hat\varepsilon_j\ \ Y]``, so the returned `LPForecast` indexes its responses against that augmented layout: `Var 2`, `Var 3`, and `Var 4` are `INDPRO`, `CPIAUCSL`, and `FEDFUNDS` in the original order, and `Shock variable 1` is the identified shock, not data column 1.

The bootstrap bands are visibly asymmetric — at ``h=6`` the funds-rate interval is ``[0.154, 0.873]`` around a point forecast of ``0.345``, with far more room above than below — which is exactly the behaviour the percentile method is meant to capture and a normal approximation would miss. The identified-shock forecast is also much smaller in magnitude than the reduced-form one above (``0.050`` against ``0.958`` at ``h=1``), because a one-unit *structural* shock is one standard deviation of the orthogonalized innovation rather than a one-unit move in the observed rate.

### Return Values (`LPForecast`)

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `Matrix{T}` | ``H \times n_{\text{resp}}`` point forecasts |
| `ci_lower` | `Matrix{T}` | Lower CI bounds |
| `ci_upper` | `Matrix{T}` | Upper CI bounds |
| `se` | `Matrix{T}` | Standard errors at each horizon |
| `horizon` | `Int` | Maximum forecast horizon ``H`` |
| `response_vars` | `Vector{Int}` | Response variable indices |
| `shock_var` | `Int` | Shock variable index |
| `shock_path` | `Vector{T}` | Assumed shock trajectory |
| `conf_level` | `T` | Confidence level |
| `ci_method` | `Symbol` | CI method used |

---

## LP-Based FEVD

Standard FEVD computes variance shares from the VMA representation, inheriting any VAR misspecification. Gorodnichenko & Lee (2019) propose an **LP-based FEVD** that estimates variance shares directly via R² regressions, inheriting LP's robustness properties. The decomposition concept, the VMA-based estimator, and the Bayesian variant are documented on [Variance Decomposition](@ref ia_fevd_page); `lp_fevd` is documented here because it is an LP estimator rather than a different view of the same object.

### The R² Estimator

At each horizon ``h``, the share of variable ``i``'s forecast error variance due to shock ``j`` is:

```math
\widehat{\text{FEVD}}_{ij}(h) = R^2\!\left(\hat{f}_{i,t+h|t-1} \sim \hat{\varepsilon}_{j,t+h}, \hat{\varepsilon}_{j,t+h-1}, \ldots, \hat{\varepsilon}_{j,t}\right)
```

where:
- ``\hat{f}_{i,t+h|t-1}`` are LP forecast error residuals for variable ``i`` at horizon ``h``
- ``\hat{\varepsilon}_{j,t+k}`` are leads and current values of structural shock ``j``
- ``R^2`` measures the fraction of forecast error variance explained by shock ``j``

### Alternative Estimators

**LP-A estimator** (Gorodnichenko & Lee 2019, Eq. 9):

```math
\hat{s}_{ij}^{A}(h) = \frac{\sum_{k=0}^{h} (\hat{\beta}_{0,ik}^{\text{LP}})^2 \, \hat{\sigma}_{\varepsilon_j}^2}{\text{Var}(\hat{f}_{i,t+h|t-1})}
```

where:
- ``\hat{\beta}_{0,ik}^{\text{LP}}`` is the LP coefficient on shock ``j`` at horizon ``k``
- ``\hat{\sigma}_{\varepsilon_j}^2`` is the variance of structural shock ``j``

**LP-B estimator** (Gorodnichenko & Lee 2019, Eq. 10):

```math
\hat{s}_{ij}^{B}(h) = \frac{\text{numerator}^A}{\text{numerator}^A + \text{Var}(\tilde{v}_{t+h})}
```

where:
- ``\tilde{v}_{t+h}`` are the residuals from the R² regression

LP-B replaces the total forecast error variance denominator with the sum of explained and unexplained components, improving finite-sample performance.

### Bias Correction

LP-FEVD estimates can be biased in finite samples. Following Kilian (1998), the package implements VAR-based bootstrap bias correction:

1. Fit a bivariate VAR(``L``) on ``(z, y)`` with HQIC-selected lag order
2. Compute the "true" FEVD from this VAR
3. Simulate ``B`` bootstrap samples and compute LP-FEVD for each
4. Estimate bias = mean(bootstrap) - true
5. Bias-corrected estimate = raw - bias

```@example lp
# R²-based LP-FEVD with bias correction, on the Cholesky-identified structural LP
# A seeded RNG keeps the bootstrap bias correction reproducible across builds.
lfevd = lp_fevd(slp, 20; method=:r2, bias_correct=true, n_boot=50,
                rng=MersenneTwister(20260801))
report(lfevd)
```

```julia
plot_result(lfevd)
```

```@raw html
<iframe src="../assets/plots/fevd_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Each variable's own shock dominates its forecast error at every horizon — 86.2%, 87.4%, and 88.8% at ``h=1``, still 75.5%, 84.9%, and 79.9% at ``h=20`` — which is the expected pattern under a Cholesky ordering on weakly-related monthly series. The monetary policy shock explains essentially none of the variance of `INDPRO` or `CPIAUCSL` at any horizon, while the price shock accounts for a growing share of funds-rate variance (3.5% at ``h=1``, 23.7% at ``h=12``), consistent with a policy rule that responds to inflation with a lag. Bootstrap standard errors in parentheses run 2-18 percentage points, so at 50 replications on 60 observations these shares are indicative rather than sharply estimated.

```@example lp
# The raw and bias-corrected estimates differ where finite-sample bias bites
(raw = round.(lfevd.proportions[1, 3, 1:5], digits=4),
 corrected = round.(lfevd.bias_corrected[1, 3, 1:5], digits=4))
```

`lfevd.proportions[i, j, h]` holds the raw R² from regressing variable ``i``'s forecast error on shock ``j``'s leads; `lfevd.bias_corrected` holds the Kilian (1998) bootstrap-corrected version, and it is the corrected array that `report` displays. For `INDPRO`'s share attributable to the policy shock the raw values climb from 0.14% to 8.66% over the first five horizons while the corrected values stay at or within 0.01% of zero — the entire raw share is finite-sample bias, since an R² is bounded below by 0 and therefore biased upward whenever the true share is near zero. That gap is largest at short horizons, exactly where the correction is designed to bite. Comparing the three estimators (`:r2`, `:lp_a`, `:lp_b`) is a further robustness check: substantial disagreement suggests the VAR specification behind the identification is unreliable, in which case the LP-based estimates are preferred.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:r2` | Estimator (`:r2`, `:lp_a`, `:lp_b`) |
| `bias_correct` | `Bool` | `true` | Apply bootstrap bias correction |
| `n_boot` | `Int` | `500` | Number of bootstrap replications |
| `conf_level` | `Real` | `0.95` | Confidence level for CIs |
| `var_lags` | `Union{Nothing,Int}` | `nothing` | Lag order of the bias-correction VAR; `nothing` selects by HQIC |
| `rng` | `AbstractRNG` | `Random.default_rng()` | Draw source for the bootstrap |

### Return Values (`LPFEVD`)

| Field | Type | Description |
|-------|------|-------------|
| `proportions` | `Array{T,3}` | ``n \times n \times H`` raw FEVD estimates |
| `bias_corrected` | `Array{T,3}` | ``n \times n \times H`` bias-corrected FEVD |
| `se` | `Array{T,3}` | Bootstrap standard errors |
| `ci_lower` | `Array{T,3}` | Lower CI bounds |
| `ci_upper` | `Array{T,3}` | Upper CI bounds |
| `method` | `Symbol` | Estimator used |
| `horizon` | `Int` | Maximum FEVD horizon |
| `n_boot` | `Int` | Number of bootstrap replications |
| `conf_level` | `T` | Confidence level |
| `bias_correction` | `Bool` | Whether bias correction was applied |
| `variables` / `shocks` | `Vector{String}` | Variable and shock labels |
| `n_requested` / `n_effective` / `n_failed` | `Int` | Bootstrap draws attempted, usable, and dropped across all cells |

---

## LP vs. VAR

Plagborg-Møller & Wolf (2021) show that under correct specification, LP and VAR IRFs are asymptotically equivalent:

```math
\sqrt{T}(\hat{\beta}_h^{\text{LP}} - \beta_h) \xrightarrow{d} N(0, V^{\text{LP}}), \qquad \sqrt{T}(\hat{\theta}_h^{\text{VAR}} - \theta_h) \xrightarrow{d} N(0, V^{\text{VAR}})
```

where:
- ``V^{\text{LP}} \geq V^{\text{VAR}}`` (VAR is weakly more efficient under correct specification)
- ``\beta_h = \theta_h`` (both target the same population IRF)

The key trade-off is bias vs. variance:

| Aspect | VAR | Local Projections |
|--------|-----|-------------------|
| **Efficiency** | More efficient if correctly specified | Less efficient, but robust |
| **Bias** | Biased if dynamics misspecified | Consistent under weak conditions |
| **Long horizons** | Compounds specification error | Each horizon estimated directly |
| **Nonlinearities** | Requires extensions | Easy to incorporate |
| **External instruments** | SVAR-IV | LP-IV |

Use LP when concerned about VAR misspecification, when incorporating external instruments or nonlinearities, when working with discrete treatments, or at long horizons where VAR error compounds. The VAR side of this comparison — estimation, identification, and bootstrap inference — is documented on the [VAR](@ref var_page) page.

The [`compare_var_lp`](@ref) helper quantifies this equivalence directly, estimating both a Cholesky-identified VAR and the matching Cholesky LP on the same data and returning their IRFs alongside the horizon-by-horizon difference:

```@example lp
cmp = compare_var_lp(Y, 20; lags=4)
(gap_first5 = round.(cmp.difference[1:5, :, 3], digits=3),  # VAR − LP, FFR shock
 max_abs_gap = round(maximum(abs.(cmp.difference)), digits=3))
```

Over the first five horizons the VAR−LP gap for the funds-rate shock never exceeds ``0.035`` in absolute value, and for the two differenced series it is ``0.001`` or smaller — the two estimators are telling the same story about this data. The largest disagreement anywhere in the ``20 \times 3 \times 3`` array is ``0.126``, and it occurs at long horizons where the LP effective sample has shrunk to 36 observations. Under correct specification these differences shrink toward zero as the sample grows; the residual gaps here reflect finite-sample bias-variance trade-offs, not a population discrepancy. A gap that *grows* with the horizon instead of shrinking is the diagnostic signature of VAR misspecification, and it is the case in which LP should be preferred.

---

## Complete Example

This example demonstrates a full LP workflow --- estimation, structural identification, IRF extraction, FEVD, and forecasting --- using FRED-MD monetary policy data.

```@example lp
# Step 1: Standard LP-IRF with Newey-West standard errors
lp_full = estimate_lp(Y, 3, 20; lags=4, cov_type=:newey_west, varnames=vnames)
irf_full = lp_irf(lp_full; conf_level=0.95)
report(irf_full)
```

```@example lp
# Step 2: Structural LP with Cholesky identification
slp_full = structural_lp(Y, 20; method=:cholesky, lags=4, varnames=vnames)
report(slp_full)
```

```@example lp
# Step 3: LP-FEVD with bias correction
lfevd_full = lp_fevd(slp_full, 20; method=:r2, bias_correct=true, n_boot=50,
                     rng=MersenneTwister(20260801))
report(lfevd_full)
```

```@example lp
# Step 4: Direct multi-step forecast from a one-period impulse
impulse_path = zeros(20); impulse_path[1] = 1.0
fc_full = forecast(lp_full, impulse_path; ci_method=:analytical, conf_level=0.95)
report(fc_full)
```

```julia
plot_result(irf_full)
plot_result(slp_full)
plot_result(lfevd_full)
plot_result(fc_full)
```

```@raw html
<iframe src="../assets/plots/irf_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/irf_structural_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/fevd_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/forecast_lp.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `estimate_lp` call fits 21 horizon-specific OLS regressions (``h = 0, \ldots, 20``) with Newey-West HAC standard errors, producing the reduced-form IRF of a federal funds rate innovation on 56 effective observations at ``h = 0``, falling to 36 at ``h = 20``. The `structural_lp` call estimates a VAR(4) for Cholesky identification, recovers orthogonalized structural shocks, and re-estimates the LP regressions using each structural shock as the impulse variable; comparing its `INDPRO` impact response of ``-0.0006`` with the reduced-form ``0.0201`` shows how much of the raw correlation was the endogenous policy response. The LP-FEVD decomposes forecast error variance without the VMA representation, and its bias correction removes the entire raw share attributed to the policy shock at short horizons. The direct forecast projects each response variable forward from the LP coefficients under a one-period unit impulse, and its intervals widen with the horizon exactly as the shrinking effective sample implies.

---

## Common Pitfalls

1. **Wider confidence intervals than VAR**: LP confidence bands are wider than VAR-based bands by construction. This reflects the efficiency cost of not imposing dynamic restrictions, not a deficiency. If LP and VAR point estimates agree but LP CIs are much wider, the VAR specification is likely correct and the VAR-based inference is more powerful.

2. **Newey-West bandwidth too small**: The default automatic bandwidth ensures ``m \geq h + 1`` at each horizon ``h``, accounting for the MA(``h-1``) serial correlation. Manually setting a small bandwidth (e.g., `bandwidth=1`) produces invalid standard errors at horizons ``h > 1``. Use `bandwidth=0` for automatic selection.

3. **State variable choice in state-dependent LP**: The state variable ``z_t`` must be predetermined (known at time ``t`` before the shock realization). Using a contemporaneous variable creates endogeneity. The standard choice is a backward-looking moving average of GDP growth, standardized to zero mean and unit variance. Check the reported `% in expansion`: an estimated transition that concentrates the sample in one regime leaves the other with too few effective observations, and the HAC covariance turns near-singular. Calibrating ``\gamma`` and setting ``c = 0`` avoids this on short samples.

4. **Propensity score overlap**: Extreme propensity scores (near 0 or 1) produce large inverse weights that inflate variance and can cause numerical instability. Always set `trimming=(0.01, 0.99)` to cap extreme weights. Check `propensity_diagnostics()` for overlap violations before interpreting ATE estimates.

5. **Effective sample shrinks with horizon**: Each horizon ``h`` loses ``h`` observations from the end of the sample. At ``h = 20`` with ``T = 100``, only 80 observations remain. With short samples and long horizons, estimates at large ``h`` are unreliable regardless of the standard error correction.

6. **Smooth LP overfitting with few knots**: Too few interior knots in the B-spline basis restrict the IRF to low-frequency shapes that cannot capture sharp impact effects. Too many knots reduce the smoothing benefit. Use `cross_validate_lambda` to select the smoothing parameter automatically.

7. **`estimate_smooth_lp` does not smooth by default**: `lambda` defaults to `0.0`, the *unpenalized* spline fit — the only smoothing then comes from the finite basis. Supply a positive `lambda`, or select one with `cross_validate_lambda`, to use the roughness penalty at all.

8. **`variance_reduction` is a ratio, not a difference**: `compare_smooth_lp` returns ``\text{mean}(\text{se}^2_{\text{smooth}}) / \text{mean}(\text{se}^2_{\text{standard}})``. Values below 1 mean the smooth IRF is more precise; comparing it to 0 always looks favourable and says nothing.

---

## References

- Montiel Olea, J. L., & Pflueger, C. (2013). A Robust Test for Weak Instruments.
  *Journal of Business & Economic Statistics*, 31(3), 358-369. [DOI](https://doi.org/10.1080/00401706.2013.806694)

- Andrews, I., Stock, J. H., & Sun, L. (2019). Weak Instruments in Instrumental Variables Regression: Theory and Practice.
  *Annual Review of Economics*, 11, 727-753. [DOI](https://doi.org/10.1146/annurev-economics-080218-025643)

- Angrist, J. D., Jordà, Ò., & Kuersteiner, G. M. (2018). Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited.
  *Journal of Business & Economic Statistics*, 36(3), 371-387. [DOI](https://doi.org/10.1080/07350015.2016.1204919)

- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative Sign Restrictions for SVARs.
  *American Economic Review*, 108(10), 2802-2829. [DOI](https://doi.org/10.1257/aer.20161852)

- Auerbach, A. J., & Gorodnichenko, Y. (2012). Measuring the Output Responses to Fiscal Policy.
  *American Economic Journal: Economic Policy*, 4(2), 1-27. [DOI](https://doi.org/10.1257/pol.4.2.1)

- Auerbach, A. J., & Gorodnichenko, Y. (2013). Fiscal Multipliers in Recession and Expansion. In *Fiscal Policy after the Financial Crisis*, 63-98. University of Chicago Press. [DOI](https://doi.org/10.7208/chicago/9780226018584.003.0003)

- Barnichon, R., & Brownlees, C. (2019). Impulse Response Estimation by Smooth Local Projections.
  *Review of Economics and Statistics*, 101(3), 522-530. [DOI](https://doi.org/10.1162/rest_a_00778)

- Blanchard, O. J., & Quah, D. (1989). The Dynamic Effects of Aggregate Demand and Supply Disturbances.
  *American Economic Review*, 79(4), 655-673. [JSTOR](https://www.jstor.org/stable/1827924)

- Gorodnichenko, Y., & Lee, B. (2019). Forecast Error Variance Decompositions with Local Projections.
  *Journal of Business & Economic Statistics*, 38(4), 921-933. [DOI](https://doi.org/10.1080/07350015.2019.1610661)

- Hirano, K., Imbens, G. W., & Ridder, G. (2003). Efficient Estimation of Average Treatment Effects Using the Estimated Propensity Score.
  *Econometrica*, 71(4), 1161-1189. [DOI](https://doi.org/10.1111/1468-0262.00442)

- Hyvärinen, A. (1999). Fast and Robust Fixed-Point Algorithms for Independent Component Analysis.
  *IEEE Transactions on Neural Networks*, 10(3), 626-634. [DOI](https://doi.org/10.1109/72.761722)

- Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections.
  *American Economic Review*, 95(1), 161-182. [DOI](https://doi.org/10.1257/0002828053828518)

- Kilian, L. (1998). Small-Sample Confidence Intervals for Impulse Response Functions.
  *Review of Economics and Statistics*, 80(2), 218-230. [DOI](https://doi.org/10.1162/003465398557465)

- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  *Econometrica*, 55(3), 703-708. [DOI](https://doi.org/10.2307/1913610)

- Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation.
  *Review of Economic Studies*, 61(4), 631-653. [DOI](https://doi.org/10.2307/2297912)

- Plagborg-Møller, M., & Wolf, C. K. (2021). Local Projections and VARs Estimate the Same Impulse Responses.
  *Econometrica*, 89(2), 955-980. [DOI](https://doi.org/10.3982/ECTA17813)

- Ramey, V. A., & Zubairy, S. (2018). Government Spending Multipliers in Good Times and in Bad: Evidence from US Historical Data.
  *Journal of Political Economy*, 126(2), 850-901. [DOI](https://doi.org/10.1086/696277)

- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of Regression Coefficients When Some Regressors Are Not Always Observed.
  *Journal of the American Statistical Association*, 89(427), 846-866. [DOI](https://doi.org/10.1080/01621459.1994.10476818)

- Stock, J. H., & Watson, M. W. (2018). Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments.
  *Economic Journal*, 128(610), 917-948. [DOI](https://doi.org/10.1111/ecoj.12593)

- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In D. W. K. Andrews & J. H. Stock (Eds.),
  *Identification and Inference for Econometric Models*, 80-108. Cambridge University Press. [DOI](https://doi.org/10.1017/CBO9780511614491.006)

- Uhlig, H. (2005). What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure.
  *Journal of Monetary Economics*, 52(2), 381-419. [DOI](https://doi.org/10.1016/j.jmoneco.2004.05.007)
