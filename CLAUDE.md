# CLAUDE.md — MacroEconometricModels.jl

## Project Overview

Julia package (v0.1.3) for macroeconomic time series analysis. Provides VAR, Bayesian VAR, Local Projections, Factor Models, ARIMA, GMM estimation, structural identification, and hypothesis testing. Requires Julia 1.10+.

- **Author**: Wookyung Chung
- **License**: MIT
- **Repo**: `github.com/chung9207/MacroEconometricModels.jl`
- **DOI**: `10.5281/zenodo.18439170`
- **Docs**: `chung9207.github.io/MacroEconometricModels.jl/dev/`

## Quick Reference

```bash
# Run tests (5107 pass, 2 pre-existing broken)
julia --project=. -e 'using Pkg; Pkg.test()'

# Load check
julia --project=. -e 'using MacroEconometricModels; println("OK")'

# Smoke test
julia --project=. -e 'using MacroEconometricModels; m = estimate_var(randn(100,3), 2); println(m)'
```

## Architecture

### Source File Map (`src/`, 9 subdirectories + 2 root files, ~12k lines)

```
src/
├── MacroEconometricModels.jl     # Module definition, includes, exports (~350 lines)
├── summary.jl                     # Cross-cutting: summary(), table(), print_table() for all types
│
├── core/                          # Shared infrastructure
│   ├── types.jl                  # Abstract type hierarchy (AbstractVARModel, AbstractAnalysisResult, etc.)
│   ├── utils.jl                  # robust_inv, safe_cholesky, companion_matrix, @float_fallback
│   ├── display.jl                # PrettyTables infrastructure (_pretty_table, _fmt, set_display_backend)
│   └── covariance.jl             # Newey-West (HAC), White (HC0), Driscoll-Kraay estimators
│
├── var/                           # VAR estimation and structural analysis
│   ├── types.jl                  # VARModel, ImpulseResponse, FEVD, BayesianIRF/FEVD, MinnesotaHyper
│   ├── estimation.jl             # estimate_var (OLS), select_lag_order (AIC/BIC/HQIC), StatsAPI
│   ├── identification.jl         # identify_cholesky, sign, narrative, long_run, Arias et al. (2018)
│   ├── irf.jl                    # IRF with bootstrap/theoretical/Bayesian CIs
│   ├── fevd.jl                   # Forecast error variance decomposition
│   └── hd.jl                     # Historical decomposition (frequentist, Bayesian, LP, Arias)
│
├── bvar/                          # Bayesian VAR
│   ├── estimation.jl             # estimate_bvar via Turing.jl MCMC, Minnesota prior support
│   ├── priors.jl                 # gen_dummy_obs, log_marginal_likelihood, optimize_hyperparameters
│   └── utils.jl                  # MCMC chain processing, posterior quantiles, threaded computation
│
├── lp/                            # Local Projections
│   ├── types.jl                  # LPModel, LPIVModel, SmoothLPModel, StateLPModel, PropensityLPModel,
│   │                             # StructuralLP, LPForecast, LPFEVD, BSplineBasis
│   ├── core.jl                   # estimate_lp (Jorda 2005), structural_lp, compare_var_lp
│   ├── iv.jl                     # LP-IV (Stock & Watson 2018)
│   ├── smooth.jl                 # Smooth LP (Barnichon & Brownlees 2019)
│   ├── state.jl                  # State-dependent LP (Auerbach & Gorodnichenko 2013)
│   ├── propensity.jl             # Propensity score LP (Angrist et al. 2018)
│   ├── forecast.jl               # LP forecasting with analytical/bootstrap CIs
│   └── fevd.jl                   # LP-FEVD (Gorodnichenko & Lee 2019)
│
├── factor/                        # Factor models
│   ├── kalman.jl                 # FactorForecast type, Kalman filter/smoother, shared forecast helpers
│   ├── static.jl                 # FactorModel struct, estimate_factors (PCA), Bai-Ng IC, forecast
│   ├── dynamic.jl                # DynamicFactorModel struct, estimation (twostep/EM), forecast
│   └── generalized.jl            # GeneralizedDynamicFactorModel struct, spectral estimation, forecast
│
├── unitroot/                      # Unit root and cointegration tests (12 files)
│   ├── types.jl                  # AbstractUnitRootTest + result structs (ADF, KPSS, PP, ZA, NgPerron, Johansen)
│   ├── critical_values.jl        # MacKinnon, KPSS, ZA, Ng-Perron, Johansen tables
│   ├── helpers.jl                # p-value computation, lag selection, bandwidth, OLS helpers
│   ├── adf.jl                    # Augmented Dickey-Fuller test
│   ├── kpss.jl                   # KPSS stationarity test
│   ├── pp.jl                     # Phillips-Perron test
│   ├── za.jl                     # Zivot-Andrews structural break test
│   ├── ngperron.jl               # Ng-Perron MZa/MZt/MSB/MPT tests
│   ├── johansen.jl               # Johansen cointegration test
│   ├── stationarity.jl           # is_stationary(::VARModel), VARStationarityResult
│   ├── convenience.jl            # unit_root_summary(), test_all_variables()
│   └── show.jl                   # Base.show methods for all unit root result types
│
├── gmm/                           # Generalized Method of Moments
│   └── gmm.jl                   # estimate_gmm (one-step/two-step/iterated), J-test, LP-GMM
│
├── arima/                         # ARIMA (univariate time series, 5 files)
│   ├── types.jl                  # AbstractARIMAModel, ARModel, MAModel, ARMAModel, ARIMAModel
│   ├── kalman.jl                 # State-space Kalman filter for exact MLE
│   ├── estimation.jl             # estimate_ar/ma/arma/arima, CSS/MLE optimization
│   ├── forecast.jl               # Multi-step forecasting, psi-weights, confidence bands
│   └── selection.jl              # auto_arima, select_arima_order, ic_table
│
└── nongaussian/                   # Non-Gaussian SVAR identification
    ├── normality.jl              # JB, Mardia, Doornik-Hansen, Henze-Zirkler, suite
    ├── ica.jl                    # FastICA, JADE, SOBI, dCov, HSIC
    ├── ml.jl                     # Student-t, mixture-normal, PML, skew-normal
    ├── heteroskedastic.jl        # Markov-switching, GARCH, smooth-transition, external
    └── tests.jl                  # Identification strength, gaussianity, independence, overid
```

### Include Order

The include order in `MacroEconometricModels.jl` matters due to dependencies:
1. `core/utils.jl` (no deps)
2. `core/types.jl` (abstract types)
3. `core/display.jl` (PrettyTables formatting)
4. `var/types.jl` (concrete VAR/IRF/FEVD types)
5. `var/estimation.jl`, `bvar/priors.jl`, `bvar/estimation.jl`
6. `unitroot/` (12 files: types → critical_values → helpers → adf → kpss → pp → za → ngperron → johansen → stationarity → convenience → show)
7. `var/identification.jl`
8. `nongaussian/` (5 files: normality → ica → ml → heteroskedastic → tests)
9. `bvar/utils.jl` (needs bayesian + identification)
10. Factor models: `factor/kalman.jl` → `factor/static.jl` → `factor/dynamic.jl` → `factor/generalized.jl`
11. `gmm/gmm.jl`
12. ARIMA: `arima/types.jl` → `arima/kalman.jl` → `arima/estimation.jl` → `arima/forecast.jl` → `arima/selection.jl`
13. `core/covariance.jl`
14. LP: `lp/types.jl` → `lp/core.jl` → `lp/iv.jl` → `lp/smooth.jl` → `lp/state.jl` → `lp/propensity.jl` → `lp/forecast.jl`
15. `var/irf.jl`, `var/fevd.jl`, `var/hd.jl` (need LP types for `lp_irf`)
16. `lp/fevd.jl` (LP-FEVD, needs irf + fevd)
17. `summary.jl` (needs all result types)

### Key Dependencies

| Dependency | Purpose |
|---|---|
| `StatsAPI` | Standard Julia stats interface (`fit`, `coef`, `predict`, etc.) |
| `Turing` (0.38–0.42) | Bayesian MCMC sampling for BVAR |
| `Optim` (1, 2) | Numerical optimization (imported, not `using`) |
| `Distributions` | Statistical distributions |
| `FFTW` | FFT for spectral factor models (GDFM) |
| `MCMCChains` | MCMC chain processing |
| `DataFrames` | Data input/output, `table()` returns DataFrames |
| `PrettyTables` | `print_table()` formatted output |
| `SpecialFunctions` | Used by unit root critical value computations |

### Type Hierarchy

```
AbstractVARModel
  └── VARModel{T}

AbstractAnalysisResult
  ├── AbstractFrequentistResult
  │     ├── ImpulseResponse{T}, FEVD{T}, HistoricalDecomposition{T}
  └── AbstractBayesianResult
        ├── BayesianImpulseResponse{T}, BayesianFEVD{T}, BayesianHistoricalDecomposition{T}

AbstractImpulseResponse
  ├── ImpulseResponse{T}, BayesianImpulseResponse{T}
  └── AbstractLPImpulseResponse → LPImpulseResponse{T}

AbstractFEVD
  ├── FEVD{T}, BayesianFEVD{T}, LPFEVD{T}

AbstractFactorModel
  ├── FactorModel{T}, DynamicFactorModel{T}, GeneralizedDynamicFactorModel{T}

FactorForecast{T}

AbstractLPModel
  ├── LPModel{T}, LPIVModel{T}, SmoothLPModel{T}, StateLPModel{T}, PropensityLPModel{T}

StructuralLP{T}
LPForecast{T}

AbstractCovarianceEstimator
  ├── NeweyWestEstimator{T}, WhiteEstimator, DriscollKraayEstimator{T}

AbstractGMMModel → GMMModel{T}

AbstractPrior → MinnesotaHyperparameters{T}

StatsAPI.RegressionModel
  └── AbstractARIMAModel
        ├── ARModel{T}, MAModel{T}, ARMAModel{T}, ARIMAModel{T}

AbstractUnitRootTest <: StatsAPI.HypothesisTest
  ├── ADFResult{T}, KPSSResult{T}, PPResult{T}, ZAResult{T}, NgPerronResult{T}, JohansenResult{T}

VARStationarityResult{T}
```

## API Quick Reference

### Estimation Functions

| Function | Description |
|---|---|
| `estimate_var(Y, p)` | VAR(p) via OLS |
| `estimate_bvar(Y, p; prior=:minnesota, hyper=..., n_samples=2000, n_adapts=500)` | Bayesian VAR via MCMC |
| `estimate_ar(y, p; method=:ols)` | AR(p), methods: `:ols`, `:mle` |
| `estimate_ma(y, q; method=:css_mle)` | MA(q), methods: `:css`, `:mle`, `:css_mle` |
| `estimate_arma(y, p, q; method=:css_mle)` | ARMA(p,q) |
| `estimate_arima(y, p, d, q; method=:css_mle)` | ARIMA(p,d,q) |
| `estimate_lp(Y, shock_var, H; lags=4, cov_type=:newey_west)` | Local Projection (Jorda 2005) |
| `estimate_lp_iv(Y, shock_var, Z, H)` | LP-IV (Stock & Watson 2018) |
| `estimate_smooth_lp(Y, shock_var, H; degree=3, n_knots=4, lambda=1.0)` | Smooth LP (Barnichon & Brownlees 2019) |
| `estimate_state_lp(Y, shock_var, state_var, H; gamma=1.5)` | State-dependent LP (Auerbach & Gorodnichenko 2013) |
| `estimate_propensity_lp(Y, treatment, covariates, H)` | Propensity score LP (Angrist et al. 2018) |
| `doubly_robust_lp(Y, treatment, covariates, H)` | Doubly robust LP estimator |
| `estimate_factors(X, r; standardize=true)` | Static factor model via PCA |
| `estimate_dynamic_factors(X, r, p; method=:twostep)` | Dynamic factor model |
| `estimate_gdfm(X, q; kernel=:bartlett)` | Generalized dynamic factor model (spectral) |
| `estimate_gmm(moment_fn, theta0, data; weighting=:two_step)` | GMM estimation |

### Structural Analysis

| Function | Description |
|---|---|
| `irf(model, H; method=:cholesky, ci_type=:bootstrap)` | Impulse response functions |
| `irf(chain, p, n, H; method=:cholesky)` | Bayesian IRF from MCMC chain |
| `fevd(model, H; method=:cholesky)` | Forecast error variance decomposition |
| `historical_decomposition(model, T)` | Historical decomposition |
| `identify_cholesky(model)` | Cholesky (recursive) identification |
| `identify_sign(model; check_func=..., n_draws=1000)` | Sign restriction identification |
| `identify_narrative(model; ...)` | Narrative sign restrictions (Antolin-Diaz & Rubio-Ramirez 2018) |
| `identify_long_run(model)` | Blanchard-Quah long-run identification |
| `identify_arias(model, restrictions)` | Arias et al. (2018) SVAR |
| `forecast(model, h)` | ARIMA forecasting |
| `forecast(fm, h; p=1, ci_method=:none)` | Static factor model forecasting (fits VAR(p) on factors) |
| `forecast(dfm, h; ci_method=:none)` | DFM forecasting (`:none/:theoretical/:bootstrap/:simulation`) |
| `forecast(gdfm, h; ci_method=:none)` | GDFM forecasting (`:none/:theoretical/:bootstrap`) |
| `structural_lp(Y, H; method=:cholesky)` | Structural LP with multi-shock IRFs (Plagborg-Møller & Wolf 2021) |
| `forecast(lp_model, h; ci_method=:analytical)` | LP direct multi-step forecasts with CIs |
| `lp_fevd(slp; estimator=:r2)` | LP-FEVD (Gorodnichenko & Lee 2019) |

### Unit Root & Cointegration Tests

| Function | Null Hypothesis | Key Options |
|---|---|---|
| `adf_test(y)` | Unit root | `lags=:aic`, `regression=:constant/:trend/:none` |
| `kpss_test(y)` | Stationarity | `regression=:constant/:trend`, `bandwidth=:auto` |
| `pp_test(y)` | Unit root | `regression=:constant/:trend` |
| `za_test(y)` | Unit root (no break) | `regression=:constant/:trend/:both`, `trim=0.15` |
| `ngperron_test(y)` | Unit root | Reports MZa, MZt, MSB, MPT |
| `johansen_test(Y, p)` | rank <= r | `deterministic=:constant/:trend/:none` |
| `unit_root_summary(y)` | Multiple tests | `tests=[:adf, :kpss, :pp]` |
| `test_all_variables(Y)` | All columns | `test=:adf` |

### Summary & Output

| Function | Description |
|---|---|
| `MacroEconometricModels.summary(obj)` | Print comprehensive summary (use qualified name to avoid `Base.summary` conflict) |
| `table(irf_result, var, shock; horizons=[1,4,8,12])` | Extract as DataFrame |
| `print_table(io, obj, ...)` | PrettyTables formatted output |
| `point_estimate(obj)` | Extract main values |
| `has_uncertainty(obj)` / `uncertainty_bounds(obj)` | Check/get CI/credible intervals |

### Diagnostics

| Function | Description |
|---|---|
| `select_lag_order(Y, max_p)` | VAR lag selection (AIC/BIC/HQIC) |
| `select_arima_order(y, max_p, max_q)` | ARIMA order selection grid |
| `auto_arima(y)` | Automatic ARIMA order selection |
| `ic_criteria(X, r_max)` | Bai-Ng info criteria for factor count (IC1/IC2/IC3) |
| `ic_criteria_dynamic(X, max_r, max_p)` | DFM factor/lag selection |
| `ic_criteria_gdfm(X, max_q)` | GDFM dynamic factor selection |
| `is_stationary(model)` | Check VAR/factor model stationarity |
| `weak_instrument_test(lpiv_model; threshold=10.0)` | LP-IV first-stage F-test |
| `sargan_test(model, h)` | Overidentification test |
| `j_test(gmm_result)` | Hansen J-test |
| `optimize_hyperparameters(Y, p)` | Optimize Minnesota prior tau via marginal likelihood |
| `cross_validate_lambda(Y, shock_var, H)` | Smooth LP lambda selection |
| `test_regime_difference(state_model)` | Test expansion vs recession differences |
| `propensity_diagnostics(model)` | Propensity score overlap/balance |
| `verify_decomposition(hd)` | Check HD identity holds |

## Test Structure

26 test files in `test/`, orchestrated by `runtests.jl`. Key test files:
- `test_core_var.jl` — VAR estimation and StatsAPI
- `test_bayesian.jl` — BVAR with different samplers
- `test_arima.jl` — ARIMA estimation and forecasting
- `test_lp.jl` — All LP variants (~38k, large)
- `test_arias2018.jl` — Arias SVAR identification (~35k, large)
- `test_unitroot.jl` — All unit root tests
- `test_aqua.jl` — Code quality (ambiguities, unbound args, etc.)
- `test_examples.jl` — Full integration examples (~17k)
- `test_factor_forecast.jl` — Factor model forecasting with CIs (all 3 types)
- `test_gmm.jl` — GMM estimation and J-test
- `test_covariance.jl` — Covariance estimators (Newey-West, White, Driscoll-Kraay)
- `test_edge_cases.jl` — Edge cases and error handling

## Conventions

- Internal helpers prefixed with `_` (e.g., `_unpack_arma_params`, `_white_noise_fit`)
- `Optim` is `import`ed (not `using`), so always qualify: `Optim.optimize`, `Optim.LBFGS()`
- StatsAPI methods use `StatsAPI.` prefix in definitions (e.g., `StatsAPI.nobs(m::AbstractARIMAModel)`)
- Type parameters use `T<:AbstractFloat` throughout
- `robust_inv` from `utils.jl` used instead of raw `inv()` for numerical safety
- `safe_cholesky` adds jitter for numerical stability
- Functions accept `AbstractVector`/`AbstractMatrix` in public API, convert to `Vector{T}`/`Matrix{T}` internally
- `@float_fallback` macro auto-generates `Float64` conversion methods for functions with typed signatures
- `summary()` conflicts with `Base.summary` — use `MacroEconometricModels.summary(obj)` or import explicitly

## Notation Conventions (used in docs and docstrings)

| Symbol | Description |
|---|---|
| `y_t` | n x 1 vector of endogenous variables at time t |
| `Y` | T x n data matrix |
| `p` | Number of lags |
| `A_i` | n x n coefficient matrix for lag i |
| `Sigma` | n x n reduced-form error covariance |
| `B_0` | n x n contemporaneous impact matrix (`u_t = B_0 * eps_t`) |
| `eps_t` | n x 1 structural shocks (unit variance) |
| `u_t` | n x 1 reduced-form residuals |
| `h`, `H` | Forecast/IRF horizon, max horizon |

## Common Patterns

- **VAR estimation**: `Y` is T x n matrix, `model = estimate_var(Y, p)` returns `VARModel` with fields `B`, `Sigma`, `residuals`, `aic`, `bic`
- **Identification**: produces rotation matrix Q; IRFs computed via `compute_irf(model, Q, horizon)`. The decomposition is `Sigma = B_0 * B_0'` where `B_0 = chol(Sigma) * Q`
- **Bayesian workflow**: `estimate_bvar` → MCMCChains object → `irf(chain, p, n, H)` / `fevd(chain, p, n, H)`. Bayesian IRF quantiles array is (H+1) x n x n x 3 where dim 4 = [16th pctl, median, 84th pctl]
- **LP workflow**: `estimate_lp(Y, shock_var, H)` → `lp_irf(lp_model)` for standard; `estimate_lp_iv` → `lp_iv_irf` for IV; `estimate_smooth_lp` → `smooth_lp_irf` for smooth; `estimate_state_lp` → `state_irf` for state-dependent; `structural_lp(Y, H)` for multi-shock structural LP; `forecast(lp_model, h)` for LP forecasting; `lp_fevd(slp)` for LP-FEVD
- **Factor model workflow**: `ic_criteria(X, r_max)` to select r, then `estimate_factors(X, r)`. For forecasting: `forecast(fm, h; ci_method=:theoretical)` returns `FactorForecast` with factor/observable forecasts and CIs. For FAVAR: `hcat(Y_key, fm.factors)` then `estimate_var`
- **Factor forecast helpers**: 5 shared helpers in `kalman.jl` — `_factor_forecast_var_theoretical` (VMA MSE), `_factor_forecast_bootstrap` (residual resampling), `_factor_forecast_obs_se` (Λ·MSE·Λ'+Σ_e), `_unstandardize_factor_forecast!`, `_build_factor_forecast`
- **Unit root workflow**: ADF+KPSS together. ADF reject + KPSS fail-to-reject = stationary; ADF fail + KPSS reject = unit root; both reject = possible structural break; both fail = inconclusive
- **ARIMA DRY helpers**: `_unpack_arma_params`/`_pack_arma_params` for parameter vectors, `_compute_aic_bic` for information criteria, `_white_noise_fit` for p=q=0 edge case, `_confidence_band` for forecast CIs, `_show_arima_model` for display

## Minnesota Prior Hyperparameters

```julia
MinnesotaHyperparameters(tau=0.5, d=2.0, omega_own=1.0, omega_cross=1.0, omega_det=1.0)
```

| Parameter | Effect | Typical range |
|---|---|---|
| `tau` | Overall shrinkage (lower = more) | 0.01 – 1.0 |
| `d` | Lag decay (higher = faster decay) | 1, 2, 3 |
| `omega_cross` | Cross-variable penalty (lower = more) | 0.5 – 1.0 |

Prior variance for coefficient (i,j) at lag l: `tau^2 / l^d` for own lags, `tau^2 * omega^2 / l^d * sigma_i^2 / sigma_j^2` for cross lags. Optimize via `optimize_hyperparameters(Y, p)` which maximizes marginal likelihood (Giannone, Lenza & Primiceri 2015).

## Documentation

### Build & Serve

```bash
# Build docs (requires dev-installing the package in docs project first)
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl

# Output goes to docs/build/
```

### File Map (`docs/src/`, 12 markdown files)

| File | Category | Content |
|---|---|---|
| `index.md` | Home | Overview, quick start examples, package structure table, notation table, references |
| `arima.md` | Theory | AR, MA, ARMA, ARIMA models; Kalman filter MLE; forecasting; order selection; StatsAPI |
| `manual.md` | Theory | VAR/SVAR (OLS, companion form, stability, IC, identification schemes, HAC) |
| `bayesian.md` | Theory | Minnesota prior, dummy observations, hyperparameter optimization, MCMC with Turing.jl |
| `lp.md` | Theory | Standard LP, LP-IV, Smooth LP (B-splines), State-dependent LP, Propensity score LP |
| `factormodels.md` | Theory | Static FM (PCA), Bai-Ng IC, Dynamic FM (two-step, EM), GDFM (spectral), FAVAR |
| `hypothesis_tests.md` | Theory | ADF, KPSS, PP, Zivot-Andrews, Ng-Perron, Johansen; combined testing workflow |
| `innovation_accounting.md` | Tools | IRF (bootstrap/Bayesian CI), FEVD, Historical Decomposition, summary tables |
| `examples.md` | Examples | 7 worked examples: VAR, BVAR, LP variants, factor models, GMM, workflow, unit root |
| `api.md` | API | Quick reference tables for all functions (organized by category) |
| `api_types.md` | API | Type docs with `@docs` blocks + type hierarchy diagram |
| `api_functions.md` | API | Function docs with `@docs`/`@autodocs` blocks |

### Navigation Structure (`make.jl` pages)

```
Home                        => index.md
Univariate Models/
  ARIMA                     => arima.md
Frequentist Models/
  VAR                       => manual.md
  Local Projections         => lp.md
  Factor Models             => factormodels.md
Bayesian Models/
  Bayesian VAR              => bayesian.md
Innovation Accounting       => innovation_accounting.md
Hypothesis Tests/
  Unit Root & Cointegration => hypothesis_tests.md
Examples                    => examples.md
API Reference/
  Overview                  => api.md
  Types                     => api_types.md
  Functions                 => api_functions.md
```

### Documentation Rules

*Informed by Stata and EViews documentation standards. Our docs should read like an econometrics textbook with working Julia code, not like an API reference.*

#### Core Principles (Stata/EViews-Inspired)

1. **Layered depth**: Each page serves multiple reader needs at increasing depth:
   - Quick Start recipes (2 min) → Model specification (5 min) → Theory + examples (30 min) → Technical details (deep study)
2. **Interleaved theory and examples**: Every theoretical concept is immediately followed by a worked example. Do NOT separate theory from examples into different sections.
3. **Interpretation is mandatory**: After every code output, include a paragraph interpreting the results in substantive terms (e.g., "The AR(1) coefficient of 0.85 indicates high persistence...").
4. **Progressive complexity**: Within each section, start with the simplest case and build to complex variants. Show the simplest function call first, then add options.
5. **Self-contained pages**: Each theory page should be readable on its own without requiring other pages. Use cross-references for related topics, not prerequisites.
6. **Exhaustive return value documentation**: Document all accessible fields of returned structs, like Stata's "Stored results" section.

#### Page Structure

1. **H1 title**: Every page starts with `# Title` — one per file
2. **Intro paragraph**: 1–3 sentences below H1 summarizing what the page covers
3. **Quick Start**: Immediately after the intro, provide 3–6 terse command recipes (no output) showing common use cases, progressing from simplest to most complex. Pattern: one-line description + indented code. (Adapted from Stata's Quick Start section.)
4. **H2 sections**: Major topics separated by `---` horizontal rules above each H2
5. **H3 subsections**: Subdivide H2 sections; do NOT use H4+ (keeps hierarchy flat)
6. **Complete Example**: Second-to-last H2 on theory pages — an end-to-end workflow combining multiple functions (e.g., unit root test → order selection → estimation → diagnostics → forecasting)
7. **References section**: Always the last H2 on theory pages

#### Section Pattern for Theory Pages

Each model/method H2 section follows this canonical order (H3 labels should be descriptive, not verbatim copies of these step names):

1. **Introduction** — motivation, key use cases, primary reference
2. **Model Specification** — display math with ```` ```math ```` blocks, immediately followed by a "where" block defining every symbol (adapted from Stata/EViews pattern):
   ```
   where
   - ``y_t`` is the ``n \times 1`` vector of endogenous variables at time ``t``
   - ``A_i`` are ``n \times n`` coefficient matrices for lag ``i``
   - ``u_t \sim \text{i.i.d.}(0, \Sigma)`` is the reduced-form error
   ```
3. **Estimation / Theory** — algorithm details, derivation. Use `!!! note "Technical Note"` admonitions for advanced digressions that casual readers can skip (adapted from Stata's Technical Note pattern).
4. **Julia Implementation** — code block with `using MacroEconometricModels`, followed by **full REPL-style output** and an **interpretation paragraph** explaining the results
5. **Return Values** — table of key fields: `| Field | Type | Description |`. Document every user-accessible field of the returned struct.
6. **`**Reference**: Author (Year)`** — inline citation at section end

#### Mathematical Notation

- Use Documenter.jl math blocks: ```` ```math ```` for display math
- Use double backticks for inline math: ``` ``y_t`` ```, ``` ``\Sigma`` ```
- Follow the notation table in `index.md` (see Notation Conventions section)
- MathJax3 is the rendering engine (configured in `make.jl`)
- After every display equation, include a "where" block defining all symbols (see Section Pattern above)
- For long derivations, separate accessible math (model spec, interpretation) from rigorous math (proofs, algorithm internals). Put rigorous derivations in a dedicated "Technical Details" H3 or `!!! note "Technical Note"` admonition.

#### Code Examples

- Always start with `using MacroEconometricModels` (and other imports if needed)
- Use `Random.seed!(42)` (or similar) for reproducibility
- Code blocks use ```` ```julia ```` fencing
- **Progressive complexity**: Start with the simplest invocation, then show variants with more options
- **Show full output**: Include REPL output or `println` results so users can compare with their own runs (adapted from Stata's complete console output pattern)
- **Interpretation paragraph**: After every example output, explain what the numbers mean in econometric terms. Do not just show output and move on.
- **Show result access**: Demonstrate how to access fields (e.g., `model.phi`, `result.pvalue`, `fc.forecast`)
- **Citation-grounded examples**: Where possible, replicate or reference published results (e.g., "Following Enders (2004, Ch. 2)...")
- **"Alternative approach" notes**: When multiple syntaxes achieve the same result, show the alternative briefly (e.g., "This can also be fit using `estimate_arima(y, 1, 0, 0)`)

#### Formatting Conventions

- **Bold**: Key term first introduction, emphasis labels like `**Reference**:`, `**Property**:`
- **Italic**: Book titles in references only
- **Code backticks**: Function names, type names, keyword arguments, field names
- **Tables**: For function options/arguments (grouped by category, not alphabetical), comparison charts, notation, return value fields
- **Admonitions**:
  - `!!! note` for important caveats (e.g., `Base.summary` conflict)
  - `!!! note "Technical Note"` for advanced digressions that casual readers can skip
  - `!!! warning` for sign convention differences, numerical pitfalls, or common errors
- **Cross-references**: `[text](@ref target)` for internal Documenter links, `[text](file.md)` for relative links. When a topic has its own dedicated page, include a brief cross-reference rather than duplicating content.
- **Contextual warnings**: Place caveats and sign convention notes exactly where the user will encounter them, not in a separate "gotchas" section (adapted from EViews pattern)

#### References Format

Full bibliographic entries at page end under `## References`. For single-topic pages, a flat alphabetical list is acceptable. For multi-topic pages, group by subtopic under H3 headings:
```
### Subtopic Name

- Author, A. B., & Author, C. D. (Year). "Article Title." *Journal Name*, Volume(Issue), Pages.
- Author, E. F. (Year). *Book Title*. Publisher.
```

Inline citations use `Author (Year)` or `(Author, Year, p. XX)` format.

#### API Documentation Pages

**`api.md`** (Overview):
- Quick reference tables with `| Function | Description |` format
- One table per functional area (estimation, structural analysis, diagnostics, etc.)
- Signature includes key arguments: `function(required; key_opt=default)`
- Options grouped by category (Model, Inference, Output), not alphabetically

**`api_types.md`** (Types):
- Use `@docs TypeName` blocks for each exported type
- Include the full type hierarchy diagram at the bottom as a code block
- For each struct, list all user-accessible fields with types and descriptions

**`api_functions.md`** (Functions):
- Use explicit `@docs function_name` blocks for public functions when filenames could collide
- Use `@autodocs Modules/Pages/Order` blocks only when the filename is unique
- **CRITICAL**: `@autodocs Pages = ["foo.jl"]` does **substring matching** — `"estimation.jl"` matches BOTH `estimation.jl` and `arima_estimation.jl`. When filenames overlap, use explicit `@docs` blocks instead.

#### When Adding a New Module's Documentation

1. Create `docs/src/<module>.md` following the page structure above (Quick Start → Theory sections with interleaved examples → Complete Example → References)
2. Add entry in `make.jl` `pages` under the appropriate nav group
3. Add the module to `index.md`:
   - Key Features bullet point
   - Quick Start example section
   - Package Structure table row
   - References subsection (if new references)
   - `@contents Pages` list
4. Add to `api.md`: quick reference table for the module's functions
5. Add to `api_types.md`: `@docs` blocks for new types + update hierarchy diagram
6. Add to `api_functions.md`: `@docs` or `@autodocs` blocks for new functions
7. Build and verify: `julia --project=docs docs/make.jl`

#### Documenter.jl Configuration (`make.jl`)

- `checkdocs=:exports` — warns about undocumented exports
- `warnonly=[:missing_docs, :cross_references, :autodocs_block, :docs_block]` — non-fatal
- `size_threshold=300 * 1024` — max HTML page size (300KB)
- `mathengine=Documenter.MathJax3()` — LaTeX rendering
- `format=Documenter.HTML(prettyurls=...)` — pretty URLs only in CI

## Key References (by module)

### VAR/SVAR

- Sims, Christopher A. 1980. "Macroeconomics and Reality." *Econometrica* 48 (1): 1–48. [https://doi.org/10.2307/1912017](https://doi.org/10.2307/1912017)
- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
- Kilian, Lutz, and Helmut Lütkepohl. 2017. *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [https://doi.org/10.1017/9781108164818](https://doi.org/10.1017/9781108164818)

### Identification

- Blanchard, Olivier Jean, and Danny Quah. 1989. "The Dynamic Effects of Aggregate Demand and Supply Disturbances." *American Economic Review* 79 (4): 655–673.
- Rubio-Ramírez, Juan F., Daniel F. Waggoner, and Tao Zha. 2010. "Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference." *Review of Economic Studies* 77 (2): 665–696. [https://doi.org/10.1111/j.1467-937X.2009.00578.x](https://doi.org/10.1111/j.1467-937X.2009.00578.x)
- Antolín-Díaz, Juan, and Juan F. Rubio-Ramírez. 2018. "Narrative Sign Restrictions for SVARs." *American Economic Review* 108 (10): 2802–2829. [https://doi.org/10.1257/aer.20161852](https://doi.org/10.1257/aer.20161852)
- Arias, Jonas E., Juan F. Rubio-Ramírez, and Daniel F. Waggoner. 2018. "Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications." *Econometrica* 86 (2): 685–720. [https://doi.org/10.3982/ECTA14468](https://doi.org/10.3982/ECTA14468)

### Bayesian

- Litterman, Robert B. 1986. "Forecasting with Bayesian Vector Autoregressions—Five Years of Experience." *Journal of Business & Economic Statistics* 4 (1): 25–38. [https://doi.org/10.1080/07350015.1986.10509491](https://doi.org/10.1080/07350015.1986.10509491)
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436–451. [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2010. "Large Bayesian Vector Auto Regressions." *Journal of Applied Econometrics* 25 (1): 71–92. [https://doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137)

### Local Projections

- Jordà, Òscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review* 95 (1): 161–182. [https://doi.org/10.1257/0002828053828518](https://doi.org/10.1257/0002828053828518)
- Stock, James H., and Mark W. Watson. 2018. "Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments." *Economic Journal* 128 (610): 917–948. [https://doi.org/10.1111/ecoj.12593](https://doi.org/10.1111/ecoj.12593)
- Barnichon, Regis, and Christian Brownlees. 2019. "Impulse Response Estimation by Smooth Local Projections." *Review of Economics and Statistics* 101 (3): 522–530. [https://doi.org/10.1162/rest_a_00778](https://doi.org/10.1162/rest_a_00778)
- Auerbach, Alan J., and Yuriy Gorodnichenko. 2013. "Fiscal Multipliers in Recession and Expansion." In *Fiscal Policy after the Financial Crisis*, edited by Alberto Alesina and Francesco Giavazzi, 63–98. Chicago: University of Chicago Press. [https://doi.org/10.7208/9780226018584-004](https://doi.org/10.7208/9780226018584-004)
- Angrist, Joshua D., Òscar Jordà, and Guido M. Kuersteiner. 2018. "Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited." *Journal of Business & Economic Statistics* 36 (3): 371–387. [https://doi.org/10.1080/07350015.2016.1204919](https://doi.org/10.1080/07350015.2016.1204919)
- Plagborg-Møller, Mikkel, and Christian K. Wolf. 2021. "Local Projections and VARs Estimate the Same Impulse Responses." *Econometrica* 89 (2): 955–980. [https://doi.org/10.3982/ECTA17813](https://doi.org/10.3982/ECTA17813)

### Factor Models

- Bai, Jushan, and Serena Ng. 2002. "Determining the Number of Factors in Approximate Factor Models." *Econometrica* 70 (1): 191–221. [https://doi.org/10.1111/1468-0262.00273](https://doi.org/10.1111/1468-0262.00273)
- Stock, James H., and Mark W. Watson. 2002. "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association* 97 (460): 1167–1179. [https://doi.org/10.1198/016214502388618960](https://doi.org/10.1198/016214502388618960)
- Forni, Mario, Marc Hallin, Marco Lippi, and Lucrezia Reichlin. 2000. "The Generalized Dynamic-Factor Model: Identification and Estimation." *Review of Economics and Statistics* 82 (4): 540–554. [https://doi.org/10.1162/003465300559037](https://doi.org/10.1162/003465300559037)
- Forni, Mario, Marc Hallin, Marco Lippi, and Lucrezia Reichlin. 2005. "The Generalized Dynamic Factor Model: One-Sided Estimation and Forecasting." *Journal of the American Statistical Association* 100 (471): 830–840. [https://doi.org/10.1198/016214504000002050](https://doi.org/10.1198/016214504000002050)

### Unit Roots

- Dickey, David A., and Wayne A. Fuller. 1979. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74 (366): 427–431. [https://doi.org/10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- MacKinnon, James G. 2010. "Critical Values for Cointegration Tests." Queen's Economics Department Working Paper No. 1227.
- Kwiatkowski, Denis, Peter C. B. Phillips, Peter Schmidt, and Yongcheol Shin. 1992. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics* 54 (1–3): 159–178. [https://doi.org/10.1016/0304-4076(92)90104-Y](https://doi.org/10.1016/0304-4076(92)90104-Y)
- Phillips, Peter C. B., and Pierre Perron. 1988. "Testing for a Unit Root in Time Series Regression." *Biometrika* 75 (2): 335–346. [https://doi.org/10.1093/biomet/75.2.335](https://doi.org/10.1093/biomet/75.2.335)
- Zivot, Eric, and Donald W. K. Andrews. 1992. "Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis." *Journal of Business & Economic Statistics* 10 (3): 251–270. [https://doi.org/10.1080/07350015.1992.10509904](https://doi.org/10.1080/07350015.1992.10509904)
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519–1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)
- Johansen, Søren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551–1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)

### HAC/Inference

- Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55 (3): 703–708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
- Newey, Whitney K., and Kenneth D. West. 1994. "Automatic Lag Selection in Covariance Matrix Estimation." *Review of Economic Studies* 61 (4): 631–653. [https://doi.org/10.2307/2297912](https://doi.org/10.2307/2297912)
- Andrews, Donald W. K. 1991. "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica* 59 (3): 817–858. [https://doi.org/10.2307/2938229](https://doi.org/10.2307/2938229)

### GMM

- Hansen, Lars Peter. 1982. "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica* 50 (4): 1029–1054. [https://doi.org/10.2307/1912775](https://doi.org/10.2307/1912775)
