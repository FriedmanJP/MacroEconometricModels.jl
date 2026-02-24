# MacroEconometricModels.jl

[![CI](https://github.com/FriedmanJP/MacroEconometricModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/FriedmanJP/MacroEconometricModels.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/FriedmanJP/MacroEconometricModels.jl/graph/badge.svg)](https://codecov.io/gh/FriedmanJP/MacroEconometricModels.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://FriedmanJP.github.io/MacroEconometricModels.jl/stable/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18439170.svg)](https://doi.org/10.5281/zenodo.18439170)

A comprehensive Julia package for macroeconomic time series analysis. Provides VAR, VECM, Bayesian VAR, Panel VAR, Local Projections, Factor Models, DSGE, ARIMA, time series filters, GMM/SMM, ARCH/GARCH/Stochastic Volatility estimation, nowcasting (DFM, BVAR, bridge equations with news decomposition), structural identification (Cholesky, sign, zero, narrative, long-run, and penalty function), statistical identification via higher moments (non-Gaussian and heteroskedasticity-based methods), hypothesis testing, typed data containers, publication-quality output with multi-format bibliographic references, and interactive D3.js visualization.

## Features

### Univariate Models
- **Time Series Filters** - Trend-cycle decomposition:
  - Hodrick-Prescott filter
  - Hamilton (2018) regression filter
  - Beveridge-Nelson decomposition (ARIMA psi-weights)
  - Baxter-King band-pass filter
  - Boosted HP filter (Phillips & Shi 2021) with ADF/BIC/fixed stopping
- **ARIMA** - AR, MA, ARMA, ARIMA estimation via CSS, exact MLE (Kalman filter), or CSS-MLE
  - Automatic order selection via `auto_arima` with grid search over (p,d,q), AIC/BIC information criteria
  - Multi-step ahead forecasting with confidence intervals via psi-weight accumulation
- **Volatility Models**:
  - ARCH - Engle (1982) ARCH(q) with MLE, ARCH-LM test, Ljung-Box squared residuals
  - GARCH - GARCH(p,q), EGARCH (Nelson 1991), GJR-GARCH (Glosten, Jagannathan & Runkle 1993)
  - Stochastic Volatility - Bayesian SV via Kim-Shephard-Chib Gibbs sampler (basic, leverage, Student-t variants)
  - Multi-step volatility forecasts with simulation CIs
  - Diagnostics: news impact curves, persistence, half-life, unconditional variance

### Multivariate Models
- **Vector Autoregression (VAR)** - OLS estimation with lag order selection (AIC, BIC, HQ)
- **Bayesian VAR (BVAR)** - Minnesota priors with hyperparameter optimization (Giannone, Lenza & Primiceri 2015)
- **Vector Error Correction Model (VECM)** - Johansen MLE and Engle-Granger two-step estimation
  - Automatic cointegrating rank selection (trace/max-eigenvalue)
  - VAR conversion (`to_var`) enabling all 18+ identification methods
  - VECM-specific forecasting preserving cointegrating relationships
  - Granger causality: short-run, long-run, and strong tests
- **Local Projections (LP)** - Jorda (2005) with extensions:
  - HAC standard errors (Newey-West, White, Driscoll-Kraay)
  - Instrumental Variables (Stock & Watson 2018)
  - Smooth IRF via B-splines (Barnichon & Brownlees 2019)
  - State-dependent LP (Auerbach & Gorodnichenko 2013)
  - Propensity Score Matching (Angrist et al. 2018)
  - Structural LP with multi-shock IRFs (Plagborg-Møller & Wolf 2021)
  - Direct multi-step forecasting with analytical/bootstrap CIs
- **Factor Models**
  - Static factors via PCA with Bai-Ng information criteria (IC1, IC2, IC3)
  - Dynamic Factor Models (two-step and EM estimation)
  - Generalized Dynamic Factor Models (spectral methods, Forni et al. 2000)
  - Unified forecasting with theoretical (analytical) and bootstrap confidence intervals for all three factor model types

### Panel Models
- **Panel VAR (PVAR)** - GMM estimation for dynamic panel data:
  - First-difference GMM (Arellano & Bond 1991) and System GMM (Blundell & Bond 1998)
  - Forward orthogonal deviations (Helmert transform) as alternative to first-differencing
  - Fixed-Effects OLS within estimator
  - Windmeijer (2005) finite-sample corrected standard errors for two-step GMM
  - Specification tests: Hansen J-test, Andrews-Lu (2001) MMSC for lag/moment selection
  - Structural analysis: OIRF, GIRF (Pesaran & Shin 1998), FEVD, stability
  - Group-level block bootstrap confidence intervals for IRFs
  - Instrument management: min/max lag truncation, collapse, PCA reduction

### DSGE
- **Model specification** - `@dsge` macro with declarative syntax for parameters, variables, shocks, and equilibrium equations
- **Steady state** - Numerical solver (Newton's method) or analytical closed-form via `steady_state` block in `@dsge` or `ss_fn` kwarg
- **Solution** - Gensys (Sims 2002) and Blanchard-Kahn linearization around steady state
- **Simulation & IRF** - `simulate`, `irf`, `fevd` for solved DSGE models; perfect foresight transition paths
- **Analytical moments** - Discrete Lyapunov equation `solve_lyapunov` for unconditional covariance; `analytical_moments` for theoretical autocovariance structure (Hamilton 1994)
- **Estimation** - Four methods via `estimate_dsge`:
  - IRF matching (minimum distance between model and empirical IRFs)
  - Euler equation GMM (moment conditions from first-order conditions)
  - Simulated Method of Moments (SMM with HAC-optimal weighting)
  - Analytical GMM (Lyapunov-based moments, no simulation noise)

### GMM
- **Generalized Method of Moments** - One-step, two-step, and iterated; Hansen J-test
- **Simulated Method of Moments (SMM)** - Simulation-based estimation with HAC-optimal weighting, two-step and iterated
- **Linear GMM** - Closed-form solver for panel IV estimation
- **Sandwich covariance** - Robust GMM variance with Windmeijer correction

### Structural Identification
- **Cholesky** (recursive) - Wold causal ordering (Sims 1980)
- **Sign restrictions** - Rotation-based identification (Rubio-Ramirez et al. 2010)
- **Narrative restrictions** - Historical event constraints on shocks (Antolin-Diaz & Rubio-Ramirez 2018)
- **Long-run restrictions** - Permanent/transitory decomposition (Blanchard & Quah 1989)
- **Zero and sign restrictions** - Joint zero+sign with importance sampling (Arias et al. 2018)
- **Penalty function** - Frequentist sign and zero restrictions via constrained optimization (Mountford & Uhlig 2009)

### Innovation Accounting
- **Impulse Response Functions (IRF)** - Bootstrap, theoretical, and Bayesian credible intervals
- **Forecast Error Variance Decomposition (FEVD)** - Frequentist and Bayesian
- **LP-FEVD** - R², LP-A, LP-B estimators with bootstrap CIs (Gorodnichenko & Lee 2019)
- **Historical Decomposition (HD)** - Decompose observed movements into structural shock contributions
- **Summary Tables** - Publication-quality output with `report()`, `table()`, `print_table()`
- **Display Backends** - Switch between text, LaTeX, and HTML table output with `set_display_backend()`
- **Bibliographic References** - `refs(model)` outputs AEA-style citations in text, LaTeX, BibTeX, or HTML

### Nowcasting
- **Dynamic Factor Model (DFM)** - EM algorithm with Kalman smoother for mixed-frequency data (Banbura & Modugno 2014)
  - Arbitrary missing patterns and ragged edges
  - Mariano-Murasawa [1 2 3 2 1] temporal aggregation for quarterly variables
  - Block factor structure, AR(1)/IID idiosyncratic components
- **Large Bayesian VAR** - GLP-style Normal-Inverse-Wishart prior (Cimadomo et al. 2022)
  - Hyperparameter optimization via marginal log-likelihood maximization
  - Minnesota shrinkage with sum-of-coefficients and co-persistence priors
- **Bridge Equations** - OLS regressions combining pairs of monthly indicators via median (Banbura et al. 2023)
- **News Decomposition** - Attribute nowcast revisions to individual data releases
- **Panel Balancing** - `balance_panel()` fills NaN in TimeSeriesData/PanelData using DFM imputation

### Statistical Identification via Higher Moments
- **Heteroskedasticity-based** - Markov-switching (Lanne & Lütkepohl 2008), GARCH (Normandin & Phaneuf 2004), smooth-transition (Lütkepohl & Netšunajev 2017), external volatility (Rigobon 2003)
- **Non-Gaussian ICA** - FastICA, JADE, SOBI, distance covariance, HSIC (Hyvärinen et al. 2010, Matteson & Tsay 2017)
- **Non-Gaussian ML** - Student-t, mixture-normal, pseudo-ML, skew-normal (Lanne et al. 2017, Gourieroux et al. 2017)
- **Multivariate normality tests** - Jarque-Bera, Mardia, Doornik-Hansen, Henze-Zirkler
- **Identifiability diagnostics** - Shock gaussianity, independence, identification strength, LR tests
- Seamless integration: `irf(model, 20; method=:fastica)` works out of the box
- See Lewis (2025) for a comprehensive review

### Hypothesis Tests
- **Unit Root Tests** - ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron
- **Cointegration** - Johansen test (trace and max-eigenvalue)
- **VAR** - Granger causality (pairwise and block Wald tests), stationarity diagnostics
- **Panel VAR** - Hansen J-test for overidentifying restrictions, Andrews-Lu MMSC for lag/moment selection
- **Model Comparison** - Likelihood ratio (LR) and Lagrange multiplier (LM/score) tests for nested models
- **Stationarity diagnostics** - `unit_root_summary()`, `test_all_variables()`

### Visualization
- **Interactive D3.js plots** - `plot_result()` renders self-contained HTML with inline D3.js v7 (no additional dependencies)
  - 31 dispatch methods covering IRF, FEVD, historical decomposition, filters, forecasts, volatility models, factor models, data containers, and nowcasting
  - Three chart types: line (with confidence bands), stacked area, and bar charts
  - Interactive tooltips, responsive layout, multi-panel grid figures
  - `save_plot(p, "file.html")` saves to disk; `display_plot(p)` opens in browser; auto-renders in Jupyter
  - Common kwargs: `var`, `shock`, `title`, `save_path`, `ncols`

### Data Management
- **Typed containers** - `TimeSeriesData`, `PanelData`, `CrossSectionData` with variable names, frequency, transformation codes, and descriptions
- **Built-in datasets** - FRED-MD (126 monthly variables), FRED-QD (245 quarterly variables), and Penn World Table (38 OECD countries, 1950–2023 balanced panel)
- **Data diagnostics** - `diagnose()` scans for NaN/Inf/constant columns; `fix()` cleans via listwise deletion, interpolation, or mean imputation
- **FRED transformations** - `apply_tcode()` / `inverse_tcode()` for all 7 FRED transformation codes
- **Filtering** - `apply_filter()` applies HP, Hamilton, BN, BK, or boosted HP per-variable to `TimeSeriesData` and `PanelData`
- **Panel support** - `xtset()` for Stata-style panel construction; balanced/unbalanced detection
- **Summary statistics** - `describe_data()` with per-variable N, mean, std, quantiles, skewness, kurtosis
- **Seamless estimation** - All estimation functions accept `TimeSeriesData` directly

## Installation

```julia
using Pkg
Pkg.add("MacroEconometricModels")
```

## Quick Start

### Time Series Filters

```julia
using MacroEconometricModels

y = cumsum(randn(200))  # simulate I(1) series

# Hodrick-Prescott filter
hp = hp_filter(y; lambda=1600.0)
trend(hp)  # trend component
cycle(hp)  # cyclical component

# Hamilton (2018) regression filter
ham = hamilton_filter(y; h=8, p=4)

# Beveridge-Nelson decomposition
bn = beveridge_nelson(y; p=:auto, q=:auto)

# Baxter-King band-pass filter
bk = baxter_king(y; pl=6, pu=32, K=12)

# Boosted HP (Phillips & Shi 2021)
bhp = boosted_hp(y; lambda=1600.0, stopping=:BIC)
```

### ARIMA Modeling

```julia
using MacroEconometricModels

# Simulate AR(1) data
y = cumsum(randn(200))

# Automatic order selection
auto = auto_arima(y; max_p=5, max_d=2, max_q=5)

# Estimate ARIMA(1,1,1)
model = estimate_arima(y, 1, 1, 1; method=:css_mle)

# Forecast 12 steps ahead
fc = forecast(model, 12)
```

### Volatility Models

```julia
using MacroEconometricModels

y = randn(500)

# GARCH(1,1)
gm = estimate_garch(y, 1, 1)
fc = forecast(gm, 10)

# EGARCH (asymmetric leverage effects)
em = estimate_egarch(y, 1, 1)

# GJR-GARCH (threshold effects)
gjr = estimate_gjr_garch(y, 1, 1)

# Stochastic Volatility (Bayesian MCMC)
sv = estimate_sv(y; variant=:basic, n_samples=1000)
```

### VAR Analysis

```julia
using MacroEconometricModels
using Random

# Generate synthetic VAR(1) data
Random.seed!(42)
T, n = 200, 3
A = [0.5 0.1 0.0; 0.0 0.6 0.1; 0.1 0.0 0.4]
Y = zeros(T, n)
for t in 2:T
    Y[t, :] = A * Y[t-1, :] + randn(n)
end

# Estimate VAR model
model = estimate_var(Y, 2)

# Innovation accounting
irf_result = irf(model, 20; ci_type=:bootstrap)
fevd_result = fevd(model, 20)
hd_result = historical_decomposition(model, 198)

# Publication-quality tables
df = table(irf_result, 1, 1; horizons=[1, 4, 8, 12, 20])
print_table(stdout, fevd_result, 1)

# Bayesian estimation
chain = estimate_bvar(Y, 2; prior=:minnesota)
```

### VECM Analysis

```julia
using MacroEconometricModels
using Random

Random.seed!(42)
# Simulate cointegrated data
T = 200
e = randn(T, 2)
x1 = cumsum(e[:, 1])
x2 = x1 + 0.5 .* e[:, 2]  # cointegrated with x1
Y = hcat(x1, x2)

# Johansen cointegration test
jt = johansen_test(Y, 2)

# Estimate VECM with automatic rank selection
vecm = estimate_vecm(Y, 2; rank=:auto)

# Structural analysis via VAR conversion
irf_result = irf(vecm, 20; method=:cholesky)
fevd_result = fevd(vecm, 20)

# VECM-specific forecasting
fc = forecast(vecm, 12; ci_method=:bootstrap)

# Granger causality (short-run, long-run, strong)
gc = granger_causality_vecm(vecm, 1, 2)
```

### Local Projections

```julia
using MacroEconometricModels

Y = randn(200, 3)
lp_result = estimate_lp(Y, 1, 20; cov_type=:newey_west)
```

```julia
# Structural LP with multi-shock IRFs (Plagborg-Moller & Wolf 2021)
slp = structural_lp(Y, 20; method=:cholesky, lags=4)
lfevd = lp_fevd(slp; estimator=:r2)  # LP-FEVD decomposition
```

### Factor Model Forecasting

```julia
using MacroEconometricModels

X = randn(200, 50)  # 200 observations, 50 variables

# Static factor model with PCA
fm = estimate_factors(X, 3)
fc = forecast(fm, 12; ci_method=:theoretical)
fc.observables       # 12x50 forecasted observables
fc.observables_lower # lower CI bounds
fc.observables_upper # upper CI bounds

# Dynamic factor model with bootstrap CIs
dfm = estimate_dynamic_factors(X, 3, 2)
fc = forecast(dfm, 12; ci_method=:bootstrap, n_boot=500)
```

### Panel VAR

```julia
using MacroEconometricModels, DataFrames, Random

Random.seed!(42)
N, T_total, m = 50, 20, 3
data = zeros(N * T_total, m)
for i in 1:N
    mu_i = randn(m)
    for t in 2:T_total
        idx = (i-1)*T_total + t
        data[idx, :] = mu_i + 0.5 * data[(i-1)*T_total + t - 1, :] + 0.1 * randn(m)
    end
end
df = DataFrame(data, ["y1", "y2", "y3"])
df.id = repeat(1:N, inner=T_total)
df.time = repeat(1:T_total, outer=N)
pd = xtset(df, :id, :time)

# Two-step GMM estimation
model = estimate_pvar(pd, 2; steps=:twostep)

# Structural analysis
irfs = pvar_oirf(model, 10)
fv = pvar_fevd(model, 10)

# Specification tests
j = pvar_hansen_j(model)
stab = pvar_stability(model)
```

### Nowcasting

```julia
using MacroEconometricModels

# Mixed-frequency data: nM monthly + nQ quarterly variables (NaN for missing)
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1)          # DFM (EM + Kalman)
bvar = nowcast_bvar(Y, nM, nQ; lags=5)           # Large BVAR
bridge = nowcast_bridge(Y, nM, nQ)                # Bridge equations

result = nowcast(dfm)                              # Current-quarter nowcast
fc = forecast(dfm, 6)                              # 6-step forecast
news = nowcast_news(X_new, X_old, dfm, T)          # News decomposition
balanced = balance_panel(ts; r=2)                   # Fill NaN via DFM
```

### Statistical Identification

```julia
using MacroEconometricModels

Y = randn(200, 3)
model = estimate_var(Y, 2)

# Test for non-Gaussianity (prerequisite)
suite = normality_test_suite(model)

# Non-Gaussianity: ICA-based identification
ica = identify_fastica(model)
irfs = irf(model, 20; method=:fastica)

# Non-Gaussianity: parametric ML identification
ml = identify_student_t(model)

# Heteroskedasticity: Markov-switching regimes
ms = identify_markov_switching(model; n_regimes=2)
```

### Unit Root Tests

```julia
using MacroEconometricModels

y = cumsum(randn(200))

# Individual tests
adf_test(y; lags=:aic)
kpss_test(y)
pp_test(y)

# Combined summary
unit_root_summary(y)
```

### Granger Causality

```julia
using MacroEconometricModels

Y = randn(200, 3)
m = estimate_var(Y, 2)

# Pairwise: does variable 1 Granger-cause variable 2?
g = granger_test(m, 1, 2)

# Block: do variables 1 and 2 jointly Granger-cause variable 3?
g_block = granger_test(m, [1, 2], 3)

# All pairwise tests at once (n×n p-value matrix)
results = granger_test_all(m)
```

### Data Management

```julia
using MacroEconometricModels

# Load built-in FRED-MD dataset (126 monthly macro variables)
md = load_example(:fred_md)
desc(md)                            # "FRED-MD Monthly Database, January 2026 Vintage..."
vardesc(md, "INDPRO")               # "IP Index"
refs(md)                            # McCracken & Ng (2016)

# Apply recommended FRED transformation codes
md_transformed = apply_tcode(md, md.tcode)

# Per-variable filtering (HP, Hamilton, BN, BK, boosted HP)
cyclical = apply_filter(md_transformed, :hp; component=:cycle)

# Diagnose and clean data
diag = diagnose(md_transformed)
clean = fix(md_transformed; method=:listwise)

# Estimate directly from TimeSeriesData
model = estimate_var(clean[:, ["INDPRO", "UNRATE", "CPIAUCSL"]], 4)

# Load Penn World Table (38 OECD countries, balanced panel)
pwt = load_example(:pwt)
usa = group_data(pwt, "USA")        # extract single country
refs(pwt)                           # Feenstra, Inklaar & Timmer (2015)
```

### Visualization

```julia
using MacroEconometricModels

Y = randn(200, 3)
m = estimate_var(Y, 2)

# Plot IRFs with bootstrap CIs
r = irf(m, 20; ci_type=:bootstrap, reps=500)
p = plot_result(r)                        # Full n×n grid
p = plot_result(r; var=1, shock=1)        # Single panel
save_plot(p, "irf.html")                  # Save to file
display_plot(p)                           # Open in browser

# FEVD (stacked area), HD (stacked bar), filters, forecasts, etc.
plot_result(fevd(m, 20))
plot_result(hp_filter(cumsum(randn(200))))

# Volatility model diagnostics
gm = estimate_garch(randn(500), 1, 1)
plot_result(gm)                           # 3-panel: returns, cond vol, std resid
```

### Bibliographic References

```julia
using MacroEconometricModels

model = estimate_var(randn(100, 3), 2)

# Get references for any model or method
refs(model)                     # AEA plain text
refs(model; format=:bibtex)     # BibTeX entries
refs(:fastica; format=:latex)   # LaTeX \bibitem
refs(:johansen; format=:html)   # HTML with DOI links
```

## Documentation

Full documentation available at [https://FriedmanJP.github.io/MacroEconometricModels.jl/dev/](https://FriedmanJP.github.io/MacroEconometricModels.jl/dev/)

## References

### Time Series Filters

- Baxter, Marianne, and Robert G. King. 1999. "Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series." *Review of Economics and Statistics* 81 (4): 575–593. [https://doi.org/10.1162/003465399558454](https://doi.org/10.1162/003465399558454)
- Beveridge, Stephen, and Charles R. Nelson. 1981. "A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components with Particular Attention to Measurement of the 'Business Cycle'." *Journal of Monetary Economics* 7 (2): 151–174. [https://doi.org/10.1016/0304-3932(81)90040-4](https://doi.org/10.1016/0304-3932(81)90040-4)
- Hamilton, James D. 2018. "Why You Should Never Use the Hodrick-Prescott Filter." *Review of Economics and Statistics* 100 (5): 831–843. [https://doi.org/10.1162/rest_a_00706](https://doi.org/10.1162/rest_a_00706)
- Hodrick, Robert J., and Edward C. Prescott. 1997. "Postwar U.S. Business Cycles: An Empirical Investigation." *Journal of Money, Credit and Banking* 29 (1): 1–16. [https://doi.org/10.2307/2953682](https://doi.org/10.2307/2953682)
- Phillips, Peter C. B., and Zhentao Shi. 2021. "Boosting: Why You Can Use the HP Filter." *International Economic Review* 62 (2): 521–570. [https://doi.org/10.1111/iere.12495](https://doi.org/10.1111/iere.12495)

### ARIMA

- Box, George E. P., and Gwilym M. Jenkins. 1970. *Time Series Analysis: Forecasting and Control*. San Francisco: Holden-Day. ISBN 978-0-8162-1094-7.
- Hyndman, Rob J., and Yeasmin Khandakar. 2008. "Automatic Time Series Forecasting: The forecast Package for R." *Journal of Statistical Software* 27 (3): 1–22. [https://doi.org/10.18637/jss.v027.i03](https://doi.org/10.18637/jss.v027.i03)

### Volatility Models

- Bollerslev, Tim. 1986. "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 31 (3): 307–327. [https://doi.org/10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1)
- Engle, Robert F. 1982. "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica* 50 (4): 987–1007. [https://doi.org/10.2307/1912773](https://doi.org/10.2307/1912773)
- Glosten, Lawrence R., Ravi Jagannathan, and David E. Runkle. 1993. "On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance* 48 (5): 1779–1801. [https://doi.org/10.1111/j.1540-6261.1993.tb05128.x](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)
- Nelson, Daniel B. 1991. "Conditional Heteroskedasticity in Asset Returns: A New Approach." *Econometrica* 59 (2): 347–370. [https://doi.org/10.2307/2938260](https://doi.org/10.2307/2938260)
- Taylor, Stephen J. 1986. *Modelling Financial Time Series*. Chichester: Wiley. ISBN 978-0-471-90993-4.

### VAR and Structural Identification

- Arias, Jonas E., Juan F. Rubio-Ramírez, and Daniel F. Waggoner. 2018. "Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications." *Econometrica* 86 (2): 685–720. [https://doi.org/10.3982/ECTA14468](https://doi.org/10.3982/ECTA14468)
- Antolín-Díaz, Juan, and Juan F. Rubio-Ramírez. 2018. "Narrative Sign Restrictions for SVARs." *American Economic Review* 108 (10): 2802–2829. [https://doi.org/10.1257/aer.20161852](https://doi.org/10.1257/aer.20161852)
- Mountford, Andrew, and Harald Uhlig. 2009. "What Are the Effects of Fiscal Policy Shocks?" *Journal of Applied Econometrics* 24 (6): 960–992. [https://doi.org/10.1002/jae.1079](https://doi.org/10.1002/jae.1079)
- Blanchard, Olivier Jean, and Danny Quah. 1989. "The Dynamic Effects of Aggregate Demand and Supply Disturbances." *American Economic Review* 79 (4): 655–673.
- Kilian, Lutz, and Helmut Lütkepohl. 2017. *Structural Vector Autoregressive Analysis*. Cambridge: Cambridge University Press. [https://doi.org/10.1017/9781108164818](https://doi.org/10.1017/9781108164818)
- Lütkepohl, Helmut. 2005. *New Introduction to Multiple Time Series Analysis*. Berlin: Springer. ISBN 978-3-540-40172-8.
- Rubio-Ramírez, Juan F., Daniel F. Waggoner, and Tao Zha. 2010. "Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference." *Review of Economic Studies* 77 (2): 665–696. [https://doi.org/10.1111/j.1467-937X.2009.00578.x](https://doi.org/10.1111/j.1467-937X.2009.00578.x)
- Sims, Christopher A. 1980. "Macroeconomics and Reality." *Econometrica* 48 (1): 1–48. [https://doi.org/10.2307/1912017](https://doi.org/10.2307/1912017)

### Bayesian Methods

- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2010. "Large Bayesian Vector Auto Regressions." *Journal of Applied Econometrics* 25 (1): 71–92. [https://doi.org/10.1002/jae.1137](https://doi.org/10.1002/jae.1137)
- Giannone, Domenico, Michele Lenza, and Giorgio E. Primiceri. 2015. "Prior Selection for Vector Autoregressions." *Review of Economics and Statistics* 97 (2): 436–451. [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
- Litterman, Robert B. 1986. "Forecasting with Bayesian Vector Autoregressions—Five Years of Experience." *Journal of Business & Economic Statistics* 4 (1): 25–38. [https://doi.org/10.1080/07350015.1986.10509491](https://doi.org/10.1080/07350015.1986.10509491)

### VECM and Cointegration

- Engle, Robert F., and Clive W. J. Granger. 1987. "Co-Integration and Error Correction: Representation, Estimation, and Testing." *Econometrica* 55 (2): 251–276. [https://doi.org/10.2307/1913236](https://doi.org/10.2307/1913236)
- Johansen, Søren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551–1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)

### Local Projections

- Angrist, Joshua D., Òscar Jordà, and Guido M. Kuersteiner. 2018. "Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited." *Journal of Business & Economic Statistics* 36 (3): 371–387. [https://doi.org/10.1080/07350015.2016.1204919](https://doi.org/10.1080/07350015.2016.1204919)
- Auerbach, Alan J., and Yuriy Gorodnichenko. 2013. "Fiscal Multipliers in Recession and Expansion." In *Fiscal Policy after the Financial Crisis*, edited by Alberto Alesina and Francesco Giavazzi, 63–98. Chicago: University of Chicago Press. [https://doi.org/10.7208/chicago/9780226018584.003.0003](https://doi.org/10.7208/chicago/9780226018584.003.0003)
- Barnichon, Regis, and Christian Brownlees. 2019. "Impulse Response Estimation by Smooth Local Projections." *Review of Economics and Statistics* 101 (3): 522–530. [https://doi.org/10.1162/rest_a_00778](https://doi.org/10.1162/rest_a_00778)
- Jordà, Òscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review* 95 (1): 161–182. [https://doi.org/10.1257/0002828053828518](https://doi.org/10.1257/0002828053828518)
- Stock, James H., and Mark W. Watson. 2018. "Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments." *Economic Journal* 128 (610): 917–948. [https://doi.org/10.1111/ecoj.12593](https://doi.org/10.1111/ecoj.12593)

### Factor Models

- Bai, Jushan, and Serena Ng. 2002. "Determining the Number of Factors in Approximate Factor Models." *Econometrica* 70 (1): 191–221. [https://doi.org/10.1111/1468-0262.00273](https://doi.org/10.1111/1468-0262.00273)
- Forni, Mario, Marc Hallin, Marco Lippi, and Lucrezia Reichlin. 2000. "The Generalized Dynamic-Factor Model: Identification and Estimation." *Review of Economics and Statistics* 82 (4): 540–554. [https://doi.org/10.1162/003465300559037](https://doi.org/10.1162/003465300559037)
- Stock, James H., and Mark W. Watson. 2002. "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association* 97 (460): 1167–1179. [https://doi.org/10.1198/016214502388618960](https://doi.org/10.1198/016214502388618960)

### Panel VAR

- Andrews, Donald W. K., and Biao Lu. 2001. "Consistent Model and Moment Selection Procedures for GMM Estimation with Application to Dynamic Panel Data Models." *Journal of Econometrics* 101 (1): 123–164. [https://doi.org/10.1016/S0304-4076(00)00077-4](https://doi.org/10.1016/S0304-4076(00)00077-4)
- Arellano, Manuel, and Stephen Bond. 1991. "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies* 58 (2): 277–297. [https://doi.org/10.2307/2297968](https://doi.org/10.2307/2297968)
- Blundell, Richard, and Stephen Bond. 1998. "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics* 87 (1): 115–143. [https://doi.org/10.1016/S0304-4076(98)00009-8](https://doi.org/10.1016/S0304-4076(98)00009-8)
- Holtz-Eakin, Douglas, Whitney Newey, and Harvey S. Rosen. 1988. "Estimating Vector Autoregressions with Panel Data." *Econometrica* 56 (6): 1371–1395. [https://doi.org/10.2307/1913103](https://doi.org/10.2307/1913103)
- Windmeijer, Frank. 2005. "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics* 126 (1): 25–51. [https://doi.org/10.1016/j.jeconom.2004.02.005](https://doi.org/10.1016/j.jeconom.2004.02.005)

### DSGE

- Blanchard, Olivier Jean, and Charles M. Kahn. 1980. "The Solution of Linear Difference Models Under Rational Expectations." *Econometrica* 48 (5): 1305–1311. [https://doi.org/10.2307/1912186](https://doi.org/10.2307/1912186)
- Hamilton, James D. 1994. *Time Series Analysis*. Princeton: Princeton University Press. ISBN 978-0-691-04289-3.
- Sims, Christopher A. 2002. "Solving Linear Rational Expectations Models." *Computational Economics* 20 (1): 1–20. [https://doi.org/10.1023/A:1020517101123](https://doi.org/10.1023/A:1020517101123)

### GMM and Covariance Estimation

- Hansen, Lars Peter. 1982. "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica* 50 (4): 1029–1054. [https://doi.org/10.2307/1912775](https://doi.org/10.2307/1912775)
- Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55 (3): 703–708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)

### Statistical Identification via Higher Moments

- Lewis, Daniel J. 2025. "Identification Based on Higher Moments in Macroeconometrics." *Annual Review of Economics* 17: 665–693. [https://doi.org/10.1146/annurev-economics-070124-051419](https://doi.org/10.1146/annurev-economics-070124-051419)
- Lewis, Daniel J. 2021. "Identifying Shocks via Time-Varying Volatility." *Review of Economic Studies* 88 (6): 3086–3124. [https://doi.org/10.1093/restud/rdab009](https://doi.org/10.1093/restud/rdab009)
- Gourieroux, Christian, Alain Monfort, and Jean-Paul Renne. 2017. "Statistical Inference for Independent Component Analysis: Application to Structural VAR Models." *Journal of Econometrics* 196 (1): 111–126. [https://doi.org/10.1016/j.jeconom.2016.09.007](https://doi.org/10.1016/j.jeconom.2016.09.007)
- Hyvärinen, Aapo. 1999. "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis." *IEEE Transactions on Neural Networks* 10 (3): 626–634. [https://doi.org/10.1109/72.761722](https://doi.org/10.1109/72.761722)
- Keweloh, Sascha A. 2021. "A Generalized Method of Moments Estimator for Structural Vector Autoregressions Based on Higher Moments." *Journal of Business & Economic Statistics* 39 (3): 772–882. [https://doi.org/10.1080/07350015.2020.1730858](https://doi.org/10.1080/07350015.2020.1730858)
- Lanne, Markku, Mika Meitz, and Pentti Saikkonen. 2017. "Identification and Estimation of Non-Gaussian Structural Vector Autoregressions." *Journal of Econometrics* 196 (2): 288–304. [https://doi.org/10.1016/j.jeconom.2016.06.002](https://doi.org/10.1016/j.jeconom.2016.06.002)
- Lanne, Markku, and Helmut Lütkepohl. 2010. "Structural Vector Autoregressions with Nonnormal Residuals." *Journal of Business & Economic Statistics* 28 (1): 159–168. [https://doi.org/10.1198/jbes.2009.06003](https://doi.org/10.1198/jbes.2009.06003)
- Rigobon, Roberto. 2003. "Identification through Heteroskedasticity." *Review of Economics and Statistics* 85 (4): 777–792. [https://doi.org/10.1162/003465303772815727](https://doi.org/10.1162/003465303772815727)

### Unit Root and Cointegration Tests

- Dickey, David A., and Wayne A. Fuller. 1979. "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association* 74 (366): 427–431. [https://doi.org/10.1080/01621459.1979.10482531](https://doi.org/10.1080/01621459.1979.10482531)
- Johansen, Søren. 1991. "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." *Econometrica* 59 (6): 1551–1580. [https://doi.org/10.2307/2938278](https://doi.org/10.2307/2938278)
- Kwiatkowski, Denis, Peter C. B. Phillips, Peter Schmidt, and Yongcheol Shin. 1992. "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics* 54 (1–3): 159–178. [https://doi.org/10.1016/0304-4076(92)90104-Y](https://doi.org/10.1016/0304-4076(92)90104-Y)
- Ng, Serena, and Pierre Perron. 2001. "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica* 69 (6): 1519–1554. [https://doi.org/10.1111/1468-0262.00256](https://doi.org/10.1111/1468-0262.00256)

### Granger Causality

- Granger, Clive W. J. 1969. "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica* 37 (3): 424–438. [https://doi.org/10.2307/1912791](https://doi.org/10.2307/1912791)

### Data Sources

- Feenstra, Robert C., Robert Inklaar, and Marcel P. Timmer. 2015. "The Next Generation of the Penn World Table." *American Economic Review* 105 (10): 3150–3182. [https://doi.org/10.1257/aer.20130954](https://doi.org/10.1257/aer.20130954)
- McCracken, Michael W., and Serena Ng. 2016. "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics* 34 (4): 574–589. [https://doi.org/10.1080/07350015.2015.1086655](https://doi.org/10.1080/07350015.2015.1086655)
- McCracken, Michael W., and Serena Ng. 2020. "FRED-QD: A Quarterly Database for Macroeconomic Research." *Federal Reserve Bank of St. Louis Working Paper* 2020-005. [https://doi.org/10.20955/wp.2020.005](https://doi.org/10.20955/wp.2020.005)

### Nowcasting

- Bańbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29 (1): 133–160. [https://doi.org/10.1002/jae.2306](https://doi.org/10.1002/jae.2306)
- Cimadomo, Jacopo, Domenico Giannone, Michele Lenza, Francesca Monti, and Andrej Sokol. 2022. "Nowcasting with Large Bayesian Vector Autoregressions." *ECB Working Paper* No. 2696.
- Bańbura, Marta, Irina Belousova, Katalin Bodnár, and Máté Barnabás Tóth. 2023. "Nowcasting Employment in the Euro Area." *ECB Working Paper* No. 2815.

### Model Comparison Tests

- Neyman, Jerzy, and Egon S. Pearson. 1933. "On the Problem of the Most Efficient Tests of Statistical Hypotheses." *Philosophical Transactions of the Royal Society A* 231 (694–706): 289–337. [https://doi.org/10.1098/rsta.1933.0009](https://doi.org/10.1098/rsta.1933.0009)
- Rao, C. Radhakrishna. 1948. "Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation." *Mathematical Proceedings of the Cambridge Philosophical Society* 44 (1): 50–57. [https://doi.org/10.1017/S0305004100023987](https://doi.org/10.1017/S0305004100023987)
- Silvey, S. D. 1959. "The Lagrangian Multiplier Test." *Annals of Mathematical Statistics* 30 (2): 389–407. [https://doi.org/10.1214/aoms/1177706259](https://doi.org/10.1214/aoms/1177706259)
- Wilks, Samuel S. 1938. "The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses." *Annals of Mathematical Statistics* 9 (1): 60–62. [https://doi.org/10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360)

## License

GPL-3.0-or-later
