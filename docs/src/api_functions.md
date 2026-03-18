# [API Functions](@id api_functions)

This page documents all functions in **MacroEconometricModels.jl**, organized by module.

---

## Data Management

### Validation and Cleaning

```@docs
diagnose
fix
validate_for_model
```

### FRED Transformations

```@docs
apply_tcode
inverse_tcode
```

### Filtering

```@docs
apply_filter
```

### Summary Statistics

```@docs
describe_data
```

### Panel Data

```@docs
xtset
isbalanced
groups
ngroups
group_data
panel_summary
```

### Data Accessors and Conversion

```@docs
MacroEconometricModels.StatsAPI.nobs(::TimeSeriesData)
nvars
varnames
frequency
time_index
obs_id
desc
vardesc
rename_vars!
set_time_index!
set_obs_id!
set_desc!
set_vardesc!
dates
set_dates!
to_matrix
to_vector
load_example
```

---

## Time Series Filters

```@docs
hp_filter
hamilton_filter
beveridge_nelson
baxter_king
boosted_hp
trend
cycle
```

---

## ARIMA Models

### Estimation

```@docs
estimate_ar
estimate_ma
estimate_arma
estimate_arima
```

### Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["arima/forecast.jl"]
Order   = [:function]
```

### Order Selection

```@docs
select_arima_order
auto_arima
ic_table
```

### ARIMA Accessors

```@docs
ar_order
ma_order
diff_order
```

### ARIMA StatsAPI Interface

```@docs
MacroEconometricModels.StatsAPI.stderror(::ARModel)
MacroEconometricModels.StatsAPI.stderror(::MAModel)
MacroEconometricModels.StatsAPI.stderror(::ARMAModel)
MacroEconometricModels.StatsAPI.stderror(::ARIMAModel)
```

---

## Cross-Sectional Models

### Estimation

```@docs
estimate_reg
estimate_iv
estimate_logit
estimate_probit
```

### Marginal Effects and Odds Ratios

```@docs
marginal_effects
odds_ratio
```

### Diagnostics

```@docs
vif
classification_table
```

---

## Ordered and Multinomial Models

```@docs
estimate_ologit
estimate_oprobit
estimate_mlogit
brant_test
hausman_iia
```

---

## VAR Estimation

### Frequentist Estimation

```@docs
estimate_var
select_lag_order
MacroEconometricModels.StatsAPI.vcov(::VARModel)
MacroEconometricModels.StatsAPI.predict
MacroEconometricModels.StatsAPI.r2(::VARModel)
MacroEconometricModels.StatsAPI.loglikelihood
MacroEconometricModels.StatsAPI.confint(::VARModel)
```

### Bayesian Estimation

```@docs
estimate_bvar
posterior_mean_model
posterior_median_model
```

### VAR/BVAR Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["var/estimation.jl", "bvar/estimation.jl"]
Order   = [:function]
```

### Forecast Accessors

```@docs
point_forecast
forecast_horizon
lower_bound
upper_bound
```

### Prior Specification

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["bvar/priors.jl"]
Order   = [:function]
```

### VECM Estimation

```@docs
estimate_vecm
to_var
select_vecm_rank
cointegrating_rank
granger_causality_vecm
```

### VECM Analysis and Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["vecm/analysis.jl", "vecm/forecast.jl"]
Order   = [:function]
```

---

## Structural Identification

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/identification.jl"]
Order   = [:function]
```

### Arias et al. (2018) Sign/Zero Restrictions

```@docs
identify_arias
identify_arias_bayesian
zero_restriction
sign_restriction
```

### Sign-Identified Set

```@docs
irf_bounds
irf_median
irf_mean
irf_percentiles
```

### Mountford-Uhlig (2009) Penalty Function

```@docs
identify_uhlig
```

---

## Innovation Accounting

### Impulse Response Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/irf.jl"]
Order   = [:function]
```

### Forecast Error Variance Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/fevd.jl"]
Order   = [:function]
```

### Historical Decomposition

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/hd.jl"]
Order   = [:function]
```

### Summary Tables

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["summary.jl"]
Order   = [:function]
```

---

## Local Projections

### Core LP Estimation and Covariance

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/core.jl"]
Order   = [:function]
```

### LP-IV (Stock & Watson 2018)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/iv.jl"]
Order   = [:function]
```

### Smooth LP (Barnichon & Brownlees 2019)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/smooth.jl"]
Order   = [:function]
```

### State-Dependent LP (Auerbach & Gorodnichenko 2013)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/state.jl"]
Order   = [:function]
```

### Propensity Score LP (Angrist et al. 2018)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/propensity.jl"]
Order   = [:function]
```

### LP Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/forecast.jl"]
Order   = [:function]
```

### LP-FEVD (Gorodnichenko & Lee 2019)

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["lp/fevd.jl"]
Order   = [:function]
```

---

## Factor Models

### Static Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factor/static.jl"]
Order   = [:function]
```

### Dynamic Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factor/dynamic.jl"]
Order   = [:function]
```

### Generalized Dynamic Factor Model

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["factor/generalized.jl"]
Order   = [:function]
```

---

## Panel VAR

### Estimation

```@docs
estimate_pvar
estimate_pvar_feols
```

### Structural Analysis

```@docs
pvar_oirf
pvar_girf
pvar_fevd
pvar_stability
```

### Bootstrap

```@docs
pvar_bootstrap_irf
```

### Specification Tests

```@docs
pvar_hansen_j
pvar_mmsc
pvar_lag_selection
```

### GMM Utilities

```@docs
linear_gmm_solve
gmm_sandwich_vcov
andrews_lu_mmsc
```

---

## Difference-in-Differences

### Estimation

```@docs
estimate_did
estimate_event_study_lp
estimate_lp_did
```

### Diagnostics

```@docs
bacon_decomposition
pretrend_test
negative_weight_check
```

### Sensitivity Analysis

```@docs
honest_did
```

---

## GMM Estimation

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["gmm/gmm.jl"]
Order   = [:function]
```

### Simulated Method of Moments

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["gmm/smm.jl"]
Order   = [:function]
```

### Parameter Transforms

```@docs
to_unconstrained
to_constrained
transform_jacobian
```

---

## Unit Root and Cointegration Tests

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["teststat/adf.jl", "teststat/kpss.jl", "teststat/pp.jl", "teststat/za.jl", "teststat/ngperron.jl", "teststat/johansen.jl", "teststat/fourier.jl", "teststat/dfgls.jl", "teststat/lm_unitroot.jl", "teststat/adf_2break.jl", "teststat/gregory_hansen.jl", "teststat/stationarity.jl", "teststat/convenience.jl"]
Order   = [:function]
```

---

## Model Comparison Tests

```@docs
lr_test
lm_test
```

---

## Granger Causality Tests

```@docs
granger_test
granger_test_all
```

---

## Volatility Models

### ARCH Estimation and Diagnostics

```@docs
estimate_arch
arch_lm_test
ljung_box_squared
```

### GARCH Estimation and Diagnostics

```@docs
estimate_garch
estimate_egarch
estimate_gjr_garch
news_impact_curve
```

### Stochastic Volatility

```@docs
estimate_sv
```

### Volatility Forecasting

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["arch/forecast.jl", "garch/forecast.jl", "sv/forecast.jl"]
Order   = [:function]
```

### Volatility Accessors

```@docs
persistence
halflife
unconditional_variance
arch_order
garch_order
```

### Volatility StatsAPI Interface

```@docs
MacroEconometricModels.StatsAPI.nobs(::ARCHModel)
MacroEconometricModels.StatsAPI.coef(::ARCHModel)
MacroEconometricModels.StatsAPI.residuals(::ARCHModel)
MacroEconometricModels.StatsAPI.aic(::ARCHModel)
MacroEconometricModels.StatsAPI.bic(::ARCHModel)
MacroEconometricModels.StatsAPI.dof(::ARCHModel)
MacroEconometricModels.StatsAPI.islinear(::ARCHModel)
MacroEconometricModels.StatsAPI.nobs(::GARCHModel)
MacroEconometricModels.StatsAPI.coef(::GARCHModel)
MacroEconometricModels.StatsAPI.residuals(::GARCHModel)
MacroEconometricModels.StatsAPI.aic(::GARCHModel)
MacroEconometricModels.StatsAPI.bic(::GARCHModel)
MacroEconometricModels.StatsAPI.dof(::GARCHModel)
MacroEconometricModels.StatsAPI.islinear(::GARCHModel)
MacroEconometricModels.StatsAPI.nobs(::EGARCHModel)
MacroEconometricModels.StatsAPI.coef(::EGARCHModel)
MacroEconometricModels.StatsAPI.residuals(::EGARCHModel)
MacroEconometricModels.StatsAPI.aic(::EGARCHModel)
MacroEconometricModels.StatsAPI.bic(::EGARCHModel)
MacroEconometricModels.StatsAPI.dof(::EGARCHModel)
MacroEconometricModels.StatsAPI.islinear(::EGARCHModel)
MacroEconometricModels.StatsAPI.nobs(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.coef(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.residuals(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.aic(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.bic(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.dof(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.islinear(::GJRGARCHModel)
MacroEconometricModels.StatsAPI.nobs(::SVModel)
MacroEconometricModels.StatsAPI.coef(::SVModel)
MacroEconometricModels.StatsAPI.residuals(::SVModel)
MacroEconometricModels.StatsAPI.islinear(::SVModel)
MacroEconometricModels.StatsAPI.stderror(m::GJRGARCHModel)
MacroEconometricModels.StatsAPI.confint(m::AbstractVolatilityModel)
```

---

## Nowcasting

### Estimation

```@docs
nowcast_dfm
nowcast_bvar
nowcast_bridge
```

### Nowcast and Forecast

```@docs
nowcast
```

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["nowcast/forecast.jl"]
Order   = [:function]
```

### News Decomposition

```@docs
nowcast_news
```

### Panel Balancing

```@docs
balance_panel
```

### Nowcast Display

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["summary_nowcast.jl"]
Order   = [:function]
```

---

## DSGE Models

### Specification

```@docs
MacroEconometricModels.@dsge
```

### Steady State

```@docs
compute_steady_state
linearize
```

### Solution Methods

```@docs
solve
gensys
blanchard_kahn
klein
perturbation_solver
MacroEconometricModels.collocation_solver
MacroEconometricModels.pfi_solver
MacroEconometricModels.perfect_foresight
evaluate_policy
max_euler_error
```

### DSGE IRF and FEVD

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["dsge/simulation.jl", "dsge/pruning.jl"]
Order   = [:function]
```

### Simulation and Analysis

```@docs
simulate
solve_lyapunov
analytical_moments
```

### Global Solution Methods

```@docs
MacroEconometricModels.vfi_solver
```

### DSGE GMM Estimation

```@docs
estimate_dsge
```

### DSGE Bayesian Estimation

```@docs
estimate_dsge_bayes
posterior_summary
marginal_likelihood
bayes_factor
prior_posterior_table
posterior_predictive
```

### Occasionally Binding Constraints

```@docs
parse_constraint
occbin_solve
occbin_irf
```

### Constraint Constructors

```@docs
variable_bound
nonlinear_constraint
```

---

## Display and References

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/display.jl"]
Order   = [:function]
```

```@docs
refs
```

### Output Tables

```@docs
table
print_table
```

---

## Non-Gaussian Identification

### Normality Tests

```@docs
jarque_bera_test
mardia_test
doornik_hansen_test
henze_zirkler_test
normality_test_suite
```

### ICA-based Identification

```@docs
identify_fastica
identify_jade
identify_sobi
identify_dcov
identify_hsic
```

### Non-Gaussian ML Identification

```@docs
identify_student_t
identify_mixture_normal
identify_pml
identify_skew_normal
identify_nongaussian_ml
```

### Heteroskedasticity Identification

```@docs
identify_markov_switching
identify_garch
identify_smooth_transition
identify_external_volatility
```

### Identifiability Tests

```@docs
test_identification_strength
test_shock_gaussianity
test_gaussian_vs_nongaussian
test_shock_independence
test_overidentification
```

---

## Covariance Estimators

```@docs
newey_west
white_vcov
driscoll_kraay
robust_vcov
long_run_variance
long_run_covariance
optimal_bandwidth_nw
register_cov_estimator!
```

---

## Plotting

### Core Plot Functions

```@docs
save_plot
display_plot
```

### Plot Dispatches

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["plotting/irf.jl", "plotting/fevd.jl", "plotting/hd.jl", "plotting/filters.jl", "plotting/forecast.jl", "plotting/models.jl", "plotting/nowcast.jl", "plotting/did.jl", "plotting/reg.jl"]
Order   = [:function]
```

---

## Utility Functions

```@autodocs
Modules = [MacroEconometricModels]
Pages   = ["core/utils.jl"]
Order   = [:function]
```

---

## Panel Regression

### Panel Linear Models

```@docs
estimate_xtreg
```

### Panel Instrumental Variables

```@docs
estimate_xtiv
```

### Panel Discrete Choice

```@docs
estimate_xtlogit
estimate_xtprobit
```

### Panel Specification Tests

```@docs
hausman_test
breusch_pagan_test
pesaran_cd_test
wooldridge_ar_test
modified_wald_test
f_test_fe
```

---

## Spectral Analysis

### Spectral Estimation

```@docs
periodogram
spectral_density
cross_spectrum
```

### Autocorrelation Functions

```@docs
acf
pacf
ccf
acf_pacf
```

### Spectral Diagnostics

```@docs
coherence
phase
gain
band_power
ideal_bandpass
transfer_function
```

---

## Portmanteau and Serial Correlation Tests

```@docs
ljung_box_test
box_pierce_test
durbin_watson_test
bartlett_white_noise_test
fisher_test
```

---

## Structural Break Tests

```@docs
andrews_test
bai_perron_test
factor_break_test
```

---

## Panel Unit Root Tests

```@docs
panic_test
pesaran_cips_test
moon_perron_test
panel_unit_root_summary
```

---

## FAVAR

```@docs
estimate_favar
favar_panel_irf
favar_panel_forecast
```

---

## Structural Dynamic Factor Models

```@docs
estimate_structural_dfm
sdfm_panel_irf
```

---

## Panel Data Utilities

```@docs
panel_lag
panel_lead
panel_diff
add_panel_lag
add_panel_lead
add_panel_diff
```
