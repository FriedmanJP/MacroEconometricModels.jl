# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
    MacroEconometricModels

A Julia package for macroeconomic time series analysis, providing tools for:
- Vector Autoregression (VAR) estimation
- Bayesian VAR (BVAR) with Minnesota priors
- Structural identification (Cholesky, sign restrictions, narrative, long-run)
- Impulse Response Functions (IRF)
- Forecast Error Variance Decomposition (FEVD)
- Factor models via Principal Component Analysis
- Local Projections (LP) with various extensions:
  - HAC standard errors (Jordà 2005)
  - Instrumental Variables (Stock & Watson 2018)
  - Smooth IRF via B-splines (Barnichon & Brownlees 2019)
  - State-dependent LP (Auerbach & Gorodnichenko 2013)
  - Propensity Score Matching (Angrist et al. 2018)
- ARIMA/ARMA model estimation, forecasting, and order selection
- Generalized Method of Moments (GMM) estimation

# Quick Start
```julia
using MacroEconometricModels

# Estimate a VAR model
Y = randn(100, 3)
model = estimate_var(Y, 2)

# Compute IRFs with bootstrap confidence intervals
irf_result = irf(model, 20; ci_type=:bootstrap)

# Local Projection IRFs with HAC standard errors
lp_result = estimate_lp(Y, 1, 20; cov_type=:newey_west)
lp_irf_result = lp_irf(lp_result)

# Bayesian estimation
post = estimate_bvar(Y, 2; prior=:minnesota)
```

# References
- Bańbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector auto regressions.
- Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
- Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural vector autoregressions.
- Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections.
- Stock, J. H., & Watson, M. W. (2018). Identification and Estimation of Dynamic Causal Effects.
- Barnichon, R., & Brownlees, C. (2019). Impulse Response Estimation by Smooth Local Projections.
- Auerbach, A. J., & Gorodnichenko, Y. (2013). Fiscal Multipliers in Recession and Expansion.
- Angrist, J. D., Jordà, Ò., & Kuersteiner, G. M. (2018). Semiparametric Estimates of Monetary Policy Effects.
"""
module MacroEconometricModels

# =============================================================================
# Dependencies
# =============================================================================

using LinearAlgebra
using Random
using Statistics
using DataFrames
using StatsAPI
using StatsAPI: fit, coef, vcov, residuals, predict, r2, aic, bic, dof, dof_residual, nobs, loglikelihood, confint, stderror, islinear
using Distributions
using SparseArrays
import Optim

# =============================================================================
# Include Source Files (Order Matters)
# =============================================================================

# Core infrastructure
include("core/utils.jl")
include("core/types.jl")
include("core/display.jl")
include("core/kalman.jl")

# Data containers, validation, transforms, panel, summary stats, examples
include("data/types.jl")
include("data/validation.jl")
include("data/transform.jl")
include("data/panel.jl")
include("data/summary_stats.jl")
include("data/examples.jl")

# VAR types and estimation
include("var/types.jl")
include("var/estimation.jl")

# Bayesian estimation
include("bvar/types.jl")
include("bvar/priors.jl")
include("bvar/estimation.jl")

# Statistical tests
include("teststat/types.jl")
include("teststat/critical_values.jl")
include("teststat/helpers.jl")
include("teststat/adf.jl")
include("teststat/kpss.jl")
include("teststat/pp.jl")
include("teststat/za.jl")
include("teststat/ngperron.jl")
include("teststat/johansen.jl")

# VECM types and estimation (after johansen.jl for JohansenResult)
include("vecm/types.jl")
include("vecm/estimation.jl")

include("teststat/stationarity.jl")
include("teststat/convenience.jl")
include("teststat/show.jl")
include("teststat/normality.jl")

# Structural identification
include("var/identification.jl")
include("var/uhlig.jl")

# Non-Gaussian identification (shared helpers first)
include("nongaussian/shared.jl")
include("nongaussian/ica.jl")
include("nongaussian/ml.jl")
include("nongaussian/heteroskedastic.jl")
include("nongaussian/tests.jl")

# Bayesian utilities (after bayesian + identification)
include("bvar/utils.jl")

# Factor models
include("factor/kalman.jl")
include("factor/static.jl")
include("factor/dynamic.jl")
include("factor/generalized.jl")

# Nowcasting (after factor models for Kalman filter reuse)
include("nowcast/types.jl")
include("nowcast/kalman_missing.jl")
include("nowcast/dfm.jl")
include("nowcast/bvar_nowcast.jl")
include("nowcast/bridge.jl")
include("nowcast/news.jl")
include("nowcast/forecast.jl")

# DSGE models
include("dsge/types.jl")
include("dsge/display.jl")
include("dsge/parser.jl")
include("dsge/steady_state.jl")
include("dsge/linearize.jl")
include("dsge/gensys.jl")
include("dsge/blanchard_kahn.jl")
include("dsge/klein.jl")
include("dsge/perfect_foresight.jl")
include("dsge/occbin.jl")

# GMM
include("gmm/transforms.jl")
include("gmm/gmm.jl")
include("gmm/smm.jl")

# ARIMA
include("arima/types.jl")
include("arima/kalman.jl")
include("arima/estimation.jl")
include("arima/forecast.jl")
include("arima/selection.jl")

# Time Series Filters (after ARIMA for beveridge_nelson, after teststat for boosted_hp ADF)
include("filters/types.jl")
include("filters/hp.jl")
include("filters/hamilton.jl")
include("filters/beveridge_nelson.jl")
include("filters/baxter_king.jl")
include("filters/boosted_hp.jl")

# ARCH models
include("arch/types.jl")
include("arch/estimation.jl")
include("arch/forecast.jl")
include("teststat/arch_diagnostics.jl")

# GARCH models
include("garch/types.jl")
include("garch/estimation.jl")
include("garch/forecast.jl")
include("garch/diagnostics.jl")

# Stochastic Volatility models
include("sv/types.jl")
include("sv/estimation.jl")
include("sv/forecast.jl")

# Model comparison tests (LR, LM)
include("teststat/model_comparison.jl")

# Granger causality tests
include("teststat/granger.jl")

# Panel VAR
include("pvar/types.jl")
include("pvar/transforms.jl")
include("pvar/instruments.jl")
include("pvar/estimation.jl")
include("pvar/analysis.jl")
include("pvar/bootstrap.jl")
include("teststat/pvar_hansen_j.jl")
include("teststat/pvar_mmsc.jl")
include("teststat/pvar_lag_selection.jl")

# Covariance estimators
include("core/covariance.jl")

# Local Projections
include("lp/types.jl")
include("lp/core.jl")
include("lp/iv.jl")
include("lp/smooth.jl")
include("lp/state.jl")
include("lp/propensity.jl")
include("lp/forecast.jl")

# Innovation accounting (after LP types for lp_irf support)
include("var/irf.jl")
include("var/fevd.jl")
include("var/hd.jl")

# VECM analysis and forecasting (after irf + fevd + hd)
include("vecm/forecast.jl")
include("vecm/analysis.jl")
include("vecm/granger.jl")

# LP-FEVD (after irf + fevd)
include("lp/fevd.jl")

# DSGE simulation, IRF/FEVD bridge (after var/irf.jl, var/fevd.jl, lp/fevd.jl)
include("dsge/simulation.jl")
include("dsge/analytical.jl")

# DSGE estimation (after estimate_var, irf, estimate_gmm)
include("dsge/estimation.jl")

# Display (after all types)
include("summary.jl")

# Data conversion and estimation dispatch wrappers (after all estimation functions)
include("data/convert.jl")

# Data filtering (after convert.jl, needs to_vector + all filter functions + group_data)
include("data/filter.jl")

# Plotting (after all types — inline D3.js rendering, no extra dependencies)
include("plotting/types.jl")
include("plotting/render.jl")
include("plotting/helpers.jl")
include("plotting/irf.jl")
include("plotting/fevd.jl")
include("plotting/hd.jl")
include("plotting/filters.jl")
include("plotting/forecast.jl")
include("plotting/models.jl")
include("plotting/nowcast.jl")

# =============================================================================
# Exports - Types
# =============================================================================

# Abstract types
export AbstractAnalysisResult, AbstractFrequentistResult, AbstractBayesianResult
export AbstractVARModel, AbstractImpulseResponse, AbstractFEVD, AbstractPrior
export AbstractForecastResult
export point_forecast, lower_bound, upper_bound, forecast_horizon

# VAR types
export VARModel, VARForecast

# IRF types
export ImpulseResponse, BayesianImpulseResponse

# FEVD types
export FEVD, BayesianFEVD, LPFEVD

# Prior types
export MinnesotaHyperparameters

# Factor model types
export AbstractFactorModel, FactorModel, DynamicFactorModel, GeneralizedDynamicFactorModel, FactorForecast

# Nowcasting types
export AbstractNowcastModel
export NowcastDFM, NowcastBVAR, NowcastBridge, NowcastResult, NowcastNews

# Nowcasting estimation
export nowcast_dfm, nowcast_bvar, nowcast_bridge

# News decomposition
export nowcast_news

# Nowcast/forecast dispatch
export nowcast

# Panel balancing
export balance_panel

# =============================================================================
# Exports - DSGE Models
# =============================================================================

# Abstract type
export AbstractDSGEModel

# Types
export DSGESpec, LinearDSGE, DSGESolution, PerfectForesightPath, DSGEEstimation

# Macro
export @dsge

# Functions
export compute_steady_state, linearize, solve, gensys, blanchard_kahn, klein
export simulate, estimate_dsge
export solve_lyapunov, analytical_moments

# Accessors
export nshocks, is_determined, is_stable

# OccBin types and functions
export OccBinConstraint, OccBinRegime, OccBinSolution, OccBinIRF
export parse_constraint, occbin_solve, occbin_irf

# Local Projection types
export AbstractLPModel, AbstractLPImpulseResponse, AbstractCovarianceEstimator
export LPModel, LPImpulseResponse, LPIVModel, SmoothLPModel, StateLPModel, PropensityLPModel
export StructuralLP, LPForecast
export NeweyWestEstimator, WhiteEstimator, DriscollKraayEstimator
export BSplineBasis, StateTransition, PropensityScoreConfig

# GMM types
export AbstractGMMModel, GMMModel, GMMWeighting

# =============================================================================
# Exports - Unit Root Tests
# =============================================================================

# Abstract type
export AbstractUnitRootTest

# Result types
export ADFResult, KPSSResult, PPResult, ZAResult, NgPerronResult
export JohansenResult, VARStationarityResult

# Univariate unit root tests
export adf_test, kpss_test, pp_test, za_test, ngperron_test

# Multivariate cointegration test
export johansen_test

# Convenience functions
export unit_root_summary, test_all_variables

# Model comparison tests
export LRTestResult, LMTestResult
export lr_test, lm_test

# Granger causality tests
export GrangerCausalityResult
export granger_test, granger_test_all

# VECM types
export VECMModel, VECMForecast, VECMGrangerResult

# =============================================================================
# Exports - VAR Estimation
# =============================================================================

export estimate_var
export select_lag_order

# =============================================================================
# Exports - VECM Estimation
# =============================================================================

export estimate_vecm, to_var, select_vecm_rank
export cointegrating_rank, granger_causality_vecm

# =============================================================================
# Exports - Bayesian Estimation
# =============================================================================

export BVARPosterior, BVARForecast
export estimate_bvar
export posterior_mean_model
export posterior_median_model

# =============================================================================
# Exports - Prior Functions
# =============================================================================

export gen_dummy_obs
export log_marginal_likelihood
export optimize_hyperparameters
export optimize_hyperparameters_full

# =============================================================================
# Exports - Structural Identification
# =============================================================================

export identify_cholesky
export identify_sign
export identify_narrative
export identify_long_run
export generate_Q
export compute_irf, compute_structural_shocks

# Arias et al. (2018) SVAR identification
export ZeroRestriction, SignRestriction, SVARRestrictions, AriasSVARResult
export identify_arias, identify_arias_bayesian
export zero_restriction, sign_restriction
export irf_percentiles, irf_mean

# Sign restriction identified set (Baumeister & Hamilton 2015)
export SignIdentifiedSet, irf_bounds, irf_median

# Mountford & Uhlig (2009) penalty function identification
export UhligSVARResult, identify_uhlig

# =============================================================================
# Exports - IRF and FEVD
# =============================================================================

export irf
export fevd
export lp_fevd

# =============================================================================
# Exports - Historical Decomposition
# =============================================================================

export AbstractHistoricalDecomposition
export HistoricalDecomposition, BayesianHistoricalDecomposition
export historical_decomposition
export contribution, total_shock_contribution, verify_decomposition

# =============================================================================
# Exports - Report, Tables, and Result Interface
# =============================================================================

export report, refs
export table, print_table
export point_estimate, has_uncertainty, uncertainty_bounds
export set_display_backend, get_display_backend

# =============================================================================
# Exports - Factor Models
# =============================================================================

export estimate_factors
export ic_criteria
export scree_plot_data

# Dynamic Factor Model functions
export estimate_dynamic_factors
export ic_criteria_dynamic
export forecast
export companion_matrix_factors
export is_stationary

# Generalized Dynamic Factor Model functions
export estimate_gdfm
export ic_criteria_gdfm
export common_variance_share
export spectral_eigenvalue_plot_data

# =============================================================================
# Exports - Utility Functions
# =============================================================================

export robust_inv
export safe_cholesky
export construct_var_matrices
export extract_ar_coefficients
export companion_matrix

# Type accessor functions
export nvars, nlags, ncoefs, effective_nobs

# =============================================================================
# Exports - Local Projections
# =============================================================================

# Core LP estimation (Jordà 2005)
export estimate_lp, lp_irf, cumulative_irf
export estimate_lp_multi, estimate_lp_cholesky, compare_var_lp
export structural_lp

# HAC covariance estimators
export newey_west, white_vcov, driscoll_kraay, optimal_bandwidth_nw
export robust_vcov, long_run_variance, long_run_covariance
export register_cov_estimator!

# LP-IV (Stock & Watson 2018)
export estimate_lp_iv, lp_iv_irf
export weak_instrument_test, sargan_test

# Smooth LP (Barnichon & Brownlees 2019)
export estimate_smooth_lp, smooth_lp_irf
export bspline_basis, cross_validate_lambda, compare_smooth_lp

# State-dependent LP (Auerbach & Gorodnichenko 2013)
export estimate_state_lp, state_irf
export logistic_transition, exponential_transition, indicator_transition
export estimate_transition_params, test_regime_difference

# Propensity score LP (Angrist et al. 2018)
export estimate_propensity_lp, propensity_irf
export estimate_propensity_score, inverse_propensity_weights
export doubly_robust_lp, propensity_diagnostics

# =============================================================================
# Exports - GMM Estimation
# =============================================================================

export estimate_gmm, gmm_objective, gmm_summary
export optimal_weighting_matrix, identity_weighting
export j_test, numerical_gradient
export estimate_lp_gmm, lp_gmm_moments
export linear_gmm_solve, gmm_sandwich_vcov, andrews_lu_mmsc
export ParameterTransform, to_unconstrained, to_constrained, transform_jacobian
export SMMModel, estimate_smm, autocovariance_moments

# =============================================================================
# Exports - Panel VAR
# =============================================================================

# Types
export PVARModel, PVARStability, PVARTestResult

# Estimation
export estimate_pvar, estimate_pvar_feols

# Analysis
export pvar_oirf, pvar_girf, pvar_fevd, pvar_stability

# Bootstrap
export pvar_bootstrap_irf

# Specification tests
export pvar_hansen_j, pvar_mmsc, pvar_lag_selection

# =============================================================================
# Exports - ARIMA Models
# =============================================================================

# Abstract type
export AbstractARIMAModel

# Model types
export ARModel, MAModel, ARMAModel, ARIMAModel
export ARIMAForecast, ARIMAOrderSelection

# Type accessors
export ar_order, ma_order, diff_order

# Estimation functions
export estimate_ar, estimate_ma, estimate_arma, estimate_arima

# Order selection
export select_arima_order, auto_arima, ic_table

# =============================================================================
# Exports - Non-Gaussian VAR Identification
# =============================================================================

# Abstract types
export AbstractNormalityTest, AbstractNonGaussianSVAR

# Result types
export NormalityTestResult, NormalityTestSuite
export ICASVARResult, NonGaussianMLResult
export MarkovSwitchingSVARResult, GARCHSVARResult
export SmoothTransitionSVARResult, ExternalVolatilitySVARResult
export IdentifiabilityTestResult

# Normality tests
export jarque_bera_test, mardia_test, doornik_hansen_test
export henze_zirkler_test, normality_test_suite

# ICA-based SVAR identification
export identify_fastica, identify_jade, identify_sobi
export identify_dcov, identify_hsic

# Non-Gaussian ML SVAR
export identify_student_t, identify_mixture_normal
export identify_pml, identify_skew_normal, identify_nongaussian_ml

# Heteroskedasticity identification
export identify_markov_switching, identify_garch
export identify_smooth_transition, identify_external_volatility

# Identifiability tests
export test_identification_strength, test_shock_gaussianity
export test_gaussian_vs_nongaussian, test_shock_independence
export test_overidentification

# =============================================================================
# Exports - Volatility Models (ARCH/GARCH/SV)
# =============================================================================

# Abstract type
export AbstractVolatilityModel

# ARCH types and estimation
export ARCHModel, VolatilityForecast
export estimate_arch
export arch_lm_test, ljung_box_squared

# GARCH types and estimation
export GARCHModel, EGARCHModel, GJRGARCHModel
export estimate_garch, estimate_egarch, estimate_gjr_garch
export news_impact_curve

# SV types and estimation
export SVModel
export estimate_sv

# Type accessors
export arch_order, garch_order, persistence, halflife, unconditional_variance

# =============================================================================
# Exports - Time Series Filters
# =============================================================================

# Abstract type
export AbstractFilterResult

# Result types
export HPFilterResult, HamiltonFilterResult, BeveridgeNelsonResult
export BaxterKingResult, BoostedHPResult

# Filter functions
export hp_filter, hamilton_filter, beveridge_nelson, baxter_king, boosted_hp

# Accessors
export trend, cycle

# =============================================================================
# Exports - StatsAPI Interface
# =============================================================================

export fit, coef, vcov, residuals, predict
export r2, aic, bic, dof, nobs
export loglikelihood, confint, stderror, islinear

# =============================================================================
# Exports - Data Module
# =============================================================================

# Data types and frequency
export AbstractMacroData, TimeSeriesData, PanelData, CrossSectionData
export Frequency, Daily, Monthly, Quarterly, Yearly, Mixed, Other

# Constructors and accessors
export varnames, frequency, time_index, obs_id, rename_vars!
export set_time_index!, set_obs_id!, desc, vardesc, set_desc!, set_vardesc!, dates, set_dates!
export xtset, isbalanced, groups, ngroups, group_data, panel_summary

# Validation
export DataDiagnostic, diagnose, fix, validate_for_model

# Transformations
export apply_tcode, inverse_tcode, apply_filter

# Summary statistics
export DataSummary, describe_data

# Conversion
export to_matrix, to_vector

# Examples
export load_example

# =============================================================================
# Exports - Plotting
# =============================================================================

export plot_result, PlotOutput, save_plot, display_plot

end # module
