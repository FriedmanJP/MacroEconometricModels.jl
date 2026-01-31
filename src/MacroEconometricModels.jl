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
chain = estimate_bvar(Y, 2; prior=:minnesota)
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
using FFTW

# =============================================================================
# Include Source Files (Order Matters)
# =============================================================================

# Core utilities (no dependencies on other package files)
include("utils.jl")

# Type definitions (depends on utils.jl for some functions)
include("types.jl")

# Estimation modules
include("estimation.jl")
include("priors.jl")
include("bayesian.jl")

# Structural analysis
include("identification.jl")

# Factor models (static, dynamic, generalized dynamic)
include("factormodels.jl")

# GMM estimation (includes GMM types)
include("gmm.jl")

# Local Projections (consolidated structure)
include("lp_types.jl")      # LP type definitions
include("lp_core.jl")       # Covariance estimators + core LP + shared utilities
include("lp_extensions.jl") # LP-IV, Smooth LP, State LP, Propensity LP

# IRF and FEVD (after LP types for lp_irf support)
include("irf.jl")
include("fevd.jl")

# =============================================================================
# Exports - Types
# =============================================================================

# Abstract types
export AbstractVARModel, AbstractImpulseResponse, AbstractFEVD, AbstractPrior

# VAR types
export VARModel

# IRF types
export ImpulseResponse, BayesianImpulseResponse

# FEVD types
export FEVD, BayesianFEVD

# Prior types
export MinnesotaHyperparameters

# Factor model types
export AbstractFactorModel, FactorModel, DynamicFactorModel, GeneralizedDynamicFactorModel

# Local Projection types
export AbstractLPModel, AbstractLPImpulseResponse, AbstractCovarianceEstimator
export LPModel, LPImpulseResponse, LPIVModel, SmoothLPModel, StateLPModel, PropensityLPModel
export NeweyWestEstimator, WhiteEstimator, DriscollKraayEstimator
export BSplineBasis, StateTransition, PropensityScoreConfig

# GMM types
export AbstractGMMModel, GMMModel, GMMWeighting

# =============================================================================
# Exports - VAR Estimation
# =============================================================================

export estimate_var
export select_lag_order

# =============================================================================
# Exports - Bayesian Estimation
# =============================================================================

export estimate_bvar
export extract_chain_parameters
export parameters_to_model
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
export compute_Q
export compute_irf
export compute_structural_shocks

# Arias et al. (2018) SVAR identification
export ZeroRestriction, SignRestriction, SVARRestrictions, AriasSVARResult
export identify_arias, identify_arias_bayesian
export zero_restriction, sign_restriction
export irf_percentiles, irf_mean

# =============================================================================
# Exports - IRF and FEVD
# =============================================================================

export irf
export fevd

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

# HAC covariance estimators
export newey_west, white_vcov, optimal_bandwidth_nw, kernel_weight
export robust_vcov, long_run_variance, long_run_covariance

# LP-IV (Stock & Watson 2018)
export estimate_lp_iv, lp_iv_irf
export first_stage_regression, weak_instrument_test, sargan_test

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

# =============================================================================
# Exports - StatsAPI Interface
# =============================================================================

export fit, coef, vcov, residuals, predict
export r2, aic, bic, dof, nobs
export loglikelihood, confint, stderror, islinear

end # module
