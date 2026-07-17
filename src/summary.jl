# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Publication-quality report and summary tables for all model results.

Provides a unified interface using multiple dispatch:
- `report(result)` - Print comprehensive summary
- `table(result, ...)` - Extract data as matrix
- `print_table(result, ...)` - Print formatted table

Also provides common interface methods for all analysis results:
- `point_estimate(result)` - Get point estimate
- `has_uncertainty(result)` - Check if uncertainty bounds available
- `uncertainty_bounds(result)` - Get (lower, upper) bounds if available
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Unified Result Interface - Common Accessors
# =============================================================================

"""
    point_estimate(result::AbstractAnalysisResult)

Get the point estimate from an analysis result.

Returns the main values/estimates (IRF values, FEVD proportions, HD contributions).
"""
point_estimate(r::AbstractAnalysisResult) = error("point_estimate not implemented for $(typeof(r))")

"""
    has_uncertainty(result::AbstractAnalysisResult) -> Bool

Check if the result includes uncertainty quantification (confidence intervals or posterior quantiles).
"""
has_uncertainty(r::AbstractAnalysisResult) = false

"""
    uncertainty_bounds(result::AbstractAnalysisResult) -> Union{Nothing, Tuple}

Get uncertainty bounds (lower, upper) if available, otherwise nothing.
"""
uncertainty_bounds(r::AbstractAnalysisResult) = nothing

# --- ImpulseResponse implementations ---

point_estimate(r::ImpulseResponse) = r.values
has_uncertainty(r::ImpulseResponse) = r.ci_type != :none
function uncertainty_bounds(r::ImpulseResponse)
    r.ci_type == :none && return nothing
    (r.ci_lower, r.ci_upper)
end

point_estimate(r::BayesianImpulseResponse) = r.point_estimate
has_uncertainty(r::BayesianImpulseResponse) = true
function uncertainty_bounds(r::BayesianImpulseResponse)
    nq = length(r.quantile_levels)
    (r.quantiles[:,:,:,1], r.quantiles[:,:,:,nq])
end

# --- FEVD implementations ---

point_estimate(r::FEVD) = r.proportions
has_uncertainty(r::FEVD) = false
uncertainty_bounds(r::FEVD) = nothing

point_estimate(r::BayesianFEVD) = r.point_estimate
has_uncertainty(r::BayesianFEVD) = true
function uncertainty_bounds(r::BayesianFEVD)
    nq = length(r.quantile_levels)
    (r.quantiles[:,:,:,1], r.quantiles[:,:,:,nq])
end

# --- HistoricalDecomposition implementations ---

point_estimate(r::HistoricalDecomposition) = r.contributions
has_uncertainty(r::HistoricalDecomposition) = false
uncertainty_bounds(r::HistoricalDecomposition) = nothing

point_estimate(r::BayesianHistoricalDecomposition) = r.point_estimate
has_uncertainty(r::BayesianHistoricalDecomposition) = true
function uncertainty_bounds(r::BayesianHistoricalDecomposition)
    nq = length(r.quantile_levels)
    (r.quantiles[:,:,:,1], r.quantiles[:,:,:,nq])
end

# =============================================================================
# Table Formatting
# =============================================================================

# _fmt, _fmt_pct, _select_horizons are defined in display_utils.jl

# =============================================================================
# report() - Comprehensive summaries
# =============================================================================

"""
    report(model::VARModel)

Print comprehensive VAR model summary including specification, per-equation
coefficient estimates with standard errors and significance, information
criteria, residual covariance, and stationarity check.
"""
report(model::VARModel) = report(stdout, model)
function report(io::IO, model::VARModel{T}) where {T}
    n, p = nvars(model), model.p
    T_eff = effective_nobs(model)
    k = ncoefs(model)

    spec_data = [
        "Variables" n;
        "Lags" p;
        "Observations (effective)" T_eff;
        "Parameters per equation" k
    ]
    _pretty_table(io, spec_data;
        title = "Vector Autoregression — VAR($p)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # --- Per-equation fit summary ---
    _, X = construct_var_matrices(model.Y, p)
    XtX_inv = robust_inv(X' * X)
    dof_r = T_eff - k

    eq_data = Matrix{Any}(undef, n, 6)
    for j in 1:n
        ssr = sum(abs2, model.U[:, j])
        sst = sum(abs2, model.U[:, j] .+ (X * model.B[:, j]) .- mean(model.Y[(p+1):end, j]))
        r2 = sst > 0 ? one(T) - ssr / sst : zero(T)
        adj_r2 = one(T) - (one(T) - r2) * (T_eff - 1) / max(dof_r, 1)
        rmse = sqrt(ssr / T_eff)
        # F-statistic: (SSR_restricted - SSR_unrestricted) / (k-1) / (SSR_unrestricted / dof_r)
        # Restricted = constant only
        f_stat = dof_r > 0 && (k - 1) > 0 ? (r2 / (k - 1)) / ((one(T) - r2) / dof_r) : zero(T)
        eq_data[j, 1] = model.varnames[j]
        eq_data[j, 2] = k
        eq_data[j, 3] = _fmt(rmse)
        eq_data[j, 4] = _fmt(r2)
        eq_data[j, 5] = _fmt(adj_r2)
        eq_data[j, 6] = _fmt(f_stat; digits=3)
    end
    _pretty_table(io, eq_data;
        title = "Equation Summary",
        column_labels = ["Equation", "Parms", "RMSE", "R²", "Adj. R²", "F-stat"],
        alignment = [:l, :r, :r, :r, :r, :r],
    )

    # --- Per-equation coefficient tables ---
    coef_names = String[_INTERCEPT_LABEL]
    for l in 1:p
        for v in 1:n
            push!(coef_names, "$(model.varnames[v]).L$l")
        end
    end

    z_crit = T(quantile(TDist(dof_r), T(0.975)))
    for j in 1:n
        se_j = sqrt.(max.(diag(XtX_inv) .* model.Sigma[j, j], zero(T)))
        coef_vals = model.B[:, j]
        _coef_table(io, "Equation: $(model.varnames[j])", coef_names, coef_vals, se_j;
                    dist=:t, dof_r=dof_r)
    end

    # --- Information Criteria ---
    # Compute log-likelihood: -(T_eff*n/2)*log(2π) - (T_eff/2)*logdet(Σ) - T_eff*n/2
    logdet_Sigma = logdet(model.Sigma)
    loglik_val = -T(T_eff * n) / 2 * log(T(2π)) - T(T_eff) / 2 * logdet_Sigma - T(T_eff * n) / 2
    ic_data = ["Log-likelihood" _fmt(loglik_val; digits=4);
               "AIC (per obs.)" _fmt(model.aic; digits=4);
               "BIC (per obs.)" _fmt(model.bic; digits=4);
               "HQIC (per obs.)" _fmt(model.hqic; digits=4)]
    _pretty_table(io, ic_data;
        title = "Information Criteria",
        column_labels = ["Criterion", "Value"],
        alignment = [:l, :r],
    )

    # --- Residual Covariance ---
    _matrix_table(io, model.Sigma, "Residual Covariance (Σ)";
        row_labels=model.varnames,
        col_labels=model.varnames)

    # --- Residual Correlation ---
    D_inv = Diagonal(one(T) ./ sqrt.(max.(diag(model.Sigma), eps(T))))
    corr_mat = D_inv * model.Sigma * D_inv
    _matrix_table(io, corr_mat, "Residual Correlation";
        row_labels=model.varnames,
        col_labels=model.varnames)

    # --- Stationarity ---
    F = companion_matrix(model.B, n, p)
    max_mod = maximum(abs.(eigvals(F)))
    stable = max_mod < 1 ? "Yes" : "No"
    stab_data = Any["Stationary" stable; "Max |λ|" _fmt(max_mod)]
    _pretty_table(io, stab_data;
        title = "Stationarity",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    # --- Notes ---
    _sig_legend(io)
end

"""
    report(post::BVARPosterior)

Print comprehensive Bayesian VAR posterior summary.
"""
report(post::BVARPosterior) = show(stdout, post)

"""
    report(vecm::VECMModel)

Print comprehensive VECM summary including cointegrating vectors, adjustment
coefficients, short-run dynamics, and diagnostics.
"""
report(m::VECMModel) = report(stdout, m)
function report(io::IO, m::VECMModel{T}) where {T}
    n = nvars(m)
    r = m.rank
    p_diff = m.p - 1
    T_eff = effective_nobs(m)

    # --- Specification ---
    spec_data = [
        "Variables" n;
        "VAR order (p)" m.p;
        "Lagged differences" p_diff;
        "Cointegrating rank (r)" r;
        "Observations (effective)" T_eff;
        "Deterministic" string(m.deterministic);
        "Method" string(m.method)
    ]
    _pretty_table(io, spec_data;
        title = "Vector Error Correction Model — VECM($p_diff), Rank $r",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # --- Cointegrating vectors (β) ---
    if r > 0
        _matrix_table(io, m.beta, "Cointegrating Vectors (β)";
            row_labels=m.varnames,
            col_labels=["β$j" for j in 1:r])

        # --- Adjustment coefficients (α) ---
        _matrix_table(io, m.alpha, "Adjustment Coefficients (α)";
            row_labels=m.varnames,
            col_labels=["α$j" for j in 1:r])

        # --- Long-run matrix (Π = αβ') ---
        _matrix_table(io, m.Pi, "Long-Run Matrix (Π = αβ')";
            row_labels=m.varnames,
            col_labels=m.varnames)
    end

    # --- Short-run dynamics ---
    for (i, Gi) in enumerate(m.Gamma)
        _matrix_table(io, Gi, "Short-Run Dynamics Γ$i";
            row_labels=m.varnames,
            col_labels=m.varnames)
    end

    # --- Intercept ---
    mu_data = Matrix{Any}(undef, n, 2)
    for i in 1:n
        mu_data[i, 1] = m.varnames[i]
        mu_data[i, 2] = _fmt(m.mu[i])
    end
    _pretty_table(io, mu_data;
        title = "Intercept (μ)",
        column_labels = ["Variable", "Value"],
        alignment = [:l, :r],
    )

    # --- Information Criteria ---
    ic_data = ["Log-likelihood" _fmt(m.loglik; digits=4);
               "AIC (per obs.)" _fmt(m.aic; digits=4);
               "BIC (per obs.)" _fmt(m.bic; digits=4);
               "HQIC (per obs.)" _fmt(m.hqic; digits=4)]
    _pretty_table(io, ic_data;
        title = "Information Criteria",
        column_labels = ["Criterion", "Value"],
        alignment = [:l, :r],
    )

    # --- Residual Covariance ---
    _matrix_table(io, m.Sigma, "Residual Covariance (Σ)";
        row_labels=m.varnames,
        col_labels=m.varnames)

    # --- Residual Correlation ---
    D_inv = Diagonal(one(T) ./ sqrt.(max.(diag(m.Sigma), eps(T))))
    corr_mat = D_inv * m.Sigma * D_inv
    _matrix_table(io, corr_mat, "Residual Correlation";
        row_labels=m.varnames,
        col_labels=m.varnames)

    # --- Notes ---
    println(io, "Note: Standard errors for α/β not available (asymptotic SEs: future release)")
end

"""
    report(f::VECMForecast)

Print VECM forecast summary.
"""
report(f::VECMForecast) = show(stdout, f)

"""
    report(g::VECMGrangerResult)

Print VECM Granger causality test results.
"""
report(g::VECMGrangerResult) = show(stdout, g)

"""
    report(res::VECMRestrictionTest)

Print a Johansen LR restriction-test summary: the statistic, degrees of freedom,
p-value, restriction description, and the restricted vs. unrestricted eigenvalues
(EV-38 / #446).
"""
report(res::VECMRestrictionTest) = report(stdout, res)
function report(io::IO, res::VECMRestrictionTest{T}) where {T}
    hdr = ["LR χ² statistic" _fmt(res.lr_stat; digits=4);
           "Degrees of freedom" res.df;
           "P-value" _format_pvalue(res.pvalue);
           "Cointegrating rank" res.rank;
           "Converged" string(res.converged)]
    _pretty_table(io, hdr;
        title = "VECM Restriction Test — $(res.description)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    if !isempty(res.eigenvalues_restricted)
        r = res.rank
        eig = Matrix{Any}(undef, r, 3)
        for i in 1:r
            eig[i, 1] = "λ$i"
            eig[i, 2] = _fmt(res.eigenvalues_unrestricted[i]; digits=6)
            eig[i, 3] = _fmt(res.eigenvalues_restricted[i]; digits=6)
        end
        _pretty_table(io, eig;
            title = "Eigenvalues",
            column_labels = ["", "Unrestricted", "Restricted"],
            alignment = [:l, :r, :r],
        )
    end
    println(io, "Note: LR = T Σᵢ ln[(1−λ*ᵢ)/(1−λᵢ)] ~ χ²($(res.df)) under H₀.")
end

# =============================================================================
# report() - Time Series Filters
# =============================================================================

"""
    report(r::HPFilterResult)

Print HP filter summary.
"""
report(r::HPFilterResult) = show(stdout, r)

"""
    report(r::HamiltonFilterResult)

Print Hamilton filter summary.
"""
report(r::HamiltonFilterResult) = show(stdout, r)

"""
    report(r::BeveridgeNelsonResult)

Print Beveridge-Nelson decomposition summary.
"""
report(r::BeveridgeNelsonResult) = show(stdout, r)

"""
    report(r::BaxterKingResult)

Print Baxter-King band-pass filter summary.
"""
report(r::BaxterKingResult) = show(stdout, r)

"""
    report(r::BoostedHPResult)

Print boosted HP filter summary.
"""
report(r::BoostedHPResult) = show(stdout, r)

"""
    report(r::X13FilterResult)

Print X-13ARIMA-SEATS seasonal adjustment summary.
"""
report(r::X13FilterResult) = show(stdout, r)

"""
    report(irf::ImpulseResponse)
    report(irf::BayesianImpulseResponse)

Print IRF summary with values at selected horizons.
"""
report(irf::ImpulseResponse) = show(stdout, irf)
report(irf::BayesianImpulseResponse) = show(stdout, irf)

"""
    report(f::FEVD)
    report(f::BayesianFEVD)

Print FEVD summary with decomposition at selected horizons.
"""
report(f::FEVD) = show(stdout, f)
report(f::BayesianFEVD) = show(stdout, f)

"""
    report(hd::HistoricalDecomposition)
    report(hd::BayesianHistoricalDecomposition)

Print HD summary with contribution statistics.
"""
report(hd::HistoricalDecomposition) = show(stdout, hd)
report(hd::BayesianHistoricalDecomposition) = show(stdout, hd)

# =============================================================================
# report() - Universal coverage for all package types
# =============================================================================

# --- Models via abstract dispatch ---
report(x::AbstractARIMAModel) = show(stdout, x)
report(x::AbstractFactorModel) = show(stdout, x)
report(x::AbstractVolatilityModel) = show(stdout, x)
report(x::AbstractMGARCHModel) = show(stdout, x)   # EV-16 (#424): CCC/DCC/BEKK
report(x::AbstractLPModel) = show(stdout, x)
report(x::AbstractGMMModel) = show(stdout, x)
# Nonparametric regression & density (EV-33, #441)
report(x::KernelDensity) = show(stdout, x)
report(x::KernelRegression) = show(stdout, x)
report(x::LowessFit) = show(stdout, x)

# --- State-space models (EV-37, #445) ---
report(ss::StateSpaceModel) = report(stdout, ss)
function report(io::IO, ss::StateSpaceModel{T}) where {T}
    if !isfitted(ss)
        println(io, "State-Space Model (unfitted spec)")
        println(io, "  n_obs = $(ss.n_obs), n_state = $(ss.n_state), init = :$(ss.init_mode)")
        return nothing
    end
    println(io, "State-Space Model — Maximum Likelihood (prediction-error decomposition)")
    println(io, "  Observations: $(ss.T_obs)   State dim: $(ss.n_state)   Obs dim: $(ss.n_obs)")
    println(io, "  Log-likelihood: $(round(ss.loglik, digits=4))   Converged: $(ss.converged)   Init: :$(ss.init_mode)")
    println(io)
    if !isempty(ss.theta)
        # Hyper-parameters (variances / builder θ̂) printed as a plain estimate table.
        # Standard errors are not delta-propagated here; the "—" columns are intentional.
        se = fill(T(NaN), length(ss.theta))
        _coef_table(io, "Hyper-parameters", ss.param_names, Vector{T}(ss.theta), se;
                    coef_label="Estimate")
    end
    # One-step-ahead residual diagnostics (drop missing).
    r = vec(ss.std_residuals)
    r = r[.!isnan.(r)]
    if length(r) > 8
        lb = ljung_box_test(r; lags=min(10, length(r) ÷ 2))
        println(io)
        println(io, "  Std. residual Ljung–Box(", lb.lags, "): Q = ", round(lb.statistic, digits=3),
                "  p = ", round(lb.pvalue, digits=4))
    end
    return nothing
end

# --- Hypothesis test results ---
report(x::AbstractUnitRootTest) = show(stdout, x)
report(x::AbstractNormalityTest) = show(stdout, x)
report(x::AbstractNonGaussianSVAR) = show(stdout, x)
# Equality-of-distribution + rank-correlation battery (EV-34, #442)
report(x::EqualityTestResult) = show(stdout, x)
report(x::CorTestResult) = show(stdout, x)

# --- Types without abstract parents ---
report(x::ARIMAForecast) = show(stdout, x)
report(x::ARIMAOrderSelection) = show(stdout, x)
report(x::FactorForecast) = show(stdout, x)
report(x::VolatilityForecast) = show(stdout, x)
report(x::VARForecast) = show(stdout, x)
report(x::BVARForecast) = show(stdout, x)
report(x::LPForecast) = show(stdout, x)
report(x::LPImpulseResponse) = show(stdout, x)
report(x::LPFEVD) = show(stdout, x)
report(x::StructuralLP) = show(stdout, x)
report(x::IdentifiabilityTestResult) = show(stdout, x)
report(x::NormalityTestSuite) = show(stdout, x)

# --- Auxiliary types ---
report(x::BSplineBasis) = show(stdout, x)
report(x::StateTransition) = show(stdout, x)
report(x::PropensityScoreConfig) = show(stdout, x)
report(x::MinnesotaHyperparameters) = show(stdout, x)
report(x::AriasSVARResult) = show(stdout, x)
report(x::UhligSVARResult) = show(stdout, x)
report(x::SVARRestrictions) = show(stdout, x)

# DSGE
report(x::DSGESolution) = show(stdout, x)
report(x::PerturbationSolution) = show(stdout, x)
report(x::ProjectionSolution) = show(stdout, x)
report(x::PerfectForesightPath) = show(stdout, x)
report(x::DSGESpec) = show(stdout, x)
report(x::DSGEEstimation) = report(stdout, x)
function report(io::IO, x::DSGEEstimation)
    show(io, x)
    println(io)
    show(io, x.solution)
end
report(x::BayesianDSGE) = show(stdout, x)
report(x::BayesianDSGESimulation) = show(stdout, x)

# Cross-sectional models
report(m::RegModel) = show(stdout, m)
report(m::LogitModel) = show(stdout, m)
report(m::ProbitModel) = show(stdout, m)
report(me::MarginalEffects) = show(stdout, me)
report(m::PenalizedRegModel) = show(stdout, m)  # EV-03 (#411)
report(r::SelectionResult) = show(stdout, r)    # EV-04 (#412)
report(io::IO, r::SelectionResult) = show(io, r)  # EV-04 (#412)
report(m::TobitModel) = show(stdout, m)         # EV-17 (#425)
report(m::TruncRegModel) = show(stdout, m)      # EV-17 (#425)
report(m::HeckmanModel) = show(stdout, m)       # EV-18 (#426)
report(m::RobustRegModel) = show(stdout, m)          # EV-40 (#448)
report(io::IO, m::RobustRegModel) = show(io, m)      # EV-40 (#448)
report(m::CointRegModel) = show(stdout, m)      # EV-10 (#418)
report(io::IO, m::CointRegModel) = show(io, m)  # EV-10 (#418)
report(m::SURModel) = show(stdout, m)           # EV-35 (#443)
report(io::IO, m::SURModel) = show(io, m)       # EV-35 (#443)
report(m::ThreeSLSModel) = show(stdout, m)      # EV-35 (#443)
report(io::IO, m::ThreeSLSModel) = show(io, m)  # EV-35 (#443)
report(m::PanelCointRegModel) = show(stdout, m)      # EV-22 (#430)
report(io::IO, m::PanelCointRegModel) = show(io, m)  # EV-22 (#430)

# Panel VAR
report(x::PVARModel) = show(stdout, x)
report(x::PVARStability) = show(stdout, x)
report(x::PVARTestResult) = show(stdout, x)

# DiD
report(x::DIDResult) = show(stdout, x)
report(x::EventStudyLP) = show(stdout, x)
report(x::LPDiDResult) = show(stdout, x)
report(x::BaconDecomposition) = show(stdout, x)
report(x::HonestDiDResult) = show(stdout, x)

# Input-Output analysis
report(x::IOData) = show(stdout, x)
report(x::LeontiefModel) = show(stdout, x)
report(x::GhoshModel) = show(stdout, x)
report(x::IOMultipliers) = show(stdout, x)
report(x::LinkageResult) = show(stdout, x)
report(x::SDAResult) = show(stdout, x)
report(x::ExtractionResult) = show(stdout, x)
report(x::BaqaeeFarhiResult) = show(stdout, x)
report(x::FootprintResult) = show(stdout, x)

# --- Show-existing types that lacked a report() dispatch (S3/T167) ---
report(x::GrangerCausalityResult) = show(stdout, x)
report(x::FAVARModel) = show(stdout, x)
report(x::BayesianFAVAR) = show(stdout, x)
report(x::ACFResult) = show(stdout, x)
report(x::SpectralDensityResult) = show(stdout, x)
report(x::CrossSpectrumResult) = show(stdout, x)
report(x::TransferFunctionResult) = show(stdout, x)
report(x::LjungBoxResult) = show(stdout, x)
report(x::BoxPierceResult) = show(stdout, x)
report(x::DurbinWatsonResult) = show(stdout, x)
report(x::LRTestResult) = show(stdout, x)
report(x::LMTestResult) = show(stdout, x)
report(x::PretrendTestResult) = show(stdout, x)

# --- Wrapped bare-return display types (S3/T167) ---
report(x::OddsRatio) = show(stdout, x)
report(x::MultinomialMarginalEffects) = show(stdout, x)
report(x::NowcastForecast) = show(stdout, x)
report(x::PanelUnitRootSummary) = show(stdout, x)

# =============================================================================
# Split Files (table extraction, display, references, nowcasting)
# =============================================================================

include("summary_tables.jl")
include("summary_display.jl")
include("summary_refs.jl")
include("summary_nowcast.jl")
