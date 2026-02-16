# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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

point_estimate(r::BayesianImpulseResponse) = r.mean
has_uncertainty(r::BayesianImpulseResponse) = true
function uncertainty_bounds(r::BayesianImpulseResponse)
    nq = length(r.quantile_levels)
    (r.quantiles[:,:,:,1], r.quantiles[:,:,:,nq])
end

# --- FEVD implementations ---

point_estimate(r::FEVD) = r.proportions
has_uncertainty(r::FEVD) = false
uncertainty_bounds(r::FEVD) = nothing

point_estimate(r::BayesianFEVD) = r.mean
has_uncertainty(r::BayesianFEVD) = true
function uncertainty_bounds(r::BayesianFEVD)
    nq = length(r.quantile_levels)
    (r.quantiles[:,:,:,1], r.quantiles[:,:,:,nq])
end

# --- HistoricalDecomposition implementations ---

point_estimate(r::HistoricalDecomposition) = r.contributions
has_uncertainty(r::HistoricalDecomposition) = false
uncertainty_bounds(r::HistoricalDecomposition) = nothing

point_estimate(r::BayesianHistoricalDecomposition) = r.mean
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
function report(model::VARModel{T}) where {T}
    n, p = nvars(model), model.p
    T_eff = effective_nobs(model)
    k = ncoefs(model)

    spec_data = [
        "Variables" n;
        "Lags" p;
        "Observations (effective)" T_eff;
        "Parameters per equation" k
    ]
    _pretty_table(stdout, spec_data;
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
        eq_data[j, 1] = "Var $j"
        eq_data[j, 2] = k
        eq_data[j, 3] = _fmt(rmse)
        eq_data[j, 4] = _fmt(r2)
        eq_data[j, 5] = _fmt(adj_r2)
        eq_data[j, 6] = _fmt(f_stat; digits=3)
    end
    _pretty_table(stdout, eq_data;
        title = "Equation Summary",
        column_labels = ["Equation", "Parms", "RMSE", "R²", "Adj. R²", "F-stat"],
        alignment = [:l, :r, :r, :r, :r, :r],
    )

    # --- Per-equation coefficient tables ---
    coef_names = String["const"]
    for l in 1:p
        for v in 1:n
            push!(coef_names, "Var$(v).L$l")
        end
    end

    z_crit = T(quantile(TDist(dof_r), T(0.975)))
    for j in 1:n
        se_j = sqrt.(max.(diag(XtX_inv) .* model.Sigma[j, j], zero(T)))
        coef_vals = model.B[:, j]
        _coef_table(stdout, "Equation: Var $j", coef_names, coef_vals, se_j;
                    dist=:t, dof_r=dof_r)
    end

    # --- Information Criteria ---
    # Compute log-likelihood: -(T_eff*n/2)*log(2π) - (T_eff/2)*logdet(Σ) - T_eff*n/2
    logdet_Sigma = logdet(model.Sigma)
    loglik_val = -T(T_eff * n) / 2 * log(T(2π)) - T(T_eff) / 2 * logdet_Sigma - T(T_eff * n) / 2
    ic_data = ["Log-likelihood" _fmt(loglik_val; digits=4);
               "AIC" _fmt(model.aic; digits=4);
               "BIC" _fmt(model.bic; digits=4);
               "HQIC" _fmt(model.hqic; digits=4)]
    _pretty_table(stdout, ic_data;
        title = "Information Criteria",
        column_labels = ["Criterion", "Value"],
        alignment = [:l, :r],
    )

    # --- Residual Covariance ---
    _matrix_table(stdout, model.Sigma, "Residual Covariance (Σ)";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Var $j" for j in 1:n])

    # --- Residual Correlation ---
    D_inv = Diagonal(one(T) ./ sqrt.(max.(diag(model.Sigma), eps(T))))
    corr_mat = D_inv * model.Sigma * D_inv
    _matrix_table(stdout, corr_mat, "Residual Correlation";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Var $j" for j in 1:n])

    # --- Stationarity ---
    F = companion_matrix(model.B, n, p)
    max_mod = maximum(abs.(eigvals(F)))
    stable = max_mod < 1 ? "Yes" : "No"
    stab_data = Any["Stationary" stable; "Max |λ|" _fmt(max_mod)]
    _pretty_table(stdout, stab_data;
        title = "Stationarity",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    # --- Notes ---
    note_data = Any["Significance" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(stdout, note_data; column_labels=["",""], alignment=[:l,:l])
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
function report(m::VECMModel{T}) where {T}
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
    _pretty_table(stdout, spec_data;
        title = "Vector Error Correction Model — VECM($p_diff), Rank $r",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # --- Cointegrating vectors (β) ---
    if r > 0
        _matrix_table(stdout, m.beta, "Cointegrating Vectors (β)";
            row_labels=["Var $i" for i in 1:n],
            col_labels=["β$j" for j in 1:r])

        # --- Adjustment coefficients (α) ---
        _matrix_table(stdout, m.alpha, "Adjustment Coefficients (α)";
            row_labels=["Var $i" for i in 1:n],
            col_labels=["α$j" for j in 1:r])

        # --- Long-run matrix (Π = αβ') ---
        _matrix_table(stdout, m.Pi, "Long-Run Matrix (Π = αβ')";
            row_labels=["Var $i" for i in 1:n],
            col_labels=["Var $j" for j in 1:n])
    end

    # --- Short-run dynamics ---
    for (i, Gi) in enumerate(m.Gamma)
        _matrix_table(stdout, Gi, "Short-Run Dynamics Γ$i";
            row_labels=["Var $i" for i in 1:n],
            col_labels=["Var $j" for j in 1:n])
    end

    # --- Intercept ---
    mu_data = Matrix{Any}(undef, n, 2)
    for i in 1:n
        mu_data[i, 1] = "Var $i"
        mu_data[i, 2] = _fmt(m.mu[i])
    end
    _pretty_table(stdout, mu_data;
        title = "Intercept (μ)",
        column_labels = ["Variable", "Value"],
        alignment = [:l, :r],
    )

    # --- Information Criteria ---
    ic_data = ["Log-likelihood" _fmt(m.loglik; digits=4);
               "AIC" _fmt(m.aic; digits=4);
               "BIC" _fmt(m.bic; digits=4);
               "HQIC" _fmt(m.hqic; digits=4)]
    _pretty_table(stdout, ic_data;
        title = "Information Criteria",
        column_labels = ["Criterion", "Value"],
        alignment = [:l, :r],
    )

    # --- Residual Covariance ---
    _matrix_table(stdout, m.Sigma, "Residual Covariance (Σ)";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Var $j" for j in 1:n])

    # --- Residual Correlation ---
    D_inv = Diagonal(one(T) ./ sqrt.(max.(diag(m.Sigma), eps(T))))
    corr_mat = D_inv * m.Sigma * D_inv
    _matrix_table(stdout, corr_mat, "Residual Correlation";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Var $j" for j in 1:n])

    # --- Notes ---
    note_data = Any["Note" "Standard errors for α/β not available (asymptotic SEs: future release)"]
    _pretty_table(stdout, note_data; column_labels=["",""], alignment=[:l,:l])
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
report(x::AbstractLPModel) = show(stdout, x)
report(x::AbstractGMMModel) = show(stdout, x)

# --- Hypothesis test results ---
report(x::AbstractUnitRootTest) = show(stdout, x)
report(x::AbstractNormalityTest) = show(stdout, x)
report(x::AbstractNonGaussianSVAR) = show(stdout, x)

# --- Types without abstract parents ---
report(x::ARIMAForecast) = show(stdout, x)
report(x::ARIMAOrderSelection) = show(stdout, x)
report(x::FactorForecast) = show(stdout, x)
report(x::VolatilityForecast) = show(stdout, x)
report(x::LPForecast) = show(stdout, x)
report(x::LPImpulseResponse) = show(stdout, x)
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


# =============================================================================
# Split Files (table extraction, display, references, nowcasting)
# =============================================================================

include("summary_tables.jl")
include("summary_display.jl")
include("summary_refs.jl")
include("summary_nowcast.jl")
