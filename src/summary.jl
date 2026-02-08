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
        eq_data[j, 6] = _fmt(f_stat; digits=2)
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

    for j in 1:n
        se_j = sqrt.(max.(diag(XtX_inv) .* model.Sigma[j, j], zero(T)))
        coef_data = Matrix{Any}(undef, k, 6)
        for i in 1:k
            est = model.B[i, j]
            se = se_j[i]
            t_stat = se > 0 ? est / se : T(NaN)
            pval = isnan(t_stat) ? T(NaN) : T(2) * (one(T) - cdf(TDist(dof_r), abs(t_stat)))
            stars = isnan(pval) ? "" : _significance_stars(pval)
            coef_data[i, 1] = coef_names[i]
            coef_data[i, 2] = _fmt(est)
            coef_data[i, 3] = _fmt(se)
            coef_data[i, 4] = isnan(t_stat) ? "—" : string(_fmt(t_stat))
            coef_data[i, 5] = isnan(pval) ? "—" : _format_pvalue(pval)
            coef_data[i, 6] = stars
        end
        _pretty_table(stdout, coef_data;
            title = "Equation: Var $j",
            column_labels = ["", "Coef.", "Std. Err.", "t", "P>|t|", ""],
            alignment = [:l, :r, :r, :r, :r, :l],
        )
    end

    # --- Information Criteria ---
    ic_data = ["AIC" _fmt(model.aic; digits=2);
               "BIC" _fmt(model.bic; digits=2);
               "HQIC" _fmt(model.hqic; digits=2)]
    _pretty_table(stdout, ic_data;
        title = "Information Criteria",
        column_labels = ["Criterion", "Value"],
        alignment = [:l, :r],
    )

    # --- Residual Covariance ---
    Sigma_data = Matrix{Any}(undef, n, n + 1)
    for i in 1:n
        Sigma_data[i, 1] = "Var $i"
        for j in 1:n
            Sigma_data[i, j + 1] = _fmt(model.Sigma[i, j])
        end
    end
    _pretty_table(stdout, Sigma_data;
        title = "Residual Covariance (Σ)",
        column_labels = vcat([""], ["Var $j" for j in 1:n]),
        alignment = vcat([:l], fill(:r, n)),
    )

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
    ic_data = ["AIC" _fmt(m.aic; digits=2);
               "BIC" _fmt(m.bic; digits=2);
               "HQIC" _fmt(m.hqic; digits=2);
               "Log-likelihood" _fmt(m.loglik; digits=2)]
    _pretty_table(stdout, ic_data;
        title = "Information Criteria",
        column_labels = ["Criterion", "Value"],
        alignment = [:l, :r],
    )

    # --- Residual Covariance ---
    _matrix_table(stdout, m.Sigma, "Residual Covariance (Σ)";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Var $j" for j in 1:n])
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
report(x::SVARRestrictions) = show(stdout, x)

# =============================================================================
# table() - Extract data as matrix
# =============================================================================

"""
    table(irf::ImpulseResponse, var, shock; horizons=nothing) -> Matrix

Extract IRF values for a variable-shock pair.
Returns matrix with columns: [Horizon, IRF] or [Horizon, IRF, CI_lo, CI_hi].
"""
function table(irf::ImpulseResponse{T}, var::Int, shock::Int;
               horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    n_vars, n_shocks = length(irf.variables), length(irf.shocks)
    @assert 1 <= var <= n_vars "Variable index out of bounds"
    @assert 1 <= shock <= n_shocks "Shock index out of bounds"

    hs = isnothing(horizons) ? (1:irf.horizon) : horizons
    has_ci = irf.ci_type != :none

    ncols = has_ci ? 4 : 2
    result = Matrix{T}(undef, length(hs), ncols)
    for (i, h) in enumerate(hs)
        result[i, 1] = h
        result[i, 2] = irf.values[h, var, shock]
        if has_ci
            result[i, 3] = irf.ci_lower[h, var, shock]
            result[i, 4] = irf.ci_upper[h, var, shock]
        end
    end
    result
end

function table(irf::ImpulseResponse, var::String, shock::String; kwargs...)
    vi, si = _validate_var_shock_indices(var, shock, irf.variables, irf.shocks)
    table(irf, vi, si; kwargs...)
end

"""
    table(irf::BayesianImpulseResponse, var, shock; horizons=nothing) -> Matrix

Extract Bayesian IRF values. Returns [Horizon, Mean, Q1, Q2, ...].
"""
function table(irf::BayesianImpulseResponse{T}, var::Int, shock::Int;
               horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    @assert 1 <= var <= length(irf.variables)
    @assert 1 <= shock <= length(irf.shocks)

    hs = isnothing(horizons) ? (1:irf.horizon) : horizons
    nq = length(irf.quantile_levels)

    result = Matrix{T}(undef, length(hs), 2 + nq)
    for (i, h) in enumerate(hs)
        result[i, 1] = h
        result[i, 2] = irf.mean[h, var, shock]
        for q in 1:nq
            result[i, 2 + q] = irf.quantiles[h, var, shock, q]
        end
    end
    result
end

function table(irf::BayesianImpulseResponse, var::String, shock::String; kwargs...)
    vi, si = _validate_var_shock_indices(var, shock, irf.variables, irf.shocks)
    table(irf, vi, si; kwargs...)
end

"""
    table(f::FEVD, var; horizons=nothing) -> Matrix

Extract FEVD proportions for a variable. Returns [Horizon, Shock1, Shock2, ...].
"""
function table(f::FEVD{T}, var::Int;
               horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    n_vars, n_shocks, H = size(f.proportions)
    @assert 1 <= var <= n_vars

    hs = isnothing(horizons) ? (1:H) : horizons

    result = Matrix{T}(undef, length(hs), n_shocks + 1)
    for (i, h) in enumerate(hs)
        result[i, 1] = h
        for j in 1:n_shocks
            result[i, j + 1] = f.proportions[var, j, h]
        end
    end
    result
end

"""
    table(f::BayesianFEVD, var; horizons=nothing, stat=:mean) -> Matrix

Extract Bayesian FEVD values. stat can be :mean or quantile index.
"""
function table(f::BayesianFEVD{T}, var::Int;
               horizons::Union{Nothing,AbstractVector{Int}}=nothing,
               stat::Union{Symbol,Int}=:mean) where {T}
    @assert 1 <= var <= length(f.variables)

    hs = isnothing(horizons) ? (1:f.horizon) : horizons
    n_shocks = length(f.shocks)

    result = Matrix{T}(undef, length(hs), n_shocks + 1)
    for (i, h) in enumerate(hs)
        result[i, 1] = h
        for j in 1:n_shocks
            result[i, j + 1] = stat == :mean ? f.mean[h, var, j] : f.quantiles[h, var, j, stat]
        end
    end
    result
end

"""
    table(hd::HistoricalDecomposition, var; periods=nothing) -> Matrix

Extract HD contributions for a variable.
Returns [Period, Actual, Shock1, ..., ShockN, Initial].
"""
function table(hd::HistoricalDecomposition{T}, var::Int;
               periods::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    @assert 1 <= var <= length(hd.variables)

    ps = isnothing(periods) ? (1:hd.T_eff) : periods
    n_shocks = length(hd.shock_names)

    result = Matrix{T}(undef, length(ps), n_shocks + 3)
    for (i, t) in enumerate(ps)
        result[i, 1] = t
        result[i, 2] = hd.actual[t, var]
        for j in 1:n_shocks
            result[i, j + 2] = hd.contributions[t, var, j]
        end
        result[i, end] = hd.initial_conditions[t, var]
    end
    result
end

"""
    table(hd::BayesianHistoricalDecomposition, var; periods=nothing, stat=:mean) -> Matrix

Extract Bayesian HD contributions. stat can be :mean or quantile index.
"""
function table(hd::BayesianHistoricalDecomposition{T}, var::Int;
               periods::Union{Nothing,AbstractVector{Int}}=nothing,
               stat::Union{Symbol,Int}=:mean) where {T}
    @assert 1 <= var <= length(hd.variables)

    ps = isnothing(periods) ? (1:hd.T_eff) : periods
    n_shocks = length(hd.shock_names)

    result = Matrix{T}(undef, length(ps), n_shocks + 3)
    for (i, t) in enumerate(ps)
        result[i, 1] = t
        result[i, 2] = hd.actual[t, var]
        for j in 1:n_shocks
            result[i, j + 2] = stat == :mean ? hd.mean[t, var, j] : hd.quantiles[t, var, j, stat]
        end
        result[i, end] = stat == :mean ? hd.initial_mean[t, var] : hd.initial_quantiles[t, var, stat]
    end
    result
end

# --- VolatilityForecast ---

"""
    table(fc::VolatilityForecast) -> Matrix

Extract volatility forecast data.
Returns matrix with columns: [Horizon, Forecast, CI_lo, CI_hi, SE].
"""
function table(fc::VolatilityForecast{T}) where {T}
    h = fc.horizon
    result = Matrix{T}(undef, h, 5)
    for i in 1:h
        result[i, 1] = i
        result[i, 2] = fc.forecast[i]
        result[i, 3] = fc.ci_lower[i]
        result[i, 4] = fc.ci_upper[i]
        result[i, 5] = fc.se[i]
    end
    result
end

# --- ARIMAForecast ---

"""
    table(fc::ARIMAForecast) -> Matrix

Extract ARIMA forecast data.
Returns matrix with columns: [Horizon, Forecast, CI_lo, CI_hi, SE].
"""
function table(fc::ARIMAForecast{T}) where {T}
    h = fc.horizon
    result = Matrix{T}(undef, h, 5)
    for i in 1:h
        result[i, 1] = i
        result[i, 2] = fc.forecast[i]
        result[i, 3] = fc.ci_lower[i]
        result[i, 4] = fc.ci_upper[i]
        result[i, 5] = fc.se[i]
    end
    result
end

# --- FactorForecast ---

"""
    table(fc::FactorForecast, var_idx::Int; type=:observable) -> Matrix

Extract factor forecast data for a single variable.
`type=:observable` returns observable forecasts, `type=:factor` returns factor forecasts.
Returns matrix with columns: [Horizon, Forecast, CI_lo, CI_hi].
"""
function table(fc::FactorForecast{T}, var_idx::Int; type::Symbol=:observable) where {T}
    h = fc.horizon
    if type == :observable
        N = size(fc.observables, 2)
        @assert 1 <= var_idx <= N "Variable index $var_idx out of bounds (1:$N)"
        result = Matrix{T}(undef, h, 4)
        for i in 1:h
            result[i, 1] = i
            result[i, 2] = fc.observables[i, var_idx]
            result[i, 3] = fc.observables_lower[i, var_idx]
            result[i, 4] = fc.observables_upper[i, var_idx]
        end
    else
        r = size(fc.factors, 2)
        @assert 1 <= var_idx <= r "Factor index $var_idx out of bounds (1:$r)"
        result = Matrix{T}(undef, h, 4)
        for i in 1:h
            result[i, 1] = i
            result[i, 2] = fc.factors[i, var_idx]
            result[i, 3] = fc.factors_lower[i, var_idx]
            result[i, 4] = fc.factors_upper[i, var_idx]
        end
    end
    result
end

# --- LPImpulseResponse ---

"""
    table(irf::LPImpulseResponse, var_idx::Int) -> Matrix

Extract LP IRF values for a response variable.
Returns matrix with columns: [Horizon, IRF, SE, CI_lo, CI_hi].
"""
function table(irf::LPImpulseResponse{T}, var_idx::Int) where {T}
    n_resp = length(irf.response_vars)
    @assert 1 <= var_idx <= n_resp "Variable index $var_idx out of bounds (1:$n_resp)"
    H = irf.horizon
    result = Matrix{T}(undef, H + 1, 5)
    for i in 0:H
        result[i + 1, 1] = i
        result[i + 1, 2] = irf.values[i + 1, var_idx]
        result[i + 1, 3] = irf.se[i + 1, var_idx]
        result[i + 1, 4] = irf.ci_lower[i + 1, var_idx]
        result[i + 1, 5] = irf.ci_upper[i + 1, var_idx]
    end
    result
end

function table(irf::LPImpulseResponse, var_name::String)
    idx = findfirst(==(var_name), irf.response_vars)
    isnothing(idx) && throw(ArgumentError("Variable '$var_name' not found in response_vars"))
    table(irf, idx)
end

# =============================================================================
# print_table() - Formatted table output
# =============================================================================

"""
    print_table([io], irf::ImpulseResponse, var, shock; horizons=nothing)

Print formatted IRF table.
"""
function print_table(io::IO, irf::ImpulseResponse{T}, var::Int, shock::Int;
                     horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    raw = table(irf, var, shock; horizons=horizons)
    has_ci = irf.ci_type != :none

    if has_ci
        col_labels = ["h", "IRF", "Lower", "Upper"]
    else
        col_labels = ["h", "IRF"]
    end

    _pretty_table(io, raw;
        title = "IRF: $(irf.variables[var]) ← $(irf.shocks[shock])",
        column_labels = col_labels,
        alignment = fill(:r, size(raw, 2)),
    )
end

print_table(irf::ImpulseResponse, var, shock; kwargs...) =
    print_table(stdout, irf, var, shock; kwargs...)

function print_table(io::IO, irf::BayesianImpulseResponse{T}, var::Int, shock::Int;
                     horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    raw = table(irf, var, shock; horizons=horizons)

    q_labels = [_fmt_pct(q; digits=0) for q in irf.quantile_levels]
    col_labels = vcat(["h", "Mean"], q_labels)

    _pretty_table(io, raw;
        title = "Bayesian IRF: $(irf.variables[var]) ← $(irf.shocks[shock])",
        column_labels = col_labels,
        alignment = fill(:r, size(raw, 2)),
    )
end

print_table(irf::BayesianImpulseResponse, var, shock; kwargs...) =
    print_table(stdout, irf, var, shock; kwargs...)

"""
    print_table([io], f::FEVD, var; horizons=nothing)

Print formatted FEVD table.
"""
function print_table(io::IO, f::FEVD{T}, var::Int;
                     horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    raw = table(f, var; horizons=horizons)
    n_shocks = size(f.proportions, 2)

    # Format percentages
    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:size(raw, 2)
            data[i, j] = _fmt_pct(raw[i, j])
        end
    end

    col_labels = vcat(["h"], ["Shock $j" for j in 1:n_shocks])
    _pretty_table(io, data;
        title = "FEVD: Variable $var",
        column_labels = col_labels,
        alignment = fill(:r, size(data, 2)),
    )
end

print_table(f::FEVD, var; kwargs...) = print_table(stdout, f, var; kwargs...)

function print_table(io::IO, f::BayesianFEVD{T}, var::Int;
                     horizons::Union{Nothing,AbstractVector{Int}}=nothing,
                     stat::Union{Symbol,Int}=:mean) where {T}
    raw = table(f, var; horizons=horizons, stat=stat)
    n_shocks = length(f.shocks)

    stat_name = stat == :mean ? "mean" : _fmt_pct(f.quantile_levels[stat]; digits=0)

    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:size(raw, 2)
            data[i, j] = _fmt_pct(raw[i, j])
        end
    end

    col_labels = vcat(["h"], f.shocks)
    _pretty_table(io, data;
        title = "Bayesian FEVD: $(f.variables[var]) ($stat_name)",
        column_labels = col_labels,
        alignment = fill(:r, size(data, 2)),
    )
end

print_table(f::BayesianFEVD, var; kwargs...) = print_table(stdout, f, var; kwargs...)

"""
    print_table([io], hd::HistoricalDecomposition, var; periods=nothing)

Print formatted HD table.
"""
function print_table(io::IO, hd::HistoricalDecomposition{T}, var::Int;
                     periods::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    raw = table(hd, var; periods=periods)
    n_shocks = length(hd.shock_names)

    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:size(raw, 2)
            data[i, j] = _fmt(raw[i, j])
        end
    end

    col_labels = vcat(["t", "Actual"], hd.shock_names, ["Initial"])
    _pretty_table(io, data;
        title = "Historical Decomposition: $(hd.variables[var])",
        column_labels = col_labels,
        alignment = fill(:r, size(data, 2)),
    )
end

print_table(hd::HistoricalDecomposition, var; kwargs...) =
    print_table(stdout, hd, var; kwargs...)

function print_table(io::IO, hd::BayesianHistoricalDecomposition{T}, var::Int;
                     periods::Union{Nothing,AbstractVector{Int}}=nothing,
                     stat::Union{Symbol,Int}=:mean) where {T}
    raw = table(hd, var; periods=periods, stat=stat)
    n_shocks = length(hd.shock_names)

    stat_name = stat == :mean ? "mean" : _fmt_pct(hd.quantile_levels[stat]; digits=0)

    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:size(raw, 2)
            data[i, j] = _fmt(raw[i, j])
        end
    end

    col_labels = vcat(["t", "Actual"], hd.shock_names, ["Initial"])
    _pretty_table(io, data;
        title = "Bayesian HD: $(hd.variables[var]) ($stat_name)",
        column_labels = col_labels,
        alignment = fill(:r, size(data, 2)),
    )
end

print_table(hd::BayesianHistoricalDecomposition, var; kwargs...) =
    print_table(stdout, hd, var; kwargs...)

# =============================================================================
# Base.show Methods for Result Types
# =============================================================================

function Base.show(io::IO, irf::ImpulseResponse{T}) where {T}
    n_vars, n_shocks = length(irf.variables), length(irf.shocks)
    H = irf.horizon

    ci_str = irf.ci_type == :none ? "None" : string(irf.ci_type)
    spec_data = ["Variables" n_vars; "Shocks" n_shocks; "Horizon" H; "CI" ci_str]
    _pretty_table(io, spec_data;
        title = "Impulse Response Functions",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    horizons_show = _select_horizons(H)
    for j in 1:n_shocks
        data = Matrix{Any}(undef, n_vars, length(horizons_show) + 1)
        for v in 1:n_vars
            data[v, 1] = irf.variables[v]
            for (hi, h) in enumerate(horizons_show)
                val = irf.values[h, v, j]
                if irf.ci_type != :none
                    lo, up = irf.ci_lower[h, v, j], irf.ci_upper[h, v, j]
                    sig = (lo > 0 || up < 0) ? "*" : ""
                    data[v, hi + 1] = string(_fmt(val), sig)
                else
                    data[v, hi + 1] = _fmt(val)
                end
            end
        end

        _pretty_table(io, data;
            title = "Shock: $(irf.shocks[j])",
            column_labels = vcat([""], ["h=$h" for h in horizons_show]),
            alignment = vcat([:l], fill(:r, length(horizons_show))),
        )
    end

    if irf.ci_type != :none
        note_data = Any["Note" "* CI excludes zero"]
        _pretty_table(io, note_data;
            column_labels = ["", ""],
            alignment = [:l, :l],
        )
    end
end

function Base.show(io::IO, irf::BayesianImpulseResponse{T}) where {T}
    n_vars, n_shocks = length(irf.variables), length(irf.shocks)
    H = irf.horizon
    nq = length(irf.quantile_levels)

    q_str = join([_fmt_pct(q; digits=0) for q in irf.quantile_levels], ", ")
    spec_data = ["Variables" n_vars; "Shocks" n_shocks; "Horizon" H; "Quantiles" q_str]
    _pretty_table(io, spec_data;
        title = "Bayesian Impulse Response Functions",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    horizons_show = _select_horizons(H)
    median_idx = nq >= 3 ? 2 : 1
    q_label = _fmt_pct(irf.quantile_levels[median_idx]; digits=0)

    for j in 1:n_shocks
        data = Matrix{Any}(undef, n_vars, length(horizons_show) + 1)
        for v in 1:n_vars
            data[v, 1] = irf.variables[v]
            for (hi, h) in enumerate(horizons_show)
                med = irf.quantiles[h, v, j, median_idx]
                lo, up = irf.quantiles[h, v, j, 1], irf.quantiles[h, v, j, nq]
                sig = (lo > 0 || up < 0) ? "*" : ""
                data[v, hi + 1] = string(_fmt(med), sig)
            end
        end

        _pretty_table(io, data;
            title = "Shock: $(irf.shocks[j]) ($q_label)",
            column_labels = vcat([""], ["h=$h" for h in horizons_show]),
            alignment = vcat([:l], fill(:r, length(horizons_show))),
        )
    end

    note_data = Any["Note" "* Credible interval excludes zero"]
    _pretty_table(io, note_data;
        column_labels = ["", ""],
        alignment = [:l, :l],
    )
end

function Base.show(io::IO, f::FEVD{T}) where {T}
    n_vars, n_shocks, H = size(f.proportions)

    spec_data = ["Variables" n_vars; "Shocks" n_shocks; "Horizon" H]
    _pretty_table(io, spec_data;
        title = "Forecast Error Variance Decomposition",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    for h in _select_horizons(H)
        data = Matrix{Any}(undef, n_vars, n_shocks + 1)
        for i in 1:n_vars
            data[i, 1] = "Var $i"
            for j in 1:n_shocks
                data[i, j + 1] = _fmt_pct(f.proportions[i, j, h])
            end
        end

        _pretty_table(io, data;
            title = "h = $h",
            column_labels = vcat([""], ["Shock $j" for j in 1:n_shocks]),
            alignment = vcat([:l], fill(:r, n_shocks)),
        )
    end
end

function Base.show(io::IO, f::BayesianFEVD{T}) where {T}
    n_vars, n_shocks = length(f.variables), length(f.shocks)
    H = f.horizon

    q_str = join([_fmt_pct(q; digits=0) for q in f.quantile_levels], ", ")
    spec_data = ["Variables" n_vars; "Shocks" n_shocks; "Horizon" H; "Quantiles" q_str]
    _pretty_table(io, spec_data;
        title = "Bayesian FEVD (posterior mean)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    for h in _select_horizons(H)
        data = Matrix{Any}(undef, n_vars, n_shocks + 1)
        for i in 1:n_vars
            data[i, 1] = f.variables[i]
            for j in 1:n_shocks
                data[i, j + 1] = _fmt_pct(f.mean[h, i, j])
            end
        end

        _pretty_table(io, data;
            title = "h = $h",
            column_labels = vcat([""], f.shocks),
            alignment = vcat([:l], fill(:r, n_shocks)),
        )
    end
end

# =============================================================================
# Structural LP Display
# =============================================================================

function Base.show(io::IO, slp::StructuralLP{T}) where {T}
    n = nvars(slp)
    H = size(slp.irf.values, 1)

    ci_str = slp.irf.ci_type == :none ? "None" : string(slp.irf.ci_type)
    spec_data = [
        "Identification" string(slp.method);
        "Variables" n;
        "IRF horizon" H;
        "LP lags" slp.lags;
        "HAC estimator" string(slp.cov_type);
        "CI" ci_str
    ]
    _pretty_table(io, spec_data;
        title = "Structural Local Projections",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Show IRF summary at selected horizons
    horizons_show = _select_horizons(H)
    for j in 1:n
        data = Matrix{Any}(undef, n, length(horizons_show) + 1)
        for v in 1:n
            data[v, 1] = slp.irf.variables[v]
            for (hi, h) in enumerate(horizons_show)
                val = slp.irf.values[h, v, j]
                se_val = slp.se[h, v, j]
                sig = abs(val) > 1.96 * se_val ? "*" : ""
                data[v, hi + 1] = string(_fmt(val), sig)
            end
        end

        _pretty_table(io, data;
            title = "Shock: $(slp.irf.shocks[j])",
            column_labels = vcat([""], ["h=$h" for h in horizons_show]),
            alignment = vcat([:l], fill(:r, length(horizons_show))),
        )
    end

    note_data = Any["Note" "* significant at 5% (|IRF/SE| > 1.96)"]
    _pretty_table(io, note_data;
        column_labels = ["", ""],
        alignment = [:l, :l],
    )
end

point_estimate(r::StructuralLP) = r.irf.values
has_uncertainty(r::StructuralLP) = r.irf.ci_type != :none
function uncertainty_bounds(r::StructuralLP)
    r.irf.ci_type == :none && return nothing
    (r.irf.ci_lower, r.irf.ci_upper)
end

"""
    print_table([io], slp::StructuralLP, var, shock; horizons=nothing)

Print formatted IRF table for a specific variable-shock pair from structural LP.
"""
function print_table(io::IO, slp::StructuralLP{T}, var::Int, shock::Int;
                     horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    print_table(io, slp.irf, var, shock; horizons=horizons)
end

print_table(slp::StructuralLP, var::Int, shock::Int; kwargs...) =
    print_table(stdout, slp, var, shock; kwargs...)

# =============================================================================
# LP Forecast Display
# =============================================================================

function Base.show(io::IO, fc::LPForecast{T}) where {T}
    H = fc.horizon
    n_resp = length(fc.response_vars)

    spec_data = [
        "Forecast horizon" H;
        "Response variables" n_resp;
        "Shock variable" fc.shock_var;
        "CI method" string(fc.ci_method);
        "Confidence level" _fmt_pct(fc.conf_level)
    ]
    _pretty_table(io, spec_data;
        title = "LP Forecast",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Forecast table
    data = Matrix{Any}(undef, H, 1 + n_resp * (fc.ci_method == :none ? 1 : 3))
    col_labels = String["h"]

    for (j, rv) in enumerate(fc.response_vars)
        if fc.ci_method == :none
            push!(col_labels, "Var $rv")
            for h in 1:H
                data[h, 1] = h
                data[h, 1 + j] = _fmt(fc.forecasts[h, j])
            end
        else
            push!(col_labels, "Var $rv")
            push!(col_labels, "Lower")
            push!(col_labels, "Upper")
            col_offset = 1 + (j - 1) * 3
            for h in 1:H
                data[h, 1] = h
                data[h, col_offset + 1] = _fmt(fc.forecasts[h, j])
                data[h, col_offset + 2] = _fmt(fc.ci_lower[h, j])
                data[h, col_offset + 3] = _fmt(fc.ci_upper[h, j])
            end
        end
    end

    _pretty_table(io, data;
        title = "Forecasts",
        column_labels = col_labels,
        alignment = fill(:r, length(col_labels)),
    )
end

"""
    print_table([io], fc::LPForecast)

Print formatted LP forecast table.
"""
function print_table(io::IO, fc::LPForecast{T}) where {T}
    show(io, fc)
end

print_table(fc::LPForecast) = print_table(stdout, fc)

# =============================================================================
# LP-FEVD Display (Gorodnichenko & Lee 2019)
# =============================================================================

function Base.show(io::IO, f::LPFEVD{T}) where {T}
    n_vars, n_shocks, H = size(f.proportions)

    method_str = f.method == :r2 ? "R²" : f.method == :lp_a ? "LP-A" : "LP-B"
    bc_str = f.bias_correction ? "Yes (VAR bootstrap)" : "No"
    spec_data = [
        "Variables" n_vars;
        "Shocks" n_shocks;
        "Horizon" H;
        "Estimator" method_str;
        "Bias corrected" bc_str;
        "Bootstrap reps" f.n_boot;
        "Conf. level" _fmt_pct(f.conf_level)
    ]
    _pretty_table(io, spec_data;
        title = "LP-FEVD (Gorodnichenko & Lee 2019)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    # Use bias-corrected values if available
    vals = f.bias_correction ? f.bias_corrected : f.proportions

    for h in _select_horizons(H)
        data = Matrix{Any}(undef, n_vars, n_shocks + 1)
        for i in 1:n_vars
            data[i, 1] = "Var $i"
            for j in 1:n_shocks
                v = vals[i, j, h]
                if f.n_boot > 0
                    se = f.se[i, j, h]
                    data[i, j + 1] = string(_fmt_pct(v), " (", _fmt(se), ")")
                else
                    data[i, j + 1] = _fmt_pct(v)
                end
            end
        end

        _pretty_table(io, data;
            title = "h = $h",
            column_labels = vcat([""], ["Shock $j" for j in 1:n_shocks]),
            alignment = vcat([:l], fill(:r, n_shocks)),
        )
    end
end

"""
    print_table([io], f::LPFEVD, var_idx; horizons=...)

Print formatted LP-FEVD table for variable `var_idx`.
"""
function print_table(io::IO, f::LPFEVD{T}, var_idx::Int;
                     horizons::Vector{Int}=collect(_select_horizons(f.horizon))) where {T}
    n_shocks = size(f.proportions, 2)
    H_sel = filter(h -> h <= f.horizon, horizons)

    vals = f.bias_correction ? f.bias_corrected : f.proportions
    header = vcat(["h"], ["Shock $j" for j in 1:n_shocks])

    if f.n_boot > 0
        # Include CIs
        header_full = String["h"]
        for j in 1:n_shocks
            push!(header_full, "Shock $j")
            push!(header_full, "Lower")
            push!(header_full, "Upper")
        end
        data = Matrix{Any}(undef, length(H_sel), 1 + 3 * n_shocks)
        for (i, h) in enumerate(H_sel)
            data[i, 1] = h
            for j in 1:n_shocks
                col = 1 + (j - 1) * 3
                data[i, col + 1] = _fmt(vals[var_idx, j, h])
                data[i, col + 2] = _fmt(f.ci_lower[var_idx, j, h])
                data[i, col + 3] = _fmt(f.ci_upper[var_idx, j, h])
            end
        end
        _pretty_table(io, data;
            title = "LP-FEVD: Variable $var_idx",
            column_labels = header_full,
            alignment = fill(:r, length(header_full)),
        )
    else
        data = Matrix{Any}(undef, length(H_sel), 1 + n_shocks)
        for (i, h) in enumerate(H_sel)
            data[i, 1] = h
            for j in 1:n_shocks
                data[i, j + 1] = _fmt(vals[var_idx, j, h])
            end
        end
        _pretty_table(io, data;
            title = "LP-FEVD: Variable $var_idx",
            column_labels = header,
            alignment = fill(:r, length(header)),
        )
    end
end

print_table(f::LPFEVD, var_idx::Int; kwargs...) =
    print_table(stdout, f, var_idx; kwargs...)

# --- VolatilityForecast ---

"""
    print_table([io], fc::VolatilityForecast)

Print formatted volatility forecast table.
"""
function print_table(io::IO, fc::VolatilityForecast{T}) where {T}
    raw = table(fc)
    data = Matrix{Any}(undef, size(raw, 1), 5)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])     # h
        data[i, 2] = _fmt(raw[i, 2])    # Forecast
        data[i, 3] = _fmt(raw[i, 5])    # SE
        data[i, 4] = _fmt(raw[i, 3])    # Lower
        data[i, 5] = _fmt(raw[i, 4])    # Upper
    end
    ci_pct = round(Int, 100 * fc.conf_level)
    _pretty_table(io, data;
        title = "Volatility Forecast ($(fc.model_type), $(ci_pct)% CI)",
        column_labels = ["h", "σ² Forecast", "Std. Err.", "Lower", "Upper"],
        alignment = fill(:r, 5),
    )
end

print_table(fc::VolatilityForecast) = print_table(stdout, fc)

# --- ARIMAForecast ---

"""
    print_table([io], fc::ARIMAForecast)

Print formatted ARIMA forecast table.
"""
function print_table(io::IO, fc::ARIMAForecast{T}) where {T}
    raw = table(fc)
    data = Matrix{Any}(undef, size(raw, 1), 5)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])     # h
        data[i, 2] = _fmt(raw[i, 2])    # Forecast
        data[i, 3] = _fmt(raw[i, 5])    # SE
        data[i, 4] = _fmt(raw[i, 3])    # Lower
        data[i, 5] = _fmt(raw[i, 4])    # Upper
    end
    ci_pct = round(Int, 100 * fc.conf_level)
    _pretty_table(io, data;
        title = "ARIMA Forecast ($(ci_pct)% CI)",
        column_labels = ["h", "Forecast", "Std. Err.", "Lower", "Upper"],
        alignment = fill(:r, 5),
    )
end

print_table(fc::ARIMAForecast) = print_table(stdout, fc)

# --- FactorForecast ---

"""
    print_table([io], fc::FactorForecast, var_idx; type=:observable)

Print formatted factor forecast table for a single variable.
"""
function print_table(io::IO, fc::FactorForecast{T}, var_idx::Int;
                     type::Symbol=:observable) where {T}
    raw = table(fc, var_idx; type=type)
    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:4
            data[i, j] = _fmt(raw[i, j])
        end
    end
    label = type == :observable ? "Observable $var_idx" : "Factor $var_idx"
    ci_str = fc.ci_method == :none ? "" : " ($(fc.ci_method))"
    _pretty_table(io, data;
        title = "Factor Forecast: $label$ci_str",
        column_labels = ["h", "Forecast", "Lower", "Upper"],
        alignment = fill(:r, 4),
    )
end

print_table(fc::FactorForecast, var_idx::Int; kwargs...) =
    print_table(stdout, fc, var_idx; kwargs...)

# --- LPImpulseResponse ---

"""
    print_table([io], irf::LPImpulseResponse, var_idx)

Print formatted LP IRF table for a response variable.
"""
function print_table(io::IO, irf::LPImpulseResponse{T}, var_idx::Int) where {T}
    raw = table(irf, var_idx)
    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:5
            data[i, j] = _fmt(raw[i, j])
        end
    end
    resp_name = irf.response_vars[var_idx]
    _pretty_table(io, data;
        title = "LP IRF: $resp_name ← $(irf.shock_var)",
        column_labels = ["h", "IRF", "Std. Err.", "Lower", "Upper"],
        alignment = fill(:r, 5),
    )
end

print_table(irf::LPImpulseResponse, var_idx::Int) =
    print_table(stdout, irf, var_idx)

function print_table(io::IO, irf::LPImpulseResponse, var_name::String)
    idx = findfirst(==(var_name), irf.response_vars)
    isnothing(idx) && throw(ArgumentError("Variable '$var_name' not found in response_vars"))
    print_table(io, irf, idx)
end

print_table(irf::LPImpulseResponse, var_name::String) =
    print_table(stdout, irf, var_name)

# =============================================================================
# Base.show Methods for LP Types
# =============================================================================

function Base.show(io::IO, m::LPModel)
    cov_name = m.cov_estimator isa NeweyWestEstimator ? "Newey-West" :
               m.cov_estimator isa WhiteEstimator ? "White (HC0)" : "Driscoll-Kraay"
    data = Any[
        "Variables" nvars(m);
        "Shock variable" m.shock_var;
        "Response variables" length(m.response_vars);
        "Horizon" m.horizon;
        "Lags" m.lags;
        "Observations" size(m.Y, 1);
        "Covariance" cov_name
    ]
    _pretty_table(io, data;
        title = "Local Projection Model (Jordà 2005)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, m::LPIVModel)
    cov_name = m.cov_estimator isa NeweyWestEstimator ? "Newey-West" :
               m.cov_estimator isa WhiteEstimator ? "White (HC0)" : "Driscoll-Kraay"
    min_F = round(minimum(m.first_stage_F), digits=2)
    max_F = round(maximum(m.first_stage_F), digits=2)
    data = Any[
        "Variables" nvars(m);
        "Shock variable" m.shock_var;
        "Instruments" n_instruments(m);
        "Horizon" m.horizon;
        "Lags" m.lags;
        "Observations" size(m.Y, 1);
        "First-stage F (min)" min_F;
        "First-stage F (max)" max_F;
        "Covariance" cov_name
    ]
    _pretty_table(io, data;
        title = "LP-IV Model (Stock & Watson 2018)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

nvars(m::LPIVModel) = size(m.Y, 2)

function Base.show(io::IO, m::SmoothLPModel)
    cov_name = m.cov_estimator isa NeweyWestEstimator ? "Newey-West" :
               m.cov_estimator isa WhiteEstimator ? "White (HC0)" : "Driscoll-Kraay"
    data = Any[
        "Variables" size(m.Y, 2);
        "Shock variable" m.shock_var;
        "Horizon" m.horizon;
        "Lags" m.lags;
        "Spline degree" m.spline_basis.degree;
        "Interior knots" m.spline_basis.n_interior_knots;
        "Lambda (penalty)" _fmt(m.lambda);
        "Basis functions" n_basis(m.spline_basis);
        "Observations" size(m.Y, 1);
        "Covariance" cov_name
    ]
    _pretty_table(io, data;
        title = "Smooth LP Model (Barnichon & Brownlees 2019)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, m::StateLPModel)
    cov_name = m.cov_estimator isa NeweyWestEstimator ? "Newey-West" :
               m.cov_estimator isa WhiteEstimator ? "White (HC0)" : "Driscoll-Kraay"
    pct_exp = round(mean(m.state.F_values) * 100, digits=1)
    data = Any[
        "Variables" size(m.Y, 2);
        "Shock variable" m.shock_var;
        "Horizon" m.horizon;
        "Lags" m.lags;
        "Transition" string(m.state.method);
        "Gamma (smoothness)" _fmt(m.state.gamma);
        "Threshold" _fmt(m.state.threshold);
        "% in expansion" string(pct_exp, "%");
        "Observations" size(m.Y, 1);
        "Covariance" cov_name
    ]
    _pretty_table(io, data;
        title = "State-Dependent LP Model (Auerbach & Gorodnichenko 2013)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, m::PropensityLPModel)
    cov_name = m.cov_estimator isa NeweyWestEstimator ? "Newey-West" :
               m.cov_estimator isa WhiteEstimator ? "White (HC0)" : "Driscoll-Kraay"
    n_t = n_treated(m)
    n_c = n_control(m)
    data = Any[
        "Variables" size(m.Y, 2);
        "Horizon" m.horizon;
        "Treated" n_t;
        "Control" n_c;
        "Covariates" size(m.covariates, 2);
        "PS method" string(m.config.method);
        "Trimming" string(m.config.trimming);
        "Observations" size(m.Y, 1);
        "Covariance" cov_name
    ]
    _pretty_table(io, data;
        title = "Propensity Score LP Model (Angrist et al. 2018)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, irf::LPImpulseResponse)
    ci_pct = round(irf.conf_level * 100, digits=0)
    data = Any[
        "Shock" irf.shock_var;
        "Response variables" length(irf.response_vars);
        "Horizon" irf.horizon;
        "CI type" string(irf.cov_type);
        "Confidence level" string(ci_pct, "%")
    ]
    _pretty_table(io, data;
        title = "LP Impulse Response",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, b::BSplineBasis)
    data = Any[
        "Degree" b.degree;
        "Interior knots" b.n_interior_knots;
        "Basis functions" n_basis(b);
        "Horizon range" string(minimum(b.horizons), ":", maximum(b.horizons))
    ]
    _pretty_table(io, data;
        title = "B-Spline Basis",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, s::StateTransition)
    pct_high = round(mean(s.F_values) * 100, digits=1)
    data = Any[
        "Transition" string(s.method);
        "Gamma" _fmt(s.gamma);
        "Threshold" _fmt(s.threshold);
        "% in high state" string(pct_high, "%");
        "Observations" length(s.state_var)
    ]
    _pretty_table(io, data;
        title = "State Transition Function",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, c::PropensityScoreConfig)
    data = Any[
        "Method" string(c.method);
        "Trimming" string(c.trimming);
        "Normalize" c.normalize ? "Yes" : "No"
    ]
    _pretty_table(io, data;
        title = "Propensity Score Configuration",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# Base.show Methods for VAR/Identification Types
# =============================================================================

function Base.show(io::IO, h::MinnesotaHyperparameters)
    data = Any[
        "tau (tightness)" _fmt(h.tau);
        "decay (lag decay)" _fmt(h.decay);
        "lambda (sum-of-coef)" _fmt(h.lambda);
        "mu (co-persistence)" _fmt(h.mu);
        "omega (covariance)" _fmt(h.omega)
    ]
    _pretty_table(io, data;
        title = "Minnesota Prior Hyperparameters",
        column_labels = ["Parameter", "Value"],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::AriasSVARResult)
    n_draws = length(r.Q_draws)
    acc_pct = round(r.acceptance_rate * 100, digits=2)
    n_zeros = length(r.restrictions.zeros)
    n_signs = length(r.restrictions.signs)
    data = Any[
        "Accepted draws" n_draws;
        "Acceptance rate" string(acc_pct, "%");
        "Zero restrictions" n_zeros;
        "Sign restrictions" n_signs;
        "Variables" r.restrictions.n_vars;
        "Shocks" r.restrictions.n_shocks
    ]
    _pretty_table(io, data;
        title = "Arias et al. (2018) SVAR Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::ZeroRestriction)
    print(io, "ZeroRestriction(var=$(r.variable), shock=$(r.shock), horizon=$(r.horizon))")
end

function Base.show(io::IO, r::SignRestriction)
    sign_str = r.sign > 0 ? "+" : "-"
    print(io, "SignRestriction(var=$(r.variable), shock=$(r.shock), horizon=$(r.horizon), sign=$(sign_str))")
end

function Base.show(io::IO, r::SVARRestrictions)
    data = Any[
        "Zero restrictions" length(r.zeros);
        "Sign restrictions" length(r.signs);
        "Variables" r.n_vars;
        "Shocks" r.n_shocks
    ]
    _pretty_table(io, data;
        title = "SVAR Restrictions",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# refs() — Multi-Format Bibliographic References
# =============================================================================

const _RefEntry = @NamedTuple{
    key::Symbol, authors::String, year::Int, title::String,
    journal::String, volume::String, issue::String, pages::String,
    doi::String, isbn::String, publisher::String, entry_type::Symbol
}

const _REFERENCES = Dict{Symbol, _RefEntry}(
    # --- VAR & Structural VAR ---
    :sims1980 => (key=:sims1980, authors="Sims, Christopher A.", year=1980,
        title="Macroeconomics and Reality", journal="Econometrica",
        volume="48", issue="1", pages="1--48", doi="10.2307/1912017",
        isbn="", publisher="", entry_type=:article),
    :lutkepohl2005 => (key=:lutkepohl2005, authors="L\\\"utkepohl, Helmut", year=2005,
        title="New Introduction to Multiple Time Series Analysis", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-3-540-40172-8", publisher="Springer", entry_type=:book),
    :blanchard_quah1989 => (key=:blanchard_quah1989, authors="Blanchard, Olivier Jean and Quah, Danny", year=1989,
        title="The Dynamic Effects of Aggregate Demand and Supply Disturbances",
        journal="American Economic Review", volume="79", issue="4", pages="655--673",
        doi="", isbn="", publisher="", entry_type=:article),
    :uhlig2005 => (key=:uhlig2005, authors="Uhlig, Harald", year=2005,
        title="What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure",
        journal="Journal of Monetary Economics", volume="52", issue="2", pages="381--419",
        doi="10.1016/j.jmoneco.2004.05.007", isbn="", publisher="", entry_type=:article),
    :antolin_diaz_rubio_ramirez2018 => (key=:antolin_diaz_rubio_ramirez2018,
        authors="Antol{\\'\\i}n-D{\\'\\i}az, Juan and Rubio-Ram{\\'\\i}rez, Juan F.", year=2018,
        title="Narrative Sign Restrictions for SVARs",
        journal="American Economic Review", volume="108", issue="10", pages="2802--2829",
        doi="10.1257/aer.20161852", isbn="", publisher="", entry_type=:article),
    :arias_rubio_ramirez_waggoner2018 => (key=:arias_rubio_ramirez_waggoner2018,
        authors="Arias, Jonas E. and Rubio-Ram{\\'\\i}rez, Juan F. and Waggoner, Daniel F.", year=2018,
        title="Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications",
        journal="Econometrica", volume="86", issue="2", pages="685--720",
        doi="10.3982/ECTA14468", isbn="", publisher="", entry_type=:article),
    :kilian1998 => (key=:kilian1998, authors="Kilian, Lutz", year=1998,
        title="Small-Sample Confidence Intervals for Impulse Response Functions",
        journal="Review of Economics and Statistics", volume="80", issue="2", pages="218--230",
        doi="10.1162/003465398557465", isbn="", publisher="", entry_type=:article),
    :kilian_lutkepohl2017 => (key=:kilian_lutkepohl2017,
        authors="Kilian, Lutz and L\\\"utkepohl, Helmut", year=2017,
        title="Structural Vector Autoregressive Analysis", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-1-107-19657-5", publisher="Cambridge University Press", entry_type=:book),
    # --- Bayesian VAR ---
    :litterman1986 => (key=:litterman1986, authors="Litterman, Robert B.", year=1986,
        title="Forecasting with Bayesian Vector Autoregressions---Five Years of Experience",
        journal="Journal of Business \\& Economic Statistics", volume="4", issue="1", pages="25--38",
        doi="10.1080/07350015.1986.10509491", isbn="", publisher="", entry_type=:article),
    :kadiyala_karlsson1997 => (key=:kadiyala_karlsson1997,
        authors="Kadiyala, K. Rao and Karlsson, Sune", year=1997,
        title="Numerical Methods for Estimation and Inference in Bayesian VAR-Models",
        journal="Journal of Applied Econometrics", volume="12", issue="2", pages="99--132",
        doi="10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A",
        isbn="", publisher="", entry_type=:article),
    # --- Local Projections ---
    :jorda2005 => (key=:jorda2005, authors="Jord\\`a, \\`Oscar", year=2005,
        title="Estimation and Inference of Impulse Responses by Local Projections",
        journal="American Economic Review", volume="95", issue="1", pages="161--182",
        doi="10.1257/0002828053828518", isbn="", publisher="", entry_type=:article),
    :stock_watson2018 => (key=:stock_watson2018,
        authors="Stock, James H. and Watson, Mark W.", year=2018,
        title="Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments",
        journal="Economic Journal", volume="128", issue="610", pages="917--948",
        doi="10.1111/ecoj.12593", isbn="", publisher="", entry_type=:article),
    :barnichon_brownlees2019 => (key=:barnichon_brownlees2019,
        authors="Barnichon, Regis and Brownlees, Christian", year=2019,
        title="Impulse Response Estimation by Smooth Local Projections",
        journal="Review of Economics and Statistics", volume="101", issue="3", pages="522--530",
        doi="10.1162/rest_a_00778", isbn="", publisher="", entry_type=:article),
    :auerbach_gorodnichenko2012 => (key=:auerbach_gorodnichenko2012,
        authors="Auerbach, Alan J. and Gorodnichenko, Yuriy", year=2012,
        title="Measuring the Output Responses to Fiscal Policy",
        journal="American Economic Journal: Economic Policy", volume="4", issue="2", pages="1--27",
        doi="10.1257/pol.4.2.1", isbn="", publisher="", entry_type=:article),
    :angrist_jorda_kuersteiner2018 => (key=:angrist_jorda_kuersteiner2018,
        authors="Angrist, Joshua D. and Jord\\`a, \\`Oscar and Kuersteiner, Guido M.", year=2018,
        title="Semiparametric Estimates of Monetary Policy Effects: String Theory Revisited",
        journal="Journal of Business \\& Economic Statistics", volume="36", issue="3", pages="371--387",
        doi="10.1080/07350015.2016.1204919", isbn="", publisher="", entry_type=:article),
    :plagborg_moller_wolf2021 => (key=:plagborg_moller_wolf2021,
        authors="Plagborg-M{\\o}ller, Mikkel and Wolf, Christian K.", year=2021,
        title="Local Projections and VARs Estimate the Same Impulse Responses",
        journal="Econometrica", volume="89", issue="2", pages="955--980",
        doi="10.3982/ECTA17813", isbn="", publisher="", entry_type=:article),
    :gorodnichenko_lee2020 => (key=:gorodnichenko_lee2020,
        authors="Gorodnichenko, Yuriy and Lee, Byoungchan", year=2020,
        title="Forecast Error Variance Decompositions with Local Projections",
        journal="Journal of Business \\& Economic Statistics", volume="38", issue="4", pages="921--933",
        doi="10.1080/07350015.2019.1610661", isbn="", publisher="", entry_type=:article),
    # --- Factor Models ---
    :bai_ng2002 => (key=:bai_ng2002, authors="Bai, Jushan and Ng, Serena", year=2002,
        title="Determining the Number of Factors in Approximate Factor Models",
        journal="Econometrica", volume="70", issue="1", pages="191--221",
        doi="10.1111/1468-0262.00273", isbn="", publisher="", entry_type=:article),
    :stock_watson2002 => (key=:stock_watson2002,
        authors="Stock, James H. and Watson, Mark W.", year=2002,
        title="Forecasting Using Principal Components from a Large Number of Predictors",
        journal="Journal of the American Statistical Association", volume="97", issue="460", pages="1167--1179",
        doi="10.1198/016214502388618960", isbn="", publisher="", entry_type=:article),
    # --- Unit Root Tests ---
    :dickey_fuller1979 => (key=:dickey_fuller1979,
        authors="Dickey, David A. and Fuller, Wayne A.", year=1979,
        title="Distribution of the Estimators for Autoregressive Time Series with a Unit Root",
        journal="Journal of the American Statistical Association", volume="74", issue="366a", pages="427--431",
        doi="10.1080/01621459.1979.10482531", isbn="", publisher="", entry_type=:article),
    :kpss1992 => (key=:kpss1992,
        authors="Kwiatkowski, Denis and Phillips, Peter C. B. and Schmidt, Peter and Shin, Yongcheol", year=1992,
        title="Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root",
        journal="Journal of Econometrics", volume="54", issue="1--3", pages="159--178",
        doi="10.1016/0304-4076(92)90104-Y", isbn="", publisher="", entry_type=:article),
    :phillips_perron1988 => (key=:phillips_perron1988,
        authors="Phillips, Peter C. B. and Perron, Pierre", year=1988,
        title="Testing for a Unit Root in Time Series Regression",
        journal="Biometrika", volume="75", issue="2", pages="335--346",
        doi="10.1093/biomet/75.2.335", isbn="", publisher="", entry_type=:article),
    :zivot_andrews1992 => (key=:zivot_andrews1992,
        authors="Zivot, Eric and Andrews, Donald W. K.", year=1992,
        title="Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis",
        journal="Journal of Business \\& Economic Statistics", volume="10", issue="3", pages="251--270",
        doi="10.1080/07350015.1992.10509904", isbn="", publisher="", entry_type=:article),
    :ng_perron2001 => (key=:ng_perron2001,
        authors="Ng, Serena and Perron, Pierre", year=2001,
        title="Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power",
        journal="Econometrica", volume="69", issue="6", pages="1519--1554",
        doi="10.1111/1468-0262.00256", isbn="", publisher="", entry_type=:article),
    :johansen1991 => (key=:johansen1991, authors="Johansen, S{\\o}ren", year=1991,
        title="Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models",
        journal="Econometrica", volume="59", issue="6", pages="1551--1580",
        doi="10.2307/2938278", isbn="", publisher="", entry_type=:article),
    :engle_granger1987 => (key=:engle_granger1987,
        authors="Engle, Robert F. and Granger, Clive W. J.", year=1987,
        title="Co-Integration and Error Correction: Representation, Estimation, and Testing",
        journal="Econometrica", volume="55", issue="2", pages="251--276",
        doi="10.2307/1913236", isbn="", publisher="", entry_type=:article),
    # --- ARIMA ---
    :box_jenkins1970 => (key=:box_jenkins1970,
        authors="Box, George E. P. and Jenkins, Gwilym M.", year=1970,
        title="Time Series Analysis: Forecasting and Control", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-0-8162-1094-7", publisher="Holden-Day", entry_type=:book),
    :hyndman_khandakar2008 => (key=:hyndman_khandakar2008,
        authors="Hyndman, Rob J. and Khandakar, Yeasmin", year=2008,
        title="Automatic Time Series Forecasting: The forecast Package for R",
        journal="Journal of Statistical Software", volume="27", issue="3", pages="1--22",
        doi="10.18637/jss.v027.i03", isbn="", publisher="", entry_type=:article),
    # --- GMM ---
    :hansen1982 => (key=:hansen1982, authors="Hansen, Lars Peter", year=1982,
        title="Large Sample Properties of Generalized Method of Moments Estimators",
        journal="Econometrica", volume="50", issue="4", pages="1029--1054",
        doi="10.2307/1912775", isbn="", publisher="", entry_type=:article),
    # --- Non-Gaussian SVAR — ICA ---
    :hyvarinen1999 => (key=:hyvarinen1999, authors="Hyv\\\"arinen, Aapo", year=1999,
        title="Fast and Robust Fixed-Point Algorithms for Independent Component Analysis",
        journal="IEEE Transactions on Neural Networks", volume="10", issue="3", pages="626--634",
        doi="10.1109/72.761722", isbn="", publisher="", entry_type=:article),
    :cardoso_souloumiac1993 => (key=:cardoso_souloumiac1993,
        authors="Cardoso, Jean-Fran{\\c{c}}ois and Souloumiac, Antoine", year=1993,
        title="Blind Beamforming for Non-Gaussian Signals",
        journal="IEE Proceedings F --- Radar and Signal Processing", volume="140", issue="6", pages="362--370",
        doi="10.1049/ip-f-2.1993.0054", isbn="", publisher="", entry_type=:article),
    :belouchrani1997 => (key=:belouchrani1997,
        authors="Belouchrani, Adel and Abed-Meraim, Karim and Cardoso, Jean-Fran{\\c{c}}ois and Moulines, Eric",
        year=1997, title="A Blind Source Separation Technique Using Second-Order Statistics",
        journal="IEEE Transactions on Signal Processing", volume="45", issue="2", pages="434--444",
        doi="10.1109/78.554307", isbn="", publisher="", entry_type=:article),
    :szekely_rizzo_bakirov2007 => (key=:szekely_rizzo_bakirov2007,
        authors="Sz{\\'e}kely, G{\\'a}bor J. and Rizzo, Maria L. and Bakirov, Nail K.", year=2007,
        title="Measuring and Testing Dependence by Correlation of Distances",
        journal="Annals of Statistics", volume="35", issue="6", pages="2769--2794",
        doi="10.1214/009053607000000505", isbn="", publisher="", entry_type=:article),
    :matteson_tsay2017 => (key=:matteson_tsay2017,
        authors="Matteson, David S. and Tsay, Ruey S.", year=2017,
        title="Independent Component Analysis via Distance Covariance",
        journal="Journal of the American Statistical Association", volume="112", issue="518", pages="623--637",
        doi="10.1080/01621459.2016.1150851", isbn="", publisher="", entry_type=:article),
    :gretton2005 => (key=:gretton2005,
        authors="Gretton, Arthur and Bousquet, Olivier and Smola, Alex and Sch\\\"olkopf, Bernhard", year=2005,
        title="Measuring Statistical Dependence with Hilbert-Schmidt Norms",
        journal="Algorithmic Learning Theory", volume="3734", issue="", pages="63--77",
        doi="10.1007/11564089_7", isbn="", publisher="", entry_type=:incollection),
    # --- Non-Gaussian SVAR — ML ---
    :lanne_meitz_saikkonen2017 => (key=:lanne_meitz_saikkonen2017,
        authors="Lanne, Markku and Meitz, Mika and Saikkonen, Pentti", year=2017,
        title="Identification and Estimation of Non-Gaussian Structural Vector Autoregressions",
        journal="Journal of Econometrics", volume="196", issue="2", pages="288--304",
        doi="10.1016/j.jeconom.2016.06.002", isbn="", publisher="", entry_type=:article),
    # --- Non-Gaussian SVAR — Heteroskedasticity ---
    :rigobon2003 => (key=:rigobon2003, authors="Rigobon, Roberto", year=2003,
        title="Identification Through Heteroskedasticity",
        journal="Review of Economics and Statistics", volume="85", issue="4", pages="777--792",
        doi="10.1162/003465303772815727", isbn="", publisher="", entry_type=:article),
    :lutkepohl_netsunajev2017 => (key=:lutkepohl_netsunajev2017,
        authors="L\\\"utkepohl, Helmut and Netsunajev, Aleksei", year=2017,
        title="Structural Vector Autoregressions with Smooth Transition in Variances",
        journal="Journal of Economic Dynamics and Control", volume="84", issue="", pages="43--57",
        doi="10.1016/j.jedc.2017.09.001", isbn="", publisher="", entry_type=:article),
    # --- Normality Tests ---
    :jarque_bera1980 => (key=:jarque_bera1980,
        authors="Jarque, Carlos M. and Bera, Anil K.", year=1980,
        title="Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals",
        journal="Economics Letters", volume="6", issue="3", pages="255--259",
        doi="10.1016/0165-1765(80)90024-5", isbn="", publisher="", entry_type=:article),
    :mardia1970 => (key=:mardia1970, authors="Mardia, Kanti V.", year=1970,
        title="Measures of Multivariate Skewness and Kurtosis with Applications",
        journal="Biometrika", volume="57", issue="3", pages="519--530",
        doi="10.1093/biomet/57.3.519", isbn="", publisher="", entry_type=:article),
    :doornik_hansen2008 => (key=:doornik_hansen2008,
        authors="Doornik, Jurgen A. and Hansen, Henrik", year=2008,
        title="An Omnibus Test for Univariate and Multivariate Normality",
        journal="Oxford Bulletin of Economics and Statistics", volume="70", issue="s1", pages="927--939",
        doi="10.1111/j.1468-0084.2008.00537.x", isbn="", publisher="", entry_type=:article),
    :henze_zirkler1990 => (key=:henze_zirkler1990,
        authors="Henze, Norbert and Zirkler, Bernhard", year=1990,
        title="A Class of Invariant Consistent Tests for Multivariate Normality",
        journal="Communications in Statistics --- Theory and Methods", volume="19", issue="10", pages="3595--3617",
        doi="10.1080/03610929008830400", isbn="", publisher="", entry_type=:article),
    # --- Covariance Estimators ---
    :newey_west1987 => (key=:newey_west1987,
        authors="Newey, Whitney K. and West, Kenneth D.", year=1987,
        title="A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
        journal="Econometrica", volume="55", issue="3", pages="703--708",
        doi="10.2307/1913610", isbn="", publisher="", entry_type=:article),
    :white1980 => (key=:white1980, authors="White, Halbert", year=1980,
        title="A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity",
        journal="Econometrica", volume="48", issue="4", pages="817--838",
        doi="10.2307/1912934", isbn="", publisher="", entry_type=:article),
    # --- Volatility Models ---
    :engle1982 => (key=:engle1982, authors="Engle, Robert F.", year=1982,
        title="Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation",
        journal="Econometrica", volume="50", issue="4", pages="987--1007",
        doi="10.2307/1912773", isbn="", publisher="", entry_type=:article),
    :bollerslev1986 => (key=:bollerslev1986, authors="Bollerslev, Tim", year=1986,
        title="Generalized Autoregressive Conditional Heteroskedasticity",
        journal="Journal of Econometrics", volume="31", issue="3", pages="307--327",
        doi="10.1016/0304-4076(86)90063-1", isbn="", publisher="", entry_type=:article),
    :nelson1991 => (key=:nelson1991, authors="Nelson, Daniel B.", year=1991,
        title="Conditional Heteroskedasticity in Asset Returns: A New Approach",
        journal="Econometrica", volume="59", issue="2", pages="347--370",
        doi="10.2307/2938260", isbn="", publisher="", entry_type=:article),
    :glosten_jagannathan_runkle1993 => (key=:glosten_jagannathan_runkle1993,
        authors="Glosten, Lawrence R. and Jagannathan, Ravi and Runkle, David E.", year=1993,
        title="On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks",
        journal="Journal of Finance", volume="48", issue="5", pages="1779--1801",
        doi="10.1111/j.1540-6261.1993.tb05128.x", isbn="", publisher="", entry_type=:article),
    :taylor1986 => (key=:taylor1986, authors="Taylor, Stephen J.", year=1986,
        title="Modelling Financial Time Series", journal="",
        volume="", issue="", pages="", doi="",
        isbn="978-0-471-90993-4", publisher="Wiley", entry_type=:book),
    :kim_shephard_chib1998 => (key=:kim_shephard_chib1998,
        authors="Kim, Sangjoon and Shephard, Neil and Chib, Siddhartha", year=1998,
        title="Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models",
        journal="Review of Economic Studies", volume="65", issue="3", pages="361--393",
        doi="10.1111/1467-937X.00050", isbn="", publisher="", entry_type=:article),
    :omori2007 => (key=:omori2007,
        authors="Omori, Yasuhiro and Chib, Siddhartha and Shephard, Neil and Nakajima, Jouchi", year=2007,
        title="Stochastic Volatility with Leverage: Fast and Efficient Likelihood Inference",
        journal="Journal of Econometrics", volume="140", issue="2", pages="425--449",
        doi="10.1016/j.jeconom.2006.07.008", isbn="", publisher="", entry_type=:article),
    :giannone_lenza_primiceri2015 => (key=:giannone_lenza_primiceri2015,
        authors="Giannone, Domenico and Lenza, Michele and Primiceri, Giorgio E.", year=2015,
        title="Prior Selection for Vector Autoregressions",
        journal="Review of Economics and Statistics", volume="97", issue="2", pages="436--451",
        doi="10.1162/REST_a_00483", isbn="", publisher="", entry_type=:article),
    # --- Time Series Filters ---
    :hodrick_prescott1997 => (key=:hodrick_prescott1997,
        authors="Hodrick, Robert J. and Prescott, Edward C.", year=1997,
        title="Postwar U.S. Business Cycles: An Empirical Investigation",
        journal="Journal of Money, Credit and Banking", volume="29", issue="1", pages="1--16",
        doi="10.2307/2953682", isbn="", publisher="", entry_type=:article),
    :hamilton2018filter => (key=:hamilton2018filter,
        authors="Hamilton, James D.", year=2018,
        title="Why You Should Never Use the Hodrick-Prescott Filter",
        journal="Review of Economics and Statistics", volume="100", issue="5", pages="831--843",
        doi="10.1162/rest_a_00706", isbn="", publisher="", entry_type=:article),
    :beveridge_nelson1981 => (key=:beveridge_nelson1981,
        authors="Beveridge, Stephen and Nelson, Charles R.", year=1981,
        title="A New Approach to Decomposition of Economic Time Series into Permanent and Transitory Components with Particular Attention to Measurement of the `Business Cycle'",
        journal="Journal of Monetary Economics", volume="7", issue="2", pages="151--174",
        doi="10.1016/0304-3932(81)90040-4", isbn="", publisher="", entry_type=:article),
    :baxter_king1999 => (key=:baxter_king1999,
        authors="Baxter, Marianne and King, Robert G.", year=1999,
        title="Measuring Business Cycles: Approximate Band-Pass Filters for Economic Time Series",
        journal="Review of Economics and Statistics", volume="81", issue="4", pages="575--593",
        doi="10.1162/003465399558454", isbn="", publisher="", entry_type=:article),
    :phillips_shi2021 => (key=:phillips_shi2021,
        authors="Phillips, Peter C. B. and Shi, Zhentao", year=2021,
        title="Boosting: Why You Can Use the HP Filter",
        journal="International Economic Review", volume="62", issue="2", pages="521--570",
        doi="10.1111/iere.12495", isbn="", publisher="", entry_type=:article),
    :mei_phillips_shi2024 => (key=:mei_phillips_shi2024,
        authors="Mei, Ziwei and Phillips, Peter C. B. and Shi, Zhentao", year=2024,
        title="The boosted HP filter is more general than you might think",
        journal="Journal of Applied Econometrics", volume="39", issue="7", pages="1260--1281",
        doi="10.1002/jae.3086", isbn="", publisher="", entry_type=:article),
    # --- Model Comparison Tests ---
    :wilks1938 => (key=:wilks1938, authors="Wilks, Samuel S.", year=1938,
        title="The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses",
        journal="Annals of Mathematical Statistics", volume="9", issue="1", pages="60--62",
        doi="10.1214/aoms/1177732360", isbn="", publisher="", entry_type=:article),
    :neyman_pearson1933 => (key=:neyman_pearson1933, authors="Neyman, Jerzy and Pearson, Egon S.", year=1933,
        title="On the Problem of the Most Efficient Tests of Statistical Hypotheses",
        journal="Philosophical Transactions of the Royal Society A", volume="231", issue="694--706", pages="289--337",
        doi="10.1098/rsta.1933.0009", isbn="", publisher="", entry_type=:article),
    :rao1948 => (key=:rao1948, authors="Rao, C. Radhakrishna", year=1948,
        title="Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation",
        journal="Mathematical Proceedings of the Cambridge Philosophical Society", volume="44", issue="1", pages="50--57",
        doi="10.1017/S0305004100023987", isbn="", publisher="", entry_type=:article),
    :silvey1959 => (key=:silvey1959, authors="Silvey, S. D.", year=1959,
        title="The Lagrangian Multiplier Test",
        journal="Annals of Mathematical Statistics", volume="30", issue="2", pages="389--407",
        doi="10.1214/aoms/1177706259", isbn="", publisher="", entry_type=:article),
)

# --- Type/method → reference keys mapping ---

const _TYPE_REFS = Dict{Symbol, Vector{Symbol}}(
    # VAR
    :VARModel => [:sims1980, :lutkepohl2005],
    :ImpulseResponse => [:lutkepohl2005, :kilian1998],
    :BayesianImpulseResponse => [:lutkepohl2005, :kilian1998],
    :FEVD => [:lutkepohl2005],
    :BayesianFEVD => [:lutkepohl2005],
    :HistoricalDecomposition => [:kilian_lutkepohl2017],
    :BayesianHistoricalDecomposition => [:kilian_lutkepohl2017],
    :AriasSVARResult => [:arias_rubio_ramirez_waggoner2018],
    :SVARRestrictions => [:arias_rubio_ramirez_waggoner2018],
    # Bayesian VAR
    :MinnesotaHyperparameters => [:litterman1986, :kadiyala_karlsson1997],
    :BVARPosterior => [:litterman1986, :kadiyala_karlsson1997, :giannone_lenza_primiceri2015],
    :bvar => [:litterman1986, :kadiyala_karlsson1997, :giannone_lenza_primiceri2015],
    # Identification methods (symbol dispatch)
    :cholesky => [:sims1980, :lutkepohl2005],
    :long_run => [:blanchard_quah1989],
    :sign => [:uhlig2005],
    :narrative => [:antolin_diaz_rubio_ramirez2018],
    :arias => [:arias_rubio_ramirez_waggoner2018],
    # Local Projections
    :LPModel => [:jorda2005],
    :LPImpulseResponse => [:jorda2005],
    :LPIVModel => [:stock_watson2018],
    :SmoothLPModel => [:barnichon_brownlees2019],
    :StateLPModel => [:auerbach_gorodnichenko2012],
    :PropensityLPModel => [:angrist_jorda_kuersteiner2018],
    :StructuralLP => [:plagborg_moller_wolf2021, :jorda2005],
    :LPForecast => [:jorda2005],
    :LPFEVD => [:gorodnichenko_lee2020],
    # Factor Models
    :FactorModel => [:bai_ng2002, :stock_watson2002],
    :DynamicFactorModel => [:stock_watson2002],
    :GeneralizedDynamicFactorModel => [:stock_watson2002],
    :FactorForecast => [:stock_watson2002],
    # Unit Root Tests
    :ADFResult => [:dickey_fuller1979],
    :KPSSResult => [:kpss1992],
    :PPResult => [:phillips_perron1988],
    :ZAResult => [:zivot_andrews1992],
    :NgPerronResult => [:ng_perron2001],
    :JohansenResult => [:johansen1991],
    :adf => [:dickey_fuller1979],
    :kpss => [:kpss1992],
    :pp => [:phillips_perron1988],
    :za => [:zivot_andrews1992],
    :ngperron => [:ng_perron2001],
    :johansen => [:johansen1991],
    # VECM
    :VECMModel => [:johansen1991, :engle_granger1987, :lutkepohl2005],
    :VECMForecast => [:johansen1991, :lutkepohl2005],
    :VECMGrangerResult => [:johansen1991, :lutkepohl2005],
    :vecm => [:johansen1991, :engle_granger1987, :lutkepohl2005],
    :engle_granger => [:engle_granger1987],
    # ARIMA
    :ARModel => [:box_jenkins1970],
    :MAModel => [:box_jenkins1970],
    :ARMAModel => [:box_jenkins1970],
    :ARIMAModel => [:box_jenkins1970],
    :ARIMAForecast => [:box_jenkins1970],
    :ARIMAOrderSelection => [:hyndman_khandakar2008],
    :auto_arima => [:hyndman_khandakar2008],
    # GMM
    :GMMModel => [:hansen1982],
    :gmm => [:hansen1982],
    # Non-Gaussian ICA methods (symbol dispatch)
    :fastica => [:hyvarinen1999],
    :jade => [:cardoso_souloumiac1993],
    :sobi => [:belouchrani1997],
    :dcov => [:szekely_rizzo_bakirov2007, :matteson_tsay2017],
    :hsic => [:gretton2005],
    # Non-Gaussian ML methods (symbol dispatch)
    :student_t => [:lanne_meitz_saikkonen2017],
    :mixture_normal => [:lanne_meitz_saikkonen2017],
    :pml => [:lanne_meitz_saikkonen2017],
    :skew_normal => [:lanne_meitz_saikkonen2017],
    :nongaussian_ml => [:lanne_meitz_saikkonen2017],
    # Non-Gaussian result types
    :ICASVARResult => [:lanne_meitz_saikkonen2017],
    :NonGaussianMLResult => [:lanne_meitz_saikkonen2017],
    # Heteroskedastic identification
    :MarkovSwitchingSVARResult => [:rigobon2003],
    :GARCHSVARResult => [:rigobon2003],
    :SmoothTransitionSVARResult => [:lutkepohl_netsunajev2017],
    :ExternalVolatilitySVARResult => [:rigobon2003],
    :markov_switching => [:rigobon2003],
    :smooth_transition => [:lutkepohl_netsunajev2017],
    :external_volatility => [:rigobon2003],
    # Normality tests
    :NormalityTestResult => [:jarque_bera1980, :mardia1970],
    :NormalityTestSuite => [:jarque_bera1980, :mardia1970, :doornik_hansen2008, :henze_zirkler1990],
    :jarque_bera => [:jarque_bera1980],
    :mardia => [:mardia1970],
    :doornik_hansen => [:doornik_hansen2008],
    :henze_zirkler => [:henze_zirkler1990],
    # Covariance estimators
    :NeweyWestEstimator => [:newey_west1987],
    :WhiteEstimator => [:white1980],
    :newey_west => [:newey_west1987],
    :white => [:white1980],
    # Volatility models
    :ARCHModel => [:engle1982],
    :GARCHModel => [:bollerslev1986],
    :EGARCHModel => [:nelson1991],
    :GJRGARCHModel => [:glosten_jagannathan_runkle1993],
    :SVModel => [:taylor1986, :kim_shephard_chib1998, :omori2007],
    :VolatilityForecast => [:engle1982, :bollerslev1986],
    :arch => [:engle1982],
    :garch => [:bollerslev1986],
    :egarch => [:nelson1991],
    :gjr_garch => [:glosten_jagannathan_runkle1993],
    :sv => [:taylor1986, :kim_shephard_chib1998, :omori2007],
    # Time Series Filters
    :HPFilterResult => [:hodrick_prescott1997],
    :HamiltonFilterResult => [:hamilton2018filter],
    :BeveridgeNelsonResult => [:beveridge_nelson1981],
    :BaxterKingResult => [:baxter_king1999],
    :BoostedHPResult => [:phillips_shi2021, :mei_phillips_shi2024],
    :hp_filter => [:hodrick_prescott1997],
    :hamilton_filter => [:hamilton2018filter],
    :beveridge_nelson => [:beveridge_nelson1981],
    :baxter_king => [:baxter_king1999],
    :boosted_hp => [:phillips_shi2021, :mei_phillips_shi2024],
    # Model comparison tests
    :LRTestResult => [:wilks1938, :neyman_pearson1933],
    :LMTestResult => [:rao1948, :silvey1959],
    :lr_test => [:wilks1938, :neyman_pearson1933],
    :lm_test => [:rao1948, :silvey1959],
)

# ICA method → additional ref keys (appended to ICASVARResult base refs)
const _ICA_METHOD_REFS = Dict{Symbol, Vector{Symbol}}(
    :fastica => [:hyvarinen1999],
    :jade => [:cardoso_souloumiac1993],
    :sobi => [:belouchrani1997],
    :dcov => [:szekely_rizzo_bakirov2007, :matteson_tsay2017],
    :hsic => [:gretton2005],
)

# ML distribution → additional ref keys
const _ML_DIST_REFS = Dict{Symbol, Vector{Symbol}}(
    :student_t => Symbol[],
    :mixture_normal => Symbol[],
    :pml => Symbol[],
    :skew_normal => Symbol[],
)

# =============================================================================
# Format Functions
# =============================================================================

function _delatex(s::String)
    out = s
    out = replace(out, "\\\"u" => "\u00fc")  # ü
    out = replace(out, "\\\"a" => "\u00e4")  # ä
    out = replace(out, "\\\"o" => "\u00f6")  # ö
    out = replace(out, "\\\"A" => "\u00c4")  # Ä
    out = replace(out, "\\\\'\\i" => "\u00ed")  # í  (for Antolín-Díaz)
    out = replace(out, "\\\\'i" => "\u00ed")   # í
    out = replace(out, "\\`a" => "\u00e0")    # à
    out = replace(out, "{\\'e}" => "\u00e9")  # é
    out = replace(out, "\\'e" => "\u00e9")    # é
    out = replace(out, "{\\o}" => "\u00f8")   # ø
    out = replace(out, "\\o" => "\u00f8")     # ø
    out = replace(out, "{\\c{c}}" => "\u00e7")  # ç
    out = replace(out, "\\&" => "&")
    out = replace(out, "---" => "\u2014")     # em-dash
    out = replace(out, "--" => "\u2013")      # en-dash
    out = replace(out, r"\{|\}" => "")        # strip remaining braces
    out
end

function _format_ref_text(io::IO, r::_RefEntry)
    a = _delatex(r.authors)
    t = _delatex(r.title)
    if r.entry_type == :book
        println(io, "$a $(r.year). $t. $(r.publisher).")
        !isempty(r.isbn) && println(io, "  ISBN: $(r.isbn)")
    else
        j = _delatex(r.journal)
        vol_str = r.volume
        !isempty(r.issue) && (vol_str *= " ($(r.issue))")
        pages = _delatex(r.pages)
        println(io, "$a $(r.year). \"$t.\" $j $vol_str: $pages.")
        !isempty(r.doi) && println(io, "  DOI: https://doi.org/$(r.doi)")
    end
end

function _format_ref_latex(io::IO, r::_RefEntry)
    key = r.key
    if r.entry_type == :book
        println(io, "\\bibitem{$key} $(r.authors). $(r.year). \\textit{$(r.title)}. $(r.publisher).",
            !isempty(r.isbn) ? " ISBN: $(r.isbn)." : "")
    else
        vol_str = r.volume
        !isempty(r.issue) && (vol_str *= " ($(r.issue))")
        pages = r.pages
        doi_str = !isempty(r.doi) ? " \\url{https://doi.org/$(r.doi)}" : ""
        println(io, "\\bibitem{$key} $(r.authors). $(r.year). ``$(r.title).'' \\textit{$(r.journal)} $vol_str: $pages.$doi_str")
    end
end

function _format_ref_bibtex(io::IO, r::_RefEntry)
    key = r.key
    if r.entry_type == :book
        println(io, "@book{$key,")
        println(io, "  author    = {$(r.authors)},")
        println(io, "  title     = {$(r.title)},")
        println(io, "  year      = {$(r.year)},")
        println(io, "  publisher = {$(r.publisher)},")
        !isempty(r.isbn) && println(io, "  isbn      = {$(r.isbn)},")
        !isempty(r.doi) && println(io, "  doi       = {$(r.doi)},")
        println(io, "}")
    else
        etype = r.entry_type == :incollection ? "incollection" : "article"
        println(io, "@$etype{$key,")
        println(io, "  author  = {$(r.authors)},")
        println(io, "  title   = {$(r.title)},")
        btype = r.entry_type == :incollection ? "booktitle" : "journal"
        println(io, "  $btype = {$(r.journal)},")
        println(io, "  year    = {$(r.year)},")
        !isempty(r.volume) && println(io, "  volume  = {$(r.volume)},")
        !isempty(r.issue) && println(io, "  number  = {$(r.issue)},")
        !isempty(r.pages) && println(io, "  pages   = {$(r.pages)},")
        !isempty(r.doi) && println(io, "  doi     = {$(r.doi)},")
        println(io, "}")
    end
end

function _format_ref_html(io::IO, r::_RefEntry)
    a = _delatex(r.authors)
    t = _delatex(r.title)
    if r.entry_type == :book
        doi_link = !isempty(r.isbn) ? " ISBN: $(r.isbn)." : ""
        println(io, "<p>$a $(r.year). <em>$t</em>. $(r.publisher).$doi_link</p>")
    else
        j = _delatex(r.journal)
        vol_str = r.volume
        !isempty(r.issue) && (vol_str *= " ($(r.issue))")
        pages = _delatex(r.pages)
        doi_link = !isempty(r.doi) ? " <a href=\"https://doi.org/$(r.doi)\">DOI</a>" : ""
        println(io, "<p>$a $(r.year). &ldquo;$t.&rdquo; <em>$j</em> $vol_str: $pages.$doi_link</p>")
    end
end

function _format_ref(io::IO, r::_RefEntry, format::Symbol)
    if format == :text
        _format_ref_text(io, r)
    elseif format == :latex
        _format_ref_latex(io, r)
    elseif format == :bibtex
        _format_ref_bibtex(io, r)
    elseif format == :html
        _format_ref_html(io, r)
    else
        throw(ArgumentError("Unknown format: $format. Use :text, :latex, :bibtex, or :html."))
    end
end

# =============================================================================
# Public refs() Methods
# =============================================================================

"""
    refs([io::IO], x; format=get_display_backend())

Print bibliographic references for a model, result, or method.

Supports four output formats via the `format` keyword:
- `:text` — AEA plain text (default, follows `get_display_backend()`)
- `:latex` — `\\bibitem{}` entries
- `:bibtex` — BibTeX `@article{}`/`@book{}` entries
- `:html` — HTML with clickable DOI links

# Dispatch
- **Instance dispatch**: `refs(model)` prints references for the model type
- **Symbol dispatch**: `refs(:fastica)` prints references for a method name

# Examples
```julia
model = estimate_var(Y, 2)
refs(model)                        # AEA text to stdout
refs(model; format=:bibtex)        # BibTeX entries

refs(:johansen)                    # Johansen (1991)
refs(:fastica; format=:latex)      # Hyvärinen (1999) as \\bibitem
```
"""
function refs(io::IO, keys::Vector{Symbol}; format::Symbol=get_display_backend())
    format = format == :bibtex ? :bibtex : format  # :bibtex is extra, not in display backend
    for k in keys
        haskey(_REFERENCES, k) || throw(ArgumentError("Unknown reference key: $k"))
        _format_ref(io, _REFERENCES[k], format)
    end
end

# --- Symbol dispatch ---
function refs(io::IO, method::Symbol; format::Symbol=get_display_backend())
    haskey(_TYPE_REFS, method) || throw(ArgumentError("Unknown method/type: $method"))
    refs(io, _TYPE_REFS[method]; format=format)
end

# --- Instance dispatch: use type name to look up refs ---
function _refs_for_type(io::IO, x; format::Symbol=get_display_backend())
    tname = Symbol(nameof(typeof(x)))
    haskey(_TYPE_REFS, tname) || throw(ArgumentError("No references available for type: $tname"))
    refs(io, _TYPE_REFS[tname]; format=format)
end

# VAR types
refs(io::IO, ::VARModel; kw...) = refs(io, _TYPE_REFS[:VARModel]; kw...)
refs(io::IO, ::ImpulseResponse; kw...) = refs(io, _TYPE_REFS[:ImpulseResponse]; kw...)
refs(io::IO, ::BayesianImpulseResponse; kw...) = refs(io, _TYPE_REFS[:BayesianImpulseResponse]; kw...)
refs(io::IO, ::FEVD; kw...) = refs(io, _TYPE_REFS[:FEVD]; kw...)
refs(io::IO, ::BayesianFEVD; kw...) = refs(io, _TYPE_REFS[:BayesianFEVD]; kw...)
refs(io::IO, ::HistoricalDecomposition; kw...) = refs(io, _TYPE_REFS[:HistoricalDecomposition]; kw...)
refs(io::IO, ::BayesianHistoricalDecomposition; kw...) = refs(io, _TYPE_REFS[:BayesianHistoricalDecomposition]; kw...)
refs(io::IO, ::AriasSVARResult; kw...) = refs(io, _TYPE_REFS[:AriasSVARResult]; kw...)
refs(io::IO, ::SVARRestrictions; kw...) = refs(io, _TYPE_REFS[:SVARRestrictions]; kw...)
refs(io::IO, ::MinnesotaHyperparameters; kw...) = refs(io, _TYPE_REFS[:MinnesotaHyperparameters]; kw...)
refs(io::IO, ::BVARPosterior; kw...) = refs(io, _TYPE_REFS[:BVARPosterior]; kw...)

# LP types
refs(io::IO, ::LPModel; kw...) = refs(io, _TYPE_REFS[:LPModel]; kw...)
refs(io::IO, ::LPImpulseResponse; kw...) = refs(io, _TYPE_REFS[:LPImpulseResponse]; kw...)
refs(io::IO, ::LPIVModel; kw...) = refs(io, _TYPE_REFS[:LPIVModel]; kw...)
refs(io::IO, ::SmoothLPModel; kw...) = refs(io, _TYPE_REFS[:SmoothLPModel]; kw...)
refs(io::IO, ::StateLPModel; kw...) = refs(io, _TYPE_REFS[:StateLPModel]; kw...)
refs(io::IO, ::PropensityLPModel; kw...) = refs(io, _TYPE_REFS[:PropensityLPModel]; kw...)
refs(io::IO, ::StructuralLP; kw...) = refs(io, _TYPE_REFS[:StructuralLP]; kw...)
refs(io::IO, ::LPForecast; kw...) = refs(io, _TYPE_REFS[:LPForecast]; kw...)
refs(io::IO, ::LPFEVD; kw...) = refs(io, _TYPE_REFS[:LPFEVD]; kw...)

# Factor models
refs(io::IO, ::FactorModel; kw...) = refs(io, _TYPE_REFS[:FactorModel]; kw...)
refs(io::IO, ::DynamicFactorModel; kw...) = refs(io, _TYPE_REFS[:DynamicFactorModel]; kw...)
refs(io::IO, ::GeneralizedDynamicFactorModel; kw...) = refs(io, _TYPE_REFS[:GeneralizedDynamicFactorModel]; kw...)
refs(io::IO, ::FactorForecast; kw...) = refs(io, _TYPE_REFS[:FactorForecast]; kw...)

# Unit root tests
refs(io::IO, ::ADFResult; kw...) = refs(io, _TYPE_REFS[:ADFResult]; kw...)
refs(io::IO, ::KPSSResult; kw...) = refs(io, _TYPE_REFS[:KPSSResult]; kw...)
refs(io::IO, ::PPResult; kw...) = refs(io, _TYPE_REFS[:PPResult]; kw...)
refs(io::IO, ::ZAResult; kw...) = refs(io, _TYPE_REFS[:ZAResult]; kw...)
refs(io::IO, ::NgPerronResult; kw...) = refs(io, _TYPE_REFS[:NgPerronResult]; kw...)
refs(io::IO, ::JohansenResult; kw...) = refs(io, _TYPE_REFS[:JohansenResult]; kw...)

# VECM
refs(io::IO, ::VECMModel; kw...) = refs(io, _TYPE_REFS[:VECMModel]; kw...)
refs(io::IO, ::VECMForecast; kw...) = refs(io, _TYPE_REFS[:VECMForecast]; kw...)
refs(io::IO, ::VECMGrangerResult; kw...) = refs(io, _TYPE_REFS[:VECMGrangerResult]; kw...)

# ARIMA
refs(io::IO, ::ARModel; kw...) = refs(io, _TYPE_REFS[:ARModel]; kw...)
refs(io::IO, ::MAModel; kw...) = refs(io, _TYPE_REFS[:MAModel]; kw...)
refs(io::IO, ::ARMAModel; kw...) = refs(io, _TYPE_REFS[:ARMAModel]; kw...)
refs(io::IO, ::ARIMAModel; kw...) = refs(io, _TYPE_REFS[:ARIMAModel]; kw...)
refs(io::IO, ::ARIMAForecast; kw...) = refs(io, _TYPE_REFS[:ARIMAForecast]; kw...)
refs(io::IO, ::ARIMAOrderSelection; kw...) = refs(io, _TYPE_REFS[:ARIMAOrderSelection]; kw...)

# GMM
refs(io::IO, ::GMMModel; kw...) = refs(io, _TYPE_REFS[:GMMModel]; kw...)

# Volatility models
refs(io::IO, ::ARCHModel; kw...) = refs(io, _TYPE_REFS[:ARCHModel]; kw...)
refs(io::IO, ::GARCHModel; kw...) = refs(io, _TYPE_REFS[:GARCHModel]; kw...)
refs(io::IO, ::EGARCHModel; kw...) = refs(io, _TYPE_REFS[:EGARCHModel]; kw...)
refs(io::IO, ::GJRGARCHModel; kw...) = refs(io, _TYPE_REFS[:GJRGARCHModel]; kw...)
refs(io::IO, ::SVModel; kw...) = refs(io, _TYPE_REFS[:SVModel]; kw...)
refs(io::IO, ::VolatilityForecast; kw...) = refs(io, _TYPE_REFS[:VolatilityForecast]; kw...)

# Covariance estimators
refs(io::IO, ::NeweyWestEstimator; kw...) = refs(io, _TYPE_REFS[:NeweyWestEstimator]; kw...)
refs(io::IO, ::WhiteEstimator; kw...) = refs(io, _TYPE_REFS[:WhiteEstimator]; kw...)

# Normality tests
refs(io::IO, ::NormalityTestResult; kw...) = refs(io, _TYPE_REFS[:NormalityTestResult]; kw...)
refs(io::IO, ::NormalityTestSuite; kw...) = refs(io, _TYPE_REFS[:NormalityTestSuite]; kw...)

# Non-Gaussian types with variant-dependent refs
function refs(io::IO, r::ICASVARResult; format::Symbol=get_display_backend())
    base = _TYPE_REFS[:ICASVARResult]
    extra = get(_ICA_METHOD_REFS, r.method, Symbol[])
    refs(io, unique(vcat(base, extra)); format=format)
end

function refs(io::IO, r::NonGaussianMLResult; format::Symbol=get_display_backend())
    base = _TYPE_REFS[:NonGaussianMLResult]
    extra = get(_ML_DIST_REFS, r.distribution, Symbol[])
    refs(io, unique(vcat(base, extra)); format=format)
end

# Heteroskedastic types (concrete type dispatch, no method field)
refs(io::IO, ::MarkovSwitchingSVARResult; kw...) = refs(io, _TYPE_REFS[:MarkovSwitchingSVARResult]; kw...)
refs(io::IO, ::GARCHSVARResult; kw...) = refs(io, _TYPE_REFS[:GARCHSVARResult]; kw...)
refs(io::IO, ::SmoothTransitionSVARResult; kw...) = refs(io, _TYPE_REFS[:SmoothTransitionSVARResult]; kw...)
refs(io::IO, ::ExternalVolatilitySVARResult; kw...) = refs(io, _TYPE_REFS[:ExternalVolatilitySVARResult]; kw...)

# Identifiability test result
refs(io::IO, ::IdentifiabilityTestResult; kw...) = refs(io, [:lanne_meitz_saikkonen2017]; kw...)

# Time series filters
refs(io::IO, ::HPFilterResult; kw...) = refs(io, _TYPE_REFS[:HPFilterResult]; kw...)
refs(io::IO, ::HamiltonFilterResult; kw...) = refs(io, _TYPE_REFS[:HamiltonFilterResult]; kw...)
refs(io::IO, ::BeveridgeNelsonResult; kw...) = refs(io, _TYPE_REFS[:BeveridgeNelsonResult]; kw...)
refs(io::IO, ::BaxterKingResult; kw...) = refs(io, _TYPE_REFS[:BaxterKingResult]; kw...)
refs(io::IO, ::BoostedHPResult; kw...) = refs(io, _TYPE_REFS[:BoostedHPResult]; kw...)

# Model comparison tests
refs(io::IO, ::LRTestResult; kw...) = refs(io, _TYPE_REFS[:LRTestResult]; kw...)
refs(io::IO, ::LMTestResult; kw...) = refs(io, _TYPE_REFS[:LMTestResult]; kw...)

# --- Convenience: stdout fallback ---
refs(x; kw...) = refs(stdout, x; kw...)
