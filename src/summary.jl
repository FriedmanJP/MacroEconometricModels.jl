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

using LinearAlgebra, Statistics

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

Print comprehensive VAR model summary including specification, information
criteria, residual covariance, and stationarity check.
"""
function report(model::VARModel{T}) where {T}
    n, p = nvars(model), model.p
    T_eff = effective_nobs(model)

    spec_data = [
        "Variables" n;
        "Lags" p;
        "Observations (effective)" T_eff;
        "Parameters per equation" ncoefs(model)
    ]
    _pretty_table(stdout, spec_data;
        title = "Vector Autoregression Model",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    ic_data = ["AIC" _fmt(model.aic; digits=2);
               "BIC" _fmt(model.bic; digits=2);
               "HQIC" _fmt(model.hqic; digits=2)]
    _pretty_table(stdout, ic_data;
        title = "Information Criteria",
        column_labels = ["Criterion", "Value"],
        alignment = [:l, :r],
    )

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

    F = companion_matrix(model.B, n, p)
    max_mod = maximum(abs.(eigvals(F)))
    stable = max_mod < 1 ? "Yes" : "No"
    stab_data = Any["Stationary" stable; "Max |λ|" _fmt(max_mod)]
    _pretty_table(stdout, stab_data;
        title = "Stationarity",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

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
        col_labels = ["h", "IRF", "CI_lo", "CI_hi"]
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
            title = "Shock: $(irf.shocks[j]) (median)",
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
            push!(col_labels, "CI_lo")
            push!(col_labels, "CI_hi")
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
            push!(header_full, "CI_lo")
            push!(header_full, "CI_hi")
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
    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:5
            data[i, j] = _fmt(raw[i, j])
        end
    end
    ci_pct = round(Int, 100 * fc.conf_level)
    _pretty_table(io, data;
        title = "Volatility Forecast ($(fc.model_type), $(ci_pct)% CI)",
        column_labels = ["h", "σ² Forecast", "CI_lo", "CI_hi", "SE"],
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
    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:5
            data[i, j] = _fmt(raw[i, j])
        end
    end
    ci_pct = round(Int, 100 * fc.conf_level)
    _pretty_table(io, data;
        title = "ARIMA Forecast ($(ci_pct)% CI)",
        column_labels = ["h", "Forecast", "CI_lo", "CI_hi", "SE"],
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
        column_labels = ["h", "Forecast", "CI_lo", "CI_hi"],
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
        column_labels = ["h", "IRF", "SE", "CI_lo", "CI_hi"],
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
