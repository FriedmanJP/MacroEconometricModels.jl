"""
Publication-quality summary tables for VAR results.

Provides a unified interface using multiple dispatch:
- `summary(result)` - Print comprehensive summary
- `table(result, ...)` - Extract data as matrix
- `print_table(result, ...)` - Print formatted table
"""

using PrettyTables, LinearAlgebra, Statistics

# =============================================================================
# Table Formatting
# =============================================================================

const _TABLE_FORMAT = TextTableFormat(
    borders = text_table_borders__borderless,
    horizontal_line_after_column_labels = true
)

_fmt(x::Real; digits::Int=4) = round(x, digits=digits)
_fmt_pct(x::Real; digits::Int=1) = string(round(x * 100, digits=digits), "%")

"""Select representative horizons for display."""
function _select_horizons(H::Int)
    H <= 5 && return collect(1:H)
    H <= 12 && return [1, 4, 8, H]
    H <= 24 && return [1, 4, 8, 12, H]
    return [1, 4, 8, 12, 24, H]
end

# =============================================================================
# summary() - Comprehensive summaries
# =============================================================================

"""
    summary(model::VARModel)

Print comprehensive VAR model summary including specification, information
criteria, residual covariance, and stationarity check.
"""
function summary(model::VARModel{T}) where {T}
    n, p = nvars(model), model.p
    T_eff = effective_nobs(model)

    println("")
    println("Vector Autoregression Model")
    println("══════════════════════════════════════════════════════════════")

    spec_data = [
        "Variables" n;
        "Lags" p;
        "Observations (effective)" T_eff;
        "Parameters per equation" ncoefs(model)
    ]
    pretty_table(stdout, spec_data;
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
        table_format = _TABLE_FORMAT
    )

    println("")
    ic_data = ["AIC" _fmt(model.aic; digits=2);
               "BIC" _fmt(model.bic; digits=2);
               "HQIC" _fmt(model.hqic; digits=2)]
    pretty_table(stdout, ic_data;
        column_labels = ["Information Criteria", "Value"],
        alignment = [:l, :r],
        table_format = _TABLE_FORMAT
    )

    println("")
    println("Residual Covariance (Σ)")
    Sigma_data = Matrix{Any}(undef, n, n + 1)
    for i in 1:n
        Sigma_data[i, 1] = "Var $i"
        for j in 1:n
            Sigma_data[i, j + 1] = _fmt(model.Sigma[i, j])
        end
    end
    pretty_table(stdout, Sigma_data;
        column_labels = vcat([""], ["Var $j" for j in 1:n]),
        alignment = vcat([:l], fill(:r, n)),
        table_format = _TABLE_FORMAT
    )

    F = companion_matrix(model.B, n, p)
    max_mod = maximum(abs.(eigvals(F)))
    println("")
    println("Stationarity: ", max_mod < 1 ? "Yes" : "No", " (max |λ| = ", _fmt(max_mod), ")")
    println("──────────────────────────────────────────────────────────────")
end

"""
    summary(irf::ImpulseResponse)
    summary(irf::BayesianImpulseResponse)

Print IRF summary with values at selected horizons.
"""
function summary(irf::ImpulseResponse{T}) where {T}
    show(stdout, irf)
end

function summary(irf::BayesianImpulseResponse{T}) where {T}
    show(stdout, irf)
end

"""
    summary(f::FEVD)
    summary(f::BayesianFEVD)

Print FEVD summary with decomposition at selected horizons.
"""
function summary(f::FEVD{T}) where {T}
    show(stdout, f)
end

function summary(f::BayesianFEVD{T}) where {T}
    show(stdout, f)
end

"""
    summary(hd::HistoricalDecomposition)
    summary(hd::BayesianHistoricalDecomposition)

Print HD summary with contribution statistics.
"""
function summary(hd::HistoricalDecomposition{T}) where {T}
    show(stdout, hd)
end

function summary(hd::BayesianHistoricalDecomposition{T}) where {T}
    show(stdout, hd)
end

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
    vi = findfirst(==(var), irf.variables)
    si = findfirst(==(shock), irf.shocks)
    isnothing(vi) && throw(ArgumentError("Variable '$var' not found"))
    isnothing(si) && throw(ArgumentError("Shock '$shock' not found"))
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
    vi = findfirst(==(var), irf.variables)
    si = findfirst(==(shock), irf.shocks)
    isnothing(vi) && throw(ArgumentError("Variable '$var' not found"))
    isnothing(si) && throw(ArgumentError("Shock '$shock' not found"))
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

    println(io, "")
    println(io, "IRF: $(irf.variables[var]) ← $(irf.shocks[shock])")
    println(io, "────────────────────────────────────────")

    if has_ci
        col_labels = ["h", "IRF", "CI_lo", "CI_hi"]
    else
        col_labels = ["h", "IRF"]
    end

    pretty_table(io, raw;
        column_labels = col_labels,
        alignment = fill(:r, size(raw, 2)),
        table_format = _TABLE_FORMAT
    )
end

print_table(irf::ImpulseResponse, var, shock; kwargs...) =
    print_table(stdout, irf, var, shock; kwargs...)

function print_table(io::IO, irf::BayesianImpulseResponse{T}, var::Int, shock::Int;
                     horizons::Union{Nothing,AbstractVector{Int}}=nothing) where {T}
    raw = table(irf, var, shock; horizons=horizons)

    println(io, "")
    println(io, "Bayesian IRF: $(irf.variables[var]) ← $(irf.shocks[shock])")
    println(io, "────────────────────────────────────────")

    q_labels = [_fmt_pct(q; digits=0) for q in irf.quantile_levels]
    col_labels = vcat(["h", "Mean"], q_labels)

    pretty_table(io, raw;
        column_labels = col_labels,
        alignment = fill(:r, size(raw, 2)),
        table_format = _TABLE_FORMAT
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

    println(io, "")
    println(io, "FEVD: Variable $var")
    println(io, "────────────────────────────────────────")

    # Format percentages
    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:size(raw, 2)
            data[i, j] = _fmt_pct(raw[i, j])
        end
    end

    col_labels = vcat(["h"], ["Shock $j" for j in 1:n_shocks])
    pretty_table(io, data;
        column_labels = col_labels,
        alignment = fill(:r, size(data, 2)),
        table_format = _TABLE_FORMAT
    )
end

print_table(f::FEVD, var; kwargs...) = print_table(stdout, f, var; kwargs...)

function print_table(io::IO, f::BayesianFEVD{T}, var::Int;
                     horizons::Union{Nothing,AbstractVector{Int}}=nothing,
                     stat::Union{Symbol,Int}=:mean) where {T}
    raw = table(f, var; horizons=horizons, stat=stat)
    n_shocks = length(f.shocks)

    stat_name = stat == :mean ? "mean" : _fmt_pct(f.quantile_levels[stat]; digits=0)
    println(io, "")
    println(io, "Bayesian FEVD: $(f.variables[var]) ($stat_name)")
    println(io, "────────────────────────────────────────")

    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:size(raw, 2)
            data[i, j] = _fmt_pct(raw[i, j])
        end
    end

    col_labels = vcat(["h"], f.shocks)
    pretty_table(io, data;
        column_labels = col_labels,
        alignment = fill(:r, size(data, 2)),
        table_format = _TABLE_FORMAT
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

    println(io, "")
    println(io, "Historical Decomposition: $(hd.variables[var])")
    println(io, "────────────────────────────────────────────────────────")

    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:size(raw, 2)
            data[i, j] = _fmt(raw[i, j])
        end
    end

    col_labels = vcat(["t", "Actual"], hd.shock_names, ["Initial"])
    pretty_table(io, data;
        column_labels = col_labels,
        alignment = fill(:r, size(data, 2)),
        table_format = _TABLE_FORMAT
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
    println(io, "")
    println(io, "Bayesian HD: $(hd.variables[var]) ($stat_name)")
    println(io, "────────────────────────────────────────────────────────")

    data = Matrix{Any}(undef, size(raw)...)
    for i in axes(raw, 1)
        data[i, 1] = Int(raw[i, 1])
        for j in 2:size(raw, 2)
            data[i, j] = _fmt(raw[i, j])
        end
    end

    col_labels = vcat(["t", "Actual"], hd.shock_names, ["Initial"])
    pretty_table(io, data;
        column_labels = col_labels,
        alignment = fill(:r, size(data, 2)),
        table_format = _TABLE_FORMAT
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

    println(io, "")
    println(io, "Impulse Response Functions")
    println(io, "══════════════════════════════════════════════════════════════")

    ci_str = irf.ci_type == :none ? "None" : string(irf.ci_type)
    spec_data = ["Variables" n_vars; "Shocks" n_shocks; "Horizon" H; "CI" ci_str]
    pretty_table(io, spec_data;
        column_labels = ["", ""],
        alignment = [:l, :r],
        table_format = _TABLE_FORMAT
    )

    horizons_show = _select_horizons(H)
    for j in 1:n_shocks
        println(io, "")
        println(io, "Shock: ", irf.shocks[j])

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

        pretty_table(io, data;
            column_labels = vcat([""], ["h=$h" for h in horizons_show]),
            alignment = vcat([:l], fill(:r, length(horizons_show))),
            table_format = _TABLE_FORMAT
        )
    end

    println(io, "──────────────────────────────────────────────────────────────")
    irf.ci_type != :none && println(io, "* CI excludes zero")
end

function Base.show(io::IO, irf::BayesianImpulseResponse{T}) where {T}
    n_vars, n_shocks = length(irf.variables), length(irf.shocks)
    H = irf.horizon
    nq = length(irf.quantile_levels)

    println(io, "")
    println(io, "Bayesian Impulse Response Functions")
    println(io, "══════════════════════════════════════════════════════════════")

    q_str = join([_fmt_pct(q; digits=0) for q in irf.quantile_levels], ", ")
    spec_data = ["Variables" n_vars; "Shocks" n_shocks; "Horizon" H; "Quantiles" q_str]
    pretty_table(io, spec_data;
        column_labels = ["", ""],
        alignment = [:l, :r],
        table_format = _TABLE_FORMAT
    )

    horizons_show = _select_horizons(H)
    median_idx = nq >= 3 ? 2 : 1

    for j in 1:n_shocks
        println(io, "")
        println(io, "Shock: ", irf.shocks[j], " (median)")

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

        pretty_table(io, data;
            column_labels = vcat([""], ["h=$h" for h in horizons_show]),
            alignment = vcat([:l], fill(:r, length(horizons_show))),
            table_format = _TABLE_FORMAT
        )
    end

    println(io, "──────────────────────────────────────────────────────────────")
    println(io, "* Credible interval excludes zero")
end

function Base.show(io::IO, f::FEVD{T}) where {T}
    n_vars, n_shocks, H = size(f.proportions)

    println(io, "")
    println(io, "Forecast Error Variance Decomposition")
    println(io, "══════════════════════════════════════════════════════════════")

    spec_data = ["Variables" n_vars; "Shocks" n_shocks; "Horizon" H]
    pretty_table(io, spec_data;
        column_labels = ["", ""],
        alignment = [:l, :r],
        table_format = _TABLE_FORMAT
    )

    for h in _select_horizons(H)
        println(io, "")
        println(io, "h = $h")

        data = Matrix{Any}(undef, n_vars, n_shocks + 1)
        for i in 1:n_vars
            data[i, 1] = "Var $i"
            for j in 1:n_shocks
                data[i, j + 1] = _fmt_pct(f.proportions[i, j, h])
            end
        end

        pretty_table(io, data;
            column_labels = vcat([""], ["Shock $j" for j in 1:n_shocks]),
            alignment = vcat([:l], fill(:r, n_shocks)),
            table_format = _TABLE_FORMAT
        )
    end

    println(io, "──────────────────────────────────────────────────────────────")
end

function Base.show(io::IO, f::BayesianFEVD{T}) where {T}
    n_vars, n_shocks = length(f.variables), length(f.shocks)
    H = f.horizon

    println(io, "")
    println(io, "Bayesian FEVD (posterior mean)")
    println(io, "══════════════════════════════════════════════════════════════")

    q_str = join([_fmt_pct(q; digits=0) for q in f.quantile_levels], ", ")
    spec_data = ["Variables" n_vars; "Shocks" n_shocks; "Horizon" H; "Quantiles" q_str]
    pretty_table(io, spec_data;
        column_labels = ["", ""],
        alignment = [:l, :r],
        table_format = _TABLE_FORMAT
    )

    for h in _select_horizons(H)
        println(io, "")
        println(io, "h = $h")

        data = Matrix{Any}(undef, n_vars, n_shocks + 1)
        for i in 1:n_vars
            data[i, 1] = f.variables[i]
            for j in 1:n_shocks
                data[i, j + 1] = _fmt_pct(f.mean[h, i, j])
            end
        end

        pretty_table(io, data;
            column_labels = vcat([""], f.shocks),
            alignment = vcat([:l], fill(:r, n_shocks)),
            table_format = _TABLE_FORMAT
        )
    end

    println(io, "──────────────────────────────────────────────────────────────")
end
