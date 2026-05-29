# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Display and report methods for Heterogeneous Agent DSGE types:
`HASteadyState`, `HADSGESolution`, `KrusellSmithSolution`.

Provides compact `show()` one-liners and detailed `report()` output
with PrettyTables formatting.
"""

# =============================================================================
# Distribution statistics helpers
# =============================================================================

"""
    _gini_coefficient(d::Vector{T}, grid::HAGrid{T}) -> T

Compute the Gini coefficient from the wealth distribution over a one-asset grid.

Sorts wealth levels by grid position, weights by distribution mass, and computes
Gini = 1 - 2 * (area under the Lorenz curve). Returns a value in [0, 1].
"""
function _gini_coefficient(d::Vector{T}, grid::HAGrid{T}) where {T<:AbstractFloat}
    a_grid = grid.grids[1]
    n_a = grid.n_points[1]
    n_e = grid.n_income

    # Aggregate distribution over income states: marginal over assets
    n_total = n_a * n_e
    length(d) == n_total || throw(ArgumentError(
        "Distribution length $(length(d)) does not match grid total $n_total"))

    # Reshape distribution: (n_a, n_e), sum across income to get marginal asset dist
    d_mat = reshape(d, n_a, n_e)
    d_asset = vec(sum(d_mat; dims=2))  # n_a-length marginal distribution

    # Sort by asset level (grid is already sorted, but be explicit)
    perm = sortperm(a_grid)
    a_sorted = a_grid[perm]
    d_sorted = d_asset[perm]

    # Normalize distribution mass
    total_mass = sum(d_sorted)
    if total_mass <= zero(T)
        return zero(T)
    end
    d_sorted ./= total_mass

    # Compute mean wealth
    mean_wealth = dot(d_sorted, a_sorted)

    if mean_wealth <= zero(T)
        return zero(T)
    end

    # Lorenz curve area via trapezoidal rule
    cum_pop = zero(T)
    cum_wealth = zero(T)
    lorenz_area = zero(T)

    for i in eachindex(d_sorted)
        prev_cum_pop = cum_pop
        prev_cum_wealth = cum_wealth
        cum_pop += d_sorted[i]
        cum_wealth += d_sorted[i] * a_sorted[i]
        # Trapezoidal area under Lorenz curve
        lorenz_area += (cum_wealth / mean_wealth + prev_cum_wealth / mean_wealth) *
                       (cum_pop - prev_cum_pop) / T(2)
    end

    gini = one(T) - T(2) * lorenz_area
    return clamp(gini, zero(T), one(T))
end

"""
    _wealth_percentile(d::Vector{T}, grid::HAGrid{T}, p::T) -> T

Compute the `p`-th percentile (0 < p < 1) of the wealth distribution.

Uses the marginal asset distribution (summed over income states) and
finds the asset level at which cumulative mass reaches `p`.
"""
function _wealth_percentile(d::Vector{T}, grid::HAGrid{T}, p::T) where {T<:AbstractFloat}
    @assert zero(T) < p < one(T) "Percentile must be in (0, 1)"

    a_grid = grid.grids[1]
    n_a = grid.n_points[1]
    n_e = grid.n_income

    n_total = n_a * n_e
    length(d) == n_total || throw(ArgumentError(
        "Distribution length $(length(d)) does not match grid total $n_total"))

    # Marginal asset distribution
    d_mat = reshape(d, n_a, n_e)
    d_asset = vec(sum(d_mat; dims=2))

    # Sort by asset level
    perm = sortperm(a_grid)
    a_sorted = a_grid[perm]
    d_sorted = d_asset[perm]

    # Normalize
    total_mass = sum(d_sorted)
    if total_mass <= zero(T)
        return a_sorted[1]
    end
    d_sorted ./= total_mass

    # Walk cumulative distribution
    cum = zero(T)
    for i in eachindex(d_sorted)
        cum += d_sorted[i]
        if cum >= p
            # Linear interpolation within bin
            if i > 1 && d_sorted[i] > zero(T)
                prev_cum = cum - d_sorted[i]
                frac = (p - prev_cum) / d_sorted[i]
                return a_sorted[i-1] + frac * (a_sorted[i] - a_sorted[i-1])
            end
            return a_sorted[i]
        end
    end

    return a_sorted[end]
end

# Allow Real argument for convenience
_wealth_percentile(d::Vector{T}, grid::HAGrid{T}, p::Real) where {T<:AbstractFloat} =
    _wealth_percentile(d, grid, T(p))

# =============================================================================
# Base.show — compact one-line displays
# =============================================================================

function Base.show(io::IO, ss::HASteadyState{T}) where {T}
    r_val = get(ss.prices, :r, NaN)
    K_val = get(ss.aggregates, :K, NaN)
    gini = _gini_coefficient(vec(ss.distribution), ss.grid)
    print(io, "HASteadyState{$T}: converged=$(ss.converged), " *
              "r=$(_fmt(r_val)), K=$(_fmt(K_val)), Gini=$(_fmt(gini; digits=2))")
end

function Base.show(io::IO, sol::HADSGESolution{T}) where {T}
    print(io, "HADSGESolution{$T}: method=:$(sol.method), " *
              "n_reduced=$(sol.n_reduced), " *
              "explained=$(_fmt_pct(sol.explained_variance))")
end

function Base.show(io::IO, sol::KrusellSmithSolution{T}) where {T}
    r2_val = if !isempty(sol.r_squared)
        _fmt(first(values(sol.r_squared)))
    else
        "N/A"
    end
    print(io, "KrusellSmithSolution{$T}: converged=$(sol.converged), R²=$r2_val")
end

# =============================================================================
# report() — detailed summaries
# =============================================================================

"""
    report(ss::HASteadyState)

Print a detailed summary of the heterogeneous agent steady state including
convergence diagnostics, equilibrium prices, aggregate quantities, and
wealth distribution statistics.
"""
function report(ss::HASteadyState{T}) where {T}
    io = stdout

    # --- Header ---
    _pretty_table(io, Any["" ""];
        title="Heterogeneous Agent Steady State",
        column_labels=["", ""],
        alignment=[:l, :r])

    # --- Convergence ---
    conv_data = Any[
        "Converged"       ss.converged ? "Yes" : "No";
        "Iterations"      ss.iterations;
        "Excess demand"   _fmt(ss.excess_demand; digits=6);
        "Euler error"     _fmt(ss.euler_error; digits=4)
    ]
    _pretty_table(io, conv_data;
        title="Convergence",
        column_labels=["", "Value"],
        alignment=[:l, :r])

    # --- Prices ---
    price_keys = sort(collect(keys(ss.prices)))
    if !isempty(price_keys)
        price_data = Matrix{Any}(undef, length(price_keys), 2)
        for (i, k) in enumerate(price_keys)
            price_data[i, 1] = string(k)
            price_data[i, 2] = _fmt(ss.prices[k]; digits=6)
        end
        _pretty_table(io, price_data;
            title="Prices",
            column_labels=["Variable", "Value"],
            alignment=[:l, :r])
    end

    # --- Aggregates ---
    agg_keys = sort(collect(keys(ss.aggregates)))
    if !isempty(agg_keys)
        agg_data = Matrix{Any}(undef, length(agg_keys), 2)
        for (i, k) in enumerate(agg_keys)
            agg_data[i, 1] = string(k)
            agg_data[i, 2] = _fmt(ss.aggregates[k]; digits=4)
        end
        _pretty_table(io, agg_data;
            title="Aggregates",
            column_labels=["Variable", "Value"],
            alignment=[:l, :r])
    end

    # --- Distribution statistics ---
    d_vec = vec(ss.distribution)
    gini = _gini_coefficient(d_vec, ss.grid)

    # Mean wealth
    a_grid = ss.grid.grids[1]
    n_a = ss.grid.n_points[1]
    n_e = ss.grid.n_income
    d_mat = reshape(d_vec, n_a, n_e)
    d_asset = vec(sum(d_mat; dims=2))
    total_mass = sum(d_asset)
    if total_mass > zero(T)
        d_asset ./= total_mass
    end
    mean_wealth = dot(d_asset, a_grid)

    # Percentiles
    pctiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    pctile_vals = [_wealth_percentile(d_vec, ss.grid, T(p)) for p in pctiles]

    dist_data = Matrix{Any}(undef, 2 + length(pctiles), 2)
    dist_data[1, 1] = "Mean wealth"
    dist_data[1, 2] = _fmt(mean_wealth; digits=4)
    dist_data[2, 1] = "Gini coefficient"
    dist_data[2, 2] = _fmt(gini; digits=4)
    for (i, (p, v)) in enumerate(zip(pctiles, pctile_vals))
        dist_data[2 + i, 1] = "P$(round(Int, p*100))"
        dist_data[2 + i, 2] = _fmt(v; digits=4)
    end
    _pretty_table(io, dist_data;
        title="Wealth Distribution",
        column_labels=["Statistic", "Value"],
        alignment=[:l, :r])

    return nothing
end

"""
    report(sol::HADSGESolution)

Print a detailed summary of the linearized HA-DSGE solution including
method, dimensionality reduction info, and the underlying steady state.
"""
function report(sol::HADSGESolution{T}) where {T}
    io = stdout

    red_data = Any[
        "Method"              string(sol.method);
        "Full state dim"      sol.n_full_states;
        "Reduced state dim"   sol.n_reduced;
        "Explained variance"  _fmt_pct(sol.explained_variance)
    ]
    _pretty_table(io, red_data;
        title="HA-DSGE Solution (Linearized)",
        column_labels=["", "Value"],
        alignment=[:l, :r])

    # Delegate to steady state report
    report(sol.steady_state)

    return nothing
end

"""
    report(sol::KrusellSmithSolution)

Print a detailed summary of the Krusell-Smith solution including PLM
coefficients, R² values, convergence, and the underlying steady state.
"""
function report(sol::KrusellSmithSolution{T}) where {T}
    io = stdout

    # --- Header ---
    conv_data = Any[
        "Converged"    sol.converged ? "Yes" : "No";
        "Iterations"   sol.iterations
    ]
    _pretty_table(io, conv_data;
        title="Krusell-Smith Solution",
        column_labels=["", "Value"],
        alignment=[:l, :r])

    # --- PLM coefficients and R² ---
    plm_keys = sort(collect(keys(sol.plm_coefficients)))
    if !isempty(plm_keys)
        # Determine max number of coefficients across variables
        max_coefs = maximum(length(sol.plm_coefficients[k]) for k in plm_keys)
        n_rows = length(plm_keys)
        # Columns: variable, coef_1, ..., coef_n, R²
        n_cols = 2 + max_coefs
        plm_data = Matrix{Any}(undef, n_rows, n_cols)
        col_labels = vcat(["Variable"],
                          ["b$i" for i in 0:(max_coefs-1)],
                          ["R²"])
        for (i, k) in enumerate(plm_keys)
            plm_data[i, 1] = string(k)
            coefs = sol.plm_coefficients[k]
            for j in 1:max_coefs
                plm_data[i, 1 + j] = j <= length(coefs) ? _fmt(coefs[j]; digits=6) : ""
            end
            plm_data[i, end] = haskey(sol.r_squared, k) ? _fmt(sol.r_squared[k]; digits=6) : "N/A"
        end
        _pretty_table(io, plm_data;
            title="Perceived Law of Motion",
            column_labels=col_labels,
            alignment=vcat([:l], fill(:r, n_cols - 1)))
    end

    # Delegate to steady state report
    report(sol.steady_state)

    return nothing
end
