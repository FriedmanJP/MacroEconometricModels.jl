# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Analysis functions for Heterogeneous Agent DSGE models.

Provides:
- `irf`, `fevd`, `simulate` dispatch for `HADSGESolution` (delegates to linear_solution)
- `distribution_irf` — wealth distribution dynamics after an aggregate shock
- `inequality_irf` — Gini and percentile responses over time
- `simulate_panel` — individual-level panel simulation from steady-state policies

# References
- Reiter, M. (2009). Solving heterogeneous-agent models by projection and
  perturbation. *Journal of Economic Dynamics and Control*, 33(3), 649–665.
- Krusell, P., & Smith, A. A. (1998). Income and wealth heterogeneity in the
  macroeconomy. *Journal of Political Economy*, 106(5), 867–896.
"""

using Random

# =============================================================================
# irf / fevd / simulate — delegate to linear_solution
# =============================================================================

# Names of the aggregate outputs carried by the Ho-Kalman observation map C_obs.
# SSJ realizes a single aggregate (the general r→K path reports K; Huggett clears
# and reports the rate r); Reiter uses the identity map over its reduced system
# (K and Z are explicit states), so its coordinate names are reported.
function _ha_obs_names(sol::HADSGESolution)
    n_out = size(sol.C_obs, 1)
    if sol.method === :ssj
        name = sol.spec.model === :huggett ? "r" : "K"
        return n_out == 1 ? [name] : ["obs_$i" for i in 1:n_out]
    else
        endog = sol.linear_solution.spec.endog
        return n_out == length(endog) ? [string(s) for s in endog] :
               ["x_$i" for i in 1:n_out]
    end
end

"""
    irf(sol::HADSGESolution{T}, horizon::Int; kwargs...) -> ImpulseResponse{T}

Impulse responses of the **aggregate outputs** of a linearized HA-DSGE solution.

The Ho-Kalman/Reiter reduced system is `x_{t+1} = A x_t + B ε_t`, `y_t = C_obs x_t
+ D_obs ε_t` (`x_0 = 0`). The reported aggregate response reproduces the realized
IRF sequence `h[k]` of the output: the impact (`h=1`) is the direct feed-through
`D_obs = h[0]`, and horizon `h ≥ 2` is `C_obs · A^(h-2) · B = h[h-1]` — i.e. in
`(K, r, Y, …)` coordinates, not the abstract reduced state `x_1..x_n` the raw
`linear_solution` would return.
"""
function irf(sol::HADSGESolution{T}, horizon::Int;
             ci_type::Symbol=:none, kwargs...) where {T<:AbstractFloat}
    lsol = sol.linear_solution
    G1 = lsol.G1
    B = lsol.impact
    n_state = size(G1, 1)
    n_shocks = size(B, 2)
    C = sol.C_obs
    D = sol.D_obs
    n_out = size(C, 1)

    point_irf = zeros(T, horizon, n_out, n_shocks)
    Gpow = Matrix{T}(I, n_state, n_state)   # at horizon h>=2, Gpow = G1^(h-2)
    for h in 1:horizon
        if h == 1
            Y_h = D                          # impact = h[0] = D_obs feed-through
        else
            Y_h = C * (Gpow * B)             # h[h-1] = C·A^(h-2)·B
            Gpow = Gpow * G1
        end
        for j in 1:n_shocks
            point_irf[h, :, j] = Y_h[:, j]
        end
    end

    var_names = _ha_obs_names(sol)
    shock_names = [string(s) for s in lsol.spec.exog]
    length(shock_names) == n_shocks || (shock_names = ["shock_$j" for j in 1:n_shocks])
    ci_lower = zeros(T, horizon, n_out, n_shocks)
    ci_upper = zeros(T, horizon, n_out, n_shocks)
    return ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                              var_names, shock_names, ci_type)
end

"""
    fevd(sol::HADSGESolution{T}, horizon::Int; kwargs...) -> FEVD{T}

Forecast error variance decomposition of the aggregate outputs (built from the
observation-mapped `irf`).
"""
function fevd(sol::HADSGESolution{T}, horizon::Int; kwargs...) where {T<:AbstractFloat}
    ir = irf(sol, horizon)
    n_vars = length(ir.variables)
    n_e = length(ir.shocks)

    decomp = zeros(T, n_vars, n_e, horizon)
    props  = zeros(T, n_vars, n_e, horizon)
    @inbounds for h in 1:horizon
        for i in 1:n_vars
            total = zero(T)
            for j in 1:n_e
                prev = h == 1 ? zero(T) : decomp[i, j, h-1]
                decomp[i, j, h] = prev + ir.values[h, i, j]^2
                total += decomp[i, j, h]
            end
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end
    return FEVD{T}(decomp, props, ir.variables, ir.shocks)
end

"""
    simulate(sol::HADSGESolution{T}, T_periods::Int; kwargs...) -> Matrix{T}

Simulate the aggregate-output deviation path `y_t = C_obs x_t + D_obs ε_t` of the
reduced HA-DSGE system. Returns a `T_periods × n_out` matrix of aggregate
deviations. `shock_draws`/`rng` are forwarded.
"""
function simulate(sol::HADSGESolution{T}, T_periods::Int;
                  shock_draws::Union{Nothing,AbstractMatrix}=nothing,
                  rng=Random.default_rng()) where {T<:AbstractFloat}
    lsol = sol.linear_solution
    G1 = lsol.G1
    B = lsol.impact
    n_state = size(G1, 1)
    n_shocks = size(B, 2)
    C = sol.C_obs
    D = sol.D_obs
    n_out = size(C, 1)

    if shock_draws !== nothing
        @assert size(shock_draws) == (T_periods, n_shocks) "shock_draws must be ($T_periods, $n_shocks)"
        e = T.(shock_draws)
    else
        e = randn(rng, T, T_periods, n_shocks)
    end

    x = zeros(T, n_state)                      # x_0 = 0
    out = zeros(T, T_periods, n_out)
    for t in 1:T_periods
        e_t = e[t, :]
        out[t, :] = C * x .+ D * e_t           # y_t = C·x_t + D·ε_t (feed-through, x pre-shock)
        x = G1 * x .+ B * e_t                  # x_{t+1} = A·x_t + B·ε_t
    end
    return out
end

# =============================================================================
# distribution_irf — wealth distribution dynamics after an aggregate shock
# =============================================================================

"""
    distribution_irf(sol::HADSGESolution{T}, horizon::Int;
                     shock_index=1, shock_size=1.0) -> Array{T,3}

Compute how the wealth distribution evolves after a one-standard-deviation
aggregate shock.

Returns an `N_a × N_e × horizon` array of distribution deviations from
steady state.

# Algorithm
Extracts the distribution dynamics from the SVD reduction basis:
`d_deviation_t = U_k * d_tilde_t`, where `d_tilde` evolves according to
the reduced system in `G1`.

# Arguments
- `sol::HADSGESolution{T}` — linearized HA-DSGE solution
- `horizon::Int` — number of periods to track

# Keyword Arguments
- `shock_index::Int` — which aggregate shock to hit (default 1)
- `shock_size::Real` — shock magnitude in standard deviations (default 1.0)
"""
function distribution_irf(sol::HADSGESolution{T}, horizon::Int;
                           shock_index::Int=1,
                           shock_size::Real=1.0) where {T<:AbstractFloat}
    lsol = sol.linear_solution
    G1 = lsol.G1
    impact = lsol.impact
    n_state = size(G1, 1)
    n_shocks = size(impact, 2)

    @assert 1 <= shock_index <= n_shocks "shock_index must be in 1:$n_shocks"

    # Simulate the reduced state path after a one-time shock at t=1
    state = zeros(T, n_state)
    state .= impact[:, shock_index] .* T(shock_size)

    grid = sol.steady_state.grid
    n_a = grid.n_points[1]
    n_e = grid.n_income

    # The reduction_basis is U_k: maps from reduced distribution coordinates
    # back to full distribution space. The first n_reduced columns of the
    # state vector correspond to distribution deviations (in reduced form).
    U_k = sol.reduction_basis
    n_red = sol.n_reduced

    # Output array
    dist_irf = zeros(T, n_a, n_e, horizon)

    for h in 1:horizon
        # Extract the reduced distribution portion of the state vector
        # The reduced distribution coordinates occupy the first n_red elements
        n_dist_coords = min(n_red, n_state)
        d_tilde = state[1:n_dist_coords]

        # Project back to full distribution space
        n_full = size(U_k, 1)
        if n_full == n_a * n_e && size(U_k, 2) >= n_dist_coords
            d_deviation = U_k[:, 1:n_dist_coords] * d_tilde
            # Reshape into (n_a, n_e)
            dist_irf[:, :, h] = reshape(d_deviation[1:min(length(d_deviation), n_a * n_e)],
                                         n_a, n_e)
        end

        # Advance state one period (no further shocks)
        state = G1 * state
    end

    return dist_irf
end

# =============================================================================
# inequality_irf — Gini and percentile responses over time
# =============================================================================

"""
    inequality_irf(sol::HADSGESolution{T}, horizon::Int;
                   shock_index=1, shock_size=1.0) -> Dict{Symbol,Vector{T}}

Compute Gini coefficient and wealth percentile responses over time after
an aggregate shock.

Returns a Dict with keys:
- `:gini` — Gini coefficient at each horizon
- `:p10`, `:p25`, `:p50`, `:p75`, `:p90` — wealth percentiles at each horizon

# Arguments
- `sol::HADSGESolution{T}` — linearized HA-DSGE solution
- `horizon::Int` — number of periods

# Keyword Arguments
- `shock_index::Int` — which aggregate shock (default 1)
- `shock_size::Real` — shock magnitude (default 1.0)
"""
function inequality_irf(sol::HADSGESolution{T}, horizon::Int;
                         shock_index::Int=1,
                         shock_size::Real=1.0) where {T<:AbstractFloat}
    ss = sol.steady_state
    grid = ss.grid

    # Get distribution IRF deviations
    d_irf = distribution_irf(sol, horizon; shock_index=shock_index,
                              shock_size=shock_size)

    # Steady-state distribution (flattened)
    d_ss = vec(ss.distribution)
    d_ss_norm = d_ss ./ sum(d_ss)

    n_a = grid.n_points[1]
    n_e = grid.n_income

    gini_path = zeros(T, horizon)
    p10_path = zeros(T, horizon)
    p25_path = zeros(T, horizon)
    p50_path = zeros(T, horizon)
    p75_path = zeros(T, horizon)
    p90_path = zeros(T, horizon)

    for h in 1:horizon
        # Perturbed distribution = steady state + deviation
        d_dev = vec(d_irf[:, :, h])
        d_h = d_ss_norm .+ d_dev

        # Ensure non-negativity and normalization
        d_h .= max.(d_h, zero(T))
        s = sum(d_h)
        if s > zero(T)
            d_h ./= s
        else
            d_h .= d_ss_norm  # fallback to SS if degenerate
        end

        gini_path[h] = _gini_coefficient(d_h, grid)
        p10_path[h] = _wealth_percentile(d_h, grid, T(0.10))
        p25_path[h] = _wealth_percentile(d_h, grid, T(0.25))
        p50_path[h] = _wealth_percentile(d_h, grid, T(0.50))
        p75_path[h] = _wealth_percentile(d_h, grid, T(0.75))
        p90_path[h] = _wealth_percentile(d_h, grid, T(0.90))
    end

    return Dict{Symbol,Vector{T}}(
        :gini => gini_path,
        :p10 => p10_path,
        :p25 => p25_path,
        :p50 => p50_path,
        :p75 => p75_path,
        :p90 => p90_path
    )
end

"""
    inequality_irf(ss::HASteadyState{T}; T_periods=50) -> Dict{Symbol,Vector{T}}

Compute inequality measures at steady state over `T_periods` periods.

Since the steady state is stationary, the Gini and percentile values are
constant over time. This method is useful as a baseline or when no aggregate
dynamics have been solved.

# Arguments
- `ss::HASteadyState{T}` — stationary equilibrium

# Keyword Arguments
- `T_periods::Int` — number of periods to report (default 50)
"""
function inequality_irf(ss::HASteadyState{T}; T_periods::Int=50) where {T<:AbstractFloat}
    grid = ss.grid
    d_vec = vec(ss.distribution)
    d_vec_norm = d_vec ./ sum(d_vec)

    gini_val = _gini_coefficient(d_vec_norm, grid)
    p10_val = _wealth_percentile(d_vec_norm, grid, T(0.10))
    p25_val = _wealth_percentile(d_vec_norm, grid, T(0.25))
    p50_val = _wealth_percentile(d_vec_norm, grid, T(0.50))
    p75_val = _wealth_percentile(d_vec_norm, grid, T(0.75))
    p90_val = _wealth_percentile(d_vec_norm, grid, T(0.90))

    return Dict{Symbol,Vector{T}}(
        :gini => fill(gini_val, T_periods),
        :p10 => fill(p10_val, T_periods),
        :p25 => fill(p25_val, T_periods),
        :p50 => fill(p50_val, T_periods),
        :p75 => fill(p75_val, T_periods),
        :p90 => fill(p90_val, T_periods)
    )
end

# =============================================================================
# simulate_panel — individual-level panel simulation
# =============================================================================

"""
    simulate_panel(ss::HASteadyState{T}; N_agents=1000, T_periods=100,
                   rng=Random.default_rng()) -> Matrix{T}

Simulate individual-level panel data from the steady-state policy functions.

Returns an `N_agents × T_periods` matrix of asset holdings.

# Algorithm
1. Draw initial assets from the steady-state marginal asset distribution.
2. Draw income state paths from the Markov chain.
3. Apply the savings policy function at each period via interpolation.

# Arguments
- `ss::HASteadyState{T}` — stationary equilibrium

# Keyword Arguments
- `N_agents::Int` — number of agents to simulate (default 1000)
- `T_periods::Int` — number of time periods (default 100)
- `rng` — random number generator (default `Random.default_rng()`)
"""
function simulate_panel(ss::HASteadyState{T};
                         N_agents::Int=1000,
                         T_periods::Int=100,
                         rng=Random.default_rng()) where {T<:AbstractFloat}
    grid = ss.grid
    @assert grid.n_dims == 1 "simulate_panel requires a one-asset model"

    a_grid = grid.grids[1]
    n_a = grid.n_points[1]
    n_e = grid.n_income
    a_min = a_grid[1]

    a_pol = ss.policies[:savings]  # n_a × n_e
    Pi = ss.income.transition      # n_e × n_e, row-stochastic

    # --- Step 1: Draw initial assets from marginal asset distribution ---
    d_vec = vec(ss.distribution)
    d_vec_pos = max.(d_vec, zero(T))
    total = sum(d_vec_pos)
    if total > zero(T)
        d_vec_pos ./= total
    else
        d_vec_pos .= one(T) / length(d_vec_pos)
    end

    # Build marginal asset distribution (sum over income states)
    d_mat = reshape(d_vec_pos, n_a, n_e)
    d_asset = vec(sum(d_mat; dims=2))
    d_asset ./= sum(d_asset)

    # Build cumulative distribution for sampling
    cdf_asset = cumsum(d_asset)
    cdf_asset[end] = one(T)  # ensure exact

    # Draw initial asset indices
    panel = zeros(T, N_agents, T_periods)
    income_states = zeros(Int, N_agents)

    for i in 1:N_agents
        u = rand(rng, T)
        idx = searchsortedfirst(cdf_asset, u)
        idx = clamp(idx, 1, n_a)
        panel[i, 1] = a_grid[idx]

        # Draw initial income state from stationary distribution
        cdf_inc = cumsum(ss.income.stationary_dist)
        cdf_inc[end] = one(T)
        u_inc = rand(rng, T)
        income_states[i] = clamp(searchsortedfirst(cdf_inc, u_inc), 1, n_e)
    end

    # --- Step 2–3: Simulate forward ---
    # Pre-compute cumulative transition rows for efficient sampling
    cdf_rows = zeros(T, n_e, n_e)
    for j in 1:n_e
        cdf_rows[j, :] = cumsum(Pi[j, :])
        cdf_rows[j, end] = one(T)
    end

    for t in 2:T_periods
        for i in 1:N_agents
            a_curr = panel[i, t-1]
            e_idx = income_states[i]

            # Apply savings policy via interpolation
            a_next = _linear_interp(a_grid, view(a_pol, :, e_idx), a_curr)
            a_next = max(a_next, a_min)
            panel[i, t] = a_next

            # Transition income state
            u = rand(rng, T)
            new_e = searchsortedfirst(view(cdf_rows, e_idx, :), u)
            income_states[i] = clamp(new_e, 1, n_e)
        end
    end

    return panel
end
