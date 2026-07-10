# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Stochastic simulation and IRF/FEVD bridge for DSGE solutions.
"""

using Random

"""
    simulate(sol::DSGESolution{T}, T_periods::Int;
             shock_draws=nothing, rng=Random.default_rng()) -> Matrix{T}

Simulate the solved DSGE model: `y_t = G1 * y_{t-1} + impact * e_t + C_sol`.

Returns `T_periods x n_endog` matrix of levels (steady state + deviations).

# Arguments
- `sol`: solved DSGE model
- `T_periods`: number of periods to simulate

# Keyword Arguments
- `shock_draws`: `T_periods x n_shocks` matrix of pre-drawn shocks (default: `nothing`, draws from N(0,1))
- `rng`: random number generator (default: `Random.default_rng()`)
"""
function simulate(sol::DSGESolution{T}, T_periods::Int;
                  shock_draws::Union{Nothing,AbstractMatrix}=nothing,
                  rng=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(sol)
    n_e = nshocks(sol)
    y_ss = sol.spec.steady_state

    # Draw or use provided shocks
    if shock_draws !== nothing
        @assert size(shock_draws) == (T_periods, n_e) "shock_draws must be ($T_periods, $n_e)"
        e = T.(shock_draws)
    else
        e = randn(rng, T, T_periods, n_e)
    end

    # Simulate deviations from steady state
    dev = zeros(T, T_periods, n)
    for t in 1:T_periods
        y_prev = t == 1 ? zeros(T, n) : dev[t-1, :]
        dev[t, :] = sol.G1 * y_prev + sol.impact * e[t, :] + sol.C_sol
    end

    # Return levels (steady state + deviations)
    levels = dev .+ y_ss'

    # Filter to original variables if augmented
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        return levels[:, orig_idx]
    end

    return levels
end

"""
    irf(sol::DSGESolution{T}, horizon::Int; kwargs...) -> ImpulseResponse{T}

Compute analytical impulse responses from a solved DSGE model.

At horizon h for shock j: `Phi_h[:,j] = G1^(h-1) * impact[:,j]`.

Returns an `ImpulseResponse{T}` compatible with `plot_result()`.

# Arguments
- `sol`: solved DSGE model
- `horizon`: number of IRF periods

# Keyword Arguments
- `ci_type::Symbol=:none`: confidence interval type (`:none` for analytical DSGE)
"""
function irf(sol::DSGESolution{T}, horizon::Int;
             ci_type::Symbol=:none, kwargs...) where {T<:AbstractFloat}
    n = nvars(sol)
    n_e = nshocks(sol)

    # Analytical IRFs: Phi_h = G1^(h-1) * impact
    point_irf = zeros(T, horizon, n, n_e)
    G1_power = Matrix{T}(I, n, n)

    for h in 1:horizon
        for j in 1:n_e
            point_irf[h, :, j] = G1_power * sol.impact[:, j]
        end
        G1_power = G1_power * sol.G1
    end

    # Filter to original variables if augmented
    if sol.spec.augmented
        orig_idx = _original_var_indices(sol.spec)
        point_irf = point_irf[:, orig_idx, :]
        var_names = [string(s) for s in sol.spec.original_endog]
        n_out = length(orig_idx)
    else
        var_names = sol.spec.varnames
        n_out = n
    end
    shock_names = [string(s) for s in sol.spec.exog]

    ci_lower = zeros(T, horizon, n_out, n_e)
    ci_upper = zeros(T, horizon, n_out, n_e)

    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                        var_names, shock_names, ci_type)
end

"""
    fevd(sol::DSGESolution{T}, horizon::Int) -> FEVD{T}

Compute forecast error variance decomposition from a solved DSGE model.

Uses the analytical IRFs from the solution to compute the proportion of
h-step forecast error variance attributable to each structural shock.

Returns a `FEVD{T}` compatible with `plot_result()`.
"""
function fevd(sol::DSGESolution{T}, horizon::Int) where {T<:AbstractFloat}
    irf_result = irf(sol, horizon)
    n_vars = length(irf_result.variables)
    n_e = nshocks(sol)

    # Compute FEVD directly — _compute_fevd assumes n_vars == n_shocks
    decomp = zeros(T, n_vars, n_e, horizon)
    props  = zeros(T, n_vars, n_e, horizon)

    @inbounds for h in 1:horizon
        for i in 1:n_vars
            total = zero(T)
            for j in 1:n_e
                prev = h == 1 ? zero(T) : decomp[i, j, h-1]
                decomp[i, j, h] = prev + irf_result.values[h, i, j]^2
                total += decomp[i, j, h]
            end
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end

    var_names = irf_result.variables
    shock_names = irf_result.shocks

    FEVD{T}(decomp, props, var_names, shock_names)
end

# =============================================================================
# ProjectionSolution simulation and IRF
# =============================================================================

"""
    simulate(sol::ProjectionSolution{T}, T_periods::Int; kwargs...) -> Matrix{T}

Simulate using the global policy function from projection solution.
Returns `T_periods x n_vars` matrix of levels.

# Arguments
- `sol`: projection solution from Chebyshev collocation
- `T_periods`: number of periods to simulate

# Keyword Arguments
- `shock_draws`: `T_periods x n_shocks` matrix of pre-drawn shocks (default: `nothing`, draws from N(0,1))
- `rng`: random number generator (default: `Random.default_rng()`)
"""
function simulate(sol::ProjectionSolution{T}, T_periods::Int;
                  shock_draws::Union{Nothing,AbstractMatrix}=nothing,
                  rng=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(sol)
    n_eps = nshocks(sol)
    nx = nstates(sol)
    ss = sol.steady_state

    if shock_draws !== nothing
        @assert size(shock_draws) == (T_periods, n_eps) "shock_draws must be ($T_periods, $n_eps)"
        e = T.(shock_draws)
    else
        e = randn(rng, T, T_periods, n_eps)
    end

    levels = zeros(T, T_periods, n)
    x_state = copy(ss[sol.state_indices])

    # Get linear impact matrix for shock propagation
    result_lin = _gensys_qz(sol.spec, sol.linear)
    impact = result_lin.impact

    for t in 1:T_periods
        # Evaluate policy at current state
        y_t = evaluate_policy(sol, x_state)
        levels[t, :] = y_t

        # Next-period state: policy state components + linear shock propagation
        x_next = y_t[sol.state_indices]
        # Add shock effect through the impact matrix (state rows)
        for (ii, si) in enumerate(sol.state_indices)
            for k in 1:n_eps
                x_next[ii] += impact[si, k] * e[t, k]
            end
        end

        # Clamp to state bounds for stability
        for d in 1:nx
            x_next[d] = clamp(x_next[d], sol.state_bounds[d, 1], sol.state_bounds[d, 2])
        end

        x_state = x_next
    end

    return levels
end

"""
    irf(sol::ProjectionSolution{T}, horizon::Int; kwargs...) -> ImpulseResponse{T}

Generalized IRF (Koop-Pesaran-Potter). For each shock and replication, draw a SHARED future
shock path and simulate a baseline and an impulse-shocked path over it; the impulse is applied
to the INITIAL state (`impact[·,j]·shock_size`, preserving the h=0 convention) and the two
paths share the same future shocks, so their difference isolates the impulse on the nonlinear
policy. Averaging over `n_sim` replications gives the GIRF. For a linear policy the shared
future shocks cancel exactly and the GIRF reduces to the deterministic IRF.

# Arguments
- `sol`: projection solution from Chebyshev collocation
- `horizon`: number of IRF periods

# Keyword Arguments
- `n_sim::Int=500`: number of GIRF replications to average
- `shock_size::Real=1.0`: impulse size in standard deviations
- `ci_type::Symbol=:none`: confidence interval type (bands are zero unless `:none`)
- `rng`: random number generator; results are reproducible for a fixed `rng`
"""
function irf(sol::ProjectionSolution{T}, horizon::Int;
             n_sim::Int=500, shock_size::Real=1.0,
             ci_type::Symbol=:none, rng=Random.default_rng(), kwargs...) where {T<:AbstractFloat}
    n = nvars(sol)
    n_eps = nshocks(sol)
    nx = nstates(sol)
    ss = sol.steady_state

    # Companion-QZ linear impact for shock propagation (correct for forward-looking models, #211).
    result_lin = _gensys_qz(sol.spec, sol.linear)
    impact = result_lin.impact

    point_irf = zeros(T, horizon, n, n_eps)
    base_seed = rand(rng, UInt64)          # one entropy draw ⇒ reproducible per (shock, rep)

    # Simulate the policy `horizon` periods from initial state `x0`, propagating the shared
    # future shocks `e` through the linear impact rows (mirrors `simulate`).
    function _girf_path(x0, e)
        x = copy(x0)
        path = zeros(T, horizon, n)
        for t in 1:horizon
            y = evaluate_policy(sol, x)
            path[t, :] = y
            x = y[sol.state_indices]
            for (ii, si) in enumerate(sol.state_indices), k in 1:n_eps
                x[ii] += impact[si, k] * e[t, k]
            end
            for d in 1:nx
                x[d] = clamp(x[d], sol.state_bounds[d, 1], sol.state_bounds[d, 2])
            end
        end
        path
    end

    for j in 1:n_eps
        irf_sum = zeros(T, horizon, n)
        for s in 1:n_sim
            rep_rng = Random.MersenneTwister(hash((j, s, base_seed)))
            e = randn(rep_rng, T, horizon, n_eps)   # shared future shocks for this replication

            x_base = copy(ss[sol.state_indices])
            # Impulse to shock j on the INITIAL state (h=0 convention), same future shocks.
            x_shock = copy(ss[sol.state_indices])
            for (ii, si) in enumerate(sol.state_indices)
                x_shock[ii] += impact[si, j] * T(shock_size)
            end
            for d in 1:nx
                x_shock[d] = clamp(x_shock[d], sol.state_bounds[d, 1], sol.state_bounds[d, 2])
            end

            irf_sum .+= (_girf_path(x_shock, e) .- _girf_path(x_base, e))
        end
        point_irf[:, :, j] = irf_sum ./ n_sim
    end

    var_names = sol.spec.varnames
    shock_names = [string(s) for s in sol.spec.exog]
    ci_lower = zeros(T, horizon, n, n_eps)
    ci_upper = zeros(T, horizon, n, n_eps)

    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                        var_names, shock_names, ci_type)
end
