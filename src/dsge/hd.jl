# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Historical decomposition for DSGE models (linear and nonlinear).

Decomposes observed variable movements into contributions from individual structural shocks
plus initial conditions.

For linear DSGE: uses the Kalman smoother to recover smoothed shocks and the
structural MA representation to attribute movements to each shock.
Theory: y_t = Σ_{s=0}^{t-1} Θ_s ε_{t-s} + initial_conditions
where Θ_s = Z_dec * G1^s * impact (structural MA coefficients at lag s).

For nonlinear DSGE: uses the FFBSi particle smoother to recover smoothed shocks and
counterfactual simulation to attribute movements to each shock.

References:
- Herbst, E. & Schorfheide, F. (2015). Bayesian Estimation of DSGE Models. Ch. 6.
- Canova, F. (2007). Methods for Applied Macroeconomic Research. Ch. 7.
- Godsill, S. J., Doucet, A., & West, M. (2004). Monte Carlo smoothing for nonlinear
  time series. JASA, 99(465), 156-168.
"""

using LinearAlgebra, Random

"""
    historical_decomposition(sol::DSGESolution{T}, data::AbstractMatrix,
                              observables::Vector{Symbol};
                              states=:observables, measurement_error=nothing) -> HistoricalDecomposition{T}

Compute historical decomposition for a linear DSGE model.

Decomposes observed data into contributions from each structural shock plus initial conditions
using the Kalman smoother to extract smoothed shocks and the structural MA representation.

# Arguments
- `sol::DSGESolution{T}`: Solved linear DSGE model
- `data::AbstractMatrix`: T_obs × n_endog matrix of data in LEVELS
- `observables::Vector{Symbol}`: Which endogenous variables are observed

# Keyword Arguments
- `states::Symbol=:observables`: Decompose `:observables` (default) or `:all` states
- `measurement_error`: Vector of measurement error std devs, or `nothing` (default: small diagonal)

# Returns
`HistoricalDecomposition{T}` with:
- `contributions`: T_obs × n_vars × n_shocks shock contribution array
- `initial_conditions`: T_obs × n_vars initial condition component
- `actual`: T_obs × n_vars actual data in deviations from steady state
- `shocks`: T_obs × n_shocks smoothed structural shocks
- `method`: `:dsge_linear`

# Example
```julia
spec = @dsge begin
    parameters: rho = 0.8
    endogenous: y
    exogenous: eps
    y[t] = rho * y[t-1] + eps[t]
end
sol = solve(spec)
sim_data = simulate(sol, 100)
hd = historical_decomposition(sol, sim_data, [:y])
verify_decomposition(hd)
```
"""
function historical_decomposition(sol::DSGESolution{T}, data::AbstractMatrix,
                                   observables::Vector{Symbol};
                                   states::Symbol=:observables,
                                   measurement_error=nothing) where {T<:AbstractFloat}
    spec = sol.spec
    n_states = spec.n_endog
    n_shocks = spec.n_exog
    T_obs = size(data, 1)

    states in (:observables, :all) || throw(ArgumentError(
        "states must be :observables or :all, got :$states"))

    # =========================================================================
    # Step 1: Build state space
    # =========================================================================
    Z, d, H = _build_observation_equation(spec, observables, measurement_error)
    ss = _build_state_space(sol, Z, d, H)
    n_obs = length(observables)

    # Observable indices in the full state vector
    obs_indices = [findfirst(==(obs), spec.endog) for obs in observables]

    # =========================================================================
    # Step 2: Convert data to deviations from steady state (n_obs × T_obs)
    # =========================================================================
    # data is T_obs × n_endog in LEVELS; extract observable columns and subtract SS
    data_dev = zeros(T, n_obs, T_obs)
    for (i, idx) in enumerate(obs_indices)
        ss_val = isempty(spec.steady_state) ? zero(T) : spec.steady_state[idx]
        for t in 1:T_obs
            data_dev[i, t] = T(data[t, idx]) - ss_val
        end
    end

    # =========================================================================
    # Step 3: Run Kalman smoother to get smoothed shocks
    # =========================================================================
    smoother_result = dsge_smoother(ss, data_dev)
    # smoothed_shocks is n_shocks × T_obs
    smoothed_shocks = smoother_result.smoothed_shocks

    # =========================================================================
    # Step 4: Compute structural MA coefficients Θ_s = Z_dec * G1^s * impact
    # =========================================================================
    # Determine the decomposition selection matrix
    if states == :all
        # Decompose all state variables
        n_vars = n_states
        Z_dec = Matrix{T}(I, n_states, n_states)
        var_names = [string(s) for s in spec.endog]
    else
        # Decompose only observables
        n_vars = n_obs
        Z_dec = Z  # n_obs × n_states selection matrix
        var_names = [string(s) for s in observables]
    end

    # Compute Θ_s for s = 0, ..., T_obs-1
    # Θ_s = Z_dec * G1^s * impact
    Theta = Vector{Matrix{T}}(undef, T_obs)
    G1_power = Matrix{T}(I, n_states, n_states)
    for s in 1:T_obs
        Theta[s] = Z_dec * G1_power * sol.impact
        G1_power = G1_power * sol.G1
    end

    # =========================================================================
    # Step 5: Compute contributions
    # HD[t, i, j] = Σ_{s=0}^{t-1} Θ_{s+1}[i, j] * ε_j(t-s)
    # =========================================================================
    # Note: shocks are n_shocks × T_obs, need T_obs × n_shocks for the loop
    shocks_mat = Matrix{T}(smoothed_shocks')  # T_obs × n_shocks

    contributions = zeros(T, T_obs, n_vars, n_shocks)
    @inbounds for t in 1:T_obs
        for i in 1:n_vars
            for j in 1:n_shocks
                val = zero(T)
                for s in 0:(t-1)
                    # Theta[s+1] is the MA coefficient at lag s
                    # shocks_mat[t-s, j] is the shock at time t-s
                    val += Theta[s+1][i, j] * shocks_mat[t-s, j]
                end
                contributions[t, i, j] = val
            end
        end
    end

    # =========================================================================
    # Step 6: Compute actual data in deviations (T_obs × n_vars)
    # =========================================================================
    if states == :all
        actual = zeros(T, T_obs, n_states)
        for i in 1:n_states
            ss_val = isempty(spec.steady_state) ? zero(T) : spec.steady_state[i]
            for t in 1:T_obs
                actual[t, i] = T(data[t, i]) - ss_val
            end
        end
    else
        # For observables: transpose data_dev back to T_obs × n_obs
        actual = Matrix{T}(data_dev')
    end

    # =========================================================================
    # Step 7: Initial conditions = actual - sum of contributions
    # =========================================================================
    initial_conditions = _compute_initial_conditions(actual, contributions)

    # Shock names
    shock_names = [string(s) for s in spec.exog]

    return HistoricalDecomposition{T}(
        contributions, initial_conditions, actual, shocks_mat, T_obs,
        var_names, shock_names, :dsge_linear
    )
end

# =============================================================================
# Nonlinear DSGE historical decomposition — counterfactual simulation
# =============================================================================

"""
    historical_decomposition(sol::PerturbationSolution{T}, data::AbstractMatrix,
                              observables::Vector{Symbol};
                              states::Symbol=:observables,
                              measurement_error=nothing,
                              N::Int=1000, N_back::Int=100,
                              rng::AbstractRNG=Random.default_rng()) where {T}

Compute historical decomposition for a nonlinear (higher-order perturbation) DSGE model.

Uses the FFBSi particle smoother to recover smoothed shocks, then performs counterfactual
simulation: for each shock j, simulates the model with shock j zeroed out.
Contribution of shock j = baseline - counterfactual_without_j.

# Arguments
- `sol::PerturbationSolution{T}`: Higher-order perturbation solution
- `data::AbstractMatrix`: T_obs × n_endog matrix of data in LEVELS
- `observables::Vector{Symbol}`: Which endogenous variables are observed

# Keyword Arguments
- `states::Symbol=:observables`: Decompose `:observables` (default) or `:all` states
- `measurement_error`: Vector of measurement error std devs, or `nothing`
- `N::Int=1000`: Number of forward particles for the smoother
- `N_back::Int=100`: Number of backward trajectories for the smoother
- `rng::AbstractRNG`: Random number generator

# Returns
`HistoricalDecomposition{T}` with `method=:dsge_nonlinear`.
"""
function historical_decomposition(sol::PerturbationSolution{T}, data::AbstractMatrix,
                                   observables::Vector{Symbol};
                                   states::Symbol=:observables,
                                   measurement_error=nothing,
                                   N::Int=1000, N_back::Int=100,
                                   rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    spec = sol.spec
    n_endog = spec.n_endog
    n_shocks = spec.n_exog
    T_obs = size(data, 1)

    states in (:observables, :all) || throw(ArgumentError(
        "states must be :observables or :all, got :$states"))

    # =========================================================================
    # Step 1: Build nonlinear state space
    # =========================================================================
    Z, d, H = _build_observation_equation(spec, observables, measurement_error)
    nss = _build_nonlinear_state_space(sol, Z, d, H)
    n_obs = length(observables)

    # Observable indices
    obs_indices = [findfirst(==(obs), spec.endog) for obs in observables]

    # =========================================================================
    # Step 2: Convert data to deviations from steady state (n_obs × T_obs)
    # =========================================================================
    data_dev = zeros(T, n_obs, T_obs)
    for (i, idx) in enumerate(obs_indices)
        ss_val = isempty(spec.steady_state) ? zero(T) : spec.steady_state[idx]
        for t in 1:T_obs
            data_dev[i, t] = T(data[t, idx]) - ss_val
        end
    end

    # =========================================================================
    # Step 3: Run particle smoother to get smoothed shocks
    # =========================================================================
    smoother_result = dsge_particle_smoother(nss, data_dev; N=N, N_back=N_back, rng=rng)
    # smoothed_shocks is n_shocks × T_obs
    shocks_mat = Matrix{T}(smoother_result.smoothed_shocks')  # T_obs × n_shocks

    # =========================================================================
    # Step 4: Baseline simulation with all smoothed shocks
    # =========================================================================
    baseline_levels = simulate(sol, T_obs; shock_draws=shocks_mat)
    # Convert to deviations from steady state: T_obs × n_endog
    baseline_dev = baseline_levels .- sol.steady_state'

    # =========================================================================
    # Step 5: Determine decomposition variables
    # =========================================================================
    if states == :all
        n_vars = n_endog
        var_indices = collect(1:n_endog)
        var_names = [string(s) for s in spec.endog]
    else
        n_vars = n_obs
        var_indices = obs_indices
        var_names = [string(s) for s in observables]
    end

    # =========================================================================
    # Step 6: Counterfactual decomposition
    # For each shock j: simulate with shock j zeroed out
    # contribution_j = baseline - counterfactual_without_j
    # =========================================================================
    contributions = zeros(T, T_obs, n_vars, n_shocks)

    for j in 1:n_shocks
        # Create shock matrix with shock j zeroed out
        shocks_cf = copy(shocks_mat)
        shocks_cf[:, j] .= zero(T)

        # Simulate counterfactual
        cf_levels = simulate(sol, T_obs; shock_draws=shocks_cf)
        cf_dev = cf_levels .- sol.steady_state'

        # Contribution = baseline - counterfactual
        for (vi, vidx) in enumerate(var_indices)
            for t in 1:T_obs
                contributions[t, vi, j] = baseline_dev[t, vidx] - cf_dev[t, vidx]
            end
        end
    end

    # =========================================================================
    # Step 7: Actual data in deviations (T_obs × n_vars)
    # =========================================================================
    if states == :all
        actual = zeros(T, T_obs, n_endog)
        for i in 1:n_endog
            ss_val = isempty(spec.steady_state) ? zero(T) : spec.steady_state[i]
            for t in 1:T_obs
                actual[t, i] = T(data[t, i]) - ss_val
            end
        end
    else
        actual = Matrix{T}(data_dev')
    end

    # =========================================================================
    # Step 8: Initial conditions = actual - sum of contributions
    # =========================================================================
    initial_conditions = _compute_initial_conditions(actual, contributions)

    # Shock names
    shock_names = [string(s) for s in spec.exog]

    return HistoricalDecomposition{T}(
        contributions, initial_conditions, actual, shocks_mat, T_obs,
        var_names, shock_names, :dsge_nonlinear
    )
end
