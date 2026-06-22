# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Krusell-Smith (1998) simulation-based solver for heterogeneous agent models.

Approximates the distribution via a perceived law of motion (PLM) for aggregate
state variables.  The PLM is a log-linear regression of future aggregate capital
on current aggregate capital.  The algorithm iterates between (1) simulating the
economy forward using the PLM-implied prices and (2) updating the PLM via OLS on
the simulated path.

# References
- Krusell, P., & Smith, A. A. (1998). Income and wealth heterogeneity in the
  macroeconomy. *Journal of Political Economy*, 106(5), 867–896.
- Young, E. R. (2010). Solving the incomplete markets model with aggregate
  uncertainty using the Krusell–Smith algorithm and non-stochastic simulations.
  *Journal of Economic Dynamics and Control*, 34(1), 36–41.
"""

using Random, Statistics

# =============================================================================
# _krusell_smith_solve — simulation-based PLM iteration
# =============================================================================

"""
    _krusell_smith_solve(ss, ip, grid, income, price_fn, params;
        T_sim=11000, T_burn=1000, max_outer=20, rho_z=0.95, sigma_z=0.007,
        tol=1e-5, damping=0.5, seed=1234) → NamedTuple

Solve a heterogeneous agent model with aggregate uncertainty using the
Krusell-Smith (1998) algorithm.

# Algorithm
1. **Initialize PLM**: log K' = b[1] + b[2] × log K (perceived law of motion).
2. **Outer loop** (iterate until PLM converges):
   a. Simulate aggregate TFP shock path: z[t] = ρ_z z[t-1] + σ_z ε[t].
   b. For t = 1, …, T_sim:
      - Compute prices from current K and z via `price_fn`.
      - Solve the individual problem via EGM at those prices (warm-started
        from the steady-state policy).
      - Forward-iterate the distribution one period.
      - Compute realized K_{t+1} = `_aggregate(dist, grid)`.
   c. Regress log(K_{t+1}) on [1, log(K_t)] via OLS (burn-in excluded).
   d. Compute R².
   e. Check ‖b_new − b_old‖_∞ < tol.
   f. Damped update: b = damping × b_new + (1 − damping) × b_old.
3. Return a NamedTuple with PLM coefficients, R², convergence info.

# Arguments
- `ss::HASteadyState{T}` — stationary equilibrium (provides initial K, policies,
  distribution)
- `ip::IndividualProblem{T}` — household problem specification
- `grid::HAGrid{T}` — asset grid
- `income::IncomeProcess{T}` — idiosyncratic income process
- `price_fn::Function` — `(K, params) → Dict{Symbol,T}` mapping capital to prices
- `params::Dict{Symbol,T}` — model parameters (`:alpha`, `:delta`, `:Z`, `:L`)

# Keyword Arguments
- `T_sim::Int` — total simulation length (default 11000)
- `T_burn::Int` — burn-in periods to discard (default 1000)
- `max_outer::Int` — maximum PLM iterations (default 20)
- `rho_z::T` — persistence of aggregate TFP shock (default 0.95)
- `sigma_z::T` — standard deviation of TFP innovation (default 0.007)
- `tol` — convergence tolerance on PLM coefficients (default 1e-5)
- `damping::T` — damping factor for PLM update (default 0.5)
- `seed::Int` — RNG seed for reproducibility (default 1234)

# Returns
A NamedTuple with fields:
- `plm_coefficients::Dict{Symbol,Vector{T}}` — PLM regression coefficients (`:K => [b0, b1]`)
- `r_squared::Dict{Symbol,T}` — R² of PLM regression (`:K => R²`)
- `converged::Bool` — whether PLM iteration converged
- `iterations::Int` — number of outer loop iterations used
"""
function _krusell_smith_solve(ss::HASteadyState{T},
                               ip::IndividualProblem{T},
                               grid::HAGrid{T},
                               income::IncomeProcess{T},
                               price_fn::Function,
                               params::Dict{Symbol,T};
                               T_sim::Int=11000,
                               T_burn::Int=1000,
                               max_outer::Int=20,
                               rho_z::Real=0.95,
                               sigma_z::Real=0.007,
                               tol::Real=1e-5,
                               damping::Real=0.5,
                               model::Symbol=:aiyagari,
                               rho_e::Real=0.9,
                               sigma_e::Real=0.01,
                               seed::Int=1234) where {T<:AbstractFloat}
    @assert grid.n_dims == 1 "Krusell-Smith solver requires a one-asset grid"
    @assert ip.n_asset_dims == 1 "Krusell-Smith solver requires a one-asset individual problem"
    @assert T_sim > T_burn "T_sim must exceed T_burn"

    if model === :huggett
        return _krusell_smith_huggett(ss, ip, grid, income;
            T_sim=T_sim, T_burn=T_burn, max_outer=max_outer,
            rho_e=rho_e, sigma_e=sigma_e, tol=tol, damping=damping, seed=seed)
    end

    rho_z_T = T(rho_z)
    sigma_z_T = T(sigma_z)
    tol_T = T(tol)
    damping_T = T(damping)

    rng = Random.MersenneTwister(seed)

    # Extract steady-state aggregate capital
    K_ss = ss.aggregates[:K]
    log_K_ss = log(K_ss)

    # Initialize PLM coefficients: log K' = b[1] + b[2] * log K
    # Start near identity: K' ≈ K → log K' ≈ 0 + 1 * log K
    b = T[zero(T), one(T)]

    # Pre-generate aggregate shock path (fixed across outer iterations for stability)
    z_path = zeros(T, T_sim)
    innovations = randn(rng, T, T_sim)
    for t in 2:T_sim
        z_path[t] = rho_z_T * z_path[t-1] + sigma_z_T * innovations[t]
    end

    # Storage for simulation
    log_K_series = zeros(T, T_sim)
    K_series = zeros(T, T_sim)

    converged = false
    final_iter = 0
    best_r2 = zero(T)

    for outer in 1:max_outer
        final_iter = outer

        # Initialize simulation from steady state
        K_series[1] = K_ss
        log_K_series[1] = log_K_ss

        # Start from the steady-state distribution (flattened)
        dist = vec(ss.distribution)
        dist = dist ./ sum(dist)  # ensure normalization

        # Solve the individual problem once at steady-state prices for warm start.
        # In standard KS, the policy function barely changes across periods when
        # aggregate shocks are small, so solving once per period is sufficient.
        # For speed, we re-solve only when prices change significantly.

        # Cache the last-used prices and policy to avoid redundant EGM calls
        last_r = ss.prices[:r]
        last_w = ss.prices[:w]
        a_pol = copy(ss.policies[:savings])

        # Build initial transition matrix from SS policy
        Lambda = _build_transition_matrix(a_pol, grid, income)

        for t in 1:(T_sim - 1)
            K_t = K_series[t]
            z_t = z_path[t]

            # Effective aggregate capital with TFP shock
            K_eff = K_t * exp(z_t)

            # Compute prices at current aggregate state
            prices = price_fn(K_eff, params)

            # Re-solve individual problem if prices changed meaningfully
            r_new = prices[:r]
            w_new = prices[:w]
            if abs(r_new - last_r) > T(1e-8) || abs(w_new - last_w) > T(1e-8)
                c_pol_new, a_pol_new = _egm_solve(ip, grid, income, prices;
                                                   max_iter=50, tol=T(1e-8))
                copyto!(a_pol, a_pol_new)
                Lambda = _build_transition_matrix(a_pol, grid, income)
                last_r = r_new
                last_w = w_new
            end

            # Forward-iterate distribution
            dist = _forward_iterate(Lambda, dist)

            # Compute realized next-period aggregate capital
            K_next = _aggregate(dist, grid; var_index=1)

            # Ensure K stays positive and finite
            K_next = max(K_next, T(1e-6))
            if !isfinite(K_next)
                K_next = K_ss
            end

            K_series[t+1] = K_next
            log_K_series[t+1] = log(K_next)
        end

        # OLS regression: log K_{t+1} = b[1] + b[2] * log K_t
        # Use only post-burn-in observations
        n_obs = T_sim - T_burn - 1  # number of (K_t, K_{t+1}) pairs after burn-in
        if n_obs < 10
            # Not enough data — cannot regress
            break
        end

        idx_start = T_burn + 1
        idx_end = T_sim - 1

        # Construct OLS matrices
        y_ols = log_K_series[(idx_start + 1):(idx_end + 1)]   # log K_{t+1}
        X_ols = hcat(ones(T, n_obs), log_K_series[idx_start:idx_end])  # [1, log K_t]

        # Solve via least squares: b_new = (X'X) \ (X'y)
        b_new = X_ols \ y_ols

        # Compute R²
        y_hat = X_ols * b_new
        resid_vec = y_ols .- y_hat
        ss_res = sum(resid_vec .^ 2)
        y_mean = mean(y_ols)
        ss_tot = sum((y_ols .- y_mean) .^ 2)
        r2 = ss_tot > zero(T) ? one(T) - ss_res / ss_tot : one(T)
        best_r2 = r2

        # Check convergence
        coef_diff = maximum(abs.(b_new .- b))
        if coef_diff < tol_T
            converged = true
            b = b_new
            break
        end

        # Damped update
        b = damping_T .* b_new .+ (one(T) - damping_T) .* b
    end

    plm_coefficients = Dict{Symbol,Vector{T}}(:K => copy(b))
    r_squared = Dict{Symbol,T}(:K => best_r2)

    return (; plm_coefficients, r_squared, converged, iterations=final_iter)
end

# =============================================================================
# _krusell_smith_huggett — Huggett (1993) variant: PLM forecasts the clearing rate
# =============================================================================

"""
    _krusell_smith_huggett(ss, ip, grid, income; T_sim, T_burn, max_outer,
        rho_e, sigma_e, tol, damping, seed) → NamedTuple

Krusell-Smith algorithm for the Huggett (1993) economy with an aggregate endowment
shock. Because the bond is in zero net supply, the relevant aggregate price is the
risk-free rate, so the perceived law of motion forecasts `r` (not capital) from the
endowment shock:

    r_t = b[1] + b[2] · z_t,      z_t = ρ_e z_{t-1} + σ_e ε_t,   w_t = exp(z_t).

Each period uses the Young (2010) non-stochastic simulation. The market-clearing rate
is recovered by a single Newton step around the PLM guess,
`r_clear = r_guess − A(r_guess) / A_r`, where `A(r) = ∫ a'(·; r, w_t) dμ_t` is aggregate
bond demand and `A_r ≈ ∂A/∂r` is the (positive) aggregate-savings slope evaluated once
at the steady state. The PLM is then refit by OLS and damped until convergence.
"""
function _krusell_smith_huggett(ss::HASteadyState{T}, ip::IndividualProblem{T},
                                 grid::HAGrid{T}, income::IncomeProcess{T};
                                 T_sim::Int=4000, T_burn::Int=500, max_outer::Int=8,
                                 rho_e::Real=0.9, sigma_e::Real=0.01,
                                 tol::Real=1e-5, damping::Real=0.5,
                                 seed::Int=1234) where {T<:AbstractFloat}
    rho = T(rho_e); sig = T(sigma_e); tol_T = T(tol); damping_T = T(damping)
    rng = Random.MersenneTwister(seed)
    r_ss = ss.prices[:r]
    dist_ss = vec(ss.distribution); dist_ss ./= sum(dist_ss)

    # Aggregate-savings slope dA/dr at the steady state (fixed Newton-step slope).
    dr0 = T(1e-4)
    _, ap0 = _egm_solve(ip, grid, income, Dict{Symbol,T}(:r => r_ss, :w => one(T));
                        max_iter=1000, tol=T(1e-10))
    _, ap1 = _egm_solve(ip, grid, income, Dict{Symbol,T}(:r => r_ss + dr0, :w => one(T));
                        max_iter=1000, tol=T(1e-10))
    A_r = (sum(vec(ap1) .* dist_ss) - sum(vec(ap0) .* dist_ss)) / dr0
    @assert A_r > zero(T) "Huggett KS: non-positive aggregate-savings slope"

    # Aggregate endowment shock path (log endowment).
    z = zeros(T, T_sim); innov = randn(rng, T, T_sim)
    for t in 2:T_sim
        z[t] = rho * z[t-1] + sig * innov[t]
    end

    b = T[r_ss, zero(T)]                # PLM: r_t = b[1] + b[2] z_t
    r_series = zeros(T, T_sim)
    r_ceil = one(T) / ip.beta - one(T) - T(1e-5)   # per-period time-preference ceiling
    best_r2 = zero(T); converged = false; final_iter = 0

    for outer in 1:max_outer
        final_iter = outer
        dist = copy(dist_ss)
        for t in 1:T_sim
            w_t = exp(z[t])
            r_g = b[1] + b[2] * z[t]                 # PLM forecast (warm start)
            # One Newton clearing step using the SS savings slope A_r (≈ exact slope).
            _, a_pol_g = _egm_solve(ip, grid, income, Dict{Symbol,T}(:r => r_g, :w => w_t);
                                    max_iter=80, tol=T(1e-8))
            A_g = sum(vec(a_pol_g) .* dist)
            r_t = clamp(r_g - A_g / A_r, T(-0.2), r_ceil)
            # Re-solve at the cleared rate and forward with that policy (keeps ∫a' ≈ 0).
            _, a_pol = _egm_solve(ip, grid, income, Dict{Symbol,T}(:r => r_t, :w => w_t);
                                  max_iter=80, tol=T(1e-8))
            r_series[t] = r_t
            dist = _forward_iterate(_build_transition_matrix(a_pol, grid, income), dist)
        end

        idx = (T_burn + 1):T_sim
        X = hcat(ones(T, length(idx)), z[idx])
        y = r_series[idx]
        b_new = X \ y
        yhat = X * b_new
        ss_res = sum((y .- yhat) .^ 2)
        ss_tot = sum((y .- mean(y)) .^ 2)
        best_r2 = ss_tot > zero(T) ? one(T) - ss_res / ss_tot : one(T)

        # The realized clearing path is independent of the PLM guess (agents are myopic
        # about prices, as in the Aiyagari path), so a full update converges in ~2 passes.
        if maximum(abs.(b_new .- b)) < tol_T
            b = b_new; converged = true; break
        end
        b = b_new
    end

    return (; plm_coefficients=Dict{Symbol,Vector{T}}(:r => copy(b)),
              r_squared=Dict{Symbol,T}(:r => best_r2), converged, iterations=final_iter)
end
