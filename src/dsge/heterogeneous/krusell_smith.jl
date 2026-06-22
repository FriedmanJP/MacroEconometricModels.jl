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
# _simulate_explicit_K — explicit Young simulation of the realized capital path
# =============================================================================

"""
    _simulate_explicit_K(ss, ip, grid, income, price_fn, params, z) → Vector

Simulate the realized aggregate capital path of an Aiyagari economy by explicitly
iterating the cross-sectional distribution (Young 2010) under the aggregate TFP shock
path `z`. Prices come from the realized capital `K_t · exp(z_t)`; the policy is re-solved
only when prices move (warm-started from the steady state). This is the **reference**
path for the Den Haan (2010) accuracy test, and the per-iteration simulation reused by
`_krusell_smith_solve`.
"""
function _simulate_explicit_K(ss::HASteadyState{T}, ip::IndividualProblem{T},
                               grid::HAGrid{T}, income::IncomeProcess{T},
                               price_fn::Function, params::Dict{Symbol,T},
                               z::AbstractVector{T}) where {T<:AbstractFloat}
    T_sim = length(z)
    K_ss = ss.aggregates[:K]
    K_series = zeros(T, T_sim)
    K_series[1] = K_ss

    dist = vec(ss.distribution); dist = dist ./ sum(dist)
    last_r = ss.prices[:r]; last_w = ss.prices[:w]
    a_pol = copy(ss.policies[:savings])
    Lambda = _build_transition_matrix(a_pol, grid, income)

    for t in 1:(T_sim - 1)
        K_eff = K_series[t] * exp(z[t])
        prices = price_fn(K_eff, params)
        r_new = prices[:r]; w_new = prices[:w]
        if abs(r_new - last_r) > T(1e-8) || abs(w_new - last_w) > T(1e-8)
            _, a_pol_new = _egm_solve(ip, grid, income, prices; max_iter=50, tol=T(1e-8))
            copyto!(a_pol, a_pol_new)
            Lambda = _build_transition_matrix(a_pol, grid, income)
            last_r = r_new; last_w = w_new
        end
        dist = _forward_iterate(Lambda, dist)
        K_next = max(_aggregate(dist, grid; var_index=1), T(1e-6))
        isfinite(K_next) || (K_next = K_ss)
        K_series[t+1] = K_next
    end
    return K_series
end

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

    # Initialize PLM coefficients: log K' = b[1] + b[2] * log K + b[3] * z
    # Start near identity: K' ≈ K → log K' ≈ 0 + 1 * log K + 0 * z
    b = T[zero(T), one(T), zero(T)]

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

        # Explicit cross-sectional (Young) simulation of the realized capital path.
        # The realized path is independent of the PLM (prices use realized K · exp(z)),
        # so this also serves as the Den Haan reference path.
        K_series = _simulate_explicit_K(ss, ip, grid, income, price_fn, params, z_path)
        log_K_series = log.(K_series)

        # OLS regression: log K_{t+1} = b[1] + b[2] * log K_t + b[3] * z_t
        # Including the aggregate shock z makes the law of motion forecastable in the
        # Den Haan (2010) sense (a z-free PLM yields a degenerate constant path).
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
        X_ols = hcat(ones(T, n_obs), log_K_series[idx_start:idx_end],
                     z_path[idx_start:idx_end])               # [1, log K_t, z_t]

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
# _simulate_explicit_r — explicit clearing-rate path (Huggett zero net supply)
# =============================================================================

"""
    _simulate_explicit_r(ss, ip, grid, income, z, b) → Vector

Simulate the explicit per-period market-clearing rate path of a Huggett economy under the
aggregate endowment shock path `z`. Each period the bond market is cleared by one Newton
step `r = r_guess − A/A_r` (with `A_r` the steady-state savings slope) and the distribution
is forwarded with the cleared policy. The PLM coefficients `b` (`r ≈ b[1] + b[2] z`) only
warm-start the inner clear; the returned path is the **reference** (Den Haan) clearing
path, also reused per outer iteration by `_krusell_smith_huggett`.
"""
function _simulate_explicit_r(ss::HASteadyState{T}, ip::IndividualProblem{T},
                               grid::HAGrid{T}, income::IncomeProcess{T},
                               z::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    T_sim = length(z)
    r_ss = ss.prices[:r]
    dist_ss = vec(ss.distribution); dist_ss = dist_ss ./ sum(dist_ss)

    # Aggregate-savings slope dA/dr at the steady state (fixed Newton-step slope).
    dr0 = T(1e-4)
    _, ap0 = _egm_solve(ip, grid, income, Dict{Symbol,T}(:r => r_ss, :w => one(T));
                        max_iter=1000, tol=T(1e-10))
    _, ap1 = _egm_solve(ip, grid, income, Dict{Symbol,T}(:r => r_ss + dr0, :w => one(T));
                        max_iter=1000, tol=T(1e-10))
    A_r = (sum(vec(ap1) .* dist_ss) - sum(vec(ap0) .* dist_ss)) / dr0
    @assert A_r > zero(T) "Huggett KS: non-positive aggregate-savings slope"

    r_ceil = one(T) / ip.beta - one(T) - T(1e-5)
    r_series = zeros(T, T_sim)
    dist = copy(dist_ss)
    for t in 1:T_sim
        w_t = exp(z[t])
        # One Newton clearing step around the PLM forecast using the SS savings slope A_r
        # (a single step from a near-converged PLM is essentially exact; iterating Newton
        # with the fixed SS slope is unstable far from the steady state). The distribution
        # is forwarded with the cleared policy, keeping ∫a' ≈ 0.
        r_g = length(b) >= 2 ? b[1] + b[2] * z[t] : r_ss
        _, a_pol_g = _egm_solve(ip, grid, income, Dict{Symbol,T}(:r => r_g, :w => w_t);
                                max_iter=80, tol=T(1e-8))
        A_g = sum(vec(a_pol_g) .* dist)
        r_t = clamp(r_g - A_g / A_r, T(-0.2), r_ceil)
        _, a_pol = _egm_solve(ip, grid, income, Dict{Symbol,T}(:r => r_t, :w => w_t);
                              max_iter=80, tol=T(1e-8))
        r_series[t] = r_t
        dist = _forward_iterate(_build_transition_matrix(a_pol, grid, income), dist)
    end
    return r_series
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
    rho = T(rho_e); sig = T(sigma_e); tol_T = T(tol)
    rng = Random.MersenneTwister(seed)
    r_ss = ss.prices[:r]

    # Aggregate endowment shock path (log endowment).
    z = zeros(T, T_sim); innov = randn(rng, T, T_sim)
    for t in 2:T_sim
        z[t] = rho * z[t-1] + sig * innov[t]
    end

    b = T[r_ss, zero(T)]                # PLM: r_t = b[1] + b[2] z_t
    best_r2 = zero(T); converged = false; final_iter = 0

    for outer in 1:max_outer
        final_iter = outer
        # Explicit per-period market clearing (the realized clearing-rate path).
        r_series = _simulate_explicit_r(ss, ip, grid, income, z, b)

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

# =============================================================================
# Den Haan (2010) accuracy test
# =============================================================================

"""
    DenHaanAccuracy{T}

Result of the Den Haan (2010) dynamic accuracy test for a Krusell-Smith solution.

# Fields
- `aggregate::Symbol` — the aggregate compared (`:K` for Aiyagari, `:r` for Huggett)
- `dh_max::T` — maximum error between the reference and PLM-only paths (percent for `:K`,
  percentage points of the rate for `:r`)
- `dh_mean::T` — mean absolute error (same units)
- `sigma_ref::T` — standard deviation of the reference aggregate
- `sigma_plm::T` — standard deviation of the PLM-only aggregate
- `ref_path::Vector{T}` — reference (explicit cross-sectional simulation) path
- `plm_path::Vector{T}` — PLM-only path (law of motion iterated on its own forecasts)
- `T_sim::Int` / `T_burn::Int` — simulation length / burn-in
"""
struct DenHaanAccuracy{T<:AbstractFloat}
    aggregate::Symbol
    dh_max::T
    dh_mean::T
    sigma_ref::T
    sigma_plm::T
    ref_path::Vector{T}
    plm_path::Vector{T}
    T_sim::Int
    T_burn::Int
end

"""
    den_haan_test(ks::KrusellSmithSolution; T_sim=10000, T_burn=1000,
                  rho_z=0.95, sigma_z=0.007, seed=98765) → DenHaanAccuracy

Den Haan (2010) accuracy test for a Krusell-Smith solution. Simulates the aggregate two
ways under the same shock path: a **reference** path from the explicit cross-sectional
(Young) simulation, and a **PLM-only** path that iterates the fitted aggregate law of
motion on its own forecasts *without* re-anchoring to the simulated cross-section. The
maximum and mean errors between the two paths are reported, together with the
standard-deviation comparison. Den Haan (2010) shows that R² and the regression standard
error can look excellent (R² ≈ 0.9999) while the implied σ of aggregate capital is off by
double digits, so this multi-step comparison is the powerful accuracy test.

The aggregate is capital `K` and errors are in percent. The PLM must include the
aggregate shock (`log K' = b₁ + b₂ log K + b₃ z`); otherwise the PLM-only path is
degenerate.

Implemented for the Aiyagari capital model (`spec.model == :aiyagari`), the setting of
Den Haan (2010). For a Huggett solution the cleared aggregate is the risk-free rate (the
bond is in zero net supply), which is driven by the wealth distribution rather than the
shock alone; a robust rate-accuracy test there requires a distribution-augmented PLM and
is out of scope, so `den_haan_test` raises an informative error for `:huggett`.

# References
- den Haan, W. J. (2010). Assessing the accuracy of the aggregate law of motion in models
  with heterogeneous agents. *Journal of Economic Dynamics and Control*, 34(1), 79–99.
"""
function den_haan_test(ks::KrusellSmithSolution{T};
                       T_sim::Int=10000, T_burn::Int=1000,
                       rho_z::Real=0.95, sigma_z::Real=0.007,
                       seed::Int=98765) where {T<:AbstractFloat}
    @assert T_sim > T_burn + 10 "T_sim must exceed T_burn by at least 10"
    if ks.spec.model === :huggett
        error("den_haan_test is implemented for Aiyagari (:aiyagari) Krusell-Smith " *
              "solutions. The Huggett clearing rate is driven by the wealth distribution, " *
              "not the aggregate shock alone, so a meaningful accuracy test there needs a " *
              "distribution-augmented PLM (out of scope).")
    end

    ss = ks.steady_state
    ip = ks.spec.individual
    grid = ks.spec.grid
    income = ks.spec.income
    rng = Random.MersenneTwister(seed)

    params = copy(ks.spec.het_params)
    for (k, v) in (:alpha => T(0.36), :delta => T(0.025), :Z => one(T), :L => one(T))
        haskey(params, k) || (params[k] = T(v))
    end
    b = ks.plm_coefficients[:K]
    @assert length(b) == 3 "den_haan_test expects a z-augmented PLM (log K' = b1 + b2 log K + b3 z)"

    z = zeros(T, T_sim); innov = randn(rng, T, T_sim)
    for t in 2:T_sim
        z[t] = T(rho_z) * z[t-1] + T(sigma_z) * innov[t]
    end

    K_ref = _simulate_explicit_K(ss, ip, grid, income,
                                 _default_cobb_douglas_price_fn, params, z)  # reference
    logK_plm = zeros(T, T_sim); logK_plm[1] = log(ss.aggregates[:K])         # PLM-only
    for t in 1:(T_sim - 1)
        logK_plm[t+1] = b[1] + b[2] * logK_plm[t] + b[3] * z[t]
    end
    K_plm = exp.(logK_plm)

    idx = (T_burn + 1):T_sim
    err = T(100) .* abs.(log.(K_ref[idx]) .- logK_plm[idx])                  # percent
    return DenHaanAccuracy{T}(:K, maximum(err), Statistics.mean(err),
                              Statistics.std(log.(K_ref[idx])),
                              Statistics.std(logK_plm[idx]),
                              K_ref, K_plm, T_sim, T_burn)
end
