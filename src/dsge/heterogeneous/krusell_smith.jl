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
# Krusell-Smith household policy over the extended state (a, e, K, z)
# =============================================================================
#
# The bug this fixes: the old solver re-solved a STATIONARY EGM at each period's
# contemporaneous realized prices, so the household policy — and the simulated
# capital path — were completely independent of the perceived law of motion (PLM).
# The outer loop was therefore vacuous (iterating the PLM changed nothing).
#
# The correct Krusell-Smith (1998) household conditions on the aggregate state
# (K, z) and forecasts next-period capital K' through the PLM
# `log K' = b1 + b2 log K + b3 z`. Its policy is a genuine function c(a, e, K, z):
# in the EGM backward step the continuation marginal utility is evaluated at the
# NEXT-period prices r(K', z'), w(K', z') implied by the PLM forecast K' and the
# aggregate-shock transition z→z', so perturbing b re-solves to a different policy.

"""
    _ks_build_z_grid(rho_z, sigma_z, n_z) → (z_grid::Vector, z_trans::Matrix)

Discretize the aggregate log-TFP AR(1) `z' = ρ_z z + σ_z ε` onto an `n_z`-state
Rouwenhorst grid for the Krusell-Smith aggregate state.
"""
function _ks_build_z_grid(rho_z::T, sigma_z::T, n_z::Int) where {T<:AbstractFloat}
    proc = rouwenhorst(rho_z, sigma_z, n_z)
    return T.(proc.states), T.(proc.transition)
end

"""
    _ks_egm_solve(ip, grid, income, b, z_grid, z_trans, K_grid, price_fn, params;
                  max_iter=100, tol=1e-6, init_policy=nothing) → (c_pol, converged)

Solve the Krusell-Smith household policy `c(a, e, K, z)` by EGM, given the PLM
coefficients `b` (`log K' = b[1] + b[2] log K + b[3] z`). Prices at an aggregate node
`(K, z)` come from the firm FOC at effective capital `K·exp(z)` (matching the
simulation convention); the Euler continuation is taken at the PLM-forecast
`K' = exp(b·[1, log K, z])` and integrated over the idiosyncratic (`Π`) and aggregate
(`z_trans`) transitions with the next-period gross return `1 + r(K', z')`.

Returns the `n_a × n_e × n_K × n_z` consumption policy and a convergence flag.
"""
function _ks_egm_solve(ip::IndividualProblem{T}, grid::HAGrid{T},
                       income::IncomeProcess{T}, b::AbstractVector{T},
                       z_grid::AbstractVector{T}, z_trans::AbstractMatrix{T},
                       K_grid::AbstractVector{T}, price_fn::Function,
                       params::Dict{Symbol,T};
                       max_iter::Int=100, tol::T=T(1e-6),
                       init_policy=nothing) where {T<:AbstractFloat}
    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = length(income.states)
    e_vals = income.states
    n_z = length(z_grid)
    n_K = length(K_grid)
    a_min = ip.borrowing_constraint[1]
    beta = ip.beta
    up = ip.utility_prime
    upi = ip.utility_prime_inv
    Pi = income.transition

    # Prices at each aggregate node (K, z): firm FOC at effective capital K·exp(z).
    r_node = zeros(T, n_K, n_z)
    w_node = zeros(T, n_K, n_z)
    for lz in 1:n_z, kK in 1:n_K
        p = price_fn(K_grid[kK] * exp(z_grid[lz]), params)
        r_node[kK, lz] = p[:r]
        w_node[kK, lz] = p[:w]
    end
    # PLM capital forecast K'(K, z), and next-period rate r(K', z') for each z'.
    Kp_node = zeros(T, n_K, n_z)
    rp_node = zeros(T, n_K, n_z, n_z)   # [kK, lz, lp] → r(K'(K,z), z')
    for lz in 1:n_z, kK in 1:n_K
        Kp = exp(b[1] + b[2] * log(K_grid[kK]) + b[3] * z_grid[lz])
        Kp_node[kK, lz] = Kp
        for lp in 1:n_z
            rp_node[kK, lz, lp] = price_fn(Kp * exp(z_grid[lp]), params)[:r]
        end
    end

    c_pol = zeros(T, n_a, n_e, n_K, n_z)
    if init_policy !== nothing
        copyto!(c_pol, init_policy)
    else
        for lz in 1:n_z, kK in 1:n_K, je in 1:n_e, ia in 1:n_a
            coh = (one(T) + r_node[kK, lz]) * a_grid[ia] + w_node[kK, lz] * e_vals[je]
            c_pol[ia, je, kK, lz] = max(T(0.05) * coh, T(1e-10))
        end
    end

    c_new = similar(c_pol)
    emu = zeros(T, n_a)
    c_endo = zeros(T, n_a)
    a_endo = zeros(T, n_a)
    converged = false
    for _ in 1:max_iter
        for lz in 1:n_z
            for kK in 1:n_K
                r = r_node[kK, lz]
                w = w_node[kK, lz]
                Kp = Kp_node[kK, lz]
                for je in 1:n_e
                    fill!(emu, zero(T))
                    @inbounds for i in 1:n_a         # a' index (on a_grid)
                        acc = zero(T)
                        for jp in 1:n_e
                            Pje = Pi[je, jp]
                            Pje < T(1e-20) && continue
                            for lp in 1:n_z
                                ztr = z_trans[lz, lp]
                                ztr < T(1e-20) && continue
                                # c'(a', e'=jp, K', z'=lp): interpolate in K at K'
                                cp = _linear_interp(K_grid, view(c_pol, i, jp, :, lp), Kp)
                                acc += Pje * ztr * (one(T) + rp_node[kK, lz, lp]) * up(cp)
                            end
                        end
                        emu[i] = acc
                    end
                    for i in 1:n_a
                        c_endo[i] = upi(beta * emu[i])
                        a_endo[i] = (c_endo[i] + a_grid[i] - w * e_vals[je]) / (one(T) + r)
                    end
                    for i in 1:n_a
                        a_val = a_grid[i]
                        if a_val < a_endo[1]
                            coh = (one(T) + r) * a_val + w * e_vals[je]
                            c_new[i, je, kK, lz] = max(coh - a_min, T(1e-10))
                        else
                            c_new[i, je, kK, lz] = _linear_interp(a_endo, c_endo, a_val)
                        end
                    end
                end
            end
        end
        max_diff = maximum(abs.(c_new .- c_pol))
        copyto!(c_pol, c_new)
        if max_diff < tol
            converged = true
            break
        end
    end
    return c_pol, converged
end

"""
    _ks_savings_at(c_pol, K_t, lz, a_grid, e_vals, K_grid, r_kz, w_kz, a_min) → Matrix

Extract the 2-D savings policy `a'(a, e)` at the realized aggregate state
`(K_t, z_index lz)` from the 4-D consumption policy by interpolating in the `K`
dimension at `K_t`.
"""
function _ks_savings_at(c_pol::Array{T,4}, K_t::T, lz::Int,
                        a_grid::AbstractVector{T}, e_vals::AbstractVector{T},
                        K_grid::AbstractVector{T}, r_kz::T, w_kz::T,
                        a_min::T) where {T<:AbstractFloat}
    n_a = length(a_grid)
    n_e = length(e_vals)
    a_pol = zeros(T, n_a, n_e)
    @inbounds for je in 1:n_e
        for ia in 1:n_a
            c = _linear_interp(K_grid, view(c_pol, ia, je, :, lz), K_t)
            coh = (one(T) + r_kz) * a_grid[ia] + w_kz * e_vals[je]
            a_pol[ia, je] = max(coh - c, a_min)
        end
    end
    return a_pol
end

"""
    _ks_simulate(c_pol, ss, grid, income, z_idx_path, z_grid, K_grid, price_fn, params)
        → K_series

Simulate the realized aggregate capital path under the converged Krusell-Smith
household policy `c_pol(a,e,K,z)` and a discrete aggregate-shock index path
`z_idx_path` (indices into `z_grid`). Each period the savings policy at the realized
`(K_t, z_t)` is obtained by K-interpolation, the Young cross-section is advanced one
period, and `K_{t+1}` is the resulting aggregate — so the path genuinely depends on
the PLM baked into `c_pol`.
"""
function _ks_simulate(c_pol::Array{T,4}, ss::HASteadyState{T}, grid::HAGrid{T},
                      income::IncomeProcess{T}, z_idx_path::AbstractVector{Int},
                      z_grid::AbstractVector{T}, K_grid::AbstractVector{T},
                      price_fn::Function, params::Dict{Symbol,T}) where {T<:AbstractFloat}
    T_sim = length(z_idx_path)
    a_grid = grid.grids[1]
    e_vals = income.states
    a_min = ss.grid.grids[1][1]
    K_ss = ss.aggregates[:K]
    K_series = zeros(T, T_sim)
    K_series[1] = K_ss
    dist = vec(ss.distribution); dist = dist ./ sum(dist)

    for t in 1:(T_sim - 1)
        K_t = K_series[t]
        lz = z_idx_path[t]
        p = price_fn(K_t * exp(z_grid[lz]), params)
        a_pol = _ks_savings_at(c_pol, K_t, lz, a_grid, e_vals, K_grid,
                               p[:r], p[:w], a_min)
        Lambda = _build_transition_matrix(a_pol, grid, income)
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
                               n_z::Int=3,
                               n_K::Int=5,
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
    K_ss = ss.aggregates[:K]

    # Aggregate-state grids: log-TFP z (Rouwenhorst) and capital K (log-spaced ±40%).
    z_grid, z_trans = _ks_build_z_grid(rho_z_T, sigma_z_T, n_z)
    K_grid = K_ss .* exp.(T.(collect(range(-T(0.4), T(0.4); length=n_K))))

    # Discrete aggregate-shock index path (Markov chain on z_grid), fixed across outer
    # iterations. Its realized z levels drive both the policy-consistent simulation and
    # the PLM regression.
    z_idx = zeros(Int, T_sim)
    z_idx[1] = div(n_z + 1, 2)                       # start at the central (z≈0) state
    z_cdf = cumsum(z_trans; dims=2)
    for t in 2:T_sim
        u = rand(rng, T)
        j = z_idx[t-1]
        z_idx[t] = clamp(searchsortedfirst(view(z_cdf, j, :), u), 1, n_z)
    end
    z_real = T[z_grid[z_idx[t]] for t in 1:T_sim]    # realized z levels

    # Initialize PLM: log K' = b[1] + b[2] log K + b[3] z (start at the identity law).
    b = T[zero(T), one(T), zero(T)]

    converged = false
    final_iter = 0
    best_r2 = zero(T)

    # Warm-start the 4-D policy from the steady-state consumption policy broadcast
    # across the (K, z) nodes — close to the solution for K near K_ss, so the EGM
    # (which contracts only at rate β ≈ 0.99 from a cold start) converges quickly.
    c_ss = ss.policies[:consumption]
    n_a0 = size(c_ss, 1); n_e0 = size(c_ss, 2)
    c_pol = Array{T,4}(undef, n_a0, n_e0, n_K, n_z)
    for lz in 1:n_z, kK in 1:n_K
        @views c_pol[:, :, kK, lz] .= c_ss
    end

    for outer in 1:max_outer
        final_iter = outer

        # Solve the household policy c(a,e,K,z) at the CURRENT PLM (warm-started), then
        # simulate the Young cross-section forward with that policy. Unlike the old
        # myopic re-solve at realized prices, this realized path genuinely depends on
        # the PLM b (perturbing b re-solves to a different policy and a different path).
        c_pol, _ = _ks_egm_solve(ip, grid, income, b, z_grid, z_trans, K_grid,
                                 price_fn, params; max_iter=1000, tol=T(1e-6),
                                 init_policy=c_pol)
        K_series = _ks_simulate(c_pol, ss, grid, income, z_idx, z_grid, K_grid,
                                price_fn, params)
        log_K_series = log.(K_series)

        n_obs = T_sim - T_burn - 1
        n_obs < 10 && break

        idx_start = T_burn + 1
        idx_end = T_sim - 1
        y_ols = log_K_series[(idx_start + 1):(idx_end + 1)]        # log K_{t+1}
        X_ols = hcat(ones(T, n_obs), log_K_series[idx_start:idx_end],
                     z_real[idx_start:idx_end])                    # [1, log K_t, z_t]
        b_new = X_ols \ y_ols

        y_hat = X_ols * b_new
        ss_res = sum((y_ols .- y_hat) .^ 2)
        ss_tot = sum((y_ols .- mean(y_ols)) .^ 2)
        best_r2 = ss_tot > zero(T) ? one(T) - ss_res / ss_tot : one(T)

        coef_diff = maximum(abs.(b_new .- b))
        if coef_diff < tol_T
            converged = true
            b = b_new
            break
        end
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

    # Aggregate-state grids (same discretization the solver used) and a discrete
    # aggregate-shock index path (Markov chain on z_grid).
    n_z = 3; n_K = 5
    z_grid, z_trans = _ks_build_z_grid(T(rho_z), T(sigma_z), n_z)
    K_ss = ss.aggregates[:K]
    K_grid = K_ss .* exp.(T.(collect(range(-T(0.4), T(0.4); length=n_K))))
    z_idx = zeros(Int, T_sim); z_idx[1] = div(n_z + 1, 2)
    z_cdf = cumsum(z_trans; dims=2)
    for t in 2:T_sim
        u = rand(rng, T); j = z_idx[t-1]
        z_idx[t] = clamp(searchsortedfirst(view(z_cdf, j, :), u), 1, n_z)
    end
    z_real = T[z_grid[z_idx[t]] for t in 1:T_sim]

    # Reference path: the PLM-DEPENDENT cross-sectional simulation under the household
    # policy re-solved at the fitted PLM b (not a myopic re-solve at realized prices),
    # warm-started from the SS policy for fast EGM convergence.
    c_ss = ss.policies[:consumption]
    c0 = Array{T,4}(undef, size(c_ss, 1), size(c_ss, 2), n_K, n_z)
    for lz in 1:n_z, kK in 1:n_K
        @views c0[:, :, kK, lz] .= c_ss
    end
    c_pol, _ = _ks_egm_solve(ip, grid, income, b, z_grid, z_trans, K_grid,
                             _default_cobb_douglas_price_fn, params;
                             max_iter=1000, tol=T(1e-6), init_policy=c0)
    K_ref = _ks_simulate(c_pol, ss, grid, income, z_idx, z_grid, K_grid,
                         _default_cobb_douglas_price_fn, params)

    # PLM-only path: iterate the fitted law on its own forecasts (no cross-section).
    logK_plm = zeros(T, T_sim); logK_plm[1] = log(K_ss)
    for t in 1:(T_sim - 1)
        logK_plm[t+1] = b[1] + b[2] * logK_plm[t] + b[3] * z_real[t]
    end
    K_plm = exp.(logK_plm)

    idx = (T_burn + 1):T_sim
    err = T(100) .* abs.(log.(K_ref[idx]) .- logK_plm[idx])                  # percent
    return DenHaanAccuracy{T}(:K, maximum(err), Statistics.mean(err),
                              Statistics.std(log.(K_ref[idx])),
                              Statistics.std(logK_plm[idx]),
                              K_ref, K_plm, T_sim, T_burn)
end
