# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Bayesian estimation dispatch for Heterogeneous Agent DSGE models.

Provides:
- `estimate_dsge_bayes(::HADSGESpec, ...)` — Bayesian estimation via Random-Walk MH
- `_build_ha_likelihood_fn` — HA-specific likelihood closure (Kalman on reduced system)
- `_update_ha_params` — update parameters across aggregate and heterogeneous blocks
- `_build_ha_observation_equation` — map observables to the reduced state space

The key insight: `HADSGESolution` contains a `DSGESolution` (the linearized reduced
system from SSJ or Reiter), so we reuse the Kalman filter infrastructure directly.

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375-2408.
- Herbst, E. & Schorfheide, F. (2014). Sequential Monte Carlo Sampling for DSGE Models.
  *Journal of Applied Econometrics*, 29(7), 1073-1098.
"""

using LinearAlgebra
using Random
using Statistics
using Distributions

# =============================================================================
# _update_ha_params — update parameters across aggregate and HA blocks
# =============================================================================

"""
    _update_ha_params(spec, param_names, theta) → HADSGESpec{T}

Create a new `HADSGESpec` with updated parameter values. Parameters may live in:
- `spec.aggregate_spec.param_values` (aggregate model params like alpha, rho_z)
- `spec.het_params` (HA-specific params)
- `spec.individual.beta` (discount factor, matched by `:beta` or `:beta_hh`)
"""
function _update_ha_params(spec::HADSGESpec{T}, param_names::Vector{Symbol},
                            theta::Vector{T}) where {T<:AbstractFloat}
    # Copy mutable containers
    new_agg_pv = copy(spec.aggregate_spec.param_values)
    new_het_pv = copy(spec.het_params)
    new_beta = spec.individual.beta

    for (i, pn) in enumerate(param_names)
        if pn === :beta || pn === :beta_hh
            new_beta = theta[i]
        elseif haskey(new_agg_pv, pn)
            new_agg_pv[pn] = theta[i]
        else
            new_het_pv[pn] = theta[i]
        end
    end

    # Rebuild aggregate spec with new param values
    agg = spec.aggregate_spec
    new_agg = DSGESpec{T}(
        agg.endog, agg.exog, agg.params, new_agg_pv,
        agg.equations, agg.residual_fns,
        agg.n_expect, agg.forward_indices, T[], agg.ss_fn;
        original_endog=agg.original_endog,
        original_equations=agg.original_equations,
        augmented=agg.augmented,
        max_lag=agg.max_lag,
        max_lead=agg.max_lead,
        linear=agg.linear
    )

    # Rebuild individual problem with updated beta
    ip = spec.individual
    new_ip = IndividualProblem{T}(
        ip.utility, ip.utility_prime, ip.utility_prime_inv,
        new_beta, ip.budget_fn, ip.borrowing_constraint,
        ip.adjustment_cost, ip.n_asset_dims
    )

    return HADSGESpec{T}(new_agg, new_ip, spec.income, spec.grid,
                          spec.aggregation, new_het_pv; model=spec.model)
end

# =============================================================================
# _build_ha_observation_equation — map observables to the reduced state space
# =============================================================================

"""
    _build_ha_observation_equation(sol, observables, measurement_error) → (Z, d, H)

Build the observation equation matrices (Z, d, H) mapping the reduced linear
state space to observables.

The challenge: the `linear_solution` inside `HADSGESolution` has synthetic
variable names (x_1, x_2, ...) from the Ho-Kalman realization. Observables
refer to aggregate variables (K, Y, r, w, etc.).

Strategy:
1. Try to match observables to the linear solution's `spec.endog` by name.
2. If no name match, use the last `n_obs` states in the reduced system
   (in SSJ, the reduced states capture the dominant modes of aggregate dynamics).

Observation equation: `y_t = Z * x_t + d + v_t`, `v_t ~ N(0, H)`.
"""
function _build_ha_observation_equation(sol::HADSGESolution{T},
                                         observables::Vector{Symbol},
                                         measurement_error) where {T<:AbstractFloat}
    linear = sol.linear_solution
    n_states = size(linear.G1, 1)
    n_obs = length(observables)
    n_obs == 0 && throw(ArgumentError("observables must be non-empty"))

    # Try to map observables to state indices via the linear solution's spec
    obs_indices = Int[]
    for obs in observables
        idx = findfirst(==(obs), linear.spec.endog)
        if idx !== nothing && idx <= n_states
            push!(obs_indices, idx)
        else
            # Fallback: assign to states from the end of the reduced system.
            # In the SSJ Ho-Kalman realization, the leading singular values
            # capture the dominant aggregate dynamics. Use the first available
            # state that has not been assigned yet.
            fallback_idx = min(length(obs_indices) + 1, n_states)
            push!(obs_indices, fallback_idx)
        end
    end

    # Build Z: selection matrix
    Z = zeros(T, n_obs, n_states)
    for (i, j) in enumerate(obs_indices)
        if j <= n_states
            Z[i, j] = one(T)
        end
    end

    # d: steady-state values of observables
    d = zeros(T, n_obs)
    for (i, obs) in enumerate(observables)
        if haskey(sol.steady_state.aggregates, obs)
            d[i] = sol.steady_state.aggregates[obs]
        elseif haskey(sol.steady_state.prices, obs)
            d[i] = sol.steady_state.prices[obs]
        end
    end

    # H: measurement error covariance
    if measurement_error === nothing
        H = T(1e-4) * Matrix{T}(I, n_obs, n_obs)
    else
        me = Vector{T}(measurement_error)
        length(me) == n_obs || throw(ArgumentError(
            "measurement_error length ($(length(me))) must match observables ($n_obs)"))
        H = diagm(me .^ 2)
    end

    return Z, d, H
end

# =============================================================================
# _build_ha_likelihood_fn — HA-specific likelihood closure
# =============================================================================

"""
    _build_ha_likelihood_fn(spec, param_names, data, observables,
                             measurement_error, ha_method, ha_kwargs) → Function

Build a closure `θ → log p(Y|θ)` for Bayesian HA-DSGE estimation.

For each parameter vector:
1. Update the `HADSGESpec` parameters via `_update_ha_params`.
2. Solve the HA model (steady state + linearize via SSJ/Reiter).
3. Extract the reduced linear system from `HADSGESolution.linear_solution`.
4. Build the observation equation mapping observables to the reduced states.
5. Evaluate the Kalman log-likelihood on the reduced linear state space.

Returns `-Inf` on any failure (non-convergence, singular matrices, etc.).
"""
function _build_ha_likelihood_fn(spec::HADSGESpec{T}, param_names::Vector{Symbol},
                                  data::AbstractMatrix, observables::Vector{Symbol},
                                  measurement_error, ha_method::Symbol,
                                  ha_kwargs::NamedTuple) where {T<:AbstractFloat}
    data_mat = Matrix{T}(data)

    function ll_fn(theta::Vector{T})
        try
            # Update parameters in both aggregate_spec and het_params
            new_spec = _update_ha_params(spec, param_names, theta)

            # Solve the HA model (steady state + linearize)
            sol = solve(new_spec; method=ha_method, ha_kwargs...)

            # Validate solution
            if !(sol isa HADSGESolution)
                return T(-Inf)
            end
            if !is_determined(sol)
                return T(-Inf)
            end

            # Extract the reduced linear system
            linear_sol = sol.linear_solution
            if linear_sol === nothing
                return T(-Inf)
            end

            # Build observation equation from the reduced state space
            Z, d, H = _build_ha_observation_equation(sol, observables, measurement_error)
            ss = _build_state_space(linear_sol, Z, d, H)

            # Evaluate Kalman log-likelihood
            ll = _kalman_loglikelihood(ss, data_mat)
            return isfinite(ll) ? ll : T(-Inf)
        catch
            return T(-Inf)
        end
    end

    return ll_fn
end

# =============================================================================
# estimate_dsge_bayes(::HADSGESpec, ...) — main entry point
# =============================================================================

"""
    estimate_dsge_bayes(spec::HADSGESpec{T}, data, θ0; priors, kwargs...) → BayesianDSGE{T}

Bayesian estimation of heterogeneous agent DSGE model parameters via
Random-Walk Metropolis-Hastings.

At each MCMC step, the full HA model is re-solved (steady state + linearization)
and the Kalman filter is evaluated on the reduced linear system. This is the
"offline" approach — computationally intensive but exact.

# Arguments
- `spec::HADSGESpec{T}` — HA-DSGE model specification
- `data::AbstractMatrix` — observed aggregate data in `T×n` (time in rows); orientation is
  resolved by matching a dimension to the number of observables (`n×T` transposed internally).
- `θ0` — initial parameter guess. Preferred: a `Dict{Symbol}`/`NamedTuple` keyed by parameter
  name (order-independent). A positional `AbstractVector` must be in sorted (alphabetical)
  prior-key order and length-matched, else an informative `ArgumentError` is thrown.

# Keywords
- `priors::Dict{Symbol,<:Distribution}` — prior distributions keyed by parameter name
- `observables::Vector{Symbol}` — which aggregate variables are observed (e.g., `[:K]`)
- `n_draws::Int=5000` — total MH draws (including burnin)
- `burnin::Int=1000` — number of burnin draws to discard
- `measurement_error::Union{Nothing,Vector{<:Real}}=nothing` — measurement error SDs
- `ha_method::Symbol=:ssj` — HA solution method (`:ssj` or `:reiter`)
- `ha_kwargs::NamedTuple=(T_horizon=50, n_reduced=15)` — HA solver options
- `proposal_scale::Float64=0.01` — initial RWMH proposal scale (σ² for proposal cov)
- `adapt_interval::Int=100` — adapt proposal covariance every N draws
- `rng::AbstractRNG=Random.default_rng()` — random number generator

# Returns
`BayesianDSGE{T}` containing posterior draws, log posterior, and the solution
at the posterior mean. The `spec` field stores the aggregate `DSGESpec` from
the posterior-mean HA solution.

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375-2408.
- Herbst, E. & Schorfheide, F. (2014). Sequential Monte Carlo Sampling for DSGE Models.
  *Journal of Applied Econometrics*, 29(7), 1073-1098.
"""
function estimate_dsge_bayes(spec::HADSGESpec{T}, data::AbstractMatrix,
                              theta0::Union{AbstractVector{<:Real},
                                            AbstractDict{Symbol,<:Real},NamedTuple};
                              priors::Dict{Symbol,<:Distribution},
                              observables::Vector{Symbol}=Symbol[],
                              n_draws::Int=5000,
                              burnin::Int=1000,
                              measurement_error::Union{Nothing,Vector{<:Real}}=nothing,
                              ha_method::Symbol=:ssj,
                              ha_kwargs::NamedTuple=(T_horizon=50, n_reduced=15),
                              proposal_scale::Float64=0.01,
                              adapt_interval::Int=100,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}

    # ── 1. Build DSGEPrior from priors dict ──────────────────────────────
    lower_dict = Dict{Symbol,Float64}()
    upper_dict = Dict{Symbol,Float64}()
    for (pn, d) in priors
        lo, hi = _infer_prior_bounds(d)
        lower_dict[pn] = lo
        upper_dict[pn] = hi
    end
    prior = DSGEPrior(priors; lower=lower_dict, upper=upper_dict)
    param_names = prior.param_names  # sorted deterministically

    # ── 2. Handle observables ────────────────────────────────────────────
    if isempty(observables)
        # Default: use first few aggregate endogenous variables
        observables = spec.aggregate_spec.endog[1:min(3, length(spec.aggregate_spec.endog))]
    end
    n_obs = length(observables)

    # ── 3. Data handling: resolve orientation by matching n_obs (E-18 / #142) ─
    # Public convention is T×n; the Kalman filter on the reduced system expects
    # n_obs × T_obs internally.
    data_mat = _orient_data(data, n_obs, T)

    # ── 4. Resolve theta0 against sorted prior keys ──────────────────────
    # Dict/NamedTuple resolved by name (order-independent); a positional vector must be
    # in sorted prior-key order and is length-validated (E-12 / H-12 / #136). (The old
    # comment claimed a reordering that never happened — it only cast T.(theta0).)
    n_params = length(param_names)
    theta0_sorted = _resolve_theta0(theta0, param_names, T)

    # ── 5. Build HA likelihood function ──────────────────────────────────
    ll_fn = _build_ha_likelihood_fn(spec, param_names, data_mat, observables,
                                     measurement_error, ha_method, ha_kwargs)

    # ── 6. Random-Walk Metropolis-Hastings ───────────────────────────────
    # Initialize
    theta_current = copy(theta0_sorted)
    ll_current = ll_fn(theta_current)
    lp_current = _log_prior(theta_current, prior)

    # If initial point has -Inf likelihood, try perturbing
    if !isfinite(ll_current) || !isfinite(lp_current)
        for attempt in 1:50
            theta_try = theta_current + T(0.01) * randn(rng, T, n_params)
            for i in 1:n_params
                theta_try[i] = clamp(theta_try[i], prior.lower[i], prior.upper[i])
            end
            ll_try = ll_fn(theta_try)
            lp_try = _log_prior(theta_try, prior)
            if isfinite(ll_try) && isfinite(lp_try)
                theta_current = theta_try
                ll_current = ll_try
                lp_current = lp_try
                break
            end
        end
    end

    # Storage
    draws = zeros(T, n_draws, n_params)
    log_posterior = zeros(T, n_draws)

    # Proposal covariance: scale * I, adapted during sampling
    c2 = T(2.38)^2 / T(n_params)
    proposal_cov = T(proposal_scale)^2 * Matrix{T}(I, n_params, n_params)
    scale_factor = one(T)
    proposal_L = cholesky(Hermitian(proposal_cov)).L

    total_accepted = 0

    for draw in 1:n_draws
        # Propose
        z = randn(rng, T, n_params)
        theta_star = theta_current + scale_factor * proposal_L * z

        # Evaluate prior
        lp_star = _log_prior(theta_star, prior)

        if isfinite(lp_star)
            # Evaluate likelihood (expensive: full HA solve)
            ll_star = ll_fn(theta_star)

            if isfinite(ll_star)
                # MH acceptance ratio
                log_alpha = (ll_star + lp_star) - (ll_current + lp_current)

                if log(rand(rng, T)) < log_alpha
                    theta_current = theta_star
                    ll_current = ll_star
                    lp_current = lp_star
                    total_accepted += 1
                end
            end
        end

        # Store
        draws[draw, :] = theta_current
        log_posterior[draw] = ll_current + lp_current

        # Adapt proposal covariance
        if draw % adapt_interval == 0 && draw >= 2 * adapt_interval
            recent_start = max(1, draw - 5 * adapt_interval)
            recent_draws = draws[recent_start:draw, :]

            if size(recent_draws, 1) > n_params + 1
                sample_cov = cov(recent_draws)
                sample_cov = (sample_cov + sample_cov') / 2
                for i in 1:n_params
                    if sample_cov[i, i] < T(1e-10)
                        sample_cov[i, i] = T(1e-10)
                    end
                end

                proposal_cov = c2 * sample_cov
                proposal_L = try
                    cholesky(Hermitian(proposal_cov)).L
                catch
                    proposal_cov += T(1e-6) * I
                    cholesky(Hermitian(proposal_cov)).L
                end
            end

            # Adapt scale factor targeting ~23.4% acceptance
            recent_acc = total_accepted / draw
            if recent_acc > T(0.30)
                scale_factor *= T(1.1)
            elseif recent_acc < T(0.15)
                scale_factor *= T(0.9)
            end
        end
    end

    acceptance_rate = T(total_accepted) / T(n_draws)

    # ── 7. Discard burn-in ───────────────────────────────────────────────
    post_draws = draws[burnin+1:end, :]
    post_log_posterior = log_posterior[burnin+1:end]

    # ── 8. Build solution at posterior mean ───────────────────────────────
    theta_mean = vec(mean(post_draws; dims=1))
    new_spec_mean = _update_ha_params(spec, param_names, theta_mean)

    # Solve at posterior mean for the result container
    sol_mean = try
        solve(new_spec_mean; method=ha_method, ha_kwargs...)
    catch
        # Fallback: solve at initial theta
        solve(spec; method=ha_method, ha_kwargs...)
    end

    # Extract the linear solution and build state space
    if sol_mean isa HADSGESolution
        linear_sol = sol_mean.linear_solution
        Z, d, H = _build_ha_observation_equation(sol_mean, observables, measurement_error)
        ss_result = _build_state_space(linear_sol, Z, d, H)

        # Log marginal likelihood via Geweke (1999) modified harmonic mean (E-04 / #130).
        # `post_log_posterior` stores the per-draw kernel log L(θ) + log π(θ); the MHM is
        # on the same additive scale as the SMC estimator. Returns NaN + @warn on a chain
        # too short to build the truncated-normal weighting density.
        log_ml = _geweke_mhm(post_draws, post_log_posterior)

        return BayesianDSGE{T}(
            post_draws, post_log_posterior, param_names, prior,
            log_ml, :rwmh, acceptance_rate,
            T[], T[],  # no ESS history or phi schedule for MH
            linear_sol.spec, linear_sol, ss_result
        )
    else
        # KrusellSmithSolution or other — build a minimal DSGESolution
        error("Bayesian estimation requires :ssj or :reiter method " *
              "(produces HADSGESolution with a linear system for Kalman filter)")
    end
end
