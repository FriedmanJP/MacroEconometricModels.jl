# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Sequential Monte Carlo (SMC) sampler and adaptive Random-Walk Metropolis-Hastings
for Bayesian DSGE estimation.

Implements:
- `_build_likelihood_fn` — closure mapping θ → log p(Y|θ) via Kalman filter
- `_log_prior` — log prior density evaluation with bound checking
- `_adaptive_tempering` — bisection for ESS-targeted tempering schedule
- `_update_proposal_cov!` — weighted covariance for RWMH proposals
- `_smc_mutation!` — parallel RWMH mutation of SMC particles
- `_smc_sample` — full SMC algorithm (Herbst & Schorfheide 2014)
- `_mh_sample` — adaptive RWMH with Roberts & Rosenthal (2001) tuning

References:
- Herbst, E. & Schorfheide, F. (2014). Sequential Monte Carlo Sampling for DSGE Models.
  Journal of Applied Econometrics, 29(7), 1073-1098.
- Roberts, G. O. & Rosenthal, J. S. (2001). Optimal Scaling for Various
  Metropolis-Hastings Algorithms. Statistical Science, 16(4), 351-367.
"""

using LinearAlgebra
using Random
using Statistics
using Distributions

# =============================================================================
# Likelihood function builder
# =============================================================================

"""
    _build_likelihood_fn(spec, param_names, data, observables, measurement_error,
                         solver, solver_kwargs)

Build a closure `θ_vec → log p(Y|θ)` for Bayesian DSGE estimation.

Updates the spec parameter values, computes steady state, solves the model,
builds a `DSGEStateSpace`, and evaluates the Kalman log-likelihood.
Returns `-Inf` on any failure (non-convergence, singular matrices, etc.).

# Arguments
- `spec::DSGESpec{T}` — model specification (template)
- `param_names::Vector{Symbol}` — which parameters θ maps to
- `data::Matrix{T}` — n_obs × T_obs data matrix
- `observables::Vector{Symbol}` — observed endogenous variables
- `measurement_error` — measurement error SDs or `nothing`
- `solver::Symbol` — solver method (e.g., `:gensys`)
- `solver_kwargs::NamedTuple` — additional keyword arguments for `solve`

# Returns
Closure `(θ::Vector{T}) → T` returning log-likelihood or `-Inf`.
"""
function _build_likelihood_fn(spec::DSGESpec{T}, param_names::Vector{Symbol},
                               data::AbstractMatrix, observables::Vector{Symbol},
                               measurement_error, solver::Symbol,
                               solver_kwargs::NamedTuple) where {T<:AbstractFloat}
    data = Matrix{T}(data)  # convert Adjoint/SubArray to concrete Matrix
    # Pre-build observation equation from spec (indices don't change across θ)
    # But steady state changes, so we rebuild each time

    function ll_fn(theta::Vector{T})
        try
            # Update parameter values
            new_pv = copy(spec.param_values)
            for (i, pn) in enumerate(param_names)
                new_pv[pn] = theta[i]
            end

            # Create new spec with updated parameters
            new_spec = DSGESpec{T}(
                spec.endog, spec.exog, spec.params, new_pv,
                spec.equations, spec.residual_fns,
                spec.n_expect, spec.forward_indices, T[], spec.ss_fn;
                original_endog=spec.original_endog,
                original_equations=spec.original_equations,
                augmented=spec.augmented,
                max_lag=spec.max_lag,
                max_lead=spec.max_lead
            )

            # Compute steady state
            new_spec = compute_steady_state(new_spec)

            # Solve model
            sol = solve(new_spec; method=solver, solver_kwargs...)
            if !is_determined(sol)
                return T(-Inf)
            end

            # Build observation equation and state space
            Z, d, H = _build_observation_equation(new_spec, observables, measurement_error)
            ss = _build_state_space(sol, Z, d, H)

            # Evaluate Kalman log-likelihood
            ll = _kalman_loglikelihood(ss, data)

            return isfinite(ll) ? ll : T(-Inf)
        catch
            return T(-Inf)
        end
    end

    return ll_fn
end

# =============================================================================
# Log prior density
# =============================================================================

"""
    _log_prior(θ::Vector{T}, prior::DSGEPrior{T}) where {T}

Evaluate the log prior density at parameter vector `θ`.

Returns the sum of marginal log-prior densities. Returns `-Inf` if any
parameter is outside its specified bounds.
"""
function _log_prior(theta::AbstractVector{T}, prior::DSGEPrior{T}) where {T<:AbstractFloat}
    n = length(theta)
    lp = zero(T)
    @inbounds for i in 1:n
        # Check bounds
        if theta[i] < prior.lower[i] || theta[i] > prior.upper[i]
            return T(-Inf)
        end
        lp_i = logpdf(prior.distributions[i], theta[i])
        if !isfinite(lp_i)
            return T(-Inf)
        end
        lp += lp_i
    end
    return lp
end

# =============================================================================
# Adaptive tempering
# =============================================================================

"""
    _adaptive_tempering(log_liks, phi_old, ess_target, N)

Find `phi_new` via bisection such that the ESS of the incremental importance
weights `exp((phi_new - phi_old) * log_liks)` equals `ess_target * N`.

If ESS at `phi = 1.0` exceeds the target, returns `1.0` (final step).
Maximum 50 bisection iterations.

# Arguments
- `log_liks::Vector{T}` — per-particle log-likelihoods
- `phi_old::T` — current tempering parameter
- `ess_target::Real` — target ESS as fraction of N (e.g., 0.5)
- `N::Int` — number of particles

# Returns
`phi_new::T` — next tempering parameter in `(phi_old, 1.0]`.
"""
function _adaptive_tempering(log_liks::AbstractVector{T}, phi_old::T,
                              ess_target::Real, N::Int) where {T<:AbstractFloat}
    target_ess = T(ess_target) * N

    # Helper: compute ESS at a given phi
    function _compute_ess(phi)
        delta_phi = phi - phi_old
        inc_log_w = delta_phi .* log_liks
        # Normalize in log space
        lse = _logsumexp(inc_log_w)
        w = exp.(inc_log_w .- lse)
        return one(T) / sum(abs2, w)
    end

    # Check if we can jump straight to phi = 1
    ess_at_one = _compute_ess(one(T))
    if ess_at_one >= target_ess
        return one(T)
    end

    # Bisection between phi_old and 1.0
    lo = phi_old
    hi = one(T)
    phi_new = (lo + hi) / 2

    for _ in 1:50
        phi_new = (lo + hi) / 2
        ess = _compute_ess(phi_new)

        if abs(ess - target_ess) < one(T)
            break
        end

        if ess > target_ess
            lo = phi_new  # ESS too high → increase phi (more tempering)
        else
            hi = phi_new  # ESS too low → decrease phi (less tempering)
        end

        if hi - lo < T(1e-12)
            break
        end
    end

    return phi_new
end

# =============================================================================
# Proposal covariance update
# =============================================================================

"""
    _update_proposal_cov!(state::SMCState{T}) where {T}

Update the SMC proposal covariance as `c² * Σ_weighted` where
`c = 2.38 / sqrt(n_params)` (Roberts & Rosenthal 2001 optimal scaling)
and `Σ_weighted` is the weighted covariance of the theta particles.

Modifies `state.proposal_cov` in-place.
"""
function _update_proposal_cov!(state::SMCState{T}) where {T<:AbstractFloat}
    n_params, N = size(state.theta_particles)

    # Normalize weights
    w = exp.(state.log_weights .- _logsumexp(state.log_weights))

    # Weighted mean
    mu = zeros(T, n_params)
    @inbounds for j in 1:N
        wj = w[j]
        @simd for i in 1:n_params
            mu[i] += wj * state.theta_particles[i, j]
        end
    end

    # Weighted covariance
    cov_mat = zeros(T, n_params, n_params)
    @inbounds for j in 1:N
        wj = w[j]
        for p in 1:n_params
            dp = state.theta_particles[p, j] - mu[p]
            @simd for q in 1:n_params
                dq = state.theta_particles[q, j] - mu[q]
                cov_mat[p, q] += wj * dp * dq
            end
        end
    end

    # Symmetrize
    cov_mat = (cov_mat + cov_mat') / 2

    # Scale by optimal factor c² = (2.38)² / n_params
    c2 = T(2.38)^2 / T(n_params)
    state.proposal_cov .= c2 .* cov_mat

    # Ensure positive definiteness with regularization
    min_diag = T(1e-10)
    @inbounds for i in 1:n_params
        if state.proposal_cov[i, i] < min_diag
            state.proposal_cov[i, i] = min_diag
        end
    end

    return nothing
end

# =============================================================================
# SMC mutation step
# =============================================================================

"""
    _smc_mutation!(state, phi, ll_fn, prior, n_mh_steps, rng)

Mutate SMC particles via `n_mh_steps` random-walk Metropolis-Hastings steps.

For each particle (parallelized with `Threads.@threads`): propose
`θ* = θ + L * z` where `L = cholesky(proposal_cov).L`, `z ~ N(0, I)`,
and accept with log ratio `(phi * ll* + lp*) - (phi * ll_old + lp_old)`.

Tracks acceptance rate across all particles and steps.

# Arguments
- `state::SMCState{T}` — SMC state (particles, weights, proposal covariance)
- `phi::T` — current tempering parameter
- `ll_fn` — log-likelihood closure `θ → log p(Y|θ)`
- `prior::DSGEPrior{T}` — prior specification
- `n_mh_steps::Int` — number of MH steps per particle
- `rng::AbstractRNG` — random number generator (used to seed per-thread RNGs)
"""
function _smc_mutation!(state::SMCState{T}, phi::T, ll_fn, prior::DSGEPrior{T},
                         n_mh_steps::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    n_params, N = size(state.theta_particles)

    # Compute Cholesky of proposal covariance
    L = try
        cholesky(Hermitian(state.proposal_cov)).L
    catch
        # Add jitter if not positive definite
        cov_jittered = state.proposal_cov + T(1e-6) * I
        cholesky(Hermitian(cov_jittered)).L
    end

    # Track total acceptances
    total_accepted = Threads.Atomic{Int}(0)
    total_proposed = Threads.Atomic{Int}(0)

    Threads.@threads for j in 1:N
        # Per-thread RNG for reproducibility
        thread_rng = Random.MersenneTwister(hash((j, rand(rng, UInt64))))

        theta_j = state.theta_particles[:, j]
        ll_j = state.log_likelihoods[j]
        lp_j = state.log_priors[j]

        for _ in 1:n_mh_steps
            # Propose: theta_star = theta_j + L * z
            z = randn(thread_rng, T, n_params)
            theta_star = theta_j + L * z

            # Evaluate prior
            lp_star = _log_prior(theta_star, prior)
            if lp_star == T(-Inf)
                Threads.atomic_add!(total_proposed, 1)
                continue
            end

            # Evaluate likelihood
            ll_star = ll_fn(theta_star)
            if ll_star == T(-Inf)
                Threads.atomic_add!(total_proposed, 1)
                continue
            end

            # MH acceptance ratio (tempered)
            log_alpha = (phi * ll_star + lp_star) - (phi * ll_j + lp_j)

            if log(rand(thread_rng, T)) < log_alpha
                theta_j = theta_star
                ll_j = ll_star
                lp_j = lp_star
                Threads.atomic_add!(total_accepted, 1)
            end

            Threads.atomic_add!(total_proposed, 1)
        end

        # Write back
        state.theta_particles[:, j] = theta_j
        state.log_likelihoods[j] = ll_j
        state.log_priors[j] = lp_j
    end

    # Record acceptance rate
    tp = total_proposed[]
    acc_rate = tp > 0 ? T(total_accepted[]) / T(tp) : zero(T)
    push!(state.acceptance_rates, acc_rate)

    return nothing
end

# =============================================================================
# Full SMC algorithm (Herbst & Schorfheide 2014)
# =============================================================================

"""
    _smc_sample(spec, data, param_names, prior, θ0; kwargs...) → SMCState

Full Sequential Monte Carlo sampler for Bayesian DSGE estimation.

Algorithm (Herbst & Schorfheide 2014):
1. Initialize `N_smc` particles from prior distributions
2. Evaluate log-likelihood for each particle (parallel with `Threads.@threads`)
3. Adaptive tempering loop:
   a. Find `phi_new` via `_adaptive_tempering` targeting `ess_target * N`
   b. Compute incremental weights, update log marginal likelihood
   c. Resample if ESS drops (using `_systematic_resample!`)
   d. Update proposal covariance via `_update_proposal_cov!`
   e. Mutate particles via `_smc_mutation!`
   f. Stop when `phi = 1`

# Arguments
- `spec::DSGESpec{T}` — model specification
- `data::Matrix{T}` — n_obs × T_obs data matrix
- `param_names::Vector{Symbol}` — parameters to estimate
- `prior::DSGEPrior{T}` — prior specification
- `θ0::Vector{T}` — initial parameter guess (used for fallback)

# Keywords
- `n_smc::Int=500` — number of SMC particles
- `n_mh_steps::Int=1` — MH steps per mutation stage
- `ess_target::Real=0.5` — target ESS fraction
- `observables::Vector{Symbol}` — observed variables
- `measurement_error` — measurement error SDs or `nothing`
- `solver::Symbol=:gensys` — DSGE solver method
- `solver_kwargs::NamedTuple` — additional solver kwargs
- `rng::AbstractRNG` — random number generator

# Returns
`SMCState{T}` with posterior particles, weights, tempering schedule, and log
marginal likelihood estimate.

# References
- Herbst, E. & Schorfheide, F. (2014). Sequential Monte Carlo Sampling for DSGE Models.
  *Journal of Applied Econometrics*, 29(7), 1073-1098.
"""
function _smc_sample(spec::DSGESpec{T}, data::AbstractMatrix,
                      param_names::Vector{Symbol}, prior::DSGEPrior{T},
                      theta0::AbstractVector{T};
                      n_smc::Int=500, n_mh_steps::Int=1,
                      ess_target::Real=0.5,
                      observables::Vector{Symbol}=spec.endog,
                      measurement_error=nothing,
                      solver::Symbol=:gensys,
                      solver_kwargs::NamedTuple=NamedTuple(),
                      rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    data = Matrix{T}(data)  # convert Adjoint/SubArray to concrete Matrix
    n_params = length(param_names)
    N = n_smc

    # Build likelihood function
    ll_fn = _build_likelihood_fn(spec, param_names, data, observables,
                                  measurement_error, solver, solver_kwargs)

    # Initialize particles from prior
    theta_particles = zeros(T, n_params, N)
    log_priors = zeros(T, N)

    for j in 1:N
        for i in 1:n_params
            # Draw from prior, clipped to bounds
            for attempt in 1:100
                draw = T(rand(rng, prior.distributions[i]))
                if draw >= prior.lower[i] && draw <= prior.upper[i]
                    theta_particles[i, j] = draw
                    break
                end
                if attempt == 100
                    # Fallback to midpoint
                    lo = isfinite(prior.lower[i]) ? prior.lower[i] : T(0)
                    hi = isfinite(prior.upper[i]) ? prior.upper[i] : T(1)
                    theta_particles[i, j] = (lo + hi) / 2
                end
            end
        end
        log_priors[j] = _log_prior(theta_particles[:, j], prior)
    end

    # Initialize weights uniformly
    log_weights = fill(-log(T(N)), N)

    # Evaluate log-likelihoods (parallel)
    log_likelihoods = fill(T(-Inf), N)
    Threads.@threads for j in 1:N
        log_likelihoods[j] = ll_fn(theta_particles[:, j])
    end

    # Initialize SMC state
    state = SMCState{T}(
        theta_particles,
        log_weights,
        log_likelihoods,
        log_priors,
        T[zero(T)],          # phi_schedule starts at 0
        T[],                  # ess_history
        T[],                  # acceptance_rates
        zero(T),              # log_marginal_likelihood
        PFWorkspace{T}[],     # pf_workspace_pool (unused for Kalman)
        Matrix{T}(I, n_params, n_params)  # proposal_cov
    )

    # Resampling buffers
    weights_normalized = fill(one(T) / N, N)
    ancestors = collect(1:N)
    cumweights = zeros(T, N)

    # Adaptive tempering loop
    phi = zero(T)

    while phi < one(T)
        # Find next tempering parameter
        # Only use particles with finite log-likelihoods for tempering
        valid_lls = copy(state.log_likelihoods)
        for j in 1:N
            if !isfinite(valid_lls[j])
                valid_lls[j] = T(-1e10)  # large negative but finite
            end
        end

        phi_new = _adaptive_tempering(valid_lls, phi, ess_target, N)
        delta_phi = phi_new - phi

        # Compute incremental log weights
        inc_log_w = delta_phi .* state.log_likelihoods
        # Handle -Inf likelihoods
        for j in 1:N
            if !isfinite(inc_log_w[j])
                inc_log_w[j] = T(-1e10)
            end
        end

        # Update log marginal likelihood: log p(Y) += log(1/N * sum exp(inc_log_w))
        state.log_marginal_likelihood += _logsumexp(inc_log_w) - log(T(N))

        # Update particle log weights
        state.log_weights .+= inc_log_w

        # Normalize weights for ESS computation and resampling
        _normalize_log_weights!(weights_normalized, state.log_weights)

        # Compute ESS
        ess = one(T) / sum(abs2, weights_normalized)
        push!(state.ess_history, ess)

        # Record tempering step
        push!(state.phi_schedule, phi_new)

        # Resample if ESS is low
        if ess < T(ess_target) * N
            _systematic_resample!(ancestors, weights_normalized, cumweights, N, rng)

            # Resample theta particles, likelihoods, priors
            theta_new = similar(state.theta_particles)
            ll_new = similar(state.log_likelihoods)
            lp_new = similar(state.log_priors)
            @inbounds for j in 1:N
                a = ancestors[j]
                theta_new[:, j] = state.theta_particles[:, a]
                ll_new[j] = state.log_likelihoods[a]
                lp_new[j] = state.log_priors[a]
            end
            state.theta_particles .= theta_new
            state.log_likelihoods .= ll_new
            state.log_priors .= lp_new

            # Reset weights to uniform
            fill!(state.log_weights, -log(T(N)))
        end

        # Update proposal covariance
        _update_proposal_cov!(state)

        # Mutate particles via RWMH
        _smc_mutation!(state, phi_new, ll_fn, prior, n_mh_steps, rng)

        phi = phi_new
    end

    return state
end

# =============================================================================
# Adaptive Random-Walk Metropolis-Hastings
# =============================================================================

"""
    _mh_sample(spec, data, param_names, prior, θ0; kwargs...) → (draws, log_posterior, acceptance_rate)

Adaptive Random-Walk Metropolis-Hastings sampler for Bayesian DSGE estimation.

Adapts the proposal covariance every `adapt_interval` steps, targeting 23.4%
acceptance rate (Roberts & Rosenthal 2001).

# Arguments
- `spec::DSGESpec{T}` — model specification
- `data::Matrix{T}` — n_obs × T_obs data matrix
- `param_names::Vector{Symbol}` — parameters to estimate
- `prior::DSGEPrior{T}` — prior specification
- `θ0::Vector{T}` — initial parameter vector

# Keywords
- `n_draws::Int=5000` — total number of draws (including burnin)
- `burnin::Int=1000` — number of burnin draws (kept in output)
- `adapt_interval::Int=100` — adapt proposal covariance every N draws
- `observables::Vector{Symbol}` — observed variables
- `measurement_error` — measurement error SDs or `nothing`
- `solver::Symbol=:gensys` — DSGE solver method
- `solver_kwargs::NamedTuple` — additional solver kwargs
- `rng::AbstractRNG` — random number generator

# Returns
- `draws::Matrix{T}` — `n_draws × n_params` matrix of posterior draws
- `log_posterior::Vector{T}` — log posterior at each draw
- `acceptance_rate::T` — overall acceptance rate

# References
- Roberts, G. O. & Rosenthal, J. S. (2001). Optimal Scaling for Various
  Metropolis-Hastings Algorithms. *Statistical Science*, 16(4), 351-367.
"""
function _mh_sample(spec::DSGESpec{T}, data::AbstractMatrix,
                     param_names::Vector{Symbol}, prior::DSGEPrior{T},
                     theta0::AbstractVector{T};
                     n_draws::Int=5000, burnin::Int=1000,
                     adapt_interval::Int=100,
                     observables::Vector{Symbol}=spec.endog,
                     measurement_error=nothing,
                     solver::Symbol=:gensys,
                     solver_kwargs::NamedTuple=NamedTuple(),
                     rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    data = Matrix{T}(data)  # convert Adjoint/SubArray to concrete Matrix
    n_params = length(param_names)

    # Build likelihood function
    ll_fn = _build_likelihood_fn(spec, param_names, data, observables,
                                  measurement_error, solver, solver_kwargs)

    # Initialize
    theta_current = copy(theta0)
    ll_current = ll_fn(theta_current)
    lp_current = _log_prior(theta_current, prior)

    # If initial point has -Inf likelihood, try perturbing
    if !isfinite(ll_current) || !isfinite(lp_current)
        for attempt in 1:50
            theta_try = theta_current + T(0.01) * randn(rng, T, n_params)
            # Clip to bounds
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

    # Proposal covariance: start with identity, scale by c²
    c2 = T(2.38)^2 / T(n_params)
    proposal_cov = c2 * Matrix{T}(I, n_params, n_params)
    scale_factor = one(T)

    # Cholesky of proposal
    proposal_L = cholesky(Hermitian(proposal_cov)).L

    total_accepted = 0

    for draw in 1:n_draws
        # Propose
        z = randn(rng, T, n_params)
        theta_star = theta_current + scale_factor * proposal_L * z

        # Evaluate
        lp_star = _log_prior(theta_star, prior)

        if isfinite(lp_star)
            ll_star = ll_fn(theta_star)

            if isfinite(ll_star)
                # MH acceptance ratio (phi = 1 for full posterior)
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
            # Use recent draws to update proposal covariance
            recent_start = max(1, draw - 5 * adapt_interval)
            recent_draws = draws[recent_start:draw, :]

            if size(recent_draws, 1) > n_params + 1
                sample_cov = cov(recent_draws)
                # Ensure positive definiteness
                sample_cov = (sample_cov + sample_cov') / 2
                for i in 1:n_params
                    if sample_cov[i, i] < T(1e-10)
                        sample_cov[i, i] = T(1e-10)
                    end
                end

                proposal_cov = c2 * sample_cov
                proposal_L_try = try
                    cholesky(Hermitian(proposal_cov)).L
                catch
                    proposal_cov += T(1e-6) * I
                    cholesky(Hermitian(proposal_cov)).L
                end
                proposal_L = proposal_L_try
            end

            # Adapt scale factor to target 23.4% acceptance
            recent_acc = total_accepted / draw
            if recent_acc > T(0.30)
                scale_factor *= T(1.1)  # accepting too much → increase step size
            elseif recent_acc < T(0.15)
                scale_factor *= T(0.9)  # accepting too little → decrease step size
            end
        end
    end

    acceptance_rate = T(total_accepted) / T(n_draws)

    return draws, log_posterior, acceptance_rate
end
