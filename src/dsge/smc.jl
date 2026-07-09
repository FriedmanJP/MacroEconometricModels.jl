# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
    _effective_obs_offset(d, new_spec, sol, observables) -> Vector

Observation-equation offset for the likelihood. For pre-linearized (`linear=true`) models
with a non-zero constant, `spec.steady_state` is zero, so the offset must instead be the
effective steady state `(I - G1)⁻¹·C_sol` at the observed variables. Shared by the Kalman
and particle-filter paths so both apply the same offset (audit E-07 / #115). Returns `d`
unchanged for nonlinear models (the `linear` guard short-circuits before touching `sol.C_sol`).
"""
function _effective_obs_offset(d::AbstractVector{T}, new_spec::DSGESpec{T},
                               sol, observables) where {T<:AbstractFloat}
    if new_spec.linear && !all(iszero, sol.C_sol)
        eff_ss = (I - sol.G1) \ sol.C_sol
        obs_idx = [findfirst(==(obs), new_spec.endog) for obs in observables]
        return T[eff_ss[j] for j in obs_idx]
    end
    return d
end

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
                               solver_kwargs::NamedTuple;
                               failures::Threads.Atomic{Int}=Threads.Atomic{Int}(0),
                               evals::Threads.Atomic{Int}=Threads.Atomic{Int}(0)) where {T<:AbstractFloat}
    data = Matrix{T}(data)  # convert Adjoint/SubArray to concrete Matrix
    # Pre-build observation equation from spec (indices don't change across θ)
    # But steady state changes, so we rebuild each time

    function ll_fn(theta::Vector{T})
        Threads.atomic_add!(evals, 1)
        try
            # Update parameter values
            new_pv = copy(spec.param_values)
            for (i, pn) in enumerate(param_names)
                new_pv[pn] = theta[i]
            end

            # Create new spec with updated parameters
            new_spec = _respec(spec, new_pv)

            # Compute steady state
            new_spec = compute_steady_state(new_spec)

            # Solve model
            sol = solve(new_spec; method=solver, solver_kwargs...)
            if !is_determined(sol)
                Threads.atomic_add!(failures, 1)
                return T(-Inf)
            end

            # Guard: Kalman filter only valid for linear (1st-order) solutions
            if sol isa PerturbationSolution && sol.order >= 2
                Threads.atomic_add!(failures, 1)
                return T(-Inf)  # use :smc2 method for nonlinear models
            end

            # Build observation equation and state space
            Z, d, H = _build_observation_equation(new_spec, observables, measurement_error)

            # linear=true models need the effective-SS observation offset (shared helper,
            # so the Kalman and particle-filter paths agree).
            d = _effective_obs_offset(d, new_spec, sol, observables)

            ss = _build_state_space(sol, Z, d, H)

            # Evaluate Kalman log-likelihood
            ll = _kalman_loglikelihood(ss, data)

            if isfinite(ll)
                return ll
            else
                Threads.atomic_add!(failures, 1)
                return T(-Inf)
            end
        catch e
            # Only benign per-θ numeric failures become -Inf (counted); bugs propagate.
            if _benign_solve_error(e)
                Threads.atomic_add!(failures, 1)
                return T(-Inf)
            end
            rethrow(e)
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
    _adaptive_tempering(log_liks, log_weights, phi_old, ess_target, N)

Find `phi_new` via bisection such that the ESS of the **updated cumulative**
importance weights `W_{n−1,j} · exp((phi_new − phi_old) · ll_j)` equals
`ess_target · N`.

The tempering reweight multiplies into the incoming (possibly non-uniform)
cumulative weights `log_weights`, so the ESS that governs the step size and the
resample decision must be evaluated on `log_weights .+ Δφ .* log_liks`, not on
`Δφ .* log_liks` alone (E-10 / #133). In the log domain, for `lw = log_weights
.+ Δφ .* log_liks`,

    log ESS(Δφ) = 2·logsumexp(lw) − logsumexp(2·lw).

Under uniform incoming weights this reduces to the old incremental-only form.
If ESS at `phi = 1.0` exceeds the target, returns `1.0` (final step). Maximum
50 bisection iterations.

# Arguments
- `log_liks::Vector{T}` — per-particle log-likelihoods
- `log_weights::Vector{T}` — current cumulative (unnormalized) log-weights carried
  into this stage, **before** folding in the tempering reweight
- `phi_old::T` — current tempering parameter
- `ess_target::Real` — target ESS as fraction of N (e.g., 0.5)
- `N::Int` — number of particles

# Returns
`phi_new::T` — next tempering parameter in `(phi_old, 1.0]`.
"""
function _adaptive_tempering(log_liks::AbstractVector{T}, log_weights::AbstractVector{T},
                              phi_old::T, ess_target::Real, N::Int) where {T<:AbstractFloat}
    target_ess = T(ess_target) * N

    # Helper: ESS of the updated cumulative weights at a given phi.
    function _compute_ess(phi)
        delta_phi = phi - phi_old
        lw = log_weights .+ delta_phi .* log_liks
        a = _logsumexp(lw)
        isfinite(a) || return zero(T)            # all weights collapsed
        b = _logsumexp(T(2) .* lw)
        return exp(T(2) * a - b)
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

"""
    _check_tempering_progress(stage, max_stages, phi, phi_new, min_dphi)

Guard the adaptive-tempering `while` loops against a degenerate likelihood that never
advances φ to 1 (E-21 / #145). Throws an informative `ErrorException` when either the
stage count exceeds `max_stages` or the chosen step `phi_new - phi < min_dphi` while
`phi_new < 1` (a legitimate final jump returns exactly `phi_new == 1`, never flagged).
Herbst & Schorfheide (2016) use a fixed ~200–500-stage schedule; `max_stages` bounds
the adaptive analogue.
"""
function _check_tempering_progress(stage::Int, max_stages::Int, phi::T,
                                    phi_new::T, min_dphi::T) where {T<:AbstractFloat}
    if stage > max_stages
        error("SMC tempering did not reach φ=1 within max_stages=$(max_stages) stages " *
              "(stalled at φ=$(phi)). The likelihood likely degenerated (mis-specified " *
              "model or bad data slice); raise max_stages only if the schedule is " *
              "legitimately long.")
    end
    if phi_new < one(T) && (phi_new - phi) < min_dphi
        error("SMC tempering stalled: adaptive step Δφ=$(phi_new - phi) < min_dphi=" *
              "$(min_dphi) at φ=$(phi) (stage $(stage)). The likelihood degenerated — " *
              "the model/data are likely mis-specified. Aborting rather than spinning.")
    end
    return nothing
end

"""
    _chunk_ranges(N, k) -> Vector{UnitRange{Int}}

Partition `1:N` into `k` contiguous, near-equal chunks (trailing ranges are empty
when `N < k`). Used to give each parallel task a **stable, dedicated** workspace by
chunk index rather than `threadid()` — the latter is unsafe under Julia's dynamic
scheduler, where a migrated task's `threadid()` can collide with another live task's
and alias the same particle-filter workspace (E-06 / #134; generalized in #146).
"""
function _chunk_ranges(N::Int, k::Int)
    ranges = Vector{UnitRange{Int}}(undef, k)
    base, rem = divrem(N, k)
    start = 1
    @inbounds for c in 1:k
        len = base + (c <= rem ? 1 : 0)
        ranges[c] = start:(start + len - 1)
        start += len
    end
    return ranges
end

"""
    _terminal_resample!(state, N, rng) -> Bool

Systematic resample of the SMC particles at termination (E-09 / #132).

When the final tempering stage reaches φ=1 without triggering a resample, the
stored θ-particles carry **non-uniform** terminal weights. Downstream consumers
(`posterior_summary` order-statistic quantiles, and the `irf`/`fevd`/`simulate`
draw selectors that pick particles uniformly) treat `theta_draws` as equally
weighted, biasing medians, credible intervals and posterior bands. Resampling
once on the normalized final weights and resetting weights to uniform makes the
stored draws equal-weighted samples from the φ=1 posterior.

Returns `true` if a resample was performed, `false` when the weights are already
(numerically) uniform — e.g. the final stage did resample — so there is no
redundant double resample. Resamples `theta_particles`, `log_likelihoods` and
`log_priors` (the fields `_smc_state_to_bayesian_dsge` consumes).
"""
function _terminal_resample!(state::SMCState{T}, N::Int,
                              rng::AbstractRNG) where {T<:AbstractFloat}
    weights = Vector{T}(undef, N)
    _normalize_log_weights!(weights, state.log_weights)
    ess = one(T) / sum(abs2, weights)
    ess >= N * (one(T) - T(1e-8)) && return false   # already uniform → no-op

    ancestors = collect(1:N)
    cumweights = zeros(T, N)
    _systematic_resample!(ancestors, weights, cumweights, N, rng)

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
    fill!(state.log_weights, -log(T(N)))
    return true
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

    # Pre-generate per-particle seeds BEFORE entering @threads
    # (MersenneTwister is NOT thread-safe, so all rng calls must be sequential)
    particle_seeds = [hash((j, rand(rng, UInt64))) for j in 1:N]

    Threads.@threads for j in 1:N
        # Per-thread RNG seeded from pre-generated seed
        thread_rng = Random.MersenneTwister(particle_seeds[j])

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
                      max_stages::Int=500, min_dphi::Real=1e-10,
                      rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    data = Matrix{T}(data)  # convert Adjoint/SubArray to concrete Matrix
    n_params = length(param_names)
    N = n_smc

    # Build likelihood function (tracking failed/total likelihood evaluations)
    lik_failures = Threads.Atomic{Int}(0)
    lik_evals = Threads.Atomic{Int}(0)
    ll_fn = _build_likelihood_fn(spec, param_names, data, observables,
                                  measurement_error, solver, solver_kwargs;
                                  failures=lik_failures, evals=lik_evals)

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
    stage = 0
    min_step = T(min_dphi)

    while phi < one(T)
        stage += 1
        # Find next tempering parameter
        # Only use particles with finite log-likelihoods for tempering
        valid_lls = copy(state.log_likelihoods)
        for j in 1:N
            if !isfinite(valid_lls[j])
                valid_lls[j] = T(-1e10)  # large negative but finite
            end
        end

        phi_new = _adaptive_tempering(valid_lls, state.log_weights, phi, ess_target, N)
        # Abort a degenerate-likelihood stall before paying for the expensive mutation (#145).
        _check_tempering_progress(stage, max_stages, phi, phi_new, min_step)
        delta_phi = phi_new - phi

        # Compute incremental log weights
        inc_log_w = delta_phi .* state.log_likelihoods
        # Handle -Inf likelihoods
        for j in 1:N
            if !isfinite(inc_log_w[j])
                inc_log_w[j] = T(-1e10)
            end
        end

        # Update log marginal likelihood with the ratio increment (E-08 / #131):
        #   ΔlogML = logsumexp(log_weights + inc) − logsumexp(log_weights)
        # using the (unnormalized) cumulative log-weights carried INTO this stage —
        # i.e. BEFORE folding inc_log_w into state.log_weights below. This reduces to
        # logsumexp(inc) − log N only when the incoming weights are uniform (right after
        # a resample); the old uniform form biased logML after any non-resampled stage.
        state.log_marginal_likelihood +=
            _logsumexp(state.log_weights .+ inc_log_w) - _logsumexp(state.log_weights)

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

    # Terminal resample so the stored draws are equal-weighted (E-09 / #132).
    _terminal_resample!(state, N, rng)

    state.n_lik_failures = lik_failures[]
    state.n_lik_evals = lik_evals[]
    return state
end

# =============================================================================
# Adaptive Random-Walk Metropolis-Hastings
# =============================================================================

"""
    _mh_sample(spec, data, param_names, prior, θ0; kwargs...) → (draws, log_posterior, acceptance_rate, diagnostics)

Adaptive Random-Walk Metropolis-Hastings sampler for Bayesian DSGE estimation.

Adapts the proposal covariance and scale every `adapt_interval` steps **during
burn-in only**, targeting 23.4% acceptance (Roberts & Rosenthal 2001). The proposal
is frozen at `draw == burnin`, so — with the burn-in discarded by the caller — the
retained chain runs a fixed proposal and targets the exact posterior (adaptive-MCMC
validity, Roberts & Rosenthal 2007). The scale is tuned from a **trailing-window**
acceptance rate (over the last `adapt_interval` draws), not the stale cumulative rate.

# Arguments
- `spec::DSGESpec{T}` — model specification
- `data::Matrix{T}` — n_obs × T_obs data matrix
- `param_names::Vector{Symbol}` — parameters to estimate
- `prior::DSGEPrior{T}` — prior specification
- `θ0::Vector{T}` — initial parameter vector

# Keywords
- `n_draws::Int=5000` — total number of draws (including burnin)
- `burnin::Int=1000` — burn-in length. `_mh_sample` returns the FULL chain; the caller
  (`estimate_dsge_bayes`) discards the first `burnin` draws unless `keep_burnin=true`.
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
- `diagnostics::NamedTuple` — the frozen end-of-burn-in proposal (`proposal_L_at_burnin`,
  `scale_at_burnin`), the final proposal (`proposal_L`, `scale_factor`), and the
  per-interval trailing-window vs cumulative acceptance signals (`window_acc_history`,
  `cum_acc_history`)

# References
- Roberts, G. O. & Rosenthal, J. S. (2001). Optimal Scaling for Various
  Metropolis-Hastings Algorithms. *Statistical Science*, 16(4), 351-367.
- Roberts, G. O. & Rosenthal, J. S. (2007). Coupling and Ergodicity of Adaptive
  Markov Chain Monte Carlo Algorithms. *Journal of Applied Probability*, 44(2), 458-475.
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

    # Build likelihood function (tracking failed/total likelihood evaluations)
    lik_failures = Threads.Atomic{Int}(0)
    lik_evals = Threads.Atomic{Int}(0)
    ll_fn = _build_likelihood_fn(spec, param_names, data, observables,
                                  measurement_error, solver, solver_kwargs;
                                  failures=lik_failures, evals=lik_evals)

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
    window_accepted = 0
    # Adaptation is confined to burn-in (Roberts & Rosenthal 2007): the proposal is
    # frozen at draw == burnin so the retained chain targets the exact posterior.
    proposal_L_at_burnin = copy(proposal_L)
    scale_at_burnin = scale_factor
    window_acc_history = T[]
    cum_acc_history = T[]

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
                    window_accepted += 1
                end
            end
        end

        # Store
        draws[draw, :] = theta_current
        log_posterior[draw] = ll_current + lp_current

        # Adapt proposal covariance and scale — ONLY during burn-in (frozen after).
        if draw <= burnin && draw % adapt_interval == 0
            # Trailing-window acceptance signal (not the stale cumulative rate); the
            # covariance adapts to recent draws, so the scale must too.
            window_acc = T(window_accepted) / T(adapt_interval)
            push!(window_acc_history, window_acc)
            push!(cum_acc_history, T(total_accepted) / T(draw))
            window_accepted = 0

            if draw >= 2 * adapt_interval
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

                # Adapt scale factor to target 23.4% acceptance (trailing window)
                if window_acc > T(0.30)
                    scale_factor *= T(1.1)  # accepting too much → increase step size
                elseif window_acc < T(0.15)
                    scale_factor *= T(0.9)  # accepting too little → decrease step size
                end
            end
        end

        # Freeze the proposal at the end of burn-in for the retained chain.
        if draw == burnin
            proposal_L_at_burnin = copy(proposal_L)
            scale_at_burnin = scale_factor
        end
    end

    acceptance_rate = T(total_accepted) / T(n_draws)

    diagnostics = (proposal_L = proposal_L, scale_factor = scale_factor,
                   proposal_L_at_burnin = proposal_L_at_burnin,
                   scale_at_burnin = scale_at_burnin,
                   window_acc_history = window_acc_history,
                   cum_acc_history = cum_acc_history,
                   n_failed = lik_failures[], n_evals = lik_evals[])

    return draws, log_posterior, acceptance_rate, diagnostics
end

# =============================================================================
# SMC² with Conditional SMC mutation (Chopin, Jacob & Papaspiliopoulos 2013)
# =============================================================================

"""
    _adapt_n_particles(N_x::Int, est_var::Real, threshold::Real) → Int

Adapt the number of inner state particles for SMC². If the **PF estimator
variance** at fixed θ (see [`_pf_estimator_variance`](@ref)) exceeds `threshold`,
double `N_x`; otherwise return `N_x` unchanged. Chopin et al. (2013) §3.4 target a
log-likelihood-estimator variance of roughly 1–3.
"""
function _adapt_n_particles(N_x::Int, est_var::Real, threshold::Real)
    return est_var > threshold ? 2 * N_x : N_x
end

"""
    _pf_estimator_variance(spec, param_names, theta_particles, observables,
                           measurement_error, solver, solver_kwargs, pool, data,
                           T_obs, rng; n_probe, n_rep) -> T

Estimate the variance of the particle-filter log-likelihood estimator **at fixed θ**
(Chopin et al. 2013 §3.4) — the quantity that should trigger `N_x` adaptation.

Probes `n_probe` θ-particles (evenly spaced through the population) with `n_rep`
fresh, independent bootstrap PFs each, and returns the mean over probes of the
per-θ variance of `log p̂(y|θ)`. This measures the **estimator noise** at a fixed
parameter, NOT `var(ll across θ)` — the spread of the likelihood across the
θ-population, which is large on any diffuse posterior and would trigger spurious
doubling (E-11 / #135). Returns `0` when fewer than two finite replicates exist.
"""
function _pf_estimator_variance(spec::DSGESpec{T}, param_names::Vector{Symbol},
        theta_particles::AbstractMatrix{T}, observables::Vector{Symbol},
        measurement_error, solver::Symbol, solver_kwargs::NamedTuple,
        pool::Vector{PFWorkspace{T}}, data::Matrix{T}, T_obs::Int,
        rng::AbstractRNG; n_probe::Int=8, n_rep::Int=2) where {T<:AbstractFloat}
    N = size(theta_particles, 2)
    n_probe = min(n_probe, N)
    (n_probe < 1 || n_rep < 2) && return zero(T)
    idx = unique(round.(Int, range(1, N; length=n_probe)))
    ws = pool[1]
    per_theta = T[]
    for i in idx
        theta_i = Vector{T}(theta_particles[:, i])
        reps = T[]
        for r in 1:n_rep
            rr = Random.MersenneTwister(hash((:nx_probe, i, r, rand(rng, UInt64))))
            ll = _solve_and_run_pf(spec, param_names, theta_i, observables,
                measurement_error, solver, solver_kwargs, ws, data, T_obs, rr)
            isfinite(ll) && push!(reps, ll)
        end
        length(reps) >= 2 && push!(per_theta, var(reps))
    end
    return isempty(per_theta) ? zero(T) : mean(per_theta)
end

"""
    _exchange_step!(state, spec, param_names, observables, measurement_error,
                    solver, solver_kwargs, pool, data, T_obs, rng) -> Nothing

Chopin et al. (2013) §3.4 **exchange step**: after `N_x` changes, re-run a fresh
unconditional bootstrap PF for EVERY θ-particle at the new `N_x` (the `pool`
workspaces must already be resized) and replace the stored `log_likelihoods`, so
the outer θ-weights are not a mix of estimates computed at two different `N_x`.
"""
function _exchange_step!(state::SMCState{T}, spec::DSGESpec{T},
        param_names::Vector{Symbol}, observables::Vector{Symbol}, measurement_error,
        solver::Symbol, solver_kwargs::NamedTuple, pool::Vector{PFWorkspace{T}},
        data::Matrix{T}, T_obs::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    N = size(state.theta_particles, 2)
    ranges = _chunk_ranges(N, length(pool))
    seeds = [hash((:exchange, j, rand(rng, UInt64))) for j in 1:N]
    Threads.@threads for c in eachindex(ranges)
        ws = pool[c]
        for j in ranges[c]
            rr = Random.MersenneTwister(seeds[j])
            state.log_likelihoods[j] = _solve_and_run_pf(spec, param_names,
                Vector{T}(state.theta_particles[:, j]), observables, measurement_error,
                solver, solver_kwargs, ws, data, T_obs, rr)
        end
    end
    return nothing
end

"""
    _solve_and_run_pf(spec, param_names, theta, observables, measurement_error,
                       solver, solver_kwargs, ws, data, T_obs, rng;
                       use_csmc=false, store_trajectory=false, return_solution=false)

Shared helper: update parameters → solve → dispatch state space → run PF/CSMC.

Creates a new `DSGESpec` with updated parameters, computes steady state, solves
the model, builds the correct state space type based on solution dispatch, and
runs either `_bootstrap_particle_filter!` or `_conditional_smc!`.

Returns the log-likelihood `T` (or `T(-Inf)` on failure). When
`return_solution=true`, returns `(ll, solution)` tuple instead.
"""
function _solve_and_run_pf(spec::DSGESpec{T}, param_names::Vector{Symbol},
                            theta::AbstractVector{T},
                            observables::Vector{Symbol},
                            measurement_error,
                            solver::Symbol,
                            solver_kwargs::NamedTuple,
                            ws::PFWorkspace{T},
                            data::Matrix{T}, T_obs::Int,
                            rng::AbstractRNG;
                            use_csmc::Bool=false,
                            store_trajectory::Bool=false,
                            return_solution::Bool=false,
                            failures::Threads.Atomic{Int}=Threads.Atomic{Int}(0),
                            evals::Threads.Atomic{Int}=Threads.Atomic{Int}(0)) where {T<:AbstractFloat}
    fail = return_solution ? (T(-Inf), nothing) : T(-Inf)
    Threads.atomic_add!(evals, 1)
    try
        # Update parameter values
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end

        # Create new spec with updated parameters
        new_spec = _respec(spec, new_pv)

        # Compute steady state and solve
        new_spec = compute_steady_state(new_spec)
        sol = solve(new_spec; method=solver, solver_kwargs...)
        if !is_determined(sol)
            Threads.atomic_add!(failures, 1)
            return fail
        end

        # Build observation equation; apply the same effective-SS offset as the Kalman path
        # so linear=true constant models are not silently mis-offset in the PF (E-07 / #115).
        Z, d_vec, H_mat = _build_observation_equation(new_spec, observables, measurement_error)
        d_vec = _effective_obs_offset(d_vec, new_spec, sol, observables)

        # Dispatch: PerturbationSolution → NonlinearStateSpace,
        #           ProjectionSolution → ProjectionStateSpace,
        #           else → DSGEStateSpace
        if sol isa PerturbationSolution
            ss = _build_nonlinear_state_space(sol, Z, d_vec, H_mat)
        elseif sol isa ProjectionSolution
            ss = _build_projection_state_space(sol, Z, d_vec, H_mat)
        else
            ss = _build_state_space(sol, Z, d_vec, H_mat)
        end

        # Run particle filter or conditional SMC
        if use_csmc
            ll = _conditional_smc!(ws, ss, data, T_obs; rng=rng)
        else
            ll = _bootstrap_particle_filter!(ws, ss, data, T_obs;
                                              store_trajectory=store_trajectory, rng=rng)
        end

        if !isfinite(ll)
            Threads.atomic_add!(failures, 1)
            ll = T(-Inf)
        end
        return return_solution ? (ll, sol) : ll
    catch e
        if _benign_solve_error(e)
            Threads.atomic_add!(failures, 1)
            return fail
        end
        rethrow(e)
    end
end

"""
    _build_pf_likelihood_fn(spec, param_names, data, observables, measurement_error,
                             solver, solver_kwargs, n_particles)

Build a closure `(θ_vec, ws, rng) → log p(Y|θ)` that evaluates the likelihood
via bootstrap particle filter instead of the Kalman filter.

The closure updates spec parameters, solves the model, builds the correct state
space type (dispatching on `PerturbationSolution`, `ProjectionSolution`, or
`DSGESolution`), and runs `_bootstrap_particle_filter!`. Returns `-Inf` on any
failure.

# Arguments
- `spec::DSGESpec{T}` — model specification (template)
- `param_names::Vector{Symbol}` — which parameters θ maps to
- `data::Matrix{T}` — n_obs × T_obs data matrix
- `observables::Vector{Symbol}` — observed endogenous variables
- `measurement_error` — measurement error SDs or `nothing`
- `solver::Symbol` — solver method
- `solver_kwargs::NamedTuple` — additional keyword arguments for `solve`
- `n_particles::Int` — number of particles for the internal PF

# Returns
Closure `(θ::Vector{T}, ws::PFWorkspace{T}, rng::AbstractRNG) → T`.
"""
function _build_pf_likelihood_fn(spec::DSGESpec{T}, param_names::Vector{Symbol},
                                   data::AbstractMatrix, observables::Vector{Symbol},
                                   measurement_error, solver::Symbol,
                                   solver_kwargs::NamedTuple,
                                   n_particles::Int;
                                   failures::Threads.Atomic{Int}=Threads.Atomic{Int}(0),
                                   evals::Threads.Atomic{Int}=Threads.Atomic{Int}(0)) where {T<:AbstractFloat}
    data = Matrix{T}(data)
    T_obs = size(data, 2)

    function pf_ll_fn(theta::Vector{T}, ws::PFWorkspace{T}, rng::AbstractRNG)
        return _solve_and_run_pf(spec, param_names, theta, observables,
                                  measurement_error, solver, solver_kwargs,
                                  ws, data, T_obs, rng;
                                  store_trajectory=true,
                                  failures=failures, evals=evals)
    end

    return pf_ll_fn
end

"""
    _smc2_sample(spec, data, param_names, prior, θ0; kwargs...) → SMCState

SMC² algorithm (Chopin, Jacob & Papaspiliopoulos 2013) that nests a particle
filter inside the outer SMC sampler. The mutation step uses Conditional SMC
(CSMC) instead of the Kalman filter for likelihood evaluation.

Algorithm:
1. Initialize N_smc θ-particles from prior
2. Evaluate log-likelihoods via bootstrap PF for each θ-particle
3. Adaptive tempering loop:
   a. Find `phi_new` via `_adaptive_tempering`
   b. Compute incremental weights, update log marginal likelihood
   c. Resample if ESS drops
   d. Update proposal covariance
   e. Mutation: for each θ-particle, propose θ*, run CSMC, MH accept/reject
   f. Adapt N_x if log-likelihood variance is high
   g. Stop when phi = 1

# Arguments
- `spec::DSGESpec{T}` — model specification
- `data::Matrix{T}` — n_obs × T_obs data matrix
- `param_names::Vector{Symbol}` — parameters to estimate
- `prior::DSGEPrior{T}` — prior specification
- `θ0::Vector{T}` — initial parameter guess

# Keywords
- `n_smc::Int=200` — number of outer SMC particles
- `n_particles::Int=100` — number of inner PF particles
- `n_mh_steps::Int=1` — MH steps per mutation stage
- `ess_target::Real=0.5` — target ESS fraction
- `observables::Vector{Symbol}` — observed variables
- `measurement_error` — measurement error SDs or `nothing`
- `solver::Symbol=:gensys` — DSGE solver method
- `solver_kwargs::NamedTuple` — additional solver kwargs
- `delayed_acceptance::Bool=false` — enable two-stage delayed acceptance MH
  (Christen & Fox 2005). Pre-screens proposals with a cheap bootstrap PF
  (`n_screen` particles) before running the expensive CSMC, preserving
  detailed balance while avoiding ~60-80% of wasted CSMC evaluations.
- `n_screen::Int=200` — number of particles for the screening PF in
  delayed acceptance (only used when `delayed_acceptance=true`)
- `rng::AbstractRNG` — random number generator

# Returns
`SMCState{T}` with posterior particles, weights, tempering schedule, and
log marginal likelihood estimate.

# References
- Chopin, N., Jacob, P. E. & Papaspiliopoulos, O. (2013). SMC²: An efficient
  algorithm for sequential analysis of state space models.
  *Journal of the Royal Statistical Society: Series B*, 75(3), 397-426.
- Christen, J. A. & Fox, C. (2005). Markov chain Monte Carlo using an
  approximation. *Journal of Computational and Graphical Statistics*,
  14(4), 795-810.
"""
function _smc2_sample(spec::DSGESpec{T}, data::AbstractMatrix,
                       param_names::Vector{Symbol}, prior::DSGEPrior{T},
                       theta0::AbstractVector{T};
                       n_smc::Int=200, n_particles::Int=100,
                       n_mh_steps::Int=1, ess_target::Real=0.5,
                       observables::Vector{Symbol}=spec.endog,
                       measurement_error=nothing,
                       solver::Symbol=:gensys,
                       solver_kwargs::NamedTuple=NamedTuple(),
                       delayed_acceptance::Bool=false,
                       n_screen::Int=200,
                       max_stages::Int=500, min_dphi::Real=1e-10,
                       rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    data = Matrix{T}(data)
    n_params = length(param_names)
    N = n_smc
    N_x = n_particles
    T_obs = size(data, 2)
    n_obs = size(data, 1)

    # Determine state space dimensions from spec
    n_states = spec.n_endog
    n_shocks = spec.n_exog

    # Detect perturbation order for pruning buffer allocation
    pf_nv = 0
    pf_nx = 0
    pf_order = 1
    pf_proj_nx = 0
    pf_proj_n_basis = 0
    pf_proj_max_degree = 0
    pf_proj_n_vars = 0
    if solver == :perturbation
        # Extract order from solver_kwargs (default 2)
        pf_order = get(solver_kwargs, :order, 2)
        # Trial solve to get nx, nv dimensions
        try
            trial_spec = compute_steady_state(spec)
            trial_sol = solve(trial_spec; method=solver, solver_kwargs...)
            if trial_sol isa PerturbationSolution
                pf_nx = length(trial_sol.state_indices)
                pf_nv = pf_nx + n_shocks
                pf_order = trial_sol.order
            end
        catch
            # If trial solve fails, estimate nx from spec
            pf_nx = n_states
            pf_nv = pf_nx + n_shocks
        end
    elseif solver in (:projection, :pfi)
        try
            trial_spec = compute_steady_state(spec)
            trial_sol = solve(trial_spec; method=solver, solver_kwargs...)
            if trial_sol isa ProjectionSolution
                pf_proj_nx = length(trial_sol.state_indices)
                pf_proj_n_basis = trial_sol.n_basis
                pf_proj_max_degree = maximum(trial_sol.multi_indices)
                pf_proj_n_vars = length(trial_sol.steady_state)
            end
        catch
            # Fallback: will be allocated at first likelihood evaluation
            pf_proj_nx = n_states
            pf_proj_n_basis = 0
            pf_proj_max_degree = 5
            pf_proj_n_vars = n_states
        end
    end

    # Build PF likelihood function (tracking failed/total authoritative likelihood evals)
    lik_failures = Threads.Atomic{Int}(0)
    lik_evals = Threads.Atomic{Int}(0)
    pf_ll_fn = _build_pf_likelihood_fn(spec, param_names, data, observables,
                                        measurement_error, solver, solver_kwargs, N_x;
                                        failures=lik_failures, evals=lik_evals)

    # Create workspace pool (one per thread for parallelism)
    n_pool = max(Threads.nthreads(), 1)
    pool = [_allocate_pf_workspace(T, n_states, n_obs, n_shocks, N_x;
                                    nv=pf_nv, nx=pf_nx, order=pf_order, T_obs=T_obs,
                                    proj_nx=pf_proj_nx, proj_n_basis=pf_proj_n_basis,
                                    proj_max_degree=pf_proj_max_degree,
                                    proj_n_vars=pf_proj_n_vars)
            for _ in 1:n_pool]

    # Per-particle solution cache for warm-starting (projection/PFI only)
    solutions = Vector{Any}(undef, N)
    fill!(solutions, nothing)

    # Delayed acceptance: screening workspace pool + cheap log-likelihoods
    screen_pool = if delayed_acceptance
        [_allocate_pf_workspace(T, n_states, n_obs, n_shocks, n_screen;
                                nv=pf_nv, nx=pf_nx, order=pf_order, T_obs=T_obs,
                                proj_nx=pf_proj_nx, proj_n_basis=pf_proj_n_basis,
                                proj_max_degree=pf_proj_max_degree,
                                proj_n_vars=pf_proj_n_vars)
         for _ in 1:n_pool]
    else
        PFWorkspace{T}[]
    end
    ll_cheap = fill(T(-Inf), N)

    # Initialize θ-particles from prior
    theta_particles = zeros(T, n_params, N)
    log_priors = zeros(T, N)

    for j in 1:N
        for i in 1:n_params
            for attempt in 1:100
                draw = T(rand(rng, prior.distributions[i]))
                if draw >= prior.lower[i] && draw <= prior.upper[i]
                    theta_particles[i, j] = draw
                    break
                end
                if attempt == 100
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

    # Evaluate log-likelihoods via PF (and cache solutions for warm-starting)
    log_likelihoods = fill(T(-Inf), N)
    for j in 1:N
        ws_idx = mod1(j, n_pool)
        thread_rng = Random.MersenneTwister(hash((j, rand(rng, UInt64))))
        if solver in (:projection, :pfi)
            ll_j, sol_j = _solve_and_run_pf(spec, param_names,
                                              theta_particles[:, j], observables,
                                              measurement_error, solver, solver_kwargs,
                                              pool[ws_idx], data, T_obs, thread_rng;
                                              store_trajectory=true, return_solution=true,
                                              failures=lik_failures, evals=lik_evals)
            log_likelihoods[j] = ll_j
            if sol_j isa ProjectionSolution
                solutions[j] = sol_j
            end
        else
            log_likelihoods[j] = pf_ll_fn(theta_particles[:, j], pool[ws_idx], thread_rng)
        end
    end

    # Compute cheap screening likelihoods for delayed acceptance
    if delayed_acceptance
        for j in 1:N
            ws_idx = mod1(j, n_pool)
            screen_rng = Random.MersenneTwister(hash((j, UInt64(0xDA), rand(rng, UInt64))))
            ll_cheap[j] = _solve_and_run_pf(spec, param_names,
                                              theta_particles[:, j], observables,
                                              measurement_error, solver, solver_kwargs,
                                              screen_pool[ws_idx], data, T_obs, screen_rng;
                                              store_trajectory=false)
        end
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
        pool,                 # pf_workspace_pool
        Matrix{T}(I, n_params, n_params)  # proposal_cov
    )

    # Resampling buffers
    weights_normalized = fill(one(T) / N, N)
    ancestors = collect(1:N)
    cumweights = zeros(T, N)

    # Adaptive tempering loop
    phi = zero(T)
    stage = 0
    min_step = T(min_dphi)

    while phi < one(T)
        stage += 1
        # Find next tempering parameter
        valid_lls = copy(state.log_likelihoods)
        for j in 1:N
            if !isfinite(valid_lls[j])
                valid_lls[j] = T(-1e10)
            end
        end

        phi_new = _adaptive_tempering(valid_lls, state.log_weights, phi, ess_target, N)
        # Abort a degenerate-likelihood stall before paying for the expensive PF mutation (#145).
        _check_tempering_progress(stage, max_stages, phi, phi_new, min_step)
        delta_phi = phi_new - phi

        # Compute incremental log weights
        inc_log_w = delta_phi .* state.log_likelihoods
        for j in 1:N
            if !isfinite(inc_log_w[j])
                inc_log_w[j] = T(-1e10)
            end
        end

        # Update log marginal likelihood with the ratio increment (E-08 / #131):
        #   ΔlogML = logsumexp(log_weights + inc) − logsumexp(log_weights)
        # using the (unnormalized) cumulative log-weights carried INTO this stage —
        # BEFORE folding inc_log_w into state.log_weights below. Reduces to the old
        # logsumexp(inc) − log N only under uniform incoming weights (post-resample).
        state.log_marginal_likelihood +=
            _logsumexp(state.log_weights .+ inc_log_w) - _logsumexp(state.log_weights)

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

            theta_new = similar(state.theta_particles)
            ll_new = similar(state.log_likelihoods)
            lp_new = similar(state.log_priors)
            sol_new = Vector{Any}(undef, N)
            ll_cheap_new = similar(ll_cheap)
            @inbounds for j in 1:N
                a = ancestors[j]
                theta_new[:, j] = state.theta_particles[:, a]
                ll_new[j] = state.log_likelihoods[a]
                lp_new[j] = state.log_priors[a]
                sol_new[j] = solutions[a]
                ll_cheap_new[j] = ll_cheap[a]
            end
            state.theta_particles .= theta_new
            state.log_likelihoods .= ll_new
            state.log_priors .= lp_new
            solutions .= sol_new
            ll_cheap .= ll_cheap_new

            fill!(state.log_weights, -log(T(N)))
        end

        # Update proposal covariance
        _update_proposal_cov!(state)

        # --- SMC² Mutation: CSMC-based MH steps ---
        L = try
            cholesky(Hermitian(state.proposal_cov)).L
        catch
            cov_jittered = state.proposal_cov + T(1e-6) * I
            cholesky(Hermitian(cov_jittered)).L
        end

        total_accepted = Threads.Atomic{Int}(0)
        total_proposed = Threads.Atomic{Int}(0)

        # Pre-generate per-particle seeds BEFORE entering @threads
        # (MersenneTwister is NOT thread-safe)
        particle_seeds_mut = [hash((j, phi_new, rand(rng, UInt64))) for j in 1:N]

        # PMMH mutation (E-06 / #134). Each θ-particle's move is a particle-marginal
        # Metropolis-Hastings step: a FRESH unconditional bootstrap PF gives an unbiased
        # likelihood estimate for θ*, compared against the stored incumbent PF estimate for
        # θ_j (kept noisy on accept — Andrieu-Doucet-Holenstein 2010). The old conditional-SMC
        # (CSMC) path is dropped from acceptance: CSMC estimates are conditional on a reference
        # trajectory and biased for parameter inference, and the pooled reference trajectory was
        # additionally contaminated across θ-particles via threadid()-indexed workspaces.
        #
        # Parallelism is chunk-based: partition the θ-particles into n_pool contiguous chunks,
        # one PF workspace per chunk, so no workspace is shared across concurrently-running tasks.
        chunk_ranges = _chunk_ranges(N, n_pool)

        Threads.@threads for c in eachindex(chunk_ranges)
            ws = pool[c]
            ws_screen = delayed_acceptance ? screen_pool[c] : ws
            for j in chunk_ranges[c]
                thread_rng = Random.MersenneTwister(particle_seeds_mut[j])

                theta_j = state.theta_particles[:, j]
                ll_j = state.log_likelihoods[j]   # incumbent unbiased PF estimate (PMMH)
                lp_j = state.log_priors[j]

                for _ in 1:n_mh_steps
                    # Propose: theta_star = theta_j + L * z (symmetric RW ⇒ q-terms cancel)
                    z = randn(thread_rng, T, n_params)
                    theta_star = theta_j + L * z

                    # Evaluate prior
                    lp_star = _log_prior(theta_star, prior)
                    if lp_star == T(-Inf)
                        Threads.atomic_add!(total_proposed, 1)
                        continue
                    end

                    # Build solver_kwargs with warm-starting from cached solution
                    mh_solver_kwargs = if solver in (:projection, :pfi) && solutions[j] isa ProjectionSolution
                        merge(solver_kwargs, (initial_coeffs=solutions[j].coefficients,))
                    else
                        solver_kwargs
                    end

                    if delayed_acceptance
                        # ── Two-stage delayed acceptance (Christen & Fox 2005) on top of PMMH ──
                        # Stage 1: cheap bootstrap-PF screen.
                        ll_cheap_star = _solve_and_run_pf(
                            spec, param_names, theta_star, observables, measurement_error,
                            solver, mh_solver_kwargs, ws_screen, data, T_obs, thread_rng)

                        if ll_cheap_star == T(-Inf)
                            Threads.atomic_add!(total_proposed, 1)
                            continue
                        end

                        # Stage 1 acceptance ratio (using cheap likelihoods)
                        log_alpha_1 = (phi_new * ll_cheap_star + lp_star) -
                                      (phi_new * ll_cheap[j] + lp_j)

                        if log(rand(thread_rng, T)) >= log_alpha_1
                            # Stage 1 rejects — skip the expensive PF
                            Threads.atomic_add!(total_proposed, 1)
                            continue
                        end

                        # Stage 2: fresh UNCONDITIONAL bootstrap PF (unbiased ⇒ PMMH-exact).
                        ll_star, sol_star = _solve_and_run_pf(
                            spec, param_names, theta_star, observables, measurement_error,
                            solver, mh_solver_kwargs, ws, data, T_obs, thread_rng;
                            use_csmc=false, return_solution=true,
                            failures=lik_failures, evals=lik_evals)

                        if ll_star == T(-Inf)
                            Threads.atomic_add!(total_proposed, 1)
                            continue
                        end

                        # Stage 2 acceptance ratio (delayed-acceptance correction term)
                        log_alpha_2 = phi_new * (ll_star - ll_cheap_star) -
                                      phi_new * (ll_j - ll_cheap[j])

                        if log(rand(thread_rng, T)) < log_alpha_2
                            theta_j = theta_star
                            ll_j = ll_star
                            lp_j = lp_star
                            ll_cheap[j] = ll_cheap_star
                            if sol_star isa ProjectionSolution
                                solutions[j] = sol_star
                            end
                            Threads.atomic_add!(total_accepted, 1)
                        end
                    else
                        # ── Standard single-stage PMMH: fresh unconditional bootstrap PF ──
                        ll_star, sol_star = _solve_and_run_pf(
                            spec, param_names, theta_star, observables, measurement_error,
                            solver, mh_solver_kwargs, ws, data, T_obs, thread_rng;
                            use_csmc=false, return_solution=true,
                            failures=lik_failures, evals=lik_evals)

                        if ll_star == T(-Inf)
                            Threads.atomic_add!(total_proposed, 1)
                            continue
                        end

                        # PMMH acceptance: unbiased PF estimate for θ* vs stored incumbent for θ_j.
                        log_alpha = (phi_new * ll_star + lp_star) - (phi_new * ll_j + lp_j)

                        if log(rand(thread_rng, T)) < log_alpha
                            theta_j = theta_star
                            ll_j = ll_star   # keep the noisy accepted estimate (PMMH)
                            lp_j = lp_star
                            if sol_star isa ProjectionSolution
                                solutions[j] = sol_star
                            end
                            Threads.atomic_add!(total_accepted, 1)
                        end
                    end

                    Threads.atomic_add!(total_proposed, 1)
                end

                # Write back
                state.theta_particles[:, j] = theta_j
                state.log_likelihoods[j] = ll_j
                state.log_priors[j] = lp_j
            end
        end

        # Record acceptance rate
        tp = total_proposed[]
        acc_rate = tp > 0 ? T(total_accepted[]) / T(tp) : zero(T)
        push!(state.acceptance_rates, acc_rate)

        # Adapt N_x on the PF ESTIMATOR variance at fixed θ (Chopin et al. 2013 §3.4),
        # NOT var(ll across θ) which just reflects posterior spread and doubles N_x
        # spuriously on any diffuse posterior (E-11 / #135).
        #
        # A cheap gate keeps the common case free: only when the PMMH acceptance rate
        # collapses (the signature of a too-noisy likelihood estimate) do we spend the
        # duplicate-run variance probe. A healthy chain — including on a diffuse posterior
        # with good acceptance — skips the probe entirely and N_x is left unchanged, so
        # N_x is never doubled merely because the posterior is diffuse.
        if acc_rate < T(0.15)
            est_var = _pf_estimator_variance(spec, param_names, state.theta_particles,
                observables, measurement_error, solver, solver_kwargs, pool, data, T_obs,
                rng; n_probe=min(N, 8), n_rep=2)   # target estimator variance ≈ 1–3
            N_x_new = _adapt_n_particles(N_x, est_var, T(3))
            if N_x_new > N_x
                N_x = N_x_new
                for ws in pool
                    _resize_pf_workspace!(ws, N_x)
                end
                # Exchange step: recompute ALL θ-likelihoods at the new N_x so the outer
                # weights are not a mix of estimates from two different N_x (no stale values).
                _exchange_step!(state, spec, param_names, observables, measurement_error,
                    solver, solver_kwargs, pool, data, T_obs, rng)
            end
        end

        phi = phi_new
    end

    # Terminal resample so the stored draws are equal-weighted (E-09 / #132).
    _terminal_resample!(state, N, rng)

    state.n_lik_failures = lik_failures[]
    state.n_lik_evals = lik_evals[]
    return state
end
