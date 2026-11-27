# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Sequential Monte Carlo (SMC) sampler and adaptive Random-Walk Metropolis-Hastings
for Bayesian DSGE estimation.

Implements:
- `_build_likelihood_fn` — closure mapping θ → log p(Y|θ) via Kalman filter
- `_log_prior` — log prior density evaluation with bound checking
- `_adaptive_tempering` — bisection for ESS-targeted tempering schedule
- `_update_proposal_cov!` — weighted covariance for RWMH proposals
- `_smc_mutation!` — parallel RWMH mutation of SMC particles
- `_smc_from_likelihood` — generic Herbst–Schorfheide tempering loop given a likelihood closure
- `_smc_sample` — RA SMC: builds a gensys/Kalman likelihood and delegates to `_smc_from_likelihood`
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
function _effective_obs_offset(d::AbstractVector{T}, new_spec::ModelSpec{T},
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
- `spec::ModelSpec{T}` — model specification (template)
- `param_names::Vector{Symbol}` — which parameters θ maps to
- `data::Matrix{T}` — n_obs × T_obs data matrix
- `observables::Vector{Symbol}` — observed endogenous variables
- `measurement_error` — measurement error SDs or `nothing`
- `solver::Symbol` — solver method (e.g., `:gensys`)
- `solver_kwargs::NamedTuple` — additional keyword arguments for `solve`

# Keywords
- `trends::Union{Nothing,ObservationTrends{T}}` — deterministic trend terms in the
  measurement equation. When supplied, `trendₜ` is re-evaluated at the current θ (its
  coefficients may be model parameters) and subtracted from the data before filtering,
  which is exactly `yₜ = d + Z sₜ + trendₜ + vₜ`.

# Returns
Closure `(θ::Vector{T}) → T` returning log-likelihood or `-Inf`.
"""
function _build_likelihood_fn(spec::ModelSpec{T}, param_names::Vector{Symbol},
                               data::AbstractMatrix, observables::Vector{Symbol},
                               measurement_error, solver::Symbol,
                               solver_kwargs::NamedTuple;
                               trends::Union{Nothing,ObservationTrends{T}}=nothing,
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

            # Deterministic observation trends (T240): y_t = d + Z s_t + trend_t + v_t.
            # Coefficients may be model parameters, so the trend is rebuilt at each θ and
            # removed from the data before filtering (equivalent to a time-varying d).
            data_t = trends === nothing ? data :
                     data .- _trend_matrix(trends, new_spec.param_values, size(data, 2), T)

            # Evaluate Kalman log-likelihood
            ll = _kalman_loglikelihood(ss, data_t)

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
# Safe incremental tempering log-weight avoiding IEEE-754 `0 * (-Inf) == NaN` (#950).
@inline function _safe_temper_inc(delta_phi::T, ll::T) where {T<:AbstractFloat}
    return isfinite(ll) ? delta_phi * ll : (delta_phi > zero(T) ? T(-Inf) : zero(T))
end

function _adaptive_tempering(log_liks::AbstractVector{T}, log_weights::AbstractVector{T},
                              phi_old::T, ess_target::Real, N::Int) where {T<:AbstractFloat}
    # Identify particles with finite cumulative weight and finite log-likelihood (#950).
    lw_active = T[]
    @inbounds for j in 1:N
        if isfinite(log_weights[j]) && isfinite(log_liks[j])
            push!(lw_active, log_weights[j])
        end
    end
    isempty(lw_active) && return phi_old

    # Effective ESS ceiling as Δφ → 0⁺ once -Inf particles receive zero weight.
    a0 = _logsumexp(lw_active)
    b0 = _logsumexp(T(2) .* lw_active)
    ess_0plus = isfinite(a0) ? exp(T(2) * a0 - b0) : zero(T)

    # Normal case: active support exceeds ess_target * N → standard target.
    # High-indeterminacy case (e.g., 60–80% of prior draws are indeterminate at φ=0):
    # bisect relative to the surviving valid support ceiling so Δφ advances cleanly,
    # resampling then eliminates all -Inf particles and ΔlogML captures log P(valid).
    nominal_target = T(ess_target) * N
    target_ess = ess_0plus > nominal_target ? nominal_target :
                 max(one(T), T(ess_target) * ess_0plus)

    # Helper: ESS of the updated cumulative weights at a given phi.
    lw_buf = Vector{T}(undef, N)
    function _compute_ess(phi)
        delta_phi = phi - phi_old
        @inbounds for j in 1:N
            lw_buf[j] = log_weights[j] + _safe_temper_inc(delta_phi, log_liks[j])
        end
        a = _logsumexp(lw_buf)
        isfinite(a) || return zero(T)            # all weights collapsed
        b = _logsumexp(T(2) .* lw_buf)
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

Update the SMC proposal covariance as `state.c_scale² * Σ_weighted` where
`state.c_scale` starts at `2.38 / sqrt(n_params)` (Roberts & Rosenthal 2001) and
adapts across tempering stages via [`_adapt_proposal_scale!`](@ref) (Herbst &
Schorfheide 2014, 2015), and `Σ_weighted` is the weighted covariance of the theta
particles.

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

    # Scale by adaptive factor c_n² (initialized to (2.38)² / n_params; #951)
    c2 = state.c_scale^2
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

"""
    _adapt_proposal_scale!(state::SMCState{T}, acc_rate::Real;
                           target_acc::Real=0.25, c_min::Real=0.05,
                           c_max::Real=5.0) -> T

Herbst & Schorfheide (2014, 2015) smooth logistic scaling update for SMC mutation:

```math
c_{n+1} = \\mathrm{clamp}\\!\\left(
    c_n \\left(0.95 + \\frac{0.10}{1 + \\exp(-16(\\hat{\\alpha}_n - \\alpha^*))}\\right),
    c_{\\min}, c_{\\max}
\\right).
```

Strictly monotone in `acc_rate`, fixed point at `acc_rate == target_acc`, bounded
step in `(0.95, 1.05)` per tempering stage. Updates `state.c_scale` in-place and
returns the new scale.
"""
function _adapt_proposal_scale!(state::SMCState{T}, acc_rate::Real;
                                target_acc::Real=0.25,
                                c_min::Real=0.05,
                                c_max::Real=5.0) where {T<:AbstractFloat}
    factor = T(0.95) + T(0.10) / (one(T) + exp(-T(16) * (T(acc_rate) - T(target_acc))))
    state.c_scale = clamp(state.c_scale * factor, T(c_min), T(c_max))
    return state.c_scale
end

"""
    _build_block_proposals(proposal_cov::Matrix{T}, n_blocks::Int, rng::AbstractRNG)
        -> (blocks::Vector{Vector{Int}}, block_Ls::Vector{LowerTriangular{T,Matrix{T}}})

Randomized parameter blocking (Herbst & Schorfheide 2014 §3.3): draw a random
permutation of `1:n_params`, partition into `B = min(n_blocks, n_params)` near-equal
blocks `I_b`, and compute the conditional covariance Cholesky factor `L_b` of
`Σ_{b|-b} = Σ_{bb} - Σ_{b,-b} Σ_{-b,-b}^{-1} Σ_{-b,b}` for each block once per
tempering stage.
"""
function _build_block_proposals(proposal_cov::Matrix{T}, n_blocks::Int,
                                 rng::AbstractRNG) where {T<:AbstractFloat}
    n_params = size(proposal_cov, 1)
    B = min(n_blocks, n_params)
    perm = randperm(rng, n_params)
    ranges = _chunk_ranges(n_params, B)
    blocks = Vector{Vector{Int}}(undef, B)
    block_Ls = Vector{LowerTriangular{T,Matrix{T}}}(undef, B)

    for b in 1:B
        I_b = perm[ranges[b]]
        blocks[b] = I_b
        if B == 1 || length(I_b) == n_params
            Σ_cond = copy(proposal_cov)
        else
            I_neg = Int[idx for idx in perm if !(idx in I_b)]
            Σ_bb = proposal_cov[I_b, I_b]
            Σ_bneg = proposal_cov[I_b, I_neg]
            Σ_negneg = proposal_cov[I_neg, I_neg]
            Σ_cond = Σ_bb - Σ_bneg * robust_inv(Symmetric(Σ_negneg); silent=true) * transpose(Σ_bneg)
        end
        Σ_cond = Matrix{T}((Σ_cond + transpose(Σ_cond)) / 2)
        min_diag = T(1e-10)
        @inbounds for i in 1:length(I_b)
            if Σ_cond[i, i] < min_diag
                Σ_cond[i, i] = min_diag
            end
        end
        L_b = try
            cholesky(Hermitian(Σ_cond)).L
        catch
            try
                cholesky(Hermitian(Σ_cond + T(1e-6) * I)).L
            catch
                LowerTriangular(Matrix{T}(safe_cholesky(Hermitian(Σ_cond); silent=true)))
            end
        end
        block_Ls[b] = L_b
    end
    return blocks, block_Ls
end

"""
    _resolve_n_screen(n_screen::Union{Nothing,Int}, n_particles::Int) -> Int

Resolve the screening particle count for two-stage delayed acceptance SMC².
When `n_screen === nothing`, chooses `clamp(n_particles ÷ 5, 20, n_particles ÷ 2)`
for `n_particles > 20` (and `max(5, n_particles ÷ 2)` for very small `n_particles <= 20`).
When an explicit integer is passed, validates `n_screen >= 1` and emits a warning if
`n_screen >= n_particles`.
"""
function _resolve_n_screen(n_screen::Union{Nothing,Int}, n_particles::Int)
    if n_screen === nothing
        return n_particles <= 20 ? max(5, n_particles ÷ 2) :
               clamp(n_particles ÷ 5, 20, n_particles ÷ 2)
    end
    n_screen >= 1 || throw(ArgumentError("n_screen must be >= 1, got $n_screen"))
    if n_screen >= n_particles
        @warn "delayed_acceptance: n_screen ($n_screen) >= n_particles ($n_particles); " *
              "screening PF is at least as expensive as the full PF. Pass n_screen < n_particles " *
              "or n_screen=nothing for automatic sizing."
    end
    return n_screen
end

# =============================================================================
# SMC mutation step
# =============================================================================

"""
    _smc_mutation!(state, phi, ll_fn, prior, n_mh_steps, rng; n_blocks=1)

Mutate SMC particles via `n_mh_steps` random-walk Metropolis-Hastings steps.

When `n_blocks == 1`, proposes full-vector updates `θ* = θ + L * z` where
`L = cholesky(proposal_cov).L`. When `n_blocks > 1` (and `n_params > 1`), partitions
parameters into `min(n_blocks, n_params)` randomized blocks per stage with conditional
covariance proposals `Σ_{b|-b}` (Herbst & Schorfheide 2014 §3.3).

Tracks acceptance rate across all particles, steps, and blocks.
"""
function _smc_mutation!(state::SMCState{T}, phi::T, ll_fn, prior::DSGEPrior{T},
                         n_mh_steps::Int, rng::AbstractRNG;
                         n_blocks::Int=1) where {T<:AbstractFloat}
    n_blocks >= 1 || throw(ArgumentError("n_blocks must be >= 1, got $n_blocks"))
    n_params, N = size(state.theta_particles)
    use_blocking = (n_blocks > 1 && n_params > 1)

    local L, blocks, block_Ls
    if use_blocking
        blocks, block_Ls = _build_block_proposals(state.proposal_cov, n_blocks, rng)
        L = LowerTriangular(zeros(T, 0, 0))
    else
        # Compute Cholesky of proposal covariance
        L = try
            cholesky(Hermitian(state.proposal_cov)).L
        catch
            # Add jitter if not positive definite
            cov_jittered = state.proposal_cov + T(1e-6) * I
            cholesky(Hermitian(cov_jittered)).L
        end
        blocks = Vector{Vector{Int}}()
        block_Ls = Vector{LowerTriangular{T,Matrix{T}}}()
    end

    # Pre-generate per-particle seeds BEFORE entering parallel work-stealing loop (#947)
    particle_seeds = [hash((j, rand(rng, UInt64))) for j in 1:N]

    nw = min(max(Threads.nthreads(), 1), max(N, 1))
    worker_acc = zeros(Int, nw)
    worker_prop = zeros(Int, nw)

    _with_single_blas_thread() do
        _parallel_for_worksteal(N; n_workers=nw) do j, w
            thread_rng = Random.Xoshiro(particle_seeds[j])

            theta_j = state.theta_particles[:, j]
            ll_j = state.log_likelihoods[j]
            lp_j = state.log_priors[j]
            acc_local = 0
            prop_local = 0

            for _ in 1:n_mh_steps
                if use_blocking
                    for b in eachindex(blocks)
                        I_b = blocks[b]
                        L_b = block_Ls[b]
                        z_b = randn(thread_rng, T, length(I_b))
                        step_b = L_b * z_b
                        theta_star = copy(theta_j)
                        @inbounds for m in eachindex(I_b)
                            theta_star[I_b[m]] += step_b[m]
                        end

                        lp_star = _log_prior(theta_star, prior)
                        if lp_star == T(-Inf)
                            prop_local += 1
                            continue
                        end

                        ll_star = ll_fn(theta_star)
                        if ll_star == T(-Inf)
                            prop_local += 1
                            continue
                        end

                        log_alpha = (phi * ll_star + lp_star) - (phi * ll_j + lp_j)
                        if log(rand(thread_rng, T)) < log_alpha
                            theta_j = theta_star
                            ll_j = ll_star
                            lp_j = lp_star
                            acc_local += 1
                        end
                        prop_local += 1
                    end
                else
                    # Propose: theta_star = theta_j + L * z
                    z = randn(thread_rng, T, n_params)
                    theta_star = theta_j + L * z

                    # Evaluate prior
                    lp_star = _log_prior(theta_star, prior)
                    if lp_star == T(-Inf)
                        prop_local += 1
                        continue
                    end

                    # Evaluate likelihood
                    ll_star = ll_fn(theta_star)
                    if ll_star == T(-Inf)
                        prop_local += 1
                        continue
                    end

                    # MH acceptance ratio (tempered)
                    log_alpha = (phi * ll_star + lp_star) - (phi * ll_j + lp_j)

                    if log(rand(thread_rng, T)) < log_alpha
                        theta_j = theta_star
                        ll_j = ll_star
                        lp_j = lp_star
                        acc_local += 1
                    end

                    prop_local += 1
                end
            end

            # Write back
            state.theta_particles[:, j] = theta_j
            state.log_likelihoods[j] = ll_j
            state.log_priors[j] = lp_j
            worker_acc[w] += acc_local
            worker_prop[w] += prop_local
        end
    end

    # Record acceptance rate (reduced once from per-worker counters)
    tp = sum(worker_prop)
    acc_rate = tp > 0 ? T(sum(worker_acc)) / T(tp) : zero(T)
    push!(state.acceptance_rates, acc_rate)

    return nothing
end

# =============================================================================
# Generic SMC algorithm (Herbst & Schorfheide 2014)
# =============================================================================

"""
    _smc_from_likelihood(ll_fn, param_names, prior, θ0; kwargs...) → SMCState

Generic Herbst–Schorfheide SMC loop driven by an arbitrary `θ → log p(Y|θ)`
closure. Shared by RA `_smc_sample` (Kalman/gensys likelihood) and HA
`method=:smc` (reduced-system likelihood). Mutation is `_smc_mutation!`.

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
"""
function _smc_from_likelihood(ll_fn, param_names::Vector{Symbol},
                              prior::DSGEPrior{T}, _theta0::AbstractVector{T};
                              n_smc::Int=500, n_mh_steps::Int=1,
                              ess_target::Real=0.5,
                              n_blocks::Int=1,
                              max_stages::Int=500, min_dphi::Real=1e-10,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n_blocks >= 1 || throw(ArgumentError("n_blocks must be >= 1, got $n_blocks"))
    n_params = length(param_names)
    N = n_smc

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
                    # Fail loudly rather than silently substituting the interval midpoint,
                    # which would distort the prior and hide a pathological prior/model
                    # region where draws never land in bounds (#150).
                    throw(ArgumentError(
                        "SMC prior initialization failed: parameter `$(param_names[i])` " *
                        "(prior $(prior.distributions[i]), support bounds " *
                        "[$(prior.lower[i]), $(prior.upper[i])]) produced 100 consecutive " *
                        "out-of-bounds draws. Check the prior support and model determinacy " *
                        "in that region."))
                end
            end
        end
        log_priors[j] = _log_prior(theta_particles[:, j], prior)
    end

    # Initialize weights uniformly
    log_weights = fill(-log(T(N)), N)

    # Evaluate log-likelihoods (parallel work-stealing under scoped single-thread BLAS; #947)
    log_likelihoods = fill(T(-Inf), N)
    _with_single_blas_thread() do
        _parallel_for_worksteal(N) do j, _
            log_likelihoods[j] = ll_fn(theta_particles[:, j])
        end
    end

    if all(!isfinite, log_likelihoods)
        error("SMC initialization failed: all $N prior particles produced non-finite " *
              "log-likelihoods (-Inf). Check model determinacy, prior support, or data scaling.")
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
        PFWorkspace{T}[],     # pf_workspace_pool (unused on this path)
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
        if all(!isfinite, state.log_likelihoods)
            error("SMC tempering failed at stage $stage (φ=$phi): all $N particles have " *
                  "non-finite log-likelihoods (-Inf).")
        end

        # Find next tempering parameter using true -Inf likelihoods (no -1e10 sentinel; #950)
        phi_new = _adaptive_tempering(state.log_likelihoods, state.log_weights, phi, ess_target, N)
        # Abort a degenerate-likelihood stall before paying for the expensive mutation (#145).
        _check_tempering_progress(stage, max_stages, phi, phi_new, min_step)
        delta_phi = phi_new - phi

        # Compute incremental log weights (preserving true -Inf for failed/indeterminate draws)
        inc_log_w = T[_safe_temper_inc(delta_phi, state.log_likelihoods[j]) for j in 1:N]

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

        # Mutate particles via RWMH (optionally blocked; #951)
        _smc_mutation!(state, phi_new, ll_fn, prior, n_mh_steps, rng; n_blocks=n_blocks)

        # Adapt proposal scale c_n toward target ~25% acceptance (Herbst & Schorfheide 2014; #951)
        _adapt_proposal_scale!(state, last(state.acceptance_rates))

        phi = phi_new
    end

    # Terminal resample so the stored draws are equal-weighted (E-09 / #132).
    _terminal_resample!(state, N, rng)

    return state
end

"""
    _smc_sample(spec, data, param_names, prior, θ0; kwargs...) → SMCState

Full Sequential Monte Carlo sampler for Bayesian DSGE estimation.

Builds a gensys/Kalman `θ → log p(Y|θ)` closure and delegates the tempering
loop to `_smc_from_likelihood`. Mutation is `_smc_mutation!`.

# Arguments
- `spec::ModelSpec{T}` — model specification
- `data::Matrix{T}` — n_obs × T_obs data matrix
- `param_names::Vector{Symbol}` — parameters to estimate
- `prior::DSGEPrior{T}` — prior specification
- `θ0::Vector{T}` — initial parameter guess (used for fallback)

# Keywords
- `n_smc::Int=500` — number of SMC particles
- `n_mh_steps::Int=1` — MH steps per mutation stage
- `ess_target::Real=0.5` — target ESS fraction
- `n_blocks::Int=1` — randomized parameter blocks per mutation stage (Herbst & Schorfheide 2014)
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
function _smc_sample(spec::ModelSpec{T}, data::AbstractMatrix,
                      param_names::Vector{Symbol}, prior::DSGEPrior{T},
                      theta0::AbstractVector{T};
                      n_smc::Int=500, n_mh_steps::Int=1,
                      ess_target::Real=0.5,
                      n_blocks::Int=1,
                      observables::Vector{Symbol}=spec.endog,
                      measurement_error=nothing,
                      solver::Symbol=:gensys,
                      solver_kwargs::NamedTuple=NamedTuple(),
                      trends::Union{Nothing,ObservationTrends{T}}=nothing,
                      max_stages::Int=500, min_dphi::Real=1e-10,
                      rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    data = Matrix{T}(data)  # convert Adjoint/SubArray to concrete Matrix

    # Build likelihood function (tracking failed/total likelihood evaluations)
    lik_failures = Threads.Atomic{Int}(0)
    lik_evals = Threads.Atomic{Int}(0)
    ll_fn = _build_likelihood_fn(spec, param_names, data, observables,
                                  measurement_error, solver, solver_kwargs;
                                  trends=trends,
                                  failures=lik_failures, evals=lik_evals)

    state = _smc_from_likelihood(ll_fn, param_names, prior, theta0;
                                 n_smc=n_smc, n_mh_steps=n_mh_steps,
                                 ess_target=ess_target, n_blocks=n_blocks,
                                 max_stages=max_stages,
                                 min_dphi=min_dphi, rng=rng)
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
- `spec::ModelSpec{T}` — model specification
- `data::Matrix{T}` — n_obs × T_obs data matrix
- `param_names::Vector{Symbol}` — parameters to estimate
- `prior::DSGEPrior{T}` — prior specification
- `θ0::Vector{T}` — initial parameter vector

# Keywords
- `n_draws::Int=5000` — total number of draws (including burnin)
- `burnin::Int=1000` — burn-in length. `_mh_sample` returns the FULL chain; the caller
  (`estimate_dsge_bayes`) discards the first `burnin` draws unless `keep_burnin=true`.
- `adapt_interval::Int=100` — adapt proposal covariance every N draws
- `init_proposal_cov::Union{Nothing,AbstractMatrix}=nothing` — initial proposal
  covariance in the SAMPLING space (already scaled, e.g. `c²·H⁻¹` from
  `posterior_mode`, transformed by the caller when `transform=true`); default `c²·I`
- `transform::Bool=false` — random-walk in the prior-transformed unconstrained
  space (log/logit via `ParameterTransform`): the target becomes
  `log posterior(θ(y)) + log|J(y)|` so the θ-space posterior is preserved, no
  proposal is ever wasted outside the support, and stored draws are
  back-transformed to θ before being returned
- `observables::Vector{Symbol}` — observed variables
- `measurement_error` — measurement error SDs or `nothing`
- `solver::Symbol=:gensys` — DSGE solver method
- `solver_kwargs::NamedTuple` — additional solver kwargs
- `rng::AbstractRNG` — random number generator

# Returns
- `draws::Matrix{T}` — `n_draws × n_params` matrix of posterior draws (θ-space)
- `log_posterior::Vector{T}` — log posterior `log L + log π` at each draw
  (θ-space kernel, no Jacobian, regardless of `transform`)
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
function _mh_sample(spec::ModelSpec{T}, data::AbstractMatrix,
                     param_names::Vector{Symbol}, prior::DSGEPrior{T},
                     theta0::AbstractVector{T};
                     n_draws::Int=5000, burnin::Int=1000,
                     adapt_interval::Int=100,
                     init_proposal_cov::Union{Nothing,AbstractMatrix}=nothing,
                     transform::Bool=false,
                     observables::Vector{Symbol}=spec.endog,
                     measurement_error=nothing,
                     solver::Symbol=:gensys,
                     solver_kwargs::NamedTuple=NamedTuple(),
                     trends::Union{Nothing,ObservationTrends{T}}=nothing,
                     rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    data = Matrix{T}(data)  # convert Adjoint/SubArray to concrete Matrix
    n_params = length(param_names)

    # Build likelihood function (tracking failed/total likelihood evaluations)
    lik_failures = Threads.Atomic{Int}(0)
    lik_evals = Threads.Atomic{Int}(0)
    ll_fn = _build_likelihood_fn(spec, param_names, data, observables,
                                  measurement_error, solver, solver_kwargs;
                                  trends=trends,
                                  failures=lik_failures, evals=lik_evals)

    # Transform layer: the walk runs on x (= θ untransformed, or y = T(θ)
    # unconstrained); acceptance uses log posterior(θ(x)) + log|J(x)|
    pt = transform ? ParameterTransform(prior.lower, prior.upper) : nothing
    to_nat = transform ? (x -> to_constrained(pt, x)) : identity
    ljac = transform ? (x -> log_jacobian(pt, x)) : (x -> zero(T))

    # Initialize (θ-space first, then map into the sampling space)
    theta_current = copy(theta0)
    if transform
        # Nudge strictly inside the support so the transform is finite
        for i in 1:n_params
            lo, hi = prior.lower[i], prior.upper[i]
            span = isfinite(lo) && isfinite(hi) ? hi - lo : one(T)
            margin = T(1e-8) * span
            isfinite(lo) && theta_current[i] <= lo && (theta_current[i] = lo + margin)
            isfinite(hi) && theta_current[i] >= hi && (theta_current[i] = hi - margin)
        end
    end
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

    x_current = transform ? to_unconstrained(pt, theta_current) : theta_current
    lj_current = ljac(x_current)

    # Storage: θ-space draws for the caller, sampling-space path for adaptation
    draws = zeros(T, n_draws, n_params)
    log_posterior = zeros(T, n_draws)
    xs = transform ? zeros(T, n_draws, n_params) : draws

    # Proposal covariance: caller-supplied (already scaled and in the sampling
    # space, e.g. c²·H⁻¹ from posterior_mode) or identity scaled by c²
    c2 = T(2.38)^2 / T(n_params)
    proposal_cov = init_proposal_cov === nothing ?
        c2 * Matrix{T}(I, n_params, n_params) : Matrix{T}(init_proposal_cov)
    scale_factor = one(T)

    # Cholesky of proposal
    proposal_L = try
        cholesky(Hermitian(proposal_cov)).L
    catch
        cholesky(Hermitian(proposal_cov + T(1e-8) * I)).L
    end

    total_accepted = 0
    window_accepted = 0
    # Adaptation is confined to burn-in (Roberts & Rosenthal 2007): the proposal is
    # frozen at draw == burnin so the retained chain targets the exact posterior.
    proposal_L_at_burnin = copy(proposal_L)
    scale_at_burnin = scale_factor
    window_acc_history = T[]
    cum_acc_history = T[]

    for draw in 1:n_draws
        # Propose in the sampling space
        z = randn(rng, T, n_params)
        x_star = x_current + scale_factor * proposal_L * z
        theta_star = to_nat(x_star)

        # Evaluate
        lp_star = _log_prior(theta_star, prior)

        if isfinite(lp_star)
            ll_star = ll_fn(theta_star)

            if isfinite(ll_star)
                lj_star = ljac(x_star)
                # MH acceptance ratio; the Jacobian terms make the y-space walk
                # target the pushforward of the θ-space posterior
                log_alpha = (ll_star + lp_star + lj_star) -
                            (ll_current + lp_current + lj_current)

                if log(rand(rng, T)) < log_alpha
                    x_current = x_star
                    theta_current = theta_star
                    ll_current = ll_star
                    lp_current = lp_star
                    lj_current = lj_star
                    total_accepted += 1
                    window_accepted += 1
                end
            end
        end

        # Store (θ-space kernel, no Jacobian — the convention downstream
        # consumers like bridge_sampling_ml rely on)
        draws[draw, :] = theta_current
        log_posterior[draw] = ll_current + lp_current
        transform && (xs[draw, :] = x_current)

        # Adapt proposal covariance and scale — ONLY during burn-in (frozen after).
        if draw <= burnin && draw % adapt_interval == 0
            # Trailing-window acceptance signal (not the stale cumulative rate); the
            # covariance adapts to recent draws, so the scale must too.
            window_acc = T(window_accepted) / T(adapt_interval)
            push!(window_acc_history, window_acc)
            push!(cum_acc_history, T(total_accepted) / T(draw))
            window_accepted = 0

            if draw >= 2 * adapt_interval
                # Update proposal covariance from recent draws in the SAMPLING
                # space (xs === draws when transform is off, unconstrained y when on).
                recent_start = max(1, draw - 5 * adapt_interval)
                recent_draws = xs[recent_start:draw, :]

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
variance** at fixed θ (see `_pf_estimator_variance`) exceeds `threshold`,
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
function _pf_estimator_variance(spec::ModelSpec{T}, param_names::Vector{Symbol},
        theta_particles::AbstractMatrix{T}, observables::Vector{Symbol},
        measurement_error, solver::Symbol, solver_kwargs::NamedTuple,
        pool::Vector{PFWorkspace{T}}, data::Matrix{T}, T_obs::Int,
        rng::AbstractRNG; n_probe::Int=8, n_rep::Int=2) where {T<:AbstractFloat}
    N = size(theta_particles, 2)
    n_probe = min(n_probe, N)
    (n_probe < 1 || n_rep < 2) && return zero(T)
    idx = unique(round.(Int, range(1, N; length=n_probe)))
    n_p = length(idx)
    total_tasks = n_p * n_rep

    # Pre-generate all per-(probe, rep) seeds sequentially so 1T == NT bit-identically (#947)
    probe_seeds = Matrix{UInt64}(undef, n_p, n_rep)
    for pi in 1:n_p
        i = idx[pi]
        for r in 1:n_rep
            probe_seeds[pi, r] = hash((:nx_probe, i, r, rand(rng, UInt64)))
        end
    end

    ll_mat = fill(T(-Inf), n_p, n_rep)
    nw = min(length(pool), total_tasks)
    _with_single_blas_thread() do
        _parallel_for_worksteal(total_tasks; n_workers=nw) do task_id, w
            pi = ((task_id - 1) ÷ n_rep) + 1
            r = ((task_id - 1) % n_rep) + 1
            i = idx[pi]
            theta_i = Vector{T}(theta_particles[:, i])
            rr = Random.Xoshiro(probe_seeds[pi, r])
            ll_mat[pi, r] = _solve_and_run_pf(spec, param_names, theta_i, observables,
                measurement_error, solver, solver_kwargs, pool[w], data, T_obs, rr)
        end
    end

    per_theta = T[]
    for pi in 1:n_p
        reps = T[ll_mat[pi, r] for r in 1:n_rep if isfinite(ll_mat[pi, r])]
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
function _exchange_step!(state::SMCState{T}, spec::ModelSpec{T},
        param_names::Vector{Symbol}, observables::Vector{Symbol}, measurement_error,
        solver::Symbol, solver_kwargs::NamedTuple, pool::Vector{PFWorkspace{T}},
        data::Matrix{T}, T_obs::Int, rng::AbstractRNG;
        phi::Real=(isempty(state.phi_schedule) ? one(T) : state.phi_schedule[end]),
        ess_target::Real=0.5,
        solutions::Union{Nothing,Vector{Any}}=nothing,
        ll_cheap::Union{Nothing,Vector{T}}=nothing,
        screen_pool::Vector{PFWorkspace{T}}=PFWorkspace{T}[],
        failures::Threads.Atomic{Int}=Threads.Atomic{Int}(0),
        evals::Threads.Atomic{Int}=Threads.Atomic{Int}(0)) where {T<:AbstractFloat}
    N = size(state.theta_particles, 2)
    ll_old = copy(state.log_likelihoods)
    ll_new = fill(T(-Inf), N)
    seeds = [hash((:exchange, j, rand(rng, UInt64))) for j in 1:N]
    refresh_cheap = (ll_cheap !== nothing && !isempty(screen_pool))
    screen_seeds = refresh_cheap ? [hash((:exchange_screen, j, rand(rng, UInt64))) for j in 1:N] : UInt64[]
    nw = min(length(pool), max(N, 1))
    _with_single_blas_thread() do
        _parallel_for_worksteal(N; n_workers=nw) do j, w
            rr = Random.Xoshiro(seeds[j])
            theta_j = Vector{T}(state.theta_particles[:, j])
            if refresh_cheap
                # Single-solve refresh of both authoritative ll_new and screening ll_cheap (#951).
                # Re-drawing auxiliary ll_cheap ~ p_screen(.|theta_j) preserves the marginal target.
                Threads.atomic_add!(evals, 1)
                ss_j, sol_j = _solve_and_build_state_space(spec, param_names, theta_j,
                    observables, measurement_error, solver, solver_kwargs)
                if ss_j === nothing
                    Threads.atomic_add!(failures, 1)
                    ll_new[j] = T(-Inf)
                    ll_cheap[j] = T(-Inf)
                else
                    ll_j = _run_pf_on_state_space(ss_j, pool[w], data, T_obs, rr;
                                                  store_trajectory=false)
                    if !isfinite(ll_j)
                        Threads.atomic_add!(failures, 1)
                        ll_j = T(-Inf)
                    end
                    ll_new[j] = ll_j
                    if solutions !== nothing && sol_j isa ProjectionSolution
                        solutions[j] = sol_j
                    end
                    s_rng = Random.Xoshiro(screen_seeds[j])
                    ll_cheap[j] = _run_pf_on_state_space(ss_j, screen_pool[w], data, T_obs,
                                                         s_rng; store_trajectory=false)
                end
            else
                ll_new[j] = _solve_and_run_pf(spec, param_names,
                    theta_j, observables, measurement_error,
                    solver, solver_kwargs, pool[w], data, T_obs, rr;
                    failures=failures, evals=evals)
            end
        end
    end
    state.log_likelihoods .= ll_new

    # Chopin, Jacob & Papaspiliopoulos (2013 §3.4) exact exchange importance weights (#950):
    #   Δlog W_j = φ · (log Ẑ_{N_new}(θ_j, u_new) - log Ẑ_{N_old}(θ_j, u_old))
    phi_T = T(phi)
    delta_lw = Vector{T}(undef, N)
    @inbounds for j in 1:N
        if phi_T == zero(T)
            delta_lw[j] = zero(T)
        elseif !isfinite(ll_new[j]) || !isfinite(ll_old[j])
            delta_lw[j] = T(-Inf)
        else
            delta_lw[j] = phi_T * (ll_new[j] - ll_old[j])
        end
    end

    lse_old = _logsumexp(state.log_weights)
    new_lw = state.log_weights .+ delta_lw
    lse_new = _logsumexp(new_lw)
    if isfinite(lse_new) && isfinite(lse_old)
        state.log_marginal_likelihood += lse_new - lse_old
    else
        state.log_marginal_likelihood = T(-Inf)
    end
    state.log_weights .= new_lw

    # Resample if exchange weights collapse ESS below ess_target * N
    if isfinite(lse_new)
        weights_norm = Vector{T}(undef, N)
        _normalize_log_weights!(weights_norm, state.log_weights)
        ess = one(T) / sum(abs2, weights_norm)
        if isfinite(ess) && ess < T(ess_target) * N
            ancestors = collect(1:N)
            cumweights = zeros(T, N)
            _systematic_resample!(ancestors, weights_norm, cumweights, N, rng)

            theta_rs = similar(state.theta_particles)
            ll_rs = similar(state.log_likelihoods)
            lp_rs = similar(state.log_priors)
            sol_rs = solutions === nothing ? nothing : Vector{Any}(undef, N)
            llc_rs = ll_cheap === nothing ? nothing : similar(ll_cheap)
            @inbounds for j in 1:N
                a = ancestors[j]
                theta_rs[:, j] = state.theta_particles[:, a]
                ll_rs[j] = state.log_likelihoods[a]
                lp_rs[j] = state.log_priors[a]
                sol_rs !== nothing && (sol_rs[j] = solutions[a])
                llc_rs !== nothing && (llc_rs[j] = ll_cheap[a])
            end
            state.theta_particles .= theta_rs
            state.log_likelihoods .= ll_rs
            state.log_priors .= lp_rs
            sol_rs !== nothing && (solutions .= sol_rs)
            llc_rs !== nothing && (ll_cheap .= llc_rs)
            fill!(state.log_weights, -log(T(N)))
        end
    end
    return nothing
end

"""
    _solve_and_build_state_space(spec, param_names, theta, observables,
                                 measurement_error, solver, solver_kwargs) -> (ss, sol)

Update parameters, compute steady state, solve the DSGE model, and construct the
appropriate state space (`NonlinearStateSpace`, `ProjectionStateSpace`, or
`DSGEStateSpace`). Returns `(nothing, nothing)` on indeterminacy or benign
numerical solve failure; non-benign exceptions propagate.
"""
function _solve_and_build_state_space(spec::ModelSpec{T}, param_names::Vector{Symbol},
                                      theta::AbstractVector{T},
                                      observables::Vector{Symbol},
                                      measurement_error,
                                      solver::Symbol,
                                      solver_kwargs::NamedTuple) where {T<:AbstractFloat}
    try
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(param_names)
            new_pv[pn] = theta[i]
        end
        new_spec = _respec(spec, new_pv)
        new_spec = compute_steady_state(new_spec)
        sol = solve(new_spec; method=solver, solver_kwargs...)
        if !is_determined(sol)
            return nothing, nothing
        end

        # Build observation equation; apply the same effective-SS offset as the Kalman path
        # so linear=true constant models are not silently mis-offset in the PF (E-07 / #115).
        Z, d_vec, H_mat = _build_observation_equation(new_spec, observables, measurement_error)
        d_vec = _effective_obs_offset(d_vec, new_spec, sol, observables)

        if sol isa PerturbationSolution
            ss = _build_nonlinear_state_space(sol, Z, d_vec, H_mat)
        elseif sol isa ProjectionSolution
            ss = _build_projection_state_space(sol, Z, d_vec, H_mat)
        else
            ss = _build_state_space(sol, Z, d_vec, H_mat)
        end
        return ss, sol
    catch e
        if _benign_solve_error(e)
            return nothing, nothing
        end
        rethrow(e)
    end
end

"""
    _run_pf_on_state_space(ss, ws, data, T_obs, rng;
                           use_csmc=false, store_trajectory=false) -> T

Run a bootstrap PF or conditional SMC on an already-constructed state space `ss`.
Returns the log-likelihood `T` (or `T(-Inf)` if non-finite or on benign numeric failure).
"""
function _run_pf_on_state_space(ss, ws::PFWorkspace{T}, data::Matrix{T},
                                T_obs::Int, rng::AbstractRNG;
                                use_csmc::Bool=false,
                                store_trajectory::Bool=false) where {T<:AbstractFloat}
    try
        ll = if use_csmc
            _conditional_smc!(ws, ss, data, T_obs; rng=rng)
        else
            _bootstrap_particle_filter!(ws, ss, data, T_obs;
                                        store_trajectory=store_trajectory, rng=rng)
        end
        return isfinite(ll) ? ll : T(-Inf)
    catch e
        if _benign_solve_error(e)
            return T(-Inf)
        end
        rethrow(e)
    end
end

"""
    _solve_and_run_pf(spec, param_names, theta, observables, measurement_error,
                       solver, solver_kwargs, ws, data, T_obs, rng;
                       use_csmc=false, store_trajectory=false, return_solution=false)

Shared helper: update parameters → solve → dispatch state space → run PF/CSMC.
Thin wrapper around [`_solve_and_build_state_space`](@ref) and
[`_run_pf_on_state_space`](@ref).

Returns the log-likelihood `T` (or `T(-Inf)` on failure). When
`return_solution=true`, returns `(ll, solution)` tuple instead.
"""
function _solve_and_run_pf(spec::ModelSpec{T}, param_names::Vector{Symbol},
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
    ss, sol = _solve_and_build_state_space(spec, param_names, theta, observables,
                                           measurement_error, solver, solver_kwargs)
    if ss === nothing
        Threads.atomic_add!(failures, 1)
        return fail
    end
    ll = _run_pf_on_state_space(ss, ws, data, T_obs, rng;
                                use_csmc=use_csmc, store_trajectory=store_trajectory)
    if !isfinite(ll)
        Threads.atomic_add!(failures, 1)
        ll = T(-Inf)
    end
    return return_solution ? (ll, sol) : ll
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
- `spec::ModelSpec{T}` — model specification (template)
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
function _build_pf_likelihood_fn(spec::ModelSpec{T}, param_names::Vector{Symbol},
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
    _smc2_init_likelihoods!(log_likelihoods, solutions, theta_particles, spec,
                            param_names, observables, measurement_error, solver,
                            solver_kwargs, pf_ll_fn, pool, data, T_obs, seeds;
                            failures, evals) -> Nothing

Evaluate each SMC² θ-particle's initial PF log-likelihood in parallel (E-23 / #147).
Partitions the particles into `length(pool)` contiguous chunks (`_chunk_ranges`), gives
each chunk its **own** PF workspace (chunk-indexed, never `threadid()`-indexed — safe
under Julia's dynamic scheduler, #146), and seeds each particle's PF from the
caller-supplied pre-generated `seeds[j]`. The result is therefore **identical to a
serial run regardless of thread count**. `pf_ll_fn` and the projection-branch
`_solve_and_run_pf` share the sampler's failure/eval atomics via `failures`/`evals`.
"""
function _smc2_init_likelihoods!(log_likelihoods::Vector{T}, solutions::Vector{Any},
        theta_particles::AbstractMatrix{T}, spec::ModelSpec{T},
        param_names::Vector{Symbol}, observables::Vector{Symbol}, measurement_error,
        solver::Symbol, solver_kwargs::NamedTuple, pf_ll_fn,
        pool::Vector{PFWorkspace{T}}, data::Matrix{T}, T_obs::Int,
        seeds::AbstractVector{<:Integer};
        ll_cheap::Union{Nothing,Vector{T}}=nothing,
        screen_pool::Vector{PFWorkspace{T}}=PFWorkspace{T}[],
        screen_seeds::Union{Nothing,AbstractVector{<:Integer}}=nothing,
        failures::Threads.Atomic{Int}=Threads.Atomic{Int}(0),
        evals::Threads.Atomic{Int}=Threads.Atomic{Int}(0)) where {T<:AbstractFloat}
    N = size(theta_particles, 2)
    nw = min(length(pool), max(N, 1))
    do_screen = (ll_cheap !== nothing && !isempty(screen_pool) && screen_seeds !== nothing)
    _with_single_blas_thread() do
        _parallel_for_worksteal(N; n_workers=nw) do j, w
            ws = pool[w]
            thread_rng = Random.Xoshiro(seeds[j])
            theta_j = Vector{T}(theta_particles[:, j])
            if do_screen
                # Single-solve Stage 0 initialization for delayed acceptance (#951):
                # solve DSGE and build state space once, then run both full & screening PF.
                Threads.atomic_add!(evals, 1)
                ss_j, sol_j = _solve_and_build_state_space(spec, param_names, theta_j,
                    observables, measurement_error, solver, solver_kwargs)
                if ss_j === nothing
                    Threads.atomic_add!(failures, 1)
                    log_likelihoods[j] = T(-Inf)
                    ll_cheap[j] = T(-Inf)
                else
                    ll_j = _run_pf_on_state_space(ss_j, ws, data, T_obs, thread_rng;
                                                  store_trajectory=true)
                    if !isfinite(ll_j)
                        Threads.atomic_add!(failures, 1)
                        ll_j = T(-Inf)
                    end
                    log_likelihoods[j] = ll_j
                    if sol_j isa ProjectionSolution
                        solutions[j] = sol_j
                    end
                    screen_rng = Random.Xoshiro(screen_seeds[j])
                    ll_cheap[j] = _run_pf_on_state_space(ss_j, screen_pool[w], data, T_obs,
                                                         screen_rng; store_trajectory=false)
                end
            elseif solver in (:projection, :pfi)
                ll_j, sol_j = _solve_and_run_pf(spec, param_names, theta_j, observables,
                    measurement_error, solver, solver_kwargs,
                    ws, data, T_obs, thread_rng;
                    store_trajectory=true, return_solution=true,
                    failures=failures, evals=evals)
                log_likelihoods[j] = ll_j
                if sol_j isa ProjectionSolution
                    solutions[j] = sol_j
                end
            else
                log_likelihoods[j] = pf_ll_fn(theta_j, ws, thread_rng)
            end
        end
    end
    return nothing
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
- `spec::ModelSpec{T}` — model specification
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
- `delayed_acceptance::Bool=false` — enable single-solve two-stage delayed acceptance MH
  (Christen & Fox 2005). Solves the DSGE model at most once per proposal and pre-screens
  with a cheap bootstrap PF (`n_screen` particles) before running the full PF on the same
  state space, preserving exact posterior detailed balance.
- `n_screen::Union{Nothing,Int}=nothing` — number of particles for the screening PF in
  delayed acceptance (`nothing` auto-sizes to `clamp(n_particles ÷ 5, 20, n_particles ÷ 2)`
  for `n_particles > 20` or `max(5, n_particles ÷ 2)` otherwise)
- `n_blocks::Int=1` — randomized parameter blocks per mutation stage (Herbst & Schorfheide 2014)
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
function _smc2_sample(spec::ModelSpec{T}, data::AbstractMatrix,
                       param_names::Vector{Symbol}, prior::DSGEPrior{T},
                       theta0::AbstractVector{T};
                       n_smc::Int=200, n_particles::Int=100,
                       n_mh_steps::Int=1, ess_target::Real=0.5,
                       n_blocks::Int=1,
                       observables::Vector{Symbol}=spec.endog,
                       measurement_error=nothing,
                       solver::Symbol=:gensys,
                       solver_kwargs::NamedTuple=NamedTuple(),
                       delayed_acceptance::Bool=false,
                       n_screen::Union{Nothing,Int}=nothing,
                       max_stages::Int=500, min_dphi::Real=1e-10,
                       rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n_blocks >= 1 || throw(ArgumentError("n_blocks must be >= 1, got $n_blocks"))
    if n_screen !== nothing && n_screen < 1
        throw(ArgumentError("n_screen must be >= 1, got $n_screen"))
    end
    n_screen_eff = delayed_acceptance ? _resolve_n_screen(n_screen, n_particles) : 0
    data = Matrix{T}(data)
    n_params = length(param_names)
    N = n_smc
    N_x = n_particles
    T_obs = size(data, 2)
    n_obs = size(data, 1)

    # Determine state space dimensions from spec
    n_states = spec.n_endog
    n_shocks = spec.n_exog

    # Prevent nested @threads oversubscription when outer SMC² runs in parallel (#947)
    eff_solver_kwargs = if solver in (:projection, :pfi) && Threads.nthreads() > 1
        merge(solver_kwargs, (threaded=false,))
    else
        solver_kwargs
    end

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
        pf_order = get(eff_solver_kwargs, :order, 2)
        # Trial solve to get nx, nv dimensions
        try
            trial_spec = compute_steady_state(spec)
            trial_sol = solve(trial_spec; method=solver, eff_solver_kwargs...)
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
            trial_sol = solve(trial_spec; method=solver, eff_solver_kwargs...)
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
                                        measurement_error, solver, eff_solver_kwargs, N_x;
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
        [_allocate_pf_workspace(T, n_states, n_obs, n_shocks, n_screen_eff;
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
                    # Fail loudly rather than silently substituting the interval midpoint (#150).
                    throw(ArgumentError(
                        "SMC prior initialization failed: parameter `$(param_names[i])` " *
                        "(prior $(prior.distributions[i]), support bounds " *
                        "[$(prior.lower[i]), $(prior.upper[i])]) produced 100 consecutive " *
                        "out-of-bounds draws. Check the prior support and model determinacy " *
                        "in that region."))
                end
            end
        end
        log_priors[j] = _log_prior(theta_particles[:, j], prior)
    end

    # Initialize weights uniformly
    log_weights = fill(-log(T(N)), N)

    # Evaluate log-likelihoods via PF (and cache solutions for warm-starting), threaded
    # with per-chunk workspaces (#147). Pre-generate the per-particle seeds BEFORE threading,
    # in j-order, so the rng-consumption order — and hence the result and downstream rng state
    # — are bit-identical to the old serial loop and independent of thread count.
    log_likelihoods = fill(T(-Inf), N)
    init_seeds = UInt64[hash((j, rand(rng, UInt64))) for j in 1:N]
    screen_seeds = delayed_acceptance ?
        UInt64[hash((j, UInt64(0xDA), rand(rng, UInt64))) for j in 1:N] : nothing
    _smc2_init_likelihoods!(log_likelihoods, solutions, theta_particles, spec,
        param_names, observables, measurement_error, solver, eff_solver_kwargs, pf_ll_fn,
        pool, data, T_obs, init_seeds;
        ll_cheap=(delayed_acceptance ? ll_cheap : nothing),
        screen_pool=screen_pool, screen_seeds=screen_seeds,
        failures=lik_failures, evals=lik_evals)

    if all(!isfinite, log_likelihoods)
        error("SMC² initialization failed: all $N prior particles produced non-finite " *
              "log-likelihoods (-Inf). Check model determinacy, prior support, measurement_error, or data scaling.")
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
        if all(!isfinite, state.log_likelihoods)
            error("SMC² tempering failed at stage $stage (φ=$phi): all $N particles have " *
                  "non-finite log-likelihoods (-Inf).")
        end

        # Find next tempering parameter using true -Inf likelihoods (no -1e10 sentinel; #950)
        phi_new = _adaptive_tempering(state.log_likelihoods, state.log_weights, phi, ess_target, N)
        # Abort a degenerate-likelihood stall before paying for the expensive PF mutation (#145).
        _check_tempering_progress(stage, max_stages, phi, phi_new, min_step)
        delta_phi = phi_new - phi

        # Compute incremental log weights (preserving true -Inf for failed/indeterminate draws)
        inc_log_w = T[_safe_temper_inc(delta_phi, state.log_likelihoods[j]) for j in 1:N]

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

        # --- SMC² Mutation: PMMH / single-solve delayed acceptance (optionally blocked; #951) ---
        use_blocking = (n_blocks > 1 && n_params > 1)
        local L, blocks, block_Ls
        if use_blocking
            blocks, block_Ls = _build_block_proposals(state.proposal_cov, n_blocks, rng)
            L = LowerTriangular(zeros(T, 0, 0))
        else
            L = try
                cholesky(Hermitian(state.proposal_cov)).L
            catch
                cov_jittered = state.proposal_cov + T(1e-6) * I
                cholesky(Hermitian(cov_jittered)).L
            end
            blocks = Vector{Vector{Int}}()
            block_Ls = Vector{LowerTriangular{T,Matrix{T}}}()
        end

        # Pre-generate per-particle seeds BEFORE entering parallel work-stealing loop
        particle_seeds_mut = [hash((j, phi_new, rand(rng, UInt64))) for j in 1:N]

        # PMMH mutation (E-06 / #134) with dynamic work-stealing and scoped single-thread BLAS (#947).
        # Each worker w owns pool[w] (and screen_pool[w]) exclusively — zero threadid() hazards,
        # zero static-chunk stragglers, and per-worker local acc/prop counters reduced once.
        nw_mut = min(n_pool, max(N, 1))
        worker_acc = zeros(Int, nw_mut)
        worker_prop = zeros(Int, nw_mut)
        worker_s2_prop = zeros(Int, nw_mut)

        function _eval_smc2_proposal(theta_star::Vector{T}, theta_j::Vector{T},
                                     ll_j::T, lp_j::T, j::Int,
                                     ws::PFWorkspace{T}, ws_screen::PFWorkspace{T},
                                     thread_rng::AbstractRNG)
            lp_star = _log_prior(theta_star, prior)
            if lp_star == T(-Inf)
                return false, false, theta_j, ll_j, lp_j
            end

            mh_solver_kwargs = if solver in (:projection, :pfi) && solutions[j] isa ProjectionSolution
                merge(eff_solver_kwargs, (initial_coeffs=solutions[j].coefficients,))
            else
                eff_solver_kwargs
            end

            if delayed_acceptance
                # Single-solve delayed acceptance (#951): solve DSGE & build state space AT MOST ONCE,
                # reusing ss_star for both Stage 1 (ws_screen) and Stage 2 (ws) (or ll_cheap refresh).
                ss_star, sol_star = _solve_and_build_state_space(
                    spec, param_names, theta_star, observables, measurement_error,
                    solver, mh_solver_kwargs)
                if ss_star === nothing
                    if !isfinite(ll_cheap[j])
                        Threads.atomic_add!(lik_evals, 1)
                        Threads.atomic_add!(lik_failures, 1)
                    end
                    return false, false, theta_j, ll_j, lp_j
                end

                if isfinite(ll_cheap[j])
                    # Stage 1: cheap screening PF on ss_star
                    ll_cheap_star = _run_pf_on_state_space(
                        ss_star, ws_screen, data, T_obs, thread_rng;
                        use_csmc=false, store_trajectory=false)
                    if !isfinite(ll_cheap_star)
                        return false, false, theta_j, ll_j, lp_j
                    end

                    log_alpha_1 = (phi_new * ll_cheap_star + lp_star) -
                                  (phi_new * ll_cheap[j] + lp_j)
                    if log(rand(thread_rng, T)) >= log_alpha_1
                        return false, false, theta_j, ll_j, lp_j
                    end

                    # Stage 2: authoritative PF on the EXACT SAME ss_star (zero second solve)
                    Threads.atomic_add!(lik_evals, 1)
                    ll_star = _run_pf_on_state_space(
                        ss_star, ws, data, T_obs, thread_rng;
                        use_csmc=false, store_trajectory=false)
                    if !isfinite(ll_star)
                        Threads.atomic_add!(lik_failures, 1)
                        return false, true, theta_j, ll_j, lp_j
                    end

                    log_alpha_2 = isfinite(ll_j) ?
                        (phi_new * (ll_star - ll_cheap_star) - phi_new * (ll_j - ll_cheap[j])) :
                        T(Inf)
                    if log(rand(thread_rng, T)) < log_alpha_2
                        ll_cheap[j] = ll_cheap_star
                        if sol_star isa ProjectionSolution
                            solutions[j] = sol_star
                        end
                        return true, true, theta_star, ll_star, lp_star
                    end
                    return false, true, theta_j, ll_j, lp_j
                else
                    # Fallback when incumbent ll_cheap[j] == -Inf (#950): single-stage PMMH + refresh ll_cheap
                    Threads.atomic_add!(lik_evals, 1)
                    ll_star = _run_pf_on_state_space(
                        ss_star, ws, data, T_obs, thread_rng;
                        use_csmc=false, store_trajectory=false)
                    if !isfinite(ll_star)
                        Threads.atomic_add!(lik_failures, 1)
                        return false, true, theta_j, ll_j, lp_j
                    end

                    log_alpha = isfinite(ll_j) ?
                        ((phi_new * ll_star + lp_star) - (phi_new * ll_j + lp_j)) :
                        T(Inf)
                    if log(rand(thread_rng, T)) < log_alpha
                        if sol_star isa ProjectionSolution
                            solutions[j] = sol_star
                        end
                        ll_cheap[j] = _run_pf_on_state_space(
                            ss_star, ws_screen, data, T_obs, thread_rng;
                            use_csmc=false, store_trajectory=false)
                        return true, true, theta_star, ll_star, lp_star
                    end
                    return false, true, theta_j, ll_j, lp_j
                end
            else
                # Standard single-stage PMMH
                ll_star, sol_star = _solve_and_run_pf(
                    spec, param_names, theta_star, observables, measurement_error,
                    solver, mh_solver_kwargs, ws, data, T_obs, thread_rng;
                    use_csmc=false, return_solution=true,
                    failures=lik_failures, evals=lik_evals)
                if ll_star == T(-Inf)
                    return false, true, theta_j, ll_j, lp_j
                end

                log_alpha = isfinite(ll_j) ?
                    ((phi_new * ll_star + lp_star) - (phi_new * ll_j + lp_j)) :
                    T(Inf)
                if log(rand(thread_rng, T)) < log_alpha
                    if sol_star isa ProjectionSolution
                        solutions[j] = sol_star
                    end
                    return true, true, theta_star, ll_star, lp_star
                end
                return false, true, theta_j, ll_j, lp_j
            end
        end

        _with_single_blas_thread() do
            _parallel_for_worksteal(N; n_workers=nw_mut) do j, w
                ws = pool[w]
                ws_screen = delayed_acceptance ? screen_pool[w] : ws
                thread_rng = Random.Xoshiro(particle_seeds_mut[j])

                theta_j = state.theta_particles[:, j]
                ll_j = state.log_likelihoods[j]   # incumbent unbiased PF estimate (PMMH)
                lp_j = state.log_priors[j]
                acc_local = 0
                prop_local = 0
                s2_local = 0

                for _ in 1:n_mh_steps
                    if use_blocking
                        for b in eachindex(blocks)
                            I_b = blocks[b]
                            L_b = block_Ls[b]
                            z_b = randn(thread_rng, T, length(I_b))
                            step_b = L_b * z_b
                            theta_star = copy(theta_j)
                            @inbounds for m in eachindex(I_b)
                                theta_star[I_b[m]] += step_b[m]
                            end
                            accepted, reached_s2, theta_j, ll_j, lp_j = _eval_smc2_proposal(
                                theta_star, theta_j, ll_j, lp_j, j, ws, ws_screen, thread_rng)
                            accepted && (acc_local += 1)
                            reached_s2 && (s2_local += 1)
                            prop_local += 1
                        end
                    else
                        # Propose: theta_star = theta_j + L * z (symmetric RW ⇒ q-terms cancel)
                        z = randn(thread_rng, T, n_params)
                        theta_star = theta_j + L * z
                        accepted, reached_s2, theta_j, ll_j, lp_j = _eval_smc2_proposal(
                            theta_star, theta_j, ll_j, lp_j, j, ws, ws_screen, thread_rng)
                        accepted && (acc_local += 1)
                        reached_s2 && (s2_local += 1)
                        prop_local += 1
                    end
                end

                # Write back
                state.theta_particles[:, j] = theta_j
                state.log_likelihoods[j] = ll_j
                state.log_priors[j] = lp_j
                worker_acc[w] += acc_local
                worker_prop[w] += prop_local
                worker_s2_prop[w] += s2_local
            end
        end

        # Record acceptance rate (reduced once from per-worker counters)
        tp = sum(worker_prop)
        tot_acc = sum(worker_acc)
        acc_rate = tp > 0 ? T(tot_acc) / T(tp) : zero(T)
        push!(state.acceptance_rates, acc_rate)

        # Adapt proposal scale c_n toward target ~25% acceptance (Herbst & Schorfheide 2014; #951)
        _adapt_proposal_scale!(state, acc_rate)

        # Adapt N_x on the PF ESTIMATOR variance at fixed θ (Chopin et al. 2013 §3.4).
        # Under delayed acceptance, gate on Stage-2 conditional acceptance as well so Stage-1
        # screening rejections don't spuriously re-probe N_x every stage; if N_x doubles,
        # resize screen_pool (when auto-sized) and single-solve refresh ll_cheap in _exchange_step!.
        s2_prop = sum(worker_s2_prop)
        acc_s2 = s2_prop > 0 ? T(tot_acc) / T(s2_prop) : zero(T)
        should_probe_nx = delayed_acceptance ? (acc_rate < T(0.10) && acc_s2 < T(0.15)) : (acc_rate < T(0.15))
        if should_probe_nx
            est_var = _pf_estimator_variance(spec, param_names, state.theta_particles,
                observables, measurement_error, solver, eff_solver_kwargs, pool, data, T_obs,
                rng; n_probe=min(N, 8), n_rep=2)   # target estimator variance ≈ 1–3
            N_x_new = _adapt_n_particles(N_x, est_var, T(3))
            if N_x_new > N_x
                N_x = N_x_new
                for ws in pool
                    _resize_pf_workspace!(ws, N_x)
                end
                if delayed_acceptance && n_screen === nothing
                    n_screen_eff = _resolve_n_screen(nothing, N_x)
                    for ws_scr in screen_pool
                        _resize_pf_workspace!(ws_scr, n_screen_eff)
                    end
                end
                # Exchange step: recompute ALL θ-likelihoods at the new N_x with exact Chopin (2013) weights (#950).
                _exchange_step!(state, spec, param_names, observables, measurement_error,
                    solver, eff_solver_kwargs, pool, data, T_obs, rng;
                    phi=phi_new, ess_target=ess_target,
                    solutions=solutions,
                    ll_cheap=(delayed_acceptance ? ll_cheap : nothing),
                    screen_pool=screen_pool,
                    failures=lik_failures, evals=lik_evals)
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
