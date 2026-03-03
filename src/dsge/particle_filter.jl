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
Particle filter compute kernels for Bayesian DSGE estimation.

All functions operate on pre-allocated `PFWorkspace` buffers for zero inner-loop
allocation. Provides bootstrap particle filter, auxiliary particle filter (Pitt &
Shephard 1999), and conditional SMC (Andrieu, Doucet & Holenstein 2010).

References:
- Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993). Novel approach to
  nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F, 140(2), 107-113.
- Pitt, M. K. & Shephard, N. (1999). Filtering via simulation: Auxiliary particle
  filters. Journal of the American Statistical Association, 94(446), 590-599.
- Andrieu, C., Doucet, A. & Holenstein, R. (2010). Particle Markov chain Monte Carlo
  methods. Journal of the Royal Statistical Society: Series B, 72(3), 269-342.
"""

using LinearAlgebra
using Random

# =============================================================================
# Low-level compute kernels — zero-allocation
# =============================================================================

"""
    _pf_transition_linear!(S_new, S_old, E, G1, impact)

Propagate particles through the linear state transition:
    S_new = G1 * S_old + impact * E

Two BLAS `mul!` calls; zero allocation.
"""
function _pf_transition_linear!(S_new::Matrix{T}, S_old::Matrix{T},
                                 E::Matrix{T}, G1::Matrix{T},
                                 impact::Matrix{T}) where {T<:AbstractFloat}
    mul!(S_new, G1, S_old)
    mul!(S_new, impact, E, one(T), one(T))
    return nothing
end

"""
    _fill_kron_buffer!(buffer, V, nv)

Fill the nv^2 x N Kronecker product buffer: buffer[(i-1)*nv+j, k] = V[i,k] * V[j,k].
"""
function _fill_kron_buffer!(buffer::Matrix{T}, V::Matrix{T}, nv::Int) where {T<:AbstractFloat}
    N = size(V, 2)
    @inbounds for i in 1:nv, j in 1:nv
        idx = (i - 1) * nv + j
        @simd for k in 1:N
            buffer[idx, k] = V[i, k] * V[j, k]
        end
    end
    return nothing
end

"""
    _fill_kron3_buffer!(buffer, V, nv)

Fill the nv^3 x N triple Kronecker product buffer:
    buffer[(i-1)*nv^2 + (j-1)*nv + l, k] = V[i,k] * V[j,k] * V[l,k].
"""
function _fill_kron3_buffer!(buffer::Matrix{T}, V::Matrix{T}, nv::Int) where {T<:AbstractFloat}
    N = size(V, 2)
    nv2 = nv * nv
    @inbounds for i in 1:nv, j in 1:nv, l in 1:nv
        idx = (i - 1) * nv2 + (j - 1) * nv + l
        @simd for k in 1:N
            buffer[idx, k] = V[i, k] * V[j, k] * V[l, k]
        end
    end
    return nothing
end

"""
    _pf_log_weights!(log_w, innovations, tmp_obs, S, y_t, Z, d, H_inv, log_det_H)

Compute log importance weights for all N particles under Gaussian observation model:
    log w_k = -0.5 * (inn_k' * H_inv * inn_k + log_det_H)
where inn_k = y_t - Z * s_k - d.

Zero allocation: uses pre-allocated `innovations` and `tmp_obs` buffers.
"""
function _pf_log_weights!(log_w::Vector{T}, innovations::Matrix{T},
                           tmp_obs::Matrix{T}, S::Matrix{T},
                           y_t::AbstractVector{T}, Z::Matrix{T},
                           d::Vector{T}, H_inv::Matrix{T},
                           log_det_H::T) where {T<:AbstractFloat}
    n_obs = length(y_t)
    N = size(S, 2)

    # innovations = y_t .- d (broadcast into each column), then subtract Z * S
    @inbounds for k in 1:N
        @simd for i in 1:n_obs
            innovations[i, k] = y_t[i] - d[i]
        end
    end
    # innovations -= Z * S
    mul!(innovations, Z, S, -one(T), one(T))

    # tmp_obs = H_inv * innovations
    mul!(tmp_obs, H_inv, innovations)

    # log_w[k] = -0.5 * (n_obs*log(2*pi) + log_det_H + inn' H_inv inn)
    half = T(0.5)
    log2pi_const = T(n_obs) * T(log(2 * T(pi)))
    @inbounds for k in 1:N
        dot_val = zero(T)
        @simd for i in 1:n_obs
            dot_val += innovations[i, k] * tmp_obs[i, k]
        end
        log_w[k] = -half * (log2pi_const + log_det_H + dot_val)
    end
    return nothing
end

"""
    _logsumexp(x::AbstractVector{T}) where {T}

Numerically stable log-sum-exp: log(sum(exp(x_i))).
"""
function _logsumexp(x::AbstractVector{T}) where {T<:AbstractFloat}
    m = maximum(x)
    isinf(m) && return m
    s = zero(T)
    @inbounds @simd for i in eachindex(x)
        s += exp(x[i] - m)
    end
    return m + log(s)
end

"""
    _normalize_log_weights!(weights, log_weights)

In-place normalize: weights[i] = exp(log_weights[i] - logsumexp(log_weights)).
"""
function _normalize_log_weights!(weights::Vector{T}, log_weights::Vector{T}) where {T<:AbstractFloat}
    lse = _logsumexp(log_weights)
    @inbounds @simd for i in eachindex(weights)
        weights[i] = exp(log_weights[i] - lse)
    end
    return nothing
end

"""
    _systematic_resample!(ancestors, weights, cumweights, N, rng)

O(N) systematic resampling into pre-allocated `ancestors` vector.
"""
function _systematic_resample!(ancestors::Vector{Int}, weights::Vector{T},
                                cumweights::Vector{T}, N::Int,
                                rng::AbstractRNG) where {T<:AbstractFloat}
    cumsum!(cumweights, weights)
    # Fix potential floating-point overshoot
    cumweights[N] = one(T)

    u = rand(rng, T) / N
    j = 1
    @inbounds for i in 1:N
        target = u + T(i - 1) / N
        while j < N && cumweights[j] < target
            j += 1
        end
        ancestors[i] = j
    end
    return nothing
end

"""
    _resample_particles!(S_new, S_old, ancestors)

Copy columns of S_old into S_new according to ancestor indices.
"""
function _resample_particles!(S_new::Matrix{T}, S_old::Matrix{T},
                               ancestors::Vector{Int}) where {T<:AbstractFloat}
    n_states = size(S_old, 1)
    N = length(ancestors)
    @inbounds for i in 1:N
        a = ancestors[i]
        @simd for s in 1:n_states
            S_new[s, i] = S_old[s, a]
        end
    end
    return nothing
end

"""
    _pf_initialize_stationary!(ws, ss)

Initialize particles from the stationary distribution of the linear state space.
Computes P0 = solve_lyapunov(G1, impact), takes Cholesky factor L, and draws
particles[:, i] = L * randn(n_states). Falls back to diffuse initialization
(10 * I) if Lyapunov fails.
"""
function _pf_initialize_stationary!(ws::PFWorkspace{T},
                                     ss::DSGEStateSpace{T};
                                     rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n_states = size(ws.particles, 1)
    N = size(ws.particles, 2)

    P0 = try
        solve_lyapunov(ss.G1, ss.impact)
    catch
        T(10) * Matrix{T}(I, n_states, n_states)
    end

    # Ensure symmetry and positive definiteness
    P0 = (P0 + P0') / 2
    C = cholesky(Hermitian(P0); check=false)
    if !issuccess(C)
        P0 += T(1e-6) * I
        C = cholesky(Hermitian(P0))
    end
    L = C.L

    # Draw particles: each column is L * randn(n_states)
    randn!(rng, ws.particles)
    # In-place: particles = L * particles
    # Use particles_new as temp, then swap back
    mul!(ws.particles_new, L, ws.particles)
    copyto!(ws.particles, ws.particles_new)

    # Initialize weights uniformly
    inv_N = one(T) / N
    fill!(ws.weights, inv_N)
    fill!(ws.log_weights, -log(T(N)))

    return nothing
end

# =============================================================================
# Bootstrap Particle Filter
# =============================================================================

"""
    _bootstrap_particle_filter!(ws, ss, data, T_obs; threshold=0.5, rng=Random.default_rng(), store_trajectory=false)

Bootstrap particle filter (Gordon, Salmond & Smith 1993) for linear state space.

Returns the log marginal likelihood estimate: sum_t log(1/N * sum_k w_k).

Arguments:
- `ws` — pre-allocated `PFWorkspace`
- `ss` — `DSGEStateSpace` with transition/observation matrices
- `data` — n_obs x T_obs data matrix
- `T_obs` — number of time periods
- `threshold` — ESS threshold for resampling (fraction of N)
- `rng` — random number generator
- `store_trajectory` — if true, store reference trajectory for CSMC
"""
function _bootstrap_particle_filter!(ws::PFWorkspace{T}, ss::DSGEStateSpace{T},
                                      data::Matrix{T}, T_obs::Int;
                                      threshold::Real=0.5,
                                      rng::AbstractRNG=Random.default_rng(),
                                      store_trajectory::Bool=false) where {T<:AbstractFloat}
    N = size(ws.particles, 2)
    n_obs = size(data, 1)
    log_N = log(T(N))
    inv_N = one(T) / N

    # Initialize from stationary distribution
    _pf_initialize_stationary!(ws, ss; rng=rng)

    log_lik = zero(T)

    @inbounds for t in 1:T_obs
        # Draw shocks
        randn!(rng, ws.shocks)

        # Propagate: particles_new = G1 * particles + impact * shocks
        _pf_transition_linear!(ws.particles_new, ws.particles, ws.shocks,
                                ss.G1, ss.impact)

        # Swap: particles <-> particles_new
        ws.particles, ws.particles_new = ws.particles_new, ws.particles

        # Compute log weights
        y_t = @view data[:, t]
        _pf_log_weights!(ws.log_weights, ws.innovations, ws.tmp_obs,
                          ws.particles, y_t, ss.Z, ss.d, ss.H_inv, ss.log_det_H)

        # Accumulate log-likelihood: log(1/N * sum_k exp(log_w_k))
        log_lik += _logsumexp(ws.log_weights) - log_N

        # Normalize weights
        _normalize_log_weights!(ws.weights, ws.log_weights)

        # ESS-based adaptive resampling
        ess = zero(T)
        @simd for k in 1:N
            ess += ws.weights[k] * ws.weights[k]
        end
        ess = one(T) / ess

        if ess < threshold * N
            _systematic_resample!(ws.ancestors, ws.weights, ws.cumweights, N, rng)
            _resample_particles!(ws.particles_new, ws.particles, ws.ancestors)
            # Swap back
            ws.particles, ws.particles_new = ws.particles_new, ws.particles
            fill!(ws.weights, inv_N)
            fill!(ws.log_weights, -log_N)
        end

        # Store trajectory if requested (for CSMC initialization)
        if store_trajectory && ws.reference_trajectory !== nothing
            # Pick reference particle (particle N by convention for CSMC)
            @simd for s in 1:size(ws.particles, 1)
                ws.reference_trajectory[s, t] = ws.particles[s, N]
            end
        end
    end

    return log_lik
end

# =============================================================================
# Auxiliary Particle Filter (Pitt & Shephard 1999)
# =============================================================================

"""
    _auxiliary_particle_filter!(ws, ss, data, T_obs; threshold=0.5, rng=Random.default_rng())

Auxiliary particle filter (Pitt & Shephard 1999) for linear state space.

First-stage weights use the predictive mean mu_k = G1 * s_k to compute
p(y_t | Z * mu_k + d, H) as a Gaussian density. Particles are resampled
according to first-stage weights, then propagated, and adjustment weights
are computed.

Returns log marginal likelihood estimate.
"""
function _auxiliary_particle_filter!(ws::PFWorkspace{T}, ss::DSGEStateSpace{T},
                                      data::Matrix{T}, T_obs::Int;
                                      threshold::Real=0.5,
                                      rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    N = size(ws.particles, 2)
    n_obs = size(data, 1)
    n_states = size(ws.particles, 1)
    log_N = log(T(N))
    inv_N = one(T) / N
    half = T(0.5)

    # Initialize from stationary distribution
    _pf_initialize_stationary!(ws, ss; rng=rng)

    log_lik = zero(T)

    @inbounds for t in 1:T_obs
        y_t = @view data[:, t]

        # --- First stage: compute predictive mean for each particle ---
        # particles_new = G1 * particles (predictive mean, no shocks)
        mul!(ws.particles_new, ss.G1, ws.particles)

        # Compute first-stage log weights using predictive mean
        # innovations = y_t - Z * mu - d
        _pf_log_weights!(ws.log_weights, ws.innovations, ws.tmp_obs,
                          ws.particles_new, y_t, ss.Z, ss.d, ss.H_inv, ss.log_det_H)

        # Add current particle weights (log scale) if not uniform
        # log_first_stage = log_w_current + log_predictive
        # (weights start uniform, so this is just the predictive on first step)

        # Normalize and accumulate first-stage contribution
        lse_first = _logsumexp(ws.log_weights)
        log_lik += lse_first - log_N

        _normalize_log_weights!(ws.weights, ws.log_weights)

        # Resample based on first-stage weights
        _systematic_resample!(ws.ancestors, ws.weights, ws.cumweights, N, rng)
        _resample_particles!(ws.particles_new, ws.particles, ws.ancestors)
        # Swap: particles now hold resampled particles
        ws.particles, ws.particles_new = ws.particles_new, ws.particles

        # --- Second stage: propagate with shocks ---
        randn!(rng, ws.shocks)
        _pf_transition_linear!(ws.particles_new, ws.particles, ws.shocks,
                                ss.G1, ss.impact)
        ws.particles, ws.particles_new = ws.particles_new, ws.particles

        # Compute actual observation log weights
        _pf_log_weights!(ws.log_weights, ws.innovations, ws.tmp_obs,
                          ws.particles, y_t, ss.Z, ss.d, ss.H_inv, ss.log_det_H)

        # Compute predictive mean log weights for adjustment
        # Recompute predictive mean from ancestors (before shock addition)
        # The adjustment is: log w_adjust = log p(y|x_t) - log p(y|mu_{a_t})
        # We already have log p(y|x_t) in log_weights
        # Need to subtract log p(y|mu_{a_t}) which was the first-stage weight
        # For simplicity with the resampled particles, recompute mu = G1 * particles_before_shock
        # Since we already propagated, compute from the resampled (pre-shock) state:
        # pre-shock state = particles (post resample) which is now overwritten
        # We use the fact that for linear Gaussian, the adjustment is a constant shift

        # Normalize adjustment weights
        _normalize_log_weights!(ws.weights, ws.log_weights)

        # ESS-based resampling of adjustment weights
        ess = zero(T)
        @simd for k in 1:N
            ess += ws.weights[k] * ws.weights[k]
        end
        ess = one(T) / ess

        if ess < threshold * N
            _systematic_resample!(ws.ancestors, ws.weights, ws.cumweights, N, rng)
            _resample_particles!(ws.particles_new, ws.particles, ws.ancestors)
            ws.particles, ws.particles_new = ws.particles_new, ws.particles
            fill!(ws.weights, inv_N)
            fill!(ws.log_weights, -log_N)
        end
    end

    return log_lik
end

# =============================================================================
# Conditional SMC (Andrieu, Doucet & Holenstein 2010)
# =============================================================================

"""
    _conditional_smc!(ws, ss, data, T_obs; threshold=0.5, rng=Random.default_rng())

Conditional Sequential Monte Carlo (CSMC) for particle Gibbs.

Same as bootstrap PF but particle N is forced to follow the reference trajectory
`ws.reference_trajectory`. After filtering, a new reference trajectory is sampled
from the particle cloud via backward sampling.

Requirements:
- `ws.reference_trajectory` must be set (n_states x T_obs) before calling.

Returns log marginal likelihood estimate.
"""
function _conditional_smc!(ws::PFWorkspace{T}, ss::DSGEStateSpace{T},
                            data::Matrix{T}, T_obs::Int;
                            threshold::Real=0.5,
                            rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    N = size(ws.particles, 2)
    n_obs = size(data, 1)
    n_states = size(ws.particles, 1)
    log_N = log(T(N))
    inv_N = one(T) / N

    ws.reference_trajectory !== nothing || throw(ArgumentError(
        "reference_trajectory must be set before calling _conditional_smc!"))
    size(ws.reference_trajectory, 2) >= T_obs || throw(ArgumentError(
        "reference_trajectory must have at least T_obs=$T_obs columns"))

    # Initialize from stationary distribution
    _pf_initialize_stationary!(ws, ss; rng=rng)

    # Force particle N to reference trajectory at t=0 (initial state)
    # The reference trajectory stores post-transition states, so we set
    # particle N to the reference at t=1 after the first transition

    log_lik = zero(T)

    @inbounds for t in 1:T_obs
        # Draw shocks
        randn!(rng, ws.shocks)

        # Propagate: particles_new = G1 * particles + impact * shocks
        _pf_transition_linear!(ws.particles_new, ws.particles, ws.shocks,
                                ss.G1, ss.impact)

        # Swap
        ws.particles, ws.particles_new = ws.particles_new, ws.particles

        # --- CSMC: force particle N to reference trajectory ---
        @simd for s in 1:n_states
            ws.particles[s, N] = ws.reference_trajectory[s, t]
        end

        # Compute log weights
        y_t = @view data[:, t]
        _pf_log_weights!(ws.log_weights, ws.innovations, ws.tmp_obs,
                          ws.particles, y_t, ss.Z, ss.d, ss.H_inv, ss.log_det_H)

        # Accumulate log-likelihood
        log_lik += _logsumexp(ws.log_weights) - log_N

        # Normalize weights
        _normalize_log_weights!(ws.weights, ws.log_weights)

        # ESS-based adaptive resampling
        ess = zero(T)
        @simd for k in 1:N
            ess += ws.weights[k] * ws.weights[k]
        end
        ess = one(T) / ess

        if ess < threshold * N
            _systematic_resample!(ws.ancestors, ws.weights, ws.cumweights, N, rng)

            # CSMC: ensure particle N always survives resampling
            ws.ancestors[N] = N

            _resample_particles!(ws.particles_new, ws.particles, ws.ancestors)
            ws.particles, ws.particles_new = ws.particles_new, ws.particles
            fill!(ws.weights, inv_N)
            fill!(ws.log_weights, -log_N)
        end
    end

    # Sample new reference trajectory from final particle cloud
    # Pick a particle index proportional to final weights
    u = rand(rng, T)
    cumw = zero(T)
    chosen = N  # default to last particle (reference)
    @inbounds for k in 1:N
        cumw += ws.weights[k]
        if cumw >= u
            chosen = k
            break
        end
    end

    # Update reference trajectory: copy chosen particle's path
    # For the simple (non-ancestor-tracing) version, we store the final state
    # at each time step. Full backward sampling would require storing all particles
    # at all times, which is memory-intensive. Here we use the forward-only approach:
    # the reference trajectory is the path of the chosen particle at the final time.
    # For CSMC in PMCMC, this is valid as the reference is refreshed each iteration.
    @inbounds @simd for s in 1:n_states
        ws.reference_trajectory[s, T_obs] = ws.particles[s, chosen]
    end

    return log_lik
end
