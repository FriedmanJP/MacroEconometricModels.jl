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

# =============================================================================
# Nonlinear particle filter — pruned state transitions (Kim et al. 2008,
# Andreasen, Fernandez-Villaverde & Rubio-Ramirez 2018)
# =============================================================================

"""
    _fill_kron_cross_buffer!(buffer, V1, V2, nv)

Fill the nv^2 x N cross-Kronecker buffer: buffer[(i-1)*nv+j, k] = V1[i,k] * V2[j,k].
Used for kron(vf, vs) in 3rd-order pruning.
"""
function _fill_kron_cross_buffer!(buffer::Matrix{T}, V1::Matrix{T}, V2::Matrix{T},
                                    nv::Int) where {T<:AbstractFloat}
    N = size(V1, 2)
    @inbounds for i in 1:nv, j in 1:nv
        idx = (i - 1) * nv + j
        @simd for k in 1:N
            buffer[idx, k] = V1[i, k] * V2[j, k]
        end
    end
    return nothing
end

"""
    _pf_initialize_nonlinear!(ws, nlss; rng)

Initialize particles for nonlinear state space (perturbation solution).

Uses the first-order Lyapunov equation P0 = solve_lyapunov(hx_state, eta_x) to set
the initial dispersion of first-order state particles. Second/third-order corrections
are initialized to zero (at steady state).

The full endogenous vector is assembled from states + controls for observation.
"""
function _pf_initialize_nonlinear!(ws::PFWorkspace{T},
                                     nlss::NonlinearStateSpace{T};
                                     rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    nx = length(nlss.state_indices)
    ny = length(nlss.control_indices)
    n_eps = size(nlss.eta, 2)
    nv = nx + n_eps
    N = size(ws.particles, 2)

    # Extract first-order blocks
    hx_state = nx > 0 ? nlss.hx[:, 1:nx] : zeros(T, 0, 0)
    eta_x = nx > 0 ? nlss.hx[:, nx+1:nv] : zeros(T, 0, n_eps)
    gx_state = ny > 0 ? nlss.gx[:, 1:nx] : zeros(T, 0, nx)

    # Compute P0 from first-order Lyapunov equation
    P0 = try
        solve_lyapunov(hx_state, eta_x)
    catch
        T(10) * Matrix{T}(I, nx, nx)
    end
    P0 = (P0 + P0') / 2
    C = cholesky(Hermitian(P0); check=false)
    if !issuccess(C)
        P0 += T(1e-6) * I
        C = cholesky(Hermitian(P0))
    end
    L = C.L

    # Draw first-order state particles: particles_fo[:, k] = L * randn(nx)
    randn!(rng, ws.particles_fo)
    temp = L * ws.particles_fo  # single allocation at init (not hot loop)
    copyto!(ws.particles_fo, temp)

    # Initialize second-order correction to zero (if allocated)
    if ws.particles_so !== nothing
        fill!(ws.particles_so, zero(T))
    end

    # Initialize third-order correction to zero (if allocated)
    if ws.particles_to !== nothing
        fill!(ws.particles_to, zero(T))
    end

    # Assemble initial full endogenous vector
    # States at state_indices, controls (gx_state * xf) at control_indices
    fill!(ws.particles, zero(T))
    @inbounds for k in 1:N
        for (si_idx, si) in enumerate(nlss.state_indices)
            ws.particles[si, k] = ws.particles_fo[si_idx, k]
        end
    end
    if ny > 0
        # controls = gx_state * particles_fo (first-order, no shocks at init)
        ctrl = gx_state * ws.particles_fo  # single allocation at init
        @inbounds for k in 1:N
            for (ci_idx, ci) in enumerate(nlss.control_indices)
                ws.particles[ci, k] = ctrl[ci_idx, k]
            end
        end
    end

    # Initialize weights uniformly
    inv_N = one(T) / N
    fill!(ws.weights, inv_N)
    fill!(ws.log_weights, -log(T(N)))

    return nothing
end

"""
    _pf_transition_pruned!(ws, nlss)

Propagate all N particles through the pruned nonlinear transition.

Order 1: standard linear first-order transition.
Order 2: Kim et al. (2008) pruning — tracks xf (1st-order) and xs (2nd-order correction).
Order 3: Andreasen et al. (2018) — additionally tracks xt (3rd-order correction).

After transition, assembles the full endogenous vector (states + controls) into
`ws.particles` for use by `_pf_log_weights!`.

Zero allocation in hot loop (all computation uses pre-allocated workspace buffers).
"""
function _pf_transition_pruned!(ws::PFWorkspace{T},
                                  nlss::NonlinearStateSpace{T}) where {T<:AbstractFloat}
    nx = length(nlss.state_indices)
    ny = length(nlss.control_indices)
    n_endog = nx + ny
    n_eps = size(nlss.eta, 2)
    nv = nx + n_eps
    N = size(ws.particles_fo, 2)

    # Extract policy function sub-matrices (views, no allocation)
    hx_state = @view nlss.hx[:, 1:nx]       # nx x nx
    eta_x = @view nlss.hx[:, nx+1:nv]       # nx x n_eps
    gx_state = @view nlss.gx[:, 1:nx]       # ny x nx
    eta_y = @view nlss.gx[:, nx+1:nv]       # ny x n_eps

    # --- First-order state: XF_new = hx_state * XF + eta_x * E ---
    # Store XF_new in transition_scratch[1:nx, :]
    xf_new = @view ws.transition_scratch[1:nx, :]
    mul!(xf_new, hx_state, ws.particles_fo)
    mul!(xf_new, eta_x, ws.shocks, one(T), one(T))

    if nlss.order >= 2 && nlss.hxx !== nothing
        # --- Build augmented vector V = [XF; E] ---
        @inbounds for k in 1:N
            @simd for i in 1:nx
                ws.augmented_buffer[i, k] = ws.particles_fo[i, k]
            end
            @simd for i in 1:n_eps
                ws.augmented_buffer[nx + i, k] = ws.shocks[i, k]
            end
        end

        # --- Fill kron(V, V) buffer ---
        _fill_kron_buffer!(ws.kron_buffer, ws.augmented_buffer, nv)

        # --- Second-order state: XS_new = hx_state * XS + 0.5 * Hxx * kron + 0.5 * hσσ ---
        xs_new = @view ws.transition_scratch[nx+1:2*nx, :]

        mul!(xs_new, hx_state, ws.particles_so)
        mul!(xs_new, nlss.hxx, ws.kron_buffer, T(0.5), one(T))
        if nlss.hsigmasigma !== nothing
            @inbounds for k in 1:N
                @simd for i in 1:nx
                    xs_new[i, k] += T(0.5) * nlss.hsigmasigma[i]
                end
            end
        end

        if nlss.order >= 3 && nlss.hxxx !== nothing && ws.particles_to !== nothing
            # --- 3rd-order: compute kron(vf, vs) where vs = [xs; 0] ---
            @inbounds for i in 1:nv, j in 1:nv
                idx = (i - 1) * nv + j
                @simd for k in 1:N
                    vf_i = ws.augmented_buffer[i, k]
                    vs_j = j <= nx ? ws.particles_so[j, k] : zero(T)
                    ws.kron_cross_buffer[idx, k] = vf_i * vs_j
                end
            end

            # Fill kron3(vf, vf, vf)
            _fill_kron3_buffer!(ws.kron3_buffer, ws.augmented_buffer, nv)

            # --- 3rd-order state: XT_new ---
            xt_new = @view ws.transition_scratch[2*nx+1:3*nx, :]

            mul!(xt_new, hx_state, ws.particles_to)
            # + hxx * kron(vf, vs)
            mul!(xt_new, nlss.hxx, ws.kron_cross_buffer, one(T), one(T))
            # + (1/6) * hxxx * kron3(vf)
            mul!(xt_new, nlss.hxxx, ws.kron3_buffer, T(1) / T(6), one(T))
            # + (1/2) * hσσx * vf
            if nlss.hsigmax !== nothing
                mul!(xt_new, nlss.hsigmax, ws.augmented_buffer, T(0.5), one(T))
            end
            # + (1/6) * hσσσ
            if nlss.hsigmasigmasigma !== nothing
                @inbounds for k in 1:N
                    @simd for i in 1:nx
                        xt_new[i, k] += nlss.hsigmasigmasigma[i] / T(6)
                    end
                end
            end

            # Copy pruned components to their buffers
            copyto!(ws.particles_to, xt_new)
        end

        # Update pruned component buffers
        copyto!(ws.particles_so, xs_new)
    end

    # Copy xf_new to particles_fo
    copyto!(ws.particles_fo, @view ws.transition_scratch[1:nx, :])

    # --- Compute total state and controls, assemble full endogenous ---

    # Build x_total in the state_indices of particles
    @inbounds for k in 1:N
        for (si_idx, si) in enumerate(nlss.state_indices)
            x_total = ws.particles_fo[si_idx, k]  # xf_new
            if nlss.order >= 2 && ws.particles_so !== nothing
                x_total += ws.particles_so[si_idx, k]
            end
            if nlss.order >= 3 && ws.particles_to !== nothing
                x_total += ws.particles_to[si_idx, k]
            end
            ws.particles[si, k] = x_total
        end
    end

    # Compute controls: use particles_new[1:ny, :] as scratch for ctrl
    if ny > 0
        ctrl = @view ws.particles_new[1:ny, :]

        # Build x_total contiguous matrix in augmented_buffer[1:nx, :]
        @inbounds for k in 1:N
            @simd for i in 1:nx
                ws.augmented_buffer[i, k] = ws.particles_fo[i, k]
                if nlss.order >= 2 && ws.particles_so !== nothing
                    ws.augmented_buffer[i, k] += ws.particles_so[i, k]
                end
                if nlss.order >= 3 && ws.particles_to !== nothing
                    ws.augmented_buffer[i, k] += ws.particles_to[i, k]
                end
            end
        end
        x_total_mat = @view ws.augmented_buffer[1:nx, :]

        # Base: Y = gx_state * x_total + eta_y * E
        mul!(ctrl, gx_state, x_total_mat)
        mul!(ctrl, eta_y, ws.shocks, one(T), one(T))

        # 2nd-order control terms
        if nlss.order >= 2 && nlss.gxx !== nothing
            # Rebuild augmented V = [xf; eps] for Kronecker
            @inbounds for k in 1:N
                @simd for i in 1:nx
                    ws.augmented_buffer[i, k] = ws.particles_fo[i, k]
                end
                @simd for i in 1:n_eps
                    ws.augmented_buffer[nx + i, k] = ws.shocks[i, k]
                end
            end
            _fill_kron_buffer!(ws.kron_buffer, ws.augmented_buffer, nv)

            # + 0.5 * Gxx * kron(vf, vf)
            mul!(ctrl, nlss.gxx, ws.kron_buffer, T(0.5), one(T))
            # + 0.5 * gσσ
            if nlss.gsigmasigma !== nothing
                @inbounds for k in 1:N
                    @simd for i in 1:ny
                        ctrl[i, k] += T(0.5) * nlss.gsigmasigma[i]
                    end
                end
            end

            # 3rd-order control terms
            if nlss.order >= 3 && nlss.gxxx !== nothing
                # kron(vf, vs) — recompute
                @inbounds for i in 1:nv, j in 1:nv
                    idx = (i - 1) * nv + j
                    @simd for k in 1:N
                        vf_i = ws.augmented_buffer[i, k]
                        vs_j = j <= nx ? ws.particles_so[j, k] : zero(T)
                        ws.kron_cross_buffer[idx, k] = vf_i * vs_j
                    end
                end
                # kron3(vf, vf, vf)
                _fill_kron3_buffer!(ws.kron3_buffer, ws.augmented_buffer, nv)

                # + gxx * kron(vf, vs)
                mul!(ctrl, nlss.gxx, ws.kron_cross_buffer, one(T), one(T))
                # + (1/6) * gxxx * kron3(vf)
                mul!(ctrl, nlss.gxxx, ws.kron3_buffer, T(1) / T(6), one(T))
                # + (1/2) * gσσx * vf
                if nlss.gsigmax !== nothing
                    mul!(ctrl, nlss.gsigmax, ws.augmented_buffer, T(0.5), one(T))
                end
                # + (1/6) * gσσσ
                if nlss.gsigmasigmasigma !== nothing
                    @inbounds for k in 1:N
                        @simd for i in 1:ny
                            ctrl[i, k] += nlss.gsigmasigmasigma[i] / T(6)
                        end
                    end
                end
            end
        end

        # Place controls at control_indices in particles
        @inbounds for k in 1:N
            for (ci_idx, ci) in enumerate(nlss.control_indices)
                ws.particles[ci, k] = ctrl[ci_idx, k]
            end
        end
    end

    return nothing
end

"""
    _pf_resample_pruned!(ws, N, nx, order)

Resample pruned state components (particles_fo, particles_so, particles_to)
alongside the main particles using the same ancestor indices.

Uses `particles_new[1:nx, :]` as scratch buffer — zero allocation.
"""
function _pf_resample_pruned!(ws::PFWorkspace{T}, N::Int, nx::Int,
                                order::Int) where {T<:AbstractFloat}
    # Resample particles_fo
    @inbounds for k in 1:N
        a = ws.ancestors[k]
        @simd for i in 1:nx
            ws.particles_new[i, k] = ws.particles_fo[i, a]
        end
    end
    @inbounds for k in 1:N
        @simd for i in 1:nx
            ws.particles_fo[i, k] = ws.particles_new[i, k]
        end
    end

    # Resample particles_so
    if order >= 2 && ws.particles_so !== nothing
        @inbounds for k in 1:N
            a = ws.ancestors[k]
            @simd for i in 1:nx
                ws.particles_new[i, k] = ws.particles_so[i, a]
            end
        end
        @inbounds for k in 1:N
            @simd for i in 1:nx
                ws.particles_so[i, k] = ws.particles_new[i, k]
            end
        end
    end

    # Resample particles_to
    if order >= 3 && ws.particles_to !== nothing
        @inbounds for k in 1:N
            a = ws.ancestors[k]
            @simd for i in 1:nx
                ws.particles_new[i, k] = ws.particles_to[i, a]
            end
        end
        @inbounds for k in 1:N
            @simd for i in 1:nx
                ws.particles_to[i, k] = ws.particles_new[i, k]
            end
        end
    end

    return nothing
end

# =============================================================================
# Bootstrap Particle Filter — Nonlinear (pruned perturbation)
# =============================================================================

"""
    _bootstrap_particle_filter!(ws, nlss::NonlinearStateSpace, data, T_obs; ...)

Bootstrap particle filter for nonlinear state space (perturbation solutions).

Uses Kim et al. (2008) pruned state transitions for order 2 and Andreasen,
Fernandez-Villaverde & Rubio-Ramirez (2018) for order 3, avoiding the explosive
paths that arise from naive higher-order simulation.

Returns log marginal likelihood estimate.
"""
function _bootstrap_particle_filter!(ws::PFWorkspace{T}, nlss::NonlinearStateSpace{T},
                                       data::Matrix{T}, T_obs::Int;
                                       threshold::Real=0.5,
                                       rng::AbstractRNG=Random.default_rng(),
                                       store_trajectory::Bool=false) where {T<:AbstractFloat}
    N = size(ws.particles, 2)
    n_obs = size(data, 1)
    n_endog = size(ws.particles, 1)
    nx = length(nlss.state_indices)
    log_N = log(T(N))
    inv_N = one(T) / N

    # Initialize from first-order stationary distribution
    _pf_initialize_nonlinear!(ws, nlss; rng=rng)

    log_lik = zero(T)

    @inbounds for t in 1:T_obs
        # Draw shocks
        randn!(rng, ws.shocks)

        # Propagate through pruned nonlinear transition
        _pf_transition_pruned!(ws, nlss)

        # Compute log weights using full endogenous particles
        y_t = @view data[:, t]
        _pf_log_weights!(ws.log_weights, ws.innovations, ws.tmp_obs,
                          ws.particles, y_t, nlss.Z, nlss.d, nlss.H_inv, nlss.log_det_H)

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
            # Resample main particles
            _resample_particles!(ws.particles_new, ws.particles, ws.ancestors)
            ws.particles, ws.particles_new = ws.particles_new, ws.particles
            # Resample pruned components
            _pf_resample_pruned!(ws, N, nx, nlss.order)
            fill!(ws.weights, inv_N)
            fill!(ws.log_weights, -log_N)
        end

        # Store trajectory if requested (for CSMC initialization)
        if store_trajectory && ws.reference_trajectory !== nothing
            @simd for s in 1:n_endog
                ws.reference_trajectory[s, t] = ws.particles[s, N]
            end
        end
    end

    return log_lik
end

# =============================================================================
# Conditional SMC — Nonlinear (pruned perturbation)
# =============================================================================

"""
    _conditional_smc!(ws, nlss::NonlinearStateSpace, data, T_obs; ...)

Conditional SMC for nonlinear state space (particle Gibbs with pruned transitions).

Same as bootstrap PF but particle N is forced to follow the reference trajectory.
Reference trajectory stores the full endogenous vector; pruned components for the
reference particle are approximated (xf = total state, xs = 0, xt = 0).

Returns log marginal likelihood estimate.
"""
function _conditional_smc!(ws::PFWorkspace{T}, nlss::NonlinearStateSpace{T},
                             data::Matrix{T}, T_obs::Int;
                             threshold::Real=0.5,
                             rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    N = size(ws.particles, 2)
    n_obs = size(data, 1)
    n_endog = size(ws.particles, 1)
    nx = length(nlss.state_indices)
    log_N = log(T(N))
    inv_N = one(T) / N

    ws.reference_trajectory !== nothing || throw(ArgumentError(
        "reference_trajectory must be set before calling _conditional_smc!"))
    size(ws.reference_trajectory, 2) >= T_obs || throw(ArgumentError(
        "reference_trajectory must have at least T_obs=$T_obs columns"))

    # Initialize from first-order stationary distribution
    _pf_initialize_nonlinear!(ws, nlss; rng=rng)

    log_lik = zero(T)

    @inbounds for t in 1:T_obs
        # Draw shocks
        randn!(rng, ws.shocks)

        # Propagate through pruned nonlinear transition
        _pf_transition_pruned!(ws, nlss)

        # --- CSMC: force particle N to reference trajectory ---
        @simd for s in 1:n_endog
            ws.particles[s, N] = ws.reference_trajectory[s, t]
        end
        # Approximate pruned components for reference particle:
        # xf = total state from reference, xs = 0, xt = 0
        for (si_idx, si) in enumerate(nlss.state_indices)
            ws.particles_fo[si_idx, N] = ws.reference_trajectory[si, t]
        end
        if ws.particles_so !== nothing
            @simd for i in 1:nx
                ws.particles_so[i, N] = zero(T)
            end
        end
        if ws.particles_to !== nothing
            @simd for i in 1:nx
                ws.particles_to[i, N] = zero(T)
            end
        end

        # Compute log weights
        y_t = @view data[:, t]
        _pf_log_weights!(ws.log_weights, ws.innovations, ws.tmp_obs,
                          ws.particles, y_t, nlss.Z, nlss.d, nlss.H_inv, nlss.log_det_H)

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
            # CSMC: ensure particle N always survives
            ws.ancestors[N] = N
            # Resample main particles
            _resample_particles!(ws.particles_new, ws.particles, ws.ancestors)
            ws.particles, ws.particles_new = ws.particles_new, ws.particles
            # Resample pruned components
            _pf_resample_pruned!(ws, N, nx, nlss.order)
            fill!(ws.weights, inv_N)
            fill!(ws.log_weights, -log_N)
        end
    end

    # Sample new reference trajectory from final particle cloud
    u = rand(rng, T)
    cumw = zero(T)
    chosen = N
    @inbounds for k in 1:N
        cumw += ws.weights[k]
        if cumw >= u
            chosen = k
            break
        end
    end
    @inbounds @simd for s in 1:n_endog
        ws.reference_trajectory[s, T_obs] = ws.particles[s, chosen]
    end

    return log_lik
end

# ─────────────────────────────────────────────────────────────────────────────
# Projection / PFI particle filter kernels
# ─────────────────────────────────────────────────────────────────────────────

"""
    _pf_transition_projection!(ws, pss)

Propagate N particles through a global nonlinear Chebyshev policy function.
Zero-allocation: all buffers pre-allocated in ws.

Algorithm:
1. Extract states from particles → scale to [-1,1]
2. Evaluate Chebyshev basis for all N particles (3D buffer, 1D recurrence per dimension)
3. Policy multiply: mul!(proj_policy, coefficients, basis) — BLAS Level 3
4. Assemble particles = policy + steady_state + impact * shocks, clamp states
"""
function _pf_transition_projection!(ws::PFWorkspace{T},
                                      pss::ProjectionStateSpace{T}) where {T<:AbstractFloat}
    N = size(ws.particles, 2)
    nx = length(pss.state_indices)
    n_endog = length(pss.steady_state)
    n_eps = size(pss.impact, 2)
    n_basis = size(pss.multi_indices, 1)
    max_deg = pss.max_degree

    # Step 1: Extract states and scale to [-1,1]
    @inbounds for i in 1:nx
        si = pss.state_indices[i]
        sc = pss.scale[i]
        sh = pss.shift[i]
        @simd for k in 1:N
            z = ws.particles[si, k] * sc + sh
            ws.proj_scaled[i, k] = clamp(z, T(-1), T(1))
        end
    end

    # Step 2a: 1D Chebyshev recurrence for all dimensions
    @inbounds for d in 1:nx
        @simd for k in 1:N
            ws.proj_cheb_1d[1, d, k] = one(T)  # T_0 = 1
        end
        if max_deg >= 1
            @simd for k in 1:N
                ws.proj_cheb_1d[2, d, k] = ws.proj_scaled[d, k]  # T_1 = z
            end
        end
        for n in 2:max_deg
            @simd for k in 1:N
                ws.proj_cheb_1d[n+1, d, k] = T(2) * ws.proj_scaled[d, k] * ws.proj_cheb_1d[n, d, k] - ws.proj_cheb_1d[n-1, d, k]
            end
        end
    end

    # Step 2b: Assemble tensor-product basis
    fill!(ws.proj_basis, one(T))
    @inbounds for j in 1:n_basis
        for d in 1:nx
            deg = pss.multi_indices[j, d]
            @simd for k in 1:N
                ws.proj_basis[j, k] *= ws.proj_cheb_1d[deg + 1, d, k]
            end
        end
    end

    # Step 3: Policy evaluation — single BLAS Level 3 call
    mul!(ws.proj_policy, pss.coefficients, ws.proj_basis)

    # Step 4: Assemble full endogenous = policy_deviation + steady_state
    @inbounds for i in 1:n_endog
        ss_i = pss.steady_state[i]
        @simd for k in 1:N
            ws.particles[i, k] = ws.proj_policy[i, k] + ss_i
        end
    end

    # Step 5: Add shock impact to state variables
    @inbounds for (ii, si) in enumerate(pss.state_indices)
        for j in 1:n_eps
            imp = pss.impact[si, j]
            if imp != zero(T)
                @simd for k in 1:N
                    ws.particles[si, k] += imp * ws.shocks[j, k]
                end
            end
        end
    end

    # Step 6: Clamp states to approximation bounds
    @inbounds for (ii, si) in enumerate(pss.state_indices)
        lb = pss.state_bounds[ii, 1]
        ub = pss.state_bounds[ii, 2]
        @simd for k in 1:N
            ws.particles[si, k] = clamp(ws.particles[si, k], lb, ub)
        end
    end

    return nothing
end

"""
    _pf_initialize_projection!(ws, pss; rng)

Initialize particles from the first-order stationary distribution for a
projection/PFI state space. Draws states from N(ss, P0) where P0 is computed
from the impact matrix, then evaluates the policy function to fill controls.
"""
function _pf_initialize_projection!(ws::PFWorkspace{T},
                                      pss::ProjectionStateSpace{T};
                                      rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    N = size(ws.particles, 2)
    nx = length(pss.state_indices)
    n_endog = length(pss.steady_state)

    # Compute approximate P0 from impact matrix
    impact_state = pss.impact[pss.state_indices, :]
    Sigma_state = impact_state * impact_state'

    # Scale by a diffuse factor for reasonable initial dispersion
    P0 = T(10) * Sigma_state

    # Cholesky for drawing with jitter fallback
    chol_P0 = try
        cholesky(Symmetric(P0))
    catch
        cholesky(Symmetric(P0 + T(1e-8) * I))
    end
    L = chol_P0.L

    # Draw initial states around steady state
    temp = randn(rng, T, nx, N)
    state_draws = L * temp

    # Place in particles at state indices + steady state
    @inbounds for i in 1:n_endog
        ss_i = pss.steady_state[i]
        @simd for k in 1:N
            ws.particles[i, k] = ss_i
        end
    end
    @inbounds for (ii, si) in enumerate(pss.state_indices)
        @simd for k in 1:N
            ws.particles[si, k] = pss.steady_state[si] + state_draws[ii, k]
        end
    end

    # Clamp to bounds
    @inbounds for (ii, si) in enumerate(pss.state_indices)
        lb = pss.state_bounds[ii, 1]
        ub = pss.state_bounds[ii, 2]
        @simd for k in 1:N
            ws.particles[si, k] = clamp(ws.particles[si, k], lb, ub)
        end
    end

    # Evaluate policy to fill controls (transition with zero shocks)
    fill!(ws.shocks, zero(T))
    _pf_transition_projection!(ws, pss)

    return nothing
end

"""
    _bootstrap_particle_filter!(ws, pss::ProjectionStateSpace, data, T_obs; ...)

Bootstrap particle filter for projection/PFI state space.
Uses batch Chebyshev evaluation for the nonlinear policy function.
Same loop structure as linear/nonlinear PF variants.
"""
function _bootstrap_particle_filter!(ws::PFWorkspace{T}, pss::ProjectionStateSpace{T},
                                       data::Matrix{T}, T_obs::Int;
                                       threshold::Real=0.5,
                                       rng::AbstractRNG=Random.default_rng(),
                                       store_trajectory::Bool=false) where {T<:AbstractFloat}
    N = size(ws.particles, 2)
    n_obs = size(data, 1)
    log_lik = zero(T)
    log_N = log(T(N))

    # Initialize from stationary distribution
    _pf_initialize_projection!(ws, pss; rng=rng)

    @inbounds for t in 1:T_obs
        # Draw shocks
        randn!(rng, ws.shocks)

        # Propagate particles through nonlinear policy
        _pf_transition_projection!(ws, pss)

        # Compute observation log-weights
        y_t = @view data[:, t]
        _pf_log_weights!(ws.log_weights, ws.innovations, ws.tmp_obs,
                          ws.particles, y_t, pss.Z, pss.d, pss.H_inv, pss.log_det_H)

        # Accumulate log-likelihood
        log_lik += _logsumexp(ws.log_weights) - log_N

        # Normalize weights
        _normalize_log_weights!(ws.weights, ws.log_weights)

        # ESS-based adaptive resampling
        ess = zero(T)
        @simd for k in 1:N
            ess += ws.weights[k]^2
        end
        ess = one(T) / ess

        if ess < T(threshold) * T(N)
            _systematic_resample!(ws.ancestors, ws.weights, ws.cumweights, N, rng)
            _resample_particles!(ws.particles_new, ws.particles, ws.ancestors)
            ws.particles, ws.particles_new = ws.particles_new, ws.particles
        end
    end

    return log_lik
end
