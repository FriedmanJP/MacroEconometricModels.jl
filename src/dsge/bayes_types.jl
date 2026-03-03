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
Type definitions for Bayesian DSGE estimation — priors, state spaces, particle filter
workspace, SMC state, and posterior result container.
"""

# =============================================================================
# DSGEPrior — prior specification for Bayesian DSGE
# =============================================================================

"""
    DSGEPrior{T}

Prior specification for Bayesian DSGE estimation.

Fields:
- `param_names::Vector{Symbol}` — parameter names
- `distributions::Vector{Distribution}` — marginal prior distributions
- `lower::Vector{T}` — lower bounds for truncation / support
- `upper::Vector{T}` — upper bounds for truncation / support
"""
struct DSGEPrior{T<:AbstractFloat}
    param_names::Vector{Symbol}
    distributions::Vector{Distribution}
    lower::Vector{T}
    upper::Vector{T}

    function DSGEPrior{T}(param_names::Vector{Symbol},
                          distributions::Vector{<:Distribution},
                          lower::Vector{T},
                          upper::Vector{T}) where {T<:AbstractFloat}
        n = length(param_names)
        @assert length(distributions) == n "distributions length must match param_names"
        @assert length(lower) == n "lower bounds length must match param_names"
        @assert length(upper) == n "upper bounds length must match param_names"
        new{T}(param_names, distributions, lower, upper)
    end
end

"""
    DSGEPrior(priors::Dict{Symbol, Distribution}; lower=Dict{Symbol,Float64}(), upper=Dict{Symbol,Float64}())

Convenience constructor from a Dict mapping parameter names to distributions.
Optional `lower` and `upper` dicts specify parameter bounds (default: -Inf / +Inf).
"""
function DSGEPrior(priors::Dict{Symbol, <:Distribution};
                   lower::Dict{Symbol, <:Real}=Dict{Symbol,Float64}(),
                   upper::Dict{Symbol, <:Real}=Dict{Symbol,Float64}())
    names = collect(keys(priors))
    sort!(names)  # deterministic ordering
    dists = [priors[n] for n in names]
    T = Float64
    lb = T[get(lower, n, T(-Inf)) for n in names]
    ub = T[get(upper, n, T(Inf)) for n in names]
    DSGEPrior{T}(names, dists, lb, ub)
end

# =============================================================================
# DSGEStateSpace — linear state space representation with cached inverses
# =============================================================================

"""
    DSGEStateSpace{T}

Linear state space for the Kalman / particle filter:
    x_{t+1} = G1 * x_t + impact * varepsilon_t
    y_t     = Z * x_t + d + measurement_error

Fields:
- `G1::Matrix{T}` — n_states x n_states transition matrix
- `impact::Matrix{T}` — n_states x n_shocks impact matrix
- `Z::Matrix{T}` — n_obs x n_states observation matrix
- `d::Vector{T}` — n_obs observation intercept
- `H::Matrix{T}` — n_obs x n_obs measurement error covariance
- `Q::Matrix{T}` — n_shocks x n_shocks shock covariance
- `H_inv::Matrix{T}` — pre-computed inverse of H
- `log_det_H::T` — pre-computed log determinant of H
"""
struct DSGEStateSpace{T<:AbstractFloat}
    G1::Matrix{T}
    impact::Matrix{T}
    Z::Matrix{T}
    d::Vector{T}
    H::Matrix{T}
    Q::Matrix{T}
    H_inv::Matrix{T}
    log_det_H::T

    function DSGEStateSpace{T}(G1::Matrix{T}, impact::Matrix{T},
                                Z::Matrix{T}, d::Vector{T},
                                H::Matrix{T}, Q::Matrix{T}) where {T<:AbstractFloat}
        n_obs = size(Z, 1)
        @assert size(H) == (n_obs, n_obs) "H must be n_obs x n_obs"
        @assert size(G1, 1) == size(G1, 2) "G1 must be square"
        @assert size(Z, 2) == size(G1, 1) "Z columns must match G1 dimension"
        @assert length(d) == n_obs "d length must match n_obs"
        H_inv = Matrix{T}(robust_inv(H))
        log_det_H = T(logdet(H))
        new{T}(G1, impact, Z, d, H, Q, H_inv, log_det_H)
    end
end

# =============================================================================
# NonlinearStateSpace — for PerturbationSolution (order 1/2/3)
# =============================================================================

"""
    NonlinearStateSpace{T}

Nonlinear state space for particle filtering with higher-order perturbation solutions.

For order k, the state transition is:
- Order 1: x_{t+1} = hx * [x_t; varepsilon_t], z_t = gx * [x_t; varepsilon_t]
- Order 2: + (1/2) * hxx * (v_t kron v_t) + (1/2) * hsigmasigma, etc.
- Order 3: + (1/6) * hxxx * (v_t kron v_t kron v_t) + ...

Observation: y_t = Z * z_full_t + d + measurement_error

Fields (1st order — always present):
- `hx, gx` — state/control first-order policy matrices
- `eta` — shock loading matrix
- `steady_state` — steady state vector
- `state_indices, control_indices` — variable partition
- `order::Int` — perturbation order (1, 2, or 3)

Fields (2nd order — nothing if order < 2):
- `hxx, gxx` — 2nd-order Kronecker coefficient matrices
- `hsigmasigma, gsigmasigma` — volatility correction vectors

Fields (3rd order — nothing if order < 3):
- `hxxx, gxxx` — 3rd-order Kronecker coefficient matrices
- `hsigmax, gsigmax` — cross terms
- `hsigmasigmasigma, gsigmasigmasigma` — 3rd-order volatility correction

Observation mapping:
- `Z, d, H, H_inv, log_det_H` — observation equation and cached inverses
"""
struct NonlinearStateSpace{T<:AbstractFloat}
    # First order
    hx::Matrix{T}
    gx::Matrix{T}
    eta::Matrix{T}
    steady_state::Vector{T}
    state_indices::Vector{Int}
    control_indices::Vector{Int}
    order::Int

    # Second order
    hxx::Union{Nothing, Matrix{T}}
    gxx::Union{Nothing, Matrix{T}}
    hsigmasigma::Union{Nothing, Vector{T}}
    gsigmasigma::Union{Nothing, Vector{T}}

    # Third order
    hxxx::Union{Nothing, Matrix{T}}
    gxxx::Union{Nothing, Matrix{T}}
    hsigmax::Union{Nothing, Matrix{T}}
    gsigmax::Union{Nothing, Matrix{T}}
    hsigmasigmasigma::Union{Nothing, Vector{T}}
    gsigmasigmasigma::Union{Nothing, Vector{T}}

    # Observation
    Z::Matrix{T}
    d::Vector{T}
    H::Matrix{T}
    H_inv::Matrix{T}
    log_det_H::T

    function NonlinearStateSpace{T}(hx, gx, eta, steady_state,
                                     state_indices, control_indices, order,
                                     hxx, gxx, hsigmasigma, gsigmasigma,
                                     hxxx, gxxx, hsigmax, gsigmax,
                                     hsigmasigmasigma, gsigmasigmasigma,
                                     Z, d, H) where {T<:AbstractFloat}
        @assert order in (1, 2, 3) "order must be 1, 2, or 3"
        n_obs = size(Z, 1)
        @assert size(H) == (n_obs, n_obs) "H must be n_obs x n_obs"
        @assert length(d) == n_obs "d length must match n_obs"
        H_inv = Matrix{T}(robust_inv(H))
        log_det_H = T(logdet(H))
        new{T}(hx, gx, eta, steady_state, state_indices, control_indices, order,
               hxx, gxx, hsigmasigma, gsigmasigma,
               hxxx, gxxx, hsigmax, gsigmax, hsigmasigmasigma, gsigmasigmasigma,
               Z, d, H, H_inv, log_det_H)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Projection / PFI state space — global nonlinear policy via Chebyshev basis
# ─────────────────────────────────────────────────────────────────────────────

"""
    ProjectionStateSpace{T}

State space for projection/PFI solutions using Chebyshev polynomial policy
functions. Pre-extracts evaluation primitives from `ProjectionSolution` for
zero-allocation batch particle filter evaluation.

Fields:
- `coefficients` — n_vars × n_basis Chebyshev coefficients
- `multi_indices` — n_basis × nx multi-index matrix (polynomial degrees per basis fn)
- `max_degree` — maximum Chebyshev polynomial degree
- `steady_state` — n_endog full steady state vector
- `state_indices` — indices of state variables in endogenous vector
- `control_indices` — indices of control variables in endogenous vector
- `impact` — n_endog × n_exog shock impact matrix (from first-order solution, cached)
- `state_bounds` — nx × 2 approximation domain bounds per state
- `scale` — pre-computed 2/(b-a) per state dimension
- `shift` — pre-computed -(a+b)/(b-a) per state dimension
- `Z, d, H` — observation equation matrices
- `H_inv, log_det_H` — cached inverse and log-determinant of H
"""
struct ProjectionStateSpace{T<:AbstractFloat}
    coefficients::Matrix{T}
    multi_indices::Matrix{Int}
    max_degree::Int

    steady_state::Vector{T}
    state_indices::Vector{Int}
    control_indices::Vector{Int}
    impact::Matrix{T}
    state_bounds::Matrix{T}

    scale::Vector{T}
    shift::Vector{T}

    Z::Matrix{T}
    d::Vector{T}
    H::Matrix{T}
    H_inv::Matrix{T}
    log_det_H::T

    function ProjectionStateSpace{T}(coefficients, multi_indices, max_degree,
                                      steady_state, state_indices, control_indices,
                                      impact, state_bounds, scale, shift,
                                      Z, d, H) where {T<:AbstractFloat}
        n_obs = size(Z, 1)
        @assert size(H) == (n_obs, n_obs) "H must be n_obs × n_obs"
        @assert length(d) == n_obs "d length must match n_obs"
        H_inv = Matrix{T}(robust_inv(H))
        log_det_H = T(logdet(H))
        new{T}(coefficients, multi_indices, max_degree,
               steady_state, state_indices, control_indices,
               impact, state_bounds, scale, shift,
               Z, d, H, H_inv, log_det_H)
    end
end

# =============================================================================
# PFWorkspace — pre-allocated workspace for zero-allocation particle filter
# =============================================================================

"""
    PFWorkspace{T}

Mutable workspace for the bootstrap / conditional SMC particle filter.
All buffers are pre-allocated to avoid allocation in the inner loop.

Core particle buffers:
- `particles, particles_new` — n_states x N current/proposed particles
- `log_weights, weights` — length-N weight vectors
- `ancestors, cumweights` — length-N resampling buffers
- `shocks` — n_shocks x N drawn innovations
- `innovations, tmp_obs` — n_obs x N observation-space temporaries

Higher-order (pruning) buffers:
- `kron_buffer` — nv^2 x N for 2nd-order Kronecker products (nothing if order < 2)
- `kron3_buffer` — nv^3 x N for 3rd-order (nothing if order < 3)
- `kron_cross_buffer` — nv^2 x N for cross-Kronecker kron(vf,vs), 3rd-order (nothing if order < 3)
- `augmented_buffer` — nv x N for building [xf; eps] vectors (nothing if order < 2)
- `transition_scratch` — 3*nx x N scratch for xf_new/xs_new/xt_new in pruned transition (nothing if not nonlinear)
- `particles_fo, particles_so` — 1st/2nd-order pruned state particles, nx x N (nothing if order < 2)
- `particles_to` — 3rd-order pruned state particles, nx x N (nothing if order < 3)

Projection/PFI buffers (Chebyshev batch evaluation):
- `proj_scaled` — nx x N states scaled to [-1,1] (nothing if not projection)
- `proj_cheb_1d` — (max_deg+1) x nx x N 1D Chebyshev polynomial values (nothing if not projection)
- `proj_basis` — n_basis x N tensor-product basis (nothing if not projection)
- `proj_policy` — n_vars x N policy deviations from steady state (nothing if not projection)

Conditional SMC (CSMC):
- `reference_trajectory` — n_states x T_obs reference path (nothing if no CSMC)
- `reference_ancestors` — length-T_obs ancestor indices (nothing if no CSMC)
"""
mutable struct PFWorkspace{T<:AbstractFloat}
    # Core
    particles::Matrix{T}
    particles_new::Matrix{T}
    log_weights::Vector{T}
    weights::Vector{T}
    ancestors::Vector{Int}
    cumweights::Vector{T}
    shocks::Matrix{T}
    innovations::Matrix{T}
    tmp_obs::Matrix{T}

    # Higher-order
    kron_buffer::Union{Nothing, Matrix{T}}
    kron3_buffer::Union{Nothing, Matrix{T}}
    kron_cross_buffer::Union{Nothing, Matrix{T}}
    augmented_buffer::Union{Nothing, Matrix{T}}
    transition_scratch::Union{Nothing, Matrix{T}}
    particles_fo::Union{Nothing, Matrix{T}}
    particles_so::Union{Nothing, Matrix{T}}
    particles_to::Union{Nothing, Matrix{T}}

    # Projection-specific (Chebyshev batch evaluation)
    proj_scaled::Union{Nothing, Matrix{T}}     # nx × N: states scaled to [-1,1]
    proj_cheb_1d::Union{Nothing, Array{T,3}}   # (max_deg+1) × nx × N: 1D Chebyshev values
    proj_basis::Union{Nothing, Matrix{T}}      # n_basis × N: tensor-product basis
    proj_policy::Union{Nothing, Matrix{T}}     # n_vars × N: policy deviations

    # CSMC
    reference_trajectory::Union{Nothing, Matrix{T}}
    reference_ancestors::Union{Nothing, Vector{Int}}
end

"""
    _allocate_pf_workspace(::Type{T}, n_states, n_obs, n_shocks, N;
                           nv=0, nx=0, order=1, T_obs=0,
                           proj_nx=0, proj_n_basis=0,
                           proj_max_degree=0, proj_n_vars=0) where {T}

Allocate a PFWorkspace with all required buffers.

Arguments:
- `n_states` — state dimension (n_endog for full endogenous vector)
- `n_obs` — observation dimension
- `n_shocks` — number of structural shocks
- `N` — number of particles
- `nv` — augmented state dimension for Kronecker products (nx + n_shocks)
- `nx` — number of perturbation state variables (for pruning buffers)
- `order` — perturbation order (1, 2, or 3)
- `T_obs` — number of time periods (>0 enables CSMC reference trajectory)
- `proj_nx` — number of projection state variables (>0 enables projection buffers)
- `proj_n_basis` — number of Chebyshev basis functions
- `proj_max_degree` — maximum Chebyshev polynomial degree
- `proj_n_vars` — number of endogenous variables in projection policy
"""
function _allocate_pf_workspace(::Type{T}, n_states::Int, n_obs::Int,
                                 n_shocks::Int, N::Int;
                                 nv::Int=0, nx::Int=0, order::Int=1,
                                 T_obs::Int=0,
                                 proj_nx::Int=0, proj_n_basis::Int=0,
                                 proj_max_degree::Int=0, proj_n_vars::Int=0) where {T<:AbstractFloat}
    particles = zeros(T, n_states, N)
    particles_new = zeros(T, n_states, N)
    log_weights = zeros(T, N)
    weights = fill(one(T) / N, N)
    ancestors = collect(1:N)
    cumweights = zeros(T, N)
    shocks = zeros(T, n_shocks, N)
    innovations = zeros(T, n_obs, N)
    tmp_obs = zeros(T, n_obs, N)

    # Higher-order buffers — pruning components are nx x N (state variables only)
    # When nx > 0 (nonlinear mode), always allocate particles_fo for state tracking
    nx_eff = nx > 0 ? nx : n_states  # fallback to n_states if nx not specified
    nonlinear = nx > 0
    kron_buffer = order >= 2 && nv > 0 ? zeros(T, nv * nv, N) : nothing
    kron3_buffer = order >= 3 && nv > 0 ? zeros(T, nv * nv * nv, N) : nothing
    kron_cross_buffer = order >= 3 && nv > 0 ? zeros(T, nv * nv, N) : nothing
    augmented_buffer = (order >= 2 || nonlinear) && nv > 0 ? zeros(T, nv, N) : nothing
    # transition_scratch: 3*nx rows for xf_new/xs_new/xt_new in pruned transition
    scratch_rows = nonlinear ? 3 * nx_eff : 0
    transition_scratch = nonlinear ? zeros(T, scratch_rows, N) : nothing
    particles_fo = (order >= 2 || nonlinear) ? zeros(T, nx_eff, N) : nothing
    particles_so = order >= 2 ? zeros(T, nx_eff, N) : nothing
    particles_to = order >= 3 ? zeros(T, nx_eff, N) : nothing

    # Projection buffers
    proj_scaled = proj_nx > 0 ? zeros(T, proj_nx, N) : nothing
    proj_cheb_1d = proj_nx > 0 ? zeros(T, proj_max_degree + 1, proj_nx, N) : nothing
    proj_basis = proj_n_basis > 0 ? zeros(T, proj_n_basis, N) : nothing
    proj_policy = proj_n_vars > 0 ? zeros(T, proj_n_vars, N) : nothing

    # CSMC
    ref_traj = T_obs > 0 ? zeros(T, n_states, T_obs) : nothing
    ref_anc = T_obs > 0 ? zeros(Int, T_obs) : nothing

    PFWorkspace{T}(particles, particles_new, log_weights, weights,
                   ancestors, cumweights, shocks, innovations, tmp_obs,
                   kron_buffer, kron3_buffer, kron_cross_buffer,
                   augmented_buffer, transition_scratch,
                   particles_fo, particles_so, particles_to,
                   proj_scaled, proj_cheb_1d, proj_basis, proj_policy,
                   ref_traj, ref_anc)
end

"""
    _resize_pf_workspace!(ws::PFWorkspace{T}, N_new::Int) where {T}

Resize all particle-count-dependent buffers in-place for adaptive N in SMC^2.
"""
function _resize_pf_workspace!(ws::PFWorkspace{T}, N_new::Int) where {T}
    n_states = size(ws.particles, 1)
    n_obs = size(ws.tmp_obs, 1)
    n_shocks = size(ws.shocks, 1)

    ws.particles = zeros(T, n_states, N_new)
    ws.particles_new = zeros(T, n_states, N_new)
    ws.log_weights = zeros(T, N_new)
    ws.weights = fill(one(T) / N_new, N_new)
    ws.ancestors = collect(1:N_new)
    ws.cumweights = zeros(T, N_new)
    ws.shocks = zeros(T, n_shocks, N_new)
    ws.innovations = zeros(T, n_obs, N_new)
    ws.tmp_obs = zeros(T, n_obs, N_new)

    if ws.kron_buffer !== nothing
        nv2 = size(ws.kron_buffer, 1)
        ws.kron_buffer = zeros(T, nv2, N_new)
    end
    if ws.kron3_buffer !== nothing
        nv3 = size(ws.kron3_buffer, 1)
        ws.kron3_buffer = zeros(T, nv3, N_new)
    end
    if ws.kron_cross_buffer !== nothing
        nv2 = size(ws.kron_cross_buffer, 1)
        ws.kron_cross_buffer = zeros(T, nv2, N_new)
    end
    if ws.augmented_buffer !== nothing
        nv = size(ws.augmented_buffer, 1)
        ws.augmented_buffer = zeros(T, nv, N_new)
    end
    if ws.transition_scratch !== nothing
        sr = size(ws.transition_scratch, 1)
        ws.transition_scratch = zeros(T, sr, N_new)
    end
    if ws.particles_fo !== nothing
        nx = size(ws.particles_fo, 1)
        ws.particles_fo = zeros(T, nx, N_new)
    end
    if ws.particles_so !== nothing
        nx = size(ws.particles_so, 1)
        ws.particles_so = zeros(T, nx, N_new)
    end
    if ws.particles_to !== nothing
        nx = size(ws.particles_to, 1)
        ws.particles_to = zeros(T, nx, N_new)
    end

    # Projection buffers
    if ws.proj_scaled !== nothing
        nx_proj = size(ws.proj_scaled, 1)
        ws.proj_scaled = zeros(T, nx_proj, N_new)
    end
    if ws.proj_cheb_1d !== nothing
        deg_p1, nx_proj, _ = size(ws.proj_cheb_1d)
        ws.proj_cheb_1d = zeros(T, deg_p1, nx_proj, N_new)
    end
    if ws.proj_basis !== nothing
        n_basis = size(ws.proj_basis, 1)
        ws.proj_basis = zeros(T, n_basis, N_new)
    end
    if ws.proj_policy !== nothing
        n_vars = size(ws.proj_policy, 1)
        ws.proj_policy = zeros(T, n_vars, N_new)
    end
    return ws
end

# =============================================================================
# SMCState — mutable state for Sequential Monte Carlo sampler
# =============================================================================

"""
    SMCState{T}

Mutable state container tracking the SMC^2 parameter particle cloud.

Fields:
- `theta_particles::Matrix{T}` — n_params x N_smc parameter particles
- `log_weights::Vector{T}` — log importance weights
- `log_likelihoods::Vector{T}` — per-particle log-likelihoods
- `log_priors::Vector{T}` — per-particle log-prior values
- `phi_schedule::Vector{T}` — tempering schedule phi_0 < ... < phi_P = 1
- `ess_history::Vector{T}` — effective sample size at each tempering step
- `acceptance_rates::Vector{T}` — MCMC acceptance rates at each step
- `log_marginal_likelihood::T` — cumulative log marginal likelihood estimate
- `pf_workspace_pool::Vector{PFWorkspace{T}}` — reusable PF workspace pool
- `proposal_cov::Matrix{T}` — MCMC mutation proposal covariance
"""
mutable struct SMCState{T<:AbstractFloat}
    theta_particles::Matrix{T}
    log_weights::Vector{T}
    log_likelihoods::Vector{T}
    log_priors::Vector{T}
    phi_schedule::Vector{T}
    ess_history::Vector{T}
    acceptance_rates::Vector{T}
    log_marginal_likelihood::T
    pf_workspace_pool::Vector{PFWorkspace{T}}
    proposal_cov::Matrix{T}
end

# =============================================================================
# BayesianDSGE — posterior result container
# =============================================================================

"""
    BayesianDSGE{T} <: AbstractDSGEModel

Bayesian DSGE estimation result container.

Fields:
- `theta_draws::Matrix{T}` — n_draws x n_params posterior draws
- `log_posterior::Vector{T}` — log posterior at each draw
- `param_names::Vector{Symbol}` — parameter names
- `priors::DSGEPrior{T}` — prior specification
- `log_marginal_likelihood::T` — log marginal likelihood (model evidence)
- `method::Symbol` — estimation method (:smc, :rwmh, :csmc, etc.)
- `acceptance_rate::T` — MCMC acceptance rate
- `ess_history::Vector{T}` — ESS history (SMC) or empty (MCMC)
- `phi_schedule::Vector{T}` — tempering schedule (SMC) or empty (MCMC)
- `spec::DSGESpec{T}` — model specification
- `solution::Union{DSGESolution{T}, PerturbationSolution{T}}` — solution at posterior mode
- `state_space::Union{DSGEStateSpace{T}, NonlinearStateSpace{T}}` — state space at posterior mode
"""
struct BayesianDSGE{T<:AbstractFloat} <: AbstractDSGEModel
    theta_draws::Matrix{T}
    log_posterior::Vector{T}
    param_names::Vector{Symbol}
    priors::DSGEPrior{T}
    log_marginal_likelihood::T
    method::Symbol
    acceptance_rate::T
    ess_history::Vector{T}
    phi_schedule::Vector{T}
    spec::DSGESpec{T}
    solution::Union{DSGESolution{T}, PerturbationSolution{T}}
    state_space::Union{DSGEStateSpace{T}, NonlinearStateSpace{T}}

    function BayesianDSGE{T}(theta_draws, log_posterior, param_names, priors,
                              log_marginal_likelihood, method, acceptance_rate,
                              ess_history, phi_schedule, spec, solution,
                              state_space) where {T<:AbstractFloat}
        n_draws, n_params = size(theta_draws)
        @assert length(log_posterior) == n_draws "log_posterior length must match n_draws"
        @assert length(param_names) == n_params "param_names length must match n_params"
        @assert method in (:smc, :rwmh, :csmc, :smc2, :importance) "unknown method: $method"
        new{T}(theta_draws, log_posterior, param_names, priors,
               log_marginal_likelihood, method, acceptance_rate,
               ess_history, phi_schedule, spec, solution, state_space)
    end
end
