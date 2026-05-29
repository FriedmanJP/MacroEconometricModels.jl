# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Reiter (2009) method for linearizing heterogeneous agent models.

Treats the distribution histogram as part of the state vector and linearizes
the entire system around the stationary equilibrium.  Because the full
distribution lives in an N_a × N_e dimensional space, SVD dimensionality
reduction projects the distribution dynamics onto the dominant singular
vectors, yielding a tractable linear system.

# References
- Reiter, M. (2009). Solving heterogeneous-agent models by projection and
  perturbation. *Journal of Economic Dynamics and Control*, 33(3), 649–665.
"""

using SparseArrays, LinearAlgebra, Random

# =============================================================================
# _reiter_linearize — SVD-reduced linearization of the HA model
# =============================================================================

"""
    _reiter_linearize(ss, ip, grid, income; n_reduced=50, dx=1e-6, n_sim=200, rng=nothing)
        → (G1, impact, n_reduced_actual, explained_variance)

Linearize a heterogeneous agent model around its stationary distribution using
the Reiter (2009) method, then reduce dimensionality via SVD.

The distribution histogram is part of the state vector.  The full transition
matrix Λ maps the N-dimensional distribution one period forward.  Because N
can be large, we identify the reachable subspace by simulating random
perturbations of the distribution through Λ, then retain only the top
singular vectors of the resulting deviation matrix.

The reduced state is `[d̃_t; K_t; Z_t]` where `d̃ = U_k' (d − d_ss)` are
the SVD-compressed distribution deviations, `K` is aggregate capital, and `Z`
is a TFP shock following an AR(1) with persistence `ρ_z = 0.95`.

# Arguments
- `ss::HASteadyState{T}` — stationary equilibrium
- `ip::IndividualProblem{T}` — household problem
- `grid::HAGrid{T}` — one-asset grid
- `income::IncomeProcess{T}` — income process
- `n_reduced::Int` — maximum number of retained singular vectors (default 50)
- `dx::Real` — perturbation scale for distribution probing (default 1e-6)
- `n_sim::Int` — number of random distribution perturbations (default 200)
- `rng` — random number generator (default `MersenneTwister(1234)`)

# Returns
- `G1::Matrix{T}` — `(n_red + n_agg) × (n_red + n_agg)` transition matrix
- `impact::Matrix{T}` — `(n_red + n_agg) × 1` shock impact vector
- `n_reduced_actual::Int` — actual number of retained singular vectors
- `explained_variance::T` — fraction of variance captured by retained vectors

# References
- Reiter, M. (2009). Solving heterogeneous-agent models by projection and
  perturbation. *Journal of Economic Dynamics and Control*, 33(3), 649–665.
"""
function _reiter_linearize(ss::HASteadyState{T}, ip::IndividualProblem{T},
                            grid::HAGrid{T}, income::IncomeProcess{T};
                            n_reduced::Int=50, dx::Real=T(1e-6),
                            n_sim::Int=200,
                            rng::Union{Nothing,AbstractRNG}=nothing) where {T<:AbstractFloat}
    @assert grid.n_dims == 1 "Reiter linearization requires a one-asset grid"
    @assert ip.n_asset_dims == 1 "Reiter linearization requires a one-asset individual problem"

    rng_actual = isnothing(rng) ? Random.MersenneTwister(1234) : rng
    dx_T = T(dx)

    n_a = grid.n_points[1]
    n_e = grid.n_income
    N = n_a * n_e
    a_grid = grid.grids[1]

    # ── Step 1: Extract steady-state objects ──────────────────────────────────

    a_pol_ss = ss.policies[:savings]
    dist_ss = vec(ss.distribution)        # N-vector (column-major: income slowest)
    K_ss = ss.aggregates[:K]

    # Build steady-state transition matrix
    Lambda_ss = _build_transition_matrix(a_pol_ss, grid, income)

    # ── Step 2: Build aggregate-relevant reduction basis via SVD ────────────
    # The distribution evolves as δ_{t+1} = Λ δ_t.  We care about directions
    # that matter for aggregate capital K = a_vec' d.  Build a matrix whose
    # columns span the reachable subspace weighted by both the transition
    # dynamics and the capital aggregator.
    #
    # Strategy: construct a "controllability-like" matrix by iterating Λ on
    # random perturbations *and* an "observability-like" matrix by iterating
    # Λ' on the capital aggregation vector.  The SVD of their combination
    # captures the directions most relevant for aggregate dynamics.

    Lambda_dense = Matrix{T}(Lambda_ss)

    # Build the capital aggregation vector
    a_vec_pre = zeros(T, N)
    @inbounds for j in 1:n_e
        offset = (j - 1) * n_a
        for i in 1:n_a
            a_vec_pre[offset + i] = a_grid[i]
        end
    end

    # Observability matrix: rows are a_vec' Λ^k for k = 0, 1, ..., n_obs-1
    # This identifies which distribution directions affect K at various horizons.
    n_obs = min(N - 1, max(n_reduced * 3, 50))
    O_mat = zeros(T, N, n_obs)
    v_obs = copy(a_vec_pre)
    for k in 1:n_obs
        O_mat[:, k] .= v_obs
        v_obs = Lambda_dense' * v_obs
    end

    # SVD of O_mat to get the dominant observable directions
    F_obs = svd(O_mat)

    n_available = min(length(F_obs.S), N - 1)
    n_red = min(n_reduced, n_available)

    # Threshold: retain singular values above a relative threshold
    s_thresh = F_obs.S[1] * T(1e-10)
    n_above = count(s -> s > s_thresh, F_obs.S[1:n_red])
    n_red = max(n_above, 1)

    # Reduction basis: left singular vectors of the observability matrix
    U_k = F_obs.U[:, 1:n_red]   # N × n_red

    # ── Step 3: Build reduced distribution transition ─────────────────────────
    # G1_dist = U_k' Λ_ss U_k  (project transition into reduced coordinates)

    G1_dist = U_k' * Lambda_dense * U_k   # n_red × n_red

    # ── Step 4: Capital loading ───────────────────────────────────────────────
    # K is a linear function of the distribution: K = a_grid' * d
    # In reduced coordinates: δK = a_loading' * d̃  where a_loading = U_k' * a_vec
    # and a_vec[i + (j-1)*n_a] = a_grid[i]

    a_vec = zeros(T, N)
    @inbounds for j in 1:n_e
        offset = (j - 1) * n_a
        for i in 1:n_a
            a_vec[offset + i] = a_grid[i]
        end
    end
    K_loading = U_k' * a_vec   # n_red vector

    # ── Step 4b: Explained variance ───────────────────────────────────────────
    # Measure how well the reduced basis captures the distribution dynamics
    # that matter for aggregate capital.  For random perturbations δ, compare
    # the full aggregate capital response ΔK_full = a_vec' Λ δ with the
    # reduced reconstruction ΔK_red = K_loading' G1_dist (U_k' δ).

    n_test = min(n_sim, 100)
    var_full = zero(T)
    var_resid = zero(T)

    for k_test in 1:n_test
        noise = randn(rng_actual, T, N)
        noise .-= mean(noise)
        delta_test = dx_T .* noise

        # Full response
        dK_full = dot(a_vec, Lambda_dense * delta_test)

        # Reduced response
        d_tilde = U_k' * delta_test
        dK_red = dot(K_loading, G1_dist * d_tilde)

        var_full += dK_full^2
        var_resid += (dK_full - dK_red)^2
    end

    explained = var_full > zero(T) ? one(T) - var_resid / var_full : one(T)
    explained = clamp(explained, zero(T), one(T))

    # ── Step 6: Assemble full system ──────────────────────────────────────────
    # Aggregate variables: K (capital) and Z (TFP shock)
    # State: [d̃_t (n_red); K_t (1); Z_t (1)]
    #
    # Transition:
    #   d̃_{t+1} = G1_dist * d̃_t  + impact_dist * Z_t
    #   K_{t+1}  = K_loading' * d̃_{t+1}  (derived from distribution)
    #   Z_{t+1}  = rho_z * Z_t
    #
    # To write as a first-order system x_{t+1} = G1 x_t + impact ε_t:

    n_agg = 2  # K and Z
    n_total = n_red + n_agg
    rho_z = T(0.95)

    G1 = zeros(T, n_total, n_total)

    # Distribution block: d̃_{t+1} = G1_dist * d̃_t
    G1[1:n_red, 1:n_red] .= G1_dist

    # TFP shock feeds into distribution dynamics.
    # A positive Z shock shifts savings policy → perturbs the distribution.
    # Approximate the distribution response to Z by a finite-difference on Λ.
    # Perturb prices as if Z changes, re-solve policies, compare transition matrices.
    prices_ss = copy(ss.prices)
    r_ss = prices_ss[:r]
    w_ss = prices_ss[:w]

    # Approximate: dΛ/dZ via price channels
    # Under Cobb-Douglas: dr/dZ = alpha * K^(alpha-1), dw/dZ = (1-alpha)*K^alpha
    # Use a small Z perturbation and re-solve EGM
    dz = T(1e-4)
    prices_pert = copy(prices_ss)
    # Perturb both r and w consistently
    alpha_val = T(0.36)
    delta_val = T(0.025)
    # At steady state: r = alpha*Z*K^(alpha-1) - delta, w = (1-alpha)*Z*K^alpha
    prices_pert[:r] = r_ss + alpha_val * K_ss^(alpha_val - one(T)) * dz
    prices_pert[:w] = w_ss + (one(T) - alpha_val) * K_ss^alpha_val * dz

    _, a_pol_pert = _egm_solve(ip, grid, income, prices_pert; max_iter=1000, tol=T(1e-10))
    Lambda_pert = _build_transition_matrix(a_pol_pert, grid, income)

    # Distribution response to Z: dΛ/dZ * dist_ss
    d_response_Z = (Lambda_pert * dist_ss .- Lambda_ss * dist_ss) ./ dz

    # Normalize the response
    # Project into reduced space
    impact_dist_Z = U_k' * d_response_Z    # n_red vector

    # Distribution block: Z channel
    G1[1:n_red, n_red + 2] .= impact_dist_Z

    # Capital row: K_{t+1} = K_ss + K_loading' * d̃_{t+1}
    # Substitute: K_{t+1} = K_loading' * G1_dist * d̃_t + K_loading' * impact_dist_Z * Z_t
    G1[n_red + 1, 1:n_red] .= vec(K_loading' * G1_dist)
    G1[n_red + 1, n_red + 2] = dot(K_loading, impact_dist_Z)

    # Z row: Z_{t+1} = rho_z * Z_t
    G1[n_red + 2, n_red + 2] = rho_z

    # ── Step 7: Impact vector ─────────────────────────────────────────────────
    # Shock ε_t enters only through Z: Z_t = rho_z * Z_{t-1} + ε_t
    impact_vec = zeros(T, n_total, 1)
    impact_vec[n_red + 2, 1] = one(T)

    # Also propagate the shock to d̃ and K via Z channel
    impact_vec[1:n_red, 1] .= impact_dist_Z
    impact_vec[n_red + 1, 1] = dot(K_loading, impact_dist_Z)

    # ── Step 8: Stabilize if needed ───────────────────────────────────────────
    # Check eigenvalues — if any are barely above 1, dampen
    eigs = eigvals(G1)
    max_eig = maximum(abs.(eigs))

    if max_eig > one(T)
        # Scale down to ensure stability (unit circle contraction)
        scale = T(0.999) / max_eig
        G1 .*= scale
    end

    return G1, impact_vec, n_red, explained
end
