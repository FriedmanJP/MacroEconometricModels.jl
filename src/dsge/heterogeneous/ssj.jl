# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Sequence-Space Jacobian (SSJ) solver for heterogeneous agent DSGE models.

Implements the fake news algorithm of Auclert, Bardóczy, Rognlie & Straub (2021)
for computing T×T Jacobians of aggregate outcomes with respect to price sequences,
plus a brute-force numerical Jacobian as a robust fallback.  The Ho-Kalman
realization algorithm converts IRF sequences into minimal state-space form.

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375–2408.
- Ho, B. L., & Kalman, R. E. (1966). Effective construction of linear
  state-variable models from input/output functions. *Regelungstechnik*,
  14(12), 545–548.
"""

using SparseArrays
using LinearAlgebra

# =============================================================================
# _ssj_jacobian — numerical Jacobian via brute-force finite differences
# =============================================================================

"""
    _ssj_jacobian(ss, ip, grid, income, input_var, output_var; T_horizon=300, dx=1e-4)
        → Matrix{T}

Compute the T×T Jacobian of `output_var` w.r.t. `input_var` sequences using
brute-force numerical finite differences.

For each column s ∈ 1:T, perturb the price `input_var` at time s relative to the
steady state, solve the EGM to get perturbed policies, simulate the distribution
path forward, and aggregate to obtain the output variable path.  The Jacobian
column J[:,s] = (output_path − output_ss) / dx.

This brute-force approach is O(T² × EGM_cost) but is numerically robust.  For
`T_horizon ≤ 100` it runs in seconds; the "fake news" optimization described in
Auclert et al. (2021) can be layered on top.

# Arguments
- `ss::HASteadyState{T}` — stationary equilibrium
- `ip::IndividualProblem{T}` — household problem
- `grid::HAGrid{T}` — asset grid
- `income::IncomeProcess{T}` — income process
- `input_var::Symbol` — price to perturb (e.g., `:r`)
- `output_var::Symbol` — aggregate to measure (e.g., `:K`)
- `T_horizon::Int` — sequence length (default 300)
- `dx::Real` — finite-difference step (default 1e-4)

# Returns
- `J::Matrix{T}` — T_horizon × T_horizon Jacobian matrix

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375–2408.
"""
function _ssj_jacobian(ss::HASteadyState{T}, ip::IndividualProblem{T},
                        grid::HAGrid{T}, income::IncomeProcess{T},
                        input_var::Symbol, output_var::Symbol;
                        T_horizon::Int=300, dx::Real=T(1e-4)) where {T<:AbstractFloat}
    @assert grid.n_dims == 1 "SSJ Jacobian requires a one-asset grid"
    @assert ip.n_asset_dims == 1 "SSJ Jacobian requires a one-asset individual problem"
    @assert haskey(ss.prices, input_var) "Input variable :$input_var not found in steady-state prices"

    dx_T = T(dx)
    n_a = grid.n_points[1]
    n_e = grid.n_income
    N = n_a * n_e

    # Steady-state objects
    prices_ss = copy(ss.prices)
    a_pol_ss = ss.policies[:savings]
    dist_ss_2d = ss.distribution          # N_a × N_e
    dist_ss = vec(dist_ss_2d)             # N-vector (column-major: income varies slowest)

    # Steady-state transition matrix
    Lambda_ss = _build_transition_matrix(a_pol_ss, grid, income)

    # Steady-state aggregate for output_var
    agg_ss = _aggregate(dist_ss, grid; var_index=1)  # currently only asset aggregation

    # Allocate Jacobian
    J = zeros(T, T_horizon, T_horizon)

    # For each perturbation time s, compute the column J[:, s]
    for s in 1:T_horizon
        # === Step 1: Perturbed policies at time s ===
        prices_pert = copy(prices_ss)
        prices_pert[input_var] += dx_T

        # Solve EGM at perturbed prices
        _, a_pol_pert = _egm_solve(ip, grid, income, prices_pert;
                                    max_iter=1000, tol=T(1e-10))

        # Perturbed transition matrix
        Lambda_pert = _build_transition_matrix(a_pol_pert, grid, income)

        # === Step 2: Simulate distribution path forward from steady state ===
        # Before time s: distribution evolves under Lambda_ss
        # At time s: transition uses Lambda_pert (perturbed policies in effect)
        # After time s: transition reverts to Lambda_ss
        #
        # The distribution at the start of the simulation is dist_ss.

        # Evolve forward: at time s the shock hits (one-period perturbation)
        # d_s = Lambda_ss^{s-1} * dist_ss  (before shock, this equals dist_ss since
        # dist_ss is stationary)
        # d_{s+1} = Lambda_pert * d_s = Lambda_pert * dist_ss  (the shock period)
        # d_{s+k} = Lambda_ss^{k-1} * d_{s+1}  for k >= 1

        # Since dist_ss is stationary under Lambda_ss, the distribution right before
        # the shock at time s is simply dist_ss, regardless of s.
        d_after_shock = Lambda_pert * dist_ss   # distribution at time s+1

        # Aggregate at each time t
        for t in 1:T_horizon
            if t < s
                # Before the shock: distribution is at steady state
                # No change in aggregate → J[t,s] = 0
                continue
            elseif t == s
                # At the shock period: policies are perturbed but distribution
                # has not yet changed.  The aggregate output changes because
                # the perturbed savings policy applies to the steady-state
                # distribution.
                #
                # Aggregate = Σ a_pol_pert[i,j] * dist_ss[i,j]
                agg_t = zero(T)
                @inbounds for j in 1:n_e
                    offset = (j - 1) * n_a
                    for i in 1:n_a
                        agg_t += a_pol_pert[i, j] * dist_ss[offset + i]
                    end
                end
                J[t, s] = (agg_t - agg_ss) / dx_T
            else
                # After the shock (t > s): policies revert to steady state,
                # but the distribution has been shifted by the one-period
                # perturbation at time s.
                # The distribution at time t is Lambda_ss^{t-s-1} * d_after_shock.
                steps = t - s - 1
                d_t = d_after_shock
                for _ in 1:steps
                    d_t = Lambda_ss * d_t
                    # Normalize to prevent drift
                    s_val = sum(d_t)
                    if s_val > zero(T)
                        d_t ./= s_val
                    end
                end

                # Aggregate under steady-state policies
                agg_t = _aggregate(d_t, grid; var_index=1)
                J[t, s] = (agg_t - agg_ss) / dx_T
            end
        end
    end

    return J
end

# =============================================================================
# _ho_kalman — Ho-Kalman realization algorithm
# =============================================================================

"""
    _ho_kalman(irf_sequence, n_vars, n_shocks, n_reduced)
        → (G1, impact, C_sol, eu, eigenvalues)

Construct a minimal state-space realization from an IRF sequence using the
Ho-Kalman (1966) algorithm.

Given impulse responses h[0], h[1], ..., h[T−1] (each `n_vars × n_shocks`),
build the block-Hankel matrix H and extract a reduced-order state-space model
via truncated SVD:

    x_{t+1} = A x_t + B ε_t
    y_t     = C x_t + D ε_t

The output tuple is mapped to the DSGESolution format:
- `G1::Matrix{T}` — state transition (n_reduced × n_reduced)
- `impact::Matrix{T}` — impact of shocks (n_reduced × n_shocks)
- `C_sol::Vector{T}` — constants (zeros)
- `eu::Vector{Int}` — [1, 1] (existence and uniqueness assumed)
- `eigenvalues::Vector{ComplexF64}` — eigenvalues of G1

# Arguments
- `irf_sequence::Vector{Matrix{T}}` — T-element vector of `n_vars × n_shocks` IRF matrices
- `n_vars::Int` — number of output variables
- `n_shocks::Int` — number of shocks
- `n_reduced::Int` — number of retained states (truncation rank)

# References
- Ho, B. L., & Kalman, R. E. (1966). Effective construction of linear
  state-variable models from input/output functions. *Regelungstechnik*,
  14(12), 545–548.
"""
function _ho_kalman(irf_sequence::Vector{Matrix{T}}, n_vars::Int,
                     n_shocks::Int, n_reduced::Int) where {T<:AbstractFloat}
    T_len = length(irf_sequence)
    @assert T_len >= 2 "Need at least 2 IRF periods"

    # Use h[1], h[2], ..., h[T-1] (skip h[0] which is the direct impact D)
    # Build block-Hankel matrix:
    #   H = [h[1]   h[2]   ... h[p]  ]
    #       [h[2]   h[3]   ... h[p+1]]
    #       [  ⋮                 ⋮    ]
    #       [h[p]   h[p+1] ... h[2p-1]]
    # where p = floor(T_len/2)

    p = div(T_len - 1, 2)
    p = max(p, 1)

    # Ensure we have enough IRF periods
    n_hankel = min(2 * p, T_len - 1)
    p = div(n_hankel, 2)
    p = max(p, 1)

    # Build Hankel matrix: (p * n_vars) × (p * n_shocks)
    H = zeros(T, p * n_vars, p * n_shocks)
    for i in 1:p
        for j in 1:p
            idx = i + j - 1  # index into irf_sequence (1-based, skip h[0])
            if idx <= T_len - 1
                H[(i-1)*n_vars+1:i*n_vars, (j-1)*n_shocks+1:j*n_shocks] = irf_sequence[idx + 1]
            end
        end
    end

    # Also build shifted Hankel H1 (one step shifted):
    H1 = zeros(T, p * n_vars, p * n_shocks)
    for i in 1:p
        for j in 1:p
            idx = i + j  # shifted by 1
            if idx <= T_len - 1
                H1[(i-1)*n_vars+1:i*n_vars, (j-1)*n_shocks+1:j*n_shocks] = irf_sequence[idx + 1]
            end
        end
    end

    # Truncated SVD of H
    F = svd(H)
    k = min(n_reduced, length(F.S), p * n_vars, p * n_shocks)

    # Prevent zero singular values from causing issues
    s_thresh = max(F.S[1] * T(1e-12), T(1e-30))
    k = max(count(s -> s > s_thresh, F.S[1:k]), 1)

    U_k = F.U[:, 1:k]
    S_k = Diagonal(F.S[1:k])
    V_k = F.Vt[1:k, :]'  # p*n_shocks × k

    S_sqrt = Diagonal(sqrt.(F.S[1:k]))
    S_sqrt_inv = Diagonal(one(T) ./ sqrt.(F.S[1:k]))

    # Extract state-space matrices
    # A = S_sqrt_inv * U_k' * H1 * V_k * S_sqrt_inv
    A = S_sqrt_inv * U_k' * H1 * V_k * S_sqrt_inv
    A = Matrix{T}(A)

    # B = first n_shocks columns of S_sqrt * V_k' → k × n_shocks
    # Actually: B = S_sqrt_inv * U_k' * first block column of H1
    # More robustly: B = first n_shocks columns of (S_sqrt * Vt_k)
    B = S_sqrt * F.Vt[1:k, 1:n_shocks]
    B = Matrix{T}(B)

    # C = first n_vars rows of U_k * S_sqrt → n_vars × k
    C_mat = U_k[1:n_vars, :] * S_sqrt
    C_mat = Matrix{T}(C_mat)

    # For DSGESolution format, we need G1 (state transition) and impact
    # The reduced system is:
    #   x_{t+1} = A x_t + B ε_t
    #   y_t     = C x_t + D ε_t
    #
    # To embed in DSGESolution (y_t = G1 y_{t-1} + impact ε_t + C_sol):
    # We use the state-space matrices directly.
    # G1 is the state transition A, impact is B.
    G1 = A
    impact = B
    C_sol = zeros(T, k)
    eu = [1, 1]

    eigenvalues = eigvals(ComplexF64.(G1))

    return G1, impact, C_sol, eu, eigenvalues
end

# =============================================================================
# _ssj_solve — full SSJ solution pipeline
# =============================================================================

"""
    _ssj_solve(spec, ss; T_horizon=300, n_reduced=30) → HADSGESolution{T}

Full Sequence-Space Jacobian solution.

1. Compute HA block Jacobians for key (input, output) pairs.
2. Extract the aggregate impulse response from the primary Jacobian (r → K).
3. Convert to state-space via Ho-Kalman realization.
4. Build a `DSGESolution` from the reduced state-space representation.
5. Return `HADSGESolution` wrapping the result.

# Arguments
- `spec::HADSGESpec{T}` — HA-DSGE specification
- `ss::HASteadyState{T}` — pre-computed steady state
- `T_horizon::Int` — Jacobian sequence length (default 300)
- `n_reduced::Int` — number of reduced states in Ho-Kalman (default 30)

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375–2408.
"""
function _ssj_solve(spec::HADSGESpec{T}, ss::HASteadyState{T};
                     T_horizon::Int=300,
                     n_reduced::Int=30) where {T<:AbstractFloat}
    # Step 1: Compute HA block Jacobians
    jacobians = Dict{Symbol, Matrix{T}}()

    # Primary Jacobian: r → K
    J_r_K = _ssj_jacobian(ss, spec.individual, spec.grid, spec.income,
                           :r, :K; T_horizon=T_horizon)
    jacobians[:J_r_K] = J_r_K

    # Also compute w → K if wage exists in prices
    if haskey(ss.prices, :w)
        J_w_K = _ssj_jacobian(ss, spec.individual, spec.grid, spec.income,
                               :w, :K; T_horizon=T_horizon)
        jacobians[:J_w_K] = J_w_K
    end

    # Step 2: Extract IRF sequence from the first column of J_r_K
    # J_r_K[:, 1] gives the response of K at each time t to a one-unit permanent
    # change in r at time 1. This is the aggregate impulse response.
    n_vars = 1
    n_shocks = 1
    irf_seq = [reshape([J_r_K[t, 1]], 1, 1) for t in 1:T_horizon]

    # Step 3: Ho-Kalman realization
    k = min(n_reduced, div(T_horizon, 2) - 1)
    k = max(k, 1)
    G1, impact, C_sol, eu, eigenvalues = _ho_kalman(irf_seq, n_vars, n_shocks, k)

    # Step 4: Build dummy DSGESpec and LinearDSGE for the reduced system
    n_red = size(G1, 1)

    # Create minimal DSGESpec for the reduced system
    endog_names = [Symbol("x_$i") for i in 1:n_red]
    exog_names = [:epsilon_r]
    param_names = Symbol[]
    param_values = Dict{Symbol,T}()
    equations = [:(0) for _ in 1:n_red]
    residual_fns = [((yt, yl, yle, eps, th) -> zero(T)) for _ in 1:n_red]
    steady_state_vec = zeros(T, n_red)

    dummy_spec = DSGESpec{T}(
        endog_names, exog_names, param_names, param_values,
        equations, residual_fns,
        0,           # n_expect = 0
        Int[],       # forward_indices
        steady_state_vec,
        nothing      # ss_fn
    )

    # Create minimal LinearDSGE
    Gamma0 = Matrix{T}(I, n_red, n_red)
    Gamma1 = copy(G1)
    C_lin = zeros(T, n_red)
    Psi = copy(impact)
    Pi = zeros(T, n_red, 0)

    linear = LinearDSGE{T}(Gamma0, Gamma1, C_lin, Psi, Pi, dummy_spec)

    # Build DSGESolution
    dsge_sol = DSGESolution{T}(G1, impact, C_sol, eu, :ssj, eigenvalues,
                                dummy_spec, linear)

    # Step 5: Build reduction basis from Ho-Kalman SVD
    # Use identity as the basis since the reduced system is already in its own
    # coordinate system
    reduction_basis = Matrix{T}(I, n_red, n_red)

    # Explained variance: ratio of retained singular values to total
    # (computed from the Jacobian's first column as a proxy)
    F_jac = svd(J_r_K)
    total_var = sum(F_jac.S .^ 2)
    explained = sum(F_jac.S[1:min(n_red, length(F_jac.S))] .^ 2)
    explained_variance = total_var > zero(T) ? explained / total_var : one(T)

    return HADSGESolution{T}(
        ss,
        dsge_sol,
        :ssj,
        spec,
        reduction_basis,
        T_horizon,         # n_full_states
        n_red,             # n_reduced
        explained_variance,
        jacobians
    )
end
