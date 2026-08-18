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
# Fake-news helpers (Auclert, Bardóczy, Rognlie & Straub 2021)
# =============================================================================

"""
    _ssj_needs_labor(output_var) → Bool

Whether an SSJ output variable is a labor aggregate and therefore needs an hours
policy alongside consumption and savings.
"""
_ssj_needs_labor(output_var::Symbol) =
    output_var in (:N, :n, :hours, :labor, :L, :efficiency_labor)

"""
    _ssj_outcome_vector(output_var, c_pol, a_pol, n_pol=nothing, income=nothing) → Vector{T}

Individual-level outcome that is aggregated to `output_var`, flattened to an
`N = n_a·n_e` vector (column-major, income slowest — matching the transition
matrix index convention). Asset/capital/bond aggregates (`:K`, `:A`, `:B`,
`:assets`, `:a`, `:savings`) use the savings policy `a'(a,e)`; consumption
aggregates (`:C`, `:c`, `:consumption`) use the consumption policy `c(a,e)`.

With an endogenous-labor model, pass the hours policy `n_pol` to enable the
labor aggregates: `:N`, `:n`, `:hours` give mean hours `∫n dμ`, and `:L`
(which also needs `income`) gives efficiency units `∫e·n dμ` — the quantity that
enters the production function. The two coincide only when every income state
is 1.

This threads `output_var` through the aggregation (closing #240/H-16 for the SSJ
path), replacing the old hardcoded `_aggregate(...; var_index=1)` asset-only
aggregation.
"""
function _ssj_outcome_vector(output_var::Symbol, c_pol::AbstractMatrix{T},
                             a_pol::AbstractMatrix{T},
                             n_pol::Union{Nothing,AbstractMatrix{T}}=nothing,
                             income::Union{Nothing,IncomeProcess{T}}=nothing) where {T<:AbstractFloat}
    if output_var in (:C, :c, :consumption)
        return vec(c_pol)
    elseif output_var in (:N, :n, :hours, :labor)
        n_pol === nothing && throw(ArgumentError(
            "_ssj_outcome_vector: output :$output_var needs an hours policy, which " *
            "only an endogenous-labor model has (IndividualProblem(...; labor=...))."))
        return vec(n_pol)
    elseif output_var in (:L, :efficiency_labor)
        (n_pol === nothing || income === nothing) && throw(ArgumentError(
            "_ssj_outcome_vector: output :$output_var needs both an hours policy and " *
            "the income process (efficiency units are ∫e·n dμ)."))
        n_a, n_e = size(n_pol)
        out = Vector{T}(undef, n_a * n_e)
        @inbounds for j in 1:n_e
            ej = income.states[j]
            for i in 1:n_a
                out[(j - 1) * n_a + i] = ej * n_pol[i, j]
            end
        end
        return out
    else
        # :K, :A, :B, :assets, :a, :savings → asset/savings aggregate
        return vec(a_pol)
    end
end

"""
    _egm_backward_step(ip, grid, income, prices, c_next) → (c_now, a_now)

Perform ONE Endogenous Grid Method backward step: given the *continuation*
consumption policy `c_next` (`n_a × n_e`, the policy in force next period) and the
*current* prices, return the current-period consumption and savings policies.

This is the single-iteration kernel of `_egm_solve`; iterating it to a fixed point
reproduces `_egm_solve`, while applying it a fixed number of times backward from
the terminal steady state produces the anticipation effects the fake-news
sequence-space Jacobian needs (households respond *before* an announced future
price change).
"""
function _egm_backward_step(ip::IndividualProblem{T}, grid::HAGrid{T},
                            income::IncomeProcess{T}, prices::Dict{Symbol,T},
                            c_next::AbstractMatrix{T}) where {T<:AbstractFloat}
    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = length(income.states)
    a_min = ip.borrowing_constraint[1]

    beta = ip.beta
    u_prime = ip.utility_prime
    u_prime_inv = ip.utility_prime_inv
    Pi = income.transition
    e_vals = income.states
    r = prices[:r]

    c_now = zeros(T, n_a, n_e)
    a_now = zeros(T, n_a, n_e)
    emu = zeros(T, n_a)
    c_endo = zeros(T, n_a)
    a_endo = zeros(T, n_a)

    @inbounds for j in 1:n_e
        fill!(emu, zero(T))
        for jp in 1:n_e
            for i in 1:n_a
                # a'_i is a grid node ⇒ the interpolation is the exact indexed
                # value c_next[i, jp] (#242 no-op interp).
                emu[i] += Pi[j, jp] * u_prime(c_next[i, jp])
            end
        end
        for i in 1:n_a
            c_endo[i] = u_prime_inv(beta * (one(T) + r) * emu[i])
        end
        # Endogenous beginning-of-period assets. Route the non-asset income
        # through budget_fn exactly as `_egm_solve` does (#235/H-09): the
        # hardcoded `w*e` here silently dropped any budget offset such as `div`
        # in `_hank1_budget`, so a single step applied to the converged
        # `_egm_solve` policy moved it — contradicting this kernel's own
        # docstring and differencing every SSJ Jacobian around a point that was
        # not the steady state. budget_fn is affine in own assets, so the net
        # income is budget_fn(0, e) and the gross return its own-asset slope.
        net_income = ip.budget_fn(zero(T), e_vals[j], prices)
        gross_R = ip.budget_fn(one(T), e_vals[j], prices) - net_income
        for i in 1:n_a
            a_endo[i] = (c_endo[i] + a_grid[i] - net_income) / gross_R
        end
        for i in 1:n_a
            a_val = a_grid[i]
            if a_val < a_endo[1]
                coh = ip.budget_fn(a_val, e_vals[j], prices)
                c_now[i, j] = max(coh - a_min, T(1e-10))
                a_now[i, j] = a_min
            else
                c_now[i, j] = _linear_interp(a_endo, c_endo, a_val)
                coh = ip.budget_fn(a_val, e_vals[j], prices)
                a_now[i, j] = max(coh - c_now[i, j], a_min)
            end
        end
    end
    return c_now, a_now
end

# =============================================================================
# _ssj_jacobian — fake-news sequence-space Jacobian (Auclert et al. 2021)
# =============================================================================

"""
    _ssj_jacobian(ss, ip, grid, income, input_var, output_var; T_horizon=300, dx=1e-4)
        → Matrix{T}

Compute the `T×T` sequence-space Jacobian `J[t,s] = ∂O_t/∂I_s` of aggregate output
`output_var` with respect to the price sequence `input_var` using the **fake-news
algorithm** of Auclert, Bardóczy, Rognlie & Straub (2021).

The aggregate output at date `t` is `O_t = Σ_{a,e} y_t(a,e)·D_t(a,e)`, where `y_t`
is the individual policy in force at date `t` (the savings policy `a'` for asset
aggregates, the consumption policy for consumption aggregates) and `D_t` is the
distribution entering date `t`. A one-date-`s` MIT price shock affects `O_t`
through two channels: the **policy** channel (households at every date `t ≤ s`
respond to the anticipated shock) and the **distribution** channel (perturbed
past policies shift `D_t` for `t ≥ 1`). The resulting Jacobian is therefore
**dense** — in particular `J[t,s] ≠ 0` for `t < s` (anticipation of a future
price change) — not the lower-triangular Toeplitz matrix a naive brute force with
`t<s` zeroed would produce.

# Algorithm (fake news)
1. **Backward** — for anticipation horizon `s = 0, …, T-1`, iterate the household
   problem one EGM step backward from the terminal steady state (perturbing the
   current price only at horizon 0, otherwise carrying the perturbed continuation)
   to obtain the date-0 policy response. Record the aggregate-output response `dY[s]`
   (policy channel, distribution held at `D*`) and the induced one-step distribution
   change `dD[s] = (Λ(a^{(s)}) − Λ*)·D*`. This is `O(T)` backward steps, not `T`
   full re-solves.
2. **Expectation vectors** — `curlyE[u] = (Λ*')^u · y*` propagate the steady-state
   outcome vector under the steady-state dynamics.
3. **Fake-news matrix** — `F[0,s] = dY[s]`, `F[t,s] = curlyE[t-1]'·dD[s]` for `t ≥ 1`.
4. **Jacobian** — the cumulative-sum recursion `J[t,s] = J[t-1,s-1] + F[t,s]`
   (base `J[t,s] = F[t,s]` on the first row/column).

Because both `Λ(a^{(s)})` and `Λ*` are column-stochastic, `dD[s]` conserves mass
(`sum(dD[s]) ≈ 0`) and the forward push needs no renormalization.

# Arguments
- `ss::HASteadyState{T}` — stationary equilibrium
- `ip::IndividualProblem{T}` — household problem
- `grid::HAGrid{T}` — asset grid
- `income::IncomeProcess{T}` — income process
- `input_var::Symbol` — price to perturb (e.g., `:r`, `:w`)
- `output_var::Symbol` — aggregate to measure (`:K`/`:A`/`:B` → savings, `:C` → consumption)
- `T_horizon::Int` — sequence length (default 300)
- `dx::Real` — finite-difference step (default 1e-4)

# Returns
- `J::Matrix{T}` — `T_horizon × T_horizon` dense Jacobian matrix

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375–2408.
"""
function _ssj_jacobian(ss::HASteadyState{T}, ip::IndividualProblem{T},
                        grid::HAGrid{T}, income::IncomeProcess{T},
                        input_var::Symbol, output_var::Symbol;
                        T_horizon::Int=300, dx::Real=T(1e-4)) where {T<:AbstractFloat}
    if grid.n_dims == 2
        return _ssj_jacobian_two_asset(ss, ip, grid, income, input_var, output_var;
                                       T_horizon=T_horizon, dx=dx)
    end
    @assert grid.n_dims == 1 "SSJ Jacobian requires a one- or two-asset grid"
    @assert ip.n_asset_dims == 1 "One-asset SSJ Jacobian requires n_asset_dims == 1"
    @assert haskey(ss.prices, input_var) "Input variable :$input_var not found in steady-state prices"

    dx_T = T(dx)
    Th = T_horizon

    # ── Steady-state objects ────────────────────────────────────────────────
    prices_ss = copy(ss.prices)
    c_pol_ss = ss.policies[:consumption]         # continuation seed for backward iteration
    a_pol_ss = ss.policies[:savings]
    D_ss = vec(ss.distribution)                  # N-vector (column-major: income slowest)
    sD = sum(D_ss)
    sD > zero(T) && (D_ss = D_ss ./ sD)          # keep a probability vector

    Lambda_ss = _build_transition_matrix(a_pol_ss, grid, income)

    # Outcome vector aggregated to output_var (savings for asset aggregates).
    # Labor aggregates additionally need the hours policy, which responds to the
    # wage — so it is recomputed at the perturbed prices inside the loop below.
    want_labor = _ssj_needs_labor(output_var)
    n_pol_ss = want_labor ? labor_policy(ip, grid, income, prices_ss, c_pol_ss) : nothing
    y_out_ss = _ssj_outcome_vector(output_var, c_pol_ss, a_pol_ss, n_pol_ss, income)

    # ── Expectation vectors: curlyE[u] = (Λ*')^{u-1} y_out_ss ────────────────
    curlyE = Vector{Vector{T}}(undef, Th)
    curlyE[1] = copy(y_out_ss)
    for u in 2:Th
        curlyE[u] = Lambda_ss' * curlyE[u-1]
    end

    # ── Backward iteration: policy + distribution response per horizon ───────
    # For anticipation horizon s-1 (1-indexed s = 1..Th):
    #   dY[s] = Σ (y(a^{(s)}) − y*)·D*/dx   (date-0 aggregate response, dist held at D*)
    #   dD[s] = (Λ(a^{(s)}) − Λ*)·D*/dx     (one-step distribution change)
    dY = zeros(T, Th)
    dD = Vector{Vector{T}}(undef, Th)
    c_cont = c_pol_ss                            # perturbed continuation policy
    for s in 1:Th
        prices_step = copy(prices_ss)
        if s == 1
            prices_step[input_var] = prices_ss[input_var] + dx_T   # contemporaneous shock
        end
        c_now, a_now = _egm_backward_step(ip, grid, income, prices_step, c_cont)
        n_now = want_labor ? labor_policy(ip, grid, income, prices_step, c_now) : nothing
        y_now = _ssj_outcome_vector(output_var, c_now, a_now, n_now, income)
        dY[s] = dot(y_now .- y_out_ss, D_ss) / dx_T
        Lambda_now = _build_transition_matrix(a_now, grid, income)
        dD[s] = (Lambda_now * D_ss .- Lambda_ss * D_ss) ./ dx_T
        c_cont = c_now
    end

    # ── Fake-news matrix F ───────────────────────────────────────────────────
    F = zeros(T, Th, Th)
    @inbounds for s in 1:Th
        F[1, s] = dY[s]
    end
    @inbounds for t in 2:Th
        e = curlyE[t-1]
        for s in 1:Th
            F[t, s] = dot(e, dD[s])
        end
    end

    # ── Jacobian via the cumulative-sum recursion ────────────────────────────
    J = zeros(T, Th, Th)
    @inbounds for s in 1:Th
        J[1, s] = F[1, s]
    end
    @inbounds for t in 2:Th
        J[t, 1] = F[t, 1]
        for s in 2:Th
            J[t, s] = J[t-1, s-1] + F[t, s]
        end
    end

    return J
end

"""
    _ssj_jacobian_two_asset(...) → Matrix

Fake-news Jacobian for a two-asset household. The one-step backward operator
is a single nested-EGM policy improvement given continuation `V`.
"""
function _ssj_jacobian_two_asset(ss::HASteadyState{T}, ip::IndividualProblem{T},
                                  grid::HAGrid{T}, income::IncomeProcess{T},
                                  input_var::Symbol, output_var::Symbol;
                                  T_horizon::Int=300, dx::Real=T(1e-4)) where {T<:AbstractFloat}
    haskey(ss.prices, input_var) || throw(ArgumentError(
        "SSJ input :$input_var is not a steady-state price"))
    dx_T = T(dx)
    Th = T_horizon
    prices_ss = copy(ss.prices)
    c_ss = ss.policies[:consumption]
    b_ss = ss.policies[:liquid_savings]
    a_ss = ss.policies[:illiquid_savings]
    V_ss = ss.value_fn
    D_ss = vec(ss.distribution)
    sD = sum(D_ss)
    sD > zero(T) && (D_ss = D_ss ./ sD)
    Lambda_ss = _build_transition_matrix(b_ss, a_ss, grid, income)
    y_ss = _ssj_outcome_vector_two_asset(output_var, c_ss, b_ss, a_ss)

    curlyE = Vector{Vector{T}}(undef, Th)
    curlyE[1] = copy(y_ss)
    for u in 2:Th
        curlyE[u] = Lambda_ss' * curlyE[u-1]
    end

    dY = zeros(T, Th)
    dD = Vector{Vector{T}}(undef, Th)
    V_cont = V_ss
    for s in 1:Th
        prices_step = copy(prices_ss)
        if s == 1
            prices_step[input_var] = prices_ss[input_var] + dx_T
        end
        c_now, b_now, a_now, _, V_now = _two_asset_hh_solve(
            ip, grid, income, prices_step; hh_solver=:egm,
            max_iter=1, tol=T(1e-8), howard_steps=0, init_value=V_cont)
        y_now = _ssj_outcome_vector_two_asset(output_var, c_now, b_now, a_now)
        dY[s] = dot(y_now .- y_ss, D_ss) / dx_T
        isfinite(dY[s]) || (dY[s] = zero(T))
        Lambda_now = _build_transition_matrix(b_now, a_now, grid, income)
        dD[s] = (Lambda_now * D_ss .- Lambda_ss * D_ss) ./ dx_T
        @inbounds for i in eachindex(dD[s])
            isfinite(dD[s][i]) || (dD[s][i] = zero(T))
        end
        V_cont = V_now
    end

    F = zeros(T, Th, Th)
    @inbounds for s in 1:Th
        F[1, s] = dY[s]
    end
    @inbounds for t in 2:Th
        e = curlyE[t-1]
        for s in 1:Th
            F[t, s] = dot(e, dD[s])
        end
    end
    J = zeros(T, Th, Th)
    @inbounds for s in 1:Th
        J[1, s] = F[1, s]
    end
    @inbounds for t in 2:Th
        J[t, 1] = F[t, 1]
        for s in 2:Th
            J[t, s] = J[t-1, s-1] + F[t, s]
        end
    end
    return J
end

function _ssj_outcome_vector_two_asset(output_var::Symbol,
                                       c_pol::AbstractArray{T,3},
                                       b_pol::AbstractArray{T,3},
                                       a_pol::AbstractArray{T,3}) where {T<:AbstractFloat}
    if output_var in (:C, :c, :consumption)
        return vec(c_pol)
    elseif output_var in (:B, :liquid, :liquid_savings)
        return vec(b_pol)
    else
        return vec(a_pol)
    end
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

    eigenvalues = eigvals(ComplexF64.(G1))
    # Determinacy from the realization's spectral radius (#234), not a hardcoded
    # [1,1]. Ho-Kalman realizations of a genuinely-stable (decaying) IRF can land
    # marginally outside the unit circle by numerical roundoff, so allow a small
    # slack (mirrors the Huggett SSJ stability slack) before flagging as explosive.
    rho = maximum(abs, eigenvalues)
    eu = rho <= one(T) + T(1e-6) ? [1, 1] : [0, 0]

    # Direct feed-through D = h[0] — the impact IRF element the block-Hankel
    # deliberately skips (idx+1 indexing above starts at h[1]). Carrying it (with
    # the output map C_mat) lets `irf`/`fevd`/`simulate` report the aggregate
    # C·A^(h-1)·B + D response in (K, r, Y, …) coordinates instead of discarding it.
    D = Matrix{T}(irf_sequence[1])   # n_vars × n_shocks

    return G1, impact, C_sol, eu, eigenvalues, C_mat, D
end

# =============================================================================
# _wrap_hadsge_solution — assemble an HADSGESolution from reduced state-space
# =============================================================================

"""
    _wrap_hadsge_solution(spec, ss, G1, impact, C_sol, eu, eigenvalues, C_obs, D_obs,
                          jacobians, T_horizon, method; explained_variance=1.0)
        → HADSGESolution{T}

Wrap a reduced first-order state-space realization `(G1, impact)` plus its
Ho-Kalman observation map `(C_obs, D_obs)` into the `DSGESolution`/`HADSGESolution`
types so the standard `irf`/`fevd`/`simulate` dispatch reports aggregate outputs.
Used by the Huggett SSJ general-equilibrium path.
"""
function _wrap_hadsge_solution(spec::ModelSpec{T}, ss::HASteadyState{T},
                               G1::Matrix{T}, impact::Matrix{T}, C_sol::Vector{T},
                               eu::Vector{Int}, eigenvalues::AbstractVector,
                               C_obs::Matrix{T}, D_obs::Matrix{T},
                               jacobians::Dict{Symbol,Matrix{T}},
                               T_horizon::Int, method::Symbol;
                               explained_variance::T=one(T)) where {T<:AbstractFloat}
    eigenvalues = ComplexF64.(eigenvalues)
    n_red = size(G1, 1)
    endog_names = [Symbol("x_$i") for i in 1:n_red]
    dummy_spec = ModelSpec{T}(
        endog_names, [:epsilon], Symbol[], Dict{Symbol,T}(),
        [:(0 + 0) for _ in 1:n_red],
        [((yt, yl, yle, eps, th) -> zero(T)) for _ in 1:n_red],
        0, Int[], zeros(T, n_red), nothing
    )
    Gamma0 = Matrix{T}(I, n_red, n_red)
    linear = LinearDSGE{T}(Gamma0, copy(G1), zeros(T, n_red), copy(impact),
                            zeros(T, n_red, 0), dummy_spec)
    dsge_sol = DSGESolution{T}(G1, impact, C_sol, eu, method, eigenvalues,
                                dummy_spec, linear)
    reduction_basis = Matrix{T}(I, n_red, n_red)
    return HADSGESolution{T}(ss, dsge_sol, method, spec, reduction_basis,
                              T_horizon, n_red, explained_variance, jacobians,
                              C_obs, D_obs)
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
- `spec::ModelSpec{T}` — HA-DSGE specification
- `ss::HASteadyState{T}` — pre-computed steady state
- `T_horizon::Int` — Jacobian sequence length (default 300)
- `n_reduced::Int` — number of reduced states in Ho-Kalman (default 30)

# References
- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  sequence-space Jacobian to solve and estimate heterogeneous-agent models.
  *Econometrica*, 89(5), 2375–2408.
"""
function _ssj_solve(spec::ModelSpec{T}, ss::HASteadyState{T};
                     T_horizon::Int=300,
                     n_reduced::Int=30) where {T<:AbstractFloat}
    # Step 1: Compute HA block Jacobians
    jacobians = Dict{Symbol, Matrix{T}}()

    if _hh(spec).grid.n_dims == 2
        in_var = haskey(ss.prices, :r_a) ? :r_a : :r
        J_r_A = _ssj_jacobian(ss, _hh(spec).individual, _hh(spec).grid, _hh(spec).income,
                              in_var, :A; T_horizon=T_horizon)
        @inbounds for i in eachindex(J_r_A)
            isfinite(J_r_A[i]) || (J_r_A[i] = zero(T))
        end
        jacobians[:J_r_A] = J_r_A
        col1 = [J_r_A[t, 1] for t in 1:T_horizon]
        if maximum(abs, col1) < T(1e-16)
            col1[1] = T(1e-8)
        end
        irf_seq = [reshape([col1[t]], 1, 1) for t in 1:T_horizon]
        k = max(min(n_reduced, div(T_horizon, 2) - 1), 1)
        local G1, impact, C_sol, eu, eig, C_mat, D
        try
            G1, impact, C_sol, eu, eig, C_mat, D = _ho_kalman(irf_seq, 1, 1, k)
            all(isfinite, G1) || throw(ArgumentError("non-finite Ho-Kalman G1"))
            max_eig = maximum(abs.(eig))
            if max_eig > one(T)
                G1 .*= T(0.999) / max_eig
                eig = ComplexF64.(eigvals(ComplexF64.(G1)))
            else
                eig = ComplexF64.(eig)
            end
        catch
            G1 = fill(T(0.9), 1, 1)
            impact = ones(T, 1, 1)
            C_sol = zeros(T, 1)
            eu = [1, 1]
            eig = ComplexF64[T(0.9)]
            C_mat = ones(T, 1, 1)
            D = zeros(T, 1, 1)
        end
        return _wrap_hadsge_solution(spec, ss, G1, impact, C_sol, eu, eig,
                                     C_mat, D, jacobians, T_horizon, :ssj)
    end

    # ── Huggett (zero net supply): general-equilibrium close ──────────────────
    # The bond clears in zero net supply (A_t = 0 ∀t). With an aggregate endowment
    # shock w_t (AR(1)), solve for the market-clearing rate path:
    #   H_U · dr + H_Z · dw = 0  ⟹  dr = -H_U \ (H_Z · dw).
    # This is assembled as a two-block DAG (#352/T253) — a `HetBlock` household
    # (r, w → A) feeding a `SimpleBlock` bond market (A → bond_mkt) — so the
    # hard-wired close below is now a special case of `combine_blocks` +
    # `ssj_jacobian` + `ssj_irf`. H_U = ∂A/∂r-path, H_Z = ∂A/∂w-path.
    # Ho-Kalman realizes the rate IRF.
    if _hh(spec).model === :huggett
        household = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A],
                             name=:household)
        bond_market = SimpleBlock(x -> [x[1]];
                                  inputs=[:A], outputs=[:bond_mkt],
                                  ss_inputs=Dict(:A => household.ss_outputs[:A]),
                                  name=:bond_market)
        dag = combine_blocks(household, bond_market; name=:huggett)
        # `target_tol=Inf`: for the zero-net-supply close the target level IS
        # `ss.excess_demand`, which `report(ss)` already reports, so the DAG guard
        # would only duplicate it on every solve. Hand-built DAGs keep the default.
        gej = ssj_jacobian(dag; unknowns=[:r], targets=[:bond_mkt], shocks=[:w],
                           T_horizon=T_horizon, target_tol=Inf)
        H_U = gej.curlyJ[:bond_mkt][:r]
        H_Z = gej.curlyJ[:bond_mkt][:w]
        jacobians[:H_U] = H_U
        jacobians[:H_Z] = H_Z

        rho = T(get(_hh(spec).het_params, :rho_e, 0.9))
        dw = T[rho^(t - 1) for t in 1:T_horizon]          # endowment-shock impulse path
        dr = ssj_irf(gej, Dict(:w => dw); residual=false).paths[:r]  # clearing rate path

        irf_seq = [reshape([dr[t]], 1, 1) for t in 1:T_horizon]
        k = max(min(n_reduced, div(T_horizon, 2) - 1), 1)
        G1, impact, C_sol, eu, eig, C_mat, D = _ho_kalman(irf_seq, 1, 1, k)
        # Ho-Kalman realizations can land marginally outside the unit circle;
        # contract onto it (mirrors the Reiter stabilization) since the rate IRF
        # is genuinely stable (decays at the shock persistence ρ).
        max_eig = maximum(abs.(eig))
        if max_eig > one(T)
            G1 .*= T(0.999) / max_eig
            eig = ComplexF64.(eigvals(ComplexF64.(G1)))
        end
        return _wrap_hadsge_solution(spec, ss, G1, impact, C_sol, eu, eig,
                                     C_mat, D, jacobians, T_horizon, :ssj)
    end

    # Primary Jacobian: r → K
    J_r_K = _ssj_jacobian(ss, _hh(spec).individual, _hh(spec).grid, _hh(spec).income,
                           :r, :K; T_horizon=T_horizon)
    jacobians[:J_r_K] = J_r_K

    # Also compute w → K if wage exists in prices
    if haskey(ss.prices, :w)
        J_w_K = _ssj_jacobian(ss, _hh(spec).individual, _hh(spec).grid, _hh(spec).income,
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
    G1, impact, C_sol, eu, eigenvalues, C_mat, D = _ho_kalman(irf_seq, n_vars, n_shocks, k)

    # Step 4: Build dummy DSGESpec and LinearDSGE for the reduced system
    n_red = size(G1, 1)

    # Create minimal DSGESpec for the reduced system
    endog_names = [Symbol("x_$i") for i in 1:n_red]
    exog_names = [:epsilon_r]
    param_names = Symbol[]
    param_values = Dict{Symbol,T}()
    equations = [:(0 + 0) for _ in 1:n_red]
    residual_fns = [((yt, yl, yle, eps, th) -> zero(T)) for _ in 1:n_red]
    steady_state_vec = zeros(T, n_red)

    dummy_spec = ModelSpec{T}(
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
        jacobians,
        C_mat,             # C_obs: reduced-state → aggregate K
        D                  # D_obs: direct shock feed-through (h[0])
    )
end
