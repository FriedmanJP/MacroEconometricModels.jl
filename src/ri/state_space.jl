# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Joint `(x_t, x̂_{t|t})` state-space and canonical ABCD representations for
steady-state LQG Rational Inattention solutions (`SimsKalmanSolution`).

Maps an equilibrium `SimsKalmanSolution` into:
- `to_statespace`: a `StateSpaceModel{T}` driven by orthogonal fundamental
  shocks `w_t` and unit-variance RI processing-noise shocks `ψ̃_t`.
- `to_abcd`: a canonical Fernández-Villaverde, Rubio-Ramírez, Sargent & Watson
  (2007) `ABCDSystem{T}` representation for IRF, FEVD, and analytical moments.
"""

# =============================================================================
# Internal Helpers (_riss_*)
# =============================================================================

@inline _riss_symmetrize(M::AbstractMatrix{T}) where {T<:AbstractFloat} =
    Matrix{T}((M + M') / 2)

function _riss_joint_matrices(sol::SimsKalmanSolution{T}) where {T<:AbstractFloat}
    TW = T <: Union{Float16, Float32} ? Float64 : T
    prob = sol.problem
    n = size(prob.A, 1)
    m = size(prob.B, 2)
    d = size(prob.H, 2)
    r_s = sol.signal_rank

    A = Matrix{TW}(prob.A)
    B = Matrix{TW}(prob.B)
    F = Matrix{TW}(sol.F_policy)
    Bu = prob.B_u === nothing ? zeros(TW, n, d) : Matrix{TW}(prob.B_u)
    BuF = Bu * F

    Y = Matrix{TW}(sol.Y_signal)
    K = Matrix{TW}(sol.K_gain)

    KYt = r_s == 0 ? zeros(TW, n, n) : K * Y'
    IKYt = Matrix{TW}(I, n, n) - KYt

    Sigma_psi_half = if r_s == 0
        zeros(TW, 0, 0)
    else
        Spsi = _riss_symmetrize(Matrix{TW}(sol.Sigma_psi))
        Matrix{TW}(safe_cholesky(Spsi; silent=true))
    end
    K_psi_half = r_s == 0 ? zeros(TW, n, 0) : K * Sigma_psi_half

    T_joint = [
        A         BuF;
        KYt * A   IKYt * A + BuF
    ]
    R_joint = [
        B         zeros(TW, n, r_s);
        KYt * B   K_psi_half
    ]

    state_names = vcat(prob.state_names, [nm * "_hat" for nm in prob.state_names])
    fund_names = ["Fundamental: " * nm for nm in prob.shock_names]
    noise_names = ["RI Noise: s$(j)" for j in 1:r_s]
    shock_names = vcat(fund_names, noise_names)

    return (
        T_joint = Matrix{T}(T_joint),
        R_joint = Matrix{T}(R_joint),
        F = Matrix{T}(F),
        Sigma_psi_half = Matrix{T}(Sigma_psi_half),
        state_names = state_names,
        shock_names = shock_names,
        n = n,
        m = m,
        d = d,
        r_s = r_s,
    )
end

function _riss_obs_matrix(sol::SimsKalmanSolution{T}, observables::Symbol) where {T<:AbstractFloat}
    observables in (:actions, :actions_and_states, :all) || throw(ArgumentError(
        "observables must be :actions, :actions_and_states, or :all, got :$observables"))

    prob = sol.problem
    n = size(prob.A, 1)
    d = size(prob.H, 2)
    F = Matrix{T}(sol.F_policy)
    In = Matrix{T}(I, n, n)
    Znn = zeros(T, n, n)
    Zdn = zeros(T, d, n)

    if observables === :actions
        Z_obs = hcat(Zdn, F)
        obs_names = copy(prob.action_names)
    elseif observables === :actions_and_states
        Z_obs = [In Znn; Zdn F]
        obs_names = vcat(prob.state_names, prob.action_names)
    else # :all
        Z_obs = [In Znn; Znn In; Zdn F]
        obs_names = vcat(prob.state_names, [nm * "_hat" for nm in prob.state_names], prob.action_names)
    end
    return (Z = Matrix{T}(Z_obs), obs_names = obs_names)
end

# =============================================================================
# Public API: to_statespace & to_abcd
# =============================================================================

"""
    to_statespace(sol::SimsKalmanSolution{T};
                  observables::Symbol=:all,
                  init_mode::Union{Nothing,Symbol}=nothing,
                  kappa::Real=1e6) where {T<:AbstractFloat} -> StateSpaceModel{T}

Construct a linear-Gaussian [`StateSpaceModel`](@ref) over the joint ``2n``-dimensional
state vector ``\\xi_t = [x_t'; \\hat{x}_{t|t}']'`` driven by ``m + r_s`` unit-variance
orthogonal shocks ``\\varepsilon_t = [w_t'; \\tilde{\\psi}_t']'``:

```math
\\xi_t = T_{\\text{joint}} \\xi_{t-1} + R_{\\text{joint}} \\varepsilon_t, \\qquad y_t = Z_{\\text{obs}} \\xi_t
```

where:
- ``\\xi_t`` is the ``2n \\times 1`` joint vector of true states ``x_t`` and posterior beliefs ``\\hat{x}_{t|t}``
- ``T_{\\text{joint}}`` is the ``2n \\times 2n`` block transition matrix coupling true dynamics and Kalman belief updating
- ``R_{\\text{joint}}`` is the ``2n \\times (m + r_s)`` shock loading matrix on fundamental shocks ``w_t`` and unit-variance RI noise ``\\tilde{\\psi}_t``
- ``\\varepsilon_t`` is the ``(m + r_s) \\times 1`` vector of orthogonal unit-variance shocks
- ``y_t`` is the ``k \\times 1`` observable vector selected by `observables`
- ``Z_{\\text{obs}}`` is the ``k \\times 2n`` observation selector matrix

# Arguments
- `sol::SimsKalmanSolution{T}`: converged steady-state Sims Kalman solution.

# Keyword Arguments
- `observables::Symbol=:all`: observation subset (`:actions`, `:actions_and_states`, or `:all`).
- `init_mode::Union{Nothing,Symbol}=nothing`: Kalman initialization mode; defaults to `:stationary` when ``\\rho(T_{\\text{joint}}) < 1 - 10^{-6}`` and `:kappa` otherwise.
- `kappa::Real=1e6`: diffuse variance scale when `init_mode === :kappa` or `:diffuse`.
"""
function to_statespace(
    sol::SimsKalmanSolution{T};
    observables::Symbol=:all,
    init_mode::Union{Nothing,Symbol}=nothing,
    kappa::Real=1e6,
) where {T<:AbstractFloat}
    jm = _riss_joint_matrices(sol)
    obs = _riss_obs_matrix(sol, observables)

    T_joint = jm.T_joint
    R_joint = jm.R_joint
    Z_obs = obs.Z
    k = size(Z_obs, 1)
    n_joint = 2 * jm.n
    n_shocks = jm.m + jm.r_s

    H_obs = zeros(T, k, k)
    Q_shocks = Matrix{T}(I, n_shocks, n_shocks)

    if init_mode === nothing
        rho = maximum(abs, eigvals(Matrix{Float64}(T_joint)); init=0.0)
        if rho < 1.0 - 1e-6
            if T <: Union{Float16, Float32}
                RQR_64 = Matrix{Float64}(R_joint) * Matrix{Float64}(R_joint)'
                P1_64 = _dlyap(Matrix{Float64}(T_joint), (RQR_64 + RQR_64') / 2;
                               warn_label="to_statespace")
                P1_sym = Matrix{T}((P1_64 + P1_64') / 2)
                return StateSpaceModel(Z_obs, H_obs, T_joint, Q_shocks;
                    R=R_joint, a1=zeros(T, n_joint), P1=P1_sym)
            else
                return StateSpaceModel(Z_obs, H_obs, T_joint, Q_shocks;
                    R=R_joint, init_mode=:stationary)
            end
        else
            return StateSpaceModel(Z_obs, H_obs, T_joint, Q_shocks;
                R=R_joint, init_mode=:kappa, kappa=kappa)
        end
    else
        return StateSpaceModel(Z_obs, H_obs, T_joint, Q_shocks;
            R=R_joint, init_mode=init_mode, kappa=kappa)
    end
end

"""
    to_abcd(sol::SimsKalmanSolution{T}; observables::Symbol=:actions) where {T<:AbstractFloat} -> ABCDSystem{T}

Convert a steady-state Rational Inattention [`SimsKalmanSolution`](@ref) into a canonical
[`ABCDSystem`](@ref) over the joint state ``\\xi_t = [x_t'; \\hat{x}_{t|t}']'`` and
orthogonal shock vector ``\\varepsilon_t = [w_t'; \\tilde{\\psi}_t']'``:

```math
\\xi_{t+1} = T_{\\text{joint}} \\xi_t + R_{\\text{joint}} \\varepsilon_{t+1}, \\qquad y_{t+1} = (Z_{\\text{obs}} T_{\\text{joint}}) \\xi_t + (Z_{\\text{obs}} R_{\\text{joint}}) \\varepsilon_{t+1}
```

where:
- ``\\xi_t`` is the ``2n \\times 1`` joint state-and-belief vector
- ``T_{\\text{joint}}`` is the ``2n \\times 2n`` joint state-belief transition matrix
- ``R_{\\text{joint}}`` is the ``2n \\times (m + r_s)`` shock impact matrix
- ``\\varepsilon_{t+1}`` is the ``(m + r_s) \\times 1`` unit-variance vector of fundamental and RI noise shocks
- ``y_{t+1}`` is the ``k \\times 1`` observable vector selected by `observables`
- ``Z_{\\text{obs}}`` is the ``k \\times 2n`` observation selection matrix

# Arguments
- `sol::SimsKalmanSolution{T}`: converged steady-state Sims Kalman solution.

# Keyword Arguments
- `observables::Symbol=:actions`: observation subset (`:actions`, `:actions_and_states`, or `:all`).
"""
function to_abcd(
    sol::SimsKalmanSolution{T};
    observables::Symbol=:actions,
) where {T<:AbstractFloat}
    jm = _riss_joint_matrices(sol)
    obs = _riss_obs_matrix(sol, observables)

    A_abcd = jm.T_joint
    B_abcd = jm.R_joint
    C_abcd = Matrix{T}(obs.Z * jm.T_joint)
    D_abcd = Matrix{T}(obs.Z * jm.R_joint)

    return ABCDSystem{T}(
        A_abcd, B_abcd, C_abcd, D_abcd;
        state_names=jm.state_names,
        obs_names=obs.obs_names,
        shock_names=jm.shock_names,
    )
end
