# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Dynamic analysis for steady-state LQG Rational Inattention solutions
(`SimsKalmanSolution`): inertial impulse response functions (`irf`),
fundamental vs. RI processing-noise variance decompositions (`fevd`),
and joint stochastic simulation (`simulate`).
"""

# =============================================================================
# Internal Helpers (_rian_*)
# =============================================================================

@inline _rian_symmetrize(M::AbstractMatrix{T}) where {T<:AbstractFloat} =
    Matrix{T}((M + M') / 2)

# =============================================================================
# Impulse Response Functions
# =============================================================================

"""
    irf(sol::SimsKalmanSolution{T}, H::Int=20;
        observables::Symbol=:all, ci_type::Symbol=:none) where {T<:AbstractFloat} -> ImpulseResponse{T}

Compute analytical impulse response functions from the joint ``(x_t, \\hat{x}_{t|t})``
state-space representation of a [`SimsKalmanSolution`](@ref).

Responses are computed to both the ``m`` fundamental shocks (`"Fundamental: ..."`) and
the ``r_s`` unit-variance endogenous Rational Inattention processing-noise shocks
(`"RI Noise: ..."`).

# Arguments
- `sol::SimsKalmanSolution{T}`: converged steady-state Sims Kalman solution.
- `H::Int=20`: impulse response horizon (number of periods, including impact `h = 1`).

# Keyword Arguments
- `observables::Symbol=:all`: observable block (`:actions`, `:actions_and_states`, or `:all`).
- `ci_type::Symbol=:none`: confidence interval indicator (`:none` for analytical IRFs).
"""
function irf(
    sol::SimsKalmanSolution{T},
    H::Int=20;
    observables::Symbol=:all,
    ci_type::Symbol=:none,
) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("Horizon H must be >= 1, got $H"))
    sys = to_abcd(sol; observables=observables)
    return irf(sys, H; ci_type=ci_type)
end

# =============================================================================
# Forecast Error Variance Decomposition
# =============================================================================

"""
    fevd(sol::SimsKalmanSolution{T}, H::Int=20;
         observables::Symbol=:actions) where {T<:AbstractFloat} -> FEVD{T}

Compute the analytical forecast error variance decomposition (FEVD) for a
[`SimsKalmanSolution`](@ref), quantifying at each horizon ``h = 1, \\dots, H``
the variance share attributable to each fundamental shock ``w_t`` versus each
endogenous Rational Inattention processing-noise shock ``\\tilde{\\psi}_t``.

# Arguments
- `sol::SimsKalmanSolution{T}`: converged steady-state Sims Kalman solution.
- `H::Int=20`: variance decomposition horizon.

# Keyword Arguments
- `observables::Symbol=:actions`: observable block (`:actions`, `:actions_and_states`, or `:all`).
"""
function fevd(
    sol::SimsKalmanSolution{T},
    H::Int=20;
    observables::Symbol=:actions,
) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("Horizon H must be >= 1, got $H"))
    ir = irf(sol, H; observables=observables)
    decomp, props = _fevd_accumulate(ir)
    return FEVD{T}(decomp, props, ir.variables, ir.shocks)
end

# =============================================================================
# Stochastic Simulation
# =============================================================================

"""
    simulate(sol::SimsKalmanSolution{T}, T_sim::Int;
             burn::Int=100,
             x0::Union{Nothing,AbstractVector{<:Real}}=nothing,
             xhat0::Union{Nothing,AbstractVector{<:Real}}=nothing,
             rng::AbstractRNG=Random.default_rng(),
             seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat} -> NamedTuple

Simulate ``T_{\\text{sim}}`` periods of true states, posterior beliefs, endogenous signals,
optimal actions, fundamental shocks, and RI processing noise from a steady-state
[`SimsKalmanSolution`](@ref):

```math
x_t = A x_{t-1} + B_u u_{t-1} + B w_t, \\qquad s_t = Y' x_t + \\psi_t, \\qquad \\hat{x}_{t|t} = \\hat{x}_{t|t-1} + K (s_t - Y' \\hat{x}_{t|t-1}), \\qquad u_t = F \\hat{x}_{t|t}
```

where:
- ``x_t`` is the ``n \\times 1`` true state vector
- ``A`` is the ``n \\times n`` state transition matrix
- ``B_u`` is the ``n \\times d`` control feedback matrix (zero if absent)
- ``u_t`` is the ``d \\times 1`` optimal action vector
- ``B`` is the ``n \\times m`` fundamental shock loading matrix
- ``w_t \\sim \\mathcal{N}(0, I_m)`` is the ``m \\times 1`` fundamental shock vector
- ``s_t`` is the ``r_s \\times 1`` endogenous signal vector
- ``Y`` is the ``n \\times r_s`` orthonormal signal loading matrix
- ``\\psi_t \\sim \\mathcal{N}(0, \\Sigma_\\psi)`` is the ``r_s \\times 1`` RI processing-noise vector
- ``\\hat{x}_{t|t}`` is the ``n \\times 1`` posterior state belief
- ``\\hat{x}_{t|t-1} = A \\hat{x}_{t-1|t-1} + B_u u_{t-1}`` is the ``n \\times 1`` prior state belief
- ``K`` is the ``n \\times r_s`` steady-state Sims Kalman gain
- ``F`` is the ``d \\times n`` optimal action policy matrix

# Arguments
- `sol::SimsKalmanSolution{T}`: converged steady-state Sims Kalman solution.
- `T_sim::Int`: number of post-burn-in periods to return.

# Keyword Arguments
- `burn::Int=100`: number of burn-in periods discarded before recording.
- `x0::Union{Nothing,AbstractVector{<:Real}}=nothing`: optional initial true state ``x_0`` (defaults to zeros).
- `xhat0::Union{Nothing,AbstractVector{<:Real}}=nothing`: optional initial belief ``\\hat{x}_{0|0}`` (defaults to zeros).
- `rng::AbstractRNG=Random.default_rng()`: random number generator.
- `seed::Union{Integer,Nothing}=nothing`: optional seed; when provided, overrides `rng` with `Xoshiro(seed)`.

# Returns
NamedTuple `(states, beliefs, signals, actions, fundamental_shocks, ri_noise)` with
time along rows (`T_sim × k`).
"""
function simulate(
    sol::SimsKalmanSolution{T},
    T_sim::Int;
    burn::Int=100,
    x0::Union{Nothing,AbstractVector{<:Real}}=nothing,
    xhat0::Union{Nothing,AbstractVector{<:Real}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
) where {T<:AbstractFloat}
    T_sim >= 1 || throw(ArgumentError("T_sim must be >= 1, got $T_sim"))
    burn >= 0 || throw(ArgumentError("burn must be >= 0, got $burn"))

    prob = sol.problem
    n = size(prob.A, 1)
    m = size(prob.B, 2)
    d = size(prob.H, 2)
    r_s = sol.signal_rank

    if x0 !== nothing
        length(x0) == n || throw(ArgumentError("x0 must have length n = $n, got $(length(x0))"))
        all(isfinite, x0) || throw(ArgumentError("x0 contains non-finite values"))
    end
    if xhat0 !== nothing
        length(xhat0) == n || throw(ArgumentError("xhat0 must have length n = $n, got $(length(xhat0))"))
        all(isfinite, xhat0) || throw(ArgumentError("xhat0 contains non-finite values"))
    end

    rng_use = _resolve_repro_rng(rng, seed)
    TW = T <: Union{Float16, Float32} ? Float64 : T

    A = Matrix{TW}(prob.A)
    B = Matrix{TW}(prob.B)
    F = Matrix{TW}(sol.F_policy)
    Bu = prob.B_u === nothing ? zeros(TW, n, d) : Matrix{TW}(prob.B_u)
    Y = Matrix{TW}(sol.Y_signal)
    K = Matrix{TW}(sol.K_gain)
    Spsi_half = r_s == 0 ? zeros(TW, 0, 0) :
        Matrix{TW}(safe_cholesky(_rian_symmetrize(Matrix{TW}(sol.Sigma_psi)); silent=true))

    total_T = burn + T_sim
    W_all = randn(rng_use, TW, total_T, m)
    Psi_std_all = r_s == 0 ? zeros(TW, total_T, 0) : randn(rng_use, TW, total_T, r_s)

    x = x0 === nothing ? zeros(TW, n) : Vector{TW}(x0)
    xhat = xhat0 === nothing ? zeros(TW, n) : Vector{TW}(xhat0)
    u_prev = F * xhat

    states = Matrix{TW}(undef, T_sim, n)
    beliefs = Matrix{TW}(undef, T_sim, n)
    signals = Matrix{TW}(undef, T_sim, r_s)
    actions = Matrix{TW}(undef, T_sim, d)
    fundamental_shocks = Matrix{TW}(undef, T_sim, m)
    ri_noise = Matrix{TW}(undef, T_sim, r_s)

    Yt = Y'
    for t in 1:total_T
        w_t = Vector{TW}(@view W_all[t, :])
        psi_t = r_s == 0 ? zeros(TW, 0) : Spsi_half * @view(Psi_std_all[t, :])

        x = A * x + Bu * u_prev + B * w_t
        xhat_prior = A * xhat + Bu * u_prev
        s_t = r_s == 0 ? zeros(TW, 0) : Yt * x + psi_t
        xhat = r_s == 0 ? xhat_prior : xhat_prior + K * (s_t - Yt * xhat_prior)
        u_t = F * xhat
        u_prev = u_t

        if t > burn
            idx = t - burn
            states[idx, :] .= x
            beliefs[idx, :] .= xhat
            signals[idx, :] .= s_t
            actions[idx, :] .= u_t
            fundamental_shocks[idx, :] .= w_t
            ri_noise[idx, :] .= psi_t
        end
    end

    return (
        states = Matrix{T}(states),
        beliefs = Matrix{T}(beliefs),
        signals = Matrix{T}(signals),
        actions = Matrix{T}(actions),
        fundamental_shocks = Matrix{T}(fundamental_shocks),
        ri_noise = Matrix{T}(ri_noise),
    )
end
