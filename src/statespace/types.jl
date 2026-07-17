# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for the public linear-Gaussian state-space module (EV-37, #445).

A thin, documented front end over the consolidated Kalman kernel
(`src/core/kalman_kernel.jl`) — it adds NO sixth filter. The general single-block
state-space form (Durbin & Koopman 2012; Harvey 1989) is

    yₜ = Z αₜ + d + εₜ,   εₜ ~ N(0, H)                    (observation)
    αₜ₊₁ = T αₜ + c + R ηₜ,   ηₜ ~ N(0, Q)                (state)

which the estimation adapter maps onto the kernel's

    xₜ = b + Tt xₜ₋₁ + η̃ₜ,   Cov(η̃) = RQR = R Q R'
    yₜ = d + Z xₜ + νₜ,       Cov(ν) = Hobs

with the issue's state offset `c` → kernel `b`, the pre-formed `RQR`, and the
observation offset `d` passed straight through.
"""

# =============================================================================
# StateSpaceModel
# =============================================================================

"""
    StateSpaceModel{T} <: AbstractStateSpaceModel{T}

A user-specified linear-Gaussian state-space model together with (optionally) the
Kalman filter / RTS smoother output and prediction-error-decomposition MLE results.

# System fields
- `Z::Matrix{T}`      — observation matrix (`n_obs × n_state`)
- `H::Matrix{T}`      — observation-noise covariance (`n_obs × n_obs`)
- `Tt::Matrix{T}`     — state-transition matrix (`n_state × n_state`)
- `Q::Matrix{T}`      — state-noise covariance (`r × r`, `r = size(R,2)`)
- `d::Vector{T}`      — observation intercept (`n_obs`, zeros if absent)
- `c::Vector{T}`      — state intercept (`n_state`, zeros if absent)
- `R::Matrix{T}`      — state-noise loading (`n_state × r`, identity if absent)
- `a1::Vector{T}`     — initial state mean (`n_state`)
- `P1::Matrix{T}`     — initial state covariance (`n_state × n_state`)
- `init_mode::Symbol` — kernel initialization mode (`:diffuse`/`:kappa`/`:stationary`/`:explicit`)

# Data & fit fields (empty when the model is an unfitted spec)
- `y::Matrix{T}`               — data (`T_obs × n_obs`, rows = time; `NaN` = missing)
- `filtered_state::Matrix{T}`  — `aₜ|ₜ` (`T_obs × n_state`)
- `filtered_cov::Array{T,3}`   — `Pₜ|ₜ` (`n_state × n_state × T_obs`)
- `smoothed_state::Matrix{T}`  — `aₜ|T` (`T_obs × n_state`)
- `smoothed_cov::Array{T,3}`   — `Pₜ|T` (`n_state × n_state × T_obs`)
- `innovations::Matrix{T}`     — one-step prediction errors `vₜ` (`T_obs × n_obs`)
- `std_residuals::Matrix{T}`   — standardized innovations `vₜ / √Fₜ` (`T_obs × n_obs`)
- `loglik::T`                  — maximized prediction-error-decomposition log-likelihood
- `theta::Vector{T}`           — estimated hyper-parameters θ̂ (empty for fixed-matrix specs)
- `param_names::Vector{String}`— labels for `theta` (for `report()`)
- `builder::Union{Nothing,Function}` — closure `θ -> NamedTuple(Z,H,T,Q,...)` (parametric form)
- `converged::Bool`            — optimizer convergence flag
- `method::Symbol`             — `:mle` (parametric fit) / `:filter` (fixed-matrix run) / `:spec`

# Dimensions
- `n_obs::Int`, `n_state::Int`, `T_obs::Int`
"""
struct StateSpaceModel{T<:AbstractFloat} <: AbstractStateSpaceModel{T}
    Z::Matrix{T}
    H::Matrix{T}
    Tt::Matrix{T}
    Q::Matrix{T}
    d::Vector{T}
    c::Vector{T}
    R::Matrix{T}
    a1::Vector{T}
    P1::Matrix{T}
    init_mode::Symbol
    y::Matrix{T}
    filtered_state::Matrix{T}
    filtered_cov::Array{T,3}
    smoothed_state::Matrix{T}
    smoothed_cov::Array{T,3}
    innovations::Matrix{T}
    std_residuals::Matrix{T}
    loglik::T
    theta::Vector{T}
    param_names::Vector{String}
    builder::Union{Nothing,Function}
    converged::Bool
    method::Symbol
    n_obs::Int
    n_state::Int
    T_obs::Int
end

# -----------------------------------------------------------------------------
# Fixed-matrix constructor — fully specifies the system (unfitted spec).
# -----------------------------------------------------------------------------
"""
    StateSpaceModel(Z, H, T_mat, Q; d=nothing, c=nothing, R=nothing,
                    a1=nothing, P1=nothing, init_mode=:diffuse, kappa=1e6)

Build an (unfitted) `StateSpaceModel` from fixed system matrices. `d`/`c` default to
zero vectors, `R` to the identity, and the initial `(a1, P1)` are formed via the
kernel's `_kalman_init` (`:diffuse` large-κ initialization by default). Filter,
smooth, and log-likelihood-evaluate it with [`estimate_statespace`](@ref); forecast
with [`forecast`](@ref) once fitted.
"""
function StateSpaceModel(Z::AbstractMatrix, H::AbstractMatrix,
                         T_mat::AbstractMatrix, Q::AbstractMatrix;
                         d=nothing, c=nothing, R=nothing,
                         a1=nothing, P1=nothing,
                         init_mode::Symbol=:kappa, kappa::Real=1e6)
    T = float(promote_type(eltype(Z), eltype(H), eltype(T_mat), eltype(Q)))
    Zm = Matrix{T}(Z); Hm = Matrix{T}(H); Tm = Matrix{T}(T_mat); Qm = Matrix{T}(Q)
    n_obs, n_state = size(Zm)
    size(Hm) == (n_obs, n_obs) || throw(ArgumentError("H must be n_obs×n_obs = $((n_obs, n_obs)), got $(size(Hm))"))
    size(Tm) == (n_state, n_state) || throw(ArgumentError("T must be n_state×n_state = $((n_state, n_state)), got $(size(Tm))"))
    Rm = R === nothing ? Matrix{T}(I, n_state, size(Qm, 1)) : Matrix{T}(R)
    size(Rm, 1) == n_state || throw(ArgumentError("R must have n_state=$n_state rows, got $(size(Rm,1))"))
    size(Qm) == (size(Rm, 2), size(Rm, 2)) || throw(ArgumentError("Q must be r×r with r=size(R,2)=$(size(Rm,2)), got $(size(Qm))"))
    dv = d === nothing ? zeros(T, n_obs) : Vector{T}(d)
    cv = c === nothing ? zeros(T, n_state) : Vector{T}(c)
    length(dv) == n_obs || throw(ArgumentError("d must have length n_obs=$n_obs"))
    length(cv) == n_state || throw(ArgumentError("c must have length n_state=$n_state"))
    RQR = Matrix{T}(Hermitian(Rm * Qm * Rm'))
    mode = init_mode
    if a1 !== nothing && P1 !== nothing
        mode = :explicit
        a_init = Vector{T}(a1); P_init = Matrix{T}(P1)
    else
        a_init, P_init = _kalman_init(mode, Tm, RQR, n_state; kappa=T(kappa))
    end
    empty3 = Array{T,3}(undef, n_state, n_state, 0)
    StateSpaceModel{T}(Zm, Hm, Tm, Qm, dv, cv, Rm, a_init, P_init, mode,
                       Matrix{T}(undef, 0, n_obs),
                       Matrix{T}(undef, 0, n_state), empty3,
                       Matrix{T}(undef, 0, n_state), empty3,
                       Matrix{T}(undef, 0, n_obs), Matrix{T}(undef, 0, n_obs),
                       T(NaN), T[], String[], nothing, false, :spec,
                       n_obs, n_state, 0)
end

# -----------------------------------------------------------------------------
# Parametric-builder constructor — carries a closure for MLE.
# -----------------------------------------------------------------------------
"""
    StateSpaceModel(build::Function, θ0::AbstractVector; init_mode=:diffuse, kappa=1e6,
                    param_names=nothing)

Build an (unfitted) parametric `StateSpaceModel`. `build(θ)` must return a
`NamedTuple` with fields `Z, H, T, Q` and optional `d, c, R, a1, P1`. The system is
materialized at `θ0`; pass the result to [`estimate_statespace`](@ref) to maximize the
log-likelihood over θ.
"""
function StateSpaceModel(build::Function, θ0::AbstractVector;
                         init_mode::Symbol=:kappa, kappa::Real=1e6,
                         param_names=nothing)
    nt = build(collect(float(eltype(θ0)).(θ0)))
    ss0 = _ss_from_namedtuple(nt; init_mode=init_mode, kappa=kappa)
    T = eltype(ss0.Z)
    names = param_names === nothing ? ["θ$i" for i in 1:length(θ0)] : collect(String.(param_names))
    StateSpaceModel{T}(ss0.Z, ss0.H, ss0.Tt, ss0.Q, ss0.d, ss0.c, ss0.R, ss0.a1, ss0.P1,
                       ss0.init_mode, ss0.y, ss0.filtered_state, ss0.filtered_cov,
                       ss0.smoothed_state, ss0.smoothed_cov, ss0.innovations, ss0.std_residuals,
                       ss0.loglik, Vector{T}(θ0), names, build, false, :spec,
                       ss0.n_obs, ss0.n_state, ss0.T_obs)
end

# Materialize a StateSpaceModel spec from a builder NamedTuple.
function _ss_from_namedtuple(nt; init_mode::Symbol=:kappa, kappa::Real=1e6)
    haskey(nt, :Z) && haskey(nt, :H) && haskey(nt, :T) && haskey(nt, :Q) ||
        throw(ArgumentError("state-space builder must return a NamedTuple with fields Z, H, T, Q"))
    StateSpaceModel(nt.Z, nt.H, nt.T, nt.Q;
                    d=get(nt, :d, nothing), c=get(nt, :c, nothing), R=get(nt, :R, nothing),
                    a1=get(nt, :a1, nothing), P1=get(nt, :P1, nothing),
                    init_mode=init_mode, kappa=kappa)
end

# -----------------------------------------------------------------------------
# StatsAPI + accessors
# -----------------------------------------------------------------------------
StatsAPI.loglikelihood(ss::StateSpaceModel) = ss.loglik
StatsAPI.nobs(ss::StateSpaceModel) = ss.T_obs
StatsAPI.coef(ss::StateSpaceModel) = ss.theta

"""
    isfitted(ss::StateSpaceModel) -> Bool

`true` once the model carries filter/smoother output (`method != :spec`).
"""
isfitted(ss::StateSpaceModel) = ss.method !== :spec && ss.T_obs > 0

# =============================================================================
# Display
# =============================================================================
function Base.show(io::IO, ss::StateSpaceModel{T}) where {T}
    if !isfitted(ss)
        print(io, "StateSpaceModel{$T} (unfitted spec: n_obs=$(ss.n_obs), n_state=$(ss.n_state))")
    else
        print(io, "StateSpaceModel{$T} ($(ss.method): T_obs=$(ss.T_obs), n_state=$(ss.n_state), loglik=$(round(ss.loglik, digits=3)))")
    end
end
