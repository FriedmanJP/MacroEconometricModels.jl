# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Prediction-error-decomposition MLE and filter/smoother driver for the public
state-space module (EV-37, #445). All filtering/smoothing is routed through the
consolidated kernel `_kalman_filter!` / `_rts_smoother` (`src/core/kalman_kernel.jl`)
— NO sixth filter is introduced here.

The Durbin–Koopman (2002) simulation smoother (a SHOULD in the EV-37 spec, for later
Bayesian reuse) is DEFERRED: this release ships the RTS mean/covariance smoother only.
"""

# -----------------------------------------------------------------------------
# Data coercion — accept Vector / Matrix / TimeSeriesData, return T_obs × n_obs.
# -----------------------------------------------------------------------------
_ss_data(y::AbstractVector{<:Real}) = reshape(Float64.(y), :, 1)
_ss_data(y::AbstractMatrix{<:Real}) = Matrix{Float64}(y)
_ss_data(td::TimeSeriesData) = Matrix{Float64}(td.data)

# -----------------------------------------------------------------------------
# Log-likelihood-only kernel run (used inside the optimizer; zero storage alloc).
# `ykern` is n_obs × T_obs; state intercept c → kernel `b`; RQR pre-formed.
# -----------------------------------------------------------------------------
function _ss_loglik(Z::Matrix{T}, Hobs::Matrix{T}, Tt::Matrix{T}, Q::Matrix{T},
                    d::Vector{T}, c::Vector{T}, R::Matrix{T},
                    a1::Vector{T}, P1::Matrix{T}, ykern::Matrix{T}) where {T<:AbstractFloat}
    RQR = Matrix{T}(Hermitian(R * Q * R'))
    Hh  = Matrix{T}(Hermitian(Hobs))
    dvec = all(iszero, d) ? nothing : d
    cvec = all(iszero, c) ? nothing : c
    return _kalman_filter!(nothing, ykern, Z, Tt, RQR, Hh; d=dvec, b=cvec, a0=a1, P0=P1)
end

# Build a spec from a builder θ and return its per-θ log-likelihood (optimizer hot loop).
function _ss_loglik_from_build(build::Function, θ::AbstractVector, ykern::Matrix{T};
                               init_mode::Symbol, kappa::Real) where {T<:AbstractFloat}
    spec = _ss_from_namedtuple(build(θ); init_mode=init_mode, kappa=kappa)
    return _ss_loglik(spec.Z, spec.H, spec.Tt, spec.Q, spec.d, spec.c, spec.R,
                      spec.a1, spec.P1, ykern)
end

# -----------------------------------------------------------------------------
# Full filter + smoother driver — populates all fit arrays for a materialized spec.
# Returns a NamedTuple consumed by the finalizers.
# -----------------------------------------------------------------------------
function _ss_filter_smooth(spec::StateSpaceModel{T}, y::Matrix{T}) where {T<:AbstractFloat}
    n_obs, n_state = spec.n_obs, spec.n_state
    T_obs = size(y, 1)
    ykern = Matrix{T}(permutedims(y))                    # n_obs × T_obs
    RQR = Matrix{T}(Hermitian(spec.R * spec.Q * spec.R'))
    Hh  = Matrix{T}(Hermitian(spec.H))
    dvec = all(iszero, spec.d) ? nothing : spec.d
    cvec = all(iszero, spec.c) ? nothing : spec.c
    store = KalmanFilterStore{T}(n_state, T_obs; innovations=true)
    ll = _kalman_filter!(store, ykern, spec.Z, spec.Tt, RQR, Hh;
                         d=dvec, b=cvec, a0=spec.a1, P0=spec.P1)
    a_smooth, P_smooth, _ = _rts_smoother(store, spec.Tt)

    filtered_state = Matrix{T}(permutedims(store.a_filt))   # T_obs × n_state
    smoothed_state = Matrix{T}(permutedims(a_smooth))
    filtered_cov = copy(store.P_filt)
    smoothed_cov = copy(P_smooth)

    innov = fill(T(NaN), T_obs, n_obs)
    stdres = fill(T(NaN), T_obs, n_obs)
    for t in 1:T_obs
        (store.v === nothing || !isassigned(store.v, t)) && continue
        v = store.v[t]; F = store.F[t]
        length(v) == n_obs || continue                      # skip partially-missing rows
        innov[t, :] = v
        if n_obs == 1
            f = F[1, 1]
            stdres[t, 1] = f > 0 ? v[1] / sqrt(f) : T(NaN)
        else
            L = safe_cholesky(Hermitian(Matrix{T}(F)))
            stdres[t, :] = L \ v
        end
    end
    return (loglik=ll, filtered_state=filtered_state, filtered_cov=filtered_cov,
            smoothed_state=smoothed_state, smoothed_cov=smoothed_cov,
            innovations=innov, std_residuals=stdres)
end

# Assemble a fitted StateSpaceModel from a spec + fit NamedTuple.
function _ss_assemble(spec::StateSpaceModel{T}, y::Matrix{T}, fit, θ::Vector{T},
                      names::Vector{String}, builder, converged::Bool,
                      method::Symbol) where {T<:AbstractFloat}
    StateSpaceModel{T}(spec.Z, spec.H, spec.Tt, spec.Q, spec.d, spec.c, spec.R,
                       spec.a1, spec.P1, spec.init_mode, y,
                       fit.filtered_state, fit.filtered_cov,
                       fit.smoothed_state, fit.smoothed_cov,
                       fit.innovations, fit.std_residuals, fit.loglik,
                       θ, names, builder, converged, method,
                       spec.n_obs, spec.n_state, size(y, 1))
end

# =============================================================================
# estimate_statespace — public MLE / filter entry points
# =============================================================================
"""
    estimate_statespace(build::Function, θ0::AbstractVector, y; init_mode=:kappa,
                        kappa=1e6, param_names=nothing, theta_transform=identity,
                        display_names=nothing, iterations=1000, g_tol=1e-8) -> StateSpaceModel

Maximize the prediction-error-decomposition log-likelihood of the parametric
state-space model `build(θ)` over θ via `Optim.optimize(..., Optim.LBFGS())` (numeric
gradient), then filter and RTS-smooth at θ̂. `build(θ)` must return a `NamedTuple` with
`Z, H, T, Q` (and optional `d, c, R, a1, P1`); it is responsible for enforcing the
positivity of variance blocks (typically by exponentiating log-variance parameters).

`theta_transform`/`display_names` control how θ̂ is presented in `report()` (e.g. the
convenience wrappers pass `exp` to show natural variances). `y` may be an
`AbstractVector`, an `AbstractMatrix` (`T_obs × n_obs`), or a `TimeSeriesData`.

    estimate_statespace(ss::StateSpaceModel, y) -> StateSpaceModel

Filter and smooth a fixed-matrix spec (no optimization); `loglik` is the evaluated
prediction-error-decomposition log-likelihood.
"""
function estimate_statespace(build::Function, θ0::AbstractVector, y;
                             init_mode::Symbol=:kappa, kappa::Real=1e6,
                             param_names=nothing, theta_transform=identity,
                             display_names=nothing, iterations::Int=1000,
                             g_tol::Real=1e-8)
    ydat = _ss_data(y)
    T = eltype(ydat)
    ykern = Matrix{T}(permutedims(ydat))
    θ0v = collect(T.(θ0))
    nll = θ -> -_ss_loglik_from_build(build, θ, ykern; init_mode=init_mode, kappa=kappa)
    res = Optim.optimize(nll, θ0v, Optim.LBFGS(),
                         Optim.Options(iterations=iterations, g_tol=T(g_tol)))
    θ̂ = Optim.minimizer(res)
    converged = Optim.converged(res)
    spec = _ss_from_namedtuple(build(θ̂); init_mode=init_mode, kappa=kappa)
    fit = _ss_filter_smooth(spec, ydat)
    θ_disp = Vector{T}(theta_transform(θ̂))
    names = display_names !== nothing ? collect(String.(display_names)) :
            (param_names !== nothing ? collect(String.(param_names)) :
             ["θ$i" for i in 1:length(θ̂)])
    return _ss_assemble(spec, ydat, fit, θ_disp, names, build, converged, :mle)
end

function estimate_statespace(ss::StateSpaceModel, y)
    ydat = _ss_data(y)
    T = eltype(ss.Z)
    ydat = Matrix{T}(ydat)
    fit = _ss_filter_smooth(ss, ydat)
    return _ss_assemble(ss, ydat, fit, ss.theta, ss.param_names, ss.builder, true, :filter)
end

# =============================================================================
# Convenience wrappers
# =============================================================================
"""
    local_level(y; init_mode=:kappa, kappa=1e6) -> StateSpaceModel

Fit the local-level (random-walk-plus-noise) model

    yₜ = μₜ + εₜ,   εₜ ~ N(0, σ²_ε);   μₜ₊₁ = μₜ + ηₜ,   ηₜ ~ N(0, σ²_η)

by prediction-error-decomposition MLE. `report()` displays σ̂²_ε and σ̂²_η. For the Nile
series (`load_example(:nile)`) this recovers σ̂²_ε ≈ 15099, σ̂²_η ≈ 1469.1
(Durbin & Koopman 2012, §2).
"""
function local_level(y; init_mode::Symbol=:kappa, kappa::Real=1e6)
    ydat = _ss_data(y)
    v0 = max(var(vec(ydat[.!isnan.(ydat[:, 1]), 1])), 1.0)
    build = θ -> (Z=reshape([1.0], 1, 1), H=reshape([exp(θ[1])], 1, 1),
                  T=reshape([1.0], 1, 1), Q=reshape([exp(θ[2])], 1, 1))
    θ0 = [log(v0 / 2), log(v0 / 2)]
    return estimate_statespace(build, θ0, ydat; init_mode=init_mode, kappa=kappa,
                               theta_transform=(θ -> exp.(θ)), display_names=["σ²_ε", "σ²_η"])
end

"""
    local_linear_trend(y; init_mode=:kappa, kappa=1e6) -> StateSpaceModel

Fit the local-linear-trend (integrated random walk with drift) structural model

    yₜ = μₜ + εₜ;   μₜ₊₁ = μₜ + βₜ + ξₜ;   βₜ₊₁ = βₜ + ζₜ

with `εₜ ~ N(0,σ²_ε)`, `ξₜ ~ N(0,σ²_ξ)` (level), `ζₜ ~ N(0,σ²_ζ)` (slope). `report()`
displays σ̂²_ε, σ̂²_ξ, σ̂²_ζ. The state is `[μₜ, βₜ]`.
"""
function local_linear_trend(y; init_mode::Symbol=:kappa, kappa::Real=1e6)
    ydat = _ss_data(y)
    v0 = max(var(vec(ydat[.!isnan.(ydat[:, 1]), 1])), 1.0)
    build = θ -> (Z=[1.0 0.0], H=reshape([exp(θ[1])], 1, 1),
                  T=[1.0 1.0; 0.0 1.0],
                  Q=[exp(θ[2]) 0.0; 0.0 exp(θ[3])],
                  R=[1.0 0.0; 0.0 1.0])
    θ0 = [log(v0 / 2), log(v0 / 10), log(v0 / 100)]
    return estimate_statespace(build, θ0, ydat; init_mode=init_mode, kappa=kappa,
                               theta_transform=(θ -> exp.(θ)), display_names=["σ²_ε", "σ²_ξ", "σ²_ζ"])
end

"""
    estimate_tvp_reg(y, X; intercept=true, init_mode=:kappa, kappa=1e6) -> StateSpaceModel

Time-varying-parameter regression with random-walk coefficients:

    yₜ = Xₜ βₜ + εₜ,   εₜ ~ N(0, σ²_ε);   βₜ₊₁ = βₜ + ηₜ,   ηₜ ~ N(0, diag(σ²_η))

The observation matrix `Zₜ = Xₜ` is time-varying, so this uses a per-period observation
row; the coefficient path `βₜ` is recovered as the smoothed state. When `intercept=true`
a leading column of ones is prepended to `X`. Hyper-parameters (`σ²_ε` and the diagonal
`σ²_η` per coefficient) are estimated by MLE; `report()` displays them. The smoothed
coefficient paths are in `smoothed_state` (columns follow `X`'s columns).
"""
function estimate_tvp_reg(y, X::AbstractMatrix; intercept::Bool=true,
                          init_mode::Symbol=:kappa, kappa::Real=1e6,
                          iterations::Int=1000, g_tol::Real=1e-8)
    ydat = _ss_data(y)
    size(ydat, 2) == 1 || throw(ArgumentError("estimate_tvp_reg requires a univariate y"))
    Tn = size(ydat, 1)
    Xm = Matrix{Float64}(X)
    size(Xm, 1) == Tn || throw(ArgumentError("X must have the same number of rows as y ($Tn)"))
    Xf = intercept ? hcat(ones(Float64, Tn), Xm) : Xm
    k = size(Xf, 2)
    Tt = Matrix{Float64}(I, k, k)
    Rr = Matrix{Float64}(I, k, k)

    # TVP has a time-varying Z_t = X_t, so filter/smooth via a bespoke time-varying
    # driver that still routes each step through the kernel's per-obs update — no
    # sixth filter. The estimator maximizes the summed one-step log-likelihood.
    v0 = max(var(vec(ydat[:, 1])), 1.0)
    θ0 = vcat(log(v0), fill(log(v0 / (100 * k)), k))   # [log σ²_ε, log σ²_η(1..k)]

    ll_of = θ -> _tvp_loglik(ydat[:, 1], Xf, θ, Tt, Rr; kappa=kappa)
    nll = θ -> -ll_of(θ)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(),
                         Optim.Options(iterations=iterations, g_tol=g_tol))
    θ̂ = Optim.minimizer(res)
    converged = Optim.converged(res)

    fit = _tvp_filter_smooth(ydat[:, 1], Xf, θ̂, Tt, Rr; kappa=kappa)
    Hh = reshape([exp(θ̂[1])], 1, 1)
    Qq = Matrix(Diagonal(exp.(θ̂[2:end])))
    spec = StateSpaceModel(Xf[1:1, :], Hh, Tt, Qq; R=Rr, init_mode=init_mode, kappa=kappa)
    θ_disp = exp.(θ̂)
    names = vcat("σ²_ε", ["σ²_η[$(j)]" for j in 1:k])
    return _ss_assemble(spec, ydat, fit, θ_disp, names, nothing, converged, :mle)
end

# Time-varying-Z Kalman: reuse the scalar kernel update logic per step.
function _tvp_loglik(y::AbstractVector{T}, X::Matrix{T}, θ::AbstractVector,
                     Tt::Matrix{T}, R::Matrix{T}; kappa::Real) where {T<:AbstractFloat}
    k = size(X, 2)
    H = exp(T(θ[1]))
    Qd = exp.(T.(θ[2:end]))
    RQR = Matrix{T}(R * Diagonal(Qd) * R')
    a = zeros(T, k)
    P = Matrix{T}(T(kappa) * Matrix{T}(I, k, k))
    ll = zero(T); log2pi = T(log(2π))
    @inbounds for t in 1:length(y)
        a_pred = Tt * a
        P_pred = Tt * P * Tt' + RQR; P_pred = (P_pred + P_pred') / 2
        yt = y[t]
        (isnan(yt)) && (a = a_pred; P = P_pred; continue)
        Zt = @view X[t, :]
        PZ = P_pred * Zt
        f = dot(Zt, PZ) + H
        f = f > 0 ? f : eps(T)
        v = yt - dot(Zt, a_pred)
        Kk = PZ ./ f
        a = a_pred + Kk * v
        IKH = I - Kk * Zt'
        P = IKH * P_pred * IKH' + H * (Kk * Kk'); P = (P + P') / 2
        ll += -T(0.5) * (log2pi + log(f) + v^2 / f)
    end
    return ll
end

function _tvp_filter_smooth(y::AbstractVector{T}, X::Matrix{T}, θ::AbstractVector,
                            Tt::Matrix{T}, R::Matrix{T}; kappa::Real) where {T<:AbstractFloat}
    k = size(X, 2); n = length(y)
    H = exp(T(θ[1]))
    Qd = exp.(T.(θ[2:end]))
    RQR = Matrix{T}(R * Diagonal(Qd) * R')
    a_pred = zeros(T, k, n); P_pred = zeros(T, k, k, n)
    a_filt = zeros(T, k, n); P_filt = zeros(T, k, k, n)
    innov = fill(T(NaN), n, 1); stdres = fill(T(NaN), n, 1)
    a = zeros(T, k); P = Matrix{T}(T(kappa) * Matrix{T}(I, k, k))
    ll = zero(T); log2pi = T(log(2π))
    @inbounds for t in 1:n
        ap = Tt * a; Pp = Tt * P * Tt' + RQR; Pp = (Pp + Pp') / 2
        a_pred[:, t] = ap; P_pred[:, :, t] = Pp
        yt = y[t]
        if isnan(yt)
            a = ap; P = Pp
        else
            Zt = @view X[t, :]
            PZ = Pp * Zt; f = dot(Zt, PZ) + H; f = f > 0 ? f : eps(T)
            v = yt - dot(Zt, ap); Kk = PZ ./ f
            a = ap + Kk * v; IKH = I - Kk * Zt'
            P = IKH * Pp * IKH' + H * (Kk * Kk'); P = (P + P') / 2
            ll += -T(0.5) * (log2pi + log(f) + v^2 / f)
            innov[t, 1] = v; stdres[t, 1] = v / sqrt(f)
        end
        a_filt[:, t] = a; P_filt[:, :, t] = P
    end
    # RTS smoother (time-invariant Tt)
    a_sm = copy(a_filt); P_sm = copy(P_filt)
    @inbounds for t in (n-1):-1:1
        Jt = _rts_smoother_gain(Matrix{T}(P_filt[:, :, t]), Tt, P_pred[:, :, t+1])
        a_sm[:, t] = a_filt[:, t] + Jt * (a_sm[:, t+1] - a_pred[:, t+1])
        Ps = P_filt[:, :, t] + Jt * (P_sm[:, :, t+1] - P_pred[:, :, t+1]) * Jt'
        P_sm[:, :, t] = (Ps + Ps') / 2
    end
    return (loglik=ll, filtered_state=Matrix{T}(permutedims(a_filt)), filtered_cov=P_filt,
            smoothed_state=Matrix{T}(permutedims(a_sm)), smoothed_cov=P_sm,
            innovations=innov, std_residuals=stdres)
end
