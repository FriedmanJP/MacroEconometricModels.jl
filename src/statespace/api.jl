# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Public API for the state-space module (EV-37, #445): out-of-sample forecasting via
the state recursion with variance accumulation.
"""

"""
    forecast(ss::StateSpaceModel, h::Integer) -> NamedTuple

`h`-step-ahead forecast of a fitted [`StateSpaceModel`](@ref). Iterates the state
recursion `α̂ₜ₊ᵢ = c + T α̂ₜ₊ᵢ₋₁` from the last filtered state `a_{T|T}`, accumulating the
predictive state covariance `Pₜ₊ᵢ = T Pₜ₊ᵢ₋₁ Tᵀ + R Q Rᵀ`, and maps each to the
observation `ŷ = d + Z α̂` with predictive covariance `Z P Zᵀ + H`.

Returns a `NamedTuple`:
- `mean`      — point forecasts (`h × n_obs`)
- `se`        — observation-forecast standard errors (`h × n_obs`, √diag of the predictive covariance)
- `state`     — forecast state paths (`h × n_state`)
- `state_cov` — predictive state covariances (`n_state × n_state × h`)
- `obs_cov`   — predictive observation covariances (`n_obs × n_obs × h`)
"""
function forecast(ss::StateSpaceModel{T}, h::Integer) where {T<:AbstractFloat}
    isfitted(ss) || throw(ArgumentError("forecast requires a fitted StateSpaceModel (call estimate_statespace / local_level / … first)"))
    h >= 1 || throw(ArgumentError("forecast horizon h must be ≥ 1, got $h"))
    n_state, n_obs = ss.n_state, ss.n_obs
    RQR = Matrix{T}(Hermitian(ss.R * ss.Q * ss.R'))
    mean = zeros(T, h, n_obs)
    se = zeros(T, h, n_obs)
    state = zeros(T, h, n_state)
    state_cov = zeros(T, n_state, n_state, h)
    obs_cov = zeros(T, n_obs, n_obs, h)
    if ss.G === nothing
        a = Vector{T}(ss.filtered_state[end, :])
        P = Matrix{T}(ss.filtered_cov[:, :, end])
        for i in 1:h
            a = ss.c .+ ss.Tt * a
            P = ss.Tt * P * ss.Tt' + RQR
            P = (P + P') / 2
            ŷ = ss.d .+ ss.Z * a
            Fy = ss.Z * P * ss.Z' + ss.H
            Fy = (Fy + Fy') / 2
            state[i, :] = a
            state_cov[:, :, i] = P
            mean[i, :] = ŷ
            obs_cov[:, :, i] = Fy
            se[i, :] = sqrt.(max.(diag(Fy), zero(T)))
        end
    else
        # Compute exact (a_{T+1|T}, P_{T+1|T}) accounting for Cov(R η_T, ε_T) = R * G
        ykern = hcat(Matrix{T}(permutedims(ss.y)), fill(T(NaN), n_obs, 1))
        Hh = Matrix{T}(Hermitian(ss.H))
        dvec = all(iszero, ss.d) ? nothing : ss.d
        cvec = all(iszero, ss.c) ? nothing : ss.c
        G_kern = Matrix{T}(ss.R * ss.G)
        store = KalmanFilterStore{T}(n_state, ss.T_obs + 1; innovations=false)
        _kalman_filter!(store, ykern, ss.Z, ss.Tt, RQR, Hh;
                        d=dvec, b=cvec, a0=ss.a1, P0=ss.P1,
                        scalar=(n_obs == 1), G=G_kern)
        a = Vector{T}(store.a_pred[:, end])
        P = Matrix{T}((store.P_pred[:, :, end] + store.P_pred[:, :, end]') / 2)
        for i in 1:h
            if i > 1
                a = ss.c .+ ss.Tt * a
                P = ss.Tt * P * ss.Tt' + RQR
                P = (P + P') / 2
            end
            ŷ = ss.d .+ ss.Z * a
            Fy = ss.Z * P * ss.Z' + ss.H
            Fy = (Fy + Fy') / 2
            state[i, :] = a
            state_cov[:, :, i] = P
            mean[i, :] = ŷ
            obs_cov[:, :, i] = Fy
            se[i, :] = sqrt.(max.(diag(Fy), zero(T)))
        end
    end
    return (mean=mean, se=se, state=state, state_cov=state_cov, obs_cov=obs_cov)
end

"""
    steady_state_kalman(ss::StateSpaceModel) -> SteadyStateKalman

Compute the steady-state Kalman filter for a [`StateSpaceModel`](@ref) using
`RQR = ss.R * ss.Q * ss.R'` and `G = ss.G === nothing ? nothing : ss.R * ss.G`.
"""
function steady_state_kalman(ss::StateSpaceModel{T}) where {T<:AbstractFloat}
    RQR = Matrix{T}(Hermitian(ss.R * ss.Q * ss.R'))
    G_kern = ss.G === nothing ? nothing : Matrix{T}(ss.R * ss.G)
    return steady_state_kalman(ss.Z, ss.H, ss.Tt, RQR; G=G_kern)
end
