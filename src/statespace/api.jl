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
    a = Vector{T}(ss.filtered_state[end, :])
    P = Matrix{T}(ss.filtered_cov[:, :, end])
    mean = zeros(T, h, n_obs)
    se = zeros(T, h, n_obs)
    state = zeros(T, h, n_state)
    state_cov = zeros(T, n_state, n_state, h)
    obs_cov = zeros(T, n_obs, n_obs, h)
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
    return (mean=mean, se=se, state=state, state_cov=state_cov, obs_cov=obs_cov)
end
