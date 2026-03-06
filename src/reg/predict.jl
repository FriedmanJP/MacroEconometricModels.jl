# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Out-of-sample prediction dispatches for cross-sectional regression models.

Extends `StatsAPI.predict` with 2-argument methods accepting new data matrices.
"""

using LinearAlgebra, Distributions, StatsAPI

# =============================================================================
# RegModel — Linear Prediction
# =============================================================================

"""
    StatsAPI.predict(m::RegModel{T}, X_new::AbstractMatrix) -> Vector{T}

Predict fitted values for new data from a linear regression model.

Returns X_new * beta.

# Arguments
- `m::RegModel{T}` — estimated OLS/WLS/IV model
- `X_new::AbstractMatrix` — new regressor matrix (n_new x k)

# Returns
`Vector{T}` of predicted values.

# Examples
```julia
m = estimate_reg(y, X)
y_hat = predict(m, X_test)
```
"""
function StatsAPI.predict(m::RegModel{T}, X_new::AbstractMatrix) where {T<:AbstractFloat}
    k = length(m.beta)
    size(X_new, 2) == k ||
        throw(ArgumentError("X_new must have $k columns (got $(size(X_new, 2)))"))
    Vector{T}(Matrix{T}(X_new) * m.beta)
end

# =============================================================================
# LogitModel — Probability Prediction
# =============================================================================

"""
    StatsAPI.predict(m::LogitModel{T}, X_new::AbstractMatrix) -> Vector{T}

Predict probabilities for new data from a logistic regression model.

Returns 1 / (1 + exp(-X_new * beta)).

# Arguments
- `m::LogitModel{T}` — estimated logit model
- `X_new::AbstractMatrix` — new regressor matrix (n_new x k)

# Returns
`Vector{T}` of predicted probabilities in (0, 1).

# Examples
```julia
m = estimate_logit(y, X)
p_hat = predict(m, X_test)
```
"""
function StatsAPI.predict(m::LogitModel{T}, X_new::AbstractMatrix) where {T<:AbstractFloat}
    k = length(m.beta)
    size(X_new, 2) == k ||
        throw(ArgumentError("X_new must have $k columns (got $(size(X_new, 2)))"))
    eta = Matrix{T}(X_new) * m.beta
    one(T) ./ (one(T) .+ exp.(-eta))
end

# =============================================================================
# ProbitModel — Probability Prediction
# =============================================================================

"""
    StatsAPI.predict(m::ProbitModel{T}, X_new::AbstractMatrix) -> Vector{T}

Predict probabilities for new data from a probit regression model.

Returns Phi(X_new * beta), where Phi is the standard normal CDF.

# Arguments
- `m::ProbitModel{T}` — estimated probit model
- `X_new::AbstractMatrix` — new regressor matrix (n_new x k)

# Returns
`Vector{T}` of predicted probabilities in (0, 1).

# Examples
```julia
m = estimate_probit(y, X)
p_hat = predict(m, X_test)
```
"""
function StatsAPI.predict(m::ProbitModel{T}, X_new::AbstractMatrix) where {T<:AbstractFloat}
    k = length(m.beta)
    size(X_new, 2) == k ||
        throw(ArgumentError("X_new must have $k columns (got $(size(X_new, 2)))"))
    eta = Matrix{T}(X_new) * m.beta
    d = Normal(zero(T), one(T))
    T[cdf(d, eta_i) for eta_i in eta]
end
