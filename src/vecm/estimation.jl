# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
VECM estimation via Johansen MLE and Engle-Granger two-step, with VAR conversion.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Core Estimation
# =============================================================================

"""
    estimate_vecm(Y, p; rank=:auto, deterministic=:constant, method=:johansen, significance=0.05)

Estimate a Vector Error Correction Model.

# Arguments
- `Y`: Data matrix (T × n) in levels
- `p`: Underlying VAR order (VECM has p-1 lagged differences)
- `rank`: Cointegrating rank; `:auto` (default) selects via Johansen trace test, or specify an integer
- `deterministic`: `:none`, `:constant` (default), or `:trend`
- `method`: `:johansen` (default) or `:engle_granger` (bivariate, rank=1 only)
- `significance`: Significance level for automatic rank selection (default 0.05)

# Returns
`VECMModel` with estimated α, β, Γ matrices, residuals, and diagnostics.

# Example
```julia
Y = cumsum(randn(200, 3), dims=1)
Y[:, 2] = Y[:, 1] + 0.1 * randn(200)
m = estimate_vecm(Y, 2)
```
"""
function estimate_vecm(Y::AbstractMatrix{T}, p::Int;
                       rank::Union{Symbol,Int}=:auto,
                       deterministic::Symbol=:constant,
                       method::Symbol=:johansen,
                       significance::Real=0.05) where {T<:AbstractFloat}

    deterministic ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("deterministic must be :none, :constant, or :trend"))
    method ∈ (:johansen, :engle_granger) ||
        throw(ArgumentError("method must be :johansen or :engle_granger"))

    T_obs, n = size(Y)
    p < 1 && throw(ArgumentError("Number of lags p must be at least 1"))
    T_obs < n + p + 10 && throw(ArgumentError("Not enough observations for VECM estimation"))

    if method == :engle_granger
        return _estimate_vecm_engle_granger(Y, p, deterministic, rank)
    end

    # === Johansen MLE ===

    # Run Johansen test for rank selection
    joh = johansen_test(Y, p; deterministic=deterministic)

    # Determine rank
    r = if rank === :auto
        # Use trace test at the given significance level
        _select_rank_trace(joh, significance)
    else
        rank::Int
    end

    r < 0 && throw(ArgumentError("rank must be non-negative"))
    r > n && throw(ArgumentError("rank cannot exceed number of variables ($n)"))

    # Handle edge cases
    if r == 0
        # No cointegration: estimate as VAR in differences
        return _vecm_rank_zero(Y, p, n, deterministic, joh)
    end

    # Construct VECM matrices (parallels johansen.jl)
    dY = diff(Y, dims=1)
    Y_lag = Y[p:end-1, :]  # Y_{t-1}

    T_eff = T_obs - p
    dY_lags = if p > 1
        hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...)
    else
        Matrix{T}(undef, T_eff, 0)
    end
    dY_eff = dY[p:end, :]

    # Deterministic terms for concentrating out
    Z = _build_deterministic_Z(T, T_eff, dY_lags, deterministic)

    # Concentrate out short-run dynamics
    if size(Z, 2) > 0
        ZtZ_inv = robust_inv(Z'Z)
        M = I - Z * (ZtZ_inv * Z')
        R0 = M * dY_eff
        R1 = M * Y_lag
    else
        R0 = copy(dY_eff)
        R1 = copy(Y_lag)
    end

    # Moment matrices
    S00 = (R0'R0) / T_eff
    S11 = (R1'R1) / T_eff
    S01 = (R0'R1) / T_eff

    # Eigendecomposition for beta
    S00_inv = robust_inv(S00)
    A = S11 \ (S01' * S00_inv * S01)
    eig = eigen(A)
    idx = sortperm(real.(eig.values), rev=true)
    eigvecs = real.(eig.vectors[:, idx])

    # Extract first r eigenvectors as beta
    beta_raw = eigvecs[:, 1:r]

    # Phillips normalization: first r rows form identity
    beta = beta_raw * robust_inv(beta_raw[1:r, :])

    # Alpha from the moment matrices
    alpha = S01 * beta * robust_inv(beta' * S11 * beta)

    # Recover Gamma and mu via OLS
    ecm = Y_lag * beta  # Error correction terms (T_eff × r)

    # Build full RHS: [ECM, dY_lags, deterministic]
    RHS = ecm
    if size(dY_lags, 2) > 0
        RHS = hcat(RHS, dY_lags)
    end

    has_const = deterministic ∈ (:constant, :trend)
    has_trend = deterministic == :trend
    if has_const
        RHS = hcat(RHS, ones(T, T_eff))
    end
    if has_trend
        RHS = hcat(RHS, T.(1:T_eff))
    end

    # OLS for full system
    B_full = robust_inv(RHS'RHS) * (RHS' * dY_eff)
    U = dY_eff - RHS * B_full

    # Extract components
    col = 1
    alpha_ols = Matrix{T}(B_full[col:col+r-1, :]')  # n × r
    col += r

    Gamma = Matrix{T}[]
    for _ in 1:(p-1)
        push!(Gamma, Matrix{T}(B_full[col:col+n-1, :]'))
        col += n
    end

    mu = if has_const
        vec(B_full[col, :])
    else
        zeros(T, n)
    end

    # Pi = alpha * beta'
    Pi = alpha_ols * beta'

    # Residual covariance
    Sigma = (U'U) / T_eff

    # Log-likelihood and information criteria
    log_det = logdet_safe(Sigma)
    k_total = r * n + n * n * (p - 1) + (has_const ? n : 0) + (has_trend ? n : 0)
    loglik = -T(T_eff * n / 2) * log(T(2π)) - T(T_eff / 2) * log_det - T(T_eff * n / 2)
    aic_val = log_det + 2 * k_total / T_eff
    bic_val = log_det + k_total * log(T_eff) / T_eff
    hqic_val = log_det + 2 * k_total * log(log(T_eff)) / T_eff

    VECMModel{T}(
        Matrix{T}(Y), p, r, alpha_ols, beta, Pi, Gamma, mu, U, Sigma,
        aic_val, bic_val, hqic_val, loglik,
        deterministic, :johansen, joh
    )
end

@float_fallback estimate_vecm Y

# =============================================================================
# Rank Selection
# =============================================================================

"""
    select_vecm_rank(Y, p; criterion=:trace, significance=0.05) -> Int

Select cointegrating rank using the Johansen trace or max-eigenvalue test.
"""
function select_vecm_rank(Y::AbstractMatrix{T}, p::Int;
                          criterion::Symbol=:trace,
                          significance::Real=0.05,
                          deterministic::Symbol=:constant) where {T<:AbstractFloat}
    criterion ∈ (:trace, :max_eigen) ||
        throw(ArgumentError("criterion must be :trace or :max_eigen"))

    joh = johansen_test(Y, p; deterministic=deterministic)

    if criterion == :trace
        _select_rank_trace(joh, significance)
    else
        _select_rank_max_eigen(joh, significance)
    end
end

select_vecm_rank(Y::AbstractMatrix, p::Int; kwargs...) = select_vecm_rank(Float64.(Y), p; kwargs...)

# =============================================================================
# VAR Conversion
# =============================================================================

"""
    to_var(vecm::VECMModel) -> VARModel

Convert VECM to VAR in levels representation.

The VECM: ΔYₜ = ΠYₜ₋₁ + Σᵢ ΓᵢΔYₜ₋ᵢ + μ + uₜ
maps to VAR: Yₜ = c + A₁Yₜ₋₁ + ... + AₚYₜ₋ₚ + uₜ

with:
- A₁ = Π + Iₙ + Γ₁
- Aᵢ = Γᵢ - Γᵢ₋₁  for i = 2, ..., p-1
- Aₚ = -Γₚ₋₁
"""
function to_var(vecm::VECMModel{T}) where {T}
    n = nvars(vecm)
    p = vecm.p

    # Build A coefficient matrices
    A = Vector{Matrix{T}}(undef, p)
    In = Matrix{T}(I, n, n)

    if p == 1
        # Only Pi, no Gamma
        A[1] = vecm.Pi + In
    else
        A[1] = vecm.Pi + In + vecm.Gamma[1]
        for i in 2:(p-1)
            A[i] = vecm.Gamma[i] - vecm.Gamma[i-1]
        end
        A[p] = -vecm.Gamma[p-1]
    end

    # Stack into B matrix: [intercept; vec(A₁'); ...; vec(Aₚ')]
    k = 1 + n * p
    B = Matrix{T}(undef, k, n)
    B[1, :] = vecm.mu

    for i in 1:p
        rows = (2 + (i-1)*n):(1 + i*n)
        B[rows, :] = A[i]'
    end

    # Compute residuals from VAR representation
    Y_eff, X = construct_var_matrices(vecm.Y, p)
    U = Y_eff - X * B

    # Covariance and information criteria
    T_eff = size(U, 1)
    Sigma = (U'U) / T_eff
    log_det = logdet_safe(Sigma)
    aic_val = log_det + 2 * k / T_eff
    bic_val = log_det + k * log(T_eff) / T_eff
    hqic_val = log_det + 2 * k * log(log(T_eff)) / T_eff

    VARModel(vecm.Y, p, B, U, Sigma, aic_val, bic_val, hqic_val)
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

StatsAPI.coef(m::VECMModel) = vcat(vec(m.alpha), [vec(G) for G in m.Gamma]..., m.mu)
StatsAPI.residuals(m::VECMModel) = m.U
StatsAPI.nobs(m::VECMModel) = size(m.Y, 1)
StatsAPI.aic(m::VECMModel) = m.aic
StatsAPI.bic(m::VECMModel) = m.bic
StatsAPI.loglikelihood(m::VECMModel) = m.loglik
StatsAPI.islinear(::VECMModel) = true

function StatsAPI.dof(m::VECMModel)
    n = nvars(m)
    r = m.rank
    has_const = m.deterministic ∈ (:constant, :trend)
    has_trend = m.deterministic == :trend
    n_per_eq = r + n * (m.p - 1) + (has_const ? 1 : 0) + (has_trend ? 1 : 0)
    n * n_per_eq
end

function StatsAPI.predict(m::VECMModel{T}) where {T}
    # In-sample fitted values (in differences)
    dY = diff(m.Y, dims=1)
    dY_fitted = dY[m.p:end, :] - m.U
    dY_fitted
end

# =============================================================================
# Internal Helpers
# =============================================================================

function _select_rank_trace(joh::JohansenResult{T}, significance::Real) where {T}
    n = length(joh.trace_stats)
    # Use 5% column (index 2) by default
    cv_col = significance <= 0.01 ? 3 : significance <= 0.05 ? 2 : 1
    r = 0
    for i in 0:(n-1)
        if joh.trace_stats[i+1] > joh.critical_values_trace[i+1, cv_col]
            r = i + 1
        else
            break
        end
    end
    r
end

function _select_rank_max_eigen(joh::JohansenResult{T}, significance::Real) where {T}
    n = length(joh.max_eigen_stats)
    cv_col = significance <= 0.01 ? 3 : significance <= 0.05 ? 2 : 1
    r = 0
    for i in 0:(n-1)
        if joh.max_eigen_stats[i+1] > joh.critical_values_max[i+1, cv_col]
            r = i + 1
        else
            break
        end
    end
    r
end

function _build_deterministic_Z(::Type{T}, T_eff::Int, dY_lags::Matrix{T}, deterministic::Symbol) where {T}
    if deterministic == :none
        dY_lags
    elseif deterministic == :constant
        isempty(dY_lags) ? ones(T, T_eff, 1) : hcat(ones(T, T_eff), dY_lags)
    else  # :trend
        trend = T.(1:T_eff)
        isempty(dY_lags) ? hcat(ones(T, T_eff), trend) : hcat(ones(T, T_eff), trend, dY_lags)
    end
end

function _vecm_rank_zero(Y::Matrix{T}, p::Int, n::Int, deterministic::Symbol,
                         joh::JohansenResult{T}) where {T}
    # No cointegration: model in first differences
    dY = diff(Y, dims=1)
    T_obs = size(Y, 1)
    T_eff = T_obs - p

    dY_lags = if p > 1
        hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...)
    else
        Matrix{T}(undef, T_eff, 0)
    end
    dY_eff = dY[p:end, :]

    has_const = deterministic ∈ (:constant, :trend)
    has_trend = deterministic == :trend

    RHS = dY_lags
    if has_const
        RHS = isempty(RHS) ? ones(T, T_eff, 1) : hcat(RHS, ones(T, T_eff))
    end
    if has_trend
        RHS = hcat(RHS, T.(1:T_eff))
    end

    if isempty(RHS) || size(RHS, 2) == 0
        U = copy(dY_eff)
        mu = zeros(T, n)
        Gamma = Matrix{T}[]
    else
        B_full = robust_inv(RHS'RHS) * (RHS' * dY_eff)
        U = dY_eff - RHS * B_full

        col = 1
        Gamma = Matrix{T}[]
        for _ in 1:(p-1)
            push!(Gamma, Matrix{T}(B_full[col:col+n-1, :]'))
            col += n
        end

        mu = has_const ? vec(B_full[col, :]) : zeros(T, n)
    end

    Sigma = (U'U) / T_eff
    log_det = logdet_safe(Sigma)
    k_total = n * n * (p - 1) + (has_const ? n : 0) + (has_trend ? n : 0)
    loglik = -T(T_eff * n / 2) * log(T(2π)) - T(T_eff / 2) * log_det - T(T_eff * n / 2)
    aic_val = log_det + 2 * k_total / T_eff
    bic_val = log_det + k_total * log(T_eff) / T_eff
    hqic_val = log_det + 2 * k_total * log(log(T_eff)) / T_eff

    VECMModel{T}(
        Y, p, 0,
        zeros(T, n, 0), zeros(T, n, 0), zeros(T, n, n),
        Gamma, mu, U, Sigma,
        aic_val, bic_val, hqic_val, loglik,
        deterministic, :johansen, joh
    )
end

function _estimate_vecm_engle_granger(Y::Matrix{T}, p::Int,
                                      deterministic::Symbol,
                                      rank_input::Union{Symbol,Int}) where {T}
    T_obs, n = size(Y)
    n < 2 && throw(ArgumentError("Engle-Granger requires at least 2 variables"))

    r = rank_input === :auto ? 1 : rank_input::Int
    r != 1 && throw(ArgumentError("Engle-Granger method only supports rank=1"))

    # Step 1: Static OLS cointegrating regression (Y₁ on Y₂,...,Yₙ)
    y_dep = Y[:, 1]
    x_regs = hcat(ones(T, T_obs), Y[:, 2:end])
    beta_ols = x_regs \ y_dep
    resid_coint = y_dep - x_regs * beta_ols

    # Cointegrating vector: [1, -β₂, ..., -βₙ] (normalized on first variable)
    beta = Matrix{T}(undef, n, 1)
    beta[1, 1] = one(T)
    beta[2:end, 1] = -beta_ols[2:end]

    # Step 2: Estimate VECM with estimated ECM term
    dY = diff(Y, dims=1)
    T_eff = T_obs - p

    ecm = resid_coint[p:end-1]  # ECM_{t-1}
    dY_eff = dY[p:end, :]

    dY_lags = if p > 1
        hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...)
    else
        Matrix{T}(undef, T_eff, 0)
    end

    has_const = deterministic ∈ (:constant, :trend)
    has_trend = deterministic == :trend

    RHS = reshape(ecm, :, 1)
    if size(dY_lags, 2) > 0
        RHS = hcat(RHS, dY_lags)
    end
    if has_const
        RHS = hcat(RHS, ones(T, T_eff))
    end
    if has_trend
        RHS = hcat(RHS, T.(1:T_eff))
    end

    B_full = robust_inv(RHS'RHS) * (RHS' * dY_eff)
    U = dY_eff - RHS * B_full

    # Extract alpha (adjustment speeds)
    alpha = Matrix{T}(B_full[1:1, :]')  # n × 1

    col = 2
    Gamma = Matrix{T}[]
    for _ in 1:(p-1)
        push!(Gamma, Matrix{T}(B_full[col:col+n-1, :]'))
        col += n
    end

    mu = has_const ? vec(B_full[col, :]) : zeros(T, n)

    Pi = alpha * beta'
    Sigma = (U'U) / T_eff

    log_det = logdet_safe(Sigma)
    k_total = n + n * n * (p - 1) + (has_const ? n : 0) + (has_trend ? n : 0)
    loglik = -T(T_eff * n / 2) * log(T(2π)) - T(T_eff / 2) * log_det - T(T_eff * n / 2)
    aic_val = log_det + 2 * k_total / T_eff
    bic_val = log_det + k_total * log(T_eff) / T_eff
    hqic_val = log_det + 2 * k_total * log(log(T_eff)) / T_eff

    VECMModel{T}(
        Y, p, 1, alpha, beta, Pi, Gamma, mu, U, Sigma,
        aic_val, bic_val, hqic_val, loglik,
        deterministic, :engle_granger, nothing
    )
end
