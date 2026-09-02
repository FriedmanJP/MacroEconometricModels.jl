# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Heteroskedasticity-based SVAR identification (Lewis 2025, Section 3).

Exploits time-varying second moments (changes in the volatility regime) to identify
structural shocks without distributional assumptions. Methods: Markov-switching,
GARCH, smooth transition, external volatility instruments.

References:
- Lewis, D. J. (2025). "Identification based on higher moments in macroeconometrics."
- Lewis, D. J. (2021). "Identifying shocks via time-varying volatility."
- Sentana, E. & Fiorentini, G. (2001). "Identification, estimation and testing of conditionally heteroskedastic factor models."
- Rigobon, R. (2003). "Identification through heteroskedasticity."
- Lanne, M. & Lütkepohl, H. (2008). "Identifying monetary policy shocks via changes in volatility."
- Normandin, M. & Phaneuf, L. (2004). "Monetary policy shocks."
- Lütkepohl, H. & Netšunajev, A. (2017). "Structural vector autoregressions with smooth transition in variances."
"""

using LinearAlgebra, Statistics, Distributions, Random
import Optim
import ForwardDiff

# =============================================================================
# Result Types
# =============================================================================

"""
    MarkovSwitchingSVARResult{T} <: AbstractNonGaussianSVAR

Result from Markov-switching heteroskedasticity SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix
- `Q::Matrix{T}` — rotation matrix
- `Sigma_regimes::Vector{Matrix{T}}` — covariance per regime
- `Lambda::Vector{Vector{T}}` — relative variances per regime
- `regime_probs::Matrix{T}` — smoothed regime probabilities (T × K)
- `transition_matrix::Matrix{T}` — Markov transition probabilities (K × K)
- `loglik::T`
- `converged::Bool`
- `iterations::Int`
- `n_regimes::Int`
- `se::Matrix{T}` — delta-method SEs of B₀
- `vcov::Matrix{T}` — parameter-space covariance (θ_Givens, log Λ₂…K)
- `classification_quality::T` — mean of max smoothed regime probability
- `shocks::Matrix{T}` — structural shocks ``ε_t = B_0^{-1} u_t``
- `shock_names::Vector{String}` — labels for the n structural shocks
"""
struct MarkovSwitchingSVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    Sigma_regimes::Vector{Matrix{T}}
    Lambda::Vector{Vector{T}}
    regime_probs::Matrix{T}
    transition_matrix::Matrix{T}
    loglik::T
    converged::Bool
    iterations::Int
    n_regimes::Int
    se::Matrix{T}
    vcov::Matrix{T}
    classification_quality::T
    shocks::Matrix{T}
    shock_names::Vector{String}
end

function MarkovSwitchingSVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_probs,
                                       transition_matrix, loglik, converged,
                                       iterations, n_regimes, se, vcov,
                                       classification_quality, shocks) where {T<:AbstractFloat}
    n = size(B0, 1)
    MarkovSwitchingSVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_probs,
                                  transition_matrix, loglik, converged, iterations,
                                  n_regimes, se, vcov, classification_quality, shocks,
                                  _default_shock_names(n))
end

function MarkovSwitchingSVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_probs,
                                       transition_matrix, loglik, converged,
                                       iterations, n_regimes, se, vcov,
                                       classification_quality) where {T<:AbstractFloat}
    n = size(B0, 1)
    MarkovSwitchingSVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_probs,
                                  transition_matrix, loglik, converged, iterations,
                                  n_regimes, se, vcov, classification_quality,
                                  zeros(T, 0, 0), _default_shock_names(n))
end

function MarkovSwitchingSVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_probs,
                                       transition_matrix, loglik, converged,
                                       iterations, n_regimes) where {T<:AbstractFloat}
    n = size(B0, 1)
    MarkovSwitchingSVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_probs,
                                  transition_matrix, loglik, converged, iterations,
                                  n_regimes, fill(T(NaN), n, n), zeros(T, 0, 0), T(NaN))
end

function MarkovSwitchingSVARResult(B0, Q, Sigma_regimes, Lambda, regime_probs,
                                    transition_matrix, loglik, converged,
                                    iterations, n_regimes)
    T = eltype(B0)
    MarkovSwitchingSVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_probs,
                                  transition_matrix, loglik, converged, iterations,
                                  n_regimes)
end

function Base.show(io::IO, r::MarkovSwitchingSVARResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Variables"  n;
        "Regimes"    r.n_regimes;
        "Log-likelihood" _fmt(r.loglik; digits=4);
        "Classification" _fmt(r.classification_quality; digits=4);
        "Converged"  r.converged ? "Yes" : "No";
        "Iterations" r.iterations
    ]
    _pretty_table(io, spec;
        title = "Markov-Switching SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _show_B0_ses(io, r.B0, r.se)
end

"""
    GARCHSVARResult{T} <: AbstractNonGaussianSVAR

Result from GARCH-based SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix
- `Q::Matrix{T}` — rotation matrix
- `garch_params::Matrix{T}` — (n × 3): [ω, α, β] per shock
- `cond_var::Matrix{T}` — (T_eff × n) conditional variances
- `shocks::Matrix{T}` — structural shocks
- `loglik::T`
- `converged::Bool`
- `iterations::Int`
- `se::Matrix{T}` — delta-method SEs of B₀
- `vcov::Matrix{T}` — parameter-space covariance (θ_Givens, GARCH)
- `shock_names::Vector{String}` — labels for the n structural shocks
"""
struct GARCHSVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    garch_params::Matrix{T}
    cond_var::Matrix{T}
    shocks::Matrix{T}
    loglik::T
    converged::Bool
    iterations::Int
    se::Matrix{T}
    vcov::Matrix{T}
    shock_names::Vector{String}
end

function GARCHSVARResult{T}(B0, Q, garch_params, cond_var, shocks, loglik,
                             converged, iterations, se, vcov) where {T<:AbstractFloat}
    n = size(B0, 1)
    GARCHSVARResult{T}(B0, Q, garch_params, cond_var, shocks, loglik, converged,
                        iterations, se, vcov, _default_shock_names(n))
end

function GARCHSVARResult{T}(B0, Q, garch_params, cond_var, shocks, loglik,
                             converged, iterations) where {T<:AbstractFloat}
    n = size(B0, 1)
    GARCHSVARResult{T}(B0, Q, garch_params, cond_var, shocks, loglik, converged,
                        iterations, fill(T(NaN), n, n), zeros(T, 0, 0))
end

function GARCHSVARResult(B0, Q, garch_params, cond_var, shocks, loglik,
                          converged, iterations)
    T = eltype(B0)
    GARCHSVARResult{T}(B0, Q, garch_params, cond_var, shocks, loglik, converged,
                        iterations)
end

function Base.show(io::IO, r::GARCHSVARResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Variables"      n;
        "Log-likelihood" _fmt(r.loglik; digits=4);
        "Converged"      r.converged ? "Yes" : "No";
        "Iterations"     r.iterations
    ]
    _pretty_table(io, spec;
        title = "GARCH-SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    garch_data = Matrix{Any}(undef, n, 4)
    for i in 1:n
        garch_data[i, 1] = "Shock $i"
        garch_data[i, 2] = _fmt(r.garch_params[i, 1])
        garch_data[i, 3] = _fmt(r.garch_params[i, 2])
        garch_data[i, 4] = _fmt(r.garch_params[i, 3])
    end
    _pretty_table(io, garch_data;
        title = "GARCH Parameters",
        column_labels = ["", "ω", "α", "β"],
        alignment = [:l, :r, :r, :r],
    )
    _show_B0_ses(io, r.B0, r.se)
end

"""
    SmoothTransitionSVARResult{T} <: AbstractNonGaussianSVAR

Result from smooth-transition heteroskedasticity SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix
- `Q::Matrix{T}` — rotation matrix
- `Sigma_regimes::Vector{Matrix{T}}` — covariance matrices for extreme regimes
- `Lambda::Vector{Vector{T}}` — relative variances per regime
- `gamma::T` — transition speed parameter
- `threshold::T` — transition location parameter
- `transition_var::Vector{T}` — transition variable values
- `G_values::Vector{T}` — transition function G(s_t) values
- `loglik::T`
- `converged::Bool`
- `iterations::Int`
- `se::Matrix{T}` — delta-method SEs of B₀
- `vcov::Matrix{T}` — parameter-space covariance (vech L, θ, log Λ, xγ, c)
- `residuals::Matrix{T}` — VAR residuals used in the likelihood
- `shock_names::Vector{String}` — labels for the n structural shocks
"""
struct SmoothTransitionSVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    Sigma_regimes::Vector{Matrix{T}}
    Lambda::Vector{Vector{T}}
    gamma::T
    threshold::T
    transition_var::Vector{T}
    G_values::Vector{T}
    loglik::T
    converged::Bool
    iterations::Int
    se::Matrix{T}
    vcov::Matrix{T}
    residuals::Matrix{T}
    shock_names::Vector{String}
end

function SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                       transition_var, G_values, loglik, converged,
                                       iterations, se, vcov, residuals) where {T<:AbstractFloat}
    n = size(B0, 1)
    SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                  transition_var, G_values, loglik, converged,
                                  iterations, se, vcov, residuals,
                                  _default_shock_names(n))
end

function SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                       transition_var, G_values, loglik, converged,
                                       iterations) where {T<:AbstractFloat}
    n = size(B0, 1)
    SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                  transition_var, G_values, loglik, converged,
                                  iterations, fill(T(NaN), n, n), zeros(T, 0, 0),
                                  zeros(T, 0, 0))
end

function SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                       transition_var, G_values, loglik, converged,
                                       iterations, se, vcov) where {T<:AbstractFloat}
    n = size(B0, 1)
    se_m, vcov_m = _coerce_se_vcov(se, vcov, n, T)
    SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                  transition_var, G_values, loglik, converged,
                                  iterations, se_m, vcov_m, zeros(T, 0, 0))
end

function SmoothTransitionSVARResult(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                    transition_var, G_values, loglik, converged,
                                    iterations)
    T = eltype(B0)
    SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                  transition_var, G_values, loglik, converged,
                                  iterations)
end

function Base.show(io::IO, r::SmoothTransitionSVARResult{T}) where {T}
    n = size(r.B0, 1)
    se_γ, se_c = _st_gamma_threshold_se(r)
    spec = Any[
        "Variables"      n;
        "γ (speed)"      _fmt(r.gamma; digits=2);
        "γ SE"           se_γ === nothing ? "—" : _fmt(se_γ; digits=2);
        "Threshold"      _fmt(r.threshold; digits=4);
        "Threshold SE"   se_c === nothing ? "—" : _fmt(se_c; digits=4);
        "Log-likelihood" _fmt(r.loglik; digits=4);
        "Converged"      r.converged ? "Yes" : "No";
        "Iterations"     r.iterations
    ]
    _pretty_table(io, spec;
        title = "Smooth-Transition SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _show_B0_ses(io, r.B0, r.se)
end

"""
    ExternalVolatilitySVARResult{T} <: AbstractNonGaussianSVAR

Result from external volatility instrument SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix
- `Q::Matrix{T}` — rotation matrix
- `Sigma_regimes::Vector{Matrix{T}}` — covariance per regime
- `Lambda::Vector{Vector{T}}` — relative variances per regime
- `regime_indices::Vector{Vector{Int}}` — observation indices per regime
- `loglik::T`
- `se::Matrix{T}` — delta-method SEs of B₀
- `vcov::Matrix{T}` — parameter-space covariance (θ_Givens, log Λ₂…K)
- `shocks::Matrix{T}` — structural shocks ``ε_t = B_0^{-1} u_t``
- `shock_names::Vector{String}` — labels for the n structural shocks
"""
struct ExternalVolatilitySVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    Sigma_regimes::Vector{Matrix{T}}
    Lambda::Vector{Vector{T}}
    regime_indices::Vector{Vector{Int}}
    loglik::T
    se::Matrix{T}
    vcov::Matrix{T}
    shocks::Matrix{T}
    shock_names::Vector{String}
end

function ExternalVolatilitySVARResult{T}(B0, Q, Sigma_regimes, Lambda,
                                          regime_indices, loglik, se,
                                          vcov, shocks) where {T<:AbstractFloat}
    n = size(B0, 1)
    ExternalVolatilitySVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_indices,
                                     loglik, se, vcov, shocks, _default_shock_names(n))
end

function ExternalVolatilitySVARResult{T}(B0, Q, Sigma_regimes, Lambda,
                                          regime_indices, loglik, se,
                                          vcov) where {T<:AbstractFloat}
    n = size(B0, 1)
    ExternalVolatilitySVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_indices,
                                     loglik, se, vcov, zeros(T, 0, 0),
                                     _default_shock_names(n))
end

function ExternalVolatilitySVARResult{T}(B0, Q, Sigma_regimes, Lambda,
                                          regime_indices, loglik) where {T<:AbstractFloat}
    n = size(B0, 1)
    ExternalVolatilitySVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_indices,
                                     loglik, fill(T(NaN), n, n), zeros(T, 0, 0))
end

function ExternalVolatilitySVARResult(B0, Q, Sigma_regimes, Lambda, regime_indices, loglik)
    T = eltype(B0)
    ExternalVolatilitySVARResult{T}(B0, Q, Sigma_regimes, Lambda, regime_indices, loglik)
end

function Base.show(io::IO, r::ExternalVolatilitySVARResult{T}) where {T}
    n = size(r.B0, 1)
    K = length(r.Sigma_regimes)
    spec = Any[
        "Variables"      n;
        "Regimes"        K;
        "Log-likelihood" _fmt(r.loglik; digits=4)
    ]
    _pretty_table(io, spec;
        title = "External Volatility SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _show_B0_ses(io, r.B0, r.se)
end

# =============================================================================
# Display / SE helpers
# =============================================================================

function _show_B0_ses(io::IO, B0::AbstractMatrix{T}, se::AbstractMatrix) where {T}
    n = size(B0, 1)
    se_m = size(se) == (n, n) ? Matrix{T}(se) : fill(T(NaN), n, n)
    names = ["B₀[$i,$j]" for j in 1:n for i in 1:n]
    _coef_table(io, "Structural Impact Matrix (B₀)", names, vec(B0), vec(se_m); dist=:z)
end

function _coerce_se_vcov(se, vcov, n::Int, ::Type{T}) where {T<:AbstractFloat}
    se_m = if se === nothing
        fill(T(NaN), n, n)
    elseif se isa AbstractMatrix && size(se) == (n, n)
        Matrix{T}(se)
    else
        fill(T(NaN), n, n)
    end
    vcov_m = if vcov === nothing || !(vcov isa AbstractMatrix)
        zeros(T, 0, 0)
    else
        Matrix{T}(vcov)
    end
    se_m, vcov_m
end

function _st_gamma_threshold_se(r::SmoothTransitionSVARResult{T}) where {T<:AbstractFloat}
    n = size(r.B0, 1)
    n_L = n * (n + 1) ÷ 2
    n_angles = n * (n - 1) ÷ 2
    pdim = n_L + n_angles + n + 2
    size(r.vcov) == (pdim, pdim) || return nothing, nothing
    idx_xγ = n_L + n_angles + n + 1
    idx_c = pdim
    se_c = sqrt(max(r.vcov[idx_c, idx_c], zero(T)))
    σs = std(r.transition_var)
    σs > 0 || return nothing, se_c
    logγ_lo = log(T(1e-3) / σs)
    logγ_hi = log(T(20) / σs)
    xγ = _st_x_from_gamma(r.gamma, logγ_lo, logγ_hi)
    dγ = ForwardDiff.derivative(x -> _st_gamma_from_x(x, logγ_lo, logγ_hi), xγ)
    se_γ = abs(dγ) * sqrt(max(r.vcov[idx_xγ, idx_xγ], zero(T)))
    se_γ, se_c
end

function _finite_se!(se::AbstractMatrix{T}) where {T<:AbstractFloat}
    @inbounds for i in eachindex(se)
        v = se[i]
        se[i] = isfinite(v) && v >= zero(T) ? v : T(NaN)
    end
    se
end

function _spd_inv(H::AbstractMatrix{T}) where {T<:AbstractFloat}
    Hs = Matrix{T}((H + H') / 2)
    λmin = try
        eigmin(Symmetric(Hs))
    catch
        T(-1)
    end
    if !isfinite(λmin) || λmin <= T(1e-10)
        jitter = T(1e-8)
        if isfinite(λmin) && λmin < zero(T)
            jitter += -λmin
        end
        Hs = Hs + jitter * I
    end
    Matrix{T}(robust_inv(Hs; silent=true))
end

function _delta_B0_se(params::Vector{T}, V::AbstractMatrix{T}, B0_fn, n::Int) where {T<:AbstractFloat}
    if size(V, 1) == 0 || size(V, 2) == 0 || all(x -> !isfinite(x) || iszero(x), V)
        return fill(T(NaN), n, n), zeros(T, 0, 0)
    end
    J = ForwardDiff.jacobian(p -> vec(B0_fn(p)), params)
    VB = J * V * J'
    se = reshape(sqrt.(max.(diag(VB), zero(T))), n, n)
    _finite_se!(se)
    se, Matrix{T}(VB)
end

# =============================================================================
# K-regime joint ML kernel (Lanne–Lütkepohl–Maciejowska 2010)
# =============================================================================

"""Two-regime start; falls back to an unchecked eigen if λ-gap is tiny."""
function _two_regime_start(Σ1::Matrix{T}, Σ2::Matrix{T}) where {T<:AbstractFloat}
    try
        return _eigendecomposition_id(Σ1, Σ2)
    catch e
        e isa IdentificationError || rethrow()
        L1 = Matrix{T}(safe_cholesky(Σ1; silent=true))
        M = Symmetric(L1 \ Σ2 / L1')
        E = eigen(M)
        λ = max.(real.(E.values), T(1e-4))
        Q = real.(E.vectors)
        idx = sortperm(λ)
        λ = λ[idx]
        Q = Q[:, idx]
        B0 = L1 * Q
        return B0, Q, λ
    end
end

"""`logdet + tr(Σ⁻¹ S)` with a large penalty if `Σ` is not PD. Dual-friendly."""
function _logdet_tr(Σ, S, Tθ)
    Hs = (Σ + Σ') * Tθ(0.5)
    d = det(Hs)
    d > zero(d) || return Tθ(1e8) + abs(d)
    log(d) + tr(Hs \ S)
end

function _logdet_quad(Σ, u, n, Tθ)
    Hs = (Σ + Σ') * Tθ(0.5)
    d = det(Hs)
    d > zero(d) || return Tθ(1e8) + abs(d)
    log(d) + dot(u, Hs \ u)
end

function _k_regime_B_from_params(params, n::Int)
    Tθ = eltype(params)
    B = Matrix{Tθ}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        B[i, j] = params[(j - 1) * n + i]
    end
    B
end

"""Negative concentrated K-regime criterion. Dual-friendly. Free `B`, `Λ₁ = I`."""
function _k_regime_nll(params, n::Int, K::Int, Tks, Sigma_hats)
    Tθ = eltype(params)
    B = _k_regime_B_from_params(params, n)
    nB = n * n
    nll = zero(Tθ)
    ridge = Tθ(1e-10)
    floor_λ = Tθ(1e-12)
    for k in 1:K
        if k == 1
            Σ = B * B'
        else
            i0 = nB + (k - 2) * n
            λ = Vector{Tθ}(undef, n)
            @inbounds for j in 1:n
                λ[j] = max(exp(params[i0 + j]), floor_λ)
            end
            Σ = B * Diagonal(λ) * B'
        end
        Σ = Σ + ridge * I
        nll += Tθ(Tks[k]) * _logdet_tr(Σ, Sigma_hats[k], Tθ)
    end
    nll
end

function _k_regime_B0(params, n::Int)
    _k_regime_B_from_params(params, n)
end

function _k_regime_pack(B, Lambdas, n::Int, K::Int)
    T = eltype(B)
    p = vec(Matrix{T}(B))
    for k in 2:K
        append!(p, log.(max.(Lambdas[k], T(1e-12))))
    end
    p
end

function _k_regime_unpack(params::AbstractVector, n::Int, K::Int)
    Tθ = eltype(params)
    B = _k_regime_B_from_params(params, n)
    nB = n * n
    Lambdas = Vector{Vector{Tθ}}(undef, K)
    Lambdas[1] = ones(Tθ, n)
    for k in 2:K
        i0 = nB + (k - 2) * n
        Lambdas[k] = [exp(params[i0 + j]) for j in 1:n]
    end
    Q = Matrix{Tθ}(I, n, n)
    try
        L = cholesky(Hermitian(B * B')).L
        Q = Matrix(L) \ B
    catch
    end
    B, Q, Lambdas
end

function _k_regime_normalize(B, Q, Lambdas, n::Int, K::Int)
    idx = sortperm(Lambdas[2])
    B = B[:, idx]
    Q = Q[:, idx]
    Lambdas = [λ[idx] for λ in Lambdas]
    for j in 1:n
        if B[j, j] < 0
            B[:, j] .*= -one(eltype(B))
            Q[:, j] .*= -one(eltype(Q))
        end
    end
    B, Q, Lambdas, idx
end

"""Joint ML of free `B` with `Λ₁ = I` and `Λ₂…K` given regime covariances."""
function _k_regime_ml(Sigma_hats::Vector{Matrix{T}}, Tks::AbstractVector;
                      max_iter::Int=200, tol::T=T(1e-8)) where {T<:AbstractFloat}
    K = length(Sigma_hats)
    n = size(Sigma_hats[1], 1)
    K >= 2 || throw(ArgumentError("need at least 2 regimes"))
    B0, Q, λ = _two_regime_start(Matrix{T}(Sigma_hats[1]), Matrix{T}(Sigma_hats[2]))
    Lambdas = Vector{Vector{T}}(undef, K)
    Lambdas[1] = ones(T, n)
    Lambdas[2] = max.(λ, T(1e-8))
    B0i = robust_inv(B0; silent=true)
    for k in 3:K
        Lambdas[k] = max.(diag(B0i * Sigma_hats[k] * B0i'), T(1e-8))
    end
    params0 = _k_regime_pack(B0, Lambdas, n, K)
    Tks_T = T[T(t) for t in Tks]
    obj = p -> begin
        v = try
            _k_regime_nll(p, n, K, Tks_T, Sigma_hats)
        catch
            T(1e20)
        end
        isfinite(v) ? v : T(1e20)
    end
    conv = true
    nit = 0
    p = params0
    # K=2 identification is the eigendecomposition (Lanne–Lütkepohl). LBFGS on
    # Optim v1 can report a lower nll at a spurious rotation (Julia 1.10 CI).
    if K > 2
        g! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
        nll0 = obj(params0)
        result = Optim.optimize(obj, g!, params0, Optim.LBFGS(),
                                Optim.Options(iterations=max_iter, g_tol=tol,
                                              f_reltol=T(1e-12)))
        p_opt = Vector{T}(Optim.minimizer(result))
        p = obj(p_opt) <= nll0 ? p_opt : params0
        conv = Optim.converged(result)
        nit = Optim.iterations(result)
    end
    B, Q, Lambdas = _k_regime_unpack(p, n, K)
    B, Q, Lambdas, idx = _k_regime_normalize(B, Q, Lambdas, n, K)
    Lfac = Matrix{T}(safe_cholesky(B * B'; silent=true))
    Q = Lfac \ B
    p = _k_regime_pack(B, Lambdas, n, K)
    (B, Q, Lambdas, p, Lfac, conv, nit, idx)
end

function _k_regime_lambda(params, n::Int, k::Int)
    Tθ = eltype(params)
    nB = n * n
    k == 1 && return ones(Tθ, n)
    i0 = nB + (k - 2) * n
    [max(exp(params[i0 + j]), Tθ(1e-12)) for j in 1:n]
end

"""Full Gaussian nll of the K-regime SVAR. Dual-friendly."""
function _k_regime_full_nll(params, U, regimes, n::Int, K::Int)
    Tθ = eltype(params)
    B = _k_regime_B_from_params(params, n)
    ridge = Tθ(1e-10)
    log2π = log(Tθ(2) * Tθ(π))
    half = Tθ(0.5)
    Σinv = Vector{Matrix{Tθ}}(undef, K)
    ld = Vector{Tθ}(undef, K)
    for k in 1:K
        λ = _k_regime_lambda(params, n, k)
        Σ = B * Diagonal(λ) * B' + ridge * I
        Hs = (Σ + Σ') * Tθ(0.5)
        d = det(Hs)
        if d <= zero(d)
            ld[k] = Tθ(1e8)
            Σinv[k] = Matrix{Tθ}(I, n, n)
        else
            ld[k] = log(d)
            Σinv[k] = Hs \ I
        end
    end
    nll = zero(Tθ)
    T_obs = size(U, 1)
    @inbounds for t in 1:T_obs
        k = regimes[t]
        u = view(U, t, :)
        nll += half * (Tθ(n) * log2π + ld[k] + dot(u, Σinv[k] * u))
    end
    nll
end

function _k_regime_obs_nll(params, u, k::Int, n::Int, K::Int)
    Tθ = eltype(params)
    B = _k_regime_B_from_params(params, n)
    λ = _k_regime_lambda(params, n, k)
    Σ = B * Diagonal(λ) * B' + Tθ(1e-10) * I
    half = Tθ(0.5)
    log2π = log(Tθ(2) * Tθ(π))
    half * (Tθ(n) * log2π + _logdet_quad(Σ, u, n, Tθ))
end

function _sandwich_meat(params::Vector{T}, U, regimes, n::Int, K::Int) where {T<:AbstractFloat}
    T_obs = size(U, 1)
    p = length(params)
    meat = zeros(T, p, p)
    for t in 1:T_obs
        g = ForwardDiff.gradient(
            θ -> _k_regime_obs_nll(θ, view(U, t, :), regimes[t], n, K), params)
        meat .+= g * g'
    end
    meat
end

function _external_vcov(p::Vector{T}, U, regimes, n::Int, K::Int) where {T<:AbstractFloat}
    nll = θ -> _k_regime_full_nll(θ, U, regimes, n, K)
    H = Matrix{T}(ForwardDiff.hessian(nll, p))
    Hinv = _spd_inv(H)
    meat = try
        _sandwich_meat(p, U, regimes, n, K)
    catch
        return Hinv
    end
    V = Hinv * meat * Hinv
    any(!isfinite, V) ? Hinv : Matrix{T}((V + V') / 2)
end

function _k_regime_conc_ll(B, Lambdas, Tks, Sigma_hats)
    T = eltype(B)
    n = size(B, 1)
    K = length(Tks)
    ll = zero(T)
    log2π = log(T(2) * T(π))
    for k in 1:K
        Σ = B * Diagonal(Lambdas[k]) * B' + T(1e-10) * I
        ll -= T(0.5) * T(Tks[k]) * (T(n) * log2π + _logdet_tr(Σ, Sigma_hats[k], T))
    end
    ll
end

"""Restricted K-regime ML with zeros on B₀. Dual-friendly objective."""
function _k_regime_restricted_ml(B0::Matrix{T}, Lambdas, mask::BitMatrix,
                                  Sigma_hats, Tks; max_iter::Int=200) where {T<:AbstractFloat}
    n = size(B0, 1)
    K = length(Tks)
    free = .!mask
    n_free = count(free)
    B_start = copy(B0)
    B_start[mask] .= zero(T)
    logΛ0 = T[]
    for k in 2:K
        append!(logΛ0, log.(max.(Lambdas[k], T(1e-8))))
    end
    p0 = vcat(B_start[free], logΛ0)
    Tks_T = T[T(t) for t in Tks]
    obj = p -> begin
        Tθ = eltype(p)
        B = zeros(Tθ, n, n)
        B[free] = p[1:n_free]
        nll = zero(Tθ)
        for k in 1:K
            if k == 1
                λ = ones(Tθ, n)
            else
                i0 = n_free + (k - 2) * n
                λ = [max(exp(p[i0 + j]), Tθ(1e-12)) for j in 1:n]
            end
            Σ = B * Diagonal(λ) * B' + Tθ(1e-10) * I
            nll += Tθ(Tks_T[k]) * _logdet_tr(Σ, Sigma_hats[k], Tθ)
        end
        isfinite(nll) ? nll : Tθ(1e20)
    end
    g! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
    res = Optim.optimize(obj, g!, p0, Optim.LBFGS(),
                         Optim.Options(iterations=max_iter, g_tol=T(1e-8),
                                       allow_f_increases=true))
    p = Vector{T}(Optim.minimizer(res))
    B = zeros(T, n, n)
    B[free] = p[1:n_free]
    Lambdas_r = Vector{Vector{T}}(undef, K)
    Lambdas_r[1] = ones(T, n)
    for k in 2:K
        i0 = n_free + (k - 2) * n
        Lambdas_r[k] = exp.(p[i0+1:i0+n])
    end
    _k_regime_conc_ll(B, Lambdas_r, Tks_T, Sigma_hats), B
end

function _st_restricted_nll(params, U, s, sigma_s, n::Int, logγ_lo, logγ_hi,
                             free::BitMatrix, n_free::Int)
    Tθ = eltype(params)
    B = zeros(Tθ, n, n)
    B[free] = params[1:n_free]
    logΛ = view(params, (n_free + 1):(n_free + n))
    xγ = params[n_free + n + 1]
    cc = params[n_free + n + 2]
    γ = _st_gamma_from_x(xγ, logγ_lo, logγ_hi)
    half = Tθ(0.5)
    log2π = log(Tθ(2) * Tθ(π))
    floor_d = Tθ(1e-12)
    nll = zero(Tθ)
    T_obs = size(U, 1)
    ridge = Tθ(1e-10)
    for t in 1:T_obs
        G_t = _logistic_transition(s[t], γ, cc, sigma_s)
        λterm = Vector{Tθ}(undef, n)
        @inbounds for j in 1:n
            λj = exp(logΛ[j])
            λterm[j] = max(one(Tθ) + G_t * (λj - one(Tθ)), floor_d)
        end
        Σ = B * Diagonal(λterm) * B' + ridge * I
        u = view(U, t, :)
        nll += half * (Tθ(n) * log2π + _logdet_quad(Σ, u, n, Tθ))
    end
    nll
end

function _garch_restricted_nll(params, U, cond_var, free::BitMatrix, n_free::Int, n::Int)
    Tθ = eltype(params)
    B = zeros(Tθ, n, n)
    B[free] = params[1:n_free]
    shocks = U * (B \ I)'
    T_obs = size(U, 1)
    nll = zero(Tθ)
    log2π = log(Tθ(2) * Tθ(π))
    half = Tθ(0.5)
    @inbounds for t in 1:T_obs, j in 1:n
        nll += half * (log2π + log(cond_var[t, j]) + shocks[t, j]^2 / cond_var[t, j])
    end
    nll += Tθ(T_obs) * log(max(abs(det(B)), Tθ(1e-16)))
    nll
end

function _ms_complete_nll(params, U, smoothed, n::Int, K::Int)
    Tθ = eltype(params)
    B = _k_regime_B_from_params(params, n)
    ridge = Tθ(1e-10)
    log2π = log(Tθ(2) * Tθ(π))
    half = Tθ(0.5)
    Σinv = Vector{Matrix{Tθ}}(undef, K)
    ld = Vector{Tθ}(undef, K)
    for k in 1:K
        λ = _k_regime_lambda(params, n, k)
        Σ = B * Diagonal(λ) * B' + ridge * I
        Hs = (Σ + Σ') * Tθ(0.5)
        d = det(Hs)
        if d <= zero(d)
            ld[k] = Tθ(1e8)
            Σinv[k] = Matrix{Tθ}(I, n, n)
        else
            ld[k] = log(d)
            Σinv[k] = Hs \ I
        end
    end
    nll = zero(Tθ)
    T_obs = size(U, 1)
    @inbounds for t in 1:T_obs
        u = view(U, t, :)
        for k in 1:K
            w = Tθ(smoothed[t, k])
            w <= 0 && continue
            nll += w * half * (Tθ(n) * log2π + ld[k] + dot(u, Σinv[k] * u))
        end
    end
    nll
end

# =============================================================================
# Hamilton Filter/Smoother
# =============================================================================

"""Hamilton (1989) forward filter for Markov-switching model."""
function _hamilton_filter(U::Matrix{T}, Sigma_regimes::Vector{Matrix{T}},
                          transition_matrix::Matrix{T}) where {T<:AbstractFloat}
    T_obs, n = size(U)
    K = length(Sigma_regimes)

    # Precompute
    Sigma_invs = [robust_inv(S) for S in Sigma_regimes]
    logdet_Sigmas = [logdet_safe(S) for S in Sigma_regimes]

    filtered_probs = zeros(T, T_obs, K)
    predicted_probs = zeros(T, T_obs, K)
    loglik = zero(T)

    # Initial probabilities: ergodic distribution
    P = transition_matrix
    A = [P' - I; ones(T, 1, K)]
    b = [zeros(T, K); one(T)]
    xi_0 = try
        (A' * A) \ (A' * b)
    catch
        fill(one(T) / K, K)
    end

    predicted_probs[1, :] = xi_0

    for t in 1:T_obs
        u = @view U[t, :]
        eta = zeros(T, K)

        for k in 1:K
            eta[k] = exp(-T(0.5) * (n * log(T(2π)) + logdet_Sigmas[k] +
                     dot(u, Sigma_invs[k] * u)))
        end

        xi_pred = t == 1 ? xi_0 : predicted_probs[t, :]
        joint = xi_pred .* eta
        margin = sum(joint)
        margin = max(margin, eps(T))
        loglik += log(margin)

        filtered_probs[t, :] = joint / margin

        if t < T_obs
            predicted_probs[t + 1, :] = P' * filtered_probs[t, :]
        end
    end

    (filtered_probs, predicted_probs, loglik)
end

"""Kim (1994) backward smoother for Markov-switching model."""
function _hamilton_smoother(filtered_probs::Matrix{T}, predicted_probs::Matrix{T},
                            transition_matrix::Matrix{T}) where {T<:AbstractFloat}
    T_obs, K = size(filtered_probs)
    smoothed = zeros(T, T_obs, K)
    smoothed[T_obs, :] = filtered_probs[T_obs, :]

    P = transition_matrix
    for t in (T_obs-1):-1:1
        for i in 1:K
            s = zero(T)
            for j in 1:K
                pred_j = max(predicted_probs[t + 1, j], eps(T))
                s += P[i, j] * smoothed[t + 1, j] / pred_j
            end
            smoothed[t, i] = filtered_probs[t, i] * s
        end
        # Normalize
        total = sum(smoothed[t, :])
        if total > 0
            smoothed[t, :] /= total
        end
    end
    smoothed
end

"""EM step for Markov-switching covariances and transition matrix.

Uses Kim (1994) joint smoothed probabilities for the transition matrix update:
  ξ_{t-1,t|T}(i,j) = ξ_{t|T}(j) · P[i,j] · ξ_{t-1|t-1}(i) / ξ_{t|t-1}(j)
"""
function _ms_em_step(U::Matrix{T}, smoothed::Matrix{T}, filtered::Matrix{T},
                      predicted::Matrix{T}, P::Matrix{T}, K::Int) where {T<:AbstractFloat}
    T_obs, n = size(U)

    # Update covariance matrices (uses smoothed probabilities as weights)
    Sigma_new = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        w = max.(smoothed[:, k], eps(T))
        w_sum = sum(w)
        Sigma_k = zeros(T, n, n)
        for t in 1:T_obs
            u = @view U[t, :]
            Sigma_k .+= w[t] * (u * u')
        end
        Sigma_new[k] = Symmetric(Sigma_k / w_sum)
        # Regularize
        Sigma_new[k] = Sigma_new[k] + eps(T) * I
    end

    # Update transition matrix using Kim (1994) / Hamilton (1994 Ch.22)
    # joint smoothed probabilities instead of marginal products
    P_new = zeros(T, K, K)
    for t in 2:T_obs
        for i in 1:K
            for j in 1:K
                pred_j = max(predicted[t, j], eps(T))
                joint_ij = smoothed[t, j] * P[i, j] * filtered[t-1, i] / pred_j
                P_new[i, j] += joint_ij
            end
        end
    end
    # Normalize rows
    for i in 1:K
        row_sum = sum(P_new[i, :])
        if row_sum > 0
            P_new[i, :] /= row_sum
        else
            P_new[i, :] .= one(T) / K
        end
    end

    (Sigma_new, P_new)
end

# =============================================================================
# Public API: Markov-Switching
# =============================================================================

function _ms_chunk_init(U::Matrix{T}, K::Int) where {T<:AbstractFloat}
    T_obs, n = size(U)
    Sigma_regimes = Vector{Matrix{T}}(undef, K)
    chunk = max(T_obs ÷ K, 1)
    for k in 1:K
        idx_start = (k - 1) * chunk + 1
        idx_end = k == K ? T_obs : min(k * chunk, T_obs)
        U_k = U[idx_start:idx_end, :]
        Sigma_regimes[k] = cov(U_k) + eps(T) * I
    end
    P = zeros(T, K, K)
    for i in 1:K, j in 1:K
        P[i, j] = i == j ? T(0.9) : T(0.1) / (K - 1)
    end
    Sigma_regimes, P
end

function _ms_dirichlet_init(U::Matrix{T}, K::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    T_obs, n = size(U)
    P = zeros(T, K, K)
    for i in 1:K
        α = ones(Float64, K)
        α[i] = 8.0
        P[i, :] = T.(rand(rng, Dirichlet(α)))
    end
    π = T.(rand(rng, Dirichlet(ones(Float64, K))))
    labels = Vector{Int}(undef, T_obs)
    @inbounds for t in 1:T_obs
        u = rand(rng)
        c = zero(T)
        labels[t] = K
        for k in 1:K
            c += π[k]
            if u <= c
                labels[t] = k
                break
            end
        end
    end
    Sigma_regimes = Vector{Matrix{T}}(undef, K)
    Σfull = cov(U) + eps(T) * I
    for k in 1:K
        idx = findall(==(k), labels)
        if length(idx) < n + 1
            Sigma_regimes[k] = Σfull + T(0.05 * k) * I
        else
            Sigma_regimes[k] = cov(U[idx, :]) + eps(T) * I
        end
    end
    Sigma_regimes, P
end

function _ms_em_run(U::Matrix{T}, K::Int, max_iter::Int, tol::T,
                     Sigma_regimes, P) where {T<:AbstractFloat}
    loglik_old = T(-Inf)
    converged = false
    iter = 0
    smoothed = zeros(T, size(U, 1), K)
    for it in 1:max_iter
        iter = it
        filtered, predicted, loglik = _hamilton_filter(U, Sigma_regimes, P)
        smoothed = _hamilton_smoother(filtered, predicted, P)
        if abs(loglik - loglik_old) < tol * abs(loglik_old + one(T))
            converged = true
            break
        end
        loglik_old = loglik
        Sigma_regimes, P = _ms_em_step(U, smoothed, filtered, predicted, P, K)
    end
    filtered, predicted, loglik = _hamilton_filter(U, Sigma_regimes, P)
    smoothed = _hamilton_smoother(filtered, predicted, P)
    (Sigma=Sigma_regimes, P=P, smoothed=smoothed, loglik=loglik,
     converged=converged, iter=iter)
end

"""Permute MS states so regime 1 has the smallest `tr(Σ)`.

EM labels are arbitrary. `B₀` is unit-variance in regime 1, so a quiet/loud
swap rescales columns — Procrustes vs the DGP `B₀` is ≈1.08 on the two-regime
recovery DGP (Julia 1.10 CI). Ties keep the original order.
"""
function _ms_order_regimes(Sigma::Vector{Matrix{T}}, P::Matrix{T},
                           smoothed::Matrix{T}) where {T<:AbstractFloat}
    idx = sortperm([tr(S) for S in Sigma])
    idx == eachindex(Sigma) && return Sigma, P, smoothed
    (Sigma[idx], P[idx, idx], smoothed[:, idx])
end

"""
    identify_markov_switching(model::VARModel; n_regimes=2, max_iter=500, tol=1e-6,
                              n_starts=5, rng=Random.default_rng()) -> MarkovSwitchingSVARResult

Identify SVAR via Markov-switching heteroskedasticity (Lanne & Lütkepohl 2008;
Lanne–Lütkepohl–Maciejowska 2010).

Estimates regime-specific covariances by EM, then identifies a common `B₀` and
`Λ₂…K` by joint ML (`Σ_k = B Λ_k B'`, `Λ₁ = I`). States are ordered so regime 1
is the lowest-trace (quiet) covariance; `B₀` is unit-variance in that regime.
Multiple Dirichlet starts (`n_starts`, default 5); the start with the highest
likelihood is returned.

**Reference**: Lanne & Lütkepohl (2008), LLM (2010), Rigobon (2003)
"""
function identify_markov_switching(model::VARModel{T}; n_regimes::Int=2,
                                    max_iter::Int=500,
                                    tol::T=T(1e-6),
                                    n_starts::Int=5,
                                    rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(model)
    K = n_regimes
    K >= 2 || throw(ArgumentError("n_regimes must be ≥ 2, got $K"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be ≥ 1, got $n_starts"))

    seeds = rand(rng, UInt64, n_starts)
    packed = Vector{NamedTuple}(undef, n_starts)
    Threads.@threads for s in 1:n_starts
        rng_s = Random.MersenneTwister(seeds[s])
        Σ0, P0 = s == 1 ? _ms_chunk_init(model.U, K) : _ms_dirichlet_init(model.U, K, rng_s)
        packed[s] = _ms_em_run(model.U, K, max_iter, tol, Σ0, P0)
    end
    best = 1
    best_ll = T(-Inf)
    for s in 1:n_starts
        if packed[s].loglik > best_ll
            best_ll = packed[s].loglik
            best = s
        end
    end
    st = packed[best]
    Sigma, P, smoothed0 = _ms_order_regimes(st.Sigma, st.P, st.smoothed)
    Tks = vec(sum(smoothed0, dims=1))
    B0, Q, Lambdas, p, _, _, _, _ = _k_regime_ml(Sigma, Tks)
    Sigma_struct = [B0 * Diagonal(Lambdas[k]) * B0' + eps(T) * I for k in 1:K]
    filtered, predicted, loglik = _hamilton_filter(model.U, Sigma_struct, P)
    smoothed = _hamilton_smoother(filtered, predicted, P)
    cq = mean(maximum(smoothed, dims=2))

    se = fill(T(NaN), n, n)
    V = zeros(T, 0, 0)
    try
        nll = θ -> _ms_complete_nll(θ, model.U, smoothed, n, K)
        V = _spd_inv(Matrix{T}(ForwardDiff.hessian(nll, p)))
        se, _ = _delta_B0_se(p, V, θ -> _k_regime_B0(θ, n), n)
    catch
        se = fill(T(NaN), n, n)
        V = zeros(T, 0, 0)
    end

    shocks = Matrix{T}((robust_inv(B0) * model.U')')
    MarkovSwitchingSVARResult{T}(B0, Q, Sigma, Lambdas, smoothed, P,
                                  loglik, st.converged, st.iter, K, se, V, T(cq),
                                  shocks)
end

# =============================================================================
# GARCH(1,1) Helpers
# =============================================================================

"""GARCH(1,1) conditional variance filter: h_t = ω + α ε²_{t-1} + β h_{t-1}. Dual-friendly."""
function _garch11_filter(omega, alpha, beta, epsilon_sq)
    T_obs = length(epsilon_sq)
    Tθ = promote_type(typeof(omega), eltype(epsilon_sq))
    h = Vector{Tθ}(undef, T_obs)
    den = one(Tθ) - Tθ(alpha) - Tθ(beta)
    den = den < Tθ(1e-12) ? Tθ(1e-12) : den
    h[1] = Tθ(omega) / den
    @inbounds for t in 2:T_obs
        ht = Tθ(omega) + Tθ(alpha) * epsilon_sq[t - 1] + Tθ(beta) * h[t - 1]
        h[t] = ht < Tθ(1e-12) ? Tθ(1e-12) : ht
    end
    h
end

function _garch_unpack_shock(p)
    Tθ = eltype(p)
    omega = exp(p[1])
    alpha = one(Tθ) / (one(Tθ) + exp(-p[2])) * Tθ(0.5)
    beta = one(Tθ) / (one(Tθ) + exp(-p[3])) * Tθ(0.99)
    omega, alpha, beta
end

function _garch_pack_shock(omega::T, alpha::T, beta::T) where {T<:AbstractFloat}
    α = clamp(alpha, T(1e-8), T(0.49))
    β = clamp(beta, T(1e-8), T(0.98))
    if α + β >= T(0.999)
        s = α + β
        α *= T(0.99) / s
        β *= T(0.99) / s
    end
    p2 = -log(T(0.5) / α - one(T))
    p3 = -log(T(0.99) / β - one(T))
    [log(max(omega, T(1e-12))), p2, p3]
end

"""GARCH-SVAR negative log-likelihood in (θ, unconstrained GARCH) params. Dual-friendly."""
function _garch_svar_nll(params, U, Lmat, n::Int)
    Tθ = eltype(params)
    n_angles = n * (n - 1) ÷ 2
    Q = n_angles == 0 ? Matrix{Tθ}(I, n, n) :
        _givens_to_orthogonal(params[1:n_angles], n)
    B = Lmat * Q
    shocks = U * (B \ I)'
    T_obs = size(U, 1)
    nll = zero(Tθ)
    log2π = log(Tθ(2) * Tθ(π))
    half = Tθ(0.5)
    logdetB = log(max(abs(det(B)), Tθ(1e-16)))
    for j in 1:n
        i0 = n_angles + (j - 1) * 3
        omega, alpha, beta = _garch_unpack_shock(params[i0+1:i0+3])
        if alpha + beta >= one(Tθ)
            return Tθ(1e20)
        end
        resid_sq = shocks[:, j] .^ 2
        h = _garch11_filter(omega, alpha, beta, resid_sq)
        @inbounds for t in 1:T_obs
            nll += half * (log2π + log(h[t]) + resid_sq[t] / h[t])
        end
    end
    nll += Tθ(T_obs) * logdetB
    nll
end

"""GARCH(1,1) log-likelihood (negative, for minimization)."""
function _garch11_loglik(params::Vector{T}, epsilon_sq::Vector{T}) where {T<:AbstractFloat}
    omega = exp(params[1])
    alpha = one(T) / (one(T) + exp(-params[2])) * T(0.5)  # constrain to (0, 0.5)
    beta = one(T) / (one(T) + exp(-params[3])) * T(0.99)   # constrain to (0, 0.99)

    # Ensure stationarity
    if alpha + beta >= one(T)
        return T(Inf)
    end

    h = _garch11_filter(omega, alpha, beta, epsilon_sq)
    T_obs = length(epsilon_sq)

    loglik = zero(T)
    for t in 1:T_obs
        loglik -= T(0.5) * (log(T(2π)) + log(h[t]) + epsilon_sq[t] / h[t])
    end
    -loglik  # negative for minimization
end

"""Estimate GARCH(1,1) parameters for a single series."""
function _estimate_garch11(epsilon_sq::Vector{T}) where {T<:AbstractFloat}
    # Stationary GARCH(1,1) start: α≈0.05, β≈0.90 ⇒ α+β=0.95 < 1.
    # Inverse of α = 0.5/(1+exp(-p2)) ⇒ p2 = -log(0.5/α - 1); same for β with cap 0.99.
    # Previous init [log(var*0.05), 0, 2] mapped to (α,β)=(0.25,0.872) with α+β>1 (Inf loglik).
    α0, β0 = T(0.05), T(0.90)
    p2_0 = -log(T(0.5) / α0 - one(T))          # logit for α-map
    p3_0 = -log(T(0.99) / β0 - one(T))          # logit for β-map
    var_eps = max(var(epsilon_sq), eps(T))
    params0 = [log(var_eps * (one(T) - α0 - β0)), p2_0, p3_0]

    obj = p -> begin
        ll = _garch11_loglik(p, epsilon_sq)
        ifelse(isfinite(ll), ll, T(1e20))  # reject Inf / NaN starts inside Nelder-Mead
    end
    result = Optim.optimize(obj, params0,
                            Optim.NelderMead(),
                            Optim.Options(iterations=500))

    p = Optim.minimizer(result)
    omega = exp(p[1])
    alpha = one(T) / (one(T) + exp(-p[2])) * T(0.5)
    beta = one(T) / (one(T) + exp(-p[3])) * T(0.99)
    # Final stationarity clip (should already hold via the barrier)
    if alpha + beta >= one(T)
        s = alpha + beta
        alpha *= T(0.99) / s
        beta  *= T(0.99) / s
    end

    h = _garch11_filter(omega, alpha, beta, epsilon_sq)
    (omega, alpha, beta, h)
end

# =============================================================================
# Public API: GARCH
# =============================================================================

"""
    identify_garch(model::VARModel; max_iter=500, tol=1e-6) -> GARCHSVARResult

Identify SVAR via GARCH-based heteroskedasticity (Normandin & Phaneuf 2004).

Iterative procedure:
1. Start with Cholesky B₀
2. Compute structural shocks ε_t = B₀⁻¹ u_t
3. Fit GARCH(1,1) to each ε_j,t
4. Use conditional covariances to re-estimate B₀
5. Repeat until convergence

**Reference**: Normandin & Phaneuf (2004)
"""
function identify_garch(model::VARModel{T}; max_iter::Int=500,
                         tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)

    # Initialize with Cholesky: B₀ = L * Q(θ), start at Q = I (θ = 0)
    L = safe_cholesky(model.Sigma)
    L_mat = Matrix(L)
    n_angles = n * (n - 1) ÷ 2
    angles = zeros(T, n_angles)
    log_det_B0_inv = -sum(log(L_mat[i, i]) for i in 1:n)

    # Precompute whitened residuals: Z = U * L⁻ᵀ, so shocks = Z * Q
    L_inv_t = Matrix{T}(robust_inv(L_mat)')
    Z = model.U * L_inv_t

    garch_params = zeros(T, n, 3)
    cond_var = ones(T, T_obs, n)
    loglik_old = T(-Inf)
    converged = false
    iter = 0

    for it in 1:max_iter
        iter = it

        # Build B₀ from current Givens angles, compute structural shocks
        Q = _givens_to_orthogonal(angles, n)
        shocks = Z * Q

        # Fit GARCH(1,1) to each structural shock series
        loglik = zero(T)
        for j in 1:n
            resid_sq = shocks[:, j] .^ 2
            omega, alpha, beta, h = _estimate_garch11(resid_sq)
            garch_params[j, :] = [omega, alpha, beta]
            cond_var[:, j] = h

            for t in 1:T_obs
                loglik -= T(0.5) * (log(T(2π)) + log(h[t]) + resid_sq[t] / h[t])
            end
        end
        loglik += T_obs * log_det_B0_inv

        # Check convergence
        if abs(loglik - loglik_old) < tol * abs(loglik_old + one(T))
            converged = true
            break
        end
        loglik_old = loglik

        # Re-estimate B₀ by optimizing Givens angles with fixed GARCH variances.
        # Since |det(Q)| = 1, the log-det term is constant w.r.t. θ.
        # The log(h) terms are also fixed. Only the weighted residual sum varies.
        if n_angles > 0
            obj = theta -> begin
                Q_t = _givens_to_orthogonal(theta, n)
                shocks_t = Z * Q_t
                val = zero(T)
                for t in 1:T_obs
                    for j in 1:n
                        val += shocks_t[t, j]^2 / cond_var[t, j]
                    end
                end
                val
            end
            result_opt = Optim.optimize(obj, angles, Optim.NelderMead(),
                                        Optim.Options(iterations=200))
            angles = Optim.minimizer(result_opt)
        end
    end

    # Final B₀ and Q with sign normalization
    Q = _givens_to_orthogonal(angles, n)
    B0 = L_mat * Q
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] *= -one(T)
            Q[:, j] *= -one(T)
        end
    end
    B0_inv = robust_inv(B0)
    shocks = (B0_inv * model.U')'

    p_garch = Vector{T}(angles)
    for j in 1:n
        append!(p_garch, _garch_pack_shock(garch_params[j, 1], garch_params[j, 2],
                                            garch_params[j, 3]))
    end
    se = fill(T(NaN), n, n)
    V = zeros(T, 0, 0)
    try
        nll = θ -> _garch_svar_nll(θ, model.U, L_mat, n)
        V = _spd_inv(Matrix{T}(ForwardDiff.hessian(nll, p_garch)))
        se, _ = _delta_B0_se(p_garch, V, θ -> begin
            n_angles = n * (n - 1) ÷ 2
            Qθ = n_angles == 0 ? Matrix{eltype(θ)}(I, n, n) :
                 _givens_to_orthogonal(θ[1:n_angles], n)
            L_mat * Qθ
        end, n)
    catch
        se = fill(T(NaN), n, n)
        V = zeros(T, 0, 0)
    end

    GARCHSVARResult{T}(B0, Q, garch_params, cond_var, shocks, loglik_old, converged, iter,
                        se, V)
end

# =============================================================================
# Smooth Transition
# =============================================================================

"""Logistic transition: G(s) = 1 / (1 + exp(-γ(s - c)/σs)). Dual-friendly (no AbstractFloat bound)."""
_logistic_transition(s, gamma, c) = _logistic_transition(s, gamma, c, one(gamma))
function _logistic_transition(s, gamma, c, sigma_s)
    z = gamma * (s - c) / sigma_s
    one(z) / (one(z) + exp(-z))
end

"""Pack lower-triangular L as `[log diag(L); strictly-lower column-major]`."""
function _st_pack_L(L::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(L, 1)
    v = Vector{T}(undef, n * (n + 1) ÷ 2)
    k = 1
    for i in 1:n
        v[k] = log(max(L[i, i], T(1e-12)))
        k += 1
    end
    for j in 1:(n - 1)
        for i in (j + 1):n
            v[k] = L[i, j]
            k += 1
        end
    end
    v
end

"""Unpack a free Cholesky factor. Dual-friendly."""
function _st_unpack_L(v, n::Int)
    Tθ = eltype(v)
    L = zeros(Tθ, n, n)
    k = 1
    for i in 1:n
        L[i, i] = exp(v[k])
        k += 1
    end
    for j in 1:(n - 1)
        for i in (j + 1):n
            L[i, j] = v[k]
            k += 1
        end
    end
    L
end

"""Forward substitution Lz = u for lower-triangular L. Dual-friendly."""
function _st_forward_sub(L, u)
    n = length(u)
    Tθ = promote_type(eltype(L), eltype(u))
    z = Vector{Tθ}(undef, n)
    @inbounds for i in 1:n
        acc = Tθ(u[i])
        for j in 1:(i - 1)
            acc -= L[i, j] * z[j]
        end
        z[i] = acc / L[i, i]
    end
    z
end

"""Negative Gaussian log-likelihood of the smooth-transition SVAR. Dual-friendly."""
function _smooth_transition_nll(params, U::AbstractMatrix, s::AbstractVector,
                                sigma_s, n::Int, logγ_lo, logγ_hi)
    n_L = n * (n + 1) ÷ 2
    n_angles = n * (n - 1) ÷ 2
    L = _st_unpack_L(view(params, 1:n_L), n)
    θ = view(params, (n_L + 1):(n_L + n_angles))
    logΛ = view(params, (n_L + n_angles + 1):(n_L + n_angles + n))
    xγ = params[n_L + n_angles + n + 1]
    cc = params[n_L + n_angles + n + 2]
    Q = _givens_to_orthogonal(θ, n)
    Tθ = eltype(params)
    half = Tθ(0.5)
    log2π = log(Tθ(2) * Tθ(π))
    floor_d = Tθ(1e-12)
    logdetL = zero(Tθ)
    for i in 1:n
        logdetL += log(L[i, i])
    end
    γ = _st_gamma_from_x(xγ, logγ_lo, logγ_hi)
    nll = zero(Tθ)
    T_obs = size(U, 1)
    for t in 1:T_obs
        G_t = _logistic_transition(s[t], γ, cc, sigma_s)
        z = _st_forward_sub(L, view(U, t, :))
        acc = zero(Tθ)
        for j in 1:n
            λj = exp(logΛ[j])
            d_j = max(one(Tθ) + G_t * (λj - one(Tθ)), floor_d)
            εj = zero(Tθ)
            for i in 1:n
                εj += Q[i, j] * z[i]
            end
            acc += log(d_j) + εj^2 / d_j
        end
        nll += half * (Tθ(n) * log2π + Tθ(2) * logdetL + acc)
    end
    nll
end

"""Map the unbounded γ-score to a bounded γ ∈ [γ_lo, γ_hi]."""
function _st_gamma_from_x(xγ, logγ_lo, logγ_hi)
    logγ = logγ_lo + (logγ_hi - logγ_lo) / (one(xγ) + exp(-xγ))
    exp(logγ)
end

function _st_x_from_gamma(γ, logγ_lo, logγ_hi)
    logγ = log(γ)
    p = (logγ - logγ_lo) / (logγ_hi - logγ_lo)
    p = clamp(p, oftype(p, 1e-8), oftype(p, 1 - 1e-8))
    log(p / (one(p) - p))
end

"""
    identify_smooth_transition(model::VARModel, transition_var::AbstractVector;
                               max_iter=500, tol=1e-6) -> SmoothTransitionSVARResult

Identify SVAR via smooth-transition heteroskedasticity (Lütkepohl & Netšunajev 2017).

Joint ML over `(θ_Givens, Λ, γ, c)` with
```math
\\Sigma_t = B_0 [I + G(s_t)(\\Lambda - I)] B_0', \\qquad B_0 = L Q(\\theta)
```
where ``G(s_t) = 1/(1 + \\exp(-\\gamma(s_t - c)/\\mathrm{std}(s)))``. `L` is
initialized from the median-split G=0 Cholesky and estimated jointly with
`(θ, Λ, γ, c)` so `B₀B₀'` is the G=0 pole, not the unconditional covariance.
`log γ` is bounded so `γ ∈ [10^{-3}, 20]/std(s)`.

Arguments:
- `transition_var` — the transition variable s_t (e.g., a lagged endogenous variable)

**Reference**: Lütkepohl & Netšunajev (2017)
"""
function identify_smooth_transition(model::VARModel{T}, transition_var::AbstractVector;
                                     max_iter::Int=500,
                                     tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)
    s = Vector{T}(transition_var[1:T_obs])
    sigma_s = std(s)
    sigma_s > zero(T) || throw(ArgumentError(
        "transition variable has zero variance; cannot scale γ."))

    # Split kernel: starting point, not the estimator.
    gamma_init = one(T) / sigma_s
    c_init = median(s)
    G_vals = [_logistic_transition(s[t], gamma_init, c_init, sigma_s) for t in 1:T_obs]
    low_idx = findall(g -> g < T(0.5), G_vals)
    high_idx = findall(g -> g >= T(0.5), G_vals)
    Sigma1 = isempty(low_idx) ? cov(model.U) : cov(model.U[low_idx, :])
    Sigma2 = isempty(high_idx) ? cov(model.U) : cov(model.U[high_idx, :])
    Sigma1 = Matrix{T}(Symmetric(Sigma1 + eps(T) * I))
    Sigma2 = Matrix{T}(Symmetric(Sigma2 + eps(T) * I))

    B0_split, Q_split, Lambda_raw = _eigendecomposition_id(Sigma1, Sigma2)
    Lambda_init = max.(Lambda_raw, T(1e-8))

    # B₀ = L Q(θ). L starts at chol(Σ_{G=0}) from the split kernel and is
    # estimated jointly so the G=0 pole is not frozen at the split/unconditional Σ.
    L_init = Matrix{T}(safe_cholesky(Sigma1))
    Q_init = Q_split
    if det(Q_init) < 0
        Q_init = copy(Q_init)
        Q_init[:, n] .*= -one(T)
    end
    θ_init = _orthogonal_to_givens(Q_init, n)

    γ_lo = T(1e-3) / sigma_s
    γ_hi = T(20) / sigma_s
    logγ_lo = log(γ_lo)
    logγ_hi = log(γ_hi)
    xγ_init = _st_x_from_gamma(gamma_init, logγ_lo, logγ_hi)

    n_L = n * (n + 1) ÷ 2
    n_angles = n * (n - 1) ÷ 2
    params0 = vcat(_st_pack_L(L_init), θ_init, log.(Lambda_init), xγ_init, c_init)

    obj = p -> _smooth_transition_nll(p, model.U, s, sigma_s, n, logγ_lo, logγ_hi)
    g! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
    result = Optim.optimize(obj, g!, params0, Optim.LBFGS(),
                            Optim.Options(iterations=max_iter, g_tol=min(tol, T(1e-8)),
                                          f_reltol=T(1e-12), allow_f_increases=true))

    p_opt = Optim.minimizer(result)
    L_mat = Matrix{T}(_st_unpack_L(view(p_opt, 1:n_L), n))
    θ_opt = p_opt[(n_L + 1):(n_L + n_angles)]
    Λ_opt = exp.(p_opt[(n_L + n_angles + 1):(n_L + n_angles + n)])
    gamma_opt = _st_gamma_from_x(p_opt[n_L + n_angles + n + 1], logγ_lo, logγ_hi)
    c_opt = p_opt[n_L + n_angles + n + 2]
    Q = _givens_to_orthogonal(θ_opt, n)
    B0 = L_mat * Q

    idx = sortperm(Λ_opt)
    Λ_opt = Λ_opt[idx]
    Q = Q[:, idx]
    B0 = B0[:, idx]
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] .*= -one(T)
            Q[:, j] .*= -one(T)
        end
    end

    Sigma_G0 = B0 * B0'
    Sigma_G1 = B0 * Diagonal(Λ_opt) * B0'
    G_final = [_logistic_transition(s[t], gamma_opt, c_opt, sigma_s) for t in 1:T_obs]
    loglik_final = -obj(p_opt)

    if det(Q) < 0
        Q = copy(Q)
        Q[:, n] .*= -one(T)
        B0 = L_mat * Q
        if B0[n, n] < 0
            B0[:, n] .*= -one(T)
            Q[:, n] .*= -one(T)
        end
    end
    θ_canon = n_angles == 0 ? T[] : _orthogonal_to_givens(Q, n)
    p_canon = vcat(_st_pack_L(L_mat), θ_canon, log.(max.(Λ_opt, T(1e-12))),
                   p_opt[n_L + n_angles + n + 1], T(c_opt))
    se = fill(T(NaN), n, n)
    V = zeros(T, 0, 0)
    try
        V = _spd_inv(Matrix{T}(ForwardDiff.hessian(obj, p_canon)))
        B0_fn = p -> begin
            Lp = _st_unpack_L(view(p, 1:n_L), n)
            Qp = n_angles == 0 ? Matrix{eltype(p)}(I, n, n) :
                 _givens_to_orthogonal(p[(n_L + 1):(n_L + n_angles)], n)
            Lp * Qp
        end
        se, _ = _delta_B0_se(p_canon, V, B0_fn, n)
    catch
        se = fill(T(NaN), n, n)
        V = zeros(T, 0, 0)
    end

    SmoothTransitionSVARResult{T}(B0, Q, [Matrix{T}(Sigma_G0), Matrix{T}(Sigma_G1)],
                                   [ones(T, n), Vector{T}(Λ_opt)],
                                   T(gamma_opt), T(c_opt), s, G_final, T(loglik_final),
                                   Optim.converged(result), Optim.iterations(result),
                                   se, V, Matrix{T}(model.U))
end

# =============================================================================
# Public API: External Volatility
# =============================================================================

"""
    identify_external_volatility(model::VARModel, regime_indicator::AbstractVector{Int};
                                 regimes=2) -> ExternalVolatilitySVARResult

Identify SVAR via externally specified volatility regimes (Rigobon 2003;
Lanne–Lütkepohl–Maciejowska 2010).

Uses a known regime indicator (e.g., NBER recessions, financial crises) to split
the sample, then jointly estimates `B₀` and `Λ₂…K` from all `K` regime covariances.

Arguments:
- `regime_indicator` — integer vector of regime labels (1, 2, ..., K)
- `regimes` — number of distinct regimes (default: 2)

A regime with fewer than `n + 1` observations throws `ArgumentError`.

**Reference**: Rigobon (2003), LLM (2010)
"""
function identify_external_volatility(model::VARModel{T},
                                       regime_indicator::AbstractVector{Int};
                                       regimes::Int=2) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)
    K = regimes

    @assert length(regime_indicator) >= T_obs "regime_indicator must have length ≥ T_obs"
    K >= 2 || throw(ArgumentError("regimes must be ≥ 2, got $K"))

    regime_indices = [findall(regime_indicator[1:T_obs] .== k) for k in 1:K]

    Sigma_regimes = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        idx = regime_indices[k]
        if length(idx) < n + 1
            throw(ArgumentError(
                "regime $k has $(length(idx)) observations; need at least $(n + 1)"))
        end
        Sigma_regimes[k] = cov(model.U[idx, :]) + eps(T) * I
    end

    Tks = T[T(length(idx)) for idx in regime_indices]
    B0, Q, Lambda_vecs, p, _, _, _, _ = _k_regime_ml(Sigma_regimes, Tks)

    regimes = Vector{Int}(regime_indicator[1:T_obs])
    loglik = -T(_k_regime_full_nll(p, model.U, regimes, n, K))

    se = fill(T(NaN), n, n)
    V = zeros(T, 0, 0)
    try
        V = _external_vcov(p, model.U, regimes, n, K)
        se, _ = _delta_B0_se(p, V, θ -> _k_regime_B0(θ, n), n)
    catch
        se = fill(T(NaN), n, n)
        V = zeros(T, 0, 0)
    end

    shocks = Matrix{T}((robust_inv(B0) * model.U')')
    ExternalVolatilitySVARResult{T}(B0, Q, Sigma_regimes, Lambda_vecs, regime_indices,
                                     loglik, se, V, shocks)
end
