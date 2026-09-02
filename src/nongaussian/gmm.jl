# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Moment-based non-Gaussian SVAR identification by GMM (Keweloh 2021; Lanne & Luoto 2021).

Parameterises ``B₀ = L Q(θ)`` with Givens angles and estimates ``θ`` from
coskewness / cokurtosis moment conditions (Keweloh 2021, eqs. 7–9) using the
package GMM kernel. Covariance moments ``E[εε' - I] = 0`` hold by the ``LQ``
parameterisation. Identification requires at most one Gaussian shock.

References:
- Keweloh, S. A. (2021). "A Generalized Method of Moments Estimator for Structural
  Vector Autoregressions Based on Higher Moments."
- Lanne, M. & Luoto, J. (2021). "GMM Estimation of Non-Gaussian Structural Vector Autoregression."
- Lewis, D. J. (2025). "Identification based on higher moments in macroeconometrics."
"""

using LinearAlgebra, Statistics, Random

# =============================================================================
# Result Type
# =============================================================================

"""
    NonGaussianGMMResult{T} <: AbstractNonGaussianSVAR

Result from moment-based non-Gaussian GMM SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix (n × n)
- `Q::Matrix{T}` — rotation matrix (`B₀ = L Q`)
- `theta::Vector{T}` — Givens angles
- `vcov::Matrix{T}` — sandwich covariance of `theta`
- `se::Matrix{T}` — delta-method standard errors for `B₀`
- `J::T` — Hansen J statistic
- `J_pvalue::T` — J-test p-value
- `moments::Symbol` — `:coskewness`, `:cokurtosis`, or `:both`
- `weighting::Symbol` — `:two_step` or `:cue`
- `shocks::Matrix{T}` — structural shocks (T_eff × n)
- `varnames::Vector{String}`
- `shock_names::Vector{String}`
"""
struct NonGaussianGMMResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    theta::Vector{T}
    vcov::Matrix{T}
    se::Matrix{T}
    J::T
    J_pvalue::T
    moments::Symbol
    weighting::Symbol
    shocks::Matrix{T}
    varnames::Vector{String}
    shock_names::Vector{String}
end

function Base.show(io::IO, r::NonGaussianGMMResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Variables"    n;
        "Moments"      string(r.moments);
        "Weighting"    string(r.weighting);
        "J-statistic"  _fmt(r.J);
        "J p-value"    _format_pvalue(r.J_pvalue);
        "Givens angles" length(r.theta)
    ]
    _pretty_table(io, spec;
        title = "Non-Gaussian GMM Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _show_B0_ses(io, r.B0, r.se)
end

# =============================================================================
# Keweloh (2021) moment conditions, eqs. 7–9
# =============================================================================

"""Number of Keweloh moment conditions for `n` shocks and a moment family."""
function _keweloh_n_moments(n::Int, moments::Symbol)
    npairs = n * (n - 1)           # ordered i ≠ j
    nsym = n * (n - 1) ÷ 2         # unordered i < j (symmetric cokurtosis)
    if moments === :coskewness
        npairs
    elseif moments === :cokurtosis
        npairs + nsym
    elseif moments === :both
        2 * npairs + nsym
    else
        throw(ArgumentError("moments must be :coskewness, :cokurtosis, or :both, got :$moments"))
    end
end

"""
Per-observation Keweloh moments.

- `:coskewness` — ``E[ε_i² ε_j] = 0`` for ``i ≠ j`` (third-moment block)
- `:cokurtosis` — ``E[ε_i³ ε_j] = 0`` for ``i ≠ j`` and ``E[ε_i² ε_j²] − 1 = 0``
  for ``i < j`` (fourth-moment block)
- `:both` — concatenate the two blocks

`data.Z` is the Cholesky-whitened residual matrix (T_eff × n); shocks are `Z Q(θ)`.
"""
function _keweloh_moment_matrix(theta::AbstractVector{T}, data) where {T<:AbstractFloat}
    Z = data.Z
    n = data.n
    kind = data.moments
    Q = _givens_to_orthogonal(theta, n)
    ε = Z * Q
    Tobs = size(ε, 1)
    q = _keweloh_n_moments(n, kind)
    M = Matrix{T}(undef, Tobs, q)
    col = 1
    if kind === :coskewness || kind === :both
        for i in 1:n, j in 1:n
            i == j && continue
            @inbounds for t in 1:Tobs
                M[t, col] = ε[t, i]^2 * ε[t, j]
            end
            col += 1
        end
    end
    if kind === :cokurtosis || kind === :both
        for i in 1:n, j in 1:n
            i == j && continue
            @inbounds for t in 1:Tobs
                M[t, col] = ε[t, i]^3 * ε[t, j]
            end
            col += 1
        end
        for i in 1:(n - 1), j in (i + 1):n
            @inbounds for t in 1:Tobs
                M[t, col] = ε[t, i]^2 * ε[t, j]^2 - one(T)
            end
            col += 1
        end
    end
    M
end

# =============================================================================
# Estimation
# =============================================================================

"""
    identify_gmm_moments(model::VARModel; moments=:coskewness|:cokurtosis|:both,
                         weighting=:two_step|:cue, se=:sandwich) -> NonGaussianGMMResult

Identify ``B₀ = L Q(θ)`` from higher-moment conditions (Keweloh 2021; Lanne & Luoto 2021).

`moments` selects the Keweloh (2021, eqs. 7–9) block: third-moment coskewness,
fourth-moment cokurtosis, or both. `weighting=:two_step` is Hansen two-step GMM;
`:cue` maps to the kernel's iterated GMM (continuously updated weighting between
steps). Sandwich standard errors for `θ` are converted to ``B₀`` SEs by the
delta method. Covariance moments are automatic under the ``LQ`` parameterisation.
"""
function identify_gmm_moments(model::VARModel{T};
                              moments::Symbol=:both,
                              weighting::Symbol=:two_step,
                              se::Symbol=:sandwich,
                              hac::Bool=true,
                              bandwidth::Int=0,
                              max_iter::Int=100,
                              tol::T=T(1e-8),
                              n_starts::Int=1,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    moments in (:coskewness, :cokurtosis, :both) || throw(ArgumentError(
        "moments must be :coskewness, :cokurtosis, or :both, got :$moments"))
    weighting in (:two_step, :cue) || throw(ArgumentError(
        "weighting must be :two_step or :cue, got :$weighting"))
    se === :sandwich || throw(ArgumentError(
        "se must be :sandwich, got :$se"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be ≥ 1"))

    n = nvars(model)
    n >= 2 || throw(ArgumentError("GMM moment identification requires n ≥ 2 variables"))
    n_angles = n * (n - 1) ÷ 2
    q = _keweloh_n_moments(n, moments)
    q >= n_angles || throw(ArgumentError(
        "moments=:$moments supplies $q conditions for $n_angles Givens angles"))

    L = Matrix{T}(safe_cholesky(model.Sigma))
    Z = Matrix{T}(model.U / L')
    data = (Z=Z, n=n, moments=moments)
    w_kernel = weighting === :cue ? :iterated : weighting

    starts = Vector{Vector{T}}(undef, n_starts)
    starts[1] = zeros(T, n_angles)
    for s in 2:n_starts
        starts[s] = T(π) .* (rand(rng, T, n_angles) .- T(0.5))
    end

    best = nothing
    for θ0 in starts
        gmm = estimate_gmm(_keweloh_moment_matrix, θ0, data;
                           weighting=w_kernel, hac=hac, bandwidth=bandwidth,
                           max_iter=max_iter, tol=tol)
        if best === nothing || gmm.J_stat < best.J_stat
            best = gmm
        end
    end

    theta = best.theta
    Q = _givens_to_orthogonal(theta, n)
    B0 = L * Q
    shocks = Z * Q
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] .*= -one(T)
            Q[:, j] .*= -one(T)
            shocks[:, j] .*= -one(T)
        end
    end

    se_B0, _ = _delta_B0_se(theta, best.vcov, ϑ -> L * _givens_to_orthogonal(ϑ, n), n)

    varnames = copy(model.varnames)
    shock_names = ["Shock $j" for j in 1:n]
    NonGaussianGMMResult{T}(B0, Q, theta, Matrix{T}(best.vcov), se_B0,
                            T(best.J_stat), T(best.J_pvalue), moments, weighting,
                            shocks, varnames, shock_names)
end
