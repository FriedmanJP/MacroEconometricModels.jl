# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Three-stage least squares (3SLS) for simultaneous systems (Zellner & Theil 1962):
instrument-project every equation's regressors, then apply the SUR GLS step.
"""

using LinearAlgebra, Statistics

"""
    estimate_3sls(eqs, Z; instruments=:common, eqnames=nothing) -> ThreeSLSModel

Estimate a simultaneous system by three-stage least squares (Zellner & Theil 1962).

# Arguments
- `eqs::Vector{<:Tuple}` — one `(y, X)` (or `(y, X, names)`) tuple per equation. `X`
  contains the equation's own regressors (endogenous and included exogenous); include an
  intercept column if wanted. All equations must share the same `T`.
- `Z` — instruments. Either a single `AbstractMatrix` shared by every equation
  (`instruments=:common`, the default) or a `Vector{<:AbstractMatrix}` of per-equation
  instrument matrices (`instruments=:perequation`). Each instrument block must have at least
  as many columns as its equation's regressors (order condition).

# Keyword arguments
- `instruments::Symbol=:common` — `:common` or `:perequation`; must match the shape of `Z`.
- `eqnames::Union{Nothing,Vector{String}}=nothing` — equation labels.

# Returns
`ThreeSLSModel{T}` — per-equation `betas`/`ses`, the 2SLS-residual cross-covariance `Sigma`
(divisor `T`), the McElroy system R², and the per-equation instrument count.

# Estimator
Project each equation's regressors onto its instrument space, `X̂ᵢ = P_{Zᵢ} Xᵢ`, form the
cross-equation covariance from equation-by-equation 2SLS residuals `Σ̂_ij = ûᵢ'ûⱼ / T`, then

    β̂ = (X̂'(Σ̂⁻¹⊗I)X̂)⁻¹ X̂'(Σ̂⁻¹⊗I)y,   V̂(β̂) = (X̂'(Σ̂⁻¹⊗I)X̂)⁻¹

with `X̂ = blkdiag(X̂₁,…,X̂_M)`. Residuals are always taken from the ORIGINAL `Xᵢ`. When the
instruments span every regressor (`P_{Zᵢ} Xᵢ = Xᵢ`) 3SLS equals SUR; when every equation is
exactly identified it equals equation-by-equation 2SLS.

# Examples
```julia
pd = load_example(:grunfeld)
ge = group_data(pd, "General Electric")
wh = group_data(pd, "Westinghouse")
Xge = hcat(ones(20), ge.data[:, 2], ge.data[:, 3])   # [1, value, capital]
Xwh = hcat(ones(20), wh.data[:, 2], wh.data[:, 3])
Z   = hcat(ones(20), ge.data[:, 3], wh.data[:, 3])   # common instruments [1, C_ge, C_wh]
m = estimate_3sls([(ge.data[:, 1], Xge, ["const", "value", "capital"]),
                   (wh.data[:, 1], Xwh, ["const", "value", "capital"])], Z;
                  eqnames = ["GE", "Westinghouse"])
report(m)
```

# References
- Zellner, A. & Theil, H. (1962). *Econometrica* 30(1), 54-78.
- McElroy, M. B. (1977). *Journal of Econometrics* 6(3), 381-387.
"""
function estimate_3sls(eqs::AbstractVector, Z;
                       instruments::Symbol=:common,
                       eqnames::Union{Nothing,AbstractVector}=nothing)
    ys64, Xs64, vns = _normalize_eqs(eqs)
    T = Float64
    ys = Vector{Vector{T}}(ys64)
    Xs = Vector{Matrix{T}}(Xs64)
    M = length(Xs)
    Tn = length(ys[1])

    # ---- Resolve instrument blocks (common vs per-equation) ----
    instruments in (:common, :perequation) ||
        throw(ArgumentError("instruments must be :common or :perequation; got :$instruments"))
    Zs = Vector{Matrix{T}}(undef, M)
    if Z isa AbstractMatrix
        instruments == :common || throw(ArgumentError(
            "instruments=:perequation requires Z to be a vector of matrices"))
        Zc = Matrix{T}(Z)
        size(Zc, 1) == Tn || throw(ArgumentError("Z must have $Tn rows; got $(size(Zc, 1))"))
        for j in 1:M
            Zs[j] = Zc
        end
    elseif Z isa AbstractVector
        length(Z) == M || throw(ArgumentError(
            "per-equation Z must have $M instrument matrices; got $(length(Z))"))
        for j in 1:M
            Zj = Matrix{T}(Z[j])
            size(Zj, 1) == Tn || throw(ArgumentError("Z[$j] must have $Tn rows; got $(size(Zj, 1))"))
            Zs[j] = Zj
        end
    else
        throw(ArgumentError("Z must be an AbstractMatrix (common) or a Vector of matrices"))
    end

    for j in 1:M
        size(Zs[j], 2) >= size(Xs[j], 2) || throw(ArgumentError(
            "equation $j order condition violated: $(size(Zs[j], 2)) instruments < $(size(Xs[j], 2)) regressors"))
    end

    # ---- Stage 1: project regressors onto the instrument space, X̂ᵢ = P_{Zᵢ} Xᵢ ----
    # `Zj \ Xs[j]` is the QR least-squares solve (Zᵢ'Zᵢ)⁻¹Zᵢ'Xᵢ, so P_Z X = Z·(Z\X) avoids
    # squaring the (large-scale Grunfeld) condition number that inv(Z'Z) would incur.
    Xhat = Vector{Matrix{T}}(undef, M)
    for j in 1:M
        Zj = Zs[j]
        Xhat[j] = Zj * (Zj \ Xs[j])
    end

    # ---- Stage 2: equation-by-equation 2SLS → Σ̂  (β = (X̂'X̂)⁻¹X̂'y = X̂ \ y) ----
    betas2 = [Xhat[j] \ ys[j] for j in 1:M]
    resids2 = [ys[j] .- Xs[j] * betas2[j] for j in 1:M]
    Sigma = _sigma_from_resids(resids2, Tn)

    # ---- Stage 3: system GLS on the projected regressors ----
    Sinv = Matrix{T}(robust_inv(Hermitian(Sigma); rcond_tol=eps(T)))
    Amat, rhs = _gls_normal_eqs(Xhat, Xhat, ys, Sinv)
    A = Matrix{T}(robust_inv(Symmetric(Amat); rcond_tol=eps(T)))
    beta = A * rhs

    betablk = _split_beta(beta, Xs)
    resids = [ys[j] .- Xs[j] * betablk[j] for j in 1:M]   # residuals from ORIGINAL X
    fitted = [Xs[j] * betablk[j] for j in 1:M]
    ses = [sqrt.(max.(diag(A)[_block_range(Xs, j)], zero(T))) for j in 1:M]

    enames = eqnames === nothing ? ["eq$(j)" for j in 1:M] : String.(collect(eqnames))
    length(enames) == M || throw(ArgumentError("eqnames must have length $M"))

    det_sigma = det(Sigma)
    mcr2 = _mcelroy_r2(resids, ys, Sinv)
    n_instr = [size(Zs[j], 2) for j in 1:M]

    ThreeSLSModel{T}(enames, vns, betablk, ses, A, Sigma, resids, fitted, Tn,
                     det_sigma, mcr2, n_instr)
end
