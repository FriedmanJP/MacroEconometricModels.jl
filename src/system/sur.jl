# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Seemingly-unrelated regressions (SUR) via feasible GLS (Zellner 1962), with
optional iteration to the Gaussian MLE and linear cross-equation restrictions.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Shared system-GLS building blocks (reused by SUR and 3SLS)
# =============================================================================

"""
    _normalize_eqs(eqs) -> (ys, Xs, varnames)

Validate a vector of `(y, X)` (or `(y, X, names)`) equation tuples and convert to
concrete `Vector{Vector{T}}` / `Vector{Matrix{T}}`. Every equation must share the
same number of observations `T` (SUR/3SLS require a balanced panel of equations).
"""
function _normalize_eqs(eqs::AbstractVector)
    isempty(eqs) && throw(ArgumentError("eqs must contain at least one equation"))
    M = length(eqs)
    ys = Vector{Vector{Float64}}(undef, M)
    Xs = Vector{Matrix{Float64}}(undef, M)
    vns = Vector{Vector{String}}(undef, M)
    Tn = -1
    for (j, e) in enumerate(eqs)
        (e isa Tuple && (length(e) == 2 || length(e) == 3)) ||
            throw(ArgumentError("equation $j must be a (y, X) or (y, X, names) tuple"))
        y = Float64.(collect(e[1]))
        X = Matrix{Float64}(e[2])
        length(y) == size(X, 1) ||
            throw(ArgumentError("equation $j: length(y)=$(length(y)) ≠ rows(X)=$(size(X, 1))"))
        if Tn < 0
            Tn = length(y)
        else
            Tn == length(y) ||
                throw(ArgumentError("all equations must share the same T; equation $j has $(length(y)), expected $Tn"))
        end
        ys[j] = y
        Xs[j] = X
        vns[j] = if length(e) == 3
            names = String.(collect(e[3]))
            length(names) == size(X, 2) ||
                throw(ArgumentError("equation $j: $(length(names)) names for $(size(X, 2)) regressors"))
            names
        else
            ["eq$(j)_x$(i)" for i in 1:size(X, 2)]
        end
    end
    (ys, Xs, vns)
end

"""
    _sigma_from_resids(resids, T) -> Σ̂

Residual cross-covariance with the classical Zellner divisor `T`:
`Σ̂_ij = ûᵢ'ûⱼ / T`. The divisor is a common scalar across all `(i,j)`, so the SUR/3SLS
*point estimates* are invariant to it; only the reported `Σ̂` and standard errors scale.
"""
function _sigma_from_resids(resids::Vector{Vector{T}}, Tn::Int) where {T<:AbstractFloat}
    M = length(resids)
    S = Matrix{T}(undef, M, M)
    @inbounds for i in 1:M, j in i:M
        s = dot(resids[i], resids[j]) / T(Tn)
        S[i, j] = s
        S[j, i] = s
    end
    S
end

"""
    _gls_normal_eqs(Ws, Xs, ys, Sigmainv) -> (A, rhs)

Assemble the system normal-equation blocks `A = W'(Σ̂⁻¹⊗I)X` and `rhs = W'(Σ̂⁻¹⊗I)y`
WITHOUT materializing the `MT×MT` Kronecker product: loop over the `M×M` equation blocks
and scale each cross-product `Wᵢ'Xⱼ` / `Wᵢ'yⱼ` by the scalar `Σ̂⁻¹_ij`. For SUR pass
`Ws = Xs`; for 3SLS pass the projected regressors `Ws = X̂s` (so `A = X̂'(Σ̂⁻¹⊗I)X̂`).
"""
function _gls_normal_eqs(Ws::Vector{Matrix{T}}, Xs::Vector{Matrix{T}},
                         ys::Vector{Vector{T}}, Sigmainv::Matrix{T}) where {T<:AbstractFloat}
    M = length(Xs)
    ks = [size(X, 2) for X in Xs]
    offs = cumsum(vcat(0, ks))            # block offsets; total K = offs[end]
    K = offs[end]
    A = zeros(T, K, K)
    rhs = zeros(T, K)
    @inbounds for i in 1:M
        ri = (offs[i] + 1):offs[i + 1]
        for j in 1:M
            cj = (offs[j] + 1):offs[j + 1]
            sij = Sigmainv[i, j]
            sij == zero(T) && continue
            A[ri, cj] .+= sij .* (Ws[i]' * Xs[j])
            rhs[ri] .+= sij .* (Ws[i]' * ys[j])
        end
    end
    (A, rhs)
end

"""
    _mcelroy_r2(resids, ys, Sigmainv) -> T

McElroy (1977) system R²: `1 − [û'(Σ̂⁻¹⊗I)û] / [ỹ'(Σ̂⁻¹⊗I)ỹ]`, where `ỹ` demeans each
equation's dependent variable. Both quadratic forms are evaluated block-wise.
"""
function _mcelroy_r2(resids::Vector{Vector{T}}, ys::Vector{Vector{T}},
                     Sigmainv::Matrix{T}) where {T<:AbstractFloat}
    M = length(resids)
    ytil = [y .- mean(y) for y in ys]
    num = zero(T)
    den = zero(T)
    @inbounds for i in 1:M, j in 1:M
        s = Sigmainv[i, j]
        num += s * dot(resids[i], resids[j])
        den += s * dot(ytil[i], ytil[j])
    end
    den <= zero(T) && return T(NaN)
    one(T) - num / den
end

"""
    _split_beta(beta, Xs) -> Vector{Vector{T}}

Split a stacked coefficient vector into per-equation blocks by regressor count.
"""
function _split_beta(beta::Vector{T}, Xs::Vector{Matrix{T}}) where {T<:AbstractFloat}
    ks = [size(X, 2) for X in Xs]
    offs = cumsum(vcat(0, ks))
    [beta[(offs[j] + 1):offs[j + 1]] for j in eachindex(Xs)]
end

# =============================================================================
# SUR estimation
# =============================================================================

"""
    estimate_sur(eqs; iterate=false, tol=1e-8, maxiter=100, restrict=nothing,
                 eqnames=nothing) -> SURModel

Estimate a seemingly-unrelated regressions system (Zellner 1962) by feasible GLS.

# Arguments
- `eqs::Vector{<:Tuple}` — one `(y, X)` (or `(y, X, names)`) tuple per equation. Each `X`
  must include its own intercept column if wanted; all equations must share the same `T`.

# Keyword arguments
- `iterate::Bool=false` — if `true`, alternate between updating `Σ̂` from the current FGLS
  residuals and re-estimating `β̂` until `max|Δβ| < tol`, which converges to the Gaussian MLE
  (iterated FGLS ≡ SUR/FIML for the linear system; Oberhofer & Kmenta 1974).
- `tol::Real=1e-8`, `maxiter::Int=100` — iteration controls.
- `restrict::Union{Nothing,Tuple{<:AbstractMatrix,<:AbstractVector}}=nothing` — linear
  cross-equation restriction `R·vec(B) = r` imposed via restricted GLS
  `β̂_R = β̂ + A R'(R A R')⁻¹(r − Rβ̂)`, `A = (X'(Σ̂⁻¹⊗I)X)⁻¹`. `R` has `q` rows and
  `K = Σ kᵢ` columns (`vec(B)` stacks the equations' coefficients in order).
- `eqnames::Union{Nothing,Vector{String}}=nothing` — equation labels.

# Returns
`SURModel{T}` — per-equation `betas`/`ses`, the residual cross-covariance `Sigma`
(classical divisor `T`), the McElroy system R², the Gaussian log-likelihood, and the
iteration count.

# Estimator
Stacking `y = Xβ + u` with `X = blkdiag(X₁,…,X_M)` and `Cov(u) = Σ ⊗ I_T`,

    Σ̂_ij = ûᵢ'ûⱼ / T           (from equation-by-equation OLS residuals)
    β̂    = (X'(Σ̂⁻¹⊗I)X)⁻¹ X'(Σ̂⁻¹⊗I)y
    V̂(β̂) = (X'(Σ̂⁻¹⊗I)X)⁻¹

Identical regressors across equations ⇒ `β̂ = OLS` exactly (Kruskal 1968). The Kronecker
product is never materialized; the normal equations are assembled block-wise.

# Examples
```julia
pd = load_example(:grunfeld)          # 10-firm investment panel
ge = group_data(pd, "General Electric")
wh = group_data(pd, "Westinghouse")
Xge = hcat(ones(20), ge.data[:, 2], ge.data[:, 3])   # [1, value, capital]
Xwh = hcat(ones(20), wh.data[:, 2], wh.data[:, 3])
m = estimate_sur([(ge.data[:, 1], Xge, ["const", "value", "capital"]),
                  (wh.data[:, 1], Xwh, ["const", "value", "capital"])];
                 eqnames = ["GE", "Westinghouse"])
report(m)
```

# References
- Zellner, A. (1962). *Journal of the American Statistical Association* 57(298), 348-368.
- Kruskal, W. (1968). *Annals of Mathematical Statistics* 39(1), 70-75.
- Oberhofer, W. & Kmenta, J. (1974). *Econometrica* 42(3), 579-590.
- McElroy, M. B. (1977). *Journal of Econometrics* 6(3), 381-387.

!!! note "Scope"
    Full-information maximum likelihood (FIML) is not implemented as a separate joint
    optimizer; iterated FGLS reaches the same Gaussian MLE for the linear SUR system.
"""
function estimate_sur(eqs::AbstractVector; iterate::Bool=false, tol::Real=1e-8,
                      maxiter::Int=100,
                      restrict::Union{Nothing,Tuple{<:AbstractMatrix,<:AbstractVector}}=nothing,
                      eqnames::Union{Nothing,AbstractVector}=nothing)
    ys64, Xs64, vns = _normalize_eqs(eqs)
    T = Float64
    ys = Vector{Vector{T}}(ys64)
    Xs = Vector{Matrix{T}}(Xs64)
    M = length(Xs)
    Tn = length(ys[1])
    K = sum(size(X, 2) for X in Xs)

    enames = eqnames === nothing ? ["eq$(j)" for j in 1:M] : String.(collect(eqnames))
    length(enames) == M || throw(ArgumentError("eqnames must have length $M"))

    # ---- Equation-by-equation OLS → starting β and Σ̂ ----
    # QR-based backslash (`X \ y`) is used for every point-estimate solve: it operates on the
    # design directly (condition κ(X)), never squaring it to κ(X)²=κ(X'X) as `inv(X'X)` would.
    # Grunfeld's raw-scale regressors give κ(X'X)≈1e8, which trips robust_inv's pinv fallback and
    # returns the wrong minimum-norm fit; the covariance inverse below relaxes rcond_tol so a
    # well-posed (merely large-scale) system is inverted rather than regularized.
    betas0 = [X \ y for (X, y) in zip(Xs, ys)]
    resids = [ys[j] .- Xs[j] * betas0[j] for j in 1:M]

    beta = vcat(betas0...)
    Sigma = _sigma_from_resids(resids, Tn)
    A = Matrix{T}(undef, K, K)
    iters = 1
    for it in 1:max(maxiter, 1)
        Sinv = Matrix{T}(robust_inv(Hermitian(Sigma); rcond_tol=eps(T)))
        Amat, rhs = _gls_normal_eqs(Xs, Xs, ys, Sinv)
        A = Matrix{T}(robust_inv(Symmetric(Amat); rcond_tol=eps(T)))
        beta_new = A * rhs
        betablk = _split_beta(beta_new, Xs)
        resids = [ys[j] .- Xs[j] * betablk[j] for j in 1:M]
        if !iterate
            beta = beta_new
            iters = 1
            break
        end
        delta = maximum(abs.(beta_new .- beta))
        beta = beta_new
        iters = it
        # Update Σ̂ for the next sweep (iterated FGLS → Gaussian MLE).
        Sigma = _sigma_from_resids(resids, Tn)
        delta < tol && break
    end

    restricted = restrict !== nothing
    if restricted
        R = Matrix{T}(restrict[1])
        r = Vector{T}(restrict[2])
        size(R, 2) == K || throw(ArgumentError(
            "restriction R must have $K columns (= total regressors); got $(size(R, 2))"))
        size(R, 1) == length(r) || throw(ArgumentError(
            "R has $(size(R, 1)) rows but r has length $(length(r))"))
        RAR = Symmetric(R * A * R')
        RARi = Matrix{T}(robust_inv(RAR; rcond_tol=eps(T)))
        correction = A * R' * (RARi * (r .- R * beta))
        beta = beta .+ correction
        # Restricted covariance: A − A R'(R A R')⁻¹ R A.
        A = A .- A * R' * RARi * (R * A)
        betablk = _split_beta(beta, Xs)
        resids = [ys[j] .- Xs[j] * betablk[j] for j in 1:M]
    end

    betablk = _split_beta(beta, Xs)
    fitted = [Xs[j] * betablk[j] for j in 1:M]

    Sinv = Matrix{T}(robust_inv(Hermitian(Sigma); rcond_tol=eps(T)))
    ses = [sqrt.(max.(diag(A)[_block_range(Xs, j)], zero(T))) for j in 1:M]
    det_sigma = det(Sigma)
    mcr2 = _mcelroy_r2(resids, ys, Sinv)
    # Gaussian system log-likelihood at Σ̂: −(MT/2)(1+ln 2π) − (T/2)ln|Σ̂|.
    loglik = -T(M * Tn) / 2 * (one(T) + log(T(2π))) - T(Tn) / 2 * log(max(det_sigma, T(1e-300)))

    SURModel{T}(enames, vns, betablk, ses, A, Sigma, resids, fitted, Tn,
                det_sigma, mcr2, loglik, iters, iterate, restricted)
end

# Per-equation index range into the stacked coefficient vector.
function _block_range(Xs::Vector{Matrix{T}}, j::Int) where {T}
    ks = [size(X, 2) for X in Xs]
    offs = cumsum(vcat(0, ks))
    (offs[j] + 1):offs[j + 1]
end
