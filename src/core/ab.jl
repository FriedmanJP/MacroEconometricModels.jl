# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
AB-model SVAR (Amisano–Giannini 1997; Lütkepohl 2005, ch. 9): ``A u_t = B ε_t``
with ``ε_t \\sim (0, I)``, estimated by concentrated ML on a zero/fixed pattern.
"""

# =============================================================================
# Pattern
# =============================================================================

"""
    SVARPattern{T}

Zero/fixed pattern for the AB-model ``A u_t = B ε_t``.

Entries of `A` and `B` are `NaN` (free) or a finite number (fixed). Optional
`long_run` imposes the same convention on the long-run impact ``C(1) A^{-1} B``
(Galí 1992; Blanchard–Quah 1989).

# Fields
- `A::Matrix{T}`: contemporaneous left-hand matrix
- `B::Matrix{T}`: contemporaneous shock matrix
- `long_run::Union{Nothing,Matrix{T}}`: long-run pattern, or `nothing`
"""
struct SVARPattern{T<:AbstractFloat}
    A::Matrix{T}
    B::Matrix{T}
    long_run::Union{Nothing,Matrix{T}}
    function SVARPattern(A::AbstractMatrix, B::AbstractMatrix;
                         long_run=nothing)
        size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
        size(A) == size(B) || throw(ArgumentError("A and B must have the same size"))
        T = promote_type(float(eltype(A)), float(eltype(B)))
        Af = Matrix{T}(A)
        Bf = Matrix{T}(B)
        lr = if long_run === nothing
            nothing
        else
            size(long_run) == size(A) ||
                throw(ArgumentError("long_run must be n×n, same as A and B"))
            Matrix{T}(long_run)
        end
        new{T}(Af, Bf, lr)
    end
end

function _ab_promote(pattern::SVARPattern, ::Type{T}) where {T<:AbstractFloat}
    eltype(pattern.A) === T && return pattern
    lr = pattern.long_run === nothing ? nothing : Matrix{T}(pattern.long_run)
    SVARPattern(Matrix{T}(pattern.A), Matrix{T}(pattern.B); long_run=lr)
end

"""
    recursive_pattern(n) -> SVARPattern

Just-identified recursive (Cholesky) AB pattern: `A` is unit lower-triangular
(ones on the diagonal, zeros above, free below) and `B` is diagonal (free
diagonal, zeros off-diagonal). The MLE reproduces [`identify_cholesky`](@ref)
(`Q ≈ I`, `lr_df == 0`).
"""
function recursive_pattern(n::Integer)
    n >= 1 || throw(ArgumentError("n must be positive"))
    T = Float64
    A = Matrix{T}(I, n, n)
    @inbounds for i in 2:n, j in 1:i-1
        A[i, j] = T(NaN)
    end
    B = zeros(T, n, n)
    @inbounds for i in 1:n
        B[i, i] = T(NaN)
    end
    SVARPattern(A, B)
end

"""
    a_model_pattern(A) -> SVARPattern

A-model: supplied pattern on `A` with `B = I`.
"""
function a_model_pattern(A::AbstractMatrix)
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("A must be square"))
    T = float(eltype(A))
    SVARPattern(A, Matrix{T}(I, n, n))
end

"""
    b_model_pattern(B) -> SVARPattern

B-model: supplied pattern on `B` with `A = I`.
"""
function b_model_pattern(B::AbstractMatrix)
    n = size(B, 1)
    size(B, 2) == n || throw(ArgumentError("B must be square"))
    T = float(eltype(B))
    SVARPattern(Matrix{T}(I, n, n), B)
end

"""
    ab_model_pattern(A, B; long_run=nothing) -> SVARPattern

AB-model with patterns on both `A` and `B`.
"""
ab_model_pattern(A::AbstractMatrix, B::AbstractMatrix; long_run=nothing) =
    SVARPattern(A, B; long_run=long_run)

"""
    blanchard_quah_pattern(n) -> SVARPattern

Just-identified Blanchard–Quah long-run pattern: `A = I`, `B` free, and
``C(1) B`` lower triangular (zeros strictly above the diagonal). The MLE
reproduces [`identify_long_run`](@ref) on a stationary VAR.
"""
function blanchard_quah_pattern(n::Integer)
    n >= 1 || throw(ArgumentError("n must be positive"))
    T = Float64
    A = Matrix{T}(I, n, n)
    B = fill(T(NaN), n, n)
    lr = fill(T(NaN), n, n)
    @inbounds for i in 1:n, j in i+1:n
        lr[i, j] = zero(T)
    end
    SVARPattern(A, B; long_run=lr)
end

# =============================================================================
# Result
# =============================================================================

"""
    SVARModel{T} <: AbstractAnalysisResult

Maximum-likelihood AB-model SVAR.

# Fields
- `A`, `B`: estimated contemporaneous matrices
- `Q`: rotation ``Q = L^{-1} A^{-1} B`` with ``L = \\mathrm{chol}(\\hat\\Sigma)``
- `vcov`: covariance of the free-parameter vector, or `nothing`
- `se`: ``n \\times 2n`` matrix `[se_A se_B]` (`NaN` on fixed entries), or `nothing`
- `loglik`: concentrated log-likelihood
- `lr_stat`, `lr_df`, `lr_pvalue`: overidentification LR test
- `pattern`: the restriction pattern
- `identification`: [`IdentificationStatus`](@ref)
- `varnames`: variable labels
"""
struct SVARModel{T<:AbstractFloat} <: AbstractAnalysisResult
    A::Matrix{T}
    B::Matrix{T}
    Q::Matrix{T}
    vcov::Union{Nothing,Matrix{T}}
    se::Union{Nothing,Matrix{T}}
    loglik::T
    lr_stat::T
    lr_df::Int
    lr_pvalue::T
    pattern::SVARPattern{T}
    identification::IdentificationStatus
    varnames::Vector{String}
end

# =============================================================================
# Free parameters
# =============================================================================

_ab_is_free(x) = isnan(x)

function _ab_n_free(pattern::SVARPattern)
    nA = count(_ab_is_free, pattern.A)
    nB = count(_ab_is_free, pattern.B)
    nA + nB
end

function _ab_n_lr(pattern::SVARPattern)
    pattern.long_run === nothing && return 0
    count(isfinite, pattern.long_run)
end

function _ab_pack(A::AbstractMatrix{S}, B::AbstractMatrix{S},
                  pattern::SVARPattern) where {S}
    n = size(A, 1)
    θ = Vector{S}(undef, _ab_n_free(pattern))
    k = 0
    @inbounds for j in 1:n, i in 1:n
        if _ab_is_free(pattern.A[i, j])
            k += 1
            θ[k] = A[i, j]
        end
    end
    @inbounds for j in 1:n, i in 1:n
        if _ab_is_free(pattern.B[i, j])
            k += 1
            θ[k] = B[i, j]
        end
    end
    θ
end

function _ab_unpack(θ::AbstractVector{S}, pattern::SVARPattern) where {S}
    n = size(pattern.A, 1)
    A = Matrix{S}(undef, n, n)
    B = Matrix{S}(undef, n, n)
    k = 0
    @inbounds for j in 1:n, i in 1:n
        if _ab_is_free(pattern.A[i, j])
            k += 1
            A[i, j] = θ[k]
        else
            A[i, j] = S(pattern.A[i, j])
        end
    end
    @inbounds for j in 1:n, i in 1:n
        if _ab_is_free(pattern.B[i, j])
            k += 1
            B[i, j] = θ[k]
        else
            B[i, j] = S(pattern.B[i, j])
        end
    end
    (A, B)
end

function _ab_free_names(pattern::SVARPattern)
    n = size(pattern.A, 1)
    names = String[]
    @inbounds for j in 1:n, i in 1:n
        _ab_is_free(pattern.A[i, j]) && push!(names, "A[$i,$j]")
    end
    @inbounds for j in 1:n, i in 1:n
        _ab_is_free(pattern.B[i, j]) && push!(names, "B[$i,$j]")
    end
    names
end

function _ab_orders(pattern::SVARPattern)
    n = size(pattern.A, 1)
    orders = zeros(Int, n)
    @inbounds for j in 1:n, i in 1:n
        aij = pattern.A[i, j]
        isfinite(aij) && aij == 0 && (orders[j] += 1)
        bij = pattern.B[i, j]
        isfinite(bij) && bij == 0 && (orders[j] += 1)
        if pattern.long_run !== nothing
            lij = pattern.long_run[i, j]
            isfinite(lij) && lij == 0 && (orders[j] += 1)
        end
    end
    orders
end

function _ab_fill(pattern::SVARPattern{T}, A_src::AbstractMatrix,
                  B_src::AbstractMatrix) where {T}
    n = size(pattern.A, 1)
    A = copy(pattern.A)
    B = copy(pattern.B)
    @inbounds for j in 1:n, i in 1:n
        _ab_is_free(pattern.A[i, j]) && (A[i, j] = T(A_src[i, j]))
        _ab_is_free(pattern.B[i, j]) && (B[i, j] = T(B_src[i, j]))
    end
    (A, B)
end

function _is_recursive_pattern(pattern::SVARPattern)
    pattern.long_run === nothing || return false
    n = size(pattern.A, 1)
    @inbounds for i in 1:n, j in 1:n
        aij, bij = pattern.A[i, j], pattern.B[i, j]
        if i == j
            (isfinite(aij) && aij == 1) || return false
            _ab_is_free(bij) || return false
        elseif i > j
            _ab_is_free(aij) || return false
            (isfinite(bij) && bij == 0) || return false
        else
            (isfinite(aij) && aij == 0) || return false
            (isfinite(bij) && bij == 0) || return false
        end
    end
    true
end

function _is_bq_pattern(pattern::SVARPattern)
    pattern.long_run !== nothing || return false
    n = size(pattern.A, 1)
    pattern.A ≈ I(n) || return false
    all(_ab_is_free, pattern.B) || return false
    lr = pattern.long_run
    @inbounds for i in 1:n, j in 1:n
        if i < j
            (isfinite(lr[i, j]) && lr[i, j] == 0) || return false
        else
            _ab_is_free(lr[i, j]) || return false
        end
    end
    true
end

# =============================================================================
# Likelihood
# =============================================================================

function _ab_concentrated_nll(A::AbstractMatrix{S}, B::AbstractMatrix{S},
                              Sigma::AbstractMatrix, Tobs::Real) where {S}
    ldA, sA = logabsdet(A)
    ldB, sB = logabsdet(B)
    (!isfinite(ldA) || !isfinite(ldB) || sA == 0 || sB == 0) && return S(Inf)
    K = try
        B \ A
    catch
        return S(Inf)
    end
    any(!isfinite, K) && return S(Inf)
    KS = K * Sigma
    trterm = zero(S)
    n = size(A, 1)
    @inbounds for j in 1:n, i in 1:n
        trterm += KS[i, j] * K[i, j]
    end
    S(Tobs) * (ldB - ldA + trterm / 2)
end

function _ab_lr_penalty(A::AbstractMatrix{S}, B::AbstractMatrix{S},
                        C1::AbstractMatrix, lr::AbstractMatrix) where {S}
    B0 = try
        A \ B
    catch
        return S(Inf)
    end
    any(!isfinite, B0) && return S(Inf)
    LR = C1 * B0
    s = zero(S)
    n = size(A, 1)
    @inbounds for j in 1:n, i in 1:n
        tij = lr[i, j]
        isfinite(tij) && (s += (LR[i, j] - S(tij))^2)
    end
    s
end

function _ab_nll_theta(θ::AbstractVector{S}, pattern::SVARPattern,
                       Sigma::AbstractMatrix, Tobs::Real,
                       C1, penalty::Real) where {S}
    A, B = _ab_unpack(θ, pattern)
    nll = _ab_concentrated_nll(A, B, Sigma, Tobs)
    if pattern.long_run !== nothing && C1 !== nothing && isfinite(nll)
        nll += S(penalty) * _ab_lr_penalty(A, B, C1, pattern.long_run)
    end
    nll
end

function _ab_sigma_r(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    B0 = A \ B
    B0 * B0'
end

# =============================================================================
# Identification (Amisano–Giannini rank/order)
# =============================================================================

function _ab_mapping(θ::AbstractVector{S}, pattern::SVARPattern, C1) where {S}
    A, B = _ab_unpack(θ, pattern)
    B0 = A \ B
    v = _vech(B0 * B0')
    if pattern.long_run !== nothing && C1 !== nothing
        LR = C1 * B0
        g = S[]
        n = size(A, 1)
        @inbounds for j in 1:n, i in 1:n
            isfinite(pattern.long_run[i, j]) && push!(g, LR[i, j])
        end
        return vcat(v, g)
    end
    v
end

function _ab_jacobian_rank(θ::AbstractVector{T}, pattern::SVARPattern,
                           C1) where {T<:AbstractFloat}
    isempty(θ) && return 0
    J = try
        ForwardDiff.jacobian(x -> _ab_mapping(x, pattern, C1), θ)
    catch
        return 0
    end
    any(!isfinite, J) && return 0
    _matrix_rank(J)
end

function _ab_chol_start(pattern::SVARPattern{T}, Sigma::AbstractMatrix{T}) where {T}
    L = safe_cholesky(Matrix{T}(Sigma))
    n = size(L, 1)
    d = diag(L)
    diag_floor = sqrt(eps(T))
    d_safe = [max(abs(d[i]), diag_floor) for i in 1:n]
    L_unit = L / Diagonal(d_safe)
    A_src = L_unit \ Matrix{T}(I, n, n)
    B_src = Matrix{T}(L)
    _ab_fill(pattern, A_src, B_src)
end

function _ab_classify(n_free::Int, n_lr::Int, n_cov::Int, rank_J::Int,
                      orders::Vector{Int})
    n_over = max(n_cov + n_lr - n_free, 0)
    status = if n_free > n_cov + n_lr || rank_J < n_free
        :under
    elseif n_over > 0
        :over
    else
        :exact
    end
    ranks = copy(orders)
    IdentificationStatus(status, ranks, orders, n_over)
end

"""
    check_identification(pattern::SVARPattern, model::VARModel; n_points=10, rng)
    check_identification(pattern::SVARPattern, n::Int)

Amisano–Giannini (1997) order and local rank conditions for an AB-model
pattern. The order condition is ``n_{\\mathrm{free}} \\le n(n+1)/2 + n_{\\mathrm{LR}}``.
The rank condition is that the Jacobian of ``(\\mathrm{vech}(A^{-1}BB'A^{-T}), g_{\\mathrm{LR}})``
with respect to the free parameters has full column rank at a generic point.

`:under` means the pattern cannot identify ``(A, B)``. Overidentified patterns
are `:over` and remain estimable (LR test).
"""
function check_identification(pattern::SVARPattern, n::Int)
    n >= 1 || throw(ArgumentError("n must be positive"))
    size(pattern.A, 1) == n || throw(ArgumentError(
        "Pattern dimension ($(size(pattern.A, 1))) must match n=$n"))
    n_free = _ab_n_free(pattern)
    n_lr = _ab_n_lr(pattern)
    n_cov = n * (n + 1) ÷ 2
    # Order-only: treat restrictions as independent (rank = n_free when feasible).
    rank_J = min(n_free, n_cov + n_lr)
    _ab_classify(n_free, n_lr, n_cov, rank_J, _ab_orders(pattern))
end

function check_identification(pattern::SVARPattern, model::VARModel{T};
                              n_points::Int=10,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(model)
    n_points >= 1 || throw(ArgumentError("n_points must be ≥ 1, got $n_points"))
    pattern = _ab_promote(pattern, T)
    size(pattern.A, 1) == n || throw(ArgumentError(
        "Pattern dimension ($(size(pattern.A, 1))) must match model ($n)"))
    n_free = _ab_n_free(pattern)
    n_lr = _ab_n_lr(pattern)
    n_cov = n * (n + 1) ÷ 2
    orders = _ab_orders(pattern)
    C1 = pattern.long_run === nothing ? nothing : _C1_from_B(model.B, n, model.p)
    rank_J = 0
    if n_free > 0 && n_free <= n_cov + n_lr
        try
            A0, B0 = _ab_chol_start(pattern, model.Sigma)
            θ0 = _ab_pack(A0, B0, pattern)
            rank_J = max(rank_J, _ab_jacobian_rank(θ0, pattern, C1))
        catch err
            _is_rejectable_draw_error(err) || rethrow(err)
        end
        for _ in 1:n_points
            try
                θ = randn(rng, T, n_free)
                A, B = _ab_unpack(θ, pattern)
                @inbounds for i in 1:n
                    _ab_is_free(pattern.A[i, i]) && (A[i, i] += T(2n))
                    _ab_is_free(pattern.B[i, i]) && (B[i, i] += T(n))
                end
                θ = _ab_pack(A, B, pattern)
                rank_J = max(rank_J, _ab_jacobian_rank(θ, pattern, C1))
            catch err
                _is_rejectable_draw_error(err) || rethrow(err)
                continue
            end
        end
    elseif n_free == 0
        rank_J = 0
    end
    # Closed-form just-identified patterns are identified independently of a
    # noisy generic Jacobian (recursive ≡ Cholesky; BQ ≡ identify_long_run).
    if n_free == n_cov + n_lr && (_is_recursive_pattern(pattern) || _is_bq_pattern(pattern))
        rank_J = n_free
    end
    _ab_classify(n_free, n_lr, n_cov, rank_J, orders)
end

# =============================================================================
# Estimation
# =============================================================================

function _ab_sign_normalize!(A::Matrix{T}, B::Matrix{T}, pattern::SVARPattern{T},
                             C1) where {T<:AbstractFloat}
    n = size(A, 1)
    B0 = A \ B
    target = if pattern.long_run !== nothing && C1 !== nothing
        C1 * B0
    else
        B0
    end
    @inbounds for j in 1:n
        target[j, j] >= 0 && continue
        if any(i -> _ab_is_free(pattern.B[i, j]), 1:n)
            B[:, j] .*= -one(T)
        elseif any(i -> _ab_is_free(pattern.A[j, i]), 1:n)
            A[j, :] .*= -one(T)
        end
    end
    (A, B)
end

function _ab_se(θ::Vector{T}, pattern::SVARPattern{T}, Sigma::Matrix{T},
                Tobs::T, C1, penalty::T) where {T<:AbstractFloat}
    isempty(θ) && return nothing, nothing
    obj = x -> _ab_nll_theta(x, pattern, Sigma, Tobs, C1, penalty)
    H = try
        ForwardDiff.hessian(obj, θ)
    catch
        return nothing, nothing
    end
    any(!isfinite, H) && return nothing, nothing
    H = Matrix{T}((H + H') / 2)
    V = try
        Matrix{T}(robust_inv(Hermitian(H); silent=true))
    catch
        return nothing, nothing
    end
    any(<(zero(T)), diag(V)) && return nothing, nothing
    n = size(pattern.A, 1)
    seAB = fill(T(NaN), n, 2n)
    k = 0
    @inbounds for j in 1:n, i in 1:n
        if _ab_is_free(pattern.A[i, j])
            k += 1
            seAB[i, j] = sqrt(max(V[k, k], zero(T)))
        end
    end
    @inbounds for j in 1:n, i in 1:n
        if _ab_is_free(pattern.B[i, j])
            k += 1
            seAB[i, n + j] = sqrt(max(V[k, k], zero(T)))
        end
    end
    (V, seAB)
end

function _ab_optimize(θ0::Vector{T}, pattern::SVARPattern{T}, Sigma::Matrix{T},
                      Tobs::T, C1, penalty::T, max_iter::Int) where {T<:AbstractFloat}
    obj = θ -> _ab_nll_theta(θ, pattern, Sigma, Tobs, C1, penalty)
    g! = (G, θ) -> ForwardDiff.gradient!(G, obj, θ)
    Optim.optimize(obj, g!, θ0, Optim.LBFGS(),
                   Optim.Options(iterations=max_iter, g_tol=T(1e-8),
                                 f_reltol=T(1e-12), allow_f_increases=true))
end

"""
    estimate_svar(model, pattern; n_starts=5, rng) -> SVARModel

Maximum-likelihood estimation of the AB-model ``A u_t = B ε_t``
(Amisano–Giannini 1997). The concentrated log-likelihood is

```math
\\ell(A,B) = -\\frac{T}{2}\\bigl[\\log|B|^2 - \\log|A|^2
+ \\mathrm{tr}(B^{-1} A \\hat\\Sigma A' B^{-T})\\bigr]
```

[`check_identification`](@ref) is called first; `:under` throws
[`IdentificationError`](@ref). Just-identified recursive and Blanchard–Quah
patterns use the Cholesky / long-run closed form (reproducing
[`identify_cholesky`](@ref) and [`identify_long_run`](@ref)). General patterns
are maximised with `Optim.LBFGS` and `ForwardDiff.gradient` from `n_starts`
starting values. Column signs are normalised so the impact (or long-run impact)
diagonal is positive.

The overidentification statistic is
``\\mathrm{LR} = T(\\log|\\hat\\Sigma_r| - \\log|\\hat\\Sigma|) \\sim \\chi^2(n_{\\mathrm{over}})``.
"""
function estimate_svar(model::VARModel{T}, pattern::SVARPattern;
                       n_starts::Int=5,
                       max_iter::Int=400,
                       rng::AbstractRNG=Random.default_rng(),
                       kwargs...) where {T<:AbstractFloat}
    n = nvars(model)
    n_starts >= 1 || throw(ArgumentError("n_starts must be ≥ 1, got $n_starts"))
    pattern = _ab_promote(pattern, T)
    size(pattern.A, 1) == n || throw(ArgumentError(
        "Pattern dimension ($(size(pattern.A, 1))) must match model ($n)"))
    st = check_identification(pattern, model; rng=copy(rng))
    if st.status === :under
        throw(IdentificationError(
            "SVAR is underidentified by the AB rank/order condition " *
            "(status=:under; n_free=$(_ab_n_free(pattern)), " *
            "n_lr=$(_ab_n_lr(pattern)), n_cov=$(n * (n + 1) ÷ 2))."))
    end

    Sigma = Matrix{T}(model.Sigma)
    Tobs = T(size(model.U, 1))
    C1 = pattern.long_run === nothing ? nothing : _C1_from_B(model.B, n, model.p)
    penalty = Tobs * T(1e6)

    A = Matrix{T}(undef, n, n)
    B = Matrix{T}(undef, n, n)
    used_closed = false
    if _is_recursive_pattern(pattern)
        A, B = _ab_chol_start(pattern, Sigma)
        used_closed = true
    elseif _is_bq_pattern(pattern)
        Q_lr = identify_long_run(model)
        L = cholesky_factor(model)
        A = Matrix{T}(I, n, n)
        B = L * Q_lr
        used_closed = true
    end

    n_free = _ab_n_free(pattern)
    if !used_closed && n_free > 0
        A_s, B_s = _ab_chol_start(pattern, Sigma)
        θ_chol = _ab_pack(A_s, B_s, pattern)
        best_nll = T(Inf)
        best_θ = copy(θ_chol)
        for s in 1:n_starts
            θ0 = s == 1 ? copy(θ_chol) :
                 θ_chol .+ T(0.25) .* randn(rng, T, n_free)
            result = try
                _ab_optimize(θ0, pattern, Sigma, Tobs, C1, penalty, max_iter)
            catch err
                _is_rejectable_draw_error(err) || rethrow(err)
                continue
            end
            nll = T(Optim.minimum(result))
            if isfinite(nll) && nll < best_nll
                best_nll = nll
                best_θ = Vector{T}(Optim.minimizer(result))
            end
        end
        isfinite(best_nll) || throw(ConvergenceError(
            "estimate_svar failed to converge from $n_starts starts"))
        A, B = _ab_unpack(best_θ, pattern)
    elseif n_free == 0
        A = copy(pattern.A)
        B = copy(pattern.B)
    end

    _ab_sign_normalize!(A, B, pattern, C1)

    L = cholesky_factor(model)
    Q = L \ (A \ B)
    nll = _ab_concentrated_nll(A, B, Sigma, Tobs)
    loglik = -nll
    Sigma_r = _ab_sigma_r(A, B)
    ld_r = logabsdet(Hermitian(Sigma_r))[1]
    ld_u = logabsdet(Hermitian(Sigma))[1]
    lr_stat = max(Tobs * (ld_r - ld_u), zero(T))
    lr_df = st.n_overidentifying
    lr_pvalue = lr_df == 0 ? one(T) : T(ccdf(Chisq(lr_df), Float64(lr_stat)))

    θ = n_free > 0 ? _ab_pack(A, B, pattern) : T[]
    vcov, se = _ab_se(θ, pattern, Sigma, Tobs, C1, penalty)

    SVARModel{T}(A, B, Matrix{T}(Q), vcov, se, T(loglik), T(lr_stat), lr_df,
                 T(lr_pvalue), pattern, st, copy(model.varnames))
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::SVARModel{T}) where {T}
    n = size(m.A, 1)
    spec = Any[
        "Variables"          n;
        "Identification"     String(m.identification.status);
        "Over-ID df"         m.lr_df;
        "Log-likelihood"     _fmt(m.loglik);
        "LR statistic"       _fmt(m.lr_stat);
        "LR p-value"         _fmt(m.lr_pvalue; digits=4)
    ]
    _pretty_table(io, spec;
        title = "AB-Model SVAR",
        column_labels = ["", ""],
        alignment = [:l, :r])
    names = _ab_free_names(m.pattern)
    if !isempty(names)
        coefs = _ab_pack(m.A, m.B, m.pattern)
        ses = if m.vcov !== nothing
            sqrt.(max.(diag(m.vcov), zero(T)))
        else
            fill(T(NaN), length(coefs))
        end
        _coef_table(io, "Free elements", names, coefs, ses)
        _sig_legend(io)
    end
    _matrix_table(io, m.A, "A"; row_labels=m.varnames, col_labels=m.varnames)
    _matrix_table(io, m.B, "B"; row_labels=m.varnames, col_labels=m.varnames)
    return nothing
end
