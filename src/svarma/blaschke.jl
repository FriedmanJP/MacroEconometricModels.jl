# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Fundamentalness regime classification, GMR (2020) bivariate Figure 6 geometry,
Lippi–Reichlin (1994) Blaschke equivalence algebra, causal IRFs/FEVD, and
user-facing `SVARMAModel` constructors.
"""

using LinearAlgebra

# =============================================================================
# Fundamentalness Regime & GMR (2020, Fig. 6) Classifier
# =============================================================================

function _svarma_resolve_theta_n(Theta, C)
    if C !== nothing
        nr, nc = size(C)
        nr == nc || throw(ArgumentError("C must be square, got $(size(C))"))
        return nr
    elseif Theta isa AbstractArray{<:Real,3}
        nr, nc, _ = size(Theta)
        nr == nc || throw(ArgumentError("Theta slices must be square, got $(size(Theta))"))
        return nr
    elseif Theta isa AbstractMatrix
        nr, nc = size(Theta)
        nr == nc || throw(ArgumentError("Theta must be square, got $(size(Theta))"))
        return nr
    elseif Theta isa AbstractVector && !isempty(Theta)
        nr, nc = size(Theta[1])
        nr == nc || throw(ArgumentError("Theta[1] must be square, got $(size(Theta[1]))"))
        return nr
    else
        throw(ArgumentError("Cannot infer dimension n when both Theta and C are empty or nothing"))
    end
end

"""
    fundamentalness_regime(Theta, C=nothing; tol=1e-8) -> NamedTuple
    fundamentalness_regime(m::SVARMAModel; tol=1e-8) -> NamedTuple

Classify the moving-average polynomial ``\\Theta(L) = I_n + \\sum_{j=1}^q \\Theta_j L^j``
by the location of its companion matrix eigenvalues relative to the unit circle:

```math
F_{\\text{ma}} = \\begin{bmatrix} -\\Theta_1 & -\\Theta_2 & \\cdots & -\\Theta_q \\\\ I_n & 0 & \\cdots & 0 \\\\ \\vdots & \\ddots & \\ddots & \\vdots \\\\ 0 & \\cdots & I_n & 0 \\end{bmatrix}
```

where:
- `F_{\\text{ma}}` is the ``nq \\times nq`` moving-average companion matrix
- `\\Theta_j` are the ``n \\times n`` moving-average lag matrices (``j = 1, \\ldots, q``)
- `I_n` is the ``n \\times n`` identity matrix

Classifies `regime` as:
- `:fundamental` — all eigenvalues satisfy ``|\\lambda_i| < 1 - \\text{tol}``
- `:mixed_nonfundamental` — both ``|\\lambda_i| < 1 - \\text{tol}`` and ``|\\lambda_i| > 1 + \\text{tol}`` eigenvalues exist
- `:purely_nonfundamental` — all eigenvalues satisfy ``|\\lambda_i| > 1 + \\text{tol}``
- `:unit_root` — at least one eigenvalue lies within `tol` of the unit circle (`||\\lambda_i| - 1| \\le \\text{tol}`)

For bivariate `VMA(1)` systems (`n == 2 && q == 1`), additionally assigns `gmr_region`
to Gouriéroux, Monfort & Renne (2020, Fig. 6, Appendix B.1) regions `:I..:V` in the
``(\\text{tr}(-\\Theta_1), \\det(-\\Theta_1))`` plane separated by ``1 \\mp \\text{tr} + \\det = 0``
and ``\\det = 1`` (`:unit_root` on boundaries, `:none` otherwise).

Returns `(regime, n_fundamental, n_nonfundamental, eigenvalues, gmr_region)`.
"""
function fundamentalness_regime(Theta, C=nothing; tol::Real=1e-8)
    n = _svarma_resolve_theta_n(Theta, C)
    T_el = float(promote_type(_svarma_eltype(Theta), _svarma_eltype(C)))
    T_real = T_el <: AbstractFloat ? T_el : Float64
    tol_T = max(T_real(tol), T_real(16) * eps(T_real))

    Theta_3d = _svarma_to_3d(Theta, n, T_real)
    q = size(Theta_3d, 3)

    if q == 0
        return (
            regime = :fundamental,
            n_fundamental = 0,
            n_nonfundamental = 0,
            eigenvalues = Complex{T_real}[],
            gmr_region = :none,
        )
    end

    F_ma = _svarma_companion(Theta_3d)
    eigs = Complex{T_real}.(eigvals(F_ma))
    moduli = abs.(eigs)
    n_unit = count(m -> abs(m - one(T_real)) <= tol_T, moduli)
    n_fund = count(m -> m < one(T_real) - tol_T, moduli)
    n_nonfund = count(m -> m > one(T_real) + tol_T, moduli)

    regime = if n_unit > 0
        :unit_root
    elseif n_nonfund == 0
        :fundamental
    elseif n_fund == 0
        :purely_nonfundamental
    else
        :mixed_nonfundamental
    end

    gmr_region = _svarma_gmr_region(Theta_3d, tol_T)
    if regime === :unit_root && n == 2 && q == 1
        gmr_region = :unit_root
    end

    return (
        regime = regime,
        n_fundamental = n_fund,
        n_nonfundamental = n_nonfund,
        eigenvalues = eigs,
        gmr_region = gmr_region,
    )
end

fundamentalness_regime(m::SVARMAModel; tol::Real=1e-8) =
    fundamentalness_regime(m.Theta, m.C; tol=tol)

# =============================================================================
# Lippi–Reichlin (1994) / GMR (2020) Blaschke Root-Flipping Algebra
# =============================================================================

function _svarma_poly_div_1mlz(W::Vector{Vector{CT}}, lam::CT) where {CT<:Complex}
    q = length(W) - 1
    n = length(W[1])
    Q = [zeros(CT, n) for _ in 1:q]
    q == 0 && return Q, norm(W[1])

    if abs(lam) <= one(real(CT))
        Q[1] = copy(W[1])
        for k in 2:q
            Q[k] = W[k] .+ lam .* Q[k - 1]
        end
        rem = norm(W[q + 1] .+ lam .* Q[q])
    else
        inv_lam = one(CT) / lam
        Q[q] = -W[q + 1] .* inv_lam
        for k in (q - 1):-1:1
            Q[k] = (Q[k + 1] .- W[k + 1]) .* inv_lam
        end
        rem = norm(Q[1] .- W[1])
    end
    return Q, rem
end

function _svarma_eval_poly_scaled(A_poly::Vector{Matrix{CT}}, lam::CT) where {CT<:Complex}
    q = length(A_poly) - 1
    n = size(A_poly[1], 1)
    M = zeros(CT, n, n)
    if abs(lam) <= one(real(CT))
        # λ^q A(1/λ) = Σ_{k=0}^q A_k λ^{q-k} keeps all powers ≤ 1
        pow = one(CT)
        for k in q:-1:0
            M .+= pow .* A_poly[k + 1]
            pow *= lam
        end
    else
        # A(1/λ) = Σ_{k=0}^q A_k (1/λ)^k keeps all powers ≤ 1
        inv_lam = one(CT) / lam
        pow = one(CT)
        for k in 0:q
            M .+= pow .* A_poly[k + 1]
            pow *= inv_lam
        end
    end
    M
end

function _svarma_flip_root(A_poly::Vector{Matrix{CT}}, lam::CT, copy_offset::Int,
                           tol_T::Real) where {CT<:Complex}
    q = length(A_poly) - 1
    n = size(A_poly[1], 1)
    M = _svarma_eval_poly_scaled(A_poly, lam)
    F = svd(M)
    tol_null = max(real(CT)(tol_T), sqrt(eps(real(CT)))) * max(F.S[1], one(real(CT)))
    nullity = count(s -> s <= tol_null, F.S)
    pick_offset = min(copy_offset, max(nullity - 1, 0))
    u = F.V[:, n - pick_offset]

    W = [A_poly[k + 1] * u for k in 0:q]
    Q_vec, _ = _svarma_poly_div_1mlz(W, lam)

    P = Matrix{CT}(I, n, n) .- u * u'
    A_new = [A_poly[k + 1] * P for k in 0:q]
    cl = conj(lam)
    if q >= 1
        A_new[1] .+= (-cl .* Q_vec[1]) * u'
        for k in 1:(q - 1)
            A_new[k + 1] .+= (Q_vec[k] .- cl .* Q_vec[k + 1]) * u'
        end
        A_new[q + 1] .+= Q_vec[q] * u'
    end
    return A_new
end

function _svarma_group_roots(eigs::Vector{Complex{T}}, tol_T::T) where {T<:AbstractFloat}
    groups = Vector{Vector{Complex{T}}}()
    flippable = filter(lam -> abs(lam) > tol_T, eigs)
    used = falses(length(flippable))
    imag_tol = max(T(10) * tol_T, sqrt(eps(T)) * T(10))

    for i in eachindex(flippable)
        used[i] && continue
        lam = flippable[i]
        used[i] = true
        if abs(imag(lam)) <= imag_tol * max(one(T), abs(lam))
            push!(groups, Complex{T}[Complex{T}(real(lam), zero(T))])
        else
            target = conj(lam)
            best_j = 0
            best_d = T(Inf)
            for j in (i + 1):length(flippable)
                used[j] && continue
                d = abs(flippable[j] - target)
                if d < best_d
                    best_d = d
                    best_j = j
                end
            end
            if best_j > 0 && best_d <= max(T(1e-3), T(100) * tol_T) * max(one(T), abs(lam))
                used[best_j] = true
                push!(groups, Complex{T}[lam, flippable[best_j]])
            else
                push!(groups, Complex{T}[lam, conj(lam)])
            end
        end
    end
    return groups
end

"""
    blaschke_representations(Theta, C=nothing; tol=1e-8) -> Vector{NamedTuple}
    blaschke_representations(m::SVARMAModel; tol=1e-8) -> Vector{NamedTuple}

Enumerate all ``2^r`` second-moment-equivalent real SVARMA moving-average
representations ``(\\Theta^*(L), C^*)`` generated by flipping subsets of the ``r``
nonzero real roots and complex-conjugate root pairs of ``\\det \\Theta(z) = 0`` across
the unit circle via Lippi–Reichlin (1994) elementary Blaschke matrices:

```math
\\Theta^*(z) C^* = \\Theta(z) C B(z), \\qquad B(e^{-i\\omega}) B(e^{-i\\omega})^* = I_n
```

where:
- `\\Theta(z)` is the ``n \\times n`` moving-average lag polynomial ``I_n + \\sum_{j=1}^q \\Theta_j z^j``
- `C` is the ``n \\times n`` nonsingular instantaneous structural impact matrix (`I_n` if omitted)
- `B(z)` is a real-rational para-unitary Blaschke matrix preserving the spectral density ``f_u(\\omega)``
- `(\\Theta^*(z), C^*)` is a second-moment-equivalent real moving-average representation with inverted flipped roots

Elementary Blaschke factors flip each real root ``\\lambda \\mapsto \\lambda^{-1}`` and each
complex-conjugate pair ``(\\lambda, \\bar{\\lambda}) \\mapsto (\\lambda^{-1}, \\bar{\\lambda}^{-1})``
while maintaining exact degree ``\\le q`` and real coefficients. The first element
corresponds to the unmodified input `(Theta, C)`.

Returns a `Vector` of `NamedTuple`s with fields
`(Theta, C, Sigma, regime, n_nonfundamental, eigenvalues, gmr_region)`.
"""
function blaschke_representations(Theta, C=nothing; tol::Real=1e-8)
    n = _svarma_resolve_theta_n(Theta, C)
    T_el = float(promote_type(_svarma_eltype(Theta), _svarma_eltype(C)))
    T_real = T_el <: AbstractFloat ? T_el : Float64
    tol_T = max(T_real(tol), T_real(16) * eps(T_real))

    Theta_3d = _svarma_to_3d(Theta, n, T_real)
    q = size(Theta_3d, 3)
    C_mat = C === nothing ? Matrix{T_real}(I, n, n) : Matrix{T_real}(C)
    _svarma_check_C_nonsingular(C_mat)

    reg0 = fundamentalness_regime(Theta_3d, C_mat; tol=tol)
    if reg0.regime === :unit_root
        throw(IdentificationError(
            "Cannot compute Blaschke representations when an MA eigenvalue lies on or near " *
            "the unit circle (tol=$tol_T)."))
    end

    if q == 0
        return [(
            Theta = Theta_3d,
            C = C_mat,
            Sigma = C_mat * transpose(C_mat),
            regime = reg0.regime,
            n_nonfundamental = reg0.n_nonfundamental,
            eigenvalues = reg0.eigenvalues,
            gmr_region = reg0.gmr_region,
        )]
    end

    groups = _svarma_group_roots(reg0.eigenvalues, tol_T)
    r = length(groups)
    n_reps = 1 << r
    CT = Complex{T_real}
    cluster_tol = max(sqrt(eps(T_real)) * T_real(100), tol_T * T_real(100))

    reps = Vector{NamedTuple}(undef, n_reps)
    for mask in 0:(n_reps - 1)
        if mask == 0
            reps[1] = (
                Theta = copy(Theta_3d),
                C = copy(C_mat),
                Sigma = C_mat * transpose(C_mat),
                regime = reg0.regime,
                n_nonfundamental = reg0.n_nonfundamental,
                eigenvalues = copy(reg0.eigenvalues),
                gmr_region = reg0.gmr_region,
            )
            continue
        end

        A_poly = Matrix{CT}[CT.(C_mat)]
        for k in 1:q
            push!(A_poly, CT.(@view(Theta_3d[:, :, k]) * C_mat))
        end

        has_complex = false
        for g_idx in 1:r
            ((mask >> (g_idx - 1)) & 1) == 1 || continue
            grp = groups[g_idx]
            length(grp) == 2 && (has_complex = true)

            # Count earlier unflipped groups sharing the same root to select distinct
            # nullspace directions for diagonalizable repeated roots.
            lam_lead = grp[1]
            unflipped_prior = 0
            for prev in 1:(g_idx - 1)
                if ((mask >> (prev - 1)) & 1) == 0 &&
                   abs(groups[prev][1] - lam_lead) <= cluster_tol * max(one(T_real), abs(lam_lead))
                    unflipped_prior += 1
                end
            end

            for lam in grp
                A_poly = _svarma_flip_root(A_poly, lam, unflipped_prior, tol_T)
            end
        end

        A_real = if !has_complex
            Matrix{T_real}[real.(M) for M in A_poly]
        else
            S2_raw = real.(A_poly[1] * A_poly[1]')
            S2 = Matrix{T_real}((S2_raw .+ transpose(S2_raw)) ./ T_real(2))
            L2 = Matrix{T_real}(safe_cholesky(S2; silent=true))
            U_unit = L2 \ A_poly[1]
            Matrix{T_real}[real.(M * U_unit') for M in A_poly]
        end

        C_alt = A_real[1]
        C_alt_inv = Matrix{T_real}(robust_inv(C_alt; silent=true))
        Theta_alt = Array{T_real,3}(undef, n, n, q)
        for k in 1:q
            @views Theta_alt[:, :, k] .= A_real[k + 1] * C_alt_inv
        end

        reg_alt = fundamentalness_regime(Theta_alt, C_alt; tol=tol)
        reps[mask + 1] = (
            Theta = Theta_alt,
            C = C_alt,
            Sigma = C_alt * transpose(C_alt),
            regime = reg_alt.regime,
            n_nonfundamental = reg_alt.n_nonfundamental,
            eigenvalues = reg_alt.eigenvalues,
            gmr_region = reg_alt.gmr_region,
        )
    end
    return reps
end

blaschke_representations(m::SVARMAModel; tol::Real=1e-8) =
    blaschke_representations(m.Theta, m.C; tol=tol)

# =============================================================================
# Causal Structural IRF & FEVD for SVARMAModel
# =============================================================================

"""
    irf(m::SVARMAModel{T}, H::Int=20) -> ImpulseResponse{T}

Compute causal structural impulse responses for an `SVARMAModel` over horizons
``h = 0, \\ldots, H - 1`` (stored at indices `1:H`):

```math
\\Psi(L) = \\Phi(L)^{-1} \\Theta(L) = \\sum_{h=0}^{\\infty} \\Psi_h L^h, \\qquad \\mathrm{IRF}_h = \\Psi_h C
```

where:
- `\\Phi(L)` is the autoregressive matrix polynomial ``I_n - \\sum_{k=1}^p \\Phi_k L^k``
- `\\Theta(L)` is the moving-average matrix polynomial ``I_n + \\sum_{j=1}^q \\Theta_j L^j``
- `\\Psi_h` is the ``n \\times n`` reduced-form Wold-type multiplier at horizon ``h``
- `C` is the ``n \\times n`` structural impact matrix (`\\mathrm{IRF}_0 = C`)
"""
function irf(m::SVARMAModel{T}, H::Int=20) where {T<:AbstractFloat}
    validate_horizon(H)
    n = nvars(m)
    p = m.p
    q = m.q

    Psi = Vector{Matrix{T}}(undef, H)
    Psi[1] = Matrix{T}(I, n, n)
    for h in 1:(H - 1)
        acc = h <= q ? copy(m.Theta[:, :, h]) : zeros(T, n, n)
        for k in 1:min(h, p)
            mul!(acc, @view(m.Phi[:, :, k]), Psi[h - k + 1], one(T), one(T))
        end
        Psi[h + 1] = acc
    end

    values = Array{T,3}(undef, H, n, n)
    @inbounds for h in 1:H
        values[h, :, :] = Psi[h] * m.C
    end
    ci_zero = zeros(T, H, n, n)
    return ImpulseResponse{T}(values, ci_zero, copy(ci_zero), H,
                              copy(m.varnames), copy(m.shock_names), :none)
end

"""
    fevd(m::SVARMAModel{T}, H::Int=20) -> FEVD{T}

Compute structural forecast error variance decomposition for an `SVARMAModel` over
horizons ``1, \\ldots, H`` from the causal structural impulse responses ``\\Psi_h C``.

Because structural shocks ``\\eta_t`` are mutually and serially independent with
unit variance, rows of `proportions[:, :, h]` sum to `1.0` at every horizon `h`.
"""
function fevd(m::SVARMAModel{T}, H::Int=20) where {T<:AbstractFloat}
    validate_horizon(H)
    ir = irf(m, H)
    n = nvars(m)
    decomp, props = _compute_fevd(ir.values, n, H)
    if n == 1
        @inbounds for h in 1:H
            if decomp[1, 1, h] == zero(T)
                props[1, 1, h] = one(T)
            end
        end
    end
    return FEVD{T}(decomp, props, copy(m.varnames), copy(m.shock_names))
end

# =============================================================================
# User-Facing Convenience Constructors for SVARMAModel
# =============================================================================

"""
    SVARMAModel(Y, Phi, Theta, C; mu=nothing, shocks=nothing, distributions=nothing,
                dist_params=nothing, loglik=NaN, aic=NaN, bic=NaN,
                varnames=nothing, shock_names=nothing, converged=true,
                method=:manual, tol=1e-8) -> SVARMAModel

Construct an `SVARMAModel{T}` accepting `Phi` and `Theta` as `Array{<:Real,3}`,
`Vector{<:AbstractMatrix}`, single `AbstractMatrix`, or `nothing`. Automatically
computes MA companion eigenvalues, `regime`, `gmr_region`, and (when `shocks === nothing`
and `Theta` has no unit roots with nonsingular `C`) recovers `shocks` via
[`ssvarma_filter_shocks`](@ref).
"""
function SVARMAModel{T}(
    Y::AbstractMatrix,
    Phi,
    Theta,
    C::AbstractMatrix;
    mu::Union{Nothing,AbstractVector}=nothing,
    shocks::Union{Nothing,AbstractMatrix}=nothing,
    distributions::Union{Nothing,AbstractVector{Symbol}}=nothing,
    dist_params::Union{Nothing,AbstractVector}=nothing,
    loglik::Real=T(NaN),
    aic::Real=T(NaN),
    bic::Real=T(NaN),
    varnames::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
    shock_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
    converged::Bool=true,
    method::Symbol=:manual,
    tol::Real=1e-8,
) where {T<:AbstractFloat}
    Y_mat = Matrix{T}(Y)
    T_obs, n = size(Y_mat)
    mu_vec = mu === nothing ? zeros(T, n) : Vector{T}(mu)
    Phi_3d = _svarma_to_3d(Phi, n, T)
    Theta_3d = _svarma_to_3d(Theta, n, T)
    C_mat = Matrix{T}(C)
    size(C_mat) == (n, n) || throw(ArgumentError("C must be $n × $n, got $(size(C_mat))"))

    p = size(Phi_3d, 3)
    q = size(Theta_3d, 3)

    reg = fundamentalness_regime(Theta_3d, C_mat; tol=tol)
    re_vec = T[T(real(z)) for z in reg.eigenvalues]
    im_vec = T[T(imag(z)) for z in reg.eigenvalues]

    shocks_mat = if shocks !== nothing
        Matrix{T}(shocks)
    elseif reg.regime !== :unit_root && T_obs > p && abs(det(C_mat)) > T(10) * eps(T)
        ssvarma_filter_shocks(Y_mat, mu_vec, Phi_3d, Theta_3d, C_mat; tol=tol).shocks
    else
        zeros(T, max(T_obs - p, 0), n)
    end

    dists_vec = distributions === nothing ? fill(:nonparametric, n) : Vector{Symbol}(distributions)
    dp_vec = dist_params === nothing ? Vector{T}[T[] for _ in 1:n] :
             Vector{T}[Vector{T}(d) for d in dist_params]
    vn_vec = varnames === nothing ? default_var_names(n) : String[string(v) for v in varnames]
    sn_vec = shock_names === nothing ? default_shock_names(n) : String[string(s) for s in shock_names]

    return SVARMAModel{T}(
        Y_mat, p, q, mu_vec, Phi_3d, Theta_3d, C_mat, shocks_mat,
        re_vec, im_vec, reg.regime, reg.gmr_region, dists_vec, dp_vec,
        T(loglik), T(aic), T(bic), vn_vec, sn_vec, converged, method,
    )
end

function SVARMAModel(
    Y::AbstractMatrix,
    Phi,
    Theta,
    C::AbstractMatrix;
    mu::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    T = float(promote_type(
        eltype(Y),
        _svarma_eltype(Phi),
        _svarma_eltype(Theta),
        eltype(C),
        mu === nothing ? Bool : eltype(mu),
    ))
    return SVARMAModel{T}(Y, Phi, Theta, C; mu=mu, kwargs...)
end
