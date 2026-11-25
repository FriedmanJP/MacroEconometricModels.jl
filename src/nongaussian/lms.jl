# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Lanne, Meitz & Saikkonen (2017) non-Gaussian SVAR identification and estimation:
- Complete identification scheme Π(B) ∈ B (Proposition 2, p. 292)
- Zero-mean unit-variance shock density families (`:student_t`, `:skewed_t`,
  `:mixture_normal_asym`, `:pearson_iv`/`:pml`, `:skew_normal`, `:ged`/`:laplace_generalized`)
- Analytical score & 16-block Hessian (Appendices B & C, pp. 301-302)
- Adaptive 3-step estimator (§4.5) and full 1-step ML estimator (§4.4)

Reference:
- Lanne, M., Meitz, M. & Saikkonen, P. (2017). "Identification and estimation of
  non-Gaussian structural vector autoregressions." Journal of Econometrics 196(2), 288-304.
"""

using LinearAlgebra, Statistics, Distributions, Random
import Optim
import SpecialFunctions

# =============================================================================
# Result Type & Interface
# =============================================================================

"""
    LMSSVARResult{T} <: AbstractNonGaussianSVAR

Result from Lanne–Meitz–Saikkonen (2017) non-Gaussian SVAR estimation.

Fields:
- `B0::Matrix{T}` — structural impact matrix `B * Diagonal(sigma)` (n × n)
- `B::Matrix{T}` — unit-diagonal normalized matrix `B(β̂) ∈ B` (n × n)
- `sigma::Vector{T}` — positive shock scale parameters `σ̂` (length n)
- `Q::Matrix{T}` — polar orthogonal factor vs `safe_cholesky(var_ols.Sigma)` (n × n)
- `Pi::Matrix{T}` — estimated VAR coefficients `(1 + n*p) × n`
- `var_ols::VARModel{T}` — reduced-form OLS VAR model
- `shocks::Matrix{T}` — standardized unit-variance shocks `z_t` (T_eff × n)
- `raw_shocks::Matrix{T}` — unscaled shocks `ε_t = B⁻¹ u_t` (T_eff × n)
- `distributions::Vector{Symbol}` — canonical shock family symbols (length n)
- `dist_params::Vector{Vector{T}}` — constrained family parameters `λ̂_i` per shock
- `theta::Vector{T}` — stacked parameter vector `[π̂; β̂; σ̂; λ̂]`
- `loglik::T` — total log-likelihood `T_eff * L_T(θ̂)`
- `loglik_gaussian::T` — Gaussian reference log-likelihood
- `vcov::Matrix{T}` — asymptotic covariance matrix of `θ̂`
- `se_B0::Matrix{T}` — delta-method standard errors for `B₀` (n × n)
- `se_B::Matrix{T}` — standard errors for `B` (`0` on fixed unit diagonal)
- `se_sigma::Vector{T}` — standard errors for `σ̂` (length n)
- `se_Pi::Matrix{T}` — standard errors for `Π̂` `(1 + n*p) × n`
- `method::Symbol` — `:three_step` or `:full_ml`
- `se_type::Symbol` — `:hessian`, `:sandwich`, or `:opg`
- `converged::Bool`
- `iterations::Int`
- `aic::T`
- `bic::T`
- `shock_names::Vector{String}`
- `manifest::Union{ReproManifest,Nothing}`
"""
struct LMSSVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    B::Matrix{T}
    sigma::Vector{T}
    Q::Matrix{T}
    Pi::Matrix{T}
    var_ols::VARModel{T}
    shocks::Matrix{T}
    raw_shocks::Matrix{T}
    distributions::Vector{Symbol}
    dist_params::Vector{Vector{T}}
    theta::Vector{T}
    loglik::T
    loglik_gaussian::T
    vcov::Matrix{T}
    se_B0::Matrix{T}
    se_B::Matrix{T}
    se_sigma::Vector{T}
    se_Pi::Matrix{T}
    method::Symbol
    se_type::Symbol
    converged::Bool
    iterations::Int
    aic::T
    bic::T
    shock_names::Vector{String}
    manifest::Union{ReproManifest,Nothing}
end

function Base.getproperty(r::LMSSVARResult, s::Symbol)
    s === :se && return getfield(r, :se_B0)
    if s === :distribution
        ds = getfield(r, :distributions)
        return length(unique(ds)) == 1 ? ds[1] : :heterogeneous
    end
    return getfield(r, s)
end

function Base.propertynames(::LMSSVARResult, private::Bool=false)
    return (fieldnames(LMSSVARResult)..., :se, :distribution)
end

nvars(r::LMSSVARResult) = size(r.B0, 1)
nlags(r::LMSSVARResult) = r.var_ols.p
effective_nobs(r::LMSSVARResult) = size(r.shocks, 1)

StatsAPI.nobs(r::LMSSVARResult) = size(r.shocks, 1)
StatsAPI.coef(r::LMSSVARResult) = r.theta
StatsAPI.vcov(r::LMSSVARResult) = r.vcov
StatsAPI.stderror(r::LMSSVARResult{T}) where {T} = sqrt.(max.(diag(r.vcov), zero(T)))
StatsAPI.residuals(r::LMSSVARResult) = r.shocks
StatsAPI.loglikelihood(r::LMSSVARResult) = r.loglik
StatsAPI.aic(r::LMSSVARResult) = r.aic
StatsAPI.bic(r::LMSSVARResult) = r.bic
StatsAPI.dof(r::LMSSVARResult) = length(r.theta)
StatsAPI.dof_residual(r::LMSSVARResult) = size(r.shocks, 1) * size(r.B0, 1) - length(r.theta)
StatsAPI.islinear(::LMSSVARResult) = false

function Base.show(io::IO, r::LMSSVARResult{T}) where {T}
    n = size(r.B0, 1)
    lr_stat = max(T(2) * (r.loglik - r.loglik_gaussian), zero(T))
    lr_df = max(sum(length, r.dist_params), 1)
    lr_pval = lr_stat > zero(T) ? T(1) - T(cdf(Chisq(lr_df), Float64(lr_stat))) : one(T)
    lr_stars = _significance_stars(lr_pval)
    dist_str = r.distribution === :heterogeneous ?
        join(string.(r.distributions), ", ") : string(r.distribution)

    spec = Any[
        "Method"             string(r.method);
        "Distribution"       dist_str;
        "SE type"            string(r.se_type);
        "Variables"          n;
        "Lags"               r.var_ols.p;
        "Observations (eff)" size(r.shocks, 1);
        "Log-likelihood"     _fmt(r.loglik; digits=4);
        "Log-lik (Gaussian)" _fmt(r.loglik_gaussian; digits=4);
        "AIC"                _fmt(r.aic; digits=2);
        "BIC"                _fmt(r.bic; digits=2);
        "Converged"          r.converged ? "Yes" : "No";
        "Iterations"         r.iterations
    ]
    _pretty_table(io, spec;
        title = "Lanne–Meitz–Saikkonen (2017) Non-Gaussian SVAR",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    lr_data = Any[
        "LR statistic" string(_fmt(lr_stat; digits=2), " ", lr_stars);
        "P-value"      _format_pvalue(lr_pval);
        "Degrees of freedom" lr_df;
        "H₀"           "Gaussian shocks (standard SVAR)"
    ]
    _pretty_table(io, lr_data;
        title = "LR Test: Non-Gaussian vs. Gaussian",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    _show_B0_ses(io, r.B0, r.se_B0)

    dist_rows = Matrix{Any}(undef, n, 4)
    for i in 1:n
        dist_rows[i, 1] = r.shock_names[i]
        dist_rows[i, 2] = string(r.distributions[i])
        dist_rows[i, 3] = _fmt(r.sigma[i])
        dist_rows[i, 4] = join([_fmt(v) for v in r.dist_params[i]], ", ")
    end
    _pretty_table(io, dist_rows;
        title = "Shock Scales & Distribution Parameters",
        column_labels = ["Shock", "Family", "σ", "λ"],
        alignment = [:l, :l, :r, :r],
    )

    _manifest_footer(io, r.manifest)
end

report(r::LMSSVARResult; repro::Bool=false) = show(IOContext(stdout, :repro => repro), r)

# =============================================================================
# Complete Identification Projection Π(B) ∈ B (LMS 2017, §3.3 Prop. 2, p. 292)
# =============================================================================

"""Column-major off-diagonal vectorization `vecd⁰(B)` of length `n(n-1)`."""
function _vecd0(B::AbstractMatrix{T}) where {T}
    n = size(B, 1)
    size(B, 2) == n || throw(ArgumentError("Matrix must be square"))
    v = Vector{T}(undef, n * (n - 1))
    k = 0
    @inbounds for j in 1:n, i in 1:n
        if i != j
            k += 1
            v[k] = B[i, j]
        end
    end
    v
end

"""Reconstruct unit-diagonal `n × n` matrix `B(β)` from `β = vecd⁰(B)`."""
function _lms_B_from_beta(beta::AbstractVector{T}, n::Int) where {T}
    length(beta) == n * (n - 1) || throw(ArgumentError(
        "beta must have length $(n * (n - 1)), got $(length(beta))"))
    B = Matrix{T}(I, n, n)
    k = 0
    @inbounds for j in 1:n, i in 1:n
        if i != j
            k += 1
            B[i, j] = beta[k]
        end
    end
    B
end

"""Selection matrix `H_lms` (`n² × n(n-1)`) satisfying `vec(B) = H_lms * β + vec(Iₙ)`."""
function _lms_H(::Type{T}, n::Int) where {T}
    H = zeros(T, n * n, n * (n - 1))
    k = 0
    @inbounds for j in 1:n, i in 1:n
        if i != j
            k += 1
            H[i + (j - 1) * n, k] = one(T)
        end
    end
    H
end

"""Commutation matrix `K_{n,n}` (`n² × n²`) satisfying `K * vec(A) = vec(A')`."""
function _lms_Knn(::Type{T}, n::Int) where {T}
    K = zeros(T, n * n, n * n)
    @inbounds for j in 1:n, i in 1:n
        K[j + (i - 1) * n, i + (j - 1) * n] = one(T)
    end
    K
end

"""
    lms_project_B(B::AbstractMatrix{<:Real}; tol::Real=1e-10)
        -> (B, beta, sigma, D1, P, D2, perm, signs)

Complete identification projection ``\\Pi(B) \\in \\mathcal{B}`` of Lanne, Meitz &
Saikkonen (2017, *JoE*, §3.3 Proposition 2, p. 292).

Maps a non-singular square matrix `B` to its unique unit-diagonal canonical
representative `B_rep = B * D1 * P * D2 ∈ B` by:
1. Checking non-singularity (`|det(B)| > tol * max(‖B‖₂ⁿ, eps)`).
2. Scaling each column to unit Euclidean norm (`C1 = B * D1`).
3. Greedily selecting row-by-row (`i = 1..n-1`) the unique remaining column with
   strictly largest magnitude in row `i`; throwing `IdentificationError` on zero
   or nonzero ties (`s₁ ≤ tol` or `s₁ - s₂ ≤ tol`) or a zero final diagonal entry.
4. Permuting columns (`C = C1 * P = C1[:, perm]`), normalizing each diagonal to `1`
   (`D2 = Diagonal(1 ./ diag(C))`), and extracting `beta = vecd⁰(B_rep)`, positive
   shock scales `sigma[i] = |C[i, i]| / D1[perm[i], perm[i]]`, and column `signs`.
"""
function lms_project_B(B::AbstractMatrix{<:Real}; tol::Real=1e-10)
    nr, nc = size(B)
    nr == nc || throw(ArgumentError("B must be square (got $(nr) × $(nc))"))
    n = nr
    n >= 1 || throw(ArgumentError("B must be non-empty"))
    T = float(eltype(B))
    tol_T = T(tol)
    tol_T > zero(T) || throw(ArgumentError("tol must be positive (got $tol)"))
    Bf = Matrix{T}(B)
    all(isfinite, Bf) || throw(IdentificationError("lms_project_B: B contains non-finite entries"))

    nrm_op = opnorm(Bf)
    scale_ref = nrm_op^n
    ref = isfinite(scale_ref) ? max(scale_ref, eps(T)) : eps(T)
    det_abs = abs(det(Bf))
    det_abs <= tol_T * ref && throw(IdentificationError(
        "lms_project_B: B is singular or near-singular (|det(B)| = $det_abs)"))

    d1 = Vector{T}(undef, n)
    @inbounds for j in 1:n
        cn = norm(@view(Bf[:, j]))
        cn > tol_T || throw(IdentificationError("lms_project_B: column $j has zero norm"))
        d1[j] = one(T) / cn
    end
    C1 = Bf * Diagonal(d1)

    rem = collect(1:n)
    perm = Vector{Int}(undef, n)
    for i in 1:(n - 1)
        mags = T[abs(C1[i, c]) for c in rem]
        ord = sortperm(mags; rev=true)
        s1 = mags[ord[1]]
        if s1 <= tol_T
            throw(IdentificationError(
                "lms_project_B: zero tie in row $i (max magnitude $s1 <= tol=$tol_T)"))
        end
        if length(rem) >= 2
            s2 = mags[ord[2]]
            if s1 - s2 <= tol_T
                throw(IdentificationError(
                    "lms_project_B: nonzero tie in row $i (s₁=$s1, s₂=$s2, gap=$(s1 - s2) <= tol=$tol_T)"))
            end
        end
        chosen = ord[1]
        perm[i] = rem[chosen]
        deleteat!(rem, chosen)
    end
    perm[n] = rem[1]
    if abs(C1[n, perm[n]]) <= tol_T
        throw(IdentificationError(
            "lms_project_B: zero entry at row $n for remaining column $(perm[n])"))
    end

    P = zeros(T, n, n)
    @inbounds for j in 1:n
        P[perm[j], j] = one(T)
    end
    C = C1 * P
    d2 = Vector{T}(undef, n)
    sigma = Vector{T}(undef, n)
    signs = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        cii = C[i, i]
        d2[i] = one(T) / cii
        sigma[i] = abs(cii) / d1[perm[i]]
        signs[i] = cii >= zero(T) ? 1 : -1
    end
    B_rep = C * Diagonal(d2)
    @inbounds for i in 1:n
        B_rep[i, i] = one(T)
    end
    beta = _vecd0(B_rep)
    D1 = Matrix{T}(Diagonal(d1))
    D2 = Matrix{T}(Diagonal(d2))

    return (B=B_rep, beta=beta, sigma=sigma, D1=D1, P=P, D2=D2, perm=perm, signs=signs)
end

# =============================================================================
# Zero-Mean Unit-Variance Shock Families & Parameter Transformations
# =============================================================================

function _lms_canonical_dist(d::Symbol)
    d === :pml && return :pearson_iv
    d === :laplace_generalized && return :ged
    if d in (:student_t, :skewed_t, :mixture_normal_asym, :pearson_iv, :skew_normal, :ged)
        return d
    end
    throw(ArgumentError(
        "Unsupported LMS distribution :$d. Supported: :student_t, :skewed_t, " *
        ":mixture_normal_asym, :pearson_iv (:pml), :skew_normal, :ged (:laplace_generalized)"))
end

function _lms_normalize_dists(distributions::Symbol, n::Int)
    fill(_lms_canonical_dist(distributions), n)
end

function _lms_normalize_dists(distributions::AbstractVector{Symbol}, n::Int)
    length(distributions) == n || throw(ArgumentError(
        "distributions vector length ($(length(distributions))) must match n ($n)"))
    Symbol[_lms_canonical_dist(d) for d in distributions]
end

@inline function _lms_dist_nparams(d::Symbol)
    d === :student_t ? 1 :
    d === :skewed_t ? 2 :
    d === :mixture_normal_asym ? 3 :
    d === :pearson_iv ? 2 :
    d === :skew_normal ? 1 :
    d === :ged ? 1 :
    throw(ArgumentError("Unsupported LMS distribution :$d"))
end

function _lms_lambda_offsets(dists::Vector{Symbol})
    n = length(dists)
    dims = Int[_lms_dist_nparams(d) for d in dists]
    offs = Vector{Int}(undef, n)
    cur = 1
    @inbounds for i in 1:n
        offs[i] = cur
        cur += dims[i]
    end
    (offs, dims, cur - 1)
end

"""Hansen (1994) zero-mean unit-variance skewed Student-t log-pdf."""
function _skewed_t_logpdf(z::Tz, nu::Tn, eta::Te) where {Tz<:Real, Tn<:Real, Te<:Real}
    T = float(promote_type(Tz, Tn, Te))
    nu_s = max(T(nu), T(2.001))
    et = T(eta)
    logc = SpecialFunctions.loggamma((nu_s + one(T)) / T(2)) -
           SpecialFunctions.loggamma(nu_s / T(2)) -
           T(0.5) * log(T(π) * (nu_s - T(2)))
    c = exp(logc)
    a = T(4) * et * c * ((nu_s - T(2)) / (nu_s - one(T)))
    b2 = max(one(T) + T(3) * et^2 - a^2, T(1e-12))
    b = sqrt(b2)
    y = b * T(z) + a
    denom = y < zero(T) ? (one(T) - et) : (one(T) + et)
    r = y / denom
    T(0.5) * log(b2) + logc - ((nu_s + one(T)) / T(2)) * log1p((r * r) / (nu_s - T(2)))
end

"""Unit-variance Generalized Error Distribution (GED) log-pdf."""
function _ged_logpdf(z::Tz, kappa::Tk) where {Tz<:Real, Tk<:Real}
    T = float(promote_type(Tz, Tk))
    k = max(T(kappa), T(0.1))
    log_lam = T(0.5) * (SpecialFunctions.loggamma(one(T) / k) -
                        SpecialFunctions.loggamma(T(3) / k))
    lam = exp(log_lam)
    log(k) - log(T(2)) - log_lam - SpecialFunctions.loggamma(one(T) / k) -
        (abs(T(z)) / lam)^k
end

"""Zero-mean unit-variance asymmetric 2-component Gaussian mixture log-pdf."""
function _lms_mixture_asym_logpdf(z::Tz, p_mix::Tp, delta::Td,
                                  log_rv::Tr) where {Tz<:Real, Tp<:Real, Td<:Real, Tr<:Real}
    T = float(promote_type(Tz, Tp, Td, Tr))
    p = T(p_mix)
    d = T(delta)
    rv = exp(T(log_rv))
    m1 = d * sqrt((one(T) - p) / p)
    m2 = -d * sqrt(p / (one(T) - p))
    v_rem = one(T) - d * d
    s1_sq = v_rem / (p + (one(T) - p) * rv)
    s2_sq = rv * s1_sq
    zt = T(z)
    log2pi = log(T(2) * T(π))
    l1 = log(p) - T(0.5) * (log2pi + log(s1_sq) + (zt - m1)^2 / s1_sq)
    l2 = log(one(T) - p) - T(0.5) * (log2pi + log(s2_sq) + (zt - m2)^2 / s2_sq)
    mx = max(l1, l2)
    mx + log(exp(l1 - mx) + exp(l2 - mx))
end

"""Zero-mean unit-variance skew-normal log-pdf."""
function _lms_skew_normal_logpdf(z::Tz, alpha::Ta) where {Tz<:Real, Ta<:Real}
    T = float(promote_type(Tz, Ta))
    a = T(alpha)
    mu_z = (a / sqrt(one(T) + a^2)) * sqrt(T(2) / T(π))
    s_z = sqrt(one(T) - mu_z^2)
    _skew_normal_logpdf(mu_z + s_z * T(z), a) + log(s_z)
end

"""Univariate log-density `log f_i(z; λ_i)` for constrained family parameters `lam`."""
@inline function _lms_logpdf_scalar(d::Symbol, z::Tz,
                                    lam::AbstractVector{Tl}) where {Tz<:Real, Tl<:Real}
    if d === :student_t
        return _student_t_logpdf(z, lam[1])
    elseif d === :skewed_t
        return _skewed_t_logpdf(z, lam[1], lam[2])
    elseif d === :mixture_normal_asym
        return _lms_mixture_asym_logpdf(z, lam[1], lam[2], lam[3])
    elseif d === :pearson_iv
        return _pearson_iv_logpdf(z, lam[1], lam[2])
    elseif d === :skew_normal
        return _lms_skew_normal_logpdf(z, lam[1])
    elseif d === :ged
        return _ged_logpdf(z, lam[1])
    end
    throw(ArgumentError("Unsupported LMS distribution :$d"))
end

"""Map unconstrained `u` to constrained `(lam, dlam_du)` for family `d`."""
function _lms_unc_to_con_1d(d::Symbol, u::AbstractVector{T}) where {T<:AbstractFloat}
    if d === :student_t
        uc = clamp(u[1], T(-20), T(10))
        e = exp(uc)
        lam = T[T(2.05) + e]
        jac = T[u[1] == uc ? e : zero(T)]
        return lam, jac
    elseif d === :skewed_t
        uc = clamp(u[1], T(-20), T(10))
        e = exp(uc)
        th = tanh(u[2])
        lam = T[T(2.05) + e, T(0.98) * th]
        jac = T[u[1] == uc ? e : zero(T), T(0.98) * (one(T) - th * th)]
        return lam, jac
    elseif d === :mixture_normal_asym
        s1 = _stable_sigmoid(u[1])
        t2 = tanh(u[2])
        t3 = tanh(u[3])
        lam = T[T(0.05) + T(0.90) * s1, T(0.95) * t2, T(8.0) * t3]
        jac = T[T(0.90) * s1 * (one(T) - s1), T(0.95) * (one(T) - t2 * t2),
                T(8.0) * (one(T) - t3 * t3)]
        return lam, jac
    elseif d === :pearson_iv
        uc = clamp(u[2], T(-20), T(9.2))
        e = exp(uc)
        lam = T[u[1], T(2.05) + e]
        jac = T[one(T), u[2] == uc ? e : zero(T)]
        return lam, jac
    elseif d === :skew_normal
        return T[u[1]], T[one(T)]
    elseif d === :ged
        uc = clamp(u[1], T(-20), T(6))
        e = exp(uc)
        lam = T[T(0.35) + e]
        jac = T[u[1] == uc ? e : zero(T)]
        return lam, jac
    end
    throw(ArgumentError("Unsupported LMS distribution :$d"))
end

"""Map constrained `lam` back to unconstrained `u` for family `d`."""
function _lms_con_to_unc_1d(d::Symbol, lam::AbstractVector{T}) where {T<:AbstractFloat}
    if d === :student_t
        return T[log(max(lam[1] - T(2.05), T(1e-8)))]
    elseif d === :skewed_t
        return T[log(max(lam[1] - T(2.05), T(1e-8))),
                 atanh(clamp(lam[2] / T(0.98), T(-0.999999), T(0.999999)))]
    elseif d === :mixture_normal_asym
        pr = clamp((lam[1] - T(0.05)) / T(0.90), T(1e-6), one(T) - T(1e-6))
        return T[log(pr / (one(T) - pr)),
                 atanh(clamp(lam[2] / T(0.95), T(-0.999999), T(0.999999))),
                 atanh(clamp(lam[3] / T(8.0), T(-0.999999), T(0.999999)))]
    elseif d === :pearson_iv
        return T[lam[1], log(max(lam[2] - T(2.05), T(1e-8)))]
    elseif d === :skew_normal
        return T[lam[1]]
    elseif d === :ged
        return T[log(max(lam[1] - T(0.35), T(1e-8)))]
    end
    throw(ArgumentError("Unsupported LMS distribution :$d"))
end

function _lms_unc_to_con(dists::Vector{Symbol}, u::AbstractVector{T}) where {T<:AbstractFloat}
    offs, dims, n_lam = _lms_lambda_offsets(dists)
    lam_vec = Vector{T}(undef, n_lam)
    jac_vec = Vector{T}(undef, n_lam)
    @inbounds for i in eachindex(dists)
        r = offs[i]:(offs[i] + dims[i] - 1)
        li, ji = _lms_unc_to_con_1d(dists[i], @view(u[r]))
        lam_vec[r] .= li
        jac_vec[r] .= ji
    end
    lam_vec, jac_vec
end

function _lms_con_to_unc(dists::Vector{Symbol}, lam_list::Vector{Vector{T}}) where {T<:AbstractFloat}
    parts = [_lms_con_to_unc_1d(dists[i], lam_list[i]) for i in eachindex(dists)]
    vcat(parts...)
end

function _lms_split_lam(dists::Vector{Symbol}, lam_vec::AbstractVector{T}) where {T<:AbstractFloat}
    offs, dims, _ = _lms_lambda_offsets(dists)
    [Vector{T}(@view(lam_vec[offs[i]:(offs[i] + dims[i] - 1)])) for i in eachindex(dists)]
end

function _lms_flip_dist_param(d::Symbol, lam::Vector{T}, sgn::Int) where {T<:AbstractFloat}
    out = copy(lam)
    sgn >= 0 && return out
    if d === :skew_normal || d === :pearson_iv
        out[1] = -out[1]
    elseif d === :skewed_t || d === :mixture_normal_asym
        out[2] = -out[2]
    end
    out
end

function _lms_default_lam_unc_1d(d::Symbol, s::Int, ::Type{T}) where {T<:AbstractFloat}
    if d === :student_t
        return T[isodd(s) ? log(T(5.0) - T(2.05)) : log(T(3.5) - T(2.05))]
    elseif d === :skewed_t
        return T[isodd(s) ? log(T(5.0) - T(2.05)) : log(T(3.5) - T(2.05)),
                 isodd(s) ? T(0.25) : T(-0.25)]
    elseif d === :mixture_normal_asym
        if s == 1
            return T[-1.0, 0.5, 0.6]
        elseif s == 2
            return T[-1.0, -0.5, 0.6]
        elseif s == 3
            return T[1.0, 0.4, -0.6]
        elseif s == 4
            return T[-0.8, -0.4, -0.6]
        else
            return isodd(s) ? T[0.3, 0.5, 0.8] : T[-0.3, -0.5, -0.8]
        end
    elseif d === :pearson_iv
        return T[isodd(s) ? T(0.3) : T(-0.3), log(T(5.0) - T(2.05))]
    elseif d === :skew_normal
        return T[isodd(s) ? T(0.5) : T(-0.5)]
    elseif d === :ged
        return T[isodd(s) ? log(T(1.5) - T(0.35)) : log(T(1.1) - T(0.35))]
    end
    throw(ArgumentError("Unsupported LMS distribution :$d"))
end

function _lms_default_lam_unc(dists::Vector{Symbol}, s::Int, ::Type{T}) where {T<:AbstractFloat}
    vcat([_lms_default_lam_unc_1d(d, s, T) for d in dists]...)
end

# =============================================================================
# Standardized Log-Likelihood, Analytical Score & 16-Block Hessian
# =============================================================================

"""
Average standardized log-likelihood ``L_T(\\theta) = T_{\\text{eff}}^{-1} \\sum_t l_t(\\theta)``
for `θ = [π; β; σ; λ]` with constrained `(σ, λ)`. Real-generic for `ForwardDiff`.
"""
function _lms_avg_loglik(theta::AbstractVector{Tθ}, Y_eff::AbstractMatrix{T},
                         X::AbstractMatrix{T}, dists::Vector{Symbol}) where {Tθ<:Real, T<:AbstractFloat}
    T_eff, n = size(Y_eff)
    k = size(X, 2)
    n_pi = n * k
    n_beta = n * (n - 1)
    offs, dims, n_lam = _lms_lambda_offsets(dists)
    length(theta) == n_pi + n_beta + n + n_lam || throw(ArgumentError(
        "theta length mismatch: got $(length(theta)), expected $(n_pi + n_beta + n + n_lam)"))

    pi_vec = @view theta[1:n_pi]
    beta_vec = @view theta[(n_pi + 1):(n_pi + n_beta)]
    sigma_vec = @view theta[(n_pi + n_beta + 1):(n_pi + n_beta + n)]
    lam_vec = @view theta[(n_pi + n_beta + n + 1):end]

    any(x -> x <= zero(Tθ), sigma_vec) && return Tθ(-Inf)
    B = _lms_B_from_beta(beta_vec, n)
    detB = det(B)
    (!isfinite(detB) || abs(detB) <= Tθ(1e-15)) && return Tθ(-Inf)

    B_inv = robust_inv(B; silent=true)
    Pi_mat = transpose(reshape(pi_vec, n, k))
    U = Y_eff .- X * Pi_mat
    Z = (U * transpose(B_inv)) * Diagonal(one(Tθ) ./ sigma_vec)

    ll = -Tθ(T_eff) * (log(abs(detB)) + sum(log, sigma_vec))
    @inbounds for i in 1:n
        li = @view lam_vec[offs[i]:(offs[i] + dims[i] - 1)]
        di = dists[i]
        for t in 1:T_eff
            ll += _lms_logpdf_scalar(di, Z[t, i], li)
        end
    end
    ll / Tθ(T_eff)
end

"""Compute average log-likelihood and analytical score `[l_π; l_β; l_σ; l_λ]` for constrained `θ`."""
function _lms_loglik_and_score(theta::AbstractVector{T}, Y_eff::AbstractMatrix{T},
                               X::AbstractMatrix{T}, dists::Vector{Symbol};
                               U_fixed::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:AbstractFloat}
    T_eff, n = size(Y_eff)
    k = size(X, 2)
    n_pi = n * k
    n_beta = n * (n - 1)
    offs, dims, n_lam = _lms_lambda_offsets(dists)
    n_total = n_pi + n_beta + n + n_lam

    pi_vec = @view theta[1:n_pi]
    beta_vec = @view theta[(n_pi + 1):(n_pi + n_beta)]
    sigma = @view theta[(n_pi + n_beta + 1):(n_pi + n_beta + n)]
    lam_vec = @view theta[(n_pi + n_beta + n + 1):end]

    if any(x -> !isfinite(x) || x <= zero(T), sigma)
        return T(-Inf), zeros(T, n_total), false
    end
    B = _lms_B_from_beta(beta_vec, n)
    detB = det(B)
    if !isfinite(detB) || abs(detB) <= T(1e-13) * max(opnorm(B)^n, eps(T))
        return T(-Inf), zeros(T, n_total), false
    end

    B_inv = Matrix{T}(robust_inv(B; silent=true))
    B_inv_T = Matrix{T}(transpose(B_inv))
    U = if U_fixed !== nothing
        U_fixed
    else
        Pi_mat = Matrix{T}(transpose(reshape(pi_vec, n, k)))
        Y_eff - X * Pi_mat
    end
    Eps = U * B_inv_T
    S_inv = one(T) ./ sigma
    Z = Eps * Diagonal(S_inv)

    Ex = Matrix{T}(undef, T_eff, n)
    lam_score = zeros(T, n_lam)
    ll_dens = zero(T)

    for i in 1:n
        di = dists[i]
        dim_i = dims[i]
        rng_i = offs[i]:(offs[i] + dim_i - 1)
        lam_i = Vector{T}(@view(lam_vec[rng_i]))
        phi_i = w -> _lms_logpdf_scalar(di, w[1], @view(w[2:(1 + dim_i)]))
        w = Vector{T}(undef, 1 + dim_i)
        copyto!(w, 2, lam_i, 1, dim_i)
        gr = ForwardDiff.DiffResults.GradientResult(w)
        cfg = ForwardDiff.GradientConfig(phi_i, w)
        @inbounds for t in 1:T_eff
            w[1] = Z[t, i]
            ForwardDiff.gradient!(gr, phi_i, w, cfg)
            val = ForwardDiff.DiffResults.value(gr)
            g = ForwardDiff.DiffResults.gradient(gr)
            ll_dens += val
            Ex[t, i] = g[1]
            for r in 1:dim_i
                lam_score[offs[i] + r - 1] += g[1 + r]
            end
        end
    end

    if !isfinite(ll_dens) || any(!isfinite, Ex) || any(!isfinite, lam_score)
        return T(-Inf), zeros(T, n_total), false
    end

    invT = one(T) / T(T_eff)
    L_T = -(log(abs(detB)) + sum(log, sigma)) + ll_dens * invT

    # Matrix-form average score over T_eff observations:
    # l_π = -vec(B⁻¹' S E_x' X / T_eff)
    BinvT_S = B_inv_T * Diagonal(S_inv)
    score_pi = -vec((BinvT_S * (Ex' * X)) .* invT)

    # l_β = -H_lms' * vec(B⁻¹' + B⁻¹' S (E_x' Eps) / T_eff)
    H_lms = _lms_H(T, n)
    S_Ex_Eps = (Diagonal(S_inv) * (Ex' * Eps)) .* invT
    score_beta = n_beta == 0 ? T[] : -vec(H_lms' * vec(B_inv_T * (Matrix{T}(I, n, n) + S_Ex_Eps)))

    # l_σ,i = -(1 + T_eff⁻¹ Σ_t z_{i,t} e_{x,i,t}) / σ_i
    z_ex_mean = vec(sum(Z .* Ex; dims=1)) .* invT
    score_sigma = -(one(T) .+ z_ex_mean) .* S_inv

    score_lam = lam_score .* invT
    score = vcat(score_pi, score_beta, score_sigma, score_lam)
    return L_T, score, all(isfinite, score) && isfinite(L_T)
end

"""
    _lms_score_hessian(theta, Y_eff, X, dists)
        -> (loglik_avg, score, hessian, opg)

Compute standardized log-likelihood ``L_T(\\theta)``, analytical score ``\\nabla L_T(\\theta)``,
exact 16-block analytical Hessian ``\\nabla^2 L_T(\\theta)`` (Lanne, Meitz & Saikkonen 2017,
Appendices B & C, pp. 301-302), and outer product of gradients ``J_T = T_{\\text{eff}}^{-1} \\sum_t S_t S_t'``
for parameter vector `theta = [π; β; σ; λ]` (`π = vec(Matrix(Pi'))`).
"""
function _lms_score_hessian(theta::AbstractVector{T}, Y_eff::AbstractMatrix{T},
                            X::AbstractMatrix{T}, dists::Vector{Symbol}) where {T<:AbstractFloat}
    T_eff, n = size(Y_eff)
    k = size(X, 2)
    n_pi = n * k
    n_beta = n * (n - 1)
    offs, dims, n_lam = _lms_lambda_offsets(dists)
    n_total = n_pi + n_beta + n + n_lam

    length(theta) == n_total || throw(ArgumentError(
        "theta length mismatch: got $(length(theta)), expected $n_total"))

    o_beta = n_pi
    o_sigma = n_pi + n_beta
    o_lam = n_pi + n_beta + n

    pi_vec = @view theta[1:n_pi]
    beta_vec = @view theta[(o_beta + 1):o_sigma]
    sigma = Vector{T}(@view(theta[(o_sigma + 1):o_lam]))
    lam_vec = @view theta[(o_lam + 1):end]

    any(x -> x <= zero(T), sigma) && throw(ArgumentError("sigma must be strictly positive"))
    B = _lms_B_from_beta(beta_vec, n)
    detB = det(B)
    abs(detB) > eps(T) || throw(IdentificationError("_lms_score_hessian: B is singular"))

    B_inv = Matrix{T}(robust_inv(B; silent=true))
    B_inv_T = Matrix{T}(transpose(B_inv))
    S_inv = one(T) ./ sigma
    S_diag = Diagonal(S_inv)
    S_sq = S_inv .^ 2
    M = S_diag * B_inv
    Mt = Matrix{T}(transpose(M))

    H_lms = _lms_H(T, n)
    Ht_lms = Matrix{T}(transpose(H_lms))
    K_nn = _lms_Knn(T, n)
    KH_lms = K_nn * H_lms

    Pi_mat = Matrix{T}(transpose(reshape(pi_vec, n, k)))
    U = Y_eff - X * Pi_mat
    Eps = U * B_inv_T
    Z = Eps * S_diag

    # Evaluate univariate derivatives via ForwardDiff.hessian! for each (t, i)
    Ex = Matrix{T}(undef, T_eff, n)
    Exx = Matrix{T}(undef, T_eff, n)
    Elam = Matrix{T}(undef, T_eff, n_lam)
    Exlam = Matrix{T}(undef, T_eff, n_lam)
    Elamlam = Vector{Array{T,3}}(undef, n)

    ll_dens = zero(T)
    for i in 1:n
        di = dists[i]
        dim_i = dims[i]
        rng_i = offs[i]:(offs[i] + dim_i - 1)
        lam_i = Vector{T}(@view(lam_vec[rng_i]))
        phi_i = w -> _lms_logpdf_scalar(di, w[1], @view(w[2:(1 + dim_i)]))
        w = Vector{T}(undef, 1 + dim_i)
        copyto!(w, 2, lam_i, 1, dim_i)
        hr = ForwardDiff.DiffResults.HessianResult(w)
        cfg = ForwardDiff.HessianConfig(phi_i, hr, w)
        ell_i = Array{T,3}(undef, T_eff, dim_i, dim_i)
        @inbounds for t in 1:T_eff
            w[1] = Z[t, i]
            ForwardDiff.hessian!(hr, phi_i, w, cfg)
            ll_dens += ForwardDiff.DiffResults.value(hr)
            g = ForwardDiff.DiffResults.gradient(hr)
            h = ForwardDiff.DiffResults.hessian(hr)
            Ex[t, i] = g[1]
            Exx[t, i] = h[1, 1]
            for r in 1:dim_i
                col = offs[i] + r - 1
                Elam[t, col] = g[1 + r]
                Exlam[t, col] = h[1, 1 + r]
                for c in 1:dim_i
                    ell_i[t, r, c] = h[1 + r, 1 + c]
                end
            end
        end
        Elamlam[i] = ell_i
    end

    score_acc = zeros(T, n_total)
    H_acc = zeros(T, n_total, n_total)
    opg_acc = zeros(T, n_total, n_total)
    S_t = Vector{T}(undef, n_total)

    # Constant Hessian piece from -log|det(B)| in l_ββ
    H_bb_const = n_beta > 0 ? Ht_lms * kron(B_inv, B_inv_T) * KH_lms : zeros(T, 0, 0)
    logdet_const = -(log(abs(detB)) + sum(log, sigma))

    r_pi = 1:n_pi
    r_beta = (o_beta + 1):o_sigma
    r_sigma = (o_sigma + 1):o_lam
    r_lam = (o_lam + 1):n_total

    @inbounds for t in 1:T_eff
        x = Vector{T}(@view(X[t, :]))
        eps_t = Vector{T}(@view(Eps[t, :]))
        z_t = Vector{T}(@view(Z[t, :]))
        ex_t = Vector{T}(@view(Ex[t, :]))
        exx_t = Vector{T}(@view(Exx[t, :]))
        v_t = Mt * ex_t

        # Score S_t
        S_t[r_pi] .= -vec(v_t * x')
        if n_beta > 0
            S_t[r_beta] .= -vec(Ht_lms * vec(B_inv_T + v_t * eps_t'))
        end
        for i in 1:n
            S_t[o_sigma + i] = -(one(T) + z_t[i] * ex_t[i]) * S_inv[i]
        end
        for r in 1:n_lam
            S_t[o_lam + r] = Elam[t, r]
        end

        score_acc .+= S_t
        opg_acc .+= S_t * S_t'

        # Hessian blocks at observation t
        W_xx = Mt * Diagonal(exx_t) * M
        xxT = x * x'
        H_acc[r_pi, r_pi] .+= kron(xxT, W_xx)

        d_sig = ex_t .+ exx_t .* z_t
        W_sig = B_inv_T * Diagonal(S_sq .* d_sig)
        h_ps = kron(x, W_sig)
        H_acc[r_pi, r_sigma] .+= h_ps

        if n_beta > 0
            h_pb = kron(x * v_t', B_inv_T) * KH_lms + kron(x * eps_t', W_xx) * H_lms
            H_acc[r_pi, r_beta] .+= h_pb

            h_bb = H_bb_const +
                   Ht_lms * (kron(eps_t * v_t', B_inv_T) * KH_lms +
                             kron(B_inv, v_t * eps_t') * KH_lms +
                             kron(eps_t * eps_t', W_xx) * H_lms)
            H_acc[r_beta, r_beta] .+= h_bb

            h_bs = Ht_lms * kron(eps_t, W_sig)
            H_acc[r_beta, r_sigma] .+= h_bs
        end

        for i in 1:n
            H_acc[o_sigma + i, o_sigma + i] +=
                S_sq[i] * (one(T) + T(2) * z_t[i] * ex_t[i] + z_t[i]^2 * exx_t[i])
            dim_i = dims[i]
            rng_i = offs[i]:(offs[i] + dim_i - 1)
            glam_i = o_lam .+ rng_i
            exlam_i = Vector{T}(@view(Exlam[t, rng_i]))

            H_acc[r_pi, glam_i] .-= kron(x, @view(Mt[:, i])) * exlam_i'
            if n_beta > 0
                H_acc[r_beta, glam_i] .-= Ht_lms * kron(eps_t, @view(Mt[:, i])) * exlam_i'
            end
            H_acc[o_sigma + i, glam_i] .-= (S_inv[i] * z_t[i]) .* exlam_i

            ell_i = Elamlam[i]
            for c in 1:dim_i, r in 1:dim_i
                H_acc[o_lam + offs[i] + r - 1, o_lam + offs[i] + c - 1] += ell_i[t, r, c]
            end
        end
    end

    # Fill symmetric lower blocks
    H_acc[r_beta, r_pi] .= transpose(@view(H_acc[r_pi, r_beta]))
    H_acc[r_sigma, r_pi] .= transpose(@view(H_acc[r_pi, r_sigma]))
    H_acc[r_lam, r_pi] .= transpose(@view(H_acc[r_pi, r_lam]))
    H_acc[r_sigma, r_beta] .= transpose(@view(H_acc[r_beta, r_sigma]))
    H_acc[r_lam, r_beta] .= transpose(@view(H_acc[r_beta, r_lam]))
    H_acc[r_lam, r_sigma] .= transpose(@view(H_acc[r_sigma, r_lam]))

    invT = one(T) / T(T_eff)
    L_T = logdet_const + ll_dens * invT
    score = score_acc .* invT
    H_T = Matrix{T}((H_acc .+ H_acc') .* (T(0.5) * invT))
    J_T = Matrix{T}((opg_acc .+ opg_acc') .* (T(0.5) * invT))

    return (loglik_avg=L_T, score=score, hessian=H_T, opg=J_T)
end

# =============================================================================
# Optimization Helpers (Step 2 Multi-Start, Step 3 Adaptive π, Full ML)
# =============================================================================

"""Generate `n_starts` initial `B0` candidate matrices (JADE, FastICA, symmetric, Cholesky, random)."""
function _lms_initial_B0s(model::VARModel{T}, n_starts::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    n = nvars(model)
    L = Matrix{T}(safe_cholesky(model.Sigma))
    starts = Vector{Matrix{T}}(undef, n_starts)

    if n == 1
        for s in 1:n_starts
            starts[s] = copy(L)
        end
        return starts
    end

    # Candidate 1: JADE
    starts[1] = try
        Matrix{T}(identify_jade(model).B0)
    catch
        copy(L)
    end
    # Candidate 2: FastICA
    if n_starts >= 2
        starts[2] = try
            Matrix{T}(identify_fastica(model; rng=rng).B0)
        catch
            L * generate_Q(n, T; rng=rng)
        end
    end
    # Candidate 3: Symmetric square root Σ^{1/2}
    if n_starts >= 3
        starts[3] = try
            E = eigen(Symmetric(Matrix{T}(model.Sigma)))
            E.vectors * Diagonal(sqrt.(max.(E.values, eps(T)))) * E.vectors'
        catch
            copy(L)
        end
    end
    # Candidate 4: Cholesky factor
    if n_starts >= 4
        starts[4] = copy(L)
    end
    # Candidates 5..n_starts: seeded random orthogonal rotations
    for s in 5:n_starts
        starts[s] = L * generate_Q(n, T; rng=rng)
    end
    starts
end

"""Project initial `B0` to `(beta, log_sigma)`, falling back to Cholesky if tied."""
function _lms_safe_project_init(B0::Matrix{T}, L::Matrix{T}, rng::AbstractRNG) where {T<:AbstractFloat}
    try
        proj = lms_project_B(B0)
        return proj.beta, log.(proj.sigma)
    catch e
        e isa IdentificationError || rethrow(e)
        try
            proj = lms_project_B(L * generate_Q(size(L, 1), T; rng=rng))
            return proj.beta, log.(proj.sigma)
        catch
            proj = lms_project_B(L)
            return proj.beta, log.(proj.sigma)
        end
    end
end

"""Shared cached L-BFGS driver compatible with Optim v1 and v2."""
function _lms_run_lbfgs(eval_fg, x0::Vector{T}, max_iter::Int, g_tol::T) where {T<:AbstractFloat}
    n_x = length(x0)
    last_x = fill(T(NaN), n_x)
    last_f = Ref(zero(T))
    last_g = Vector{T}(undef, n_x)

    function ensure_eval!(x)
        if x != last_x
            copyto!(last_x, x)
            val, ok = eval_fg(last_g, x)
            if ok && isfinite(val)
                last_f[] = val
            else
                last_f[] = T(Inf)
                fill!(last_g, zero(T))
            end
        end
    end

    f(x) = (ensure_eval!(x); last_f[])
    g!(G, x) = (ensure_eval!(x); copyto!(G, last_g))

    opts = Optim.Options(iterations=max_iter, g_tol=g_tol, f_reltol=zero(T),
                         allow_f_increases=true, show_trace=false)
    Optim.optimize(f, g!, x0, Optim.LBFGS(), opts)
end

"""Optimize `(β, log_σ, λ_unc)` holding `π` and `U` fixed."""
function _lms_optimize_gamma(pi_vec::Vector{T}, beta0::Vector{T}, log_sig0::Vector{T},
                             lam_unc0::Vector{T}, Y_eff::Matrix{T}, X::Matrix{T},
                             U_fixed::Matrix{T}, dists::Vector{Symbol};
                             max_iter::Int=500, g_tol::T=T(1e-8)) where {T<:AbstractFloat}
    n = size(Y_eff, 2)
    n_beta = n * (n - 1)
    n_lam = length(lam_unc0)
    n_gamma = n_beta + n + n_lam
    n_pi = length(pi_vec)
    theta_buf = Vector{T}(undef, n_pi + n_gamma)
    copyto!(theta_buf, 1, pi_vec, 1, n_pi)

    x0 = vcat(beta0, log_sig0, lam_unc0)
    idx_b = 1:n_beta
    idx_s = (n_beta + 1):(n_beta + n)
    idx_l = (n_beta + n + 1):n_gamma

    function eval_fg(G, x)
        if n_beta > 0
            copyto!(theta_buf, n_pi + 1, x, 1, n_beta)
        end
        @inbounds for i in 1:n
            theta_buf[n_pi + n_beta + i] = exp(clamp(x[n_beta + i], T(-20), T(20)))
        end
        lam_con, lam_jac = _lms_unc_to_con(dists, @view(x[idx_l]))
        copyto!(theta_buf, n_pi + n_beta + n + 1, lam_con, 1, n_lam)

        L_T, sc, ok = _lms_loglik_and_score(theta_buf, Y_eff, X, dists; U_fixed=U_fixed)
        ok || return (T(Inf), false)
        if n_beta > 0
            @inbounds for j in 1:n_beta
                G[j] = -sc[n_pi + j]
            end
        end
        @inbounds for i in 1:n
            ls = x[n_beta + i]
            G[n_beta + i] = (-20 <= ls <= 20) ? -sc[n_pi + n_beta + i] * theta_buf[n_pi + n_beta + i] : zero(T)
        end
        @inbounds for j in 1:n_lam
            G[n_beta + n + j] = -sc[n_pi + n_beta + n + j] * lam_jac[j]
        end
        return (-L_T, true)
    end

    res = _lms_run_lbfgs(eval_fg, x0, max_iter, g_tol)
    x_opt = Vector{T}(Optim.minimizer(res))
    beta_opt = n_beta > 0 ? x_opt[idx_b] : T[]
    sig_opt = exp.(clamp.(x_opt[idx_s], T(-20), T(20)))
    lam_opt, _ = _lms_unc_to_con(dists, @view(x_opt[idx_l]))
    return (nll=T(Optim.minimum(res)), beta=beta_opt, sigma=sig_opt,
            lam_con=lam_opt, converged=Optim.converged(res), iters=Optim.iterations(res))
end

"""Adaptive Step 3: optimize `π` holding `(β, σ, λ)` fixed."""
function _lms_optimize_pi(pi0::Vector{T}, beta::Vector{T}, sigma::Vector{T},
                          lam_con::Vector{T}, Y_eff::Matrix{T}, X::Matrix{T},
                          dists::Vector{Symbol};
                          max_iter::Int=500, g_tol::T=T(1e-8)) where {T<:AbstractFloat}
    n_pi = length(pi0)
    n_beta = length(beta)
    n = length(sigma)
    n_lam = length(lam_con)
    theta_buf = Vector{T}(undef, n_pi + n_beta + n + n_lam)
    n_beta > 0 && copyto!(theta_buf, n_pi + 1, beta, 1, n_beta)
    copyto!(theta_buf, n_pi + n_beta + 1, sigma, 1, n)
    copyto!(theta_buf, n_pi + n_beta + n + 1, lam_con, 1, n_lam)

    function eval_fg(G, pi_curr)
        copyto!(theta_buf, 1, pi_curr, 1, n_pi)
        L_T, sc, ok = _lms_loglik_and_score(theta_buf, Y_eff, X, dists)
        ok || return (T(Inf), false)
        @inbounds for j in 1:n_pi
            G[j] = -sc[j]
        end
        return (-L_T, true)
    end

    res = _lms_run_lbfgs(eval_fg, copy(pi0), max_iter, g_tol)
    return (nll=T(Optim.minimum(res)), pi=Vector{T}(Optim.minimizer(res)),
            converged=Optim.converged(res), iters=Optim.iterations(res))
end

"""Full 1-step ML: jointly optimize `(π, β, log_σ, λ_unc)`."""
function _lms_optimize_joint(pi0::Vector{T}, beta0::Vector{T}, log_sig0::Vector{T},
                             lam_unc0::Vector{T}, Y_eff::Matrix{T}, X::Matrix{T},
                             dists::Vector{Symbol};
                             max_iter::Int=500, g_tol::T=T(1e-8)) where {T<:AbstractFloat}
    n = size(Y_eff, 2)
    n_pi = length(pi0)
    n_beta = length(beta0)
    n_lam = length(lam_unc0)
    n_total = n_pi + n_beta + n + n_lam
    theta_buf = Vector{T}(undef, n_total)
    x0 = vcat(pi0, beta0, log_sig0, lam_unc0)

    o_beta = n_pi
    o_sig = n_pi + n_beta
    o_lam = n_pi + n_beta + n

    function eval_fg(G, x)
        copyto!(theta_buf, 1, x, 1, n_pi)
        n_beta > 0 && copyto!(theta_buf, o_beta + 1, x, o_beta + 1, n_beta)
        @inbounds for i in 1:n
            theta_buf[o_sig + i] = exp(clamp(x[o_sig + i], T(-20), T(20)))
        end
        lam_con, lam_jac = _lms_unc_to_con(dists, @view(x[(o_lam + 1):n_total]))
        copyto!(theta_buf, o_lam + 1, lam_con, 1, n_lam)

        L_T, sc, ok = _lms_loglik_and_score(theta_buf, Y_eff, X, dists)
        ok || return (T(Inf), false)
        @inbounds for j in 1:(n_pi + n_beta)
            G[j] = -sc[j]
        end
        @inbounds for i in 1:n
            ls = x[o_sig + i]
            G[o_sig + i] = (-20 <= ls <= 20) ? -sc[o_sig + i] * theta_buf[o_sig + i] : zero(T)
        end
        @inbounds for j in 1:n_lam
            G[o_lam + j] = -sc[o_lam + j] * lam_jac[j]
        end
        return (-L_T, true)
    end

    res = _lms_run_lbfgs(eval_fg, x0, max_iter, g_tol)
    x_opt = Vector{T}(Optim.minimizer(res))
    pi_opt = x_opt[1:n_pi]
    beta_opt = n_beta > 0 ? x_opt[(o_beta + 1):o_sig] : T[]
    sig_opt = exp.(clamp.(x_opt[(o_sig + 1):o_lam], T(-20), T(20)))
    lam_opt, _ = _lms_unc_to_con(dists, @view(x_opt[(o_lam + 1):n_total]))
    return (nll=T(Optim.minimum(res)), pi=pi_opt, beta=beta_opt, sigma=sig_opt,
            lam_con=lam_opt, converged=Optim.converged(res), iters=Optim.iterations(res))
end

"""Project `(beta, sigma, lam_con, dists)` onto `B` via `lms_project_B`, reordering consistently."""
function _lms_realign_to_B(beta::Vector{T}, sigma::Vector{T}, lam_con::Vector{T},
                           dists::Vector{Symbol}) where {T<:AbstractFloat}
    n = length(sigma)
    B0_curr = _lms_B_from_beta(beta, n) * Diagonal(sigma)
    proj = lms_project_B(B0_curr)
    perm = proj.perm
    signs = proj.signs
    dists_new = dists[perm]
    lam_split = _lms_split_lam(dists, lam_con)
    lam_new = [_lms_flip_dist_param(dists_new[i], lam_split[perm[i]], signs[i]) for i in 1:n]
    (beta=proj.beta, sigma=proj.sigma, lam_con=vcat(lam_new...), lam_list=lam_new,
     dists=dists_new, changed=(perm != collect(1:n) || any(!=(1), signs)))
end

# =============================================================================
# Public Estimation API
# =============================================================================

"""
    estimate_lms_svar(model::VARModel; distributions=:student_t, method=:three_step,
                      se=:hessian, n_starts=5, max_iter=500, tol=nothing,
                      shock_names=nothing, rng=Random.default_rng(), seed=nothing) -> LMSSVARResult
    estimate_lms_svar(Y::AbstractMatrix, p::Int; ...) -> LMSSVARResult

Estimate a non-Gaussian SVAR using the Lanne–Meitz–Saikkonen (2017) complete
identification scheme `Π(B) ∈ B`, analytical score/16-block Hessian, and either:
- `method=:three_step` (LMS §4.5): OLS step 1, multi-start `(β, σ, λ)` ML + projection
  polish step 2, and adaptive VAR slope `π` ML step 3; or
- `method=:full_ml` (LMS §4.4): full joint ML warm-started from `:three_step`.

# Supported `distributions` (`Symbol` or `Vector{Symbol}` of length `n`)
- `:student_t` — standardized Student-t (`ν > 2.05`)
- `:skewed_t` — Hansen (1994) zero-mean unit-variance skewed Student-t (`ν > 2.05, η ∈ (-0.98, 0.98)`)
- `:mixture_normal_asym` — zero-mean unit-variance asymmetric 2-component Gaussian mixture (`p, δ, log_rv`)
- `:pearson_iv` (alias `:pml`) — zero-mean unit-variance Pearson Type IV (`skew, m > 2.05`)
- `:skew_normal` — zero-mean unit-variance skew-normal (`α`)
- `:ged` (alias `:laplace_generalized`) — exact unit-variance Generalized Error Distribution (`κ > 0.35`)

# Standard Errors (`se`)
- `:hessian` — `(-H_T)⁻¹ / T_eff`
- `:sandwich` — `(-H_T)⁻¹ J_T (-H_T)⁻¹ / T_eff`
- `:opg` — `J_T⁻¹ / T_eff`
"""
function estimate_lms_svar(model::VARModel{T};
                           distributions::Union{Symbol,AbstractVector{Symbol}}=:student_t,
                           method::Symbol=:three_step,
                           se::Symbol=:hessian,
                           n_starts::Int=5,
                           max_iter::Int=500,
                           tol::Union{Nothing,Real}=nothing,
                           shock_names::Union{Nothing,Vector{String}}=nothing,
                           rng::AbstractRNG=Random.default_rng(),
                           seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    n = nvars(model)
    dists = _lms_normalize_dists(distributions, n)
    method in (:three_step, :full_ml) || throw(ArgumentError(
        "method must be :three_step or :full_ml, got :$method"))
    se in (:hessian, :sandwich, :opg) || throw(ArgumentError(
        "se must be :hessian, :sandwich, or :opg, got :$se"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be ≥ 1 (got $n_starts)"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1 (got $max_iter)"))

    snames = if shock_names !== nothing
        length(shock_names) == n || throw(ArgumentError(
            "shock_names length ($(length(shock_names))) must match n ($n)"))
        Vector{String}(shock_names)
    else
        _default_shock_names(n)
    end

    rng = _resolve_repro_rng(rng, seed)
    tol_T = tol === nothing ? (T === Float32 ? T(1e-5) : T(1e-8)) : T(tol)
    polish_tol = min(tol_T, T === Float32 ? T(1e-6) : T(1e-10))

    Y_eff, X = construct_var_matrices(model.Y, model.p)
    T_eff, k = size(X)
    n_pi = n * k
    n_beta = n * (n - 1)

    # Step 1: OLS π̃_LS
    pi_ls = vec(Matrix{T}(transpose(model.B)))
    U_ls = Matrix{T}(model.U)
    L_chol = Matrix{T}(safe_cholesky(model.Sigma))

    # Step 2: Multi-start LBFGS on (β, log_σ, λ_unc) holding π = π̃_LS fixed
    cand_B0s = _lms_initial_B0s(model, n_starts, rng)
    best_nll = T(Inf)
    best_step2 = nothing
    for s in 1:n_starts
        beta0, log_sig0 = _lms_safe_project_init(cand_B0s[s], L_chol, rng)
        lam_unc0 = _lms_default_lam_unc(dists, s, T)
        res_s = try
            _lms_optimize_gamma(pi_ls, beta0, log_sig0, lam_unc0, Y_eff, X, U_ls, dists;
                                max_iter=max_iter, g_tol=tol_T)
        catch err
            _is_recoverable_draw_error(err) || rethrow(err)
            nothing
        end
        res_s === nothing && continue
        if isfinite(res_s.nll) && (!isfinite(best_nll) || res_s.nll < best_nll - T(1e-10))
            best_nll = res_s.nll
            best_step2 = res_s
        end
    end

    if best_step2 === nothing
        beta0, log_sig0 = _lms_safe_project_init(L_chol, L_chol, rng)
        lam_unc0 = _lms_default_lam_unc(dists, 1, T)
        best_step2 = _lms_optimize_gamma(pi_ls, beta0, log_sig0, lam_unc0, Y_eff, X, U_ls, dists;
                                         max_iter=max_iter, g_tol=tol_T)
    end

    # Project Step 2 onto B and polish with analytical gradients so ∇_γ L_T = 0
    aligned2 = _lms_realign_to_B(best_step2.beta, best_step2.sigma, best_step2.lam_con, dists)
    dists_curr = aligned2.dists
    lam_unc_pol = _lms_con_to_unc(dists_curr, aligned2.lam_list)
    res_polish = _lms_optimize_gamma(pi_ls, aligned2.beta, log.(aligned2.sigma), lam_unc_pol,
                                     Y_eff, X, U_ls, dists_curr;
                                     max_iter=max(max_iter, 500), g_tol=polish_tol)
    aligned2b = _lms_realign_to_B(res_polish.beta, res_polish.sigma, res_polish.lam_con, dists_curr)
    dists_curr = aligned2b.dists
    beta_hat = aligned2b.beta
    sigma_hat = aligned2b.sigma
    lam_con_hat = aligned2b.lam_con
    lam_list_hat = aligned2b.lam_list

    # Step 3: Adaptive optimization over π holding γ̃_T fixed
    res_step3 = _lms_optimize_pi(pi_ls, beta_hat, sigma_hat, lam_con_hat, Y_eff, X, dists_curr;
                                 max_iter=max_iter, g_tol=tol_T)
    pi_hat = res_step3.pi
    converged = res_polish.converged && res_step3.converged
    total_iters = best_step2.iters + res_polish.iters + res_step3.iters

    # Full 1-step ML (warm-started from :three_step)
    if method === :full_ml
        lam_unc_joint = _lms_con_to_unc(dists_curr, lam_list_hat)
        res_ml = _lms_optimize_joint(pi_hat, beta_hat, log.(sigma_hat), lam_unc_joint,
                                     Y_eff, X, dists_curr;
                                     max_iter=max_iter, g_tol=tol_T)
        total_iters += res_ml.iters
        aligned_ml = _lms_realign_to_B(res_ml.beta, res_ml.sigma, res_ml.lam_con, dists_curr)
        if aligned_ml.changed
            dists_curr = aligned_ml.dists
            lam_unc_ml2 = _lms_con_to_unc(dists_curr, aligned_ml.lam_list)
            res_ml2 = _lms_optimize_joint(res_ml.pi, aligned_ml.beta, log.(aligned_ml.sigma),
                                          lam_unc_ml2, Y_eff, X, dists_curr;
                                          max_iter=max_iter, g_tol=polish_tol)
            total_iters += res_ml2.iters
            pi_hat = res_ml2.pi
            beta_hat = res_ml2.beta
            sigma_hat = res_ml2.sigma
            lam_con_hat = res_ml2.lam_con
            lam_list_hat = _lms_split_lam(dists_curr, lam_con_hat)
            converged = res_ml2.converged
        else
            pi_hat = res_ml.pi
            beta_hat = aligned_ml.beta
            sigma_hat = aligned_ml.sigma
            lam_con_hat = aligned_ml.lam_con
            lam_list_hat = aligned_ml.lam_list
            converged = res_ml.converged
        end
    end

    # Assemble structural matrices, shocks, and polar Q
    B_hat = _lms_B_from_beta(beta_hat, n)
    B0_hat = B_hat * Diagonal(sigma_hat)
    Pi_hat = Matrix{T}(transpose(reshape(pi_hat, n, k)))
    B_inv_T = Matrix{T}(transpose(robust_inv(B_hat; silent=true)))
    U_hat = Y_eff - X * Pi_hat
    raw_shocks = Matrix{T}(U_hat * B_inv_T)
    shocks = Matrix{T}(raw_shocks * Diagonal(one(T) ./ sigma_hat))

    Q_raw = L_chol \ B0_hat
    F_polar = svd(Q_raw)
    Q_hat = Matrix{T}(F_polar.U * F_polar.Vt)

    theta_hat = vcat(pi_hat, beta_hat, sigma_hat, lam_con_hat)
    sh = _lms_score_hessian(theta_hat, Y_eff, X, dists_curr)
    loglik = T(T_eff) * sh.loglik_avg
    loglik_g = _gaussian_loglik(model)

    # Asymptotic covariance of θ̂
    V_theta = if se === :hessian
        _spd_inv(-sh.hessian) ./ T(T_eff)
    elseif se === :opg
        _spd_inv(sh.opg) ./ T(T_eff)
    else  # :sandwich
        H_inv = _spd_inv(-sh.hessian)
        V_raw = (H_inv * sh.opg * H_inv) ./ T(T_eff)
        Matrix{T}((V_raw .+ V_raw') .* T(0.5))
    end

    se_theta = sqrt.(max.(diag(V_theta), zero(T)))
    se_pi_vec = @view se_theta[1:n_pi]
    se_Pi = Matrix{T}(transpose(reshape(se_pi_vec, n, k)))
    se_beta_vec = n_beta > 0 ? @view(se_theta[(n_pi + 1):(n_pi + n_beta)]) : T[]
    se_B = zeros(T, n, n)
    if n_beta > 0
        idx_k = 0
        @inbounds for j in 1:n, i in 1:n
            if i != j
                idx_k += 1
                se_B[i, j] = se_beta_vec[idx_k]
            end
        end
    end
    se_sigma = Vector{T}(@view(se_theta[(n_pi + n_beta + 1):(n_pi + n_beta + n)]))

    # Delta-method standard errors for B₀ = B(β) * Diagonal(σ)
    idx_bs = (n_pi + 1):(n_pi + n_beta + n)
    params_bs = Vector{T}(@view(theta_hat[idx_bs]))
    V_bs = Matrix{T}(@view(V_theta[idx_bs, idx_bs]))
    B0_fn = bs -> _lms_B_from_beta(@view(bs[1:n_beta]), n) *
                  Diagonal(@view(bs[(n_beta + 1):(n_beta + n)]))
    se_B0, _ = _delta_B0_se(params_bs, V_bs, B0_fn, n)

    n_params = length(theta_hat)
    aic_val = -T(2) * loglik + T(2) * T(n_params)
    bic_val = -T(2) * loglik + log(T(T_eff)) * T(n_params)

    manifest = capture_manifest(; seed=seed,
        settings=Dict{String,Any}(
            "distributions" => String.(dists_curr),
            "method"        => String(method),
            "se"            => String(se),
            "n_starts"      => n_starts,
        ))

    LMSSVARResult{T}(
        B0_hat, B_hat, sigma_hat, Q_hat, Pi_hat, model, shocks, raw_shocks,
        dists_curr, lam_list_hat, theta_hat, loglik, loglik_g, V_theta,
        se_B0, se_B, se_sigma, se_Pi, method, se, converged, total_iters,
        aic_val, bic_val, snames, manifest,
    )
end

function estimate_lms_svar(Y::AbstractMatrix, p::Int;
                           check_stability::Bool=true,
                           varnames::Union{Nothing,Vector{String}}=nothing,
                           kwargs...)
    Ym = eltype(Y) <: AbstractFloat ? Y : Float64.(Y)
    model = estimate_var(Ym, p; check_stability=check_stability, varnames=varnames)
    estimate_lms_svar(model; kwargs...)
end
