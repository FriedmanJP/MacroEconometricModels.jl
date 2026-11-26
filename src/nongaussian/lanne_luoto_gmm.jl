# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Full-system non-Gaussian SVAR estimation by Generalized Method of Moments
(Lanne & Luoto 2021), Andrews (1999) + Hall et al. (2007) co-kurtosis moment
selection, Pham–Garat (1997) column normalization, and Newey–West (1987) GMM
LR-type distance hypothesis testing.

References:
- Lanne, M., & Luoto, J. (2021). "GMM Estimation of Non-Gaussian Structural Vector
  Autoregression." Journal of Business & Economic Statistics, 39(3), 772–783.
- Andrews, D. W. K. (1999). "Consistent Moment Selection Procedures for Generalized
  Method of Moments Estimation." Econometrica, 67(3), 543–564.
- Hall, A. R., Inoue, A., Jana, K., & Shin, C. (2007). "Information in Generalized
  Method of Moments Estimation and Entropy-Based Moment Selection." Journal of
  Econometrics, 138(2), 488–512.
- Newey, W. K., & West, K. D. (1987). "Hypothesis Testing with Efficient Method of
  Moments Estimation." International Economic Review, 28(3), 777–787.
- Pham, D. T., & Garat, P. (1997). "Blind separation of mixture of independent sources
  through a quasi-maximum likelihood approach." IEEE TSP, 45(7), 1712–1725.
"""

using LinearAlgebra, Statistics, Distributions, Random
import Optim
import ForwardDiff

# =============================================================================
# Exported Result Types (RSER-01 Compatible)
# =============================================================================

"""
    GMMMomentSelectionResult{T<:AbstractFloat}

Result from two-stage Andrews (1999) MSC and Hall, Inoue, Jana & Shin (2007) RMSC
co-kurtosis moment selection for Lanne–Luoto (2021) non-Gaussian SVAR GMM.

# Fields
- `asym_pairs::Vector{Tuple{Int,Int}}` — selected asymmetric co-kurtosis pairs `(i, j)` for ``E[\\varepsilon_{it}^3 \\varepsilon_{jt}] = 0``
- `sym_pairs::Vector{Tuple{Int,Int}}` — selected symmetric co-kurtosis pairs `(i, j)` with ``i < j`` for ``E[\\varepsilon_{it}^2 \\varepsilon_{jt}^2 - 1] = 0``
- `msc_asym_pairs::Vector{Tuple{Int,Int}}` — Stage-1 MSC selected asymmetric pairs
- `msc_sym_pairs::Vector{Tuple{Int,Int}}` — Stage-1 MSC selected symmetric pairs
- `msc_score::T` — Stage-1 minimum Andrews (1999) MSC value
- `rmsc_score::T` — Stage-2 minimum Hall et al. (2007) RMSC value
- `J_stat::T` — Hansen ``J``-test statistic at selected moments
- `J_pvalue::T` — ``J``-test p-value at selected moments
- `df::Int` — overidentifying degrees of freedom ``q - k``
- `n_moments::Int` — total moment conditions ``q`` of selected specification
- `n_params::Int` — total parameters ``k = n + (p + 1) n^2``
- `bandwidth::Int` — Newey–West HAC bandwidth ``b_T`` used in RMSC
- `method::Symbol` — selection strategy (`:msc_rmsc` or `:rmsc_seq`)
- `candidates::Vector{NamedTuple}` — evaluated candidate moment sets and diagnostics
"""
struct GMMMomentSelectionResult{T<:AbstractFloat}
    asym_pairs::Vector{Tuple{Int,Int}}
    sym_pairs::Vector{Tuple{Int,Int}}
    msc_asym_pairs::Vector{Tuple{Int,Int}}
    msc_sym_pairs::Vector{Tuple{Int,Int}}
    msc_score::T
    rmsc_score::T
    J_stat::T
    J_pvalue::T
    df::Int
    n_moments::Int
    n_params::Int
    bandwidth::Int
    method::Symbol
    candidates::Vector{NamedTuple{(:asym_pairs, :sym_pairs, :n_moments, :df, :J_stat, :J_pvalue, :msc, :rmsc, :rank_ok),
                                  Tuple{Vector{Tuple{Int,Int}}, Vector{Tuple{Int,Int}}, Int, Int, T, T, T, T, Bool}}}

    function GMMMomentSelectionResult{T}(
        asym_pairs::AbstractVector,
        sym_pairs::AbstractVector,
        msc_asym_pairs::AbstractVector,
        msc_sym_pairs::AbstractVector,
        msc_score::Real,
        rmsc_score::Real,
        J_stat::Real,
        J_pvalue::Real,
        df::Integer,
        n_moments::Integer,
        n_params::Integer,
        bandwidth::Integer,
        method::Symbol,
        candidates::AbstractVector,
    ) where {T<:AbstractFloat}
        ap = Tuple{Int,Int}[(Int(a[1]), Int(a[2])) for a in asym_pairs]
        sp = Tuple{Int,Int}[(Int(a[1]), Int(a[2])) for a in sym_pairs]
        map_p = Tuple{Int,Int}[(Int(a[1]), Int(a[2])) for a in msc_asym_pairs]
        msp_p = Tuple{Int,Int}[(Int(a[1]), Int(a[2])) for a in msc_sym_pairs]
        NT = NamedTuple{(:asym_pairs, :sym_pairs, :n_moments, :df, :J_stat, :J_pvalue, :msc, :rmsc, :rank_ok),
                        Tuple{Vector{Tuple{Int,Int}}, Vector{Tuple{Int,Int}}, Int, Int, T, T, T, T, Bool}}
        cands = NT[
            (
                asym_pairs = Tuple{Int,Int}[(Int(x[1]), Int(x[2])) for x in c.asym_pairs],
                sym_pairs  = Tuple{Int,Int}[(Int(x[1]), Int(x[2])) for x in c.sym_pairs],
                n_moments  = Int(c.n_moments),
                df         = Int(c.df),
                J_stat     = T(c.J_stat),
                J_pvalue   = T(c.J_pvalue),
                msc        = T(c.msc),
                rmsc       = T(c.rmsc),
                rank_ok    = Bool(c.rank_ok),
            ) for c in candidates
        ]
        new{T}(ap, sp, map_p, msp_p, T(msc_score), T(rmsc_score), T(J_stat), T(J_pvalue),
               Int(df), Int(n_moments), Int(n_params), Int(bandwidth), method, cands)
    end
end

GMMMomentSelectionResult(args...) =
    GMMMomentSelectionResult{typeof(float(args[5]))}(args...)

function Base.show(io::IO, r::GMMMomentSelectionResult{T}) where {T}
    spec = Any[
        "Method"             string(r.method);
        "Total moments (q)"  r.n_moments;
        "Parameters (k)"     r.n_params;
        "Overid. df (q - k)" r.df;
        "Selected asym."     length(r.asym_pairs);
        "Selected sym."      length(r.sym_pairs);
        "MSC score"          _fmt(r.msc_score);
        "RMSC score"         _fmt(r.rmsc_score);
        "Hansen J-stat"      _fmt(r.J_stat);
        "Hansen J p-value"   _format_pvalue(r.J_pvalue);
        "HAC bandwidth b_T"  r.bandwidth;
        "Candidates tested"  length(r.candidates)
    ]
    _pretty_table(io, spec;
        title = "Lanne–Luoto (2021) GMM Co-Kurtosis Moment Selection",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    pair_rows = Any[
        "Asymmetric (εᵢ³εⱼ)" isempty(r.asym_pairs) ? "none" : join(string.(r.asym_pairs), ", ");
        "Symmetric (εᵢ²εⱼ²−1)" isempty(r.sym_pairs) ? "none" : join(string.(r.sym_pairs), ", ")
    ]
    _pretty_table(io, pair_rows;
        title = "Selected Co-Kurtosis Conditions",
        column_labels = ["Block", "Pairs"],
        alignment = [:l, :l],
    )
end

report(io::IO, r::GMMMomentSelectionResult) = (show(io, r); nothing)
report(r::GMMMomentSelectionResult) = report(stdout, r)

"""
    LanneLuotoGMMResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR

Result from full-system Lanne & Luoto (2021) non-Gaussian SVAR GMM estimation.

Estimates the full ``k = n + (p + 1) n^2`` parameter vector
``\\theta = (\\nu', \\operatorname{vec}(A_1)', \\dots, \\operatorname{vec}(A_p)', \\operatorname{vec}(B)')'``
(or A-model ``A_0 u_t = \\operatorname{diag}(\\sigma) \\varepsilon_t`` with ``A_{0,ii} = 1``)
from reduced-form orthogonality, variance/covariance, and selected fourth-order
co-kurtosis moment conditions with HAC covariance weighting.

# Fields
- `B0::Matrix{T}` — Pham–Garat normalized structural impact matrix ``B_0`` (``n \\times n``, ``B_{0,ii} > 0``)
- `A0::Matrix{T}` — structural contemporaneous matrix ``A_0`` (``A_{0,ii} = 1`` in `:a_model`, ``B_0^{-1}`` in `:b_model`)
- `Q::Matrix{T}` — polar orthogonal factor relative to `safe_cholesky(Sigma)` (``n \\times n``)
- `sigma::Vector{T}` — structural shock scale vector (length ``n``)
- `nu::Vector{T}` — GMM-estimated VAR intercept ``\\hat{\\nu}`` (length ``n``)
- `A::Vector{Matrix{T}}` — GMM-estimated VAR lag matrices ``[\\hat{A}_1, \\dots, \\hat{A}_p]``
- `Pi::Matrix{T}` — stacked reduced-form coefficient matrix `(1 + n*p) × n` matching `VARModel.B`
- `Sigma::Matrix{T}` — implied reduced-form residual covariance ``\\hat{B}_0 \\hat{B}_0'`` (``n \\times n``)
- `var_ols::VARModel{T}` — underlying OLS reduced-form VAR model
- `shocks::Matrix{T}` — estimated structural shocks ``\\hat{\\varepsilon}_t`` (`T_eff × n`)
- `theta::Vector{T}` — full GMM parameter vector ``\\hat{\\theta}`` (length ``k``)
- `vcov::Matrix{T}` — finite-sample asymptotic covariance ``\\hat{V}_{\\theta} / T_{\\text{eff}}`` (``k \\times k``)
- `se_B0::Matrix{T}` — asymptotic standard errors for ``B_0`` (``n \\times n``)
- `se_theta::Vector{T}` — asymptotic standard errors for ``\\hat{\\theta}`` (length ``k``)
- `W::Matrix{T}` — final GMM weighting matrix (``q \\times q``)
- `S_hac::Matrix{T}` — HAC moment covariance matrix ``\\hat{S}_T(\\hat{\\theta})`` (``q \\times q``)
- `G_jac::Matrix{T}` — analytical moment Jacobian ``G_T(\\hat{\\theta})`` (``q \\times k``)
- `g_bar::Vector{T}` — sample moment vector ``\\bar{g}_T(\\hat{\\theta})`` (length ``q``)
- `J_stat::T` — Hansen ``J``-test statistic ``T_{\\text{eff}} \\bar{g}_T(\\hat{\\theta})' \\hat{W} \\bar{g}_T(\\hat{\\theta})``
- `J_pvalue::T` — ``J``-test p-value (``1.0`` when just-identified)
- `df::Int` — overidentifying degrees of freedom ``q - k``
- `n_moments::Int` — total moment conditions ``q``
- `n_params::Int` — total parameters ``k``
- `jacobian_rank::Int` — numerical rank of ``G_T(\\hat{\\theta})``
- `rank_ok::Bool` — `true` when `jacobian_rank == n_params`
- `asym_pairs::Vector{Tuple{Int,Int}}` — active asymmetric co-kurtosis pairs ``(i, j)``
- `sym_pairs::Vector{Tuple{Int,Int}}` — active symmetric co-kurtosis pairs ``(i, j)``
- `selection::Union{Nothing,GMMMomentSelectionResult{T}}` — moment selection summary when performed
- `form::Symbol` — `:b_model` or `:a_model`
- `weighting::Symbol` — `:two_step`, `:iterated`, or `:cue`
- `kernel::Symbol` — HAC kernel (`:bartlett`, `:parzen`, `:quadratic_spectral`, or `:tukey_hanning`)
- `bandwidth::Int` — HAC truncation lag ``b_T``
- `converged::Bool` — optimizer convergence flag
- `iterations::Int` — total optimization iterations
- `varnames::Vector{String}` — variable names
- `shock_names::Vector{String}` — structural shock names
- `manifest::Union{ReproManifest,Nothing}` — reproducibility manifest
"""
struct LanneLuotoGMMResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    A0::Matrix{T}
    Q::Matrix{T}
    sigma::Vector{T}
    nu::Vector{T}
    A::Vector{Matrix{T}}
    Pi::Matrix{T}
    Sigma::Matrix{T}
    var_ols::VARModel{T}
    shocks::Matrix{T}
    theta::Vector{T}
    vcov::Matrix{T}
    se_B0::Matrix{T}
    se_theta::Vector{T}
    W::Matrix{T}
    S_hac::Matrix{T}
    G_jac::Matrix{T}
    g_bar::Vector{T}
    J_stat::T
    J_pvalue::T
    df::Int
    n_moments::Int
    n_params::Int
    jacobian_rank::Int
    rank_ok::Bool
    asym_pairs::Vector{Tuple{Int,Int}}
    sym_pairs::Vector{Tuple{Int,Int}}
    selection::Union{Nothing,GMMMomentSelectionResult{T}}
    form::Symbol
    weighting::Symbol
    kernel::Symbol
    bandwidth::Int
    converged::Bool
    iterations::Int
    varnames::Vector{String}
    shock_names::Vector{String}
    manifest::Union{ReproManifest,Nothing}

    function LanneLuotoGMMResult{T}(
        B0::AbstractMatrix,
        A0::AbstractMatrix,
        Q::AbstractMatrix,
        sigma::AbstractVector,
        nu::AbstractVector,
        A::AbstractVector,
        Pi::AbstractMatrix,
        Sigma::AbstractMatrix,
        var_ols::VARModel{T},
        shocks::AbstractMatrix,
        theta::AbstractVector,
        vcov::AbstractMatrix,
        se_B0::AbstractMatrix,
        se_theta::AbstractVector,
        W::AbstractMatrix,
        S_hac::AbstractMatrix,
        G_jac::AbstractMatrix,
        g_bar::AbstractVector,
        J_stat::Real,
        J_pvalue::Real,
        df::Integer,
        n_moments::Integer,
        n_params::Integer,
        jacobian_rank::Integer,
        rank_ok::Bool,
        asym_pairs::AbstractVector,
        sym_pairs::AbstractVector,
        selection::Union{Nothing,GMMMomentSelectionResult{T}},
        form::Symbol,
        weighting::Symbol,
        kernel::Symbol,
        bandwidth::Integer,
        converged::Bool,
        iterations::Integer,
        varnames::AbstractVector,
        shock_names::AbstractVector,
        manifest::Union{ReproManifest,Nothing}=nothing,
    ) where {T<:AbstractFloat}
        A_mats = Matrix{T}[Matrix{T}(Al) for Al in A]
        ap = Tuple{Int,Int}[(Int(x[1]), Int(x[2])) for x in asym_pairs]
        sp = Tuple{Int,Int}[(Int(x[1]), Int(x[2])) for x in sym_pairs]
        vn = String[string(s) for s in varnames]
        sn = String[string(s) for s in shock_names]
        new{T}(
            Matrix{T}(B0), Matrix{T}(A0), Matrix{T}(Q),
            Vector{T}(sigma), Vector{T}(nu), A_mats,
            Matrix{T}(Pi), Matrix{T}(Sigma), var_ols,
            Matrix{T}(shocks), Vector{T}(theta), Matrix{T}(vcov),
            Matrix{T}(se_B0), Vector{T}(se_theta),
            Matrix{T}(W), Matrix{T}(S_hac), Matrix{T}(G_jac),
            Vector{T}(g_bar), T(J_stat), T(J_pvalue),
            Int(df), Int(n_moments), Int(n_params),
            Int(jacobian_rank), rank_ok, ap, sp, selection,
            form, weighting, kernel, Int(bandwidth),
            converged, Int(iterations), vn, sn, manifest,
        )
    end
end

LanneLuotoGMMResult(args...) =
    LanneLuotoGMMResult{eltype(args[1])}(args...)

function Base.getproperty(r::LanneLuotoGMMResult, s::Symbol)
    s === :se && return getfield(r, :se_B0)
    s === :B && return getfield(r, :B0)
    s === :J && return getfield(r, :J_stat)
    s === :residuals && return getfield(r, :shocks)
    return getfield(r, s)
end

function Base.propertynames(::LanneLuotoGMMResult, private::Bool=false)
    return (fieldnames(LanneLuotoGMMResult)..., :se, :B, :J, :residuals)
end

nvars(r::LanneLuotoGMMResult) = size(r.B0, 1)
nlags(r::LanneLuotoGMMResult) = r.var_ols.p
effective_nobs(r::LanneLuotoGMMResult) = size(r.shocks, 1)

StatsAPI.nobs(r::LanneLuotoGMMResult) = size(r.shocks, 1)
StatsAPI.coef(r::LanneLuotoGMMResult) = r.theta
StatsAPI.vcov(r::LanneLuotoGMMResult) = r.vcov
StatsAPI.stderror(r::LanneLuotoGMMResult) = r.se_theta
StatsAPI.residuals(r::LanneLuotoGMMResult) = r.shocks
StatsAPI.dof(r::LanneLuotoGMMResult) = r.n_params
StatsAPI.dof_residual(r::LanneLuotoGMMResult) = size(r.shocks, 1) * size(r.B0, 1) - r.n_params
StatsAPI.islinear(::LanneLuotoGMMResult) = false

function StatsAPI.confint(r::LanneLuotoGMMResult{T}; level::Real=DEFAULT_CI_MACRO) where {T}
    lv = _validate_conf_level(level)
    z = T(_conf_z(lv))
    hcat(r.theta .- z .* r.se_theta, r.theta .+ z .* r.se_theta)
end

function Base.show(io::IO, r::LanneLuotoGMMResult{T}) where {T}
    n = size(r.B0, 1)
    p = r.var_ols.p
    spec = Any[
        "Model form"         string(r.form);
        "Weighting"          string(r.weighting);
        "Variables"          n;
        "Lags"               p;
        "Observations (eff)" size(r.shocks, 1);
        "Parameters (k)"     r.n_params;
        "Moments (q)"        r.n_moments;
        "Overid. df"         r.df;
        "Jacobian rank"      "$(r.jacobian_rank) / $(r.n_params) ($(r.rank_ok ? "full" : "DEFICIENT"))";
        "HAC kernel"         "$(r.kernel) (bw=$(r.bandwidth))";
        "Hansen J-stat"      _fmt(r.J_stat);
        "Hansen J p-value"   _format_pvalue(r.J_pvalue);
        "Converged"          r.converged ? "Yes" : "No";
        "Iterations"         r.iterations
    ]
    _pretty_table(io, spec;
        title = "Lanne–Luoto (2021) Full-System Non-Gaussian SVAR GMM",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _show_B0_ses(io, r.B0, r.se_B0)
    if r.form === :a_model
        _matrix_table(io, r.A0, "Contemporaneous Matrix A₀ (unit diagonal)";
            row_labels=r.varnames, col_labels=r.shock_names)
    end
    pair_rows = Any[
        "Asymmetric (εᵢ³εⱼ)" isempty(r.asym_pairs) ? "none" : join(string.(r.asym_pairs), ", ");
        "Symmetric (εᵢ²εⱼ²−1)" isempty(r.sym_pairs) ? "none" : join(string.(r.sym_pairs), ", ")
    ]
    _pretty_table(io, pair_rows;
        title = "Active Co-Kurtosis Conditions",
        column_labels = ["Block", "Pairs"],
        alignment = [:l, :l],
    )
    if !r.rank_ok
        _show_note(io, "WARNING: Jacobian rank $(r.jacobian_rank) < $(r.n_params); parameters may be weakly or under-identified.")
    end
    _manifest_footer(io, r.manifest)
end

report(io::IO, r::LanneLuotoGMMResult; repro::Bool=false) = show(IOContext(io, :repro => repro), r)
report(r::LanneLuotoGMMResult; repro::Bool=false) = report(stdout, r; repro=repro)

# =============================================================================
# Internal Pair, Normalization & Indexing Helpers (_llgmm_*)
# =============================================================================

_llgmm_all_asym_pairs(n::Int) = Tuple{Int,Int}[(i, j) for i in 1:n for j in 1:n if i != j]
_llgmm_all_sym_pairs(n::Int) = Tuple{Int,Int}[(i, j) for i in 1:(n - 1) for j in (i + 1):n]

function _llgmm_validate_pairs(pairs, n::Int; symmetric::Bool)
    out = Tuple{Int,Int}[]
    for item in pairs
        length(item) == 2 || throw(ArgumentError("pair must have length 2, got $item"))
        i, j = Int(item[1]), Int(item[2])
        (1 <= i <= n && 1 <= j <= n) || throw(ArgumentError("pair ($i, $j) out of bounds 1:$n"))
        i == j && throw(ArgumentError("co-kurtosis pair ($i, $j) must satisfy i ≠ j"))
        p_norm = symmetric ? (min(i, j), max(i, j)) : (i, j)
        push!(out, p_norm)
    end
    length(unique(out)) == length(out) || throw(ArgumentError("duplicate co-kurtosis pairs supplied"))
    out
end

function _llgmm_n_base_moments(n::Int, p::Int)
    n * (1 + n * p) + n + (n * (n - 1)) ÷ 2
end

function _llgmm_n_params(n::Int, p::Int)
    n + (p + 1) * n * n
end

"""Check that every unordered pair {i, j} (1 <= i < j <= n) has at least one asymmetric condition."""
function _llgmm_covers_all_pairs(asym_pairs::Vector{Tuple{Int,Int}}, n::Int)
    covered = falses(n, n)
    @inbounds for (i, j) in asym_pairs
        a, b = min(i, j), max(i, j)
        covered[a, b] = true
    end
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        covered[i, j] || return false
    end
    return true
end

"""Column-major off-diagonal vectorization matching `_vecd0`."""
function _llgmm_vecd0(M::AbstractMatrix{T}) where {T}
    n = size(M, 1)
    v = Vector{T}(undef, n * (n - 1))
    k = 0
    @inbounds for j in 1:n, i in 1:n
        if i != j
            k += 1
            v[k] = M[i, j]
        end
    end
    v
end

function _llgmm_unit_diag_from_off(off::AbstractVector{Tθ}, n::Int) where {Tθ<:Real}
    M = Matrix{Tθ}(I, n, n)
    k = 0
    @inbounds for j in 1:n, i in 1:n
        if i != j
            k += 1
            M[i, j] = off[k]
        end
    end
    M
end

function _llgmm_unpack_theta(theta::AbstractVector{Tθ}, n::Int, p::Int, form::Symbol) where {Tθ<:Real}
    nu = @view theta[1:n]
    n_lag_tot = p * n * n
    # Use clean reshaped matrices (Real-generic)
    Pi_T = Matrix{Tθ}(undef, n, 1 + n * p)
    @inbounds for i in 1:n
        Pi_T[i, 1] = nu[i]
    end
    @inbounds for l in 1:p
        off_l = n + (l - 1) * n * n
        col0 = 1 + (l - 1) * n
        for j in 1:n, i in 1:n
            Pi_T[i, col0 + j] = theta[off_l + i + (j - 1) * n]
        end
    end
    s_off = n + n_lag_tot
    if form === :b_model
        B = Matrix{Tθ}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            B[i, j] = theta[s_off + i + (j - 1) * n]
        end
        return Pi_T, B, nothing, nothing
    else
        n_off = n * (n - 1)
        A0 = _llgmm_unit_diag_from_off(@view(theta[(s_off + 1):(s_off + n_off)]), n)
        sigma = @view theta[(s_off + n_off + 1):(s_off + n_off + n)]
        return Pi_T, nothing, A0, sigma
    end
end

function _llgmm_pack_theta(nu::AbstractVector{T}, A_mats::Vector{Matrix{T}},
                           Smat::AbstractMatrix{T}, sigma::Union{Nothing,AbstractVector{T}},
                           form::Symbol) where {T<:AbstractFloat}
    n = length(nu)
    p = length(A_mats)
    parts = Vector{T}[Vector{T}(nu)]
    for l in 1:p
        push!(parts, vec(A_mats[l]))
    end
    if form === :b_model
        push!(parts, vec(Smat))
    else
        push!(parts, _llgmm_vecd0(Smat))
        push!(parts, Vector{T}(sigma))
    end
    vcat(parts...)
end

function _llgmm_shocks_and_resids(theta::AbstractVector{Tθ}, Y_eff::AbstractMatrix{T},
                                  X::AbstractMatrix{T}, n::Int, p::Int,
                                  form::Symbol) where {Tθ<:Real, T<:AbstractFloat}
    Pi_T, B, A0, sigma = _llgmm_unpack_theta(theta, n, p, form)
    Pi = transpose(Pi_T)
    U = Y_eff .- X * Pi
    if form === :b_model
        all(isfinite, B) || return zeros(Tθ, size(Y_eff)), U, zeros(Tθ, n, n), nothing, nothing, false
        detB = det(B)
        ref = max(opnorm(B, 1)^n, Tθ(1e-12))
        if !isfinite(detB) || abs(detB) <= Tθ(1e-12) * ref
            return zeros(Tθ, size(Y_eff)), U, zeros(Tθ, n, n), nothing, nothing, false
        end
        B_inv = robust_inv(B; silent=true)
        all(isfinite, B_inv) || return zeros(Tθ, size(Y_eff)), U, B_inv, nothing, nothing, false
        E = U * transpose(B_inv)
        return E, U, B_inv, B, nothing, true
    else
        (all(isfinite, A0) && all(isfinite, sigma)) ||
            return zeros(Tθ, size(Y_eff)), U, zeros(Tθ, n, n), A0, sigma, false
        any(s -> s <= Tθ(1e-8), sigma) &&
            return zeros(Tθ, size(Y_eff)), U, zeros(Tθ, n, n), A0, sigma, false
        detA = det(A0)
        ref = max(opnorm(A0, 1)^n, Tθ(1e-12))
        if !isfinite(detA) || abs(detA) <= Tθ(1e-12) * ref
            return zeros(Tθ, size(Y_eff)), U, zeros(Tθ, n, n), A0, sigma, false
        end
        M_A = Diagonal(one(Tθ) ./ sigma) * A0
        E = U * transpose(M_A)
        return E, U, M_A, A0, sigma, true
    end
end

"""
Pham & Garat (1997) column normalization of structural impact matrix `B`:
find column permutation maximizing ``\\prod_{i=1}^n |B_{ii}|`` and enforce ``B_{ii} > 0``.
"""
function _llgmm_pham_garat(B::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(B, 1)
    size(B, 2) == n || throw(ArgumentError("B must be square"))
    best_perm = collect(1:n)
    floor_v = T(1e-15)
    if n <= 8
        best_score = T(-Inf)
        for perm in _permutations(n)
            sc = zero(T)
            @inbounds for i in 1:n
                sc += log(max(abs(B[i, perm[i]]), floor_v))
            end
            if sc > best_score + T(1e-12)
                best_score = sc
                best_perm = copy(perm)
            end
        end
    else
        # Greedy log-magnitude assignment + pairwise 2-opt swap refinement
        used = falses(n)
        for i in 1:n
            best_c = 1
            best_val = T(-Inf)
            for c in 1:n
                used[c] && continue
                v = log(max(abs(B[i, c]), floor_v))
                if v > best_val
                    best_val = v
                    best_c = c
                end
            end
            best_perm[i] = best_c
            used[best_c] = true
        end
        changed = true
        while changed
            changed = false
            for i in 1:(n - 1), j in (i + 1):n
                pi, pj = best_perm[i], best_perm[j]
                cur = log(max(abs(B[i, pi]), floor_v)) + log(max(abs(B[j, pj]), floor_v))
                swp = log(max(abs(B[i, pj]), floor_v)) + log(max(abs(B[j, pi]), floor_v))
                if swp > cur + T(1e-12)
                    best_perm[i], best_perm[j] = pj, pi
                    changed = true
                end
            end
        end
    end
    B_perm = Matrix{T}(B[:, best_perm])
    signs = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        if B_perm[j, j] < zero(T)
            signs[j] = -1
            B_perm[:, j] .*= -one(T)
        else
            signs[j] = 1
        end
    end
    (B=B_perm, perm=best_perm, signs=signs)
end

function _llgmm_remap_pairs(asym_pairs::Vector{Tuple{Int,Int}},
                            sym_pairs::Vector{Tuple{Int,Int}},
                            perm::Vector{Int})
    invp = invperm(perm)
    asym_new = Tuple{Int,Int}[(invp[i], invp[j]) for (i, j) in asym_pairs]
    sym_new = Tuple{Int,Int}[let a = invp[i], b = invp[j]; (min(a, b), max(a, b)) end for (i, j) in sym_pairs]
    (asym_new, sym_new)
end

"""Convert a Pham–Garat normalized B0 matrix to A-model `(A0, sigma)` with `A_{0,ii} = 1`."""
function _llgmm_b0_to_a0(B0::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(B0, 1)
    B0_inv = Matrix{T}(robust_inv(B0; silent=true))
    d = diag(B0_inv)
    A0 = similar(B0_inv)
    sigma = Vector{T}(undef, n)
    B0_adj = copy(B0)
    signs_flip = ones(Int, n)
    @inbounds for i in 1:n
        di = d[i]
        abs_di = abs(di)
        if abs_di < T(1e-12)
            # Small pivot fallback: rescale by largest row entry magnitude (degenerate case)
            di_use = di >= zero(T) ? T(1e-12) : -T(1e-12)
        else
            di_use = di
        end
        if di_use < zero(T)
            signs_flip[i] = -1
            B0_adj[:, i] .*= -one(T)
        end
        scale_i = one(T) / di_use
        sigma[i] = abs(scale_i)
        for j in 1:n
            A0[i, j] = i == j ? one(T) : B0_inv[i, j] * scale_i
        end
    end
    # Verify exact identity A0 * B0_adj = Diagonal(sigma)
    (A0=A0, sigma=sigma, B0=B0_adj, signs_flip=signs_flip)
end

# =============================================================================
# Per-Observation Moments & Exact Analytical Jacobian (LL Appendix eqs. A.1–A.4)
# =============================================================================

function _llgmm_moment_matrix(theta::AbstractVector{Tθ}, Y_eff::AbstractMatrix{T},
                              X::AbstractMatrix{T}, n::Int, p::Int, form::Symbol,
                              asym_pairs::Vector{Tuple{Int,Int}},
                              sym_pairs::Vector{Tuple{Int,Int}}) where {Tθ<:Real, T<:AbstractFloat}
    E, _, _, _, _, ok = _llgmm_shocks_and_resids(theta, Y_eff, X, n, p, form)
    T_eff = size(Y_eff, 1)
    nx = size(X, 2)
    q = _llgmm_n_base_moments(n, p) + length(asym_pairs) + length(sym_pairs)
    ok || return zeros(Tθ, T_eff, q), false

    M = Matrix{Tθ}(undef, T_eff, q)
    col = 1

    # (5a) VAR orthogonality: E[ε_t ⊗ x_{t-1}] = 0
    @inbounds for i in 1:n, m in 1:nx
        for t in 1:T_eff
            M[t, col] = E[t, i] * X[t, m]
        end
        col += 1
    end

    # (5b) Unit shock variances: E[ε_{it}^2 - 1] = 0
    @inbounds for i in 1:n
        for t in 1:T_eff
            M[t, col] = E[t, i]^2 - one(Tθ)
        end
        col += 1
    end

    # (5c) Uncorrelated shocks: E[ε_{it} ε_{jt}] = 0 (i < j)
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        for t in 1:T_eff
            M[t, col] = E[t, i] * E[t, j]
        end
        col += 1
    end

    # (6) Asymmetric co-kurtosis: E[ε_{it}^3 ε_{jt}] = 0 (i != j)
    @inbounds for (i, j) in asym_pairs
        for t in 1:T_eff
            e_i = E[t, i]
            M[t, col] = e_i * e_i * e_i * E[t, j]
        end
        col += 1
    end

    # (7) Symmetric co-kurtosis: E[ε_{it}^2 ε_{jt}^2 - 1] = 0 (i < j)
    @inbounds for (i, j) in sym_pairs
        for t in 1:T_eff
            e_i = E[t, i]
            e_j = E[t, j]
            M[t, col] = e_i * e_i * e_j * e_j - one(Tθ)
        end
        col += 1
    end

    return M, all(isfinite, M)
end

"""
Compute weighted or unweighted Jacobian row contributions from shock derivative weights
`(row -> [(shock_idx, T_eff-vector of ∂m_t/∂ε_{shock,t})])`.
When `time_weights === nothing`, computes `G_T(θ) = T⁻¹ Σ_t ∂g_t/∂θ'` (`q × k`).
When `time_weights::Vector{T}` is supplied, computes `T⁻¹ Σ_t time_weights[t] ∂g_t/∂θ'`
(used by exact CUE derivative `-v' (∂S/∂θ) v`).
"""
function _llgmm_jacobian_impl(theta::AbstractVector{T}, Y_eff::AbstractMatrix{T},
                              X::AbstractMatrix{T}, n::Int, p::Int, form::Symbol,
                              asym_pairs::Vector{Tuple{Int,Int}},
                              sym_pairs::Vector{Tuple{Int,Int}};
                              time_weights::Union{Nothing,AbstractVector{T}}=nothing) where {T<:AbstractFloat}
    E, U, M_inv, _, sigma, ok = _llgmm_shocks_and_resids(theta, Y_eff, X, n, p, form)
    T_eff = size(Y_eff, 1)
    nx = size(X, 2)
    n_pi = n * nx
    k = _llgmm_n_params(n, p)
    q = _llgmm_n_base_moments(n, p) + length(asym_pairs) + length(sym_pairs)
    ok || return zeros(T, T_eff, q), zeros(T, q), zeros(T, q, k), false

    M = Matrix{T}(undef, T_eff, q)
    G_jac = zeros(T, q, k)
    invT = one(T) / T(T_eff)

    # Pre-scale design / residual / shock blocks if time_weights provided
    Xw = time_weights === nothing ? X : X .* time_weights
    Ew = time_weights === nothing ? E : E .* time_weights
    Uw = time_weights === nothing ? U : U .* time_weights

    # Precompute basic cross-products for base moments
    # C_XX_w = (X' * Xw) / T_eff  [nx × nx]
    C_XX = (transpose(X) * Xw) .* invT
    # C_XE_w = (X' * Ew) / T_eff  [nx × n]
    C_XE = (transpose(X) * Ew) .* invT
    # C_EE_w = (E' * Ew) / T_eff  [n × n]
    C_EE = (transpose(E) * Ew) .* invT

    # Helper to fill Jacobian row `row` given moment-row average `T⁻¹ Σ_t w_{t,s} * (∂ε_{st}/∂θ')`
    # For shock `s`, we need:
    #   avg_w_X  = (1/T) Σ_t w_{t,s} * X[t, :]          (length nx)
    #   avg_w_E  = (1/T) Σ_t w_{t,s} * E[t, :]          (length n)   [:b_model]
    #   avg_w_U  = (1/T) Σ_t w_{t,s} * U[t, :]          (length n)   [:a_model]
    #   avg_w_Es = (1/T) Σ_t w_{t,s} * E[t, s]          (scalar)     [:a_model]
    @inline function _add_shock_contrib!(row::Int, s::Int,
                                         avg_w_X::AbstractVector{T},
                                         avg_w_E::AbstractVector{T},
                                         avg_w_U::Union{Nothing,AbstractVector{T}})
        # π block: ∂ε_{st}/∂Π'_{r, m} = -M_inv[s, r] * X[t, m]
        @inbounds for m in 1:nx
            x_val = avg_w_X[m]
            col_off = (m - 1) * n
            for r in 1:n
                G_jac[row, col_off + r] -= M_inv[s, r] * x_val
            end
        end
        if form === :b_model
            # B block: ∂ε_{st}/∂B_{r, c} = -M_inv[s, r] * E[t, c]
            @inbounds for c in 1:n
                e_val = avg_w_E[c]
                col_off = n_pi + (c - 1) * n
                for r in 1:n
                    G_jac[row, col_off + r] -= M_inv[s, r] * e_val
                end
            end
        else
            # A0 off-diagonal block: ∂ε_{st}/∂A0_{r, c} = δ_{s, r} * U[t, c] / σ_s (r ≠ c)
            inv_sig = one(T) / sigma[s]
            @inbounds for c in 1:n
                c == s && continue
                idx_a0 = (c - 1) * (n - 1) + (s < c ? s : s - 1)
                G_jac[row, n_pi + idx_a0] += avg_w_U[c] * inv_sig
            end
            # σ block: ∂ε_{st}/∂σ_s = -E[t, s] / σ_s
            n_off = n * (n - 1)
            G_jac[row, n_pi + n_off + s] -= avg_w_E[s] * inv_sig
        end
    end

    C_XU = form === :a_model ? (transpose(X) * Uw) .* invT : zeros(T, 0, 0)
    C_EU = form === :a_model ? (transpose(E) * Uw) .* invT : zeros(T, 0, 0)

    col = 1
    # (5a) E[ε_{it} x_{t-1, m}] = 0: shock i enters with w_{t,i} = X[t, m]
    @inbounds for i in 1:n, m in 1:nx
        for t in 1:T_eff
            M[t, col] = E[t, i] * X[t, m]
        end
        avg_X = @view C_XX[:, m]
        avg_E = @view C_XE[m, :]
        avg_U = form === :a_model ? @view(C_XU[m, :]) : nothing
        _add_shock_contrib!(col, i, avg_X, avg_E, avg_U)
        col += 1
    end

    # (5b) E[ε_{it}^2 - 1] = 0: shock i enters with w_{t,i} = 2 ε_{it}
    @inbounds for i in 1:n
        for t in 1:T_eff
            M[t, col] = E[t, i]^2 - one(T)
        end
        avg_X = T(2) .* @view(C_XE[:, i])
        avg_E = T(2) .* @view(C_EE[:, i])
        avg_U = form === :a_model ? T(2) .* @view(C_EU[i, :]) : nothing
        _add_shock_contrib!(col, i, avg_X, avg_E, avg_U)
        col += 1
    end

    # (5c) E[ε_{it} ε_{jt}] = 0 (i < j): w_{t,i} = ε_{jt}, w_{t,j} = ε_{it}
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        for t in 1:T_eff
            M[t, col] = E[t, i] * E[t, j]
        end
        _add_shock_contrib!(col, i, @view(C_XE[:, j]), @view(C_EE[:, j]),
                            form === :a_model ? @view(C_EU[j, :]) : nothing)
        _add_shock_contrib!(col, j, @view(C_XE[:, i]), @view(C_EE[:, i]),
                            form === :a_model ? @view(C_EU[i, :]) : nothing)
        col += 1
    end

    # Co-kurtosis moments: batch weight buffers to keep allocations minimal
    n_asym = length(asym_pairs)
    n_sym = length(sym_pairs)
    n_ck = n_asym + n_sym
    if n_ck > 0
        # For each co-kurtosis moment, there are 2 shock weights w1 and w2
        W1 = Matrix{T}(undef, T_eff, n_ck)
        W2 = Matrix{T}(undef, T_eff, n_ck)
        idx_ck = 1
        @inbounds for (i, j) in asym_pairs
            for t in 1:T_eff
                e_i = E[t, i]
                e_j = E[t, j]
                e_i2 = e_i * e_i
                e_i3 = e_i2 * e_i
                M[t, col] = e_i3 * e_j
                W1[t, idx_ck] = T(3) * e_i2 * e_j
                W2[t, idx_ck] = e_i3
            end
            col += 1
            idx_ck += 1
        end
        @inbounds for (i, j) in sym_pairs
            for t in 1:T_eff
                e_i = E[t, i]
                e_j = E[t, j]
                e_i2 = e_i * e_i
                e_j2 = e_j * e_j
                M[t, col] = e_i2 * e_j2 - one(T)
                W1[t, idx_ck] = T(2) * e_i * e_j2
                W2[t, idx_ck] = T(2) * e_i2 * e_j
            end
            col += 1
            idx_ck += 1
        end

        C_W1X = (transpose(W1) * Xw) .* invT
        C_W2X = (transpose(W2) * Xw) .* invT
        C_W1E = (transpose(W1) * Ew) .* invT
        C_W2E = (transpose(W2) * Ew) .* invT
        C_W1U = form === :a_model ? (transpose(W1) * Uw) .* invT : zeros(T, 0, 0)
        C_W2U = form === :a_model ? (transpose(W2) * Uw) .* invT : zeros(T, 0, 0)

        row_base = _llgmm_n_base_moments(n, p)
        idx_ck = 1
        @inbounds for (i, j) in asym_pairs
            r = row_base + idx_ck
            _add_shock_contrib!(r, i, @view(C_W1X[idx_ck, :]), @view(C_W1E[idx_ck, :]),
                                form === :a_model ? @view(C_W1U[idx_ck, :]) : nothing)
            _add_shock_contrib!(r, j, @view(C_W2X[idx_ck, :]), @view(C_W2E[idx_ck, :]),
                                form === :a_model ? @view(C_W2U[idx_ck, :]) : nothing)
            idx_ck += 1
        end
        @inbounds for (i, j) in sym_pairs
            r = row_base + idx_ck
            _add_shock_contrib!(r, i, @view(C_W1X[idx_ck, :]), @view(C_W1E[idx_ck, :]),
                                form === :a_model ? @view(C_W1U[idx_ck, :]) : nothing)
            _add_shock_contrib!(r, j, @view(C_W2X[idx_ck, :]), @view(C_W2E[idx_ck, :]),
                                form === :a_model ? @view(C_W2U[idx_ck, :]) : nothing)
            idx_ck += 1
        end
    end

    g_bar = vec(mean(M; dims=1))
    ok_all = all(isfinite, M) && all(isfinite, g_bar) && all(isfinite, G_jac)
    return M, g_bar, G_jac, ok_all
end

# =============================================================================
# HAC Covariance Matrix & Bandwidth Selection
# =============================================================================

function _llgmm_resolve_bw(M_demean::AbstractMatrix{T}, ki::Symbol, bandwidth) where {T<:AbstractFloat}
    T_eff = size(M_demean, 1)
    if bandwidth isa Symbol
        bw = if bandwidth === :auto || bandwidth === :nw94
            optimal_bandwidth_nw94(M_demean; kernel=ki)
        elseif bandwidth === :andrews
            optimal_bandwidth_nw(M_demean; kernel=ki)
        else
            throw(ArgumentError("bandwidth Symbol must be :auto, :nw94, or :andrews; got :$bandwidth"))
        end
        schwert_cap = floor(Int, 12 * (T(T_eff) / 100)^(one(T) / 4))
        return clamp(bw, 0, min(schwert_cap, max(0, T_eff - 2)))
    else
        bw_i = Int(bandwidth)
        bw_i >= 0 || throw(ArgumentError("bandwidth must be ≥ 0, got $bandwidth"))
        return min(bw_i, max(0, T_eff - 2))
    end
end

function _llgmm_hac_cov(M::AbstractMatrix{T}, kernel::Symbol, bandwidth;
                        fixed_bw::Union{Nothing,Int}=nothing) where {T<:AbstractFloat}
    T_eff, q = size(M)
    M_demean = M .- mean(M; dims=1)
    ki = _lrv_kernel(kernel)
    bw = fixed_bw === nothing ? _llgmm_resolve_bw(M_demean, ki, bandwidth) : fixed_bw
    if T_eff < 2
        return Matrix{T}(I, q, q), 0
    end
    S_raw = if bw == 0 && ki !== :quadratic_spectral
        Matrix{T}((transpose(M_demean) * M_demean) ./ T(T_eff))
    else
        _lrv_accumulate(M_demean, bw, ki)[1]
    end
    S_psd = _psd_project(S_raw)
    return S_psd, bw
end

function _llgmm_inv_S(S::AbstractMatrix{T}) where {T<:AbstractFloat}
    q = size(S, 1)
    Ss = Matrix{T}((S + transpose(S)) .* T(0.5))
    scale_ref = max(one(T), tr(Ss) / T(max(q, 1)))
    eigvals_S = eigvals(Symmetric(Ss))
    min_ev = isempty(eigvals_S) ? zero(T) : minimum(eigvals_S)
    if !isfinite(min_ev) || min_ev <= T(1e-11) * scale_ref
        jitter = T(1e-8) * scale_ref
        if isfinite(min_ev) && min_ev < zero(T)
            jitter += -min_ev
        end
        Ss = Ss + jitter * I
    end
    W = Matrix{T}(robust_inv(Ss; silent=true))
    Matrix{T}((W + transpose(W)) .* T(0.5))
end

function _llgmm_jacobian_rank(G_jac::AbstractMatrix{T}) where {T<:AbstractFloat}
    isempty(G_jac) && return 0
    sv = svdvals(G_jac)
    isempty(sv) && return 0
    tol_rk = T(100) * T(max(size(G_jac)...)) * eps(T) * max(one(T), sv[1])
    count(>(tol_rk), sv)
end

# =============================================================================
# Initialization & Optimization Solvers (Fixed-W Gauss-Newton + CUE Analytical)
# =============================================================================

function _llgmm_cokr_discrepancy(Q::AbstractMatrix{Tθ}, Z::AbstractMatrix{T},
                                 asym_pairs::Vector{Tuple{Int,Int}},
                                 sym_pairs::Vector{Tuple{Int,Int}}) where {Tθ<:Real, T<:AbstractFloat}
    E = Z * Q
    T_eff = size(E, 1)
    invT = one(Tθ) / Tθ(T_eff)
    loss = zero(Tθ)
    @inbounds for (i, j) in asym_pairs
        s = zero(Tθ)
        for t in 1:T_eff
            e_i = E[t, i]
            s += e_i * e_i * e_i * E[t, j]
        end
        m = s * invT
        loss += m * m
    end
    @inbounds for (i, j) in sym_pairs
        s = zero(Tθ)
        for t in 1:T_eff
            e_i = E[t, i]
            e_j = E[t, j]
            s += e_i * e_i * e_j * e_j - one(Tθ)
        end
        m = s * invT
        loss += m * m
    end
    loss
end

function _llgmm_init_theta(model::VARModel{T}, Y_eff::AbstractMatrix{T}, X::AbstractMatrix{T},
                           form::Symbol, asym_pairs::Vector{Tuple{Int,Int}},
                           sym_pairs::Vector{Tuple{Int,Int}},
                           n_starts::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    n = nvars(model)
    p = model.p
    nu_ols = Vector{T}(@view(model.B[1, :]))
    A_ols = extract_ar_coefficients(model.B, n, p)
    L = Matrix{T}(safe_cholesky(model.Sigma; silent=true))
    Z = Matrix{T}(model.U / transpose(L))
    n_angles = (n * (n - 1)) ÷ 2

    best_Q = Matrix{T}(I, n, n)
    best_loss = T(Inf)

    if n_angles > 0
        starts = Vector{Vector{T}}()
        push!(starts, zeros(T, n_angles))
        if n == 2
            # Dense coarse grid over [-π/4, π/4] guards against local rotation basins in n=2
            for ang in range(T(-0.7), T(0.7); length=max(7, n_starts))
                push!(starts, T[ang])
            end
        else
            for _ in 2:max(n_starts, 2)
                push!(starts, T(π) .* (rand(rng, T, n_angles) .- T(0.5)))
            end
        end

        obj_giv = angs -> _llgmm_cokr_discrepancy(_givens_to_orthogonal(angs, n), Z, asym_pairs, sym_pairs)
        for ang0 in starts
            res = try
                cfg = ForwardDiff.GradientConfig(obj_giv, ang0)
                g! = (G, x) -> ForwardDiff.gradient!(G, obj_giv, x, cfg)
                Optim.optimize(obj_giv, g!, ang0, Optim.LBFGS(),
                               Optim.Options(iterations=150, g_tol=T(1e-7)))
            catch
                nothing
            end
            res === nothing && continue
            val = T(Optim.minimum(res))
            if isfinite(val) && val < best_loss
                best_loss = val
                best_Q = _givens_to_orthogonal(Vector{T}(Optim.minimizer(res)), n)
            end
        end
    end

    B_raw = L * best_Q
    pg = _llgmm_pham_garat(B_raw)
    B_init = pg.B

    if form === :b_model
        return _llgmm_pack_theta(nu_ols, A_ols, B_init, nothing, :b_model)
    else
        a_conv = _llgmm_b0_to_a0(B_init)
        return _llgmm_pack_theta(nu_ols, A_ols, a_conv.A0, a_conv.sigma, :a_model)
    end
end

"""
Run `Optim.LBFGS` on a combined `(G, x) -> val` callback (fills `G` if non-`nothing`),
caching the most recent evaluation so `Optim.optimize(f, g!, ...)` avoids duplicate work
and does not depend on `NLSolversBase.only_fg!`.
"""
function _llgmm_lbfgs_fg(fg_eval!, x0::Vector{T}, iterations::Int, g_tol::T) where {T<:AbstractFloat}
    last_x = Vector{T}()
    last_val = Ref{T}(T(NaN))
    last_g = Vector{T}(undef, length(x0))
    function ensure!(x)
        xv = Vector{T}(x)
        if length(last_x) != length(xv) || xv != last_x
            fill!(last_g, zero(T))
            val = T(fg_eval!(last_g, xv))
            resize!(last_x, length(xv))
            copyto!(last_x, xv)
            last_val[] = val
        end
    end
    f_obj(x) = (ensure!(x); last_val[])
    g_obj!(G, x) = (ensure!(x); copyto!(G, last_g))
    Optim.optimize(
        f_obj, g_obj!, x0, Optim.LBFGS(),
        Optim.Options(iterations=iterations, g_tol=g_tol,
                      f_reltol=zero(T), allow_f_increases=true),
    )
end

function _llgmm_solve_fixed_W(theta0::Vector{T}, Y_eff::AbstractMatrix{T}, X::AbstractMatrix{T},
                              W::AbstractMatrix{T}, n::Int, p::Int, form::Symbol,
                              asym_pairs::Vector{Tuple{Int,Int}},
                              sym_pairs::Vector{Tuple{Int,Int}};
                              max_iter::Int=100, tol::T=T(1e-7)) where {T<:AbstractFloat}
    k = length(theta0)
    theta = copy(theta0)
    M, g_bar, G_jac, ok = _llgmm_jacobian_impl(theta, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
    ok || throw(IdentificationError("Initial GMM parameter vector yields singular B0/A0"))

    Q_curr = T(dot(g_bar, W * g_bar))
    lambda = T(1e-10)
    iters = 0
    converged = false
    gn_tol = min(tol, T === Float32 ? T(1e-5) : T(1e-9))

    for it in 1:max_iter
        iters += 1
        Wg = W * g_bar
        grad_half = transpose(G_jac) * Wg
        grad_inf = maximum(abs, grad_half) * T(2)
        if grad_inf <= gn_tol
            converged = true
            break
        end

        H = Matrix{T}(transpose(G_jac) * W * G_jac)
        H = Matrix{T}((H + transpose(H)) .* T(0.5))
        scale_diag = max.(diag(H), one(T))

        accepted = false
        for _ in 1:8
            H_reg = H + Diagonal(lambda .* scale_diag)
            step = -Vector{T}(robust_inv(H_reg; silent=true) * grad_half)
            if !all(isfinite, step)
                lambda *= T(10)
                continue
            end
            # Require descent direction
            if dot(grad_half, step) >= zero(T)
                step = -Vector{T}(grad_half)
            end

            alpha = one(T)
            for _ in 1:15
                theta_cand = theta .+ alpha .* step
                M_c, g_c, G_c, ok_c = _llgmm_jacobian_impl(theta_cand, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
                if ok_c
                    Q_cand = T(dot(g_c, W * g_c))
                    if isfinite(Q_cand) && Q_cand <= Q_curr + T(1e-14) * max(one(T), Q_curr)
                        dtheta = maximum(abs, alpha .* step)
                        theta = theta_cand
                        M, g_bar, G_jac = M_c, g_c, G_c
                        Q_curr = Q_cand
                        lambda = max(T(1e-12), lambda * T(0.3))
                        accepted = true
                        if dtheta <= gn_tol && maximum(abs, transpose(G_jac) * (W * g_bar)) * T(2) <= max(gn_tol * T(10), T(1e-7))
                            converged = true
                        end
                        break
                    end
                end
                alpha *= T(0.5)
            end
            accepted && break
            lambda *= T(10)
            lambda > T(1e8) && break
        end

        converged && break
        !accepted && break
    end

    # L-BFGS polish with exact analytical gradient if Gauss-Newton did not reach tolerance
    grad_inf_now = maximum(abs, transpose(G_jac) * (W * g_bar)) * T(2)
    if !converged && grad_inf_now > gn_tol
        function eval_fg_polish!(G, xv::Vector{T})
            _, g_p, G_p, ok_p = _llgmm_jacobian_impl(xv, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
            ok_p || return T(Inf)
            Wg_p = W * g_p
            val = T(dot(g_p, Wg_p))
            isfinite(val) || return T(Inf)
            if G !== nothing
                copyto!(G, T(2) .* (transpose(G_p) * Wg_p))
            end
            return val
        end
        res_opt = try
            _llgmm_lbfgs_fg(eval_fg_polish!, theta, max_iter, gn_tol)
        catch
            nothing
        end
        if res_opt !== nothing
            iters += Optim.iterations(res_opt)
            x_opt = Vector{T}(Optim.minimizer(res_opt))
            M_p, g_p, G_p, ok_p = _llgmm_jacobian_impl(x_opt, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
            if ok_p
                Q_p = T(dot(g_p, W * g_p))
                if isfinite(Q_p) && Q_p <= Q_curr
                    theta = x_opt
                    M, g_bar, G_jac = M_p, g_p, G_p
                    Q_curr = Q_p
                end
            end
            converged = converged || Optim.converged(res_opt)
        end
    end

    (theta=theta, M=M, g_bar=g_bar, G_jac=G_jac, Q=Q_curr, converged=converged, iterations=iters)
end

function _llgmm_cue_eval_fg!(G_out, theta::AbstractVector{T}, Y_eff::AbstractMatrix{T},
                             X::AbstractMatrix{T}, n::Int, p::Int, form::Symbol,
                             asym_pairs::Vector{Tuple{Int,Int}},
                             sym_pairs::Vector{Tuple{Int,Int}},
                             ki::Symbol, bw_fixed::Int) where {T<:AbstractFloat}
    T_eff = size(Y_eff, 1)
    M, g_bar, G_jac, ok = _llgmm_jacobian_impl(Vector{T}(theta), Y_eff, X, n, p, form, asym_pairs, sym_pairs)
    ok || return T(Inf)

    M_demean = M .- g_bar'
    S_raw = (bw_fixed == 0 && ki !== :quadratic_spectral) ?
        Matrix{T}((transpose(M_demean) * M_demean) ./ T(T_eff)) :
        _lrv_accumulate(M_demean, bw_fixed, ki)[1]
    S_psd = _psd_project(S_raw)
    W = _llgmm_inv_S(S_psd)
    v = W * g_bar
    Q_val = T(dot(g_bar, v))
    isfinite(Q_val) || return T(Inf)

    if G_out !== nothing
        # Exact analytical derivative of ḡ(θ)' S(θ)⁻¹ ḡ(θ):
        # 2 G(θ)' v  -  v' (∂S/∂θ) v
        # where v'(∂S/∂θ)v = (2/T) Σ_t (α_t - ᾱ) v' (∂g_t/∂θ'), so total is
        # (2/T) Σ_t [1 - (α_t - ᾱ)] v' (∂g_t/∂θ')
        d_t = M_demean * v
        alpha = copy(d_t)
        jmax = ki === :quadratic_spectral ? (T_eff - 1) : bw_fixed
        @inbounds for j in 1:jmax
            wj = kernel_weight(j, bw_fixed, ki, T)
            wj == zero(T) && continue
            for t in 1:(T_eff - j)
                alpha[t] += wj * d_t[t + j]
            end
            for t in (j + 1):T_eff
                alpha[t] += wj * d_t[t - j]
            end
        end
        alpha_bar = mean(alpha)
        omega = one(T) .- (alpha .- alpha_bar)
        _, _, G_weighted, ok_w = _llgmm_jacobian_impl(
            Vector{T}(theta), Y_eff, X, n, p, form, asym_pairs, sym_pairs;
            time_weights=omega,
        )
        ok_w || return T(Inf)
        grad = T(2) .* (transpose(G_weighted) * v)
        all(isfinite, grad) || return T(Inf)
        copyto!(G_out, grad)
    end
    return Q_val
end

function _llgmm_solve_cue(theta_init::Vector{T}, Y_eff::AbstractMatrix{T}, X::AbstractMatrix{T},
                          n::Int, p::Int, form::Symbol,
                          asym_pairs::Vector{Tuple{Int,Int}},
                          sym_pairs::Vector{Tuple{Int,Int}},
                          kernel::Symbol, bandwidth;
                          max_iter::Int=100, tol::T=T(1e-7)) where {T<:AbstractFloat}
    ki = _lrv_kernel(kernel)
    M0, _ = _llgmm_moment_matrix(theta_init, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
    M0_d = M0 .- mean(M0; dims=1)
    bw_curr = _llgmm_resolve_bw(M0_d, ki, bandwidth)
    theta_curr = copy(theta_init)
    total_iters = 0
    converged = false
    g_tol_eff = min(tol, T === Float32 ? T(1e-5) : T(1e-8))

    n_passes = (bandwidth isa Symbol) ? 3 : 1
    for _ in 1:n_passes
        bw_pass = bw_curr
        fg! = (G, x) -> _llgmm_cue_eval_fg!(G, x, Y_eff, X, n, p, form, asym_pairs, sym_pairs, ki, bw_pass)
        res = try
            _llgmm_lbfgs_fg(fg!, theta_curr, max(max_iter, 150), g_tol_eff)
        catch
            nothing
        end
        if res !== nothing
            total_iters += Optim.iterations(res)
            cand = Vector{T}(Optim.minimizer(res))
            _, _, _, _, _, ok_c = _llgmm_shocks_and_resids(cand, Y_eff, X, n, p, form)
            if ok_c && isfinite(Optim.minimum(res))
                theta_curr = cand
                converged = Optim.converged(res)
            end
        end
        bandwidth isa Symbol || break
        M_new, _ = _llgmm_moment_matrix(theta_curr, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
        bw_new = _llgmm_resolve_bw(M_new .- mean(M_new; dims=1), ki, bandwidth)
        bw_new == bw_curr && break
        bw_curr = bw_new
    end

    M_f, g_bar_f, G_jac_f, _ = _llgmm_jacobian_impl(theta_curr, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
    S_f, bw_f = _llgmm_hac_cov(M_f, kernel, bandwidth)
    W_f = _llgmm_inv_S(S_f)
    Q_f = T(dot(g_bar_f, W_f * g_bar_f))
    (theta=theta_curr, M=M_f, g_bar=g_bar_f, G_jac=G_jac_f, S_hac=S_f, W=W_f, bw=bw_f,
     Q=Q_f, converged=converged, iterations=total_iters)
end

# =============================================================================
# Core Estimation Pipeline (Shared by Public Estimator & Moment Selector)
# =============================================================================

function _llgmm_var_for_Y(Ym::AbstractMatrix{T}, p::Int;
                          check_stability::Bool=true,
                          varnames::Union{Nothing,Vector{String}}=nothing) where {T<:AbstractFloat}
    _validate_data(Ym, "Y")
    T_obs, n = size(Ym)
    p < 0 && throw(ArgumentError("p must be >= 0, got p=$p"))
    if p == 0
        T_obs >= 4 || throw(ArgumentError("Need at least 4 observations for p=0, got $T_obs"))
        mu = mean(Ym; dims=1)
        B = Matrix{T}(mu)
        U = Ym .- mu
        Sigma = Matrix{T}((transpose(U) * U) ./ T(T_obs))
        log_det = logdet_safe(Sigma)
        k_sys = n
        aic = log_det + 2 * k_sys / T(T_obs)
        bic = log_det + k_sys * log(T(T_obs)) / T(T_obs)
        hqic = log_det + 2 * k_sys * log(log(T(T_obs))) / T(T_obs)
        vn = something(varnames, ["y$i" for i in 1:n])
        return VARModel(Matrix{T}(Ym), 0, B, U, Sigma, T(aic), T(bic), T(hqic), vn)
    else
        return estimate_var(Ym, p; check_stability=check_stability, varnames=varnames)
    end
end

function _llgmm_construct_matrices(model::VARModel{T}) where {T<:AbstractFloat}
    if model.p == 0
        T_obs = size(model.Y, 1)
        return Matrix{T}(model.Y), ones(T, T_obs, 1)
    else
        return construct_var_matrices(model.Y, model.p)
    end
end

function _llgmm_estimate_core(model::VARModel{T}, Y_eff::AbstractMatrix{T}, X::AbstractMatrix{T},
                              asym_pairs::Vector{Tuple{Int,Int}},
                              sym_pairs::Vector{Tuple{Int,Int}};
                              form::Symbol=:b_model,
                              weighting::Symbol=:two_step,
                              kernel::Symbol=:bartlett,
                              bandwidth=:auto,
                              max_iter::Int=100,
                              tol::T=T(1e-7),
                              n_starts::Int=5,
                              rng::AbstractRNG=Random.default_rng(),
                              theta_warm::Union{Nothing,Vector{T}}=nothing) where {T<:AbstractFloat}
    n = nvars(model)
    p = model.p
    T_eff = size(Y_eff, 1)
    k = _llgmm_n_params(n, p)
    q = _llgmm_n_base_moments(n, p) + length(asym_pairs) + length(sym_pairs)
    q >= k || throw(ArgumentError(
        "Under-identified GMM: $q moment conditions < $k parameters. " *
        "Provide at least $( (n * (n - 1)) ÷ 2 ) co-kurtosis pairs."))

    tol_eff = T === Float32 ? max(tol, T(1e-5)) : tol

    theta0 = if theta_warm !== nothing
        copy(theta_warm)
    else
        _llgmm_init_theta(model, Y_eff, X, form, asym_pairs, sym_pairs, n_starts, rng)
    end

    # Step 1: Initial weighting matrix evaluated at consistent VAR + co-kurtosis initializer θ₀
    # (scales 2nd- and 4th-moment blocks properly before the two-step efficient update)
    M0_init, ok0 = _llgmm_moment_matrix(theta0, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
    W1 = if ok0
        S0_init, _ = _llgmm_hac_cov(M0_init, kernel, bandwidth)
        _llgmm_inv_S(S0_init)
    else
        Matrix{T}(I, q, q)
    end
    step1 = _llgmm_solve_fixed_W(theta0, Y_eff, X, W1, n, p, form, asym_pairs, sym_pairs;
                                 max_iter=max_iter, tol=tol_eff)
    total_iters = step1.iterations

    theta_hat = step1.theta
    M_f = step1.M
    g_bar_f = step1.g_bar
    G_jac_f = step1.G_jac
    S_f, bw_f = _llgmm_hac_cov(M_f, kernel, bandwidth)
    W_f = _llgmm_inv_S(S_f)
    converged = step1.converged

    if weighting === :two_step
        step2 = _llgmm_solve_fixed_W(step1.theta, Y_eff, X, W_f, n, p, form, asym_pairs, sym_pairs;
                                     max_iter=max_iter, tol=tol_eff)
        total_iters += step2.iterations
        theta_hat = step2.theta
        M_f = step2.M
        g_bar_f = step2.g_bar
        G_jac_f = step2.G_jac
        S_f, bw_f = _llgmm_hac_cov(M_f, kernel, bandwidth)
        converged = step2.converged
    elseif weighting === :iterated
        theta_curr = step1.theta
        W_curr = W_f
        for _ in 1:max_iter
            step_it = _llgmm_solve_fixed_W(theta_curr, Y_eff, X, W_curr, n, p, form, asym_pairs, sym_pairs;
                                           max_iter=max_iter, tol=tol_eff)
            total_iters += step_it.iterations
            diff = maximum(abs, step_it.theta - theta_curr)
            theta_curr = step_it.theta
            M_f = step_it.M
            g_bar_f = step_it.g_bar
            G_jac_f = step_it.G_jac
            S_f, bw_f = _llgmm_hac_cov(M_f, kernel, bandwidth)
            W_curr = _llgmm_inv_S(S_f)
            converged = step_it.converged
            diff <= tol_eff && break
        end
        theta_hat = theta_curr
        W_f = W_curr
    elseif weighting === :cue
        # Warm-start CUE from 2-step consistent estimate for stability
        step2_w = _llgmm_solve_fixed_W(step1.theta, Y_eff, X, W_f, n, p, form, asym_pairs, sym_pairs;
                                       max_iter=max_iter, tol=tol_eff)
        total_iters += step2_w.iterations
        cue_res = _llgmm_solve_cue(step2_w.theta, Y_eff, X, n, p, form, asym_pairs, sym_pairs,
                                   kernel, bandwidth; max_iter=max_iter, tol=tol_eff)
        total_iters += cue_res.iterations
        theta_hat = cue_res.theta
        M_f = cue_res.M
        g_bar_f = cue_res.g_bar
        G_jac_f = cue_res.G_jac
        S_f = cue_res.S_hac
        W_f = cue_res.W
        bw_f = cue_res.bw
        converged = cue_res.converged
    end

    # Extract structural matrices and apply Pham-Garat (1997) canonical normalization
    Pi_T, B_raw, A0_raw, sig_raw = _llgmm_unpack_theta(theta_hat, n, p, form)
    B0_pre = form === :b_model ? Matrix{T}(B_raw) : Matrix{T}(robust_inv(A0_raw; silent=true) * Diagonal(sig_raw))
    pg = _llgmm_pham_garat(B0_pre)
    asym_out = copy(asym_pairs)
    sym_out = copy(sym_pairs)

    if pg.perm != collect(1:n) || any(!=(1), pg.signs)
        asym_out, sym_out = _llgmm_remap_pairs(asym_pairs, sym_pairs, pg.perm)
        nu_cur = Vector{T}(@view(Pi_T[:, 1]))
        A_cur = Matrix{T}[Matrix{T}(@view(Pi_T[:, (2 + (l - 1) * n):(1 + l * n)])) for l in 1:p]
        if form === :b_model
            theta_hat = _llgmm_pack_theta(nu_cur, A_cur, pg.B, nothing, :b_model)
        else
            a_conv = _llgmm_b0_to_a0(pg.B)
            theta_hat = _llgmm_pack_theta(nu_cur, A_cur, a_conv.A0, a_conv.sigma, :a_model)
        end
        # Permute W_f rows/cols if weighting was :two_step (or recompute S_f, W_f at normalized theta_hat)
        M_f, g_bar_f, G_jac_f, _ = _llgmm_jacobian_impl(theta_hat, Y_eff, X, n, p, form, asym_out, sym_out)
        S_f, bw_f = _llgmm_hac_cov(M_f, kernel, bandwidth)
        W_f = _llgmm_inv_S(S_f)
    end

    # Recompute final outputs at canonical normalized theta_hat
    E_f, _, _, _, _, _ = _llgmm_shocks_and_resids(theta_hat, Y_eff, X, n, p, form)
    Pi_Tf, Bf_u, A0f_u, sigf_u = _llgmm_unpack_theta(theta_hat, n, p, form)

    B0 = form === :b_model ? Matrix{T}(Bf_u) : Matrix{T}(robust_inv(A0f_u; silent=true) * Diagonal(sigf_u))
    A0 = form === :b_model ? Matrix{T}(robust_inv(B0; silent=true)) : Matrix{T}(A0f_u)
    sigma_vec = form === :b_model ? Vector{T}(abs.(diag(B0))) : Vector{T}(sigf_u)
    nu_vec = Vector{T}(@view(Pi_Tf[:, 1]))
    A_vec = Matrix{T}[Matrix{T}(@view(Pi_Tf[:, (2 + (l - 1) * n):(1 + l * n)])) for l in 1:p]
    Pi_mat = Matrix{T}(transpose(Pi_Tf))
    Sigma_imp = Matrix{T}((B0 * transpose(B0) + transpose(B0 * transpose(B0))) .* T(0.5))
    L_imp = Matrix{T}(safe_cholesky(Sigma_imp; silent=true))
    Q_raw = L_imp \ B0
    F_pol = svd(Q_raw)
    Q_mat = Matrix{T}(F_pol.U * F_pol.Vt)

    # Efficient asymptotic variance V_theta / T_eff = (G' W G)^{-1} / T_eff
    # and rank check (LL Prop. 1)
    jac_rank = _llgmm_jacobian_rank(G_jac_f)
    rank_ok = jac_rank == k

    info_mat = Matrix{T}(transpose(G_jac_f) * W_f * G_jac_f)
    info_sym = Matrix{T}((info_mat + transpose(info_mat)) .* T(0.5))
    V_asymp = _spd_inv(info_sym)  # (G' W G)^{-1}
    V_theta = Matrix{T}((V_asymp .+ transpose(V_asymp)) .* (T(0.5) / T(T_eff)))
    se_theta = sqrt.(max.(diag(V_theta), zero(T)))

    n_pi = n * (1 + n * p)
    se_B0 = if form === :b_model
        reshape(se_theta[(n_pi + 1):end], n, n)
    else
        V_bs = Matrix{T}(@view(V_theta[(n_pi + 1):end, (n_pi + 1):end]))
        n_off = n * (n - 1)
        B0_fn = bs -> robust_inv(_llgmm_unit_diag_from_off(@view(bs[1:n_off]), n); silent=true) *
                      Diagonal(@view(bs[(n_off + 1):(n_off + n)]))
        se_b0, _ = _delta_B0_se(Vector{T}(@view(theta_hat[(n_pi + 1):end])), V_bs, B0_fn, n)
        se_b0
    end

    df_overid = q - k
    J_stat = df_overid > 0 ? max(zero(T), T(T_eff) * T(dot(g_bar_f, W_f * g_bar_f))) : zero(T)
    J_pval = df_overid > 0 ? T(ccdf(Chisq(df_overid), Float64(J_stat))) : one(T)

    return (
        B0=B0, A0=A0, Q=Q_mat, sigma=sigma_vec, nu=nu_vec, A=A_vec, Pi=Pi_mat, Sigma=Sigma_imp,
        shocks=Matrix{T}(E_f), theta=theta_hat, vcov=V_theta, V_asymp=V_asymp,
        se_B0=se_B0, se_theta=se_theta, W=W_f, S_hac=S_f, G_jac=G_jac_f, g_bar=g_bar_f,
        J_stat=J_stat, J_pvalue=J_pval, df=df_overid, n_moments=q, n_params=k,
        jacobian_rank=jac_rank, rank_ok=rank_ok, asym_pairs=asym_out, sym_pairs=sym_out,
        bw=bw_f, converged=converged, iterations=total_iters,
    )
end

# =============================================================================
# Two-Stage Moment Selection: Andrews (1999) MSC + Hall et al. (2007) RMSC
# =============================================================================

function _llgmm_rmsc_penalty(T_eff::Int, b_T::Int, df::Int, ::Type{T}) where {T<:AbstractFloat}
    df <= 0 && return zero(T)
    ratio = T(T_eff) / T(max(b_T, 1))
    T(df) * (T(0.5) * log(ratio)) / sqrt(ratio)
end

function _llgmm_logdet_V(V_asymp::AbstractMatrix{T}) where {T<:AbstractFloat}
    Vs = Matrix{T}((V_asymp + transpose(V_asymp)) .* T(0.5))
    L = try
        cholesky(Hermitian(Vs)).L
    catch
        return T(Inf)
    end
    d = diag(L)
    any(x -> !isfinite(x) || x <= zero(T), d) && return T(Inf)
    T(2) * sum(log, d)
end

function _llgmm_enumerate_stage1_subsets(n::Int)
    pairs = Tuple{Int,Int}[(i, j) for i in 1:(n - 1) for j in (i + 1):n]
    m = length(pairs)
    n_req = m  # n*(n-1)/2
    # Each unordered pair (i,j) has 3 asymmetric choices: [(i,j)], [(j,i)], or both
    out = Tuple{Vector{Tuple{Int,Int}}, Vector{Tuple{Int,Int}}}[]
    for a_mask in Iterators.product(ntuple(_ -> 1:3, m)...)
        asym = Tuple{Int,Int}[]
        for (idx, ch) in enumerate(a_mask)
            if ch == 1
                push!(asym, (pairs[idx][1], pairs[idx][2]))
            elseif ch == 2
                push!(asym, (pairs[idx][2], pairs[idx][1]))
            else
                push!(asym, (pairs[idx][1], pairs[idx][2]))
                push!(asym, (pairs[idx][2], pairs[idx][1]))
            end
        end
        for s_bits in 0:((1 << m) - 1)
            sym = Tuple{Int,Int}[]
            for idx in 1:m
                ((s_bits >> (idx - 1)) & 1 == 1) && push!(sym, pairs[idx])
            end
            # Over-identified strictly: q > k <=> length(asym) + length(sym) > n_req
            (length(asym) + length(sym) > n_req) && push!(out, (asym, sym))
        end
    end
    out
end

function _llgmm_enumerate_stage2_subsets(asym_msc::Vector{Tuple{Int,Int}},
                                         sym_msc::Vector{Tuple{Int,Int}},
                                         n::Int)
    # Build all subsets of (asym_msc, sym_msc) covering every unordered pair {i, j}
    pairs = Tuple{Int,Int}[(i, j) for i in 1:(n - 1) for j in (i + 1):n]
    m = length(pairs)
    asym_set = Set(asym_msc)
    per_pair_choices = Vector{Vector{Vector{Tuple{Int,Int}}}}(undef, m)
    for (idx, (i, j)) in enumerate(pairs)
        chs = Vector{Tuple{Int,Int}}[]
        has_ij = (i, j) in asym_set
        has_ji = (j, i) in asym_set
        has_ij && push!(chs, [(i, j)])
        has_ji && push!(chs, [(j, i)])
        has_ij && has_ji && push!(chs, [(i, j), (j, i)])
        per_pair_choices[idx] = chs
    end
    n_sym = length(sym_msc)
    out = Tuple{Vector{Tuple{Int,Int}}, Vector{Tuple{Int,Int}}}[]
    for a_tuple in Iterators.product(per_pair_choices...)
        asym = Tuple{Int,Int}[]
        for ch in a_tuple
            append!(asym, ch)
        end
        for s_bits in 0:((1 << n_sym) - 1)
            sym = Tuple{Int,Int}[]
            for idx in 1:n_sym
                ((s_bits >> (idx - 1)) & 1 == 1) && push!(sym, sym_msc[idx])
            end
            push!(out, (asym, sym))
        end
    end
    out
end

"""Removable single conditions from `(asym, sym)` preserving all-pair coverage and optional `min_ck`."""
function _llgmm_removable_neighbors(asym::Vector{Tuple{Int,Int}},
                                    sym::Vector{Tuple{Int,Int}},
                                    min_ck::Int)
    n_tot = length(asym) + length(sym)
    n_tot <= min_ck && return Tuple{Vector{Tuple{Int,Int}}, Vector{Tuple{Int,Int}}}[]
    asym_set = Set(asym)
    neighbors = Tuple{Vector{Tuple{Int,Int}}, Vector{Tuple{Int,Int}}}[]
    # Any symmetric condition can always be removed
    for idx in eachindex(sym)
        sym_new = deleteat!(copy(sym), idx)
        push!(neighbors, (copy(asym), sym_new))
    end
    # An asymmetric condition (i, j) can be removed if (j, i) is also present
    for idx in eachindex(asym)
        (i, j) = asym[idx]
        if (j, i) in asym_set
            asym_new = deleteat!(copy(asym), idx)
            push!(neighbors, (asym_new, copy(sym)))
        end
    end
    neighbors
end

"""
    select_gmm_moments(model::VARModel; form=:b_model, weighting=:two_step,
                       method=:msc_rmsc, kernel=:bartlett, bandwidth=:auto,
                       alpha=0.05, max_iter=100, tol=1e-7, n_starts=5,
                       rng=Random.default_rng(), seed=nothing) -> GMMMomentSelectionResult
    select_gmm_moments(Y::AbstractMatrix, p=nothing; ...) -> GMMMomentSelectionResult

Select the optimal, non-redundant co-kurtosis moment set for Lanne–Luoto (2021)
non-Gaussian SVAR GMM using the two-stage Andrews (1999) MSC and Hall, Inoue,
Jana & Shin (2007) entropy-based RMSC procedure (§3.3, eqs. 13–14).

Stage 1 minimizes Andrews' Moment Selection Criterion:

```math
\\mathrm{MSC}(c) = J_T(c) - (q - k) \\ln(T)
```

where:
- ``J_T(c)`` is Hansen's overidentifying restrictions test statistic for candidate set ``c``
- ``q`` is the total number of moment conditions in candidate set ``c``
- ``k = n + (p + 1)n^2`` is the total number of estimated SVAR parameters
- ``T`` is the effective sample size

Stage 2 minimizes Hall et al.'s Relevant Moment Selection Criterion among subsets
of the MSC-selected set not rejected by ``J_T`` at significance level `alpha`:

```math
\\mathrm{RMSC}(c) = \\ln \\left| \\hat{V}_{\\theta,T}(c) \\right| + (q - k) \\ln\\!\\left[ \\left(\\frac{T}{b_T}\\right)^{1/2} \\right] \\left(\\frac{T}{b_T}\\right)^{-1/2}
```

where:
- ``\\hat{V}_{\\theta,T}(c) = (G_{T,c}' \\hat{S}_{T,c}^{-1} G_{T,c})^{-1}`` is the asymptotic covariance estimator
- ``q - k`` is the number of overidentifying restrictions
- ``T`` is the effective sample size
- ``b_T`` is the Newey–West HAC bandwidth

For ``n \\ge 4``, `method=:msc_rmsc` automatically switches to Lanne & Luoto's (2021, §3.3)
tractable sequential backward-elimination procedure.
"""
function select_gmm_moments(model::VARModel{T};
                            form::Symbol=:b_model,
                            weighting::Symbol=:two_step,
                            method::Symbol=:msc_rmsc,
                            kernel::Symbol=:bartlett,
                            bandwidth=:auto,
                            alpha::Real=0.05,
                            max_iter::Int=100,
                            tol::Real=1e-7,
                            n_starts::Int=5,
                            rng::AbstractRNG=Random.default_rng(),
                            seed::Union{Integer,Nothing}=nothing,
                            fixed_sym_pairs::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing,
                            fixed_asym_pairs::Union{Nothing,Vector{Tuple{Int,Int}}}=nothing) where {T<:AbstractFloat}
    form in (:b_model, :a_model) || throw(ArgumentError("form must be :b_model or :a_model, got :$form"))
    weighting in (:two_step, :iterated, :cue) || throw(ArgumentError(
        "weighting must be :two_step, :iterated, or :cue, got :$weighting"))
    method in (:msc_rmsc, :rmsc_seq) || throw(ArgumentError(
        "method must be :msc_rmsc or :rmsc_seq, got :$method"))
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0, 1), got $alpha"))

    n = nvars(model)
    n >= 2 || throw(ArgumentError("Lanne–Luoto GMM moment selection requires n ≥ 2 variables"))
    p = model.p
    Y_eff, X = _llgmm_construct_matrices(model)
    T_eff = size(Y_eff, 1)
    k = _llgmm_n_params(n, p)
    rng_eff = _resolve_repro_rng(rng, seed)
    tol_T = T(tol)
    sel_weighting = weighting === :cue ? :two_step : weighting

    all_asym = fixed_asym_pairs === nothing ? _llgmm_all_asym_pairs(n) : fixed_asym_pairs
    all_sym = fixed_sym_pairs === nothing ? _llgmm_all_sym_pairs(n) : fixed_sym_pairs

    # Anchor warm-start on all asymmetric moments (consistent even under common SV)
    anchor_sym = fixed_sym_pairs === nothing ? Tuple{Int,Int}[] : fixed_sym_pairs
    anchor = _llgmm_estimate_core(model, Y_eff, X, all_asym, anchor_sym;
                                  form=form, weighting=:two_step, kernel=kernel,
                                  bandwidth=bandwidth, max_iter=max_iter, tol=tol_T,
                                  n_starts=n_starts, rng=rng_eff)
    theta_anchor = anchor.theta

    NT = NamedTuple{(:asym_pairs, :sym_pairs, :n_moments, :df, :J_stat, :J_pvalue, :msc, :rmsc, :rank_ok),
                    Tuple{Vector{Tuple{Int,Int}}, Vector{Tuple{Int,Int}}, Int, Int, T, T, T, T, Bool}}
    eval_cache = Dict{Tuple{Vector{Tuple{Int,Int}}, Vector{Tuple{Int,Int}}}, Any}()

    function _eval_cand(asym::Vector{Tuple{Int,Int}}, sym::Vector{Tuple{Int,Int}})
        key = (sort(asym), sort(sym))
        if haskey(eval_cache, key)
            return eval_cache[key]
        end
        est = _llgmm_estimate_core(model, Y_eff, X, asym, sym;
                                   form=form, weighting=sel_weighting, kernel=kernel,
                                   bandwidth=bandwidth, max_iter=max_iter, tol=tol_T,
                                   n_starts=1, rng=rng_eff, theta_warm=theta_anchor)
        msc = est.df > 0 ? est.J_stat - T(est.df) * log(T(T_eff)) : zero(T)
        out = (est=est, msc=msc, asym=asym, sym=sym)
        eval_cache[key] = out
        return out
    end

    use_exhaustive = (n <= 3) && (method === :msc_rmsc) && (fixed_asym_pairs === nothing) && (fixed_sym_pairs === nothing)
    n_pairs_req = (n * (n - 1)) ÷ 2

    # --- Stage 1: Andrews (1999) MSC ---
    msc_asym = copy(all_asym)
    msc_sym = copy(all_sym)
    msc_best_eval = nothing

    if method === :msc_rmsc
        if use_exhaustive
            s1_subsets = _llgmm_enumerate_stage1_subsets(n)
            best_msc = T(Inf)
            for (asym_c, sym_c) in s1_subsets
                ev = _eval_cand(asym_c, sym_c)
                if ev.est.rank_ok && ev.msc < best_msc
                    best_msc = ev.msc
                    msc_best_eval = ev
                end
            end
            if msc_best_eval === nothing
                msc_best_eval = _eval_cand(all_asym, Tuple{Int,Int}[])
            end
            msc_asym = msc_best_eval.asym
            msc_sym = msc_best_eval.sym
        else
            # Tractable sequential backward elimination for n >= 4 (or fixed subset restrictions)
            cand_full = _eval_cand(all_asym, all_sym)
            cand_asym = (length(all_asym) > n_pairs_req) ? _eval_cand(all_asym, anchor_sym) : cand_full
            cur = (cand_asym.msc < cand_full.msc && cand_asym.est.rank_ok) ? cand_asym : cand_full
            while true
                best_nb = nothing
                best_nb_msc = cur.msc
                for (a_nb, s_nb) in _llgmm_removable_neighbors(cur.asym, cur.sym, n_pairs_req + 1)
                    fixed_asym_pairs !== nothing && a_nb != fixed_asym_pairs && continue
                    fixed_sym_pairs !== nothing && s_nb != fixed_sym_pairs && continue
                    ev = _eval_cand(a_nb, s_nb)
                    if ev.est.rank_ok && ev.msc < best_nb_msc - T(1e-8)
                        best_nb_msc = ev.msc
                        best_nb = ev
                    end
                end
                best_nb === nothing && break
                cur = best_nb
            end
            msc_best_eval = cur
            msc_asym = cur.asym
            msc_sym = cur.sym
        end
    else
        # :rmsc_seq starts from full set (pruning invalid symmetric conditions if rejected by J)
        cand_full = _eval_cand(all_asym, all_sym)
        cand_asym = (length(all_asym) > n_pairs_req) ? _eval_cand(all_asym, anchor_sym) : cand_full
        cur = cand_full
        if cur.est.J_pvalue < T(alpha) && cand_asym.est.J_pvalue >= T(alpha)
            cur = cand_asym
        end
        while cur.est.J_pvalue < T(alpha)
            best_nb = nothing
            best_J = cur.est.J_stat
            for (a_nb, s_nb) in _llgmm_removable_neighbors(cur.asym, cur.sym, n_pairs_req + 1)
                fixed_asym_pairs !== nothing && a_nb != fixed_asym_pairs && continue
                fixed_sym_pairs !== nothing && s_nb != fixed_sym_pairs && continue
                ev = _eval_cand(a_nb, s_nb)
                if ev.est.rank_ok && ev.est.J_stat < best_J
                    best_J = ev.est.J_stat
                    best_nb = ev
                end
            end
            best_nb === nothing && break
            cur = best_nb
        end
        msc_best_eval = cur
        msc_asym = cur.asym
        msc_sym = cur.sym
    end

    # Common HAC bandwidth b_T from Stage-1 MSC fit for Hall et al. (2007) RMSC
    b_T = clamp(msc_best_eval.est.bw, 1, max(1, T_eff ÷ 2))

    # --- Stage 2: Hall et al. (2007) Entropy-Based RMSC ---
    final_eval = msc_best_eval
    best_rmsc = _llgmm_logdet_V(msc_best_eval.est.V_asymp) +
                _llgmm_rmsc_penalty(T_eff, b_T, msc_best_eval.est.df, T)

    if use_exhaustive
        s2_subsets = _llgmm_enumerate_stage2_subsets(msc_asym, msc_sym, n)
        best_rmsc_valid = T(Inf)
        best_ev_valid = nothing
        for (asym_c, sym_c) in s2_subsets
            ev = _eval_cand(asym_c, sym_c)
            (ev.est.rank_ok && ev.est.J_pvalue >= T(alpha)) || continue
            ldv = _llgmm_logdet_V(ev.est.V_asymp)
            rmsc = ldv + _llgmm_rmsc_penalty(T_eff, b_T, ev.est.df, T)
            if isfinite(rmsc) && rmsc < best_rmsc_valid
                best_rmsc_valid = rmsc
                best_ev_valid = ev
            end
        end
        if best_ev_valid !== nothing
            final_eval = best_ev_valid
            best_rmsc = best_rmsc_valid
        end
    else
        # Sequential backward-elimination RMSC (LL §3.3, p. 6; automatic for n >= 4)
        cur = msc_best_eval
        cur_rmsc = _llgmm_logdet_V(cur.est.V_asymp) + _llgmm_rmsc_penalty(T_eff, b_T, cur.est.df, T)
        while true
            best_nb = nothing
            best_nb_rmsc = cur_rmsc
            for (a_nb, s_nb) in _llgmm_removable_neighbors(cur.asym, cur.sym, n_pairs_req)
                fixed_asym_pairs !== nothing && a_nb != fixed_asym_pairs && continue
                fixed_sym_pairs !== nothing && s_nb != fixed_sym_pairs && continue
                ev = _eval_cand(a_nb, s_nb)
                (ev.est.rank_ok && ev.est.J_pvalue >= T(alpha)) || continue
                ldv = _llgmm_logdet_V(ev.est.V_asymp)
                rmsc = ldv + _llgmm_rmsc_penalty(T_eff, b_T, ev.est.df, T)
                if isfinite(rmsc) && rmsc < best_nb_rmsc - T(1e-8)
                    best_nb_rmsc = rmsc
                    best_nb = ev
                end
            end
            best_nb === nothing && break
            cur = best_nb
            cur_rmsc = best_nb_rmsc
        end
        final_eval = cur
        best_rmsc = cur_rmsc
    end

    # Build candidate summary table
    cand_list = NT[]
    for (_, ev) in sort!(collect(eval_cache); by=kv -> (length(kv[1][1]) + length(kv[1][2]), kv[1][1], kv[1][2]))
        ldv = _llgmm_logdet_V(ev.est.V_asymp)
        rmsc_c = ldv + _llgmm_rmsc_penalty(T_eff, b_T, ev.est.df, T)
        push!(cand_list, (
            asym_pairs = ev.asym,
            sym_pairs  = ev.sym,
            n_moments  = ev.est.n_moments,
            df         = ev.est.df,
            J_stat     = ev.est.J_stat,
            J_pvalue   = ev.est.J_pvalue,
            msc        = ev.msc,
            rmsc       = rmsc_c,
            rank_ok    = ev.est.rank_ok,
        ))
    end

    reported_method = (n >= 4 && method === :msc_rmsc) ? :rmsc_seq : method

    GMMMomentSelectionResult{T}(
        final_eval.asym,
        final_eval.sym,
        msc_asym,
        msc_sym,
        msc_best_eval.msc,
        best_rmsc,
        final_eval.est.J_stat,
        final_eval.est.J_pvalue,
        final_eval.est.df,
        final_eval.est.n_moments,
        k,
        b_T,
        reported_method,
        cand_list,
    )
end

function select_gmm_moments(Y::AbstractMatrix, p::Union{Int,Nothing}=nothing;
                            check_stability::Bool=true,
                            varnames::Union{Nothing,Vector{String}}=nothing,
                            kwargs...)
    Ym = eltype(Y) <: AbstractFloat ? Y : Float64.(Y)
    T_obs, n = size(Ym)
    p_use = if p === nothing
        max_p = min(8, max(1, (T_obs - 10) ÷ max(3 * n, 1)))
        max_p >= 1 && T_obs > max_p + 2 ? select_lag_order(Ym, max_p; criterion=:bic) : 1
    else
        Int(p)
    end
    model = _llgmm_var_for_Y(Ym, p_use; check_stability=check_stability, varnames=varnames)
    select_gmm_moments(model; kwargs...)
end

# =============================================================================
# Public Estimation Entry Point
# =============================================================================

"""
    estimate_lanne_luoto_gmm(model::VARModel; form=:b_model, weighting=:two_step,
                             asym_pairs=:auto, sym_pairs=:auto,
                             moment_selection=:msc_rmsc, kernel=:bartlett,
                             bandwidth=:auto, max_iter=100, tol=1e-7,
                             n_starts=5, shock_names=nothing,
                             rng=Random.default_rng(), seed=nothing) -> LanneLuotoGMMResult
    estimate_lanne_luoto_gmm(Y::AbstractMatrix, p::Union{Int,Nothing}=nothing; ...) -> LanneLuotoGMMResult

Estimate a full-system non-Gaussian structural VAR(``p``) by Generalized Method of
Moments (Lanne & Luoto 2021).

Jointly estimates all ``k = n + (p + 1)n^2`` reduced-form and structural parameters
``\\theta = (\\nu', \\operatorname{vec}(A_1)', \\dots, \\operatorname{vec}(A_p)', \\operatorname{vec}(B)')'``
from the base orthogonality/covariance moments and selected fourth-order co-kurtosis
conditions:

```math
E\\left[\\varepsilon_t \\otimes x_{t-1}\\right] = 0, \\quad
E\\left[\\varepsilon_{it}^2 - 1\\right] = 0, \\quad
E\\left[\\varepsilon_{it} \\varepsilon_{jt}\\right] = 0 \\; (i < j), \\quad
E\\left[\\varepsilon_{it}^3 \\varepsilon_{jt}\\right] = 0 \\; (i \\ne j), \\quad
E\\left[\\varepsilon_{it}^2 \\varepsilon_{jt}^2 - 1\\right] = 0 \\; (i < j)
```

where:
- ``\\varepsilon_t = B^{-1} u_t`` (or ``\\operatorname{diag}(\\sigma)^{-1} A_0 u_t`` with ``A_{0,ii} = 1`` in `:a_model`) is the ``n \\times 1`` vector of structural shocks
- ``x_{t-1} = (1, y_{t-1}', \\dots, y_{t-p}')'`` is the ``(1 + np) \\times 1`` vector of VAR regressors
- ``n`` is the number of endogenous variables and ``p`` is the lag order

Supports `:two_step`, `:iterated`, and genuine `:cue` (Continuous Updating Estimator
minimizing ``\\bar{g}_T(\\theta)' \\hat{S}_T(\\theta)^{-1} \\bar{g}_T(\\theta)`` directly),
Newey–West HAC weighting with automatic bandwidth, analytical Jacobian rank verification
(Proposition 1), Pham–Garat (1997) column normalization, and two-stage Andrews MSC +
Hall et al. RMSC moment selection via [`select_gmm_moments`](@ref).

# Keyword Arguments
- `form::Symbol=:b_model` — `:b_model` (``u_t = B \\varepsilon_t``) or `:a_model` (``A_0 u_t = \\operatorname{diag}(\\sigma) \\varepsilon_t``, ``A_{0,ii} = 1``)
- `weighting::Symbol=:two_step` — `:two_step`, `:iterated`, or `:cue`
- `asym_pairs=:auto` — asymmetric co-kurtosis pairs `Vector{Tuple{Int,Int}}` or `:auto`
- `sym_pairs=:auto` — symmetric co-kurtosis pairs `Vector{Tuple{Int,Int}}` or `:auto`
- `moment_selection::Symbol=:msc_rmsc` — `:msc_rmsc`, `:rmsc_seq`, or `:none` (used when pairs are `:auto`)
- `kernel::Symbol=:bartlett` — HAC kernel (`:bartlett`, `:parzen`, `:qs`/`:quadratic_spectral`, `:tukey_hanning`)
- `bandwidth=:auto` — HAC lag truncation (`:auto`, `:nw94`, `:andrews`, or non-negative `Int`)
- `max_iter::Int=100` — maximum optimizer / iteration steps
- `tol::Real=1e-7` — convergence tolerance
- `n_starts::Int=5` — multi-starts for initial Givens co-kurtosis basin search
- `shock_names::Union{Nothing,Vector{String}}=nothing` — optional structural shock labels
- `rng::AbstractRNG=Random.default_rng()` — random number generator for initial starts
- `seed::Union{Integer,Nothing}=nothing` — optional reproducibility seed
"""
function estimate_lanne_luoto_gmm(model::VARModel{T};
                                  form::Symbol=:b_model,
                                  weighting::Symbol=:two_step,
                                  asym_pairs=:auto,
                                  sym_pairs=:auto,
                                  moment_selection::Symbol=:msc_rmsc,
                                  kernel::Symbol=:bartlett,
                                  bandwidth=:auto,
                                  max_iter::Int=100,
                                  tol::Real=1e-7,
                                  n_starts::Int=5,
                                  shock_names::Union{Nothing,Vector{String}}=nothing,
                                  rng::AbstractRNG=Random.default_rng(),
                                  seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    form in (:b_model, :a_model) || throw(ArgumentError("form must be :b_model or :a_model, got :$form"))
    weighting in (:two_step, :iterated, :cue) || throw(ArgumentError(
        "weighting must be :two_step, :iterated, or :cue, got :$weighting"))
    moment_selection in (:msc_rmsc, :rmsc_seq, :none) || throw(ArgumentError(
        "moment_selection must be :msc_rmsc, :rmsc_seq, or :none, got :$moment_selection"))
    ki = _lrv_kernel(kernel)
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    tol > 0 || throw(ArgumentError("tol must be > 0, got $tol"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be ≥ 1, got $n_starts"))

    n = nvars(model)
    n >= 2 || throw(ArgumentError("Lanne–Luoto GMM requires n ≥ 2 variables"))
    p = model.p
    snames = if shock_names !== nothing
        length(shock_names) == n || throw(ArgumentError(
            "shock_names length ($(length(shock_names))) must match n ($n)"))
        Vector{String}(shock_names)
    else
        _default_shock_names(n)
    end

    rng_eff = _resolve_repro_rng(rng, seed)
    tol_T = T(tol)
    Y_eff, X = _llgmm_construct_matrices(model)

    fixed_asym = asym_pairs === :auto ? nothing : _llgmm_validate_pairs(asym_pairs, n; symmetric=false)
    fixed_sym = sym_pairs === :auto ? nothing : _llgmm_validate_pairs(sym_pairs, n; symmetric=true)

    sel_res = nothing
    active_asym, active_sym = if (asym_pairs === :auto || sym_pairs === :auto) && moment_selection !== :none
        sel_res = select_gmm_moments(model;
                                     form=form, weighting=weighting, method=moment_selection,
                                     kernel=kernel, bandwidth=bandwidth, max_iter=max_iter,
                                     tol=tol, n_starts=n_starts, rng=rng_eff,
                                     fixed_asym_pairs=fixed_asym, fixed_sym_pairs=fixed_sym)
        (sel_res.asym_pairs, sel_res.sym_pairs)
    else
        (fixed_asym === nothing ? _llgmm_all_asym_pairs(n) : fixed_asym,
         fixed_sym === nothing ? _llgmm_all_sym_pairs(n) : fixed_sym)
    end

    est = _llgmm_estimate_core(model, Y_eff, X, active_asym, active_sym;
                               form=form, weighting=weighting, kernel=kernel,
                               bandwidth=bandwidth, max_iter=max_iter, tol=tol_T,
                               n_starts=n_starts, rng=rng_eff)

    if !est.rank_ok
        @warn "Lanne–Luoto GMM Jacobian is rank-deficient ($(est.jacobian_rank) < $(est.n_params)); " *
              "shock identification may fail (e.g. a Gaussian shock enters only in third power)." maxlog=1
    end

    manifest = capture_manifest(; seed=seed,
        settings=Dict{String,Any}(
            "form"             => String(form),
            "weighting"        => String(weighting),
            "moment_selection" => String(moment_selection),
            "kernel"           => String(ki),
            "bandwidth"        => bandwidth isa Symbol ? String(bandwidth) : Int(bandwidth),
            "n_moments"        => est.n_moments,
            "n_params"         => est.n_params,
        ))

    LanneLuotoGMMResult{T}(
        est.B0, est.A0, est.Q, est.sigma, est.nu, est.A, est.Pi, est.Sigma,
        model, est.shocks, est.theta, est.vcov, est.se_B0, est.se_theta,
        est.W, est.S_hac, est.G_jac, est.g_bar, est.J_stat, est.J_pvalue,
        est.df, est.n_moments, est.n_params, est.jacobian_rank, est.rank_ok,
        est.asym_pairs, est.sym_pairs, sel_res, form, weighting, ki, est.bw,
        est.converged, est.iterations, copy(model.varnames), snames, manifest,
    )
end

function estimate_lanne_luoto_gmm(Y::AbstractMatrix, p::Union{Int,Nothing}=nothing;
                                  check_stability::Bool=true,
                                  varnames::Union{Nothing,Vector{String}}=nothing,
                                  kwargs...)
    Ym = eltype(Y) <: AbstractFloat ? Y : Float64.(Y)
    T_obs, n = size(Ym)
    p_use = if p === nothing
        max_p = min(8, max(1, (T_obs - 10) ÷ max(3 * n, 1)))
        max_p >= 1 && T_obs > max_p + 2 ? select_lag_order(Ym, max_p; criterion=:bic) : 1
    else
        Int(p)
    end
    model = _llgmm_var_for_Y(Ym, p_use; check_stability=check_stability, varnames=varnames)
    estimate_lanne_luoto_gmm(model; kwargs...)
end

# =============================================================================
# Structural Ecosystem Integration: IRF, FEVD, Historical Decomposition, Labelling
# =============================================================================

function _llgmm_effective_var_and_Q(r::LanneLuotoGMMResult{T}) where {T<:AbstractFloat}
    Y_eff, X = _llgmm_construct_matrices(r.var_ols)
    U_gmm = Y_eff - X * r.Pi
    m_eff = VARModel(r.var_ols.Y, r.var_ols.p, r.Pi, U_gmm, r.Sigma,
                     r.var_ols.aic, r.var_ols.bic, r.var_ols.hqic, r.varnames)
    L = safe_cholesky(r.Sigma; silent=true)
    Q_eff = Matrix{T}(L \ r.B0)
    (m_eff, Q_eff)
end

function irf(r::LanneLuotoGMMResult{T}, horizon::Int=20; kwargs...) where {T<:AbstractFloat}
    m_eff, Q_eff = _llgmm_effective_var_and_Q(r)
    kw = Dict{Symbol,Any}(pairs(kwargs))
    if !haskey(kw, :shock_names) || kw[:shock_names] === nothing
        kw[:shock_names] = r.shock_names
    end
    irf(m_eff, horizon; method=:svar, _svar_Q=Q_eff, kw...)
end

function fevd(r::LanneLuotoGMMResult{T}, horizon::Int=20; kwargs...) where {T<:AbstractFloat}
    m_eff, Q_eff = _llgmm_effective_var_and_Q(r)
    kw = Dict{Symbol,Any}(pairs(kwargs))
    if !haskey(kw, :shock_names) || kw[:shock_names] === nothing
        kw[:shock_names] = r.shock_names
    end
    fevd(m_eff, horizon; method=:svar, _svar_Q=Q_eff, kw...)
end

function historical_decomposition(r::LanneLuotoGMMResult{T},
                                  horizon::Int=effective_nobs(r);
                                  kwargs...) where {T<:AbstractFloat}
    m_eff, Q_eff = _llgmm_effective_var_and_Q(r)
    kw = Dict{Symbol,Any}(pairs(kwargs))
    if !haskey(kw, :shock_names) || kw[:shock_names] === nothing
        kw[:shock_names] = r.shock_names
    end
    historical_decomposition(m_eff, horizon; method=:svar, _svar_Q=Q_eff, kw...)
end

function _signed_perm_result(result::LanneLuotoGMMResult{T}, perm::Vector{Int},
                             signs::Vector{Int};
                             shock_names::Union{Nothing,Vector{String}}=nothing) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    p = result.var_ols.p
    B0_new = _col_signed(result.B0, perm, signs)
    Q_new = _col_signed(result.Q, perm, signs)
    shocks_new = _col_signed(result.shocks, perm, signs)
    names_new = shock_names === nothing ? result.shock_names[perm] : Vector{String}(shock_names)
    asym_new, sym_new = _llgmm_remap_pairs(result.asym_pairs, result.sym_pairs, perm)

    A0_new = result.form === :b_model ? Matrix{T}(robust_inv(B0_new; silent=true)) : result.A0
    sig_new = result.sigma[perm]
    theta_new = result.theta
    vcov_new = result.vcov
    se_B0_new = _col_signed(result.se_B0, perm, ones(Int, n))
    se_theta_new = result.se_theta
    if result.form === :b_model
        theta_new = _llgmm_pack_theta(result.nu, result.A, B0_new, nothing, :b_model)
        # Transform vec(B) block of vcov under column permutation & sign flip
        k = length(result.theta)
        n_pi = n * (1 + n * p)
        P_theta = zeros(T, k, k)
        @inbounds for j in 1:n_pi
            P_theta[j, j] = one(T)
        end
        @inbounds for c_new in 1:n
            c_old = perm[c_new]
            sgn = T(signs[c_new])
            for r_i in 1:n
                P_theta[n_pi + r_i + (c_new - 1) * n, n_pi + r_i + (c_old - 1) * n] = sgn
            end
        end
        vcov_new = Matrix{T}((P_theta * result.vcov * transpose(P_theta)))
        vcov_new = Matrix{T}((vcov_new + transpose(vcov_new)) .* T(0.5))
        se_theta_new = sqrt.(max.(diag(vcov_new), zero(T)))
        se_B0_new = reshape(se_theta_new[(n_pi + 1):end], n, n)
    else
        a_conv = _llgmm_b0_to_a0(B0_new)
        A0_new = a_conv.A0
        sig_new = a_conv.sigma
        theta_new = _llgmm_pack_theta(result.nu, result.A, A0_new, sig_new, :a_model)
    end

    LanneLuotoGMMResult{T}(
        B0_new, A0_new, Q_new, sig_new, result.nu, result.A, result.Pi, result.Sigma,
        result.var_ols, shocks_new, theta_new, vcov_new, se_B0_new, se_theta_new,
        result.W, result.S_hac, result.G_jac, result.g_bar, result.J_stat, result.J_pvalue,
        result.df, result.n_moments, result.n_params, result.jacobian_rank, result.rank_ok,
        asym_new, sym_new, result.selection, result.form, result.weighting, result.kernel,
        result.bandwidth, result.converged, result.iterations, result.varnames, names_new,
        result.manifest,
    )
end

function test_overidentification(model::VARModel{T}, res::LanneLuotoGMMResult{T};
                                 restrictions=nothing,
                                 n_bootstrap::Int=999,
                                 seed::Union{Integer,Nothing}=nothing,
                                 rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    _resolve_repro_rng(rng, seed)
    if restrictions === nothing
        identified = isnan(res.J_pvalue) ? true : res.J_pvalue >= T(0.05)
        return IdentifiabilityTestResult{T}(
            :overidentification, res.J_stat, res.J_pvalue, identified,
            Dict{Symbol,Any}(:method => :hansen_j, :df => res.df,
                             :asym_pairs => res.asym_pairs, :sym_pairs => res.sym_pairs,
                             :just_identified => res.df == 0),
        )
    end
    ht = test_svar_hypothesis(res, restrictions; method=:gmm_lr, seed=seed, rng=rng)
    return IdentifiabilityTestResult{T}(
        :overidentification, ht.statistic, ht.pvalue, ht.pvalue >= T(0.05),
        Dict{Symbol,Any}(:method => :gmm_lr, :df => ht.df, :just_identified => false),
    )
end

# =============================================================================
# GMM LR-Type Distance Test & Wald Test (Newey & West 1987; LL §3.2 eq. 12)
# =============================================================================

function _llgmm_B0_from_theta(theta::AbstractVector{Tθ}, n::Int, p::Int, form::Symbol) where {Tθ<:Real}
    Pi_T, B, A0, sigma = _llgmm_unpack_theta(theta, n, p, form)
    B0 = form === :b_model ? B : robust_inv(A0; silent=true) * Diagonal(sigma)
    (Pi_T, B0)
end

function _llgmm_eval_hyp_spec(sp::_SVARHypothesisSpec, theta::AbstractVector{Tθ},
                              n::Int, p::Int, form::Symbol, perm::Vector{Int}) where {Tθ<:Real}
    Pi_T, B0 = _llgmm_B0_from_theta(theta, n, p, form)
    if sp.kind === :recursiveness
        return Tθ[B0[perm[i], perm[j]] for i in 1:(n - 1) for j in (i + 1):n]
    elseif sp.kind === :impact_zero
        return Tθ[B0[i, perm[j]] for (i, j) in sp.zeros_idx]
    elseif sp.kind === :impact_linear
        d = diag(B0)
        B_norm = B0 * Diagonal(one(Tθ) ./ d)
        beta = _llgmm_vecd0(B_norm)
        return Matrix{Tθ}(sp.R_beta) * beta .- Vector{Tθ}(sp.r_beta)
    elseif sp.kind === :long_run
        A_sum = zeros(Tθ, n, n)
        @inbounds for l in 1:p
            A_sum .+= @view(Pi_T[:, (2 + (l - 1) * n):(1 + l * n)])
        end
        A1 = Matrix{Tθ}(I, n, n) - A_sum
        Theta_LR = robust_inv(A1; silent=true) * B0
        return Tθ[Theta_LR[i, perm[j]] for (i, j) in sp.zeros_idx]
    elseif sp.kind === :horizon_irf
        H_max = maximum(r -> r[1], sp.horizon_idx)
        Phi = Vector{Matrix{Tθ}}(undef, H_max + 1)
        Phi[1] = Matrix{Tθ}(I, n, n)
        @inbounds for hh in 1:H_max
            P_h = zeros(Tθ, n, n)
            for l in 1:min(p, hh)
                P_h .+= @view(Pi_T[:, (2 + (l - 1) * n):(1 + l * n)]) * Phi[hh + 1 - l]
            end
            Phi[hh + 1] = P_h
        end
        return Tθ[dot(@view(Phi[h + 1][i, :]), @view(B0[:, perm[j]])) for (h, i, j) in sp.horizon_idx]
    elseif sp.kind === :nonlinear
        n_pi = n * (1 + n * p)
        val = if applicable(sp.g_func, theta)
            sp.g_func(theta)
        else
            sp.g_func(@view(theta[1:n_pi]), vec(B0))
        end
        return (val isa AbstractVector || val isa Tuple) ? collect(Tθ, val) : Tθ[val]
    elseif sp.kind === :mixed
        parts = [_llgmm_eval_hyp_spec(sub, theta, n, p, form,
                                      sub.ordering isa Vector{Int} ? sub.ordering : perm)
                 for sub in sp.subspecs]
        return vcat(parts...)
    end
    throw(ArgumentError("Unsupported hypothesis kind :$(sp.kind)"))
end

function _llgmm_resolve_hyp_ordering(hyp::_SVARHypothesisSpec, res::LanneLuotoGMMResult{T}) where {T<:AbstractFloat}
    n = nvars(res)
    p = res.var_ols.p
    if hyp.kind === :mixed && any(sub -> sub.ordering === :best_match, hyp.subspecs)
        resolved = [_llgmm_resolve_hyp_ordering(sub, res)[1] for sub in hyp.subspecs]
        hmix = mixed_svar_hypothesis(resolved)
        perm = hmix.ordering isa Vector{Int} ? hmix.ordering : collect(1:n)
        return hmix, perm
    end
    hyp.ordering === :best_match || begin
        perm = hyp.ordering isa Vector{Int} ? hyp.ordering : collect(1:n)
        return hyp, perm
    end
    best_perm = collect(1:n)
    best_val = T(Inf)
    found = false
    for perm in _permutations(n)
        cand = try
            if hyp.kind === :recursiveness
                recursiveness_hypothesis(n; ordering=perm)
            elseif hyp.kind === :impact_zero
                impact_zero_hypothesis(n, hyp.zeros_idx; ordering=perm)
            elseif hyp.kind === :impact_linear
                R_perm = _lmshyp_permute_R_beta(hyp.R_beta, perm, n)
                _SVARHypothesisSpec(:impact_linear, n, hyp.df, perm, Tuple{Int,Int}[],
                                    Tuple{Int,Int,Int}[], R_perm, hyp.r_beta,
                                    _SVARHypothesisSpec[], nothing, hyp.description)
            elseif hyp.kind === :long_run
                long_run_hypothesis(n, hyp.zeros_idx; ordering=perm)
            elseif hyp.kind === :horizon_irf
                horizon_irf_hypothesis(n, hyp.horizon_idx; ordering=perm)
            else
                hyp
            end
        catch e
            e isa ArgumentError ? nothing : rethrow(e)
        end
        cand === nothing && continue
        val = sum(abs2, _llgmm_eval_hyp_spec(cand, res.theta, n, p, res.form, perm))
        if val < best_val
            best_val = val
            best_perm = perm
            found = true
        end
    end
    found || (best_perm = collect(1:n))
    cand_best = if hyp.kind === :recursiveness
        recursiveness_hypothesis(n; ordering=best_perm)
    elseif hyp.kind === :impact_zero
        impact_zero_hypothesis(n, hyp.zeros_idx; ordering=best_perm)
    elseif hyp.kind === :impact_linear
        R_perm = _lmshyp_permute_R_beta(hyp.R_beta, best_perm, n)
        _SVARHypothesisSpec(:impact_linear, n, hyp.df, best_perm, Tuple{Int,Int}[],
                            Tuple{Int,Int,Int}[], R_perm, hyp.r_beta,
                            _SVARHypothesisSpec[], nothing, hyp.description)
    elseif hyp.kind === :long_run
        long_run_hypothesis(n, hyp.zeros_idx; ordering=best_perm)
    elseif hyp.kind === :horizon_irf
        horizon_irf_hypothesis(n, hyp.horizon_idx; ordering=best_perm)
    else
        hyp
    end
    return cand_best, best_perm
end

function _llgmm_extract_linear_R(hyp::_SVARHypothesisSpec, n::Int, p::Int,
                                  form::Symbol, perm::Vector{Int}, ::Type{T}) where {T<:AbstractFloat}
    form === :b_model || return nothing
    k = _llgmm_n_params(n, p)
    n_pi = n * (1 + n * p)
    if hyp.kind === :recursiveness
        pairs = Tuple{Int,Int}[(perm[i], perm[j]) for i in 1:(n - 1) for j in (i + 1):n]
        q = length(pairs)
        R = zeros(T, q, k)
        @inbounds for (row, (i, j)) in enumerate(pairs)
            R[row, n_pi + i + (j - 1) * n] = one(T)
        end
        return (R, zeros(T, q))
    elseif hyp.kind === :impact_zero
        q = length(hyp.zeros_idx)
        R = zeros(T, q, k)
        @inbounds for (row, (i, j)) in enumerate(hyp.zeros_idx)
            R[row, n_pi + i + (perm[j] - 1) * n] = one(T)
        end
        return (R, zeros(T, q))
    end
    nothing
end

function _llgmm_solve_linear_restricted(theta_init::Vector{T}, R::Matrix{T}, r::Vector{T},
                                        Y_eff::AbstractMatrix{T}, X::AbstractMatrix{T},
                                        W::AbstractMatrix{T}, n::Int, p::Int, form::Symbol,
                                        asym_pairs::Vector{Tuple{Int,Int}},
                                        sym_pairs::Vector{Tuple{Int,Int}},
                                        max_iter::Int, tol::T) where {T<:AbstractFloat}
    q_r, k = size(R)
    F = svd(R; full=true)
    sv_tol = max(T(1e-10), eps(T) * max(T(q_r), T(k))) * max(one(T), isempty(F.S) ? zero(T) : F.S[1])
    rk = count(>(sv_tol), F.S)
    rk == q_r || throw(ArgumentError("Restriction matrix R is rank-deficient (rank $rk < $q_r)"))
    theta0 = Vector{T}(F.V[:, 1:rk] * ((transpose(F.U[:, 1:rk]) * r) ./ F.S[1:rk]))
    M_null = rk < k ? Matrix{T}(F.V[:, (rk + 1):end]) : zeros(T, k, 0)
    n_free = size(M_null, 2)

    if n_free == 0
        _, g_bar, _, ok = _llgmm_jacobian_impl(theta0, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
        Q_val = ok ? T(dot(g_bar, W * g_bar)) : T(Inf)
        return (theta=theta0, Q=Q_val, converged=true, iters=0)
    end

    Mt = transpose(M_null)
    gamma = Vector{T}(Mt * (theta_init - theta0))
    theta_cur = theta0 + M_null * gamma
    _, g_bar, G_jac, ok = _llgmm_jacobian_impl(theta_cur, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
    if !ok
        # Fall back to theta0 if projection is near-singular
        gamma = zeros(T, n_free)
        theta_cur = copy(theta0)
        _, g_bar, G_jac, ok = _llgmm_jacobian_impl(theta_cur, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
    end

    Q_cur = ok ? T(dot(g_bar, W * g_bar)) : T(Inf)
    lambda = T(1e-10)
    iters = 0
    converged = false
    gn_tol = min(tol, T === Float32 ? T(1e-5) : T(1e-9))

    for _ in 1:max_iter
        iters += 1
        G_red = G_jac * M_null
        Wg = W * g_bar
        grad_half = transpose(G_red) * Wg
        if maximum(abs, grad_half) * T(2) <= gn_tol
            converged = true
            break
        end
        H = Matrix{T}(transpose(G_red) * W * G_red)
        H = Matrix{T}((H + transpose(H)) .* T(0.5))
        scale_d = max.(diag(H), one(T))

        accepted = false
        for _ in 1:8
            H_reg = H + Diagonal(lambda .* scale_d)
            step = -Vector{T}(robust_inv(H_reg; silent=true) * grad_half)
            all(isfinite, step) || (lambda *= T(10); continue)
            dot(grad_half, step) >= zero(T) && (step = -Vector{T}(grad_half))

            alpha = one(T)
            for _ in 1:15
                g_cand = gamma .+ alpha .* step
                th_cand = theta0 + M_null * g_cand
                _, g_c, G_c, ok_c = _llgmm_jacobian_impl(th_cand, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
                if ok_c
                    Q_cand = T(dot(g_c, W * g_c))
                    if isfinite(Q_cand) && Q_cand <= Q_cur + T(1e-14) * max(one(T), Q_cur)
                        gamma = g_cand
                        theta_cur = th_cand
                        g_bar = g_c
                        G_jac = G_c
                        Q_cur = Q_cand
                        lambda = max(T(1e-12), lambda * T(0.3))
                        accepted = true
                        if maximum(abs, alpha .* step) <= gn_tol &&
                           maximum(abs, transpose(G_jac * M_null) * (W * g_bar)) * T(2) <= max(gn_tol * T(10), T(1e-7))
                            converged = true
                        end
                        break
                    end
                end
                alpha *= T(0.5)
            end
            accepted && break
            lambda *= T(10)
            lambda > T(1e8) && break
        end
        (converged || !accepted) && break
    end

    # L-BFGS polish on reduced γ-space
    if !converged
        fg_red! = (G_out, g_in::Vector{T}) -> begin
            th_p = theta0 + M_null * g_in
            _, g_p, G_p, ok_p = _llgmm_jacobian_impl(th_p, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
            ok_p || return T(Inf)
            Wg_p = W * g_p
            val = T(dot(g_p, Wg_p))
            isfinite(val) || return T(Inf)
            if G_out !== nothing
                copyto!(G_out, T(2) .* (Mt * (transpose(G_p) * Wg_p)))
            end
            return val
        end
        res_opt = try
            _llgmm_lbfgs_fg(fg_red!, gamma, max(max_iter, 200), gn_tol)
        catch
            nothing
        end
        if res_opt !== nothing
            iters += Optim.iterations(res_opt)
            th_opt = theta0 + M_null * Vector{T}(Optim.minimizer(res_opt))
            _, g_p, _, ok_p = _llgmm_jacobian_impl(th_opt, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
            if ok_p
                Q_p = T(dot(g_p, W * g_p))
                if isfinite(Q_p) && Q_p <= Q_cur
                    theta_cur = th_opt
                    Q_cur = Q_p
                end
            end
            converged = converged || Optim.converged(res_opt)
        end
    end

    (theta=theta_cur, Q=Q_cur, converged=converged, iters=iters)
end

function _llgmm_solve_auglag_restricted(theta_init::Vector{T}, g_fn, q_r::Int,
                                        Y_eff::AbstractMatrix{T}, X::AbstractMatrix{T},
                                        W::AbstractMatrix{T}, n::Int, p::Int, form::Symbol,
                                        asym_pairs::Vector{Tuple{Int,Int}},
                                        sym_pairs::Vector{Tuple{Int,Int}},
                                        max_iter::Int, tol::T) where {T<:AbstractFloat}
    k = length(theta_init)
    theta_curr = copy(theta_init)
    nu_mult = zeros(T, q_r)
    rho = T(50.0)
    rho_max = T === Float32 ? T(1e7) : T(1e12)
    c_tol = T === Float32 ? T(1e-5) : T(1e-9)
    g_inner = min(tol, T === Float32 ? T(1e-5) : T(1e-8))
    iters_total = 0
    converged_flag = false
    prev_c = T(Inf)

    jac_res = ForwardDiff.DiffResults.JacobianResult(zeros(T, q_r), theta_curr)
    jac_cfg = ForwardDiff.JacobianConfig(g_fn, theta_curr)

    inner_iters = clamp(max_iter, 150, 400)
    for outer in 1:18
        fg_aug! = (G_out, xv::Vector{T}) -> begin
            _, g_p, G_p, ok_p = _llgmm_jacobian_impl(xv, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
            ok_p || return T(Inf)
            ForwardDiff.jacobian!(jac_res, g_fn, xv, jac_cfg)
            c_val = ForwardDiff.DiffResults.value(jac_res)
            J_c = ForwardDiff.DiffResults.jacobian(jac_res)
            (all(isfinite, c_val) && all(isfinite, J_c)) || return T(Inf)

            Wg_p = W * g_p
            Q_p = T(dot(g_p, Wg_p))
            mult = nu_mult .+ rho .* c_val
            aug_val = Q_p + T(dot(nu_mult, c_val)) + T(0.5) * rho * T(dot(c_val, c_val))
            isfinite(aug_val) || return T(Inf)
            if G_out !== nothing
                grad = T(2) .* (transpose(G_p) * Wg_p) .+ transpose(J_c) * mult
                copyto!(G_out, grad)
            end
            return aug_val
        end

        res_opt = try
            _llgmm_lbfgs_fg(fg_aug!, theta_curr, inner_iters, g_inner)
        catch
            nothing
        end
        res_opt === nothing && break
        iters_total += Optim.iterations(res_opt)
        x_next = Vector{T}(Optim.minimizer(res_opt))

        _, g_p, G_p, ok_p = _llgmm_jacobian_impl(x_next, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
        if !ok_p
            rho = min(rho * T(5), rho_max)
            continue
        end
        ForwardDiff.jacobian!(jac_res, g_fn, x_next, jac_cfg)
        c_val = ForwardDiff.DiffResults.value(jac_res)
        J_c = ForwardDiff.DiffResults.jacobian(jac_res)
        c_norm = maximum(abs, c_val)

        theta_curr = x_next
        nu_mult .+= rho .* c_val

        grad_Q = T(2) .* (transpose(G_p) * (W * g_p))
        JJt = Matrix{T}(J_c * transpose(J_c))
        nu_ls = -(robust_inv(JJt; silent=true) * (J_c * grad_Q))
        kkt_res = maximum(abs, grad_Q .+ transpose(J_c) * nu_ls)
        kkt_tol = max(tol * T(10), T === Float32 ? T(5e-4) : T(1e-6))

        if c_norm <= c_tol && (kkt_res <= kkt_tol || outer >= 10)
            converged_flag = true
            break
        end
        if c_norm > c_tol
            rho = (outer == 1 || c_norm > T(0.25) * prev_c) ? min(rho * T(5), rho_max) : min(rho * T(2), rho_max)
        end
        prev_c = c_norm
    end

    if !converged_flag
        c_final = g_fn(theta_curr)
        converged_flag = maximum(abs, c_final) <= max(c_tol * T(100), T(1e-4))
    end
    _, g_f, _, _ = _llgmm_jacobian_impl(theta_curr, Y_eff, X, n, p, form, asym_pairs, sym_pairs)
    Q_f = T(dot(g_f, W * g_f))
    (theta=theta_curr, Q=Q_f, converged=converged_flag, iters=iters_total)
end

function _llgmm_run_hyp_test(res::LanneLuotoGMMResult{T}, hyp_kind::Symbol, g_fn, q_r::Int,
                             perm::Vector{Int}, desc::String,
                             lin_Rr::Union{Nothing,Tuple{Matrix{T},Vector{T}}},
                             method::Symbol, max_iter::Int, tol::T,
                             seed::Union{Integer,Nothing}) where {T<:AbstractFloat}
    n = nvars(res)
    p = res.var_ols.p
    Y_eff, X = _llgmm_construct_matrices(res.var_ols)
    T_eff = size(Y_eff, 1)

    if method === :wald
        g_hat = Vector{T}(g_fn(res.theta))
        length(g_hat) == q_r || throw(ArgumentError("g(θ) returned length $(length(g_hat)), expected $q_r"))
        J_h = Matrix{T}(ForwardDiff.jacobian(g_fn, res.theta))
        V_g = Matrix{T}(J_h * res.vcov * transpose(J_h))
        V_g_sym = Matrix{T}((V_g .+ transpose(V_g)) .* T(0.5))
        V_g_inv = Matrix{T}(robust_inv(V_g_sym; silent=true))
        stat = max(zero(T), T(dot(g_hat, V_g_inv * g_hat)))
        pval = T(ccdf(Chisq(q_r), Float64(stat)))
        manifest = seed !== nothing ? capture_manifest(; seed=seed, settings=Dict{String,Any}(
            "hypothesis_kind" => String(hyp_kind), "method" => "wald")) : nothing
        return SVARHypothesisTest{T}(
            hyp_kind, :wald, stat, pval, q_r, nothing, :none, 0,
            T(-0.5) * T(T_eff) * T(dot(res.g_bar, res.W * res.g_bar)),
            nothing, copy(res.theta), nothing, g_hat, nothing,
            perm, :hac, true, 0, T_eff, n, desc, manifest,
        )
    else
        # Newey-West (1987) GMM LR-type distance test (LL §3.2 eq. 12):
        # D_T = T [Q_T(θ̃_T; Ŵ_unres) - Q_T(θ̂_T; Ŵ_unres)] ~ χ²_s
        W_unres = res.W
        theta_u = res.theta
        Q_u = T(dot(res.g_bar, W_unres * res.g_bar))
        if res.weighting === :cue
            # Under CUE, polish unrestricted θ under fixed Ŵ_unres so Q_u ≤ Q_r strictly
            pol_u = _llgmm_solve_fixed_W(res.theta, Y_eff, X, W_unres, n, p, res.form,
                                         res.asym_pairs, res.sym_pairs; max_iter=max_iter, tol=tol)
            if isfinite(pol_u.Q) && pol_u.Q <= Q_u
                theta_u = pol_u.theta
                Q_u = pol_u.Q
            end
        end

        rsol = if lin_Rr !== nothing
            _llgmm_solve_linear_restricted(theta_u, lin_Rr[1], lin_Rr[2],
                                           Y_eff, X, W_unres, n, p, res.form,
                                           res.asym_pairs, res.sym_pairs, max_iter, tol)
        else
            _llgmm_solve_auglag_restricted(theta_u, g_fn, q_r,
                                           Y_eff, X, W_unres, n, p, res.form,
                                           res.asym_pairs, res.sym_pairs, max_iter, tol)
        end

        D_T = max(zero(T), T(T_eff) * (rsol.Q - Q_u))
        pval = T(ccdf(Chisq(q_r), Float64(D_T)))
        g_val = Vector{T}(g_fn(rsol.theta))
        manifest = seed !== nothing ? capture_manifest(; seed=seed, settings=Dict{String,Any}(
            "hypothesis_kind" => String(hyp_kind), "method" => "gmm_lr")) : nothing
        return SVARHypothesisTest{T}(
            hyp_kind, :gmm_lr, D_T, pval, q_r, nothing, :none, 0,
            T(-0.5) * T(T_eff) * Q_u, T(-0.5) * T(T_eff) * rsol.Q,
            theta_u, rsol.theta, g_val, nothing,
            perm, :hac, rsol.converged, rsol.iters, T_eff, n, desc, manifest,
        )
    end
end

"""
    test_svar_hypothesis(res::LanneLuotoGMMResult{T}, hyp;
                         method=:gmm_lr, max_iter=200, tol=1e-8,
                         rng=Random.default_rng(), seed=nothing) -> SVARHypothesisTest{T}

Test a structural hypothesis `hyp` on a fitted Lanne–Luoto (2021) non-Gaussian SVAR GMM
result `res`.

Supports two test principles (both asymptotically ``\\chi^2(s)`` under ``H_0``):
- `:gmm_lr` (default, Newey & West 1987; Lanne & Luoto 2021 §3.2 eq. 12): GMM LR-type
  distance statistic computed using the **same unrestricted efficient weighting matrix**
  ``\\hat{W}_{\\text{unres}} = \\hat{S}_{\\text{HAC}}(\\hat{\\theta}_T)^{-1}``:

```math
D_T = T_{\\text{eff}} \\left[ Q_T(\\tilde{\\theta}_T; \\hat{W}_{\\text{unres}}) - Q_T(\\hat{\\theta}_T; \\hat{W}_{\\text{unres}}) \\right] \\xrightarrow{d} \\chi^2(s)
```

where:
- ``\\tilde{\\theta}_T`` is the constrained GMM estimator minimizing ``Q_T(\\theta; \\hat{W}_{\\text{unres}})`` subject to ``H_0``
- ``\\hat{\\theta}_T`` is the unrestricted GMM estimator
- ``\\hat{W}_{\\text{unres}}`` is the unrestricted efficient HAC weighting matrix
- ``s`` is the number of structural restrictions

- `:wald`: Wald statistic ``g(\\hat{\\theta})' [J \\hat{V}_{\\theta} J']^{-1} g(\\hat{\\theta})``.

`hyp` may be any hypothesis specification from [`recursiveness_hypothesis`](@ref),
[`impact_zero_hypothesis`](@ref), [`impact_linear_hypothesis`](@ref),
[`long_run_hypothesis`](@ref), [`horizon_irf_hypothesis`](@ref),
[`mixed_svar_hypothesis`](@ref), or [`nonlinear_svar_hypothesis`](@ref), or an
`n × n` matrix mask of target values for ``B_0`` (`NaN`/`missing` for free entries;
`BitMatrix` for zeros), or linear matrices `(R, r)` acting on ``\\theta`` or ``\\operatorname{vec}(B_0)``.
"""
function test_svar_hypothesis(res::LanneLuotoGMMResult{T}, hyp::_SVARHypothesisSpec;
                              method::Symbol=:gmm_lr,
                              max_iter::Int=200,
                              tol::Real=1e-8,
                              rng::AbstractRNG=Random.default_rng(),
                              seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    method in (:gmm_lr, :wald) || throw(ArgumentError("method must be :gmm_lr or :wald, got :$method"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    tol > 0 || throw(ArgumentError("tol must be > 0, got $tol"))
    _resolve_repro_rng(rng, seed)

    n = nvars(res)
    p = res.var_ols.p
    (hyp.n === nothing || hyp.n == n) || throw(ArgumentError(
        "Hypothesis n=$(hyp.n) does not match LanneLuotoGMMResult n=$n"))

    hyp_eff, perm = _llgmm_resolve_hyp_ordering(hyp, res)
    g_fn = th -> _llgmm_eval_hyp_spec(hyp_eff, th, n, p, res.form, perm)
    lin_Rr = _llgmm_extract_linear_R(hyp_eff, n, p, res.form, perm, T)
    tol_T = max(T(tol), T === Float32 ? T(1e-6) : T(1e-10))

    _llgmm_run_hyp_test(res, hyp_eff.kind, g_fn, hyp_eff.df, perm, hyp_eff.description,
                        lin_Rr, method, max_iter, tol_T, seed)
end

function test_svar_hypothesis(res::LanneLuotoGMMResult{T}, restrictions::AbstractMatrix;
                              method::Symbol=:gmm_lr,
                              max_iter::Int=200,
                              tol::Real=1e-8,
                              rng::AbstractRNG=Random.default_rng(),
                              seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    method in (:gmm_lr, :wald) || throw(ArgumentError("method must be :gmm_lr or :wald, got :$method"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    tol > 0 || throw(ArgumentError("tol must be > 0, got $tol"))
    _resolve_repro_rng(rng, seed)

    n = nvars(res)
    p = res.var_ols.p
    size(restrictions) == (n, n) || throw(ArgumentError(
        "restrictions matrix must be $n × $n, got $(size(restrictions))"))

    targets = Tuple{Int,Int,T}[]
    if eltype(restrictions) <: Bool
        @inbounds for j in 1:n, i in 1:n
            restrictions[i, j] && push!(targets, (i, j, zero(T)))
        end
    else
        @inbounds for j in 1:n, i in 1:n
            v = restrictions[i, j]
            if v !== missing && v isa Number && isfinite(v)
                push!(targets, (i, j, T(v)))
            end
        end
    end
    isempty(targets) && throw(ArgumentError("restrictions matrix contains no finite target restrictions"))

    q_r = length(targets)
    k = res.n_params
    n_pi = n * (1 + n * p)
    lin_Rr = if res.form === :b_model
        R = zeros(T, q_r, k)
        r = Vector{T}(undef, q_r)
        @inbounds for (row, (i, j, val)) in enumerate(targets)
            R[row, n_pi + i + (j - 1) * n] = one(T)
            r[row] = val
        end
        (R, r)
    else
        nothing
    end
    g_fn = th -> begin
        _, B0 = _llgmm_B0_from_theta(th, n, p, res.form)
        eltype(th)[B0[i, j] - val for (i, j, val) in targets]
    end
    desc = "Impact matrix B₀ restrictions (df=$q_r)"
    tol_T = max(T(tol), T === Float32 ? T(1e-6) : T(1e-10))
    _llgmm_run_hyp_test(res, :impact_linear, g_fn, q_r, collect(1:n), desc,
                        lin_Rr, method, max_iter, tol_T, seed)
end

function test_svar_hypothesis(res::LanneLuotoGMMResult{T}, R::AbstractMatrix, r::AbstractVector;
                              method::Symbol=:gmm_lr,
                              max_iter::Int=200,
                              tol::Real=1e-8,
                              rng::AbstractRNG=Random.default_rng(),
                              seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    method in (:gmm_lr, :wald) || throw(ArgumentError("method must be :gmm_lr or :wald, got :$method"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    tol > 0 || throw(ArgumentError("tol must be > 0, got $tol"))
    _resolve_repro_rng(rng, seed)

    n = nvars(res)
    p = res.var_ols.p
    k = res.n_params
    q_r, ncol = size(R)
    q_r >= 1 || throw(ArgumentError("R must have at least 1 row"))
    length(r) == q_r || throw(ArgumentError("r length ($(length(r))) must match size(R, 1) ($q_r)"))
    ncol == n * (n - 1) && return test_svar_hypothesis(
        res, impact_linear_hypothesis(R, r); method=method, max_iter=max_iter, tol=tol, rng=rng, seed=seed)

    (ncol == k || ncol == n * n) || throw(ArgumentError(
        "R must have $k (full θ), $(n * n) (vec(B₀)), or $(n * (n - 1)) (vecd⁰(B)) columns, got $ncol"))

    Rf = Matrix{T}(R)
    rf = Vector{T}(r)
    sv = svdvals(Rf)
    rk = count(>(T(1e-10) * max(one(T), isempty(sv) ? zero(T) : sv[1])), sv)
    rk == q_r || throw(ArgumentError("R must have full row rank $q_r (got $rk)"))

    n_pi = n * (1 + n * p)
    lin_Rr = if ncol == k
        (Rf, rf)
    elseif res.form === :b_model
        R_full = zeros(T, q_r, k)
        R_full[:, (n_pi + 1):k] .= Rf
        (R_full, rf)
    else
        nothing
    end
    g_fn = th -> begin
        if ncol == k
            return Matrix{eltype(th)}(Rf) * th .- Vector{eltype(th)}(rf)
        else
            _, B0 = _llgmm_B0_from_theta(th, n, p, res.form)
            return Matrix{eltype(th)}(Rf) * vec(B0) .- Vector{eltype(th)}(rf)
        end
    end
    desc = "Linear GMM SVAR hypothesis (df=$q_r)"
    tol_T = max(T(tol), T === Float32 ? T(1e-6) : T(1e-10))
    _llgmm_run_hyp_test(res, :impact_linear, g_fn, q_r, collect(1:n), desc,
                        lin_Rr, method, max_iter, tol_T, seed)
end
