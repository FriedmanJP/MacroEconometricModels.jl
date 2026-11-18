# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Structural hypothesis testing and residual diagnostics for Lanne–Meitz–Saikkonen (2017)
non-Gaussian SVAR models:
- Composable short-run, linear, Blanchard–Quah long-run, horizon IRF, mixed, and custom
  nonlinear hypothesis specifications with `Π(B) ∈ B` diagonal interiority guards
- Classical test triad (`:wald`, `:lr`, `:lm`) propagating full joint `(π, β)` covariance
- Optional residual bootstrap (`:iid`, `:wild`) under the null hypothesis
- Per-shock and multivariate residual diagnostic battery (`diagnose_nongaussian_svar`)

References:
- Lanne, M., Meitz, M., & Saikkonen, P. (2017). "Identification and estimation of
  non-Gaussian structural vector autoregressions." Journal of Econometrics 196(2), 288–304 (§4.6, §5.2).
- Bjørnland, H. C., & Leitemo, K. (2009). "Identifying the interdependence between US
  monetary policy and the stock market." Journal of Monetary Economics 56(2), 275–282.
"""

using LinearAlgebra, Statistics, Distributions, Random
import Optim
import ForwardDiff

# =============================================================================
# Internal Hypothesis Specification (Unexported — may hold Function)
# =============================================================================

struct _SVARHypothesisSpec
    kind::Symbol
    n::Union{Nothing,Int}
    df::Int
    ordering::Union{Nothing,Symbol,Vector{Int}}
    zeros_idx::Vector{Tuple{Int,Int}}
    horizon_idx::Vector{Tuple{Int,Int,Int}}
    R_beta::Union{Nothing,Matrix{Float64}}
    r_beta::Union{Nothing,Vector{Float64}}
    subspecs::Vector{_SVARHypothesisSpec}
    g_func::Union{Nothing,Function}
    description::String
end

# =============================================================================
# Exported Result Types (Purely Serializable — RSER-01)
# =============================================================================

"""
    SVARHypothesisTest{T<:AbstractFloat} <: StatsAPI.HypothesisTest

Result from a structural hypothesis test on an LMS non-Gaussian SVAR (`LMSSVARResult{T}`).

# Fields
- `hypothesis_kind::Symbol` — `:recursiveness`, `:impact_zero`, `:impact_linear`, `:long_run`, `:horizon_irf`, `:mixed`, or `:nonlinear`
- `test_type::Symbol` — `:wald`, `:lr`, or `:lm`
- `statistic::T` — chi-squared test statistic
- `pvalue::T` — asymptotic ``\\chi^2(\\text{df})`` p-value
- `df::Int` — degrees of freedom (number of restrictions)
- `bootstrap_pvalue::Union{Nothing,T}` — residual bootstrap p-value when requested, else `nothing`
- `bootstrap_scheme::Symbol` — `:none`, `:iid`, or `:wild`
- `n_boot::Int` — number of bootstrap replications (`0` when `bootstrap=:none`)
- `loglik_unrestricted::T` — total log-likelihood at unrestricted MLE
- `loglik_restricted::Union{Nothing,T}` — total log-likelihood at constrained MLE (`nothing` for non-bootstrapped `:wald`)
- `theta_unrestricted::Vector{T}` — unrestricted parameter vector ``\\hat{\\theta}``
- `theta_restricted::Union{Nothing,Vector{T}}` — constrained MLE ``\\tilde{\\theta}`` (`nothing` for non-bootstrapped `:wald`)
- `g_value::Vector{T}` — restriction residual ``g(\\hat{\\theta})`` (or ``g(\\tilde{\\theta})`` for `:lm`)
- `score_norm::Union{Nothing,T}` — Euclidean norm of average score at ``\\tilde{\\theta}`` (for `:lm`)
- `shock_permutation::Vector{Int}` — column/variable permutation used
- `vcov_type::Symbol` — covariance estimator (`:hessian`, `:sandwich`, or `:opg`)
- `converged::Bool` — whether optimization converged
- `iterations::Int` — constrained optimizer iterations
- `nobs::Int` — effective sample size ``T_{\\text{eff}}``
- `nvars::Int` — number of endogenous variables ``n``
- `description::String` — human-readable description of ``H_0``
- `manifest::Union{ReproManifest,Nothing}` — reproducibility manifest when seeded or bootstrapped
"""
struct SVARHypothesisTest{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    hypothesis_kind::Symbol
    test_type::Symbol
    statistic::T
    pvalue::T
    df::Int
    bootstrap_pvalue::Union{Nothing,T}
    bootstrap_scheme::Symbol
    n_boot::Int
    loglik_unrestricted::T
    loglik_restricted::Union{Nothing,T}
    theta_unrestricted::Vector{T}
    theta_restricted::Union{Nothing,Vector{T}}
    g_value::Vector{T}
    score_norm::Union{Nothing,T}
    shock_permutation::Vector{Int}
    vcov_type::Symbol
    converged::Bool
    iterations::Int
    nobs::Int
    nvars::Int
    description::String
    manifest::Union{ReproManifest,Nothing}
end

StatsAPI.pvalue(r::SVARHypothesisTest) = r.pvalue
StatsAPI.dof(r::SVARHypothesisTest) = r.df
StatsAPI.nobs(r::SVARHypothesisTest) = r.nobs

function Base.show(io::IO, r::SVARHypothesisTest{T}) where {T}
    stars = _significance_stars(r.pvalue)
    test_label = r.test_type === :lr ? "Likelihood Ratio (LR)" :
                 r.test_type === :lm ? "Lagrange Multiplier (Rao Score)" : "Wald"
    spec = Any[
        "H₀"             r.description;
        "Hypothesis"     string(r.hypothesis_kind);
        "Test"           test_label;
        "Covariance"     string(r.vcov_type);
        "Statistic"      string(_fmt(r.statistic; digits=4), stars == "" ? "" : " $stars");
        "Degrees of freedom" r.df;
        "Asymptotic p-value" _format_pvalue(r.pvalue);
        "Observations"   r.nobs;
        "Converged"      r.converged ? "Yes" : "No"
    ]
    if r.bootstrap_pvalue !== nothing
        bstars = _significance_stars(r.bootstrap_pvalue)
        spec = vcat(spec, Any[
            "Bootstrap scheme" string(r.bootstrap_scheme, " (B=", r.n_boot, ")");
            "Bootstrap p-value" string(_format_pvalue(r.bootstrap_pvalue), bstars == "" ? "" : " $bstars")
        ])
    end
    if r.loglik_restricted !== nothing
        spec = vcat(spec, Any[
            "Log-lik (unrestricted)" _fmt(r.loglik_unrestricted; digits=4);
            "Log-lik (restricted)"   _fmt(r.loglik_restricted; digits=4)
        ])
    end
    _pretty_table(io, spec;
        title = "LMS (2017) Structural SVAR Hypothesis Test",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    eff_p = r.bootstrap_pvalue !== nothing ? r.bootstrap_pvalue : r.pvalue
    conc = eff_p < T(0.05) ?
        "Reject H₀ at 5% level ($(r.description) is rejected)" :
        "Fail to reject H₀ at 5% level (data are consistent with H₀)"
    conc_data = Any["Conclusion" conc; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["", ""], alignment=[:l, :l])
    _manifest_footer(io, r.manifest)
end

report(io::IO, r::SVARHypothesisTest; repro::Bool=false) = show(IOContext(io, :repro => repro), r)
report(r::SVARHypothesisTest; repro::Bool=false) = report(stdout, r; repro=repro)

"""
    NonGaussianDiagnostics{T<:AbstractFloat}

Residual specification and identification strength diagnostics for an `LMSSVARResult{T}`
(Lanne, Meitz & Saikkonen 2017, §5.2).

# Fields
- `jb_stats::Vector{T}` — per-shock Jarque–Bera test statistics
- `jb_pvals::Vector{T}` — per-shock Jarque–Bera p-values
- `jb_pvals_holm::Vector{T}` — Holm-adjusted per-shock Jarque–Bera p-values
- `skewness::Vector{T}` — per-shock sample skewness
- `excess_kurtosis::Vector{T}` — per-shock sample excess kurtosis
- `multivariate_jb_stat::T` — joint multivariate Jarque–Bera statistic
- `multivariate_jb_pval::T` — joint multivariate Jarque–Bera p-value
- `lb_levels_stats::Vector{T}` — per-shock Ljung–Box `Q(lags)` on levels ``\\hat{z}_{it}``
- `lb_levels_pvals::Vector{T}` — p-values for Ljung–Box on levels
- `lb_squares_stats::Vector{T}` — per-shock Ljung–Box `Q(lags)` on squares ``\\hat{z}_{it}^2`` (ARCH check)
- `lb_squares_pvals::Vector{T}` — p-values for Ljung–Box on squares
- `lb_abs_stats::Vector{T}` — per-shock Ljung–Box `Q(lags)` on absolute shocks ``|\\hat{z}_{it}|``
- `lb_abs_pvals::Vector{T}` — p-values for Ljung–Box on absolute shocks
- `portmanteau_stat::T` — multivariate Portmanteau statistic on ``\\hat{z}_t``
- `portmanteau_pval::T` — multivariate Portmanteau p-value
- `portmanteau_df::Int` — degrees of freedom of multivariate Portmanteau test
- `bds_stats::Vector{T}` — per-shock BDS independence test statistics (`m = 2`)
- `bds_pvals::Vector{T}` — per-shock BDS independence p-values
- `cross_corr_matrix::Matrix{T}` — `n × n` shock level correlation matrix
- `cross_sq_corr_matrix::Matrix{T}` — `n × n` squared-shock correlation matrix
- `cross_moment_stat::T` — joint pairwise squared-shock cross-moment statistic
- `cross_moment_pval::T` — pairwise cross-moment independence p-value
- `cross_moment_df::Int` — degrees of freedom ``n(n-1)/2``
- `n_gaussian_shocks::Int` — count of shocks failing to reject Gaussianity at 5% (Holm-adjusted JB)
- `weak_id_warning::Bool` — `true` when `n ≥ 2` and `n_gaussian_shocks ≥ 2` (LMS Prop. 1 non-identification warning)
- `lags::Int` — lag order used for serial correlation diagnostics
- `nobs::Int` — effective sample size ``T_{\\text{eff}}``
- `nvars::Int` — number of shocks ``n``
- `shock_names::Vector{String}` — shock labels
"""
struct NonGaussianDiagnostics{T<:AbstractFloat}
    jb_stats::Vector{T}
    jb_pvals::Vector{T}
    jb_pvals_holm::Vector{T}
    skewness::Vector{T}
    excess_kurtosis::Vector{T}
    multivariate_jb_stat::T
    multivariate_jb_pval::T
    lb_levels_stats::Vector{T}
    lb_levels_pvals::Vector{T}
    lb_squares_stats::Vector{T}
    lb_squares_pvals::Vector{T}
    lb_abs_stats::Vector{T}
    lb_abs_pvals::Vector{T}
    portmanteau_stat::T
    portmanteau_pval::T
    portmanteau_df::Int
    bds_stats::Vector{T}
    bds_pvals::Vector{T}
    cross_corr_matrix::Matrix{T}
    cross_sq_corr_matrix::Matrix{T}
    cross_moment_stat::T
    cross_moment_pval::T
    cross_moment_df::Int
    n_gaussian_shocks::Int
    weak_id_warning::Bool
    lags::Int
    nobs::Int
    nvars::Int
    shock_names::Vector{String}
end

StatsAPI.nobs(d::NonGaussianDiagnostics) = d.nobs
nvars(d::NonGaussianDiagnostics) = d.nvars

function Base.show(io::IO, d::NonGaussianDiagnostics{T}) where {T}
    n = d.nvars
    summary_data = Any[
        "Observations"         d.nobs;
        "Shocks"               n;
        "Diagnostic lags"      d.lags;
        "Gaussian shocks (5%)" d.n_gaussian_shocks;
        "Weak ID warning"      d.weak_id_warning ? "YES (≥2 Gaussian shocks)" : "No";
        "Multivariate JB"      string(_fmt(d.multivariate_jb_stat; digits=2), " (p=", _format_pvalue(d.multivariate_jb_pval), ")");
        "Portmanteau Q($(d.lags))" string(_fmt(d.portmanteau_stat; digits=2), " (df=", d.portmanteau_df, ", p=", _format_pvalue(d.portmanteau_pval), ")");
        "Cross-moment indep."  string(_fmt(d.cross_moment_stat; digits=2), " (df=", d.cross_moment_df, ", p=", _format_pvalue(d.cross_moment_pval), ")")
    ]
    _pretty_table(io, summary_data;
        title = "Non-Gaussian SVAR Residual Diagnostics (LMS §5.2)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    rows = Matrix{Any}(undef, n, 8)
    for i in 1:n
        rows[i, 1] = d.shock_names[i]
        rows[i, 2] = _fmt(d.skewness[i]; digits=3)
        rows[i, 3] = _fmt(d.excess_kurtosis[i]; digits=3)
        rows[i, 4] = string(_fmt(d.jb_stats[i]; digits=2), _significance_stars(d.jb_pvals_holm[i]))
        rows[i, 5] = _format_pvalue(d.jb_pvals[i])
        rows[i, 6] = string(_fmt(d.lb_levels_stats[i]; digits=2), " (", _format_pvalue(d.lb_levels_pvals[i]), ")")
        rows[i, 7] = string(_fmt(d.lb_squares_stats[i]; digits=2), " (", _format_pvalue(d.lb_squares_pvals[i]), ")")
        rows[i, 8] = isnan(d.bds_stats[i]) ? "NaN" : string(_fmt(d.bds_stats[i]; digits=2), " (", _format_pvalue(d.bds_pvals[i]), ")")
    end
    _pretty_table(io, rows;
        title = "Per-Shock Higher Moments, Serial Correlation & Independence",
        column_labels = ["Shock", "Skew", "Ex.Kurt", "JB", "JB p", "Q(z)", "Q(z²)", "BDS(2)"],
        alignment = [:l, :r, :r, :r, :r, :r, :r, :r],
    )

    if d.weak_id_warning
        _show_note(io, "WARNING: ≥ 2 shocks appear Gaussian; B₀ may suffer from weak or non-identification (LMS Prop. 1).")
    end
end

report(io::IO, d::NonGaussianDiagnostics) = (show(io, d); nothing)
report(d::NonGaussianDiagnostics) = report(stdout, d)

# =============================================================================
# Internal Validation & Indexing Helpers (_lmshyp_*)
# =============================================================================

@inline function _lmshyp_beta_index(i::Int, j::Int, n::Int)
    (j - 1) * (n - 1) + (i < j ? i : i - 1)
end

@inline function _lmshyp_beta_coords(idx::Int, n::Int)
    q, rem = divrem(idx - 1, n - 1)
    j = q + 1
    r = rem + 1
    i = r < j ? r : r + 1
    (i, j)
end

function _lmshyp_validate_ordering(ordering::Union{Nothing,Symbol,AbstractVector{Int}}, n::Int)
    if ordering === nothing || ordering === :canonical
        return collect(1:n)
    elseif ordering === :best_match
        return :best_match
    elseif ordering isa Symbol
        throw(ArgumentError("ordering Symbol must be :canonical or :best_match, got :$ordering"))
    else
        perm = collect(Int, ordering)
        (length(perm) == n && sort(perm) == collect(1:n)) ||
            throw(ArgumentError("ordering must be a permutation of 1:$n, got $ordering"))
        return perm
    end
end

function _lmshyp_diagonal_guard(i::Int, j::Int, col::Int, perm::Vector{Int})
    if i == col
        throw(ArgumentError(
            "LMS (2017, §4.6) B ∈ B interiority violation: restriction at ($i, $j) under " *
            "ordering=$perm maps to canonical diagonal entry B[$i, $i] = 1, which is fixed to " *
            "unity by complete identification Π(B). To test a zero on variable $i's impact from " *
            "a different shock, pass a column permutation `ordering` mapping shock $j to a " *
            "column ≠ $i (e.g., swapping column $j with another column)."))
    end
end

function _lmshyp_infer_n_from_nbeta(n_beta::Int)
    n_beta >= 2 || throw(ArgumentError(
        "R_beta must have n*(n-1) ≥ 2 columns for n ≥ 2 (got $n_beta)"))
    disc = 1 + 4 * n_beta
    s = isqrt(disc)
    if s * s == disc && iseven(1 + s)
        n = (1 + s) ÷ 2
        if n >= 2 && n * (n - 1) == n_beta
            return n
        end
    end
    sq = isqrt(n_beta)
    if sq * sq == n_beta
        throw(ArgumentError(
            "R_beta has $n_beta = $(sq)^2 columns, which appears to index vec(B) rather than " *
            "β = vecd⁰(B). Under LMS (2017, §4.6) B ∈ B normalization, diagonal entries B_{ii} = 1 " *
            "are fixed; R_beta must have n*(n-1) columns acting on off-diagonal β."))
    end
    throw(ArgumentError(
        "R_beta column count ($n_beta) does not equal n*(n-1) for any integer n ≥ 2"))
end

function _lmshyp_build_R_from_pairs(pairs::Vector{Tuple{Int,Int}}, n::Int)
    q = length(pairs)
    n_beta = n * (n - 1)
    R = zeros(Float64, q, n_beta)
    @inbounds for (row, (i, col)) in enumerate(pairs)
        R[row, _lmshyp_beta_index(i, col, n)] = 1.0
    end
    R, zeros(Float64, q)
end

function _lmshyp_permute_R_beta(R_in::Matrix{Float64}, perm::Vector{Int}, n::Int)
    q, n_beta = size(R_in)
    R_out = zeros(Float64, q, n_beta)
    for k_ord in 1:n_beta
        i, j = _lmshyp_beta_coords(k_ord, n)
        col_canon = perm[j]
        if i == col_canon
            if any(!iszero, @view(R_in[:, k_ord]))
                throw(ArgumentError(
                    "LMS (2017, §4.6) B ∈ B interiority violation: R_beta places nonzero weight " *
                    "on ($i, $j) which under ordering=$perm maps to fixed diagonal B[$i, $i] = 1. " *
                    "Permute `ordering` so shock $j maps to a column ≠ $i."))
            end
        else
            R_out[:, _lmshyp_beta_index(i, col_canon, n)] .+= @view(R_in[:, k_ord])
        end
    end
    R_out
end

# =============================================================================
# Public Hypothesis Builder Functions
# =============================================================================

"""
    recursiveness_hypothesis(n::Int; ordering=nothing) -> _SVARHypothesisSpec

Construct a structural hypothesis testing Cholesky recursive (lower-triangular) impact
ordering ``B_{ij} = 0`` for ``i < j`` (Lanne, Meitz & Saikkonen 2017, §4.6 & §5.2), with
``\\text{df} = n(n-1)/2`` restrictions.

# Arguments
- `n::Int` — number of endogenous variables (`n ≥ 2`)
- `ordering::Union{Nothing,Symbol,AbstractVector{Int}}=nothing` — variable ordering
  permutation of `1:n` (default `nothing` / `:canonical` uses `1:n`; `:best_match`
  selects the permutation minimizing upper-triangle norm)
"""
function recursiveness_hypothesis(n::Int;
                                  ordering::Union{Nothing,Symbol,AbstractVector{Int}}=nothing)
    n >= 2 || throw(ArgumentError("recursiveness_hypothesis requires n ≥ 2, got n=$n"))
    ord = _lmshyp_validate_ordering(ordering, n)
    q = n * (n - 1) ÷ 2
    if ord === :best_match
        return _SVARHypothesisSpec(
            :recursiveness, n, q, :best_match, Tuple{Int,Int}[], Tuple{Int,Int,Int}[],
            nothing, nothing, _SVARHypothesisSpec[], nothing,
            "Recursive (lower-triangular) impact B (best-match ordering)",
        )
    end
    pairs = Tuple{Int,Int}[(ord[i], ord[j]) for i in 1:(n - 1) for j in (i + 1):n]
    R, r = _lmshyp_build_R_from_pairs(pairs, n)
    desc = ord == collect(1:n) ?
        "Recursive (lower-triangular) B (df=$q)" :
        "Recursive B under ordering $ord (df=$q)"
    _SVARHypothesisSpec(
        :recursiveness, n, q, ord, pairs, Tuple{Int,Int,Int}[],
        R, r, _SVARHypothesisSpec[], nothing, desc,
    )
end

"""
    impact_zero_hypothesis(n::Int, zeros_idx::AbstractVector{<:Tuple{Int,Int}};
                           ordering=nothing) -> _SVARHypothesisSpec

Construct a zero-restriction hypothesis ``B_{i, \\pi(j)} = 0`` on the structural impact
matrix ``B \\in \\mathcal{B}`` (Lanne, Meitz & Saikkonen 2017, §4.6).

Enforces the LMS ``B \\in \\mathcal{B}`` diagonal interiority guard: any restriction
mapping to canonical diagonal ``B_{ii} = 1`` directly or through `ordering` throws an
`ArgumentError`.

# Arguments
- `n::Int` — number of endogenous variables
- `zeros_idx::AbstractVector{<:Tuple{Int,Int}}` — `(var, shock)` index pairs restricted to zero
- `ordering::Union{Nothing,Symbol,AbstractVector{Int}}=nothing` — column permutation of `1:n`
"""
function impact_zero_hypothesis(n::Int, zeros_idx::AbstractVector{<:Tuple{Int,Int}};
                                ordering::Union{Nothing,Symbol,AbstractVector{Int}}=nothing)
    n >= 1 || throw(ArgumentError("n must be ≥ 1, got $n"))
    isempty(zeros_idx) && throw(ArgumentError("zeros_idx must be non-empty"))
    zlist = Tuple{Int,Int}[(Int(i), Int(j)) for (i, j) in zeros_idx]
    for (i, j) in zlist
        (1 <= i <= n && 1 <= j <= n) || throw(ArgumentError(
            "restriction ($i, $j) out of bounds for n=$n"))
    end
    length(unique(zlist)) == length(zlist) || throw(ArgumentError(
        "zeros_idx contains duplicate (i, j) entries"))
    ord = _lmshyp_validate_ordering(ordering, n)
    q = length(zlist)
    if ord === :best_match
        n < 2 && throw(ArgumentError(
            "LMS (2017, §4.6) B ∈ B interiority violation: cannot restrict B[1,1] = 0 when n=1"))
        return _SVARHypothesisSpec(
            :impact_zero, n, q, :best_match, zlist, Tuple{Int,Int,Int}[],
            nothing, nothing, _SVARHypothesisSpec[], nothing,
            "Impact zero restrictions on B (best-match ordering, df=$q)",
        )
    end
    canon_pairs = Tuple{Int,Int}[]
    for (i, j) in zlist
        col = ord[j]
        _lmshyp_diagonal_guard(i, j, col, ord)
        push!(canon_pairs, (i, col))
    end
    R, r = _lmshyp_build_R_from_pairs(canon_pairs, n)
    _SVARHypothesisSpec(
        :impact_zero, n, q, ord, zlist, Tuple{Int,Int,Int}[],
        R, r, _SVARHypothesisSpec[], nothing,
        "Impact zero restrictions on B (df=$q)",
    )
end

"""
    impact_linear_hypothesis(R_beta::AbstractMatrix{<:Real},
                             r_beta::AbstractVector{<:Real}=zeros(size(R_beta, 1));
                             ordering=nothing) -> _SVARHypothesisSpec

Construct a general linear hypothesis on the off-diagonal impact parameters
``\\beta = \\operatorname{vecd}^0(B) \\in \\mathbb{R}^{n(n-1)}``:

```math
R_{\\beta} \\, \\beta = r_{\\beta}
```

where:
- ``R_{\\beta}`` is a full row-rank ``q \\times n(n-1)`` restriction matrix for ``n`` variables and ``q`` restrictions
- ``\\beta = \\operatorname{vecd}^0(B)`` stacks the off-diagonal entries of ``B \\in \\mathcal{B}`` in column-major order
- ``r_{\\beta}`` is the length-``q`` target vector
"""
function impact_linear_hypothesis(R_beta::AbstractMatrix{<:Real},
                                  r_beta::AbstractVector{<:Real}=zeros(Float64, size(R_beta, 1));
                                  ordering::Union{Nothing,Symbol,AbstractVector{Int}}=nothing)
    q, n_beta = size(R_beta)
    q >= 1 || throw(ArgumentError("R_beta must have at least 1 row"))
    length(r_beta) == q || throw(ArgumentError(
        "r_beta length ($(length(r_beta))) must match size(R_beta, 1) ($q)"))
    n = _lmshyp_infer_n_from_nbeta(n_beta)
    Rf = Matrix{Float64}(R_beta)
    rf = Vector{Float64}(r_beta)
    all(isfinite, Rf) && all(isfinite, rf) || throw(ArgumentError("R_beta and r_beta must be finite"))
    sv = svdvals(Rf)
    rk = count(>(1e-10 * max(1.0, isempty(sv) ? 0.0 : sv[1])), sv)
    rk == q || throw(ArgumentError("R_beta must have full row rank $q (got rank $rk)"))

    ord = _lmshyp_validate_ordering(ordering, n)
    if ord === :best_match
        return _SVARHypothesisSpec(
            :impact_linear, n, q, :best_match, Tuple{Int,Int}[], Tuple{Int,Int,Int}[],
            Rf, rf, _SVARHypothesisSpec[], nothing,
            "Linear impact hypothesis R_β β = r_β (best-match ordering, df=$q)",
        )
    end
    R_canon = ord == collect(1:n) ? Rf : _lmshyp_permute_R_beta(Rf, ord, n)
    _SVARHypothesisSpec(
        :impact_linear, n, q, ord, Tuple{Int,Int}[], Tuple{Int,Int,Int}[],
        R_canon, rf, _SVARHypothesisSpec[], nothing,
        "Linear impact hypothesis R_β β = r_β (df=$q)",
    )
end

"""
    long_run_hypothesis(n::Int, zeros_idx::AbstractVector{<:Tuple{Int,Int}};
                        ordering=nothing) -> _SVARHypothesisSpec

Construct a Blanchard–Quah long-run cumulative zero-restriction hypothesis on
``\\Theta_\\infty(\\pi, \\beta) = \\left(I_n - \\sum_{l=1}^{p} A_l\\right)^{-1} B(\\beta)``:

```math
g_k(\\pi, \\beta) = \\iota_{i_k}' \\left(I_n - \\sum_{l=1}^{p} A_l\\right)^{-1} B(\\beta) \\, \\iota_{\\pi(j_k)} = 0
```

where:
- ``g_k(\\pi, \\beta)`` is the ``k``-th long-run zero restriction residual
- ``I_n`` is the ``n \\times n`` identity matrix
- ``A_1, \\dots, A_p`` are the VAR(``p``) reduced-form lag coefficient matrices extracted from ``\\pi``
- ``B(\\beta) \\in \\mathcal{B}`` is the unit-diagonal structural impact matrix
- ``\\iota_{i_k}`` and ``\\iota_{\\pi(j_k)}`` are standard basis vectors in ``\\mathbb{R}^n`` for response variable ``i_k`` and ordered shock ``\\pi(j_k)``
"""
function long_run_hypothesis(n::Int, zeros_idx::AbstractVector{<:Tuple{Int,Int}};
                             ordering::Union{Nothing,Symbol,AbstractVector{Int}}=nothing)
    n >= 2 || throw(ArgumentError("long_run_hypothesis requires n ≥ 2, got n=$n"))
    isempty(zeros_idx) && throw(ArgumentError("zeros_idx must be non-empty"))
    zlist = Tuple{Int,Int}[(Int(i), Int(j)) for (i, j) in zeros_idx]
    for (i, j) in zlist
        (1 <= i <= n && 1 <= j <= n) || throw(ArgumentError(
            "long-run restriction ($i, $j) out of bounds for n=$n"))
    end
    length(unique(zlist)) == length(zlist) || throw(ArgumentError(
        "zeros_idx contains duplicate (i, j) entries"))
    # Guard against restricting an entire row or column of Θ_∞ to zero (forces singular B)
    for row in 1:n
        count(z -> z[1] == row, zlist) == n && throw(ArgumentError(
            "Cannot restrict an entire row ($row) of long-run matrix Θ_∞ to zero"))
    end
    for col in 1:n
        count(z -> z[2] == col, zlist) == n && throw(ArgumentError(
            "Cannot restrict an entire column ($col) of long-run matrix Θ_∞ to zero"))
    end
    ord = _lmshyp_validate_ordering(ordering, n)
    q = length(zlist)
    _SVARHypothesisSpec(
        :long_run, n, q, ord, zlist, Tuple{Int,Int,Int}[],
        nothing, nothing, _SVARHypothesisSpec[], nothing,
        "Blanchard–Quah long-run zero restrictions (df=$q)",
    )
end

"""
    horizon_irf_hypothesis(n::Int, restrictions::AbstractVector;
                           ordering=nothing) -> _SVARHypothesisSpec

Construct a hypothesis imposing zero restrictions `(h, i, j)` at specific horizons ``h \\ge 0``
on the structural impulse response matrices ``\\Theta_h(\\pi, \\beta) = \\Phi_h(\\pi) B(\\beta)``:

```math
\\left[\\Phi_h(\\pi) B(\\beta)\\right]_{i, \\pi(j)} = 0
```

where:
- ``\\Phi_h(\\pi)`` is the reduced-form moving-average coefficient matrix at horizon ``h \\ge 0`` (``\\Phi_0 = I_n``)
- ``B(\\beta) \\in \\mathcal{B}`` is the unit-diagonal structural impact matrix
- ``i`` is the response variable index and ``\\pi(j)`` is the shock column index selected by `ordering`

Restrictions at impact (`h == 0`) enforce the LMS ``B \\in \\mathcal{B}`` diagonal interiority guard.
"""
function horizon_irf_hypothesis(n::Int, restrictions::AbstractVector;
                                ordering::Union{Nothing,Symbol,AbstractVector{Int}}=nothing)
    n >= 1 || throw(ArgumentError("n must be ≥ 1, got $n"))
    isempty(restrictions) && throw(ArgumentError("restrictions must be non-empty"))
    rlist = Tuple{Int,Int,Int}[]
    for item in restrictions
        length(item) == 3 || throw(ArgumentError(
            "Each restriction must be a 3-tuple (h, i, j), got $item"))
        h, i, j = Int(item[1]), Int(item[2]), Int(item[3])
        h >= 0 || throw(ArgumentError("Horizon h must be ≥ 0, got $h"))
        (1 <= i <= n && 1 <= j <= n) || throw(ArgumentError(
            "Restriction ($h, $i, $j) out of bounds for n=$n"))
        push!(rlist, (h, i, j))
    end
    length(unique(rlist)) == length(rlist) || throw(ArgumentError(
        "restrictions contains duplicate (h, i, j) entries"))
    ord = _lmshyp_validate_ordering(ordering, n)
    q = length(rlist)
    if ord isa Vector{Int}
        for (h, i, j) in rlist
            if h == 0
                _lmshyp_diagonal_guard(i, j, ord[j], ord)
            end
        end
    end
    if all(r -> r[1] == 0, rlist) && ord isa Vector{Int}
        canon_pairs = Tuple{Int,Int}[(i, ord[j]) for (_, i, j) in rlist]
        R, r = _lmshyp_build_R_from_pairs(canon_pairs, n)
        return _SVARHypothesisSpec(
            :horizon_irf, n, q, ord, Tuple{Int,Int}[], rlist,
            R, r, _SVARHypothesisSpec[], nothing,
            "Horizon IRF zero restrictions at h=0 (df=$q)",
        )
    end
    _SVARHypothesisSpec(
        :horizon_irf, n, q, ord, Tuple{Int,Int}[], rlist,
        nothing, nothing, _SVARHypothesisSpec[], nothing,
        "Horizon IRF zero restrictions (df=$q)",
    )
end

"""
    mixed_svar_hypothesis(specs::AbstractVector) -> _SVARHypothesisSpec
    mixed_svar_hypothesis(; short_run=nothing, long_run=nothing, horizon=nothing) -> _SVARHypothesisSpec

Combine multiple structural SVAR hypothesis specifications (e.g., short-run impact +
Blanchard–Quah long-run + horizon IRF restrictions as in Bjørnland & Leitemo 2009),
verifying that the combined restriction Jacobian has full row rank.
"""
function mixed_svar_hypothesis(specs::AbstractVector)
    isempty(specs) && throw(ArgumentError("specs must be non-empty"))
    flat = _SVARHypothesisSpec[]
    for sp in specs
        sp isa _SVARHypothesisSpec || throw(ArgumentError(
            "All elements of specs must be SVAR hypothesis specifications from builder functions"))
        if sp.kind === :mixed
            append!(flat, sp.subspecs)
        else
            push!(flat, sp)
        end
    end
    ns = Int[sp.n for sp in flat if sp.n !== nothing]
    n_common = if isempty(ns)
        nothing
    else
        allequal(ns) || throw(ArgumentError(
            "All component hypotheses in mixed_svar_hypothesis must share the same n (got $ns)"))
        ns[1]
    end
    q_total = sum(sp -> sp.df, flat)

    # Check linear-beta subset consistency and rank
    if all(sp -> sp.R_beta !== nothing, flat) && n_common !== nothing
        R_all = vcat([sp.R_beta for sp in flat]...)
        r_all = vcat([sp.r_beta for sp in flat]...)
        sv = svdvals(R_all)
        rk = count(>(1e-10 * max(1.0, isempty(sv) ? 0.0 : sv[1])), sv)
        rk == q_total || throw(ArgumentError(
            "Combined linear impact restrictions are rank-deficient (rank $rk < $q_total)"))
        sol = R_all \ r_all
        norm(R_all * sol - r_all) <= 1e-8 * (1.0 + norm(r_all)) || throw(ArgumentError(
            "Combined linear impact restrictions are mutually inconsistent"))
        ord_common = _lmshyp_common_ordering(flat, n_common)
        return _SVARHypothesisSpec(
            :mixed, n_common, q_total, ord_common, Tuple{Int,Int}[], Tuple{Int,Int,Int}[],
            R_all, r_all, flat, nothing,
            "Mixed SVAR restrictions (df=$q_total)",
        )
    end

    # Generic Jacobian rank check at a non-degenerate deterministic (π, β) test point when n is known
    has_custom_or_best = any(sp -> sp.kind === :nonlinear || sp.ordering === :best_match, flat)
    if n_common !== nothing && !has_custom_or_best
        _lmshyp_check_mixed_jacobian(flat, n_common, q_total)
    end

    ord_common = n_common === nothing ? nothing : _lmshyp_common_ordering(flat, n_common)
    _SVARHypothesisSpec(
        :mixed, n_common, q_total, ord_common, Tuple{Int,Int}[], Tuple{Int,Int,Int}[],
        nothing, nothing, flat, nothing,
        "Mixed SVAR restrictions (df=$q_total)",
    )
end

function mixed_svar_hypothesis(; short_run=nothing, long_run=nothing, horizon=nothing)
    specs = _SVARHypothesisSpec[]
    short_run !== nothing && push!(specs, short_run)
    long_run !== nothing && push!(specs, long_run)
    horizon !== nothing && push!(specs, horizon)
    isempty(specs) && throw(ArgumentError(
        "Provide at least one hypothesis via `short_run`, `long_run`, or `horizon`"))
    mixed_svar_hypothesis(specs)
end

"""
    nonlinear_svar_hypothesis(g_func::Function, q_df::Int;
                              description::String="Custom nonlinear SVAR restriction") -> _SVARHypothesisSpec

Construct a custom smooth nonlinear hypothesis ``H_0: g(\\theta) = 0`` with ``q``
restrictions differentiated via `ForwardDiff.jacobian`.

`g_func` may accept either the full stacked parameter vector `theta = [π; β; σ; λ]`
(`g_func(theta)`) or the reduced-form and off-diagonal structural blocks `(pi, beta)`
(`g_func(pi, beta)`). Scalar returns are automatically wrapped when `q_df == 1`.
"""
function nonlinear_svar_hypothesis(g_func::Function, q_df::Int;
                                   description::String="Custom nonlinear SVAR restriction")
    q_df >= 1 || throw(ArgumentError("q_df must be ≥ 1, got $q_df"))
    _SVARHypothesisSpec(
        :nonlinear, nothing, q_df, nothing, Tuple{Int,Int}[], Tuple{Int,Int,Int}[],
        nothing, nothing, _SVARHypothesisSpec[], g_func, description,
    )
end

# =============================================================================
# Restriction Evaluation & Mixed Rank Check Helpers
# =============================================================================

function _lmshyp_common_ordering(flat::Vector{_SVARHypothesisSpec}, n::Int)
    ords = [sp.ordering for sp in flat if sp.ordering !== nothing]
    isempty(ords) && return collect(1:n)
    first_ord = ords[1]
    all(==(first_ord), ords) ? first_ord : collect(1:n)
end

function _lmshyp_eval_user_g(g_func::Function, theta::AbstractVector{Tθ},
                             n_pi::Int, n_beta::Int) where {Tθ<:Real}
    val = if applicable(g_func, @view(theta[1:n_pi]), @view(theta[(n_pi + 1):(n_pi + n_beta)]))
        g_func(@view(theta[1:n_pi]), @view(theta[(n_pi + 1):(n_pi + n_beta)]))
    else
        g_func(theta)
    end
    (val isa AbstractVector || val isa Tuple) ? collect(Tθ, val) : Tθ[val]
end

function _lmshyp_eval_subspec(sp::_SVARHypothesisSpec, theta::AbstractVector{Tθ},
                              n::Int, p::Int, n_pi::Int, n_beta::Int,
                              perm::Vector{Int}) where {Tθ<:Real}
    k = 1 + n * p
    if sp.R_beta !== nothing
        beta_vec = @view theta[(n_pi + 1):(n_pi + n_beta)]
        return Matrix{Tθ}(sp.R_beta) * beta_vec .- Vector{Tθ}(sp.r_beta)
    elseif sp.kind === :long_run
        pi_vec = @view theta[1:n_pi]
        beta_vec = @view theta[(n_pi + 1):(n_pi + n_beta)]
        Pi_T = reshape(pi_vec, n, k)
        A_sum = zeros(Tθ, n, n)
        @inbounds for l in 1:p
            A_sum .+= @view(Pi_T[:, (2 + (l - 1) * n):(1 + l * n)])
        end
        A1 = Matrix{Tθ}(I, n, n) - A_sum
        B = _lms_B_from_beta(beta_vec, n)
        Theta_LR = robust_inv(A1; silent=true) * B
        return Tθ[Theta_LR[i, perm[j]] for (i, j) in sp.zeros_idx]
    elseif sp.kind === :horizon_irf
        pi_vec = @view theta[1:n_pi]
        beta_vec = @view theta[(n_pi + 1):(n_pi + n_beta)]
        Pi_T = reshape(pi_vec, n, k)
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
        B = _lms_B_from_beta(beta_vec, n)
        return Tθ[dot(@view(Phi[h + 1][i, :]), @view(B[:, perm[j]])) for (h, i, j) in sp.horizon_idx]
    elseif sp.kind === :nonlinear
        return _lmshyp_eval_user_g(sp.g_func, theta, n_pi, n_beta)
    elseif sp.kind === :mixed
        parts = [_lmshyp_eval_subspec(sub, theta, n, p, n_pi, n_beta,
                                      sub.ordering isa Vector{Int} ? sub.ordering : perm)
                 for sub in sp.subspecs]
        return vcat(parts...)
    end
    throw(ArgumentError("Unsupported hypothesis kind :$(sp.kind)"))
end

function _lmshyp_make_g_fn(hyp::_SVARHypothesisSpec, n::Int, p::Int, perm::Vector{Int})
    n_pi = n * (1 + n * p)
    n_beta = n * (n - 1)
    if hyp.kind === :mixed
        return theta -> begin
            parts = [_lmshyp_eval_subspec(sub, theta, n, p, n_pi, n_beta,
                                          sub.ordering isa Vector{Int} ? sub.ordering : perm)
                     for sub in hyp.subspecs]
            vcat(parts...)
        end
    else
        return theta -> _lmshyp_eval_subspec(hyp, theta, n, p, n_pi, n_beta, perm)
    end
end

function _lmshyp_check_mixed_jacobian(flat::Vector{_SVARHypothesisSpec}, n::Int, q_total::Int)
    p_test = 2
    n_pi = n * (1 + n * p_test)
    n_beta = n * (n - 1)
    Pi_T = zeros(Float64, n, 1 + n * p_test)
    @inbounds for i in 1:n
        Pi_T[i, 1] = 0.05 * i
        for l in 1:p_test, j in 1:n
            Pi_T[i, 1 + (l - 1) * n + j] = (i == j ? 0.3 / l : 0.04 * sin(1.3 * i + 0.7 * j + l))
        end
    end
    beta_test = Float64[0.11 * cos(1.7 * k + 0.3) for k in 1:n_beta]
    th_test = vcat(vec(Pi_T), beta_test, ones(Float64, n))
    g_test = theta -> begin
        parts = [_lmshyp_eval_subspec(sub, theta, n, p_test, n_pi, n_beta,
                                      sub.ordering isa Vector{Int} ? sub.ordering : collect(1:n))
                 for sub in flat]
        vcat(parts...)
    end
    J = ForwardDiff.jacobian(g_test, th_test)
    sv = svdvals(J)
    rk = count(>(1e-9 * max(1.0, isempty(sv) ? 0.0 : sv[1])), sv)
    rk == q_total || throw(ArgumentError(
        "Combined restrictions in mixed_svar_hypothesis are rank-deficient (Jacobian rank $rk < $q_total)"))
end

function _lmshyp_resolve_best_match(hyp::_SVARHypothesisSpec, res::LMSSVARResult{T},
                                    n::Int, p::Int) where {T<:AbstractFloat}
    if hyp.kind === :mixed && any(sub -> sub.ordering === :best_match, hyp.subspecs)
        resolved_subs = [_lmshyp_resolve_best_match(sub, res, n, p)[1] for sub in hyp.subspecs]
        hmix = mixed_svar_hypothesis(resolved_subs)
        perm = hmix.ordering isa Vector{Int} ? hmix.ordering : collect(1:n)
        return hmix, perm
    end
    hyp.ordering === :best_match || begin
        perm = hyp.ordering isa Vector{Int} ? hyp.ordering : collect(1:n)
        return hyp, perm
    end
    perms = _permutations(n)
    best_perm = collect(1:n)
    best_val = T(Inf)
    found = false
    for perm in perms
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
        g_fn = _lmshyp_make_g_fn(cand, n, p, perm)
        val = sum(abs2, g_fn(res.theta))
        if val < best_val
            best_val = val
            best_perm = perm
            found = true
        end
    end
    found || throw(ArgumentError(
        "No valid permutation satisfies LMS B ∈ B interiority for the requested restrictions"))
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

# =============================================================================
# Constrained ML Solvers (Linear Nullspace & Augmented Lagrangian)
# =============================================================================

function _lmshyp_nullspace_params(R::Matrix{T}, r::Vector{T}) where {T<:AbstractFloat}
    q, n_beta = size(R)
    F = svd(R; full=true)
    sv_tol = max(T(1e-10), eps(T) * max(T(q), T(n_beta))) * max(one(T), isempty(F.S) ? zero(T) : F.S[1])
    rk = count(>(sv_tol), F.S)
    rk == q || throw(ArgumentError("Restriction matrix R_beta is rank-deficient (rank $rk < $q)"))
    beta0 = Vector{T}(F.V[:, 1:rk] * ((F.U[:, 1:rk]' * r) ./ F.S[1:rk]))
    M = rk < n_beta ? Matrix{T}(F.V[:, (rk + 1):end]) : zeros(T, n_beta, 0)
    (beta0, M)
end

function _lmshyp_cholesky_feasible_start(Sigma::AbstractMatrix{T}, perm::Vector{Int},
                                         beta0::Vector{T}, M::Matrix{T},
                                         n::Int) where {T<:AbstractFloat}
    invp = invperm(perm)
    S_p = Matrix{T}(Sigma[perm, perm])
    Lp = Matrix{T}(safe_cholesky(S_p; silent=true))
    D_inv = Diagonal(one(T) ./ max.(abs.(diag(Lp)), T(1e-8)))
    B_p = Lp * D_inv
    @inbounds for i in 1:n
        B_p[i, i] = one(T)
    end
    B_cand = B_p[invp, invp]
    beta_raw = _vecd0(B_cand)
    isempty(M) ? copy(beta0) : Vector{T}(beta0 + M * (transpose(M) * (beta_raw - beta0)))
end

function _lmshyp_solve_linear_restricted(theta_init::Vector{T}, Y_eff::Matrix{T},
                                         X::Matrix{T}, Sigma::AbstractMatrix{T},
                                         dists::Vector{Symbol},
                                         R_beta::Matrix{T}, r_beta::Vector{T},
                                         perm::Vector{Int},
                                         max_iter::Int, g_tol::T;
                                         skip_stage1::Bool=false) where {T<:AbstractFloat}
    T_eff, n = size(Y_eff)
    k = size(X, 2)
    n_pi = n * k
    n_beta = n * (n - 1)
    offs, dims, n_lam = _lms_lambda_offsets(dists)
    n_total = n_pi + n_beta + n + n_lam

    beta0, M = _lmshyp_nullspace_params(R_beta, r_beta)
    n_free = size(M, 2)
    Mt = Matrix{T}(transpose(M))

    pi_init = Vector{T}(@view(theta_init[1:n_pi]))
    beta_init = Vector{T}(@view(theta_init[(n_pi + 1):(n_pi + n_beta)]))
    sig_init = Vector{T}(@view(theta_init[(n_pi + n_beta + 1):(n_pi + n_beta + n)]))
    lam_list_init = _lms_split_lam(dists, @view(theta_init[(n_pi + n_beta + n + 1):n_total]))
    lam_unc_init = _lms_con_to_unc(dists, lam_list_init)

    gamma_proj = n_free > 0 ? Vector{T}(Mt * (beta_init - beta0)) : T[]
    best_gamma = gamma_proj
    best_logsig = log.(max.(sig_init, T(1e-6)))
    best_lamunc = lam_unc_init
    iters_total = 0

    theta_buf = Vector{T}(undef, n_total)

    if !skip_stage1
        beta_proj = n_free > 0 ? beta0 + M * gamma_proj : copy(beta0)
        cand_gammas = Vector{T}[gamma_proj]
        if maximum(abs, beta_proj - beta_init; init=zero(T)) > T(0.05) ||
           abs(det(_lms_B_from_beta(beta_proj, n))) <= T(1e-6)
            beta_chol = _lmshyp_cholesky_feasible_start(Sigma, perm, beta0, M, n)
            gamma_chol = n_free > 0 ? Vector{T}(Mt * (beta_chol - beta0)) : T[]
            if maximum(abs, gamma_chol - gamma_proj; init=zero(T)) > T(1e-3)
                push!(cand_gammas, gamma_chol)
            end
        end

        Pi_init = Matrix{T}(transpose(reshape(pi_init, n, k)))
        U_init = Y_eff - X * Pi_init
        best_s1_nll = T(Inf)
        copyto!(theta_buf, 1, pi_init, 1, n_pi)

        idx_g = 1:n_free
        idx_s = (n_free + 1):(n_free + n)
        idx_l = (n_free + n + 1):(n_free + n + n_lam)
        s1_iters = min(max_iter, 150)
        s1_gtol = max(g_tol, T === Float32 ? T(1e-4) : T(1e-5))

        for g_cand in cand_gammas
            beta_c = n_free > 0 ? beta0 + M * g_cand : copy(beta0)
            B_c = _lms_B_from_beta(beta_c, n)
            abs(det(B_c)) <= T(1e-12) && continue
            B_inv_T = Matrix{T}(transpose(robust_inv(B_c; silent=true)))
            eps_c = U_init * B_inv_T
            sig_c = T[max(std(@view(eps_c[:, i])), T(1e-4)) for i in 1:n]
            logsig_c = log.(sig_c)

            x0_s1 = vcat(g_cand, logsig_c, lam_unc_init)
            function eval_fg_s1(G, x)
                if n_free > 0
                    b = beta0 + M * @view(x[idx_g])
                    copyto!(theta_buf, n_pi + 1, b, 1, n_beta)
                else
                    copyto!(theta_buf, n_pi + 1, beta0, 1, n_beta)
                end
                @inbounds for i in 1:n
                    theta_buf[n_pi + n_beta + i] = exp(clamp(x[n_free + i], T(-20), T(20)))
                end
                lam_c, lam_j = _lms_unc_to_con(dists, @view(x[idx_l]))
                copyto!(theta_buf, n_pi + n_beta + n + 1, lam_c, 1, n_lam)

                L_T, sc, ok = _lms_loglik_and_score(theta_buf, Y_eff, X, dists; U_fixed=U_init)
                ok || return (T(Inf), false)
                if n_free > 0
                    g_b = Mt * @view(sc[(n_pi + 1):(n_pi + n_beta)])
                    @inbounds for j in 1:n_free
                        G[j] = -g_b[j]
                    end
                end
                @inbounds for i in 1:n
                    ls = x[n_free + i]
                    G[n_free + i] = (-20 <= ls <= 20) ?
                        -sc[n_pi + n_beta + i] * theta_buf[n_pi + n_beta + i] : zero(T)
                end
                @inbounds for j in 1:n_lam
                    G[n_free + n + j] = -sc[n_pi + n_beta + n + j] * lam_j[j]
                end
                return (-L_T, true)
            end
            res_s1 = try
                _lms_run_lbfgs(eval_fg_s1, x0_s1, s1_iters, s1_gtol)
            catch err
                _is_recoverable_draw_error(err) || rethrow(err)
                nothing
            end
            res_s1 === nothing && continue
            iters_total += Optim.iterations(res_s1)
            val = T(Optim.minimum(res_s1))
            if isfinite(val) && val < best_s1_nll
                best_s1_nll = val
                x_opt = Vector{T}(Optim.minimizer(res_s1))
                best_gamma = n_free > 0 ? x_opt[idx_g] : T[]
                best_logsig = x_opt[idx_s]
                best_lamunc = x_opt[idx_l]
            end
        end
    end

    # Stage 2: joint constrained optimization over (π, γ, log σ, λ_unc)
    x0_s2 = vcat(pi_init, best_gamma, best_logsig, best_lamunc)
    o_g = n_pi
    o_s = n_pi + n_free
    o_l = n_pi + n_free + n

    function eval_fg_s2(G, x)
        copyto!(theta_buf, 1, x, 1, n_pi)
        if n_free > 0
            b = beta0 + M * @view(x[(o_g + 1):o_s])
            copyto!(theta_buf, n_pi + 1, b, 1, n_beta)
        else
            copyto!(theta_buf, n_pi + 1, beta0, 1, n_beta)
        end
        @inbounds for i in 1:n
            theta_buf[n_pi + n_beta + i] = exp(clamp(x[o_s + i], T(-20), T(20)))
        end
        lam_c, lam_j = _lms_unc_to_con(dists, @view(x[(o_l + 1):end]))
        copyto!(theta_buf, n_pi + n_beta + n + 1, lam_c, 1, n_lam)

        L_T, sc, ok = _lms_loglik_and_score(theta_buf, Y_eff, X, dists)
        ok || return (T(Inf), false)
        @inbounds for j in 1:n_pi
            G[j] = -sc[j]
        end
        if n_free > 0
            g_b = Mt * @view(sc[(n_pi + 1):(n_pi + n_beta)])
            @inbounds for j in 1:n_free
                G[o_g + j] = -g_b[j]
            end
        end
        @inbounds for i in 1:n
            ls = x[o_s + i]
            G[o_s + i] = (-20 <= ls <= 20) ?
                -sc[n_pi + n_beta + i] * theta_buf[n_pi + n_beta + i] : zero(T)
        end
        @inbounds for j in 1:n_lam
            G[o_l + j] = -sc[n_pi + n_beta + n + j] * lam_j[j]
        end
        return (-L_T, true)
    end

    res_s2 = _lms_run_lbfgs(eval_fg_s2, x0_s2, max(max_iter, 500), g_tol)
    iters_total += Optim.iterations(res_s2)
    x_opt = Vector{T}(Optim.minimizer(res_s2))

    pi_opt = x_opt[1:n_pi]
    beta_opt = n_free > 0 ? Vector{T}(beta0 + M * @view(x_opt[(o_g + 1):o_s])) : copy(beta0)
    sig_opt = exp.(clamp.(x_opt[(o_s + 1):o_l], T(-20), T(20)))
    lam_opt, _ = _lms_unc_to_con(dists, @view(x_opt[(o_l + 1):end]))
    theta_r = vcat(pi_opt, beta_opt, sig_opt, lam_opt)
    ll_r = T(T_eff) * (-T(Optim.minimum(res_s2)))
    (theta=theta_r, loglik=ll_r, converged=Optim.converged(res_s2), iters=iters_total)
end

function _lmshyp_solve_auglag(theta_init::Vector{T}, Y_eff::Matrix{T}, X::Matrix{T},
                              dists::Vector{Symbol}, g_fn, q::Int,
                              max_iter::Int, g_tol::T) where {T<:AbstractFloat}
    T_eff, n = size(Y_eff)
    k = size(X, 2)
    n_pi = n * k
    n_beta = n * (n - 1)
    offs, dims, n_lam = _lms_lambda_offsets(dists)
    n_total = n_pi + n_beta + n + n_lam
    o_beta = n_pi
    o_sig = n_pi + n_beta
    o_lam = n_pi + n_beta + n

    pi_0 = Vector{T}(@view(theta_init[1:n_pi]))
    beta_0 = Vector{T}(@view(theta_init[(o_beta + 1):o_sig]))
    logsig_0 = log.(max.(Vector{T}(@view(theta_init[(o_sig + 1):o_lam])), T(1e-6)))
    lam_list_0 = _lms_split_lam(dists, @view(theta_init[(o_lam + 1):n_total]))
    lam_unc_0 = _lms_con_to_unc(dists, lam_list_0)

    x_curr = vcat(pi_0, beta_0, logsig_0, lam_unc_0)
    nu = zeros(T, q)
    rho = T(25.0)
    rho_max = T === Float32 ? T(1e6) : T(1e10)
    c_tol = T === Float32 ? T(1e-5) : T(1e-8)
    g_inner = max(g_tol, T === Float32 ? T(1e-5) : T(1e-8))

    theta_buf = Vector{T}(undef, n_total)
    jac_res = ForwardDiff.DiffResults.JacobianResult(zeros(T, q), theta_buf)
    jac_cfg = ForwardDiff.JacobianConfig(g_fn, theta_buf)

    prev_c = T(Inf)
    iters_total = 0
    converged_flag = false
    ll_curr = T(-Inf)
    theta_curr = copy(theta_init)

    function unpack_x!(th::Vector{T}, x::AbstractVector{T})
        copyto!(th, 1, x, 1, n_pi)
        n_beta > 0 && copyto!(th, o_beta + 1, x, o_beta + 1, n_beta)
        @inbounds for i in 1:n
            th[o_sig + i] = exp(clamp(x[o_sig + i], T(-20), T(20)))
        end
        lam_c, lam_j = _lms_unc_to_con(dists, @view(x[(o_lam + 1):n_total]))
        copyto!(th, o_lam + 1, lam_c, 1, n_lam)
        lam_j
    end

    inner_iters = clamp(max_iter, 150, 400)
    max_outer = 18

    for outer in 1:max_outer
        function eval_fg_aug(G, x)
            lam_j = unpack_x!(theta_buf, x)
            L_T, sc, ok = _lms_loglik_and_score(theta_buf, Y_eff, X, dists)
            ok || return (T(Inf), false)

            ForwardDiff.jacobian!(jac_res, g_fn, theta_buf, jac_cfg)
            c_val = ForwardDiff.DiffResults.value(jac_res)
            J_th = ForwardDiff.DiffResults.jacobian(jac_res)
            (all(isfinite, c_val) && all(isfinite, J_th)) || return (T(Inf), false)

            mult = nu .+ rho .* c_val
            aug_obj = -L_T + dot(nu, c_val) + T(0.5) * rho * dot(c_val, c_val)
            g_aug_th = (-sc) .+ transpose(J_th) * mult

            @inbounds for j in 1:(n_pi + n_beta)
                G[j] = g_aug_th[j]
            end
            @inbounds for i in 1:n
                ls = x[o_sig + i]
                G[o_sig + i] = (-20 <= ls <= 20) ? g_aug_th[o_sig + i] * theta_buf[o_sig + i] : zero(T)
            end
            @inbounds for j in 1:n_lam
                G[o_lam + j] = g_aug_th[o_lam + j] * lam_j[j]
            end
            return (aug_obj, true)
        end

        res_opt = try
            _lms_run_lbfgs(eval_fg_aug, x_curr, inner_iters, g_inner)
        catch err
            _is_recoverable_draw_error(err) || rethrow(err)
            nothing
        end
        res_opt === nothing && break
        iters_total += Optim.iterations(res_opt)
        x_next = Vector{T}(Optim.minimizer(res_opt))

        unpack_x!(theta_buf, x_next)
        L_T, sc, ok = _lms_loglik_and_score(theta_buf, Y_eff, X, dists)
        if !ok
            rho = min(rho * T(5), rho_max)
            continue
        end

        ForwardDiff.jacobian!(jac_res, g_fn, theta_buf, jac_cfg)
        c_val = ForwardDiff.DiffResults.value(jac_res)
        J_th = ForwardDiff.DiffResults.jacobian(jac_res)
        c_norm = maximum(abs, c_val)

        x_curr = x_next
        theta_curr = copy(theta_buf)
        ll_curr = T(T_eff) * L_T
        nu .+= rho .* c_val

        # Check projected KKT gradient in θ-space
        JJt = Matrix{T}(J_th * transpose(J_th))
        nu_ls = -(robust_inv(JJt; silent=true) * (J_th * (-sc)))
        kkt_res = maximum(abs, (-sc) .+ transpose(J_th) * nu_ls)
        kkt_tol = max(g_tol * T(10), T === Float32 ? T(5e-4) : T(1e-6))

        if c_norm <= c_tol && (kkt_res <= kkt_tol || outer >= 10)
            converged_flag = true
            break
        end
        if c_norm > c_tol
            if outer == 1 || c_norm > T(0.25) * prev_c
                rho = min(rho * T(5), rho_max)
            else
                rho = min(rho * T(2), rho_max)
            end
        end
        prev_c = c_norm
    end

    if !converged_flag
        unpack_x!(theta_buf, x_curr)
        c_final = g_fn(theta_buf)
        c_norm = maximum(abs, c_final)
        L_T, _, ok = _lms_loglik_and_score(theta_buf, Y_eff, X, dists)
        if ok
            theta_curr = copy(theta_buf)
            ll_curr = T(T_eff) * L_T
            converged_flag = c_norm <= max(c_tol * T(100), T(1e-4))
        end
    end

    (theta=theta_curr, loglik=ll_curr, converged=converged_flag, iters=iters_total)
end

function _lmshyp_solve_constrained(hyp::_SVARHypothesisSpec, theta_init::Vector{T},
                                   Y_eff::Matrix{T}, X::Matrix{T}, Sigma::AbstractMatrix{T},
                                   dists::Vector{Symbol}, perm::Vector{Int},
                                   g_fn, max_iter::Int, g_tol::T;
                                   skip_stage1::Bool=false) where {T<:AbstractFloat}
    if hyp.R_beta !== nothing
        return _lmshyp_solve_linear_restricted(
            theta_init, Y_eff, X, Sigma, dists,
            Matrix{T}(hyp.R_beta), Vector{T}(hyp.r_beta),
            perm, max_iter, g_tol; skip_stage1=skip_stage1,
        )
    else
        return _lmshyp_solve_auglag(
            theta_init, Y_eff, X, dists, g_fn, hyp.df, max_iter, g_tol,
        )
    end
end

function _lmshyp_polish_unrestricted(theta_init::Vector{T}, res_loglik::T,
                                     Y_eff::Matrix{T}, X::Matrix{T},
                                     dists::Vector{Symbol}, n::Int, n_pi::Int,
                                     n_beta::Int, max_iter::Int, g_tol::T) where {T<:AbstractFloat}
    T_eff = size(Y_eff, 1)
    pi0 = Vector{T}(@view(theta_init[1:n_pi]))
    beta0 = n_beta > 0 ? Vector{T}(@view(theta_init[(n_pi + 1):(n_pi + n_beta)])) : T[]
    logsig0 = log.(max.(Vector{T}(@view(theta_init[(n_pi + n_beta + 1):(n_pi + n_beta + n)])), T(1e-6)))
    lam_list0 = _lms_split_lam(dists, @view(theta_init[(n_pi + n_beta + n + 1):end]))
    lam_unc0 = _lms_con_to_unc(dists, lam_list0)
    opt_u = _lms_optimize_joint(pi0, beta0, logsig0, lam_unc0, Y_eff, X, dists;
                                max_iter=max_iter, g_tol=g_tol)
    ll_u = T(T_eff) * (-opt_u.nll)
    if isfinite(ll_u) && ll_u >= res_loglik - T(1e-8)
        theta_u = vcat(opt_u.pi, opt_u.beta, opt_u.sigma, opt_u.lam_con)
        return (theta=theta_u, loglik=ll_u, converged=opt_u.converged, iters=opt_u.iters)
    end
    (theta=copy(theta_init), loglik=res_loglik, converged=true, iters=0)
end

function _lmshyp_compute_vcov(theta::Vector{T}, Y_eff::Matrix{T}, X::Matrix{T},
                              dists::Vector{Symbol}, vcov_type::Symbol) where {T<:AbstractFloat}
    T_eff = size(Y_eff, 1)
    sh = _lms_score_hessian(theta, Y_eff, X, dists)
    if vcov_type === :hessian
        return _spd_inv(-sh.hessian) ./ T(T_eff)
    elseif vcov_type === :opg
        return _spd_inv(sh.opg) ./ T(T_eff)
    else
        H_inv = _spd_inv(-sh.hessian)
        V_raw = (H_inv * sh.opg * H_inv) ./ T(T_eff)
        return Matrix{T}((V_raw .+ V_raw') .* T(0.5))
    end
end

function _lmshyp_eval_stat(hyp::_SVARHypothesisSpec, theta_u::Vector{T}, ll_u::T,
                           V_theta::Matrix{T}, Y_eff::Matrix{T}, X::Matrix{T},
                           Sigma::AbstractMatrix{T}, dists::Vector{Symbol},
                           perm::Vector{Int}, method::Symbol, vcov_mode::Symbol,
                           max_iter::Int, g_tol::T;
                           theta_r_warm::Union{Nothing,Vector{T}}=nothing) where {T<:AbstractFloat}
    T_eff, n = size(Y_eff)
    k = size(X, 2)
    n_pi = n * k
    n_beta = n * (n - 1)
    p = (k - 1) ÷ n
    g_fn = _lmshyp_make_g_fn(hyp, n, p, perm)
    q = hyp.df

    if method === :wald
        g_hat = Vector{T}(g_fn(theta_u))
        length(g_hat) == q || throw(ArgumentError(
            "g(θ) returned length $(length(g_hat)), expected q_df=$q"))
        G = Matrix{T}(ForwardDiff.jacobian(g_fn, theta_u))
        V_g = Matrix{T}(G * V_theta * transpose(G))
        V_g_sym = Matrix{T}((V_g .+ transpose(V_g)) .* T(0.5))
        V_g_inv = Matrix{T}(robust_inv(V_g_sym; silent=true))
        stat = max(zero(T), T(dot(g_hat, V_g_inv * g_hat)))
        pval = T(ccdf(Chisq(q), Float64(stat)))
        return (stat=stat, pval=pval, theta_r=nothing, ll_r=nothing,
                g_val=g_hat, score_norm=nothing, converged=true, iters=0)
    elseif method === :lr
        warm = theta_r_warm === nothing ? theta_u : theta_r_warm
        rsol = _lmshyp_solve_constrained(hyp, warm, Y_eff, X, Sigma, dists, perm, g_fn,
                                         max_iter, g_tol; skip_stage1=(theta_r_warm !== nothing))
        stat = max(zero(T), T(2) * (ll_u - rsol.loglik))
        pval = T(ccdf(Chisq(q), Float64(stat)))
        g_val = Vector{T}(g_fn(rsol.theta))
        return (stat=stat, pval=pval, theta_r=rsol.theta, ll_r=rsol.loglik,
                g_val=g_val, score_norm=nothing, converged=rsol.converged, iters=rsol.iters)
    else  # :lm
        warm = theta_r_warm === nothing ? theta_u : theta_r_warm
        rsol = _lmshyp_solve_constrained(hyp, warm, Y_eff, X, Sigma, dists, perm, g_fn,
                                         max_iter, g_tol; skip_stage1=(theta_r_warm !== nothing))
        sh = _lms_score_hessian(rsol.theta, Y_eff, X, dists)
        G = Matrix{T}(ForwardDiff.jacobian(g_fn, rsol.theta))
        I_obs = -sh.hessian
        lmin_H = try
            eigmin(Symmetric(I_obs))
        catch
            T(-1)
        end
        I_inv, V_G = if vcov_mode === :opg
            I_i = _spd_inv(sh.opg)
            (I_i, Matrix{T}(G * I_i * transpose(G)))
        elseif vcov_mode === :sandwich
            I_i = _spd_inv(I_obs)
            (I_i, Matrix{T}(G * I_i * sh.opg * I_i * transpose(G)))
        else  # :hessian
            I_i = (isfinite(lmin_H) && lmin_H > T(1e-10)) ? _spd_inv(I_obs) : _spd_inv(sh.opg)
            (I_i, Matrix{T}(G * I_i * transpose(G)))
        end
        V_G_sym = Matrix{T}((V_G .+ transpose(V_G)) .* T(0.5))
        s_proj = G * (I_inv * sh.score)
        stat = max(zero(T), T(T_eff) * T(dot(s_proj, Matrix{T}(robust_inv(V_G_sym; silent=true)) * s_proj)))
        pval = T(ccdf(Chisq(q), Float64(stat)))
        g_val = Vector{T}(g_fn(rsol.theta))
        return (stat=stat, pval=pval, theta_r=rsol.theta, ll_r=rsol.loglik,
                g_val=g_val, score_norm=T(norm(sh.score)), converged=rsol.converged, iters=rsol.iters)
    end
end

# =============================================================================
# Residual Bootstrap Under H₀
# =============================================================================

function _lmshyp_bootstrap_pvalue(hyp::_SVARHypothesisSpec, stat_obs::T,
                                  theta_null::Vector{T}, res::LMSSVARResult{T},
                                  perm::Vector{Int}, method::Symbol, vcov_mode::Symbol,
                                  bootstrap::Symbol, n_boot::Int, rng::AbstractRNG,
                                  max_iter::Int, g_tol::T) where {T<:AbstractFloat}
    n = size(res.B, 1)
    p = res.var_ols.p
    Y_eff, X = construct_var_matrices(res.var_ols.Y, p)
    T_eff, k = size(X)
    n_pi = n * k
    n_beta = n * (n - 1)
    dists = res.distributions

    Pi_r = Matrix{T}(transpose(reshape(@view(theta_null[1:n_pi]), n, k)))
    B_r = _lms_B_from_beta(@view(theta_null[(n_pi + 1):(n_pi + n_beta)]), n)
    sigma_r = Vector{T}(@view(theta_null[(n_pi + n_beta + 1):(n_pi + n_beta + n)]))
    B0_r = B_r * Diagonal(sigma_r)
    B0_r_inv_T = Matrix{T}(transpose(robust_inv(B0_r; silent=true)))

    U_r = Y_eff - X * Pi_r
    Z_r = U_r * B0_r_inv_T
    Z_c = Z_r .- mean(Z_r; dims=1)
    @inbounds for j in 1:n
        sd_j = std(@view(Z_c[:, j]))
        sd_j > T(1e-12) && (Z_c[:, j] ./= sd_j)
    end

    has_skew = any(d -> d in (:skewed_t, :skew_normal, :pearson_iv, :mixture_normal_asym), dists)
    wild_dist = has_skew ? :mammen : :rademacher

    Y_init = Matrix{T}(res.var_ols.Y[1:p, :])
    B_dgp = Matrix{T}(Pi_r)
    B0_r_T = Matrix{T}(transpose(B0_r))

    seeds = rand(rng, UInt64, n_boot)
    boot_stats = fill(T(NaN), n_boot)
    boot_max_iter = min(max_iter, 250)

    Threads.@threads for b in 1:n_boot
        local_rng = Random.Xoshiro(seeds[b])
        boot_stats[b] = _suppress_warnings() do
            try
                Z_star = Matrix{T}(undef, T_eff, n)
                if bootstrap === :iid
                    @inbounds for j in 1:n
                        idx = rand(local_rng, 1:T_eff, T_eff)
                        Z_star[:, j] .= @view(Z_c[idx, j])
                    end
                else
                    @inbounds for j in 1:n
                        w = _wild_weights(local_rng, T_eff, wild_dist, T)
                        Z_star[:, j] .= @view(Z_c[:, j]) .* w
                    end
                end
                U_star = Z_star * B0_r_T
                Y_star = _simulate_var(Y_init, B_dgp, U_star, T_eff + p)
                Y_eff_s, X_s = construct_var_matrices(Y_star, p)
                B_ols_s = X_s \ Y_eff_s
                U_ols_s = Y_eff_s - X_s * B_ols_s
                Sigma_s = Matrix{T}((transpose(U_ols_s) * U_ols_s) ./ T(max(T_eff - k, 1)))

                if method === :lm
                    ev = _lmshyp_eval_stat(hyp, theta_null, zero(T), zeros(T, 0, 0),
                                           Y_eff_s, X_s, Sigma_s, dists, perm,
                                           :lm, vcov_mode, boot_max_iter, g_tol;
                                           theta_r_warm=theta_null)
                    ev.stat
                else
                    u_pol = _lmshyp_polish_unrestricted(theta_null, T(-Inf), Y_eff_s, X_s,
                                                        dists, n, n_pi, n_beta,
                                                        boot_max_iter, g_tol)
                    V_s = method === :wald ?
                        _lmshyp_compute_vcov(u_pol.theta, Y_eff_s, X_s, dists, vcov_mode) :
                        zeros(T, 0, 0)
                    ev = _lmshyp_eval_stat(hyp, u_pol.theta, u_pol.loglik, V_s,
                                           Y_eff_s, X_s, Sigma_s, dists, perm,
                                           method, vcov_mode, boot_max_iter, g_tol;
                                           theta_r_warm=theta_null)
                    ev.stat
                end
            catch err
                _is_recoverable_draw_error(err) || rethrow(err)
                T(NaN)
            end
        end
    end

    valid = filter(isfinite, boot_stats)
    isempty(valid) && return T(NaN)
    T((1 + count(>=(stat_obs), valid)) / (1 + length(valid)))
end

# =============================================================================
# Public Hypothesis Testing Entry Point
# =============================================================================

"""
    test_svar_hypothesis(res::LMSSVARResult{T}, hyp;
                         method=:wald, bootstrap=:none, n_boot=499,
                         rng=Random.default_rng(), seed=nothing,
                         max_iter=500, g_tol=1e-7, vcov_type=nothing) -> SVARHypothesisTest{T}

Test a structural hypothesis `hyp` on a fitted Lanne–Meitz–Saikkonen (2017) non-Gaussian
SVAR result `res`.

Three classical test principles are supported (all asymptotically ``\\chi^2(q)`` under ``H_0``):
- `:wald`: Evaluates ``g(\\hat{\\theta})`` and ``G = \\partial g(\\hat{\\theta})/\\partial \\theta'``
  via `ForwardDiff.jacobian`, propagating the full joint parameter covariance ``\\hat{V}_{\\theta}``
  across ``(\\pi, \\beta)``.
- `:lr`: Likelihood ratio ``\\text{LR} = \\max\\{0, 2(\\ell(\\hat{\\theta}) - \\ell(\\tilde{\\theta}))\\}``
  using exact nullspace reparameterization for linear ``\\beta`` constraints or an
  augmented-Lagrangian `Optim.LBFGS()` solver for nonlinear ``(\\pi, \\beta)`` constraints.
- `:lm`: Rao Lagrange multiplier (score) test at the constrained MLE ``\\tilde{\\theta}``
  using the analytical score and 16-block Hessian/OPG from `_lms_score_hessian`.

The Wald statistic is computed as:

```math
W = g(\\hat{\\theta})' \\left[ G(\\hat{\\theta}) \\hat{V}_{\\theta} G(\\hat{\\theta})' \\right]^{-1} g(\\hat{\\theta})
```

where:
- ``g(\\hat{\\theta})`` is the ``q``-dimensional structural restriction vector at ``\\hat{\\theta}``
- ``G(\\hat{\\theta}) = \\partial g(\\hat{\\theta}) / \\partial \\theta'`` is the ``q \\times k_{\\theta}`` Jacobian
- ``\\hat{V}_{\\theta}`` is the asymptotic covariance matrix of ``\\hat{\\theta}``

# Keyword Arguments
- `method::Symbol=:wald` — `:wald`, `:lr`, or `:lm`
- `bootstrap::Symbol=:none` — `:none`, `:iid`, or `:wild` residual bootstrap under ``H_0``
- `n_boot::Int=499` — number of bootstrap replications when `bootstrap != :none`
- `rng::AbstractRNG=Random.default_rng()` — random number generator for bootstrap
- `seed::Union{Integer,Nothing}=nothing` — reproducible seed override via `_resolve_repro_rng`
- `max_iter::Int=500` — maximum optimizer iterations for constrained ML
- `g_tol::Real=1e-7` — gradient convergence tolerance
- `vcov_type::Union{Nothing,Symbol}=nothing` — covariance type override (`:hessian`, `:sandwich`, `:opg`); `nothing` uses `res.se_type`
"""
function test_svar_hypothesis(res::LMSSVARResult{T}, hyp::_SVARHypothesisSpec;
                              method::Symbol=:wald,
                              bootstrap::Symbol=:none,
                              n_boot::Int=499,
                              rng::AbstractRNG=Random.default_rng(),
                              seed::Union{Integer,Nothing}=nothing,
                              max_iter::Int=500,
                              g_tol::Real=1e-7,
                              vcov_type::Union{Nothing,Symbol}=nothing) where {T<:AbstractFloat}
    method in (:wald, :lr, :lm) || throw(ArgumentError(
        "method must be :wald, :lr, or :lm, got :$method"))
    bootstrap in (:none, :iid, :wild) || throw(ArgumentError(
        "bootstrap must be :none, :iid, or :wild, got :$bootstrap"))
    n_boot >= 1 || throw(ArgumentError("n_boot must be ≥ 1, got $n_boot"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1, got $max_iter"))
    g_tol > 0 || throw(ArgumentError("g_tol must be > 0, got $g_tol"))
    vcov_type !== nothing && !(vcov_type in (:hessian, :sandwich, :opg)) && throw(ArgumentError(
        "vcov_type must be :hessian, :sandwich, :opg, or nothing; got :$vcov_type"))

    n = size(res.B, 1)
    p = res.var_ols.p
    (hyp.n === nothing || hyp.n == n) || throw(ArgumentError(
        "Hypothesis n=$(hyp.n) does not match LMSSVARResult n=$n"))

    hyp_eff, perm = _lmshyp_resolve_best_match(hyp, res, n, p)
    rng_eff = _resolve_repro_rng(rng, seed)

    Y_eff, X = construct_var_matrices(res.var_ols.Y, p)
    T_eff, k = size(X)
    n_pi = n * k
    n_beta = n * (n - 1)
    dists = res.distributions
    g_tol_T = max(T(g_tol), T === Float32 ? T(1e-6) : T(1e-10))
    vcov_mode = vcov_type === nothing ? res.se_type : vcov_type

    theta_u = Vector{T}(res.theta)
    ll_u = res.loglik

    # For :lr on :three_step, polish unrestricted joint MLE so ℓ_u and ℓ_r optimize the same objective
    if method === :lr && res.method !== :full_ml
        pol = _lmshyp_polish_unrestricted(theta_u, ll_u, Y_eff, X, dists, n, n_pi, n_beta,
                                          max_iter, g_tol_T)
        theta_u = pol.theta
        ll_u = pol.loglik
    end

    V_theta = if method === :wald
        vcov_type === nothing ? Matrix{T}(res.vcov) :
            _lmshyp_compute_vcov(theta_u, Y_eff, X, dists, vcov_mode)
    else
        zeros(T, 0, 0)
    end

    ev = _lmshyp_eval_stat(hyp_eff, theta_u, ll_u, V_theta, Y_eff, X, res.var_ols.Sigma,
                           dists, perm, method, vcov_mode, max_iter, g_tol_T)

    boot_pval = if bootstrap !== :none
        theta_null = if ev.theta_r !== nothing
            ev.theta_r
        else
            g_fn = _lmshyp_make_g_fn(hyp_eff, n, p, perm)
            _lmshyp_solve_constrained(hyp_eff, theta_u, Y_eff, X, res.var_ols.Sigma,
                                      dists, perm, g_fn, max_iter, g_tol_T).theta
        end
        _lmshyp_bootstrap_pvalue(hyp_eff, ev.stat, theta_null, res, perm, method,
                                 vcov_mode, bootstrap, n_boot, rng_eff, max_iter, g_tol_T)
    else
        nothing
    end

    manifest = (bootstrap !== :none || seed !== nothing) ?
        capture_manifest(; seed=seed,
            settings=Dict{String,Any}(
                "hypothesis_kind" => String(hyp_eff.kind),
                "method"          => String(method),
                "bootstrap"       => String(bootstrap),
                "n_boot"          => bootstrap === :none ? 0 : n_boot,
                "vcov_type"       => String(vcov_mode),
            )) : nothing

    SVARHypothesisTest{T}(
        hyp_eff.kind,
        method,
        ev.stat,
        ev.pval,
        hyp_eff.df,
        boot_pval,
        bootstrap,
        bootstrap === :none ? 0 : n_boot,
        ll_u,
        ev.ll_r,
        theta_u,
        ev.theta_r,
        ev.g_val,
        ev.score_norm,
        perm,
        vcov_mode,
        ev.converged,
        ev.iters,
        T_eff,
        n,
        hyp_eff.description,
        manifest,
    )
end

# =============================================================================
# LMS §5.2 Residual Specification & Identification Strength Diagnostics
# =============================================================================

function _lmshyp_multivariate_portmanteau(Z::Matrix{T}, lags::Int, p_var::Int) where {T<:AbstractFloat}
    T_eff, n = size(Z)
    L = min(lags, max(1, T_eff - 2))
    Zc = Z .- mean(Z; dims=1)
    C0 = Matrix{T}((transpose(Zc) * Zc) ./ T(T_eff))
    C0_inv = Matrix{T}(robust_inv(C0; silent=true))
    Q_m = zero(T)
    @inbounds for h in 1:L
        Ch = Matrix{T}((transpose(@view(Zc[(h + 1):end, :])) * @view(Zc[1:(end - h), :])) ./ T(T_eff))
        Q_m += tr(transpose(Ch) * C0_inv * Ch * C0_inv) / T(T_eff - h)
    end
    Q_m *= T(T_eff)^2
    Q_m = max(zero(T), Q_m)
    df = L > p_var ? n * n * (L - p_var) : n * n * L
    pval = T(ccdf(Chisq(max(df, 1)), Float64(Q_m)))
    (Q_m, pval, df)
end

"""
    diagnose_nongaussian_svar(res::LMSSVARResult{T}; lags::Int=4) -> NonGaussianDiagnostics{T}

Compute the Lanne–Meitz–Saikkonen (2017, §5.2) residual diagnostic battery on the
standardized structural shocks ``\\hat{z}_t`` from `res`:
- Per-shock skewness, excess kurtosis, Jarque–Bera statistics & p-values (raw and Holm-adjusted),
  plus joint multivariate Jarque–Bera
- Per-shock Ljung–Box `Q(lags)` on levels ``\\hat{z}_{it}``, squares ``\\hat{z}_{it}^2`` (ARCH check),
  and absolute values ``|\\hat{z}_{it}|``
- Multivariate Portmanteau test on ``\\hat{z}_t``
- Per-shock BDS independence test (`m = 2`)
- Pairwise shock and squared-shock correlation matrices plus a joint squared cross-moment
  independence test
- `weak_id_warning`: flagged `true` when `n ≥ 2` and `≥ 2` shocks fail to reject Gaussianity at 5%
"""
function diagnose_nongaussian_svar(res::LMSSVARResult{T}; lags::Int=4) where {T<:AbstractFloat}
    lags >= 1 || throw(ArgumentError("lags must be ≥ 1, got $lags"))
    Z = Matrix{T}(res.shocks)
    T_eff, n = size(Z)
    T_eff >= 6 || throw(ArgumentError("Need at least 6 observations for diagnostics, got $T_eff"))
    L = min(lags, max(1, T_eff - 2))

    skew_vec = Vector{T}(undef, n)
    kurt_vec = Vector{T}(undef, n)
    jb_stats = Vector{T}(undef, n)
    jb_pvals = Vector{T}(undef, n)
    lb_lev_s = Vector{T}(undef, n)
    lb_lev_p = Vector{T}(undef, n)
    lb_sq_s = Vector{T}(undef, n)
    lb_sq_p = Vector{T}(undef, n)
    lb_abs_s = Vector{T}(undef, n)
    lb_abs_p = Vector{T}(undef, n)
    bds_s = Vector{T}(undef, n)
    bds_p = Vector{T}(undef, n)

    for j in 1:n
        s = @view Z[:, j]
        sd = std(s)
        sd > zero(T) || throw(ArgumentError("Shock $j has zero variance"))
        z_std = (s .- mean(s)) ./ sd
        sk = mean(z_std .^ 3)
        ek = mean(z_std .^ 4) - T(3)
        jb = T(T_eff) * (sk^2 / T(6) + ek^2 / T(24))
        skew_vec[j] = sk
        kurt_vec[j] = ek
        jb_stats[j] = jb
        jb_pvals[j] = T(ccdf(Chisq(2), Float64(jb)))

        lb_l = ljung_box_test(s; lags=L)
        lb_lev_s[j] = T(lb_l.statistic)
        lb_lev_p[j] = T(lb_l.pvalue)

        lb_q = ljung_box_squared(s, L)
        lb_sq_s[j] = T(lb_q.statistic)
        lb_sq_p[j] = T(lb_q.pvalue)

        lb_a = ljung_box_test(abs.(s); lags=L)
        lb_abs_s[j] = T(lb_a.statistic)
        lb_abs_p[j] = T(lb_a.pvalue)

        s_bds = T_eff > 2500 ? @view(s[(end - 2499):end]) : s
        bds_res = _suppress_warnings() do
            try
                bds_test(s_bds; m=[2], eps_frac=T(0.7))
            catch
                nothing
            end
        end
        if bds_res === nothing
            bds_s[j] = T(NaN)
            bds_p[j] = T(NaN)
        else
            bds_s[j] = T(bds_res.statistic[1, 1])
            bds_p[j] = T(bds_res.pvalue[1, 1])
        end
    end

    jb_holm = _holm_adjust(jb_pvals)
    n_gauss = count(p -> p >= T(0.05), jb_holm)
    weak_id = (n >= 2) && (n_gauss >= 2)

    mv_jb = jarque_bera_test(Z; method=:multivariate)
    pm_stat, pm_pval, pm_df = _lmshyp_multivariate_portmanteau(Z, L, res.var_ols.p)

    corr_mat = Matrix{T}(cor(Z))
    Z_sq = Z .^ 2
    sq_corr_mat = Matrix{T}(cor(Z_sq))

    cm_df = n * (n - 1) ÷ 2
    cm_stat = zero(T)
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        r2_ij = sq_corr_mat[i, j]
        isfinite(r2_ij) && (cm_stat += T(T_eff) * r2_ij^2)
    end
    cm_pval = cm_df > 0 ? T(ccdf(Chisq(cm_df), Float64(cm_stat))) : one(T)

    NonGaussianDiagnostics{T}(
        jb_stats,
        jb_pvals,
        jb_holm,
        skew_vec,
        kurt_vec,
        T(mv_jb.statistic),
        T(mv_jb.pvalue),
        lb_lev_s,
        lb_lev_p,
        lb_sq_s,
        lb_sq_p,
        lb_abs_s,
        lb_abs_p,
        pm_stat,
        pm_pval,
        pm_df,
        bds_s,
        bds_p,
        corr_mat,
        sq_corr_mat,
        cm_stat,
        cm_pval,
        cm_df,
        n_gauss,
        weak_id,
        L,
        T_eff,
        n,
        copy(res.shock_names),
    )
end
