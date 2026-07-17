# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
ARDL(p, q₁…q_k) estimation by OLS on lagged levels, IC-based lag selection on a
common effective sample, long-run coefficients with delta-method standard
errors, and the conditional error-correction re-parameterisation. Bounds testing
lives in `bounds.jl`.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Deterministics per PSS (2001) case
# =============================================================================

# Which deterministics enter the levels regression as regressors:
#   case 1 → none; case 2 → intercept (restricted, but still estimated);
#   case 3 → intercept; case 4 → intercept + trend (trend restricted);
#   case 5 → intercept + trend.
function _case_trend(case::Int)
    case == 1 && return :none
    (case == 2 || case == 3) && return :const
    (case == 4 || case == 5) && return :trend
    throw(ArgumentError("case must be an integer in 1:5; got $case"))
end

# =============================================================================
# Design-matrix construction (levels ARDL)
# =============================================================================

"""
    _ardl_design(y, X0, p, q, case; row_start) -> (Xd, yeff, ar_idx, x_idx, det_idx,
                                                    intercept_col, trend_col, names)

Build the levels-ARDL design matrix over rows `row_start:N` (1-based time index),
where `N = length(y)`. Column order is `[deterministics; y_{t-1..p}; x_j lags]`.
`row_start` must exceed `max(p, maximum(q))` so every lag is in-sample; passing a
common `row_start` across candidates aligns the sample for comparable IC values.
"""
function _ardl_design(y::AbstractVector{T}, X0::AbstractMatrix{T}, p::Int,
                      q::AbstractVector{<:Integer}, case::Int;
                      row_start::Int, xnames::Vector{String},
                      yname::String) where {T<:AbstractFloat}
    N = length(y)
    k = size(X0, 2)
    rows = row_start:N
    n = length(rows)

    cols = Vector{Vector{T}}()
    names = String[]
    det_idx = Int[]
    intercept_col = 0
    trend_col = 0

    trend = _case_trend(case)
    if trend == :const || trend == :trend
        push!(cols, ones(T, n)); push!(names, "(Intercept)")
        intercept_col = length(cols); push!(det_idx, intercept_col)
    end
    if trend == :trend
        push!(cols, T.(collect(rows))); push!(names, "trend")
        trend_col = length(cols); push!(det_idx, trend_col)
    end

    ar_idx = Int[]
    for i in 1:p
        push!(cols, T[y[t-i] for t in rows]); push!(names, "L$(i).$(yname)")
        push!(ar_idx, length(cols))
    end

    x_idx = Vector{Vector{Int}}(undef, k)
    for j in 1:k
        idxj = Int[]
        for l in 0:q[j]
            push!(cols, T[X0[t-l, j] for t in rows])
            push!(names, l == 0 ? xnames[j] : "L$(l).$(xnames[j])")
            push!(idxj, length(cols))
        end
        x_idx[j] = idxj
    end

    Xd = reduce(hcat, cols)
    yeff = T[y[t] for t in rows]
    (Xd, yeff, ar_idx, x_idx, det_idx, intercept_col, trend_col, names)
end

# =============================================================================
# OLS core + Gaussian IC
# =============================================================================

"""Return `(coef, vcov, resid, fitted, ssr, sigma2)` for OLS of `yeff` on `Xd`."""
function _ardl_ols(Xd::Matrix{T}, yeff::Vector{T}) where {T<:AbstractFloat}
    n, K = size(Xd)
    XtX = Symmetric(Xd' * Xd)
    XtXinv = Matrix{T}(robust_inv(XtX))
    coef = XtXinv * (Xd' * yeff)
    fitted = Xd * coef
    resid = yeff .- fitted
    ssr = sum(abs2, resid)
    dof = max(n - K, 1)
    sigma2 = ssr / dof
    vcov = sigma2 .* XtXinv
    (coef, vcov, resid, fitted, ssr, sigma2)
end

"""Gaussian log-likelihood, AIC, BIC for `ssr` with `K` parameters over `n` obs."""
function _ardl_ic(ssr::T, n::Int, K::Int) where {T<:AbstractFloat}
    sigma2 = ssr / n
    ll = -T(n) / 2 * (log(2 * T(π)) + log(max(sigma2, floatmin(T))) + one(T))
    aic = -2 * ll + 2 * (K + 1)          # +1 for the estimated variance
    bic = -2 * ll + log(T(n)) * (K + 1)
    (ll, aic, bic)
end

# =============================================================================
# Grid search over (p, q₁…q_k) on a COMMON effective sample
# =============================================================================

"""
    _ardl_grid(y, X0, max_p, max_q, case, ic; xnames, yname) -> (p, q)

Search `p ∈ 1:max_p`, `q_j ∈ 0:max_q` (all regressors), scoring every candidate
on the **same** effective sample (`row_start = max(max_p, max_q) + 1`) so the AIC/BIC
values are directly comparable, and return the minimiser. Trimming to the common
sample is essential: scoring each candidate on its own longer sample biases
selection toward shorter lags.
"""
function _ardl_grid(y::AbstractVector{T}, X0::AbstractMatrix{T}, max_p::Int,
                    max_q::Int, case::Int, ic::Symbol;
                    xnames::Vector{String}, yname::String) where {T<:AbstractFloat}
    k = size(X0, 2)
    row_start = max(max_p, max_q) + 1
    n_common = length(y) - row_start + 1
    n_common > 0 || throw(ArgumentError("sample too short for max_p=$max_p, max_q=$max_q"))

    best_score = T(Inf)
    best_p = 1
    best_q = fill(0, k)
    # iterate q ∈ (0:max_q)^k via a mixed-radix counter
    qcombos = Iterators.product(ntuple(_ -> 0:max_q, k)...)
    for p in 1:max_p, qtup in qcombos
        q = collect(Int, qtup)
        Xd, yeff, = _ardl_design(y, X0, p, q, case; row_start=row_start,
                                 xnames=xnames, yname=yname)
        _, _, _, _, ssr, = _ardl_ols(Xd, yeff)
        ll, aic, bic = _ardl_ic(ssr, n_common, size(Xd, 2))
        score = ic == :bic ? bic : aic
        if score < best_score
            best_score = score
            best_p = p
            best_q = q
        end
    end
    (best_p, best_q)
end

# =============================================================================
# Public estimator
# =============================================================================

"""
    estimate_ardl(y, X; p=:auto, q=:auto, max_p=4, max_q=4, ic=:aic, case=3,
                  trend=:none, xnames=nothing, yname="y") -> ARDLModel{T}

Estimate an autoregressive distributed-lag model `ARDL(p, q₁…q_k)` by OLS on the
lagged **levels** of `y` and the columns of `X`.

# Arguments
- `y::AbstractVector` — dependent variable (length `N`).
- `X::AbstractMatrix` — `N × k` matrix of distributed-lag regressors (no intercept column).

# Keywords
- `p` — autoregressive order (`Int ≥ 1`) or `:auto` for IC selection.
- `q` — distributed-lag order: an `Int` applied to every regressor, a length-`k`
  vector, or `:auto` for IC selection.
- `max_p::Int=4`, `max_q::Int=4` — grid bounds when `p`/`q` are `:auto`.
- `ic::Symbol=:aic` — information criterion for `:auto` selection (`:aic` or `:bic`).
  Candidates are scored on a common effective sample so the IC values are comparable.
- `case::Int=3` — Pesaran–Shin–Smith (2001) deterministic case `∈ 1:5`
  (I: none; II: restricted intercept; III: unrestricted intercept; IV: unrestricted
  intercept + restricted trend; V: unrestricted intercept + trend). `case` sets which
  deterministics enter and which bounds table [`bounds_test`](@ref) uses.
- `trend::Symbol=:none` — informational label; the deterministics that actually enter
  are governed by `case`.
- `xnames`, `yname` — labels for display.

# Returns
`ARDLModel{T}` (`T = float(eltype(y))`) carrying the OLS coefficients/vcov, residuals,
information criteria, the coefficient-block bookkeeping, and a cached long-run block.

# References
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). *Journal of Applied Econometrics* 16, 289–326.
"""
function estimate_ardl(y::AbstractVector{T}, X::AbstractMatrix{T};
                       p::Union{Symbol,Integer}=:auto,
                       q::Union{Symbol,Integer,AbstractVector}=:auto,
                       max_p::Int=4, max_q::Int=4, ic::Symbol=:aic,
                       case::Int=3, trend::Symbol=:none,
                       xnames::Union{Nothing,Vector{String}}=nothing,
                       yname::AbstractString="y") where {T<:AbstractFloat}
    N, k = size(X)
    length(y) == N || throw(ArgumentError("y has length $(length(y)); X has $N rows"))
    (1 <= case <= 5) || throw(ArgumentError("case must be in 1:5; got $case"))
    ic in (:aic, :bic) || throw(ArgumentError("ic must be :aic or :bic; got :$ic"))
    max_p >= 1 || throw(ArgumentError("max_p must be ≥ 1"))
    max_q >= 0 || throw(ArgumentError("max_q must be ≥ 0"))
    vnames = xnames === nothing ? ["x$j" for j in 1:k] : xnames
    length(vnames) == k || throw(ArgumentError("xnames must have length $k"))
    yn = String(yname)

    # ---- resolve (p, q) ----
    selected = (p === :auto) || (q === :auto)
    pp::Int = 0
    qq::Vector{Int} = Int[]
    if p === :auto || q === :auto
        gp, gq = _ardl_grid(y, X, max_p, max_q, case, ic; xnames=vnames, yname=yn)
        pp = p === :auto ? gp : Int(p)
        qq = q === :auto ? gq :
             q isa Integer ? fill(Int(q), k) : collect(Int, q)
    else
        pp = Int(p)
        qq = q isa Integer ? fill(Int(q), k) : collect(Int, q)
    end
    pp >= 1 || throw(ArgumentError("p must be ≥ 1; got $pp"))
    length(qq) == k || throw(ArgumentError("q must have length $k; got $(length(qq))"))
    all(>=(0), qq) || throw(ArgumentError("every q must be ≥ 0"))

    # ---- final fit on the SELECTED model's own effective sample ----
    L = max(pp, maximum(qq))
    row_start = L + 1
    row_start <= N || throw(ArgumentError("effective sample empty: need N > $(L)"))
    Xd, yeff, ar_idx, x_idx, det_idx, intercept_col, trend_col, names =
        _ardl_design(y, X, pp, qq, case; row_start=row_start, xnames=vnames, yname=yn)
    n, K = size(Xd)
    n > K || throw(ArgumentError("effective sample ($n) ≤ #coefficients ($K); reduce lags"))

    coef, vcov, resid, fitted, ssr, sigma2 = _ardl_ols(Xd, yeff)
    ll, aic, bic = _ardl_ic(ssr, n, K)

    m = ARDLModel{T}(yeff, Xd, coef, vcov, resid, fitted, pp, qq, case,
                     _case_trend(case), ssr, sigma2, ll, aic, bic, n, K,
                     ar_idx, x_idx, det_idx, intercept_col, trend_col,
                     names, vnames, yn, selected, ic, nothing)
    m.longrun = _compute_long_run(m)
    m
end

# Integer / mixed-eltype convenience
estimate_ardl(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    estimate_ardl(float.(collect(y)), float.(collect(X)); kwargs...)
estimate_ardl(y::AbstractVector, x::AbstractVector; kwargs...) =
    estimate_ardl(y, reshape(collect(x), :, 1); kwargs...)

# =============================================================================
# Long-run coefficients (delta method)
# =============================================================================

"""Compute the long-run block θ_j = (Σ_ℓ β_{jℓ})/(1 − Σ_i φ_i) with delta-method SEs."""
function _compute_long_run(m::ARDLModel{T}) where {T<:AbstractFloat}
    b = m.coef
    V = m.vcov
    K = m.K
    denom = one(T) - sum(@view b[m.ar_idx])          # 1 − Σφ  (= −α, positive when stable)
    k = length(m.x_idx)
    theta = zeros(T, k)
    se = zeros(T, k)
    for j in 1:k
        num = sum(@view b[m.x_idx[j]])                # Σ_ℓ β_{jℓ}
        theta[j] = num / denom
        # Jacobian g of θ_j w.r.t. the full coefficient vector.
        g = zeros(T, K)
        for c in m.x_idx[j]
            g[c] = one(T) / denom                     # ∂θ/∂β_{jℓ}
        end
        for c in m.ar_idx
            g[c] = num / denom^2                      # ∂θ/∂φ_i = num/denom²
        end
        var = dot(g, V * g)
        se[j] = sqrt(max(var, zero(T)))
    end
    ARDLLongRun{T}(theta, se, denom, copy(m.xnames))
end

"""
    long_run(m::ARDLModel) -> ARDLLongRun

Long-run (level) multipliers `θ̂_j = (Σ_ℓ β̂_{jℓ}) / (1 − Σ_i φ̂_i)` with
delta-method standard errors from the OLS variance matrix. The denominator
`1 − Σφ̂` is the negative of the error-correction speed of adjustment; a value
near zero (a near-unit-root `y`) inflates the multipliers and their SEs.
"""
long_run(m::ARDLModel) = m.longrun === nothing ? _compute_long_run(m) : m.longrun

# =============================================================================
# Conditional error-correction (ECM) re-parameterisation
# =============================================================================

"""
    ecm_form(m::ARDLModel) -> NamedTuple

Recover the conditional error-correction re-parameterisation of the levels ARDL
without re-fitting (the ECM is an exact linear map of the levels model, so the
fit is identical). Returns the speed of adjustment `alpha = Σφ̂ − 1` (the
coefficient on `y_{t-1}` in the ECM form) with its standard error and `t`-ratio,
and the long-run level term.

# Returned fields
- `alpha::T`, `alpha_se::T`, `alpha_t::T`: speed of adjustment and its `t`-ratio.
- `longrun::ARDLLongRun`: the cached long-run block.
"""
function ecm_form(m::ARDLModel{T}) where {T<:AbstractFloat}
    b = m.coef
    V = m.vcov
    # α = Σφ − 1, a linear functional r'b − 1 with r = indicator of AR columns.
    r = zeros(T, m.K)
    for c in m.ar_idx
        r[c] = one(T)
    end
    alpha = dot(r, b) - one(T)
    alpha_se = sqrt(max(dot(r, V * r), zero(T)))
    alpha_t = alpha / alpha_se
    (alpha=alpha, alpha_se=alpha_se, alpha_t=alpha_t, longrun=long_run(m))
end

# =============================================================================
# StatsAPI interface
# =============================================================================

StatsAPI.coef(m::ARDLModel) = m.coef
StatsAPI.vcov(m::ARDLModel) = m.vcov
StatsAPI.residuals(m::ARDLModel) = m.residuals
StatsAPI.predict(m::ARDLModel) = m.fitted
StatsAPI.nobs(m::ARDLModel) = m.n
StatsAPI.dof(m::ARDLModel) = m.K
StatsAPI.dof_residual(m::ARDLModel) = m.n - m.K
StatsAPI.loglikelihood(m::ARDLModel) = m.loglik
StatsAPI.aic(m::ARDLModel) = m.aic
StatsAPI.bic(m::ARDLModel) = m.bic
StatsAPI.islinear(::ARDLModel) = true
StatsAPI.stderror(m::ARDLModel) = sqrt.(diag(m.vcov))

# =============================================================================
# Display
# =============================================================================

const _ARDL_CASE_DESC = Dict(
    1 => "I — no intercept, no trend",
    2 => "II — restricted intercept, no trend",
    3 => "III — unrestricted intercept, no trend",
    4 => "IV — unrestricted intercept, restricted trend",
    5 => "V — unrestricted intercept + trend",
)

function Base.show(io::IO, m::ARDLModel{T}) where {T}
    qstr = join(m.q, ", ")
    spec = Any[
        "Model"          "ARDL($(m.p); $(qstr))";
        "Dependent"      m.yname;
        "Regressors"     length(m.q);
        "Case"           get(_ARDL_CASE_DESC, m.case, string(m.case));
        "Selection"      m.selected ? uppercase(string(m.ic)) * " grid" : "fixed";
        "Observations"   m.n;
        "Coefficients"   m.K;
        "σ̂²"             _fmt(m.sigma2);
        "Log-lik."       _fmt(m.loglik; digits=2);
        "AIC"            _fmt(m.aic; digits=2);
        "BIC"            _fmt(m.bic; digits=2)
    ]
    _pretty_table(io, spec; title="ARDL Model", column_labels=["Specification", ""],
                  alignment=[:l, :r])

    _coef_table(io, "Coefficients (levels form)", m.coefnames, m.coef, stderror(m);
                dist=:t, dof_r=max(m.n - m.K, 1))

    lr = long_run(m)
    _coef_table(io, "Long-run coefficients", lr.varnames, lr.theta, lr.se; dist=:z)

    ecm = ecm_form(m)
    ecm_data = Any[
        "Speed of adj. α (ΔECₜ₋₁)"  _fmt(ecm.alpha);
        "  Std.Err."                _fmt(ecm.alpha_se);
        "  t-ratio"                 _fmt(ecm.alpha_t);
        "1 − Σφ̂ (long-run denom.)"  _fmt(lr.denom)
    ]
    _pretty_table(io, ecm_data; title="Error-correction form",
                  column_labels=["", "Value"], alignment=[:l, :r])
    _sig_legend(io)
end

"""
    report(m::ARDLModel)

Print the ARDL specification, the levels-form coefficient table, the long-run
coefficients with delta-method standard errors, and the error-correction summary
(speed of adjustment α).
"""
report(m::ARDLModel) = show(stdout, m)
report(io::IO, m::ARDLModel) = show(io, m)
