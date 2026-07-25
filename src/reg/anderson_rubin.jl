# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Anderson-Rubin weak-instrument-robust inference (T244 / #343).

Weak instruments do not merely inflate the first-stage F: they invalidate the 2SLS Wald
confidence interval, whose coverage can be far below nominal no matter how large the
sample. The Anderson-Rubin (1949) test has correct size **regardless of instrument
strength**, and inverting it gives a confidence set with correct coverage. This completes
the weak-IV suite begun with the partial first-stage F, Cragg-Donald, Kleibergen-Paap and
Stock-Yogo critical values.

Provides:
- `AndersonRubinTest` / `anderson_rubin_test` — the test at a hypothesized value
- `AndersonRubinCI` / `anderson_rubin_ci` — the confidence set obtained by inverting it

References:
- Anderson, T. W. & Rubin, H. (1949). Estimation of the Parameters of a Single Equation in
  a Complete System of Stochastic Equations. *Annals of Mathematical Statistics*, 20(1), 46-63.
- Andrews, I., Stock, J. H. & Sun, L. (2019). Weak Instruments in Instrumental Variables
  Regression: Theory and Practice. *Annual Review of Economics*, 11, 727-753.
"""

using LinearAlgebra
using Statistics
using Distributions

# =============================================================================
# Result containers
# =============================================================================

"""
    AndersonRubinTest{T}

Anderson-Rubin test of `H₀: β_endog = β₀`, valid under arbitrarily weak instruments.

# Fields
- `beta0::Vector{T}` — the hypothesized coefficient vector on the endogenous regressors
- `statistic::T` — the AR statistic in `F` form (Wald divided by `q`)
- `p_value::T` — p-value under the reference distribution
- `df1::Int` — `q`, the number of excluded instruments
- `df2::Int` — residual degrees of freedom `n − m` (`F` reference only)
- `distribution::Symbol` — `:F` (homoskedastic) or `:chisq` (robust / clustered)
- `cov_type::Symbol` — covariance used to weight the statistic
- `endog_names::Vector{String}` — names of the endogenous regressors
"""
struct AndersonRubinTest{T<:AbstractFloat}
    beta0::Vector{T}
    statistic::T
    p_value::T
    df1::Int
    df2::Int
    distribution::Symbol
    cov_type::Symbol
    endog_names::Vector{String}
end

function Base.show(io::IO, t::AndersonRubinTest{T}) where {T}
    dist = t.distribution === :F ? "F($(t.df1), $(t.df2))" : "χ²($(t.df1))/$(t.df1)"
    data = Any[
        "H₀"             join(["$(n) = $(_fmt(b))" for (n, b) in zip(t.endog_names, t.beta0)], ", ");
        "AR statistic"   _fmt(t.statistic);
        "Reference"      dist;
        "p-value"        _fmt(t.p_value; digits=4);
        "Covariance"     string(t.cov_type)
    ]
    _pretty_table(io, data;
        title = "Anderson-Rubin Test (weak-instrument robust)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

"""
    AndersonRubinCI{T}

Anderson-Rubin confidence set for the coefficient on a single endogenous regressor,
obtained by inverting [`anderson_rubin_test`](@ref) over a grid.

Unlike a Wald interval, an AR confidence set need not be a bounded interval. Under weak
identification it can be **unbounded** on one or both sides, a **union** of disjoint
components, the **whole real line**, or **empty** (the last signalling that no value of
`β` is consistent with the over-identifying restrictions). All of these are represented
rather than collapsed into a `[lo, hi]`.

# Fields
- `intervals::Vector{Tuple{T,T}}` — the connected components, in increasing order. An
  unbounded side carries `-Inf` / `Inf`
- `is_empty::Bool` — no value in the searched range is accepted
- `is_whole_line::Bool` — every value in the searched range is accepted
- `bounded::Bool` — the set is contained strictly inside the searched range
- `level::T` — nominal coverage
- `critical_value::T` — the AR critical value at `level`
- `grid_lo::T` / `grid_hi::T` — the searched range
- `wald_lower::T` / `wald_upper::T` — the 2SLS Wald interval, for comparison
- `estimate::T` — the 2SLS point estimate
- `df1::Int` — `q`, the number of excluded instruments
- `distribution::Symbol` — `:F` or `:chisq`
- `endog_name::String` — name of the endogenous regressor
"""
struct AndersonRubinCI{T<:AbstractFloat}
    intervals::Vector{Tuple{T,T}}
    is_empty::Bool
    is_whole_line::Bool
    bounded::Bool
    level::T
    critical_value::T
    grid_lo::T
    grid_hi::T
    wald_lower::T
    wald_upper::T
    estimate::T
    df1::Int
    distribution::Symbol
    endog_name::String
end

"""Render a confidence set as `[a, b]`, `[a, b] ∪ [c, d]`, `(-∞, ∞)`, or `∅`."""
function _ar_set_string(ci::AndersonRubinCI)
    ci.is_empty && return "∅ (empty)"
    ci.is_whole_line && return "(-∞, ∞)"
    parts = String[]
    for (a, b) in ci.intervals
        lo = isfinite(a) ? _fmt(a) : "-∞"
        hi = isfinite(b) ? _fmt(b) : "∞"
        push!(parts, (isfinite(a) ? "[" : "(") * lo * ", " * hi * (isfinite(b) ? "]" : ")"))
    end
    return join(parts, " ∪ ")
end

function Base.show(io::IO, ci::AndersonRubinCI{T}) where {T}
    pct = round(Int, 100 * ci.level)
    shape = ci.is_empty ? "empty" :
            ci.is_whole_line ? "whole line" :
            !ci.bounded ? "unbounded" :
            length(ci.intervals) > 1 ? "disjoint ($(length(ci.intervals)) components)" :
            "bounded interval"
    data = Any[
        "Endogenous regressor"  ci.endog_name;
        "2SLS estimate"         _fmt(ci.estimate);
        "$(pct)% AR set"        _ar_set_string(ci);
        "Shape"                 shape;
        "$(pct)% Wald CI"       "[$(_fmt(ci.wald_lower)), $(_fmt(ci.wald_upper))]";
        "Excluded instruments"  ci.df1;
        "Critical value"        _fmt(ci.critical_value);
        "Searched range"        "[$(_fmt(ci.grid_lo)), $(_fmt(ci.grid_hi))]"
    ]
    _pretty_table(io, data;
        title = "Anderson-Rubin Confidence Set",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    if !ci.bounded && !ci.is_empty
        println(io, "Note: the AR set reaches the edge of the searched range — the " *
                    "instruments are weak enough that the set is unbounded. Trust the AR " *
                    "set, not the Wald interval.")
    end
    return nothing
end

# =============================================================================
# Core machinery
# =============================================================================

"""
    _ar_excluded_basis(X, Z, endogenous) -> (Zb, W, q)

Orthonormal basis `Zb` (n × q) of the excluded-instrument space `M_W Z`, where `W` holds
the included exogenous regressors. Working in this basis makes the AR auxiliary regression
an orthogonal projection (`Zb'Zb = I`) and sidesteps the rank deficiency of `M_W Z`.
"""
function _ar_excluded_basis(X::Matrix{T}, Z::Matrix{T},
                            endogenous::Vector{Int}) where {T<:AbstractFloat}
    n = size(Z, 1)
    k = size(X, 2)
    incl = setdiff(1:k, endogenous)
    W = X[:, incl]
    Zt = isempty(incl) ? Z : _partial_out(Z, W)
    U, S, _ = svd(Zt)
    tol = maximum(S) * n * eps(T)
    q = count(>(tol), S)
    q == 0 && throw(ArgumentError(
        "no excluded instruments: every instrument is spanned by the included exogenous " *
        "regressors, so the Anderson-Rubin test is not defined"))
    return Matrix{T}(U[:, 1:q]), W, q
end

"""
    _ar_stat(y, Xen, W, Zb, beta0, cov_type, m; clusters=nothing) -> (stat, df2, dist)

The AR statistic at `beta0`.

Form `ỹ = y − X_en β₀`; under `H₀` this is `Wγ + u`, so the excluded instruments have zero
coefficients in the auxiliary regression of `ỹ` on `[W, Z_excl]`. The statistic is the Wald
test of that restriction, divided by `q`:

- `:ols` — the classical homoskedastic AR, `(γ'γ/q) / (r'r/(n−m))`, referred to `F(q, n−m)`
- `:hc0`–`:hc3` — heteroskedasticity-robust meat `Σᵢ Zbᵢ rᵢ² Zbᵢ'`, referred to `χ²_q/q`
- `:cluster` — cluster meat `Σ_g (Zb_g'r_g)(Zb_g'r_g)'`, referred to `χ²_q/q`

Its size is correct at any instrument strength because `γ = 0` under `H₀` whatever the
first stage looks like.
"""
function _ar_stat(y::Vector{T}, Xen::Matrix{T}, W::Matrix{T}, Zb::Matrix{T},
                  beta0::Vector{T}, cov_type::Symbol, m::Int;
                  clusters::Union{Nothing,AbstractVector}=nothing) where {T<:AbstractFloat}
    n, q = size(Zb)
    ytil = y .- Xen * beta0
    ytil_perp = isempty(W) ? ytil : _partial_out(ytil, W)
    gamma = Zb' * ytil_perp                     # Zb'Zb = I ⇒ these are the OLS coefficients
    resid = ytil_perp .- Zb * gamma
    df2 = n - m

    if cov_type === :ols
        df2 > 0 || throw(ArgumentError("n − m must be positive for the homoskedastic AR test"))
        s2 = dot(resid, resid) / T(df2)
        stat = dot(gamma, gamma) / (T(q) * s2)
        return stat, df2, :F
    end

    V = zeros(T, q, q)
    if cov_type === :cluster
        clusters === nothing && throw(ArgumentError(
            "cov_type=:cluster requires the clusters keyword"))
        length(clusters) == n || throw(ArgumentError("clusters must have length $n"))
        uc = unique(clusters)
        G = length(uc)
        G >= 2 || throw(ArgumentError("Need at least 2 clusters, got $G"))
        score = Vector{T}(undef, q)
        for g in uc
            idx = findall(==(g), clusters)
            fill!(score, zero(T))
            @inbounds for i in idx
                ri = resid[i]
                for a in 1:q
                    score[a] += Zb[i, a] * ri
                end
            end
            BLAS.ger!(one(T), score, score, V)
        end
        V .*= T(G) / T(G - 1) * T(n - 1) / T(max(df2, 1))
    else
        @inbounds for i in 1:n
            w2 = resid[i]^2
            for a in 1:q, b in 1:q
                V[a, b] += Zb[i, a] * Zb[i, b] * w2
            end
        end
        cov_type === :hc1 && df2 > 0 && (V .*= T(n) / T(df2))
    end
    wald = dot(gamma, robust_inv(Symmetric(V)) * gamma)
    return wald / T(q), df2, :chisq
end

_ar_pvalue(stat::T, q::Int, df2::Int, dist::Symbol) where {T} =
    dist === :F ? T(ccdf(FDist(q, df2), stat)) : T(ccdf(Chisq(q), stat * q))

_ar_critical(level::Real, q::Int, df2::Int, dist::Symbol, ::Type{T}) where {T} =
    dist === :F ? T(quantile(FDist(q, df2), level)) : T(quantile(Chisq(q), level) / q)

# =============================================================================
# Model unpacking
# =============================================================================

"""Pull `(y, X, Z, endogenous, cov_type, varnames)` out of an IV-fitted model."""
function _ar_unpack(m::RegModel{T}) where {T<:AbstractFloat}
    m.Z === nothing && throw(ArgumentError(
        "the model was not estimated by IV — anderson_rubin_test requires a model from " *
        "estimate_iv"))
    m.endogenous === nothing && throw(ArgumentError("the model carries no endogenous indices"))
    return (Vector{T}(m.y), Matrix{T}(m.X), Matrix{T}(m.Z),
            Vector{Int}(m.endogenous), m.cov_type, m.varnames)
end

function _ar_unpack(m::PanelIVModel{T}) where {T<:AbstractFloat}
    endog = [findfirst(==(nm), m.varnames) for nm in m.endog_names]
    any(isnothing, endog) && throw(ArgumentError(
        "could not locate endogenous regressors $(m.endog_names) among $(m.varnames)"))
    return (Vector{T}(m.y), Matrix{T}(m.X), Matrix{T}(m.Z),
            Vector{Int}(endog), m.cov_type, m.varnames)
end

# =============================================================================
# Public API
# =============================================================================

"""
    anderson_rubin_test(model, beta0; cov_type=model.cov_type, clusters=nothing)
        -> AndersonRubinTest

Anderson-Rubin (1949) test of `H₀: β_endog = beta0` on an IV-fitted model, valid under
arbitrarily weak instruments.

Subtracting the hypothesized effect gives `ỹ = y − X_endog β₀`, which under `H₀` depends on
the excluded instruments only through the error. The AR statistic is the Wald test that
the excluded instruments' coefficients are jointly zero in the auxiliary regression of `ỹ`
on the full instrument set. Because that restriction holds under `H₀` whatever the first
stage looks like, the test's size is correct at any instrument strength — unlike the 2SLS
Wald test.

# Arguments
- `model` — a [`RegModel`](@ref) from `estimate_iv`, or a [`PanelIVModel`](@ref)
- `beta0` — hypothesized value(s): a scalar for one endogenous regressor, else a vector

# Keywords
- `cov_type::Symbol` — `:ols` (classical `F`), `:hc0`–`:hc3`, or `:cluster`; defaults to
  the covariance the model was fitted with, so the test matches the reported standard errors
- `clusters::AbstractVector` — required when `cov_type=:cluster`

# Returns
[`AndersonRubinTest`](@ref).

# References
- Anderson, T. W. & Rubin, H. (1949). *Annals of Mathematical Statistics*, 20(1), 46-63.
"""
function anderson_rubin_test(model, beta0;
                             cov_type::Union{Nothing,Symbol}=nothing,
                             clusters::Union{Nothing,AbstractVector}=nothing)
    y, X, Z, endog, model_cov, varnames = _ar_unpack(model)
    T = eltype(y)
    ct = cov_type === nothing ? model_cov : cov_type
    ct in (:ols, :hc0, :hc1, :hc2, :hc3, :cluster) || throw(ArgumentError(
        "cov_type must be :ols, :hc0, :hc1, :hc2, :hc3, or :cluster; got :$ct"))

    b0 = beta0 isa Number ? T[T(beta0)] : Vector{T}(beta0)
    length(b0) == length(endog) || throw(ArgumentError(
        "beta0 has length $(length(b0)) but the model has $(length(endog)) endogenous " *
        "regressors"))

    Zb, W, q = _ar_excluded_basis(X, Z, endog)
    stat, df2, dist = _ar_stat(y, X[:, endog], W, Zb, b0, ct, size(Z, 2); clusters=clusters)
    p = _ar_pvalue(stat, q, df2, dist)
    return AndersonRubinTest{T}(b0, stat, p, q, df2, dist, ct, varnames[endog])
end

"""
    anderson_rubin_ci(model; level=0.95, n_grid=1001, span=20, grid=nothing,
                      cov_type=model.cov_type, clusters=nothing) -> AndersonRubinCI

Anderson-Rubin confidence set for the coefficient on a **single** endogenous regressor,
obtained by inverting [`anderson_rubin_test`](@ref): the set of `β₀` the AR test does not
reject at `level`.

The set is **not forced to be an interval**. With weak instruments it is routinely
unbounded on one or both sides, and with over-identification it can be a union of disjoint
components or empty. The result reports whichever shape actually obtains, plus the 2SLS
Wald interval for comparison — the contrast between the two is the diagnostic.

The default search range spans `β̂ ± span·se` around the 2SLS estimate. Components touching
the edge of that range are reported as unbounded; widen `span` or pass an explicit `grid`
to search further. Component boundaries strictly inside the range are refined by bisection.

# Keywords
- `level::Real=0.95` — nominal coverage
- `n_grid::Int=1001` — grid points over the search range
- `span::Real=20` — half-width of the default range, in 2SLS standard errors
- `grid::AbstractVector` — explicit grid, overriding `span`/`n_grid`
- `cov_type`, `clusters` — as in [`anderson_rubin_test`](@ref)

# Returns
[`AndersonRubinCI`](@ref).
"""
function anderson_rubin_ci(model;
                           level::Real=0.95,
                           n_grid::Int=1001,
                           span::Real=20,
                           grid::Union{Nothing,AbstractVector}=nothing,
                           cov_type::Union{Nothing,Symbol}=nothing,
                           clusters::Union{Nothing,AbstractVector}=nothing)
    y, X, Z, endog, model_cov, varnames = _ar_unpack(model)
    T = eltype(y)
    length(endog) == 1 || throw(ArgumentError(
        "anderson_rubin_ci inverts over a single endogenous coefficient; this model has " *
        "$(length(endog)). Use anderson_rubin_test at specific vectors instead."))
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1)"))
    n_grid >= 5 || throw(ArgumentError("n_grid must be at least 5"))

    ct = cov_type === nothing ? model_cov : cov_type
    j = endog[1]
    Zb, W, q = _ar_excluded_basis(X, Z, endog)
    Xen = X[:, endog]
    m_inst = size(Z, 2)

    arstat(b0) = _ar_stat(y, Xen, W, Zb, T[b0], ct, m_inst; clusters=clusters)[1]
    _, df2, dist = _ar_stat(y, Xen, W, Zb, T[zero(T)], ct, m_inst; clusters=clusters)
    crit = _ar_critical(level, q, df2, dist, T)

    beta_hat = T(coef(model)[j])
    se_hat = T(stderror(model)[j])
    z = T(quantile(Normal(), (1 + level) / 2))
    wald_lo, wald_hi = beta_hat - z * se_hat, beta_hat + z * se_hat

    gvec = if grid === nothing
        half = isfinite(se_hat) && se_hat > 0 ? T(span) * se_hat : T(span)
        collect(range(beta_hat - half, beta_hat + half; length=n_grid))
    else
        sort(Vector{T}(grid))
    end
    glo, ghi = first(gvec), last(gvec)

    inside = [arstat(b) <= crit for b in gvec]

    if !any(inside)
        return AndersonRubinCI{T}(Tuple{T,T}[], true, false, true, T(level), crit,
                                  glo, ghi, wald_lo, wald_hi, beta_hat, q, dist,
                                  varnames[j])
    end
    if all(inside)
        return AndersonRubinCI{T}([(T(-Inf), T(Inf))], false, true, false, T(level), crit,
                                  glo, ghi, wald_lo, wald_hi, beta_hat, q, dist,
                                  varnames[j])
    end

    # Refine a boundary between an accepted and a rejected grid point
    function refine(a, b)
        for _ in 1:40
            mid = (a + b) / 2
            if arstat(mid) <= crit
                a = mid
            else
                b = mid
            end
        end
        return a          # last accepted point
    end

    intervals = Tuple{T,T}[]
    i = 1
    N = length(gvec)
    while i <= N
        if inside[i]
            first_in = i
            while i < N && inside[i+1]
                i += 1
            end
            last_in = i
            lo = first_in == 1 ? T(-Inf) : refine(gvec[first_in], gvec[first_in-1])
            hi = last_in == N ? T(Inf) : refine(gvec[last_in], gvec[last_in+1])
            push!(intervals, (lo, hi))
        end
        i += 1
    end
    bounded = all(t -> isfinite(t[1]) && isfinite(t[2]), intervals)

    return AndersonRubinCI{T}(intervals, false, false, bounded, T(level), crit,
                              glo, ghi, wald_lo, wald_hi, beta_hat, q, dist, varnames[j])
end
