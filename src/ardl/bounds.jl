# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Pesaran–Shin–Smith (2001) bounds test for the existence of a level (long-run)
relationship. The `F`-statistic is a joint Wald/F test that all error-correction
level terms are zero; the `t`-statistic is the Dickey–Fuller-type `t`-ratio on
the lagged level of the dependent variable. Both are compared **only** to the
asymptotic I(0)/I(1) critical-value bounds — the null distributions are
non-standard, so no p-value is defined or reported.

The critical-value tables are transcribed from Pesaran, Shin & Smith (2001),
*Journal of Applied Econometrics* 16, 289–326, Tables CI(i)–CI(v) (`F`) and
CII(i)/CII(iii)/CII(v) (`t`). Row `k` is the number of distributed-lag
regressors; columns are the four significance levels 10 / 5 / 2.5 / 1 %.
Case III (unrestricted intercept, no trend) is the default; its k=1 5% `F`-bounds
(4.94, 5.73) and `t`-bounds (−2.86, −3.22) are the canonical, widely-reproduced
reference values asserted in the test suite.
"""

using LinearAlgebra

# Significance levels the tables are indexed by (descending significance risk).
const _PSS_LEVELS = [0.10, 0.05, 0.025, 0.01]

# -----------------------------------------------------------------------------
# F-statistic bounds — PSS (2001) Tables CI(i)–CI(v).
# Each matrix row is one k (1..10); the 8 columns are
#   [I0_10, I1_10, I0_5, I1_5, I0_2.5, I1_2.5, I0_1, I1_1].
# -----------------------------------------------------------------------------

# Case I — no intercept, no trend  (CI(i))
const _PSS_F1 = [
    2.44 3.28  3.15 4.11  3.88 4.92  4.81 6.02
    2.17 3.19  2.72 3.83  3.22 4.50  3.88 5.30
    2.01 3.10  2.45 3.63  2.87 4.16  3.42 4.84
    1.90 3.01  2.26 3.48  2.62 3.90  3.07 4.44
    1.81 2.93  2.14 3.34  2.44 3.71  2.82 4.21
    1.75 2.87  2.04 3.24  2.32 3.59  2.66 4.05
    1.70 2.83  1.97 3.18  2.22 3.49  2.54 3.91
    1.66 2.79  1.91 3.11  2.15 3.40  2.45 3.79
    1.63 2.75  1.86 3.05  2.08 3.33  2.34 3.68
    1.60 2.72  1.82 2.99  2.02 3.27  2.26 3.60
]

# Case II — restricted intercept, no trend  (CI(ii))
const _PSS_F2 = [
    3.02 3.51  3.62 4.16  4.18 4.79  4.94 5.58
    2.63 3.35  3.10 3.87  3.55 4.38  4.13 5.00
    2.37 3.20  2.79 3.67  3.15 4.08  3.65 4.66
    2.20 3.09  2.56 3.49  2.88 3.87  3.29 4.37
    2.08 3.00  2.39 3.38  2.70 3.73  3.06 4.15
    1.99 2.94  2.27 3.28  2.55 3.61  2.88 3.99
    1.92 2.89  2.17 3.21  2.43 3.51  2.73 3.90
    1.85 2.85  2.11 3.15  2.33 3.42  2.62 3.77
    1.80 2.80  2.04 3.08  2.24 3.35  2.50 3.68
    1.76 2.77  1.98 3.04  2.18 3.28  2.41 3.61
]

# Case III — unrestricted intercept, no trend  (CI(iii)) — DEFAULT
const _PSS_F3 = [
    4.04 4.78  4.94 5.73  5.77 6.68  6.84 7.84
    3.17 4.14  3.79 4.85  4.41 5.52  5.15 6.36
    2.72 3.77  3.23 4.35  3.69 4.89  4.29 5.61
    2.45 3.52  2.86 4.01  3.25 4.49  3.74 5.06
    2.26 3.35  2.62 3.79  2.96 4.18  3.41 4.68
    2.12 3.23  2.45 3.61  2.75 3.99  3.15 4.43
    2.03 3.13  2.32 3.50  2.60 3.84  2.96 4.26
    1.95 3.06  2.22 3.39  2.48 3.70  2.79 4.10
    1.88 2.99  2.14 3.30  2.37 3.60  2.65 3.97
    1.83 2.94  2.06 3.24  2.27 3.50  2.54 3.86
]

# Case IV — unrestricted intercept, restricted trend  (CI(iv))
const _PSS_F4 = [
    4.05 4.49  4.68 5.15  5.30 5.83  6.10 6.73
    3.38 4.02  3.88 4.61  4.37 5.16  5.00 5.90
    2.97 3.74  3.38 4.23  3.80 4.68  4.30 5.23
    2.68 3.53  3.05 3.97  3.40 4.36  3.81 4.92
    2.49 3.38  2.81 3.76  3.11 4.13  3.50 4.63
    2.33 3.25  2.63 3.62  2.90 3.94  3.27 4.39
    2.22 3.17  2.50 3.50  2.76 3.81  3.07 4.23
    2.13 3.09  2.38 3.41  2.62 3.70  2.93 4.06
    2.05 3.02  2.30 3.33  2.52 3.60  2.79 3.93
    1.98 2.97  2.22 3.26  2.42 3.52  2.68 3.84
]

# Case V — unrestricted intercept + trend  (CI(v))
const _PSS_F5 = [
    5.59 6.26  6.56 7.30  7.46 8.27  8.74 9.63
    4.19 5.06  4.87 5.85  5.49 6.59  6.34 7.52
    3.47 4.45  4.01 4.98  4.52 5.46  5.17 6.36
    3.03 4.06  3.47 4.57  3.89 5.07  4.40 5.72
    2.75 3.79  3.12 4.25  3.47 4.67  3.93 5.23
    2.53 3.59  2.87 4.00  3.19 4.38  3.60 4.90
    2.38 3.45  2.69 3.83  2.98 4.16  3.34 4.63
    2.26 3.34  2.55 3.68  2.82 3.99  3.15 4.43
    2.16 3.24  2.43 3.56  2.67 3.87  2.97 4.24
    2.07 3.16  2.33 3.46  2.56 3.76  2.84 4.15
]

const _PSS_F = Dict(1 => _PSS_F1, 2 => _PSS_F2, 3 => _PSS_F3, 4 => _PSS_F4, 5 => _PSS_F5)

# -----------------------------------------------------------------------------
# t-statistic bounds — PSS (2001) Tables CII(i)/CII(iii)/CII(v).
# The bounds test t-ratio is only tabulated for the three cases without a
# restricted deterministic (I, III, V); cases II and IV have no standard t-bounds.
# Columns: [I0_10, I1_10, I0_5, I1_5, I0_2.5, I1_2.5, I0_1, I1_1].
# The I(0) (lower) bound depends only on the case, not on k.
# -----------------------------------------------------------------------------

# Case I — no intercept, no trend  (CII(i))
const _PSS_T1 = [
    -1.62 -2.28  -1.95 -2.60  -2.25 -2.90  -2.58 -3.22
    -1.62 -2.68  -1.95 -3.02  -2.25 -3.31  -2.58 -3.66
    -1.62 -3.00  -1.95 -3.33  -2.25 -3.64  -2.58 -3.97
    -1.62 -3.26  -1.95 -3.60  -2.25 -3.89  -2.58 -4.23
    -1.62 -3.49  -1.95 -3.83  -2.25 -4.12  -2.58 -4.44
    -1.62 -3.70  -1.95 -4.04  -2.25 -4.34  -2.58 -4.67
    -1.62 -3.90  -1.95 -4.23  -2.25 -4.54  -2.58 -4.88
    -1.62 -4.09  -1.95 -4.43  -2.25 -4.72  -2.58 -5.07
    -1.62 -4.26  -1.95 -4.61  -2.25 -4.89  -2.58 -5.25
    -1.62 -4.42  -1.95 -4.76  -2.25 -5.06  -2.58 -5.44
]

# Case III — unrestricted intercept, no trend  (CII(iii)) — DEFAULT
const _PSS_T3 = [
    -2.57 -2.91  -2.86 -3.22  -3.13 -3.50  -3.43 -3.82
    -2.57 -3.21  -2.86 -3.53  -3.13 -3.80  -3.43 -4.10
    -2.57 -3.46  -2.86 -3.78  -3.13 -4.05  -3.43 -4.37
    -2.57 -3.66  -2.86 -3.99  -3.13 -4.26  -3.43 -4.60
    -2.57 -3.86  -2.86 -4.19  -3.13 -4.46  -3.43 -4.79
    -2.57 -4.04  -2.86 -4.38  -3.13 -4.66  -3.43 -4.99
    -2.57 -4.23  -2.86 -4.57  -3.13 -4.85  -3.43 -5.19
    -2.57 -4.40  -2.86 -4.72  -3.13 -5.02  -3.43 -5.37
    -2.57 -4.56  -2.86 -4.88  -3.13 -5.18  -3.43 -5.54
    -2.57 -4.69  -2.86 -5.03  -3.13 -5.34  -3.43 -5.68
]

# Case V — unrestricted intercept + trend  (CII(v))
const _PSS_T5 = [
    -3.13 -3.40  -3.41 -3.69  -3.65 -3.96  -3.96 -4.26
    -3.13 -3.63  -3.41 -3.95  -3.65 -4.20  -3.96 -4.53
    -3.13 -3.84  -3.41 -4.16  -3.65 -4.42  -3.96 -4.73
    -3.13 -4.04  -3.41 -4.36  -3.65 -4.62  -3.96 -4.96
    -3.13 -4.21  -3.41 -4.52  -3.65 -4.79  -3.96 -5.13
    -3.13 -4.37  -3.41 -4.69  -3.65 -4.96  -3.96 -5.31
    -3.13 -4.53  -3.41 -4.85  -3.65 -5.12  -3.96 -5.47
    -3.13 -4.68  -3.41 -4.99  -3.65 -5.26  -3.96 -5.61
    -3.13 -4.82  -3.41 -5.13  -3.65 -5.40  -3.96 -5.75
    -3.13 -4.96  -3.41 -5.27  -3.65 -5.54  -3.96 -5.89
]

const _PSS_T = Dict(1 => _PSS_T1, 3 => _PSS_T3, 5 => _PSS_T5)

# -----------------------------------------------------------------------------
# Table lookup
# -----------------------------------------------------------------------------

"""Return `(f_lower, f_upper, t_lower, t_upper)` bound vectors (one entry per level in
`_PSS_LEVELS`) for a given `case`/`k`. `t_*` are `NaN` when the case has no t-bounds."""
function _pss_bounds(case::Int, k::Int)
    (1 <= case <= 5) || throw(ArgumentError("case must be in 1:5; got $case"))
    haskey(_PSS_F, case) || throw(ArgumentError("no F-bounds tabulated for case $case"))
    FT = _PSS_F[case]
    (1 <= k <= size(FT, 1)) ||
        throw(ArgumentError("PSS bounds are tabulated for k ∈ 1:$(size(FT,1)); got k=$k"))
    frow = @view FT[k, :]
    f_lower = collect(frow[1:2:end])
    f_upper = collect(frow[2:2:end])
    if haskey(_PSS_T, case)
        trow = @view _PSS_T[case][k, :]
        t_lower = collect(trow[1:2:end])
        t_upper = collect(trow[2:2:end])
    else
        t_lower = fill(NaN, length(_PSS_LEVELS))
        t_upper = fill(NaN, length(_PSS_LEVELS))
    end
    (f_lower, f_upper, t_lower, t_upper)
end

# -----------------------------------------------------------------------------
# Restriction machinery + public bounds_test
# -----------------------------------------------------------------------------

"""Build the level-restriction matrix `R` and target `r0` (`R·coef = r0` under H₀ of no
level relationship) plus the row `r_ρ` selecting the lagged-y level functional Σφ."""
function _bounds_restrictions(m::ARDLModel{T}) where {T<:AbstractFloat}
    K = m.K
    rows = Vector{Vector{T}}()
    r0 = T[]

    # ρ = 0  ⇔  Σφ = 1  (Dickey–Fuller-type restriction on the lagged y level)
    r_rho = zeros(T, K)
    for c in m.ar_idx
        r_rho[c] = one(T)
    end
    push!(rows, copy(r_rho)); push!(r0, one(T))

    # restricted deterministic under the null: intercept (case II) or trend (case IV)
    if m.case == 2 && m.intercept_col > 0
        e = zeros(T, K); e[m.intercept_col] = one(T); push!(rows, e); push!(r0, zero(T))
    elseif m.case == 4 && m.trend_col > 0
        e = zeros(T, K); e[m.trend_col] = one(T); push!(rows, e); push!(r0, zero(T))
    end

    # δ_j = Σ_ℓ β_{jℓ} = 0 for every regressor
    for idxj in m.x_idx
        e = zeros(T, K)
        for c in idxj
            e[c] = one(T)
        end
        push!(rows, e); push!(r0, zero(T))
    end

    R = permutedims(reduce(hcat, rows))               # (m_r × K)
    (R, r0, r_rho)
end

_decide(stat, lo, hi) = stat > hi ? :cointegrated :
                        stat < lo ? :not_cointegrated : :inconclusive
# t-statistic decision: more negative than the (negative) I(1) bound ⇒ cointegrated.
_decide_t(t, lo, hi) = t < hi ? :cointegrated :
                       t > lo ? :not_cointegrated : :inconclusive

"""
    bounds_test(m::ARDLModel; case=m.case, level=0.05, cv_source=:pss) -> ARDLBoundsTest

Pesaran–Shin–Smith (2001) bounds test on an [`ARDLModel`](@ref).

The `F`-statistic is the joint Wald/F test that every error-correction **level**
coefficient is zero — the lagged dependent level `y_{t-1}` (equivalently the
restriction `Σφ = 1`) and each lagged regressor level `x_{j,t-1}` (`Σ_ℓ β_{jℓ}=0`),
plus the restricted intercept (case II) or trend (case IV). The `t`-statistic is
the classical `t`-ratio on the lagged `y` level.

Both statistics are compared **only** to the tabulated I(0)/I(1) bounds: a value
above the I(1) upper bound ⇒ `:cointegrated`; below the I(0) lower bound ⇒
`:not_cointegrated`; in between ⇒ `:inconclusive`. **No p-value is defined** — the
null distributions are non-standard functionals of Brownian motion, not `F`/`t`.

# Keywords
- `case::Int=m.case` — deterministic case selecting the bounds table (defaults to the
  model's own case). Must match how the model's deterministics were specified.
- `level::Real=0.05` — significance level used for the returned decision (`0.10`, `0.05`,
  `0.025`, or `0.01`); every level's bounds are stored regardless.
- `cv_source::Symbol=:pss` — critical-value source. Only `:pss` (the asymptotic PSS 2001
  tables) is bundled; `:narayan` (finite-sample bounds) is reserved for a future build.

# References
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). *Journal of Applied Econometrics* 16, 289–326.
"""
function bounds_test(m::ARDLModel{T}; case::Int=m.case, level::Real=0.05,
                     cv_source::Symbol=:pss) where {T<:AbstractFloat}
    if cv_source == :narayan
        throw(ArgumentError("cv_source=:narayan (Narayan 2005 finite-sample bounds) is " *
              "not bundled in this build; use cv_source=:pss (asymptotic PSS 2001 bounds). " *
              "Finite-sample tables were omitted rather than transcribed unverified."))
    end
    cv_source == :pss ||
        throw(ArgumentError("cv_source must be :pss (or :narayan); got :$cv_source"))
    (1 <= case <= 5) || throw(ArgumentError("case must be in 1:5; got $case"))
    lev_idx = findfirst(≈(level), _PSS_LEVELS)
    lev_idx === nothing &&
        throw(ArgumentError("level must be one of $(_PSS_LEVELS); got $level"))

    k = length(m.q)
    R, r0, r_rho = _bounds_restrictions(m)
    b = m.coef
    V = m.vcov
    m_r = size(R, 1)

    # Joint Wald / F on the level block: (Rb−r0)'(RVR')⁻¹(Rb−r0) / m_r.
    d = R * b .- r0
    RVR = Symmetric(R * V * transpose(R))
    wald = dot(d, Matrix{T}(robust_inv(RVR)) * d)
    fstat = wald / m_r

    # t-ratio on the lagged y level: (Σφ − 1) / se(Σφ).
    rho = dot(r_rho, b) - one(T)
    se_rho = sqrt(max(dot(r_rho, V * r_rho), zero(T)))
    tstat = rho / se_rho

    f_lower, f_upper, t_lower, t_upper = _pss_bounds(case, k)
    f_decision = _decide(fstat, f_lower[lev_idx], f_upper[lev_idx])
    t_decision = isnan(t_lower[lev_idx]) ? :undefined :
                 _decide_t(tstat, t_lower[lev_idx], t_upper[lev_idx])

    ARDLBoundsTest{T}(T(fstat), T(tstat), k, case, cv_source, T.(_PSS_LEVELS),
                     T.(f_lower), T.(f_upper), T.(t_lower), T.(t_upper),
                     T(level), f_decision, t_decision, m.n)
end

# =============================================================================
# Display
# =============================================================================

# Format a significance level as a percent label, keeping the ".5" for 2.5%.
function _pct_label(l::Real)
    v = 100 * l
    r = round(Int, v)
    return isapprox(v, r; atol=1e-9) ? "$(r)%" : "$(round(v; digits=1))%"
end

const _DECISION_LABEL = Dict(
    :cointegrated => "cointegrated (reject H₀: no level relationship)",
    :not_cointegrated => "no cointegration (do not reject H₀)",
    :inconclusive => "inconclusive (statistic between the bounds)",
    :undefined => "not defined for this case",
)

function Base.show(io::IO, r::ARDLBoundsTest{T}) where {T}
    lev_pct = _pct_label(r.level)
    li = findfirst(≈(r.level), r.levels)

    spec = Any[
        "Statistic"       "PSS (2001) bounds test";
        "Case"            get(_ARDL_CASE_DESC, r.case, string(r.case));
        "Regressors (k)"  r.k;
        "Observations"    r.n;
        "CV source"       uppercase(string(r.cv_source));
        "Decision level"  lev_pct
    ]
    _pretty_table(io, spec; title="ARDL Bounds Test", column_labels=["", ""],
                  alignment=[:l, :r])

    # F-statistic against its bounds at the decision level
    fdata = Any[
        "F-statistic"         _fmt(r.fstat);
        "I(0) bound ($(lev_pct))"  _fmt(r.f_lower[li]);
        "I(1) bound ($(lev_pct))"  _fmt(r.f_upper[li]);
        "Decision (F)"        _DECISION_LABEL[r.f_decision]
    ]
    _pretty_table(io, fdata; title="Bounds F-test (level relationship)",
                  column_labels=["", "Value"], alignment=[:l, :r])

    # t-statistic block (only when tabulated for the case)
    if !isnan(r.t_lower[li])
        tdata = Any[
            "t-statistic"         _fmt(r.tstat);
            "I(0) bound ($(lev_pct))"  _fmt(r.t_lower[li]);
            "I(1) bound ($(lev_pct))"  _fmt(r.t_upper[li]);
            "Decision (t)"        _DECISION_LABEL[r.t_decision]
        ]
        _pretty_table(io, tdata; title="Bounds t-test (on lagged y level)",
                      column_labels=["", "Value"], alignment=[:l, :r])
    end

    # Full bounds table across all significance levels
    lvls = [_pct_label(l) for l in r.levels]
    tbl = Matrix{Any}(undef, length(r.levels), 5)
    for i in eachindex(r.levels)
        tbl[i, 1] = lvls[i]
        tbl[i, 2] = _fmt(r.f_lower[i]); tbl[i, 3] = _fmt(r.f_upper[i])
        tbl[i, 4] = isnan(r.t_lower[i]) ? "—" : _fmt(r.t_lower[i])
        tbl[i, 5] = isnan(r.t_upper[i]) ? "—" : _fmt(r.t_upper[i])
    end
    _pretty_table(io, tbl; title="Critical-value bounds (PSS 2001)",
                  column_labels=["Level", "F I(0)", "F I(1)", "t I(0)", "t I(1)"],
                  alignment=[:l, :r, :r, :r, :r])

    _show_note(io, "Non-standard test: compare the statistic to the I(0)/I(1) bounds " *
                   "only — no p-value is defined. Above I(1) ⇒ level relationship; " *
                   "below I(0) ⇒ none; between ⇒ inconclusive.")
end

"""
    report(r::ARDLBoundsTest)

Print the bounds `F`- and `t`-statistics with their bracketing I(0)/I(1) critical
values and the cointegration decision. Emits **no p-value** — the bounds test is a
non-standard test compared only to the tabulated bounds.
"""
report(r::ARDLBoundsTest) = show(stdout, r)
report(io::IO, r::ARDLBoundsTest) = show(io, r)
