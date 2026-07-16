# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for the ARDL (autoregressive distributed lag) module
(`src/ardl/`).

This file defines the shared result types for the ARDL / bounds-testing path.
The module scaffold (this file + `estimation.jl` + `bounds.jl`) is created by
EV-08 (#416) and extended by later nonlinear-ARDL (NARDL, EV-09) and
pooled/mean-group panel-ARDL (PMG, EV-23) estimators. The design keeps the
coefficient-block bookkeeping (`ar_idx`, `x_idx`, `det_idx`) explicit so those
extensions can reuse `long_run` / `bounds_test` on a re-parameterised design.
"""

# =============================================================================
# ARDLLongRun — long-run multipliers with delta-method standard errors
# =============================================================================

"""
    ARDLLongRun{T<:AbstractFloat}

Long-run (level) coefficients of an [`ARDLModel`](@ref),

```math
\\hat\\theta_j = \\frac{\\sum_\\ell \\hat\\beta_{j\\ell}}{1 - \\sum_i \\hat\\varphi_i},
```

with delta-method standard errors from the OLS variance matrix. Returned by
[`long_run`](@ref); prints as a `_coef_table`-style block.

# Fields
- `theta::Vector{T}`: long-run multipliers (`k`-vector).
- `se::Vector{T}`: delta-method standard errors.
- `denom::T`: the common denominator `1 − Σφ̂` (the negative speed of adjustment).
- `varnames::Vector{String}`: regressor labels.
"""
struct ARDLLongRun{T<:AbstractFloat}
    theta::Vector{T}
    se::Vector{T}
    denom::T
    varnames::Vector{String}
end

# =============================================================================
# ARDLModel — single-equation ARDL(p, q₁…q_k) estimated by OLS on lagged levels
# =============================================================================

"""
    ARDLModel{T<:AbstractFloat}

Autoregressive distributed-lag model

```math
y_t = c + \\delta t + \\sum_{i=1}^{p} \\varphi_i\\, y_{t-i}
        + \\sum_{j=1}^{k} \\sum_{\\ell=0}^{q_j} \\beta_{j\\ell}\\, x_{j,t-\\ell} + u_t,
```

estimated by OLS on the lagged **levels** over the effective sample
`t = L+1 … N`, with `L = max(p, max_j q_j)`. The deterministic terms `c` and
`δ t` enter according to the Pesaran–Shin–Smith (2001) `case` (see
[`estimate_ardl`](@ref)). Long-run coefficients, the conditional error-correction
re-parameterisation, and the bounds test are recovered from the stored OLS
`coef` / `vcov` through the index-bookkeeping fields.

# Fields
- `y::Vector{T}`: dependent variable over the effective sample (`n`-vector).
- `X::Matrix{T}`: levels design matrix (`n × K`), column order
  `[deterministics; y lags; x₁ lags; …; x_k lags]`.
- `coef::Vector{T}`, `vcov::Matrix{T}`: OLS coefficients and `σ̂²(X'X)⁻¹`.
- `residuals::Vector{T}`, `fitted::Vector{T}`: OLS residuals / fitted values.
- `p::Int`: autoregressive order; `q::Vector{Int}`: distributed-lag orders per regressor (`k`-vector).
- `case::Int`: PSS (2001) deterministic case `∈ 1:5`.
- `trend::Symbol`: `:none`, `:const`, or `:trend` — the deterministics actually in `X`.
- `ssr::T`, `sigma2::T`: residual sum of squares and `σ̂² = ssr/(n−K)`.
- `loglik::T`, `aic::T`, `bic::T`: Gaussian log-likelihood and information criteria.
- `n::Int`, `K::Int`: effective sample size and number of coefficients.
- `ar_idx::Vector{Int}`: columns of `X` holding `y_{t-1..p}`.
- `x_idx::Vector{Vector{Int}}`: columns of `X` holding each regressor's `0..q_j` lags.
- `det_idx::Vector{Int}`: columns of `X` holding deterministics.
- `intercept_col::Int`, `trend_col::Int`: column of the intercept / trend (`0` if absent).
- `coefnames::Vector{String}`: coefficient labels aligned with `coef`.
- `xnames::Vector{String}`: regressor labels (`k`-vector); `yname::String`: dependent-variable label.
- `selected::Bool`: whether `(p, q)` were chosen by IC grid search; `ic::Symbol`: `:aic` / `:bic`.
- `longrun::Union{Nothing,ARDLLongRun{T}}`: cached long-run block.

# References
- Pesaran, M. H. & Shin, Y. (1999). *Econometrics and Economic Theory in the 20th Century*.
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). *Journal of Applied Econometrics* 16, 289–326.
"""
mutable struct ARDLModel{T<:AbstractFloat}
    y::Vector{T}
    X::Matrix{T}
    coef::Vector{T}
    vcov::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    p::Int
    q::Vector{Int}
    case::Int
    trend::Symbol
    ssr::T
    sigma2::T
    loglik::T
    aic::T
    bic::T
    n::Int
    K::Int
    ar_idx::Vector{Int}
    x_idx::Vector{Vector{Int}}
    det_idx::Vector{Int}
    intercept_col::Int
    trend_col::Int
    coefnames::Vector{String}
    xnames::Vector{String}
    yname::String
    selected::Bool
    ic::Symbol
    longrun::Union{Nothing,ARDLLongRun{T}}
end

# =============================================================================
# ARDLBoundsTest — Pesaran–Shin–Smith (2001) bounds test
# =============================================================================

"""
    ARDLBoundsTest{T<:AbstractFloat}

Result of the Pesaran–Shin–Smith (2001) bounds test for the existence of a
level relationship. The `F`-statistic is the joint (non-standard) Wald/F test
that **all** error-correction level terms are zero — the lagged dependent level
`y_{t-1}` and every lagged regressor level `x_{j,t-1}` (plus the restricted
intercept/trend in cases II/IV). The `t`-statistic is the (Dickey–Fuller-type)
`t`-ratio on the `y_{t-1}` level term.

Both statistics are compared **only** to the asymptotic I(0)/I(1) critical-value
bounds (PSS Tables CI(i)–CI(v) for `F`, CII(i)/CII(iii)/CII(v) for `t`); the
distributions are non-standard, so **no p-value is defined or reported**.

# Fields
- `fstat::T`: bounds `F`-statistic.
- `tstat::T`: bounds `t`-statistic on the lagged level of `y`.
- `k::Int`: number of distributed-lag regressors (indexes the CV tables).
- `case::Int`: PSS deterministic case `∈ 1:5`.
- `cv_source::Symbol`: critical-value source (`:pss`).
- `levels::Vector{T}`: significance levels of the tabulated bounds (`[0.10,0.05,0.025,0.01]`).
- `f_lower::Vector{T}`, `f_upper::Vector{T}`: I(0)/I(1) `F`-bounds at each level.
- `t_lower::Vector{T}`, `t_upper::Vector{T}`: I(0)/I(1) `t`-bounds (`NaN` when undefined for the case).
- `level::T`: significance level used for the reported decision.
- `f_decision::Symbol`: `:cointegrated`, `:not_cointegrated`, or `:inconclusive` (from `F`).
- `t_decision::Symbol`: same, from the `t`-statistic (`:undefined` for cases II/IV).
- `n::Int`: effective sample size.

# References
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). *Journal of Applied Econometrics* 16, 289–326.
- Narayan, P. K. (2005). *Applied Economics* 37, 1979–1990.
"""
struct ARDLBoundsTest{T<:AbstractFloat}
    fstat::T
    tstat::T
    k::Int
    case::Int
    cv_source::Symbol
    levels::Vector{T}
    f_lower::Vector{T}
    f_upper::Vector{T}
    t_lower::Vector{T}
    t_upper::Vector{T}
    level::T
    f_decision::Symbol
    t_decision::Symbol
    n::Int
end

# =============================================================================
# Nonlinear ARDL (NARDL) — Shin–Yu–Greenwood-Nimmo (2014) — EV-09 (#417)
# =============================================================================

"""
    NARDLModel{T<:AbstractFloat}

Nonlinear autoregressive distributed-lag (NARDL) model of Shin, Yu &
Greenwood-Nimmo (2014). Each regressor flagged `asymmetric` is decomposed into
positive/negative partial sums

```math
x^{+}_{j,t} = \\sum_{s\\le t}\\max(\\Delta x_{j,s},0), \\qquad
x^{-}_{j,t} = \\sum_{s\\le t}\\min(\\Delta x_{j,s},0), \\qquad x^{+}_{j,0}=x^{-}_{j,0}=0,
```

and the pair `(x⁺_j, x⁻_j)` replaces `x_j` in an ARDL conditional error-correction
model. The enlarged design is estimated by the reused EV-08 [`estimate_ardl`](@ref)
machinery, so the long-run coefficients, the ECM re-parameterisation, and the
[`bounds_test`](@ref) all operate on the split regressors. Crucially, an
asymmetric regressor contributes **two** columns to the number-of-regressors `k`
that indexes the Pesaran–Shin–Smith bounds table — the enlarged `k` is what the
stored [`ARDLBoundsTest`](@ref) uses.

# Fields
- `ardl::ARDLModel{T}`: the underlying ARDL fit on the enlarged (split) design.
- `bounds::ARDLBoundsTest{T}`: PSS bounds test on the enlarged specification (`k` enlarged).
- `y::Vector{T}`: dependent variable (full sample, `N`-vector).
- `X::Matrix{T}`: original regressors (`N × k₀`).
- `Xsplit::Matrix{T}`: enlarged design handed to `estimate_ardl` (`N × k`).
- `asym::Vector{Int}`: original-column indices that were split into partial sums.
- `meta::Vector{Tuple{Int,Symbol}}`: one entry per enlarged regressor — `(orig_index, kind)`
  with `kind ∈ (:pos, :neg, :sym)`, aligned with `ardl.q` / `ardl.x_idx`.
- `k_orig::Int`, `k::Int`: original and enlarged regressor counts.
- `xnames::Vector{String}`: original regressor labels (`k₀`-vector).
- `enames::Vector{String}`: enlarged regressor labels (`k`-vector, e.g. `"x1_POS"`).
- `yname::String`: dependent-variable label.

# References
- Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). *Modelling Asymmetric Cointegration
  and Dynamic Multipliers in a Nonlinear ARDL Framework.* In Festschrift in Honor of
  Peter Schmidt, Springer, 281–314.
"""
struct NARDLModel{T<:AbstractFloat}
    ardl::ARDLModel{T}
    bounds::ARDLBoundsTest{T}
    y::Vector{T}
    X::Matrix{T}
    Xsplit::Matrix{T}
    asym::Vector{Int}
    meta::Vector{Tuple{Int,Symbol}}
    k_orig::Int
    k::Int
    xnames::Vector{String}
    enames::Vector{String}
    yname::String
end

"""
    NARDLSymmetryTest{T<:AbstractFloat}

Long- and short-run symmetry Wald tests for a [`NARDLModel`](@ref), one row per
asymmetric regressor. The long-run test is `H₀: θ⁺_j = θ⁻_j` (a delta-method Wald
on the long-run coefficients, whose Jacobian carries the `1 − Σφ̂` denominator);
the short-run test is `H₀: Σ_ℓ π⁺_{jℓ} = Σ_ℓ π⁻_{jℓ}` on the differenced-term
coefficients of the conditional ECM (a linear restriction on the levels
coefficients). Each single-restriction statistic is reported both as a `χ²(1)` and
as an `F(1, n−K)` with the matching p-value.

# Fields
- `reg_index::Vector{Int}`, `reg_names::Vector{String}`: asymmetric regressors tested.
- `lr_stat::Vector{T}`, `lr_p_chi2::Vector{T}`, `lr_p_f::Vector{T}`: long-run Wald `χ²`, χ²- and F-p-values.
- `sr_stat::Vector{T}`, `sr_p_chi2::Vector{T}`, `sr_p_f::Vector{T}`: short-run Wald `χ²`, χ²- and F-p-values.
- `theta_pos::Vector{T}`, `theta_neg::Vector{T}`: long-run θ⁺_j / θ⁻_j.
- `df::Int`: numerator degrees of freedom (1 per single-restriction test).
- `dof_resid::Int`: residual degrees of freedom `n − K` (F-denominator).

# References
- Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014).
"""
struct NARDLSymmetryTest{T<:AbstractFloat}
    reg_index::Vector{Int}
    reg_names::Vector{String}
    lr_stat::Vector{T}
    lr_p_chi2::Vector{T}
    lr_p_f::Vector{T}
    sr_stat::Vector{T}
    sr_p_chi2::Vector{T}
    sr_p_f::Vector{T}
    theta_pos::Vector{T}
    theta_neg::Vector{T}
    df::Int
    dof_resid::Int
end

"""
    NARDLMultipliers{T<:AbstractFloat}

Cumulative dynamic multipliers of a [`NARDLModel`](@ref): the response of `y` to a
unit permanent (step) change in each asymmetric regressor's positive and negative
partial sum, obtained by recursively iterating the estimated ARDL difference
equation. `m⁺_{j,h}` and `m⁻_{j,h}` converge to the long-run θ⁺_j / θ⁻_j as
`h → ∞`. Optional pointwise percentile bands come from a recursive-design
(condition-on-`x`) residual bootstrap.

# Fields
- `horizons::Vector{Int}`: `0:H`.
- `reg_index::Vector{Int}`, `reg_names::Vector{String}`: asymmetric regressors.
- `m_pos::Matrix{T}`, `m_neg::Matrix{T}`: multipliers, `n_asym × (H+1)`.
- `m_diff::Matrix{T}`: the asymmetry curve `m⁺ − m⁻`.
- `m_pos_lo/hi`, `m_neg_lo/hi`, `m_diff_lo/hi::Matrix{T}`: bootstrap bands (empty if `nreps==0`).
- `theta_pos::Vector{T}`, `theta_neg::Vector{T}`: long-run limits.
- `nreps::Int`, `level::T`: bootstrap replications and band coverage.

# References
- Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014).
"""
struct NARDLMultipliers{T<:AbstractFloat}
    horizons::Vector{Int}
    reg_index::Vector{Int}
    reg_names::Vector{String}
    m_pos::Matrix{T}
    m_neg::Matrix{T}
    m_diff::Matrix{T}
    m_pos_lo::Matrix{T}
    m_pos_hi::Matrix{T}
    m_neg_lo::Matrix{T}
    m_neg_hi::Matrix{T}
    m_diff_lo::Matrix{T}
    m_diff_hi::Matrix{T}
    theta_pos::Vector{T}
    theta_neg::Vector{T}
    nreps::Int
    level::T
end

# PMGModel — dynamic heterogeneous panel ARDL (PMG / MG / DFE), EV-23 (#431)
# =============================================================================

"""
    PMGModel{T<:AbstractFloat}

Dynamic heterogeneous-panel ARDL estimated in error-correction form,

```math
\\Delta y_{it} = \\varphi_i\\,\\bigl(y_{i,t-1} - \\theta' x_{i,t-1}\\bigr)
              + \\sum_{j=1}^{p-1}\\xi_{ij}\\,\\Delta y_{i,t-j}
              + \\sum_{j=0}^{q-1}\\psi_{ij}'\\,\\Delta x_{i,t-j}
              + \\mu_i + \\varepsilon_{it},
```

for `i = 1…N` cross-sectional units. Three estimators (selected by `method`):

- **`:pmg`** — Pooled Mean Group (Pesaran, Shin & Smith 1999): the long-run vector
  `θ` is **common** across units while the speed of adjustment `φ_i`, the short-run
  dynamics, and `σ²_i` are heterogeneous. `θ` maximises the concentrated (profile)
  Gaussian likelihood by alternating a pooled-GLS update of `θ` with per-unit OLS
  updates of `(φ_i, short-run, σ²_i)` to convergence; `θ`'s covariance is the inverse
  information `[Σ_i (φ_i²/σ²_i) X_{i,-1}' N_i X_{i,-1}]⁻¹` (`N_i` projects out the
  short-run block and the EC direction).
- **`:mg`** — Mean Group (Pesaran & Smith 1995): an **unrestricted** ARDL is fit per
  unit (via [`estimate_ardl`](@ref)); `θ̄ = N⁻¹Σθ_i` with the Swamy between-unit
  covariance `(N(N-1))⁻¹Σ(θ_i-θ̄)(θ_i-θ̄)'` (diagonal SE `= std(θ_i)/√N`).
- **`:dfe`** — Dynamic Fixed Effects: a **pooled** within-transformed EC regression with
  common `φ, θ`, short-run, unit intercepts, and cluster-robust (by unit) standard
  errors via `_panel_cluster_vcov`.

Units with `φ_i ≥ 0` (non-error-correcting) are counted in `n_nonconv` and flagged by
[`report`](@ref).

# Fields
- `method::Symbol`: `:pmg`, `:mg`, or `:dfe`.
- `yname::String`, `xnames::Vector{String}`: dependent / long-run regressor labels (`k`).
- `srnames::Vector{String}`: short-run coefficient labels (deterministics + Δ terms).
- `theta::Vector{T}`, `theta_se::Vector{T}`, `theta_vcov::Matrix{T}`: long-run block
  (common for `:pmg`/`:dfe`, averaged for `:mg`) with covariance.
- `theta_i::Matrix{T}`: `N×k` per-unit long-run coefficients (populated for `:mg`; empty
  `0×k` otherwise).
- `phi_i::Vector{T}`: per-unit speed of adjustment (`= φ` repeated for `:dfe`).
- `phi::T`, `phi_se::T`: averaged (or common, `:dfe`) speed of adjustment and its SE.
- `sr::Vector{T}`, `sr_se::Vector{T}`: averaged (common for `:dfe`) short-run block.
- `sr_i::Matrix{T}`: `N×m` per-unit short-run coefficients (empty `0×m` for `:dfe`).
- `sigma2_i::Vector{T}`: per-unit residual variances.
- `loglik::T`: Gaussian log-likelihood (concentrated, at the estimate).
- `N::Int`: number of units; `T_i::Vector{Int}`: per-unit effective sample sizes.
- `p::Int`, `q::Int`: ARDL AR / distributed-lag orders.
- `n_nonconv::Int`: number of units with `φ_i ≥ 0`.
- `converged::Bool`, `iters::Int`: PMG outer-loop convergence flag / iteration count
  (`true`/`0` for `:mg`/`:dfe`).

# References
- Pesaran, M. H., Shin, Y. & Smith, R. P. (1999). *Journal of the American Statistical
  Association* 94(446), 621–634.
- Pesaran, M. H. & Smith, R. (1995). *Journal of Econometrics* 68(1), 79–113.
- Blackburne, E. F. & Frank, M. W. (2007). *Stata Journal* 7(2), 197–208.
"""
struct PMGModel{T<:AbstractFloat}
    method::Symbol
    yname::String
    xnames::Vector{String}
    srnames::Vector{String}
    theta::Vector{T}
    theta_se::Vector{T}
    theta_vcov::Matrix{T}
    theta_i::Matrix{T}
    phi_i::Vector{T}
    phi::T
    phi_se::T
    sr::Vector{T}
    sr_se::Vector{T}
    sr_i::Matrix{T}
    sigma2_i::Vector{T}
    loglik::T
    N::Int
    T_i::Vector{Int}
    p::Int
    q::Int
    n_nonconv::Int
    converged::Bool
    iters::Int
end
