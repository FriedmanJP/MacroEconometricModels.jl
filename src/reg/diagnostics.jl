# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Diagnostic tools for cross-sectional regression models: Variance Inflation
Factors (VIF) for multicollinearity detection and classification tables for
binary response models.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Variance Inflation Factors
# =============================================================================

"""
    vif(m::RegModel{T}) -> Vector{T}

Compute Variance Inflation Factors for each non-intercept regressor.

VIF_j = 1 / (1 - R^2_j), where R^2_j is the R-squared from regressing x_j on
all other regressors (excluding the intercept column). A VIF > 10 indicates
severe multicollinearity (Belsley, Kuh & Welsch 1980).

# Arguments
- `m::RegModel{T}` — estimated OLS/WLS model

# Returns
`Vector{T}` of VIF values, one per non-intercept regressor. The order matches
`m.varnames` with the intercept column removed.

# Examples
```julia
m = estimate_reg(y, X; varnames=["const", "x1", "x2"])
v = vif(m)  # VIF for x1, x2
```

# References
- Belsley, D. A., Kuh, E. & Welsch, R. E. (1980). *Regression Diagnostics*.
  Wiley.
- Greene, W. H. (2018). *Econometric Analysis*. 8th ed. Pearson, ch. 4.
"""
function vif(m::RegModel{T}) where {T<:AbstractFloat}
    X = m.X
    n, k = size(X)

    # Detect intercept column(s): all values equal
    is_intercept = Bool[all(X[:, j] .== X[1, j]) for j in 1:k]
    non_intercept = findall(.!is_intercept)

    length(non_intercept) >= 1 ||
        throw(ArgumentError("VIF requires at least one non-intercept regressor"))

    vif_vals = Vector{T}(undef, length(non_intercept))

    for (idx, j) in enumerate(non_intercept)
        # Dependent variable: column j
        x_j = X[:, j]

        # Regressors: all other columns (including intercept)
        other_cols = setdiff(1:k, j)
        X_other = X[:, other_cols]

        # OLS regression of x_j on X_other
        XtXinv = robust_inv(X_other' * X_other)
        beta_j = XtXinv * (X_other' * x_j)
        fitted_j = X_other * beta_j
        resid_j = x_j .- fitted_j

        # R^2_j
        x_bar = mean(x_j)
        tss_j = sum((xi - x_bar)^2 for xi in x_j)
        tss_j = max(tss_j, T(1e-300))
        ssr_j = dot(resid_j, resid_j)
        r2_j = one(T) - ssr_j / tss_j

        # VIF_j = 1 / (1 - R^2_j), clamped to avoid division by zero
        vif_vals[idx] = one(T) / max(one(T) - r2_j, T(1e-10))
    end

    vif_vals
end

# =============================================================================
# Classification Table
# =============================================================================

"""
    classification_table(m::Union{LogitModel{T},ProbitModel{T}}; threshold=0.5) -> Dict{String,Any}

Compute a classification table (confusion matrix) and summary metrics for a
binary response model.

# Arguments
- `m` — estimated LogitModel or ProbitModel
- `threshold::Real` — classification threshold (default 0.5)

# Returns
Dict with keys:
- `"confusion"` — 2x2 confusion matrix [[TN, FP], [FN, TP]]
- `"accuracy"` — (TP + TN) / N
- `"sensitivity"` — TP / (TP + FN) (true positive rate / recall)
- `"specificity"` — TN / (TN + FP) (true negative rate)
- `"precision"` — TP / (TP + FP) (positive predictive value)
- `"f1_score"` — 2 * precision * recall / (precision + recall)
- `"n"` — number of observations
- `"threshold"` — classification threshold used

# Examples
```julia
m = estimate_logit(y, X)
ct = classification_table(m)
ct["accuracy"]    # overall accuracy
ct["confusion"]   # 2x2 confusion matrix
```

# References
- Agresti, A. (2002). *Categorical Data Analysis*. 2nd ed. Wiley.
"""
function classification_table(m::Union{LogitModel{T},ProbitModel{T}};
                               threshold::Real=0.5) where {T<:AbstractFloat}
    y = m.y
    p_hat = m.fitted
    n = length(y)
    thresh = T(threshold)

    # Predicted classes
    y_pred = T.(p_hat .>= thresh)

    # Confusion matrix elements
    tp = zero(T)
    tn = zero(T)
    fp = zero(T)
    fn = zero(T)

    @inbounds for i in 1:n
        if y[i] == one(T) && y_pred[i] == one(T)
            tp += one(T)
        elseif y[i] == zero(T) && y_pred[i] == zero(T)
            tn += one(T)
        elseif y[i] == zero(T) && y_pred[i] == one(T)
            fp += one(T)
        else  # y == 1, pred == 0
            fn += one(T)
        end
    end

    # Metrics
    accuracy = (tp + tn) / T(n)

    sensitivity = (tp + fn) > zero(T) ? tp / (tp + fn) : zero(T)
    specificity = (tn + fp) > zero(T) ? tn / (tn + fp) : zero(T)
    prec = (tp + fp) > zero(T) ? tp / (tp + fp) : zero(T)

    f1 = (prec + sensitivity) > zero(T) ?
        2 * prec * sensitivity / (prec + sensitivity) : zero(T)

    # Confusion matrix: rows = actual (0, 1), cols = predicted (0, 1)
    confusion = Matrix{T}([tn fp; fn tp])

    Dict{String,Any}(
        "confusion"   => confusion,
        "accuracy"    => accuracy,
        "sensitivity" => sensitivity,
        "specificity" => specificity,
        "precision"   => prec,
        "f1_score"    => f1,
        "n"           => n,
        "threshold"   => thresh
    )
end

# =============================================================================
# OLS Residual Diagnostics — EV-31 (#439)
# =============================================================================
# White (1980), Breusch-Pagan (1979) / Koenker (1981), Glejser (1969),
# Harvey (1976), Breusch (1978) & Godfrey (1978), Ramsey (1969, RESET).
#
# All consume a fitted `RegModel{T}`, run an auxiliary OLS regression, and
# report an nR^2 chi-squared statistic and/or an F-form. Raw-data `(resid, X)`
# convenience methods delegate to the same kernels.
#
# NAME-COLLISION NOTE: `breusch_pagan_test(::PanelRegModel)` in src/preg/tests.jl
# is a *different* test (the random-effects LM test). The method below dispatches
# on `RegModel`; the two coexist by argument type and must never be merged.

"""
    RegDiagnosticResult{T} <: StatsAPI.HypothesisTest

Result of an OLS residual-diagnostic test (heteroskedasticity, serial
correlation, or functional form) on a fitted [`RegModel`](@ref).

# Fields
- `test_name::String` — e.g. `"White test"`.
- `h0::String` — null hypothesis in words.
- `statistic::T` — primary test statistic. When `df` is an `Int` this is an
  nR² χ² statistic; when `df` is a `Tuple{Int,Int}` it is an F statistic.
- `pvalue::T` — p-value of the primary statistic.
- `df::Union{Int,Tuple{Int,Int}}` — χ² degrees of freedom (`Int`) or `(df1, df2)`
  for an F-form primary statistic.
- `f_stat::Union{Nothing,T}` — secondary F-form statistic (reported alongside the
  χ² statistic, e.g. for the Breusch–Godfrey test); `nothing` when not applicable.
- `f_pvalue::Union{Nothing,T}` — p-value of `f_stat`.
- `f_df::Union{Nothing,Tuple{Int,Int}}` — `(df1, df2)` of `f_stat`.
- `aux_r2::T` — R² of the auxiliary regression.
- `n::Int` — number of observations used in the auxiliary regression.

# See also
[`white_test`](@ref), [`breusch_pagan_test`](@ref), [`glejser_test`](@ref),
[`harvey_test`](@ref), [`breusch_godfrey_test`](@ref), [`reset_test`](@ref).
"""
struct RegDiagnosticResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    test_name::String
    h0::String
    statistic::T
    pvalue::T
    df::Union{Int,Tuple{Int,Int}}
    f_stat::Union{Nothing,T}
    f_pvalue::Union{Nothing,T}
    f_df::Union{Nothing,Tuple{Int,Int}}
    aux_r2::T
    n::Int
end

# --- Internal helpers ---------------------------------------------------------

# Indices of the non-intercept columns of a regressor matrix (an intercept
# column is one whose entries are all equal). Mirrors the `vif` convention.
function _nonintercept_cols(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    k = size(X, 2)
    is_intercept = Bool[all(X[:, j] .== X[1, j]) for j in 1:k]
    findall(.!is_intercept)
end

# Pivoted-QR rank-revealing selection of linearly independent columns of `M`.
# Returns the (sorted) column indices to keep. Used to guard singular White
# cross-term designs (e.g. a dummy regressor whose square equals itself).
function _indep_cols(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    n, p = size(M)
    p == 0 && return Int[]
    F = qr(Matrix{T}(M), ColumnNorm())
    absr = abs.(diag(F.R))
    isempty(absr) && return Int[]
    thresh = maximum(absr) * eps(real(T)) * max(n, p)
    r = count(>(thresh), absr)
    sort(F.p[1:r])
end

# Centered R² of regressing `dep` on the columns of `Z` (with an implicit
# intercept), computed on demeaned data so the intercept is handled exactly and
# collinear `Z` columns are dropped. Returns `(r2, df, tss)` where `df` is the
# number of retained (independent) non-constant regressors.
function _aux_centered_r2(dep::AbstractVector{T}, Z::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = length(dep)
    dc = dep .- mean(dep)
    tss = max(dot(dc, dc), T(1e-300))
    size(Z, 2) == 0 && return (zero(T), 0, tss)
    Zc = Z .- mean(Z, dims = 1)
    keep = _indep_cols(Zc)
    isempty(keep) && return (zero(T), 0, tss)
    Zk = Zc[:, keep]
    beta = robust_inv(Zk' * Zk) * (Zk' * dc)
    resid = dc .- Zk * beta
    ssr = dot(resid, resid)
    r2 = one(T) - ssr / tss
    (max(r2, zero(T)), length(keep), tss)
end

_chisq_p(stat::T, df::Int) where {T<:AbstractFloat} =
    (df < 1 || !(stat > zero(T)) || !isfinite(stat)) ? one(T) : T(ccdf(Chisq(df), stat))

function _f_p(f::T, df1::Int, df2::Int) where {T<:AbstractFloat}
    (df1 < 1 || df2 < 1 || !(f > zero(T)) || !isfinite(f)) && return one(T)
    T(ccdf(FDist(df1, df2), f))
end

# =============================================================================
# White (1980) heteroskedasticity test
# =============================================================================

"""
    white_test(m::RegModel; cross_terms=true) -> RegDiagnosticResult
    white_test(resid, X; cross_terms=true) -> RegDiagnosticResult

White's (1980) general test for heteroskedasticity. Regresses the squared OLS
residuals on the original (non-intercept) regressors, their squares, and — when
`cross_terms=true` — their pairwise cross-products. The statistic is
`nR²_aux ~ χ²(df)`, with `df` equal to the number of auxiliary regressors
(excluding the constant) that survive collinearity pruning.

With `cross_terms=true` the auxiliary design has up to `k(k+3)/2` terms; columns
that are linearly dependent (e.g. the square of a 0/1 dummy, which equals the
dummy itself) are dropped via a rank-revealing pivoted QR and `df` is reduced
accordingly.

# Arguments
- `m::RegModel` — fitted OLS/WLS model (or pass `resid`, `X` directly).
- `cross_terms::Bool=true` — include pairwise cross-products.

H₀: homoskedasticity (error variance unrelated to the regressors).

# References
- White, H. (1980). *Econometrica* 48(4), 817–838.
"""
function white_test(resid::AbstractVector{T}, X::AbstractMatrix{T};
                    cross_terms::Bool=true) where {T<:AbstractFloat}
    ni = _nonintercept_cols(X)
    Z = Matrix{T}(X[:, ni])
    n, kz = size(Z)
    dep = resid .^ 2
    # Auxiliary regressors: levels, squares, (optional) cross-products.
    cols = Vector{Vector{T}}()
    for j in 1:kz
        push!(cols, Z[:, j])
    end
    for j in 1:kz
        push!(cols, Z[:, j] .^ 2)
    end
    if cross_terms
        for i in 1:kz, j in (i+1):kz
            push!(cols, Z[:, i] .* Z[:, j])
        end
    end
    aux = isempty(cols) ? zeros(T, n, 0) : reduce(hcat, cols)
    r2, df, _ = _aux_centered_r2(dep, aux)
    stat = T(n) * r2
    pval = _chisq_p(stat, df)
    h0 = "Homoskedasticity (error variance unrelated to regressors)"
    RegDiagnosticResult{T}("White test" * (cross_terms ? "" : " (no cross-terms)"),
                           h0, stat, pval, df, nothing, nothing, nothing, r2, n)
end

white_test(m::RegModel{T}; cross_terms::Bool=true) where {T<:AbstractFloat} =
    white_test(m.residuals, m.X; cross_terms=cross_terms)

# =============================================================================
# Breusch–Pagan (1979) / Koenker (1981) heteroskedasticity test
# =============================================================================

"""
    breusch_pagan_test(m::RegModel; studentized=true, het_regressors=nothing) -> RegDiagnosticResult
    breusch_pagan_test(resid, X; studentized=true, het_regressors=nothing) -> RegDiagnosticResult

Breusch–Pagan (1979) test for heteroskedasticity, defaulting to the Koenker
(1981) studentized version. Regresses the squared residuals on `het_regressors`
(default = the model's non-intercept regressors).

- `studentized=true` (default) — Koenker's `nR²_aux ~ χ²(p)` form, robust to
  non-normal errors. This matches R `lmtest::bptest(..., studentize=TRUE)`.
- `studentized=false` — the original BP form `½·ESS ~ χ²(p)`, where `ESS` is the
  explained sum of squares from regressing `resid²/σ̂²` (with `σ̂² = SSR/n`) on the
  regressors. Valid only under Gaussian errors.

!!! note "Name collision"
    A **different** `breusch_pagan_test(::PanelRegModel)` exists in
    `src/preg/tests.jl` — the random-effects Lagrange-multiplier test. That
    method dispatches on `PanelRegModel`; this one dispatches on `RegModel`. They
    are distinct tests that share only a name. See also that method's docstring.

H₀: homoskedasticity.

# References
- Breusch, T. S. & Pagan, A. R. (1979). *Econometrica* 47(5), 1287–1294.
- Koenker, R. (1981). *Journal of Econometrics* 17(1), 107–112.
"""
function breusch_pagan_test(resid::AbstractVector{T}, X::AbstractMatrix{T};
                            studentized::Bool=true,
                            het_regressors::Union{Nothing,AbstractMatrix}=nothing) where {T<:AbstractFloat}
    Z = het_regressors === nothing ? Matrix{T}(X[:, _nonintercept_cols(X)]) : Matrix{T}(het_regressors)
    n = length(resid)
    if studentized
        dep = resid .^ 2
        r2, df, _ = _aux_centered_r2(dep, Z)
        stat = T(n) * r2
        name = "Breusch-Pagan test (Koenker studentized)"
    else
        sigma2 = dot(resid, resid) / T(n)          # MLE variance (σ̂² = SSR/n)
        g = (resid .^ 2) ./ max(sigma2, T(1e-300))
        r2, df, tss = _aux_centered_r2(g, Z)
        ess = r2 * tss                             # explained SS of the g-regression
        stat = T(0.5) * ess
        name = "Breusch-Pagan test (original)"
    end
    pval = _chisq_p(stat, df)
    RegDiagnosticResult{T}(name, "Homoskedasticity", stat, pval, df,
                           nothing, nothing, nothing, zero(T), n)
end

breusch_pagan_test(m::RegModel{T}; studentized::Bool=true,
                   het_regressors::Union{Nothing,AbstractMatrix}=nothing) where {T<:AbstractFloat} =
    breusch_pagan_test(m.residuals, m.X; studentized=studentized, het_regressors=het_regressors)

# =============================================================================
# Glejser (1969) heteroskedasticity test
# =============================================================================

"""
    glejser_test(m::RegModel) -> RegDiagnosticResult
    glejser_test(resid, X) -> RegDiagnosticResult

Glejser's (1969) test for heteroskedasticity: regress the absolute residuals
`|resid|` on the (non-intercept) regressors and F-test the joint significance of
the slopes. The primary statistic is the F-form `F(p, n−p−1)`; the auxiliary R²
is stored in `aux_r2`.

H₀: homoskedasticity (|resid| unrelated to the regressors).

# References
- Glejser, H. (1969). *JASA* 64(325), 316–323.
"""
function glejser_test(resid::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
    Z = Matrix{T}(X[:, _nonintercept_cols(X)])
    n = length(resid)
    dep = abs.(resid)
    r2, df, _ = _aux_centered_r2(dep, Z)
    df1 = df
    df2 = n - df1 - 1                              # minus the constant
    f = (df1 >= 1 && df2 >= 1 && r2 < one(T)) ?
        (r2 / df1) / ((one(T) - r2) / T(df2)) : zero(T)
    pval = _f_p(f, df1, df2)
    RegDiagnosticResult{T}("Glejser test", "Homoskedasticity (|resid| unrelated to regressors)",
                           f, pval, (df1, df2), nothing, nothing, nothing, r2, n)
end

glejser_test(m::RegModel{T}) where {T<:AbstractFloat} = glejser_test(m.residuals, m.X)

# =============================================================================
# Harvey (1976) multiplicative-heteroskedasticity test
# =============================================================================

"""
    harvey_test(m::RegModel) -> RegDiagnosticResult
    harvey_test(resid, X) -> RegDiagnosticResult

Harvey's (1976) test for multiplicative heteroskedasticity: regress
`log(resid²)` on the (non-intercept) regressors; the statistic is
`nR²_aux ~ χ²(p)`.

H₀: homoskedasticity (`log(resid²)` unrelated to the regressors).

# References
- Harvey, A. C. (1976). *Econometrica* 44(3), 461–465.
"""
function harvey_test(resid::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
    Z = Matrix{T}(X[:, _nonintercept_cols(X)])
    n = length(resid)
    dep = log.(max.(resid .^ 2, T(1e-300)))       # guard log(0) for exact-zero residuals
    r2, df, _ = _aux_centered_r2(dep, Z)
    stat = T(n) * r2
    pval = _chisq_p(stat, df)
    RegDiagnosticResult{T}("Harvey test", "Homoskedasticity (log resid² unrelated to regressors)",
                           stat, pval, df, nothing, nothing, nothing, r2, n)
end

harvey_test(m::RegModel{T}) where {T<:AbstractFloat} = harvey_test(m.residuals, m.X)

# =============================================================================
# Breusch (1978) – Godfrey (1978) serial-correlation LM test
# =============================================================================

"""
    breusch_godfrey_test(m::RegModel; lags=1) -> RegDiagnosticResult
    breusch_godfrey_test(resid, X; lags=1) -> RegDiagnosticResult

Breusch (1978)–Godfrey (1978) Lagrange-multiplier test for serial correlation up
to order `lags`. Regresses the residuals on the original regressors `X` **and**
`lags` lagged residuals; missing pre-sample lags are **zero-padded** (the
standard `lmtest::bgtest` convention).

Reports both the χ² LM form `nR²_aux ~ χ²(p)` (primary `statistic`/`df`) and the
F-form (in `f_stat`/`f_pvalue`/`f_df`), the latter from comparing the auxiliary
regression against the restricted regression of the residuals on `X` alone.

H₀: no serial correlation up to order `lags`.

# References
- Breusch, T. S. (1978). *Australian Economic Papers* 17(31), 334–355.
- Godfrey, L. G. (1978). *Econometrica* 46(6), 1293–1301.
"""
function breusch_godfrey_test(resid::AbstractVector{T}, X::AbstractMatrix{T};
                              lags::Int=1) where {T<:AbstractFloat}
    lags >= 1 || throw(ArgumentError("lags must be ≥ 1 (got $lags)"))
    n = length(resid)
    lags < n || throw(ArgumentError("lags ($lags) must be < n ($n)"))
    Xm = Matrix{T}(X)
    # Lagged residuals with zero-padding for pre-sample values.
    E = zeros(T, n, lags)
    for j in 1:lags
        @inbounds for i in (j+1):n
            E[i, j] = resid[i-j]
        end
    end
    # Auxiliary: resid ~ X + lagged residuals. Centered R² (X carries its own
    # intercept column, so use it directly, dropping collinear columns).
    aux_full = hcat(Xm, E)
    keep = _indep_cols(aux_full)
    Ak = aux_full[:, keep]
    beta = robust_inv(Ak' * Ak) * (Ak' * resid)
    r_aux = resid .- Ak * beta
    ssr1 = dot(r_aux, r_aux)
    dbar = mean(resid)
    tss = max(sum((r - dbar)^2 for r in resid), T(1e-300))
    r2 = max(one(T) - ssr1 / tss, zero(T))
    stat = T(n) * r2
    df = lags
    pval = _chisq_p(stat, df)
    # F-form: restricted regression of resid on X alone.
    keep0 = _indep_cols(Xm)
    Xk = Xm[:, keep0]
    beta0 = robust_inv(Xk' * Xk) * (Xk' * resid)
    r0 = resid .- Xk * beta0
    ssr0 = dot(r0, r0)
    kk = length(keep)
    df2 = n - kk
    fstat = (df2 >= 1 && ssr1 > zero(T)) ?
        ((ssr0 - ssr1) / T(lags)) / (ssr1 / T(df2)) : zero(T)
    fpval = _f_p(fstat, lags, df2)
    RegDiagnosticResult{T}("Breusch-Godfrey test (lags=$lags)",
                           "No serial correlation up to order $lags",
                           stat, pval, df, fstat, fpval, (lags, df2), r2, n)
end

breusch_godfrey_test(m::RegModel{T}; lags::Int=1) where {T<:AbstractFloat} =
    breusch_godfrey_test(m.residuals, m.X; lags=lags)

# =============================================================================
# Ramsey (1969) RESET functional-form test
# =============================================================================

"""
    reset_test(m::RegModel; powers=2:4) -> RegDiagnosticResult

Ramsey's (1969) RESET test for functional-form misspecification. Augments the
original regression of `y` on `X` with powers `ŷ^2 … ŷ^k` of the **fitted
values** (not of individual regressors) and F-tests that the added powers are
jointly zero.

The primary statistic is the F-form `F(q, n−k_aug)` where `q = length(powers)`
and `k_aug` is the number of columns of the augmented (collinearity-pruned)
design.

H₀: correct functional form (added power terms jointly zero).

# References
- Ramsey, J. B. (1969). *JRSS-B* 31(2), 350–371.
"""
function reset_test(m::RegModel{T}; powers=2:4) where {T<:AbstractFloat}
    pw = collect(powers)
    all(p -> p >= 2, pw) || throw(ArgumentError("RESET powers must all be ≥ 2 (got $pw)"))
    y = m.y
    X = Matrix{T}(m.X)
    yhat = m.fitted
    n = length(y)
    q = length(pw)
    # Augmented design: X and yhat^p for p in powers.
    P = reduce(hcat, [yhat .^ T(p) for p in pw])
    aug = hcat(X, P)
    keep = _indep_cols(aug)
    Ak = aug[:, keep]
    beta = robust_inv(Ak' * Ak) * (Ak' * y)
    r1 = y .- Ak * beta
    ssr1 = dot(r1, r1)
    ssr0 = m.ssr                                  # restricted SSR from the fitted model
    kk = length(keep)
    df2 = n - kk
    fstat = (df2 >= 1 && ssr1 > zero(T)) ?
        ((ssr0 - ssr1) / T(q)) / (ssr1 / T(df2)) : zero(T)
    pval = _f_p(fstat, q, df2)
    # Auxiliary R² of the augmented regression (for reference).
    ybar = mean(y)
    tss = max(sum((yi - ybar)^2 for yi in y), T(1e-300))
    r2 = max(one(T) - ssr1 / tss, zero(T))
    RegDiagnosticResult{T}("Ramsey RESET test (powers=$(pw))",
                           "Correct functional form (added power terms jointly zero)",
                           fstat, pval, (q, df2), nothing, nothing, nothing, r2, n)
end

# =============================================================================
# Display — RegDiagnosticResult
# =============================================================================

function Base.show(io::IO, r::RegDiagnosticResult{T}) where {T}
    is_f = r.df isa Tuple
    stat_label = is_f ? "F statistic" : "χ² statistic (nR²)"
    df_str = is_f ? "($(r.df[1]), $(r.df[2]))" : string(r.df)
    reject = r.pvalue < T(0.05)
    concl = reject ?
        "Reject H0 (p=$(_format_pvalue(r.pvalue)))" :
        "Fail to reject H0 (p=$(_format_pvalue(r.pvalue)))"
    rows = Any[
        "Test"        r.test_name;
        "H0"          r.h0;
        stat_label    _fmt(r.statistic);
        "df"          df_str;
        "p-value"     _format_pvalue(r.pvalue);
    ]
    if r.f_stat !== nothing
        rows = vcat(rows, Any[
            "F statistic"  _fmt(r.f_stat);
            "F df"         "($(r.f_df[1]), $(r.f_df[2]))";
            "F p-value"    _format_pvalue(r.f_pvalue);
        ])
    end
    rows = vcat(rows, Any["Conclusion"  concl])
    _pretty_table(io, rows; title = r.test_name,
                  column_labels = ["", ""], alignment = [:l, :r])
end

"""
    report(r::RegDiagnosticResult)

Print an OLS residual-diagnostic test result to stdout.
"""
report(r::RegDiagnosticResult) = show(stdout, r)
