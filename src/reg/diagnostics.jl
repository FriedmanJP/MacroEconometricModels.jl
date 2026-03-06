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

using LinearAlgebra, Statistics

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
