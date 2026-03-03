# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Structural break tests for factor models.

Implements three tests for structural instability in factor models:
- Breitung & Eickmeier (2011): Loading stability via CUSUM fluctuation
- Chen, Dolado & Gonzalo (2014): Change in the number of factors
- Han & Inoue (2015): Loading instability with unknown break (sup-Wald)

References:
- Breitung, J., & Eickmeier, S. (2011). Testing for structural breaks in dynamic
  factor models. Journal of Econometrics, 163(1), 71-84.
- Chen, L., Dolado, J. J., & Gonzalo, J. (2014). Detecting big structural breaks
  in large factor models. Journal of Econometrics, 180(1), 30-48.
- Han, X., & Inoue, A. (2015). Tests for parameter instability in dynamic factor
  models. Econometric Theory, 31(5), 1117-1152.
"""

# =============================================================================
# Main API
# =============================================================================

"""
    factor_break_test(X, r; method=:breitung_eickmeier) -> FactorBreakResult
    factor_break_test(fm::FactorModel; method=:breitung_eickmeier) -> FactorBreakResult
    factor_break_test(X; method=:chen_dolado_gonzalo) -> FactorBreakResult

Test for structural breaks in factor models.

# Methods
- `:breitung_eickmeier` — Breitung & Eickmeier (2011) loading stability CUSUM test
- `:chen_dolado_gonzalo` — Chen, Dolado & Gonzalo (2014) change in number of factors
- `:han_inoue` — Han & Inoue (2015) sup-Wald loading instability test

# Arguments
- `X`: Data matrix (T × N), observations × variables
- `r`: Number of factors (required for :breitung_eickmeier and :han_inoue)
- `fm`: Estimated `FactorModel` (alternative to providing X and r)

# Returns
`FactorBreakResult{T}` with test statistic, p-value, estimated break date, and method.

# Examples
```julia
X = randn(200, 50)
result = factor_break_test(X, 3; method=:breitung_eickmeier)
result.pvalue < 0.05 && println("Reject loading stability at 5%")

# Using FactorModel dispatch
fm = estimate_factors(X, 3)
result = factor_break_test(fm; method=:han_inoue)

# Chen-Dolado-Gonzalo does not require r
result = factor_break_test(X; method=:chen_dolado_gonzalo)
```
"""
function factor_break_test(X::AbstractMatrix{T}, r::Int;
                           method::Symbol=:breitung_eickmeier) where {T<:AbstractFloat}
    method ∈ (:breitung_eickmeier, :chen_dolado_gonzalo, :han_inoue) ||
        throw(ArgumentError("method must be :breitung_eickmeier, :chen_dolado_gonzalo, or :han_inoue; got :$method"))

    T_obs, N = size(X)
    T_obs < 30 && throw(ArgumentError("Time series too short (T=$T_obs), need at least 30 observations"))

    if method == :breitung_eickmeier
        return _breitung_eickmeier_test(X, r)
    elseif method == :chen_dolado_gonzalo
        return _chen_dolado_gonzalo_test(X)
    else  # :han_inoue
        return _han_inoue_test(X, r)
    end
end

# FactorModel dispatch
function factor_break_test(fm::FactorModel{T};
                           method::Symbol=:breitung_eickmeier) where {T<:AbstractFloat}
    factor_break_test(fm.X, fm.r; method=method)
end

# Matrix-only dispatch (default to chen_dolado_gonzalo which doesn't need r)
function factor_break_test(X::AbstractMatrix{T};
                           method::Symbol=:chen_dolado_gonzalo) where {T<:AbstractFloat}
    method ∈ (:breitung_eickmeier, :chen_dolado_gonzalo, :han_inoue) ||
        throw(ArgumentError("method must be :breitung_eickmeier, :chen_dolado_gonzalo, or :han_inoue; got :$method"))

    if method ∈ (:breitung_eickmeier, :han_inoue)
        throw(ArgumentError("Method :$method requires the number of factors r. Use factor_break_test(X, r; method=:$method)"))
    end

    T_obs, N = size(X)
    T_obs < 30 && throw(ArgumentError("Time series too short (T=$T_obs), need at least 30 observations"))

    return _chen_dolado_gonzalo_test(X)
end

# Float64 fallbacks
factor_break_test(X::AbstractMatrix, r::Int; kwargs...) =
    factor_break_test(Float64.(X), r; kwargs...)
factor_break_test(X::AbstractMatrix; kwargs...) =
    factor_break_test(Float64.(X); kwargs...)

# =============================================================================
# Breitung-Eickmeier (2011) — Loading stability CUSUM test
# =============================================================================

function _breitung_eickmeier_test(X::AbstractMatrix{T}, r::Int) where {T<:AbstractFloat}
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)

    # Estimate factors from full sample
    fm = estimate_factors(X, r; standardize=true)
    F_hat = fm.factors          # T × r
    Lambda_full = fm.loadings   # N × r

    # Full-sample loadings: Λ̂(T) = X' F̂ (F̂' F̂)^{-1}
    # (equivalent to fm.loadings when standardized)
    X_std = _standardize(X)
    FtF_full = F_hat' * F_hat   # r × r
    FtF_full_inv = robust_inv(FtF_full)
    Lambda_T = X_std' * F_hat * FtF_full_inv   # N × r

    # Estimate variance of loading regression residuals
    resid_full = X_std - F_hat * Lambda_T'  # T × N
    sigma2_hat = sum(resid_full .^ 2) / (T_obs * N)

    # Trimmed range: [0.15T, 0.85T]
    trim = max(round(Int, 0.15 * T_obs), r + 1)
    t_start = trim
    t_end = T_obs - trim

    if t_start >= t_end
        # Insufficient observations for meaningful test
        return FactorBreakResult{T}(zero(T), one(T), nothing, :breitung_eickmeier,
                                    r, T_obs, N)
    end

    # Compute CUSUM fluctuation statistics
    n_params = N * r
    fluct_path = Vector{T}(undef, t_end - t_start + 1)

    for (idx, t) in enumerate(t_start:t_end)
        F_sub = F_hat[1:t, :]           # t × r
        X_sub = X_std[1:t, :]           # t × N
        FtF_sub = F_sub' * F_sub        # r × r
        FtF_sub_inv = robust_inv(FtF_sub)
        Lambda_t = X_sub' * F_sub * FtF_sub_inv  # N × r

        # Fluctuation: scaled difference in vectorized loadings
        diff_vec = vec(Lambda_t - Lambda_T)
        # Scale by sqrt(t) for Brownian bridge normalization
        fluct_path[idx] = sqrt(T(t) / T_obs) * norm(diff_vec) / sqrt(max(sigma2_hat, T(1e-10)))
    end

    # Sup statistic
    stat = maximum(fluct_path)
    break_idx = argmax(fluct_path)
    break_date = t_start + break_idx - 1

    # P-value: asymptotic distribution is sup of Bessel process
    # Approximate using chi-squared with df = N*r (loading parameters)
    # The statistic squared is approximately chi-squared under the null
    pval = _breitung_eickmeier_pvalue(stat, n_params)

    FactorBreakResult{T}(stat, pval, break_date, :breitung_eickmeier, r, T_obs, N)
end

"""
Approximate p-value for Breitung-Eickmeier CUSUM statistic.

Uses the asymptotic distribution of the supremum of a Brownian bridge process.
For the multivariate case with `k` parameters, approximated via
chi-squared(k) applied to the squared statistic.
"""
function _breitung_eickmeier_pvalue(stat::T, k::Int) where {T<:AbstractFloat}
    # The squared CUSUM stat is asymptotically related to sup of
    # squared Brownian bridge. Use Kolmogorov-Smirnov-type approximation:
    # P(sup|B(s)| > x) ≈ 2 * sum_{j=1}^∞ (-1)^{j+1} exp(-2j²x²)
    # For multivariate generalization, use chi-squared approximation
    stat_sq = stat^2
    pval = ccdf(Chisq(k), stat_sq)
    clamp(pval, zero(T), one(T))
end

# =============================================================================
# Chen-Dolado-Gonzalo (2014) — Number of factors change
# =============================================================================

function _chen_dolado_gonzalo_test(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    T_obs, N = size(X)

    # Determine r_max
    r_max = min(floor(Int, sqrt(min(T_obs, N))), 10)
    r_max = max(r_max, 1)

    # Standardize the full data
    X_std = _standardize(X)

    # Trimmed range
    trim = max(round(Int, 0.15 * T_obs), r_max + 2)
    t_start = trim
    t_end = T_obs - trim

    if t_start >= t_end
        return FactorBreakResult{T}(zero(T), one(T), nothing, :chen_dolado_gonzalo,
                                    r_max, T_obs, N)
    end

    # Compute eigenvalue ratios for each candidate break
    stat_path = Vector{T}(undef, t_end - t_start + 1)

    for (idx, t) in enumerate(t_start:t_end)
        # Subsample 1: [1, t]
        X1 = X_std[1:t, :]
        eig1 = _sorted_eigenvalues(X1)

        # Subsample 2: [t+1, T]
        X2 = X_std[(t+1):T_obs, :]
        eig2 = _sorted_eigenvalues(X2)

        # Compare eigenvalue ratios across subsamples
        # ER_k = lambda_k / lambda_{k+1}
        max_diff = zero(T)
        n_eig = min(length(eig1), length(eig2), r_max)
        for k in 1:n_eig
            if k < length(eig1) && k < length(eig2)
                er1 = eig1[k] / max(eig1[k+1], T(1e-10))
                er2 = eig2[k] / max(eig2[k+1], T(1e-10))
                max_diff = max(max_diff, abs(er1 - er2))
            end
        end

        # Normalize by variance estimate (use geometric mean of subsample sizes)
        norm_factor = sqrt(T(t) * T(T_obs - t) / T(T_obs))
        stat_path[idx] = norm_factor * max_diff
    end

    # Test statistic: maximum over candidate breaks
    stat = maximum(stat_path)

    # P-value: chi-squared approximation with df = r_max
    pval = ccdf(Chisq(r_max), stat)
    pval = clamp(pval, zero(T), one(T))

    # Chen-Dolado-Gonzalo doesn't precisely localize the break
    break_idx = argmax(stat_path)
    break_date = t_start + break_idx - 1

    FactorBreakResult{T}(stat, pval, break_date, :chen_dolado_gonzalo, r_max, T_obs, N)
end

"""
Compute sorted eigenvalues (descending) of sample covariance of X.
"""
function _sorted_eigenvalues(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    T_obs, N = size(X)
    Sigma = (X'X) / T_obs
    eig_vals = eigvals(Symmetric(Sigma))
    sort(eig_vals; rev=true)
end

# =============================================================================
# Han-Inoue (2015) — Loading instability sup-Wald
# =============================================================================

function _han_inoue_test(X::AbstractMatrix{T}, r::Int) where {T<:AbstractFloat}
    T_obs, N = size(X)
    validate_factor_inputs(T_obs, N, r)

    # Estimate factors from full sample
    fm = estimate_factors(X, r; standardize=true)
    F_hat = fm.factors    # T × r
    X_std = _standardize(X)

    # Trimmed range: [0.15T, 0.85T]
    trim = max(round(Int, 0.15 * T_obs), r + 1)
    t_start = trim
    t_end = T_obs - trim

    if t_start >= t_end
        return FactorBreakResult{T}(zero(T), one(T), nothing, :han_inoue,
                                    r, T_obs, N)
    end

    # For each candidate break date, compute sum of individual Wald statistics
    wald_path = Vector{T}(undef, t_end - t_start + 1)

    # Precompute full-sample quantities for each unit
    # Loading regression: X_i = F_hat * lambda_i + e_i
    FtF = F_hat' * F_hat  # r × r
    FtF_inv = robust_inv(FtF)

    for (idx, t) in enumerate(t_start:t_end)
        F1 = F_hat[1:t, :]           # t × r
        F2 = F_hat[(t+1):T_obs, :]   # (T-t) × r

        F1tF1 = F1' * F1             # r × r
        F2tF2 = F2' * F2             # r × r
        F1tF1_inv = robust_inv(F1tF1)
        F2tF2_inv = robust_inv(F2tF2)

        W_t = zero(T)
        for i in 1:N
            x_i = X_std[:, i]   # T × 1

            # Subsample loadings
            lambda1 = F1tF1_inv * (F1' * x_i[1:t])        # r × 1
            lambda2 = F2tF2_inv * (F2' * x_i[(t+1):T_obs]) # r × 1

            # Full-sample residual variance for unit i
            lambda_full = FtF_inv * (F_hat' * x_i)
            resid_i = x_i - F_hat * lambda_full
            sigma2_i = max(sum(resid_i .^ 2) / T_obs, T(1e-10))

            # Wald statistic for H0: lambda1 = lambda2
            diff_lambda = lambda1 - lambda2
            # Variance of difference: sigma2_i * (inv(F1'F1) + inv(F2'F2))
            V_diff = sigma2_i * (F1tF1_inv + F2tF2_inv)
            V_diff_inv = robust_inv(V_diff)

            W_i = diff_lambda' * V_diff_inv * diff_lambda
            W_t += max(W_i, zero(T))
        end

        wald_path[idx] = W_t / N
    end

    # Sup-Wald statistic
    stat = maximum(wald_path)
    break_idx = argmax(wald_path)
    break_date = t_start + break_idx - 1

    # P-value: Andrews (1993) sup-Wald distribution with df = r
    pval = _han_inoue_pvalue(stat, r)

    FactorBreakResult{T}(stat, pval, break_date, :han_inoue, r, T_obs, N)
end

"""
Approximate p-value for Han-Inoue sup-Wald statistic using
Hansen (1997) / Andrews (1993) critical values.
"""
function _han_inoue_pvalue(stat::T, k::Int) where {T<:AbstractFloat}
    # Use HANSEN_ANDREWS_CV table for k = 1,...,10
    k_clamped = clamp(k, 1, 10)
    cv = HANSEN_ANDREWS_CV[k_clamped]

    # Interpolate p-value from critical values at 1%, 5%, 10%
    cv1  = T(cv[1])   # 1% critical value
    cv5  = T(cv[5])   # 5% critical value
    cv10 = T(cv[10])  # 10% critical value

    if stat >= cv1
        # Beyond 1% CV — use chi-squared tail for extrapolation
        pval = ccdf(Chisq(k), stat)
        return clamp(pval, zero(T), T(0.01))
    elseif stat >= cv5
        # Between 1% and 5%: linear interpolation
        frac = (stat - cv5) / max(cv1 - cv5, T(1e-10))
        return T(0.05) - frac * T(0.04)
    elseif stat >= cv10
        # Between 5% and 10%: linear interpolation
        frac = (stat - cv10) / max(cv5 - cv10, T(1e-10))
        return T(0.10) - frac * T(0.05)
    else
        # Below 10% CV: use chi-squared for approximate large p-value
        pval = ccdf(Chisq(k), stat)
        return clamp(pval, T(0.10), one(T))
    end
end
