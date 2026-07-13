# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Johansen cointegration test for VAR systems.
"""

using LinearAlgebra


"""
    _johansen_pvalue(stat, m, deterministic, test) -> p

Asymptotic p-value for the Johansen trace or maximum-eigenvalue statistic via
the Doornik (1998) gamma approximation: the asymptotic distribution for
`m = n - r` common trends under the null is approximated by
`Gamma(mean^2/var, var/mean)` with `mean` and `var` from case-specific response
surfaces, reproducing the MacKinnon-Haug-Michelis (1999) numerical
distributions to simulation accuracy.

Deterministic cases map to Doornik/gretl rows: `:none` -> no constant,
`:constant` -> restricted constant, `:trend` -> restricted trend (matching the
Osterwald-Lenum critical-value tables used for the rank decision).
"""
function _johansen_pvalue(stat::T, m::Int, deterministic::Symbol,
                          test::Symbol) where {T<:AbstractFloat}
    case = deterministic == :none ? 1 :
           deterministic == :constant ? 2 : 4
    x = (Float64(m)^2, Float64(m), 1.0,
         m == 1 ? 1.0 : 0.0, m == 2 ? 1.0 : 0.0, sqrt(Float64(m)))
    local mn::Float64, vr::Float64
    if test == :trace
        mn = sum(x[i] * JOHANSEN_TRACE_M_COEF[case, i] for i in 1:6)
        vr = sum(x[i] * JOHANSEN_TRACE_V_COEF[case, i] for i in 1:6)
    else
        mn = sum(x[i+1] * JOHANSEN_MAXEV_M_COEF[case, i] for i in 1:5)
        vr = sum(x[i+1] * JOHANSEN_MAXEV_V_COEF[case, i] for i in 1:5)
    end
    (isfinite(stat) && stat > zero(T)) || return one(T)
    p = 1.0 - cdf(Gamma(mn^2 / vr, vr / mn), Float64(stat))
    clamp(T(p), zero(T), one(T))
end

"""
    johansen_test(Y, p; deterministic=:constant) -> JohansenResult

Johansen cointegration test for VAR system.

Tests for the number of cointegrating relationships among variables using
trace and maximum eigenvalue tests.

# Arguments
- `Y`: Data matrix (T × n)
- `p`: Number of lags in the VECM representation
- `deterministic`: Specification for deterministic terms
  - :none - No deterministic terms
  - :constant - Constant in cointegrating relation (default)
  - :trend - Linear trend in levels

# Returns
`JohansenResult` containing trace and max-eigenvalue statistics, cointegrating
vectors, adjustment coefficients, and estimated rank.

# Example
```julia
# Generate cointegrated system
n, T = 3, 200
Y = randn(T, n)
Y[:, 2] = Y[:, 1] + 0.1 * randn(T)  # Y2 cointegrated with Y1

result = johansen_test(Y, 2)
result.rank  # Should detect 1 or 2 cointegrating relations
```

# References
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration
  vectors in Gaussian vector autoregressive models. Econometrica, 59(6), 1551-1580.
- Osterwald-Lenum, M. (1992). A note with quantiles of the asymptotic
  distribution of the ML cointegration rank test statistics. Oxford BEJM.
- MacKinnon, J.G., Haug, A.A. & Michelis, L. (1999). Numerical distribution
  functions of likelihood ratio tests for cointegration. JAE, 14(5), 563-577.
- Doornik, J.A. (1998). Approximations to the asymptotic distributions of
  cointegration tests. Journal of Economic Surveys, 12(5), 573-593.
"""
function johansen_test(Y::AbstractMatrix{T}, p::Int;
                       deterministic::Symbol=:constant,
                       significance::Real=0.05) where {T<:AbstractFloat}

    deterministic ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("deterministic must be :none, :constant, or :trend"))

    T_obs, n = size(Y)
    T_obs < n + p + 10 && throw(ArgumentError("Not enough observations for Johansen test"))
    p < 1 && throw(ArgumentError("Number of lags p must be at least 1"))

    # VECM representation: ΔYₜ = αβ'Yₜ₋₁ + Σᵢ Γᵢ ΔYₜ₋ᵢ + det + εₜ
    # Johansen (1991) Cases:
    #   :none    = Case 1: no deterministic terms
    #   :constant = Case 2: restricted constant in cointegrating relation
    #   :trend   = Case 4: restricted trend + unrestricted constant

    # Construct matrices
    dY = diff(Y, dims=1)  # ΔY: (T-1) × n
    Y_lag = Y[p:end-1, :]  # Y_{t-1}: (T-p) × n

    # Lagged differences
    T_eff = T_obs - p
    dY_lags = if p > 1
        hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...)
    else
        Matrix{eltype(Y)}(undef, T_eff, 0)
    end

    # Dependent variable
    dY_eff = dY[p:end, :]

    # Deterministic terms and augmented Y_lag
    # Case 1 (:none): Z = lagged diffs only; Y_lag unaugmented
    # Case 2 (:constant): constant restricted to cointegrating space (augment Y_lag)
    # Case 4 (:trend): trend restricted, constant unrestricted in Z
    # Use T_float for the element type to avoid confusion with dimensions
    T_float = T

    if deterministic == :none
        Z = dY_lags
        Y_lag_aug = Y_lag
    elseif deterministic == :constant
        # Case 2: restrict constant to cointegrating relation
        # Augment Y_lag with ones so constant enters β'[Y_{t-1}; 1]
        Z = dY_lags  # no unrestricted deterministic terms
        Y_lag_aug = hcat(Y_lag, ones(T_float, T_eff))
    else  # :trend
        # Case 4: restrict trend, keep constant unrestricted
        Z = if isempty(dY_lags)
            reshape(ones(T_float, T_eff), :, 1)
        else
            hcat(ones(T_float, T_eff), dY_lags)
        end
        Y_lag_aug = hcat(Y_lag, T_float.(1:T_eff))
    end

    # Concentrate out short-run dynamics via least squares projection
    if size(Z, 2) > 0
        R0 = dY_eff - Z * (Z \ dY_eff)
        R1 = Y_lag_aug - Z * (Z \ Y_lag_aug)
    else
        R0 = dY_eff
        R1 = Y_lag_aug
    end

    # Dimension of the augmented system (n + number of restricted deterministic terms)
    n_aug = size(Y_lag_aug, 2)

    # Moment matrices
    S00 = (R0'R0) / T_eff
    S11 = (R1'R1) / T_eff
    S01 = (R0'R1) / T_eff
    S10 = S01'

    # Solve generalized eigenvalue problem
    # |λS₁₁ - S₁₀S₀₀⁻¹S₀₁| = 0
    S00_inv = robust_inv(S00)
    A = S11 \ (S10 * S00_inv * S01)

    # Eigendecomposition (n_aug × n_aug for augmented systems)
    eig = eigen(A)
    idx = sortperm(real.(eig.values), rev=true)
    eigenvalues_all = real.(eig.values[idx])
    eigenvectors_all = real.(eig.vectors[:, idx])

    # Use only the first n eigenvalues for test statistics
    eigenvalues = clamp.(eigenvalues_all[1:n], 0, 1 - eps(T))

    # Test statistics
    trace_stats = Vector{T}(undef, n)
    max_eigen_stats = Vector{T}(undef, n)

    for r in 0:(n-1)
        # Trace statistic: -T Σᵢ₌ᵣ₊₁ⁿ ln(1 - λᵢ)
        trace_stats[r+1] = -T_eff * sum(log.(1 .- eigenvalues[(r+1):n]))
        # Max eigenvalue statistic: -T ln(1 - λᵣ₊₁)
        max_eigen_stats[r+1] = -T_eff * log(1 - eigenvalues[r+1])
    end

    # Select critical values based on deterministic specification
    cv_trace_tbl, cv_max_tbl = if deterministic == :none
        JOHANSEN_TRACE_CV_NONE, JOHANSEN_MAX_CV_NONE
    elseif deterministic == :constant
        JOHANSEN_TRACE_CV_CONSTANT, JOHANSEN_MAX_CV_CONSTANT
    else  # :trend
        JOHANSEN_TRACE_CV_TREND, JOHANSEN_MAX_CV_TREND
    end

    cv_trace = Matrix{T}(undef, n, 3)
    cv_max = Matrix{T}(undef, n, 3)

    for r in 0:(n-1)
        n_minus_r = n - r
        if haskey(cv_trace_tbl, n_minus_r)
            cv_trace[r+1, :] = T.(cv_trace_tbl[n_minus_r])
            cv_max[r+1, :] = T.(cv_max_tbl[n_minus_r])
        else
            # Extrapolate for large systems (approximate)
            cv_trace[r+1, :] = T.([6.5 + 10*n_minus_r, 8.18 + 10*n_minus_r, 11.65 + 12*n_minus_r])
            cv_max[r+1, :] = T.([6.5 + 6*n_minus_r, 8.18 + 6*n_minus_r, 11.65 + 7*n_minus_r])
        end
    end

    # P-values: Doornik (1998) gamma approximation to the MHM (1999) asymptotic
    # distributions (m = n - r common trends under each null). The rank decision
    # below still uses the tabulated Osterwald-Lenum critical values, which can
    # differ slightly from the asymptotic surfaces (OL simulations drift low for
    # large m), so p-values near a CV need not equal the nominal level exactly.
    trace_pvalues = Vector{T}(undef, n)
    max_pvalues = Vector{T}(undef, n)

    for r in 1:n
        m_ct = n - (r - 1)
        trace_pvalues[r] = _johansen_pvalue(trace_stats[r], m_ct, deterministic, :trace)
        max_pvalues[r] = _johansen_pvalue(max_eigen_stats[r], m_ct, deterministic, :max)
    end

    # Estimated cointegration rank = number of leading trace-test rejections (B1/T171):
    # rejecting H₀: rank ≤ r (trace stat > CV) implies rank ≥ r+1. Shared with the VECM
    # selector so the two never disagree.
    rank = _rank_from_trace(trace_stats, cv_trace, significance)

    # Cointegrating vectors and adjustment coefficients
    # For augmented systems, extract only the n variable rows from eigenvectors
    r_eff = max(1, rank)
    beta_aug = eigenvectors_all[:, 1:r_eff]  # full augmented eigenvectors
    beta = beta_aug[1:n, :]  # β: cointegrating vectors (n × r)
    alpha = S01 * beta_aug * robust_inv(beta_aug' * S11 * beta_aug)  # α: adjustment (n × r)

    JohansenResult(
        trace_stats, trace_pvalues,
        max_eigen_stats, max_pvalues,
        rank, beta, alpha, eigenvalues,
        cv_trace, cv_max,
        deterministic, p, T_eff
    )
end

johansen_test(Y::AbstractMatrix, p::Int; kwargs...) = johansen_test(Float64.(Y), p; kwargs...)

# Estimated cointegration rank = number of leading trace-test rejections (B1/T171).
# Rejecting H₀: rank ≤ r (trace stat > CV) implies rank ≥ r+1. Single source shared by
# johansen_test and estimate_vecm(rank=:auto) so the reported rank never disagrees.
function _rank_from_trace(trace_stats::AbstractVector, cv_trace::AbstractMatrix, significance::Real)
    cv_col = significance <= 0.01 ? 3 : significance <= 0.05 ? 2 : 1
    r = 0
    for i in 0:(length(trace_stats) - 1)
        if trace_stats[i+1] > cv_trace[i+1, cv_col]
            r = i + 1
        else
            break
        end
    end
    r
end

_select_rank_trace(joh::JohansenResult, significance::Real) =
    _rank_from_trace(joh.trace_stats, joh.critical_values_trace, significance)
