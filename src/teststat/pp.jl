# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Phillips-Perron unit root test.
"""

using LinearAlgebra, Statistics

"""
    pp_test(y; regression=:constant, bandwidth=:auto) -> PPResult

Phillips-Perron test for unit root with non-parametric correction.

Tests H₀: y has a unit root against H₁: y is stationary.

# Arguments
- `y`: Time series vector
- `regression`: :none, :constant (default), or :trend
- `bandwidth`: Newey-West bandwidth, or :auto for automatic selection

# Returns
`PPResult` containing test statistic (Zt), p-value, critical values, etc.

# Example
```julia
y = cumsum(randn(200))  # Random walk
result = pp_test(y)
result.pvalue > 0.05  # Should fail to reject H₀
```

# References
- Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time
  series regression. Biometrika, 75(2), 335-346.
"""
function pp_test(y::AbstractVector{T};
                 regression::Symbol=:constant,
                 bandwidth::Union{Int,Symbol}=:auto) where {T<:AbstractFloat}

    regression ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("regression must be :none, :constant, or :trend"))

    n = length(y)
    n < 20 && throw(ArgumentError("Time series too short (n=$n), need at least 20 observations"))

    # Build regression: y_t = α + β*t + ρ*y_{t-1} + u_t
    y_lag = y[1:end-1]
    y_curr = y[2:end]
    nobs = n - 1

    if regression == :none
        X = reshape(y_lag, :, 1)
    elseif regression == :constant
        X = hcat(ones(T, nobs), y_lag)
    else  # :trend
        t = T.(1:nobs)
        X = hcat(ones(T, nobs), t, y_lag)
    end

    # OLS via QR (same conditioning hazard as adf_test; #584): normal equations
    # square cond(X) and a pinv fallback would silently corrupt the statistic.
    qrX = qr(X)
    B = qrX \ y_curr
    Rinv = inv(UpperTriangular(Matrix(qrX.R)))
    XtX_inv = Rinv * Rinv'
    resid = y_curr - X * B

    # Standard error under homoskedasticity
    sigma2 = sum(resid.^2) / (nobs - size(X, 2))
    se = sqrt.(sigma2 * diag(XtX_inv))

    # Coefficient on y_{t-1}
    rho_idx = regression == :none ? 1 : (regression == :constant ? 2 : 3)
    rho = B[rho_idx]
    rho_se = se[rho_idx]

    # t-statistic (uncorrected)
    t_rho = (rho - 1) / rho_se

    # Bandwidth for long-run variance
    bw = bandwidth == :auto ? _nw_bandwidth(resid) : bandwidth

    # Long-run variance and short-run variance
    gamma0 = var(resid; corrected=false)
    lambda2 = _long_run_variance(resid, bw)

    # Phillips-Perron Z_t (Phillips 1987 / Hamilton 17.6.8):
    #   Z_t = √(γ₀/λ²)·t_ρ − (λ²−γ₀)·T·se(ρ̂) / (2·λ·s)
    # With se(ρ̂) = s·√[(X'X)⁻¹]_ρρ = s/√S̃ the correction is
    # (λ²−γ₀)·T / (2·λ·√S̃), where S̃ = 1/(X'X)⁻¹_ρρ is the residualized
    # Σ ỹ²_{t−1} — scale-invariant AND O(1) in T (#576: an earlier revision
    # omitted the factor T, making the correction vanish asymptotically; the
    # pre-#576 form divided by se_ρ·√T and carried the units of y).
    Sll_eff = one(T) / max(XtX_inv[rho_idx, rho_idx], eps(T))
    stat = sqrt(gamma0 / lambda2) * t_rho -
           (lambda2 - gamma0) * nobs / (2 * sqrt(lambda2) * sqrt(Sll_eff))

    # Critical values (same as ADF with 0 lags)
    cv = adf_critical_values(regression, nobs, 0, T)
    pval = adf_pvalue(stat, regression, nobs, 0)

    PPResult(stat, pval, regression, cv, bw, nobs)
end

pp_test(y::AbstractVector; kwargs...) = pp_test(Float64.(y); kwargs...)
