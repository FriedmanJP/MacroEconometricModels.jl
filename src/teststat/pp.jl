# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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

    # OLS estimation
    XtX = X'X
    XtX_inv = inv(XtX)
    B = XtX_inv * (X'y_curr)
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

    # Phillips-Perron Zt statistic
    # Zt = sqrt(γ₀/λ²) * t_ρ - (λ² - γ₀) / (2λ * se_ρ * sqrt(T))
    stat = sqrt(gamma0 / lambda2) * t_rho -
           (lambda2 - gamma0) / (2 * sqrt(lambda2) * rho_se * sqrt(nobs))

    # Critical values (same as ADF)
    cv = adf_critical_values(regression, nobs, T)
    pval = adf_pvalue(stat, regression, nobs)

    PPResult(stat, pval, regression, cv, bw, nobs)
end

pp_test(y::AbstractVector; kwargs...) = pp_test(Float64.(y); kwargs...)
