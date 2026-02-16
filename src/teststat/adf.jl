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
Augmented Dickey-Fuller (ADF) unit root test.
"""

"""
    adf_test(y; lags=:aic, max_lags=nothing, regression=:constant) -> ADFResult

Augmented Dickey-Fuller test for unit root.

Tests H₀: y has a unit root (non-stationary) against H₁: y is stationary.

# Arguments
- `y`: Time series vector
- `lags`: Number of augmenting lags, or :aic/:bic/:hqic for automatic selection
- `max_lags`: Maximum lags for automatic selection (default: floor(12*(T/100)^0.25))
- `regression`: Deterministic terms - :none, :constant (default), or :trend

# Returns
`ADFResult` containing test statistic, p-value, critical values, etc.

# Example
```julia
y = cumsum(randn(200))  # Random walk (has unit root)
result = adf_test(y)
result.pvalue > 0.05  # Should fail to reject H₀
```

# References
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for
  autoregressive time series with a unit root. JASA, 74(366), 427-431.
- MacKinnon, J. G. (2010). Critical values for cointegration tests.
  Queen's Economics Department Working Paper No. 1227.
"""
function adf_test(y::AbstractVector{T};
                  lags::Union{Int,Symbol}=:aic,
                  max_lags::Union{Int,Nothing}=nothing,
                  regression::Symbol=:constant) where {T<:AbstractFloat}

    regression ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("regression must be :none, :constant, or :trend"))

    n = length(y)
    n < 20 && throw(ArgumentError("Time series too short (n=$n), need at least 20 observations"))

    # Determine maximum lags for selection
    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags

    # Select lag length
    p = if lags isa Symbol
        lags ∈ (:aic, :bic, :hqic) ||
            throw(ArgumentError("lags must be an integer or :aic/:bic/:hqic"))
        adf_select_lags(y, max_p, regression, lags)
    else
        lags
    end

    # Compute first differences
    dy = diff(y)

    # Build regression matrix
    X = _build_adf_matrix(y, dy, p, regression)
    Y = dy[(p+1):end]
    nobs = length(Y)

    # OLS estimation
    XtX = X'X
    XtX_inv = inv(XtX)
    B = XtX_inv * (X'Y)
    resid = Y - X * B

    # Compute standard error of γ coefficient
    sigma2 = sum(resid.^2) / (nobs - size(X, 2))
    se = sqrt.(sigma2 * diag(XtX_inv))

    # γ is the coefficient on y_{t-1}
    gamma_idx = regression == :none ? 1 : (regression == :constant ? 2 : 3)
    gamma = B[gamma_idx]
    gamma_se = se[gamma_idx]

    # ADF test statistic
    stat = gamma / gamma_se

    # Critical values and p-value
    cv = adf_critical_values(regression, nobs, T)
    pval = adf_pvalue(stat, regression, nobs)

    ADFResult(stat, pval, p, regression, cv, nobs)
end

# Float64 fallback
adf_test(y::AbstractVector; kwargs...) = adf_test(Float64.(y); kwargs...)
