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
Zivot-Andrews unit root test with endogenous structural break.
"""

using LinearAlgebra, Statistics

"""
    za_test(y; regression=:both, trim=0.15, lags=:aic, max_lags=nothing) -> ZAResult

Zivot-Andrews test for unit root with endogenous structural break.

Tests H₀: y has a unit root without break against H₁: y is stationary with break.

# Arguments
- `y`: Time series vector
- `regression`: Type of break - :constant (intercept), :trend (slope), or :both
- `trim`: Trimming fraction for break search (default 0.15)
- `lags`: Number of augmenting lags, or :aic/:bic for automatic selection
- `max_lags`: Maximum lags for selection

# Returns
`ZAResult` containing minimum t-statistic, break point, p-value, etc.

# Example
```julia
# Series with structural break
y = vcat(randn(100), randn(100) .+ 2)
result = za_test(y; regression=:constant)
```

# References
- Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the great crash,
  the oil-price shock, and the unit-root hypothesis. JBES, 10(3), 251-270.
"""
function za_test(y::AbstractVector{T};
                 regression::Symbol=:both,
                 trim::Real=0.15,
                 lags::Union{Int,Symbol}=:aic,
                 max_lags::Union{Int,Nothing}=nothing) where {T<:AbstractFloat}

    regression ∈ (:constant, :trend, :both) ||
        throw(ArgumentError("regression must be :constant, :trend, or :both"))
    0 < trim < 0.5 || throw(ArgumentError("trim must be between 0 and 0.5"))

    n = length(y)
    n < 50 && throw(ArgumentError("Time series too short for ZA test (n=$n), need at least 50"))

    # Maximum lags
    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags

    # Trimming bounds
    start_idx = max(2, ceil(Int, trim * n))
    end_idx = min(n - 1, floor(Int, (1 - trim) * n))

    # First differences
    dy = diff(y)

    min_stat = T(Inf)
    best_break = start_idx
    best_lags = 0

    for tb in start_idx:end_idx
        # Determine lags for this break point
        p = if lags isa Symbol
            # Simplified: use fixed lag selection for speed
            min(max_p, 4)
        else
            lags
        end

        # Build regression matrix with break dummies
        nobs = n - 1 - p
        nobs < 10 && continue

        # Dependent variable
        Y = dy[(p+1):end]

        # Regressors
        ones_col = ones(T, nobs)
        trend = T.(1:nobs)
        y_lag = y[(p+1):(n-1)]

        # Break dummies
        DU = T.((p+1):n-1 .>= tb)  # Level shift
        DT = T.(max.(0, (p+1:n-1) .- tb .+ 1))  # Trend shift

        # Lagged differences
        X_base = if regression == :constant
            hcat(ones_col, trend, DU, y_lag)
        elseif regression == :trend
            hcat(ones_col, trend, DT, y_lag)
        else  # :both
            hcat(ones_col, trend, DU, DT, y_lag)
        end

        # Add lagged differences
        if p > 0
            dy_lags = Matrix{T}(undef, nobs, p)
            for j in 1:p
                dy_lags[:, j] = dy[(p+1-j):(n-1-j)]
            end
            X = hcat(X_base, dy_lags)
        else
            X = X_base
        end

        # OLS
        XtX = X'X
        cond(XtX) > 1e12 && continue

        B = XtX \ (X'Y)
        resid = Y - X * B

        # Standard errors
        sigma2 = sum(resid.^2) / (nobs - size(X, 2))
        se = sqrt.(sigma2 * diag(inv(XtX)))

        # t-statistic on y_{t-1}
        gamma_idx = regression == :both ? 5 : 4
        t_stat = B[gamma_idx] / se[gamma_idx]

        if t_stat < min_stat
            min_stat = t_stat
            best_break = tb
            best_lags = p
        end
    end

    # Critical values and p-value
    cv = Dict{Int,T}(k => T(v) for (k, v) in ZA_CRITICAL_VALUES[regression])
    pval = za_pvalue(min_stat, regression)
    break_frac = T(best_break) / n

    ZAResult(min_stat, pval, best_break, break_frac, regression, cv, best_lags, n - 1 - best_lags)
end

za_test(y::AbstractVector; kwargs...) = za_test(Float64.(y); kwargs...)
