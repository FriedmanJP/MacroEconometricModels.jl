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
Hamilton (2018) regression-based filter for trend-cycle decomposition.

Regresses y_{t+h} on [1, y_t, y_{t-1}, ..., y_{t-p+1}].
Residuals = cyclical component, fitted values = trend.

Reference: Hamilton, James D. 2018.
"Why You Should Never Use the Hodrick-Prescott Filter."
*Review of Economics and Statistics* 100 (5): 831–843.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    hamilton_filter(y::AbstractVector; h=8, p=4) -> HamiltonFilterResult

Apply the Hamilton (2018) regression filter for trend-cycle decomposition.

Regresses \$y_{t+h}\$ on \$[1, y_t, y_{t-1}, \\ldots, y_{t-p+1}]\$ by OLS.
The residuals form the cyclical component and the fitted values form the trend.

The filter loses `h + p - 1` observations at the start of the sample.

# Arguments
- `y::AbstractVector`: Time series data

# Keywords
- `h::Int=8`: Forecast horizon (default 8 for quarterly data = 2 years)
- `p::Int=4`: Number of lags in the regression (default 4 for quarterly data)

# Returns
- `HamiltonFilterResult{T}` with fields `trend`, `cycle`, `beta`, `h`, `p`,
  `T_obs`, `valid_range`

# Examples
```julia
y = cumsum(randn(200))
result = hamilton_filter(y)                 # quarterly defaults
result = hamilton_filter(y; h=24, p=12)     # monthly data (2-year horizon)
```

# References
- Hamilton, J. D. (2018). *REStat* 100(5): 831–843.
"""
function hamilton_filter(y::AbstractVector{T}; h::Int=8, p::Int=4) where {T<:AbstractFloat}
    T_obs = length(y)
    h < 1 && throw(ArgumentError("Forecast horizon h must be positive, got $h"))
    p < 1 && throw(ArgumentError("Number of lags p must be positive, got $p"))
    T_obs < h + p + 1 && throw(ArgumentError(
        "Not enough observations ($T_obs) for h=$h, p=$p. Need at least $(h + p + 1)."))

    yv = Vector{T}(y)

    # Build regression: y_{t+h} on [1, y_t, y_{t-1}, ..., y_{t-p+1}]
    # Valid range: t from p to T_obs - h, so dependent variable indices are (p+h):T_obs
    n_eff = T_obs - h - p + 1
    dep_start = h + p
    dep_end = T_obs

    # Dependent variable: y_{t+h}
    Y_dep = yv[dep_start:dep_end]

    # Regressors: [1, y_t, y_{t-1}, ..., y_{t-p+1}] where t = p:(T_obs-h)
    X = Matrix{T}(undef, n_eff, 1 + p)
    X[:, 1] .= one(T)
    @inbounds for lag in 0:(p - 1)
        # y_{t-lag} where t = p:(T_obs-h)
        col_start = p - lag
        col_end = T_obs - h - lag
        X[:, 2 + lag] .= yv[col_start:col_end]
    end

    # OLS: beta = (X'X)^{-1} X'Y
    beta = robust_inv(X' * X) * (X' * Y_dep)
    fitted = X * beta
    resid = Y_dep .- fitted

    valid_range = dep_start:dep_end

    HamiltonFilterResult(fitted, resid, beta, h, p, T_obs, valid_range)
end

# Float64 fallback for non-float input
hamilton_filter(y::AbstractVector; kwargs...) = hamilton_filter(Float64.(y); kwargs...)
