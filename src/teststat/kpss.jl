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
KPSS stationarity test.
"""

using Statistics

"""
    kpss_test(y; regression=:constant, bandwidth=:auto) -> KPSSResult

Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

Tests H₀: y is stationary against H₁: y has a unit root.

# Arguments
- `y`: Time series vector
- `regression`: :constant (level stationarity) or :trend (trend stationarity)
- `bandwidth`: Bartlett kernel bandwidth, or :auto for Newey-West selection

# Returns
`KPSSResult` containing test statistic, p-value, critical values, etc.

# Example
```julia
y = randn(200)  # Stationary series
result = kpss_test(y)
result.pvalue > 0.05  # Should fail to reject H₀ (stationarity)
```

# References
- Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). Testing
  the null hypothesis of stationarity against the alternative of a unit root.
  Journal of Econometrics, 54(1-3), 159-178.
"""
function kpss_test(y::AbstractVector{T};
                   regression::Symbol=:constant,
                   bandwidth::Union{Int,Symbol}=:auto) where {T<:AbstractFloat}

    regression ∈ (:constant, :trend) ||
        throw(ArgumentError("regression must be :constant or :trend"))

    n = length(y)
    n < 10 && throw(ArgumentError("Time series too short (n=$n), need at least 10 observations"))

    # Detrend the series
    if regression == :constant
        resid = y .- mean(y)
    else  # :trend
        t = T.(1:n)
        X = hcat(ones(T, n), t)
        B = X \ y
        resid = y - X * B
    end

    # Compute partial sums
    S = cumsum(resid)

    # Determine bandwidth
    bw = bandwidth == :auto ? _nw_bandwidth(resid) : bandwidth

    # Long-run variance estimate
    lrv = _long_run_variance(resid, bw)

    # KPSS statistic
    stat = sum(S.^2) / (n^2 * lrv)

    # Critical values and p-value
    cv = Dict{Int,T}(k => T(v) for (k, v) in KPSS_CRITICAL_VALUES[regression])
    pval = kpss_pvalue(stat, regression)

    KPSSResult(stat, pval, regression, cv, bw, n)
end

kpss_test(y::AbstractVector; kwargs...) = kpss_test(Float64.(y); kwargs...)
