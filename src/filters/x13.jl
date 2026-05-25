# X-13ARIMA-SEATS seasonal adjustment
#
# Pure-Julia port of the US Census Bureau's X-13ARIMA-SEATS program.
# Provides both X-11 (iterative moving average) and SEATS (Wiener-Kolmogorov
# spectral decomposition) methods for seasonal adjustment.
#
# Original software: US Census Bureau X-13ARIMA-SEATS v1.1 Build 62
# Dagum & Bianconcini (2016), "Seasonal Adjustment Methods and Real Time
# Trend-Cycle Estimation", Springer.

"""
    x13_filter(y; frequency=12, method=:seats, kwargs...) -> X13FilterResult{T}

X-13ARIMA-SEATS seasonal adjustment.

Decomposes a seasonal time series into trend, seasonal, and irregular components
using automatic ARIMA model identification (TRAMO-style) followed by either
X-11 (iterative moving average) or SEATS (spectral decomposition).

# Arguments
- `y::AbstractVector`: Time series data (length ≥ 3 × frequency)

# Keyword Arguments
- `frequency::Int=12`: Seasonal period (4 for quarterly, 12 for monthly)
- `method::Symbol=:seats`: Preferred decomposition (`:seats` or `:x11`; both are computed, this selects the output)
- `start::Tuple{Int,Int}=(1,1)`: Start date as (year, period)
- `transform::Symbol=:auto`: Pre-transformation (`:auto`, `:log`, or `:none`)
- `model::Symbol=:auto`: ARIMA model (`:auto` for automatic identification)
- `trading_day::Bool=false`: Include trading day regressors
- `easter::Bool=false`: Include Easter effect regressor
- `outliers::Bool=true`: Detect additive outliers, level shifts, temporary changes

# Returns
An `X13FilterResult{T}` with fields:
- `trend`: Trend-cycle component
- `seasonal`: Seasonal component
- `irregular`: Irregular component
- `adjusted`: Seasonally adjusted series
- `original`: Original input series
- `method`: Decomposition method used
- `arima_order`: Fitted (p,d,q,P,D,Q) specification

# Examples
```julia
y = 100.0 .+ 10.0 .* sin.(2π .* (1:120) ./ 12) .+ randn(120)
r = x13_filter(y; frequency=12)
report(r)
sa = r.adjusted    # seasonally adjusted series
```
"""
function x13_filter(y::AbstractVector{T};
                    frequency::Int=12,
                    method::Symbol=:seats,
                    start::Tuple{Int,Int}=(1,1),
                    transform::Symbol=:auto,
                    model::Symbol=:auto,
                    trading_day::Bool=false,
                    easter::Bool=false,
                    easter_window::Int=8,
                    outliers::Bool=true,
                    critical_value::Float64=0.0) where {T<:AbstractFloat}
    n = length(y)
    n < 3 * frequency && throw(ArgumentError(
        "x13_filter requires at least 3 × frequency = $(3 * frequency) observations, got $n"))
    frequency ∉ (4, 12) && throw(ArgumentError(
        "x13_filter supports frequency 4 (quarterly) or 12 (monthly), got $frequency"))
    method ∉ (:seats, :x11) && throw(ArgumentError(
        "method must be :seats or :x11, got :$method"))

    yf = Float64.(y)

    result = _x13_run(yf; frequency=frequency, start=start, transform=transform,
                      model=model === :auto ? :auto : :auto,
                      trading_day=trading_day, easter=easter, easter_window=easter_window,
                      outliers=outliers, critical_value=critical_value,
                      x11=true, seats=true)

    spec = result.model.spec
    arima_order = (spec.p, spec.d, spec.q, spec.P, spec.D, spec.Q)
    trans_sym = result.transform == _X13_LOG ? :log : :none

    # Extract components from preferred method
    used_method = method
    if method == :seats && result.seats !== nothing
        s = result.seats
        trend_v = result.transform == _X13_LOG ? exp.(s.trend) : s.trend
        seas_v = result.transform == _X13_LOG ? exp.(s.seasonal) : s.seasonal
        irr_v = result.transform == _X13_LOG ? exp.(s.irregular) : s.irregular
        adj_v = trend_v .+ irr_v
        if result.transform == _X13_LOG
            adj_v = yf ./ seas_v
        end
    elseif result.x11 !== nothing
        used_method = :x11
        x = result.x11
        trend_v = x.trend
        seas_v = x.seasonal
        irr_v = x.irregular
        adj_v = x.seasonally_adjusted
    else
        throw(ErrorException("X-13 decomposition failed: neither X-11 nor SEATS produced results"))
    end

    X13FilterResult{T}(
        T.(trend_v), T.(seas_v), T.(irr_v), T.(adj_v), Vector{T}(y),
        used_method, arima_order, frequency, trans_sym,
        T(result.model.sigma2), T(result.model.aic),
        length(result.outliers), n
    )
end

x13_filter(y::AbstractVector; kwargs...) = x13_filter(Float64.(y); kwargs...)

"""
    seasonal(r::X13FilterResult) -> Vector

Return the seasonal component from an X-13 result.
"""
seasonal(r::X13FilterResult) = r.seasonal

"""
    adjusted(r::X13FilterResult) -> Vector

Return the seasonally adjusted series from an X-13 result.
"""
adjusted(r::X13FilterResult) = r.adjusted
