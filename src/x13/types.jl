# X-13ARIMA-SEATS types
# Ported from X-13ARIMA-SEATS v1.1 Build 62 (US Census Bureau)

@enum _X13Transform _X13_LOG _X13_NONE _X13_AUTO
@enum _X13OutlierType _X13_AO _X13_LS _X13_TC

struct _X13TimeSeries
    y::Vector{Float64}
    start::Tuple{Int,Int}
    frequency::Int
end

struct _X13ARIMASpec
    p::Int; d::Int; q::Int
    P::Int; D::Int; Q::Int
    frequency::Int
end

mutable struct _X13ARIMAModel
    spec::_X13ARIMASpec
    ar::Vector{Float64}
    ma::Vector{Float64}
    sar::Vector{Float64}
    sma::Vector{Float64}
    sigma2::Float64
    loglik::Float64
    aic::Float64
    aicc::Float64
    converged::Bool
    niter::Int
end

function _X13ARIMAModel(spec::_X13ARIMASpec)
    _X13ARIMAModel(spec,
        zeros(spec.p), zeros(spec.q),
        zeros(spec.P), zeros(spec.Q),
        NaN, NaN, NaN, NaN, false, 0)
end

struct _X13RegressionSpec
    trading_day::Bool
    easter::Bool
    easter_window::Int
    user::Matrix{Float64}
end

function _X13RegressionSpec(; trading_day=false, easter=false, easter_window=8,
                             user=Matrix{Float64}(undef, 0, 0))
    _X13RegressionSpec(trading_day, easter, easter_window, user)
end

struct _X13Outlier
    type::_X13OutlierType
    position::Int
    tstat::Float64
    coefficient::Float64
end

struct _X13X11Spec
    mode::Symbol
    seasonal_filter::Symbol
    henderson_length::Int
    sigma_lower::Float64
    sigma_upper::Float64
end

function _X13X11Spec(; mode=:auto, seasonal_filter=:auto, henderson_length=0,
                      sigma_lower=1.5, sigma_upper=2.5)
    _X13X11Spec(mode, seasonal_filter, henderson_length, sigma_lower, sigma_upper)
end

struct _X13SEATSSpec
    approximation::Symbol
    tasman::Float64
end

_X13SEATSSpec(; approximation=:none, tasman=1.0) = _X13SEATSSpec(approximation, tasman)

struct _X13X11Result
    seasonal::Vector{Float64}
    seasonally_adjusted::Vector{Float64}
    trend::Vector{Float64}
    irregular::Vector{Float64}
end

struct _X13SEATSResult
    trend::Vector{Float64}
    seasonal::Vector{Float64}
    irregular::Vector{Float64}
    trend_filter::Vector{Float64}
    seasonal_filter::Vector{Float64}
end

struct _X13InternalResult
    series::_X13TimeSeries
    transform::_X13Transform
    model::_X13ARIMAModel
    residuals::Vector{Float64}
    outliers::Vector{_X13Outlier}
    regression_coefficients::Dict{String,Float64}
    x11::Union{Nothing,_X13X11Result}
    seats::Union{Nothing,_X13SEATSResult}
end

struct _X13CompositeSpec
    weights::Vector{Float64}
    operation::Symbol
end
