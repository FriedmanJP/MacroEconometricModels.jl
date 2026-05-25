# X-13ARIMA-SEATS main API (internal)

function _x13_run(y::Vector{Float64};
                  frequency::Int=12, start::Tuple{Int,Int}=(1,1),
                  transform::Symbol=:auto,
                  model::Union{_X13ARIMASpec,Symbol}=:auto,
                  trading_day::Bool=false, easter::Bool=false, easter_window::Int=8,
                  user_regressors::Matrix{Float64}=Matrix{Float64}(undef, 0, 0),
                  outliers::Bool=true,
                  outlier_types::Vector{_X13OutlierType}=_X13OutlierType[_X13_AO, _X13_LS, _X13_TC],
                  critical_value::Float64=0.0,
                  x11::Bool=true, x11_mode::Symbol=:auto,
                  x11_seasonal_filter::Symbol=:auto, x11_henderson::Int=0,
                  seats::Bool=true, forecast_horizon::Int=0)
    ts = _X13TimeSeries(copy(y), start, frequency)
    n = length(y)
    working_y = copy(y); selected_transform = _X13_NONE
    if transform == :auto
        all(y .> 0) && (selected_transform, _ = _x13_auto_transform(y, frequency))
    elseif transform == :log
        selected_transform = _X13_LOG
    end
    selected_transform == _X13_LOG && (working_y = log.(y))
    reg_spec = _X13RegressionSpec(trading_day=trading_day, easter=easter,
                                  easter_window=easter_window, user=user_regressors)
    X = _x13_build_regressors(ts, reg_spec)
    detected_outliers = _X13Outlier[]
    local est_model::_X13ARIMAModel
    if model === :auto
        est_model, detected_outliers = _x13_auto_model(working_y, frequency)
    else
        est_model = _X13ARIMAModel(model)
        _x13_estimate!(est_model, working_y, X)
        if outliers
            detected_outliers = _x13_detect_outliers!(est_model, working_y, X;
                types=outlier_types, critical_value=critical_value)
        end
    end
    residuals_vec = _x13_compute_final_residuals(est_model, working_y, X)
    h = forecast_horizon > 0 ? forecast_horizon : frequency
    fc = try _x13_forecast(est_model, working_y, h) catch; zeros(h) end
    bc = try _x13_backcast(est_model, working_y, h) catch; zeros(h) end
    x11_result = nothing
    if x11
        x11_spec = _X13X11Spec(mode=x11_mode, seasonal_filter=x11_seasonal_filter,
                                henderson_length=x11_henderson)
        x11_y = selected_transform == _X13_LOG ? exp.(working_y) : working_y
        x11_fc = selected_transform == _X13_LOG ? exp.(fc) : fc
        x11_bc = selected_transform == _X13_LOG ? exp.(bc) : bc
        try x11_result = _x13_x11(x11_y, frequency, x11_spec; forecasts=x11_fc, backcasts=x11_bc)
        catch; end
    end
    seats_result = nothing
    if seats
        seats_spec = _X13SEATSSpec()
        try seats_result = _x13_seats(est_model, working_y, frequency, seats_spec)
        catch; end
    end
    return _X13InternalResult(ts, selected_transform, est_model, residuals_vec,
                              detected_outliers, Dict{String,Float64}(), x11_result, seats_result)
end

function _x13_compute_final_residuals(model, y, X)
    n = length(y)
    try
        filtered_y, filtered_X, na, _, info = _x13_armafl!(copy(y), model, copy(X))
        (info != 0 || na == 0) && return zeros(n)
        if size(filtered_X, 2) > 0
            ncxy = size(filtered_X, 2) + 1
            Xy = hcat(filtered_X, filtered_y)
            b = zeros(size(filtered_X, 2))
            pxpx = ncxy * (ncxy + 1) ÷ 2; chlxpx = zeros(pxpx)
            _x13_olsreg!(Xy, na, ncxy, ncxy, b, chlxpx, pxpx)
            r = zeros(na)
            _x13_compute_residuals!(Xy, na, ncxy, ncxy, 1, ncxy-1, -1.0, b, r)
            return vcat(zeros(n - na), r)
        else
            return vcat(zeros(n - length(filtered_y)), filtered_y)
        end
    catch; return zeros(n); end
end
