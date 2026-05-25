# ARIMA forecasting and backcasting for X-13

function _x13_forecast(model::_X13ARIMAModel, y::Vector{Float64}, h::Int)
    spec = model.spec; s = spec.frequency
    diff_poly = [1.0]
    for _ in 1:spec.d; diff_poly = _x13_poly_multiply(diff_poly, [1.0, -1.0]); end
    for _ in 1:spec.D
        seas_diff = zeros(s + 1); seas_diff[1] = 1.0; seas_diff[end] = -1.0
        diff_poly = _x13_poly_multiply(diff_poly, seas_diff)
    end
    ar_poly = zeros(spec.p + 1); ar_poly[1] = 1.0
    for i in 1:spec.p; ar_poly[i + 1] = -model.ar[i]; end
    sar_poly = if spec.P > 0
        sp = zeros(spec.P * s + 1); sp[1] = 1.0
        for i in 1:spec.P; sp[i * s + 1] = -model.sar[i]; end; sp
    else [1.0] end
    full_ar = _x13_poly_multiply(_x13_poly_multiply(ar_poly, sar_poly), diff_poly)
    order = length(full_ar) - 1; n = length(y)
    yext = vcat(y, zeros(h))
    for t in (n + 1):(n + h)
        val = 0.0
        for k in 1:min(order, t - 1); val -= full_ar[k + 1] * yext[t - k]; end
        yext[t] = val
    end
    return yext[(n + 1):end]
end

function _x13_backcast(model::_X13ARIMAModel, y::Vector{Float64}, h::Int)
    return reverse(_x13_forecast(model, reverse(y), h))
end
