# Henderson moving-average filter weights ported from X-13ARIMA-SEATS

function _x13_henderson_weights(n::Int)
    y  = (n + 3) / 2.0
    m  = (n + 1) ÷ 2
    y1 = (y - 1.0)^2; y2 = y^2; y3 = (y + 1.0)^2
    y4 = 3.0 * y2 - 16.0; y5 = 4.0 * y2
    denom = 8.0 * y * (y2 - 1.0) * (y5 - 1.0) * (y5 - 9.0) * (y5 - 25.0) / 315.0
    w = Vector{Float64}(undef, m)
    for i in 1:m
        x = Float64(i - 1)^2
        w[i] = (y1 - x) * (y2 - x) * (y3 - x) * (y4 - 11.0 * x) / denom
    end
    return w
end

function _x13_henderson_full_weights(n::Int)
    w = _x13_henderson_weights(n)
    return vcat(reverse(w[2:end]), w)
end

function _x13_apply_henderson!(y::AbstractVector{<:Real}, n::Int, trend::AbstractVector{<:Real})
    T_len = length(y)
    fw = _x13_henderson_full_weights(n)
    half = (n - 1) ÷ 2
    for i in 1:half; trend[i] = NaN; end
    for i in (T_len - half + 1):T_len; trend[i] = NaN; end
    for i in (half + 1):(T_len - half)
        s = 0.0
        for k in 1:n; s += fw[k] * y[i - half - 1 + k]; end
        trend[i] = s
    end
    return trend
end
