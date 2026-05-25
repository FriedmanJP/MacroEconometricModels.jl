# X-11 Seasonal Decomposition ported from X-13ARIMA-SEATS

function _x13_resolve_mode(y::AbstractVector{<:Real}, mode::Symbol)
    mode == :auto && return all(x -> x > 0.0, y) ? :multiplicative : :additive
    return mode
end

function _x13_centered_ma_2xs!(y::AbstractVector{<:Real}, s::Int, trend::AbstractVector{<:Real})
    n = length(y); half = s ÷ 2
    for i in 1:half; trend[i] = NaN; end
    for i in (n - half + 1):n; trend[i] = NaN; end
    for t in (half + 1):(n - half)
        total = 0.5 * y[t - half] + 0.5 * y[t + half]
        for j in (-half + 1):(half - 1); total += y[t + j]; end
        trend[t] = total / s
    end
    return trend
end

function _x13_fill_nan_endpoints!(x::AbstractVector{<:Real})
    n = length(x)
    first_valid = findfirst(!isnan, x)
    first_valid === nothing && return x
    for i in 1:(first_valid - 1); x[i] = x[first_valid]; end
    last_valid = findlast(!isnan, x)
    last_valid === nothing && return x
    for i in (last_valid + 1):n; x[i] = x[last_valid]; end
    return x
end

function _x13_compute_si!(y, trend, si, mode)
    for i in eachindex(y)
        if isnan(trend[i]); si[i] = NaN
        elseif mode == :multiplicative; si[i] = trend[i] == 0.0 ? 1.0 : y[i] / trend[i]
        else; si[i] = y[i] - trend[i]; end
    end
    return si
end

function _x13_normalize_seasonal!(seasonal, frequency, mode)
    n = length(seasonal); nyears = cld(n, frequency)
    for year in 0:(nyears - 1)
        i1 = year * frequency + 1; i2 = min((year + 1) * frequency, n)
        nperiods = i2 - i1 + 1; nperiods < 1 && continue
        if mode == :multiplicative
            avg = mean(@view seasonal[i1:i2])
            avg != 0.0 && (for i in i1:i2; seasonal[i] /= avg; end)
        else
            avg = mean(@view seasonal[i1:i2])
            for i in i1:i2; seasonal[i] -= avg; end
        end
    end
    return seasonal
end

function _x13_compute_seasonal_factors!(si, frequency, seasonal, mode)
    n = length(si); nyears = cld(n, frequency)
    for p in 1:frequency
        vals = Float64[]
        for year in 0:(nyears - 1)
            idx = p + year * frequency
            idx <= n && !isnan(si[idx]) && push!(vals, si[idx])
        end
        factor = isempty(vals) ? (mode == :multiplicative ? 1.0 : 0.0) : mean(vals)
        for year in 0:(nyears - 1)
            idx = p + year * frequency; idx <= n && (seasonal[idx] = factor)
        end
    end
    _x13_normalize_seasonal!(seasonal, frequency, mode)
    return seasonal
end

function _x13_deseasonalize!(y, seasonal, adjusted, mode)
    for i in eachindex(y)
        if mode == :multiplicative
            adjusted[i] = seasonal[i] == 0.0 ? y[i] : y[i] / seasonal[i]
        else; adjusted[i] = y[i] - seasonal[i]; end
    end
    return adjusted
end

function _x13_apply_henderson_filled!(y, hlen, trend)
    _x13_apply_henderson!(y, hlen, trend); _x13_fill_nan_endpoints!(trend)
    return trend
end

function _x13_select_henderson_length(frequency::Int, spec_length::Int)
    spec_length > 0 && return spec_length
    frequency == 4 ? 5 : 13
end

function _x13_replace_extremes!(si, seasonal, frequency, sigma_lower, sigma_upper, mode)
    n = length(si); nyears = cld(n, frequency)
    for p in 1:frequency
        indices = Int[]; vals = Float64[]
        for year in 0:(nyears - 1)
            idx = p + year * frequency
            idx <= n && (push!(indices, idx); push!(vals, si[idx]))
        end
        length(vals) < 3 && continue
        med = median(vals); σ = std(vals); σ < 1e-12 && continue
        for (j, idx) in enumerate(indices)
            abs(vals[j] - med) / σ > sigma_upper && (si[idx] = med)
        end
    end
    return si
end

function _x13_compute_seasonal_factors_ma!(si, frequency, seasonal, mode)
    n = length(si); nyears = cld(n, frequency)
    for p in 1:frequency
        indices = Int[]; vals = Float64[]
        for year in 0:(nyears - 1)
            idx = p + year * frequency
            idx <= n && (push!(indices, idx);
                push!(vals, isnan(si[idx]) ? (mode == :multiplicative ? 1.0 : 0.0) : si[idx]))
        end
        nv = length(vals); nv == 0 && continue
        smoothed = similar(vals)
        if nv >= 3
            smoothed[1] = (vals[1] + vals[2]) / 2.0
            for j in 2:(nv - 1); smoothed[j] = (vals[j-1] + vals[j] + vals[j+1]) / 3.0; end
            smoothed[nv] = (vals[nv-1] + vals[nv]) / 2.0
        elseif nv == 2; smoothed .= mean(vals)
        else; smoothed .= vals; end
        for (j, idx) in enumerate(indices); seasonal[idx] = smoothed[j]; end
    end
    _x13_normalize_seasonal!(seasonal, frequency, mode)
    return seasonal
end

function _x13_x11(y::AbstractVector{<:Real}, frequency::Int, spec::_X13X11Spec;
                  forecasts::Union{Nothing,AbstractVector{<:Real}}=nothing,
                  backcasts::Union{Nothing,AbstractVector{<:Real}}=nothing)
    n_orig = length(y)
    n_back = backcasts === nothing ? 0 : length(backcasts)
    n_fore = forecasts === nothing ? 0 : length(forecasts)
    if n_back > 0 || n_fore > 0
        y_ext = Vector{Float64}(undef, n_back + n_orig + n_fore)
        n_back > 0 && (y_ext[1:n_back] .= backcasts)
        y_ext[(n_back + 1):(n_back + n_orig)] .= y
        n_fore > 0 && (y_ext[(n_back + n_orig + 1):end] .= forecasts)
    else
        y_ext = Float64.(y)
    end
    n = length(y_ext)
    mode = _x13_resolve_mode(y_ext, spec.mode)
    hlen = _x13_select_henderson_length(frequency, spec.henderson_length)
    iseven(hlen) && (hlen += 1); hlen = max(hlen, 3)

    # Phase 1
    trend1 = zeros(n); _x13_centered_ma_2xs!(y_ext, frequency, trend1); _x13_fill_nan_endpoints!(trend1)
    si1 = zeros(n); _x13_compute_si!(y_ext, trend1, si1, mode)
    seas1 = zeros(n); _x13_compute_seasonal_factors!(si1, frequency, seas1, mode)
    sa1 = zeros(n); _x13_deseasonalize!(y_ext, seas1, sa1, mode)

    # Phase 2
    trend2 = zeros(n); _x13_apply_henderson_filled!(sa1, hlen, trend2)
    si2 = zeros(n); _x13_compute_si!(y_ext, trend2, si2, mode)
    seas2 = zeros(n); _x13_compute_seasonal_factors_ma!(si2, frequency, seas2, mode)
    si2_cleaned = copy(si2)
    _x13_replace_extremes!(si2_cleaned, seas2, frequency, spec.sigma_lower, spec.sigma_upper, mode)
    seas2b = zeros(n); _x13_compute_seasonal_factors_ma!(si2_cleaned, frequency, seas2b, mode)
    sa2 = zeros(n); _x13_deseasonalize!(y_ext, seas2b, sa2, mode)

    # Phase 3
    trend3 = zeros(n); _x13_apply_henderson_filled!(sa2, hlen, trend3)
    si3 = zeros(n); _x13_compute_si!(y_ext, trend3, si3, mode)
    seas3 = zeros(n); _x13_compute_seasonal_factors_ma!(si3, frequency, seas3, mode)
    si3_cleaned = copy(si3)
    _x13_replace_extremes!(si3_cleaned, seas3, frequency, spec.sigma_lower, spec.sigma_upper, mode)
    seas_final = zeros(n); _x13_compute_seasonal_factors_ma!(si3_cleaned, frequency, seas_final, mode)

    # Phase 4
    sa_final = zeros(n); _x13_deseasonalize!(y_ext, seas_final, sa_final, mode)
    irr_final = zeros(n)
    for i in 1:n
        if mode == :multiplicative
            irr_final[i] = trend3[i] == 0.0 ? 1.0 : sa_final[i] / trend3[i]
        else; irr_final[i] = sa_final[i] - trend3[i]; end
    end
    i1 = n_back + 1; i2 = n_back + n_orig
    return _X13X11Result(seas_final[i1:i2], sa_final[i1:i2], trend3[i1:i2], irr_final[i1:i2])
end
