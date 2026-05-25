# Regression construction: trading day, Easter, and length-of-month regressors

function _x13_easter_date(year::Int)
    a = year % 19
    b, c = divrem(year, 100)
    d, e = divrem(b, 4)
    f = (b + 8) ÷ 25
    g = (b - f + 1) ÷ 3
    h = (19a + b - d - g + 15) % 30
    i, k = divrem(c, 4)
    l = (32 + 2e + 2i - h - k) % 7
    m = (a + 11h + 22l) ÷ 451
    month, day = divrem(h + l - 7m + 114, 31)
    return Date(year, month, day + 1)
end

function _x13_period_to_daterange(start::Tuple{Int,Int}, t::Int, frequency::Int)
    yr = start[1] + (start[2] + t - 2) ÷ frequency
    per = mod1(start[2] + t - 1, frequency)
    if frequency == 12
        d1 = Date(yr, per, 1); d2 = lastdayofmonth(d1)
    elseif frequency == 4
        m1 = (per - 1) * 3 + 1
        d1 = Date(yr, m1, 1); d2 = lastdayofmonth(Date(yr, m1 + 2, 1))
    else
        error("Unsupported frequency: $frequency")
    end
    return d1, d2
end

function _x13_trading_day_matrix(start::Tuple{Int,Int}, nobs::Int, frequency::Int)
    td = zeros(nobs, 6)
    for t in 1:nobs
        d1, d2 = _x13_period_to_daterange(start, t, frequency)
        counts = zeros(Int, 7); d = d1
        while d <= d2; counts[dayofweek(d)] += 1; d += Day(1); end
        sun = counts[7]
        for i in 1:6; td[t, i] = counts[i] - sun; end
    end
    return td
end

function _x13_easter_regressor(start::Tuple{Int,Int}, nobs::Int, window::Int, frequency::Int)
    e = zeros(nobs)
    for t in 1:nobs
        d1, d2 = _x13_period_to_daterange(start, t, frequency)
        for yr in (year(d1) - 1):(year(d2) + 1)
            edate = _x13_easter_date(yr)
            win_start = edate - Day(window); win_end = edate - Day(1)
            overlap_start = max(win_start, d1); overlap_end = min(win_end, d2)
            if overlap_start <= overlap_end
                e[t] += (Dates.value(overlap_end - overlap_start) + 1) / window
            end
        end
    end
    return e
end

function _x13_build_regressors(ts::_X13TimeSeries, spec::_X13RegressionSpec)
    nobs = length(ts.y); parts = Matrix{Float64}[]
    spec.trading_day && push!(parts, _x13_trading_day_matrix(ts.start, nobs, ts.frequency))
    spec.easter && push!(parts, reshape(_x13_easter_regressor(ts.start, nobs, spec.easter_window, ts.frequency), nobs, 1))
    length(spec.user) > 0 && push!(parts, spec.user)
    isempty(parts) && return Matrix{Float64}(undef, nobs, 0)
    return hcat(parts...)
end
