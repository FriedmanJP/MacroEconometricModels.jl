# SEATS — Signal Extraction in ARIMA Time Series

function _x13_seats(model::_X13ARIMAModel, y::Vector{Float64}, frequency::Int,
                    spec::_X13SEATSSpec)
    n = length(y); s = frequency
    ar_poly, ma_poly = _x13_build_full_polynomials(model, s)
    trend_denom, seas_denom = _x13_build_component_denominators(model, s)
    nfreq = 2048; freqs = range(0.0, π, length=nfreq)
    ma_num = Vector{Float64}(undef, nfreq)
    for k in 1:nfreq; ma_num[k] = _x13_poly_mod2(ma_poly, freqs[k]); end
    trend_spec, seas_spec, irr_spec = _x13_partial_fractions(ma_num, trend_denom, seas_denom, model.sigma2)
    total_spec = trend_spec .+ seas_spec .+ irr_spec
    trend_wk = zeros(nfreq); seas_wk = zeros(nfreq)
    for k in 1:nfreq
        denom = total_spec[k]
        if denom > 0.0; trend_wk[k] = trend_spec[k] / denom; seas_wk[k] = seas_spec[k] / denom; end
    end
    nweights = min(3 * s, n ÷ 2 - 1, 120); nweights = max(nweights, s)
    trend_tw = _x13_freq_to_time_weights(trend_wk, freqs, nweights)
    seas_tw = _x13_freq_to_time_weights(seas_wk, freqs, nweights)
    trend_comp = _x13_apply_symmetric_filter(y, trend_tw, nweights)
    seas_comp = _x13_apply_symmetric_filter(y, seas_tw, nweights)
    irr_comp = y .- trend_comp .- seas_comp
    return _X13SEATSResult(trend_comp, seas_comp, irr_comp, trend_tw, seas_tw)
end

function _x13_build_full_polynomials(model::_X13ARIMAModel, s::Int)
    sp = model.spec
    ar_poly = zeros(sp.p + 1); ar_poly[1] = 1.0
    for i in 1:sp.p; ar_poly[i + 1] = -model.ar[i]; end
    sar_poly = if sp.P > 0
        sap = zeros(sp.P * s + 1); sap[1] = 1.0
        for i in 1:sp.P; sap[i * s + 1] = -model.sar[i]; end; sap
    else [1.0] end
    diff_poly = [1.0]
    for _ in 1:sp.d; diff_poly = _x13_poly_multiply(diff_poly, [1.0, -1.0]); end
    for _ in 1:sp.D
        sd = zeros(s + 1); sd[1] = 1.0; sd[end] = -1.0
        diff_poly = _x13_poly_multiply(diff_poly, sd)
    end
    full_ar = _x13_poly_multiply(_x13_poly_multiply(ar_poly, sar_poly), diff_poly)
    ma_poly = zeros(sp.q + 1); ma_poly[1] = 1.0
    for i in 1:sp.q; ma_poly[i + 1] = model.ma[i]; end
    sma_poly = if sp.Q > 0
        smp = zeros(sp.Q * s + 1); smp[1] = 1.0
        for i in 1:sp.Q; smp[i * s + 1] = model.sma[i]; end; smp
    else [1.0] end
    full_ma = _x13_poly_multiply(ma_poly, sma_poly)
    return full_ar, full_ma
end

function _x13_build_component_denominators(model::_X13ARIMAModel, s::Int)
    sp = model.spec
    trend_denom = [1.0]
    for _ in 1:sp.d; trend_denom = _x13_poly_multiply(trend_denom, [1.0, -1.0]); end
    if sp.p > 0
        ap = zeros(sp.p + 1); ap[1] = 1.0
        for i in 1:sp.p; ap[i + 1] = -model.ar[i]; end
        trend_denom = _x13_poly_multiply(trend_denom, ap)
    end
    seas_denom = [1.0]
    for _ in 1:sp.D
        sd = zeros(s + 1); sd[1] = 1.0; sd[end] = -1.0
        seas_denom = _x13_poly_multiply(seas_denom, sd)
    end
    if sp.P > 0
        sap = zeros(sp.P * s + 1); sap[1] = 1.0
        for i in 1:sp.P; sap[i * s + 1] = -model.sar[i]; end
        seas_denom = _x13_poly_multiply(seas_denom, sap)
    end
    return trend_denom, seas_denom
end

function _x13_freq_to_time_weights(filter_freq::Vector{Float64},
                                   freqs::AbstractRange{Float64}, nweights::Int)
    nf = length(freqs); dω = freqs[2] - freqs[1]
    weights = Vector{Float64}(undef, 2 * nweights + 1)
    for (k, tau) in enumerate(-nweights:nweights)
        w = 0.0
        for j in 1:nf
            coeff = (j == 1 || j == nf) ? 0.5 : 1.0
            w += coeff * filter_freq[j] * cos(freqs[j] * tau) * dω
        end
        weights[k] = w / π
    end
    return weights
end

function _x13_apply_symmetric_filter(y::Vector{Float64}, weights::Vector{Float64}, nweights::Int)
    n = length(y); result = Vector{Float64}(undef, n)
    for t in 1:n
        val = 0.0
        for (k, tau) in enumerate(-nweights:nweights)
            idx = t + tau
            if idx < 1; idx = clamp(2 - idx, 1, n)
            elseif idx > n; idx = clamp(2 * n - idx, 1, n); end
            val += weights[k] * y[idx]
        end
        result[t] = val
    end
    return result
end
