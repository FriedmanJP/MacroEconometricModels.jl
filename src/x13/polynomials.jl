# Polynomial operations for ARMA algebra and SEATS decomposition

function _x13_poly_multiply(a::Vector{Float64}, b::Vector{Float64})
    na, nb = length(a), length(b)
    c = zeros(na + nb - 1)
    for i in 1:na, j in 1:nb; c[i+j-1] += a[i] * b[j]; end
    return c
end

function _x13_poly_mod2(coeffs::Vector{Float64}, ω::Float64)
    re = 0.0; im = 0.0
    for (j, c) in enumerate(coeffs)
        k = j - 1; re += c * cos(k * ω); im -= c * sin(k * ω)
    end
    return re * re + im * im
end

function _x13_partial_fractions(num_spectrum::Vector{Float64},
                                denom_trend::Vector{Float64},
                                denom_seasonal::Vector{Float64},
                                sigma2::Float64)
    nf = length(num_spectrum)
    freqs = range(0.0, π, length=nf)
    trend_spec = zeros(nf); seasonal_spec = zeros(nf); irregular_spec = zeros(nf)
    for k in 1:nf
        ω = freqs[k]
        dt2 = _x13_poly_mod2(denom_trend, ω)
        ds2 = _x13_poly_mod2(denom_seasonal, ω)
        inv_t = dt2 == 0.0 ? 0.0 : 1.0 / dt2
        inv_s = ds2 == 0.0 ? 0.0 : 1.0 / ds2
        inv_irr = 1.0
        total_inv = inv_t + inv_s + inv_irr
        f = sigma2 / (2π) * num_spectrum[k]
        trend_spec[k] = f * inv_t / total_inv
        seasonal_spec[k] = f * inv_s / total_inv
        irregular_spec[k] = f * inv_irr / total_inv
    end
    return (trend_spec, seasonal_spec, irregular_spec)
end
