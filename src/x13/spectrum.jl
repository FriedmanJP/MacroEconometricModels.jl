# Spectral density and autocovariance for ARMA models (X-13 internal)

function _x13_spectral_density(ar::Vector{Float64}, ma::Vector{Float64},
                               freqs, sigma2::Float64)
    nf = length(freqs)
    sd = Vector{Float64}(undef, nf)
    for k in 1:nf
        ω = freqs[k]
        ma_re = 1.0; ma_im = 0.0
        for j in eachindex(ma); ma_re += ma[j] * cos(j * ω); ma_im -= ma[j] * sin(j * ω); end
        num = ma_re * ma_re + ma_im * ma_im
        ar_re = 1.0; ar_im = 0.0
        for j in eachindex(ar); ar_re -= ar[j] * cos(j * ω); ar_im += ar[j] * sin(j * ω); end
        den = ar_re * ar_re + ar_im * ar_im
        sd[k] = sigma2 / (2π) * num / den
    end
    return sd
end
