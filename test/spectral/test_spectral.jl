# MacroEconometricModels.jl — Spectral Analysis & ACF/PACF Tests
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, Random, Statistics, LinearAlgebra

@testset "ACF" begin
    rng = Random.MersenneTwister(1001)
    # White noise: ACF near zero at all lags
    wn = randn(rng, 500)
    r = acf(wn; lags=20)
    @test r isa MacroEconometricModels.ACFResult
    @test length(r.acf) == 20
    @test all(abs.(r.acf) .< 0.15)
    @test r.nobs == 500
    @test r.ci > 0

    # AR(1) with rho=0.8: ACF(1) should be close to 0.8
    rng2 = Random.MersenneTwister(1002)
    n = 1000
    rho = 0.8
    ar1 = zeros(n)
    ar1[1] = randn(rng2)
    for t in 2:n
        ar1[t] = rho * ar1[t-1] + randn(rng2)
    end
    r2 = acf(ar1; lags=10)
    @test abs(r2.acf[1] - rho) < 0.1
    # ACF should decay geometrically
    @test abs(r2.acf[2]) < abs(r2.acf[1])
end

@testset "PACF" begin
    # AR(2): PACF should cut off after lag 2
    rng = Random.MersenneTwister(2001)
    n = 2000
    ar2 = zeros(n)
    ar2[1] = randn(rng)
    ar2[2] = randn(rng)
    for t in 3:n
        ar2[t] = 0.5 * ar2[t-1] - 0.3 * ar2[t-2] + randn(rng)
    end

    # Levinson method
    r_lev = pacf(ar2; lags=10, method=:levinson)
    @test r_lev isa MacroEconometricModels.ACFResult
    @test length(r_lev.pacf) == 10
    @test abs(r_lev.pacf[1]) > 0.2  # significant at lag 1
    @test abs(r_lev.pacf[2]) > 0.1  # significant at lag 2
    # Lags 4+ should be near zero for AR(2)
    @test all(abs.(r_lev.pacf[4:end]) .< 0.15)

    # OLS method
    r_ols = pacf(ar2; lags=10, method=:ols)
    @test length(r_ols.pacf) == 10

    # Levinson vs OLS should broadly agree
    @test all(abs.(r_lev.pacf[1:5] .- r_ols.pacf[1:5]) .< 0.15)
end

@testset "acf_pacf combined" begin
    rng = Random.MersenneTwister(3001)
    y = randn(rng, 300)
    # Inject mild AR(1) structure
    for t in 2:300
        y[t] += 0.4 * y[t-1]
    end
    r = acf_pacf(y; lags=15)
    @test r isa MacroEconometricModels.ACFResult
    @test length(r.acf) == 15
    @test length(r.pacf) == 15
    # Q-stats should be populated
    @test length(r.q_stats) == 15
    @test length(r.q_pvalues) == 15
    @test all(r.q_stats .>= 0)
    @test all(0 .<= r.q_pvalues .<= 1)
end

@testset "CCF" begin
    rng = Random.MersenneTwister(4001)
    n = 500
    x = randn(rng, n)
    y = randn(rng, n)
    # y partly driven by x with lag 1
    for t in 2:n
        y[t] += 0.6 * x[t-1]
    end
    r = ccf(x, y; lags=10)
    @test r isa MacroEconometricModels.ACFResult
    @test r.ccf !== nothing
    @test length(r.ccf) == 21  # -10:10
    @test length(r.lags) == 21
    @test r.lags[1] == -10
    @test r.lags[end] == 10
    # CCF at lag -1 (y_t correlates with x_{t-1}) should be positive and significant
    # lag -1 corresponds to index (-1)+10+1 = 10
    @test r.ccf[10] > 0.15
end

@testset "Periodogram" begin
    # Sinusoid + noise: periodogram should spike at the sinusoid frequency
    rng = Random.MersenneTwister(5001)
    n = 256
    freq_true = 0.1  # cycles per sample
    t_grid = collect(1:n)
    y = 3.0 .* sin.(2pi .* freq_true .* t_grid) .+ 0.5 .* randn(rng, n)
    r = periodogram(y)
    @test r isa MacroEconometricModels.SpectralDensityResult
    @test r.method == :periodogram
    @test r.nobs == n
    @test length(r.freq) == div(n, 2) + 1
    @test length(r.density) == length(r.freq)
    @test length(r.ci_lower) == length(r.freq)
    @test length(r.ci_upper) == length(r.freq)
    # All densities should be non-negative
    @test all(r.density .>= 0)
    # Peak should be near the planted frequency
    # Planted freq in radians = 2pi * 0.1 = 0.6283
    peak_idx = argmax(r.density)
    @test abs(r.freq[peak_idx] - 2pi * freq_true) < 0.15
end

@testset "Welch spectral density" begin
    rng = Random.MersenneTwister(6001)
    y = randn(rng, 512)
    r = spectral_density(y; method=:welch)
    @test r isa MacroEconometricModels.SpectralDensityResult
    @test r.method == :welch
    @test all(r.density .>= 0)
    # CIs should bracket the density
    @test all(r.ci_lower .<= r.density)
    @test all(r.ci_upper .>= r.density)
end

@testset "AR spectral density" begin
    # AR(1) spectrum should peak at omega=0
    rng = Random.MersenneTwister(6002)
    n = 2000
    ar1 = zeros(n)
    ar1[1] = randn(rng)
    for t in 2:n
        ar1[t] = 0.9 * ar1[t-1] + randn(rng)
    end
    r = spectral_density(ar1; method=:ar, order=1)
    @test r.method == :ar
    @test all(r.density .>= 0)
    # Density should be highest at low frequencies for positive AR(1)
    # First 10% of frequencies should have higher average than last 10%
    nf = length(r.density)
    low_freq_mean = mean(r.density[1:max(1, nf÷10)])
    high_freq_mean = mean(r.density[9*nf÷10:end])
    @test low_freq_mean > high_freq_mean
end

@testset "Smoothed spectral density" begin
    rng = Random.MersenneTwister(6003)
    y = randn(rng, 256)
    r = spectral_density(y; method=:smoothed)
    @test r.method == :smoothed
    @test all(r.density .>= 0)
    @test length(r.density) > 0
end

@testset "Cross-spectrum" begin
    rng = Random.MersenneTwister(7001)
    n = 512
    x = randn(rng, n)
    # y = alpha * x + noise  (high coherence expected)
    alpha = 2.5
    y = alpha .* x .+ 0.3 .* randn(rng, n)

    cs = cross_spectrum(x, y)
    @test cs isa MacroEconometricModels.CrossSpectrumResult
    @test cs.nobs == n
    @test length(cs.freq) > 0
    @test length(cs.co_spectrum) == length(cs.freq)
    @test length(cs.quad_spectrum) == length(cs.freq)
    @test length(cs.coherence) == length(cs.freq)
    @test length(cs.phase) == length(cs.freq)
    @test length(cs.gain) == length(cs.freq)

    # Coherence should be high (y is mostly a scaled copy of x)
    mean_coh = mean(cs.coherence)
    @test mean_coh > 0.5

    # All coherence values in [0,1]
    @test all(0 .<= cs.coherence .<= 1)
end

@testset "Windows" begin
    n = 64
    for wtype in [:rectangular, :bartlett, :hann, :hanning, :hamming, :blackman, :tukey, :flat_top]
        w = MacroEconometricModels._spectral_window(n, wtype)
        @test length(w) == n
        @test all(isfinite.(w))
        # Symmetric (up to floating point)
        @test all(abs.(w .- reverse(w)) .< 1e-12)
    end
    # Unknown window should error
    @test_throws ArgumentError MacroEconometricModels._spectral_window(10, :bogus)
end

@testset "Ljung-Box test" begin
    # White noise: high p-value (fail to reject H0: no autocorrelation)
    rng = Random.MersenneTwister(8001)
    wn = randn(rng, 500)
    lb = ljung_box_test(wn; lags=10)
    @test lb isa MacroEconometricModels.LjungBoxResult
    @test lb.pvalue > 0.01  # should generally not reject

    # AR(1): low p-value (reject H0)
    rng2 = Random.MersenneTwister(8002)
    ar1 = zeros(500)
    ar1[1] = randn(rng2)
    for t in 2:500
        ar1[t] = 0.8 * ar1[t-1] + randn(rng2)
    end
    lb2 = ljung_box_test(ar1; lags=10)
    @test lb2.pvalue < 0.05
end

@testset "Box-Pierce test" begin
    rng = Random.MersenneTwister(8003)
    wn = randn(rng, 500)
    bp = box_pierce_test(wn; lags=10)
    @test bp isa MacroEconometricModels.BoxPierceResult
    @test bp.pvalue > 0.01
end

@testset "Durbin-Watson test" begin
    rng = Random.MersenneTwister(8004)
    # iid residuals: DW near 2
    resid = randn(rng, 300)
    dw = durbin_watson_test(resid)
    @test dw isa MacroEconometricModels.DurbinWatsonResult
    @test abs(dw.statistic - 2.0) < 0.5
    @test dw.pvalue > 0.01
end

@testset "Fisher's test" begin
    # Planted sinusoid: should detect periodicity (low p-value)
    rng = Random.MersenneTwister(9001)
    n = 200
    t_grid = collect(1:n)
    y_sin = 5.0 .* sin.(2pi .* 0.1 .* t_grid) .+ 0.5 .* randn(rng, n)
    ft = fisher_test(y_sin)
    @test ft isa MacroEconometricModels.FisherTestResult
    @test ft.pvalue < 0.05  # detect the sinusoid
    @test ft.statistic > 0
    @test ft.peak_freq > 0

    # White noise: should not detect periodicity (high p-value, usually)
    rng2 = Random.MersenneTwister(9002)
    wn = randn(rng2, 200)
    ft2 = fisher_test(wn)
    @test ft2.pvalue > 0.01
end

@testset "Bartlett's white-noise test" begin
    # White noise: should pass (high p-value)
    rng = Random.MersenneTwister(9003)
    wn = randn(rng, 300)
    bt = bartlett_white_noise_test(wn)
    @test bt isa MacroEconometricModels.BartlettWhiteNoiseResult
    @test bt.pvalue > 0.01

    # AR(1) with strong autocorrelation: should fail (low p-value)
    rng2 = Random.MersenneTwister(9004)
    ar1 = zeros(300)
    ar1[1] = randn(rng2)
    for t in 2:300
        ar1[t] = 0.9 * ar1[t-1] + randn(rng2)
    end
    bt2 = bartlett_white_noise_test(ar1)
    @test bt2.pvalue < 0.05
end

@testset "band_power" begin
    rng = Random.MersenneTwister(10001)
    y = randn(rng, 256)
    r = periodogram(y)
    # Band power over full range
    bp_full = band_power(r, 0.0, Float64(pi))
    @test bp_full > 0
    # Band power over partial range should be less
    bp_partial = band_power(r, 0.5, 1.5)
    @test 0 < bp_partial < bp_full
end

@testset "ideal_bandpass" begin
    rng = Random.MersenneTwister(11001)
    n = 256
    t_grid = collect(1:n)
    # Two sinusoids at different frequencies
    f_in = 0.1   # cycles/sample -> 2pi*0.1 radians
    f_out = 0.4  # cycles/sample -> 2pi*0.4 radians
    y = 3.0 .* sin.(2pi .* f_in .* t_grid) .+ 3.0 .* sin.(2pi .* f_out .* t_grid) .+ 0.1 .* randn(rng, n)

    # Bandpass to keep only f_in (roughly 0.5 to 0.8 radians)
    y_filt = ideal_bandpass(y, 0.4, 0.9)
    @test length(y_filt) == n
    @test all(isfinite.(y_filt))

    # Filtered signal should retain the in-band component
    # and kill the out-of-band component
    # Check via energy: the filtered signal should have roughly half the energy
    energy_orig = sum(y .^ 2)
    energy_filt = sum(y_filt .^ 2)
    @test energy_filt < energy_orig
    @test energy_filt > 0
end

@testset "transfer_function" begin
    # HP filter
    tf_hp = transfer_function(:hp; lambda=1600)
    @test tf_hp isa MacroEconometricModels.TransferFunctionResult
    @test tf_hp.filter == :hp
    @test length(tf_hp.freq) == 256
    @test length(tf_hp.gain) == 256
    # HP gain at omega=0 should be near 0 (cycle component vanishes at DC)
    @test tf_hp.gain[1] < 0.01

    # BK filter
    tf_bk = transfer_function(:bk)
    @test tf_bk.filter == :bk
    @test length(tf_bk.gain) == 256

    # Hamilton filter
    tf_ham = transfer_function(:hamilton; h=8)
    @test tf_ham.filter == :hamilton
    @test length(tf_ham.gain) == 256
end

@testset "Display (show methods)" begin
    rng = Random.MersenneTwister(12001)
    y = randn(rng, 200)

    # ACFResult
    r_acf = acf(y; lags=10)
    io = IOBuffer()
    show(io, r_acf)
    @test length(String(take!(io))) > 0

    # SpectralDensityResult
    r_sd = periodogram(y)
    show(io, r_sd)
    @test length(String(take!(io))) > 0

    # CrossSpectrumResult
    rng2 = Random.MersenneTwister(12002)
    x = randn(rng2, 200)
    cs = cross_spectrum(x, y)
    show(io, cs)
    @test length(String(take!(io))) > 0

    # TransferFunctionResult
    tf = transfer_function(:hp)
    show(io, tf)
    @test length(String(take!(io))) > 0

    # FisherTestResult
    ft = fisher_test(y)
    show(io, ft)
    @test length(String(take!(io))) > 0

    # BartlettWhiteNoiseResult
    bt = bartlett_white_noise_test(y)
    show(io, bt)
    @test length(String(take!(io))) > 0

    # LjungBoxResult
    lb = ljung_box_test(y; lags=5)
    show(io, lb)
    @test length(String(take!(io))) > 0

    # BoxPierceResult
    bp = box_pierce_test(y; lags=5)
    show(io, bp)
    @test length(String(take!(io))) > 0

    # DurbinWatsonResult
    dw = durbin_watson_test(y)
    show(io, dw)
    @test length(String(take!(io))) > 0
end

@testset "Plotting (plot_result)" begin
    rng = Random.MersenneTwister(13001)
    y = randn(rng, 200)

    # ACFResult
    r_acf = acf_pacf(y; lags=10)
    p = plot_result(r_acf)
    @test p isa MacroEconometricModels.PlotOutput

    # CCF plot
    rng2 = Random.MersenneTwister(13002)
    x = randn(rng2, 200)
    r_ccf = ccf(x, y; lags=10)
    p2 = plot_result(r_ccf)
    @test p2 isa MacroEconometricModels.PlotOutput

    # SpectralDensityResult
    r_sd = periodogram(y)
    p3 = plot_result(r_sd)
    @test p3 isa MacroEconometricModels.PlotOutput

    # CrossSpectrumResult
    cs = cross_spectrum(x, y)
    p4 = plot_result(cs)
    @test p4 isa MacroEconometricModels.PlotOutput

    # TransferFunctionResult
    tf = transfer_function(:hp)
    p5 = plot_result(tf)
    @test p5 isa MacroEconometricModels.PlotOutput
end

@testset "TimeSeriesData dispatch" begin
    rng = Random.MersenneTwister(14001)
    y_vec = randn(rng, 200)
    td = TimeSeriesData(y_vec)

    # ACF dispatch should match raw vector results
    r_vec = acf(y_vec; lags=10)
    r_td = acf(td, 10)
    @test r_vec.acf == r_td.acf
    @test r_vec.nobs == r_td.nobs

    # PACF dispatch
    r_vec_p = pacf(y_vec; lags=10)
    r_td_p = pacf(td, 10)
    @test r_vec_p.pacf == r_td_p.pacf

    # acf_pacf dispatch
    r_vec_ap = acf_pacf(y_vec; lags=10)
    r_td_ap = acf_pacf(td, 10)
    @test r_vec_ap.acf == r_td_ap.acf
    @test r_vec_ap.pacf == r_td_ap.pacf

    # spectral_density dispatch
    r_vec_sd = spectral_density(y_vec; method=:welch)
    r_td_sd = spectral_density(td; method=:welch)
    @test r_vec_sd.density == r_td_sd.density

    # periodogram dispatch
    r_vec_pg = periodogram(y_vec)
    r_td_pg = periodogram(td)
    @test r_vec_pg.density == r_td_pg.density
end

@testset "Float32 fallback" begin
    rng = Random.MersenneTwister(15001)
    y32 = Float32.(randn(rng, 200))

    # Float32 <: AbstractFloat, so primary methods accept it directly
    # producing Float32-parameterized results
    r_acf = acf(y32; lags=10)
    @test r_acf isa MacroEconometricModels.ACFResult{Float32}

    r_pacf = pacf(y32; lags=10)
    @test r_pacf isa MacroEconometricModels.ACFResult{Float32}

    r_ap = acf_pacf(y32; lags=10)
    @test r_ap isa MacroEconometricModels.ACFResult{Float32}

    r_pg = periodogram(y32)
    @test r_pg isa MacroEconometricModels.SpectralDensityResult{Float32}

    r_sd = spectral_density(y32; method=:welch)
    @test r_sd isa MacroEconometricModels.SpectralDensityResult{Float32}

    r_ft = fisher_test(y32)
    @test r_ft isa MacroEconometricModels.FisherTestResult{Float32}

    r_bt = bartlett_white_noise_test(y32)
    @test r_bt isa MacroEconometricModels.BartlettWhiteNoiseResult{Float32}

    r_lb = ljung_box_test(y32; lags=5)
    @test r_lb isa MacroEconometricModels.LjungBoxResult{Float32}

    r_bp = box_pierce_test(y32; lags=5)
    @test r_bp isa MacroEconometricModels.BoxPierceResult{Float32}

    r_dw = durbin_watson_test(y32)
    @test r_dw isa MacroEconometricModels.DurbinWatsonResult{Float32}

    # Integer vector should upcast to Float64 via the <:Real fallback
    y_int = round.(Int, randn(rng, 200) .* 10)
    r_int = acf(y_int; lags=5)
    @test r_int isa MacroEconometricModels.ACFResult{Float64}
end

@testset "FFTW fix (estimate_gdfm)" begin
    # estimate_gdfm uses FFTW internally; should work without explicit `using FFTW`
    rng = Random.MersenneTwister(16001)
    X = randn(rng, 100, 5)
    gdfm = estimate_gdfm(X, 2)
    @test gdfm !== nothing
end
