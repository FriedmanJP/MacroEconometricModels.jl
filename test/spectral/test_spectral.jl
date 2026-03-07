# MacroEconometricModels.jl — Spectral Analysis & ACF/PACF Tests
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, Random, Statistics, LinearAlgebra, StatsAPI, DataFrames

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

# =============================================================================
# Coverage expansion: input validation, edge cases, untested code paths
# =============================================================================

@testset "Input validation — ArgumentError" begin
    short2 = [1.0, 2.0]
    short3 = [1.0, 2.0, 3.0]

    # acf/pacf/acf_pacf require n >= 3
    @test_throws ArgumentError acf(short2)
    @test_throws ArgumentError pacf(short2)
    @test_throws ArgumentError acf_pacf(short2)

    # ccf requires n >= 3 and equal length
    @test_throws ArgumentError ccf(short2, short2)
    @test_throws DimensionMismatch ccf([1.0, 2.0, 3.0], [1.0, 2.0])

    # periodogram requires n >= 4
    @test_throws ArgumentError periodogram(short3)

    # spectral_density requires n >= 4
    @test_throws ArgumentError spectral_density(short3; method=:welch)
    @test_throws ArgumentError spectral_density(short3; method=:smoothed)
    @test_throws ArgumentError spectral_density(short3; method=:ar)

    # cross_spectrum requires n >= 4 and equal length
    @test_throws ArgumentError cross_spectrum(short3, short3)
    @test_throws DimensionMismatch cross_spectrum([1.0,2.0,3.0,4.0], [1.0,2.0,3.0])

    # fisher_test / bartlett require n >= 4
    @test_throws ArgumentError fisher_test(short3)
    @test_throws ArgumentError bartlett_white_noise_test(short3)

    # ideal_bandpass requires n >= 4
    @test_throws ArgumentError ideal_bandpass(short3, 0.5, 1.5)

    # ideal_bandpass invalid frequency bounds
    @test_throws ArgumentError ideal_bandpass(randn(100), 1.5, 0.5)  # f_low >= f_high
    @test_throws ArgumentError ideal_bandpass(randn(100), -0.1, 1.0) # f_low < 0
    @test_throws ArgumentError ideal_bandpass(randn(100), 0.5, 4.0)  # f_high > π

    # Invalid method for pacf
    @test_throws ArgumentError pacf(randn(100); method=:invalid)
    @test_throws ArgumentError acf_pacf(randn(100); method=:invalid)

    # Invalid method for spectral_density
    @test_throws ArgumentError spectral_density(randn(100); method=:invalid)

    # Invalid filter for transfer_function
    @test_throws ArgumentError transfer_function(:bogus)

    # band_power: f_low >= f_high
    rng = Random.MersenneTwister(99001)
    r = periodogram(randn(rng, 256))
    @test_throws ArgumentError band_power(r, 1.5, 0.5)
end

@testset "Degenerate data — constant series" begin
    # Constant series → zero variance → _zero_acf_result
    const_series = fill(5.0, 200)

    r_acf = acf(const_series; lags=10)
    @test all(r_acf.acf .== 0)
    @test all(r_acf.q_stats .== 0)
    @test r_acf.ci > 0

    r_pacf = pacf(const_series; lags=10, method=:levinson)
    @test all(r_pacf.pacf .== 0)

    r_ap = acf_pacf(const_series; lags=10)
    @test all(r_ap.acf .== 0)
    @test all(r_ap.pacf .== 0)

    # CCF with one constant series
    rng = Random.MersenneTwister(20001)
    r_ccf = ccf(const_series, randn(rng, 200))
    @test all(r_ccf.ccf .== 0)
end

@testset "Convenience extractors: coherence, phase, gain" begin
    rng = Random.MersenneTwister(21001)
    n = 256
    x = randn(rng, n)
    y = 2.0 .* x .+ 0.3 .* randn(rng, n)

    freq_c, coh_vals = coherence(x, y)
    @test length(freq_c) == length(coh_vals)
    @test all(0 .<= coh_vals .<= 1)

    freq_p, phase_vals = phase(x, y)
    @test length(freq_p) == length(phase_vals)

    freq_g, gain_vals = gain(x, y)
    @test length(freq_g) == length(gain_vals)
    @test all(gain_vals .>= 0)
end

@testset "StatsAPI interface — nobs, pvalue" begin
    rng = Random.MersenneTwister(22001)
    y = randn(rng, 200)

    ft = fisher_test(y)
    @test StatsAPI.nobs(ft) == 200
    @test 0 <= StatsAPI.pvalue(ft) <= 1

    bt = bartlett_white_noise_test(y)
    @test StatsAPI.nobs(bt) == 200
    @test 0 <= StatsAPI.pvalue(bt) <= 1
end

@testset "Custom conf_level" begin
    rng = Random.MersenneTwister(23001)
    y = randn(rng, 300)

    r90 = acf(y; lags=10, conf_level=0.90)
    r99 = acf(y; lags=10, conf_level=0.99)
    @test r90.ci < r99.ci  # 99% CI should be wider than 90%
    @test r90.acf == r99.acf  # ACF values unchanged

    r90p = pacf(y; lags=10, conf_level=0.90)
    r99p = pacf(y; lags=10, conf_level=0.99)
    @test r90p.ci < r99p.ci
end

@testset "Welch — segment options" begin
    rng = Random.MersenneTwister(24001)
    y = randn(rng, 512)

    # Custom segment_length
    r1 = spectral_density(y; method=:welch, segment_length=64)
    @test r1.method == :welch
    @test all(r1.density .>= 0)

    # segment_length = n (single segment, like periodogram)
    r2 = spectral_density(y; method=:welch, segment_length=512, overlap=0.0)
    @test r2.method == :welch
    @test length(r2.density) == div(512, 2) + 1

    # Custom overlap
    r3 = spectral_density(y; method=:welch, overlap=0.75)
    @test r3.method == :welch
    @test all(r3.density .>= 0)

    # Very small overlap
    r4 = spectral_density(y; method=:welch, overlap=0.0)
    @test r4.method == :welch
end

@testset "Smoothed — custom bandwidth" begin
    rng = Random.MersenneTwister(25001)
    y = randn(rng, 256)

    r1 = spectral_density(y; method=:smoothed, bandwidth=3)
    r2 = spectral_density(y; method=:smoothed, bandwidth=20)
    @test r1.method == :smoothed
    @test r2.method == :smoothed
    # Larger bandwidth should produce smoother (lower peak) spectrum
    @test maximum(r2.density) <= maximum(r1.density) * 1.5
end

@testset "AR spectrum — custom order and AIC selection" begin
    rng = Random.MersenneTwister(26001)
    n = 500
    y = zeros(n)
    y[1] = randn(rng)
    for t in 2:n
        y[t] = 0.7 * y[t-1] + randn(rng)
    end

    # Explicit AR order
    r_ar1 = spectral_density(y; method=:ar, order=1)
    @test r_ar1.method == :ar
    @test all(r_ar1.density .>= 0)

    # Higher AR order
    r_ar5 = spectral_density(y; method=:ar, order=5)
    @test r_ar5.method == :ar

    # AIC-selected (default order=0)
    r_aic = spectral_density(y; method=:ar)
    @test r_aic.method == :ar
    @test all(r_aic.density .>= 0)

    # Custom n_freq
    r_nf = spectral_density(y; method=:ar, n_freq=128)
    @test length(r_nf.density) == 128
end

@testset "Periodogram with windows" begin
    rng = Random.MersenneTwister(27001)
    y = randn(rng, 256)

    for win in [:rectangular, :bartlett, :hann, :hamming, :blackman, :tukey, :flat_top]
        r = periodogram(y; window=win)
        @test r.method == :periodogram
        @test all(r.density .>= 0)
    end
end

@testset "Cross-spectrum with custom segment options" begin
    rng = Random.MersenneTwister(28001)
    n = 512
    x = randn(rng, n)
    y = randn(rng, n)

    # Custom segment_length
    cs1 = cross_spectrum(x, y; segment_length=64)
    @test length(cs1.coherence) == div(64, 2) + 1

    # Custom window
    cs2 = cross_spectrum(x, y; window=:hamming)
    @test all(0 .<= cs2.coherence .<= 1)

    # Non-Float64 fallback (same type)
    x32 = Float32.(x)
    y32 = Float32.(y)
    cs3 = cross_spectrum(x32, y32)
    @test cs3 isa MacroEconometricModels.CrossSpectrumResult

    # Mixed Float64 + Float32 → promotes to Float64
    cs4 = cross_spectrum(x, y32)
    @test cs4 isa MacroEconometricModels.CrossSpectrumResult{Float64}

    cs5 = cross_spectrum(x32, y)
    @test cs5 isa MacroEconometricModels.CrossSpectrumResult{Float64}

    # Integer fallback (both Int → Float64)
    x_int = round.(Int, x .* 10)
    y_int = round.(Int, y .* 10)
    cs6 = cross_spectrum(x_int, y_int)
    @test cs6 isa MacroEconometricModels.CrossSpectrumResult

    # Float + Integer
    cs7 = cross_spectrum(x, y_int)
    @test cs7 isa MacroEconometricModels.CrossSpectrumResult{Float64}

    cs8 = cross_spectrum(x_int, y)
    @test cs8 isa MacroEconometricModels.CrossSpectrumResult{Float64}
end

@testset "Transfer function — parameter variations" begin
    # HP with different lambda
    tf1 = transfer_function(:hp; lambda=6.25)  # annual
    tf2 = transfer_function(:hp; lambda=1600)  # quarterly
    tf3 = transfer_function(:hp; lambda=129600) # monthly
    # All should have gain near 1 at Nyquist (ω=π)
    @test tf1.gain[end] > 0.9
    @test tf2.gain[end] > 0.9
    @test tf3.gain[end] > 0.9

    # BK with different K
    tf_bk4 = transfer_function(:bk; K=4)
    tf_bk24 = transfer_function(:bk; K=24)
    @test length(tf_bk4.gain) == 256
    @test length(tf_bk24.gain) == 256

    # Hamilton with different h
    tf_h4 = transfer_function(:hamilton; h=4)
    tf_h16 = transfer_function(:hamilton; h=16)
    @test length(tf_h4.gain) == 256
    @test length(tf_h16.gain) == 256

    # Custom n_freq
    tf_nf = transfer_function(:hp; n_freq=64)
    @test length(tf_nf.gain) == 64
    @test length(tf_nf.freq) == 64
end

@testset "Window edge cases" begin
    # n=1
    w1 = MacroEconometricModels._spectral_window(1, :hann)
    @test w1 == [1.0]

    # n=2
    w2 = MacroEconometricModels._spectral_window(2, :hann)
    @test length(w2) == 2

    # n < 1
    @test_throws ArgumentError MacroEconometricModels._spectral_window(0, :hann)
end

@testset "Levinson-Durbin early break" begin
    # Series that triggers denominator < 1e-15 in Levinson-Durbin
    # A unit root series with near-perfect ACF[1] ≈ 1
    rng = Random.MersenneTwister(29001)
    n = 100
    rw = cumsum(randn(rng, n))
    # Still should return valid result without error
    r = pacf(rw; lags=10)
    @test length(r.pacf) == 10
    @test all(isfinite.(r.pacf))
end

@testset "OLS PACF with short series / high lags" begin
    rng = Random.MersenneTwister(30001)
    # Request more lags than feasible for OLS
    y = randn(rng, 20)
    r = pacf(y; lags=15, method=:ols)
    @test length(r.pacf) == 15
    @test all(isfinite.(r.pacf))
    # High lags should be zero (neff < k+1 break)
    @test r.pacf[end] == 0.0
end

@testset "Fisher test — edge cases" begin
    rng = Random.MersenneTwister(31001)
    # Very short series (n=4, m=1)
    y4 = randn(rng, 4)
    ft4 = fisher_test(y4)
    @test ft4 isa MacroEconometricModels.FisherTestResult
    @test ft4.nobs == 4

    # n=5 (m=2)
    y5 = randn(rng, 5)
    ft5 = fisher_test(y5)
    @test ft5.nobs == 5
    @test ft5.statistic > 0
end

@testset "Bartlett test — edge cases" begin
    rng = Random.MersenneTwister(32001)
    # Minimum valid n=4
    y4 = randn(rng, 4)
    bt4 = bartlett_white_noise_test(y4)
    @test bt4 isa MacroEconometricModels.BartlettWhiteNoiseResult
    @test bt4.nobs == 4
end

@testset "band_power — edge cases" begin
    rng = Random.MersenneTwister(33001)
    r = periodogram(randn(rng, 256))

    # Very narrow band
    bp_narrow = band_power(r, 1.0, 1.01)
    @test bp_narrow >= 0

    # Band at boundaries
    bp_low = band_power(r, 0.0, 0.5)
    bp_high = band_power(r, 2.5, Float64(π))
    @test bp_low >= 0
    @test bp_high >= 0
end

@testset "ideal_bandpass — full band and edge cases" begin
    rng = Random.MersenneTwister(34001)
    n = 128
    y = randn(rng, n)

    # Near-full band should preserve most energy
    y_full = ideal_bandpass(y, 0.01, Float64(π) - 0.01)
    energy_orig = sum((y .- mean(y)) .^ 2)
    energy_filt = sum(y_full .^ 2)
    @test energy_filt > 0.5 * energy_orig

    # Very narrow band should kill most energy
    y_narrow = ideal_bandpass(y, 1.0, 1.05)
    energy_narrow = sum(y_narrow .^ 2)
    @test energy_narrow < 0.5 * energy_orig
end

@testset "Non-Float64 fallbacks for estimation" begin
    rng = Random.MersenneTwister(35001)
    y_int = round.(Int, randn(rng, 200) .* 10)

    # Integer → Float64 via fallback
    r_pg = periodogram(y_int)
    @test r_pg isa MacroEconometricModels.SpectralDensityResult{Float64}

    r_sd = spectral_density(y_int; method=:welch)
    @test r_sd isa MacroEconometricModels.SpectralDensityResult{Float64}

    r_ar = spectral_density(y_int; method=:ar)
    @test r_ar isa MacroEconometricModels.SpectralDensityResult{Float64}

    r_sm = spectral_density(y_int; method=:smoothed)
    @test r_sm isa MacroEconometricModels.SpectralDensityResult{Float64}

    # Integer fisher/bartlett
    ft = fisher_test(y_int)
    @test ft isa MacroEconometricModels.FisherTestResult{Float64}

    bt = bartlett_white_noise_test(y_int)
    @test bt isa MacroEconometricModels.BartlettWhiteNoiseResult{Float64}

    # Integer ideal_bandpass
    y_filt = ideal_bandpass(y_int, 0.5, 1.5)
    @test eltype(y_filt) == Float64

    # Integer ccf
    x_int = round.(Int, randn(rng, 200) .* 10)
    r_ccf = ccf(x_int, y_int)
    @test r_ccf isa MacroEconometricModels.ACFResult{Float64}
end

@testset "PanelData dispatch — acf, spectral_density" begin
    rng = Random.MersenneTwister(36001)
    # Create simple PanelData with 3 groups, 50 time periods each
    n_groups = 3
    n_time = 50
    df = DataFrames.DataFrame(
        group = repeat(1:n_groups, inner=n_time),
        time = repeat(1:n_time, outer=n_groups),
        y = randn(rng, n_groups * n_time),
        x = randn(rng, n_groups * n_time),
    )
    pd = xtset(df, :group, :time)

    # acf dispatch → Dict{group, ACFResult}
    result_acf = acf(pd, 10; var=:y)
    @test result_acf isa Dict
    @test length(result_acf) == n_groups
    for (gname, r) in result_acf
        @test r isa MacroEconometricModels.ACFResult
        @test length(r.acf) == 10
    end

    # spectral_density dispatch → Dict{group, SpectralDensityResult}
    result_sd = spectral_density(pd; var=:y, method=:welch)
    @test result_sd isa Dict
    @test length(result_sd) == n_groups
    for (gname, r) in result_sd
        @test r isa MacroEconometricModels.SpectralDensityResult
        @test all(r.density .>= 0)
    end
end

@testset "Default lag selection" begin
    rng = Random.MersenneTwister(37001)
    # lags=0 → auto-select min(n-1, 10*log10(n))
    y50 = randn(rng, 50)
    r50 = acf(y50)  # default lags
    expected_maxlag = min(49, round(Int, 10 * log10(50)))
    @test length(r50.acf) == expected_maxlag

    y1000 = randn(rng, 1000)
    r1000 = acf(y1000)
    expected_maxlag2 = min(999, round(Int, 10 * log10(1000)))
    @test length(r1000.acf) == expected_maxlag2
end

@testset "Burg coefficient edge case" begin
    rng = Random.MersenneTwister(38001)
    y = randn(rng, 50)
    # AR order close to n should throw
    @test_throws ArgumentError MacroEconometricModels._burg_coefficients(y, 50)

    # High but valid order
    a, sigma2 = MacroEconometricModels._burg_coefficients(y, 15)
    @test length(a) == 15
    @test isfinite(sigma2)
end

@testset "report() for spectral types" begin
    rng = Random.MersenneTwister(39001)
    y = randn(rng, 200)

    # report() should work for all spectral result types
    r_acf = acf_pacf(y; lags=10)
    io = IOBuffer()
    show(io, r_acf)
    s = String(take!(io))
    @test occursin("Correlogram", s)

    # ACF-only → different title
    r_acf_only = acf(y; lags=10)
    show(io, r_acf_only)
    s2 = String(take!(io))
    @test occursin("Autocorrelation Function", s2)

    # PACF-only → different title
    r_pacf_only = pacf(y; lags=10)
    show(io, r_pacf_only)
    s3 = String(take!(io))
    @test occursin("Partial Autocorrelation Function", s3)

    # CCF display
    rng2 = Random.MersenneTwister(39002)
    x = randn(rng2, 200)
    r_ccf = ccf(x, y; lags=10)
    show(io, r_ccf)
    s4 = String(take!(io))
    @test occursin("Cross-Correlation", s4)

    # Spectral density
    r_sd = periodogram(y)
    show(io, r_sd)
    s5 = String(take!(io))
    @test occursin("Spectral Density", s5)

    # Cross-spectrum
    cs = cross_spectrum(x, y)
    show(io, cs)
    s6 = String(take!(io))
    @test occursin("Cross-Spectral", s6)

    # Transfer function
    tf = transfer_function(:hp)
    show(io, tf)
    s7 = String(take!(io))
    @test occursin("Transfer Function", s7)

    # Fisher
    ft = fisher_test(y)
    show(io, ft)
    s8 = String(take!(io))
    @test occursin("Fisher", s8)

    # Bartlett
    bt = bartlett_white_noise_test(y)
    show(io, bt)
    s9 = String(take!(io))
    @test occursin("Bartlett", s9)

    # Ljung-Box
    lb = ljung_box_test(y; lags=5)
    show(io, lb)
    s10 = String(take!(io))
    @test occursin("Ljung-Box", s10)

    # Box-Pierce
    bp = box_pierce_test(y; lags=5)
    show(io, bp)
    s11 = String(take!(io))
    @test occursin("Box-Pierce", s11)

    # Durbin-Watson
    dw = durbin_watson_test(y)
    show(io, dw)
    s12 = String(take!(io))
    @test occursin("Durbin-Watson", s12)
end

@testset "DW show branches" begin
    # Positive autocorrelation (DW < 1.5)
    rng = Random.MersenneTwister(40001)
    n = 300
    ar_pos = zeros(n)
    ar_pos[1] = randn(rng)
    for t in 2:n; ar_pos[t] = 0.95 * ar_pos[t-1] + randn(rng); end
    dw_pos = durbin_watson_test(ar_pos)
    io = IOBuffer()
    show(io, dw_pos)
    s = String(take!(io))
    @test occursin("Positive autocorrelation", s) || occursin("No strong evidence", s)

    # Negative autocorrelation (DW > 2.5)
    ar_neg = zeros(n)
    ar_neg[1] = randn(rng)
    for t in 2:n; ar_neg[t] = -0.5 * ar_neg[t-1] + randn(rng); end
    dw_neg = durbin_watson_test(ar_neg)
    show(io, dw_neg)
    s2 = String(take!(io))
    @test occursin("Negative autocorrelation", s2) || occursin("No strong evidence", s2)
end

@testset "CCF type fallbacks" begin
    rng = Random.MersenneTwister(41001)
    x64 = randn(rng, 100)
    y_int = round.(Int, randn(rng, 100) .* 10)

    # Integer + Integer → Float64 via ccf(x::<:Real, y::<:Real)
    x_int = round.(Int, randn(rng, 100) .* 10)
    r1 = ccf(x_int, y_int)
    @test r1 isa MacroEconometricModels.ACFResult{Float64}

    # Float32 + Float32 (same type, dispatches to primary method)
    x32 = Float32.(randn(rng, 100))
    y32 = Float32.(randn(rng, 100))
    r2 = ccf(x32, y32)
    @test r2 isa MacroEconometricModels.ACFResult{Float32}

    # Float64 + Float32 → promotes to Float64
    r3 = ccf(x64, y32)
    @test r3 isa MacroEconometricModels.ACFResult{Float64}

    # Float32 + Float64 → promotes to Float64
    r4 = ccf(y32, x64)
    @test r4 isa MacroEconometricModels.ACFResult{Float64}

    # Float64 + Integer
    r5 = ccf(x64, y_int)
    @test r5 isa MacroEconometricModels.ACFResult{Float64}

    # Integer + Float64
    r6 = ccf(y_int, x64)
    @test r6 isa MacroEconometricModels.ACFResult{Float64}
end
