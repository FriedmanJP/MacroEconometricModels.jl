# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

const _RSER08 = ("HPFilterResult", "HamiltonFilterResult", "BeveridgeNelsonResult",
                 "BaxterKingResult", "BoostedHPResult", "X13FilterResult",
                 "ACFResult", "SpectralDensityResult", "CrossSpectrumResult",
                 "TransferFunctionResult", "FisherTestResult", "BartlettWhiteNoiseResult",
                 "KernelDensity", "KernelRegression", "LowessFit",
                 "DataSummary", "DataDiagnostic")

function _rser08_series(; n=80, seed=781)
    rng = MersenneTwister(seed)
    t = 1:n
    trend = 0.02 .* collect(t)
    cycle = 1.5 .* sin.(2π .* collect(t) ./ 16)
    return trend .+ cycle .+ 0.3 .* randn(rng, n)
end

function _rser08_seasonal(; n=96, seed=7812)
    rng = MersenneTwister(seed)
    t = 1:n
    return 100.0 .+ 0.05 .* collect(t) .+ 8.0 .* sin.(2π .* collect(t) ./ 12) .+
           0.4 .* randn(rng, n)
end

@testset "RSER-08 filter/spectral/nonparametric serialization (#781)" begin
    @testset "registry" begin
        @test length(_RSER08) == 17
        for name in _RSER08
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
        @test !any(v == "pending RSER-08" for v in values(_MEM._SERIALIZATION_EXCLUDED))
        @test !any(startswith(k, "_X13") for k in keys(_MEM._SERIALIZABLE_TYPES))
        @test !any(startswith(k, "_X13") for k in keys(_MEM._SERIALIZATION_EXCLUDED))
    end

    y = _rser08_series()

    @testset "HPFilterResult" begin
        @test _from_serializable_is_generic(HPFilterResult)
        r = hp_filter(y; lambda=1600.0)
        @test r isa HPFilterResult{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test trend(r2) == trend(r)
        @test cycle(r2) == cycle(r)
        @test r2.trend .+ r2.cycle ≈ y
    end

    @testset "HamiltonFilterResult UnitRange" begin
        @test _from_serializable_is_generic(HamiltonFilterResult)
        r = hamilton_filter(y; h=8, p=4)
        @test r isa HamiltonFilterResult{Float64}
        @test r.valid_range isa UnitRange{Int}
        payload = _MEM._to_serializable(r)
        @test payload["valid_range"] isa AbstractDict
        @test payload["valid_range"]["__unitrange__"] === true
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.valid_range isa UnitRange{Int}
        @test r2.valid_range == r.valid_range
        @test r2.trend .+ r2.cycle ≈ y[r2.valid_range]
    end

    @testset "BeveridgeNelsonResult arima_order Tuple" begin
        @test _from_serializable_is_generic(BeveridgeNelsonResult)
        r = beveridge_nelson(y; p=1, q=0)
        @test r isa BeveridgeNelsonResult{Float64}
        @test r.arima_order isa Tuple{Int,Int,Int}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.arima_order isa Tuple
        @test r2.arima_order == r.arima_order
        @test r2.permanent .+ r2.transitory ≈ y
        @test trend(r2) == r2.permanent
        @test cycle(r2) == r2.transitory
    end

    @testset "BaxterKingResult UnitRange" begin
        @test _from_serializable_is_generic(BaxterKingResult)
        r = baxter_king(y; pl=6, pu=32, K=8)
        @test r isa BaxterKingResult{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.valid_range isa UnitRange{Int}
        @test r2.valid_range == r.valid_range
        @test r2.trend .+ r2.cycle ≈ y[r2.valid_range]
    end

    @testset "BoostedHPResult" begin
        @test _from_serializable_is_generic(BoostedHPResult)
        r = boosted_hp(y; stopping=:fixed, max_iter=2)
        @test r isa BoostedHPResult{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.stopping === :fixed
        @test r2.iterations == 2
        @test r2.trend .+ r2.cycle ≈ y
    end

    @testset "X13FilterResult disk + NTuple{6,Int}" begin
        @test _from_serializable_is_generic(X13FilterResult)
        ys = _rser08_seasonal()
        r = x13_filter(ys; frequency=12, method=:x11)
        @test r isa X13FilterResult{Float64}
        @test r.arima_order isa NTuple{6,Int}
        _MEM._assert_plain_payload(_MEM._build_container(r))
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.arima_order isa NTuple{6,Int}
        @test r2.arima_order == r.arima_order
        @test r2.method === :x11
        @test seasonal(r2) == r.seasonal
        @test adjusted(r2) == r.adjusted
        seats = x13_filter(ys; frequency=12, method=:seats)
        s2 = _assert_roundtrip(seats)
        _assert_consumers(seats, s2)
        @test s2.method === :seats
        let path = joinpath(mktempdir(), "x13_filter.jld2")
            save_model(r, path)
            r3 = load_model(path)
            @test r3 isa X13FilterResult{Float64}
            @test r3.arima_order isa NTuple{6,Int}
            _assert_report_equal(r, r3)
            _assert_plot_equal(r, r3)
        end
    end

    @testset "ACFResult ccf nothing and Vector" begin
        @test _from_serializable_is_generic(ACFResult)
        r = acf_pacf(y; lags=12)
        @test r isa ACFResult{Float64}
        @test r.ccf === nothing
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.ccf === nothing
        @test r2.lags == r.lags

        x = _rser08_series(seed=7813)
        cc = ccf(x, y; lags=8)
        @test cc.ccf isa Vector{Float64}
        cc2 = _assert_roundtrip(cc)
        _assert_consumers(cc, cc2)
        @test cc2.ccf isa Vector{Float64}
        @test cc2.ccf == cc.ccf
    end

    @testset "SpectralDensityResult disk" begin
        @test _from_serializable_is_generic(SpectralDensityResult)
        r = periodogram(y)
        @test r isa SpectralDensityResult{Float64}
        @test r.method === :periodogram
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        welch = spectral_density(y; method=:welch)
        w2 = _assert_roundtrip(welch)
        _assert_consumers(welch, w2)
        @test w2.method === :welch
        let path = joinpath(mktempdir(), "spectral_density.jld2")
            save_model(r, path)
            r3 = load_model(path)
            @test r3 isa SpectralDensityResult{Float64}
            _assert_report_equal(r, r3)
            _assert_plot_equal(r, r3)
        end
    end

    @testset "CrossSpectrumResult" begin
        @test _from_serializable_is_generic(CrossSpectrumResult)
        x = _rser08_series(seed=7814)
        r = cross_spectrum(x, y)
        @test r isa CrossSpectrumResult{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test all(0 .<= r2.coherence .<= 1)
    end

    @testset "TransferFunctionResult" begin
        @test _from_serializable_is_generic(TransferFunctionResult)
        r = transfer_function(:hp; lambda=1600)
        @test r isa TransferFunctionResult{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.filter === :hp
        bk = transfer_function(:bk)
        _assert_consumers(bk, _assert_roundtrip(bk))
    end

    @testset "FisherTestResult" begin
        @test _from_serializable_is_generic(FisherTestResult)
        r = fisher_test(y)
        @test r isa FisherTestResult{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.nobs == r.nobs
        @test r2.peak_freq == r.peak_freq
    end

    @testset "BartlettWhiteNoiseResult" begin
        @test _from_serializable_is_generic(BartlettWhiteNoiseResult)
        r = bartlett_white_noise_test(y)
        @test r isa BartlettWhiteNoiseResult{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.statistic == r.statistic
        @test r2.pvalue == r.pvalue
    end

    @testset "KernelDensity" begin
        @test _from_serializable_is_generic(KernelDensity)
        xd = sin.(0.3 .* (1:60)) .+ 0.5 .* cos.(0.7 .* (1:60))
        r = kernel_density(xd; bw=:silverman)
        @test r isa KernelDensity{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.kernel === :gaussian
        @test r2.bw_method === :silverman
        @test r2.nobs == 60
    end

    @testset "KernelRegression" begin
        @test _from_serializable_is_generic(KernelRegression)
        xr = (1:50) ./ 10
        yr = sin.(xr) .+ 0.2 .* sin.(11 .* (1:50))
        r = kernel_reg(yr, xr; method=:ll, bw=:rot)
        @test r isa KernelRegression{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.method === :ll
        @test r2.bw_method === :rot
    end

    @testset "LowessFit" begin
        @test _from_serializable_is_generic(LowessFit)
        xr = (1:50) ./ 10
        yr = sin.(xr) .+ 0.2 .* sin.(11 .* (1:50))
        r = lowess(yr, xr; f=2 / 3, iter=3)
        @test r isa LowessFit{Float64}
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.iter == 3
        @test r2.nobs == 50
    end

    @testset "DataSummary Frequency enum" begin
        @test _from_serializable_is_generic(DataSummary)
        tsd = TimeSeriesData(hcat(y, _rser08_series(seed=7815));
                             varnames=["y", "x"], frequency=Quarterly)
        s = describe_data(tsd)
        @test s isa DataSummary
        @test s.frequency == Quarterly
        s2 = _assert_roundtrip(s)
        _assert_consumers(s, s2)
        @test s2.frequency == Quarterly
        @test s2.varnames == ["y", "x"]
        @test s2.n_vars == 2
    end

    @testset "DataDiagnostic" begin
        @test _from_serializable_is_generic(DataDiagnostic)
        tsd = TimeSeriesData(hcat(y, _rser08_series(seed=7816));
                             varnames=["y", "x"])
        d = diagnose(tsd)
        @test d isa DataDiagnostic
        @test d.is_clean
        d2 = _assert_roundtrip(d)
        _assert_consumers(d, d2)
        @test d2.is_clean
        @test d2.varnames == ["y", "x"]

        dirty_mat = hcat(y, _rser08_series(seed=7817))
        dirty_mat[5, 1] = NaN
        dirty_mat[10, 2] = Inf
        dirty = diagnose(TimeSeriesData(dirty_mat; varnames=["y", "x"]))
        @test !dirty.is_clean
        dirty2 = _assert_roundtrip(dirty)
        _assert_consumers(dirty, dirty2)
        @test !dirty2.is_clean
        @test dirty2.n_nan == dirty.n_nan
        @test dirty2.n_inf == dirty.n_inf
    end
end
