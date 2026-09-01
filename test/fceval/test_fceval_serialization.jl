# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

const _RSER04 = ("VARForecast", "BVARForecast", "VECMForecast", "ARIMAForecast",
                 "LPForecast", "MidasForecast", "FactorForecast", "VolatilityForecast",
                 "ThresholdForecast", "STARForecast", "MSForecast",
                 "ConditionalForecast", "ForecastCondition",
                 "ForecastEvaluation", "DMTestResult", "ClarkWestResult",
                 "MincerZarnowitzResult", "ForecastEncompassingResult",
                 "ForecastCombination")

@testset "RSER-04 forecast-evaluation serialization (#777)" begin
    @testset "registry" begin
        for name in _RSER04
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
        @test !any(v == "pending RSER-04" for v in values(_MEM._SERIALIZATION_EXCLUDED))
        @test haskey(_MEM._SERIALIZATION_EXCLUDED, "NowcastForecast")
        @test _MEM._SERIALIZATION_EXCLUDED["NowcastForecast"] == "pending RSER-05"
        @test !haskey(_MEM._SERIALIZABLE_TYPES, "NowcastForecast")
    end

    t = collect(1.0:40.0)
    actual = 2.0 .+ 0.5 .* sin.(0.2 .* t)
    e1 = sin.(0.3 .* t)
    e2 = 0.8 .* sin.(0.3 .* t .+ 0.5)
    f1 = actual .+ e1
    f2 = actual .+ e2

    @testset "ForecastEvaluation" begin
        ev = forecast_evaluate(actual, hcat(f1, f2); model_names=["m1", "m2"])
        @test _from_serializable_is_generic(ForecastEvaluation)
        args = Any[getfield(ev, i) for i in 1:nfields(ev)]
        @test _MEM._infer_float_param(args) === Float64
        ev2 = _assert_roundtrip(ev)
        _assert_consumers(ev, ev2)
        let path = joinpath(mktempdir(), "fceval.jld2")
            save_model(ev, path)
            ev3 = load_model(path)
            @test ev3 isa ForecastEvaluation{Float64}
            _assert_report_equal(ev, ev3)
            _assert_plot_equal(ev, ev3)
            @test ev3.values == ev.values
        end
    end

    @testset "DMTestResult" begin
        dm = diebold_mariano(e1, e2; h=1, loss=:se)
        @test _from_serializable_is_generic(DMTestResult)
        dm2 = _assert_roundtrip(dm)
        _assert_consumers(dm, dm2)
        @test dm2.statistic == dm.statistic
        @test dm2.hln == dm.hln
    end

    @testset "ClarkWestResult" begin
        f_adj = f1 .- f2
        cw = clark_west(e1, e2, f_adj; h=1)
        @test _from_serializable_is_generic(ClarkWestResult)
        cw2 = _assert_roundtrip(cw)
        _assert_consumers(cw, cw2)
        @test cw2.statistic == cw.statistic
    end

    @testset "MincerZarnowitzResult" begin
        mz = mincer_zarnowitz(actual, f1; lags=2, kernel=:bartlett)
        @test _from_serializable_is_generic(MincerZarnowitzResult)
        mz2 = _assert_roundtrip(mz)
        _assert_consumers(mz, mz2)
        @test mz2.a == mz.a && mz2.b == mz.b
    end

    @testset "ForecastEncompassingResult" begin
        enc = forecast_encompassing(actual, f1, f2; lags=2, kernel=:bartlett)
        @test _from_serializable_is_generic(ForecastEncompassingResult)
        enc2 = _assert_roundtrip(enc)
        _assert_consumers(enc, enc2)
        @test enc2.b1 == enc.b1 && enc2.b2 == enc.b2
    end

    @testset "ForecastCombination" begin
        comb = combine_forecasts(hcat(f1, f2), actual; method=:equal)
        @test _from_serializable_is_generic(ForecastCombination)
        comb2 = _assert_roundtrip(comb)
        _assert_consumers(comb, comb2)
        @test comb2.weights == comb.weights
        @test comb2.method === :equal
    end
end
