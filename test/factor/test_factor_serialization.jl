# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-04 FactorForecast serialization (#777)" begin
    X = dgp_dynamic_factors(MersenneTwister(77701); N=8, T=60).X
    fm = estimate_factors(X, 2)
    fc = forecast(fm, 4)
    @test _from_serializable_is_generic(FactorForecast)
    fc2 = _assert_roundtrip(fc)
    _assert_consumers(fc, fc2)
    @test long_table(fc2) isa DataFrame
    @test fc2.ci_method === fc.ci_method
    @test fc2.observables == fc.observables
end

@testset "RSER-14 factor IC leftovers (#787)" begin
    @testset "registry" begin
        for name in ("HallinLiskaResult", "BaiNgQResult", "AmengualWatsonResult")
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
    end

    @testset "HallinLiskaResult" begin
        @test _from_serializable_is_generic(HallinLiskaResult)
        r = HallinLiskaResult(1, [1, 1, 2], [0.0, 1.0, 2.0], [0.1, 0.0, 0.2],
                              (0.5, 1.5), 0.01, [1 1 2; 1 2 2], :p1)
        r2 = _assert_roundtrip(r)
        @test r2.q == 1
        @test r2.stability_interval == (0.5, 1.5)
        @test r2.penalty === :p1
        _assert_report_equal(r, r2)
        let path = joinpath(mktempdir(), "hl.jld2")
            save_model(r, path)
            @test load_model(path).q == 1
        end
    end

    @testset "BaiNgQResult" begin
        @test _from_serializable_is_generic(BaiNgQResult)
        r = BaiNgQResult(1, 2, [0.5, 0.3], [0.4, 0.2], 0.1, 3)
        r2 = _assert_roundtrip(r)
        @test r2.q_D1 == 1 && r2.q_D2 == 2 && r2.r == 3
        _assert_report_equal(r, r2)
    end

    @testset "AmengualWatsonResult" begin
        @test _from_serializable_is_generic(AmengualWatsonResult)
        r = AmengualWatsonResult(1, 2, 2, 3, 1)
        r2 = _assert_roundtrip(r)
        @test r2.q == 1 && r2.r_IC2 == 2 && r2.p == 1
        _assert_report_equal(r, r2)
    end
end
