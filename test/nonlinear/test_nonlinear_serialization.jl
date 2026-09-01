# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-02 nonlinear serialization" begin
    @testset "HansenLinearityTest" begin
        n = 80
        y = randn(MersenneTwister(5), n)
        X = hcat(ones(n - 1), y[1:n-1])
        ht = hansen_linearity_test(y[2:end], X, y[1:n-1]; reps=20,
                                   rng=MersenneTwister(6))
        ht2 = _assert_roundtrip(ht)
        _assert_report_equal(ht, ht2)
        @test sprint(io -> refs(io, ht)) == sprint(io -> refs(io, ht2))
    end

    @testset "STARModel" begin
        y = randn(MersenneTwister(414), 120)
        m = estimate_star(y, 1; d=1, type=:auto, n_gamma=6, n_c=6)
        @test m.sel_pvalues isa NTuple{3,Float64}
        m2 = _assert_roundtrip(m)
        _assert_report_equal(m, m2)
        @test plot_result(m2) isa PlotOutput
        @test m2.sel_pvalues isa NTuple{3,Float64}
        @test coef(m2) == coef(m)
        f1 = forecast(m, 4; reps=30, rng=MersenneTwister(9))
        f2 = forecast(m2, 4; reps=30, rng=MersenneTwister(9))
        @test f1.forecast == f2.forecast
        @test sprint(io -> refs(io, m)) == sprint(io -> refs(io, m2))
    end

    @testset "MSRegModel" begin
        @test !_from_serializable_is_generic(MSRegModel)
        y = randn(MersenneTwister(415), 90)
        m = estimate_ms_ar(y, 1; k_regimes=2)
        m2 = _assert_roundtrip(m)
        _assert_report_equal(m, m2)
        @test plot_result(m2) isa PlotOutput
        @test coef(m2) == coef(m)
        @test fitted(m2) == fitted(m)
        f1 = forecast(m, 4; reps=30, rng=MersenneTwister(10))
        f2 = forecast(m2, 4; reps=30, rng=MersenneTwister(10))
        @test f1.forecast == f2.forecast
        @test sprint(io -> refs(io, m)) == sprint(io -> refs(io, m2))
        let path = joinpath(mktempdir(), "msreg.jld2")
            save_model(m, path)
            m3 = load_model(path)
            @test m3 isa MSRegModel{Float64}
            @test sprint(show, m3) == sprint(show, m)
            @test forecast(m3, 4; reps=30, rng=MersenneTwister(10)).forecast == f1.forecast
        end
    end
end
