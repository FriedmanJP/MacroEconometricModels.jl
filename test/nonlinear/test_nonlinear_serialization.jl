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
        _assert_plot_equal(m, m2)
        @test m2.sel_pvalues isa NTuple{3,Float64}
        @test coef(m2) == coef(m)
        f1 = forecast(m, 4; reps=30, rng=MersenneTwister(9))
        f2 = forecast(m2, 4; reps=30, rng=MersenneTwister(9))
        @test f1.forecast == f2.forecast
        @test sprint(io -> refs(io, m)) == sprint(io -> refs(io, m2))
        @test _from_serializable_is_generic(STARForecast)
        f1b = _assert_roundtrip(f1)
        _assert_consumers(f1, f1b)
        @test f1b.reps == f1.reps
    end

    @testset "MSRegModel" begin
        @test !_from_serializable_is_generic(MSRegModel)
        y = randn(MersenneTwister(415), 90)
        m = estimate_ms_ar(y, 1; k_regimes=2)
        m2 = _assert_roundtrip(m)
        _assert_report_equal(m, m2)
        _assert_plot_equal(m, m2)
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
        @test _from_serializable_is_generic(MSForecast)
        f1b = _assert_roundtrip(f1)
        _assert_consumers(f1, f1b)
        @test f1b.regime_prob == f1.regime_prob
    end
end

@testset "RSER-04 ThresholdForecast serialization (#777)" begin
    rng = MersenneTwister(5)
    y = zeros(120)
    for t in 2:120
        y[t] = (y[t-1] <= 0 ? 0.3 : 0.7) * y[t-1] + 0.4 * randn(rng)
    end
    m = estimate_setar(y, 1, 1; linearity=false)
    fc = forecast(m, 4; reps=20, rng=MersenneTwister(6))
    @test _from_serializable_is_generic(ThresholdForecast)
    fc2 = _assert_roundtrip(fc)
    _assert_consumers(fc, fc2)
    @test fc2.reps == fc.reps
    @test long_table(fc2) isa DataFrame
end
