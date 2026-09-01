# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-02 IGARCHModel serialization" begin
    yv = randn(MersenneTwister(423), 300)
    m = estimate_igarch(yv, 1, 1)
    m2 = _assert_roundtrip(m)
    _assert_report_equal(m, m2)
    _assert_plot_equal(m, m2)
    @test coef(m2) == coef(m)
    @test stderror(m2) == stderror(m)
    f1 = forecast(m, 5; n_sim=40, rng=MersenneTwister(3))
    f2 = forecast(m2, 5; n_sim=40, rng=MersenneTwister(3))
    @test f1.forecast == f2.forecast
    @test sprint(io -> refs(io, m)) == sprint(io -> refs(io, m2))
end

@testset "RSER-04 VolatilityForecast serialization (#777)" begin
    yv = randn(MersenneTwister(423), 200)
    m = estimate_garch(yv, 1, 1)
    fc = forecast(m, 5; n_sim=40, rng=MersenneTwister(3))
    @test _from_serializable_is_generic(VolatilityForecast)
    fc2 = _assert_roundtrip(fc)
    _assert_consumers(fc, fc2)
    @test long_table(fc2) isa DataFrame
    @test fc2.model_type === fc.model_type
end
