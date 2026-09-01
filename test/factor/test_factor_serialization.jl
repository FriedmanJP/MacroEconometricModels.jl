# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-04 FactorForecast serialization (#777)" begin
    X = randn(MersenneTwister(77701), 60, 8)
    fm = estimate_factors(X, 2)
    fc = forecast(fm, 4)
    @test _from_serializable_is_generic(FactorForecast)
    fc2 = _assert_roundtrip(fc)
    _assert_consumers(fc, fc2)
    @test long_table(fc2) isa DataFrame
    @test fc2.ci_method === fc.ci_method
    @test fc2.observables == fc.observables
end
