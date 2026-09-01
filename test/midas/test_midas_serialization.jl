# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-04 MidasForecast serialization (#777)" begin
    rng = MersenneTwister(409)
    m, K, T_lf = 3, 4, 40
    x = randn(rng, m * T_lf)
    y = randn(rng, T_lf)
    model = estimate_midas(y, x; m=m, K=K, weights=:umidas, p_ar=0)
    fc = forecast(model, randn(rng, K))
    @test _from_serializable_is_generic(MidasForecast)
    fc2 = _assert_roundtrip(fc)
    _assert_consumers(fc, fc2)
    @test long_table(fc2) isa DataFrame
    @test fc2.horizon == fc.horizon
end
