# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-04 VECMForecast serialization (#777)" begin
    # Cointegrated bivariate truth (DGP-04 #793) instead of independent RWs.
    Y = dgp_vecm(MersenneTwister(777); alpha=[-0.3, 0.1], beta=[1.0, -1.0],
                 Gamma=Matrix(0.2 * I, 2, 2), T=80).Y
    vecm = estimate_vecm(Y, 2; rank=1)
    fc = forecast(vecm, 5)
    @test _from_serializable_is_generic(VECMForecast)
    fc2 = _assert_roundtrip(fc)
    _assert_consumers(fc, fc2)
    @test long_table(fc2) isa DataFrame
    @test fc2.levels == fc.levels
    @test fc2.ci_method === fc.ci_method
end
