# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-03 LP innovation-accounting serialization (#776)" begin
    Y = randn(MersenneTwister(776), 80, 2)

    @testset "LPImpulseResponse" begin
        lp = estimate_lp(Y, 1, 6; lags=1)
        lir = lp_irf(lp)
        lir2 = _assert_roundtrip(lir)
        _assert_consumers(lir, lir2)
        @test lir2.shock_var == lir.shock_var
        @test lir2.cov_type === lir.cov_type
    end

    @testset "StructuralLP nested VARModel / LPModel / ImpulseResponse" begin
        slp = structural_lp(Y, 6; method=:cholesky, lags=1, var_lags=2)
        slp2 = _assert_roundtrip(slp)
        _assert_consumers(slp, slp2)
        @test slp2.irf isa ImpulseResponse{Float64}
        @test slp2.var_model isa VARModel{Float64}
        @test slp2.lp_models isa Vector{<:LPModel}
        @test length(slp2.lp_models) == 2
        @test slp2.method === :cholesky
        @test slp2.Q == slp.Q
        @test irf(slp2.var_model, 4).values == irf(slp.var_model, 4).values
        @test coef(slp2.lp_models[1]) == coef(slp.lp_models[1])
    end

    @testset "LPFEVD" begin
        slp = structural_lp(Y, 6; method=:cholesky, lags=1, var_lags=2)
        lf = lp_fevd(slp, 6; method=:r2, n_boot=0)
        lf2 = _assert_roundtrip(lf)
        _assert_consumers(lf, lf2)
        @test lf2.method === :r2
        @test lf2.n_boot == 0
        @test lf2.bias_correction == lf.bias_correction
    end
end
