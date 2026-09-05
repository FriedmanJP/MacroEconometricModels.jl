# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# FAST-friendly Codecov gap tests: exception show, tolerances, wild bootstrap
# helpers, GARCH show, panel-probit validation.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics
using DataFrames

const M = MacroEconometricModels

@testset "Codecov gaps" begin

    @testset "default_abstol / default_reltol" begin
        @test M.default_abstol() == 1e-8
        @test M.default_abstol(Float64) == 1e-8
        @test M.default_abstol(1.0) == 1e-8
        @test M.default_abstol(Float32) == sqrt(eps(Float32))
        @test M.default_reltol() == sqrt(eps(Float64))
        @test M.default_reltol(0.5) == sqrt(eps(Float64))
        @test M.default_reltol(Float32) == sqrt(eps(Float32))
    end

    @testset "typed exception showerror" begin
        ce = M.ConvergenceError("nope"; iters=3, residual=1.2)
        ie = M.IdentificationError("unid")
        se = M.SingularSystemError("sing"; cond=1e16)
        xe = M.SerializationError("bad file")
        @test sprint(showerror, ce) == "ConvergenceError: nope (iters=3) (residual=1.2)"
        @test sprint(showerror, ie) == "IdentificationError: unid"
        @test occursin("cond≈", sprint(showerror, se))
        @test sprint(showerror, xe) == "SerializationError: bad file"
        @test M._is_recoverable_draw_error(ce)
        @test M._is_recoverable_draw_error(ie)
        @test M._is_recoverable_draw_error(se)
        @test M._is_recoverable_draw_error(LinearAlgebra.SingularException(1))
        @test M._is_recoverable_draw_error(DomainError(0.0, "x"))
        @test !M._is_recoverable_draw_error(MethodError(sin, (1,)))
    end

    @testset "wild bootstrap helpers" begin
        @test M._default_block_length(1) == 1
        @test M._default_block_length(1000) == ceil(Int, cbrt(1000))
        rng = MersenneTwister(7)
        w_r = M._wild_weights(rng, 20, :rademacher, Float64)
        @test length(w_r) == 20
        @test all(abs.(w_r) .≈ 1)
        w_m = M._wild_weights(MersenneTwister(8), 50, :mammen, Float64)
        @test length(w_m) == 50
        @test isapprox(mean(w_m), 0.0; atol=0.5)
        @test_throws ArgumentError M._wild_weights(rng, 4, :bogus, Float64)
        U = randn(MersenneTwister(9), 30, 2)
        Uw = M._resample_residuals(U, :wild, MersenneTwister(9))
        Ub = M._resample_residuals(U, :block, MersenneTwister(9); block_length=5)
        @test size(Uw) == size(U) && size(Ub) == size(U)
    end

    @testset "GARCHModel show / StatsAPI" begin
        rng = MersenneTwister(11)
        y = randn(rng, 200)
        m = estimate_garch(y, 1, 1)
        @test m isa M.GARCHModel
        s = sprint(show, m)
        @test occursin("GARCH", s)
        @test length(m.fitted) == length(y)
        @test length(m.residuals) == length(y)
    end

    @testset "xtprobit validation" begin
        df = DataFrame(id=repeat(1:10, inner=4), t=repeat(1:4, 10),
                       x=randn(Random.MersenneTwister(1429), 40),
                       y=Float64.(rand(Random.MersenneTwister(1430), 40) .< 0.5))
        pd = xtset(df, :id, :t)
        @test_throws ArgumentError estimate_xtprobit(pd, :y, [:x]; model=:fe)
        @test_throws ArgumentError estimate_xtprobit(pd, :missing, [:x])
        @test_throws ArgumentError estimate_xtprobit(pd, :y, [:nope])
        df.y[1] = 2.0
        pd2 = xtset(df, :id, :t)
        @test_throws ArgumentError estimate_xtprobit(pd2, :y, [:x])
    end

end
