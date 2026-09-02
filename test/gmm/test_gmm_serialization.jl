# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

const _RSER11_GMM = ("GMMWeighting", "ParameterTransform")

@testset "RSER-11 GMM leftovers serialization (#784)" begin
    @testset "registry" begin
        for name in _RSER11_GMM
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
    end

    @testset "GMMWeighting" begin
        @test _from_serializable_is_generic(GMMWeighting)
        gw = GMMWeighting()
        @test gw isa GMMWeighting{Float64}
        gw2 = _assert_roundtrip(gw)
        _assert_consumers(gw, gw2)
        @test gw2.method === :two_step
        @test gw2.max_iter == 100
        @test gw2.tol == 1e-8

        gw_it = GMMWeighting(method=:iterated, max_iter=50, tol=1e-6)
        gw_it2 = _assert_roundtrip(gw_it)
        @test gw_it2.method === :iterated
        @test gw_it2.max_iter == 50
        @test gw_it2.tol == 1e-6
    end

    @testset "GMMWeighting nested in GMMModel" begin
        rng = MersenneTwister(784)
        n = 80
        X = randn(rng, n, 2)
        beta_true = [1.0, -0.5]
        y = X * beta_true + randn(rng, n)
        data = hcat(y, X)
        moment_fn(theta, d) = begin
            resid = d[:, 1] - d[:, 2:3] * theta
            d[:, 2:3] .* resid
        end
        m = estimate_gmm(moment_fn, [0.0, 0.0], data; weighting=:identity)
        @test m.weighting isa GMMWeighting{Float64}
        w2 = _assert_roundtrip(m.weighting)
        @test w2.method === :identity
        m2 = _assert_roundtrip(m)
        _assert_consumers(m, m2)
        @test m2.weighting isa GMMWeighting{Float64}
        @test m2.weighting.method === m.weighting.method
        @test m2.theta == m.theta
    end

    @testset "ParameterTransform including Inf bounds" begin
        @test _from_serializable_is_generic(ParameterTransform)
        pt = ParameterTransform([0.0, -Inf, -1.0], [1.0, Inf, Inf])
        @test pt isa ParameterTransform{Float64}
        pt2 = _assert_roundtrip(pt)
        _assert_consumers(pt, pt2)
        @test pt2.lower == pt.lower
        @test pt2.upper == pt.upper
        @test isinf(pt2.lower[2]) && pt2.lower[2] < 0
        @test isinf(pt2.upper[2]) && pt2.upper[2] > 0
        @test to_constrained(pt2, to_unconstrained(pt, [0.5, 2.0, 0.0])) ≈ [0.5, 2.0, 0.0]
    end

    @testset "disk round-trip ParameterTransform / GMMWeighting" begin
        pt = ParameterTransform([-Inf], [0.0])
        gw = GMMWeighting(method=:optimal, max_iter=10, tol=1e-4)
        dir = mktempdir()
        save_model(pt, joinpath(dir, "pt.jld2"))
        save_model(gw, joinpath(dir, "gw.jld2"))
        pt2 = load_model(joinpath(dir, "pt.jld2"))
        gw2 = load_model(joinpath(dir, "gw.jld2"))
        @test pt2 isa ParameterTransform{Float64}
        @test isinf(pt2.lower[1]) && pt2.upper[1] == 0.0
        @test gw2.method === :optimal
    end
end
