# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-02 BVAR / FAVAR posterior serialization" begin
    @testset "MinnesotaHyperparameters / GLPHyperparameters" begin
        h = MinnesotaHyperparameters(; tau=0.3, decay=1.0, lambda=1.2, mu=0.8, omega=2.0)
        h2 = _assert_roundtrip(h)
        _assert_report_equal(h, h2)
        @test sprint(io -> refs(io, h)) == sprint(io -> refs(io, h2))

        Yg = randn(MersenneTwister(351), 80, 2)
        glp = optimize_hyperparameters_glp(Yg, 1; starts=1, max_iter=40, verbose=false)
        glp2 = _assert_roundtrip(glp)
        _assert_report_equal(glp, glp2)
        @test glp2.hyper isa MinnesotaHyperparameters{Float64}
        @test glp2.hyper.tau == glp.hyper.tau
    end

    @testset "BayesianFAVAR" begin
        rng = MersenneTwister(525)
        X = randn(rng, 60, 8)
        bf = estimate_favar(X, [1, 2], 1, 1; method=:bayesian, n_draws=12, burnin=5)
        bf2 = _assert_roundtrip(bf)
        _assert_report_equal(bf, bf2)
        _assert_plot_equal(bf, bf2)
        r1, r2 = irf(bf, 4), irf(bf2, 4)
        @test _deep_equal(r1.point_estimate, r2.point_estimate)
        @test sprint(io -> refs(io, bf)) == sprint(io -> refs(io, bf2))
    end

    @testset "MFVARPosterior" begin
        T_hf = 48
        Z = zeros(T_hf, 2)
        rng = MersenneTwister(350)
        A = [0.6 0.1; 0.15 0.5]
        for t in 2:T_hf
            Z[t, :] = A * Z[t-1, :] + 0.3 .* randn(rng, 2)
        end
        data = copy(Z)
        for t in 1:T_hf
            data[t, 2] = t % 3 == 0 ? sum(Z[max(t-2, 1):t, 2]) : NaN
        end
        post = estimate_mfvar(data, 1; low_freq=[2], aggregation=:flow, freq_ratio=3,
                              n_draws=12, n_burn=12, rng=MersenneTwister(4))
        post2 = _assert_roundtrip(post)
        _assert_report_equal(post, post2)
        r1, r2 = irf(post, 4), irf(post2, 4)
        @test _deep_equal(r1.point_estimate, r2.point_estimate)
    end

    @testset "TVPVARPosterior" begin
        Y = randn(MersenneTwister(349), 70, 2)
        post = estimate_tvpvar(Y, 1; tvp=true, sv=true,
                               n_draws=12, n_burn=12, rng=MersenneTwister(21))
        args = Any[getfield(post, i) for i in 1:nfields(post)]
        @test _MEM._infer_float_param(args) === Float64
        post2 = _assert_roundtrip(post)
        _assert_report_equal(post, post2)
        r1 = irf(post, 4; n_draws=8)
        r2 = irf(post2, 4; n_draws=8)
        @test _deep_equal(r1.point_estimate, r2.point_estimate)
        let path = joinpath(mktempdir(), "tvpvar.jld2")
            save_model(post, path)
            post3 = load_model(path)
            @test post3 isa TVPVARPosterior{Float64}
            @test sprint(show, post3) == sprint(show, post)
            @test _deep_equal(irf(post3, 4; n_draws=8).point_estimate, r1.point_estimate)
        end
    end
end

@testset "RSER-04 BVARForecast serialization (#777)" begin
    Y = randn(MersenneTwister(777), 60, 2)
    post = estimate_bvar(Y, 1; n_draws=24, seed=2)
    fc = forecast(post, 4; store_draws=true, rng=MersenneTwister(1))
    @test fc._draws isa Array{Float64,3}
    payload = _MEM._capture_fields(fc)
    @test haskey(payload, "_draws")
    @test payload["_draws"] !== nothing
    @test _from_serializable_is_generic(BVARForecast)
    fc2 = _assert_roundtrip(fc)
    _assert_consumers(fc, fc2)
    @test fc2._draws == fc._draws
    @test long_table(fc2) isa DataFrame
    let path = joinpath(mktempdir(), "bvar_fc.jld2")
        save_model(fc, path)
        fc3 = load_model(path)
        @test fc3 isa BVARForecast{Float64}
        @test fc3._draws == fc._draws
        _assert_report_equal(fc, fc3)
        _assert_plot_equal(fc, fc3)
    end
end
