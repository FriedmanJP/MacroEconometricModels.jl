# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using JLD2                # exercises the weak-dependency disk backend
using Random
using LinearAlgebra

const _MEM = MacroEconometricModels

# Pure in-memory round-trip through the versioned container (no disk / no backend).
_roundtrip(m) = _MEM._reconstruct_from_container(_MEM._build_container(m))

@testset "Versioned serialization (T248/#347)" begin

    @testset "container round-trip reconstructs public fields exactly — all types" begin
        Y = randn(MersenneTwister(1), 120, 2)

        model = estimate_var(Y, 2)
        v2 = _roundtrip(model)
        @test v2 isa VARModel
        @test v2.Y == model.Y && v2.B == model.B && v2.U == model.U
        @test v2.Sigma == model.Sigma && v2.p == model.p
        @test v2.aic == model.aic && v2.bic == model.bic && v2.hqic == model.hqic
        @test v2.varnames == model.varnames

        post = estimate_bvar(Y, 2; n_draws=50, seed=7)
        b2 = _roundtrip(post)
        @test b2 isa BVARPosterior
        @test b2.B_draws == post.B_draws && b2.Sigma_draws == post.Sigma_draws
        @test b2.n_draws == post.n_draws && b2.data == post.data
        @test b2.prior == post.prior && b2.sampler == post.sampler
        @test b2.manifest isa ReproManifest && b2.manifest.seed == 7

        X = hcat(ones(100), randn(MersenneTwister(2), 100, 2))
        yv = X * [1.0, 0.5, -0.3] .+ 0.1 .* randn(MersenneTwister(3), 100)
        reg = estimate_reg(yv, X)
        r2 = _roundtrip(reg)
        @test r2 isa RegModel
        @test r2.beta == reg.beta && r2.vcov_mat == reg.vcov_mat
        @test r2.residuals == reg.residuals && r2.r2 == reg.r2
        @test r2.method == reg.method && r2.cov_type == reg.cov_type
        @test r2.weights === reg.weights    # nothing survives as nothing

        yb = Float64.((X * [0.0, 1.5, -1.5] .+ 0.3 .* randn(MersenneTwister(4), 100)) .> 0)
        logit = estimate_logit(yb, X)
        l2 = _roundtrip(logit)
        @test l2 isa LogitModel
        @test l2.beta == logit.beta && l2.vcov_mat == logit.vcov_mat
        @test l2.converged == logit.converged && l2.iterations == logit.iterations

        probit = estimate_probit(yb, X)
        pr2 = _roundtrip(probit)
        @test pr2 isa ProbitModel
        @test pr2.beta == probit.beta && pr2.loglik == probit.loglik

        lp = estimate_lp(Y, 1, 6)
        lp2 = _roundtrip(lp)
        @test lp2 isa LPModel
        @test lp2.B == lp.B && lp2.residuals == lp.residuals && lp2.vcov == lp.vcov
        @test lp2.horizon == lp.horizon && lp2.lags == lp.lags
        @test lp2.cov_estimator isa typeof(lp.cov_estimator)
    end

    @testset "save_model / load_model disk round-trip via JLD2" begin
        Y = randn(MersenneTwister(5), 120, 3)

        model = estimate_var(Y, 2)
        path = joinpath(mktempdir(), "var.jld2")
        @test save_model(model, path) == path
        @test isfile(path)
        m2 = load_model(path)
        @test m2 isa VARModel
        @test m2.Y == model.Y && m2.B == model.B && m2.Sigma == model.Sigma
        @test m2.aic == model.aic && m2.varnames == model.varnames

        post = estimate_bvar(Y, 2; n_draws=40, seed=99)
        pp = joinpath(mktempdir(), "bvar.jld2")
        save_model(post, pp)
        p2 = load_model(pp)
        @test p2.B_draws == post.B_draws
        @test p2.manifest isa ReproManifest && p2.manifest.seed == 99   # manifest persisted

        # A reloaded BVAR still reproduces from its persisted seed
        @test reproduce(p2).matched === true
    end

    @testset "container metadata header" begin
        Y = randn(MersenneTwister(6), 80, 2)
        c = _MEM._build_container(estimate_var(Y, 2))
        @test c["format_version"] == SERIALIZATION_FORMAT_VERSION
        @test c["type"] == "VARModel"
        @test !isempty(c["package_version"])
        @test !isempty(c["julia_version"])
        @test haskey(c, "payload") && c["payload"] isa AbstractDict
    end

    @testset "top-level manifest travels with the container" begin
        Y = randn(MersenneTwister(7), 80, 2)
        post = estimate_bvar(Y, 2; n_draws=30, seed=11)
        c = _MEM._build_container(post)
        @test c["manifest"] isa AbstractDict
        @test c["manifest"]["seed"] == 11
        # a deterministic result has no manifest
        cv = _MEM._build_container(estimate_var(Y, 2))
        @test cv["manifest"] === nothing
    end

    @testset "unknown format_version and type raise a typed, informative error" begin
        Y = randn(MersenneTwister(8), 80, 2)
        c = _MEM._build_container(estimate_var(Y, 2))

        bad_ver = copy(c); bad_ver["format_version"] = 999
        err = try
            _MEM._reconstruct_from_container(bad_ver); nothing
        catch e
            e
        end
        @test err isa SerializationError
        @test occursin("999", err.msg)
        @test occursin(string(SERIALIZATION_FORMAT_VERSION), err.msg)

        bad_type = copy(c); bad_type["type"] = "NopeModel"
        @test_throws SerializationError _MEM._reconstruct_from_container(bad_type)

        no_ver = copy(c); delete!(no_ver, "format_version")
        @test_throws SerializationError _MEM._reconstruct_from_container(no_ver)
    end

    @testset "unsupported save target and missing file raise SerializationError" begin
        @test_throws SerializationError _MEM._build_container(3.14)
        @test_throws SerializationError load_model(joinpath(mktempdir(), "does_not_exist.jld2"))
    end
end
