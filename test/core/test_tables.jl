# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using DataFrames
using Tables
using StatsAPI
using Random
using LinearAlgebra
using DelimitedFiles

@testset "Tables.jl integration (T247/#346)" begin

    # ── Coefficient-bearing models: DataFrame(result) ───────────────────────────
    @testset "DataFrame(RegModel) matches report inputs" begin
        rng = MersenneTwister(11)
        X = randn(rng, 90, 3); y = X * [1.0, -0.5, 0.3] .+ randn(rng, 90)
        m = estimate_reg(y, X)
        df = DataFrame(m)
        @test names(df) == ["term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test nrow(df) == length(coef(m))
        @test df.estimate ≈ coef(m)
        @test df.std_error ≈ stderror(m)
        @test df.stat ≈ coef(m) ./ stderror(m)
        @test all(df.ci_lower .< df.ci_upper)
        # Tables source protocol.
        @test Tables.istable(m)
        @test Tables.columnaccess(typeof(m))
        @test Set(Tables.columnnames(Tables.columns(m))) ==
              Set([:term, :estimate, :std_error, :stat, :p_value, :ci_lower, :ci_upper])
        @test Tables.schema(m) !== nothing
    end

    @testset "DataFrame(LogitModel/ProbitModel)" begin
        rng = MersenneTwister(12)
        X = randn(rng, 150, 2); z = X * [0.8, -0.6]
        y = Float64.(rand(rng, 150) .< 1 ./ (1 .+ exp.(-z)))
        lm = estimate_logit(y, X)
        dl = DataFrame(lm)
        @test dl.estimate ≈ coef(lm)
        @test dl.std_error ≈ stderror(lm)
        pm = estimate_probit(y, X)
        @test DataFrame(pm).estimate ≈ coef(pm)
    end

    @testset "DataFrame(MarginalEffects) drops non-finite rows" begin
        rng = MersenneTwister(13)
        X = randn(rng, 150, 2); z = X * [0.8, -0.6]
        y = Float64.(rand(rng, 150) .< 1 ./ (1 .+ exp.(-z)))
        me = marginal_effects(estimate_logit(y, X))
        dme = DataFrame(me)
        keep = findall(isfinite, me.effects)
        @test nrow(dme) == length(keep)
        @test dme.estimate ≈ me.effects[keep]
        @test dme.p_value ≈ me.p_values[keep]
        @test dme.ci_lower ≈ me.ci_lower[keep]
    end

    @testset "DataFrame(OrderedModel) — two blocks" begin
        rng = MersenneTwister(14)
        n = 300; X = randn(rng, n, 2); latent = X * [1.0, -0.8] .+ randn(rng, n)
        y = [v < -0.7 ? 1 : v < 0.7 ? 2 : 3 for v in latent]
        om = estimate_ologit(y, X)
        d = DataFrame(om)
        @test "block" in names(d)
        @test Set(d.block) == Set(["coef", "cutpoint"])
        @test count(==("coef"), d.block) == length(om.beta)
        @test count(==("cutpoint"), d.block) == length(om.cutpoints)
        @test d.estimate ≈ vcat(om.beta, om.cutpoints)
    end

    @testset "DataFrame(MultinomialLogitModel) — per-alternative blocks" begin
        rng = MersenneTwister(15)
        n = 400; X = randn(rng, n, 2)
        # 3-category DGP.
        u2 = X * [1.0, 0.0]; u3 = X * [0.0, 1.0]
        y = map(1:n) do i
            e = -log.(-log.(rand(rng, 3)))
            argmax([0.0 + e[1], u2[i] + e[2], u3[i] + e[3]])
        end
        ml = estimate_mlogit(y, X)
        d = DataFrame(ml)
        @test "alternative" in names(d)
        @test nrow(d) == length(ml.varnames) * size(ml.beta, 2)
        @test length(unique(d.alternative)) == size(ml.beta, 2)
        @test d.estimate ≈ vec(ml.beta)
    end

    @testset "DataFrame(VARModel) — one row per (equation, term)" begin
        rng = MersenneTwister(16)
        Y = randn(rng, 120, 2); vm = estimate_var(Y, 2)
        d = DataFrame(vm)
        @test "equation" in names(d)
        @test Set(d.equation) == Set(vm.varnames)
        @test nrow(d) == length(vm.varnames) * (1 + 2 * 2)   # (intercept + n*p) per equation
        # First equation's estimates equal B[:,1].
        d1 = d[d.equation .== vm.varnames[1], :]
        @test d1.estimate ≈ vm.B[:, 1]
    end

    # ── long_table for array-valued results ─────────────────────────────────────
    @testset "long_table(ImpulseResponse)" begin
        rng = MersenneTwister(17)
        vm = estimate_var(randn(rng, 120, 3), 2)
        ir = irf(vm, 10; method=:cholesky)
        lt = long_table(ir)
        @test names(lt) == ["horizon", "variable", "shock", "value", "lower", "upper"]
        @test nrow(lt) == 10 * 3 * 3
        @test Set(lt.horizon) == Set(1:10)
        # Bands present when ci_type != :none, else missing.
        if ir.ci_type == :none
            @test all(ismissing, lt.lower)
        end
    end

    @testset "long_table(FEVD) and forecast" begin
        rng = MersenneTwister(18)
        vm = estimate_var(randn(rng, 120, 2), 2)
        lf = long_table(fevd(vm, 8))
        @test names(lf) == ["horizon", "variable", "shock", "value"]
        @test nrow(lf) == 8 * 2 * 2
        @test all(0 .<= lf.value .<= 1 .+ 1e-8)
        lfc = long_table(forecast(vm, 6))
        @test names(lfc) == ["horizon", "variable", "value", "lower", "upper"]
        @test nrow(lfc) == 6 * 2
    end

    @testset "long_table(LPImpulseResponse) — direct construction" begin
        vals = reshape(collect(1.0:12.0), 6, 2)
        lpir = MacroEconometricModels.LPImpulseResponse{Float64}(vals, vals .- 1, vals .+ 1,
                                               fill(0.5, 6, 2), 5, ["y1", "y2"], "shock", :hac, 0.95)
        lt = long_table(lpir)
        @test names(lt) == ["horizon", "variable", "shock", "value", "se", "lower", "upper"]
        @test nrow(lt) == 6 * 2
        @test Set(lt.horizon) == Set(0:5)          # LP horizons are 0-based (impact included)
        @test all(lt.shock .== "shock")
    end

    # ── write_csv ───────────────────────────────────────────────────────────────
    @testset "write_csv round-trips through a co-author read-back" begin
        rng = MersenneTwister(19)
        X = randn(rng, 80, 2); y = X * [1.0, -0.5] .+ randn(rng, 80)
        m = estimate_reg(y, X)
        path = tempname() * ".csv"
        @test write_csv(m, path) == path
        raw, hdr = readdlm(path, ',', header=true)
        @test vec(hdr) == ["term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test size(raw, 1) == length(coef(m))
        @test Float64.(raw[:, 2]) ≈ coef(m)

        # Passing an array-valued result directly routes through long_table.
        vm = estimate_var(randn(rng, 100, 2), 2)
        ipath = tempname() * ".csv"
        write_csv(irf(vm, 5; method=:cholesky), ipath)
        iraw, ihdr = readdlm(ipath, ',', header=true)
        @test vec(ihdr) == ["horizon", "variable", "shock", "value", "lower", "upper"]
        @test size(iraw, 1) == 5 * 2 * 2
    end
end
