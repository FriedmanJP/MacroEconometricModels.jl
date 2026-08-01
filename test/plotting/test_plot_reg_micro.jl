# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# PLT-39 domain lane — base cross-sectional dispatches in reg.jl. This lane fills
# the genuinely-untested binary-choice / marginal-effect dispatches:
#   LogitModel, ProbitModel ★, MarginalEffects ★
# (RegModel diagnostics are exercised in Lane E; the panel/LDV coefficient forest
#  plots — Ordered/Multinomial/OddsRatio/Heckman/Tobit/SUR — in Lane D.)
# ★ = added by PLT-39 (no prior plot_result smoke test in the suite).
#
# Uses shared plot_test_helpers.jl assertions (Testing Rules 1-7).

using Test, Random

isdefined(@__MODULE__, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

@testset "Plotting — Reg micro (Logit / Probit / MarginalEffects)" begin
    Random.seed!(45)
    X = randn(200, 2)
    xb = X * [0.8, -0.5]
    yb = Float64.((xb .+ 0.3 .* randn(200)) .> 0)

    @testset "LogitModel" begin
        m = estimate_logit(yb, X)
        p = plot_result(m)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Logit", p.html)
        @test length(panel_titles(p.html)) == 2         # sorted probs + distribution
        # red stays reserved for reference elements, not an outcome series (PLT-13).
        @test !occursin("\"name\":\"y = 0\",\"color\":\"#d62728\"", p.html)
        # hostile figure title escapes at the HTML sink (A8).
        @test !occursin("<c>", plot_result(m; title=HOSTILE_NAME).html)
    end

    @testset "ProbitModel ★" begin
        m = estimate_probit(yb, X)
        p = plot_result(m)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Probit", p.html)
        @test length(panel_titles(p.html)) == 2
        # save_path returns the PlotOutput (C8).
        tmp = tempname() * ".html"
        p2 = plot_result(m; save_path=tmp)
        @test p2 isa PlotOutput && isfile(tmp)
        rm(tmp; force=true)
    end

    @testset "MarginalEffects (Logit) ★" begin
        me = marginal_effects(estimate_logit(yb, X; varnames=["x1", "x2"]))
        @test me isa MarginalEffects{Float64}
        p = plot_result(me)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Marginal Effects", p.html)
        @test occursin("ci_lo", p.html)                 # dot-and-whisker rows
        # the intercept row (non-finite marginal effect) is dropped, not serialized.
        for (nm, lit) in extract_json_blocks(p.html)
            nm == "data" || continue
            @test !occursin(r"[:\[,]\s*NaN", lit)
        end
    end

    @testset "MarginalEffects (Probit) + escaping ★" begin
        me = marginal_effects(estimate_probit(yb, X; varnames=[HOSTILE_NAME, "x2"]))
        p = plot_result(me)
        check_plot(p); assert_all_json_valid(p)
        assert_escapes(p)                                # hostile regressor name (A7/A8)
    end
end
