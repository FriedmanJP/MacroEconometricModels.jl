# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# PLT-39 domain lane — models.jl core dispatches + the genuinely-untested
# dispatches this rebuild fills:
#   Volatility: ARCH / GARCH / EGARCH / GJR-GARCH / SV  (+ FIGARCH / FIEGARCH ★)
#   Factor:     FactorModel / DynamicFactorModel        (+ FAVAR / BayesianFAVAR ★ / StructuralDFM ★)
#   Data:       TimeSeriesData / PanelData
#   Infra:      _json helpers, _resolve_var, _make_plot, PlotOutput, save_plot
# ★ = added by PLT-39 (no prior plot_result smoke test in the suite).
#
# Uses shared plot_test_helpers.jl assertions (Testing Rules 1-7).

using Test, Random, DataFrames

isdefined(@__MODULE__, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

const _MEM_MODELS = MacroEconometricModels

@testset "Plotting — Models / Factor / Data / Infra" begin
    Random.seed!(44)

    # =========================================================================
    # Volatility models
    # =========================================================================
    @testset "ARCHModel" begin
        p = plot_result(estimate_arch(randn(300), 2))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("ARCH", p.html)
    end

    @testset "GARCHModel" begin
        p = plot_result(estimate_garch(randn(500), 1, 1))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("GARCH", p.html)
    end

    @testset "EGARCHModel" begin
        p = plot_result(estimate_egarch(randn(300), 1, 1))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("EGARCH", p.html)
    end

    @testset "GJRGARCHModel" begin
        p = plot_result(estimate_gjr_garch(randn(300), 1, 1))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("GJR-GARCH", p.html)
    end

    @testset "SVModel" begin
        p = plot_result(estimate_sv(randn(200); n_samples=100, burnin=50))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Stochastic Volatility", p.html)
    end

    # ── Genuine gaps filled by PLT-39 ────────────────────────────────────────
    @testset "FIGARCHModel ★" begin
        m = estimate_figarch(randn(300); truncation=100)
        p = plot_result(m)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("FIGARCH", p.html)
        @test_throws ArgumentError plot_result(m; view=:bogus)
    end

    @testset "FIEGARCHModel ★" begin
        m = estimate_fiegarch(randn(300); truncation=100)
        p = plot_result(m)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("FIEGARCH", p.html)
        @test_throws ArgumentError plot_result(m; view=:bogus)
    end

    # =========================================================================
    # Factor models
    # =========================================================================
    @testset "FactorModel" begin
        p = plot_result(estimate_factors(randn(200, 20), 3))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Factor", p.html)
    end

    @testset "DynamicFactorModel" begin
        p = plot_result(estimate_dynamic_factors(randn(200, 20), 2, 1))
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Dynamic", p.html)
    end

    # ── Genuine gaps filled by PLT-39 ────────────────────────────────────────
    @testset "FAVARModel ★" begin
        X = randn(120, 12)
        m = estimate_favar(X, [1, 5], 2, 2)
        p = plot_result(m)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("FAVAR", p.html)
        @test occursin("Factor 1", p.html)
        # save_path returns the PlotOutput and writes a file (C8).
        tmp = tempname() * ".html"
        p2 = plot_result(m; save_path=tmp)
        @test p2 isa PlotOutput && isfile(tmp)
        rm(tmp; force=true)
        @test occursin("Custom Title", plot_result(m; title="Custom Title").html)
    end

    @testset "BayesianFAVAR ★" begin
        X = randn(100, 10)
        bf = estimate_favar(X, [1, 5], 2, 1; method=:bayesian, n_draws=50, burnin=20)
        @test bf isa BayesianFAVAR{Float64}
        p = plot_result(bf)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Bayesian FAVAR", p.html)
        @test occursin("68% CI", p.html)             # posterior credible band label
        @test occursin("\"lo_key\"", p.html)         # band drawn
    end

    @testset "StructuralDFM ★" begin
        # Small GDFM panel with q=2 common shocks.
        rng = Random.MersenneTwister(7)
        T_obs, N, q = 120, 12, 2
        F = zeros(T_obs, q); F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.5 .* F[t-1, :] .+ randn(rng, q)
        end
        X = F * randn(rng, N, q)' .+ 0.3 .* randn(rng, T_obs, N)
        sdfm = estimate_structural_dfm(X, q; identification=:cholesky, p=1, H=12)
        @test sdfm isa StructuralDFM{Float64}
        p = plot_result(sdfm)                        # delegates to the IRF dispatch
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Structural DFM", p.html)
        # var selection delegates through the IRF dispatch (C3).
        check_plot(plot_result(sdfm; var=1))
    end

    # =========================================================================
    # Data containers
    # =========================================================================
    @testset "TimeSeriesData" begin
        d = TimeSeriesData(randn(100, 3); varnames=["GDP", "CPI", "RATE"])
        p = plot_result(d)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("GDP", p.html)
        check_plot(plot_result(d; vars=["GDP", "CPI"]))
    end

    @testset "PanelData" begin
        df = DataFrame(group=repeat(1:3, inner=20), time=repeat(1:20, 3),
                       x=randn(60), y=randn(60))
        p = plot_result(xtset(df, :group, :time))
        check_plot(p); assert_all_json_valid(p)
    end

    # =========================================================================
    # Infrastructure — serializer helpers, resolution, composition, output
    # =========================================================================
    @testset "_json_obj / _json_array_of_objects" begin
        obj = _MEM_MODELS._json_obj([Pair("a", "1"), Pair("b", "\"x\"")])
        @test occursin("\"a\":1", obj)
        arr = _MEM_MODELS._json_array_of_objects([[Pair("x", "1")], [Pair("x", "2")]])
        @test startswith(arr, "[") && endswith(arr, "]")
        assert_strict_json(arr)
    end

    @testset "_resolve_var edge cases (bounds-checked, Int + String)" begin
        names = ["GDP", "CPI", "RATE"]
        @test _MEM_MODELS._resolve_var(nothing, names) == 1
        @test _MEM_MODELS._resolve_var(2, names) == 2
        @test _MEM_MODELS._resolve_var("CPI", names) == 2
        @test_throws ArgumentError _MEM_MODELS._resolve_var("INVALID", names)
        @test_throws ArgumentError _MEM_MODELS._resolve_var(99, names)
    end

    @testset "_make_plot source + note; PlotOutput; save_plot" begin
        panel = _MEM_MODELS._PanelSpec("test_id", "Test Title", "// js code")
        p = _MEM_MODELS._make_plot([panel, panel]; title="Test",
                                   source="Source: Author", note="Note: test")
        @test occursin("Source: Author", p.html)
        @test occursin("Note: test", p.html)

        pt = PlotOutput("<html></html>")
        @test pt isa PlotOutput && pt.html == "<html></html>"

        # save_plot returns the path and writes a self-contained document.
        p3 = plot_result(hp_filter(randn(120)))
        tmp = tempname() * ".html"
        @test save_plot(p3, tmp) == tmp
        content = read(tmp, String)
        @test startswith(strip(content), "<!DOCTYPE html>")
        @test !occursin("cdnjs", content)                 # vendored D3 (A12), no CDN
        @test occursin("append('svg')", content)
        rm(tmp; force=true)

        # show methods: text/plain summary + default show (text/html embed-safety is
        # exercised in test_plot_render.jl PLT-03).
        io = IOBuffer(); show(io, MIME"text/plain"(), p3)
        s = String(take!(io))
        @test occursin("PlotOutput", s) && occursin("bytes", s)
        io2 = IOBuffer(); show(io2, p3)
        @test occursin("PlotOutput", String(take!(io2)))
    end

    # =========================================================================
    # Escaping — hostile series name through a data container (A7/A8)
    # =========================================================================
    @testset "escaping (TimeSeriesData)" begin
        de = TimeSeriesData(randn(40, 2); varnames=[HOSTILE_NAME, "y"])
        assert_escapes(plot_result(de))
    end
end
