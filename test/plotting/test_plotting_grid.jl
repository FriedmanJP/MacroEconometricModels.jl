# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# Unit tests for Milestone 5:
#   Issue #880: Plotting Grid Layout and ncols Defaults
#   Issue #881: Asset Scoping, D3 Deduplication & Multi-Plot Saving

using LinearAlgebra
using MacroEconometricModels
using Random
using Test

# Include shared plotting assertions if not already loaded
if !isdefined(@__MODULE__, :check_plot)
    include(joinpath(@__DIR__, "plot_test_helpers.jl"))
end

const MEM = MacroEconometricModels

@testset "Milestone 5: Plotting Grid Layout & Asset Scoping (#880, #881)" begin

    # =========================================================================
    # 1. IRF Single-Shock Multi-Column Layout (#880)
    # =========================================================================
    @testset "IRF Single Shock Default ncols" begin
        rng = Xoshiro(8801)
        sim = dgp_var(rng; T=150)
        m = estimate_var(sim.Y, 1)
        r = irf(m, 8; ci_type=:none)

        # 3 variables, 1 shock: default ncols = min(3, 3) = 3 -> 33%
        p_1shock = plot_result(r; shock=1)
        check_plot(p_1shock)
        @test occursin("--mem-panel-width: 33%", p_1shock.html)
        @test length(panel_titles(p_1shock.html)) == 3

        # 2 variables, 1 shock: default ncols = min(2, 3) = 2 -> 50%
        p_2vars = plot_result(r; var=[1, 2], shock=1)
        check_plot(p_2vars)
        @test occursin("--mem-panel-width: 50%", p_2vars.html)
        @test length(panel_titles(p_2vars.html)) == 2

        # 1 variable, 1 shock: default ncols = min(1, 3) = 1 -> 100%
        p_1var = plot_result(r; var=1, shock=1)
        check_plot(p_1var)
        @test occursin("--mem-panel-width: 100%", p_1var.html)
        @test length(panel_titles(p_1var.html)) == 1

        # Explicit ncols override takes precedence
        p_override = plot_result(r; shock=1, ncols=2)
        @test occursin("--mem-panel-width: 50%", p_override.html)

        # BayesianImpulseResponse single shock default ncols
        post = estimate_bvar(sim.Y, 1; n_draws=40, rng=Xoshiro(8802))
        r_bayes = irf(post, 8)
        pb_1shock = plot_result(r_bayes; shock=1)
        check_plot(pb_1shock)
        @test occursin("--mem-panel-width: 33%", pb_1shock.html)
    end

    # =========================================================================
    # 2. IRF Row-Major Matrix Layout (#880)
    # =========================================================================
    @testset "IRF Row-Major Order (Vars x Shocks)" begin
        rng = Xoshiro(8803)
        sim = dgp_var(rng; T=150)
        m = estimate_var(sim.Y, 1)
        r = irf(m, 8; ci_type=:none)

        p = plot_result(r)
        check_plot(p)
        titles = panel_titles(p.html)
        @test length(titles) == 9

        # Row-major order: outer loop is variables, inner loop is shocks
        # Row 1: y1 with shocks 1, 2, 3
        @test occursin("y1", titles[1]) && occursin("Shock 1", titles[1])
        @test occursin("y1", titles[2]) && occursin("Shock 2", titles[2])
        @test occursin("y1", titles[3]) && occursin("Shock 3", titles[3])
        # Row 2: y2 with shocks 1, 2, 3
        @test occursin("y2", titles[4]) && occursin("Shock 1", titles[4])
        @test occursin("y2", titles[5]) && occursin("Shock 2", titles[5])
        @test occursin("y2", titles[6]) && occursin("Shock 3", titles[6])
        # Row 3: y3 with shocks 1, 2, 3
        @test occursin("y3", titles[7]) && occursin("Shock 1", titles[7])
        @test occursin("y3", titles[8]) && occursin("Shock 2", titles[8])
        @test occursin("y3", titles[9]) && occursin("Shock 3", titles[9])

        # Default ncols for multiple shocks equals number of shocks
        @test occursin("--mem-panel-width: 33%", p.html)

        # Custom subset and ordering preserved row-major
        p_sub = plot_result(r; var=[3, 1], shock=[2, 1])
        sub_titles = panel_titles(p_sub.html)
        @test length(sub_titles) == 4
        @test occursin("y3", sub_titles[1]) && occursin("Shock 2", sub_titles[1])
        @test occursin("y3", sub_titles[2]) && occursin("Shock 1", sub_titles[2])
        @test occursin("y1", sub_titles[3]) && occursin("Shock 2", sub_titles[3])
        @test occursin("y1", sub_titles[4]) && occursin("Shock 1", sub_titles[4])

        # BayesianImpulseResponse row-major order
        post = estimate_bvar(sim.Y, 1; n_draws=40, rng=Xoshiro(8804))
        r_bayes = irf(post, 8)
        pb = plot_result(r_bayes)
        b_titles = panel_titles(pb.html)
        @test occursin("y1", b_titles[1]) && occursin("y1", b_titles[1])
        @test occursin("y1", b_titles[2]) && occursin("y2", b_titles[2])
        @test occursin("y2", b_titles[4]) && occursin("y1", b_titles[4])
    end

    # =========================================================================
    # 3. HD 2-Column Side-by-Side Layout (#880)
    # =========================================================================
    @testset "HD 2-Column Layout Defaults" begin
        rng = Xoshiro(8805)
        sim = dgp_var(rng; T=150)
        m = estimate_var(sim.Y, 1)
        hd_res = historical_decomposition(m)

        # Default ncols for HistoricalDecomposition is 2
        p_hd = plot_result(hd_res)
        check_plot(p_hd)
        @test occursin("--mem-panel-width: 50%", p_hd.html)

        # Single variable filter: 2 panels (stacked bars + actual line), 2 columns
        p_hd_1var = plot_result(hd_res; var=1)
        check_plot(p_hd_1var)
        @test length(panel_titles(p_hd_1var.html)) == 2
        @test occursin("--mem-panel-width: 50%", p_hd_1var.html)

        # Explicit ncols override
        p_hd_col1 = plot_result(hd_res; var=1, ncols=1)
        @test occursin("--mem-panel-width: 100%", p_hd_col1.html)

        # BayesianHistoricalDecomposition views
        post = estimate_bvar(sim.Y, 1; n_draws=40, rng=Xoshiro(8806))
        bhd = historical_decomposition(post)

        # view=:stacked defaults to ncols=2
        p_bhd_stacked = plot_result(bhd; view=:stacked)
        check_plot(p_bhd_stacked)
        @test occursin("--mem-panel-width: 50%", p_bhd_stacked.html)

        # view=:fan defaults to ncols=min(n_vars, 3) for single shock
        p_bhd_fan_1s = plot_result(bhd; view=:fan, shock=1)
        check_plot(p_bhd_fan_1s)
        @test occursin("--mem-panel-width: 33%", p_bhd_fan_1s.html)
    end

    # =========================================================================
    # 4. OccBin Keyword Contracts (#880)
    # =========================================================================
    @testset "OccBinIRF and OccBinSolution ncols and var Contracts" begin
        oirf = OccBinIRF{Float64}(
            zeros(15, 3),
            zeros(15, 3),
            falses(15),
            ["output", "inflation", "rate"],
        )

        # 3 variables default to min(3, 3) = 3 columns
        p_oirf = plot_result(oirf)
        check_plot(p_oirf)
        @test occursin("--mem-panel-width: 33%", p_oirf.html)
        @test length(panel_titles(p_oirf.html)) == 3

        # 2 variables default to 2 columns
        p_oirf_2v = plot_result(oirf; var=["output", "inflation"])
        check_plot(p_oirf_2v)
        @test occursin("--mem-panel-width: 50%", p_oirf_2v.html)
        @test length(panel_titles(p_oirf_2v.html)) == 2

        # 1 variable by index defaults to 1 column
        p_oirf_1v = plot_result(oirf; var=2)
        check_plot(p_oirf_1v)
        @test occursin("--mem-panel-width: 100%", p_oirf_1v.html)
        @test length(panel_titles(p_oirf_1v.html)) == 1

        # Explicit ncols override
        p_oirf_col1 = plot_result(oirf; ncols=1)
        @test occursin("--mem-panel-width: 100%", p_oirf_col1.html)

        # Invalid variable specification raises ArgumentError
        @test_throws ArgumentError plot_result(oirf; var="nonexistent")
        @test_throws ArgumentError plot_result(oirf; var=99)

        # OccBinSolution contract
        osol = OccBinSolution{Float64}(
            zeros(20, 2),
            zeros(20, 2),
            falses(20),
            ["c", "k"],
        )

        # 2 variables default to 2 columns
        p_osol = plot_result(osol)
        check_plot(p_osol)
        @test occursin("--mem-panel-width: 50%", p_osol.html)
        @test length(panel_titles(p_osol.html)) == 2

        # Filter by string name
        p_osol_c = plot_result(osol; var="c")
        check_plot(p_osol_c)
        @test occursin("--mem-panel-width: 100%", p_osol_c.html)
        @test length(panel_titles(p_osol_c.html)) == 1
    end

    # =========================================================================
    # 5. CSS Variable Scoping & Fragment Isolation (#881)
    # =========================================================================
    @testset "CSS Variable Scoping & Embedding Isolation" begin
        rng = Xoshiro(8807)
        sim = dgp_var(rng; T=150)
        m = estimate_var(sim.Y, 1)
        r = irf(m, 8; ci_type=:none)

        p3 = plot_result(r; ncols=3)
        p1 = plot_result(r; ncols=1)

        # CSS template must define .panel using var(--mem-panel-width, 100%)
        @test occursin("--mem-panel-width", p3.html)
        @test occursin("var(--mem-panel-width", p3.html)
        @test occursin("--mem-panel-width", p1.html)
        @test occursin("var(--mem-panel-width", p1.html)

        # Global .panel rule should NOT hardcode fixed percentage widths
        @test !occursin("width: calc(33% - 16px);", p3.html)
        @test !occursin("width: calc(100% - 16px);", p1.html)

        # Fragments must carry their respective inline scoping on .panel-grid
        frag3 = MEM._render_fragment(p3)
        frag1 = MEM._render_fragment(p1)
        @test occursin("--mem-panel-width: 33%", frag3)
        @test occursin("--mem-panel-width: 100%", frag1)

        # Combined document with both fragments preserves scoped styles
        combined = """
        <!DOCTYPE html>
        <html>
        <head><title>Combined Scoping Test</title></head>
        <body>
        $(frag3)
        $(frag1)
        </body>
        </html>
        """
        @test occursin("--mem-panel-width: 33%", combined)
        @test occursin("--mem-panel-width: 100%", combined)
    end

    # =========================================================================
    # 6. D3 Deduplication & Guarding (#881)
    # =========================================================================
    @testset "D3 Guarding and Deduplication" begin
        rng = Xoshiro(8808)
        sim = dgp_var(rng; T=150)
        m = estimate_var(sim.Y, 1)
        r = irf(m, 8; ci_type=:none)
        p = plot_result(r)

        # Fragment must guard D3 load against duplicate re-evaluation
        frag = MEM._render_fragment(p)
        @test occursin("if (typeof window.d3 === 'undefined')", frag)

        # Multi-plot rendering helper must include D3 script once
        p_a = plot_result(r; var=1, shock=1)
        p_b = plot_result(r; var=2, shock=2)
        multi_html = MEM._render_multi_plot_html([p_a, p_b]; title="Multi Test")

        # Must have exactly one <!DOCTYPE html> and one <head>
        @test length(collect(eachmatch(r"<!DOCTYPE html>"i, multi_html))) == 1
        @test length(collect(eachmatch(r"<head>"i, multi_html))) == 1
        @test length(collect(eachmatch(r"</head>"i, multi_html))) == 1

        # Deduplicated multi-plot HTML should have single D3 definition
        d3_count = length(collect(eachmatch(r"typeof window\.d3"i, multi_html)))
        @test d3_count == 1 || d3_count == 0
    end

    # =========================================================================
    # 7. Multi-Plot save_plot Overloads (#881)
    # =========================================================================
    @testset "Multi-Plot save_plot and Overloads" begin
        rng = Xoshiro(8809)
        sim = dgp_var(rng; T=150)
        m = estimate_var(sim.Y, 1)
        r = irf(m, 8; ci_type=:none)

        p1 = plot_result(r; var=1, shock=1, title="Plot 1")
        p2 = plot_result(r; var=2, shock=2, title="Plot 2")

        # Vector overload save_plot
        tmp_vec = tempname() * ".html"
        save_plot([p1, p2], tmp_vec; title="Batch Saved Plots")
        @test isfile(tmp_vec)
        content_vec = read(tmp_vec, String)
        @test startswith(strip(content_vec), "<!DOCTYPE html>")
        @test occursin("Batch Saved Plots", content_vec)
        @test occursin("Plot 1", content_vec)
        @test occursin("Plot 2", content_vec)
        rm(tmp_vec; force=true)

        # Tuple overload save_plot
        tmp_tup = tempname() * ".html"
        save_plot((p1, p2), tmp_tup; title="Tuple Saved Plots")
        @test isfile(tmp_tup)
        content_tup = read(tmp_tup, String)
        @test startswith(strip(content_tup), "<!DOCTYPE html>")
        @test occursin("Tuple Saved Plots", content_tup)
        rm(tmp_tup; force=true)

        # Empty collection handling
        tmp_empty = tempname() * ".html"
        @test_throws ArgumentError save_plot(PlotOutput[], tmp_empty)
        @test_throws ArgumentError save_plot((), tmp_empty)
        @test_throws ArgumentError display_plot(PlotOutput[])
        @test_throws ArgumentError display_plot(())
    end
end
