# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Wave-2 Lane C plotting tests (PLT-29 PVAR, PLT-30 break tests, PLT-31 set-ID
# SVAR). Uses the shared assertions in plot_test_helpers.jl (Testing Rules 1-7):
# check_plot, assert_all_json_valid, assert_escapes, assert_nan_becomes_null,
# series_count/series_names, panel_titles, HOSTILE_NAME.
# =============================================================================

using Test, Random, DataFrames, LinearAlgebra

# Self-bootstrap the shared helpers when run standalone.
if !isdefined(@__MODULE__, :check_plot)
    include(joinpath(@__DIR__, "plot_test_helpers.jl"))
end

# Small stationary panel VAR for the PVAR dispatches.
function _lanec_panel(; N=12, Tt=16, m=2, varnames=nothing, seed=7)
    rng = MersenneTwister(seed)
    A1 = 0.3 * I(m) + 0.03 * randn(rng, m, m)
    dm = zeros(N * Tt, m)
    for i in 1:N
        mu = randn(rng, m); off = (i - 1) * Tt
        dm[off + 1, :] = mu
        for t in 2:Tt
            dm[off + t, :] = mu + A1 * dm[off + t - 1, :] + 0.1 * randn(rng, m)
        end
    end
    names = varnames === nothing ? ["y$i" for i in 1:m] : varnames
    df = DataFrame(dm, names)
    df.id = repeat(1:N, inner=Tt); df.time = repeat(1:Tt, outer=N)
    xtset(df, :id, :time)
end

@testset "Wave-2 Lane C — PVAR / break tests / set-ID SVAR" begin

    # =========================================================================
    # PLT-29 — Panel VAR
    # =========================================================================
    @testset "PLT-29 PVAR" begin
        mdl = estimate_pvar_feols(_lanec_panel(), 1)

        @testset "views render + JSON valid" begin
            for v in (:oirf, :girf, :fevd, :stability)
                p = plot_result(mdl; view=v, H=6)
                check_plot(p)
                assert_all_json_valid(p)
            end
        end

        @testset "IRF form + auto title" begin
            p = plot_result(mdl; view=:oirf, H=6)
            @test occursin("Panel VAR Orthogonalized", p.html)
            # one panel per (response ← shock) = m*m
            @test length(panel_titles(p.html)) == mdl.m * mdl.m
            g = plot_result(mdl; view=:girf, H=6)
            @test occursin("Generalized", g.html)
        end

        @testset "bootstrap CI bands" begin
            bs = pvar_bootstrap_irf(mdl, 6; irf_type=:oirf, n_draws=20)
            p = plot_result(mdl; view=:oirf, H=6, ci=bs)
            check_plot(p)
            @test occursin("bootstrap CI", p.html)
            @test occursin("ci_lo", p.html)   # band keys present
        end

        @testset "var/shock selection Int + String" begin
            p_i = plot_result(mdl; view=:oirf, H=6, var=1, shock=2)
            p_s = plot_result(mdl; view=:oirf, H=6, var="y1", shock="y2")
            @test length(panel_titles(p_i.html)) == 1
            @test length(panel_titles(p_s.html)) == 1
            # FEVD by var name resolves through _resolve_var
            pf = plot_result(mdl; view=:fevd, H=6, var="y2")
            @test length(panel_titles(pf.html)) == 1
        end

        @testset "bad selection / view → ArgumentError" begin
            @test_throws ArgumentError plot_result(mdl; view=:oirf, var="nope")
            @test_throws ArgumentError plot_result(mdl; view=:oirf, var=99)
            @test_throws ArgumentError plot_result(mdl; view=:fevd, var=99)
            @test_throws ArgumentError plot_result(mdl; view=:bogus)
            # per-view kwarg guards (plotrule C5)
            @test_throws ArgumentError plot_result(mdl; view=:fevd, shock=1)
            @test_throws ArgumentError plot_result(mdl; view=:stability, var=1)
        end

        @testset "wrappers build canonical types" begin
            r = pvar_irf(mdl, 6)
            @test r isa ImpulseResponse
            @test r.ci_type == :none
            f = pvar_fevd_result(mdl, 6)
            @test f isa FEVD
            @test size(f.proportions) == (mdl.m, mdl.m, 7)
            # rows sum to 1 (proportions)
            @test all(isapprox.(sum(f.proportions[1, :, :], dims=1), 1; atol=1e-6))
            @test_throws ArgumentError pvar_irf(mdl, 6; irf_type=:bad)
        end

        @testset "stability panel + escaping + save_path" begin
            p = plot_result(mdl; view=:stability)
            pt = panel_titles(p.html)
            @test any(t -> occursin("Companion eigenvalues", t), pt)
            @test any(t -> occursin("|λ|", t), pt)
            # hostile variable names survive every sink
            mdl_h = estimate_pvar_feols(_lanec_panel(varnames=[HOSTILE_NAME, "y2"]), 1)
            ph = plot_result(mdl_h; view=:oirf, H=5)
            assert_escapes(ph)
            # save_path writes a file and returns the PlotOutput
            tmp = tempname() * ".html"
            p2 = plot_result(mdl; view=:oirf, H=5, save_path=tmp)
            @test p2 isa PlotOutput
            @test isfile(tmp); rm(tmp; force=true)
        end

        @testset "1×1 PVAR + NaN eigenvalue" begin
            m1 = estimate_pvar_feols(_lanec_panel(m=1), 1)
            check_plot(plot_result(m1; view=:oirf, H=5))
            check_plot(plot_result(m1; view=:stability))
            # NaN eigenvalue → null in the scatter data
            s = PVARStability{Float64}([complex(NaN, NaN), complex(0.5, 0.0)],
                                       [NaN, 0.5], false)
            p = plot_result(s)
            assert_nan_becomes_null(p)
        end
    end

    # =========================================================================
    # PLT-30 — unit-root & structural-break tests
    # =========================================================================
    @testset "PLT-30 break tests" begin
        za = ZAResult(-4.2, 0.03, 40, 0.4, :both,
                      Dict(1 => -5.34, 5 => -4.8, 10 => -4.58), 2, 100)
        and = AndrewsResult(18.0, 0.01, 55, 0.55, :supwald,
                            Dict(1 => 16.0, 5 => 12.0, 10 => 10.0),
                            Float64[5, 8, 12, 18, 14, 9, 6], 0.15, 120, 2)
        bp = BaiPerronResult(2, [30, 70], [(25, 35), (65, 75)], [Float64[]], [Float64[]],
                             Float64[10, 8], Float64[9], Float64[7], Float64[0.4],
                             Float64[100, 95, 93], Float64[102, 98, 99], 0.15, 120)
        adf2 = ADF2BreakResult(-5.1, 0.02, 30, 70, 0.3, 0.7, 2, :both,
                               Dict(1 => -5.7, 5 => -5.2, 10 => -4.9), 100)
        gh = GregoryHansenResult(-5.5, 0.02, -5.3, 0.03, -40.0, 0.02, 45, 46, 44,
                                 :cshift, 1, Dict(1 => -5.7, 5 => -5.3, 10 => -5.0),
                                 Dict(1 => -50.0, 5 => -45.0, 10 => -40.0), 100)
        joh = JohansenResult(Float64[35, 10], Float64[0.01, 0.3], Float64[25, 10],
                             Float64[0.01, 0.3], 1, zeros(2, 2), zeros(2, 2),
                             Float64[0.3, 0.1],
                             [20.0 15.5 12.0; 9.0 6.5 4.0], [18.0 14.0 11.0; 9.0 6.0 4.0],
                             :constant, 2, 100)
        fadf = FourierADFResult(-3.8, 0.04, 1, 12.0, 0.001, 2, :constant,
                                Dict(1 => -4.5, 5 => -3.9, 10 => -3.6), Dict(5 => 6.0), 100)
        fkpss = FourierKPSSResult(0.15, 0.08, 1, 10.0, 0.002, :constant,
                                  Dict(1 => 0.27, 5 => 0.17, 10 => 0.12), Dict(5 => 6.0), 4, 100)

        @testset "each dispatch renders + JSON valid" begin
            for r in (za, and, adf2, gh, joh, fadf, fkpss)
                p = plot_result(r)
                check_plot(p)
                assert_all_json_valid(p)
            end
        end

        @testset "Andrews sequential path + break/CV refs" begin
            p = plot_result(and)
            @test occursin("Candidate break index", p.html)
            @test occursin("sup-Wald", p.html)
            # break vline (axis:"x") and CV ref line present
            @test occursin("\"axis\":\"x\"", p.html)
            @test occursin("\"axis\":\"y\"", p.html)
            @test any(t -> occursin("Break at obs 55", t), panel_titles(p.html))
        end

        @testset "Bai-Perron multi-view" begin
            pc = plot_result(bp; view=:criteria)
            @test occursin("Number of breaks", pc.html)
            @test series_count(pc.html) == 2               # BIC + LWZ
            @test Set(series_names(pc.html)) == Set(["BIC", "LWZ"])
            pb = plot_result(bp; view=:breaks)
            @test any(t -> occursin("2 break", t), panel_titles(pb.html))
            @test_throws ArgumentError plot_result(bp; view=:nope)
            # 0-break degenerate breaks view still renders
            bp0 = BaiPerronResult(0, Int[], Tuple{Int,Int}[], [Float64[]], [Float64[]],
                                  Float64[], Float64[], Float64[], Float64[],
                                  Float64[100.0], Float64[102.0], 0.15, 80)
            p0 = plot_result(bp0; view=:breaks)
            @test any(t -> occursin("No structural breaks", t), panel_titles(p0.html))
        end

        @testset "grouped statistic-vs-CV bars" begin
            pg = plot_result(gh)
            @test series_count(pg.html) == 2
            @test Set(series_names(pg.html)) == Set(["Statistic", "5% CV"])
            @test any(t -> occursin("Break at obs 45", t), panel_titles(pg.html))
            pj = plot_result(joh)
            @test series_count(pj.html) == 2
            @test any(t -> occursin("rank = 1", t), panel_titles(pj.html))
        end

        @testset "Fourier honest subtitle (no phantom series)" begin
            for r in (fadf, fkpss)
                p = plot_result(r)
                @test any(t -> occursin("not stored", t), panel_titles(p.html))
            end
        end

        @testset "NaN in a sequence → null; degenerate single-point" begin
            and_nan = AndrewsResult(18.0, 0.01, 3, 0.5, :supwald,
                                    Dict(1 => 16.0, 5 => 12.0, 10 => 10.0),
                                    Float64[5, NaN, 12], 0.15, 20, 2)
            assert_nan_becomes_null(plot_result(and_nan))
            bp_nan = BaiPerronResult(1, [30], [(25, 35)], [Float64[]], [Float64[]],
                                     Float64[10], Float64[9], Float64[7], Float64[0.4],
                                     Float64[100, NaN], Float64[102, 98], 0.15, 80)
            assert_nan_becomes_null(plot_result(bp_nan; view=:criteria))
            # single-element sequence: no exception
            and1 = AndrewsResult(9.0, 0.2, 50, 0.5, :meanwald,
                                 Dict(5 => 12.0), Float64[9.0], 0.15, 100, 1)
            check_plot(plot_result(and1))
        end
    end

    # =========================================================================
    # PLT-31 — set-identified SVAR IRFs
    # =========================================================================
    @testset "PLT-31 set-ID SVAR" begin
        H = 8; n = 2; nd = 60
        draws = randn(MersenneTwister(3), nd, H, n, n) .* 0.5
        sis = SignIdentifiedSet{Float64}([randn(n, n) for _ in 1:nd], draws, nd, 120,
                                         nd / 120, ["gdp", "infl"], ["demand", "supply"])
        restr = SVARRestrictions(n)
        arias = AriasSVARResult{Float64}([randn(n, n) for _ in 1:nd], draws,
                                         fill(1 / nd, nd), nd / 120, restr)
        uh = UhligSVARResult{Float64}(randn(n, n), randn(H, n, n), -1.5,
                                      Float64[-0.7, -0.8], restr, true)

        @testset "SignIdentifiedSet fan" begin
            p = plot_result(sis)
            check_plot(p); assert_all_json_valid(p)
            @test length(panel_titles(p.html)) == n * n
            @test occursin("68% band", p.html)                # C7 draw count / band
            @test occursin("950 draws", p.html) == false      # sanity: uses actual count
            @test occursin("$(nd) draws", p.html)
            @test occursin("Median", p.html)                  # central line label
            # single-panel selection Int + String
            @test length(panel_titles(plot_result(sis; var="gdp", shock=1).html)) == 1
            @test length(panel_titles(plot_result(sis; var=2, shock="supply").html)) == 1
            @test_throws ArgumentError plot_result(sis; var="nope")
            @test_throws ArgumentError plot_result(sis; shock=99)
        end

        @testset "quantiles override (nested bands)" begin
            p = plot_result(sis; quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])
            check_plot(p)
            @test occursin("90% band", p.html)
            @test occursin("5–95%", p.html)                   # outer band legend label
            @test occursin("16–84%", p.html)                  # inner band legend label
        end

        @testset "Arias reuses weighted helpers + name synthesis/override" begin
            p = plot_result(arias)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("Mean", p.html)                    # central = weighted mean
            @test occursin("Var 1", p.html) && occursin("Shock 1", p.html)
            # override names + selection by name
            po = plot_result(arias; variables=["a", "b"], shocks=["s1", "s2"], var="a", shock="s2")
            @test length(panel_titles(po.html)) == 1
            @test occursin("a ← s2", po.html)
            @test_throws ArgumentError plot_result(arias; variables=["only1"])
        end

        @testset "Uhlig single rotation — line only, no band" begin
            p = plot_result(uh)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("converged", p.html)
            @test series_count(p.html) == 1                   # one line
            @test !occursin("\"lo_key\"", p.html)             # no CI band drawn (single rotation)
        end

        @testset "escaping + NaN draw + single accepted draw" begin
            sis_h = SignIdentifiedSet{Float64}([randn(n, n) for _ in 1:nd], draws, nd, 120,
                                               nd / 120, [HOSTILE_NAME, "infl"],
                                               ["demand", HOSTILE_NAME])
            assert_escapes(plot_result(sis_h))
            arias_h = AriasSVARResult{Float64}([randn(n, n) for _ in 1:nd], draws,
                                               fill(1 / nd, nd), nd / 120, restr)
            assert_escapes(plot_result(arias_h; variables=[HOSTILE_NAME, "b"],
                                       shocks=["s1", HOSTILE_NAME]))
            # NaN draw → null (SignIdentifiedSet pointwise-quantile path)
            dn = copy(draws); dn[1, 3, 1, 1] = NaN
            sis_n = SignIdentifiedSet{Float64}([randn(n, n) for _ in 1:nd], dn, nd, 120,
                                               nd / 120, ["gdp", "infl"], ["demand", "supply"])
            assert_nan_becomes_null(plot_result(sis_n))
            # single accepted draw does not throw
            d1 = randn(1, H, n, n)
            sis1 = SignIdentifiedSet{Float64}([randn(n, n)], d1, 1, 10, 0.1,
                                              ["a", "b"], ["c", "d"])
            check_plot(plot_result(sis1))
        end
    end
end
