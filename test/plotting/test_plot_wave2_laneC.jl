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
using MacroEconometricModels

# Self-bootstrap the shared helpers when run standalone.
if !isdefined(@__MODULE__, :check_plot)
    include(joinpath(@__DIR__, "plot_test_helpers.jl"))
end

# Small stationary panel VAR for the PVAR dispatches.
function _lanec_panel(; N=12, Tt=16, m=2, varnames=nothing, seed=7)
    rng = Xoshiro(seed)
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
                @test any(t -> occursin("k=", t) && occursin("5%:", t), panel_titles(p.html))
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

        @testset "Unit-root bars (#841 PR1)" begin
            adf = ADFResult(-3.5, 0.01, 4, :constant,
                            Dict(1 => -3.5, 5 => -2.9, 10 => -2.6), 200)
            kpss = KPSSResult(0.2, 0.05, :constant,
                              Dict(1 => 0.74, 5 => 0.46, 10 => 0.35), 4, 200)
            pp = PPResult(-2.0, 0.2, :constant,
                          Dict(1 => -3.5, 5 => -2.9, 10 => -2.6), 4, 200)
            ers = ERSResult(2.5, 0.03, :constant,
                            Dict(1 => 1.9, 5 => 3.0, 10 => 4.2), 200)
            for r in (adf, kpss, pp, ers)
                p = plot_result(r)
                check_plot(p)
                assert_all_json_valid(p)
            end
            # left-tailed ADF/PP/ERS reject below CV; right-tailed KPSS above CV
            @test any(t -> occursin("5%: reject", t),
                      panel_titles(plot_result(adf).html))
            @test any(t -> occursin("5%: fail to reject", t),
                      panel_titles(plot_result(pp).html))
            @test any(t -> occursin("5%: fail to reject", t),
                      panel_titles(plot_result(kpss).html))
            @test any(t -> occursin("5%: reject", t),
                      panel_titles(plot_result(ers).html))
            # level selects the decision CV: ADF -3.5 rejects at 10% only
            @test any(t -> occursin("1%: fail to reject", t),
                      panel_titles(plot_result(adf; level=1).html))
            @test any(t -> occursin("10%: reject", t),
                      panel_titles(plot_result(adf; level=10).html))
            # subtitles carry the distinguishing spec
            @test any(t -> occursin("lags=4", t),
                      panel_titles(plot_result(adf).html))
            @test any(t -> occursin("bandwidth=4", t),
                      panel_titles(plot_result(kpss).html))
            # integration: real estimators render
            y = randn(Xoshiro(841), 200)
            for r in (adf_test(y), kpss_test(y), pp_test(y), ers_test(y))
                check_plot(plot_result(r))
            end
        end

        @testset "Multi-stat + panel unit-root bars (#841 PR2)" begin
            ngp = NgPerronResult(-9.0, -2.0, 0.24, 3.5, :constant,
                Dict(:MZa => Dict(1 => -13.8, 5 => -8.1, 10 => -5.7),
                     :MZt => Dict(1 => -2.58, 5 => -1.98, 10 => -1.62),
                     :MSB => Dict(1 => 0.17, 5 => 0.23, 10 => 0.28),
                     :MPT => Dict(1 => 1.78, 5 => 3.17, 10 => 4.45)), 200)
            hegy = HEGYResult(4, :const_trend_seas, 2, [0.1, 0.2, 0.3], -2.5, -1.8,
                Dict(1 => -3.5, 5 => -2.9, 10 => -2.6),
                Dict(1 => -3.0, 5 => -2.5, 10 => -2.2),
                [Float64(pi / 2)], [4.5], Dict(1 => 7.0, 5 => 5.0, 10 => 4.0),
                5.5, 6.5, 100)
            llc = LLCResult(-2.5, 0.006, -2.0, -0.05, 1.0, 0.0, 1.0, 50.0,
                            [2, 2, 2], :constant, 60, 3)
            ips = IPSResult(-2.2, 0.014, -2.0, [-2.1, -2.0, -1.9], 0.0, 1.0,
                            [2, 2, 2], :constant, 60, 3)
            br = BreitungPanelResult(-1.9, 0.029, 1, :constant, 60, 3)
            fp = FisherPanelResult(25.0, 0.01, 25.0, 0.01, -2.0, 0.02, -1.8, 0.04,
                3.0, 0.001, [0.01, 0.02, 0.03], :adf, :mw, 60, 3)
            ha = HadriResult(2.5, 0.006, 0.5, 0.16, 0.1, false, :constant, 60, 3)
            for r in (ngp, hegy, llc, ips, br, fp, ha)
                p = plot_result(r)
                check_plot(p)
                assert_all_json_valid(p)
            end
            # Ng-Perron counts rejections across the four stats (2/4 here)
            @test any(t -> occursin("2/4 reject", t),
                      panel_titles(plot_result(ngp).html))
            # HEGY draws two panels (t-stats + pair Fs)
            @test length(panel_titles(plot_result(hegy).html)) == 2
            @test any(t -> occursin("F_seasonal=", t),
                      panel_titles(plot_result(hegy).html))
            # N(0,1) tails: LLC left rejects, Hadri right rejects
            @test any(t -> occursin("5%: reject", t),
                      panel_titles(plot_result(llc).html))
            @test any(t -> occursin("5%: reject", t),
                      panel_titles(plot_result(ha).html))
            # Fisher α selects the count: 4/4 at 5%, 1/4 at 1%
            @test any(t -> occursin("4/4 reject", t),
                      panel_titles(plot_result(fp).html))
            @test any(t -> occursin("1/4 reject", t),
                      panel_titles(plot_result(fp; level=1).html))
            # integration: real estimators render
            y = randn(Xoshiro(842), 200)
            check_plot(plot_result(ngperron_test(y)))
            check_plot(plot_result(hegy_test(y)))
            X = randn(Xoshiro(843), 60, 3)
            for r in (llc_test(X), ips_test(X), breitung_panel_test(X),
                      fisher_panel_test(X), hadri_test(X))
                check_plot(plot_result(r))
            end
        end

        @testset "Panel-2 + cointegration bars (#841 PR3)" begin
            mp = MoonPerronResult(-2.0, -1.5, 0.02, 0.07, 2, 60, 5)
            pa = PANICResult([-2.0, -1.5], [0.02, 0.07], -2.2, 0.014,
                             [-2.0, -1.8, -1.5], [0.02, 0.04, 0.07],
                             1, :pooled, 60, 3)
            ci = PesaranCIPSResult(-2.4, 0.01, [-2.5, -2.3, -2.4],
                                   Dict(1 => -2.6, 5 => -2.2, 10 => -2.0),
                                   1, :constant, 60, 3)
            ka = KaoResult(["DFrho", "DFt", "DFrho_star", "DFt_star", "ADF"],
                           [-2.0, -2.5, -1.8, -2.2, -2.4],
                           [0.02, 0.006, 0.04, 0.014, 0.008],
                           0.9, -2.0, -2.4, 1.0, 1.2, 1, 1, 2, 60, 5)
            pe = PedroniResult(["panel-v", "panel-rho", "panel-t", "panel-adf",
                                "group-rho", "group-t", "group-adf"],
                               zeros(7), [2.0, -2.0, -2.2, -2.4, -1.9, -2.1, -1.2],
                               [0.02, 0.02, 0.014, 0.008, 0.03, 0.018, 0.115],
                               zeros(7), ones(7), :constant, 2, 3, 1, 60, 5)
            we = WesterlundResult(["Gt", "Ga", "Pt", "Pa"],
                                  [-1.5, -2.0, -6.0, -7.0], [-2.0, -2.2, -2.4, -2.6],
                                  [0.02, 0.014, 0.008, 0.005],
                                  [NaN, NaN, NaN, NaN],
                                  :constant, 2, 1, 1, 3, 0, 0, 60, 5)
            eg = EngleGrangerResult(-3.5, 0.02, 1, :constant, 2, 3, 100)
            po = PhillipsOuliarisResult(-3.2, 0.03, -15.0, 0.04, :constant,
                                        :bartlett, 3.0, 2, 3, 100)
            pk = ParkAddedResult(8.5, 0.04, 2, 1, :constant, :const, 2, 100)
            for r in (mp, pa, ci, ka, pe, we, eg, po, pk)
                p = plot_result(r)
                check_plot(p)
                assert_all_json_valid(p)
            end
            # Moon-Perron: t_a rejects, t_b fails at 5%
            @test any(t -> occursin("1/2 reject", t),
                      panel_titles(plot_result(mp).html))
            # PANIC pooled + 3 units: 0.07 fails at 5%
            @test any(t -> occursin("3/4 reject", t),
                      panel_titles(plot_result(pa).html))
            # CIPS below 5% CV rejects
            @test any(t -> occursin("5%: reject", t),
                      panel_titles(plot_result(ci).html))
            # Pedroni panel-v is right-tailed: 2.0 rejects while -1.2 fails
            @test any(t -> occursin("6/7 reject", t),
                      panel_titles(plot_result(pe).html))
            # Westerlund at 1%: only Pt/Pa clear -2.33
            @test any(t -> occursin("2/4 reject", t),
                      panel_titles(plot_result(we; level=1).html))
            # EG rejects at 5%; PO 2/2 at 10%; Park rejects at 5%
            @test any(t -> occursin("1/1 reject", t),
                      panel_titles(plot_result(eg).html))
            @test any(t -> occursin("2/2 reject", t),
                      panel_titles(plot_result(po; level=10).html))
            @test any(t -> occursin("1/1 reject", t),
                      panel_titles(plot_result(pk).html))
            # integration: matrix/vector estimators render
            Xp = randn(Xoshiro(844), 60, 5)
            for r in (moon_perron_test(Xp), panic_test(Xp), pesaran_cips_test(Xp))
                check_plot(plot_result(r))
            end
            xc = cumsum(randn(Xoshiro(845), 100))
            Xc = hcat(xc, cumsum(randn(Xoshiro(846), 100)))
            yc = 1.5 .* xc .+ randn(Xoshiro(847), 100)
            check_plot(plot_result(engle_granger_test(yc, Xc)))
            check_plot(plot_result(phillips_ouliaris_test(yc, Xc)))
            check_plot(plot_result(park_added_test(estimate_cointreg(yc, xc))))
        end

        @testset "Portmanteau + causality bars (#841 PR4)" begin
            lb = LjungBoxResult(18.5, 0.02, 10, 10, 200)
            bp = BoxPierceResult(15.0, 0.13, 10, 10, 200)
            dw = DurbinWatsonResult(1.87, 0.30, 200)
            bds = BDSResult([2, 3], [0.5, 1.0], [0.5, 1.0], 1.0,
                            [2.5 1.2; 2.0 0.8], [0.01 0.23; 0.045 0.42],
                            fill(NaN, 2, 2), ones(2, 2), 300, false, 0, 0)
            vr = VarianceRatioResult([2, 4], [1.1, 1.2], [1.0, 1.5], [0.9, 1.4],
                                     [0.32, 0.13], [0.37, 0.16],
                                     1.5, 0.13, 1.4, 0.16,
                                     :lomackinlay, true, false,
                                     Float64[], Float64[], Float64[],
                                     Float64[], Float64[], Float64[],
                                     0, :rademacher, 0, Float64[], NaN, 200)
            gc = GrangerCausalityResult(12.5, 0.006, 4, [2], 1, 3, 4, 200, :block)
            vg = VECMGrangerResult(9.0, 0.01, 2, 2.0, 0.16, 1, 12.0, 0.007, 3, 2, 1)
            for r in (lb, bp, dw, bds, vr, gc, vg)
                p = plot_result(r)
                check_plot(p)
                assert_all_json_valid(p)
            end
            # χ² tails: LB rejects at 5% but not 1%; BP fails at 5%
            @test any(t -> occursin("5%: reject", t),
                      panel_titles(plot_result(lb).html))
            @test any(t -> occursin("1%: fail to reject", t),
                      panel_titles(plot_result(lb; level=1).html))
            @test any(t -> occursin("5%: fail to reject", t),
                      panel_titles(plot_result(bp).html))
            # DW subtitle carries the statistic; BDS 2/4 cells reject
            @test any(t -> occursin("DW=1.870", t),
                      panel_titles(plot_result(dw).html))
            @test any(t -> occursin("2/4 reject", t),
                      panel_titles(plot_result(bds).html))
            # VR robust branch: nothing rejects at 5%
            @test any(t -> occursin("0/3 reject", t),
                      panel_titles(plot_result(vr).html))
            # Granger rejects; VECM long-run fails → 2/3
            @test any(t -> occursin("5%: reject", t),
                      panel_titles(plot_result(gc).html))
            @test any(t -> occursin("2/3 reject", t),
                      panel_titles(plot_result(vg).html))
            # integration: series tests render
            y = randn(Xoshiro(848), 200)
            for r in (ljung_box_test(y; lags=5), box_pierce_test(y; lags=5),
                      durbin_watson_test(y), bds_test(y), variance_ratio_test(y))
                check_plot(plot_result(r))
            end
            # integration: VAR/VECM causality renders
            Y = randn(Xoshiro(849), 100, 3)
            m = estimate_var(Y, 2)
            check_plot(plot_result(granger_test(m, 2, 1)))
            v = estimate_vecm(Y, 2; rank=1)
            check_plot(plot_result(granger_causality_vecm(v, 2, 1)))
        end

        @testset "Spec/misc bars + stationarity scatter (#841 PR5)" begin
            lm = LMTestResult(9.5, 0.02, 3, 200, 0.8)
            lr = LRTestResult(7.2, 0.03, 2, -100.0, -96.4, 3, 5, 200, 200)
            eq = EqualityTestResult(:anova, 4.2, 0.01, 2.0, 57.0, 3,
                                    [20, 20, 20], false, "one-way ANOVA")
            hi = HansenInstabilityResult(0.35, 0.20, :constant, :const, 4, 2, 100)
            nr = NormalityTestResult(:jarque_bera, 6.5, 0.04, 2, 2, 100,
                                     [3.0, 3.5], [0.08, 0.03])
            fb = FactorBreakResult(12.0, 0.03, 45, :han_inoue, 2, 120, 10)
            pt = PanelTestResult("CD test", 1.8, 0.07, 5, "Pesaran CD")
            pj = PVARTestResult("Hansen J-test", 8.0, 0.09, 4, 12, 8)
            vs_bad = VARStationarityResult(false, ComplexF64[1.2 + 0im], 1.2,
                                           zeros(1, 1))
            for r in (lm, lr, eq, hi, nr, fb, pt, pj, vs_bad)
                p = plot_result(r)
                check_plot(p)
                assert_all_json_valid(p)
            end
            # χ² tails: LM rejects at 5% but not 1%; LR rejects; J fails
            @test any(t -> occursin("5%: reject", t),
                      panel_titles(plot_result(lm).html))
            @test any(t -> occursin("1%: fail to reject", t),
                      panel_titles(plot_result(lm; level=1).html))
            @test any(t -> occursin("5%: reject", t),
                      panel_titles(plot_result(lr).html))
            @test any(t -> occursin("5%: fail to reject", t),
                      panel_titles(plot_result(pj).html))
            # p-value bars: equality rejects; Hansen fails; normality 2/3
            @test any(t -> occursin("1/1 reject", t),
                      panel_titles(plot_result(eq).html))
            @test any(t -> occursin("ANOVA", t),
                      panel_titles(plot_result(eq).html))
            @test any(t -> occursin("0/1 reject", t),
                      panel_titles(plot_result(hi).html))
            @test any(t -> occursin("L_c=0.350", t),
                      panel_titles(plot_result(hi).html))
            @test any(t -> occursin("2/3 reject", t),
                      panel_titles(plot_result(nr).html))
            # break date + stationarity status in subtitles
            @test any(t -> occursin("break=45", t),
                      panel_titles(plot_result(fb).html))
            @test any(t -> occursin("Pesaran CD", t),
                      panel_titles(plot_result(pt).html))
            @test any(t -> occursin("NON-STATIONARY", t),
                      panel_titles(plot_result(vs_bad).html))
            # integration: nested-model, group, residual, and VAR checks render
            ya = randn(Xoshiro(850), 200)
            a2 = estimate_ar(ya, 2; method=:mle)
            a4 = estimate_ar(ya, 4; method=:mle)
            check_plot(plot_result(lr_test(a2, a4)))
            check_plot(plot_result(lm_test(a2, a4)))
            check_plot(plot_result(equality_test(ya, repeat([1, 2, 3, 4]; inner=50); test=:anova)))
            check_plot(plot_result(jarque_bera_test(reshape(ya, 200, 1))))
            xc = cumsum(randn(Xoshiro(851), 100))
            yc = 1.5 .* xc .+ randn(Xoshiro(852), 100)
            check_plot(plot_result(hansen_instability_test(estimate_cointreg(yc, xc))))
            check_plot(plot_result(is_stationary(estimate_var(randn(Xoshiro(853), 100, 2), 1))))
            dk = load_example(:denmark)
            dky = Matrix(dk[:, ["LRM", "LRY", "IBO", "IDE"]])
            vm = estimate_vecm(dky, 2; rank=1, deterministic=:constant,
                               varnames=["LRM", "LRY", "IBO", "IDE"])
            vbr = test_beta_restriction(vm, Float64[1 0 0; 0 1 0; 0 0 1; 0 0 -1])
            check_plot(plot_result(vbr))
            @test any(t -> occursin("rank=1", t),
                      panel_titles(plot_result(vbr).html))
        end
    end

    # =========================================================================
    # PLT-31 — set-identified SVAR IRFs
    # =========================================================================
    @testset "PLT-31 set-ID SVAR" begin
        H = 8; n = 2; nd = 60
        draws = randn(Xoshiro(3), nd, H, n, n) .* 0.5
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

        @testset "SID-17 view=:joint and view=:median_target" begin
            p0 = plot_result(sis; view=:default)
            check_plot(p0)
            pj = plot_result(sis; view=:joint)
            check_plot(pj); assert_all_json_valid(pj)
            @test occursin("joint", lowercase(pj.html))
            @test occursin("68% joint", pj.html)
            @test !occursin("16–84%", pj.html)
            @test !occursin("16-84%", pj.html)
            pj90 = plot_result(sis; view=:joint, level=0.90)
            check_plot(pj90)
            @test occursin("90% joint", pj90.html)
            @test !occursin("16–84%", pj90.html)
            pmt = plot_result(sis; view=:median_target)
            check_plot(pmt); assert_all_json_valid(pmt)
            @test occursin("Median-target", pmt.html) || occursin("median-target", lowercase(pmt.html))
            @test !occursin("\"lo_key\"", pmt.html)
            @test_throws ArgumentError plot_result(sis; view=:nope)
        end
    end
end
