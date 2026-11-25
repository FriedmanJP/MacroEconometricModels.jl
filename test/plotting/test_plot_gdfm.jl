# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.
#
# GDFM-08 (#900): Plotting tests for all 10 GDFM-series plot_result methods
# across spectral.jl, nowcast.jl, and teststat.jl.
# Uses shared plot_test_helpers.jl assertions (Testing Rules 1-7).

using Test
using Random
using MacroEconometricModels

isdefined(@__MODULE__, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

@testset "Plotting — GDFM Suite (#900)" begin

    # =========================================================================
    # Shared Small Fixtures (all RNG draws use explicit rng::Xoshiro)
    # =========================================================================
    rng = Random.Xoshiro(90001)
    d_stat = dgp_infinite_gdfm(rng; T_obs=120, N=12, q=2, a_coef=0.5, signal_share=0.75)
    d_i1 = dgp_infinite_gdfm(rng; T_obs=120, N=12, q=2, n_permanent=1, a_coef=0.4, signal_share=0.8)
    vnames = ["Series_$i" for i in 1:12]

    # 1. FHLZModel
    fhlz = estimate_gdfm_fhlz(d_stat.X, 2; p_max=3, n_perm=5, irf_horizon=16,
                              varnames=vnames, seed=42)

    # 2. CointegratedGDFM
    gdfm_i1 = estimate_gdfm_i1(d_i1.X, 2; n_permanent=1, p_max=2, n_perm=5,
                               irf_horizon=16, varnames=vnames, seed=42)

    # 3..5. GeneralizedDynamicFactorModel -> BandPassGDFMResult, EurocoinResult, SpectralFEVDResult
    gdfm = estimate_gdfm(d_stat.X, 2; r=3, varnames=vnames)
    bp = bandpass_gdfm(gdfm, (0.0, π / 4))
    target_vec = vec(d_stat.X[:, 1])
    ec = eurocoin_indicator(d_stat.X, target_vec, 2; cutoff=π / 4, r=3)
    sdfm = estimate_structural_dfm(gdfm; method=:fglr, r=3, p=1, H=16)
    sfevd = spectral_fevd(sdfm; include_idiosyncratic=true)
    sfevd_noid = spectral_fevd(sdfm; include_idiosyncratic=false)

    # 6. TimeVaryingGDFM
    tv = estimate_tv_gdfm(d_stat.X, 2; n_grid=8, varnames=vnames)

    # 7. GDFMNowcastResult
    X_rag = copy(d_stat.X)
    X_rag[end, 1:3] .= NaN
    X_rag[end - 1, 2] = NaN
    gnc = nowcast_gdfm(gdfm, X_rag, 0)
    gnc_fc = nowcast_gdfm(gdfm, X_rag, 2)
    gnc_bc = nowcast_gdfm(gdfm, X_rag, -1)
    gnc_1r = nowcast_gdfm(gdfm, X_rag[end:end, :], 0)

    # 8..10. Selection tests: OnatskiQTestResult, AhnHorensteinResult, OnatskiEDResult
    oq_rej = onatski_q_test(d_stat.X, 0, 4)
    oq_norej = onatski_q_test(d_stat.X, 2, 5)
    ah = ahn_horenstein(d_stat.X, 6)
    oed = onatski_ed(d_stat.X, 6)

    # =========================================================================
    # 1. FHLZModel
    # =========================================================================
    @testset "FHLZModel plot_result" begin
        for v in (:common, :innovations, :factors, :irf)
            p = plot_result(fhlz; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        @test occursin("Common Component", plot_result(fhlz; view=:common).html)
        @test occursin("Showing 4 of 12", plot_result(fhlz; view=:common).html)
        @test occursin("Spectral Variance Explained", plot_result(fhlz; view=:innovations).html)

        # Selection by Int, range, String; custom horizon/shock/title/ncols/save_path
        p_sub = plot_result(fhlz; view=:irf, vars=["Series_2", "Series_5"], shock=2, horizon=8, ncols=2, title="My FHLZ")
        check_plot(p_sub); assert_all_json_valid(p_sub)
        @test length(panel_titles(p_sub.html)) == 2
        @test occursin("My FHLZ", p_sub.html)
        @test occursin("Shock 2", p_sub.html)

        tmp = tempname() * ".html"
        @test plot_result(fhlz; save_path=tmp) isa PlotOutput
        @test isfile(tmp); rm(tmp; force=true)

        # Validation guards
        @test_throws ArgumentError plot_result(fhlz; view=:bogus)
        @test_throws ArgumentError plot_result(fhlz; view=:common, vars=["MissingVar"])
        @test_throws ArgumentError plot_result(fhlz; view=:common, vars=[99])
        @test_throws ArgumentError plot_result(fhlz; view=:irf, shock=99)
        @test_throws ArgumentError plot_result(fhlz; view=:irf, horizon=0)
        @test_throws ArgumentError plot_result(fhlz; view=:irf, horizon=999)

        # NaN -> null and hostile escaping
        fhlz_nan = deepcopy(fhlz)
        fhlz_nan.common_component[5, 1] = NaN
        assert_nan_becomes_null(plot_result(fhlz_nan; view=:common, vars=1:2))

        fhlz_h = FHLZModel{Float64}(
            fhlz.X, fhlz.q, fhlz.common_component, fhlz.idiosyncratic, fhlz.innovations,
            fhlz.impulse_filters, fhlz.p_blocks, fhlz.n_perm, fhlz.bandwidth,
            fhlz.kernel, fhlz.standardized, fhlz.variance_explained,
            vcat([HOSTILE_NAME], vnames[2:end]), fhlz.manifest,
        )
        assert_escapes(plot_result(fhlz_h; view=:common, vars=1:2))
    end

    # =========================================================================
    # 2. CointegratedGDFM
    # =========================================================================
    @testset "CointegratedGDFM plot_result" begin
        for v in (:common, :irf, :idiosyncratic)
            p = plot_result(gdfm_i1; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        @test occursin("Showing 4 of 12", plot_result(gdfm_i1; view=:common).html)
        p_str = plot_result(gdfm_i1; view=:irf, vars=["Series_1", "Series_3"],
                            shock=gdfm_i1.shock_names[2], ncols=2, title="I1 IRF")
        check_plot(p_str); assert_all_json_valid(p_str)
        @test length(panel_titles(p_str.html)) == 2
        @test occursin("I1 IRF", p_str.html)

        @test_throws ArgumentError plot_result(gdfm_i1; view=:bad)
        @test_throws ArgumentError plot_result(gdfm_i1; view=:irf, shock="UnknownShock")
        @test_throws ArgumentError plot_result(gdfm_i1; view=:irf, shock=99)
        @test_throws ArgumentError plot_result(gdfm_i1; view=:common, vars=[99])

        i1_nan = deepcopy(gdfm_i1)
        i1_nan.irf_matrices[3, 1, 1] = NaN
        assert_nan_becomes_null(plot_result(i1_nan; view=:irf, vars=1:2))

        i1_h = CointegratedGDFM{Float64}(
            gdfm_i1.X, gdfm_i1.common_levels, gdfm_i1.idiosyncratic, gdfm_i1.irf_matrices,
            gdfm_i1.long_run_impact, gdfm_i1.B0, gdfm_i1.shocks, gdfm_i1.trend_coef,
            gdfm_i1.p_blocks, gdfm_i1.q, gdfm_i1.n_permanent, gdfm_i1.n_perm,
            gdfm_i1.bandwidth, gdfm_i1.kernel, gdfm_i1.det_trend, gdfm_i1.identification,
            gdfm_i1.irf_horizon, vcat([HOSTILE_NAME], vnames[2:end]),
            gdfm_i1.shock_names, gdfm_i1.manifest,
        )
        assert_escapes(plot_result(i1_h; view=:common, vars=1:2))
    end

    # =========================================================================
    # 3. BandPassGDFMResult
    # =========================================================================
    @testset "BandPassGDFMResult plot_result" begin
        for v in (:comparison, :variance_share)
            p = plot_result(bp; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        @test length(panel_titles(plot_result(bp; view=:comparison).html)) == 4
        @test occursin("Showing 4 of 12", plot_result(bp; view=:comparison).html)
        p_vs = plot_result(bp; view=:variance_share, vars=["Series_2", "Series_4"], title="BP Share")
        check_plot(p_vs); assert_all_json_valid(p_vs)
        @test occursin("BP Share", p_vs.html)

        @test_throws ArgumentError plot_result(bp; view=:bogus)
        @test_throws ArgumentError plot_result(bp; view=:comparison, vars=["Missing"])

        bp_nan = BandPassGDFMResult{Float64}(
            let c = copy(bp.common_twosided); c[4, 1] = NaN; c end,
            bp.common_onesided, bp.variance_share, bp.band, bp.r, bp.varnames,
        )
        assert_nan_becomes_null(plot_result(bp_nan; view=:comparison, vars=1:2))

        bp_h = BandPassGDFMResult{Float64}(
            bp.common_twosided, bp.common_onesided, bp.variance_share, bp.band, bp.r,
            vcat([HOSTILE_NAME], vnames[2:end]),
        )
        assert_escapes(plot_result(bp_h; view=:variance_share))
    end

    # =========================================================================
    # 4. EurocoinResult
    # =========================================================================
    @testset "EurocoinResult plot_result" begin
        p = plot_result(ec; view=:indicator)
        check_plot(p); assert_all_json_valid(p)
        @test "Eurocoin indicator" in series_names(p.html)
        @test "Target low-pass" in series_names(p.html)
        @test occursin("Custom EC", plot_result(ec; title="Custom EC").html)
        @test_throws ArgumentError plot_result(ec; view=:bad)

        ec_nan = EurocoinResult{Float64}(
            let ind = copy(ec.indicator); ind[10] = NaN; ind end,
            ec.target_lowpass, ec.weights, ec.r2, ec.cutoff, ec.q, ec.r,
        )
        assert_nan_becomes_null(plot_result(ec_nan))
    end

    # =========================================================================
    # 5. SpectralFEVDResult
    # =========================================================================
    @testset "SpectralFEVDResult plot_result" begin
        for res in (sfevd, sfevd_noid)
            for md in (:stacked, :grouped)
                p = plot_result(res; mode=md)
                check_plot(p); assert_all_json_valid(p)
            end
        end
        @test "Idiosyncratic" in series_names(plot_result(sfevd).html)
        p_cap = plot_result(sfevd; max_vars=5)
        @test occursin("Showing 5 of 12", p_cap.html)
        @test_throws ArgumentError plot_result(sfevd; horizon=4)
        @test_throws ArgumentError plot_result(sfevd; mode=:pie)

        sfevd_nan = SpectralFEVDResult{Float64}(
            let sh = copy(sfevd.shares); sh[1, 1] = NaN; sh end,
            sfevd.power_band, sfevd.total_power_band, sfevd.band,
            sfevd.include_idiosyncratic, sfevd.varnames, sfevd.shock_names,
        )
        assert_nan_becomes_null(plot_result(sfevd_nan))

        sfevd_h = SpectralFEVDResult{Float64}(
            sfevd.shares, sfevd.power_band, sfevd.total_power_band, sfevd.band,
            sfevd.include_idiosyncratic, vcat([HOSTILE_NAME], vnames[2:end]),
            sfevd.shock_names,
        )
        assert_escapes(plot_result(sfevd_h))
    end

    # =========================================================================
    # 6. TimeVaryingGDFM
    # =========================================================================
    @testset "TimeVaryingGDFM plot_result" begin
        for v in (:heatmap, :variance, :connectedness)
            p = plot_result(tv; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        p_hm = plot_result(tv; view=:heatmap, freq_idx=3, ncols=2, title="TV Heatmap")
        check_plot(p_hm); assert_all_json_valid(p_hm)
        @test occursin("scaleType = \"sequential\"", p_hm.html)
        @test occursin("legend-gradient", p_hm.html)
        @test occursin("TV Heatmap", p_hm.html)
        @test length(panel_titles(p_hm.html)) == 2

        @test_throws ArgumentError plot_result(tv; view=:bogus)
        @test_throws ArgumentError plot_result(tv; view=:heatmap, freq_idx=0)
        @test_throws ArgumentError plot_result(tv; view=:heatmap, freq_idx=999)

        tv_nan = deepcopy(tv)
        tv_nan.eigenvalues_path[2, 1, 2] = NaN
        assert_nan_becomes_null(plot_result(tv_nan; view=:heatmap))
    end

    # =========================================================================
    # 7. GDFMNowcastResult
    # =========================================================================
    @testset "GDFMNowcastResult plot_result" begin
        for res in (gnc, gnc_fc, gnc_bc, gnc_1r)
            p = plot_result(res)
            check_plot(p); assert_all_json_valid(p)
        end
        # Regression check: default vars plots min(6, length(nowcast)) == 6 panels (not 1!)
        p_def = plot_result(gnc)
        @test length(panel_titles(p_def.html)) == 6
        @test occursin("Showing 6 of 12", p_def.html)
        @test occursin("95% CI", p_def.html)
        @test occursin("missing at T", p_def.html)

        p_sel = plot_result(gnc; vars=["Series_1", "Series_4"], ncols=2, title="Custom NC")
        check_plot(p_sel); assert_all_json_valid(p_sel)
        @test length(panel_titles(p_sel.html)) == 2
        @test occursin("Custom NC", p_sel.html)

        @test_throws ArgumentError plot_result(gnc; vars=["NoSuchVar"])
        @test_throws ArgumentError plot_result(gnc; vars=[99])

        gnc_nan = GDFMNowcastResult{Float64}(
            let nw = copy(gnc.nowcast); nw[1] = NaN; nw end,
            gnc.se, gnc.panel_imputed, gnc.horizon, gnc.n_lags,
            gnc.observed_mask, gnc.varnames,
        )
        assert_nan_becomes_null(plot_result(gnc_nan; vars=1:2))

        gnc_h = GDFMNowcastResult{Float64}(
            gnc.nowcast, gnc.se, gnc.panel_imputed, gnc.horizon, gnc.n_lags,
            gnc.observed_mask, vcat([HOSTILE_NAME], vnames[2:end]),
        )
        assert_escapes(plot_result(gnc_h; vars=1:2))
    end

    # =========================================================================
    # 8. OnatskiQTestResult
    # =========================================================================
    @testset "OnatskiQTestResult plot_result" begin
        for r in (oq_rej, oq_norej)
            p = plot_result(r)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("10% CV", p.html)
            @test occursin("5% CV", p.html)
            @test occursin("1% CV", p.html)
        end
        @test occursin("reject H", plot_result(oq_rej).html)
        @test occursin("fail to reject", plot_result(oq_norej).html)
        @test occursin("Custom OQ", plot_result(oq_rej; title="Custom OQ").html)

        oq_nan = OnatskiQTestResult{Float64}(
            NaN, oq_rej.pvalue, oq_rej.crit_vals, oq_rej.alpha_levels,
            oq_rej.k0, oq_rej.k1, oq_rej.n_obs, oq_rej.n_vars,
            oq_rej.bandwidth, oq_rej.n_blocks, oq_rej.rejected_05,
        )
        assert_nan_becomes_null(plot_result(oq_nan))
    end

    # =========================================================================
    # 9. AhnHorensteinResult
    # =========================================================================
    @testset "AhnHorensteinResult plot_result" begin
        for v in (:ratios, :eigenvalues)
            p = plot_result(ah; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        pr = plot_result(ah; view=:ratios, ncols=1, title="AH Custom")
        check_plot(pr); assert_all_json_valid(pr)
        @test length(panel_titles(pr.html)) == 2
        @test occursin("ER(k)", pr.html)
        @test occursin("GR(k)", pr.html)
        @test occursin("AH Custom", pr.html)
        @test_throws ArgumentError plot_result(ah; view=:bogus)

        ah_nan = AhnHorensteinResult{Float64}(
            ah.eigenvalues, ah.mock_eigenvalue,
            let er = copy(ah.er); er[2] = NaN; er end,
            ah.gr, ah.r_er, ah.r_gr, ah.k_max, ah.n_obs, ah.n_vars,
        )
        assert_nan_becomes_null(plot_result(ah_nan; view=:ratios))
    end

    # =========================================================================
    # 10. OnatskiEDResult
    # =========================================================================
    @testset "OnatskiEDResult plot_result" begin
        p = plot_result(oed)
        check_plot(p); assert_all_json_valid(p)
        @test length(panel_titles(p.html)) == 2
        @test occursin("Scree Eigenvalues", p.html)
        @test occursin("Eigenvalue Gaps", p.html)
        @test occursin("Custom ED", plot_result(oed; ncols=1, title="Custom ED").html)

        # Edge cases: r_ed = 0 and single-eigenvalue result
        oed0 = OnatskiEDResult{Float64}(oed.eigenvalues, 0, oed.delta, 1, false,
                                        oed.k_max, oed.n_obs, oed.n_vars)
        check_plot(plot_result(oed0)); assert_all_json_valid(plot_result(oed0))

        oed_1 = OnatskiEDResult{Float64}([5.0], 0, 0.1, 1, true, 1, 50, 1)
        check_plot(plot_result(oed_1)); assert_all_json_valid(plot_result(oed_1))

        oed_nan = OnatskiEDResult{Float64}(
            let ev = copy(oed.eigenvalues); ev[2] = NaN; ev end,
            oed.r_ed, oed.delta, oed.iterations, oed.converged,
            oed.k_max, oed.n_obs, oed.n_vars,
        )
        assert_nan_becomes_null(plot_result(oed_nan))
    end
end
