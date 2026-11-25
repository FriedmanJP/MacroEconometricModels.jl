# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

using Test, Random, LinearAlgebra, Statistics, DataFrames, StatsAPI
using MacroEconometricModels

isdefined(@__MODULE__, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

@testset "Plotting — QFM, QVAR, QFAVAR & Growth-at-Risk (#891, #892)" begin
    rng = Random.Xoshiro(89100)

    # Shared synthetic fixtures (small & fast, deterministic via explicit rng)
    T_obs, N = 80, 18
    F_raw = randn(rng, T_obs, 2)
    L_raw = randn(rng, N, 2)
    e_raw = 0.35 .* randn(rng, T_obs, N)
    X = F_raw * L_raw' .+ e_raw
    vnames = ["S_$i" for i in 1:N]

    @testset "Single-positional-arg signature contract (plotrule / plotting.md)" begin
        qtypes = (
            QuantileFactorModel,
            QuantileFactorGrid,
            QFMICResult,
            QFMSpanningResult,
            QFMLoadingConstancyTest,
            QuantileFactorPanelModel,
            GrowthAtRiskResult,
            QuantileImpulseResponse,
        )
        for QT in qtypes
            ms = collect(methods(plot_result, Tuple{QT{Float64}}))
            @test length(ms) == 1
            @test length(Base.unwrap_unionall(ms[1].sig).parameters) == 2
        end
        @test all(m -> length(Base.unwrap_unionall(m.sig).parameters) == 2, methods(plot_result))
    end

    @testset "QuantileFactorModel" begin
        qfm = estimate_qfm(X, 2, 0.25; varnames=vnames, se=:none, seed=11)
        qfm_se = estimate_qfm(X, 2, 0.25; varnames=vnames, se=:asymptotic, hac_bandwidth=0, seed=11)

        for mod in (qfm, qfm_se), v in (:factors, :loadings, :variance, :scree)
            p = plot_result(mod; view=v)
            check_plot(p); assert_all_json_valid(p)
        end

        # var selection by Int and String + bounds checking
        p_f_v = plot_result(qfm; view=:factors, var="S_3")
        check_plot(p_f_v); assert_all_json_valid(p_f_v)
        @test occursin("S_3", p_f_v.html)

        p_l_i = plot_result(qfm_se; view=:loadings, var=2)
        p_l_s = plot_result(qfm_se; view=:loadings, var="S_2")
        check_plot(p_l_i); check_plot(p_l_s)
        @test panel_titles(p_l_i.html) == panel_titles(p_l_s.html)
        @test_throws ArgumentError plot_result(qfm; view=:loadings, var=0)
        @test_throws ArgumentError plot_result(qfm; view=:loadings, var=N + 1)
        @test_throws ArgumentError plot_result(qfm; view=:loadings, var="NOPE")
        @test_throws ArgumentError plot_result(qfm; view=:bogus)

        # C7 caps
        p_cap = plot_result(qfm; view=:loadings, max_factors=1, max_vars=5)
        check_plot(p_cap); assert_all_json_valid(p_cap)
        @test occursin("Showing 5 of $N", p_cap.html)
        @test occursin("Showing 1 of 2", p_cap.html)

        # title, ncols, save_path
        tmp = tempname() * ".html"
        p_sv = plot_result(qfm; view=:variance, ncols=1, title="QFM Custom", save_path=tmp)
        @test p_sv isa PlotOutput && isfile(tmp)
        @test occursin("QFM Custom", p_sv.html)
        rm(tmp; force=true)

        # NaN -> null & hostile variable name escaping
        fac_nan = copy(qfm.factors); fac_nan[3, 1] = NaN
        qfm_nan = QuantileFactorModel(
            X=qfm.X, tau=qfm.tau, factors=fac_nan, loadings=qfm.loadings,
            sigma_sq=qfm.sigma_sq, common_component=qfm.common_component,
            residuals=qfm.residuals, objective=qfm.objective, pseudo_r2=qfm.pseudo_r2,
            r=qfm.r, varnames=vcat([HOSTILE_NAME], vnames[2:end]),
        )
        assert_nan_becomes_null(plot_result(qfm_nan; view=:factors))
        assert_escapes(plot_result(qfm_nan; view=:loadings, var=1))
        assert_escapes(plot_result(qfm_nan; view=:variance))
    end

    @testset "QuantileFactorGrid" begin
        grid = estimate_qfm(X, [2, 1, 2], [0.9, 0.5, 0.1]; varnames=vnames, seed=21)
        for v in (:factors, :loadings, :ic, :rank)
            p = plot_result(grid; view=v)
            check_plot(p); assert_all_json_valid(p)
        end

        # factor=2 handles quantile-varying rank [2, 1, 2] where tau=0.5 has r=1
        p_f2 = plot_result(grid; view=:factors, factor=2)
        check_plot(p_f2); assert_all_json_valid(p_f2)

        # Int vs String var selection + validation
        p_li = plot_result(grid; view=:loadings, var=3)
        p_ls = plot_result(grid; view=:loadings, var="S_3")
        @test panel_titles(p_li.html) == panel_titles(p_ls.html)
        @test_throws ArgumentError plot_result(grid; view=:bogus)
        @test_throws ArgumentError plot_result(grid; factor=0)
        @test_throws ArgumentError plot_result(grid; factor=99)
        @test_throws ArgumentError plot_result(grid; view=:loadings, var=99)
        @test_throws ArgumentError plot_result(grid; view=:loadings, var="NOPE")

        # C7 cap on max_taus and max_factors
        p_cap = plot_result(grid; view=:factors, max_taus=2)
        check_plot(p_cap); assert_all_json_valid(p_cap)
        @test occursin("Showing 2 of 3", p_cap.html)

        # Single-tau grid + SE bands on loadings
        grid_1 = estimate_qfm(X, 2, [0.5]; varnames=vnames, se=:asymptotic, hac_bandwidth=0, seed=22)
        for v in (:factors, :loadings, :ic)
            check_plot(plot_result(grid_1; view=v))
        end

        # Hostile varnames & NaN -> null
        grid_se = estimate_qfm(X, 2, [0.1, 0.9]; varnames=vcat([HOSTILE_NAME], vnames[2:end]),
                               se=:asymptotic, hac_bandwidth=0, seed=23)
        p_h = plot_result(grid_se; view=:loadings, var=1)
        check_plot(p_h); assert_escapes(p_h)

        # save_path & title
        tmp = tempname() * ".html"
        pg = plot_result(grid; view=:ic, ncols=1, title="Grid IC", save_path=tmp)
        @test isfile(tmp) && occursin("Grid IC", pg.html)
        rm(tmp; force=true)
    end

    @testset "QFMICResult" begin
        ic = ic_criteria_qfm(X, 4, 0.25; warn_boundary=false, seed=31)
        for v in (:ic, :scree, :loss)
            p = plot_result(ic; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        @test_throws ArgumentError plot_result(ic; view=:bogus)

        tmp = tempname() * ".html"
        pic = plot_result(ic; view=:ic, ncols=2, title="Custom QFM IC", save_path=tmp)
        @test isfile(tmp) && occursin("Custom QFM IC", pic.html)
        rm(tmp; force=true)

        ic_nan = QFMICResult{Float64}(
            ic.tau, ic.r_max, vcat([NaN], ic.sigma_sq[2:end]), ic.threshold_PNT, ic.r_rank,
            ic.loss, ic.IC1, ic.IC2, ic.IC3, ic.r_IC1, ic.r_IC2, ic.r_IC3,
        )
        assert_nan_becomes_null(plot_result(ic_nan; view=:scree))
    end

    @testset "QFMSpanningResult" begin
        grid = estimate_qfm(X, [2, 1, 2], [0.1, 0.5, 0.9]; seed=41)
        sp = test_qfm_spanning(grid, X; r_pca=2)
        p = plot_result(sp)
        check_plot(p); assert_all_json_valid(p)

        # Single-tau spanning + max_bars C7 cap
        sp1 = test_qfm_spanning(grid[1], X; r_pca=2)
        check_plot(plot_result(sp1)); assert_all_json_valid(plot_result(sp1))
        p_cap = plot_result(sp; max_bars=3)
        check_plot(p_cap); assert_all_json_valid(p_cap)
        @test occursin("Showing 3 of 5", p_cap.html)

        # Empty cancor edge case & NaN -> null
        sp_nan = QFMSpanningResult{Float64}(
            sp.taus, sp.r_qfm, 0,
            [[NaN, 0.2], sp.ols_r2[2], sp.ols_r2[3]],
            sp.quantile_r1, [Float64[], Float64[], Float64[]],
            sp.unspanned, sp.threshold, sp.n_obs, sp.n_vars,
        )
        assert_nan_becomes_null(plot_result(sp_nan))
    end

    @testset "QFMLoadingConstancyTest" begin
        grid = estimate_qfm(X, 2, [0.1, 0.5]; seed=51)
        ct = test_qfm_loading_constancy(grid, 0.1, 0.5; n_boot=19, threaded=false, seed=51)
        p = plot_result(ct)
        check_plot(p); assert_all_json_valid(p)

        # Zero-spread bootstrap draws (tau1 == tau2)
        ct_same = test_qfm_loading_constancy(grid, 0.5, 0.5; n_boot=10, seed=51)
        p_same = plot_result(ct_same)
        check_plot(p_same); assert_all_json_valid(p_same)

        # Empty boot_stats fallback
        ct_empty = QFMLoadingConstancyTest{Float64}(
            0.1, 0.5, 2, 2, 0.42, 0.15, Float64[], 0, 4, :block, T_obs, N, 10, nothing,
        )
        p_emp = plot_result(ct_empty; title="Empty Boot", ncols=1)
        check_plot(p_emp); assert_all_json_valid(p_emp)
        @test occursin("Empty Boot", p_emp.html)
    end

    @testset "QuantileFactorPanelModel" begin
        N_p, T_p, p_k = 20, 30, 2
        f_p = randn(rng, T_p, 1)
        l_p = 0.5 .+ rand(rng, N_p, 1)
        lf = l_p * f_p'
        W_arr = Array{Float64,3}(undef, N_p, T_p, p_k)
        W_arr[:, :, 1] = 0.8 .* lf .+ randn(rng, N_p, T_p)
        W_arr[:, :, 2] = 0.3 .* lf .+ randn(rng, N_p, T_p)
        Y_mat = 1.2 .* W_arr[:, :, 1] .- 0.6 .* W_arr[:, :, 2] .+ lf .+ 0.4 .* randn(rng, N_p, T_p)

        m_hom = estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5;
                                        varnames=[HOSTILE_NAME, "x2"], seed=61)
        m_het = estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5;
                                        slope=:heterogeneous, varnames=["x1", "x2"], seed=61)

        for mod in (m_hom, m_het), v in (:coef, :factors, :loadings)
            p = plot_result(mod; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        assert_escapes(plot_result(m_hom; view=:coef))
        @test_throws ArgumentError plot_result(m_hom; view=:bogus)
        @test_throws ArgumentError plot_result(m_hom; conf_level=1.5)

        # C7 cap on :loadings
        p_cap = plot_result(m_hom; view=:loadings, max_vars=5)
        check_plot(p_cap); assert_all_json_valid(p_cap)
        @test occursin("Showing 5 of $N_p", p_cap.html)
    end

    @testset "GrowthAtRiskResult" begin
        taus_g = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        vnames_g = vcat([HOSTILE_NAME], ["G_$i" for i in 2:N])
        gar_st = growth_at_risk(X, 1, 1; r=2, taus=taus_g, density=:skewed_t,
                                varnames=vnames_g, seed=71)
        gar_nm = growth_at_risk(X, 1, 1; r=2, taus=[0.1, 0.5, 0.9], density=:normal, seed=72)
        gar_no = growth_at_risk(X, 1, 1; r=2, taus=[0.25], density=:none, seed=73)

        for gr in (gar_st, gar_nm, gar_no), v in (:fan, :gar, :density)
            p = plot_result(gr; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        check_plot(plot_result(gar_st; view=:fan, quantile_idx=2))
        check_plot(plot_result(gar_st; view=:density, quantile_idx=2))
        assert_escapes(plot_result(gar_st; view=:fan))
        assert_escapes(plot_result(gar_st; view=:density))

        @test_throws ArgumentError plot_result(gar_st; view=:bogus)
        @test_throws ArgumentError plot_result(gar_st; quantile_idx=99)

        # NaN -> null
        act_nan = copy(gar_st.actual_target); act_nan[4] = NaN
        gar_nan = GrowthAtRiskResult{Float64}(
            gar_st.y, gar_st.target_name, gar_st.taus, gar_st.horizon, gar_st.lags,
            gar_st.alpha, gar_st.density, gar_st.rearrange, gar_st.fitted_quantiles,
            act_nan, gar_st.oos_quantiles, gar_st.coefficients, gar_st.density_params,
            gar_st.oos_density_params, gar_st.gar, gar_st.expected_shortfall,
            gar_st.upside_entropy, gar_st.downside_entropy, gar_st.oos_gar,
            gar_st.oos_expected_shortfall, gar_st.oos_upside_entropy,
            gar_st.oos_downside_entropy, gar_st.quantile_score, gar_st.crps,
            gar_st.oos_split, gar_st.pit, gar_st.oos_quantile_scores, gar_st.oos_crps,
            gar_st.n_obs, gar_st.n_eff, gar_st.manifest,
        )
        assert_nan_becomes_null(plot_result(gar_nan; view=:fan))
    end

    @testset "QuantileImpulseResponse" begin
        Y_qv = randn(rng, 100, 3)
        qv = estimate_qvar(Y_qv, 1, [0.10, 0.50, 0.90];
                           varnames=[HOSTILE_NAME, "CPI", "FFR"], se=:none, seed=81)
        ir_none = qirf(qv, 6; tau_profile=0.10)
        ir_boot = qirf(qv, 6; tau_profile=[0.10, 0.25], ci_type=:bootstrap, n_boot=15, seed=82)

        for ir in (ir_none, ir_boot)
            p = plot_result(ir)
            check_plot(p); assert_all_json_valid(p)
        end

        # Int & String selection on var and shock + bounds errors
        p_is = plot_result(ir_boot; var=2, shock=3)
        p_ss = plot_result(ir_boot; var="CPI", shock="FFR")
        @test panel_titles(p_is.html) == panel_titles(p_ss.html)
        @test length(panel_titles(p_is.html)) == 1
        @test_throws ArgumentError plot_result(ir_boot; var=0)
        @test_throws ArgumentError plot_result(ir_boot; var="NOPE")
        @test_throws ArgumentError plot_result(ir_boot; shock=99)
        @test_throws ArgumentError plot_result(ir_boot; shock="NOPE")

        # Escaping & NaN -> null
        assert_escapes(plot_result(ir_none))
        vals_nan = copy(ir_none.values); vals_nan[2, 1, 1] = NaN
        ir_nan = QuantileImpulseResponse{Float64}(
            vals_nan, ir_none.ci_lower, ir_none.ci_upper, ir_none.horizon,
            ir_none.tau_profile, ir_none.baseline_tau, ir_none.shock_size,
            ir_none.method, ir_none.ci_type, ir_none.conf_level,
            ir_none.variables, ir_none.shocks,
        )
        assert_nan_becomes_null(plot_result(ir_nan; var=1, shock=1))

        # Wide QFAVAR panel IRF hits C7 max_vars cap cleanly
        dq = dgp_qfavar(rng; T_obs=90, N=18, r=1, p=1)
        qf = estimate_qfavar(dq.X_panel, 1:dq.n_key, dq.r, dq.p, [0.10, 0.50, 0.90];
                             method=:twostep, seed=83)
        pir = qfavar_panel_irf(qf, 5; shock=2, tau_profile=0.10)
        p_wide = plot_result(pir; max_vars=6)
        check_plot(p_wide); assert_all_json_valid(p_wide)
        @test occursin("Showing 6 of 18", p_wide.html)
        @test length(panel_titles(p_wide.html)) == 6
    end
end
