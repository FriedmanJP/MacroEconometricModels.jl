# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
CI-default policy audit, static drift gate, monotonicity battery, and regression
tests for the 68% macro (incl. LP-DiD) / 90% micro / 5% hypothesis-test standard
(issues #936-#942).
"""

using Test
using MacroEconometricModels
using DataFrames
using Distributions
using LinearAlgebra
using Random
using Statistics
using StatsAPI

const MEM = MacroEconometricModels

@testset "CI Defaults & Universal Configurability Audit (#936-#942)" begin

    # =========================================================================
    # 1. Core Constants, Validators & Formatters (CI-01)
    # =========================================================================
    @testset "Policy constants and validators" begin
        @test DEFAULT_CI_MACRO == 0.68
        @test DEFAULT_CI_MICRO == 0.90
        @test DEFAULT_QUANTILES_MACRO == [0.16, 0.5, 0.84]
        @test DEFAULT_QUANTILES_MICRO == [0.05, 0.5, 0.95]

        @test MEM._validate_conf_level(0.68) === 0.68
        @test isapprox(MEM._validate_conf_level(Float32(0.90)), 0.90; atol=1e-6)
        for bad in (0.0, 1.0, -0.1, 1.2, NaN, Inf, -Inf, true, false)
            @test_throws ArgumentError MEM._validate_conf_level(bad)
            @test_throws ArgumentError MEM._conf_z(bad)
        end

        @test MEM._validate_quantiles([0.16, 0.5, 0.84]) == [0.16, 0.5, 0.84]
        @test_throws ArgumentError MEM._validate_quantiles(Float64[])
        @test_throws ArgumentError MEM._validate_quantiles([0.0, 0.5])
        @test_throws ArgumentError MEM._validate_quantiles([0.5, 1.0])
        @test_throws ArgumentError MEM._validate_quantiles([true])

        @test MEM._conf_level_to_quantiles(0.68) ≈ [0.16, 0.5, 0.84] atol=1e-12
        @test MEM._conf_level_to_quantiles(0.90) ≈ [0.05, 0.5, 0.95] atol=1e-12
        @test MEM._conf_level_to_quantiles(0.95) ≈ [0.025, 0.5, 0.975] atol=1e-12

        @test MEM._ci_pct(0.68) == "68"
        @test MEM._ci_pct(0.90) == "90"
        @test MEM._ci_pct(0.95) == "95"
        @test MEM._ci_pct(0.975) == "97.5"
        @test MEM._quantile_pct_label(0.16) == "16%"
        @test MEM._quantile_pct_label(0.025) == "2.5%"
        @test MEM._quantile_pct_label(0.975) == "97.5%"
    end

    # =========================================================================
    # 2. Static Literal Drift Gate (CI-06)
    # =========================================================================
    @testset "Static literal drift gate across src/" begin
        src_root = normpath(joinpath(@__DIR__, "..", "src"))

        # Allowlisted files where 1.96 appears only in tabulated critical values or prose
        allow_196 = Set([
            "teststat/critical_values.jl",
            "teststat/ips.jl",
            "factor/selection.jl",
            "dsge/mcmc_diagnostics.jl",
        ])

        # Allowlisted files where 0.025 / 0.975 appear as tabulated test CVs,
        # depreciation delta = 0.025, SV prior b0 = 0.025, seasonal filter weights,
        # or docstring examples in ci_defaults.jl.
        allow_025_975 = Set([
            "teststat/critical_values.jl",
            "teststat/ips.jl",
            "teststat/helpers.jl",
            "teststat/bai_perron.jl",
            "ardl/bounds.jl",
            "ardl/types.jl",
            "reg/stability.jl",
            "sv/estimation.jl",
            "x13/seasonalma.jl",
            "core/ci_defaults.jl",
            "dsge/heterogeneous/reiter.jl",
            "dsge/heterogeneous/two_asset_ge.jl",
            "dsge/heterogeneous/krusell_smith.jl",
            "dsge/heterogeneous/parser.jl",
            "dsge/heterogeneous/examples.jl",
            "dsge/heterogeneous/types.jl",
            "dsge/heterogeneous/ssj.jl",
            "dsge/heterogeneous/steady_state.jl",
        ])

        offenders_196 = String[]
        offenders_025 = String[]
        offenders_095 = String[]
        re_095_kw = r"(?:conf_level|ci_level|\blevel|\bci)\s*(?:::[^=]+)?=\s*0\.95\b"

        for (root, _, files) in walkdir(src_root)
            for f in files
                endswith(f, ".jl") || continue
                full = joinpath(root, f)
                rel = replace(relpath(full, src_root), '\\' => '/')
                for (ln, line) in enumerate(eachline(full))
                    stripped = strip(line)
                    startswith(stripped, '#') && continue
                    if occursin(r"\b1\.96\b", line) && !(rel in allow_196)
                        push!(offenders_196, "$rel:$ln: $stripped")
                    end
                    if occursin(r"\b(?:0\.025|0\.975)\b", line) && !(rel in allow_025_975)
                        push!(offenders_025, "$rel:$ln: $stripped")
                    end
                    if occursin(re_095_kw, line) && rel != "core/tables.jl"
                        push!(offenders_095, "$rel:$ln: $stripped")
                    end
                end
            end
        end

        @test isempty(offenders_196)
        @test isempty(offenders_025)
        @test isempty(offenders_095)
    end

    # =========================================================================
    # 3. Empirical Bug Regressions (CI-01..CI-05)
    # =========================================================================
    @testset "Bug regression: Hansen (2000) closed-form critical values & SETAR" begin
        @test MEM._hansen2000_crit(0.68) ≈ -2.0 * log(1.0 - sqrt(0.68)) atol=1e-12
        @test isapprox(MEM._hansen2000_crit(0.68), 3.4816; atol=1e-3)
        @test isapprox(MEM._hansen2000_crit(0.90), 5.9390; atol=1e-2)
        @test isapprox(MEM._hansen2000_crit(0.95), 7.3552; atol=1e-2)
        @test isapprox(MEM._hansen2000_crit(0.99), 10.589; atol=1e-2)
        @test MEM._hansen2000_crit(0.68) < MEM._hansen2000_crit(0.90) < MEM._hansen2000_crit(0.95)
        @test_throws ArgumentError MEM._hansen2000_crit(0.0)
        @test_throws ArgumentError MEM._hansen2000_crit(1.0)
        @test_throws ArgumentError MEM._hansen2000_crit(true)

        rng = Xoshiro(94201)
        n = 160
        y = zeros(n)
        for t in 2:n
            y[t] = (y[t - 1] <= 0.0 ? 0.3 : -0.4) * y[t - 1] + 0.5 * randn(rng)
        end
        m68 = estimate_setar(y, 1, 1; linearity=false)
        @test m68.gamma_ci_level == 0.68
        @test default_ci_level(m68) == 0.68
        m90 = estimate_setar(y, 1, 1; linearity=false, ci_level=0.90)
        m95 = estimate_setar(y, 1, 1; linearity=false, ci_level=0.95)
        @test (m68.gamma_ci[2] - m68.gamma_ci[1]) <= (m90.gamma_ci[2] - m90.gamma_ci[1]) <=
              (m95.gamma_ci[2] - m95.gamma_ci[1])

        ci_def = confint(m68)
        ci_95 = confint(m68; level=0.95)
        @test all((ci_def[:, 2] .- ci_def[:, 1]) .< (ci_95[:, 2] .- ci_95[:, 1]))
        fc = forecast(m68, 3; reps=40, rng=Xoshiro(94202))
        @test fc.conf_level == 0.68
    end

    @testset "Bug regression: degenerate ACF respects conf_level" begin
        r68 = acf(ones(50))
        @test r68.ci ≈ quantile(Normal(), 0.84) / sqrt(50) atol=1e-12
        r95 = acf(ones(50); conf_level=0.95)
        @test r95.ci ≈ quantile(Normal(), 0.975) / sqrt(50) atol=1e-12
        @test r68.ci < r95.ci
        @test occursin("CI (68%)", sprint(show, r68))
        @test occursin("CI (95%)", sprint(show, r95))
        @test_throws ArgumentError acf(ones(50); conf_level=1.0)
        @test_throws ArgumentError acf(ones(50); conf_level=true)
    end

    @testset "Bug regression: QuantileRegModel alpha, DataFrame, confint" begin
        rng = Xoshiro(94203)
        n = 120
        X = hcat(ones(n), randn(rng, n, 2))
        y = 1.0 .+ 2.0 .* X[:, 2] .- 1.5 .* X[:, 3] .+ randn(rng, n)
        q_def = estimate_qreg(y, X, 0.5)
        @test q_def.alpha ≈ 0.10
        @test default_ci_level(q_def) ≈ 0.90
        df_def = DataFrame(q_def)
        ci_90 = confint(q_def; level=0.90)
        @test df_def.ci_lower ≈ ci_90[:, 1]
        @test df_def.ci_upper ≈ ci_90[:, 2]

        q_95 = estimate_qreg(y, X, 0.5; alpha=0.05)
        @test q_95.alpha ≈ 0.05
        @test default_ci_level(q_95) ≈ 0.95
        df_95 = DataFrame(q_95)
        ci_95 = confint(q_95; level=0.95)
        @test df_95.ci_lower ≈ ci_95[:, 1]
        @test df_95.ci_upper ≈ ci_95[:, 2]
        @test all((df_def.ci_upper .- df_def.ci_lower) .< (df_95.ci_upper .- df_95.ci_lower))
        @test occursin("90% CI Low", sprint(io -> report(io, q_def)))
        @test occursin("95% CI Low", sprint(io -> report(io, q_95)))
    end

    @testset "Bug regression: DSGEVARPosterior Tables.jl, coef_table & uncertainty_bounds" begin
        rng = Xoshiro(94204)
        sys = ABCDSystem([0.6 0.0; 0.0 0.5], [1.0 0.0; 0.0 1.0],
                         [1.0 0.2; 0.0 0.8], [1.0 0.1; 0.0 1.0])
        Y = randn(rng, 80, 2)
        post = estimate_dsge_var(sys, Y, 1; lambda=1.0, n_draws=60, rng=rng)
        @test default_ci_level(post) == 0.68

        df68 = DataFrame(post)
        df95 = DataFrame(coef_table(post; conf_level=0.95))
        @test :ci_lower in propertynames(df68)
        @test :ci_upper in propertynames(df68)
        @test all((df68.ci_upper .- df68.ci_lower) .< (df95.ci_upper .- df95.ci_lower))

        lo68, hi68 = uncertainty_bounds(post)
        lo95, hi95 = uncertainty_bounds(post; conf_level=0.95)
        @test all((hi68 .- lo68) .< (hi95 .- lo95))

        s68 = sprint(show, post)
        @test occursin("16%", s68) && occursin("84%", s68)
        s95 = sprint(io -> report(io, post; conf_level=0.95))
        @test occursin("2.5%", s95) && occursin("97.5%", s95)
    end

    # =========================================================================
    # 4. Macro vs Micro Defaults & Monotonicity Battery (CI-02, CI-03, CI-04)
    # =========================================================================
    @testset "Macro 68% defaults & monotonicity (VAR, LP, LP-DiD, BVAR, ARIMA, Spectral, Bai-Perron)" begin
        rng = Xoshiro(94205)
        Y = zeros(120, 2)
        for t in 2:120
            Y[t, 1] = 0.5 * Y[t - 1, 1] + 0.1 * Y[t - 1, 2] + randn(rng)
            Y[t, 2] = 0.2 * Y[t - 1, 1] + 0.4 * Y[t - 1, 2] + randn(rng)
        end

        # VARModel
        m_var = estimate_var(Y, 1)
        @test default_ci_level(m_var) == 0.68
        w68 = confint(m_var)[:, 2] .- confint(m_var)[:, 1]
        w90 = confint(m_var; level=0.90)[:, 2] .- confint(m_var; level=0.90)[:, 1]
        w95 = confint(m_var; level=0.95)[:, 2] .- confint(m_var; level=0.95)[:, 1]
        @test all(w68 .< w90 .< w95)
        df_v68 = DataFrame(m_var)
        df_v68_exp = DataFrame(coef_table(m_var; conf_level=0.68))
        df_v90 = DataFrame(coef_table(m_var; conf_level=0.90))
        df_v95 = DataFrame(coef_table(m_var; conf_level=0.95))
        @test df_v68.ci_lower ≈ df_v68_exp.ci_lower
        @test df_v68.ci_upper ≈ df_v68_exp.ci_upper
        @test all((df_v68.ci_upper .- df_v68.ci_lower) .<
                  (df_v90.ci_upper .- df_v90.ci_lower) .<
                  (df_v95.ci_upper .- df_v95.ci_lower))
        @test occursin("68% CI Low", sprint(io -> report(io, m_var)))

        fc68 = forecast(m_var, 4; ci_method=:analytic)
        fc90 = forecast(m_var, 4; ci_method=:analytic, conf_level=0.90)
        fc95 = forecast(m_var, 4; ci_method=:analytic, conf_level=0.95)
        @test fc68.conf_level == 0.68
        @test all((fc68.ci_upper .- fc68.ci_lower) .< (fc90.ci_upper .- fc90.ci_lower) .<
                  (fc95.ci_upper .- fc95.ci_lower))

        # LP
        lp_m = estimate_lp(Y, 1, 4; lags=1)
        ir68 = lp_irf(lp_m)
        ir90 = lp_irf(lp_m; conf_level=0.90)
        ir95 = lp_irf(lp_m; conf_level=0.95)
        @test ir68.conf_level == 0.68
        nz_lp_ir = findall(>(1e-10), ir68.se)
        @test !isempty(nz_lp_ir)
        @test all((ir68.ci_upper[nz_lp_ir] .- ir68.ci_lower[nz_lp_ir]) .<
                  (ir90.ci_upper[nz_lp_ir] .- ir90.ci_lower[nz_lp_ir]) .<
                  (ir95.ci_upper[nz_lp_ir] .- ir95.ci_lower[nz_lp_ir]))
        @test occursin("68% CI excludes zero", sprint(show, lp_m))

        # BVAR
        bpost = estimate_bvar(Y, 1; n_draws=60, rng=Xoshiro(94206))
        bfc68 = forecast(bpost, 4; rng=Xoshiro(94207))
        @test bfc68.conf_level == 0.68
        sb68 = sprint(show, bpost)
        @test occursin("16%", sb68) && occursin("84%", sb68)
        sb95 = sprint(io -> report(io, bpost; conf_level=0.95))
        @test occursin("2.5%", sb95) && occursin("97.5%", sb95)

        # ARIMA
        m_ar = estimate_ar(Y[:, 1], 1)
        @test default_ci_level(m_ar) == 0.68
        af68 = forecast(m_ar, 4)
        af95 = forecast(m_ar, 4; conf_level=0.95)
        @test af68.conf_level == 0.68
        @test all((af68.ci_upper .- af68.ci_lower) .< (af95.ci_upper .- af95.ci_lower))

        # Spectral
        r_acf68 = acf(Y[:, 1]; lags=5)
        r_acf90 = acf(Y[:, 1]; lags=5, conf_level=0.90)
        r_acf95 = acf(Y[:, 1]; lags=5, conf_level=0.95)
        @test r_acf68.ci < r_acf90.ci < r_acf95.ci
        sp68 = spectral_density(Y[:, 1])
        sp95 = spectral_density(Y[:, 1]; conf_level=0.95)
        @test all((sp68.ci_upper .- sp68.ci_lower) .<= (sp95.ci_upper .- sp95.ci_lower))

        # Bai-Perron break-date CI (macro 68% default, alpha=0.05 sequential test unchanged)
        y_bp = vcat(fill(1.0, 60), fill(5.0, 60)) .+ 0.4 .* randn(Xoshiro(94208), 120)
        X_bp = ones(120, 1)
        bp68 = bai_perron_test(y_bp, X_bp; max_breaks=2)
        bp95 = bai_perron_test(y_bp, X_bp; max_breaks=2, conf_level=0.95)
        @test bp68.conf_level == 0.68
        @test bp95.conf_level == 0.95
        @test bp68.n_breaks == bp95.n_breaks
        if bp68.n_breaks > 0
            @test (bp68.break_cis[1][2] - bp68.break_cis[1][1]) <=
                  (bp95.break_cis[1][2] - bp95.break_cis[1][1])
            @test occursin("68% CI", sprint(show, bp68))
            @test occursin("95% CI", sprint(show, bp95))
        end
    end

    @testset "Micro 90% defaults & LP-DiD (68%) vs DiD (90%) split" begin
        rng = Xoshiro(94209)
        n = 120
        X = hcat(ones(n), randn(rng, n, 2))
        y = 0.5 .+ 1.2 .* X[:, 2] .- 0.8 .* X[:, 3] .+ randn(rng, n)

        # OLS
        m_ols = estimate_reg(y, X)
        @test default_ci_level(m_ols) == 0.90
        w68 = confint(m_ols; level=0.68)[:, 2] .- confint(m_ols; level=0.68)[:, 1]
        w90 = confint(m_ols)[:, 2] .- confint(m_ols)[:, 1]
        w95 = confint(m_ols; level=0.95)[:, 2] .- confint(m_ols; level=0.95)[:, 1]
        @test all(w68 .< w90 .< w95)
        @test DataFrame(m_ols).ci_lower ≈ confint(m_ols; level=0.90)[:, 1]
        @test DataFrame(coef_table(m_ols; conf_level=0.95)).ci_lower ≈ confint(m_ols; level=0.95)[:, 1]
        @test occursin("90% CI Low", sprint(show, m_ols))

        # Logit + MarginalEffects + OddsRatio
        yb = Float64.(y .> median(y))
        m_log = estimate_logit(yb, X)
        @test default_ci_level(m_log) == 0.90
        me = marginal_effects(m_log)
        @test me.conf_level == 0.90
        @test default_ci_level(me) == 0.90
        or_res = odds_ratio(m_log)
        @test or_res.conf_level == 0.90
        @test size(confint(me)) == (length(me.effects), 2)
        @test size(confint(or_res)) == (length(or_res.or), 2)

        # Threshold regression (cross-sectional -> 90% micro)
        q = randn(rng, n)
        m_th = estimate_threshold(y, X, q; linearity=false)
        @test m_th.gamma_ci_level == 0.90
        @test default_ci_level(m_th) == 0.90

        # RDD (90% micro + plot_result honors r.level)
        x_rdd = randn(rng, 200)
        y_rdd = 2.0 .* (x_rdd .>= 0.0) .+ 0.5 .* x_rdd .+ 0.5 .* randn(rng, 200)
        r_rdd = estimate_rdd(y_rdd, x_rdd; cutoff=0.0)
        @test r_rdd.level == 0.90
        @test default_ci_level(r_rdd) == 0.90
        @test size(confint(r_rdd)) == (2, 2)
        @test occursin("90% CI Low", sprint(io -> report(io, r_rdd)))

        # Wild cluster bootstrap (90% micro)
        clusters = repeat(1:15, inner=8)
        wcb = wild_cluster_bootstrap(m_ols, 2; clusters=clusters, n_boot=99, rng=Xoshiro(94210))
        @test wcb.level == 0.90

        # DiD (non-LP = 90% micro) vs LP-DiD (68% macro)
        N_u, T_u = 30, 10
        df_p = DataFrame(
            unit = repeat(1:N_u, inner=T_u),
            time = repeat(1:T_u, outer=N_u),
            treat_time = repeat(vcat(fill(6, 15), fill(0, 15)), inner=T_u),
        )
        df_p.treat = Float64.((df_p.treat_time .> 0) .& (df_p.time .>= df_p.treat_time))
        df_p.y = 1.0 .+ 2.0 .* df_p.treat .+ 0.3 .* randn(Xoshiro(94211), nrow(df_p))
        pd = xtset(df_p, :unit, :time)

        did_res = estimate_did(pd, :y, :treat_time; method=:twfe, leads=2, horizon=2)
        @test did_res.conf_level == 0.90
        @test default_ci_level(did_res) == 0.90
        @test occursin("90% CI Low", sprint(show, did_res))
        ci_did_68 = confint(did_res; level=0.68)
        ci_did_90 = confint(did_res)
        ci_did_95 = confint(did_res; level=0.95)
        nz = findall(>(0), did_res.se)
        @test all((ci_did_68[nz, 2] .- ci_did_68[nz, 1]) .<
                  (ci_did_90[nz, 2] .- ci_did_90[nz, 1]) .<
                  (ci_did_95[nz, 2] .- ci_did_95[nz, 1]))

        lpdid_res = estimate_lp_did(pd, :y, :treat, 2; pre_window=2, post_window=2)
        @test lpdid_res.conf_level == 0.68
        @test default_ci_level(lpdid_res) == 0.68
        @test occursin("68% CI Low", sprint(show, lpdid_res))
        ci_lp_68 = confint(lpdid_res)
        ci_lp_95 = confint(lpdid_res; level=0.95)
        nz_lp = findall(>(0), lpdid_res.se)
        @test all((ci_lp_68[nz_lp, 2] .- ci_lp_68[nz_lp, 1]) .<
                  (ci_lp_95[nz_lp, 2] .- ci_lp_95[nz_lp, 1]))
    end

    # =========================================================================
    # 5. D3 Plotting conf_level Configurability & Defaults (CI-05)
    # =========================================================================
    @testset "Plotting conf_level configurability & domain defaults" begin
        rng = Xoshiro(94212)

        # KernelRegression (micro 90%)
        xk = sort(randn(rng, 50))
        yk = sin.(xk) .+ 0.3 .* randn(rng, 50)
        kr = kernel_reg(yk, xk)
        p_kr90 = plot_result(kr)
        p_kr95 = plot_result(kr; conf_level=0.95)
        @test occursin("90% CI", p_kr90.html)
        @test occursin("95% CI", p_kr95.html)
        @test p_kr90.html != p_kr95.html
        @test_throws ArgumentError plot_result(kr; conf_level=1.0)

        # RDDResult respects r.level and supports override
        x_r = randn(rng, 160)
        y_r = 1.5 .* (x_r .>= 0.0) .+ 0.4 .* randn(rng, 160)
        rdd = estimate_rdd(y_r, x_r; cutoff=0.0)
        p_rdd_def = plot_result(rdd)
        p_rdd_95 = plot_result(rdd; conf_level=0.95)
        @test occursin("90% CI", p_rdd_def.html)
        @test occursin("95% CI", p_rdd_95.html)

        # KalmanSmootherResult (macro 68%)
        cov3 = cat([Matrix{Float64}(I, 2, 2) for _ in 1:20]...; dims=3)
        ks = KalmanSmootherResult(
            randn(rng, 2, 20), cov3, randn(rng, 1, 20),
            randn(rng, 2, 20), cov3,
            randn(rng, 2, 20), cov3,
            -10.0,
        )
        p_ks68 = plot_result(ks; varnames=["s1", "s2"])
        p_ks95 = plot_result(ks; varnames=["s1", "s2"], conf_level=0.95)
        @test occursin("68% CI", p_ks68.html)
        @test occursin("95% CI", p_ks95.html)
        @test !occursin("1.96", p_ks68.html)
        @test_throws ArgumentError plot_result(ks; varnames=["s1", "s2"], conf_level=0.0)

        # GDFMNowcastResult (macro 68%)
        nc = GDFMNowcastResult(
            randn(rng, 3), fill(0.2, 3), randn(rng, 15, 3),
            0, 2, trues(15, 3), ["x1", "x2", "x3"],
        )
        p_nc68 = plot_result(nc)
        p_nc95 = plot_result(nc; conf_level=0.95)
        @test occursin("68% CI", p_nc68.html)
        @test occursin("95% CI", p_nc95.html)
        @test_throws ArgumentError plot_result(nc; conf_level=-0.5)

        # DMTestResult (macro 68%)
        e1 = randn(rng, 40)
        e2 = randn(rng, 40) .* 1.3
        dm = diebold_mariano(e1, e2)
        p_dm68 = plot_result(dm)
        p_dm95 = plot_result(dm; conf_level=0.95)
        @test occursin("68% CI", p_dm68.html)
        @test occursin("95% CI", p_dm95.html)
    end
end
