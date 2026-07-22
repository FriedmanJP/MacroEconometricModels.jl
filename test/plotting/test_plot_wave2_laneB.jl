# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# PLT Wave-2 Lane B tests: ARIMA (PLT-25), MCMC diagnostics (PLT-26), and
# static-Bayesian prior/posterior & predictive checks (PLT-27). Structural
# assertions parse EXTRACTED JSON literals via the shared plot_test_helpers, never
# raw p.html (plotrule Testing Rules 1-7).

using Test, Random, Distributions, LinearAlgebra

const MEM = MacroEconometricModels

# First `const data = [...]` literal of a figure, parsed to a Julia value (used to
# count points and inspect coordinates without a raw-HTML scan).
function _laneB_first_data(html::AbstractString)
    for (nm, lit) in extract_json_blocks(html)
        nm == "data" && return assert_strict_json(lit)
    end
    return Any[]
end
_laneB_first_data_len(html::AbstractString) = length(_laneB_first_data(html))

@testset "PLT Wave-2 Lane B (ARIMA / MCMC / Bayes-static)" begin
    Random.seed!(20260718)

    # =========================================================================
    # PLT-25 — ARIMA model plots
    # =========================================================================
    @testset "PLT-25 ARIMA" begin
        y = cumsum(randn(140)) .+ 0.3 .* randn(140)
        yc = y .- sum(y) / length(y)
        m = estimate_arma(yc, 2, 1)

        # :fit — fitted-vs-actual, two series, integer x, strict JSON
        pf = plot_result(m; view=:fit)
        check_plot(pf)
        assert_all_json_valid(pf)
        @test series_count(pf.html) == 2
        @test "Actual" in series_names(pf.html)
        @test "Fitted" in series_names(pf.html)

        # :resid — ACF + PACF, two panels
        pr = plot_result(m; view=:resid)
        check_plot(pr)
        assert_all_json_valid(pr)
        @test any(t -> occursin("ACF", t), panel_titles(pr.html))
        @test any(t -> occursin("PACF", t), panel_titles(pr.html))

        # :diagnostics — the shared four-panel residual figure
        pd = plot_result(m; view=:diagnostics)
        check_plot(pd)
        assert_all_json_valid(pd)
        @test length(panel_titles(pd.html)) == 4

        # :roots — stationary ARMA fit stays inside the circle (no UNSTABLE flag)
        pk = plot_result(m; view=:roots)
        check_plot(pk)
        assert_all_json_valid(pk)
        @test occursin("Inverse Roots", pk.html)
        @test !occursin("UNSTABLE", pk.html)

        # :roots — a nonstationary AR(1) (phi = 1.2) puts an inverse root OUTSIDE
        # the unit circle; assert it from the parsed data + the alert flagging.
        am = MEM.ARModel(randn(60), 1, 0.0, [1.2], 1.0, randn(59), randn(59),
                         -10.0, 1.0, 1.0, :mle, true, 1)
        pku = plot_result(am; view=:roots)
        check_plot(pku)
        assert_all_json_valid(pku)
        @test occursin("UNSTABLE", pku.html)
        @test occursin("Outside unit circle", pku.html)
        @test occursin(MEM._PLOT_ALERT, pku.html)          # alert color present
        d = _laneB_first_data(pku.html)
        @test any(o -> o isa AbstractDict &&
                       (Float64(get(o, "x", 0.0))^2 + Float64(get(o, "y", 0.0))^2) > 1.0, d)

        # Unknown view → ArgumentError
        @test_throws ArgumentError plot_result(m; view=:bogus)

        # ARIMAOrderSelection — IC grid heatmap (:aic/:bic), best-cell marked
        sel = select_arima_order(y, 2, 2)
        pa = plot_result(sel; view=:aic)
        check_plot(pa)
        assert_all_json_valid(pa)
        @test occursin("sequential", pa.html)              # PLT-15 sequential ramp
        @test occursin("bestCell", pa.html)                # best-cell marker
        @test occursin("best p=$(sel.best_p_aic), q=$(sel.best_q_aic)", pa.html)

        pb = plot_result(sel; view=:bic)
        check_plot(pb)
        assert_all_json_valid(pb)

        @test_throws ArgumentError plot_result(sel; view=:nope)

        # NaN cell (a failed fit is stored as Inf; NaN behaves the same) → grey/null
        A = copy(sel.aic_matrix); A[1, 1] = NaN
        seln = MEM.ARIMAOrderSelection(sel.best_p_aic, sel.best_q_aic,
                                       sel.best_p_bic, sel.best_q_bic,
                                       A, sel.bic_matrix,
                                       sel.best_model_aic, sel.best_model_bic)
        pn = plot_result(seln; view=:aic)
        assert_nan_becomes_null(pn)
    end

    # =========================================================================
    # PLT-26 — MCMC diagnostics plots
    # =========================================================================
    @testset "PLT-26 MCMC — BVARPosterior" begin
        post = estimate_bvar(randn(120, 2), 2; n_draws=80)

        for v in (:trace, :density, :running, :acf)
            p = plot_result(post; view=v)
            check_plot(p)
            assert_all_json_valid(p)
        end

        # thin honored: thin=2 halves the plotted trace points (C4 live kwarg)
        n = post.n_draws
        p1 = plot_result(post; view=:trace, params=[1])
        p2 = plot_result(post; view=:trace, params=[1], thin=2)
        @test _laneB_first_data_len(p1.html) == length(1:1:n)
        @test _laneB_first_data_len(p2.html) == length(1:2:n)
        @test _laneB_first_data_len(p2.html) < _laneB_first_data_len(p1.html)

        # Selection by Int and by String resolve to the same coefficient panel
        labels = MEM._bvar_param_labels(post)
        pI = plot_result(post; view=:trace, params=[2])
        pS = plot_result(post; view=:trace, params=[labels[2]])
        @test panel_titles(pI.html) == panel_titles(pS.html)

        # Bad selectors → ArgumentError (C3)
        @test_throws ArgumentError plot_result(post; view=:trace, params=[999])
        @test_throws ArgumentError plot_result(post; view=:trace, params=["nope"])
        @test_throws ArgumentError plot_result(post; view=:nope)

        # A hostile coefficient label round-trips into valid HTML (A8)
        posth = MEM.BVARPosterior{Float64}(post.B_draws, post.Sigma_draws,
                    post.n_draws, post.p, post.n, post.data, post.prior,
                    post.sampler, [HOSTILE_NAME, "y2"])
        ph = plot_result(posth; view=:trace, params=[1])
        assert_escapes(ph)

        # A draw with NaN → null (no bare NaN in a data literal)
        B2 = copy(post.B_draws); B2[3, 1, 1] = NaN
        postn = MEM.BVARPosterior{Float64}(B2, post.Sigma_draws, post.n_draws,
                    post.p, post.n, post.data, post.prior, post.sampler, post.varnames)
        pnan = plot_result(postn; view=:trace, params=[1])
        assert_nan_becomes_null(pnan)
    end

    @testset "PLT-26 MCMC — BayesianDSGE" begin
        # Cheap valid (spec, sol, ss) from a 1-param AR(1); attach a 2-parameter
        # draws matrix + prior so the fixture exercises multi-parameter selection.
        spec = MEM.compute_steady_state(MEM.@dsge begin
            parameters: rho = 0.5, sig = 0.5
            endogenous: y
            exogenous: eps
            y[t] = rho * y[t-1] + sig * eps[t]
            steady_state = [0.0]
        end)
        sol, ss = MEM._build_solution_at_theta(spec, [:rho], [0.5], [:y],
                                               nothing, :gensys, NamedTuple())
        N = 60
        td = hcat(0.4 .+ 0.1 .* randn(N), 0.5 .+ 0.1 .* randn(N))
        prior = MEM.DSGEPrior{Float64}([:rho, :sig],
                    Distribution[Normal(0.5, 0.2), Normal(0.5, 0.2)],
                    [0.0, 0.0], [1.0, 1.0])
        bd = MEM.BayesianDSGE{Float64}(td, zeros(N), [:rho, :sig], prior, 0.0,
                    :rwmh, 0.5, Float64[], Float64[], spec, sol, ss)

        for v in (:trace, :density, :running, :acf)
            p = plot_result(bd; view=v)
            check_plot(p)
            assert_all_json_valid(p)
        end
        @test_throws ArgumentError plot_result(bd; view=:bogus)

        # params=:rho vs "rho" vs 1 select the SAME single parameter panel
        pSym = plot_result(bd; view=:trace, params=:rho)
        pStr = plot_result(bd; view=:trace, params="rho")
        pInt = plot_result(bd; view=:trace, params=1)
        @test panel_titles(pSym.html) == panel_titles(pStr.html) == panel_titles(pInt.html)
        @test length(panel_titles(pSym.html)) == 1

        # :density draws the prior overlay + a posterior-mean vertical line (axis:"x")
        pden = plot_result(bd; view=:density, params=:rho)
        @test "Prior" in series_names(pden.html)
        @test "Posterior" in series_names(pden.html)
        @test occursin("\"axis\":\"x\"", pden.html)

        # NaN draw → null on a BayesianDSGE trace too
        td2 = copy(td); td2[3, 1] = NaN
        bdn = MEM.BayesianDSGE{Float64}(td2, zeros(N), [:rho, :sig], prior, 0.0,
                    :rwmh, 0.5, Float64[], Float64[], spec, sol, ss)
        assert_nan_becomes_null(plot_result(bdn; view=:trace, params=1))
    end

    @testset "PLT-26 MCMC — MCMCDiagnostics" begin
        d = MEM.MCMCDiagnostics{Float64}([:alpha, :beta, :gamma],
                [1.0, 1.02, 1.005],                 # R-hat: beta > 1.01
                [500.0, 300.0, 800.0],              # ESS bulk: beta < 400
                [450.0, 200.0, 600.0],              # ESS tail: beta < 400
                zeros(3), ones(3), zeros(3), ones(3), 1000, :rwmh)
        p = plot_result(d)
        check_plot(p)
        assert_all_json_valid(p)
        # Three horizontal panels with threshold flags in the alert color
        titles = panel_titles(p.html)
        @test any(t -> occursin("R-hat", t) && occursin("flagged", t), titles)
        @test any(t -> occursin("ESS (bulk)", t), titles)
        @test any(t -> occursin("ESS (tail)", t), titles)
        @test occursin(MEM._PLOT_ALERT, p.html)            # threshold reference line
    end

    # =========================================================================
    # PLT-27 — Prior/posterior & predictive-check plots
    # =========================================================================
    @testset "PLT-27 Bayes static" begin
        # PriorPosteriorOverlap — horizontal bar + threshold line, flagged annotated
        o = MEM.PriorPosteriorOverlap{Float64}([:a, :b, :c],
                [0.30, 0.90, 0.85], [false, true, true], 0.8)
        po = plot_result(o)
        check_plot(po)
        assert_all_json_valid(po)
        @test occursin("flagged", po.html)
        @test occursin("threshold 0.8", po.html)           # _fmt threshold (C9)
        @test occursin(MEM._PLOT_ALERT, po.html)           # alert reference line

        # NaN overlap → null, no bare NaN
        onan = MEM.PriorPosteriorOverlap{Float64}([:a, :b], [NaN, 0.9],
                [false, true], 0.8)
        assert_nan_becomes_null(plot_result(onan))

        # PriorPredictiveResult — one histogram/density panel per statistic
        pp = MEM.PriorPredictiveResult{Float64}(["mean", "var", "ac1"],
                randn(80, 3), 100, 80, 50)
        ppp = plot_result(pp)
        check_plot(ppp)
        assert_all_json_valid(ppp)
        @test length(panel_titles(ppp.html)) == 3
        @test_throws ArgumentError plot_result(pp; view=:bad)

        # C7 cap: > 12 statistics → cap note in the title
        big = MEM.PriorPredictiveResult{Float64}(
                String["s$(i)" for i in 1:15], randn(40, 15), 50, 40, 30)
        pbig = plot_result(big)
        @test occursin("showing 12 of 15", pbig.html)

        # PosteriorPredictiveCheck — observed vline (axis:"x") + p-value in title
        ppc = MEM.PosteriorPredictiveCheck{Float64}(["mean", "var"],
                [0.10, 1.20], randn(80, 2), [0.40, 0.03], 100, 80)
        pc = plot_result(ppc)
        check_plot(pc)
        assert_all_json_valid(pc)
        @test occursin("\"axis\":\"x\"", pc.html)          # observed reference line
        @test occursin("p = 0.4000", pc.html)              # _fmt p-value (C9)
        # NaN p-value renders as "n/a", not a serialized NaN
        ppcn = MEM.PosteriorPredictiveCheck{Float64}(["mean"], [0.1],
                randn(50, 1), [NaN], 60, 50)
        pcn = plot_result(ppcn)
        check_plot(pcn)
        @test occursin("p = n/a", pcn.html)

        # A statistic name with a quote round-trips (A8)
        pph = MEM.PriorPredictiveResult{Float64}([HOSTILE_NAME, "b"],
                randn(60, 2), 70, 60, 40)
        assert_escapes(plot_result(pph))

        # IdentificationDiagnostics — log-scaled singular values + tol line + verdict
        idg = MEM.IdentificationDiagnostics{Float64}([:a, :b, :c],
                [0.5, 0.5, 0.5], 2, 3, 10, 5, [1.2, 0.4, 1e-9], 1e-6, zeros(3, 1), false)
        pid = plot_result(idg)
        check_plot(pid)
        assert_all_json_valid(pid)
        @test occursin("UNIDENTIFIED", pid.html)
        @test occursin("rank 2/3", pid.html)
        @test occursin("logx = true", pid.html)            # log scale
        @test occursin(MEM._PLOT_ALERT, pid.html)          # tol reference line

        # PosteriorMode — dot-and-whisker (mode ± 1.96·√diag(inv_hessian))
        pm = MEM.PosteriorMode{Float64}([0.5, 0.3],
                [0.01 0.0; 0.0 0.04], [100.0 0.0; 0.0 25.0],
                -10.0, -8.0, -12.0, [:a, :b], true, 20)
        pmo = plot_result(pm)
        check_plot(pmo)
        assert_all_json_valid(pmo)
        dm = _laneB_first_data(pmo.html)
        @test all(o -> o isa AbstractDict && haskey(o, "ci_lo") && haskey(o, "ci_hi"), dm)
        # 95% interval half-width for a=0.5, var 0.01 ⇒ 1.96*0.1 = 0.196
        row_a = first(x for x in dm if x isa AbstractDict && get(x, "name", "") == "a")
        @test isapprox(Float64(row_a["ci_hi"]) - Float64(row_a["effect"]), 1.96 * 0.1; atol=1e-6)
    end
end
