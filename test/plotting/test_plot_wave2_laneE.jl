# =============================================================================
# Lane E — PLT-24 (generic residual diagnostics), PLT-28 (Bayesian nested fans),
# PLT-37 (GMM/SMM + GARCH news-impact), PLT-38 (forecast-eval + LP extras).
#
# Uses the shared assertions from plot_test_helpers.jl (Testing Rules 1-7): check_plot
# (Rule 1 + A12), assert_all_json_valid (Rule 6), assert_nan_becomes_null (Rule 4),
# assert_escapes (Rule 5, HOSTILE_NAME), series_count/panel_titles (Rule 2),
# @test_throws ArgumentError for bad view/regime (Rule 3 / C5). Structural assertions
# parse EXTRACTED JSON literals, never raw p.html.
# =============================================================================

using MacroEconometricModels, Test, Random, DataFrames
const _MEM = MacroEconometricModels

# Extract the `const fan = [...]` literal (the fan renderer's band-spec array, which
# `extract_json_blocks` does not name) using the shared balance helper, so band counts
# are asserted against strict JSON — not a raw-html scan.
function _extract_fan_block(html::AbstractString)
    anchor = "const fan ="
    r = findfirst(anchor, html)
    r === nothing && return nothing
    j = nextind(html, last(r))
    while j <= lastindex(html) && isspace(html[j]); j = nextind(html, j); end
    k = _tj_balance(html, j)
    k == 0 ? nothing : String(html[j:k])
end

# Values of `key` across the first extracted `data` block (for stat=:median/:mean diffs).
function _first_data_key(html::AbstractString, key::AbstractString)
    for (nm, lit) in extract_json_blocks(html)
        nm == "data" || continue
        val, _ = _tj_parse_value(lit, firstindex(lit))
        return Float64[Float64(o[key]) for o in val if o isa AbstractDict && haskey(o, key) && o[key] !== nothing]
    end
    return Float64[]
end

@testset "Lane E — PLT-24/28/37/38" begin

    # =========================================================================
    # PLT-24 — generic residual-diagnostics view (view=:diagnostics)
    # =========================================================================
    @testset "PLT-24 residual diagnostics" begin
        Random.seed!(2401)

        @testset "RegModel (shared helper, no scale-clone)" begin
            X = randn(80, 3); y = X * [1.0, -0.5, 0.3] .+ randn(80)
            m = estimate_reg(y, X)
            p = plot_result(m)                       # :diagnostics is the only view
            check_plot(p)
            assert_all_json_valid(p)
            # Four panels from the shared helper; no bespoke bar-histogram scale-clone.
            @test length(panel_titles(p.html)) >= 4
            @test occursin("Residual ACF", p.html)
            @test_throws ArgumentError plot_result(m; view=:bogus)
        end

        @testset "VARModel / VECMModel (eq= selector)" begin
            Y = randn(120, 3)
            v = estimate_var(Y, 2)
            p = plot_result(v; view=:diagnostics)     # eq=nothing → first eq + C7 note
            check_plot(p); assert_all_json_valid(p)
            @test occursin("Showing 1 of 3 equations", p.html)   # cap note (C7)
            check_plot(plot_result(v; view=:diagnostics, eq="y2"))
            check_plot(plot_result(v; view=:diagnostics, eq=3))
            @test_throws ArgumentError plot_result(v; view=:diagnostics, eq=9)
            @test_throws ArgumentError plot_result(v; view=:bogus)

            Yv = cumsum(randn(120, 3); dims=1)
            vec_ = estimate_vecm(Yv, 2; rank=1)
            check_plot(plot_result(vec_; view=:diagnostics))
        end

        @testset "PanelRegModel / PanelIVModel" begin
            df = DataFrame(g=repeat(1:20, inner=10), t=repeat(1:10, 20),
                           y=randn(200), x=randn(200), z=randn(200))
            pd = xtset(df, :g, :t)
            pr = estimate_xtreg(pd, :y, [:x])
            p = plot_result(pr; view=:diagnostics)
            check_plot(p); assert_all_json_valid(p)
            @test_throws ArgumentError plot_result(pr; view=:bogus)
            piv = estimate_xtiv(pd, :y, Symbol[], [:x]; instruments=[:z])
            check_plot(plot_result(piv; view=:diagnostics))
        end

        @testset "GARCH family (standardized) + default unchanged" begin
            g = estimate_garch(randn(200), 1, 1)
            check_plot(plot_result(g))                         # :default (3-panel) intact
            @test occursin("Diagnostic", plot_result(g).html)
            pd = plot_result(g; view=:diagnostics)
            check_plot(pd); assert_all_json_valid(pd)
            @test occursin("Std. Residual", pd.html)           # standardized label
            @test_throws ArgumentError plot_result(g; view=:bogus)
            for est in (estimate_arch(randn(200), 1),
                        estimate_egarch(randn(200), 1, 1),
                        estimate_gjr_garch(randn(200), 1, 1))
                check_plot(plot_result(est; view=:diagnostics))
            end
        end

        @testset "SVModel (already standardized)" begin
            sv = estimate_sv(randn(120); n_samples=200, burnin=100)
            check_plot(plot_result(sv))                         # :default
            check_plot(plot_result(sv; view=:diagnostics))
            @test_throws ArgumentError plot_result(sv; view=:bogus)
        end

        @testset "MIDAS / Threshold / STAR / MSReg / NARDL" begin
            y = randn(200)
            check_plot(plot_result(estimate_setar(y, 1, 1; linearity=false); view=:diagnostics))
            check_plot(plot_result(estimate_star(y, 1; type=:lstr1); view=:diagnostics))
            check_plot(plot_result(estimate_ms(y; k_regimes=2); view=:diagnostics))

            m = 3; K = 6; ny = 80
            mu = estimate_midas(randn(ny), randn(ny * m); m=m, K=K, weights=:umidas, p_ar=0)
            check_plot(plot_result(mu; view=:diagnostics))
            @test_throws ArgumentError plot_result(mu; view=:bogus)

            yn = cumsum(randn(120)); xn = cumsum(randn(120))
            nm = estimate_nardl(yn, reshape(xn, :, 1); asymmetric=:all, p=1, q=1, case=3)
            check_plot(plot_result(nm; view=:diagnostics))
            @test_throws ArgumentError plot_result(nm; view=:bogus)
        end
    end

    # =========================================================================
    # PLT-28 — Bayesian nested-fan band upgrades
    # =========================================================================
    @testset "PLT-28 Bayesian nested fans" begin
        Random.seed!(2801)
        Y = randn(90, 3)
        post = estimate_bvar(Y, 2; n_draws=120)

        @testset "BayesianImpulseResponse → fan, all bands, stat, draws" begin
            r = irf(post, 8)
            p = plot_result(r)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("Bayesian", p.html)                 # regression: title kept
            @test occursin("const fan", p.html)                # routed through fan renderer
            fanlit = _extract_fan_block(p.html)
            @test fanlit !== nothing
            assert_strict_json(fanlit)
            @test count("lo_key", fanlit) >= 1                 # ≥1 nested band per pair

            @test_throws ArgumentError plot_result(r; stat=:bogus)

            # stat=:median vs :mean changes the central line (live kwarg C4)
            pmed = plot_result(r; var=1, shock=1, stat=:median)
            pmean = plot_result(r; var=1, shock=1, stat=:mean)
            med_vals = _first_data_key(pmed.html, "med")
            mean_vals = _first_data_key(pmean.html, "med")
            @test length(med_vals) > 0 && med_vals != mean_vals

            # draws overlay caps at ≤200 with a visible per-panel note
            pdr = plot_result(r; var=1, shock=1, draws=500)
            check_plot(pdr); assert_all_json_valid(pdr)
            @test series_count(pdr.html) <= 201                # central + ≤200 draws
            @test any(t -> occursin("draws", t), panel_titles(pdr.html))
        end

        @testset "BayesianFEVD / BayesianHistoricalDecomposition → fans" begin
            f = fevd(post, 8)
            pf = plot_result(f)
            check_plot(pf); assert_all_json_valid(pf)
            @test occursin("Bayesian", pf.html) && occursin("const fan", pf.html)
            @test_throws ArgumentError plot_result(f; stat=:bogus)

            hdr = historical_decomposition(post, size(Y, 1) - 2)
            ph = plot_result(hdr)
            check_plot(ph); assert_all_json_valid(ph)
            @test occursin("Bayesian", ph.html) && occursin("const fan", ph.html)
            @test_throws ArgumentError plot_result(hdr; stat=:bogus)
        end

        @testset "BayesianDSGESimulation (NEW) — bands, NaN, escaping" begin
            Tn = 16; nv = 2; nd = 500
            levels = [0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975]   # 3 symmetric pairs
            ap = randn(nd, Tn, nv)
            q = zeros(Tn, nv, length(levels)); pe = zeros(Tn, nv)
            for t in 1:Tn, v in 1:nv
                col = sort(ap[:, t, v])
                for (qi, l) in enumerate(levels); q[t, v, qi] = _MEM._quantile(col, l); end
                pe[t, v] = _MEM._quantile(col, 0.5)
            end
            sim = _MEM.BayesianDSGESimulation{Float64}(q, pe, Tn,
                                                       [HOSTILE_NAME, "b"], levels, ap)
            p = plot_result(sim)
            check_plot(p); assert_all_json_valid(p)
            fanlit = _extract_fan_block(p.html)
            @test fanlit !== nothing && count("lo_key", fanlit) == 3   # 3 pairs → 3 bands
            assert_escapes(p)                                          # hostile var name (A8)
            @test_throws ArgumentError plot_result(sim; stat=:bogus)

            # draws cap ≤200 with note
            pdr = plot_result(sim; stat=:mean, draws=500)
            check_plot(pdr)
            @test series_count(pdr.html) <= 201
            @test any(t -> occursin("draws", t), panel_titles(pdr.html))

            # NaN quantile → null gap (Rule 4)
            q2 = copy(q); q2[3, 1, 1] = NaN
            sim2 = _MEM.BayesianDSGESimulation{Float64}(q2, pe, Tn, ["a", "b"], levels, ap)
            assert_nan_becomes_null(plot_result(sim2; var=1))
        end
    end

    # =========================================================================
    # PLT-37 — GMM/SMM moment fit + GARCH news-impact
    # =========================================================================
    @testset "PLT-37 GMM/SMM + news impact" begin
        Random.seed!(3701)

        @testset "GMMModel moment discrepancy bar + J note" begin
            n = 300; X = randn(n, 2); yv = X * [1.0, -0.5] .+ randn(n)
            Z = hcat(X, randn(n)); data = hcat(yv, Z)     # 3 moments, 2 params (over-id)
            mfn(theta, d) = begin
                y_d = d[:, 1]; X_d = d[:, 2:3]; Zd = d[:, 2:4]
                Zd .* (y_d - X_d * theta)
            end
            g = estimate_gmm(mfn, [0.0, 0.0], data; weighting=:two_step)
            p = plot_result(g)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("J =", p.html) && occursin("df = 1", p.html)   # over-id df
            # figure title escapes a hostile user title (HTML sink)
            ph = plot_result(g; title=HOSTILE_NAME)
            @test !occursin("<c>", ph.html)
        end

        @testset "GARCH-family news-impact view + shocks=0 vline" begin
            for est in (estimate_garch(randn(250), 1, 1),
                        estimate_egarch(randn(250), 1, 1),
                        estimate_gjr_garch(randn(250), 1, 1),
                        estimate_igarch(randn(250), 1, 1))
                p = plot_result(est; view=:news_impact)
                check_plot(p); assert_all_json_valid(p)
                # vertical reference line at the shock=0 axis:"x"
                @test occursin("\"axis\":\"x\"", p.html)
            end
        end
    end

    # =========================================================================
    # PLT-38 — forecast-evaluation & LP extras
    # =========================================================================
    @testset "PLT-38 forecast-eval + LP" begin
        Random.seed!(3801)
        actual = randn(60)
        fc1 = actual .+ 0.3 .* randn(60)
        fc2 = actual .+ 0.6 .* randn(60)

        @testset "ForecastEvaluation :metrics / :theil (+ escaping)" begin
            ev = forecast_evaluate(actual, hcat(fc1, fc2);
                                   model_names=[HOSTILE_NAME, "M2"])
            pm = plot_result(ev)                        # :metrics (grouped)
            check_plot(pm); assert_all_json_valid(pm); assert_escapes(pm)
            check_plot(plot_result(ev; view=:metrics, metric="RMSE"))
            pt = plot_result(ev; view=:theil)
            check_plot(pt); assert_all_json_valid(pt)
            @test_throws ArgumentError plot_result(ev; view=:bogus)
        end

        @testset "MincerZarnowitz efficiency line (no scatter, NaN)" begin
            mz = mincer_zarnowitz(actual, fc1)
            p = plot_result(mz)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("45°", p.html) && occursin("Wald p", p.html)
            # No fabricated scatter points (C6): the data block is only the 2-point line grid.
            @test length(_first_data_key(p.html, "eff")) == 2
            # NaN coefficient → null in the line data (Rule 4)
            mz_nan = _MEM.MincerZarnowitzResult{Float64}(NaN, mz.b, mz.se, mz.wald,
                mz.pvalue_wald, mz.fstat, mz.pvalue_f, mz.lags, mz.kernel, mz.T_obs)
            assert_nan_becomes_null(plot_result(mz_nan))
        end

        @testset "DM / ClarkWest dot-whisker" begin
            dm = diebold_mariano(actual .- fc1, actual .- fc2)
            check_plot(plot_result(dm)); assert_all_json_valid(plot_result(dm))
            cw = clark_west(actual .- fc1, actual .- fc2, fc1 .- fc2)
            check_plot(plot_result(cw))
        end

        @testset "ForecastCombination weights (negative-safe) + escaping" begin
            fcomb = combine_forecasts(hcat(fc1, fc2), actual;
                                      method=:granger_ramanathan,
                                      model_names=[HOSTILE_NAME, "M2"])
            p = plot_result(fcomb)
            check_plot(p); assert_all_json_valid(p); assert_escapes(p)
        end

        @testset "StateLPModel two-regime IRF" begin
            Yl = randn(150, 2); statev = cumsum(randn(150))
            slp = estimate_state_lp(Yl, 1, statev, 8; lags=2,
                                    varnames=[HOSTILE_NAME, "y2"])
            p = plot_result(slp)                        # regime=:both
            check_plot(p); assert_all_json_valid(p); assert_escapes(p)
            check_plot(plot_result(slp; regime=:recession, var=1))
            check_plot(plot_result(slp; regime=:expansion, var="y2"))
            @test_throws ArgumentError plot_result(slp; regime=:bogus)
            @test_throws ArgumentError plot_result(slp; var=9)
        end

        @testset "LP-family delegation (+ LP diagnostics)" begin
            Yl = randn(150, 2)
            lp = estimate_lp(Yl, 1, 8; lags=2)
            check_plot(plot_result(lp))                 # :irf → LPImpulseResponse plot
            check_plot(plot_result(lp; view=:diagnostics, h=3))
            @test_throws ArgumentError plot_result(lp; view=:bogus)

            Z = randn(150, 1)
            lpiv = estimate_lp_iv(Yl, 1, Z, 8; lags=2)
            check_plot(plot_result(lpiv))
            check_plot(plot_result(lpiv; view=:diagnostics))
            check_plot(plot_result(estimate_smooth_lp(Yl, 1, 8; lags=2)))
            trt = rand(Bool, 150)
            check_plot(plot_result(estimate_propensity_lp(Yl, trt, randn(150, 1), 8; lags=2)))
        end
    end
end
