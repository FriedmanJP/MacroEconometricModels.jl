# Coverage tests targeting 4 source files with low coverage:
#   src/plotting/render.jl, src/plotting/models.jl,
#   src/summary.jl, src/summary_refs.jl

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

if !@isdefined(M)
    const M = MacroEconometricModels
end

@testset "Display Coverage" begin

    # =========================================================================
    # 1. plotting/render.jl — CSS, HTML rendering, show methods
    # =========================================================================
    @testset "render.jl — _render_css" begin
        css1 = M._render_css(1)
        @test occursin("100%", css1)
        @test occursin("font-family", css1)

        css3 = M._render_css(3)
        @test occursin("33%", css3)

        css2 = M._render_css(2)
        @test occursin("50%", css2)
    end

    @testset "render.jl — _render_html" begin
        html = M._render_html(; title="Test", css="body{}", body="<p>hi</p>", scripts="")
        @test occursin("<!DOCTYPE html>", html)
        @test occursin("<title>Test</title>", html)
        @test occursin("d3.min.js", html)
        @test occursin("<p>hi</p>", html)
    end

    @testset "render.jl — _render_js_core" begin
        js = M._render_js_core()
        @test occursin("tooltip", js)
        @test occursin("fmt(v)", js)
        @test occursin("showTip", js)
        @test occursin("hideTip", js)
    end

    @testset "render.jl — show methods for PlotOutput" begin
        p = PlotOutput("<html><body>test</body></html>")

        # text/html
        io = IOBuffer()
        show(io, MIME"text/html"(), p)
        s = String(take!(io))
        @test s == "<html><body>test</body></html>"

        # text/plain
        io2 = IOBuffer()
        show(io2, MIME"text/plain"(), p)
        s2 = String(take!(io2))
        @test occursin("PlotOutput", s2)
        @test occursin("bytes", s2)

        # default show
        io3 = IOBuffer()
        show(io3, p)
        s3 = String(take!(io3))
        @test occursin("PlotOutput", s3)
    end

    @testset "render.jl — save_plot" begin
        rng = Random.MersenneTwister(9001)
        y = cumsum(randn(rng, 200))
        r = hp_filter(y)
        p = plot_result(r)
        tmpfile = tempname() * ".html"
        result = save_plot(p, tmpfile)
        @test result == tmpfile
        @test isfile(tmpfile)
        @test filesize(tmpfile) > 100
        rm(tmpfile)
    end

    @testset "render.jl — _render_body_single and _render_body_figure" begin
        panel = M._PanelSpec("id1", "Title1", "// js")
        body1 = M._render_body_single(panel; title="Fig Title")
        @test occursin("Fig Title", body1)
        @test occursin("id1", body1)

        body2 = M._render_body_figure([panel, panel]; title="Multi", source="Src", note="Nt")
        @test occursin("Multi", body2)
        @test occursin("Src", body2)
        @test occursin("Nt", body2)
        @test occursin("panel-grid", body2)
    end

    @testset "render.jl — _make_plot auto columns" begin
        panel = M._PanelSpec("tid", "T", "// js")
        # single panel -> ncols auto = 1
        p1 = M._make_plot([panel]; title="Single")
        @test p1 isa PlotOutput
        @test occursin("Single", p1.html)

        # 4 panels -> auto ncols = 3
        p4 = M._make_plot([panel, panel, panel, panel]; title="Four")
        @test p4 isa PlotOutput
    end

    # =========================================================================
    # 2. plotting/models.jl — plot_result for model types
    # =========================================================================
    @testset "models.jl — ARCHModel plot" begin
        rng = Random.MersenneTwister(9010)
        y = randn(rng, 300)
        m = estimate_arch(y, 2)
        p = plot_result(m)
        @test p isa PlotOutput
        @test occursin("ARCH", p.html)
    end

    @testset "models.jl — GARCHModel plot" begin
        rng = Random.MersenneTwister(9011)
        y = randn(rng, 500)
        m = estimate_garch(y, 1, 1)
        p = plot_result(m)
        @test p isa PlotOutput
        @test occursin("GARCH", p.html)
    end

    @testset "models.jl — EGARCHModel plot" begin
        rng = Random.MersenneTwister(9012)
        y = randn(rng, 300)
        m = estimate_egarch(y, 1, 1)
        p = plot_result(m)
        @test p isa PlotOutput
        @test occursin("EGARCH", p.html)
    end

    @testset "models.jl — GJRGARCHModel plot" begin
        rng = Random.MersenneTwister(9013)
        y = randn(rng, 300)
        m = estimate_gjr_garch(y, 1, 1)
        p = plot_result(m)
        @test p isa PlotOutput
        @test occursin("GJR-GARCH", p.html)
    end

    if !FAST
        @testset "models.jl — SVModel plot" begin
            rng = Random.MersenneTwister(9014)
            y = randn(rng, 200)
            m = estimate_sv(y; n_samples=100, burnin=50)
            p = plot_result(m)
            @test p isa PlotOutput
            @test occursin("Stochastic Volatility", p.html)
        end
    end

    @testset "models.jl — FactorModel plot" begin
        rng = Random.MersenneTwister(9015)
        X = randn(rng, 200, 20)
        fm = estimate_factors(X, 3)
        p = plot_result(fm)
        @test p isa PlotOutput
        @test occursin("Factor", p.html)
        @test occursin("Scree", p.html)
    end

    @testset "models.jl — DynamicFactorModel plot" begin
        rng = Random.MersenneTwister(9016)
        X = randn(rng, 200, 20)
        fm = estimate_dynamic_factors(X, 2, 1)
        p = plot_result(fm)
        @test p isa PlotOutput
        @test occursin("Dynamic Factor", p.html)
    end

    @testset "models.jl — TimeSeriesData plot" begin
        rng = Random.MersenneTwister(9017)
        d = TimeSeriesData(randn(rng, 100, 3); varnames=["GDP", "CPI", "RATE"])
        p = plot_result(d)
        @test p isa PlotOutput
        @test occursin("GDP", p.html)

        # With var selection
        p2 = plot_result(d; vars=["GDP", "CPI"])
        @test p2 isa PlotOutput
    end

    @testset "models.jl — PanelData plot" begin
        rng = Random.MersenneTwister(9018)
        df = DataFrame(group=repeat(1:3, inner=20), time=repeat(1:20, 3),
            x=randn(rng, 60), y=randn(rng, 60))
        pd = xtset(df, :group, :time)
        p = plot_result(pd)
        @test p isa PlotOutput
    end

    @testset "models.jl — FAVARModel plot" begin
        rng = Random.MersenneTwister(9019)
        Y_slow = randn(rng, 150, 3)
        Y_fast = randn(rng, 150, 10)
        m = estimate_favar(Y_slow, Y_fast, 2, 2)
        p = plot_result(m)
        @test p isa PlotOutput
        @test occursin("FAVAR", p.html)
    end

    @testset "models.jl — plot with save_path kwarg" begin
        rng = Random.MersenneTwister(9020)
        y = randn(rng, 300)
        m = estimate_arch(y, 1)
        tmpfile = tempname() * ".html"
        p = plot_result(m; save_path=tmpfile)
        @test isfile(tmpfile)
        rm(tmpfile)
    end

    @testset "models.jl — plot with custom title" begin
        rng = Random.MersenneTwister(9021)
        y = randn(rng, 300)
        m = estimate_garch(y, 1, 1)
        p = plot_result(m; title="Custom Title")
        @test occursin("Custom Title", p.html)
    end

    # =========================================================================
    # 3. summary.jl — report() dispatches, point_estimate/has_uncertainty
    # =========================================================================
    @testset "summary.jl — report(VECMModel)" begin
        rng = Random.MersenneTwister(9030)
        Y = cumsum(randn(rng, 150, 3), dims=1)
        vecm = estimate_vecm(Y, 2; rank=1)
        redirect_stdout(devnull) do
            report(vecm)
        end
        @test true
    end

    @testset "summary.jl — report(VECMForecast)" begin
        rng = Random.MersenneTwister(9031)
        Y = cumsum(randn(rng, 150, 3), dims=1)
        vecm = estimate_vecm(Y, 2; rank=1)
        fc = forecast(vecm, 10)
        redirect_stdout(devnull) do
            report(fc)
        end
        @test true
    end

    @testset "summary.jl — report(LP variants)" begin
        rng = Random.MersenneTwister(9032)
        Y = randn(rng, 100, 3)
        lp = estimate_lp(Y, 1, 8; lags=2)
        redirect_stdout(devnull) do
            report(lp)
        end
        @test true
    end

    @testset "summary.jl — report(ARIMA)" begin
        rng = Random.MersenneTwister(9033)
        y = randn(rng, 200)
        ar = estimate_ar(y, 2)
        redirect_stdout(devnull) do
            report(ar)
        end
        @test true
    end

    @testset "summary.jl — report(FactorModel)" begin
        rng = Random.MersenneTwister(9034)
        X = randn(rng, 100, 10)
        fm = estimate_factors(X, 3)
        redirect_stdout(devnull) do
            report(fm)
        end
        @test true
    end

    @testset "summary.jl — report(GARCHModel)" begin
        rng = Random.MersenneTwister(9035)
        y = randn(rng, 500)
        m = estimate_garch(y, 1, 1)
        redirect_stdout(devnull) do
            report(m)
        end
        @test true
    end

    @testset "summary.jl — report(filter types)" begin
        rng = Random.MersenneTwister(9036)
        y = cumsum(randn(rng, 200))
        redirect_stdout(devnull) do
            report(hp_filter(y))
            report(hamilton_filter(y))
            report(beveridge_nelson(y))
            report(baxter_king(y))
            report(boosted_hp(y))
        end
        @test true
    end

    @testset "summary.jl — report(IRF/FEVD/HD)" begin
        rng = Random.MersenneTwister(9037)
        Y = randn(rng, 100, 3)
        m = estimate_var(Y, 2)
        irf_r = irf(m, 8)
        fevd_r = fevd(m, 8)
        hd_r = historical_decomposition(m, size(m.Y, 1) - m.p)
        redirect_stdout(devnull) do
            report(irf_r)
            report(fevd_r)
            report(hd_r)
        end
        @test true
    end

    @testset "summary.jl — report(DSGESolution)" begin
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        redirect_stdout(devnull) do
            report(sol)
            report(spec)
        end
        @test true
    end

    @testset "summary.jl — report(BVARPosterior)" begin
        rng = Random.MersenneTwister(9039)
        Y = randn(rng, 100, 3)
        post = estimate_bvar(Y, 2; n_draws=50)
        redirect_stdout(devnull) do
            report(post)
        end
        @test true
    end

    @testset "summary.jl — report(GMM)" begin
        rng = Random.MersenneTwister(9040)
        data_gmm = randn(rng, 200, 3)
        g = (theta, data) -> data[:, 2:3] .* (data[:, 1] .- theta[1])
        gmm_m = estimate_gmm(g, [0.0], data_gmm)
        redirect_stdout(devnull) do
            report(gmm_m)
        end
        @test true
    end

    @testset "summary.jl — report(VECMGrangerResult)" begin
        rng = Random.MersenneTwister(9041)
        Y = cumsum(randn(rng, 150, 3), dims=1)
        vecm = estimate_vecm(Y, 2; rank=1)
        gc = granger_causality_vecm(vecm, 1, 2)
        redirect_stdout(devnull) do
            report(gc)
        end
        @test true
    end

    @testset "summary.jl — point_estimate/has_uncertainty/uncertainty_bounds" begin
        rng = Random.MersenneTwister(9042)
        Y = randn(rng, 100, 2)
        m = estimate_var(Y, 2)

        # ImpulseResponse without CI
        irf_no = irf(m, 8)
        @test M.point_estimate(irf_no) == irf_no.values
        @test M.has_uncertainty(irf_no) == false
        @test M.uncertainty_bounds(irf_no) === nothing

        # ImpulseResponse with CI
        irf_ci = irf(m, 8; ci_type=:bootstrap, reps=50)
        @test M.point_estimate(irf_ci) == irf_ci.values
        @test M.has_uncertainty(irf_ci) == true
        bounds = M.uncertainty_bounds(irf_ci)
        @test bounds !== nothing
        @test bounds[1] == irf_ci.ci_lower

        # FEVD
        fevd_r = fevd(m, 8)
        @test M.point_estimate(fevd_r) == fevd_r.proportions
        @test M.has_uncertainty(fevd_r) == false

        # HD
        hd_r = historical_decomposition(m, size(m.Y, 1) - m.p)
        @test M.point_estimate(hd_r) == hd_r.contributions
        @test M.has_uncertainty(hd_r) == false
    end

    @testset "summary.jl — Bayesian point_estimate/uncertainty_bounds" begin
        rng = Random.MersenneTwister(9043)
        Y = randn(rng, 100, 2)
        post = estimate_bvar(Y, 2; n_draws=50)

        birf = irf(post, 8)
        @test M.point_estimate(birf) == birf.point_estimate
        @test M.has_uncertainty(birf) == true
        bb = M.uncertainty_bounds(birf)
        @test bb !== nothing

        bfevd = fevd(post, 8)
        @test M.point_estimate(bfevd) == bfevd.point_estimate
        @test M.has_uncertainty(bfevd) == true

        bhd = historical_decomposition(post, size(Y, 1) - 2)
        @test M.point_estimate(bhd) == bhd.point_estimate
        @test M.has_uncertainty(bhd) == true
    end

    @testset "summary.jl — report(DSGEEstimation)" begin
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        T_sim = 100
        sim_data = M.simulate(sol, T_sim)
        Y_obs = sim_data[:, 1:1]

        est = M.estimate_dsge(spec, Y_obs, [:rho];
            method=:euler_gmm, weighting=:identity)
        redirect_stdout(devnull) do
            report(est)
        end
        @test true
    end

    # =========================================================================
    # 4. summary_refs.jl — refs() dispatches for new types, format functions
    # =========================================================================
    @testset "refs.jl — _delatex" begin
        @test M._delatex("L\\\"utkepohl") == "L\u00fctkepohl"
        @test M._delatex("\\&") == "&"
        @test M._delatex("em---dash") == "em\u2014dash"
        @test M._delatex("en--dash") == "en\u2013dash"
        @test M._delatex("{braces}") == "braces"
    end

    @testset "refs.jl — 4 formats on VARModel" begin
        rng = Random.MersenneTwister(9050)
        Y = randn(rng, 100, 3)
        m = estimate_var(Y, 2)
        for fmt in (:text, :latex, :bibtex, :html)
            r = refs(m; format=fmt)
            @test r isa String
            @test !isempty(r)
        end
        # Spot-check format-specific content
        @test occursin("\\bibitem", refs(m; format=:latex))
        @test occursin("@article", refs(m; format=:bibtex)) || occursin("@book", refs(m; format=:bibtex))
        @test occursin("<p>", refs(m; format=:html))
    end

    @testset "refs.jl — symbol dispatch" begin
        r = refs(:johansen)
        @test r isa String
        @test !isempty(r)

        r2 = refs(:fastica; format=:bibtex)
        @test occursin("@article", r2)

        r3 = refs(:gensys; format=:latex)
        @test occursin("Sims", r3)
    end

    @testset "refs.jl — AndrewsResult" begin
        rng = Random.MersenneTwister(9051)
        y = randn(rng, 200)
        X = hcat(ones(200), randn(rng, 200))
        ar = andrews_test(y, X; test=:supwald)
        for fmt in (:text, :bibtex)
            r = refs(ar; format=fmt)
            @test r isa String
            @test !isempty(r)
        end
    end

    @testset "refs.jl — BaiPerronResult" begin
        rng = Random.MersenneTwister(9052)
        y = randn(rng, 200)
        X = hcat(ones(200), randn(rng, 200))
        bp = bai_perron_test(y, X; max_breaks=2, trimming=0.15)
        r = refs(bp)
        @test occursin("Bai", r)
    end

    @testset "refs.jl — PANICResult" begin
        rng = Random.MersenneTwister(9053)
        X = randn(rng, 100, 10)
        pr = panic_test(X; r=2)
        for fmt in (:text, :html)
            r = refs(pr; format=fmt)
            @test r isa String
            @test !isempty(r)
        end
    end

    @testset "refs.jl — PesaranCIPSResult" begin
        rng = Random.MersenneTwister(9054)
        X = randn(rng, 100, 10)
        pc = pesaran_cips_test(X; lags=1)
        r = refs(pc)
        @test occursin("Pesaran", r)
    end

    @testset "refs.jl — MoonPerronResult" begin
        rng = Random.MersenneTwister(9055)
        X = randn(rng, 100, 10)
        mp = moon_perron_test(X; r=2)
        r = refs(mp)
        @test occursin("Moon", r)
    end

    @testset "refs.jl — FactorBreakResult" begin
        rng = Random.MersenneTwister(9056)
        X = randn(rng, 200, 20)
        fb = factor_break_test(X, 3)
        for fmt in (:text, :latex, :bibtex, :html)
            r = refs(fb; format=fmt)
            @test r isa String
            @test !isempty(r)
        end
    end

    @testset "refs.jl — FactorModel refs" begin
        rng = Random.MersenneTwister(9057)
        X = randn(rng, 200, 20)
        fm = estimate_factors(X, 3)
        r = refs(fm)
        @test occursin("Bai", r) || occursin("Stock", r)
    end

    @testset "refs.jl — LP types refs" begin
        rng = Random.MersenneTwister(9058)
        Y = randn(rng, 100, 3)
        lp = estimate_lp(Y, 1, 8; lags=2)
        r = refs(lp)
        @test occursin("Jord", r)

        slp = structural_lp(Y, 8; method=:cholesky, lags=2)
        r2 = refs(slp)
        @test r2 isa String
    end

    @testset "refs.jl — Volatility refs" begin
        rng = Random.MersenneTwister(9059)
        y = randn(rng, 300)
        arch_m = estimate_arch(y, 1)
        garch_m = estimate_garch(y, 1, 1)
        egarch_m = estimate_egarch(y, 1, 1)
        gjr_m = estimate_gjr_garch(y, 1, 1)

        @test occursin("Engle", refs(arch_m))
        @test occursin("Bollerslev", refs(garch_m))
        @test occursin("Nelson", refs(egarch_m))
        @test occursin("Glosten", refs(gjr_m))
    end

    @testset "refs.jl — Filter refs" begin
        rng = Random.MersenneTwister(9060)
        y = cumsum(randn(rng, 200))
        @test !isempty(refs(hp_filter(y)))
        @test !isempty(refs(hamilton_filter(y)))
        @test !isempty(refs(beveridge_nelson(y)))
        @test !isempty(refs(baxter_king(y)))
        @test !isempty(refs(boosted_hp(y)))
    end

    @testset "refs.jl — DSGE type refs" begin
        spec = @dsge begin
            parameters: rho = 0.9
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        r = refs(sol)
        @test occursin("Sims", r)
        r2 = refs(spec)
        @test r2 isa String
    end

    @testset "refs.jl — OccBin refs via symbol" begin
        r = refs(:occbin)
        @test occursin("Guerrieri", r)
    end

    @testset "refs.jl — VECM refs" begin
        rng = Random.MersenneTwister(9062)
        Y = cumsum(randn(rng, 150, 3), dims=1)
        vecm = estimate_vecm(Y, 2; rank=1)
        r = refs(vecm)
        @test occursin("Johansen", r) || occursin("Engle", r)
    end

    @testset "refs.jl — Nowcast refs" begin
        rng = Random.MersenneTwister(9063)
        nM = 4; nQ = 1
        Y_nc = randn(rng, 100, nM + nQ)
        Y_nc[end, end] = NaN
        dfm_nc = nowcast_dfm(Y_nc, nM, nQ; r=2, p=1)
        r = refs(dfm_nc)
        @test r isa String
        @test !isempty(r)
    end

    @testset "refs.jl — DiD refs via symbol" begin
        r = refs(:callaway_santanna)
        @test occursin("Callaway", r)

        r2 = refs(:twfe)
        @test occursin("Goodman", r2) || occursin("Bacon", r2)
    end

    @testset "refs.jl — ARIMA refs" begin
        rng = Random.MersenneTwister(9064)
        y = randn(rng, 200)
        ar = estimate_ar(y, 2)
        r = refs(ar)
        @test occursin("Box", r)
    end

    @testset "refs.jl — Granger refs" begin
        r = refs(:granger)
        @test occursin("Granger", r)
    end

    @testset "refs.jl — _format_ref with book and article" begin
        # Article
        ref_article = M._REFERENCES[:sims1980]
        io = IOBuffer()
        M._format_ref_text(io, ref_article)
        @test occursin("Sims", String(take!(io)))

        io = IOBuffer()
        M._format_ref_latex(io, ref_article)
        @test occursin("\\bibitem", String(take!(io)))

        io = IOBuffer()
        M._format_ref_bibtex(io, ref_article)
        @test occursin("@article", String(take!(io)))

        io = IOBuffer()
        M._format_ref_html(io, ref_article)
        @test occursin("<p>", String(take!(io)))

        # Book
        ref_book = M._REFERENCES[:lutkepohl2005]
        io = IOBuffer()
        M._format_ref_text(io, ref_book)
        out = String(take!(io))
        @test occursin("Springer", out)
        @test occursin("ISBN", out)

        io = IOBuffer()
        M._format_ref_latex(io, ref_book)
        @test occursin("\\textit{", String(take!(io)))

        io = IOBuffer()
        M._format_ref_bibtex(io, ref_book)
        out = String(take!(io))
        @test occursin("@book", out)
        @test occursin("publisher", out)

        io = IOBuffer()
        M._format_ref_html(io, ref_book)
        @test occursin("<em>", String(take!(io)))
    end

    @testset "refs.jl — _format_ref dispatcher" begin
        ref = M._REFERENCES[:sims1980]
        io = IOBuffer()
        M._format_ref(io, ref, :text)
        @test !isempty(String(take!(io)))

        io = IOBuffer()
        M._format_ref(io, ref, :latex)
        @test !isempty(String(take!(io)))

        io = IOBuffer()
        M._format_ref(io, ref, :bibtex)
        @test !isempty(String(take!(io)))

        io = IOBuffer()
        M._format_ref(io, ref, :html)
        @test !isempty(String(take!(io)))

        @test_throws ArgumentError M._format_ref(IOBuffer(), ref, :unknown)
    end

    @testset "refs.jl — incollection entry type" begin
        # Test a reference with :incollection entry type
        ref = M._REFERENCES[:gretton2005]
        io = IOBuffer()
        M._format_ref_bibtex(io, ref)
        out = String(take!(io))
        @test occursin("@incollection", out)
        @test occursin("booktitle", out)
    end

    @testset "refs.jl — error handling" begin
        @test_throws ArgumentError refs(:nonexistent_method_xyz)
        @test_throws ArgumentError M._format_ref(IOBuffer(), M._REFERENCES[:sims1980], :badformat)
    end

    @testset "refs.jl — Panel VAR refs" begin
        r = refs(:pvar)
        @test occursin("Holtz-Eakin", r) || occursin("Arellano", r)
    end

    @testset "refs.jl — GMM/SMM refs" begin
        rng = Random.MersenneTwister(9065)
        data_gmm = randn(rng, 200, 3)
        g = (theta, data) -> data[:, 2:3] .* (data[:, 1] .- theta[1])
        gmm_m = estimate_gmm(g, [0.0], data_gmm)
        r = refs(gmm_m)
        @test occursin("Hansen", r)
    end

    @testset "refs.jl — BayesianDSGE refs via symbol" begin
        r = refs(:estimate_dsge_bayes)
        @test occursin("Herbst", r) || occursin("Schorfheide", r)
    end

    @testset "refs.jl — Normality test refs" begin
        r = refs(:jarque_bera)
        @test occursin("Jarque", r)

        r2 = refs(:doornik_hansen)
        @test occursin("Doornik", r2)
    end

end  # Display Coverage
