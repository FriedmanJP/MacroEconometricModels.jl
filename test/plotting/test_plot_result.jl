# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

using Test, Random, DataFrames

@testset "Plotting — plot_result()" begin
    Random.seed!(42)

    # =========================================================================
    # Helper: check PlotOutput validity
    # =========================================================================
    function check_plot(p::PlotOutput; min_size=500, check_d3=true)
        @test p isa PlotOutput
        @test length(p.html) >= min_size
        @test occursin("<!DOCTYPE html>", p.html)
        # D3 creates <svg> elements at runtime; check for the JS creation pattern
        @test occursin("append('svg')", p.html)
        if check_d3
            @test occursin("d3.min.js", p.html)
        end
    end

    # =========================================================================
    # IRF types
    # =========================================================================
    @testset "ImpulseResponse" begin
        Y = randn(100, 3)
        m = estimate_var(Y, 2)
        r = irf(m, 10; ci_type=:bootstrap, reps=50)

        # Full grid
        p = plot_result(r)
        check_plot(p)

        # Single var+shock
        p2 = plot_result(r; var=1, shock=1)
        check_plot(p2)

        # String-based selection
        p3 = plot_result(r; var="y1", shock="y2")
        check_plot(p3)
        @test occursin("y1", p3.html)
        @test occursin("y2", p3.html)

        # Custom title
        p4 = plot_result(r; title="My Custom IRF")
        @test occursin("My Custom IRF", p4.html)

        # save_path
        tmpfile = tempname() * ".html"
        p5 = plot_result(r; var=1, shock=1, save_path=tmpfile)
        @test isfile(tmpfile)
        @test filesize(tmpfile) > 0
        rm(tmpfile)
    end

    @testset "BayesianImpulseResponse" begin
        Y = randn(100, 3)
        post = estimate_bvar(Y, 2; n_draws=100)
        r = irf(post, 10)
        p = plot_result(r)
        check_plot(p)
        @test occursin("Bayesian", p.html)

        # Single var
        p2 = plot_result(r; var=1, shock=1)
        check_plot(p2)
    end

    @testset "LPImpulseResponse" begin
        Y = randn(100, 3)
        lp = estimate_lp(Y, 1, 10; lags=2)
        r = lp_irf(lp)
        p = plot_result(r)
        check_plot(p)
        @test occursin("LP", p.html)

        p2 = plot_result(r; var=1)
        check_plot(p2)
    end

    @testset "StructuralLP" begin
        Y = randn(100, 3)
        slp = structural_lp(Y, 10; method=:cholesky, lags=2)
        p = plot_result(slp)
        check_plot(p)
        @test occursin("Structural LP", p.html)
    end

    # =========================================================================
    # FEVD types
    # =========================================================================
    @testset "FEVD" begin
        Y = randn(100, 3)
        m = estimate_var(Y, 2)
        f = fevd(m, 10)
        p = plot_result(f)
        check_plot(p)
        @test occursin("FEVD", p.html) || occursin("Variance", p.html)

        p2 = plot_result(f; var=1)
        check_plot(p2)
    end

    @testset "BayesianFEVD" begin
        Y = randn(100, 3)
        post = estimate_bvar(Y, 2; n_draws=100)
        f = fevd(post, 10)
        p = plot_result(f)
        check_plot(p)
        @test occursin("Bayesian", p.html)
    end

    @testset "LPFEVD" begin
        Y = randn(100, 3)
        slp = structural_lp(Y, 10; method=:cholesky, lags=2)
        f = lp_fevd(slp, 10)
        p = plot_result(f)
        check_plot(p)
    end

    # =========================================================================
    # Historical Decomposition
    # =========================================================================
    @testset "HistoricalDecomposition" begin
        Y = randn(100, 3)
        m = estimate_var(Y, 2)
        T_eff = size(m.Y, 1) - m.p
        hd_res = historical_decomposition(m, T_eff)
        p = plot_result(hd_res)
        check_plot(p)
        @test occursin("Historical Decomposition", p.html)

        p2 = plot_result(hd_res; var=1)
        check_plot(p2)
    end

    @testset "BayesianHistoricalDecomposition" begin
        Y = randn(100, 3)
        post = estimate_bvar(Y, 2; n_draws=100)
        T_eff = size(Y, 1) - 2
        hd_res = historical_decomposition(post, T_eff)
        p = plot_result(hd_res)
        check_plot(p)
        @test occursin("Bayesian", p.html)
    end

    # =========================================================================
    # Filters
    # =========================================================================
    @testset "HPFilterResult" begin
        y = cumsum(randn(200))
        r = hp_filter(y)
        p = plot_result(r)
        check_plot(p)
        @test occursin("Hodrick-Prescott", p.html)
    end

    @testset "HamiltonFilterResult" begin
        y = cumsum(randn(200))
        r = hamilton_filter(y)
        p = plot_result(r; original=y)
        check_plot(p)
        @test occursin("Hamilton", p.html)
    end

    @testset "BeveridgeNelsonResult" begin
        y = cumsum(randn(200))
        r = beveridge_nelson(y)
        p = plot_result(r)
        check_plot(p)
        @test occursin("Beveridge-Nelson", p.html)
    end

    @testset "BaxterKingResult" begin
        y = cumsum(randn(200))
        r = baxter_king(y)
        p = plot_result(r)
        check_plot(p)
        @test occursin("Baxter-King", p.html)
    end

    @testset "BoostedHPResult" begin
        y = cumsum(randn(200))
        r = boosted_hp(y)
        p = plot_result(r)
        check_plot(p)
        @test occursin("Boosted HP", p.html)
    end

    # =========================================================================
    # Forecasts
    # =========================================================================
    @testset "ARIMAForecast" begin
        y = randn(200)
        ar = estimate_ar(y, 2)
        fc = forecast(ar, 20)

        # Without history
        p = plot_result(fc)
        check_plot(p)

        # With history
        p2 = plot_result(fc; history=y, n_history=30)
        check_plot(p2)
    end

    @testset "VolatilityForecast" begin
        y = randn(300)
        gm = estimate_garch(y, 1, 1)
        fc = forecast(gm, 10)
        p = plot_result(fc)
        check_plot(p)

        # With history
        p2 = plot_result(fc; history=gm.conditional_variance)
        check_plot(p2)
    end

    @testset "VECMForecast" begin
        Y = cumsum(randn(150, 3), dims=1)
        vecm_m = estimate_vecm(Y, 2; rank=1)
        fc = forecast(vecm_m, 10)
        p = plot_result(fc)
        check_plot(p)

        p2 = plot_result(fc; var=1)
        check_plot(p2)
    end

    @testset "FactorForecast" begin
        X = randn(200, 20)
        fm = estimate_dynamic_factors(X, 2, 1)
        fc = forecast(fm, 10)

        p = plot_result(fc)
        check_plot(p)

        p2 = plot_result(fc; type=:observable, var=1)
        check_plot(p2)
    end

    @testset "LPForecast" begin
        Y = randn(100, 3)
        lp = estimate_lp(Y, 1, 10; lags=2)
        shock_path = zeros(10); shock_path[1] = 1.0
        fc = forecast(lp, shock_path)
        p = plot_result(fc)
        check_plot(p)
    end

    # =========================================================================
    # Models
    # =========================================================================
    @testset "ARCHModel" begin
        y = randn(300)
        m = estimate_arch(y, 2)
        p = plot_result(m)
        check_plot(p)
        @test occursin("ARCH", p.html)
    end

    @testset "GARCHModel" begin
        y = randn(500)
        m = estimate_garch(y, 1, 1)
        p = plot_result(m)
        check_plot(p)
        @test occursin("GARCH", p.html)
    end

    @testset "EGARCHModel" begin
        y = randn(300)
        m = estimate_egarch(y, 1, 1)
        p = plot_result(m)
        check_plot(p)
        @test occursin("EGARCH", p.html)
    end

    @testset "GJRGARCHModel" begin
        y = randn(300)
        m = estimate_gjr_garch(y, 1, 1)
        p = plot_result(m)
        check_plot(p)
        @test occursin("GJR-GARCH", p.html)
    end

    @testset "SVModel" begin
        y = randn(200)
        m = estimate_sv(y; n_samples=100, burnin=50)
        p = plot_result(m)
        check_plot(p)
        @test occursin("Stochastic Volatility", p.html)
    end

    @testset "FactorModel" begin
        X = randn(200, 20)
        fm = estimate_factors(X, 3)
        p = plot_result(fm)
        check_plot(p)
        @test occursin("Factor", p.html)
    end

    @testset "DynamicFactorModel" begin
        X = randn(200, 20)
        fm = estimate_dynamic_factors(X, 2, 1)
        p = plot_result(fm)
        check_plot(p)
        @test occursin("Dynamic", p.html)
    end

    @testset "TimeSeriesData" begin
        d = TimeSeriesData(randn(100, 3); varnames=["GDP", "CPI", "RATE"])
        p = plot_result(d)
        check_plot(p)
        @test occursin("GDP", p.html)

        # Subset selection
        p2 = plot_result(d; vars=["GDP", "CPI"])
        check_plot(p2)
    end

    @testset "PanelData" begin
        df = DataFrame(group=repeat(1:3, inner=20), time=repeat(1:20, 3),
            x=randn(60), y=randn(60))
        pd = xtset(df, :group, :time)
        p = plot_result(pd)
        check_plot(p)
    end

    # =========================================================================
    # Nowcasting
    # =========================================================================
    @testset "NowcastResult" begin
        nM = 4; nQ = 1
        Y_nc = randn(100, nM + nQ)
        Y_nc[end, end] = NaN
        dfm_nc = nowcast_dfm(Y_nc, nM, nQ; r=2, p=1)
        nr = nowcast(dfm_nc)
        p = plot_result(nr)
        check_plot(p)
        @test occursin("Nowcast", p.html)
    end

    @testset "NowcastNews" begin
        X_old = randn(100, 5)
        X_old[end, end] = NaN
        X_new = copy(X_old)
        X_new[end, end] = 0.5
        dfm2 = nowcast_dfm(X_old, 4, 1; r=2, p=1)
        nn = nowcast_news(X_new, X_old, dfm2, 5)
        p = plot_result(nn)
        check_plot(p)
        @test occursin("News", p.html) || occursin("news", p.html)
    end

    # =========================================================================
    # Infrastructure
    # =========================================================================
    @testset "save_plot" begin
        y = randn(200)
        r = hp_filter(y)
        p = plot_result(r)

        tmpfile = tempname() * ".html"
        result_path = save_plot(p, tmpfile)
        @test result_path == tmpfile
        @test isfile(tmpfile)
        content = read(tmpfile, String)
        @test occursin("d3.min.js", content)
        @test occursin("append('svg')", content)
        rm(tmpfile)
    end

    @testset "show methods" begin
        y = randn(200)
        r = hp_filter(y)
        p = plot_result(r)

        # text/plain
        io = IOBuffer()
        show(io, MIME"text/plain"(), p)
        s = String(take!(io))
        @test occursin("PlotOutput", s)
        @test occursin("bytes", s)

        # text/html
        io2 = IOBuffer()
        show(io2, MIME"text/html"(), p)
        s2 = String(take!(io2))
        @test occursin("<!DOCTYPE html>", s2)

        # Default show
        io3 = IOBuffer()
        show(io3, p)
        s3 = String(take!(io3))
        @test occursin("PlotOutput", s3)
    end

    # =========================================================================
    # Coverage — internal helpers and edge cases
    # =========================================================================
    @testset "_json edge cases" begin
        @test MacroEconometricModels._json(NaN) == "null"
        @test MacroEconometricModels._json(Inf) == "null"
        @test MacroEconometricModels._json(-Inf) == "null"
        @test MacroEconometricModels._json(nothing) == "null"
        @test MacroEconometricModels._json(missing) == "null"
        @test MacroEconometricModels._json(true) == "true"
        @test MacroEconometricModels._json(false) == "false"
        @test MacroEconometricModels._json(:test) == "\"test\""
        @test MacroEconometricModels._json(42) == "42"
        @test MacroEconometricModels._json(3.14) == "3.14"
        @test MacroEconometricModels._json("hello") == "\"hello\""
        @test MacroEconometricModels._json("he\"llo") == "\"he\\\"llo\""
        @test MacroEconometricModels._json("line\nnewline") == "\"line\\nnewline\""
        @test MacroEconometricModels._json([1,2,3]) == "[1,2,3]"
    end

    @testset "_json_obj and _json_array_of_objects" begin
        obj = MacroEconometricModels._json_obj([Pair("a","1"), Pair("b","\"x\"")])
        @test occursin("\"a\":1", obj)
        rows = [
            [Pair("x","1")],
            [Pair("x","2")]
        ]
        arr = MacroEconometricModels._json_array_of_objects(rows)
        @test startswith(arr, "[")
        @test endswith(arr, "]")
    end

    @testset "VARForecast plot" begin
        Y = randn(100, 3)
        m = estimate_var(Y, 2)
        fc = forecast(m, 10)
        p = plot_result(fc)
        check_plot(p)
        # With specific var
        p2 = plot_result(fc; var=1)
        check_plot(p2)
    end

    @testset "BVARForecast plot with point_estimate=:mean" begin
        Y = randn(100, 3)
        post = estimate_bvar(Y, 2; n_draws=100)
        fc = forecast(post, 10; point_estimate=:mean)
        p = plot_result(fc)
        check_plot(p)
        @test occursin("Posterior mean", p.html) || occursin("mean", p.html)
    end

    @testset "FactorForecast type=:factor plot" begin
        X = randn(200, 20)
        fm = estimate_dynamic_factors(X, 2, 1)
        fc = forecast(fm, 10)
        p = plot_result(fc; type=:factor)
        check_plot(p)
        @test occursin("Factor", p.html)
    end

    @testset "_resolve_var edge cases" begin
        names = ["GDP", "CPI", "RATE"]
        @test MacroEconometricModels._resolve_var(nothing, names) == 1
        @test MacroEconometricModels._resolve_var(2, names) == 2
        @test MacroEconometricModels._resolve_var("CPI", names) == 2
        @test_throws ArgumentError MacroEconometricModels._resolve_var("INVALID", names)
    end

    @testset "Hamilton and BaxterKing filters without original=" begin
        y = cumsum(randn(200))
        r = hamilton_filter(y)
        p_no_orig = plot_result(r)
        check_plot(p_no_orig)

        r2 = baxter_king(y)
        p_bk = plot_result(r2)
        check_plot(p_bk)
    end

    @testset "_make_plot with source and note" begin
        panel = MacroEconometricModels._PanelSpec("test_id", "Test Title", "// js code")
        p = MacroEconometricModels._make_plot([panel, panel]; title="Test", source="Source: Author", note="Note: test")
        @test occursin("Source: Author", p.html)
        @test occursin("Note: test", p.html)
    end

    @testset "IRF with ci_type=:none" begin
        Y = randn(100, 3)
        m = estimate_var(Y, 2)
        r = irf(m, 10; ci_type=:none)
        p = plot_result(r; var=1, shock=1)
        check_plot(p)
    end

    @testset "PlotOutput type" begin
        p = PlotOutput("<html></html>")
        @test p isa PlotOutput
        @test p.html == "<html></html>"
    end
end
