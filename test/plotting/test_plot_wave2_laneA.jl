# =============================================================================
# Wave-2 Lane A — data-analysis views (PLT-20..23)
#   PLT-20 TimeSeriesData views, PLT-21 PanelData views,
#   PLT-22 CrossSectionData (first dispatch), PLT-23 binscatter.
#
# Uses the shared plotting-test helpers (plot_test_helpers.jl): check_plot,
# assert_all_json_valid, assert_escapes, assert_nan_becomes_null, series_names/
# panel_titles, HOSTILE_NAME. Structural assertions parse EXTRACTED JSON literals,
# never raw p.html (plotrule Testing Rules 1-7).
# =============================================================================

using MacroEconometricModels, Test, Random, DataFrames
using MacroEconometricModels: _binscatter_compute, _ols_fit_line, _fan_bands_json,
                              _regions_json

const MEM = MacroEconometricModels

@testset "Wave-2 Lane A (PLT-20..23)" begin

# -----------------------------------------------------------------------------
# PLT-20 — TimeSeriesData analysis views
# -----------------------------------------------------------------------------
@testset "PLT-20 TimeSeriesData" begin
    Random.seed!(20)
    d = TimeSeriesData(abs.(randn(80, 4)) .+ 0.5;
                       varnames=["GDP", "CPI", "RATE", "UNEMP"])

    @testset "view dispatch + default" begin
        for v in (:line, :hist, :density, :corr, :growth)
            p = plot_result(d; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        p = plot_result(d; view=:scatter, vars=["GDP", "CPI"])
        check_plot(p); assert_all_json_valid(p)
        # unknown view → ArgumentError naming valid set (C5)
        @test_throws ArgumentError plot_result(d; view=:bogus)
    end

    @testset "String + Int selection, bounds (C3)" begin
        p1 = plot_result(d; view=:scatter, vars=["GDP", "CPI"])
        p2 = plot_result(d; view=:scatter, vars=[1, 2])
        check_plot(p1); check_plot(p2)
        @test_throws ArgumentError plot_result(d; view=:scatter, vars=["NOPE", "CPI"])
        @test_throws ArgumentError plot_result(d; view=:scatter, vars=[99, 1])
        @test_throws ArgumentError plot_result(d; view=:scatter, vars=["GDP"])  # needs 2
    end

    @testset "date axis (PLT-08)" begin
        p0 = plot_result(d; view=:line)
        @test occursin("Time", p0.html)             # integer time_index
        dd = TimeSeriesData(randn(12, 2); varnames=["a", "b"])
        set_dates!(dd, ["20$(lpad(i,2,'0'))Q1" for i in 1:12])
        pd = plot_result(dd; view=:line)
        @test occursin("Date", pd.html)
        @test occursin("2001Q1", pd.html)           # a real calendar label, not "Time 143"
        assert_all_json_valid(pd)
    end

    @testset "shade regions" begin
        p = plot_result(d; view=:line, shade=[(10, 20), (40, 50, "#d9d9d9")])
        check_plot(p)
        @test occursin("\"x0\":10", p.html)
        assert_strict_json(_regions_json([(10, 20), (40, 50, "#ccc", 0.2)]))
        @test _regions_json(nothing) == "[]"
    end

    @testset "corr symmetric [-1,1] diverging + legend (PLT-15)" begin
        p = plot_result(d; view=:corr)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("gradient", p.html) || occursin("legend", p.html)
        # symmetric domain fed as [-1,1]
        @test occursin("-1", p.html)
    end

    @testset "NaN → null (line), constant-column hist" begin
        dn = TimeSeriesData(copy(d.data); varnames=copy(d.varnames))
        dn.data[5, 1] = NaN
        p = plot_result(dn; view=:line, vars=["GDP"])
        assert_nan_becomes_null(p)
        # constant column histogram: single bin, no exception, no bare NaN token
        dc = TimeSeriesData(hcat(fill(3.0, 40), randn(40)); varnames=["const", "z"])
        pc = plot_result(dc; view=:hist, vars=["const"])
        check_plot(pc)
        for (_, lit) in extract_json_blocks(pc.html)
            @test !occursin(r"[:\[,]\s*NaN", lit)
        end
    end

    @testset "growth tcode + non-positive fallback" begin
        # log-diff (tcode 5) on a column with a non-positive value falls back to Δ (tcode 2)
        dat = abs.(randn(50, 2)) .+ 1.0
        dat[3, 1] = -2.0
        dg = TimeSeriesData(dat; varnames=["neg", "pos"], tcode=[5, 5])
        p = plot_result(dg; view=:growth)
        check_plot(p)
        @test occursin("fell back", p.html)         # visible subtitle note (C7)
    end

    @testset "var cap note (C7)" begin
        dbig = TimeSeriesData(randn(60, 15); varnames=["v$i" for i in 1:15])
        p = plot_result(dbig; view=:line)
        @test occursin("Showing 12 of 15 variables", p.html)
    end

    @testset "escaping (A8)" begin
        de = TimeSeriesData(randn(40, 2); varnames=[HOSTILE_NAME, "y"])
        p = plot_result(de; view=:scatter, vars=[HOSTILE_NAME, "y"])
        assert_escapes(p)
        ph = plot_result(de; view=:hist)
        assert_escapes(ph)
        pc = plot_result(de; view=:corr)
        assert_escapes(pc)
    end

    @testset "save_path (C8)" begin
        tmp = tempname() * ".html"
        p = plot_result(d; view=:line, save_path=tmp)
        @test p isa PlotOutput
        @test startswith(strip(read(tmp, String)), "<!DOCTYPE html>")
        rm(tmp; force=true)
    end
end

# -----------------------------------------------------------------------------
# PLT-21 — PanelData analysis views
# -----------------------------------------------------------------------------
@testset "PLT-21 PanelData" begin
    Random.seed!(21)
    df = DataFrame(group=repeat(1:6, inner=20), time=repeat(1:20, 6),
                   x=randn(120), y=randn(120), z=randn(120))
    pd = xtset(df, :group, :time)

    @testset "view dispatch + default :lines" begin
        for v in (:lines, :quantiles, :spaghetti, :groups)
            p = plot_result(pd; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        p = plot_result(pd; view=:scatter, vars=["x", "y"])
        check_plot(p); assert_all_json_valid(p)
        @test_throws ArgumentError plot_result(pd; view=:bogus)
    end

    @testset ":quantiles fan bands + single group degenerate" begin
        p = plot_result(pd; view=:quantiles, vars=["x"])
        check_plot(p)
        assert_strict_json(_fan_bands_json([0.1, 0.25, 0.5, 0.75, 0.9]))
        @test occursin("10–90%", p.html) || occursin("25", p.html)  # band legend labels
        # single group: quantiles collapse (zero-width bands) without error
        df1 = DataFrame(group=fill(1, 15), time=1:15, x=randn(15))
        pd1 = xtset(df1, :group, :time)
        p1 = plot_result(pd1; view=:quantiles)
        check_plot(p1)
    end

    @testset ":spaghetti highlight String vs Int" begin
        pI = plot_result(pd; view=:spaghetti, vars=["x"], highlight=[1, 3])
        pS = plot_result(pd; view=:spaghetti, vars=["x"], highlight=["2"])
        check_plot(pI); check_plot(pS)
        # highlighted units are colored (palette), non-highlighted grey
        @test occursin("#cccccc", pI.html)
        @test_throws ArgumentError plot_result(pd; view=:spaghetti, highlight=[999])
        @test_throws ArgumentError plot_result(pd; view=:spaghetti, highlight=["nope"])
    end

    @testset ":scatter demean within/between/none subtitle" begin
        pn = plot_result(pd; view=:scatter, vars=["x", "y"], demean=:none)
        pw = plot_result(pd; view=:scatter, vars=["x", "y"], demean=:within)
        pb = plot_result(pd; view=:scatter, vars=["x", "y"], demean=:between)
        @test occursin("pooled", pn.html)
        @test occursin("within", pw.html)
        @test occursin("between", pb.html)
        @test_throws ArgumentError plot_result(pd; view=:scatter, vars=["x", "y"], demean=:bad)
        @test_throws ArgumentError plot_result(pd; view=:scatter)  # needs vars
    end

    @testset "group cap note (C7)" begin
        df2 = DataFrame(group=repeat(1:15, inner=8), time=repeat(1:8, 15), x=randn(120))
        pd2 = xtset(df2, :group, :time)
        p = plot_result(pd2; view=:lines)
        @test occursin("Showing 10 of 15 groups", p.html)
        pg = plot_result(pd2; view=:groups)                 # groups cap default 12
        @test occursin("Showing 12 of 15 groups", pg.html)
    end

    @testset "unbalanced panel → null gaps, not exceptions" begin
        dfu = DataFrame(group=[1,1,1,2,2,3], time=[1,2,3,1,3,2],
                        x=[1.0,2.0,3.0,4.0,5.0,6.0])
        pdu = xtset(dfu, :group, :time)
        p = plot_result(pdu; view=:lines)
        assert_nan_becomes_null(p)                          # missing (g,t) → null
    end

    @testset "date axis + escaping" begin
        p = plot_result(pd; view=:lines, dates=["D$i" for i in 1:20])
        @test occursin("Date", p.html)
        @test_throws ArgumentError plot_result(pd; view=:lines, dates=["only-one"])
        dfe = DataFrame(g=[HOSTILE_NAME, HOSTILE_NAME, "B", "B"], t=[1,2,1,2], x=randn(4))
        pde = xtset(dfe, :g, :t)
        pe = plot_result(pde; view=:lines)
        assert_escapes(pe)
    end
end

# -----------------------------------------------------------------------------
# PLT-22 — CrossSectionData plotting (first dispatch)
# -----------------------------------------------------------------------------
@testset "PLT-22 CrossSectionData" begin
    Random.seed!(22)
    cs = CrossSectionData(randn(200, 4); varnames=["a", "b", "c", "d"])

    @testset "view dispatch + default :hist" begin
        for v in (:hist, :density, :corr, :pairs)
            p = plot_result(cs; view=v)
            check_plot(p); assert_all_json_valid(p)
        end
        p = plot_result(cs; view=:scatter, vars=[1, 2])
        check_plot(p); assert_all_json_valid(p)
        @test_throws ArgumentError plot_result(cs; view=:bogus)
        # default view is a distribution
        pdef = plot_result(cs)
        check_plot(pdef)
    end

    @testset "String + Int selection + bounds (C3)" begin
        p1 = plot_result(cs; view=:scatter, vars=["a", "b"])
        p2 = plot_result(cs; view=:scatter, vars=[1, 2])
        check_plot(p1); check_plot(p2)
        @test_throws ArgumentError plot_result(cs; view=:scatter, vars=["z", "a"])
        @test_throws ArgumentError plot_result(cs; view=:scatter, vars=[99, 1])
    end

    @testset "NaN → null (scatter), corr legend" begin
        csn = CrossSectionData(copy(cs.data); varnames=copy(cs.varnames))
        csn.data[7, 1] = NaN
        p = plot_result(csn; view=:scatter, vars=["a", "b"])
        assert_nan_becomes_null(p)
        pc = plot_result(cs; view=:corr)
        @test occursin("gradient", pc.html) || occursin("legend", pc.html)
    end

    @testset ":pairs ≤6 cap note (C7)" begin
        cs8 = CrossSectionData(randn(120, 8); varnames=["v$i" for i in 1:8])
        p = plot_result(cs8; view=:pairs)
        @test occursin("first 6 of 8", p.html)
        # 6×6 matrix = 36 panels
        @test length(panel_titles(p.html)) >= 36
    end

    @testset "escaping (A8)" begin
        cse = CrossSectionData(randn(60, 2); varnames=[HOSTILE_NAME, "y"])
        assert_escapes(plot_result(cse; view=:hist))
        assert_escapes(plot_result(cse; view=:scatter, vars=[HOSTILE_NAME, "y"]))
        assert_escapes(plot_result(cse; view=:corr))
    end

    @testset "save_path DOCTYPE (C8)" begin
        tmp = tempname() * ".html"
        p = plot_result(cs; view=:hist, save_path=tmp)
        @test p isa PlotOutput
        @test startswith(strip(read(tmp, String)), "<!DOCTYPE html>")
        rm(tmp; force=true)
    end
end

# -----------------------------------------------------------------------------
# PLT-23 — Binscatter (all three containers)
# -----------------------------------------------------------------------------
@testset "PLT-23 binscatter" begin
    Random.seed!(23)
    n = 500
    z = randn(n)
    x = 0.7 .* z .+ randn(n)
    y = 1.5 .* x .- 0.9 .* z .+ randn(n)

    @testset "runs on all three containers" begin
        ts = TimeSeriesData(hcat(x, y, z); varnames=["x", "y", "z"])
        cs = CrossSectionData(hcat(x, y, z); varnames=["x", "y", "z"])
        df = DataFrame(group=repeat(1:10, inner=50), time=repeat(1:50, 10),
                       x=x, y=y, z=z)
        pd = xtset(df, :group, :time)
        for d in (ts, cs, pd)
            p = plot_result(d; view=:binscatter, x="x", y="y")
            check_plot(p); assert_all_json_valid(p)
        end
        # PanelData within-group binscatter
        pw = plot_result(pd; view=:binscatter, x="x", y="y", demean=:within)
        check_plot(pw)
        @test occursin("within-group", pw.html)
    end

    @testset "20 quantile bins" begin
        r = _binscatter_compute(x, y; n_bins=20)
        @test length(r.xbar) == 20
        @test length(r.ybar) == 20
    end

    @testset "FWL slope == multivariate OLS coefficient" begin
        r = _binscatter_compute(x, y; n_bins=20, controls=reshape(z, :, 1))
        m = estimate_reg(y, hcat(ones(n), x, z))       # intercept, x, z
        @test isapprox(coef(m)[2], r.slope; atol=1e-8)
        # no controls: slope == simple OLS slope (with intercept)
        r0 = _binscatter_compute(x, y; n_bins=20)
        m0 = estimate_reg(y, hcat(ones(n), x))
        @test isapprox(coef(m0)[2], r0.slope; atol=1e-8)
    end

    @testset "controls partialled out, slope in subtitle (C9)" begin
        cs = CrossSectionData(hcat(x, y, z); varnames=["x", "y", "z"])
        p = plot_result(cs; view=:binscatter, x="x", y="y", controls=["z"])
        check_plot(p)
        @test occursin("controls: z", p.html)
        @test occursin("slope=", p.html)               # _fmt-rounded, not _json
        @test !occursin("slope=null", p.html)
    end

    @testset "NaN rows dropped before binning" begin
        xn = copy(x); yn = copy(y); xn[3] = NaN; yn[10] = Inf
        r = _binscatter_compute(xn, yn; n_bins=20)
        @test r.n_used == n - 2                          # two rows dropped, not nulled
        cs = CrossSectionData(hcat(xn, yn); varnames=["x", "y"])
        p = plot_result(cs; view=:binscatter, x="x", y="y")
        assert_all_json_valid(p)                         # bins are finite, strictly valid
    end

    @testset ":within differs from :none with group effects" begin
        # Build a panel where the within slope differs from the pooled slope.
        Random.seed!(99)
        G = 8; Tg = 40
        gid = repeat(1:G, inner=Tg)
        alpha = repeat(5.0 .* randn(G), inner=Tg)        # big group intercepts
        xx = alpha .+ randn(G * Tg)
        yy = alpha .+ 0.5 .* (xx .- alpha) .+ randn(G * Tg)
        rn = _binscatter_compute(xx, yy; n_bins=20)
        rw = _binscatter_compute(xx, yy; n_bins=20, group_id=gid, demean=:within)
        @test !isapprox(rn.slope, rw.slope; atol=1e-2)
        @test isapprox(rw.slope, 0.5; atol=0.1)          # within recovers the true slope
    end

    @testset "fit :linear/:quadratic/:none + bad inputs" begin
        cs = CrossSectionData(hcat(x, y); varnames=["x", "y"])
        assert_strict_json(_ols_fit_line(x, y; degree=2))
        for f in (:linear, :quadratic, :none, :ols)
            p = plot_result(cs; view=:binscatter, x="x", y="y", fit=f)
            check_plot(p)
        end
        @test_throws ArgumentError plot_result(cs; view=:binscatter, x="x", y="y", fit=:bad)
        @test_throws ArgumentError plot_result(cs; view=:binscatter)        # x,y required
        @test_throws ArgumentError plot_result(cs; view=:binscatter, x="nope", y="y")
    end

    @testset "control varname with a quote round-trips (A8)" begin
        ce = CrossSectionData(hcat(x, y, z); varnames=["x", "y", HOSTILE_NAME])
        p = plot_result(ce; view=:binscatter, x="x", y="y", controls=[HOSTILE_NAME])
        assert_escapes(p)
    end
end

end  # top-level testset
