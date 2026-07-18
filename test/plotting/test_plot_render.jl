# =============================================================================
# Renderer + output-safety unit tests (PLT-01 vendored D3, PLT-02 escaping,
# PLT-03 embed-safe multi-plot). Renderer-option tests live here per plotrule
# Testing Rules ("Renderer changes additionally require one test per option").
# =============================================================================

using Test, Random, Dates
include(joinpath(@__DIR__, "plot_test_helpers.jl"))

const M = MacroEconometricModels

@testset "Plotting output safety (PLT-01/02/03)" begin

    # -------------------------------------------------------------------------
    # PLT-02 — escaping at all three sinks (A7/A8)
    # -------------------------------------------------------------------------
    @testset "PLT-02 _json (JSON / JS-string sink, A7)" begin
        @test M._json("a\nb")      == "\"a\\nb\""
        @test M._json("a\rb")      == "\"a\\rb\""
        @test M._json("a\tb")      == "\"a\\tb\""
        @test M._json("\x01")      == "\"\\u0001\""
        @test M._json("</script>") == "\"\\u003c/script>\""     # script terminator neutralized
        @test occursin("\\u003c", M._json("x<y"))
        @test !occursin("<", M._json("x<y"))                    # every '<' escaped
        @test M._json(string(Char(0x2028)))    == "\"\\u2028\""             # JS line separators
        @test M._json(string(Char(0x2029)))    == "\"\\u2029\""
        @test M._json("réal Δlog") == "\"réal Δlog\""           # unicode verbatim
        @test !occursin("</script>", M._json(HOSTILE_NAME))
        assert_strict_json("[" * M._json(HOSTILE_NAME) * "]")   # still strict JSON
    end

    @testset "PLT-02 _esc_html (HTML-text sink, A8)" begin
        @test M._esc_html("a&b")   == "a&amp;b"                 # '&' escaped first
        @test M._esc_html("x<y>z") == "x&lt;y&gt;z"
        @test M._esc_html("\"q\"") == "&quot;q&quot;"
        @test M._esc_html("it's")  == "it&#39;s"
        @test !occursin("<", M._esc_html(HOSTILE_NAME))
        @test !occursin("</script>", M._esc_html(HOSTILE_NAME))
    end

    @testset "PLT-02 _axis_labels_js (JS-string label sink, A8)" begin
        @test M._axis_labels_js("", "") == ""                   # empty labels omitted (Julia-side)
        out = M._axis_labels_js("Horizon", "Resp<onse")
        @test occursin(M._json("Horizon"), out)                 # label arrives as JSON literal
        @test occursin(M._json("Resp<onse"), out)
        @test !occursin("<onse", out)                           # '<' escaped in the JS string
        @test occursin("\\u003c", out)
        out2 = M._axis_labels_js("X", "Y"; g="g_7", w="w_7", h="h_7", yl_y="-99")
        @test occursin("g_7.append", out2)
        @test occursin("w_7/2", out2)
        @test occursin("-99", out2)
    end

    @testset "PLT-02 HTML-text sink escapes titles/source/note" begin
        panel = M._PanelSpec("panel_9", HOSTILE_NAME, "// js")
        body = M._render_body_single(panel; title=HOSTILE_NAME)
        @test !occursin("<c>", body)                            # figure + panel title escaped
        @test occursin("&lt;c&gt;", body)
        @test occursin("&quot;", body)                          # embedded " escaped
        fig = M._render_body_figure([panel, panel]; title=HOSTILE_NAME,
                                    source=HOSTILE_NAME, note=HOSTILE_NAME)
        @test !occursin("<c>", fig)
        @test occursin("&lt;/script&gt;", fig)                  # </script> neutralized in HTML sink
    end

    @testset "PLT-02 renderer escapes hostile series name + NaN→null" begin
        rows = [["x" => M._json(0), "y" => M._json(1.5)],
                ["x" => M._json(1), "y" => M._json(NaN)]]       # NaN → null
        data_json = M._json_array_of_objects(rows)
        series_json = "[" * M._json_obj([
            "name"  => M._json(HOSTILE_NAME),
            "color" => M._json("#1f77b4"),
            "key"   => M._json("y"),
            "dash"  => M._json("")]) * "]"
        js = M._render_line_js("panel_1", data_json, series_json;
                               xlabel=HOSTILE_NAME, ylabel="Value")
        @test !occursin("</script>", js)                        # no raw script terminator
        @test occursin("null", js)                              # NaN serialized to null
        @test !occursin(r"[:\[,]\s*NaN", js)
        for (_, lit) in extract_json_blocks(js)                 # embedded literals strict-valid
            assert_strict_json(lit)
        end
    end

    # -------------------------------------------------------------------------
    # PLT-01 — vendored, self-contained D3 (A12)
    # -------------------------------------------------------------------------
    @testset "PLT-01 self-contained vendored D3 (A12)" begin
        Y = randn(60, 2); m = estimate_var(Y, 2); r = irf(m, 6; ci_type=:none)
        p = plot_result(r; var=1, shock=1)
        @test !occursin("cdnjs", p.html)
        @test !occursin(r"src=\"https?://", p.html)
        @test !occursin(r"href=\"https?://", p.html)
        @test occursin("d3js.org", p.html)                      # inline D3 source banner
        @test occursin("Mike Bostock", p.html)                  # inline D3 source present
        assert_self_contained(p.html)
        @test !isempty(M._d3_source())                          # lazy Ref populated on first use
        @test length(M._d3_source()) > 200_000                  # full ~280 KB blob inlined
    end

    # -------------------------------------------------------------------------
    # PLT-03 — embed-safe, multi-plot-safe output (A11)
    # -------------------------------------------------------------------------
    @testset "PLT-03 embed-safe multi-plot (A11)" begin
        Y = randn(80, 2); m = estimate_var(Y, 2); r = irf(m, 8; ci_type=:none)
        p1 = plot_result(r; var=1, shock=1, title="Plot One")
        p2 = plot_result(r; var=2, shock=2, title="Plot Two")

        frag1 = sprint(show, MIME"text/html"(), p1)
        frag2 = sprint(show, MIME"text/html"(), p2)

        for frag in (frag1, frag2)                              # fragment has NO doc wrapper
            @test !occursin("<!DOCTYPE", frag)
            @test !occursin("<html", frag)
            @test !occursin("<head>", frag)
            @test !occursin("<body>", frag)
        end
        @test occursin("window.__mem_core", frag1)              # guarded shared core
        @test !occursin("const tooltip", frag1)                 # no redeclarable global const
        @test count(r"typeof window\.__mem_core === 'undefined'", frag1) == 1

        combined = frag1 * "\n" * frag2                         # two plots, one page
        @test occursin("Plot One", combined)
        @test occursin("Plot Two", combined)
        @test !occursin("const tooltip", combined)              # no duplicate-const collision
        ids = String[String(mm.captures[1])
                     for mm in eachmatch(r"<div id=\"([^\"]+)\"></div>", combined)]
        @test length(ids) >= 2
        @test length(unique(ids)) == length(ids)                # unique DOM ids (A11)

        tmp = tempname() * ".html"; save_plot(p1, tmp)          # saved file still standalone
        @test startswith(strip(read(tmp, String)), "<!DOCTYPE html>")
        rm(tmp; force=true)
        @test startswith(p1.html, "<!DOCTYPE html>")

        pd = PlotOutput("<html>x</html>")                       # empty fragment → full html
        @test sprint(show, MIME"text/html"(), pd) == "<html>x</html>"
    end

    # -------------------------------------------------------------------------
    # Shared helpers exercised end-to-end on a real plot (Rules 1,2,4,6)
    # -------------------------------------------------------------------------
    @testset "helpers on a real IRF plot" begin
        Y = randn(90, 3); m = estimate_var(Y, 2)
        r = irf(m, 10; ci_type=:bootstrap, reps=30)
        p = plot_result(r)
        check_plot(p)
        blocks = assert_all_json_valid(p)
        @test !isempty(blocks)
        @test series_count(p.html) >= 1
        assert_escapes(p)                                       # no hostile chars in a clean plot

        # Rule 4: a NaN in the data serializes to null, never a bare NaN token — and
        # the check scans only the embedded data literals (immune to the inlined D3).
        rows = [["x" => M._json(0), "y" => M._json(1.0)],
                ["x" => M._json(1), "y" => M._json(NaN)]]      # NaN → null
        dj = M._json_array_of_objects(rows)
        sj = "[" * M._json_obj(["name" => M._json("s"), "color" => M._json("#1f77b4"),
                                "key" => M._json("y"), "dash" => M._json("")]) * "]"
        pnan = M._make_plot([M._PanelSpec("p_nan", "NaN", M._render_line_js("p_nan", dj, sj))])
        assert_nan_becomes_null(pnan)
    end

    # -------------------------------------------------------------------------
    # strict-JSON validator self-check
    # -------------------------------------------------------------------------
    @testset "assert_strict_json catches dupes / NaN / undefined" begin
        @test_throws Exception _tj_parse_value("{\"a\":1,\"a\":2}", firstindex("{\"a\":1,\"a\":2}"))
        @test_throws Exception _tj_parse_value("[NaN]", firstindex("[NaN]"))
        @test_throws Exception _tj_parse_value("[Infinity]", firstindex("[Infinity]"))
        @test_throws Exception _tj_parse_value("[undefined]", firstindex("[undefined]"))
        v, _ = _tj_parse_value("[1,2,null]", firstindex("[1,2,null]"))
        @test v == Any[1.0, 2.0, nothing]
        o, _ = _tj_parse_value("{\"k\":\"v\\u003cx\"}", firstindex("{\"k\":\"v\\u003cx\"}"))
        @test o["k"] == "v<x"                                   # < decodes back to '<'
    end
end

# =============================================================================
# Renderer-option + dispatch tests for PLT-04..08 (plotrule Testing Rules:
# "Renderer changes additionally require one test per option").
# =============================================================================
@testset "Plotting renderer options + dispatch (PLT-04..08)" begin

    _line_sj() = "[" * M._json_obj(["name" => M._json("s"), "color" => M._json("#1f77b4"),
                                    "key" => M._json("y"), "dash" => M._json("")]) * "]"
    _line_dj(xs) = M._json_array_of_objects([["x" => M._json(x), "y" => M._json(float(x))] for x in xs])

    # -------------------------------------------------------------------------
    # PLT-05 — vertical (axis:"x") reference line on the line renderer
    # -------------------------------------------------------------------------
    @testset "PLT-05 line renderer axis:x vertical ref" begin
        dj = _line_dj(0:5); sj = _line_sj()
        jsx = M._render_line_js("p_vx", dj, sj;
                    ref_lines_json="[{\"value\":2,\"axis\":\"x\",\"color\":\"#d62728\"}]")
        @test occursin("(r.axis||'y') === 'x'", jsx)                       # branch present
        @test occursin(".attr('x1',x(r.value)).attr('x2',x(r.value))", jsx) # vertical draw
        @test occursin("if((r.axis||'y') !== 'x') allYVals.push(r.value)", jsx)  # x-ref not in y-domain
        # default (axis:"y") still draws a HORIZONTAL line
        jsy = M._render_line_js("p_vy", dj, sj; ref_lines_json="[{\"value\":0.5}]")
        @test occursin(".attr('y1',y(r.value)).attr('y2',y(r.value))", jsy)
        for (_, lit) in extract_json_blocks(jsx); assert_strict_json(lit); end
    end

    # -------------------------------------------------------------------------
    # PLT-05 — Bayesian FEVD/HD `stat` is a live kwarg (:mean/:median), title tracks it
    # -------------------------------------------------------------------------
    @testset "PLT-05 Bayesian FEVD/HD stat kwarg" begin
        H = 4; nv = 2; ns = 2; levels = [0.16, 0.5, 0.84]; nq = 3
        pe = zeros(H, nv, ns); pe[:, :, 1] .= 0.8; pe[:, :, 2] .= 0.2
        q  = zeros(H, nv, ns, nq); q[:, :, 1, :] .= 0.2; q[:, :, 2, :] .= 0.8
        f  = M.BayesianFEVD{Float64}(q, pe, H, ["a", "b"], ["s1", "s2"], levels)
        pm  = plot_result(f; stat=:mean)
        pmd = plot_result(f; stat=:median)
        @test occursin("posterior mean", pm.html)
        @test occursin("posterior median", pmd.html)
        @test pm.html != pmd.html                                    # statistic really switched
        @test_throws ArgumentError plot_result(f; stat=:bogus)
        f2 = M.BayesianFEVD{Float64}(q, pe, H, ["a", "b"], ["s1", "s2"], [0.16, 0.84, 0.9])
        @test_throws ArgumentError plot_result(f2; stat=:median)     # no 0.5 level

        Te = 5
        hq  = zeros(Te, nv, ns, nq); hq[:, :, 1, :] .= 0.3; hq[:, :, 2, :] .= 0.7
        hpe = zeros(Te, nv, ns); hpe[:, :, 1] .= 0.9; hpe[:, :, 2] .= 0.1
        iq  = zeros(Te, nv, nq); ipe = zeros(Te, nv); spe = zeros(Te, ns); act = zeros(Te, nv)
        hd  = M.BayesianHistoricalDecomposition{Float64}(hq, hpe, iq, ipe, spe, act,
                Te, ["a", "b"], ["s1", "s2"], levels, :cholesky)
        hm  = plot_result(hd; stat=:mean)
        hmd = plot_result(hd; stat=:median)
        @test occursin("posterior mean", hm.html)
        @test occursin("posterior median", hmd.html)
        @test hm.html != hmd.html
        @test_throws ArgumentError plot_result(hd; stat=:bogus)
    end

    # -------------------------------------------------------------------------
    # PLT-06 — horizontal bar orientation + log-scale option
    # -------------------------------------------------------------------------
    @testset "PLT-06 bar orientation=h + logscale" begin
        dj = M._json_array_of_objects([["x" => M._json("Alpha"), "impact" => M._json(1.2)],
                                       ["x" => M._json("Beta"),  "impact" => M._json(-0.6)]])
        sj = M._series_json(["Impact"], ["#1f77b4"]; keys=["impact"])
        jh = M._render_bar_js("p_h", dj, sj; mode="grouped", orientation="h", xlabel="Impact")
        @test occursin("d3.scaleBand().domain(data.map(d=>d.x)).range([0,h])", jh)  # y = band(names)
        @test occursin("d3.axisLeft(y)", jh)                                        # names on y-axis
        @test occursin("orientation = 'h'", jh)
        # default vertical: categories on the x-band (unchanged)
        jv = M._render_bar_js("p_v", dj, sj; mode="grouped")
        @test occursin("d3.scaleBand().domain(data.map(d=>d.x)).range([0,w])", jv)  # x = band(names)
        @test occursin("orientation = 'v'", jv)                                     # default vertical
        # log-scale value axis (for singular-value bars)
        jl = M._render_bar_js("p_l", dj, sj; mode="grouped", orientation="h", logscale=true)
        @test occursin("d3.scaleLog()", jl)
        for (_, lit) in extract_json_blocks(jh); assert_strict_json(lit); end
    end

    # -------------------------------------------------------------------------
    # PLT-04 — filter Original-line alignment (offset → calendar)
    # -------------------------------------------------------------------------
    @testset "PLT-04 filter Original alignment" begin
        tr = collect(1.0:5.0); cyc = fill(0.1, 5)
        orig = collect(10.0:16.0)                          # length 7 = n(5)+offset(2)
        dj = M._filter_data_json(tr, cyc; original=orig, offset=2)
        val, _ = _tj_parse_value(dj, firstindex(dj))
        @test val[1]["x"] == 3.0                            # first drawn x = offset+1
        @test val[1]["orig"] == orig[3]                     # aligned to calendar position 3
        # too-short original → ArgumentError (never silently shift/truncate)
        @test_throws ArgumentError M._filter_data_json(tr, cyc; original=collect(1.0:5.0), offset=2)

        # Hamilton with offset>0 and no original ⇒ Original omitted + note, no "orig" key
        y = cumsum(randn(200)); r = hamilton_filter(y)
        p = plot_result(r)
        @test occursin("original not supplied", p.html)
        for (nm, lit) in extract_json_blocks(p.html)
            nm == "data" && @test !occursin("\"orig\"", lit)
        end
        # correct-length original still renders fine
        check_plot(plot_result(r; original=y))
    end

    # -------------------------------------------------------------------------
    # PLT-08 — Date serializer + x_ticks/regions renderer options + dated dispatch
    # -------------------------------------------------------------------------
    @testset "PLT-08 _json(Date) + x_ticks/regions options" begin
        @test M._json(Date(2020, 3, 31)) == "\"2020-03-31\""
        @test occursin("2020-03-31T", M._json(DateTime(2020, 3, 31, 12)))

        dj = _line_dj(1:4); sj = _line_sj()
        # null path: integer-tick fallback (byte-identical axis line, no tickValues)
        jnull = M._render_line_js("p_tn", dj, sj)
        @test occursin("d3.axisBottom(x).ticks(Math.min(xVals.length,8))", jnull)
        @test !occursin("tickValues", jnull)
        # provided: ticks drawn only at data x-values, with supplied labels
        xt = M._x_ticks_json([1, 2, 3, 4], ["2020Q1", "2020Q2", "2020Q3", "2020Q4"])
        jt = M._render_line_js("p_tt", dj, sj; x_ticks_json=xt)
        @test occursin("tickValues", jt)
        @test occursin("2020Q1", jt)
        @test !occursin("d3.axisBottom(x).ticks(Math.min(xVals.length,8))", jt)
        @test M._x_ticks_json([1, 2, 3], nothing) == "null"
        @test_throws ArgumentError M._x_ticks_json([1, 2, 3], ["a", "b"])

        # regions_json shade option (recession bands; PLT-20 dependency)
        jr = M._render_line_js("p_rg", dj, sj;
                    regions_json="[{\"x0\":2,\"x1\":3,\"color\":\"#888\",\"alpha\":0.12}]")
        @test occursin("regions.forEach", jr)
        @test occursin("x(rg.x0)", jr)
    end

    @testset "PLT-08 dated TimeSeriesData shows dates not integers" begin
        # (the tickValues/null-axis toggle is asserted at the renderer level above;
        # here we check the DISPATCH picks up d.dates. "tickValues" cannot be scanned
        # in the full html — the inlined D3 source legitimately contains that token.)
        d = TimeSeriesData(cumsum(randn(6)); varname="gdp")
        p0 = plot_result(d)                                   # undated → integer "Time" axis
        @test occursin(".text(\"Time\")", p0.html)
        @test !occursin(".text(\"Date\")", p0.html)
        @test !occursin("2019Q1", p0.html)
        set_dates!(d, ["2019Q1", "2019Q2", "2019Q3", "2019Q4", "2020Q1", "2020Q2"])
        p1 = plot_result(d)                                   # dated → date-string axis
        @test occursin(".text(\"Date\")", p1.html)
        @test !occursin(".text(\"Time\")", p1.html)           # x-label switched
        @test occursin("2019Q1", p1.html)                     # date label present
    end

    # -------------------------------------------------------------------------
    # PLT-07 — HA wealth distribution conserves mass (bin-aggregate, not point-sample)
    # -------------------------------------------------------------------------
    @testset "PLT-07 HA wealth mass conservation" begin
        ss = compute_steady_state(load_ha_example(:krusell_smith))
        n_a = ss.grid.n_points[1]
        @test n_a > 60                                        # binning path exercised
        p = plot_result(ss)                                   # view=:distribution
        total = sum(ss.distribution)
        masssum = 0.0; found = false
        for (nm, lit) in extract_json_blocks(p.html)
            nm == "data" || continue
            v, _ = _tj_parse_value(lit, firstindex(lit))
            for o in v; masssum += o["mass"]; end
            found = true
        end
        @test found
        @test isapprox(masssum, total; atol=1e-8)             # displayed mass == distribution mass
        @test occursin("grid nodes in", p.html)               # binning note visible (C7)
    end
end

# =============================================================================
# PLT-19 — new renderer primitives (scatter overlays/shapes, vbar relocation,
# histogram, box, fan) + the pre-shipped statistical converters. Renderer tests
# per plotrule A9 ("its own unit test"); converter tests per A5/A6.
# =============================================================================
@testset "Plotting PLT-19 renderer primitives + converters" begin

    _pt_dj(xs) = M._json_array_of_objects(
        [["x" => M._json(float(x)), "y" => M._json(float(x)), "group" => M._json("g")] for x in xs])
    _grp1() = "[{\"name\":\"g\",\"color\":\"#1f77b4\"}]"

    # -------------------------------------------------------------------------
    # A1 — every renderer now lives ONLY in render.jl
    # -------------------------------------------------------------------------
    @testset "PLT-19/A1 renderers relocated into render.jl" begin
        srcdir = joinpath(dirname(pathof(MacroEconometricModels)), "plotting")
        defs = String[]
        for f in readdir(srcdir; join=true)
            endswith(f, ".jl") || continue
            for ln in eachline(f)
                m = match(r"^function (_render_\w+_js)", ln)
                m === nothing || push!(defs, basename(f) * ":" * m.captures[1])
            end
        end
        # scatter + vbar must now be defined in render.jl (not did.jl/spectral.jl)
        @test "render.jl:_render_scatter_js" in defs
        @test "render.jl:_render_vbar_js" in defs
        @test "render.jl:_render_histogram_js" in defs
        @test "render.jl:_render_box_js" in defs
        @test "render.jl:_render_fan_js" in defs
        @test !("did.jl:_render_scatter_js" in defs)          # deleted from did.jl
        @test !("spectral.jl:_render_vbar_js" in defs)        # deleted from spectral.jl
    end

    # -------------------------------------------------------------------------
    # Scatter: sloped overlay (A4), ref circle, big-N cap (C7), integer_x
    # -------------------------------------------------------------------------
    @testset "PLT-19 scatter overlays / shapes / big-N / integer_x" begin
        dj = _pt_dj(1:5); gj = _grp1()

        ov = "[{\"x1\":0,\"y1\":0,\"x2\":5,\"y2\":5,\"color\":\"#d62728\",\"dash\":\"4,3\"}]"
        js = M._render_scatter_js("p_ov", dj, gj; line_overlays_json=ov, xlabel="x", ylabel="y")
        @test occursin("lineOverlays.forEach", js)
        @test occursin(".attr('x1', x(o.x1)).attr('y1', y(o.y1))", js)  # data coords → own scales
        @test count("const margin", js) == 1                            # single scale block (A4, no clone)

        rs = "[{\"type\":\"circle\",\"cx\":0,\"cy\":0,\"r\":1}]"
        jsc = M._render_scatter_js("p_c", dj, gj; ref_shapes_json=rs)
        @test occursin("g.append('ellipse')", jsc)                      # circle → ellipse thru scales
        @test occursin("s.cx + s.r", jsc)

        jsi = M._render_scatter_js("p_i", dj, gj; integer_x=true)
        @test occursin("tickFormat(d3.format('d'))", jsi)
        jsd = M._render_scatter_js("p_d", dj, gj)
        @test occursin("d3.axisBottom(x).ticks(8)", jsd)
        @test !occursin("tickFormat(d3.format('d'))", jsd)

        big = M._json_array_of_objects(
            [["x" => M._json(float(i)), "y" => M._json(float(i)), "group" => M._json("g")] for i in 1:3000])
        jsb = M._render_scatter_js("p_b", big, gj)
        @test occursin("data.length > CAP", jsb)
        @test occursin("showing '+drawn.length+' of '+subN+' points", jsb)  # C7 note
        assert_strict_json(big)                                          # full 3000-point data valid
        for (_, lit) in extract_json_blocks(jsb); assert_strict_json(lit); end

        # relocated scatter renders end-to-end into a full standalone document
        sp = M._PanelSpec("p_se", "Scatter", M._render_scatter_js("p_se", dj, gj; xlabel="x", ylabel="y"))
        check_plot(M._make_plot([sp]))
    end

    # -------------------------------------------------------------------------
    # Histogram renderer + _histogram_bins / _kde_line converters
    # -------------------------------------------------------------------------
    @testset "PLT-19 histogram renderer + bins/kde converters" begin
        bins = M._histogram_bins([1.0, 2.0, NaN, Inf, -Inf, 3.0, 2.5])   # NaN/Inf dropped
        @test !occursin("null", bins)
        @test !occursin("NaN", bins)
        assert_strict_json(bins)

        binsd = M._histogram_bins(randn(500); density=true)             # density integrates to 1
        v, _ = _tj_parse_value(binsd, firstindex(binsd))
        @test isapprox(sum(o["y"] * (o["x1"] - o["x0"]) for o in v), 1.0; atol=1e-6)

        @test M._histogram_bins(Float64[]) == "[]"                      # empty → no bins
        @test M._histogram_bins([NaN, Inf]) == "[]"                     # all dropped
        assert_strict_json(M._histogram_bins([5.0]))                    # single obs, no error
        bc = M._histogram_bins(fill(3.0, 10))                           # constant → one bin
        vc, _ = _tj_parse_value(bc, firstindex(bc))
        @test length(vc) == 1

        xs, dens = M._kde_line(randn(400))
        @test isapprox(sum(dens) * (xs[2] - xs[1]), 1.0; atol=0.05)     # KDE integrates to ~1
        xs2, dens2 = M._kde_line([1.0, NaN, 2.0, Inf, 3.0])            # NaN dropped
        @test all(isfinite, dens2)

        sj = "[{\"name\":\"Data\",\"color\":\"#1f77b4\"},{\"name\":\"KDE\",\"color\":\"#d62728\"}]"
        jr = M._render_histogram_js("p_hist", binsd, sj;
                 density_json=M._kde_line_json(randn(200)),
                 ref_lines_json="[{\"value\":0,\"axis\":\"x\",\"color\":\"#d62728\"}]",
                 xlabel="Value", ylabel="Density")
        @test occursin("rect.bar", jr)
        @test occursin("d3.line()", jr)                                # density overlay curve
        @test occursin(".attr('x1',x(r.value)).attr('x2',x(r.value))", jr)  # axis:x vertical ref
        for (_, lit) in extract_json_blocks(jr); assert_strict_json(lit); end

        sjh = "[" * M._json_obj(["name" => M._json(HOSTILE_NAME), "color" => M._json("#1f77b4")]) * "]"
        jh = M._render_histogram_js("p_he", binsd, sjh; xlabel=HOSTILE_NAME)
        @test !occursin("</script>", jh)
        @test !occursin("<c>", jh)
    end

    # -------------------------------------------------------------------------
    # Box renderer + _boxplot_stats converter
    # -------------------------------------------------------------------------
    @testset "PLT-19 box renderer + boxplot stats" begin
        s = M._boxplot_stats([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])         # 100 = outlier
        @test s.q1 <= s.med <= s.q3
        @test 100.0 in s.outliers
        @test s.whishi < 100.0

        sc = M._boxplot_stats(fill(7.0, 20))                           # constant collapses to median
        @test sc.whislo == sc.q1 == sc.med == sc.q3 == sc.whishi == 7.0
        @test isempty(sc.outliers)

        sn = M._boxplot_stats([1.0, NaN, 2.0, Inf, 3.0])              # NaN dropped
        @test sn.med == 2.0

        Random.seed!(1)
        bj = M._boxes_json(["A", "B"], [randn(50), randn(50) .+ 2])
        assert_strict_json(bj)
        jv = M._render_box_js("p_bv", bj; orientation=:v, xlabel="Group", ylabel="Value")
        @test occursin("const horiz = false", jv)
        jhz = M._render_box_js("p_bh", bj; orientation=:h, tip_label="Sector")
        @test occursin("const horiz = true", jhz)
        @test occursin("Sector", jhz)                                  # tip_label prefix (A2)

        bjc = M._boxes_json(["C"], [fill(3.0, 10)])                    # constant series renders
        assert_strict_json(bjc)
        @test occursin("append('svg')", M._render_box_js("p_bc", bjc))

        bjh = M._boxes_json([HOSTILE_NAME], [randn(20)])              # escaping
        @test !occursin("</script>", M._render_box_js("p_bhe", bjh))
    end

    # -------------------------------------------------------------------------
    # Fan renderer + _fan_data_json / _fan_bands_json converters
    # -------------------------------------------------------------------------
    @testset "PLT-19 fan renderer + fan converters" begin
        levels = [0.05, 0.25, 0.5, 0.75, 0.95]; Tn = 8; nq = 5
        q = zeros(Tn, nq)
        for t in 1:Tn, j in 1:nq; q[t, j] = levels[j]; end
        central = fill(0.5, Tn)

        dj = M._fan_data_json(q, levels, central)
        fj = M._fan_bands_json(levels)
        assert_strict_json(dj); assert_strict_json(fj)
        fv, _ = _tj_parse_value(fj, firstindex(fj))
        @test length(fv) == 2                                          # 5 levels → 2 nested bands
        @test any(o -> endswith(o["label"], "95%"), fv)
        @test any(o -> endswith(o["label"], "75%"), fv)
        @test fv[1]["alpha"] < fv[2]["alpha"]                          # outer→inner alpha ramp
        @test fv[1]["lo_key"] == "q1" && fv[1]["hi_key"] == "q5"       # keys match sorted re-keying

        @test_throws ArgumentError M._fan_data_json(q, levels[1:4], central)  # level/col mismatch

        js = M._render_fan_js("p_fan", dj, fj; xlabel="Horizon", ylabel="Response")
        @test occursin("fan.forEach", js)
        @test occursin("95%", js) && occursin("75%", js)               # k legend-labelled bands
        @test occursin("Median", js)                                   # central label (default)
        jsm = M._render_fan_js("p_fan2", dj, fj; central_label="Mean")
        @test occursin("Mean", jsm)                                    # honest central label (C6)
        for (_, lit) in extract_json_blocks(js); assert_strict_json(lit); end
    end

    # -------------------------------------------------------------------------
    # Line renderer integer_x option (PLT-17 prep; scatter shares it)
    # -------------------------------------------------------------------------
    @testset "PLT-19 line integer_x option" begin
        lsj = "[" * M._json_obj(["name" => M._json("s"), "color" => M._json("#1f77b4"),
                                 "key" => M._json("y"), "dash" => M._json("")]) * "]"
        ldj = M._json_array_of_objects([["x" => M._json(x), "y" => M._json(float(x))] for x in 0:5])
        ji = M._render_line_js("p_lix", ldj, lsj; integer_x=true)
        @test occursin("tickFormat(d3.format('d'))", ji)
        @test occursin("Math.ceil(_xd[0])", ji)
        jd = M._render_line_js("p_lid", ldj, lsj)                      # default unchanged
        @test occursin("d3.axisBottom(x).ticks(Math.min(xVals.length,8))", jd)
        @test !occursin("tickFormat(d3.format('d'))", jd)
        xt = M._x_ticks_json(collect(0:5), ["a","b","c","d","e","f"])  # date map still wins
        jt = M._render_line_js("p_lit", ldj, lsj; x_ticks_json=xt, integer_x=true)
        @test occursin("tickValues(_xtv", jt)
        @test !occursin("tickFormat(d3.format('d'))", jt)
    end

    # -------------------------------------------------------------------------
    # Shared statistical converters: OLS fit line, FWL, correlation
    # -------------------------------------------------------------------------
    @testset "PLT-19 stat converters (ols / fwl / corr)" begin
        x = collect(1.0:10.0); y = 2.0 .* x .+ 1.0
        ov = M._ols_fit_line(x, y; degree=1)
        ovv, _ = _tj_parse_value(ov, firstindex(ov))
        @test length(ovv) == 1                                         # degree 1 = one segment
        sg = ovv[1]
        @test isapprox((sg["y2"] - sg["y1"]) / (sg["x2"] - sg["x1"]), 2.0; atol=1e-8)
        ov2 = M._ols_fit_line(x, y .+ 0.1 .* x .^ 2; degree=2)
        ov2v, _ = _tj_parse_value(ov2, firstindex(ov2))
        @test length(ov2v) > 1                                         # degree 2 = polyline
        assert_strict_json(M._ols_fit_line([1.0, 2.0, NaN, 4.0], [1.0, 2.0, 3.0, 4.0]))

        Random.seed!(11)
        z = randn(120); xreg = 0.5 .* z .+ randn(120)
        yreg = 3.0 .* xreg .+ 2.0 .* z .+ 0.1 .* randn(120)
        rx = M._fwl_residualize(xreg, z); ry = M._fwl_residualize(yreg, z)
        b_fwl = sum(rx .* ry) / sum(rx .^ 2)
        b_mv = (hcat(ones(120), xreg, z) \ yreg)[2]
        @test isapprox(b_fwl, b_mv; atol=1e-8)                         # FWL theorem

        Random.seed!(5)
        Msamp = hcat(randn(80), randn(80))
        cj = M._corr_matrix_json(Msamp; labels=["a", "b"])
        assert_strict_json(cj)
        cv, _ = _tj_parse_value(cj, firstindex(cj))
        @test all(o -> isapprox(o["v"], 1.0; atol=1e-8), [o for o in cv if o["x"] == o["y"]])
        assert_strict_json(M._corr_matrix_json([1.0 2.0; NaN 4.0; 3.0 6.0]; labels=["a", "b"]))
        @test_throws ArgumentError M._corr_matrix_json(Msamp; labels=["a"])
    end

    # -------------------------------------------------------------------------
    # Shared 4-panel residual-diagnostics helper (A6, pre-shipped for PLT-24/25)
    # -------------------------------------------------------------------------
    @testset "PLT-19 _residual_diagnostics_panels" begin
        Random.seed!(3)
        panels = M._residual_diagnostics_panels(randn(120), randn(120); varname="gdp")
        @test length(panels) == 4
        @test panels[1].title == "Residual vs Fitted"
        @test panels[3].title == "Normal Q-Q"
        @test panels[4].title == "Residual ACF"
        @test occursin("lineOverlays.forEach", panels[3].js)          # Q-Q 45° via overlay (A4)
        @test count("const margin", panels[3].js) == 1                # no scale-clone
        @test occursin("gdp Residual", panels[2].js)                  # varname threaded (live kwarg)
        p = M._make_plot(panels; title="Residual Diagnostics", ncols=2)
        check_plot(p)
        assert_all_json_valid(p)
        pc = M._residual_diagnostics_panels(fill(0.5, 50), randn(50))  # constant resid, no error
        @test length(pc) == 4
    end
end
