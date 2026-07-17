# =============================================================================
# Renderer + output-safety unit tests (PLT-01 vendored D3, PLT-02 escaping,
# PLT-03 embed-safe multi-plot). Renderer-option tests live here per plotrule
# Testing Rules ("Renderer changes additionally require one test per option").
# =============================================================================

using Test, Random
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
