# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using Random

# Helper: run f(backend) for each backend, always resetting to :text
function _with_each_backend(f)
    for be in (:text, :latex, :html)
        set_display_backend(be)
        try
            f(be)
        finally
            set_display_backend(:text)
        end
    end
end

@testset "Display Backend Switching" begin
    Random.seed!(42)
    Y = randn(100, 3)
    m = estimate_var(Y, 2)

    @testset "Default backend is :text" begin
        @test get_display_backend() == :text
    end

    @testset "Text backend output" begin
        buf = IOBuffer()
        show(buf, m)
        text_out = String(take!(buf))
        @test occursin("VAR(2) Model", text_out)
        @test !occursin("<table>", text_out)
        @test !occursin("\\begin{tabular}", text_out)
    end

    @testset "LaTeX backend output" begin
        set_display_backend(:latex)
        try
            @test get_display_backend() == :latex
            buf = IOBuffer()
            show(buf, m)
            latex_out = String(take!(buf))
            @test occursin("tabular", latex_out)
            @test occursin("VAR", latex_out)
        finally
            set_display_backend(:text)
        end
    end

    @testset "HTML backend output" begin
        set_display_backend(:html)
        try
            @test get_display_backend() == :html
            buf = IOBuffer()
            show(buf, m)
            html_out = String(take!(buf))
            @test occursin("<table>", html_out)
            @test occursin("VAR", html_out)
        finally
            set_display_backend(:text)
        end
    end

    @testset "Invalid backend throws ArgumentError" begin
        @test_throws ArgumentError set_display_backend(:pdf)
        @test_throws ArgumentError set_display_backend(:csv)
    end

    @testset "Reset works" begin
        set_display_backend(:latex)
        try
            @test get_display_backend() == :latex
        finally
            set_display_backend(:text)
        end
        @test get_display_backend() == :text
    end

    @testset "VARModel renders in all backends" begin
        _with_each_backend() do be
            buf = IOBuffer()
            show(buf, m)
            out = String(take!(buf))
            @test length(out) > 0
            @test occursin("VAR", out)
        end
    end

    @testset "IRF renders in all backends" begin
        irf_result = irf(m, 10)
        _with_each_backend() do be
            buf = IOBuffer()
            show(buf, irf_result)
            out = String(take!(buf))
            @test length(out) > 0
        end
    end

    @testset "FEVD renders in all backends" begin
        fevd_result = fevd(m, 10)
        _with_each_backend() do be
            buf = IOBuffer()
            show(buf, fevd_result)
            out = String(take!(buf))
            @test length(out) > 0
        end
    end

    @testset "ARIMA models render in all backends" begin
        y = randn(200)
        ar = estimate_ar(y, 2)
        _with_each_backend() do be
            buf = IOBuffer()
            show(buf, ar)
            out = String(take!(buf))
            @test length(out) > 0
            @test occursin("AR", out)
        end
        # Verify publication-quality columns in text mode
        buf = IOBuffer(); show(buf, ar); out = String(take!(buf))
        @test occursin("Std.Err.", out)
        @test occursin("CI]", out)
    end

    @testset "Unit root tests render in all backends" begin
        y = cumsum(randn(200))
        adf = adf_test(y)
        _with_each_backend() do be
            buf = IOBuffer()
            show(buf, adf)
            out = String(take!(buf))
            @test length(out) > 0
        end
    end

    @testset "Factor model renders in all backends" begin
        X = randn(100, 10)
        fm = estimate_factors(X, 3)
        _with_each_backend() do be
            buf = IOBuffer()
            show(buf, fm)
            out = String(take!(buf))
            @test length(out) > 0
        end
    end

    @testset "Historical decomposition renders in all backends" begin
        hd_result = historical_decomposition(m, size(Y, 1) - m.p)
        _with_each_backend() do be
            buf = IOBuffer()
            show(buf, hd_result)
            out = String(take!(buf))
            @test length(out) > 0
        end
    end

    @testset "print_table works in all backends" begin
        irf_result = irf(m, 10)
        _with_each_backend() do be
            buf = IOBuffer()
            print_table(buf, irf_result, 1, 1)
            out = String(take!(buf))
            @test length(out) > 0
        end
    end

    @testset "report() does not error" begin
        # report(VARModel) writes to stdout; backend switching already tested via show(buf, m)
        @test (redirect_stdout(devnull) do
            report(m)
        end; true)
    end

    @testset "ARIMA show in all backends" begin
        Random.seed!(9901)
        y = randn(100)
        ar_m = estimate_ar(y, 1)
        ma_m = estimate_ma(y, 1)
        arma_m = estimate_arma(y, 1, 1)
        _with_each_backend() do be
            for model in [ar_m, ma_m, arma_m]
                buf = IOBuffer()
                show(buf, model)
                @test length(String(take!(buf))) > 0
            end
        end
    end

    @testset "Unit root result show in all backends" begin
        Random.seed!(9902)
        y = randn(100)
        adf_r = adf_test(y)
        kpss_r = kpss_test(y)
        pp_r = pp_test(y)
        _with_each_backend() do be
            for r in [adf_r, kpss_r, pp_r]
                buf = IOBuffer()
                show(buf, r)
                @test length(String(take!(buf))) > 0
            end
        end
    end

    @testset "Factor model show in all backends" begin
        Random.seed!(9903)
        X = randn(100, 10)
        fm = estimate_factors(X, 2)
        _with_each_backend() do be
            buf = IOBuffer()
            show(buf, fm)
            @test length(String(take!(buf))) > 0
        end
    end

    @testset "Non-Gaussian result show in all backends" begin
        Random.seed!(9904)
        Y = randn(100, 2)
        var_m = estimate_var(Y, 1)
        ica_r = identify_fastica(var_m)
        ml_r = identify_student_t(var_m; max_iter=50)
        _with_each_backend() do be
            for r in [ica_r, ml_r]
                buf = IOBuffer()
                show(buf, r)
                @test length(String(take!(buf))) > 0
            end
        end
    end

    @testset "refs() bibliographic references" begin
        Random.seed!(42)
        model = estimate_var(randn(100, 2), 2)

        # Text format
        io = IOBuffer(); refs(io, model; format=:text)
        s = String(take!(io))
        @test occursin("Sims", s)
        @test occursin("DOI:", s)

        # BibTeX format
        io = IOBuffer(); refs(io, model; format=:bibtex)
        s = String(take!(io))
        @test occursin("@article{sims1980", s)
        @test occursin("@book{lutkepohl2005", s)

        # LaTeX format
        io = IOBuffer(); refs(io, model; format=:latex)
        s = String(take!(io))
        @test occursin("\\bibitem{sims1980}", s)

        # HTML format
        io = IOBuffer(); refs(io, model; format=:html)
        s = String(take!(io))
        @test occursin("<a href=", s)
        @test occursin("<em>", s)

        # Symbol dispatch
        io = IOBuffer(); refs(io, :fastica; format=:text)
        s = String(take!(io))
        @test occursin("rinen", s)  # Hyvärinen

        io = IOBuffer(); refs(io, :johansen; format=:text)
        s = String(take!(io))
        @test occursin("Johansen", s)

        # Unit root test refs
        y = cumsum(randn(200))
        adf_r = adf_test(y)
        io = IOBuffer(); refs(io, adf_r; format=:text)
        s = String(take!(io))
        @test occursin("Dickey", s)

        # ARIMA refs
        ar_m = estimate_ar(randn(100), 1)
        io = IOBuffer(); refs(io, ar_m; format=:text)
        s = String(take!(io))
        @test occursin("Box", s)

        # Volatility model refs
        io = IOBuffer(); refs(io, :garch; format=:bibtex)
        s = String(take!(io))
        @test occursin("@article{bollerslev1986", s)

        # ICA variant-dependent refs
        var_m2 = estimate_var(randn(200, 2), 1)
        ica_r2 = identify_fastica(var_m2)
        io = IOBuffer(); refs(io, ica_r2; format=:text)
        s = String(take!(io))
        @test occursin("rinen", s)  # Hyvärinen from FastICA method-specific ref

        # Unknown symbol throws
        @test_throws ArgumentError refs(IOBuffer(), :nonexistent)

        # Unknown format throws
        @test_throws ArgumentError refs(IOBuffer(), model; format=:pdf)

        # Convenience stdout form does not error
        @test (redirect_stdout(devnull) do; refs(model); end; true)
        @test (redirect_stdout(devnull) do; refs(:johansen); end; true)
    end
end

@testset "with_display_backend (scoped, concurrency-safe) — #249" begin
    Random.seed!(2049)
    m = estimate_var(randn(80, 2), 1)
    set_display_backend(:text)   # process default

    @testset "scoped override restores on exit" begin
        @test get_display_backend() == :text
        out = with_display_backend(:latex) do
            @test get_display_backend() == :latex
            sprint(show, m)
        end
        @test occursin("tabular", out)
        @test get_display_backend() == :text
    end

    @testset "nesting pops back to the enclosing scope" begin
        with_display_backend(:latex) do
            @test get_display_backend() == :latex
            with_display_backend(:html) do
                @test get_display_backend() == :html
            end
            @test get_display_backend() == :latex
        end
        @test get_display_backend() == :text
    end

    @testset "invalid backend throws" begin
        @test_throws ArgumentError with_display_backend(identity, :pdf)
    end

    @testset "scope wins over the process default" begin
        set_display_backend(:html)
        try
            @test get_display_backend() == :html
            with_display_backend(:latex) do
                @test get_display_backend() == :latex
            end
            @test get_display_backend() == :html
        finally
            set_display_backend(:text)
        end
    end

    @testset "child tasks inherit the scope" begin
        out = with_display_backend(:latex) do
            fetch(Threads.@spawn sprint(show, m))
        end
        @test occursin("tabular", out) && !occursin("<table>", out)
    end

    @testset "concurrent tasks under different backends do not collide" begin
        # The old single shared Ref let one task's backend clobber another's read under
        # interleaving (finding G-08). A ScopedValue keeps each task's scope isolated;
        # the explicit yield() forces the interleaving that would expose a shared Ref.
        n = 32
        latex_out = Vector{String}(undef, n)
        html_out  = Vector{String}(undef, n)
        @sync for i in 1:n
            Threads.@spawn (latex_out[i] = with_display_backend(:latex) do
                yield(); sprint(show, m)
            end)
            Threads.@spawn (html_out[i] = with_display_backend(:html) do
                yield(); sprint(show, m)
            end)
        end
        @test all(s -> occursin("tabular", s) && !occursin("<table>", s), latex_out)
        @test all(s -> occursin("<table>", s) && !occursin("tabular", s), html_out)
        @test get_display_backend() == :text   # default untouched throughout
    end

    @testset "default :text render is deterministic and unpolluted" begin
        a = sprint(show, m)
        b = sprint(show, m)
        @test a == b
        @test !occursin("tabular", a) && !occursin("<table>", a)
    end
end

@testset "Non-TTY cropping disabled (S1/T161)" begin
    set_display_backend(:text)

    # (A) Wide 8-column coefficient table at a NARROW width. With PrettyTables' fit-to-
    #     display crop ON (the pre-fix default) the trailing significance/CI columns are
    #     dropped with a "N columns omitted" footer. Crop OFF renders the full table.
    Random.seed!(1)
    X = randn(200, 4)
    y = X * [1.0, -0.8, 0.6, 0.4] .+ 0.1 .* randn(200)
    mr = estimate_reg(y, X)
    buf = IOBuffer(); show(IOContext(buf, :displaysize => (24, 50), :color => false), mr)
    out = String(take!(buf))
    @test !occursin("omitted", out)          # horizontal crop off → no dropped columns
    @test occursin("Std.Err.", out)          # full-width table present
    @test occursin("P>|", out)               # p-value column (last-but-two) survives

    # (B) GARCH show spans >24 lines → the pre-fix vertical crop drops interior rows.
    Random.seed!(2)
    mg = estimate_garch(randn(400))
    buf = IOBuffer(); show(IOContext(buf, :displaysize => (24, 80), :color => false), mg)
    out = String(take!(buf))
    @test !occursin("omitted", out)          # vertical crop off

    # (C) ACF with 25 lags → long table (audit V04 vertical-crop victim).
    ra = acf(randn(300); lags = 25)
    buf = IOBuffer(); show(IOContext(buf, :displaysize => (24, 80), :color => false), ra)
    out = String(take!(buf))
    @test !occursin("omitted", out)
    @test count(==('\n'), out) > 20          # all 25 lag rows rendered, not clipped to a 24-line box

    # (D) Regression guard: the :latex/:html branches must still render (they must NOT
    #     receive the text-only fit_table_in_display_* kwargs, which would error there).
    for be in (:latex, :html)
        s = with_display_backend(be) do
            sprint(show, mr)
        end
        @test length(s) > 0
    end

    set_display_backend(:text)
end

@testset "Empty-header suppression + legend-as-text (S7/T162)" begin
    set_display_backend(:text)

    # (A) _sig_legend is a single plain line, not a multi-line table.
    s = sprint(MacroEconometricModels._sig_legend)
    @test occursin("Significance: *** p<0.01, ** p<0.05, * p<0.10", s)
    @test count(==('\n'), s) <= 1

    # (B) _show_note is plain text, not a 2-column "Note | text" table.
    s = sprint(MacroEconometricModels._show_note, "* CI excludes zero")
    @test occursin("* CI excludes zero", s)
    @test !occursin("Note", s)          # the old wrapper cell is gone
    @test count(==('\n'), s) <= 1

    # (C) All-empty column labels suppress the header row (no stray blank band); a
    #     key/value spec table still prints its data rows.
    s = sprint(Any["Stationary" "Yes"; "Max |λ|" "0.9"]) do io, d
        MacroEconometricModels._pretty_table(io, d;
            title = "Stationarity", column_labels = ["", ""], alignment = [:l, :r])
    end
    @test occursin("Stationary", s) && occursin("0.9", s)   # data rows survive
    @test !occursin("\n\n\n", s)                            # no extra empty-header blank band
                                                            # (one blank after the title is the format's own spacing)

    # (D) Non-empty labels are untouched — a real header still renders.
    s = sprint(Any[1 2 3]) do io, d
        MacroEconometricModels._pretty_table(io, d;
            column_labels = ["", "Coef.", "SE"], alignment = [:l, :r, :r])
    end
    @test occursin("Coef.", s)

    set_display_backend(:text)
end

@testset "Fixed-decimal number formatting (S2/T163)" begin
    set_display_backend(:text)
    # _matrix_table routes every cell through _fmt → constant 4-decimal columns; the A01
    # ragged case (round() printed "1.0 / -0.033 / 0.1829") is fixed.
    M = [1.0 -0.033; 0.1829 1.0]
    s = sprint(io -> MacroEconometricModels._matrix_table(io, M, "Residual Correlation";
                        row_labels = ["y1", "y2"], col_labels = ["y1", "y2"]))
    @test occursin("1.0000", s)      # was "1.0"
    @test occursin("-0.0330", s)     # was "-0.033"
    @test occursin("0.1829", s)
    set_display_backend(:text)
end

@testset "Shared display conventions (S8/T166)" begin
    set_display_backend(:text)
    MEM = MacroEconometricModels
    # label dictionary + prettify fallback for unknown symbols
    @test MEM._label(:hc1) == "HC1 (robust)"
    @test MEM._label(:css_mle) == "CSS-MLE"
    @test MEM._label(:ar1) == "AR(1)"
    @test MEM._label(:jarque_bera) == "Jarque–Bera"
    @test MEM._label(:some_unknown_thing) == "Some Unknown Thing"   # prettified, not raw
    @test MEM._label("normal") == "Normal"                          # String dispatch
    # Yes/No booleans (no raw true/false)
    @test MEM._yesno(true) == "Yes" && MEM._yesno(false) == "No"
    # one canonical p-value formatter
    @test MEM._fmt_pvalue(0.0001) == "<0.001"
    @test MEM._fmt_pvalue === MEM._format_pvalue
    # intercept convention: internal tokens map to (Intercept), others pass through
    @test MEM._INTERCEPT_LABEL == "(Intercept)"
    @test MEM._display_intercept("const") == "(Intercept)"
    @test MEM._display_intercept("_cons") == "(Intercept)"
    @test MEM._display_intercept("Intercept (c)") == "(Intercept)"
    @test MEM._display_intercept("gdp") == "gdp"
    set_display_backend(:text)
end

@testset "_select_horizons deduplication (B2/T165)" begin
    MEM = MacroEconometricModels
    # endpoint no longer doubled when H coincides with a fixed anchor
    @test MEM._select_horizons(8) == [1, 4, 8]        # was [1, 4, 8, 8]
    @test MEM._select_horizons(12) == [1, 4, 8, 12]
    @test MEM._select_horizons(4) == [1, 2, 3, 4]
    @test MEM._select_horizons(24) == [1, 4, 8, 12, 24]
    @test MEM._select_horizons(40) == [1, 4, 8, 12, 24, 40]
    # no duplicated horizons for any H (the doubled-row defect)
    @test all(allunique(MEM._select_horizons(H)) for H in 1:60)
end

@testset "Dust/reference-row guard + degenerate-fit banner (S6/T164)" begin
    set_display_backend(:text)
    MEM = MacroEconometricModels

    # (A) A genuinely-estimated dust row (huge z from se≈1e-15) must NOT print a computed
    #     stat/p/stars ("significant zero"); the real row above it keeps its stars.
    s = sprint(io -> MEM._coef_table(io, "T", ["a", "dust"],
                    [1.0, 8.145e-15], [0.2, 1.0e-15]))
    @test occursin("*", s)                                   # the real row keeps stars
    dust_line = only(filter(l -> occursin("dust", l), split(s, '\n')))
    @test occursin("—", dust_line) && !occursin("*", dust_line)

    # (B) Reference row labeled "(ref)" with dashed estimate/stat.
    s = sprint(io -> MEM._coef_table(io, "T", ["e=-1", "e=0"],
                    [0.0, 0.5], [0.0, 0.1]; ref_rows=[1]))
    @test occursin("(ref)", s)
    ref_line = only(filter(l -> occursin("(ref)", l), split(s, '\n')))
    @test occursin("—", ref_line) && !occursin("*", ref_line)

    # (C) Degenerate-fit banner fires on exploded/non-finite coefficients, silent on a
    #     healthy fit; a finite coefficient (any SE) never triggers it.
    @test occursin("degenerate fit", sprint(io -> MEM._degenerate_fit_banner(io, [4.72533e114, 1.0])))
    @test occursin("degenerate fit", sprint(io -> MEM._degenerate_fit_banner(io, [NaN, 1.0])))
    @test isempty(strip(sprint(io -> MEM._degenerate_fit_banner(io, [0.5, -0.3, 1.2]))))
    @test isempty(strip(sprint(io -> MEM._degenerate_fit_banner(io, [0.5, 0.0]))))
    set_display_backend(:text)
end
