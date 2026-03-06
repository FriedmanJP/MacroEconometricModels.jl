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
