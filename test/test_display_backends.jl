using MacroEconometricModels
using Test
using Random

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
        @test get_display_backend() == :latex
        buf = IOBuffer()
        show(buf, m)
        latex_out = String(take!(buf))
        @test occursin("tabular", latex_out)
        @test occursin("VAR", latex_out)
        set_display_backend(:text)
    end

    @testset "HTML backend output" begin
        set_display_backend(:html)
        @test get_display_backend() == :html
        buf = IOBuffer()
        show(buf, m)
        html_out = String(take!(buf))
        @test occursin("<table>", html_out)
        @test occursin("VAR", html_out)
        set_display_backend(:text)
    end

    @testset "Invalid backend throws ArgumentError" begin
        @test_throws ArgumentError set_display_backend(:pdf)
        @test_throws ArgumentError set_display_backend(:csv)
    end

    @testset "Reset works" begin
        set_display_backend(:latex)
        @test get_display_backend() == :latex
        set_display_backend(:text)
        @test get_display_backend() == :text
    end

    @testset "VARModel renders in all backends" begin
        for be in (:text, :latex, :html)
            set_display_backend(be)
            buf = IOBuffer()
            show(buf, m)
            out = String(take!(buf))
            @test length(out) > 0
            @test occursin("VAR", out)
        end
        set_display_backend(:text)
    end

    @testset "IRF renders in all backends" begin
        irf_result = irf(m, 10)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            buf = IOBuffer()
            show(buf, irf_result)
            out = String(take!(buf))
            @test length(out) > 0
        end
        set_display_backend(:text)
    end

    @testset "FEVD renders in all backends" begin
        fevd_result = fevd(m, 10)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            buf = IOBuffer()
            show(buf, fevd_result)
            out = String(take!(buf))
            @test length(out) > 0
        end
        set_display_backend(:text)
    end

    @testset "ARIMA models render in all backends" begin
        y = randn(200)
        ar = estimate_ar(y, 2)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            buf = IOBuffer()
            show(buf, ar)
            out = String(take!(buf))
            @test length(out) > 0
            @test occursin("AR", out)
        end
        set_display_backend(:text)
    end

    @testset "Unit root tests render in all backends" begin
        y = cumsum(randn(200))
        adf = adf_test(y)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            buf = IOBuffer()
            show(buf, adf)
            out = String(take!(buf))
            @test length(out) > 0
        end
        set_display_backend(:text)
    end

    @testset "Factor model renders in all backends" begin
        X = randn(100, 10)
        fm = estimate_factors(X, 3)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            buf = IOBuffer()
            show(buf, fm)
            out = String(take!(buf))
            @test length(out) > 0
        end
        set_display_backend(:text)
    end

    @testset "Historical decomposition renders in all backends" begin
        hd_result = historical_decomposition(m, size(Y, 1) - m.p)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            buf = IOBuffer()
            show(buf, hd_result)
            out = String(take!(buf))
            @test length(out) > 0
        end
        set_display_backend(:text)
    end

    @testset "print_table works in all backends" begin
        irf_result = irf(m, 10)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            buf = IOBuffer()
            print_table(buf, irf_result, 1, 1)
            out = String(take!(buf))
            @test length(out) > 0
        end
        set_display_backend(:text)
    end

    @testset "report() does not error in any backend" begin
        for be in (:text, :latex, :html)
            set_display_backend(be)
            # report(VARModel) prints to stdout — just verify no errors
            @test (redirect_stdout(devnull) do
                report(m)
            end; true)
        end
        set_display_backend(:text)
    end

    @testset "ARIMA show in all backends" begin
        Random.seed!(9901)
        y = randn(100)
        ar_m = estimate_ar(y, 1)
        ma_m = estimate_ma(y, 1)
        arma_m = estimate_arma(y, 1, 1)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            for model in [ar_m, ma_m, arma_m]
                buf = IOBuffer()
                show(buf, model)
                @test length(String(take!(buf))) > 0
            end
        end
        set_display_backend(:text)
    end

    @testset "Unit root result show in all backends" begin
        Random.seed!(9902)
        y = randn(100)
        adf_r = adf_test(y)
        kpss_r = kpss_test(y)
        pp_r = pp_test(y)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            for r in [adf_r, kpss_r, pp_r]
                buf = IOBuffer()
                show(buf, r)
                @test length(String(take!(buf))) > 0
            end
        end
        set_display_backend(:text)
    end

    @testset "Factor model show in all backends" begin
        Random.seed!(9903)
        X = randn(100, 10)
        fm = estimate_factors(X, 2)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            buf = IOBuffer()
            show(buf, fm)
            @test length(String(take!(buf))) > 0
        end
        set_display_backend(:text)
    end

    @testset "Non-Gaussian result show in all backends" begin
        Random.seed!(9904)
        Y = randn(100, 2)
        var_m = estimate_var(Y, 1)
        ica_r = identify_fastica(var_m)
        ml_r = identify_student_t(var_m; max_iter=50)
        for be in (:text, :latex, :html)
            set_display_backend(be)
            for r in [ica_r, ml_r]
                buf = IOBuffer()
                show(buf, r)
                @test length(String(take!(buf))) > 0
            end
        end
        set_display_backend(:text)
    end
end
