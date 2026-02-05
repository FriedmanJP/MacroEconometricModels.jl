using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics

@testset "Summary Tables Tests" begin

    @testset "summary(VARModel)" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        # summary() should not throw - use devnull
        redirect_stdout(devnull) do
            MacroEconometricModels.summary(model)
        end
        @test true
    end

    @testset "IRF table and print_table" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        # Frequentist IRF
        irf_result = irf(model, 12)

        # Test show method
        io = IOBuffer()
        show(io, irf_result)
        output = String(take!(io))
        @test occursin("Impulse Response Functions", output)

        # Test table()
        t = table(irf_result, 1, 1)
        @test size(t, 1) == 12
        @test size(t, 2) == 2  # No CI

        # With specific horizons
        t_h = table(irf_result, 1, 1; horizons=[1, 4, 8])
        @test size(t_h, 1) == 3

        # With bootstrap CI
        irf_ci = irf(model, 8; ci_type=:bootstrap, reps=50)
        t_ci = table(irf_ci, 1, 1)
        @test size(t_ci, 2) == 4  # With CI

        # Test print_table()
        io = IOBuffer()
        print_table(io, irf_result, 1, 1; horizons=[1, 4, 8])
        output = String(take!(io))
        @test occursin("IRF:", output)

        # String indexing
        t_str = table(irf_result, "Var 1", "Shock 1")
        @test t_str == t
    end

    @testset "FEVD table and print_table" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        fevd_result = fevd(model, 12)

        # Test show method
        io = IOBuffer()
        show(io, fevd_result)
        output = String(take!(io))
        @test occursin("Forecast Error Variance Decomposition", output)

        # Test table()
        t = table(fevd_result, 1)
        @test size(t, 1) == 12
        @test size(t, 2) == 3  # Horizon + 2 shocks

        t_h = table(fevd_result, 1; horizons=[1, 4, 8])
        @test size(t_h, 1) == 3

        # Test print_table()
        io = IOBuffer()
        print_table(io, fevd_result, 1; horizons=[1, 4, 8])
        output = String(take!(io))
        @test occursin("FEVD", output)
    end

    @testset "HD table and print_table" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        hd = historical_decomposition(model, 98)

        # Test show method
        io = IOBuffer()
        show(io, hd)
        output = String(take!(io))
        @test occursin("Historical Decomposition", output)

        # Test table()
        t = table(hd, 1)
        @test size(t, 1) == 98
        @test size(t, 2) == 5  # Period, Actual, 2 shocks, Initial

        t_p = table(hd, 1; periods=90:98)
        @test size(t_p, 1) == 9

        # Test print_table()
        io = IOBuffer()
        print_table(io, hd, 1; periods=90:98)
        output = String(take!(io))
        @test occursin("Historical Decomposition", output)
    end

    @testset "summary() for all types" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        irf_result = irf(model, 8)
        fevd_result = fevd(model, 8)
        hd_result = historical_decomposition(model, 98)

        # All summary() calls should work - use devnull
        redirect_stdout(devnull) do
            MacroEconometricModels.summary(model)
            MacroEconometricModels.summary(irf_result)
            MacroEconometricModels.summary(fevd_result)
            MacroEconometricModels.summary(hd_result)
        end
        @test true
    end

    @testset "_select_horizons" begin
        @test MacroEconometricModels._select_horizons(3) == [1, 2, 3]
        @test MacroEconometricModels._select_horizons(5) == [1, 2, 3, 4, 5]
        @test MacroEconometricModels._select_horizons(10) == [1, 4, 8, 10]
        @test MacroEconometricModels._select_horizons(20) == [1, 4, 8, 12, 20]
        @test MacroEconometricModels._select_horizons(40) == [1, 4, 8, 12, 24, 40]
    end

end
