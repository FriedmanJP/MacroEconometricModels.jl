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

    # =================================================================
    # point_estimate / has_uncertainty / uncertainty_bounds
    # =================================================================

    @testset "point_estimate, has_uncertainty, uncertainty_bounds" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        # ImpulseResponse without CI
        irf_no_ci = irf(model, 8)
        pe = MacroEconometricModels.point_estimate(irf_no_ci)
        @test pe == irf_no_ci.values
        @test MacroEconometricModels.has_uncertainty(irf_no_ci) == false
        @test MacroEconometricModels.uncertainty_bounds(irf_no_ci) === nothing

        # ImpulseResponse with bootstrap CI
        irf_ci = irf(model, 8; ci_type=:bootstrap, reps=50)
        pe_ci = MacroEconometricModels.point_estimate(irf_ci)
        @test pe_ci == irf_ci.values
        @test MacroEconometricModels.has_uncertainty(irf_ci) == true
        bounds = MacroEconometricModels.uncertainty_bounds(irf_ci)
        @test bounds !== nothing
        @test bounds[1] == irf_ci.ci_lower
        @test bounds[2] == irf_ci.ci_upper

        # FEVD
        fevd_result = fevd(model, 8)
        pe_fevd = MacroEconometricModels.point_estimate(fevd_result)
        @test pe_fevd == fevd_result.proportions
        @test MacroEconometricModels.has_uncertainty(fevd_result) == false
        @test MacroEconometricModels.uncertainty_bounds(fevd_result) === nothing

        # HistoricalDecomposition
        hd = historical_decomposition(model, 98)
        pe_hd = MacroEconometricModels.point_estimate(hd)
        @test pe_hd == hd.contributions
        @test MacroEconometricModels.has_uncertainty(hd) == false
        @test MacroEconometricModels.uncertainty_bounds(hd) === nothing
    end

    # =================================================================
    # Bayesian IRF show / table / print_table
    # =================================================================

    @testset "BayesianImpulseResponse" begin
        # Construct a synthetic BayesianImpulseResponse
        H, n = 8, 2
        nq = 3
        quantiles_arr = randn(H, n, n, nq)
        # Ensure ordered quantiles
        for h in 1:H, i in 1:n, j in 1:n
            vals = sort(quantiles_arr[h, i, j, :])
            quantiles_arr[h, i, j, :] = vals
        end
        mean_arr = randn(H, n, n)
        vars = ["Var 1", "Var 2"]
        shocks = ["Shock 1", "Shock 2"]
        q_levels = [0.16, 0.5, 0.84]

        birf = BayesianImpulseResponse{Float64}(quantiles_arr, mean_arr, H, vars, shocks, q_levels)

        # show
        io = IOBuffer()
        show(io, birf)
        output = String(take!(io))
        @test occursin("Bayesian Impulse Response Functions", output)
        @test occursin("Quantiles", output)
        @test occursin("Shock 1", output)

        # table - integer indices
        t = table(birf, 1, 1)
        @test size(t, 1) == H
        @test size(t, 2) == 2 + nq  # Horizon, Mean, Q1, Q2, Q3

        # table - specific horizons
        t_h = table(birf, 1, 1; horizons=[1, 4, 8])
        @test size(t_h, 1) == 3

        # table - string indices
        t_str = table(birf, "Var 1", "Shock 1")
        @test t_str == t

        # table - invalid string
        @test_throws ArgumentError table(birf, "NonExistent", "Shock 1")
        @test_throws ArgumentError table(birf, "Var 1", "NonExistent")

        # print_table
        io = IOBuffer()
        print_table(io, birf, 1, 1)
        output = String(take!(io))
        @test occursin("Bayesian IRF", output)
        @test occursin("Mean", output)

        # point_estimate / has_uncertainty / uncertainty_bounds
        pe = MacroEconometricModels.point_estimate(birf)
        @test pe == birf.mean
        @test MacroEconometricModels.has_uncertainty(birf) == true
        bounds = MacroEconometricModels.uncertainty_bounds(birf)
        @test bounds[1] == birf.quantiles[:, :, :, 1]
        @test bounds[2] == birf.quantiles[:, :, :, nq]

        # summary
        redirect_stdout(devnull) do
            MacroEconometricModels.summary(birf)
        end
        @test true
    end

    # =================================================================
    # Bayesian FEVD show / table / print_table
    # =================================================================

    @testset "BayesianFEVD" begin
        # Construct a synthetic BayesianFEVD
        H, n = 8, 2
        nq = 3
        quantiles_arr = abs.(randn(H, n, n, nq))
        mean_arr = abs.(randn(H, n, n))
        vars = ["Var 1", "Var 2"]
        shocks = ["Shock 1", "Shock 2"]
        q_levels = [0.16, 0.5, 0.84]

        bfevd = BayesianFEVD{Float64}(quantiles_arr, mean_arr, H, vars, shocks, q_levels)

        # show
        io = IOBuffer()
        show(io, bfevd)
        output = String(take!(io))
        @test occursin("Bayesian FEVD", output)
        @test occursin("posterior mean", output)

        # table - mean stat
        t = table(bfevd, 1)
        @test size(t, 1) == H
        @test size(t, 2) == n + 1  # Horizon + n shocks

        # table - specific horizons
        t_h = table(bfevd, 1; horizons=[1, 4])
        @test size(t_h, 1) == 2

        # table - quantile stat
        t_q = table(bfevd, 1; stat=2)  # Median
        @test size(t_q, 1) == H

        # print_table - mean
        io = IOBuffer()
        print_table(io, bfevd, 1)
        output = String(take!(io))
        @test occursin("Bayesian FEVD", output)
        @test occursin("Var 1", output)

        # print_table - quantile
        io = IOBuffer()
        print_table(io, bfevd, 1; stat=1)
        output = String(take!(io))
        @test occursin("Bayesian FEVD", output)

        # point_estimate / has_uncertainty / uncertainty_bounds
        pe = MacroEconometricModels.point_estimate(bfevd)
        @test pe == bfevd.mean
        @test MacroEconometricModels.has_uncertainty(bfevd) == true
        bounds = MacroEconometricModels.uncertainty_bounds(bfevd)
        @test bounds[1] == bfevd.quantiles[:, :, :, 1]
        @test bounds[2] == bfevd.quantiles[:, :, :, nq]

        # summary
        redirect_stdout(devnull) do
            MacroEconometricModels.summary(bfevd)
        end
        @test true
    end

    # =================================================================
    # Bayesian HD show / table / print_table
    # =================================================================

    @testset "BayesianHistoricalDecomposition summary and table" begin
        # Construct a synthetic BayesianHistoricalDecomposition
        T_eff, n = 50, 2
        nq = 3
        quantiles_arr = randn(T_eff, n, n, nq)
        mean_arr = randn(T_eff, n, n)
        initial_q = randn(T_eff, n, nq)
        initial_m = randn(T_eff, n)
        shocks_m = randn(T_eff, n)
        actual = randn(T_eff, n)
        vars = ["Var 1", "Var 2"]
        shock_names = ["Shock 1", "Shock 2"]
        q_levels = [0.16, 0.5, 0.84]

        bhd = BayesianHistoricalDecomposition{Float64}(
            quantiles_arr, mean_arr, initial_q, initial_m,
            shocks_m, actual, T_eff, vars, shock_names, q_levels, :cholesky
        )

        # show
        io = IOBuffer()
        show(io, bhd)
        output = String(take!(io))
        @test occursin("Bayesian Historical Decomposition", output)
        @test occursin("cholesky", output)
        @test occursin("Var 1", output)

        # table - mean
        t = table(bhd, 1)
        @test size(t, 1) == T_eff
        @test size(t, 2) == n + 3  # Period, Actual, n shocks, Initial

        # table - specific periods
        t_p = table(bhd, 1; periods=1:10)
        @test size(t_p, 1) == 10

        # table - quantile stat
        t_q = table(bhd, 1; stat=2)
        @test size(t_q, 1) == T_eff

        # print_table - mean
        io = IOBuffer()
        print_table(io, bhd, 1; periods=1:5)
        output = String(take!(io))
        @test occursin("Bayesian HD", output)

        # print_table - quantile
        io = IOBuffer()
        print_table(io, bhd, 1; stat=1)
        output = String(take!(io))
        @test occursin("Bayesian HD", output)

        # point_estimate / has_uncertainty / uncertainty_bounds
        pe = MacroEconometricModels.point_estimate(bhd)
        @test pe == bhd.mean
        @test MacroEconometricModels.has_uncertainty(bhd) == true
        bounds = MacroEconometricModels.uncertainty_bounds(bhd)
        @test bounds[1] == bhd.quantiles[:, :, :, 1]
        @test bounds[2] == bhd.quantiles[:, :, :, nq]

        # summary
        redirect_stdout(devnull) do
            MacroEconometricModels.summary(bhd)
        end
        @test true
    end

end
