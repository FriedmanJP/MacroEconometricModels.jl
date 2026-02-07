using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics

@testset "Summary Tables Tests" begin

    @testset "report(VARModel)" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        # report() should not throw - use devnull
        redirect_stdout(devnull) do
            report(model)
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

    @testset "report() for all types" begin
        Y = randn(100, 2)
        model = estimate_var(Y, 2)

        irf_result = irf(model, 8)
        fevd_result = fevd(model, 8)
        hd_result = historical_decomposition(model, 98)

        # All report() calls should work - use devnull
        redirect_stdout(devnull) do
            report(model)
            report(irf_result)
            report(fevd_result)
            report(hd_result)
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

        # report
        redirect_stdout(devnull) do
            report(birf)
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

        # report
        redirect_stdout(devnull) do
            report(bfevd)
        end
        @test true
    end

    # =================================================================
    # Bayesian HD show / table / print_table
    # =================================================================

    @testset "BayesianHistoricalDecomposition report and table" begin
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

        # report
        redirect_stdout(devnull) do
            report(bhd)
        end
        @test true
    end

    # =================================================================
    # report() coverage for all types
    # =================================================================

    @testset "report() coverage for models and results" begin
        # --- ARIMA models ---
        y = randn(200)
        ar_model = estimate_ar(y, 2)
        redirect_stdout(devnull) do
            report(ar_model)
        end
        @test true

        # --- Factor model ---
        X = randn(100, 10)
        fm = estimate_factors(X, 3)
        redirect_stdout(devnull) do
            report(fm)
        end
        @test true

        # --- ARCH model ---
        arch_m = estimate_arch(randn(200), 1)
        redirect_stdout(devnull) do
            report(arch_m)
        end
        @test true

        # --- GMM model ---
        n_obs = 200
        x_gmm = randn(n_obs, 2)
        z_gmm = randn(n_obs, 3)
        g = (theta, x, z) -> z .* (x[:, 1] .- theta[1])
        gmm_m = estimate_gmm(g, [0.0], x_gmm, z_gmm)
        redirect_stdout(devnull) do
            report(gmm_m)
        end
        @test true

        # --- Unit root test ---
        adf_r = adf_test(cumsum(randn(200)))
        redirect_stdout(devnull) do
            report(adf_r)
        end
        @test true

        # --- LP model ---
        Y_lp = randn(100, 3)
        lp_m = estimate_lp(Y_lp, 1, 10)
        redirect_stdout(devnull) do
            report(lp_m)
        end
        @test true

        # --- Volatility forecast ---
        vf = forecast(arch_m, 5)
        redirect_stdout(devnull) do
            report(vf)
        end
        @test true

        # --- ARIMA forecast ---
        af = forecast(ar_model, 5)
        redirect_stdout(devnull) do
            report(af)
        end
        @test true

        # --- LP IRF ---
        lp_irf_r = lp_irf(lp_m)
        redirect_stdout(devnull) do
            report(lp_irf_r)
        end
        @test true

        # --- Auxiliary types ---
        redirect_stdout(devnull) do
            report(MinnesotaHyperparameters())
        end
        @test true
    end

    # =================================================================
    # table() and print_table() for forecast types
    # =================================================================

    @testset "table() for VolatilityForecast" begin
        arch_m = estimate_arch(randn(200), 1)
        vf = forecast(arch_m, 5)
        t = table(vf)
        @test size(t) == (5, 5)
        @test t[:, 1] == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test t[:, 2] == vf.forecast
        @test t[:, 3] == vf.ci_lower
        @test t[:, 4] == vf.ci_upper
        @test t[:, 5] == vf.se
    end

    @testset "print_table() for VolatilityForecast" begin
        arch_m = estimate_arch(randn(200), 1)
        vf = forecast(arch_m, 5)
        io = IOBuffer()
        print_table(io, vf)
        output = String(take!(io))
        @test occursin("Volatility Forecast", output)
        @test occursin("σ² Forecast", output)
    end

    @testset "table() for ARIMAForecast" begin
        y = randn(200)
        ar_m = estimate_ar(y, 2)
        af = forecast(ar_m, 5)
        t = table(af)
        @test size(t) == (5, 5)
        @test t[:, 1] == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test t[:, 2] == af.forecast
        @test t[:, 3] == af.ci_lower
        @test t[:, 4] == af.ci_upper
        @test t[:, 5] == af.se
    end

    @testset "print_table() for ARIMAForecast" begin
        y = randn(200)
        ar_m = estimate_ar(y, 2)
        af = forecast(ar_m, 5)
        io = IOBuffer()
        print_table(io, af)
        output = String(take!(io))
        @test occursin("ARIMA Forecast", output)
        @test occursin("Forecast", output)
    end

    @testset "table() for FactorForecast" begin
        X = randn(100, 10)
        fm = estimate_factors(X, 3)
        fc = forecast(fm, 5)
        # Observable table
        t = table(fc, 1)
        @test size(t) == (5, 4)
        @test t[:, 1] == [1.0, 2.0, 3.0, 4.0, 5.0]
        # Factor table
        t_f = table(fc, 1; type=:factor)
        @test size(t_f) == (5, 4)
        # Bounds check
        @test_throws AssertionError table(fc, 100)
    end

    @testset "print_table() for FactorForecast" begin
        X = randn(100, 10)
        fm = estimate_factors(X, 3)
        fc = forecast(fm, 5)
        io = IOBuffer()
        print_table(io, fc, 1)
        output = String(take!(io))
        @test occursin("Factor Forecast", output)
        @test occursin("Observable 1", output)
        # Factor type
        io2 = IOBuffer()
        print_table(io2, fc, 1; type=:factor)
        output2 = String(take!(io2))
        @test occursin("Factor 1", output2)
    end

    @testset "table() for LPImpulseResponse" begin
        Y_lp = randn(100, 3)
        lp_m = estimate_lp(Y_lp, 1, 8)
        lp_irf_r = lp_irf(lp_m)
        t = table(lp_irf_r, 1)
        @test size(t, 1) == lp_irf_r.horizon + 1
        @test size(t, 2) == 5  # h, IRF, SE, CI_lo, CI_hi
        # String indexing
        t_str = table(lp_irf_r, lp_irf_r.response_vars[1])
        @test t_str == t
        # Invalid string
        @test_throws ArgumentError table(lp_irf_r, "NonExistent")
    end

    @testset "print_table() for LPImpulseResponse" begin
        Y_lp = randn(100, 3)
        lp_m = estimate_lp(Y_lp, 1, 8)
        lp_irf_r = lp_irf(lp_m)
        io = IOBuffer()
        print_table(io, lp_irf_r, 1)
        output = String(take!(io))
        @test occursin("LP IRF", output)
        @test occursin("←", output)
    end

end
