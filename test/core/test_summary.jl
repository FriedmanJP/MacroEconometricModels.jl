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
        t_str = table(irf_result, "y1", "y1")
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
        data_gmm = randn(n_obs, 3)
        g = (theta, data) -> data[:, 2:3] .* (data[:, 1] .- theta[1])
        gmm_m = estimate_gmm(g, [0.0], data_gmm)
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

    # =================================================================
    # refs() Returns String (Issue #16)
    # =================================================================
    @testset "refs() Returns String" begin
        Y = randn(100, 3)
        model = estimate_var(Y, 2)

        # refs() should return a String, not print to stdout
        result = refs(model)
        @test result isa String
        @test !isempty(result)
        @test occursin("Sims", result) || occursin("Lütkepohl", result) || length(result) > 10

        # BibTeX format
        bib = refs(model; format=:bibtex)
        @test bib isa String
        @test occursin("@", bib)

        # Multiple formats
        for fmt in (:text, :bibtex, :html, :latex)
            r = refs(model; format=fmt)
            @test r isa String
            @test !isempty(r)
        end
    end

    # =================================================================
    # report() for VECM
    # =================================================================
    @testset "report(VECMModel)" begin
        Y = cumsum(randn(150, 3), dims=1)
        vecm = estimate_vecm(Y, 2; rank=1)
        redirect_stdout(devnull) do
            report(vecm)
        end
        @test true
    end

    # =================================================================
    # report() for all 5 filter types
    # =================================================================
    @testset "report() for filter types" begin
        y = cumsum(randn(200))
        redirect_stdout(devnull) do
            report(hp_filter(y))
            report(hamilton_filter(y))
            report(beveridge_nelson(y))
            report(baxter_king(y))
            report(boosted_hp(y))
        end
        @test true
    end

    # =================================================================
    # report() for VARForecast and BVARForecast
    # =================================================================
    @testset "report() for VARForecast and BVARForecast" begin
        Y = randn(100, 3)
        m = estimate_var(Y, 2)
        fc = forecast(m, 5)
        redirect_stdout(devnull) do
            report(fc)
        end
        @test true

        post = estimate_bvar(Y, 2; n_draws=50)
        bfc = forecast(post, 5)
        redirect_stdout(devnull) do
            report(bfc)
        end
        @test true
    end

    # =================================================================
    # report() for LP variants
    # =================================================================
    @testset "report() for LP variants" begin
        Y = randn(100, 3)

        # LPIVModel
        Z = randn(100, 1)
        lp_iv = estimate_lp_iv(Y, 1, Z, 8; lags=2)
        redirect_stdout(devnull) do
            report(lp_iv)
        end
        @test true

        # SmoothLPModel
        slp = estimate_smooth_lp(Y, 1, 8; lags=2, degree=3, n_knots=3)
        redirect_stdout(devnull) do
            report(slp)
        end
        @test true

        # StateLPModel
        state_var = Y[:, 2]
        state_lp = estimate_state_lp(Y, 1, state_var, 8; lags=2)
        redirect_stdout(devnull) do
            report(state_lp)
        end
        @test true

        # PropensityLPModel
        treatment = Float64.(randn(100) .> 0)
        covariates = randn(100, 2)
        prop_lp = estimate_propensity_lp(Y, treatment, covariates, 8)
        redirect_stdout(devnull) do
            report(prop_lp)
        end
        @test true
    end

    # =================================================================
    # report() for StructuralLP, LPForecast, LPFEVD
    # =================================================================
    @testset "show/report for StructuralLP, LPForecast, LPFEVD" begin
        Y = randn(100, 3)
        slp = structural_lp(Y, 8; method=:cholesky, lags=2)
        # StructuralLP has show() but no report() — use show() directly
        io = IOBuffer()
        show(io, slp)
        @test length(String(take!(io))) > 0

        lp = estimate_lp(Y, 1, 8; lags=2)
        shock_path = zeros(8); shock_path[1] = 1.0
        lp_fc = forecast(lp, shock_path)
        redirect_stdout(devnull) do
            report(lp_fc)
        end
        @test true

        lp_f = lp_fevd(slp, 8; n_boot=0)
        # LPFEVD has show() but no report() — use show() directly
        io = IOBuffer()
        show(io, lp_f)
        @test length(String(take!(io))) > 0
    end

    # =================================================================
    # show() for LP model types
    # =================================================================
    @testset "show() for LP model types" begin
        Y = randn(100, 3)

        # LPModel
        lp = estimate_lp(Y, 1, 8; lags=2)
        io = IOBuffer()
        show(io, lp)
        out = String(take!(io))
        @test occursin("Local Projection", out)
        @test occursin("Newey-West", out) || occursin("Jordà", out)

        # LPIVModel
        Z = randn(100, 1)
        lp_iv = estimate_lp_iv(Y, 1, Z, 8; lags=2)
        io = IOBuffer()
        show(io, lp_iv)
        out = String(take!(io))
        @test occursin("LP-IV", out) || occursin("Stock", out)

        # SmoothLPModel
        slp = estimate_smooth_lp(Y, 1, 8; lags=2, degree=3, n_knots=3)
        io = IOBuffer()
        show(io, slp)
        out = String(take!(io))
        @test occursin("Smooth LP", out) || occursin("Barnichon", out)

        # StateLPModel
        state_var = Y[:, 2]
        state_lp = estimate_state_lp(Y, 1, state_var, 8; lags=2)
        io = IOBuffer()
        show(io, state_lp)
        out = String(take!(io))
        @test occursin("State-Dependent", out) || occursin("Auerbach", out)

        # PropensityLPModel
        treatment = Float64.(randn(100) .> 0)
        covariates = randn(100, 2)
        prop_lp = estimate_propensity_lp(Y, treatment, covariates, 8)
        io = IOBuffer()
        show(io, prop_lp)
        out = String(take!(io))
        @test occursin("Propensity", out) || occursin("Angrist", out)
    end

    # =================================================================
    # show() for supporting types
    # =================================================================
    @testset "show() for supporting types" begin
        # ZeroRestriction
        zr = ZeroRestriction(1, 2, 0)
        io = IOBuffer()
        show(io, zr)
        out = String(take!(io))
        @test occursin("ZeroRestriction", out)

        # SignRestriction
        sr = SignRestriction(1, 1, 0, 1)
        io = IOBuffer()
        show(io, sr)
        out = String(take!(io))
        @test occursin("SignRestriction", out)
        @test occursin("+", out)

        # SignRestriction negative
        sr_neg = SignRestriction(1, 1, 0, -1)
        io = IOBuffer()
        show(io, sr_neg)
        out = String(take!(io))
        @test occursin("-", out)

        # SVARRestrictions
        restrictions = SVARRestrictions(
            [zr], [sr], 3, 3
        )
        io = IOBuffer()
        show(io, restrictions)
        out = String(take!(io))
        @test occursin("SVAR Restrictions", out)

        # NeweyWestEstimator (positional args: bandwidth, kernel, prewhiten)
        nw = NeweyWestEstimator{Float64}(0, :bartlett, false)
        io = IOBuffer()
        show(io, nw)
        out = String(take!(io))
        @test occursin("NeweyWest", out)
        @test occursin("automatic", out)

        # WhiteEstimator
        we = WhiteEstimator()
        io = IOBuffer()
        show(io, we)
        out = String(take!(io))
        @test occursin("White", out)

        # DriscollKraayEstimator (positional args: bandwidth, kernel)
        dk = DriscollKraayEstimator{Float64}(5, :bartlett)
        io = IOBuffer()
        show(io, dk)
        out = String(take!(io))
        @test occursin("DriscollKraay", out)
        @test occursin("5", out)

        # GMMWeighting (positional args: method, max_iter, tol)
        gw = GMMWeighting{Float64}(:two_step, 100, 1e-6)
        io = IOBuffer()
        show(io, gw)
        out = String(take!(io))
        @test occursin("GMMWeighting", out)
        @test occursin("two_step", out)
    end

    # =================================================================
    # show() for BSplineBasis, StateTransition, PropensityScoreConfig
    # =================================================================
    @testset "show() for BSplineBasis, StateTransition, PropensityScoreConfig" begin
        Y = randn(100, 3)

        # BSplineBasis - from smooth LP
        slp = estimate_smooth_lp(Y, 1, 8; lags=2, degree=3, n_knots=3)
        io = IOBuffer()
        show(io, slp.spline_basis)
        out = String(take!(io))
        @test occursin("B-Spline", out)

        # StateTransition - from state LP
        state_var = Y[:, 2]
        state_lp = estimate_state_lp(Y, 1, state_var, 8; lags=2)
        io = IOBuffer()
        show(io, state_lp.state)
        out = String(take!(io))
        @test occursin("State Transition", out)

        # PropensityScoreConfig - from propensity LP
        treatment = Float64.(randn(100) .> 0)
        covariates = randn(100, 2)
        prop_lp = estimate_propensity_lp(Y, treatment, covariates, 8)
        io = IOBuffer()
        show(io, prop_lp.config)
        out = String(take!(io))
        @test occursin("Propensity Score", out)
    end

    # =================================================================
    # refs() comprehensive format and dispatch
    # =================================================================
    @testset "refs() comprehensive format and dispatch" begin
        # Symbol dispatch
        r = refs(:johansen)
        @test r isa String
        @test !isempty(r)

        # All four formats for various types
        Y = randn(100, 3)
        lp = estimate_lp(Y, 1, 8)
        for fmt in (:text, :latex, :bibtex, :html)
            r = refs(lp; format=fmt)
            @test r isa String
            @test !isempty(r)
        end

        # refs for BVAR posterior
        post = estimate_bvar(Y, 2; n_draws=50)
        r = refs(post)
        @test r isa String
        @test !isempty(r)

        # refs for filter types
        y = cumsum(randn(200))
        for f in [hp_filter(y), hamilton_filter(y)]
            r = refs(f)
            @test r isa String
            @test !isempty(r)
        end

        # LaTeX format should have \bibitem
        r_latex = refs(Y |> x -> estimate_var(x, 2); format=:latex)
        @test occursin("\\bibitem", r_latex)

        # BibTeX format should have @article or @book
        r_bib = refs(estimate_var(Y, 2); format=:bibtex)
        @test occursin("@article", r_bib) || occursin("@book", r_bib)

        # HTML format should have <p> tags
        r_html = refs(estimate_var(Y, 2); format=:html)
        @test occursin("<p>", r_html)

        # refs for ARIMA models
        ar = estimate_ar(randn(200), 2)
        r = refs(ar; format=:bibtex)
        @test r isa String
        @test !isempty(r)

        # refs for volatility models
        gm = estimate_garch(randn(300), 1, 1)
        r = refs(gm)
        @test r isa String
        @test !isempty(r)

        # refs for unit root tests
        adf_r = adf_test(randn(200))
        r = refs(adf_r)
        @test r isa String
    end

    # =================================================================
    # _delatex() Unicode replacements
    # =================================================================
    @testset "_delatex() Unicode replacements" begin
        @test MacroEconometricModels._delatex("L\\\"utkepohl") == "Lütkepohl"
        @test MacroEconometricModels._delatex("Jord\\'e") == "Jordé"
        @test MacroEconometricModels._delatex("em---dash") == "em\u2014dash"
        @test MacroEconometricModels._delatex("en--dash") == "en\u2013dash"
        @test MacroEconometricModels._delatex("\\&") == "&"
        @test MacroEconometricModels._delatex("{braces}") == "braces"
    end

    # =================================================================
    # Nowcast show() and report()
    # =================================================================
    @testset "Nowcast show() and report()" begin
        nM = 4; nQ = 1
        Y_nc = randn(100, nM + nQ)
        Y_nc[end, end] = NaN

        # NowcastDFM show
        dfm = nowcast_dfm(Y_nc, nM, nQ; r=2, p=1)
        io = IOBuffer()
        show(io, dfm)
        out = String(take!(io))
        @test occursin("DFM", out) || occursin("Dynamic Factor", out)

        # NowcastResult show
        nr = nowcast(dfm)
        io = IOBuffer()
        show(io, nr)
        out = String(take!(io))
        @test occursin("Nowcast", out)

        # report() for nowcast model and result
        redirect_stdout(devnull) do
            report(dfm)
            report(nr)
        end
        @test true

        # NowcastNews show
        X_old = randn(100, 5)
        X_old[end, end] = NaN
        X_new = copy(X_old)
        X_new[end, end] = 0.5
        dfm2 = nowcast_dfm(X_old, 4, 1; r=2, p=1)
        nn = nowcast_news(X_new, X_old, dfm2, 5)
        io = IOBuffer()
        show(io, nn)
        out = String(take!(io))
        @test occursin("News", out) || occursin("Revision", out)
        redirect_stdout(devnull) do
            report(nn)
        end
        @test true
    end

    # =================================================================
    # show() and print_table() for StructuralLP
    # =================================================================
    @testset "show() and print_table() for StructuralLP" begin
        Y = randn(100, 3)
        slp = structural_lp(Y, 8; method=:cholesky, lags=2)
        io = IOBuffer()
        show(io, slp)
        out = String(take!(io))
        @test occursin("Structural Local Projections", out) || occursin("Structural LP", out)

        # print_table
        io = IOBuffer()
        print_table(io, slp, 1, 1)
        out = String(take!(io))
        @test length(out) > 0
    end

    # =================================================================
    # show() and print_table() for LPForecast
    # =================================================================
    @testset "show() and print_table() for LPForecast" begin
        Y = randn(100, 3)
        lp = estimate_lp(Y, 1, 8; lags=2)
        shock_path = zeros(8); shock_path[1] = 1.0
        fc = forecast(lp, shock_path)
        io = IOBuffer()
        show(io, fc)
        out = String(take!(io))
        @test occursin("LP Forecast", out)

        io = IOBuffer()
        print_table(io, fc)
        out = String(take!(io))
        @test length(out) > 0
    end

    # =================================================================
    # show() and print_table() for LPFEVD
    # =================================================================
    @testset "show() and print_table() for LPFEVD" begin
        Y = randn(100, 3)
        slp = structural_lp(Y, 8; method=:cholesky, lags=2)
        f = lp_fevd(slp, 8; n_boot=0)
        io = IOBuffer()
        show(io, f)
        out = String(take!(io))
        @test occursin("LP-FEVD", out) || occursin("Gorodnichenko", out)

        # print_table for LPFEVD (no bootstrap)
        io = IOBuffer()
        print_table(io, f, 1)
        out = String(take!(io))
        @test length(out) > 0

        # Also test with n_boot>0 for CI columns
        f2 = lp_fevd(slp, 8; n_boot=20)
        io = IOBuffer()
        print_table(io, f2, 1)
        out = String(take!(io))
        @test length(out) > 0
    end

    # =================================================================
    # show() for AriasSVARResult and UhligSVARResult
    # =================================================================
    @testset "show() for AriasSVARResult and UhligSVARResult" begin
        Y = randn(100, 3)
        m = estimate_var(Y, 2)

        # Uhlig
        restrictions = SVARRestrictions(
            [ZeroRestriction(1, 2, 0)],
            [SignRestriction(1, 1, 0, 1)],
            3, 3
        )
        uhlig_r = identify_uhlig(m, restrictions, 10)
        io = IOBuffer()
        show(io, uhlig_r)
        out = String(take!(io))
        @test occursin("Mountford-Uhlig", out) || occursin("Uhlig", out)

        # Arias
        restrictions2 = SVARRestrictions(
            ZeroRestriction[],
            [SignRestriction(1, 1, 0, 1)],
            3, 3
        )
        arias_r = identify_arias(m, restrictions2, 10; n_draws=20)
        io = IOBuffer()
        show(io, arias_r)
        out = String(take!(io))
        @test occursin("Arias", out)
    end

    # =================================================================
    # table() for BayesianFEVD with stat=1
    # =================================================================
    @testset "table() for BayesianFEVD quantile stat" begin
        H, n = 8, 2
        nq = 3
        quantiles_arr = abs.(randn(H, n, n, nq))
        mean_arr = abs.(randn(H, n, n))
        vars = ["Var 1", "Var 2"]
        shocks = ["Shock 1", "Shock 2"]
        q_levels = [0.16, 0.5, 0.84]
        bfevd = BayesianFEVD{Float64}(quantiles_arr, mean_arr, H, vars, shocks, q_levels)

        # stat=1 uses first quantile
        t = table(bfevd, 1; stat=1)
        @test size(t, 1) == H
        @test size(t, 2) == n + 1
    end

end
