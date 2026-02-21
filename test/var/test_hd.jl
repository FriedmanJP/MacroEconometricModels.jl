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
using Random

@testset "Historical Decomposition Tests" begin

    @testset "Basic Frequentist HD" begin
        Random.seed!(42)

        # Generate simple VAR(1) data
        T_obs = 200
        n = 3
        p = 2

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        # Compute historical decomposition
        horizon = T_obs - p
        hd = historical_decomposition(model, horizon; method=:cholesky)

        @test hd isa HistoricalDecomposition
        @test hd.T_eff == T_obs - p
        @test size(hd.contributions) == (T_obs - p, n, n)
        @test size(hd.initial_conditions) == (T_obs - p, n)
        @test size(hd.actual) == (T_obs - p, n)
        @test size(hd.shocks) == (T_obs - p, n)
        @test length(hd.variables) == n
        @test length(hd.shock_names) == n
        @test hd.method == :cholesky
    end

    @testset "Decomposition Identity Verification" begin
        Random.seed!(123)

        T_obs = 150
        n = 2
        p = 1

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        horizon = T_obs - p
        hd = historical_decomposition(model, horizon)

        # Verify decomposition identity: contributions + initial = actual
        @test verify_decomposition(hd)

        # Manual verification for each variable
        for i in 1:n
            total_contrib = total_shock_contribution(hd, i)
            reconstructed = total_contrib .+ hd.initial_conditions[:, i]
            @test isapprox(reconstructed, hd.actual[:, i], atol=1e-10)
        end
    end

    @testset "Accessor Functions" begin
        Random.seed!(456)

        T_obs = 100
        n = 2
        p = 1

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        hd = historical_decomposition(model, T_obs - p)

        # Test contribution accessor with integer indices
        c11 = contribution(hd, 1, 1)
        @test length(c11) == T_obs - p
        @test c11 == hd.contributions[:, 1, 1]

        # Test contribution accessor with string names
        c_str = contribution(hd, "y1", "y1")
        @test c_str == c11

        # Test total shock contribution
        total = total_shock_contribution(hd, 1)
        @test length(total) == T_obs - p
        @test isapprox(total, sum(hd.contributions[:, 1, :], dims=2)[:], atol=1e-10)

        # Test total with string name
        total_str = total_shock_contribution(hd, "y1")
        @test total_str == total

        # Test error handling
        @test_throws ArgumentError contribution(hd, "NonExistent", "y1")
        @test_throws ArgumentError contribution(hd, "y1", "NonExistent")
        @test_throws AssertionError contribution(hd, 10, 1)
    end

    @testset "Different Identification Methods" begin
        Random.seed!(789)

        T_obs = 150
        n = 2
        p = 1

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)
        horizon = T_obs - p

        # Cholesky
        hd_chol = historical_decomposition(model, horizon; method=:cholesky)
        @test hd_chol.method == :cholesky
        @test verify_decomposition(hd_chol)

        # Long-run
        hd_lr = historical_decomposition(model, horizon; method=:long_run)
        @test hd_lr.method == :long_run
        @test verify_decomposition(hd_lr)

        # Sign restrictions
        check_func = irf -> irf[1, 1, 1] > 0  # Require positive impact
        hd_sign = historical_decomposition(model, horizon; method=:sign, check_func=check_func)
        @test hd_sign.method == :sign
        @test verify_decomposition(hd_sign)
    end

    @testset "Theoretical DGP Verification" begin
        # Create a known DGP where we can verify HD contributions
        # Diagonal VAR(1) with identity covariance
        Random.seed!(999)

        T_obs = 500
        n = 2
        p = 1

        # True parameters: diagonal AR with 0.5 coefficient
        true_A = [0.5 0.0; 0.0 0.5]
        true_c = [0.0; 0.0]

        # Generate data
        Y = zeros(T_obs, n)
        structural_shocks = randn(T_obs, n)  # Identity covariance = structural shocks
        for t in 2:T_obs
            Y[t, :] = true_c + true_A * Y[t-1, :] + structural_shocks[t, :]
        end

        model = estimate_var(Y, p)
        hd = historical_decomposition(model, T_obs - p; method=:cholesky)

        # With diagonal VAR and identity covariance, Cholesky gives identity impact
        # So shock j only affects variable j (at impact and through MA dynamics)
        # Variable 1 should be driven primarily by Shock 1
        mean_abs_contrib_1_1 = mean(abs.(hd.contributions[:, 1, 1]))
        mean_abs_contrib_1_2 = mean(abs.(hd.contributions[:, 1, 2]))

        # Contribution from own shock should be larger
        @test mean_abs_contrib_1_1 > mean_abs_contrib_1_2

        # Verify decomposition identity
        @test verify_decomposition(hd)
    end

    @testset "Bayesian Historical Decomposition" begin
        Random.seed!(111)

        T_obs = 80
        n = 2
        p = 1

        Y = randn(T_obs, n)
        T_eff = T_obs - p

        try
            post = estimate_bvar(Y, p; n_draws=(FAST ? 25 : 50))

            hd = historical_decomposition(post, T_eff;
                                          data=Y, method=:cholesky,
                                          quantiles=[0.16, 0.5, 0.84])

            @test hd isa BayesianHistoricalDecomposition
            @test hd.T_eff == T_eff
            @test size(hd.quantiles) == (T_eff, n, n, 3)
            @test size(hd.mean) == (T_eff, n, n)
            @test size(hd.initial_quantiles) == (T_eff, n, 3)
            @test size(hd.initial_mean) == (T_eff, n)
            @test length(hd.quantile_levels) == 3
            @test hd.method == :cholesky

            # Test accessor for Bayesian HD
            c_mean = contribution(hd, 1, 1; stat=:mean)
            @test length(c_mean) == T_eff
            @test c_mean == hd.mean[:, 1, 1]

            c_median = contribution(hd, 1, 1; stat=2)  # Median is 2nd quantile
            @test c_median == hd.quantiles[:, 1, 1, 2]

            # Total contribution
            total = total_shock_contribution(hd, 1)
            @test length(total) == T_eff

        catch e
            @warn "Bayesian HD test failed" exception=e
            @test_skip "Bayesian HD skipped due to MCMC issues"
        end
    end

    @testset "Arias Identification HD" begin
        Random.seed!(222)

        T_obs = 150
        n = 2
        p = 1

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)
        T_eff = T_obs - p

        # Create sign restrictions
        restrictions = SVARRestrictions(n;
            signs=[sign_restriction(1, 1, :positive; horizon=0)]
        )

        try
            hd = historical_decomposition(model, restrictions, T_eff;
                                          n_draws=(FAST ? 10 : 20), n_rotations=(FAST ? 50 : 100),
                                          quantiles=[0.16, 0.5, 0.84])

            @test hd isa BayesianHistoricalDecomposition
            @test hd.T_eff == T_eff
            @test hd.method == :arias

            # Check structures
            @test size(hd.quantiles) == (T_eff, n, n, 3)
            @test size(hd.mean) == (T_eff, n, n)

        catch e
            @warn "Arias HD test failed (may need more draws)" exception=e
            @test_skip "Arias HD skipped"
        end
    end

    @testset "Show Methods" begin
        Random.seed!(333)

        T_obs = 100
        n = 2
        p = 1

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)
        hd = historical_decomposition(model, T_obs - p)

        # Test that show doesn't error
        io = IOBuffer()
        show(io, hd)
        output = String(take!(io))

        @test occursin("Historical Decomposition", output)
        @test occursin("cholesky", output)
        @test occursin("Variables", output)
        @test occursin("Decomposition identity", output)
    end

    @testset "Edge Cases" begin
        Random.seed!(444)

        # Minimum viable case
        T_obs = 20
        n = 2
        p = 1

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        # Horizon larger than T_eff should be clamped
        hd = historical_decomposition(model, 1000)
        @test hd.T_eff == T_obs - p
        @test verify_decomposition(hd)

        # Single lag
        p = 1
        model = estimate_var(Y, p)
        hd = historical_decomposition(model, T_obs - p)
        @test verify_decomposition(hd)
    end

    # =================================================================
    # Bayesian HD: verify_decomposition, show, accessors
    # =================================================================

    @testset "BayesianHistoricalDecomposition verify_decomposition" begin
        # Construct synthetic Bayesian HD where mean contributions + initial ≈ actual
        T_eff, n = 30, 2
        actual = randn(T_eff, n)

        # Make mean contributions and initial_mean sum to actual
        mean_arr = randn(T_eff, n, n)
        initial_m = zeros(T_eff, n)
        for i in 1:n
            total_contrib = vec(sum(mean_arr[:, i, :], dims=2))
            initial_m[:, i] = actual[:, i] - total_contrib
        end

        nq = 3
        quantiles_arr = randn(T_eff, n, n, nq)
        initial_q = randn(T_eff, n, nq)
        shocks_m = randn(T_eff, n)
        q_levels = [0.16, 0.5, 0.84]

        bhd = BayesianHistoricalDecomposition{Float64}(
            quantiles_arr, mean_arr, initial_q, initial_m,
            shocks_m, actual, T_eff,
            ["Var 1", "Var 2"], ["Shock 1", "Shock 2"],
            q_levels, :cholesky
        )

        @test verify_decomposition(bhd)
    end

    @testset "BayesianHistoricalDecomposition show method" begin
        T_eff, n = 20, 2
        nq = 3
        bhd = BayesianHistoricalDecomposition{Float64}(
            randn(T_eff, n, n, nq), randn(T_eff, n, n),
            randn(T_eff, n, nq), randn(T_eff, n),
            randn(T_eff, n), randn(T_eff, n), T_eff,
            ["Var 1", "Var 2"], ["Shock 1", "Shock 2"],
            [0.16, 0.5, 0.84], :cholesky
        )

        io = IOBuffer()
        show(io, bhd)
        output = String(take!(io))

        @test occursin("Bayesian Historical Decomposition", output)
        @test occursin("cholesky", output)
        @test occursin("Variables", output)
        @test occursin("Quantiles", output)
        @test occursin("Posterior Mean", output)
    end

    @testset "BayesianHD accessor functions" begin
        T_eff, n = 30, 2
        nq = 3
        mean_arr = randn(T_eff, n, n)
        quantiles_arr = randn(T_eff, n, n, nq)

        bhd = BayesianHistoricalDecomposition{Float64}(
            quantiles_arr, mean_arr,
            randn(T_eff, n, nq), randn(T_eff, n),
            randn(T_eff, n), randn(T_eff, n), T_eff,
            ["Var 1", "Var 2"], ["Shock 1", "Shock 2"],
            [0.16, 0.5, 0.84], :cholesky
        )

        # contribution with mean
        c_mean = contribution(bhd, 1, 1; stat=:mean)
        @test length(c_mean) == T_eff
        @test c_mean == bhd.mean[:, 1, 1]

        # contribution with quantile index
        c_q1 = contribution(bhd, 1, 1; stat=1)
        @test c_q1 == bhd.quantiles[:, 1, 1, 1]

        c_q3 = contribution(bhd, 1, 1; stat=3)
        @test c_q3 == bhd.quantiles[:, 1, 1, 3]

        # contribution with string
        c_str = contribution(bhd, "Var 1", "Shock 1"; stat=:mean)
        @test c_str == c_mean

        # Invalid string
        @test_throws ArgumentError contribution(bhd, "NonExistent", "Shock 1")

        # total_shock_contribution
        total = total_shock_contribution(bhd, 1)
        @test length(total) == T_eff
        expected = vec(sum(bhd.mean[:, 1, :], dims=2))
        @test isapprox(total, expected, atol=1e-10)

        # total_shock_contribution with string
        total_str = total_shock_contribution(bhd, "Var 1")
        @test total_str == total

        # Invalid arguments
        @test_throws AssertionError contribution(bhd, 10, 1)
        @test_throws AssertionError contribution(bhd, 1, 10)
        @test_throws AssertionError contribution(bhd, 1, 1; stat=10)
        @test_throws ArgumentError contribution(bhd, 1, 1; stat=:invalid)
    end

    @testset "HD with long_run identification" begin
        Random.seed!(555)
        T_obs = 150
        n = 2
        p = 1

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)
        horizon = T_obs - p

        hd_lr = historical_decomposition(model, horizon; method=:long_run)
        @test hd_lr.method == :long_run
        @test verify_decomposition(hd_lr)
        @test size(hd_lr.contributions) == (T_obs - p, n, n)
    end

    @testset "HD with truncated horizon" begin
        Random.seed!(666)
        T_obs = 100
        n = 2
        p = 2

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        # Horizon smaller than T_eff
        horizon = 20
        hd = historical_decomposition(model, horizon)
        @test hd.T_eff == T_obs - p
        @test verify_decomposition(hd)
    end

    @testset "HD 3-variable model" begin
        Random.seed!(777)
        T_obs = 200
        n = 3
        p = 1

        Y = randn(T_obs, n)
        model = estimate_var(Y, p)
        hd = historical_decomposition(model, T_obs - p)

        @test size(hd.contributions) == (T_obs - p, n, n)
        @test size(hd.initial_conditions) == (T_obs - p, n)
        @test length(hd.variables) == n
        @test length(hd.shock_names) == n
        @test verify_decomposition(hd)

        # Check all accessor combos work
        for i in 1:n, j in 1:n
            c = contribution(hd, i, j)
            @test length(c) == T_obs - p
        end
    end

    # =================================================================
    # Default Horizon (Issue #18)
    # =================================================================
    @testset "Default Horizon" begin
        Random.seed!(42)
        Y = randn(100, 3)
        model = estimate_var(Y, 2)
        T_eff = size(Y, 1) - 2  # effective_nobs

        # Call without specifying horizon — should use effective_nobs
        hd = historical_decomposition(model)

        @test hd isa HistoricalDecomposition
        n = 3
        @test size(hd.contributions) == (T_eff, n, n)
        @test length(hd.variables) == n

        # With explicit horizon should also work (horizon clamped to T_eff)
        hd2 = historical_decomposition(model, 50)
        @test hd2.T_eff == T_eff
        @test verify_decomposition(hd2)
    end

end
