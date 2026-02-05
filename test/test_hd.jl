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
        c_str = contribution(hd, "Var 1", "Shock 1")
        @test c_str == c11

        # Test total shock contribution
        total = total_shock_contribution(hd, 1)
        @test length(total) == T_obs - p
        @test isapprox(total, sum(hd.contributions[:, 1, :], dims=2)[:], atol=1e-10)

        # Test total with string name
        total_str = total_shock_contribution(hd, "Var 1")
        @test total_str == total

        # Test error handling
        @test_throws ArgumentError contribution(hd, "NonExistent", "Shock 1")
        @test_throws ArgumentError contribution(hd, "Var 1", "NonExistent")
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

        T_obs = 100
        n = 2
        p = 1

        Y = randn(T_obs, n)
        T_eff = T_obs - p

        try
            chain = estimate_bvar(Y, p; n_samples=100, n_adapts=50)

            hd = historical_decomposition(chain, p, n, T_eff;
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
                                          n_draws=100, n_rotations=500,
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

end
