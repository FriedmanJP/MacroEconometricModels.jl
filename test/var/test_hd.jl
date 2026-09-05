# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using Random

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "Historical Decomposition Tests" begin

    @testset "Basic Frequentist HD" begin
        rng = MersenneTwister(42)  # DGP-02: explicit rng

        # Generate simple VAR(1) data
        T_obs = 200
        n = 3
        p = 2

        Y = randn(rng, T_obs, n)
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
        rng = MersenneTwister(123)  # DGP-02: explicit rng

        T_obs = 150
        n = 2
        p = 1

        Y = randn(rng, T_obs, n)
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
        rng = MersenneTwister(456)  # DGP-02: explicit rng

        T_obs = 100
        n = 2
        p = 1

        Y = randn(rng, T_obs, n)
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
        rng = MersenneTwister(789)  # DGP-02: explicit rng

        T_obs = 150
        n = 2
        p = 1

        Y = randn(rng, T_obs, n)
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

        # Sign restrictions — set-aware median (SID-05); adding-up identity not required
        check_func = irf -> irf[1, 1, 1] > 0  # Require positive impact
        hd_sign = historical_decomposition(model, horizon; method=:sign, check_func=check_func)
        @test hd_sign.method == :sign
        @test hd_sign.n_effective > 0

        hd_sign_md = historical_decomposition(model, horizon; method=:sign, check_func=check_func,
                                              max_draws=200, rng=MersenneTwister(734))
        @test hd_sign_md.method == :sign
        @test hd_sign_md.n_effective > 0
    end

    @testset "Theoretical DGP Verification" begin
        # Create a known DGP where we can verify HD contributions
        # Diagonal VAR(1) with identity covariance
        rng = MersenneTwister(999)  # DGP-02: explicit rng

        T_obs = 500
        n = 2
        p = 1

        # True parameters: diagonal AR with 0.5 coefficient
        true_A = [0.5 0.0; 0.0 0.5]
        true_c = [0.0; 0.0]

        # Generate data
        Y = zeros(T_obs, n)
        structural_shocks = randn(rng, T_obs, n)  # Identity covariance = structural shocks
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

    @testset "HD recovery on known (A, B0) DGP" begin
        # Non-diagonal A + non-identity B0: shock ordering matters, so a
        # transposed B0 or wrong ordering fails this testset (DGP-02 #791).
        rng = MersenneTwister(7320)
        # T = 4000: the weakest cell (variable 1 ← shock 3, indirect only)
        # clears 0.9; estimation noise scales 1/√T (0.85 at T = 2000).
        d = dgp_var(rng; T=4000)
        model = estimate_var(d.Y, 1)
        T_eff = size(d.Y, 1) - 1
        hd = historical_decomposition(model, T_eff; method=:cholesky)
        truth = var_hd(d.A, d.B0, d.eps)
        # Estimator rows 1:T_eff ↔ sample rows 2:T (first lag seeds initials);
        # late sample: initial-condition term decayed (max eig ≈ 0.6).
        late_hd = (T_eff - 499):T_eff
        late_true = ((T_eff - 499) + 1):size(d.Y, 1)
        for i in 1:3, j in 1:3
            @test cor(hd.contributions[late_hd, i, j], truth[late_true, i, j]) > 0.9
        end
        @test mean(abs.(hd.contributions[late_hd, :, :])) ≈
              mean(abs.(truth[late_true, :, :])) rtol=0.2
    end

    @testset "Bayesian Historical Decomposition" begin
        rng = MersenneTwister(111)  # DGP-02: explicit rng

        T_obs = 80
        n = 2
        p = 1

        # Structured DGP (DGP-02): no longer white noise.
        d = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], B0=[1.0 0.0; 0.3 1.0], T=T_obs)
        Y = d.Y
        T_eff = T_obs - p

        post = estimate_bvar(Y, p; n_draws=(FAST ? 25 : 50))

        hd = historical_decomposition(post, T_eff;
                                      data=Y, method=:cholesky,
                                      quantiles=[0.16, 0.5, 0.84])

        @test hd isa BayesianHistoricalDecomposition
        @test hd.T_eff == T_eff
        @test size(hd.quantiles) == (T_eff, n, n, 3)
        @test size(hd.point_estimate) == (T_eff, n, n)
        @test size(hd.initial_quantiles) == (T_eff, n, 3)
        @test size(hd.initial_point_estimate) == (T_eff, n)
        @test length(hd.quantile_levels) == 3
        @test hd.method == :cholesky

        # Test accessor for Bayesian HD
        c_mean = contribution(hd, 1, 1; stat=:mean)
        @test length(c_mean) == T_eff
        @test c_mean == hd.point_estimate[:, 1, 1]

        c_median = contribution(hd, 1, 1; stat=2)  # Median is 2nd quantile
        @test c_median == hd.quantiles[:, 1, 1, 2]

        # Total contribution
        total = total_shock_contribution(hd, 1)
        @test length(total) == T_eff
    end

    @testset "Arias Identification HD" begin
        rng = MersenneTwister(222)  # DGP-02: explicit rng

        T_obs = 150
        n = 2
        p = 1

        # Structured DGP (DGP-02): the (1,1)-positive restriction below holds
        # at the truth (B0[1,1] = 1), so this is not self-fulfilling.
        d = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], B0=[1.0 0.0; 0.3 1.0], T=T_obs)
        Y = d.Y
        model = estimate_var(Y, p)
        T_eff = T_obs - p

        # Create sign restrictions
        restrictions = SVARRestrictions(n;
            signs=[sign_restriction(1, 1, :positive; horizon=0)]
        )

        hd = historical_decomposition(model, restrictions, T_eff;
                                      n_draws=(FAST ? 10 : 20), n_rotations=(FAST ? 50 : 100),
                                      quantiles=[0.16, 0.5, 0.84])

        @test hd isa BayesianHistoricalDecomposition
        @test hd.T_eff == T_eff
        @test hd.method == :arias

        # Check structures
        @test size(hd.quantiles) == (T_eff, n, n, 3)
        @test size(hd.point_estimate) == (T_eff, n, n)
    end

    @testset "Show Methods" begin
        rng = MersenneTwister(333)  # DGP-02: explicit rng

        T_obs = 100
        n = 2
        p = 1

        Y = randn(rng, T_obs, n)
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
        rng = MersenneTwister(444)  # DGP-02: explicit rng

        # Minimum viable case
        T_obs = 20
        n = 2
        p = 1

        Y = randn(rng, T_obs, n)
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
        rng = MersenneTwister(7310)  # DGP-02: explicit rng (synthetic struct)
        # Construct synthetic Bayesian HD where mean contributions + initial ≈ actual
        T_eff, n = 30, 2
        actual = randn(rng, T_eff, n)

        # Make point_estimate contributions and initial_point_estimate sum to actual
        mean_arr = randn(rng, T_eff, n, n)
        initial_m = zeros(T_eff, n)
        for i in 1:n
            total_contrib = vec(sum(mean_arr[:, i, :], dims=2))
            initial_m[:, i] = actual[:, i] - total_contrib
        end

        nq = 3
        quantiles_arr = randn(rng, T_eff, n, n, nq)
        initial_q = randn(rng, T_eff, n, nq)
        shocks_m = randn(rng, T_eff, n)
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
        rng = MersenneTwister(7311)  # DGP-02: explicit rng (synthetic struct)
        T_eff, n = 20, 2
        nq = 3
        bhd = BayesianHistoricalDecomposition{Float64}(
            randn(rng, T_eff, n, n, nq), randn(rng, T_eff, n, n),
            randn(rng, T_eff, n, nq), randn(rng, T_eff, n),
            randn(rng, T_eff, n), randn(rng, T_eff, n), T_eff,
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
        rng = MersenneTwister(7312)  # DGP-02: explicit rng (synthetic struct)
        T_eff, n = 30, 2
        nq = 3
        mean_arr = randn(rng, T_eff, n, n)
        quantiles_arr = randn(rng, T_eff, n, n, nq)

        bhd = BayesianHistoricalDecomposition{Float64}(
            quantiles_arr, mean_arr,
            randn(rng, T_eff, n, nq), randn(rng, T_eff, n),
            randn(rng, T_eff, n), randn(rng, T_eff, n), T_eff,
            ["Var 1", "Var 2"], ["Shock 1", "Shock 2"],
            [0.16, 0.5, 0.84], :cholesky
        )

        # contribution with mean
        c_mean = contribution(bhd, 1, 1; stat=:mean)
        @test length(c_mean) == T_eff
        @test c_mean == bhd.point_estimate[:, 1, 1]

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
        expected = vec(sum(bhd.point_estimate[:, 1, :], dims=2))
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
        rng = MersenneTwister(555)  # DGP-02: explicit rng
        T_obs = 150
        n = 2
        p = 1

        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)
        horizon = T_obs - p

        hd_lr = historical_decomposition(model, horizon; method=:long_run)
        @test hd_lr.method == :long_run
        @test verify_decomposition(hd_lr)
        @test size(hd_lr.contributions) == (T_obs - p, n, n)
    end

    @testset "HD with truncated horizon" begin
        rng = MersenneTwister(666)  # DGP-02: explicit rng
        T_obs = 100
        n = 2
        p = 2

        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        # Horizon smaller than T_eff
        horizon = 20
        hd = historical_decomposition(model, horizon)
        @test hd.T_eff == T_obs - p
        @test verify_decomposition(hd)
    end

    @testset "HD 3-variable model" begin
        rng = MersenneTwister(777)  # DGP-02: explicit rng
        T_obs = 200
        n = 3
        p = 1

        Y = randn(rng, T_obs, n)
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
        rng = MersenneTwister(42)  # DGP-02: explicit rng
        Y = randn(rng, 100, 3)
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

    @testset "SID-05 set-aware sign HD" begin
        rng = MersenneTwister(734)  # DGP-02: explicit rng
        m = estimate_var(randn(rng, 150, 2), 1)
        chk(irf) = irf[1, 1, 1] > 0
        s = identify_sign(m, effective_nobs(m), chk; store_all=true, rng=MersenneTwister(1), max_draws=200)
        hd = historical_decomposition(m; method=:sign, check_func=chk, rng=MersenneTwister(1), max_draws=200)
        @test hd.n_effective == s.n_accepted
        @test s.n_accepted > 1
        T_eff = effective_nobs(m)
        actual = m.Y[(m.p + 1):end, :]
        n = 2
        acc = Array{Float64,4}(undef, s.n_accepted, T_eff, n, n)
        for (i, Q) in enumerate(s.Q_draws)
            contrib, _, _ = MacroEconometricModels._hd_from_Q(m, Q, T_eff, actual)
            acc[i, :, :, :] = contrib
        end
        med = similar(hd.contributions)
        for t in 1:T_eff, i in 1:n, j in 1:n
            med[t, i, j] = quantile(@view(acc[:, t, i, j]), 0.5)
        end
        @test hd.contributions ≈ med
        contrib1, _, _ = MacroEconometricModels._hd_from_Q(m, s.Q_draws[1], T_eff, actual)
        @test hd.contributions ≉ contrib1
    end

    @testset "SID-19 arias/uhlig HD" begin
        rng = MersenneTwister(748)  # DGP-02: explicit rng
        m = estimate_var(randn(rng, 80, 2), 1)
        r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        hda = historical_decomposition(m; method=:arias, restrictions=r,
                                       max_draws=20, rng=MersenneTwister(1))
        @test hda isa HistoricalDecomposition
        @test size(hda.contributions, 2) == 2
        hdu = historical_decomposition(m; method=:uhlig, restrictions=r,
                                       rng=MersenneTwister(2),
                                       n_starts=FAST ? 3 : 8, n_refine=1,
                                       max_iter_coarse=80, max_iter_fine=200)
        @test hdu isa HistoricalDecomposition
        @test verify_decomposition(hdu)
    end

end
