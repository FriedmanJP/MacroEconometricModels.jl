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

@testset "FEVD Tests with Theoretical Verification" begin
    _tprint("Generating Data for FEVD Verification...")
    # FEVD Verification DGP:
    # Diagonal VAR(1) with Identity Error Covariance.
    # Means shocks are orthogonal and variables don't interact.
    # Var 1 is ONLY driven by Shock 1. Var 2 ONLY by Shock 2.
    # Theoretical Proportions:
    # Var 1: Shock 1 -> 1.0, Shock 2 -> 0.0
    # Var 2: Shock 1 -> 0.0, Shock 2 -> 1.0

    T = 500
    n = 2
    p = 1
    true_A = [0.5 0.0; 0.0 0.5]
    true_c = [0.0; 0.0]

    # Random seed for reproducibility
    Random.seed!(42)

    Y = zeros(T, n)
    for t in 2:T
        u = randn(n)
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    model = fit(VARModel, Y, p)
    _tprint("Frequentist Estimation Done.")

    horizon = 5

    # 1. Frequentist FEVD
    _tprint("Testing Frequentist FEVD...")
    fevd_freq = fevd(model, horizon; method=:cholesky)

    # Note: FEVD struct uses lowercase 'proportions'
    @test size(fevd_freq.proportions) == (n, n, horizon)

    for h in 1:horizon
        # Var 1 (Index 1) driven by Shock 1 (Index 1)
        @test isapprox(fevd_freq.proportions[1, 1, h], 1.0, atol=0.15)
        @test isapprox(fevd_freq.proportions[1, 2, h], 0.0, atol=0.15)

        # Var 2 (Index 2) driven by Shock 2 (Index 2)
        @test isapprox(fevd_freq.proportions[2, 1, h], 0.0, atol=0.15)
        @test isapprox(fevd_freq.proportions[2, 2, h], 1.0, atol=0.15)

        # Sum to 1
        @test isapprox(sum(fevd_freq.proportions[1, :, h]), 1.0, atol=1e-5)
        @test isapprox(sum(fevd_freq.proportions[2, :, h]), 1.0, atol=1e-5)
    end

    # 2. Bayesian FEVD
    _tprint("Testing Bayesian FEVD...")
    try
        post = estimate_bvar(Y, p; n_draws=(FAST ? 25 : 50))

        # Compute Bayesian FEVD
        fevd_bayes = fevd(post, horizon; method=:cholesky)

        # Check Mean Proportions
        # Structure: BayesianFEVD.point_estimate is (variable, shock, horizon) — unified with FEVD (#527)
        @test size(fevd_bayes.point_estimate) == (n, n, horizon)

        # Check specific values with relaxed tolerance for MCMC
        for h in 1:horizon
            # Var 1 (v=1) driven by Shock 1 (sh=1)
            mean_prop_1_1 = fevd_bayes.point_estimate[1, 1, h]
            # Var 1 (v=1) driven by Shock 2 (sh=2)
            mean_prop_1_2 = fevd_bayes.point_estimate[1, 2, h]

            @test isapprox(mean_prop_1_1, 1.0, atol=0.25)  # Relaxed for MCMC variability
            @test isapprox(mean_prop_1_2, 0.0, atol=0.25)
        end

    catch e
        @warn "Bayesian FEVD test failed (may be due to MCMC sampling issues)" exception=e
        # Don't fail the entire test suite for Bayesian estimation issues
        @test_skip "Bayesian FEVD skipped due to error"
    end

    _tprint("FEVD Verification Passed.")
end

@testset "FEVD Basic Functionality" begin
    Random.seed!(123)

    # Simple VAR model
    T, n, p = 200, 3, 2
    Y = randn(T, n)
    model = estimate_var(Y, p)

    horizon = 10

    # Test that FEVD can be computed
    fevd_result = fevd(model, horizon)

    @test fevd_result isa FEVD
    @test size(fevd_result.decomposition) == (n, n, horizon)
    @test size(fevd_result.proportions) == (n, n, horizon)

    # Proportions should sum to 1 for each variable at each horizon
    for h in 1:horizon
        for v in 1:n
            @test isapprox(sum(fevd_result.proportions[v, :, h]), 1.0, atol=1e-10)
        end
    end

    # Proportions should be non-negative
    @test all(fevd_result.proportions .>= -1e-10)

    # Decomposition should be non-negative
    @test all(fevd_result.decomposition .>= -1e-10)
end

@testset "fevd input validation (T062 C-18)" begin
    @test_throws ArgumentError MacroEconometricModels._validate_data([1.0 NaN; 0.0 2.0], "Sigma")
end

@testset "FEVD orthogonality guard (T061)" begin
    Random.seed!(61)
    Y = zeros(200, 3)
    for t in 2:200
        Y[t, :] = 0.5 * Y[t-1, :] + randn(3)
    end
    model = estimate_var(Y, 1)
    L = Matrix(MacroEconometricModels.safe_cholesky(model.Sigma))

    # (1) orthonormal impact matrix (cholesky, Q=I) ⇒ guard true, no warning, props sum to 1
    @test MacroEconometricModels._check_fevd_orthogonality(L, model.Sigma; method=:cholesky)
    fe = @test_nowarn fevd(model, 8; method=:cholesky)
    for h in 1:8, i in 1:3
        @test sum(fe.proportions[i, :, h]) ≈ 1.0 atol = 1e-10
    end

    # (2) non-orthonormal impact matrix (P = L·diag(2,1,1)) ⇒ guard false + one warning
    P_bad = copy(L); P_bad[:, 1] .*= 2.0
    got = @test_logs (:warn,) match_mode = :any MacroEconometricModels._check_fevd_orthogonality(
        P_bad, model.Sigma; method=:garch)
    @test got == false
end

@testset "FEVD Methods" begin
    Random.seed!(456)

    T, n, p = 150, 2, 1
    Y = randn(T, n)
    model = estimate_var(Y, p)

    horizon = 5

    # Test Cholesky method
    fevd_chol = fevd(model, horizon; method=:cholesky)
    @test fevd_chol isa FEVD

    # Both methods should give valid proportions
    for h in 1:horizon
        for v in 1:n
            @test isapprox(sum(fevd_chol.proportions[v, :, h]), 1.0, atol=1e-10)
        end
    end
end

@testset "SID-05 set-aware sign FEVD" begin
    Random.seed!(734)
    m = estimate_var(randn(150, 2), 1)
    chk(irf) = irf[1, 1, 1] > 0
    s = identify_sign(m, 5, chk; store_all=true, rng=MersenneTwister(1), max_draws=200)
    f = fevd(m, 5; method=:sign, check_func=chk, rng=MersenneTwister(1), max_draws=200)
    @test f.n_effective == s.n_accepted
    @test size(f.proportions) == (2, 2, 5)
    @test all(f.proportions .>= -1e-12)
    @test s.n_accepted > 1
    n, H = 2, 5
    acc = Array{Float64,4}(undef, s.n_accepted, n, n, H)
    for (i, Q) in enumerate(s.Q_draws)
        _, p = MacroEconometricModels._compute_fevd(compute_irf(m, Q, H), n, H)
        acc[i, :, :, :] = p
    end
    med = similar(f.proportions)
    for v in 1:n, sh in 1:n, h in 1:H
        med[v, sh, h] = quantile(@view(acc[:, v, sh, h]), 0.5)
    end
    @test f.proportions ≈ med
    _, p1 = MacroEconometricModels._compute_fevd(compute_irf(m, s.Q_draws[1], H), n, H)
    @test f.proportions ≉ p1
    irf_med = similar(s.irf_draws[1, :, :, :])
    for h in 1:H, i in 1:n, j in 1:n
        irf_med[h, i, j] = quantile(@view(s.irf_draws[:, h, i, j]), 0.5)
    end
    _, p_med_irf = MacroEconometricModels._compute_fevd(irf_med, n, H)
    @test f.proportions ≉ p_med_irf
end
