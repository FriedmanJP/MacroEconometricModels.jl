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

rng = MersenneTwister(42)  # DGP-03: explicit rng

@testset "Minnesota Prior Tests" begin
    _tprint("Generating Data for Minnesota Test...")
    T = 50
    n = 2
    p = 1
    true_A = [0.8 0.0; 0.0 0.8] # Persistent
    # Reference DGP (DGP-03 #792): persistent diagonal A, B0 = 0.5I.
    Y = dgp_var(rng; A=true_A, B0=0.5 * Matrix{Float64}(I, n, n), T=T).Y

    # 1. Test Dummy Generation
    hyper = MinnesotaHyperparameters(tau=1.0, lambda=1.0, mu=1.0)
    Y_d, X_d = gen_dummy_obs(Y, p, hyper)

    # Check dimensions
    # n=2, p=1
    # AR priors: n*p = 2 rows
    # Sum coeffs: n = 2 rows
    # Dummy initial: 1 row
    # Covariance: n = 2 rows
    # Total = 2 + 2 + 1 + 2 = 7 rows
    @test size(Y_d, 1) == 7
    @test size(Y_d, 2) == 2
    @test size(X_d, 1) == 7
    @test size(X_d, 2) == 3 # 1 + 2*1

    _tprint("Dummy Observations Generated.")

    # 2. Test Estimation with Minnesota Prior
    _tprint("Estimating BVAR with Minnesota...")
    post = estimate_bvar(Y, p; n_draws=(FAST ? 50 : 100), prior=:minnesota, hyper=hyper, rng=rng)

    @test post isa BVARPosterior
    @test post.n_draws == (FAST ? 50 : 100)
    @test post.prior == :minnesota

    # Basic check: posterior mean should be somewhat reasonable (within bounds)
    _tprint("Estimation Complete.")

    # 3. Test optimize_hyperparameters_full
    @testset "Full Hyperparameter Optimization" begin
        _tprint("Testing optimize_hyperparameters_full...")

        # Generate data with clear VAR structure (DGP-03 #792: shared simulator)
        T_full = 100
        Y_full = dgp_var(rng; A=true_A, B0=0.5 * Matrix{Float64}(I, n, n), T=T_full).Y

        # Test with small grids for speed
        best_hyper, best_ml = optimize_hyperparameters_full(Y_full, p;
            tau_grid=range(0.1, 2.0, length=3),
            lambda_grid=[1.0, 5.0],
            mu_grid=[1.0, 2.0]
        )

        @test best_hyper isa MinnesotaHyperparameters
        @test best_hyper.tau > 0
        @test best_hyper.lambda > 0
        @test best_hyper.mu > 0
        @test isfinite(best_ml)
        @test best_ml > -Inf

        # Verify the returned hyperparameters are from the grid
        @test best_hyper.tau in range(0.1, 2.0, length=3)
        @test best_hyper.lambda in [1.0, 5.0]
        @test best_hyper.mu in [1.0, 2.0]

        # Compare with single-parameter optimization
        simple_hyper = optimize_hyperparameters(Y_full, p; grid_size=(FAST ? 3 : 5))
        @test simple_hyper isa MinnesotaHyperparameters

        # Full optimization should find at least as good (or better) marginal likelihood
        ml_full = log_marginal_likelihood(Y_full, p, best_hyper)
        ml_simple = log_marginal_likelihood(Y_full, p, simple_hyper)
        # Note: Not strictly >= because grids differ, but both should be finite
        @test isfinite(ml_full)
        @test isfinite(ml_simple)

        _tprint("Full Hyperparameter Optimization Test Complete.")
    end

    @testset "Shrinkage monotonicity and ML location (DGP-03 #792)" begin
        # White-noise truth (A = 0) under an RW-centred Minnesota prior: as tau
        # falls the posterior mean moves from the OLS estimate (≈ 0) to the
        # prior mean (I), and posterior variance shrinks. 500 direct draws make
        # posterior-mean MC error negligible next to the gaps (realized
        # d-to-I: 0.03 vs 1.37; d-to-0: 1.39 vs 0.14; variance 4x).
        rng = MersenneTwister(4455)
        Y_wn = dgp_var(rng; A=zeros(2, 2), B0=0.5 * Matrix{Float64}(I, 2, 2), T=200).Y
        post_tight = estimate_bvar(Y_wn, 1; n_draws=500, sampler=:direct,
            prior=:minnesota, hyper=MinnesotaHyperparameters(tau=0.01), rng=rng)
        post_loose = estimate_bvar(Y_wn, 1; n_draws=500, sampler=:direct,
            prior=:minnesota, hyper=MinnesotaHyperparameters(tau=10.0), rng=rng)
        A_tight = Matrix(dropdims(mean(post_tight.B_draws; dims=1); dims=1)[2:3, :]')
        A_loose = Matrix(dropdims(mean(post_loose.B_draws; dims=1); dims=1)[2:3, :]')
        # Tight prior pins the posterior at the RW prior mean I ...
        @test norm(A_tight - I(2)) < norm(A_loose - I(2))
        # ... while the loose prior lets it sit at the OLS estimate ≈ 0.
        @test norm(A_loose) < norm(A_tight)
        # Posterior variance shrinks with tau.
        @test sum(vec(var(post_tight.B_draws; dims=1))) <
              sum(vec(var(post_loose.B_draws; dims=1)))
        # Marginal likelihood on white-noise data rises as the RW prior loosens
        # (realized −429 → −354 → −307; closed form, zero MC noise).
        ml_001 = log_marginal_likelihood(Y_wn, 1, MinnesotaHyperparameters(tau=0.01))
        ml_01 = log_marginal_likelihood(Y_wn, 1, MinnesotaHyperparameters(tau=0.1))
        ml_10 = log_marginal_likelihood(Y_wn, 1, MinnesotaHyperparameters(tau=1.0))
        @test ml_10 > ml_01 > ml_001
        # ... while on RW truth (A = 0.9I) it peaks at an INTERIOR tau
        # (realized argmax 0.1: −326.5 beats −329.6 at 0.01 and −337.8 at 10).
        Y_rw = dgp_var(rng; A=0.9 * Matrix{Float64}(I, 2, 2),
                       B0=0.5 * Matrix{Float64}(I, 2, 2), T=200).Y
        ml_rw001 = log_marginal_likelihood(Y_rw, 1, MinnesotaHyperparameters(tau=0.01))
        ml_rw01 = log_marginal_likelihood(Y_rw, 1, MinnesotaHyperparameters(tau=0.1))
        ml_rw10 = log_marginal_likelihood(Y_rw, 1, MinnesotaHyperparameters(tau=10.0))
        @test ml_rw01 > ml_rw001
        @test ml_rw01 > ml_rw10
    end

    @testset "Extreme hyperparameters" begin
        rng = MersenneTwister(4456)  # DGP-03: explicit rng
        # White-noise DGP (DGP-03 #792).
        Y_ex = dgp_var(rng; A=zeros(2, 2), B0=0.5 * Matrix{Float64}(I, 2, 2), T=80).Y
        p_ex = 1

        # Very tight prior (tau=0.001)
        hyper_tight = MinnesotaHyperparameters(tau=0.001, decay=2.0, omega=0.5)
        ml_tight = log_marginal_likelihood(Y_ex, p_ex, hyper_tight)
        @test isfinite(ml_tight)

        # Very loose prior (tau=100)
        hyper_loose = MinnesotaHyperparameters(tau=100.0, decay=1.0, omega=1.0)
        ml_loose = log_marginal_likelihood(Y_ex, p_ex, hyper_loose)
        @test isfinite(ml_loose)
    end

    @testset "optimize_hyperparameters returns valid type" begin
        rng = MersenneTwister(4457)  # DGP-03: explicit rng
        # White-noise DGP (DGP-03 #792).
        Y_opt = dgp_var(rng; A=zeros(2, 2), B0=0.5 * Matrix{Float64}(I, 2, 2), T=80).Y
        hyper_opt = optimize_hyperparameters(Y_opt, 1; grid_size=(FAST ? 2 : 3))
        @test hyper_opt isa MinnesotaHyperparameters
        @test hyper_opt.tau > 0
        @test hyper_opt.decay > 0
    end
end
