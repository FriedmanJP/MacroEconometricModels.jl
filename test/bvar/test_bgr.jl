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

@testset "BGR 2010 Optimization" begin
    _tprint("Testing BGR 2010 Hyperparameter Optimization...")
    rng = MersenneTwister(1001)  # DGP-03: explicit rng

    # Generate synthetic data (VAR(1))
    T = 60
    n = 3
    p = 1

    # Small stationary VAR(1) (DGP-03 #792: shared simulator, no A' idiom).
    Y = dgp_var(rng; A=0.4 * Matrix{Float64}(I, n, n),
                B0=0.5 * Matrix{Float64}(I, n, n), T=T).Y

    # 1. Test Log Marginal Likelihood
    _tprint("Testing Marginal Likelihood...")
    hyper = MinnesotaHyperparameters(tau=0.2)
    ml = log_marginal_likelihood(Y, p, hyper)
    _tprint("ML (tau=0.2): ", ml)
    @test ml isa Float64
    @test !isnan(ml)

    # 2. Test Optimization matching
    _tprint("Testing Optimization...")
    best_hyper = optimize_hyperparameters(Y, p; grid_size=(FAST ? 5 : 10))
    _tprint("Optimal Tau: ", best_hyper.tau)

    @test best_hyper.tau > 0
    @test best_hyper isa MinnesotaHyperparameters

    # Comparison: Very tight prior (tau -> 0) vs Loose prior (tau -> Inf) relative to data info.
    # Usually a balanced tau is found.

    ml_opt = log_marginal_likelihood(Y, p, best_hyper)
    ml_bad = log_marginal_likelihood(Y, p, MinnesotaHyperparameters(tau=100.0))

    _tprint("ML Optimal: ", ml_opt)
    _tprint("ML Loose:   ", ml_bad)

    # Ideally optimization found a peak.
    @test ml_opt >= ml_bad
end

@testset "BGR 2010: Large Sparse VAR" begin
    _tprint("\nTesting Large Sparse VAR (N=20)...")

    # 3. Large Sparse DGP
    # Replicating BGR style environment: Many vars, short T relative to params
    # DGP: Matrix of 20 series, mostly Random Walk (diagonal A=I)
    T_large = 100
    n_large = 20
    p_large = 1

    # Near-RW diagonal VAR (DGP-03 #792: shared simulator, no A' idiom;
    # rng first — the old code drew Y_large[1,:] BEFORE Random.seed!).
    rng = MersenneTwister(999)
    Y_large = dgp_var(rng; A=0.9 * Matrix{Float64}(I, n_large, n_large),
                      B0=Matrix{Float64}(I, n_large, n_large), T=T_large).Y

    # Optimize Hyperparameters
    _tprint("Optimizing Hyperparameters for Large VAR...")
    # This might take a moment due to larger matrix inversions
    @time best_hyper_large = optimize_hyperparameters(Y_large, p_large; grid_size=(FAST ? 5 : 10))

    _tprint("Optimal Tau (Large): ", best_hyper_large.tau)

    # For Large VARs, we expect tighter priors (smaller tau) to prevent overfitting
    # compared to loose priors, especially if N is very large.
    # BGR finding: As N increases, optimal lambda (tau) decreases.

    ml_opt = log_marginal_likelihood(Y_large, p_large, best_hyper_large)
    ml_loose = log_marginal_likelihood(Y_large, p_large, MinnesotaHyperparameters(tau=10.0))

    _tprint("ML Optimal (Large): ", ml_opt)
    _tprint("ML Loose (Large):   ", ml_loose)

    @test ml_opt > ml_loose
    @test !isnan(ml_opt)
    @test best_hyper_large.tau < 10.0 # Should prefer some shrinkage

    # BGR finding (DGP-03 #792): the large system wants strictly tighter
    # shrinkage than the small stationary one (realized 0.01 vs 2.51/1.12 —
    # the large grid optimum sits at the floor in both grid modes).
    Y_small = dgp_var(MersenneTwister(1001); A=0.4 * Matrix{Float64}(I, 3, 3),
                      B0=0.5 * Matrix{Float64}(I, 3, 3), T=60).Y
    best_hyper_small = optimize_hyperparameters(Y_small, 1; grid_size=(FAST ? 5 : 10))
    @test best_hyper_large.tau < best_hyper_small.tau
end
