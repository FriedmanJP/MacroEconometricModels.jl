using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using Random

@testset "BGR 2010 Optimization" begin
    println("Testing BGR 2010 Hyperparameter Optimization...")

    # Generate synthetic data (VAR(1))
    T = 60
    n = 3
    p = 1

    # Stable VAR
    A = 0.4 * I(n)
    Y = zeros(T, n)
    for t in 2:T
        Y[t, :] = Y[t-1, :]' * A + 0.5 * randn(n)'
    end

    # 1. Test Log Marginal Likelihood
    println("Testing Marginal Likelihood...")
    hyper = MinnesotaHyperparameters(tau=0.2)
    ml = log_marginal_likelihood(Y, p, hyper)
    println("ML (tau=0.2): ", ml)
    @test ml isa Float64
    @test !isnan(ml)

    # 2. Test Optimization matching
    println("Testing Optimization...")
    best_hyper = optimize_hyperparameters(Y, p; grid_size=10)
    println("Optimal Tau: ", best_hyper.tau)

    @test best_hyper.tau > 0
    @test best_hyper isa MinnesotaHyperparameters

    # Comparison: Very tight prior (tau -> 0) vs Loose prior (tau -> Inf) relative to data info.
    # Usually a balanced tau is found.

    ml_opt = log_marginal_likelihood(Y, p, best_hyper)
    ml_bad = log_marginal_likelihood(Y, p, MinnesotaHyperparameters(tau=100.0))

    println("ML Optimal: ", ml_opt)
    println("ML Loose:   ", ml_bad)

    # Ideally optimization found a peak.
    @test ml_opt >= ml_bad
end

@testset "BGR 2010: Large Sparse VAR" begin
    println("\nTesting Large Sparse VAR (N=20)...")

    # 3. Large Sparse DGP
    # Replicating BGR style environment: Many vars, short T relative to params
    # DGP: Matrix of 20 series, mostly Random Walk (diagonal A=I)
    T_large = 100
    n_large = 20
    p_large = 1

    # Sparse Transition: Diagonal 0.9 (Persistent), rest 0
    A_large = zeros(n_large, n_large)
    for i in 1:n_large
        A_large[i, i] = 0.9
    end

    Y_large = zeros(T_large, n_large)
    # Initialize near mean
    Y_large[1, :] = randn(n_large)

    Random.seed!(999)
    for t in 2:T_large
        Y_large[t, :] = Y_large[t-1, :]' * A_large + randn(n_large)'
    end

    # Optimize Hyperparameters
    println("Optimizing Hyperparameters for Large VAR...")
    # This might take a moment due to larger matrix inversions
    @time best_hyper_large = optimize_hyperparameters(Y_large, p_large; grid_size=10)

    println("Optimal Tau (Large): ", best_hyper_large.tau)

    # For Large VARs, we expect tighter priors (smaller tau) to prevent overfitting
    # compared to loose priors, especially if N is very large.
    # BGR finding: As N increases, optimal lambda (tau) decreases.

    ml_opt = log_marginal_likelihood(Y_large, p_large, best_hyper_large)
    ml_loose = log_marginal_likelihood(Y_large, p_large, MinnesotaHyperparameters(tau=10.0))

    println("ML Optimal (Large): ", ml_opt)
    println("ML Loose (Large):   ", ml_loose)

    @test ml_opt > ml_loose
    @test !isnan(ml_opt)
    @test best_hyper_large.tau < 10.0 # Should prefer some shrinkage
end
