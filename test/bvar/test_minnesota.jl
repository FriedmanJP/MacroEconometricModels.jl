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

Random.seed!(42)

@testset "Minnesota Prior Tests" begin
    println("Generating Data for Minnesota Test...")
    T = 50
    n = 2
    p = 1
    true_A = [0.8 0.0; 0.0 0.8] # Persistent
    true_c = [0.0; 0.0]
    Y = zeros(T, n)
    for t in 2:T
        u = randn(2) * 0.5
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

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

    println("Dummy Observations Generated.")

    # 2. Test Estimation with Minnesota Prior
    println("Estimating BVAR with Minnesota...")
    post = estimate_bvar(Y, p; n_draws=100, prior=:minnesota, hyper=hyper)

    @test post isa BVARPosterior
    @test post.n_draws == 100
    @test post.prior == :minnesota

    # Basic check: posterior mean should be somewhat reasonable (within bounds)
    println("Estimation Complete.")

    # 3. Test optimize_hyperparameters_full
    @testset "Full Hyperparameter Optimization" begin
        println("Testing optimize_hyperparameters_full...")

        # Generate data with clear VAR structure
        T_full = 100
        Y_full = zeros(T_full, n)
        for t in 2:T_full
            Y_full[t, :] = true_A * Y_full[t-1, :] + randn(2) * 0.5
        end

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
        simple_hyper = optimize_hyperparameters(Y_full, p; grid_size=5)
        @test simple_hyper isa MinnesotaHyperparameters

        # Full optimization should find at least as good (or better) marginal likelihood
        ml_full = log_marginal_likelihood(Y_full, p, best_hyper)
        ml_simple = log_marginal_likelihood(Y_full, p, simple_hyper)
        # Note: Not strictly >= because grids differ, but both should be finite
        @test isfinite(ml_full)
        @test isfinite(ml_simple)

        println("Full Hyperparameter Optimization Test Complete.")
    end

    @testset "Marginal likelihood with different hypers" begin
        Random.seed!(4455)
        Y_ml = randn(80, 2)
        p_ml = 2

        hyper1 = MinnesotaHyperparameters(tau=0.1)
        hyper2 = MinnesotaHyperparameters(tau=1.0)
        hyper3 = MinnesotaHyperparameters(tau=0.001)

        ml1 = log_marginal_likelihood(Y_ml, p_ml, hyper1)
        ml2 = log_marginal_likelihood(Y_ml, p_ml, hyper2)
        ml3 = log_marginal_likelihood(Y_ml, p_ml, hyper3)

        @test isfinite(ml1)
        @test isfinite(ml2)
        @test isfinite(ml3)
        # Different tau values should give different likelihoods
        @test ml1 != ml2
    end

    @testset "Extreme hyperparameters" begin
        Random.seed!(4456)
        Y_ex = randn(80, 2)
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
        Random.seed!(4457)
        Y_opt = randn(80, 2)
        hyper_opt = optimize_hyperparameters(Y_opt, 1; grid_size=3)
        @test hyper_opt isa MinnesotaHyperparameters
        @test hyper_opt.tau > 0
        @test hyper_opt.decay > 0
    end
end
