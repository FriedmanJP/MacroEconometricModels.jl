# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
using Random

@testset "Bayesian Samplers Tests" begin
    println("Testing BVAR samplers...")

    # Generate small synthetic data for speed
    T = 50
    n = 2
    p = 1
    Random.seed!(123)

    true_A = [0.5 0.0; 0.0 0.5]
    true_c = [0.0; 0.0]
    Y = zeros(T, n)
    for t in 2:T
        u = randn(2)
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    # Test Direct sampler (default, most commonly used)
    @testset "Direct Sampler" begin
        println("Testing sampler: direct")
        post = estimate_bvar(Y, p; n_draws=100, sampler=:direct)
        @test post isa BVARPosterior
        @test post.sampler == :direct
        @test post.n_draws == 100
        @test post.p == p
        @test post.n == n
        @test size(post.B_draws) == (100, 1 + n*p, n)
        @test size(post.Sigma_draws) == (100, n, n)
        @test all(isfinite.(post.B_draws))
        @test all(isfinite.(post.Sigma_draws))
        println("  -> Passed")
    end

    # Test Gibbs sampler
    @testset "Gibbs Sampler" begin
        println("Testing sampler: gibbs")
        post = estimate_bvar(Y, p; n_draws=50, sampler=:gibbs, burnin=50, thin=1)
        @test post isa BVARPosterior
        @test post.sampler == :gibbs
        @test post.n_draws == 50
        @test all(isfinite.(post.B_draws))
        @test all(isfinite.(post.Sigma_draws))
        println("  -> Passed")
    end

    # Test Gibbs with thinning
    @testset "Gibbs with Thinning" begin
        println("Testing sampler: gibbs with thin=2")
        post = estimate_bvar(Y, p; n_draws=30, sampler=:gibbs, burnin=50, thin=2)
        @test post isa BVARPosterior
        @test post.n_draws == 30
        println("  -> Passed")
    end

    # Test default burnin for Gibbs
    @testset "Gibbs Default Burnin" begin
        println("Testing gibbs default burnin (200 when not specified)")
        post = estimate_bvar(Y, p; n_draws=30, sampler=:gibbs)
        @test post isa BVARPosterior
        @test post.n_draws == 30
        println("  -> Passed")
    end

    # Test with Minnesota prior
    @testset "Direct with Minnesota Prior" begin
        println("Testing direct sampler with Minnesota prior")
        hyper = MinnesotaHyperparameters(tau=0.5)
        post = estimate_bvar(Y, p; n_draws=50, sampler=:direct, prior=:minnesota, hyper=hyper)
        @test post isa BVARPosterior
        @test post.prior == :minnesota
        println("  -> Passed")
    end

    @testset "Gibbs with Minnesota Prior" begin
        println("Testing gibbs sampler with Minnesota prior")
        hyper = MinnesotaHyperparameters(tau=0.5)
        post = estimate_bvar(Y, p; n_draws=30, sampler=:gibbs, burnin=30,
                             prior=:minnesota, hyper=hyper)
        @test post isa BVARPosterior
        @test post.prior == :minnesota
        println("  -> Passed")
    end

    # Test error for unknown sampler
    @testset "Unknown Sampler Error" begin
        @test_throws ArgumentError estimate_bvar(Y, p; sampler=:nonexistent, n_draws=50)
    end

    # Test Sigma positive definiteness
    @testset "Sigma Positive Definiteness" begin
        post = estimate_bvar(Y, p; n_draws=50, sampler=:direct)
        for s in 1:post.n_draws
            S = post.Sigma_draws[s, :, :]
            @test isapprox(S, S', atol=1e-10)  # Symmetric
            eigs = eigvals(Symmetric(S))
            @test all(eigs .> -1e-10)  # PD (up to numerical precision)
        end
    end
end
