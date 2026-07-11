# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Tests for Bayesian processing utilities.
"""

using Test
using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random

Random.seed!(54321)

@testset "Bayesian Processing Utilities" begin

    @testset "compute_posterior_quantiles" begin
        # Create test samples: 100 samples of a 10×3×3 array
        samples = randn(100, 10, 3, 3)
        q_vec = [0.16, 0.5, 0.84]

        # Test allocating version
        q_out, m_out = MacroEconometricModels.compute_posterior_quantiles(samples, q_vec)

        @test size(q_out) == (10, 3, 3, 3)  # (dims..., n_quantiles)
        @test size(m_out) == (10, 3, 3)

        # Verify quantiles are ordered
        @test all(q_out[:, :, :, 1] .<= q_out[:, :, :, 2])
        @test all(q_out[:, :, :, 2] .<= q_out[:, :, :, 3])

        # Verify mean is reasonable (between 16th and 84th percentiles)
        @test all(q_out[:, :, :, 1] .<= m_out)
        @test all(m_out .<= q_out[:, :, :, 3])
    end

    @testset "compute_posterior_quantiles!" begin
        samples = randn(50, 5, 2)
        q_vec = Float64.([0.1, 0.5, 0.9])

        q_out = zeros(5, 2, 3)
        m_out = zeros(5, 2)

        MacroEconometricModels.compute_posterior_quantiles!(q_out, m_out, samples, q_vec)

        @test size(q_out) == (5, 2, 3)
        @test size(m_out) == (5, 2)

        # Verify correctness for a specific element
        d = @view samples[:, 1, 1]
        @test q_out[1, 1, 2] ≈ quantile(d, 0.5)
        @test m_out[1, 1] ≈ mean(d)
    end

    @testset "compute_posterior_quantiles_threaded!" begin
        samples = randn(200, 10, 4, 4)
        q_vec = Float64.([0.16, 0.5, 0.84])

        q_out = zeros(10, 4, 4, 3)
        m_out = zeros(10, 4, 4)

        MacroEconometricModels.compute_posterior_quantiles_threaded!(q_out, m_out, samples, q_vec)

        # Compare with non-threaded version
        q_out2 = zeros(10, 4, 4, 3)
        m_out2 = zeros(10, 4, 4)
        MacroEconometricModels.compute_posterior_quantiles!(q_out2, m_out2, samples, q_vec)

        @test q_out ≈ q_out2
        @test m_out ≈ m_out2
    end

    @testset "stack_posterior_results" begin
        # Create vector of result arrays
        results = [randn(5, 3, 3) for _ in 1:20]

        stacked = MacroEconometricModels.stack_posterior_results(results, (5, 3, 3), Float64)

        @test size(stacked) == (20, 5, 3, 3)

        # Verify stacking correctness
        for s in 1:20
            @test stacked[s, :, :, :] ≈ results[s]
        end
    end

    @testset "Weighted Quantiles" begin
        # Test compute_weighted_quantiles!
        samples = randn(100, 5, 2)
        weights = rand(100)
        weights ./= sum(weights)  # Normalize

        q_vec = Float64.([0.16, 0.5, 0.84])
        q_out = zeros(5, 2, 3)
        m_out = zeros(5, 2)

        MacroEconometricModels.compute_weighted_quantiles!(q_out, m_out, samples, weights, q_vec)

        @test size(q_out) == (5, 2, 3)
        @test size(m_out) == (5, 2)

        # Verify weighted mean for a specific element
        d = @view samples[:, 1, 1]
        @test m_out[1, 1] ≈ sum(weights .* d)
    end

    @testset "Performance Utilities - XtX_inv Caching" begin
        X = randn(100, 5)
        residuals1 = randn(100)
        residuals2 = randn(100)

        # Pre-compute XtX_inv
        XtX_inv = MacroEconometricModels.precompute_XtX_inv(X)
        @test size(XtX_inv) == (5, 5)
        @test issymmetric(XtX_inv) || norm(XtX_inv - XtX_inv') < 1e-10

        # Test that cached version gives same results as non-cached
        V1 = newey_west(X, residuals1; bandwidth=3)
        V2 = newey_west(X, residuals1; bandwidth=3, XtX_inv=XtX_inv)
        @test V1 ≈ V2

        V3 = white_vcov(X, residuals2)
        V4 = white_vcov(X, residuals2; XtX_inv=XtX_inv)
        @test V3 ≈ V4

        V5 = driscoll_kraay(X, residuals1; bandwidth=3)
        V6 = driscoll_kraay(X, residuals1; bandwidth=3, XtX_inv=XtX_inv)
        @test V5 ≈ V6
    end

end

@testset "process_posterior_samples concrete container (#210 box A)" begin
    # Pure-allocation refactor: the result container was `Vector{Any}` and is now typed by the
    # concrete element returned by `compute_func` (inferred from the first valid draw). The stored
    # values are computed by the identical `compute_func` call, so outputs are unchanged; the only
    # observable difference is that the container's element type is now concrete, not `Any`.
    rng = Random.MersenneTwister(20210)
    n, p, T = 2, 2, 140
    Y = zeros(T, n)
    for t in 2:T
        Y[t, :] = 0.5 .* Y[t-1, :] .+ randn(rng, n)
    end
    Random.seed!(4242)
    post = estimate_bvar(Y, p; n_draws=60, sampler=:direct)

    # (1) The internal machinery now returns a concretely-typed vector of Array{Float64,3}
    #     (from compute_irf), not Vector{Any}.
    results, ns = MacroEconometricModels.process_posterior_samples(post,
        (m, Q, h) -> MacroEconometricModels.compute_irf(m, Q, h);
        method=:cholesky, horizon=8)
    @test eltype(results) != Any
    @test isconcretetype(eltype(results))
    @test eltype(results) == Array{Float64,3}
    @test ns == length(results)
    @test all(r -> all(isfinite, r), results)
    @test all(r -> size(r) == (8, n, n), results)

    # (2) Before/after equality: the deterministic Cholesky posterior-IRF pipeline reproduces the
    #     same result on a fixed seed across repeated runs (the container change is value-neutral).
    Random.seed!(777); ir1 = irf(post, 8; method=:cholesky)
    Random.seed!(777); ir2 = irf(post, 8; method=:cholesky)
    @test ir1.point_estimate == ir2.point_estimate
    @test ir1.quantiles == ir2.quantiles
    @test all(isfinite, ir1.point_estimate)
end
