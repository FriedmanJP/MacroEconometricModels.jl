# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Tests for Bayesian processing utilities.
"""

using Test
using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random

rng = Xoshiro(54321)  # DGP-03: explicit rng

@testset "Bayesian Processing Utilities" begin

    @testset "compute_posterior_quantiles" begin
        # Create test samples: 100 samples of a 10×3×3 array
        samples = randn(rng, 100, 10, 3, 3)
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
        samples = randn(rng, 50, 5, 2)
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
        samples = randn(rng, 200, 10, 4, 4)
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
        results = [randn(rng, 5, 3, 3) for _ in 1:20]

        stacked = MacroEconometricModels.stack_posterior_results(results, (5, 3, 3), Float64)

        @test size(stacked) == (20, 5, 3, 3)

        # Verify stacking correctness
        for s in 1:20
            @test stacked[s, :, :, :] ≈ results[s]
        end
    end

    @testset "Weighted Quantiles" begin
        # Test compute_weighted_quantiles!
        samples = randn(rng, 100, 5, 2)
        weights = rand(rng, 100)
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
        X = randn(rng, 100, 5)
        residuals1 = randn(rng, 100)
        residuals2 = randn(rng, 100)

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
    rng = Random.Xoshiro(20210)
    n, p, T = 2, 2, 140
    # Persistent VAR(1) truth (DGP-03 #792: shared simulator).
    Y = dgp_var(rng; A=[0.5 0.1; 0.05 0.4], B0=Matrix{Float64}(I, n, n), T=T).Y
    post = estimate_bvar(Y, p; n_draws=60, sampler=:direct, rng=Xoshiro(4242))

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
    # T159: Cholesky impact Gram reproduces the draw's Sigma (observed ≤7e-16 in VAR IRF-CI).
    for d in eachindex(results)
        @test maximum(abs, results[d][1, :, :] * results[d][1, :, :]' .- post.Sigma_draws[d, :, :]) < 1e-12
    end

    # (2) Before/after equality: the deterministic Cholesky posterior-IRF pipeline reproduces the
    #     same result across repeated runs with NO reseeding (the container change is value-neutral;
    #     DGP-03 #792: the old global reseeds were a crutch — determinism must not need them).
    ir1 = irf(post, 8; method=:cholesky)
    ir2 = irf(post, 8; method=:cholesky)
    @test ir1.point_estimate == ir2.point_estimate
    @test ir1.quantiles == ir2.quantiles
    @test all(isfinite, ir1.point_estimate)
    # T159: default point estimate is the mean over the per-draw IRFs (observed maxdiff 1e-16).
    stack_t159 = cat(results...; dims=4)
    @test maximum(abs, ir1.point_estimate .- dropdims(mean(stack_t159; dims=4); dims=4)) < 1e-12
end

@testset "BVAR threaded=false vs threaded=true bit-exact parity (#945)" begin
    rng = Random.Xoshiro(94501)
    n, p, T = 3, 2, 120
    A_true = [0.5 0.1 0.0; 0.05 0.4 0.05; 0.0 0.1 0.45]
    Y = dgp_var(rng; A=A_true, B0=Matrix{Float64}(I, n, n), T=T).Y
    post = estimate_bvar(Y, p; n_draws=40, sampler=:direct, rng=Xoshiro(94502))

    # Cholesky IRF / FEVD / HD bit-exact parity
    ir_s = irf(post, 10; method=:cholesky, threaded=false)
    ir_t = irf(post, 10; method=:cholesky, threaded=true)
    @test ir_s.point_estimate == ir_t.point_estimate
    @test ir_s.quantiles == ir_t.quantiles
    @test ir_s._draws == ir_t._draws

    fv_s = fevd(post, 10; method=:cholesky, threaded=false)
    fv_t = fevd(post, 10; method=:cholesky, threaded=true)
    @test fv_s.point_estimate == fv_t.point_estimate
    @test fv_s.quantiles == fv_t.quantiles

    hd_s = historical_decomposition(post; method=:cholesky, threaded=false)
    hd_t = historical_decomposition(post; method=:cholesky, threaded=true)
    @test hd_s.point_estimate == hd_t.point_estimate
    @test hd_s.quantiles == hd_t.quantiles
    @test hd_s.initial_point_estimate == hd_t.initial_point_estimate

    # Long-run IRF & HD bit-exact parity
    lr_s = irf(post, 8; method=:long_run, threaded=false)
    lr_t = irf(post, 8; method=:long_run, threaded=true)
    @test lr_s.point_estimate == lr_t.point_estimate
    @test lr_s.quantiles == lr_t.quantiles

    # Sign-restricted IRF & HD bit-exact parity with explicit RNG
    chk = ir -> (ir[1, 1, 1] > 0 && ir[1, 2, 2] > 0)
    sg_s = irf(post, 8; method=:sign, check_func=chk, max_draws=200,
               rng=Xoshiro(94503), threaded=false)
    sg_t = irf(post, 8; method=:sign, check_func=chk, max_draws=200,
               rng=Xoshiro(94503), threaded=true)
    @test sg_s.point_estimate == sg_t.point_estimate
    @test sg_s.quantiles == sg_t.quantiles

    hds_s = historical_decomposition(post; method=:sign, check_func=chk, max_draws=200,
                                     rng=Xoshiro(94504), threaded=false)
    hds_t = historical_decomposition(post; method=:sign, check_func=chk, max_draws=200,
                                     rng=Xoshiro(94504), threaded=true)
    @test hds_s.point_estimate == hds_t.point_estimate
    @test hds_s.quantiles == hds_t.quantiles
end
