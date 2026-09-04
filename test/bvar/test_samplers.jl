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

@testset "Bayesian Samplers Tests" begin
    _tprint("Testing BVAR samplers...")

    # Reference DGP (DGP-03 #792): non-diagonal A, non-identity B0, burn-in.
    rng = MersenneTwister(123)
    T = 400
    n = 2
    p = 1

    true_A = [0.5 0.1; 0.0 0.4]
    B0_true = [1.0 0.0; 0.3 1.0]
    Y = dgp_var(rng; A=true_A, B0=B0_true, T=T).Y

    # Test Direct sampler (default, most commonly used)
    @testset "Direct Sampler" begin
        _tprint("Testing sampler: direct")
        post = estimate_bvar(Y, p; n_draws=(FAST ? 50 : 100), sampler=:direct, rng=rng)
        @test post isa BVARPosterior
        @test post.sampler == :direct
        @test post.n_draws == (FAST ? 50 : 100)
        @test post.p == p
        @test post.n == n
        @test size(post.B_draws) == ((FAST ? 50 : 100), 1 + n*p, n)
        @test size(post.Sigma_draws) == ((FAST ? 50 : 100), n, n)
        @test all(isfinite.(post.B_draws))
        @test all(isfinite.(post.Sigma_draws))
        _tprint("  -> Passed")
    end

    # Test Gibbs sampler
    @testset "Gibbs Sampler" begin
        _tprint("Testing sampler: gibbs")
        post = estimate_bvar(Y, p; n_draws=(FAST ? 25 : 50), sampler=:gibbs, burnin=(FAST ? 25 : 50), thin=1, rng=rng)
        @test post isa BVARPosterior
        @test post.sampler == :gibbs
        @test post.n_draws == (FAST ? 25 : 50)
        @test all(isfinite.(post.B_draws))
        @test all(isfinite.(post.Sigma_draws))
        _tprint("  -> Passed")
    end

    # Test Gibbs with thinning
    @testset "Gibbs with Thinning" begin
        _tprint("Testing sampler: gibbs with thin=2")
        post = estimate_bvar(Y, p; n_draws=(FAST ? 15 : 30), sampler=:gibbs, burnin=(FAST ? 25 : 50), thin=2, rng=rng)
        @test post isa BVARPosterior
        @test post.n_draws == (FAST ? 15 : 30)
        _tprint("  -> Passed")
    end

    # Test default burnin for Gibbs
    @testset "Gibbs Default Burnin" begin
        _tprint("Testing gibbs default burnin (200 when not specified)")
        post = estimate_bvar(Y, p; n_draws=(FAST ? 15 : 30), sampler=:gibbs, rng=rng)
        @test post isa BVARPosterior
        @test post.n_draws == (FAST ? 15 : 30)
        _tprint("  -> Passed")
    end

    # Test with Minnesota prior
    @testset "Direct with Minnesota Prior" begin
        _tprint("Testing direct sampler with Minnesota prior")
        hyper = MinnesotaHyperparameters(tau=0.5)
        post = estimate_bvar(Y, p; n_draws=(FAST ? 25 : 50), sampler=:direct, prior=:minnesota, hyper=hyper, rng=rng)
        @test post isa BVARPosterior
        @test post.prior == :minnesota
        _tprint("  -> Passed")
    end

    @testset "Gibbs with Minnesota Prior" begin
        _tprint("Testing gibbs sampler with Minnesota prior")
        hyper = MinnesotaHyperparameters(tau=0.5)
        post = estimate_bvar(Y, p; n_draws=(FAST ? 15 : 30), sampler=:gibbs, burnin=(FAST ? 15 : 30),
                             prior=:minnesota, hyper=hyper, rng=rng)
        @test post isa BVARPosterior
        @test post.prior == :minnesota
        _tprint("  -> Passed")
    end

    # DGP-03 #792: direct (NIW) and Gibbs target the SAME posterior. On the
    # same data, posterior means agree within 3 Monte-Carlo SEs (ESS-corrected
    # for Gibbs autocorrelation), variances within rtol 0.3, and both recover
    # A within 2 posterior sd at T=400. A Gibbs sampler with a wrong
    # conditional (e.g. drawing Sigma from the prior) fails the agreement.
    @testset "Direct-Gibbs equivalence and recovery" begin
        _tprint("Testing direct-vs-Gibbs posterior equivalence")
        nd = 2000
        post_d = estimate_bvar(Y, p; n_draws=nd, sampler=:direct, rng=MersenneTwister(31))
        post_g = estimate_bvar(Y, p; n_draws=nd, sampler=:gibbs, burnin=500, rng=MersenneTwister(32))

        Bd, Bg = post_d.B_draws, post_g.B_draws
        @test size(Bd) == size(Bg) == (nd, 1 + n * p, n)
        for j in axes(Bd, 2), k in axes(Bd, 3)
            a, b = Bd[:, j, k], Bg[:, j, k]
            # ESS via lag-1 autocorrelation (direct draws are iid; Gibbs may
            # autocorrelate, which only widens the bound).
            ra = cor(a[1:end-1], a[2:end])
            rb = cor(b[1:end-1], b[2:end])
            ea = isfinite(ra) ? nd * (1 - ra) / (1 + ra) : nd
            eb = isfinite(rb) ? nd * (1 - rb) / (1 + rb) : nd
            se = sqrt(var(a) / ea + var(b) / eb)
            @test abs(mean(a) - mean(b)) <= 3 * se  # 3-MCSE agreement
            @test isapprox(var(a), var(b); rtol=0.3)  # variance agreement
        end
        # Sigma posterior means agree (realized max diff 0.01 ≈ 1% of Sigma).
        Sd = dropdims(mean(post_d.Sigma_draws; dims=1); dims=1)
        Sg = dropdims(mean(post_g.Sigma_draws; dims=1); dims=1)
        @test Sd ≈ Sg atol = 0.03
        # Both recover A within 2 posterior sd (realized max 1.35).
        for post in (post_d, post_g)
            Bm = dropdims(mean(post.B_draws; dims=1); dims=1)
            Bs = dropdims(std(post.B_draws; dims=1); dims=1)
            @test maximum(abs.(Bm[2:3, :]' .- true_A) ./ Bs[2:3, :]') < 2
        end
        _tprint("  -> Passed")
    end

    # Test error for unknown sampler
    @testset "Unknown Sampler Error" begin
        @test_throws ArgumentError estimate_bvar(Y, p; sampler=:nonexistent, n_draws=50)
    end

    # Test Sigma positive definiteness
    @testset "Sigma Positive Definiteness" begin
        post = estimate_bvar(Y, p; n_draws=(FAST ? 25 : 50), sampler=:direct, rng=rng)
        for s in 1:post.n_draws
            S = post.Sigma_draws[s, :, :]
            @test isapprox(S, S', atol=1e-10)  # Symmetric
            eigs = eigvals(Symmetric(S))
            @test all(eigs .> -1e-10)  # PD (up to numerical precision)
        end
    end
end
