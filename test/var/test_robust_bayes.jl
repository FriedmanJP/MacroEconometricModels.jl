# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Tests for Giacomini–Kitagawa (2021) robust Bayes set-identified SVARs (SID-18 / #747).
"""

using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

# n=2 VAR with a known contemporaneous covariance. Impact IRFs depend only on Σ.
function _gk_n2_model(; Sigma=[1.0 0.4; 0.4 1.0], B=nothing)
    n, p = 2, 1
    Bmat = B === nothing ? [0.0 0.0; 0.3 0.1; 0.05 0.4] : B
    Y = zeros(20, n)
    U = zeros(19, n)
    VARModel(Y, p, Bmat, U, Matrix{Float64}(Sigma), 0.0, 0.0, 0.0, ["y1", "y2"])
end

# Closed-form impact bounds for shock 1 under IRF[0,1,1]>0 and IRF[0,2,1]>0
# (Gafarov–Meier–Montiel Olea 2018, n=2 one-shock program).
function _gmmo_n2_shock1_impact(Sigma)
    L = cholesky(Hermitian(Sigma)).L
    l11, l21, l22 = L[1, 1], L[2, 1], L[2, 2]
    y2_max = l21 > 0 ? hypot(l21, l22) : l22
    y1_max = l11
    return (L=L, y1_min=0.0, y1_max=y1_max, y2_min=0.0, y2_max=y2_max)
end

@testset "SID-18 Giacomini–Kitagawa robust Bayes" begin

    @testset "n=2 analytical impact bounds (solver=:optimize)" begin
        Sigma = [1.0 0.4; 0.4 1.0]
        m = _gk_n2_model(; Sigma=Sigma)
        r = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 1, :positive),
        ])
        tru = _gmmo_n2_shock1_impact(m.Sigma)
        lo, hi = identified_set_bounds(m, r, 1; solver=:optimize)
        @test size(lo) == (1, 2, 2)
        @test size(hi) == (1, 2, 2)
        @test lo[1, 2, 1] ≈ tru.y2_min atol=1e-6
        @test hi[1, 2, 1] ≈ tru.y2_max atol=1e-6
        @test lo[1, 1, 1] ≈ tru.y1_min atol=1e-6
        @test hi[1, 1, 1] ≈ tru.y1_max atol=1e-6
        @test all(lo .<= hi)
    end

    @testset "draw envelope converges to analytical bounds" begin
        Sigma = [1.0 0.4; 0.4 1.0]
        m = _gk_n2_model(; Sigma=Sigma)
        r = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 1, :positive),
        ])
        lo_opt, hi_opt = identified_set_bounds(m, r, 1; solver=:optimize)
        lo50, hi50 = identified_set_bounds(m, r, 1; solver=:draws, n_draws=50,
                                          rng=MersenneTwister(747))
        n_big = FAST ? 300 : 800
        lo_big, hi_big = identified_set_bounds(m, r, 1; solver=:draws, n_draws=n_big,
                                              rng=MersenneTwister(747))
        # Envelope is an inner approximation of the true set. Haar draws are
        # stochastic; require both envelopes inside the opt set and that more
        # draws are not a much worse inner approx.
        @test lo50[1, 2, 1] >= lo_opt[1, 2, 1] - 1e-6
        @test hi50[1, 2, 1] <= hi_opt[1, 2, 1] + 1e-6
        @test lo_big[1, 2, 1] >= lo_opt[1, 2, 1] - 1e-6
        @test hi_big[1, 2, 1] <= hi_opt[1, 2, 1] + 1e-6
        @test lo_big[1, 2, 1] <= lo50[1, 2, 1] + 0.05
        @test hi_big[1, 2, 1] >= hi50[1, 2, 1] - 0.05
        @test hi_big[1, 2, 1] > hi_opt[1, 2, 1] - 0.15
        @test lo_big[1, 2, 1] < lo_opt[1, 2, 1] + 0.15
    end

    @testset "empty identified set" begin
        m = _gk_n2_model()
        r_imp = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(1, 1, :negative),
        ])
        @test_throws IdentificationError identified_set_bounds(m, r_imp, 1; solver=:optimize)
        @test_throws IdentificationError identified_set_bounds(m, r_imp, 1; solver=:draws,
                                                              n_draws=20, rng=MersenneTwister(1))
    end

    @testset "identify_robust_bayes" begin
        rng = MersenneTwister(747)
        Y = randn(rng, 70, 2)
        post = estimate_bvar(Y, 1; n_draws=FAST ? 8 : 14, burnin=4, seed=747)
        r = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 1, :positive),
        ])
        nrot = FAST ? 15 : 30
        rng = MersenneTwister(747)
        res = identify_robust_bayes(post, r, 3; level=0.68, solver=:optimize,
                                    n_rotations=nrot, rng=copy(rng))
        @test res isa RobustBayesResult
        @test res isa AbstractAnalysisResult
        @test size(res.lower) == (3, 2, 2)
        @test size(res.upper) == (3, 2, 2)
        @test size(res.robust_lower) == (3, 2, 2)
        @test size(res.robust_upper) == (3, 2, 2)
        @test size(res.single_prior_lower) == (3, 2, 2)
        @test size(res.single_prior_upper) == (3, 2, 2)
        @test res.level == 0.68
        @test res.empty_set_prob == 0
        @test 0 <= res.informativeness <= 1
        @test all(res.lower .<= res.upper)
        @test all(res.robust_lower .<= res.robust_upper)
        @test all(res.single_prior_lower .<= res.single_prior_upper)
        # Single-prior interval is identify_arias_bayesian (all nonempty draws,
        # pooled weights, equal-tailed). GK guarantees Haar coverage of the CR,
        # not that the equal-tailed Haar interval ⊂ CR.
        rand(rng, UInt64, post.n_draws)
        q_lo = (1 - 0.68) / 2
        arias = identify_arias_bayesian(post, r, 3; n_rotations=nrot,
                                        quantiles=[q_lo, 0.5, 1 - q_lo],
                                        compute_weights=true, rng=rng)
        @test res.single_prior_lower ≈ arias.irf_quantiles[:, :, :, 1]
        @test res.single_prior_upper ≈ arias.irf_quantiles[:, :, :, end]
        w = arias.weights
        n_acc = size(arias.irf_draws, 1)
        H, n, _ = size(res.robust_lower)
        for h in 1:H, i in 1:n, j in 1:n
            cl = res.robust_lower[h, i, j]
            cu = res.robust_upper[h, i, j]
            (isfinite(cl) && isfinite(cu)) || continue
            mass = zero(eltype(w))
            for s in 1:n_acc
                η = arias.irf_draws[s, h, i, j]
                if cl - 1e-10 <= η <= cu + 1e-10
                    mass += w[s]
                end
            end
            @test mass >= res.level - 1e-12
        end
        @test has_uncertainty(res)
        @test point_estimate(res) ≈ (res.lower .+ res.upper) ./ 2
        rb = uncertainty_bounds(res)
        @test rb[1] === res.robust_lower
        @test rb[2] === res.robust_upper
    end

    @testset "empty-set probability" begin
        rng = MersenneTwister(748)
        Y = randn(rng, 60, 2)
        post = estimate_bvar(Y, 1; n_draws=FAST ? 6 : 10, burnin=3, seed=748)
        r_ok = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 1, :positive),
        ])
        ok = identify_robust_bayes(post, r_ok, 2; solver=:optimize, n_rotations=10,
                                   rng=MersenneTwister(748))
        @test ok.empty_set_prob == 0
        r_bad = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(1, 1, :negative),
        ])
        bad = identify_robust_bayes(post, r_bad, 2; solver=:optimize, n_rotations=5,
                                    rng=MersenneTwister(748))
        @test bad.empty_set_prob > 0
        @test bad.empty_set_prob == 1
    end

    @testset "report / refs / plot_result" begin
        rng = MersenneTwister(749)
        Y = randn(rng, 50, 2)
        post = estimate_bvar(Y, 1; n_draws=FAST ? 6 : 10, burnin=3, seed=749)
        r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        res = identify_robust_bayes(post, r, 2; solver=:optimize, n_rotations=10,
                                    rng=MersenneTwister(749))
        sh = sprint(show, res)
        @test occursin("Giacomini", sh) || occursin("Robust Bayes", sh)
        @test occursin("Informativeness", sh) || occursin("informativeness", sh)
        report(res)
        rs = sprint(refs, res)
        @test occursin("Giacomini", rs)
        @test occursin("Kitagawa", rs)
        @test occursin("Gafarov", rs)
        @test occursin("Baumeister", rs)
        p = plot_result(res)
        @test p isa PlotOutput
        p1 = plot_result(res; var=1, shock=1)
        @test p1 isa PlotOutput
    end

    @testset "argument checks" begin
        m = _gk_n2_model()
        r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        @test_throws ArgumentError identified_set_bounds(m, r, 1; solver=:bogus)
        @test_throws ArgumentError identified_set_bounds(m, r, 0; solver=:optimize)
        rng = MersenneTwister(1)
        Y = randn(rng, 40, 2)
        post = estimate_bvar(Y, 1; n_draws=4, burnin=1, seed=1)
        @test_throws ArgumentError identify_robust_bayes(post, r, 2; level=1.5)
        @test_throws ArgumentError identify_robust_bayes(post, r, 2; level=0)
    end

    @testset "smallest covering interval" begin
        lows = [0.0, 0.1, 0.0, 5.0, 0.2]
        highs = [1.0, 1.1, 1.2, 20.0, 0.9]
        cl, cu = MacroEconometricModels._smallest_covering_interval(lows, highs, 0.68)
        # 68% of 5 → k=4: drop the [5, 20] outlier; cover [0, 1.2].
        @test cl ≈ 0.0
        @test cu ≈ 1.2
        # A Haar-like inner point of the covered sets sits inside the CR.
        @test cl <= 0.5 <= cu
    end

    @testset "n>2 optimize vs draws envelope" begin
        n, p = 3, 1
        Sigma = [1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0]
        B = zeros(1 + n * p, n)
        B[2, 1] = 0.25
        B[3, 2] = 0.15
        B[4, 3] = 0.30
        Y = zeros(15, n)
        U = zeros(14, n)
        m = VARModel(Y, p, B, U, Matrix{Float64}(Sigma), 0.0, 0.0, 0.0,
                     ["y1", "y2", "y3"])
        r = SVARRestrictions(3; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 1, :positive),
        ])
        n_env = FAST ? 16 : 32
        seed = 747
        lo_d, hi_d = identified_set_bounds(m, r, 1; solver=:draws, n_draws=n_env,
                                           rng=MersenneTwister(seed), threaded=false)
        lo_o, hi_o = identified_set_bounds(m, r, 1; solver=:optimize, n_rotations=n_env,
                                           n_starts=2, rng=MersenneTwister(seed))
        @test all(lo_o .<= hi_o)
        @test all(lo_d .<= hi_d)
        # n=3 optimize with few starts can miss the Haar envelope; both are
        # approximations, so allow a modest optimizer miss.
        @test all(lo_o .<= lo_d .+ 0.15)
        @test all(hi_o .>= hi_d .- 0.15)
    end
end
