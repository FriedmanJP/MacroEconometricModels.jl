# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-28 (#436): Brock-Dechert-Scheinkman-LeBaron (BDS) test for iid/independence.
#
# Oracle discipline (no invented numerics; see the EV-28 spec's oracle_strategy):
#
#   (1) MICRO HAND-COMPUTATION (PRIMARY). For a tiny T=6, m=2 series with NO tied
#       distances every quantity is an exact rational, computed by hand below and
#       pinned to 1e-12. This guards the two documented BDS failure points: the K
#       estimator and the σ²_m cross-term. See `@testset "Hand-computed T=6 m=2"`.
#
#   (2) INDEPENDENT RECOMPUTATION. A naive O(T³) brute-force reference
#       (`naive_bds`) recomputes Θ, C_1, C_m, K, σ_m and w_m from scratch under
#       the SAME convention (full-sample C_1, √T, two-sided N(0,1)); the vectorised
#       engine must match it to 1e-10 on several seeded series and multiple (m,ε).
#
#   (3) ANALYTIC MONTE-CARLO (needs no external tool). Seeded iid Normal(0,1)
#       ⇒ w_2 is ≈ N(0,1) (mean≈0, sd≈1) with a rejection rate near nominal; the
#       deterministic logistic map y_{t+1}=4y_t(1−y_t) ⇒ |w_m| huge, rejected on
#       every seed.
#
# PUBLISHED-NOT-LIVE NOTE: R `tseries::bds.test` (the conventional oracle) is NOT
# installable in this environment (its toolchain deps fail to build — same problem
# noted for the EV-11/EV-20/EV-21 tests), and neither scipy.stats nor statsmodels
# ships a BDS routine. There is therefore NO live cross-implementation. The exact
# rational hand-computation in (1) is the authoritative correctness oracle for the
# variance formula and the K estimator; the convention is Brock-Dechert-Scheinkman-
# LeBaron (1996), Econometric Reviews 15(3), eqs. for C_m, K and σ²_m (full-sample
# C_1). The closed form σ²_2 = 4(K − C_1²)² is a well-known BDS identity used as a
# secondary analytic check.

using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics, Distributions
import StatsAPI

const _M = MacroEconometricModels

# -----------------------------------------------------------------------------
# Naive O(T³) reference implementation (independent recomputation oracle).
# Deliberately unvectorised and literal to the BDS definitions.
# -----------------------------------------------------------------------------
function naive_bds(y::Vector{Float64}, m::Int, eps::Float64)
    T = length(y)
    Θ = [abs(y[i] - y[j]) < eps for i in 1:T, j in 1:T]   # strict <
    # C_1 (full sample): off-diagonal upper-triangle fraction.
    c1cnt = 0
    for i in 1:T, j in (i+1):T
        c1cnt += Θ[i, j]
    end
    C1 = 2 * c1cnt / (T * (T - 1))
    # C_m over the T_m = T-m+1 embedding vectors.
    Tm = T - m + 1
    cmcnt = 0
    for s in 1:Tm, t in (s+1):Tm
        allin = true
        for k in 0:(m-1)
            if !Θ[s+k, t+k]
                allin = false
                break
            end
        end
        cmcnt += allin
    end
    Cm = 2 * cmcnt / (Tm * (Tm - 1))
    # K estimator: k_i = # points within ε of i (j≠i); K = Σ k_i(k_i-1)/(T(T-1)(T-2)).
    K = 0.0
    for i in 1:T
        ki = 0
        for j in 1:T
            j == i && continue
            ki += Θ[i, j]
        end
        K += ki * (ki - 1)
    end
    K /= (T * (T - 1) * (T - 2))
    # σ²_m = 4[ K^m + 2 Σ_{j=1}^{m-1} K^{m-j} C^{2j} + (m-1)² C^{2m} - m² K C^{2m-2} ].
    c = C1
    s = K^m + (m - 1)^2 * c^(2m) - m^2 * K * c^(2m - 2)
    for j in 1:(m-1)
        s += 2 * K^(m - j) * c^(2j)
    end
    σ2 = 4 * s
    w = sqrt(T) * (Cm - C1^m) / sqrt(σ2)
    return (C1=C1, Cm=Cm, K=K, sigma=sqrt(σ2), w=w)
end

@testset "BDS independence test (EV-28)" begin

    # =========================================================================
    # (1) PRIMARY: exact hand-computation, T=6, m=2.
    #
    # y = [0.0, 0.1, 0.2, 1.0, 1.1, 5.0], ε = 0.5  (no tied distances).
    # Pairwise |yᵢ−yⱼ| < 0.5 holds only for {(1,2),(1,3),(2,3),(4,5)} ⇒ 4 pairs.
    #   C_1 = 2·4/(6·5) = 4/15.
    #   neighbour counts k = [2,2,2,1,1,0]  ⇒  K = Σk(k−1)/(6·5·4) = 6/120 = 1/20.
    #   m=2, T₂=5 embedding vectors; only (1,2) has Θ[1,2] & Θ[2,3] ⇒ C_2 = 2·1/(5·4) = 1/10.
    #   σ²_2 = 4(K − C_1²)² with K−C_1² = 1/20 − 16/225 = −19/900 ⇒ σ_2 = 19/450.
    #   w_2 = √6 (C_2 − C_1²)/σ_2 = √6 (13/450)/(19/450) = √6·13/19.
    # =========================================================================
    @testset "Hand-computed T=6 m=2 (rationals, 1e-12)" begin
        y = [0.0, 0.1, 0.2, 1.0, 1.1, 5.0]
        Θ = _M._bds_theta(y, 0.5)
        C1, K = _M._bds_c1_k(Θ, Float64)
        Cm = _M._bds_cm(Θ, [1, 2], Float64)
        @test C1 ≈ 4 / 15 atol = 1e-12
        @test K ≈ 1 / 20 atol = 1e-12
        @test Cm[1] ≈ 4 / 15 atol = 1e-12     # C_1 via the C_m path too
        @test Cm[2] ≈ 1 / 10 atol = 1e-12
        σ2 = _M._bds_sigma2(C1, K, 2)
        @test sqrt(σ2) ≈ 19 / 450 atol = 1e-12
        @test σ2 ≈ 4 * (K - C1^2)^2 atol = 1e-14      # analytic m=2 identity
        # Full statistic through the internal assembler.
        W, Cmat = _M._bds_stats(y, [2], [0.5])
        @test W[1, 1] ≈ sqrt(6) * 13 / 19 atol = 1e-12
        @test Cmat[1, 1] ≈ 1 / 10 atol = 1e-12
    end

    # =========================================================================
    # (2) Vectorised engine ≡ naive brute force, several seeds / m / ε.
    # =========================================================================
    @testset "Engine matches naive brute force" begin
        for seed in (11, 42, 907)
            rng = Random.MersenneTwister(seed)
            y = randn(rng, 60)
            sd = std(y)
            for m in (2, 3, 5), frac in (0.5, 1.0, 1.5)
                eps = frac * sd
                ref = naive_bds(y, m, eps)
                W, C = _M._bds_stats(y, [m], [eps])
                @test C[1, 1] ≈ ref.Cm atol = 1e-10
                @test W[1, 1] ≈ ref.w atol = 1e-9
            end
        end
    end

    # =========================================================================
    # (3a) Analytic MC: iid Normal ⇒ w_2 ≈ N(0,1), rejection near nominal.
    # =========================================================================
    @testset "iid Normal ⇒ standard-normal statistic" begin
        nseed = 200
        ws = Float64[]
        nreject = 0
        for s in 1:nseed
            rng = Random.MersenneTwister(3000 + s)
            y = randn(rng, 400)
            r = bds_test(y; m=2, eps_frac=1.0)          # T≥200 ⇒ no warning
            w = r.statistic[1, 1]
            push!(ws, w)
            (r.pvalue[1, 1] < 0.05) && (nreject += 1)
        end
        # Distributional moments (stable): under iid, w_2 ~ N(0,1).
        @test abs(mean(ws)) < 0.25
        @test 0.75 < std(ws) < 1.35
        # Rejection rate: near 5%, loose band guarding against gross mis-sizing.
        rate = nreject / nseed
        @test 0.005 < rate < 0.20
    end

    # =========================================================================
    # (3b) Deterministic logistic map y_{t+1}=4y_t(1−y_t): massively rejected.
    # =========================================================================
    @testset "Logistic map ⇒ overwhelming rejection" begin
        function logistic(T, y0)
            y = Vector{Float64}(undef, T)
            y[1] = y0
            for t in 2:T
                y[t] = 4 * y[t-1] * (1 - y[t-1])
            end
            return y
        end
        nreject = 0
        nseed = 20
        for s in 1:nseed
            rng = Random.MersenneTwister(700 + s)
            y0 = 0.05 + 0.9 * rand(rng)
            y = logistic(500, y0)
            r = bds_test(y; m=2:3, eps_frac=0.7)
            # Chaos ⇒ huge positive statistic, p ≈ 0 for both m.
            if all(w -> abs(w) > 5, r.statistic) && all(p -> p < 1e-4, r.pvalue)
                nreject += 1
            end
        end
        @test nreject == nseed
    end

    # =========================================================================
    # API surface: table shape, multi-ε, StatsAPI, show, refs.
    # =========================================================================
    @testset "API, table shape, StatsAPI, show, refs" begin
        rng = Random.MersenneTwister(1)
        y = randn(rng, 300)
        r = bds_test(y; m=2:4, eps_frac=[0.5, 1.0, 1.5])
        @test r isa BDSResult
        @test r.m == [2, 3, 4]
        @test size(r.statistic) == (3, 3)
        @test size(r.pvalue) == (3, 3)
        @test length(r.eps) == 3
        @test r.eps ≈ std(y) .* [0.5, 1.0, 1.5]
        @test all(isfinite, r.statistic)
        @test all(0 .<= r.pvalue .<= 1)
        @test !r.small_sample                    # T=300 ≥ 200
        # StatsAPI
        @test StatsAPI.nobs(r) == 300
        @test StatsAPI.pvalue(r) == minimum(r.pvalue)
        # show renders one row per (m,ε) without error
        io = IOBuffer()
        show(io, r)
        str = String(take!(io))
        @test occursin("BDS Independence Test", str)
        @test occursin("H₀", str)
        # refs
        rio = IOBuffer()
        refs(rio, r)
        rstr = String(take!(rio))
        @test occursin("Brock", rstr)
    end

    # =========================================================================
    # Small-sample warning + permutation bootstrap.
    # =========================================================================
    @testset "small-sample flag & bootstrap" begin
        rng = Random.MersenneTwister(5)
        y = randn(rng, 80)
        local r
        @test_logs (:warn,) match_mode=:any begin
            r = bds_test(y; m=2, eps_frac=1.0)
        end
        @test r.small_sample
        # Bootstrap under iid: p-value is a valid fraction; iid ⇒ not tiny.
        rng2 = Random.MersenneTwister(6)
        yb = randn(rng2, 150)
        rb = bds_test(yb; m=2, eps_frac=1.0, bootstrap=300, seed=99)
        @test rb.bootstrap == 300
        @test isfinite(rb.boot_pvalue[1, 1])
        @test 0 <= rb.boot_pvalue[1, 1] <= 1

        # Bootstrap detects dependence: logistic map ⇒ bootstrap p ≈ 0.
        function logistic(T, y0)
            y = Vector{Float64}(undef, T); y[1] = y0
            for t in 2:T; y[t] = 4 * y[t-1] * (1 - y[t-1]); end
            return y
        end
        rc = bds_test(logistic(300, 0.31); m=2, eps_frac=0.7, bootstrap=300, seed=7)
        @test rc.boot_pvalue[1, 1] < 0.05
    end

    # =========================================================================
    # Model dispatches: ARIMA residuals and GARCH standardized residuals.
    # =========================================================================
    @testset "ARIMA & GARCH residual dispatches" begin
        # White-noise data ⇒ AR(1) residuals ≈ iid ⇒ do not reject.
        rng = Random.MersenneTwister(20)
        y = randn(rng, 400)
        ar = estimate_ar(y, 1)
        r_arima = bds_test(ar; m=2, eps_frac=1.0)
        @test r_arima isa BDSResult
        @test r_arima.nobs == length(StatsAPI.residuals(ar))
        @test isfinite(r_arima.statistic[1, 1])

        # GARCH dispatch tests STANDARDIZED residuals (documented behaviour).
        rng2 = Random.MersenneTwister(21)
        n = 600
        e = randn(rng2, n)
        h = ones(n); ret = zeros(n)
        for t in 2:n
            h[t] = 0.05 + 0.1 * ret[t-1]^2 + 0.85 * h[t-1]
            ret[t] = sqrt(h[t]) * e[t]
        end
        g = estimate_garch(ret, 1, 1)
        r_g = bds_test(g; m=2, eps_frac=1.0)
        @test r_g isa BDSResult
        @test r_g.nobs == length(g.standardized_residuals)
        # Confirm it really used standardized residuals, not raw returns.
        r_std = bds_test(g.standardized_residuals; m=2, eps_frac=1.0)
        @test r_g.statistic ≈ r_std.statistic
    end

    # =========================================================================
    # Edge cases / argument validation.
    # =========================================================================
    @testset "edge cases & validation" begin
        rng = Random.MersenneTwister(9)
        y = randn(rng, 250)
        # Very large ε ⇒ Θ all-ones ⇒ degenerate variance ⇒ NaN statistic.
        r = bds_test(y; m=2, eps_frac=1e6)
        @test isnan(r.statistic[1, 1])
        @test isnan(r.pvalue[1, 1])
        # Zero-variance series errors.
        @test_throws ArgumentError bds_test(fill(3.0, 50))
        # Too-short series errors.
        @test_throws ArgumentError bds_test([1.0, 2.0]; m=5)
        # Negative bootstrap errors.
        @test_throws ArgumentError bds_test(y; bootstrap=-1)
        # AbstractVector{<:Real} conversion (Int input works).
        ri = bds_test(collect(1:300) .+ 0; m=2, eps_frac=1.0)
        @test ri isa BDSResult
    end
end
