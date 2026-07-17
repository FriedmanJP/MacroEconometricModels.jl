# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-27 (#435): Variance-ratio / random-walk tests — Lo-MacKinlay (1988),
# Chow-Denning (1993), Wright (2000) rank/sign, Kim (2006) wild bootstrap.
#
# Oracle discipline (no invented numerics; see the EV-27 spec oracle_strategy):
#   (1) ANALYTIC PROPERTY (primary, per the issue acceptance criteria): on a long
#       fixed-seed random walk (T≥5000) VR(q)≈1 within a loose band for every q and
#       the robust Chow-Denning does NOT reject; on a fixed-seed AR(1) level with
#       ρ=0.5 the robust Z*(q) and Chow-Denning REJECT.
#   (2) INDEPENDENT HAND-RECOMPUTATION inside the test (mirrors the scipy-recompute
#       pattern at test/teststat/test_unitroot.jl:681): σ²_a, the overlapping σ²_c
#       with the unbiased normalizer m=q(N−q+1)(1−q/N), VR=σ²_c/σ²_a, the
#       homoskedastic Z(q), and the robust θ̂(q)/Z*(q) are recomputed from scratch on
#       a small fixed-seed series and asserted equal to the module to ~1e-10.
#   (3) PUBLISHED-WITH-CITATION (⚠ R `vrtest` is NOT installed and does not build in
#       this environment — confirmed requireNamespace=FALSE — so NO live vrtest
#       numbers are generated here). Instead the population variance-ratio identity
#       for an AR(1) return process is pinned from Campbell, Lo & MacKinlay (1997),
#       "The Econometrics of Financial Markets", Ch. 2, eq. (2.4.19)-(2.4.20):
#           VR(q) = 1 + 2 Σ_{k=1}^{q-1} (1 − k/q) ρ(k),   ρ(k)=φ^k for AR(1),
#       and the sample VR(q) on a long AR(1)-return series is checked to converge to
#       it. The Chow-Denning SMM-complement p-value 1−(2Φ(CD)−1)^m is recomputed by
#       hand (published closed form, Chow & Denning 1993) and asserted to match the
#       module — and to DIFFER from the naive Bonferroni value.

using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics, Distributions

const MEM = MacroEconometricModels

# --- Independent, fully self-contained VR recomputation from a LEVEL series -----
# Deliberately written from the level vector (not reusing the module's return/cumsum
# path) so it is a genuine cross-check of _vr_lomac.
function _ref_vr(y::Vector{Float64}, q::Int)
    N = length(y) - 1                      # number of returns
    x = [y[t+1] - y[t] for t in 1:N]       # returns
    μ = sum(x) / N                          # = (y_N − y_0)/N
    σ2a = sum((x .- μ) .^ 2) / (N - 1)
    m = q * (N - q + 1) * (1 - q / N)
    sc = 0.0
    for t in q:N                            # return indices t=q..N
        # y_t − y_{t−q} in 1-based level indexing:
        d = y[t+1] - y[t-q+1] - q * μ
        sc += d^2
    end
    σ2c = sc / m
    vr = σ2c / σ2a
    z = sqrt(N) * (vr - 1) / sqrt(2 * (2q - 1) * (q - 1) / (3q))
    # robust θ̂(q)
    ss = sum((x .- μ) .^ 2)
    θ = 0.0
    for j in 1:(q-1)
        num = 0.0
        for t in (j+1):N
            num += (x[t] - μ)^2 * (x[t-j] - μ)^2
        end
        δj = N * num / ss^2
        θ += (2 * (q - j) / q)^2 * δj
    end
    zstar = sqrt(N) * (vr - 1) / sqrt(θ)
    return (; vr, z, zstar, m, σ2a, σ2c)
end

# Campbell-Lo-MacKinlay (1997) population VR for a return process with autocorr ρ(·).
_pop_vr(ρ::Function, q::Int) = 1 + 2 * sum((1 - k / q) * ρ(k) for k in 1:(q-1))

@testset "Variance-Ratio Tests (EV-27)" begin

    # ------------------------------------------------------------------------
    # (2) Independent hand-recomputation oracle
    # ------------------------------------------------------------------------
    @testset "hand-recomputation matches module (independent)" begin
        rng = MersenneTwister(4242)
        y = cumsum(0.3 .+ randn(rng, 250))          # drifting level series
        r = variance_ratio_test(y; q=[2, 3, 5, 10])
        @test r.nobs == length(y)
        for (i, q) in enumerate(r.q)
            ref = _ref_vr(y, q)
            @test r.vr[i] ≈ ref.vr atol = 1e-10
            @test r.z[i] ≈ ref.z atol = 1e-10
            @test r.z_star[i] ≈ ref.zstar atol = 1e-10
        end
        # the unbiased normalizer m is exactly q(N−q+1)(1−q/N)
        N = length(y) - 1
        @test _ref_vr(y, 4).m ≈ 4 * (N - 4 + 1) * (1 - 4 / N) atol = 1e-10
    end

    # ------------------------------------------------------------------------
    # (1) Analytic property: random walk ⇒ VR(q)≈1, do not reject
    # ------------------------------------------------------------------------
    @testset "random walk: VR(q)≈1, no rejection" begin
        rng = MersenneTwister(12345)
        y = cumsum(randn(rng, 6000))                # T ≥ 5000 random walk
        r = variance_ratio_test(y; q=[2, 4, 8, 16])
        for v in r.vr
            @test abs(v - 1) < 0.08                 # loose band
        end
        # robust and homoskedastic Z within a normal-ish range; joint test insignificant
        @test all(abs.(r.z_star) .< 3.5)
        @test r.cd_star_pvalue > 0.05
        @test r.cd_pvalue > 0.05
        # under iid homoskedastic returns Z* ≈ Z (δ̂_j → 1)
        @test maximum(abs.(r.z_star .- r.z)) < 0.5
    end

    # ------------------------------------------------------------------------
    # (1) Analytic property: AR(1) ρ=0.5 level ⇒ reject
    # ------------------------------------------------------------------------
    @testset "AR(1) ρ=0.5 level: robust test rejects" begin
        rng = MersenneTwister(999)
        T = 4000
        z = zeros(T)
        for t in 2:T
            z[t] = 0.5 * z[t-1] + randn(rng)
        end
        r = variance_ratio_test(z; q=[2, 4, 8, 16])
        @test any(abs.(r.z_star) .> 3)              # strong per-q rejection
        @test r.cd_star_pvalue < 0.01               # joint robust test rejects
        # a stationary (mean-reverting) level pushes VR below 1
        @test r.vr[end] < 0.9
    end

    # ------------------------------------------------------------------------
    # (3) PUBLISHED: population VR identity (Campbell-Lo-MacKinlay 1997)
    # ------------------------------------------------------------------------
    @testset "sample VR → population VR for AR(1) returns (CLM 1997)" begin
        # Build returns as AR(1) with φ=0.4, then levels = cumsum(returns).
        rng = MersenneTwister(2024)
        φ = 0.4
        T = 40_000
        x = zeros(T)
        for t in 2:T
            x[t] = φ * x[t-1] + randn(rng)
        end
        y = cumsum(x)                               # level series fed to the test
        r = variance_ratio_test(y; q=[2, 4, 8])
        ρ(k) = φ^k
        for (i, q) in enumerate(r.q)
            pop = _pop_vr(ρ, q)                      # CLM (1997) eq. 2.4.19
            @test isapprox(r.vr[i], pop; rtol=0.05)
        end
        # sanity: population VR for AR(1) φ=0.4, q=2 is 1+2·(1/2)·0.4 = 1.4
        @test _pop_vr(ρ, 2) ≈ 1.4 atol = 1e-12
    end

    # ------------------------------------------------------------------------
    # (3) PUBLISHED closed form: Chow-Denning SMM-complement p-value
    # ------------------------------------------------------------------------
    @testset "Chow-Denning SMM p-value = 1-(2Φ(CD)-1)^m (not Bonferroni)" begin
        rng = MersenneTwister(77)
        y = cumsum(randn(rng, 800))
        qv = [2, 4, 8, 16]
        r = variance_ratio_test(y; q=qv)
        m = length(qv)
        # homoskedastic branch
        cd = r.cd_stat
        p_smm = 1 - (2 * cdf(Normal(), cd) - 1)^m
        @test r.cd_pvalue ≈ p_smm atol = 1e-12
        # robust branch
        cds = r.cd_star_stat
        p_smm_star = 1 - (2 * cdf(Normal(), cds) - 1)^m
        @test r.cd_star_pvalue ≈ p_smm_star atol = 1e-12
        # SMM p-value differs from the naive Bonferroni bound m·2·(1−Φ(CD))
        p_bonf = m * 2 * ccdf(Normal(), cd)
        @test !isapprox(r.cd_pvalue, p_bonf; atol=1e-6)
        # CD is exactly max_q |Z(q)|
        @test r.cd_stat ≈ maximum(abs.(r.z)) atol = 1e-12
        @test r.cd_star_stat ≈ maximum(abs.(r.z_star)) atol = 1e-12
    end

    # ------------------------------------------------------------------------
    # Per-q asymptotic p-values are the two-sided normal tails
    # ------------------------------------------------------------------------
    @testset "per-q asymptotic p-values (two-sided normal)" begin
        rng = MersenneTwister(31)
        y = cumsum(randn(rng, 600))
        r = variance_ratio_test(y; q=[2, 4, 8])
        for i in eachindex(r.q)
            @test r.z_pvalue[i] ≈ 2 * ccdf(Normal(), abs(r.z[i])) atol = 1e-12
            @test r.z_star_pvalue[i] ≈ 2 * ccdf(Normal(), abs(r.z_star[i])) atol = 1e-12
        end
        # primary pvalue() is the robust Chow-Denning by default
        @test MEM.StatsAPI.pvalue(r) == r.cd_star_pvalue
        r_homo = variance_ratio_test(y; q=[2, 4, 8], robust=false)
        @test MEM.StatsAPI.pvalue(r_homo) == r_homo.cd_pvalue
        @test MEM.StatsAPI.nobs(r) == length(y)
        @test MEM.StatsAPI.dof(r) == length(r.q)
    end

    # ------------------------------------------------------------------------
    # Wright (2000) rank / sign statistics + simulated-null caching
    # ------------------------------------------------------------------------
    @testset "Wright rank/sign statistics and cached iid nulls" begin
        rng = MersenneTwister(555)
        # AR(1) returns give the rank/sign tests power
        T = 1200
        x = zeros(T)
        for t in 2:T
            x[t] = 0.35 * x[t-1] + randn(rng)
        end
        y = cumsum(x)
        r = variance_ratio_test(y; q=[2, 4, 8], method=:wright)
        @test r.wright
        @test length(r.R1) == 3 && length(r.R2) == 3 && length(r.S1) == 3
        @test all(0 .< r.R1_pvalue .<= 1)
        @test all(0 .< r.S1_pvalue .<= 1)
        # positive-autocorrelation series ⇒ at least one rank test flags significance
        @test minimum(r.R2_pvalue) < 0.10

        # Null-distribution cache: same (N,q,kind) → identical simulated draws
        N = length(y) - 1
        d1 = MEM._wright_null(N, 4, :r1)
        d2 = MEM._wright_null(N, 4, :r1)
        @test d1 === d2                              # served from the cache
        @test haskey(MEM._WRIGHT_NULL_CACHE, (N, 4, :r1))
        @test length(d1) == MEM._WRIGHT_NDRAWS
        # cache is bounded
        @test length(MEM._WRIGHT_NULL_CACHE) <= MEM._WRIGHT_CACHE_CAP

        # Re-running the whole test is reproducible (deterministic seeds)
        r_again = variance_ratio_test(y; q=[2, 4, 8], method=:wright)
        @test r_again.R1_pvalue == r.R1_pvalue
        @test r_again.S1_pvalue == r.S1_pvalue

        # default method omits Wright stats
        r_lm = variance_ratio_test(y; q=[2, 4, 8])
        @test !r_lm.wright
        @test isempty(r_lm.R1)
    end

    # Wright null is a genuine iid-null: on a random walk it should almost never
    # reject at 5% (well-calibrated simulated p-values).
    @testset "Wright null calibration on a random walk" begin
        rng = MersenneTwister(8)
        y = cumsum(randn(rng, 1500))
        r = variance_ratio_test(y; q=[2, 4], method=:wright)
        @test all(r.R1_pvalue .> 0.05)
        @test all(r.S1_pvalue .> 0.05)
    end

    # ------------------------------------------------------------------------
    # Kim (2006) wild bootstrap
    # ------------------------------------------------------------------------
    @testset "Kim wild bootstrap p-values (reproducible)" begin
        rng = MersenneTwister(61)
        y = cumsum(randn(rng, 500))
        r1 = variance_ratio_test(y; q=[2, 4, 8], bootstrap=299, seed=2024)
        r2 = variance_ratio_test(y; q=[2, 4, 8], bootstrap=299, seed=2024)
        @test r1.bootstrap == 299
        @test length(r1.z_star_boot_pvalue) == 3
        @test all(0 .< r1.z_star_boot_pvalue .<= 1)
        @test 0 < r1.cd_boot_pvalue <= 1
        # deterministic under a fixed seed
        @test r1.z_star_boot_pvalue == r2.z_star_boot_pvalue
        @test r1.cd_boot_pvalue == r2.cd_boot_pvalue
        # different seed ⇒ generally different p-values
        r3 = variance_ratio_test(y; q=[2, 4, 8], bootstrap=299, seed=7)
        @test r3.cd_boot_pvalue != r1.cd_boot_pvalue
        # normal weights also run
        rn = variance_ratio_test(y; q=[2, 4], bootstrap=199, boot_weights=:normal, seed=1)
        @test 0 < rn.cd_boot_pvalue <= 1

        # Power: AR(1) level ⇒ wild-bootstrap Chow-Denning rejects
        rng2 = MersenneTwister(3)
        z = zeros(2000); for t in 2:2000; z[t] = 0.5 * z[t-1] + randn(rng2); end
        rar = variance_ratio_test(z; q=[2, 4, 8, 16], bootstrap=299, seed=11)
        @test rar.cd_boot_pvalue < 0.05
    end

    # ------------------------------------------------------------------------
    # Argument validation
    # ------------------------------------------------------------------------
    @testset "argument validation" begin
        y = cumsum(randn(MersenneTwister(1), 100))
        @test_throws ArgumentError variance_ratio_test(y; q=[1, 2])          # q ≥ 2
        @test_throws ArgumentError variance_ratio_test(y; q=[2, 200])        # q < N
        @test_throws ArgumentError variance_ratio_test(y; method=:bogus)
        @test_throws ArgumentError variance_ratio_test(y; boot_weights=:x)
        @test_throws ArgumentError variance_ratio_test(y; bootstrap=-1)
        @test_throws ArgumentError variance_ratio_test([1.0, 2.0])           # too few obs
        # accepts integer / range inputs and non-Float vectors
        r = variance_ratio_test(collect(1:120) .+ 0.0; q=2:2:8)
        @test r.q == [2, 4, 6, 8]
    end

    # ------------------------------------------------------------------------
    # Display + refs render without error
    # ------------------------------------------------------------------------
    @testset "show / report / refs render" begin
        rng = MersenneTwister(2)
        y = cumsum(randn(rng, 400))
        r = variance_ratio_test(y; q=[2, 4, 8], method=:wright, bootstrap=99)
        s = sprint(show, r)
        @test occursin("Variance-Ratio", s)
        @test occursin("Chow-Denning", s)
        @test occursin("Wright", s)
        @test occursin("VR(q)", s)
        rb = sprint(io -> refs(io, r))
        @test occursin("MacKinlay", rb)
        @test occursin("Chow", rb) || occursin("Denning", rb)
        @test occursin("Wright", rb)
    end
end
