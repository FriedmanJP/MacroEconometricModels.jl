# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# T243 (#342): wild cluster bootstrap (Cameron-Gelbach-Miller) for few-cluster inference.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics
using DataFrames

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _M = MacroEconometricModels

"""Few-cluster DGP with a within-cluster error component."""
function _wcb_sim(G::Int, n_per::Int; beta::Float64=0.0, seed::Int=1, rho::Float64=0.5)
    rng = Random.MersenneTwister(seed)
    n = G * n_per
    cl = repeat(1:G, inner=n_per)
    x = randn(rng, n)
    u = zeros(n)
    for g in 1:G
        a = randn(rng)
        idx = findall(==(g), cl)
        u[idx] .= sqrt(rho) * a .+ sqrt(1 - rho) .* randn(rng, length(idx))
    end
    y = 1.0 .+ beta .* x .+ u
    return y, hcat(ones(n), x), cl
end

"""
Independent, deliberately naive WCR bootstrap used only as a test oracle: refits by `\\`
each replication, recomputes the cluster sandwich from scratch, and enumerates every sign
vector. Shares no code path with the implementation under test.
"""
function _wcb_bruteforce(y, X, cl, j, r0)
    n, k = size(X)
    ucl = unique(cl)
    G = length(ucl)
    idxs = [findall(==(g), cl) for g in ucl]

    function cluster_se(bvec, resid)
        B = zeros(k, k)
        for idx in idxs
            sc = X[idx, :]' * resid[idx]
            B .+= sc * sc'
        end
        B .*= (G / (G - 1)) * ((n - 1) / (n - k))
        Vm = inv(X' * X) * B * inv(X' * X)
        sqrt(Vm[j, j])
    end

    beta = X \ y
    resid = y - X * beta
    t_obs = (beta[j] - r0) / cluster_se(beta, resid)

    # Restricted LS: minimize ‖y - Xb‖² subject to b[j] = r0
    keep = setdiff(1:k, j)
    yr = y .- r0 .* X[:, j]
    br = zeros(k)
    br[j] = r0
    br[keep] = X[:, keep] \ yr
    fit_r = X * br
    resid_r = y - fit_r

    ts = Float64[]
    for b in 0:(2^G - 1)
        w = [((b >> (g - 1)) & 1) == 1 ? -1.0 : 1.0 for g in 1:G]
        ystar = copy(fit_r)
        for (g, idx) in enumerate(idxs)
            ystar[idx] .+= w[g] .* resid_r[idx]
        end
        bs = X \ ystar
        rs = ystar - X * bs
        push!(ts, (bs[j] - r0) / cluster_se(bs, rs))
    end
    # Same tie rule as the implementation, written independently: the all-+1 sign
    # vector reproduces the sample so ±t_obs are exactly attained and must be counted.
    tol = max(abs(t_obs), 1.0) * 1e-9
    p = (1 + count(t -> abs(t) >= abs(t_obs) - tol, ts)) / (length(ts) + 1)
    return t_obs, p, ts
end

@testset "Wild Cluster Bootstrap" begin

# ─────────────────────────────────────────────────────────────────────────────
# Weights
# ─────────────────────────────────────────────────────────────────────────────

@testset "weight matrix: exact enumeration and Webb support" begin
    rng = Random.MersenneTwister(1)

    # 2^G ≤ n_boot with Rademacher weights ⇒ every sign vector exactly once
    V, enumerated = _M._wcb_weight_matrix(4, 999, :rademacher, rng, Float64)
    @test enumerated
    @test size(V) == (4, 16)
    @test all(v -> v == 1.0 || v == -1.0, V)
    cols = Set(Tuple(V[:, b]) for b in 1:16)
    @test length(cols) == 16                     # all distinct
    @test Tuple(ones(4)) in cols                 # includes the all-positive vector

    # Too many clusters to enumerate ⇒ random draws of the requested count
    V2, enum2 = _M._wcb_weight_matrix(30, 99, :rademacher, rng, Float64)
    @test !enum2
    @test size(V2) == (30, 99)

    # Webb weights are never enumerated and take the 6-point support
    V3, enum3 = _M._wcb_weight_matrix(4, 999, :webb, rng, Float64)
    @test !enum3
    @test size(V3) == (4, 999)
    support = sort(unique(round.(vec(V3), digits=10)))
    expected = sort(round.([-sqrt(1.5), -1.0, -sqrt(0.5), sqrt(0.5), 1.0, sqrt(1.5)], digits=10))
    @test support == expected
    @test abs(mean(V3)) < 0.1                    # mean zero
    @test abs(var(V3; corrected=false) - 1.0) < 0.1   # unit variance
end

# ─────────────────────────────────────────────────────────────────────────────
# Correctness against an independent brute-force implementation
# ─────────────────────────────────────────────────────────────────────────────

@testset "matches an independent brute-force WCR implementation" begin
    y, X, cl = _wcb_sim(5, 25; beta=0.0, seed=77)
    m = estimate_reg(y, X; cov_type=:cluster, clusters=cl, varnames=["const", "x"])

    for r0 in (0.0, 0.1, -0.25)
        b = wild_cluster_bootstrap(m, "x", r0; clusters=cl, ci=false,
                                   rng=Random.MersenneTwister(1))
        t_ref, p_ref, ts_ref = _wcb_bruteforce(y, X, cl, 2, r0)
        @test b.enumerated
        @test b.n_boot == 2^5
        @test b.t_stat ≈ t_ref atol = 1e-10
        @test b.p_value ≈ p_ref atol = 1e-12
        @test sort(b.t_boot) ≈ sort(ts_ref) atol = 1e-9
    end

    # The observed t also matches the model's own cluster-robust t at r0 = 0
    b0 = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, ci=false,
                                rng=Random.MersenneTwister(1))
    @test b0.t_stat ≈ coef(m)[2] / stderror(m)[2] atol = 1e-8
    @test b0.estimate ≈ coef(m)[2] atol = 1e-10
end

@testset "enumerated and simulated bootstraps agree" begin
    y, X, cl = _wcb_sim(6, 30; beta=0.0, seed=11)
    m = estimate_reg(y, X; cov_type=:cluster, clusters=cl, varnames=["const", "x"])

    exact = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, ci=false,
                                   rng=Random.MersenneTwister(1))
    @test exact.enumerated && exact.n_boot == 64

    # Drawing many Rademacher vectors samples the same 64-point space, so the simulated
    # p-value converges on the exact one. Enumeration must be switched off explicitly:
    # the default enumerates whenever 2^G ≤ n_boot, so a LARGER n_boot would be exact.
    sim = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, n_boot=20000, ci=false,
                                 enumerate=false, rng=Random.MersenneTwister(2))
    @test !sim.enumerated
    @test sim.n_boot == 20000
    @test sim.p_value ≈ exact.p_value atol = 0.02

    # enumerate=true is honored, and rejected when it cannot be satisfied
    @test wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, n_boot=64, ci=false,
                                 enumerate=true, rng=Random.MersenneTwister(2)).enumerated
    @test_throws ArgumentError wild_cluster_bootstrap(m, "x", 0.0; clusters=cl,
                                                      n_boot=10, enumerate=true)
    @test_throws ArgumentError wild_cluster_bootstrap(m, "x", 0.0; clusters=cl,
                                                      weights=:webb, enumerate=true)
end

# ─────────────────────────────────────────────────────────────────────────────
# The point of the procedure: size with few clusters
# ─────────────────────────────────────────────────────────────────────────────

@testset "better size than the cluster-robust t with G=6" begin
    nrep = FAST ? 60 : 200
    rej_boot = 0
    rej_asy = 0
    for s in 1:nrep
        y, X, cl = _wcb_sim(6, 30; beta=0.0, seed=1000 + s)
        m = estimate_reg(y, X; cov_type=:cluster, clusters=cl, varnames=["const", "x"])
        b = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, ci=false,
                                   rng=Random.MersenneTwister(s))
        rej_boot += b.p_value < 0.05
        rej_asy += b.p_value_asymptotic < 0.05
    end
    size_boot = rej_boot / nrep
    size_asy = rej_asy / nrep

    # The cluster-robust normal over-rejects badly at G=6 (Cameron-Gelbach-Miller);
    # the wild cluster bootstrap is close to nominal.
    @test size_asy > 0.10
    @test size_boot < 0.10
    @test size_boot < size_asy
end

# ─────────────────────────────────────────────────────────────────────────────
# Confidence interval by test inversion
# ─────────────────────────────────────────────────────────────────────────────

@testset "CI inverts the test" begin
    y, X, cl = _wcb_sim(6, 30; beta=0.0, seed=11)
    m = estimate_reg(y, X; cov_type=:cluster, clusters=cl, varnames=["const", "x"])
    b = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, rng=Random.MersenneTwister(1))

    @test isfinite(b.ci_lower) && isfinite(b.ci_upper)
    @test b.ci_lower < b.estimate < b.ci_upper

    pat(v) = wild_cluster_bootstrap(m, "x", v; clusters=cl, ci=false,
                                    rng=Random.MersenneTwister(1)).p_value

    # Inside the interval the null is not rejected; just outside it is. With G=6 the
    # enumerated p-value is a step function on multiples of 1/65, so the endpoint p sits
    # at the smallest attainable step at or above 0.05 rather than exactly 0.05.
    @test pat(b.ci_lower) >= 0.05
    @test pat(b.ci_upper) >= 0.05
    width = b.ci_upper - b.ci_lower
    @test pat(b.ci_lower - 0.05 * width) < 0.05
    @test pat(b.ci_upper + 0.05 * width) < 0.05

    # A wider level gives a wider interval
    b90 = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, level=0.90,
                                 rng=Random.MersenneTwister(1))
    @test b90.ci_upper - b90.ci_lower <= width + 1e-8
    @test b90.level == 0.90

    # ci=false leaves the bounds unset
    bn = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, ci=false,
                                rng=Random.MersenneTwister(1))
    @test isnan(bn.ci_lower) && isnan(bn.ci_upper)
end

# ─────────────────────────────────────────────────────────────────────────────
# Variants
# ─────────────────────────────────────────────────────────────────────────────

@testset "WCR is the default; WCU available" begin
    y, X, cl = _wcb_sim(6, 30; beta=0.0, seed=11)
    m = estimate_reg(y, X; cov_type=:cluster, clusters=cl, varnames=["const", "x"])

    wcr = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, ci=false,
                                 rng=Random.MersenneTwister(1))
    wcu = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, imposenull=false, ci=false,
                                 rng=Random.MersenneTwister(1))
    @test wcr.imposenull
    @test !wcu.imposenull
    @test wcr.t_stat ≈ wcu.t_stat atol = 1e-10       # same observed statistic
    # The bootstrap DGPs differ, so the bootstrap t DISTRIBUTIONS must differ. The
    # p-values need not: with G=6 the 2^G sign vectors are enumerated and the
    # p-value lives on a 64-point grid, so the two procedures can land on the same
    # grid point by coincidence — they did on Julia 1.10's stream.
    @test wcr.t_boot != wcu.t_boot

    # Under WCU the bootstrap DGP is the unrestricted fit, so the bootstrap t
    # distribution is centered on zero by construction of the recentering.
    @test abs(median(wcu.t_boot)) < 1.0
end

@testset "Webb weights" begin
    y, X, cl = _wcb_sim(4, 30; beta=0.0, seed=5)
    m = estimate_reg(y, X; cov_type=:cluster, clusters=cl, varnames=["const", "x"])
    b = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, weights=:webb, n_boot=499,
                               ci=false, rng=Random.MersenneTwister(3))
    @test b.weighttype === :webb
    @test !b.enumerated
    @test b.n_boot == 499
    @test 0 < b.p_value <= 1
end

@testset "equal-tail p-value" begin
    y, X, cl = _wcb_sim(6, 30; beta=0.8, seed=31)
    m = estimate_reg(y, X; cov_type=:cluster, clusters=cl, varnames=["const", "x"])
    b = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, ci=false,
                               rng=Random.MersenneTwister(1))
    @test 0 < b.p_value_equaltail <= 1
    # A strong one-sided effect is detected by both p-values
    @test b.p_value < 0.10
    @test b.p_value_equaltail < 0.10
end

# ─────────────────────────────────────────────────────────────────────────────
# Panel dispatch
# ─────────────────────────────────────────────────────────────────────────────

@testset "panel FE dispatch matches the within-transformed design" begin
    y, X, cl = _wcb_sim(6, 30; beta=0.3, seed=41)
    df = DataFrame(id=cl, t=repeat(1:30, outer=6), y=y, x=X[:, 2])
    pd = xtset(df, :id, :t)
    pm = estimate_xtreg(pd, :y, [:x])

    bp = wild_cluster_bootstrap(pm, "x", 0.0; ci=false, rng=Random.MersenneTwister(1))
    @test bp.n_clusters == 6
    @test bp.enumerated
    @test bp.estimate ≈ coef(pm)[1] atol = 1e-8

    # Same answer as bootstrapping the manually within-demeaned design directly
    ug = sort(unique(cl))
    y_dm, _ = _M._within_demean(y, cl, ug)
    X_dm, _ = _M._within_demean_matrix(reshape(X[:, 2], :, 1), cl, ug)
    manual = _M._wild_cluster_bootstrap(y_dm, X_dm, ["x"], cl, "x", 0.0;
                                        ci=false, rng=Random.MersenneTwister(1))
    @test bp.t_stat ≈ manual.t_stat atol = 1e-10
    @test bp.p_value ≈ manual.p_value atol = 1e-12

    # Non-FE panel estimators are rejected with a message naming the supported case
    pm_re = estimate_xtreg(pd, :y, [:x]; model=:re)
    err = try
        wild_cluster_bootstrap(pm_re, "x", 0.0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(":fe", sprint(showerror, err))
end

# ─────────────────────────────────────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────────────────────────────────────

@testset "argument handling, validation and display" begin
    y, X, cl = _wcb_sim(5, 25; beta=0.0, seed=77)
    m = estimate_reg(y, X; cov_type=:cluster, clusters=cl, varnames=["const", "x"])

    # Coefficient selectable by name, Symbol, or index
    by_name = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, ci=false,
                                     rng=Random.MersenneTwister(1))
    by_sym = wild_cluster_bootstrap(m, :x, 0.0; clusters=cl, ci=false,
                                    rng=Random.MersenneTwister(1))
    by_idx = wild_cluster_bootstrap(m, 2, 0.0; clusters=cl, ci=false,
                                    rng=Random.MersenneTwister(1))
    @test by_name.p_value == by_sym.p_value == by_idx.p_value
    @test by_idx.coefindex == 2 && by_idx.coefname == "x"

    # null_value defaults to zero
    @test wild_cluster_bootstrap(m, "x"; clusters=cl, ci=false,
                                 rng=Random.MersenneTwister(1)).null_value == 0.0

    # Reproducible under a seeded rng (simulated path)
    a1 = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, n_boot=200, ci=false,
                                rng=Random.MersenneTwister(9))
    a2 = wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, n_boot=200, ci=false,
                                rng=Random.MersenneTwister(9))
    @test a1.p_value == a2.p_value

    @test_throws ArgumentError wild_cluster_bootstrap(m, "x", 0.0)          # clusters required
    @test_throws ArgumentError wild_cluster_bootstrap(m, "zz", 0.0; clusters=cl)
    @test_throws ArgumentError wild_cluster_bootstrap(m, 9, 0.0; clusters=cl)
    @test_throws ArgumentError wild_cluster_bootstrap(m, "x", 0.0; clusters=cl[1:10])
    @test_throws ArgumentError wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, weights=:bogus)
    @test_throws ArgumentError wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, n_boot=0)
    @test_throws ArgumentError wild_cluster_bootstrap(m, "x", 0.0; clusters=cl, level=1.5)
    @test_throws ArgumentError wild_cluster_bootstrap(m, "x", 0.0; clusters=fill(1, length(cl)))

    out = sprint(show, by_name)
    @test occursin("Wild Cluster Bootstrap", out)
    @test occursin("WCR", out)
    @test occursin("bootstrap, symmetric", out)
    @test report(by_name) === nothing
end

end  # @testset "Wild Cluster Bootstrap"
