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

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "FEVD Tests with Theoretical Verification" begin
    # Non-diagonal A + non-identity B0 (DGP-02 #791): every share is interior,
    # so the FEVD must match var_fevd truth — an identity-returning estimator
    # fails here. fevd(model,H).proportions[:,:,h] accumulates horizons 0..h-1,
    # i.e. var_fevd row h.
    _tprint("Generating Data for FEVD Verification...")
    rng = MersenneTwister(7501)  # DGP-02: explicit rng
    d = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], B0=[1.0 0.0; 0.3 1.0], T=2000)
    Y, n, p = d.Y, 2, 1
    truth_fevd = var_fevd(d.A, d.B0, 5)

    model = fit(VARModel, Y, p)
    _tprint("Frequentist Estimation Done.")

    horizon = 5

    # 1. Frequentist FEVD
    _tprint("Testing Frequentist FEVD...")
    fevd_freq = fevd(model, horizon; method=:cholesky)

    # Note: FEVD struct uses lowercase 'proportions'
    @test size(fevd_freq.proportions) == (n, n, horizon)

    # T = 2000: OLS se ≈ 0.02; shares within 0.1 of truth at every horizon.
    for h in 1:horizon
        @test fevd_freq.proportions[:, :, h] ≈ truth_fevd[h, :, :] atol=0.1
    end

    # 2. Bayesian FEVD
    _tprint("Testing Bayesian FEVD...")
    post = estimate_bvar(Y, p; n_draws=(FAST ? 25 : 50))

    # Compute Bayesian FEVD
    fevd_bayes = fevd(post, horizon; method=:cholesky)

    # Check Mean Proportions
    # Structure: BayesianFEVD.point_estimate is (variable, shock, horizon) — unified with FEVD (#527)
    @test size(fevd_bayes.point_estimate) == (n, n, horizon)

    # Posterior median shares track truth (relaxed for MCMC variability).
    for h in 1:horizon
        @test fevd_bayes.point_estimate[:, :, h] ≈ truth_fevd[h, :, :] atol=0.25
    end

    _tprint("FEVD Verification Passed.")
end

@testset "FEVD identity corner (B0 = I)" begin
    # Kept as the closed-form corner: with B0 = I the true FEVD is the
    # identity at every horizon under every identification scheme.
    rng = MersenneTwister(7502)  # DGP-02: explicit rng
    d = dgp_var(rng; A=[0.3 0.0; 0.0 0.3], B0=Matrix{Float64}(I, 2, 2), T=2000)
    model = fit(VARModel, d.Y, 1)
    fe = fevd(model, 5; method=:cholesky)
    for h in 1:5
        @test fe.proportions[1, 1, h] ≈ 1.0 atol=0.1
        @test fe.proportions[2, 2, h] ≈ 1.0 atol=0.1
        @test fe.proportions[1, 2, h] ≈ 0.0 atol=0.1
        @test fe.proportions[2, 1, h] ≈ 0.0 atol=0.1
    end
end

@testset "FEVD Basic Functionality" begin
    rng = MersenneTwister(123)  # DGP-02: explicit rng

    # Simple VAR model
    T, n, p = 200, 3, 2
    Y = randn(rng, T, n)
    model = estimate_var(Y, p)

    horizon = 10

    # Test that FEVD can be computed
    fevd_result = fevd(model, horizon)

    @test fevd_result isa FEVD
    @test size(fevd_result.decomposition) == (n, n, horizon)
    @test size(fevd_result.proportions) == (n, n, horizon)

    # Proportions should sum to 1 for each variable at each horizon
    for h in 1:horizon
        for v in 1:n
            @test isapprox(sum(fevd_result.proportions[v, :, h]), 1.0, atol=1e-10)
        end
    end

    # Proportions should be non-negative
    @test all(fevd_result.proportions .>= -1e-10)

    # Decomposition should be non-negative
    @test all(fevd_result.decomposition .>= -1e-10)
end

@testset "fevd input validation (T062 C-18)" begin
    @test_throws ArgumentError MacroEconometricModels._validate_data([1.0 NaN; 0.0 2.0], "Sigma")
end

@testset "FEVD orthogonality guard (T061)" begin
    rng = MersenneTwister(61)  # DGP-02: explicit rng
    Y = zeros(200, 3)
    for t in 2:200
        Y[t, :] = 0.5 * Y[t-1, :] + randn(rng, 3)
    end
    model = estimate_var(Y, 1)
    L = Matrix(MacroEconometricModels.safe_cholesky(model.Sigma))

    # (1) orthonormal impact matrix (cholesky, Q=I) ⇒ guard true, no warning, props sum to 1
    @test MacroEconometricModels._check_fevd_orthogonality(L, model.Sigma; method=:cholesky)
    fe = @test_nowarn fevd(model, 8; method=:cholesky)
    for h in 1:8, i in 1:3
        @test sum(fe.proportions[i, :, h]) ≈ 1.0 atol = 1e-10
    end

    # (2) non-orthonormal impact matrix (P = L·diag(2,1,1)) ⇒ guard false + one warning
    P_bad = copy(L); P_bad[:, 1] .*= 2.0
    got = @test_logs (:warn,) match_mode = :any MacroEconometricModels._check_fevd_orthogonality(
        P_bad, model.Sigma; method=:garch)
    @test got == false
end

@testset "FEVD Methods" begin
    rng = MersenneTwister(456)  # DGP-02: explicit rng

    T, n, p = 150, 2, 1
    Y = randn(rng, T, n)
    model = estimate_var(Y, p)

    horizon = 5

    # Test Cholesky method
    fevd_chol = fevd(model, horizon; method=:cholesky)
    @test fevd_chol isa FEVD

    # Both methods should give valid proportions
    for h in 1:horizon
        for v in 1:n
            @test isapprox(sum(fevd_chol.proportions[v, :, h]), 1.0, atol=1e-10)
        end
    end
end

@testset "SID-05 set-aware sign FEVD" begin
    rng = MersenneTwister(734)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 150, 2), 1)
    chk(irf) = irf[1, 1, 1] > 0
    s = identify_sign(m, 5, chk; store_all=true, rng=MersenneTwister(1), max_draws=200)
    f = fevd(m, 5; method=:sign, check_func=chk, rng=MersenneTwister(1), max_draws=200)
    @test f.n_effective == s.n_accepted
    @test size(f.proportions) == (2, 2, 5)
    @test all(f.proportions .>= -1e-12)
    @test s.n_accepted > 1
    n, H = 2, 5
    acc = Array{Float64,4}(undef, s.n_accepted, n, n, H)
    for (i, Q) in enumerate(s.Q_draws)
        _, p = MacroEconometricModels._compute_fevd(compute_irf(m, Q, H), n, H)
        acc[i, :, :, :] = p
    end
    med = similar(f.proportions)
    for v in 1:n, sh in 1:n, h in 1:H
        med[v, sh, h] = quantile(@view(acc[:, v, sh, h]), 0.5)
    end
    @test f.proportions ≈ med
    _, p1 = MacroEconometricModels._compute_fevd(compute_irf(m, s.Q_draws[1], H), n, H)
    @test f.proportions ≉ p1
    irf_med = similar(s.irf_draws[1, :, :, :])
    for h in 1:H, i in 1:n, j in 1:n
        irf_med[h, i, j] = quantile(@view(s.irf_draws[:, h, i, j]), 0.5)
    end
    _, p_med_irf = MacroEconometricModels._compute_fevd(irf_med, n, H)
    @test f.proportions ≉ p_med_irf
end

@testset "SID-19 arias/uhlig FEVD" begin
    rng = MersenneTwister(748)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 80, 2), 1)
    r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
    fa = fevd(m, 5; method=:arias, restrictions=r, max_draws=20, rng=MersenneTwister(1))
    @test fa isa FEVD
    @test size(fa.proportions) == (2, 2, 5)
    fu = fevd(m, 5; method=:uhlig, restrictions=r, rng=MersenneTwister(2),
              n_starts=FAST ? 3 : 8, n_refine=1, max_iter_coarse=80, max_iter_fine=200)
    @test fu isa FEVD
    @test size(fu.proportions) == (2, 2, 5)
end
