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

Random.seed!(42)

@testset "IRF Tests with Theoretical Verification" begin
    _tprint("Generating Data for IRF Verification...")
    # 1. Setup Data with Known DGP
    # VAR(1): Y_t = A Y_{t-1} + u_t, u_t ~ N(0, I)
    # A = 0.5 * I
    T = 500
    n = 2
    p = 1
    true_A = [0.5 0.0; 0.0 0.5]
    true_c = [0.0; 0.0]
    Sigma_true = [1.0 0.0; 0.0 1.0] # Identity
    L_true = [1.0 0.0; 0.0 1.0]      # Cholesky of Identity is Identity

    Y = zeros(T, n)
    for t in 2:T
        u = randn(2)
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    model = estimate_var(Y, p)
    _tprint("Frequentist Estimation Done.")

    # 2. Frequentist IRF (Cholesky) vs Theoretical
    _tprint("Testing Frequentist IRF (Cholesky)...")
    irf_freq = irf(model, 6; method=:cholesky) # Horizon 6 (lags 0 to 5)

    # Theoretical IRF: Phi_h * P
    # P = L_true = I
    # Phi_h = A^h
    # Since A is diagonal 0.5:
    # IRF at h (lag h-1) = 0.5^(h-1) * I

    for h in 1:6
        lag = h - 1
        theoretical_impact = (0.5^lag) * I(2)
        estimated_impact = irf_freq.values[h, :, :]

        # Check diagonal elements
        @test isapprox(estimated_impact[1, 1], theoretical_impact[1, 1], atol=0.1)
        @test isapprox(estimated_impact[2, 2], theoretical_impact[2, 2], atol=0.1)

        # Check off-diagonal (should be close to 0)
        @test abs(estimated_impact[1, 2]) < 0.1
        @test abs(estimated_impact[2, 1]) < 0.1
    end

    # 3. Frequentist IRF (Sign) - Basic check logic remains
    _tprint("Testing Frequentist IRF (Sign)...")
    check_func(irf) = irf[1, 1, 1] > 0
    irf_sign_res = irf(model, 6; method=:sign, check_func=check_func)
    @test irf_sign_res.values[1, 1, 1] > 0

    # 4. Bayesian IRF
    _tprint("Testing Bayesian Estimation...")
    try
        post = estimate_bvar(Y, p; n_draws=50)
        _tprint("Bayesian Estimation Done.")

        _tprint("Testing Bayesian IRF...")
        irf_bayes = irf(post, 6; method=:cholesky)
        _tprint("Bayesian IRF Done.")

        @test irf_bayes isa BayesianImpulseResponse

        # Check Mean IRF against Theoretical
        for h in 1:6
            lag = h - 1
            theoretical_impact = (0.5^lag) * I(2)
            bayes_mean = irf_bayes.point_estimate[h, :, :]

            # Allow larger tolerance for smaller chain
            @test isapprox(bayes_mean[1, 1], theoretical_impact[1, 1], atol=0.3)
            @test isapprox(bayes_mean[2, 2], theoretical_impact[2, 2], atol=0.3)
        end

    catch e
        _tprint("ERROR CAUGHT:")
        showerror(stdout, e)
        _tprint()
        rethrow(e)
    end
end

# =============================================================================
# Cumulative IRF (Issue #15 + #31 fix: cumulate draws before quantile extraction)
# =============================================================================
@testset "Cumulative IRF" begin
    Random.seed!(42)
    Y = randn(200, 3)
    model = estimate_var(Y, 2)
    H = 20

    @testset "VAR cumulative IRF - no CI" begin
        irf_result = irf(model, H)
        cirf = cumulative_irf(irf_result)

        @test cirf isa ImpulseResponse
        @test size(cirf.values) == size(irf_result.values)
        @test cirf.horizon == irf_result.horizon

        # Verify cumulative = cumsum along horizon dimension
        expected = cumsum(irf_result.values, dims=1)
        @test cirf.values ≈ expected

        # No CI case: ci_lower/ci_upper are zeros, cumsum of zeros is zeros
        @test all(cirf.ci_lower .== 0)
        @test all(cirf.ci_upper .== 0)
    end

    @testset "VAR cumulative IRF - bootstrap CI (Issue #31)" begin
        Random.seed!(12345)
        irf_boot = irf(model, H; ci_type=:bootstrap, reps=200, conf_level=0.90)

        # Raw draws should be stored
        @test irf_boot._draws !== nothing
        @test size(irf_boot._draws, 1) == 200

        cirf = cumulative_irf(irf_boot)

        # Point estimate is still cumsum of original
        @test cirf.values ≈ cumsum(irf_boot.values, dims=1)

        # CI bands must be properly ordered
        @test all(cirf.ci_lower .<= cirf.ci_upper)

        # Key test: cumulative CIs should NOT equal naive cumsum of original CIs
        # (because quantiles are not additive)
        naive_cum_lower = cumsum(irf_boot.ci_lower, dims=1)
        naive_cum_upper = cumsum(irf_boot.ci_upper, dims=1)
        # At later horizons, the difference should be noticeable
        @test !(cirf.ci_lower ≈ naive_cum_lower)
        @test !(cirf.ci_upper ≈ naive_cum_upper)

        # The correct cumulative bands should be tighter than naive cumsum
        # (sub-additivity of quantiles in most cases)
        correct_width = mean(cirf.ci_upper .- cirf.ci_lower)
        naive_width = mean(naive_cum_upper .- naive_cum_lower)
        @test correct_width < naive_width * 1.5  # not drastically wider
    end

    @testset "Bayesian cumulative IRF (Issue #31)" begin
        post = estimate_bvar(Y, 2; n_draws=200)
        birf = irf(post, H)

        # Raw draws should be stored
        @test birf._draws !== nothing

        bcirf = cumulative_irf(birf)

        @test bcirf isa BayesianImpulseResponse
        @test size(bcirf.point_estimate) == size(birf.point_estimate)

        # Mean is additive, so cumsum of mean should equal mean of cumsum
        @test bcirf.point_estimate ≈ cumsum(birf.point_estimate, dims=1)

        # Quantiles should be properly ordered
        for qi in 1:(length(birf.quantile_levels)-1)
            @test all(bcirf.quantiles[:, :, :, qi] .<= bcirf.quantiles[:, :, :, qi+1])
        end

        # Key test: cumulative quantiles should NOT equal naive cumsum
        naive_cum_quantiles = cumsum(birf.quantiles, dims=1)
        @test !(bcirf.quantiles ≈ naive_cum_quantiles)
    end
end

# =============================================================================
# compute_irf exported (Issue #20)
# =============================================================================
@testset "Bootstrap is uncorrected residual bootstrap (T060)" begin
    Random.seed!(606)
    Tn, n, p = 200, 2, 1
    A = [0.5 0.1; 0.0 0.4]
    Y = zeros(Tn, n)
    for t in 2:Tn
        Y[t, :] = A * Y[t-1, :] + randn(n)
    end
    model = estimate_var(Y, p)
    H = 6
    reps = 400
    res = irf(model, H; method=:cholesky, ci_type=:bootstrap, stationary_only=true, reps=reps)

    @test res.ci_type == :bootstrap
    @test res._draws !== nothing
    @test all(isfinite, res._draws)

    # Uncorrected residual bootstrap ⇒ draws are centered at the estimated B̂, so the
    # bootstrap mean tracks the point IRF. A bias-corrected (Kilian 1998) bootstrap would
    # shift the draws away from the point estimate; this guard locks the current behavior.
    n_draws = size(res._draws, 1)
    boot_mean = dropdims(sum(res._draws; dims=1); dims=1) ./ n_draws
    scale = maximum(abs, res.values)
    @test isapprox(boot_mean, res.values; atol=3 * scale / sqrt(n_draws) + 0.03)
end

@testset "compute_irf buffer rewrite equivalence (T063)" begin
    # The preallocated-buffer/mul! rewrite must reproduce the analytic VAR(1) IRF
    # IRF[h] = A₁^(h-1)·P exactly (behavior-preserving).
    Random.seed!(63)
    A1 = [0.5 0.1; 0.0 0.4]
    Y = zeros(200, 2)
    for t in 2:200
        Y[t, :] = A1 * Y[t-1, :] + randn(2)
    end
    model = estimate_var(Y, 1)
    Q = Matrix{Float64}(I, 2, 2)
    IRF = MacroEconometricModels.compute_irf(model, Q, 5)
    P = Matrix(MacroEconometricModels.safe_cholesky(model.Sigma)) * Q
    A1hat = MacroEconometricModels.extract_ar_coefficients(model.B, 2, 1)[1]
    @test IRF[1, :, :] ≈ P
    @test IRF[2, :, :] ≈ A1hat * P atol = 1e-12
    @test IRF[3, :, :] ≈ A1hat * A1hat * P atol = 1e-12
end

@testset "Core numerics batch (T062: C-14/C-16/C-18)" begin
    # C-14: generate_Q never zeroes a rotation column (explicit ±1 map, not sign(0)=0)
    Random.seed!(614)
    Q4 = MacroEconometricModels.generate_Q(4)
    @test Q4' * Q4 ≈ I(4) atol = 1e-10
    @test rank(Q4) == 4
    Rdiag = [0.0, -2.0, 3.0]
    d = [r < 0 ? -1.0 : 1.0 for r in Rdiag]
    @test all(!iszero, d)
    @test d == [1.0, -1.0, 1.0]      # old sign.(Rdiag) would give [0,-1,1]

    # C-16: triangular solves reproduce the inverse-based results exactly, and the
    #       long-run rotation stays orthonormal (L⁻¹(I−ΣA)D · (...)' = I).
    Random.seed!(615)
    Y = zeros(200, 3)
    for t in 2:200
        Y[t, :] = 0.4 * Y[t-1, :] + randn(3)
    end
    model = estimate_var(Y, 1)
    Q = MacroEconometricModels.generate_Q(3)
    L = MacroEconometricModels.safe_cholesky(model.Sigma)
    eps_shocks = MacroEconometricModels.compute_structural_shocks(model, Q)
    @test eps_shocks ≈ (Q' * inv(Matrix(L)) * model.U')' atol = 1e-9
    Q_lr = MacroEconometricModels.identify_long_run(model)
    @test Q_lr * Q_lr' ≈ I(3) atol = 1e-8

    # C-18: the irf/fevd/hd input-validation guard rejects NaN/Inf in model matrices
    @test_throws ArgumentError MacroEconometricModels._validate_data([1.0 NaN; 2.0 3.0], "Sigma")
    @test_throws ArgumentError MacroEconometricModels._validate_data([Inf, 1.0], "B")
end

@testset "compute_irf exported" begin
    Random.seed!(42)
    Y = randn(200, 3)
    model = estimate_var(Y, 2)
    n = 3

    Q = Matrix{Float64}(I, n, n)
    result = compute_irf(model, Q, 10)
    @test size(result) == (10, n, n)
    @test !any(isnan, result)
end

# =============================================================================
# Sign Identified Set (Issue #21)
# =============================================================================
@testset "Sign Identified Set" begin
    Random.seed!(42)
    Y = randn(200, 3)
    model = estimate_var(Y, 2)
    n = 3
    H = 10

    # Accept-all check function for testing
    check_all(irf_result) = true

    result = identify_sign(model, H, check_all; max_draws=50, store_all=true)

    @test result isa SignIdentifiedSet
    @test result.n_accepted == 50
    @test result.n_total == 50
    @test result.acceptance_rate ≈ 1.0
    @test length(result.Q_draws) == 50
    @test size(result.irf_draws) == (50, H, n, n)

    # irf_bounds
    lower, upper = irf_bounds(result)
    @test size(lower) == (H, n, n)
    @test size(upper) == (H, n, n)
    @test all(lower .<= upper)

    # irf_median
    med = irf_median(result)
    @test size(med) == (H, n, n)
    @test !any(isnan, med)

    # show method
    io = IOBuffer()
    show(io, result)
    output = String(take!(io))
    @test occursin("Sign-Identified Set", output)
    @test occursin("50", output)
end

@testset "irf bootstrap CI is reproducible + thread-invariant (C-02/#243)" begin
    mkrng() = Random.MersenneTwister(7)
    Random.seed!(123)
    Y = zeros(120, 2)
    for t in 2:120
        Y[t, 1] = 0.5Y[t-1, 1] + 0.1Y[t-1, 2] + randn()
        Y[t, 2] = -0.2Y[t-1, 1] + 0.4Y[t-1, 2] + randn()
    end
    model = estimate_var(Y, 2)
    # Same seed -> bitwise-identical CI bands. The rejection loops now seed each iteration by
    # index and stage by index, so the kept subset is invariant to thread scheduling / thread
    # count (the old atomic accept-counter kept a scheduling-dependent subset).
    b1 = irf(model, 10; ci_type=:bootstrap, reps=50, rng=mkrng())
    b2 = irf(model, 10; ci_type=:bootstrap, reps=50, rng=mkrng())
    @test b1.ci_lower == b2.ci_lower
    @test b1.ci_upper == b2.ci_upper
    # stationary-only rejection path (the C-02 site) is deterministic too
    s1 = irf(model, 10; ci_type=:bootstrap, stationary_only=true, reps=30, rng=mkrng())
    s2 = irf(model, 10; ci_type=:bootstrap, stationary_only=true, reps=30, rng=mkrng())
    @test s1.ci_lower == s2.ci_lower && s1.ci_upper == s2.ci_upper
    # theoretical path
    t1 = irf(model, 10; ci_type=:theoretical, reps=50, rng=mkrng())
    t2 = irf(model, 10; ci_type=:theoretical, reps=50, rng=mkrng())
    @test t1.ci_lower == t2.ci_lower
    # different seed -> different draws
    b3 = irf(model, 10; ci_type=:bootstrap, reps=50, rng=Random.MersenneTwister(99))
    @test b1.ci_lower != b3.ci_lower
    # :sign identification now threads rng through compute_Q -> identify_sign -> generate_Q
    # (previously the rotation draw leaked to the global RNG, so sign CIs were non-reproducible)
    check = ir -> ir[1, 1, 1] > 0
    sg1 = irf(model, 8; ci_type=:bootstrap, method=:sign, check_func=check, reps=20, rng=mkrng(),
              set_inference=:bootstrap_x_rotations)
    sg2 = irf(model, 8; ci_type=:bootstrap, method=:sign, check_func=check, reps=20, rng=mkrng(),
              set_inference=:bootstrap_x_rotations)
    @test sg1.ci_lower == sg2.ci_lower
end

@testset "SID-06 theoretical CI vs residual-based ID" begin
    Random.seed!(735)
    m = estimate_var(randn(120, 2), 1)
    chk(irf) = irf[1, 1, 1] > 0
    @test_throws ArgumentError irf(m, 5; method=:fastica, ci_type=:theoretical)
    @test_throws ArgumentError irf(m, 5; method=:student_t, ci_type=:theoretical)
    @test_throws ArgumentError irf(m, 5; method=:markov_switching, ci_type=:theoretical)
    @test_throws ArgumentError irf(m, 5; method=:narrative, ci_type=:theoretical,
                                   check_func=chk, narrative_check=_ -> true)
    r = irf(m, 5; method=:cholesky, ci_type=:theoretical, reps=FAST ? 10 : 30)
    @test r.ci_type === :theoretical
end

@testset "SID-08 long-run on cointegrated systems" begin
    Random.seed!(737)
    Tlen, n = 200, 2
    trend = cumsum(randn(Tlen))
    Yc = [trend .+ 0.3 .* randn(Tlen)  trend .+ 0.3 .* randn(Tlen)]
    vecm = estimate_vecm(Yc, 2; rank=1)
    @test_throws IdentificationError identify_long_run(to_var(vecm))
    Ys = randn(200, 2)
    ms = estimate_var(Ys, 1)
    Q = identify_long_run(ms)
    @test norm(Q' * Q - I(2)) < 1e-8
end

@testset "SID-05 set-aware sign IRFs" begin
    Random.seed!(734)
    m = estimate_var(randn(150, 2), 1)
    chk(irf) = irf[1, 1, 1] > 0
    rng = MersenneTwister(1)
    s = identify_sign(m, 5, chk; store_all=true, rng=MersenneTwister(1), max_draws=200)
    r = irf(m, 5; method=:sign, check_func=chk, seed=1, max_draws=200)
    @test r.ci_type === :identified_set
    @test r.values ≈ irf_median(s)
    @test_throws ArgumentError irf(m, 5; method=:sign, check_func=chk, ci_type=:bootstrap)
    @test_throws ArgumentError irf(m, 5; method=:sign, check_func=chk, ci_type=:theoretical)
    r2 = irf(m, 5; method=:sign, check_func=chk, max_draws=500, seed=2)
    @test r2.manifest.settings["max_draws"] == 500
end

@testset "SID-05 identify_narrative store_all" begin
    Random.seed!(734)
    m = estimate_var(randn(150, 2), 1)
    chk(irf) = irf[1, 1, 1] > 0
    s = identify_narrative(m, 5, chk, _ -> true; store_all=true, max_draws=80, rng=MersenneTwister(3))
    @test s isa SignIdentifiedSet
    r = irf(m, 5; method=:narrative, check_func=chk, narrative_check=_ -> true, seed=3, max_draws=80)
    @test r.ci_type === :identified_set
    @test r.values ≈ irf_median(s)
    @test_throws ArgumentError irf(m, 5; method=:narrative, check_func=chk,
                                   narrative_check=_ -> true, ci_type=:bootstrap)
end

@testset "SID-19 one identification API" begin
    Random.seed!(748)
    m = estimate_var(randn(80, 2), 1)
    n = nvars(m)
    @test identify_cholesky(m) ≈ Matrix{Float64}(I, n, n)
    L = cholesky_factor(m)
    @test istriu(L')
    @test L * L' ≈ m.Sigma atol=1e-8

    Qkw = MacroEconometricModels.compute_Q(m, :cholesky; horizon=5)
    @test Qkw ≈ I(n)
    Qpos = @test_deprecated MacroEconometricModels.compute_Q(m, :cholesky, 5, nothing, nothing)
    @test Qpos ≈ I(n)

    @test MacroEconometricModels._needs_residuals(:cholesky) == false
    @test MacroEconometricModels._needs_residuals(:fastica) == true
    @test MacroEconometricModels._is_set_identified(:arias)
    @test !MacroEconometricModels._is_set_identified(:uhlig)
    @test !MacroEconometricModels._is_partial(:cholesky)
    @test_throws ArgumentError MacroEconometricModels._needs_residuals(:not_a_method)
    @test haskey(MacroEconometricModels.IDENTIFICATION_REGISTRY, :proxy)
    @test MacroEconometricModels._needs_residuals(:proxy)
    @test !MacroEconometricModels._is_set_identified(:proxy)
    @test MacroEconometricModels._is_partial(:proxy)
    @test haskey(MacroEconometricModels.IDENTIFICATION_REGISTRY, :ab)
    @test !MacroEconometricModels._needs_residuals(:ab)
    @test !MacroEconometricModels._is_set_identified(:ab)
    @test !MacroEconometricModels._is_partial(:ab)
    @test haskey(MacroEconometricModels.IDENTIFICATION_REGISTRY, :max_share)
    @test !MacroEconometricModels._needs_residuals(:max_share)
    @test !MacroEconometricModels._is_set_identified(:max_share)
    @test MacroEconometricModels._is_partial(:max_share)
    @test !MacroEconometricModels._should_match_columns(:max_share)
    @test haskey(MacroEconometricModels.IDENTIFICATION_REGISTRY, :svec)
    @test !MacroEconometricModels._needs_residuals(:svec)
    @test !MacroEconometricModels._is_set_identified(:svec)
    @test !MacroEconometricModels._is_partial(:svec)

    r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
    ra = irf(m, 5; method=:arias, restrictions=r, max_draws=30, seed=1)
    @test ra.ci_type === :identified_set
    @test ra.values[1, 1, 1] > 0
    @test size(ra.values) == (5, n, n)
    # n_rotations used to collide with n_rotations=max_draws (repeated-keyword MethodError).
    ra_nr = irf(m, 5; method=:arias, restrictions=r, n_rotations=20, max_draws=10, seed=1)
    @test ra_nr isa ImpulseResponse
    @test ra_nr.ci_type === :identified_set
    Qa = MacroEconometricModels.compute_Q(m, :arias; restrictions=r, n_rotations=20)
    @test Qa isa AbstractMatrix
    @test size(Qa) == (n, n)

    rng_u = MersenneTwister(748)
    uhlig_kw = (n_starts=FAST ? 3 : 8, n_refine=1, max_iter_coarse=80, max_iter_fine=200)
    u = identify_uhlig(m, r, 5; rng=copy(rng_u), uhlig_kw...)
    ru = irf(m, 5; method=:uhlig, restrictions=r, rng=copy(rng_u), uhlig_kw...)
    @test ru.values ≈ u.irf
    @test ru.values[1, 1, 1] > 0

    # Same-count quantile levels must be recomputed, not reused from stored 16/50/84.
    post = estimate_bvar(randn(80, 2), 1; n_draws=FAST ? 16 : 30, burnin=5)
    ar = identify_arias_bayesian(post, r, 4; n_rotations=FAST ? 20 : 40,
                                 rng=MersenneTwister(748))
    ir_def = irf(ar)
    ir_wide = irf(ar; quantiles=[0.05, 0.5, 0.95])
    @test ir_wide.quantile_levels ≈ [0.05, 0.5, 0.95]
    @test ir_wide.quantiles != ar.irf_quantiles
    @test ir_wide.quantiles != ir_def.quantiles
end
