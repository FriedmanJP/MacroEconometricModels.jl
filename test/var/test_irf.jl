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

@testset "IRF Tests with Theoretical Verification" begin
    # Non-diagonal A + non-identity B0 (DGP-02 #791): shock ordering matters,
    # so a transposed B0 or wrong rotation fails this testset.
    rng = MersenneTwister(7401)  # DGP-02: explicit rng
    _tprint("Generating Data for IRF Verification...")
    d = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], B0=[1.0 0.0; 0.3 1.0], T=2000)
    Y, p = d.Y, 1
    truth_irf = var_irf(d.A, d.B0, 5)

    model = estimate_var(Y, p)
    _tprint("Frequentist Estimation Done.")

    # 2. Frequentist IRF (Cholesky) vs Theoretical
    _tprint("Testing Frequentist IRF (Cholesky)...")
    irf_freq = irf(model, 6; method=:cholesky) # Horizon 6 (lags 0 to 5)

    # T = 2000: OLS se ≈ 0.02; atol 0.1 keeps 5x margin.
    for h in 1:6
        @test irf_freq.values[h, :, :] ≈ truth_irf[h, :, :] atol=0.1
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
            bayes_mean = irf_bayes.point_estimate[h, :, :]

            # Allow larger tolerance for smaller chain
            @test isapprox(bayes_mean[1, 1], truth_irf[h, 1, 1], atol=0.3)
            @test isapprox(bayes_mean[2, 2], truth_irf[h, 2, 2], atol=0.3)
        end

    catch e
        _tprint("ERROR CAUGHT:")
        showerror(stdout, e)
        _tprint()
        rethrow(e)
    end
end

@testset "IRF closed-form corner (A = 0.5I, Sigma = I)" begin
    # Kept as the analytic corner: IRF at lag h-1 is exactly 0.5^(h-1)·I.
    rng = MersenneTwister(7402)  # DGP-02: explicit rng
    d = dgp_var(rng; A=Matrix{Float64}(0.5 * I, 2, 2), B0=Matrix{Float64}(I, 2, 2),
                T=2000)
    model = estimate_var(d.Y, 1)
    irf_c = irf(model, 6; method=:cholesky)
    for h in 1:6
        @test irf_c.values[h, :, :] ≈ (0.5^(h - 1)) * I(2) atol=0.1
    end
end

# =============================================================================
# Cumulative IRF (Issue #15 + #31 fix: cumulate draws before quantile extraction)
# =============================================================================
@testset "Cumulative IRF" begin
    rng = MersenneTwister(42)  # DGP-02: explicit rng
    Y = randn(rng, 200, 3)
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
        rng = MersenneTwister(12345)  # DGP-02: explicit rng
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
    rng = MersenneTwister(606)  # DGP-02: explicit rng
    Tn, n, p = 200, 2, 1
    A = [0.5 0.1; 0.0 0.4]
    Y = zeros(Tn, n)
    for t in 2:Tn
        Y[t, :] = A * Y[t-1, :] + randn(rng, n)
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
    rng = MersenneTwister(63)  # DGP-02: explicit rng
    A1 = [0.5 0.1; 0.0 0.4]
    Y = zeros(200, 2)
    for t in 2:200
        Y[t, :] = A1 * Y[t-1, :] + randn(rng, 2)
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
    rng = MersenneTwister(614)  # DGP-02: explicit rng
    Q4 = MacroEconometricModels.generate_Q(4)
    @test Q4' * Q4 ≈ I(4) atol = 1e-10
    @test rank(Q4) == 4
    Rdiag = [0.0, -2.0, 3.0]
    d = [r < 0 ? -1.0 : 1.0 for r in Rdiag]
    @test all(!iszero, d)
    @test d == [1.0, -1.0, 1.0]      # old sign.(Rdiag) would give [0,-1,1]

    # C-16: triangular solves reproduce the inverse-based results exactly, and the
    #       long-run rotation stays orthonormal (L⁻¹(I−ΣA)D · (...)' = I).
    rng = MersenneTwister(615)  # DGP-02: explicit rng
    Y = zeros(200, 3)
    for t in 2:200
        Y[t, :] = 0.4 * Y[t-1, :] + randn(rng, 3)
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
    rng = MersenneTwister(42)  # DGP-02: explicit rng
    Y = randn(rng, 200, 3)
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
    # Structured DGP (DGP-02 #791): the restriction holds at the truth
    # (B0[1,1] = 1 > 0.5) but a random rotation violates it, so the
    # acceptance rate is strictly interior — the old check_all=true tautology
    # (n_accepted == 50 by construction) is gone.
    rng = MersenneTwister(42)  # DGP-02: explicit rng
    d = dgp_var(rng; T=500)
    Y = d.Y
    model = estimate_var(Y, 2)
    n = 3
    H = 10

    check_some(irf_result) = irf_result[1, 1, 1] > 0.5

    result = identify_sign(model, H, check_some; max_draws=50, store_all=true, rng=rng)

    @test result isa SignIdentifiedSet
    @test 0 < result.n_accepted < result.n_total
    @test 0.0 < result.acceptance_rate < 1.0
    @test length(result.Q_draws) == result.n_accepted
    @test size(result.irf_draws) == (result.n_accepted, H, n, n)

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
    @test occursin(string(result.n_accepted), output)
end

@testset "irf bootstrap CI is reproducible + thread-invariant (C-02/#243)" begin
    mkrng() = Random.MersenneTwister(7)
    rng = MersenneTwister(123)  # DGP-02: explicit rng
    Y = zeros(120, 2)
    for t in 2:120
        Y[t, 1] = 0.5Y[t-1, 1] + 0.1Y[t-1, 2] + randn(rng, )
        Y[t, 2] = -0.2Y[t-1, 1] + 0.4Y[t-1, 2] + randn(rng, )
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
    rng = MersenneTwister(735)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 120, 2), 1)
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
    rng = MersenneTwister(737)  # DGP-02: explicit rng
    Tlen, n = 200, 2
    trend = cumsum(randn(rng, Tlen))
    Yc = [trend .+ 0.3 .* randn(rng, Tlen)  trend .+ 0.3 .* randn(rng, Tlen)]
    vecm = estimate_vecm(Yc, 2; rank=1)
    @test_throws IdentificationError identify_long_run(to_var(vecm))
    Ys = randn(rng, 200, 2)
    ms = estimate_var(Ys, 1)
    Q = identify_long_run(ms)
    @test norm(Q' * Q - I(2)) < 1e-8
end

@testset "SID-05 set-aware sign IRFs" begin
    rng = MersenneTwister(734)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 150, 2), 1)
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
    rng = MersenneTwister(734)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 150, 2), 1)
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
    rng = MersenneTwister(748)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 80, 2), 1)
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
    post = estimate_bvar(randn(rng, 80, 2), 1; n_draws=FAST ? 16 : 30, burnin=5)
    ar = identify_arias_bayesian(post, r, 4; n_rotations=FAST ? 20 : 40,
                                 rng=MersenneTwister(748))
    ir_def = irf(ar)
    ir_wide = irf(ar; quantiles=[0.05, 0.5, 0.95])
    @test ir_wide.quantile_levels ≈ [0.05, 0.5, 0.95]
    @test ir_wide.quantiles != ar.irf_quantiles
    @test ir_wide.quantiles != ir_def.quantiles
end

# =============================================================================
# SID-17 (#746): Fry–Pagan / Inoue–Kilian set-ID summaries
# =============================================================================
@testset "SID-17 set-ID summaries" begin
    rng = MersenneTwister(746)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 120, 2), 1)
    chk(irf) = irf[1, 1, 1] > 0
    s = identify_sign(m, 6, chk; store_all=true, max_draws=80, rng=MersenneTwister(746))

    @testset "SignIdentifiedSet weights back-compat" begin
        @test hasfield(typeof(s), :weights)
        @test length(s.weights) == s.n_accepted
        @test s.weights ≈ fill(1 / s.n_accepted, s.n_accepted)
        @test s.ess ≈ s.n_accepted
        @test s.ess_fraction ≈ 1
        @test s.restrictions === nothing
        s7 = SignIdentifiedSet{Float64}(s.Q_draws, s.irf_draws, s.n_accepted, s.n_total,
                                        s.acceptance_rate, s.variables, s.shocks)
        @test s7.weights ≈ fill(1 / s.n_accepted, s.n_accepted)
        @test s7.ess ≈ Float64(s.n_accepted)
        @test s7.ess_fraction ≈ 1.0
        @test s7.restrictions === nothing
        med = irf_median(s)
        for h in axes(s.irf_draws, 2), i in axes(s.irf_draws, 3), j in axes(s.irf_draws, 4)
            @test med[h, i, j] ≈ quantile(view(s.irf_draws, :, h, i, j), 0.5)
        end
    end

    @testset "median_target is an admissible Q closest to irf_median" begin
        mt = median_target(s)
        @test mt.Q === s.Q_draws[mt.index]
        @test any(Q -> Q === mt.Q, s.Q_draws)
        @test mt.irf ≈ s.irf_draws[mt.index, :, :, :]
        med = irf_median(s)
        n_d = s.n_accepted
        H, nv, ns = size(med)
        σ = [std(view(s.irf_draws, :, h, i, j)) for h in 1:H, i in 1:nv, j in 1:ns]
        dist(d) = sum(((s.irf_draws[d, h, i, j] - med[h, i, j]) /
                       (σ[h, i, j] > 0 ? σ[h, i, j] : 1.0))^2
                      for h in 1:H, i in 1:nv, j in 1:ns)
        d_star = dist(mt.index)
        @test all(d -> dist(d) >= d_star - 1e-12, 1:n_d)
    end

    @testset "joint_band contains median-target and covers level jointly" begin
        nd, H, n = 5, 2, 2
        Qs = [Matrix{Float64}(I, n, n) for _ in 1:nd]
        draws = zeros(nd, H, n, n)
        for i in 1:nd
            draws[i, :, :, :] .= Float64(i)
        end
        tiny = SignIdentifiedSet{Float64}(Qs, draws, nd, nd, 1.0, ["y1", "y2"], ["e1", "e2"])
        mt = median_target(tiny)
        @test mt.index == 3
        lo, hi = joint_band(tiny; level=0.6, loss=:absolute)
        @test size(lo) == (H, n, n) && size(hi) == (H, n, n)
        @test all(lo .<= mt.irf .<= hi)
        inside(d) = all(lo[h, i, j] <= tiny.irf_draws[d, h, i, j] <= hi[h, i, j]
                        for h in 1:H, i in 1:n, j in 1:n)
        @test count(inside, 1:nd) / nd >= 0.6 - 1e-12
        @test_throws ArgumentError joint_band(tiny; loss=:quadratic)
    end

    @testset "joint_band skips zero-weight L1 outliers" begin
        nd, H, n = 4, 2, 2
        Qs = [Matrix{Float64}(I, n, n) for _ in 1:nd]
        draws = zeros(nd, H, n, n)
        draws[1, :, :, :] .= 1.0
        draws[2, :, :, :] .= 2.0
        draws[3, :, :, :] .= 3.0
        draws[4, :, :, :] .= 100.0   # L1 outlier
        w = [0.4, 0.3, 0.3, 0.0]
        tiny = SignIdentifiedSet{Float64}(Qs, draws, nd, nd, 1.0, ["y1", "y2"], ["e1", "e2"],
                                          w, 3.0, 0.75, nothing)
        lo, hi = joint_band(tiny; level=0.68)
        @test all(hi .< 50)
        @test all(lo .> 0.5)
        # Same mass without the zero-weight draw → identical envelope
        tiny3 = SignIdentifiedSet{Float64}(Qs[1:3], draws[1:3, :, :, :], 3, 3, 1.0,
                                           ["y1", "y2"], ["e1", "e2"],
                                           w[1:3], 3.0, 1.0, nothing)
        lo3, hi3 = joint_band(tiny3; level=0.68)
        @test lo ≈ lo3
        @test hi ≈ hi3
        # Zero-weight far-side draw visited before HPD mass is reached (tied L1
        # with the opposite-side member) must still not widen the envelope.
        Qs2 = [Matrix{Float64}(I, 1, 1) for _ in 1:3]
        d2 = zeros(3, 1, 1, 1)
        d2[1, 1, 1, 1] = 0.0
        d2[2, 1, 1, 1] = 2.0
        d2[3, 1, 1, 1] = 4.0
        s2 = SignIdentifiedSet{Float64}(Qs2, d2, 3, 3, 1.0, ["y"], ["e"],
                                        [0.0, 0.5, 0.5], 2.0, 2 / 3, nothing)
        lo2, hi2 = joint_band(s2; level=0.68)
        @test lo2[1, 1, 1] ≈ 2.0
        @test hi2[1, 1, 1] ≈ 4.0
    end

    @testset "modal_model and sup_t_band" begin
        mm = modal_model(s)
        @test mm.Q === s.Q_draws[mm.index]
        @test mm.irf ≈ s.irf_draws[mm.index, :, :, :]
        mm2 = modal_model(s; bandwidth=1.0)
        @test mm2.index isa Int
        lo, hi = sup_t_band(s; level=0.68)
        @test size(lo) == size(irf_median(s))
        @test all(lo .<= hi)
        mt = median_target(s)
        # sup-t is simultaneous around the pointwise median; median-target need not
        # sit inside, but the band is nonempty and finite.
        @test all(isfinite, lo) && all(isfinite, hi)
    end

    @testset "fevd(model, s, H) weighted median and adding-up" begin
        H = 6
        fv = fevd(m, s, H)
        @test fv isa BayesianFEVD
        n = nvars(m)
        n_d = s.n_accepted
        props = Array{Float64}(undef, n_d, n, n, H)
        for i in 1:n_d
            _, p = MacroEconometricModels._compute_fevd(s.irf_draws[i, :, :, :], n, H)
            props[i, :, :, :] = p
            for h in 1:H, v in 1:n
                @test sum(p[v, :, h]) ≈ 1 atol=1e-10
            end
        end
        for v in 1:n, sh in 1:n, h in 1:H
            @test fv.point_estimate[v, sh, h] ≈ quantile(view(props, :, v, sh, h), 0.5)
        end
    end

    @testset "historical_decomposition and structural_shocks on the set" begin
        hd = historical_decomposition(m, s)
        @test hd isa BayesianHistoricalDecomposition
        @test hd.n_effective == s.n_accepted
        @test size(hd.point_estimate, 1) == effective_nobs(m)
        sh = structural_shocks(m, s)
        @test size(sh.median) == (effective_nobs(m), nvars(m))
        @test size(sh.lower) == size(sh.median)
        @test size(sh.upper) == size(sh.median)
        @test all(sh.lower .<= sh.median .<= sh.upper)
    end

    @testset "Uhlig is a one-draw set" begin
        r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        u = identify_uhlig(m, r, 4; n_starts=3, n_refine=1,
                           max_iter_coarse=40, max_iter_fine=80, rng=MersenneTwister(746))
        mt = median_target(u)
        @test mt.Q === u.Q
        @test mt.irf ≈ u.irf
        @test mt.index == 1
        mm = modal_model(u)
        @test mm.Q === u.Q
        @test_throws ArgumentError joint_band(u)
        @test_throws ArgumentError sup_t_band(u)
        fv = fevd(m, u, 4)
        @test fv isa FEVD
        hd = historical_decomposition(m, u)
        @test hd isa HistoricalDecomposition
        sh = structural_shocks(m, u)
        point = MacroEconometricModels.compute_structural_shocks(m, u.Q)
        @test sh.median == sh.lower == sh.upper
        @test sh.median ≈ point
        @test size(sh.quantiles, 3) == 3
        @test sh.quantiles[:, :, 1] == sh.median
    end
end

@testset "SID-20 unit-effect IRF and structural_shocks(model, Q)" begin
    rng = MersenneTwister(749)  # DGP-02: explicit rng
    Tobs = FAST ? 250 : 400
    n = 3
    Y = zeros(Tobs, n)
    A = 0.4 * Matrix{Float64}(I, n, n)
    for t in 2:Tobs
        Y[t, :] = A * Y[t-1, :] + randn(rng, n)
    end
    m = estimate_var(Y, 1; varnames=["GDP", "CPI", "FEDFUNDS"])
    k = findfirst(==("FEDFUNDS"), m.varnames)
    @test k == 3

    ir_uv = irf(m, 6; method=:cholesky)
    @test ir_uv.values[1, 1, 1] != 0.25

    ir_ue = irf(m, 6; method=:cholesky, normalize=:unit_effect,
                shock_size=("FEDFUNDS" => 0.25))
    hits = [j for j in 1:n if isapprox(ir_ue.values[1, k, j], 0.25; atol=1e-10)]
    @test !isempty(hits)
    jstar = hits[1]
    @test ir_ue.values[1, k, jstar] ≈ 0.25 atol = 1e-10

    ir_idx = irf(m, 4; method=:cholesky, normalize=:unit_effect,
                 shock_size=(k => 0.25))
    @test ir_idx.values[1, k, jstar] ≈ 0.25 atol = 1e-10

    ir_fast = irf(m, 6; method=:fastica, normalize=:unit_effect,
                  shock_size=("FEDFUNDS" => 0.25), seed=749)
    @test any(j -> isapprox(ir_fast.values[1, k, j], 0.25; atol=1e-10), 1:n)

    ir_band = irf(m, 4; method=:cholesky, normalize=:unit_effect,
                  shock_size=("FEDFUNDS" => 0.25), ci_type=:bootstrap, reps=8,
                  seed=750)
    @test ir_band.values[1, k, jstar] ≈ 0.25 atol = 1e-10
    @test ir_band.ci_lower[1, k, jstar] ≈ 0.25 atol = 1e-8
    @test ir_band.ci_upper[1, k, jstar] ≈ 0.25 atol = 1e-8
    @test any(ir_band.ci_lower .!= ir_band.ci_upper)

    fv = MacroEconometricModels._suppress_warnings() do
        fevd(m, 4; method=:cholesky, normalize=:unit_effect,
             shock_size=("FEDFUNDS" => 0.25))
    end
    @test fv isa FEVD
    @test size(fv.proportions, 1) == n

    cum = cumulative_irf(ir_ue)
    @test cum.values[1, k, jstar] ≈ 0.25 atol = 1e-10

    Q = identify_cholesky(m)
    sh_q = structural_shocks(m, Q)
    @test sh_q ≈ MacroEconometricModels.compute_structural_shocks(m, Q)
    @test size(sh_q) == (size(m.U, 1), n)

    post = estimate_bvar(Y, 1; n_draws=FAST ? 12 : 24, burnin=4)
    bir = irf(post, 4; method=:cholesky, normalize=:unit_effect,
              shock_size=(1 => 0.25))
    @test bir isa BayesianImpulseResponse
    @test any(j -> isapprox(bir.point_estimate[1, 1, j], 0.25; atol=1e-8), 1:n)
    hd_b = historical_decomposition(post, 4; method=:cholesky,
                                    normalize=:unit_effect, shock_size=(1 => 0.25))
    @test hd_b isa BayesianHistoricalDecomposition
end
