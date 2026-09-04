# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using Random
using LinearAlgebra
using Statistics
using Distributions

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _suppress_warnings = MacroEconometricModels._suppress_warnings

@testset "IRF Confidence Intervals" begin
    # Non-diagonal A + non-identity B0 via dgp_var (DGP-02 #791): every CI
    # method faces genuine dynamics and shock correlation.
    T_obs = 200
    n = 3
    p = 1
    rng = MersenneTwister(12345)  # DGP-02: explicit rng
    d = dgp_var(rng; T=T_obs)
    Y = d.Y

    model = estimate_var(Y, p)
    H = 10

    # =========================================================================
    # 1. Cholesky Identification
    # =========================================================================

    @testset "Cholesky - Bootstrap CI" begin
        _suppress_warnings() do
            irf_boot = irf(model, H; method=:cholesky, ci_type=:bootstrap, reps=(FAST ? 100 : 200), conf_level=0.90, seed=12346)

            @test irf_boot isa ImpulseResponse
            @test irf_boot.ci_type == :bootstrap
            @test size(irf_boot.values) == (H, n, n)
            @test size(irf_boot.ci_lower) == (H, n, n)
            @test size(irf_boot.ci_upper) == (H, n, n)
            # CI ordering: lower <= upper everywhere
            @test all(irf_boot.ci_lower .<= irf_boot.ci_upper)
            # Point estimate should generally lie within CIs
            frac_inside = mean(irf_boot.ci_lower .<= irf_boot.values .<= irf_boot.ci_upper)
            @test frac_inside > 0.8  # most point estimates should be inside
        end
    end

    @testset "Cholesky - Theoretical CI" begin
        _suppress_warnings() do
            irf_theo = irf(model, H; method=:cholesky, ci_type=:theoretical, reps=(FAST ? 200 : 500), conf_level=0.90, seed=12347)

            @test irf_theo isa ImpulseResponse
            @test irf_theo.ci_type == :theoretical
            @test all(irf_theo.ci_lower .<= irf_theo.ci_upper)

            # Symmetricity test: for theoretical (asymptotic normal) CIs,
            # the interval should be symmetric around the point estimate
            width_lower = irf_theo.values .- irf_theo.ci_lower  # distance below
            width_upper = irf_theo.ci_upper .- irf_theo.values  # distance above
            # Both should be non-negative
            @test all(width_lower .>= -1e-10)
            @test all(width_upper .>= -1e-10)
            # Symmetry: |lower_width - upper_width| / max_width should be small
            max_width = max.(width_lower, width_upper, 1e-15)
            asymmetry = abs.(width_lower .- width_upper) ./ max_width
            # Allow some asymmetry due to quantile estimation from finite draws
            @test mean(asymmetry) < 0.3  # average asymmetry should be modest
        end
    end

    @testset "Cholesky - Theoretical vs Bootstrap consistency" begin
        _suppress_warnings() do
            irf_boot = irf(model, H; method=:cholesky, ci_type=:bootstrap, reps=(FAST ? 200 : 500), conf_level=0.90, seed=12348)
            irf_theo = irf(model, H; method=:cholesky, ci_type=:theoretical, reps=(FAST ? 200 : 500), conf_level=0.90, seed=12348)

            # Point estimates should be identical (same model, same Q)
            @test irf_boot.values ≈ irf_theo.values

            # Both bands estimate the same asymptotic variance, so their mean
            # widths agree within 40% (was: order of magnitude — vacuous).
            boot_width = irf_boot.ci_upper .- irf_boot.ci_lower
            theo_width = irf_theo.ci_upper .- irf_theo.ci_lower
            ratio = mean(boot_width) / mean(theo_width)
            @test 0.7 < ratio < 1.4
        end
    end

    @testset "Cholesky - Confidence level affects width" begin
        _suppress_warnings() do
            # Common seed: same bootstrap draws, so the width ordering is exact.
            irf_90 = irf(model, H; method=:cholesky, ci_type=:bootstrap, reps=(FAST ? 100 : 200), conf_level=0.90, seed=12349)
            irf_68 = irf(model, H; method=:cholesky, ci_type=:bootstrap, reps=(FAST ? 100 : 200), conf_level=0.68, seed=12349)

            width_90 = mean(irf_90.ci_upper .- irf_90.ci_lower)
            width_68 = mean(irf_68.ci_upper .- irf_68.ci_lower)
            # 90% CI should be wider than 68% CI
            @test width_90 > width_68
        end
    end

    @testset "Cholesky - No CI" begin
        irf_none = irf(model, H; method=:cholesky, ci_type=:none)
        @test irf_none isa ImpulseResponse
        @test irf_none.ci_type == :none
        @test all(irf_none.ci_lower .== 0)
        @test all(irf_none.ci_upper .== 0)
    end

    # =========================================================================
    # 2. Long-Run Identification (Blanchard-Quah)
    # =========================================================================

    @testset "Long-run - Bootstrap CI" begin
        _suppress_warnings() do
            irf_lr = irf(model, H; method=:long_run, ci_type=:bootstrap, reps=(FAST ? 50 : 100), conf_level=0.90, seed=12350)

            @test irf_lr isa ImpulseResponse
            @test size(irf_lr.values) == (H, n, n)
            @test all(irf_lr.ci_lower .<= irf_lr.ci_upper)
        end
    end

    @testset "Long-run - Theoretical CI" begin
        _suppress_warnings() do
            # 1500 draws: quantile asymmetry is Monte Carlo noise scaling as
            # 1/sqrt(reps) (0.35 at 300 reps, 0.25 at 1500 on the reference DGP).
            irf_lr_theo = irf(model, H; method=:long_run, ci_type=:theoretical, reps=(FAST ? 500 : 1500), conf_level=0.90, seed=12351)

            @test irf_lr_theo isa ImpulseResponse
            @test all(irf_lr_theo.ci_lower .<= irf_lr_theo.ci_upper)

            # Symmetricity test for theoretical CIs
            width_lower = irf_lr_theo.values .- irf_lr_theo.ci_lower
            width_upper = irf_lr_theo.ci_upper .- irf_lr_theo.values
            @test all(width_lower .>= -1e-10)
            @test all(width_upper .>= -1e-10)
            max_width = max.(width_lower, width_upper, 1e-15)
            asymmetry = abs.(width_lower .- width_upper) ./ max_width
            # Symmetry is a property of material cells: for near-zero IRF
            # entries the width itself is Monte Carlo quantile noise, so the
            # relative asymmetry there measures noise, not the method.
            material = max_width .> 0.05
            @test any(material)
            @test mean(asymmetry[material]) < 0.3
        end
    end

    # =========================================================================
    # 3. Sign Restriction Identification
    # =========================================================================

    @testset "Sign restrictions - Bootstrap CI" begin
        _suppress_warnings() do
            # check_func takes a single arg: IRF array (H x n x n)
            # Sign restriction: shock 1 has positive impact on variable 1 at horizon 1
            check_fn = irf_vals -> irf_vals[1, 1, 1] > 0

            irf_sign = irf(model, H; method=:sign, ci_type=:bootstrap, reps=(FAST ? 20 : 50),
                           conf_level=0.90, check_func=check_fn, seed=12352,
                           set_inference=:bootstrap_x_rotations)

            @test irf_sign isa ImpulseResponse
            @test size(irf_sign.values) == (H, n, n)
            @test all(irf_sign.ci_lower .<= irf_sign.ci_upper)
        end
    end

    @testset "Sign restrictions - Theoretical CI" begin
        check_fn = irf_vals -> irf_vals[1, 1, 1] > 0
        @test_throws ArgumentError irf(model, H; method=:sign, ci_type=:theoretical,
                                       reps=(FAST ? 50 : 100), conf_level=0.90,
                                       check_func=check_fn)
    end

    # =========================================================================
    # 4. Non-Gaussian ICA Identification (FastICA)
    # =========================================================================

    @testset "FastICA - Bootstrap CI" begin
        _suppress_warnings() do
            # Non-Gaussian shocks: ICA has something to identify (DGP-02 #791).
            dng = dgp_nongaussian_var(MersenneTwister(12354); T=T_obs)
            mng = estimate_var(dng.Y, p)
            irf_ica = irf(mng, H; method=:fastica, ci_type=:bootstrap, reps=(FAST ? 20 : 50), conf_level=0.90, seed=12354)

            @test irf_ica isa ImpulseResponse
            @test size(irf_ica.values) == (H, n, n)
            @test all(irf_ica.ci_lower .<= irf_ica.ci_upper)
        end
    end

    @testset "FastICA - Theoretical CI rejected (SID-06)" begin
        @test_throws ArgumentError irf(model, H; method=:fastica, ci_type=:theoretical, reps=10)
    end

    # =========================================================================
    # 5. JADE Identification
    # =========================================================================

    @testset "JADE - Bootstrap CI" begin
        _suppress_warnings() do
            # Non-Gaussian shocks: JADE has something to identify (DGP-02 #791).
            dng = dgp_nongaussian_var(MersenneTwister(12356); T=T_obs)
            mng = estimate_var(dng.Y, p)
            irf_jade = irf(mng, H; method=:jade, ci_type=:bootstrap, reps=(FAST ? 20 : 50), conf_level=0.90, seed=12356)

            @test irf_jade isa ImpulseResponse
            @test size(irf_jade.values) == (H, n, n)
            @test all(irf_jade.ci_lower .<= irf_jade.ci_upper)
        end
    end

    # =========================================================================
    # 6. Cross-method point estimate comparison
    # =========================================================================

    @testset "All methods produce valid IRFs" begin
        _suppress_warnings() do
            for method in [:cholesky, :long_run, :fastica, :jade]
                ir = irf(model, H; method=method, ci_type=:none, seed=12357)
                @test ir isa ImpulseResponse
                @test all(isfinite, ir.values)
                @test size(ir.values) == (H, n, n)
                # Impact response (h=1) should be non-trivial for at least some entries
                @test any(abs.(ir.values[1, :, :]) .> 1e-10)
            end
        end
    end

    # =========================================================================
    # 7. Theoretical CI symmetry - comprehensive test across methods
    # =========================================================================

    @testset "Theoretical CI symmetry - $method" for method in [:cholesky, :long_run]
        _suppress_warnings() do
            ir = irf(model, H; method=method, ci_type=:theoretical, reps=(FAST ? 300 : 500), conf_level=0.90, seed=12358 + (method === :cholesky ? 1 : 2))
            @test ir isa ImpulseResponse

            # Symmetry check
            width_lower = ir.values .- ir.ci_lower
            width_upper = ir.ci_upper .- ir.values
            # Non-negative widths
            @test all(width_lower .>= -1e-10)
            @test all(width_upper .>= -1e-10)
            # Symmetry metric over material cells only (near-zero IRF entries
            # carry only Monte Carlo quantile noise in relative terms).
            max_width = max.(width_lower, width_upper, 1e-15)
            asymmetry = abs.(width_lower .- width_upper) ./ max_width
            material = max_width .> 0.05
            @test any(material)
            @test mean(asymmetry[material]) < 0.3
        end
    end

    # =========================================================================
    # 8. Stationarity Filtering (Issue #45)
    # =========================================================================

    @testset "stationary_only - Bootstrap" begin
        _suppress_warnings() do
            # Standard model should have most draws stationary
            irf_stat = irf(model, H; method=:cholesky, ci_type=:bootstrap, reps=(FAST ? 20 : 50),
                           conf_level=0.90, stationary_only=true, seed=12361)
            @test irf_stat isa ImpulseResponse
            @test size(irf_stat.values) == (H, n, n)
            @test all(irf_stat.ci_lower .<= irf_stat.ci_upper)

            # Draws should be stored and all should be from stationary models
            @test irf_stat._draws !== nothing
            @test size(irf_stat._draws, 1) > 0
        end
    end

    @testset "stationary_only - Theoretical" begin
        _suppress_warnings() do
            irf_stat_theo = irf(model, H; method=:cholesky, ci_type=:theoretical, reps=(FAST ? 20 : 50),
                                conf_level=0.90, stationary_only=true, seed=12362)
            @test irf_stat_theo isa ImpulseResponse
            @test all(irf_stat_theo.ci_lower .<= irf_stat_theo.ci_upper)
        end
    end

    @testset "stationary_only=false is default" begin
        _suppress_warnings() do
            irf_default = irf(model, H; method=:cholesky, ci_type=:bootstrap, reps=(FAST ? 20 : 50), conf_level=0.90, seed=12363)
            @test irf_default isa ImpulseResponse
            # Default behavior should work as before
            @test irf_default._draws !== nothing
            @test size(irf_default._draws, 1) == (FAST ? 20 : 50)
        end
    end

    # =========================================================================
    # 9. Bayesian IRF with posterior credible intervals
    # =========================================================================

    @testset "Bayesian IRF - Credible Intervals" begin
        _suppress_warnings() do
            post = estimate_bvar(Y, p; n_draws=(FAST ? 25 : 50), seed=12360)
            irf_bayes = irf(post, H)

            @test size(irf_bayes.quantiles, 4) == 3  # [16th, 50th, 84th percentile]
            # Ordering: 16th <= 50th <= 84th
            @test all(irf_bayes.quantiles[:, :, :, 1] .<= irf_bayes.quantiles[:, :, :, 2])
            @test all(irf_bayes.quantiles[:, :, :, 2] .<= irf_bayes.quantiles[:, :, :, 3])

            # Credible interval width should be positive
            width = irf_bayes.quantiles[:, :, :, 3] .- irf_bayes.quantiles[:, :, :, 1]
            @test all(width .>= 0)
        end
    end

    @testset "Band coverage MC (DGP-02 #791)" begin
        # Seeded MC: nominal-90% bands contain the TRUE IRF (not the point
        # estimate) — a band 3x too narrow fails. Bootstrap covers h in
        # {1, 4, 8}; theoretical conditions on Sigma-hat (draws only VAR
        # coefficients, so the impact row is degenerate) and covers {4, 8}.
        # Nominal 90% with 200 reps: MC se ≈ 0.02; [0.80, 0.97] is safe.
        nrep = FAST ? 25 : 200
        Hmc, Tmc = 8, 200
        hs_boot, hs_theo = [1, 4, 8], [4, 8]
        cover_boot = zeros(length(hs_boot))
        cover_theo = zeros(length(hs_theo))
        for r in 1:nrep
            rng = MersenneTwister(7600 + r)
            dmc = dgp_var(rng; T=Tmc)
            mmc = estimate_var(dmc.Y, 1)
            truth = var_irf(dmc.A, dmc.B0, Hmc - 1)
            b = irf(mmc, Hmc; method=:cholesky, ci_type=:bootstrap, reps=100,
                    conf_level=0.90, seed=7600 + r, stationary_only=true)
            t = irf(mmc, Hmc; method=:cholesky, ci_type=:theoretical, reps=300,
                    conf_level=0.90, seed=7600 + r)
            for (k, h) in enumerate(hs_boot)
                cover_boot[k] += mean(b.ci_lower[h, :, :] .<= truth[h, :, :] .<= b.ci_upper[h, :, :])
            end
            for (k, h) in enumerate(hs_theo)
                cover_theo[k] += mean(t.ci_lower[h, :, :] .<= truth[h, :, :] .<= t.ci_upper[h, :, :])
            end
        end
        cover_boot ./= nrep
        cover_theo ./= nrep
        for k in eachindex(hs_boot)
            @test 0.80 <= cover_boot[k] <= 0.97
        end
        for k in eachindex(hs_theo)
            @test 0.80 <= cover_theo[k] <= 0.97
        end
        # Theoretical impact row is degenerate by construction (Sigma-hat held
        # fixed): document it rather than covering it.
        t0 = irf(model, Hmc; method=:cholesky, ci_type=:theoretical, reps=300,
                 conf_level=0.90, seed=1)
        @test all(t0.ci_upper[1, :, :] .== t0.ci_lower[1, :, :])
    end
end

@testset "SID-06 theoretical CI vs residual-based ID" begin
    rng = MersenneTwister(735)  # DGP-02: explicit rng (throws-only data)
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

@testset "SID-04 column matching" begin
    # No draws in this testset (pure linear algebra); the old global seed is dropped.
    P = [1.0 0.2; 0.1 1.0]
    P_b = [-P[:, 2] P[:, 1]]          # permutation + sign flip
    perm, signs = MacroEconometricModels._match_columns(P, P_b)
    Q = P_b[:, perm] .* signs'
    @test all(diag(P' * Q) .> 0)
    @test MacroEconometricModels._procrustes_distance(Q, P) < 1e-12
end

@testset "SID-04 FastICA bootstrap bands after matching" begin
    _suppress_warnings() do
        rng = MersenneTwister(733)  # DGP-02: explicit rng
        n, p, Tobs, H = 2, 1, FAST ? 200 : 300, 6
        true_A = [0.5 0.1; 0.0 0.4]
        B0 = [1.0 0.3; 0.2 1.0]
        Y = zeros(Tobs, n)
        for t in 2:Tobs
            Y[t, :] = true_A * Y[t-1, :] + B0 * rand(rng, TDist(3), n)
        end
        m = estimate_var(Y, p)
        reps = FAST ? 20 : 100
        ir_ica = irf(m, H; method=:fastica, ci_type=:bootstrap, reps=reps, seed=733)
        ir_chol = irf(m, H; method=:cholesky, ci_type=:bootstrap, reps=reps, seed=733)
        w_ica = mean(ir_ica.ci_upper[1, :, :] .- ir_ica.ci_lower[1, :, :])
        w_chol = mean(ir_chol.ci_upper[1, :, :] .- ir_chol.ci_lower[1, :, :])
        @test w_ica < 2 * w_chol
        @test ir_ica.manifest !== nothing
        @test haskey(ir_ica.manifest.settings, "relabeled_fraction")
    end
end
