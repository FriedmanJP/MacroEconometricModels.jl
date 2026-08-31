# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Distributions
using StatsAPI

const _suppress_warnings = MacroEconometricModels._suppress_warnings
# Allow standalone include (runtests.jl also defines FAST at the suite root).
if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end
if !@isdefined(simulate_two_regime)
    include(joinpath(@__DIR__, "..", "var", "id_dgps.jl"))
end

@testset "Non-Gaussian SVAR Identification" begin
    Random.seed!(54321)

    # Generate VAR data
    n_obs = 300
    Y = randn(n_obs, 3)
    model = estimate_var(Y, 2)
    n = 3

    @testset "ICA-based Identification" begin
        _suppress_warnings() do
            @testset "FastICA deflation" begin
                result = identify_fastica(model; approach=:deflation)
                @test result isa ICASVARResult{Float64}
                @test result.method == :fastica
                @test size(result.B0) == (n, n)
                @test size(result.W) == (n, n)
                @test size(result.Q) == (n, n)
                @test size(result.shocks) == (n_obs - 2, n)

                # Q should be orthogonal
                @test norm(result.Q' * result.Q - I) < 1e-6

                # B₀ B₀' ≈ Σ
                @test norm(result.B0 * result.B0' - model.Sigma) / norm(model.Sigma) < 0.5

                # Show method
                buf = IOBuffer()
                show(buf, result)
                @test occursin("ICA-SVAR", String(take!(buf)))
            end

            @testset "FastICA symmetric" begin
                result = identify_fastica(model; approach=:symmetric)
                @test result isa ICASVARResult{Float64}
                @test norm(result.Q' * result.Q - I) < 1e-6
            end

            @testset "FastICA contrasts" begin
                for contrast in [:logcosh, :exp, :kurtosis]
                    result = identify_fastica(model; contrast=contrast)
                    @test result isa ICASVARResult{Float64}
                    @test norm(result.Q' * result.Q - I) < 1e-6
                end
            end

            @testset "JADE" begin
                result = identify_jade(model)
                @test result isa ICASVARResult{Float64}
                @test result.method == :jade
                @test size(result.B0) == (n, n)
                @test norm(result.Q' * result.Q - I) < 1e-6
            end

            @testset "SOBI" begin
                result = identify_sobi(model; lags=1:5)
                @test result isa ICASVARResult{Float64}
                @test result.method == :sobi
                @test size(result.B0) == (n, n)
                @test norm(result.Q' * result.Q - I) < 1e-6
            end

            @testset "dCov" begin
                result = identify_dcov(model)
                @test result isa ICASVARResult{Float64}
                @test result.method == :dcov
                @test size(result.B0) == (n, n)
                @test norm(result.Q' * result.Q - I) < 1e-6
                @test result.objective >= 0
            end

            @testset "HSIC" begin
                result = identify_hsic(model)
                @test result isa ICASVARResult{Float64}
                @test result.method == :hsic
                @test size(result.B0) == (n, n)
                @test norm(result.Q' * result.Q - I) < 1e-6
            end
        end
    end

    @testset "Non-Gaussian ML Identification" begin
        _suppress_warnings() do
            @testset "Student-t" begin
                result = identify_student_t(model)
                @test result isa NonGaussianMLResult{Float64}
                @test result.distribution == :student_t
                @test size(result.B0) == (n, n)
                @test size(result.Q) == (n, n)
                @test norm(result.Q' * result.Q - I) < 1e-6
                @test haskey(result.dist_params, :nu)
                @test length(result.dist_params[:nu]) == n
                @test all(result.dist_params[:nu] .> 2)
                @test result.loglik > -Inf
                @test size(result.se) == (n, n)
                @test result.aic > -Inf
                @test result.bic > -Inf

                # Show method
                buf = IOBuffer()
                show(buf, result)
                @test occursin("Non-Gaussian ML", String(take!(buf)))
            end

            @testset "Mixture normal" begin
                result = identify_mixture_normal(model)
                @test result isa NonGaussianMLResult{Float64}
                @test result.distribution == :mixture_normal
                @test norm(result.Q' * result.Q - I) < 1e-6

                # Verify unit variance constraint: σ₂² > 0 (sigmoid bound)
                p_mix = result.dist_params[:p_mix]
                sigma1 = result.dist_params[:sigma1]
                for j in 1:n
                    sigma2_sq = (1.0 - p_mix[j] * sigma1[j]^2) / (1.0 - p_mix[j])
                    @test sigma2_sq > 0
                end
            end

            @testset "PML" begin
                result = identify_pml(model)
                @test result isa NonGaussianMLResult{Float64}
                @test result.distribution == :pml
                @test haskey(result.dist_params, :kappa)
                @test haskey(result.dist_params, :nu)
            end

            @testset "Skew normal" begin
                result = identify_skew_normal(model)
                @test result isa NonGaussianMLResult{Float64}
                @test result.distribution == :skew_normal
                @test haskey(result.dist_params, :alpha)
                @test norm(result.Q' * result.Q - I) < 1e-6
            end

            @testset "Unified dispatcher" begin
                for dist in [:student_t, :mixture_normal, :pml, :skew_normal]
                    result = identify_nongaussian_ml(model; distribution=dist)
                    @test result isa NonGaussianMLResult{Float64}
                    @test result.distribution == dist
                end
            end
        end
    end

    @testset "Heteroskedasticity Identification" begin
        _suppress_warnings() do
            @testset "Markov-switching" begin
                result = identify_markov_switching(model; n_regimes=2)
                @test result isa MarkovSwitchingSVARResult{Float64}
                @test size(result.B0) == (n, n)
                @test size(result.Q) == (n, n)
                @test length(result.Sigma_regimes) == 2
                @test all(size(S) == (n, n) for S in result.Sigma_regimes)
                @test size(result.regime_probs) == (n_obs - 2, 2)
                @test size(result.transition_matrix) == (2, 2)
                @test result.n_regimes == 2
                @test result.loglik > -Inf

                # Regime probs should sum to ~1
                @test all(sum(result.regime_probs, dims=2) .≈ 1.0)

                # Transition matrix rows sum to 1
                @test all(isapprox.(sum(result.transition_matrix, dims=2), 1.0, atol=1e-6))

                @test size(result.se) == (n, n)
                @test all(s -> isnan(s) || s >= 0, result.se)
                @test 0 < result.classification_quality <= 1

                # Show method
                buf = IOBuffer()
                show(buf, result)
                txt = String(take!(buf))
                @test occursin("Markov-Switching", txt)
                @test occursin("Std.Err.", txt)
            end

            @testset "GARCH" begin
                result = identify_garch(model)
                @test result isa GARCHSVARResult{Float64}
                @test size(result.B0) == (n, n)
                @test size(result.garch_params) == (n, 3)
                @test size(result.cond_var) == (n_obs - 2, n)
                @test size(result.shocks) == (n_obs - 2, n)
                @test all(result.cond_var .> 0)  # conditional variances positive

                # GARCH params: omega > 0, alpha >= 0, beta >= 0
                @test all(result.garch_params[:, 1] .> 0)
                @test all(result.garch_params[:, 2] .>= 0)
                @test all(result.garch_params[:, 3] .>= 0)

                # B₀ = L * Q relationship (Givens-based parametrization)
                L = MacroEconometricModels.safe_cholesky(model.Sigma)
                @test Matrix(L) * result.Q ≈ result.B0 atol=1e-10

                # Q should be orthogonal
                @test norm(result.Q' * result.Q - I) < 1e-10

                @test size(result.se) == (n, n)
                @test all(s -> isnan(s) || s >= 0, result.se)

                buf = IOBuffer()
                show(buf, result)
                txt = String(take!(buf))
                @test occursin("GARCH-SVAR", txt)
                @test occursin("Std.Err.", txt)
            end

            @testset "Smooth transition" begin
                Random.seed!(99)
                s = randn(n_obs)
                result = identify_smooth_transition(model, s)
                @test result isa SmoothTransitionSVARResult{Float64}
                @test size(result.B0) == (n, n)
                @test length(result.Sigma_regimes) == 2
                @test result.gamma > 0
                @test length(result.G_values) == n_obs - 2
                @test all(0 .<= result.G_values .<= 1)  # logistic in [0,1]
                @test size(result.se) == (n, n)
                @test size(result.vcov, 1) == size(result.vcov, 2)
                @test all(s -> isnan(s) || s >= 0, result.se)

                buf = IOBuffer()
                show(buf, result)
                txt = String(take!(buf))
                @test occursin("Smooth-Transition", txt)
                @test occursin("Std.Err.", txt)
            end

            @testset "External volatility" begin
                regime = vcat(fill(1, 150), fill(2, 150))
                result = identify_external_volatility(model, regime)
                @test result isa ExternalVolatilitySVARResult{Float64}
                @test size(result.B0) == (n, n)
                @test length(result.Sigma_regimes) == 2
                @test length(result.Lambda) == 2
                @test length(result.regime_indices) == 2
                @test result.loglik > -Inf

                @test size(result.se) == (n, n)
                @test all(s -> isnan(s) || s >= 0, result.se)

                buf = IOBuffer()
                show(buf, result)
                txt = String(take!(buf))
                @test occursin("External Volatility", txt)
                @test occursin("Std.Err.", txt)
            end
        end
    end

    @testset "Identifiability Tests" begin
        _suppress_warnings() do
            ica = identify_fastica(model)
            ml = identify_student_t(model)

            @testset "Shock gaussianity - ICA" begin
                result = test_shock_gaussianity(ica)
                @test result isa IdentifiabilityTestResult{Float64}
                @test result.test_name == :shock_gaussianity
                @test result.statistic >= 0
                @test 0 <= result.pvalue <= 1
                @test haskey(result.details, :jb_stats)
                @test haskey(result.details, :n_gaussian)

                buf = IOBuffer()
                show(buf, result)
                @test occursin("Identifiability Test", String(take!(buf)))
            end

            @testset "Shock gaussianity - ML" begin
                result = test_shock_gaussianity(ml)
                @test result isa IdentifiabilityTestResult{Float64}
                @test result.test_name == :shock_gaussianity
            end

            @testset "Gaussian vs non-Gaussian LR" begin
                result = test_gaussian_vs_nongaussian(model; distribution=:student_t)
                @test result isa IdentifiabilityTestResult{Float64}
                @test result.test_name == :gaussian_vs_nongaussian
                @test result.statistic >= 0
                @test 0 <= result.pvalue <= 1
                @test haskey(result.details, :df)
            end

            @testset "Shock independence - ICA" begin
                result = test_shock_independence(ica; max_lag=5)
                @test result isa IdentifiabilityTestResult{Float64}
                @test result.test_name == :shock_independence
                @test result.statistic >= 0
                @test 0 <= result.pvalue <= 1
                @test haskey(result.details, :cc_statistic)
                @test haskey(result.details, :dcov_statistic)
            end

            @testset "Shock independence - ML" begin
                result = test_shock_independence(ml; max_lag=5)
                @test result isa IdentifiabilityTestResult{Float64}
            end

            @testset "Overidentification" begin
                # ICA is just-identified; overid falls back to label-stability (no p-value)
                result = test_overidentification(model, ica; n_bootstrap=(FAST ? 9 : 19),
                                                 rng=MersenneTwister(75101))
                @test result isa IdentifiabilityTestResult{Float64}
                @test result.test_name == :overidentification
                @test result.statistic >= 0
                @test isnan(result.pvalue)
                @test get(result.details, :fallback, nothing) == :label_stability
                @test haskey(result.details, :match_fraction)
            end

            @testset "Identification strength" begin
                result = test_identification_strength(model; method=:fastica,
                                                      n_bootstrap=(FAST ? 9 : 19),
                                                      rng=MersenneTwister(75102))
                @test result isa IdentifiabilityTestResult{Float64}
                @test result.test_name == :label_stability
                @test 0 <= result.statistic <= 1
                @test isnan(result.pvalue)
                @test haskey(result.details, :n_bootstrap)
                @test haskey(result.details, :match_fraction)
            end
        end
    end

    @testset "compute_Q Integration" begin
        _suppress_warnings() do
            # Test that ICA methods work through compute_Q → irf pipeline
            for method in [:fastica, :jade, :sobi, :dcov, :hsic]
                Q = MacroEconometricModels.compute_Q(model, method, 10, nothing, nothing)
                @test size(Q) == (n, n)
                @test norm(Q' * Q - I) < 1e-4
            end

            # Non-Gaussian ML methods through compute_Q
            for method in [:student_t, :mixture_normal, :pml, :skew_normal]
                Q = MacroEconometricModels.compute_Q(model, method, 10, nothing, nothing)
                @test size(Q) == (n, n)
                @test norm(Q' * Q - I) < 1e-4
            end

            # Heteroskedasticity methods
            for method in [:markov_switching, :garch]
                Q = MacroEconometricModels.compute_Q(model, method, 10, nothing, nothing)
                @test size(Q) == (n, n)
            end

            # irf integration
            irf_result = irf(model, 10; method=:fastica)
            @test size(irf_result.values) == (10, n, n)
        end
    end

    @testset "Type Hierarchy" begin
        _suppress_warnings() do
            @test AbstractNormalityTest <: StatsAPI.HypothesisTest
            @test AbstractNonGaussianSVAR isa DataType

            ica = identify_fastica(model)
            @test ica isa AbstractNonGaussianSVAR

            ml = identify_student_t(model)
            @test ml isa AbstractNonGaussianSVAR

            ms = identify_markov_switching(model)
            @test ms isa AbstractNonGaussianSVAR

            garch = identify_garch(model)
            @test garch isa AbstractNonGaussianSVAR

            st = identify_smooth_transition(model, randn(n_obs))
            @test st isa AbstractNonGaussianSVAR

            ev = identify_external_volatility(model, vcat(fill(1, 150), fill(2, 150)))
            @test ev isa AbstractNonGaussianSVAR
        end
    end

    @testset "Bivariate model" begin
        _suppress_warnings() do
            Y2 = randn(200, 2)
            model2 = estimate_var(Y2, 1)

            ica2 = identify_fastica(model2)
            @test size(ica2.B0) == (2, 2)
            @test norm(ica2.Q' * ica2.Q - I) < 1e-6

            ml2 = identify_student_t(model2)
            @test size(ml2.B0) == (2, 2)

            ms2 = identify_markov_switching(model2)
            @test size(ms2.B0) == (2, 2)
        end
    end

    @testset "FastICA symmetric + contrasts" begin
        _suppress_warnings() do
            for contrast in [:exp, :kurtosis]
                result = identify_fastica(model; approach=:symmetric, contrast=contrast)
                @test result isa ICASVARResult{Float64}
                @test norm(result.Q' * result.Q - I) < 1e-6
            end
        end
    end

    @testset "SOBI with different lag ranges" begin
        _suppress_warnings() do
            result_short = identify_sobi(model; lags=1:3)
            @test result_short isa ICASVARResult{Float64}
            @test result_short.method == :sobi

            result_long = identify_sobi(model; lags=1:20)
            @test result_long isa ICASVARResult{Float64}
        end
    end

    @testset "HSIC with explicit sigma" begin
        _suppress_warnings() do
            result = identify_hsic(model; sigma=2.0)
            @test result isa ICASVARResult{Float64}
            @test result.method == :hsic
        end
    end

    @testset "Identification strength with jade and sobi" begin
        _suppress_warnings() do
            for method in [:jade, :sobi]
                result = test_identification_strength(model; method=method,
                                                      n_bootstrap=(FAST ? 5 : 9),
                                                      rng=MersenneTwister(75103))
                @test result isa IdentifiabilityTestResult{Float64}
                @test result.test_name == :label_stability
                @test isnan(result.pvalue)
            end
        end
    end

    @testset "Gaussian vs non-Gaussian LR with other distributions" begin
        _suppress_warnings() do
            for dist in [:mixture_normal, :pml, :skew_normal]
                result = test_gaussian_vs_nongaussian(model; distribution=dist)
                @test result isa IdentifiabilityTestResult{Float64}
                @test result.statistic >= 0
                @test 0 <= result.pvalue <= 1
            end
        end
    end

    @testset "Markov-switching with 3 regimes" begin
        _suppress_warnings() do
            result = identify_markov_switching(model; n_regimes=3, max_iter=(FAST ? 20 : 50))
            @test result isa MarkovSwitchingSVARResult{Float64}
            @test result.n_regimes == 3
            @test length(result.Sigma_regimes) == 3
            @test size(result.transition_matrix) == (3, 3)
        end
    end

    @testset "External volatility with 3 regimes" begin
        _suppress_warnings() do
            regime3 = vcat(fill(1, 100), fill(2, 100), fill(3, 100))
            result = identify_external_volatility(model, regime3; regimes=3)
            @test result isa ExternalVolatilitySVARResult{Float64}
            @test length(result.Sigma_regimes) == 3
            @test length(result.Lambda) == 3
        end
    end

    @testset "External volatility with small regime" begin
        _suppress_warnings() do
            # Regime 3 has fewer than n+1 residual observations
            regime_small = vcat(fill(1, 148), fill(2, 148), fill(3, 4))
            @test_throws ArgumentError identify_external_volatility(model, regime_small; regimes=3)
        end
    end

    @testset "SID-09 smooth-transition joint ML" begin
        Random.seed!(738)
        m = estimate_var(randn(80, 2), 1)
        @test_throws ArgumentError identify_external_volatility(m, vcat(fill(1, 78), fill(2, 2)))
        m3 = estimate_var(randn(120, 2), 1)
        ri3 = vcat(fill(1, 40), fill(2, 40), fill(3, 40))
        ev3 = identify_external_volatility(m3, ri3; regimes=3)
        @test length(ev3.Lambda) == 3
        @test size(ev3.se) == (2, 2)
        st = identify_smooth_transition(m, randn(size(m.U, 1)))
        @test size(st.se) == (2, 2)
        @test st.vcov isa AbstractMatrix
        B0_reid, _, _ = MacroEconometricModels._eigendecomposition_id(
            Matrix(st.Sigma_regimes[1]), Matrix(st.Sigma_regimes[2]))
        @test norm(st.B0 - B0_reid) < 1e-6
    end

    @testset "SID-10 K-regime ML, SEs, and tests" begin
        Random.seed!(739)
        n_sid = 2
        B_rec = [1.0 0.0; 0.5 1.0]
        Λ = [0.4, 3.0]
        A = [0.4 * Matrix{Float64}(I, n_sid, n_sid)]
        Ysid, regime = simulate_two_regime(B_rec, A, Λ; Tobs=800, split=0.5,
                                           rng=MersenneTwister(739))
        modelsid = estimate_var(Ysid, 1)
        ri = regime[2:end]
        ev = identify_external_volatility(modelsid, ri; regimes=2)
        @test size(ev.se) == (n_sid, n_sid)
        @test all(isfinite, ev.se)
        @test all(ev.se .>= 0)

        w = test_lambda_distinct(ev; pairs=:all)
        @test w isa NamedTuple
        @test haskey(w, :statistic) && haskey(w, :pvalue_bonferroni)
        @test all(w.pvalue_bonferroni .>= 0)
        @test all(w.pvalue_bonferroni .<= 1)

        mask_true = [NaN 0.0; NaN NaN]
        mask_false = [NaN NaN; 0.0 NaN]
        lr_true = test_restrictions(ev, mask_true)
        lr_false = test_restrictions(ev, mask_false)
        @test lr_true.pvalue > 0.05
        @test lr_false.pvalue < 0.05

        ms = identify_markov_switching(modelsid; n_regimes=2, n_starts=2,
                                       rng=MersenneTwister(739), max_iter=(FAST ? 20 : 80))
        @test ms.classification_quality > 0
        ms2 = identify_markov_switching(modelsid; n_regimes=2, n_starts=2,
                                        rng=MersenneTwister(739), max_iter=(FAST ? 20 : 80))
        @test ms.B0 ≈ ms2.B0 atol=1e-8

        st = identify_smooth_transition(modelsid, modelsid.U[:, 1])
        garch = identify_garch(modelsid; max_iter=(FAST ? 5 : 30))
        for r in (ev, ms, st, garch)
            buf = IOBuffer()
            show(buf, r)
            @test occursin("Std.Err.", String(take!(buf)))
        end

        # Empty / all-zero vcov must not produce a confident Wald
        nB2 = n_sid * n_sid
        pdim2 = nB2 + n_sid
        ev_empty = ExternalVolatilitySVARResult{Float64}(
            ev.B0, ev.Q, ev.Sigma_regimes, ev.Lambda, ev.regime_indices,
            ev.loglik, fill(NaN, n_sid, n_sid), zeros(Float64, 0, 0))
        ev_zero = ExternalVolatilitySVARResult{Float64}(
            ev.B0, ev.Q, ev.Sigma_regimes, ev.Lambda, ev.regime_indices,
            ev.loglik, fill(NaN, n_sid, n_sid), zeros(Float64, pdim2, pdim2))
        for rbad in (ev_empty, ev_zero)
            try
                wbad = test_lambda_distinct(rbad; pairs=:all)
                @test all(isnan, wbad.statistic)
                @test all(p -> isnan(p) || ismissing(p), wbad.pvalue)
            catch e
                @test e isa ArgumentError
            end
        end

        # K=3: Λ₂ tied, Λ₃ separates — must not report failure from Λ₂ alone
        B_k3 = [1.0 0.0; 0.3 1.0]
        Λ_k3 = [[1.0, 1.0], [2.0, 2.0], [0.5, 4.0]]
        Σ_k3 = [B_k3 * Diagonal(λ) * B_k3' for λ in Λ_k3]
        pdim3 = n_sid * n_sid + n_sid * (length(Λ_k3) - 1)
        V_k3 = Matrix{Float64}(I, pdim3, pdim3) * 0.01
        ev_k3 = ExternalVolatilitySVARResult{Float64}(
            B_k3, Matrix{Float64}(I, n_sid, n_sid), Σ_k3, Λ_k3,
            [collect(1:3) for _ in 1:3], -10.0, fill(NaN, n_sid, n_sid), V_k3)
        w_k3 = test_lambda_distinct(ev_k3; pairs=[(1, 2)])
        @test isfinite(w_k3.pvalue[1])
        @test w_k3.pvalue[1] < 0.05

        # GARCH LR uses cond_var columns matching the signed permutation of B₀
        n_g = size(garch.B0, 1)
        perm_g = collect(n_g:-1:1)
        g_perm = GARCHSVARResult{Float64}(
            garch.B0[:, perm_g], garch.Q[:, perm_g],
            garch.garch_params[perm_g, :], garch.cond_var[:, perm_g],
            garch.shocks[:, perm_g], garch.loglik, garch.converged,
            garch.iterations)
        mask_g = fill(NaN, n_g, n_g)
        mask_g[1, n_g] = 0.0
        lr_g0 = test_restrictions(garch, mask_g)
        lr_gp = test_restrictions(g_perm, mask_g)
        @test lr_g0.statistic ≈ lr_gp.statistic rtol=1e-6 atol=1e-8

        if !FAST
            n_reps = 200
            Tobs_w = 500
            # Size: two components with equal relative variances
            n_rej_size = 0
            n_ok_size = 0
            for r in 1:n_reps
                rng_r = MersenneTwister(73900 + r)
                Ys, regs = simulate_two_regime(B_rec, A, [1.5, 1.5]; Tobs=Tobs_w,
                                               split=0.5, rng=rng_r)
                m_s = estimate_var(Ys, 1)  # size/power draws; does not clobber `model`
                try
                    ev_s = identify_external_volatility(m_s, regs[2:end]; regimes=2)
                    wt = test_lambda_distinct(ev_s; pairs=[(1, 2)])
                    n_ok_size += 1
                    n_rej_size += Int(wt.pvalue[1] < 0.05)
                catch
                    continue
                end
            end
            @test n_ok_size >= 100
            size_est = n_rej_size / n_ok_size
            @test 0.01 <= size_est <= 0.12

            n_rej_pow = 0
            n_ok_pow = 0
            for r in 1:100
                rng_r = MersenneTwister(73950 + r)
                Yp, regp = simulate_two_regime(B_rec, A, [1.0, 2.0]; Tobs=Tobs_w,
                                               split=0.5, rng=rng_r)
                m_p = estimate_var(Yp, 1)
                try
                    ev_p = identify_external_volatility(m_p, regp[2:end]; regimes=2)
                    wt = test_lambda_distinct(ev_p; pairs=[(1, 2)])
                    n_ok_pow += 1
                    n_rej_pow += Int(wt.pvalue[1] < 0.05)
                catch
                    continue
                end
            end
            @test n_ok_pow >= 50
            @test n_rej_pow / n_ok_pow > 0.80
        end
    end

    @testset "Smooth transition edge cases" begin
        _suppress_warnings() do
            Random.seed!(99999)
            # Extreme transition variable (all same sign)
            s_edge = abs.(randn(n_obs)) .+ 5.0
            result = identify_smooth_transition(model, s_edge)
            @test result isa SmoothTransitionSVARResult{Float64}
            @test result.gamma > 0
        end
    end

    @testset "GARCH max_iter=1" begin
        _suppress_warnings() do
            result = identify_garch(model; max_iter=1)
            @test result isa GARCHSVARResult{Float64}
            @test result.iterations == 1
        end
    end

    @testset "4-variable scalability" begin
        _suppress_warnings() do
            Random.seed!(55555)
            Y4 = randn(200, 4)
            model4 = estimate_var(Y4, 1)
            ica4 = identify_fastica(model4)
            @test size(ica4.B0) == (4, 4)
            @test norm(ica4.Q' * ica4.Q - I) < 1e-4

            ml4 = identify_student_t(model4; max_iter=(FAST ? 30 : 100))
            @test size(ml4.B0) == (4, 4)
        end
    end

    # =================================================================
    # Integration Tests: Non-Gaussian through FEVD / HD / BVAR / LP
    # =================================================================

    @testset "Non-Gaussian FEVD Integration" begin
        _suppress_warnings() do
            for method in [:fastica, :student_t, :markov_switching]
                f = fevd(model, 10; method=method)
                @test f isa MacroEconometricModels.FEVD
                @test size(f.proportions) == (n, n, 10)
                # Each variable's FEVD proportions sum to ~1 at each horizon
                for h in 1:10, i in 1:n
                    @test sum(f.proportions[i, :, h]) ≈ 1.0 atol=1e-10
                end
            end
        end
    end

    @testset "Non-Gaussian HD Integration" begin
        _suppress_warnings() do
            T_eff = n_obs - 2  # p=2
            for method in [:fastica, :student_t]
                hd_r = historical_decomposition(model, T_eff; method=method)
                @test hd_r isa MacroEconometricModels.HistoricalDecomposition
                @test verify_decomposition(hd_r)
                @test hd_r.method == method
            end
        end
    end

    @testset "compute_Q new methods" begin
        _suppress_warnings() do
            # :nongaussian_ml
            Q_ngml = MacroEconometricModels.compute_Q(model, :nongaussian_ml, 10, nothing, nothing)
            @test size(Q_ngml) == (n, n)
            @test norm(Q_ngml' * Q_ngml - I) < 1e-4

            # :smooth_transition with transition_var
            tv = randn(n_obs)
            Q_st = MacroEconometricModels.compute_Q(model, :smooth_transition, 10, nothing, nothing;
                                                     transition_var=tv)
            @test size(Q_st) == (n, n)

            # :external_volatility with regime_indicator
            ri = vcat(fill(1, 150), fill(2, 150))
            Q_ev = MacroEconometricModels.compute_Q(model, :external_volatility, 10, nothing, nothing;
                                                     regime_indicator=ri)
            @test size(Q_ev) == (n, n)

            # Missing kwargs should error
            @test_throws ArgumentError MacroEconometricModels.compute_Q(model, :smooth_transition, 10, nothing, nothing)
            @test_throws ArgumentError MacroEconometricModels.compute_Q(model, :external_volatility, 10, nothing, nothing)
        end
    end

    @testset "Hetero-ID through irf/fevd/hd" begin
        _suppress_warnings() do
            tv = randn(n_obs)
            ri = vcat(fill(1, 150), fill(2, 150))
            T_eff = n_obs - 2

            irf_st = irf(model, 10; method=:smooth_transition, transition_var=tv)
            @test irf_st isa MacroEconometricModels.ImpulseResponse
            @test size(irf_st.values) == (10, n, n)

            fevd_st = fevd(model, 10; method=:smooth_transition, transition_var=tv)
            @test fevd_st isa MacroEconometricModels.FEVD

            hd_st = historical_decomposition(model, T_eff; method=:smooth_transition, transition_var=tv)
            @test hd_st isa MacroEconometricModels.HistoricalDecomposition
            @test verify_decomposition(hd_st)

            irf_ev = irf(model, 10; method=:external_volatility, regime_indicator=ri)
            @test irf_ev isa MacroEconometricModels.ImpulseResponse

            fevd_ev = fevd(model, 10; method=:external_volatility, regime_indicator=ri)
            @test fevd_ev isa MacroEconometricModels.FEVD

            hd_ev = historical_decomposition(model, T_eff; method=:external_volatility, regime_indicator=ri)
            @test hd_ev isa MacroEconometricModels.HistoricalDecomposition
            @test verify_decomposition(hd_ev)
        end
    end

    @testset "BVAR Non-Gaussian Identification" begin
        _suppress_warnings() do
            Random.seed!(77777)
            post = estimate_bvar(Y, 2; n_draws=(FAST ? 15 : 30))
            for method in [:fastica, :student_t]
                irf_r = irf(post, 10; method=method)
                @test irf_r isa MacroEconometricModels.BayesianImpulseResponse

                f = fevd(post, 10; method=method)
                @test f isa MacroEconometricModels.BayesianFEVD

                hd_r = historical_decomposition(post, n_obs - 2; data=Y, method=method)
                @test hd_r isa MacroEconometricModels.BayesianHistoricalDecomposition
            end
        end
    end

    @testset "Structural LP Non-Gaussian" begin
        _suppress_warnings() do
            Random.seed!(88888)
            Y_lp = randn(200, 3)
            for method in [:fastica, :student_t]
                slp = structural_lp(Y_lp, 8; method=method, lags=2)
                @test slp isa MacroEconometricModels.StructuralLP
                @test slp.method == method

                f = fevd(slp, 8)
                @test f isa MacroEconometricModels.LPFEVD

                T_eff_lp = size(Y_lp, 1) - 2
                hd_r = historical_decomposition(slp, T_eff_lp)
                @test hd_r isa MacroEconometricModels.HistoricalDecomposition
            end
        end
    end

    @testset "SID-21 GMM moments" begin
        _suppress_warnings() do
            Random.seed!(750)
            Y2 = randn(250, 2)
            model2 = estimate_var(Y2, 1)
            n2 = 2
            n_angles = n2 * (n2 - 1) ÷ 2

            @testset "API and result fields" begin
                result = identify_gmm_moments(model2; moments=:cokurtosis, weighting=:two_step)
                @test result isa NonGaussianGMMResult{Float64}
                @test result isa AbstractNonGaussianSVAR
                @test size(result.B0) == (n2, n2)
                @test size(result.Q) == (n2, n2)
                @test length(result.theta) == n_angles
                @test size(result.vcov) == (n_angles, n_angles)
                @test size(result.se) == (n2, n2)
                @test all(s -> isnan(s) || s >= 0, result.se)
                @test size(result.shocks) == (size(model2.U, 1), n2)
                @test result.moments == :cokurtosis
                @test result.weighting == :two_step
                @test result.J >= 0
                @test isnan(result.J_pvalue) || (0 <= result.J_pvalue <= 1)
                @test length(result.varnames) == n2
                @test length(result.shock_names) == n2
                @test norm(result.Q' * result.Q - I) < 1e-8
                L = MacroEconometricModels.safe_cholesky(model2.Sigma)
                @test norm(result.B0 * result.B0' - model2.Sigma) / norm(model2.Sigma) < 1e-6
            end

            @testset "moment and weighting variants" begin
                for mom in (:coskewness, :cokurtosis, :both)
                    r = identify_gmm_moments(model2; moments=mom)
                    @test r.moments == mom
                    @test r isa NonGaussianGMMResult
                end
                r_cue = identify_gmm_moments(model2; moments=:cokurtosis, weighting=:cue)
                @test r_cue.weighting == :cue
                @test r_cue isa NonGaussianGMMResult
            end

            @testset "invalid kwargs" begin
                @test_throws ArgumentError identify_gmm_moments(model2; moments=:skewness)
                @test_throws ArgumentError identify_gmm_moments(model2; weighting=:identity)
                @test_throws ArgumentError identify_gmm_moments(model2; se=:bootstrap)
            end

            @testset "show / report prints J and SEs" begin
                result = identify_gmm_moments(model2; moments=:both, weighting=:two_step)
                buf = IOBuffer()
                show(buf, result)
                txt = String(take!(buf))
                @test occursin("GMM", txt)
                @test occursin("J", txt)
                @test occursin("both", txt)
                @test occursin("two_step", txt)
                @test occursin("Std.Err.", txt)
            end

            @testset "registry and compute_Q" begin
                @test haskey(MacroEconometricModels.IDENTIFICATION_REGISTRY, :gmm_moments)
                @test MacroEconometricModels._needs_residuals(:gmm_moments)
                @test !MacroEconometricModels._is_set_identified(:gmm_moments)
                @test !MacroEconometricModels._is_partial(:gmm_moments)
                @test MacroEconometricModels._should_match_columns(:gmm_moments)
                Q = MacroEconometricModels.compute_Q(model2, :gmm_moments)
                @test size(Q) == (n2, n2)
                @test norm(Q' * Q - I) < 1e-6
                ir = irf(model2, 8; method=:gmm_moments)
                @test size(ir.values) == (8, n2, n2)
            end

            @testset "plot_result mixing heatmap" begin
                result = identify_gmm_moments(model2; moments=:cokurtosis)
                p = plot_result(result)
                @test occursin("Mixing Matrix", p.html)
                @test_throws ArgumentError plot_result(result; view=:shocks)
            end

            @testset "test_gaussian_shock_count" begin
                Tobs = FAST ? 400 : 2000
                rng = MersenneTwister(75021)
                n = 3
                # Two Gaussian shocks + one t(5): identification fails
                shocks_two = randn(rng, Tobs, n)
                ν = 5.0
                shocks_two[:, 3] = rand(rng, TDist(ν), Tobs) .* sqrt((ν - 2) / ν)
                dummy_two = NonGaussianGMMResult{Float64}(
                    Matrix{Float64}(I, n, n), Matrix{Float64}(I, n, n),
                    zeros(3), Matrix{Float64}(I, 3, 3), zeros(n, n),
                    0.0, 1.0, :both, :two_step, shocks_two,
                    ["y$i" for i in 1:n], ["Shock $j" for j in 1:n])
                cnt_two = test_gaussian_shock_count(dummy_two)
                @test cnt_two isa IdentifiabilityTestResult{Float64}
                @test cnt_two.test_name == :gaussian_shock_count
                @test cnt_two.identified == false
                @test cnt_two.details[:n_gaussian] >= 2

                # All t(5): at most one Gaussian → identification holds
                shocks_t = rand(rng, TDist(ν), Tobs, n) .* sqrt((ν - 2) / ν)
                dummy_t = NonGaussianGMMResult{Float64}(
                    Matrix{Float64}(I, n, n), Matrix{Float64}(I, n, n),
                    zeros(3), Matrix{Float64}(I, 3, 3), zeros(n, n),
                    0.0, 1.0, :cokurtosis, :two_step, shocks_t,
                    ["y$i" for i in 1:n], ["Shock $j" for j in 1:n])
                cnt_t = test_gaussian_shock_count(dummy_t)
                @test cnt_t.identified == true
                @test cnt_t.details[:n_gaussian] <= 1
            end
        end
    end

    @testset "SID-20 shock labels and structural_shocks" begin
        B_true = [1.2 0.35 -0.25; 0.20 1.05 0.40; -0.30 0.25 1.10]
        n3 = 3
        perm0 = [2, 3, 1]
        signs0 = [1, -1, 1]
        B_shuf = B_true[:, perm0] .* signs0'
        Q_shuf = Matrix{Float64}(I, n3, n3)[:, perm0] .* signs0'
        W_shuf = Matrix{Float64}(I, n3, n3)
        shocks_shuf = randn(20, n3)
        ica_shuf = ICASVARResult{Float64}(B_shuf, W_shuf, Q_shuf, shocks_shuf,
                                          :fastica, true, 1, 0.0)
        @test length(ica_shuf.shock_names) == n3
        @test ica_shuf.shock_names == ["Shock $j" for j in 1:n3]

        S = Int.(sign.(B_true))
        lab = label_shocks(ica_shuf; by=:restrictions, restrictions=S)
        @test lab isa ICASVARResult{Float64}
        @test lab.B0 ≈ B_true atol = 1e-12
        @test lab.Q ≈ Matrix{Float64}(I, n3, n3) atol = 1e-12

        lab_imp = label_shocks(ica_shuf; by=:max_impact)
        @test lab_imp.B0 ≈ B_true atol = 1e-12

        lab_ref = label_shocks(ica_shuf; by=:reference, B_ref=B_true)
        @test lab_ref.B0 ≈ B_true atol = 1e-12

        # `:reference` keeps `_match_columns` signs (negative own-effect stays).
        B_neg = copy(B_true)
        B_neg[:, 2] .*= -1
        @test B_neg[2, 2] < 0
        B_shuf_neg = B_neg[:, perm0] .* signs0'
        ica_neg = ICASVARResult{Float64}(B_shuf_neg, W_shuf, Q_shuf, shocks_shuf,
                                         :fastica, true, 1, 0.0)
        lab_neg = label_shocks(ica_neg; by=:reference, B_ref=B_neg)
        @test lab_neg.B0[2, 2] < 0
        @test lab_neg.B0 ≈ B_neg atol = 1e-12

        named = label_shocks(ica_shuf; by=:max_impact,
                             shock_names=["Demand", "Supply", "MP"])
        @test named.shock_names == ["Demand", "Supply", "MP"]
        buf = IOBuffer()
        show(buf, named)
        @test occursin("Demand", String(take!(buf)))
        p = plot_result(named)
        @test occursin("Demand", p.html)

        rs = SVARRestrictions(n3; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(1, 2, :positive),
            sign_restriction(1, 3, :negative),
            sign_restriction(2, 1, :positive),
            sign_restriction(2, 2, :positive),
            sign_restriction(2, 3, :positive),
            sign_restriction(3, 1, :negative),
            sign_restriction(3, 2, :positive),
            sign_restriction(3, 3, :positive),
        ])
        lab_sv = label_shocks(ica_shuf; by=:restrictions, restrictions=rs)
        @test lab_sv.B0 ≈ B_true atol = 1e-12

        @test structural_shocks(named) == named.shocks

        # Det-reversing signed permutation: Q stays L\B0; Givens theta is not
        # overwritten with SO(n) angles that reconstruct a different rotation.
        n2 = 2
        L2 = [1.2 0.0; 0.4 1.1]
        Q2 = Matrix{Float64}(I, n2, n2)
        B2 = L2 * Q2
        theta0 = zeros(1)
        gmm0 = NonGaussianGMMResult{Float64}(
            B2, Q2, copy(theta0), Matrix{Float64}(I, 1, 1), zeros(n2, n2),
            0.0, 1.0, :cokurtosis, :two_step, randn(12, n2),
            ["y1", "y2"], ["Shock 1", "Shock 2"])
        B_swap = B2[:, [2, 1]]
        lab_g = @test_logs (:warn, r"det\(Q\)") match_mode = :any begin
            label_shocks(gmm0; by=:reference, B_ref=B_swap)
        end
        @test det(lab_g.Q) < 0
        @test lab_g.Q ≈ L2 \ lab_g.B0 atol = 1e-12
        @test lab_g.B0 ≈ B_swap atol = 1e-12
        @test all(isnan, lab_g.theta)

        ml_old = NonGaussianMLResult{Float64}(
            B_true, Matrix{Float64}(I, n3, n3), randn(12, n3), :student_t,
            -1.0, -2.0, Dict{Symbol,Any}(), zeros(n3, n3), zeros(n3, n3),
            true, 1, 1.0, 2.0)
        @test length(ml_old.shock_names) == n3
        @test structural_shocks(ml_old) == ml_old.shocks

        # DGP recovery: FastICA + sign-pattern labelling restores column order
        # without a further Procrustes search (SID-20 acceptance).
        if !FAST
            B_dgp = [1.0 0.35; -0.40 1.15]
            A = [0.4 * Matrix{Float64}(I, 2, 2)]
            rng_d = MersenneTwister(74920)
            Yd, _ = simulate_svar(B_dgp, A; Tobs=2000, shocks=:t, rng=rng_d)
            md = estimate_var(Yd, 1)
            ica_d = identify_fastica(md; rng=MersenneTwister(74921))
            Sd = Int.(sign.(B_dgp))
            lab_d = label_shocks(ica_d; by=:restrictions, restrictions=Sd)
            @test norm(lab_d.B0 - B_dgp) < 0.1
            lab_m = label_shocks(ica_d; by=:max_impact)
            @test norm(lab_m.B0 - B_dgp) < 0.1
        end
    end

    @testset "SID-22 principled identification and overidentification tests" begin
        _suppress_warnings() do
            Random.seed!(751)

            @testset "label-stability reports match fraction and no p-value" begin
                Y2 = randn(MersenneTwister(75110), 180, 2)
                m2 = estimate_var(Y2, 1)
                stab = test_label_stability(m2; method=:fastica, n_bootstrap=(FAST ? 7 : 15),
                                            rng=MersenneTwister(75111))
                @test stab isa IdentifiabilityTestResult{Float64}
                @test stab.test_name == :label_stability
                @test 0 <= stab.statistic <= 1
                @test isnan(stab.pvalue)
                @test stab.details[:match_fraction] == stab.statistic
                @test stab.details[:n_bootstrap] > 0
                @test 0 <= stab.details[:n_identity] <= stab.details[:n_bootstrap]
                buf = IOBuffer()
                show(buf, stab)
                txt = String(take!(buf))
                @test occursin("label_stability", txt)
                @test occursin("Match fraction", txt)
                @test !occursin("***", txt)  # no significance stars without a p-value
            end

            @testset "identity permutation is the match criterion" begin
                B = [1.2 0.3; 0.4 1.1]
                perm_id, _ = MacroEconometricModels._match_columns(B, B)
                @test perm_id == [1, 2]
                Bswap = B[:, [2, 1]]
                perm_sw, _ = MacroEconometricModels._match_columns(B, Bswap)
                @test perm_sw == [2, 1]
            end

            @testset "deprecated strength wrapper: ICA → label-stability" begin
                Y2 = randn(MersenneTwister(75120), 160, 2)
                m2 = estimate_var(Y2, 1)
                r = test_identification_strength(m2; method=:fastica, n_bootstrap=5,
                                                 rng=MersenneTwister(75121))
                @test r.test_name == :label_stability
                @test isnan(r.pvalue)
            end

            @testset "deprecated strength wrapper: hetero → λ-distinct" begin
                Yh, rh = simulate_two_regime([1.0 0.0; 0.4 1.0],
                                             [0.4 * Matrix{Float64}(I, 2, 2)],
                                             [0.5, 3.0]; Tobs=400, split=0.5,
                                             rng=MersenneTwister(75130))
                mh = estimate_var(Yh, 1)
                ev = identify_external_volatility(mh, rh[2:end]; regimes=2)
                r = test_identification_strength(ev)
                @test r.test_name == :lambda_distinct
                @test r isa IdentifiabilityTestResult{Float64}
                @test haskey(r.details, :pvalue_bonferroni)
            end

            @testset "deprecated strength wrapper: non-Gaussian → Gaussian count + Holm" begin
                shocks_t = rand(MersenneTwister(75140), TDist(5.0), FAST ? 400 : 800, 3)
                shocks_t .*= sqrt(3 / 5)
                dummy = NonGaussianGMMResult{Float64}(
                    Matrix{Float64}(I, 3, 3), Matrix{Float64}(I, 3, 3),
                    zeros(3), Matrix{Float64}(I, 3, 3), zeros(3, 3),
                    0.0, 1.0, :cokurtosis, :two_step, shocks_t,
                    ["y$i" for i in 1:3], ["Shock $j" for j in 1:3])
                r = test_identification_strength(dummy)
                @test r.test_name == :gaussian_shock_count
                @test r.identified == true
                @test haskey(r.details, :jb_pvals_holm)
                @test haskey(r.details, :kurt_pvals)
                @test r.details[:n_gaussian] <= 1
            end

            @testset "shock independence/gaussianity dispatch including MS" begin
                Ym = randn(MersenneTwister(75150), 220, 2)
                mm = estimate_var(Ym, 1)
                ms = identify_markov_switching(mm; n_regimes=2, n_starts=1,
                                               rng=MersenneTwister(75151),
                                               max_iter=(FAST ? 15 : 40))
                @test size(ms.shocks, 2) == 2
                @test size(ms.shocks, 1) == size(mm.U, 1)
                indep_ms = test_shock_independence(ms; max_lag=3,
                                                   rng=MersenneTwister(75152))
                @test indep_ms isa IdentifiabilityTestResult{Float64}
                @test indep_ms.test_name == :shock_independence
                @test 0 <= indep_ms.pvalue <= 1
                gauss_ms = test_shock_gaussianity(ms)
                @test gauss_ms.test_name == :shock_gaussianity

                garch = identify_garch(mm; max_iter=(FAST ? 5 : 20))
                @test test_shock_independence(garch; max_lag=2,
                                              rng=MersenneTwister(75153)).test_name ==
                      :shock_independence
                @test test_shock_gaussianity(garch).test_name == :shock_gaussianity

                st = identify_smooth_transition(mm, mm.U[:, 1])
                @test test_shock_gaussianity(st).test_name == :shock_gaussianity

                ev = identify_external_volatility(mm, vcat(fill(1, 110), fill(2, size(mm.U, 1) - 110)))
                @test size(ev.shocks, 1) == size(mm.U, 1)
                @test test_shock_independence(ev; max_lag=2,
                                              rng=MersenneTwister(75154)).test_name ==
                      :shock_independence
            end

            @testset "overidentification: ML just-identified and AB LR" begin
                Y2 = randn(MersenneTwister(75160), 200, 2)
                m2 = estimate_var(Y2, 1)
                ml = identify_student_t(m2)
                oid_ml = test_overidentification(m2, ml)
                @test oid_ml.test_name == :overidentification
                @test get(oid_ml.details, :just_identified, false) == true
                @test oid_ml.pvalue == 1.0

                svar = estimate_svar(m2, recursive_pattern(2); rng=MersenneTwister(75161))
                oid_ab = test_overidentification(m2, svar)
                @test oid_ab.test_name == :overidentification
                @test get(oid_ab.details, :method, nothing) == :lr
                @test oid_ab.details[:df] == svar.lr_df
                @test oid_ab.statistic ≈ svar.lr_stat
                @test oid_ab.pvalue ≈ svar.lr_pvalue
                @test get(oid_ab.details, :pattern, nothing) == :stored

                mask_b = [NaN 0.0; NaN NaN]
                oid_ab_mask = test_overidentification(m2, svar; restrictions=mask_b,
                                                      rng=MersenneTwister(75162))
                @test get(oid_ab_mask.details, :pattern, nothing) == :reestimated
                @test oid_ab_mask.test_name == :overidentification
                @test_throws ArgumentError test_overidentification(svar; restrictions=mask_b)
            end

            @testset "overidentification: hetero LR + Wald on a true zero" begin
                Yh, rh = simulate_two_regime([1.0 0.0; 0.45 1.0],
                                             [0.4 * Matrix{Float64}(I, 2, 2)],
                                             [0.4, 3.0]; Tobs=500, split=0.5,
                                             rng=MersenneTwister(75170))
                mh = estimate_var(Yh, 1)
                ev = identify_external_volatility(mh, rh[2:end]; regimes=2)
                mask_true = [NaN 0.0; NaN NaN]
                oid = test_overidentification(mh, ev; restrictions=mask_true)
                @test oid.test_name == :overidentification
                @test get(oid.details, :method, nothing) == :lr
                @test haskey(oid.details, :wald_statistic)
                @test oid.pvalue > 0.01
            end

            @testset "ICA overidentification says it falls back to label-stability" begin
                Y2 = randn(MersenneTwister(75180), 150, 2)
                m2 = estimate_var(Y2, 1)
                ica2 = identify_fastica(m2; rng=MersenneTwister(75181))
                oid = test_overidentification(m2, ica2; n_bootstrap=5,
                                              rng=MersenneTwister(75182))
                @test oid.details[:fallback] == :label_stability
                @test isnan(oid.pvalue)
                buf = IOBuffer()
                show(buf, oid)
                txt = String(take!(buf))
                @test occursin("label-stability", txt) || occursin("label_stability", txt)
            end

            if !FAST
                # Size/power: LR overid of a true recursive restriction under Student-t
                n_reps = 200
                Tobs_lr = 500
                B_rec = [1.0 0.0; 0.45 1.0]
                A_lr = [0.4 * Matrix{Float64}(I, 2, 2)]
                mask_true = [NaN 0.0; NaN NaN]
                mask_false = [NaN NaN; 0.0 NaN]
                n_rej_size = 0
                n_ok_size = 0
                n_rej_pow = 0
                n_ok_pow = 0
                for r in 1:n_reps
                    rng_r = MersenneTwister(75100 + r)
                    Ys, _ = simulate_svar(B_rec, A_lr; Tobs=Tobs_lr, shocks=:t, rng=rng_r)
                    m_s = estimate_var(Ys, 1)
                    try
                        ml_s = identify_student_t(m_s)
                        ot = test_overidentification(m_s, ml_s; restrictions=mask_true)
                        n_ok_size += 1
                        n_rej_size += Int(ot.pvalue < 0.05)
                    catch
                        continue
                    end
                end
                @test n_ok_size >= 100
                size_est = n_rej_size / n_ok_size
                @test 0.01 <= size_est <= 0.12

                for r in 1:100
                    rng_r = MersenneTwister(75300 + r)
                    Yp, _ = simulate_svar(B_rec, A_lr; Tobs=Tobs_lr, shocks=:t, rng=rng_r)
                    m_p = estimate_var(Yp, 1)
                    try
                        ml_p = identify_student_t(m_p)
                        op = test_overidentification(m_p, ml_p; restrictions=mask_false)
                        n_ok_pow += 1
                        n_rej_pow += Int(op.pvalue < 0.05)
                    catch
                        continue
                    end
                end
                @test n_ok_pow >= 50
                @test n_rej_pow / n_ok_pow > 0.80
            end
        end
    end
end
