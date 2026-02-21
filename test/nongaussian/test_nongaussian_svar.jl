# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Distributions
using StatsAPI

@testset "Non-Gaussian SVAR Identification" begin
    Random.seed!(54321)

    # Generate VAR data
    n_obs = 300
    Y = randn(n_obs, 3)
    model = estimate_var(Y, 2)
    n = 3

    @testset "ICA-based Identification" begin
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

    @testset "Non-Gaussian ML Identification" begin
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

    @testset "Heteroskedasticity Identification" begin
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

            # Show method
            buf = IOBuffer()
            show(buf, result)
            @test occursin("Markov-Switching", String(take!(buf)))
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

            buf = IOBuffer()
            show(buf, result)
            @test occursin("GARCH-SVAR", String(take!(buf)))
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

            buf = IOBuffer()
            show(buf, result)
            @test occursin("Smooth-Transition", String(take!(buf)))
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

            buf = IOBuffer()
            show(buf, result)
            @test occursin("External Volatility", String(take!(buf)))
        end
    end

    @testset "Identifiability Tests" begin
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
            result = test_overidentification(model, ica; n_bootstrap=(FAST ? 49 : 99))
            @test result isa IdentifiabilityTestResult{Float64}
            @test result.test_name == :overidentification
            @test result.statistic >= 0
            @test 0 <= result.pvalue <= 1
        end

        @testset "Identification strength" begin
            result = test_identification_strength(model; method=:fastica, n_bootstrap=(FAST ? 19 : 49))
            @test result isa IdentifiabilityTestResult{Float64}
            @test result.test_name == :identification_strength
            @test result.statistic >= 0
            @test 0 <= result.pvalue <= 1
            @test haskey(result.details, :n_bootstrap)
        end
    end

    @testset "compute_Q Integration" begin
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

    @testset "Type Hierarchy" begin
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

    @testset "Bivariate model" begin
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

    @testset "FastICA symmetric + contrasts" begin
        for contrast in [:exp, :kurtosis]
            result = identify_fastica(model; approach=:symmetric, contrast=contrast)
            @test result isa ICASVARResult{Float64}
            @test norm(result.Q' * result.Q - I) < 1e-6
        end
    end

    @testset "SOBI with different lag ranges" begin
        result_short = identify_sobi(model; lags=1:3)
        @test result_short isa ICASVARResult{Float64}
        @test result_short.method == :sobi

        result_long = identify_sobi(model; lags=1:20)
        @test result_long isa ICASVARResult{Float64}
    end

    @testset "HSIC with explicit sigma" begin
        result = identify_hsic(model; sigma=2.0)
        @test result isa ICASVARResult{Float64}
        @test result.method == :hsic
    end

    @testset "Identification strength with jade and sobi" begin
        for method in [:jade, :sobi]
            result = test_identification_strength(model; method=method, n_bootstrap=(FAST ? 9 : 19))
            @test result isa IdentifiabilityTestResult{Float64}
            @test result.test_name == :identification_strength
        end
    end

    @testset "Gaussian vs non-Gaussian LR with other distributions" begin
        for dist in [:mixture_normal, :pml, :skew_normal]
            result = test_gaussian_vs_nongaussian(model; distribution=dist)
            @test result isa IdentifiabilityTestResult{Float64}
            @test result.statistic >= 0
            @test 0 <= result.pvalue <= 1
        end
    end

    @testset "Markov-switching with 3 regimes" begin
        result = identify_markov_switching(model; n_regimes=3, max_iter=(FAST ? 20 : 50))
        @test result isa MarkovSwitchingSVARResult{Float64}
        @test result.n_regimes == 3
        @test length(result.Sigma_regimes) == 3
        @test size(result.transition_matrix) == (3, 3)
    end

    @testset "External volatility with 3 regimes" begin
        regime3 = vcat(fill(1, 100), fill(2, 100), fill(3, 100))
        result = identify_external_volatility(model, regime3; regimes=3)
        @test result isa ExternalVolatilitySVARResult{Float64}
        @test length(result.Sigma_regimes) == 3
        @test length(result.Lambda) == 3
    end

    @testset "External volatility with small regime" begin
        # Regime 3 has very few observations (fallback to overall cov)
        regime_small = vcat(fill(1, 148), fill(2, 148), fill(3, 4))
        result = identify_external_volatility(model, regime_small; regimes=3)
        @test result isa ExternalVolatilitySVARResult{Float64}
    end

    @testset "Smooth transition edge cases" begin
        Random.seed!(99999)
        # Extreme transition variable (all same sign)
        s_edge = abs.(randn(n_obs)) .+ 5.0
        result = identify_smooth_transition(model, s_edge)
        @test result isa SmoothTransitionSVARResult{Float64}
        @test result.gamma > 0
    end

    @testset "GARCH max_iter=1" begin
        result = identify_garch(model; max_iter=1)
        @test result isa GARCHSVARResult{Float64}
        @test result.iterations == 1
    end

    @testset "4-variable scalability" begin
        Random.seed!(55555)
        Y4 = randn(200, 4)
        model4 = estimate_var(Y4, 1)
        ica4 = identify_fastica(model4)
        @test size(ica4.B0) == (4, 4)
        @test norm(ica4.Q' * ica4.Q - I) < 1e-4

        ml4 = identify_student_t(model4; max_iter=(FAST ? 30 : 100))
        @test size(ml4.B0) == (4, 4)
    end

    # =================================================================
    # Integration Tests: Non-Gaussian through FEVD / HD / BVAR / LP
    # =================================================================

    @testset "Non-Gaussian FEVD Integration" begin
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

    @testset "Non-Gaussian HD Integration" begin
        T_eff = n_obs - 2  # p=2
        for method in [:fastica, :student_t]
            hd_r = historical_decomposition(model, T_eff; method=method)
            @test hd_r isa MacroEconometricModels.HistoricalDecomposition
            @test verify_decomposition(hd_r)
            @test hd_r.method == method
        end
    end

    @testset "compute_Q new methods" begin
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

    @testset "Hetero-ID through irf/fevd/hd" begin
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

    @testset "BVAR Non-Gaussian Identification" begin
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

    @testset "Structural LP Non-Gaussian" begin
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
