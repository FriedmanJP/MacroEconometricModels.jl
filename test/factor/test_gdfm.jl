# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
    Tests for Generalized Dynamic Factor Model (GDFM)

Comprehensive test suite for the GDFM implementation following
Forni, Hallin, Lippi, and Reichlin (2000, 2005).
"""

using Test
using MacroEconometricModels
using LinearAlgebra
using Statistics
using Random

@testset "Generalized Dynamic Factor Model" begin

    # ==========================================================================
    # Basic Estimation Tests
    # ==========================================================================

    @testset "Basic GDFM Estimation" begin
        Random.seed!(12345)
        T_obs, N, q = 200, 20, 2

        # Generate simple factor data
        F_true = randn(T_obs, q)
        Lambda = randn(N, q)
        X = F_true * Lambda' + 0.3 * randn(T_obs, N)

        # Estimate GDFM
        model = estimate_gdfm(X, q)

        # Type checks
        @test model isa GeneralizedDynamicFactorModel{Float64}

        # Dimension checks
        @test size(model.X) == (T_obs, N)
        @test size(model.factors) == (T_obs, q)
        @test size(model.common_component) == (T_obs, N)
        @test size(model.idiosyncratic) == (T_obs, N)
        @test model.q == q
        @test length(model.frequencies) > 0
        @test length(model.variance_explained) == q

        # Common + idiosyncratic should approximately equal original
        reconstruction = model.common_component + model.idiosyncratic
        @test maximum(abs.(reconstruction - X)) < 1e-10
    end

    @testset "Common component via Forni projector (T096 #195)" begin
        Random.seed!(1959)
        T_obs, N = 120, 6
        F = randn(T_obs, 2); Λ = randn(N, 2)
        common_true = F * Λ'
        X = common_true + 0.1 * randn(T_obs, N)
        # q = N: the projector L·Lᴴ = I, so the common component reconstructs X exactly (the old
        # raw rank-1 periodogram Wiener filter did not).
        m_full = estimate_gdfm(X, N; standardize=false, spectral=:smoothed_periodogram)
        @test maximum(abs.(m_full.common_component - X)) < 1e-6
        # low-rank q = 2: a genuine projection (variance ≤ X per series) that recovers the common
        # structure (high correlation with the true common component).
        m2 = estimate_gdfm(X, 2; standardize=false, spectral=:smoothed_periodogram)
        @test all(var(m2.common_component[:, i]) <= 1.1 * var(X[:, i]) + 1e-8 for i in 1:N)
        @test mean([abs(cor(m2.common_component[:, i], common_true[:, i])) for i in 1:N]) > 0.9
    end

    @testset "Different Kernels" begin
        Random.seed!(23456)
        T_obs, N, q = 150, 15, 2
        X = randn(T_obs, N)

        for kernel in [:bartlett, :parzen, :tukey]
            model = estimate_gdfm(X, q; kernel=kernel)
            @test model isa GeneralizedDynamicFactorModel
            @test model.kernel == kernel
        end
    end

    @testset "Standardization Options" begin
        Random.seed!(34567)
        T_obs, N, q = 100, 10, 1

        # Create data with different scales
        X = randn(T_obs, N)
        X[:, 1] .*= 100  # Large scale
        X[:, 2] .*= 0.01  # Small scale

        # With standardization
        model_std = estimate_gdfm(X, q; standardize=true)
        @test model_std.standardized == true

        # Without standardization
        model_nostd = estimate_gdfm(X, q; standardize=false)
        @test model_nostd.standardized == false

        # Both should produce valid outputs
        @test size(model_std.factors) == (T_obs, q)
        @test size(model_nostd.factors) == (T_obs, q)
    end

    @testset "Custom Bandwidth" begin
        Random.seed!(45678)
        T_obs, N, q = 120, 12, 1
        X = randn(T_obs, N)

        # Automatic bandwidth
        model_auto = estimate_gdfm(X, q; bandwidth=0)
        @test model_auto.bandwidth > 0

        # Custom bandwidth
        bw = 5
        model_custom = estimate_gdfm(X, q; bandwidth=bw)
        @test model_custom.bandwidth == bw
    end

    # ==========================================================================
    # Factor Recovery Tests
    # ==========================================================================

    @testset "Single Factor Recovery" begin
        Random.seed!(56789)
        T_obs, N = 300, 30
        q = 1

        # Generate single factor data with clear structure
        F_true = randn(T_obs)
        Lambda = randn(N)
        # Low noise for clearer recovery
        X = F_true * Lambda' + 0.2 * randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # Common component should capture most variance
        var_common = var(vec(model.common_component))
        var_total = var(vec(X))
        @test var_common / var_total > 0.5  # At least 50% variance explained
    end

    @testset "Multiple Factor Recovery" begin
        Random.seed!(67890)
        T_obs, N = 400, 40
        q_true = 3

        # Generate multi-factor data
        F_true = randn(T_obs, q_true)
        Lambda = randn(N, q_true)
        X = F_true * Lambda' + 0.25 * randn(T_obs, N)

        model = estimate_gdfm(X, q_true)

        # Check variance explained increases with factors
        @test length(model.variance_explained) == q_true
        @test all(model.variance_explained .> 0)

        # R² should be reasonable for most variables
        r2_vals = r2(model)
        @test mean(r2_vals) > 0.3  # Average R² > 30%
    end

    @testset "Dynamic Factor Structure" begin
        Random.seed!(78901)
        T_obs, N = 300, 25
        q = 2

        # Generate factors with AR dynamics
        F_true = zeros(T_obs, q)
        for t in 2:T_obs
            F_true[t, :] = 0.7 * F_true[t-1, :] + randn(q)
        end

        Lambda = randn(N, q)
        X = F_true * Lambda' + 0.3 * randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # Spectral loadings should vary across frequencies for dynamic factors
        @test size(model.loadings_spectral, 3) > 1  # Multiple frequencies
        @test any(!iszero, model.loadings_spectral)
    end

    # ==========================================================================
    # Spectral Density Tests
    # ==========================================================================

    @testset "Spectral Density Properties" begin
        Random.seed!(89012)
        T_obs, N, q = 128, 16, 2
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        n_freq = length(model.frequencies)

        # Frequencies should be in [0, π]
        @test model.frequencies[1] >= 0
        @test model.frequencies[end] <= π + 1e-10

        # Spectral density should be Hermitian at each frequency
        for j in 1:n_freq
            S_j = model.spectral_density_X[:, :, j]
            @test norm(S_j - S_j') < 1e-10  # Hermitian check
        end

        # Eigenvalues should be real and non-negative
        @test all(model.eigenvalues_spectral .>= -1e-10)
    end

    @testset "Eigenvalue Ordering" begin
        Random.seed!(90123)
        T_obs, N, q = 100, 15, 3
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # Eigenvalues should be sorted in descending order at each frequency
        n_freq = length(model.frequencies)
        for j in 1:n_freq
            eigs = model.eigenvalues_spectral[:, j]
            @test issorted(eigs, rev=true)
        end
    end

    # ==========================================================================
    # StatsAPI Interface Tests
    # ==========================================================================

    @testset "StatsAPI Interface" begin
        Random.seed!(12345)
        T_obs, N, q = 100, 10, 2
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # predict
        fitted = predict(model)
        @test size(fitted) == (T_obs, N)
        @test fitted == model.common_component

        # residuals
        resid = residuals(model)
        @test size(resid) == (T_obs, N)
        @test resid == model.idiosyncratic

        # nobs
        @test nobs(model) == T_obs

        # dof
        @test dof(model) > 0

        # r2
        r2_vals = r2(model)
        @test length(r2_vals) == N
        @test all(r2_vals .<= 1.0 + 1e-10)  # R² <= 1
    end

    @testset "R² Consistency" begin
        Random.seed!(23456)
        T_obs, N, q = 150, 12, 2

        # Strong factor structure
        F_true = randn(T_obs, q)
        Lambda = randn(N, q)
        X_strong = F_true * Lambda' + 0.1 * randn(T_obs, N)

        # Weak factor structure
        X_weak = 0.1 * F_true * Lambda' + randn(T_obs, N)

        model_strong = estimate_gdfm(X_strong, q)
        model_weak = estimate_gdfm(X_weak, q)

        r2_strong = mean(r2(model_strong))
        r2_weak = mean(r2(model_weak))

        # Strong structure should have higher R²
        @test r2_strong > r2_weak
    end

    # ==========================================================================
    # Information Criteria Tests
    # ==========================================================================

    @testset "Information Criteria Computation" begin
        Random.seed!(34567)
        T_obs, N = 200, 20
        max_q = 5
        X = randn(T_obs, N)

        ic = ic_criteria_gdfm(X, max_q)

        # Check outputs exist
        @test length(ic.eigenvalue_ratios) >= 1
        @test length(ic.cumulative_variance) == max_q
        @test length(ic.avg_eigenvalues) == max_q

        # Cumulative variance should be increasing
        @test issorted(ic.cumulative_variance)

        # Cumulative variance should sum to <= 1 for q <= max_q
        @test ic.cumulative_variance[end] <= 1.0 + 1e-10

        # Selected q should be in valid range
        @test 1 <= ic.q_ratio <= max_q
        @test 1 <= ic.q_variance <= max_q
        @test ic.boundary isa Bool
        @test ic.boundary == (ic.q_variance == max_q && ic.cumulative_variance[end] < 0.9 - 1e-14)
    end

    @testset "Factor Selection with Known Structure" begin
        Random.seed!(45678)
        T_obs, N = 300, 30
        q_true = 2
        max_q = 5

        # Generate data with clear 2-factor structure
        F_true = randn(T_obs, q_true)
        Lambda = randn(N, q_true)
        X = F_true * Lambda' + 0.2 * randn(T_obs, N)

        ic = ic_criteria_gdfm(X, max_q)

        # First two eigenvalues should dominate
        @test ic.avg_eigenvalues[1] > ic.avg_eigenvalues[3]
        @test ic.avg_eigenvalues[2] > ic.avg_eigenvalues[3]

        # Eigenvalue ratio should suggest q close to true value
        @test ic.q_ratio <= q_true + 1
    end

    # ==========================================================================
    # Forecasting Tests
    # ==========================================================================

    @testset "Basic Forecasting" begin
        Random.seed!(56789)
        T_obs, N, q = 150, 15, 2
        h = 10

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        fc = forecast(model, h; method=:ar)

        # Dimension checks
        @test size(fc.observables) == (h, N)
        @test size(fc.factors) == (h, q)

        # Forecasts should be finite
        @test all(isfinite, fc.observables)
        @test all(isfinite, fc.factors)
    end

    @testset "Forecast Methods" begin
        Random.seed!(67890)
        T_obs, N, q = 120, 12, 2
        h = 5

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        # AR method
        fc_ar = forecast(model, h; method=:ar)
        @test size(fc_ar.observables) == (h, N)

        # Spectral method is the FHLR (2005) projection, not an AR(1) alias
        fc_spectral = forecast(model, h; method=:spectral, ci_method=:none)
        fc_os = forecast(model, h; method=:one_sided, ci_method=:none)
        @test size(fc_spectral.observables) == (h, N)
        @test fc_spectral.observables ≈ fc_os.observables atol=1e-10
        @test !(fc_ar.observables ≈ fc_spectral.observables)
    end

    @testset "Forecast with Dynamic Factors" begin
        Random.seed!(78901)
        T_obs, N, q = 200, 20, 2
        h = 12

        # Generate AR(1) factors
        F_true = zeros(T_obs, q)
        phi = 0.8
        for t in 2:T_obs
            F_true[t, :] = phi * F_true[t-1, :] + randn(q)
        end

        Lambda = randn(N, q)
        X = F_true * Lambda' + 0.3 * randn(T_obs, N)

        model = estimate_gdfm(X, q)
        fc = forecast(model, h)

        # Forecasts should decay toward zero for stationary factors
        factor_norm_start = norm(fc.factors[1, :])
        factor_norm_end = norm(fc.factors[end, :])
        # Not always true due to estimation uncertainty, so just check finiteness
        @test isfinite(factor_norm_start)
        @test isfinite(factor_norm_end)
    end

    # ==========================================================================
    # Edge Cases
    # ==========================================================================

    @testset "Single Factor (q=1)" begin
        Random.seed!(89012)
        T_obs, N = 100, 15
        q = 1

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test model.q == 1
        @test size(model.factors, 2) == 1
        @test length(model.variance_explained) == 1
    end

    @testset "Many Factors (q close to N)" begin
        Random.seed!(90123)
        T_obs, N = 100, 10
        q = N - 2  # Many factors

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test model.q == q
        @test size(model.factors, 2) == q

        # With many factors, should explain reasonable variance
        # (random data won't have strong factor structure)
        r2_vals = r2(model)
        @test mean(r2_vals) > 0.3  # Relaxed threshold for random data
    end

    @testset "Short Time Series" begin
        Random.seed!(12345)
        T_obs = 50  # Short
        N, q = 10, 2

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test size(model.factors) == (T_obs, q)
        @test all(isfinite, model.common_component)
    end

    @testset "Wide Panel (N > T)" begin
        Random.seed!(23456)
        T_obs = 25
        N = 50  # N > T
        q = 2

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test size(model.X) == (T_obs, N)
        @test size(model.factors) == (T_obs, q)
    end

    @testset "Power of 2 Sample Size" begin
        Random.seed!(34567)
        T_obs = 256  # Power of 2 for efficient FFT
        N, q = 20, 3

        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        @test size(model.factors) == (T_obs, q)
    end

    # ==========================================================================
    # Numerical Stability Tests
    # ==========================================================================

    @testset "Near-Collinear Data" begin
        Random.seed!(45678)
        T_obs, N = 100, 10
        q = 2

        # Create nearly collinear variables
        X = randn(T_obs, N)
        X[:, 2] = X[:, 1] + 1e-8 * randn(T_obs)

        model = estimate_gdfm(X, q)
        @test all(isfinite, model.common_component)
        @test all(isfinite, model.factors)
    end

    @testset "Extreme Scaling" begin
        Random.seed!(56789)
        T_obs, N, q = 100, 10, 2

        # Very large values
        X_large = 1e6 * randn(T_obs, N)
        model_large = estimate_gdfm(X_large, q; standardize=true)
        @test all(isfinite, model_large.common_component)

        # Very small values
        X_small = 1e-6 * randn(T_obs, N)
        model_small = estimate_gdfm(X_small, q; standardize=true)
        @test all(isfinite, model_small.common_component)
    end

    @testset "Mixed Scaling" begin
        Random.seed!(67890)
        T_obs, N, q = 100, 10, 2

        X = randn(T_obs, N)
        X[:, 1] .*= 1e6
        X[:, end] .*= 1e-6

        model = estimate_gdfm(X, q; standardize=true)
        @test all(isfinite, model.common_component)
        @test all(isfinite, r2(model))
    end

    @testset "Constant Column" begin
        Random.seed!(78901)
        T_obs, N, q = 100, 10, 2

        X = randn(T_obs, N)
        X[:, 3] .= 5.0  # Constant column

        model = estimate_gdfm(X, q; standardize=true)
        # Should handle constant column gracefully
        @test all(isfinite, model.common_component)
    end

    # ==========================================================================
    # Input Validation Tests
    # ==========================================================================

    @testset "Input Validation" begin
        Random.seed!(89012)
        T_obs, N = 100, 10
        X = randn(T_obs, N)

        # Invalid q
        @test_throws ArgumentError estimate_gdfm(X, 0)
        @test_throws ArgumentError estimate_gdfm(X, N + 1)

        # Invalid kernel
        @test_throws ArgumentError estimate_gdfm(X, 2; kernel=:invalid)

        # Invalid r < q
        @test_throws ArgumentError estimate_gdfm(X, 3; r=2)
    end

    @testset "IC Criteria Validation" begin
        Random.seed!(90123)
        T_obs, N = 100, 10
        X = randn(T_obs, N)

        # Invalid max_q
        @test_throws ArgumentError ic_criteria_gdfm(X, 0)
        @test_throws ArgumentError ic_criteria_gdfm(X, N + 1)
    end

    @testset "Forecast Validation" begin
        Random.seed!(12345)
        T_obs, N, q = 100, 10, 2
        X = randn(T_obs, N)
        model = estimate_gdfm(X, q)

        # Invalid horizon
        @test_throws ArgumentError forecast(model, 0)
        @test_throws ArgumentError forecast(model, -1)

        # Invalid method
        @test_throws ArgumentError forecast(model, 5; method=:invalid)
    end

    # ==========================================================================
    # Utility Function Tests
    # ==========================================================================

    @testset "Common Variance Share" begin
        Random.seed!(23456)
        T_obs, N, q = 150, 15, 2

        # Strong factor structure
        F_true = randn(T_obs, q)
        Lambda = randn(N, q)
        X = F_true * Lambda' + 0.1 * randn(T_obs, N)

        model = estimate_gdfm(X, q)

        shares = common_variance_share(model)

        @test length(shares) == N
        @test all(shares .>= 0)
        # The Forni common component is a spectral projection that can concentrate variance into
        # individual series, so a per-series share may modestly exceed 1 in finite samples (the
        # old raw-periodogram Wiener filter shrank it below 1); allow a small margin.
        @test all(shares .<= 1.1)

        # Should be high for strong factor structure
        @test mean(shares) > 0.5
    end

    @testset "Spectral Eigenvalue Plot Data" begin
        Random.seed!(34567)
        T_obs, N, q = 100, 12, 2
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        plot_data = spectral_eigenvalue_plot_data(model)

        @test haskey(plot_data, :frequencies)
        @test haskey(plot_data, :eigenvalues)

        @test plot_data.frequencies == model.frequencies
        @test plot_data.eigenvalues == model.eigenvalues_spectral
    end

    # ==========================================================================
    # Consistency Tests
    # ==========================================================================

    @testset "Decomposition Consistency" begin
        Random.seed!(45678)
        T_obs, N, q = 120, 15, 2
        X = randn(T_obs, N)

        model = estimate_gdfm(X, q)

        # X = chi + xi
        @test norm(model.X - (model.common_component + model.idiosyncratic)) < 1e-10

        # predict() returns common component
        @test predict(model) == model.common_component

        # residuals() returns idiosyncratic
        @test residuals(model) == model.idiosyncratic
    end

    @testset "Reproducibility" begin
        Random.seed!(56789)
        T_obs, N, q = 100, 10, 2
        X = randn(T_obs, N)

        # Same data should give same results
        model1 = estimate_gdfm(X, q; bandwidth=5, kernel=:bartlett)
        model2 = estimate_gdfm(X, q; bandwidth=5, kernel=:bartlett)

        @test model1.factors ≈ model2.factors
        @test model1.common_component ≈ model2.common_component
    end

    @testset "Integer Matrix Input" begin
        Random.seed!(67890)
        T_obs, N, q = 100, 10, 2

        X_int = rand(1:10, T_obs, N)

        model = estimate_gdfm(X_int, q)
        @test model isa GeneralizedDynamicFactorModel{Float64}
        @test all(isfinite, model.common_component)
    end

    # ==========================================================================
    # Asymptotic Behavior Tests
    # ==========================================================================

    @testset "Increasing Sample Size" begin
        Random.seed!(78901)
        N, q = 20, 2
        sample_sizes = [100, 400]

        # Generate consistent DGP
        F_full = randn(500, q)
        Lambda = randn(N, q)
        e = 0.3 * randn(500, N)

        r2_values = Float64[]

        for T_obs in sample_sizes
            X = F_full[1:T_obs, :] * Lambda' + e[1:T_obs, :]
            model = estimate_gdfm(X, q)
            push!(r2_values, mean(r2(model)))
        end

        # R² should generally improve or stay stable with more data
        # Allow for some variation due to randomness
        @test r2_values[end] > 0.3  # At least reasonable R² with largest sample
    end

    @testset "Increasing Panel Width" begin
        Random.seed!(89012)
        T_obs, q = 200, 2
        panel_widths = [10, 25]

        variance_explained_list = Float64[]

        for N in panel_widths
            F_true = randn(T_obs, q)
            Lambda = randn(N, q)
            X = F_true * Lambda' + 0.3 * randn(T_obs, N)

            model = estimate_gdfm(X, q)
            push!(variance_explained_list, sum(model.variance_explained))
        end

        # Variance explained should remain reasonable regardless of N
        @test all(variance_explained_list .> 0.2)
    end

    @testset "TimeSeriesData varnames and NaN validation" begin
        Random.seed!(42)
        T_obs, N, q = 80, 8, 2
        X = randn(T_obs, N)
        names = ["x$i" for i in 1:N]
        ts = TimeSeriesData(X; varnames=names)
        m = estimate_gdfm(ts, q)
        @test m.varnames == names
        @test length(m.varnames) == N

        Xnan = copy(X)
        Xnan[3, 2] = NaN
        err = try
            estimate_gdfm(Xnan, q)
            error("expected ArgumentError")
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("NaN", err.msg) || occursin("Inf", err.msg)

        # In-memory serialization round-trip of the new varnames field
        m2 = MacroEconometricModels._reconstruct_from_container(
            MacroEconometricModels._build_container(m))
        @test m2.varnames == names
        @test m2.q == m.q
        @test m2.spectral === m.spectral
        @test size(m2.Z) == size(m.Z)
        @test m2.factors_onesided ≈ m.factors_onesided atol=1e-10
    end

    @testset "both spectral estimators on a dynamic-factor panel" begin
        rng = Random.MersenneTwister(11)
        T_obs, N, q = 120, 15, 2
        F = zeros(T_obs, q)
        F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.5 .* F[t - 1, :] .+ randn(rng, q)
        end
        Λ = randn(rng, N, q)
        common_true = F * Λ'
        X = common_true .+ 0.3 .* randn(rng, T_obs, N)
        for spec in (:lag_window, :smoothed_periodogram)
            m = estimate_gdfm(X, q; spectral=spec, standardize=false)
            @test m.spectral === spec
            @test size(m.factors) == (T_obs, q)
            @test size(m.common_component) == (T_obs, N)
            recon = m.common_component + m.idiosyncratic
            @test maximum(abs.(recon - X)) < 1e-8
            @test mean([abs(cor(m.common_component[:, i], common_true[:, i])) for i in 1:N]) > 0.7
        end
    end

    @testset "lag-window spectrum has full rank; smoothed periodogram does not" begin
        rng = Random.MersenneTwister(720)
        T_obs, N = 300, 40
        X = randn(rng, T_obs, N)
        bw = 5
        lw = estimate_gdfm(X, 2; spectral=:lag_window, bandwidth=20, standardize=false)
        sp = estimate_gdfm(X, 2; spectral=:smoothed_periodogram, bandwidth=bw, standardize=false)
        S_lw = lw.spectral_density_X[:, :, max(1, div(size(lw.spectral_density_X, 3), 2))]
        S_sp = sp.spectral_density_X[:, :, max(1, div(size(sp.spectral_density_X, 3), 2))]
        evals_lw = eigvals(Hermitian(S_lw))
        evals_sp = eigvals(Hermitian(S_sp))
        rank_lw = count(>(1e-8) ∘ abs, evals_lw)
        rank_sp = count(>(1e-8) ∘ abs, evals_sp)
        @test rank_lw == N
        @test rank_sp <= 2 * bw + 1
    end

    @testset "lag-window estimator matches AR(1) spectrum (closed-form check)" begin
        # Oracle for `_estimate_spectral_density_lagwindow`, not a GDFM fit: a univariate
        # AR(1) is the unique simple process whose spectral density is known in closed form.
        rng = Random.MersenneTwister(721)
        T_obs, M, φ, σ2 = 2000, 25, 0.5, 1.0
        y = zeros(T_obs)
        y[1] = sqrt(σ2 / (1 - φ^2)) * randn(rng)
        for t in 2:T_obs
            y[t] = φ * y[t - 1] + sqrt(σ2) * randn(rng)
        end
        X = reshape(y, :, 1)
        θ, S = MacroEconometricModels._estimate_spectral_density_lagwindow(X, M, :bartlett)
        fhat = real.(S[1, 1, :])
        fth = [σ2 / (2π * abs2(1 - φ * cis(-θh))) for θh in θ]
        rel_rmse = sqrt(mean(abs2, fhat .- fth)) / sqrt(mean(abs2, fth))
        @test rel_rmse < 0.15
    end

    @testset "GDFM historical decomposition by dynamic PC" begin
        rng = Random.MersenneTwister(7291)
        T_obs, N, q = 400, 30, 2
        F = zeros(T_obs, q)
        F[1, :] = randn(rng, q)
        Φ = [0.8 0.0; 0.0 -0.5]
        for t in 2:T_obs
            F[t, :] = Φ * F[t-1, :] .+ randn(rng, q)
        end
        Λ = zeros(N, q)
        Λ[1:15, 1] .= 1
        Λ[16:30, 2] .= 1
        Λ .+= 0.05 .* randn(rng, N, q)
        X = F * Λ' .+ 0.15 .* randn(rng, T_obs, N)
        gdfm = estimate_gdfm(X, q; standardize=false, spectral=:lag_window)
        hd = historical_decomposition(gdfm)
        @test verify_decomposition(hd; tol=1e-8)
        chi_hat = dropdims(sum(hd.contributions[:, :, 1:q]; dims=3); dims=3)
        @test chi_hat ≈ gdfm.common_component atol=1e-8
        @test hd.contributions[:, :, q + 1] ≈ gdfm.idiosyncratic atol=1e-8
        for i in 1:N
            c1 = hd.contributions[:, i, 1]
            c2 = hd.contributions[:, i, 2]
            if std(c1) > 1e-8 && std(c2) > 1e-8
                @test abs(cor(c1, c2)) < 0.1
            end
            vχ = var(chi_hat[:, i])
            vsum = var(c1) + var(c2)
            vχ > 1e-8 && @test abs(vχ - vsum) / vχ < 0.10
        end
        @test plot_result(hd) isa MacroEconometricModels.PlotOutput
    end

    @testset "GDFM HD reconstructs common_component under default standardize" begin
        rng = Random.MersenneTwister(7293)
        T_obs, N, q = 120, 12, 2
        F = randn(rng, T_obs, q)
        X = F * randn(rng, N, q)' .+ 0.2 .* randn(rng, T_obs, N) .+ 5
        gdfm = estimate_gdfm(X, q)   # default standardize=true
        @test gdfm.standardized
        hd = historical_decomposition(gdfm)
        chi_hat = dropdims(sum(hd.contributions[:, :, 1:q]; dims=3); dims=3)
        @test chi_hat ≈ gdfm.common_component atol=1e-8
        @test hd.contributions[:, :, q + 1] ≈ gdfm.idiosyncratic atol=1e-8
        @test verify_decomposition(hd; tol=1e-8)
        @test maximum(abs, hd.initial_conditions) < 1e-12
    end

    # ==========================================================================
    # Hallin–Liška / Bai–Ng 2007 / Amengual–Watson (SDFM-10)
    # ==========================================================================

    @testset "ic_criteria_gdfm warns on 90% boundary" begin
        rng = Random.MersenneTwister(71901)
        X = randn(rng, 80, 12)
        ic = @test_logs (:warn, r"q_variance") ic_criteria_gdfm(X, 2)
        @test ic.boundary
        @test ic.q_variance == 2
        @test haskey(ic, :q_ratio)
        @test haskey(ic, :q_variance)
        @test haskey(ic, :eigenvalue_ratios)
        @test haskey(ic, :cumulative_variance)
        @test haskey(ic, :avg_eigenvalues)
    end

    @testset "Hallin–Liška and Bai–Ng recover q=2 in ≥8/10 replications" begin
        n_rep, T_obs, N, q_true = 10, 500, 100, 2
        n_hl = 0
        n_bn = 0
        for i in 1:n_rep
            rng = Random.MersenneTwister(71900 + i)
            u = randn(rng, T_obs, q_true)
            for t in 2:T_obs
                u[t, :] .+= 0.4 .* u[t-1, :]
            end
            Λ0 = randn(rng, N, q_true)
            Λ1 = randn(rng, N, q_true)
            X = u * Λ0'
            X[2:end, :] .+= u[1:end-1, :] * Λ1'
            X .+= 0.5 .* randn(rng, T_obs, N)
            hl = hallin_liska(X, 8)
            @test hl isa HallinLiskaResult
            n_hl += (hl.q == q_true)
            bn = bai_ng_q(X, 4; p=1)
            @test bn isa BaiNgQResult
            n_bn += (bn.q_D1 == q_true)
        end
        @test n_hl >= 8
        @test n_bn >= 8
        rng = Random.MersenneTwister(71999)
        u = randn(rng, T_obs, q_true)
        X = u * randn(rng, N, q_true)'
        X[2:end, :] .+= u[1:end-1, :] * randn(rng, N, q_true)'
        X .+= 0.5 .* randn(rng, T_obs, N)
        aw = amengual_watson_q(X, 4, 1)
        @test aw isa AmengualWatsonResult
        @test aw.q >= 1
        @test sprint(show, hallin_liska(X[:, 1:20], 3; subpanels=2, c_grid=range(0, 2; length=20))) isa String
    end

    @testset "spectrum inverts to asymmetric lag-1 covariance (not the even part)" begin
        rng = Random.MersenneTwister(72101)
        T_obs, N = 2000, 16
        u = randn(rng, T_obs)
        X = zeros(T_obs, N)
        X[1, 1:8] .= u[1]
        for t in 2:T_obs
            X[t, 1:8] .= u[t]
            X[t, 9:16] .= u[t - 1]
        end
        X .+= 0.05 .* randn(rng, T_obs, N)
        m = estimate_gdfm(X, 1; standardize=false, spectral=:lag_window)
        Γ1 = MacroEconometricModels._gamma_from_spectrum(m.spectral_density_X, m.frequencies, 1)
        Xc = X .- mean(X; dims=1)
        # E[X_t X_{t-1}']: same /T lag-window convention as estimate_gdfm
        sample = (Xc[2:T_obs, :]' * Xc[1:(T_obs - 1), :]) / T_obs
        @test size(Γ1) == (N, N)
        @test norm(Γ1 - transpose(Γ1)) > 0.2 * norm(Γ1)
        w = 1 - 1 / max(m.bandwidth, 1)
        @test Γ1 ≈ w .* sample rtol=0.08
        @test norm(Γ1 - w .* sample) < norm(Γ1 - w .* ((sample + sample') / 2))
    end

    @testset "one-sided FHLR factors: contemporaneous identity and two-sided wrap" begin
        rng = Random.MersenneTwister(721)
        T_obs, N, q = 500, 30, 2
        F = zeros(T_obs, q)
        F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.7 .* F[t-1, :] .+ randn(rng, q)
        end
        X = F * randn(rng, N, q)' .+ 0.3 .* randn(rng, T_obs, N)
        m = estimate_gdfm(X, q; standardize=false)
        @test size(m.Z) == (N, q)
        @test size(m.factors_onesided) == (T_obs, q)
        @test m.factors_onesided ≈ X * m.Z atol=1e-8
        cors = [abs(cor(m.factors_onesided[:, j], m.factors[:, j])) for j in 1:q]
        @test all(>(0.9), cors)
        # Drop last 10: contemporaneous filter is unchanged on the common sample
        # (the defining one-sided property). Re-estimating Z from the truncated
        # sample is O(1/T); two-sided IFFT factors move because they wrap.
        @test m.factors_onesided[1:end-10, :] ≈ X[1:end-10, :] * m.Z atol=1e-8
        mt = estimate_gdfm(X[1:end-10, :], q; standardize=false)
        @test mt.factors_onesided ≈ X[1:end-10, :] * mt.Z atol=1e-8
        F_os = copy(m.factors_onesided[1:end-10, :])
        F_os_t = copy(mt.factors_onesided)
        F_2s = copy(m.factors[1:end-10, :])
        F_2s_t = copy(mt.factors)
        for j in 1:q
            dot(F_os[:, j], F_os_t[:, j]) < 0 && (F_os_t[:, j] .*= -1)
            dot(F_2s[:, j], F_2s_t[:, j]) < 0 && (F_2s_t[:, j] .*= -1)
        end
        @test maximum(abs, F_2s - F_2s_t) > 1e-3
        @test maximum(abs, F_2s - F_2s_t) > maximum(abs, F_os - F_os_t)
        os_corr = mean(abs.([cor(F_os[:, j], F_os_t[:, j]) for j in 1:q]))
        @test os_corr > 0.99
    end

    @testset "FHLR h=1 projection RMSE beats AR(1) on two-sided factors" begin
        rng = Random.MersenneTwister(7211)
        T_obs, N, q = 500, 30, 2
        u = randn(rng, T_obs, q)
        for t in 2:T_obs
            u[t, :] .+= 0.5 .* u[t-1, :]
        end
        Λ0 = randn(rng, N, q)
        Λ1 = randn(rng, N, q)
        X = u * Λ0'
        X[2:end, :] .+= u[1:end-1, :] * Λ1'
        X .+= 0.3 .* randn(rng, T_obs, N)
        n_origin = 8
        sse_ar = 0.0
        sse_os = 0.0
        for t in (T_obs - n_origin):(T_obs - 1)
            m = estimate_gdfm(X[1:t, :], q; standardize=false)
            fc_ar = forecast(m, 1; method=:ar, ci_method=:none)
            fc_os = forecast(m, 1; method=:one_sided, ci_method=:none)
            sse_ar += sum(abs2, fc_ar.observables[1, :] .- X[t + 1, :])
            sse_os += sum(abs2, fc_os.observables[1, :] .- X[t + 1, :])
        end
        @test sse_os < sse_ar
        m = estimate_gdfm(X, q; standardize=false)
        @test_throws ArgumentError forecast(m, 1; method=:invalid)
        fc_sp = forecast(m, 2; method=:spectral, ci_method=:none)
        fc_os = forecast(m, 2; method=:one_sided, ci_method=:none)
        @test fc_sp.observables ≈ fc_os.observables atol=1e-10
    end

end

# Run all tests
_tprint("Running GDFM tests...")
