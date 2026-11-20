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

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "Generalized Dynamic Factor Model" begin

    # ==========================================================================
    # Basic Estimation Tests
    # ==========================================================================

    @testset "Basic GDFM Estimation" begin
        # DGP-06: shared dynamic-factor simulator (was: bespoke iid loop).
        rng = Random.Xoshiro(12345)
        T_obs, N, q = 200, 20, 2

        # Generate simple factor data
        X = dgp_dynamic_factors(rng; N=N, T=T_obs, idio_sd=0.3).X

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
        # DGP-06: shared simulator; the true common component comes from the
        # simulator's factors/loadings (was: bespoke iid loop).
        rng = Random.Xoshiro(1959)
        T_obs, N = 120, 6
        # O(1) loadings keep the strong-signal regime the 0.9 bound was
        # written for (the simulator default targets a 0.7 common share).
        Lambda = randn(rng, N, 2)
        d = dgp_dynamic_factors(rng; Lambda=Lambda, N=N, T=T_obs,
                                idio_sd=0.1)
        common_true = d.F * d.Lambda'
        X = d.X
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
        rng = Random.Xoshiro(23456)
        T_obs, N, q = 150, 15, 2
        X = randn(rng, T_obs, N)

        for kernel in [:bartlett, :parzen, :tukey]
            model = estimate_gdfm(X, q; kernel=kernel)
            @test model isa GeneralizedDynamicFactorModel
            @test model.kernel == kernel
        end
    end

    @testset "Standardization Options" begin
        rng = Random.Xoshiro(34567)
        T_obs, N, q = 100, 10, 1

        # Create data with different scales
        X = randn(rng, T_obs, N)
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
        rng = Random.Xoshiro(45678)
        T_obs, N, q = 120, 12, 1
        X = randn(rng, T_obs, N)

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
        # DGP-06: shared simulator (was: bespoke iid loop).
        rng = Random.Xoshiro(56789)
        T_obs, N = 300, 30
        q = 1

        # Generate single factor data with clear structure (low noise)
        X = dgp_dynamic_factors(rng; A=reshape([0.5], 1, 1), N=N, T=T_obs,
                                idio_sd=0.2).X

        model = estimate_gdfm(X, q)

        # Common component should capture most variance
        var_common = var(vec(model.common_component))
        var_total = var(vec(X))
        @test var_common / var_total > 0.5  # At least 50% variance explained
    end

    @testset "Multiple Factor Recovery" begin
        # DGP-06: shared simulator (was: bespoke iid loop).
        rng = Random.Xoshiro(67890)
        T_obs, N = 400, 40
        q_true = 3

        # Generate multi-factor data
        A3 = [0.5 0.1 0.0; 0.05 0.5 0.1; 0.0 0.05 0.5]
        X = dgp_dynamic_factors(rng; A=A3, N=N, T=T_obs, idio_sd=0.25).X

        model = estimate_gdfm(X, q_true)

        # Check variance explained increases with factors
        @test length(model.variance_explained) == q_true
        @test all(model.variance_explained .> 0)

        # R² should be reasonable for most variables
        r2_vals = r2(model)
        @test mean(r2_vals) > 0.3  # Average R² > 30%
    end

    @testset "Dynamic Factor Structure" begin
        # DGP-06: shared simulator (was: bespoke AR loop).
        rng = Random.Xoshiro(78901)
        T_obs, N = 300, 25
        q = 2

        # Generate factors with AR dynamics
        X = dgp_dynamic_factors(rng; A=[0.7 0.0; 0.0 0.7], N=N, T=T_obs,
                                idio_sd=0.3).X

        model = estimate_gdfm(X, q)

        # Spectral loadings should vary across frequencies for dynamic factors
        @test size(model.loadings_spectral, 3) > 1  # Multiple frequencies
        @test any(!iszero, model.loadings_spectral)
    end

    # ==========================================================================
    # Spectral Density Tests
    # ==========================================================================

    @testset "Spectral Density Properties" begin
        rng = Random.Xoshiro(89012)
        T_obs, N, q = 128, 16, 2
        X = randn(rng, T_obs, N)

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
        rng = Random.Xoshiro(90123)
        T_obs, N, q = 100, 15, 3
        X = randn(rng, T_obs, N)

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
        rng = Random.Xoshiro(12345)
        T_obs, N, q = 100, 10, 2
        X = randn(rng, T_obs, N)

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
        # DGP-06: shared simulator for the strong panel; the weak panel scales
        # the same common component down against unit noise (was: bespoke loop).
        rng = Random.Xoshiro(23456)
        T_obs, N, q = 150, 12, 2

        # Strong factor structure
        d = dgp_dynamic_factors(rng; N=N, T=T_obs, idio_sd=0.1)
        X_strong = d.X

        # Weak factor structure
        X_weak = 0.1 * (d.F * d.Lambda') + randn(rng, T_obs, N)

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
        rng = Random.Xoshiro(34567)
        T_obs, N = 200, 20
        max_q = 5
        X = randn(rng, T_obs, N)

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
        # DGP-06: shared simulator (was: bespoke iid loop).
        rng = Random.Xoshiro(45678)
        T_obs, N = 300, 30
        q_true = 2
        max_q = 5

        # Generate data with clear 2-factor structure
        X = dgp_dynamic_factors(rng; N=N, T=T_obs, idio_sd=0.2).X

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
        rng = Random.Xoshiro(56789)
        T_obs, N, q = 150, 15, 2
        h = 10

        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        fc = forecast(model, h; method=:ar)

        # Dimension checks
        @test size(fc.observables) == (h, N)
        @test size(fc.factors) == (h, q)

        # Forecasts should be finite
        @test all(isfinite, fc.observables)
        @test all(isfinite, fc.factors)
        # T159: white-noise input ⇒ AR forecasts stay near zero (observed 0.014/0.16).
        @test maximum(abs, fc.factors) < 0.05
        @test maximum(abs, fc.observables) < 0.5
    end

    @testset "Forecast Methods" begin
        rng = Random.Xoshiro(67890)
        T_obs, N, q = 120, 12, 2
        h = 5

        X = randn(rng, T_obs, N)
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
        # DGP-06: shared simulator (was: bespoke AR loop).
        rng = Random.Xoshiro(78901)
        T_obs, N, q = 200, 20, 2
        h = 12

        # Generate AR(1) factors
        phi = 0.8
        X = dgp_dynamic_factors(rng; A=[phi 0.0; 0.0 phi], N=N, T=T_obs,
                                idio_sd=0.3).X

        model = estimate_gdfm(X, q)
        fc = forecast(model, h)

        # Forecasts should decay toward zero for stationary factors
        factor_norm_start = norm(fc.factors[1, :])
        factor_norm_end = norm(fc.factors[end, :])
        # Not always true due to estimation uncertainty, so just check finiteness
        # T159: kept — the decay property is MC-flaky by design (see comment); #256 territory.
        @test isfinite(factor_norm_start)
        @test isfinite(factor_norm_end)
    end

    # ==========================================================================
    # Edge Cases
    # ==========================================================================

    @testset "Single Factor (q=1)" begin
        rng = Random.Xoshiro(89012)
        T_obs, N = 100, 15
        q = 1

        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        @test model.q == 1
        @test size(model.factors, 2) == 1
        @test length(model.variance_explained) == 1
    end

    @testset "Many Factors (q close to N)" begin
        rng = Random.Xoshiro(90123)
        T_obs, N = 100, 10
        q = N - 2  # Many factors

        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        @test model.q == q
        @test size(model.factors, 2) == q

        # With many factors, should explain reasonable variance
        # (random data won't have strong factor structure)
        r2_vals = r2(model)
        @test mean(r2_vals) > 0.3  # Relaxed threshold for random data
    end

    @testset "Short Time Series" begin
        rng = Random.Xoshiro(12345)
        T_obs = 50  # Short
        N, q = 10, 2

        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        @test size(model.factors) == (T_obs, q)
        @test all(isfinite, model.common_component)
        @test model.idiosyncratic == model.X .- model.common_component  # T159: X = C + E by construction (bit-exact)
    end

    @testset "Wide Panel (N > T)" begin
        rng = Random.Xoshiro(23456)
        T_obs = 25
        N = 50  # N > T
        q = 2

        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        @test size(model.X) == (T_obs, N)
        @test size(model.factors) == (T_obs, q)
    end

    @testset "Power of 2 Sample Size" begin
        rng = Random.Xoshiro(34567)
        T_obs = 256  # Power of 2 for efficient FFT
        N, q = 20, 3

        X = randn(rng, T_obs, N)
        model = estimate_gdfm(X, q)

        @test size(model.factors) == (T_obs, q)
    end

    # ==========================================================================
    # Numerical Stability Tests
    # ==========================================================================

    @testset "Near-Collinear Data" begin
        rng = Random.Xoshiro(45678)
        T_obs, N = 100, 10
        q = 2

        # Create nearly collinear variables
        X = randn(rng, T_obs, N)
        X[:, 2] = X[:, 1] + 1e-8 * randn(rng, T_obs)

        model = estimate_gdfm(X, q)
        @test all(isfinite, model.common_component)
        @test model.idiosyncratic == model.X .- model.common_component  # T159: X = C + E by construction
        # T159: kept — adversarial 1e-8-collinear input; no factor reference exists.
        @test all(isfinite, model.factors)
    end

    @testset "Extreme Scaling" begin
        rng = Random.Xoshiro(56789)
        T_obs, N, q = 100, 10, 2

        # Very large values
        X_large = 1e6 * randn(rng, T_obs, N)
        model_large = estimate_gdfm(X_large, q; standardize=true)
        @test all(isfinite, model_large.common_component)
        @test model_large.idiosyncratic == model_large.X .- model_large.common_component  # T159: X = C + E by construction

        # Very small values
        X_small = 1e-6 * randn(rng, T_obs, N)
        model_small = estimate_gdfm(X_small, q; standardize=true)
        @test all(isfinite, model_small.common_component)
        @test model_small.idiosyncratic == model_small.X .- model_small.common_component  # T159: X = C + E by construction
    end

    @testset "Mixed Scaling" begin
        rng = Random.Xoshiro(67890)
        T_obs, N, q = 100, 10, 2

        X = randn(rng, T_obs, N)
        X[:, 1] .*= 1e6
        X[:, end] .*= 1e-6

        model = estimate_gdfm(X, q; standardize=true)
        @test all(isfinite, model.common_component)
        @test model.idiosyncratic == model.X .- model.common_component  # T159: X = C + E by construction
        @test all(isfinite, r2(model))
        @test all(0 .<= r2(model) .<= 1)  # T159: R² shares live in [0,1] (observed [0.18, 0.99])
    end

    @testset "Constant Column" begin
        rng = Random.Xoshiro(78901)
        T_obs, N, q = 100, 10, 2

        X = randn(rng, T_obs, N)
        X[:, 3] .= 5.0  # Constant column

        model = estimate_gdfm(X, q; standardize=true)
        # Should handle constant column gracefully
        @test all(isfinite, model.common_component)
        @test model.idiosyncratic == model.X .- model.common_component  # T159: X = C + E by construction
    end

    # ==========================================================================
    # Input Validation Tests
    # ==========================================================================

    @testset "Input Validation" begin
        rng = Random.Xoshiro(89012)
        T_obs, N = 100, 10
        X = randn(rng, T_obs, N)

        # Invalid q
        @test_throws ArgumentError estimate_gdfm(X, 0)
        @test_throws ArgumentError estimate_gdfm(X, N + 1)

        # Invalid kernel
        @test_throws ArgumentError estimate_gdfm(X, 2; kernel=:invalid)

        # Invalid r < q
        @test_throws ArgumentError estimate_gdfm(X, 3; r=2)
    end

    @testset "IC Criteria Validation" begin
        rng = Random.Xoshiro(90123)
        T_obs, N = 100, 10
        X = randn(rng, T_obs, N)

        # Invalid max_q
        @test_throws ArgumentError ic_criteria_gdfm(X, 0)
        @test_throws ArgumentError ic_criteria_gdfm(X, N + 1)
    end

    @testset "Forecast Validation" begin
        rng = Random.Xoshiro(12345)
        T_obs, N, q = 100, 10, 2
        X = randn(rng, T_obs, N)
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
        # DGP-06: shared simulator (was: bespoke iid loop).
        rng = Random.Xoshiro(23456)
        T_obs, N, q = 150, 15, 2

        # Strong factor structure
        X = dgp_dynamic_factors(rng; N=N, T=T_obs, idio_sd=0.1).X

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
        rng = Random.Xoshiro(34567)
        T_obs, N, q = 100, 12, 2
        X = randn(rng, T_obs, N)

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
        rng = Random.Xoshiro(45678)
        T_obs, N, q = 120, 15, 2
        X = randn(rng, T_obs, N)

        model = estimate_gdfm(X, q)

        # X = chi + xi
        @test norm(model.X - (model.common_component + model.idiosyncratic)) < 1e-10

        # predict() returns common component
        @test predict(model) == model.common_component

        # residuals() returns idiosyncratic
        @test residuals(model) == model.idiosyncratic
    end

    @testset "Reproducibility" begin
        rng = Random.Xoshiro(56789)
        T_obs, N, q = 100, 10, 2
        X = randn(rng, T_obs, N)

        # Same data should give same results
        model1 = estimate_gdfm(X, q; bandwidth=5, kernel=:bartlett)
        model2 = estimate_gdfm(X, q; bandwidth=5, kernel=:bartlett)

        @test model1.factors ≈ model2.factors
        @test model1.common_component ≈ model2.common_component
    end

    @testset "Integer Matrix Input" begin
        rng = Random.Xoshiro(67890)
        T_obs, N, q = 100, 10, 2

        X_int = rand(rng, 1:10, T_obs, N)

        model = estimate_gdfm(X_int, q)
        @test model isa GeneralizedDynamicFactorModel{Float64}
        @test all(isfinite, model.common_component)
        @test model.idiosyncratic == model.X .- model.common_component  # T159: X = C + E by construction
    end

    # ==========================================================================
    # Asymptotic Behavior Tests
    # ==========================================================================

    @testset "Increasing Sample Size" begin
        # DGP-06: shared simulator with FIXED loadings across sample sizes, so
        # only T varies (was: bespoke nested truncation). Each span is drawn
        # fresh from the same DGP parameters.
        rng = Random.Xoshiro(78901)
        N, q = 20, 2
        sample_sizes = [100, 400]

        # Generate consistent DGP
        Lambda = randn(rng, N, q)

        r2_values = Float64[]

        for T_obs in sample_sizes
            X = dgp_dynamic_factors(rng; Lambda=Lambda, N=N, T=T_obs,
                                    idio_sd=0.3).X
            model = estimate_gdfm(X, q)
            push!(r2_values, mean(r2(model)))
        end

        # R² should generally improve or stay stable with more data
        # Allow for some variation due to randomness
        @test r2_values[end] > 0.3  # At least reasonable R² with largest sample
    end

    @testset "Increasing Panel Width" begin
        # DGP-06: shared simulator (was: bespoke iid loop per width).
        rng = Random.Xoshiro(89012)
        T_obs, q = 200, 2
        panel_widths = [10, 25]

        variance_explained_list = Float64[]

        for N in panel_widths
            X = dgp_dynamic_factors(rng; N=N, T=T_obs, idio_sd=0.3).X

            model = estimate_gdfm(X, q)
            push!(variance_explained_list, sum(model.variance_explained))
        end

        # Variance explained should remain reasonable regardless of N
        @test all(variance_explained_list .> 0.2)
    end

    @testset "TimeSeriesData varnames and NaN validation" begin
        rng = Random.Xoshiro(42)
        T_obs, N, q = 80, 8, 2
        X = randn(rng, T_obs, N)
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
        # DGP-06: shared simulator; truth common component from its factors
        # (was: bespoke AR loop).
        rng = Random.Xoshiro(11)
        T_obs, N, q = 120, 15, 2
        d = dgp_dynamic_factors(rng; A=[0.5 0.0; 0.0 0.5], N=N, T=T_obs,
                                idio_sd=0.3)
        common_true = d.F * d.Lambda'
        X = d.X
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
        rng = Random.Xoshiro(720)
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
        rng = Random.Xoshiro(721)
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
        # DGP-06: shared simulator with the same block loadings and AR matrix
        # (was: bespoke loop). The simulator burns in; the assertions are HD
        # identities plus loose orthogonality/variance-ratio bounds.
        rng = Random.Xoshiro(7291)
        T_obs, N, q = 400, 30, 2
        Φ = [0.8 0.0; 0.0 -0.5]
        Λ = zeros(N, q)
        Λ[1:15, 1] .= 1
        Λ[16:30, 2] .= 1
        Λ .+= 0.05 .* randn(rng, N, q)
        X = dgp_dynamic_factors(rng; A=Φ, Lambda=Λ, N=N, T=T_obs,
                                idio_sd=0.15).X
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
        # DGP-06: shared zero-dynamics simulator plus a nonzero level
        # (was: bespoke iid loop). The assertions are HD identities.
        rng = Random.Xoshiro(7293)
        T_obs, N, q = 120, 12, 2
        X = dgp_dynamic_factors(rng; A=zeros(q, q), N=N, T=T_obs,
                                idio_sd=0.2).X .+ 5
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
        rng = Random.Xoshiro(71901)
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
        n_rep, T_obs, N, q_true = (FAST ? 5 : 10), 500, 100, 2
        n_hl = 0
        n_bn = 0
        for i in 1:n_rep
            rng = Random.Xoshiro(71900 + i)
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
        @test n_hl >= (FAST ? 4 : 8)
        @test n_bn >= (FAST ? 4 : 8)
        rng = Random.Xoshiro(71999)
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
        rng = Random.Xoshiro(72101)
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

    @testset "one-sided FHLR factors: contemporaneous identity and two-sided leads" begin
        # DGP-06: shared simulator (was: bespoke AR loop). O(1) loadings keep
        # the strong-signal regime the 0.9 one-sided/two-sided bound needs.
        rng = Random.Xoshiro(721)
        T_obs, N, q = 500, 30, 2
        Lambda = randn(rng, N, q)
        X = dgp_dynamic_factors(rng; A=[0.7 0.0; 0.0 0.7], Lambda=Lambda,
                                N=N, T=T_obs, idio_sd=0.3).X
        m = estimate_gdfm(X, q; standardize=false)
        @test size(m.Z) == (N, q)
        @test size(m.factors_onesided) == (T_obs, q)
        @test m.factors_onesided ≈ X * m.Z atol=1e-8
        # One-sided and two-sided factors span the same space. Asserted as a
        # subspace distance (rotation/ordering-invariant): the old per-column
        # `all(cors .> 0.9)` was MC-fragile by construction — nearby streams
        # score anywhere in 0.75-1.0 under BOTH the old bespoke loop and this
        # DGP (column-order flips, not worse estimates). Calibrated: 0.13-0.38
        # over seeds 721-725; a broken estimator scores ≈ 1.
        Qo = Matrix(qr(m.factors_onesided).Q); Qt = Matrix(qr(m.factors).Q)
        @test opnorm((I - Qo * Qo') * Qt) < 0.5
        # Drop last 10: contemporaneous filter is unchanged on the common sample
        # (the defining one-sided property). Re-estimating Z from the truncated
        # sample is O(1/T); two-sided finite-lag factors move because of future
        # leads and boundary reflection.
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
        rng = Random.Xoshiro(7211)
        T_obs, N, q = 600, 30, 2
        u = randn(rng, T_obs, q)
        for t in 2:T_obs
            u[t, :] .+= 0.6 .* u[t-1, :]
        end
        Λ0 = randn(rng, N, q)
        Λ1 = randn(rng, N, q)
        X = u * Λ0'
        X[2:end, :] .+= u[1:end-1, :] * Λ1'
        X .+= 0.15 .* randn(rng, T_obs, N)
        n_origin = FAST ? 8 : 20
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

    @testset "Non-circular finite-lag filter: 20σ end-of-sample spike invariance (#893)" begin
        rng = Random.Xoshiro(89310)
        T_obs, N, q = 150, 12, 2
        X = dgp_dynamic_factors(rng; N=N, T=T_obs, idio_sd=0.2).X
        X_spike = copy(X)
        X_spike[end, :] .+= 20.0 .* vec(std(X; dims=1))

        for spec in (:lag_window, :smoothed_periodogram)
            m = estimate_gdfm(X, q; spectral=spec, standardize=false)
            n_freq = size(m.loadings_spectral, 3)
            M = MacroEconometricModels._gdfm_filter_halfwidth(T_obs, n_freq, spec, m.bandwidth)
            t_hi = T_obs - M - 1
            @test t_hi >= 1

            # Fixed-loadings time-domain common component reconstruction
            chi_base = MacroEconometricModels._reconstruct_time_domain(
                m.loadings_spectral, X, m.frequencies; spectral=spec, bandwidth=m.bandwidth)
            chi_spike = MacroEconometricModels._reconstruct_time_domain(
                m.loadings_spectral, X_spike, m.frequencies; spectral=spec, bandwidth=m.bandwidth)
            @test maximum(abs, chi_spike[1:t_hi, :] - chi_base[1:t_hi, :]) < 1e-12

            # Fixed-loadings factor filter kernels (unnormalized and normalized up to global std)
            B0, B_pos, B_neg = MacroEconometricModels._factor_filter_kernels(
                m.loadings_spectral, T_obs, spec, M)
            X_pad_b = MacroEconometricModels._reflect_pad(X, M)
            X_pad_s = MacroEconometricModels._reflect_pad(X_spike, M)
            F_raw_b = X * transpose(B0)
            F_raw_s = X_spike * transpose(B0)
            for k in 1:M
                F_raw_b .+= @view(X_pad_b[(M + 1 - k):(M + T_obs - k), :]) * transpose(B_pos[k])
                F_raw_b .+= @view(X_pad_b[(M + 1 + k):(M + T_obs + k), :]) * transpose(B_neg[k])
                F_raw_s .+= @view(X_pad_s[(M + 1 - k):(M + T_obs - k), :]) * transpose(B_pos[k])
                F_raw_s .+= @view(X_pad_s[(M + 1 + k):(M + T_obs + k), :]) * transpose(B_neg[k])
            end
            @test maximum(abs, F_raw_s[1:t_hi, :] - F_raw_b[1:t_hi, :]) < 1e-12

            F_norm_b = MacroEconometricModels._extract_time_domain_factors(
                X, m.loadings_spectral, m.frequencies; spectral=spec, bandwidth=m.bandwidth)
            F_norm_s = MacroEconometricModels._extract_time_domain_factors(
                X_spike, m.loadings_spectral, m.frequencies; spectral=spec, bandwidth=m.bandwidth)
            for j in 1:q
                s_j = dot(F_norm_b[1:t_hi, j], F_norm_s[1:t_hi, j]) /
                      dot(F_norm_b[1:t_hi, j], F_norm_b[1:t_hi, j])
                @test maximum(abs, F_norm_s[1:t_hi, j] .- s_j .* F_norm_b[1:t_hi, j]) < 1e-12
            end
        end
    end

    @testset "Dynamic-lag reconstruction & q == N exactness (#893)" begin
        rng = Random.Xoshiro(89311)
        T_obs = 500
        u = randn(rng, T_obs)
        X_dyn = hcat(u, vcat(0.0, u[1:end-1]))
        for spec in (:lag_window, :smoothed_periodogram)
            m = estimate_gdfm(X_dyn, 1; spectral=spec, bandwidth=10, standardize=false)
            rmse = sqrt(mean(abs2, m.common_component[15:end-15, :] - X_dyn[15:end-15, :]))
            @test rmse < 0.1
        end

        for T_exact in (80, 81)
            X_ex = randn(Random.Xoshiro(89312 + T_exact), T_exact, 5)
            for spec in (:lag_window, :smoothed_periodogram)
                m_full = estimate_gdfm(X_ex, 5; spectral=spec, standardize=false)
                @test maximum(abs, m_full.common_component - X_ex) < 1e-12
            end
        end
    end

    @testset "GDFM HD exactness (< 1e-10) for both spectral estimators (#893)" begin
        rng = Random.Xoshiro(89313)
        X = dgp_dynamic_factors(rng; N=12, T=150, idio_sd=0.2).X .+ 3.0
        for spec in (:lag_window, :smoothed_periodogram), std_flag in (true, false)
            gdfm = estimate_gdfm(X, 2; spectral=spec, standardize=std_flag)
            hd = historical_decomposition(gdfm)
            @test verify_decomposition(hd; tol=1e-10)
            chi_hat = dropdims(sum(hd.contributions[:, :, 1:2]; dims=3); dims=3)
            @test maximum(abs, chi_hat - gdfm.common_component) < 1e-10
            @test maximum(abs, dropdims(sum(hd.contributions; dims=3); dims=3) - gdfm.X) < 1e-10
        end
    end

    # =========================================================================
    # Band-Pass GDFM, Eurocoin Real-Time Indicator & Spectral FEVD (#897)
    # =========================================================================

    @testset "_gdfmbp_gamma_band & bandpass_gdfm partition additivity (#897)" begin
        rng = Random.Xoshiro(89701)
        X_even = dgp_dynamic_factors(rng; N=14, T=160, idio_sd=0.25).X .+ 2.5
        X_odd = X_even[1:159, :]

        for (X_case, label) in ((X_even, "even"), (X_odd, "odd"))
            for spec in (:lag_window, :smoothed_periodogram), std_flag in (true, false)
                gdfm = estimate_gdfm(X_case, 2; spectral=spec, standardize=std_flag)
                ω0 = π / 4
                bp_lo = bandpass_gdfm(gdfm, (0.0, ω0))
                bp_hi = bandpass_gdfm(gdfm, (ω0, π))

                @test bp_lo isa BandPassGDFMResult{Float64}
                @test size(bp_lo.common_twosided) == size(gdfm.X)
                @test size(bp_lo.common_onesided) == size(gdfm.X)
                @test length(bp_lo.variance_share) == size(gdfm.X, 2)
                @test bp_lo.band == (0.0, ω0)
                @test bp_lo.r == gdfm.r
                @test bp_lo.varnames == gdfm.varnames
                @test all(s -> 0.0 <= s <= 1.0, bp_lo.variance_share)
                @test all(s -> 0.0 <= s <= 1.0, bp_hi.variance_share)

                # Exact partition additivity: (0, ω₀) + (ω₀, π) == common_component
                @test maximum(abs, (bp_lo.common_twosided .+ bp_hi.common_twosided) .- gdfm.common_component) < 1e-11
                @test maximum(abs, (bp_lo.variance_share .+ bp_hi.variance_share) .- 1.0) < 1e-11
                if spec === :lag_window || label == "even"
                    @test maximum(abs, (bp_lo.common_onesided .+ bp_hi.common_onesided) .- gdfm.common_component_onesided) < 1e-11
                end
            end
        end

        # Direct _gdfmbp_gamma_band partition additivity & match with _gamma_from_spectrum
        gdfm_lw = estimate_gdfm(X_even, 2; spectral=:lag_window, standardize=true)
        for k in 0:2
            G_full = MacroEconometricModels._gdfmbp_gamma_band(
                gdfm_lw.spectral_density_chi, gdfm_lw.frequencies, k, 0.0, π)
            G_ref = MacroEconometricModels._gamma_from_spectrum(
                gdfm_lw.spectral_density_chi, gdfm_lw.frequencies, k)
            @test maximum(abs, G_full - G_ref) < 1e-11
            G_1 = MacroEconometricModels._gdfmbp_gamma_band(
                gdfm_lw.spectral_density_chi, gdfm_lw.frequencies, k, 0.0, π / 3)
            G_2 = MacroEconometricModels._gdfmbp_gamma_band(
                gdfm_lw.spectral_density_chi, gdfm_lw.frequencies, k, π / 3, π)
            @test maximum(abs, (G_1 + G_2) - G_full) < 1e-12
        end

        # Frequency selectivity on synthetic low + high frequency factor panel
        T_sel, N_sel = 240, 20
        t_grid = collect(1.0:T_sel)
        X_freq, f_slow = let rng = Random.Xoshiro(89702)
            fs = sin.(2π .* t_grid ./ 24.0) .+ 0.4 .* cos.(2π .* t_grid ./ 36.0)
            ff = sin.(2π .* t_grid ./ 3.0) .+ 0.3 .* cos.(2π .* t_grid ./ 2.5)
            L_slow = 0.8 .+ 0.4 .* rand(rng, N_sel)
            L_fast = (-0.5) .+ rand(rng, N_sel)
            L_fast[1] = 1.2
            Xf = fs * L_slow' .+ ff * L_fast' .+ 0.1 .* randn(rng, T_sel, N_sel)
            Xf, fs
        end
        gdfm_freq = estimate_gdfm(X_freq, 2; spectral=:lag_window, bandwidth=16)
        bp_low = bandpass_gdfm(gdfm_freq, (0.0, π / 6); r=3)
        bp_high = bandpass_gdfm(gdfm_freq, (π / 2, π); r=3)
        @test bp_low.r == 3
        @test cor(bp_low.common_twosided[25:end-25, 1], f_slow[25:end-25]) > 0.90
        @test mean(abs2, bp_low.common_twosided[:, 1]) > 0.1
        @test mean(bp_high.variance_share) > 0.1

        # Display, report, RSER-01 field constructor, Float32 & validation errors
        buf = IOBuffer()
        show(buf, bp_low)
        @test occursin("Band-Pass GDFM Decomposition", String(take!(buf)))
        report(bp_low)
        vals = ntuple(i -> getfield(bp_low, fieldnames(BandPassGDFMResult)[i]), fieldcount(BandPassGDFMResult))
        @test BandPassGDFMResult{Float64}(vals...).variance_share == bp_low.variance_share

        gdfm32 = estimate_gdfm(Float32.(X_even), 2)
        bp32 = bandpass_gdfm(gdfm32, (0.0f0, Float32(π / 4)))
        @test bp32 isa BandPassGDFMResult{Float32}

        @test_throws ArgumentError bandpass_gdfm(gdfm_lw, (-0.1, 1.0))
        @test_throws ArgumentError bandpass_gdfm(gdfm_lw, (1.0, 0.5))
        @test_throws ArgumentError bandpass_gdfm(gdfm_lw, (0.0, π + 0.1))
        @test_throws ArgumentError bandpass_gdfm(gdfm_lw, (0.0, 1.0); r=1)
        @test_throws ArgumentError bandpass_gdfm(gdfm_lw, (0.0, 1.0); r=100)
    end

    @testset "Eurocoin real-time indicator (Altissimo et al. 2010) (#897)" begin
        rng = Random.Xoshiro(89703)
        T_obs, N = 240, 25
        t_grid = collect(1.0:T_obs)

        # Factor 1: smooth medium/long-run cycle (ω ≈ 2π/20 < π/6)
        # Factor 2: high-frequency noise oscillation (ω ≈ 2π/3 > π/6)
        phi_low = 0.85
        u_low = randn(rng, T_obs)
        f_smooth = sin.(2π .* t_grid ./ 22.0) .+ 0.6 .* cos.(2π .* t_grid ./ 35.0)
        for t in 2:T_obs
            f_smooth[t] += 0.25 * (phi_low * (f_smooth[t - 1] - sin(2π * (t - 1) / 22.0)) + 0.3 * u_low[t])
        end
        f_hf = 1.4 .* sin.(2π .* t_grid ./ 3.0) .+ 0.7 .* cos.(2π .* t_grid ./ 2.5)

        lam_smooth = 0.7 .+ 0.6 .* rand(rng, N)
        lam_hf = randn(rng, N)
        lam_hf[1:2] .= [1.2, -1.0]
        idio = 0.2 .* randn(rng, T_obs, N)

        X_ec = f_smooth * lam_smooth' .+ f_hf * lam_hf' .+ idio .+ 1.5
        target_smooth_true = 1.0 .+ 1.2 .* f_smooth
        target_hf_true = 1.1 .* f_hf
        target_obs = target_smooth_true .+ target_hf_true .+ 0.15 .* randn(rng, T_obs)

        ec = eurocoin_indicator(X_ec, target_obs, 2; cutoff=π / 6)
        @test ec isa EurocoinResult{Float64}
        @test length(ec.indicator) == T_obs
        @test length(ec.target_lowpass) == T_obs
        @test length(ec.weights) == N
        @test ec.cutoff ≈ π / 6
        @test ec.q == 2
        @test ec.r == 2
        @test 0.0 <= ec.r2 <= 1.0
        @test length(ec) == T_obs
        @test ec[end] == ec.indicator[end]
        @test ec[1:5] == ec.indicator[1:5]

        # Exact linear contemporaneous representation from raw demeaned panel
        X_c = X_ec .- mean(X_ec; dims=1)
        @test maximum(abs, (X_c * ec.weights .+ mean(target_obs)) .- ec.indicator) < 1e-11

        # High tracking correlation (> 0.90) with both target_lowpass and true smooth component
        @test cor(ec.indicator, ec.target_lowpass) > 0.90
        @test cor(ec.indicator, target_smooth_true) > 0.90

        # End-of-sample high-frequency variance suppression > 85%
        end_idx = (T_obs - 35):T_obs
        hf_resid_ec = ec.indicator[end_idx] .- target_smooth_true[end_idx]
        hf_resid_raw = target_obs[end_idx] .- target_smooth_true[end_idx]
        suppression = 1.0 - var(hf_resid_ec) / var(hf_resid_raw)
        @test suppression > 0.85

        # Unstandardized option, custom static rank r, Float32, show/report, constructor
        ec_unstd = eurocoin_indicator(X_ec, target_obs, 2; cutoff=π / 6, r=4, standardize=false, kernel=:parzen)
        @test ec_unstd.r == 4
        @test cor(ec_unstd.indicator, ec_unstd.target_lowpass) > 0.90

        ec32 = eurocoin_indicator(Float32.(X_ec), Float32.(target_obs), 2; cutoff=Float32(π / 6))
        @test ec32 isa EurocoinResult{Float32}

        buf = IOBuffer()
        show(buf, ec)
        @test occursin("Eurocoin Real-Time Low-Frequency Indicator", String(take!(buf)))
        report(ec)
        vals = ntuple(i -> getfield(ec, fieldnames(EurocoinResult)[i]), fieldcount(EurocoinResult))
        @test EurocoinResult{Float64}(vals...).r2 == ec.r2

        # Validation checks
        @test_throws ArgumentError eurocoin_indicator(X_ec, target_obs[1:end-1], 2)
        @test_throws ArgumentError eurocoin_indicator(X_ec, target_obs, 0)
        @test_throws ArgumentError eurocoin_indicator(X_ec, target_obs, 2; cutoff=0.0)
        @test_throws ArgumentError eurocoin_indicator(X_ec, target_obs, 2; cutoff=π)
        @test_throws ArgumentError eurocoin_indicator(X_ec, target_obs, 2; r=1)
        @test_throws ArgumentError eurocoin_indicator(X_ec, target_obs, 2; kernel=:bad)
    end

    @testset "spectral_fevd across StructuralDFM methods & unit-sum invariants (#897)" begin
        rng = Random.Xoshiro(89704)
        A = [0.75 0.0; 0.0 0.2]
        sim = dgp_dynamic_factors(rng; A=A, N=18, T=180, idio_sd=0.35)
        gdfm = estimate_gdfm(sim.X, 2; r=3, spectral=:lag_window, bandwidth=8)

        sdfm_fglr = estimate_structural_dfm(gdfm; method=:fglr, r=3, p=2, H=24)
        sdfm_var = estimate_structural_dfm(gdfm; method=:gdfm_var, p=2, H=24)
        sdfm_fhlz = estimate_structural_dfm(gdfm; method=:fhlz, p=2, n_perm=6, H=24, seed=123)

        for (m, label) in ((sdfm_fglr, "fglr"), (sdfm_var, "gdfm_var"), (sdfm_fhlz, "fhlz"))
            s_no = spectral_fevd(m)
            @test s_no isa SpectralFEVDResult{Float64}
            @test s_no isa AbstractFEVD
            @test size(s_no.shares) == (18, 2)
            @test s_no.include_idiosyncratic == false
            @test length(s_no.shock_names) == 2
            @test all(>=(0.0), s_no.shares)
            @test all(abs.(sum(s_no.shares; dims=2) .- 1.0) .< 1e-11)
            @test point_estimate(s_no) == s_no.shares
            @test has_uncertainty(s_no) == false
            @test uncertainty_bounds(s_no) === nothing

            s_id = spectral_fevd(m, (2π / 32, 2π / 6); include_idiosyncratic=true)
            @test size(s_id.shares) == (18, 3)
            @test s_id.include_idiosyncratic == true
            @test s_id.shock_names[end] == "Idiosyncratic"
            @test all(>=(0.0), s_id.shares)
            @test all(>=(0.0), s_id.power_band)
            @test all(abs.(sum(s_id.shares; dims=2) .- 1.0) .< 1e-11)
        end

        # Standardized units match and band differences (low-freq vs high-freq)
        sdfm_std = estimate_structural_dfm(gdfm; method=:fglr, r=3, p=2, units=:standardized)
        s_lr = spectral_fevd(sdfm_std, (0.0, 2π / 24); include_idiosyncratic=true)
        s_hf = spectral_fevd(sdfm_std, (2π / 4, π); include_idiosyncratic=true)
        @test all(abs.(sum(s_lr.shares; dims=2) .- 1.0) .< 1e-11)
        @test all(abs.(sum(s_hf.shares; dims=2) .- 1.0) .< 1e-11)
        @test any(abs.(s_lr.shares .- s_hf.shares) .> 1e-3)

        # Display, report, RSER-01 positional constructor, Float32 & validation errors
        s_show = spectral_fevd(sdfm_fglr; include_idiosyncratic=true)
        buf = IOBuffer()
        show(buf, s_show)
        @test occursin("Spectral FEVD", String(take!(buf)))
        report(s_show)
        vals = ntuple(i -> getfield(s_show, fieldnames(SpectralFEVDResult)[i]), fieldcount(SpectralFEVDResult))
        @test SpectralFEVDResult{Float64}(vals...).shares == s_show.shares

        gdfm32 = estimate_gdfm(Float32.(sim.X), 2; r=3, bandwidth=8)
        fv64 = sdfm_fglr.factor_var
        fv32 = VARModel(Float32.(fv64.Y), fv64.p, Float32.(fv64.B), Float32.(fv64.U),
                        Float32.(fv64.Sigma), Float32(fv64.aic), Float32(fv64.bic),
                        Float32(fv64.hqic), fv64.varnames)
        sdfm32 = StructuralDFM{Float32}(
            gdfm32, fv32, Float32.(sdfm_fglr.B0), Float32.(sdfm_fglr.Q),
            sdfm_fglr.identification, Float32.(sdfm_fglr.structural_irf),
            Float32.(sdfm_fglr.loadings_td), sdfm_fglr.p_var, sdfm_fglr.shock_names,
            sdfm_fglr.varnames, Float32.(sdfm_fglr.K), sdfm_fglr.r, sdfm_fglr.method,
            Float32.(sdfm_fglr.static_factors), Float32.(sdfm_fglr.loadings_static),
            Float32(sdfm_fglr.shock_variance_share), sdfm_fglr.units, nothing,
            Float32(sdfm_fglr.acceptance_rate), sdfm_fglr.id_order,
            sdfm_fglr.lag_criterion, Float32(sdfm_fglr.max_eigenvalue_modulus),
            nothing, Float32(sdfm_fglr.first_stage_F), nothing)
        s32 = spectral_fevd(sdfm32; include_idiosyncratic=true)
        @test s32 isa SpectralFEVDResult{Float32}
        @test all(abs.(sum(s32.shares; dims=2) .- 1.0f0) .< 1.0f-5)

        @test_throws ArgumentError spectral_fevd(sdfm_fglr, (-0.1, 1.0))
        @test_throws ArgumentError spectral_fevd(sdfm_fglr, (1.0, 1.0))
        @test_throws ArgumentError spectral_fevd(sdfm_fglr, (0.5, π + 0.1))
    end

    # ==========================================================================
    # Ragged-Edge EM-Spectral GDFM, Multi-Lag Projection & nowcast_gdfm (#899)
    # ==========================================================================

    @testset "EM-spectral GDFM missing-data & ragged-edge recovery (#899)" begin
        rng = Random.Xoshiro(89901)
        T_obs, N, q = 250, 40, 2
        d = dgp_dynamic_factors(rng; A=[0.7 0.1; 0.05 0.65], N=N, T=T_obs, idio_sd=0.3)
        X = d.X
        chi_true = d.F * d.Lambda'

        X_nan = copy(X)
        for t in 1:(T_obs - 1), j in 1:N
            rand(rng) < 0.10 && (X_nan[t, j] = NaN)
        end
        tail_idx = sort(randperm(rng, N)[1:round(Int, 0.30 * N)])
        X_nan[T_obs, tail_idx] .= NaN
        miss_mask = isnan.(X_nan)
        obs_mask = .!miss_mask

        # Default allow_missing=false rejects NaN and Inf
        @test_throws ArgumentError estimate_gdfm(X_nan, q)
        X_inf = copy(X); X_inf[5, 3] = Inf
        @test_throws ArgumentError estimate_gdfm(X_inf, q)
        @test_throws ArgumentError estimate_gdfm(X_inf, q; allow_missing=true)

        # All-NaN row/column and invalid EM keyword arguments
        X_bad_row = copy(X); X_bad_row[10, :] .= NaN
        @test_throws ArgumentError estimate_gdfm(X_bad_row, q; allow_missing=true)
        X_bad_col = copy(X); X_bad_col[:, 4] .= NaN
        @test_throws ArgumentError estimate_gdfm(X_bad_col, q; allow_missing=true)
        @test_throws ArgumentError estimate_gdfm(X_nan, q; allow_missing=true, em_max_iter=0)
        @test_throws ArgumentError estimate_gdfm(X_nan, q; allow_missing=true, em_tol=0.0)
        @test_throws ArgumentError estimate_gdfm(X, q; n_lags_proj=-1)
        @test_throws ArgumentError estimate_gdfm(X, q; n_lags_proj=T_obs)

        m_em = estimate_gdfm(X_nan, q; allow_missing=true)
        @test m_em isa GeneralizedDynamicFactorModel{Float64}
        @test all(isfinite, m_em.X)
        @test all(isfinite, m_em.common_component)
        @test all(isfinite, m_em.common_component_onesided)
        @test all(isfinite, m_em.factors)
        @test m_em.X[obs_mask] ≈ X[obs_mask]
        @test m_em.common_component + m_em.idiosyncratic ≈ m_em.X atol=1e-10

        @test cor(m_em.common_component[miss_mask], chi_true[miss_mask]) > 0.88

        # Downstream methods work cleanly on EM-imputed GDFM without NaN
        fc_em = forecast(m_em, 4; method=:one_sided)
        @test all(isfinite, fc_em.observables)
        @test all(isfinite, fc_em.observables_se)
        bp_em = bandpass_gdfm(m_em, (0.0, π / 4))
        @test all(isfinite, bp_em.common_twosided)
        hd_em = historical_decomposition(m_em)
        @test verify_decomposition(hd_em; tol=1e-9)

        # Clean data with allow_missing=true matches allow_missing=false
        m_clean_a = estimate_gdfm(X, q; allow_missing=false)
        m_clean_b = estimate_gdfm(X, q; allow_missing=true)
        @test m_clean_a.common_component ≈ m_clean_b.common_component atol=1e-12
        @test m_clean_a.Z ≈ m_clean_b.Z atol=1e-12

        # Unstandardized allow_missing=true
        m_nostd = estimate_gdfm(X_nan, q; allow_missing=true, standardize=false)
        @test all(isfinite, m_nostd.X)
        @test cor(m_nostd.common_component[miss_mask], chi_true[miss_mask]) > 0.88
    end

    @testset "Multi-lag stacked FHLR projection (n_lags_proj) (#899)" begin
        rng = Random.Xoshiro(89902)
        T_obs, N, q = 200, 18, 2
        d = dgp_dynamic_factors(rng; A=[0.7 0.1; 0.05 0.65], N=N, T=T_obs, idio_sd=0.25)
        X = d.X

        m0 = estimate_gdfm(X, q; r=3, n_lags_proj=0)
        m0_def = estimate_gdfm(X, q; r=3)
        @test m0.Z ≈ m0_def.Z atol=1e-12
        @test size(m0.Z) == (N, 3)

        m2 = estimate_gdfm(X, q; r=3, n_lags_proj=2)
        @test size(m2.Z) == (N * 3, 3)
        @test size(m2.factors_onesided) == (T_obs, 3)
        @test size(m2.common_component_onesided) == (T_obs, N)
        @test all(isfinite, m2.Z)
        @test all(isfinite, m2.factors_onesided)
        @test all(isfinite, m2.common_component_onesided)

        # Stacked projection forecasts agree across stored vs on-the-fly n_lags_proj and methods
        fc2_stored = forecast(m2, 4; method=:one_sided, ci_method=:theoretical)
        fc2_onthefly = forecast(m0, 4; method=:one_sided, n_lags_proj=2, ci_method=:theoretical)
        fc2_spectral = forecast(m2, 4; method=:spectral, ci_method=:theoretical)
        @test fc2_stored.observables ≈ fc2_onthefly.observables atol=1e-10
        @test fc2_stored.observables_se ≈ fc2_onthefly.observables_se atol=1e-10
        @test fc2_stored.observables ≈ fc2_spectral.observables atol=1e-10
        @test all(isfinite, fc2_stored.observables)
        @test all(fc2_stored.observables_se .> 0)

        fc2_none = forecast(m2, 4; method=:one_sided, ci_method=:none)
        @test all(fc2_none.observables_lower .== 0)
        @test fc2_none.observables ≈ fc2_stored.observables atol=1e-12

        fc2_boot = forecast(m2, 4; method=:one_sided, ci_method=:bootstrap, n_boot=30, seed=7)
        @test all(isfinite, fc2_boot.observables_se)
        @test all(fc2_boot.observables_upper .>= fc2_boot.observables_lower)

        # Smoothed periodogram with n_lags_proj=2
        m2_sp = estimate_gdfm(X, q; spectral=:smoothed_periodogram, n_lags_proj=2)
        @test size(m2_sp.Z) == (N * 3, q)
        @test all(isfinite, forecast(m2_sp, 3; method=:one_sided).observables)
        @test_throws ArgumentError forecast(m0, 3; method=:one_sided, n_lags_proj=-1)
    end

    @testset "nowcast_gdfm Wiener-Kolmogorov nowcasting, RSER-01, Float32 & display (#899)" begin
        rng = Random.Xoshiro(89903)
        T_obs, N, q = 250, 40, 2
        d = dgp_dynamic_factors(rng; A=[0.7 0.1; 0.05 0.65], N=N, T=T_obs, idio_sd=0.3)
        X = d.X
        names = ["v$i" for i in 1:N]

        X_nan = copy(X)
        for t in 1:(T_obs - 1), j in 1:N
            rand(rng) < 0.10 && (X_nan[t, j] = NaN)
        end
        tail_idx = sort(randperm(rng, N)[1:round(Int, 0.30 * N)])
        X_nan[T_obs, tail_idx] .= NaN

        gdfm_clean = estimate_gdfm(X, q; varnames=names)
        nc = nowcast_gdfm(gdfm_clean, X_nan, 0)
        @test nc isa GDFMNowcastResult{Float64}
        @test fieldnames(GDFMNowcastResult) == (
            :nowcast, :se, :panel_imputed, :horizon, :n_lags, :observed_mask, :varnames)
        @test length(nc.nowcast) == N
        @test length(nc.se) == N
        @test size(nc.panel_imputed) == size(X_nan)
        @test nc.horizon == 0
        @test nc.n_lags == 2
        @test nc.observed_mask == BitMatrix(.!isnan.(X_nan))
        @test nc.varnames == names
        @test all(isfinite, nc.nowcast)
        @test all(>(0.0), nc.se)
        @test all(isfinite, nc.panel_imputed)
        @test nc.panel_imputed[nc.observed_mask] == X_nan[nc.observed_mask]

        # Lower RMSE on masked tail entries across rolling origins than univariate AR(2) benchmark
        se_nc_sq = Float64[]
        se_ar2_sq = Float64[]
        for t_eval in (T_obs - 11):T_obs
            X_eval = copy(X_nan[1:t_eval, :])
            X_eval[t_eval, tail_idx] .= NaN
            nc_eval = nowcast_gdfm(gdfm_clean, X_eval, 0)
            for (idx_pos, j) in enumerate(tail_idx)
                push!(se_nc_sq, abs2(nc_eval.nowcast[j] - X[t_eval, j]))
                y = X_eval[1:(t_eval - 1), j]
                medj = median(filter(!isnan, y))
                yc = [isnan(v) ? medj : v for v in y]
                Y_t = yc[3:end]
                Z_t = hcat(ones(length(Y_t)), yc[2:(end - 1)], yc[1:(end - 2)])
                b = Z_t \ Y_t
                push!(se_ar2_sq, abs2((b[1] + b[2] * yc[end] + b[3] * yc[end - 1]) - X[t_eval, j]))
            end
        end
        rmse_nc = sqrt(mean(se_nc_sq))
        rmse_ar2 = sqrt(mean(se_ar2_sq))
        @test rmse_nc < rmse_ar2

        # Multi-horizon (h > 0 forecast, h < 0 backcast), unstandardized, smoothed_periodogram
        nc_h1 = nowcast_gdfm(gdfm_clean, X_nan, 1; n_lags=3)
        nc_back = nowcast_gdfm(gdfm_clean, X_nan, -1; n_lags=2)
        @test nc_h1.horizon == 1 && nc_h1.n_lags == 3 && all(isfinite, nc_h1.nowcast)
        @test nc_back.horizon == -1 && all(isfinite, nc_back.nowcast)
        @test all(nc_h1.se .> 0) && all(nc_back.se .> 0)

        gdfm_nostd = estimate_gdfm(X, q; standardize=false, spectral=:smoothed_periodogram)
        nc_nostd = nowcast_gdfm(gdfm_nostd, X_nan, 0)
        @test all(isfinite, nc_nostd.nowcast) && all(>(0.0), nc_nostd.se)

        # Single-row ragged panel and all-NaN tail row edge cases
        nc_1row = nowcast_gdfm(gdfm_clean, X_nan[end:end, :], 0)
        @test size(nc_1row.panel_imputed) == (1, N) && all(isfinite, nc_1row.nowcast)
        X_allnan_tail = copy(X_nan)
        X_allnan_tail[end, :] .= NaN
        nc_allnan = nowcast_gdfm(gdfm_clean, X_allnan_tail, 0; n_lags=2)
        @test all(isfinite, nc_allnan.nowcast) && all(isfinite, nc_allnan.panel_imputed)

        # Display, report, RSER-01 _generic_construct & Float32
        buf = IOBuffer()
        show(buf, nc)
        @test occursin("GDFM Ragged-Edge Nowcast", String(take!(buf)))
        redirect_stdout(devnull) do
            report(nc)
        end
        vals = ntuple(i -> getfield(nc, fieldnames(GDFMNowcastResult)[i]), fieldcount(GDFMNowcastResult))
        @test GDFMNowcastResult{Float64}(vals...).nowcast == nc.nowcast
        @test MacroEconometricModels._generic_construct(GDFMNowcastResult, Any[vals...]) isa GDFMNowcastResult{Float64}

        gdfm32 = estimate_gdfm(Float32.(X_nan), q; allow_missing=true, n_lags_proj=2)
        @test gdfm32 isa GeneralizedDynamicFactorModel{Float32}
        @test size(gdfm32.Z) == (N * 3, q)
        nc32 = nowcast_gdfm(gdfm32, Float32.(X_nan), 0)
        @test nc32 isa GDFMNowcastResult{Float32}
        @test eltype(nc32.nowcast) === Float32
        @test eltype(nc32.se) === Float32
        @test eltype(nc32.panel_imputed) === Float32
        vals32 = ntuple(i -> getfield(nc32, fieldnames(GDFMNowcastResult)[i]), fieldcount(GDFMNowcastResult))
        @test MacroEconometricModels._generic_construct(GDFMNowcastResult, Any[vals32...]) isa GDFMNowcastResult{Float32}

        # Validation errors
        X_bad_inf = copy(X_nan); X_bad_inf[2, 2] = -Inf
        @test_throws ArgumentError nowcast_gdfm(gdfm_clean, X_bad_inf, 0)
        @test_throws ArgumentError nowcast_gdfm(gdfm_clean, X_nan[:, 1:(N - 1)], 0)
        @test_throws ArgumentError nowcast_gdfm(gdfm_clean, zeros(0, N), 0)
        @test_throws ArgumentError nowcast_gdfm(gdfm_clean, X_nan, 0; n_lags=-1)
    end

end

# Run all tests
_tprint("Running GDFM tests...")
