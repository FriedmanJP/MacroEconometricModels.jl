# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

# Use MacroEconometricModels versions of StatsAPI functions
const fm_residuals = MacroEconometricModels.residuals
const fm_r2 = MacroEconometricModels.r2
const fm_predict = MacroEconometricModels.predict
const fm_nobs = MacroEconometricModels.nobs
const fm_dof = MacroEconometricModels.dof

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

# DGP-06 (#795): sine of the largest principal angle between the column spaces
# of A and B (0 = same subspace, 1 = orthogonal). PCA identifies loadings only
# up to rotation, so recovery is asserted on the subspace, never elementwise.
function _subspace_dist(A::AbstractMatrix, B::AbstractMatrix)
    QA = Matrix(qr(A).Q)
    QB = Matrix(qr(B).Q)
    size(QA, 2) == size(QB, 2) ||
        throw(DimensionMismatch("_subspace_dist needs equal column counts"))
    return opnorm((I - QA * QA') * QB)
end

# Shared stationary factor-transition matrices (dgp_dynamic_factors sizes r from A).
const _FM_A2 = [0.6 0.15; 0.1 0.5]
const _FM_A3 = [0.5 0.1 0.0; 0.05 0.5 0.1; 0.0 0.05 0.5]

@testset "Factor Model Tests" begin

    @testset "Basic Factor Model Estimation" begin
        rng = Random.MersenneTwister(123)
        # DGP-06: VAR(1) factors with known loadings instead of iid draws.
        T, N, r_true = 100, 20, 3
        d = dgp_dynamic_factors(rng; A=_FM_A3, N=N, T=T, idio_sd=0.3)
        X = d.X

        # Estimate factor model
        model = estimate_factors(X, r_true)

        @test model isa FactorModel
        @test size(model.factors) == (T, r_true)
        @test size(model.loadings) == (N, r_true)
        @test length(model.eigenvalues) == N
        @test model.r == r_true
        @test model.standardized == true

        # Factors should have reasonable magnitude (not exploding or collapsing)
        @test all(isfinite, model.factors)
        @test maximum(abs.(model.factors)) < 100  # Reasonable bound

        # Variance explained should sum to 1
        @test isapprox(sum(model.explained_variance), 1.0, atol=1e-10)

        # Cumulative variance should be increasing
        @test issorted(model.cumulative_variance)
        @test model.cumulative_variance[end] ≈ 1.0
    end

    @testset "PCA recovery on a dynamic-factor DGP (T=500)" begin
        # DGP-06 (#795): the white-noise smoke test becomes a truth assertion —
        # PCA recovers the loadings subspace of a VAR-factor DGP. Realized
        # distance ≈ 0.06 at seed 7; the FAST bound is deliberately loose for
        # cross-platform LAPACK variation, the full bound is the honest one.
        rng = Random.MersenneTwister(7)
        T, N, r_true = 500, 30, 3
        d = dgp_dynamic_factors(rng; A=_FM_A3, N=N, T=T)
        model = estimate_factors(d.X, r_true)
        Xs = vec(std(d.X, dims=1))
        # The estimator works in standardized space: truth loadings are Λ/σ.
        @test _subspace_dist(d.Lambda ./ Xs, model.loadings) < (FAST ? 0.3 : 0.15)
    end

    @testset "Factor Model without Standardization" begin
        rng = Random.MersenneTwister(234)
        T, N, r = 50, 10, 2
        X = dgp_dynamic_factors(rng; A=_FM_A2, N=N, T=T, idio_sd=0.5).X

        model = estimate_factors(X, r; standardize=false)

        @test model.standardized == false
        @test size(model.factors) == (T, r)
        @test size(model.loadings) == (N, r)
    end

    @testset "Prediction and Residuals" begin
        rng = Random.MersenneTwister(345)
        T, N, r = 80, 15, 3
        X = dgp_dynamic_factors(rng; A=_FM_A3, N=N, T=T, idio_sd=0.2).X

        model = estimate_factors(X, r)

        # Test prediction
        X_fitted = fm_predict(model)
        @test size(X_fitted) == (T, N)

        # Test residuals
        resid = fm_residuals(model)
        @test size(resid) == (T, N)

        # Residuals should be finite
        @test all(isfinite, resid)

        # Residuals should have reasonable magnitude (not exploding)
        @test maximum(abs.(resid)) < 10
    end

    @testset "R-squared Computation" begin
        rng = Random.MersenneTwister(456)
        T, N, r = 100, 10, 2
        # Low idiosyncratic noise for reasonable R².
        X = dgp_dynamic_factors(rng; A=_FM_A2, N=N, T=T, idio_sd=0.1).X

        model = estimate_factors(X, r)
        r2_vals = fm_r2(model)

        @test length(r2_vals) == N
        # R² should be bounded (allow small negative values due to numerical issues)
        @test all(r2_vals .>= -0.1)
        @test all(r2_vals .<= 1.1)
        # R² values should be finite
        @test all(isfinite, r2_vals)
    end

    @testset "Information Criteria" begin
        rng = Random.MersenneTwister(567)
        T, N = 100, 20
        r_true = 3
        X = dgp_dynamic_factors(rng; A=_FM_A3, N=N, T=T, idio_sd=0.3).X

        max_r = 8
        ic = ic_criteria(X, max_r)

        @test length(ic.IC1) == max_r
        @test length(ic.IC2) == max_r
        @test length(ic.IC3) == max_r

        @test 1 <= ic.r_IC1 <= max_r
        @test 1 <= ic.r_IC2 <= max_r
        @test 1 <= ic.r_IC3 <= max_r

        # IC should be finite
        @test all(isfinite.(ic.IC1))
        @test all(isfinite.(ic.IC2))
        @test all(isfinite.(ic.IC3))
    end

    @testset "Scree Plot Data" begin
        rng = Random.MersenneTwister(678)
        T, N, r = 100, 15, 5
        X = dgp_dynamic_factors(rng; A=0.5 * Matrix{Float64}(I, r, r), N=N, T=T).X

        model = estimate_factors(X, r)
        scree_data = scree_plot_data(model)

        @test length(scree_data.factors) == N
        @test length(scree_data.explained_variance) == N
        @test length(scree_data.cumulative_variance) == N

        # Cumulative variance should be monotonically increasing
        @test issorted(scree_data.cumulative_variance)

        # Last value should be 1
        @test scree_data.cumulative_variance[end] ≈ 1.0
    end

    @testset "StatsAPI Interface" begin
        rng = Random.MersenneTwister(789)
        T, N, r = 100, 12, 3
        X = dgp_dynamic_factors(rng; A=_FM_A3, N=N, T=T).X

        model = estimate_factors(X, r)

        # Test nobs
        @test fm_nobs(model) == T

        # Test dof
        df = fm_dof(model)
        @test df == N * r + T * r - r^2
        @test df > 0
    end

    @testset "Input Validation" begin
        rng = Random.MersenneTwister(890)
        T, N = 50, 10
        X = randn(rng, T, N)

        # Test invalid number of factors
        @test_throws ArgumentError estimate_factors(X, 0)
        @test_throws ArgumentError estimate_factors(X, N + 1)
        @test_throws ArgumentError estimate_factors(X, -1)

        # Test IC criteria with invalid max_factors
        @test_throws ArgumentError ic_criteria(X, 0)
        @test_throws ArgumentError ic_criteria(X, min(T, N) + 1)
    end

    @testset "Edge Cases" begin
        rng = Random.MersenneTwister(901)
        # Single factor
        T, N = 100, 10
        X = dgp_dynamic_factors(rng; A=reshape([0.5], 1, 1), N=N, T=T).X
        model = estimate_factors(X, 1)
        @test size(model.factors) == (T, 1)
        @test size(model.loadings) == (N, 1)

        # Maximum number of factors
        T, N = 50, 20
        X = randn(rng, T, N)
        r_max = min(T, N)
        model = estimate_factors(X, r_max)
        @test size(model.factors) == (T, r_max)
    end

    @testset "Constant Series Handling" begin
        rng = Random.MersenneTwister(12)
        T, N = 100, 10
        X = randn(rng, T, N)

        # Add a constant series
        X[:, 1] .= 5.0

        # Should not throw error due to zero variance
        model = estimate_factors(X, 2)
        @test model isa FactorModel
    end

    @testset "Explained Variance Properties" begin
        rng = Random.MersenneTwister(23)
        T, N, r = 100, 20, 5
        X = dgp_dynamic_factors(rng; A=0.5 * Matrix{Float64}(I, r, r), N=N, T=T).X

        model = estimate_factors(X, r)

        # First r factors should explain more variance than later ones
        @test model.explained_variance[1] >= model.explained_variance[r]

        # Explained variance should be in descending order (eigenvalues sorted)
        @test issorted(model.explained_variance[1:r], rev=true)

        # Cumulative variance at r should equal sum of first r explained variances
        @test model.cumulative_variance[r] ≈ sum(model.explained_variance[1:r])
    end

    @testset "Reconstruction Quality" begin
        rng = Random.MersenneTwister(34)
        T, N = 150, 15  # More observations for stability
        r_true = 3
        X = dgp_dynamic_factors(rng; A=_FM_A3, N=N, T=T, idio_sd=0.1).X

        model = estimate_factors(X, r_true)
        X_fitted = fm_predict(model)

        # Fitted values should be finite
        @test all(isfinite, X_fitted)
        @test size(X_fitted) == size(X)

        # R² should be computed without errors
        r2_vals = fm_r2(model)
        @test length(r2_vals) == N
        @test all(isfinite, r2_vals)
    end

    @testset "Type Stability" begin
        rng = Random.MersenneTwister(45)
        T, N, r = 50, 10, 2

        # Float64
        X64 = randn(rng, Float64, T, N)
        model64 = estimate_factors(X64, r)
        @test eltype(model64.factors) == Float64
        @test eltype(model64.loadings) == Float64

        # Float32
        X32 = randn(rng, Float32, T, N)
        model32 = estimate_factors(X32, r)
        @test eltype(model32.factors) == Float32
        @test eltype(model32.loadings) == Float32
    end

    @testset "Integer Input Conversion" begin
        rng = Random.MersenneTwister(56)
        T, N, r = 50, 10, 2
        X_int = rand(rng, 1:10, T, N)

        model = estimate_factors(X_int, r)
        @test model isa FactorModel{Float64}
        @test all(isfinite, model.factors)
    end

    @testset "F-06 regression: PCA reconstruction == projection & correct Bai-Ng" begin
        # Reconstruction F·Λ' must equal the true PCA projection X·Vᵣ·Vᵣ', and
        # factors must be unit-variance (F'F/T = I). Mis-scaled factors (the F-06 bug)
        # break predict/residuals/r2 and make ic_criteria pick the wrong factor count.
        rng = Random.MersenneTwister(20260623)
        T, N, rtrue = 200, 30, 3
        F0 = randn(rng, T, rtrue); Λ0 = randn(rng, N, rtrue)
        X = F0 * Λ0' + randn(rng, T, N)

        m = estimate_factors(X, rtrue; standardize=false)
        # projection onto the top-rtrue principal directions
        ev = eigen(Symmetric(X'X / T)); idx = sortperm(ev.values; rev=true)
        Vr = ev.vectors[:, idx[1:rtrue]]
        proj = X * Vr * Vr'
        @test isapprox(fm_predict(m), proj; rtol=1e-8, atol=1e-8)
        # factors are unit-variance: F'F/T ≈ I
        @test isapprox(m.factors' * m.factors / T, Matrix(I, rtrue, rtrue); atol=1e-8)
        # Bai-Ng recovers the true number of factors on a clear factor structure
        ic = ic_criteria(X, 8; standardize=false)
        @test ic.r_IC1 == rtrue
        @test ic.r_IC2 == rtrue
        @test ic.r_IC3 == rtrue
    end

end
