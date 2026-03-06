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

@testset "Structural DFM Tests" begin

    # =========================================================================
    # Shared test data generation
    # =========================================================================
    function make_sdfm_data(; T_obs=200, N=20, q=3, seed=42)
        rng = Random.MersenneTwister(seed)
        # Generate factor structure with some serial correlation
        F = zeros(T_obs, q)
        F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.5 * F[t-1, :] + randn(rng, q)
        end
        Lambda = randn(rng, N, q)
        noise = 0.3 * randn(rng, T_obs, N)
        X = F * Lambda' + noise
        return X, q
    end

    # =========================================================================
    # StructuralDFM Type Construction
    # =========================================================================

    @testset "StructuralDFM type construction" begin
        X, q = make_sdfm_data()
        sdfm = estimate_structural_dfm(X, q; p=1, H=20)

        @test sdfm isa StructuralDFM{Float64}
        @test sdfm.gdfm isa GeneralizedDynamicFactorModel{Float64}
        @test sdfm.factor_var isa VARModel{Float64}
        @test sdfm.identification == :cholesky
        @test sdfm.p_var == 1
        @test length(sdfm.shock_names) == q
    end

    # =========================================================================
    # One-Step Estimation (From Raw Data)
    # =========================================================================

    @testset "From raw data (one-step)" begin
        X, q = make_sdfm_data()
        T_obs, N = size(X)

        sdfm = estimate_structural_dfm(X, q; identification=:cholesky, p=2, H=30)

        @test sdfm.p_var == 2
        @test size(sdfm.structural_irf) == (30, N, q)
        @test size(sdfm.B0) == (q, q)
        @test size(sdfm.Q) == (q, q)
        @test size(sdfm.loadings_td) == (N, q)

        # Factor VAR should have q variables
        @test nvars(sdfm.factor_var) == q
        @test sdfm.factor_var.p == 2
    end

    # =========================================================================
    # Two-Step Estimation (From Existing GDFM)
    # =========================================================================

    @testset "From existing GDFM (two-step)" begin
        X, q = make_sdfm_data()

        # Step 1: Estimate GDFM separately
        gdfm = estimate_gdfm(X, q)
        @test gdfm isa GeneralizedDynamicFactorModel{Float64}

        # Step 2: Build Structural DFM on top
        sdfm = estimate_structural_dfm(gdfm; identification=:cholesky, p=1, H=25)

        @test sdfm.gdfm === gdfm  # Same reference
        @test size(sdfm.structural_irf, 1) == 25
        @test size(sdfm.structural_irf, 2) == size(X, 2)
        @test size(sdfm.structural_irf, 3) == q
    end

    # =========================================================================
    # Cholesky Identification
    # =========================================================================

    @testset "Cholesky identification (B0 lower-triangular)" begin
        X, q = make_sdfm_data()
        sdfm = estimate_structural_dfm(X, q; identification=:cholesky, p=1, H=20)

        # B0 = chol(Sigma) * Q. With Cholesky, Q=I, so B0 = chol(Sigma) = lower triangular
        B0 = sdfm.B0
        for i in 1:q
            for j in (i+1):q
                @test abs(B0[i, j]) < 1e-10
            end
        end

        # Q should be identity for Cholesky
        @test sdfm.Q ≈ Matrix{Float64}(I, q, q)
    end

    # =========================================================================
    # Sign Restrictions
    # =========================================================================

    @testset "Sign restrictions" begin
        X, q = make_sdfm_data(; q=2)

        # Define a sign check: first shock has positive impact on first factor at h=1
        sign_check = irf_result -> irf_result[1, 1, 1] > 0

        sdfm = estimate_structural_dfm(X, 2;
            identification=:sign, p=1, H=20,
            sign_check=sign_check, max_draws=5000)

        @test sdfm.identification == :sign

        # Verify sign restriction is satisfied in the factor IRF
        factor_irf = compute_irf(sdfm.factor_var, sdfm.Q, 20)
        @test factor_irf[1, 1, 1] > 0

        # Q should be orthogonal
        QQt = sdfm.Q * sdfm.Q'
        @test QQt ≈ Matrix{Float64}(I, 2, 2) atol=1e-10
    end

    # =========================================================================
    # irf Dispatch
    # =========================================================================

    @testset "irf dispatch returns ImpulseResponse" begin
        X, q = make_sdfm_data()
        T_obs, N = size(X)
        sdfm = estimate_structural_dfm(X, q; p=1, H=30)

        # Test irf dispatch
        irf_result = irf(sdfm, 20)
        @test irf_result isa ImpulseResponse{Float64}
        @test irf_result.horizon == 20
        @test size(irf_result.values) == (20, N, q)
        @test length(irf_result.variables) == N
        @test length(irf_result.shocks) == q
        @test irf_result.ci_type == :none

        # Requesting horizon beyond stored should truncate
        irf_long = irf(sdfm, 50)
        @test irf_long.horizon == 30  # stored H=30

        # Values should match stored structural_irf
        @test irf_result.values ≈ sdfm.structural_irf[1:20, :, :]
    end

    # =========================================================================
    # fevd Dispatch
    # =========================================================================

    @testset "fevd dispatch returns FEVD" begin
        X, q = make_sdfm_data()
        sdfm = estimate_structural_dfm(X, q; p=1, H=20)

        fevd_result = fevd(sdfm, 15)
        @test fevd_result isa FEVD{Float64}

        # FEVD is on the factor VAR, so q variables and q shocks
        @test length(fevd_result.variables) == q
        @test length(fevd_result.shocks) == q

        # FEVD proportions: shape (n_var, n_shock, horizon)
        # Proportions should sum to 1 across shocks at each horizon for each variable
        for h in 1:15
            for i in 1:q
                @test sum(fevd_result.proportions[i, :, h]) ≈ 1.0 atol=1e-10
            end
        end
    end

    # =========================================================================
    # Validation Errors
    # =========================================================================

    @testset "Validation errors" begin
        X, q = make_sdfm_data()

        # Invalid identification method
        @test_throws ArgumentError estimate_structural_dfm(X, q; identification=:invalid)

        # Sign identification without sign_check
        @test_throws ArgumentError estimate_structural_dfm(X, q; identification=:sign)

        # Invalid p
        @test_throws ArgumentError estimate_structural_dfm(X, q; p=0)

        # Invalid H
        @test_throws ArgumentError estimate_structural_dfm(X, q; H=0)
    end

    # =========================================================================
    # Display Output
    # =========================================================================

    @testset "Display output" begin
        X, q = make_sdfm_data()
        sdfm = estimate_structural_dfm(X, q; p=1, H=20)

        io = IOBuffer()
        show(io, sdfm)
        output = String(take!(io))

        @test occursin("Structural DFM", output)
        @test occursin("Cholesky", output)
        @test occursin("Dynamic factors", output)
        @test occursin("Impact Matrix B0", output)
        @test occursin("Variance Explained", output)
    end

    # =========================================================================
    # Dimensions
    # =========================================================================

    @testset "Dimensions consistency" begin
        T_obs, N, q = 150, 15, 2
        X, _ = make_sdfm_data(; T_obs=T_obs, N=N, q=q)
        H = 25

        sdfm = estimate_structural_dfm(X, q; p=1, H=H)

        # structural_irf: H x N x q
        @test size(sdfm.structural_irf) == (H, N, q)

        # B0: q x q
        @test size(sdfm.B0) == (q, q)

        # Q: q x q
        @test size(sdfm.Q) == (q, q)

        # loadings_td: N x q
        @test size(sdfm.loadings_td) == (N, q)

        # factor_var has q variables
        @test nvars(sdfm.factor_var) == q

        # GDFM factors: T_obs x q
        @test size(sdfm.gdfm.factors) == (T_obs, q)
    end

    # =========================================================================
    # StatsAPI Interface
    # =========================================================================

    @testset "StatsAPI interface" begin
        X, q = make_sdfm_data()
        sdfm = estimate_structural_dfm(X, q; p=1, H=20)

        @test nobs(sdfm) == size(X, 1)
        @test dof(sdfm) > 0
        @test length(r2(sdfm)) == size(X, 2)
        @test all(x -> 0 <= x <= 1, r2(sdfm))
    end

    # =========================================================================
    # Different VAR Lag Orders
    # =========================================================================

    @testset "Different VAR lag orders" begin
        X, q = make_sdfm_data()

        for p in [1, 2, 4]
            sdfm = estimate_structural_dfm(X, q; p=p, H=20)
            @test sdfm.p_var == p
            @test sdfm.factor_var.p == p
        end
    end

    # =========================================================================
    # Structural IRF Non-Trivial
    # =========================================================================

    @testset "Structural IRFs are non-trivial" begin
        X, q = make_sdfm_data()
        sdfm = estimate_structural_dfm(X, q; p=1, H=20)

        # IRFs should not be all zeros
        @test sum(abs.(sdfm.structural_irf)) > 0

        # Impact (h=1) should be non-zero for at least some variables
        impact = sdfm.structural_irf[1, :, :]
        @test maximum(abs.(impact)) > 1e-10

        # IRFs should generally decay (mean absolute IRF decreases)
        mean_abs_early = mean(abs.(sdfm.structural_irf[1:5, :, :]))
        mean_abs_late = mean(abs.(sdfm.structural_irf[16:20, :, :]))
        # With AR structure, later IRFs should not be much larger
        @test mean_abs_late < 10 * mean_abs_early
    end

end
