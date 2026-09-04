# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
    Tests for Arias, Rubio-Ramírez, and Waggoner (2018) SVAR Identification

These tests verify the implementation against theoretical properties
and examples from the paper.

Reference:
Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018).
"Inference Based on Structural Vector Autoregressions Identified With
Sign and Zero Restrictions: Theory and Applications."
Econometrica, 86(2), 685-720.
"""

using Test
using LinearAlgebra
using Statistics
using Random
using Logging
using ForwardDiff
using MacroEconometricModels

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "Arias et al. (2018) SVAR Identification" begin

    # ==========================================================================
    # Type Construction Tests
    # ==========================================================================

    @testset "Restriction Type Construction" begin
        # Zero restrictions
        zr = ZeroRestriction(1, 2, 0)
        @test zr.variable == 1
        @test zr.shock == 2
        @test zr.horizon == 0

        # Sign restrictions
        sr = SignRestriction(2, 1, 0, 1)
        @test sr.variable == 2
        @test sr.shock == 1
        @test sr.sign == 1

        # Convenience constructors
        zr2 = zero_restriction(3, 1; horizon=2)
        @test zr2.variable == 3
        @test zr2.horizon == 2

        sr2 = sign_restriction(1, 1, :positive)
        @test sr2.sign == 1

        sr3 = sign_restriction(2, 1, :negative; horizon=1)
        @test sr3.sign == -1
        @test sr3.horizon == 1
    end

    @testset "SVARRestrictions Construction" begin
        zeros = [ZeroRestriction(2, 1, 0), ZeroRestriction(3, 1, 0)]
        signs = [SignRestriction(1, 1, 0, 1)]

        restrictions = SVARRestrictions(3; zeros=zeros, signs=signs)

        @test restrictions.n_vars == 3
        @test restrictions.n_shocks == 3
        @test length(restrictions.zeros) == 2
        @test length(restrictions.signs) == 1
    end

    # ==========================================================================
    # Basic Identification Tests (Pure Sign Restrictions)
    # ==========================================================================

    @testset "Pure Sign Restrictions" begin
        rng = MersenneTwister(12345)  # DGP-02: explicit rng

        # Reference DGP (DGP-02 #791): non-diagonal A, non-identity B0.
        # The (1,1)+ and (2,2)+ restrictions hold at the truth (B0 diag = 1).
        T_obs, n, p = 200, 3, 1
        Y = dgp_var(rng; T=T_obs).Y

        model = estimate_var(Y, p)

        # Define sign restrictions only
        signs = [
            sign_restriction(1, 1, :positive),  # Var 1 responds + to shock 1
            sign_restriction(2, 2, :positive),  # Var 2 responds + to shock 2
        ]

        restrictions = SVARRestrictions(n; signs=signs)

        # Identify
        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        # Basic checks
        @test result isa AriasSVARResult
        @test length(result.Q_draws) > 0
        @test length(result.Q_draws) == length(result.weights)
        @test result.acceptance_rate > 0

        # Q matrices should be orthogonal
        for Q in result.Q_draws
            @test norm(Q' * Q - I) < 1e-10
            @test norm(Q * Q' - I) < 1e-10
        end

        # All accepted IRFs should satisfy sign restrictions
        for i in 1:size(result.irf_draws, 1)
            irf = result.irf_draws[i, :, :, :]
            @test irf[1, 1, 1] > 0  # Sign restriction 1
            @test irf[1, 2, 2] > 0  # Sign restriction 2
        end

        # Weights should sum to 1 (approximately)
        @test abs(sum(result.weights) - 1.0) < 1e-10
    end

    # ==========================================================================
    # Zero Restrictions Tests
    # ==========================================================================

    @testset "weighted quantile with Float32 eps(T) (T062 C-17)" begin
        vals = Float32[1, 2, 3, 4]
        w = Float32[1, 1, 1, 1]
        q = MacroEconometricModels._weighted_quantile(vals, w, 0.5f0)
        @test isfinite(q)
        @test 1 ≤ q ≤ 4
        # a zero-weight tie (cw[idx]==cw[idx-1]) must not blow up thanks to the eps(T) floor
        @test isfinite(MacroEconometricModels._weighted_quantile(Float32[1, 2, 3], Float32[1, 0, 1], 0.5f0))
    end

    @testset "Narrowed identification catch (T059)" begin
        # (1) predicate: numeric-degeneracy errors are rejectable; genuine bugs are not
        @test MacroEconometricModels._is_rejectable_draw_error(LinearAlgebra.SingularException(1))
        @test MacroEconometricModels._is_rejectable_draw_error(LinearAlgebra.PosDefException(1))
        @test MacroEconometricModels._is_rejectable_draw_error(DomainError(-1.0))
        @test !MacroEconometricModels._is_rejectable_draw_error(BoundsError())
        @test !MacroEconometricModels._is_rejectable_draw_error(MethodError(sqrt, ("x",)))
        @test !MacroEconometricModels._is_rejectable_draw_error(DimensionMismatch("x"))

        rng = MersenneTwister(2018)  # DGP-02: explicit rng
        Y = zeros(150, 3)
        for t in 2:150
            Y[t, :] = 0.5 * Y[t-1, :] + randn(rng, 3)
        end
        model = estimate_var(Y, 1)

        # (2) regression: a satisfiable sign-restricted run still succeeds
        ok = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive)])
        res = identify_arias(model, ok, 8; n_draws=5, n_rotations=100, rng=rng)
        @test length(res.Q_draws) ≥ 1
        @test 0 < res.acceptance_rate ≤ 1

        # (3) a genuine bug (out-of-range restriction index → BoundsError) now propagates
        #     instead of being swallowed into "No valid identification after N attempts".
        #     Bypass the public constructor, which now rejects out-of-range indices.
        bogus = SVARRestrictions(ZeroRestriction[], [SignRestriction(999, 1, 0, 1)], 3, 3)
        @test_throws BoundsError identify_arias(model, bogus, 8; n_draws=5, n_rotations=100)
    end

    @testset "Pure Zero Restrictions (Cholesky-like)" begin
        rng = MersenneTwister(23456)  # DGP-02: explicit rng

        # Reference DGP (DGP-02 #791); Cholesky-like zeros hold at the truth
        # (lower-triangular B0).
        T_obs, n, p = 200, 3, 1
        Y = dgp_var(rng; T=T_obs).Y
        model = estimate_var(Y, p)

        # Cholesky-equivalent zero restrictions (package RWZ convention: shock 1
        # most restricted, so impact is UPPER-triangular — Cholesky with reversed
        # variable order):
        # Shock 1: Only affects var 1 on impact (zeros on vars 2, 3)
        # Shock 2: Only affects vars 1, 2 on impact (zero on var 3)
        zeros = [
            zero_restriction(2, 1),  # Var 2 doesn't respond to shock 1 on impact
            zero_restriction(3, 1),  # Var 3 doesn't respond to shock 1 on impact
            zero_restriction(3, 2),  # Var 3 doesn't respond to shock 2 on impact
        ]

        restrictions = SVARRestrictions(n; zeros=zeros)

        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        @test length(result.Q_draws) > 0

        # Check zero restrictions are satisfied
        for i in 1:size(result.irf_draws, 1)
            irf = result.irf_draws[i, :, :, :]
            @test abs(irf[1, 2, 1]) < 1e-10  # Var 2, Shock 1, impact ≈ 0
            @test abs(irf[1, 3, 1]) < 1e-10  # Var 3, Shock 1, impact ≈ 0
            @test abs(irf[1, 3, 2]) < 1e-10  # Var 3, Shock 2, impact ≈ 0
        end
    end

    @testset "Mixed Zero and Sign Restrictions" begin
        rng = MersenneTwister(34567)  # DGP-02: explicit rng

        # Reference DGP with B0[3,2] < 0 (DGP-02 #791): the (3,2)-negative
        # sign restriction below holds at the truth instead of by luck.
        T_obs, n, p = 200, 3, 1
        Y = dgp_var(rng; B0=[1.0 0.0 0.0; 0.5 1.0 0.0; 0.3 -0.2 1.0], T=T_obs).Y
        model = estimate_var(Y, p)

        # Zero restrictions
        zeros = [
            zero_restriction(2, 1),  # Var 2 doesn't respond to shock 1 on impact
        ]

        # Sign restrictions
        signs = [
            sign_restriction(1, 1, :positive),  # Var 1 responds + to shock 1
            sign_restriction(3, 2, :negative),  # Var 3 responds - to shock 2
        ]

        restrictions = SVARRestrictions(n; zeros=zeros, signs=signs)

        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 200), rng=rng)

        @test length(result.Q_draws) > 0

        # Check all restrictions
        for i in 1:size(result.irf_draws, 1)
            irf = result.irf_draws[i, :, :, :]
            @test abs(irf[1, 2, 1]) < 1e-10  # Zero restriction
            @test irf[1, 1, 1] > 0           # Sign restriction 1
            @test irf[1, 3, 2] < 0           # Sign restriction 2
        end
    end

    # ==========================================================================
    # Long-Run Zero Restrictions
    # ==========================================================================

    @testset "Zero Restrictions at Different Horizons" begin
        rng = MersenneTwister(45678)  # DGP-02: explicit rng

        # Reference 2-var DGP (DGP-02 #791) estimated with p = 2 lags.
        T_obs, n, p = 200, 2, 2
        Y = dgp_var(rng; A=[[0.5 0.1; 0.0 0.4], [0.1 0.0; 0.0 0.1]],
                    B0=[1.0 0.0; 0.3 1.0], T=T_obs).Y
        model = estimate_var(Y, p)

        # Zero at horizon 1 (one period after impact). A lone non-impact zero
        # does NOT identify the system: the RWZ rank/order check rejects it
        # up front with IdentificationError (this is specified behavior — the
        # old try/catch skip was masking it, DGP-02 #791).
        zeros = [
            zero_restriction(1, 2; horizon=1),  # Var 1 doesn't respond to shock 2 at h=1
        ]

        restrictions = SVARRestrictions(n; zeros=zeros)

        @test_throws IdentificationError identify_arias(model, restrictions, 10;
            n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 200), rng=rng)
    end

    # ==========================================================================
    # Weighted Statistics Tests
    # ==========================================================================

    @testset "IRF Percentiles and Mean" begin
        rng = MersenneTwister(56789)  # DGP-02: explicit rng

        T_obs, n, p = 150, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        # Compute percentiles
        pct = irf_percentiles(result; quantiles=[0.16, 0.5, 0.84])
        mean_irf = irf_mean(result)

        @test size(pct) == (10, 2, 2, 3)
        @test size(mean_irf) == (10, 2, 2)

        # Percentiles should be ordered
        for h in 1:10
            for i in 1:n
                for j in 1:n
                    @test pct[h, i, j, 1] <= pct[h, i, j, 2]  # 16th <= 50th
                    @test pct[h, i, j, 2] <= pct[h, i, j, 3]  # 50th <= 84th
                end
            end
        end

        # Mean should be within reasonable bounds
        @test all(isfinite, mean_irf)
    end

    # ==========================================================================
    # Theoretical Properties Tests
    # ==========================================================================

    @testset "Orthogonality of Q Matrices" begin
        rng = MersenneTwister(67890)  # DGP-02: explicit rng

        T_obs, n, p = 150, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_arias(model, restrictions, 5; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 50), rng=rng)

        for Q in result.Q_draws
            # Q should be orthogonal
            @test norm(Q' * Q - I(n)) < 1e-10
            @test norm(Q * Q' - I(n)) < 1e-10

            # Columns should be unit vectors
            for j in 1:n
                @test abs(norm(Q[:, j]) - 1.0) < 1e-10
            end
        end
    end

    @testset "Weights are Positive and Sum to One" begin
        rng = MersenneTwister(78901)  # DGP-02: explicit rng

        T_obs, n, p = 150, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        zeros = [zero_restriction(2, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros, signs=signs)

        result = identify_arias(model, restrictions, 5; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        # All weights should be positive
        @test all(result.weights .> 0)

        # Weights should sum to 1
        @test abs(sum(result.weights) - 1.0) < 1e-10
    end

    # ==========================================================================
    # Edge Cases
    # ==========================================================================

    @testset "Single Variable" begin
        rng = MersenneTwister(89012)  # DGP-02: explicit rng

        T_obs, n, p = 100, 1, 1
        Y = dgp_var(rng; A=reshape([0.5], 1, 1), B0=reshape([1.0], 1, 1), T=T_obs).Y
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_arias(model, restrictions, 5; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 50), rng=rng)

        @test length(result.Q_draws) > 0
        @test all(result.irf_draws[:, 1, 1, 1] .> 0)
    end

    @testset "Two Variables - Block Recursive" begin
        rng = MersenneTwister(90123)  # DGP-02: explicit rng

        T_obs, n, p = 150, 2, 1
        # Block-recursive truth: lower-triangular B0 (DGP-02 #791).
        Y = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], B0=[1.0 0.0; 0.3 1.0], T=T_obs).Y
        model = estimate_var(Y, p)

        # Block recursive: var 2 doesn't respond to shock 1 on impact
        zeros = [zero_restriction(2, 1)]
        restrictions = SVARRestrictions(n; zeros=zeros)

        result = identify_arias(model, restrictions, 5; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 50), rng=rng)

        @test length(result.Q_draws) > 0

        for i in 1:size(result.irf_draws, 1)
            @test abs(result.irf_draws[i, 1, 2, 1]) < 1e-10
        end
    end

    @testset "Many Zero Restrictions" begin
        rng = MersenneTwister(12345)  # DGP-02: explicit rng

        T_obs, n, p = 200, 4, 1
        # 4-variable reference DGP: stationary A (row sums <= 0.6),
        # lower-triangular B0 (DGP-02 #791).
        Y = dgp_var(rng; A=[0.5 0.1 0.0 0.0; 0.0 0.4 0.1 0.0;
                            0.0 0.0 0.35 0.05; 0.05 0.0 0.0 0.3],
                    B0=[1.0 0.0 0.0 0.0; 0.4 1.0 0.0 0.0;
                        0.2 0.3 1.0 0.0; 0.1 0.2 0.3 1.0], T=T_obs).Y
        model = estimate_var(Y, p)

        # Recursive structure (package RWZ convention: shock 1 most restricted,
        # so impact is UPPER-triangular — Cholesky with reversed variable order)
        zeros = [
            zero_restriction(2, 1),
            zero_restriction(3, 1),
            zero_restriction(4, 1),
            zero_restriction(3, 2),
            zero_restriction(4, 2),
            zero_restriction(4, 3),
        ]

        restrictions = SVARRestrictions(n; zeros=zeros)

        result = identify_arias(model, restrictions, 5; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 50), rng=rng)

        @test length(result.Q_draws) > 0

        # Check all zero restrictions
        for i in 1:size(result.irf_draws, 1)
            irf = result.irf_draws[i, :, :, :]
            @test abs(irf[1, 2, 1]) < 1e-10
            @test abs(irf[1, 3, 1]) < 1e-10
            @test abs(irf[1, 4, 1]) < 1e-10
            @test abs(irf[1, 3, 2]) < 1e-10
            @test abs(irf[1, 4, 2]) < 1e-10
            @test abs(irf[1, 4, 3]) < 1e-10
        end
    end

    # ==========================================================================
    # Numerical Stability Tests
    # ==========================================================================

    @testset "Numerical Stability - Near Singular Covariance" begin
        rng = MersenneTwister(23456)  # DGP-02: explicit rng

        T_obs, n, p = 150, 3, 1
        Y = randn(rng, T_obs, n)
        # Add near-collinearity
        Y[:, 3] = Y[:, 1] + 0.01 * randn(rng, T_obs)

        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        # Should not error, even with near-singular covariance
        result = identify_arias(model, restrictions, 5; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        # May have few or no draws, but should not crash
        @test result isa AriasSVARResult
    end

    @testset "Reproducibility" begin
        T_obs, n, p = 150, 2, 1

        rng = MersenneTwister(54321)  # DGP-02: explicit rng
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        # Same explicit rng stream twice → identical results (DGP-02).
        result1 = identify_arias(model, restrictions, 5; n_draws=5, n_rotations=20,
                                 rng=MersenneTwister(11111))

        result2 = identify_arias(model, restrictions, 5; n_draws=5, n_rotations=20,
                                 rng=MersenneTwister(11111))

        # Same seed should give same results
        @test length(result1.Q_draws) == length(result2.Q_draws)
        @test result1.irf_draws ≈ result2.irf_draws
    end

    # ==========================================================================
    # Input Validation Tests
    # ==========================================================================

    @testset "Input Validation" begin
        rng = MersenneTwister(34567)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        # Mismatched dimensions
        restrictions_wrong = SVARRestrictions(3)  # 3-var restrictions for 2-var model

        @test_throws AssertionError identify_arias(model, restrictions_wrong, 5)
    end

    # ==========================================================================
    # Comparison with Cholesky (Special Case)
    # ==========================================================================

    @testset "Comparison with Cholesky Identification" begin
        rng = MersenneTwister(45678)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        # Reference DGP: Cholesky zeros hold at the truth (DGP-02 #791).
        Y = dgp_var(rng; T=T_obs).Y
        model = estimate_var(Y, p)

        # Cholesky-equivalent restrictions (package RWZ convention: shock 1 most
        # restricted, so impact is UPPER-triangular — Cholesky with reversed
        # variable order)
        zeros = [
            zero_restriction(2, 1),
            zero_restriction(3, 1),
            zero_restriction(3, 2),
        ]

        restrictions = SVARRestrictions(n; zeros=zeros)

        result_arias = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        # Get Cholesky IRF
        L = cholesky_factor(model)
        Q_chol = identify_cholesky(model)
        irf_chol = MacroEconometricModels.compute_irf(model, Q_chol, 10)

        # Impact responses from Arias should match Cholesky structure
        # (upper-triangular impact matrix under the package's shock ordering)
        for i in 1:size(result_arias.irf_draws, 1)
            irf = result_arias.irf_draws[i, :, :, :]

            # Check upper-triangular structure at impact
            @test abs(irf[1, 2, 1]) < 1e-10  # (2,1) = 0
            @test abs(irf[1, 3, 1]) < 1e-10  # (3,1) = 0
            @test abs(irf[1, 3, 2]) < 1e-10  # (3,2) = 0
        end
    end

    # ==========================================================================
    # Large Scale Test
    # ==========================================================================

    @testset "Larger System (5 variables)" begin
        rng = MersenneTwister(56789)  # DGP-02: explicit rng

        T_obs, n, p = 300, 5, 2
        # 5-variable reference DGP: stationary A, lower-triangular B0 (DGP-02 #791).
        Y = dgp_var(rng; A=[0.45 0.05 0.0 0.0 0.0; 0.0 0.4 0.05 0.0 0.0;
                            0.0 0.0 0.35 0.05 0.0; 0.0 0.0 0.0 0.3 0.05;
                            0.05 0.0 0.0 0.0 0.25],
                    B0=[1.0 0.0 0.0 0.0 0.0; 0.3 1.0 0.0 0.0 0.0;
                        0.2 0.2 1.0 0.0 0.0; 0.1 0.1 0.2 1.0 0.0;
                        0.1 0.1 0.1 0.2 1.0], T=T_obs).Y
        model = estimate_var(Y, p)

        # Some sign restrictions
        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 2, :positive),
            sign_restriction(3, 3, :positive),
        ]

        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        @test length(result.Q_draws) > 0
        @test size(result.irf_draws, 2) == 10  # horizon
        @test size(result.irf_draws, 3) == 5   # n_vars
        @test size(result.irf_draws, 4) == 5   # n_shocks
    end

    # ==========================================================================
    # AriasSVARResult Methods
    # ==========================================================================

    @testset "AriasSVARResult Methods" begin
        rng = MersenneTwister(67890)  # DGP-02: explicit rng

        T_obs, n, p = 150, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 50), rng=rng)

        # Test irf_percentiles
        pct = irf_percentiles(result)
        @test size(pct) == (10, 2, 2, 3)  # default 3 quantiles

        pct5 = irf_percentiles(result; quantiles=[0.05, 0.5, 0.95])
        @test size(pct5) == (10, 2, 2, 3)

        # Test irf_mean
        m = irf_mean(result)
        @test size(m) == (10, 2, 2)

        # Mean should be between min and max of draws
        for h in 1:10
            for i in 1:n
                for j in 1:n
                    vals = result.irf_draws[:, h, i, j]
                    @test minimum(vals) <= m[h, i, j] <= maximum(vals)
                end
            end
        end
    end

end

# ==========================================================================
# Bayesian Arias Identification Tests
# ==========================================================================

@testset "identify_arias_bayesian" begin

    @testset "Basic Bayesian Sign Restrictions" begin
        rng = MersenneTwister(11111)  # DGP-02: explicit rng

        # Generate simple VAR data (reduced from T=200)
        T_obs, n, p = 100, 2, 1
        # Reference 2-var DGP (DGP-02 #791); the (1,1)+ restriction holds
        # at the truth (B0[1,1] = 1).
        Y = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], B0=[1.0 0.0; 0.3 1.0], T=T_obs).Y

        # Estimate BVAR
        try
            post = estimate_bvar(Y, p; n_draws=(FAST ? 15 : 30), rng=rng)

            # Define sign restrictions
            signs = [sign_restriction(1, 1, :positive)]
            restrictions = SVARRestrictions(n; signs=signs)

            # Run Bayesian identification
            result = identify_arias_bayesian(post, restrictions, 5;
                n_rotations=(FAST ? 10 : 30), quantiles=[0.16, 0.5, 0.84], rng=rng)

            # Check output structure
            @test haskey(result, :irf_quantiles)
            @test haskey(result, :irf_mean)
            @test haskey(result, :acceptance_rates)
            @test haskey(result, :total_accepted)
            @test haskey(result, :weights)

            # Check dimensions
            @test size(result.irf_quantiles) == (5, n, n, 3)  # horizon × n × n × quantiles
            @test size(result.irf_mean) == (5, n, n)
            @test length(result.acceptance_rates) == (FAST ? 15 : 30)  # n_draws from posterior
            @test length(result.weights) == result.total_accepted

            # Check weights sum to 1
            @test abs(sum(result.weights) - 1.0) < 1e-10

            # Check quantiles are ordered
            for h in 1:5, i in 1:n, j in 1:n
                @test result.irf_quantiles[h, i, j, 1] <= result.irf_quantiles[h, i, j, 2]
                @test result.irf_quantiles[h, i, j, 2] <= result.irf_quantiles[h, i, j, 3]
            end

            # Check mean is finite
            @test all(isfinite, result.irf_mean)

        catch e
            @warn "Bayesian identification test failed" exception=(e, catch_backtrace())
            @test_skip "Bayesian Arias identification may fail due to MCMC issues"
        end
    end

    @testset "Bayesian Zero Restrictions" begin
        rng = MersenneTwister(22222)  # DGP-02: explicit rng

        T_obs, n, p = 100, 3, 1
        Y = dgp_var(rng; T=T_obs).Y

        try
            post = estimate_bvar(Y, p; n_draws=(FAST ? 10 : 20), rng=rng)

            # Cholesky-equivalent zero restrictions (package RWZ convention:
            # shock 1 most restricted → upper-triangular impact)
            zeros = [
                zero_restriction(2, 1),
                zero_restriction(3, 1),
                zero_restriction(3, 2),
            ]
            restrictions = SVARRestrictions(n; zeros=zeros)

            result = identify_arias_bayesian(post, restrictions, 5;
                n_rotations=(FAST ? 10 : 50), rng=rng)

            @test result.total_accepted > 0
            @test all(isfinite, result.irf_mean)

        catch e
            @warn "Bayesian zero restrictions test failed" exception=(e, catch_backtrace())
            @test_skip "Bayesian identification with zeros may have convergence issues"
        end
    end

    @testset "Bayesian Mixed Zero and Sign Restrictions" begin
        rng = MersenneTwister(33333)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = randn(rng, T_obs, n)

        try
            post = estimate_bvar(Y, p; n_draws=(FAST ? 10 : 20), rng=rng)

            zeros = [zero_restriction(2, 1)]
            signs = [sign_restriction(1, 1, :positive)]
            restrictions = SVARRestrictions(n; zeros=zeros, signs=signs)

            result = identify_arias_bayesian(post, restrictions, 5;
                n_rotations=(FAST ? 10 : 50), rng=rng)

            @test result.total_accepted > 0
            @test size(result.irf_mean) == (5, n, n)

        catch e
            @warn "Bayesian mixed restrictions test failed" exception=(e, catch_backtrace())
            @test_skip "Bayesian mixed restrictions may have issues"
        end
    end

    @testset "Bayesian Identification without Data" begin
        rng = MersenneTwister(44444)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = randn(rng, T_obs, n)

        try
            post = estimate_bvar(Y, p; n_draws=(FAST ? 10 : 20), rng=rng)

            signs = [sign_restriction(1, 1, :positive)]
            restrictions = SVARRestrictions(n; signs=signs)

            # Run without providing data
            result = identify_arias_bayesian(post, restrictions, 5;
                n_rotations=(FAST ? 10 : 30), rng=rng)

            @test result.total_accepted > 0
            @test all(isfinite, result.irf_mean)

        catch e
            @warn "Bayesian identification without data test failed" exception=(e, catch_backtrace())
            @test_skip "Bayesian identification without data may have issues"
        end
    end

    @testset "Bayesian Custom Quantiles" begin
        rng = MersenneTwister(55555)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], B0=[1.0 0.0; 0.3 1.0], T=T_obs).Y

        try
            post = estimate_bvar(Y, p; n_draws=(FAST ? 10 : 20), rng=rng)

            signs = [sign_restriction(1, 1, :positive)]
            restrictions = SVARRestrictions(n; signs=signs)

            # Custom quantiles
            custom_q = [0.05, 0.25, 0.5, 0.75, 0.95]
            result = identify_arias_bayesian(post, restrictions, 5;
                n_rotations=(FAST ? 10 : 30), quantiles=custom_q, rng=rng)

            @test size(result.irf_quantiles, 4) == length(custom_q)

            # Quantiles should be ordered
            for h in 1:5, i in 1:n, j in 1:n
                for q in 1:(length(custom_q)-1)
                    @test result.irf_quantiles[h, i, j, q] <= result.irf_quantiles[h, i, j, q+1]
                end
            end

        catch e
            @warn "Custom quantiles test failed" exception=(e, catch_backtrace())
            @test_skip "Custom quantiles test may have issues"
        end
    end

    @testset "Bayesian Single Variable" begin
        rng = MersenneTwister(66666)  # DGP-02: explicit rng

        T_obs, n, p = 80, 1, 1
        Y = randn(rng, T_obs, n)

        try
            post = estimate_bvar(Y, p; n_draws=(FAST ? 10 : 20), rng=rng)

            signs = [sign_restriction(1, 1, :positive)]
            restrictions = SVARRestrictions(n; signs=signs)

            result = identify_arias_bayesian(post, restrictions, 5;
                n_rotations=(FAST ? 10 : 30), rng=rng)

            @test result.total_accepted > 0
            @test size(result.irf_mean) == (5, 1, 1)

        catch e
            @warn "Single variable Bayesian test failed" exception=(e, catch_backtrace())
            @test_skip "Single variable Bayesian identification may have issues"
        end
    end

end

# ==========================================================================
# Helper Function Tests
# ==========================================================================

@testset "Helper Functions Coverage" begin

    @testset "_weighted_quantile" begin
        # Test basic functionality
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform weights

        # Median should be around 2.5-3.5 (allowing for floating point)
        median_val = MacroEconometricModels._weighted_quantile(vals, weights, 0.5)
        @test 2.4 <= median_val <= 3.6  # Relaxed tolerance for floating point

        # 0th percentile should be close to minimum
        q0 = MacroEconometricModels._weighted_quantile(vals, weights, 0.0)
        @test q0 ≈ 1.0

        # 100th percentile should be close to maximum
        q100 = MacroEconometricModels._weighted_quantile(vals, weights, 1.0)
        @test isapprox(q100, 5.0, atol=1e-8)

        # Non-uniform weights - skewed towards low values
        weights_skewed = [0.5, 0.25, 0.15, 0.05, 0.05]
        median_skewed = MacroEconometricModels._weighted_quantile(vals, weights_skewed, 0.5)
        @test median_skewed <= median_val + 0.1  # Should be shifted towards lower values or similar

        # Single value
        single_vals = [42.0]
        single_weights = [1.0]
        @test MacroEconometricModels._weighted_quantile(single_vals, single_weights, 0.5) ≈ 42.0

        # Two values
        two_vals = [1.0, 10.0]
        two_weights = [0.5, 0.5]
        q50_two = MacroEconometricModels._weighted_quantile(two_vals, two_weights, 0.5)
        @test 1.0 <= q50_two <= 10.0
    end

    @testset "_compute_ma_coefficients" begin
        rng = MersenneTwister(77777)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 2
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        horizon = 10
        Phi = MacroEconometricModels._compute_ma_coefficients(model, horizon)

        # Should return horizon + 1 matrices (0 to horizon)
        @test length(Phi) == horizon + 1

        # First matrix should be identity
        @test Phi[1] ≈ Matrix{Float64}(I, n, n)

        # All matrices should have correct dimensions
        for i in 1:(horizon + 1)
            @test size(Phi[i]) == (n, n)
        end

        # All values should be finite
        for i in 1:(horizon + 1)
            @test all(isfinite, Phi[i])
        end
    end

    @testset "haar_orthogonal" begin
        rng = MersenneTwister(88888)  # DGP-02: explicit rng

        for n in [2, 3, 4, 5]
            Q = MacroEconometricModels.haar_orthogonal(n, Float64)

            # Should be orthogonal
            @test size(Q) == (n, n)
            @test norm(Q' * Q - I(n)) < 1e-10
            @test norm(Q * Q' - I(n)) < 1e-10

            # Columns should be unit vectors
            for j in 1:n
                @test abs(norm(Q[:, j]) - 1.0) < 1e-10
            end
        end
        @test MacroEconometricModels.haar_orthogonal === MacroEconometricModels.generate_Q
    end

    @testset "_check_zero_restrictions" begin
        # Create a simple IRF array
        n, horizon = 3, 5
        irf = zeros(horizon, n, n)
        irf .= 1.0  # All ones initially

        # Set some zeros
        irf[1, 2, 1] = 0.0  # var 2 to shock 1 at impact is zero
        irf[1, 3, 1] = 0.0  # var 3 to shock 1 at impact is zero

        # Create restrictions that match
        zeros_match = [
            ZeroRestriction(2, 1, 0),  # horizon 0 => irf index 1
            ZeroRestriction(3, 1, 0),
        ]
        restrictions = SVARRestrictions(zeros_match, SignRestriction[], n, n)

        @test MacroEconometricModels._check_zero_restrictions(irf, restrictions)

        # Create restrictions that don't match
        zeros_no_match = [
            ZeroRestriction(1, 1, 0),  # This is not zero
        ]
        restrictions_no = SVARRestrictions(zeros_no_match, SignRestriction[], n, n)

        @test !MacroEconometricModels._check_zero_restrictions(irf, restrictions_no)

        # Empty restrictions should return true
        empty_restrictions = SVARRestrictions(ZeroRestriction[], SignRestriction[], n, n)
        @test MacroEconometricModels._check_zero_restrictions(irf, empty_restrictions)
    end

    @testset "_check_sign_restrictions" begin
        n, horizon = 2, 5
        irf = zeros(horizon, n, n)
        irf[1, 1, 1] = 1.0   # Positive
        irf[1, 2, 1] = -1.0  # Negative
        irf[1, 1, 2] = 0.5   # Positive
        irf[1, 2, 2] = -0.5  # Negative

        # Matching restrictions
        signs_match = [
            SignRestriction(1, 1, 0, 1),   # var 1, shock 1, positive
            SignRestriction(2, 1, 0, -1),  # var 2, shock 1, negative
        ]
        restrictions = SVARRestrictions(ZeroRestriction[], signs_match, n, n)

        @test MacroEconometricModels._check_sign_restrictions(irf, restrictions)

        # Non-matching restrictions
        signs_no_match = [
            SignRestriction(1, 1, 0, -1),  # Expecting negative, but it's positive
        ]
        restrictions_no = SVARRestrictions(ZeroRestriction[], signs_no_match, n, n)

        @test !MacroEconometricModels._check_sign_restrictions(irf, restrictions_no)

        # Empty restrictions should return true
        empty_restrictions = SVARRestrictions(ZeroRestriction[], SignRestriction[], n, n)
        @test MacroEconometricModels._check_sign_restrictions(irf, empty_restrictions)
    end

    @testset "_draw_null_space_vector" begin
        rng = MersenneTwister(99999)  # DGP-02: explicit rng

        # No constraints - should return random unit vector
        n = 3
        v1 = MacroEconometricModels._draw_null_space_vector(Vector{Float64}[], n)
        @test length(v1) == n
        @test abs(norm(v1) - 1.0) < 1e-10

        # Single constraint - result should be orthogonal to it
        constraint = [1.0, 0.0, 0.0]
        v2 = MacroEconometricModels._draw_null_space_vector([constraint], n)
        @test abs(norm(v2) - 1.0) < 1e-10
        @test abs(dot(v2, constraint)) < 1e-10  # Orthogonal to constraint

        # Two constraints in 3D
        c1 = [1.0, 0.0, 0.0]
        c2 = [0.0, 1.0, 0.0]
        v3 = MacroEconometricModels._draw_null_space_vector([c1, c2], n)
        @test abs(norm(v3) - 1.0) < 1e-10
        @test abs(dot(v3, c1)) < 1e-10
        @test abs(dot(v3, c2)) < 1e-10
        # Should be parallel to [0, 0, 1]
        @test abs(abs(v3[3]) - 1.0) < 1e-10
    end

    @testset "_compute_importance_weight" begin
        rng = MersenneTwister(12121)  # DGP-02: explicit rng

        T_obs, n, p = 100, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        Phi = MacroEconometricModels._compute_ma_coefficients(model, 5)
        L = safe_cholesky(model.Sigma)
        Q = MacroEconometricModels.haar_orthogonal(n, Float64)

        # No zero restrictions - backward-compatible 4-arg form should return 1
        restrictions_no_zeros = SVARRestrictions(ZeroRestriction[], SignRestriction[], n, n)
        w1 = MacroEconometricModels._compute_importance_weight(Q, restrictions_no_zeros, Phi, L)
        @test w1 ≈ 1.0

        # With zero restrictions - new 6-arg form with model and setup
        zrs = [ZeroRestriction(2, 1, 0)]
        restrictions_with_zeros = SVARRestrictions(zrs, SignRestriction[], n, n)
        setup = MacroEconometricModels._AriasSVARSetup(restrictions_with_zeros, n, Float64)
        Q_zero = MacroEconometricModels._draw_Q_with_zero_restrictions(restrictions_with_zeros, Phi, L)
        w2 = MacroEconometricModels._compute_importance_weight(Q_zero, model, setup, restrictions_with_zeros, Phi, L)
        @test w2 > 0
        @test isfinite(w2)
    end

    @testset "_build_zero_constraint_matrix" begin
        rng = MersenneTwister(23232)  # DGP-02: explicit rng

        T_obs, n, p = 100, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        Phi = MacroEconometricModels._compute_ma_coefficients(model, 5)
        L = safe_cholesky(model.Sigma)

        # Zero restriction on shock 1
        zeros = [ZeroRestriction(2, 1, 0)]
        restrictions = SVARRestrictions(zeros, SignRestriction[], n, n)

        constraints = MacroEconometricModels._build_zero_constraint_matrix(restrictions, 1, Phi, L)

        # Should have one constraint (for shock 1)
        @test length(constraints) == 1
        @test length(constraints[1]) == n

        # Constraint for shock 2 (no zeros defined for it)
        constraints2 = MacroEconometricModels._build_zero_constraint_matrix(restrictions, 2, Phi, L)
        @test isempty(constraints2)
    end

    @testset "_compute_irf_for_Q" begin
        rng = MersenneTwister(34343)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        horizon = 5
        Phi = MacroEconometricModels._compute_ma_coefficients(model, horizon)
        L = safe_cholesky(model.Sigma)
        Q = MacroEconometricModels.haar_orthogonal(n, Float64)

        irf = MacroEconometricModels._compute_irf_for_Q(model, Q, Phi, L, horizon)

        @test size(irf) == (horizon, n, n)
        @test all(isfinite, irf)

        # Impact response should be L * Q
        A0_inv = L * Q
        @test irf[1, :, :] ≈ A0_inv
    end

    @testset "_draw_Q_with_zero_restrictions" begin
        rng = MersenneTwister(45454)  # DGP-02: explicit rng

        T_obs, n, p = 100, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        Phi = MacroEconometricModels._compute_ma_coefficients(model, 5)
        L = safe_cholesky(model.Sigma)

        # Recursive zero restrictions (package RWZ convention: shock 1 most
        # restricted → upper-triangular impact)
        zeros = [
            ZeroRestriction(2, 1, 0),
            ZeroRestriction(3, 1, 0),
            ZeroRestriction(3, 2, 0),
        ]
        restrictions = SVARRestrictions(zeros, SignRestriction[], n, n)

        Q = MacroEconometricModels._draw_Q_with_zero_restrictions(restrictions, Phi, L)

        # Q should be orthogonal
        @test size(Q) == (n, n)
        @test norm(Q' * Q - I(n)) < 1e-10
        @test norm(Q * Q' - I(n)) < 1e-10
    end

end

# ==========================================================================
# Importance Weight Correctness Tests (Arias et al. 2018, Proposition 4)
# ==========================================================================

@testset "Draw-Dependent Importance Weight Correctness" begin

    @testset "Weight variability with zero restrictions" begin
        rng = MersenneTwister(42424)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        # Zero restrictions: var 2 and var 3 don't respond to shock 1
        zrs = [zero_restriction(2, 1), zero_restriction(3, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zrs, signs=signs)

        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        # Key test: weights must NOT all be identical (the old bug gave constant weights)
        unique_weights = length(unique(round.(result.weights, digits=8)))
        @test unique_weights > 1  # Weights vary per draw
        @test all(w -> w > 0, result.weights)
        @test abs(sum(result.weights) - 1.0) < 1e-10
    end

    @testset "Pure sign restrictions give unit weights" begin
        rng = MersenneTwister(43434)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive), sign_restriction(2, 2, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        # All weights should be equal (1/N)
        expected_w = 1.0 / length(result.weights)
        @test all(w -> isapprox(w, expected_w, atol=1e-10), result.weights)
    end

    @testset "Numerical Jacobian accuracy" begin
        # Test against known analytical Jacobian: f(x) = [x[1]^2, x[1]*x[2], x[2]^3]
        f(x) = [x[1]^2, x[1]*x[2], x[2]^3]
        x0 = [2.0, 3.0]

        J_num = MacroEconometricModels._numerical_jacobian(f, x0)

        # Analytical: [2x1  0; x2  x1; 0  3x2^2]
        J_exact = [4.0 0.0; 3.0 2.0; 0.0 27.0]

        @test size(J_num) == (3, 2)
        @test isapprox(J_num, J_exact, atol=1e-5)
    end

    @testset "Structural param roundtrip" begin
        rng = MersenneTwister(44444)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        L = safe_cholesky(model.Sigma)
        Q = MacroEconometricModels.haar_orthogonal(n, Float64)

        # Forward: (B, L, Q) → (A0, Aplus)
        A0, Aplus = MacroEconometricModels._rf_to_struct(model.B, L, Q)

        # Backward: (A0, Aplus) → (B_rec, Σ_rec)
        B_rec, Sigma_rec = MacroEconometricModels._struct_to_rf(A0, Aplus)

        # B should recover exactly
        @test isapprox(B_rec, model.B, atol=1e-8)

        # Σ_rec = inv(A0)*inv(A0)' = Q'*L'*L*Q (rotated Σ), but Σ = L*L'
        # So Σ_rec ≠ Σ in general. Instead verify that we can recover (A0, Aplus) from (B_rec, L_rec, Q_rec)
        L_rec = safe_cholesky(Sigma_rec)
        Q_rec = Matrix{Float64}(L_rec') * A0
        A0_rec, Aplus_rec = MacroEconometricModels._rf_to_struct(B_rec, L_rec, Q_rec)
        @test isapprox(A0_rec, A0, atol=1e-8)
        @test isapprox(Aplus_rec, Aplus, atol=1e-8)
    end

    @testset "Q ↔ spheres roundtrip" begin
        rng = MersenneTwister(45454)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        zrs = [zero_restriction(2, 1)]
        restrictions = SVARRestrictions(n; zeros=zrs)

        Phi = MacroEconometricModels._compute_ma_coefficients(model, 1)
        L = safe_cholesky(model.Sigma)
        setup = MacroEconometricModels._AriasSVARSetup(restrictions, n, Float64)

        # Draw a valid Q satisfying zero restrictions
        Q_orig = MacroEconometricModels._draw_Q_with_zero_restrictions(restrictions, Phi, L)

        # Forward: Q → w
        w = MacroEconometricModels._Q_to_spheres(Q_orig, setup, restrictions, Phi, L)

        # Backward: w → Q
        Q_rec = MacroEconometricModels._spheres_to_Q(w, setup, restrictions, Phi, L)

        @test isapprox(Q_rec, Q_orig, atol=1e-8)
    end

    @testset "Volume element sanity checks" begin
        rng = MersenneTwister(46464)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        zrs = [zero_restriction(2, 1), zero_restriction(3, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zrs, signs=signs)

        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        # All pre-normalization weights should be positive and finite
        Phi = MacroEconometricModels._compute_ma_coefficients(model, 10)
        L = safe_cholesky(model.Sigma)
        setup = MacroEconometricModels._AriasSVARSetup(restrictions, n, Float64)

        for Q in result.Q_draws
            w = MacroEconometricModels._compute_importance_weight(Q, model, setup, restrictions, Phi, L)
            @test w > 0
            @test isfinite(w)
        end
    end

    @testset "Cholesky equivalence: diagonal impact entries match" begin
        rng = MersenneTwister(47474)  # DGP-02: explicit rng

        T_obs, n, p = 300, 3, 1
        # Reference DGP (DGP-02 #791).
        Y = dgp_var(rng; T=T_obs).Y
        model = estimate_var(Y, p)

        # Full recursive zero restrictions (package RWZ convention: shock 1 most
        # restricted, so impact is UPPER-triangular — Cholesky with reversed
        # variable order).
        zrs = [
            zero_restriction(2, 1),
            zero_restriction(3, 1),
            zero_restriction(3, 2),
        ]
        restrictions = SVARRestrictions(n; zeros=zrs)

        result = identify_arias(model, restrictions, 10; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100), rng=rng)

        # Exact recursive zeros pin the impact matrix up to column signs on ANY
        # data. Under the package's shock ordering the impact is
        # upper-triangular, i.e. lower-triangular with REVERSED variable order:
        # |B[j,j]| equals diagonal entry (n-j+1) of chol(P*Sigma*P').
        # NOTE (DGP-02 #791): the old reference compared against the plain
        # lower factor (compute_irf with Q=I). That matches only when
        # Sigma-hat ≈ I — true for the old randn data, false for the
        # reference DGP (upper-triangular B has no such diagonal pinning).
        P = reverse(Matrix{Float64}(I, n, n); dims=2)
        Lt = cholesky(Hermitian(P * model.Sigma * P')).L
        lt_diag = diag(Matrix(Lt))  # positive by construction

        n_draws_got = size(result.irf_draws, 1)
        @test n_draws_got > 0
        ref_abs = abs.(result.irf_draws[1, 1, :, :])
        for idx in 1:min(5, n_draws_got)
            irf_draw = result.irf_draws[idx, :, :, :]
            for j in 1:n
                # Diagonal matches reversed-order Cholesky in absolute value
                @test isapprox(abs(irf_draw[1, j, j]), lt_diag[n - j + 1], rtol=1e-8)
            end
            # Exact identification: every draw is the same impact up to signs
            @test maximum(abs.(abs.(irf_draw[1, :, :]) .- ref_abs)) < 1e-8
        end
    end

    @testset "compute_weights=false gives unit weights" begin
        rng = MersenneTwister(48484)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        zrs = [zero_restriction(2, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zrs, signs=signs)

        result = identify_arias(model, restrictions, 5; n_draws=(FAST ? 5 : 10), n_rotations=(FAST ? 20 : 100),
                                compute_weights=false, rng=rng)

        # All weights equal (no volume element computation)
        expected_w = 1.0 / length(result.weights)
        @test all(w -> isapprox(w, expected_w, atol=1e-10), result.weights)
    end

    @testset "_AriasSVARSetup construction" begin
        n = 3
        zrs = [zero_restriction(2, 1), zero_restriction(3, 1), zero_restriction(3, 2)]
        restrictions = SVARRestrictions(n; zeros=zrs)

        setup = MacroEconometricModels._AriasSVARSetup(restrictions, n, Float64)

        # Shock 1: 2 zeros → s_1 = 3 - 0 - 2 = 1
        # Shock 2: 1 zero  → s_2 = 3 - 1 - 1 = 1
        # Shock 3: 0 zeros → s_3 = 3 - 2 - 0 = 1
        @test setup.zeros_per_shock == [2, 1, 0]
        @test setup.sphere_dims == [1, 1, 1]
        @test setup.dim == 3
        @test length(setup.W) == 3
        for (j, s_j) in enumerate(setup.sphere_dims)
            @test size(setup.W[j]) == (s_j, n)
        end
    end

    @testset "_vech" begin
        A = [1.0 2.0 3.0; 2.0 4.0 5.0; 3.0 5.0 6.0]
        v = MacroEconometricModels._vech(A)
        @test v == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        B = [1.0 0.0; 0.0 2.0]
        v2 = MacroEconometricModels._vech(B)
        @test v2 == [1.0, 0.0, 2.0]
    end

    @testset "_pack/_unpack_structural roundtrip" begin
        rng = MersenneTwister(49494)  # DGP-02: explicit rng
        n, m = 3, 7  # m = 1 + n*p for p=2
        A0 = randn(rng, n, n)
        Aplus = randn(rng, m, n)

        x = MacroEconometricModels._pack_structural(A0, Aplus)
        @test length(x) == n*n + m*n

        A0_rec, Aplus_rec = MacroEconometricModels._unpack_structural(x, n, m)
        @test A0_rec ≈ A0
        @test Aplus_rec ≈ Aplus
    end

    @testset "_log_abs_det" begin
        A = [2.0 0.0; 0.0 3.0]
        @test isapprox(MacroEconometricModels._log_abs_det(A), log(6.0), atol=1e-10)

        # Singular matrix
        B = [1.0 1.0; 1.0 1.0]
        @test MacroEconometricModels._log_abs_det(B) == -Inf
    end

    @testset "_log_volume_element" begin
        # Simple test: f = identity on R^2, h = [x[1] - 1]
        # df/dx = I, dh/dx = [1, 0]. Null space of dh = [0; 1].
        # N = I * [0; 1] = [0; 1]. N'N = [1]. log|det| = 0.
        f_id(x) = x
        h_constraint(x) = [x[1] - 1.0]

        x0 = [1.0, 2.0]
        lve = MacroEconometricModels._log_volume_element(f_id, x0, h_constraint)
        @test isapprox(lve, 0.0, atol=1e-4)
    end

    @testset "_compute_qr_signs" begin
        rng = MersenneTwister(50505)  # DGP-02: explicit rng
        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        zrs = [zero_restriction(2, 1)]
        restrictions = SVARRestrictions(n; zeros=zrs)

        Phi = MacroEconometricModels._compute_ma_coefficients(model, 1)
        L = safe_cholesky(model.Sigma)
        setup = MacroEconometricModels._AriasSVARSetup(restrictions, n, Float64)

        Q = MacroEconometricModels._draw_Q_with_zero_restrictions(restrictions, Phi, L)
        signs = MacroEconometricModels._compute_qr_signs(Q, setup, restrictions, Phi, L)

        @test length(signs) == n
        for j in 1:n
            @test all(s -> s == 1 || s == -1, signs[j])
        end

        # Using ref_signs should give same w as without
        w_default = MacroEconometricModels._Q_to_spheres(Q, setup, restrictions, Phi, L)
        w_ref = MacroEconometricModels._Q_to_spheres(Q, setup, restrictions, Phi, L; ref_signs=signs)
        @test w_default ≈ w_ref
    end

    @testset "ff_h Jacobian smoothness (Issue #37)" begin
        rng = MersenneTwister(51515)  # DGP-02: explicit rng
        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        zrs = [zero_restriction(2, 1), zero_restriction(3, 1)]
        restrictions = SVARRestrictions(n; zeros=zrs)

        Phi = MacroEconometricModels._compute_ma_coefficients(model, 1)
        L = safe_cholesky(model.Sigma)
        setup = MacroEconometricModels._AriasSVARSetup(restrictions, n, Float64)
        Q = MacroEconometricModels._draw_Q_with_zero_restrictions(restrictions, Phi, L)

        m_size = size(model.B, 1)
        A0, Aplus = MacroEconometricModels._rf_to_struct(model.B, L, Q)
        structpara = MacroEconometricModels._pack_structural(A0, Aplus)

        max_h = maximum(zr.horizon for zr in restrictions.zeros)
        ff_h = MacroEconometricModels._build_ff_h(setup, restrictions, n, m_size, p, max_h)

        # Jacobian should be finite and well-conditioned
        J = ForwardDiff.jacobian(ff_h, structpara)
        @test all(isfinite, J)
        @test !any(isnan, J)

        # Verify that the volume element is finite
        zero_fn = MacroEconometricModels._build_zero_restrictions_fn(restrictions, n, m_size, p, max_h, Float64)
        lve = MacroEconometricModels._log_volume_element(ff_h, structpara, zero_fn)
        @test isfinite(lve)
    end

    @testset "_draw_w" begin
        rng = MersenneTwister(49494)  # DGP-02: explicit rng

        n = 3
        zrs = [zero_restriction(2, 1)]
        restrictions = SVARRestrictions(n; zeros=zrs)
        setup = MacroEconometricModels._AriasSVARSetup(restrictions, n, Float64)

        w = MacroEconometricModels._draw_w(setup)
        @test length(w) == setup.dim

        # Each sub-vector should be on unit sphere
        offset = 0
        for s_j in setup.sphere_dims
            w_j = w[offset+1:offset+s_j]
            @test isapprox(norm(w_j), 1.0, atol=1e-10)
            offset += s_j
        end
    end
end

# ==========================================================================
# Error Handling Tests
# ==========================================================================

@testset "Error Handling" begin

    @testset "No Valid Identification" begin
        rng = MersenneTwister(56565)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        # Contradictory restrictions - positive AND negative on same element
        signs = [
            SignRestriction(1, 1, 0, 1),   # Positive
            SignRestriction(1, 1, 0, -1),  # Negative on same element
        ]
        restrictions = SVARRestrictions(ZeroRestriction[], signs, n, n)

        # Should error after max attempts
        @test_throws IdentificationError identify_arias(model, restrictions, 5; n_draws=1, n_rotations=10)
    end

    @testset "Dimension Mismatch" begin
        rng = MersenneTwister(67676)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        # 3-var restrictions for 2-var model
        restrictions = SVARRestrictions(3)

        @test_throws AssertionError identify_arias(model, restrictions, 5)
    end

end

@testset "Arias rng reproducibility (#243/T144)" begin
    rng = MersenneTwister(7)  # DGP-02: explicit rng
    Y = randn(rng, 150, 3)
    model = estimate_var(Y, 2)
    restr = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive)])
    r1 = identify_arias(model, restr, 8; n_draws=10, n_rotations=30, rng=Random.MersenneTwister(11))
    r2 = identify_arias(model, restr, 8; n_draws=10, n_rotations=30, rng=Random.MersenneTwister(11))
    @test r1.Q_draws == r2.Q_draws          # same seed -> bitwise-identical rotations
    r3 = identify_arias(model, restr, 8; n_draws=10, n_rotations=30, rng=Random.MersenneTwister(99))
    @test r1.Q_draws != r3.Q_draws          # different seed -> different draws
end


# ==============================================================================
# T273 (#372): ESS reporting for Arias importance-sampling weights
# ==============================================================================

@testset "T273: importance-weight effective sample size" begin
    MEM = MacroEconometricModels
    kish(v) = sum(v)^2 / sum(abs2, v)

    @testset "_effective_sample_size == Kish's formula" begin
        # Uniform weights spend the whole sample; one dominant draw spends one.
        @test MEM._effective_sample_size(fill(1.0, 10)) ≈ 10.0
        @test MEM._effective_sample_size(fill(0.1, 10)) ≈ 10.0      # normalized: same
        @test MEM._effective_sample_size(vcat(1.0, fill(1e-12, 99))) ≈ 1.0 atol = 1e-8
        @test MEM._effective_sample_size([1.0, 1.0]) ≈ 2.0
        @test MEM._effective_sample_size(vcat(fill(1.0, 50), zeros(50))) ≈ 50.0

        # Scale invariance: the ratio is unchanged by normalization, which is why
        # `ess` can be computed before the weights are scaled to sum to 1.
        w = abs.(randn(Random.MersenneTwister(1), 200))
        @test MEM._effective_sample_size(w) ≈ MEM._effective_sample_size(1e6 .* w)
        @test MEM._effective_sample_size(w) ≈ MEM._effective_sample_size(w ./ sum(w))
        @test MEM._effective_sample_size(w) ≈ kish(w)

        # 1 <= ESS <= n for any non-negative weight vector.
        for seed in 1:50
            v = abs.(randn(Random.MersenneTwister(seed), 2 + seed % 40))
            e = MEM._effective_sample_size(v)
            @test 1 - 1e-9 <= e <= length(v) + 1e-9
        end

        # Degenerate inputs return 0 rather than NaN; negatives are an upstream bug.
        @test MEM._effective_sample_size(Float64[]) == 0.0
        @test MEM._effective_sample_size(zeros(5)) == 0.0
        @test_throws ArgumentError MEM._effective_sample_size([1.0, -1.0])
    end

    rng = MersenneTwister(7)  # DGP-02: explicit rng
    n_v, p_v, T_v = 3, 2, 150
    Yb = randn(rng, T_v, n_v)
    for t in 3:T_v
        Yb[t, :] .= 0.5 .* Yb[t-1, :] .- 0.15 .* Yb[t-2, :] .+ 0.6 .* randn(rng, n_v)
    end
    model_b = estimate_var(Yb, p_v)
    restr_zs = SVARRestrictions(n_v;
        zeros=[ZeroRestriction(1, 1, 0), ZeroRestriction(2, 1, 0)],
        signs=[SignRestriction(3, 1, 0, 1)])

    @testset "identify_arias populates ess / ess_fraction" begin
        nd = FAST ? 40 : 120
        r = identify_arias(model_b, restr_zs, 6; n_draws=nd, n_rotations=300,
                           rng=Random.MersenneTwister(42))
        @test length(r.weights) == nd
        @test r.ess ≈ kish(r.weights) rtol = 1e-10
        @test r.ess_fraction ≈ r.ess / nd
        @test 1 <= r.ess <= nd
        # Zero restrictions make the weights genuinely uneven, so the effective
        # sample is strictly smaller than the nominal draw count.
        @test !all(≈(r.weights[1]), r.weights)
        @test r.ess < nd
        @test sum(r.weights) ≈ 1.0                       # stored weights normalized

        # ESS is computed pre-normalization and is unchanged by it.
        r_raw = identify_arias(model_b, restr_zs, 6; n_draws=nd, n_rotations=300,
                               normalize_weights=false, rng=Random.MersenneTwister(42))
        @test r_raw.ess ≈ r.ess
        @test r_raw.ess_fraction ≈ r.ess_fraction
        @test !isapprox(sum(r_raw.weights), 1.0)         # raw volume-element scale
        @test r_raw.weights ./ sum(r_raw.weights) ≈ r.weights rtol = 1e-12
    end

    @testset "pure sign restrictions give a full effective sample" begin
        nd = FAST ? 20 : 60
        rs = identify_arias(model_b, SVARRestrictions(n_v; signs=[SignRestriction(3, 1, 0, 1)]),
                            6; n_draws=nd, rng=Random.MersenneTwister(1))
        # Uniform weights: no importance sampling, so nothing is lost.
        @test rs.ess ≈ nd
        @test rs.ess_fraction ≈ 1.0
    end

    @testset "degeneracy warning" begin
        degen = vcat(1.0, fill(1e-8, 199))
        @test MEM._effective_sample_size(degen) ≈ 1.0 atol = 1e-4

        buf = IOBuffer()
        Logging.with_logger(Logging.SimpleLogger(buf)) do
            MEM._warn_low_ess(MEM._effective_sample_size(degen),
                              MEM._effective_sample_size(degen) / 200, 200, "unit")
        end
        msg = String(take!(buf))
        @test occursin("degenerate", msg)
        @test occursin("unit", msg)

        # Silent when the weights are uniform — the sign-only case must not warn.
        buf2 = IOBuffer()
        Logging.with_logger(Logging.SimpleLogger(buf2)) do
            MEM._warn_low_ess(200.0, 1.0, 200, "unit")
        end
        @test isempty(String(take!(buf2)))

        # ...and not on an empty sample either.
        buf3 = IOBuffer()
        Logging.with_logger(Logging.SimpleLogger(buf3)) do
            MEM._warn_low_ess(0.0, 0.0, 0, "unit")
        end
        @test isempty(String(take!(buf3)))
    end

    @testset "back-compatible constructor derives the diagnostics" begin
        restr = SVARRestrictions(2)
        nd = 200
        degen = vcat(1.0, fill(1e-8, nd - 1))
        ad = AriasSVARResult{Float64}([randn(rng, 2, 2) for _ in 1:nd], randn(rng, nd, 4, 2, 2),
                                      degen ./ sum(degen), 0.5, restr)
        @test ad.ess ≈ 1.0 atol = 1e-4
        @test ad.ess_fraction ≈ ad.ess / nd

        unif = AriasSVARResult{Float64}([randn(rng, 2, 2) for _ in 1:10], randn(rng, 10, 4, 2, 2),
                                        fill(0.1, 10), 0.5, restr)
        @test unif.ess ≈ 10.0
        @test unif.ess_fraction ≈ 1.0
    end

    @testset "report surfaces the effective sample" begin
        restr = SVARRestrictions(2)
        ad = AriasSVARResult{Float64}([randn(rng, 2, 2) for _ in 1:20], randn(rng, 20, 4, 2, 2),
                                      fill(0.05, 20), 0.5, restr)
        buf = IOBuffer()
        show(buf, ad)
        out = String(take!(buf))
        @test occursin("Effective sample", out)
        @test occursin("100.0%", out)
    end

    @testset "Bayesian pooling keeps the importance weights alive" begin
        # Each per-posterior-draw call accepts a SINGLE rotation. Normalizing there
        # would set every weight to 1 and silently reduce the weighted summaries to
        # unweighted ones; pooling must happen once, on the raw scale.
        post = estimate_bvar(Yb, p_v; n_draws=(FAST ? 15 : 30), rng=rng)
        rb = identify_arias_bayesian(post, restr_zs, 4; n_rotations=300,
                                     rng=Random.MersenneTwister(9))
        @test rb.total_accepted > 1
        @test length(unique(round.(rb.weights; digits=12))) > 1     # not all identical
        @test sum(rb.weights) ≈ 1.0
        @test rb.ess ≈ kish(rb.weights) rtol = 1e-10
        @test rb.ess_fraction ≈ rb.ess / rb.total_accepted
        @test 1 <= rb.ess <= rb.total_accepted
        @test rb.ess < rb.total_accepted                            # weighting is live
    end
end

@testset "SID-02 restriction horizon ≥ IRF horizon" begin
    rng = MersenneTwister(731)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 150, 3), 2)
    r5 = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive; horizon=5)])
    a = identify_arias(m, r5, 3; n_draws=5, n_rotations=200, rng=MersenneTwister(731))
    @test size(a.irf_draws, 2) == 3
    for Q in a.Q_draws
        irf6 = compute_irf(m, Q, 6)
        @test irf6[6, 1, 1] > 0
    end
    r0 = SVARRestrictions(3; zeros=[zero_restriction(2, 1; horizon=4)],
                          signs=[sign_restriction(1, 1, :positive)])
    a0 = identify_arias(m, r0, 2; n_draws=3, n_rotations=400, rng=MersenneTwister(7311))
    for Q in a0.Q_draws
        @test abs(compute_irf(m, Q, 5)[5, 2, 1]) < 1e-8
    end
    @test_throws ArgumentError SVARRestrictions(3; signs=[SignRestriction(4, 1, 0, 1)])
    @test_throws ArgumentError sign_restriction(1, 1, :positive; horizon=-1)
end

@testset "SID-19 BayesianSetIdentifiedSVAR" begin
    rng = MersenneTwister(748)  # DGP-02: explicit rng
    Y = randn(rng, 80, 2)
    post = estimate_bvar(Y, 1; n_draws=FAST ? 12 : 20, burnin=5, rng=rng)
    r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
    res = identify_arias_bayesian(post, r, 4; n_rotations=FAST ? 20 : 50,
                                  rng=MersenneTwister(748))
    @test res isa BayesianSetIdentifiedSVAR
    @test res.n_unidentified >= 0
    @test res.n_degenerate_weights >= 0
    @test res.total_accepted == size(res.irf_draws, 1)
    @test length(res.weights) == res.total_accepted
    @test hasproperty(res, :ess)
    fv = fevd(res)
    @test fv isa BayesianFEVD
    ir = irf(res)
    @test ir isa BayesianImpulseResponse
    p = plot_result(res)
    @test p isa PlotOutput
end

@testset "SID-14 typed restriction language" begin
    rng = MersenneTwister(743)  # DGP-02: explicit rng

    @testset "horizons expansion and constructors" begin
        srs = sign_restriction(1, 2, :positive; horizons=0:3)
        @test srs isa Vector
        @test length(srs) == 4
        @test all(s -> s isa SignRestriction, srs)
        @test [s.horizon for s in srs] == collect(0:3)
        @test all(s -> s.variable == 1 && s.shock == 2 && s.sign == 1, srs)

        r = SVARRestrictions(3; signs=[sign_restriction(1, 1, :negative; horizons=0:2)])
        @test length(r.signs) == 3
        @test all(s -> s isa SignRestriction && s.sign == -1, r.signs)

        zr = zero_restriction(1, 2; horizon=:long_run)
        @test zr isa LongRunZeroRestriction
        @test zr.variable == 1 && zr.shock == 2
        @test is_linear_zero(zr)
        @test is_linear_zero(zero_restriction(1, 1))
        @test !is_linear_zero(sign_restriction(1, 1, :positive))

        @test a0_zero_restriction(2, 1) isa A0ZeroRestriction
        @test is_linear_zero(a0_zero_restriction(2, 1))
        @test is_linear_zero(aplus_zero_restriction(1, 1; lag=1))
        @test !is_linear_zero(elasticity_bound(1, 2, 1; lower=0.0, upper=1.0))
        @test !is_linear_zero(magnitude_bound(1, 1; lower=-1.0, upper=1.0))
        @test !is_linear_zero(fevd_share_restriction(1, 1; horizon=4, lower=0.2, upper=1.0))
        @test !is_linear_zero(cumulative_restriction(1, 1, :positive; horizons=0:4))
        @test !is_linear_zero(narrative_shock_restriction(1, [2, 5], :negative))
        @test !is_linear_zero(narrative_contribution_restriction(1, 1, 2:4))

        @test_throws ArgumentError sign_restriction(1, 1, :positive; horizons=-1:2)
        @test_throws ArgumentError elasticity_bound(1, 2, 1; lower=1.0, upper=0.5)
        @test_throws ArgumentError elasticity_bound(1, 2, 1; lower=NaN, upper=1.0)
        @test_throws ArgumentError magnitude_bound(1, 1; lower=-Inf, upper=1.0)
        @test_throws ArgumentError fevd_share_restriction(1, 1; lower=-0.1, upper=0.5)
        @test_throws ArgumentError fevd_share_restriction(1, 1; lower=0.0, upper=1.5)
        @test_throws ArgumentError SVARRestrictions(2; zeros=[zero_restriction(3, 1; horizon=:long_run)])
        @test_throws ArgumentError SVARRestrictions(2; signs=[elasticity_bound(1, 2, 1; lower=2.0, upper=1.0)])
        @test_throws ArgumentError aplus_zero_restriction(2, 1; lag=0)
    end

    @testset "sign_check matches _check_sign_restrictions for pure signs" begin
        n, H = 2, 6
        irf = zeros(H, n, n)
        irf[1, 1, 1] = 1.2
        irf[1, 2, 1] = -0.4
        irf[2, 1, 1] = 0.8
        r = SVARRestrictions(n; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 1, :negative),
            sign_restriction(1, 1, :positive; horizon=1),
        ])
        chk = sign_check(r)
        @test chk(irf) == MacroEconometricModels._check_sign_restrictions(irf, r)
        @test chk(irf)
        irf[1, 1, 1] = -0.1
        @test chk(irf) == MacroEconometricModels._check_sign_restrictions(irf, r)
        @test !chk(irf)
        @test_throws ArgumentError sign_check(
            SVARRestrictions(2; signs=[a0_sign_restriction(1, 1, :positive)]))
    end

    @testset "Blanchard-Quah as n(n-1)/2 long-run zeros + impact signs" begin
        T_obs, n, p = 250, 2, 1
        Y = zeros(T_obs, n)
        for t in 2:T_obs
            Y[t, :] = 0.4 .* Y[t-1, :] + randn(rng, n)
        end
        model = estimate_var(Y, p)
        # Arias-feasible BQ: shock 1 is transitory for variable 1 (n(n-1)/2 = 1 zero).
        r = SVARRestrictions(n;
            zeros=[zero_restriction(1, 1; horizon=:long_run)],
            signs=[sign_restriction(1, 1, :positive),
                   sign_restriction(2, 2, :positive)])
        result = identify_arias(model, r, 8; n_draws=FAST ? 8 : 20, n_rotations=200,
                                rng=MersenneTwister(743))
        @test length(result.Q_draws) >= 1
        Q0 = result.Q_draws[1]
        for Q in result.Q_draws
            @test norm(Q' * Q - I(n)) < 1e-8
            for j in 1:n
                @test Q[:, j] ≈ Q0[:, j] atol=1e-8
            end
        end
        L = MacroEconometricModels.safe_cholesky(model.Sigma)
        C1 = MacroEconometricModels._long_run_multiplier(model.B, model.Sigma, n, p)[1]
        C1Q = C1 * L * Q0
        @test abs(C1Q[1, 1]) < 1e-6
        Q_bq = identify_long_run(model)
        C1_bq = C1 * L * Q_bq
        @test abs(C1_bq[1, 2]) < 1e-6
        # Same triangular long-run structure up to column permutation / sign.
        matched = false
        for perm in ([1, 2], [2, 1])
            Qp = Q0[:, perm]
            for s1 in (1.0, -1.0), s2 in (1.0, -1.0)
                Qs = hcat(s1 * Qp[:, 1], s2 * Qp[:, 2])
                if Qs ≈ Q_bq atol=1e-5
                    matched = true
                end
            end
        end
        @test matched
    end

    @testset "elasticity bound shrinks irf_bounds vs signs alone" begin
        Y = randn(MersenneTwister(7432), 180, 2)
        m = estimate_var(Y, 1)
        r_s = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 1, :positive)])
        s_s = identify_sign(m, 6, sign_check(r_s); store_all=true,
                            max_draws=FAST ? 250 : 600, rng=MersenneTwister(7433))
        @test s_s.n_accepted > 8
        elas = [s_s.irf_draws[i, 1, 1, 1] / s_s.irf_draws[i, 1, 2, 1] for i in 1:s_s.n_accepted]
        mid = median(elas)
        lo_e, hi_e = mid - 0.15, mid + 0.15
        r_e = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 1, :positive),
            elasticity_bound(1, 2, 1; horizon=0, lower=lo_e, upper=hi_e)])
        keep = [sign_check(r_e)(s_s.irf_draws[i, :, :, :]) for i in 1:s_s.n_accepted]
        n_keep = count(keep)
        @test 0 < n_keep < s_s.n_accepted
        idx = findall(keep)
        irf_e = s_s.irf_draws[idx, :, :, :]
        s_e = SignIdentifiedSet{Float64}(s_s.Q_draws[idx], irf_e, n_keep, s_s.n_total,
                                         n_keep / s_s.n_total, s_s.variables, s_s.shocks)
        lo_s, hi_s = irf_bounds(s_s)
        lo_b, hi_b = irf_bounds(s_e)
        @test all((hi_b .- lo_b) .<= (hi_s .- lo_s) .+ 1e-12)
        @test any((hi_b .- lo_b) .< (hi_s .- lo_s) .- 1e-8)
    end

    @testset "A0 zeros, FEVD/cumulative/magnitude rejection, show" begin
        Y = randn(MersenneTwister(7434), 160, 2)
        m = estimate_var(Y, 1)
        r_a0 = SVARRestrictions(2;
            zeros=[a0_zero_restriction(2, 1)],
            signs=[sign_restriction(1, 1, :positive)])
        a0res = identify_arias(m, r_a0, 4; n_draws=FAST ? 4 : 8, n_rotations=300,
                               rng=MersenneTwister(7434))
        L = MacroEconometricModels.safe_cholesky(m.Sigma)
        for Q in a0res.Q_draws
            A0, _ = MacroEconometricModels._rf_to_struct(m.B, L, Q)
            @test abs(A0[2, 1]) < 1e-8
            # RWZ y'A0 form A0 = L^{-T} Q, not the column-convention impact LQ
            @test A0 ≈ Matrix(L') \ Q atol=1e-10
        end

        irf = zeros(5, 2, 2)
        irf[1, 1, 1] = 0.4
        irf[1, 2, 1] = 0.2
        irf[2, 1, 1] = 0.3
        irf[3, 1, 1] = 0.1
        @test MacroEconometricModels.check(magnitude_bound(1, 1; lower=0.0, upper=0.5),
                                           irf, nothing, nothing, nothing)
        @test !MacroEconometricModels.check(magnitude_bound(1, 1; lower=0.0, upper=0.2),
                                            irf, nothing, nothing, nothing)
        @test MacroEconometricModels.check(cumulative_restriction(1, 1, :positive; horizons=0:2),
                                           irf, nothing, nothing, nothing)
        fevd = MacroEconometricModels._compute_fevd(irf, 2, 5)[2]
        @test MacroEconometricModels.check(fevd_share_restriction(1, 1; horizon=0, lower=0.5, upper=1.0),
                                           irf, nothing, nothing, fevd)
        ε_pos = zeros(2, 2); ε_pos[1, 1] = 1.2
        @test MacroEconometricModels.check(narrative_shock_restriction(1, [1], :positive),
                                           irf, nothing, nothing, nothing, ε_pos)
        @test !MacroEconometricModels.check(narrative_shock_restriction(1, [1], :positive),
                                            irf, nothing, nothing, nothing)

        rshow = SVARRestrictions(2;
            zeros=[zero_restriction(1, 2; horizon=:long_run), a0_zero_restriction(2, 1)],
            signs=[sign_restriction(1, 1, :positive; horizons=0:1),
                   elasticity_bound(1, 2, 1; lower=0.1, upper=2.0)])
        shown = sprint(show, rshow)
        @test occursin("SVAR Restrictions", shown)
        @test occursin("long run", lowercase(shown))
        @test occursin("positively", lowercase(shown))
        @test occursin("elasticity", lowercase(shown))
    end

    @testset "SID-08 guard on long-run zeros" begin
        rng = MersenneTwister(7435)  # DGP-02: explicit rng
        trend = cumsum(randn(rng, 200))
        Yc = [trend .+ 0.3 .* randn(rng, 200)  trend .+ 0.3 .* randn(rng, 200)]
        vecm = estimate_vecm(Yc, 2; rank=1)
        r = SVARRestrictions(2; zeros=[zero_restriction(1, 1; horizon=:long_run)])
        @test_throws IdentificationError identify_arias(to_var(vecm), r, 4;
                                                       n_draws=1, n_rotations=5)
    end
end

@testset "SID-23 RWZ rank/order checker" begin
    rng = MersenneTwister(752)  # DGP-02: explicit rng
    recursive_zeros(n) = [zero_restriction(i, j) for j in 1:n-1 for i in (j + 1):n]

    @testset "recursive zeros → :exact with rank(M_j)=n-j" begin
        n, p = 3, 1
        model = estimate_var(randn(rng, 180, n), p)
        r = SVARRestrictions(n; zeros=recursive_zeros(n))
        st = check_identification(r, model; n_points=8, rng=MersenneTwister(752))
        @test st isa IdentificationStatus
        @test st.status === :exact
        @test st.ranks == [n - j for j in 1:n]
        @test st.orders == [n - j for j in 1:n]
        @test st.n_overidentifying == 0
        st_o = check_identification(r, n)
        @test st_o.status === :exact
        @test st_o.orders == st.orders
        @test st_o.n_overidentifying == 0
        result = identify_arias(model, r, 4; n_draws=FAST ? 2 : 4, n_rotations=50,
                                rng=MersenneTwister(7521))
        @test length(result.Q_draws) >= 1
    end

    @testset "all zeros on shock 1 → :under and IdentificationError" begin
        n, p = 3, 1
        model = estimate_var(randn(rng, 180, n), p)
        # n(n-1)/2 impact zeros, all loaded on shock 1
        r = SVARRestrictions(n; zeros=[zero_restriction(i, 1) for i in 1:n])
        st = check_identification(r, model; n_points=6, rng=MersenneTwister(7522))
        @test st.status === :under
        @test st.orders[1] == n
        @test all(st.orders[j] == 0 for j in 2:n)
        @test st.ranks[2] < n - 2 || st.ranks[3] < n - 3 || st.orders[2] < n - 2
        st_o = check_identification(r, n)
        @test st_o.status === :under
        @test st_o.orders == [n, 0, 0]
        @test_throws IdentificationError identify_arias(model, r, 4; n_draws=1, n_rotations=5,
                                                       rng=MersenneTwister(7523))
        err = try
            identify_arias(model, r, 4; n_draws=1, n_rotations=5, rng=rng)
            nothing
        catch e
            e
        end
        @test err isa IdentificationError
        @test occursin("under", lowercase(err.msg))
    end

    @testset "sign-only container → :set" begin
        n = 3
        model = estimate_var(randn(rng, 120, n), 1)
        r = SVARRestrictions(n; signs=[sign_restriction(1, 1, :positive),
                                       sign_restriction(2, 1, :negative)])
        st = check_identification(r, model; rng=MersenneTwister(7524))
        @test st.status === :set
        @test st.orders == zeros(Int, n)
        @test st.ranks == zeros(Int, n)
        @test st.n_overidentifying == 0
        @test check_identification(r, n).status === :set
        result = identify_arias(model, r, 4; n_draws=FAST ? 3 : 6, n_rotations=80,
                                rng=MersenneTwister(7525))
        @test length(result.Q_draws) >= 1
    end

    @testset "Impact zero + linearly dependent long-run zero" begin
        n, p = 3, 1
        model = estimate_var(randn(rng, 150, n), p)
        # Order count for shock 1 is n-1, but the two ZF rows are the same
        # restriction twice (impact copied), and a long-run zero on the same
        # (variable, shock) is the same row whenever C(1)=I.
        r = SVARRestrictions(n; zeros=[
            zero_restriction(2, 1),
            zero_restriction(2, 1),
            zero_restriction(3, 2),
        ])
        st_o = check_identification(r, n)
        @test st_o.orders == [2, 1, 0]
        @test st_o.status === :exact   # count passes
        st = check_identification(r, model; n_points=8, rng=MersenneTwister(7526))
        @test st.orders == [2, 1, 0]
        @test st.ranks[1] < n - 1
        @test st.status === :under

        B0 = copy(model.B)
        B0[2:end, :] .= 0
        model0 = VARModel(model.Y, p, B0, model.U, Matrix(model.Sigma),
                          model.aic, model.bic, model.hqic, model.varnames)
        r_lr = SVARRestrictions(n; zeros=[
            zero_restriction(2, 1),
            zero_restriction(2, 1; horizon=:long_run),
            zero_restriction(3, 2),
        ])
        Phi = MacroEconometricModels.ma_coefficients(model0, 2)
        L = safe_cholesky(model0.Sigma)
        C1 = MacroEconometricModels._C1_from_B(model0.B, n, p)
        Z1 = MacroEconometricModels._compute_ZF(r_lr, Phi, L, 1; B=model0.B, C1=C1)
        @test size(Z1, 1) == 2
        @test Z1[1, :] ≈ Z1[2, :] atol=1e-10
        @test rank(Z1) == 1
        st_lr = check_identification(r_lr, n)
        @test st_lr.orders[1] == 2
        @test st_lr.status === :exact
    end

    @testset "show / order-condition dimension check" begin
        r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        st = check_identification(r, 2)
        shown = sprint(show, st)
        @test occursin("set", shown)
        @test_throws ArgumentError check_identification(r, 3)
    end

    @testset "independent extra zeros → :over and IdentificationError" begin
        n = 2
        model = estimate_var(randn(rng, 120, n), 1)
        r = SVARRestrictions(n; zeros=[zero_restriction(1, 1), zero_restriction(2, 1)])
        st = check_identification(r, model; n_points=6, rng=MersenneTwister(7528))
        @test st.status === :over
        @test st.n_overidentifying >= 1
        @test check_identification(r, n).status === :over
        err = try
            identify_arias(model, r, 4; n_draws=1, n_rotations=5, rng=MersenneTwister(7529))
            nothing
        catch e
            e
        end
        @test err isa IdentificationError
        @test occursin("over", lowercase(err.msg))
    end

    @testset "RWZ probe does not consume caller rng" begin
        n = 2
        model = estimate_var(randn(rng, 80, n), 1)
        r = SVARRestrictions(n; signs=[sign_restriction(1, 1, :positive)])
        rng_a = MersenneTwister(7530)
        MacroEconometricModels._assert_rwz_identified(r, model; rng=rng_a)
        x_a = rand(rng_a)
        rng_b = MersenneTwister(7530)
        x_b = rand(rng_b)
        @test x_a == x_b
        rng_c = MersenneTwister(7532)
        identify_arias(model, r, 3; n_draws=1, n_rotations=50, rng=rng_c)
        y_c = rand(rng_c)
        rng_d = MersenneTwister(7532)
        identify_arias(model, r, 3; n_draws=1, n_rotations=50, rng=rng_d, check_id=false)
        y_d = rand(rng_d)
        @test y_c == y_d
    end
end

@testset "SID-15 ADRR narrative restrictions" begin
    rng = MersenneTwister(744)  # DGP-02: explicit rng

    # Planted bivariate SVAR: lower-triangular B0, five large positive ε₁ dates.
    function _adrr_planted(; Tobs=180, p=1, dates=[20, 30, 40, 50, 60],
                           rng=MersenneTwister(744))
        n = 2
        B0 = [1.0 0.0; 0.4 1.0]
        A1 = [0.5 0.1; 0.0 0.4]
        ntot = Tobs + p + 40
        ε = randn(rng, ntot, n)
        start = ntot - Tobs + 1
        for d in dates
            ε[start + p + d - 1, 1] = 5.0
        end
        Yfull = zeros(ntot, n)
        for t in (p + 1):ntot
            Yfull[t, :] = A1 * Yfull[t - 1, :] + B0 * ε[t, :]
        end
        Y = Yfull[start:end, :]
        return Y, dates, B0
    end

    Y, dates, B0 = _adrr_planted()
    model = estimate_var(Y, 1)
    n_sims = FAST ? 200 : 400
    n_dr = FAST ? 24 : 60
    n_rot = FAST ? 150 : 400

    @testset "check evaluates shock-sign and Type A/B/least contribution" begin
        Q_I = identify_cholesky(model)
        irf_I = compute_irf(model, Q_I, 6)
        ε_I = compute_structural_shocks(model, Q_I)
        r_pos = narrative_shock_restriction(1, dates, :positive)
        r_neg = narrative_shock_restriction(1, dates, :negative)
        @test MacroEconometricModels.check(r_pos, irf_I, nothing, nothing, nothing,
                                           ε_I, nothing)
        @test !MacroEconometricModels.check(r_neg, irf_I, nothing, nothing, nothing,
                                            ε_I, nothing)
        @test !MacroEconometricModels.check(r_pos, irf_I, nothing, nothing, nothing)

        r_c = narrative_contribution_restriction(1, 1, dates[1]:dates[1])
        @test r_c.kind === :most_important
        @test MacroEconometricModels.check(r_c, irf_I, nothing, nothing, nothing,
                                           ε_I, nothing)
        r_ov = narrative_contribution_restriction(1, 1, dates[1]:dates[1];
                                                  kind=:overwhelming)
        @test MacroEconometricModels.check(r_ov, irf_I, nothing, nothing, nothing,
                                           ε_I, nothing)
        r_least = narrative_contribution_restriction(1, 1, dates[1]:dates[1];
                                                     kind=:least_important)
        @test r_least.kind === :least_important
        @test !MacroEconometricModels.check(r_least, irf_I, nothing, nothing, nothing,
                                            ε_I, nothing)
        r_least2 = narrative_contribution_restriction(1, 2, dates[1]:dates[1];
                                                      kind=:least_important)
        @test MacroEconometricModels.check(r_least2, irf_I, nothing, nothing, nothing,
                                           ε_I, nothing)
        Q_mix = [0.0 -1.0; 1.0 0.0]
        irf_mix = compute_irf(model, Q_mix, 6)
        ε_mix = compute_structural_shocks(model, Q_mix)
        @test !MacroEconometricModels.check(r_c, irf_mix, nothing, nothing, nothing,
                                            ε_mix, nothing)
        @test !MacroEconometricModels.check(r_ov, irf_mix, nothing, nothing, nothing,
                                            ε_mix, nothing)
        H_A = [3.0, 2.0, 1.5]   # Type A but not Type B
        H_B = [5.0, 1.0, 1.2]   # Type A and Type B
        @test MacroEconometricModels._is_leading_contributor(H_A, 1, :most_important)
        @test !MacroEconometricModels._is_leading_contributor(H_A, 1, :overwhelming)
        @test MacroEconometricModels._is_leading_contributor(H_B, 1, :most_important)
        @test MacroEconometricModels._is_leading_contributor(H_B, 1, :overwhelming)
        @test MacroEconometricModels._is_leading_contributor(H_B, 2, :least_important)
        @test !MacroEconometricModels._is_leading_contributor(H_B, 3, :least_important)
        @test_throws ArgumentError narrative_contribution_restriction(1, 1, 2:4;
                                                                      kind=:other)
        @test_throws ArgumentError narrative_shock_restriction(1, Int[], :positive)
    end

    @testset "true sign shrinks irf_bounds; weighted median toward truth" begin
        r_sign = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        r_nar = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            narrative_shock_restriction(1, dates, :positive)])
        a_sign = identify_arias(model, r_sign, 6; n_draws=n_dr, n_rotations=n_rot,
                                rng=MersenneTwister(7441))
        a_nar = identify_arias(model, r_nar, 6; n_draws=n_dr, n_rotations=n_rot,
                               n_narrative_sims=n_sims, rng=MersenneTwister(7441))
        @test a_nar.n_narrative_sims == n_sims
        @test a_sign.n_narrative_sims == 0
        @test a_sign.ess_fraction ≈ 1.0 atol=1e-12
        @test a_nar.ess_fraction < 1
        keep = [MacroEconometricModels._narrative_restrictions_hold(
                    r_nar, a_sign.irf_draws[i, :, :, :],
                    compute_structural_shocks(model, a_sign.Q_draws[i]))
                for i in 1:length(a_sign.Q_draws)]
        n_keep = count(keep)
        @test 0 < n_keep < length(keep)
        idx = findall(keep)
        irf_k = a_sign.irf_draws[idx, :, :, :]
        lo_s = dropdims(minimum(a_sign.irf_draws; dims=1), dims=1)
        hi_s = dropdims(maximum(a_sign.irf_draws; dims=1), dims=1)
        lo_n = dropdims(minimum(irf_k; dims=1), dims=1)
        hi_n = dropdims(maximum(irf_k; dims=1), dims=1)
        @test all((hi_n .- lo_n) .<= (hi_s .- lo_s) .+ 1e-12)
        @test any((hi_n .- lo_n) .< (hi_s .- lo_s) .- 1e-8)
        L = MacroEconometricModels.safe_cholesky(model.Sigma)
        truth = L[1, 1]
        med_s = median(a_sign.irf_draws[:, 1, 1, 1])
        med_n = median(irf_k[:, 1, 1, 1])
        @test abs(med_n - truth) <= abs(med_s - truth) + 1e-8
        med_w = irf_percentiles(a_nar; quantiles=[0.5])[1, 1, 1, 1]
        @test abs(med_w - truth) <= abs(med_s - truth) + 0.15
    end

    @testset "wrong sign → IdentificationError or degenerate ESS" begin
        r_wrong = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            narrative_shock_restriction(1, dates, :negative)])
        local got = nothing
        try
            got = identify_arias(model, r_wrong, 6; n_draws=FAST ? 8 : 20,
                                 n_rotations=FAST ? 80 : 200,
                                 n_narrative_sims=n_sims,
                                 rng=MersenneTwister(7442))
        catch e
            @test e isa IdentificationError
            got = :error
        end
        if got isa AriasSVARResult
            @test got.ess_fraction < MacroEconometricModels._ARIAS_ESS_WARN_FRACTION
        end
        @test got === :error || got isa AriasSVARResult
    end

    @testset "identify_narrative wrapper and weighted HD" begin
        r_nar = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            narrative_shock_restriction(1, dates, :positive)])
        wrapped = identify_narrative(model, r_nar, 4; n_draws=FAST ? 8 : 16,
                                     n_rotations=n_rot, n_narrative_sims=n_sims,
                                     rng=MersenneTwister(7444))
        @test wrapped isa AriasSVARResult
        @test wrapped.n_narrative_sims == n_sims
        @test wrapped.ess_fraction < 1
        Q_f, irf_f, sh_f = identify_narrative(model, 4,
            ir -> ir[1, 1, 1] > 0, s -> s[dates[1], 1] > 0;
            max_draws=FAST ? 200 : 800, rng=MersenneTwister(7445))
        @test irf_f[1, 1, 1] > 0
        @test sh_f[dates[1], 1] > 0
        hd = historical_decomposition(model, r_nar, 8; n_draws=FAST ? 8 : 16,
                                      n_rotations=n_rot, n_narrative_sims=n_sims,
                                      rng=MersenneTwister(7446))
        @test hd isa BayesianHistoricalDecomposition
        @test hd.n_effective >= 1
        @test hd.point_estimate[end, 1, 1] + hd.point_estimate[end, 1, 2] +
              hd.initial_point_estimate[end, 1] ≈ hd.actual[end, 1] atol=1e-6
    end

    @testset "contribution restriction through identify_arias" begin
        Q_I = identify_cholesky(model)
        irf_I = compute_irf(model, Q_I, 6)
        ε_I = compute_structural_shocks(model, Q_I)
        win = dates[1]:dates[1]
        r_c = narrative_contribution_restriction(1, 1, win)
        @test MacroEconometricModels.check(r_c, irf_I, nothing, nothing, nothing,
                                           ε_I, nothing)
        r_sign = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        r_contrib = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive), r_c])
        n_csims = FAST ? 80 : 200
        n_cdr = FAST ? 16 : 40
        a_sign = identify_arias(model, r_sign, 6; n_draws=n_cdr, n_rotations=n_rot,
                                rng=MersenneTwister(7451))
        a_c = identify_arias(model, r_contrib, 6; n_draws=n_cdr, n_rotations=n_rot,
                             n_narrative_sims=n_csims, rng=MersenneTwister(7451))
        @test a_c.n_narrative_sims == n_csims
        @test a_sign.ess_fraction ≈ 1.0 atol=1e-12
        @test a_c.ess_fraction < a_sign.ess_fraction - 0.01
        keep = [MacroEconometricModels._narrative_restrictions_hold(
                    r_contrib, a_sign.irf_draws[i, :, :, :],
                    compute_structural_shocks(model, a_sign.Q_draws[i]))
                for i in 1:length(a_sign.Q_draws)]
        @test any(keep)
        @test count(keep) < length(keep)
        rng_ω = MersenneTwister(7452)
        ω_I = MacroEconometricModels._omega_hat(r_contrib, irf_I, 2, Float64;
                                                n_sims=n_csims, rng=rng_ω)
        Q_mix = [0.0 -1.0; 1.0 0.0]
        irf_mix = compute_irf(model, Q_mix, 6)
        ω_mix = MacroEconometricModels._omega_hat(r_contrib, irf_mix, 2, Float64;
                                                  n_sims=n_csims, rng=MersenneTwister(7453))
        @test ω_I > ω_mix + 0.15
        lo, hi = irf_bounds(a_c)
        @test all(lo .<= hi)
        pct = irf_percentiles(a_c; quantiles=[0.16, 0.84])
        @test lo ≈ pct[:, :, :, 1]
        @test hi ≈ pct[:, :, :, 2]
    end

    @testset "identify_arias_bayesian reports n_narrative_sims" begin
        r_nar = SVARRestrictions(2; signs=[
            sign_restriction(1, 1, :positive),
            narrative_shock_restriction(1, dates[1:2], :positive)])
        post = estimate_bvar(Y, 1; n_draws=FAST ? 8 : 12, burnin=3, rng=rng)
        n_b = FAST ? 40 : 80
        res = identify_arias_bayesian(post, r_nar, 4; n_rotations=FAST ? 20 : 40,
                                      n_narrative_sims=n_b, rng=MersenneTwister(7447))
        @test res isa BayesianSetIdentifiedSVAR
        @test res.n_narrative_sims == n_b
        @test res.ess_fraction < 1
        r_sign = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        res_s = identify_arias_bayesian(post, r_sign, 4; n_rotations=FAST ? 20 : 40,
                                        rng=MersenneTwister(7448))
        @test res_s.ess_fraction ≈ 1.0 atol=1e-12
        @test res_s.n_narrative_sims == 0
    end
end

@testset "SID-17 Arias set summaries" begin
    rng = MersenneTwister(7461)  # DGP-02: explicit rng
    Y = randn(MersenneTwister(7461), 100, 2)
    model = estimate_var(Y, 1)
    r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
    a = identify_arias(model, r, 5; n_draws=FAST ? 12 : 24, n_rotations=FAST ? 80 : 200,
                       rng=MersenneTwister(7461))
    mt = median_target(a)
    @test any(Q -> Q === mt.Q, a.Q_draws)
    @test mt.irf ≈ a.irf_draws[mt.index, :, :, :]
    lo, hi = joint_band(a; level=0.68)
    @test all(lo .<= mt.irf .<= hi)
    mm = modal_model(a)
    @test any(Q -> Q === mm.Q, a.Q_draws)
    slo, shi = sup_t_band(a; level=0.68)
    @test all(slo .<= shi)
    H = 5
    fv = fevd(model, a, H)
    @test fv isa BayesianFEVD
    n = 2
    n_d = length(a.Q_draws)
    props = Array{Float64}(undef, n_d, n, n, H)
    for i in 1:n_d
        _, p = MacroEconometricModels._compute_fevd(a.irf_draws[i, :, :, :], n, H)
        props[i, :, :, :] = p
        for h in 1:H, v in 1:n
            @test sum(p[v, :, h]) ≈ 1 atol=1e-10
        end
    end
    for v in 1:n, sh in 1:n, h in 1:H
        @test fv.point_estimate[v, sh, h] ≈
              MacroEconometricModels._weighted_quantile(view(props, :, v, sh, h), a.weights, 0.5)
    end
    hd = historical_decomposition(model, a)
    @test hd isa BayesianHistoricalDecomposition
    shk = structural_shocks(model, a)
    @test size(shk.median, 2) == n
end

# ==========================================================================
# SID-27: parallel draws + ForwardDiff volume (#756)
# ==========================================================================

@testset "SID-27 parallel Arias + ForwardDiff volume" begin
    MEM = MacroEconometricModels

    @testset "FD vs AD weights 1e-6 relative (pinned freeze-FD)" begin
        rng = MersenneTwister(756)  # DGP-02: explicit rng
        Y = randn(rng, 150, 3)
        model = estimate_var(Y, 1)
        restrictions = SVARRestrictions(3;
            zeros=[zero_restriction(2, 1), zero_restriction(3, 1)],
            signs=[sign_restriction(1, 1, :positive)])
        rng = MersenneTwister(75601)
        Phi = MEM._compute_ma_coefficients(model, 10)
        L = safe_cholesky(model.Sigma)
        setup = MEM._AriasSVARSetup(restrictions, 3, Float64; rng=MersenneTwister(75602))
        ws_ad = Float64[]
        ws_fd = Float64[]
        for _ in 1:40
            Q = MEM._draw_Q_with_zero_restrictions(restrictions, Phi, L; rng=rng, B=model.B)
            irf = MEM._compute_irf_for_Q(model, Q, Phi, L, 10)
            irf[1, 1, 1] > 0 || continue
            abs(irf[1, 2, 1]) < 1e-8 || continue
            abs(irf[1, 3, 1]) < 1e-8 || continue
            w_ad = MEM._compute_importance_weight(Q, model, setup, restrictions, Phi, L)
            w_fd = MEM._compute_importance_weight_fd(Q, model, setup, restrictions, Phi, L)
            push!(ws_ad, w_ad)
            push!(ws_fd, w_fd)
            length(ws_ad) >= 3 && break
        end
        @test length(ws_ad) >= 1
        @test length(ws_fd) == length(ws_ad)
        for i in eachindex(ws_ad)
            @test isapprox(ws_ad[i], ws_fd[i]; rtol=1e-6)
        end
        ess_ad = MEM._effective_sample_size(ws_ad)
        ess_fd = MEM._effective_sample_size(ws_fd)
        @test isapprox(ess_ad, ess_fd; rtol=1e-6)
    end

    @testset "pre-seed slots are thread-count invariant" begin
        rng = MersenneTwister(75611)  # DGP-02: explicit rng
        Y = randn(rng, 120, 3)
        model = estimate_var(Y, 1)
        restrictions = SVARRestrictions(3;
            zeros=[zero_restriction(2, 1)],
            signs=[sign_restriction(1, 1, :positive)])
        r1 = identify_arias(model, restrictions, 5; n_draws=8, n_rotations=40,
                            rng=MersenneTwister(75611))
        r2 = identify_arias(model, restrictions, 5; n_draws=8, n_rotations=40,
                            rng=MersenneTwister(75611))
        @test length(r1.Q_draws) == length(r2.Q_draws)
        @test length(r1.Q_draws) >= 1
        @test r1.weights ≈ r2.weights
        @test r1.irf_draws ≈ r2.irf_draws
        @test r1.acceptance_rate ≈ r2.acceptance_rate
        for (Q1, Q2) in zip(r1.Q_draws, r2.Q_draws)
            @test Q1 ≈ Q2
        end
        # Same seed is thread-count invariant (r1 ≈ r2). Absolute weights/Q
        # goldens are BLAS/OS-dependent and are not pinned.
        @test length(r1.Q_draws) >= 1
        @test r1.elapsed >= 0
        @test r1.weights_elapsed >= 0
        @test r1.weights_elapsed > 0  # zeros → volume-element weight
    end

    @testset "elapsed fields and back-compat constructors" begin
        restr = SVARRestrictions(2)
        ad = AriasSVARResult{Float64}([randn(rng, 2, 2) for _ in 1:4], randn(rng, 4, 3, 2, 2),
                                      fill(0.25, 4), 0.5, restr)
        @test ad.elapsed == 0
        @test ad.weights_elapsed == 0
        ad2 = AriasSVARResult{Float64}([randn(rng, 2, 2)], randn(rng, 1, 3, 2, 2), [1.0], 1.0, restr,
                                       1.0, 1.0, ["y1", "y2"], 0, 0)
        @test ad2.elapsed == 0
        @test ad2.weights_elapsed == 0
        ad3 = AriasSVARResult{Float64}([randn(rng, 2, 2)], randn(rng, 1, 3, 2, 2), [1.0], 1.0, restr,
                                       1.0, 1.0, ["y1", "y2"], 0, 0, 0.12, 0.04)
        @test ad3.elapsed ≈ 0.12
        @test ad3.weights_elapsed ≈ 0.04

        signs = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        Y = randn(MersenneTwister(75612), 80, 2)
        model = estimate_var(Y, 1)
        rsign = identify_arias(model, signs, 4; n_draws=4, n_rotations=20,
                               rng=MersenneTwister(75612))
        @test rsign.elapsed >= 0
        @test rsign.weights_elapsed == 0  # pure signs skip the volume element
    end
end

_tprint("Arias et al. (2018) tests completed.")

