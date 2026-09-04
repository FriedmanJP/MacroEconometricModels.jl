# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
    Tests for Mountford & Uhlig (2009) Penalty Function SVAR Identification

Reference:
Mountford, A. & Uhlig, H. (2009). "What Are the Effects of Fiscal Policy Shocks?"
Journal of Applied Econometrics 24(6): 960–992.
"""

using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "Mountford-Uhlig (2009) Penalty Function Identification" begin

    # ==========================================================================
    # Spherical Coordinate Tests
    # ==========================================================================

    @testset "Spherical coordinate unit norm" begin
        for m in [2, 3, 4, 5]
            theta = rand(MersenneTwister(42 + m), m - 1) .* 2π  # DGP-02: explicit rng
            x = MacroEconometricModels._spherical_to_unit_vector(theta, m)
            @test length(x) == m
            @test isapprox(norm(x), 1.0, atol=1e-12)
        end

        # m=1 special case
        x1 = MacroEconometricModels._spherical_to_unit_vector(Float64[], 1)
        @test x1 == [1.0]
    end

    @testset "Spherical coordinates cover full space" begin
        # Different angles should produce different unit vectors
        rng = MersenneTwister(100)  # DGP-02: explicit rng
        m = 3
        vecs = [MacroEconometricModels._spherical_to_unit_vector(rand(m-1) .* 2π, m) for _ in 1:10]
        # Not all the same
        @test !all(v -> isapprox(v, vecs[1], atol=1e-8), vecs[2:end])
    end

    # ==========================================================================
    # Q Orthogonality Tests
    # ==========================================================================

    @testset "Q orthogonality — no zero restrictions" begin
        rng = MersenneTwister(12345)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 2, :positive),
        ]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300), rng=rng)

        # Q should be orthogonal
        @test norm(result.Q' * result.Q - I(n)) < 1e-8
        @test norm(result.Q * result.Q' - I(n)) < 1e-8

        # Columns should be unit vectors
        for j in 1:n
            @test isapprox(norm(result.Q[:, j]), 1.0, atol=1e-8)
        end
    end

    @testset "Q orthogonality — with zero restrictions" begin
        rng = MersenneTwister(23456)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        zeros_r = [zero_restriction(2, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300), rng=rng)

        @test norm(result.Q' * result.Q - I(n)) < 1e-8
        @test norm(result.Q * result.Q' - I(n)) < 1e-8
    end

    # ==========================================================================
    # Zero Restriction Enforcement
    # ==========================================================================

    @testset "Zero restrictions enforced exactly" begin
        rng = MersenneTwister(34567)  # DGP-02: explicit rng

        # Cholesky zeros hold at the truth (lower-triangular B0, DGP-02 #791).
        T_obs, n, p = 200, 3, 1
        Y = dgp_var(rng; T=T_obs).Y
        model = estimate_var(Y, p)

        zeros_r = [
            zero_restriction(2, 1),  # Var 2 doesn't respond to shock 1 on impact
            zero_restriction(3, 1),  # Var 3 doesn't respond to shock 1 on impact
        ]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300), rng=rng)

        # Zero restrictions must be satisfied exactly
        @test abs(result.irf[1, 2, 1]) < 1e-8  # Var 2, Shock 1, impact
        @test abs(result.irf[1, 3, 1]) < 1e-8  # Var 3, Shock 1, impact
    end

    @testset "Zero restrictions at non-zero horizon" begin
        rng = MersenneTwister(45678)  # DGP-02: explicit rng

        # Use n=3 to avoid over-constraining (n=2 with 1 zero on shock 2
        # leaves 0 free dimensions for column 2: 2-1-1=0)
        T_obs, n, p = 200, 3, 2
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        # Zero at horizon 1 on shock 1 (which has no orthogonality constraints)
        zeros_r = [zero_restriction(1, 1; horizon=1)]
        signs = [sign_restriction(2, 2, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300), rng=rng)

        # Zero restriction at h=1 (index 2): var 1, shock 1
        @test abs(result.irf[2, 1, 1]) < 1e-8
    end

    # ==========================================================================
    # Pure Sign Restrictions
    # ==========================================================================

    @testset "Pure sign restrictions — convergence" begin
        rng = MersenneTwister(56789)  # DGP-02: explicit rng

        # Reference DGP (DGP-02 #791): (1,1)+ and (2,2)+ hold at the truth.
        T_obs, n, p = 200, 3, 1
        Y = dgp_var(rng; T=T_obs).Y
        model = estimate_var(Y, p)

        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 2, :positive),
        ]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 15), n_refine=(FAST ? 1 : 3), max_iter_coarse=(FAST ? 50 : 150), max_iter_fine=(FAST ? 100 : 500), rng=rng)

        @test result isa UhligSVARResult
        @test result.converged == true
        @test result.irf[1, 1, 1] > 0
        @test result.irf[1, 2, 2] > 0
        @test isfinite(result.penalty)
    end

    # ==========================================================================
    # Mixed Zero + Sign Restrictions
    # ==========================================================================

    @testset "Mixed zero and sign restrictions" begin
        rng = MersenneTwister(67890)  # DGP-02: explicit rng

        # B0[3,2] < 0: the (3,2)-negative restriction holds at the truth (DGP-02 #791).
        T_obs, n, p = 200, 3, 1
        Y = dgp_var(rng; B0=[1.0 0.0 0.0; 0.5 1.0 0.0; 0.3 -0.2 1.0], T=T_obs).Y
        model = estimate_var(Y, p)

        zeros_r = [zero_restriction(2, 1)]
        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(3, 2, :negative),
        ]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300), rng=rng)

        # Zero restriction must hold
        @test abs(result.irf[1, 2, 1]) < 1e-8

        # Sign restrictions should be satisfied if converged
        if result.converged
            @test result.irf[1, 1, 1] > 0
            @test result.irf[1, 3, 2] < 0
        end
    end

    # ==========================================================================
    # Cholesky Equivalence
    # ==========================================================================

    @testset "Full Cholesky zeros ≈ Cholesky identification" begin
        rng = MersenneTwister(78901)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        # Full lower-triangular zero restrictions
        zeros_r = [
            zero_restriction(2, 1),
            zero_restriction(3, 1),
            zero_restriction(3, 2),
        ]
        # Need at least one sign for penalty function
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300), rng=rng)

        # Cholesky IRF for comparison
        Q_chol = Matrix{Float64}(I, n, n)
        irf_chol = MacroEconometricModels.compute_irf(model, Q_chol, 10)

        # Zero elements enforced
        @test abs(result.irf[1, 2, 1]) < 1e-8
        @test abs(result.irf[1, 3, 1]) < 1e-8
        @test abs(result.irf[1, 3, 2]) < 1e-8

        # Diagonal entries should match Cholesky in absolute value
        for j in 1:n
            @test isapprox(abs(result.irf[1, j, j]), abs(irf_chol[1, j, j]), rtol=0.05)
        end
    end

    # ==========================================================================
    # Consistency with Arias
    # ==========================================================================

    @testset "Uhlig Q satisfies same restrictions as Arias" begin
        rng = MersenneTwister(89012)  # DGP-02: explicit rng

        # Reference DGP (DGP-02 #791).
        T_obs, n, p = 200, 3, 1
        Y = dgp_var(rng; T=T_obs).Y
        model = estimate_var(Y, p)

        zeros_r = [zero_restriction(3, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 15), n_refine=(FAST ? 1 : 3), max_iter_coarse=(FAST ? 50 : 150), max_iter_fine=(FAST ? 100 : 500), rng=rng)

        if result.converged
            # Verify zero restriction
            @test MacroEconometricModels._check_zero_restrictions(result.irf, restrictions)
            # Verify sign restriction
            @test MacroEconometricModels._check_sign_restrictions(result.irf, restrictions)
        end
    end

    # ==========================================================================
    # Edge Cases
    # ==========================================================================

    @testset "n=2 system" begin
        rng = MersenneTwister(90123)  # DGP-02: explicit rng

        T_obs, n, p = 150, 2, 1
        Y = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], B0=[1.0 0.0; 0.3 1.0], T=T_obs).Y
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 5;
            n_starts=(FAST ? 3 : 8), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300), rng=rng)

        @test result isa UhligSVARResult
        @test size(result.Q) == (n, n)
        @test size(result.irf) == (5, n, n)
        @test norm(result.Q' * result.Q - I(n)) < 1e-8
    end

    @testset "No sign restrictions throws error" begin
        rng = MersenneTwister(12321)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        # Only zero restrictions, no signs
        zeros_r = [zero_restriction(2, 1)]
        restrictions = SVARRestrictions(n; zeros=zeros_r)

        @test_throws ArgumentError identify_uhlig(model, restrictions, 5)
    end

    @testset "Dimension mismatch throws error" begin
        rng = MersenneTwister(23232)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        restrictions = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive)])
        @test_throws AssertionError identify_uhlig(model, restrictions, 5)
    end

    @testset "Over-constrained zero restrictions" begin
        rng = MersenneTwister(34343)  # DGP-02: explicit rng

        T_obs, n, p = 100, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        # Two zero restrictions on shock 1 in a 2-var system: leaves 0 free dims
        zeros_r = [
            zero_restriction(1, 1),
            zero_restriction(2, 1),
        ]
        signs = [sign_restriction(1, 2, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        @test_throws IdentificationError identify_uhlig(model, restrictions, 5)
    end

    # ==========================================================================
    # Reproducibility
    # ==========================================================================

    @testset "Reproducibility with same seed" begin
        T_obs, n, p = 150, 2, 1

        rng = MersenneTwister(54321)  # DGP-02: explicit rng
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        # Same explicit rng stream twice → identical results (DGP-02).
        result1 = identify_uhlig(model, restrictions, 5;
            n_starts=3, n_refine=1, max_iter_coarse=50, max_iter_fine=100,
            rng=MersenneTwister(11111))

        result2 = identify_uhlig(model, restrictions, 5;
            n_starts=3, n_refine=1, max_iter_coarse=50, max_iter_fine=100,
            rng=MersenneTwister(11111))

        @test result1.Q ≈ result2.Q
        @test result1.irf ≈ result2.irf
        @test result1.penalty ≈ result2.penalty
    end

    # ==========================================================================
    # Penalty Diagnostics
    # ==========================================================================

    @testset "Penalty values are finite and negative" begin
        rng = MersenneTwister(65432)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 2, :positive),
        ]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300), rng=rng)

        @test isfinite(result.penalty)
        @test result.penalty < 0  # Satisfied restrictions yield large negative penalties

        @test length(result.shock_penalties) == n
        @test all(isfinite, result.shock_penalties)
    end

    # ==========================================================================
    # Display Tests
    # ==========================================================================

    @testset "show() output" begin
        rng = MersenneTwister(76543)  # DGP-02: explicit rng

        T_obs, n, p = 150, 3, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        zeros_r = [zero_restriction(2, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 8), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200), rng=rng)

        io = IOBuffer()
        show(io, result)
        output = String(take!(io))

        @test contains(output, "Mountford-Uhlig")
        @test contains(output, "Variables")
        @test contains(output, "Converged")
        @test contains(output, "Per-Shock")
    end

    @testset "report() dispatches to show()" begin
        rng = MersenneTwister(87654)  # DGP-02: explicit rng

        T_obs, n, p = 150, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 5;
            n_starts=(FAST ? 3 : 5), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200), rng=rng)

        # report() should not error
        io = IOBuffer()
        redirect_stdout(devnull) do
            report(result)
        end
        @test true  # No error
    end

    @testset "refs() output" begin
        rng = MersenneTwister(98765)  # DGP-02: explicit rng

        T_obs, n, p = 150, 2, 1
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 5;
            n_starts=(FAST ? 3 : 5), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200), rng=rng)

        io = IOBuffer()
        refs(io, result)
        output = String(take!(io))

        @test contains(output, "Mountford")
        @test contains(output, "Uhlig")
        @test contains(output, "2009")
    end

    # ==========================================================================
    # _uhlig_n_params Tests
    # ==========================================================================

    @testset "_uhlig_n_params computation" begin
        # 3-var, no zeros:
        #   shock 1: free_dim = 3 - 0 - 0 = 3, angles = 2
        #   shock 2: free_dim = 3 - 1 - 0 = 2, angles = 1
        #   shock 3: free_dim = 3 - 2 - 0 = 1, angles = 0
        # Total = 2 + 1 + 0 = 3
        restrictions = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive)])
        @test MacroEconometricModels._uhlig_n_params(3, restrictions) == 3

        # 3-var, 1 zero on shock 1:
        #   shock 1: free_dim = 3 - 0 - 1 = 2, angles = 1
        #   shock 2: free_dim = 3 - 1 - 0 = 2, angles = 1
        #   shock 3: free_dim = 3 - 2 - 0 = 1, angles = 0
        # Total = 1 + 1 + 0 = 2
        zeros_r = [zero_restriction(2, 1)]
        restrictions2 = SVARRestrictions(3; zeros=zeros_r, signs=[sign_restriction(1, 1, :positive)])
        @test MacroEconometricModels._uhlig_n_params(3, restrictions2) == 2

        # 2-var, no zeros:
        #   shock 1: free_dim = 2 - 0 - 0 = 2, angles = 1
        #   shock 2: free_dim = 2 - 1 - 0 = 1, angles = 0
        # Total = 1
        restrictions3 = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
        @test MacroEconometricModels._uhlig_n_params(2, restrictions3) == 1
    end

    # ==========================================================================
    # Larger System
    # ==========================================================================

    @testset "Larger system (4 variables)" begin
        rng = MersenneTwister(11111)  # DGP-02: explicit rng

        # 4-variable reference DGP (DGP-02 #791); the optimizer enforces the
        # sign pattern below by construction (shock-3 sign flip is free).
        T_obs, n, p = 300, 4, 1
        Y = dgp_var(rng; A=[0.5 0.1 0.0 0.0; 0.0 0.4 0.1 0.0;
                            0.0 0.0 0.35 0.05; 0.05 0.0 0.0 0.3],
                    B0=[1.0 0.0 0.0 0.0; 0.4 1.0 0.0 0.0;
                        0.2 0.3 1.0 0.0; 0.1 0.2 0.3 1.0], T=T_obs).Y
        model = estimate_var(Y, p)

        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 2, :positive),
            sign_restriction(3, 3, :negative),
        ]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 12), n_refine=(FAST ? 1 : 3), max_iter_coarse=(FAST ? 50 : 150), max_iter_fine=(FAST ? 100 : 400), rng=rng)

        @test result isa UhligSVARResult
        @test size(result.irf) == (10, 4, 4)
        @test norm(result.Q' * result.Q - I(n)) < 1e-8

        if result.converged
            @test result.irf[1, 1, 1] > 0
            @test result.irf[1, 2, 2] > 0
            @test result.irf[1, 3, 3] < 0
        end
    end

    # ==========================================================================
    # Numerical Stability
    # ==========================================================================

    @testset "Near-singular covariance doesn't crash" begin
        rng = MersenneTwister(22222)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 1
        Y = randn(rng, T_obs, n)
        Y[:, 3] = Y[:, 1] + 0.01 * randn(rng, T_obs)  # Near-collinear

        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 5;
            n_starts=(FAST ? 3 : 8), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200), rng=rng)

        @test result isa UhligSVARResult
        @test all(isfinite, result.irf)
    end

    # ==========================================================================
    # IRF Structure
    # ==========================================================================

    @testset "IRF dimensions and finiteness" begin
        rng = MersenneTwister(33333)  # DGP-02: explicit rng

        T_obs, n, p = 200, 3, 2
        Y = randn(rng, T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        horizon = 20
        result = identify_uhlig(model, restrictions, horizon;
            n_starts=(FAST ? 3 : 8), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200), rng=rng)

        @test size(result.irf) == (horizon, n, n)
        @test all(isfinite, result.irf)

        # Impact response = Phi[1] * L * Q = I * L * Q = L * Q
        L = MacroEconometricModels.safe_cholesky(model.Sigma)
        expected_impact = L * result.Q
        @test isapprox(result.irf[1, :, :], expected_impact, atol=1e-8)
    end

end

@testset "SID-02 restriction horizon ≥ IRF horizon" begin
    rng = MersenneTwister(731)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 150, 3), 2)
    r5 = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive; horizon=5)])
    u = identify_uhlig(m, r5, 3; n_starts=10, n_refine=2,
                       max_iter_coarse=100, max_iter_fine=300, rng=MersenneTwister(731))
    @test size(u.irf, 1) == 3
    irf6 = compute_irf(m, u.Q, 6)
    @test irf6[6, 1, 1] > 0
    r0 = SVARRestrictions(3; zeros=[zero_restriction(2, 1; horizon=4)],
                          signs=[sign_restriction(1, 1, :positive)])
    u0 = identify_uhlig(m, r0, 2; n_starts=10, n_refine=2,
                        max_iter_coarse=100, max_iter_fine=300, rng=MersenneTwister(7311))
    @test abs(compute_irf(m, u0.Q, 5)[5, 2, 1]) < 1e-8
    @test_throws ArgumentError SVARRestrictions(3; signs=[SignRestriction(4, 1, 0, 1)])
    @test_throws ArgumentError sign_restriction(1, 1, :positive; horizon=-1)
end

@testset "SID-03 Uhlig penalty weights" begin
    # Two :positive restrictions, normalised responses (1.00, 0.01) vs (1.05, -0.05)
    # Satisfied candidate must have the lower (better) penalty.
    # Construct via a tiny helper that injects responses — or call _uhlig_penalty
    # on hand-built Qs once a VAR is estimated.
    # Xoshiro(732) reproduces this testset's historical global-RNG stream
    # exactly (verified); it stays on white noise because the r_uniq block
    # below is seed-sensitive at ~50/50 across streams — see #814 (filed
    # from DGP-02 #791).
    xrng = Xoshiro(732)  # DGP-02: explicit rng
    n = 2
    Y = randn(xrng, 200, n)
    m = estimate_var(Y, 1)
    r = SVARRestrictions(n; signs=[
        sign_restriction(1, 1, :positive),
        sign_restriction(2, 1, :positive),
    ])
    # Direct f(x) table
    f(x; w=100) = x <= 0 ? x : w * x
    # candidate A: both satisfied, normalized = (1.00, 0.01) → x = -normalized
    penA = f(-1.00) + f(-0.01)
    penB = f(-1.05) + f(0.05)
    @test penA < penB
    @test penA ≈ -1.01
    @test penB ≈ 3.95

    # If _uhlig_penalty still uses inverted weights, a violating rotation can score better.
    # After the fix, the satisfied Q has the lower penalty.
    # Negative residual correlation: θ≈0 maximises the first response but violates
    # the second sign; a modest rotation satisfies both with a smaller first response.
    ρ = -0.5
    Sigma = [1.0 ρ; ρ 1.0]
    B = zeros(1 + n * 1, n)
    U = randn(xrng, 199, n)
    m_corr = VARModel(Y, 1, B, U, Sigma, 0.0, 0.0, 0.0)
    horizon = 1
    Phi = MacroEconometricModels._compute_ma_coefficients(m_corr, horizon)
    L = MacroEconometricModels.safe_cholesky(m_corr.Sigma)
    θA = [π / 6 + 0.05]          # both signs satisfied
    θB = [0.0]                   # var 2 on impact is negative
    penA_fn = MacroEconometricModels._uhlig_penalty(θA, r, Phi, L, m_corr, horizon, n)
    penB_fn = MacroEconometricModels._uhlig_penalty(θB, r, Phi, L, m_corr, horizon, n)
    QA = MacroEconometricModels._uhlig_build_Q(θA, r, Phi, L, n)
    QB = MacroEconometricModels._uhlig_build_Q(θB, r, Phi, L, n)
    @test QA[1, 1] > 0 && (MacroEconometricModels.safe_cholesky(m_corr.Sigma) * QA)[2, 1] > 0
    @test QB[1, 1] > 0 && (MacroEconometricModels.safe_cholesky(m_corr.Sigma) * QB)[2, 1] < 0
    @test penA_fn < penB_fn
    penB_heavy = MacroEconometricModels._uhlig_penalty(θB, r, Phi, L, m_corr, horizon, n;
                                                      penalty_weight=1000)
    @test penB_heavy > penB_fn
    spA = MacroEconometricModels._uhlig_shock_penalties(QA, r, Phi, L, m_corr, horizon)
    spB = MacroEconometricModels._uhlig_shock_penalties(QB, r, Phi, L, m_corr, horizon)
    @test spA[1] < spB[1]

    # Unique admissible rotation: zero on (2,1) plus a positive impact sign on (1,1)
    # is Cholesky up to the sign, which the restriction pins.
    r_uniq = SVARRestrictions(n;
        zeros=[zero_restriction(2, 1)],
        signs=[sign_restriction(1, 1, :positive)])
    u = identify_uhlig(m, r_uniq, 5; n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2),
                       max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300),
                       rng=xrng)
    @test u.converged == true
    @test u.irf[1, 1, 1] > 0
    @test abs(u.irf[1, 2, 1]) < 1e-8
    @test all(sr -> sr.sign * u.irf[sr.horizon + 1, sr.variable, sr.shock] > 0,
              r_uniq.signs)

    io = IOBuffer()
    show(io, u)
    @test occursin("lower", lowercase(String(take!(io))))
end

@testset "SID-14 Uhlig rejects non-sign rejection types" begin
    rng = MersenneTwister(743)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 80, 2), 1)
    s1 = sign_restriction(1, 1, :positive)
    mixed = [
        SVARRestrictions(2; signs=[s1, elasticity_bound(1, 2, 1; lower=0.0, upper=1.0)]),
        SVARRestrictions(2; signs=[s1, magnitude_bound(1, 1; lower=-1.0, upper=1.0)]),
        SVARRestrictions(2; signs=[s1, fevd_share_restriction(1, 1; horizon=0, lower=0.2, upper=1.0)]),
        SVARRestrictions(2; signs=[s1, cumulative_restriction(1, 1, :positive; horizons=0:2)]),
        SVARRestrictions(2; signs=[s1, a0_sign_restriction(1, 1, :positive)]),
    ]
    for r in mixed
        err = try
            identify_uhlig(m, r, 4; n_starts=1, n_refine=1, rng=rng)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("SignRestriction", sprint(showerror, err))
    end
    # Linear A0 zeros still go through the null space
    r_a0z = SVARRestrictions(2;
        zeros=[a0_zero_restriction(2, 1)],
        signs=[sign_restriction(1, 1, :positive)])
    u = identify_uhlig(m, r_a0z, 4; n_starts=(FAST ? 3 : 8), n_refine=1,
                       max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200),
                       rng=MersenneTwister(743))
    L = MacroEconometricModels.safe_cholesky(m.Sigma)
    A0, _ = MacroEconometricModels._rf_to_struct(m.B, L, u.Q)
    @test abs(A0[2, 1]) < 1e-8
    @test A0 ≈ Matrix(L') \ u.Q atol=1e-10
end

@testset "SID-19 irf method=:uhlig" begin
    rng = MersenneTwister(748)  # DGP-02: explicit rng
    m = estimate_var(randn(rng, 80, 2), 1)
    r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
    rng = MersenneTwister(748)
    uhlig_kw = (n_starts=FAST ? 3 : 8, n_refine=1, max_iter_coarse=80, max_iter_fine=200)
    u = identify_uhlig(m, r, 5; rng=copy(rng), uhlig_kw...)
    ru = irf(m, 5; method=:uhlig, restrictions=r, rng=copy(rng), uhlig_kw...)
    @test ru.values ≈ u.irf
end

@testset "SID-23 Uhlig RWZ checker" begin
    rng = MersenneTwister(752)  # DGP-02: explicit rng
    n = 3
    m = estimate_var(randn(rng, 120, n), 1)

    @testset "sign-only is :set and report notes the set" begin
        r = SVARRestrictions(n; signs=[sign_restriction(1, 1, :positive)])
        @test check_identification(r, n).status === :set
        u = identify_uhlig(m, r, 4; n_starts=(FAST ? 3 : 6), n_refine=1,
                           max_iter_coarse=(FAST ? 40 : 80), max_iter_fine=(FAST ? 80 : 160),
                           rng=MersenneTwister(7527))
        shown = sprint(show, u)
        @test occursin("set", lowercase(shown))
        @test occursin("point", lowercase(shown))
    end

    @testset "zeros all on shock 1 → IdentificationError" begin
        r = SVARRestrictions(n;
            zeros=[zero_restriction(i, 1) for i in 1:n],
            signs=[sign_restriction(1, 2, :positive)])
        @test check_identification(r, n).status === :under
        @test_throws IdentificationError identify_uhlig(m, r, 4; n_starts=1, n_refine=1)
    end

    @testset "extra independent zeros → :over IdentificationError" begin
        m2 = estimate_var(randn(rng, 80, 2), 1)
        r = SVARRestrictions(2;
            zeros=[zero_restriction(1, 1), zero_restriction(2, 1)],
            signs=[sign_restriction(1, 2, :positive)])
        @test check_identification(r, 2).status === :over
        @test_throws IdentificationError identify_uhlig(m2, r, 4; n_starts=1, n_refine=1)
    end

    @testset "report uses stored rank status, not order-only checker" begin
        r = SVARRestrictions(n; zeros=[
            zero_restriction(2, 1),
            zero_restriction(2, 1),
            zero_restriction(3, 2),
        ], signs=[sign_restriction(1, 1, :positive)])
        @test check_identification(r, n).status === :exact
        st = check_identification(r, m; n_points=8, rng=MersenneTwister(7531))
        @test st.status === :set
        uh = UhligSVARResult{Float64}(Matrix{Float64}(I, n, n), randn(rng, 4, n, n), -1.0,
                                      zeros(n), r, true, ["v$i" for i in 1:n], st)
        @test uh.id_status.status === :set
        shown = sprint(show, uh)
        @test occursin("set", lowercase(shown))
        @test occursin("point", lowercase(shown))
    end
end

_tprint("Mountford-Uhlig (2009) tests completed.")
