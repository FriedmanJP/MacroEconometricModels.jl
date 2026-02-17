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
            Random.seed!(42 + m)
            theta = rand(m - 1) .* 2π
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
        Random.seed!(100)
        m = 3
        vecs = [MacroEconometricModels._spherical_to_unit_vector(rand(m-1) .* 2π, m) for _ in 1:10]
        # Not all the same
        @test !all(v -> isapprox(v, vecs[1], atol=1e-8), vecs[2:end])
    end

    # ==========================================================================
    # Q Orthogonality Tests
    # ==========================================================================

    @testset "Q orthogonality — no zero restrictions" begin
        Random.seed!(12345)

        T_obs, n, p = 200, 3, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 2, :positive),
        ]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300))

        # Q should be orthogonal
        @test norm(result.Q' * result.Q - I(n)) < 1e-8
        @test norm(result.Q * result.Q' - I(n)) < 1e-8

        # Columns should be unit vectors
        for j in 1:n
            @test isapprox(norm(result.Q[:, j]), 1.0, atol=1e-8)
        end
    end

    @testset "Q orthogonality — with zero restrictions" begin
        Random.seed!(23456)

        T_obs, n, p = 200, 3, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        zeros_r = [zero_restriction(2, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300))

        @test norm(result.Q' * result.Q - I(n)) < 1e-8
        @test norm(result.Q * result.Q' - I(n)) < 1e-8
    end

    # ==========================================================================
    # Zero Restriction Enforcement
    # ==========================================================================

    @testset "Zero restrictions enforced exactly" begin
        Random.seed!(34567)

        T_obs, n, p = 200, 3, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        zeros_r = [
            zero_restriction(2, 1),  # Var 2 doesn't respond to shock 1 on impact
            zero_restriction(3, 1),  # Var 3 doesn't respond to shock 1 on impact
        ]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300))

        # Zero restrictions must be satisfied exactly
        @test abs(result.irf[1, 2, 1]) < 1e-8  # Var 2, Shock 1, impact
        @test abs(result.irf[1, 3, 1]) < 1e-8  # Var 3, Shock 1, impact
    end

    @testset "Zero restrictions at non-zero horizon" begin
        Random.seed!(45678)

        # Use n=3 to avoid over-constraining (n=2 with 1 zero on shock 2
        # leaves 0 free dimensions for column 2: 2-1-1=0)
        T_obs, n, p = 200, 3, 2
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        # Zero at horizon 1 on shock 1 (which has no orthogonality constraints)
        zeros_r = [zero_restriction(1, 1; horizon=1)]
        signs = [sign_restriction(2, 2, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300))

        # Zero restriction at h=1 (index 2): var 1, shock 1
        @test abs(result.irf[2, 1, 1]) < 1e-8
    end

    # ==========================================================================
    # Pure Sign Restrictions
    # ==========================================================================

    @testset "Pure sign restrictions — convergence" begin
        Random.seed!(56789)

        T_obs, n, p = 200, 3, 1
        Y = zeros(T_obs, n)
        for t in 2:T_obs
            Y[t, :] = 0.5 * Y[t-1, :] + randn(n)
        end
        model = estimate_var(Y, p)

        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 2, :positive),
        ]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 15), n_refine=(FAST ? 1 : 3), max_iter_coarse=(FAST ? 50 : 150), max_iter_fine=(FAST ? 100 : 500))

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
        Random.seed!(67890)

        T_obs, n, p = 200, 3, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        zeros_r = [zero_restriction(2, 1)]
        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(3, 2, :negative),
        ]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 15), n_refine=(FAST ? 1 : 3), max_iter_coarse=(FAST ? 50 : 150), max_iter_fine=(FAST ? 100 : 500))

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
        Random.seed!(78901)

        T_obs, n, p = 200, 3, 1
        Y = randn(T_obs, n)
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
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300))

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
        Random.seed!(89012)

        T_obs, n, p = 200, 3, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        zeros_r = [zero_restriction(3, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 15), n_refine=(FAST ? 1 : 3), max_iter_coarse=(FAST ? 50 : 150), max_iter_fine=(FAST ? 100 : 500))

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
        Random.seed!(90123)

        T_obs, n, p = 150, 2, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 5;
            n_starts=(FAST ? 3 : 8), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300))

        @test result isa UhligSVARResult
        @test size(result.Q) == (n, n)
        @test size(result.irf) == (5, n, n)
        @test norm(result.Q' * result.Q - I(n)) < 1e-8
    end

    @testset "No sign restrictions throws error" begin
        Random.seed!(12321)

        T_obs, n, p = 100, 2, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        # Only zero restrictions, no signs
        zeros_r = [zero_restriction(2, 1)]
        restrictions = SVARRestrictions(n; zeros=zeros_r)

        @test_throws ArgumentError identify_uhlig(model, restrictions, 5)
    end

    @testset "Dimension mismatch throws error" begin
        Random.seed!(23232)

        T_obs, n, p = 100, 2, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        restrictions = SVARRestrictions(3; signs=[sign_restriction(1, 1, :positive)])
        @test_throws AssertionError identify_uhlig(model, restrictions, 5)
    end

    @testset "Over-constrained zero restrictions" begin
        Random.seed!(34343)

        T_obs, n, p = 100, 2, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        # Two zero restrictions on shock 1 in a 2-var system: leaves 0 free dims
        zeros_r = [
            zero_restriction(1, 1),
            zero_restriction(2, 1),
        ]
        signs = [sign_restriction(1, 2, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        @test_throws ErrorException identify_uhlig(model, restrictions, 5)
    end

    # ==========================================================================
    # Reproducibility
    # ==========================================================================

    @testset "Reproducibility with same seed" begin
        T_obs, n, p = 150, 2, 1

        Random.seed!(54321)
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        Random.seed!(11111)
        result1 = identify_uhlig(model, restrictions, 5;
            n_starts=8, n_refine=2, max_iter_coarse=100, max_iter_fine=300)

        Random.seed!(11111)
        result2 = identify_uhlig(model, restrictions, 5;
            n_starts=8, n_refine=2, max_iter_coarse=100, max_iter_fine=300)

        @test result1.Q ≈ result2.Q
        @test result1.irf ≈ result2.irf
        @test result1.penalty ≈ result2.penalty
    end

    # ==========================================================================
    # Penalty Diagnostics
    # ==========================================================================

    @testset "Penalty values are finite and negative" begin
        Random.seed!(65432)

        T_obs, n, p = 200, 3, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 2, :positive),
        ]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 10), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 300))

        @test isfinite(result.penalty)
        @test result.penalty < 0  # Satisfied restrictions yield large negative penalties

        @test length(result.shock_penalties) == n
        @test all(isfinite, result.shock_penalties)
    end

    # ==========================================================================
    # Display Tests
    # ==========================================================================

    @testset "show() output" begin
        Random.seed!(76543)

        T_obs, n, p = 150, 3, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        zeros_r = [zero_restriction(2, 1)]
        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; zeros=zeros_r, signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 8), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200))

        io = IOBuffer()
        show(io, result)
        output = String(take!(io))

        @test contains(output, "Mountford-Uhlig")
        @test contains(output, "Variables")
        @test contains(output, "Converged")
        @test contains(output, "Per-Shock")
    end

    @testset "report() dispatches to show()" begin
        Random.seed!(87654)

        T_obs, n, p = 150, 2, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 5;
            n_starts=(FAST ? 3 : 5), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200))

        # report() should not error
        io = IOBuffer()
        redirect_stdout(devnull) do
            report(result)
        end
        @test true  # No error
    end

    @testset "refs() output" begin
        Random.seed!(98765)

        T_obs, n, p = 150, 2, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 5;
            n_starts=(FAST ? 3 : 5), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200))

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
        Random.seed!(11111)

        T_obs, n, p = 300, 4, 1
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        signs = [
            sign_restriction(1, 1, :positive),
            sign_restriction(2, 2, :positive),
            sign_restriction(3, 3, :negative),
        ]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 10;
            n_starts=(FAST ? 3 : 12), n_refine=(FAST ? 1 : 3), max_iter_coarse=(FAST ? 50 : 150), max_iter_fine=(FAST ? 100 : 400))

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
        Random.seed!(22222)

        T_obs, n, p = 200, 3, 1
        Y = randn(T_obs, n)
        Y[:, 3] = Y[:, 1] + 0.01 * randn(T_obs)  # Near-collinear

        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        result = identify_uhlig(model, restrictions, 5;
            n_starts=(FAST ? 3 : 8), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200))

        @test result isa UhligSVARResult
        @test all(isfinite, result.irf)
    end

    # ==========================================================================
    # IRF Structure
    # ==========================================================================

    @testset "IRF dimensions and finiteness" begin
        Random.seed!(33333)

        T_obs, n, p = 200, 3, 2
        Y = randn(T_obs, n)
        model = estimate_var(Y, p)

        signs = [sign_restriction(1, 1, :positive)]
        restrictions = SVARRestrictions(n; signs=signs)

        horizon = 20
        result = identify_uhlig(model, restrictions, horizon;
            n_starts=(FAST ? 3 : 8), n_refine=(FAST ? 1 : 2), max_iter_coarse=(FAST ? 50 : 100), max_iter_fine=(FAST ? 100 : 200))

        @test size(result.irf) == (horizon, n, n)
        @test all(isfinite, result.irf)

        # Impact response = Phi[1] * L * Q = I * L * Q = L * Q
        L = MacroEconometricModels.safe_cholesky(model.Sigma)
        expected_impact = L * result.Q
        @test isapprox(result.irf[1, :, :], expected_impact, atol=1e-8)
    end

end

_tprint("Mountford-Uhlig (2009) tests completed.")
