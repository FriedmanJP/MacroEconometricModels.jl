# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Unit tests for DSGE analytical moments, GMM observable subsetting, and
HA-DSGE Reiter aggregate observation map (#874, #876).
"""

using LinearAlgebra
using MacroEconometricModels
using Random
using Test

@testset "DSGE Analytical Moments & GMM Observable Subsetting (#874)" begin
    # Setup standard 2-variable model: y, pi
    spec_dsge = @dsge begin
        parameters: rho = 0.7, phi = 0.5
        endogenous: y, pi
        exogenous: e_y, e_pi
        y[t] = rho * y[t-1] + e_y[t]
        pi[t] = phi * y[t] + e_pi[t]
    end
    spec_ss = compute_steady_state(spec_dsge)
    sol = solve(spec_ss; method=:gensys)

    @testset "sol.Sigma Dynamic Property Access" begin
        @test hasproperty(sol, :Sigma)
        @test sol.Sigma ≈ solve_lyapunov(sol.G1, sol.impact) atol=1e-12
        @test :Sigma in propertynames(sol)
    end

    @testset "Observable Subsetting Mathematical Invariants" begin
        # Dimension 1: Full subsetting identity vs unsubsetted moments
        m_full = analytical_moments(sol; lags=1)
        m_sub_all = analytical_moments(sol; lags=1, observable_indices=[1, 2])
        @test length(m_full) == 3 + 2  # 3 cov + 2 acov
        @test length(m_sub_all) == 5
        @test m_sub_all ≈ m_full atol=1e-12

        # Dimension 2: Single observable subsetting [1]
        m_sub1 = analytical_moments(sol; lags=1, observable_indices=[1])
        @test length(m_sub1) == 2  # 1 var + 1 acov
        @test m_sub1[1] ≈ sol.Sigma[1, 1] atol=1e-12
        @test m_sub1[1] > 0.0

        # Single observable subsetting [2]
        m_sub2 = analytical_moments(sol; lags=1, observable_indices=[2])
        @test length(m_sub2) == 2
        @test m_sub2[1] ≈ sol.Sigma[2, 2] atol=1e-12
        @test m_sub2[1] > 0.0

        # Multi-lag subsetting
        m_sub1_lags3 = analytical_moments(sol; lags=3, observable_indices=[1])
        @test length(m_sub1_lags3) == 4  # 1 var + 3 acovs
        @test m_sub1_lags3[1] ≈ m_sub1[1] atol=1e-12

        # Zero lag boundary (contemporaneous only)
        m_lag0 = analytical_moments(sol; lags=0, observable_indices=[1])
        @test length(m_lag0) == 1
        @test m_lag0[1] ≈ sol.Sigma[1, 1] atol=1e-12
    end

    @testset "Validation & Error Paths" begin
        # Out-of-bounds observable index throws BoundsError
        @test_throws BoundsError analytical_moments(
            sol; lags=1, observable_indices=[0],
        )
        @test_throws BoundsError analytical_moments(
            sol; lags=1, observable_indices=[3],
        )
        @test_throws BoundsError analytical_moments(
            sol; lags=1, observable_indices=[1, 4],
        )

        # Empty observable indices throws ArgumentError
        @test_throws ArgumentError analytical_moments(
            sol; lags=1, observable_indices=Int[],
        )

        # Negative lags throws ArgumentError
        @test_throws ArgumentError analytical_moments(
            sol; lags=-1, observable_indices=[1],
        )
    end

    @testset "PerturbationSolution Observable Subsetting" begin
        pert_sol = solve(spec_ss; method=:perturbation, order=1)
        m_pert_sub1 = analytical_moments(
            pert_sol; lags=1, observable_indices=[1],
        )
        @test length(m_pert_sub1) == 2
        @test m_pert_sub1[1] ≈ sol.Sigma[1, 1] atol=1e-10

        # PerturbationSolution gmm format
        m_pert_gmm = analytical_moments(
            pert_sol; lags=1, format=:gmm, observable_indices=[1],
        )
        # format=:gmm produces 1 mean + 1 product moment + 1 acov = 3
        @test length(m_pert_gmm) == 3
        @test isapprox(m_pert_gmm[1], 0.0; atol=1e-10)  # zero mean
        @test m_pert_gmm[2] ≈ sol.Sigma[1, 1] atol=1e-10
    end

    @testset "GMM Parameter Recovery with Observable Subsetting" begin
        rng = Xoshiro(42)
        sim_data = simulate(sol, 200; rng=rng)
        # Pre-subsetted data: pass only column 1 (y)
        data_y = sim_data[:, [1]]

        est = estimate_dsge(
            spec_ss,
            data_y,
            [:rho];
            method=:analytical_gmm,
            observable_indices=[1],
            bounds=Dict(:rho => (0.2, 0.95)),
        )

        @test est isa DSGEEstimation{Float64}
        @test est.method === :analytical_gmm
        @test est.converged
        @test abs(est.theta[1] - 0.7) < 0.2
        @test all(isfinite, est.vcov)

        # Full-dimension data with observable subsetting [1]
        est_full_data = estimate_dsge(
            spec_ss,
            sim_data,
            [:rho];
            method=:analytical_gmm,
            observable_indices=[1],
            bounds=Dict(:rho => (0.2, 0.95)),
        )
        # Full-dimension data with unsubsetted / natural order [1, 2]
        est_natural = estimate_dsge(
            spec_ss,
            sim_data,
            [:rho];
            method=:analytical_gmm,
            observable_indices=[1, 2],
            bounds=Dict(:rho => (0.2, 0.95)),
        )
        @test est_natural.converged
        @test abs(est_natural.theta[1] - 0.7) < 0.2

        # Full-dimension data with permuted observable indices [2, 1]
        est_perm = estimate_dsge(
            spec_ss,
            sim_data,
            [:rho];
            method=:analytical_gmm,
            observable_indices=[2, 1],
            bounds=Dict(:rho => (0.2, 0.95)),
        )
        @test est_perm isa DSGEEstimation{Float64}
        @test est_perm.converged
        @test abs(est_perm.theta[1] - 0.7) < 0.2
        @test all(isfinite, est_perm.vcov)
        @test est_perm.theta[1] ≈ est_natural.theta[1] atol=1e-8
    end
end

@testset "HA-DSGE Reiter Aggregate Observation Map (#876)" begin
    ha_spec = load_ha_example(:krusell_smith)
    ha_ss = compute_steady_state(
        ha_spec;
        r_bounds=(-0.02, 0.04),
        max_iter=60,
        tol=1e-3,
    )
    reiter_sol = solve(
        ha_spec;
        method=:reiter,
        ss=ha_ss,
        n_reduced=8,
    )

    @testset "Reiter Solution and Aggregate Names" begin
        @test reiter_sol isa HADSGESolution{Float64}
        @test reiter_sol.method === :reiter
        @test reiter_sol.linear_solution isa DSGESolution{Float64}
        @test hasproperty(reiter_sol, :variables)
        @test reiter_sol.variables == ["K"]
        @test :variables in propertynames(reiter_sol)

        ir = irf(reiter_sol, 10)
        @test ir.variables == ["K"]
        @test !occursin("x_1", ir.variables[1])
        @test ir.shocks == ["epsilon"]

        # Huggett model dynamic property returns ["r"]
        huggett_spec = load_ha_example(:huggett)
        huggett_ss = compute_steady_state(
            huggett_spec;
            r_bounds=(-0.05, 0.03),
            max_iter=60,
            tol=1e-3,
        )
        huggett_sol = solve(
            huggett_spec;
            method=:reiter,
            ss=huggett_ss,
            n_reduced=6,
        )
        @test hasproperty(huggett_sol, :variables)
        @test huggett_sol.variables == ["r"]
        @test :variables in propertynames(huggett_sol)
    end

    @testset "Non-Zero Impact at h=1 and Dynamic Continuity" begin
        ir = irf(reiter_sol, 10)
        @test size(ir.values) == (10, 1, 1)

        # Non-zero impact at h=1
        @test !iszero(ir.values[1, 1, 1])
        @test isfinite(ir.values[1, 1, 1])
        @test ir.values[1, 1, 1] ≈ reiter_sol.D_obs[1, 1] atol=1e-10

        # Horizon h=2 continuity matching C_obs * impact
        expected_h2 = (reiter_sol.C_obs * reiter_sol.linear_solution.impact)[1, 1]
        @test ir.values[2, 1, 1] ≈ expected_h2 atol=1e-10
    end

    @testset "Boundary Conditions & Validation" begin
        # Horizon H=1 boundary returns impact-only slice
        ir_h1 = irf(reiter_sol, 1)
        @test size(ir_h1.values) == (1, 1, 1)
        @test ir_h1.values[1, 1, 1] ≈ reiter_sol.D_obs[1, 1] atol=1e-10

        # Non-positive horizon throws ArgumentError
        @test_throws ArgumentError irf(reiter_sol, 0)
        @test_throws ArgumentError irf(reiter_sol, -5)

        # Simulation with unit shock reproduces IRF bit-for-bit
        sim_unit = simulate(
            reiter_sol,
            10;
            shock_draws=reshape([1.0; zeros(9)], 10, 1),
        )
        ir_comp = irf(reiter_sol, 10)
        @test sim_unit[:, 1] ≈ ir_comp.values[:, 1, 1] atol=1e-8

        # FEVD for single shock is identically 1.0
        fv = fevd(reiter_sol, 10)
        @test fv.variables == ["K"]
        @test all(==(1.0), fv.proportions)
    end
end
