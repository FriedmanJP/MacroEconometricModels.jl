# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _suppress_warnings = MacroEconometricModels._suppress_warnings

@testset "ABCD Representation & Minimal State Extraction (#909)" begin

    # =========================================================================
    # 1. ABCDSystem construction, validation, queries, display
    # =========================================================================
    @testset "ABCDSystem struct, constructors, and queries" begin
        A = [0.8 0.1; 0.0 0.5]
        B = [1.0; 0.5;;]
        C = [1.0 0.0; 0.0 1.0]
        D = [0.2; 0.3;;]
        sys = ABCDSystem(A, B, C, D)

        @test sys isa ABCDSystem{Float64}
        @test nstates(sys) == 2
        @test nobservables(sys) == 2
        @test nvars(sys) == 2
        @test nshocks(sys) == 1
        @test is_square(sys) == false
        @test is_stable(sys) == true
        @test sys.c_state == [0.0, 0.0]
        @test sys.d_obs == [0.0, 0.0]
        @test sys.state_names == ["x_1", "x_2"]
        @test sys.obs_names == ["y_1", "y_2"]
        @test sys.shock_names == ["w_1"]

        # Integer inputs promote to Float64
        sys_int = ABCDSystem([0 0; 0 0], [1 0; 0 1], [1 0; 0 1], [1 0; 0 1])
        @test sys_int isa ABCDSystem{Float64}
        @test is_square(sys_int) == true
        @test is_stable(sys_int) == true

        # Custom intercepts & labels (including Symbols)
        sys_kw = ABCDSystem(A, B, C, D;
            c_state=[0.1, -0.2], d_obs=[1.0, 2.0],
            state_names=[:k, :z], obs_names=[:y, :c], shock_names=[:eps_z])
        @test sys_kw.c_state ≈ [0.1, -0.2]
        @test sys_kw.d_obs ≈ [1.0, 2.0]
        @test sys_kw.state_names == ["k", "z"]
        @test sys_kw.obs_names == ["y", "c"]
        @test sys_kw.shock_names == ["eps_z"]

        # Unstable / unit-root checks
        sys_ur = ABCDSystem([1.0;;], [1.0;;], [1.0;;], [0.5;;])
        @test is_stable(sys_ur) == false
        @test is_square(sys_ur) == true

        # Zero-state (n == 0) system
        sys_zero = ABCDSystem(zeros(0, 0), zeros(0, 2), zeros(2, 0), [1.0 0.0; 0.0 2.0])
        @test nstates(sys_zero) == 0
        @test nobservables(sys_zero) == 2
        @test nshocks(sys_zero) == 2
        @test is_square(sys_zero) == true
        @test is_stable(sys_zero) == true
        ir0 = irf(sys_zero, 5)
        @test ir0.values[1, :, :] ≈ [1.0 0.0; 0.0 2.0]
        @test all(iszero, ir0.values[2:end, :, :])
        fv0 = fevd(sys_zero, 5)
        @test fv0.proportions[:, :, 1] ≈ [1.0 0.0; 0.0 1.0]
        mom0 = analytical_moments(sys_zero; lags=2, format=:matrices)
        @test length(mom0) == 3
        @test mom0[1] ≈ [1.0 0.0; 0.0 4.0]
        @test all(iszero, mom0[2])
        @test all(iszero, mom0[3])

        # Dimension mismatch validation
        @test_throws ArgumentError ABCDSystem(zeros(2, 3), zeros(2, 1), zeros(1, 2), zeros(1, 1))
        @test_throws ArgumentError ABCDSystem(zeros(2, 2), zeros(3, 1), zeros(1, 2), zeros(1, 1))
        @test_throws ArgumentError ABCDSystem(zeros(2, 2), zeros(2, 1), zeros(1, 3), zeros(1, 1))
        @test_throws ArgumentError ABCDSystem(zeros(2, 2), zeros(2, 1), zeros(1, 2), zeros(2, 1))
        @test_throws ArgumentError ABCDSystem(zeros(2, 2), zeros(2, 1), zeros(1, 2), zeros(1, 1); c_state=[1.0])
        @test_throws ArgumentError ABCDSystem(zeros(2, 2), zeros(2, 1), zeros(1, 2), zeros(1, 1); d_obs=[1.0, 2.0])
        @test_throws ArgumentError ABCDSystem(zeros(2, 2), zeros(2, 1), zeros(1, 2), zeros(1, 1); state_names=["a"])
        @test_throws ArgumentError ABCDSystem(zeros(2, 2), zeros(2, 1), zeros(1, 2), zeros(1, 1); obs_names=["a", "b"])
        @test_throws ArgumentError ABCDSystem(zeros(2, 2), zeros(2, 1), zeros(1, 2), zeros(1, 1); shock_names=["a", "b"])

        # show and report
        io = IOBuffer()
        show(io, sys)
        s = String(take!(io))
        @test occursin("ABCDSystem{Float64}", s)
        @test occursin("n_states=2", s)
        @test occursin("n_obs=2", s)
        @test occursin("n_shocks=1", s)

        report(io, sys)
        r = String(take!(io))
        @test occursin("ABCD State-Space System", r)
        @test occursin("States (n)", r)

        report(io, sys_zero)
        r0 = String(take!(io))
        @test occursin("ABCD State-Space System", r0)
    end

    # =========================================================================
    # 2. FVRSW (2007, §II eqs. 13-14) Permanent Income Model
    # =========================================================================
    @testset "FVRSW (2007) Permanent Income Model" begin
        spec_fv = @dsge begin
            parameters: R_gross = 1.2, sigma_w = 1.0
            endogenous: y, c, y_m_c
            exogenous: w

            c[t] = c[t-1] + sigma_w * (1 - 1/R_gross) * w[t]
            y_m_c[t] = -c[t-1] + sigma_w * (1/R_gross) * w[t]
            y_m_c[t] = y[t] - c[t]

            steady_state = [0.0, 0.0, 0.0]
        end
        sol_fv = solve(spec_fv; method=:gensys)
        @test is_determined(sol_fv)

        abcd = abcd_representation(sol_fv, [:y_m_c])
        @test nstates(abcd) == 1
        @test nobservables(abcd) == 1
        @test nshocks(abcd) == 1
        @test is_square(abcd) == true
        @test is_stable(abcd) == false
        @test abcd.state_names == ["c"]
        @test abcd.obs_names == ["y_m_c"]
        @test abcd.shock_names == ["w"]
        @test abcd.A ≈ [1.0;;] atol=1e-10
        @test abcd.B ≈ [1.0 - 1.0/1.2;;] atol=1e-10
        @test abcd.C ≈ [-1.0;;] atol=1e-10
        @test abcd.D ≈ [1.0/1.2;;] atol=1e-10

        # Also test :surplus naming variant
        spec_surplus = @dsge begin
            parameters: R_gross = 1.2, sigma_w = 1.0
            endogenous: y, c, surplus
            exogenous: w

            c[t] = c[t-1] + sigma_w * (1 - 1/R_gross) * w[t]
            surplus[t] = -c[t-1] + sigma_w * (1/R_gross) * w[t]
            surplus[t] = y[t] - c[t]

            steady_state = [0.0, 0.0, 0.0]
        end
        sol_surplus = solve(spec_surplus; method=:gensys)
        abcd_surplus = abcd_representation(sol_surplus, [:surplus])
        @test nstates(abcd_surplus) == 1
        @test abcd_surplus.state_names == ["c"]
        @test abcd_surplus.A ≈ [1.0;;] atol=1e-10
        @test abcd_surplus.B ≈ [1.0 - 1.0/1.2;;] atol=1e-10
        @test abcd_surplus.C ≈ [-1.0;;] atol=1e-10
        @test abcd_surplus.D ≈ [1.0/1.2;;] atol=1e-10

        # IRF parity (< 1e-12) against DSGESolution
        ir_abcd = irf(abcd, 8)
        ir_sol = irf(sol_fv, 8)
        @test ir_abcd.values[:, 1, 1] ≈ ir_sol.values[:, 3, 1] atol=1e-12

        # All-variable ABCD parity (IRF and FEVD) and String/Int selectors
        abcd_all = abcd_representation(sol_fv)
        @test nobservables(abcd_all) == 3
        @test irf(abcd_all, 8).values ≈ ir_sol.values atol=1e-12
        @test fevd(abcd_all, 8).proportions ≈ fevd(sol_fv, 8).proportions atol=1e-12

        abcd_str = abcd_representation(sol_fv, ["y_m_c"])
        abcd_int = abcd_representation(sol_fv, [3])
        @test abcd_str.A ≈ abcd.A
        @test abcd_int.C ≈ abcd.C

        # reduction=:none preserves all 3 states
        abcd_none = abcd_representation(sol_fv, [:y_m_c]; reduction=:none)
        @test nstates(abcd_none) == 3
        @test irf(abcd_none, 8).values ≈ ir_abcd.values atol=1e-12

        # Non-stationary ABCD throws in analytical_moments
        @test_throws ArgumentError analytical_moments(abcd)
    end

    # =========================================================================
    # 3. Pre-linearized ModelSpec(...; linear=true) (Smets-Wouters style)
    # =========================================================================
    @testset "Pre-linearized ModelSpec (linear=true)" begin
        ctrend = 0.45
        spec_lin = ModelSpec{Float64}(
            [:y, :dy, :a], [:eps], [:rho, :ctrend_p],
            Dict{Symbol,Float64}(:rho => 0.85, :ctrend_p => ctrend),
            Expr[:(0 + 0) for _ in 1:3],
            Function[
                (yt, yl, yle, e, th) -> yt[1] - th[:rho] * yl[1] - e[1],
                (yt, yl, yle, e, th) -> yt[2] - yt[1] + yl[1] - th[:ctrend_p],
                (yt, yl, yle, e, th) -> yt[3] - th[:rho] * yl[3] - e[1],
            ],
            0, Int[], Float64[], nothing; linear=true
        )
        sol_lin = solve(spec_lin; method=:gensys)
        @test is_determined(sol_lin)

        abcd_lin = abcd_representation(sol_lin, [:dy, :y])
        @test nstates(abcd_lin) == 2  # y and a have nonzero G1 columns; dy does not
        @test nobservables(abcd_lin) == 2
        @test irf(abcd_lin, 12).values ≈ irf(sol_lin, 12).values[:, [2, 1], :] atol=1e-12

        # Effective steady state E[y] = d_obs + C * (I - A)^{-1} * c_state matches (I - G1) \ C_sol
        eff_ss_sol = ((I - sol_lin.G1) \ sol_lin.C_sol)[[2, 1]]
        eff_ss_abcd = abcd_lin.d_obs + abcd_lin.C * ((I - abcd_lin.A) \ abcd_lin.c_state)
        @test eff_ss_abcd ≈ [ctrend, 0.0] atol=1e-10
        @test eff_ss_abcd ≈ eff_ss_sol atol=1e-10

        # Minimal reduction removes duplicate/redundant state a!
        abcd_lin_min = abcd_representation(sol_lin, [:dy, :y]; reduction=:minimal)
        @test nstates(abcd_lin_min) == 1
        @test irf(abcd_lin_min, 12).values ≈ irf(abcd_lin, 12).values atol=1e-10
        @test analytical_moments(abcd_lin_min; lags=2, format=:matrices) ≈
              analytical_moments(abcd_lin; lags=2, format=:matrices) atol=1e-10

        # Forward-looking linear=true model with constant
        spec_fwd = ModelSpec{Float64}(
            [:y], [:eps], [:a, :b, :c],
            Dict{Symbol,Float64}(:a => 0.4, :b => 0.35, :c => 1.2),
            Expr[:(0 + 0)],
            Function[(yt, yl, yle, e, th) -> yt[1] - th[:a] * yle[1] - th[:b] * yl[1] - th[:c] - e[1]],
            1, [1], Float64[], nothing; linear=true,
        )
        sol_fwd = solve(spec_fwd; method=:gensys)
        abcd_fwd = abcd_representation(sol_fwd)
        @test nstates(abcd_fwd) == 1
        eff_fwd = (abcd_fwd.d_obs + abcd_fwd.C * ((I - abcd_fwd.A) \ abcd_fwd.c_state))[1]
        @test eff_fwd ≈ ((I - sol_fwd.G1) \ sol_fwd.C_sol)[1] atol=1e-12
        @test eff_fwd ≈ 1.2 / (1 - 0.4 - 0.35) atol=1e-7
        @test irf(abcd_fwd, 10).values ≈ irf(sol_fwd, 10).values atol=1e-12
        @test analytical_moments(abcd_fwd; lags=2) ≈ analytical_moments(sol_fwd; lags=2) atol=1e-10
    end

    # =========================================================================
    # 4. NK 3-equation (DSGESolution & PerturbationSolution order 1)
    # =========================================================================
    @testset "NK 3-equation: DSGESolution and PerturbationSolution (order 1)" begin
        # 4a. Purely forward-looking 3-eq NK (zero predetermined states)
        spec_nk3 = compute_steady_state(@dsge begin
            parameters: sigma_y = 1.0, beta = 0.99, kappa = 0.15, phi_pi = 1.5
            endogenous: y, pi_v, i_rate
            exogenous: eps_m
            linear: true

            y[t] = y[t+1] - sigma_y * (i_rate[t] - pi_v[t+1])
            pi_v[t] = beta * pi_v[t+1] + kappa * y[t]
            i_rate[t] = phi_pi * pi_v[t] + eps_m[t]
        end)
        sol_nk3 = solve(spec_nk3; method=:gensys)
        psol_nk3 = perturbation_solver(spec_nk3; order=1)

        abcd_d3 = abcd_representation(sol_nk3)
        abcd_p3 = abcd_representation(psol_nk3)
        @test nstates(abcd_d3) == 0
        @test nstates(abcd_p3) == 0
        @test is_stable(abcd_d3) == true
        @test irf(abcd_d3, 8).values ≈ irf(sol_nk3, 8).values atol=1e-12
        @test irf(abcd_p3, 8).values ≈ irf(psol_nk3, 8).values atol=1e-12
        @test analytical_moments(abcd_d3; lags=2) ≈ analytical_moments(sol_nk3; lags=2) atol=1e-10
        @test analytical_moments(abcd_p3; lags=2) ≈ analytical_moments(psol_nk3; lags=2) atol=1e-10

        # 4b. NK with persistence + interest-rate smoothing (2 predetermined states)
        spec_nk = compute_steady_state(@dsge begin
            parameters: rho = 0.9, sigma_y = 0.5, kappa = 0.3, phi_pi = 1.5, rho_r = 0.7
            endogenous: y, pi_v, i_rate, a
            exogenous: eps_y
            linear: true

            y[t] = y[t+1] - sigma_y * (i_rate[t] - pi_v[t+1])
            pi_v[t] = 0.99 * pi_v[t+1] + kappa * y[t]
            i_rate[t] = rho_r * i_rate[t-1] + (1.0 - rho_r) * phi_pi * pi_v[t] + a[t]
            a[t] = rho * a[t-1] + eps_y[t]
        end)
        sol_nk = solve(spec_nk; method=:gensys)
        psol_nk = perturbation_solver(spec_nk; order=1)

        abcd_dsge = abcd_representation(sol_nk)
        abcd_pert = abcd_representation(psol_nk)
        @test nstates(abcd_dsge) == 2
        @test nstates(abcd_pert) == 2
        @test nobservables(abcd_dsge) == 4
        @test nobservables(abcd_pert) == 4

        @test irf(abcd_dsge, 12).values ≈ irf(sol_nk, 12).values atol=1e-10
        @test irf(abcd_pert, 12).values ≈ irf(psol_nk, 12).values atol=1e-10
        @test fevd(abcd_dsge, 12).proportions ≈ fevd(sol_nk, 12).proportions atol=1e-10
        @test fevd(abcd_pert, 12).proportions ≈ fevd(psol_nk, 12).proportions atol=1e-10
        @test analytical_moments(abcd_dsge; lags=1:3) ≈ analytical_moments(sol_nk; lags=1:3) atol=1e-9
        @test analytical_moments(abcd_pert; lags=1:3) ≈ analytical_moments(psol_nk; lags=1:3) atol=1e-9

        # Subset observables & reduction=:none on PerturbationSolution
        abcd_sub = abcd_representation(psol_nk, [:y, :pi_v])
        @test nobservables(abcd_sub) == 2
        @test abcd_sub.obs_names == ["y", "pi_v"]
        @test irf(abcd_sub, 10).values ≈ irf(psol_nk, 10).values[:, 1:2, :] atol=1e-10

        abcd_pert_none = abcd_representation(psol_nk; reduction=:none)
        @test nstates(abcd_pert_none) == 4
        @test irf(abcd_pert_none, 10).values ≈ irf(psol_nk, 10).values atol=1e-10

        # Augmented (2-lag) model defaults to original_endog
        spec_aug = compute_steady_state(@dsge begin
            parameters: rho1 = 0.6, rho2 = 0.2
            endogenous: y, c
            exogenous: eps
            linear: true

            y[t] = rho1 * y[t-1] + rho2 * y[t-2] + eps[t]
            c[t] = 0.5 * y[t]
        end)
        sol_aug = solve(spec_aug; method=:gensys)
        psol_aug = perturbation_solver(spec_aug; order=1)
        abcd_aug_d = abcd_representation(sol_aug)
        abcd_aug_p = abcd_representation(psol_aug)
        @test abcd_aug_d.obs_names == ["y", "c"]
        @test abcd_aug_p.obs_names == ["y", "c"]
        @test irf(abcd_aug_d, 10).values ≈ irf(sol_aug, 10).values atol=1e-10
        @test irf(abcd_aug_p, 10).values ≈ irf(psol_aug, 10).values atol=1e-10
    end

    # =========================================================================
    # 5. HADSGESolution (load_ha_example(:krusell_smith))
    # =========================================================================
    @testset "HADSGESolution (Krusell-Smith SSJ)" begin
        ha_spec = load_ha_example(:krusell_smith)
        ha_ss = compute_steady_state(ha_spec; max_iter=100, tol=1e-4)
        ha_sol = solve(ha_spec; method=:ssj, ss=ha_ss, T_horizon=50, n_reduced=10)

        abcd_ha = abcd_representation(ha_sol)
        @test abcd_ha isa ABCDSystem{Float64}
        @test abcd_ha.obs_names == ["K"]
        @test is_square(abcd_ha) == true
        @test nstates(abcd_ha) == size(ha_sol.linear_solution.G1, 1)

        ir_ha = irf(ha_sol, 20)
        ir_abcd = irf(abcd_ha, 20)
        @test ir_abcd.values ≈ ir_ha.values atol=1e-12
        @test fevd(abcd_ha, 20).proportions ≈ fevd(ha_sol, 20).proportions atol=1e-12

        # Minimal reduction on HA system preserves IRF
        abcd_ha_min = abcd_representation(ha_sol; reduction=:minimal, tol=1e-12)
        @test nstates(abcd_ha_min) <= nstates(abcd_ha)
        @test irf(abcd_ha_min, 20).values ≈ ir_ha.values atol=1e-8

        # Measurement error on HA system
        abcd_ha_me = abcd_representation(ha_sol, [:K]; measurement_error=[0.02])
        @test nshocks(abcd_ha_me) == 2
        @test abcd_ha_me.shock_names[end] == "me_K"
        @test abcd_ha_me.D[1, 2] ≈ 0.02

        # Fallback to linear_solution.spec.endog symbols
        abcd_ha_red = abcd_representation(ha_sol, [:x_1, :x_2])
        @test abcd_ha_red.obs_names == ["x_1", "x_2"]
        @test irf(abcd_ha_red, 10).values ≈ irf(ha_sol.linear_solution, 10).values[:, 1:2, :] atol=1e-12

        # Synthetic HA lag-1 feedthrough test (C_obs nonzero where G1 column is zero)
        n_syn = 2
        dummy_spec = ModelSpec{Float64}(
            [:x_1, :x_2], [:eps], Symbol[], Dict{Symbol,Float64}(),
            [:(0+0), :(0+0)], [((yt, yl, yle, e, th) -> 0.0) for _ in 1:2],
            0, Int[], zeros(2), nothing
        )
        G1_syn = [0.0 0.0; 0.0 0.8]
        imp_syn = [1.0; 0.5;;]
        ld_syn = LinearDSGE{Float64}(Matrix{Float64}(I, 2, 2), copy(G1_syn), zeros(2), copy(imp_syn), zeros(2, 0), dummy_spec)
        lsol_syn = DSGESolution{Float64}(G1_syn, imp_syn, zeros(2), [1, 1], :ssj, ComplexF64.(eigvals(G1_syn)), dummy_spec, ld_syn)
        C_syn = [0.7 1.0]
        D_syn = [0.2;;]
        ha_syn = HADSGESolution{Float64}(ha_ss, lsol_syn, :ssj, ha_spec, Matrix{Float64}(I, 2, 2), 2, 2, 1.0, nothing, C_syn, D_syn)
        abcd_syn = abcd_representation(ha_syn)
        @test irf(abcd_syn, 8).values ≈ irf(ha_syn, 8).values atol=1e-12
    end

    # =========================================================================
    # 6. StateSpaceModel (G = nothing and G !== nothing)
    # =========================================================================
    @testset "StateSpaceModel: uncorrelated (G=nothing) and correlated (G!=nothing)" begin
        # 6a. Uncorrelated (G = nothing), rank-deficient RQR, nonzero intercepts
        Z = [1.0 0.5; 0.0 1.0]
        H = [0.04 0.0; 0.0 0.0]   # partially zero observation noise
        Tt = [0.75 0.1; -0.05 0.6]
        R = [1.0; 0.25;;]
        Q = [0.5;;]
        c_vec = [0.1, -0.2]
        d_vec = [1.5, -0.5]
        ss_uncorr = StateSpaceModel(Z, H, Tt, Q; d=d_vec, c=c_vec, R=R)

        abcd_u = abcd_representation(ss_uncorr)
        @test abcd_u.A ≈ Tt atol=1e-12
        @test abcd_u.C ≈ Z atol=1e-12
        @test abcd_u.c_state ≈ c_vec atol=1e-12
        @test abcd_u.d_obs ≈ d_vec atol=1e-12
        @test abcd_u.obs_names == ["obs_1", "obs_2"]
        @test abcd_u.B * abcd_u.B' ≈ R * Q * R' atol=1e-12
        @test abcd_u.D * abcd_u.D' ≈ H atol=1e-12
        @test abcd_u.B * abcd_u.D' ≈ zeros(2, 2) atol=1e-12

        # Exact autocovariances match theoretical StateSpaceModel moments
        Sigma_alpha = MacroEconometricModels._lyapunov_from_cov(Tt, R * Q * R')
        Gamma0_true = Z * Sigma_alpha * Z' + H
        Gamma1_true = Z * (Tt * Sigma_alpha * Z')
        Gamma2_true = Z * Tt * (Tt * Sigma_alpha * Z')
        moms_u = analytical_moments(abcd_u; lags=2, format=:matrices)
        @test moms_u[1] ≈ Gamma0_true atol=1e-11
        @test moms_u[2] ≈ Gamma1_true atol=1e-11
        @test moms_u[3] ≈ Gamma2_true atol=1e-11

        # 6b. Correlated (G !== nothing), non-identity R (n_state × r)
        rng = MersenneTwister(909)
        n_s, r_s, k_s = 3, 2, 2
        Tt_c = [0.6 0.1 0.0; 0.0 0.5 -0.1; 0.05 0.0 0.4]
        Z_c = [1.0 0.0 0.5; 0.0 1.0 -0.2]
        R_c = [1.0 0.0; 0.3 1.0; 0.0 0.5]
        F_joint = randn(rng, r_s + k_s, r_s + k_s)
        Sig_joint = F_joint * F_joint' + 0.1 * I
        Q_c = Sig_joint[1:r_s, 1:r_s]
        G_c = Sig_joint[1:r_s, r_s+1:end]
        H_c = Sig_joint[r_s+1:end, r_s+1:end]
        c_c = [0.2, -0.1, 0.05]
        d_c = [0.5, -0.3]

        ss_corr = StateSpaceModel(Z_c, H_c, Tt_c, Q_c; d=d_c, c=c_c, R=R_c, G=G_c)
        abcd_c = abcd_representation(ss_corr)
        @test abcd_c.A ≈ Tt_c atol=1e-12
        @test abcd_c.C ≈ Z_c atol=1e-12
        @test abcd_c.c_state ≈ c_c atol=1e-12
        @test abcd_c.d_obs ≈ d_c atol=1e-12
        @test abcd_c.B * abcd_c.B' ≈ R_c * Q_c * R_c' atol=1e-11
        @test abcd_c.D * abcd_c.D' ≈ H_c atol=1e-11
        @test abcd_c.B * abcd_c.D' ≈ R_c * G_c atol=1e-11

        Sigma_c = MacroEconometricModels._lyapunov_from_cov(Tt_c, R_c * Q_c * R_c')
        G0_c_true = Z_c * Sigma_c * Z_c' + H_c
        G1_c_true = Z_c * (Tt_c * Sigma_c * Z_c' + R_c * G_c)
        G2_c_true = Z_c * Tt_c * (Tt_c * Sigma_c * Z_c' + R_c * G_c)
        moms_c = analytical_moments(abcd_c; lags=2, format=:matrices)
        @test moms_c[1] ≈ G0_c_true atol=1e-11
        @test moms_c[2] ≈ G1_c_true atol=1e-11
        @test moms_c[3] ≈ G2_c_true atol=1e-11

        # Subset observables on StateSpaceModel
        abcd_c_sub = abcd_representation(ss_corr, [:obs_2])
        @test nobservables(abcd_c_sub) == 1
        @test abcd_c_sub.obs_names == ["obs_2"]
        @test abcd_c_sub.D * abcd_c_sub.D' ≈ H_c[2:2, 2:2] atol=1e-11
        @test abcd_c_sub.B * abcd_c_sub.D' ≈ (R_c * G_c)[:, 2:2] atol=1e-11

        # Minimal reduction on StateSpaceModel
        abcd_c_min = abcd_representation(ss_corr; reduction=:minimal)
        @test analytical_moments(abcd_c_min; lags=2, format=:matrices) ≈ moms_c atol=1e-10

        # Predetermined reduction on StateSpaceModel keeps lag-1 Z-active state and drops dead state
        ss_pred = StateSpaceModel([1.0 0.5 0.0], [0.05;;], [0.8 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [1.0 0.0; 0.0 1.0];
                                  R=[1.0 0.0; 0.0 1.0; 0.0 0.0])
        abcd_pred = abcd_representation(ss_pred; reduction=:predetermined)
        @test nstates(abcd_pred) == 2
        @test abcd_pred.state_names == ["x_1", "x_2"]

        # Non-PSD joint covariance raises ArgumentError
        ss_bad = StateSpaceModel([1.0;;], [0.01;;], [0.5;;], [0.01;;]; G=[1.0;;])
        @test_throws ArgumentError abcd_representation(ss_bad)
    end

    # =========================================================================
    # 7. Minimal Staircase Reduction (_abcd_minimal_realization & :minimal)
    # =========================================================================
    @testset "Minimal staircase realization" begin
        # 4-state system: mode 1 minimal, mode 2 uncontrollable, mode 3 unobservable, mode 4 both
        A_diag = diagm([0.8, 0.6, 0.4, 0.2])
        B_diag = [1.0; 0.0; 1.0; 0.0;;]
        C_diag = [1.0 1.0 0.0 0.0]
        D_diag = [0.25;;]

        # Rotate by a non-trivial orthogonal matrix to test non-diagonal staircase
        rng = MersenneTwister(42)
        Q_orth = Matrix(qr(randn(rng, 4, 4)).Q)
        A_rot = Q_orth * A_diag * Q_orth'
        B_rot = Q_orth * B_diag
        C_rot = C_diag * Q_orth'

        A_m, B_m, C_m, names_m, U_tot = MacroEconometricModels._abcd_minimal_realization(
            A_rot, B_rot, C_rot, ["s1", "s2", "s3", "s4"]; tol=1e-10
        )
        @test size(A_m) == (1, 1)
        @test size(B_m) == (1, 1)
        @test size(C_m) == (1, 1)
        @test names_m == ["x_1"]
        @test eigvals(A_m)[1] ≈ 0.8 atol=1e-10
        @test U_tot' * U_tot ≈ [1.0;;] atol=1e-12

        sys_full = ABCDSystem(A_rot, B_rot, C_rot, D_diag)
        sys_min = ABCDSystem(A_m, B_m, C_m, D_diag; state_names=names_m)
        @test irf(sys_min, 15).values ≈ irf(sys_full, 15).values atol=1e-10
        @test analytical_moments(sys_min; lags=3, format=:matrices) ≈
              analytical_moments(sys_full; lags=3, format=:matrices) atol=1e-10

        # 2-state controllable & observable coupled chain inside 4-state system
        A_chain = [0.7 0.3 0.0 0.0; 0.0 0.5 0.0 0.0; 0.0 0.0 0.4 0.0; 0.0 0.0 0.0 0.2]
        B_chain = [0.0; 1.0; 0.0; 1.0;;]
        C_chain = [1.0 0.0 1.0 0.0]
        A_cm, B_cm, C_cm, nm_cm, _ = MacroEconometricModels._abcd_minimal_realization(
            Q_orth * A_chain * Q_orth', Q_orth * B_chain, C_chain * Q_orth',
            ["s1", "s2", "s3", "s4"]
        )
        @test size(A_cm) == (2, 2)
        @test nm_cm == ["x_1", "x_2"]
        @test sort(real.(eigvals(A_cm))) ≈ [0.5, 0.7] atol=1e-10
        sys_chain_full = ABCDSystem(Q_orth * A_chain * Q_orth', Q_orth * B_chain, C_chain * Q_orth', [0.1;;])
        sys_chain_min = ABCDSystem(A_cm, B_cm, C_cm, [0.1;;]; state_names=nm_cm)
        @test irf(sys_chain_min, 12).values ≈ irf(sys_chain_full, 12).values atol=1e-10

        # Zero-rank (completely uncontrollable) collapses to 0 states
        A_0, B_0, C_0, nm_0, U_0 = MacroEconometricModels._abcd_minimal_realization(
            A_diag, zeros(4, 1), C_diag, ["s1", "s2", "s3", "s4"]
        )
        @test size(A_0) == (0, 0)
        @test size(B_0) == (0, 1)
        @test size(C_0) == (1, 0)
        @test nm_0 == String[]
        @test size(U_0) == (4, 0)

        # Zero-rank (controllable but completely unobservable) collapses to 0 states
        A_0b, B_0b, C_0b, nm_0b, _ = MacroEconometricModels._abcd_minimal_realization(
            A_diag, B_diag, zeros(2, 4), ["s1", "s2", "s3", "s4"]
        )
        @test size(A_0b) == (0, 0)
        @test size(B_0b) == (0, 1)
        @test size(C_0b) == (2, 0)
        @test nm_0b == String[]
    end

    # =========================================================================
    # 8. Measurement Error Augmentation
    # =========================================================================
    @testset "Measurement error augmentation" begin
        spec_nk = compute_steady_state(@dsge begin
            parameters: rho = 0.85, sigma_y = 0.5, kappa = 0.3, phi_pi = 1.5, rho_r = 0.6
            endogenous: y, pi_v, i_rate, a
            exogenous: eps_y
            linear: true

            y[t] = y[t+1] - sigma_y * (i_rate[t] - pi_v[t+1])
            pi_v[t] = 0.99 * pi_v[t+1] + kappa * y[t]
            i_rate[t] = rho_r * i_rate[t-1] + (1.0 - rho_r) * phi_pi * pi_v[t] + a[t]
            a[t] = rho * a[t-1] + eps_y[t]
        end)
        sol_nk = solve(spec_nk; method=:gensys)
        psol_nk = perturbation_solver(spec_nk; order=1)

        abcd_base = abcd_representation(sol_nk, [:y, :pi_v])
        m0_base = analytical_moments(abcd_base; lags=2, format=:matrices)

        # Vector SDs
        me_vec = [0.05, 0.12]
        abcd_me_vec = abcd_representation(sol_nk, [:y, :pi_v]; measurement_error=me_vec)
        @test nshocks(abcd_me_vec) == 3
        @test abcd_me_vec.shock_names == ["eps_y", "me_y", "me_pi_v"]
        @test all(iszero, abcd_me_vec.B[:, 2:3])
        @test abcd_me_vec.D[:, 2:3] ≈ Diagonal(me_vec)
        m0_vec = analytical_moments(abcd_me_vec; lags=2, format=:matrices)
        @test m0_vec[1] ≈ m0_base[1] + Diagonal(me_vec .^ 2) atol=1e-11
        @test m0_vec[2] ≈ m0_base[2] atol=1e-11
        @test m0_vec[3] ≈ m0_base[3] atol=1e-11

        # Correlated k × k measurement error covariance matrix
        me_cov = [0.04 0.015; 0.015 0.09]
        abcd_me_mat = abcd_representation(psol_nk, [:y, :pi_v]; measurement_error=me_cov)
        @test nshocks(abcd_me_mat) == 3
        @test abcd_me_mat.shock_names == ["eps_y", "me_y", "me_pi_v"]
        @test abcd_me_mat.D[:, 2:3] * abcd_me_mat.D[:, 2:3]' ≈ me_cov atol=1e-12
        m0_mat = analytical_moments(abcd_me_mat; lags=1, format=:matrices)
        @test m0_mat[1] ≈ m0_base[1] + me_cov atol=1e-10

        # Validation checks
        @test_throws ArgumentError abcd_representation(sol_nk, [:y, :pi_v]; measurement_error=[0.1])
        @test_throws ArgumentError abcd_representation(sol_nk, [:y, :pi_v]; measurement_error=[-0.1, 0.1])
        @test_throws ArgumentError abcd_representation(sol_nk, [:y, :pi_v]; measurement_error=ones(3, 3))
        @test_throws ArgumentError abcd_representation(sol_nk, [:y, :pi_v]; measurement_error=[0.1 0.5; 0.0 0.1])
        @test_throws ArgumentError abcd_representation(sol_nk, [:y, :pi_v]; measurement_error=:auto)
    end

    # =========================================================================
    # 9. analytical_moments, fevd, Float32, and error handling
    # =========================================================================
    @testset "analytical_moments, fevd, Float32, and errors" begin
        A = [0.7 0.2; -0.1 0.5]
        B = [1.0 0.0; 0.2 0.8]
        C = [1.0 0.5; 0.0 1.0]
        D = [0.5 0.1; 0.0 0.4]
        sys = ABCDSystem(A, B, C, D)

        # Format consistency (:vector vs :matrices) and non-contiguous lags
        m_vec0 = analytical_moments(sys; lags=0, format=:vector)
        m_mat0 = analytical_moments(sys; lags=0, format=:matrices)
        @test length(m_vec0) == 3
        @test length(m_mat0) == 1
        @test m_vec0 ≈ [m_mat0[1][1, 1], m_mat0[1][1, 2], m_mat0[1][2, 2]]

        m_vec_nc = analytical_moments(sys; lags=[1, 3], format=:vector)
        m_mat_3 = analytical_moments(sys; lags=3, format=:matrices)
        @test length(m_mat_3) == 4
        @test m_vec_nc ≈ [
            m_mat_3[1][1, 1], m_mat_3[1][1, 2], m_mat_3[1][2, 2],
            m_mat_3[2][1, 1], m_mat_3[2][2, 2],
            m_mat_3[4][1, 1], m_mat_3[4][2, 2],
        ]

        # FEVD row normalization
        fv = fevd(sys, 10)
        @test size(fv.decomposition) == (2, 2, 10)
        @test size(fv.proportions) == (2, 2, 10)
        for h in 1:10, i in 1:2
            @test sum(fv.proportions[i, :, h]) ≈ 1.0 atol=1e-12
        end

        # Float32 precision preservation across struct & all dispatches
        A32 = Float32[0.7 0.1; 0.0 0.5]
        B32 = Float32[1.0; 0.3;;]
        C32 = Float32[1.0 0.2; 0.0 1.0]
        D32 = Float32[0.4; 0.2;;]
        sys32 = ABCDSystem(A32, B32, C32, D32)
        @test sys32 isa ABCDSystem{Float32}
        @test irf(sys32, 6) isa ImpulseResponse{Float32}
        @test fevd(sys32, 6) isa FEVD{Float32}
        @test analytical_moments(sys32; lags=1) isa Vector{Float32}
        @test analytical_moments(sys32; lags=1, format=:matrices) isa Vector{Matrix{Float32}}

        spec32 = ModelSpec{Float32}(
            [:y, :pi_v, :i_rate, :a], [:eps_y],
            [:rho, :sigma_y, :kappa, :phi_pi, :rho_r],
            Dict{Symbol,Float32}(:rho => 0.9f0, :sigma_y => 0.5f0, :kappa => 0.3f0, :phi_pi => 1.5f0, :rho_r => 0.7f0),
            [:(0+0) for _ in 1:4],
            Function[
                (yt, yl, yle, e, th) -> yt[1] - yle[1] + th[:sigma_y] * (yt[3] - yle[2]),
                (yt, yl, yle, e, th) -> yt[2] - 0.99f0 * yle[2] - th[:kappa] * yt[1],
                (yt, yl, yle, e, th) -> yt[3] - th[:rho_r] * yl[3] - (1.0f0 - th[:rho_r]) * th[:phi_pi] * yt[2] - yt[4],
                (yt, yl, yle, e, th) -> yt[4] - th[:rho] * yl[4] - e[1],
            ],
            2, [1, 2], Float32[0, 0, 0, 0]; linear=true
        )
        sol32 = solve(spec32; method=:gensys)
        psol32 = perturbation_solver(spec32; order=1)
        abcd_d32 = abcd_representation(sol32)
        abcd_p32 = abcd_representation(psol32)
        @test abcd_d32 isa ABCDSystem{Float32}
        @test abcd_p32 isa ABCDSystem{Float32}
        @test irf(abcd_d32, 8).values ≈ irf(sol32, 8).values atol=1e-5
        @test irf(abcd_p32, 8).values ≈ irf(psol32, 8).values atol=1e-5

        ss32 = StateSpaceModel(Float32[1.0 0.0], Float32[0.1;;], Float32[0.8 0.0; 0.0 0.5], Float32[0.2;;];
                               R=Float32[1.0; 0.0;;])
        abcd_ss32 = abcd_representation(ss32)
        @test abcd_ss32 isa ABCDSystem{Float32}

        # Error handling
        @test_throws ArgumentError abcd_representation(sol32, [:nonexistent])
        @test_throws BoundsError abcd_representation(sol32, [99])
        @test_throws ArgumentError abcd_representation(sol32, [:y]; reduction=:invalid)
        @test_throws ArgumentError abcd_representation(psol32, [:y]; reduction=:invalid)
        @test_throws ArgumentError abcd_representation(ss32; reduction=:invalid)
        @test_throws ArgumentError irf(sys, 0)
        @test_throws ArgumentError fevd(sys, 0)
        @test_throws ArgumentError analytical_moments(sys; lags=-1)
        @test_throws ArgumentError analytical_moments(sys; format=:bad)
    end
end

@testset "FVRSW Poor Man's Invertibility & Komunjer-Ng Zeros (#910)" begin
    MEM = MacroEconometricModels

    # =========================================================================
    # 1. FVRSW (2007, pp. 1023-1024) Permanent Income Model
    # =========================================================================
    @testset "FVRSW (2007) Permanent Income: square non-invertible, invertible, tall" begin
        spec_fv = @dsge begin
            parameters: R_gross = 1.2, sigma_w = 1.0
            endogenous: y, c, y_m_c
            exogenous: w

            c[t] = c[t-1] + sigma_w * (1 - 1/R_gross) * w[t]
            y_m_c[t] = -c[t-1] + sigma_w * (1/R_gross) * w[t]
            y_m_c[t] = y[t] - c[t]

            steady_state = [0.0, 0.0, 0.0]
        end
        sol_fv = solve(spec_fv; method=:gensys)

        # (a) Observable = y_m_c (saving): M = [1.2], non-invertible!
        res_ymc = poor_mans_invertibility(sol_fv, [:y_m_c])
        @test res_ymc isa InvertibilityResult{Float64}
        @test res_ymc isa MEM.StatsAPI.HypothesisTest
        @test res_ymc.status === :non_invertible_zeros
        @test !res_ymc.invertible
        @test !is_invertible(res_ymc)
        @test !is_invertible(sol_fv, [:y_m_c])
        @test res_ymc.M ≈ [1.2;;] atol=1e-10
        @test res_ymc.max_modulus ≈ 1.2 atol=1e-10
        @test res_ymc.rho ≈ 1.2 atol=1e-10
        @test res_ymc.n_unstable == 1
        @test res_ymc.n_infinite_zeros == 0
        @test res_ymc.eigenvalues ≈ [1.2 + 0.0im] atol=1e-10
        @test res_ymc.transmission_zeros ≈ [1.2 + 0.0im] atol=1e-10
        @test isempty(res_ymc.var_lag_matrices)
        @test MEM.StatsAPI.pvalue(res_ymc) == 0.0

        # (b) Observable = c (consumption): M = [0.0], exact AR(1) random walk!
        res_c = poor_mans_invertibility(sol_fv, [:c]; max_var_lags=10)
        @test res_c.status === :invertible
        @test res_c.invertible
        @test is_invertible(res_c)
        @test is_invertible(sol_fv, [:c])
        @test res_c.M ≈ [0.0;;] atol=1e-12
        @test res_c.max_modulus < 1e-12
        @test res_c.n_unstable == 0
        @test res_c.Omega ≈ [1.0 / (1.0 - 1.0 / 1.2);;] atol=1e-10
        @test length(res_c.var_lag_matrices) == 10
        @test res_c.Pi[1] ≈ [1.0;;] atol=1e-10
        for j in 2:10
            @test norm(res_c.Pi[j]) < 1e-12
        end
        @test MEM.StatsAPI.pvalue(res_c) == 1.0

        # (c) Observable = y (white noise) with minimal reduction -> 0 states, invertible
        res_y_min = poor_mans_invertibility(sol_fv, [:y]; reduction=:minimal)
        @test res_y_min.status === :invertible
        @test res_y_min.invertible
        @test res_y_min.n_states == 0
        @test res_y_min.max_modulus == 0.0

        # (d) Tall pair [:c, :y_m_c] (k=2, m=1): observable c_t kills the zero -> invertible_tall!
        res_tall = poor_mans_invertibility(sol_fv, [:c, :y_m_c])
        @test res_tall.status === :invertible_tall
        @test res_tall.invertible
        @test is_invertible(sol_fv, [:c, :y_m_c])
        @test isempty(res_tall.transmission_zeros)
        @test res_tall.n_obs == 2
        @test res_tall.n_shocks == 1
        @test res_tall.rank_D == 1
    end

    # =========================================================================
    # 2. Square invertible VAR(∞) lag matrices & borderline unit circle
    # =========================================================================
    @testset "Square ARMA(1,1) VAR(∞) parity and borderline unit circle" begin
        # y_{t+1} = (1 + 0.4L)/(1 - 0.7L) w_{t+1} => x_{t+1} = 0.7 x_t + w_{t+1}, y_{t+1} = 1.1 x_t + w_{t+1}
        sys_arma = ABCDSystem([0.7;;], [1.0;;], [1.1;;], [1.0;;])
        res_arma = poor_mans_invertibility(sys_arma; max_var_lags=15)
        @test res_arma.status === :invertible
        @test res_arma.invertible
        @test res_arma.M ≈ [-0.4;;] atol=1e-12
        @test res_arma.max_modulus ≈ 0.4 atol=1e-12
        for j in 1:15
            @test res_arma.var_lag_matrices[j] ≈ [1.1 * (-0.4)^(j - 1);;] atol=1e-12
        end

        # Complex conjugate eigenvalues inside unit circle
        A_cx = [0.5 0.3; -0.3 0.5]
        B_cx = [1.0 0.0; 0.0 1.0]
        C_cx = [0.2 0.1; -0.1 0.2]
        D_cx = [1.0 0.0; 0.0 1.0]
        sys_cx = ABCDSystem(A_cx, B_cx, C_cx, D_cx)
        res_cx = poor_mans_invertibility(sys_cx; max_var_lags=5)
        @test res_cx.status === :invertible
        @test !all(iszero, res_cx.zero_im)
        @test res_cx.eigenvalues ≈ eigvals(A_cx - B_cx * inv(D_cx) * C_cx) atol=1e-12

        # Over-differenced MA(1): y_{t+1} = w_{t+1} - w_t => M = [1.0], unit circle
        sys_uc = ABCDSystem([0.0;;], [1.0;;], [-1.0;;], [1.0;;])
        res_uc = poor_mans_invertibility(sys_uc)
        @test res_uc.status === :borderline_unit_circle
        @test !res_uc.invertible
        @test res_uc.max_modulus ≈ 1.0 atol=1e-12
        @test res_uc.n_unstable == 0
        @test isempty(res_uc.var_lag_matrices)
    end

    # =========================================================================
    # 3. Underdetermined (k < m) and singular D / news-shock regimes
    # =========================================================================
    @testset "Underdetermined (k < m) and singular D / news shocks" begin
        # Wide system k = 1 < m = 2
        sys_wide = ABCDSystem([0.5;;], [1.0 0.5], [1.0;;], [1.0 0.5])
        res_wide = poor_mans_invertibility(sys_wide)
        @test res_wide.status === :non_invertible_underdetermined
        @test !res_wide.invertible
        @test isnan(res_wide.max_modulus)
        @test isempty(res_wide.eigenvalues)
        @test size(res_wide.Omega) == (2, 1)
        @test isempty(res_wide.var_lag_matrices)

        # Pure delayed news shock (GMR 2020 eq. 2.10 root at L = 0): y_{t+1} = w_t
        sys_delay1 = ABCDSystem([0.0;;], [1.0;;], [1.0;;], [0.0;;])
        res_d1 = poor_mans_invertibility(sys_delay1)
        @test res_d1.status === :non_invertible_singular_D
        @test !res_d1.invertible
        @test res_d1.rank_D == 0
        @test res_d1.n_infinite_zeros == 1
        @test isempty(res_d1.transmission_zeros)
        @test isinf(res_d1.max_modulus)

        # Delayed news shock with finite zero: y_{t+1} = w_t - 0.5 w_{t-1}
        A_d2 = [0.0 0.0; 1.0 0.0]
        B_d2 = [1.0; 0.0;;]
        C_d2 = [1.0 -0.5]
        D_d2 = [0.0;;]
        sys_d2 = ABCDSystem(A_d2, B_d2, C_d2, D_d2)
        res_d2 = poor_mans_invertibility(sys_d2)
        @test res_d2.status === :non_invertible_singular_D
        @test !res_d2.invertible
        @test res_d2.n_infinite_zeros == 1
        @test res_d2.transmission_zeros ≈ [0.5 + 0.0im] atol=1e-12
        @test res_d2.max_modulus ≈ 0.5 atol=1e-12
        @test res_d2.n_unstable == 0

        # Delayed news shock with unstable finite zero: y_{t+1} = w_t - 1.4 w_{t-1}
        sys_d2u = ABCDSystem(A_d2, B_d2, [1.0 -1.4], D_d2)
        res_d2u = poor_mans_invertibility(sys_d2u)
        @test res_d2u.status === :non_invertible_singular_D
        @test res_d2u.transmission_zeros ≈ [1.4 + 0.0im] atol=1e-12
        @test res_d2u.n_unstable == 1

        # Singular square pencil (collinear rows + unobserved mode at -0.6)
        A_sp = [0.4 0.0; 0.0 0.7]
        B_sp = [1.0 0.0; 0.0 1.0]
        C_sp = [1.0 0.0; 2.0 0.0]
        D_sp = [1.0 0.0; 2.0 0.0]
        res_sp = poor_mans_invertibility(ABCDSystem(A_sp, B_sp, C_sp, D_sp))
        @test res_sp.status === :non_invertible_singular_D
        @test !res_sp.invertible
        @test res_sp.rank_D == 1
        @test res_sp.transmission_zeros ≈ [-0.6 + 0.0im] atol=1e-12

        # Tall rank-deficient D (k = 3 > m = 2, rank(D) = 1)
        A_tr = [0.5 0.0; 0.0 0.3]
        B_tr = [1.0 0.0; 0.0 1.0]
        C_tr = [1.0 0.0; 0.0 1.0; 1.0 2.0]
        D_tr = [1.0 0.0; 0.0 0.0; 0.0 0.0]
        res_tr = poor_mans_invertibility(ABCDSystem(A_tr, B_tr, C_tr, D_tr))
        @test res_tr.status === :non_invertible_singular_D
        @test !res_tr.invertible
        @test res_tr.rank_D == 1
        @test res_tr.n_infinite_zeros >= 1
        @test isempty(res_tr.transmission_zeros)
    end

    # =========================================================================
    # 4. Komunjer & Ng (2011) tall full-column-rank systems (k > m, rank(D) == m)
    # =========================================================================
    @testset "Komunjer-Ng (2011) tall full-column-rank systems" begin
        # 4a. Unstable eigenvalue in A_red = [1.5] that IS observable via C_2 -> NO zeros -> invertible_tall!
        sys_tall_obs = ABCDSystem([1.5;;], [1.0;;], [0.0; 1.0;;], [1.0; 0.0;;])
        res_to = poor_mans_invertibility(sys_tall_obs)
        @test res_to.status === :invertible_tall
        @test res_to.invertible
        @test isempty(res_to.transmission_zeros)
        @test res_to.max_modulus == 0.0
        @test res_to.Omega * sys_tall_obs.D ≈ [1.0;;] atol=1e-12

        # 4b. Tall system with stable unobservable zero at 0.4 (under orthogonal rotation)
        rng = MersenneTwister(910)
        Q_st = Matrix(qr(randn(rng, 2, 2)).Q)
        Q_ob = Matrix(qr(randn(rng, 3, 3)).Q)
        A_red_st = [0.4 0.25; 0.0 0.6]
        B_st = [1.0; 0.8;;]
        C1_st = [0.5 0.3]
        C2_st = [0.0 1.0; 0.0 0.4]
        R1_st = [2.0;;]
        A_raw_st = A_red_st + B_st * inv(R1_st) * C1_st
        C_raw_st = vcat(C1_st, C2_st)
        D_raw_st = vcat(R1_st, zeros(2, 1))
        sys_tall_st = ABCDSystem(Q_st * A_raw_st * Q_st', Q_st * B_st, Q_ob * C_raw_st * Q_st', Q_ob * D_raw_st)
        res_tall_st = poor_mans_invertibility(sys_tall_st)
        @test res_tall_st.status === :invertible_tall
        @test res_tall_st.invertible
        @test res_tall_st.transmission_zeros ≈ [0.4 + 0.0im] atol=1e-10
        @test res_tall_st.max_modulus ≈ 0.4 atol=1e-10
        @test res_tall_st.n_unstable == 0
        @test res_tall_st.Omega * sys_tall_st.D ≈ [1.0;;] atol=1e-12

        # 4c. Tall system with unstable unobservable zero at 1.5 (under orthogonal rotation)
        A_red_un = [1.5 0.3; 0.0 0.4]
        A_raw_un = A_red_un + B_st * inv(R1_st) * C1_st
        sys_tall_un = ABCDSystem(Q_st * A_raw_un * Q_st', Q_st * B_st, Q_ob * C_raw_st * Q_st', Q_ob * D_raw_st)
        res_tall_un = poor_mans_invertibility(sys_tall_un)
        @test res_tall_un.status === :non_invertible_tall_zeros
        @test !res_tall_un.invertible
        @test res_tall_un.transmission_zeros ≈ [1.5 + 0.0im] atol=1e-10
        @test res_tall_un.max_modulus ≈ 1.5 atol=1e-10
        @test res_tall_un.n_unstable == 1
    end

    # =========================================================================
    # 5. Convenience dispatches, Float32, display/report, RSER-01, error paths
    # =========================================================================
    @testset "Convenience dispatches, Float32, display, RSER-01, and validation" begin
        spec_nk = compute_steady_state(@dsge begin
            parameters: rho = 0.9, sigma_y = 0.5, kappa = 0.3, phi_pi = 1.5, rho_r = 0.7
            endogenous: y, pi_v, i_rate, a
            exogenous: eps_y
            linear: true

            y[t] = y[t+1] - sigma_y * (i_rate[t] - pi_v[t+1])
            pi_v[t] = 0.99 * pi_v[t+1] + kappa * y[t]
            i_rate[t] = rho_r * i_rate[t-1] + (1.0 - rho_r) * phi_pi * pi_v[t] + a[t]
            a[t] = rho * a[t-1] + eps_y[t]
        end)
        sol_nk = solve(spec_nk; method=:gensys)
        psol_nk = perturbation_solver(spec_nk; order=1)

        r_dsge = poor_mans_invertibility(sol_nk, [:y])
        r_pert = poor_mans_invertibility(psol_nk, [:y])
        @test r_dsge.status === r_pert.status
        @test r_dsge.max_modulus ≈ r_pert.max_modulus atol=1e-8
        @test is_invertible(sol_nk, [:y]) == r_dsge.invertible

        # Adding measurement error makes 1-obs / 2-shock underdetermined
        r_me = poor_mans_invertibility(sol_nk, [:y]; measurement_error=[0.05])
        @test r_me.status === :non_invertible_underdetermined
        @test !is_invertible(sol_nk, [:y]; measurement_error=[0.05])

        # StateSpaceModel dispatch
        ss = StateSpaceModel([1.0 0.0], [0.2;;], [0.6 0.1; 0.0 0.4], [0.5;;]; R=[1.0; 0.2;;])
        r_ss = poor_mans_invertibility(ss)
        @test r_ss.status === :non_invertible_underdetermined  # 1 obs, 2 uncorrelated shocks (state + H)
        @test !is_invertible(ss)

        # HADSGESolution dispatch
        ha_spec = load_ha_example(:krusell_smith)
        ha_ss = compute_steady_state(ha_spec; max_iter=100, tol=1e-4)
        ha_sol = solve(ha_spec; method=:ssj, ss=ha_ss, T_horizon=50, n_reduced=10)
        r_ha = poor_mans_invertibility(ha_sol)
        @test r_ha isa InvertibilityResult{Float64}
        @test is_invertible(ha_sol) == r_ha.invertible

        # Float32 precision preservation
        sys32 = ABCDSystem(Float32[0.7;;], Float32[1.0;;], Float32[1.1;;], Float32[1.0;;])
        res32 = poor_mans_invertibility(sys32; max_var_lags=5)
        @test res32 isa InvertibilityResult{Float32}
        @test res32.invertible
        @test eltype(res32.zero_re) === Float32
        @test eltype(res32.zero_im) === Float32
        @test eltype(res32.eigenvalues) === ComplexF32
        @test eltype(res32.M) === Float32
        @test eltype(res32.Omega) === Float32
        @test eltype(res32.var_lag_matrices[1]) === Float32
        @test res32.max_modulus ≈ 0.4f0 atol=1e-5
        @test MEM.StatsAPI.pvalue(res32) === 1.0f0

        # show and report across text and LaTeX backends
        res_d2_loc = poor_mans_invertibility(ABCDSystem([0.0 0.0; 1.0 0.0], [1.0; 0.0;;], [1.0 -0.5], [0.0;;]))
        res_wide_loc = poor_mans_invertibility(ABCDSystem([0.5;;], [1.0 0.5], [1.0;;], [1.0 0.5]))

        io = IOBuffer()
        show(io, res32)
        s_txt = String(take!(io))
        @test occursin("Poor Man", s_txt)
        @test occursin("invertible", s_txt)

        report(io, res_d2_loc)
        r_txt = String(take!(io))
        @test occursin("Transmission Zeros", r_txt)

        report(io, res32)
        r32_txt = String(take!(io))
        @test occursin("Shock Recovery Matrix", r32_txt)

        with_display_backend(:latex) do
            show(io, res_d2_loc)
            l_txt = String(take!(io))
            @test occursin("tabular", l_txt)
            report(io, res32)
            l_rep = String(take!(io))
            @test occursin("tabular", l_rep)
        end

        # RSER-01 positional constructor & plain-field round-trip check
        for obj in (res32, res_d2_loc, res_wide_loc)
            payload = Dict{String,Any}(string(f) => MEM._ser_field(getfield(obj, f)) for f in fieldnames(typeof(obj)))
            MEM._assert_plain_payload(payload)
            reconstructed = MEM._from_serializable(InvertibilityResult, payload, MEM.SERIALIZATION_FORMAT_VERSION)
            @test typeof(reconstructed) === typeof(obj)
            @test reconstructed.invertible == obj.invertible
            @test reconstructed.status === obj.status
            @test reconstructed.zero_re == obj.zero_re
            @test reconstructed.zero_im == obj.zero_im
            @test reconstructed.var_lag_matrices == obj.var_lag_matrices
            @test isequal(reconstructed.max_modulus, obj.max_modulus)
        end

        # Propertynames check
        pnames = propertynames(res32)
        @test :eigenvalues in pnames
        @test :transmission_zeros in pnames
        @test :rho in pnames
        @test :Pi in pnames

        # Validation errors
        @test_throws ArgumentError poor_mans_invertibility(sys32; tol=-1e-5)
        @test_throws ArgumentError poor_mans_invertibility(sys32; max_var_lags=-1)
    end
end

@testset "Hansen-Sargent DARE Innovations Representation (#911)" begin
    MEM = MacroEconometricModels

    # =========================================================================
    # 1. FVRSW (2007, §II eqs. 15-16) Permanent Income analytical benchmark
    # =========================================================================
    @testset "FVRSW (2007) Permanent Income closed-form DARE & forecast_sufficiency" begin
        spec_fv = @dsge begin
            parameters: R_gross = 1.2, sigma_w = 1.0
            endogenous: y, c, y_m_c
            exogenous: w

            c[t] = c[t-1] + sigma_w * (1 - 1/R_gross) * w[t]
            y_m_c[t] = -c[t-1] + sigma_w * (1/R_gross) * w[t]
            y_m_c[t] = y[t] - c[t]

            steady_state = [0.0, 0.0, 0.0]
        end
        sol_fv = solve(spec_fv; method=:gensys)
        abcd_fv = abcd_representation(sol_fv, [:y_m_c])
        innov = innovations_representation(abcd_fv)

        @test innov isa InnovationsRepresentation{Float64}
        @test !innov.invertible
        @test innov.dare_residual < 1e-12
        @test innov.P ≈ [11.0 / 36.0;;] atol=1e-11
        @test innov.K ≈ [-1.0 / 6.0;;] atol=1e-11
        @test innov.Sigma_u ≈ [1.0;;] atol=1e-11
        @test innov.Sigma_u_chol ≈ [1.0;;] atol=1e-11
        @test innov.information_loss ≈ [11.0 / 36.0] atol=1e-11

        # Derived getproperty aliases matching FVRSW eqs. (9)-(12)
        @test innov.Sigma_state ≈ [11.0 / 36.0;;] atol=1e-11
        @test innov.Omega_innov ≈ [1.0;;] atol=1e-11
        @test innov.D_hat ≈ [1.0;;] atol=1e-11
        @test innov.B_hat ≈ [-1.0 / 6.0;;] atol=1e-11
        @test innov.M_innov ≈ [1.0 / 1.2;;] atol=1e-11
        @test maximum(abs, eigvals(innov.A - innov.K * innov.C)) ≈ 1.0 / 1.2 atol=1e-11

        # Wold-fundamental ABCD conversion & IRF
        wold_fv = to_abcd(innov)
        @test wold_fv isa ABCDSystem{Float64}
        @test is_invertible(wold_fv)
        @test poor_mans_invertibility(wold_fv).M ≈ [1.0 / 1.2;;] atol=1e-11
        ir_wold = irf(innov, 8)
        @test ir_wold isa ImpulseResponse{Float64}
        @test ir_wold.values ≈ irf(wold_fv, 8).values atol=1e-12
        @test ir_wold.values[1, 1, 1] ≈ 1.0 atol=1e-11
        @test !(ir_wold.values ≈ irf(abcd_fv, 8).values)

        # Upgraded forecast_sufficiency on unit-root FVRSW runs with zero warnings
        fs_fv = @test_nowarn forecast_sufficiency(sol_fv, [:y_m_c]; H=20)
        @test fs_fv isa ForecastSufficiency{Float64}
        @test !fs_fv.invertible
        @test fs_fv.one_step_ratio[1] ≈ 1.44 atol=1e-11  # 1.0 / (25/36) = 36/25

        # Invertible observable (:c) -> zero state error, zero info loss, exact Sigma_u = D*D'
        abcd_c = abcd_representation(sol_fv, [:c])
        innov_c = innovations_representation(sol_fv, [:c])
        @test innov_c.invertible
        @test norm(innov_c.P) < 1e-12
        @test innov_c.Sigma_u ≈ abcd_c.D * abcd_c.D' atol=1e-11
        @test maximum(innov_c.information_loss) < 1e-12
        @test irf(innov_c, 8).values ≈ irf(abcd_c, 8).values atol=1e-11
    end

    # =========================================================================
    # 2. Invertible benchmark & Leeper-Walker-Yang (2013) fiscal foresight
    # =========================================================================
    @testset "Leeper-Walker-Yang (2013) fiscal foresight & autocovariance parity" begin
        # Stationary invertible ARMA(1,1)
        sys_arma = ABCDSystem([0.7;;], [1.0;;], [1.1;;], [1.0;;])
        innov_arma = innovations_representation(sys_arma)
        @test innov_arma.invertible
        @test norm(innov_arma.P) < 1e-12
        @test innov_arma.Sigma_u ≈ [1.0;;] atol=1e-12
        @test maximum(innov_arma.information_loss) < 1e-12
        @test irf(innov_arma, 12).values ≈ irf(sys_arma, 12).values atol=1e-12

        # Leeper-Walker-Yang (2013) 2-period fiscal foresight news shock:
        # k_t responds to announced tax news before tax rate tau_t moves -> rank(D) = 1 < 2
        alpha = 0.5
        kappa = 0.4
        theta = 0.8
        A_lwy = [alpha -kappa * theta -kappa; 0.0 0.0 0.0; 0.0 1.0 0.0]
        B_lwy = [1.0 -kappa * theta^2; 0.0 1.0; 0.0 0.0]
        C_lwy = [alpha -kappa * theta -kappa; 0.0 0.0 1.0]
        D_lwy = [1.0 -kappa * theta^2; 0.0 0.0]
        abcd_lwy = ABCDSystem(A_lwy, B_lwy, C_lwy, D_lwy;
                              state_names=[:k, :eps_tau, :eps_tau_lag],
                              obs_names=[:k_obs, :tau_obs],
                              shock_names=[:u_A, :eps_tau_news])

        @test !is_invertible(abcd_lwy)
        @test poor_mans_invertibility(abcd_lwy).status === :non_invertible_singular_D

        innov_lwy = innovations_representation(abcd_lwy)
        @test !innov_lwy.invertible
        @test innov_lwy.dare_residual < 1e-12
        @test maximum(abs, eigvals(innov_lwy.A - innov_lwy.K * innov_lwy.C)) < 1.0
        @test minimum(eigvals(Symmetric(innov_lwy.Sigma_u))) > 0.0
        @test all(>=(0.0), innov_lwy.information_loss)
        @test maximum(innov_lwy.information_loss) > 0.1

        # Wold representation is square & invertible and matches ALL autocovariances < 1e-11
        wold_lwy = to_abcd(innov_lwy)
        @test is_invertible(wold_lwy)
        moms_orig = analytical_moments(abcd_lwy; lags=6, format=:matrices)
        moms_wold = analytical_moments(wold_lwy; lags=6, format=:matrices)
        @test length(moms_wold) == length(moms_orig) == 7
        for h in eachindex(moms_orig)
            @test moms_wold[h] ≈ moms_orig[h] atol=1e-11
        end

        # Default horizon H=20 IRF matches to_abcd(innov_lwy)
        ir_def = irf(innov_lwy)
        @test ir_def.horizon == 20
        @test size(ir_def.values) == (20, 2, 2)
        @test ir_def.values ≈ irf(wold_lwy, 20).values atol=1e-12
    end

    # =========================================================================
    # 3. Convenience dispatches, n==0, Float32, display/report, RSER-01, errors
    # =========================================================================
    @testset "Convenience dispatches, Float32, display, RSER-01, and validation" begin
        spec_nk = compute_steady_state(@dsge begin
            parameters: rho = 0.9, sigma_y = 0.5, kappa = 0.3, phi_pi = 1.5, rho_r = 0.7
            endogenous: y, pi_v, i_rate, a
            exogenous: eps_y
            linear: true

            y[t] = y[t+1] - sigma_y * (i_rate[t] - pi_v[t+1])
            pi_v[t] = 0.99 * pi_v[t+1] + kappa * y[t]
            i_rate[t] = rho_r * i_rate[t-1] + (1.0 - rho_r) * phi_pi * pi_v[t] + a[t]
            a[t] = rho * a[t-1] + eps_y[t]
        end)
        sol_nk = solve(spec_nk; method=:gensys)
        psol_nk = perturbation_solver(spec_nk; order=1)

        inn_dsge = innovations_representation(sol_nk, [:y])
        inn_pert = innovations_representation(psol_nk, [:y])
        @test inn_dsge.Sigma_u ≈ inn_pert.Sigma_u atol=1e-8
        @test inn_dsge.P ≈ inn_pert.P atol=1e-8

        # StateSpaceModel dispatch
        ss_corr = StateSpaceModel([1.0 0.2], [0.1;;], [0.7 0.1; 0.0 0.5], [0.4;;];
                                  R=[1.0; 0.3;;], G=[0.05;;])
        inn_ss = innovations_representation(ss_corr)
        @test inn_ss isa InnovationsRepresentation{Float64}
        @test inn_ss.dare_residual < 1e-12
        @test analytical_moments(to_abcd(inn_ss); lags=4, format=:matrices) ≈
              analytical_moments(abcd_representation(ss_corr); lags=4, format=:matrices) atol=1e-11

        # Zero-state system (n == 0)
        sys_zero = ABCDSystem(zeros(0, 0), zeros(0, 2), zeros(2, 0), [1.0 0.0; 0.0 2.0])
        inn_zero = innovations_representation(sys_zero)
        @test size(inn_zero.P) == (0, 0)
        @test inn_zero.Sigma_u ≈ [1.0 0.0; 0.0 4.0] atol=1e-12
        @test inn_zero.invertible
        @test all(iszero, inn_zero.information_loss)
        @test irf(inn_zero, 5).values[1, :, :] ≈ [1.0 0.0; 0.0 2.0] atol=1e-12

        # Float32 precision preservation (both invertible and non-invertible FVRSW)
        sys32_ni = ABCDSystem(Float32[1.0;;], Float32[1.0 - 1.0 / 1.2;;],
                              Float32[-1.0;;], Float32[1.0 / 1.2;;])
        inn32 = innovations_representation(sys32_ni)
        @test inn32 isa InnovationsRepresentation{Float32}
        @test eltype(inn32.P) === Float32
        @test eltype(inn32.K) === Float32
        @test eltype(inn32.Sigma_u) === Float32
        @test eltype(inn32.Sigma_u_chol) === Float32
        @test eltype(inn32.information_loss) === Float32
        @test inn32.dare_residual isa Float32
        @test inn32.P ≈ Float32[11.0 / 36.0;;] atol=1f-5
        @test inn32.Sigma_u ≈ Float32[1.0;;] atol=1f-5
        @test to_abcd(inn32) isa ABCDSystem{Float32}
        @test irf(inn32, 8) isa ImpulseResponse{Float32}

        # show and report across text and LaTeX backends
        io = IOBuffer()
        show(io, inn32)
        s_txt = String(take!(io))
        @test occursin("Wold Innovations", s_txt)

        report(io, inn32)
        r_txt = String(take!(io))
        @test occursin("Information Loss", r_txt)
        @test occursin("Kalman Gain", r_txt)

        with_display_backend(:latex) do
            show(io, inn_ss)
            l_txt = String(take!(io))
            @test occursin("tabular", l_txt)
            report(io, inn32)
            l_rep = String(take!(io))
            @test occursin("tabular", l_rep)
        end

        # RSER-01 positional constructor & plain-field round-trip check
        for obj in (inn32, inn_ss, inn_zero)
            payload = Dict{String,Any}(string(f) => MEM._ser_field(getfield(obj, f)) for f in fieldnames(typeof(obj)))
            MEM._assert_plain_payload(payload)
            reconstructed = MEM._from_serializable(InnovationsRepresentation, payload, MEM.SERIALIZATION_FORMAT_VERSION)
            @test typeof(reconstructed) === typeof(obj)
            @test reconstructed.A == obj.A
            @test reconstructed.K == obj.K
            @test reconstructed.Sigma_u == obj.Sigma_u
            @test reconstructed.P == obj.P
            @test reconstructed.invertible == obj.invertible
            @test reconstructed.information_loss == obj.information_loss
        end

        # Propertynames check
        pnames = propertynames(inn32)
        @test :Sigma_state in pnames
        @test :Omega_innov in pnames
        @test :D_hat in pnames
        @test :B_hat in pnames
        @test :M_innov in pnames

        # Validation errors
        @test_throws ArgumentError innovations_representation(sys32_ni; rcond_tol=0.0)
        @test_throws ArgumentError irf(inn32, 0)
        @test_throws ArgumentError InnovationsRepresentation{Float64}(
            zeros(2, 3), zeros(2, 1), zeros(1, 2), zeros(1, 1), zeros(1, 1), zeros(2, 2),
            0.0, true, [0.0], ["x_1", "x_2"], ["y_1"],
        )
    end
end
