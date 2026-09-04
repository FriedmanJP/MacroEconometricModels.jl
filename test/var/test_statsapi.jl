# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using StatsAPI
using LinearAlgebra
using Statistics
using Random

@testset "StatsAPI Compatibility" begin
    # Reference DGP (DGP-02 #791): non-diagonal A, non-identity B0, intercept.
    rng = MersenneTwister(42)
    T, n, p = 100, 2, 1
    Y = dgp_var(rng; A=[0.5 0.1; 0.0 0.5], B0=[0.1 0.0; 0.05 0.1],
                c=[0.1, 0.1], T=T).Y

    model = StatsAPI.fit(VARModel, Y, p)

    # 1. Basic Interface
    @test StatsAPI.coef(model) isa Matrix
    @test StatsAPI.residuals(model) isa Matrix
    @test size(StatsAPI.coef(model)) == (1 + n * p, n)

    # 2. Test dof, nobs
    _tprint("Testing dof/nobs...")
    @test StatsAPI.nobs(model) == T
    @test StatsAPI.dof(model) == (1 + n * p) * n

    # 3. Test vcov
    _tprint("Testing vcov()...")
    V = StatsAPI.vcov(model)
    @test size(V) == ((1 + n * p) * n, (1 + n * p) * n)
    @test issymmetric(V) || norm(V - V') < 1e-10  # Should be symmetric

    # 4. Test predict (in-sample)
    _tprint("Testing predict() in-sample...")
    y_hat = StatsAPI.predict(model)
    @test size(y_hat) == (T - p, n) # Effective sample size
    @test all(isfinite, y_hat)

    # 5. Test predict (forecast)
    _tprint("Testing predict() forecast...")
    steps = 5
    y_fcast = StatsAPI.predict(model, steps)
    @test size(y_fcast) == (steps, n)
    @test all(isfinite, y_fcast)

    # 6. Test loglikelihood
    _tprint("Testing loglikelihood...")
    ll = StatsAPI.loglikelihood(model)
    @test ll isa Float64
    @test isfinite(ll)

    # 7. Test stderror
    _tprint("Testing stderror...")
    se = StatsAPI.stderror(model)
    @test length(se) == length(vec(StatsAPI.coef(model)))
    @test all(se .> 0)
    @test all(isfinite, se)

    # 8. Test confint
    _tprint("Testing confint...")
    ci = StatsAPI.confint(model; level=0.95)
    @test size(ci) == (length(se), 2)
    @test all(ci[:, 1] .< ci[:, 2])  # Lower < Upper

    # Check that confidence intervals are reasonable (contain plausible values)
    # B structure: [c1 c2; A11 A12; A21 A22]
    # vec(B) = [c1, A11, A21, c2, A12, A22]
    b_vec = vec(StatsAPI.coef(model))

    # Check that CI widths are reasonable (not too narrow, not too wide)
    ci_widths = ci[:, 2] - ci[:, 1]
    @test all(ci_widths .> 0)
    @test all(ci_widths .< 10)  # Reasonable upper bound

    # F-01 regression: vcov/stderror must use the dof-adjusted residual covariance
    # U'U/(T_eff−k), not the ML covariance U'U/T_eff (which made SEs too small).
    _, Xd = MacroEconometricModels.construct_var_matrices(model.Y, model.p)
    Teff, kk = size(Xd, 1), size(Xd, 2)
    XtXi = MacroEconometricModels.robust_inv(Xd' * Xd)
    V_dof = kron((model.U' * model.U) / (Teff - kk), XtXi)
    @test isapprox(StatsAPI.vcov(model), V_dof; rtol=1e-10)
    se_ml = sqrt.(diag(kron((model.U' * model.U) / Teff, XtXi)))
    @test all(StatsAPI.stderror(model) .> se_ml)            # dof-adjusted SEs strictly larger
    @test isapprox(StatsAPI.stderror(model) ./ se_ml, fill(sqrt(Teff / (Teff - kk)), length(se_ml)); rtol=1e-10)

    # 9. Test islinear
    _tprint("Testing islinear...")
    @test StatsAPI.islinear(model)

    _tprint("StatsAPI Tests Passed.")
end

@testset "StatsAPI r2 for VARModel" begin
    # r2 IS implemented for VARModel (per-equation R² vector), so assert the
    # contract directly — no try/catch skip (DGP-02 #791).
    rng = MersenneTwister(123)
    T, n, p = 1000, 2, 1
    A_ref = [0.5 0.1; 0.0 0.5]
    B0_ref = [0.3 0.0; 0.1 0.2]
    sim = dgp_var(rng; A=A_ref, B0=B0_ref, T=T)
    model = estimate_var(sim.Y, p)

    r2_val = StatsAPI.r2(model)
    @test r2_val isa AbstractVector
    @test length(r2_val) == n
    @test all(isfinite, r2_val)
    @test all(x -> 0 <= x <= 1, r2_val)
    _tprint("r2 for VAR: ", r2_val)

    # Population truth R²_i = 1 - Σ_ii/Γ0_ii; atol 0.05 covers sampling noise
    # of the variance ratio at T=1000 (SE ≈ √(2/T) ≈ 0.045).
    G0 = lyapunov_gamma0(Matrix(A_ref), sim.Sigma)
    r2_pop = [1 - sim.Sigma[i, i] / G0[i, i] for i in 1:n]
    @test r2_val ≈ r2_pop atol = 0.05
end
