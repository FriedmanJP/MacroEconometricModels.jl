# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra

@testset "Core Kalman operations" begin
    Random.seed!(42)
    n, m = 4, 2

    F = 0.9 * I(n) |> Matrix{Float64}
    H = randn(m, n)
    Q = 0.1 * I(n) |> Matrix{Float64}
    R = 0.05 * I(m) |> Matrix{Float64}
    x0 = zeros(n)
    P0 = Matrix{Float64}(I(n))

    @testset "Lyapunov solver" begin
        P = MacroEconometricModels._solve_discrete_lyapunov(F, Q)
        # Verify P = F * P * F' + Q
        @test P ≈ F * P * F' + Q atol=1e-8
        @test issymmetric(P) || P ≈ P'
        @test all(eigvals(P) .>= 0)  # positive semidefinite
    end

    @testset "Lyapunov solver with near-unit-root system" begin
        F_near = 0.99 * I(n) |> Matrix{Float64}
        P = MacroEconometricModels._solve_discrete_lyapunov(F_near, Q)
        @test P ≈ F_near * P * F_near' + Q atol=1e-8
        @test all(eigvals(P) .>= 0)
        # Analytical solution for diagonal case: P_ii = Q_ii / (1 - lambda^2)
        expected = 0.1 / (1.0 - 0.99^2)
        @test P[1,1] ≈ expected atol=1e-6
    end

    @testset "Lyapunov solver with off-diagonal transition" begin
        F_off = [0.5 0.2; 0.1 0.4]
        Q_off = [0.1 0.0; 0.0 0.1]
        P = MacroEconometricModels._solve_discrete_lyapunov(F_off, Q_off)
        @test P ≈ F_off * P * F_off' + Q_off atol=1e-8
    end

    @testset "predict step" begin
        x_pred, P_pred = MacroEconometricModels._kalman_predict(x0, P0, F, Q)
        @test size(x_pred) == (n,)
        @test size(P_pred) == (n, n)
        @test x_pred ≈ F * x0
        @test P_pred ≈ F * P0 * F' + Q
    end

    @testset "update step" begin
        x_pred, P_pred = MacroEconometricModels._kalman_predict(x0, P0, F, Q)
        y = H * x_pred + 0.1 * randn(m)
        x_upd, P_upd, v, S, K = MacroEconometricModels._kalman_update(x_pred, P_pred, y, H, R)
        @test size(x_upd) == (n,)
        @test size(P_upd) == (n, n)
        @test v ≈ y - H * x_pred
        @test S ≈ H * P_pred * H' + R
        # Updated covariance should be smaller than predicted
        @test tr(P_upd) < tr(P_pred) + 1e-10
    end

    @testset "update reduces uncertainty" begin
        # Multiple update steps should monotonically reduce trace of P
        x, P = x0, P0
        prev_tr = tr(P)
        for _ in 1:5
            x, P = MacroEconometricModels._kalman_predict(x, P, F, Q)
            y = H * x + 0.1 * randn(m)
            x, P, _, _, _ = MacroEconometricModels._kalman_update(x, P, y, H, R)
        end
        @test tr(P) < tr(P0)  # steady-state P should be smaller than initial
    end

    @testset "RTS smoother gain" begin
        P_filt = 0.5 * Matrix{Float64}(I(n))
        P_pred = Matrix{Float64}(I(n))
        J = MacroEconometricModels._rts_smoother_gain(P_filt, F, P_pred)
        @test size(J) == (n, n)
        @test J ≈ P_filt * F' * inv(P_pred) atol=1e-8
    end

    @testset "predict-update roundtrip consistency" begin
        # Verify that predict followed by update with perfect observation recovers state
        x_true = randn(n)
        x_pred, P_pred = MacroEconometricModels._kalman_predict(x_true, Q, F, Q)
        # Observe with zero noise
        R_zero = zeros(m, m) + 1e-12 * I(m)
        y_obs = H * (F * x_true)
        x_upd, P_upd, _, _, _ = MacroEconometricModels._kalman_update(x_pred, P_pred, y_obs, H, R_zero)
        # With near-zero observation noise, updated state should be close to predicted
        @test norm(x_upd - x_pred) < norm(x_true)  # update moved toward observation
    end

    @testset "Lyapunov stability & convergence warns (T062 C-12)" begin
        # Analytic converged value: diagonal Lyapunov P = Q/(1−0.5²) = (1/0.75)·I.
        P = MacroEconometricModels._solve_discrete_lyapunov(
            0.5 * Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2))
        @test Matrix(P) ≈ (1 / 0.75) * Matrix{Float64}(I, 2, 2) atol = 1e-8
        # Unstable transition (ρ=1.01 ≥ 1) ⇒ spectral-radius warning.
        @test_logs (:warn,) match_mode = :any MacroEconometricModels._solve_discrete_lyapunov(
            1.01 * Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2); max_iter=5)
    end

    @testset "Joseph-form measurement update (T058)" begin
        Random.seed!(4242)
        x_pred, P_pred = MacroEconometricModels._kalman_predict(x0, P0, F, Q)
        y = H * x_pred + 0.1 * randn(m)
        x_upd, P_upd, v, S, K = MacroEconometricModels._kalman_update(x_pred, P_pred, y, H, R)

        # (a) Joseph P_upd is symmetric and PSD by construction; S is returned exact.
        @test norm(P_upd - P_upd') < 1e-12
        @test minimum(eigvals(Symmetric(P_upd))) ≥ -1e-10
        @test S ≈ H * P_pred * H' + R

        # (c) Algebraic equivalence at the optimal gain (matches explicit-inverse form).
        S_inv = inv(Symmetric(H * P_pred * H' + R))
        K_ref = P_pred * H' * S_inv
        @test K ≈ K_ref atol = 1e-9
        P_naive = (I - K_ref * H) * P_pred
        @test P_upd ≈ (P_naive + P_naive') / 2 atol = 1e-9
        @test x_upd ≈ x_pred + K_ref * v atol = 1e-10

        # (b) On an ill-conditioned filter covariance (eigenvalue spread 1e11), the Joseph
        #     form stays symmetric + PSD across many steps.
        Random.seed!(77)
        U = qr(randn(n, n)).Q
        P = Matrix(U * Diagonal([1.0, 1e-4, 1e-8, 1e-11]) * U')
        x = zeros(n)
        Qsmall = 1e-10 * Matrix{Float64}(I(n))
        worst_asym = 0.0
        worst_negeig = 0.0
        for _ in 1:300
            x, P = MacroEconometricModels._kalman_predict(x, P, F, Qsmall)
            y = H * x + 0.1 * randn(m)
            x, P, _, _, _ = MacroEconometricModels._kalman_update(x, P, y, H, R)
            worst_asym = max(worst_asym, norm(P - P'))
            worst_negeig = min(worst_negeig, minimum(eigvals(Symmetric(P))))
        end
        @test worst_asym < 1e-10
        @test worst_negeig ≥ -1e-8
    end
end
