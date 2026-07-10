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

    @testset "Consolidated kernel (T147/#246) matches the core primitive" begin
        MEM = MacroEconometricModels
        rng = Random.MersenneTwister(246)
        # small stable linear-Gaussian system WITH both intercepts (b state, d obs)
        Tt = [0.5 0.1; -0.2 0.4]
        RQR = [0.30 0.05; 0.05 0.20]
        Z = [1.0 0.0; 0.3 1.0]
        Hobs = [0.10 0.0; 0.0 0.15]
        b = [0.2, -0.1]; d = [0.05, 0.02]
        n_state = 2; n_obs = 2; T_obs = 40
        a0, P0 = MEM._kalman_init(:stationary, Tt, RQR, n_state)
        LR = cholesky(Symmetric(RQR)).L; LH = cholesky(Symmetric(Hobs)).L
        y = Matrix{Float64}(undef, n_obs, T_obs); x = copy(a0)
        for t in 1:T_obs
            x = b + Tt * x + LR * randn(rng, n_state)
            y[:, t] = d + Z * x + LH * randn(rng, n_obs)
        end

        # reference forward filter assembled from the core _kalman_update primitive
        function ref_filter(y, Z, Tt, RQR, Hobs, b, d, a0, P0; skip=0)
            xr = copy(a0); Pr = Matrix(P0); ll = 0.0
            for t in 1:size(y, 2)
                x_pred = b + Tt * xr
                P_pred = Tt * Pr * Tt' + RQR; P_pred = (P_pred + P_pred') / 2
                if t == skip
                    xr = x_pred; Pr = P_pred; continue
                end
                xu, Pu, v, S, _ = MEM._kalman_update(x_pred, P_pred, y[:, t] - d, Z, Hobs)
                L = cholesky(Symmetric((S + S') / 2)).L
                ll += -0.5 * (length(v) * log(2π) + 2sum(log, diag(L)) + sum(abs2, L \ v))
                xr = xu; Pr = Pu
            end
            return ll, xr, Pr
        end
        ll_ref, xf_ref, Pf_ref = ref_filter(y, Z, Tt, RQR, Hobs, b, d, a0, P0)

        store = MEM.KalmanFilterStore{Float64}(n_state, T_obs)
        ll_k = MEM._kalman_filter!(store, y, Z, Tt, RQR, Hobs; d=d, b=b, a0=a0, P0=P0, scalar=false)
        @test ll_k ≈ ll_ref rtol = 1e-10
        @test store.a_filt[:, end] ≈ xf_ref rtol = 1e-9
        @test store.P_filt[:, :, end] ≈ Pf_ref rtol = 1e-9
        @test norm(store.P_filt[:, :, end] - store.P_filt[:, :, end]') < 1e-12
        @test minimum(eigvals(Symmetric(store.P_filt[:, :, end]))) ≥ -1e-10

        # scalar rank-1 path == multivariate path on a single-observation series
        Zs = reshape([1.0, 0.4], 1, 2); Hs = reshape([0.12], 1, 1); ds = [0.03]
        ys = Matrix{Float64}(undef, 1, T_obs); xs = copy(a0)
        for t in 1:T_obs
            xs = b + Tt * xs + LR * randn(rng, 2)
            ys[1, t] = ds[1] + (Zs * xs)[1] + sqrt(Hs[1, 1]) * randn(rng)
        end
        ll_scalar = MEM._kalman_filter!(nothing, ys, Zs, Tt, RQR, Hs; d=ds, b=b, a0=a0, P0=P0, scalar=true)
        ll_multiv = MEM._kalman_filter!(nothing, ys, Zs, Tt, RQR, Hs; d=ds, b=b, a0=a0, P0=P0, scalar=false)
        @test ll_scalar ≈ ll_multiv rtol = 1e-10

        # missing data: an all-NaN step contributes 0 to the log-likelihood
        y_miss = copy(y); y_miss[:, 10] .= NaN
        ll_miss = MEM._kalman_filter!(nothing, y_miss, Z, Tt, RQR, Hobs; d=d, b=b, a0=a0, P0=P0, scalar=false)
        @test ll_miss ≈ ref_filter(y, Z, Tt, RQR, Hobs, b, d, a0, P0; skip=10)[1] rtol = 1e-10

        # predict_first=false: a0/P0 are the prior a_{1|0}/P_{1|0} (BN-style seed at first obs)
        a1 = [0.3, -0.2]; P1 = [0.5 0.1; 0.1 0.4]
        function ref_pf_false(y, Z, Tt, RQR, Hobs, b, d, a1, P1)
            ll = 0.0; xr = copy(a1); Pr = Matrix(P1)
            for t in 1:size(y, 2)
                if t > 1
                    x_pred = b + Tt * xr; P_pred = Tt * Pr * Tt' + RQR; P_pred = (P_pred + P_pred') / 2
                else
                    x_pred = copy(xr); P_pred = (Pr + Pr') / 2       # a1/P1 taken as a_{1|0}
                end
                xu, Pu, v, S, _ = MEM._kalman_update(x_pred, P_pred, y[:, t] - d, Z, Hobs)
                L = cholesky(Symmetric((S + S') / 2)).L
                ll += -0.5 * (length(v) * log(2π) + 2sum(log, diag(L)) + sum(abs2, L \ v))
                xr = xu; Pr = Pu
            end
            return ll
        end
        ll_pf = MEM._kalman_filter!(nothing, y, Z, Tt, RQR, Hobs; d=d, b=b, a0=a1, P0=P1,
                                    scalar=false, predict_first=false)
        @test ll_pf ≈ ref_pf_false(y, Z, Tt, RQR, Hobs, b, d, a1, P1) rtol = 1e-10

        # _rts_smoother reproduces a reference RTS backward pass over the stored moments
        store2 = MEM.KalmanFilterStore{Float64}(n_state, T_obs)
        MEM._kalman_filter!(store2, y, Z, Tt, RQR, Hobs; d=d, b=b, a0=a0, P0=P0, scalar=false)
        as, Ps = MEM._rts_smoother(store2, Tt)
        as_ref = similar(store2.a_filt); as_ref[:, end] = store2.a_filt[:, end]
        for t in (T_obs-1):-1:1
            J = store2.P_filt[:, :, t] * Tt' * inv(Symmetric(store2.P_pred[:, :, t+1]))
            as_ref[:, t] = store2.a_filt[:, t] + J * (as_ref[:, t+1] - store2.a_pred[:, t+1])
        end
        @test as ≈ as_ref rtol = 1e-9
        @test as[:, end] == store2.a_filt[:, end]

        # init modes
        Ps = MEM._kalman_init(:stationary, Tt, RQR, 2)[2]
        @test norm(Ps - (Tt * Ps * Tt' + RQR)) < 1e-8                       # Lyapunov fixed point
        @test MEM._kalman_init(:kappa, Tt, RQR, 2; kappa=1e6)[2] ≈ 1e6 * Matrix(I, 2, 2)
        a_e, P_e = MEM._kalman_init(:explicit, Tt, RQR, 2; a0=[1.0, 2.0], P0=[2.0 0.0; 0.0 3.0])
        @test a_e == [1.0, 2.0] && P_e == [2.0 0.0; 0.0 3.0]
        @test_throws ArgumentError MEM._kalman_init(:bogus, Tt, RQR, 2)
        @test_throws ArgumentError MEM._kalman_init(:stationary, [1.01 0.0; 0.0 0.5], RQR, 2)
    end
end
