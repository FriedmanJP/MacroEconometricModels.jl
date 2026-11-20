# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "Core Kalman operations" begin
    rng = Xoshiro(42)  # DGP-02: explicit rng
    n, m = 4, 2

    F = 0.9 * I(n) |> Matrix{Float64}
    H = randn(rng, m, n)
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
        y = H * x_pred + 0.1 * randn(rng, m)
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
            y = H * x + 0.1 * randn(rng, m)
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
        # DGP-02 #791: the old version observed H·(F·x_true) with ZERO noise, so
        # the innovation was identically zero and a no-op update "passed". Now
        # the observation carries real noise and the prior mean is deliberately
        # wrong with a diffuse covariance — only a genuine update shrinks the
        # state error toward the true predicted state F·x_true.
        # Fully deterministic (no draws): a single random draw decides this
        # comparison by luck, so the state and noise are fixed instead. NOTE:
        # this testset no longer consumes the file-level rng; the only later
        # user of that stream is the DSGE byte-equivalence guard below, which
        # compares two filters on identical data and is stream-agnostic.
        x_true = [0.5, -0.3, 0.8, 0.1]
        x_wrong = x_true + [1.0, -1.0, 0.5, 0.25]  # fixed bad-prior offset
        P_big = 10.0 * Matrix{Float64}(I(n))       # diffuse prior covariance
        x_pred, P_pred = MacroEconometricModels._kalman_predict(x_wrong, P_big, F, Q)
        v_noise = [0.05, -0.04]  # small but nonzero observation noise
        y_obs = H * (F * x_true) + v_noise
        x_upd, P_upd, _, _, _ = MacroEconometricModels._kalman_update(x_pred, P_pred, y_obs, H, R)
        @test norm(x_upd - F * x_true) < norm(x_pred - F * x_true)  # update helps
        @test tr(P_upd) < tr(P_pred)  # and uncertainty shrinks
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
        rng = Xoshiro(4242)  # DGP-02: explicit rng
        x_pred, P_pred = MacroEconometricModels._kalman_predict(x0, P0, F, Q)
        y = H * x_pred + 0.1 * randn(rng, m)
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
        rng = Xoshiro(77)  # DGP-02: explicit rng
        U = qr(randn(rng, n, n)).Q
        P = Matrix(U * Diagonal([1.0, 1e-4, 1e-8, 1e-11]) * U')
        x = zeros(n)
        Qsmall = 1e-10 * Matrix{Float64}(I(n))
        worst_asym = 0.0
        worst_negeig = 0.0
        for _ in 1:300
            x, P = MacroEconometricModels._kalman_predict(x, P, F, Qsmall)
            y = H * x + 0.1 * randn(rng, m)
            x, P, _, _, _ = MacroEconometricModels._kalman_update(x, P, y, H, R)
            worst_asym = max(worst_asym, norm(P - P'))
            worst_negeig = min(worst_negeig, minimum(eigvals(Symmetric(P))))
        end
        @test worst_asym < 1e-10
        @test worst_negeig ≥ -1e-8
    end

    @testset "Consolidated kernel (T147/#246) matches the core primitive" begin
        MEM = MacroEconometricModels
        rng = Random.Xoshiro(246)
        # small stable linear-Gaussian system WITH both intercepts (b state, d obs)
        Tt = [0.5 0.1; -0.2 0.4]
        RQR = [0.30 0.05; 0.05 0.20]
        Z = [1.0 0.0; 0.3 1.0]
        Hobs = [0.10 0.0; 0.0 0.15]
        b = [0.2, -0.1]; d = [0.05, 0.02]
        n_state = 2; n_obs = 2; T_obs = 40
        a0, P0 = MEM._kalman_init(:stationary, Tt, RQR, n_state)
        # Shared state-space DGP (DGP-02 #791): bit-identical draws to the old
        # hand loop, plus the TRUE state path the smoother must recover.
        sim = dgp_state_space(rng; F=Tt, H=Z, Q=RQR, R=Hobs, b=b, d=d, x0=a0, T=T_obs)
        y = Matrix(sim.y')        # n_obs × T_obs layout used below
        Xtrue = Matrix(sim.x')    # n_state × T_obs true states

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
        sims = dgp_state_space(rng; F=Tt, H=Zs, Q=RQR, R=Hs, b=b, d=ds, x0=a0, T=T_obs)
        ys = Matrix(sims.y')
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

        # DGP-02 #791: the smoother recovers the SIMULATED states (the old test
        # built the true path and discarded it). R² floor 0.7 (realized 0.83 at
        # seed 246 — short T=40 sample; the margin covers cross-platform LAPACK
        # variation while a prior-mean "smoother" would score ≈ 0).
        resid_s = as - Xtrue
        resid_f = store2.a_filt - Xtrue
        R2_s = 1 - sum(abs2, resid_s) / sum(abs2, Xtrue .- mean(Xtrue))
        @test R2_s > 0.7
        # Smoother MSE below filter MSE holds in EXPECTATION (RTS theory), not
        # on every single short-T draw — so the comparison is averaged over 10
        # seeds, where the systematic gap dominates draw noise.
        mses = map(1:10) do s
            sms = dgp_state_space(Random.Xoshiro(246 + s); F=Tt, H=Z, Q=RQR,
                                  R=Hobs, b=b, d=d, x0=a0, T=T_obs)
            ys = Matrix(sms.y')
            Xs = Matrix(sms.x')
            st = MEM.KalmanFilterStore{Float64}(n_state, T_obs)
            MEM._kalman_filter!(st, ys, Z, Tt, RQR, Hobs; d=d, b=b, a0=a0,
                               P0=P0, scalar=false)
            as, _ = MEM._rts_smoother(st, Tt)
            (sum(abs2, as - Xs) / length(Xs),
             sum(abs2, st.a_filt - Xs) / length(Xs))
        end
        @test mean(first.(mses)) < mean(last.(mses))

        # nlag: lag-1 smoothed cross-cov Plag[1][:,:,t] = Cov(x_t,x_{t-1}|Y_T) = P_{t|T} J_{t-1}'
        _, Ps_n, Plag = MEM._rts_smoother(store2, Tt; nlag=1)
        @test length(Plag) == 1
        @test all(Plag[1][:, :, 1] .== 0)                 # no lag-1 cross-cov at t=1
        for t in 2:T_obs
            Jm1 = store2.P_filt[:, :, t-1] * Tt' * inv(Symmetric(store2.P_pred[:, :, t]))
            @test Plag[1][:, :, t] ≈ Ps_n[:, :, t] * Jm1' rtol = 1e-9
        end

        # DSGE _kalman_loglikelihood is kept hand-tuned (a measured perf exception: a kernel
        # migration is byte-equivalent but 2.4x slower / 29x allocation on the Bayesian MLE
        # hot loop). This guards it byte-equivalent to the consolidated kernel so it cannot
        # drift from the shared numerics.
        nsd, nobs2, nshk = 6, 3, 3
        Gd = randn(rng, nsd, nsd); Gd = 0.6 * Gd / maximum(abs, eigvals(Gd))
        imp = randn(rng, nsd, nshk); Zd = randn(rng, nobs2, nsd)
        dd = zeros(nobs2); Hd = Matrix(0.1 * I, nobs2, nobs2); Qd = Matrix(1.0 * I, nshk, nshk)
        ssd = MEM.DSGEStateSpace{Float64}(Gd, imp, Zd, dd, Hd, Qd)
        RQRd = imp * Qd * imp'
        datad = zeros(nobs2, 50); xd = zeros(nsd)
        for tt in 1:50
            xd = Gd * xd + imp * randn(rng, nshk)
            datad[:, tt] = Zd * xd + cholesky(Symmetric(Hd)).L * randn(rng, nobs2)
        end
        P0d = Matrix(MEM.solve_lyapunov(Gd, imp))
        ll_dsge = MEM._kalman_loglikelihood(ssd, datad)
        ll_kern = MEM._kalman_filter!(nothing, datad, Zd, Gd, RQRd, Hd; d=dd, a0=zeros(nsd), P0=P0d, scalar=false)
        @test ll_dsge ≈ ll_kern rtol = 1e-10

        # init modes
        Ps = MEM._kalman_init(:stationary, Tt, RQR, 2)[2]
        @test norm(Ps - (Tt * Ps * Tt' + RQR)) < 1e-8                       # Lyapunov fixed point
        @test MEM._kalman_init(:kappa, Tt, RQR, 2; kappa=1e6)[2] ≈ 1e6 * Matrix(I, 2, 2)
        a_e, P_e = MEM._kalman_init(:explicit, Tt, RQR, 2; a0=[1.0, 2.0], P0=[2.0 0.0; 0.0 3.0])
        @test a_e == [1.0, 2.0] && P_e == [2.0 0.0; 0.0 3.0]
        @test_throws ArgumentError MEM._kalman_init(:bogus, Tt, RQR, 2)
        @test_throws ArgumentError MEM._kalman_init(:stationary, [1.01 0.0; 0.0 0.5], RQR, 2)
    end

    @testset "Innovations form: stored (v, F) reproduce the prediction-error likelihood" begin
        # DGP-06 (#795): the kernel's log-likelihood IS the innovations form
        # Σ_t log p(y_t | y_{1:t-1}); recompute it by hand from the stored
        # one-step errors v_t and covariances F_t, and check their definitions.
        MEM = MacroEconometricModels
        rng = Random.Xoshiro(1246)
        Tt = [0.5 0.1; -0.2 0.4]
        RQR = [0.30 0.05; 0.05 0.20]
        Z = [1.0 0.0; 0.3 1.0]
        Hobs = [0.10 0.0; 0.0 0.15]
        b = [0.2, -0.1]; d = [0.05, 0.02]
        T_obs = 60
        a0, P0 = MEM._kalman_init(:stationary, Tt, RQR, 2)
        sim = dgp_state_space(rng; F=Tt, H=Z, Q=RQR, R=Hobs, b=b, d=d, x0=a0, T=T_obs)
        y = Matrix(sim.y')

        store = MEM.KalmanFilterStore{Float64}(2, T_obs; innovations=true)
        ll = MEM._kalman_filter!(store, y, Z, Tt, RQR, Hobs; d=d, b=b, a0=a0, P0=P0,
                                 scalar=false)
        @test all(v -> length(v) == 2, store.v)       # fully observed: no ragged steps
        ll_hand = 0.0
        for t in 1:T_obs
            v, Ft = store.v[t], store.F[t]
            # definitions: v_t = y_t − d − Z a_{t|t-1}, F_t = Z P_{t|t-1} Z' + H
            @test v ≈ y[:, t] - d - Z * store.a_pred[:, t] rtol = 1e-9
            @test Ft ≈ Z * store.P_pred[:, :, t] * Z' + Hobs rtol = 1e-9
            L = cholesky(Symmetric((Ft + Ft') / 2)).L
            w = L \ v
            ll_hand += -0.5 * (2 * log(2π) + 2 * sum(log, diag(L)) + dot(w, w))
        end
        @test ll_hand ≈ ll rtol = 1e-10
    end

    @testset "Diffuse initialization" begin
        # DGP-06 (#795): :diffuse collapses to the stationary Lyapunov solution
        # on stable transitions, and puts κ on unit-root directions (with a
        # one-time warning, silenced here — the warning itself is covered by
        # the #512 testset in test/statespace/test_statespace.jl).
        MEM = MacroEconometricModels
        Tt = [0.5 0.1; -0.2 0.4]
        RQR = [0.30 0.05; 0.05 0.20]
        Pd = MEM._kalman_init(:diffuse, Tt, RQR, 2)[2]
        Ps = MEM._kalman_init(:stationary, Tt, RQR, 2)[2]
        @test Pd ≈ Ps rtol = 1e-8
        @test Pd != 1e6 * Matrix(I, 2, 2)          # genuinely not the κ prior
        Pu = MEM._suppress_warnings() do
            MEM._kalman_init(:diffuse, reshape([1.0], 1, 1), reshape([0.5], 1, 1),
                             1; kappa=1e6)[2]
        end
        @test Pu ≈ fill(1e6, 1, 1) rtol = 1e-8    # κ on the unit-root direction
    end

    @testset "Simulation smoother: draw mean matches RTS, draws vary" begin
        # DGP-06 (#795): Durbin–Koopman draws α̃ ~ p(α|y). The draw average must
        # recover the RTS smoother mean and the draw covariance the smoother
        # covariance; a mean-only "smoother" fails the variance checks.
        # Calibrated (seed 1247): max|drawmean − RTS|/scale ≈ 0.043 at S=200.
        MEM = MacroEconometricModels
        rng = Random.Xoshiro(1247)
        Tt = [0.5 0.1; -0.2 0.4]
        RQR = [0.30 0.05; 0.05 0.20]
        Z = [1.0 0.0; 0.3 1.0]
        Hobs = [0.10 0.0; 0.0 0.15]
        b = [0.2, -0.1]; d = [0.05, 0.02]
        T_obs = 60
        a0, P0 = MEM._kalman_init(:stationary, Tt, RQR, 2)
        sim = dgp_state_space(rng; F=Tt, H=Z, Q=RQR, R=Hobs, b=b, d=d, x0=a0, T=T_obs)
        y = Matrix(sim.y')
        store = MEM.KalmanFilterStore{Float64}(2, T_obs)
        MEM._kalman_filter!(store, y, Z, Tt, RQR, Hobs; d=d, b=b, a0=a0, P0=P0,
                            scalar=false)
        as, Ps, _ = MEM._rts_smoother(store, Tt)

        S = FAST ? 64 : 400
        acc = zeros(2, T_obs); acc2 = zeros(2, T_obs)
        for s in 1:S
            dr = MEM._simulation_smoother(rng, y, Z, Tt, RQR, Hobs; d=d, b=b,
                                          a0=a0, P0=P0)
            # T159: kept — per-draw smoke; the mean/variance-moment pins below carry the weight.
            @test all(isfinite, dr)
            acc .+= dr; acc2 .+= dr .^ 2
        end
        dmean = acc / S
        dvar = acc2 / S - dmean .^ 2
        scale = maximum(abs, as)
        @test maximum(abs, dmean - as) / scale < (FAST ? 0.2 : 0.08)
        # Draws genuinely vary, with roughly the smoother variance (MC-loose:
        # Var of a variance estimate over S draws is ~2/S).
        @test all(dvar .> 0)
        mid = T_obs ÷ 2
        for i in 1:2
            @test isapprox(dvar[i, mid], Ps[i, i, mid]; rtol=0.5)
        end
    end

    @testset "_solve_dare & steady_state_kalman (Issue #916)" begin
        MEM = MacroEconometricModels
        rng = Random.Xoshiro(916)

        # 1. Singular transition A = [0 1; 0 0] (uncorrelated and correlated S)
        A_sing = [0.0 1.0; 0.0 0.0]
        Q_sing = [0.5 0.1; 0.1 0.3]
        C_sing = [1.0 0.5]
        R_sing = reshape([0.2], 1, 1)
        P_s, K_s, F_s, res_s = MEM._solve_dare(A_sing, Q_sing, C_sing, R_sing)
        @test res_s < 1e-12
        @test issymmetric(P_s) || P_s ≈ P_s'
        @test minimum(eigvals(Symmetric(P_s))) >= -1e-12
        @test maximum(abs, eigvals(A_sing - K_s * C_sing)) < 1.0

        S_sing = [0.15; 0.05;;]
        P_sc, K_sc, F_sc, res_sc = MEM._solve_dare(A_sing, Q_sing, C_sing, R_sing; S=S_sing)
        @test res_sc < 1e-12
        @test maximum(abs, eigvals(A_sing - K_sc * C_sing)) < 1.0

        # 2. Unit-root A = [1.0] FVRSW (2007) Permanent Income with S = B * D' != 0
        R_gross = 1.2
        A_fv = reshape([1.0], 1, 1)
        B_fv = reshape([1.0 - 1.0 / R_gross], 1, 1)
        C_fv = reshape([-1.0], 1, 1)
        D_fv = reshape([1.0 / R_gross], 1, 1)
        S_fv = B_fv * D_fv'
        P_fv, K_fv, F_fv, res_fv = MEM._solve_dare(A_fv, B_fv * B_fv', C_fv, D_fv * D_fv'; S=S_fv)
        @test res_fv < 1e-12
        @test P_fv[1, 1] ≈ 11.0 / 36.0 atol = 1e-12
        @test K_fv[1, 1] ≈ -1.0 / 6.0 atol = 1e-12
        @test F_fv[1, 1] ≈ 1.0 atol = 1e-12
        @test maximum(abs, eigvals(A_fv - K_fv * C_fv)) < 1.0

        # 3. Singular / zero R via Van Dooren QR-compressed extended pencil
        P_r0, K_r0, F_r0, res_r0 = MEM._solve_dare(A_sing, Q_sing, C_sing, zeros(1, 1))
        @test res_r0 < 1e-12
        @test maximum(abs, eigvals(A_sing - K_r0 * C_sing)) < 1.0

        # 3b. Rank-deficient [C'; -S; R] (< m) adversarial cases (Issue #916)
        # (a) R = 0 with m = 3 > n = 2
        A_r3 = [0.5 0.1; 0.0 0.6]
        Q_r3 = [1.0 0.2; 0.2 1.0]
        C_r3 = [1.0 0.0; 0.0 1.0; 1.0 1.0]
        P_r3, K_r3, F_r3, res_r3 = MEM._solve_dare(A_r3, Q_r3, C_r3, zeros(3, 3))
        @test res_r3 < 1e-12
        @test minimum(eigvals(Symmetric(P_r3))) >= -1e-12
        @test maximum(abs, eigvals(A_r3 - K_r3 * C_r3)) < 1.0

        # (b) R = 0 with collinear C
        C_col = [1.0 0.0; 2.0 0.0]
        P_col, K_col, F_col, res_col = MEM._solve_dare(A_r3, Q_r3, C_col, zeros(2, 2))
        @test res_col < 1e-12
        @test minimum(eigvals(Symmetric(P_col))) >= -1e-12
        @test maximum(abs, eigvals(A_r3 - K_col * C_col)) < 1.0

        # (c) Rank-1 R = d*d' with rank-2 C (m = n = 3, d in colspace(C))
        A_dd = [0.5 0.1 0.0; 0.0 0.6 0.1; 0.0 0.0 0.4]
        Q_dd = Matrix{Float64}(I, 3, 3)
        C_dd = [1.0 0.0 0.0; 0.0 1.0 0.0; 1.0 1.0 0.0]
        d_dd = [1.0, 0.0, 1.0]
        P_dd, K_dd, F_dd, res_dd = MEM._solve_dare(A_dd, Q_dd, C_dd, d_dd * d_dd')
        @test res_dd < 1e-12
        @test minimum(eigvals(Symmetric(P_dd))) >= -1e-12
        @test maximum(abs, eigvals(A_dd - K_dd * C_dd)) < 1.0

        # (d) Rank-0 observation block (C = 0, R = 0)
        P_z0, K_z0, F_z0, res_z0 = MEM._solve_dare(A_r3, Q_r3, zeros(2, 2), zeros(2, 2))
        @test res_z0 < 1e-12
        @test P_z0 ≈ A_r3 * P_z0 * A_r3' + Q_r3 atol = 1e-11
        @test iszero(K_z0)

        # Unstabilizable / undetectable systems must throw ArgumentError
        @test_throws ArgumentError MEM._solve_dare([1.2 0.0; 0.0 0.5], Matrix{Float64}(I, 2, 2), zeros(0, 2), zeros(0, 0))
        @test_throws ArgumentError MEM._solve_dare([2.0 0.0; 0.0 0.5], zeros(2, 2), [0.0 1.0], ones(1, 1))
        @test_throws ArgumentError MEM._solve_dare([1.5 0.0; 0.0 0.5], Matrix{Float64}(I, 2, 2), [0.0 1.0], ones(1, 1))
        @test_throws ArgumentError MEM._solve_dare([1.2 0.0; 0.0 0.5], Matrix{Float64}(I, 2, 2), zeros(2, 2), zeros(2, 2))

        # 4. Dense n=20, m=5 correlated system: < 1e-12 residual in < 5 ms (single-thread BLAS)
        n20, m20 = 20, 5
        M20 = randn(rng, n20, n20)
        A20 = 0.8 * M20 / maximum(abs, eigvals(M20))
        C20 = randn(rng, m20, n20)
        W20 = randn(rng, n20 + m20, n20 + m20)
        Jnt20 = W20 * W20' / (n20 + m20) + 0.1 * I
        Q20 = Jnt20[1:n20, 1:n20]
        S20 = Jnt20[1:n20, (n20+1):end]
        R20 = Jnt20[(n20+1):end, (n20+1):end]
        P20, K20, F20, res20 = MEM._solve_dare(A20, Q20, C20, R20; S=S20)
        @test res20 < 1e-12
        @test maximum(abs, eigvals(A20 - K20 * C20)) < 1.0
        old_blas_threads = BLAS.get_num_threads()
        t_min_s = try
            BLAS.set_num_threads(1)
            MEM._solve_dare(A20, Q20, C20, R20; S=S20)  # single-thread warmup
            minimum(@elapsed(MEM._solve_dare(A20, Q20, C20, R20; S=S20)) for _ in 1:10)
        finally
            BLAS.set_num_threads(old_blas_threads)
        end
        @test t_min_s < 0.005

        # Dimension guards
        @test_throws ArgumentError MEM._solve_dare(ones(2, 3), Q_sing, C_sing, R_sing)
        @test_throws ArgumentError MEM._solve_dare(A_sing, ones(3, 3), C_sing, R_sing)
        @test_throws ArgumentError MEM._solve_dare(A_sing, Q_sing, ones(1, 3), R_sing)
        @test_throws ArgumentError MEM._solve_dare(A_sing, Q_sing, C_sing, ones(2, 2))
        @test_throws ArgumentError MEM._solve_dare(A_sing, Q_sing, C_sing, R_sing; S=ones(2, 2))

        # 5. steady_state_kalman & T=500 correlated _kalman_filter! convergence
        Tt = [0.6 0.15 -0.05; 0.1 0.5 0.2; -0.05 0.1 0.4]
        Z = [1.0 0.2 0.0; 0.0 1.0 -0.3]
        B_j = randn(rng, 5, 5)
        Sig_j = B_j * B_j' / 5 + 0.05 * I
        RQR = Sig_j[1:3, 1:3]; G = Sig_j[1:3, 4:5]; Hobs = Sig_j[4:5, 4:5]
        ssk = steady_state_kalman(Z, Hobs, Tt, RQR; G=G)
        @test ssk isa SteadyStateKalman{Float64}
        @test all(abs.(ssk.closed_loop_eigs) .< 1.0)
        @test ssk.F_innov ≈ Z * ssk.P_pred * Z' + Hobs atol = 1e-12

        L_j = cholesky(Symmetric(Sig_j)).L
        T_long = 500
        y_long = zeros(2, T_long)
        x_cur = zeros(3)
        for t in 1:T_long
            w = L_j * randn(rng, 5)
            y_long[:, t] = Z * x_cur + w[4:5]
            x_cur = Tt * x_cur + w[1:3]
        end
        st_long = MEM.KalmanFilterStore{Float64}(3, T_long; innovations=true)
        MEM._kalman_filter!(st_long, y_long, Z, Tt, RQR, Hobs;
                            a0=zeros(3), P0=Matrix{Float64}(I, 3, 3), G=G)
        @test norm(st_long.P_pred[:, :, end] - ssk.P_pred) < 1e-10
        @test norm(st_long.K[end] - ssk.K_gain) < 1e-10
        @test norm(st_long.P_filt[:, :, end] - ssk.P_filt) < 1e-10

        # Scalar correlated path: T=500 convergence + scalar==multivariate + G=0 equivalence
        Z1 = [1.0 0.3]
        Tt1 = [0.65 0.1; -0.05 0.5]
        B1 = randn(rng, 3, 3); Sig1 = B1 * B1' / 3 + 0.05 * I
        RQR1 = Sig1[1:2, 1:2]; G1 = Sig1[1:2, 3:3]; H1 = Sig1[3:3, 3:3]
        ssk1 = steady_state_kalman(Z1, H1, Tt1, RQR1; G=G1)
        y1 = randn(rng, 1, T_long)
        st1_sc = MEM.KalmanFilterStore{Float64}(2, T_long; innovations=true)
        st1_mv = MEM.KalmanFilterStore{Float64}(2, T_long; innovations=true)
        ll1_sc = MEM._kalman_filter!(st1_sc, y1, Z1, Tt1, RQR1, H1;
                                     a0=zeros(2), P0=Matrix{Float64}(I, 2, 2), scalar=true, G=G1)
        ll1_mv = MEM._kalman_filter!(st1_mv, y1, Z1, Tt1, RQR1, H1;
                                     a0=zeros(2), P0=Matrix{Float64}(I, 2, 2), scalar=false, G=G1)
        @test ll1_sc ≈ ll1_mv rtol = 1e-12
        @test norm(st1_sc.P_pred[:, :, end] - ssk1.P_pred) < 1e-10
        @test norm(st1_sc.K[end] - ssk1.K_gain) < 1e-10

        # G = zeros(...) matches G === nothing
        st_n = MEM.KalmanFilterStore{Float64}(3, 30; innovations=true)
        st_z = MEM.KalmanFilterStore{Float64}(3, 30; innovations=true)
        ll_n = MEM._kalman_filter!(st_n, y_long[:, 1:30], Z, Tt, RQR, Hobs;
                                   a0=zeros(3), P0=Matrix{Float64}(I, 3, 3))
        ll_z = MEM._kalman_filter!(st_z, y_long[:, 1:30], Z, Tt, RQR, Hobs;
                                   a0=zeros(3), P0=Matrix{Float64}(I, 3, 3), G=zeros(3, 2))
        @test ll_n ≈ ll_z atol = 1e-12
        @test st_n.a_filt ≈ st_z.a_filt atol = 1e-12
        @test st_n.P_filt ≈ st_z.P_filt atol = 1e-12
        as_n, Ps_n, Pl_n = MEM._rts_smoother(st_n, Tt; nlag=1)
        as_z, Ps_z, Pl_z = MEM._rts_smoother(st_z, Tt; nlag=1, G=zeros(3, 2), y=y_long[:, 1:30])
        @test as_n ≈ as_z atol = 1e-12
        @test Ps_n ≈ Ps_z atol = 1e-12
        @test Pl_n[1] ≈ Pl_z[1] atol = 1e-12
    end

    @testset "Correlated _rts_smoother (exact GLS oracle) & _simulation_smoother" begin
        MEM = MacroEconometricModels
        rng = Random.Xoshiro(1916)
        T6 = 6; ns = 2; no = 2
        Tt6 = [0.7 0.1; -0.1 0.6]; Z6 = [1.0 0.3; -0.2 1.0]
        B6 = randn(rng, ns + no, ns + no); Sig6 = B6 * B6' / 4 + 0.1 * I
        RQR6 = Sig6[1:ns, 1:ns]; G6 = Sig6[1:ns, (ns+1):end]; H6 = Sig6[(ns+1):end, (ns+1):end]
        b6 = [0.2, -0.1]; d6 = [0.05, -0.03]; a06 = [0.1, -0.2]; P06 = [0.5 0.1; 0.1 0.4]
        y6 = randn(rng, no, T6)

        st6 = MEM.KalmanFilterStore{Float64}(ns, T6; innovations=true)
        ll6 = MEM._kalman_filter!(st6, y6, Z6, Tt6, RQR6, H6; d=d6, b=b6, a0=a06, P0=P06, G=G6)
        as6, Ps6, Plag6 = MEM._rts_smoother(st6, Tt6; nlag=1, G=G6, y=y6)

        # Build exact stacked Gaussian joint (alpha_{1:T}, y_{1:T}) oracle
        N_a = ns * T6; N_y = no * T6
        mu_a = zeros(N_a); mu_y = zeros(N_y)
        mu_cur = b6 + Tt6 * a06
        for t in 1:T6
            mu_a[(t-1)*ns+1:t*ns] = mu_cur
            mu_y[(t-1)*no+1:t*no] = d6 + Z6 * mu_cur
            mu_cur = b6 + Tt6 * mu_cur
        end
        n_u = ns + ns + (T6 - 1) * (ns + no) + no
        Sigma_u = zeros(n_u, n_u)
        Sigma_u[1:ns, 1:ns] = P06
        Sigma_u[(ns+1):2ns, (ns+1):2ns] = RQR6
        for t in 1:(T6-1)
            idx = 2ns + (t - 1) * (ns + no)
            Sigma_u[(idx+1):(idx+ns+no), (idx+1):(idx+ns+no)] = [RQR6 G6; G6' H6]
        end
        idxT = n_u - no
        Sigma_u[(idxT+1):n_u, (idxT+1):n_u] = H6

        M_a = zeros(N_a, n_u); M_y = zeros(N_y, n_u)
        M_a[1:ns, 1:ns] = Tt6
        M_a[1:ns, (ns+1):2ns] = Matrix{Float64}(I, ns, ns)
        for t in 1:(T6-1)
            idx = 2ns + (t - 1) * (ns + no)
            M_y[(t-1)*no+1:t*no, :] = Z6 * M_a[(t-1)*ns+1:t*ns, :]
            M_y[(t-1)*no+1:t*no, (idx+ns+1):(idx+ns+no)] = Matrix{Float64}(I, no, no)
            M_a[t*ns+1:(t+1)*ns, :] = Tt6 * M_a[(t-1)*ns+1:t*ns, :]
            M_a[t*ns+1:(t+1)*ns, (idx+1):(idx+ns)] = Matrix{Float64}(I, ns, ns)
        end
        M_y[(T6-1)*no+1:T6*no, :] = Z6 * M_a[(T6-1)*ns+1:T6*ns, :]
        M_y[(T6-1)*no+1:T6*no, (idxT+1):n_u] = Matrix{Float64}(I, no, no)

        M_z = vcat(M_a, M_y)
        Sigma_z = M_z * Sigma_u * M_z'
        Saa = Sigma_z[1:N_a, 1:N_a]
        Say = Sigma_z[1:N_a, (N_a+1):end]
        Syy = Sigma_z[(N_a+1):end, (N_a+1):end]
        mu_cond = mu_a + Say * (Syy \ (vec(y6) - mu_y))
        P_cond = Saa - Say * (Syy \ Say')
        Lyy = cholesky(Symmetric(Syy)).L
        ll_exact = -0.5 * (N_y * log(2π) + 2 * sum(log, diag(Lyy)) + sum(abs2, Lyy \ (vec(y6) - mu_y)))

        @test ll6 ≈ ll_exact atol = 1e-12
        @test vec(as6) ≈ mu_cond atol = 1e-12
        for t in 1:T6
            @test Ps6[:, :, t] ≈ P_cond[(t-1)*ns+1:t*ns, (t-1)*ns+1:t*ns] atol = 1e-12
        end
        for t in 2:T6
            @test Plag6[1][:, :, t] ≈ P_cond[(t-1)*ns+1:t*ns, (t-2)*ns+1:(t-1)*ns] atol = 1e-12
        end

        # Partial and all-NaN missingness vs marginal Gaussian oracle
        y_nan = copy(y6); y_nan[1, 2] = NaN; y_nan[:, 4] .= NaN
        st_nan = MEM.KalmanFilterStore{Float64}(ns, T6; innovations=true)
        ll_nan = MEM._kalman_filter!(st_nan, y_nan, Z6, Tt6, RQR6, H6; d=d6, b=b6, a0=a06, P0=P06, G=G6)
        as_nan, Ps_nan, Plag_nan = MEM._rts_smoother(st_nan, Tt6; nlag=1, G=G6, y=y_nan)
        obs_idx = findall(!isnan, vec(y_nan))
        Syy_o = Syy[obs_idx, obs_idx]; Say_o = Say[:, obs_idx]
        mu_cond_nan = mu_a + Say_o * (Syy_o \ (vec(y_nan)[obs_idx] - mu_y[obs_idx]))
        P_cond_nan = Saa - Say_o * (Syy_o \ Say_o')
        Lyy_o = cholesky(Symmetric(Syy_o)).L
        ll_exact_nan = -0.5 * (length(obs_idx) * log(2π) + 2 * sum(log, diag(Lyy_o)) + sum(abs2, Lyy_o \ (vec(y_nan)[obs_idx] - mu_y[obs_idx])))
        @test ll_nan ≈ ll_exact_nan atol = 1e-12
        @test vec(as_nan) ≈ mu_cond_nan atol = 1e-12
        for t in 1:T6
            @test Ps_nan[:, :, t] ≈ P_cond_nan[(t-1)*ns+1:t*ns, (t-1)*ns+1:t*ns] atol = 1e-12
        end
        for t in 2:T6
            @test Plag_nan[1][:, :, t] ≈ P_cond_nan[(t-1)*ns+1:t*ns, (t-2)*ns+1:(t-1)*ns] atol = 1e-12
        end

        # Error checks for _rts_smoother with G
        st_noinnov = MEM.KalmanFilterStore{Float64}(ns, T6; innovations=false)
        @test_throws ArgumentError MEM._rts_smoother(st_noinnov, Tt6; G=G6)
        @test_throws ArgumentError MEM._rts_smoother(st_nan, Tt6; G=G6)

        # Correlated _simulation_smoother: sample mean & midpoint variance match RTS smoother
        S_sim = FAST ? 150 : 600
        acc = zeros(ns, T6); acc2 = zeros(ns, T6)
        for _ in 1:S_sim
            dr = MEM._simulation_smoother(rng, y6, Z6, Tt6, RQR6, H6;
                                          d=d6, b=b6, a0=a06, P0=P06, G=G6)
            @test all(isfinite, dr)
            acc .+= dr; acc2 .+= dr .^ 2
        end
        dmean = acc / S_sim
        dvar = acc2 / S_sim - dmean .^ 2
        @test maximum(abs, dmean - as6) < (FAST ? 0.25 : 0.12)
        for i in 1:ns
            @test isapprox(dvar[i, 3], Ps6[i, i, 3]; rtol=0.45)
        end

        # _simulation_smoother parity (G === nothing vs G = zeros) & NaN/non-stationary P0 oracle
        dr_nothing = MEM._simulation_smoother(MersenneTwister(42), y_nan, Z6, Tt6, RQR6, H6;
                                              d=d6, b=b6, a0=a06, P0=P06, G=nothing)
        dr_zeros = MEM._simulation_smoother(MersenneTwister(42), y_nan, Z6, Tt6, RQR6, H6;
                                            d=d6, b=b6, a0=a06, P0=P06, G=zeros(ns, no))
        @test dr_nothing == dr_zeros

        # G === nothing Monte Carlo vs exact stacked-Gaussian GLS oracle with NaNs, non-stationary P0, rank-def RQR
        ns3 = 3; no3 = 2; T5 = 5
        Tt3 = [0.7 0.1 0.0; 0.0 0.5 0.1; 0.0 0.0 0.4]
        Z3 = [1.0 0.2 0.0; 0.0 1.0 -0.3]
        R3 = [1.0 0.0; 0.3 1.0; 0.0 0.5]
        RQR3 = R3 * [0.5 0.1; 0.1 0.4] * R3'  # rank 2 < 3
        H3 = [0.2 0.0; 0.0 0.15]
        b3 = [0.1, -0.05, 0.02]; d3 = [0.03, -0.02]
        a03 = [0.5, -0.3, 0.2]; P03 = [1.2 0.2 0.0; 0.2 0.8 0.1; 0.0 0.1 0.6]  # non-stationary P0
        y3_nan = randn(rng, no3, T5)
        y3_nan[:, 3] .= NaN
        y3_nan[1, 4] = NaN

        # Stacked Gaussian oracle for G=0
        Na3 = ns3 * T5; Ny3 = no3 * T5
        mu_a3 = zeros(Na3); mu_y3 = zeros(Ny3)
        mc3 = b3 + Tt3 * a03
        for t in 1:T5
            mu_a3[(t-1)*ns3+1:t*ns3] = mc3
            mu_y3[(t-1)*no3+1:t*no3] = d3 + Z3 * mc3
            mc3 = b3 + Tt3 * mc3
        end
        nu3 = ns3 + T5 * ns3 + T5 * no3
        Su3 = zeros(nu3, nu3)
        Su3[1:ns3, 1:ns3] = P03
        for t in 1:T5
            Su3[ns3+(t-1)*ns3+1:ns3+t*ns3, ns3+(t-1)*ns3+1:ns3+t*ns3] = RQR3
            Su3[ns3+T5*ns3+(t-1)*no3+1:ns3+T5*ns3+t*no3, ns3+T5*ns3+(t-1)*no3+1:ns3+T5*ns3+t*no3] = H3
        end
        Ma3 = zeros(Na3, nu3); My3 = zeros(Ny3, nu3)
        Ma3[1:ns3, 1:ns3] = Tt3
        Ma3[1:ns3, ns3+1:2ns3] = I(ns3)
        for t in 2:T5
            Ma3[(t-1)*ns3+1:t*ns3, :] = Tt3 * Ma3[(t-2)*ns3+1:(t-1)*ns3, :]
            Ma3[(t-1)*ns3+1:t*ns3, ns3+(t-1)*ns3+1:ns3+t*ns3] = I(ns3)
        end
        for t in 1:T5
            My3[(t-1)*no3+1:t*no3, :] = Z3 * Ma3[(t-1)*ns3+1:t*ns3, :]
            My3[(t-1)*no3+1:t*no3, ns3+T5*ns3+(t-1)*no3+1:ns3+T5*ns3+t*no3] = I(no3)
        end
        Mz3 = vcat(Ma3, My3)
        Sz3 = Mz3 * Su3 * Mz3'
        obs3 = findall(!isnan, vec(y3_nan))
        Saa3 = Sz3[1:Na3, 1:Na3]; Say3 = Sz3[1:Na3, Na3 .+ obs3]; Syy3 = Sz3[Na3 .+ obs3, Na3 .+ obs3]
        mu_post3 = mu_a3 + Say3 * (Syy3 \ (vec(y3_nan)[obs3] - mu_y3[obs3]))
        P_post3 = Saa3 - Say3 * (Syy3 \ Say3')

        S_mc = FAST ? 300 : 2000
        draws3 = zeros(Na3, S_mc)
        rng_mc = Random.Xoshiro(424242)
        for s in 1:S_mc
            draws3[:, s] = vec(MEM._simulation_smoother(rng_mc, y3_nan, Z3, Tt3, RQR3, H3;
                                                        d=d3, b=b3, a0=a03, P0=P03))
        end
        mc_mean3 = vec(mean(draws3; dims=2))
        mc_cov3 = cov(draws3; dims=2)
        @test maximum(abs, mc_mean3 - mu_post3) < (FAST ? 0.15 : 0.06)
        @test maximum(abs, mc_cov3 - P_post3) < (FAST ? 0.18 : 0.07)
    end
end
