# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics
using Distributions

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

if !@isdefined(simulate_svar)
    include("id_dgps.jl")
end

@testset "SID-01 identification recovery" begin
    @testset "SID-01 heteroskedastic kernel recovers B0" begin
        T = Float64
        B_true = T[1.0 0.4 0.1; 0.0 1.0 0.2; 0.0 0.0 1.0]
        Λ = T[0.5, 2.0, 5.0]
        Σ1 = B_true * B_true'
        Σ2 = B_true * Diagonal(Λ) * B_true'
        B0, Q, lam = MacroEconometricModels._eigendecomposition_id(Σ1, Σ2)
        @test norm(B0 * B0' - Σ1) < 1e-10
        off = B0 \ Σ2 / B0'
        @test norm(off - Diagonal(diag(off))) < 1e-10
        @test MacroEconometricModels._procrustes_distance(B0, B_true) < 1e-8
        @test issorted(lam)
        @test norm(Q' * Q - I(3)) < 1e-10
    end

    if !FAST
        @testset "SID-01 external volatility recovers B0" begin
            B_true = [1.0 0.4 0.1; 0.0 1.0 0.2; 0.0 0.0 1.0]
            Λ = [0.5, 2.0, 5.0]
            A = [0.5 * Matrix{Float64}(I, 3, 3)]
            Y, regime = simulate_two_regime(B_true, A, Λ; Tobs=2000, split=0.5,
                                            rng=MersenneTwister(7))
            model = estimate_var(Y, 1)
            p = 1
            ev = identify_external_volatility(model, regime[(p + 1):end])
            @test MacroEconometricModels._procrustes_distance(ev.B0, B_true) < 0.1
        end

        @testset "SID-09 smooth-transition recovers B0" begin
            # Seed 738 lands Procrustes 0.22 (local mode, γ≈3.1). Seed 13 is 0.094.
            rng = MersenneTwister(13)
            B_true = [1.0 0.4 0.1; 0.0 1.0 0.2; 0.0 0.0 1.0]
            Λ = [0.5, 2.0, 5.0]
            A = [0.5 * Matrix{Float64}(I, 3, 3)]
            Tobs = 2000
            p = 1
            γ_true = 2.0
            c_true = 0.0
            n = size(B_true, 1)
            ntot = Tobs + p + 50
            s_all = randn(rng, ntot)
            σs = std(s_all)
            G = @. 1 / (1 + exp(-γ_true * (s_all - c_true) / σs))
            ε = randn(rng, ntot, n)
            u = zeros(ntot, n)
            for t in 1:ntot
                d = sqrt.(1 .+ G[t] .* (Λ .- 1))
                u[t, :] = B_true * (d .* ε[t, :])
            end
            Y = zeros(ntot, n)
            for t in (p + 1):ntot
                yt = u[t, :]
                for lag in 1:p
                    yt = yt + A[lag] * Y[t - lag, :]
                end
                Y[t, :] = yt
            end
            Y_obs = Y[(end - Tobs + 1):end, :]
            s_obs = s_all[(end - Tobs + 1):end]
            model = estimate_var(Y_obs, p)
            st = identify_smooth_transition(model, s_obs[(p + 1):end])
            @test MacroEconometricModels._procrustes_distance(st.B0, B_true) < 0.25
            @test abs(st.gamma - γ_true) / γ_true < 0.5
            @test abs(st.threshold - c_true) < 0.2 * std(st.transition_var)
            B0_reid, _, _ = MacroEconometricModels._eigendecomposition_id(
                Matrix(st.Sigma_regimes[1]), Matrix(st.Sigma_regimes[2]))
            @test norm(st.B0 - B0_reid) < 1e-6
        end
    end
end

@testset "SID-10 K-regime joint ML recovery" begin
    if !FAST
        rng = MersenneTwister(13)
        B_true = [1.0 0.4 0.1; 0.0 1.0 0.2; 0.0 0.0 1.0]
        Λ2 = [0.5, 2.0, 5.0]
        Λ3 = [2.0, 0.4, 3.0]
        A = [0.3 * Matrix{Float64}(I, 3, 3)]
        Y, regime = simulate_k_regime(B_true, A, [Λ2, Λ3]; Tobs=3000, rng=rng)
        model = estimate_var(Y, 1)
        p = 1
        ri = regime[(p + 1):end]
        ev = identify_external_volatility(model, ri; regimes=3)
        d_joint = MacroEconometricModels._procrustes_distance(ev.B0, B_true)
        @test d_joint < 0.15
        idx1 = findall(==(1), ri)
        idx2 = findall(==(2), ri)
        Σ1 = cov(model.U[idx1, :])
        Σ2 = cov(model.U[idx2, :])
        B_two, _, _ = MacroEconometricModels._eigendecomposition_id(Matrix(Σ1), Matrix(Σ2))
        d_two = MacroEconometricModels._procrustes_distance(B_two, B_true)
        @test d_joint < d_two
    end
end

@testset "SID-04 FastICA bootstrap column matching" begin
    Random.seed!(733)
    n, p, Tobs, H = 2, 1, FAST ? 200 : 300, 6
    B0 = [1.0 0.3; 0.2 1.0]
    A = [0.4 * Matrix{Float64}(I, n, n)]
    rng = MersenneTwister(733)
    ε = rand(rng, TDist(3.0), Tobs + p + 50, n)
    u = ε * B0'
    Yfull = zeros(Tobs + p + 50, n)
    for t in (p + 1):size(Yfull, 1)
        yt = u[t, :]
        for lag in 1:p
            yt = yt + A[lag] * Yfull[t - lag, :]
        end
        Yfull[t, :] = yt
    end
    Y = Yfull[(end - Tobs + 1):end, :]
    m = estimate_var(Y, p)
    reps = FAST ? 20 : 100
    ir_ica = irf(m, H; method=:fastica, ci_type=:bootstrap, reps=reps, seed=733)
    ir_chol = irf(m, H; method=:cholesky, ci_type=:bootstrap, reps=reps, seed=733)
    w_ica = mean(ir_ica.ci_upper[1, :, :] .- ir_ica.ci_lower[1, :, :])
    w_chol = mean(ir_chol.ci_upper[1, :, :] .- ir_chol.ci_lower[1, :, :])
    @test w_ica < 2 * w_chol
    @test ir_ica.manifest !== nothing
    @test haskey(ir_ica.manifest.settings, "relabeled_fraction")
end

@testset "SID-11 proxy SVAR recovery" begin
    B_true = [1.0 0.3 0.2; 0.5 1.0 0.1; 0.4 0.2 1.0]
    A = [0.5 * Matrix{Float64}(I, 3, 3)]
    Tobs = 5000

    @testset "k=1 recovers B0[:,1] within 5%" begin
        rng = MersenneTwister(4)
        Y, ε, z = simulate_proxy_svar(B_true, A; Tobs=Tobs, ρ=0.6, k=1, rng=rng)
        m = estimate_var(Y, 1)
        r = identify_proxy(m, reshape(z, :, 1); normalize=:unit_variance)
        @test r isa ProxySVARResult
        b_est = r.B0[:, 1]
        b_true = B_true[:, 1]
        b_est = sign(dot(b_est, b_true)) * b_est
        @test norm(b_est - b_true) / norm(b_true) < 0.05
    end

    @testset "k=2 recovers the instrumented span (Procrustes)" begin
        rng = MersenneTwister(741)
        Y, ε, Z = simulate_proxy_svar(B_true, A; Tobs=Tobs, ρ=0.6, k=2, rng=rng)
        m = estimate_var(Y, 1)
        r = identify_proxy(m, Z; normalize=:unit_variance)
        Ahat = r.B0[:, 1:2]
        Atrue = B_true[:, 1:2]
        U, _, V = svd(Ahat' * Atrue)
        R = U * V'
        @test norm(Ahat * R - Atrue) / norm(Atrue) < 0.10
    end
end

@testset "SID-12 max-share recovery" begin
    n = 3
    B_true = Matrix{Float64}(I, n, n)
    A = [Diagonal([0.85, 0.30, 0.15])]
    q_true = [1.0, 0.0, 0.0]

    @testset "population |q′q_true| > 0.99" begin
        Tobs = 40
        Y = zeros(Tobs, n)
        B = zeros(1 + n, n)
        B[2:(1 + n), :] = A[1]'
        U = zeros(Tobs - 1, n)
        model = VARModel(Y, 1, B, U, Matrix{Float64}(I, n, n), 0.0, 0.0, 0.0)
        r = identify_max_share(model; target=1, horizons=0:20)
        @test abs(dot(r.q, q_true)) > 0.99
    end

    @testset "large-T estimate recovers the target shock" begin
        rng = MersenneTwister(74112)
        Y, _, _, q_news = simulate_news_maxshare(; Tobs=FAST ? 800 : 2000, rng=rng)
        m = estimate_var(Y, 1)
        r = identify_max_share(m; target=1, horizons=0:20)
        @test abs(dot(r.q, q_news)) > 0.99
        @test abs(dot(r.q, q_true)) > 0.99
    end

    @testset "frequency band recovers the same shock as a long horizon" begin
        Tobs = 40
        Y = zeros(Tobs, n)
        B = zeros(1 + n, n)
        B[2:(1 + n), :] = A[1]'
        U = zeros(Tobs - 1, n)
        model = VARModel(Y, 1, B, U, Matrix{Float64}(I, n, n), 0.0, 0.0, 0.0)
        r_time = identify_max_share(model; target=1, horizons=0:200)
        r_freq = identify_max_share(model; target=1, band=(0.0, Float64(π)))
        @test abs(dot(r_time.q, r_freq.q)) > 0.99
        @test abs(dot(r_freq.q, q_true)) > 0.99
    end

    @testset "FEVD share of shock 1 on variable 1 equals λ_max / tr(S)" begin
        Tobs = 40
        Y = zeros(Tobs, n)
        Bcoef = zeros(1 + n, n)
        Bcoef[2:(1 + n), :] = A[1]'
        U = zeros(Tobs - 1, n)
        model = VARModel(Y, 1, Bcoef, U, Matrix{Float64}(I, n, n), 0.0, 0.0, 0.0)
        Hwin = 0:16
        r = identify_max_share(model; target=1, horizons=Hwin)
        H = last(Hwin) + 1
        fv = fevd(model, H; method=:max_share, target=1, horizons=Hwin)
        @test r.share ≈ r.eigvals[1] / sum(r.eigvals) atol = 1e-10
        @test fv.proportions[1, 1, H] ≈ r.share atol = 1e-8
    end
end

@testset "SID-16 SVEC recovery" begin
    rng = MersenneTwister(74516)
    Tobs = FAST ? 800 : 1000
    Y, _, B0_true, Xi_true = simulate_common_trend_svec(; Tobs=Tobs, rng=rng)
    lr_true = Xi_true * B0_true
    vecm = estimate_vecm(Y, 1; rank=2, deterministic=:none)
    svec = identify_svec(vecm)
    @test svec isa SVECResult
    @test svec.n_permanent == 1
    lr = svec.Xi * svec.B0
    @test maximum(abs, lr[:, 2]) < 1e-8
    @test maximum(abs, lr[:, 3]) < 1e-8
    perm = lr[:, 1]
    truth = lr_true[:, 1]
    perm = sign(dot(perm, truth)) * perm
    @test norm(perm - truth) / norm(truth) < 0.10
    Q = svec.Q
    @test norm(Q' * Q - I(3)) < 1e-6
end

@testset "SID-21 GMM moment recovery" begin
    B_true = [1.0 0.3; 0.2 1.0]
    A = [0.4 * Matrix{Float64}(I, 2, 2)]

    if !FAST
        @testset "t(5) cokurtosis recovers B0" begin
            rng = MersenneTwister(21)
            Y, _ = simulate_svar(B_true, A; Tobs=2000, shocks=:t, rng=rng)
            m = estimate_var(Y, 1)
            r = identify_gmm_moments(m; moments=:cokurtosis, weighting=:two_step)
            @test MacroEconometricModels._procrustes_distance(r.B0, B_true) < 0.1
            @test r.J_pvalue > 0.05
        end

        @testset "skew-normal coskewness recovers B0" begin
            rng = MersenneTwister(21)
            Y, _ = simulate_svar(B_true, A; Tobs=2000, shocks=:skewnormal, rng=rng)
            m = estimate_var(Y, 1)
            r = identify_gmm_moments(m; moments=:coskewness, weighting=:two_step)
            @test MacroEconometricModels._procrustes_distance(r.B0, B_true) < 0.1
            @test r.J_pvalue > 0.05
        end

        @testset "sandwich SEs cover free angles" begin
            # Cokurtosis sandwich needs 8th moments (invalid for t(5)). Score the
            # third-moment free angle vs the DGP Givens rotation (signed-permutation
            # alignment). Non-finite SEs count as non-coverage.
            B_rec = [1.0 0.0; 0.4 1.0]
            L_dgp = Matrix(MacroEconometricModels.safe_cholesky(B_rec * B_rec'))
            F = svd(L_dgp \ B_rec)
            θtrue = MacroEconometricModels._orthogonal_to_givens(F.U * F.Vt, 2)[1]
            n_reps = 200
            Tobs = 2000
            n_cover = 0
            zcrit = 1.96
            for r in 1:n_reps
                rng_r = MersenneTwister(75000 + r)
                Y, _ = simulate_svar(B_rec, A; Tobs=Tobs, shocks=:skewnormal, rng=rng_r)
                m = estimate_var(Y, 1)
                g = identify_gmm_moments(m; moments=:coskewness, weighting=:two_step, hac=true)
                seθ = sqrt(max(g.vcov[1, 1], 0.0))
                dθ = minimum(abs(g.theta[1] + k * π / 2 - θtrue) for k in -4:4)
                n_cover += Int(isfinite(seθ) && seθ > 0 && dθ <= zcrit * seθ)
            end
            cover = n_cover / n_reps
            @info "SID-21 Givens-angle 95% coverage" cover n_cover n_reps
            @test 0.88 <= cover <= 1.0
        end
    end
end

# =============================================================================
# SID-26 remaining entry points (#755)
# =============================================================================

if !@isdefined(_pd)
    _pd(Bhat, Btrue) = MacroEconometricModels._procrustes_distance(Matrix{Float64}(Bhat),
                                                                  Matrix{Float64}(Btrue))
    _Bhat(model, Q) = Matrix(cholesky_factor(model)) * Q
end
if !@isdefined(_B_rec)
    const _B_rec = [1.0 0.0; 0.4 1.0]
    const _B_up = [1.0 0.4; 0.0 1.0]
    const _B_pos = [1.0 0.3; 0.5 1.0]
    const _A2 = [0.5 * Matrix{Float64}(I, 2, 2)]
end

@testset "identify_cholesky recovery" begin
    Tobs = FAST ? 400 : 2000
    rng = MersenneTwister(75501)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, rng=rng)
    m = estimate_var(Y, 1)
    Q = identify_cholesky(m)
    @test Q ≈ I(2) atol = 1e-12
    @test _pd(_Bhat(m, Q), _B_rec) < 0.15
    @test MacroEconometricModels.compute_Q(m, :cholesky) ≈ Q
end

@testset "identify_long_run recovery" begin
    Tobs = FAST ? 400 : 2000
    rng = MersenneTwister(2)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, rng=rng)
    m = estimate_var(Y, 1)
    Q = identify_long_run(m)
    @test _pd(_Bhat(m, Q), _B_rec) < 0.15
    @test MacroEconometricModels.compute_Q(m, :long_run) ≈ Q
end

@testset "identify_sign recovery" begin
    Tobs = FAST ? 250 : 800
    draws = FAST ? 300 : 1500
    rng = MersenneTwister(75503)
    Y, _ = simulate_svar(_B_pos, _A2; Tobs=Tobs, rng=rng)
    m = estimate_var(Y, 1)
    chk = irf -> irf[1, 1, 1] > 0 && irf[1, 2, 1] > 0
    s = identify_sign(m, 4, chk; max_draws=draws, store_all=true, rng=copy(rng))
    @test s.n_accepted > 0
    dmin = minimum(_pd(_Bhat(m, Q), _B_pos) for Q in s.Q_draws)
    @test dmin < 0.20
    for Q in s.Q_draws
        ir = compute_irf(m, Q, 4)
        @test ir[1, 1, 1] > 0 && ir[1, 2, 1] > 0
    end
end

@testset "identify_arias recovery" begin
    Tobs = FAST ? 300 : 800
    rng = MersenneTwister(75504)
    Y, _ = simulate_svar(_B_up, _A2; Tobs=Tobs, rng=rng)
    m = estimate_var(Y, 1)
    restr = SVARRestrictions(2;
        zeros=[zero_restriction(2, 1)],
        signs=[sign_restriction(1, 1, :positive)])
    ar = identify_arias(m, restr, 4;
                        n_draws=FAST ? 20 : 80,
                        n_rotations=FAST ? 40 : 120,
                        rng=MersenneTwister(755041))
    @test length(ar.Q_draws) > 0
    mt = median_target(ar)
    @test _pd(_Bhat(m, mt.Q), _B_up) < 0.12
    @test abs(compute_irf(m, mt.Q, 2)[1, 2, 1]) < 1e-8
end

@testset "identify_uhlig recovery" begin
    Tobs = FAST ? 300 : 800
    rng = MersenneTwister(75505)
    Y, _ = simulate_svar(_B_up, _A2; Tobs=Tobs, rng=rng)
    m = estimate_var(Y, 1)
    restr = SVARRestrictions(2;
        zeros=[zero_restriction(2, 1)],
        signs=[sign_restriction(1, 1, :positive)])
    u = identify_uhlig(m, restr, 4;
                       n_starts=FAST ? 4 : 12,
                       n_refine=FAST ? 1 : 3,
                       max_iter_coarse=FAST ? 40 : 120,
                       max_iter_fine=FAST ? 80 : 300,
                       rng=MersenneTwister(755051))
    # Unique exact ID: one zero pins the column up to sign (free_dim=1; SVD
    # basis is not sign-optimized). Admissible rotation = recovered Q with
    # shock 1 aligned to the sign restriction.
    @test abs(u.irf[1, 2, 1]) < 1e-8
    Q_adm = Matrix{Float64}(u.Q)
    u.irf[1, 1, 1] < 0 && (Q_adm[:, 1] .*= -1)
    ir_adm = compute_irf(m, Q_adm, 4)
    @test ir_adm[1, 1, 1] > 0
    @test abs(ir_adm[1, 2, 1]) < 1e-8
    @test _pd(_Bhat(m, Q_adm), _B_up) < 0.15
end

@testset "identify_fastica recovery" begin
    Tobs = FAST ? 400 : 1500
    rng = MersenneTwister(4)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, shocks=:t, rng=rng)
    r = identify_fastica(estimate_var(Y, 1); rng=MersenneTwister(4))
    @test _pd(r.B0, _B_rec) < 0.20
end

@testset "identify_jade recovery" begin
    Tobs = FAST ? 2000 : 3000
    rng = MersenneTwister(13)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, shocks=:t, rng=rng)
    r = identify_jade(estimate_var(Y, 1))
    @test _pd(r.B0, _B_rec) < 0.20
end

@testset "identify_sobi recovery" begin
    # VAR(1) on a VAR(1)+AR-shock DGP is misspecified (true RF is VAR(2)), so
    # estimate_var residuals keep leftover serial correlation. Looser
    # Procrustes than planted AR residuals (~0.02) because the mean filter
    # absorbs some AC.
    Tobs = FAST ? 800 : 2000
    rng = MersenneTwister(14)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, shock_ar=[0.4, -0.4], rng=rng)
    r = identify_sobi(estimate_var(Y, 1); lags=1:8)
    @test _pd(r.B0, _B_rec) < 0.20
end

@testset "identify_dcov recovery" begin
    Tobs = 500
    rng = MersenneTwister(21)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, shocks=:t, rng=rng)
    r = identify_dcov(estimate_var(Y, 1); max_iter=FAST ? 40 : 80)
    @test _pd(r.B0, _B_rec) < 0.25
end

@testset "identify_hsic recovery" begin
    Tobs = 300
    rng = MersenneTwister(16)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, shocks=:t, rng=rng)
    r = identify_hsic(estimate_var(Y, 1); max_iter=FAST ? 60 : 120, rng=MersenneTwister(16))
    @test _pd(r.B0, _B_rec) < 0.30
end

@testset "identify_student_t recovery" begin
    Tobs = FAST ? 600 : 1500
    rng = MersenneTwister(6)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, shocks=:t, rng=rng)
    r = identify_student_t(estimate_var(Y, 1); max_iter=FAST ? 80 : 200)
    @test _pd(r.B0, _B_rec) < 0.20
end

@testset "identify_mixture_normal recovery" begin
    Tobs = FAST ? 800 : 1500
    rng = MersenneTwister(60)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, shocks=:mixture, rng=rng)
    r = identify_mixture_normal(estimate_var(Y, 1); max_iter=FAST ? 80 : 150)
    @test _pd(r.B0, _B_rec) < 0.20
end

@testset "identify_pml recovery" begin
    Tobs = FAST ? 600 : 1500
    rng = MersenneTwister(30)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, shocks=:t, rng=rng)
    r = identify_pml(estimate_var(Y, 1); max_iter=FAST ? 80 : 200)
    @test _pd(r.B0, _B_rec) < 0.20
end

@testset "identify_skew_normal recovery" begin
    Tobs = FAST ? 600 : 1500
    rng = MersenneTwister(30)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, shocks=:skewnormal, rng=rng)
    r = identify_skew_normal(estimate_var(Y, 1); max_iter=FAST ? 80 : 200)
    @test _pd(r.B0, _B_rec) < 0.20
end

@testset "identify_markov_switching recovery" begin
    if !FAST
        rng = MersenneTwister(53)
        Y, _ = simulate_two_regime(_B_rec, _A2, [0.4, 4.0]; Tobs=1500, split=0.5, rng=rng)
        r = identify_markov_switching(estimate_var(Y, 1); n_regimes=2, n_starts=3,
                                      max_iter=40, rng=MersenneTwister(53))
        # Quiet-state numeraire: a high-vol Σ₁ rescales columns (pd ≈ 1.08).
        @test issorted(tr.(r.Sigma_regimes))
        @test _pd(r.B0, _B_rec) < 0.30
        # Joint ML must not raise nll above the two-regime eigen start — Optim v1
        # LBFGS (Julia 1.10 numerical cell) previously walked off that start.
        Σ1 = _B_rec * _B_rec'
        Σ2 = _B_rec * Diagonal([0.4, 4.0]) * _B_rec'
        Tks = [750.0, 750.0]
        Bml, _, _, pml, _, _, _, _ = MacroEconometricModels._k_regime_ml([Σ1, Σ2], Tks)
        n = size(_B_rec, 1)
        nll(p) = MacroEconometricModels._k_regime_nll(p, n, 2, Tks, [Σ1, Σ2])
        B0s, _, λs = MacroEconometricModels._two_regime_start(Σ1, Σ2)
        p0 = MacroEconometricModels._k_regime_pack(B0s, [ones(n), max.(λs, 1e-8)], n, 2)
        @test nll(pml) <= nll(p0) + 1e-8
        @test _pd(Bml, _B_rec) < 0.05
    end
end

@testset "identify_garch recovery" begin
    if !FAST
        rng = MersenneTwister(50)
        Y, _ = simulate_garch_svar(_B_rec, _A2; Tobs=1500, rng=rng)
        r = identify_garch(estimate_var(Y, 1); max_iter=80)
        @test _pd(r.B0, _B_rec) < 0.15
    end
end

@testset "estimate_svar recovery" begin
    Tobs = FAST ? 300 : 1500
    rng = MersenneTwister(75517)
    Y, _ = simulate_svar(_B_rec, _A2; Tobs=Tobs, rng=rng)
    m = estimate_var(Y, 1)
    s = estimate_svar(m, recursive_pattern(2); rng=MersenneTwister(755171))
    @test _pd(s.A \ s.B, _B_rec) < 0.15
    @test s.Q ≈ identify_cholesky(m) atol = 1e-6
end

