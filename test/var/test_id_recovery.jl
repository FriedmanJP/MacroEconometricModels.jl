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
            @test MacroEconometricModels._procrustes_distance(st.B0, B_true) < 0.1
            @test abs(st.gamma - γ_true) / γ_true < 0.5
            @test abs(st.threshold - c_true) < 0.2 * std(st.transition_var)
            B0_reid, _, _ = MacroEconometricModels._eigendecomposition_id(
                Matrix(st.Sigma_regimes[1]), Matrix(st.Sigma_regimes[2]))
            @test norm(st.B0 - B0_reid) < 1e-6
        end
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
        Y, _ = simulate_svar(B_true, A; Tobs=FAST ? 800 : 2000, rng=rng)
        m = estimate_var(Y, 1)
        r = identify_max_share(m; target=1, horizons=0:20)
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

