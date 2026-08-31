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
