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

const MEM = MacroEconometricModels

if !@isdefined(simulate_svar)
    include(joinpath(@__DIR__, "..", "var", "id_dgps.jl"))
end

_ar1_coef(x::AbstractVector) = cor(x[1:(end - 1)], x[2:end])

@testset "TVV common infra (#823)" begin

    @testset "simulate_sv_shocks" begin
        E1, H1 = MEM.simulate_sv_shocks(Xoshiro(11), 500, 2)
        E2, H2 = MEM.simulate_sv_shocks(Xoshiro(11), 500, 2)
        @test E1 == E2 && H1 == H2  # determinism under explicit seed
        @test size(E1) == (500, 2) && size(H1) == (500, 2)
        @test all(isfinite, E1) && all(isfinite, H1)

        # Distinct default persistence: AR(1) of log-vol recovers default rhos
        _, Hb = MEM.simulate_sv_shocks(Xoshiro(12), 20_000, 2)
        rho_hat = [_ar1_coef(Hb[:, j]) for j in 1:2]
        @test rho_hat[1] ≈ 0.97 atol = 0.02
        @test rho_hat[2] ≈ 0.85 atol = 0.02
        @test abs(rho_hat[1] - rho_hat[2]) > 0.1

        # sigma = 0 yields a homoskedastic shock (constant log-vol)
        _, H0 = MEM.simulate_sv_shocks(Xoshiro(13), 200, 2; sigmas=[0.2, 0.0])
        @test all(iszero, H0[:, 2])

        @test_throws ArgumentError MEM.simulate_sv_shocks(Xoshiro(1), 100, 2; rhos=[0.9, 1.0])
        @test_throws ArgumentError MEM.simulate_sv_shocks(Xoshiro(1), 100, 2; sigmas=[0.2, -0.1])
        @test_throws ArgumentError MEM.simulate_sv_shocks(Xoshiro(1), 100, 2; rhos=[0.9])
    end

    @testset "simulate_tvv_dgp" begin
        for kind in (:sv, :markov)
            Y1, A1, B1, U1 = MEM.simulate_tvv_dgp(Xoshiro(21), 2, 1, 500; kind=kind)
            Y2, _, _, _ = MEM.simulate_tvv_dgp(Xoshiro(21), 2, 1, 500; kind=kind)
            @test Y1 == Y2  # determinism
            @test size(Y1) == (500, 2) && size(U1) == (500, 2)
            @test size(B1) == (2, 2) && length(A1) == 1
            # Recovered shocks have ~unit variance (standardized pre-trim)
            Erec = U1 / B1'
            @test vec(mean(Erec .^ 2; dims=1)) ≈ ones(2) atol = 0.15
            # VAR recursion consistency
            @test Y1[2, :] ≈ A1[1] * Y1[1, :] + U1[2, :]
        end
        @test_throws ArgumentError MEM.simulate_tvv_dgp(Xoshiro(1), 2, 1, 100; kind=:bogus)
        B0c = [1.0 0.5; -0.2 1.0]
        _, _, Bgot, _ = MEM.simulate_tvv_dgp(Xoshiro(2), 2, 1, 100; B0=B0c)
        @test Bgot == B0c
    end

    @testset "align_Q / q_distance" begin
        rng = Xoshiro(31)
        Q0 = MEM.generate_Q(3; rng=rng)
        P = [2, 3, 1]
        S = [1.0, -1.0, 1.0]
        Qh = Q0[:, P] .* S'
        Qa, p_out, s_out = MEM.align_Q(Qh, Q0)
        @test p_out == P && s_out == S
        @test Qa ≈ Qh
        @test MEM.q_distance(Qh, Q0) < 1e-12
        Qrand = MEM.generate_Q(3; rng=rng)
        @test MEM.q_distance(Qrand, Q0) > 1e-3
        # Triangle sanity
        Qmid = MEM.generate_Q(3; rng=rng)
        @test MEM.q_distance(Qh, Qrand) <=
            MEM.q_distance(Qh, Qmid) + MEM.q_distance(Qmid, Qrand) + 1e-12
        # Greedy path (n > 5) recovers an exact signed permutation
        Q6 = MEM.generate_Q(6; rng=rng)
        Q6t = Q6[:, [3, 1, 2, 6, 4, 5]] .* [1.0 -1.0 1.0 1.0 -1.0 1.0]
        @test MEM.q_distance(Q6t, Q6) < 1e-10
        Qa6, p6, _ = MEM.align_Q(Q6t, Q6)
        @test p6 == [3, 1, 2, 6, 4, 5]
        @test Qa6 ≈ Q6t
        @test_throws DimensionMismatch MEM.align_Q(randn(Xoshiro(84), 3, 3), randn(Xoshiro(85), 3, 2))
        @test_throws DimensionMismatch MEM.q_distance(randn(Xoshiro(86), 3, 3), randn(Xoshiro(87), 2, 2))
    end

    @testset "check_orthogonal" begin
        Q = MEM.generate_Q(4; rng=Xoshiro(41))
        @test MEM.check_orthogonal(Q)
        @test MEM.check_orthogonal(Q; atol=1e-12)
        @test !MEM.check_orthogonal(randn(Xoshiro(42), 4, 4))
        @test !MEM.check_orthogonal(randn(Xoshiro(43), 4, 3))
        @test_throws ArgumentError MEM.check_orthogonal(Q; atol=-1.0)
    end

    @testset "estimate_gmm kernel covers TVV needs (gap check)" begin
        # Tiny linear-moment problem: all weighting modes + HAC run with no new code
        rng = Xoshiro(51)
        X = randn(rng, 200, 2)
        β = [1.5, -2.0]
        y = X * β + randn(rng, 200)
        data = (X=X, y=y)
        mfn(θ, d) = (d.y .- d.X * θ) .* d.X
        for w in (:identity, :two_step, :iterated)
            g = MEM.estimate_gmm(mfn, [0.0, 0.0], data; weighting=w, hac=true)
            @test isfinite(g.J_stat)
            @test g.theta ≈ β atol = 0.3
        end
    end

    @testset "generate_tvv_var / generate_sv_var fixtures" begin
        Y, B0t, Q0t, At = generate_tvv_var(; n=2, p=1, Tobs=500, rng=Xoshiro(61))
        @test size(Y) == (500, 2) && size(B0t) == (2, 2) && size(Q0t) == (2, 2)
        @test length(At) == 1
        @test norm(Q0t' * Q0t - I) < 1e-12
        @test B0t ≈ cholesky(Symmetric(B0t * B0t')).L * Q0t
        Yb, _, _, _ = generate_tvv_var(; n=2, p=1, Tobs=500, rng=Xoshiro(61))
        @test Y == Yb
        Ys, _, Q0s, As = generate_sv_var(; n=3, p=2, Tobs=400, rng=Xoshiro(62))
        @test size(Ys) == (400, 3) && length(As) == 2
        @test norm(Q0s' * Q0s - I) < 1e-12
    end

end
