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

include(joinpath(@__DIR__, "..", "var", "id_dgps.jl"))

@testset "Lewis TVV-ID (#824)" begin
    # No FAST reduction in this file: recovery thresholds need full T (see #824).

    @testset "moments vanish at truth, not at random Q" begin
        _, _, B0t, Ut = MEM.simulate_tvv_dgp(Xoshiro(101), 2, 1, 10_000; kind=:sv)
        E = Matrix{Float64}(Ut / B0t')  # true unit-variance shocks
        data = (Z=E, n=2, lags=collect(1:5))
        M0 = MEM._lewis_tvv_moments(zeros(1), data)
        Mr = MEM._lewis_tvv_moments([0.7], data)
        n0 = norm(vec(mean(M0; dims=1)))
        nr = norm(vec(mean(Mr; dims=1)))
        @test n0 < 0.5 * nr   # truth an order below a 40° rotation
        @test nr > 0.1        # wrong Q gives materially nonzero moments
        @test n0 < 0.5
    end

    @testset "recovers B on SV fixture (n=2)" begin
        # Lewis-GMM moments vanish to first order at truth (fourth-order
        # objective), so recovery is draw-sensitive; assert mean/max over draws.
        qs = Float64[]
        for s in (102, 202, 302)
            Y, _, Q0t, _ = generate_sv_var(; n=2, Tobs=20000, rng=Xoshiro(s),
                rhos=[0.97, 0.85], sigmas=[0.25, 0.20])
            m = MEM.estimate_var(Y, 1)
            r = MEM.identify_lewis_tvv(m; rng=Xoshiro(s + 1))
            push!(qs, MEM.q_distance(r.Q, Q0t))
            @test r.weak_id == false
        end
        @test maximum(qs) < 0.4   # no breakdown on any draw
        @test sum(qs) / length(qs) < 0.25
        Y, _, _, _ = generate_sv_var(; n=2, Tobs=20000, rng=Xoshiro(102),
            rhos=[0.97, 0.85], sigmas=[0.25, 0.20])
        r = MEM.identify_lewis_tvv(MEM.estimate_var(Y, 1); rng=Xoshiro(103))
        @test r.converged == true
        @test occursin("Lewis", sprint(show, r))
    end

    @testset "recovers B on SV fixture (n=3)" begin
        qs = Float64[]
        for s in (104, 204, 304)
            Y, _, Q0t, _ = generate_sv_var(; n=3, Tobs=40000, rng=Xoshiro(s),
                rhos=[0.98, 0.96, 0.94], sigmas=[0.22, 0.32, 0.38])
            m = MEM.estimate_var(Y, 1)
            r = MEM.identify_lewis_tvv(m; rng=Xoshiro(s + 1))
            push!(qs, MEM.q_distance(r.Q, Q0t))
            @test r.weak_id == false
        end
        @test maximum(qs) < 0.25
        @test sum(qs) / length(qs) < 0.2
    end

    @testset "determinism" begin
        Y, _, _, _ = generate_tvv_var(; Tobs=2000, rng=Xoshiro(106))
        m = MEM.estimate_var(Y, 1)
        r1 = MEM.identify_lewis_tvv(m; n_starts=3, rng=Xoshiro(107))
        r2 = MEM.identify_lewis_tvv(m; n_starts=3, rng=Xoshiro(107))
        @test r1.B0 == r2.B0
    end

    @testset "weighting modes" begin
        Y, _, _, _ = generate_tvv_var(; Tobs=2000, rng=Xoshiro(108))
        m = MEM.estimate_var(Y, 1)
        for w in (:one_step, :two_step, :cue)
            r = MEM.identify_lewis_tvv(m; weighting=w, n_starts=2, rng=Xoshiro(109))
            @test MEM.check_orthogonal(r.Q)
            @test isfinite(r.J)
        end
        @test_throws ArgumentError MEM.identify_lewis_tvv(m; weighting=:bogus)
    end

    @testset "weak-ID flag (both sides)" begin
        # Homoskedastic Gaussian DGP -> weak
        B0h = [1.0 0.3; 0.2 1.0]
        Yh, _ = simulate_svar(B0h, [0.4 * Matrix{Float64}(I, 2, 2)]; Tobs=2000, rng=Xoshiro(110))
        mh = MEM.estimate_var(Yh, 1)
        rh = MEM.identify_lewis_tvv(mh; n_starts=3, rng=Xoshiro(111))
        @test rh.weak_id == true
        @test occursin("weak identification", rh.message)
        # TVV DGP -> identified
        Yt, _, _, _ = generate_tvv_var(; Tobs=2000, rng=Xoshiro(112))
        mt = MEM.estimate_var(Yt, 1)
        rt = MEM.identify_lewis_tvv(mt; n_starts=3, rng=Xoshiro(113))
        @test rt.weak_id == false
    end

    @testset "orthogonality and sign normalization" begin
        Y, _, _, _ = generate_tvv_var(; Tobs=2000, rng=Xoshiro(114))
        m = MEM.estimate_var(Y, 1)
        r = MEM.identify_lewis_tvv(m; n_starts=3, rng=Xoshiro(115))
        @test MEM.check_orthogonal(r.Q)
        @test all(diag(r.B0) .> 0)
        @test r.B0 ≈ MEM.safe_cholesky(m.Sigma) * r.Q
    end

    @testset "input validation" begin
        Y, _, _, _ = generate_tvv_var(; Tobs=500, rng=Xoshiro(116))
        m = MEM.estimate_var(Y, 1)
        @test_throws ArgumentError MEM.identify_lewis_tvv(m; lags=Int[])
        @test_throws ArgumentError MEM.identify_lewis_tvv(m; lags=[1, -1])
        @test_throws ArgumentError MEM.identify_lewis_tvv(m; n_starts=0)
        Ytiny, _, _, _ = generate_tvv_var(; Tobs=50, rng=Xoshiro(117))
        mtiny = MEM.estimate_var(Ytiny, 1)
        @test_throws ArgumentError MEM.identify_lewis_tvv(mtiny)  # < 100 usable obs
    end

    @testset "registry and compute_Q (#826)" begin
        Y, _, _, _ = generate_tvv_var(; Tobs=1500, rng=Xoshiro(118))
        m = MEM.estimate_var(Y, 1)
        @test haskey(MEM.IDENTIFICATION_REGISTRY, :lewis_tvv)
        @test MEM._needs_residuals(:lewis_tvv)
        @test !MEM._is_set_identified(:lewis_tvv)
        @test !MEM._is_partial(:lewis_tvv)
        @test MEM._should_match_columns(:lewis_tvv)
        Q = MEM.compute_Q(m, :lewis_tvv; n_starts=3, rng=Xoshiro(119))
        @test size(Q) == (2, 2)
        @test norm(Q' * Q - I) < 1e-6
        Q2 = MEM.compute_Q(m, :lewis_tvv; n_starts=3, rng=Xoshiro(119))
        @test Q == Q2  # determinism under explicit Xoshiro
        S = MEM.compute_structural_shocks(m, Q)
        @test size(S) == (1499, 2)
        IR = MEM.compute_irf(m, Q, 8)
        @test size(IR) == (8, 2, 2)
        ir = irf(m, 4; method=:lewis_tvv, n_starts=3, rng=Xoshiro(119))
        @test size(ir.values) == (4, 2, 2)
        @test_throws MethodError MEM.compute_Q(m, :lewis_tvv; bogus_kw=1)
    end

end
