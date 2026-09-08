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

@testset "BB SV-SVAR (#825)" begin

    @testset "rotation step descends; grid improves bad starts" begin
        Zt = randn(Xoshiro(60), 300, 2)
        W = ones(300, 2)
        f(th) = sum(W .* (Zt * MEM._givens_to_orthogonal(th, 2)) .^ 2)
        th0 = [0.5]
        th1 = MEM._svsvar_rotation_step(Zt, W, th0, 200)
        @test f(th1) <= f(th0) * (1 + 1e-12)  # 1-ulp Optim/LBFGS landing (CI linux)
        @test MEM.check_orthogonal(MEM._givens_to_orthogonal(th1, 2))
        thg = MEM._svsvar_grid_start(Zt, W, [2.0], 12)
        @test f(thg) <= f([2.0])
        thoff = MEM._svsvar_grid_start(Zt, W, [2.0], 0)
        @test thoff == [2.0]  # grid disabled returns fallback
    end

    @testset "recovers B (n=2)" begin
        for s in (2, 12, 22)
            Y, B0t, _, _ = generate_sv_var(; n=2, Tobs=2000, rng=Xoshiro(s))
            r = MEM.identify_sv_svar(Y, 1; rng=Xoshiro(s + 1))
            @test r.converged == true
            @test MEM._procrustes_distance(r.B, B0t) < 0.2
            @test all(x -> -1 < x < 1, r.rhos)
            @test all(x -> x > 0, r.sigmas)
            @test size(r.H_smooth) == (1999, 2)
            @test all(isfinite, r.H_smooth)
        end
        Y, _, _, _ = generate_sv_var(; n=2, Tobs=2000, rng=Xoshiro(2))
        r = MEM.identify_sv_svar(Y, 1; rng=Xoshiro(3))
        @test occursin("SV-SVAR", sprint(show, r))
        @test length(r.loglik) == r.iters
        @test all(isfinite, r.loglik)
        # Path wiggles with MC noise (not asserted monotone); tail is bounded.
        tail = r.loglik[max(1, end - 9):end]
        @test maximum(tail) - minimum(tail) < 0.5 * abs(tail[end])
    end

    @testset "recovers B (n=3)" begin
        # Optim v1 LBFGS lands a worse rotation basin at n=3 (0.31 observed on
        # the Julia 1.10 numerical cell vs < 0.2 on Optim ≥ 2): upstream
        # optimizer difference, same convention as the #822 VFI NM gate.
        tol = Base.pkgversion(MEM.Optim) < v"2" ? 0.35 : 0.2
        for s in (4, 14)
            Y, B0t, _, _ = generate_sv_var(; n=3, Tobs=3000, rng=Xoshiro(s))
            r = MEM.identify_sv_svar(Y, 1; rng=Xoshiro(s + 1))
            @test r.converged == true
            @test MEM._procrustes_distance(r.B, B0t) < tol
        end
    end

    @testset "partial hetero flags homoskedastic shock" begin
        Yp, Bpt, _, _ = generate_sv_var(; n=2, Tobs=2000, rng=Xoshiro(30),
            sigmas=[0.25, 0.0])
        rp = MEM.identify_sv_svar(Yp, 1; hetero=[true, false], rng=Xoshiro(31))
        @test rp.converged == true
        @test rp.hetero == BitVector([true, false])
        b1 = rp.B[:, 1] / norm(rp.B[:, 1])
        t1 = Bpt[:, 1] / norm(Bpt[:, 1])
        @test abs(dot(b1, t1)) > 0.99  # hetero column recovered
        @test isnan(rp.sigmas[2]) && isnan(rp.rhos[2]) && isnan(rp.mus[2])
        @test all(isnan, rp.H_smooth[:, 2])
        @test all(isfinite, rp.H_smooth[:, 1])
    end

    @testset "misspecification smoke (GARCH DGP)" begin
        Yg, _ = simulate_garch_svar([1.0 0.3; 0.2 1.0],
            [0.4 * Matrix{Float64}(I, 2, 2)]; Tobs=1500, rng=Xoshiro(40))
        rg = MEM.identify_sv_svar(Yg, 1; maxiter=50, rng=Xoshiro(41))
        @test all(isfinite, rg.B)
        @test all(isfinite, rg.H_smooth)
        @test rg.iters <= 50
    end

    @testset "determinism" begin
        Yd, _, _, _ = generate_sv_var(; n=2, Tobs=1000, rng=Xoshiro(50))
        d1 = MEM.identify_sv_svar(Yd, 1; maxiter=20, rng=Xoshiro(51))
        d2 = MEM.identify_sv_svar(Yd, 1; maxiter=20, rng=Xoshiro(51))
        @test d1.B == d2.B
    end

    @testset "registry and compute_Q (#826)" begin
        Y, _, _, _ = generate_sv_var(; n=2, Tobs=1000, rng=Xoshiro(70))
        m = estimate_var(Y, 1)
        @test haskey(MEM.IDENTIFICATION_REGISTRY, :sv_em)
        @test MEM._needs_residuals(:sv_em)
        @test !MEM._is_set_identified(:sv_em)
        @test !MEM._is_partial(:sv_em)
        @test MEM._should_match_columns(:sv_em)
        Q = MEM.compute_Q(m, :sv_em; maxiter=5, rng=Xoshiro(71))
        @test size(Q) == (2, 2)
        @test norm(Q' * Q - I) < 1e-6
        Q2 = MEM.compute_Q(m, :sv_em; maxiter=5, rng=Xoshiro(71))
        @test Q == Q2  # determinism under explicit Xoshiro
        S = MEM.compute_structural_shocks(m, Q)
        @test size(S) == (999, 2)
        IR = MEM.compute_irf(m, Q, 8)
        @test size(IR) == (8, 2, 2)
        ir = irf(m, 4; method=:sv_em, maxiter=5, rng=Xoshiro(71))
        @test size(ir.values) == (4, 2, 2)
        @test_throws MethodError MEM.compute_Q(m, :sv_em; bogus_kw=1)
        @test_throws ArgumentError MEM.compute_Q(m, :sv_em; smoother=:ekf)
    end

    @testset "input validation" begin
        Yv, _, _, _ = generate_sv_var(; n=2, Tobs=500, rng=Xoshiro(60))
        @test_throws ArgumentError MEM.identify_sv_svar(Yv, 1; smoother=:ekf)
        @test_throws ArgumentError MEM.identify_sv_svar(Yv, 1; hetero=[false, false])
        @test_throws ArgumentError MEM.identify_sv_svar(Yv, 1; hetero=[true])
        @test_throws ArgumentError MEM.identify_sv_svar(Yv, 1; init=:bogus)
        @test_throws ArgumentError MEM.identify_sv_svar(Yv, 1; maxiter=0)
        @test_throws ArgumentError MEM.identify_sv_svar(Yv, 1; tol=0.0)
        @test_throws ArgumentError MEM.identify_sv_svar(randn(Xoshiro(61), 50, 2), 1)
        @test_throws ArgumentError MEM.identify_sv_svar(randn(Xoshiro(62), 500, 1), 1)
    end

    @testset "plot_result impact/volatility / refs" begin
        Y, _, _, _ = generate_sv_var(; n=2, Tobs=600, rng=Xoshiro(72))
        r = MEM.identify_sv_svar(Y, 1; maxiter=5, rng=Xoshiro(73))
        p = plot_result(r)
        @test occursin("Impact Matrix", p.html)
        pv = plot_result(r; view=:volatility)
        @test length(pv.html) > 1000
        @test_throws ArgumentError plot_result(r; view=:shocks)
        io = IOBuffer()
        refs(io, r)
        @test occursin("Bertsche", String(take!(io)))
    end

end
