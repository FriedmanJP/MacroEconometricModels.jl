# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using SparseArrays

const _CT = MacroEconometricModels

@testset "Continuous-time Aiyagari (Achdou et al. 2022)" begin

    m = CTAiyagari(; I=200, a_max=30.0, sigma=2.0, rho=0.05, delta=0.05)

    @testset "HJB implicit upwind" begin
        r = 0.03
        kl = (0.36 / (r + 0.05))^(1 / 0.64)
        w = 0.64 * kl^0.36
        v, c, s, A, a, ok = _CT.ct_hjb(m, r, w)
        @test ok                                    # HJB converged
        @test all(c .> 0)                           # positive consumption
        @test size(v) == (200, 2) && size(c) == (200, 2)
        # The generator is a valid infinitesimal generator: rows sum to ~0.
        @test maximum(abs.(vec(sum(A; dims=2)))) < 1e-8
        # Saving drift is (weakly) decreasing in wealth (concave policy) at the top.
        @test s[end, 1] <= 1e-8 && s[end, 2] <= 1e-8   # no saving past a_max
    end

    @testset "KFE stationary distribution" begin
        r = 0.03
        kl = (0.36 / (r + 0.05))^(1 / 0.64)
        w = 0.64 * kl^0.36
        _, _, _, A, a, _ = _CT.ct_hjb(m, r, w)
        da = a[2] - a[1]
        g = _CT.ct_kfe(A, m.I, da)
        @test size(g) == (200, 2)
        @test minimum(g) >= -1e-10                  # nonnegative density
        @test isapprox(sum(g) * da, 1.0; atol=1e-6) # integrates to 1
    end

    @testset "Steady-state equilibrium" begin
        ss = ct_steady_state(m; tol=1e-5)
        @test ss isa CTSteadyState{Float64}
        @test ss.converged
        @test ss.K > 0
        @test 0 < ss.r < m.rho                      # r below the discount rate
        # Market clears: household capital ≈ firm capital demand at the equilibrium r.
        kl_eq = (0.36 / (ss.r + 0.05))^(1 / 0.64)
        @test isapprox(ss.K, kl_eq * ss.L; rtol=1e-2)
        # Fraction at the borrowing constraint is a sensible probability.
        da = ss.a[2] - ss.a[1]
        constrained = (ss.g[1, 1] + ss.g[1, 2]) * da
        @test 0 < constrained < 1
    end

    @testset "More risk raises precautionary saving" begin
        # Wider income spread (same mean) ⟹ lower equilibrium r (more saving).
        m_lo = CTAiyagari(; I=200, z=[0.13, 0.17], lambda=[0.5, 0.5])
        m_hi = CTAiyagari(; I=200, z=[0.05, 0.25], lambda=[0.5, 0.5])
        r_lo = ct_steady_state(m_lo; tol=1e-5).r
        r_hi = ct_steady_state(m_hi; tol=1e-5).r
        @test r_hi < r_lo                           # more risk ⟹ lower r
    end

    @testset "Display" begin
        ss = ct_steady_state(m; tol=1e-4)
        io = IOBuffer(); show(io, ss)
        @test occursin("CTSteadyState", String(take!(io)))
        # G-17 (#254): report is io-routed (report(io, obj)); stdout convenience form still works
        iob = IOBuffer(); report(iob, ss)
        @test occursin("Continuous-Time Aiyagari", String(take!(iob)))
        @test (redirect_stdout(devnull) do; report(ss); end; true)
    end

    @testset "MIT-shock transition" begin
        ss0 = ct_steady_state(m; tol=1e-6)
        N = 60
        # Zero shock ⟹ the transition path is flat at the steady state.
        tr0 = ct_mit_shock(m, ss0, fill(m.Z, N + 1); dt=0.5, max_iter=50, tol=1e-7, relax=0.5)
        @test tr0.converged
        @test length(tr0.t) == N + 1
        @test maximum(abs.(tr0.K .- ss0.K)) < 1e-3
        # Transitory positive TFP shock ⟹ capital accumulates then returns; r up on impact.
        Z_shock = [m.Z * (1 + 0.03 * exp(-0.4 * (n - 1) * 0.5)) for n in 1:(N+1)]
        tr = ct_mit_shock(m, ss0, Z_shock; dt=0.5, max_iter=400, tol=1e-6, relax=0.3)
        @test tr.converged
        @test isapprox(tr.K[1], ss0.K; atol=1e-4)          # K_0 pinned by initial dist
        @test maximum(tr.K) > ss0.K + 1e-4                 # capital accumulates (hump)
        @test isapprox(tr.K[end], ss0.K; rtol=3e-2)        # returns toward the steady state
        @test tr.r[1] > ss0.r                              # positive TFP ⟹ higher r on impact
        io = IOBuffer(); show(io, tr)
        @test occursin("CTTransition", String(take!(io)))
    end

    @testset "Two-asset KMV-style solver" begin
        m2 = CTTwoAsset(; Ib=30, Ia=30, r_a=0.05, r_b=0.02, chi=2.0, rho=0.08)
        s = ct_two_asset_solve(m2; tol=1e-6)
        @test s isa CTTwoAssetSolution{Float64}
        @test s.hjb_converged
        # Valid infinitesimal generator (rows sum to ~0).
        @test maximum(abs.(vec(sum(s.gen; dims=2)))) < 1e-8
        db = s.b[2] - s.b[1]; da = s.a[2] - s.a[1]
        @test isapprox(sum(s.g) * db * da, 1.0; atol=1e-6)   # joint density integrates to 1
        @test minimum(s.g) >= -1e-10
        @test all(s.c .> 0)
        @test s.A > 0 && s.B >= 0
        @test s.A / (s.A + s.B) > 0.3                        # illiquidity premium ⟹ illiquid wealth
        # A larger illiquidity premium raises the illiquid share.
        s2 = ct_two_asset_solve(CTTwoAsset(; Ib=30, Ia=30, r_a=0.07, r_b=0.02, chi=2.0, rho=0.08); tol=1e-6)
        @test s2.hjb_converged
        @test s2.A / (s2.A + s2.B) > s.A / (s.A + s.B)
        io = IOBuffer(); show(io, s)
        @test occursin("CTTwoAssetSolution", String(take!(io)))
        report(s)
    end

end
