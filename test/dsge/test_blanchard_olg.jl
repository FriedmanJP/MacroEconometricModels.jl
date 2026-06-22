# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra

@testset "Blanchard OLG" begin

    @testset "Ramsey limit (γ=1)" begin
        m = BlanchardOLG(; gamma=1.0, beta=0.96, alpha=0.36, delta=0.08, Z=1.0)
        ss = blanchard_steady_state(m)
        @test ss.converged
        @test isapprox(ss.r, 1 / 0.96 - 1; atol=1e-4)            # r = 1/β − 1 (Ramsey)
        k_ramsey = (0.36 * 1.0 / (ss.r + 0.08))^(1 / (1 - 0.36))
        @test isapprox(ss.k, k_ramsey; rtol=1e-4)
        @test ss.C > 0 && ss.w > 0
        @test isapprox(ss.C, ss.r * ss.k + ss.w; rtol=1e-6)      # SS budget (b=0)
        @test isapprox(ss.mpc, 1 - 0.96; atol=1e-12)             # MPC = 1 − βγ
    end

    @testset "Finite horizon: r > 1/β − 1" begin
        m = BlanchardOLG(; gamma=0.98, beta=0.96)
        ss = blanchard_steady_state(m)
        @test ss.converged
        @test ss.r > 1 / 0.96 - 1                                # finite horizons raise r
        @test ss.k > 0 && ss.C > 0
    end

    @testset "Interest rate rises with mortality" begin
        rs = [blanchard_steady_state(BlanchardOLG(; gamma=g, beta=0.96)).r
              for g in (0.99, 0.97, 0.95, 0.93)]
        @test issorted(rs)                                        # more death → higher r
        @test all(r -> r > 1 / 0.96 - 1, rs)
    end

    @testset "Non-Ricardian debt" begin
        ss0 = blanchard_steady_state(BlanchardOLG(; gamma=0.98, beta=0.96, b=0.0))
        ss1 = blanchard_steady_state(BlanchardOLG(; gamma=0.98, beta=0.96, b=0.05))
        ss2 = blanchard_steady_state(BlanchardOLG(; gamma=0.98, beta=0.96, b=0.10))
        @test ss1.converged && ss2.converged
        @test ss1.r > ss0.r && ss2.r > ss1.r                     # debt raises r
        @test ss1.k < ss0.k && ss2.k < ss1.k                     # debt crowds out capital
    end

    @testset "Saddle-path dynamics" begin
        m = BlanchardOLG(; gamma=0.98, beta=0.96)
        ss = blanchard_steady_state(m)
        sol = blanchard_solve(m, ss)
        @test sol.determinate                                     # exactly one stable root
        @test abs(sol.stable_eig) < 1
        @test count(<(1.0 - 1e-9), abs.(sol.eigenvalues)) == 1
        # Transition from below the steady state converges monotonically toward k*.
        tr = blanchard_transition(m, sol, 0.8 * ss.k; H=80)
        @test abs(tr.k[end] - ss.k) < abs(tr.k[1] - ss.k)
        @test isapprox(tr.k[end], ss.k; atol=1e-2)
        @test all(diff(tr.k) .>= -1e-10)                          # monotone toward SS
        @test tr.r[1] > ss.r                                      # low k ⟹ high r initially
    end

    @testset "Display" begin
        ss = blanchard_steady_state(BlanchardOLG())
        @test ss isa BlanchardOLGSteadyState{Float64}
        io = IOBuffer(); show(io, ss)
        @test occursin("BlanchardOLGSteadyState", String(take!(io)))
        report(ss)                                                # smoke test
    end

end
