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

    @testset "report is io-routed (#254 G-17)" begin
        ss = blanchard_steady_state(BlanchardOLG(; gamma=0.98, beta=0.96))
        iob = IOBuffer(); report(iob, ss)
        @test occursin("Blanchard", String(take!(iob)))
        @test (redirect_stdout(devnull) do; report(ss); end; true)   # stdout convenience form
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

    @testset "Debt-service C correction (#237/T137)" begin
        # Aggregate consumption is C = r·k + w = f(k) − δk: debt is net wealth and
        # taxes r·b service it, so the debt-service terms cancel. The old code
        # subtracted a spurious −r·b, wrong for b ≠ 0 (a no-op only at b = 0).
        mb = BlanchardOLG(; gamma=0.98, beta=0.96, alpha=0.36, delta=0.08, Z=1.0, b=0.1)
        ss = blanchard_steady_state(mb)
        @test ss.converged
        @test isapprox(ss.C, ss.r * ss.k + ss.w; rtol=1e-8)          # NOT r·k+w−r·b
        @test isapprox(ss.C, mb.Z * ss.k^mb.alpha - mb.delta * ss.k; rtol=1e-8)  # f(k)−δk
        @test !isapprox(ss.C, ss.r * ss.k + ss.w - ss.r * mb.b; rtol=1e-6)  # ≠ old value
        # Human wealth H keeps −r·b (after-tax labour income): legitimate reduction.
        @test isapprox(ss.H, (ss.w - ss.r * mb.b) * (1 + ss.r) / (1 + ss.r - mb.gamma); rtol=1e-8)
        # M[1,1] must stay (1+r) — the issue's proposed −b·r'(k) term is WRONG.
        sol = blanchard_solve(mb, ss)
        @test isapprox(sol.M[1, 1], 1 + ss.r)
        @test count(<(1.0 - 1e-9), abs.(sol.eigenvalues)) == 1       # one stable root
    end

    @testset "λ and γ→1 Ramsey limit (#240/H-20)" begin
        beta = 0.96
        # Independent references (Blanchard 1985): the individual marginal
        # propensity to consume out of wealth is 1 − βγ, and as γ → 1 the model
        # collapses to the representative-agent Ramsey economy (the wedge λ → 0,
        # r* → 1/β − 1, k* → k_ramsey).
        for gamma in (0.90, 0.95, 0.99)
            m = BlanchardOLG(; gamma=gamma, beta=beta, alpha=0.36, delta=0.08, Z=1.0)
            ss = blanchard_steady_state(m)
            @test isapprox(ss.mpc, 1 - beta * gamma; atol=1e-12)   # MPC = 1 − βγ
            @test ss.r > 1 / beta - 1                               # finite horizon raises r
        end
        # γ → 1 continuity: k* → Ramsey k evaluated at r = 1/β − 1
        r_ram = 1 / beta - 1
        k_ram = (0.36 / (r_ram + 0.08))^(1 / (1 - 0.36))
        ks = [blanchard_steady_state(BlanchardOLG(; gamma=g, beta=beta, alpha=0.36, delta=0.08)).k
              for g in (0.98, 0.995, 0.999)]
        @test issorted(ks)                                          # k → k_ramsey as γ → 1
        @test abs(ks[end] - k_ram) < abs(ks[1] - k_ram)
        @test isapprox(blanchard_steady_state(BlanchardOLG(; gamma=1.0, beta=beta, alpha=0.36, delta=0.08)).k,
                       k_ram; rtol=1e-4)                            # γ = 1 IS Ramsey
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
