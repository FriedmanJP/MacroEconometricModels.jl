# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using SparseArrays
using Logging

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
        # NOTE (#509): this calibration is deliberately the DIVERGENT one --
        # a_max = 20 sits above a* = 1/(chi*r_a) = 10, so the level-quadratic cost
        # cannot stop illiquid wealth from growing. It is kept because it exercises
        # the solver mechanics (generator, mass, positivity) on a dense interior,
        # but its aggregates are grid artifacts and are asserted as such below.
        # `check_stationarity=false` silences the warning the solver now emits.
        m2 = CTTwoAsset(; Ib=30, Ia=30, r_a=0.05, r_b=0.02, chi=2.0, rho=0.08)
        @test !ct_two_asset_stationarity(m2).ok
        s = ct_two_asset_solve(m2; tol=1e-6, check_stationarity=false)
        @test s isa CTTwoAssetSolution{Float64}
        @test s.hjb_converged
        # Valid infinitesimal generator (rows sum to ~0).
        @test maximum(abs.(vec(sum(s.gen; dims=2)))) < 1e-8
        # The joint density integrates to 1 under the TRAPEZOIDAL weights `bdelta`/`adelta`
        # (KMV's `adelta(1) = 0.5*dagrid(1)`), which are half-width at the grid edges and
        # are what the solver normalizes against. A flat `db*da` is not the right measure —
        # it over-counts the boundary rows, and on this calibration, where most mass sits at
        # b = 0, it reads 2.43 rather than 1.
        mass = sum(s.g[i, j, k] * s.bdelta[i] * s.adelta[j]
                   for i in eachindex(s.b), j in eachindex(s.a), k in 1:2)
        @test isapprox(mass, 1.0; atol=1e-6)
        @test length(s.bdelta) == length(s.b) && length(s.adelta) == length(s.a)
        @test isapprox(sum(s.bdelta), s.b[end] - s.b[1]; rtol=1e-12)
        @test isapprox(sum(s.adelta), s.a[end] - s.a[1]; rtol=1e-12)
        @test minimum(s.g) >= -1e-10
        @test all(s.c .> 0)
        @test s.A > 0 && s.B >= 0
        # The old assertion here was `s.A / (s.A + s.B) > 0.3`, which passes at 0.94
        # while 65% of the mass sits ON the illiquid ceiling -- it could not tell an
        # illiquidity premium from a divergence. State what this calibration actually
        # is instead (#509).
        @test s.A / (s.A + s.B) > 0.9
        @test ceiling_mass(s).illiquid > 0.5                 # measured 0.654: a grid artifact
        # A larger illiquidity premium raises the illiquid share -- and, on a divergent
        # calibration, also the share pinned to the ceiling. Both are recorded.
        s2 = ct_two_asset_solve(CTTwoAsset(; Ib=30, Ia=30, r_a=0.07, r_b=0.02, chi=2.0,
                                           rho=0.08); tol=1e-6, check_stationarity=false)
        @test s2.hjb_converged
        @test s2.A / (s2.A + s2.B) > s.A / (s.A + s.B)
        @test ceiling_mass(s2).illiquid > ceiling_mass(s).illiquid   # 0.981 vs 0.654
        io = IOBuffer(); show(io, s)
        @test occursin("CTTwoAssetSolution", String(take!(io)))
        report(s)
    end


@testset "CT two-asset: KMV kinked cost, GE and MIT (#358/T259)" begin
    M = MacroEconometricModels

    @testset "power-spaced grids and trapezoidal weights" begin
        # KMV `PowerSpacedGrid`: k = 1 is uniform, k → 0 is L-shaped.
        g1 = M._ct2_power_grid(Float64, 0.0, 10.0, 11, 1.0)
        @test g1 ≈ collect(range(0.0, 10.0; length=11))
        for k in (0.15, 0.35, 0.5)
            g = M._ct2_power_grid(Float64, 0.0, 10.0, 21, k)
            @test length(g) == 21
            @test g[1] == 0.0 && g[end] == 10.0
            @test all(diff(g) .> 0)
            # L-shaped ⇒ denser at the bottom than a uniform grid
            @test count(<=(2.0), g) > count(<=(2.0), g1)
            @test diff(g)[1] < diff(g)[end]
        end
        # smaller k ⇒ more concentrated
        @test count(<=(1.0), M._ct2_power_grid(Float64, 0.0, 10.0, 41, 0.15)) >
              count(<=(1.0), M._ct2_power_grid(Float64, 0.0, 10.0, 41, 0.5))
        @test_throws ArgumentError M._ct2_power_grid(Float64, 0.0, 1.0, 1, 0.5)
        @test M._ct2_power_grid(Float64, 0.0, 1.0, 2, 0.5) == [0.0, 1.0]

        # `_ct2_deltas`: the weights partition the domain exactly, KMV's half-width at the ends
        for k in (1.0, 0.35)
            g = M._ct2_power_grid(Float64, 0.0, 8.0, 17, k)
            dg, delta = M._ct2_deltas(g)
            @test length(dg) == 16 && length(delta) == 17
            @test sum(delta) ≈ g[end] - g[1] rtol = 1e-14
            @test delta[1] ≈ dg[1] / 2
            @test delta[end] ≈ dg[end] / 2
            @test all(delta .> 0)
            # integrating a linear function is exact under these weights
            @test sum(g .* delta) ≈ (g[end]^2 - g[1]^2) / 2 rtol = 1e-12
        end
    end

    @testset "KMV adjustment cost and its inverse" begin
        m = CTTwoAsset(; Ib=6, Ia=6, cost=:kinked, chi0=0.05, chi1=0.5, chi2=2.0, a_kink=1.0)
        a_eff = 3.0
        # the FOC inverse returns EXACTLY zero inside the inaction band |R-1| <= chi0
        # Strictly inside the band the deposit is EXACTLY zero. (Right at the edge,
        # `R - 1` is not exactly ±chi0 in binary — `1.0 + (-0.05)` rounds — so the FOC
        # returns ~1e-8 rather than 0. Test the interior and the crossing separately.)
        for x in (-0.049, -0.03, 0.0, 0.03, 0.049)
            @test M._ct2_deposit(m, 1.0 + x, a_eff) == 0.0
        end
        @test abs(M._ct2_deposit(m, 1.0 + 0.05, a_eff)) < 1e-7      # at the edge: ~0
        @test M._ct2_deposit(m, 1.0 + 0.06, a_eff) > 0              # outside: adjusts
        @test M._ct2_deposit(m, 1.0 - 0.06, a_eff) < 0
        # the band has positive width, and widening chi0 widens it
        mw = CTTwoAsset(; Ib=6, Ia=6, cost=:kinked, chi0=0.20, chi1=0.5, chi2=2.0, a_kink=1.0)
        @test M._ct2_deposit(mw, 1.0 + 0.10, a_eff) == 0.0          # inside the wider band
        @test M._ct2_deposit(m, 1.0 + 0.10, a_eff) > 0              # outside the narrow one
        # cost is zero at d=0, positive and convex otherwise, and symmetric here
        @test M._ct2_adj_cost(m, 0.0, a_eff) == 0.0
        @test M._ct2_adj_cost(m, 0.4, a_eff) > 0
        @test M._ct2_adj_cost(m, 0.4, a_eff) ≈ M._ct2_adj_cost(m, -0.4, a_eff)
        @test M._ct2_adj_cost(m, 0.8, a_eff) > 2 * M._ct2_adj_cost(m, 0.4, a_eff)  # convex
        # deposit and marginal cost invert each other: chi'(d(y)) == y outside the band
        for y in (0.2, 0.5, 1.0)
            d = M._ct2_deposit(m, 1.0 + y, a_eff)
            marg = m.chi0 + ((d / a_eff) / m.chi1)^m.chi2      # KMV `adjcostfn1`
            @test marg ≈ y rtol = 1e-10
        end
        # `a_kink` is a FLOOR on the scale (KMV `max(kappa3, la)`), not an offset
        @test M._ct2_adj_scale(m, 0.0) == m.a_kink
        @test M._ct2_adj_scale(m, 5.0) == 5.0
        # the deposit scales with the illiquid stock — the property that lets a withdrawal
        # offset the accruing return r_a*a at any level of a
        d1 = M._ct2_deposit(m, 1.5, 2.0); d2 = M._ct2_deposit(m, 1.5, 8.0)
        @test d2 ≈ 4 * d1 rtol = 1e-12
        # the level-quadratic cost has NO inaction region and NO scaling with a
        mq = CTTwoAsset(; Ib=6, Ia=6, cost=:quadratic, chi=2.0)
        @test M._ct2_deposit(mq, 1.0 + 1e-9, 3.0) > 0        # adjusts for any R != 1
        @test M._ct2_deposit(mq, 1.0, 3.0) == 0.0
        @test M._ct2_deposit(mq, 1.5, 2.0) == M._ct2_deposit(mq, 1.5, 8.0)
        @test M._ct2_adj_cost(mq, 0.5, 2.0) ≈ (2.0 / 2) * 0.25
        # dmax cap (KMV `Parameters.f90`)
        mc = CTTwoAsset(; Ib=6, Ia=6, cost=:kinked, chi2=0.40176, dmax=0.75)
        @test abs(M._ct2_deposit(mc, 50.0, 10.0)) <= 0.75
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError CTTwoAsset(; cost=:bogus)
        @test_throws ArgumentError CTTwoAsset(; cost=:kinked, chi1=0.0)
        @test_throws ArgumentError CTTwoAsset(; cost=:kinked, chi2=0.0)
        @test_throws ArgumentError CTTwoAsset(; cost=:kinked, chi0=-0.1)
        @test_throws ArgumentError CTTwoAsset(; cost=:kinked, a_kink=0.0)
        @test_throws ArgumentError CTTwoAsset(; a_power=0.0)
        @test_throws ArgumentError CTTwoAsset(; a_power=1.5)
        @test_throws ArgumentError CTTwoAsset(; b_power=-0.2)
        @test_throws AssertionError CTTwoAsset(; r_a=0.01, r_b=0.02)   # premium must be > 0
    end

    @testset "stationarity diagnostic" begin
        # quadratic: withdrawals cap at the CONSTANT 1/chi, so illiquid wealth diverges
        # above a* = 1/(chi*r_a) — see #509
        st = ct_two_asset_stationarity(CTTwoAsset(; chi=2.0, r_a=0.05, a_max=20.0))
        @test st.bound ≈ 1 / (2.0 * 0.05)
        @test !st.ok                                    # a_max = 20 > a* = 10
        @test occursin("509", st.message)
        @test ct_two_asset_stationarity(CTTwoAsset(; chi=2.0, r_a=0.05, a_max=5.0)).ok
        @test st.a_star ≈ st.bound

        # kinked: the withdrawal SCALES with a, so the condition is on the withdrawal
        # RATE. In the KMV parameterization the implemented FOC is
        #   |d| = chi1 * (|V_a/V_b - 1| - chi0)^(1/chi2) * (a + a_kink),
        # so chi1 MULTIPLIES the withdrawal — a larger chi1 is MORE stationary, not less.
        # The bound is the maximum rate chi1*(1-chi0)^(1/chi2), which must exceed r_a.
        rate(c1, c0, c2) = c1 * (1 - c0)^(1 / c2)
        stk = ct_two_asset_stationarity(CTTwoAsset(; cost=:kinked, chi0=0.05, chi1=30.0,
                                                   r_a=0.05))
        @test stk.bound ≈ rate(30.0, 0.05, 0.40176)
        @test stk.bound > 26                          # ≈ 26.40
        @test stk.ok                                  # ... comfortably above r_a = 0.05
        @test isinf(stk.a_star)

        # A SMALL chi1 is the divergent case: the rate falls below r_a.
        stk_bad = ct_two_asset_stationarity(CTTwoAsset(; cost=:kinked, chi0=0.05, chi1=0.01,
                                                       r_a=0.05))
        @test stk_bad.bound ≈ rate(0.01, 0.05, 0.40176)
        @test stk_bad.bound < 0.05
        @test !stk_bad.ok
        @test stk_bad.a_star == 0.0
        @test occursin("RAISE chi1", stk_bad.message)

        # The absolute dmax cap re-imposes a constant withdrawal above a* = dmax/r_a even
        # when the rate condition holds.
        stk_dmax = ct_two_asset_stationarity(CTTwoAsset(; cost=:kinked, chi0=0.05, chi1=5.0,
                                                        r_a=0.05, dmax=0.1, a_max=20.0))
        @test !stk_dmax.ok
        @test stk_dmax.a_star ≈ 0.1 / 0.05
        @test occursin("dmax", stk_dmax.message)
    end

    @testset "solution diagnostics on a converged solve" begin
        m = CTTwoAsset(; Ib=25, Ia=25, r_a=0.05, r_b=0.02, chi=2.0, rho=0.10,
                       a_max=6.0, b_max=6.0, z=[0.6, 1.4], lambda=[0.4, 0.4])
        s = ct_two_asset_solve(m; tol=1e-6, max_iter=300)
        @test s.hjb_converged
        @test s.hjb_iterations > 0
        @test isfinite(s.kfe_residual) && s.kfe_residual < 1e-6
        # mass integrates to 1 under the trapezoidal weights, and the generator is valid
        mass = sum(s.g[i, j, k] * s.bdelta[i] * s.adelta[j]
                   for i in eachindex(s.b), j in eachindex(s.a), k in 1:2)
        @test mass ≈ 1.0 atol = 1e-8
        @test maximum(abs, vec(sum(s.gen; dims=2))) < 1e-8
        @test all(s.c .> 0)
        # aggregates equal the weighted integrals of the policy
        Achk = sum(s.a[j] * s.g[i, j, k] * s.bdelta[i] * s.adelta[j]
                   for i in eachindex(s.b), j in eachindex(s.a), k in 1:2)
        Bchk = sum(s.b[i] * s.g[i, j, k] * s.bdelta[i] * s.adelta[j]
                   for i in eachindex(s.b), j in eachindex(s.a), k in 1:2)
        @test s.A ≈ Achk rtol = 1e-12
        @test s.B ≈ Bchk rtol = 1e-12
        # hand-to-mouth shares partition the low-liquid mass
        htm = hand_to_mouth(s)
        @test htm.total ≈ htm.poor + htm.wealthy
        @test 0 <= htm.total <= 1
        @test htm.b_threshold > 0 && htm.a_threshold > 0
        htm2 = hand_to_mouth(s; b_threshold=1e-8, a_threshold=1e-8)
        @test htm2.total <= htm.total          # a tighter threshold cannot include more
        cm = ceiling_mass(s)
        @test 0 <= cm.liquid <= 1 && 0 <= cm.illiquid <= 1
        # warm start reproduces the same fixed point
        # Warm-starting lands on the same fixed point, but the tolerance has to be tied to
        # the SOLVER's, not to the identity: `tol` bounds ‖ΔV‖∞, and the pushforward to the
        # stationary distribution and then to an aggregate amplifies it (measured 1.8e-5
        # relative at tol = 1e-6).
        s_warm = ct_two_asset_solve(m; tol=1e-6, max_iter=300, V_init=s.V)
        @test s_warm.hjb_converged
        @test s_warm.A ≈ s.A rtol = 1e-3
        @test s_warm.B ≈ s.B rtol = 1e-3
        @test s_warm.hjb_iterations <= s.hjb_iterations
        @test_throws ArgumentError ct_two_asset_solve(m; V_init=zeros(3, 3, 2))
    end

    @testset "general equilibrium clears both markets" begin
        m = CTTwoAsset(; Ib=25, Ia=25, a_max=6.0, b_max=5.0, rho=0.06, sigma=2.0,
                       z=[0.6, 1.4], lambda=[0.4, 0.4], alpha=0.36, delta=0.05, Z=1.0,
                       B_supply=0.69, chi=2.0, a_power=0.5, b_power=0.5)
        ge = ct_two_asset_ge(m; max_iter=120, tol=1e-3, relax_K=0.3, relax_rb=0.05)
        @test ge isa MacroEconometricModels.CTTwoAssetGE{Float64}
        @test ge.markets_cleared
        @test abs(ge.resid_illiquid) < 1e-3          # measured 8.2e-04
        @test abs(ge.resid_liquid) < 1e-3            # measured -2.3e-04

        # the firm's first-order conditions hold EXACTLY at the reported K
        @test ge.r_a ≈ m.alpha * m.Z * (ge.K / ge.L)^(m.alpha - 1) - m.delta rtol = 1e-14
        @test ge.w ≈ (1 - m.alpha) * m.Z * (ge.K / ge.L)^m.alpha rtol = 1e-14
        @test ge.Y ≈ m.Z * ge.K^m.alpha * ge.L^(1 - m.alpha) rtol = 1e-14
        # the government budget balances and labor is the stationary mean of the income process
        @test ge.tau ≈ ge.r_b * m.B_supply rtol = 1e-14
        la = m.income.lambda; zz = m.income.z
        @test ge.L ≈ zz[1] * la[2] / (la[1] + la[2]) + zz[2] * la[1] / (la[1] + la[2]) rtol = 1e-14
        # the illiquidity premium is positive and the liquid return respects the bounds
        @test ge.r_b < ge.r_a
        @test ge.r_b <= m.rho
        @test ge.B ≈ ge.solution.B rtol = 1e-14
        io = IOBuffer(); show(io, ge)
        @test occursin("CTTwoAssetGE", String(take!(io)))
        report(ge)
    end

    @testset "MIT transition returns to the terminal steady state" begin
        m = CTTwoAsset(; Ib=25, Ia=25, a_max=6.0, b_max=5.0, rho=0.06, sigma=2.0,
                       z=[0.6, 1.4], lambda=[0.4, 0.4], alpha=0.36, delta=0.05, Z=1.0,
                       B_supply=0.69, chi=2.0, a_power=0.5, b_power=0.5)
        ge = ct_two_asset_ge(m; max_iter=120, tol=1e-3, relax_K=0.3, relax_rb=0.05)

        N = 40
        rho_z = 0.6
        Z = [1.0 + 0.02 * rho_z^(n - 1) for n in 1:(N + 1)]; Z[1] = 1.0
        tr = ct_two_asset_mit(m, ge, Z; dt=0.5, max_iter=80, tol=1e-5,
                              relax_K=0.3, relax_rb=0.02)
        @test tr isa MacroEconometricModels.CTTwoAssetTransition{Float64}
        @test length(tr.t) == N + 1
        @test tr.t[1] == 0.0 && tr.t[2] ≈ 0.5
        @test tr.Z == Z

        # K_0 is PINNED by the predetermined distribution and cannot jump on impact
        @test tr.K[1] ≈ ge.K rtol = 1e-12
        # the path returns to the terminal steady state (measured 3.7e-08)
        @test abs(tr.K[end] - ge.K) < 1e-5
        # capital accumulates above the steady state, then comes back down
        ip = argmax(tr.K)
        @test tr.K[ip] > ge.K
        @test 1 < ip < N + 1
        @test tr.K[end] < tr.K[ip]
        # correct impact signs: higher TFP raises the marginal product of capital and the wage
        @test tr.r_a[2] > ge.r_a
        @test tr.w[2] > ge.w
        # prices are consistent with the firm FOCs along the whole path
        for n in 1:(N + 1)
            @test tr.r_a[n] ≈ m.alpha * Z[n] * (tr.K[n] / ge.L)^(m.alpha - 1) - m.delta rtol = 1e-12
            @test tr.w[n] ≈ (1 - m.alpha) * Z[n] * (tr.K[n] / ge.L)^m.alpha rtol = 1e-12
        end
        @test all(isfinite, tr.C) && all(tr.C .> 0)
        @test all(isfinite, tr.B)
        @test all(tr.r_b .< tr.r_a)
        io = IOBuffer(); show(io, tr)
        @test occursin("CTTwoAssetTransition", String(take!(io)))

        # a ZERO shock leaves the economy at rest: the flat path is as flat as the initial
        # steady state is accurate (its own clearing residual is ~1e-3).
        tr0 = ct_two_asset_mit(m, ge, fill(m.Z, 17); dt=0.5, max_iter=60, tol=1e-6,
                               relax_K=0.3, relax_rb=0.02)
        @test tr0.K[1] ≈ ge.K rtol = 1e-12
        @test maximum(abs.(tr0.K .- ge.K)) < 1e-2
        @test maximum(abs.(tr0.r_a .- ge.r_a)) < 1e-3

        @test_throws ArgumentError ct_two_asset_mit(m, ge, [1.0])
    end

    @testset "kinked cost produces an inaction region the quadratic cannot" begin
        # This is a property of the FOC, so it holds independently of whether the aggregate
        # solve converges: count the grid cells the deposit policy leaves exactly at zero.
        function inaction_cells(; cost, chi0=0.05, kw...)
            m = CTTwoAsset(; Ib=20, Ia=20, r_a=0.05, r_b=0.01, rho=0.07, a_max=10.0,
                           b_max=5.0, z=[0.5, 1.5], lambda=[0.3, 0.3],
                           cost=cost, chi0=chi0, chi2=2.0, kw...)
            # This testset is about the FOC's inaction band, not about stationarity;
            # a_max = 10 sits exactly at a* for the quadratic leg, so silence #509's
            # warning rather than let it fire on every call.
            s = ct_two_asset_solve(m; tol=1e-6, max_iter=60, check_stationarity=false)
            return count(iszero, s.d) / length(s.d)
        end
        # the smooth level-quadratic cost adjusts almost everywhere
        frac_q = inaction_cells(; cost=:quadratic)
        # the kink opens a band, and widening chi0 widens it
        f1 = inaction_cells(; cost=:kinked, chi0=0.02, chi1=1.0)
        f2 = inaction_cells(; cost=:kinked, chi0=0.20, chi1=1.0)
        @test f2 > f1
        @test f2 > frac_q
    end
end
end

@testset "#509: level-quadratic illiquid divergence is detected, not shipped silently" begin
    # The level-quadratic cost chi(d) = (chi/2)d^2 has FOC d = (V_a/V_b - 1)/chi, and
    # V_a/V_b >= 0, so the largest WITHDRAWAL is the constant 1/chi. The no-deposit
    # illiquid drift is r_a*a, which grows without bound in a. A constant cap cannot
    # offset a return that scales with a, so illiquid wealth necessarily diverges above
    #     a* = 1/(chi*r_a).
    mk(amax) = CTTwoAsset(; Ib=20, Ia=20, r_a=0.05, r_b=0.02, chi=8.0, rho=0.08,
                          a_max=amax, b_max=5.0)
    a_star = 1 / (8.0 * 0.05)                                  # = 2.5
    @test ct_two_asset_stationarity(mk(2.0)).bound ≈ a_star

    @testset "the solver warns instead of returning a converged artifact" begin
        @test_logs (:warn, r"diverges") match_mode = :any begin
            ct_two_asset_solve(mk(6.0); tol=1e-6, max_iter=50)
        end
        # ... and the warning is suppressible for tests that want the artifact.
        @test_logs min_level = Logging.Warn begin
            ct_two_asset_solve(mk(2.0); tol=1e-6, max_iter=50, check_stationarity=false)
        end
    end

    @testset "divergence signature: raising a_max moves the pile, it does not reveal a tail" begin
        # If the distribution were bounded and merely truncated, raising a_max would
        # expose the tail and A/a_max would FALL. Under divergence it RISES toward 1
        # and the excess mass sits exactly on the ceiling.
        s3 = ct_two_asset_solve(mk(3.0); tol=1e-6, max_iter=300, check_stationarity=false)
        s6 = ct_two_asset_solve(mk(6.0); tol=1e-6, max_iter=300, check_stationarity=false)
        @test s3.hjb_converged && s6.hjb_converged
        @test !ct_two_asset_stationarity(mk(3.0)).ok
        @test !ct_two_asset_stationarity(mk(6.0)).ok

        @test s6.A / 6.0 > s3.A / 3.0                 # measured 0.642 vs 0.234
        @test s6.A > 2 * s3.A                         # 3.853 vs 0.701
        # The ceiling mass IS the excess: A/a_max and the ceiling mass coincide.
        @test ceiling_mass(s3).illiquid ≈ s3.A / 3.0 rtol = 0.05
        @test ceiling_mass(s6).illiquid ≈ s6.A / 6.0 rtol = 0.05
        @test ceiling_mass(s6).illiquid > 0.6
    end

    @testset "below a* the ceiling is clean and A is insensitive to a_max" begin
        # The other half of the dichotomy, and the answer to "is :quadratic usable":
        # inside a* the ceiling mass is machine-zero and A does not move with a_max --
        # because A is essentially ZERO. The level-quadratic cost is stationary only
        # where it supports no illiquid holdings at all, which is why a calibration
        # with a realistic illiquid tail needs cost=:kinked.
        s1 = ct_two_asset_solve(mk(1.0); tol=1e-6, max_iter=300, check_stationarity=false)
        s2 = ct_two_asset_solve(mk(2.0); tol=1e-6, max_iter=300, check_stationarity=false)
        @test ct_two_asset_stationarity(mk(1.0)).ok
        @test ct_two_asset_stationarity(mk(2.0)).ok
        @test ceiling_mass(s1).illiquid < 1e-15       # measured 6.7e-26
        @test ceiling_mass(s2).illiquid < 1e-15       # measured 1.2e-22
        @test abs(s2.A - s1.A) < 1e-6                 # both ~0: insensitive to a_max
        @test s2.A < 1e-6
    end
end

@testset "G-04: to_spec ContinuousHouseholdSystem (#641)" begin
    @testset "CTAiyagari" begin
        m = CTAiyagari(; I=50)
        spec = to_spec(m)
        @test spec isa ModelSpec
        @test spec.n_endog == 0 && spec.n_exog == 0
        @test isempty(spec.equations) && isempty(spec.residual_fns)
        @test spec.ir.clock === :continuous
        @test only(keys(spec.agents)) === :household
        hh = only(values(spec.agents))
        @test hh isa ContinuousHouseholdSystem
        @test hh.model === m
        @test MacroEconometricModels.has_kind(spec, ContinuousHouseholdSystem)

        spec_named = to_spec(m; agent_name=:hh)
        @test only(keys(spec_named.agents)) === :hh
        @test only(values(spec_named.agents)).model === m

        ss = ct_steady_state(hh.model; tol=1e-5)
        ss_direct = ct_steady_state(m; tol=1e-5)
        @test ss.r ≈ ss_direct.r
        @test ss.K ≈ ss_direct.K
        @test ss.w ≈ ss_direct.w
        ss_solve = solve(spec; tol=1e-5)
        @test ss_solve isa CTSteadyState
        @test ss_solve.r ≈ ss_direct.r
        @test ss_solve.K ≈ ss_direct.K
    end

    @testset "CTTwoAsset" begin
        # Coarse grid; same household payload as the family constructor so
        # `ct_two_asset_ge` on the wrapped model matches the direct call (G-05).
        m = CTTwoAsset(; Ib=20, Ia=20, a_max=6.0, b_max=5.0, rho=0.06, sigma=2.0,
                       z=[0.6, 1.4], lambda=[0.4, 0.4], alpha=0.36, delta=0.05,
                       Z=1.0, B_supply=0.69, chi=2.0, a_power=0.5, b_power=0.5)
        spec = to_spec(m)
        @test spec isa ModelSpec
        @test spec.n_endog == 0 && spec.n_exog == 0
        @test isempty(spec.equations)
        @test spec.ir.clock === :continuous
        hh = only(values(spec.agents))
        @test hh isa ContinuousHouseholdSystem
        @test hh.model === m
        @test hh.model.Ib == m.Ib && hh.model.Ia == m.Ia
        @test hh.model.a_max == m.a_max && hh.model.b_max == m.b_max
        @test hh.model.a_power == m.a_power && hh.model.b_power == m.b_power

        # One GE pass on the wrapped payload: grid nodes and K are those of
        # `ct_two_asset_ge(m)` (same object). Identity wrap ⇒ no second solve.
        ge = ct_two_asset_ge(hh.model; max_iter=120, tol=1e-3, relax_K=0.3, relax_rb=0.05)
        @test length(ge.solution.a) == m.Ia && length(ge.solution.b) == m.Ib
        @test ge.solution.a[end] ≈ m.a_max && ge.solution.b[end] ≈ m.b_max
        @test isapprox(ge.K, ge.solution.A; atol=1e-3)
        @test ge.K > 0
        @test solve(spec; max_iter=1, tol=1.0) isa CTTwoAssetGE
    end
end

@testset "G-14: CT irf wraps MIT (#648)" begin
    m = CTAiyagari(; I=50)
    ss = ct_steady_state(m; tol=1e-5)
    resp = irf(m, 16; ss=ss, shock_size=0.02, persist=0.6, dt=0.5, max_iter=80, tol=1e-5)
    @test resp isa ImpulseResponse
    @test resp.variables == ["K", "r", "w", "C", "Z"]
    @test resp.shocks == ["Z"]
    @test size(resp.values) == (16, 5, 1)
    @test all(isfinite, resp.values)
    @test resp.values[1, 1, 1] ≈ 0 atol=1e-4          # K_0 pinned
    @test resp.values[1, 5, 1] ≈ 0.02 * m.Z atol=1e-10
    fv = MacroEconometricModels._fevd_from_irf(resp)
    @test fv isa FEVD
    @test all(isfinite, fv.proportions)
end
