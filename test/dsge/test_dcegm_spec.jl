# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels

@testset "G-02: to_spec(::DCEGMProblem) (#639)" begin
    prob = dcegm_retirement_model(; n_a=40, n_periods=6)
    spec = to_spec(prob)

    @test spec isa ModelSpec
    @test spec.n_endog == 0
    @test spec.n_exog == 0
    @test isempty(spec.equations)
    @test isempty(spec.residual_fns)
    @test MacroEconometricModels.has_agents(spec)
    @test MacroEconometricModels.has_kind(spec, DCEGMSystem)
    @test keys(spec.agents) == (:household,)
    @test only(values(spec.agents)) isa DCEGMSystem
    @test only(values(spec.agents)).problem === prob

    spec_named = to_spec(prob; agent_name=:retirees)
    @test keys(spec_named.agents) == (:retirees,)
    @test spec_named.agents.retirees.problem === prob

    wrapped = only(values(spec.agents)).problem
    @test wrapped.options == [:retire, :work]
    @test wrapped.absorbing == [true, false]
    @test wrapped.asset_grid[1] == prob.asset_grid[1]
    @test wrapped.asset_grid[end] == prob.asset_grid[end]
    @test wrapped.n_periods == 6

    sol1 = dcegm_solve(prob)
    sol2 = dcegm_solve(wrapped)
    sol3 = solve(spec)
    @test sol1.n_periods == sol2.n_periods
    @test sol1.converged && sol2.converged
    for t in (1, 3, 6), d in 1:2, M in (5.0, 20.0, 40.0)
        c1, v1 = dcegm_policy(sol1, t, d, 1, M)
        c2, v2 = dcegm_policy(sol2, t, d, 1, M)
        @test c1 ≈ c2 atol=0 rtol=0
        @test v1 ≈ v2 atol=0 rtol=0
        c3, v3 = dcegm_policy(sol3, t, d, 1, M)
        @test c1 ≈ c3 atol=0 rtol=0
        @test v1 ≈ v3 atol=0 rtol=0
    end
    @test sol3 isa DCEGMSolution
end

@testset "G-10: DCEGM market clearing (#644)" begin
    firm = DCEGMFirm(; alpha=0.36, delta=0.08, Z=1.0, L=1.0)
    @test firm isa DCEGMFirm{Float64}
    @test occursin("DCEGMFirm", sprint(show, firm))

    # Cobb-Douglas identities: K^d falls in r, w = (1-α)Z (K/L)^α, r recovers.
    r0 = 0.05
    Kd = dcegm_capital_demand(firm, r0)
    w0 = dcegm_firm_wage(firm, r0)
    kl = Kd / firm.L
    @test w0 ≈ (1 - firm.alpha) * firm.Z * kl^firm.alpha
    @test r0 ≈ firm.alpha * firm.Z * kl^(firm.alpha - 1) - firm.delta atol=1e-12
    @test dcegm_capital_demand(firm, 0.10) < Kd
    @test dcegm_firm_wage(firm, 0.10) < w0
    @test !isfinite(dcegm_capital_demand(firm, -firm.delta))
    @test dcegm_firm_wage(firm, -firm.delta) == 0.0
    @test_throws ArgumentError DCEGMFirm(; alpha=1.5)
    @test_throws ArgumentError DCEGMFirm(; L=0.0)

    # Stationary OLG cross-section of the retirement model + CD firm.
    # Household wage stays at the PE default (20); only R = 1+r is the
    # equilibrium object. Newborns enter with zero assets (cash-on-hand = wage).
    prob = dcegm_retirement_model(; n_periods=6, n_a=30, a_max=40.0, beta=0.96,
                                    wage=20.0, pension=2.0, disutility=0.5)
    eq = dcegm_steady_state(prob, firm; r_bounds=(0.06, 0.14), tol=1e-3, max_iter=30)
    @test eq isa DCEGMEquilibrium
    @test eq.converged
    @test eq.solution.converged
    @test abs(eq.K - eq.K_demand) < 5e-3
    @test eq.K ≈ eq.K_demand atol=5e-3          # A = K^d(r)
    @test eq.excess_demand ≈ eq.K - eq.K_demand atol=0 rtol=0
    @test eq.r > 0.06 && eq.r < 0.14
    @test eq.solution.prob.R ≈ 1 + eq.r atol=1e-14
    @test eq.Y ≈ firm.Z * eq.K^firm.alpha * eq.L^(1 - firm.alpha)
    @test eq.L == firm.L                        # labor=:exogenous default
    @test 0.08 < eq.r < 0.12                    # calibration bracket from the sweep

    # report prints r and K (the acceptance numbers).
    txt = sprint(report, eq)
    @test occursin("Interest rate r", txt)
    @test occursin("Capital K", txt)
    @test occursin("DCEGM General Equilibrium", txt)
    @test occursin("r=", sprint(show, eq))
    @test occursin("K=", sprint(show, eq))

    # Same clearing from a ModelSpec and from a (R, w) factory.
    spec = to_spec(prob)
    eq_spec = dcegm_steady_state(spec, firm; r_bounds=(0.06, 0.14), tol=1e-3, max_iter=30)
    @test eq_spec.converged
    @test eq_spec.r ≈ eq.r atol=2e-3
    @test eq_spec.K ≈ eq.K atol=5e-3

    make = (R, w) -> dcegm_retirement_model(; n_periods=6, n_a=30, a_max=40.0,
                                              beta=0.96, R=R, wage=20.0,
                                              pension=2.0, disutility=0.5)
    eq_fn = dcegm_steady_state(make, firm; r_bounds=(0.06, 0.14), tol=1e-3, max_iter=30)
    @test eq_fn.converged
    @test abs(eq_fn.K - eq_fn.K_demand) < 5e-3

    # Measured labor uses the work-option share, not firm.L.
    eq_m = dcegm_steady_state(prob, DCEGMFirm(; alpha=0.36, delta=0.08, Z=1.0, L=1.0);
                              labor=:measured, r_bounds=(0.06, 0.14),
                              tol=2e-3, max_iter=30)
    @test eq_m.converged
    @test 0 < eq_m.L < 1                        # last-period retirement cuts the work share
    @test abs(eq_m.K - eq_m.K_demand) < 1e-2

    @test_throws ArgumentError dcegm_steady_state(prob, firm; r_bounds=(0.1, 0.05))
    @test_throws ArgumentError dcegm_steady_state(prob, firm; r_bounds=(-0.09, 0.05))
    @test_throws ArgumentError dcegm_steady_state(prob, firm; labor=:hours)
    @test_throws ArgumentError dcegm_steady_state(prob, firm; reprice_wage=true,
                                                  work_option=:farm)
end
