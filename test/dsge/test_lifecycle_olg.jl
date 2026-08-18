# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# [T255]/#354 — true life-cycle OLG with age-dependent EGM.

using Test
using MacroEconometricModels
using LinearAlgebra

const _MFLC = MacroEconometricModels

# Degenerate single-state income process: the deterministic life-cycle model, which
# has a closed-form solution and therefore an exact oracle for the backward sweep.
_lc_det_income() = _MFLC.IncomeProcess{Float64}(ones(1, 1), [1.0], [1.0], :income)

"Simulate the age path of consumption and assets from `a_1 = 0` for a one-state model."
function _lc_age_path(m, c_pol, a_pol)
    a_grid = m.grid.grids[1]
    cs = Float64[]; as = Float64[0.0]
    av = 0.0
    for j in 1:m.J
        push!(cs, _MFLC._lc_interp(a_grid, c_pol[:, 1, j], av))
        av = _MFLC._lc_interp(a_grid, a_pol[:, 1, j], av)
        push!(as, av)
    end
    return cs, as
end

@testset "Life-Cycle OLG" begin

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: income and survival helpers
# ─────────────────────────────────────────────────────────────────────────────

@testset "income and survival helpers" begin
    inc = lifecycle_income(0.95, 0.2, 5)
    @test inc isa _MFLC.IncomeProcess{Float64}
    @test length(inc.states) == 5
    @test all(inc.states .> 0)                              # LEVELS, not logs
    @test dot(inc.stationary_dist, inc.states) ≈ 1.0        # normalized to unit mean
    @test all(≈(1.0), sum(inc.transition; dims=2))
    # The raw chain is in logs and averages to zero — the exact trap this helper avoids.
    @test abs(dot(rouwenhorst(0.95, 0.2, 5).stationary_dist,
                  rouwenhorst(0.95, 0.2, 5).states)) < 1e-10
    @test lifecycle_income(0.9, 0.1, 3; method=:tauchen) isa _MFLC.IncomeProcess
    @test_throws ArgumentError lifecycle_income(0.9, 0.1, 3; method=:nope)

    s = lifecycle_survival(65)
    @test length(s) == 65
    @test s[end] == 0.0                                     # nobody survives the last age
    @test all(0 .<= s .<= 1)
    @test issorted(s[1:64]; rev=true)                       # mortality rises with age
    @test s[1] > 0.999 && s[64] < 0.95                      # steep enough to matter
    @test_throws ArgumentError lifecycle_survival(1)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: ANALYTIC ORACLE — deterministic, unconstrained life cycle
# ─────────────────────────────────────────────────────────────────────────────

@testset "backward EGM sweep vs closed form" begin
    J = 12; r = 0.04; w = 1.0; beta = 0.96; sigma = 2.0
    kappa = [1.0 + 0.05 * (j - 1) for j in 1:J]
    m = LifeCycleOLG(; J=J, J_retire=J+1, survival=ones(J), earnings=kappa,
                     income=_lc_det_income(), a_max=40.0, n_a=400, beta=beta,
                     sigma=sigma, replacement=0.0, credit_limit=-20.0)
    c_pol, a_pol = lifecycle_policies(m, r, w)
    cs, as = _lc_age_path(m, c_pol, a_pol)

    # With no risk and no binding constraint the Euler equation holds exactly:
    # c_{j+1}/c_j = (β(1+r))^{1/σ}. This is an EXACT identity, not an approximation.
    growth = (beta * (1 + r))^(1 / sigma)
    for j in 1:(J-1)
        @test cs[j+1] / cs[j] ≈ growth rtol=1e-10
    end

    # Households may borrow during life but cannot die in debt, so the lifetime
    # budget binds with terminal assets exactly zero.
    @test as[end] ≈ 0.0 atol=1e-8
    pv_c = sum(cs[j] / (1 + r)^(j - 1) for j in 1:J)
    pv_y = sum(w * kappa[j] / (1 + r)^(j - 1) for j in 1:J)
    @test pv_c ≈ pv_y rtol=1e-8

    # Backward induction is a FINITE sweep: the terminal age is the known rule,
    # not the outcome of a fixed point.
    @test all(a_pol[:, 1, J] .≈ 0.0)
    a_grid = m.grid.grids[1]
    coh_J = (1 + r) .* a_grid .+ w * kappa[J]
    # Only where terminal cash-on-hand is positive: deep in debt the agent cannot
    # repay and still consume, and consumption is floored rather than made negative.
    feasible = coh_J .> 1e-6
    @test any(feasible)
    @test c_pol[feasible, 1, J] ≈ coh_J[feasible]
    @test all(c_pol[.!feasible, 1, J] .> 0)

    # β(1+r) = 1 ⇒ perfectly flat consumption (the permanent-income benchmark).
    m2 = LifeCycleOLG(; J=J, J_retire=J+1, survival=ones(J), earnings=fill(1.0, J),
                      income=_lc_det_income(), a_max=40.0, n_a=400,
                      beta=1 / (1 + r), sigma=sigma, replacement=0.0,
                      credit_limit=-20.0)
    cs2, _ = _lc_age_path(m2, lifecycle_policies(m2, r, w)...)
    @test maximum(abs, cs2 .- cs2[1]) < 1e-10
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: demographics and the age-extended histogram
# ─────────────────────────────────────────────────────────────────────────────

@testset "cohort mass and distribution" begin
    J = 10
    s = collect(range(0.99, 0.90; length=J)); s[J] = 0.0
    m = LifeCycleOLG(; J=J, J_retire=8, survival=s, income=lifecycle_income(0.9, 0.2, 3),
                     a_max=30.0, n_a=60, n_pop=0.01)
    mu = _MFLC._lc_cohort_mass(m)
    @test length(mu) == J
    @test sum(mu) ≈ 1.0
    for j in 1:(J-1)
        @test mu[j+1] / mu[j] ≈ s[j] / (1 + m.n_pop) rtol=1e-12
    end
    @test issorted(mu; rev=true)                    # mortality + growth shrink cohorts

    _, a_pol = lifecycle_policies(m, 0.03, 1.0)
    dist = lifecycle_distribution(m, a_pol)
    @test size(dist) == (length(m.grid.grids[1]), 3, J)
    @test sum(dist) ≈ 1.0
    @test all(dist .>= 0)
    for j in 1:J
        @test sum(@view dist[:, :, j]) ≈ mu[j] rtol=1e-10   # age slices carry cohort mass
    end
    # Newborns hold zero assets and are drawn from the stationary productivity law.
    @test sum(@view dist[1, :, 1]) ≈ mu[1] rtol=1e-10
    @test vec(sum(@view(dist[:, :, 1]); dims=1)) ./ mu[1] ≈ m.income.stationary_dist rtol=1e-10
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: stationary equilibrium
# ─────────────────────────────────────────────────────────────────────────────

@testset "stationary equilibrium" begin
    spec = LifeCycleOLG(; J=40, J_retire=31, survival=0.995,
                        income=lifecycle_income(0.95, 0.2, 3), a_max=50.0, n_a=120,
                        beta=0.97, sigma=2.0, replacement=0.4)
    ss = lifecycle_steady_state(spec; r_bounds=(-0.01, 0.10), tol=1e-6, max_iter=45)
    @test ss isa LifeCycleSteadyState{Float64}
    @test ss.converged
    @test abs(ss.excess_demand) < 1e-6
    @test 0.0 < ss.r < 0.10 && ss.w > 0
    @test 2.0 < ss.K / ss.Y < 5.0                     # plausible capital-output ratio

    # ACCOUNTING IDENTITY: the reported aggregate capital IS the integral of the
    # age-asset distribution — not an independently maintained number.
    a_grid = spec.grid.grids[1]
    K_dist = sum(ss.dist[i, ie, j] * a_grid[i]
                 for i in eachindex(a_grid), ie in 1:length(spec.income.states),
                     j in 1:spec.J)
    @test K_dist ≈ ss.K rtol=1e-12
    # …and it clears against firm demand to the requested tolerance.
    @test abs(ss.K - (ss.K / ss.L) * ss.L) < 1e-12
    @test sum(ss.dist) ≈ 1.0
    @test sum(ss.cohort_mass) ≈ 1.0

    # Labor supply counts working ages only, weighted by the earnings profile.
    mean_e = dot(spec.income.stationary_dist, spec.income.states)
    L_hand = sum(ss.cohort_mass[j] * spec.earnings[j] * mean_e for j in 1:(spec.J_retire-1))
    @test ss.L ≈ L_hand rtol=1e-12

    # Pay-as-you-go social security balances: τ w L = pension × retired mass.
    mass_ret = sum(@view ss.cohort_mass[spec.J_retire:end])
    @test ss.tau * ss.w * ss.L ≈ ss.pension * mass_ret rtol=1e-10
    @test ss.transfer == 0.0                          # annuities ⇒ no accidental bequests

    # Assets are hump-shaped over the life cycle and peak just before retirement.
    @test 1 < argmax(ss.asset_profile) < spec.J
    @test argmax(ss.asset_profile) <= spec.J_retire
    @test ss.asset_profile[1] ≈ 0.0 atol=1e-10
    @test ss.asset_profile[end] < maximum(ss.asset_profile)
    @test all(isfinite, ss.consumption_profile) && all(ss.consumption_profile .> 0)
    # Retirees receive the pension, workers the after-tax wage bill.
    @test ss.income_profile[end] ≈ ss.pension rtol=1e-10

    report(ss)                                        # display smoke tests
    @test occursin("LifeCycleSteadyState", sprint(show, ss))
    @test occursin("LifeCycleOLG", sprint(show, spec))

    for v in (:profiles, :distribution, :policy)
        p = plot_result(ss; view=v)
        @test p isa MacroEconometricModels.PlotOutput
        @test occursin("svg", lowercase(p.html)) || occursin("d3", lowercase(p.html))
    end
    @test_throws ArgumentError plot_result(ss; view=:nope)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: the consumption hump requires imperfect annuitization
# ─────────────────────────────────────────────────────────────────────────────

@testset "consumption hump and annuity neutrality" begin
    J = 65
    surv = lifecycle_survival(J)
    common = (J=J, J_retire=45, survival=surv, income=lifecycle_income(0.95, 0.2, 3),
              a_max=60.0, n_a=110, beta=0.97, sigma=2.0, replacement=0.4)

    # Actuarially fair annuities pay the gross return (1+r)/s_j, so β·s_j·R_j = β(1+r)
    # EXACTLY: survival cancels out of the Euler equation and mortality cannot bend
    # the consumption path. With β(1+r) > 1 consumption must then rise monotonically.
    ss_a = lifecycle_steady_state(LifeCycleOLG(; common..., annuities=true);
                                  r_bounds=(-0.01, 0.20), tol=1e-6, max_iter=50)
    @test ss_a.converged
    @test ss_a.spec.beta * (1 + ss_a.r) > 1
    @test argmax(ss_a.consumption_profile) == J
    @test issorted(ss_a.consumption_profile)

    # Without annuities the Euler growth factor is (β·s_j·(1+r))^{1/σ}, which falls
    # below one once late-life mortality bites — the classic life-cycle hump.
    ss_b = lifecycle_steady_state(LifeCycleOLG(; common..., annuities=false);
                                  r_bounds=(-0.01, 0.20), tol=1e-6, max_iter=50)
    @test ss_b.converged
    @test ss_b.transfer > 0                            # accidental bequests are rebated
    cp = ss_b.consumption_profile
    peak = argmax(cp)
    @test 1 < peak < J                                 # interior peak: a genuine hump
    @test cp[peak] > cp[1] * 1.2
    @test cp[end] < cp[peak] * 0.95
    # Terciles capture the shape without demanding pointwise monotonicity.
    ter(a, b) = sum(@view cp[a:b]) / (b - a + 1)
    @test ter(1, 21) < ter(22, 43) > ter(44, J)
    # Strict monotonicity holds away from the retirement date. Right at it the
    # cross-sectional mean has a small discontinuity — everyone switches to the same
    # flat pension, so the composition of income changes — which is a feature of the
    # profile, not noise; the largest pre-peak dip here is 0.3% of peak consumption.
    @test issorted(cp[1:(common.J_retire - 8)])        # rises through working life…
    @test issorted(cp[common.J_retire:end]; rev=true)  # …and falls monotonically in retirement
    @test peak >= common.J_retire - 8
    # Mortality is what turns the path over: the growth factor crosses one late.
    @test ss_b.spec.beta * surv[1] * (1 + ss_b.r) > 1
    @test ss_b.spec.beta * surv[J-1] * (1 + ss_b.r) < 1
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: validation and honest non-convergence
# ─────────────────────────────────────────────────────────────────────────────

@testset "validation and non-convergence" begin
    # `rouwenhorst` returns LOG states; feeding one in would silently zero aggregate
    # labor and every factor price, so it is rejected with a pointer to the fix.
    @test_throws ArgumentError LifeCycleOLG(; J=10, income=rouwenhorst(0.9, 0.2, 3))
    ok = lifecycle_income(0.9, 0.2, 3)
    @test_throws ArgumentError LifeCycleOLG(; J=1, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, J_retire=1, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, beta=1.2, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, sigma=0.0, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, alpha=1.5, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, delta=2.0, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, n_pop=-2.0, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, replacement=-0.1, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, n_a=2, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, a_max=-1.0, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, survival=1.5, income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, survival=ones(3), income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, survival=[2.0; ones(9)], income=ok)
    @test_throws ArgumentError LifeCycleOLG(; J=10, earnings=ones(3), income=ok)

    m = LifeCycleOLG(; J=10, J_retire=8, income=ok, a_max=20.0, n_a=40)
    @test_throws ArgumentError lifecycle_steady_state(m; r_bounds=(0.05, 0.01))
    @test_throws ArgumentError lifecycle_steady_state(m; r_bounds=(-0.5, 0.05))

    # A bracket that does not straddle the equilibrium is reported, not papered over.
    local ss_bad
    @test_logs (:warn, r"does not change sign") match_mode=:any begin
        ss_bad = lifecycle_steady_state(m; r_bounds=(0.20, 0.30), tol=1e-8, max_iter=5)
    end
    @test !ss_bad.converged
    @test abs(ss_bad.excess_demand) > 1e-8
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: to_spec → ModelSpec with LifeCycleSystem (#640 / G-03)
# ─────────────────────────────────────────────────────────────────────────────

@testset "to_spec wraps LifeCycleSystem (#640)" begin
    m0 = LifeCycleOLG()
    spec0 = to_spec(m0)
    @test spec0 isa ModelSpec
    @test spec0.n_endog == 0 && spec0.n_exog == 0
    @test isempty(spec0.equations) && isempty(spec0.residual_fns)
    @test only(keys(spec0.agents)) === :households
    @test only(values(spec0.agents)) isa LifeCycleSystem
    @test only(values(spec0.agents)).model === m0
    @test spec0.ir.horizon === :ages

    m = LifeCycleOLG(; J=40, J_retire=31, survival=0.995,
                     income=lifecycle_income(0.95, 0.2, 3), a_max=50.0, n_a=120,
                     beta=0.97, sigma=2.0, replacement=0.4)
    spec = to_spec(m)
    @test only(values(spec.agents)) isa LifeCycleSystem
    @test only(values(spec.agents)).model === m
    @test MacroEconometricModels.has_kind(spec, LifeCycleSystem)

    spec_named = to_spec(m; agent_name=:cohorts)
    @test only(keys(spec_named.agents)) === :cohorts
    @test only(values(spec_named.agents)).model === m

    ss_m = lifecycle_steady_state(m; r_bounds=(-0.01, 0.10), tol=1e-6, max_iter=45)
    ss_w = lifecycle_steady_state(only(values(spec.agents)).model;
                                  r_bounds=(-0.01, 0.10), tol=1e-6, max_iter=45)
    @test ss_m.converged && ss_w.converged
    @test ss_w.r ≈ ss_m.r atol=1e-6
    ss_solve = solve(spec; r_bounds=(-0.01, 0.10), tol=1e-6, max_iter=45)
    @test ss_solve isa LifeCycleSteadyState
    @test ss_solve.converged
    @test ss_solve.r ≈ ss_m.r atol=1e-6
end

end # @testset "Life-Cycle OLG"
