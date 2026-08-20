# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using Statistics

"""Small Bewley-bank fixture: one permanent κ, Rouwenhorst ξ, coarse n-grid."""
function _bb_sys(; n_xi=3, zeta1=0.02, n_n=21)
    return IntermediarySystem(;
        n_min=0.08, n_max=6.0, n_n=n_n,
        n_xi=n_xi, rho_xi=0.55, sigma_xi=0.08,
        kappa=1.0, beta=0.99, sigma=0.94, lambda=0.20,
        zeta1=zeta1, zeta2=2.0,
        R=1.01, rk=0.05, Z=0.25, alpha=0.33)
end

@testset "G-19: IntermediarySystem kind (#653)" begin
    @test IntermediarySystem <: AbstractAgentSystem
    sys = _bb_sys()
    @test sys isa IntermediarySystem{Float64}
    @test sys.model === :bewley_banks
    @test sys.distribution === :young
    @test sys.grid.n_dims == 1
    @test length(sys.xi.states) == 3
    @test all(>(0), sys.xi.states)
    @test MacroEconometricModels.grid(sys) === sys.grid
    @test MacroEconometricModels.idiosyncratic(sys) === sys.xi
    @test MacroEconometricModels.ssj_inputs(sys) == [:R, :rk]
    @test :L in MacroEconometricModels.ssj_outputs(sys)

    spec = to_spec(sys)
    @test spec isa ModelSpec
    @test spec.n_endog == 0
    @test isempty(spec.equations)
    @test MacroEconometricModels.has_kind(spec, IntermediarySystem)
    @test !MacroEconometricModels.has_kind(spec, HouseholdSystem)
    @test keys(spec.agents) == (:banks,)
    @test only(values(spec.agents)) === sys

    spec2 = to_spec(sys; agent_name=:dealers)
    @test keys(spec2.agents) == (:dealers,)
    @test spec2.agents.dealers isa IntermediarySystem
    @test occursin("Jamilov", string(@doc IntermediarySystem))
end

@testset "G-19: PE franchise V(n, ξ) given (R, rᵏ)" begin
    sys = _bb_sys()
    pe = intermediary_pe(sys; R=1.01, rk=0.06, max_iter=180, tol=1e-6)
    @test pe isa IntermediaryPE
    @test pe.converged
    @test all(isfinite, pe.V)
    @test all(>(0), pe.V)
    @test all(isfinite, pe.l_policy)
    @test all(>=(0), pe.l_policy)
    # GK incentive constraint λ l ≤ V
    @test maximum(sys.lambda .* pe.l_policy .- pe.V) <= 1e-6
    # Value rises in net worth (monotone on a coarse grid, weak test)
    @test pe.V[end, 1] > pe.V[1, 1]
    # Balance sheet: b = l − n
    @test pe.b_policy ≈ pe.l_policy .- sys.grid.grids[1]
end

@testset "G-19: stationary credit clearing, leverage, n dispersion" begin
    sys = _bb_sys()
    spec = to_spec(sys)
    ss = intermediary_steady_state(sys; tol=2e-3, max_iter=18,
                                   pe_max_iter=140, pe_tol=1e-5)
    @test ss isa IntermediarySteadyState
    @test isfinite(ss.aggregates[:leverage])
    @test isfinite(ss.aggregates[:L])
    @test ss.aggregates[:L] > 0
    @test ss.aggregates[:leverage] > 0
    @test all(isfinite, ss.V)
    @test isapprox(sum(ss.distribution), 1; atol=1e-8)

    # n distribution has positive dispersion when ξ is on
    n_grid = ss.grid.grids[1]
    mass_n = vec(sum(ss.distribution; dims=2))
    μn = sum(n_grid .* mass_n)
    vn = sum((n_grid .- μn).^2 .* mass_n)
    @test vn > 1e-6

    # solve / compute_steady_state dispatch on the kind
    sol = solve(spec; tol=2e-3, max_iter=18, pe_max_iter=140, pe_tol=1e-5)
    @test sol isa IntermediarySteadyState
    @test isfinite(sol.aggregates[:L])
    css = compute_steady_state(spec; tol=2e-3, max_iter=18,
                               pe_max_iter=140, pe_tol=1e-5)
    @test css isa IntermediarySteadyState
end

@testset "G-19: TFP IRF of L via MIT" begin
    sys = _bb_sys()
    ss = intermediary_steady_state(sys; tol=2e-3, max_iter=16,
                                   pe_max_iter=120, pe_tol=1e-5)
    resp = irf(ss, 8; shock_size=0.02, persist=0.5, pe_max_iter=50, pe_tol=1e-4)
    @test resp isa ImpulseResponse
    @test all(isfinite, resp.values)
    @test resp.variables == ["L", "Y", "Z"]
    @test size(resp.values, 1) == 8
    # impact on Z is the shock; L path is finite (may be small)
    @test resp.values[1, 3, 1] ≈ ss.system.Z * 0.02 atol=1e-10

    spec = to_spec(sys)
    # façade: irf(spec) → solve → irf(::IntermediarySteadyState)
    resp2 = irf(ss, 6; shock_size=0.01, persist=0.0,
                pe_max_iter=40, pe_tol=1e-4)
    @test all(isfinite, resp2.values)
    @test MacroEconometricModels.has_kind(spec, IntermediarySystem)
end

@testset "G-19: GK 2011 is the nested representative case" begin
    # ζ₁ = 0 and ξ off → scale-invariant leverage (IC binds, V ∝ n)
    sys = _bb_sys(; n_xi=1, zeta1=0.0, n_n=17)
    pe = intermediary_pe(sys; R=1.01, rk=0.08, max_iter=200, tol=1e-6)
    @test pe.converged
    n_grid = sys.grid.grids[1]
    lev = pe.l_policy[3:end-2, 1] ./ n_grid[3:end-2]
    @test all(isfinite, lev)
    @test mean(lev) > 1
    # With ζ₁ = 0 the IC binds: λ l ≈ V (scale-invariant GK nest).
    @test maximum(abs.(sys.lambda .* pe.l_policy .- pe.V)) < 1e-5
end

@testset "intermediary_steady_state no-sign-change is not converged (MSR-09)" begin
    sys = _bb_sys()
    ss = @test_logs (:warn, r"no sign change") match_mode=:any begin
        intermediary_steady_state(sys; r_bounds=(5.0, 6.0), max_iter=4,
                                  pe_max_iter=20, pe_tol=1e-3)
    end
    @test ss.converged == false
    buf = IOBuffer()
    report(buf, ss)
    out = lowercase(String(take!(buf)))
    @test occursin("no", out) || occursin("false", out)
end

@testset "intermediary default mass goes to entry (MSR-10)" begin
    sys = IntermediarySystem(;
        n_min=0.08, n_max=6.0, n_n=11, n_enter=1.0,
        n_xi=3, rho_xi=0.1, sigma_xi=0.4,
        kappa=1.0, beta=0.99, sigma=0.94, lambda=0.20,
        zeta1=0.5, zeta2=3.0,
        R=1.05, rk=0.02, Z=0.25, alpha=0.33)
    n_grid = sys.grid.grids[1]
    n_n = length(n_grid)
    n_e = length(sys.xi.states)
    # Large lending so n' is negative for a bad ξ'
    l_pol = fill(8.0, n_n, n_e)
    Λ = MacroEconometricModels._intermediary_transition(l_pol, sys, sys.R, sys.rk)
    @test all(abs.(vec(sum(Λ; dims=1)) .- 1) .< 1e-12)
    ke, _, _ = MacroEconometricModels._young_bracket(n_grid, sys.n_enter)
    @test ke > 1
    entry_mass = sum(Λ[ke, :]) + (ke < n_n ? sum(Λ[ke + 1, :]) : 0.0)
    floor_mass = sum(Λ[1, :])
    @test entry_mass > floor_mass
end

@testset "ξ chain mean is μ exactly (MSR-27)" begin
    for (rho, sig, n, mu) in ((0.553, 0.085, 3, 1.0),
                              (0.2, 0.3, 7, 1.25),
                              (0.9, 0.05, 5, 0.8))
        xi = MacroEconometricModels._xi_process(rho, sig, n; mu=mu)
        @test abs(sum(xi.stationary_dist .* xi.states) - mu) < 1e-12
    end
    xi1 = MacroEconometricModels._xi_process(0.5, 0.1, 1; mu=1.7)
    @test xi1.states == [1.7]
    @test abs(sum(xi1.stationary_dist .* xi1.states) - 1.7) < 1e-12
    sys = IntermediarySystem(; n_xi=5, rho_xi=0.4, sigma_xi=0.2, xi_mean=1.3,
                             n_n=9, n_min=0.1, n_max=3.0)
    @test abs(sum(sys.xi.stationary_dist .* sys.xi.states) - 1.3) < 1e-12
end

@testset "lending scan density / l_cap_mult (MSR-27)" begin
    # Wide ξ + deposit rate above the bad-state claim return: limited
    # liability kinks n'(l), so the franchise payoff is not concave in l.
    sys = IntermediarySystem(;
        n_min=0.1, n_max=2.0, n_n=13,
        n_xi=5, rho_xi=0.10, sigma_xi=0.60,
        kappa=1.0, beta=0.99, sigma=0.90, lambda=0.18,
        zeta1=0.01, zeta2=2.0,
        R=1.08, rk=0.05, Z=0.25, alpha=0.33)
    n_grid = sys.grid.grids[1]
    n_n = length(n_grid)
    n_e = length(sys.xi.states)
    V = [0.4 * n_grid[i] for i in 1:n_n, _ in 1:n_e]
    n = n_grid[div(n_n, 2)]
    j = 1
    l_hi = 20.0
    kinked = false
    for l in range(0.2, l_hi; length=50)
        np_lo = MacroEconometricModels._bank_nprime(
            n, l, minimum(sys.xi.states), sys.kappa, sys.rk, sys.R,
            sys.zeta1, sys.zeta2)
        np_hi = MacroEconometricModels._bank_nprime(
            n, l, maximum(sys.xi.states), sys.kappa, sys.rk, sys.R,
            sys.zeta1, sys.zeta2)
        if np_lo <= 0 < np_hi
            kinked = true
            break
        end
    end
    @test kinked

    l16, v16 = MacroEconometricModels._best_lending(
        n, j, l_hi, V, n_grid, sys.xi, sys.kappa, sys.rk, sys.R,
        sys.zeta1, sys.zeta2, sys.beta, sys.sigma; n_try=16)
    l32, v32 = MacroEconometricModels._best_lending(
        n, j, l_hi, V, n_grid, sys.xi, sys.kappa, sys.rk, sys.R,
        sys.zeta1, sys.zeta2, sys.beta, sys.sigma; n_try=32)
    @test isfinite(l16) && isfinite(l32)
    @test isapprox(l16, l32; rtol=1e-3, atol=1e-4)
    @test v32 >= v16 - 1e-10

    # Documented escape: l_cap_mult / n_try are honored and forwarded.
    mild = _bb_sys(; n_n=11)
    n_max = mild.grid.grids[1][end]
    pe_cap = intermediary_pe(mild; l_cap_mult=0.25, n_try=8,
                             max_iter=20, tol=1e-3)
    @test maximum(pe_cap.l_policy) <= 0.25 * n_max + 1e-12
    @test_throws ArgumentError intermediary_pe(mild; n_try=0)
    @test_throws ArgumentError intermediary_pe(mild; l_cap_mult=0)
    ss = intermediary_steady_state(mild; n_try=8, l_cap_mult=10,
                                   max_iter=3, pe_max_iter=12, pe_tol=1e-3)
    @test ss isa IntermediarySteadyState
    @test maximum(ss.l_policy) <= 10 * n_max + 1e-12
end
