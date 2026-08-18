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
