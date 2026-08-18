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
