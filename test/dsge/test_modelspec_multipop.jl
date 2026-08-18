# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# G-17 / #651 — solve() with more than one HouseholdSystem.

using Test
using MacroEconometricModels

const _MEM = MacroEconometricModels

function _two_huggett_spec(; n_a::Int=30, a_max::Float64=6.0, credit_limit::Float64=-2.0,
                           beta_u::Float64=0.96, beta_h::Float64=0.985)
    s1 = _MEM._huggett_example(; credit_limit=credit_limit, a_max=a_max, n_a=n_a,
                               beta=beta_u)
    s2 = _MEM._huggett_example(; credit_limit=credit_limit, a_max=a_max, n_a=n_a,
                               beta=beta_h)
    hh_u = only(values(s1.agents))
    hh_h = only(values(s2.agents))
    return ModelSpec{Float64}(
        Symbol[], Symbol[], s1.params, copy(s1.param_values),
        NamedEquation[], Function[], 0, Int[], Float64[];
        agents=(unconstrained=hh_u, htm=hh_h))
end

@testset "G-17: _hh unique or agents_of (#651)" begin
    spec1 = _MEM._huggett_example(; n_a=20, a_max=4.0, credit_limit=-1.0, beta=0.96)
    hh = only(values(spec1.agents))
    spec_named = ModelSpec{Float64}(
        Symbol[], Symbol[], spec1.params, copy(spec1.param_values),
        NamedEquation[], Function[], 0, Int[], Float64[];
        agents=(workers=hh,))
    @test _MEM._hh(spec_named) === hh
    @test !haskey(spec_named.agents, :household)

    spec = _two_huggett_spec()
    @test Set(keys(spec.agents)) == Set((:unconstrained, :htm))
    @test !haskey(spec.agents, :household)
    @test length(collect(_MEM.agents_of(spec, HouseholdSystem))) == 2
    err = try
        _MEM._hh(spec)
        ErrorException("expected _hh to throw")
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("agents_of", msg)
    @test occursin("#651", msg)
    @test occursin("unconstrained", msg) || occursin("htm", msg)
end

@testset "G-17: two named HouseholdSystems clear via solve(:ssj) (#651)" begin
    spec = _two_huggett_spec()
    @test spec.n_endog == 0
    @test isempty(spec.residual_fns)
    @test all(a -> a isa HouseholdSystem, values(spec.agents))

    sol = solve(spec; method=:ssj, T_horizon=12, n_reduced=6,
                max_iter=50, tol=1e-4)
    @test sol isa HADSGESolution
    @test sol.method === :ssj
    @test sol.steady_state.converged
    @test abs(sol.steady_state.excess_demand) < 5e-3
    @test isfinite(sol.steady_state.prices[:r])
    @test sol.steady_state.prices[:r] <
          1 / maximum(hh.individual.beta for hh in values(spec.agents)) - 1

    hh_by_name = Dict(k => v for (k, v) in _MEM._named_households(spec))
    ss_map = Dict{Symbol,Any}()
    for (k, hh) in hh_by_name
        pe = _MEM._household_asset_supply(hh, sol.steady_state.prices)
        ss_map[k] = _MEM._household_ss_from_pe(hh, sol.steady_state.prices, pe)
    end
    hu = hh_by_name[:unconstrained]
    hh = hh_by_name[:htm]
    hb_u = HetBlock(ss_map[:unconstrained], hu.individual, hu.grid, hu.income;
                    inputs=[:r, :w], outputs=[:A], name=:unconstrained)
    hb_h = HetBlock(ss_map[:htm], hh.individual, hh.grid, hh.income;
                    inputs=[:r, :w], outputs=[:B], name=:htm)
    mkt = SimpleBlock(x -> [x[1] + x[2]];
                      inputs=[:A, :B], outputs=[:bond_mkt],
                      ss_inputs=Dict(:A => hb_u.ss_outputs[:A],
                                     :B => hb_h.ss_outputs[:B]),
                      name=:bond_market)
    dag = combine_blocks(hb_u, hb_h, mkt; name=:two_hh, ss_tol=Inf)
    @test Set(b.name for b in dag.blocks) == Set((:unconstrained, :htm, :bond_market))
    @test abs(dag.ss_values[:bond_mkt]) < 5e-3

    Th = 8
    gej = ssj_jacobian(dag; unknowns=[:r], targets=[:bond_mkt], shocks=[:w],
                       T_horizon=Th, target_tol=Inf)
    @test all(isfinite, gej.H_U) && all(isfinite, gej.H_Z)
    dw = [0.9^(t - 1) for t in 1:Th]
    ir = ssj_irf(gej, Dict(:w => dw); residual=false)
    @test all(isfinite, ir.paths[:r])
    @test maximum(abs, gej.H_U * ir.paths[:r] .+ gej.H_Z * dw) < 1e-8

    # irf/fevd still assume a unique household; they name #651 until retargeted.
    err_irf = try
        irf(sol, 8)
        ErrorException("expected irf to throw")
    catch e
        e
    end
    @test err_irf isa ArgumentError
    @test occursin("#651", sprint(showerror, err_irf))
    @test occursin("agents_of", sprint(showerror, err_irf))
end

@testset "G-17: callers that assume one household name #651" begin
    spec = _two_huggett_spec()
    err_reiter = try
        solve(spec; method=:reiter)
        ErrorException("expected reiter to throw")
    catch e
        e
    end
    @test err_reiter isa ArgumentError
    @test occursin("#651", sprint(showerror, err_reiter))
    @test occursin("agents_of", sprint(showerror, err_reiter))

    prob = dcegm_retirement_model(; n_a=16, n_periods=4)
    mixed = ModelSpec{Float64}(
        Symbol[], Symbol[], Symbol[], Dict{Symbol,Float64}(),
        NamedEquation[], Function[], 0, Int[], Float64[];
        agents=(unconstrained=spec.agents.unconstrained,
                discrete=DCEGMSystem(prob)))
    err_mix = try
        solve(mixed)
        ErrorException("expected mixed kinds to throw")
    catch e
        e
    end
    @test err_mix isa ArgumentError
    mixmsg = sprint(showerror, err_mix)
    @test occursin("#651", mixmsg)
    @test occursin("agents_of", mixmsg)
end
