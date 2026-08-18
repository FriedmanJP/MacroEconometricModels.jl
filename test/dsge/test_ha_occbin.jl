# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels

const M = MacroEconometricModels
const _suppress_warnings = M._suppress_warnings

# -----------------------------------------------------------------------------
# Thin programmatic fixtures (not shipped examples.jl)
# -----------------------------------------------------------------------------

function _tiny_household()
    T = Float64
    inc = rouwenhorst(0.9, 0.1, 3)
    grid = HAGrid(; assets=(0.0, 10.0, 12), income_states=3, grid_type=:linear)
    ip = IndividualProblem{T}(
        c -> log(max(c, 1e-15)),
        c -> inv(max(c, 1e-15)),
        m -> inv(max(m, 1e-15)),
        0.99,
        (a, e, prices) -> (one(T) + get(prices, :r, T(0.01))) * a +
                          get(prices, :w, one(T)) * e,
        [zero(T)], nothing, 1)
    hh = HouseholdSystem{T}(
        ip, inc, grid,
        Pair{Symbol,Function}[:K => (dist, g) -> zero(T)],
        Dict{Symbol,T}(:r => T(0.01), :w => one(T)))
    return hh
end

"""Partial-GE HA spec: HouseholdSystem, no aggregate residuals (`n_endog == 0`)."""
function _empty_endog_ha()
    T = Float64
    hh = _tiny_household()
    ModelSpec{T}(
        Symbol[], Symbol[], Symbol[], Dict{Symbol,T}(),
        M.NamedEquation[], Function[],
        0, Int[], T[];
        agents=NamedTuple{(:household,)}((hh,)))
end

"""
NK-HANK fixture: `HouseholdSystem` + Taylor `i` + NKPC.

Demand is an AR(1) so the binding (`i[t] = 0`) regime stays determinate.
OccBin runs on these named aggregates, not dummy Cobb–Douglas residuals.
"""
function _nk_hank_fixture()
    T = Float64
    hh = _tiny_household()
    pv = Dict{Symbol,T}(
        :beta => T(0.99), :kappa => T(0.10),
        :phi_pi => T(1.5), :phi_y => T(0.5),
        :rho => T(0.5), :i_bar => T(0.01))
    fns = Function[
        (yt, yl, yle, ε, θ) -> yt[1] - θ[:rho] * yl[1] - (isempty(ε) ? zero(T) : ε[1]),
        (yt, yl, yle, ε, θ) -> yt[2] - θ[:beta] * yle[2] - θ[:kappa] * yt[1],
        (yt, yl, yle, ε, θ) -> yt[3] - θ[:i_bar] - θ[:phi_pi] * yt[2] - θ[:phi_y] * yt[1],
    ]
    eqs = M.NamedEquation[
        M.NamedEquation(:y, :y, :(y[t] - rho * y[t-1] - e[t]), fns[1]),
        M.NamedEquation(:nkpc, :pi, :(pi[t] - beta * pi[t+1] - kappa * y[t]), fns[2];
                        timing=M.TimingInfo(0, 1, true)),
        M.NamedEquation(:taylor, :i, :(i[t] - i_bar - phi_pi * pi[t] - phi_y * y[t]), fns[3]),
    ]
    ModelSpec{T}(
        [:y, :pi, :i], [:e],
        [:beta, :kappa, :phi_pi, :phi_y, :rho, :i_bar], pv,
        eqs, fns, 1, [2], T[],
        θ -> T[zero(T), zero(T), θ[:i_bar]];
        agents=NamedTuple{(:household,)}((hh,)))
end

function _throws_654(f)
    err = try
        f()
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("#654", msg)
    @test !occursin("HADSGESolution", msg)
    @test !occursin("has no field", msg)
    return err
end

@testset "G-20 OccBin on HA aggregate equations (#654)" begin

    @testset "n_endog == 0 throws ArgumentError #654" begin
        spec0 = _empty_endog_ha()
        @test spec0.n_endog == 0
        @test M.has_kind(spec0, HouseholdSystem)
        _throws_654(() -> occbin_solve(spec0, :(i[t] >= 0)))
        c = OccBinConstraint{Float64}(:(i[t] >= 0), :i, 0.0, :geq, :(i[t] = 0.0))
        _throws_654(() -> occbin_solve(spec0, c))
    end

    @testset "shipped real-HANK throws ArgumentError #654" begin
        for name in (:one_asset_hank, :krusell_smith)
            spec = load_ha_example(name)
            @test M.has_kind(spec, HouseholdSystem)
            @test :i ∉ spec.endog
            @test :r in spec.endog
            _throws_654(() -> occbin_solve(spec, :(i[t] >= 0)))
            _throws_654(() -> occbin_solve(spec, :(r[t] >= 0)))
        end
    end

    @testset "NK-HANK ELB piecewise path differs when bound binds" begin
        spec = _nk_hank_fixture()
        @test M.has_kind(spec, HouseholdSystem)
        @test :i in spec.endog
        @test any(eq -> eq.defines === :i, spec.equations)
        @test any(eq -> eq.defines === :pi, spec.equations)

        shock_path = zeros(40, 1)
        shock_path[1, 1] = -2.0   # large negative demand → unconstrained i < 0

        sol = _suppress_warnings() do
            occbin_solve(spec, :(i[t] >= 0); shock_path=shock_path, nperiods=40)
        end
        @test sol isa OccBinSolution{Float64}
        @test size(sol.linear_path) == (40, 3)
        @test size(sol.piecewise_path) == (40, 3)
        @test sol.converged

        i_idx = findfirst(==("i"), sol.varnames)
        @test i_idx !== nothing
        i_lin = sol.linear_path[:, i_idx] .+ sol.steady_state[i_idx]
        i_pw  = sol.piecewise_path[:, i_idx] .+ sol.steady_state[i_idx]
        @test minimum(i_lin) < 0
        @test any(sol.regime_history[:, 1] .== 1)
        @test minimum(i_pw) >= -1e-8
        @test sol.linear_path != sol.piecewise_path
        @test maximum(abs, sol.linear_path .- sol.piecewise_path) > 1e-8
    end
end
