# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using Random

struct DummyAgent{T} <: AbstractAgentSystem{T}
    x::Int
end

@testset "Follow-on agent kinds (#631)" begin
    @test DCEGMSystem <: AbstractAgentSystem
    @test LifeCycleSystem <: AbstractAgentSystem
    @test ContinuousHouseholdSystem <: AbstractAgentSystem
    prob = dcegm_retirement_model(; n_a=20, n_periods=4)
    sys = DCEGMSystem(prob)
    @test sys isa DCEGMSystem
    @test sys.problem === prob
end

@testset "G-06: empty endog/exog HA is partial GE (#643)" begin
    spec = @dsge begin
        parameters: beta_hh = 0.99
        heterogeneous: a in [0.0, 10.0], n_grid = 20, utility = log, discount = beta_hh, borrowing = 0.0
        idiosyncratic: e ~ Rouwenhorst(0.9, 0.1, 3)
        aggregation: K = sum(a)
    end
    @test spec isa ModelSpec
    @test spec.n_endog == 0
    @test spec.n_exog == 0
    @test isempty(spec.equations)
    @test MacroEconometricModels.has_kind(spec, HouseholdSystem)
end

@testset "G-06: RA still requires endog/exog (#643)" begin
    @test_throws LoadError eval(:(@dsge begin
        parameters: ρ = 0.9
        y[t] = ρ * y[t-1]
    end))
end

@testset "G-06: HA one-sided empty lists error (#643)" begin
    @test_throws LoadError eval(:(@dsge begin
        parameters: beta_hh = 0.99
        endogenous: K
        heterogeneous: a in [0.0, 10.0], n_grid = 20, utility = log, discount = beta_hh, borrowing = 0.0
        idiosyncratic: e ~ Rouwenhorst(0.9, 0.1, 3)
        aggregation: K = sum(a)
    end))
end

@testset "G-05: solve dispatches by kind (#642)" begin
    @testset "DCEGMSystem" begin
        prob = dcegm_retirement_model(; n_a=20, n_periods=4)
        spec = to_spec(prob)
        sol = solve(spec)
        ref = dcegm_solve(prob)
        @test sol isa DCEGMSolution
        @test sol.n_periods == ref.n_periods
        @test sol.converged && ref.converged
        c1, v1 = dcegm_policy(sol, 1, 1, 1, 20.0)
        c2, v2 = dcegm_policy(ref, 1, 1, 1, 20.0)
        @test c1 ≈ c2 atol=0 rtol=0
        @test v1 ≈ v2 atol=0 rtol=0
    end

    @testset "LifeCycleSystem" begin
        m = LifeCycleOLG(; J=40, J_retire=31, survival=0.995,
                         income=lifecycle_income(0.95, 0.2, 3), a_max=50.0, n_a=80,
                         beta=0.97, sigma=2.0, replacement=0.4)
        spec = to_spec(m)
        sol = solve(spec; r_bounds=(-0.01, 0.10), tol=1e-5, max_iter=40)
        ref = lifecycle_steady_state(m; r_bounds=(-0.01, 0.10), tol=1e-5, max_iter=40)
        @test sol isa LifeCycleSteadyState
        @test sol.r ≈ ref.r atol=1e-6
    end

    @testset "ContinuousHouseholdSystem (one-asset)" begin
        m = CTAiyagari(; I=50)
        spec = to_spec(m)
        sol = solve(spec; tol=1e-5)
        ref = ct_steady_state(m; tol=1e-5)
        @test sol isa CTSteadyState
        @test sol.r ≈ ref.r
        @test sol.K ≈ ref.K
    end

    @testset "unknown kind names the type" begin
        spec = ModelSpec{Float64}(
            Symbol[], Symbol[], Symbol[], Dict{Symbol,Float64}(),
            NamedEquation[], Function[], 0, Int[], Float64[];
            agents=(dummy=DummyAgent{Float64}(0),))
        err = try
            solve(spec)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("DummyAgent", sprint(showerror, err))
    end

    @testset "multiple populations error until G-17" begin
        prob = dcegm_retirement_model(; n_a=20, n_periods=4)
        spec = ModelSpec{Float64}(
            Symbol[], Symbol[], Symbol[], Dict{Symbol,Float64}(),
            NamedEquation[], Function[], 0, Int[], Float64[];
            agents=(a=DCEGMSystem(prob), b=DCEGMSystem(prob)))
        @test_throws ArgumentError solve(spec)
    end
end

@testset "G-13a: Blanchard TFP irf via solve(to_spec) (#647a)" begin
    m = BlanchardOLG(; gamma=0.98, beta=0.96)
    spec = to_spec(m; rho_z=0.9, sigma_z=0.01)
    sol = solve(spec)
    @test sol isa DSGESolution
    @test is_determined(sol)
    @test sol.eu == [1, 1]
    ss = blanchard_steady_state(m)
    @test sol.spec.steady_state[1] ≈ ss.k atol=1e-8
    @test sol.spec.steady_state[2] ≈ ss.C atol=1e-8
    @test sol.spec.steady_state[3] ≈ ss.r atol=1e-8
    fam = blanchard_solve(m, ss)
    @test any(λ -> isapprox(abs(λ), fam.stable_eig; atol=1e-6), sol.eigenvalues)
    resp = irf(sol, 20)
    @test all(isfinite, resp.values)
    @test maximum(abs, resp.values) > 0
    @test resp.shocks == ["eps_Z"]
    @test resp.variables == ["k", "C", "r", "w", "Z"]
    # Z is the AR(1): impact = σ, then ρ, ρ², …
    @test resp.values[1, 5, 1] ≈ 0.01 atol=1e-8
    @test resp.values[2, 5, 1] ≈ 0.009 atol=1e-8
end

@testset "G-14: irf/fevd/simulate façade (#648)" begin
    m = BlanchardOLG(; gamma=0.98, beta=0.96)
    spec = to_spec(m; rho_z=0.9, sigma_z=0.01)
    sol = solve(spec)
    resp = irf(spec, 12)
    @test resp isa ImpulseResponse
    @test all(isfinite, resp.values)
    @test resp.variables == ["k", "C", "r", "w", "Z"]
    fv = fevd(spec, 12)
    @test fv isa FEVD
    @test all(isfinite, fv.proportions)
    path = simulate(sol, 20; rng=Random.MersenneTwister(1))
    @test size(path) == (20, 5)
    @test all(isfinite, path)

    pe = dcegm_solve(dcegm_retirement_model(; n_a=20, n_periods=4))
    @test_throws ArgumentError irf(pe, 8)
end

@testset "G-23 leftover @dsge keys (#657)" begin
    @testset "clock:/horizon: set ModelIR flags only" begin
        spec = @dsge begin
            clock: continuous
            horizon: ages
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y, a
            exogenous: ε
            y[t] = a[t]
            a[t] = ρ * a[t-1] + σ * ε[t]
        end
        @test spec isa ModelSpec{Float64,NoAgents}
        @test spec.ir.clock === :continuous
        @test spec.ir.horizon === :ages
        @test !MacroEconometricModels.has_kind(spec, ContinuousHouseholdSystem)
        @test !MacroEconometricModels.has_kind(spec, LifeCycleSystem)
        kinds = [d.kind for d in spec.ir.declarations]
        @test :clock in kinds
        @test :horizon in kinds
    end

    @testset "horizon: compound keys stored, not compiled" begin
        spec = @dsge begin
            horizon: ages, J = 60, retire = 45, survival = 0.995
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y, a
            exogenous: ε
            y[t] = a[t]
            a[t] = ρ * a[t-1] + σ * ε[t]
        end
        @test spec.ir.horizon === :ages
        @test spec isa ModelSpec{Float64,NoAgents}
        @test !MacroEconometricModels.has_kind(spec, LifeCycleSystem)
        hor = only(d for d in spec.ir.declarations if d.kind === :horizon)
        @test hor.payload isa Expr
    end

    @testset "discrete:/absorbing: extra_decls only" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y, a
            exogenous: ε
            discrete: work, retire
            absorbing: retire
            y[t] = a[t]
            a[t] = ρ * a[t-1] + σ * ε[t]
        end
        @test spec isa ModelSpec{Float64,NoAgents}
        @test !MacroEconometricModels.has_kind(spec, DCEGMSystem)
        kinds = [d.kind for d in spec.ir.declarations]
        @test :discrete in kinds
        @test :absorbing in kinds
    end

    @testset "invalid clock/horizon rejected" begin
        @test_throws LoadError eval(:(@dsge begin
            clock: monthly
            parameters: ρ = 0.9
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + ε[t]
        end))
        @test_throws LoadError eval(:(@dsge begin
            horizon: overlapping
            parameters: ρ = 0.9
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + ε[t]
        end))
    end

    @testset "HA leftover keys stored; still HouseholdSystem" begin
        spec = @dsge begin
            parameters: beta_hh = 0.99
            clock: continuous
            horizon: ages, J = 40, survival = 0.995
            heterogeneous: a in [0.0, 10.0], n_grid = 20, utility = log, discount = beta_hh, borrowing = 0.0
            idiosyncratic: e ~ Rouwenhorst(0.9, 0.1, 3)
            aggregation: K = sum(a)
            discrete: work, retire
            absorbing: retire
        end
        @test spec isa ModelSpec
        @test MacroEconometricModels.has_kind(spec, HouseholdSystem)
        @test !MacroEconometricModels.has_kind(spec, DCEGMSystem)
        @test !MacroEconometricModels.has_kind(spec, LifeCycleSystem)
        @test !MacroEconometricModels.has_kind(spec, ContinuousHouseholdSystem)
        @test spec.ir.clock === :continuous
        @test spec.ir.horizon === :ages
        kinds = [d.kind for d in spec.ir.declarations]
        @test :discrete in kinds
        @test :absorbing in kinds
        @test :horizon in kinds
        @test :clock in kinds
    end

    @testset "HA heterogeneous whitelist" begin
        @test_throws LoadError eval(:(@dsge begin
            parameters: beta_hh = 0.99
            heterogeneous: a in [0.0, 10.0], n_grid = 20, utility = log, discount = beta_hh, liquid = 5.0
            idiosyncratic: e ~ Rouwenhorst(0.9, 0.1, 3)
            aggregation: K = sum(a)
        end))
        @test_throws LoadError eval(:(@dsge begin
            parameters: beta_hh = 0.99
            heterogeneous: a in [0.0, 10.0], n_grid = 20, utility = log, discount = beta_hh, labor = ghh
            idiosyncratic: e ~ Rouwenhorst(0.9, 0.1, 3)
            aggregation: K = sum(a)
        end))
        @test_throws LoadError eval(:(@dsge begin
            parameters: beta_hh = 0.99
            heterogeneous: a in [0.0, 10.0], n_grid = 20, utility = log, discount = beta_hh, model = khan_thomas
            idiosyncratic: e ~ Rouwenhorst(0.9, 0.1, 3)
            aggregation: K = sum(a)
        end))
    end
end
