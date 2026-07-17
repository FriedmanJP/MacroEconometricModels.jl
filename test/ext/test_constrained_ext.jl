# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Constrained-DSGE weakdep extension tests (JuMP/Ipopt + PATHSolver).
#
# Consolidated here (issue #309/T210) so the JuMP/Ipopt/PATH cold-load
# (~1-3 min weakdep compile) is paid ONCE in a dedicated Extensions group
# instead of twice (formerly duplicated across test/dsge/test_dsge.jl and
# test/coverage/test_gmm_ext_coverage.jl). This is a verbatim MOVE of those
# testsets — names and assertions are unchanged.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics
using StatsAPI

Random.seed!(9006)

const _suppress_warnings = MacroEconometricModels._suppress_warnings

# JuMP + Ipopt availability (single consolidated guard)
_jump_available = try
    @eval import JuMP
    @eval import Ipopt
    true
catch
    false
end

# PATHSolver availability (single consolidated guard)
_path_available = try
    @eval import PATHSolver
    true
catch
    false
end

@testset "Constrained Extensions (JuMP/Ipopt/PATH)" begin

if _jump_available

@testset "Constrained Steady State (JuMP)" begin
    _suppress_warnings() do
    # Simple AR(1) with non-binding constraint
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    # Non-binding: SS is 0, lower=-1 doesn't bind (explicit Ipopt for JuMP test)
    spec_c = compute_steady_state(spec; constraints=[variable_bound(:y, lower=-1.0)], solver=:ipopt)
    @test abs(spec_c.steady_state[1]) < 0.01

    # Growth model with positivity constraints (non-binding at true SS)
    spec2 = @dsge begin
        parameters: alpha = 0.36, beta_disc = 0.99, delta = 0.025, gamma = 2.0, sigma_e = 0.01
        endogenous: k, c
        exogenous: epsilon
        steady_state = begin
            k_ss = (alpha / (1/beta_disc - 1 + delta))^(1 / (1 - alpha))
            c_ss = k_ss^alpha - delta * k_ss
            [k_ss, c_ss]
        end
        c[t]^(-gamma) - beta_disc * c[t+1]^(-gamma) * (alpha * k[t]^(alpha - 1) + 1 - delta) = 0
        k[t] - k[t-1]^alpha - (1 - delta) * k[t-1] + c[t] - sigma_e * epsilon[t] = 0
    end
    spec2 = compute_steady_state(spec2)
    k_ss = spec2.steady_state[1]
    c_ss = spec2.steady_state[2]

    spec2_c = compute_steady_state(spec2;
        constraints=[variable_bound(:k, lower=0.1), variable_bound(:c, lower=0.1)], solver=:ipopt)
    @test abs(spec2_c.steady_state[1] - k_ss) / k_ss < 0.05
    @test abs(spec2_c.steady_state[2] - c_ss) / c_ss < 0.05
    end # _suppress_warnings
end

@testset "Constrained Perfect Foresight (JuMP)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    # Moderate shock: unconstrained path stays above -2
    shocks = zeros(20, 1)
    shocks[1, 1] = -1.0

    pf_unc = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks)
    @test pf_unc.converged

    # Non-binding lower bound: constraint doesn't alter the path (explicit Ipopt)
    pf_con = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks,
                    constraints=[variable_bound(:y, lower=-5.0)], solver=:ipopt)
    @test pf_con isa PerfectForesightPath
    @test pf_con.converged
    @test all(pf_con.path[:, 1] .>= -5.0 - 1e-4)  # bound respected

    # Path dimensions
    @test size(pf_con.path) == (20, 1)

    # Constrained path matches unconstrained when bound is non-binding
    @test maximum(abs.(pf_con.path .- pf_unc.path)) < 0.01

    # Terminal convergence to SS
    @test abs(pf_con.path[end, 1]) < 0.5

    # Upper bound test: constrain y <= 5 (non-binding, explicit Ipopt)
    shocks2 = zeros(20, 1)
    shocks2[1, 1] = 2.0
    pf_upper = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks2,
                      constraints=[variable_bound(:y, upper=5.0)], solver=:ipopt)
    @test pf_upper isa PerfectForesightPath
    @test pf_upper.converged
    @test all(pf_upper.path[:, 1] .<= 5.0 + 1e-4)
    end # _suppress_warnings
end

@testset "Binding Steady-State Constraint (JuMP)" begin
    _suppress_warnings() do
    # AR(1) SS is 0.  Add binding lower bound y >= 0.5 → constrained SS at bound.
    # SS objective: min (y - 0.9y)² = (0.1y)² subject to y >= 0.5  →  y* = 0.5
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    @test abs(spec.steady_state[1]) < 1e-8  # unconstrained SS = 0

    spec_bound = compute_steady_state(spec; constraints=[variable_bound(:y, lower=0.5)], solver=:ipopt)
    @test spec_bound.steady_state[1] ≈ 0.5 atol=0.01  # bound binds

    # Binding PF with Ipopt: conflicting constraint → solver reports infeasibility
    shocks = zeros(10, 1)
    shocks[1, 1] = -3.0
    pf = solve(spec; method=:perfect_foresight, T_periods=10, shock_path=shocks,
                constraints=[variable_bound(:y, lower=0.0)], solver=:ipopt)
    @test pf isa PerfectForesightPath
    @test !pf.converged  # infeasible: equality constraints + binding bound conflict
    end # _suppress_warnings
end

@testset "NonlinearConstraint Integration (JuMP)" begin
    _suppress_warnings() do
    # AR(1) with nonlinear constraint on SS: y - 0.5 <= 0 (caps y at 0.5)
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    # Non-binding NL constraint: SS=0 < 1.5
    nlc_nb = nonlinear_constraint(
        (y, y_lag, y_lead, e, theta) -> y[1] - 1.5;
        label="cap_y"
    )
    spec_c = compute_steady_state(spec; constraints=[nlc_nb])
    @test abs(spec_c.steady_state[1]) < 0.01

    # Binding NL constraint: SS=0 but force y >= 0.3 via -(y - 0.3) <= 0 → y >= 0.3
    nlc_bind = nonlinear_constraint(
        (y, y_lag, y_lead, e, theta) -> -(y[1] - 0.3);
        label="floor_y"
    )
    spec_floor = compute_steady_state(spec; constraints=[nlc_bind])
    @test spec_floor.steady_state[1] ≈ 0.3 atol=0.01

    # Non-binding NL constraint in PF
    shocks = zeros(20, 1)
    shocks[1, 1] = 1.0
    pf = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks,
                constraints=[nlc_nb])
    @test pf isa PerfectForesightPath
    @test pf.converged
    @test all(pf.path[:, 1] .<= 1.5 + 1e-3)
    end # _suppress_warnings
end

@testset "JuMP Extension coverage" begin

    @testset "Constrained SS with initial_guess kwarg" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 0.01
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # Provide explicit initial_guess (exercises that branch in the extension)
            spec_c = compute_steady_state(spec;
                constraints=[variable_bound(:y, lower=-1.0)],
                initial_guess=[0.5])
            @test abs(spec_c.steady_state[1]) < 0.1
        end
    end

    @testset "Constrained SS with only upper bound" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 0.01
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # Upper bound only — exercises the c.upper !== nothing branch
            spec_c = compute_steady_state(spec;
                constraints=[variable_bound(:y, upper=1.0)])
            @test spec_c.steady_state[1] <= 1.0 + 1e-4
        end
    end

    @testset "Constrained PF with both bounds and NL constraint" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 1.0
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            shocks = zeros(10, 1)
            shocks[1, 1] = 2.0

            # Variable bound + nonlinear constraint together
            nlc = nonlinear_constraint(
                (y, y_lag, y_lead, e, theta) -> y[1] - 5.0;
                label="cap_y")
            pf = solve(spec; method=:perfect_foresight, T_periods=10,
                       shock_path=shocks,
                       constraints=[variable_bound(:y, lower=-1.0), nlc])
            @test pf isa PerfectForesightPath
            @test size(pf.path) == (10, 1)
        end
    end

    @testset "Constrained PF with upper-only variable bound" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 1.0
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            shocks = zeros(15, 1)
            shocks[1, 1] = 3.0

            # Upper-only bound
            pf = solve(spec; method=:perfect_foresight, T_periods=15,
                       shock_path=shocks,
                       constraints=[variable_bound(:y, upper=5.0)])
            @test pf isa PerfectForesightPath
            @test pf.converged
            @test all(pf.path[:, 1] .<= 5.0 + 1e-4)
        end
    end
end

end # _jump_available

if _path_available

@testset "MCP Steady State (PATH)" begin
    _suppress_warnings() do
    # Non-binding: SS is 0, lower=-1 doesn't bind → same as unconstrained
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    spec_mcp = compute_steady_state(spec; constraints=[variable_bound(:y, lower=-1.0)],
                                     solver=:path)
    @test abs(spec_mcp.steady_state[1]) < 0.01

    # Binding lower bound: SS=0 but lower=0.5 → MCP finds y=0.5 at bound
    spec_bind = compute_steady_state(spec; constraints=[variable_bound(:y, lower=0.5)],
                                      solver=:path)
    @test spec_bind.steady_state[1] ≈ 0.5 atol=0.01

    # Binding upper bound: SS=0 but upper=-0.5 → MCP finds y=-0.5 at bound
    spec_upper = compute_steady_state(spec; constraints=[variable_bound(:y, upper=-0.5)],
                                       solver=:path)
    @test spec_upper.steady_state[1] ≈ -0.5 atol=0.01

    # Multiple variables, mixed binding/non-binding
    spec2 = @dsge begin
        parameters: ρ1 = 0.9, ρ2 = 0.9, σ1 = 0.01, σ2 = 0.01
        endogenous: y1, y2
        exogenous: ε1, ε2
        y1[t] = ρ1 * y1[t-1] + σ1 * ε1[t]
        y2[t] = ρ2 * y2[t-1] + σ2 * ε2[t]
        steady_state: [0.0, 0.0]
    end
    spec2 = compute_steady_state(spec2)
    spec2_mcp = compute_steady_state(spec2;
        constraints=[variable_bound(:y1, lower=0.3), variable_bound(:y2, lower=-1.0)],
        solver=:path)
    @test spec2_mcp.steady_state[1] ≈ 0.3 atol=0.01   # binding
    @test abs(spec2_mcp.steady_state[2]) < 0.01         # non-binding
    end # _suppress_warnings
end

@testset "MCP Perfect Foresight (PATH)" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    # Non-binding lower bound: path not affected
    shocks = zeros(20, 1)
    shocks[1, 1] = -1.0
    pf_unc = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks)
    pf_mcp = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks,
                    constraints=[variable_bound(:y, lower=-5.0)], solver=:path)
    @test pf_mcp isa PerfectForesightPath
    @test pf_mcp.converged
    @test maximum(abs.(pf_mcp.path .- pf_unc.path)) < 0.1

    # ZLB BINDING: large negative shock + y >= 0 → y clamped at 0
    shocks_big = zeros(30, 1)
    shocks_big[1, 1] = -3.0
    pf_unc2 = solve(spec; method=:perfect_foresight, T_periods=30, shock_path=shocks_big)
    @test minimum(pf_unc2.path[:, 1]) < -0.5  # unconstrained goes negative

    pf_zlb = solve(spec; method=:perfect_foresight, T_periods=30, shock_path=shocks_big,
                    constraints=[variable_bound(:y, lower=0.0)], solver=:path)
    @test pf_zlb isa PerfectForesightPath
    @test pf_zlb.converged
    @test all(pf_zlb.path[:, 1] .>= -1e-6)     # bound respected
    @test pf_zlb.path[1, 1] ≈ 0.0 atol=1e-3    # bound binds at impact

    # Path dimensions
    @test size(pf_zlb.path) == (30, 1)

    # Terminal convergence to SS
    @test abs(pf_zlb.path[end, 1]) < 0.5

    # Upper bound binding
    shocks_pos = zeros(20, 1)
    shocks_pos[1, 1] = 3.0
    pf_cap = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks_pos,
                    constraints=[variable_bound(:y, upper=0.5)], solver=:path)
    @test pf_cap isa PerfectForesightPath
    @test pf_cap.converged
    @test all(pf_cap.path[:, 1] .<= 0.5 + 1e-4)
    end # _suppress_warnings
end

@testset "Auto-detection defaults to NonlinearSolve" begin
    # Bounds-only now defaults to :nonlinearsolve; PATH still available via override
    bounds_only = [variable_bound(:y, lower=0.0)]
    @test MacroEconometricModels._select_solver(bounds_only, nothing) == :nonlinearsolve
    @test MacroEconometricModels._select_solver(bounds_only, :path) == :path
end

@testset "PATH rejects NonlinearConstraint" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    mixed = [variable_bound(:y, lower=0.0),
             nonlinear_constraint((y, yl, yld, e, th) -> y[1] - 1.0; label="cap")]
    @test_throws ArgumentError compute_steady_state(spec; constraints=mixed, solver=:path)

    shocks = zeros(10, 1)
    @test_throws ArgumentError solve(spec; method=:perfect_foresight,
        T_periods=10, shock_path=shocks, constraints=mixed, solver=:path)
end

@testset "PATH Extension coverage" begin

    @testset "MCP SS with initial_guess kwarg" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 0.01
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # Provide explicit initial_guess (exercises that branch in PATH ext)
            spec_c = compute_steady_state(spec;
                constraints=[variable_bound(:y, lower=-1.0)],
                solver=:path,
                initial_guess=[0.2])
            @test abs(spec_c.steady_state[1]) < 0.1
        end
    end

    @testset "MCP SS with upper-only bound" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 0.01
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # Upper bound only on PATH
            spec_c = compute_steady_state(spec;
                constraints=[variable_bound(:y, upper=1.0)],
                solver=:path)
            @test spec_c.steady_state[1] <= 1.0 + 1e-4
        end
    end

    @testset "MCP PF single-period (t==1 && t==T_periods)" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 1.0
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            # T_periods=1 triggers the t==1 && t==T_periods branch (line 107)
            shocks = zeros(1, 1)
            shocks[1, 1] = 0.5
            pf = solve(spec; method=:perfect_foresight, T_periods=1,
                       shock_path=shocks,
                       constraints=[variable_bound(:y, lower=-5.0)],
                       solver=:path)
            @test pf isa PerfectForesightPath
            @test size(pf.path) == (1, 1)
        end
    end

    @testset "MCP PF with upper bound" begin
        _suppress_warnings() do
            spec = @dsge begin
                parameters: rho = 0.9, sigma = 1.0
                endogenous: y
                exogenous: eps
                y[t] = rho * y[t-1] + sigma * eps[t]
                steady_state: [0.0]
            end
            spec = compute_steady_state(spec)

            shocks = zeros(10, 1)
            shocks[1, 1] = 3.0

            pf = solve(spec; method=:perfect_foresight, T_periods=10,
                       shock_path=shocks,
                       constraints=[variable_bound(:y, upper=0.5)],
                       solver=:path)
            @test pf isa PerfectForesightPath
            @test pf.converged
            @test all(pf.path[:, 1] .<= 0.5 + 1e-4)
        end
    end
end

end # _path_available

end # Constrained Extensions
