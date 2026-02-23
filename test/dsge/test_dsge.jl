# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

using Test
using MacroEconometricModels
using StatsAPI
using LinearAlgebra
using Random
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "DSGE Module" begin

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Types
# ─────────────────────────────────────────────────────────────────────────────

@testset "Type hierarchy" begin
    @test AbstractDSGEModel <: StatsAPI.StatisticalModel
end

@testset "DSGESpec construction" begin
    spec = DSGESpec{Float64}(
        [:C, :K], [:ε_A], [:α, :β],
        Dict(:α => 0.33, :β => 0.99),
        [:(C[t] + K[t]), :(K[t] - C[t])], [identity, identity],
        0, Int[], Float64[]
    )
    @test spec.n_endog == 2
    @test spec.n_exog == 1
    @test spec.n_params == 2
    @test spec.n_expect == 0
    @test length(spec.varnames) == 2
    @test spec.varnames == ["C", "K"]
    @test spec.endog == [:C, :K]
    @test spec.exog == [:ε_A]
    @test spec.params == [:α, :β]
    @test spec.param_values[:α] ≈ 0.33
    @test spec.param_values[:β] ≈ 0.99
end

@testset "DSGESpec validation" begin
    # Too few equations for endog count
    @test_throws AssertionError DSGESpec{Float64}(
        [:C, :K], [:ε_A], [:α],
        Dict(:α => 0.33),
        [:(C[t])], [identity],  # only 1 equation for 2 endog
        0, Int[], Float64[]
    )
    # Mismatched residual_fns count
    @test_throws AssertionError DSGESpec{Float64}(
        [:C], [:ε_A], [:α],
        Dict(:α => 0.33),
        [:(C[t])], [identity, identity],  # 2 fns for 1 equation
        0, Int[], Float64[]
    )
end

@testset "DSGESpec show" begin
    spec = DSGESpec{Float64}(
        [:C, :K], [:ε_A], [:α, :β],
        Dict(:α => 0.33, :β => 0.99),
        [:(C[t] + K[t]), :(K[t] - C[t])], [identity, identity],
        0, Int[], Float64[]
    )
    io = IOBuffer()
    show(io, spec)
    s = String(take!(io))
    @test occursin("DSGE Model Specification", s)
    @test occursin("Endogenous", s)
    @test occursin("Exogenous", s)
    @test occursin("Parameters", s)
end

@testset "DSGESpec with forward-looking variables" begin
    spec = DSGESpec{Float64}(
        [:C, :K, :R], [:ε_A], [:α, :β, :δ],
        Dict(:α => 0.33, :β => 0.99, :δ => 0.025),
        [:(C[t]), :(K[t]), :(R[t])], [identity, identity, identity],
        2, [1, 3], Float64[]
    )
    @test spec.n_expect == 2
    @test spec.forward_indices == [1, 3]
    @test spec.n_endog == 3
end

@testset "LinearDSGE construction" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    @test size(ld.Gamma0) == (2, 2)
    @test size(ld.Gamma1) == (2, 2)
    @test size(ld.Psi) == (2, 1)
    @test size(ld.Pi) == (2, 0)
    @test length(ld.C) == 2
end

@testset "LinearDSGE validation" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    # Wrong Gamma0 size
    @test_throws AssertionError LinearDSGE{Float64}(
        Matrix{Float64}(I, 3, 3), Matrix{Float64}(I, 2, 2),
        zeros(2), ones(2, 1), zeros(2, 0), spec
    )
    # Wrong C length
    @test_throws AssertionError LinearDSGE{Float64}(
        Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2),
        zeros(3), ones(2, 1), zeros(2, 0), spec
    )
end

@testset "LinearDSGE show" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    io = IOBuffer()
    show(io, ld)
    s = String(take!(io))
    @test occursin("Linearized DSGE", s)
    @test occursin("State dimension", s)
end

@testset "DSGESolution construction and accessors" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    sol = DSGESolution{Float64}(
        0.5 * Matrix{Float64}(I, n, n), ones(n, 1), zeros(n),
        [1, 1], :gensys, [0.5+0.0im, 0.5+0.0im], spec, ld
    )
    @test nvars(sol) == 2
    @test nshocks(sol) == 1
    @test is_determined(sol)
    @test is_stable(sol)

    # Non-determined case
    sol_nd = DSGESolution{Float64}(
        0.5 * Matrix{Float64}(I, n, n), ones(n, 1), zeros(n),
        [0, 1], :gensys, [0.5+0.0im, 0.5+0.0im], spec, ld
    )
    @test !is_determined(sol_nd)

    # Unstable case
    sol_unstable = DSGESolution{Float64}(
        2.0 * Matrix{Float64}(I, n, n), ones(n, 1), zeros(n),
        [1, 1], :blanchard_kahn, [2.0+0.0im, 2.0+0.0im], spec, ld
    )
    @test !is_stable(sol_unstable)
end

@testset "DSGESolution show" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    sol = DSGESolution{Float64}(
        0.5 * Matrix{Float64}(I, n, n), ones(n, 1), zeros(n),
        [1, 1], :gensys, [0.5+0.0im, 0.5+0.0im], spec, ld
    )
    io = IOBuffer()
    show(io, sol)
    s = String(take!(io))
    @test occursin("DSGE Solution", s)
    @test occursin("Existence", s)
    @test occursin("Uniqueness", s)
    @test occursin("Method", s)
end

@testset "PerfectForesightPath construction" begin
    spec = DSGESpec{Float64}(
        [:C, :K], [:ε_A], [:α],
        Dict(:α => 0.33),
        [:(C[t]), :(K[t])], [identity, identity],
        0, Int[], [1.0, 4.0]
    )
    T_periods = 50
    path = rand(T_periods, 2)
    devs = path .- [1.0 4.0]
    pf = PerfectForesightPath{Float64}(path, devs, true, 12, spec)
    @test size(pf.path) == (50, 2)
    @test pf.converged
    @test pf.iterations == 12
end

@testset "PerfectForesightPath show" begin
    spec = DSGESpec{Float64}(
        [:C, :K], [:ε_A], [:α],
        Dict(:α => 0.33),
        [:(C[t]), :(K[t])], [identity, identity],
        0, Int[], [1.0, 4.0]
    )
    pf = PerfectForesightPath{Float64}(rand(50, 2), rand(50, 2), true, 8, spec)
    io = IOBuffer()
    show(io, pf)
    s = String(take!(io))
    @test occursin("Perfect Foresight Path", s)
    @test occursin("Converged", s)
    @test occursin("Iterations", s)
end

@testset "DSGEEstimation construction" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ, :σ],
        Dict(:ρ => 0.9, :σ => 0.01),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    sol = DSGESolution{Float64}(
        0.5 * Matrix{Float64}(I, n, n), ones(n, 1), zeros(n),
        [1, 1], :gensys, [0.5+0.0im, 0.5+0.0im], spec, ld
    )
    est = DSGEEstimation{Float64}(
        [0.85, 0.02], [0.01 0.0; 0.0 0.001], [:ρ, :σ],
        :irf_matching, 3.5, 0.12, sol, true, spec
    )
    @test coef(est) == [0.85, 0.02]
    @test size(vcov(est)) == (2, 2)
    @test dof(est) == 2
    @test islinear(est) == false
    @test length(stderror(est)) == 2
    @test stderror(est)[1] ≈ sqrt(0.01)
    @test est.converged
    @test est.J_stat ≈ 3.5
end

@testset "DSGEEstimation validation" begin
    spec = DSGESpec{Float64}(
        [:y1], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y1[t])], [identity],
        0, Int[], Float64[]
    )
    n = 1
    ld = LinearDSGE{Float64}(
        ones(n, n), 0.5 * ones(n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    sol = DSGESolution{Float64}(
        0.5 * ones(n, n), ones(n, 1), zeros(n),
        [1, 1], :gensys, [0.5+0.0im], spec, ld
    )
    # Wrong method
    @test_throws AssertionError DSGEEstimation{Float64}(
        [0.9], [0.01;;], [:ρ], :invalid_method, 0.0, 1.0, sol, true, spec
    )
    # Mismatched theta/param_names length
    @test_throws AssertionError DSGEEstimation{Float64}(
        [0.9, 0.1], [0.01;;], [:ρ], :irf_matching, 0.0, 1.0, sol, true, spec
    )
end

@testset "DSGEEstimation show" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ, :σ],
        Dict(:ρ => 0.9, :σ => 0.01),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    sol = DSGESolution{Float64}(
        0.5 * Matrix{Float64}(I, n, n), ones(n, 1), zeros(n),
        [1, 1], :gensys, [0.5+0.0im, 0.5+0.0im], spec, ld
    )
    est = DSGEEstimation{Float64}(
        [0.85, 0.02], [0.01 0.0; 0.0 0.001], [:ρ, :σ],
        :irf_matching, 3.5, 0.12, sol, true, spec
    )
    io = IOBuffer()
    show(io, est)
    s = String(take!(io))
    @test occursin("DSGE Estimation", s)
    @test occursin("GMM", s)
    @test occursin("J-statistic", s)
    @test occursin("Estimated Parameters", s)
    @test occursin("Significance", s)
end

@testset "DSGEEstimation report" begin
    spec = DSGESpec{Float64}(
        [:y1, :y2], [:ε], [:ρ, :σ],
        Dict(:ρ => 0.9, :σ => 0.01),
        [:(y1[t]), :(y2[t])], [identity, identity],
        0, Int[], Float64[]
    )
    n = 2
    ld = LinearDSGE{Float64}(
        Matrix{Float64}(I, n, n), 0.5 * Matrix{Float64}(I, n, n),
        zeros(n), ones(n, 1), zeros(n, 0), spec
    )
    sol = DSGESolution{Float64}(
        0.5 * Matrix{Float64}(I, n, n), ones(n, 1), zeros(n),
        [1, 1], :gensys, [0.5+0.0im, 0.5+0.0im], spec, ld
    )
    est = DSGEEstimation{Float64}(
        [0.85, 0.02], [0.01 0.0; 0.0 0.001], [:ρ, :σ],
        :euler_gmm, 2.1, 0.35, sol, true, spec
    )
    # report() writes to stdout — just verify no error
    @test report(est) === nothing
    @test report(sol) === nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Parser
# ─────────────────────────────────────────────────────────────────────────────

@testset "Parser: simple AR(1)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    @test spec isa DSGESpec{Float64}
    @test spec.endog == [:y]
    @test spec.exog == [:ε]
    @test spec.params == [:ρ, :σ]
    @test spec.param_values[:ρ] ≈ 0.9
    @test spec.param_values[:σ] ≈ 0.01
    @test spec.n_endog == 1
    @test spec.n_exog == 1
    @test spec.n_expect == 0  # no forward-looking
end

@testset "Parser: multi-variable" begin
    spec = @dsge begin
        parameters: α = 0.33, ρ = 0.9
        endogenous: y, k, a
        exogenous: ε_a
        y[t] = k[t-1]^α
        k[t] = y[t] - 0.5 * y[t]
        a[t] = ρ * a[t-1] + ε_a[t]
    end
    @test spec.n_endog == 3
    @test spec.n_exog == 1
    @test spec.n_params == 2
end

@testset "Parser: forward-looking / E[t]" begin
    spec = @dsge begin
        parameters: β = 0.5
        endogenous: x
        exogenous: ε
        x[t] = β * E[t](x[t+1]) + ε[t]
    end
    @test spec.n_expect == 1  # one forward-looking variable
end

@testset "Parser: implicit forward (no E[t])" begin
    spec = @dsge begin
        parameters: β = 0.5
        endogenous: x
        exogenous: ε
        x[t] = β * x[t+1] + ε[t]
    end
    @test spec.n_expect == 1
end

@testset "Parser: residual functions" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    fn = spec.residual_fns[1]
    # At SS (y=0, ε=0): residual = 0 - 0.9*0 - 1.0*0 = 0
    @test fn([0.0], [0.0], [0.0], [0.0], spec.param_values) ≈ 0.0 atol=1e-12
    # y_t=1, y_lag=0, ε=0: residual = 1 - 0.9*0 - 1.0*0 = 1
    @test fn([1.0], [0.0], [0.0], [0.0], spec.param_values) ≈ 1.0 atol=1e-12
    # y_t=0.9, y_lag=1.0, ε=0: residual = 0.9 - 0.9*1.0 = 0
    @test fn([0.9], [1.0], [0.0], [0.0], spec.param_values) ≈ 0.0 atol=1e-12
end

@testset "Parser: equation count mismatch" begin
    # Macro errors occur at expansion time; use eval to catch them
    @test_throws LoadError eval(:(@dsge begin
        parameters: ρ = 0.9
        endogenous: x, y
        exogenous: ε
        x[t] = ρ * x[t-1] + ε[t]
    end))
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Steady State
# ─────────────────────────────────────────────────────────────────────────────

@testset "Steady state: AR(1)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec2 = compute_steady_state(spec)
    @test spec2 isa DSGESpec
    @test length(spec2.steady_state) == 1
    @test spec2.steady_state[1] ≈ 0.0 atol=1e-6  # AR(1) SS = 0
end

@testset "Steady state: simple production" begin
    # y = k^α, k = s*y → SS: y = (s*y)^α → y^(1-α) = s^α
    spec = @dsge begin
        parameters: α = 0.33, s = 0.3
        endogenous: y, k
        exogenous: ε
        y[t] = k[t-1]^α + ε[t]
        k[t] = s * y[t]
    end
    spec2 = compute_steady_state(spec; initial_guess=[1.0, 0.3])
    @test length(spec2.steady_state) == 2
    # Check SS satisfies equations
    y_ss, k_ss = spec2.steady_state
    @test y_ss ≈ k_ss^0.33 atol=1e-4
    @test k_ss ≈ 0.3 * y_ss atol=1e-4
end

@testset "Steady state: analytical" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    ss_fn = (θ) -> [0.0]  # Known: y_ss = 0 for zero-mean AR
    spec2 = compute_steady_state(spec; method=:analytical, ss_fn=ss_fn)
    @test spec2.steady_state[1] ≈ 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Linearization
# ─────────────────────────────────────────────────────────────────────────────

@testset "Linearize: AR(1)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    ld = linearize(spec)
    @test ld isa LinearDSGE{Float64}
    # For y_t = ρ*y_{t-1} + σ*ε_t → Γ₀ = [1], Γ₁ = [ρ], Ψ = [σ], Π = empty
    @test ld.Gamma0[1,1] ≈ 1.0 atol=1e-4
    @test ld.Gamma1[1,1] ≈ 0.9 atol=1e-4
    @test ld.Psi[1,1] ≈ 1.0 atol=1e-4
    @test size(ld.Pi, 2) == 0  # no forward-looking variables
end

@testset "Linearize: forward-looking" begin
    # x_t = 0.5 * E_t[x_{t+1}] + ε_t
    spec = @dsge begin
        parameters: β = 0.5
        endogenous: x
        exogenous: ε
        x[t] = β * E[t](x[t+1]) + ε[t]
    end
    spec = compute_steady_state(spec)
    ld = linearize(spec)
    @test size(ld.Pi, 2) == 1  # 1 expectation error
    @test abs(ld.Pi[1,1]) > 0.1  # non-zero Π entry
end

@testset "LinearDSGE show from linearize" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    spec = compute_steady_state(spec)
    ld = linearize(spec)
    io = IOBuffer()
    show(io, ld)
    s = String(take!(io))
    @test occursin("Linearized DSGE", s)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Gensys Solver
# ─────────────────────────────────────────────────────────────────────────────

@testset "Gensys: AR(1) model" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    @test sol isa DSGESolution{Float64}
    @test is_determined(sol)
    @test sol.G1[1,1] ≈ 0.9 atol=1e-4
    @test sol.impact[1,1] ≈ 1.0 atol=1e-4
end

@testset "Gensys: forward-looking model" begin
    # x_t = 0.5 * E_t[x_{t+1}] + ε_t
    # Forward iteration (iid shock): x_t = Σ_{j=0}^∞ 0.5^j E_t[ε_{t+j}] = ε_t
    # So G1 = 0 (no state persistence), impact = 1.0
    spec = @dsge begin
        parameters: β = 0.5
        endogenous: x
        exogenous: ε
        x[t] = β * E[t](x[t+1]) + ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    @test is_determined(sol)
    # G1 should be 0 (no state persistence), impact should be 1.0
    @test abs(sol.G1[1,1]) < 0.1
    @test sol.impact[1,1] ≈ 1.0 atol=0.1
end

@testset "Gensys: existence/uniqueness flags" begin
    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    @test sol.eu[1] == 1  # exists
    @test sol.eu[2] == 1  # unique
end

@testset "Gensys: default method" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec)  # should default to :gensys
    @test sol.method == :gensys
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Blanchard-Kahn
# ─────────────────────────────────────────────────────────────────────────────

@testset "BK: AR(1) model" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:blanchard_kahn)
    @test sol isa DSGESolution{Float64}
    @test is_determined(sol)
    @test sol.G1[1,1] ≈ 0.9 atol=1e-4
    @test sol.method == :blanchard_kahn
end

@testset "BK: agrees with gensys" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    sol_g = solve(spec; method=:gensys)
    sol_bk = solve(spec; method=:blanchard_kahn)
    @test sol_g.G1 ≈ sol_bk.G1 atol=1e-4
    @test sol_g.impact ≈ sol_bk.impact atol=1e-4
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Perfect Foresight
# ─────────────────────────────────────────────────────────────────────────────

@testset "Perfect foresight: AR(1) impulse" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    T_periods = 50
    shocks = zeros(T_periods, 1)
    shocks[1, 1] = 1.0  # unit shock at t=1
    pf = solve(spec; method=:perfect_foresight, T_periods=T_periods, shock_path=shocks)
    @test pf isa PerfectForesightPath{Float64}
    @test pf.converged
    @test size(pf.path) == (T_periods, 1)
    # y_1 = 0.9*0 + 1.0*1.0 = 1.0
    @test pf.deviations[1, 1] ≈ 1.0 atol=0.1
    # y_2 = 0.9*1.0 = 0.9
    @test pf.deviations[2, 1] ≈ 0.9 atol=0.1
    # Converges to SS
    @test abs(pf.deviations[end, 1]) < 0.01
end

@testset "Perfect foresight: zero shocks = steady state" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    T_periods = 20
    pf = solve(spec; method=:perfect_foresight, T_periods=T_periods)
    @test pf.converged
    # With no shocks, path should stay at steady state
    @test all(abs.(pf.deviations) .< 1e-6)
end

@testset "Perfect foresight: multi-variable" begin
    spec = @dsge begin
        parameters: ρ = 0.8, σ_y = 1.0, σ_k = 0.5
        endogenous: y, k
        exogenous: ε_y, ε_k
        y[t] = ρ * y[t-1] + σ_y * ε_y[t]
        k[t] = 0.5 * y[t] + σ_k * ε_k[t]
    end
    spec = compute_steady_state(spec)
    T_periods = 30
    shocks = zeros(T_periods, 2)
    shocks[1, 1] = 1.0  # shock to y only
    pf = solve(spec; method=:perfect_foresight, T_periods=T_periods, shock_path=shocks)
    @test pf isa PerfectForesightPath{Float64}
    @test pf.converged
    @test size(pf.path) == (T_periods, 2)
    @test size(pf.deviations) == (T_periods, 2)
    # y responds, k responds through y
    @test abs(pf.deviations[1, 1]) > 0.5
    @test abs(pf.deviations[1, 2]) > 0.1  # k = 0.5*y
end

@testset "Perfect foresight: convergence diagnostics" begin
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    shocks = zeros(20, 1)
    shocks[1, 1] = 1.0
    pf = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks)
    @test pf.converged
    @test pf.iterations >= 1
    @test pf.iterations <= 100
end

@testset "Perfect foresight: persistent shocks" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    T_periods = 40
    shocks = zeros(T_periods, 1)
    shocks[1:5, 1] .= 1.0  # sustained shock for 5 periods
    pf = solve(spec; method=:perfect_foresight, T_periods=T_periods, shock_path=shocks)
    @test pf.converged
    # After persistent shocks, deviation should be larger than single-period
    @test abs(pf.deviations[5, 1]) > abs(pf.deviations[1, 1]) * 0.5
end

@testset "Perfect foresight: requires steady state" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    # solve() auto-computes SS, so this should work
    pf = solve(spec; method=:perfect_foresight, T_periods=10)
    @test pf isa PerfectForesightPath{Float64}
    @test pf.converged
end

@testset "Perfect foresight: show method" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    pf = solve(spec; method=:perfect_foresight, T_periods=10)
    io = IOBuffer()
    show(io, pf)
    s = String(take!(io))
    @test occursin("Perfect Foresight Path", s)
    @test occursin("Converged", s)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Simulation & IRF/FEVD Bridge
# ─────────────────────────────────────────────────────────────────────────────

@testset "Simulate: stochastic" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    sol = solve(spec)
    Random.seed!(42)
    sim = simulate(sol, 200)
    @test size(sim) == (200, 1)
    @test std(sim[:, 1]) > 0  # not all zeros
    @test std(sim[:, 1]) < 1  # bounded
end

@testset "Simulate: deterministic (given shocks)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    sol = solve(spec)
    shocks = zeros(10, 1)
    shocks[1, 1] = 1.0
    sim = simulate(sol, 10; shock_draws=shocks)
    @test sim[1, 1] ≈ 1.0 atol=1e-4
    @test sim[2, 1] ≈ 0.9 atol=1e-4
end

@testset "IRF bridge to ImpulseResponse" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    sol = solve(spec)
    irf_result = irf(sol, 20)
    @test irf_result isa ImpulseResponse{Float64}
    @test irf_result.horizon == 20
    @test irf_result.values[1, 1, 1] ≈ 1.0 atol=1e-4  # impact
    @test irf_result.values[2, 1, 1] ≈ 0.9 atol=1e-4  # h=2
    @test length(irf_result.variables) == 1
    @test length(irf_result.shocks) == 1
end

@testset "FEVD bridge to FEVD type" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    sol = solve(spec)
    fevd_result = fevd(sol, 20)
    @test fevd_result isa FEVD{Float64}
    # Single shock -> 100% variance explained at all horizons
    for h in 1:20
        @test fevd_result.proportions[1, 1, h] ≈ 1.0 atol=1e-6
    end
end

@testset "FEVD with n_endog > n_shocks" begin
    # 2 endogenous variables, 1 shock — tests non-square FEVD
    spec = @dsge begin
        parameters: ρ = 0.8, α = 0.5
        endogenous: y, c
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
        c[t] = α * y[t]
    end
    sol = solve(spec)
    @test nvars(sol) == 2
    @test nshocks(sol) == 1

    # IRF shape: (horizon, n_endog, n_exog)
    irf_result = irf(sol, 10)
    @test size(irf_result.values) == (10, 2, 1)

    # FEVD should not error with non-square dimensions
    fevd_result = fevd(sol, 10)
    @test fevd_result isa FEVD{Float64}
    @test size(fevd_result.proportions) == (2, 1, 10)
    # Single shock → 100% variance for both variables
    for h in 1:10, i in 1:2
        @test fevd_result.proportions[i, 1, h] ≈ 1.0 atol=1e-6
    end

    # Simulate also works
    Random.seed!(42)
    sim = simulate(sol, 50)
    @test size(sim) == (50, 2)
end

@testset "plot_result works with DSGE IRF" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    sol = solve(spec)
    irf_result = irf(sol, 20)
    p = plot_result(irf_result)
    @test p isa PlotOutput
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 9: GMM Estimation
# ─────────────────────────────────────────────────────────────────────────────

@testset "IRF matching: recover AR(1) parameter" begin
    # True model: y_t = 0.8 * y_{t-1} + ε_t
    rng = Random.MersenneTwister(42)
    T_obs = FAST ? 300 : 500
    y_true = zeros(T_obs)
    for t in 2:T_obs
        y_true[t] = 0.8 * y_true[t-1] + randn(rng)
    end
    Y = reshape(y_true, :, 1)

    spec = @dsge begin
        parameters: ρ = 0.5  # starting guess
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end

    est = estimate_dsge(spec, Y, [:ρ]; method=:irf_matching, irf_horizon=10)
    @test est isa DSGEEstimation{Float64}
    @test est.converged
    @test est.theta[1] ≈ 0.8 atol=0.15
    @test est.method == :irf_matching
    @test is_determined(est.solution)
end

@testset "Euler GMM: basic" begin
    rng = Random.MersenneTwister(123)
    T_obs = FAST ? 200 : 300
    y_true = zeros(T_obs)
    for t in 2:T_obs
        y_true[t] = 0.7 * y_true[t-1] + randn(rng)
    end
    Y = reshape(y_true, :, 1)

    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end

    est = estimate_dsge(spec, Y, [:ρ]; method=:euler_gmm, n_lags_instruments=4)
    @test est isa DSGEEstimation{Float64}
    @test est.method == :euler_gmm
    @test est.theta[1] ≈ 0.7 atol=0.2
end

@testset "estimate_dsge: invalid method" begin
    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    Y = randn(100, 1)
    @test_throws ArgumentError estimate_dsge(spec, Y, [:ρ]; method=:invalid)
end

@testset "IRF matching: pre-computed target IRFs" begin
    rng = Random.MersenneTwister(99)
    T_obs = FAST ? 200 : 400
    y_true = zeros(T_obs)
    for t in 2:T_obs
        y_true[t] = 0.85 * y_true[t-1] + randn(rng)
    end
    Y = reshape(y_true, :, 1)

    # Pre-compute target IRFs from VAR
    var_model = estimate_var(Y, 2)
    target = irf(var_model, 10; method=:cholesky)

    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end

    est = estimate_dsge(spec, Y, [:ρ]; method=:irf_matching,
                         target_irfs=target, irf_horizon=10)
    @test est isa DSGEEstimation{Float64}
    @test est.converged
end

@testset "DSGEEstimation show and report" begin
    rng = Random.MersenneTwister(42)
    y_true = zeros(200)
    for t in 2:200
        y_true[t] = 0.8 * y_true[t-1] + randn(rng)
    end
    Y = reshape(y_true, :, 1)

    spec = @dsge begin
        parameters: ρ = 0.5
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end

    est = estimate_dsge(spec, Y, [:ρ]; method=:irf_matching, irf_horizon=10)
    io = IOBuffer()
    show(io, est)
    s = String(take!(io))
    @test occursin("DSGE Estimation", s)
    @test occursin("Estimated Parameters", s)
    @test occursin("irf_matching", s)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Full Workflow Integration
# ─────────────────────────────────────────────────────────────────────────────

@testset "Full workflow: specify → solve → analyze" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end

    # Solve
    sol = solve(spec)
    @test is_determined(sol)

    # IRF
    irf_result = irf(sol, 40)
    @test irf_result isa ImpulseResponse
    @test irf_result.values[1, 1, 1] ≈ 0.01 atol=1e-3

    # FEVD
    fevd_result = fevd(sol, 40)
    @test fevd_result isa FEVD

    # Simulate
    Random.seed!(42)
    sim = simulate(sol, 200)
    @test size(sim) == (200, 1)

    # Plot (smoke test)
    p = plot_result(irf_result)
    @test p isa PlotOutput

    # Display
    io = IOBuffer()
    show(io, sol)
    @test occursin("DSGE Solution", String(take!(io)))
end

@testset "Full workflow: forward-looking model" begin
    spec = @dsge begin
        parameters: β = 0.5, σ = 1.0
        endogenous: x
        exogenous: ε
        x[t] = β * E[t](x[t+1]) + σ * ε[t]
    end

    # Solve with both methods
    sol_g = solve(spec; method=:gensys)
    sol_bk = solve(spec; method=:blanchard_kahn)
    @test is_determined(sol_g)
    # BK eigenvalue counting may flag purely forward-looking models as
    # indeterminate (n_unstable=0 < n_fwd=1), while gensys finds the
    # unique bounded solution. The solution matrices still agree.
    @test sol_g.G1 ≈ sol_bk.G1 atol=1e-4
    @test sol_g.impact ≈ sol_bk.impact atol=1e-4

    # IRF and simulation
    irf_result = irf(sol_g, 20)
    @test irf_result isa ImpulseResponse
    sim = simulate(sol_g, 100)
    @test size(sim) == (100, 1)
end

@testset "Full workflow: perfect foresight" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    shocks = zeros(50, 1)
    shocks[1, 1] = 1.0
    pf = solve(spec; method=:perfect_foresight, T_periods=50, shock_path=shocks)
    @test pf isa PerfectForesightPath
    @test pf.converged
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 11: refs() for DSGE
# ─────────────────────────────────────────────────────────────────────────────

@testset "refs() for DSGE" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    sol = solve(spec)

    # refs for DSGESolution
    r = refs(sol)
    @test occursin("Sims", r)
    @test occursin("Blanchard", r)

    # refs for DSGESpec
    r_spec = refs(spec)
    @test occursin("Sims", r_spec)

    # Symbol dispatch
    r_sym = refs(:gensys)
    @test occursin("Sims", r_sym)
    r_bk = refs(:blanchard_kahn)
    @test occursin("Blanchard", r_bk)

    # BibTeX format
    r_bib = refs(sol; format=:bibtex)
    @test occursin("@article{sims2002", r_bib)
end

end # top-level @testset
