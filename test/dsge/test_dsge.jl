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

@testset "Parser: steady_state single-line" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    @test spec isa DSGESpec{Float64}
    @test spec.ss_fn !== nothing
    @test spec.ss_fn(spec.param_values) == [0.0]
    @test spec.n_endog == 1
    @test spec.n_expect == 0
end

@testset "Parser: steady_state multi-line begin...end" begin
    spec = @dsge begin
        parameters: α = 0.33, δ = 0.025
        endogenous: y, k
        exogenous: ε
        y[t] = k[t-1]^α + ε[t]
        k[t] = y[t] - δ * k[t-1]
        steady_state = begin
            k_ss = (1.0 / δ)^(1 / (1 - α))
            y_ss = k_ss^α
            [y_ss, k_ss]
        end
    end
    @test spec.ss_fn !== nothing
    ss = spec.ss_fn(spec.param_values)
    @test length(ss) == 2
    @test ss[2] ≈ (1.0 / 0.025)^(1 / (1 - 0.33)) atol=1e-6
end

@testset "Parser: steady_state auto-detected by compute_steady_state" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
        steady_state: [0.0]
    end
    spec2 = compute_steady_state(spec)
    @test spec2.steady_state[1] ≈ 0.0
end

@testset "Parser: no steady_state block → ss_fn is nothing" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    @test spec.ss_fn === nothing
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

@testset "Steady state: ss_fn field on DSGESpec" begin
    # DSGESpec with ss_fn=nothing (backward compat)
    spec = DSGESpec{Float64}(
        [:y], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y[t])], [identity],
        0, Int[], Float64[]
    )
    @test spec.ss_fn === nothing

    # DSGESpec with explicit ss_fn
    my_ss = (θ) -> [0.0]
    spec2 = DSGESpec{Float64}(
        [:y], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y[t])], [identity],
        0, Int[], Float64[], my_ss
    )
    @test spec2.ss_fn === my_ss
    @test spec2.ss_fn(spec2.param_values) == [0.0]
end

@testset "Steady state: auto-detect ss_fn on spec" begin
    spec = DSGESpec{Float64}(
        [:y], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y[t])], [identity],
        0, Int[], Float64[], (θ) -> [0.0]
    )
    spec2 = compute_steady_state(spec)
    @test spec2.steady_state[1] ≈ 0.0
    @test spec2.ss_fn !== nothing  # ss_fn propagated
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

# ─────────────────────────────────────────────────────────────────────────────
# Section 12: SMM Estimation
# ─────────────────────────────────────────────────────────────────────────────

@testset "DSGE SMM Estimation" begin
    # Simple AR(1): y_t = rho * y_{t-1} + sigma * e_t
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: rho = 0.7, sigma = 1.0
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + sigma * e[t]
    end
    spec = compute_steady_state(spec)

    # Simulate data from the model
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 500; rng=rng)

    # Estimate via SMM (only rho, hold sigma fixed)
    # Use bounds to keep rho in stationary region (-1, 1)
    bounds = ParameterTransform([-0.99], [0.99])
    est = estimate_dsge(spec, sim_data, [:rho];
                        method=:smm, sim_ratio=5, burn=100,
                        bounds=bounds,
                        rng=Random.MersenneTwister(123))

    @test est isa DSGEEstimation{Float64}
    @test est.method == :smm
    @test est.converged
    @test abs(est.theta[1] - 0.7) < 0.25  # reasonable recovery
    @test is_determined(est.solution)
end

@testset "DSGE SMM: invalid method still errors" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    Y = randn(100, 1)
    @test_throws ArgumentError estimate_dsge(spec, Y, [:rho]; method=:invalid)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 13: Analytical Moments
# ─────────────────────────────────────────────────────────────────────────────

@testset "solve_lyapunov: 1D AR(1)" begin
    # y_t = rho * y_{t-1} + sigma * e_t => Sigma = sigma^2 / (1 - rho^2)
    rho = 0.9
    sigma = 1.0
    G1 = fill(rho, 1, 1)
    impact = fill(sigma, 1, 1)
    Sigma = solve_lyapunov(G1, impact)
    @test size(Sigma) == (1, 1)
    @test Sigma[1, 1] ≈ sigma^2 / (1 - rho^2) atol=1e-10
end

@testset "solve_lyapunov: 2D VAR(1)" begin
    G1 = [0.8 0.1; 0.0 0.5]
    impact = [1.0 0.0; 0.0 1.0]
    Sigma = solve_lyapunov(G1, impact)
    @test size(Sigma) == (2, 2)
    @test issymmetric(Sigma)
    @test all(eigvals(Sigma) .> 0)
    # Verify Sigma = G1 * Sigma * G1' + impact * impact'
    residual = Sigma - G1 * Sigma * G1' - impact * impact'
    @test norm(residual) < 1e-10
end

@testset "solve_lyapunov: unstable system errors" begin
    G1 = fill(1.5, 1, 1)
    impact = fill(1.0, 1, 1)
    @test_throws ArgumentError solve_lyapunov(G1, impact)
end

@testset "analytical_moments: AR(1) matches theory" begin
    spec = @dsge begin
        parameters: rho = 0.9, sigma = 1.0
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + sigma * e[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)

    m = analytical_moments(sol; lags=1)
    expected_var = 1.0 / (1 - 0.81)
    expected_autocov = 0.9 * expected_var
    @test length(m) == 2
    @test m[1] ≈ expected_var atol=1e-10
    @test m[2] ≈ expected_autocov atol=1e-10
end

@testset "analytical_moments: 2D matches simulation" begin
    spec = @dsge begin
        parameters: rho = 0.8, sigma = 1.0
        endogenous: y, x
        exogenous: e_y, e_x
        y[t] = rho * y[t-1] + sigma * e_y[t]
        x[t] = 0.5 * y[t-1] + 0.5 * x[t-1] + e_x[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)

    m_analytical = analytical_moments(sol; lags=2)
    # k=2, lags=2: k*(k+1)/2 + k*lags = 3 + 4 = 7 moments
    @test length(m_analytical) == 7

    # Cross-check with long simulation
    rng = Random.MersenneTwister(42)
    sim_data = simulate(sol, 100_000; rng=rng)
    m_simulated = autocovariance_moments(sim_data; lags=2)
    for i in eachindex(m_analytical)
        @test m_analytical[i] ≈ m_simulated[i] rtol=0.05
    end
end

@testset "analytical_moments: lags=0" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    m = analytical_moments(sol; lags=0)
    @test length(m) == 1
    @test m[1] ≈ 1.0 / (1 - 0.25) atol=1e-10
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 14: Analytical GMM Estimation
# ─────────────────────────────────────────────────────────────────────────────

@testset "DSGE Analytical GMM Estimation" begin
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: rho = 0.7, sigma = 1.0
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + sigma * e[t]
    end
    spec = compute_steady_state(spec)

    # Simulate data
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 500; rng=rng)

    # Estimate rho via analytical GMM
    bounds = ParameterTransform([-0.99], [0.99])
    est = estimate_dsge(spec, sim_data, [:rho];
                        method=:analytical_gmm,
                        bounds=bounds)

    @test est isa DSGEEstimation{Float64}
    @test est.method == :analytical_gmm
    @test est.converged
    @test abs(est.theta[1] - 0.7) < 0.2  # reasonable recovery
    @test is_determined(est.solution)
end

@testset "DSGE Analytical GMM: lags kwarg" begin
    spec = @dsge begin
        parameters: rho = 0.8
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)
    sim_data = simulate(sol, 300; rng=Random.MersenneTwister(99))

    est = estimate_dsge(spec, sim_data, [:rho];
                        method=:analytical_gmm, lags=2)
    @test est isa DSGEEstimation{Float64}
    @test est.method == :analytical_gmm
end

@testset "DSGE estimate_dsge: invalid method error includes analytical_gmm" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    Y = randn(100, 1)
    @test_throws ArgumentError estimate_dsge(spec, Y, [:rho]; method=:invalid)
end

# Section 15: OccBin Types
@testset "OccBin: types" begin
    # OccBinConstraint
    oc = MacroEconometricModels.OccBinConstraint{Float64}(
        :(R >= 0.0), :R, 0.0, :geq, :(R = 0.0)
    )
    @test oc.variable == :R
    @test oc.bound == 0.0
    @test oc.direction == :geq
    @test oc.expr == :(R >= 0.0)
    @test oc.bind_expr == :(R = 0.0)

    # OccBinRegime
    A = [1.0 0.0; 0.0 1.0]
    B = [0.5 0.1; 0.2 0.6]
    C = [0.3 0.0; 0.0 0.4]
    D = [1.0 0.0; 0.0 1.0]
    regime = MacroEconometricModels.OccBinRegime{Float64}(A, B, C, D)
    @test size(regime.A) == (2, 2)
    @test size(regime.B) == (2, 2)
    @test size(regime.C) == (2, 2)
    @test size(regime.D) == (2, 2)
    @test regime.B[1, 1] == 0.5

    # OccBinSolution
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)

    T_periods = 20
    n_endog = 1
    n_constraints = 1
    linear_path = randn(T_periods, n_endog)
    piecewise_path = randn(T_periods, n_endog)
    ss = zeros(n_endog)
    regime_hist = zeros(Int, T_periods, n_constraints)
    regime_hist[5:8, 1] .= 1  # binding in periods 5-8

    sol = MacroEconometricModels.OccBinSolution{Float64}(
        linear_path, piecewise_path, ss, regime_hist,
        true, 7, spec, ["y"]
    )
    @test sol.converged == true
    @test sol.iterations == 7
    @test size(sol.piecewise_path) == (20, 1)
    @test sum(sol.regime_history .> 0) == 4

    # show method
    io = IOBuffer()
    show(io, sol)
    str = String(take!(io))
    @test occursin("OccBin Piecewise-Linear Solution", str)
    @test occursin("Converged", str)
    @test occursin("Yes", str)

    # report dispatches to show (test that it doesn't error)
    @test report(sol) === nothing

    # OccBinIRF
    H = 20
    linear_irf = randn(H, n_endog)
    pw_irf = randn(H, n_endog)
    regime_irf = zeros(Int, H, n_constraints)
    regime_irf[1:3, 1] .= 1

    oirf = MacroEconometricModels.OccBinIRF{Float64}(
        linear_irf, pw_irf, regime_irf, ["y"], "e"
    )
    @test oirf.shock_name == "e"
    @test size(oirf.piecewise) == (20, 1)
    @test size(oirf.linear) == (20, 1)

    # show method
    io3 = IOBuffer()
    show(io3, oirf)
    str3 = String(take!(io3))
    @test occursin("OccBin IRF Comparison", str3)
    @test occursin("e", str3)
    @test occursin("Binding periods", str3)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 16: OccBin Constraint Parsing & Regime Derivation
# ─────────────────────────────────────────────────────────────────────────────

@testset "OccBin: parse_constraint" begin
    spec = @dsge begin
        parameters: rho = 0.5, phi = 1.5, sigma = 0.01
        endogenous: y, pi_var, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        pi_var[t] = 0.5 * y[t]
        i[t] = phi * pi_var[t]
    end

    c = parse_constraint(:(i[t] >= 0), spec)
    @test c.variable == :i
    @test c.bound == 0.0
    @test c.direction == :geq

    c2 = parse_constraint(:(y[t] <= 1.0), spec)
    @test c2.variable == :y
    @test c2.bound == 1.0
    @test c2.direction == :leq

    @test_throws ArgumentError parse_constraint(:(z[t] >= 0), spec)
end

@testset "OccBin: _derive_alternative_regime" begin
    spec = @dsge begin
        parameters: rho = 0.5, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)

    c = parse_constraint(:(i[t] >= 0), spec)
    alt_spec = MacroEconometricModels._derive_alternative_regime(spec, c)

    @test alt_spec.n_endog == spec.n_endog
    @test alt_spec.endog == spec.endog

    y_ss = alt_spec.steady_state
    theta = alt_spec.param_values
    eps_zero = zeros(spec.n_exog)
    resid = alt_spec.residual_fns[2](y_ss, y_ss, y_ss, eps_zero, theta)
    @test abs(resid) < 1e-10
end

@testset "OccBin: _extract_regime" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)

    regime = MacroEconometricModels._extract_regime(spec)
    @test isa(regime, OccBinRegime{Float64})
    @test size(regime.A) == (1, 1)
    @test size(regime.B) == (1, 1)
    @test size(regime.C) == (1, 1)
    @test size(regime.D) == (1, 1)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 17: OccBin One-Constraint Solver
# ─────────────────────────────────────────────────────────────────────────────

@testset "OccBin: _map_regime" begin
    violvec = BitVector([0, 0, 1, 1, 1, 0, 0, 1, 0, 0])
    regimes, starts = MacroEconometricModels._map_regime(violvec)
    @test regimes == [0, 1, 0, 1, 0]
    @test starts == [1, 3, 6, 8, 9]

    # All zeros
    v0 = falses(5)
    r0, s0 = MacroEconometricModels._map_regime(v0)
    @test r0 == [0]
    @test s0 == [1]

    # All ones
    v1 = trues(4)
    r1, s1 = MacroEconometricModels._map_regime(v1)
    @test r1 == [1]
    @test s1 == [1]

    # Empty
    ve = BitVector([])
    re, se = MacroEconometricModels._map_regime(ve)
    @test isempty(re)
    @test isempty(se)

    # Single element
    vs = BitVector([1])
    rs, ss = MacroEconometricModels._map_regime(vs)
    @test rs == [1]
    @test ss == [1]
end

@testset "OccBin: one-constraint ZLB" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)

    shock_path = zeros(40, spec.n_exog)
    shock_path[1, 1] = -2.0

    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=40)
    @test isa(sol, OccBinSolution{Float64})
    @test sol.converged
    @test size(sol.piecewise_path) == (40, 2)
    @test size(sol.linear_path) == (40, 2)

    # The linear path should violate the ZLB (i goes negative)
    i_idx = 2
    @test minimum(sol.linear_path[:, i_idx]) < 0.0

    # The piecewise path should respect the ZLB (i >= 0, up to numerical tolerance)
    @test minimum(sol.piecewise_path[:, i_idx]) >= -1e-8

    # There should be some binding periods
    @test sum(sol.regime_history[:, 1]) > 0
end

@testset "OccBin: no-binding case" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = 0.5 * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= -100.0), spec)

    shock_path = zeros(20, 1)
    shock_path[1, 1] = 0.01
    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)
    @test sol.converged
    @test sum(sol.regime_history) == 0
    # When no binding, piecewise should equal linear
    @test sol.piecewise_path ≈ sol.linear_path atol=1e-10
end

@testset "OccBin: explicit alt_spec variant" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)
    alt_spec = MacroEconometricModels._derive_alternative_regime(spec, constraint)

    shock_path = zeros(40, spec.n_exog)
    shock_path[1, 1] = -2.0

    sol = occbin_solve(spec, constraint, alt_spec;
                       shock_path=shock_path, nperiods=40)
    @test isa(sol, OccBinSolution{Float64})
    @test sol.converged
end

@testset "OccBin: leq constraint" begin
    spec = @dsge begin
        parameters: rho = 0.9
        endogenous: y, cap
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        cap[t] = y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(cap[t] <= 0.5), spec)

    shock_path = zeros(30, 1)
    shock_path[1, 1] = 2.0  # large positive shock pushes above cap

    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=30)
    @test sol.converged
    cap_idx = 2
    ss_cap = spec.steady_state[cap_idx]
    # Linear path should violate the cap (in levels)
    @test maximum(sol.linear_path[:, cap_idx] .+ ss_cap) > 0.5
    # Piecewise path should respect the cap in levels (up to numerical tolerance)
    @test maximum(sol.piecewise_path[:, cap_idx] .+ ss_cap) <= 0.5 + 1e-4
end

@testset "OccBin: show and report on solution" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = 0.5 * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= -100.0), spec)
    shock_path = zeros(10, 1)
    shock_path[1, 1] = 0.1
    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=10)

    io = IOBuffer()
    show(io, sol)
    str = String(take!(io))
    @test occursin("OccBin Piecewise-Linear Solution", str)
    @test occursin("Converged", str)
    @test report(sol) === nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 18: OccBin Two-Constraint Solver
# ─────────────────────────────────────────────────────────────────────────────

@testset "OccBin: two-constraint" begin
    spec = @dsge begin
        parameters: rho = 0.8, phi = 1.5
        endogenous: y, i, c
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
        c[t] = 0.5 * y[t]
    end
    spec = compute_steady_state(spec)

    c1 = parse_constraint(:(i[t] >= 0), spec)
    c2 = parse_constraint(:(c[t] >= 0), spec)

    shock_path = zeros(30, 1)
    shock_path[1, 1] = -3.0

    sol = occbin_solve(spec, c1, c2; shock_path=shock_path, nperiods=30)
    @test isa(sol, OccBinSolution{Float64})
    @test sol.converged
    @test size(sol.regime_history, 2) == 2

    # Both constraints should bind for some periods
    @test sum(sol.regime_history[:, 1]) > 0
    @test sum(sol.regime_history[:, 2]) > 0

    # Piecewise should respect both constraints
    i_idx = findfirst(==(:i), spec.endog)
    c_idx = findfirst(==(:c), spec.endog)
    @test minimum(sol.piecewise_path[:, i_idx]) >= -1e-8
    @test minimum(sol.piecewise_path[:, c_idx]) >= -1e-8
end

@testset "OccBin: two-constraint no-binding" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, i, c
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = 0.5 * y[t]
        c[t] = 0.3 * y[t]
    end
    spec = compute_steady_state(spec)

    c1 = parse_constraint(:(i[t] >= -100.0), spec)
    c2 = parse_constraint(:(c[t] >= -100.0), spec)

    shock_path = zeros(20, 1)
    shock_path[1, 1] = 0.01

    sol = occbin_solve(spec, c1, c2; shock_path=shock_path, nperiods=20)
    @test sol.converged
    @test sum(sol.regime_history) == 0
    @test sol.piecewise_path ≈ sol.linear_path atol=1e-10
end

@testset "OccBin: two-constraint with explicit alt_specs" begin
    spec = @dsge begin
        parameters: rho = 0.8, phi = 1.5
        endogenous: y, i, c
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
        c[t] = 0.5 * y[t]
    end
    spec = compute_steady_state(spec)

    c1 = parse_constraint(:(i[t] >= 0), spec)
    c2 = parse_constraint(:(c[t] >= 0), spec)

    alt1 = MacroEconometricModels._derive_alternative_regime(spec, c1)
    alt2 = MacroEconometricModels._derive_alternative_regime(spec, c2)
    alt12 = MacroEconometricModels._derive_alternative_regime(alt1, c2)

    alt_specs = Dict((1, 0) => alt1, (0, 1) => alt2, (1, 1) => alt12)

    shock_path = zeros(30, 1)
    shock_path[1, 1] = -3.0

    sol = occbin_solve(spec, c1, c2, alt_specs; shock_path=shock_path, nperiods=30)
    @test isa(sol, OccBinSolution{Float64})
    @test sol.converged
    @test size(sol.regime_history, 2) == 2
end

@testset "OccBin: two-constraint curb_retrench" begin
    spec = @dsge begin
        parameters: rho = 0.8, phi = 1.5
        endogenous: y, i, c
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
        c[t] = 0.5 * y[t]
    end
    spec = compute_steady_state(spec)

    c1 = parse_constraint(:(i[t] >= 0), spec)
    c2 = parse_constraint(:(c[t] >= 0), spec)

    shock_path = zeros(30, 1)
    shock_path[1, 1] = -3.0

    sol = occbin_solve(spec, c1, c2; shock_path=shock_path, nperiods=30,
                       curb_retrench=true)
    @test isa(sol, OccBinSolution{Float64})
    @test sol.converged
end

@testset "OccBin: _find_last_binding_two" begin
    violvec = falses(10, 2)
    @test MacroEconometricModels._find_last_binding_two(violvec) == 0

    violvec[3, 1] = true
    @test MacroEconometricModels._find_last_binding_two(violvec) == 3

    violvec[7, 2] = true
    @test MacroEconometricModels._find_last_binding_two(violvec) == 7

    violvec[10, 1] = true
    @test MacroEconometricModels._find_last_binding_two(violvec) == 10
end

@testset "OccBin: two-constraint show/report" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, i, c
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = 0.5 * y[t]
        c[t] = 0.3 * y[t]
    end
    spec = compute_steady_state(spec)

    c1 = parse_constraint(:(i[t] >= -100.0), spec)
    c2 = parse_constraint(:(c[t] >= -100.0), spec)

    shock_path = zeros(10, 1)
    shock_path[1, 1] = 0.1
    sol = occbin_solve(spec, c1, c2; shock_path=shock_path, nperiods=10)

    io = IOBuffer()
    show(io, sol)
    str = String(take!(io))
    @test occursin("OccBin Piecewise-Linear Solution", str)
    @test occursin("Converged", str)
    @test report(sol) === nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 19: OccBin refs()
# ─────────────────────────────────────────────────────────────────────────────

@testset "OccBin: refs()" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)
    shock_path = zeros(20, 1)
    shock_path[1, 1] = -1.0
    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)

    r = refs(sol)
    @test occursin("Guerrieri", r)
    @test occursin("Iacoviello", r)
    @test occursin("2015", r)

    # Symbol dispatch
    r2 = refs(:occbin)
    @test occursin("Guerrieri", r2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 20: OccBin IRF
# ─────────────────────────────────────────────────────────────────────────────

@testset "OccBin: occbin_irf one-constraint" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)

    # Large negative shock triggers ZLB
    oirf = occbin_irf(spec, constraint, 1, 30; magnitude=-2.0)
    @test isa(oirf, OccBinIRF{Float64})
    @test size(oirf.linear, 1) == 30
    @test size(oirf.piecewise, 1) == 30
    @test oirf.shock_name == "e"

    # Linear should go negative for i, piecewise should be clamped
    i_idx = 2
    @test minimum(oirf.linear[:, i_idx]) < 0.0
    @test minimum(oirf.piecewise[:, i_idx]) >= -1e-8

    # Some binding periods
    @test sum(oirf.regime_history) > 0
end

@testset "OccBin: occbin_irf no-binding" begin
    spec = @dsge begin
        parameters: rho = 0.5, phi = 1.0
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= -100.0), spec)

    oirf = occbin_irf(spec, constraint, 1, 20; magnitude=0.5)
    @test oirf.linear ≈ oirf.piecewise atol=1e-10
    @test sum(oirf.regime_history) == 0
end

@testset "OccBin: occbin_irf two-constraint" begin
    spec = @dsge begin
        parameters: rho = 0.8, phi = 1.5
        endogenous: y, i, c
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
        c[t] = 0.5 * y[t]
    end
    spec = compute_steady_state(spec)
    c1 = parse_constraint(:(i[t] >= 0), spec)
    c2 = parse_constraint(:(c[t] >= 0), spec)

    oirf = occbin_irf(spec, c1, c2, 1, 30; magnitude=-3.0)
    @test isa(oirf, OccBinIRF{Float64})
    @test size(oirf.regime_history, 2) == 2
    @test oirf.shock_name == "e"
end

@testset "OccBin: occbin_irf invalid shock_idx" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(y[t] >= -10.0), spec)

    @test_throws ArgumentError occbin_irf(spec, constraint, 0, 20)
    @test_throws ArgumentError occbin_irf(spec, constraint, 2, 20)
end

@testset "OccBin: occbin_irf show" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)
    oirf = occbin_irf(spec, constraint, 1, 20; magnitude=-2.0)

    io = IOBuffer()
    show(io, oirf)
    output = String(take!(io))
    @test occursin("OccBin IRF", output)
    @test occursin("e", output)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 21: OccBin Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

@testset "OccBin: explicit alternative spec" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)

    # Explicit alternative: i[t] = 0 (ZLB)
    alt_spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = 0.0
    end
    alt_spec = compute_steady_state(alt_spec)

    constraint = parse_constraint(:(i[t] >= 0), spec)

    shock_path = zeros(30, 1)
    shock_path[1, 1] = -2.0
    sol = occbin_solve(spec, constraint, alt_spec; shock_path=shock_path, nperiods=30)
    @test sol.converged
    @test minimum(sol.piecewise_path[:, 2]) >= -1e-8
end

@testset "OccBin: <= constraint direction" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, debt
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        debt[t] = 0.8 * y[t]
    end
    spec = compute_steady_state(spec)

    constraint = parse_constraint(:(debt[t] <= 1.0), spec)
    @test constraint.direction == :leq
    @test constraint.bound == 1.0

    shock_path = zeros(20, 1)
    shock_path[1, 1] = 3.0  # positive shock pushes debt above bound

    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)
    @test sol.converged
    debt_idx = 2
    @test maximum(sol.piecewise_path[:, debt_idx]) <= 1.0 + 1e-8
end

@testset "OccBin: maxiter warning" begin
    spec = @dsge begin
        parameters: rho = 0.99
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = 1.5 * y[t]
    end
    spec = compute_steady_state(spec)

    constraint = parse_constraint(:(i[t] >= 0), spec)
    shock_path = zeros(5, 1)
    shock_path[1, 1] = -5.0

    # With maxiter=1, should warn about not converging
    sol = @test_warn r"did not converge" occbin_solve(spec, constraint;
        shock_path=shock_path, nperiods=5, maxiter=1)
    @test !sol.converged
end

@testset "OccBin: show/report for OccBinSolution" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)
    shock_path = zeros(20, 1)
    shock_path[1, 1] = -2.0
    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)

    io = IOBuffer()
    show(io, sol)
    output = String(take!(io))
    @test occursin("OccBin", output)
    @test occursin("Converged", output)
    @test occursin("Yes", output)

    # report should not error
    report(sol)
end

@testset "OccBin: constraint parsing edge cases" begin
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y, z
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        z[t] = y[t]
    end

    # Invalid variable
    @test_throws ArgumentError parse_constraint(:(w[t] >= 0), spec)

    # Invalid operator
    @test_throws ArgumentError parse_constraint(:(y[t] > 0), spec)

    # Missing time index
    @test_throws ArgumentError parse_constraint(:(y >= 0), spec)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 22: OccBin Plotting
# ─────────────────────────────────────────────────────────────────────────────

@testset "OccBin: plot_result OccBinIRF" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)
    oirf = occbin_irf(spec, constraint, 1, 30; magnitude=-2.0)

    p = plot_result(oirf)
    @test isa(p, PlotOutput)
    @test occursin("OccBin", p.html)
    @test occursin("d3", p.html)
end

@testset "OccBin: plot_result OccBinSolution" begin
    spec = @dsge begin
        parameters: rho = 0.9, phi = 1.5
        endogenous: y, i
        exogenous: e
        y[t] = rho * y[t-1] + e[t]
        i[t] = phi * y[t]
    end
    spec = compute_steady_state(spec)
    constraint = parse_constraint(:(i[t] >= 0), spec)
    shock_path = zeros(30, 1)
    shock_path[1, 1] = -2.0
    sol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=30)

    p = plot_result(sol)
    @test isa(p, PlotOutput)
    @test occursin("OccBin", p.html)
end

end # top-level @testset
