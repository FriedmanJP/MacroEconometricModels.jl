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
    # report() writes to stdout — redirect to devnull, just verify no error
    @test redirect_stdout(devnull) do; report(est) end === nothing
    @test redirect_stdout(devnull) do; report(sol) end === nothing
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
    @test redirect_stdout(devnull) do; report(sol) end === nothing

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
    @test redirect_stdout(devnull) do; report(sol) end === nothing
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
    @test redirect_stdout(devnull) do; report(sol) end === nothing
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
    @test redirect_stdout(devnull) do; report(sol) end === nothing
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

@testset "DSGE Display (#57)" begin
    spec = @dsge begin
        parameters: β = 0.99, σ = 1.0, κ = 0.3, φ_π = 1.5, φ_y = 0.5, ρ = 0.8
        endogenous: y, π, R, d
        exogenous: ε_d

        y[t] = y[t+1] - (1 / σ) * (R[t] - π[t+1]) + d[t]
        π[t] = β * π[t+1] + κ * y[t]
        R[t] = φ_π * π[t] + φ_y * y[t]
        d[t] = ρ * d[t-1] + σ * ε_d[t]
    end

    # ── _sym_to_latex ──
    @testset "_sym_to_latex" begin
        @test MacroEconometricModels._sym_to_latex(:β) == "\\beta"
        @test MacroEconometricModels._sym_to_latex(:σ) == "\\sigma"
        @test MacroEconometricModels._sym_to_latex(:φ_π) == "\\phi_{\\pi}"
        @test MacroEconometricModels._sym_to_latex(:σ_c) == "\\sigma_{c}"
        @test MacroEconometricModels._sym_to_latex(:A) == "A"
        @test MacroEconometricModels._sym_to_latex(:ρ_d) == "\\rho_{d}"
    end

    # ── _format_num_display ──
    @testset "_format_num_display" begin
        @test MacroEconometricModels._format_num_display(1) == "1"
        @test MacroEconometricModels._format_num_display(1.0) == "1"
        @test MacroEconometricModels._format_num_display(0.99) == "0.99"
        @test MacroEconometricModels._format_num_display(0.3) == "0.3"
    end

    # ── _expr_to_text ──
    @testset "_expr_to_text" begin
        endog = [:y, :π, :R, :d]
        exog = [:ε_d]
        params = [:β, :σ, :κ, :φ_π, :φ_y, :ρ]

        # Simple ref
        @test MacroEconometricModels._expr_to_text(:(y[t]), endog, exog, params) == "y_t"
        @test occursin("t-1", MacroEconometricModels._expr_to_text(:(y[t-1]), endog, exog, params))

        # Forward-looking → E_t
        txt = MacroEconometricModels._expr_to_text(:(y[t+1]), endog, exog, params)
        @test occursin("E_t", txt)
        @test occursin("t+1", txt)

        # Shock with subscript
        txt_shock = MacroEconometricModels._expr_to_text(:(ε_d[t]), endog, exog, params)
        @test occursin("ε", txt_shock)
        @test occursin("d", txt_shock)

        # Full equation (RHS only — = sign rendered by _equation_to_display)
        eq_text = MacroEconometricModels._expr_to_text(spec.equations[2], endog, exog, params)
        @test occursin("β", eq_text) || occursin("π", eq_text)
        @test occursin("κ", eq_text)
    end

    # ── _expr_to_latex ──
    @testset "_expr_to_latex" begin
        endog = [:y, :π, :R, :d]
        exog = [:ε_d]
        params = [:β, :σ, :κ, :φ_π, :φ_y, :ρ]

        @test MacroEconometricModels._expr_to_latex(:(y[t]), endog, exog, params) == "y_{t}"

        # Forward-looking → \mathbb{E}_t
        latex = MacroEconometricModels._expr_to_latex(:(y[t+1]), endog, exog, params)
        @test occursin("\\mathbb{E}_t", latex)

        # Greek parameter
        @test MacroEconometricModels._expr_to_latex(:β, endog, exog, params) == "\\beta"

        # Division → \frac
        latex_div = MacroEconometricModels._expr_to_latex(:(1 / σ), endog, exog, params)
        @test occursin("\\frac", latex_div)

        # Subscripted shock
        eq_latex = MacroEconometricModels._expr_to_latex(spec.equations[4], endog, exog, params)
        @test occursin("\\rho", eq_latex)
        @test occursin("\\varepsilon", eq_latex)
    end

    # ── show() text backend ──
    @testset "show text backend" begin
        set_display_backend(:text)
        io = IOBuffer()
        show(io, spec)
        output = String(take!(io))

        @test occursin("DSGE Model", output)
        @test occursin("4", output)  # 4 equations
        @test occursin("Calibration", output)
        @test occursin("β", output)
        @test occursin("Equations", output)
        @test occursin("(1)", output)
        @test occursin("(4)", output)
        @test occursin("E_t", output)

        # report() should not error
        @test redirect_stdout(devnull) do; report(spec) end === nothing
    end

    # ── show() LaTeX backend ──
    @testset "show latex backend" begin
        set_display_backend(:latex)
        io = IOBuffer()
        show(io, spec)
        output = String(take!(io))

        @test occursin("\\begin{align}", output)
        @test occursin("\\end{align}", output)
        @test occursin("\\mathbb{E}_t", output)
        @test occursin("\\beta", output)
        @test occursin("\\frac", output)
        @test occursin("\\begin{tabular}", output)

        set_display_backend(:text)  # restore
    end

    # ── show() HTML backend ──
    @testset "show html backend" begin
        set_display_backend(:html)
        io = IOBuffer()
        show(io, spec)
        output = String(take!(io))

        @test occursin("<div", output)
        @test occursin("\\begin{align}", output)
        @test occursin("<table>", output)
        @test occursin("\\mathbb{E}_t", output)

        set_display_backend(:text)  # restore
    end

    # ── Spec without steady state ──
    @testset "show without steady state" begin
        spec_noss = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        io = IOBuffer()
        show(io, spec_noss)
        output = String(take!(io))
        @test occursin("1", output)  # 1 equation
        @test !occursin("Steady State", output)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 23: Augmentation — Deep Lags, Deep Leads, News Shocks (#54)
# ─────────────────────────────────────────────────────────────────────────────

@testset "Augmentation: @dsge with news shock (#54)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ_0 = 0.01, σ_3 = 0.007
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ_0 * ε[t] + σ_3 * ε[t-3]
    end
    @test spec.augmented == true
    @test spec.n_original_endog == 1
    @test spec.original_endog == [:y]
    @test spec.n_endog == 4  # y + 3 news auxiliaries
    @test spec.max_lag == 3
    @test length(spec.original_equations) == 1
    @test length(spec.equations) == 4  # 1 user + 3 identity
end

@testset "Augmentation: @dsge with deep endogenous lag (#54)" begin
    spec = @dsge begin
        parameters: a1 = 0.5, a2 = 0.3, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = a1 * y[t-1] + a2 * y[t-2] + σ * ε[t]
    end
    @test spec.augmented == true
    @test spec.n_original_endog == 1
    @test spec.n_endog == 2  # y + 1 lag auxiliary
    @test spec.max_lag == 2
end

@testset "Augmentation: backward compat — no augmentation needed (#54)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    @test spec.augmented == false
    @test spec.n_original_endog == spec.n_endog
    @test spec.original_endog == spec.endog
    @test spec.max_lag == 1
    @test spec.max_lead == 1
end

@testset "Augmentation: @dsge with deep lead (#54)" begin
    spec = @dsge begin
        parameters: a = 0.3, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = a * y[t+2] + σ * ε[t]
    end
    @test spec.augmented == true
    @test spec.n_original_endog == 1
    @test spec.n_endog == 2  # y + 1 fwd auxiliary
    @test spec.max_lead == 2
end

@testset "Augmentation: solve and simulate AR(2) (#54)" begin
    spec = @dsge begin
        parameters: a1 = 0.5, a2 = 0.3, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = a1 * y[t-1] + a2 * y[t-2] + σ * ε[t]
    end
    spec = compute_steady_state(spec; initial_guess=zeros(spec.n_endog))
    sol = solve(spec; method=:gensys)

    @test is_determined(sol)

    # IRF should only show original variable 'y'
    ir = irf(sol, 20)
    @test length(ir.variables) == 1
    @test ir.variables == ["y"]
    @test size(ir.values, 2) == 1  # only 1 variable

    # FEVD should only show original variable
    fv = fevd(sol, 20)
    @test length(fv.variables) == 1
    @test fv.variables == ["y"]

    # Simulate should return only original variable
    sim = simulate(sol, 100; shock_draws=zeros(100, 1))
    @test size(sim, 2) == 1  # only 'y' column
end

@testset "Augmentation: news shock IRF timing (#54)" begin
    # Pure news shock model: y[t] = σ_0 * ε[t] + σ_3 * ε[t-3]
    # No persistence (no y[t-1] term), so shock effects are isolated
    spec = @dsge begin
        parameters: σ_0 = 1.0, σ_3 = 0.5
        endogenous: y
        exogenous: ε
        y[t] = σ_0 * ε[t] + σ_3 * ε[t-3]
    end
    spec = compute_steady_state(spec; initial_guess=zeros(spec.n_endog))
    sol = solve(spec; method=:gensys)

    @test is_determined(sol)

    ir = irf(sol, 10)
    @test length(ir.variables) == 1  # only 'y'
    @test ir.variables == ["y"]

    # At h=1: immediate impact σ_0 = 1.0
    @test abs(ir.values[1, 1, 1] - 1.0) < 0.05
    # At h=4: delayed news impact σ_3 = 0.5
    # ε drawn at t=1 feeds __news_ε_3 at t=4, which enters y via the lag chain
    @test abs(ir.values[4, 1, 1] - 0.5) < 0.05
end

@testset "Augmentation: display shows original equations (#54)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ_0 = 0.01, σ_3 = 0.007
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ_0 * ε[t] + σ_3 * ε[t-3]
    end

    # Text display
    io = IOBuffer()
    show(io, spec)
    output = String(take!(io))

    # Should show original equation count (1), not augmented (4)
    @test occursin("Equations:             1", output)
    # Should show original variable (y), not auxiliaries
    @test occursin("(y)", output)
    @test !occursin("__news", output)
    # Should show augmented state dim
    @test occursin("Augmented state dim", output)

    # LaTeX display
    set_display_backend(:latex)
    io2 = IOBuffer()
    show(io2, spec)
    latex_output = String(take!(io2))
    @test occursin("Augmented state dimension", latex_output)
    @test !occursin("__news", latex_output)

    # HTML display
    set_display_backend(:html)
    io3 = IOBuffer()
    show(io3, spec)
    html_output = String(take!(io3))
    @test occursin("Augmented state dim", html_output)
    @test !occursin("__news", html_output)

    # Reset
    set_display_backend(:text)

    # Non-augmented model should NOT show augmented state dim
    spec2 = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    io4 = IOBuffer()
    show(io4, spec2)
    output2 = String(take!(io4))
    @test !occursin("Augmented state dim", output2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section: Full Integration Tests for News Shocks and Augmentation (#54)
# ─────────────────────────────────────────────────────────────────────────────

@testset "News Shocks: Full Pipeline (#54)" begin
    # Beaudry-Portier style: technology with news component
    spec = @dsge begin
        parameters: ρ = 0.9, σ_0 = 0.01, σ_4 = 0.007
        endogenous: A, Y
        exogenous: ε_A
        A[t] = ρ * A[t-1] + σ_0 * ε_A[t] + σ_4 * ε_A[t-4]
        Y[t] = A[t]
        steady_state = begin
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # A, Y + 4 news aux
        end
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)

    @test is_determined(sol)
    @test sol.spec.augmented
    @test sol.spec.n_original_endog == 2

    # IRF: only A and Y shown
    ir = irf(sol, 20)
    @test length(ir.variables) == 2
    @test ir.variables == ["A", "Y"]

    # News shock timing: impact at h=1 from σ_0
    @test abs(ir.values[1, 1, 1]) > 0  # immediate impact on A

    # FEVD
    fv = fevd(sol, 20)
    @test length(fv.variables) == 2

    # Simulate
    sim = simulate(sol, 50; shock_draws=zeros(50, 1))
    @test size(sim, 2) == 2

    # Display
    io = IOBuffer()
    show(io, spec)
    output = String(take!(io))
    @test occursin("Augmented state dim", output)
    @test !occursin("__news", output)
end

@testset "Higher-Order Lead: y[t+2] (#54)" begin
    spec = @dsge begin
        parameters: a = 0.3, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = a * y[t+2] + σ * ε[t]
        steady_state = [0.0, 0.0]  # y + 1 fwd auxiliary
    end
    spec = compute_steady_state(spec)
    @test spec.augmented
    @test spec.max_lead == 2
    @test spec.n_endog == 2

    # y[t] = a*y[t+2] may be indeterminate — test what gensys reports
    sol = solve(spec; method=:gensys)

    # Whether determined or not, IRF filtering should work
    ir = irf(sol, 10)
    @test length(ir.variables) == 1
    @test ir.variables == ["y"]
end

@testset "Mixed: deep lag + news shock (#54)" begin
    spec = @dsge begin
        parameters: a1 = 0.4, a2 = 0.2, σ_0 = 1.0, σ_2 = 0.5
        endogenous: y
        exogenous: ε
        y[t] = a1 * y[t-1] + a2 * y[t-2] + σ_0 * ε[t] + σ_2 * ε[t-2]
        steady_state = begin
            zeros(4)  # y + 1 lag aux + 2 news aux
        end
    end
    spec = compute_steady_state(spec)
    @test spec.augmented
    @test spec.n_original_endog == 1
    @test spec.n_endog == 4  # y + __lag_y_1 + __news_ε_1 + __news_ε_2

    sol = solve(spec; method=:gensys)
    @test is_determined(sol)

    ir = irf(sol, 20)
    @test length(ir.variables) == 1
end

@testset "Augmentation: analytical_moments with news (#54)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ_0 = 0.01, σ_2 = 0.005
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ_0 * ε[t] + σ_2 * ε[t-2]
        steady_state = [0.0, 0.0, 0.0]  # y + 2 news aux
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:gensys)

    @test is_determined(sol)

    # analytical_moments should work on augmented system (uses Lyapunov)
    # Returns Vector{T} — length = k*(k+1)/2 + k*lags where k = n_endog (augmented)
    moments = analytical_moments(sol)
    k = sol.spec.n_endog  # 3 (augmented)
    expected_len = div(k * (k + 1), 2) + k  # upper-tri variance + 1 lag autocov
    @test length(moments) == expected_len
    @test all(isfinite, moments)
end

@testset "Augmentation: metadata correctness (#54)" begin
    # No augmentation needed
    spec1 = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    @test !spec1.augmented
    @test spec1.n_original_endog == spec1.n_endog
    @test spec1.original_endog == spec1.endog
    @test spec1.max_lag == 1
    @test spec1.max_lead == 1

    # Deep lag only
    spec2 = @dsge begin
        parameters: a1 = 0.5, a2 = 0.3, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = a1 * y[t-1] + a2 * y[t-2] + σ * ε[t]
    end
    @test spec2.augmented
    @test spec2.max_lag == 2
    @test spec2.max_lead == 1
    @test spec2.n_original_endog == 1
    @test spec2.n_endog == 2

    # News shock only
    spec3 = @dsge begin
        parameters: σ_0 = 0.01, σ_3 = 0.007
        endogenous: y
        exogenous: ε
        y[t] = σ_0 * ε[t] + σ_3 * ε[t-3]
        steady_state = [0.0, 0.0, 0.0, 0.0]
    end
    @test spec3.augmented
    @test spec3.max_lag == 3
    @test spec3.n_original_endog == 1
    @test spec3.n_endog == 4  # y + 3 news aux

    # Deep lead only
    spec4 = @dsge begin
        parameters: a = 0.3, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = a * y[t+2] + σ * ε[t]
    end
    @test spec4.augmented
    @test spec4.max_lead == 2
    @test spec4.n_endog == 2  # y + 1 fwd aux
end

# ─────────────────────────────────────────────────────────────────────────────
# Section: Klein (2000) Solver (#49)
# ─────────────────────────────────────────────────────────────────────────────

@testset "Klein (2000) Solver (#49)" begin
    @testset "Predetermined variable detection" begin
        # AR(1): y[t] = ρ*y[t-1] + σ*ε[t] — 1 predetermined
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        ld = linearize(spec)
        @test MacroEconometricModels._count_predetermined(ld) == 1

        # Purely forward-looking: x[t] = β*E[t](x[t+1]) + ε[t] — 0 predetermined
        spec2 = @dsge begin
            parameters: β = 0.5, σ = 1.0
            endogenous: x
            exogenous: ε
            x[t] = β * x[t+1] + σ * ε[t]
        end
        spec2 = compute_steady_state(spec2)
        ld2 = linearize(spec2)
        @test MacroEconometricModels._count_predetermined(ld2) == 0

        # NK model: 2 equations, 1 predetermined (y[t-1])
        spec3 = @dsge begin
            parameters: β = 0.99, κ = 0.5, φ_π = 1.5, ρ = 0.8, σ = 0.01
            endogenous: π, y
            exogenous: ε
            π[t] = β * π[t+1] + κ * y[t] + σ * ε[t]
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec3 = compute_steady_state(spec3)
        ld3 = linearize(spec3)
        @test MacroEconometricModels._count_predetermined(ld3) == 1
    end

    @testset "Equivalence with gensys — AR(1)" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        sol_g = solve(spec; method=:gensys)
        sol_k = solve(spec; method=:klein)

        @test sol_k.method == :klein
        @test is_determined(sol_k)
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-8
        @test sol_k.impact ≈ sol_g.impact atol=1e-8
        @test sol_k.C_sol ≈ sol_g.C_sol atol=1e-8
    end

    @testset "Equivalence with gensys — forward-looking" begin
        spec = @dsge begin
            parameters: β = 0.5, σ = 1.0
            endogenous: x
            exogenous: ε
            x[t] = β * x[t+1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        sol_g = solve(spec; method=:gensys)
        sol_k = solve(spec; method=:klein)

        # Klein reports eu=[1,0] for purely forward-looking models (n_stable > n_predetermined=0)
        # because BK counting flags indeterminacy, but gensys resolves it via Pi.
        # The solution matrices still match.
        @test sol_k.eu[1] == 1  # existence
        @test !is_determined(sol_k)  # Klein BK counting flags indeterminacy
        @test is_determined(sol_g)   # gensys resolves it via Pi rank check
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-8
        @test sol_k.impact ≈ sol_g.impact atol=1e-8
    end

    @testset "Equivalence with gensys — NK 3-equation" begin
        spec = @dsge begin
            parameters: β = 0.99, κ = 0.3, φ_π = 1.5, φ_y = 0.125, ρ_v = 0.5, σ_v = 0.25
            endogenous: π, y, i
            exogenous: ε_v
            π[t] = β * π[t+1] + κ * y[t]
            y[t] = y[t+1] - (i[t] - π[t+1]) + σ_v * ε_v[t]
            i[t] = φ_π * π[t] + φ_y * y[t] + ρ_v * ε_v[t]
            steady_state = [0.0, 0.0, 0.0]
        end
        spec = compute_steady_state(spec)

        sol_g = solve(spec; method=:gensys)
        sol_k = solve(spec; method=:klein)

        # Purely forward-looking NK model: Klein BK counting differs from gensys
        # (n_stable > n_predetermined=0), but solution matrices match.
        @test sol_k.eu[1] == 1  # existence
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-6
        @test sol_k.impact ≈ sol_g.impact atol=1e-6
    end

    @testset "BK condition — eu flags" begin
        # Determined model
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:klein)
        @test sol.eu == [1, 1]

        # Explosive model: ρ > 1 with no forward-looking vars
        spec2 = @dsge begin
            parameters: ρ = 1.5, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec2 = compute_steady_state(spec2)
        sol2 = solve(spec2; method=:klein)
        @test sol2.eu[1] == 0  # no stable solution
    end

    @testset "Downstream: simulate, irf, fevd" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:klein)

        # Simulate
        sim = simulate(sol, 100; shock_draws=zeros(100, 1))
        @test size(sim) == (100, 1)
        @test all(abs.(sim) .< 1e-10)  # zero shocks → stays at SS (≈ 0)

        # IRF
        ir = irf(sol, 20)
        @test length(ir.variables) == 1
        @test ir.variables == ["y"]
        @test ir.values[1, 1, 1] ≈ 1.0 atol=0.01  # σ=1 impact

        # FEVD
        fv = fevd(sol, 20)
        @test length(fv.variables) == 1
        @test all(fv.proportions[:, 1, :] .≈ 1.0)  # single shock = 100%
    end

    @testset "Augmented model compatibility (#54)" begin
        spec = @dsge begin
            parameters: a1 = 0.5, a2 = 0.3, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = a1 * y[t-1] + a2 * y[t-2] + σ * ε[t]
            steady_state = [0.0, 0.0]
        end
        spec = compute_steady_state(spec)
        @test spec.augmented

        sol_g = solve(spec; method=:gensys)
        sol_k = solve(spec; method=:klein)

        @test is_determined(sol_k)
        @test sol_k.G1 ≈ sol_g.G1 atol=1e-8
        @test sol_k.impact ≈ sol_g.impact atol=1e-8

        # IRF should show only original variable
        ir = irf(sol_k, 20)
        @test length(ir.variables) == 1
        @test ir.variables == ["y"]
    end

    @testset "Display shows :klein method" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:klein)

        io = IOBuffer()
        show(io, sol)
        output = String(take!(io))
        @test occursin("klein", output)
    end
end

@testset "Higher-Order Perturbation (#48)" begin
    @testset "Order 1 equivalence with gensys" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 1.0
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        sol_g = solve(spec; method=:gensys)
        sol_p = solve(spec; method=:perturbation, order=1)

        @test sol_p isa MacroEconometricModels.PerturbationSolution
        @test sol_p.order == 1
        @test sol_p.method == :perturbation
        @test is_determined(sol_p)
        # First-order coefficients should match gensys
        # gensys G1[1,1] should equal hx[1,1] (state transition)
        @test sol_p.hx[1,1] ≈ sol_g.G1[1,1] atol=1e-6
    end

    @testset "Second-order AR(1)" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        sol2 = solve(spec; method=:perturbation, order=2)

        @test sol2.order == 2
        @test is_determined(sol2)
        @test sol2.gxx !== nothing || sol2.hxx !== nothing
        @test sol2.hσσ !== nothing
        # For linear AR(1), second-order terms should be ~zero
        if sol2.hxx !== nothing
            @test maximum(abs.(sol2.hxx)) < 0.01
        end
    end

    @testset "State/control partition" begin
        spec = @dsge begin
            parameters: β = 0.99, κ = 0.3, φ_π = 1.5, φ_y = 0.125, ρ_v = 0.5, σ_v = 0.25
            endogenous: π, y, i
            exogenous: ε_v
            π[t] = β * π[t+1] + κ * y[t]
            y[t] = y[t+1] - (i[t] - π[t+1]) + σ_v * ε_v[t]
            i[t] = φ_π * π[t] + φ_y * y[t] + ρ_v * ε_v[t]
            steady_state = [0.0, 0.0, 0.0]
        end
        spec = compute_steady_state(spec)
        ld = linearize(spec)
        s_idx, c_idx = MacroEconometricModels._state_control_indices(ld)
        # NK model: no predetermined variables (all forward-looking)
        @test length(s_idx) == 0
        @test length(c_idx) == 3
    end

    @testset "Display" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:perturbation, order=2)
        io = IOBuffer()
        show(io, sol)
        output = String(take!(io))
        @test occursin("Perturbation", output) || occursin("perturbation", output)
        @test occursin("2", output)  # order 2
    end

    @testset "Pruned simulation" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:perturbation, order=2)

        # Zero-shock simulation stays near SS
        sim = simulate(sol, 100; shock_draws=zeros(100, 1))
        @test size(sim) == (100, 1)

        # Stochastic simulation doesn't explode
        sim2 = simulate(sol, 10000; rng=Random.MersenneTwister(42))
        @test all(isfinite.(sim2))
        @test std(sim2[:, 1]) < 1.0

        # Antithetic shocks
        sim_anti = simulate(sol, 1000; antithetic=true, rng=Random.MersenneTwister(42))
        @test size(sim_anti) == (1000, 1)
        @test all(isfinite.(sim_anti))

        # IRF works
        ir = irf(sol, 20)
        @test length(ir.variables) == 1
        @test size(ir.values) == (20, 1, 1)

        # GIRF
        ir_g = irf(sol, 20; irf_type=:girf, n_draws=100)
        @test size(ir_g.values) == (20, 1, 1)

        # FEVD works
        fv = fevd(sol, 20)
        @test all(fv.proportions[:, 1, :] .≈ 1.0)  # single shock = 100%
    end

    @testset "Closed-form moments" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        # Compare first-order perturbation moments with existing analytical_moments
        sol1 = solve(spec; method=:gensys)
        sol_p1 = solve(spec; method=:perturbation, order=1)

        mom1 = analytical_moments(sol1; lags=1)
        mom_p1 = analytical_moments(sol_p1; lags=1)

        # Both should have same length: k*(k+1)/2 + k*lags = 1 + 1 = 2
        @test length(mom_p1) == length(mom1)

        # Variance should match: σ²/(1-ρ²) = 0.01²/(1-0.81) ≈ 5.2632e-4
        theoretical_var = 0.01^2 / (1 - 0.9^2)
        @test mom_p1[1] ≈ theoretical_var atol=1e-8
        @test mom_p1[1] ≈ mom1[1] atol=1e-8

        # Autocovariance at lag 1: ρ * variance
        theoretical_autocov = 0.9 * theoretical_var
        @test mom_p1[2] ≈ theoretical_autocov atol=1e-8
        @test mom_p1[2] ≈ mom1[2] atol=1e-8

        # Multiple lags
        mom_p1_3 = analytical_moments(sol_p1; lags=3)
        @test length(mom_p1_3) == 1 + 3  # k*(k+1)/2 + k*lags = 1 + 3

        # Second-order moments (simulation-based for order >= 2)
        sol2 = solve(spec; method=:perturbation, order=2)
        mom2 = analytical_moments(sol2; lags=1)
        @test all(isfinite.(mom2))
        @test length(mom2) == length(mom1)
        # For a linear model, 2nd-order moments should be close to 1st-order
        @test mom2[1] ≈ mom1[1] atol=1e-3  # relaxed tolerance for simulation-based

        # Test _dlyap_doubling directly
        A = [0.9;;]  # 1x1 matrix
        B = [0.01^2;;]
        Sigma = MacroEconometricModels._dlyap_doubling(A, B)
        @test Sigma[1,1] ≈ theoretical_var atol=1e-10

        # Test _dlyap_doubling matches solve_lyapunov for multivariate case
        G1_test = [0.8 0.1; 0.05 0.7]
        impact_test = [0.01 0.0; 0.0 0.02]
        Sigma_kron = solve_lyapunov(G1_test, impact_test)
        Sigma_doub = MacroEconometricModels._dlyap_doubling(G1_test, impact_test * impact_test')
        @test Sigma_doub ≈ Sigma_kron atol=1e-10
    end

    # =====================================================================
    # Additional comprehensive tests (Task 6)
    # =====================================================================

    @testset "Two-variable model (RBC-like)" begin
        spec = @dsge begin
            parameters: ρ = 0.95, σ = 0.01, α = 0.5
            endogenous: k, c
            exogenous: ε
            c[t] = c[t+1] - α * (k[t] - k[t-1]) + σ * ε[t]
            k[t] = (1 + α) * k[t-1] - c[t] + σ * ε[t]
            steady_state = [0.0, 0.0]
        end
        spec = compute_steady_state(spec)

        sol2 = solve(spec; method=:perturbation, order=2)
        @test sol2 isa MacroEconometricModels.PerturbationSolution
        @test sol2.order == 2
        @test is_determined(sol2)
        @test nvars(sol2) == 2
        @test nshocks(sol2) == 1
        @test MacroEconometricModels.nstates(sol2) + MacroEconometricModels.ncontrols(sol2) == 2

        # Simulation
        sim = simulate(sol2, 1000; rng=Random.MersenneTwister(42))
        @test size(sim, 2) == 2
        @test all(isfinite.(sim))

        # IRF — 2 variables, 1 shock
        ir = irf(sol2, 40)
        @test size(ir.values) == (40, 2, 1)
        @test all(isfinite.(ir.values))

        # FEVD — single shock should explain 100% of variance for both variables
        fv = fevd(sol2, 40)
        @test all(isfinite.(fv.proportions))
        for h in 1:40, i in 1:2
            @test fv.proportions[i, 1, h] ≈ 1.0 atol=1e-8
        end

        # Moments
        mom = analytical_moments(sol2; lags=2)
        @test all(isfinite.(mom))
    end

    @testset "Pruning stability — long simulation" begin
        spec = @dsge begin
            parameters: ρ = 0.95, σ = 0.1
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        sol2 = solve(spec; method=:perturbation, order=2)
        # Long simulation should not explode — key pruning stability test
        sim = simulate(sol2, 100000; rng=Random.MersenneTwister(42))
        @test all(isfinite.(sim))
        @test std(sim[:, 1]) < 10.0  # bounded variance
    end

    @testset "Order 1 downstream equivalence" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        sol_g = solve(spec; method=:gensys)
        sol_p = solve(spec; method=:perturbation, order=1)

        # Same simulation with same shocks
        shocks = randn(Random.MersenneTwister(42), 100, 1)
        sim_g = simulate(sol_g, 100; shock_draws=shocks)
        sim_p = simulate(sol_p, 100; shock_draws=shocks)
        @test sim_g ≈ sim_p atol=1e-6

        # Same IRF
        ir_g = irf(sol_g, 20)
        ir_p = irf(sol_p, 20)
        @test ir_g.values ≈ ir_p.values atol=1e-6

        # Same FEVD
        fv_g = fevd(sol_g, 20)
        fv_p = fevd(sol_p, 20)
        @test fv_g.proportions ≈ fv_p.proportions atol=1e-6

        # Same moments
        mom_g = analytical_moments(sol_g; lags=1)
        mom_p = analytical_moments(sol_p; lags=1)
        @test mom_g ≈ mom_p atol=1e-6
    end

    @testset "Display — comprehensive" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:perturbation, order=2)
        io = IOBuffer()
        show(io, sol)
        output = String(take!(io))
        @test occursin("Perturbation", output) || occursin("perturbation", output)
        @test occursin("2", output)  # order 2
        @test occursin("States", output) || occursin("state", output)
        @test occursin("Controls", output) || occursin("control", output)
        @test occursin("Stable", output) || occursin("stable", output)

        # Order 1 display also works
        sol1 = solve(spec; method=:perturbation, order=1)
        io1 = IOBuffer()
        show(io1, sol1)
        out1 = String(take!(io1))
        @test occursin("1", out1)  # order 1
        @test occursin("Perturbation", out1) || occursin("perturbation", out1)
    end

    @testset "Edge cases — invalid order" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        # Order 3 is recognized but not yet implemented
        @test_throws ArgumentError solve(spec; method=:perturbation, order=3)
        # Order 4 is out of valid range
        @test_throws ArgumentError solve(spec; method=:perturbation, order=4)
        # Order 0 is invalid
        @test_throws ArgumentError solve(spec; method=:perturbation, order=0)
    end

    @testset "GIRF vs analytical IRF" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:perturbation, order=2)

        ir_a = irf(sol, 20; irf_type=:analytical)
        ir_g = irf(sol, 20; irf_type=:girf, n_draws=500)

        # For a linear model, GIRF and analytical should agree closely
        @test size(ir_a.values) == size(ir_g.values)
        @test ir_a.values ≈ ir_g.values atol=0.01

        # Invalid irf_type raises
        @test_throws ArgumentError irf(sol, 20; irf_type=:invalid)
    end

    @testset "Zero-shock simulation stays at steady state" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        # Order 1
        sol1 = solve(spec; method=:perturbation, order=1)
        sim1 = simulate(sol1, 50; shock_draws=zeros(50, 1))
        @test all(abs.(sim1) .< 1e-10)

        # Order 2 — should also stay near SS (hσσ correction may shift mean slightly)
        sol2 = solve(spec; method=:perturbation, order=2)
        sim2 = simulate(sol2, 50; shock_draws=zeros(50, 1))
        @test all(abs.(sim2) .< 0.1)  # relaxed: 2nd-order constant correction
    end

    @testset "Multiple shocks — FEVD sums to 1" begin
        # Use the NK model spec which has forward-looking variables only
        spec = @dsge begin
            parameters: β = 0.99, κ = 0.3, φ_π = 1.5, φ_y = 0.125, σ_d = 0.01, σ_s = 0.01
            endogenous: π, y, i
            exogenous: ε_d, ε_s
            π[t] = β * π[t+1] + κ * y[t] + σ_s * ε_s[t]
            y[t] = y[t+1] - (i[t] - π[t+1]) + σ_d * ε_d[t]
            i[t] = φ_π * π[t] + φ_y * y[t]
            steady_state = [0.0, 0.0, 0.0]
        end
        spec = compute_steady_state(spec)

        sol = solve(spec; method=:perturbation, order=2)
        @test nshocks(sol) == 2
        @test nvars(sol) == 3

        fv = fevd(sol, 30)
        # FEVD proportions should sum to 1 across shocks for each variable/horizon
        for h in 1:30, i in 1:3
            total = sum(fv.proportions[i, :, h])
            @test total ≈ 1.0 atol=1e-6
        end
    end

    @testset "Blanchard-Kahn first-order solver in perturbation" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        # perturbation_solver can use blanchard_kahn as the first-order solver
        sol_bk = MacroEconometricModels.perturbation_solver(spec; order=2, method=:blanchard_kahn)
        sol_gs = MacroEconometricModels.perturbation_solver(spec; order=2, method=:gensys)

        @test sol_bk isa MacroEconometricModels.PerturbationSolution
        @test sol_gs isa MacroEconometricModels.PerturbationSolution

        # Both should produce same first-order coefficients
        @test sol_bk.hx ≈ sol_gs.hx atol=1e-6
        @test sol_bk.gx ≈ sol_gs.gx atol=1e-6

        # Second-order terms should also match
        if sol_bk.hxx !== nothing && sol_gs.hxx !== nothing
            @test sol_bk.hxx ≈ sol_gs.hxx atol=1e-6
        end
        if sol_bk.hσσ !== nothing && sol_gs.hσσ !== nothing
            @test sol_bk.hσσ ≈ sol_gs.hσσ atol=1e-6
        end
    end

    @testset "Antithetic simulation reduces variance" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.1
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:perturbation, order=2)

        # Antithetic simulation should have lower mean absolute value
        # (variance reduction technique)
        rng1 = Random.MersenneTwister(123)
        sim_anti = simulate(sol, 2000; antithetic=true, rng=rng1)
        @test size(sim_anti) == (2000, 1)
        @test all(isfinite.(sim_anti))

        # Mean should be closer to zero than raw simulation (on average)
        # Just verify it's finite and reasonable
        @test abs(mean(sim_anti[:, 1])) < 1.0
    end

    @testset "is_stable check" begin
        # Stable AR(1)
        spec_stable = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec_stable = compute_steady_state(spec_stable)
        sol_stable = solve(spec_stable; method=:perturbation, order=2)
        @test is_stable(sol_stable)

        # Pure forward-looking model (no states) is trivially stable
        spec_fwd = @dsge begin
            parameters: β = 0.99, κ = 0.3, φ_π = 1.5, φ_y = 0.125, σ_v = 0.25
            endogenous: π, y, i
            exogenous: ε_v
            π[t] = β * π[t+1] + κ * y[t]
            y[t] = y[t+1] - (i[t] - π[t+1]) + σ_v * ε_v[t]
            i[t] = φ_π * π[t] + φ_y * y[t]
            steady_state = [0.0, 0.0, 0.0]
        end
        spec_fwd = compute_steady_state(spec_fwd)
        sol_fwd = solve(spec_fwd; method=:perturbation, order=2)
        @test is_stable(sol_fwd)  # nx=0 => trivially stable
    end
end

@testset "GMM Higher-Order Moments" begin

    @testset "DSGEEstimation with PerturbationSolution" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol_p = solve(spec; method=:perturbation, order=2)

        # Should be able to construct DSGEEstimation with PerturbationSolution
        est = MacroEconometricModels.DSGEEstimation{Float64}(
            [0.9], [0.01;;], [:ρ], :analytical_gmm,
            0.0, 1.0, sol_p, true, spec
        )
        @test est.theta == [0.9]
        @test est.method == :analytical_gmm
        @test est.solution isa MacroEconometricModels.PerturbationSolution

        # show() should work without error
        io = IOBuffer()
        show(io, est)
        output = String(take!(io))
        @test occursin("DSGE Estimation", output)
        @test occursin("analytical_gmm", output)
    end

    @testset "GMM moment format" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        # Order 1: GMM format includes means (should be zero)
        sol1 = solve(spec; method=:perturbation, order=1)
        mom_gmm1 = analytical_moments(sol1; lags=1, format=:gmm)
        # ny=1: 1 mean + 1 product moment + 1 autocov = 3
        @test length(mom_gmm1) == 3
        @test abs(mom_gmm1[1]) < 1e-10  # mean ≈ 0 for order 1

        # Product moment E[y²] = Var(y) + E[y]² ≈ Var(y)
        theoretical_var = 0.01^2 / (1 - 0.9^2)
        @test mom_gmm1[2] ≈ theoretical_var atol=1e-8

        # Autocov: E[y_t * y_{t-1}] = Cov(y_t,y_{t-1}) + E[y]²
        # For AR(1): Cov(lag=1) = ρ * Var(y)
        @test mom_gmm1[3] ≈ 0.9 * theoretical_var atol=1e-8

        # Order 2: GMM format — mean may be non-zero
        sol2 = solve(spec; method=:perturbation, order=2)
        mom_gmm2 = analytical_moments(sol2; lags=1, format=:gmm)
        @test length(mom_gmm2) == 3
        @test all(isfinite.(mom_gmm2))

        # Default format (:covariance) still works and is backward-compatible
        mom_cov = analytical_moments(sol1; lags=1)
        @test length(mom_cov) == 2  # k*(k+1)/2 + k*lags = 1 + 1

        # Invalid format throws
        @test_throws ArgumentError analytical_moments(sol1; format=:invalid)
    end

    @testset "Data moments computation" begin
        Random.seed!(42)
        data = randn(500, 1)
        m_data = MacroEconometricModels._compute_data_moments(data; lags=[1])
        # 1 mean + 1 product moment + 1 autocov = 3
        @test length(m_data) == 3
        @test m_data[1] ≈ sum(data[:, 1]) / 500 atol=1e-10
        @test m_data[2] ≈ dot(data[:, 1], data[:, 1]) / 500 atol=1e-10

        # Multi-variable data moments
        data2 = randn(500, 2)
        m_data2 = MacroEconometricModels._compute_data_moments(data2; lags=[1, 3])
        # ny=2: 2 means + 3 product moments + 2*2 autocov = 2 + 3 + 4 = 9
        @test length(m_data2) == 9

        # observable_indices filtering
        data3 = randn(500, 3)
        m_sub = MacroEconometricModels._compute_data_moments(data3; lags=[1], observable_indices=[1, 3])
        # 2 means + 3 product moments + 2 autocov = 7
        @test length(m_sub) == 7
    end

    @testset "Perturbation-order analytical GMM" begin
        spec = @dsge begin
            parameters: ρ = 0.85, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        # Generate data from known parameters
        Random.seed!(42)
        sol_true = solve(spec; method=:perturbation, order=2)
        data = simulate(sol_true, 500; rng=Random.MersenneTwister(42))

        # Estimate with perturbation order 2
        bounds = ParameterTransform{Float64}([0.01], [0.999])
        est = estimate_dsge(spec, data, [:ρ];
                             method=:analytical_gmm,
                             solve_method=:perturbation,
                             solve_order=2,
                             auto_lags=[1],
                             bounds=bounds)

        @test est.converged
        @test est.method == :analytical_gmm
        @test est.solution isa MacroEconometricModels.PerturbationSolution
        # Parameter should be recovered within CI
        @test abs(est.theta[1] - 0.85) < 0.15
        @test est.J_stat >= 0.0
        @test 0.0 <= est.J_pvalue <= 1.0
    end

    @testset "Innovation variance 2nd order" begin
        hx_s = [0.9;;]
        eta_x = [0.01;;]
        Var_xf = MacroEconometricModels._dlyap_doubling(hx_s, eta_x * eta_x')

        Var_inov = MacroEconometricModels._innovation_variance_2nd(
            hx_s, eta_x, Var_xf, 1, 1)

        # nz = 2*1 + 1 = 3
        @test size(Var_inov) == (3, 3)
        # Block (1,1) = eta_x * eta_x' = 0.0001
        @test Var_inov[1, 1] ≈ 0.01^2 atol=1e-12
        # Symmetric
        @test Var_inov ≈ Var_inov' atol=1e-15
        # Positive semi-definite
        @test all(eigvals(Symmetric(Var_inov)) .>= -1e-12)
    end

    @testset "Extract xx block" begin
        # nx=2, n_eps=1, nv=3 → nv²=9, nx²=4
        M = reshape(collect(1.0:18.0), 2, 9)
        Mxx = MacroEconometricModels._extract_xx_block(M, 2, 3)
        @test size(Mxx) == (2, 4)
        # Column (1,1) of v⊗v = column 1 of M → column 1 of Mxx
        @test Mxx[:, 1] == M[:, 1]
        # Column (1,2) of v⊗v = column 2 of M → column 2 of Mxx
        @test Mxx[:, 2] == M[:, 2]
        # Column (2,1) of v⊗v = column 4 of M (=(2-1)*3+1) → column 3 of Mxx
        @test Mxx[:, 3] == M[:, 4]
        # Column (2,2) of v⊗v = column 5 of M (=(2-1)*3+2) → column 4 of Mxx
        @test Mxx[:, 4] == M[:, 5]
    end

    @testset "2nd-order risk correction: non-zero mean" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.1
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol2 = solve(spec; method=:perturbation, order=2)

        result = MacroEconometricModels._augmented_moments_2nd(sol2; lags=[1])

        # Mean exists and is finite
        @test all(isfinite.(result[:E_y]))
        # Variance is positive
        @test all(diag(result[:Var_y]) .> 0)
    end

    @testset "Data moments match analytical for generated data" begin
        Random.seed!(123)
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol1 = solve(spec; method=:perturbation, order=1)

        # Generate long simulation
        data = simulate(sol1, 100_000; rng=Random.MersenneTwister(123))

        # Data moments should converge to model moments
        m_model = analytical_moments(sol1; lags=1, format=:gmm)
        m_data = MacroEconometricModels._compute_data_moments(data; lags=[1])

        @test length(m_model) == length(m_data)
        # Mean ≈ 0 for order 1
        @test abs(m_data[1]) < 0.01
        # Product moment ≈ theoretical variance
        theoretical_var = 0.01^2 / (1 - 0.9^2)
        @test m_data[2] ≈ theoretical_var atol=0.05 * theoretical_var
    end

    @testset "Closed-form 2nd-order matches simulation" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol2 = solve(spec; method=:perturbation, order=2)

        # Closed-form moments
        mom_cf = analytical_moments(sol2; lags=1, format=:gmm)

        # Simulation-based moments (long run)
        sim = simulate(sol2, 500_000; rng=Random.MersenneTwister(99))
        m_sim = MacroEconometricModels._compute_data_moments(sim; lags=[1])

        @test length(mom_cf) == length(m_sim)
        # Should match within sampling error (generous tolerance for mean near zero)
        for i in eachindex(mom_cf)
            @test mom_cf[i] ≈ m_sim[i] atol=max(abs(m_sim[i]) * 0.15, 1e-4)
        end
    end

    @testset "Multi-variable model moments" begin
        spec = @dsge begin
            parameters: ρ₁ = 0.8, ρ₂ = 0.7, σ₁ = 0.01, σ₂ = 0.02
            endogenous: x, y
            exogenous: ε₁, ε₂
            x[t] = ρ₁ * x[t-1] + σ₁ * ε₁[t]
            y[t] = ρ₂ * y[t-1] + σ₂ * ε₂[t]
        end
        spec = compute_steady_state(spec)
        sol2 = solve(spec; method=:perturbation, order=2)

        mom = analytical_moments(sol2; lags=2, format=:gmm)
        # ny=2: 2 means + 3 product moments + 2*2 autocov = 2 + 3 + 4 = 9
        @test length(mom) == 9
        @test all(isfinite.(mom))
    end

    @testset "Backward compatibility: default format unchanged" begin
        spec = @dsge begin
            parameters: ρ = 0.9, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        # Order 1: default format matches existing behavior
        sol1 = solve(spec; method=:perturbation, order=1)
        sol_g = solve(spec; method=:gensys)
        mom_p = analytical_moments(sol1; lags=1)
        mom_g = analytical_moments(sol_g; lags=1)
        @test length(mom_p) == length(mom_g)
        @test mom_p ≈ mom_g atol=1e-8

        # Order 2: default format uses simulation (backward compatible)
        sol2 = solve(spec; method=:perturbation, order=2)
        mom_sim = analytical_moments(sol2; lags=1)
        @test length(mom_sim) == length(mom_g)  # same format
    end

    @testset "Existing analytical_gmm still works" begin
        spec = @dsge begin
            parameters: ρ = 0.85, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)
        sol = solve(spec; method=:gensys)
        Random.seed!(42)
        data = simulate(sol, 200; rng=Random.MersenneTwister(42))

        # Old API: estimate_dsge with analytical_gmm, no perturbation kwargs
        bounds = ParameterTransform{Float64}([0.01], [0.999])
        est = estimate_dsge(spec, data, [:ρ];
                             method=:analytical_gmm,
                             bounds=bounds)
        @test est.converged
        @test est.method == :analytical_gmm
        @test est.solution isa MacroEconometricModels.DSGESolution
    end

    @testset "Round-trip perturbation GMM estimation" begin
        spec = @dsge begin
            parameters: ρ = 0.85, σ = 0.01
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + σ * ε[t]
        end
        spec = compute_steady_state(spec)

        # Generate data from known model
        sol_true = solve(spec; method=:perturbation, order=2)
        Random.seed!(7777)
        data = simulate(sol_true, 1000; rng=Random.MersenneTwister(7777))

        # Estimate ρ with perturbation order 2, multiple autocov lags
        bounds = ParameterTransform{Float64}([0.01], [0.999])
        est = estimate_dsge(spec, data, [:ρ];
                             method=:analytical_gmm,
                             solve_method=:perturbation,
                             solve_order=2,
                             auto_lags=[1, 3],
                             bounds=bounds)

        @test est.converged
        @test abs(est.theta[1] - 0.85) < 0.2
        @test est.J_stat >= 0.0

        # Show works
        io = IOBuffer()
        show(io, est)
        output = String(take!(io))
        @test occursin("analytical_gmm", output)
    end

end

# ─────────────────────────────────────────────────────────────────────────────
# Section: Projection Methods (Chebyshev Collocation)
# ─────────────────────────────────────────────────────────────────────────────

@testset "Projection Methods" begin

@testset "ProjectionSolution type" begin
    @test isdefined(MacroEconometricModels, :ProjectionSolution)
    @test ProjectionSolution <: Any
end

@testset "Quadrature" begin
    @testset "Gauss-Hermite nodes and weights" begin
        for n in [3, 5, 7]
            nodes, weights = MacroEconometricModels._gauss_hermite_nodes_weights(n)
            @test length(nodes) == n
            @test length(weights) == n
            # Weights sum to √π
            @test sum(weights) ≈ sqrt(π) atol=1e-12
            # Nodes are symmetric around 0
            @test sort(nodes) ≈ sort(-reverse(nodes)) atol=1e-12
        end
        # Error for n < 1
        @test_throws ArgumentError MacroEconometricModels._gauss_hermite_nodes_weights(0)
    end

    @testset "Gauss-Hermite polynomial exactness" begin
        nodes5, w5 = MacroEconometricModels._gauss_hermite_nodes_weights(5)
        # ∫ exp(-x²) dx = √π
        @test dot(w5, ones(5)) ≈ sqrt(π) atol=1e-12
        # ∫ x² exp(-x²) dx = √π/2
        @test dot(w5, nodes5.^2) ≈ sqrt(π) / 2 atol=1e-12
        # ∫ x⁴ exp(-x²) dx = 3√π/4
        @test dot(w5, nodes5.^4) ≈ 3 * sqrt(π) / 4 atol=1e-12
        # ∫ x⁸ exp(-x²) dx = 105√π/16 (degree 8 ≤ 2*5-1=9)
        @test dot(w5, nodes5.^8) ≈ 105 * sqrt(π) / 16 atol=1e-10
    end

    @testset "Gauss-Hermite scaled" begin
        Σ = [1.0 0.0; 0.0 1.0]
        nodes, weights = MacroEconometricModels._gauss_hermite_scaled(3, Σ)
        @test size(nodes) == (9, 2)
        @test length(weights) == 9
        # Weights sum to 1 (probability measure)
        @test sum(weights) ≈ 1.0 atol=1e-12
        # E[x_i] = 0
        for j in 1:2
            @test abs(dot(weights, nodes[:, j])) < 1e-12
        end
        # E[x_i²] = 1 (unit variance)
        for j in 1:2
            @test dot(weights, nodes[:, j].^2) ≈ 1.0 atol=1e-10
        end

        # Non-identity covariance
        Σ2 = [2.0 0.5; 0.5 1.0]
        nodes2, w2 = MacroEconometricModels._gauss_hermite_scaled(3, Σ2)
        @test sum(w2) ≈ 1.0 atol=1e-12
        # E[x₁²] = Σ₁₁ = 2
        @test dot(w2, nodes2[:, 1].^2) ≈ 2.0 atol=1e-10
        # E[x₂²] = Σ₂₂ = 1
        @test dot(w2, nodes2[:, 2].^2) ≈ 1.0 atol=1e-10
        # E[x₁ x₂] = Σ₁₂ = 0.5
        @test dot(w2, nodes2[:, 1] .* nodes2[:, 2]) ≈ 0.5 atol=1e-10
    end

    @testset "Monomial rule" begin
        for n_eps in [1, 2, 3, 5]
            nodes, weights = MacroEconometricModels._monomial_nodes_weights(n_eps)
            @test size(nodes, 1) == 2 * n_eps + 1
            @test size(nodes, 2) == n_eps
            @test length(weights) == 2 * n_eps + 1
            # Weights sum to 1
            @test sum(weights) ≈ 1.0 atol=1e-12
            # E[x_j] = 0
            for j in 1:n_eps
                @test dot(weights, nodes[:, j]) ≈ 0.0 atol=1e-12
            end
            # E[x_j²] = 1
            for j in 1:n_eps
                @test dot(weights, nodes[:, j].^2) ≈ 1.0 atol=1e-12
            end
        end
        # Error for n_eps < 1
        @test_throws ArgumentError MacroEconometricModels._monomial_nodes_weights(0)
    end
end

@testset "Chebyshev basis" begin
    @testset "Chebyshev nodes" begin
        nodes = MacroEconometricModels._chebyshev_nodes(5)
        @test length(nodes) == 5
        @test nodes[1] ≈ 1.0 atol=1e-14
        @test nodes[5] ≈ -1.0 atol=1e-14
        @test nodes[3] ≈ 0.0 atol=1e-14
        @test all(-1 .<= nodes .<= 1)
        # Error for n < 2
        @test_throws ArgumentError MacroEconometricModels._chebyshev_nodes(1)
    end

    @testset "Chebyshev polynomial evaluation" begin
        x = 0.5
        vals = MacroEconometricModels._chebyshev_eval(x, 4)
        @test vals[1] ≈ 1.0 atol=1e-14       # T_0
        @test vals[2] ≈ 0.5 atol=1e-14       # T_1
        @test vals[3] ≈ 2*0.25 - 1 atol=1e-14 # T_2 = -0.5
        @test vals[4] ≈ 4*0.125 - 3*0.5 atol=1e-14  # T_3 = 4x^3-3x
        @test length(vals) == 5  # T_0 through T_4

        # Boundary values: T_n(1) = 1 for all n
        vals1 = MacroEconometricModels._chebyshev_eval(1.0, 5)
        @test all(v -> v ≈ 1.0, vals1)

        # Boundary values: T_n(-1) = (-1)^n
        valsm1 = MacroEconometricModels._chebyshev_eval(-1.0, 5)
        for k in 0:5
            @test valsm1[k + 1] ≈ (-1.0)^k atol=1e-14
        end
    end

    @testset "Scale/unscale round-trip" begin
        bounds = [1.0 5.0; -2.0 3.0]
        x_phys = [3.0, 0.5]
        z = MacroEconometricModels._scale_to_unit(x_phys, bounds)
        @test all(-1 .<= z .<= 1)
        x_back = MacroEconometricModels._scale_from_unit(z, bounds)
        @test x_back ≈ x_phys atol=1e-14

        # Boundary checks
        x_lo = [1.0, -2.0]
        @test MacroEconometricModels._scale_to_unit(x_lo, bounds) ≈ [-1.0, -1.0] atol=1e-14
        x_hi = [5.0, 3.0]
        @test MacroEconometricModels._scale_to_unit(x_hi, bounds) ≈ [1.0, 1.0] atol=1e-14

        # Matrix versions
        X_phys = [3.0 0.5; 1.0 -2.0; 5.0 3.0]
        Z = MacroEconometricModels._scale_to_unit(X_phys, bounds)
        @test size(Z) == (3, 2)
        X_back = MacroEconometricModels._scale_from_unit(Z, bounds)
        @test X_back ≈ X_phys atol=1e-14
    end

    @testset "Tensor-product basis matrix" begin
        nodes1d = MacroEconometricModels._chebyshev_nodes(3)
        X = reshape(nodes1d, 3, 1)
        mi = reshape([0; 1; 2], 3, 1)
        B = MacroEconometricModels._chebyshev_basis_multi(X, mi)
        @test size(B) == (3, 3)
        @test cond(B) < 100

        # 2D test
        nodes2d, mi2d = MacroEconometricModels._tensor_grid(2, 2)
        B2d = MacroEconometricModels._chebyshev_basis_multi(nodes2d, mi2d)
        @test size(B2d) == (9, 9)
        @test cond(B2d) < 1e6  # should be reasonably conditioned
    end
end

@testset "Grid construction" begin
    @testset "Tensor grid" begin
        for nx in [1, 2, 3]
            degree = 3
            nodes, mi = MacroEconometricModels._tensor_grid(nx, degree)
            expected_nodes = (degree + 1)^nx
            @test size(nodes, 1) == expected_nodes
            @test size(nodes, 2) == nx
            @test size(mi, 1) == expected_nodes
            @test size(mi, 2) == nx
            @test all(-1 .<= nodes .<= 1)
            @test all(0 .<= mi .<= degree)
        end
    end

    @testset "Smolyak grid" begin
        for (nx, mu) in [(2, 2), (2, 3), (3, 2)]
            nodes, mi = MacroEconometricModels._smolyak_grid(nx, mu)
            tensor_nodes, _ = MacroEconometricModels._tensor_grid(nx, mu + nx)
            # Smolyak grid should have fewer points than the full tensor grid
            @test size(nodes, 1) < size(tensor_nodes, 1)
            @test size(nodes, 2) == nx
            @test all(-1 .<= nodes .<= 1)
        end

        # 2D mu=1: known node count
        nodes_21, mi_21 = MacroEconometricModels._smolyak_grid(2, 1)
        @test size(nodes_21, 2) == 2
        @test size(nodes_21, 1) >= 5  # at least the cross pattern
    end
end

@testset "Linear AR(1) projection" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol = solve(spec; method=:projection, degree=5, verbose=false)

    @test sol isa MacroEconometricModels.ProjectionSolution
    @test sol.converged
    @test sol.residual_norm < 1e-8
    @test sol.method == :projection
    @test sol.iterations <= 10

    # evaluate_policy at steady state should return steady state
    y_ss = evaluate_policy(sol, [0.0])
    @test length(y_ss) == 1
    @test abs(y_ss[1]) < 1e-6

    # Linear model: projection should recover linear policy
    pert_sol = solve(spec; method=:gensys)
    for x_val in [-0.02, -0.01, 0.0, 0.01, 0.02]
        y_proj = evaluate_policy(sol, [x_val])
        y_pert = pert_sol.G1[1, 1] * x_val  # linear: y = G1 * x (deviation)
        @test abs(y_proj[1] - y_pert) < 1e-4
    end

    # max_euler_error should be small
    euler_err = max_euler_error(sol; n_test=100, rng=Random.MersenneTwister(42))
    @test euler_err < 1e-6
end

@testset "Projection simulate and irf" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    sol = solve(spec; method=:projection, degree=5, verbose=false)

    # simulate returns T × n matrix
    Random.seed!(42)
    Y_sim = simulate(sol, 100)
    @test size(Y_sim) == (100, 1)
    @test all(abs.(Y_sim) .< 1.0)

    # simulate with explicit shock_draws
    shocks = zeros(50, 1)
    shocks[1, 1] = 1.0
    Y_det = simulate(sol, 50; shock_draws=shocks)
    @test size(Y_det) == (50, 1)

    # irf returns ImpulseResponse
    irfs = irf(sol, 20; n_sim=200)
    @test irfs isa ImpulseResponse
    @test size(irfs.values) == (20, 1, 1)
    # First period: should be close to σ = 0.01 (impact of unit shock)
    @test abs(irfs.values[1, 1, 1] - 0.01) < 0.005
    # IRF should decay
    @test abs(irfs.values[20, 1, 1]) < abs(irfs.values[1, 1, 1])
end

@testset "API integration" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    @testset "show() works" begin
        sol = solve(spec; method=:projection, degree=3, verbose=false)
        io = IOBuffer()
        show(io, sol)
        output = String(take!(io))
        @test occursin("Projection", output)
        @test occursin("Chebyshev", output) || occursin("projection", lowercase(output))
        @test occursin("Converged", output)
    end

    @testset "Grid auto-selection" begin
        sol = solve(spec; method=:projection, degree=3, grid=:auto, verbose=false)
        @test sol.grid_type == :tensor  # 1 state → tensor
    end

    @testset "Quadrature auto-selection" begin
        sol = solve(spec; method=:projection, degree=3, quadrature=:auto, verbose=false)
        @test sol.quadrature == :gauss_hermite  # 1 shock → GH
    end

    @testset "Accessors" begin
        sol = solve(spec; method=:projection, degree=3, verbose=false)
        @test nvars(sol) == 1
        @test nshocks(sol) == 1
        @test MacroEconometricModels.nstates(sol) >= 1
        @test MacroEconometricModels.ncontrols(sol) >= 0
        @test is_determined(sol) == sol.converged
        @test is_stable(sol) == sol.converged
    end

    @testset "evaluate_policy matrix input" begin
        sol = solve(spec; method=:projection, degree=3, verbose=false)
        X_mat = reshape([-0.02, -0.01, 0.0, 0.01, 0.02], 5, 1)
        Y = evaluate_policy(sol, X_mat)
        @test size(Y) == (5, 1)
        # Each row should match single-point evaluation
        for i in 1:5
            y_single = evaluate_policy(sol, [X_mat[i, 1]])
            @test Y[i, 1] ≈ y_single[1] atol=1e-14
        end
    end

    @testset "Backward compatibility" begin
        sol_gensys = solve(spec; method=:gensys)
        @test sol_gensys isa DSGESolution
        sol_bk = solve(spec; method=:blanchard_kahn)
        @test sol_bk isa DSGESolution
        sol_pert = solve(spec; method=:perturbation, order=2)
        @test sol_pert isa PerturbationSolution
    end
end

@testset "Nonlinear growth model" begin
    # Neoclassical growth model (standard timing: k[t-1] is beginning-of-period capital)
    # Euler: c[t]^(-γ) = β * c[t+1]^(-γ) * (α * k[t]^(α-1) + 1 - δ)
    # Resource: k[t] = k[t-1]^α + (1-δ)*k[t-1] - c[t] + σ_e*ε[t]
    spec = @dsge begin
        parameters: α = 0.36, β = 0.99, δ = 0.025, γ = 2.0, σ_e = 0.01
        endogenous: k, c
        exogenous: ε
        c[t]^(-γ) - β * c[t+1]^(-γ) * (α * k[t]^(α - 1) + 1 - δ) = 0
        k[t] - k[t-1]^α - (1 - δ) * k[t-1] + c[t] - σ_e * ε[t] = 0
        steady_state = begin
            k_ss = (α / (1/β - 1 + δ))^(1 / (1 - α))
            c_ss = k_ss^α - δ * k_ss
            [k_ss, c_ss]
        end
    end
    spec = compute_steady_state(spec)
    k_ss = spec.steady_state[1]
    c_ss = spec.steady_state[2]

    sol = solve(spec; method=:projection, degree=5, scale=3.0, verbose=false, tol=1e-3)

    @test sol isa ProjectionSolution
    @test sol.converged
    @test sol.residual_norm < 1e-3

    # Policy at SS should return approximately SS
    y_at_ss = evaluate_policy(sol, [k_ss])
    @test abs(y_at_ss[1] - k_ss) / k_ss < 0.01  # within 1%
    @test abs(y_at_ss[2] - c_ss) / c_ss < 0.01

    # Euler error check
    euler_err = max_euler_error(sol; n_test=200, rng=Random.MersenneTwister(123))
    @test euler_err < 1e-2
end

@testset "Projection vs perturbation accuracy" begin
    spec = @dsge begin
        parameters: α = 0.36, β = 0.99, δ = 0.025, γ = 2.0, σ_e = 0.05
        endogenous: k, c
        exogenous: ε
        c[t]^(-γ) - β * c[t+1]^(-γ) * (α * k[t]^(α - 1) + 1 - δ) = 0
        k[t] - k[t-1]^α - (1 - δ) * k[t-1] + c[t] - σ_e * ε[t] = 0
        steady_state = begin
            k_ss = (α / (1/β - 1 + δ))^(1 / (1 - α))
            c_ss = k_ss^α - δ * k_ss
            [k_ss, c_ss]
        end
    end
    spec = compute_steady_state(spec)

    sol_proj = solve(spec; method=:projection, degree=5, scale=3.0, verbose=false, tol=1e-3)
    sol_pert = solve(spec; method=:gensys)

    k_ss = spec.steady_state[1]
    c_ss = spec.steady_state[2]

    # Both agree near steady state
    y_proj_ss = evaluate_policy(sol_proj, [k_ss])
    @test abs(y_proj_ss[1] - k_ss) / k_ss < 0.01
    @test abs(y_proj_ss[2] - c_ss) / c_ss < 0.01

    # At state bounds, projection and perturbation should both produce valid values
    k_low = sol_proj.state_bounds[1, 1]
    y_proj_low = evaluate_policy(sol_proj, [k_low])

    # Perturbation: G1 maps full y_{t-1} deviations to y_t deviations
    state_idx = sol_proj.state_indices
    y_lag_dev = zeros(spec.n_endog)
    y_lag_dev[state_idx[1]] = k_low - k_ss
    y_pert_low = sol_pert.G1 * y_lag_dev .+ [k_ss, c_ss]

    # Both should produce valid (finite) values
    @test all(isfinite.(y_proj_low))
    @test all(isfinite.(y_pert_low))
    @test length(y_proj_low) == 2
    @test length(y_pert_low) == 2
end

end # Projection Methods

# ─────────────────────────────────────────────────────────────────────────────
# Section: Policy Function Iteration (PFI / Time Iteration)
# ─────────────────────────────────────────────────────────────────────────────

@testset "Policy Function Iteration" begin

@testset "Linear AR(1) PFI" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol = solve(spec; method=:pfi, degree=5, verbose=false)

    @test sol isa ProjectionSolution
    @test sol.method == :pfi
    @test sol.converged
    @test sol.residual_norm < 1e-6
    @test sol.iterations <= 10

    # evaluate_policy at steady state should return steady state
    y_ss = evaluate_policy(sol, [0.0])
    @test length(y_ss) == 1
    @test abs(y_ss[1]) < 1e-6

    # Linear model: PFI should recover linear policy
    pert_sol = solve(spec; method=:gensys)
    for x_val in [-0.02, -0.01, 0.0, 0.01, 0.02]
        y_pfi = evaluate_policy(sol, [x_val])
        y_pert = pert_sol.G1[1, 1] * x_val
        @test abs(y_pfi[1] - y_pert) < 1e-4
    end

    # Euler error
    euler_err = max_euler_error(sol; n_test=100, rng=Random.MersenneTwister(42))
    @test euler_err < 1e-6
end

@testset "PFI vs projection agreement" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    sol_pfi = solve(spec; method=:pfi, degree=5, verbose=false)
    sol_proj = solve(spec; method=:projection, degree=5, verbose=false)

    @test sol_pfi.converged
    @test sol_proj.converged

    # Policy functions should agree on a linear model
    for x_val in [-0.02, 0.0, 0.02]
        y_pfi = evaluate_policy(sol_pfi, [x_val])
        y_proj = evaluate_policy(sol_proj, [x_val])
        @test abs(y_pfi[1] - y_proj[1]) < 1e-4
    end
end

@testset "PFI damping" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    # damping=1.0 (no damping) should converge
    sol1 = solve(spec; method=:pfi, degree=5, damping=1.0, verbose=false)
    @test sol1.converged

    # damping=0.5 should also converge (possibly more iterations)
    sol05 = solve(spec; method=:pfi, degree=5, damping=0.5, verbose=false)
    @test sol05.converged

    # Both should give same policy
    y1 = evaluate_policy(sol1, [0.01])
    y05 = evaluate_policy(sol05, [0.01])
    @test abs(y1[1] - y05[1]) < 1e-4
end

@testset "PFI API integration" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    @testset "show() displays pfi" begin
        sol = solve(spec; method=:pfi, degree=3, verbose=false)
        io = IOBuffer()
        show(io, sol)
        output = String(take!(io))
        @test occursin("Projection", output) || occursin("projection", lowercase(output))
        @test occursin("Converged", output)
    end

    @testset "Grid auto-selection" begin
        sol = solve(spec; method=:pfi, degree=3, grid=:auto, verbose=false)
        @test sol.grid_type == :tensor  # 1 state → tensor
    end

    @testset "Quadrature auto-selection" begin
        sol = solve(spec; method=:pfi, degree=3, quadrature=:auto, verbose=false)
        @test sol.quadrature == :gauss_hermite  # 1 shock → GH
    end

    @testset "Accessors" begin
        sol = solve(spec; method=:pfi, degree=3, verbose=false)
        @test nvars(sol) == 1
        @test nshocks(sol) == 1
        @test MacroEconometricModels.nstates(sol) >= 1
        @test is_determined(sol) == sol.converged
    end

    @testset "Backward compatibility" begin
        sol_gensys = solve(spec; method=:gensys)
        @test sol_gensys isa DSGESolution
        sol_proj = solve(spec; method=:projection, degree=3, verbose=false)
        @test sol_proj isa ProjectionSolution
        @test sol_proj.method == :projection
    end
end

@testset "Nonlinear growth model PFI" begin
    spec = @dsge begin
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
    spec = compute_steady_state(spec)

    k_ss = spec.steady_state[1]
    c_ss = spec.steady_state[2]

    sol = solve(spec; method=:pfi, degree=5, scale=3.0, verbose=false, tol=1e-3, max_iter=500)

    @test sol isa ProjectionSolution
    @test sol.method == :pfi
    @test sol.converged

    # Policy at SS should return approximately SS
    y_at_ss = evaluate_policy(sol, [k_ss])
    @test abs(y_at_ss[1] - k_ss) / k_ss < 0.02
    @test abs(y_at_ss[2] - c_ss) / c_ss < 0.02

    # Euler error check
    euler_err = max_euler_error(sol; n_test=200, rng=Random.MersenneTwister(123))
    @test euler_err < 1e-2

    # Simulation should produce bounded values
    Random.seed!(42)
    Y_sim = simulate(sol, 100)
    @test size(Y_sim) == (100, 2)
    @test all(isfinite.(Y_sim))
end

end # Policy Function Iteration

# ─────────────────────────────────────────────────────────────────────────────
# Section: DSGE Constraint Types
# ─────────────────────────────────────────────────────────────────────────────

@testset "DSGE Constraint Types" begin

@testset "VariableBound" begin
    vb = variable_bound(:i, lower=0.0)
    @test vb isa VariableBound{Float64}
    @test vb.var_name == :i
    @test vb.lower == 0.0
    @test vb.upper === nothing

    vb2 = variable_bound(:h, lower=0.0, upper=1.0)
    @test vb2.lower == 0.0
    @test vb2.upper == 1.0

    vb3 = variable_bound(:c, upper=10.0)
    @test vb3.lower === nothing
    @test vb3.upper == 10.0

    # Error: no bounds specified
    @test_throws ArgumentError variable_bound(:x)

    # Error: lower > upper
    @test_throws ArgumentError variable_bound(:x, lower=2.0, upper=1.0)
end

@testset "NonlinearConstraint" begin
    fn = (y, y_lag, y_lead, e, theta) -> y[1] - 0.8
    nc = nonlinear_constraint(fn; label="test")
    @test nc isa NonlinearConstraint{Float64}
    @test nc.label == "test"
    @test nc.fn([1.0], [1.0], [1.0], [0.0], Dict()) ≈ 0.2

    nc2 = nonlinear_constraint(fn)
    @test nc2.label == "constraint"
end

@testset "Constraint validation" begin
    spec = @dsge begin
        parameters: rho = 0.9, sigma = 0.01
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + sigma * e[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    # Valid constraint
    MacroEconometricModels._validate_constraints(spec, [variable_bound(:y, lower=-1.0)])

    # Invalid variable name
    @test_throws ArgumentError MacroEconometricModels._validate_constraints(
        spec, [variable_bound(:z, lower=0.0)])
end

@testset "Backward compatibility" begin
    spec = @dsge begin
        parameters: rho = 0.9, sigma = 0.01
        endogenous: y
        exogenous: e
        y[t] = rho * y[t-1] + sigma * e[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    @test length(spec.steady_state) == 1
    @test abs(spec.steady_state[1]) < 1e-6

    pf = solve(spec; method=:perfect_foresight, T_periods=10)
    @test pf isa PerfectForesightPath
    @test pf.converged
end

# JuMP integration tests — only run if JuMP + Ipopt are available
_jump_available = try
    @eval import JuMP
    @eval import Ipopt
    true
catch
    false
end

if _jump_available

@testset "Constrained Steady State (JuMP)" begin
    # Simple AR(1) with non-binding constraint
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)

    # Non-binding: SS is 0, lower=-1 doesn't bind
    spec_c = compute_steady_state(spec; constraints=[variable_bound(:y, lower=-1.0)])
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
        constraints=[variable_bound(:k, lower=0.1), variable_bound(:c, lower=0.1)])
    @test abs(spec2_c.steady_state[1] - k_ss) / k_ss < 0.05
    @test abs(spec2_c.steady_state[2] - c_ss) / c_ss < 0.05
end

@testset "Constrained Perfect Foresight (JuMP)" begin
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

    # Non-binding lower bound: constraint doesn't alter the path
    pf_con = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks,
                    constraints=[variable_bound(:y, lower=-5.0)])
    @test pf_con isa PerfectForesightPath
    @test pf_con.converged
    @test all(pf_con.path[:, 1] .>= -5.0 - 1e-4)  # bound respected

    # Path dimensions
    @test size(pf_con.path) == (20, 1)

    # Constrained path matches unconstrained when bound is non-binding
    @test maximum(abs.(pf_con.path .- pf_unc.path)) < 0.01

    # Terminal convergence to SS
    @test abs(pf_con.path[end, 1]) < 0.5

    # Upper bound test: constrain y <= 5 (non-binding)
    shocks2 = zeros(20, 1)
    shocks2[1, 1] = 2.0
    pf_upper = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks2,
                      constraints=[variable_bound(:y, upper=5.0)])
    @test pf_upper isa PerfectForesightPath
    @test pf_upper.converged
    @test all(pf_upper.path[:, 1] .<= 5.0 + 1e-4)
end

@testset "Binding Steady-State Constraint (JuMP)" begin
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

    spec_bound = compute_steady_state(spec; constraints=[variable_bound(:y, lower=0.5)])
    @test spec_bound.steady_state[1] ≈ 0.5 atol=0.01  # bound binds

    # Binding PF: conflicting constraint → solver reports infeasibility
    shocks = zeros(10, 1)
    shocks[1, 1] = -3.0
    pf = solve(spec; method=:perfect_foresight, T_periods=10, shock_path=shocks,
                constraints=[variable_bound(:y, lower=0.0)])
    @test pf isa PerfectForesightPath
    @test !pf.converged  # infeasible: equality constraints + binding bound conflict
end

@testset "NonlinearConstraint Integration (JuMP)" begin
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
end

end # _jump_available

end # DSGE Constraint Types

end # top-level @testset
