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

end # top-level @testset
