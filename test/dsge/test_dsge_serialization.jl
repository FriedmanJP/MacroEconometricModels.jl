# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, Random, MacroEconometricModels

# Task 4 helpers are not on this commit — copy only `_MEM` / `_roundtrip`
# from test/core/test_serialization.jl (do not include that file).
const _MEM = MacroEconometricModels
_roundtrip(m) = _MEM._reconstruct_from_container(_MEM._build_container(m))

# Residual closures index θ by parameter name (`_θ_[:β]`), matching linearize.
# The plan's Vector construction would MethodError on both original and reload.

@testset "DSER-02 ModelSpec recompile" begin
    spec = @dsge begin
        parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.03
        endogenous: k, c, z
        exogenous: ε
        z[t] = ρ * z[t-1] + σ * ε[t]
        k[t] = exp(z[t]) * k[t-1]^α + (1 - δ) * k[t-1] - c[t]
        1 / c[t] = β * (1 / c[t+1]) * (α * exp(z[t+1]) * k[t]^(α - 1) + 1 - δ)
    end
    @test haskey(_MEM._SERIALIZABLE_TYPES, "ModelSpec")
    spec2 = _roundtrip(spec)
    rng = MersenneTwister(760)
    θ = spec.param_values
    for i in 1:length(spec.residual_fns)
        y = randn(rng, spec.n_endog); lag = randn(rng, spec.n_endog); lead = randn(rng, spec.n_endog)
        e = randn(rng, spec.n_exog)
        @test spec2.residual_fns[i](y, lag, lead, e, θ) == spec.residual_fns[i](y, lag, lead, e, θ)
    end
    sol, sol2 = solve(spec), solve(spec2)
    @test sol.G1 == sol2.G1
    @test sol.impact == sol2.impact
    @test_throws SerializationError _MEM._sanitize_residual_ast(:(eval("x")), :eq1)
end

function _dser02_collect_dsge_blocks(root)
    blocks = Expr[]
    stack = Any[root]
    while !isempty(stack)
        ex = pop!(stack)
        ex isa Expr || continue
        if ex.head === :macrocall && !isempty(ex.args) && string(ex.args[1]) == "@dsge"
            block = ex.args[end]
            block isa Expr && push!(blocks, block)
            continue
        end
        append!(stack, ex.args)
    end
    return blocks
end

function _dser02_equation_exprs(block::Expr)
    eqs = Expr[]
    stmts = filter(a -> !(a isa LineNumberNode), block.args)
    for stmt in stmts
        stmt isa Expr || continue
        label = _MEM._detect_declaration(stmt)
        label === :steady_state && continue
        label !== nothing && label in _MEM._RESERVED_DSGE_LABELS && continue
        stmt.head === :(=) && stmt.args[1] === :steady_state && continue
        stmt.head === :(=) && push!(eqs, stmt)
    end
    return eqs
end

@testset "DSER-02 Dynare residual AST allowlist" begin
    dir = joinpath(@__DIR__, "..", "dynare_replication")
    tier_files = sort(filter(p -> startswith(basename(p), "tier") && endswith(p, ".jl"),
                             readdir(dir; join=true)))
    @test !isempty(tier_files)
    for path in tier_files
        parsed = Meta.parseall(read(path, String); filename=path)
        blocks = _dser02_collect_dsge_blocks(parsed)
        # SW07 is built from closures, not `@dsge` equation ASTs.
        isempty(blocks) && continue
        eqs = Expr[]
        for block in blocks
            append!(eqs, _dser02_equation_exprs(block))
        end
        @test !isempty(eqs)
        for eq in eqs
            @test _MEM._sanitize_residual_ast(eq, :dynare) isa Expr
        end
    end
end

# helpers.jl `compare_steady_state` uses atol=1e-6
const _DSER03_SS_ATOL = 1e-6

@testset "DSER-03 ModelIR steady_state recompile" begin
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
    kinds = [d.kind for d in spec.ir.declarations]
    @test :steady_state in kinds
    @test spec.ss_fn !== nothing
    spec2 = _roundtrip(spec)
    θ = spec.param_values
    @test spec2.ss_fn !== nothing
    @test spec2.ss_fn(θ) == spec.ss_fn(θ)
    ss1 = compute_steady_state(spec; method=:analytical)
    ss2 = compute_steady_state(spec2; method=:analytical)
    @test ss2.steady_state == ss1.steady_state
    @test :steady_state in [d.kind for d in spec2.ir.declarations]
end

@testset "DSER-03 linear: true zeros ss_fn" begin
    spec = @dsge begin
        parameters: rho = 0.9, sigma_y = 0.5, kappa = 0.3, phi_pi = 1.5
        endogenous: y, pi_v, i_rate
        exogenous: eps_y
        linear: true
        y[t] = y[t+1] - sigma_y * (i_rate[t] - pi_v[t+1])
        pi_v[t] = 0.99 * pi_v[t+1] + kappa * y[t]
        i_rate[t] = phi_pi * pi_v[t] + eps_y[t]
    end
    @test :linear in [d.kind for d in spec.ir.declarations]
    spec2 = _roundtrip(spec)
    @test spec2.linear
    @test spec2.ss_fn(spec.param_values) == zeros(spec.n_endog)
    @test spec2.ss_fn(spec.param_values) == spec.ss_fn(spec.param_values)
end

@testset "DSER-03 SW07-style programmatic linear zeros" begin
    # Same constructor shape as test/dynare_replication/sw07_model.jl
    n = 40
    spec = ModelSpec{Float64}(
        [Symbol(:y, i) for i in 1:n], [:e], [:ρ],
        Dict(:ρ => 0.9),
        [:(0 + 0) for _ in 1:n],
        [((yt, yl, yle, e, th) -> 0.0) for _ in 1:n],
        0, Int[], zeros(n), nothing;
        linear=true,
    )
    spec2 = _roundtrip(spec)
    @test spec2.ss_fn(spec.param_values) == zeros(n)
    @test spec2.steady_state == zeros(n)
end

@testset "DSER-03 Bellman / varnames IR decls" begin
    spec = @dsge begin
        parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.95, σ = 0.007
        endogenous: c, k, a
        exogenous: ε
        utility: log(c)
        beta: β
        controls: c
        varnames: Consumption, Capital, TFP
        1 / c[t] = β * (1 / c[t+1]) * (α * exp(a[t+1]) * k[t]^(α - 1) + 1 - δ)
        c[t] + k[t] = exp(a[t]) * k[t-1]^α + (1 - δ) * k[t-1]
        a[t] = ρ * a[t-1] + σ * ε[t]
    end
    kinds = [d.kind for d in spec.ir.declarations]
    @test :utility in kinds
    @test :beta in kinds
    @test :controls in kinds
    @test :varnames in kinds
    spec2 = _roundtrip(spec)
    kinds2 = [d.kind for d in spec2.ir.declarations]
    @test :utility in kinds2 && :beta in kinds2 && :controls in kinds2 && :varnames in kinds2
    @test spec2.bellman_utility === spec.bellman_utility
    @test spec2.bellman_beta === spec.bellman_beta
    @test spec2.bellman_controls == spec.bellman_controls
    @test spec2.varnames == spec.varnames
end

@testset "DSER-03 programmatic ss_fn closure dropped" begin
    my_ss = θ -> [0.0]
    spec = ModelSpec{Float64}(
        [:y], [:ε], [:ρ],
        Dict(:ρ => 0.9),
        [:(y[t] - ρ * y[t-1])], [identity],
        0, Int[], [1.23], my_ss,
    )
    spec2 = @test_logs (:warn, r"ss_fn was a Julia closure") _roundtrip(spec)
    @test spec2.ss_fn === nothing
    @test spec2.steady_state == [1.23]
end

@testset "DSER-03 sanitizer rejects run in steady_state" begin
    @test_throws SerializationError _MEM._sanitize_residual_ast(:(run("x")), :steady_state)
    @test_throws SerializationError _MEM._compile_ss_fn(:(run("x")), [:ρ], 1, false)
end

@testset "DSER-03 Dynare SS after save/load" begin
    dir = joinpath(@__DIR__, "..", "dynare_replication")
    tier_files = sort(filter(p -> startswith(basename(p), "tier") && endswith(p, ".jl"),
                             readdir(dir; join=true)))
    n_checked = Ref(0)
    for path in tier_files
        parsed = Meta.parseall(read(path, String); filename=path)
        blocks = _dser02_collect_dsge_blocks(parsed)
        isempty(blocks) && continue
        @testset "$(basename(path))" begin
            for block in blocks
                spec = @eval @dsge $block
                spec.ss_fn === nothing && continue
                θ = spec.param_values
                y0 = spec.ss_fn(θ)
                spec_ss = compute_steady_state(spec; method=:analytical)
                spec2 = _roundtrip(spec)
                @test spec2.ss_fn !== nothing
                @test spec2.ss_fn(θ) ≈ y0 atol=_DSER03_SS_ATOL
                spec2_ss = compute_steady_state(spec2; method=:analytical)
                @test spec2_ss.steady_state ≈ spec_ss.steady_state atol=_DSER03_SS_ATOL
                spec_ss2 = _roundtrip(spec_ss)
                @test spec_ss2.steady_state ≈ spec_ss.steady_state atol=_DSER03_SS_ATOL
                n_checked[] += 1
            end
        end
    end
    @test n_checked[] >= 20
end
