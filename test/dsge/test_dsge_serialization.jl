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
