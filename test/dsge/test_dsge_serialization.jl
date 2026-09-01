# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, Random, LinearAlgebra, Distributions, MacroEconometricModels

# Task 4 helpers are not on this commit — copy `_MEM` / `_roundtrip`
# from test/core/test_serialization.jl (do not include that file).
const _MEM = MacroEconometricModels
_roundtrip(m) = _MEM._reconstruct_from_container(_MEM._build_container(m))
const _suppress_warnings = _MEM._suppress_warnings

# Recursive equality that treats recompiled closures as matching and strips
# Expr line numbers (the Expr codec drops them).
function _deep_equal(a, b)
    a === nothing && return b === nothing
    a isa Missing && return b isa Missing
    a isa Function && return b isa Function
    if a isa Expr
        b isa Expr || return false
        return Base.remove_linenums!(deepcopy(a)) == Base.remove_linenums!(deepcopy(b))
    end
    if a isa Number
        return (isnan(a) && b isa Number && isnan(b)) || isequal(a, b)
    end
    (a isa AbstractString || a isa Symbol || a isa Enum || a isa Bool) && return isequal(a, b)
    if a isa AbstractArray
        b isa AbstractArray || return false
        size(a) == size(b) || return false
        return all(_deep_equal(a[i], b[i]) for i in eachindex(a))
    end
    if a isa AbstractDict
        b isa AbstractDict || return false
        Set(keys(a)) == Set(keys(b)) || return false
        return all(_deep_equal(a[k], b[k]) for k in keys(a))
    end
    if a isa NamedTuple
        b isa NamedTuple || return false
        keys(a) === keys(b) || keys(a) == keys(b) || return false
        return all(_deep_equal(a[k], b[k]) for k in keys(a))
    end
    a isa Tuple && return length(a) == length(b) && all(_deep_equal(a[i], b[i]) for i in eachindex(a))
    if isstructtype(typeof(a)) && parentmodule(typeof(a)) === _MEM
        typeof(a).name === typeof(b).name || return false
        return all(_deep_equal(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)))
    end
    return isequal(a, b)
end

function _assert_roundtrip(m; skip::Vector{Symbol}=Symbol[])
    m2 = _roundtrip(m)
    @test typeof(m2).name === typeof(m).name
    for f in fieldnames(typeof(m))
        f in skip && continue
        @test _deep_equal(getfield(m, f), getfield(m2, f))
    end
    return m2
end

function _dser04_rbc()
    spec = @dsge begin
        parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.03
        endogenous: k, c, z
        exogenous: ε
        z[t] = ρ * z[t-1] + σ * ε[t]
        k[t] = exp(z[t]) * k[t-1]^α + (1 - δ) * k[t-1] - c[t]
        1 / c[t] = β * (1 / c[t+1]) * (α * exp(z[t+1]) * k[t]^(α - 1) + 1 - δ)
    end
    return compute_steady_state(spec)
end

function _dser04_ar1()
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    return compute_steady_state(spec)
end

function _dser04_bellman_rbc()
    spec = @dsge begin
        parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.95, σ = 0.007
        endogenous: c, k, a
        exogenous: ε
        utility: log(c)
        beta: β
        controls: c
        euler: 1 / c[t] = β * (1 / c[t+1]) * (α * exp(a[t+1]) * k[t]^(α - 1) + 1 - δ)
        k[t] = exp(a[t]) * k[t-1]^α + (1 - δ) * k[t-1] - c[t]
        a[t] = ρ * a[t-1] + σ * ε[t]
    end
    return compute_steady_state(spec)
end

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

# =============================================================================
# DSER-04 — RA solution family (#762)
# =============================================================================

const _DSER04_TYPES = (
    "LinearDSGE", "DSGESolution", "PerturbationSolution", "ProjectionSolution",
    "PerfectForesightPath", "PrunedStateSpace", "DeterminacyMap",
    "KalmanSmootherResult", "DSGEEstimation", "OccBinConstraint",
    "OccBinRegime", "OccBinSolution", "OccBinIRF",
)

@testset "DSER-04 RA family registered" begin
    for name in _DSER04_TYPES
        @test haskey(_MEM._SERIALIZABLE_TYPES, name)
        @test string(nameof(_MEM._SERIALIZABLE_TYPES[name])) == name
    end
    if isdefined(_MEM, :_SERIALIZATION_EXCLUDED)
        for name in _DSER04_TYPES
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
    end
    F = cholesky(Matrix{Float64}(I, 2, 2))
    @test _MEM._ser_field(F) === nothing
end

@testset "DSER-04 LinearDSGE / DSGESolution solvers" begin
    spec = _dser04_rbc()
    ld = linearize(spec)
    @test ld isa LinearDSGE
    _assert_roundtrip(ld)
    ld2 = _roundtrip(ld)
    @test ld2.Gamma0 == ld.Gamma0 && ld2.Gamma1 == ld.Gamma1
    @test ld2.Psi == ld.Psi && ld2.Pi == ld.Pi && ld2.C == ld.C

    for method in (:gensys, :klein, :blanchard_kahn)
        sol = solve(spec; method=method)
        @test sol isa DSGESolution
        sol2 = _roundtrip(sol)
        @test sol2.G1 ≈ sol.G1 atol=0
        @test sol2.impact ≈ sol.impact atol=0
        @test sol2.C_sol ≈ sol.C_sol atol=0
        @test sol2.eu == sol.eu
        @test sol2.method == sol.method
        @test sol2.eigenvalues == sol.eigenvalues
        @test eltype(sol2.eigenvalues) <: Complex
        @test is_determined(sol2) == is_determined(sol)
        @test is_stable(sol2) == is_stable(sol)
        @test irf(sol2, 20).values == irf(sol, 20).values
        @test fevd(sol2, 20).proportions == fevd(sol, 20).proportions
        @test simulate(sol2, 50; rng=MersenneTwister(1)) == simulate(sol, 50; rng=MersenneTwister(1))
        @test analytical_moments(sol2) == analytical_moments(sol)
        @test sprint(show, sol2) == sprint(show, sol)
    end

    sol = solve(spec)
    mktemp() do p, _
        save_model(sol, p)
        sol3 = load_model(p)
        @test sol3 isa DSGESolution
        @test sol3.G1 ≈ sol.G1 atol=0
        @test sol3.impact ≈ sol.impact atol=0
        @test sol3.eigenvalues == sol.eigenvalues
    end
end

@testset "DSER-04 PerturbationSolution orders 1-3 + pruned" begin
    spec = _dser04_rbc()
    for order in 1:3
        psol = perturbation_solver(spec; order=order)
        @test psol isa PerturbationSolution
        @test psol.order == order
        p2 = _roundtrip(psol)
        @test p2.order == order
        @test p2.gx ≈ psol.gx atol=0
        @test p2.hx ≈ psol.hx atol=0
        @test p2.eta == psol.eta
        @test p2.steady_state == psol.steady_state
        if order >= 2
            @test p2.gxx == psol.gxx && p2.hxx == psol.hxx
        else
            @test p2.gxx === nothing && psol.gxx === nothing
        end
        if order == 3
            @test p2.gxxx == psol.gxxx
        end
    end
    pss = pruned_state_space(perturbation_solver(spec; order=2))
    @test pss isa PrunedStateSpace
    pss2 = _roundtrip(pss)
    @test pss2.hx_state == pss.hx_state
    @test pss2.gx_state == pss.gx_state
    @test pss2.hxx == pss.hxx && pss2.gxx == pss.gxx
    @test pss2.spec.n_endog == pss.spec.n_endog
end

@testset "DSER-04 ProjectionSolution tensor / Smolyak / PFI / VFI" begin
    ar1 = _dser04_ar1()
    tens = collocation_solver(ar1; grid=:tensor, degree=5, verbose=false)
    @test tens isa ProjectionSolution
    t2 = _roundtrip(tens)
    @test t2.coefficients == tens.coefficients
    @test t2.grid_type == :tensor
    @test t2.method == tens.method
    x = [0.01]
    @test evaluate_policy(t2, x) == evaluate_policy(tens, x)
    @test max_euler_error(t2; n_test=50, rng=MersenneTwister(762)) ==
          max_euler_error(tens; n_test=50, rng=MersenneTwister(762))
    mktemp() do p, _
        save_model(tens, p)
        t3 = load_model(p)
        @test t3 isa ProjectionSolution
        @test t3.coefficients == tens.coefficients
        @test evaluate_policy(t3, x) == evaluate_policy(tens, x)
    end

    rbc = _dser04_bellman_rbc()
    smol = collocation_solver(rbc; grid=:smolyak, smolyak_mu=1, max_iter=40, verbose=false)
    @test smol.grid_type == :smolyak
    s2 = _roundtrip(smol)
    @test s2.smolyak_levels == smol.smolyak_levels
    @test s2.coefficients == smol.coefficients
    xss = rbc.steady_state[smol.state_indices]
    @test evaluate_policy(s2, xss) == evaluate_policy(smol, xss)

    pfi = pfi_solver(ar1; next_state=:linear, degree=4, max_iter=40, verbose=false)
    @test pfi isa ProjectionSolution
    p2 = _roundtrip(pfi)
    @test p2.coefficients == pfi.coefficients
    @test evaluate_policy(p2, [0.01]) == evaluate_policy(pfi, [0.01])

    vfi = _suppress_warnings() do
        vfi_solver(rbc; next_state=:residual, degree=2, n_grid=6, max_iter=80,
                   howard_steps=5, n_choice=9, verbose=false)
    end
    @test vfi isa ProjectionSolution
    @test !isempty(vfi.value_fn) && !isempty(vfi.value_coefficients)
    v2 = _roundtrip(vfi)
    @test v2.value_fn == vfi.value_fn
    @test v2.value_coefficients == vfi.value_coefficients
    @test v2.method === vfi.method
end

@testset "DSER-04 PerfectForesightPath" begin
    spec = _dser04_ar1()
    shocks = zeros(30, 1); shocks[1, 1] = 1.0
    pf = _suppress_warnings() do
        solve(spec; method=:perfect_foresight, T_periods=30, shock_path=shocks)
    end
    @test pf isa PerfectForesightPath
    pf2 = _roundtrip(pf)
    @test pf2.path == pf.path && pf2.deviations == pf.deviations
    @test pf2.converged == pf.converged && pf2.iterations == pf.iterations
    p = plot_result(pf2)
    @test p isa PlotOutput
end

@testset "DSER-04 OccBin family" begin
    _suppress_warnings() do
        spec = @dsge begin
            parameters: rho = 0.9, phi = 1.5
            endogenous: y, i
            exogenous: e
            y[t] = rho * y[t-1] + e[t]
            i[t] = phi * y[t]
        end
        spec = compute_steady_state(spec)
        constraint = parse_constraint(:(i[t] >= 0), spec)
        @test constraint isa OccBinConstraint
        c2 = _roundtrip(constraint)
        @test c2.variable == constraint.variable
        @test c2.bound == constraint.bound
        @test c2.direction == constraint.direction
        @test _deep_equal(c2.expr, constraint.expr)
        @test _deep_equal(c2.bind_expr, constraint.bind_expr)

        A = [1.0 0.0; 0.0 1.0]; B = [0.5 0.1; 0.2 0.6]
        C = [0.3 0.0; 0.0 0.4]; D = [1.0 0.0; 0.0 1.0]
        regime = OccBinRegime{Float64}(A, B, C, D)
        r2 = _roundtrip(regime)
        @test r2.A == A && r2.B == B && r2.C == C && r2.D == D

        shock_path = zeros(20, spec.n_exog); shock_path[1, 1] = -2.0
        osol = occbin_solve(spec, constraint; shock_path=shock_path, nperiods=20)
        @test osol isa OccBinSolution
        o2 = _roundtrip(osol)
        @test o2.linear_path == osol.linear_path
        @test o2.piecewise_path == osol.piecewise_path
        @test o2.regime_history == osol.regime_history
        @test length(o2.constraints) == length(osol.constraints)
        @test o2.constraints[1].variable == osol.constraints[1].variable

        oirf = occbin_irf(spec, constraint, 1, 20; magnitude=-2.0)
        @test oirf isa OccBinIRF
        i2 = _roundtrip(oirf)
        @test i2.linear == oirf.linear && i2.piecewise == oirf.piecewise
        @test i2.regime_history == oirf.regime_history
        @test i2.shock_name == oirf.shock_name
        @test plot_result(i2) isa PlotOutput
    end
end

@testset "DSER-04 DeterminacyMap" begin
    nk = @dsge begin
        parameters: β = 0.99, σ_c = 1.0, κ = 0.3, φ_π = 1.5, φ_y = 0.5,
                    ρ_d = 0.8, σ_d = 0.01
        endogenous: y, π, R, d
        exogenous: ε_d
        y[t] = y[t+1] - (1 / σ_c) * (R[t] - π[t+1]) + d[t]
        R[t] = φ_π * π[t] + φ_y * y[t]
        π[t] = β * π[t+1] + κ * y[t]
        d[t] = ρ_d * d[t-1] + σ_d * ε_d[t]
    end
    nk = compute_steady_state(nk)
    grid = range(0.5, 2.0; length=5)
    m = determinacy_region(nk; params=:φ_π, grids=grid)
    @test m isa DeterminacyMap
    m2 = _roundtrip(m)
    @test m2.params == m.params
    @test m2.axes == m.axes
    @test m2.verdict == m.verdict
    @test m2.eu == m.eu
    @test m2.failures == m.failures
    @test m2.base_values == m.base_values
    @test m2.div == m.div && m2.method == m.method
    @test plot_result(m2) isa PlotOutput
    # Nested Vector{Vector{T}} is enough to infer the float parameter.
    @test _MEM._infer_float_param(([:φ_π], [[0.5, 1.0], [1.5]])) === Float64
end

@testset "DSER-04 KalmanSmootherResult (HD path)" begin
    spec = _dser04_ar1()
    sol = solve(spec)
    rng = MersenneTwister(762)
    sim = simulate(sol, 40; rng=rng)
    Z, d, H = _MEM._build_observation_equation(spec, [:y], nothing)
    ss = _MEM._build_state_space(sol, Z, d, H)
    data_dev = Matrix{Float64}(sim' .- sol.spec.steady_state)
    sm = dsge_smoother(ss, data_dev)
    @test sm isa KalmanSmootherResult
    sm2 = _roundtrip(sm)
    @test sm2.smoothed_states == sm.smoothed_states
    @test sm2.smoothed_shocks == sm.smoothed_shocks
    @test sm2.filtered_states == sm.filtered_states
    @test sm2.predicted_states == sm.predicted_states
    @test sm2.log_likelihood == sm.log_likelihood
    @test sm2.smoothed_covariances == sm.smoothed_covariances
end

@testset "DSER-04 DSGEEstimation Union variants" begin
    spec = _dser04_ar1()
    lin = solve(spec)
    pert = perturbation_solver(spec; order=1)
    proj = collocation_solver(spec; grid=:tensor, degree=3, verbose=false)
    theta = [0.9]; vcov = [0.01;;]; names = [:ρ]
    variants = (
        DSGEEstimation{Float64}(theta, vcov, names, :irf_matching, 0.0, 1.0, lin, true, spec),
        DSGEEstimation{Float64}(theta, vcov, names, :analytical_gmm, 0.0, 1.0, pert, true, spec),
        DSGEEstimation{Float64}(theta, vcov, names, :smm, 0.0, 1.0, proj, true, spec),
    )
    expected = (DSGESolution, PerturbationSolution, ProjectionSolution)
    for (est, S) in zip(variants, expected)
        @test est.solution isa S
        e2 = _roundtrip(est)
        @test e2 isa DSGEEstimation
        @test e2.solution isa S
        @test e2.theta == est.theta && e2.vcov == est.vcov
        @test e2.param_names == est.param_names
        @test e2.method == est.method
        @test e2.converged == est.converged
    end
    mktemp() do p, _
        save_model(variants[1], p)
        e3 = load_model(p)
        @test e3 isa DSGEEstimation
        @test e3.solution isa DSGESolution
        @test e3.theta == variants[1].theta
        @test e3.solution.G1 == lin.G1
    end

    _suppress_warnings() do
        rng = MersenneTwister(762)
        y = zeros(60)
        for t in 2:60
            y[t] = 0.8 * y[t-1] + 0.01 * randn(rng)
        end
        est = estimate_dsge(spec, reshape(y, :, 1), [:ρ];
                            method=:irf_matching, irf_horizon=4, var_lags=1, n_boot=5)
        @test est isa DSGEEstimation
        @test est.solution isa DSGESolution
        e2 = _roundtrip(est)
        @test e2.theta == est.theta
        @test e2.solution.G1 == est.solution.G1
    end
end

# =============================================================================
# DSER-05 — Bayesian DSGE + Distribution codec (#763)
# =============================================================================

const _DSER05_TYPES = (
    "DSGEPrior", "DSGEStateSpace", "NonlinearStateSpace", "ProjectionStateSpace",
    "BayesianDSGE", "PosteriorMode", "BayesianDSGESimulation", "MCMCDiagnostics",
    "IdentificationDiagnostics", "LearningRateCheck", "PriorPosteriorOverlap",
    "PriorPredictiveResult", "PosteriorPredictiveCheck", "PrefilterSpec",
    "ObservationTrends",
)

function _dser05_test_points(d)
    lo = minimum(d); hi = maximum(d)
    finite_lo = isfinite(lo) ? Float64(lo) : -5.0
    finite_hi = isfinite(hi) ? Float64(hi) : 5.0
    if finite_hi <= finite_lo
        finite_hi = finite_lo + 1.0
    end
    # Interior of the support so logpdf is finite for truncated / bounded priors.
    span = finite_hi - finite_lo
    return [finite_lo + span * f for f in (0.15, 0.30, 0.50, 0.70, 0.85)]
end

function _dser05_assert_dist(d1, d2)
    @test nameof(typeof(d1)) == nameof(typeof(d2))
    if d1 isa InverseGamma1
        @test d2 isa InverseGamma1
        @test d1.s == d2.s && d1.nu == d2.nu
    else
        @test collect(params(d1)) == collect(params(d2))
    end
    @test minimum(d1) == minimum(d2)
    @test maximum(d1) == maximum(d2)
    for x in _dser05_test_points(d1)
        @test logpdf(d1, x) == logpdf(d2, x)
    end
end

struct DSER05DummyDist <: Distribution{Univariate, Continuous} end

@testset "DSER-05 family registered" begin
    for name in _DSER05_TYPES
        @test haskey(_MEM._SERIALIZABLE_TYPES, name)
        @test string(nameof(_MEM._SERIALIZABLE_TYPES[name])) == name
    end
    if isdefined(_MEM, :_SERIALIZATION_EXCLUDED)
        for name in _DSER05_TYPES
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
    end
end

@testset "DSER-05 Distribution codec" begin
    ser, deser = _MEM._ser_field, _MEM._deser_field

    kinds = Any[
        dynare_prior(:normal, 0.0, 1.0),
        dynare_prior(:gamma, 1.5, 0.25),
        dynare_prior(:beta, 0.7, 0.1),
        dynare_prior(:inv_gamma, 0.02, 0.05),
        dynare_prior(:inv_gamma2, 0.02, 0.05),
        dynare_prior(:uniform, 0.5, 0.1),
        dynare_prior(:beta, 0.5, 0.1; lower=0.0, upper=0.9),
        InverseGamma(2, 3),
        truncated(Normal(0, 1), 0, 1),
        0.2 + 0.7 * Beta(2, 3),
        LogNormal(0.0, 1.0),
        Exponential(2.0),
        TDist(5.0),
        Cauchy(0.0, 1.0),
    ]
    for d in kinds
        enc = ser(d)
        @test enc isa AbstractDict
        @test haskey(enc, "__distribution__")
        _MEM._assert_plain_payload(enc)
        d2 = deser(enc)
        _dser05_assert_dist(d, d2)
    end

    mix = MixtureModel([Normal(), Normal(1, 1)])
    @test_throws SerializationError ser(mix)
    err = try
        ser(DSER05DummyDist())
        nothing
    catch e
        e
    end
    @test err isa SerializationError
    @test occursin("DSER05DummyDist", sprint(showerror, err))
end

@testset "DSER-05 DSGEPrior + state spaces" begin
    prior = _MEM.DSGEPrior(Dict(
        :ρ => dynare_prior(:beta, 0.7, 0.1),
        :σ => dynare_prior(:inv_gamma, 0.02, 0.05),
        :α => dynare_prior(:normal, 0.3, 0.05),
        :φ => InverseGamma(2, 3),
        :ψ => truncated(Normal(0.5, 0.2), 0.0, 1.0),
        :θ => 0.2 + 0.7 * Beta(2, 3),
    ))
    @test prior isa _MEM.DSGEPrior
    p2 = _assert_roundtrip(prior)
    @test p2 isa _MEM.DSGEPrior
    @test p2.param_names == prior.param_names
    for (d1, d2) in zip(prior.distributions, p2.distributions)
        _dser05_assert_dist(d1, d2)
    end

    spec = _dser04_ar1()
    sol = solve(spec)
    Z, d, H = _MEM._build_observation_equation(spec, [:y], [0.01])
    ss = _MEM._build_state_space(sol, Z, d, H)
    @test ss isa _MEM.DSGEStateSpace
    ss2 = _roundtrip(ss)
    @test ss2.G1 == ss.G1 && ss2.impact == ss.impact
    @test ss2.Z == ss.Z && ss2.d == ss.d && ss2.H == ss.H && ss2.Q == ss.Q
    @test ss2.H_inv ≈ ss.H_inv
    @test ss2.log_det_H ≈ ss.log_det_H

    psol = perturbation_solver(spec; order=2)
    nss = _MEM._build_nonlinear_state_space(psol, Z, d, H)
    @test nss isa _MEM.NonlinearStateSpace
    nss2 = _roundtrip(nss)
    @test nss2.hx == nss.hx && nss2.gx == nss.gx && nss2.order == 2
    @test nss2.H_inv ≈ nss.H_inv
    @test nss2.log_det_H ≈ nss.log_det_H

    proj = collocation_solver(spec; grid=:tensor, degree=3, verbose=false)
    pss = _MEM._build_projection_state_space(proj, Z, d, H)
    @test pss isa _MEM.ProjectionStateSpace
    pss2 = _roundtrip(pss)
    @test pss2.coefficients == pss.coefficients
    @test pss2.H_inv ≈ pss.H_inv
    @test pss2.log_det_H ≈ pss.log_det_H
end

@testset "DSER-05 companion types" begin
    pm = PosteriorMode{Float64}([0.5], [0.01;;], [100.0;;], -1.2, -0.8, -3.0,
                                [:ρ], true, 12)
    _assert_roundtrip(pm)

    diag = MCMCDiagnostics{Float64}([:ρ], [1.01], [50.0], [40.0], [0.1], [0.9],
                                    [0.5], [0.1], 30, :smc)
    _assert_roundtrip(diag)

    idd = IdentificationDiagnostics{Float64}([:ρ], [0.5], 1, 1, 3, 2,
                                             [1.0], 1e-8, zeros(1, 0), true)
    _assert_roundtrip(idd)

    lrc = LearningRateCheck{Float64}([:ρ], [50, 100], [0.1 0.05], [0.9], [false], 0.2)
    _assert_roundtrip(lrc)

    ppo = PriorPosteriorOverlap{Float64}([:ρ], [0.3], [false], 0.8)
    _assert_roundtrip(ppo)

    ppr = PriorPredictiveResult{Float64}(["mean_y"], reshape([0.1], 1, 1), 10, 8, 50)
    _assert_roundtrip(ppr)

    ppc = PosteriorPredictiveCheck{Float64}(["mean_y"], [0.1], reshape([0.0], 1, 1),
                                            [0.5], 10, 8)
    _assert_roundtrip(ppc)

    bsim = BayesianDSGESimulation{Float64}(
        zeros(5, 1, 4), zeros(5, 1), 5, ["y"],
        Float64[0.05, 0.16, 0.84, 0.95], zeros(3, 5, 1))
    _assert_roundtrip(bsim)

    Y = randn(MersenneTwister(763), 1, 20)
    _, pf = apply_prefilter(Y, :demean; observables=[:y])
    @test pf isa PrefilterSpec
    pf2 = _assert_roundtrip(pf)
    @test pf2.transform === :demean
    @test invert_prefilter(pf2, zeros(1, 2); time_offset=20) ==
          invert_prefilter(pf, zeros(1, 2); time_offset=20)

    tr = observation_trends(Dict(:y => (constant=1.0, linear=:g, quadratic=0.0)), [:y])
    @test tr.constants[1] == 1.0 && tr.linears[1] === :g
    tr2 = _assert_roundtrip(tr)
    @test tr2.constants[1] == 1.0
    @test tr2.linears[1] === :g
    mktemp() do p, _
        save_model(tr, p)
        tr3 = load_model(p)
        @test tr3 isa ObservationTrends
        @test tr3.constants[1] == 1.0
        @test tr3.linears[1] === :g
        @test eltype(tr3.constants) <: Union{AbstractFloat,Symbol}
    end
end

@testset "DSER-05 BayesianDSGE Kalman SMC + consumers" begin
    b = _suppress_warnings() do
        spec = _dser04_ar1()
        rng = MersenneTwister(763)
        sim = simulate(solve(spec), 40; rng=rng)
        priors = Dict(:ρ => Beta(2, 2))
        estimate_dsge_bayes(spec, sim, [0.5];
            priors=priors, method=:smc, observables=[:y],
            n_smc=30, n_mh_steps=1, ess_target=0.5,
            measurement_error=[0.01],
            rng=MersenneTwister(7631))
    end
    @test b isa BayesianDSGE
    @test b.state_space isa _MEM.DSGEStateSpace
    b2 = _assert_roundtrip(b; skip=[:state_space])
    @test b2.state_space.H_inv ≈ b.state_space.H_inv
    @test b2.state_space.log_det_H ≈ b.state_space.log_det_H
    @test b2.solver_kwargs == b.solver_kwargs

    @test posterior_summary(b2) == posterior_summary(b)
    @test marginal_likelihood(b2) == marginal_likelihood(b)
    @test bayes_factor(b2, b2) == 0.0
    @test prior_posterior_table(b2) == prior_posterior_table(b)
    @test trace(b2, :ρ) == trace(b, :ρ)
    @test sprint(show, b2) == sprint(show, b)

    rng = MersenneTwister(1)
    @test posterior_predictive(b2, 5; T_periods=8, rng=MersenneTwister(1)) ==
          posterior_predictive(b, 5; T_periods=8, rng=MersenneTwister(1))

    d1 = mcmc_diagnostics(b)
    d2 = mcmc_diagnostics(b2)
    @test d2.rhat == d1.rhat && d2.ess_bulk == d1.ess_bulk

    hd = historical_decomposition(b2, Matrix(b.data'), [:y]; mode_only=true, n_draws=5)
    @test hd isa HistoricalDecomposition

    birf = irf(b2, 5; n_draws=5, rng=MersenneTwister(2))
    @test birf isa BayesianImpulseResponse
    bsim = simulate(b2, 8; n_draws=5, rng=MersenneTwister(3))
    @test bsim isa BayesianDSGESimulation
    ppc = posterior_predictive_check(b2; n_draws=5, rng=MersenneTwister(4))
    @test ppc isa PosteriorPredictiveCheck
    idd = identification_diagnostics(b2.spec, b2.param_names; observables=b2.observables)
    @test idd isa IdentificationDiagnostics
    ppo = prior_posterior_overlap(b2)
    @test ppo isa PriorPosteriorOverlap
    @test plot_result(b2) isa PlotOutput

    mktemp() do p, _
        save_model(b, p)
        b3 = load_model(p)
        @test b3 isa BayesianDSGE
        @test sprint(show, b3) == sprint(show, b)
        @test b3.state_space.H_inv ≈ b.state_space.H_inv
        proj = dirname(Base.active_project())
        cmd = `$(Base.julia_cmd()) --project=$proj -e "using MacroEconometricModels; m = load_model(raw\"$p\"); print(sprint(show, m))"`
        loaded = read(cmd, String)
        @test loaded == sprint(show, b)
    end
end

@testset "DSER-05 NonlinearStateSpace / ProjectionStateSpace BayesianDSGE" begin
    _suppress_warnings() do
        spec = _dser04_ar1()
        data_obs = randn(MersenneTwister(42), 1, 24) .* 0.02
        priors = Dict(:ρ => Normal(0.5, 0.2))
        θ0 = [0.5]

        bp = estimate_dsge_bayes(spec, data_obs, θ0;
            priors=priors, method=:smc2, observables=[:y],
            n_smc=8, n_particles=20, n_mh_steps=1, ess_target=0.5,
            measurement_error=[0.005],
            solver=:perturbation, solver_kwargs=(order=2,),
            rng=MersenneTwister(7632))
        @test bp.state_space isa _MEM.NonlinearStateSpace
        bp2 = _assert_roundtrip(bp; skip=[:state_space])
        @test bp2.state_space isa _MEM.NonlinearStateSpace
        @test bp2.state_space.H_inv ≈ bp.state_space.H_inv
        @test bp2.solver_kwargs == bp.solver_kwargs

        bj = estimate_dsge_bayes(spec, data_obs, θ0;
            priors=priors, method=:smc2, observables=[:y],
            n_smc=8, n_particles=20, n_mh_steps=1, ess_target=0.5,
            measurement_error=[0.005],
            solver=:projection, solver_kwargs=(degree=3, scale=5.0),
            rng=MersenneTwister(7633))
        @test bj.state_space isa _MEM.ProjectionStateSpace
        bj2 = _assert_roundtrip(bj; skip=[:state_space])
        @test bj2.state_space isa _MEM.ProjectionStateSpace
        @test bj2.state_space.H_inv ≈ bj.state_space.H_inv
    end
end

@testset "DSER-05 prefilter= and trends= on BayesianDSGE" begin
    _suppress_warnings() do
        spec = _dser04_ar1()
        rng = MersenneTwister(7634)
        sim = simulate(solve(spec), 50; rng=rng)
        y_level = sim .+ 0.5 .+ 0.01 .* collect(1.0:50)
        priors = Dict(:ρ => Beta(2, 2))

        bpf = estimate_dsge_bayes(spec, y_level, [0.5];
            priors=priors, method=:mh, n_draws=40, burnin=10,
            observables=[:y], prefilter=:linear_detrend,
            warn_trends=false, rng=MersenneTwister(7635))
        @test bpf.prefilter isa PrefilterSpec
        bpf2 = _assert_roundtrip(bpf; skip=[:state_space])
        @test bpf2.prefilter isa PrefilterSpec
        @test bpf2.prefilter.transform === :linear_detrend
        @test bpf2.prefilter.slopes == bpf.prefilter.slopes

        btr = estimate_dsge_bayes(spec, y_level, [0.5];
            priors=priors, method=:mh, n_draws=40, burnin=10,
            observables=[:y],
            observation_trends=Dict(:y => (constant=0.5, linear=0.01)),
            warn_trends=false, rng=MersenneTwister(7636))
        @test btr.trends isa ObservationTrends
        btr2 = _assert_roundtrip(btr; skip=[:state_space])
        @test btr2.trends isa ObservationTrends
        @test btr2.trends.constants == btr.trends.constants
        @test btr2.trends.linears == btr.trends.linears
    end
end

# =============================================================================
# DSER-06 — CRRAUtility callable structs (#764)
# =============================================================================

# Named `Main` budget so the Function codec round-trips in this session.
function _dser06_named_budget(a::Float64, e::Float64, prices::Dict{Symbol,Float64})
    (1.0 + prices[:r]) * a + prices[:w] * e
end

const _DSER06_TYPES = ("HouseholdSystem", "IndividualProblem")

function _dser06_prices(hh)
    Dict{Symbol,Float64}(:r => 0.02, :w => 1.1, :div => 0.05,
                         :r_b => 0.01, :tau => 0.0)
end

function _dser06_assert_callables(hh, hh2)
    ip, ip2 = hh.individual, hh2.individual
    @test ip2.utility(1.3) == ip.utility(1.3)
    @test ip2.utility_prime(1.3) == ip.utility_prime(1.3)
    @test ip2.utility_prime_inv(0.7) == ip.utility_prime_inv(0.7)
    prices = _dser06_prices(hh)
    if ip.n_asset_dims == 1
        @test ip2.budget_fn(1.0, 1.0, prices) == ip.budget_fn(1.0, 1.0, prices)
    else
        @test ip2.budget_fn(1.0, 1.0, 1.0, prices) == ip.budget_fn(1.0, 1.0, 1.0, prices)
        @test ip2.adjustment_cost(0.1, 2.0) == ip.adjustment_cost(0.1, 2.0)
    end
    @test ip2.budget_fn === ip.budget_fn
    return hh2
end

@testset "DSER-06 CRRAUtility structs and household round-trip" begin
    @testset "registry" begin
        for name in _DSER06_TYPES
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test string(nameof(_MEM._SERIALIZABLE_TYPES[name])) == name
        end
        if isdefined(_MEM, :_SERIALIZATION_EXCLUDED)
            for name in _DSER06_TYPES
                @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
            end
        end
        @test CRRAUtility isa Type
        @test CRRAMarginalUtility isa Type
        @test CRRAInverseMarginalUtility isa Type
    end

    @testset "CRRA values match the 1e-15 floor and sigma==1 log branch" begin
        u, up, upi = _MEM._crra_utility(1.0)
        @test u isa CRRAUtility
        @test up isa CRRAMarginalUtility
        @test upi isa CRRAInverseMarginalUtility
        @test u(1.3) == log(1.3)
        @test up(1.3) == 1.0 / 1.3
        @test upi(0.7) == 1.0 / 0.7
        @test u(0.0) == log(1e-15)
        @test up(0.0) == 1.0 / 1e-15
        @test upi(0.0) == 1.0 / 1e-15

        u2, up2, upi2 = _MEM._crra_utility(2.0)
        @test u2(1.3) == 1.3^(1.0 - 2.0) / (1.0 - 2.0)
        @test up2(1.3) == 1.3^(-2.0)
        @test upi2(0.7) == 0.7^(-1.0 / 2.0)
        @test u2(0.0) == (1e-15)^(1.0 - 2.0) / (1.0 - 2.0)
        enc = _MEM._ser_field(u)
        @test enc isa AbstractDict && enc["__struct__"] == "CRRAUtility"
        _MEM._assert_plain_payload(enc)
        @test _MEM._deser_field(enc).sigma == u.sigma
    end

    @testset "HAGrid / IncomeProcess / LaborSupply are generic" begin
        grid = HAGrid(; assets=(0.0, 10.0, 5), income_states=2, grid_type=:linear)
        g2 = _MEM._deser_field(_MEM._ser_field(grid))
        @test g2 isa HAGrid
        @test g2.grids == grid.grids && g2.bounds == grid.bounds
        @test g2.total_individual_states == grid.total_individual_states

        inc = rouwenhorst(0.9, 0.1, 3)
        i2 = _MEM._deser_field(_MEM._ser_field(inc))
        @test i2 isa IncomeProcess
        @test i2.states == inc.states && i2.transition == inc.transition

        ls = LaborSupply(; kind=:ghh, psi=3.0, frisch=0.5)
        ls2 = _MEM._deser_field(_MEM._ser_field(ls))
        @test ls2 isa LaborSupply
        @test ls2.kind === :ghh && ls2.psi == ls.psi && ls2.frisch == ls.frisch
    end

    @testset "example households round-trip" begin
        for name in (:krusell_smith, :one_asset_hank, :two_asset_hank, :huggett)
            spec = load_ha_example(name)
            hh = first(values(spec.agents))
            hh2 = _assert_roundtrip(hh)
            _dser06_assert_callables(hh, hh2)
        end
        spec_el = _MEM._endogenous_labor_example()
        hh = first(values(spec_el.agents))
        hh2 = _assert_roundtrip(hh)
        _dser06_assert_callables(hh, hh2)
        @test hh2.individual.labor isa LaborSupply
        @test hh2.individual.labor.kind === hh.individual.labor.kind
    end

    @testset "reloaded household reproduces KS steady state" begin
        spec = load_ha_example(:krusell_smith)
        hh = first(values(spec.agents))
        hh2 = _roundtrip(hh)
        spec2 = _MEM._copy_model_spec(spec; agents=(household=hh2,))
        ss = compute_steady_state(spec)
        ss2 = compute_steady_state(spec2)
        @test ss2.converged && ss.converged
        for k in keys(ss.prices)
            @test ss2.prices[k] ≈ ss.prices[k] atol=1e-12
        end
        for k in keys(ss.aggregates)
            @test ss2.aggregates[k] ≈ ss.aggregates[k] atol=1e-12
        end
        @test ss2.distribution ≈ ss.distribution atol=1e-12
    end

    @testset "anonymous IndividualProblem is a SerializationError" begin
        ip = IndividualProblem{Float64}(
            c -> log(max(c, 1e-15)), c -> 1.0 / max(c, 1e-15),
            m -> 1.0 / max(m, 1e-15), 0.99,
            (a, e, prices) -> (1 + prices[:r]) * a + prices[:w] * e,
            [0.0], nothing, 1)
        err = try
            _MEM._ser_field(ip)
            nothing
        catch e
            e
        end
        @test err isa SerializationError
        msg = sprint(showerror, err)
        @test occursin("IndividualProblem.utility", msg)
        @test occursin("anonymous function", msg)
        @test occursin("CRRAUtility", msg)
    end

    @testset "Main-named budget function round-trips" begin
        u, up, upi = _MEM._crra_utility(1.0)
        ip = IndividualProblem{Float64}(u, up, upi, 0.99, _dser06_named_budget,
                                        [0.0], nothing, 1)
        ip2 = _assert_roundtrip(ip)
        prices = Dict{Symbol,Float64}(:r => 0.02, :w => 1.1)
        @test ip2.budget_fn === _dser06_named_budget
        @test ip2.budget_fn(1.0, 1.0, prices) == ip.budget_fn(1.0, 1.0, prices)
        @test ip2.utility(1.3) == ip.utility(1.3)
    end
end

# =============================================================================
# DSER-07 — HA results (#765)
# =============================================================================

const _DSER07_TYPES = (
    "HASteadyState", "HADSGESolution", "KrusellSmithSolution",
    "WinberryFamily", "DenHaanAccuracy", "HAGridDiagnostics",
    "HAGrid", "IncomeProcess",
    "HouseholdSystem", "IndividualProblem",
)

_dser07_huggett() = _MEM._huggett_example(; credit_limit=-2.0, a_max=8.0, n_a=40)

function _dser07_two_huggett()
    s1 = _MEM._huggett_example(; credit_limit=-2.0, a_max=6.0, n_a=20, beta=0.96)
    s2 = _MEM._huggett_example(; credit_limit=-2.0, a_max=6.0, n_a=20, beta=0.985)
    hh_u = only(values(s1.agents))
    hh_h = only(values(s2.agents))
    return ModelSpec{Float64}(
        Symbol[], Symbol[], s1.params, copy(s1.param_values),
        NamedEquation[], Function[], 0, Int[], Float64[];
        agents=(unconstrained=hh_u, htm=hh_h))
end

function _dser07_assert_ss_types(ss)
    T = eltype(ss.distribution)
    @test ss.policies isa Dict{Symbol,Array{T}}
    @test ss.prices isa Dict{Symbol,T}
    @test ss.aggregates isa Dict{Symbol,T}
    @test ss.grid isa HAGrid{T}
    @test ss.income isa IncomeProcess{T}
    for v in values(ss.policies)
        @test v isa Array{T}
    end
end

@testset "DSER-07 HA results registered" begin
    for name in _DSER07_TYPES
        @test haskey(_MEM._SERIALIZABLE_TYPES, name)
        @test string(nameof(_MEM._SERIALIZABLE_TYPES[name])) == name
    end
    if isdefined(_MEM, :_SERIALIZATION_EXCLUDED)
        for name in _DSER07_TYPES
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
    end
end

@testset "DSER-07 HAGrid / IncomeProcess standalone save_model" begin
    grid = HAGrid(; assets=(0.0, 10.0, 5), income_states=2, grid_type=:linear)
    g2 = _assert_roundtrip(grid)
    @test g2.total_individual_states == grid.total_individual_states
    mktemp() do p, _
        save_model(grid, p)
        g3 = load_model(p)
        @test g3 isa HAGrid
        @test g3.grids == grid.grids
    end

    inc = rouwenhorst(0.9, 0.1, 3)
    i2 = _assert_roundtrip(inc)
    @test i2.states == inc.states && i2.transition == inc.transition
    mktemp() do p, _
        save_model(inc, p)
        i3 = load_model(p)
        @test i3 isa IncomeProcess
        @test i3.labels === inc.labels
    end
end

@testset "DSER-07 WinberryFamily / DenHaanAccuracy / HAGridDiagnostics" begin
    pd = ParametricDensity{Float64}([0.1, 0.2], [1.0, 0.5], 1.0, 0.5, 0.0, true, 3, 1e-8)
    wf = WinberryFamily{Float64}([pd, pd], [0.6, 0.4], 2, [0.0, 1.0], [0.5, 0.5],
                                 (0.0, 1.0), true)
    wf2 = _assert_roundtrip(wf)
    @test wf2.densities isa Vector{<:ParametricDensity}
    @test wf2.n_moments == 2
    @test wf2.bounds == (0.0, 1.0)
    @test wf2.densities[1].lambda == pd.lambda

    dh = DenHaanAccuracy{Float64}(:K, 0.02, 0.01, 0.1, 0.09,
                                  [1.0, 1.1, 1.05], [1.0, 1.08, 1.04],
                                  3, 1; source=:plm)
    dh2 = _assert_roundtrip(dh)
    @test dh2.aggregate === :K
    @test dh2.source === :plm
    @test dh2.ref_path == dh.ref_path
    @test dh2.dh_max == dh.dh_max
    report(dh2)

    gd = HAGridDiagnostics{Float64}(0.0, 10.0, 5, 2, 0.0, 0.1, 0, 0,
                                    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 9.0,
                                    1e-6, 1e-6, true)
    gd2 = _assert_roundtrip(gd)
    @test gd2.adequate && gd2.n_a == 5
    @test gd2.a_max == 10.0
end

@testset "DSER-07 ModelSpec.agents round-trip" begin
    for name in (:krusell_smith, :one_asset_hank, :two_asset_hank, :huggett)
        spec = load_ha_example(name)
        spec2 = _roundtrip(spec)
        @test has_kind(spec2, HouseholdSystem)
        @test keys(spec2.agents) == keys(spec.agents)
        hh = first(values(spec.agents))
        hh2 = first(values(spec2.agents))
        @test hh2 isa HouseholdSystem
        @test hh2.model === hh.model
        @test hh2.individual.beta == hh.individual.beta
    end

    spec_w = load_ha_example(:krusell_smith; distribution=:winberry)
    spec_w2 = _roundtrip(spec_w)
    @test first(values(spec_w2.agents)).distribution === :winberry

    mp = _dser07_two_huggett()
    mp2 = _roundtrip(mp)
    @test keys(mp2.agents) == keys(mp.agents)
    @test collect(keys(mp2.agents)) == [:unconstrained, :htm]
    @test has_kind(mp2, HouseholdSystem)
    @test length(collect(agents_of(mp2, HouseholdSystem))) == 2
end

@testset "DSER-07 HASteadyState round-trip" begin
    _suppress_warnings() do
        for name in (:krusell_smith, :one_asset_hank, :two_asset_hank, :huggett)
            spec = if name === :two_asset_hank
                _MEM._two_asset_hank_example(; n_liquid=6, n_illiquid=5, n_e=2, B_supply=1.0)
            elseif name === :huggett
                _dser07_huggett()
            else
                load_ha_example(name)
            end
            ss_kw = name === :two_asset_hank ?
                (max_iter=15, tol=5e-2, grid_check=:none) :
                (max_iter=80, tol=1e-4, grid_check=:none)
            ss = compute_steady_state(spec; ss_kw...)
            @test ss isa HASteadyState
            ss2 = _assert_roundtrip(ss)
            _dser07_assert_ss_types(ss2)
            @test typeof(ss2.policies) === typeof(ss.policies)
            @test typeof(ss2.prices) === typeof(ss.prices)
            @test typeof(ss2.aggregates) === typeof(ss.aggregates)
            @test ss2.converged == ss.converged
            report(ss2)
            if ss.grid.n_dims == 1
                @test plot_result(ss2; view=:distribution) isa PlotOutput
                @test plot_result(ss2; view=:lorenz) isa PlotOutput
                @test plot_result(ss2; view=:policy) isa PlotOutput
                gd = ha_grid_diagnostics(ss)
                gd2 = _assert_roundtrip(gd)
                @test gd2.n_a == gd.n_a
            end
        end

        spec_w = _MEM._replace_household(_dser07_huggett(); distribution=:winberry)
        ss_w = compute_steady_state(spec_w; max_iter=40, tol=1e-4, grid_check=:none)
        @test ss_w.parametric isa WinberryFamily
        ss_w2 = _assert_roundtrip(ss_w)
        @test ss_w2.parametric isa WinberryFamily
        @test ss_w2.parametric.n_moments == ss_w.parametric.n_moments
        @test ss_w2.parametric.mass == ss_w.parametric.mass
    end
end

@testset "DSER-07 reloaded spec reproduces SS and SSJ G1" begin
    _suppress_warnings() do
        spec = _dser07_huggett()
        spec2 = _roundtrip(spec)
        @test has_kind(spec2, HouseholdSystem)
        @test keys(spec2.agents) == keys(spec.agents)
        ss = compute_steady_state(spec; max_iter=40, tol=1e-6, grid_check=:none)
        ss2 = compute_steady_state(spec2; max_iter=40, tol=1e-6, grid_check=:none)
        @test ss2.converged && ss.converged
        for k in keys(ss.prices)
            @test ss2.prices[k] ≈ ss.prices[k] atol=1e-12
        end
        for k in keys(ss.aggregates)
            @test ss2.aggregates[k] ≈ ss.aggregates[k] atol=1e-12
        end
        @test ss2.distribution ≈ ss.distribution atol=1e-12

        sol = solve(spec; method=:ssj, ss=ss, T_horizon=12, n_reduced=4)
        sol_from_spec2 = solve(spec2; method=:ssj, ss=ss2, T_horizon=12, n_reduced=4)
        @test sol_from_spec2.linear_solution.G1 ≈ sol.linear_solution.G1 atol=1e-10
    end
end

@testset "DSER-07 HADSGESolution ssj/reiter + consumers" begin
    _suppress_warnings() do
        spec = _dser07_huggett()
        ss = compute_steady_state(spec; max_iter=40, tol=1e-4, grid_check=:none)
        for method in (:ssj, :reiter)
            sol = if method === :ssj
                solve(spec; method=:ssj, ss=ss, T_horizon=12, n_reduced=4)
            else
                solve(spec; method=:reiter, ss=ss, n_reduced=4)
            end
            @test sol isa HADSGESolution
            sol2 = _assert_roundtrip(sol)
            @test sol2.method === sol.method
            @test sol2.linear_solution.G1 == sol.linear_solution.G1
            @test sol2.C_obs == sol.C_obs && sol2.D_obs == sol.D_obs
            ir = irf(sol, 20)
            ir2 = irf(sol2, 20)
            @test ir2.values == ir.values
            @test ir2.variables == ir.variables
            fv = fevd(sol, 12)
            fv2 = fevd(sol2, 12)
            @test fv2.proportions == fv.proportions
            sim = simulate(sol, 8; shock_draws=zeros(8, nshocks(sol)))
            sim2 = simulate(sol2, 8; shock_draws=zeros(8, nshocks(sol2)))
            @test sim2 == sim
        end

        sol = solve(spec; method=:ssj, ss=ss, T_horizon=12, n_reduced=4)
        mktemp() do p, _
            save_model(sol, p)
            sol3 = load_model(p)
            @test sol3 isa HADSGESolution
            @test sol3.linear_solution.G1 == sol.linear_solution.G1
            @test irf(sol3, 20).values == irf(sol, 20).values
            report(sol3)
        end
    end
end

@testset "DSER-07 KrusellSmithSolution round-trip" begin
    _suppress_warnings() do
        spec = _dser07_huggett()
        ss = compute_steady_state(spec; max_iter=40, tol=1e-4, grid_check=:none)
        ks = solve(spec; method=:krusell_smith, ss=ss, T_sim=40, T_burn=8, max_outer=1)
        @test ks isa KrusellSmithSolution
        ks2 = _assert_roundtrip(ks)
        @test ks2.plm_coefficients isa Dict{Symbol,Vector{Float64}}
        @test ks2.r_squared isa Dict{Symbol,Float64}
        @test ks2.plm_coefficients == ks.plm_coefficients
        @test ks2.r_squared == ks.r_squared
        @test ks2.converged == ks.converged
        report(ks2)
    end
end

# =============================================================================
# DSER-08 — SSJ block objects (#766)
# =============================================================================

const _DSER08_TYPES = (
    "SSJModel", "SimpleBlock", "HetBlock", "MitBlock",
    "SSJGEJacobian", "SSJImpulseResponse",
)

# Named Main callables so SimpleBlock / MitBlock round-trip (Function codec).
_dser08_identity(x) = [x[1]]
_dser08_asset_mkt(x) = [x[1] - x[2]]
function _dser08_ks_firm(x)
    α, δ = 0.36, 0.025
    K, Z = x[1], x[2]
    [α * Z * K^(α - 1) - δ, (1 - α) * Z * K^α, Z * K^α]
end
function _dser08_mit_eval(paths, Th)
    r = paths[:r]
    Dict(:A => copy(r), :C => 0.5 .* r)
end

_dser08_report_text(obj) = sprint(report, obj)

function _dser08_plot_ir(r::SSJImpulseResponse)
    vars = sort(collect(keys(r.paths)); by=string)
    H = r.T_horizon
    n = length(vars)
    T = eltype(first(values(r.paths)))
    vals = Array{T}(undef, H, n, 1)
    for (j, v) in enumerate(vars)
        vals[:, j, 1] = r.paths[v]
    end
    shock = isempty(r.shocks) ? "shock" : string(first(sort(collect(keys(r.shocks)); by=string)))
    ir = ImpulseResponse{T}(vals, zeros(T, H, n, 1), zeros(T, H, n, 1),
                            H, string.(vars), [shock], :none)
    return plot_result(ir)
end

@testset "DSER-08 SSJ types registered" begin
    for name in _DSER08_TYPES
        @test haskey(_MEM._SERIALIZABLE_TYPES, name)
        @test string(nameof(_MEM._SERIALIZABLE_TYPES[name])) == name
    end
    if isdefined(_MEM, :_SERIALIZATION_EXCLUDED)
        for name in _DSER08_TYPES
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
    end
end

@testset "DSER-08 lambda SimpleBlock is a SerializationError" begin
    blk = SimpleBlock(x -> [x[1]]; inputs=[:a], outputs=[:b],
                      ss_inputs=Dict(:a => 1.0), name=:firm)
    err = try
        _MEM._ser_field(blk)
        nothing
    catch e
        e
    end
    @test err isa SerializationError
    msg = sprint(showerror, err)
    @test occursin("SimpleBlock :firm", msg)
    @test occursin("anonymous", msg)
end

@testset "DSER-08 SimpleBlock / MitBlock named functions" begin
    sb = SimpleBlock(_dser08_identity; inputs=[:a], outputs=[:b],
                     ss_inputs=Dict(:a => 1.5), name=:id)
    sb2 = _assert_roundtrip(sb)
    @test sb2 isa SimpleBlock
    @test sb2.f === _dser08_identity
    @test sb2.ss_outputs[:b] == sb.ss_outputs[:b]
    @test sb2.arg_order == sb.arg_order
    @test sb2.f([2.0]) == sb.f([2.0])

    mb = MitBlock(_dser08_mit_eval, Float64;
                  inputs=[:r, :w], outputs=[:A, :C],
                  ss_inputs=Dict(:r => 0.02, :w => 1.0),
                  ss_outputs=Dict(:A => 0.02, :C => 0.01),
                  name=:lc)
    mb2 = _assert_roundtrip(mb)
    @test mb2 isa MitBlock
    @test mb2.evaluate === _dser08_mit_eval
    @test mb2.ss_outputs[:A] == mb.ss_outputs[:A]
    Th = 3
    paths = Dict(:r => fill(0.02, Th), :w => fill(1.0, Th))
    @test mb2.evaluate(paths, Th)[:A] == mb.evaluate(paths, Th)[:A]
end

@testset "DSER-08 SSJModel / Jacobian / IRF named DAG" begin
    _suppress_warnings() do
        b1 = SimpleBlock(_dser08_identity; inputs=[:u], outputs=[:y],
                         ss_inputs=Dict(:u => 1.0), name=:one)
        b2 = SimpleBlock(_dser08_asset_mkt; inputs=[:y, :z], outputs=[:q],
                         ss_inputs=Dict(:y => 1.0, :z => 1.0), name=:two)
        model = combine_blocks(b1, b2; name=:toy)
        @test model.blocks isa Vector{AbstractSSJBlock}
        model2 = _assert_roundtrip(model)
        @test model2.blocks isa Vector{AbstractSSJBlock}
        @test eltype(model2.blocks) === AbstractSSJBlock
        @test [b.name for b in model2.blocks] == [b.name for b in model.blocks]
        @test _dser08_report_text(model2) == _dser08_report_text(model)

        gej = ssj_jacobian(model; unknowns=[:u], targets=[:q], shocks=[:z],
                           T_horizon=8)
        gej2 = _assert_roundtrip(gej; skip=[:H_U_fact])
        @test gej2.H_U == gej.H_U
        @test gej2.H_Z == gej.H_Z
        @test typeof(gej2.curlyJ) === typeof(gej.curlyJ)
        rng = MersenneTwister(766)
        v = randn(rng, size(gej.H_U, 1))
        @test gej2.H_U_fact \ v ≈ gej.H_U_fact \ v atol=1e-12

        dz = Dict(:z => [0.01 * 0.9^(t - 1) for t in 1:8])
        ir = ssj_irf(gej, dz; residual=false)
        ir_from2 = ssj_irf(gej2, dz; residual=false)
        @test ir_from2.paths[:u] ≈ ir.paths[:u] atol=1e-12
        @test ir_from2.paths[:y] ≈ ir.paths[:y] atol=1e-12
        ir2 = _assert_roundtrip(ir)
        @test ir2.paths[:u] == ir.paths[:u]
        @test ir2.order == ir.order
        @test _dser08_plot_ir(ir2) isa PlotOutput
    end
end

@testset "DSER-08 Huggett HetBlock DAG (named bond market)" begin
    _suppress_warnings() do
        spec = _dser07_huggett()
        ss = compute_steady_state(spec; max_iter=40, tol=1e-4, grid_check=:none)
        hh = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A], name=:household)
        hh2 = _assert_roundtrip(hh)
        @test hh2 isa HetBlock
        @test hh2.ss_outputs[:A] ≈ hh.ss_outputs[:A] atol=1e-12

        bond = SimpleBlock(_dser08_identity;
                           inputs=[:A], outputs=[:bond_mkt],
                           ss_inputs=Dict(:A => hh.ss_outputs[:A]),
                           name=:bond_market)
        model = combine_blocks(hh, bond; name=:huggett)
        gej = ssj_jacobian(model; unknowns=[:r], targets=[:bond_mkt], shocks=[:w],
                           T_horizon=12, target_tol=1e-2)
        gej2 = _assert_roundtrip(gej; skip=[:H_U_fact])
        @test gej2.model.blocks isa Vector{AbstractSSJBlock}
        @test gej2.H_U ≈ gej.H_U atol=1e-12
        rng = MersenneTwister(7661)
        v = randn(rng, size(gej.H_U, 1))
        @test gej2.H_U_fact \ v ≈ gej.H_U_fact \ v atol=1e-10
        dw = Dict(:w => [0.02 * 0.9^(t - 1) for t in 1:12])
        ir = ssj_irf(gej, dw; residual=false)
        ir2 = ssj_irf(gej2, dw; residual=false)
        @test ir2.paths[:r] ≈ ir.paths[:r] atol=1e-10
        @test _dser08_report_text(model) == _dser08_report_text(_roundtrip(model))
    end
end
