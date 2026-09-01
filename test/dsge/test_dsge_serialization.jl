# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, Random, LinearAlgebra, MacroEconometricModels

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
