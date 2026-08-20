# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# #659 — sparse colored PF Jacobian, cached CSC, block-tridiagonal solve.
# Regression copies of the AR(1) / zero-shock / multi-variable PF cases from
# test/dsge/test_dsge.jl (L2466–2524). Do not include that file here.

using Test
using LinearAlgebra
using SparseArrays
using MacroEconometricModels
import NonlinearSolve

const _MEM = MacroEconometricModels

# Chain of n neighbor-coupled AR(1)s with a contemporaneous tridiagonal block,
# a diagonal lag block, and a diagonal lead block.
# n=16 T=40: structural nnz = T*(3n-2) + (T-1)*n + (T-1)*n
#           = 40*46 + 39*16 + 39*16 = 1840 + 1248 = 3088
# Dense-block bound 3*T*n^2 = 30720.
function _sparse_chain_spec(n::Int)
    endog = [Symbol(:y, i) for i in 1:n]
    exog = [:ε]
    params = [:ρ, :α, :γ, :σ]
    θ = Dict(:ρ => 0.5, :α => 0.1, :γ => 0.05, :σ => 1.0)
    fns = Function[
        let i = i
            (y_t, y_lag, y_lead, ε_t, th) -> begin
                r = y_t[i] - th[:ρ] * y_lag[i] - th[:γ] * y_lead[i]
                i > 1 && (r -= th[:α] * y_t[i - 1])
                i < n && (r -= th[:α] * y_t[i + 1])
                i == 1 && (r -= th[:σ] * ε_t[1])
                return r
            end
        end
        for i in 1:n
    ]
    eqs = [Expr(:call, :(-), Expr(:ref, endog[i], :t), 0) for i in 1:n]
    return ModelSpec{Float64}(endog, exog, params, θ, eqs, fns, 0, Int[], zeros(n))
end

@testset "PF regression: AR(1) impulse" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    T_periods = 50
    shocks = zeros(T_periods, 1)
    shocks[1, 1] = 1.0
    pf = solve(spec; method=:perfect_foresight, T_periods=T_periods, shock_path=shocks)
    @test pf isa PerfectForesightPath{Float64}
    @test pf.converged
    @test size(pf.path) == (T_periods, 1)
    # Analytical: y_t = ρ^{t-1}  (y_0 = 0, unit shock at t=1)
    @test pf.deviations[1, 1] ≈ 1.0 atol=1e-8
    @test pf.deviations[2, 1] ≈ 0.9 atol=1e-8
    for t in 1:T_periods
        @test pf.deviations[t, 1] ≈ 0.9^(t - 1) atol=1e-8
    end
    @test abs(pf.deviations[end, 1]) < 0.01
end

@testset "PF regression: zero shocks = steady state" begin
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
    @test all(abs.(pf.deviations) .< 1e-8)
end

@testset "PF regression: multi-variable" begin
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
    shocks[1, 1] = 1.0
    pf = solve(spec; method=:perfect_foresight, T_periods=T_periods, shock_path=shocks)
    @test pf isa PerfectForesightPath{Float64}
    @test pf.converged
    @test size(pf.path) == (T_periods, 2)
    @test pf.deviations[1, 1] ≈ 1.0 atol=1e-8
    @test pf.deviations[1, 2] ≈ 0.5 atol=1e-8
    @test pf.deviations[2, 1] ≈ 0.8 atol=1e-8
    @test pf.deviations[2, 2] ≈ 0.4 atol=1e-8
end

@testset "PF sparse Jacobian: chain n=16 T=40" begin
    n = 16
    Tp = 40
    spec = _sparse_chain_spec(n)
    shocks = zeros(Tp, 1)
    shocks[1, 1] = 1.0
    x0 = repeat(spec.steady_state, Tp)

    cache = _MEM._pf_make_cache(spec, Tp)
    J = cache.J
    @test J isa SparseMatrixCSC
    # nnz comment: structural pattern is contemporaneous tridiagonal (3n-2),
    # plus diagonal lag and lead on interior periods. For n=16 T=40:
    #   nnz = 40*(3*16-2) + 39*16 + 39*16 = 3088
    #   dense-block bound 3*T*n^2 = 30720
    dense_bound = 3 * Tp * n * n
    @test nnz(J) < dense_bound ÷ 2
    @test nnz(J) == Tp * (3n - 2) + 2 * (Tp - 1) * n

    _MEM._pf_assemble_jacobian!(J, x0, spec, shocks, Tp, cache)
    F = zeros(Tp * n)
    _MEM._pf_residual_packed!(F, x0, spec, shocks, Tp)
    u_bt = zeros(Tp * n)
    @test _MEM._pf_bt_solve!(u_bt, cache, F)
    u_lu = J \ F
    @test u_bt ≈ u_lu rtol=1e-8 atol=1e-8

    pf = perfect_foresight(spec; T_periods=Tp, shock_path=shocks)
    @test pf.converged
    @test size(pf.path) == (Tp, n)
    # First variable absorbs the shock; neighbors respond contemporaneously.
    @test abs(pf.deviations[1, 1]) > 0.5
    @test maximum(abs, pf.deviations[end, :]) < 1e-6
end

@testset "PF sparse Jacobian: in-place CSC pattern is stable" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    Tp = 10
    shocks = zeros(Tp, 1)
    shocks[1, 1] = 1.0
    cache = _MEM._pf_make_cache(spec, Tp)
    J1 = cache.J
    ptr1 = copy(J1.colptr)
    rv1 = copy(J1.rowval)
    x = repeat(spec.steady_state, Tp)
    _MEM._pf_assemble_jacobian!(J1, x, spec, shocks, Tp, cache)
    @test J1.colptr == ptr1
    @test J1.rowval == rv1
    # Second fill must reuse the same nzval buffer.
    nz_ptr = pointer(nonzeros(J1))
    _MEM._pf_assemble_jacobian!(J1, x, spec, shocks, Tp, cache)
    @test pointer(nonzeros(J1)) == nz_ptr
    @test J1.colptr == ptr1
end

@testset "PF sparse Jacobian: NonlinearSolve algorithm kwarg still works" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    shocks = zeros(30, 1)
    shocks[1, 1] = 1.0
    pf = solve(spec; method=:perfect_foresight, T_periods=30, shock_path=shocks,
               algorithm=NonlinearSolve.NewtonRaphson())
    @test pf.converged
    @test pf.deviations[1, 1] ≈ 1.0 atol=1e-8
end

# Windows CI is a single process with JULIA_NUM_THREADS=auto and @spawn
# workers. @threads :static cannot run concurrently, so two residual /
# Jacobian fills from different tasks must not throw.
@testset "PF residual is safe under concurrent callers" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    Tp = 20
    shocks = zeros(Tp, 1)
    n = spec.n_endog
    function _one()
        x = repeat(spec.steady_state, Tp)
        F = zeros(n * Tp)
        _MEM._pf_residual_packed!(F, x, spec, shocks, Tp)
        cache = _MEM._pf_make_cache(spec, Tp)
        _MEM._pf_compute_blocks!(cache, x, spec, shocks, Tp)
        return F
    end
    results = Vector{Vector{Float64}}(undef, 2)
    @sync begin
        Threads.@spawn (results[1] = _one())
        Threads.@spawn (results[2] = _one())
    end
    @test results[1] == results[2]
end

@testset "PF sparsity: log residual with small SS (MSR-04)" begin
    fn = (y_t, y_lag, y_lead, ε_t, θ) ->
        log(y_t[1]) - log(θ[:rss]) - θ[:ρ] * (log(y_lag[1]) - log(θ[:rss])) - ε_t[1]
    spec = ModelSpec{Float64}(
        [:r], [:ε], [:ρ, :rss], Dict(:ρ => 0.5, :rss => 0.05),
        [:(log(r[t]) - log(rss))], [fn], 0, Int[], [0.05])
    pf = perfect_foresight(spec; T_periods=8)
    @test all(isfinite, pf.path)
end

@testset "PF sparsity=:dense matches :auto on a smooth model (MSR-04)" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
    end
    spec = compute_steady_state(spec)
    Tp = 20
    shocks = zeros(Tp, 1)
    shocks[1, 1] = 1.0
    pf_s = perfect_foresight(spec; T_periods=Tp, shock_path=shocks, sparsity=:auto)
    pf_d = perfect_foresight(spec; T_periods=Tp, shock_path=shocks, sparsity=:dense)
    @test maximum(abs, pf_s.path .- pf_d.path) < 1e-8
end

@testset "PF kink succeeds with sparsity=:dense (MSR-04)" begin
    fn = (y_t, y_lag, y_lead, ε_t, θ) ->
        y_t[1] - θ[:ρ] * y_lag[1] - max(y_t[1] - 0.5, 0.0) * 0.1 - ε_t[1]
    spec = ModelSpec{Float64}(
        [:y], [:ε], [:ρ], Dict(:ρ => 0.5),
        [:(y[t] - ρ * y[t-1])], [fn], 0, Int[], [0.0])
    pf = perfect_foresight(spec; T_periods=10, sparsity=:dense)
    @test all(isfinite, pf.path)
end
