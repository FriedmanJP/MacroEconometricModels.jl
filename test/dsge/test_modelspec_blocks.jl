# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# G-16 / #650 — HetBlock / MitBlock from non-household AbstractAgentSystem.

using Test
using MacroEconometricModels
using LinearAlgebra

function _cd_firm_block(K_ss, L, Z, alpha, delta)
    SimpleBlock(
        x -> begin
            Klag, Zt = x[1], x[2]
            kl = Klag / L
            r = alpha * Zt * kl^(alpha - 1) - delta
            w = (1 - alpha) * Zt * kl^alpha
            Y = Zt * Klag^alpha * L^(1 - alpha)
            [r, w, Y]
        end;
        inputs=[:K, :Z], outputs=[:r, :w, :Y],
        lags=Dict(:K => [1]),
        ss_inputs=Dict(:K => K_ss, :Z => Z),
        name=:firm)
end

@testset "G-16: MitBlock from LifeCycleSystem (#650)" begin
    inc = MacroEconometricModels.IncomeProcess{Float64}(ones(1, 1), [1.0], [1.0], :income)
    m = LifeCycleOLG(; J=16, J_retire=12, survival=0.995,
                     income=inc, a_max=40.0, n_a=40, beta=0.97, sigma=2.0,
                     replacement=0.0, n_pop=0.0, annuities=true)
    spec = to_spec(m)
    @test MacroEconometricModels.has_kind(spec, LifeCycleSystem)
    ss = lifecycle_steady_state(m; r_bounds=(-0.01, 0.15), tol=1e-4, max_iter=40)
    @test ss.converged
    @test ss.K > 0 && ss.L > 0

    hh = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A, :C], name=:households)
    @test hh isa MitBlock
    @test hh isa AbstractSSJBlock
    @test hh.ss_inputs[:r] == ss.r
    @test hh.ss_inputs[:w] == ss.w
    @test isfinite(hh.ss_outputs[:A]) && isfinite(hh.ss_outputs[:C])
    @test occursin("MitBlock", sprint(show, hh))

    Th = 6
    flat = Dict(:r => fill(ss.r, Th), :w => fill(ss.w, Th))
    base = MacroEconometricModels._block_evaluate(hh, flat, Th)
    @test all(isfinite, base[:A]) && all(isfinite, base[:C])
    @test maximum(abs, base[:A] .- hh.ss_outputs[:A]) < 0.25 * abs(hh.ss_outputs[:A])

    Jb = block_jacobian(hh, Th)
    @test Set(keys(Jb)) == Set([(:A, :r), (:A, :w), (:C, :r), (:C, :w)])
    @test all(isfinite, Jb[(:A, :r)]) && all(isfinite, Jb[(:C, :w)])

    firm = _cd_firm_block(ss.K, ss.L, m.Z, m.alpha, m.delta)
    mkt = SimpleBlock(x -> [x[1] - x[2]];
                      inputs=[:A, :K], outputs=[:asset_mkt],
                      ss_inputs=Dict(:A => hh.ss_outputs[:A], :K => ss.K),
                      name=:asset_market)
    dag = combine_blocks(firm, hh, mkt; name=:lc_cd, ss_tol=1e-5)
    @test [b.name for b in dag.blocks] == [:firm, :households, :asset_market]
    @test dag.exogenous == [:K, :Z]
    @test :A in dag.endogenous && :asset_mkt in dag.endogenous

    gej = ssj_jacobian(dag; unknowns=[:K], targets=[:asset_mkt], shocks=[:Z],
                       T_horizon=Th, target_tol=Inf)
    @test all(isfinite, gej.H_U) && all(isfinite, gej.H_Z)
    @test size(gej.H_U) == (Th, Th)

    dZ = Dict(:Z => [0.01 * 0.8^(t - 1) for t in 1:Th])
    ir = ssj_irf(gej, dZ; residual=false)
    @test all(isfinite, ir.paths[:K])
    @test all(isfinite, ir.paths[:A])
    @test all(isfinite, ir.paths[:r])
    @test all(isfinite, ir.paths[:Y])
    @test maximum(abs, gej.H_U * ir.paths[:K] .+ gej.H_Z * dZ[:Z]) < 1e-8

    @test_throws ArgumentError HetBlock(spec, ss; outputs=[:N])
    @test_throws ArgumentError HetBlock(spec, ss; outputs=[:nonsense])
end

@testset "G-16: MitBlock from ContinuousHouseholdSystem (#650)" begin
    m = CTAiyagari(; I=25, a_max=20.0)
    spec = to_spec(m)
    @test MacroEconometricModels.has_kind(spec, ContinuousHouseholdSystem)
    ss = ct_steady_state(m; tol=1e-4, max_iter=40)
    @test ss.converged
    @test ss.K > 0 && ss.L > 0

    hh = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A, :C], name=:household)
    @test hh isa MitBlock
    @test hh.ss_inputs[:r] == ss.r

    Th = 6
    firm = _cd_firm_block(ss.K, ss.L, m.Z, m.alpha, m.delta)
    mkt = SimpleBlock(x -> [x[1] - x[2]];
                      inputs=[:A, :K], outputs=[:asset_mkt],
                      ss_inputs=Dict(:A => hh.ss_outputs[:A], :K => ss.K),
                      name=:asset_market)
    dag = combine_blocks(firm, hh, mkt; name=:ct_cd, ss_tol=1e-5)
    gej = ssj_jacobian(dag; unknowns=[:K], targets=[:asset_mkt], shocks=[:Z],
                       T_horizon=Th, target_tol=Inf)
    @test all(isfinite, gej.H_U) && all(isfinite, gej.H_Z)
    dZ = Dict(:Z => [0.01 * 0.8^(t - 1) for t in 1:Th])
    ir = ssj_irf(gej, dZ; residual=false)
    @test all(isfinite, ir.paths[:K]) && all(isfinite, ir.paths[:A])
    @test all(isfinite, ir.paths[:r])
end

@testset "G-16: DCEGM HetBlock throws G-11 (#650)" begin
    prob = dcegm_retirement_model(; n_a=20, n_periods=4)
    spec = to_spec(prob)
    @test MacroEconometricModels.has_kind(spec, DCEGMSystem)
    sol = dcegm_solve(prob)
    err = try
        HetBlock(spec, sol)
        ErrorException("expected HetBlock(DCEGM) to throw")
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("G-11", sprint(showerror, err))
    err2 = try
        MitBlock(spec, sol)
        ErrorException("expected MitBlock(DCEGM) to throw")
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("G-11", sprint(showerror, err2))
    @test_throws ArgumentError HetBlock(spec, prob)
end

# Aqua LTS empirical: MitBlock(evaluate, ::Type{T}) overlapped
# MitBlock(spec::ModelSpec, ss) on (ModelSpec, Type{<:AbstractFloat}).
@testset "MitBlock(spec, Type) is not ambiguous" begin
    spec = @dsge begin
        parameters: ρ = 0.9
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    err = try
        MitBlock(spec, Float64)
        ErrorException("expected MitBlock(spec, Float64) to throw")
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("MitBlock(spec, ss)", sprint(showerror, err))
end

@testset "HA @dsge default CD injection requires canonical names (MSR-17)" begin
    spec = @dsge begin
        parameters: alpha = 0.36, beta_hh = 0.99, delta = 0.025, rho_z = 0.95, sigma_z = 0.007
        endogenous: Y, K, r, w, Z
        exogenous: eps_Z
        heterogeneous: a in [0.0, 50.0], n_grid = 20, utility = log, discount = beta_hh, borrowing = 0.0
        idiosyncratic: e ~ Rouwenhorst(0.9, 0.3, 3)
        aggregation: K = sum(a)
    end
    @test spec isa ModelSpec
    @test spec.endog == [:Y, :K, :r, :w, :Z]
    @test length(spec.equations) == 5

    err = try
        @eval @dsge begin
            parameters: alpha = 0.36, beta_hh = 0.99
            endogenous: K, Y, r, w, Z
            exogenous: eps_Z
            heterogeneous: a in [0.0, 50.0], n_grid = 20, utility = log, discount = beta_hh, borrowing = 0.0
            idiosyncratic: e ~ Rouwenhorst(0.9, 0.3, 3)
            aggregation: K = sum(a)
        end
        error("should have thrown")
    catch e
        e
    end
    inner = err isa LoadError ? err.error : err
    @test inner isa ArgumentError
    @test occursin("Y, K, r, w, Z", sprint(showerror, inner))
end
