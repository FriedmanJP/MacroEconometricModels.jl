# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, LinearAlgebra, Statistics, Random

@testset "Maćkowiak–Wiederholt Multi-Block Attention Allocation (#921)" begin

    # =========================================================================
    # 1. Input Validation & Block Formats
    # =========================================================================
    @testset "Input Validation & Block Specifications" begin
        agg0 = (; rho=0.85, sigma=0.01)
        idio0 = (; rho=0.80, sigma=0.10)

        @test_throws ArgumentError solve_mw_attention(agg0, idio0)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; capacity=2.0, omega=0.1)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; capacity=0.0)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; capacity=-1.0)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; omega=0.0)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; omega=-0.5)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; capacity=2.0, strategic_complementarity=-0.1)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; capacity=2.0, strategic_complementarity=1.0)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; capacity=2.0, horizon=0)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; capacity=2.0, tol=0.0)
        @test_throws ArgumentError solve_mw_attention(agg0, idio0; capacity=2.0, maxiter=0)
        @test_throws ArgumentError solve_mw_attention((; rho=0.8, sigma=0.01, units=:bits),
                                                      (; rho=0.8, sigma=0.10, units=:nats); capacity=2.0)
        @test_throws ArgumentError solve_mw_attention((; rho=0.8, sigma=0.01, process=:invalid),
                                                      idio0; capacity=2.0)
        @test_throws ArgumentError solve_mw_attention("bad", idio0; capacity=2.0)
        @test_throws ArgumentError solve_mw_attention((; rho=0.8, sigma=0.0), (; rho=0.8, sigma=0.0); capacity=1.0)

        # Equivalent specifications: NamedTuple, Tuple, RationalInattentionProblem, (A, G, Q, H, R)
        r_nt = solve_mw_attention((; rho=0.85, sigma=0.01, weight=1.0, beta=0.99),
                                  (; rho=0.80, sigma=0.10, weight=1.0, beta=0.99);
                                  capacity=2.0, strategic_complementarity=0.0, horizon=20)
        r_tup = solve_mw_attention((0.85, 0.01, 1.0), (0.80, 0.10, 1.0);
                                   capacity=2.0, strategic_complementarity=0.0, horizon=20)
        p_agg = RationalInattentionProblem(0.85, 0.01, 1.0; M=1.0, beta=0.99, capacity=1.0)
        p_id = RationalInattentionProblem(0.80, 0.10, 1.0; M=1.0, beta=0.99, capacity=1.0)
        r_prob = solve_mw_attention(p_agg, p_id; capacity=2.0, strategic_complementarity=0.0, horizon=20)
        r_mat = solve_mw_attention(
            (; A=[0.85;;], G=[1.0;;], Q=[0.01^2;;], H=[1.0;;], R=[1.0;;], beta=0.99),
            (; A=[0.80;;], B=[0.10;;], H=[1.0;;], M=[1.0;;], beta=0.99);
            capacity=2.0, strategic_complementarity=0.0, horizon=20,
        )

        for r_other in (r_tup, r_prob, r_mat)
            @test isapprox(r_other.kappa_agg, r_nt.kappa_agg; atol=1e-7)
            @test isapprox(r_other.kappa_idio, r_nt.kappa_idio; atol=1e-7)
            @test isapprox(r_other.omega, r_nt.omega; rtol=1e-6)
            @test isapprox(r_other.agg_price_irf, r_nt.agg_price_irf; atol=1e-7)
            @test isapprox(r_other.idio_price_irf, r_nt.idio_price_irf; atol=1e-7)
        end
    end

    # =========================================================================
    # 2. Water-Filling Optimality & Fixed-Omega / Nats Consistency
    # =========================================================================
    @testset "Water-Filling Optimality & Fixed-Omega Round-Trip" begin
        agg = (; rho=0.88, sigma=0.02, weight=1.0, beta=0.99)
        idio = (; rho=0.75, sigma=0.10, weight=1.0, beta=0.99)

        # Partial equilibrium (r = 0): exact agreement with independent solve_ri_kalman at common ω
        res_pe = solve_mw_attention(agg, idio; capacity=2.5, strategic_complementarity=0.0, horizon=25)
        @test res_pe.converged
        @test res_pe.units === :bits
        @test isapprox(res_pe.total_kappa, 2.5; atol=1e-6)
        @test isapprox(res_pe.kappa_agg + res_pe.kappa_idio, res_pe.total_kappa; atol=1e-12)
        @test isapprox(res_pe.agg_solution.omega, res_pe.omega; rtol=1e-9)
        @test isapprox(res_pe.idio_solution.omega, res_pe.omega; rtol=1e-9)

        sol_a_chk = solve_ri_kalman(RationalInattentionProblem(0.88, 0.02, 1.0; beta=0.99, omega=res_pe.omega))
        sol_i_chk = solve_ri_kalman(RationalInattentionProblem(0.75, 0.10, 1.0; beta=0.99, omega=res_pe.omega))
        @test isapprox(sol_a_chk.capacity_bits, res_pe.kappa_agg; atol=1e-9)
        @test isapprox(sol_i_chk.capacity_bits, res_pe.kappa_idio; atol=1e-9)

        # GE capacity <-> omega round-trip
        res_cap = solve_mw_attention(agg, idio; capacity=2.5, strategic_complementarity=0.75, horizon=30)
        @test res_cap.converged
        @test isapprox(res_cap.total_kappa, 2.5; atol=1e-6)
        res_om = solve_mw_attention(agg, idio; omega=res_cap.omega, strategic_complementarity=0.75, horizon=30)
        @test res_om.converged
        @test isapprox(res_om.kappa_agg, res_cap.kappa_agg; atol=1e-6)
        @test isapprox(res_om.kappa_idio, res_cap.kappa_idio; atol=1e-6)
        @test isapprox(res_om.agg_price_irf, res_cap.agg_price_irf; atol=1e-7)
        @test isapprox(res_om.real_output_irf, res_cap.real_output_irf; atol=1e-7)

        # Units = :nats conversion (κ_nats = κ_bits * ln 2)
        res_nats = solve_mw_attention(
            (; rho=0.88, sigma=0.02, beta=0.99, units=:nats),
            (; rho=0.75, sigma=0.10, beta=0.99, units=:nats);
            capacity=2.5 * log(2.0), strategic_complementarity=0.75, horizon=30,
        )
        @test res_nats.units === :nats
        @test isapprox(res_nats.kappa_agg / log(2.0), res_cap.kappa_agg; atol=1e-5)
        @test isapprox(res_nats.kappa_idio / log(2.0), res_cap.kappa_idio; atol=1e-5)
    end

    # =========================================================================
    # 3. Maćkowiak & Wiederholt (2009) Quantitative Benchmark
    # =========================================================================
    @testset "MW (2009) Benchmark (σ_idio = 10 σ_agg, κ = 3 bits, r = 0.85)" begin
        sigma_agg = 0.01
        sigma_idio = 10.0 * sigma_agg # 0.10

        # Persistent damped-growth AR(2) nominal demand (1 - 0.94 L)(1 - 0.60 L) q_t = σ_agg w_t
        agg_mw = (; rho=(0.94, 0.60), sigma=sigma_agg, beta=0.99)
        idio_mw = (; rho=0.85, sigma=sigma_idio, beta=0.99)

        res = solve_mw_attention(
            agg_mw, idio_mw;
            capacity=3.0, strategic_complementarity=0.85, horizon=40, tol=1e-8, maxiter=200,
        )
        @test res isa MWAttentionResult{Float64}
        @test res.converged
        @test res.iterations >= 1
        @test isapprox(res.total_kappa, 3.0; atol=1e-5)

        # AC: > 85% of bit capacity allocated to κ_idio
        @test res.kappa_idio / res.total_kappa > 0.85
        @test res.kappa_agg > 0.0
        @test res.kappa_agg + res.kappa_idio ≈ res.total_kappa

        # AC: fast (~1 period) idiosyncratic pass-through (> 90% on impact h = 1)
        @test length(res.idio_price_irf) == 40
        @test res.idio_price_irf[1] / sigma_idio > 0.90

        # AC: sluggish aggregate price + persistent hump-shaped real output response
        @test length(res.agg_price_irf) == 40
        @test length(res.real_output_irf) == 40
        @test res.agg_price_irf[1] / sigma_agg < 0.35
        @test all(x -> x > 0.0, res.real_output_irf[1:10])
        _, peak_h = findmax(res.real_output_irf)
        @test peak_h > 1
        @test res.real_output_irf[peak_h] > res.real_output_irf[1]
        @test minimum(res.real_output_irf) > -0.05 * res.real_output_irf[peak_h]
        @test res.real_output_irf[end] < 0.5 * res.real_output_irf[peak_h]

        # Strategic complementarity comparative static: higher r reduces κ_agg and slows agg price pass-through
        res_r0 = solve_mw_attention(
            agg_mw, idio_mw;
            capacity=3.0, strategic_complementarity=0.0, horizon=40, tol=1e-8,
        )
        @test res.kappa_agg < res_r0.kappa_agg
        @test res.agg_price_irf[1] < res_r0.agg_price_irf[1]
        @test sum(res.real_output_irf) > sum(res_r0.real_output_irf)

        # Growth process Δq_t = ρ Δq_{t-1} + σ_agg w_t also converges cleanly with hump-shaped output
        res_gr = solve_mw_attention(
            (; rho=0.55, sigma=sigma_agg, process=:growth, beta=0.95),
            (; rho=0.80, sigma=sigma_idio, beta=0.95);
            capacity=3.0, strategic_complementarity=0.85, horizon=40,
        )
        @test res_gr.converged
        @test res_gr.kappa_idio > res_gr.kappa_agg
        _, peak_gr = findmax(res_gr.real_output_irf)
        @test peak_gr > 1
    end

    # =========================================================================
    # 4. RSER-01, Float32, Display & Ambiguities
    # =========================================================================
    @testset "RSER-01, Float32 & Display" begin
        res = solve_mw_attention(
            (; rho=(0.92, 0.55), sigma=0.01),
            (; rho=0.80, sigma=0.10);
            capacity=2.5, strategic_complementarity=0.5, horizon=20, tol=1e-6,
        )

        # Exact positional constructor (RSER-01)
        flds = [getfield(res, f) for f in fieldnames(MWAttentionResult)]
        rec = MWAttentionResult{Float64}(flds...)
        for f in fieldnames(MWAttentionResult)
            @test isequal(getfield(rec, f), getfield(res, f))
        end

        # Display & report across backends
        @test occursin("Maćkowiak–Wiederholt", sprint(show, res))
        rep_txt = sprint(report, res)
        @test occursin("Equilibrium Impulse Responses", rep_txt)
        rep_ltx = with_display_backend(:latex) do; sprint(report, res); end
        rep_htm = with_display_backend(:html) do; sprint(report, res); end
        @test occursin("\\begin{tabular}", rep_ltx)
        @test occursin("<table", rep_htm)

        # Float32 precision support
        p_agg32 = RationalInattentionProblem(0.85f0, 0.02f0, 1.0f0; beta=0.99f0, omega=0.05f0)
        p_id32 = RationalInattentionProblem(0.80f0, 0.10f0, 1.0f0; beta=0.99f0, omega=0.05f0)
        res32 = solve_mw_attention(p_agg32, p_id32; capacity=2.0f0, strategic_complementarity=0.5f0, horizon=15, tol=1e-5)
        @test res32 isa MWAttentionResult{Float32}
        @test eltype(res32.agg_price_irf) === Float32
        @test eltype(res32.idio_price_irf) === Float32
        @test eltype(res32.real_output_irf) === Float32
        @test res32.converged
    end
end
