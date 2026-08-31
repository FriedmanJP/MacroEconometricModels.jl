# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

if !@isdefined(simulate_proxy_svar)
    include("id_dgps.jl")
end

const MEM = MacroEconometricModels

@testset "SID-11 proxy SVAR" begin
    B_true = [1.0 0.3 0.2; 0.5 1.0 0.1; 0.4 0.2 1.0]
    A = [0.5 * Matrix{Float64}(I, 3, 3)]

    @testset "vector method still returns NamedTuple" begin
        rng = MersenneTwister(7401)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=400, ρ=0.8, rng=rng)
        m = estimate_var(Y, 1)
        nt = identify_proxy(m, z)
        @test nt isa NamedTuple
        @test keys(nt) == (:Q, :b1, :first_stage_F, :z_eff)
        @test size(nt.Q) == (3, 3)
        @test length(nt.b1) == 3
        @test isfinite(nt.first_stage_F)
    end

    @testset "matrix method returns ProxySVARResult" begin
        rng = MersenneTwister(7402)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=400, ρ=0.8, rng=rng)
        m = estimate_var(Y, 1)
        r = identify_proxy(m, reshape(z, :, 1))
        @test r isa ProxySVARResult
        @test r.k == 1
        @test r.is_partial
        @test size(r.Q) == (3, 3)
        @test size(r.B0) == (3, 3)
        @test r.first_stage_F > 10
        @test 0 < r.reliability < 1
        @test length(r.varnames) == 3
        @test length(r.shock_names) == 3
        @test length(r.instruments_names) == 1
    end

    @testset "vector and matrix k=1 unit-effect agree" begin
        rng = MersenneTwister(7403)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=400, ρ=0.8, rng=rng)
        m = estimate_var(Y, 1)
        nt = identify_proxy(m, z; normalize=1, normalize_value=1)
        r = identify_proxy(m, reshape(z, :, 1); normalize=:unit_effect, normalize_var=1)
        @test r.Q[:, 1] ≈ nt.Q[:, 1] atol = 1e-8
        @test r.B0[1, 1] ≈ 1 atol = 1e-8
    end

    @testset "first-stage F matches first_stage_regression" begin
        rng = MersenneTwister(7404)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=500, ρ=0.7, rng=rng)
        m = estimate_var(Y, 1)
        r = identify_proxy(m, reshape(z, :, 1); normalize=:unit_effect, normalize_var=1)
        U = m.U
        z_eff = MEM._align_instrument(z, size(m.Y, 1), m.p, size(U, 1))
        mask = [isfinite(z_eff[t]) && all(isfinite, @view U[t, :]) for t in 1:size(U, 1)]
        y = U[mask, 1]
        Zc = reshape(z_eff[mask], :, 1)
        controls = zeros(eltype(y), length(y), 0)
        fs = MEM.first_stage_regression(y, Zc, controls)
        @test r.first_stage_F ≈ fs.F_stat rtol = 1e-6
    end

    @testset "reliability rises with ρ" begin
        rels = Float64[]
        for (i, ρ) in enumerate((0.3, 0.9))
            rng = MersenneTwister(7405 + i)
            Y, _, z = simulate_proxy_svar(B_true, A; Tobs=800, ρ=ρ, rng=rng)
            m = estimate_var(Y, 1)
            r = identify_proxy(m, reshape(z, :, 1); normalize=:unit_variance)
            push!(rels, r.reliability)
        end
        @test rels[2] > rels[1]
    end

    @testset "registry flags and compute_Q" begin
        @test haskey(MEM.IDENTIFICATION_REGISTRY, :proxy)
        @test MEM._needs_residuals(:proxy)
        @test !MEM._is_set_identified(:proxy)
        @test MEM._is_partial(:proxy)
        rng = MersenneTwister(7406)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=300, ρ=0.8, rng=rng)
        m = estimate_var(Y, 1)
        Q = MEM.compute_Q(m, :proxy; instruments=reshape(z, :, 1), normalize=:unit_variance)
        @test size(Q) == (3, 3)
        @test norm(Q' * Q - I(3)) < 1e-8
        @test_throws ArgumentError MEM.compute_Q(m, :proxy)
    end

    @testset "irf/fevd/hd method=:proxy" begin
        rng = MersenneTwister(7407)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=300, ρ=0.8, rng=rng)
        m = estimate_var(Y, 1)
        Z = reshape(z, :, 1)
        ir = irf(m, 8; method=:proxy, instruments=Z, normalize=:unit_variance)
        @test ir isa ImpulseResponse
        @test size(ir.values) == (8, 3, 3)
        @test isfinite(ir.values[1, 1, 1])
        fv = fevd(m, 8; method=:proxy, instruments=Z, normalize=:unit_variance)
        @test fv isa FEVD
        hd = historical_decomposition(m, 20; method=:proxy, instruments=Z, normalize=:unit_variance)
        @test hd isa HistoricalDecomposition
    end

    @testset "align=true drops NaN instrument rows" begin
        rng = MersenneTwister(7408)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=400, ρ=0.8, rng=rng)
        z[1:20] .= NaN
        m = estimate_var(Y, 1)
        r = identify_proxy(m, reshape(z, :, 1); align=true)
        @test r isa ProxySVARResult
        @test isfinite(r.first_stage_F)
    end

    @testset "weak instrument warns" begin
        rng = MersenneTwister(7409)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=300, ρ=0.05, rng=rng)
        m = estimate_var(Y, 1)
        @test_logs (:warn, r"(?i)weak") identify_proxy(m, reshape(z, :, 1))
    end

    @testset "report / refs / plot_result" begin
        rng = MersenneTwister(7410)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=250, ρ=0.8, rng=rng)
        m = estimate_var(Y, 1)
        r = identify_proxy(m, reshape(z, :, 1))
        buf = IOBuffer()
        report(buf, r)
        s = String(take!(buf))
        @test occursin("Proxy", s) || occursin("proxy", lowercase(s))
        @test occursin("B", s)
        rs = sprint(refs, r)
        @test occursin("Mertens", rs)
        @test occursin("Stock", rs)
        p = plot_result(r)
        @test p isa MEM.PlotOutput
        p2 = plot_result(r; view=:B0)
        @test p2 isa MEM.PlotOutput
        @test_throws ArgumentError plot_result(r; view=:bogus)
    end

    @testset "k=1 Anderson-Rubin bands reuse lp_iv_ar_band" begin
        rng = MersenneTwister(7411)
        Y, _, z = simulate_proxy_svar(B_true, A; Tobs=400, ρ=0.8, rng=rng)
        m = estimate_var(Y, 1)
        band = proxy_ar_band(m, z; horizon=2, normalize_var=1, n_grid=81, span=8)
        @test band isa LPIVARBand
        @test size(band.point, 1) == 3
        @test all(isfinite, band.point)
    end

    @testset "LP-IV and proxy impact agree on :mp_shocks" begin
        td = load_example(:mp_shocks)
        # mp1 sample 1988Q4–2012Q2; keep ygap, infl, ffr on that window
        row(yr, q) = (yr - 1960) * 4 + q
        rows = row(1988, 4):row(2012, 2)
        Y = td.data[rows, 1:3]
        z = td.data[rows, 6]
        @test !any(isnan, Y)
        @test !any(isnan, z)
        p = 4
        m = estimate_var(Y, p; varnames=["ygap", "infl", "ffr"])
        r = identify_proxy(m, reshape(z, :, 1); normalize=:unit_effect, normalize_var=3)
        lp = estimate_lp_iv(Y, 3, reshape(z, :, 1), 0; lags=p, varnames=["ygap", "infl", "ffr"])
        lir = lp_iv_irf(lp)
        b = r.B0[:, 1]
        # unit-effect on ffr: relative impacts should match LP-IV h=0
        @test abs(b[3] - 1) < 1e-8
        @test b[1] ≈ lir.values[1, 1] atol = 0.15
        @test b[2] ≈ lir.values[1, 2] atol = 0.15
        @test b[3] ≈ lir.values[1, 3] atol = 0.15
    end

    if !FAST
        @testset "MBB 90% coverage at T=400 (gated)" begin
            true_imp = B_true[2, 1]   # non-normalised variable
            nrep = 200
            ncover = 0
            for r in 1:nrep
                rng = MersenneTwister(7500 + r)
                Y, _, z = simulate_proxy_svar(B_true, A; Tobs=400, ρ=0.7, rng=rng)
                m = estimate_var(Y, 1)
                ir = irf(m, 1; method=:proxy, instruments=reshape(z, :, 1),
                         normalize=:unit_variance, ci_type=:bootstrap, bootstrap=:block,
                         reps=99, conf_level=0.9, seed=7500 + r)
                lo, hi = ir.ci_lower[1, 2, 1], ir.ci_upper[1, 2, 1]
                s = sign(ir.values[1, 1, 1])
                target = s * true_imp
                ncover += Int(lo <= target <= hi)
            end
            @test ncover / nrep >= 0.80
        end
    end
end
