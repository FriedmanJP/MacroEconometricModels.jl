# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-23 (#431): Panel ARDL — Pooled Mean Group (PMG), Mean Group (MG), Dynamic
# Fixed Effects (DFE), and the generalized Hausman selection test.
#
# Oracle strategy (no invented numerics):
#   (1) Cross-implementation against EV-08: the MG long-run estimate must equal the
#       hand-computed mean of the per-unit `estimate_ardl` long-run coefficients to
#       MACHINE tolerance (the MG estimator is defined as that mean).
#   (2) Analytic identity: the MG standard error must equal the cross-unit
#       std(θ_i)/√N (Pesaran–Smith 1995 Swamy between-unit variance) EXACTLY.
#   (3) Estimator property on a fixed-seed HOMOGENEOUS-long-run EC DGP: PMG recovers
#       the true common θ within a loose stated tolerance, all φ_i are negative
#       (stable error-correction), and the PMG-vs-MG Hausman test FAILS to reject
#       H0 (long-run homogeneity) at 5%.
#   (4) Efficiency ordering: the pooled PMG long-run variance is no larger than the
#       MG between-unit variance on the homogeneous DGP (so V_MG − V_PMG is PSD and
#       the Hausman quadratic form is well-defined / non-negative).
#
# NOTE on the Stata `xtpmg` oracle: the acceptance criterion suggests pinning
#   . xtpmg d.y ..., lr(l.y l.x) ec(ec) replace pmg   (Blackburne–Frank 2007)
#   . xtpmg ..., mg      // Mean Group
#   . xtpmg ..., dfe     // Dynamic Fixed Effects
#   . hausman pmg mg     // long-run homogeneity
# Stata is NOT available in this environment, so — following the EV-08 precedent and
# the project's oracle discipline (never fabricate reference numerics) — no `xtpmg`
# output is hard-coded. The exact commands above are recorded for provenance; the
# tests rest on the analytic identities (1)–(4), which pin the estimators uniquely.

using Test, MacroEconometricModels, Random, Statistics, LinearAlgebra, DataFrames

# ---------------------------------------------------------------------------
# Fixed-seed EC-form panel DGP.
#   Δy_it = φ_i (y_{i,t-1} − θ' x_{i,t-1}) + g_i Δx_{i,t} + ε_it,  x ~ I(1).
# `homog=true` ⇒ common long-run θ; `homog=false` ⇒ θ_i spread around θ.
# ---------------------------------------------------------------------------
function _pmg_dgp(; N=20, T=60, theta=1.5, seed=123, homog=true, spread=0.8)
    rng = MersenneTwister(seed)
    ys = Vector{Vector{Float64}}(); xs = Vector{Matrix{Float64}}()
    for i in 1:N
        phi = -(0.2 + 0.4 * rand(rng))                    # φ_i ∈ (-0.6, -0.2)
        th  = homog ? theta : theta + spread * (rand(rng) - 0.5)
        g   = 0.3 * randn(rng)
        x = zeros(T); y = zeros(T)
        x[1] = randn(rng); y[1] = th * x[1] + randn(rng)
        for t in 2:T
            x[t] = x[t-1] + randn(rng)
            y[t] = y[t-1] + phi * (y[t-1] - th * x[t-1]) + g * (x[t] - x[t-1]) + 0.3 * randn(rng)
        end
        push!(ys, y); push!(xs, reshape(x, :, 1))
    end
    (ys, xs)
end

# Flatten per-unit series into (yv, Xv, id, time) long form.
function _pmg_long(ys, xs)
    N = length(ys); T = length(ys[1]); k = size(xs[1], 2)
    yv = Float64[]; id = Int[]; tm = Int[]
    Xv = Matrix{Float64}(undef, N * T, k)
    r = 0
    for i in 1:N, t in 1:T
        r += 1
        yv = push!(yv, ys[i][t]); Xv[r, :] .= xs[i][t, :]
        push!(id, i); push!(tm, t)
    end
    (yv, Xv, id, tm)
end

@testset "Panel ARDL (PMG / MG / DFE)" begin
    ys, xs = _pmg_dgp(; N=20, T=60, theta=1.5, seed=123, homog=true)
    yv, Xv, id, tm = _pmg_long(ys, xs)

    mg  = estimate_pmg(yv, Xv, id, tm; p=1, q=1, method=:mg,  xnames=["x"])
    pmg = estimate_pmg(yv, Xv, id, tm; p=1, q=1, method=:pmg, xnames=["x"])
    dfe = estimate_pmg(yv, Xv, id, tm; p=1, q=1, method=:dfe, xnames=["x"])

    @testset "types / plumbing" begin
        @test mg isa PMGModel{Float64}
        @test pmg.method === :pmg && mg.method === :mg && dfe.method === :dfe
        @test length(mg.theta) == 1 && length(pmg.theta) == 1
        @test mg.N == 20 && pmg.N == 20
        @test all(mg.T_i .== 59)                      # T − max(p,q) = 60 − 1
        @test pmg.converged
        @test coef(pmg) == pmg.theta
        @test nobs(mg) == sum(mg.T_i)
    end

    @testset "(1) MG ≡ mean of per-unit estimate_ardl long-run [machine tol]" begin
        lr_i = [long_run(estimate_ardl(ys[i], xs[i]; p=1, q=1, case=3)).theta[1]
                for i in 1:20]
        @test mg.theta[1] ≈ mean(lr_i) atol=1e-12
        @test mg.theta_i[:, 1] ≈ lr_i atol=1e-12
    end

    @testset "(2) MG SE = std(θ_i)/√N [exact identity]" begin
        θi = mg.theta_i[:, 1]
        @test mg.theta_se[1] ≈ std(θi; corrected=true) / sqrt(20.0) atol=1e-14
        # Swamy between-unit covariance diagonal reconstructs the SE².
        @test mg.theta_vcov[1, 1] ≈ mg.theta_se[1]^2 atol=1e-14
    end

    @testset "(3) PMG recovers true θ; φ_i < 0; Hausman fails to reject" begin
        @test pmg.theta[1] ≈ 1.5 atol=0.06          # loose recovery on homogeneous DGP
        @test all(pmg.phi_i .< 0)                    # stable error-correction
        @test all(mg.phi_i .< 0)
        @test pmg.n_nonconv == 0
        h = hausman_test(pmg, mg)
        @test h isa MacroEconometricModels.PanelTestResult
        @test h.pvalue > 0.05                        # long-run homogeneity not rejected
        @test h.statistic ≥ 0                        # V_MG − V_PMG PSD here
    end

    @testset "(4) PMG is more efficient than MG on homogeneous DGP" begin
        @test pmg.theta_vcov[1, 1] ≤ mg.theta_vcov[1, 1] + 1e-10
    end

    @testset "(5) Hausman REJECTS on heterogeneous long-run (power arm)" begin
        # The homog=false DGP existed but was never called (DGP-04 #793): on
        # the shared hetero panel (N=30, T=100) the homogeneity null is false
        # and the test must reject (probed p = 0.003).
        dh = dgp_pmg(MersenneTwister(123); N=30, T=100, homogeneous=false)
        tmh = repeat(1:100, outer=30)
        Xh = reshape(dh.X, :, 1)
        pmgh = estimate_pmg(dh.Y, Xh, dh.id, tmh; p=1, q=1, method=:pmg,
                            xnames=["x"])
        mgh = estimate_pmg(dh.Y, Xh, dh.id, tmh; p=1, q=1, method=:mg,
                           xnames=["x"])
        h = hausman_test(pmgh, mgh)
        @test h.pvalue < 0.05
        # MG stays consistent for the mean long-run under heterogeneity.
        @test mgh.theta[1] ≈ mean(dh.theta_i) atol=0.2
    end

    @testset "DFE: pooled EC with clustered SEs" begin
        @test dfe.theta[1] ≈ 1.5 atol=0.08
        @test dfe.phi < 0
        @test all(dfe.phi_i .== dfe.phi)             # common speed
        @test dfe.theta_se[1] > 0
        # Hausman DFE-vs-MG runs (sign not asserted — finite-sample dV may be non-PSD).
        h2 = hausman_test(dfe, mg)
        @test isfinite(h2.statistic)
        @test 0.0 ≤ h2.pvalue ≤ 1.0
    end

    @testset "PanelData (xtset) path == long-matrix path" begin
        df = DataFrame(id=id, time=tm, y=yv, x=Xv[:, 1])
        pd = xtset(df, :id, :time)
        pmg_pd = estimate_pmg(pd, :y, :x; p=1, q=1, method=:pmg)
        @test pmg_pd.theta[1] ≈ pmg.theta[1] atol=1e-10
        @test pmg_pd.phi ≈ pmg.phi atol=1e-10
    end

    @testset "multi-regressor, higher lags (k=2, p=2, q=2)" begin
        rng = MersenneTwister(99)
        N = 15; T = 70
        rec = NamedTuple[]
        for i in 1:N
            phi = -(0.25 + 0.3 * rand(rng)); th = [1.0, -0.5]
            x1 = zeros(T); x2 = zeros(T); y = zeros(T)
            x1[1] = randn(rng); x2[1] = randn(rng)
            y[1] = th[1]*x1[1] + th[2]*x2[1] + randn(rng)
            x1[2] = x1[1]+randn(rng); x2[2] = x2[1]+randn(rng); y[2] = y[1]+0.1*randn(rng)
            for t in 3:T
                x1[t] = x1[t-1]+randn(rng); x2[t] = x2[t-1]+randn(rng)
                y[t] = y[t-1] + phi*(y[t-1]-th[1]*x1[t-1]-th[2]*x2[t-1]) +
                       0.2*(y[t-1]-y[t-2]) + 0.3*(x1[t]-x1[t-1]) + 0.3*randn(rng)
            end
            for t in 1:T; push!(rec, (id=i, time=t, y=y[t], x1=x1[t], x2=x2[t])); end
        end
        pd = xtset(DataFrame(rec), :id, :time)
        pmg2 = estimate_pmg(pd, :y, :x1, :x2; p=2, q=2, method=:pmg)
        mg2  = estimate_pmg(pd, :y, :x1, :x2; p=2, q=2, method=:mg)
        @test pmg2.theta ≈ [1.0, -0.5] atol=0.12
        @test length(pmg2.srnames) == 6              # int + L1.D.y + (D.x + L1.D.x)*2
        @test "(Intercept)" in pmg2.srnames && "L1.D.y" in pmg2.srnames
        @test all(pmg2.phi_i .< 0)
        # MG oracle still holds for the first regressor with k=2, p=2, q=2.
        # xtset drops id/time, so pd.varnames == ["y","x1","x2"] (cols 1,2,3).
        lr1 = [long_run(estimate_ardl(pd.data[pd.group_id .== i, 1],
                                      pd.data[pd.group_id .== i, 2:3];
                                      p=2, q=[2,2], case=3)).theta[1] for i in 1:N]
        @test mg2.theta[1] ≈ mean(lr1) atol=1e-10
    end

    @testset "trend variants" begin
        pmg_n = estimate_pmg(yv, Xv, id, tm; p=1, q=1, method=:pmg, trend=:none, xnames=["x"])
        pmg_t = estimate_pmg(yv, Xv, id, tm; p=1, q=1, method=:pmg, trend=:trend, xnames=["x"])
        @test !("(Intercept)" in pmg_n.srnames)
        @test "trend" in pmg_t.srnames && "(Intercept)" in pmg_t.srnames
        @test isfinite(pmg_n.theta[1]) && isfinite(pmg_t.theta[1])
        # Trend short-run coefficient ≈ 0 on the trendless DGP (DGP-04 #793;
        # probed −0.00068), and scales with a planted trend: in EC form
        # τ = −φδ, so τ̂/(−φ̂) ≈ δ (probed 0.048 vs planted 0.05).
        it = findfirst(==("trend"), pmg_t.srnames)
        @test abs(pmg_t.sr[it]) < 0.01
        T = length(ys[1])
        yv2 = vcat([ys[i] .+ 0.05 .* collect(1.0:T) for i in 1:length(ys)]...)
        pmg_t2 = estimate_pmg(yv2, Xv, id, tm; p=1, q=1, method=:pmg,
                              trend=:trend, xnames=["x"])
        @test pmg_t2.sr[it] > 0.01
        @test pmg_t2.sr[it] / (-pmg_t2.phi) ≈ 0.05 atol=0.02
    end

    @testset "display / refs render" begin
        io = IOBuffer()
        for m in (mg, pmg, dfe)
            report(io, m)
        end
        s = String(take!(io))
        @test occursin("Long-run coefficients", s)
        @test occursin("Error-correction speed", s)
        r = sprint(io -> refs(io, pmg))
        @test occursin("Pooled Mean Group", r)
        @test occursin("Blackburne", r)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError estimate_pmg(yv, Xv, id, tm; method=:bad, xnames=["x"])
        @test_throws ArgumentError estimate_pmg(yv, Xv, id, tm; p=0, xnames=["x"])
        @test_throws ArgumentError hausman_test(pmg, dfe)   # second must be :mg
        df = DataFrame(id=id, time=tm, y=yv, x=Xv[:, 1])
        pd = xtset(df, :id, :time)
        @test_throws ArgumentError estimate_pmg(pd, :nope, :x)
    end

    @testset "non-error-correcting flag (φ_i ≥ 0)" begin
        # Explosive units (AR 1.08) cannot error-correct, so n_nonconv > 0
        # UNCONDITIONALLY (DGP-04 #793) — the old `> 0 && @test` guard
        # evaporated on pure noise (n_nonconv == 0 for seeds 5, 6, 7).
        rng = MersenneTwister(5)
        N = 6; T = 25
        rec = NamedTuple[]
        for i in 1:N, t in 1:T
            push!(rec, (id=i, time=t, y=randn(rng), x=randn(rng)))
        end
        ys_x = Dict{Int,Vector{Float64}}()
        for i in 1:N
            ei = [r.y for r in rec if r.id == i]
            ye = zeros(T); ye[1] = ei[1]
            for t in 2:T
                ye[t] = 1.08 * ye[t-1] + ei[t]
            end
            ys_x[i] = ye
        end
        rec2 = [(id=r.id, time=r.time, y=ys_x[r.id][r.time], x=r.x) for r in rec]
        pd = xtset(DataFrame(rec2), :id, :time)
        m = estimate_pmg(pd, :y, :x; p=1, q=1, method=:mg)
        @test m.n_nonconv > 0                       # probed 4 on this draw
        @test m.n_nonconv == count(>=(0.0), m.phi_i)
        io = IOBuffer(); report(io, m)
        @test occursin("non-error-correcting", String(take!(io)))
    end
end
