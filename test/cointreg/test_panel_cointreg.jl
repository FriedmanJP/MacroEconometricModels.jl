# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-22 (#430): panel cointegrating regression — panel FMOLS / DOLS
# (group-mean between-dimension + pooled within-dimension).
#
# ORACLE DISCIPLINE. All numeric oracles here are in-env ANALYTIC IDENTITIES, no
# invented constants:
#   (1) N=1 degeneracy — pooled DOLS with a single unit reproduces EV-10's
#       single-equation DOLS slope EXACTLY (machine tolerance). Likewise pooled
#       FMOLS with N=1 reproduces the EV-10 FMOLS slope and its covariance.
#   (2) group-mean β̄ equals the hand-computed arithmetic mean of the per-unit
#       EV-10 `estimate_cointreg` coefficient vectors (machine tolerance).
#   (3) the reported group-mean t equals Σᵢ tᵢ / √N exactly, recomputed from the
#       per-unit t-ratios tᵢ = coef_i / se_i.
# The Pedroni (2000/2001) group-mean/pooled CONSTRUCTIONS are the reference; see
# the type docstring for citations. No external Stata/R constants are hard-coded
# because the analytic identities pin the estimator to machine precision.

using Test
using MacroEconometricModels
using MacroEconometricModels: estimate_cointreg, estimate_xtcointreg,
                              PanelCointRegModel, CointRegModel
using LinearAlgebra, Statistics, Random
using DataFrames
import StatsAPI as S

# -----------------------------------------------------------------------------
# Fixed-seed heterogeneous cointegrated panel:
#   y_{it} = a_i + b_i·x_{it} + u_{it},   x_{it} a random walk,
#   u_{it} = ρ_i u_{i,t-1} + e_{it} + φ_i v_{it}   (serial corr + endogeneity).
# Common cointegrating slope b_i ≡ β0 (so the panel long-run coef is well defined),
# heterogeneous intercepts / dynamics / endogeneity across units.
# -----------------------------------------------------------------------------
function coint_panel(; seed::Int=20260716, N::Int=6, T::Int=120, beta0::Float64=1.5)
    rng = MersenneTwister(seed)
    yv = Float64[]
    xv = Float64[]
    idv = Int[]
    tv = Int[]
    for i in 1:N
        v = randn(rng, T)
        e = randn(rng, T)
        x = cumsum(v)
        a_i = 1.0 + 0.5 * i
        rho = 0.2 + 0.05 * i
        phi = 0.3 + 0.05 * i
        u = zeros(T)
        for t in 1:T
            ulag = t == 1 ? 0.0 : u[t-1]
            u[t] = rho * ulag + e[t] + phi * v[t]
        end
        y = a_i .+ beta0 .* x .+ u
        append!(yv, y)
        append!(xv, x)
        append!(idv, fill(i, T))
        append!(tv, 1:T)
    end
    return yv, xv, idv, tv
end

# Build the identical per-unit (y_i, x_i) list the panel estimator sees.
function _unit_series(yv, xv, idv, tv)
    ids = sort(unique(idv))
    units = Tuple{Vector{Float64},Vector{Float64}}[]
    for g in ids
        rows = findall(==(g), idv)
        ord = sortperm(tv[rows])
        rr = rows[ord]
        push!(units, (yv[rr], xv[rr]))
    end
    return units
end

@testset "Panel Cointegrating Regression (EV-22)" begin
    yv, xv, idv, tv = coint_panel()
    N = length(unique(idv))
    units = _unit_series(yv, xv, idv, tv)
    bw = 3   # fixed bandwidth for exact reproducibility across paths

    # ---------------------------------------------------------------------
    # (2) group-mean β̄ = arithmetic mean of per-unit EV-10 coefficient vectors
    # (3) group-mean t = Σ tᵢ / √N   (tᵢ = coef_i / se_i)
    # ---------------------------------------------------------------------
    @testset "group-mean FMOLS ≡ mean of per-unit estimate_cointreg" begin
        per = [estimate_cointreg(yi, xi; method=:fmols, trend=:const, bandwidth=bw)
               for (yi, xi) in units]
        Cmat = hcat([S.coef(m) for m in per]...)          # (d+k)×N
        beta_bar = vec(mean(Cmat; dims=2))
        Tmat = hcat([S.coef(m) ./ S.stderror(m) for m in per]...)
        t_gm = vec(sum(Tmat; dims=2)) ./ sqrt(N)

        m = estimate_xtcointreg(yv, xv, idv, tv; method=:fmols, pooling=:group,
                                trend=:const, bandwidth=bw)
        @test m isa PanelCointRegModel
        @test m.N == N
        @test length(S.coef(m)) == 2                       # [const, slope]
        @test S.coef(m) ≈ beta_bar rtol=0 atol=1e-12       # (2) exact mean
        @test m.tstats ≈ t_gm rtol=0 atol=1e-10            # (3) Σtᵢ/√N exactly
        # display se is consistent with the reported t for well-behaved data
        @test S.coef(m) ./ m.se ≈ m.tstats rtol=1e-10
    end

    @testset "group-mean DOLS ≡ mean of per-unit DOLS + Σtᵢ/√N" begin
        per = [estimate_cointreg(yi, xi; method=:dols, trend=:const, bandwidth=bw)
               for (yi, xi) in units]
        Cmat = hcat([S.coef(m) for m in per]...)
        beta_bar = vec(mean(Cmat; dims=2))
        Tmat = hcat([S.coef(m) ./ S.stderror(m) for m in per]...)
        t_gm = vec(sum(Tmat; dims=2)) ./ sqrt(N)

        m = estimate_xtcointreg(yv, xv, idv, tv; method=:dols, pooling=:group,
                                trend=:const, bandwidth=bw)
        @test S.coef(m) ≈ beta_bar rtol=0 atol=1e-12
        @test m.tstats ≈ t_gm rtol=0 atol=1e-10
        # the slope point estimate is near the true β0 = 1.5
        @test isapprox(S.coef(m)[2], 1.5; atol=0.1)
    end

    # ---------------------------------------------------------------------
    # (1) N=1 degeneracy: pooled ≡ EV-10 single-equation estimator EXACTLY
    # ---------------------------------------------------------------------
    @testset "N=1 pooled DOLS ≡ EV-10 estimate_cointreg(:dols)" begin
        yi, xi = units[1]
        single = estimate_cointreg(yi, xi; method=:dols, trend=:const, bandwidth=bw)
        slope_single = S.coef(single)[2]                   # [const, slope]

        id1 = ones(Int, length(yi))
        t1 = collect(1:length(yi))
        m = estimate_xtcointreg(yi, xi, id1, t1; method=:dols, pooling=:pooled,
                                trend=:const, bandwidth=bw)
        @test m.N == 1
        @test length(S.coef(m)) == 1                       # pooled reports slope only
        @test S.coef(m)[1] ≈ slope_single rtol=0 atol=1e-10  # (1) machine tol
    end

    @testset "N=1 pooled FMOLS ≡ EV-10 estimate_cointreg(:fmols) slope + vcov" begin
        yi, xi = units[2]
        single = estimate_cointreg(yi, xi; method=:fmols, trend=:const, bandwidth=bw)
        slope_single = S.coef(single)[2]
        vcov_slope_single = S.vcov(single)[2, 2]           # FMOLS slope variance

        id1 = ones(Int, length(yi))
        t1 = collect(1:length(yi))
        m = estimate_xtcointreg(yi, xi, id1, t1; method=:fmols, pooling=:pooled,
                                trend=:const, bandwidth=bw)
        @test length(S.coef(m)) == 1
        @test S.coef(m)[1] ≈ slope_single rtol=0 atol=1e-10
        @test S.vcov(m)[1, 1] ≈ vcov_slope_single rtol=0 atol=1e-10
    end

    # ---------------------------------------------------------------------
    # Pooled estimators on the full panel — sanity + covariance structure
    # ---------------------------------------------------------------------
    @testset "pooled FMOLS on full panel" begin
        m = estimate_xtcointreg(yv, xv, idv, tv; method=:fmols, pooling=:pooled,
                                trend=:const, bandwidth=bw)
        @test m.pooling == :pooled
        @test m.method == :fmols
        @test length(S.coef(m)) == 1
        @test isapprox(S.coef(m)[1], 1.5; atol=0.1)        # near true β0
        # Pedroni (2000) pooled FMOLS covariance is (Σ wᵢ S_xx,i)⁻¹ — SPD, so se > 0.
        @test m.se[1] > 0
        @test isfinite(m.tstats[1])
        @test size(m.unit_coefs) == (1, N)                 # per-unit slopes stored
    end

    @testset "pooled DOLS on full panel" begin
        m = estimate_xtcointreg(yv, xv, idv, tv; method=:dols, pooling=:pooled,
                                trend=:const, bandwidth=bw)
        @test length(S.coef(m)) == 1
        @test isapprox(S.coef(m)[1], 1.5; atol=0.1)
        @test m.se[1] > 0
        @test m.nobs == sum(m.T_i)
    end

    # ---------------------------------------------------------------------
    # PanelData dispatch matches the long-format dispatch
    # ---------------------------------------------------------------------
    @testset "PanelData dispatch ≡ long-format dispatch" begin
        df = DataFrame(country=idv, year=tv, ly=yv, lx=xv)
        pd = xtset(df, :country, :year)
        mp = estimate_xtcointreg(pd, :ly, :lx; method=:fmols, pooling=:group,
                                 trend=:const, bandwidth=bw)
        ml = estimate_xtcointreg(yv, xv, idv, tv; method=:fmols, pooling=:group,
                                 trend=:const, bandwidth=bw)
        @test S.coef(mp) ≈ S.coef(ml) rtol=0 atol=1e-12
        @test mp.tstats ≈ ml.tstats rtol=0 atol=1e-10
        @test S.coefnames(mp) == ["const", "lx"]           # uses panel var names
    end

    # ---------------------------------------------------------------------
    # Display / API smoke tests
    # ---------------------------------------------------------------------
    @testset "report / refs / show" begin
        m = estimate_xtcointreg(yv, xv, idv, tv; method=:fmols, pooling=:group,
                                trend=:const, bandwidth=bw)
        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test occursin("Panel Cointegrating Regression", s)
        @test occursin("Pedroni", s) || occursin("group-mean", s)

        io2 = IOBuffer()
        refs(io2, m)
        rs = String(take!(io2))
        @test occursin("Pedroni", rs)
        @test S.nobs(m) == length(yv)
        @test S.islinear(m)
        ci = S.confint(m)
        @test size(ci) == (2, 2)
        @test all(ci[:, 1] .<= ci[:, 2])
    end

    # ---------------------------------------------------------------------
    # Argument validation
    # ---------------------------------------------------------------------
    @testset "argument validation" begin
        @test_throws ArgumentError estimate_xtcointreg(yv, xv, idv, tv; method=:bogus)
        @test_throws ArgumentError estimate_xtcointreg(yv, xv, idv, tv; pooling=:bogus)
        @test_throws ArgumentError estimate_xtcointreg(yv, xv, idv, tv; trend=:bogus)
    end
end
