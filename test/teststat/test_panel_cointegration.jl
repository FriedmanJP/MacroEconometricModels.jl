# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-21 (#429): Panel cointegration tests — Pedroni / Kao / Westerlund /
# Fisher-Johansen.
#
# Oracle discipline (no invented numerics; see the EV-21 spec's oracle_strategy).
# A live cross-implementation reference is NOT used because the local R
# toolchain has no panel-cointegration package (only `cointReg`, a single-series
# cointegrating-regression estimator; `plm`/`urca`/`pco` are absent and, per the
# EV-20 note, cannot be compiled here) and Stata is unavailable. Provenance is
# instead preserved through:
#   (1) Machine-tolerance RECONSTRUCTION oracles. The reported p-values are exact
#       functions of the standardized statistics and the tail convention, so we
#       reconstruct them from Distributions' Normal()/Chisq() CDFs. This pins the
#       SIGN TRAP directly: Pedroni panel-v (index 1) is RIGHT-tailed
#       (p = ccdf), the other six Pedroni statistics + all Kao + all Westerlund
#       are LEFT-tailed (p = cdf); Fisher Maddala-Wu is upper-tailed χ²(2N).
#   (2) Published-table spot asserts. The Pedroni (1999, 2004) Table 2 (μ, v)
#       adjustment moments and Kao's (1999) closed-form standardization constants
#       are transcribed from the source papers / the `pco` reference package and
#       spot-checked against the hard-coded table.
#   (3) Degenerate equivalence. `fisher_johansen_test` with N = 1 reproduces the
#       single-unit `johansen_test` trace/max p-values EXACTLY (Maddala-Wu
#       P_r = -2 ln p_{1,r}, whose χ²(2) upper tail is p_{1,r}; Choi
#       Z_r = Φ⁻¹(p_{1,r})/√1, whose normal CDF is p_{1,r}).
#   (4) Analytic DGP behaviour. A cointegrated panel rejects H0(no cointegration)
#       while independent random walks do not (Kao/Westerlund left-tailed; Fisher
#       recovers the true rank).

using Test
using MacroEconometricModels
using StatsAPI
using DataFrames
using Random
using Statistics
using Distributions

const _M = MacroEconometricModels

# Build a PanelData from a T×N response `Y` and a T×N×k regressor array `X`.
function _mk_coint_panel(Y::AbstractMatrix, X::AbstractArray{<:Any,3})
    Tobs, N = size(Y)
    k = size(X, 3)
    idv = Int[]; tv = Int[]; yy = Float64[]
    xx = [Float64[] for _ in 1:k]
    for i in 1:N, t in 1:Tobs
        push!(idv, i); push!(tv, t); push!(yy, Y[t, i])
        for j in 1:k
            push!(xx[j], X[t, i, j])
        end
    end
    df = DataFrame(id = idv, t = tv, y = yy)
    for j in 1:k
        df[!, Symbol("x", j)] = xx[j]
    end
    xtset(df, :id, :t)
end

# Cointegrated DGP: y_it = β' x_it + stationary AR(1) error; x_it random walks.
function _coint_dgp(rng, Tobs, N; k = 1, beta = ones(k), rho_e = 0.3)
    Y = zeros(Tobs, N)
    X = zeros(Tobs, N, k)
    for i in 1:N
        e = zeros(Tobs)
        for t in 2:Tobs
            e[t] = rho_e * e[t-1] + randn(rng)
        end
        yi = copy(e)
        for j in 1:k
            xj = cumsum(randn(rng, Tobs))
            X[:, i, j] = xj
            yi .+= beta[j] .* xj
        end
        Y[:, i] = yi
    end
    Y, X
end

# No-cointegration DGP: y and each x are independent random walks (spurious).
function _nocoint_dgp(rng, Tobs, N; k = 1)
    Y = zeros(Tobs, N)
    X = zeros(Tobs, N, k)
    for i in 1:N
        Y[:, i] = cumsum(randn(rng, Tobs))
        for j in 1:k
            X[:, i, j] = cumsum(randn(rng, Tobs))
        end
    end
    Y, X
end

# Two-series panel for Fisher-Johansen: b_it = a_it + noise (coint) or an
# independent random walk (no coint); a_it always a random walk.
function _fj_panel(rng, Tobs, N; coint = true, noise = 0.5)
    idv = Int[]; tv = Int[]; av = Float64[]; bv = Float64[]
    for i in 1:N
        a = cumsum(randn(rng, Tobs))
        b = coint ? a .+ noise .* randn(rng, Tobs) : cumsum(randn(rng, Tobs))
        for t in 1:Tobs
            push!(idv, i); push!(tv, t); push!(av, a[t]); push!(bv, b[t])
        end
    end
    xtset(DataFrame(id = idv, t = tv, a = av, b = bv), :id, :t)
end

@testset "Panel Cointegration Tests (EV-21)" begin

    # =========================================================================
    # (2) Published-table spot asserts — Pedroni (1999, 2004) Table 2 (μ, v).
    # Transcribed from the `pco` R package (arrays stamm/stavv), which follows
    # Pedroni (1999) Table 2 (p. 666) verbatim. Row order of the 7 statistics:
    # panel-v, panel-ρ, panel-t, panel-ADF, group-ρ, group-t, group-ADF.
    # =========================================================================
    @testset "Pedroni (1999) Table 2 moments" begin
        mu, v = _M._pedroni_moments(1, :constant)   # k=1, intercept
        @test mu == [11.754, -9.495, -2.177, -2.177, -12.938, -2.453, -2.453]
        @test v == [104.546, 57.61, 0.964, 0.964, 51.49, 0.618, 0.618]
        # panel-t and panel-ADF share moments (rows 3 & 4); likewise group-t /
        # group-ADF (rows 6 & 7) — a structural property of Pedroni's table.
        @test mu[3] == mu[4]
        @test mu[6] == mu[7]
        @test v[3] == v[4]
        @test v[6] == v[7]
        # A different (trend, k) slice.
        mu2, v2 = _M._pedroni_moments(2, :trend)
        @test mu2[1] == 24.556
        @test v2[1] == 198.167
        # :none case, k=1, panel-v mean.
        mu0, _ = _M._pedroni_moments(1, :none)
        @test mu0[1] == 6.982
        # Out-of-range regressor count errors.
        @test_throws ArgumentError _M._pedroni_moments(7, :constant)
        @test_throws ArgumentError _M._pedroni_moments(0, :constant)
    end

    # =========================================================================
    # (1) Machine-tolerance reconstruction oracle — pins the tail convention.
    # =========================================================================
    @testset "Pedroni: p-values reconstruct from N(0,1), correct tails" begin
        rng = MersenneTwister(2024)
        Y, X = _coint_dgp(rng, 60, 20)
        r = pedroni_test(_mk_coint_panel(Y, X), :y, :x1; trend = :constant)
        @test length(r.statistics) == 7
        @test r.names[1] == "panel-v"
        # SIGN TRAP: panel-v (index 1) is RIGHT-tailed.
        @test r.pvalues[1] ≈ ccdf(Normal(), r.statistics[1]) rtol = 1e-12
        # The other six are LEFT-tailed.
        for s in 2:7
            @test r.pvalues[s] ≈ cdf(Normal(), r.statistics[s]) rtol = 1e-12
        end
        # Standardization: std = (raw - μ√N)/√v with the stored moments.
        N = r.n_units
        for s in 1:7
            @test r.statistics[s] ≈ (r.raw[s] - r.mu[s] * sqrt(N)) / sqrt(r.v[s]) rtol = 1e-12
        end
    end

    @testset "Kao: p-values reconstruct from N(0,1) (all left-tailed)" begin
        rng = MersenneTwister(11)
        Y, X = _coint_dgp(rng, 60, 20)
        r = kao_test(_mk_coint_panel(Y, X), :y, :x1)
        @test length(r.statistics) == 5
        @test r.names == ["DFrho", "DFt", "DFrho_star", "DFt_star", "ADF"]
        for s in 1:5
            @test r.pvalues[s] ≈ cdf(Normal(), r.statistics[s]) rtol = 1e-12
        end
        # Kao (1999) closed-form constants for the unadjusted DFρ / DFt:
        #   DFρ = (√N T (ρ̂-1) + 3√N)/√(10.2),   10.2 = 51/5   (Kao 1999, Thm 4)
        #   DFt = √1.25 · t_ρ + √(1.875 N),      1.25 = 5/4, 1.875 = 15/8
        N = r.n_units; T = r.nobs
        dfrho_manual = (sqrt(N) * T * (r.rho - 1) + 3 * sqrt(N)) / sqrt(51 / 5)
        @test r.statistics[1] ≈ dfrho_manual rtol = 1e-10
        dft_manual = sqrt(5 / 4) * r.t_rho + sqrt(15 / 8 * N)
        @test r.statistics[2] ≈ dft_manual rtol = 1e-10
    end

    @testset "Westerlund: p-values reconstruct from N(0,1) (all left-tailed)" begin
        rng = MersenneTwister(101)
        Y, X = _coint_dgp(rng, 60, 20)
        r = westerlund_test(_mk_coint_panel(Y, X), :y, :x1)
        @test r.names == ["Gt", "Ga", "Pt", "Pa"]
        for s in 1:4
            @test r.pvalues[s] ≈ cdf(Normal(), r.statistics[s]) rtol = 1e-12
        end
        # No bootstrap requested ⇒ bootstrap p-values are NaN.
        @test all(isnan, r.bootstrap_pvalues)
    end

    # =========================================================================
    # (3) Degenerate equivalence — Fisher-Johansen with N=1 ≡ single johansen.
    # =========================================================================
    @testset "Fisher-Johansen N=1 ≡ single johansen_test (exact)" begin
        rng = MersenneTwister(7)
        Tobs = 80
        a = cumsum(randn(rng, Tobs))
        b = a .+ 0.5 .* randn(rng, Tobs)
        c = cumsum(randn(rng, Tobs))
        pd = xtset(DataFrame(id = fill(1, Tobs), t = 1:Tobs, a = a, b = b, c = c), :id, :t)
        jt = johansen_test(hcat(a, b, c), 2; deterministic = :constant)

        fj = fisher_johansen_test(pd, :a, :b, :c; lags = 2, deterministic = :constant, combine = :mw)
        @test fj.n_units == 1
        @test fj.n_series == 3
        # Maddala-Wu: ccdf(Chisq(2), -2 ln p_1) == p_1 exactly.
        @test fj.trace_pvalues ≈ jt.trace_pvalues rtol = 1e-9
        @test fj.max_pvalues ≈ jt.max_eigen_pvalues rtol = 1e-9
        @test fj.trace_statistics ≈ -2 .* log.(jt.trace_pvalues) rtol = 1e-9

        # Choi inverse-normal at N=1: cdf(Normal(), Φ⁻¹(p_1)) == p_1 exactly.
        fjc = fisher_johansen_test(pd, :a, :b, :c; lags = 2, combine = :choi)
        @test fjc.trace_pvalues ≈ jt.trace_pvalues rtol = 1e-9
        @test fjc.max_pvalues ≈ jt.max_eigen_pvalues rtol = 1e-9
    end

    @testset "Fisher-Johansen: MW upper-tailed χ²(2N) reconstruction" begin
        rng = MersenneTwister(303)
        fj = fisher_johansen_test(_fj_panel(rng, 60, 12; coint = true), :a, :b; lags = 2)
        N = fj.n_units
        for r in eachindex(fj.ranks)
            @test fj.trace_pvalues[r] ≈ ccdf(Chisq(2N), fj.trace_statistics[r]) rtol = 1e-12
            @test fj.max_pvalues[r] ≈ ccdf(Chisq(2N), fj.max_statistics[r]) rtol = 1e-12
        end
        @test fj.ranks == [0, 1]
    end

    # =========================================================================
    # (4) Analytic DGP behaviour — cointegration rejects, random walks do not.
    # =========================================================================
    @testset "Cointegrated panel rejects H0(no cointegration)" begin
        rng = MersenneTwister(42)
        Yc, Xc = _coint_dgp(rng, 60, 20)
        pdc = _mk_coint_panel(Yc, Xc)

        pc = pedroni_test(pdc, :y, :x1)
        # The workhorse group statistics reject strongly.
        @test pc.pvalues[4] < 0.01     # panel-ADF
        @test pc.pvalues[6] < 0.01     # group-t
        @test pc.pvalues[7] < 0.01     # group-ADF

        kc = kao_test(pdc, :y, :x1)
        @test kc.pvalues[5] < 0.01     # ADF
        @test all(kc.pvalues .< 0.05)

        wc = westerlund_test(pdc, :y, :x1)
        @test all(wc.pvalues .< 0.05)  # Gt, Ga, Pt, Pa
    end

    @testset "Independent random walks fail to reject H0" begin
        rng = MersenneTwister(42)
        Yn, Xn = _nocoint_dgp(rng, 60, 20)
        pdn = _mk_coint_panel(Yn, Xn)

        # Group statistics (most reliable finite-sample) do not reject.
        pn = pedroni_test(pdn, :y, :x1)
        @test pn.pvalues[5] > 0.10     # group-ρ
        @test pn.pvalues[6] > 0.10     # group-t
        @test pn.pvalues[7] > 0.10     # group-ADF

        kn = kao_test(pdn, :y, :x1)
        @test kn.pvalues[5] > 0.10     # ADF

        wn = westerlund_test(pdn, :y, :x1)
        @test wn.pvalues[1] > 0.10     # Gt
        @test wn.pvalues[2] > 0.10     # Ga

        # Fisher-Johansen recovers rank 0 for independent walks, rank>=1 when
        # cointegrated.
        rng2 = MersenneTwister(55)
        @test fisher_johansen_test(_fj_panel(rng2, 60, 15; coint = false), :a, :b; lags = 2).rank == 0
        @test fisher_johansen_test(_fj_panel(rng2, 60, 15; coint = true), :a, :b; lags = 2).rank >= 1
    end

    # =========================================================================
    # Deterministic specifications, multiple regressors, and options.
    # =========================================================================
    @testset "Trend / none deterministics and k=2 regressors" begin
        rng = MersenneTwister(88)
        Yc, Xc = _coint_dgp(rng, 70, 15; k = 2, beta = [1.0, -0.5])
        pd = _mk_coint_panel(Yc, Xc)
        for tr in (:none, :constant, :trend)
            r = pedroni_test(pd, :y, :x1, :x2; trend = tr)
            @test r.n_regressors == 2
            @test r.trend == tr
            @test r.pvalues[7] < 0.05          # group-ADF still rejects
        end
        @test kao_test(pd, :y, :x1, :x2).n_regressors == 2
        wr = westerlund_test(pd, :y, :x1, :x2; trend = :trend, lags = 1, leads = 1)
        @test wr.n_regressors == 2
        @test wr.leads == 1
    end

    @testset "Westerlund seeded bootstrap is reproducible" begin
        rng = MersenneTwister(909)
        Yc, Xc = _coint_dgp(rng, 50, 12)
        pd = _mk_coint_panel(Yc, Xc)
        w1 = westerlund_test(pd, :y, :x1; bootstrap = 30, seed = 314)
        w2 = westerlund_test(pd, :y, :x1; bootstrap = 30, seed = 314)
        @test w1.bootstrap_pvalues == w2.bootstrap_pvalues
        @test all(0 .< w1.bootstrap_pvalues .<= 1)
        # Different seed generally gives a (possibly) different draw but valid p.
        w3 = westerlund_test(pd, :y, :x1; bootstrap = 30, seed = 271)
        @test all(0 .< w3.bootstrap_pvalues .<= 1)
    end

    # =========================================================================
    # StatsAPI interface, lag options, and input validation.
    # =========================================================================
    @testset "StatsAPI accessors" begin
        rng = MersenneTwister(5)
        Yc, Xc = _coint_dgp(rng, 55, 10)
        pd = _mk_coint_panel(Yc, Xc)
        pr = pedroni_test(pd, :y, :x1)
        @test StatsAPI.nobs(pr) == 55
        @test StatsAPI.dof(pr) == 10
        @test StatsAPI.pvalue(pr) == minimum(pr.pvalues)
        kr = kao_test(pd, :y, :x1)
        @test StatsAPI.pvalue(kr) == minimum(kr.pvalues)
        fj = fisher_johansen_test(_fj_panel(rng, 55, 8; coint = true), :a, :b; lags = 2)
        @test StatsAPI.dof(fj) == 2 * fj.n_units
        @test StatsAPI.pvalue(fj) == fj.trace_pvalues[1]
    end

    @testset "Explicit lag / bandwidth options" begin
        rng = MersenneTwister(17)
        Yc, Xc = _coint_dgp(rng, 60, 12)
        pd = _mk_coint_panel(Yc, Xc)
        pr = pedroni_test(pd, :y, :x1; lags = 3, adf_lags = 3)
        @test pr.bandwidth == 3
        @test pr.adf_lags == 3
        kr = kao_test(pd, :y, :x1; lags = 2, kernel_lags = 4)
        @test kr.lags == 2
        @test kr.kernel_lags == 4
    end

    @testset "Input validation" begin
        rng = MersenneTwister(9)
        Yc, Xc = _coint_dgp(rng, 40, 6)
        pd = _mk_coint_panel(Yc, Xc)
        @test_throws ArgumentError pedroni_test(pd, :y)                    # no regressor
        @test_throws ArgumentError pedroni_test(pd, :y, :x1; trend = :bad)
        @test_throws ArgumentError kao_test(pd, :y)
        @test_throws ArgumentError westerlund_test(pd, :y)
        @test_throws ArgumentError westerlund_test(pd, :y, :x1; trend = :bad)
        @test_throws ArgumentError fisher_johansen_test(pd, :y)           # needs >=2 series
        @test_throws ArgumentError fisher_johansen_test(pd, :y, :x1; combine = :bad)
        @test_throws ArgumentError fisher_johansen_test(pd, :y, :x1; lags = 0)
    end

    @testset "show / refs render without error" begin
        rng = MersenneTwister(21)
        Yc, Xc = _coint_dgp(rng, 50, 10)
        pd = _mk_coint_panel(Yc, Xc)
        for r in (pedroni_test(pd, :y, :x1), kao_test(pd, :y, :x1),
                  westerlund_test(pd, :y, :x1))
            io = IOBuffer()
            show(io, r)
            @test !isempty(String(take!(io)))
            rio = IOBuffer()
            refs(rio, r)
            @test !isempty(String(take!(rio)))
        end
        fj = fisher_johansen_test(_fj_panel(rng, 50, 8; coint = true), :a, :b; lags = 2)
        io = IOBuffer(); show(io, fj)
        @test occursin("Fisher", String(take!(io)))
    end

end
