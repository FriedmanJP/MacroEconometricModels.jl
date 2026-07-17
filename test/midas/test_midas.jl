# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# MIDAS regression tests (EV-01, issue #409).
#
# Oracle strategy (R `midasr` is NOT installed in this environment, so no live
# cross-implementation check is possible; the repo oracle harness in test/oracle/
# is Octave/Python only). Correctness is proven with analytic-property oracles
# and a fixed-seed parameter-recovery end-to-end oracle — no invented numerics:
#   (O1) `_midas_weights` sums to 1 for every kind.
#   (O2) exp-Almon at θ=0 equals fill(1/K, K) exactly (degenerate ⇒ equal weights).
#   (O3) analytic Jacobians agree with ForwardDiff.jacobian to machine tolerance.
#   (O4) weights=:umidas equals plain OLS (estimate_reg) on the stacked lags to 1e-8.
#   (O5) `_align_hf` block layout / ragged-edge trimming identities.
#   (O6) fixed-seed simulated monthly→quarterly DGP (m=3,K=6,:expalmon,p_ar=1):
#        estimate_midas recovers the true weight curve & coefficients; report() prints.
#
# The equivalent R reference (were midasr installed) would be, for the exp-Almon
# case in O6 (chronological x, quarterly y):
#   library(midasr)
#   midas_r(y ~ mls(x, 0:5, 3, nealmon) + mls(y, 1:1, 1),
#           start = list(x = c(1, 0.3, -0.05)))
# — generate offline and paste with provenance if midasr becomes available.

using Test
using MacroEconometricModels
using MacroEconometricModels: _midas_weights, _midas_weights_jac, _align_hf,
                              _ar_block, dof_residual
using LinearAlgebra, Statistics, Random, Distributions
import ForwardDiff

@testset "MIDAS Regression (EV-01)" begin

    # =========================================================================
    # O1 — normalization: every weight kind sums to 1
    # =========================================================================
    @testset "weights sum to 1 (all kinds)" begin
        K = 8
        cases = [
            (:expalmon, [0.3, -0.05]),
            (:expalmon, [-0.2, 0.01]),
            (:beta2,    [1.0, 4.0]),
            (:beta2,    [2.0, 2.0]),
            (:beta3,    [1.0, 5.0, 0.02]),
            (:almon,    [1.0, -0.1, 0.01]),
            (:umidas,   Float64[]),
        ]
        for (kind, th) in cases
            w = _midas_weights(th, K, kind)
            @test length(w) == K
            @test sum(w) ≈ 1.0 atol = 1e-12
        end
    end

    # =========================================================================
    # O2 — exp-Almon degenerate case: θ=0 ⇒ exactly equal weights
    # =========================================================================
    @testset "exp-Almon θ=0 ⇒ fill(1/K, K) exactly" begin
        for K in (3, 6, 12)
            w = _midas_weights([0.0, 0.0], K, :expalmon)
            @test w == fill(1.0 / K, K)   # exact, not approx
        end
    end

    # =========================================================================
    # O3 — analytic Jacobian ≡ ForwardDiff.jacobian
    # =========================================================================
    @testset "analytic Jacobian ≡ ForwardDiff" begin
        K = 7
        for (kind, th) in [(:expalmon, [0.25, -0.04]),
                           (:expalmon, [-0.3, 0.02]),
                           (:beta2,    [1.5, 3.0]),
                           (:beta2,    [0.8, 2.5]),
                           (:beta3,    [1.2, 4.0, 0.03])]
            Ja = _midas_weights_jac(th, K, kind)
            Jf = ForwardDiff.jacobian(t -> _midas_weights(t, K, kind), th)
            @test size(Ja) == size(Jf)
            @test Ja ≈ Jf atol = 1e-9
        end
        # Jacobian columns sum to ~0 (since Σw ≡ 1 ⇒ Σ ∂w/∂θ = 0)
        Ja = _midas_weights_jac([0.25, -0.04], K, :expalmon)
        @test all(abs.(vec(sum(Ja, dims=1))) .< 1e-10)
    end

    # =========================================================================
    # O5 — frequency alignment structure & ragged-edge trimming
    # =========================================================================
    @testset "_align_hf block layout & trimming" begin
        m, K = 3, 4
        T_lf = 10
        # HF series = 1,2,...,m*T_lf so blocks are easy to read
        x = collect(1.0:(m * T_lf))
        y = collect(1.0:T_lf)
        Xlags, retained = _align_hf(y, x; m=m, K=K)
        # Last LF period anchors to last HF obs (= m*T_lf).
        # Most-recent-first: row for t has [x[t*m], x[t*m-1], ...].
        @test retained[end] == T_lf
        @test Xlags[end, :] == [Float64(m * T_lf) - j for j in 0:(K - 1)]
        # First retained t must have a full K-block: t*m - K + 1 ≥ 1 ⇒ t ≥ ceil(K/m).
        @test retained[1] == cld(K, m)
        @test size(Xlags, 1) == length(retained)
        # Each row is strictly most-recent-first (descending here since x increasing).
        @test all(Xlags[i, 1] > Xlags[i, 2] for i in 1:size(Xlags, 1))

        # Ragged front: extra leading HF obs are dropped, alignment unchanged.
        x2 = vcat([-99.0, -98.0], x)   # 2 spurious early HF obs
        Xlags2, retained2 = _align_hf(y, x2; m=m, K=K)
        @test Xlags2[end, :] == Xlags[end, :]   # last block identical (anchored at end)
    end

    # =========================================================================
    # O4 — U-MIDAS ≡ plain OLS on stacked lags (to 1e-8)
    # =========================================================================
    @testset "U-MIDAS ≡ estimate_reg on stacked lags" begin
        rng = MersenneTwister(20240716)
        m, K, T_lf = 3, 5, 120
        x = randn(rng, m * T_lf)
        y = randn(rng, T_lf)

        mu = estimate_midas(y, x; m=m, K=K, weights=:umidas, p_ar=0)

        # Reconstruct the exact aligned design and run estimate_reg directly.
        Xlags, retained = _align_hf(y, x; m=m, K=K)
        Wlin, keep, y_used = _ar_block(y, retained, 0)
        Xk = Xlags[keep, :]
        M = hcat(ones(length(y_used)), Xk)     # [const, lag1..lagK]
        reg = estimate_reg(y_used, M; cov_type=:ols)

        @test mu.beta ≈ reg.beta atol = 1e-8
        @test length(mu.beta) == K + 1
        @test mu.weights_kind == :umidas
    end

    # =========================================================================
    # O6 — end-to-end fixed-seed recovery (m=3, K=6, exp-Almon, p_ar=1)
    # =========================================================================
    @testset "exp-Almon ADL-MIDAS recovery + report()" begin
        rng = MersenneTwister(424242)
        m, K = 3, 6
        T_lf = 320
        θ_true = [0.35, -0.06]
        w_true = _midas_weights(θ_true, K, :expalmon)
        β0, β1, ρ = 0.5, 2.0, 0.4
        σ = 0.20

        x = randn(rng, m * T_lf)
        # aggregated MIDAS signal for each fully-covered LF period
        s = zeros(T_lf)
        for t in 1:T_lf
            hi = t * m
            hi - K + 1 >= 1 || continue
            block = x[hi:-1:(hi - K + 1)]
            s[t] = dot(block, w_true)
        end
        y = zeros(T_lf)
        for t in 2:T_lf
            y[t] = β0 + β1 * s[t] + ρ * y[t - 1] + σ * randn(rng)
        end

        model = estimate_midas(y, x; m=m, K=K, weights=:expalmon, p_ar=1)

        @test model isa MidasModel
        @test model.converged
        # Recovered weight curve matches the DGP (identified through normalization).
        @test norm(model.w - w_true) < 0.04
        @test sum(model.w) ≈ 1.0 atol = 1e-10
        # Coefficients: β = [const, β₁, ρ]
        @test model.beta[2] ≈ β1 atol = 0.20   # HF loading
        @test model.beta[3] ≈ ρ  atol = 0.10   # AR(1)
        @test model.r2 > 0.8

        # report() prints without error and produces content
        buf = IOBuffer()
        show(buf, model)
        out = String(take!(buf))
        @test occursin("MIDAS Regression", out)
        @test occursin("Exponential Almon", out)
        @test occursin("Weight parameters", out)

        # StatsAPI plumbing
        @test nobs(model) == length(model.y)
        @test dof(model) == length(model.beta) + length(model.theta)
        @test length(stderror(model)) == dof(model)
        @test midas_weights(model) == model.w

        # ------- forecast: point + NLS prediction interval -------
        Xnew = randn(rng, K)                 # fresh HF block, most-recent-first
        fc = forecast(model, Xnew)
        @test fc isa MidasForecast
        @test length(fc.forecast) == 1
        @test fc.ci_lower[1] < fc.forecast[1] < fc.ci_upper[1]
        @test fc.se[1] > 0
        buff = IOBuffer(); show(buff, fc)
        @test occursin("MIDAS Forecast", String(take!(buff)))
    end

    # =========================================================================
    # Beta and polynomial-Almon estimators run end-to-end
    # =========================================================================
    @testset "Beta & Almon estimation run" begin
        rng = MersenneTwister(999)
        m, K, T_lf = 3, 6, 200
        w_true = _midas_weights([1.0, 4.0], K, :beta2)
        x = randn(rng, m * T_lf)
        s = zeros(T_lf)
        for t in 1:T_lf
            hi = t * m
            hi - K + 1 >= 1 || continue
            s[t] = dot(x[hi:-1:(hi - K + 1)], w_true)
        end
        y = 1.0 .+ 1.5 .* s .+ 0.15 .* randn(rng, T_lf)

        mb = estimate_midas(y, x; m=m, K=K, weights=:beta2, p_ar=0)
        @test mb isa MidasModel
        @test sum(mb.w) ≈ 1.0 atol = 1e-10
        @test mb.r2 > 0.7

        ma = estimate_midas(y, x; m=m, K=K, weights=:almon, p_ar=0, poly_degree=2)
        @test ma isa MidasModel
        @test sum(ma.w) ≈ 1.0 atol = 1e-10
        @test length(ma.theta) == 3   # degree 2 ⇒ 3 params
    end

    # =========================================================================
    # midas_weights public wrapper & plotting smoke test
    # =========================================================================
    @testset "public wrapper + plotting" begin
        w = midas_weights([0.3, -0.05], 6; kind=:expalmon)
        @test sum(w) ≈ 1.0 atol = 1e-12
        @test w ≈ _midas_weights([0.3, -0.05], 6, :expalmon)

        rng = MersenneTwister(7)
        m, K, T_lf = 3, 5, 90
        x = randn(rng, m * T_lf); y = randn(rng, T_lf)
        model = estimate_midas(y, x; m=m, K=K, weights=:expalmon, p_ar=0)
        p1 = plot_result(model; view=:weights)
        p2 = plot_result(model; view=:fit)
        @test p1 isa PlotOutput
        @test p2 isa PlotOutput
        @test_throws ArgumentError plot_result(model; view=:nonsense)
    end
end
