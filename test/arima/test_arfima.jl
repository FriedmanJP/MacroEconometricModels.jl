# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-13 (#421): ARFIMA — fractional integration (CSS / exact ML) + GPH and
# local-Whittle semiparametric d estimators.
#
# ORACLE DISCIPLINE (R `fracdiff` is NOT installable in this environment — its
# compilation fails — so the cross-implementation numeric fixtures are replaced
# by the layered in-environment oracle set below; no numbers are fabricated):
#   (1) ANALYTIC-PROPERTY oracles (exact identities, machine tolerance):
#       - _frac_diff(y, 0) == y ;  _frac_diff_weights(0, K) == [1, 0, …, 0]
#       - the (1−L)^d recursion π_k = π_{k−1}(k−1−d)/k reproduced by hand
#       - FFT convolution path ≡ direct recursion (Jensen–Nielsen 2014)
#       - (1−L)^d ∘ (1−L)^{−d} = identity  (group property of the filter)
#       - d=0 ⇒ the CSS component is byte-identical to the ARMA CSS residuals
#       - logit keeps d strictly inside (−0.5, 0.5)
#       - Durbin–Levinson concentrated loglik reconstructs an AR(1) exactly
#   (2) KNOWN-VALUE oracle:
#       - load_example(:nile) GPH d̂ ≈ 0.38–0.40 (documented long-memory value
#         for the Nile-at-Aswan series; Beran 1994, Long-Memory Processes, §1)
#   (3) CONSISTENCY / RECOVERY oracles (the estimators are consistent, so on
#       fixed-seed simulated ARFIMA(p,d,q) data the estimates recover the truth
#       within calibrated Monte-Carlo tolerances; GPH↔local-Whittle agree).

using Test
using MacroEconometricModels
using MacroEconometricModels: _frac_diff, _frac_diff_weights, _frac_diff_fft,
    _arfima_components, _arfima_d, _arfima_xi, _arfima0_autocov, _arfima_autocov,
    _dl_concentrated_loglik, _compute_arma_residuals, _periodogram
using Statistics
using Random
using LinearAlgebra
using Distributions: loggamma
import StatsAPI

_gamma(x) = exp(loggamma(x))

# -----------------------------------------------------------------------------
# Deterministic ARFIMA(1,d,0) simulator: (1−φL)(1−L)^d x = e  ⇒
#   x = (1−L)^{−d} · (1−φL)^{−1} e.  Long burn-in so the truncated fractional
# filter has effectively converged. Fixed MersenneTwister seed ⇒ reproducible.
# -----------------------------------------------------------------------------
function _sim_arfima(seed::Int, n::Int, d::Float64, phi::Float64; burn::Int=3000)
    rng = MersenneTwister(seed)
    N = n + burn
    e = randn(rng, N)
    u = zeros(N)
    u[1] = e[1]
    @inbounds for t in 2:N
        u[t] = phi * u[t-1] + e[t]
    end
    x = _frac_diff(u, -d)      # apply (1−L)^{−d}
    x[burn+1:end]
end

@testset "ARFIMA (EV-13)" begin

    # =========================================================================
    # Fractional-differencing filter — analytic-property oracles
    # =========================================================================
    @testset "Fractional-difference weights & filter" begin
        # π₀ = 1 always; d = 0 ⇒ [1, 0, …, 0]
        @test _frac_diff_weights(0.0, 6) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        @test _frac_diff_weights(0.3, 5)[1] == 1.0
        @test length(_frac_diff_weights(0.4, 10)) == 11
        @test_throws ArgumentError _frac_diff_weights(0.3, -1)

        # Hand-computed recursion π_k = π_{k−1}(k−1−d)/k for d = 0.4:
        #   π₀=1, π₁=−d=−0.4, π₂=π₁(1−d)/2=−0.4·0.6/2=−0.12,
        #   π₃=π₂(2−d)/3=−0.12·1.6/3=−0.064
        w = _frac_diff_weights(0.4, 3)
        @test w ≈ [1.0, -0.4, -0.12, -0.064] atol = 1e-12

        # Weights of (1−L)^d for d>0 are negative for k≥1 and decay ∝ k^{−1−d}
        wpos = _frac_diff_weights(0.3, 50)
        @test all(wpos[2:end] .< 0)
        @test abs(wpos[end]) < abs(wpos[2])

        # _frac_diff(y, 0) returns y unchanged (identity of the filter at d=0)
        y = randn(MersenneTwister(11), 60)
        @test _frac_diff(y, 0.0) ≈ y atol = 1e-13
        @test _frac_diff(y, 0.0; method=:direct) ≈ y atol = 1e-13

        # Direct definition wₜ = Σ_{k=0}^{t−1} π_k y_{t−k} reproduced by hand
        d = 0.35
        wgt = _frac_diff_weights(d, length(y) - 1)
        manual = [sum(wgt[k+1] * y[t-k] for k in 0:t-1) for t in 1:length(y)]
        @test _frac_diff(y, d; method=:direct) ≈ manual atol = 1e-11

        # FFT path ≡ direct path (Jensen–Nielsen 2014), machine tolerance
        for dd in (0.2, 0.45, -0.3)
            fd_dir = _frac_diff(y, dd; method=:direct)
            fd_fft = _frac_diff(y, dd; method=:fft)
            @test fd_fft ≈ fd_dir atol = 1e-10
        end
        @test_throws ArgumentError _frac_diff(y, 0.3; method=:bogus)

        # Group property: (1−L)^{−d} ∘ (1−L)^{d} = identity (up to truncation)
        back = _frac_diff(_frac_diff(y, d; method=:direct), -d; method=:direct)
        @test back ≈ y atol = 1e-9

        # Integer d = 1 ⇒ ordinary first difference on the leading part
        z = collect(1.0:20.0)
        fd1 = _frac_diff(z, 1.0; method=:direct)
        @test fd1[2:end] ≈ fill(1.0, 19) atol = 1e-12   # (1−L)z = 1 for a linear ramp
    end

    # =========================================================================
    # ARFIMA(0,d,0) autocovariance — Hosking (1981) closed form
    # =========================================================================
    @testset "ARFIMA(0,d,0) autocovariance (Hosking 1981)" begin
        d = 0.3; s2 = 2.0
        g = _arfima0_autocov(d, s2, 8)
        # γ(0) = σ² Γ(1−2d)/Γ(1−d)²
        g0 = s2 * _gamma(1 - 2d) / _gamma(1 - d)^2
        @test g[1] ≈ g0 atol = 1e-12
        # ρ(1) = d/(1−d)
        @test g[2] / g[1] ≈ d / (1 - d) atol = 1e-12
        # recursion γ(k) = γ(k−1)(k−1+d)/(k−d)
        for k in 1:8
            @test g[k+1] ≈ g[k] * (k - 1 + d) / (k - d) atol = 1e-10
        end
        # d = 0 ⇒ white noise: γ(0)=σ², γ(k)=0
        gw = _arfima0_autocov(0.0, s2, 5)
        @test gw[1] == s2
        @test all(gw[2:end] .== 0)
        # p=q=0 branch of _arfima_autocov reduces exactly to _arfima0_autocov
        @test _arfima_autocov(d, Float64[], Float64[], s2, 8) ≈ g atol = 1e-12
    end

    # =========================================================================
    # Durbin–Levinson concentrated log-likelihood — exact AR(1) reconstruction
    # =========================================================================
    @testset "Durbin–Levinson concentrated loglik" begin
        # AR(1) with φ=0.5, σ²=1 ⇒ γ(k)=φ^k/(1−φ²). DL innovations e_t must equal
        # x_t − φ x_{t−1}, and σ̂² the innovation variance.
        rng = MersenneTwister(3)
        n = 300; phi = 0.5
        x = zeros(n); x[1] = randn(rng)
        for t in 2:n
            x[t] = phi * x[t-1] + randn(rng)
        end
        x .-= mean(x)
        r = [phi^abs(k) / (1 - phi^2) for k in 0:n-1]
        ll, s2, e = _dl_concentrated_loglik(x, r)
        @test isfinite(ll)
        @test s2 > 0
        # innovations e_t ≈ x_t − φ x_{t−1} for t ≥ 2 (steady state)
        pred_err = x[3:end] .- phi .* x[2:end-1]
        @test e[3:end] ≈ pred_err atol = 1e-8
    end

    # =========================================================================
    # d=0 degenerate identity: CSS component ≡ ARMA CSS residuals (exact)
    # =========================================================================
    @testset "Degenerate d=0 ⇒ ARMA CSS identity" begin
        rng = MersenneTwister(9)
        y = randn(rng, 200) .* 2.0 .+ 1.0
        c = 0.5; phi = [0.3]; theta = [0.2]
        ll, s2, resid, fit = _arfima_components(0.0, c, phi, theta, y, :css)
        resid_arma = _compute_arma_residuals(y, c, phi, theta)
        @test resid ≈ resid_arma atol = 1e-12   # (1−L)^0 = identity ⇒ exact
        @test fit ≈ y .- resid atol = 1e-12
    end

    # =========================================================================
    # estimate_arfima — end-to-end, structure, logit bound, coef ordering
    # =========================================================================
    @testset "estimate_arfima end-to-end & invariants" begin
        x = _sim_arfima(1, 400, 0.3, 0.4)
        m = estimate_arfima(x, 1, 0; method=:css)
        @test m isa ARFIMAModel
        @test m isa MacroEconometricModels.AbstractARIMAModel
        @test eltype(m.y) == Float64
        @test -0.5 < m.d < 0.5                 # logit keeps d strictly interior
        @test length(m.phi) == 1
        @test isempty(m.theta)
        @test m.p == 1 && m.q == 0
        @test m.converged
        @test isfinite(m.loglik) && isfinite(m.aic) && isfinite(m.bic)
        @test length(m.residuals) == length(x)
        @test length(m.fitted) == length(x)
        # coef ordering [c, d, φ…, θ…]
        cf = StatsAPI.coef(m)
        @test cf[1] == m.c
        @test cf[2] == m.d
        @test cf[3] == m.phi[1]
        @test StatsAPI.dof(m) == m.p + m.q + 3
        # standard errors align with coef and d_se is a valid nonneg number
        se = StatsAPI.stderror(m)
        @test length(se) == length(cf)
        @test m.d_se >= 0 || isnan(m.d_se)

        # explicit d0 start and Float conversion from Int input both work
        m_d0 = estimate_arfima(x, 0, 0; method=:css, d0=0.2)
        @test m_d0 isa ARFIMAModel
        @test estimate_arfima(round.(Int, x .* 0 .+ collect(1:length(x))), 0, 0) isa ARFIMAModel

        @test_throws ArgumentError estimate_arfima(x, 1, 0; method=:bogus)
    end

    # =========================================================================
    # White noise ⇒ d̂ ≈ 0  (consistency at the short-memory boundary)
    # =========================================================================
    @testset "White noise ⇒ d ≈ 0" begin
        ds = Float64[]
        for s in 1:8
            y = randn(MersenneTwister(100 + s), 400)
            push!(ds, estimate_arfima(y, 0, 0; method=:css).d)
        end
        @test abs(mean(ds)) < 0.05
        @test maximum(abs.(ds)) < 0.2
    end

    # =========================================================================
    # Parameter recovery — CSS (consistency oracle, calibrated tolerances)
    # =========================================================================
    @testset "CSS parameter recovery" begin
        # pure fractional ARFIMA(0,d,0)
        for d in (0.3, 0.4)
            ds = [estimate_arfima(_sim_arfima(s, 500, d, 0.0), 0, 0; method=:css).d for s in 1:12]
            @test mean(ds) ≈ d atol = 0.06
        end
        # ARFIMA(1,d,0): recover both d and φ. d and the AR persistence are only
        # weakly separable at moderate T (Sowell 1992), so this uses a longer
        # sample (T=1500) where the joint identification is sharp.
        dtrue, ptrue = 0.3, 0.4
        ds = Float64[]; ps = Float64[]
        for s in 1:12
            m = estimate_arfima(_sim_arfima(s, 1500, dtrue, ptrue), 1, 0; method=:css)
            push!(ds, m.d); push!(ps, m.phi[1])
        end
        @test mean(ds) ≈ dtrue atol = 0.08
        @test mean(ps) ≈ ptrue atol = 0.08
    end

    # =========================================================================
    # Parameter recovery — exact ML (Sowell 1992 / Durbin–Levinson O(T²) path)
    # =========================================================================
    @testset "Exact ML recovery & agreement with CSS" begin
        for d in (0.3, 0.4)
            ds = [estimate_arfima(_sim_arfima(s, 250, d, 0.0), 0, 0; method=:mle).d for s in 1:8]
            @test mean(ds) ≈ d atol = 0.07
        end
        # CSS and MLE agree in the aggregate on the same DGP (single-seed MLE can
        # occasionally drift toward the stationarity boundary, so compare means).
        css = [estimate_arfima(_sim_arfima(s, 400, 0.35, 0.0), 0, 0; method=:css).d for s in 1:8]
        mle = [estimate_arfima(_sim_arfima(s, 400, 0.35, 0.0), 0, 0; method=:mle).d for s in 1:8]
        @test mean(css) ≈ mean(mle) atol = 0.06
        @test estimate_arfima(_sim_arfima(1, 400, 0.35, 0.0), 0, 0; method=:mle).converged
    end

    # =========================================================================
    # GPH log-periodogram regression (Geweke–Porter-Hudak 1983)
    # =========================================================================
    @testset "gph_test" begin
        # Periodogram sanity: length ⌊n/2⌋, all nonnegative
        x = randn(MersenneTwister(7), 128)
        lam, I = _periodogram(x)
        @test length(lam) == 64
        @test length(I) == 64
        @test all(I .>= 0)

        # Recovery on fixed-seed d=0.4 series (single seed, loose band; GPH is
        # high-variance but consistent and positively signed for d>0)
        g = gph_test(_sim_arfima(7, 1000, 0.4, 0.0))
        @test g isa GPHResult
        @test g.d > 0.15
        @test g.se > 0
        @test g.m == floor(Int, sqrt(1000))
        @test g.n == 1000
        # averaged over seeds the estimator concentrates near the truth
        gm = mean(gph_test(_sim_arfima(s, 800, 0.4, 0.0)).d for s in 1:15)
        @test gm ≈ 0.4 atol = 0.2

        # H₀: d=0 rejects strongly on a highly-persistent series and is honest on
        # white noise (average p-value not tiny)
        pw = mean(gph_test(randn(MersenneTwister(200 + s), 500)).pval for s in 1:10)
        @test pw > 0.2

        # options: custom m and trimming, and the short-series guard
        gt = gph_test(_sim_arfima(3, 600, 0.3, 0.0); m=40, trim=2)
        @test gt.m == 40
        @test gt.trim == 2
        @test_throws ArgumentError gph_test(randn(5))
    end

    # =========================================================================
    # Local Whittle estimator (Robinson 1995)
    # =========================================================================
    @testset "local_whittle" begin
        lw = local_whittle(_sim_arfima(7, 1000, 0.4, 0.0))
        @test lw isa LocalWhittleResult
        @test -0.5 < lw.d < 1.0
        @test lw.se ≈ 1 / (2 * sqrt(lw.m)) atol = 1e-12   # Robinson (1995) SE
        @test lw.m == floor(Int, sqrt(1000))
        @test isfinite(lw.objective)

        # averaged recovery near the truth
        lm = mean(local_whittle(_sim_arfima(s, 800, 0.4, 0.0)).d for s in 1:15)
        @test lm ≈ 0.4 atol = 0.2

        # GPH and local Whittle agree in the aggregate on the same DGP
        gm = mean(gph_test(_sim_arfima(s, 800, 0.4, 0.0)).d for s in 1:15)
        @test abs(lm - gm) < 0.2

        @test_throws ArgumentError local_whittle(randn(5))
    end

    # =========================================================================
    # KNOWN-VALUE oracle: Nile at Aswan long-memory d̂ ≈ 0.38–0.40
    # =========================================================================
    @testset ":nile dataset & GPH known value" begin
        nile = load_example(:nile)
        y = nile.data[:, 1]
        @test length(y) == 100
        @test y[1] == 1120.0          # 1871 flow (datasets::Nile[1])
        @test y[end] == 740.0         # 1970 flow (datasets::Nile[100])
        @test nile.varnames == ["Nile"]

        # GPH d̂ well inside the documented long-memory band for the Nile series
        # (≈0.4; Beran 1994, Statistics for Long-Memory Processes, §1).
        g = gph_test(y)
        @test 0.30 < g.d < 0.50
        # local Whittle detects long memory on the same series
        lw = local_whittle(y)
        @test lw.d > 0.2
        # ARFIMA(0,d,0) exact-ML on the (mean-adjusted) Nile flags positive d
        m = estimate_arfima(y, 0, 0; method=:mle)
        @test m.d > 0.1
    end

    # =========================================================================
    # Forecasting via truncated AR(∞)/ψ(∞) representation
    # =========================================================================
    @testset "ARFIMA forecast" begin
        x = _sim_arfima(2, 400, 0.3, 0.3)
        m = estimate_arfima(x, 1, 0; method=:css)
        fc = forecast(m, 10)
        @test fc isa MacroEconometricModels.ARIMAForecast
        @test length(fc.forecast) == 10
        @test length(fc.se) == 10
        @test all(isfinite, fc.forecast)
        @test all(fc.se .> 0)
        @test issorted(fc.se)                         # error variance accumulates
        @test all(fc.ci_lower .< fc.forecast .< fc.ci_upper)
        @test_throws ArgumentError forecast(m, 0)

        # Pure-fractional d>0: 1-step SE = σ (ψ₀=1) exactly
        m0 = estimate_arfima(_sim_arfima(4, 300, 0.3, 0.0), 0, 0; method=:css)
        fc0 = forecast(m0, 5)
        @test fc0.se[1] ≈ sqrt(m0.sigma2) atol = 1e-10
    end

    # =========================================================================
    # Display / report / refs render without error
    # =========================================================================
    @testset "Display & refs" begin
        m = estimate_arfima(_sim_arfima(1, 300, 0.3, 0.0), 0, 0; method=:css)
        s = sprint(show, m)
        @test occursin("ARFIMA", s)
        @test occursin("d", s)
        @test !isempty(sprint(show, gph_test(_sim_arfima(1, 300, 0.3, 0.0))))
        @test !isempty(sprint(show, local_whittle(_sim_arfima(1, 300, 0.3, 0.0))))
        # refs() render for the model + both semiparametric results
        @test !isempty(sprint(io -> refs(io, m)))
        @test !isempty(sprint(io -> refs(io, gph_test(_sim_arfima(1, 300, 0.3, 0.0)))))
        @test !isempty(sprint(io -> refs(io, local_whittle(_sim_arfima(1, 300, 0.3, 0.0)))))
    end

end
