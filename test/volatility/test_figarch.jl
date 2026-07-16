# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-14 (#422): FIGARCH / FIEGARCH fractionally-integrated volatility.
#
# Oracle strategy (per docs/plans/2026-07-16-ev-series-specs.json), no invented numerics:
#   (1) ANALYTIC, MACHINE-TOL [primary]: at d=0 the ARCH(∞) λ-weights of a
#       FIGARCH(1,0,1) collapse EXACTLY to the GARCH(1,1) weights λ_i=(φ−β)β^{i−1}
#       (so α=φ−β); the FIEGARCH MA(∞) ψ-weights collapse to ψ_k=(β−φ)β^{k−1},
#       ψ_0=1. Reconstructed to ~1e-12, validating the whole weight construction
#       (which reuses EV-13's `_frac_diff_weights`).
#   (2) NESTED LOGLIK: on a GARCH(1,1)-simulated series, the truncated FIGARCH
#       log-likelihood at (ω,β,φ=α+β,d→0) equals the GARCH(1,1) log-likelihood to
#       within the pre-sample backcast / truncation edge effect (< 0.1 abs on
#       T=3000, i.e. relative ~4e-6).
#   (3) PARAMETER RECOVERY: a fixed-seed T=10000 FIGARCH(1,d,1) path recovers
#       (ω, d, φ, β) within a loose stated tolerance.
#   (4) POSITIVITY: a parameterization violating the BBM non-negativity conditions
#       yields a warning + recorded negative-λ count (no throw).
#   (5) CROSS-IMPL KNOWN-VALUE (documented, NOT asserted): R `rugarch` `fiGARCH`
#       spec. `rugarch` is NOT installed in this environment (only the Octave/BVAR
#       oracle harness is, per test/oracle/README.md), and the classic BBM (1996)
#       DM/£ estimates depend on their proprietary dataset, so no decimals are
#       hard-coded (fabrication is disallowed). The reference cross-check would be
#       (Ghalanos, rugarch ≥ 1.4, on the built-in `dmbp[,1]` series):
#           library(rugarch)
#           spec <- ugarchspec(
#             variance.model = list(model="fiGARCH", garchOrder=c(1,1),
#                                   submodel="FIGARCH"),
#             mean.model = list(armaOrder=c(0,0), include.mean=TRUE),
#             distribution.model = "norm")
#           fit <- ugarchfit(spec, data = dmbp[,1])
#           # coef(fit): mu, omega, alpha1(=φ), beta1(=β), delta(=d)  [rugarch names]
#       reproducing σ²_t = ω/(1−β) + [1 − (1−βL)⁻¹(1−φL)(1−L)^d] ε²_t with the
#       sample-variance backcast. Oracles (1)–(4) pin the estimator to ground truth
#       instead.

using Test
using MacroEconometricModels
using Random
using Statistics
using LinearAlgebra
using StatsAPI

const _M = MacroEconometricModels

# --- fixed-seed simulators ---------------------------------------------------

"""Simulate a FIGARCH(1,d,1) path from the truncated ARCH(∞) recursion."""
function _sim_figarch(T::Int; omega=0.05, d=0.4, phi=0.2, beta=0.5,
                      K=1500, burn=3000, seed=1)
    Random.seed!(seed)
    lam = _M._figarch_lambda(float(d), [float(phi)], [float(beta)], K)
    ostar = omega / (1 - beta)
    n = T + burn
    e2 = fill(ostar, n)
    r = zeros(n)
    @inbounds for t in 1:n
        h = ostar
        for i in 1:min(K, t - 1)
            h += lam[i] * e2[t-i]
        end
        h = max(h, 1e-12)
        r[t] = sqrt(h) * randn()
        e2[t] = r[t]^2
    end
    r[burn+1:end]
end

"""Simulate a GARCH(1,1) path."""
function _sim_garch11(T::Int; omega=0.05, alpha=0.08, beta=0.9, burn=1500, seed=1)
    Random.seed!(seed)
    n = T + burn
    h = omega / (1 - alpha - beta)
    r = zeros(n)
    ep = 0.0
    @inbounds for t in 1:n
        h = omega + alpha * ep^2 + beta * h
        r[t] = sqrt(h) * randn()
        ep = r[t]
    end
    r[burn+1:end]
end

"""Simulate a FIEGARCH path from the truncated log-variance MA(∞)."""
function _sim_fiegarch(T::Int; omega=-0.1, theta=-0.08, gamma=0.15, d=0.35,
                       phi=0.2, beta=0.4, K=1000, burn=2000, seed=1)
    Random.seed!(seed)
    psi = _M._fiegarch_psi(float(d), [float(phi)], [float(beta)], K)
    Eabsz = sqrt(2 / pi)
    n = T + burn
    r = zeros(n)
    gz = zeros(n)
    @inbounds for t in 1:n
        lg = omega
        for j in 0:K
            idx = t - 1 - j
            idx >= 1 && (lg += psi[j+1] * gz[idx])
        end
        lg = clamp(lg, -50.0, 50.0)
        h = exp(lg)
        z = randn()
        r[t] = sqrt(h) * z
        gz[t] = theta * z + gamma * (abs(z) - Eabsz)
    end
    r[burn+1:end]
end

@testset "FIGARCH / FIEGARCH (EV-14, #422)" begin

    # =========================================================================
    # Oracle 1 — analytic machine-tol weight identities (d = 0 nesting)
    # =========================================================================
    @testset "λ / ψ weight identities at d=0" begin
        for (phi, beta) in ((0.35, 0.6), (0.25, 0.5), (0.4, 0.55))
            K = 60
            lam = _M._figarch_lambda(0.0, [phi], [beta], K)
            # FIGARCH(1,0,1) ≡ GARCH(1,1) with α = φ − β:  λ_i = (φ−β)β^{i−1}
            target = [(phi - beta) * beta^(i - 1) for i in 1:K]
            @test maximum(abs.(lam .- target)) < 1e-11
            @test lam[1] ≈ phi - beta atol=1e-12

            psi = _M._fiegarch_psi(0.0, [phi], [beta], K)
            # FIEGARCH at d=0: ψ_0=1, ψ_k=(β−φ)β^{k−1}
            @test psi[1] ≈ 1.0 atol=1e-12
            target_psi = [(beta - phi) * beta^(k - 1) for k in 1:K]
            @test maximum(abs.(psi[2:end] .- target_psi)) < 1e-11
        end

        # weights genuinely reuse EV-13's _frac_diff_weights (hyperbolic memory):
        # at d>0 the λ-weights decay slowly and stay positive for a valid spec.
        lam = _M._figarch_lambda(0.4, [0.2], [0.5], 200)
        @test all(lam .> 0)
        @test lam[1] > lam[50] > lam[150] > 0            # monotone hyperbolic decay
        # δ_1 of (1−L)^d is −d — sign carried into the construction
        δ = _M._frac_diff_weights(0.4, 5)
        @test δ[2] ≈ -0.4 atol=1e-12
    end

    # =========================================================================
    # Oracle 2 — nested log-likelihood: FIGARCH(d→0) == GARCH(1,1)
    # =========================================================================
    @testset "d→0 loglik nests GARCH(1,1)" begin
        r = _sim_garch11(3000; seed=3)
        gm = estimate_garch(r, 1, 1)
        α, β, ω, μ = gm.alpha[1], gm.beta[1], gm.omega, gm.mu
        K = length(r) - 1
        logit(v) = log(v / (1 - v))
        for dsmall in (1e-8, 1e-5)
            params = [μ, log(ω), logit(α + β), logit(β), logit(dsmall)]
            ll_fig = -_M._figarch_negloglik(params, r, 1, 1, K)
            # difference is only the pre-sample backcast / truncation edge effect
            @test abs(ll_fig - gm.loglik) < 0.1
        end
    end

    # =========================================================================
    # Oracle 3 — parameter recovery on a long simulated FIGARCH path
    # =========================================================================
    @testset "FIGARCH parameter recovery (T=10000)" begin
        for seed in (1, 7, 42)
            r = _sim_figarch(10000; omega=0.05, d=0.4, phi=0.2, beta=0.5,
                             K=1500, seed=seed)
            m = estimate_figarch(r; truncation=1500)
            @test m.converged
            @test 0.0 < m.d < 1.0
            @test isapprox(m.d, 0.4; atol=0.10)
            @test isapprox(m.phi[1], 0.2; atol=0.10)
            @test isapprox(m.beta[1], 0.5; atol=0.12)
            @test isapprox(m.omega, 0.05; atol=0.05)
        end
    end

    # =========================================================================
    # Oracle 4 — positivity (BBM non-negativity): warn + count, never throw
    # =========================================================================
    @testset "positivity warning + count (no throw)" begin
        # φ=0.9, β=0.1, small d ⇒ many negative λ-weights (BBM violation)
        bad = _M._figarch_lambda(0.2, [0.9], [0.1], 100)
        @test count(<(-1e-8), bad) > 0
        # helper warns and returns the count, does not throw
        n_neg = @test_logs (:warn,) _M._figarch_nonneg_count(bad; warn=true)
        @test n_neg == count(<(-sqrt(eps(Float64))), bad)
        # silent mode: no warning, still counts
        @test _M._figarch_nonneg_count(bad; warn=false) == n_neg
        # a valid spec triggers no warning and zero count
        good = _M._figarch_lambda(0.4, [0.2], [0.5], 100)
        @test (@test_logs _M._figarch_nonneg_count(good; warn=true)) == 0
    end

    # =========================================================================
    # API / display / StatsAPI / forecast / NIC — FIGARCH
    # =========================================================================
    @testset "FIGARCH API surface" begin
        r = _sim_figarch(2000; seed=11, K=1000)
        m = estimate_figarch(r; truncation=800)
        @test m isa FIGARCHModel{Float64}
        @test m isa MacroEconometricModels.AbstractVolatilityModel
        @test m.truncation == 800
        @test length(m.lambda) == 800
        @test length(m.conditional_variance) == length(r)
        @test all(m.conditional_variance .> 0)

        # StatsAPI
        @test length(coef(m)) == 5                       # μ, ω, φ, β, d
        @test coef(m)[end] == m.d
        @test StatsAPI.dof(m) == 5
        @test nobs(m) == length(r)
        @test loglikelihood(m) == m.loglik
        @test isfinite(aic(m)) && isfinite(bic(m))
        @test predict(m) === m.conditional_variance
        @test residuals(m) == m.residuals

        # standard errors (BW sandwich + delta method), all finite & positive
        se = stderror(m)
        @test length(se) == 5
        @test all(isfinite, se) && all(se .>= 0)
        se_h = stderror(m; cov_type=:hessian)
        @test all(isfinite, se_h)
        @test_throws ArgumentError stderror(m; cov_type=:bogus)
        @test _M.d_stderror(m) == se[end]

        # persistence / frac_order accessors
        @test _M.persistence(m) == m.d
        @test _M.frac_order(m) == m.d

        # forecast reuses the shared VolatilityForecast infrastructure
        fc = forecast(m, 8)
        @test fc isa MacroEconometricModels.VolatilityForecast
        @test length(fc.forecast) == 8
        @test all(fc.forecast .> 0)
        @test length(predict(m, 8)) == 8                 # StatsAPI predict(h) smoke
        @test_throws ArgumentError forecast(m, 0)

        # news impact curve (symmetric parabola)
        nic = news_impact_curve(m)
        @test length(nic.shocks) == length(nic.variance) == 200
        @test all(nic.variance .> 0)
        # symmetry: NIC(+e) == NIC(−e)
        @test nic.variance[1] ≈ nic.variance[end] atol=1e-8

        # display + refs render without error
        io = IOBuffer(); show(io, m); s = String(take!(io))
        @test occursin("FIGARCH", s)
        @test occursin("d (frac. int.)", s)
        rb = refs(m)
        @test occursin("Baillie", rb)

        # invalid dist rejected
        @test_throws ArgumentError estimate_figarch(r; dist=:t)
    end

    # =========================================================================
    # FIEGARCH — runs, identifies leverage sign, forecast positive
    # (d, φ, β weakly identified in the log-variance form ⇒ lenient recovery)
    # =========================================================================
    @testset "FIEGARCH API + fit" begin
        r = _sim_fiegarch(4000; theta=-0.08, gamma=0.15, d=0.35, seed=5, K=1000)
        m = estimate_fiegarch(r; truncation=1000)
        @test m isa FIEGARCHModel{Float64}
        @test m.converged
        @test 0.0 < m.d < 1.0
        @test isfinite(m.loglik)
        @test all(m.conditional_variance .> 0)
        @test length(m.psi) == 1001                      # ψ_0 .. ψ_K

        # StatsAPI
        @test length(coef(m)) == 7                       # μ, ω, θ, γ, φ, β, d
        @test StatsAPI.dof(m) == 7
        se = stderror(m)
        @test length(se) == 7
        @test all(isfinite, se)

        # forecast + NIC
        fc = forecast(m, 6)
        @test all(fc.forecast .> 0)
        nic = news_impact_curve(m)
        @test length(nic.variance) == 200
        @test all(nic.variance .> 0)
        # asymmetry from θ≠0: the curve is generally not symmetric in the shock sign
        @test nic.variance[1] != nic.variance[end]

        # display + refs
        io = IOBuffer(); show(io, m); s = String(take!(io))
        @test occursin("FIEGARCH", s)
        rb = refs(m)
        @test occursin("Bollerslev", rb)

        @test_throws ArgumentError estimate_fiegarch(r; dist=:t)
    end

    # =========================================================================
    # Input validation
    # =========================================================================
    @testset "input validation" begin
        r = _sim_figarch(500; seed=2, K=400)
        @test_throws ArgumentError estimate_figarch(randn(5))          # too few obs
        @test_throws ArgumentError estimate_figarch(r; truncation=0)
        @test_throws ArgumentError estimate_figarch(r; d0=1.5)
        @test_throws ArgumentError estimate_fiegarch(r; d0=0.0)
        # Integer input auto-converts to Float64
        m = estimate_figarch(round.(Int, 100 .* r) ./ 100; truncation=300)
        @test m isa FIGARCHModel{Float64}
    end
end
