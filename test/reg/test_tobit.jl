# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-17 (#425): Tobit (censored) and truncated-normal regression.

using Test
using MacroEconometricModels
using MacroEconometricModels: _mills, _lambda
using LinearAlgebra, Statistics, Distributions
import StatsAPI
import ForwardDiff

# =============================================================================
# Deterministic DGP shared with the R oracle (no RNG => byte-identical in R and
# Julia). Left-censored at 0; a two-sided [0, 2] version; and the uncensored
# subsample used for the truncated-regression fit.
#
# Julia:
#   x1 = sin(0.3i+0.7); x2 = 0.5cos(0.2i)+0.3sin(0.5i);
#   e  = 0.9sin(1.3i+0.5)+0.45cos(2.1i+0.2); ystar = 0.4 + x1 - 0.7 x2 + e
#   y  = max(ystar,0);  y2 = clamp(ystar,0,2)
# n=200 → 63 left-censored, 137 uncensored, and (two-sided) 63 left + 17 right.
# =============================================================================
function _tobit_dgp()
    n = 200
    x1 = [sin(0.3i + 0.7) for i in 1:n]
    x2 = [0.5cos(0.2i) + 0.3sin(0.5i) for i in 1:n]
    e  = [0.9sin(1.3i + 0.5) + 0.45cos(2.1i + 0.2) for i in 1:n]
    ystar = [0.4 + 1.0 * x1[i] - 0.7 * x2[i] + e[i] for i in 1:n]
    y  = max.(ystar, 0.0)
    y2 = clamp.(ystar, 0.0, 2.0)
    X = hcat(ones(n), x1, x2)
    (; n, X, ystar, y, y2)
end

# Plain censored-normal Tobit log-likelihood in (β, σ) for the score-at-MLE check.
function _tobit_ll_ref(beta, sigma, y, X, L, U)
    N = Normal(zero(eltype(beta)), one(eltype(beta)))
    ll = zero(eltype(beta))
    for i in 1:length(y)
        xb = dot(@view(X[i, :]), beta)
        if isfinite(L) && y[i] <= L
            ll += logcdf(N, (L - xb) / sigma)
        elseif isfinite(U) && y[i] >= U
            ll += logccdf(N, (U - xb) / sigma)
        else
            ll += logpdf(N, (y[i] - xb) / sigma) - log(sigma)
        end
    end
    ll
end

@testset "EV-17 Tobit + truncated regression (#425)" begin
    d = _tobit_dgp()
    X = d.X

    # -------------------------------------------------------------------------
    @testset "shared helpers: _mills / _lambda" begin
        # Inverse Mills ratio at 0: φ(0)/Φ(0) = (1/√(2π)) / 0.5.
        @test _mills(0.0) ≈ (1 / sqrt(2π)) / 0.5 atol = 1e-12
        # Identity vs Distributions.
        for a in (-3.0, -1.0, 0.5, 2.0)
            N = Normal()
            @test _mills(a) ≈ pdf(N, a) / cdf(N, a) atol = 1e-12
        end
        # Deep left tail stays finite (no 0/0 blow-up).
        @test isfinite(_mills(-30.0)) && _mills(-30.0) > 0
        # Two-sided hazard collapses to one-sided limits.
        @test _lambda(0.5, Inf) ≈ _mills(-0.5) atol = 1e-9      # b=∞ ⇒ φ(a)/Φ(-a)
        @test _lambda(-Inf, 1.0) ≈ -_mills(1.0) atol = 1e-9     # a=-∞ ⇒ -φ(b)/Φ(b)
        # E[z | a<z<b] for standard normal, checked by numeric quadrature.
        a, b = -0.7, 1.3
        N = Normal()
        zs = range(a, b; length = 200_001); dz = step(zs)
        num = sum(z * pdf(N, z) for z in zs) * dz
        den = sum(pdf(N, z) for z in zs) * dz
        @test _lambda(a, b) ≈ num / den atol = 1e-4
    end

    # -------------------------------------------------------------------------
    @testset "Tobit left-censored — R AER::tobit / survreg oracle" begin
        # R (survival 4.5.0, the AER::tobit backend):
        #   d <- data.frame(x1=..., x2=..., y=...)   # same deterministic DGP
        #   fit <- survreg(Surv(y, y > 0, type="left") ~ x1 + x2, data=d, dist="gaussian")
        #   coef(fit)   # (Intercept) x1 x2
        #   fit$scale   # sigma
        #   logLik(fit)
        #   sqrt(diag(vcov(fit)))  # se(Intercept,x1,x2, Log(scale))
        R_beta  = [0.450813234144, 0.929294250864, -0.641802139701]
        R_sigma = 0.681848872058
        R_ll    = -181.080929285
        R_se    = [0.0545417840433, 0.0781789723515, 0.1200959385490]  # β block
        R_se_logsigma = 0.0626283375003

        m = estimate_tobit(d.y, X; lower = 0.0, varnames = ["const", "x1", "x2"])
        @test m.converged
        @test m.n_censored_left == 63
        @test m.n_censored_right == 0
        @test m.beta ≈ R_beta atol = 1e-5
        @test m.sigma ≈ R_sigma atol = 1e-5
        @test m.loglik ≈ R_ll atol = 1e-3
        @test StatsAPI.stderror(m) ≈ R_se atol = 1e-5
        # survreg reports SE on Log(scale); compare on that scale.
        @test m.sigma_se / m.sigma ≈ R_se_logsigma atol = 1e-4

        # Score (gradient of the log-likelihood) is ~0 at the MLE.
        gβ = ForwardDiff.gradient(b -> _tobit_ll_ref(b, m.sigma, d.y, X, 0.0, Inf), m.beta)
        gσ = ForwardDiff.derivative(s -> _tobit_ll_ref(m.beta, s, d.y, X, 0.0, Inf), m.sigma)
        @test maximum(abs, gβ) < 1e-4
        @test abs(gσ) < 1e-4
    end

    # -------------------------------------------------------------------------
    @testset "Tobit two-sided censoring [0, 2] — survreg interval2 oracle" begin
        # R: low/up per obs, fit <- survreg(Surv(low, up, type="interval2") ~ x1 + x2, dist="gaussian")
        R_beta  = [0.450344325510, 0.952721759508, -0.664989841037]
        R_sigma = 0.711485035511
        R_ll    = -185.008081723

        m = estimate_tobit(d.y2, X; lower = 0.0, upper = 2.0, varnames = ["const", "x1", "x2"])
        @test m.converged
        @test m.n_censored_left == 63
        @test m.n_censored_right == 17
        @test m.beta ≈ R_beta atol = 1e-5
        @test m.sigma ≈ R_sigma atol = 1e-5
        @test m.loglik ≈ R_ll atol = 1e-3

        # Score = 0 at the MLE (two-sided likelihood).
        gβ = ForwardDiff.gradient(b -> _tobit_ll_ref(b, m.sigma, d.y2, X, 0.0, 2.0), m.beta)
        @test maximum(abs, gβ) < 1e-4
    end

    # -------------------------------------------------------------------------
    @testset "Degenerate: no censoring ⇒ Tobit ≡ OLS" begin
        yun = d.ystar .+ 5.0                 # all > 0 ⇒ nothing censored at lower=0
        m = estimate_tobit(yun, X; lower = 0.0, varnames = ["const", "x1", "x2"])
        @test m.n_censored_left == 0 && m.n_censored_right == 0
        ols = X \ yun
        @test m.beta ≈ ols atol = 1e-6
        # Tobit σ (MLE, ÷n) matches the OLS residual σ (÷n).
        resid = yun .- X * ols
        @test m.sigma ≈ sqrt(sum(abs2, resid) / length(yun)) atol = 1e-6
        # And the log-likelihood equals the plain Gaussian log-likelihood.
        n = length(yun)
        gauss_ll = -n / 2 * (log(2π) + 1 + 2 * log(m.sigma))
        @test m.loglik ≈ gauss_ll atol = 1e-4
    end

    # -------------------------------------------------------------------------
    @testset "Truncated regression — truncreg::truncreg oracle" begin
        # R (truncreg 0.2-5):
        #   du <- d[d$y > 0, ]
        #   ft <- truncreg(y ~ x1 + x2, data=du, point=0, direction="left")
        #   coef(ft)  # (Intercept) x1 x2 sigma ;  logLik(ft) ; sqrt(diag(vcov(ft)))
        R_beta  = [0.266151738962, 1.046272716992, -0.788379025447]
        R_sigma = 0.755079975030
        R_ll    = -95.8720598769
        R_se    = [0.187871251031, 0.194484978868, 0.191018208448]
        R_sigma_se = 0.078672673666

        keep = d.ystar .> 0
        yt = d.ystar[keep]
        Xt = X[keep, :]
        m = estimate_truncreg(yt, Xt; lower = 0.0, varnames = ["const", "x1", "x2"])
        @test m.converged
        @test StatsAPI.nobs(m) == count(keep)
        # atol 2e-4 on β: our optimizer attains a marginally HIGHER log-likelihood than R's
        # truncreg (which stops a touch early); see the score check below — ours is the true MLE.
        @test m.beta ≈ R_beta atol = 2e-4
        @test m.sigma ≈ R_sigma atol = 2e-4
        @test m.loglik ≈ R_ll atol = 1e-3
        @test m.loglik >= R_ll                       # our fit is at least as good as R's
        @test StatsAPI.stderror(m) ≈ R_se atol = 1e-3
        @test m.sigma_se ≈ R_sigma_se atol = 1e-3

        # Score = 0 at our MLE (truncated-normal log-likelihood), to machine precision.
        function trunc_ll(beta, sigma)
            N = Normal(); s = zero(eltype(beta))
            for i in 1:length(yt)
                xb = dot(@view(Xt[i, :]), beta)
                s += logpdf(N, (yt[i] - xb) / sigma) - log(sigma) - logccdf(N, (0.0 - xb) / sigma)
            end
            s
        end
        gβ = ForwardDiff.gradient(b -> trunc_ll(b, m.sigma), m.beta)
        gσ = ForwardDiff.derivative(s -> trunc_ll(m.beta, s), m.sigma)
        @test maximum(abs, gβ) < 1e-5
        @test abs(gσ) < 1e-5
    end

    # -------------------------------------------------------------------------
    @testset "McDonald–Moffitt marginal effects" begin
        m = estimate_tobit(d.y, X; lower = 0.0, varnames = ["const", "x1", "x2"])
        σ = m.sigma
        z = (X * m.beta) ./ σ                       # left-censoring at 0 ⇒ z = x'β/σ
        N = Normal()

        me_u = marginal_effects(m; which = :unconditional)
        me_p = marginal_effects(m; which = :probability)
        me_c = marginal_effects(m; which = :conditional)

        # Intercept row carries no marginal effect.
        @test isnan(me_u.effects[1]) && isnan(me_p.effects[1]) && isnan(me_c.effects[1])

        # Closed forms (Greene 19.3.3) for left-censoring at 0, averaged over the sample:
        #   unconditional: β_j · mean Φ(z_i);  probability: β_j · mean φ(z_i)/σ
        for j in 2:3
            @test me_u.effects[j] ≈ m.beta[j] * mean(cdf.(N, z)) atol = 1e-8
            @test me_p.effects[j] ≈ m.beta[j] * mean(pdf.(N, z)) / σ atol = 1e-8
            # Conditional (McDonald–Moffitt): β_j · mean{1 − λ(z)[z + λ(z)]}, λ=φ/Φ.
            λ = pdf.(N, z) ./ cdf.(N, z)
            @test me_c.effects[j] ≈ m.beta[j] * mean(1 .- λ .* (z .+ λ)) atol = 1e-8
        end

        # SEs positive and finite; all three effect types available with SEs.
        for me in (me_u, me_p, me_c)
            @test all(isfinite, me.se[2:3])
            @test all(>(0), me.se[2:3])
        end
    end

    # -------------------------------------------------------------------------
    @testset "Upper-only + logistic-dist paths run" begin
        # Upper-only censoring (lower = -Inf).
        yhi = min.(d.ystar, 1.0)
        m_hi = estimate_tobit(yhi, X; lower = -Inf, upper = 1.0, varnames = ["const", "x1", "x2"])
        @test m_hi.converged
        @test m_hi.n_censored_left == 0
        @test m_hi.n_censored_right == count(>=(1.0), yhi)

        # Logistic Tobit (direct (β, logσ) optimization, no Olsen) runs and is finite.
        m_lo = estimate_tobit(d.y, X; lower = 0.0, dist = :logistic, varnames = ["const", "x1", "x2"])
        @test m_lo.converged
        @test all(isfinite, m_lo.beta) && isfinite(m_lo.sigma)
        @test all(isfinite, StatsAPI.stderror(m_lo))
    end

    # -------------------------------------------------------------------------
    @testset "Input validation & errors" begin
        @test_throws ArgumentError estimate_tobit(d.y, X; lower = 0.0, upper = 0.0)  # lower==upper
        @test_throws ArgumentError estimate_tobit(d.y, X; dist = :cauchy)
        @test_throws ArgumentError estimate_tobit(d.y, X; varnames = ["a", "b"])     # wrong length
        # truncated regression rejects points outside the truncation region.
        @test_throws ArgumentError estimate_truncreg(d.y, X; lower = 0.0)            # y contains 0s
        m = estimate_tobit(d.y, X; lower = 0.0)
        @test_throws ArgumentError marginal_effects(m; which = :bogus)
    end

    # -------------------------------------------------------------------------
    @testset "Display & StatsAPI interface" begin
        m = estimate_tobit(d.y, X; lower = 0.0, varnames = ["const", "x1", "x2"])
        keep = d.ystar .> 0
        mt = estimate_truncreg(d.ystar[keep], X[keep, :]; lower = 0.0, varnames = ["const", "x1", "x2"])

        for mod in (m, mt)
            io = IOBuffer(); show(io, mod); s = String(take!(io))
            @test occursin("sigma", s)
            @test occursin("Coefficients", s)
            io2 = IOBuffer(); refs(io2, mod); @test !isempty(String(take!(io2)))
            @test length(StatsAPI.coef(mod)) == 3
            @test size(StatsAPI.vcov(mod)) == (3, 3)
            @test StatsAPI.dof(mod) == 4         # 3 β + σ
            ci = StatsAPI.confint(mod)
            @test size(ci) == (3, 2)
            @test all(ci[:, 1] .< ci[:, 2])
        end
        # Tobit header shows censoring counts.
        io = IOBuffer(); show(io, m); s = String(take!(io))
        @test occursin("Left-censored", s)
    end
end
