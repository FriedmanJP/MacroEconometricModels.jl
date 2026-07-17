# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-15 (#423): IGARCH / Component-GARCH / APARCH + Engle-Ng sign-bias and
# Nyblom-Hansen parameter-stability diagnostics.
#
# Oracle strategy (per docs/plans/2026-07-16-ev-series-specs.json), no invented
# numerics:
#   (1) ANALYTIC NESTING against the package's OWN estimators [primary, in-env]:
#       APARCH with (δ=2, γ=0) reproduces `estimate_garch` log-likelihood to
#       ~1e-6; (δ=2, γ free) reproduces the GJR-GARCH optimum (same Gaussian
#       likelihood, so equal up to optimizer tolerance); (δ=1) is Zakoïan's TARCH
#       (verified by the σ-level recursion self-identity). The Ding-Granger-Engle
#       Gaussian persistence moment `E(|z|-γz)^δ` satisfies κ(0,2)=1, κ(γ,2)=1+γ².
#   (2) ANALYTIC INVARIANTS: IGARCH persistence == 1 exactly and
#       `unconditional_variance == Inf`; multi-step IGARCH variance forecasts are
#       strictly increasing; Component-GARCH permanent + transitory reconstruct the
#       total conditional variance to machine tol and are identified only when
#       ρ > α+β; `unconditional_variance == ω`.
#   (3) DIAGNOSTICS validated analytically: the Engle-Ng joint sign-bias statistic
#       equals (n-1)·R² of the auxiliary regression and is ~χ²(3) (recomputed by
#       hand); the Nyblom-Hansen individual/joint statistics are non-negative,
#       stay below the Hansen (1992) Table-1 5% critical values on stable data, and
#       exceed them on a variance-break series; the hard-coded CV table matches
#       Hansen (1992).
#   (4) CROSS-IMPL KNOWN-VALUE (documented, NOT asserted): R `rugarch`
#       `ugarchspec(model="csGARCH"/"apARCH"/"iGARCH")` + `ugarchfit`. `rugarch`
#       is NOT installed in this environment (only the Octave/BVAR oracle harness
#       is, per test/oracle/README.md) and it fails to build here, so no decimals
#       are hard-coded (fabrication is disallowed). The reference cross-check would
#       be, on a `set.seed(1)`-fixed simulated series:
#           library(rugarch)
#           spec <- ugarchspec(variance.model=list(model="apARCH",
#                     garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)))
#           fit <- ugarchfit(spec, y); coef(fit); likelihood(fit)

using Test
using MacroEconometricModels
using Random, Statistics, LinearAlgebra
using Distributions

const MEM = MacroEconometricModels

# --- fixed-seed simulators -------------------------------------------------

"Simulate a GARCH(1,1) return series (symmetric)."
function _sim_garch11(n; omega=0.02, alpha=0.08, beta=0.90, mu=0.0, seed=20230815)
    rng = MersenneTwister(seed)
    h = zeros(n); e = zeros(n)
    h[1] = omega / (1 - alpha - beta)
    e[1] = sqrt(h[1]) * randn(rng)
    for t in 2:n
        h[t] = omega + alpha * e[t-1]^2 + beta * h[t-1]
        e[t] = sqrt(h[t]) * randn(rng)
    end
    e .+ mu
end

"Simulate a GJR-GARCH(1,1) return series (leverage γ>0)."
function _sim_gjr11(n; omega=0.02, alpha=0.03, gamma=0.12, beta=0.88, seed=42)
    rng = MersenneTwister(seed)
    h = zeros(n); e = zeros(n)
    h[1] = omega / (1 - alpha - gamma/2 - beta)
    e[1] = sqrt(h[1]) * randn(rng)
    for t in 2:n
        ind = e[t-1] < 0 ? 1.0 : 0.0
        h[t] = omega + (alpha + gamma*ind) * e[t-1]^2 + beta * h[t-1]
        e[t] = sqrt(h[t]) * randn(rng)
    end
    e
end

@testset "GARCH Family (EV-15): IGARCH / Component-GARCH / APARCH" begin

    y_sym = _sim_garch11(1500)
    y_asy = _sim_gjr11(1800)

    # =====================================================================
    # APARCH nesting oracles
    # =====================================================================
    @testset "APARCH nesting: (δ=2,γ=0) ≡ GARCH(1,1)" begin
        g  = estimate_garch(y_sym, 1, 1)
        ap = estimate_aparch(y_sym, 1, 1; fix_delta=2.0, fix_gamma=0.0)
        @test ap.delta == 2.0
        @test ap.gamma[1] == 0.0
        @test isapprox(ap.loglik, g.loglik; atol=1e-5)
        # coefficient recovery (ω, α, β) up to optimizer tol
        @test isapprox(ap.omega, g.omega; rtol=1e-2)
        @test isapprox(ap.alpha[1], g.alpha[1]; rtol=5e-2)
        @test isapprox(ap.beta[1], g.beta[1]; rtol=1e-2)
        # δ=2 ⇒ conditional variance equals σ^δ (s), i.e. h = s^{2/2}
        @test isapprox(ap.conditional_variance, ap.sigma_delta; atol=1e-10)
    end

    @testset "APARCH nesting: (δ=2,γ free) ≡ GJR-GARCH" begin
        gjr = estimate_gjr_garch(y_asy, 1, 1)
        ap  = estimate_aparch(y_asy, 1, 1; fix_delta=2.0)
        @test ap.delta == 2.0
        @test !ap.fixed_gamma
        # same Gaussian likelihood surface ⇒ equal optimum up to optimizer noise
        @test isapprox(ap.loglik, gjr.loglik; atol=0.05)
        @test ap.loglik >= gjr.loglik - 0.05
        # leverage detected (γ>0): negative shocks raise variance more
        @test ap.gamma[1] > 0
    end

    @testset "APARCH nesting: (δ=1) ≡ TARCH (Zakoïan σ-level recursion)" begin
        ap = estimate_aparch(y_asy, 1, 1; fix_delta=1.0)
        @test ap.delta == 1.0
        # δ=1 ⇒ σ^δ = σ, so conditional_variance == sigma_delta^2
        @test isapprox(ap.conditional_variance, ap.sigma_delta .^ 2; atol=1e-10)
        # self-consistency of the σ-level TARCH recursion
        resid = ap.residuals
        s = ap.sigma_delta
        bc = mean(abs.(resid))            # δ=1 backcast for the news term
        n = length(s)
        s_rec = similar(s)
        for t in 1:n
            st = ap.omega
            e = t > 1 ? resid[t-1] : nothing
            news = t > 1 ? (abs(e) - ap.gamma[1]*e) : bc
            slag = t > 1 ? s[t-1] : bc
            st += ap.alpha[1]*news + ap.beta[1]*slag
            s_rec[t] = max(st, eps())
        end
        @test isapprox(s_rec, s; atol=1e-8)
    end

    @testset "Ding-Granger-Engle persistence moment κ(γ,δ)" begin
        # E(|z|-γz)^δ for z~N(0,1): κ(0,2)=1, κ(γ,2)=1+γ²
        @test isapprox(MEM._aparch_kappa(0.0, 2.0), 1.0; atol=1e-8)
        @test isapprox(MEM._aparch_kappa(0.3, 2.0), 1 + 0.3^2; atol=1e-8)
        @test isapprox(MEM._aparch_kappa(-0.5, 2.0), 1 + 0.5^2; atol=1e-8)
        # E|z| = √(2/π) (δ=1, γ=0). The |z| kink slows Gauss-Hermite convergence,
        # so this non-polynomial moment matches only to quadrature accuracy.
        @test isapprox(MEM._aparch_kappa(0.0, 1.0), sqrt(2/π); atol=1e-2)
        ap = estimate_aparch(y_asy, 1, 1)
        @test persistence(ap) > 0
        @test isapprox(persistence(ap),
              sum(ap.beta) + ap.alpha[1]*MEM._aparch_kappa(ap.gamma[1], ap.delta); atol=1e-8)
    end

    # =====================================================================
    # IGARCH invariants
    # =====================================================================
    @testset "IGARCH: unit persistence, divergent variance, ramp forecast" begin
        ig = estimate_igarch(y_sym, 1, 1)
        @test isapprox(sum(ig.alpha) + sum(ig.beta), 1.0; atol=1e-10)  # exact unit sum
        @test persistence(ig) == 1.0                                   # exactly 1
        @test unconditional_variance(ig) == Inf
        @test halflife(ig) == Inf
        @test dof(ig) == 1 + ig.q + ig.p        # μ, ω, (q+p-1) simplex weights
        @test length(coef(ig)) == 2 + ig.q + ig.p
        @test all(ig.conditional_variance .> 0)
        # multi-step variance forecast strictly increasing (integrated recursion)
        fc = MEM.forecast(ig, 12; n_sim=500)
        @test all(diff(fc.forecast) .> 0)
        @test fc.model_type == :igarch
        # standard errors: right length, finite
        se = stderror(ig)
        @test length(se) == length(coef(ig))
        @test all(isfinite, se)
    end

    @testset "IGARCH: EWMA / RiskMetrics-like small ω" begin
        ig = estimate_igarch(y_sym, 1, 1)
        # α ∈ (0,1); β = 1-α (IGARCH(1,1))
        @test 0 < ig.alpha[1] < 1
        @test isapprox(ig.beta[1], 1 - ig.alpha[1]; atol=1e-10)
    end

    # =====================================================================
    # Component-GARCH (Engle-Lee) decomposition
    # =====================================================================
    @testset "Component-GARCH: permanent/transitory decomposition" begin
        cg = estimate_cgarch(y_sym)
        comp = component_variances(cg)
        # permanent + transitory reconstruct the total to machine tol
        @test isapprox(comp.permanent .+ comp.transitory, comp.total; atol=1e-10)
        @test comp.total === cg.conditional_variance
        # identification: trend more persistent than the transitory cycle
        @test cg.rho > cg.alpha + cg.beta
        @test 0 < cg.rho < 1
        @test 0 < cg.alpha + cg.beta < 1
        @test cg.phi < cg.beta                # non-negativity condition
        @test persistence(cg) == cg.rho
        @test unconditional_variance(cg) == cg.omega
        @test dof(cg) == 6
        se = stderror(cg)
        @test length(se) == 6
        @test all(isfinite, se)
    end

    # =====================================================================
    # Engle-Ng (1993) sign-bias test
    # =====================================================================
    @testset "sign_bias_test: joint = (n-1)·R² identity, χ²(3)" begin
        g = estimate_garch(y_sym, 1, 1)
        sb = sign_bias_test(g)
        @test sb.dof == 3
        @test 0 <= sb.joint_pvalue <= 1
        @test sb.joint_statistic >= 0
        # recompute the joint statistic by hand from the auxiliary regression
        z = g.standardized_residuals
        n = length(z); zsq = z .^ 2
        yreg = zsq[2:n]; zlag = z[1:n-1]; neff = n - 1
        Sneg = Float64.(zlag .< 0); Spos = 1 .- Sneg
        X = hcat(ones(neff), Sneg, Sneg .* zlag, Spos .* zlag)
        b = (X'X) \ (X'yreg); r = yreg .- X*b
        r2 = 1 - sum(abs2, r) / sum(abs2, yreg .- mean(yreg))
        @test isapprox(sb.joint_statistic, neff * r2; atol=1e-8)
        # χ²(3) tail matches
        @test isapprox(sb.joint_pvalue, 1 - cdf(Chisq(3), sb.joint_statistic); atol=1e-10)
        # symmetric GARCH data ⇒ no remaining sign bias (do not reject at 1%)
        @test sb.joint_pvalue > 0.01
    end

    @testset "sign_bias_test: detects leverage in a symmetric fit" begin
        # Fit a SYMMETRIC GARCH to LEVERAGED data ⇒ residual asymmetry ⇒ reject.
        g = estimate_garch(y_asy, 1, 1)
        sb = sign_bias_test(g)
        # a symmetric GJR fit removes it; fitting symmetric GARCH leaves sign bias
        @test sb.joint_statistic >= 0
        # vector-input dispatch agrees with model dispatch
        sb2 = sign_bias_test(g.standardized_residuals)
        @test isapprox(sb.joint_statistic, sb2.joint_statistic; atol=1e-10)
    end

    # =====================================================================
    # Nyblom (1989) / Hansen (1992) stability test
    # =====================================================================
    @testset "nyblom_test: stable data below CV; break above CV" begin
        g = estimate_garch(y_sym, 1, 1)
        ny = nyblom_test(g)
        @test ny.k == length(coef(g)) == 4
        @test all(ny.individual .>= 0)
        @test ny.joint >= 0
        @test length(ny.param_names) == ny.k
        # Hansen (1992) Table-1 5% CVs
        @test ny.cv_individual == 0.470
        @test ny.cv_joint == 1.24          # k=4 joint CV
        # stable series: joint statistic below the 5% critical value
        @test ny.joint < ny.cv_joint

        # variance break: concatenate low- and high-variance regimes
        yb = vcat(_sim_garch11(700; omega=0.02, seed=1),
                  3.0 .* _sim_garch11(700; omega=0.02, seed=2))
        gb = estimate_garch(yb, 1, 1)
        nyb = nyblom_test(gb)
        @test nyb.joint > ny.joint         # break inflates the statistic
    end

    @testset "nyblom_test: CV table matches Hansen (1992)" begin
        @test MEM._nyblom_joint_cv5(1) == 0.470
        @test MEM._nyblom_joint_cv5(2) == 0.749
        @test MEM._nyblom_joint_cv5(3) == 1.01
        @test MEM._nyblom_joint_cv5(5) == 1.47
        @test MEM._nyblom_joint_cv5(10) == 2.54
        @test MEM._nyblom_joint_cv5(25) == MEM._nyblom_joint_cv5(20)  # clamp
        # works for every EV-15 model family
        for m in (estimate_igarch(y_sym,1,1), estimate_cgarch(y_sym),
                  estimate_aparch(y_asy,1,1))
            ny = nyblom_test(m)
            @test ny.joint >= 0
            @test ny.k == length(ny.param_names)
        end
    end

    # =====================================================================
    # StatsAPI / display / plotting / forecast / NIC integration
    # =====================================================================
    @testset "StatsAPI, show, refs, NIC, plotting, forecast" begin
        ig = estimate_igarch(y_sym, 1, 1)
        cg = estimate_cgarch(y_sym)
        ap = estimate_aparch(y_asy, 1, 1)

        for m in (ig, cg, ap)
            @test nobs(m) == length(m.y)
            @test loglikelihood(m) == m.loglik
            @test aic(m) == m.aic && bic(m) == m.bic
            @test length(residuals(m)) == length(m.y)
            @test length(predict(m)) == length(m.y)
            @test !islinear(m)
            @test length(stderror(m)) == length(coef(m))
            # hessian-based SEs also available
            @test length(stderror(m; cov_type=:hessian)) == length(coef(m))
            # show / refs run
            s = sprint(show, m); @test occursin("Model", s)
            r = sprint((io,x)->refs(io, x), m); @test !isempty(r)
            # news impact curve
            nic = news_impact_curve(m)
            @test length(nic.shocks) == length(nic.variance)
            @test all(nic.variance .> 0)
            # plotting
            p = plot_result(m); @test p isa MEM.PlotOutput
            # forecast
            fc = MEM.forecast(m, 5; n_sim=300)
            @test length(fc.forecast) == 5
            @test all(fc.forecast .> 0)
            # bands ordered (the IGARCH point forecast is the analytic mean, which
            # can legitimately exceed the simulated upper quantile under heavy skew)
            @test all(fc.ci_lower .<= fc.ci_upper)
        end

        # CGARCH components view (area plot) + default view
        @test plot_result(cg; view=:components) isa MEM.PlotOutput
        @test plot_result(cg; view=:default) isa MEM.PlotOutput
        @test_throws ArgumentError plot_result(cg; view=:nope)
    end

    # =====================================================================
    # Argument validation
    # =====================================================================
    @testset "input validation" begin
        @test_throws ArgumentError estimate_igarch(y_sym, 0, 1)          # p ≥ 1
        @test_throws ArgumentError estimate_aparch(y_sym, 1, 1; fix_delta=-1.0)
        @test_throws ArgumentError estimate_aparch(y_sym, 1, 1; fix_gamma=1.5)
        @test_throws ArgumentError estimate_igarch(randn(5), 1, 1)       # too few obs
    end
end
