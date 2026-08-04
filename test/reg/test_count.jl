# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-19 (#427): count-data regression — Poisson (QML) and Negative Binomial 2.

using Test
using MacroEconometricModels
using MacroEconometricModels: _poisson_loglik, _poisson_deviance, _nb2_loglik,
    _ct_moment_alpha, _irls_poisson
const SA = MacroEconometricModels.StatsAPI
using LinearAlgebra, Statistics, Random, Distributions, DelimitedFiles

# =============================================================================
# Fixed oracle data, generated once in R 4.5.0 and committed verbatim as
# data/count_oracle.csv (columns: y_pois, y_nb, y_off, x1, x2, expo; n = 400):
#
#   set.seed(20260731)
#   x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4); expo <- runif(n, 0.5, 2.0)
#   mu <- exp(0.5 + 0.6*x1 - 0.4*x2)
#   y_pois <- rpois(n, mu)
#   y_nb   <- MASS::rnegbin(n, mu = mu, theta = 2.0)     # alpha = 1/theta = 0.5
#   y_off  <- rpois(n, expo * mu)
#
# Committing the data (rather than the seed) is deliberate: neither R's nor
# Julia's RNG stream is stable across versions, so every reference number below
# is reproducible on any Julia and any R.
#
# Reference values come from `glm(family = poisson)` and `MASS::glm.nb`, with the
# exact call quoted above each block.
# =============================================================================
function _count_oracle_data()
    d = readdlm(joinpath(@__DIR__, "data", "count_oracle.csv"), ',', Float64)
    n = size(d, 1)
    (y_pois=d[:, 1], y_nb=d[:, 2], y_off=d[:, 3],
     x1=d[:, 4], x2=d[:, 5], expo=d[:, 6],
     X=hcat(ones(n), d[:, 4], d[:, 5]), n=n)
end

@testset "Count-data models (EV-19)" begin

    D = _count_oracle_data()
    VN = ["const", "x1", "x2"]

    # =========================================================================
    # 1. R cross-implementation oracle
    # =========================================================================
    @testset "R oracle — Poisson (glm family=poisson, R 4.5.0)" begin
        # glm(y_pois ~ x1 + x2, family = poisson)
        m = estimate_poisson(D.y_pois, D.X; varnames=VN, cov_type=:mle)

        @test isapprox(coef(m), [0.4857077197, 0.5985126201, -0.3426680646]; rtol=1e-8)
        @test isapprox(loglikelihood(m), -604.3424512009; rtol=1e-10)
        @test isapprox(m.deviance, 453.6955286285; rtol=1e-9)
        @test isapprox(m.null_deviance, 696.4032366243; rtol=1e-9)
        @test isapprox(m.aic, 1214.6849024019; rtol=1e-10)
        @test m.converged

        # summary(glm(...))$coefficients[,2]. R reuses the last IRLS QR, so its
        # information is evaluated one step behind the converged beta; we evaluate
        # it AT beta-hat, which is what the estimate's information matrix actually
        # is. The residual gap is that staleness, not a disagreement about the
        # formula: measured 1.7e-6 relative here and 1.7e-5 on the overdispersed
        # series below, so 1e-4 covers it with room for platform arithmetic.
        @test isapprox(stderror(m), [0.0504167939, 0.0405518195, 0.0838311671]; rtol=1e-4)

        # The QMLE sandwich, hand-computed in R at fitted(mp) so that both sides
        # evaluate the same quantity:
        #   X <- model.matrix(mp); mu <- fitted(mp)
        #   A <- t(X)%*%(X*mu); B <- t(X)%*%(X*(y-mu)^2); Ai <- solve(A)
        #   sqrt(diag(Ai %*% B %*% Ai))
        mr = estimate_poisson(D.y_pois, D.X; varnames=VN)      # :robust is the default
        @test mr.cov_type == :robust
        @test isapprox(stderror(mr), [0.0494848497, 0.0400881259, 0.0860413529]; rtol=1e-8)

        # Same meat, HC1 finite-sample scaling n/(n-k).
        mh = estimate_poisson(D.y_pois, D.X; cov_type=:hc1)
        @test isapprox(stderror(mh), [0.0496714683, 0.0402393073, 0.0863658343]; rtol=1e-8)
    end

    @testset "R oracle — Poisson with exposure offset" begin
        # glm(y_off ~ x1 + x2 + offset(log(expo)), family = poisson)
        m = estimate_poisson(D.y_off, D.X; exposure=D.expo, varnames=VN, cov_type=:mle)
        @test isapprox(coef(m), [0.5870555995, 0.5275610523, -0.3811976246]; rtol=1e-8)
        @test isapprox(loglikelihood(m), -659.6666001919; rtol=1e-10)
        @test isapprox(stderror(m), [0.0433567565, 0.0351500392, 0.0740877069]; rtol=1e-4)

        # exposure = e is exactly offset = log(e)
        m2 = estimate_poisson(D.y_off, D.X; offset=log.(D.expo), cov_type=:mle)
        @test coef(m2) ≈ coef(m) atol = 1e-12
        @test m.offset !== nothing && m2.offset !== nothing

        # ... and dropping it changes the fit, so the offset is not being ignored.
        m3 = estimate_poisson(D.y_off, D.X; cov_type=:mle)
        @test !isapprox(coef(m3), coef(m); rtol=1e-3)
    end

    @testset "R oracle — NegBin2 (MASS::glm.nb, R 4.5.0)" begin
        # MASS::glm.nb(y_nb ~ x1 + x2)
        m = estimate_nbreg(D.y_nb, D.X; varnames=VN)

        @test isapprox(coef(m), [0.6067325634, 0.6953706931, -0.4575861023]; rtol=1e-7)
        # R reports theta = 1/alpha.
        @test isapprox(m.alpha, 1 / 1.9301234233; rtol=1e-7)
        @test isapprox(1 / m.alpha, 1.9301234233; rtol=1e-7)
        @test isapprox(loglikelihood(m), -683.1739777513; rtol=1e-9)
        @test isapprox(m.aic, 1374.3479555026; rtol=1e-9)
        @test isapprox(m.deviance, 432.402429181802; rtol=1e-8)
        @test isapprox(m.null_deviance, 616.277536148403; rtol=1e-8)
        @test m.converged

        # Standard errors follow a DIFFERENT convention from MASS, deliberately.
        # `glm.nb` reports the IRLS errors at theta held fixed; we report the
        # (beta,beta) block of the inverse JOINT (beta, log alpha) Hessian, which
        # charges for alpha having been estimated (Stata's `nbreg` convention).
        # The expected information is block diagonal (Lawless 1987), so the two
        # agree asymptotically; here they differ by ~0.25%.
        se_joint = stderror(m)
        se_mass = [0.0673092742, 0.0581454590, 0.1157962196]
        @test isapprox(se_joint, se_mass; rtol=5e-3)

        # Proof that the gap is the convention and not an error: rebuilding the
        # fixed-alpha information reproduces MASS to 8 digits.
        w = m.fitted ./ (1 .+ m.alpha .* m.fitted)
        se_fixed = sqrt.(diag(inv(D.X' * Diagonal(w) * D.X)))
        @test isapprox(se_fixed, se_mass; rtol=1e-6)

        # alpha's own standard error, against R's SE.theta pushed through the
        # delta method SE(alpha) = SE(theta)/theta^2.
        @test isapprox(m.alpha_se, 0.3272890575 / 1.9301234233^2; rtol=1e-3)
        @test m.alpha_se > 0
    end

    @testset "R oracle — Cameron-Trivedi (1990) dispersion test" begin
        # Poisson on the overdispersed series, then
        #   z <- ((y - mu)^2 - y)/mu
        #   lm(z ~ mu - 1)   # NB2 form
        #   lm(z ~ 1)        # NB1 form
        m = estimate_poisson(D.y_nb, D.X; cov_type=:mle)
        @test isapprox(coef(m), [0.6073515832, 0.6972867334, -0.4599179412]; rtol=1e-7)
        @test isapprox(loglikelihood(m), -738.4963897233; rtol=1e-9)

        dt = dispersion_test(m)
        @test isapprox(dt.nb2.alpha, 0.3947902689; rtol=1e-8)
        @test isapprox(dt.nb2.t_stat, 4.9326118377; rtol=1e-8)
        @test isapprox(dt.nb1.alpha, 0.9779036795; rtol=1e-8)
        @test isapprox(dt.nb1.t_stat, 4.9854874016; rtol=1e-8)
        @test dt.n == D.n
        # Overdispersion is detected, and in the right direction.
        @test dt.nb2.p_value < 1e-5
        @test dt.nb2.alpha > 0

        # On the genuinely equidispersed series the test does not fire.
        dt0 = dispersion_test(estimate_poisson(D.y_pois, D.X))
        @test dt0.nb2.p_value > 0.05
        @test abs(dt0.nb2.alpha) < 0.1
    end

    @testset "R oracle — marginal effects and incidence-rate ratios" begin
        m = estimate_poisson(D.y_pois, D.X; varnames=VN, cov_type=:mle)
        me = marginal_effects(m)

        # Continuous x1: mean(mu) * beta_1
        @test isapprox(me.effects[2], 1.009990046436; rtol=1e-9)
        # Binary x2: the discrete change mean(exp(eta|x2=1) - exp(eta|x2=0)), the
        # same convention marginal_effects already uses for logit/probit/Tobit.
        @test isapprox(me.effects[3], -0.550357586342; rtol=1e-9)
        # Intercept has no marginal effect.
        @test isnan(me.effects[1]) && isnan(me.se[1])
        @test all(me.se[2:3] .> 0)
        @test me.type == :ame

        irr = incidence_rate_ratio(m)
        @test isapprox(irr.or, [1.625324876296, 1.819410632060, 0.709873804700]; rtol=1e-9)
        # delta method: SE(IRR) = IRR * SE(beta); CI formed on the log scale
        @test irr.se ≈ irr.or .* stderror(m) atol = 1e-12
        @test irr.ci_lower[2] < irr.or[2] < irr.ci_upper[2]
        # #546: count models label the ratio table as IRR, not Odds Ratios
        @test irr.title == "Incidence Rate Ratios"
        @test irr.ratio_label == "IRR"
        @test occursin("Incidence Rate", sprint(show, irr))
        @test !occursin("Odds Ratio", sprint(show, irr))

        # NegBin2 has the same conditional mean, so the same AME formula applies.
        mn = estimate_nbreg(D.y_nb, D.X; varnames=VN)
        men = marginal_effects(mn)
        @test isapprox(men.effects[2], mean(mn.fitted) * coef(mn)[2]; rtol=1e-9)
        @test all(isfinite, men.se[2:3])
        irrn = incidence_rate_ratio(mn)
        @test irrn.or ≈ exp.(coef(mn)) atol = 1e-12
    end

    # =========================================================================
    # 2. Analytic oracles — exact, no external engine involved
    # =========================================================================
    @testset "analytic: intercept-only Poisson is log(ybar)" begin
        y = D.y_pois
        m = estimate_poisson(y, reshape(ones(D.n), D.n, 1); cov_type=:mle)
        @test coef(m)[1] ≈ log(mean(y)) atol = 1e-9
        # ... and its information-matrix SE is exactly 1/sqrt(sum(y))
        @test stderror(m)[1] ≈ 1 / sqrt(sum(y)) rtol = 1e-8
    end

    @testset "analytic: score identity mean(mu) == mean(y) with an intercept" begin
        # The first-order condition for the intercept is sum(y - mu) = 0.
        for m in (estimate_poisson(D.y_pois, D.X), estimate_poisson(D.y_nb, D.X))
            @test mean(m.fitted) ≈ mean(m.y) atol = 1e-8
        end
        # Holds for NegBin2 too: its intercept score is sum((y-mu)/(1+alpha*mu)),
        # which is NOT the same identity, so this one is checked directly.
        mn = estimate_nbreg(D.y_nb, D.X)
        w = 1 ./ (1 .+ mn.alpha .* mn.fitted)
        @test abs(sum((mn.y .- mn.fitted) .* w)) < 1e-5
    end

    @testset "analytic: saturated design reproduces the cell means" begin
        # With a full set of group dummies the Poisson MLE is exactly the group
        # mean, so this checks the IRLS fixed point against a closed form.
        g = repeat(1:4, inner=25)
        y = Float64[2, 5, 1, 3, 0, 4, 2, 6, 1, 2, 3, 3, 7, 2, 1, 4, 0, 2, 5, 3,
                    1, 2, 4, 3, 2]
        y = vcat(y, y .+ 1, y .* 2, y .+ 3)
        Xd = Float64[gi == j for gi in g, j in 1:4]
        m = estimate_poisson(y, Xd; cov_type=:mle)
        cell_means = [mean(y[g .== j]) for j in 1:4]
        @test exp.(coef(m)) ≈ cell_means rtol = 1e-9
        @test m.deviance >= 0
    end

    @testset "analytic: NegBin2 log-likelihood equals the Poisson limit as alpha -> 0" begin
        y = D.y_pois; X = D.X
        b = [0.5, 0.6, -0.4]
        off = zeros(D.n)
        mu = exp.(X * b)
        ll_pois = _poisson_loglik(y, mu)
        for a in (1e-4, 1e-5, 1e-6)
            @test isapprox(_nb2_loglik(b, a, y, X, off), ll_pois; atol=1e-2)
        end
        @test abs(_nb2_loglik(b, 1e-7, y, X, off) - ll_pois) <
              abs(_nb2_loglik(b, 1e-4, y, X, off) - ll_pois)
    end

    # =========================================================================
    # 3. Properties
    # =========================================================================
    @testset "NegBin2 collapses to Poisson on equidispersed data" begin
        # MASS::glm.nb(y_pois ~ x1 + x2): theta = 312.88, i.e. alpha = 0.0031961
        mn = estimate_nbreg(D.y_pois, D.X; varnames=VN)
        mp = estimate_poisson(D.y_pois, D.X; varnames=VN, cov_type=:mle)
        @test mn.alpha < 0.02
        @test isapprox(coef(mn), [0.4855665908, 0.5982314136, -0.3420423567]; rtol=1e-6)
        @test isapprox(coef(mn), coef(mp); rtol=1e-2)
    end

    @testset "robust and MLE errors diverge exactly where they should" begin
        # Equidispersed: the sandwich and the information agree closely.
        r_eq = stderror(estimate_poisson(D.y_pois, D.X))
        m_eq = stderror(estimate_poisson(D.y_pois, D.X; cov_type=:mle))
        @test maximum(abs.(r_eq ./ m_eq .- 1)) < 0.05

        # Overdispersed: the naive errors understate uncertainty badly. R's
        # summary(glm(y_nb ~ x1+x2, family=poisson)) gives 0.04776/0.03821/0.07975,
        # against the hand-built sandwich 0.06680/0.05223/0.11629.
        r_od = stderror(estimate_poisson(D.y_nb, D.X))
        m_od = stderror(estimate_poisson(D.y_nb, D.X; cov_type=:mle))
        @test isapprox(r_od, [0.066795507561, 0.052226907658, 0.116293872206]; rtol=1e-7)
        @test isapprox(m_od, [0.047760471605, 0.038208564526, 0.079747232295]; rtol=1e-4)
        @test all(r_od .> 1.3 .* m_od)      # ~40% wider
        # ... which is exactly why :robust is the default.
        @test estimate_poisson(D.y_nb, D.X).cov_type == :robust
    end

    @testset "cluster-robust errors are available and differ from :robust" begin
        clusters = repeat(1:20, inner=20)
        mc = estimate_poisson(D.y_nb, D.X; cov_type=:cluster, clusters=clusters)
        @test mc.cov_type == :cluster
        @test all(isfinite, stderror(mc))
        @test !isapprox(stderror(mc), stderror(estimate_poisson(D.y_nb, D.X)); rtol=1e-6)
        @test_throws ArgumentError estimate_poisson(D.y_nb, D.X; cov_type=:cluster)
    end

    @testset "frequency-weighted duplication leaves the coefficients alone" begin
        # Stacking the sample twice doubles the log-likelihood and leaves beta
        # unchanged — a basic consistency check on the IRLS objective.
        y2 = vcat(D.y_pois, D.y_pois)
        X2 = vcat(D.X, D.X)
        m1 = estimate_poisson(D.y_pois, D.X; cov_type=:mle)
        m2 = estimate_poisson(y2, X2; cov_type=:mle)
        @test coef(m2) ≈ coef(m1) rtol = 1e-8
        @test loglikelihood(m2) ≈ 2 * loglikelihood(m1) rtol = 1e-8
        @test stderror(m2) ≈ stderror(m1) ./ sqrt(2) rtol = 1e-6
    end

    # =========================================================================
    # 4. Input validation
    # =========================================================================
    @testset "errors and edge cases" begin
        X = D.X
        @test_throws ArgumentError estimate_poisson([1.0, 2.0, -1.0], ones(3, 1))
        @test_throws ArgumentError estimate_poisson([1.0, 2.5, 3.0], ones(3, 1))
        @test_throws ArgumentError estimate_nbreg([1.0, 2.0, -3.0], ones(3, 1))
        @test_throws ArgumentError estimate_nbreg([0.5, 1.0, 2.0], ones(3, 1))
        # both offset and exposure
        @test_throws ArgumentError estimate_poisson(D.y_pois, X;
                                                    offset=zeros(D.n), exposure=ones(D.n))
        # nonpositive exposure
        @test_throws ArgumentError estimate_poisson(D.y_pois, X;
                                                    exposure=vcat(0.0, ones(D.n - 1)))
        # wrong lengths
        @test_throws ArgumentError estimate_poisson(D.y_pois, X; offset=zeros(3))
        @test_throws ArgumentError estimate_poisson(D.y_pois, X; varnames=["a"])
        @test_throws ArgumentError estimate_poisson(D.y_pois, X; cov_type=:nonsense)
        @test_throws ArgumentError estimate_poisson(D.y_pois, X[1:10, :])

        # Integer counts and integer-valued floats are both accepted.
        yi = Int.(round.(D.y_pois))
        @test coef(estimate_poisson(yi, D.X)) ≈ coef(estimate_poisson(D.y_pois, D.X)) atol = 1e-12
        # An all-zero response is degenerate but must not throw.
        mz = estimate_poisson(zeros(50), hcat(ones(50), randn(MersenneTwister(1), 50)))
        @test all(mz.fitted .< 1e-3)
    end

    # =========================================================================
    # 5. StatsAPI, prediction, display
    # =========================================================================
    @testset "StatsAPI surface" begin
        m = estimate_poisson(D.y_pois, D.X; varnames=VN, cov_type=:mle)
        mn = estimate_nbreg(D.y_nb, D.X; varnames=VN)

        @test nobs(m) == D.n
        @test dof(m) == 3
        @test dof_residual(m) == D.n - 3
        @test dof(mn) == 4                        # beta + alpha
        @test dof_residual(mn) == D.n - 4
        @test length(residuals(m)) == D.n
        @test length(fitted(m)) == D.n
        @test predict(m) === m.fitted
        @test size(vcov(m)) == (3, 3)
        @test size(vcov(mn)) == (3, 3)            # beta block only
        @test size(mn.vcov_mat) == (4, 4)         # joint (beta, alpha)
        @test size(confint(m)) == (3, 2)
        @test !SA.islinear(m)
        @test 0 <= m.pseudo_r2 <= 1
        @test SA.deviance(m) ≈ m.deviance
        @test SA.nulldeviance(m) ≈ m.null_deviance

        # Deviance residuals: sum of squares is the deviance.
        @test sum(abs2, residuals(m)) ≈ m.deviance rtol = 1e-9

        # predict on new data
        Xnew = [1.0 0.5 0.0; 1.0 -1.0 1.0]
        p = predict(m, Xnew)
        @test p ≈ exp.(Xnew * coef(m)) atol = 1e-12
        @test predict(m, Xnew; exposure=[2.0, 3.0]) ≈ p .* [2.0, 3.0] atol = 1e-10
        @test_throws ArgumentError predict(m, ones(2, 5))
    end

    @testset "display and refs" begin
        m = estimate_poisson(D.y_pois, D.X; varnames=VN)
        mn = estimate_nbreg(D.y_nb, D.X; varnames=VN)
        dt = dispersion_test(estimate_poisson(D.y_nb, D.X))

        for (obj, needle) in ((m, "Poisson"), (mn, "Negative Binomial"),
                              (dt, "Overdispersion"))
            io = IOBuffer(); show(io, obj); s = String(take!(io))
            @test occursin(needle, s)
            @test !isempty(strip(s))
            @test !occursin("omitted", s)           # no PrettyTables cropping
        end

        io = IOBuffer(); show(io, m); s = String(take!(io))
        @test occursin("Robust (QMLE)", s)          # cov_type via _label, not the raw symbol
        io = IOBuffer(); show(io, mn); s = String(take!(io))
        @test occursin("alpha", s)
        io = IOBuffer(); show(io, dt); s = String(take!(io))
        @test occursin("estimate_nbreg", s)         # the verdict names the remedy

        io = IOBuffer(); refs(io, m); @test !isempty(String(take!(io)))
        io = IOBuffer(); refs(io, mn); @test !isempty(String(take!(io)))
        io = IOBuffer(); refs(io, dt); @test !isempty(String(take!(io)))

        # report(io, x) is wired for all three (the docs call it).
        for obj in (m, mn, dt)
            io = IOBuffer(); report(io, obj); s = String(take!(io))
            @test !isempty(strip(s))
        end
    end

    @testset "internal helpers" begin
        y = D.y_pois
        mu = fill(mean(y), D.n)
        @test _poisson_deviance(y, y .+ 0.0 .+ eps()) < 1e-6      # perfect fit -> 0
        @test _poisson_deviance(y, mu) > 0
        # zero counts contribute 2*mu and must not produce NaN
        @test isfinite(_poisson_deviance([0.0, 0.0], [1.0, 2.0]))
        @test _poisson_deviance([0.0], [1.5]) ≈ 3.0 atol = 1e-12
        # the moment estimator of alpha recovers the truth on NB data
        mp = estimate_poisson(D.y_nb, D.X)
        @test 0.2 < _ct_moment_alpha(D.y_nb, mp.fitted) < 0.8
        # IRLS starting value is finite even when every count is zero
        b, mu0, w0, ll, cv, it = _irls_poisson(zeros(20), ones(20, 1), zeros(20))
        @test all(isfinite, mu0) && all(isfinite, w0) && isfinite(ll)
    end
end
