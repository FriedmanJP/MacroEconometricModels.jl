# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-18 (#426): Heckman (1979) sample-selection model (two-step Heckit + FIML).

using Test
using MacroEconometricModels
using MacroEconometricModels: _mills, _heckman_negll
using LinearAlgebra, Statistics, Distributions, Random, Logging
import StatsAPI

# =============================================================================
# ORACLE — Wooldridge (2010) Example 17.5 / Greene sample-selection example on the
# Mroz (1987) 753-obs extract (== sampleSelection::Mroz87 == wooldridge::mroz).
# Specification:
#   selection: inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6
#   outcome:   lwage ~ educ + exper + expersq
#
# The two-step (heckit) reference is a *deterministic* base-R replication of
# sampleSelection::heckit (identical procedure); the exact R calls are:
#
#   library(wooldridge); data(mroz)
#   sel  <- glm(inlf ~ nwifeinc+educ+exper+expersq+age+kidslt6+kidsge6,
#               family=binomial(link="probit"), data=mroz)
#   imr  <- dnorm(predict(sel))/pnorm(predict(sel))
#   out  <- lm(lwage ~ educ+exper+expersq+imr, data=subset(mroz, inlf==1))
#   # delta <- imr*(imr+zg); sigma^2 = e'e/n_sel + mean(delta)*b_imr^2; rho = b_imr/sigma
#
# (sampleSelection could not be installed in-CI — its systemfit/VGAM/car deps do
#  not build headless — so the point estimates are reproduced by the identical
#  base-R Heckit procedure above, which yields sampleSelection::heckit's numbers.)
# =============================================================================

# Two-step (base-R heckit) reference values.
const HECK_GAMMA = [0.2700736, -0.0120236, 0.1309040, 0.1233472,
                    -0.0018871, -0.0528524, -0.8683247, 0.0360056]   # selection
const HECK_BETA  = [-0.5781023, 0.1090655, 0.0438873, -0.0008591]    # outcome (educ,exper,expersq)
const HECK_SIGMA = 0.6636287
const HECK_RHO   = 0.0486136
const HECK_LAMBDA = 0.0322614   # = rho*sigma = coefficient on the Mills ratio

function _mroz_design()
    d = load_example(:mroz)
    n = d.N_obs
    inlf  = d[:, "inlf"]
    lwage = d[:, "lwage"]
    X = hcat(ones(n), d[:, "educ"], d[:, "exper"], d[:, "expersq"])
    Z = hcat(ones(n), d[:, "nwifeinc"], d[:, "educ"], d[:, "exper"],
             d[:, "expersq"], d[:, "age"], d[:, "kidslt6"], d[:, "kidsge6"])
    on = ["const", "educ", "exper", "expersq"]
    sn = ["const", "nwifeinc", "educ", "exper", "expersq", "age", "kidslt6", "kidsge6"]
    (; d, n, inlf, lwage, X, Z, on, sn)
end

@testset "EV-18 Heckman selection model" begin

    @testset "load_example(:mroz)" begin
        d = load_example(:mroz)
        @test d isa CrossSectionData
        @test d.N_obs == 753
        @test d.n_vars == 22
        # lwage / wage are NaN for the 325 non-participants.
        @test count(isfinite, d[:, "lwage"]) == 428
        @test count(!isfinite, d[:, "lwage"]) == 325
        @test sum(d[:, "inlf"]) == 428
        @test occursin("Mroz", first(d.desc))
        @test :mroz1987 in d.source_refs
    end

    @testset "Two-step Heckit — oracle match (Wooldridge Ex 17.5)" begin
        s = _mroz_design()
        m = estimate_heckman(s.lwage, s.X, s.inlf, s.Z; method=:twostep,
                             outcome_names=s.on, select_names=s.sn)
        @test m isa HeckmanModel
        @test m.method === :twostep
        @test m.n_selected == 428
        @test m.n_total == 753
        # Selection (probit) coefficients.
        @test m.gamma ≈ HECK_GAMMA atol=1e-4
        # Outcome coefficients [const, educ, exper, expersq].
        @test m.beta ≈ HECK_BETA atol=1e-4
        @test m.sigma ≈ HECK_SIGMA atol=1e-4
        @test m.rho ≈ HECK_RHO atol=1e-4
        @test m.lambda ≈ HECK_LAMBDA atol=1e-4
        @test m.converged
    end

    @testset "sigma^2 / rho closed-form recovery" begin
        s = _mroz_design()
        m = estimate_heckman(s.lwage, s.X, s.inlf, s.Z; method=:twostep)
        # Reconstruct sigma^2 = e'e/n_sel + mean(delta)*beta_lambda^2, rho = beta_lambda/sigma
        # independently from the fitted Mills ratio and outcome regression.
        sel = findall(==(1.0), s.inlf)
        Xs = s.X[sel, :]; ys = s.lwage[sel]
        W = hcat(Xs, m.mills)
        bstar = W \ ys
        resid = ys .- W * bstar
        blam = bstar[end]
        # zg over selected obs from the fitted probit index.
        zg_s = (s.Z * m.gamma)[sel]
        delta = m.mills .* (m.mills .+ zg_s)
        sig2 = dot(resid, resid) / length(ys) + mean(delta) * blam^2
        @test sqrt(sig2) ≈ m.sigma atol=1e-8
        @test blam / sqrt(sig2) ≈ m.rho atol=1e-8
        @test blam ≈ m.lambda atol=1e-10
        @test all(0 .< delta .< 1)          # delta_i ∈ (0,1)
    end

    @testset "Greene corrected covariance ≠ naive OLS, and is PD" begin
        s = _mroz_design()
        m = estimate_heckman(s.lwage, s.X, s.inlf, s.Z; method=:twostep)
        # Greene-corrected covariance must be symmetric PD.
        V = m.vcov_beta
        @test issymmetric(Symmetric(V))
        @test minimum(eigvals(Symmetric(V))) > 0
        # Naive second-stage OLS SE on the Mills coefficient.
        sel = findall(==(1.0), s.inlf)
        Xs = s.X[sel, :]; ys = s.lwage[sel]
        W = hcat(Xs, m.mills)
        bstar = W \ ys; resid = ys .- W * bstar
        s2 = dot(resid, resid) / (length(ys) - size(W, 2))
        Vn = s2 * inv(W'W)
        naive_lambda_se = sqrt(Vn[end, end])
        # The correction is ACTIVE (not a stub): corrected λ SE differs from naive OLS.
        @test abs(m.lambda_se - naive_lambda_se) > 1e-4
        @test m.lambda_se > 0
    end

    @testset "Two-step λ t-test — H0: no selection" begin
        s = _mroz_design()
        m = estimate_heckman(s.lwage, s.X, s.inlf, s.Z; method=:twostep)
        tstat = m.lambda / m.lambda_se
        # On Mroz the selection term is small and insignificant (rho ≈ 0.05):
        # positive point estimate, |t| well under 1 (H0 not rejected).
        @test m.lambda > 0
        @test 0 < tstat < 1
    end

    @testset "FIML MLE — improves the two-step and matches its own optimum" begin
        s = _mroz_design()
        mt = estimate_heckman(s.lwage, s.X, s.inlf, s.Z; method=:twostep)
        mm = estimate_heckman(s.lwage, s.X, s.inlf, s.Z; method=:mle)
        @test mm.method === :mle
        @test mm.converged
        # FIML log-likelihood ≥ the two-step profile value at the two-step point
        # (R's optim gets stuck at the two-step in this flat-rho region; our BFGS
        #  finds a strictly higher likelihood — documented rho≈0 pathology).
        @test mm.loglik ≥ mt.loglik - 1e-6
        @test mm.loglik ≈ -832.8851 atol=1e-2
        # Estimates stay close to the two-step (small rho ⇒ near-collinear).
        @test mm.beta ≈ mt.beta atol=0.05
        @test mm.gamma ≈ mt.gamma atol=0.01
        @test mm.sigma ≈ mt.sigma atol=0.02
        @test 0 ≤ mm.rho < 0.1
        # MLE delivers rho/sigma standard errors directly (two-step does not).
        @test isfinite(mm.rho_se) && mm.rho_se > 0
        @test isfinite(mm.sigma_se) && mm.sigma_se > 0
        @test isfinite(mm.lambda_se) && mm.lambda_se > 0
        @test isnan(mt.rho_se) && isnan(mt.sigma_se)   # two-step: not identified directly
    end

    @testset "Simulated selection DGP — parameter recovery" begin
        Random.seed!(20260717)
        n = 6000
        z2 = randn(n); x2 = randn(n)
        Z = hcat(ones(n), z2, x2)          # z2 is the exclusion restriction
        X = hcat(ones(n), x2)
        gam = [0.3, 0.8, 0.5]; bet = [1.0, 2.0]
        rho_true = 0.6; sig_true = 1.5
        L = [1.0 0.0; rho_true*sig_true sig_true*sqrt(1-rho_true^2)]
        e = randn(n, 2) * L'
        u = e[:, 1]; eps2 = e[:, 2]
        d = Float64.((Z * gam .+ u) .> 0)
        y = X * bet .+ eps2
        y[d .== 0] .= NaN                   # unobserved outcome for non-selected
        mm = estimate_heckman(y, X, d, Z; method=:mle)
        @test mm.converged
        @test mm.rho ≈ rho_true atol=0.08
        @test mm.sigma ≈ sig_true atol=0.08
        @test mm.beta ≈ bet atol=0.08
        # Two-step recovers the same neighborhood.
        mt = estimate_heckman(y, X, d, Z; method=:twostep)
        @test mt.rho ≈ rho_true atol=0.12
        @test mt.beta ≈ bet atol=0.1
    end

    @testset "Exclusion-restriction warning (Z ⊆ span(X))" begin
        s = _mroz_design()
        # Z == X (up to intercept) ⇒ no exclusion restriction ⇒ @warn.
        @test_logs (:warn, r"no exclusion restriction") match_mode=:any estimate_heckman(
            s.lwage, s.X, s.inlf, s.X; method=:twostep)
        # A genuine exclusion (Z ⊋ X) must NOT warn about exclusion.
        @test_logs min_level=Logging.Warn estimate_heckman(
            s.lwage, s.X, s.inlf, s.Z; method=:twostep)
    end

    @testset "report / refs / StatsAPI" begin
        s = _mroz_design()
        m = estimate_heckman(s.lwage, s.X, s.inlf, s.Z; method=:twostep,
                             outcome_names=s.on, select_names=s.sn)
        io = IOBuffer(); show(io, m); str = String(take!(io))
        @test occursin("Heckman Selection Model", str)
        @test occursin("Selection equation", str)
        @test occursin("Outcome equation", str)
        @test occursin("no selection", str)          # two-step footer
        r = refs(m)
        @test occursin("Heckman", r)
        @test occursin("Mroz", r) || occursin("1987", r)
        # StatsAPI
        @test StatsAPI.coef(m) === m.beta
        @test size(StatsAPI.vcov(m)) == (length(m.beta), length(m.beta))
        @test length(StatsAPI.stderror(m)) == length(m.beta)
        @test StatsAPI.nobs(m) == 428
        @test StatsAPI.loglikelihood(m) == m.loglik
        @test size(StatsAPI.confint(m)) == (length(m.beta), 2)
        # MLE report shows the Wald footer.
        mm = estimate_heckman(s.lwage, s.X, s.inlf, s.Z; method=:mle,
                             outcome_names=s.on, select_names=s.sn)
        io2 = IOBuffer(); show(io2, mm); str2 = String(take!(io2))
        @test occursin("Wald", str2)
        @test occursin("Maximum likelihood", str2)
    end

    @testset "Input validation & Float fallback" begin
        s = _mroz_design()
        @test_throws ArgumentError estimate_heckman(s.lwage, s.X, s.inlf, s.Z; method=:bogus)
        # Non-binary selection indicator.
        bad = copy(s.inlf); bad[1] = 2.0
        @test_throws ArgumentError estimate_heckman(s.lwage, s.X, bad, s.Z)
        # Mismatched rows.
        @test_throws ArgumentError estimate_heckman(s.lwage, s.X[1:100, :], s.inlf, s.Z)
        # Int selection indicator (Float fallback path).
        di = Int.(s.inlf)
        m = estimate_heckman(s.lwage, s.X, di, s.Z; method=:twostep)
        @test m.beta ≈ HECK_BETA atol=1e-4
    end
end
