# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-16 (#424): Multivariate GARCH — CCC (Bollerslev 1990), DCC/cDCC (Engle 2002 /
# Aielli 2013), scalar/diagonal BEKK(1,1) (Engle-Kroner 1995).
#
# Oracle strategy (per docs/plans/2026-07-16-ev-series-specs.json), no invented numerics:
#   (1) NESTING, MACHINE-TOL [primary]: CCC ≡ DCC with a=b=0. The DCC correlation
#       recursion at a=b=0 gives R_t = corr(Q̄) constant = the CCC correlation,
#       reproduced to ~1e-12.
#   (2) CORRELATION TARGETING IDENTITY, MACHINE-TOL: Q̄ = (1/T) Σ z_t z_t' recovered
#       exactly from the standardized residuals.
#   (3) MARGIN REPRODUCTION, MACHINE-TOL: each CCC/DCC margin equals the standalone
#       `estimate_garch` fit on that column (same ω,α,β), since the margins ARE the
#       reused univariate estimator.
#   (4) POSITIVE SEMIDEFINITENESS: every conditional Σ_t on the fitted path has
#       min-eigenvalue ≥ −tol (CCC, DCC, scalar & diagonal BEKK).
#   (5) CORRELATION BOUNDS: every conditional correlation ∈ [−1,1] with unit diagonal.
#   (6) PARAMETER RECOVERY: a fixed-seed T=3000 DCC(a=0.05,b=0.90,ρ̄=0.3) path recovers
#       (a,b) within a loose stated tolerance, and DCC log-lik ≥ CCC log-lik (nesting).
#   (7) CROSS-IMPL KNOWN-VALUE (documented, NOT asserted): R `rmgarch::dccfit`.
#       `rmgarch` is NOT installed in this environment — its dependency tree
#       (`rugarch`, `Rmpfr`, `SkewHyperbolic`, `GeneralizedHyperbolic`, …) fails to
#       build within the harness time budget — so no decimals are hard-coded
#       (fabrication is disallowed). The reference cross-check would be
#       (Ghalanos, rmgarch ≥ 1.3, standard DCC(1,1) with GARCH(1,1) norm margins):
#           library(rmgarch)
#           uspec <- multispec(replicate(2, ugarchspec(
#               variance.model=list(model="sGARCH", garchOrder=c(1,1)),
#               mean.model=list(armaOrder=c(0,0), include.mean=TRUE),
#               distribution.model="norm")))
#           dccspec2 <- dccspec(uspec, dccOrder=c(1,1), distribution="mvnorm")
#           fit <- dccfit(dccspec2, data = Y)
#           # coef(fit): ..., [Joint]dcca1 (=a), [Joint]dccb1 (=b); likelihood(fit);
#           # rcor(fit)[,,t] gives R_t.
#       reproducing Q_t = (1−a−b)Q̄ + a z_{t−1}z_{t−1}' + b Q_{t−1},
#       R_t = diag(Q_t)^{−1/2} Q_t diag(Q_t)^{−1/2}. Oracles (1)–(6) pin the estimator
#       to ground truth instead.

using Test
using MacroEconometricModels
using Random
using Statistics
using LinearAlgebra
using StatsAPI

const _MG = MacroEconometricModels

# --- fixed-seed simulators ---------------------------------------------------

"""Bivariate series with CONSTANT correlation (drives CCC and the DCC→CCC collapse)."""
function _sim_ccc(T::Int; rho=0.4, seed=42)
    rng = MersenneTwister(seed)
    Y = zeros(T, 2); h1 = 1.0; h2 = 1.0
    for t in 2:T
        h1 = 0.05 + 0.10 * Y[t-1, 1]^2 + 0.85 * h1
        h2 = 0.05 + 0.10 * Y[t-1, 2]^2 + 0.85 * h2
        z1 = randn(rng); z2 = randn(rng)
        Y[t, 1] = sqrt(h1) * z1
        Y[t, 2] = sqrt(h2) * (rho * z1 + sqrt(1 - rho^2) * z2)
    end
    Y
end

"""Bivariate DCC(1,1) process with time-varying correlation and GARCH(1,1) margins."""
function _sim_dcc(T::Int; a=0.05, b=0.90, rho=0.3, seed=7)
    rng = MersenneTwister(seed)
    Qbar = [1.0 rho; rho 1.0]; Q = copy(Qbar)
    Y = zeros(T, 2); h = [1.0, 1.0]
    for t in 1:T
        d = sqrt.(diag(Q)); R = Q ./ (d * d'); R = (R + R') / 2
        L = cholesky(Symmetric(R)).L
        zt = L * randn(rng, 2)
        e = sqrt.(h) .* zt
        Y[t, :] = e
        Q = (1 - a - b) * Qbar + a * (zt * zt') + b * Q; Q = (Q + Q') / 2
        for i in 1:2
            h[i] = 0.02 + 0.08 * e[i]^2 + 0.90 * h[i]
        end
    end
    Y
end

_min_eig(H) = minimum(eigen(Symmetric(Matrix(H))).values)

@testset "Multivariate GARCH (EV-16)" begin
    tol = 1e-10

    @testset "CCC estimation & structure" begin
        Y = _sim_ccc(700)
        m = estimate_ccc(Y)
        @test m.kind === :ccc
        @test m.n == 2
        @test length(m.margins) == 2
        @test size(m.H) == (2, 2, 700)
        @test isfinite(m.loglik)
        @test m.converged

        # (5) correlation bounds + unit diagonal (constant R)
        @test m.R[1, 1] ≈ 1 atol = tol
        @test m.R[2, 2] ≈ 1 atol = tol
        @test -1 ≤ m.R[1, 2] ≤ 1
        @test issymmetric(m.R)

        # (4) every Σ_t PSD
        @test all(_min_eig(m.H[:, :, t]) ≥ -1e-8 for t in 1:size(m.H, 3))

        # accessors
        @test covariances(m) === m.H
        R3 = correlations(m)
        @test size(R3) == (2, 2, 700)
        @test all(R3[:, :, t] ≈ m.R for t in (1, 350, 700))  # constant broadcast
        V = variances(m)
        @test size(V) == (700, 2)
        @test all(V[t, i] ≈ m.H[i, i, t] for t in (1, 200), i in 1:2)
        @test all(V .> 0)
    end

    @testset "(3) CCC margins reproduce standalone estimate_garch" begin
        Y = _sim_ccc(700)
        m = estimate_ccc(Y)
        for i in 1:2
            g = estimate_garch(Y[:, i], 1, 1)
            @test m.margins[i].omega ≈ g.omega atol = 1e-8
            @test m.margins[i].alpha[1] ≈ g.alpha[1] atol = 1e-8
            @test m.margins[i].beta[1] ≈ g.beta[1] atol = 1e-8
        end
    end

    @testset "(2) correlation-targeting identity Q̄ = (1/T)Σ zₜz'ₜ" begin
        Y = _sim_ccc(500)
        m = estimate_ccc(Y)
        z = hcat(m.margins[1].standardized_residuals, m.margins[2].standardized_residuals)
        Qbar = _MG._uncentered_Q(z)
        @test Qbar ≈ (transpose(z) * z) / size(z, 1) atol = tol
        @test issymmetric(Qbar)
        # CCC R is the normalization of Q̄
        @test m.R ≈ _MG._corr_from_cov(Qbar) atol = tol
    end

    @testset "(1) NESTING: CCC ≡ DCC with a=b=0" begin
        Y = _sim_ccc(600)
        m = estimate_ccc(Y)
        z = hcat(m.margins[1].standardized_residuals, m.margins[2].standardized_residuals)
        Qbar = _MG._uncentered_Q(z)
        Rpath = _MG._dcc_R_path(z, 0.0, 0.0, Qbar; correction = :none)
        # a=b=0 ⇒ Q_t constant = Q̄ ⇒ R_t = CCC correlation, for every t
        @test all(isapprox(Rpath[:, :, t], m.R; atol = tol) for t in 1:size(Rpath, 3))
    end

    @testset "DCC estimation, bounds & PSD" begin
        Y = _sim_dcc(1500)
        m = estimate_dcc(Y)
        @test m.kind === :dcc
        @test m.correction === :none
        @test length(m.params) == 2
        a, b = m.params
        @test a ≥ 0
        @test b ≥ 0
        @test a + b < 1
        @test size(m.R) == (2, 2, 1500)   # time-varying

        # (5) correlations in [-1,1], unit diagonal across the whole path
        R = correlations(m)
        @test all(isapprox(R[1, 1, t], 1; atol = 1e-9) for t in 1:size(R, 3))
        @test all(isapprox(R[2, 2, t], 1; atol = 1e-9) for t in 1:size(R, 3))
        @test all(-1 ≤ R[1, 2, t] ≤ 1 for t in 1:size(R, 3))

        # (4) PSD across path
        @test all(_min_eig(m.H[:, :, t]) ≥ -1e-8 for t in 1:size(m.H, 3))

        # second-stage SEs finite & positive (interior estimate)
        se = stderror(m)
        @test length(se) == 2
        @test all(isfinite, se)
        @test all(se .> 0)
    end

    @testset "(6) DCC parameter recovery + nesting log-lik" begin
        Y = _sim_dcc(3000; a = 0.05, b = 0.90, rho = 0.3)
        mc = estimate_ccc(Y)
        md = estimate_dcc(Y)
        a, b = md.params
        # loose recovery tolerance (two-step QMLE, finite sample)
        @test isapprox(a, 0.05; atol = 0.03)
        @test isapprox(b, 0.90; atol = 0.06)
        @test a + b < 1
        # DCC nests CCC ⇒ log-lik must be ≥ CCC (strictly higher for dynamic corr DGP)
        @test md.loglik ≥ mc.loglik - 1e-6
        @test md.loglik > mc.loglik + 10   # dynamic correlation is real here
    end

    @testset "cDCC (Aielli 2013) correction runs & stays valid" begin
        Y = _sim_dcc(1500)
        m = estimate_dcc(Y; correction = :aielli)
        @test m.correction === :aielli
        a, b = m.params
        @test a ≥ 0 && b ≥ 0 && a + b < 1
        R = correlations(m)
        @test all(-1 ≤ R[1, 2, t] ≤ 1 for t in 1:size(R, 3))
        @test all(_min_eig(m.H[:, :, t]) ≥ -1e-8 for t in 1:size(m.H, 3))
    end

    @testset "scalar BEKK estimation & PSD" begin
        Y = _sim_dcc(1200)
        m = estimate_bekk(Y)
        @test m.kind === :bekk
        @test m.bekk_kind === :scalar
        @test isempty(m.margins)          # BEKK models covariance directly
        @test length(m.params) == 2
        a, b = m.params
        @test a ≥ 0 && b ≥ 0 && a + b < 1
        @test size(m.H) == (2, 2, 1200)
        # (4) PSD across the whole path
        @test all(_min_eig(m.H[:, :, t]) ≥ -1e-8 for t in 1:size(m.H, 3))
        # unconditional covariance targeting: sample cov ≈ H̄
        resid = Y .- vec(mean(Y; dims = 1))'
        Sigbar = cov(resid; corrected = false)
        Hbar = dropdims(mean(m.H; dims = 3); dims = 3)
        @test isapprox(Hbar, Sigbar; rtol = 0.15)
        se = stderror(m)
        @test all(isfinite, se) && all(se .> 0)
    end

    @testset "diagonal BEKK estimation & PSD" begin
        Y = _sim_dcc(1200)
        m = estimate_bekk(Y; kind = :diagonal)
        @test m.bekk_kind === :diagonal
        @test length(m.params) == 4       # [a1,a2,b1,b2]
        @test all(m.params .≥ 0)
        @test size(m.H) == (2, 2, 1200)
        @test all(_min_eig(m.H[:, :, t]) ≥ -1e-8 for t in 1:size(m.H, 3))
    end

    @testset "forecast shapes & PSD" begin
        Y = _sim_dcc(1000)
        for m in (estimate_ccc(Y), estimate_dcc(Y), estimate_bekk(Y),
                  estimate_bekk(Y; kind = :diagonal))
            f = forecast(m, 8)
            @test size(f) == (2, 2, 8)
            @test all(_min_eig(f[:, :, k]) ≥ -1e-8 for k in 1:8)
            @test all(issymmetric(round.(f[:, :, k]; digits = 10)) for k in 1:8)
        end
        @test_throws ArgumentError forecast(estimate_ccc(Y), 0)
    end

    @testset "display, report & refs" begin
        Y = _sim_dcc(800)
        for m in (estimate_ccc(Y), estimate_dcc(Y), estimate_bekk(Y))
            io = IOBuffer()
            show(io, m)
            s = String(take!(io))
            @test occursin("Multivariate GARCH", s)
            @test occursin("Log-likelihood", s)
            rio = IOBuffer()
            refs(rio, m)
            @test length(String(take!(rio))) > 0
        end
    end

    @testset "plot_result views" begin
        Y = _sim_dcc(600)
        m = estimate_dcc(Y)
        p1 = plot_result(m; view = :correlations)
        @test p1 isa MacroEconometricModels.PlotOutput
        @test occursin("svg", p1.html)
        p2 = plot_result(m; view = :covariance_heatmap)
        @test p2 isa MacroEconometricModels.PlotOutput
        p3 = plot_result(m; view = :covariance_heatmap, at = 10)
        @test p3 isa MacroEconometricModels.PlotOutput
        @test_throws ArgumentError plot_result(m; view = :bogus)
        @test_throws ArgumentError plot_result(m; view = :covariance_heatmap, at = 10_000)
    end

    @testset "input validation & StatsAPI" begin
        @test_throws ArgumentError estimate_ccc(randn(100, 1))   # need ≥2 series
        @test_throws ArgumentError estimate_dcc(randn(100, 2); correction = :bad)
        @test_throws ArgumentError estimate_bekk(randn(100, 2); kind = :bad)
        Y = _sim_dcc(500)
        m = estimate_dcc(Y)
        @test StatsAPI.nobs(m) == 500
        @test StatsAPI.coef(m) == m.params
        @test StatsAPI.loglikelihood(m) == m.loglik
        @test StatsAPI.dof(m) > 2
        @test !StatsAPI.islinear(m)
        @test isempty(stderror(estimate_ccc(Y)))   # CCC has no second-stage params
    end
end
