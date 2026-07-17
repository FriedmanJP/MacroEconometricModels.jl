# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-10 (#418): single-equation cointegrating regression — FMOLS / CCR / DOLS.

using Test
using MacroEconometricModels
using MacroEconometricModels: _cointreg_deter, _cointreg_lrv, _make_leadlag,
                              _dols_select_leadlag
using LinearAlgebra, Statistics, Random, DelimitedFiles
import StatsAPI as S

# =============================================================================
# Deterministic cointegrated DGP — byte-identical Julia code was used to write
# the CSVs fed to R's cointReg, so the hard-coded oracle constants below apply
# to exactly these series. y_t = 2 + 1.5·x_t + u_t, x_t a random walk.
#   endog=true : u_t = 0.4 u_{t-1} + e_t + 0.6 v_t (serial corr + endogeneity)
#   endog=false: u_t = e_t (strictly exogenous, u ⟂ v, i.i.d.)
# =============================================================================
function coint_dgp(; seed::Int=20260716, T::Int=200, endog::Bool=true)
    # The endogenous default is the R cointReg oracle; its exact (y,x) is pinned to
    # a committed fixture (test/gen_ev_fixtures.jl) because MersenneTwister is not
    # stable across Julia versions. The exogenous variant stays RNG-driven.
    if endog && seed == 20260716 && T == 200
        d = readdlm(joinpath(@__DIR__, "data", "coint_dgp_endog.csv"), ',', Float64)
        return d[:, 1], d[:, 2]
    end
    rng = MersenneTwister(seed)
    v = randn(rng, T)
    e = randn(rng, T)
    x = cumsum(v)
    u = zeros(T)
    if endog
        for t in 1:T
            ulag = t == 1 ? 0.0 : u[t-1]
            u[t] = 0.4 * ulag + e[t] + 0.6 * v[t]
        end
    else
        u .= e
    end
    y = 2.0 .+ 1.5 .* x .+ u
    return y, x
end

# -----------------------------------------------------------------------------
# NOTE ON THE BANDWIDTH CONVENTION (for the R oracle to match to machine tol):
# cointReg's Bartlett weight is 1 − j/bw (lags j = 1…ceil(bw)−1); this package's
# `lrcov`/`lrvar` use the Newey–West normalisation 1 − j/(bw+1) (lags j = 1…bw).
# They coincide when our `bandwidth = b` equals cointReg's `bandwidth = b+1`.
# The oracle therefore calls estimate_cointreg(...; bandwidth=3) against
# cointReg(..., bandwidth = 4).
# -----------------------------------------------------------------------------

@testset "Cointegrating Regression (FMOLS/CCR/DOLS)" begin
    y, x = coint_dgp(; endog=true)
    n = length(y)

    @testset "FMOLS — Phillips–Hansen vs R cointReg" begin
        # R (cointReg 0.2.0, Aschersleben & Wagner):
        #   d <- read.csv("coint_endog.csv", header=FALSE); y<-d[[1]]; x<-d[[2]]
        #   deter <- matrix(1, nrow=length(y), ncol=1)
        #   cointRegFM(x=x, y=y, deter=deter, kernel="ba", bandwidth=4, demeaning=FALSE)
        #   $theta   = (2.178624, 1.555044)   # (const, slope)
        #   $sd.theta= (0.11973959, 0.03897627)
        #   $omega.u.v = 2.844837
        m = estimate_cointreg(y, x; method=:fmols, trend=:const,
                              kernel=:bartlett, bandwidth=3)
        @test coef(m) ≈ [2.178624, 1.555044]      atol=1e-5
        @test S.stderror(m) ≈ [0.11973959, 0.03897627] atol=1e-6
        @test m.omega_uv ≈ 2.844837               atol=1e-5
        @test m.method === :fmols
        @test m.nobs == n
        @test length(m.residuals) == n
        @test m.fitted ≈ hcat(ones(n), x) * coef(m)
    end

    @testset "DOLS — Stock–Watson vs R cointReg" begin
        # R:  cointRegD(x=x, y=y, deter=deter, kernel="ba", bandwidth=4,
        #               n.lead=2, n.lag=2, demeaning=FALSE)
        #   $theta    = (2.206254, 1.531501)
        #   $sd.theta = (0.11106549, 0.03872308)
        m = estimate_cointreg(y, x; method=:dols, leads=2, lags=2,
                              kernel=:bartlett, bandwidth=3)
        @test coef(m) ≈ [2.206254, 1.531501]        atol=1e-5
        @test S.stderror(m) ≈ [0.11106549, 0.03872308] atol=1e-6
        @test m.leads == 2 && m.lags == 2

        # Automatic lead/lag selection reproduces cointReg's getLeadLag (k4/AIC):
        #   cointRegD(..., bandwidth="and", kmax="k4", info.crit="AIC")
        #   $lead.lag => n.lag=4, n.lead=0 ; $theta = (2.188488, 1.515048)
        ma = estimate_cointreg(y, x; method=:dols, bandwidth=:andrews, ic=:aic)
        @test ma.lags == 4 && ma.leads == 0
        @test coef(ma) ≈ [2.188488, 1.515048]       atol=1e-4
    end

    @testset "DEGENERATE: DOLS(0,0) ≡ OLS on levels" begin
        Z = hcat(ones(n), x)
        b_ols = Z \ y
        m = estimate_cointreg(y, x; method=:dols, leads=0, lags=0, bandwidth=3)
        @test coef(m) ≈ b_ols atol=1e-10       # coefficients identical to machine tol
        @test m.fitted ≈ Z * b_ols atol=1e-9
    end

    @testset "PROPERTY: FMOLS/CCR ≈ OLS under strict exogeneity" begin
        ye, xe = coint_dgp(; endog=false)
        ne = length(ye)
        b_ols = hcat(ones(ne), xe) \ ye
        # R oracle (informational): cointRegFM(..., bandwidth="and") on coint_exog.csv
        #   $theta = (2.104670, 1.527412); OLS = (2.108115, 1.530036).
        mf = estimate_cointreg(ye, xe; method=:fmols, bandwidth=:andrews)
        @test coef(mf) ≈ b_ols atol=2e-2       # no endogeneity/serial corr ⇒ FMOLS ≈ OLS
        mc = estimate_cointreg(ye, xe; method=:ccr, bandwidth=:andrews)
        @test coef(mc) ≈ b_ols atol=2e-2
    end

    @testset "PROPERTY: CCR ≈ FMOLS (asymptotically equivalent)" begin
        mf = estimate_cointreg(y, x; method=:fmols, bandwidth=3)
        mc = estimate_cointreg(y, x; method=:ccr, bandwidth=3)
        @test coef(mc) ≈ coef(mf) rtol=5e-2
        @test mc.omega_uv ≈ mf.omega_uv atol=1e-8   # same LRV pieces feed both
    end

    @testset "Deterministics: none / const / linear" begin
        mn = estimate_cointreg(y, x; method=:fmols, trend=:none, bandwidth=3)
        @test length(coef(mn)) == 1
        @test mn.varnames == ["x"]
        mc = estimate_cointreg(y, x; method=:fmols, trend=:const, bandwidth=3)
        @test mc.varnames == ["const", "x"]
        ml = estimate_cointreg(y, x; method=:fmols, trend=:linear, bandwidth=3)
        @test length(coef(ml)) == 3
        @test ml.varnames == ["const", "trend", "x"]
        # deterministic-block builder
        D, names = _cointreg_deter(10, :linear, Float64)
        @test size(D) == (10, 2)
        @test D[:, 1] == ones(10)
        @test D[:, 2] == collect(1.0:10.0)
        @test names == ["const", "trend"]
    end

    @testset "Stored long-run covariance pieces (EV-11/EV-22 layout)" begin
        m = estimate_cointreg(y, x; method=:fmols, bandwidth=3)
        # stacked (u, Δx): 2×2 here; Ω symmetric PSD; Ω = Λ + Λ' − Γ₀ (non-prewhitened).
        @test size(m.Omega) == (2, 2)
        @test size(m.Lambda) == (2, 2)
        @test size(m.Sigma) == (2, 2)
        @test m.Omega ≈ m.Omega'                    atol=1e-10
        @test m.Omega ≈ m.Lambda + m.Lambda' - m.Sigma atol=1e-8
        @test isposdef(m.Omega + 1e-8I)
        # conditional LR variance identity
        Ωuu = m.Omega[1, 1]; Ωuv = m.Omega[1, 2]; Ωvv = m.Omega[2, 2]
        @test m.omega_uv ≈ Ωuu - Ωuv^2 / Ωvv        atol=1e-8
    end

    @testset "DOLS robust (HAC) standard errors" begin
        m = estimate_cointreg(y, x; method=:dols, leads=2, lags=2,
                              bandwidth=3, dols_se=:robust)
        @test all(S.stderror(m) .> 0)
        @test length(coef(m)) == 2
        # coefficients are unchanged by the SE option
        ml = estimate_cointreg(y, x; method=:dols, leads=2, lags=2,
                               bandwidth=3, dols_se=:lrv)
        @test coef(m) ≈ coef(ml) atol=1e-12
    end

    @testset "Multi-regressor system" begin
        # Pinned to a committed fixture (test/gen_ev_fixtures.jl): the slope-recovery
        # assertion (atol 0.05) is seed-specific and MersenneTwister is not stable
        # across Julia versions. Equivalent to MersenneTwister(11), Tn=220 with
        # x1,x2 random walks and an AR(1) error uu.
        dmr = readdlm(joinpath(@__DIR__, "data", "cointreg_multireg.csv"), ',', Float64)
        Tn = size(dmr, 1)
        x1 = dmr[:, 1]; x2 = dmr[:, 2]; uu = dmr[:, 3]
        yy = 1.0 .+ 0.8 .* x1 .- 0.5 .* x2 .+ uu
        X = hcat(x1, x2)
        for meth in (:fmols, :ccr, :dols)
            m = estimate_cointreg(yy, X; method=meth, bandwidth=3)
            @test length(coef(m)) == 3               # const + 2 slopes
            @test m.varnames == ["const", "x1", "x2"]
            # slope block recovers the true cointegrating vector (the intercept on
            # I(1) regressors converges more slowly, so it is not pinned here)
            @test coef(m)[2:3] ≈ [0.8, -0.5] atol=0.05
            @test all(isfinite, coef(m))
            @test size(m.Omega) == (3, 3)
        end
    end

    @testset "StatsAPI accessors + display" begin
        m = estimate_cointreg(y, x; method=:fmols, bandwidth=3)
        @test S.nobs(m) == n
        @test S.dof(m) == 2
        @test S.dof_residual(m) == n - 2
        @test S.coefnames(m) == ["const", "x"]
        @test length(S.residuals(m)) == n
        @test S.predict(m) ≈ S.fitted(m)
        ci = S.confint(m; level=0.95)
        @test size(ci) == (2, 2)
        @test all(ci[:, 1] .< coef(m) .< ci[:, 2])
        # report / refs render without error
        buf = IOBuffer(); show(buf, m)
        s = String(take!(buf))
        @test occursin("Cointegrating Regression", s)
        @test occursin("FMOLS", s)
        rbuf = IOBuffer(); refs(rbuf, m)
        @test occursin("Phillips", String(take!(rbuf)))
    end

    @testset "Input validation" begin
        @test_throws ArgumentError estimate_cointreg(y, x; method=:bogus)
        @test_throws ArgumentError estimate_cointreg(y, x; trend=:quadratic)
        @test_throws ArgumentError estimate_cointreg(y, x; method=:dols, ic=:hqic)
        @test_throws ArgumentError estimate_cointreg(y, x; method=:dols, dols_se=:bogus)
        @test_throws DimensionMismatch estimate_cointreg(y, x[1:end-1])
        @test_throws ArgumentError estimate_cointreg(y[1:4], x[1:4])
    end
end
