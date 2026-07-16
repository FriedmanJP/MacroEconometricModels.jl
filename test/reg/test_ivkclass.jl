# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-36 (#444) — IV k-class estimators: LIML / Fuller / generic k-class.
#
# Oracle discipline (three layers, no invented numerics):
#   (1) Analytic identities — k-class nests OLS (k=0) and 2SLS (k=1) exactly; κ̂ ≥ 1;
#       just-identified ⇒ κ̂ → 1 and LIML ≡ 2SLS; the (I − k·M_Z) closed form recomputed by an
#       INDEPENDENT route (literal n×n residual-maker) matches the implementation.
#   (2) Golden guard — method=:tsls reproduces the classic 2SLS path (also re-run test_reg.jl).
#   (3) Cross-implementation known values — Python `linearmodels` 7.0 `IVLIML` on a fixed-seed
#       over-identified DGP (n=500, 1 endogenous, 3 excluded instruments). linearmodels is a
#       live, verifiable external LIML/Fuller reference (Kevin Sheppard); the exact reference
#       call and the reproducible DGP are recorded below. (No Stata / rugarch / R LIML package
#       builds in this environment; linearmodels is the named cross-impl oracle used instead.)

using Test
using MacroEconometricModels
using LinearAlgebra, Statistics, Random, Distributions

# ---------------------------------------------------------------------------------------------
# Fixed-seed over-identified DGP (must reproduce the CSV fed to linearmodels bit-for-bit).
#   rng = MersenneTwister(20240736); draws in order z1,z2,z3,w,u,v.
#   X = [const, w, xend]  (linearmodels param order: exog [const, w] then endog);  endogenous = col 3.
#   Z = [const, w, z1, z2, z3]  (5 instruments total ⇒ m = 5, q = 3 excluded).
# ---------------------------------------------------------------------------------------------
function _ev36_dgp()
    rng = MersenneTwister(20240736)
    n = 500
    z1 = randn(rng, n); z2 = randn(rng, n); z3 = randn(rng, n)
    w  = randn(rng, n)
    u  = randn(rng, n)
    v  = randn(rng, n)
    xend = 0.6 .* z1 .+ 0.4 .* z2 .+ 0.3 .* z3 .+ 0.5 .* w .+ 0.7 .* u .+ v
    y = 1.0 .+ 2.0 .* xend .+ 0.5 .* w .+ u
    X = hcat(ones(n), w, xend)
    Z = hcat(ones(n), w, z1, z2, z3)
    return y, X, Z
end

# Independent LIML κ̂ via the literal determinantal problem (different route than the
# Cholesky-whitened implementation): κ̂ = min generalized eigenvalue of (Ȳ'M_X1 Ȳ, Ȳ'M_Z Ȳ).
function _kappa_reference(y, X, Z, endog)
    n = length(y); k = size(X, 2)
    incl = setdiff(1:k, endog)
    X1 = X[:, incl]
    Ybar = hcat(y, X[:, endog])
    M(A) = I - A * (inv(A' * A) * A')
    B = Ybar' * M(X1) * Ybar
    W = Ybar' * M(Z) * Ybar
    minimum(real.(eigvals(B, W)))     # generalized eigenvalues, symmetric pencil
end

# Independent β̂(k) via the literal n×n residual-maker M_Z (different route than
# (1−k)X'X + k·X_hat'X used inside estimate_iv).
function _kclass_beta_reference(y, X, Z, kval)
    n = size(X, 1)
    MZ = I - Z * (inv(Z' * Z) * Z')
    A = X' * (I - kval * MZ) * X
    rhs = X' * (I - kval * MZ) * y
    A \ rhs
end

@testset "IV k-class: LIML / Fuller / generic k-class (EV-36)" begin

    y, X, Z = _ev36_dgp()
    endog = [3]

    # =========================================================================================
    # Layer 3 — cross-implementation known values from Python `linearmodels` 7.0.
    #
    #   import numpy as np; from linearmodels.iv import IVLIML
    #   exog=[const,w]; endog=xend; instr=[z1,z2,z3]      # same fixed-seed DGP above
    #   IVLIML(y, exog, endog, instr).fit(cov_type="unadjusted")     # LIML
    #   IVLIML(y, exog, endog, instr, fuller=1).fit(cov_type="unadjusted")  # Fuller(1)
    #
    # Reported (param order [const, w, xend]):
    #   LIML     κ̂ = 1.003189427028863
    #            β  = [0.9836179550694701, 0.3977326073891603, 2.0709576638448346]
    #            se = [0.03961384348770555, 0.04667028469402005, 0.048231649546826356]  (unadjusted, σ̂²=RSS/n)
    #   Fuller(1) κ = 1.001169225008661  ( = κ̂ − 1/(n−m) = 1.003189… − 1/495 )
    #            β  = [0.9836434156430230, 0.3968588790533110, 2.0725518949422500]
    #            se = [0.03957075624270732, 0.04658778430127276, 0.04807678903579573]  (unadjusted)
    # =========================================================================================
    @testset "LIML vs linearmodels (over-identified)" begin
        m = estimate_iv(y, X, Z; endogenous=endog, method=:liml, cov_type=:ols)
        @test isapprox(m.kappa_hat, 1.003189427028863; atol=1e-4)
        @test isapprox(m.kclass_k, m.kappa_hat; atol=0)            # LIML: k == κ̂ exactly
        @test isapprox(m.beta, [0.9836179550694701, 0.3977326073891603, 2.0709576638448346]; atol=1e-4)
        @test isapprox(sqrt.(diag(m.vcov_mat)),
                       [0.03961384348770555, 0.04667028469402005, 0.048231649546826356]; atol=1e-4)
    end

    @testset "Fuller(1) vs linearmodels (over-identified)" begin
        m = estimate_iv(y, X, Z; endogenous=endog, method=:fuller, fuller_a=1.0, cov_type=:ols)
        # k = κ̂ − a/(n−m), m = size(Z,2) = 5 total instruments.
        @test isapprox(m.kclass_k, m.kappa_hat - 1.0 / (500 - 5); atol=1e-12)
        @test isapprox(m.kclass_k, 1.001169225008661; atol=1e-4)
        @test isapprox(m.beta, [0.9836434156430230, 0.3968588790533110, 2.0725518949422500]; atol=1e-4)
        @test isapprox(sqrt.(diag(m.vcov_mat)),
                       [0.03957075624270732, 0.04658778430127276, 0.04807678903579573]; atol=1e-4)
    end

    # =========================================================================================
    # Layer 1 — analytic identities.
    # =========================================================================================
    @testset "k-class nests OLS (k=0) and 2SLS (k=1)" begin
        # k = 0  ⇒  OLS on X (ignores instruments): matches estimate_reg exactly.
        m0 = estimate_iv(y, X, Z; endogenous=endog, method=:kclass, k=0.0, cov_type=:ols)
        ols = estimate_reg(y, X; cov_type=:ols)
        @test isapprox(m0.beta, ols.beta; atol=1e-9)
        @test isapprox(m0.kclass_k, 0.0; atol=0)
        @test m0.kappa_hat === nothing                            # no κ̂ for generic k-class

        # k = 1  ⇒  2SLS: matches the classic :tsls path.
        m1 = estimate_iv(y, X, Z; endogenous=endog, method=:kclass, k=1.0, cov_type=:ols)
        tsls = estimate_iv(y, X, Z; endogenous=endog, method=:tsls, cov_type=:ols)
        @test isapprox(m1.beta, tsls.beta; atol=1e-8)
    end

    @testset "κ̂ ≥ 1 and independent-route recomputation" begin
        m = estimate_iv(y, X, Z; endogenous=endog, method=:liml)
        @test m.kappa_hat >= 1.0
        # κ̂ from a literal generalized-eigenvalue route (not the whitened one in the impl).
        @test isapprox(m.kappa_hat, _kappa_reference(y, X, Z, endog); atol=1e-8)
        # β̂(κ̂) from the literal n×n M_Z route.
        @test isapprox(m.beta, _kclass_beta_reference(y, X, Z, m.kappa_hat); atol=1e-8)
        # generic k: β̂(k) closed form recomputed independently.
        for kv in (0.3, 0.75, 0.95, 1.05, 1.2)
            mk = estimate_iv(y, X, Z; endogenous=endog, method=:kclass, k=kv)
            @test isapprox(mk.beta, _kclass_beta_reference(y, X, Z, kv); atol=1e-8)
        end
    end

    @testset "just-identified ⇒ κ̂ = 1 and LIML ≡ 2SLS" begin
        # One excluded instrument for one endogenous regressor ⇒ exactly identified.
        Zjust = hcat(ones(500), X[:, 2], Z[:, 3])          # [const, w, z1]
        ml = estimate_iv(y, X, Zjust; endogenous=endog, method=:liml)          # default :hc1
        ts = estimate_iv(y, X, Zjust; endogenous=endog, method=:tsls)          # default :hc1
        @test isapprox(ml.kappa_hat, 1.0; atol=1e-6)
        @test isapprox(ml.beta, ts.beta; atol=1e-8)                            # coefficients
        @test isapprox(sqrt.(diag(ml.vcov_mat)), sqrt.(diag(ts.vcov_mat)); atol=1e-6)  # SEs (hc1)
    end

    # =========================================================================================
    # Structural property — Fuller's k may fall BELOW 1 and must NOT be clamped (Fuller 1977).
    # =========================================================================================
    @testset "Fuller k not clamped below 1" begin
        # On the main DGP, a=4 gives k = κ̂ − 4/(n−m) = 1.003189… − 4/495 ≈ 0.9951 < 1.
        m = estimate_iv(y, X, Z; endogenous=endog, method=:fuller, fuller_a=4.0)
        @test isapprox(m.kclass_k, m.kappa_hat - 4.0 / (500 - 5); atol=1e-12)
        @test m.kclass_k < 1.0                                    # NOT clamped to [1, ∞)
    end

    # =========================================================================================
    # HC-robust covariance path reduces to 2SLS at k=1 (WX-sandwich, golden-consistent).
    # =========================================================================================
    @testset "robust k-class VCV at k=1 matches 2SLS" begin
        for ct in (:hc0, :hc1, :hc2, :hc3)
            mk = estimate_iv(y, X, Z; endogenous=endog, method=:kclass, k=1.0, cov_type=ct)
            ts = estimate_iv(y, X, Z; endogenous=endog, method=:tsls, cov_type=ct)
            @test isapprox(mk.vcov_mat, ts.vcov_mat; atol=1e-8)
        end
    end

    # =========================================================================================
    # Reporting / refs / validation.
    # =========================================================================================
    @testset "report / refs render for k-class" begin
        m = estimate_iv(y, X, Z; endogenous=endog, method=:liml)
        io = IOBuffer()
        @test (show(io, m); true)
        s = String(take!(io))
        @test occursin("κ̂", s) || occursin("LIML", s)            # κ̂ line present
        @test occursin("Anderson LR", s)
        rio = IOBuffer()
        @test (refs(rio, m); true)
        rs = String(take!(rio))
        @test occursin("Fuller", rs) || occursin("Anderson", rs)  # k-class references surfaced
        # Anderson LR statistic value: n·ln(κ̂).
        @test m.kappa_hat > 1.0
    end

    @testset "input validation" begin
        @test_throws ArgumentError estimate_iv(y, X, Z; endogenous=endog, method=:bogus)
        @test_throws ArgumentError estimate_iv(y, X, Z; endogenous=endog, method=:kclass)  # k required
    end

    @testset "Float32 path" begin
        m = estimate_iv(Float32.(y), Float32.(X), Float32.(Z); endogenous=endog, method=:liml)
        @test eltype(m.beta) == Float32
        @test m.kappa_hat >= 1.0f0
    end
end
