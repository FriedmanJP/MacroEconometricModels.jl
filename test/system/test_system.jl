# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-35 (#443): systems estimation — SUR (Zellner 1962) + 3SLS (Zellner-Theil 1962).
#
# =============================================================================
# ORACLE DISCIPLINE — no invented numerics. Three layers:
#
# (1) CROSS-IMPLEMENTATION (independent live recomputation + published reference).
#     R's `systemfit` is NOT installed in this environment, so the SUR/3SLS reference
#     estimates were recomputed independently in base-R matrix algebra (FGLS normal
#     equations with the classical Zellner residual-covariance divisor T=20) on the
#     canonical General-Electric / Westinghouse Grunfeld two-equation system, and the
#     one-step SUR estimates additionally match the PUBLISHED Zellner (1962, JASA
#     57(298)) values reproduced in Greene, "Econometric Analysis" and by
#     systemfit(..., method="SUR") in Henningsen & Hamann (2007, J. Stat. Software
#     23(4)). Constants below are the R output (digits=9); assert to 1e-4.
#
#     Exact R oracle (base R, divisor T; equivalent systemfit call annotated):
#       # systemfit( list(GE = I_ge ~ F_ge + C_ge, WH = I_wh ~ F_wh + C_wh),
#       #            method = "SUR", methodResidCovDiag = TRUE,
#       #            residCovWeighted = FALSE )   # divisor T, no dof correction
#       S <- crossprod(cbind(u1,u2)) / T          # Sigma_ij = u_i'u_j / T
#       Si <- solve(S)
#       A  <- solve( blockdiag over i,j of  Si[i,j] * t(X_i) %*% X_j )
#       beta <- A %*% ( blockstack of  sum_j Si[i,j] * t(X_i) %*% y_j )
#
# (2) ANALYTIC PROPERTY (in-env): Kruskal (1968) — identical regressors in every
#     equation ⇒ SUR ≡ equation-by-equation OLS exactly (tight tol). Restricted GLS
#     satisfies R·vec(B)=r to 1e-8. Full-instrument-span ⇒ 3SLS ≡ SUR; exactly
#     identified ⇒ 3SLS ≡ 2SLS.
#
# (3) INDEPENDENT HAND-RECOMPUTATION of the FGLS / McElroy-R² / log-likelihood
#     formulae inside the test (explicit 2×2 Kronecker), matched to the estimator.
# =============================================================================

using Test
using MacroEconometricModels
using MacroEconometricModels: _sigma_from_resids, _gls_normal_eqs, _mcelroy_r2, robust_inv
using LinearAlgebra, Statistics, Random
import StatsAPI

# ---- Grunfeld GE / Westinghouse design from the built-in dataset ----
const _PD = load_example(:grunfeld)
const _GE = group_data(_PD, "General Electric")
const _WH = group_data(_PD, "Westinghouse")
const _T  = 20

# y = invest (col 1); regressors [1, value (col 2), capital (col 3)]
y1 = _GE.data[:, 1]; X1 = hcat(ones(_T), _GE.data[:, 2], _GE.data[:, 3])
y2 = _WH.data[:, 1]; X2 = hcat(ones(_T), _WH.data[:, 2], _WH.data[:, 3])
grunfeld_eqs() = [(y1, X1, ["const", "value", "capital"]),
                  (y2, X2, ["const", "value", "capital"])]

# ---- R oracle constants (base R, divisor T) ----
# Equation-by-equation OLS (verified: matches published Grunfeld GE/WH OLS)
const OLS_GE = [-9.956306455, 0.026551189, 0.151693870]
const OLS_WH = [-0.509390184, 0.052894126, 0.092406492]
# One-step SUR
const SUR_GE  = [-27.7193171237, 0.0383102065, 0.1390362741]
const SUR_WH  = [-1.2519882281, 0.0576297963, 0.0639780665]
const SUR_SE  = [27.032828001, 0.013290114, 0.023035588, 6.956346688, 0.013411012, 0.048900998]
const SUR_SIG = [660.8293885 176.4490614; 176.4490614 88.6616965]
const SUR_DET = 27455.9834
# Iterated SUR (Gaussian MLE)
const ISUR_GE  = [-30.7484629270, 0.0405106939, 0.1359307281]
const ISUR_WH  = [-1.7016098801, 0.0593521099, 0.0557354721]
const ISUR_SIG = [702.2340586 195.3519806; 195.3519806 90.9531072]
# 3SLS just-identified, common Z = [1, C_ge, C_wh]  (equals equation-by-equation 2SLS)
const TSLS_JI_GE = [191.880268488, -0.081777152, 0.172845265]
const TSLS_JI_WH = [-40.763974329, 0.143546791, -0.147728903]
# 3SLS overidentified, common Z = [1, GE_C, WH_C, WH_F]
const TSLS_OI_GE  = [-14.5835893704, 0.0313903694, 0.1397807891]
const TSLS_OI_WH  = [0.8856419477, 0.0529220779, 0.0758980241]
const TWOSLS_OI_GE = [-18.5597708013, 0.0311687815, 0.1507922732]  # eq-by-eq 2SLS (≠ 3SLS)
const TSLS_OI_SE  = [30.985759302, 0.015573021, 0.023105637, 7.356168244, 0.014480672, 0.051046289]

@testset "EV-35 SUR & 3SLS systems estimation" begin

    @testset "Grunfeld dataset sanity" begin
        @test MacroEconometricModels.ngroups(_PD) == 10
        @test MacroEconometricModels.nobs(_PD) == 200
        @test MacroEconometricModels.nvars(_PD) == 3
        @test _GE.data[1, 1] == 33.1     # GE 1935 investment
        @test _WH.data[1, 2] == 191.5    # Westinghouse 1935 firm value
        # OLS on each equation reproduces the published Grunfeld estimates (QR solve;
        # inv(X'X) is ill-conditioned at Grunfeld's raw scale, see the estimator notes).
        @test X1 \ y1 ≈ OLS_GE atol = 1e-6
        @test X2 \ y2 ≈ OLS_WH atol = 1e-6
    end

    @testset "SUR one-step vs R oracle (divisor T)" begin
        m = estimate_sur(grunfeld_eqs(); eqnames = ["GE", "Westinghouse"])
        @test m isa SURModel{Float64}
        @test m.iterations == 1
        @test !m.iterated
        @test m.betas[1] ≈ SUR_GE atol = 1e-4
        @test m.betas[2] ≈ SUR_WH atol = 1e-4
        @test vcat(m.ses...) ≈ SUR_SE atol = 1e-4
        @test m.Sigma ≈ SUR_SIG atol = 1e-3
        @test m.det_sigma ≈ SUR_DET rtol = 1e-5
        @test m.eqnames == ["GE", "Westinghouse"]
        @test m.varnames[1] == ["const", "value", "capital"]
        # residuals from the ORIGINAL X
        @test m.residuals[1] ≈ y1 .- X1 * m.betas[1] atol = 1e-8
    end

    @testset "Kruskal (1968): identical regressors ⇒ SUR ≡ OLS" begin
        # Same regressor matrix in both equations (different y). SUR must collapse to OLS.
        Xc = hcat(ones(_T), _GE.data[:, 2], _GE.data[:, 3])
        m = estimate_sur([(y1, Xc), (y2, Xc)])
        ols1 = Xc \ y1
        ols2 = Xc \ y2
        @test m.betas[1] ≈ ols1 atol = 1e-9
        @test m.betas[2] ≈ ols2 atol = 1e-9
    end

    @testset "Independent FGLS hand-recomputation (explicit 2×2 Kronecker)" begin
        m = estimate_sur(grunfeld_eqs())
        # Rebuild Σ̂ from OLS residuals and solve the stacked GLS system directly (QR solves).
        b1 = X1 \ y1
        b2 = X2 \ y2
        u1 = y1 .- X1 * b1; u2 = y2 .- X2 * b2
        S = [dot(u1, u1) dot(u1, u2); dot(u2, u1) dot(u2, u2)] ./ _T
        Si = inv(S)
        A = [Si[1,1]*(X1'X1) Si[1,2]*(X1'X2); Si[2,1]*(X2'X1) Si[2,2]*(X2'X2)]
        rhs = vcat(Si[1,1]*(X1'y1) + Si[1,2]*(X1'y2), Si[2,1]*(X2'y1) + Si[2,2]*(X2'y2))
        beta_hand = A \ rhs
        se_hand = sqrt.(diag(inv(A)))
        @test vcat(m.betas...) ≈ beta_hand atol = 1e-8
        @test vcat(m.ses...) ≈ se_hand atol = 1e-8
        @test m.Sigma ≈ S atol = 1e-8
    end

    @testset "McElroy (1977) system R² + log-likelihood recomputation" begin
        m = estimate_sur(grunfeld_eqs())
        Si = inv(m.Sigma)
        yt1 = y1 .- mean(y1); yt2 = y2 .- mean(y2)
        r1, r2 = m.residuals[1], m.residuals[2]
        num = Si[1,1]*dot(r1,r1) + 2*Si[1,2]*dot(r1,r2) + Si[2,2]*dot(r2,r2)
        den = Si[1,1]*dot(yt1,yt1) + 2*Si[1,2]*dot(yt1,yt2) + Si[2,2]*dot(yt2,yt2)
        @test m.mcelroy_r2 ≈ 1 - num/den atol = 1e-8
        @test 0 < m.mcelroy_r2 < 1
        ll = -Float64(2 * _T)/2 * (1 + log(2π)) - _T/2 * log(det(m.Sigma))
        @test m.loglik ≈ ll atol = 1e-8
    end

    @testset "Iterated FGLS → Gaussian MLE" begin
        m1 = estimate_sur(grunfeld_eqs())
        mi = estimate_sur(grunfeld_eqs(); iterate = true, tol = 1e-10, maxiter = 500)
        @test mi.iterated
        @test mi.iterations > 1
        @test mi.betas[1] ≈ ISUR_GE atol = 1e-4
        @test mi.betas[2] ≈ ISUR_WH atol = 1e-4
        @test mi.Sigma ≈ ISUR_SIG atol = 1e-3
        # Iterated Σ̂ differs from the one-step Σ̂; coefficients move.
        @test !isapprox(mi.Sigma, m1.Sigma; atol = 1e-2)
        @test !isapprox(mi.betas[1], m1.betas[1]; atol = 1e-2)
    end

    @testset "Linear cross-equation restrictions (restricted GLS)" begin
        # R·vec(B)=r with vec(B) = [c1,v1,k1, c2,v2,k2]; impose value coef equal across eqs.
        R = reshape(Float64[0, 1, 0, 0, -1, 0], 1, 6)
        r = [0.0]
        m = estimate_sur(grunfeld_eqs(); restrict = (R, r))
        @test m.restricted
        beta = vcat(m.betas...)
        @test (R * beta)[1] ≈ 0.0 atol = 1e-8           # restriction satisfied
        @test m.betas[1][2] ≈ m.betas[2][2] atol = 1e-8 # value coefs equal
        # Two-restriction case: also force GE intercept = 0.
        R2 = Float64[0 1 0 0 -1 0; 1 0 0 0 0 0]
        r2 = [0.0, 0.0]
        m2 = estimate_sur(grunfeld_eqs(); restrict = (R2, r2))
        @test R2 * vcat(m2.betas...) ≈ r2 atol = 1e-8
        @test m2.betas[1][1] ≈ 0.0 atol = 1e-8
    end

    @testset "3SLS ≡ SUR when instruments span all regressors" begin
        # Common Z spanning both firms' regressors ⇒ P_Z X_i = X_i ⇒ 3SLS = SUR.
        Zfull = hcat(ones(_T), _GE.data[:, 2], _GE.data[:, 3], _WH.data[:, 2], _WH.data[:, 3])
        msur = estimate_sur(grunfeld_eqs())
        m3 = estimate_3sls(grunfeld_eqs(), Zfull)
        @test m3 isa ThreeSLSModel{Float64}
        @test m3.betas[1] ≈ msur.betas[1] atol = 1e-6
        @test m3.betas[2] ≈ msur.betas[2] atol = 1e-6
    end

    @testset "3SLS exactly identified ≡ 2SLS (R oracle)" begin
        # Common Z = [1, C_ge, C_wh]: each equation just-identified ⇒ 3SLS = eq-by-eq 2SLS.
        Z = hcat(ones(_T), _GE.data[:, 3], _WH.data[:, 3])
        m = estimate_3sls(grunfeld_eqs(), Z; eqnames = ["GE", "Westinghouse"])
        @test m.betas[1] ≈ TSLS_JI_GE atol = 1e-4
        @test m.betas[2] ≈ TSLS_JI_WH atol = 1e-4
        @test m.n_instruments == [3, 3]
        # Cross-check: differs sharply from SUR (genuine instrument projection).
        msur = estimate_sur(grunfeld_eqs())
        @test !isapprox(m.betas[1], msur.betas[1]; atol = 1.0)
    end

    @testset "3SLS overidentified vs R oracle (GLS weighting active)" begin
        # Common Z = [1, GE_C, WH_C, WH_F]: GE equation overidentified ⇒ 3SLS ≠ 2SLS.
        Z = hcat(ones(_T), _GE.data[:, 3], _WH.data[:, 3], _WH.data[:, 2])
        m = estimate_3sls(grunfeld_eqs(), Z)
        @test m.betas[1] ≈ TSLS_OI_GE atol = 1e-4
        @test m.betas[2] ≈ TSLS_OI_WH atol = 1e-4
        @test vcat(m.ses...) ≈ TSLS_OI_SE atol = 1e-4
        @test m.n_instruments == [4, 4]
        # GLS weighting genuinely moves the GE estimate away from equation-by-equation 2SLS.
        @test !isapprox(m.betas[1], TWOSLS_OI_GE; atol = 1e-2)
    end

    @testset "3SLS per-equation instruments" begin
        Zge = hcat(ones(_T), _GE.data[:, 3], _WH.data[:, 3])
        Zwh = hcat(ones(_T), _GE.data[:, 3], _WH.data[:, 3])
        m = estimate_3sls(grunfeld_eqs(), [Zge, Zwh]; instruments = :perequation)
        # Identical per-equation instruments = the common just-identified case.
        mc = estimate_3sls(grunfeld_eqs(), hcat(ones(_T), _GE.data[:, 3], _WH.data[:, 3]))
        @test m.betas[1] ≈ mc.betas[1] atol = 1e-8
        @test m.betas[2] ≈ mc.betas[2] atol = 1e-8
    end

    @testset "report() / refs() render" begin
        m = estimate_sur(grunfeld_eqs(); eqnames = ["GE", "WH"])
        io = IOBuffer(); report(io, m); s = String(take!(io))
        @test occursin("SUR", s)
        @test occursin("GE", s)
        @test occursin("McElroy", s)
        io2 = IOBuffer(); show(io2, m); @test !isempty(String(take!(io2)))
        m3 = estimate_3sls(grunfeld_eqs(), hcat(ones(_T), _GE.data[:, 3], _WH.data[:, 3]))
        io3 = IOBuffer(); report(io3, m3); s3 = String(take!(io3))
        @test occursin("3SLS", s3)
        rs = refs(m); @test occursin("Zellner", rs)
        rs3 = refs(m3); @test occursin("Theil", rs3)
    end

    @testset "Input validation" begin
        @test_throws ArgumentError estimate_sur([])
        # mismatched T across equations
        @test_throws ArgumentError estimate_sur([(y1, X1), (y2[1:10], X2[1:10, :])])
        # length(y) ≠ rows(X)
        @test_throws ArgumentError estimate_sur([(y1[1:10], X1)])
        # bad restriction width
        badR = reshape(Float64[1, 0, 0], 1, 3)
        @test_throws ArgumentError estimate_sur(grunfeld_eqs(); restrict = (badR, [0.0]))
        # 3SLS order condition (fewer instruments than regressors)
        Zsmall = hcat(ones(_T), _GE.data[:, 3])
        @test_throws ArgumentError estimate_3sls(grunfeld_eqs(), Zsmall)
        # per-equation shape mismatch
        @test_throws ArgumentError estimate_3sls(grunfeld_eqs(),
            [hcat(ones(_T), _GE.data[:, 3], _WH.data[:, 3])]; instruments = :perequation)
    end
end
