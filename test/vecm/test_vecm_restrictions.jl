# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-38 (#446): Johansen LR restriction tests on the VECM cointegrating structure.
#
# ORACLE DISCIPLINE
# -----------------
# R `urca` (blrtest/alrtest/ablrtest) is NOT installed in this environment, so the
# numeric oracle is an INDEPENDENT base-R reimplementation of the Johansen
# reduced-rank restriction LR statistics, run LIVE at authoring time on the same
# `:denmark` data with the identical estimator convention (deterministic=:constant
# => unrestricted constant concentrated out, p=2 => one lagged difference):
#   - β restriction:  closed-form transformed eigenproblem
#                     |λ H'S11H − H'S10 S00⁻¹ S01 H| = 0            (Johansen 1991)
#   - α / weak-exog:  CONDITIONAL partial-system reduced-rank      (Johansen 1995, Ch.8)
#   - known β:        H = b degenerate closed form
# The exact R script lives in the commit message / scratch `oracle_final.R`; its
# output (matched here to 1e-6) was:
#   BETA IBO=-IDE     LR=1.30281619 df=1 p=0.25369945
#   KNOWN [1,-1,0,0]  LR=29.34908089 df=3 p=0.00000189
#   WEAKEXOG LRY      LR=0.31648200 df=1 p=0.57372954
#   ALPHA rows3,4=0   LR=1.54812162 df=2 p=0.46113667
# Each α/weak-exog value was cross-checked against a brute-force Johansen–Juselius
# switching MLE (also base R) to 1e-8, and against this package's own switching
# `test_joint_restriction(H, A)` (below), so the closed-form and iterative routes
# agree independently.
#
# PUBLISHED (not live): Johansen & Juselius (1990, Oxford Bull. Econ. Statist.
# 52(2), 169–210) report weak-exogeneity/long-run restriction LR tests on this very
# dataset; their headline finding is that income (LRY) and prices are NOT weakly
# exogenous while the interest-rate block is closer to it. Their exact statistics
# use a different deterministic/seasonal specification (restricted constant +
# quarterly dummies) so are cited for provenance, not matched numerically.

using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics, Distributions

# ---------------------------------------------------------------------------
# Fixture: Johansen–Juselius (1990) Danish money demand, rank-1 VECM.
# ---------------------------------------------------------------------------
const _DK = load_example(:denmark)
const _DKY = Matrix(_DK[:, ["LRM", "LRY", "IBO", "IDE"]])
_dk_vecm() = estimate_vecm(_DKY, 2; rank=1, deterministic=:constant,
                           varnames=["LRM", "LRY", "IBO", "IDE"])

@testset "VECM restriction tests (EV-38)" begin

    @testset "dataset :denmark loads correctly" begin
        @test nobs(_DK) == 55
        @test Set(["LRM", "LRY", "LPY", "IBO", "IDE"]) == Set(varnames(_DK))
        @test size(_DKY) == (55, 4)
        # spot-check first row against urca's `denmark` (public domain)
        @test _DKY[1, 1] ≈ 11.63255023 atol = 1e-6   # LRM 1974:01
        @test _DKY[1, 2] ≈ 5.903658491 atol = 1e-6   # LRY 1974:01
    end

    m = _dk_vecm()
    @test cointegrating_rank(m) == 1

    # ---- (a) β restriction: IBO = -IDE  ->  H = [I3; row -e3] -------------
    @testset "test_beta_restriction — live R oracle" begin
        H = Float64[1 0 0; 0 1 0; 0 0 1; 0 0 -1]
        res = test_beta_restriction(m, H)
        @test res isa VECMRestrictionTest
        @test res.kind == :beta
        @test res.df == 1                              # r(p-s) = 1*(4-3)
        @test res.lr_stat ≈ 1.30281619 atol = 1e-6
        @test res.pvalue ≈ 0.25369945 atol = 1e-6
        # restricted β lies in span(H): IBO coeff = -IDE coeff
        b = res.beta_restricted
        @test b[3, 1] ≈ -b[4, 1] atol = 1e-8
        @test norm(b - H * (H \ b)) < 1e-8             # β ∈ span(H)
        # switching route agrees (joint with α unrestricted)
        rj = test_joint_restriction(m, H, Matrix{Float64}(I, 4, 4))
        @test rj.lr_stat ≈ res.lr_stat atol = 1e-5
        @test rj.df == res.df
    end

    # ---- (b) α restriction: rows 3,4 (IBO,IDE) weakly exogenous ----------
    @testset "test_alpha_restriction — live R oracle" begin
        A = Float64[1 0; 0 1; 0 0; 0 0]                # α ∈ span(A): rows 3,4 = 0
        res = test_alpha_restriction(m, A)
        @test res.kind == :alpha
        @test res.df == 2                              # r(p-a) = 1*(4-2)
        @test res.lr_stat ≈ 1.54812162 atol = 1e-6
        @test res.pvalue ≈ 0.46113667 atol = 1e-6
        # restricted α has zero rows 3,4
        @test all(abs.(res.restricted_model.alpha[3:4, :]) .< 1e-10)
        # switching route agrees
        rj = test_joint_restriction(m, Matrix{Float64}(I, 4, 4), A)
        @test rj.lr_stat ≈ res.lr_stat atol = 1e-5
    end

    # ---- weak exogeneity of LRY (row 2) ----------------------------------
    @testset "test_weak_exogeneity — live R oracle & df = r·m" begin
        res = test_weak_exogeneity(m, "LRY")
        @test res.kind == :weak_exogeneity
        @test res.df == 1                              # r·m = 1*1
        @test res.lr_stat ≈ 0.31648200 atol = 1e-6
        @test res.pvalue ≈ 0.57372954 atol = 1e-6
        @test all(abs.(res.restricted_model.alpha[2, :]) .< 1e-10)
        # by-index and by-name agree
        @test test_weak_exogeneity(m, 2).lr_stat ≈ res.lr_stat atol = 1e-10
        # df = r·m for a two-variable weak-exogeneity hypothesis
        res2 = test_weak_exogeneity(m, ["IBO", "IDE"])
        @test res2.df == 2                             # r·m = 1*2
        # equals the equivalent α restriction keeping rows 1,2
        ra = test_alpha_restriction(m, Float64[1 0; 0 1; 0 0; 0 0])
        @test res2.lr_stat ≈ ra.lr_stat atol = 1e-8
    end

    # ---- (c) known β: b = [1,-1,0,0] -------------------------------------
    @testset "test_known_beta — live R oracle" begin
        b = reshape(Float64[1, -1, 0, 0], 4, 1)
        res = test_known_beta(m, b)
        @test res.kind == :known_beta
        @test res.df == 3                              # r(p-r) = 1*(4-1)
        @test res.lr_stat ≈ 29.34908089 atol = 1e-6
        @test res.pvalue ≈ 0.00000189 atol = 1e-7
        # restricted model uses β = b (Phillips-normalized, here already normalized)
        @test res.restricted_model.beta ≈ b atol = 1e-10
        # switching cross-check: joint with H=b, α free
        rj = test_joint_restriction(m, b, Matrix{Float64}(I, 4, 4))
        @test rj.lr_stat ≈ res.lr_stat atol = 1e-5
        @test rj.df == 3
    end

    # ---- (d) joint β/α switching -----------------------------------------
    @testset "test_joint_restriction — switching algorithm" begin
        H = Float64[1 0 0; 0 1 0; 0 0 1; 0 0 -1]
        A = Float64[1 0 0; 0 1 0; 0 0 1; 0 0 0]        # IDE weakly exogenous (row 4=0)
        res = test_joint_restriction(m, H, A)
        @test res.kind == :joint
        @test res.converged
        @test res.df == 1 * (4 - 3) + 1 * (4 - 3)      # r(p-s) + r(p-a) = 2
        @test res.lr_stat > 0
        # joint LR ≥ each single-restriction LR (nested)
        @test res.lr_stat ≥ test_beta_restriction(m, H).lr_stat - 1e-6
        @test res.lr_stat ≥ test_weak_exogeneity(m, "IDE").lr_stat - 1e-6
        # restricted β ∈ span(H), restricted α row 4 = 0
        b = res.beta_restricted
        @test norm(b - H * (H \ b)) < 1e-7
        @test all(abs.(res.restricted_model.alpha[4, :]) .< 1e-10)
    end

    # ---- Analytic invariants (no reference numerics) ---------------------
    @testset "analytic invariants" begin
        # non-binding H = I_p  =>  LR ≈ 0, df = 0, p = 1
        res = test_beta_restriction(m, Matrix{Float64}(I, 4, 4))
        @test res.df == 0
        @test res.lr_stat ≈ 0.0 atol = 1e-10
        @test res.pvalue == 1.0
        # non-binding α = I_p  =>  LR ≈ 0, df = 0
        ra = test_alpha_restriction(m, Matrix{Float64}(I, 4, 4))
        @test ra.df == 0
        @test ra.lr_stat ≈ 0.0 atol = 1e-8
        # known β = β̂  =>  LR ≈ 0 (restricted space equals estimated space)
        rk = test_known_beta(m, m.beta)
        @test rk.lr_stat ≈ 0.0 atol = 1e-6
        # LR is always non-negative
        @test test_beta_restriction(m, Float64[1 0 0; 0 1 0; 0 0 1; 0 0 -1]).lr_stat ≥ 0
        # unrestricted eigenvalues match estimate_vecm's Johansen eigenvalues
        @test res.eigenvalues_unrestricted[1] ≈ 0.44821426 atol = 1e-6
    end

    # ---- restricted model feeds irf/fevd/hd ------------------------------
    @testset "restricted model → irf / fevd / hd" begin
        H = Float64[1 0 0; 0 1 0; 0 0 1; 0 0 -1]
        rm = test_beta_restriction(m, H).restricted_model
        @test rm isa VECMModel
        @test cointegrating_rank(rm) == 1
        ir = irf(rm, 8)
        @test size(ir.values, 1) == 8                 # 8 horizons
        @test size(ir.values, 2) == 4                 # 4 variables
        @test all(isfinite, ir.values)
        fv = fevd(rm, 8)
        @test all(isfinite, fv.decomposition)
        hd = historical_decomposition(rm)
        @test hd !== nothing
        # α-restricted model also runs
        ra = test_weak_exogeneity(m, "LRY").restricted_model
        @test all(isfinite, irf(ra, 6).values)
    end

    # ---- display: report / refs / show -----------------------------------
    @testset "report / refs / show render" begin
        res = test_beta_restriction(m, Float64[1 0 0; 0 1 0; 0 0 1; 0 0 -1])
        io = IOBuffer()
        report(io, res)
        s = String(take!(io))
        @test occursin("Restriction Test", s)
        @test occursin("LR", s)
        show(IOBuffer(), res)
        refs(IOBuffer(), res)
        rio = IOBuffer(); refs(rio, res)
        @test occursin("Johansen", String(take!(rio)))
    end

    # ---- input validation -------------------------------------------------
    @testset "argument validation" begin
        @test_throws DimensionMismatch test_beta_restriction(m, Float64[1 0; 0 1])
        @test_throws ArgumentError test_beta_restriction(m, ones(4, 0))   # s < r
        @test_throws DimensionMismatch test_known_beta(m, reshape(Float64[1, 0, 0, 0, 0], 5, 1))
        @test_throws ArgumentError test_weak_exogeneity(m, "NOPE")
    end
end
