# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-09 (#417): Nonlinear ARDL (NARDL) — Shin, Yu & Greenwood-Nimmo (2014).
#
# Oracle strategy (no invented numerics):
#   (1) Analytic self-consistency (primary):
#         • partial-sum identity  x_t = x_1 + x⁺_t + x⁻_t  to machine precision;
#         • the k-enlargement check — an asymmetric regressor contributes TWO
#           columns to the bounds-table k (reusing EV-08 bounds_test without
#           enlarging k is the classic NARDL bug); assert bounds.k == enlarged k;
#         • multiplier convergence m⁺_H → θ⁺, m⁻_H → θ⁻ as H grows (the recursive
#           ARDL difference equation's fixed point IS the long-run coefficient).
#   (2) Size spot-check (seeded single draw, NOT a Monte Carlo): on a SYMMETRIC
#       DGP (θ⁺ = θ⁻) the long-run symmetry Wald does not reject at 5%.
#   (3) Cross-implementation known-value oracle: on the fixed-seed asymmetric DGP
#       below, θ⁺, θ⁻ and the bounds F/t are matched against an independent R
#       `lm` fit of the identical levels design. R is used because the `nardl`
#       CRAN package would not build in this environment; the equivalent
#       hand-rolled R computation is reproduced verbatim in the comment block so
#       the numbers are provenanced, not invented.
#
# R oracle (R 4.5.0), run on the SAME data this test regenerates
# (MersenneTwister(987654321), N=250; CSV = hcat(y,x)):
#   d <- read.csv("nardl.csv", header=FALSE); y<-d$V1; x<-d$V2; N<-length(x)
#   dx <- c(0,diff(x)); xp <- cumsum(pmax(dx,0)); xn <- cumsum(pmin(dx,0))
#   t<-2:N; fit <- lm(y[t] ~ y[t-1] + xp[t] + xp[t-1] + xn[t] + xn[t-1])
#   b<-coef(fit); denom<-1-b[2]
#   (b[3]+b[4])/denom  ->  theta+ = 1.1835065373
#   (b[5]+b[6])/denom  ->  theta- = -0.7151284394
#   # bounds F: joint Wald {b[2]=1, b[3]+b[4]=0, b[5]+b[6]=0}/3 = 27.4905998610
#   # bounds t: (b[2]-1)/se(b[2]) = -8.9676996239

using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics

@testset "NARDL — Nonlinear ARDL (EV-09 #417)" begin

    # -------------------------------------------------------------------------
    # Fixed-seed DGP generators
    # -------------------------------------------------------------------------
    # Asymmetric error-correction DGP with KNOWN long-run θ⁺, θ⁻ and speed φ.
    function _nardl_dgp(seed::Int, N::Int; θp, θn, φ=0.35, ψ=0.25, σ=0.5)
        rng = MersenneTwister(seed)
        x = zeros(N)
        for t in 2:N
            x[t] = x[t-1] + randn(rng)
        end
        dx = diff(x)
        xp = zeros(N); xn = zeros(N)
        for t in 2:N
            xp[t] = xp[t-1] + max(dx[t-1], 0.0)
            xn[t] = xn[t-1] + min(dx[t-1], 0.0)
        end
        y = zeros(N)
        for t in 2:N
            ec = y[t-1] - (θp * xp[t-1] + θn * xn[t-1])
            y[t] = y[t-1] - φ * ec + ψ * dx[t-1] + σ * randn(rng)
        end
        (y, x)
    end

    # =========================================================================
    # (1a) Partial-sum decomposition identity — machine tolerance
    # =========================================================================
    @testset "partial-sum identity" begin
        rng = MersenneTwister(42)
        x = cumsum(randn(rng, 300))
        xp, xn = MacroEconometricModels._partial_sums(x)
        # x_t = x_1 + x⁺_t + x⁻_t  exactly (baseline is the first level)
        @test maximum(abs.(x .- (x[1] .+ xp .+ xn))) < 1e-10
        @test xp[1] == 0.0 && xn[1] == 0.0            # x⁺_0 = x⁻_0 = 0
        @test all(diff(xp) .>= -1e-14)                # x⁺ is non-decreasing
        @test all(diff(xn) .<= 1e-14)                 # x⁻ is non-increasing
        # a hand-built Δx path: [+2, -1, +3, -4]
        xh = [0.0, 2.0, 1.0, 4.0, 0.0]
        php, phn = MacroEconometricModels._partial_sums(xh)
        @test php ≈ [0.0, 2.0, 2.0, 5.0, 5.0]
        @test phn ≈ [0.0, 0.0, -1.0, -1.0, -5.0]
        @test xh ≈ xh[1] .+ php .+ phn
    end

    # =========================================================================
    # (1b) + (3) Long-run θ⁺/θ⁻ recovery and R cross-implementation oracle
    # =========================================================================
    @testset "long-run θ⁺/θ⁻ + R oracle" begin
        y, x = _nardl_dgp(987654321, 250; θp=1.2, θn=-0.7)
        m = estimate_nardl(y, reshape(x, :, 1); asymmetric=:all, p=1, q=1, case=3)

        lr = long_run(m)
        @test lr.varnames == ["x1_POS", "x1_NEG"]

        # --- cross-implementation known values (R lm, see header comment) ---
        @test isapprox(lr.theta[1],  1.1835065373; atol=1e-6)   # θ⁺
        @test isapprox(lr.theta[2], -0.7151284394; atol=1e-6)   # θ⁻
        @test isapprox(m.bounds.fstat, 27.4905998610; atol=1e-5)
        @test isapprox(m.bounds.tstat, -8.9676996239; atol=1e-5)

        # --- recovery of the true DGP long-run coefficients ---
        @test isapprox(lr.theta[1],  1.2; atol=0.15)
        @test isapprox(lr.theta[2], -0.7; atol=0.15)
        @test all(lr.se .> 0)
    end

    # =========================================================================
    # (1c) k-enlargement — the classic NARDL bounds bug
    # =========================================================================
    @testset "bounds k counts partial sums separately" begin
        # two regressors: x1 asymmetric (→ 2 cols), x2 symmetric (→ 1 col) ⇒ k=3
        y, x1 = _nardl_dgp(555, 220; θp=1.0, θn=-0.4)
        rng = MersenneTwister(556)
        x2 = cumsum(randn(rng, 220))
        X = hcat(x1, x2)
        m = estimate_nardl(y, X; asymmetric=[1], p=1, q=1, case=3)
        @test m.k_orig == 2
        @test m.k == 3                                 # 2 (split x1) + 1 (x2)
        @test m.bounds.k == 3                          # PSS table indexed at enlarged k
        @test length(m.ardl.q) == 3
        @test m.enames == ["x1_POS", "x1_NEG", "x2"]

        # all-asymmetric: k doubles
        m2 = estimate_nardl(y, X; asymmetric=:all, p=1, q=1, case=3)
        @test m2.k == 4
        @test m2.bounds.k == 4
    end

    # =========================================================================
    # (1d) Cumulative dynamic multiplier convergence to θ⁺/θ⁻
    # =========================================================================
    @testset "multiplier convergence m±_H → θ±" begin
        y, x = _nardl_dgp(2024, 300; θp=1.5, θn=-0.5)
        m = estimate_nardl(y, reshape(x, :, 1); asymmetric=:all, p=1, q=1, case=3)
        lr = long_run(m)
        mm = dynamic_multipliers(m, 250; bootstrap=false)

        @test mm.horizons == collect(0:250)
        # fixed point of the ARDL difference equation IS the long-run coefficient
        @test isapprox(mm.m_pos[1, end], lr.theta[1]; atol=1e-6)
        @test isapprox(mm.m_neg[1, end], lr.theta[2]; atol=1e-6)
        @test isapprox(mm.theta_pos[1], lr.theta[1]; atol=1e-10)
        @test isapprox(mm.theta_neg[1], lr.theta[2]; atol=1e-10)
        # monotone-ish approach: far-horizon multiplier closer than early one
        @test abs(mm.m_pos[1, end] - lr.theta[1]) <= abs(mm.m_pos[1, 1] - lr.theta[1])
        # asymmetry curve is the difference
        @test mm.m_diff[1, :] ≈ mm.m_pos[1, :] .- mm.m_neg[1, :]
    end

    # =========================================================================
    # (2) Size spot-check — SYMMETRIC DGP does not reject long-run symmetry
    # =========================================================================
    @testset "symmetry Wald: size spot-check (symmetric DGP)" begin
        # NOTE: seeded SINGLE draw — a size spot-check, NOT a Monte Carlo.
        # θ⁺ = θ⁻ ⇒ the long-run symmetry Wald should not reject at 5%.
        y, x = _nardl_dgp(31337, 300; θp=0.9, θn=0.9)
        m = estimate_nardl(y, reshape(x, :, 1); asymmetric=:all, p=1, q=1, case=3)
        st = symmetry_test(m)
        @test st.reg_names == ["x1"]
        @test st.df == 1
        @test st.lr_p_chi2[1] > 0.05                   # do not reject symmetry
        @test st.sr_p_chi2[1] >= 0.0 && st.sr_p_chi2[1] <= 1.0
        # θ⁺ and θ⁻ should be close under the symmetric DGP
        @test isapprox(st.theta_pos[1], st.theta_neg[1]; atol=0.2)
    end

    # =========================================================================
    # symmetry Wald: power — asymmetric DGP DOES reject
    # =========================================================================
    @testset "symmetry Wald: rejects under asymmetry" begin
        y, x = _nardl_dgp(987654321, 250; θp=1.2, θn=-0.7)
        m = estimate_nardl(y, reshape(x, :, 1); asymmetric=:all, p=1, q=1, case=3)
        st = symmetry_test(m)
        @test st.lr_p_chi2[1] < 0.01                   # strongly reject symmetry
        @test st.lr_stat[1] > 0
        # χ²(1) and F(1, n−K) p-values both defined and ordered sensibly
        @test 0.0 <= st.lr_p_f[1] <= 1.0
        @test st.lr_p_f[1] >= st.lr_p_chi2[1] - 1e-8   # F p-value ≥ χ² p-value (finite dof)
    end

    # =========================================================================
    # Bootstrap dynamic-multiplier bands
    # =========================================================================
    @testset "recursive-design residual bootstrap bands" begin
        y, x = _nardl_dgp(2024, 260; θp=1.5, θn=-0.5)
        m = estimate_nardl(y, reshape(x, :, 1); asymmetric=:all, p=1, q=1, case=3)
        rng = MersenneTwister(7)
        mm = dynamic_multipliers(m, 24; bootstrap=true, nreps=500, level=0.90, rng=rng)

        @test mm.nreps == 500
        @test size(mm.m_pos_lo) == (1, 25)
        # bands bracket the point estimate at every horizon
        @test all(mm.m_pos_lo[1, :] .<= mm.m_pos[1, :] .+ 1e-8)
        @test all(mm.m_pos[1, :] .<= mm.m_pos_hi[1, :] .+ 1e-8)
        @test all(mm.m_neg_lo[1, :] .<= mm.m_neg[1, :] .+ 1e-8)
        @test all(mm.m_neg[1, :] .<= mm.m_neg_hi[1, :] .+ 1e-8)
        @test all(mm.m_diff_lo[1, :] .<= mm.m_diff_hi[1, :] .+ 1e-8)
        # bands have positive width away from h=0
        @test mm.m_pos_hi[1, end] > mm.m_pos_lo[1, end]

        # bootstrap=false ⇒ empty bands
        mm0 = dynamic_multipliers(m, 10; bootstrap=false)
        @test mm0.nreps == 0
        @test size(mm0.m_pos_lo) == (0, 0)
    end

    # =========================================================================
    # asymmetric column selection + symmetric passthrough
    # =========================================================================
    @testset "asymmetric selection" begin
        y, x1 = _nardl_dgp(111, 200; θp=1.0, θn=-0.3)
        rng = MersenneTwister(112)
        x2 = cumsum(randn(rng, 200))
        X = hcat(x1, x2)
        m = estimate_nardl(y, X; asymmetric=[2], p=1, q=1, case=3,
                           xnames=["oil", "fx"])
        @test m.asym == [2]
        @test m.enames == ["oil", "fx_POS", "fx_NEG"]   # x1 passes through, x2 split
        st = symmetry_test(m)
        @test st.reg_names == ["fx"]                    # only the asymmetric one is tested
        @test_throws ArgumentError estimate_nardl(y, X; asymmetric=Int[])
        @test_throws ArgumentError estimate_nardl(y, X; asymmetric=[3])
    end

    # =========================================================================
    # Display / refs / plotting smoke tests
    # =========================================================================
    @testset "report / refs / plot_result" begin
        y, x = _nardl_dgp(2024, 260; θp=1.5, θn=-0.5)
        m = estimate_nardl(y, reshape(x, :, 1); asymmetric=:all, p=1, q=1, case=3)
        st = symmetry_test(m)
        rng = MersenneTwister(9)
        mm = dynamic_multipliers(m, 20; bootstrap=true, nreps=200, level=0.90, rng=rng)

        io = IOBuffer()
        show(io, m);   s = String(take!(io)); @test occursin("NARDL", s)
        @test occursin("θ⁺", s) || occursin("_POS", s)
        show(io, st);  @test occursin("Symmetry", String(take!(io)))
        show(io, mm);  @test occursin("multiplier", lowercase(String(take!(io))))

        refs(io, m);   @test occursin("Shin", String(take!(io)))

        p = plot_result(mm; view=:multipliers)
        @test p isa MacroEconometricModels.PlotOutput
        p2 = plot_result(m; view=:multipliers, H=12, bootstrap=false)
        @test p2 isa MacroEconometricModels.PlotOutput
        @test_throws ArgumentError plot_result(mm; view=:bogus)
        @test_throws ArgumentError dynamic_multipliers(m, -1)
    end

    # =========================================================================
    # Integer inputs + vector convenience
    # =========================================================================
    @testset "input conversions" begin
        y, x = _nardl_dgp(77, 180; θp=1.1, θn=-0.6)
        # vector-x convenience path
        m = estimate_nardl(y, x; asymmetric=:all, p=1, q=1, case=3)
        @test m.k == 2
        @test coef(m) == coef(m.ardl)
        @test nobs(m) == m.ardl.n
    end
end
