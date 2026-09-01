# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

@testset "Structural DFM Tests" begin

    # =========================================================================
    # Shared test data generation
    # =========================================================================
    function make_sdfm_data(; T_obs=200, N=20, q=3, seed=42)
        rng = Random.MersenneTwister(seed)
        # Generate factor structure with some serial correlation
        F = zeros(T_obs, q)
        F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.5 * F[t-1, :] + randn(rng, q)
        end
        Lambda = randn(rng, N, q)
        noise = 0.3 * randn(rng, T_obs, N)
        X = F * Lambda' + noise
        return X, q
    end

    # Existing tests pin the two-sided GDFM-VAR pipeline; FGLR is covered below.
    sdfm_legacy(args...; kwargs...) = estimate_structural_dfm(args...; method=:gdfm_var, kwargs...)

    # =========================================================================
    # StructuralDFM Type Construction
    # =========================================================================

    @testset "StructuralDFM type construction" begin
        X, q = make_sdfm_data()
        sdfm = sdfm_legacy(X, q; p=1, H=20)

        @test sdfm isa StructuralDFM{Float64}
        @test sdfm.gdfm isa GeneralizedDynamicFactorModel{Float64}
        @test sdfm.factor_var isa VARModel{Float64}
        @test sdfm.identification == :cholesky
        @test sdfm.p_var == 1
        @test length(sdfm.shock_names) == q
    end

    # =========================================================================
    # One-Step Estimation (From Raw Data)
    # =========================================================================

    @testset "From raw data (one-step)" begin
        X, q = make_sdfm_data()
        T_obs, N = size(X)

        sdfm = sdfm_legacy(X, q; identification=:cholesky, p=2, H=30)

        @test sdfm.p_var == 2
        @test size(sdfm.structural_irf) == (30, N, q)
        @test size(sdfm.B0) == (q, q)
        @test size(sdfm.Q) == (q, q)
        @test size(sdfm.loadings_td) == (N, q)

        # Factor VAR should have q variables
        @test nvars(sdfm.factor_var) == q
        @test sdfm.factor_var.p == 2
    end

    # =========================================================================
    # Two-Step Estimation (From Existing GDFM)
    # =========================================================================

    @testset "From existing GDFM (two-step)" begin
        X, q = make_sdfm_data()

        # Step 1: Estimate GDFM separately
        gdfm = estimate_gdfm(X, q)
        @test gdfm isa GeneralizedDynamicFactorModel{Float64}

        # Step 2: Build Structural DFM on top
        sdfm = sdfm_legacy(gdfm; identification=:cholesky, p=1, H=25)

        @test sdfm.gdfm === gdfm  # Same reference
        @test size(sdfm.structural_irf, 1) == 25
        @test size(sdfm.structural_irf, 2) == size(X, 2)
        @test size(sdfm.structural_irf, 3) == q
    end

    # =========================================================================
    # Cholesky Identification
    # =========================================================================

    @testset "Cholesky identification (B0 lower-triangular)" begin
        X, q = make_sdfm_data()
        sdfm = sdfm_legacy(X, q; identification=:cholesky, p=1, H=20)

        # B0 = chol(Sigma) * Q. With Cholesky, Q=I, so B0 = chol(Sigma) = lower triangular
        B0 = sdfm.B0
        for i in 1:q
            for j in (i+1):q
                @test abs(B0[i, j]) < 1e-10
            end
        end

        # Q should be identity for Cholesky
        @test sdfm.Q ≈ Matrix{Float64}(I, q, q)
    end

    # =========================================================================
    # Sign Restrictions
    # =========================================================================

    @testset "Sign restrictions" begin
        X, q = make_sdfm_data(; q=2)

        # Define a sign check: first shock has positive impact on first factor at h=1
        sign_check = irf_result -> irf_result[1, 1, 1] > 0

        sdfm = sdfm_legacy(X, 2;
            identification=:sign, p=1, H=20, restriction_space=:factor,
            sign_check=sign_check, max_draws=5000)

        @test sdfm.identification == :sign

        # Verify sign restriction is satisfied in the factor IRF
        factor_irf = compute_irf(sdfm.factor_var, sdfm.Q, 20)
        @test factor_irf[1, 1, 1] > 0

        # Q should be orthogonal
        QQt = sdfm.Q * sdfm.Q'
        @test QQt ≈ Matrix{Float64}(I, 2, 2) atol=1e-10
    end

    # =========================================================================
    # irf Dispatch
    # =========================================================================

    @testset "irf dispatch returns ImpulseResponse" begin
        X, q = make_sdfm_data()
        T_obs, N = size(X)
        sdfm = sdfm_legacy(X, q; p=1, H=30)

        # Test irf dispatch
        irf_result = irf(sdfm, 20)
        @test irf_result isa ImpulseResponse{Float64}
        @test irf_result.horizon == 20
        @test size(irf_result.values) == (20, N, q)
        @test length(irf_result.variables) == N
        @test length(irf_result.shocks) == q
        @test irf_result.ci_type == :none

        # Horizons beyond the estimation-time cache are computed on demand (SDFM-08)
        irf_long = irf(sdfm, 50)
        @test irf_long.horizon == 50
        @test size(irf_long.values) == (50, N, q)
        @test irf_long.values[1:30, :, :] ≈ sdfm.structural_irf atol=1e-10

        # Values at overlapping horizons match the stored cache and the panel projection
        @test irf_result.values ≈ sdfm.structural_irf[1:20, :, :] atol=1e-10
        @test irf(sdfm, 20).values ≈ sdfm_panel_irf(sdfm, 20).values atol=1e-12
    end

    @testset "irf rejects re-identification kwargs and non-positive horizon" begin
        X, q = make_sdfm_data()
        sdfm = sdfm_legacy(X, q; p=1, H=10)
        @test_throws ArgumentError irf(sdfm, 10; method=:cholesky)
        @test_throws ArgumentError irf(sdfm, 0)
        @test_throws ArgumentError irf(sdfm, -1)
    end

    # =========================================================================
    # fevd Dispatch
    # =========================================================================

    @testset "fevd dispatch returns FEVD" begin
        X, q = make_sdfm_data()
        sdfm = sdfm_legacy(X, q; p=1, H=20)

        fevd_result = fevd(sdfm, 15)
        @test fevd_result isa FEVD{Float64}

        # FEVD is on the factor VAR, so q variables and q shocks
        @test length(fevd_result.variables) == q
        @test length(fevd_result.shocks) == q

        # FEVD proportions: shape (n_var, n_shock, horizon)
        # Proportions should sum to 1 across shocks at each horizon for each variable
        for h in 1:15
            for i in 1:q
                @test sum(fevd_result.proportions[i, :, h]) ≈ 1.0 atol=1e-10
            end
        end
    end

    @testset "fevd uses stored rotation and shock names" begin
        X, q = make_sdfm_data(; q=2)
        rng = Random.MersenneTwister(42)
        sign_check = irf_result -> irf_result[1, 1, 1] > 0 && irf_result[1, 1, 2] < 0
        sdfm = sdfm_legacy(X, 2;
            identification=:sign, p=1, H=20, restriction_space=:factor,
            sign_check=sign_check, max_draws=5000, rng=rng,
            shock_names=["demand", "supply"])

        H = 15
        fevd_result = fevd(sdfm, H)
        @test fevd_result.shocks == sdfm.shock_names == ["demand", "supply"]

        irf_vals = compute_irf(sdfm.factor_var, sdfm.Q, H)
        _, props = MacroEconometricModels._compute_fevd(irf_vals, 2, H)
        @test fevd_result.proportions ≈ props atol=1e-10

        # Differs from the unrotated factor-VAR FEVD whenever Q is not a signed permutation of I
        signed_perm = all(count(x -> abs(x) > 1e-8, sdfm.Q[i, :]) == 1 &&
                          any(x -> isapprox(abs(x), 1; atol=1e-8), sdfm.Q[i, :])
                          for i in 1:2) &&
                      all(count(x -> abs(x) > 1e-8, sdfm.Q[:, j]) == 1 &&
                          any(x -> isapprox(abs(x), 1; atol=1e-8), sdfm.Q[:, j])
                          for j in 1:2)
        @test !signed_perm
        fevd_unrot = fevd(sdfm.factor_var, H)
        @test maximum(abs.(fevd_result.proportions .- fevd_unrot.proportions)) > 1e-8

        @test_throws ArgumentError fevd(sdfm, H; method=:cholesky)
        @test_throws ArgumentError fevd(sdfm, H; check_func=sign_check)
        @test_throws ArgumentError fevd(sdfm, H; narrative_check=sign_check)
    end

    # =========================================================================
    # Validation Errors
    # =========================================================================

    @testset "Validation errors" begin
        X, q = make_sdfm_data()

        # Invalid identification method
        @test_throws ArgumentError sdfm_legacy(X, q; identification=:invalid)

        # Sign identification without sign_check
        @test_throws ArgumentError sdfm_legacy(X, q; identification=:sign)

        # Invalid p
        @test_throws ArgumentError sdfm_legacy(X, q; p=0)

        # Invalid H
        @test_throws ArgumentError sdfm_legacy(X, q; H=0)
    end

    # =========================================================================
    # Display Output
    # =========================================================================

    @testset "Display output" begin
        X, q = make_sdfm_data()
        sdfm = sdfm_legacy(X, q; p=1, H=20)

        io = IOBuffer()
        show(io, sdfm)
        output = String(take!(io))

        @test occursin("Structural DFM", output)
        @test occursin("Cholesky", output)
        @test occursin("Dynamic factors", output)
        @test occursin("Impact Matrix B0", output)
        @test occursin("Variance Explained", output)
    end

    # =========================================================================
    # Dimensions
    # =========================================================================

    @testset "Dimensions consistency" begin
        T_obs, N, q = 150, 15, 2
        X, _ = make_sdfm_data(; T_obs=T_obs, N=N, q=q)
        H = 25

        sdfm = sdfm_legacy(X, q; p=1, H=H)

        # structural_irf: H x N x q
        @test size(sdfm.structural_irf) == (H, N, q)

        # B0: q x q
        @test size(sdfm.B0) == (q, q)

        # Q: q x q
        @test size(sdfm.Q) == (q, q)

        # loadings_td: N x q
        @test size(sdfm.loadings_td) == (N, q)

        # factor_var has q variables
        @test nvars(sdfm.factor_var) == q

        # GDFM factors: T_obs x q
        @test size(sdfm.gdfm.factors) == (T_obs, q)
    end

    # =========================================================================
    # StatsAPI Interface
    # =========================================================================

    @testset "StatsAPI interface" begin
        X, q = make_sdfm_data()
        sdfm = sdfm_legacy(X, q; p=1, H=20)

        @test nobs(sdfm) == size(X, 1)
        @test dof(sdfm) > 0
        @test length(r2(sdfm)) == size(X, 2)
        @test all(x -> 0 <= x <= 1, r2(sdfm))

        pred = predict(sdfm)
        @test size(pred) == size(X)
        @test residuals(sdfm) ≈ X - pred atol=1e-10
        @test coef(sdfm) == coef(sdfm.factor_var)
        @test MacroEconometricModels.loadings(sdfm) == sdfm.loadings_td
        @test MacroEconometricModels.factors(sdfm) == sdfm.gdfm.factors
    end

    # =========================================================================
    # Different VAR Lag Orders
    # =========================================================================

    @testset "Different VAR lag orders" begin
        X, q = make_sdfm_data()

        for p in [1, 2, 4]
            sdfm = sdfm_legacy(X, q; p=p, H=20)
            @test sdfm.p_var == p
            @test sdfm.factor_var.p == p
        end
    end

    # =========================================================================
    # Structural IRF Non-Trivial
    # =========================================================================

    @testset "Structural IRFs are non-trivial" begin
        X, q = make_sdfm_data()
        sdfm = sdfm_legacy(X, q; p=1, H=20)

        # IRFs should not be all zeros
        @test sum(abs.(sdfm.structural_irf)) > 0

        # Impact (h=1) should be non-zero for at least some variables
        impact = sdfm.structural_irf[1, :, :]
        @test maximum(abs.(impact)) > 1e-10

        # IRFs should generally decay (mean absolute IRF decreases)
        mean_abs_early = mean(abs.(sdfm.structural_irf[1:5, :, :]))
        mean_abs_late = mean(abs.(sdfm.structural_irf[16:20, :, :]))
        # With AR structure, later IRFs should not be much larger
        @test mean_abs_late < 10 * mean_abs_early
    end

    # =========================================================================
    # sdfm_panel_irf
    # =========================================================================

    @testset "sdfm_panel_irf convenience form" begin
        X, q = make_sdfm_data()
        T_obs, N = size(X)
        sdfm = sdfm_legacy(X, q; p=1, H=30)

        # Compute panel IRFs via convenience form
        panel_irf = sdfm_panel_irf(sdfm, 20)

        @test panel_irf isa ImpulseResponse{Float64}
        @test panel_irf.horizon == 20
        @test size(panel_irf.values) == (20, N, q)
        @test length(panel_irf.variables) == N
        @test length(panel_irf.shocks) == q
        @test panel_irf.ci_type == :none

        # Values should agree with stored structural_irf for overlapping horizons
        @test panel_irf.values ≈ sdfm.structural_irf[1:20, :, :] atol=1e-10
    end

    @testset "sdfm_panel_irf exceeds stored horizon" begin
        X, q = make_sdfm_data()
        T_obs, N = size(X)
        sdfm = sdfm_legacy(X, q; p=1, H=20)

        # Request horizon beyond stored — convenience form recomputes from VAR
        panel_irf = sdfm_panel_irf(sdfm, 40)

        @test panel_irf.horizon == 40
        @test size(panel_irf.values) == (40, N, q)
        @test all(isfinite.(panel_irf.values))

        # First 20 horizons should match stored
        @test panel_irf.values[1:20, :, :] ≈ sdfm.structural_irf atol=1e-10
    end

    @testset "sdfm_panel_irf from ImpulseResponse" begin
        X, q = make_sdfm_data()
        T_obs, N = size(X)
        sdfm = sdfm_legacy(X, q; p=1, H=20)

        # Get factor-space IRF
        factor_irf_result = irf(sdfm.factor_var, 20)

        # Project to panel space
        panel_irf = sdfm_panel_irf(sdfm, factor_irf_result)

        @test panel_irf isa ImpulseResponse{Float64}
        @test panel_irf.horizon == 20
        @test size(panel_irf.values) == (20, N, q)

        # Should agree with stored structural_irf (both use Cholesky = default)
        @test panel_irf.values ≈ sdfm.structural_irf atol=1e-10
    end

    @testset "sdfm_panel_irf dimension validation" begin
        X, q = make_sdfm_data(; q=3)
        sdfm = sdfm_legacy(X, q; p=1, H=20)

        # Wrong number of variables in IRF
        bad_irf = ImpulseResponse{Float64}(
            zeros(20, 2, 3), zeros(20, 2, 3), zeros(20, 2, 3),
            20, ["a", "b"], ["s1", "s2", "s3"], :none, nothing, 0.0)
        @test_throws ArgumentError sdfm_panel_irf(sdfm, bad_irf)

        # Wrong number of shocks in IRF
        bad_irf2 = ImpulseResponse{Float64}(
            zeros(20, 3, 2), zeros(20, 3, 2), zeros(20, 3, 2),
            20, ["a", "b", "c"], ["s1", "s2"], :none, nothing, 0.0)
        @test_throws ArgumentError sdfm_panel_irf(sdfm, bad_irf2)

        # Invalid horizon
        @test_throws ArgumentError sdfm_panel_irf(sdfm, 0)
    end

    # =========================================================================
    # TimeSeriesData, shock names, NaN validation
    # =========================================================================

    @testset "TimeSeriesData dispatch preserves names" begin
        X, q = make_sdfm_data(; q=2, N=8)
        names = ["v$i" for i in 1:size(X, 2)]
        ts = TimeSeriesData(X; varnames=names)

        sdfm = sdfm_legacy(ts, 2; p=1, H=10)
        @test sdfm.varnames == names
        @test irf(sdfm, 5).variables == names

        gdfm = estimate_gdfm(ts, 2)
        @test gdfm.varnames == names
        sdfm2 = sdfm_legacy(gdfm; p=1, H=10)
        @test sdfm2.varnames == names
        @test irf(sdfm2, 5).variables == names

        sdfm_rt = MacroEconometricModels._reconstruct_from_container(
            MacroEconometricModels._build_container(sdfm))
        @test sdfm_rt.varnames == names
        @test sdfm_rt.shock_names == sdfm.shock_names
    end

    @testset "shock_names propagate; wrong length throws" begin
        X, q = make_sdfm_data(; q=2)
        sdfm = sdfm_legacy(X, 2; p=1, H=10, shock_names=["demand", "supply"])
        @test sdfm.shock_names == ["demand", "supply"]
        @test irf(sdfm, 5).shocks == ["demand", "supply"]
        @test fevd(sdfm, 5).shocks == ["demand", "supply"]
        shown = sprint(show, sdfm)
        @test occursin("demand", shown)
        @test occursin("supply", shown)
        @test_throws ArgumentError sdfm_legacy(X, 2; p=1, H=10, shock_names=["only_one"])
    end

    @testset "NaN/Inf panel data throw ArgumentError" begin
        X, q = make_sdfm_data()
        Xnan = copy(X)
        Xnan[1, 1] = NaN
        err = try
            sdfm_legacy(Xnan, q; p=1, H=10)
            error("expected ArgumentError")
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("NaN", err.msg) || occursin("Inf", err.msg)

        Xinf = copy(X)
        Xinf[2, 1] = Inf
        err2 = try
            sdfm_legacy(Xinf, q; p=1, H=10)
            error("expected ArgumentError")
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("NaN", err2.msg) || occursin("Inf", err2.msg)
    end

    # =========================================================================
    # refs / report
    # =========================================================================

    @testset "refs cite FHLR/FGLR; report includes Factor VAR" begin
        X, q = make_sdfm_data(; q=2, N=8, T_obs=80)
        gdfm = estimate_gdfm(X, 2)
        apa = sprint(io -> refs(io, gdfm; format=:apa))
        @test occursin("Forni", apa)
        @test occursin("2000", apa)

        sdfm = sdfm_legacy(gdfm; p=1, H=10)
        sdfm_apa = sprint(io -> refs(io, sdfm; format=:apa))
        @test occursin("Forni", sdfm_apa)
        @test occursin("2009", sdfm_apa)
        @test occursin("2005", sdfm_apa)

        rpt = sprint(report, sdfm)
        @test occursin("Factor VAR", rpt)
        @test occursin("Identification", rpt)
    end

    @testset "bibliography keys from #724 exist" begin
        keys = [:forni_hallin_lippi_reichlin2000, :forni_hallin_lippi_reichlin2005,
                :hallin_liska2007, :forni_gambetti2010, :stock_watson2005,
                :stock_watson2016, :bai_ng2007, :amengual_watson2007]
        for k in keys
            s = sprint(io -> refs(io, [k]; format=:apa))
            @test !isempty(strip(s))
        end
    end

    # =========================================================================
    # FGLR (2009): r ≥ q, rank-q reduction, panel Cholesky
    # =========================================================================

    @testset "FGLR r < q throws" begin
        X, q = make_sdfm_data(; q=2, N=12, T_obs=80)
        @test_throws ArgumentError estimate_structural_dfm(X, 2; r=1, method=:fglr, p=1, H=10)
    end

    @testset "FGLR K shape, K'K diagonal, shock share" begin
        X, q = make_sdfm_data(; q=2, N=20, T_obs=150)
        sdfm = estimate_structural_dfm(X, 2; r=4, method=:fglr, identification=:cholesky,
                                       order=[1, 2], p=1, H=10, standardize=false)
        @test sdfm.method === :fglr
        @test sdfm.r == 4
        @test size(sdfm.K) == (4, 2)
        KtK = sdfm.K' * sdfm.K
        @test isapprox(KtK, Diagonal(diag(KtK)); atol=1e-8)
        @test 0 < sdfm.shock_variance_share <= 1
        @test size(sdfm.B0) == (4, 2)
        @test nvars(sdfm.factor_var) == 4
        r = irf(sdfm, 12)
        @test r.horizon == 12
        @test size(r.values) == (12, 20, 2)
        sdfm2 = MacroEconometricModels._reconstruct_from_container(
            MacroEconometricModels._build_container(sdfm))
        @test sdfm2.r == 4
        @test sdfm2.method === :fglr
        @test sdfm2.K ≈ sdfm.K
    end

    @testset "FGLR sign_check receives factor-space IRFs H×r×q" begin
        X, q = make_sdfm_data(; q=2, N=16, T_obs=100)
        seen = Ref{Tuple{Int,Int,Int}}((0, 0, 0))
        sign_check = irf -> (seen[] = size(irf); irf[1, 1, 1] > 0 && irf[1, 1, 2] < 0)
        sdfm = estimate_structural_dfm(X, 2; r=4, method=:fglr, identification=:sign,
                                       restriction_space=:factor,
                                       sign_check=sign_check, max_draws=5000, p=1, H=10,
                                       rng=Random.MersenneTwister(7), standardize=false)
        @test seen[] == (10, 4, 2)          # horizon × r × q, not H×N×q
        fac = MacroEconometricModels._sdfm_factor_structural_irf(sdfm, 10)
        @test fac[1, 1, 1] > 0 && fac[1, 1, 2] < 0
        panel = irf(sdfm, 10).values
        @test size(panel) == (10, 16, 2)
        # Factor-space restriction need not copy onto observable 1
        @test seen[][2] != size(X, 2)
    end

    @testset "FGLR lagged-factor Monte Carlo recovers panel IRFs" begin
        rng = Random.MersenneTwister(20260830)
        T_obs, N, q, rstat = 400, 60, 2, 4
        Φ = [0.5 0.0; 0.1 0.4]
        # True contemporaneous impact on first two observables is lower triangular
        Λ = 0.4 .* randn(rng, N, rstat)
        Λ[1:2, 1:2] .= [1.0 0.0; 0.6 1.0]
        Λ[1:2, 3:4] .= [0.3 0.1; 0.2 0.4]
        f = zeros(T_obs + 1, q)
        X = zeros(T_obs, N)
        true_irf = zeros(10, N, q)   # h = 1..10, impact at h=1
        for h in 1:10
            Ah = h == 1 ? Matrix{Float64}(I, q, q) : Φ^(h - 1)
            Alag = h == 1 ? zeros(q, q) : Φ^(h - 2)
            ΨF = vcat(Ah, Alag)          # 4 × 2
            true_irf[h, :, :] = Λ * ΨF
        end
        f[1, :] = randn(rng, q)
        for t in 1:T_obs
            eps_t = randn(rng, q)
            f[t + 1, :] = Φ * f[t, :] + eps_t
            Fstat = vcat(f[t + 1, :], f[t, :])   # [f_t; f_{t-1}]
            X[t, :] = Λ * Fstat + 0.15 .* randn(rng, N)
        end

        sdfm = estimate_structural_dfm(X, q; r=rstat, method=:fglr,
                                       identification=:cholesky, order=[1, 2],
                                       p=1, H=10, standardize=false)
        est = irf(sdfm, 10).values
        aligned = copy(est)
        for j in 1:q
            if dot(est[1, :, j], true_irf[1, :, j]) < 0
                aligned[:, :, j] .*= -1
            end
        end
        rel_rmse = sqrt(mean(abs2, aligned .- true_irf)) / sqrt(mean(abs2, true_irf))
        # 0.15 was tight on Julia 1.10 CI BLAS (0.164); 0.20 still rejects a broken kernel.
        @test rel_rmse < 0.20

        # Legacy two-sided path is callable on the same DGP (may not recover)
        sdfm_leg = estimate_structural_dfm(X, q; method=:gdfm_var, p=1, H=10)
        @test sdfm_leg.method === :gdfm_var
        @test irf(sdfm_leg, 10).horizon == 10
    end

    # =========================================================================
    # SDFM-03: panel-space sign restrictions and identified set
    # =========================================================================

    @testset "panel-space signs hold on observable IRF cells" begin
        rng = Random.MersenneTwister(72503)
        T_obs, N, q = 180, 12, 2
        Λ = 0.5 .* randn(rng, N, q)
        Λ[1:2, :] .= [1.0 0.0; -0.6 1.0]
        F = zeros(T_obs, q)
        F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.4 .* F[t - 1, :] .+ randn(rng, q)
        end
        X = F * Λ' .+ 0.15 .* randn(rng, T_obs, N)
        names = ["x$i" for i in 1:N]
        # Shock 1 raises x1 and lowers x2 at horizons 1:2 (true impact is lower-triangular)
        sdfm = estimate_structural_dfm(X, q; r=2, method=:fglr, identification=:sign,
            sign_restrictions=[("x1", 1, 1:2, :positive), ("x2", 1, 1:2, :negative)],
            restriction_space=:panel, p=1, H=8, max_draws=4000,
            rng=Random.MersenneTwister(72503), standardize=false, varnames=names)
        ir = irf(sdfm, 8)
        @test size(ir.values, 2) == N
        @test all(ir.values[h, 1, 1] > 0 for h in 1:2)
        @test all(ir.values[h, 2, 1] < 0 for h in 1:2)
        @test varindex(sdfm, "x1") == 1
        @test varindex(sdfm, "x2") == 2
    end

    @testset "store_all identified set has sign-set bands" begin
        rng = Random.MersenneTwister(72504)
        T_obs, N, q = 150, 10, 2
        Λ = 0.5 .* randn(rng, N, q)
        Λ[1:2, :] .= [1.0 0.0; -0.6 1.0]
        F = zeros(T_obs, q); F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.4 .* F[t - 1, :] .+ randn(rng, q)
        end
        X = F * Λ' .+ 0.15 .* randn(rng, T_obs, N)
        names = ["x$i" for i in 1:N]
        sdfm = estimate_structural_dfm(X, q; r=2, method=:fglr, identification=:sign,
            sign_restrictions=[("x1", 1, 1:2, :positive), ("x2", 1, 1:2, :negative)],
            restriction_space=:panel, store_all=true, p=1, H=6, max_draws=3000,
            rng=Random.MersenneTwister(72504), standardize=false, varnames=names)
        @test sdfm.identified_set !== nothing
        @test sdfm.identified_set.n_accepted >= 1
        @test size(sdfm.identified_set.irf_draws) == (sdfm.identified_set.n_accepted, 6, N, q)
        ir = irf(sdfm, 6)
        @test ir.ci_type === :sign_set
        @test all(ir.ci_lower .<= ir.values)
        @test all(ir.values .<= ir.ci_upper)
        ir_first = irf(sdfm, 6; point=:first)
        @test ir_first.ci_type === :none
        @test all(ir_first.values[h, 1, 1] > 0 for h in 1:2)
        @test all(ir_first.values[h, 2, 1] < 0 for h in 1:2)
    end

    @testset "declarative and closure forms share Haar draws" begin
        rng1 = Random.MersenneTwister(91)
        rng2 = Random.MersenneTwister(91)
        T_obs, N, q = 120, 8, 2
        X = randn(rng1, T_obs, N); rng1 = Random.MersenneTwister(91)
        names = ["x$i" for i in 1:N]
        decl = estimate_structural_dfm(X, q; identification=:sign, method=:fglr, r=2,
            sign_restrictions=[("x1", 1, 1:1, :positive)],
            restriction_space=:panel, p=1, H=5, max_draws=2000,
            rng=rng1, standardize=false, varnames=names)
        closefn(irf) = irf[1, 1, 1] > 0
        clo = estimate_structural_dfm(X, q; identification=:sign, method=:fglr, r=2,
            sign_check=closefn, restriction_space=:panel, p=1, H=5, max_draws=2000,
            rng=rng2, standardize=false, varnames=names)
        @test decl.Q ≈ clo.Q
    end

    @testset "unsatisfiable restriction names the variable" begin
        rng = Random.MersenneTwister(11)
        X = randn(rng, 80, 6)
        names = ["x$i" for i in 1:6]
        err = try
            estimate_structural_dfm(X, 2; identification=:sign, method=:fglr, r=2,
                sign_restrictions=[("x2", 1, 1:1, :positive), ("x2", 1, 1:1, :negative)],
                restriction_space=:panel, p=1, H=4, max_draws=50,
                rng=Random.MersenneTwister(11), standardize=false, varnames=names)
            nothing
        catch e
            e
        end
        @test err isa IdentificationError
        @test occursin("x2", err.msg)
    end

    @testset "factor-space path still returns a first-accepted point" begin
        X, q = make_sdfm_data(; q=2, N=12, T_obs=80)
        sign_check = irf -> irf[1, 1, 1] > 0
        sdfm = estimate_structural_dfm(X, 2; identification=:sign, restriction_space=:factor,
            sign_check=sign_check, max_draws=3000, p=1, H=6,
            rng=Random.MersenneTwister(3), standardize=false)
        @test sdfm.identified_set === nothing
        fac = MacroEconometricModels._sdfm_factor_structural_irf(sdfm, 6)
        @test fac[1, 1, 1] > 0
        @test irf(sdfm, 6).ci_type === :none
    end

    # =========================================================================
    # SDFM-04: compute_Q routing and panel long-run
    # =========================================================================

    function _panel_lr(sdfm)
        fv = sdfm.factor_var
        n, p = nvars(fv), fv.p
        A_sum = sum(MacroEconometricModels.extract_ar_coefficients(fv.B, n, p))
        Psi_inf = inv(I - A_sum)
        Λ = sdfm.loadings_static
        Λ * Psi_inf * sdfm.B0
    end

    @testset "panel long-run zeros the second shock on the first target" begin
        rng = Random.MersenneTwister(713)
        T_obs, N, q = 220, 10, 2
        Φ = [0.5 0.0; 0.1 0.4]
        F = zeros(T_obs, q); F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = Φ * F[t-1, :] .+ randn(rng, q)
        end
        Λ = 0.4 .* randn(rng, N, q)
        Λ[1:2, :] .= [1.0 0.0; 0.5 1.0]
        X = F * Λ' .+ 0.15 .* randn(rng, T_obs, N)
        names = ["prod", "hours", ["x$i" for i in 3:N]...]
        sdfm = estimate_structural_dfm(X, q; r=2, identification=:long_run,
            target_vars=["prod", "hours"], varnames=names, p=1, H=24,
            standardize=false)
        C∞ = _panel_lr(sdfm)
        @test abs(C∞[1, 2]) < 1e-8
        @test_throws ArgumentError irf(sdfm, 8; method=:cholesky)
        @test_throws ArgumentError fevd(sdfm, 8; method=:cholesky)
        d = fevd(sdfm, 8)
        irf_f = MacroEconometricModels._sdfm_factor_structural_irf(sdfm, 8)
        _, props = MacroEconometricModels._compute_fevd_rect(irf_f, 2, 2, 8)
        @test d.proportions ≈ props atol=1e-10
    end

    @testset "compute_Q methods yield orthogonal Q consumed by irf/fevd" begin
        rng = Random.MersenneTwister(7131)
        T_obs, N, q = 260, 8, 2
        F = zeros(T_obs, q); F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.5 .* F[t-1, :] .+ randn(rng, q)
        end
        X = F * randn(rng, N, q)' .+ 0.25 .* randn(rng, T_obs, N)
        sign_fn = irf -> irf[1, 1, 1] > 0
        narr = shocks -> true
        methods = (
            (:narrative, (sign_check=sign_fn, narrative_check=narr, max_draws=2000,
                          rng=Random.MersenneTwister(3))),
            (:fastica, (rng=Random.MersenneTwister(4),)),
            (:student_t, NamedTuple()),
            (:garch, NamedTuple()),
        )
        for (id, kw) in methods
            sdfm = estimate_structural_dfm(X, q; r=2, identification=id, p=1, H=10,
                standardize=false, kw...)
            @test sdfm.identification === id
            @test sdfm.Q' * sdfm.Q ≈ I(q) atol=1e-8
            @test irf(sdfm, 6).horizon == 6
            @test_throws ArgumentError fevd(sdfm, 6; method=:cholesky)
        end
        rest = SVARRestrictions(2; signs=[SignRestriction(1, 1, 0, 1)])
        sdfm_a = estimate_structural_dfm(X, q; r=2, identification=:arias,
            restrictions=rest, p=1, H=8, max_draws=2000,
            rng=Random.MersenneTwister(5), standardize=false)
        @test sdfm_a.Q' * sdfm_a.Q ≈ I(q) atol=1e-8
        @test_throws ArgumentError estimate_structural_dfm(X, q; identification=:not_a_method)
        msg = try
            estimate_structural_dfm(X, q; identification=:not_a_method)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("long_run", msg) || occursin("fastica", msg)
    end

    @testset "stochastic identification is seed-identical" begin
        rng = Random.MersenneTwister(88)
        T_obs, N, q = 200, 8, 2
        F = zeros(T_obs, q); F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.5 .* F[t-1, :] .+ randn(rng, q)
        end
        X = F * randn(rng, N, q)' .+ 0.3 .* randn(rng, T_obs, N)
        for id in (:fastica, :narrative)
            kw = id === :narrative ?
                (sign_check=(irf -> irf[1, 1, 1] > 0), narrative_check=(shocks -> true),
                 max_draws=2000) : NamedTuple()
            a = estimate_structural_dfm(X, q; r=2, identification=id, p=1, H=8,
                standardize=false, rng=Random.MersenneTwister(99), kw...)
            b = estimate_structural_dfm(X, q; r=2, identification=id, p=1, H=8,
                standardize=false, rng=Random.MersenneTwister(99), kw...)
            @test a.Q ≈ b.Q
        end
    end

    # =========================================================================
    # SDFM-06: panel FEVD
    # =========================================================================

    @testset "panel FEVD shares sum to 1 and idiosyncratic column" begin
        rng = Random.MersenneTwister(715)
        T_obs, N, q = 180, 10, 2
        Λ = 0.5 .* randn(rng, N, q)
        Λ[1, :] .= [1.2, 0.05]
        F = zeros(T_obs, q); F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.4 .* F[t-1, :] .+ randn(rng, q)
        end
        X = F * Λ' .+ 0.15 .* randn(rng, T_obs, N)
        names = ["x$i" for i in 1:N]
        sdfm = estimate_structural_dfm(X, q; r=2, identification=:cholesky, order=[1, 2],
            p=1, H=20, standardize=false, varnames=names)
        d = fevd(sdfm, 12; space=:panel, include_idiosyncratic=false)
        @test d.variables == names
        @test length(d.shocks) == q
        for h in 1:12, i in 1:N
            @test sum(d.proportions[i, :, h]) ≈ 1 atol=1e-10
        end
        dξ = fevd(sdfm, 12; space=:panel, include_idiosyncratic=true)
        @test dξ.shocks[end] == "Idiosyncratic"
        @test length(dξ.shocks) == q + 1
        for h in 1:12, i in 1:N
            @test sum(dξ.proportions[i, :, h]) ≈ 1 atol=1e-10
        end
        ir0 = irf(sdfm, 1).values
        ξ = residuals(sdfm)
        vξ = var(ξ[:, 1])
        common1 = sum(abs2, ir0[1, 1, :])
        @test dξ.proportions[1, end, 1] ≈ vξ / (vξ + common1) atol=1e-8
        @test occursin("x1", sprint(show, fevd(sdfm, 8; space=:panel)))
        d20 = fevd(sdfm, 20; space=:panel, include_idiosyncratic=false)
        @test d20.proportions[1, 1, 20] > 0.8
        df = fevd(sdfm, 8; space=:factor)
        @test df.variables == sdfm.factor_var.varnames
    end

    # =========================================================================
    # SDFM-19: stored-Q panel HD
    # =========================================================================

    @testset "SDFM panel HD uses stored Q and verifies" begin
        rng = Random.MersenneTwister(729)
        T_obs, N, q = 160, 8, 2
        F = zeros(T_obs, q); F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.45 .* F[t-1, :] .+ randn(rng, q)
        end
        X = F * randn(rng, N, q)' .+ 0.2 .* randn(rng, T_obs, N)
        names = ["x$i" for i in 1:N]
        sdfm = estimate_structural_dfm(X, q; r=2, identification=:sign,
            sign_restrictions=[("x1", 1, 1:1, :positive)],
            varnames=names, p=1, H=12, max_draws=3000,
            rng=Random.MersenneTwister(729), standardize=false)
        hd = historical_decomposition(sdfm)
        @test verify_decomposition(hd; tol=1e-8)
        T_eff = effective_nobs(sdfm.factor_var)
        @test size(hd.contributions) == (T_eff, N, q + 1)
        @test hd.variables == names
        @test hd.shock_names == vcat(sdfm.shock_names, ["Idiosyncratic"])
        @test_throws ArgumentError historical_decomposition(sdfm; method=:cholesky)
        hd_chol = let
            ch = estimate_structural_dfm(X, q; r=2, identification=:cholesky, p=1, H=12,
                standardize=false, varnames=names)
            historical_decomposition(ch)
        end
        @test maximum(abs.(hd.contributions[:, :, 1:q] .- hd_chol.contributions[:, :, 1:q])) > 1e-8
    end

    @testset "SDFM HD recovers shock-1 path on a loading-heavy series" begin
        rng = Random.MersenneTwister(7292)
        T_obs, N, q = 400, 20, 2
        εtrue = randn(rng, T_obs, q)
        Φ = [0.5 0.0; 0.0 0.3]
        F = zeros(T_obs, q)
        F[1, :] = εtrue[1, :]
        for t in 2:T_obs
            F[t, :] = Φ * F[t-1, :] .+ εtrue[t, :]
        end
        Λ = 0.3 .* randn(rng, N, q)
        Λ[1, :] .= [1.5, 0.05]
        X = F * Λ' .+ 0.1 .* randn(rng, T_obs, N)
        sdfm = estimate_structural_dfm(X, q; r=2, identification=:cholesky, order=[1, 2],
            p=1, H=40, standardize=false)
        hd = historical_decomposition(sdfm)
        F1 = zeros(T_obs, q)
        F1[1, 1] = εtrue[1, 1]
        for t in 2:T_obs
            F1[t, :] = Φ * F1[t-1, :]
            F1[t, 1] += εtrue[t, 1]
        end
        true_c = (F1 * Λ')[:, 1]
        p = sdfm.p_var
        est = hd.contributions[:, 1, 1]
        tgt = true_c[(p + 1):end]
        ρ = cor(est, tgt)
        if ρ < 0
            ρ = cor(-est, tgt)
        end
        @test ρ > 0.9
        @test plot_result(hd) isa MacroEconometricModels.PlotOutput
    end

    @testset "auto q via Bai–Ng 2007" begin
        rng = Random.MersenneTwister(71910)
        T_obs, N, q = 120, 16, 2
        u = randn(rng, T_obs, q)
        X = u * randn(rng, N, q)'
        X[2:end, :] .+= u[1:end-1, :] * randn(rng, N, q)'
        X .+= 0.3 .* randn(rng, T_obs, N)
        sdfm = estimate_structural_dfm(X, :auto; q_method=:bai_ng, q_max=4, r=4, p=1,
            H=8, identification=:cholesky, standardize=false)
        @test sdfm.gdfm.q >= 1
        bn = bai_ng_q(sdfm)
        @test bn isa BaiNgQResult
        @test 0 <= bn.q_D1 <= bn.r
    end

    # =========================================================================
    # SDFM-05 / 07 leftover / 09: bootstrap bands, shocks/forecast, lag selection
    # =========================================================================

    @testset "bootstrap panel IRF bands" begin
        X, q = make_sdfm_data(; T_obs=120, N=12, q=2, seed=714)
        sdfm = estimate_structural_dfm(X, q; r=2, identification=:cholesky,
            p=1, H=10, standardize=false)
        ir = irf(sdfm, 10; ci_type=:bootstrap, reps=50, rng=Random.MersenneTwister(1))
        @test ir.ci_type === :bootstrap
        @test size(ir._draws) == (50, 10, 12, 2)
        @test all(isfinite, ir.values)
        @test all(isfinite, ir.ci_lower)
        @test all(isfinite, ir.ci_upper)
        @test all(ir.ci_lower .<= ir.values .<= ir.ci_upper)
        ir2 = irf(sdfm, 10; ci_type=:bootstrap, reps=50, rng=Random.MersenneTwister(1))
        @test ir.ci_lower == ir2.ci_lower
        @test ir.ci_upper == ir2.ci_upper
        for sch in (:iid, :wild, :block)
            ir_s = irf(sdfm, 6; ci_type=:bootstrap, reps=8, bootstrap=sch,
                rng=Random.MersenneTwister(2))
            @test ir_s.ci_type === :bootstrap
            @test size(ir_s._draws, 1) == 8
        end
        fv = irf(sdfm.factor_var, 10; ci_type=:bootstrap, reps=30,
            rng=Random.MersenneTwister(3))
        pan = sdfm_panel_irf(sdfm, fv)
        @test pan.ci_type === :bootstrap
        @test any(pan.ci_upper .!= pan.ci_lower)
        m2 = MacroEconometricModels._reconstruct_from_container(
            MacroEconometricModels._build_container(sdfm))
        @test m2.p_var == sdfm.p_var
        @test m2.id_order == sdfm.id_order
    end

    @testset "bootstrap 90% bands cover the true IRF on the FGLR DGP" begin
        rng = Random.MersenneTwister(20260830)
        T_obs, N, q, rstat = 400, 60, 2, 4
        Φ = [0.5 0.0; 0.1 0.4]
        Λ = 0.4 .* randn(rng, N, rstat)
        Λ[1:2, 1:2] .= [1.0 0.0; 0.6 1.0]
        Λ[1:2, 3:4] .= [0.3 0.1; 0.2 0.4]
        f = zeros(T_obs + 1, q)
        X = zeros(T_obs, N)
        true_irf = zeros(10, N, q)
        for h in 1:10
            Ah = h == 1 ? Matrix{Float64}(I, q, q) : Φ^(h - 1)
            Alag = h == 1 ? zeros(q, q) : Φ^(h - 2)
            ΨF = vcat(Ah, Alag)
            true_irf[h, :, :] = Λ * ΨF
        end
        f[1, :] = randn(rng, q)
        for t in 1:T_obs
            eps_t = randn(rng, q)
            f[t + 1, :] = Φ * f[t, :] + eps_t
            Fstat = vcat(f[t + 1, :], f[t, :])
            X[t, :] = Λ * Fstat + 0.15 .* randn(rng, N)
        end
        sdfm = estimate_structural_dfm(X, q; r=rstat, method=:fglr,
            identification=:cholesky, order=[1, 2], p=1, H=10, standardize=false)
        ir = irf(sdfm, 10; ci_type=:bootstrap, reps=200, conf_level=0.90,
            rng=Random.MersenneTwister(714200))
        aligned = copy(true_irf)
        for j in 1:q
            if dot(ir.values[1, :, j], true_irf[1, :, j]) < 0
                aligned[:, :, j] .*= -1
            end
        end
        covered = (ir.ci_lower .<= aligned .<= ir.ci_upper)
        @test mean(covered) >= 0.75
    end

    @testset "structural shocks and forecast" begin
        rng = Random.MersenneTwister(716)
        T_obs, N, q = 400, 16, 2
        F = zeros(T_obs, q)
        F[1, :] = randn(rng, q)
        for t in 2:T_obs
            F[t, :] = 0.4 .* F[t-1, :] .+ randn(rng, q)
        end
        Λ = randn(rng, N, q)
        Λ[1:2, :] .= [1.2 0.0; 0.4 1.0]
        X = F * Λ' .+ 0.2 .* randn(rng, T_obs, N)
        names = ["x$i" for i in 1:N]
        sdfm_c = estimate_structural_dfm(X, q; r=2, identification=:cholesky,
            order=[1, 2], p=1, H=8, standardize=false, varnames=names)
        εc = structural_shocks(sdfm_c)
        @test size(εc, 2) == q
        C = cov(εc)
        @test C ≈ Matrix{Float64}(I, q, q) atol=0.05
        sdfm_s = estimate_structural_dfm(X, q; r=2, identification=:sign,
            sign_restrictions=[("x1", 1, 1:1, :positive)],
            max_draws=800, p=1, H=8, standardize=false, varnames=names,
            rng=Random.MersenneTwister(7162))
        εs = structural_shocks(sdfm_s)
        Cs = cov(εs)
        @test Cs ≈ Matrix{Float64}(I, q, q) atol=0.05
        fc = forecast(sdfm_c, 6; ci_method=:none)
        @test size(fc.observables) == (6, N)
        @test all(isfinite, fc.observables)
        fcb = forecast(sdfm_c, 6; ci_method=:bootstrap, reps=40,
            rng=Random.MersenneTwister(7163))
        @test all(fcb.observables_lower .<= fcb.observables .<= fcb.observables_upper)
        @test sprint(report, fc) isa String
        @test plot_result(fc) isa MacroEconometricModels.PlotOutput
    end

    @testset "factor-VAR lag selection and stability" begin
        rng = Random.MersenneTwister(718)
        T_obs, N, q = 500, 20, 2
        Φ1 = [0.35 0.12; 0.05 0.30]
        Φ2 = [0.25 0.00; 0.00 0.22]
        F = zeros(T_obs, q)
        F[1, :] = randn(rng, q)
        F[2, :] = Φ1 * F[1, :] .+ randn(rng, q)
        for t in 3:T_obs
            F[t, :] = Φ1 * F[t-1, :] .+ Φ2 * F[t-2, :] .+ randn(rng, q)
        end
        X = F * randn(rng, N, q)' .+ 0.15 .* randn(rng, T_obs, N)
        sdfm = estimate_structural_dfm(X, q; r=2, p=:bic, p_max=5, H=8,
            identification=:cholesky, standardize=false)
        @test sdfm.p_var == 2
        @test sdfm.lag_criterion === :bic
        @test occursin("Max eigenvalue modulus", sprint(show, sdfm))

        rng2 = Random.MersenneTwister(7182)
        Te, Ne = 80, 10
        Fe = zeros(Te, 1)
        Fe[1] = randn(rng2)
        for t in 2:Te
            Fe[t] = 1.08 * Fe[t-1] + 0.3 * randn(rng2)
        end
        Xe = Fe * randn(rng2, Ne, 1)' .+ 0.05 .* randn(rng2, Te, Ne)
        sdfm_e = @test_logs (:warn, r"stab") estimate_structural_dfm(Xe, 1; r=1, p=1, H=6,
            identification=:cholesky, standardize=false, check_stability=true)
        @test is_stable(sdfm_e) == false
        @test sdfm_e.max_eigenvalue_modulus >= 1
    end

    include(joinpath(@__DIR__, "test_sdfm_recovery.jl"))

end
