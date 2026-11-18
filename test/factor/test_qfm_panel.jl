# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using LinearAlgebra
using Statistics
using Distributions
using Random
using DataFrames
using StatsAPI
using MacroEconometricModels

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const M_PAN = MacroEconometricModels

@testset "Panel Quantile Regression with Interactive Factors (QFM-06)" begin

    @testset "OVB benchmark (N=60, T=80) across quantiles τ ∈ {0.25, 0.50, 0.75}" begin
        rng = Random.Xoshiro(88801)
        N, T_obs, p = 60, 80, 2
        beta0 = [1.5, -0.8]

        f_true = randn(rng, T_obs, 1)
        lam_true = 0.6 .+ 1.2 .* rand(rng, N, 1)
        lf = lam_true * f_true'

        W_arr = Array{Float64,3}(undef, N, T_obs, p)
        W_arr[:, :, 1] = 1.0 .* lf .+ randn(rng, N, T_obs)
        W_arr[:, :, 2] = 0.4 .* lf .+ randn(rng, N, T_obs)

        Y_mat = beta0[1] .* W_arr[:, :, 1] .+ beta0[2] .* W_arr[:, :, 2] .+
                lf .+ 0.5 .* randn(rng, N, T_obs)

        # Build PanelData via xtset
        ids = repeat(1:N, outer=T_obs)
        ts = repeat(1:T_obs, inner=N)
        df = DataFrame(
            id=ids, t=ts,
            y=vec(Y_mat),
            x1=vec(W_arr[:, :, 1]),
            x2=vec(W_arr[:, :, 2]),
        )
        pd = xtset(df, :id, :t)

        # Pooled quantile regression design [1, W_1, W_2]
        W_flat = hcat(ones(N * T_obs), reshape(W_arr, N * T_obs, p))
        y_flat = vec(Y_mat)

        for tau in (0.25, 0.50, 0.75)
            qreg_pool = estimate_qreg(y_flat, W_flat, tau)
            ovb_err = maximum(abs.(StatsAPI.coef(qreg_pool)[2:3] .- beta0))
            @test ovb_err > 0.3

            m = estimate_xtqreg_factors(pd, :y, [:x1, :x2], 1, tau; seed=42)
            @test m isa QuantileFactorPanelModel{Float64}
            @test m isa StatsAPI.RegressionModel
            @test m.converged
            @test m.r == 1
            @test m.tau ≈ tau
            @test m.slope === :homogeneous
            @test m.n_units == N
            @test m.n_periods == T_obs
            @test size(m.beta_i) == (0, 0)

            xt_err = maximum(abs.(StatsAPI.coef(m) .- beta0))
            @test xt_err < 0.05

            # CDG factor normalization invariant
            @test norm(m.factors' * m.factors / T_obs - I(1)) < 1e-10
            @test isfinite(m.objective) && m.objective > 0
            @test m.fitted .+ m.residuals ≈ Y_mat atol=1e-10
        end

        # Raw matrix interface matches PanelData interface identically
        m_pd = estimate_xtqreg_factors(pd, :y, [:x1, :x2], 1, 0.5; seed=42)
        m_mat = estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5; varnames=["x1", "x2"], seed=42)
        @test StatsAPI.coef(m_pd) == StatsAPI.coef(m_mat)
        @test StatsAPI.vcov(m_pd) == StatsAPI.vcov(m_mat)
        @test m_pd.factors == m_mat.factors
        @test m_pd.loadings == m_mat.loadings

        # Quantile-varying heteroskedastic factor & quantile slope shift
        rng_h = Random.Xoshiro(88802)
        f_mean = randn(rng_h, T_obs)
        l_mean = 0.5 .+ rand(rng_h, N)
        f_vol = 0.2 .+ 1.8 .* Float64.(rand(rng_h, T_obs) .> 0.5)
        l_vol = 0.5 .+ 0.8 .* rand(rng_h, N)
        W_het = Array{Float64,3}(undef, N, T_obs, 1)
        W_het[:, :, 1] = 0.8 .* (l_mean * f_mean') .+ randn(rng_h, N, T_obs)
        Y_het = 2.0 .* W_het[:, :, 1] .+ (l_mean * f_mean') .+
                0.4 .* (l_vol * f_vol') .* randn(rng_h, N, T_obs)
        m_het_05 = estimate_xtqreg_factors(Y_het, W_het, 1, 0.5; seed=10)
        @test abs(StatsAPI.coef(m_het_05)[1] - 2.0) < 0.05
    end

    @testset "Automatic factor rank selection (r = :auto) & re-polishing" begin
        rng = Random.Xoshiro(88803)
        N, T_obs, p = 50, 60, 2
        beta0 = [1.2, -0.6]
        F0 = randn(rng, T_obs, 2)
        L0 = hcat(0.5 .+ 1.2 .* rand(rng, N), randn(rng, N))
        LF = L0 * F0'

        W_arr = Array{Float64,3}(undef, N, T_obs, p)
        W_arr[:, :, 1] = 0.9 .* LF .+ randn(rng, N, T_obs)
        W_arr[:, :, 2] = 0.5 .* (L0[:, 2] * F0[:, 2]') .+ randn(rng, N, T_obs)
        Y_mat = beta0[1] .* W_arr[:, :, 1] .+ beta0[2] .* W_arr[:, :, 2] .+
                1.5 .* LF .+ 0.35 .* randn(rng, N, T_obs)

        m_auto = estimate_xtqreg_factors(Y_mat, W_arr, :auto, 0.5; r_max=4, seed=11)
        @test m_auto.r == 2
        @test size(m_auto.factors) == (T_obs, 2)
        @test size(m_auto.loadings) == (N, 2)
        @test maximum(abs.(StatsAPI.coef(m_auto) .- beta0)) < 0.08
    end

    @testset "Heterogeneous slopes (:heterogeneous), Swamy SE & unbalanced PanelData" begin
        rng = Random.Xoshiro(88804)
        N, T_obs, p = 45, 70, 2
        beta_bar = [1.5, -0.5]
        beta_i_true = repeat(beta_bar', N, 1) .+ 0.25 .* randn(rng, N, p)

        f0 = randn(rng, T_obs, 1)
        l0 = 0.5 .+ 1.0 .* rand(rng, N, 1)
        lf = l0 * f0'
        W_arr = Array{Float64,3}(undef, N, T_obs, p)
        W_arr[:, :, 1] = 0.7 .* lf .+ randn(rng, N, T_obs)
        W_arr[:, :, 2] = 0.3 .* lf .+ randn(rng, N, T_obs)

        Y_mat = Matrix{Float64}(undef, N, T_obs)
        for i in 1:N, t in 1:T_obs
            Y_mat[i, t] = beta_i_true[i, 1] * W_arr[i, t, 1] +
                          beta_i_true[i, 2] * W_arr[i, t, 2] +
                          lf[i, t] + 0.4 * randn(rng)
        end

        m_het = estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5; slope=:heterogeneous, seed=21)
        @test m_het.slope === :heterogeneous
        @test size(m_het.beta_i) == (N, p)
        @test m_het.beta ≈ vec(mean(m_het.beta_i; dims=1))
        @test maximum(abs.(StatsAPI.coef(m_het) .- vec(mean(beta_i_true; dims=1)))) < 0.08
        @test all(>(0), StatsAPI.stderror(m_het))
        @test StatsAPI.dof(m_het) == N * p + 1 * (N + T_obs - 1)

        # Unbalanced PanelData (drop rows from t = 1 and t = T_obs for unit 1)
        df = DataFrame(
            id=repeat(1:N, outer=T_obs),
            t=repeat(1:T_obs, inner=N),
            y=vec(Y_mat),
            x1=vec(W_arr[:, :, 1]),
            x2=vec(W_arr[:, :, 2]),
        )
        df_unb = df[.!((df.id .== 1) .& ((df.t .== 1) .| (df.t .== T_obs))), :]
        pd_unb = xtset(df_unb, :id, :t)
        @test !isbalanced(pd_unb)
        m_unb = estimate_xtqreg_factors(pd_unb, :y, [:x1, :x2], 1, 0.5; seed=21)
        @test m_unb.n_units == N
        @test m_unb.n_periods == T_obs - 2
    end

    @testset "Bootstrap SE (:bootstrap), reproducibility, threading parity & StatsAPI/Tables" begin
        rng = Random.Xoshiro(88805)
        N, T_obs, p = 35, 50, 2
        beta0 = [1.0, 0.7]
        f0 = randn(rng, T_obs, 1)
        l0 = 0.5 .+ rand(rng, N, 1)
        W_arr = Array{Float64,3}(undef, N, T_obs, p)
        W_arr[:, :, 1] = 0.8 .* (l0 * f0') .+ randn(rng, N, T_obs)
        W_arr[:, :, 2] = randn(rng, N, T_obs)
        Y_mat = beta0[1] .* W_arr[:, :, 1] .+ beta0[2] .* W_arr[:, :, 2] .+
                (l0 * f0') .+ 0.4 .* randn(rng, N, T_obs)

        m_asym = estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5;
                                         varnames=["x1", "x2"], se=:asymptotic, seed=31)
        n_b = FAST ? 29 : 49
        m_boot_par = estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5;
                                             varnames=["x1", "x2"], se=:bootstrap,
                                             n_boot=n_b, threaded=true, seed=31)
        m_boot_seq = estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5;
                                             varnames=["x1", "x2"], se=:bootstrap,
                                             n_boot=n_b, threaded=false, seed=31)

        @test StatsAPI.stderror(m_boot_par) ≈ StatsAPI.stderror(m_boot_seq) atol=1e-12
        @test StatsAPI.vcov(m_boot_par) ≈ StatsAPI.vcov(m_boot_seq) atol=1e-12
        @test all(>(0), StatsAPI.stderror(m_asym))
        @test all(>(0), StatsAPI.stderror(m_boot_par))
        @test m_asym.manifest isa ReproManifest
        @test m_asym.manifest.seed == 31

        # Complete StatsAPI interface
        @test StatsAPI.coef(m_asym) == m_asym.beta
        @test StatsAPI.coefnames(m_asym) == ["x1", "x2"]
        @test StatsAPI.vcov(m_asym) == m_asym.vcov_mat
        @test StatsAPI.stderror(m_asym) == m_asym.stderr
        @test StatsAPI.nobs(m_asym) == N * T_obs
        @test StatsAPI.dof(m_asym) == p + 1 * (N + T_obs - 1)
        @test StatsAPI.dof_residual(m_asym) == N * T_obs - StatsAPI.dof(m_asym)
        @test StatsAPI.islinear(m_asym) === false
        @test StatsAPI.residuals(m_asym) === m_asym.residuals
        @test StatsAPI.fitted(m_asym) === m_asym.fitted
        @test StatsAPI.predict(m_asym) === m_asym.fitted
        @test StatsAPI.predict(m_asym, W_arr) ≈ m_asym.fitted atol=1e-12
        W_cross = randn(rng, 10, p)
        @test StatsAPI.predict(m_asym, W_cross) ≈ W_cross * m_asym.beta

        ci = StatsAPI.confint(m_asym; level=0.95)
        @test size(ci) == (p, 2)
        @test all(ci[:, 1] .< m_asym.beta .< ci[:, 2])

        # Tables.jl / DataFrame(m)
        df_tbl = DataFrame(m_asym)
        @test nrow(df_tbl) == p
        @test df_tbl.term == ["x1", "x2"]
        @test df_tbl.estimate ≈ Float64.(StatsAPI.coef(m_asym))
        @test df_tbl.std_error ≈ Float64.(StatsAPI.stderror(m_asym))
        @test df_tbl.ci_lower ≈ Float64.(ci[:, 1])
        @test df_tbl.ci_upper ≈ Float64.(ci[:, 2])

        # RSER-01 positional constructor matching fieldnames order
        m_copy = QuantileFactorPanelModel{Float64}(
            m_asym.beta, m_asym.beta_i, m_asym.vcov_mat, m_asym.stderr,
            m_asym.factors, m_asym.loadings, m_asym.tau, m_asym.r, m_asym.slope,
            m_asym.residuals, m_asym.fitted, m_asym.objective, m_asym.converged,
            m_asym.iterations, m_asym.n_units, m_asym.n_periods, m_asym.varnames,
            m_asym.manifest,
        )
        @test StatsAPI.coef(m_copy) == StatsAPI.coef(m_asym)
        @test m_copy.r == m_asym.r && m_copy.tau == m_asym.tau

        # Display across backends and repro=true
        for be in (:text, :latex, :html)
            M_PAN.with_display_backend(be) do
                @test !isempty(sprint(show, m_asym))
                @test !isempty(sprint(report, m_asym))
            end
        end
        @test occursin("Reproducibility: seed=31", sprint(io -> report(io, m_asym; repro=true)))

        # Float32 precision & Integer fallback
        Y32 = Float32.(Y_mat)
        W32 = Float32.(W_arr)
        m32 = estimate_xtqreg_factors(Y32, W32, 1, 0.5f0; seed=32)
        @test m32 isa QuantileFactorPanelModel{Float32}
        @test eltype(m32.beta) === Float32

        Y_int = round.(Int, 10 .* Y_mat)
        W_int = round.(Int, 10 .* W_arr)
        m_int = estimate_xtqreg_factors(Y_int, W_int, 1, 0.5; seed=1)
        @test m_int isa QuantileFactorPanelModel{Float64}

        # Validation errors
        pd_small = xtset(DataFrame(id=repeat(1:N, outer=T_obs), t=repeat(1:T_obs, inner=N),
                                   y=vec(Y_mat), x1=vec(W_arr[:, :, 1]), x2=vec(W_arr[:, :, 2])),
                         :id, :t)
        @test_throws ArgumentError estimate_xtqreg_factors(pd_small, :missing_y, [:x1], 1, 0.5)
        @test_throws ArgumentError estimate_xtqreg_factors(pd_small, :y, Symbol[], 1, 0.5)
        @test_throws ArgumentError estimate_xtqreg_factors(pd_small, :y, [:x1, :missing_x], 1, 0.5)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.0)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, 1, 1.0)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, 0, 0.5)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, N, 0.5)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, :bogus, 0.5)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5; slope=:bogus)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5; se=:bogus)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5; n_boot=0)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5; tol=-1.0)
        @test_throws ArgumentError estimate_xtqreg_factors(Y_mat, W_arr, 1, 0.5; varnames=["only_one"])
        @test_throws DimensionMismatch estimate_xtqreg_factors(Y_mat, zeros(N + 1, T_obs, p), 1, 0.5)
        @test_throws DimensionMismatch StatsAPI.predict(m_asym, zeros(N, T_obs + 1, p))
        @test_throws DimensionMismatch StatsAPI.predict(m_asym, zeros(5, p + 1))
    end

    @testset "No method ambiguities in MacroEconometricModels" begin
        @test isempty(Test.detect_ambiguities(MacroEconometricModels))
    end
end
