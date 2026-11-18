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
using MacroEconometricModels

const M = MacroEconometricModels
const qfm_predict = MacroEconometricModels.predict
const qfm_fitted = MacroEconometricModels.fitted
const qfm_residuals = MacroEconometricModels.residuals
const qfm_r2 = MacroEconometricModels.r2
const qfm_nobs = MacroEconometricModels.nobs
const qfm_dof = MacroEconometricModels.dof

function _qfm_trace_r2(A::AbstractMatrix, B::AbstractMatrix)
    QA = Matrix(qr(A).Q)
    QB = Matrix(qr(B).Q)
    size(QA, 2) == size(QB, 2) ||
        throw(DimensionMismatch("_qfm_trace_r2 requires equal column counts"))
    return tr((QA' * QB) * (QB' * QA)) / size(B, 2)
end

_qfm_vec_r2(a::AbstractVector, b::AbstractVector) = cor(a, b)^2

function _qfm_check_norm_invariants(m::QuantileFactorModel{T}; tol::Real=1e-10) where {T}
    T_obs, N = size(m.X)
    r = m.r
    F = m.factors
    L = m.loadings
    @test size(F) == (T_obs, r)
    @test size(L) == (N, r)
    @test length(m.sigma_sq) == r
    @test norm(F' * F / T(T_obs) - Matrix{T}(I, r, r)) < tol
    @test norm(L' * L / T(N) - Diagonal(m.sigma_sq)) < tol
    @test issorted(m.sigma_sq; rev=true)
    @test all(x -> x >= zero(T), m.sigma_sq)
    for j in 1:r
        col = @view L[:, j]
        idx = findfirst(x -> abs(x) > T(1e-8), col)
        i = idx === nothing ? argmax(abs.(col)) : idx
        if abs(L[i, j]) > zero(T)
            @test L[i, j] > zero(T)
        end
    end
end

@testset "Quantile Factor Model (QFM-02)" begin

    @testset "IQR objective monotonicity across iterations" begin
        rng = Random.Xoshiro(88401)
        T_obs, N = 120, 40
        F_true = randn(rng, T_obs, 3)
        L_true = randn(rng, N, 3)
        X = F_true * L_true' .+ 0.5 .* rand(rng, TDist(3), T_obs, N)

        for tau in (0.1, 0.5, 0.9), r in (1, 2, 3), init in (:pca, :quantile_svd, :random)
            m, trace = M._qfm_iqr_trace(X, r, tau; init=init, n_starts=1, seed=42)
            @test m.converged
            @test length(trace) == m.iterations
            @test m.iterations >= 2
            @test trace[end] ≈ m.objective atol=1e-13
            @test maximum(diff(trace)) <= 1e-12
            _qfm_check_norm_invariants(m)
        end

        # Standardized runs are also monotone
        X_scaled = X .* collect(1.0:N)' .+ collect(-19.5:1.0:20.0)'
        for rob in (true, false)
            m_s, trace_s = M._qfm_iqr_trace(X_scaled, 2, 0.25;
                                            standardize=true, robust_scale=rob,
                                            n_starts=1, seed=99)
            @test m_s.converged
            @test length(trace_s) == m_s.iterations
            @test maximum(diff(trace_s)) <= 1e-12
            _qfm_check_norm_invariants(m_s)
        end
    end

    @testset "CDG normalization helper invariants and exact product preservation" begin
        rng = Random.Xoshiro(88402)
        T_obs, N, r = 80, 30, 3
        F_raw = randn(rng, T_obs, r) * [1.2 -0.5 0.3; 0.0 2.0 0.7; 0.4 -0.3 0.8]
        L_raw = randn(rng, N, r) * [0.8 0.2 -0.4; -0.1 1.5 0.2; 0.3 0.1 0.9]
        # Force negative first nonzero entry in column 1 and near-zero head in column 2
        L_raw[1, :] .= [-2.5, 0.0, 1.0]
        L_raw[1, 2] = 1e-12

        F_norm, L_norm, sigma_sq = M._qfm_normalize(F_raw, L_raw)
        @test norm(F_norm * L_norm' - F_raw * L_raw') < 1e-13
        @test norm(F_norm' * F_norm / T_obs - I(r)) < 1e-12
        @test norm(L_norm' * L_norm / N - Diagonal(sigma_sq)) < 1e-12
        @test issorted(sigma_sq; rev=true)
        @test all(>(0), sigma_sq)
        for j in 1:r
            idx = findfirst(x -> abs(x) > 1e-8, @view(L_norm[:, j]))
            i = idx === nothing ? argmax(abs.(@view(L_norm[:, j]))) : idx
            @test L_norm[i, j] > 0.0
        end

        # Tiny / zero column fallback in sign normalization
        F_zero = randn(rng, 40, 2)
        L_zero = hcat(randn(rng, 15), fill(1e-12, 15))
        L_zero[7, 2] = -5e-10
        F_zn, L_zn, sq_z = M._qfm_normalize(F_zero, L_zero)
        @test norm(F_zn * L_zn' - F_zero * L_zero') < 1e-13
        @test norm(F_zn' * F_zn / 40 - I(2)) < 1e-12
        idx2 = findfirst(x -> abs(x) > 1e-8, @view(L_zn[:, 2]))
        i2 = idx2 === nothing ? argmax(abs.(@view(L_zn[:, 2]))) : idx2
        @test L_zn[i2, 2] >= 0.0
    end

    @testset "CDG (2021) location-scale DGP with t(2) and Cauchy errors" begin
        T_obs, N = 200, 80

        # 1. Heavy-tailed t(2) errors
        rng_t = Random.Xoshiro(88403)
        f1_t = randn(rng_t, T_obs)
        g_t = exp.(0.8 .* randn(rng_t, T_obs)) .+ 0.2
        lam1_t = randn(rng_t, N) .* 2.0
        lam2_t = rand(rng_t, Uniform(0.8, 1.5), N)
        X_t2 = f1_t * lam1_t' .+ (g_t * lam2_t') .* (0.3 .* rand(rng_t, TDist(2), T_obs, N))

        qfm_t_05 = estimate_qfm(X_t2, 1, 0.5; init=:quantile_svd, seed=1)
        qfm_t_01 = estimate_qfm(X_t2, 2, 0.1; init=:quantile_svd, seed=2)
        qfm_t_09 = estimate_qfm(X_t2, 2, 0.9; init=:quantile_svd, seed=3)
        @test _qfm_vec_r2(vec(qfm_t_05.factors), f1_t) > 0.90
        @test _qfm_trace_r2(qfm_t_01.factors, hcat(f1_t, g_t)) > 0.90
        @test _qfm_trace_r2(qfm_t_09.factors, hcat(f1_t, g_t)) > 0.90
        _qfm_check_norm_invariants(qfm_t_05)
        _qfm_check_norm_invariants(qfm_t_01)
        _qfm_check_norm_invariants(qfm_t_09)

        # 2. Heavy-tailed Cauchy errors: QFM recovers f1 at tau=0.5 and span(f1, g) at tau=0.1, 0.9
        rng_c = Random.Xoshiro(2021)
        f1_c = randn(rng_c, T_obs)
        g_c = exp.(0.8 .* randn(rng_c, T_obs))
        lam1_c = randn(rng_c, N) .* 2.0
        lam2_c = rand(rng_c, Uniform(0.5, 1.5), N)
        X_c = f1_c * lam1_c' .+ (g_c * lam2_c') .* (0.15 .* rand(rng_c, Cauchy(0, 1), T_obs, N))

        qfm_c_05 = estimate_qfm(X_c, 1, 0.5; init=:quantile_svd, seed=1)
        qfm_c_01 = estimate_qfm(X_c, 2, 0.1; init=:quantile_svd, seed=1)
        qfm_c_09 = estimate_qfm(X_c, 2, 0.9; init=:quantile_svd, seed=1)
        pca_c_raw = estimate_factors(X_c, 2; standardize=false)

        @test _qfm_vec_r2(vec(qfm_c_05.factors), f1_c) > 0.90
        @test _qfm_trace_r2(qfm_c_01.factors, hcat(f1_c, g_c)) > 0.90
        @test _qfm_trace_r2(qfm_c_09.factors, hcat(f1_c, g_c)) > 0.90
        @test _qfm_trace_r2(pca_c_raw.factors, hcat(f1_c, g_c)) < 0.40

        # Also check standard Cauchy(0, 1) where default estimate_factors(X, 2) fails (< 0.40)
        rng_ch = Random.Xoshiro(8841)
        f1_ch = randn(rng_ch, T_obs)
        g_ch = rand(rng_ch, Uniform(0.5, 4.5), T_obs)
        lam1_ch = randn(rng_ch, N) .* 4.0
        lam2_ch = rand(rng_ch, Uniform(0.4, 0.8), N)
        X_ch = f1_ch * lam1_ch' .+ (g_ch * lam2_ch') .* rand(rng_ch, Cauchy(0, 1), T_obs, N)
        pca_ch_def = estimate_factors(X_ch, 2)
        qfm_ch_05 = estimate_qfm(X_ch, 1, 0.5; init=:quantile_svd, seed=1)
        @test _qfm_trace_r2(pca_ch_def.factors, hcat(f1_ch, g_ch)) < 0.40
        @test _qfm_vec_r2(vec(qfm_ch_05.factors), f1_ch) > 0.90
    end

    @testset "Standardization (robust_scale=true vs false) & scale fallbacks" begin
        rng = Random.Xoshiro(88404)
        T_obs, N, r = 150, 30, 2
        F_true = randn(rng, T_obs, r)
        L_true = randn(rng, N, r)
        scales = exp.(range(-2.0, 3.0; length=N))
        shifts = collect(range(-50.0, 50.0; length=N))
        X = (F_true * L_true' .+ 0.25 .* randn(rng, T_obs, N)) .* scales' .+ shifts'

        m_rob = estimate_qfm(X, r, 0.5; standardize=true, robust_scale=true, seed=11)
        m_sd = estimate_qfm(X, r, 0.5; standardize=true, robust_scale=false, seed=11)
        @test m_rob.standardized == true
        @test m_sd.standardized == true
        @test _qfm_trace_r2(m_rob.factors, F_true) > 0.95
        @test _qfm_trace_r2(m_sd.factors, F_true) > 0.95
        # common_component is unstandardized in raw X units
        @test isapprox(m_rob.common_component + m_rob.residuals, X; atol=1e-11)
        @test isapprox(m_sd.common_component + m_sd.residuals, X; atol=1e-11)
        @test mean(m_rob.pseudo_r2) > 0.70
        @test mean(m_sd.pseudo_r2) > 0.70

        # Scale fallback chain:
        # col 3 -> 20% negative, 60% zero, 20% positive -> IQR = 0, MAD = 0 -> SD fallback
        # col 4 -> constant -> IQR = 0, MAD = 0, SD = 0 -> clamped at 1e-8
        X_fb = copy(X)
        n_tail = Int(0.2 * T_obs)
        X_fb[1:n_tail, 3] .= -10.0
        X_fb[(n_tail + 1):(T_obs - n_tail), 3] .= 0.0
        X_fb[(T_obs - n_tail + 1):T_obs, 3] .= 10.0
        X_fb[:, 4] .= 7.5
        X_proc, c_fb, s_fb = M._qfm_standardize(X_fb, true)
        @test all(isfinite, X_proc)
        @test all(isfinite, c_fb)
        @test all(>=(1e-8), s_fb)
        @test s_fb[3] ≈ std(@view(X_fb[:, 3])) atol=1e-12
        @test s_fb[4] == 1e-8
        @test all(iszero, X_proc[:, 4])

        m_fb = estimate_qfm(X_fb, 2, 0.5; standardize=true, robust_scale=true, seed=5)
        @test all(isfinite, m_fb.factors)
        @test all(isfinite, m_fb.loadings)
        @test all(isfinite, m_fb.common_component)
        @test all(isfinite, m_fb.pseudo_r2)
    end

    @testset "Initializations (:pca, :quantile_svd, :random) and multi-start" begin
        rng = Random.Xoshiro(88405)
        T_obs, N, r = 120, 35, 2
        F_true = randn(rng, T_obs, r)
        L_true = randn(rng, N, r)
        X = F_true * L_true' .+ 0.2 .* randn(rng, T_obs, N)

        for init_opt in (:pca, :quantile_svd, :random)
            m = estimate_qfm(X, r, 0.5; init=init_opt, n_starts=2, seed=2025)
            @test m.converged
            @test _qfm_trace_r2(m.factors, F_true) > 0.95
            _qfm_check_norm_invariants(m)
        end

        m1 = estimate_qfm(X, r, 0.25; init=:random, n_starts=1, seed=77)
        m3 = estimate_qfm(X, r, 0.25; init=:random, n_starts=3, seed=77)
        @test m3.objective <= m1.objective + 1e-14
    end

    @testset "Float32 support and Integer @float_fallback" begin
        rng = Random.Xoshiro(88406)
        T_obs, N, r = 100, 25, 2
        F32 = Float32.(randn(rng, T_obs, r))
        L32 = Float32.(randn(rng, N, r))
        X32 = F32 * L32' .+ 0.2f0 .* Float32.(randn(rng, T_obs, N))

        m32, trace32 = M._qfm_iqr_trace(X32, r, 0.5f0; standardize=true, seed=32)
        @test m32 isa QuantileFactorModel{Float32}
        @test eltype(m32.X) === Float32
        @test typeof(m32.tau) === Float32
        @test eltype(m32.factors) === Float32
        @test eltype(m32.loadings) === Float32
        @test eltype(m32.sigma_sq) === Float32
        @test eltype(m32.common_component) === Float32
        @test eltype(m32.residuals) === Float32
        @test typeof(m32.objective) === Float32
        @test eltype(m32.pseudo_r2) === Float32
        @test m32.converged
        @test maximum(diff(trace32)) <= 1f-5
        _qfm_check_norm_invariants(m32; tol=1f-4)
        @test _qfm_trace_r2(m32.factors, F32) > 0.95

        # Integer matrix fallback -> Float64
        X_int = rand(rng, -5:5, 60, 15)
        m_int = estimate_qfm(X_int, 2, 0.5; seed=10)
        @test m_int isa QuantileFactorModel{Float64}
        @test all(isfinite, m_int.factors)

        # SubArray view input
        m_view = estimate_qfm(@view(X32[1:80, 1:20]), 2, 0.5f0; seed=12)
        @test m_view isa QuantileFactorModel{Float32}
    end

    @testset "Seed reproducibility, manifest, and threading parity" begin
        rng = Random.Xoshiro(88407)
        X = randn(rng, 80, 20)

        m_a = estimate_qfm(X, 2, 0.75; init=:random, n_starts=3, seed=1234, threaded=true)
        m_b = estimate_qfm(X, 2, 0.75; init=:random, n_starts=3, seed=1234, threaded=true)
        m_c = estimate_qfm(X, 2, 0.75; init=:random, n_starts=3, seed=1234, threaded=false)

        @test m_a.factors == m_b.factors
        @test m_a.loadings == m_b.loadings
        @test m_a.sigma_sq == m_b.sigma_sq
        @test m_a.common_component == m_b.common_component
        @test m_a.objective == m_b.objective
        @test m_a.pseudo_r2 == m_b.pseudo_r2
        @test m_a.iterations == m_b.iterations

        @test m_a.factors == m_c.factors
        @test m_a.loadings == m_c.loadings
        @test m_a.objective == m_c.objective

        @test m_a.manifest isa ReproManifest
        @test m_a.manifest.seed == 1234
        @test m_a.manifest.settings["r"] == 2
        @test m_a.manifest.settings["tau"] == 0.75
        @test m_a.manifest.settings["init"] == "random"
        @test m_a.manifest.settings["n_starts"] == 3

        m_noseed = estimate_qfm(X, 2, 0.5; rng=Random.Xoshiro(1))
        @test m_noseed.manifest isa ReproManifest
        @test m_noseed.manifest.seed === nothing
    end

    @testset "StatsAPI interface, accessors, pseudo-R², display, and constructors" begin
        rng = Random.Xoshiro(88408)
        T_obs, N, r = 90, 24, 2
        F_true = randn(rng, T_obs, r)
        L_true = randn(rng, N, r)
        X = F_true * L_true' .+ 0.3 .* randn(rng, T_obs, N)
        vnames = ["Series_$i" for i in 1:N]

        m = estimate_qfm(X, r, 0.3; varnames=vnames, seed=9)
        @test m isa AbstractFactorModel
        @test qfm_nobs(m) == T_obs
        @test qfm_dof(m) == N * r + T_obs * r - r^2
        @test qfm_predict(m) === m.common_component
        @test qfm_fitted(m) === m.common_component
        @test qfm_residuals(m) === m.residuals
        @test qfm_r2(m) === m.pseudo_r2
        @test M.factors(m) === m.factors
        @test M.loadings(m) === m.loadings
        @test isapprox(qfm_fitted(m) + qfm_residuals(m), m.X; atol=1e-13)
        @test m.varnames == vnames
        @test m.method === :iqr
        @test m.se_loadings === nothing
        @test m.se_factors === nothing

        # Independent check of Koenker-Machado pseudo-R²
        for i in 1:N
            v_hat = M._check_loss(X[:, i] .- m.common_component[:, i], 0.3)
            v_tilde = M._check_loss(X[:, i] .- quantile(X[:, i], 0.3), 0.3)
            @test m.pseudo_r2[i] ≈ 1.0 - v_hat / v_tilde atol=1e-12
        end

        # show and report across backends
        s_show = sprint(show, m)
        @test occursin("Quantile Factor Model", s_show)
        @test occursin("Factor Strength", s_show)
        @test occursin("Pseudo-R", s_show)
        @test occursin("Series_", s_show)
        @test !occursin("Reproducibility:", s_show)

        s_rep = sprint(io -> report(io, m))
        @test s_rep == s_show
        s_rep_repro = sprint(io -> report(io, m; repro=true))
        @test occursin("Reproducibility: seed=9", s_rep_repro)

        redirect_stdout(devnull) do
            report(m)
            report(m; repro=true)
        end

        for be in (:latex, :html)
            out_be = with_display_backend(be) do
                sprint(show, m)
            end
            @test !isempty(out_be)
        end

        # Backward-compatible & keyword constructors
        m17 = QuantileFactorModel{Float64}(
            m.X, m.tau, m.factors, m.loadings, m.sigma_sq, m.common_component,
            m.residuals, m.objective, m.pseudo_r2, m.r, m.method, m.standardized,
            m.converged, m.iterations, nothing, nothing, m.varnames,
        )
        @test m17.manifest === nothing
        m15 = QuantileFactorModel{Float64}(
            m.X, m.tau, m.factors, m.loadings, m.sigma_sq, m.common_component,
            m.residuals, m.objective, m.pseudo_r2, m.r, m.method, m.standardized,
            m.converged, m.iterations, m.varnames,
        )
        @test m15.varnames == vnames && m15.se_loadings === nothing
        m14 = QuantileFactorModel{Float64}(
            m.X, m.tau, m.factors, m.loadings, m.sigma_sq, m.common_component,
            m.residuals, m.objective, m.pseudo_r2, m.r, m.method, m.standardized,
            m.converged, m.iterations,
        )
        @test m14.varnames == ["Var $i" for i in 1:N]
        m_kw = QuantileFactorModel(
            X=m.X, tau=m.tau, factors=m.factors, loadings=m.loadings,
            sigma_sq=m.sigma_sq, common_component=m.common_component,
            residuals=m.residuals, objective=m.objective, pseudo_r2=m.pseudo_r2,
            r=m.r,
        )
        @test m_kw.method === :iqr && m_kw.manifest === nothing
    end

    @testset "Input validation and edge cases" begin
        rng = Random.Xoshiro(88409)
        X = randn(rng, 40, 12)

        X_nan = copy(X); X_nan[1, 1] = NaN
        X_inf = copy(X); X_inf[2, 2] = Inf
        @test_throws ArgumentError estimate_qfm(X_nan, 2)
        @test_throws ArgumentError estimate_qfm(X_inf, 2)
        @test_throws ArgumentError estimate_qfm(X, 0)
        @test_throws ArgumentError estimate_qfm(X, -1)
        @test_throws ArgumentError estimate_qfm(X, 13)
        @test_throws ArgumentError estimate_qfm(X, 2, 0.0)
        @test_throws ArgumentError estimate_qfm(X, 2, 1.0)
        @test_throws ArgumentError estimate_qfm(X, 2, -0.1)
        @test_throws ArgumentError estimate_qfm(X, 2, 1.1)
        @test_throws ArgumentError estimate_qfm(X, 2, NaN)
        @test_throws ArgumentError estimate_qfm(X, 2, 0.5; init=:bogus)
        @test_throws ArgumentError estimate_qfm(X, 2, 0.5; n_starts=0)
        @test_throws ArgumentError estimate_qfm(X, 2, 0.5; max_iter=0)
        @test_throws ArgumentError estimate_qfm(X, 2, 0.5; tol=0.0)
        @test_throws ArgumentError estimate_qfm(X, 2, 0.5; tol=-1e-5)
        @test_throws ArgumentError estimate_qfm(X, 2, 0.5; varnames=["too_few"])

        # r = 1 edge case
        m1 = estimate_qfm(X, 1, 0.5; seed=1)
        @test size(m1.factors) == (40, 1)
        @test size(m1.loadings) == (12, 1)
        _qfm_check_norm_invariants(m1)

        # r = min(T, N) edge case
        X_sm = randn(rng, 25, 6)
        m_max = estimate_qfm(X_sm, 6, 0.5; seed=1)
        @test size(m_max.factors) == (25, 6)
        _qfm_check_norm_invariants(m_max)

        # Noiseless exact rank-r matrix -> near-zero objective
        F_exact = randn(rng, 50, 2)
        L_exact = randn(rng, 20, 2)
        X_exact = F_exact * L_exact'
        m_ex = estimate_qfm(X_exact, 2, 0.5; seed=3)
        @test m_ex.converged
        @test m_ex.objective < 1e-12
        @test _qfm_trace_r2(m_ex.factors, F_exact) > 0.9999

        # All-zero panel
        m_zero = estimate_qfm(zeros(30, 10), 2, 0.5; seed=4)
        @test m_zero.objective == 0.0
        @test all(iszero, m_zero.loadings)
        @test all(iszero, m_zero.pseudo_r2)
        @test norm(m_zero.factors' * m_zero.factors / 30 - I(2)) < 1e-12
    end

    @testset "No method ambiguities in MacroEconometricModels" begin
        @test isempty(Test.detect_ambiguities(MacroEconometricModels))
    end
end
