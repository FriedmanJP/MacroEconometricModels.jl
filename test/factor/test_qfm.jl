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
using ForwardDiff
using MacroEconometricModels

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

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

function _qfm_chordal_dist(A::AbstractMatrix, B::AbstractMatrix)
    QA = Matrix(qr(A).Q)
    QB = Matrix(qr(B).Q)
    r = size(QA, 2)
    r == size(QB, 2) || throw(DimensionMismatch("_qfm_chordal_dist requires equal column counts"))
    svals = svdvals(QA' * QB)
    return sqrt(max(0.0, r - sum(abs2, svals))), minimum(svals)
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

    @testset "Convolution-Smoothed QFA & Standard Errors (QFM-03)" begin

        @testset "Analytical gradient vs ForwardDiff.gradient (< 1e-10) for all kernels" begin
            rng = Random.Xoshiro(88501)
            T_obs, N, r = 35, 14, 2
            X = randn(rng, T_obs, N)
            # Include large-magnitude entries to exercise overflow safety of :logistic and tails of :epanechnikov
            X[1, 1] = 250.0
            X[2, 2] = -250.0
            X[3, 3] = 0.0
            F0 = randn(rng, T_obs, r)
            L0 = randn(rng, N, r)
            nF = T_obs * r

            for kern in (:gaussian, :logistic, :epanechnikov), tau in (0.1, 0.5, 0.9), h in (0.1, 0.4)
                gf, gl = M._qfm_smooth_grad(X, F0, L0, tau, h, kern)
                g_anal = vcat(vec(gf), vec(gl))
                theta0 = vcat(vec(F0), vec(L0))
                g_fd = ForwardDiff.gradient(theta -> begin
                    Fm = reshape(@view(theta[1:nF]), T_obs, r)
                    Lm = reshape(@view(theta[(nF + 1):end]), N, r)
                    M._qfm_smooth_objective(X, Fm, Lm, tau, h, kern)
                end, theta0)
                @test all(isfinite, g_anal)
                @test maximum(abs.(g_anal .- g_fd)) < 1e-10
            end
        end

        @testset "Smoothed QFA vs IQR subspace agreement & h -> 0+ convergence" begin
            rng = Random.Xoshiro(88502)
            T_obs, N, r = 200, 80, 2
            F_true = sqrt(T_obs) .* Matrix(qr(randn(rng, T_obs, r)).Q)
            L_true = randn(rng, N, r)
            e_base = randn(rng, T_obs, N)

            # Median with bandwidth=:auto and tail quantiles with small h
            for tau in (0.2, 0.5, 0.8)
                e_shifted = 0.5 .* (e_base .- quantile(Normal(), tau))
                X = F_true * L_true' .+ e_shifted
                m_iqr = estimate_qfm(X, r, tau; method=:iqr, seed=11)

                for kern in (:gaussian, :logistic, :epanechnikov)
                    bw = tau == 0.5 ? :auto : 0.05
                    m_sm = estimate_qfm(X, r, tau; method=:smoothed_iqr, kernel=kern,
                                        bandwidth=bw, seed=11)
                    @test m_sm.method === :smoothed_iqr
                    @test m_sm.converged
                    _qfm_check_norm_invariants(m_sm; tol=1e-9)
                    @test m_sm.objective ≈ M._qfm_objective(X, m_sm.factors, m_sm.loadings, tau) atol=1e-12
                    d_chord, min_sval = _qfm_chordal_dist(m_iqr.factors, m_sm.factors)
                    @test d_chord < 0.05
                    @test min_sval > 0.995
                    @test _qfm_trace_r2(m_sm.factors, F_true) > 0.95
                end
            end

            # Convergence as h -> 0+
            X_h = F_true * L_true' .+ 0.5 .* (e_base .- quantile(Normal(), 0.25))
            m_iqr_h = estimate_qfm(X_h, r, 0.25; method=:iqr, seed=12)
            m_coarse = estimate_qfm(X_h, r, 0.25; method=:smoothed_iqr, bandwidth=0.5, seed=12)
            m_fine = estimate_qfm(X_h, r, 0.25; method=:smoothed_iqr, bandwidth=0.05, seed=12)
            d_coarse, _ = _qfm_chordal_dist(m_iqr_h.factors, m_coarse.factors)
            d_fine, _ = _qfm_chordal_dist(m_iqr_h.factors, m_fine.factors)
            @test d_fine < d_coarse
            @test d_fine < 0.05
        end

        @testset "Monte Carlo 95% coverage of CDG asymptotic SEs (N=80, T=200)" begin
            T_mc, N_mc, r_mc = 200, 80, 2
            tau = 0.25
            q_shift = quantile(Normal(), tau)
            n_rep = FAST ? 10 : 20
            cov_iqr_L = Float64[]
            cov_iqr_F = Float64[]
            cov_sm_L = Float64[]

            for rep in 1:n_rep
                r_rng = Random.Xoshiro(88510 + rep)
                F0 = sqrt(T_mc) .* Matrix(qr(randn(r_rng, T_mc, r_mc)).Q)
                L0 = randn(r_rng, N_mc, r_mc)
                e = randn(r_rng, T_mc, N_mc) .- q_shift
                X = F0 * L0' .+ 0.5 .* e

                m_iqr = estimate_qfm(X, r_mc, tau; method=:iqr, se=:asymptotic,
                                     hac_bandwidth=:auto, n_starts=1, seed=rep)
                m_sm = estimate_qfm(X, r_mc, tau; method=:smoothed_iqr, se=:asymptotic,
                                    hac_bandwidth=0, n_starts=1, seed=rep)

                @test size(m_iqr.se_loadings) == (N_mc, r_mc)
                @test size(m_iqr.se_factors) == (T_mc, r_mc)
                @test all(x -> isfinite(x) && x > 0, m_iqr.se_loadings)
                @test all(x -> isfinite(x) && x > 0, m_iqr.se_factors)

                H_i = (F0' * m_iqr.factors) / T_mc
                L_rot_i = L0 * inv(H_i)'
                F_rot_i = F0 * H_i
                push!(cov_iqr_L, mean(abs.(m_iqr.loadings .- L_rot_i) .<= 1.96 .* m_iqr.se_loadings))
                push!(cov_iqr_F, mean(abs.(m_iqr.factors .- F_rot_i) .<= 1.96 .* m_iqr.se_factors))

                H_s = (F0' * m_sm.factors) / T_mc
                L_rot_s = L0 * inv(H_s)'
                push!(cov_sm_L, mean(abs.(m_sm.loadings .- L_rot_s) .<= 1.96 .* m_sm.se_loadings))
            end

            @test 0.88 <= mean(cov_iqr_L) <= 0.99
            @test 0.88 <= mean(cov_iqr_F) <= 0.99
            @test 0.88 <= mean(cov_sm_L) <= 0.99
        end

        @testset "HAC adjustment for serially correlated idiosyncratic errors" begin
            rng = Random.Xoshiro(88530)
            T_obs, N, r = 240, 50, 2
            # Both factors and idiosyncratic errors follow AR(1) so score s_{it} = ψ_τ(e_{it}) f_t is serially correlated
            F_raw = zeros(T_obs, r)
            rho_f = 0.75
            f_sd = sqrt(1 - rho_f^2)
            F_raw[1, :] = randn(rng, r)
            for t in 2:T_obs
                F_raw[t, :] = rho_f .* F_raw[t - 1, :] .+ f_sd .* randn(rng, r)
            end
            F0 = sqrt(T_obs) .* Matrix(qr(F_raw).Q)
            L0 = randn(rng, N, r)
            e = zeros(T_obs, N)
            rho_e = 0.75
            e_sd = sqrt(1 - rho_e^2)
            e[1, :] = randn(rng, N)
            for t in 2:T_obs
                e[t, :] = rho_e .* e[t - 1, :] .+ e_sd .* randn(rng, N)
            end
            X = F0 * L0' .+ 0.5 .* e

            m_white = estimate_qfm(X, r, 0.5; se=:asymptotic, hac_bandwidth=0, seed=1)
            m_hac_auto = estimate_qfm(X, r, 0.5; se=:asymptotic, hac_bandwidth=:auto, seed=1)
            m_hac_fixed = estimate_qfm(X, r, 0.5; se=:asymptotic, hac_bandwidth=6, seed=1)

            @test m_white.se_factors == m_hac_auto.se_factors
            @test median(m_hac_fixed.se_loadings) > median(m_white.se_loadings)
            @test median(m_hac_auto.se_loadings) > median(m_white.se_loadings)
        end

        @testset "Bootstrap SEs, Procrustes alignment, and seed/threading parity" begin
            rng = Random.Xoshiro(88540)
            T_obs, N, r = 140, 45, 2
            F0 = sqrt(T_obs) .* Matrix(qr(randn(rng, T_obs, r)).Q)
            L0 = randn(rng, N, r)
            X = F0 * L0' .+ 0.4 .* (randn(rng, T_obs, N) .- quantile(Normal(), 0.3))

            m_asy = estimate_qfm(X, r, 0.3; method=:iqr, se=:asymptotic, hac_bandwidth=0, seed=42)
            m_boot1 = estimate_qfm(X, r, 0.3; method=:iqr, se=:bootstrap, n_boot=60,
                                   threaded=true, seed=42)
            m_boot2 = estimate_qfm(X, r, 0.3; method=:iqr, se=:bootstrap, n_boot=60,
                                   threaded=false, seed=42)
            m_boot_sm = estimate_qfm(X, r, 0.3; method=:smoothed_iqr, se=:bootstrap, n_boot=40,
                                     seed=42)

            @test m_boot1.factors == m_asy.factors
            @test m_boot1.loadings == m_asy.loadings
            @test m_boot1.se_loadings == m_boot2.se_loadings
            @test m_boot1.se_factors == m_boot2.se_factors
            @test size(m_boot1.se_loadings) == (N, r)
            @test size(m_boot1.se_factors) == (T_obs, r)
            @test all(x -> isfinite(x) && x > 0, m_boot1.se_loadings)
            @test all(x -> isfinite(x) && x > 0, m_boot1.se_factors)

            ratio_L = median(m_boot1.se_loadings ./ m_asy.se_loadings)
            ratio_F = median(m_boot1.se_factors ./ m_asy.se_factors)
            @test 0.6 <= ratio_L <= 1.6
            @test 0.6 <= ratio_F <= 1.6

            @test size(m_boot_sm.se_loadings) == (N, r)
            @test all(x -> isfinite(x) && x > 0, m_boot_sm.se_loadings)
        end

        @testset "Standardization SE invariance, Float32 support, and display with SEs" begin
            rng = Random.Xoshiro(88550)
            T_obs, N, r = 120, 35, 2
            F0 = sqrt(T_obs) .* Matrix(qr(randn(rng, T_obs, r)).Q)
            L0 = randn(rng, N, r)
            X_base = F0 * L0' .+ 0.3 .* randn(rng, T_obs, N)
            scales = exp.(range(-2.0, 4.0; length=N))
            shifts = collect(range(-100.0, 100.0; length=N))
            X_scaled = X_base .* scales' .+ shifts'

            m_std_1 = estimate_qfm(X_base, r, 0.5; standardize=true, se=:asymptotic, hac_bandwidth=0, seed=5)
            m_std_2 = estimate_qfm(X_scaled, r, 0.5; standardize=true, se=:asymptotic, hac_bandwidth=0, seed=5)
            @test isapprox(m_std_1.se_loadings, m_std_2.se_loadings; rtol=1e-10)
            @test isapprox(m_std_1.se_factors, m_std_2.se_factors; rtol=1e-10)
            m_std_a1 = estimate_qfm(X_base, r, 0.5; standardize=true, se=:asymptotic, seed=5)
            m_std_a2 = estimate_qfm(X_scaled, r, 0.5; standardize=true, se=:asymptotic, seed=5)
            @test isapprox(m_std_a1.se_loadings, m_std_a2.se_loadings; rtol=0.05)

            # Display with SEs across backends
            s_se = sprint(show, m_std_1)
            @test occursin("Standard Errors Summary", s_se)
            @test occursin("Std.Err.", s_se)
            for be in (:latex, :html)
                out_be = with_display_backend(be) do
                    sprint(show, m_std_1)
                end
                @test !isempty(out_be)
            end

            # Float32 support across smoothed kernels & SE modes
            X32 = Float32.(X_base)
            for kern in (:gaussian, :logistic, :epanechnikov)
                m32_sm = estimate_qfm(X32, r, 0.5f0; method=:smoothed_iqr, kernel=kern,
                                      se=:asymptotic, seed=8)
                @test m32_sm isa QuantileFactorModel{Float32}
                @test eltype(m32_sm.se_loadings) === Float32
                @test eltype(m32_sm.se_factors) === Float32
                @test all(isfinite, m32_sm.se_loadings)
                @test all(isfinite, m32_sm.se_factors)
                _qfm_check_norm_invariants(m32_sm; tol=1f-4)
            end
            m32_boot = estimate_qfm(X32, r, 0.5f0; method=:smoothed_iqr, se=:bootstrap,
                                    n_boot=20, seed=8)
            @test m32_boot isa QuantileFactorModel{Float32}
            @test eltype(m32_boot.se_loadings) === Float32
            @test eltype(m32_boot.se_factors) === Float32

            # Validation errors
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; method=:bogus)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; kernel=:bogus)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; se=:bogus)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; bandwidth=:bogus)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; bandwidth=0.0)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; bandwidth=-0.5)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; bandwidth=true)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; hac_bandwidth=:bogus)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; hac_bandwidth=-1)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; n_boot=0)
            @test_throws ArgumentError estimate_qfm(X_base, r, 0.5; se=:bootstrap, n_boot=1)
        end
    end

    # =========================================================================
    # QFM-04 (#886): Rank-minimization & check-loss IC factor selection
    # =========================================================================
    @testset "Quantile Factor Selection & IC (QFM-04)" begin
        @testset "Location-scale DGP rank detection across quantiles" begin
            rng = MersenneTwister(8861)
            T_obs, N = 200, 80
            z = randn(rng, T_obs)
            f1 = 1.5 .* randn(rng, T_obs)
            f2 = exp.(0.6 .* z) .+ 0.3
            e = randn(rng, T_obs, N)

            # Two-loading location-scale DGP: quantile-specific rank (r=1 at tau=0.5, r=2 at tails)
            lam1 = 1.5 .* randn(rng, N)
            lam2 = 0.6 .+ 0.8 .* rand(rng, N)
            X_ls = f1 * lam1' .+ (f2 * lam2') .* e

            ic_low = ic_criteria_qfm(X_ls, 5, 0.1)
            ic_med = ic_criteria_qfm(X_ls, 5, 0.5)
            ic_high = ic_criteria_qfm(X_ls, 5, 0.9)

            @test ic_low isa QFMICResult{Float64}
            @test ic_low.tau == 0.1
            @test ic_low.r_max == 5
            @test length(ic_low.sigma_sq) == 5
            @test length(ic_low.loss) == 6
            @test length(ic_low.IC1) == 6
            @test length(ic_low.IC2) == 6
            @test length(ic_low.IC3) == 6
            @test issorted(ic_low.loss; rev=true)
            @test ic_low.r_rank == 2
            @test ic_med.r_rank == 1
            @test ic_high.r_rank == 2
            @test ic_med.r_IC1 == 1
            @test ic_med.r_IC2 == 1
            @test ic_low.r_IC1 >= 2
            @test ic_high.r_IC1 >= 2

            # Heavy-tailed t(3) location-scale DGP: rank=1 at median, rank=2 at tail quantiles
            rng2 = MersenneTwister(8862)
            T2, N2 = 200, 80
            f1_t = 1.2 .* randn(rng2, T2)
            f2_t = exp.(0.6 .* randn(rng2, T2)) .+ 0.3
            lam1_t = randn(rng2, N2)
            lam2_t = 0.7 .+ 1.1 .* rand(rng2, N2)
            X_t3 = f1_t * lam1_t' .+ (f2_t * lam2_t') .* (0.5 .* rand(rng2, TDist(3), T2, N2))

            ic_t_low = ic_criteria_qfm(X_t3, 5, 0.1)
            ic_t_med = ic_criteria_qfm(X_t3, 5, 0.5)
            ic_t_high = ic_criteria_qfm(X_t3, 5, 0.9)
            @test ic_t_low.r_rank == 2
            @test ic_t_med.r_rank == 1
            @test ic_t_high.r_rank == 2
        end

        @testset "Pure location DGP and zero-factor noise panel" begin
            rng = MersenneTwister(8863)
            T_obs, N = 150, 60
            F_true = randn(rng, T_obs, 2)
            L_true = randn(rng, N, 2)
            X_loc = F_true * L_true' .+ 0.35 .* randn(rng, T_obs, N)

            ic_loc = ic_criteria_qfm(X_loc, 5, 0.5)
            @test ic_loc.r_rank == 2
            @test ic_loc.r_IC1 == 2
            @test ic_loc.r_IC2 == 2
            @test ic_loc.r_IC3 == 2

            ic_sm = ic_criteria_qfm(X_loc, 4, 0.5; method=:smoothed_iqr)
            @test ic_sm.r_rank == 2
            @test ic_sm.r_IC2 == 2

            # Pure Gaussian noise: Bai-Ng check-loss IC selects r=0
            X_noise = randn(rng, T_obs, N)
            ic_noise = ic_criteria_qfm(X_noise, 4, 0.5; warn_boundary=false)
            @test ic_noise.r_IC1 == 0
            @test ic_noise.r_IC2 == 0
            @test ic_noise.r_IC3 == 0
            @test ic_noise.loss[1] >= ic_noise.loss[end]
        end

        @testset "Threshold scale, boundary warnings, and :auto dispatch" begin
            rng = MersenneTwister(8864)
            T_obs, N = 120, 50
            F3 = randn(rng, T_obs, 3)
            L3 = randn(rng, N, 3)
            X3 = F3 * L3' .+ 0.3 .* randn(rng, T_obs, N)

            # Upper boundary r_max < r_true emits @warn unless warn_boundary=false
            ic_bnd = @test_logs (:warn, r"boundary|r_max") ic_criteria_qfm(X3, 2, 0.5)
            @test ic_bnd.r_rank == 2 || ic_bnd.r_IC1 == 2
            ic_nowarn = @test_nowarn ic_criteria_qfm(X3, 2, 0.5; warn_boundary=false)
            @test ic_nowarn.r_rank == ic_bnd.r_rank

            # threshold_scale proportionality & monotonicity
            ic_s1 = ic_criteria_qfm(X3, 5, 0.5; threshold_scale=1.0)
            ic_s2 = ic_criteria_qfm(X3, 5, 0.5; threshold_scale=2.0)
            ic_big = ic_criteria_qfm(X3, 5, 0.5; threshold_scale=100.0)
            @test ic_s2.threshold_PNT ≈ 2.0 * ic_s1.threshold_PNT rtol=1e-12
            @test ic_s2.r_rank <= ic_s1.r_rank
            @test ic_big.r_rank == 0

            # estimate_qfm(X, :auto, tau; ...) across criteria and kwarg forwarding
            qfm_auto_rank = estimate_qfm(X3, :auto, 0.5; r_max=5, criterion=:rank)
            @test qfm_auto_rank isa QuantileFactorModel{Float64}
            @test qfm_auto_rank.r == 3
            @test qfm_auto_rank.tau == 0.5

            qfm_auto_ic2 = estimate_qfm(X3, :auto, 0.5; r_max=5, criterion=:ic2,
                                        method=:smoothed_iqr, se=:asymptotic)
            @test qfm_auto_ic2.r == 3
            @test qfm_auto_ic2.method == :smoothed_iqr
            @test !isempty(qfm_auto_ic2.se_loadings)
            @test !isempty(qfm_auto_ic2.se_factors)
            @test size(qfm_auto_ic2.se_loadings) == (N, 3)
            @test size(qfm_auto_ic2.se_factors) == (T_obs, 3)

            for crit in (:ic1, :ic3)
                q_c = estimate_qfm(X3, :auto, 0.5; r_max=5, criterion=crit)
                @test q_c.r == 3
            end

            # Zero-factor clamp fallback warning on pure noise with criterion=:ic2
            X_noise = randn(rng, 100, 40)
            q_zero = @test_logs (:warn, r"clamping to r=1") estimate_qfm(
                X_noise, :auto, 0.5; r_max=3, criterion=:ic2, warn_boundary=false,
            )
            @test q_zero.r == 1
        end

        @testset "RSER-01 constructor, display, Float32, fallback, and validation" begin
            rng = MersenneTwister(8865)
            T_obs, N = 100, 40
            F2 = randn(rng, T_obs, 2)
            L2 = randn(rng, N, 2)
            X = F2 * L2' .+ 0.35 .* randn(rng, T_obs, N)

            ic = ic_criteria_qfm(X, 4, 0.25)

            # Default positional constructor QFMICResult{T}(fields...) matching fieldnames order
            ic_copy = QFMICResult{Float64}(
                ic.tau, ic.r_max, ic.sigma_sq, ic.threshold_PNT, ic.r_rank,
                ic.loss, ic.IC1, ic.IC2, ic.IC3, ic.r_IC1, ic.r_IC2, ic.r_IC3,
            )
            @test ic_copy.tau == ic.tau
            @test ic_copy.r_rank == ic.r_rank
            @test ic_copy.IC2 == ic.IC2

            # Display backends & report
            for backend in (:text, :latex, :html)
                MacroEconometricModels.with_display_backend(backend) do
                    s = sprint(show, ic)
                    @test !isempty(s)
                    s_rep = sprint(report, ic)
                    @test !isempty(s_rep)
                end
            end

            # Float32 precision preservation
            X32 = Float32.(X)
            ic32 = ic_criteria_qfm(X32, 4, 0.25f0)
            @test ic32 isa QFMICResult{Float32}
            @test eltype(ic32.sigma_sq) === Float32
            @test eltype(ic32.IC1) === Float32
            @test typeof(ic32.threshold_PNT) === Float32
            @test ic32.r_rank == 2

            qfm32 = estimate_qfm(X32, :auto, 0.25f0; r_max=4)
            @test qfm32 isa QuantileFactorModel{Float32}
            @test qfm32.r == 2

            # Integer input @float_fallback
            X_int = round.(Int, 10 .* X)
            ic_int = ic_criteria_qfm(X_int, 4, 0.5)
            @test ic_int isa QFMICResult{Float64}

            # Validation errors
            @test_throws ArgumentError ic_criteria_qfm(X, 0, 0.5)
            @test_throws ArgumentError ic_criteria_qfm(X, N + 1, 0.5)
            @test_throws ArgumentError ic_criteria_qfm(X, 4, 0.0)
            @test_throws ArgumentError ic_criteria_qfm(X, 4, 1.0)
            @test_throws ArgumentError ic_criteria_qfm(X, 4, 0.5; threshold_scale=0.0)
            @test_throws ArgumentError ic_criteria_qfm(X, 4, 0.5; threshold_scale=-1.0)
            @test_throws ArgumentError ic_criteria_qfm(X, 4, 0.5; method=:invalid_method)
            @test_throws ArgumentError estimate_qfm(X, :bogus, 0.5)
            @test_throws ArgumentError estimate_qfm(X, :auto, 0.5; criterion=:invalid_crit)
        end
    end

    @testset "No method ambiguities in MacroEconometricModels" begin
        @test isempty(Test.detect_ambiguities(MacroEconometricModels))
    end
end
