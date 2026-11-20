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

@testset "FHLZ One-Sided Infinite-r GDFM Tests" begin

    function make_infinite_r_panel(; N::Int=60, T_obs::Int=300, q::Int=2,
                                   idio_sd::Float64=0.3, burn::Int=100, seed::Int=2026)
        rng = Random.Xoshiro(seed)
        alphas = -0.75 .+ 1.5 .* rand(rng, N)
        B_load = randn(rng, N, q)
        B_load[1:2, :] .= [1.0 0.0; 0.5 1.0]
        chi = zeros(T_obs + burn, N)
        shocks = randn(rng, T_obs + burn, q)
        @inbounds for t in 1:(T_obs + burn), i in 1:N
            prev = t > 1 ? alphas[i] * chi[t - 1, i] : 0.0
            chi[t, i] = prev + dot(B_load[i, :], shocks[t, :])
        end
        X = chi[(burn + 1):end, :] .+ idio_sd .* randn(rng, T_obs, N)
        (X=X, chi=chi[(burn + 1):end, :], shocks=shocks[(burn + 1):end, :],
         alphas=alphas, B_load=B_load)
    end

    function align_irf_cols(est::Array{T,3}, truth::Array{T,3}) where {T}
        out = copy(est)
        @inbounds for j in 1:size(est, 3)
            if dot(@view(est[1, :, j]), @view(truth[1, :, j])) < 0
                out[:, :, j] .*= -one(T)
            end
        end
        out
    end

    # =========================================================================
    # 1. Standalone FHLZModel Construction & API
    # =========================================================================

    @testset "FHLZModel construction, dimensions, and StatsAPI" begin
        rng = Random.Xoshiro(101)
        d = dgp_dynamic_factors(rng; A=0.5 * Matrix{Float64}(I, 2, 2), N=20, T=150, idio_sd=0.3)
        X = d.X
        names = ["v$i" for i in 1:20]

        m = estimate_gdfm_fhlz(X, 2; p_max=4, n_perm=15, irf_horizon=30,
                               varnames=names, seed=7)

        @test m isa FHLZModel{Float64}
        @test m isa AbstractFactorModel
        @test m.q == 2
        @test m.n_perm == 15
        @test m.kernel === :bartlett
        @test m.standardized == true
        @test m.varnames == names
        @test m.manifest isa ReproManifest
        @test m.manifest.seed == 7

        T_eff = 150 - 4
        @test size(m.X) == (150, 20)
        @test size(m.common_component) == (T_eff, 20)
        @test size(m.idiosyncratic) == (T_eff, 20)
        @test size(m.innovations) == (T_eff, 2)
        @test size(m.impulse_filters) == (30, 20, 2)
        @test size(m.p_blocks) == (15, div(20, 3))
        @test all(1 .<= m.p_blocks .<= 4)
        @test length(m.variance_explained) == 2
        @test all(0 .<= m.variance_explained .<= 1)

        # Orthonormal innovations and additive decomposition
        U = m.innovations
        @test (U' * U) / T_eff ≈ Matrix{Float64}(I, 2, 2) atol=1e-10
        @test m.common_component + m.idiosyncratic ≈ X[5:end, :] atol=1e-12

        # StatsAPI & utilities
        @test nobs(m) == 150
        @test effective_nobs(m) == T_eff
        @test predict(m) === m.common_component
        @test residuals(m) === m.idiosyncratic
        @test dof(m) > 0
        @test length(r2(m)) == 20
        @test length(common_variance_share(m)) == 20
        @test all(0 .<= common_variance_share(m) .<= 1.1)

        shown = sprint(show, m)
        @test occursin("FHLZ One-Sided Infinite-r GDFM", shown)
        @test occursin("Variance Explained", shown)
        rpt = sprint(report, m)
        @test occursin("FHLZ", rpt)
    end

    @testset "Kernels, unstandardized, and MA reconstruction" begin
        rng = Random.Xoshiro(202)
        d = dgp_dynamic_factors(rng; A=0.45 * Matrix{Float64}(I, 2, 2), N=15, T=140, idio_sd=0.25)
        X = d.X

        for kern in (:bartlett, :parzen, :tukey)
            mk = estimate_gdfm_fhlz(X, 2; kernel=kern, bandwidth=8, n_perm=8, seed=11)
            @test mk.kernel === kern
            @test mk.bandwidth == 8
            @test all(isfinite, mk.common_component)
        end

        m_raw = estimate_gdfm_fhlz(X, 2; standardize=false, n_perm=10, irf_horizon=40, seed=12)
        @test m_raw.standardized == false
        @test size(m_raw.impulse_filters) == (40, 15, 2)
        @test all(isfinite, m_raw.common_component)

        # Check raw-unit identity χ̂_t ≈ μ + Σ B_h u_{t-h} on standardized fit
        m_std = estimate_gdfm_fhlz(X, 2; standardize=true, n_perm=10, irf_horizon=40, seed=12)
        μ = vec(mean(X; dims=1))
        T_eff = size(m_std.common_component, 1)
        recon = repeat(μ', T_eff, 1)
        for t in 1:T_eff, h in 0:min(t - 1, 39)
            recon[t, :] .+= m_std.impulse_filters[h + 1, :, :] * m_std.innovations[t - h, :]
        end
        @test mean([cor(m_std.common_component[:, i], recon[:, i]) for i in 1:15]) > 0.99
    end

    @testset "Input validation errors" begin
        X = randn(Random.Xoshiro(3), 40, 8)
        @test_throws ArgumentError estimate_gdfm_fhlz(X, 0)
        @test_throws ArgumentError estimate_gdfm_fhlz(X, 9)
        @test_throws ArgumentError estimate_gdfm_fhlz(X, 2; kernel=:bad)
        @test_throws ArgumentError estimate_gdfm_fhlz(X, 2; lag_criterion=:hqic)
        @test_throws ArgumentError estimate_gdfm_fhlz(X, 2; p_max=0)
        @test_throws ArgumentError estimate_gdfm_fhlz(X, 2; n_perm=0)
        @test_throws ArgumentError estimate_gdfm_fhlz(X, 2; irf_horizon=0)
        @test_throws ArgumentError estimate_gdfm_fhlz(X, 2; varnames=["a"])
        Xnan = copy(X); Xnan[3, 2] = NaN
        @test_throws ArgumentError estimate_gdfm_fhlz(Xnan, 2)
        Xshort = randn(Random.Xoshiro(4), 4, 8)
        @test_throws ArgumentError estimate_gdfm_fhlz(Xshort, 3; p_max=5)
    end

    # =========================================================================
    # 2. Criterion 1: Strict One-Sidedness
    # =========================================================================

    @testset "Strict one-sidedness under fixed causal filter" begin
        rng = Random.Xoshiro(314)
        X = randn(rng, 120, 15)
        q = 2
        p_max = 5
        μ = vec(mean(X; dims=1))
        σ = max.(vec(std(X; dims=1)), 1e-10)
        X_proc = (X .- μ') ./ σ'

        M_bw = MacroEconometricModels._select_lag_window(size(X, 1))
        freq, Sx = MacroEconometricModels._fhlz_lag_window_spectrum(X_proc, M_bw, :bartlett, p_max)
        ev, evec = MacroEconometricModels._spectral_eigendecomposition(Sx)
        Schi = MacroEconometricModels._compute_common_spectral_density(evec[:, 1:q, :], max.(ev[1:q, :], 0.0))
        Gpos = [MacroEconometricModels._gamma_from_spectrum(Schi, freq, k) for k in 0:p_max]

        perm_res = MacroEconometricModels._fhlz_fit_permutation(
            X_proc, Gpos, q, p_max, :bic, 20, UInt64(98765))

        chi_full, u_full = MacroEconometricModels._fhlz_apply_causal_filter(
            X_proc, perm_res.blocks, perm_res.Phi_blocks,
            perm_res.V_q, perm_res.W_whiten, perm_res.R_m, p_max)
        chi_trunc, u_trunc = MacroEconometricModels._fhlz_apply_causal_filter(
            X_proc[1:(end - 5), :], perm_res.blocks, perm_res.Phi_blocks,
            perm_res.V_q, perm_res.W_whiten, perm_res.R_m, p_max)

        T_common = size(chi_trunc, 1)
        @test size(chi_full, 1) - T_common == 5
        @test maximum(abs.(chi_full[1:T_common, :] .- chi_trunc)) < 1e-12
        @test maximum(abs.(u_full[1:T_common, :] .- u_trunc)) < 1e-12
    end

    # =========================================================================
    # 3. Criterion 2: Infinite-r Benchmark (FHLZ vs finite-r FGLR)
    # =========================================================================

    @testset "Infinite-r benchmark: FHLZ beats finite-r FGLR (r=6)" begin
        d = make_infinite_r_panel(; N=60, T_obs=300, q=2, idio_sd=0.3, burn=100, seed=2026)
        X = d.X
        H_eval = 15
        N, q = size(X, 2), 2
        true_irf = [((d.alphas[i]^(h - 1)) * d.B_load[i, j]) for h in 1:H_eval, i in 1:N, j in 1:q]

        sdfm_fhlz = estimate_structural_dfm(X, q; method=:fhlz, identification=:cholesky,
            order=[1, 2], H=H_eval, standardize=true, seed=42)
        sdfm_fglr = estimate_structural_dfm(X, q; method=:fglr, r=6, identification=:cholesky,
            order=[1, 2], p=1, H=H_eval, standardize=true)

        est_fhlz = align_irf_cols(irf(sdfm_fhlz, H_eval).values, true_irf)
        est_fglr = align_irf_cols(irf(sdfm_fglr, H_eval).values, true_irf)

        rmse_fhlz = sqrt(mean(abs2, est_fhlz .- true_irf))
        rmse_fglr = sqrt(mean(abs2, est_fglr .- true_irf))
        @test rmse_fhlz < rmse_fglr
        @test rmse_fhlz < 0.06
    end

    # =========================================================================
    # 4. Criterion 3: Thread Determinism & AIC vs BIC
    # =========================================================================

    @testset "Thread determinism (threaded=true vs false)" begin
        X = randn(Random.Xoshiro(512), 140, 20)
        mt = estimate_gdfm_fhlz(X, 2; n_perm=16, threaded=true, seed=42)
        ms = estimate_gdfm_fhlz(X, 2; n_perm=16, threaded=false, seed=42)

        @test mt.p_blocks == ms.p_blocks
        @test maximum(abs.(mt.common_component .- ms.common_component)) < 1e-12
        @test maximum(abs.(mt.idiosyncratic .- ms.idiosyncratic)) < 1e-12
        @test maximum(abs.(mt.innovations .- ms.innovations)) < 1e-12
        @test maximum(abs.(mt.impulse_filters .- ms.impulse_filters)) < 1e-12
    end

    @testset "AIC vs BIC lag selection" begin
        d = make_infinite_r_panel(; N=30, T_obs=200, q=2, idio_sd=0.25, seed=909)
        m_bic = estimate_gdfm_fhlz(d.X, 2; p_max=5, lag_criterion=:bic, n_perm=12, seed=42)
        m_aic = estimate_gdfm_fhlz(d.X, 2; p_max=5, lag_criterion=:aic, n_perm=12, seed=42)

        @test all(m_bic.p_blocks .<= m_aic.p_blocks)
        @test mean(m_aic.p_blocks) >= mean(m_bic.p_blocks)
    end

    # =========================================================================
    # 5. Criterion 4: Float32 & Full StructuralDFM Workflow
    # =========================================================================

    @testset "Float32 precision support" begin
        X32 = randn(Random.Xoshiro(3232), Float32, 100, 12)
        m32 = estimate_gdfm_fhlz(X32, 2; n_perm=8, irf_horizon=20, seed=11)
        @test m32 isa FHLZModel{Float32}
        @test eltype(m32.common_component) === Float32
        @test eltype(m32.innovations) === Float32
        @test eltype(m32.impulse_filters) === Float32
        @test all(isfinite, m32.common_component)
        @test all(isfinite, m32.impulse_filters)

        sdfm32 = estimate_structural_dfm(X32, 2; method=:fhlz, n_perm=8, H=10, seed=11)
        @test sdfm32 isa StructuralDFM{Float32}
        @test irf(sdfm32, 10) isa ImpulseResponse{Float32}
        @test fevd(sdfm32, 10; space=:panel) isa FEVD{Float32}
        @test historical_decomposition(sdfm32) isa HistoricalDecomposition{Float32}
        @test forecast(sdfm32, 4) isa FactorForecast{Float32}
    end

    @testset "Full estimate_structural_dfm(method=:fhlz) workflow" begin
        d = make_infinite_r_panel(; N=24, T_obs=220, q=2, idio_sd=0.25, seed=777)
        X = d.X
        T_obs, N = size(X)
        q = 2
        names = ["x$i" for i in 1:N]

        # Cholesky + downstream irf, fevd, HD, forecast, report
        sdfm_chol = estimate_structural_dfm(X, q; method=:fhlz, identification=:cholesky,
            order=[1, 2], H=12, n_perm=15, varnames=names, seed=42)
        @test sdfm_chol isa StructuralDFM{Float64}
        @test sdfm_chol.method === :fhlz
        @test sdfm_chol.identification === :cholesky
        @test sdfm_chol.Q' * sdfm_chol.Q ≈ I(q) atol=1e-10
        @test is_stable(sdfm_chol)

        ir12 = irf(sdfm_chol, 12)
        @test size(ir12.values) == (12, N, q)
        @test abs(ir12.values[1, 1, 2]) < 1e-10
        @test irf(sdfm_chol, 80).horizon == 80
        @test sdfm_panel_irf(sdfm_chol, 12).values ≈ ir12.values atol=1e-12
        @test_throws ArgumentError irf(sdfm_chol, 10; method=:cholesky)

        irb = irf(sdfm_chol, 10; ci_type=:bootstrap, reps=25, rng=Random.Xoshiro(9))
        @test irb.ci_type === :bootstrap
        @test all(irb.ci_lower .<= irb.values .<= irb.ci_upper)

        fv_panel = fevd(sdfm_chol, 10; space=:panel, include_idiosyncratic=true)
        @test size(fv_panel.proportions) == (N, q + 1, 10)
        for h in 1:10, i in 1:N
            @test sum(fv_panel.proportions[i, :, h]) ≈ 1.0 atol=1e-10
        end
        fv_fac = fevd(sdfm_chol, 10; space=:factor)
        @test size(fv_fac.proportions) == (q, q, 10)

        hd = historical_decomposition(sdfm_chol)
        @test verify_decomposition(hd; tol=1e-8)
        @test hd.variables == names
        @test hd.shock_names == vcat(sdfm_chol.shock_names, ["Idiosyncratic"])

        fc = forecast(sdfm_chol, 6; ci_method=:none)
        fcb = forecast(sdfm_chol, 6; ci_method=:bootstrap, reps=25, rng=Random.Xoshiro(10))
        @test size(fc.observables) == (6, N)
        @test all(isfinite, fc.observables)
        @test fc.observables == fcb.observables
        @test all(fcb.observables_lower .<= fcb.observables .<= fcb.observables_upper)

        @test occursin("Structural DFM", sprint(report, sdfm_chol))
        @test occursin("Innovation", sprint(show, sdfm_chol))
        @test size(structural_shocks(sdfm_chol), 2) == q

        # Sign restrictions (panel space + store_all)
        sdfm_sign = estimate_structural_dfm(X, q; method=:fhlz, identification=:sign,
            sign_restrictions=[("x1", 1, 1:1, :positive)],
            store_all=true, H=10, n_perm=10, max_draws=2000,
            varnames=names, seed=13)
        @test sdfm_sign.identification === :sign
        @test sdfm_sign.identified_set !== nothing
        ir_set = irf(sdfm_sign, 10)
        @test ir_set.ci_type === :sign_set
        @test ir_set.values[1, 1, 1] > 0

        # Statistical ID methods: :fastica & :student_t
        for id_stat in (:fastica, :student_t)
            sdfm_st = estimate_structural_dfm(X, q; method=:fhlz, identification=id_stat,
                H=10, n_perm=10, varnames=names, seed=21)
            @test sdfm_st.identification === id_stat
            @test sdfm_st.Q' * sdfm_st.Q ≈ I(q) atol=1e-8
            @test size(irf(sdfm_st, 10).values) == (10, N, q)
        end

        # Max-share identification (time domain and frequency band)
        sdfm_ms = estimate_structural_dfm(X, q; method=:fhlz, identification=:max_share,
            H=12, n_perm=10, varnames=names, id_kwargs=(target="x1", horizons=0:10), seed=31)
        @test sdfm_ms.identification === :max_share
        @test sdfm_ms.Q' * sdfm_ms.Q ≈ I(q) atol=1e-8
        @test irf(sdfm_ms, 12).values[1, 1, 1] >= 0

        sdfm_ms_band = estimate_structural_dfm(X, q; method=:fhlz, identification=:max_share,
            H=12, n_perm=10, varnames=names, target=2, id_kwargs=(band=(0.1, 1.0),), seed=31)
        @test sdfm_ms_band.Q' * sdfm_ms_band.Q ≈ I(q) atol=1e-8

        # Long-run and proxy identification
        sdfm_lr = estimate_structural_dfm(X, q; method=:fhlz, identification=:long_run,
            target_vars=["x1", "x2"], H=12, n_perm=10, varnames=names, seed=41)
        @test sdfm_lr.Q' * sdfm_lr.Q ≈ I(q) atol=1e-8

        z = d.shocks[:, 1] .+ 0.3 .* randn(Random.Xoshiro(99), T_obs)
        sdfm_px = estimate_structural_dfm(X, q; method=:fhlz, identification=:proxy,
            instrument=z, normalize=("x1", 1.0), H=10, n_perm=10, varnames=names, seed=51)
        @test sdfm_px.identification === :proxy
        @test isfinite(sdfm_px.first_stage_F)
        @test irf(sdfm_px, 10).values[1, 1, 1] ≈ 1.0 atol=1e-10
    end

    # =========================================================================
    # 6. EM Missing-Data / Ragged-Edge Support (#899)
    # =========================================================================

    @testset "EM-FHLZ missing-data & ragged-edge recovery (#899)" begin
        rng = Random.Xoshiro(89911)
        T_obs, N, q = 250, 40, 2
        d = dgp_dynamic_factors(rng; A=[0.7 0.1; 0.05 0.65], N=N, T=T_obs, idio_sd=0.3)
        X = d.X
        chi_true = d.F * d.Lambda'

        X_nan = copy(X)
        for t in 1:(T_obs - 1), j in 1:N
            rand(rng) < 0.10 && (X_nan[t, j] = NaN)
        end
        tail_idx = sort(randperm(rng, N)[1:round(Int, 0.30 * N)])
        X_nan[T_obs, tail_idx] .= NaN
        miss_mask = isnan.(X_nan)
        obs_mask = .!miss_mask

        # Default allow_missing=false rejects NaN and Inf
        @test_throws ArgumentError estimate_gdfm_fhlz(X_nan, q; n_perm=8, seed=42)
        X_inf = copy(X); X_inf[3, 4] = Inf
        @test_throws ArgumentError estimate_gdfm_fhlz(X_inf, q; allow_missing=true, n_perm=8, seed=42)

        # All-NaN row/column and invalid EM kwargs
        X_bad_row = copy(X); X_bad_row[7, :] .= NaN
        @test_throws ArgumentError estimate_gdfm_fhlz(X_bad_row, q; allow_missing=true, n_perm=4, seed=42)
        X_bad_col = copy(X); X_bad_col[:, 5] .= NaN
        @test_throws ArgumentError estimate_gdfm_fhlz(X_bad_col, q; allow_missing=true, n_perm=4, seed=42)
        @test_throws ArgumentError estimate_gdfm_fhlz(X_nan, q; allow_missing=true, em_max_iter=0, n_perm=4, seed=42)
        @test_throws ArgumentError estimate_gdfm_fhlz(X_nan, q; allow_missing=true, em_tol=0.0, n_perm=4, seed=42)

        m = estimate_gdfm_fhlz(X_nan, q; allow_missing=true, n_perm=8, seed=42)
        @test m isa FHLZModel{Float64}
        @test all(isfinite, m.X)
        @test all(isfinite, m.common_component)
        @test all(isfinite, m.idiosyncratic)
        @test all(isfinite, m.innovations)
        @test all(isfinite, m.impulse_filters)
        @test m.X[obs_mask] ≈ X[obs_mask]

        offset = T_obs - size(m.common_component, 1)
        @test m.common_component + m.idiosyncratic ≈ m.X[(offset + 1):end, :] atol=1e-10
        miss_eff = miss_mask[(offset + 1):end, :]
        @test cor(m.common_component[miss_eff], chi_true[(offset + 1):end, :][miss_eff]) > 0.85

        # Thread determinism under allow_missing=true
        m_seq = estimate_gdfm_fhlz(X_nan, q; allow_missing=true, n_perm=8, threaded=false, seed=42)
        @test m.common_component ≈ m_seq.common_component atol=1e-10
        @test m.X ≈ m_seq.X atol=1e-10

        # Complete data with allow_missing=true matches allow_missing=false
        m_c1 = estimate_gdfm_fhlz(X, q; allow_missing=false, n_perm=6, seed=9)
        m_c2 = estimate_gdfm_fhlz(X, q; allow_missing=true, n_perm=6, seed=9)
        @test m_c1.common_component ≈ m_c2.common_component atol=1e-12

        # Float32 precision support under allow_missing=true
        m32 = estimate_gdfm_fhlz(Float32.(X_nan), q; allow_missing=true, n_perm=4, seed=42)
        @test m32 isa FHLZModel{Float32}
        @test all(isfinite, m32.X)
        @test all(isfinite, m32.common_component)
    end

    # =========================================================================
    # 7. dgp_infinite_gdfm Recovery & Dynamic Commonality (#900)
    # =========================================================================

    @testset "dgp_infinite_gdfm recovery & dynamic commonality (#900)" begin
        rng = Random.Xoshiro(9001)
        d1 = dgp_infinite_gdfm(rng; T_obs=300, N=50, q=2, a_coef=0.6, b_coef=0.5,
                               signal_share=0.8)
        d2 = dgp_infinite_gdfm(Random.Xoshiro(9001); T_obs=300, N=50, q=2, a_coef=0.6,
                               b_coef=0.5, signal_share=0.8)
        d3 = dgp_infinite_gdfm(; T_obs=300, N=50, q=2, a_coef=0.6, b_coef=0.5,
                               signal_share=0.8, seed=9001)

        @test d1.X == d2.X
        @test d1.X == d3.X
        @test d1.common == d2.common
        @test d1.irf_true == d2.irf_true
        @test size(d1.X) == (300, 50)
        @test size(d1.common) == (300, 50)
        @test size(d1.idiosyncratic) == (300, 50)
        @test size(d1.innovations) == (300, 2)
        @test d1.q == 2
        @test d1.n_permanent == 0
        @test size(d1.irf_true) == (41, 50, 2)
        @test size(d1.spectral_density) == (50, 50, 33)
        @test d1.X ≈ d1.common + d1.idiosyncratic atol=1e-12

        # Signal share matches target 0.8 (finite-sample tolerance across T=300, N=50)
        emp_share = 1.0 - sum(var(d1.idiosyncratic; dims=1)) / sum(var(d1.X; dims=1))
        @test emp_share ≈ 0.8 atol=0.1

        # Exact rational IRF identity and spectral dynamic commonality (q-vs-(q+1) gap)
        @test d1.irf_true[1, :, :] ≈ d1.b
        @test d1.irf_true[2, :, :] ≈ d1.b .* d1.alpha
        for h in 1:size(d1.spectral_density, 3)
            ev = sort(real.(eigvals(Hermitian(d1.spectral_density[:, :, h]))); rev=true)
            @test ev[2] > 2.5 * ev[3]
        end

        # FHLZ recovers infinite-dimensional common component & dynamic IRFs up to rotation
        m_fhlz = estimate_gdfm_fhlz(d1.X, 2; p_max=4, n_perm=15, irf_horizon=20, seed=42)
        offset = size(d1.X, 1) - size(m_fhlz.common_component, 1)
        chi_true_eff = d1.common[(offset + 1):end, :]
        cor_mean = mean([abs(cor(m_fhlz.common_component[:, i], chi_true_eff[:, i])) for i in 1:50])
        @test cor_mean > 0.80

        # Procrustes-aligned IRF recovery across horizons h = 0..10
        H_eval = 10
        B_est = m_fhlz.impulse_filters[1:H_eval, :, :]
        B_true = d1.irf_true[1:H_eval, :, :]
        M_cross = reshape(B_est, H_eval * 50, 2)' * reshape(B_true, H_eval * 50, 2)
        sv = svd(M_cross)
        O_rot = sv.U * sv.Vt
        B_aligned = similar(B_est)
        for h in 1:H_eval
            B_aligned[h, :, :] = B_est[h, :, :] * O_rot
        end
        @test cor(vec(B_aligned), vec(B_true)) > 0.85

        # AR(1) idiosyncratic noise option
        d_ar = dgp_infinite_gdfm(Random.Xoshiro(9003); T_obs=300, N=10, q=1, idio_ar=0.6)
        mean_ar1 = mean([cor(d_ar.idiosyncratic[1:end-1, j], d_ar.idiosyncratic[2:end, j]) for j in 1:10])
        @test mean_ar1 ≈ 0.6 atol=0.15

        # Validation errors
        @test_throws ArgumentError dgp_infinite_gdfm(Random.Xoshiro(1); T_obs=1)
        @test_throws ArgumentError dgp_infinite_gdfm(Random.Xoshiro(1); q=0)
        @test_throws ArgumentError dgp_infinite_gdfm(Random.Xoshiro(1); q=2, n_permanent=3)
        @test_throws ArgumentError dgp_infinite_gdfm(Random.Xoshiro(1); a_coef=1.0)
        @test_throws ArgumentError dgp_infinite_gdfm(Random.Xoshiro(1); signal_share=0.0)
        @test_throws ArgumentError dgp_infinite_gdfm(Random.Xoshiro(1); idio_ar=1.0)
    end
end
