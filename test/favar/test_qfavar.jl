# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

using Test
using LinearAlgebra
using Statistics
using Random
using StatsAPI
using MacroEconometricModels

const M_QF = MacroEconometricModels

@testset "Quantile VAR & Quantile FAVAR (QFM-08 / #890)" begin

    # =========================================================================
    # 1. Quantile VAR (Chavleishvili & Manganelli 2024) & qirf
    # =========================================================================

    @testset "QuantileVARModel estimation, ordering, SEs, and StatsAPI" begin
        rng = Random.Xoshiro(89001)
        T_obs, n, p = 240, 3, 2
        A1 = [0.5 0.1 0.0; 0.2 0.4 0.05; 0.0 0.15 0.5]
        A2 = [0.1 0.0 0.0; 0.0 0.1 0.0; 0.0 0.0 0.1]
        B_true = [1.0 0.0 0.0; 0.4 1.0 0.0; -0.3 0.5 1.0]
        Y = zeros(T_obs + 40, n)
        for t in 3:(T_obs + 40)
            Y[t, :] = A1 * Y[t - 1, :] + A2 * Y[t - 2, :] + B_true * randn(rng, n)
        end
        Y = Y[41:end, :]

        qv = estimate_qvar(Y, p, [0.10, 0.50, 0.90]; se=:asymptotic, varnames=["y1", "y2", "y3"], seed=11)
        @test qv isa QuantileVARModel{Float64}
        @test qv isa AbstractVARModel
        @test nvars(qv) == 3
        @test nlags(qv) == 2
        @test ncoefs(qv) == 1 + 3 * 2
        @test effective_nobs(qv) == T_obs - p
        @test varnames(qv) == ["y1", "y2", "y3"]
        @test StatsAPI.nobs(qv) == T_obs
        @test StatsAPI.islinear(qv) === false
        @test StatsAPI.dof(qv) > 0
        @test StatsAPI.dof_residual(qv) > 0
        @test size(StatsAPI.coef(qv)) == (7, 3, 3)
        @test size(StatsAPI.stderror(qv)) == (7, 3, 3)
        @test size(StatsAPI.residuals(qv)) == (T_obs - p, 3, 3)
        @test all(isfinite, StatsAPI.coef(qv))
        @test all(>(0), StatsAPI.stderror(qv))
        @test qv.manifest isa ReproManifest
        @test qv.manifest.seed == 11

        # StatsAPI.vcov and StatsAPI.confint (multi-tau)
        V_multi = StatsAPI.vcov(qv)
        @test V_multi isa Vector{Matrix{Float64}}
        @test length(V_multi) == 3
        @test size(V_multi[1]) == (7 * 3, 7 * 3)
        @test isdiag(V_multi[1])
        @test diag(V_multi[1]) ≈ vec(qv.se_list[1]) .^ 2
        ci95 = StatsAPI.confint(qv)
        ci90 = StatsAPI.confint(qv; level=0.90)
        @test size(ci95) == (7 * 3, 2, 3)
        @test size(ci90) == (7 * 3, 2, 3)
        for m in 1:3
            @test all(ci95[:, 1, m] .<= vec(qv.B_list[m]) .<= ci95[:, 2, m])
            @test all(ci95[:, 2, m] .- ci95[:, 1, m] .> ci90[:, 2, m] .- ci90[:, 1, m])
        end
        @test_throws ArgumentError StatsAPI.confint(qv; level=0.0)
        @test_throws ArgumentError StatsAPI.confint(qv; level=1.0)
        @test_throws ArgumentError StatsAPI.confint(qv; level=true)

        # Single-tau vcov/confint & Float32
        qv_1 = estimate_qvar(Y, p, [0.50]; se=:asymptotic)
        V_1 = StatsAPI.vcov(qv_1)
        ci_1 = StatsAPI.confint(qv_1)
        @test V_1 isa Matrix{Float64}
        @test size(V_1) == (7 * 3, 7 * 3)
        @test diag(V_1) ≈ vec(qv_1.se_list[1]) .^ 2
        @test ci_1 isa Matrix{Float64}
        @test size(ci_1) == (7 * 3, 2)
        @test all(ci_1[:, 1] .<= vec(qv_1.B_list[1]) .<= ci_1[:, 2])

        qv_f32 = estimate_qvar(Float32.(Y), 1, Float32[0.50]; se=:asymptotic)
        @test StatsAPI.vcov(qv_f32) isa Matrix{Float32}
        @test StatsAPI.confint(qv_f32) isa Matrix{Float32}

        # Exact structural equation identity: Y_eff * A0' - X_lag * B ≈ resid
        Y_eff, X_lag = construct_var_matrices(Y, p)
        for m in 1:3
            @test Y_eff * qv.A0_list[m]' - X_lag * qv.B_list[m] ≈ qv.resid_list[m] atol=1e-11
            @test diag(qv.A0_list[m]) ≈ ones(n)
            # Default ordering 1:3 -> A0 is unit lower triangular
            @test norm(triu(qv.A0_list[m], 1)) == 0.0
        end

        # Non-identity recursive ordering [3, 1, 2]
        ord = [3, 1, 2]
        qv_ord = estimate_qvar(Y, p, [0.50]; ordering=ord, se=:none)
        @test qv_ord.ordering == ord
        @test size(StatsAPI.coef(qv_ord)) == (7, 3)
        @test all(==(0), StatsAPI.stderror(qv_ord))
        A0_perm = qv_ord.A0_list[1][ord, ord]
        @test diag(A0_perm) ≈ ones(n)
        @test norm(triu(A0_perm, 1)) == 0.0
        @test Y_eff * qv_ord.A0_list[1]' - X_lag * qv_ord.B_list[1] ≈ qv_ord.resid_list[1] atol=1e-11

        # Bootstrap SEs & seed reproducibility
        qv_b1 = estimate_qvar(Y, 1, [0.50]; se=:bootstrap, n_boot=30, seed=77)
        qv_b2 = estimate_qvar(Y, 1, [0.50]; se=:bootstrap, n_boot=30, seed=77)
        @test qv_b1.se_list == qv_b2.se_list
        @test all(>(0), qv_b1.se_list[1])

        # Median QVAR IRF vs OLS VAR unit-effect IRF on Gaussian linear VAR
        ols_var = estimate_var(Y, p)
        ols_irf = irf(ols_var, 10; method=:cholesky, normalize=:unit_effect)
        q_irf_ana = qirf(qv, 10; tau_profile=0.50, method=:analytical)
        q_irf_sim = qirf(qv, 10; tau_profile=0.50, method=:simulation, n_boot=300, seed=12)
        @test q_irf_ana isa QuantileImpulseResponse{Float64}
        @test size(q_irf_ana.values) == (10, 3, 3)
        @test q_irf_ana.values[1, :, :] ≈ inv(qv.A0_list[2]) atol=1e-12
        @test cor(vec(q_irf_ana.values), vec(ols_irf.values)) > 0.97
        @test cor(vec(q_irf_sim.values), vec(ols_irf.values)) > 0.95

        # Vector tau_profile, single shock, shock_size, and bootstrap CIs
        q_irf_vec = qirf(qv, 8; shock=2, shock_size=2.0, tau_profile=[0.10, 0.25],
                         baseline_tau=0.50, ci_type=:bootstrap, n_boot=25, seed=99)
        @test size(q_irf_vec.values) == (8, 3, 1)
        @test q_irf_vec.values[1, 2, 1] ≈ 2.0 atol=1e-12
        @test q_irf_vec.tau_profile ≈ [0.10, 0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        @test q_irf_vec.ci_type === :bootstrap
        @test all(q_irf_vec.ci_lower .<= q_irf_vec.ci_upper)
    end

    # =========================================================================
    # 2. Acceptance Criterion 1: Symmetric Gaussian Linear FAVAR (N=50, T=300, r=2)
    # =========================================================================

    @testset "AC1: Symmetric Gaussian FAVAR τ=0.50 vs favar_panel_irf (> 0.92 corr)" begin
        rng = Random.Xoshiro(89002)
        N, T_obs, r, p, H = 50, 300, 2, 1, 12
        d = dgp_dynamic_factors(rng; A=[0.6 0.1; 0.05 0.55], N=N, T=T_obs, idio_sd=0.4)
        X = d.X
        key_idx = [1]

        favar = estimate_favar(X, key_idx, r, p)
        ols_panel = favar_panel_irf(favar, irf(favar, H))

        qf_2step = estimate_qfavar(X, key_idx, r, p, [0.10, 0.50, 0.90];
                                   method=:twostep, standardize=true, seed=42)
        @test qf_2step isa QuantileFAVAR{Float64}
        qir_2step = qfavar_panel_irf(qf_2step, H; tau_profile=0.50, method=:analytical)
        @test qir_2step isa QuantileImpulseResponse{Float64}
        @test size(qir_2step.values) == (H, N, r + 1)

        qf_vb = estimate_qfavar(X, key_idx, r, p, [0.10, 0.50, 0.90];
                                method=:vb, standardize=true, vb_maxiter=80, n_draws=60, seed=42)
        @test qf_vb isa VBQuantileFAVAR{Float64}
        qir_vb = qfavar_panel_irf(qf_vb, H; tau_profile=0.50, method=:analytical)
        @test qir_vb isa QuantileImpulseResponse{Float64}
        @test size(qir_vb.values) == (H, N, r + 1)

        # Correlation on the key-variable shock (rotation-invariant across PCA/QFM/BBE)
        corr_2step_key = cor(vec(qir_2step.values[:, :, r + 1]), vec(ols_panel.values[:, :, r + 1]))
        corr_vb_key = cor(vec(qir_vb.values[:, :, r + 1]), vec(ols_panel.values[:, :, r + 1]))
        @test corr_2step_key > 0.92
        @test corr_vb_key > 0.92

        # Also check simulation QIRF at τ=0.50 matches closely
        qir_2step_sim = qfavar_panel_irf(qf_2step, H; tau_profile=0.50, method=:simulation, n_boot=250, seed=10)
        @test cor(vec(qir_2step_sim.values[:, :, r + 1]), vec(ols_panel.values[:, :, r + 1])) > 0.92

        # Matrix Y_key dispatch matches column-index dispatch
        qf_mat = estimate_qfavar(X, X[:, key_idx], r, p, [0.10, 0.50, 0.90];
                                 method=:twostep, standardize=true, seed=42)
        @test qf_mat.Y_key_indices == key_idx
        @test qfavar_panel_irf(qf_mat, H; tau_profile=0.50).values ≈ qir_2step.values atol=1e-10
    end

    # =========================================================================
    # 3. Acceptance Criterion 2: Asymmetric Tail-Risk DGP (left-tail dispersion)
    # =========================================================================

    @testset "AC2: Asymmetric tail-risk DGP (strong negative τ=0.10, near-zero τ=0.50/0.90)" begin
        rng = Random.Xoshiro(89003)
        T_obs, N, r, p, H = 350, 40, 1, 1, 6

        # Y_key is an exogenous stress factor; positive Y_key shocks amplify left-tail
        # dispersion without shifting the conditional mean, median, or 90th percentile.
        Y_key = zeros(T_obs)
        F_true = zeros(T_obs, r)
        for t in 2:T_obs
            Y_key[t] = 0.5 * Y_key[t - 1] + randn(rng)
            stress = max(Y_key[t - 1], 0.0)
            jump = 1.2 + 2.2 * stress
            u = rand(rng)
            # Zero-mean asymmetric mixture: 18% left tail, 4% compensating extreme upper tail (>τ=0.90),
            # 78% symmetric N(0, 0.25^2) body covering τ=0.50 and τ=0.90.
            eta = if u < 0.18
                -jump + 0.2 * randn(rng)
            elseif u < 0.22
                4.5 * jump + 0.2 * randn(rng)
            else
                0.25 * randn(rng)
            end
            F_true[t, 1] = 0.4 * F_true[t - 1, 1] + eta
        end

        # Panel X has positive loadings on F_true plus Y_key as column 1
        Lam = 0.8 .+ 0.6 .* rand(rng, N - 1, r)
        X_nonkey = F_true * Lam' .+ 0.2 .* randn(rng, T_obs, N - 1)
        X_panel = hcat(Y_key, X_nonkey)

        # Direct QVAR on [F_true, Y_key]
        Y_var = hcat(F_true, Y_key)
        qv_tail = estimate_qvar(Y_var, p, [0.10, 0.50, 0.90]; se=:none)
        ir_10 = qirf(qv_tail, H; shock=2, tau_profile=0.10)
        ir_50 = qirf(qv_tail, H; shock=2, tau_profile=0.50)
        ir_90 = qirf(qv_tail, H; shock=2, tau_profile=0.90)

        # At h=1 (index 2), factor 1 has strong negative response at τ=0.10, near-zero at τ=0.50/0.90
        @test ir_10.values[2, 1, 1] < -0.5
        @test abs(ir_50.values[2, 1, 1]) < 0.25
        @test abs(ir_90.values[2, 1, 1]) < 0.25
        @test ir_10.values[2, 1, 1] < ir_50.values[2, 1, 1] - 0.3

        # Quantile FAVAR (:twostep and :vb) on full panel X_panel
        qf_tail_2s = estimate_qfavar(X_panel, [1], r, p, [0.10, 0.50, 0.90];
                                     method=:twostep, standardize=true, seed=101)
        qf_tail_vb = estimate_qfavar(X_panel, [1], r, p, [0.10, 0.50, 0.90];
                                     method=:vb, standardize=true, vb_maxiter=80, n_draws=50, seed=101)

        for qfm_mod in (qf_tail_2s, qf_tail_vb)
            pir_10 = qfavar_panel_irf(qfm_mod, H; shock=2, tau_profile=0.10)
            pir_50 = qfavar_panel_irf(qfm_mod, H; shock=2, tau_profile=0.50)
            pir_90 = qfavar_panel_irf(qfm_mod, H; shock=2, tau_profile=0.90)

            mean_resp_10 = mean(pir_10.values[2, 2:end, 1])
            mean_resp_50 = mean(pir_50.values[2, 2:end, 1])
            mean_resp_90 = mean(pir_90.values[2, 2:end, 1])

            @test mean_resp_10 < -0.3
            @test abs(mean_resp_50) < 0.25
            @test abs(mean_resp_90) < 0.25
            @test mean_resp_10 < mean_resp_50 - 0.25
        end
    end

    # =========================================================================
    # 4. Acceptance Criterion 3: #528 Purging & Exact 1-for-1 Unit Impact Invariant
    # =========================================================================

    @testset "AC3: #528 purging invariant & exact unit impact at h=0" begin
        rng = Random.Xoshiro(89004)
        T_obs, N, r, p, H = 180, 25, 2, 1, 6
        d = dgp_dynamic_factors(rng; A=[0.5 0.1; 0.0 0.5], N=N, T=T_obs, idio_sd=0.3)
        # Heterogeneous column scales to verify raw-panel unit rescaling
        scales = collect(range(0.5, 8.0; length=N))'
        offsets = collect(range(-10.0, 10.0; length=N))'
        X = d.X .* scales .+ offsets
        key_indices = [3, 12]

        for std_flag in (true, false)
            qf_2s = estimate_qfavar(X, key_indices, r, p, [0.10, 0.50, 0.90];
                                    method=:twostep, standardize=std_flag, seed=55)
            qf_vb = estimate_qfavar(X, key_indices, r, p, [0.10, 0.50, 0.90];
                                    method=:vb, standardize=std_flag, vb_maxiter=60, n_draws=40, seed=55)

            # Purged factors in :twostep are orthogonal to Y_key
            Y_k = X[:, key_indices]
            for m_idx in 1:3, f_j in 1:r, k_j in 1:length(key_indices)
                @test abs(cor(qf_2s.factors_list[m_idx][:, f_j], Y_k[:, k_j])) < 1e-10
            end

            # VB anchors satisfy Λ[anchor, :] == I_r and Λ_y[anchor, :] == 0
            anchor = setdiff(1:N, key_indices)[1:r]
            Ir = Matrix{Float64}(I, r, r)
            for m_idx in 1:3
                @test qf_vb.loadings_mean[m_idx][anchor, :] ≈ Ir atol=1e-12
                @test norm(qf_vb.lambda_y_mean[m_idx][anchor, :]) == 0.0
            end

            # Exact 1-for-1 unit impact at h=0 (index 1) for both :twostep and :vb
            for mod in (qf_2s, qf_vb), tau_q in (0.10, 0.50, 0.90)
                pir = qfavar_panel_irf(mod, H; tau_profile=tau_q)
                aug_ir = qirf(mod.qvar, H; tau_profile=tau_q)
                for (k, panel_idx) in enumerate(key_indices)
                    shock_col = r + k
                    # Exact 1.0 unit impact on own shock at h=0
                    @test pir.values[1, panel_idx, shock_col] ≈ 1.0 atol=1e-12
                    # Exact 1-for-1 match with augmented VAR response at all horizons & shocks
                    @test pir.values[:, panel_idx, :] ≈ aug_ir.values[:, r + k, :] atol=1e-12
                end
            end
        end
    end

    # =========================================================================
    # 5. Panel Quantile Forecasts & Uncertainty Intervals (Credible + Bootstrap)
    # =========================================================================

    @testset "qfavar_panel_forecast (rearrangement) and CI bands" begin
        rng = Random.Xoshiro(89005)
        X = dgp_dynamic_factors(rng; A=[0.5 0.0; 0.0 0.5], N=20, T=160, idio_sd=0.4).X
        key_idx = [2, 9]

        qf_2s = estimate_qfavar(X, key_idx, 2, 1, [0.10, 0.50, 0.90];
                                method=:twostep, seed=31)
        qf_vb = estimate_qfavar(X, key_idx, 2, 1, [0.10, 0.50, 0.90];
                                method=:vb, vb_maxiter=60, n_draws=40, seed=31)

        for mod in (qf_2s, qf_vb)
            fc = qfavar_panel_forecast(mod, 6; rearrange=true)
            @test size(fc.quantiles) == (6, 20, 3)
            @test size(fc.forecast) == (6, 20)
            @test size(fc.ci_lower) == (6, 20)
            @test size(fc.ci_upper) == (6, 20)
            @test size(fc.factors) == (6, 2, 3)
            @test fc.horizon == 6
            @test fc.rearranged === true
            @test all(isfinite, fc.quantiles)
            # Non-crossing quantiles guaranteed when rearrange=true
            @test all(fc.quantiles[:, :, 1] .<= fc.quantiles[:, :, 2] .<= fc.quantiles[:, :, 3])
            @test fc.forecast == fc.quantiles[:, :, 2]
        end

        # Unsorted taus with rearrangement
        qf_unsorted = estimate_qfavar(X, [1], 2, 1, [0.90, 0.10, 0.50]; method=:twostep, seed=32)
        fc_u = qfavar_panel_forecast(qf_unsorted, 4; rearrange=true)
        @test all(fc_u.quantiles[:, :, 2] .<= fc_u.quantiles[:, :, 3] .<= fc_u.quantiles[:, :, 1])
        @test fc_u.ci_lower == fc_u.quantiles[:, :, 2]
        @test fc_u.ci_upper == fc_u.quantiles[:, :, 1]
        @test fc_u.forecast == fc_u.quantiles[:, :, 3]

        # Bootstrap & Credible bands on qfavar_panel_irf
        pir_boot = qfavar_panel_irf(qf_2s, 6; tau_profile=0.50, ci_type=:bootstrap, n_boot=20, seed=7)
        @test pir_boot.ci_type === :bootstrap
        @test all(pir_boot.ci_lower .<= pir_boot.ci_upper)

        pir_cred = qfavar_panel_irf(qf_vb, 6; tau_profile=0.10, ci_type=:credible, conf_level=0.90)
        @test pir_cred.ci_type === :credible
        @test pir_cred.conf_level ≈ 0.90
        @test all(pir_cred.ci_lower .<= pir_cred.ci_upper)
    end

    # =========================================================================
    # 6. AC4: RSER-01 _generic_construct, Float32, display/report, seed, validations, ambiguities
    # =========================================================================

    @testset "AC4: RSER-01 _generic_construct, Float32, display, seed, errors, ambiguities" begin
        rng = Random.Xoshiro(89006)
        X = dgp_dynamic_factors(rng; A=[0.5 0.0; 0.0 0.5], N=15, T=120, idio_sd=0.4).X

        qv = estimate_qvar(X[:, 1:3], 1, [0.10, 0.50, 0.90]; se=:asymptotic, seed=42)
        qir = qirf(qv, 5; tau_profile=0.10)
        qf = estimate_qfavar(X, [1], 2, 1, [0.10, 0.50, 0.90]; method=:twostep, seed=42)
        vb = estimate_qfavar(X, [1], 2, 1, [0.10, 0.50, 0.90]; method=:vb, vb_maxiter=40, n_draws=25, seed=42)

        # RSER-01 positional constructor & _generic_construct check for all 4 structs
        for obj in (qv, qir, qf, vb)
            T_obj = typeof(obj)
            Tw = Base.typename(T_obj).wrapper
            args = Any[getfield(obj, i) for i in 1:nfields(obj)]
            rebuilt = M_QF._generic_construct(Tw, args)
            @test typeof(rebuilt) === T_obj
            for f in fieldnames(T_obj)
                @test isequal(getfield(rebuilt, f), getfield(obj, f))
            end
        end

        # Float32 precision & Integer fallback
        X32 = Float32.(X)
        qv32 = estimate_qvar(X32[:, 1:3], 1, Float32[0.10, 0.50, 0.90]; se=:asymptotic, seed=42)
        @test qv32 isa QuantileVARModel{Float32}
        qir32 = qirf(qv32, 4; tau_profile=0.5f0)
        @test qir32 isa QuantileImpulseResponse{Float32}
        qf32 = estimate_qfavar(X32, [1], 2, 1, Float32[0.10, 0.50, 0.90]; method=:twostep, seed=42)
        @test qf32 isa QuantileFAVAR{Float32}
        vb32 = estimate_qfavar(X32, [1], 2, 1, Float32[0.10, 0.50, 0.90]; method=:vb, vb_maxiter=30, n_draws=20, seed=42)
        @test vb32 isa VBQuantileFAVAR{Float32}
        @test qfavar_panel_irf(qf32, 4; tau_profile=0.5f0) isa QuantileImpulseResponse{Float32}
        @test eltype(qfavar_panel_forecast(vb32, 3).quantiles) === Float32

        # _generic_construct preserves Float32
        for obj32 in (qv32, qir32, qf32, vb32)
            T32 = typeof(obj32)
            Tw32 = Base.typename(T32).wrapper
            args32 = Any[getfield(obj32, i) for i in 1:nfields(obj32)]
            @test typeof(M_QF._generic_construct(Tw32, args32)) === T32
        end

        # Integer matrix fallback -> Float64
        X_int = round.(Int, 10 .* X)
        @test estimate_qvar(X_int[:, 1:3], 1, [0.5]) isa QuantileVARModel{Float64}
        @test estimate_qfavar(X_int, [1], 2, 1, [0.5]) isa QuantileFAVAR{Float64}

        # Display and report across all backends + repro=true footer
        for be in (:text, :latex, :html)
            M_QF.with_display_backend(be) do
                for obj in (qv, qir, qf, vb)
                    @test !isempty(sprint(show, obj))
                    @test !isempty(sprint(report, obj))
                end
            end
        end
        @test occursin("Reproducibility: seed=42", sprint(io -> report(io, qv; repro=true)))
        @test occursin("Reproducibility: seed=42", sprint(io -> report(io, qf; repro=true)))
        @test occursin("Reproducibility: seed=42", sprint(io -> report(io, vb; repro=true)))

        # Seed reproducibility
        qf2 = estimate_qfavar(X, [1], 2, 1, [0.10, 0.50, 0.90]; method=:twostep, seed=42)
        vb2 = estimate_qfavar(X, [1], 2, 1, [0.10, 0.50, 0.90]; method=:vb, vb_maxiter=40, n_draws=25, seed=42)
        @test qf.qvar.B_list == qf2.qvar.B_list
        @test vb.B_draws == vb2.B_draws
        @test vb.loadings_draws == vb2.loadings_draws

        # ArgumentError validations
        @test_throws ArgumentError estimate_qvar(X[:, 1:3], 0)
        @test_throws ArgumentError estimate_qvar(X[:, 1:3], 1, Float64[])
        @test_throws ArgumentError estimate_qvar(X[:, 1:3], 1, [0.0, 0.5])
        @test_throws ArgumentError estimate_qvar(X[:, 1:3], 1, [0.5, 1.0])
        @test_throws ArgumentError estimate_qvar(X[:, 1:3], 1; ordering=[1, 1, 2])
        @test_throws ArgumentError estimate_qvar(X[:, 1:3], 1; se=:bogus)
        @test_throws ArgumentError estimate_qvar(X[:, 1:3], 1; se=:bootstrap, n_boot=1)
        @test_throws ArgumentError estimate_qvar(X[:, 1:3], 1; varnames=["a"])

        @test_throws ArgumentError qirf(qv, 0)
        @test_throws ArgumentError qirf(qv, 5; shock=0)
        @test_throws ArgumentError qirf(qv, 5; shock=10)
        @test_throws ArgumentError qirf(qv, 5; tau_profile=0.0)
        @test_throws ArgumentError qirf(qv, 5; tau_profile=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        @test_throws ArgumentError qirf(qv, 5; baseline_tau=1.0)
        @test_throws ArgumentError qirf(qv, 5; method=:bogus)
        @test_throws ArgumentError qirf(qv, 5; ci_type=:bogus)
        @test_throws ArgumentError qirf(qv, 5; ci_type=:bootstrap, conf_level=1.5)

        @test_throws ArgumentError estimate_qfavar(X, Int[], 2, 1)
        @test_throws ArgumentError estimate_qfavar(X, [0], 2, 1)
        @test_throws ArgumentError estimate_qfavar(X, [1, 1], 2, 1)
        @test_throws ArgumentError estimate_qfavar(X, [99], 2, 1)
        @test_throws ArgumentError estimate_qfavar(X, randn(rng, 50, 1), 2, 1)
        @test_throws ArgumentError estimate_qfavar(X, [1], 0, 1)
        @test_throws ArgumentError estimate_qfavar(X, [1], 2, 0)
        @test_throws ArgumentError estimate_qfavar(X, [1], 2, 1; method=:bogus)
        @test_throws ArgumentError estimate_qfavar(X, [1], 2, 1; panel_varnames=["x"])
        @test_throws ArgumentError estimate_qfavar(X, [1], 2, 1; key_names=["a", "b"])
        @test_throws ArgumentError estimate_qfavar(X[:, 1:3], [1, 2], 2, 1; method=:vb)
        X_nan = copy(X); X_nan[1, 1] = NaN
        @test_throws ArgumentError estimate_qfavar(X_nan, [1], 2, 1)

        @test_throws ArgumentError qfavar_panel_irf(qf, 0)
        @test_throws ArgumentError qfavar_panel_irf(qf, 5; ci_type=:credible)
        @test_throws ArgumentError qfavar_panel_forecast(qf, 0)
    end

    @testset "dgp_qfavar recovery & tail amplification (#891, #892)" begin
        # Calling forms, determinism, and validation
        rng = Random.Xoshiro(42)
        d1 = dgp_qfavar(rng; T_obs=80, N=20, r=2, p=1, n_key=1, seed=99)
        d2 = dgp_qfavar(; T_obs=80, N=20, r=2, p=1, n_key=1, seed=99)
        @test d1.X_panel == d2.X_panel
        @test d1.X == d1.X_panel
        @test d1.Y_key == d2.Y_key
        @test d1.seed == 99
        @test d1.X_panel[:, d1.Y_key_indices] == d1.Y_key
        @test size(d1.X_panel) == (80, 20)
        @test size(d1.B_lower) == (4, 3)
        @test norm(d1.B_lower[2:end, 1:2]) > norm(d1.B_median[2:end, 1:2])
        @test d1.B_median[2:end, :] == d1.B_upper[2:end, :]
        @test all(d1.B_upper[1, :] .> d1.B_median[1, :] .> d1.B_lower[1, :])
        @test d1.true_B(0.10) == d1.B_lower
        @test d1.true_qirf(0.10, 4)[1, :, :] ≈ d1.true_qirf_impact
        @test_throws ArgumentError dgp_qfavar(; T_obs=3, p=1)
        @test_throws ArgumentError dgp_qfavar(; N=3, r=2, n_key=1)
        @test_throws ArgumentError dgp_qfavar(; r=0)
        @test_throws ArgumentError dgp_qfavar(; p=0)
        @test_throws ArgumentError dgp_qfavar(; df_idio=0.0)
        @test_throws ArgumentError dgp_qfavar(; tail_amplification=0.0)
        @test_throws ArgumentError d1.true_B(0.0)
        @test_throws ArgumentError d1.true_qirf(0.5, 0)

        d_p2 = dgp_qfavar(; T_obs=60, N=25, r=2, p=2, n_key=2, seed=12)
        @test size(d_p2.B_median) == (1 + 4 * 2, 4)
        @test size(d_p2.Y_key) == (60, 2)
        @test d_p2.X_panel[:, d_p2.Y_key_indices] == d_p2.Y_key

        # Recovery of tail-amplified VAR transmission & QFAVAR panel IRFs
        rng = Random.Xoshiro(42)
        dq = dgp_qfavar(rng; T_obs=250, N=40, r=2, p=1, n_key=1, tail_amplification=1.5)
        qv_dgp = estimate_qvar(hcat(dq.F, dq.Y_key), 1, [0.10, 0.50, 0.90]; seed=42)
        @test norm(qv_dgp.B_list[1][4, 1:2]) > norm(qv_dgp.B_list[2][4, 1:2])
        ir_qv_10 = qirf(qv_dgp, 5; tau_profile=0.10, baseline_tau=0.50, shock=3)
        ir_qv_50 = qirf(qv_dgp, 5; tau_profile=0.50, baseline_tau=0.50, shock=3)
        @test norm(ir_qv_10.values[2, 1:2, 1]) > norm(ir_qv_50.values[2, 1:2, 1])

        qf_dgp = estimate_qfavar(dq.X_panel, dq.Y_key_indices, 2, 1, [0.10, 0.50, 0.90]; seed=42)
        # Joint span [F_tilde(0.50), Y_key] recovers true latent factor space
        Q_est = Matrix(qr(hcat(qf_dgp.factors_list[2], dq.Y_key)).Q)
        for j in 1:2
            r2_f = sum(abs2, Q_est' * dq.F[:, j]) / sum(abs2, dq.F[:, j])
            @test r2_f > 0.85
        end
        ir10 = qfavar_panel_irf(qf_dgp, 5; shock=3, tau_profile=0.10)
        ir50 = qfavar_panel_irf(qf_dgp, 5; shock=3, tau_profile=0.50)
        ir90 = qfavar_panel_irf(qf_dgp, 5; shock=3, tau_profile=0.90)
        @test ir10.values[1, 1, 1] ≈ 1.0 atol=1e-10
        @test ir50.values[1, 1, 1] ≈ 1.0 atol=1e-10
        @test sum(abs, ir10.values[2:4, 2:end, 1]) > sum(abs, ir50.values[2:4, 2:end, 1])
        @test sum(abs, ir10.values[2:4, 2:end, 1]) > sum(abs, ir90.values[2:4, 2:end, 1])
    end
end
