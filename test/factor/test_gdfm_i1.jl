# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using LinearAlgebra
using Statistics
using Random
using StatsAPI
using MacroEconometricModels

@testset "Cointegrated I(1) GDFM Tests (#896)" begin

    """
    Generate a BLL (2021) cointegrated I(1) panel with:
    - `n_perm` independent random-walk permanent stochastic trends
    - `q - n_perm` stationary AR(1) transitory factors
    - Series-specific error-correction adjustments `(1 - rho_i) χ_{i,t-1} -> χ_t - Λ F_t`
      or heterogeneous AR(1) propagation in levels + idiosyncratic I(0) noise.
    """
    function make_i1_cointegrated_panel(;
        N::Int=45,
        T_obs::Int=250,
        q::Int=2,
        phi::Float64=0.6,
        idio_sd::Float64=0.25,
        drift_scale::Float64=0.0,
        const_scale::Float64=1.0,
        burn::Int=60,
        seed::Int=202604,
    )
        @assert q == 2
        rng = Random.Xoshiro(seed)
        T_tot = T_obs + burn

        lambda1 = 0.6 .+ 0.8 .* rand(rng, N)             # strictly positive loadings on permanent RW trend
        lambda2 = randn(rng, N)
        lambda2[1] = 1.5
        lambda2[2] = -1.2

        # Moderate heterogeneous short-run adjustment ρ_i ∈ [-0.3, 0.4] on Δχ_{i,t} or stationary part
        shocks = randn(rng, T_tot, 2)                    # col 1 = permanent, col 2 = transitory
        F1 = cumsum(@view(shocks[:, 1]))                 # I(1) RW factor
        F2 = zeros(T_tot)
        @inbounds for t in 1:T_tot
            F2[t] = (t > 1 ? phi * F2[t - 1] : 0.0) + shocks[t, 2]
        end

        chi = zeros(T_tot, N)
        @inbounds for t in 1:T_tot, i in 1:N
            chi[t, i] = lambda1[i] * F1[t] + lambda2[i] * F2[t]
        end

        consts = const_scale .* randn(rng, N)
        drifts = drift_scale .* (0.5 .+ rand(rng, N))
        idio = zeros(T_obs, N)
        e_idio = idio_sd .* randn(rng, T_obs, N)
        @inbounds for t in 1:T_obs, i in 1:N
            prev = t > 1 ? 0.3 * idio[t - 1, i] : 0.0
            idio[t, i] = prev + e_idio[t, i]
        end

        X = zeros(T_obs, N)
        @inbounds for t in 1:T_obs, i in 1:N
            X[t, i] = consts[i] + drifts[i] * t + chi[burn + t, i] + idio[t, i]
        end

        # Exact true level IRFs for h = 0..40 (41 × N × 2)
        H_max = 40
        C_true = zeros(H_max + 1, N, 2)
        @inbounds for h in 0:H_max, i in 1:N
            C_true[h + 1, i, 1] = lambda1[i]
            C_true[h + 1, i, 2] = lambda2[i] * (phi^h)
        end

        (X=X, chi=chi[(burn + 1):end, :], shocks=shocks[(burn + 1):end, :],
         C_true=C_true, lambda1=lambda1, lambda2=lambda2)
    end

    # =========================================================================
    # 1. Type Construction, RSER-01 Field-Order Constructor & StatsAPI
    # =========================================================================
    @testset "CointegratedGDFM construction, RSER-01 constructor, and StatsAPI" begin
        sim = make_i1_cointegrated_panel(; N=18, T_obs=120, q=2, seed=101)
        names = ["s$i" for i in 1:18]
        snames = ["Perm", "Trans"]

        m = estimate_gdfm_i1(sim.X, 2; n_perm=8, p_max=3, irf_horizon=20,
                             varnames=names, shock_names=snames, seed=11)

        @test m isa CointegratedGDFM{Float64}
        @test m isa AbstractFactorModel
        @test nvars(m) == 18
        @test nobs(m) == 120
        @test StatsAPI.nobs(m) == 120
        @test effective_nobs(m) == size(m.shocks, 1)
        @test effective_nobs(m) < 120
        @test varnames(m) == names
        @test m.shock_names == snames
        @test size(m.common_levels) == (120, 18)
        @test size(m.idiosyncratic) == (120, 18)
        @test m.common_levels + m.idiosyncratic ≈ m.X
        @test predict(m) == m.common_levels
        @test residuals(m) == m.idiosyncratic
        @test size(m.irf_matrices) == (21, 18, 2)
        @test size(m.long_run_impact) == (18, 2)
        @test size(m.B0) == (2, 2)
        @test size(m.trend_coef) == (2, 18)
        @test size(m.p_blocks, 1) == 8
        @test m.q == 2
        @test m.n_perm == 8
        @test m.det_trend === :constant
        @test m.identification === :permanent_transitory
        @test m.irf_horizon == 20
        @test m.manifest !== nothing
        @test StatsAPI.dof(m) > 0
        @test length(StatsAPI.r2(m)) == 18
        @test all(x -> 0.0 <= x <= 1.0, StatsAPI.r2(m))
        cvs = common_variance_share(m)
        @test length(cvs) == 18
        @test all(isfinite, cvs)

        # RSER-01 positional constructor invariant
        vals = ntuple(i -> getfield(m, fieldnames(CointegratedGDFM)[i]), fieldcount(CointegratedGDFM))
        m_rebuilt = CointegratedGDFM{Float64}(vals...)
        @test m_rebuilt isa CointegratedGDFM{Float64}
        @test m_rebuilt.irf_matrices == m.irf_matrices
        @test m_rebuilt.manifest == m.manifest

        # Display and report
        buf = IOBuffer()
        show(buf, m)
        txt = String(take!(buf))
        @test occursin("Cointegrated I(1) GDFM", txt)
        @test occursin("Shock Norms", txt)
        report(m)
    end

    # =========================================================================
    # 2. Input Validation & Edge Errors
    # =========================================================================
    @testset "Validation and error handling" begin
        rng = Random.Xoshiro(202)
        X_small = cumsum(randn(rng, 40, 10); dims=1)
        X_tiny = randn(rng, 4, 10)

        @test_throws ArgumentError estimate_gdfm_i1(X_tiny, 2)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 0)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 11)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; kernel=:bad_kernel)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; det_trend=:quadratic)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; identification=:bogus)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; lag_criterion=:hqic)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; p_max=0)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; n_perm=0)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; irf_horizon=0)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; n_permanent=3)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; identification=:sign)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; identification=:proxy)
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; varnames=["a", "b"])
        @test_throws ArgumentError estimate_gdfm_i1(X_small, 2; shock_names=["only_one"])

        X_nan = copy(X_small)
        X_nan[5, 3] = NaN
        @test_throws ArgumentError estimate_gdfm_i1(X_nan, 2)

        m = estimate_gdfm_i1(X_small, 2; n_perm=4, p_max=2, irf_horizon=10, seed=1)
        @test_throws ArgumentError irf(m, 0)
        @test_throws ArgumentError fevd(m, 0)
        @test_throws ArgumentError forecast(m, 0)
        @test_throws ArgumentError forecast(m, 5; ci_method=:invalid)
    end

    # =========================================================================
    # 3. BLL (2021) I(1) Cointegrated Panel Benchmark (N=45, T=250, q=2)
    # =========================================================================
    @testset "BLL I(1) cointegrated panel benchmark (N=45, T=250, q=2)" begin
        sim = make_i1_cointegrated_panel(;
            N=45, T_obs=250, q=2, phi=0.6, idio_sd=0.25, seed=89601)

        m = estimate_gdfm_i1(
            sim.X, 2;
            p_max=3,
            n_perm=25,
            det_trend=:constant,
            irf_horizon=40,
            identification=:permanent_transitory,
            n_permanent=1,
            seed=89601,
        )

        @test m.n_permanent == 1
        @test m.shock_names == ["P1", "T1"]
        @test all(isfinite, m.irf_matrices)
        @test all(isfinite, m.long_run_impact)

        # Permanent shock: C_h -> C_∞ != 0, transitory shock: C_h -> 0 at h = 40
        C_est = m.irf_matrices                      # 41 × 45 × 2
        perm_h0_norm = norm(@view(C_est[1, :, 1]))
        perm_h40_norm = norm(@view(C_est[41, :, 1]))
        trans_h0_norm = norm(@view(C_est[1, :, 2]))
        trans_h40_norm = norm(@view(C_est[41, :, 2]))

        @test perm_h40_norm > 0.5 * perm_h0_norm
        @test norm(@view(m.long_run_impact[:, 1])) > 0.1
        @test norm(@view(m.long_run_impact[:, 2])) < 1e-8 * max(norm(@view(m.long_run_impact[:, 1])), 1.0)
        @test trans_h40_norm < 0.25 * max(trans_h0_norm, 1e-12)

        # Match sign per shock column against true level IRFs
        s1 = dot(@view(C_est[1, :, 1]), @view(sim.C_true[1, :, 1])) >= 0 ? 1.0 : -1.0
        s2 = dot(@view(C_est[1, :, 2]), @view(sim.C_true[1, :, 2])) >= 0 ? 1.0 : -1.0
        est_perm = s1 .* @view(C_est[:, :, 1])
        est_trans = s2 .* @view(C_est[:, :, 2])
        true_perm = @view(sim.C_true[:, :, 1])
        true_trans = @view(sim.C_true[:, :, 2])

        corr_perm = cor(vec(est_perm), vec(true_perm))
        corr_trans = cor(vec(est_trans), vec(true_trans))
        @test corr_perm > 0.85
        @test corr_trans > 0.85

        # Compare against naive first-differenced FHLZ accumulated IRFs:
        # without cointegration / error-correction structure, diff(X) misses the permanent/transitory split
        dX = diff(sim.X; dims=1)
        fhlz_diff = estimate_gdfm_fhlz(dX, 2; p_max=3, n_perm=15, irf_horizon=41, seed=89601)
        cum_fhlz = cumsum(fhlz_diff.impulse_filters[1:41, :, :]; dims=1)
        @test norm(@view(m.long_run_impact[:, 2])) < 1e-6 * max(norm(@view(cum_fhlz[41, :, 2])), 1e-8)

        # Automatic Johansen n_permanent detection also detects 1 permanent trend
        m_auto = estimate_gdfm_i1(sim.X, 2; p_max=3, n_perm=10, det_trend=:constant,
                                  irf_horizon=20, seed=89601)
        @test 1 <= m_auto.n_permanent <= 2
        @test m_auto.n_permanent == 1
    end

    # =========================================================================
    # 4. Detrending Options, Alternative Identifications & Determinism
    # =========================================================================
    @testset "Detrending (:none, :linear), kernels & identification schemes" begin
        sim_lin = make_i1_cointegrated_panel(;
            N=20, T_obs=150, q=2, drift_scale=0.45, const_scale=2.0, seed=303)
        m_lin = estimate_gdfm_i1(sim_lin.X, 2; det_trend=:linear, n_perm=6,
                                 p_max=2, irf_horizon=15, seed=303)
        @test m_lin.det_trend === :linear
        @test mean(m_lin.trend_coef[2, :]) > 0.10
        @test cor(vec(m_lin.common_levels), vec(sim_lin.X)) > 0.90

        sim_none = make_i1_cointegrated_panel(;
            N=20, T_obs=140, q=2, const_scale=0.0, seed=304)
        m_none = estimate_gdfm_i1(sim_none.X, 2; det_trend=:none, n_perm=6,
                                  p_max=2, kernel=:parzen, seed=304)
        @test iszero(m_none.trend_coef)
        @test m_none.kernel === :parzen

        # Identification schemes
        m_chol = estimate_gdfm_i1(sim_none.X, 2; identification=:cholesky,
                                  n_perm=6, p_max=2, n_permanent=1, seed=305)
        @test abs(m_chol.irf_matrices[1, 1, 2]) < 1e-10

        m_lr = estimate_gdfm_i1(sim_none.X, 2; identification=:long_run,
                                n_perm=6, p_max=2, n_permanent=2,
                                target_vars=[1, 2], seed=306)
        @test abs(m_lr.long_run_impact[1, 2]) < 1e-10

        # Sign identification
        sgn_fn = irf_arr -> irf_arr[1, 1, 1] > 0 && irf_arr[1, 2, 2] > 0
        m_sign = estimate_gdfm_i1(sim_none.X, 2; identification=:sign,
                                  sign_check=sgn_fn, n_perm=6, p_max=2,
                                  n_permanent=1, seed=307)
        @test m_sign.irf_matrices[1, 1, 1] > 0
        @test m_sign.irf_matrices[1, 2, 2] > 0

        # Proxy identification (correlate with true permanent shock padded to T_obs)
        rng = Random.Xoshiro(308)
        z_inst = sim_none.shocks[:, 1] .+ 0.2 .* randn(rng, 140)
        m_proxy = estimate_gdfm_i1(sim_none.X, 2; identification=:proxy,
                                   instrument=z_inst, normalize=(1, 1.0),
                                   n_perm=6, p_max=2, n_permanent=1, seed=308)
        @test m_proxy.irf_matrices[1, 1, 1] ≈ 1.0 atol=1e-8
        @test size(m_proxy.shocks, 2) == 2

        # Threaded vs serial seed determinism (#243)
        m_th = estimate_gdfm_i1(sim_none.X, 2; n_perm=8, p_max=2, threaded=true, seed=999)
        m_sr = estimate_gdfm_i1(sim_none.X, 2; n_perm=8, p_max=2, threaded=false, seed=999)
        @test m_th.irf_matrices ≈ m_sr.irf_matrices atol=1e-12
        @test m_th.shocks ≈ m_sr.shocks atol=1e-12
    end

    # =========================================================================
    # 5. Downstream Methods: irf, fevd, historical_decomposition, forecast, Float32
    # =========================================================================
    @testset "irf, fevd, historical_decomposition, forecast & Float32" begin
        sim = make_i1_cointegrated_panel(; N=20, T_obs=150, q=2, seed=404)
        m = estimate_gdfm_i1(sim.X, 2; n_perm=8, p_max=2, irf_horizon=25,
                             n_permanent=1, seed=404)

        # irf (default + extended beyond irf_horizon + 1)
        ir = irf(m)
        @test ir isa ImpulseResponse{Float64}
        @test size(ir.values) == (26, 20, 2)
        @test ir.values == m.irf_matrices
        ir_ext = irf(m, 35)
        @test size(ir_ext.values) == (35, 20, 2)
        @test ir_ext.values[35, :, :] == m.long_run_impact

        # fevd (with and without idiosyncratic)
        fv = fevd(m, 15)
        @test fv isa FEVD{Float64}
        @test size(fv.proportions) == (20, 2, 15)
        @test all(x -> x >= -1e-12, fv.proportions)
        @test all(abs.(sum(fv.proportions; dims=2) .- 1.0) .< 1e-10)

        fv_idio = fevd(m, 15; include_idiosyncratic=true)
        @test size(fv_idio.proportions) == (20, 3, 15)
        @test fv_idio.shocks[end] == "Idiosyncratic"
        @test all(abs.(sum(fv_idio.proportions; dims=2) .- 1.0) .< 1e-10)

        # historical_decomposition exact additivity
        hd = historical_decomposition(m)
        @test hd isa HistoricalDecomposition{Float64}
        @test verify_decomposition(hd; tol=1e-10)
        @test size(hd.contributions, 3) == 3

        hd_no_idio = historical_decomposition(m; include_idiosyncratic=false)
        @test verify_decomposition(hd_no_idio; tol=1e-10)
        @test size(hd_no_idio.contributions, 3) == 2

        # forecast: none, theoretical, bootstrap (level cointegration & monotone CI width)
        fc0 = forecast(m, 8; ci_method=:none)
        @test fc0 isa FactorForecast{Float64}
        @test size(fc0.observables) == (8, 20)
        @test size(fc0.factors) == (8, 2)

        fc_th = forecast(m, 8; ci_method=:theoretical, conf_level=0.95)
        @test all(fc_th.observables_upper .>= fc_th.observables)
        @test all(fc_th.observables .>= fc_th.observables_lower)
        # In levels with a permanent shock, forecast SE grows with horizon
        @test fc_th.observables_se[8, 1] >= fc_th.observables_se[1, 1]

        fc_b1 = forecast(m, 6; ci_method=:bootstrap, reps=50, seed=555)
        fc_b2 = forecast(m, 6; ci_method=:bootstrap, reps=50, seed=555)
        @test fc_b1.observables_upper == fc_b2.observables_upper
        @test all(fc_b1.observables_upper .>= fc_b1.observables)
        @test all(fc_b1.observables .>= fc_b1.observables_lower)

        # Float32 precision support
        X32 = Float32.(sim.X)
        m32 = estimate_gdfm_i1(X32, 2; n_perm=5, p_max=2, irf_horizon=12, seed=405)
        @test m32 isa CointegratedGDFM{Float32}
        @test irf(m32, 10) isa ImpulseResponse{Float32}
        @test fevd(m32, 10) isa FEVD{Float32}
        @test forecast(m32, 4) isa FactorForecast{Float32}
    end

    # =========================================================================
    # 6. dgp_infinite_gdfm I(1) Cointegrated Recovery (n_permanent > 0, #900)
    # =========================================================================
    @testset "dgp_infinite_gdfm I(1) cointegrated recovery (n_permanent > 0)" begin
        rng = Random.Xoshiro(9002)
        d_i1 = dgp_infinite_gdfm(rng; T_obs=250, N=45, q=2, n_permanent=1,
                                 a_coef=0.4, b_coef=0.4, signal_share=0.8)
        @test d_i1.q == 2
        @test d_i1.n_permanent == 1
        @test d_i1.X ≈ d_i1.common + d_i1.idiosyncratic atol=1e-12

        # Permanent shock IRF stabilizes at nonzero long-run level; transitory decays to 0
        @test norm(@view(d_i1.irf_true[41, :, 1])) > 0.5 * norm(@view(d_i1.irf_true[1, :, 1]))
        @test norm(@view(d_i1.irf_true[41, :, 2])) < 1e-6 * norm(@view(d_i1.irf_true[1, :, 2]))

        m_i1 = estimate_gdfm_i1(d_i1.X, 2; n_permanent=1, p_max=3, n_perm=15,
                                irf_horizon=25, det_trend=:constant, seed=9002)
        @test m_i1.n_permanent == 1
        @test cor(vec(m_i1.common_levels), vec(d_i1.common)) > 0.90
        @test norm(@view(m_i1.long_run_impact[:, 1])) > 0.1
        @test norm(@view(m_i1.long_run_impact[:, 2])) < 1e-8 * max(norm(@view(m_i1.long_run_impact[:, 1])), 1.0)

        # Sign-aligned level IRF recovery for permanent and transitory shocks
        C_est = m_i1.irf_matrices
        C_true = d_i1.irf_true[1:26, :, :]
        s1 = dot(@view(C_est[1, :, 1]), @view(C_true[1, :, 1])) >= 0 ? 1.0 : -1.0
        s2 = dot(@view(C_est[1, :, 2]), @view(C_true[1, :, 2])) >= 0 ? 1.0 : -1.0
        @test cor(vec(s1 .* @view(C_est[:, :, 1])), vec(@view(C_true[:, :, 1]))) > 0.80
        @test cor(vec(s2 .* @view(C_est[:, :, 2])), vec(@view(C_true[:, :, 2]))) > 0.75

        # Automatic Johansen detection identifies n_permanent == 1
        m_auto = estimate_gdfm_i1(d_i1.X, 2; p_max=3, n_perm=10, det_trend=:constant,
                                  irf_horizon=15, seed=9002)
        @test m_auto.n_permanent == 1
    end
end
