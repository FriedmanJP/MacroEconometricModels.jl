# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

# Targeted v1.1.0 Codecov coverage harness (Coverage-C, runs in CI serialization+coverage cell).
# Exercises TimeSeriesData/PanelData dispatch wrappers, reproduce() manifests, DGP variants,
# Tables.jl / long_table / report / refs / plot_result consumers, correlated-noise StateSpace
# forecasting, RI Jacobians, and branch variants across all five v1.1.0 series.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using Tables
using StatsAPI

const _COV_M = MacroEconometricModels
const _cov_suppress = _COV_M._suppress_warnings

@testset "v1.1.0 Coverage Harness (#883-#928)" begin
    rng = Xoshiro(11001)

    # =========================================================================
    # 1. Synthetic DGPs (src/dgp/dgp_factors.jl, dgp_gdfm.jl, dgp_svar_nongauss.jl)
    # =========================================================================
    sim_qf_t = dgp_quantile_factors(Xoshiro(1101); T_obs=70, N=10,
                                    location_factors=1, scale_factors=1, asym_factors=1)
    sim_qf_n = dgp_quantile_factors(Xoshiro(1102); T_obs=60, N=10, df_idio=Inf)
    @test size(sim_qf_t.X) == (70, 10)
    @test size(sim_qf_n.X) == (60, 10)

    sim_qfv_p1 = dgp_qfavar(Xoshiro(1103); T_obs=80, N=10, r=1, p=1, n_key=1)
    sim_qfv_p2 = dgp_qfavar(Xoshiro(1104); T_obs=70, N=10, r=2, p=2, n_key=2, df_idio=Inf)
    @test size(sim_qfv_p1.true_B(0.25)) == (3, 2)
    @test size(sim_qfv_p2.true_qirf(0.25, 4)) == (4, 4, 4)

    sim_gdfm = dgp_infinite_gdfm(Xoshiro(1105); T_obs=80, N=10, q=2)
    sim_gdfm_i1 = dgp_infinite_gdfm(Xoshiro(1106); T_obs=80, N=10, q=2, n_permanent=1)
    @test size(sim_gdfm.X) == (80, 10)
    @test size(sim_gdfm_i1.X) == (80, 10)

    sim_sv_t = dgp_ssvarma(Xoshiro(1107); T=90, Phi=[0.35 0.0; 0.0 0.30],
                           Theta=zeros(2, 2, 0), C=[1.0 0.2; -0.15 0.9],
                           dist=:student_t, nu=5.0)
    sim_sv_ma = dgp_ssvarma(Xoshiro(1108); T=90, Phi=[0.35 0.0; 0.0 0.30],
                            Theta=[-0.4 0.0; 0.0 0.3], C=[1.0 0.2; -0.15 0.9],
                            dist=:mixture_normal_asym)
    sim_sv_sn = dgp_ssvarma(Xoshiro(1109); T=90, Phi=nothing,
                            Theta=[-0.5 0.0; 0.8 -1.6], C=[0.6 1.0; 0.8 0.4],
                            dist=:skew_normal)
    @test size(sim_sv_t.Y) == (90, 2)
    @test size(sim_sv_ma.Y) == (90, 2)
    @test size(sim_sv_sn.Y) == (90, 2)

    # =========================================================================
    # 2. TimeSeriesData & PanelData Dispatch Wrappers (src/data/convert.jl)
    # =========================================================================
    ts_qf = TimeSeriesData(sim_qf_t.X; varnames=["x$(i)" for i in 1:10])
    ts_gdfm = TimeSeriesData(sim_gdfm.X; varnames=["g$(i)" for i in 1:10])
    ts_i1 = TimeSeriesData(sim_gdfm_i1.X; varnames=["c$(i)" for i in 1:10])
    ts_sv = TimeSeriesData(sim_sv_t.Y; varnames=["s1", "s2"])
    ts_sv2 = TimeSeriesData(sim_sv_ma.Y; varnames=["s1", "s2"])
    ts_pan = TimeSeriesData(randn(rng, 90, 6); varnames=["p$(i)" for i in 1:6])

    _cov_suppress() do
        qfm_ts = estimate_qfm(ts_qf, 2, 0.25; method=:iqr, se=:asymptotic, max_iter=25, seed=11)
        qfm_auto_ts = estimate_qfm(ts_qf, :auto, 0.5; r_max=3, max_iter=20, seed=11)
        qfm_grid_ts = estimate_qfm(ts_qf, 2, [0.25, 0.50, 0.75]; max_iter=20, seed=11)
        qfm_ic_ts = ic_criteria_qfm(ts_qf, 3, 0.25; n_starts=1, max_iter=20, warn_boundary=false)
        @test qfm_ts isa QuantileFactorModel
        @test qfm_auto_ts isa QuantileFactorModel
        @test qfm_grid_ts isa QuantileFactorGrid
        @test qfm_ic_ts isa QFMICResult

        @test onatski_q_test(ts_gdfm, 1, 3) isa OnatskiQTestResult
        @test onatski_q_select(ts_gdfm) isa Tuple{Int, <:AbstractVector}
        @test onatski_q_select(ts_gdfm, 3) isa Tuple{Int, <:AbstractVector}
        @test ahn_horenstein(ts_gdfm, 4) isa AhnHorensteinResult
        @test onatski_ed(ts_gdfm, 4) isa OnatskiEDResult

        fhlz_ts = estimate_gdfm_fhlz(ts_gdfm, 2; p_max=2, n_perm=3, bandwidth=3, seed=11)
        i1_ts = estimate_gdfm_i1(ts_i1, 2; n_permanent=1, n_perm=3, p_max=2, irf_horizon=6, seed=11)
        eur_ts = eurocoin_indicator(ts_gdfm, vec(sim_gdfm.X[:, 1]), 2; cutoff=π / 4, r=3)
        tv_ts = estimate_tv_gdfm(ts_gdfm, 2; n_grid=4)
        gdfm_base = estimate_gdfm(ts_gdfm, 2; r=3)
        X_rag_ts = copy(sim_gdfm.X); X_rag_ts[end, 1] = NaN
        nc_ts = nowcast_gdfm(gdfm_base, TimeSeriesData(X_rag_ts; varnames=ts_gdfm.varnames), 0)
        @test fhlz_ts isa FHLZModel
        @test i1_ts isa CointegratedGDFM
        @test eur_ts isa EurocoinResult
        @test tv_ts isa TimeVaryingGDFM
        @test nc_ts isa GDFMNowcastResult

        lms_ts = estimate_lms_svar(ts_sv, 1; distributions=[:student_t, :student_t],
                                   method=:three_step, n_starts=1, max_iter=40, seed=11)
        ms_ts = select_gmm_moments(ts_sv, 1)
        ll_ts = estimate_lanne_luoto_gmm(ts_sv, 1; n_starts=1, max_iter=40, seed=11)
        ssv_ts = estimate_ssvarma(ts_sv2, 1, 1; method=:ml, n_starts=1, max_iter=40, seed=11)
        @test lms_ts isa LMSSVARResult
        @test ms_ts isa GMMMomentSelectionResult
        @test ll_ts isa LanneLuotoGMMResult
        @test ssv_ts isa SVARMAModel

        si1 = test_sufficient_information(lms_ts, ts_pan; r_max=2, p_f=2)
        si2 = test_sufficient_information(ts_sv, ts_pan.data; r_max=2, p_f=2)
        si3 = test_sufficient_information(ts_sv, ts_pan; r_max=2, p_f=2)
        @test si1 isa SufficientInformationTestResult
        @test si2 isa SufficientInformationTestResult
        @test si3 isa SufficientInformationTestResult

        gar_ts = growth_at_risk(ts_qf, "x1", 1; r=2, taus=[0.1, 0.5, 0.9], density=:normal, seed=11)
        qv_ts = estimate_qvar(ts_sv, 1, [0.25, 0.75]; se=:asymptotic, seed=11)
        ts_qfv_panel = TimeSeriesData(sim_qfv_p1.X_panel; varnames=["v$(i)" for i in 1:10])
        ts_qfv_key = TimeSeriesData(sim_qfv_p1.Y_key; varnames=["key1"])
        qfv_ts1 = estimate_qfavar(ts_qfv_panel, [1], 1, 1, [0.25, 0.75]; seed=11)
        qfv_ts2 = estimate_qfavar(ts_qfv_panel, ts_qfv_key, 1, 1, [0.25, 0.75]; seed=11)
        @test gar_ts isa GrowthAtRiskResult
        @test qv_ts isa QuantileVARModel
        @test qfv_ts1 isa QuantileFAVAR
        @test qfv_ts2 isa QuantileFAVAR

        # PanelData wrappers (src/factor/quantile_panel.jl & pre existing)
        N_p, T_p = 10, 15
        df_p = DataFrame(
            id = repeat(1:N_p, inner=T_p),
            t = repeat(1:T_p, outer=N_p),
            y = randn(rng, N_p * T_p),
            w1 = randn(rng, N_p * T_p),
        )
        pd_q = xtset(df_p, :id, :t)
        qpan_pd = estimate_xtqreg_factors(pd_q, :y, [:w1], 1, 0.5; maxiter=15, seed=11)
        @test qpan_pd isa QuantileFactorPanelModel
        @test coef(qpan_pd) == qpan_pd.beta
        @test StatsAPI.coefnames(qpan_pd) == qpan_pd.varnames
        @test nobs(qpan_pd) == N_p * T_p
        @test !islinear(qpan_pd)
        @test dof(qpan_pd) >= 1

        # RI & DSGE-VAR TimeSeriesData wrappers
        prob_ri = RationalInattentionProblem(0.85, 1.0, 1.0; capacity=0.8)
        sol_ri = solve_ri_kalman(prob_ri)
        sim_ri = simulate(sol_ri, 50; burn=20, seed=11)
        ts_ri = TimeSeriesData(hcat(sim_ri.states, sim_ri.actions); varnames=["x", "a"])
        rie_ts = estimate_ri(prob_ri, ts_ri; method=:mle, estimate_param=:capacity)
        @test rie_ts isa RIEstimationResult
        @test rie_ts isa RIEstimationResult && ts_sv isa TimeSeriesData

        A_ab = [0.7 0.0; 0.0 0.5]; B_ab = [1.0 0.0; 0.0 1.0]
        C_ab = [1.0 0.3; 0.0 0.8]; D_ab = [1.0 0.1; 0.0 1.0]
        sys_ab = ABCDSystem(A_ab, B_ab, C_ab, D_ab)
        ts_dv = TimeSeriesData(randn(rng, 50, 2); varnames=["y1", "y2"])
        @test isfinite(dsge_var_logml(dsge_var_prior(sys_ab, 1, 1.0, 49), ts_dv))
        @test isfinite(dsge_var_logml(sys_ab, ts_dv, 1, 1.0))
        dv_prof_ts = select_dsge_var_lambda(sys_ab, ts_dv, 1; lambda_grid=[0.5, 2.0, Inf])
        dv_post_ts = estimate_dsge_var(sys_ab, ts_dv, 1; lambda=1.0, n_draws=15, seed=11)
        @test dv_prof_ts isa DSGEVARLambdaProfile
        @test dv_post_ts isa DSGEVARPosterior
        # =====================================================================
        # 3. Correlated-Noise StateSpace API, RI Jacobians & Identification Branches
        # =====================================================================
        prob_ri2 = RationalInattentionProblem([0.85 0.05; 0.0 0.70], [1.0 0.0; 0.0 1.0],
                                              [1.0, 0.5]; capacity=1.0)
        sol_ri2 = solve_ri_kalman(prob_ri2)
        ss_ri = to_statespace(sol_ri2; observables=:all)
        abcd_ri = to_abcd(sol_ri2; observables=:actions_and_states)
        @test ss_ri isa StateSpaceModel
        @test abcd_ri isa ABCDSystem
        @test irf(sol_ri2, 6) isa ImpulseResponse
        @test fevd(sol_ri2, 6) isa FEVD

        # Fitted StateSpaceModel with correlated shock-measurement noise G !== nothing
        Z_g = [1.0;;]; H_g = [0.5;;]; T_g = [0.7;;]; R_g = [1.0;;]; Q_g = [0.8;;]; G_g = [0.3;;]
        y_g = randn(rng, 40)
        ss_g_spec = StateSpaceModel(Z_g, H_g, T_g, Q_g; R=R_g, G=G_g, init_mode=:stationary)
        ss_g_fit = estimate_statespace(ss_g_spec, y_g)
        @test _COV_M.isfitted(ss_g_fit)
        fc_g = forecast(ss_g_fit, 3)
        @test size(fc_g.mean) == (3, 1) && all(isfinite, fc_g.se)
        ssk_g = steady_state_kalman(ss_g_fit)
        @test ssk_g isa SteadyStateKalman

        # Behavioral RI Jacobian modes (:gain & :capacity)
        J_ssj = Matrix{Float64}(I, 6, 6)
        J_gain = rational_inattention_jacobian(J_ssj, 0.4; mode=:gain)
        J_kal = rational_inattention_jacobian(J_ssj, 0.8; mode=:capacity, rho_news=0.8)
        @test size(J_gain) == (6, 6) && size(J_kal) == (6, 6)

        # Identification dispatch & compute_Q variants + LMS hypotheses & LL-GMM
        var_m = estimate_var(sim_sv_t.Y, 1)
        @test size(compute_Q(var_m, :cvm; max_iter=30, seed=11)) == (2, 2)
        @test size(compute_Q(var_m, :mixture_normal_asym; max_iter=30, seed=11)) == (2, 2)
        @test size(compute_Q(var_m, :lms; n_starts=1, max_iter=30, seed=11)) == (2, 2)

        lms1 = estimate_lms_svar(var_m; method=:full_ml, n_starts=1, max_iter=40, seed=11)
        lms3 = estimate_lms_svar(var_m; method=:three_step, n_starts=1, max_iter=40, seed=11)
        @test lms1 isa LMSSVARResult && lms3 isa LMSSVARResult
        @test test_svar_hypothesis(lms3, impact_linear_hypothesis([1.0 0.0]); method=:wald) isa SVARHypothesisTest
        @test test_svar_hypothesis(lms3, horizon_irf_hypothesis(2, [(0, 1, 2)]); method=:wald) isa SVARHypothesisTest
        h_mix = mixed_svar_hypothesis(short_run=impact_zero_hypothesis(2, [(1, 2)]))
        h_nl = nonlinear_svar_hypothesis(theta -> theta[3], 1)
        @test test_svar_hypothesis(lms3, h_mix; method=:lr) isa SVARHypothesisTest
        @test test_svar_hypothesis(lms3, h_nl; method=:lm) isa SVARHypothesisTest
        @test test_svar_hypothesis(lms3, recursiveness_hypothesis(2); bootstrap=:wild, n_boot=9, seed=11) isa SVARHypothesisTest

        ll_it = estimate_lanne_luoto_gmm(var_m; weighting=:iterated, moment_selection=:rmsc_seq,
                                         n_starts=1, max_iter=30, seed=11)
        ll_cue = estimate_lanne_luoto_gmm(var_m; weighting=:cue, moment_selection=:none,
                                          n_starts=1, max_iter=30, seed=11)
        @test ll_it isa LanneLuotoGMMResult && ll_cue isa LanneLuotoGMMResult
        @test _COV_M._llgmm_covers_all_pairs([(1, 2)], 2)

        # =====================================================================
        # 4. Option & Consumer Coverage Across QFM, GDFM, SSVARMA, ABCD-VAR & RI
        # =====================================================================
        X_q = sim_qf_t.X
        qfm_boot = estimate_qfm(X_q, 2, 0.25; method=:smoothed_iqr, kernel=:epanechnikov,
                                se=:bootstrap, n_boot=6, max_iter=15, seed=11)
        qfm_log = estimate_qfm(X_q, 2, 0.75; method=:smoothed_iqr, kernel=:logistic,
                               se=:asymptotic, max_iter=15, seed=11)
        @test size(qfm_boot.se_loadings) == (10, 2) && size(qfm_log.se_factors) == (70, 2)
        fc_q_th = forecast(qfm_log, 3; p=1, ci_method=:none)
        fc_q_bt = forecast(qfm_log, 2; p=1, ci_method=:bootstrap, n_boot=6, seed=11)
        @test fc_q_th isa FactorForecast && fc_q_bt isa FactorForecast

        gar_emp = growth_at_risk(X_q, 1, 1; r=2, taus=[0.1, 0.5, 0.9], density=:none, seed=11)
        gar_skt = growth_at_risk(X_q, 1, 1; r=2, taus=[0.05, 0.25, 0.5, 0.75, 0.95],
                                 density=:skewed_t, oos_split=50, seed=11)
        @test gar_emp isa GrowthAtRiskResult && gar_skt isa GrowthAtRiskResult
        @test isfinite(_COV_M._skewt_logpdf(0.1, 0.0, 1.0, -0.5, 5.0))

        qv_bt = estimate_qvar(sim_sv_t.Y, 1, [0.25, 0.75]; se=:bootstrap, n_boot=6, seed=11)
        @test length(vcov(qv_bt)) == 2 && size(confint(qv_bt)) == (6, 2, 2)
        qi_sim = qirf(qv_bt, 4; method=:simulation, ci_type=:bootstrap, n_boot=6, seed=11)
        @test qi_sim isa QuantileImpulseResponse

        qf_vb = estimate_qfavar(sim_qfv_p1.X_panel, [1], 1, 1, [0.25, 0.75];
                                method=:vb, vb_maxiter=15, n_draws=12, seed=11)
        qf_p_irf = qfavar_panel_irf(qf_vb, 4; ci_type=:credible)
        qf_p_fc = qfavar_panel_forecast(qf_vb, 3)
        @test qf_p_irf isa QuantileImpulseResponse && qf_p_fc isa NamedTuple

        gdfm_1s = estimate_gdfm(sim_gdfm.X, 2; r=3, spectral=:lag_window, bandwidth=6)
        @test length(common_variance_share(gdfm_1s)) == size(sim_gdfm.X, 2)
        @test haskey(spectral_eigenvalue_plot_data(gdfm_1s), :eigenvalues)
        fc_g_1s = forecast(gdfm_1s, 3; method=:one_sided, ci_method=:theoretical)
        fc_g_bt = forecast(gdfm_1s, 2; method=:one_sided, ci_method=:bootstrap, n_boot=6, seed=11)
        @test fc_g_1s isa FactorForecast && fc_g_bt isa FactorForecast

        m_sv = estimate_ssvarma(sim_sv_ma.Y, 1, 1; method=:pml, distribution=:pearson_iv,
                                se=:opg, n_starts=1, max_iter=40, seed=11)
        @test length(blaschke_representations(m_sv.Theta, m_sv.C)) >= 1
        @test size(ssvarma_filter_shocks(m_sv).shocks) == (size(sim_sv_ma.Y, 1) - 1, 2)
        @test test_fundamentalness(m_sv; max_lag=2, n_boot=9, seed=11) isa FundamentalnessTestResult

        sys_sq = ABCDSystem([0.5;;], [1.0;;], [2.0;;], [1.0;;])
        Y_dv = randn(rng, 45, 1)
        dv_rw = estimate_dsge_var(sys_sq, Y_dv, 1; lambda=1.5, n_draws=20, seed=11)
        @test irf(dv_rw, 4) isa BayesianImpulseResponse
        @test fevd(dv_rw, 4) isa BayesianFEVD
        @test forecast(dv_rw, 3) isa BVARForecast
        @test poor_mans_invertibility(sys_sq) isa InvertibilityResult
        @test innovations_representation(sys_sq) isa InnovationsRepresentation
        @test shock_recoverability(sys_sq) isa ShockRecoverabilityResult
        @test compare_dsge_svar_irf(sys_sq, 1, 4) isa DSGEVARIRFAudit
        @test search_invertible_observables(
            ABCDSystem([0.5 0.0; 0.0 0.4], [1.0 0.0; 0.0 1.0],
                       [1.0 0.0; 1.5 0.5; 0.0 1.0], [1.0 0.0; 0.0 1.0; 0.5 0.5]), 2
        ) isa ObservableSearchResult

        # Full Tables / report / refs / reproduce / plot_result sweep on remaining new types
        bp = bandpass_gdfm(gdfm_1s, (0.0, Float64(π / 4)))
        sdfm_g = estimate_structural_dfm(gdfm_1s; method=:fglr, r=3, p=1, H=6)
        sf = spectral_fevd(sdfm_g)
        qgrid = estimate_qfm(X_q, 2, [0.25, 0.75]; max_iter=15, seed=11)
        qsp = test_qfm_spanning(qgrid, X_q)
        qct = test_qfm_loading_constancy(qgrid, 0.25, 0.75; n_boot=5, seed=11)
        mw_ri = solve_mw_attention((; rho=0.85, sigma=0.01, units=:nats),
                                   (; rho=0.75, sigma=0.10, process=:growth, units=:nats);
                                   capacity=2.0, strategic_complementarity=0.5, horizon=12)
        dc_ri = solve_ri_discrete([1.0 0.0; 0.0 1.0; 0.2 0.2], [0.4, 0.6]; capacity=0.5)
        iia_ri = ri_iia_diagnostic([1.0 0.0; 0.0 1.0], [0.5, 0.5], [1.0, 0.0]; omega=0.8)
        m_ril = estimate_ri_logit([1, 2, 1, 2], randn(rng, 2, 2, 2), [1, 1, 2, 2])
        d_ri = solve_ri_transition(RationalInattentionProblem(0.8, 1.0, 1.0; capacity=1.0), [2.0;;]; T_path=4)

        nd_s = diagnose_nongaussian_svar(lms3; lags=2)
        cg_s = coibion_gorodnichenko_test(randn(rng, 30), randn(rng, 30), randn(rng, 30))

        for repro_obj in (qfm_log, fhlz_ts, lms3, m_ril, test_svar_hypothesis(lms3, recursiveness_hypothesis(2)),
                          qgrid, qct, qpan_pd, i1_ts, ll_it, rie_ts, gar_skt, qv_bt, qfv_ts1, qf_vb)
            rep_res = reproduce(repro_obj)
            @test rep_res isa ReproReport
        end

        all_consumers = (
            qfm_log, qfm_ic_ts, qgrid, qsp, qct, qpan_pd, gar_skt, qv_bt, qi_sim, qfv_ts1, qf_vb,
            fhlz_ts, i1_ts, bp, eur_ts, sf, tv_ts, nc_ts,
            lms3, nd_s, ll_it, ll_it.selection, m_sv,
            sys_sq, dv_rw, mw_ri, dc_ri, iia_ri, m_ril, d_ri, cg_s, prob_ri, sol_ri, rie_ts,
        )
        for obj in all_consumers
            @test Tables.istable(typeof(obj))
            @test DataFrame(obj) isa DataFrame
            if applicable(long_table, obj)
                @test long_table(obj) isa DataFrame
            end
            txt = applicable(report, IOBuffer(), obj) ? sprint(report, obj) : sprint(show, obj)
            @test !isempty(txt)
            if applicable(refs, IOBuffer(), obj)
                @test !isempty(sprint(io -> refs(io, obj; format=:bibtex)))
            end
            if applicable(plot_result, obj)
                @test plot_result(obj) isa PlotOutput
            end
        end
    end
end
