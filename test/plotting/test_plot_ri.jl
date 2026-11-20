# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under the GPL-3.0-or-later license.

using Test, Random, LinearAlgebra
using MacroEconometricModels

if !@isdefined(check_plot)
    include(joinpath(@__DIR__, "plot_test_helpers.jl"))
end

@testset "Rational Inattention & Kalman Plots (#923)" begin
    rng = Random.Xoshiro(92301)

    # Shared RI LQG problem
    A = [0.85 0.10; 0.00 0.70]
    B = [1.0 0.0; 0.0 0.7]
    H = [1.0 0.25; 0.35 0.90]
    prob = RationalInattentionProblem(
        A, B, H;
        capacity=1.2,
        state_names=["Output", "Cost"],
        action_names=["Price", "Wage"],
        shock_names=["Demand", "Supply"],
    )
    sol = solve_ri_kalman(prob)

    # 1. RationalInattentionProblem{T}
    @testset "RationalInattentionProblem" begin
        p = plot_result(prob)
        check_plot(p)
        assert_all_json_valid(p)
        @test length(panel_titles(p.html)) == 2
        @test occursin("State Transition A", p.html)
        @test occursin("Target Action Loading H", p.html)

        p_tit = plot_result(prob; title="Custom RI Spec")
        check_plot(p_tit)
        @test occursin("Custom RI Spec", p_tit.html)

        tmp = mktempdir()
        sp = joinpath(tmp, "ri_prob.html")
        plot_result(prob; save_path=sp)
        @test isfile(sp)
        @test occursin("<!DOCTYPE html>", read(sp, String))

        prob32 = RationalInattentionProblem(Float32.(A), Float32.(B), Float32.(H); capacity=1.0f0)
        p32 = plot_result(prob32)
        check_plot(p32)
        assert_all_json_valid(p32)

        prob_hostile = RationalInattentionProblem(
            A, B, H;
            capacity=1.2,
            state_names=[HOSTILE_NAME, "s2"],
            action_names=[HOSTILE_NAME, "a2"],
        )
        p_esc = plot_result(prob_hostile; title=HOSTILE_NAME)
        check_plot(p_esc)
        assert_all_json_valid(p_esc)
        assert_escapes(p_esc)
    end

    # 2. SteadyStateKalman{T}
    @testset "SteadyStateKalman" begin
        # Complex conjugate stable eigenvalues + explicit unstable eigenvalue test
        ssk = SteadyStateKalman{Float64}(
            [0.5 0.1; 0.1 0.4],
            [0.3 0.0; 0.0 0.2],
            [0.4 0.1; 0.0 0.5],
            [1.0 0.1; 0.1 1.0],
            [ComplexF64(0.6, 0.35), ComplexF64(0.6, -0.35), ComplexF64(1.05, 0.0)],
        )
        p = plot_result(ssk)
        check_plot(p)
        assert_all_json_valid(p)
        @test occursin("Inside unit circle", p.html)
        @test occursin("On/outside unit circle", p.html)
        @test occursin("UNSTABLE", p.html)

        # Via StateSpaceModel
        Z_ss = [1.0 0.0]
        H_ss = reshape([0.25], 1, 1)
        T_ss = [0.85 0.1; -0.05 0.7]
        RQR_ss = [0.5 0.0; 0.0 0.3]
        ssk_ss = steady_state_kalman(Z_ss, H_ss, T_ss, RQR_ss)
        p_ss = plot_result(ssk_ss)
        check_plot(p_ss)
        assert_all_json_valid(p_ss)

        tmp = mktempdir()
        sp = joinpath(tmp, "ssk.html")
        plot_result(ssk; title="Kalman Poles", save_path=sp)
        @test isfile(sp)

        ssk32 = SteadyStateKalman{Float32}(
            Float32.(ssk.P_pred), Float32.(ssk.P_filt), Float32.(ssk.K_gain),
            Float32.(ssk.F_innov), ComplexF32.(ssk.closed_loop_eigs),
        )
        p32 = plot_result(ssk32)
        check_plot(p32)
        assert_all_json_valid(p32)

        p_esc = plot_result(ssk; title=HOSTILE_NAME)
        check_plot(p_esc)
        assert_all_json_valid(p_esc)
        assert_escapes(p_esc)

        ssk_nan = SteadyStateKalman{Float64}(
            ssk.P_pred, ssk.P_filt, ssk.K_gain, ssk.F_innov,
            [ComplexF64(NaN, 0.0), ComplexF64(0.5, 0.2)],
        )
        p_nan = plot_result(ssk_nan)
        assert_all_json_valid(p_nan)
        assert_nan_becomes_null(p_nan)
    end

    # 3. SimsKalmanSolution{T} — 3 views
    @testset "SimsKalmanSolution" begin
        # :irf_by_capacity
        p_irf = plot_result(sol)
        check_plot(p_irf)
        assert_all_json_valid(p_irf)
        @test occursin("Full info", p_irf.html)

        p_irf_sub = plot_result(sol; view=:irf_by_capacity, capacities=[0.3, sol.capacity_bits, Inf],
                                horizon=12, var="Price", shock="Demand", ncols=1, title="Price IRF")
        check_plot(p_irf_sub)
        assert_all_json_valid(p_irf_sub)
        @test occursin("Price IRF", p_irf_sub.html)
        @test length(panel_titles(p_irf_sub.html)) == 1

        p_irf_int = plot_result(sol; view=:irf_by_capacity, var=2, shock=2, horizon=10)
        check_plot(p_irf_int)
        assert_all_json_valid(p_irf_int)

        # :fevd
        p_fevd = plot_result(sol; view=:fevd, horizon=15)
        check_plot(p_fevd)
        assert_all_json_valid(p_fevd)
        @test length(panel_titles(p_fevd.html)) == 2
        p_fevd_v = plot_result(sol; view=:fevd, var="Wage", ncols=1, horizon=12)
        check_plot(p_fevd_v)
        assert_all_json_valid(p_fevd_v)
        @test length(panel_titles(p_fevd_v.html)) == 1

        # :water_filling
        p_wf = plot_result(sol; view=:water_filling)
        check_plot(p_wf)
        assert_all_json_valid(p_wf)
        @test occursin("Water-Filling", p_wf.html)

        # Zero-attention edge case (signal_rank == 0)
        prob_zero = RationalInattentionProblem(A, B, H; omega=500.0)
        sol_zero = solve_ri_kalman(prob_zero)
        @test sol_zero.signal_rank == 0
        for vw in (:irf_by_capacity, :fevd, :water_filling)
            pz = plot_result(sol_zero; view=vw)
            check_plot(pz)
            assert_all_json_valid(pz)
        end

        # Save path & Float32
        tmp = mktempdir()
        sp = joinpath(tmp, "sims_wf.html")
        plot_result(sol; view=:water_filling, save_path=sp)
        @test isfile(sp)

        prob32 = RationalInattentionProblem(Float32.(A), Float32.(B), Float32.(H); capacity=1.0f0)
        sol32 = solve_ri_kalman(prob32)
        for vw in (:irf_by_capacity, :fevd, :water_filling)
            p32 = plot_result(sol32; view=vw)
            check_plot(p32)
            assert_all_json_valid(p32)
        end

        # Hostile escaping across all views
        prob_h = RationalInattentionProblem(
            A, B, H;
            capacity=1.2,
            state_names=[HOSTILE_NAME, "s2"],
            action_names=[HOSTILE_NAME, "a2"],
            shock_names=[HOSTILE_NAME, "w2"],
        )
        sol_h = solve_ri_kalman(prob_h)
        for vw in (:irf_by_capacity, :fevd, :water_filling)
            p_esc = plot_result(sol_h; view=vw, title=HOSTILE_NAME)
            check_plot(p_esc)
            assert_all_json_valid(p_esc)
            assert_escapes(p_esc)
        end

        # NaN -> null propagation
        F_nan = copy(sol.F_policy)
        F_nan[1, 1] = NaN
        sol_nan_irf = SimsKalmanSolution{Float64}(
            sol.problem, sol.Sigma_prior, sol.Sigma_post, sol.Omega_benefit,
            sol.omega, sol.capacity_bits, sol.capacity_nats, sol.signal_rank,
            sol.Y_signal, sol.Sigma_psi, sol.K_gain, F_nan,
            sol.expected_loss, sol.converged, sol.iterations, sol.method,
        )
        p_nan_irf = plot_result(sol_nan_irf; view=:irf_by_capacity, capacities=[sol.capacity_bits])
        assert_all_json_valid(p_nan_irf)
        assert_nan_becomes_null(p_nan_irf)

        Sp_nan = copy(sol.Sigma_prior)
        Sp_nan[1, 1] = NaN
        sol_nan_wf = SimsKalmanSolution{Float64}(
            sol.problem, Sp_nan, sol.Sigma_post, sol.Omega_benefit,
            sol.omega, sol.capacity_bits, sol.capacity_nats, sol.signal_rank,
            sol.Y_signal, sol.Sigma_psi, sol.K_gain, sol.F_policy,
            sol.expected_loss, sol.converged, sol.iterations, sol.method,
        )
        p_nan_wf = plot_result(sol_nan_wf; view=:water_filling)
        assert_all_json_valid(p_nan_wf)
        assert_nan_becomes_null(p_nan_wf)

        # Error paths
        @test_throws ArgumentError plot_result(sol; view=:invalid_view)
        @test_throws ArgumentError plot_result(sol; var="MissingVar")
        @test_throws ArgumentError plot_result(sol; var=99)
        @test_throws ArgumentError plot_result(sol; shock="MissingShock")
        @test_throws ArgumentError plot_result(sol; shock=99)
    end

    # 4. DynamicRISolution{T} — 2 views
    @testset "DynamicRISolution" begin
        prob_om = RationalInattentionProblem(A, B, H; omega=0.4)
        sol_ss = solve_ri_kalman(prob_om)
        dyn = solve_ri_transition(prob_om, 3.0 * sol_ss.Sigma_prior; T_path=30)

        p_att = plot_result(dyn; view=:attention_path)
        check_plot(p_att)
        assert_all_json_valid(p_att)
        @test length(panel_titles(p_att.html)) == 2

        p_om = plot_result(dyn; view=:omega)
        check_plot(p_om)
        assert_all_json_valid(p_om)
        @test length(panel_titles(p_om.html)) == 2

        tmp = mktempdir()
        sp = joinpath(tmp, "dyn_ri.html")
        plot_result(dyn; title="Transition Path", save_path=sp)
        @test isfile(sp)

        prob_om32 = RationalInattentionProblem(Float32.(A), Float32.(B), Float32.(H); omega=0.4f0)
        sol_ss32 = solve_ri_kalman(prob_om32)
        dyn32 = solve_ri_transition(prob_om32, 2.5f0 * sol_ss32.Sigma_prior; T_path=20)
        for vw in (:attention_path, :omega)
            p32 = plot_result(dyn32; view=vw)
            check_plot(p32)
            assert_all_json_valid(p32)
        end

        for vw in (:attention_path, :omega)
            p_esc = plot_result(dyn; view=vw, title=HOSTILE_NAME)
            check_plot(p_esc)
            assert_all_json_valid(p_esc)
            assert_escapes(p_esc)
        end

        cap_nan = copy(dyn.capacity_bits_path)
        cap_nan[4] = NaN
        om_nan = copy(dyn.omega_path)
        om_nan[4] = NaN
        dyn_nan = DynamicRISolution{Float64}(
            dyn.Sigma_prior_path, dyn.Sigma_post_path, dyn.Xi_path, dyn.K_gain_path,
            dyn.Y_terminal, dyn.Sigma_psi_terminal, om_nan, cap_nan,
            dyn.capacity_nats_path, dyn.signal_rank_path, dyn.Sigma_init, dyn.T_path,
            dyn.converged, dyn.iterations, dyn.residual,
            dyn.state_names, dyn.action_names, dyn.shock_names,
        )
        for vw in (:attention_path, :omega)
            p_nan = plot_result(dyn_nan; view=vw)
            assert_all_json_valid(p_nan)
            assert_nan_becomes_null(p_nan)
        end

        @test_throws ArgumentError plot_result(dyn; view=:bad)
    end

    # 5. MWAttentionResult{T}
    @testset "MWAttentionResult" begin
        agg = RationalInattentionProblem(reshape([0.9], 1, 1), reshape([1.0], 1, 1), reshape([1.0], 1, 1); omega=0.2)
        idio = RationalInattentionProblem(reshape([0.7], 1, 1), reshape([1.5], 1, 1), reshape([1.0], 1, 1); omega=0.2)
        mw = solve_mw_attention(agg, idio; capacity=1.5, strategic_complementarity=0.5, horizon=15, tol=1e-5)

        p = plot_result(mw)
        check_plot(p)
        assert_all_json_valid(p)
        @test length(panel_titles(p.html)) == 2
        @test occursin("Price Impulse Responses", p.html)
        @test occursin("Real Output Response", p.html)

        tmp = mktempdir()
        sp = joinpath(tmp, "mw.html")
        plot_result(mw; title="MW Benchmark", save_path=sp)
        @test isfile(sp)

        agg32 = RationalInattentionProblem(reshape([0.9f0], 1, 1), reshape([1.0f0], 1, 1), reshape([1.0f0], 1, 1); omega=0.2f0)
        idio32 = RationalInattentionProblem(reshape([0.7f0], 1, 1), reshape([1.5f0], 1, 1), reshape([1.0f0], 1, 1); omega=0.2f0)
        mw32 = solve_mw_attention(agg32, idio32; capacity=1.5f0, strategic_complementarity=0.5f0, horizon=12, tol=1e-4)
        p32 = plot_result(mw32)
        check_plot(p32)
        assert_all_json_valid(p32)

        p_esc = plot_result(mw; title=HOSTILE_NAME)
        check_plot(p_esc)
        assert_all_json_valid(p_esc)
        assert_escapes(p_esc)

        agg_nan = copy(mw.agg_price_irf)
        agg_nan[3] = NaN
        mw_nan = MWAttentionResult{Float64}(
            mw.kappa_agg, mw.kappa_idio, mw.total_kappa, mw.omega,
            mw.strategic_complementarity, mw.units,
            agg_nan, mw.idio_price_irf, mw.real_output_irf,
            mw.agg_solution, mw.idio_solution, mw.converged, mw.iterations,
        )
        p_nan = plot_result(mw_nan)
        assert_all_json_valid(p_nan)
        assert_nan_becomes_null(p_nan)
    end

    # 6. RIEstimationResult{T}
    @testset "RIEstimationResult" begin
        prob_est = RationalInattentionProblem(reshape([0.85], 1, 1), reshape([1.0], 1, 1), reshape([1.0], 1, 1); capacity=0.8)
        sol_est = solve_ri_kalman(prob_est)
        sim_est = simulate(sol_est, 120; rng=rng)
        res = estimate_ri(prob_est, sim_est.actions; method=:mle, estimate_param=:capacity)

        p = plot_result(res)
        check_plot(p)
        assert_all_json_valid(p)
        p90 = plot_result(res; conf_level=0.90, title="MLE Capacity")
        check_plot(p90)
        assert_all_json_valid(p90)
        @test occursin("90% CI", p90.html)
        @test_throws ArgumentError plot_result(res; conf_level=0.0)

        tmp = mktempdir()
        sp = joinpath(tmp, "ri_est.html")
        plot_result(res; save_path=sp)
        @test isfile(sp)

        # Multi-parameter + hostile name + NaN + Float32
        res32 = RIEstimationResult{Float32}(
            Float32[0.85, 0.35], ["capacity", "rho"], Float32[0.01 0.0; 0.0 0.02], Float32[0.1, 0.1414],
            0.85f0, 0.25f0, -40.0f0, -35.0f0, 10.0f0, 0.002f0, 84.0f0, 89.0f0,
            120, :mle, :capacity, solve_ri_kalman(RationalInattentionProblem(reshape([0.85f0], 1, 1), reshape([1.0f0], 1, 1), reshape([1.0f0], 1, 1); capacity=0.85f0)),
            zeros(Float32, 0, 0), NaN32, true, nothing,
        )
        p32 = plot_result(res32)
        check_plot(p32)
        assert_all_json_valid(p32)

        res_esc = RIEstimationResult{Float64}(
            res.params, [HOSTILE_NAME], res.vcov_mat, res.stderr,
            res.capacity, res.omega, res.loglik, res.loglik_fullinfo,
            res.lr_stat, res.lr_pvalue, res.aic, res.bic, res.n_obs,
            res.method, res.estimate_param, res.solution, res.draws,
            res.acceptance_rate, res.converged, res.manifest,
        )
        p_esc = plot_result(res_esc; title=HOSTILE_NAME)
        check_plot(p_esc)
        assert_all_json_valid(p_esc)
        assert_escapes(p_esc)

        res_nan = RIEstimationResult{Float64}(
            res.params, res.param_names, res.vcov_mat, [NaN],
            res.capacity, res.omega, res.loglik, res.loglik_fullinfo,
            res.lr_stat, res.lr_pvalue, res.aic, res.bic, res.n_obs,
            res.method, res.estimate_param, res.solution, res.draws,
            res.acceptance_rate, res.converged, res.manifest,
        )
        p_nan = plot_result(res_nan)
        assert_all_json_valid(p_nan)
        assert_nan_becomes_null(p_nan)
    end

    # 7. CGTestResult{T}
    @testset "CGTestResult" begin
        n_cg = 80
        f_now = cumsum(randn(rng, n_cg))
        f_lag = vcat(0.0, f_now[1:end-1])
        noise = randn(rng, n_cg)
        realized = f_now .+ 0.6 .* (f_now .- f_lag) .+ 0.5 .* noise
        cg = coibion_gorodnichenko_test(realized, f_now, f_lag)

        p = plot_result(cg)
        check_plot(p)
        assert_all_json_valid(p)
        p90 = plot_result(cg; conf_level=0.90, title="CG Regression")
        check_plot(p90)
        assert_all_json_valid(p90)
        @test occursin("90% HAC CI", p90.html)
        @test_throws ArgumentError plot_result(cg; conf_level=1.5)

        tmp = mktempdir()
        sp = joinpath(tmp, "cg.html")
        plot_result(cg; save_path=sp)
        @test isfile(sp)

        cg32 = CGTestResult{Float32}(0.05f0, 0.04f0, 0.60f0, 0.12f0, 5.0f0, 1f-5, n_cg, 4, 0.625f0, 0.708f0)
        p32 = plot_result(cg32)
        check_plot(p32)
        assert_all_json_valid(p32)

        p_esc = plot_result(cg; title=HOSTILE_NAME)
        check_plot(p_esc)
        assert_all_json_valid(p_esc)
        assert_escapes(p_esc)

        cg_nan = CGTestResult{Float64}(cg.alpha, cg.se_alpha, cg.beta_cg, NaN, cg.t_stat, cg.pvalue,
                                       cg.n_obs, cg.hac_lags, cg.implied_gain, cg.implied_capacity)
        p_nan = plot_result(cg_nan)
        assert_all_json_valid(p_nan)
        assert_nan_becomes_null(p_nan)
    end

    # 8. RIDiscreteChoiceSolution{T} — 2 views
    @testset "RIDiscreteChoiceSolution" begin
        U_dc = [2.0 0.2 1.0; 0.3 2.1 0.8; -5.0 -5.0 -5.0]
        mu_dc = [0.4, 0.35, 0.25]
        dc_sol = solve_ri_discrete(U_dc, mu_dc; omega=0.45)

        p_cc = plot_result(dc_sol; view=:choice_curves)
        check_plot(p_cc)
        assert_all_json_valid(p_cc)
        @test length(panel_titles(p_cc.html)) == 2
        @test occursin("State-Contingent Choice Probabilities", p_cc.html)

        p_kkt = plot_result(dc_sol; view=:kkt)
        check_plot(p_kkt)
        assert_all_json_valid(p_kkt)
        @test length(panel_titles(p_kkt.html)) == 1
        @test occursin("KKT Certificate", p_kkt.html)

        tmp = mktempdir()
        sp = joinpath(tmp, "dc_kkt.html")
        plot_result(dc_sol; view=:kkt, title="Discrete RI KKT", save_path=sp)
        @test isfile(sp)

        dc32 = solve_ri_discrete(Float32.(U_dc), Float32.(mu_dc); omega=0.45f0)
        for vw in (:choice_curves, :kkt)
            p32 = plot_result(dc32; view=vw)
            check_plot(p32)
            assert_all_json_valid(p32)
        end

        for vw in (:choice_curves, :kkt)
            p_esc = plot_result(dc_sol; view=vw, title=HOSTILE_NAME)
            check_plot(p_esc)
            assert_all_json_valid(p_esc)
            assert_escapes(p_esc)
        end

        cp_nan = copy(dc_sol.cond_probs)
        cp_nan[1, 1] = NaN
        kkt_nan = copy(dc_sol.kkt_multipliers)
        kkt_nan[2] = NaN
        dc_nan = RIDiscreteChoiceSolution{Float64}(
            dc_sol.payoffs, dc_sol.prior, cp_nan, dc_sol.uncond_probs,
            dc_sol.consideration_set, kkt_nan, dc_sol.omega,
            dc_sol.mutual_info_bits, dc_sol.mutual_info_nats,
            dc_sol.expected_utility, dc_sol.net_welfare,
            dc_sol.converged, dc_sol.iterations,
        )
        for vw in (:choice_curves, :kkt)
            p_nan = plot_result(dc_nan; view=vw)
            assert_all_json_valid(p_nan)
            assert_nan_becomes_null(p_nan)
        end

        @test_throws ArgumentError plot_result(dc_sol; view=:bad_view)
    end

    # 9. RIIIADiagnostic{T}
    @testset "RIIIADiagnostic" begin
        diag = ri_iia_diagnostic([1.0 0.0; 0.0 1.0], [0.5, 0.5], [0.0, 1.0]; omega=0.25, units=:nats)

        p = plot_result(diag)
        check_plot(p)
        assert_all_json_valid(p)
        @test occursin("Base", p.html)
        @test occursin("Expanded", p.html)

        tmp = mktempdir()
        sp = joinpath(tmp, "ri_iia.html")
        plot_result(diag; title="Red Bus / Blue Bus", save_path=sp)
        @test isfile(sp)

        diag32 = ri_iia_diagnostic(Float32[1.0 0.0; 0.0 1.0], Float32[0.5, 0.5], Float32[0.0, 1.0]; omega=0.25f0, units=:nats)
        p32 = plot_result(diag32)
        check_plot(p32)
        assert_all_json_valid(p32)

        p_esc = plot_result(diag; title=HOSTILE_NAME)
        check_plot(p_esc)
        assert_all_json_valid(p_esc)
        assert_escapes(p_esc)

        ub_nan = copy(diag.base_solution.uncond_probs)
        ub_nan[1] = NaN
        base_nan = RIDiscreteChoiceSolution{Float64}(
            diag.base_solution.payoffs, diag.base_solution.prior, diag.base_solution.cond_probs,
            ub_nan, diag.base_solution.consideration_set, diag.base_solution.kkt_multipliers,
            diag.base_solution.omega, diag.base_solution.mutual_info_bits, diag.base_solution.mutual_info_nats,
            diag.base_solution.expected_utility, diag.base_solution.net_welfare,
            diag.base_solution.converged, diag.base_solution.iterations,
        )
        diag_nan = RIIIADiagnostic{Float64}(
            base_nan, diag.expanded_solution,
            diag.uncond_odds_ratio_shift, diag.cond_odds_ratio_max_diff, diag.max_iia_violation,
        )
        p_nan = plot_result(diag_nan)
        assert_all_json_valid(p_nan)
        assert_nan_becomes_null(p_nan)
    end

    # 10. RILogitModel{T}
    @testset "RILogitModel" begin
        # Minimal synthetic discrete-choice RI logit dataset
        M_alt, S_st, K_feat = 3, 2, 2
        E_env = 4
        beta_true = [1.2, -0.8]
        omega_true = 0.5
        menu = randn(rng, M_alt, S_st, K_feat, E_env)
        priors_mat = fill(0.5, S_st, E_env)
        env_id = repeat(1:E_env, inner=60)
        state_id = repeat(1:S_st, outer=120)
        choices = Vector{Int}(undef, length(env_id))
        for e in 1:E_env
            U_e = [sum(beta_true[k] * menu[j, s, k, e] for k in 1:K_feat) for j in 1:M_alt, s in 1:S_st]
            sol_e = solve_ri_discrete(U_e, priors_mat[:, e]; omega=omega_true, units=:nats)
            for idx in findall(==(e), env_id)
                s = state_id[idx]
                u_draw = rand(rng)
                cdf = cumsum(sol_e.cond_probs[:, s])
                choices[idx] = findfirst(>=(u_draw), cdf)
                choices[idx] === nothing && (choices[idx] = M_alt)
            end
        end

        m = estimate_ri_logit(choices, menu, env_id; state_id=state_id, priors=priors_mat,
                              normalize=:omega_one, varnames=["Attr1", "Attr2"])
        p = plot_result(m)
        check_plot(p)
        assert_all_json_valid(p)
        p90 = plot_result(m; conf_level=0.90, title="RI Logit Coefs")
        check_plot(p90)
        assert_all_json_valid(p90)
        @test occursin("90% CI", p90.html)
        @test_throws ArgumentError plot_result(m; conf_level=1.5)

        tmp = mktempdir()
        sp = joinpath(tmp, "ri_logit.html")
        plot_result(m; save_path=sp)
        @test isfile(sp)

        # K == 0 base_payoff-only model
        bp3 = [sum(beta_true[k] * menu[j, s, k, e] for k in 1:K_feat) for j in 1:M_alt, s in 1:S_st, e in 1:E_env]
        m0 = estimate_ri_logit(choices, bp3, env_id; state_id=state_id, priors=priors_mat)
        p0 = plot_result(m0)
        check_plot(p0)
        assert_all_json_valid(p0)
        @test occursin("omega", p0.html)

        # Float32, hostile varnames, NaN vcov
        m32 = RILogitModel{Float32}(
            m.choices, Float32.(m.menu_attributes), nothing, m.normalize,
            m.env_id, m.state_id, Float32.(m.priors),
            Float32.(m.beta), Float32(m.omega), Float32.(m.vcov_mat),
            RIDiscreteChoiceSolution{Float32}[], Float32(m.loglik), Float32(m.loglik_mnl),
            Float32(m.lr_vs_mnl), Float32(m.lr_pval), Float32(m.aic), Float32(m.bic),
            Float32(m.pseudo_r2), m.cov_type, m.varnames, m.converged, m.iterations, m.n_obs,
        )
        p32 = plot_result(m32)
        check_plot(p32)
        assert_all_json_valid(p32)

        m_esc = RILogitModel{Float64}(
            m.choices, m.menu_attributes, m.base_payoff, m.normalize,
            m.env_id, m.state_id, m.priors,
            m.beta, m.omega, m.vcov_mat, m.solutions, m.loglik, m.loglik_mnl,
            m.lr_vs_mnl, m.lr_pval, m.aic, m.bic, m.pseudo_r2, m.cov_type,
            [HOSTILE_NAME, "Attr2"], m.converged, m.iterations, m.n_obs,
        )
        p_esc = plot_result(m_esc; title=HOSTILE_NAME)
        check_plot(p_esc)
        assert_all_json_valid(p_esc)
        assert_escapes(p_esc)

        vc_nan = copy(m.vcov_mat)
        vc_nan[1, 1] = NaN
        m_nan = RILogitModel{Float64}(
            m.choices, m.menu_attributes, m.base_payoff, m.normalize,
            m.env_id, m.state_id, m.priors,
            m.beta, m.omega, vc_nan, m.solutions, m.loglik, m.loglik_mnl,
            m.lr_vs_mnl, m.lr_pval, m.aic, m.bic, m.pseudo_r2, m.cov_type,
            m.varnames, m.converged, m.iterations, m.n_obs,
        )
        p_nan = plot_result(m_nan)
        assert_all_json_valid(p_nan)
        assert_nan_becomes_null(p_nan)
    end
end
