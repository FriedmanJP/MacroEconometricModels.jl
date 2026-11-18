# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Plotting tests for all 9 ABCD State-Space & DSGE-VAR structs (#915 / ABCD-07)
#
# Covers plotrule Testing Rules 1-7:
#   1. ABCDSystem{T}                  (:eigenvalues, :singular_values)
#   2. InvertibilityResult{T}         (:eigenvalues)
#   3. InnovationsRepresentation{T}   (:eigenvalues, :gain)
#   4. ShockRecoverabilityResult{T}   (:leads, :contemp)
#   5. ObservableSearchResult{T}      (metric=:min_R2, :mean_R2, :rho)
#   6. DSGEVARIRFAudit{T}             (:irf_comparison, :wedges)
#   7. DSGEVARPrior{T}                (:phi, :sigma, :both)
#   8. DSGEVARLambdaProfile{T}        (curve + lambda_hat + logml_inf + empty grid)
#   9. DSGEVARPosterior{T}            (:irf, :draws, :lambda — fixed-θ & joint)
# =============================================================================

using Test
using LinearAlgebra
using Random
using Distributions
using MacroEconometricModels
const _MAB = MacroEconometricModels

isdefined(Main, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

@testset "ABCD & DSGE-VAR Plotting (#915 / ABCD-07)" begin

    # Shared canonical fixtures (FVRSW Permanent Income + 3-eq NK DSGE)
    spec_fv = @dsge begin
        parameters: R_gross = 1.2, sigma_w = 1.0
        endogenous: y, c, surplus
        exogenous: w

        c[t] = c[t-1] + sigma_w * (1 - 1/R_gross) * w[t]
        surplus[t] = -c[t-1] + sigma_w * (1/R_gross) * w[t]
        surplus[t] = y[t] - c[t]

        steady_state = [0.0, 0.0, 0.0]
    end
    sol_fv = solve(spec_fv; method=:gensys)
    abcd_fv_ni = abcd_representation(sol_fv, [:surplus])
    abcd_fv_inv = abcd_representation(sol_fv, [:c])

    nk_spec = @dsge begin
        parameters: beta = 0.99, kappa = 0.30, psi1 = 1.50, psi2 = 0.25,
                    rho_r = 0.50, rho_d = 0.70, sig_r = 0.01, sig_d = 0.01, sig_pi = 0.01
        endogenous: y, pi, r, xd
        exogenous: e_r, e_d, e_pi
        steady_state: [0.0, 0.0, 0.0, 0.0]

        y[t] = y[t+1] - (r[t] - pi[t+1]) + xd[t]
        pi[t] = beta * pi[t+1] + kappa * y[t] + sig_pi * e_pi[t]
        r[t] = rho_r * r[t-1] + (1 - rho_r) * (psi1 * pi[t] + psi2 * y[t]) + sig_r * e_r[t]
        xd[t] = rho_d * xd[t-1] + sig_d * e_d[t]
    end
    sol_nk = solve(nk_spec)
    obs_nk = [:y, :pi, :r]
    abcd_nk = abcd_representation(sol_nk, obs_nk)

    sim_all = simulate(sol_nk, 350; seed=2026)
    obs_idx = [findfirst(==(sym), sol_nk.spec.endog) for sym in obs_nk]
    Y_nk = Matrix{Float64}(sim_all[201:end, obs_idx])
    p_var = 2
    T_eff = size(Y_nk, 1) - p_var

    # -------------------------------------------------------------------------
    # Contract check: every new plot_result method has EXACTLY 1 positional arg
    # -------------------------------------------------------------------------
    @testset "Single positional argument contract" begin
        abcd_types = (
            ABCDSystem, InvertibilityResult, InnovationsRepresentation,
            ShockRecoverabilityResult, ObservableSearchResult, DSGEVARIRFAudit,
            DSGEVARPrior, DSGEVARLambdaProfile, DSGEVARPosterior,
        )
        for T_ab in abcd_types
            ms = [m for m in methods(_MAB.plot_result)
                  if let sig = Base.unwrap_unionall(m.sig)
                      sig isa DataType && length(sig.parameters) >= 2 &&
                      (let arg = sig.parameters[2]
                           ua = arg isa UnionAll ? Base.unwrap_unionall(arg) : arg
                           ua isa DataType && Base.typename(ua).wrapper === T_ab
                       end)
                  end]
            @test length(ms) == 1
            sig = Base.unwrap_unionall(ms[1].sig)
            @test length(sig.parameters) == 2
        end
    end

    # -------------------------------------------------------------------------
    # 1. ABCDSystem
    # -------------------------------------------------------------------------
    @testset "1. ABCDSystem" begin
        p_eig = plot_result(abcd_nk)
        check_plot(p_eig); assert_all_json_valid(p_eig)
        @test occursin("ellipse", p_eig.html)
        @test occursin("STABLE", p_eig.html)
        @test occursin("n=$(nstates(abcd_nk)), k=$(nobservables(abcd_nk)), m=$(nshocks(abcd_nk))", p_eig.html)

        p_sv = plot_result(abcd_nk; view=:singular_values)
        check_plot(p_sv); assert_all_json_valid(p_sv)
        @test length(panel_titles(p_sv.html)) == 2
        @test occursin("Impact Matrix D Singular Values", p_sv.html)
        @test occursin("State Transition A Singular Values", p_sv.html)

        # Unit-root FVRSW & zero-state system
        p_ur = plot_result(abcd_fv_ni; view=:eigenvalues)
        check_plot(p_ur); assert_all_json_valid(p_ur)
        @test occursin("UNSTABLE", p_ur.html)

        sys_zero = ABCDSystem(zeros(0, 0), zeros(0, 2), zeros(2, 0), [1.0 0.0; 0.0 2.0])
        p_z1 = plot_result(sys_zero; view=:eigenvalues)
        p_z2 = plot_result(sys_zero; view=:singular_values)
        check_plot(p_z1); assert_all_json_valid(p_z1)
        check_plot(p_z2); assert_all_json_valid(p_z2)
        @test length(panel_titles(p_z2.html)) == 1
        @test occursin("STATIC", p_z1.html)

        # NaN handling -> null
        sys_nan = ABCDSystem([NaN 0.0; 0.0 0.5], [1.0; 0.0;;], [1.0 0.0], [NaN;;])
        assert_nan_becomes_null(plot_result(sys_nan; view=:eigenvalues))
        assert_nan_becomes_null(plot_result(sys_nan; view=:singular_values))

        # Escaping, title, save_path, bad view
        tmp = tempname() * ".html"
        p_t = plot_result(abcd_nk; title=HOSTILE_NAME, save_path=tmp)
        @test isfile(tmp) && startswith(strip(read(tmp, String)), "<!DOCTYPE html>")
        rm(tmp; force=true)
        assert_escapes(p_t)
        @test_throws ArgumentError plot_result(abcd_nk; view=:bogus)
    end

    # -------------------------------------------------------------------------
    # 2. InvertibilityResult
    # -------------------------------------------------------------------------
    @testset "2. InvertibilityResult" begin
        inv_ni = poor_mans_invertibility(abcd_fv_ni)
        p_ni = plot_result(inv_ni)
        check_plot(p_ni); assert_all_json_valid(p_ni)
        @test occursin("NON-INVERTIBLE", p_ni.html)
        @test occursin("ellipse", p_ni.html)

        inv_ok = poor_mans_invertibility(abcd_fv_inv)
        p_ok = plot_result(inv_ok)
        check_plot(p_ok); assert_all_json_valid(p_ok)
        @test occursin("INVERTIBLE", p_ok.html)

        # Complex zeros
        sys_cx = ABCDSystem([0.5 0.3; -0.3 0.5], [1.0 0.0; 0.0 1.0],
                            [0.2 0.1; -0.1 0.2], [1.0 0.0; 0.0 1.0])
        p_cx = plot_result(poor_mans_invertibility(sys_cx))
        check_plot(p_cx); assert_all_json_valid(p_cx)

        # Infinite zeros (delayed news shock)
        sys_delay = ABCDSystem([0.0 0.0; 1.0 0.0], [1.0; 0.0;;], [1.0 -0.5], [0.0;;])
        p_infz = plot_result(poor_mans_invertibility(sys_delay))
        check_plot(p_infz); assert_all_json_valid(p_infz)
        @test occursin("infinite zero", p_infz.html)

        # Underdetermined (k < m -> empty zeros, NaN max_modulus)
        sys_wide = ABCDSystem([0.5;;], [1.0 0.5], [1.0;;], [1.0 0.5])
        p_wide = plot_result(poor_mans_invertibility(sys_wide))
        check_plot(p_wide); assert_all_json_valid(p_wide)
        @test occursin("Underdetermined", p_wide.html)

        # NaN in zero_re -> null + hostile observable name escaping
        inv_nan = _MAB.InvertibilityResult{Float64}(
            false, :non_invertible_zeros, NaN, [NaN, 0.5], [0.0, 0.2],
            0, 0, [0.5 0.0; 0.0 0.5], [1.0 0.0; 0.0 1.0], Matrix{Float64}[],
            2, 2, 2, 2, 1e-8, [HOSTILE_NAME, "y2"], ["w1", "w2"],
        )
        p_nan = plot_result(inv_nan)
        assert_nan_becomes_null(p_nan)
        assert_escapes(p_nan)

        @test_throws ArgumentError plot_result(inv_ok; view=:bogus)
    end

    # -------------------------------------------------------------------------
    # 3. InnovationsRepresentation
    # -------------------------------------------------------------------------
    @testset "3. InnovationsRepresentation" begin
        inn_ni = innovations_representation(abcd_fv_ni)
        p_eig = plot_result(inn_ni; view=:eigenvalues)
        check_plot(p_eig); assert_all_json_valid(p_eig)
        @test occursin("ellipse", p_eig.html)
        @test occursin("non-invertible", p_eig.html)

        p_gain = plot_result(inn_ni; view=:gain)
        check_plot(p_gain); assert_all_json_valid(p_gain)
        @test occursin("legend-gradient", p_gain.html)
        @test occursin("scaleType = \"diverging\"", p_gain.html)

        inn_nk = innovations_representation(abcd_nk)
        p_nk_g = plot_result(inn_nk; view=:gain)
        check_plot(p_nk_g); assert_all_json_valid(p_nk_g)

        # Zero-state system: :eigenvalues succeeds, :gain throws ArgumentError
        inn_zero = innovations_representation(
            ABCDSystem(zeros(0, 0), zeros(0, 2), zeros(2, 0), [1.0 0.0; 0.0 2.0]))
        check_plot(plot_result(inn_zero; view=:eigenvalues))
        @test_throws ArgumentError plot_result(inn_zero; view=:gain)

        # NaN in K -> null in both views + hostile escaping
        inn_nan = _MAB.InnovationsRepresentation{Float64}(
            [0.5 0.0; 0.0 0.4], [NaN 0.1; 0.2 -0.3], [1.0 0.0; 0.0 1.0],
            [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0], [0.1 0.0; 0.0 0.1],
            0.0, true, [0.1, 0.2], [HOSTILE_NAME, "x2"], [HOSTILE_NAME, "y2"],
        )
        assert_nan_becomes_null(plot_result(inn_nan; view=:eigenvalues))
        p_gnan = plot_result(inn_nan; view=:gain)
        assert_nan_becomes_null(p_gnan)
        assert_escapes(p_gnan)

        @test_throws ArgumentError plot_result(inn_ni; view=:bogus)
    end

    # -------------------------------------------------------------------------
    # 4. ShockRecoverabilityResult
    # -------------------------------------------------------------------------
    @testset "4. ShockRecoverabilityResult" begin
        rec_fv = shock_recoverability(abcd_fv_ni; max_lead=12)
        p_fv = plot_result(rec_fv)
        check_plot(p_fv); assert_all_json_valid(p_fv)
        @test occursin("d3.format('d')", p_fv.html)
        @test occursin("non-invertible", p_fv.html)

        # Multi-shock Leeper-Walker-Yang fiscal foresight
        alpha, kappa, theta = 0.5, 0.4, 0.8
        A_lwy = [alpha -kappa * theta -kappa; 0.0 0.0 0.0; 0.0 1.0 0.0]
        B_lwy = [1.0 -kappa * theta^2; 0.0 1.0; 0.0 0.0]
        C_lwy = [alpha -kappa * theta -kappa; 0.0 0.0 1.0]
        D_lwy = [1.0 -kappa * theta^2; 0.0 0.0]
        abcd_lwy = ABCDSystem(A_lwy, B_lwy, C_lwy, D_lwy;
                              obs_names=[:k_obs, :tau_obs],
                              shock_names=[:u_A, :eps_tau_news])
        rec_lwy = shock_recoverability(abcd_lwy; max_lead=8)

        p_leads = plot_result(rec_lwy; view=:leads)
        check_plot(p_leads); assert_all_json_valid(p_leads)
        @test "u_A" in series_names(p_leads.html)
        @test "eps_tau_news" in series_names(p_leads.html)

        p_cont = plot_result(rec_lwy; view=:contemp)
        check_plot(p_cont); assert_all_json_valid(p_cont)
        @test occursin("orientation", p_cont.html)

        # Shock selection Int & String + errors + max_shocks cap (C7)
        @test series_names(plot_result(rec_lwy; shock=1).html) == ["u_A"]
        @test series_names(plot_result(rec_lwy; shock="eps_tau_news").html) == ["eps_tau_news"]
        @test_throws ArgumentError plot_result(rec_lwy; shock=99)
        @test_throws ArgumentError plot_result(rec_lwy; shock="nope")
        p_cap = plot_result(rec_lwy; max_shocks=1)
        @test occursin("Showing 1 of 2 shocks", p_cap.html)

        # max_lead == 0 falls back cleanly to contemporaneous bar
        rec_0 = shock_recoverability(abcd_lwy; max_lead=0)
        check_plot(plot_result(rec_0; view=:leads))
        check_plot(plot_result(rec_0; view=:contemp))

        # NaN + hostile escaping
        r2m_nan = copy(rec_lwy.r2_matrix)
        r2m_nan[3, 1] = NaN
        rec_nan = _MAB.ShockRecoverabilityResult{Float64}(
            r2m_nan, [NaN, rec_lwy.r2_contemp[2]],
            [HOSTILE_NAME, "w2"], ["y1", "y2"], false, 8,
        )
        p_rnan = plot_result(rec_nan; view=:leads)
        assert_nan_becomes_null(p_rnan)
        assert_escapes(p_rnan)
        assert_nan_becomes_null(plot_result(rec_nan; view=:contemp))

        @test_throws ArgumentError plot_result(rec_lwy; view=:bogus)
    end

    # -------------------------------------------------------------------------
    # 5. ObservableSearchResult
    # -------------------------------------------------------------------------
    @testset "5. ObservableSearchResult" begin
        res_fv = search_invertible_observables(sol_fv, 1)
        for met in (:min_R2, :mean_R2, :rho)
            pm = plot_result(res_fv; metric=met)
            check_plot(pm); assert_all_json_valid(pm)
            @test "Invertible" in series_names(pm.html)
            @test "Non-invertible" in series_names(pm.html)
            @test occursin("[✓]", pm.html)
            @test occursin("[✗]", pm.html)
        end

        # max_candidates cap (C7)
        pc = plot_result(res_fv; max_candidates=2)
        @test occursin("(2 of 3)", pc.html)
        @test occursin("Showing 2 of 3 candidate subsets", pc.html)

        # NaN + Inf + hostile escaping
        res_edge = _MAB.ObservableSearchResult{Float64}(
            [[Symbol(HOSTILE_NAME)], [:surplus]],
            [true, false],
            [0.4, Inf],
            [NaN, 0.69],
            [0.9, 0.69],
            [1.0, Inf],
            [:invertible, :non_invertible_singular_D],
            1, 2, 2, false, [1], ["w"],
        )
        pe = plot_result(res_edge; metric=:min_R2)
        assert_nan_becomes_null(pe)
        assert_escapes(pe)
        check_plot(plot_result(res_edge; metric=:rho)); assert_all_json_valid(plot_result(res_edge; metric=:rho))

        @test_throws ArgumentError plot_result(res_fv; metric=:bogus)
        @test_throws ArgumentError plot_result(res_fv; max_candidates=0)
    end

    # -------------------------------------------------------------------------
    # 6. DSGEVARIRFAudit
    # -------------------------------------------------------------------------
    @testset "6. DSGEVARIRFAudit" begin
        audit_fv = compare_dsge_svar_irf(sol_fv, 4, 15; observables=[:surplus])
        p_cmp = plot_result(audit_fv; view=:irf_comparison)
        check_plot(p_cmp); assert_all_json_valid(p_cmp)
        @test "DSGE (True)" in series_names(p_cmp.html)
        @test "Wold (best rot.)" in series_names(p_cmp.html)
        @test "SVAR(4)" in series_names(p_cmp.html)
        @test occursin("d3.format('d')", p_cmp.html)

        p_wdg = plot_result(audit_fv; view=:wedges)
        check_plot(p_wdg); assert_all_json_valid(p_wdg)
        @test "Total RMSE" in series_names(p_wdg.html)
        @test "Non-invertibility" in series_names(p_wdg.html)
        @test "Lag truncation" in series_names(p_wdg.html)
        @test "Identification" in series_names(p_wdg.html)

        # Multi-variable identified NK audit + var/shock selection + C7 cap + C5 guard
        audit_nk = compare_dsge_svar_irf(abcd_nk, 2, 10; identification=:cholesky)
        p_nk_all = plot_result(audit_nk; view=:irf_comparison)
        check_plot(p_nk_all); assert_all_json_valid(p_nk_all)
        @test length(panel_titles(p_nk_all.html)) == 9
        @test "Wold (cholesky)" in series_names(p_nk_all.html)

        p_sub = plot_result(audit_nk; var="y", shock=1)
        @test length(panel_titles(p_sub.html)) == 1
        @test panel_titles(p_sub.html) == ["y ← e_r"]
        p_sub2 = plot_result(audit_nk; var=2, shock="e_pi")
        @test panel_titles(p_sub2.html) == ["pi ← e_pi"]

        p_cap = plot_result(audit_nk; max_panels=4)
        @test length(panel_titles(p_cap.html)) == 4
        @test occursin("Showing 4 of 9 response-shock panels", p_cap.html)

        @test_throws ArgumentError plot_result(audit_nk; var="bad")
        @test_throws ArgumentError plot_result(audit_nk; shock=99)
        @test_throws ArgumentError plot_result(audit_nk; view=:wedges, var=1)
        @test_throws ArgumentError plot_result(audit_nk; view=:wedges, shock="e_r")
        @test_throws ArgumentError plot_result(audit_nk; view=:bogus)

        # NaN + hostile escaping
        id_nan = copy(audit_fv.irf_dsge); id_nan[3, 1, 1] = NaN
        nbh_nan = copy(audit_fv.noninv_by_horizon); nbh_nan[2] = NaN
        audit_nan = _MAB.DSGEVARIRFAudit{Float64}(
            id_nan, audit_fv.irf_wold_best, audit_fv.irf_wold_id, audit_fv.irf_svar,
            audit_fv.noninvertibility_wedge, audit_fv.truncation_wedge,
            audit_fv.identification_wedge, audit_fv.total_rmse,
            audit_fv.rmse_by_horizon, nbh_nan, audit_fv.trunc_by_horizon, audit_fv.id_by_horizon,
            audit_fv.Q_best, audit_fv.Q_id_inf, audit_fv.Q_id_p,
            audit_fv.p, audit_fv.horizon, audit_fv.identification, audit_fv.is_invertible,
            [HOSTILE_NAME], [HOSTILE_NAME],
        )
        pan_nan = plot_result(audit_nan; view=:irf_comparison)
        assert_nan_becomes_null(pan_nan)
        assert_escapes(pan_nan)
        assert_nan_becomes_null(plot_result(audit_nan; view=:wedges))
    end

    # -------------------------------------------------------------------------
    # 7. DSGEVARPrior
    # -------------------------------------------------------------------------
    @testset "7. DSGEVARPrior" begin
        prior = dsge_var_prior(sol_nk, p_var, 1.5, T_eff; observables=obs_nk)
        for vw in (:phi, :sigma, :both)
            pp = plot_result(prior; view=vw)
            check_plot(pp); assert_all_json_valid(pp)
            @test occursin("legend-gradient", pp.html)
            @test occursin("scaleType = \"diverging\"", pp.html)
        end
        @test length(panel_titles(plot_result(prior; view=:both).html)) == 2
        @test occursin("Φ*", plot_result(prior; view=:phi).html)
        @test occursin("Σ_u*", plot_result(prior; view=:sigma).html)

        # Lambda = Inf & no-intercept
        prior_inf = dsge_var_prior(sol_nk, p_var, Inf, T_eff;
                                   observables=obs_nk, include_intercept=false)
        check_plot(plot_result(prior_inf; view=:both))
        @test occursin("∞", plot_result(prior_inf).html)

        # NaN + hostile escaping
        phi_nan = copy(prior.Phi_star); phi_nan[1, 1] = NaN
        sig_nan = copy(prior.Sigma_u_star); sig_nan[1, 2] = NaN
        prior_nan = _MAB.DSGEVARPrior{Float64}(
            prior.Gamma_yy, prior.Gamma_xy, prior.Gamma_xx,
            phi_nan, sig_nan, prior.Omega_star,
            prior.lambda, prior.lambda_min, prior.T_obs, prior.p,
            [HOSTILE_NAME, "pi", "r"], prior.shock_names,
        )
        p_pn = plot_result(prior_nan; view=:both)
        assert_nan_becomes_null(p_pn)
        assert_escapes(p_pn)

        @test_throws ArgumentError plot_result(prior; view=:bogus)
    end

    # -------------------------------------------------------------------------
    # 8. DSGEVARLambdaProfile
    # -------------------------------------------------------------------------
    @testset "8. DSGEVARLambdaProfile" begin
        prof = select_dsge_var_lambda(sol_nk, Y_nk, p_var; observables=obs_nk)
        pl = plot_result(prof)
        check_plot(pl); assert_all_json_valid(pl)
        @test "Log ML(λ)" in series_names(pl.html)
        @test "Log ML(λ = ∞)" in series_names(pl.html)
        @test occursin("λ̂", pl.html)

        # Infinite-only profile (empty finite grid)
        prof_inf = select_dsge_var_lambda(sol_nk, Y_nk, p_var;
                                          observables=obs_nk, lambda_grid=[Inf])
        pi_inf = plot_result(prof_inf)
        check_plot(pi_inf); assert_all_json_valid(pi_inf)
        @test occursin("No finite λ grid points", pi_inf.html)

        # NaN in curve -> null + hostile title escaping
        lmls_nan = copy(prof.log_marginal_likelihoods); lmls_nan[2] = NaN
        prof_nan = _MAB.DSGEVARLambdaProfile{Float64}(
            prof.lambdas, lmls_nan, prof.lambda_hat, prof.logml_hat, prof.logml_inf,
            prof.lambda_min, prof.p, prof.T_obs, [HOSTILE_NAME, "pi", "r"],
        )
        p_ln = plot_result(prof_nan; title=HOSTILE_NAME)
        assert_nan_becomes_null(p_ln)
        assert_escapes(p_ln)
    end

    # -------------------------------------------------------------------------
    # 9. DSGEVARPosterior
    # -------------------------------------------------------------------------
    @testset "9. DSGEVARPosterior" begin
        post_fixed = estimate_dsge_var(sol_nk, Y_nk, p_var;
                                       lambda=1.5, observables=obs_nk, n_draws=40, seed=11)

        # :irf view (posterior bands + dogmatic DSGE line)
        p_irf = plot_result(post_fixed; view=:irf, horizon=10)
        check_plot(p_irf); assert_all_json_valid(p_irf)
        @test length(panel_titles(p_irf.html)) == 9
        @test "DSGE-VAR Median" in series_names(p_irf.html)
        @test "Dogmatic DSGE (λ=∞)" in series_names(p_irf.html)
        @test occursin("d3.format('d')", p_irf.html)

        p_mean = plot_result(post_fixed; view=:irf, var="y", shock="e_r", horizon=8, stat=:mean)
        check_plot(p_mean); assert_all_json_valid(p_mean)
        @test length(panel_titles(p_mean.html)) == 1
        @test "DSGE-VAR Mean" in series_names(p_mean.html)

        p_chol = plot_result(post_fixed; view=:irf, var=1, shock=2, horizon=8, method=:cholesky)
        check_plot(p_chol); assert_all_json_valid(p_chol)

        # Cap on un-filtered grid (C7)
        p_cap = plot_result(post_fixed; view=:irf, horizon=6, max_panels=3)
        @test length(panel_titles(p_cap.html)) == 3
        @test occursin("Showing 3 of 9 response-shock panels", p_cap.html)

        # :draws & :lambda in fixed-θ mode
        p_dr_f = plot_result(post_fixed; view=:draws, max_panels=4)
        check_plot(p_dr_f); assert_all_json_valid(p_dr_f)
        @test length(panel_titles(p_dr_f.html)) == 4
        @test occursin("Showing 2 of", p_dr_f.html)

        p_lam_f = plot_result(post_fixed; view=:lambda)
        check_plot(p_lam_f); assert_all_json_valid(p_lam_f)
        @test "Log ML(λ)" in series_names(p_lam_f.html)

        # Joint (θ, Φ, Σ_u) posterior
        priors_j = Dict{Symbol,Distribution}(
            :psi1 => Distributions.Uniform(1.15, 2.50),
            :rho_r => Distributions.Beta(4.0, 4.0),
        )
        post_joint = estimate_dsge_var(nk_spec, Y_nk, p_var;
                                       priors=priors_j, lambda=1.5, method=:rwmh,
                                       observables=obs_nk, n_draws=30, burnin=15, seed=321)
        check_plot(plot_result(post_joint; view=:irf, horizon=8))
        p_dr_j = plot_result(post_joint; view=:draws, params=[:psi1], thin=2)
        check_plot(p_dr_j); assert_all_json_valid(p_dr_j)
        @test length(panel_titles(p_dr_j.html)) == 2
        @test occursin("psi1", p_dr_j.html)

        p_lam_j = plot_result(post_joint; view=:lambda)
        check_plot(p_lam_j); assert_all_json_valid(p_lam_j)
        @test occursin("Log Posterior Trace", p_lam_j.html)
        @test occursin("Log Posterior Density", p_lam_j.html)

        # Dogmatic λ = Inf posterior
        post_inf = estimate_dsge_var(sol_nk, Y_nk, p_var;
                                     lambda=Inf, observables=obs_nk, n_draws=20, seed=42)
        check_plot(plot_result(post_inf; view=:irf, var=1, shock=1, horizon=6))
        check_plot(plot_result(post_inf; view=:lambda))

        # Validation errors (C3 / C5)
        @test_throws ArgumentError plot_result(post_fixed; view=:bogus)
        @test_throws ArgumentError plot_result(post_fixed; view=:irf, var="nope")
        @test_throws ArgumentError plot_result(post_fixed; view=:irf, shock=99)
        @test_throws ArgumentError plot_result(post_fixed; view=:irf, stat=:bogus)
        @test_throws ArgumentError plot_result(post_fixed; view=:irf, params=1)
        @test_throws ArgumentError plot_result(post_fixed; view=:draws, var=1)
        @test_throws ArgumentError plot_result(post_fixed; view=:lambda, shock=1)
        @test_throws ArgumentError plot_result(post_joint; view=:draws, params="missing_p")

        # NaN & hostile escaping (row 2 = lag-1 AR coef affects dogmatic IRF)
        phi_nan = copy(post_joint.dsge_prior.Phi_star); phi_nan[2, 1] = NaN
        pr_nan = _MAB.DSGEVARPrior{Float64}(
            post_joint.dsge_prior.Gamma_yy, post_joint.dsge_prior.Gamma_xy,
            post_joint.dsge_prior.Gamma_xx, phi_nan, post_joint.dsge_prior.Sigma_u_star,
            post_joint.dsge_prior.Omega_star, post_joint.dsge_prior.lambda,
            post_joint.dsge_prior.lambda_min, post_joint.dsge_prior.T_obs,
            post_joint.dsge_prior.p, [HOSTILE_NAME, "pi", "r"],
            [HOSTILE_NAME, "e_d", "e_pi"],
        )
        td_nan = copy(post_joint.theta_draws); td_nan[2, 1] = NaN
        lp_nan = copy(post_joint.log_posterior); lp_nan[3] = NaN
        sig_nan = copy(post_joint.Sigma_draws); sig_nan[2, 1, 1] = NaN
        post_nan = _MAB.DSGEVARPosterior{Float64}(
            post_joint.B_draws, sig_nan, post_joint.B0_draws,
            post_joint.lambda, post_joint.log_marginal_likelihood,
            post_joint.n_draws, post_joint.p, post_joint.n, post_joint.data,
            post_joint.method, [HOSTILE_NAME, "pi", "r"], [HOSTILE_NAME, "e_d", "e_pi"],
            td_nan, [Symbol(HOSTILE_NAME), :rho_r], lp_nan,
            post_joint.acceptance_rate, pr_nan, post_joint.manifest,
        )
        p_in = plot_result(post_nan; view=:irf, var=1, shock=1, horizon=6)
        assert_nan_becomes_null(p_in); assert_escapes(p_in)
        p_dn = plot_result(post_nan; view=:draws)
        assert_nan_becomes_null(p_dn); assert_escapes(p_dn)
        assert_nan_becomes_null(plot_result(post_nan; view=:lambda))

        # Fixed-θ non-finite Sigma_draws fallback in view=:lambda
        sig_fnan = copy(post_fixed.Sigma_draws); sig_fnan[2, 1, 1] = NaN
        post_fnan = _MAB.DSGEVARPosterior{Float64}(
            post_fixed.B_draws, sig_fnan, post_fixed.B0_draws,
            post_fixed.lambda, post_fixed.log_marginal_likelihood,
            post_fixed.n_draws, post_fixed.p, post_fixed.n, zeros(0, 3),
            post_fixed.method, post_fixed.varnames, post_fixed.shock_names,
            nothing, Symbol[], nothing, post_fixed.acceptance_rate,
            post_fixed.dsge_prior, post_fixed.manifest,
        )
        assert_nan_becomes_null(plot_result(post_fnan; view=:lambda))
    end
end
