# =============================================================================
# NGID & SSVARMA plotting tests (#908 / NGID-08)
#
# Covers plotrule Testing Rules 1–7 for all 8 new plot_result dispatches:
#   1. SVARMAModel{T}                 (:regime_map, :shock_densities, :roots)
#   2. LMSSVARResult{T}               (:qq, :densities)
#   3. LanneLuotoGMMResult{T}         (:qq, :densities)
#   4. NonGaussianDiagnostics{T}
#   5. SVARHypothesisTest{T}
#   6. GMMMomentSelectionResult{T}
#   7. FundamentalnessTestResult{T}
#   8. SufficientInformationTestResult{T}
# =============================================================================

using Test
using Random
using LinearAlgebra
using Distributions
using MacroEconometricModels

const _MN = MacroEconometricModels
const _suppress_warnings = _MN._suppress_warnings

# Shared plotting test assertions (self-bootstrap when run standalone)
isdefined(Main, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

@testset "NGID & SSVARMA Plots (plot_result)" begin

    # -------------------------------------------------------------------------
    # Shared fast fixtures (deterministic, DGP-lint compliant)
    # -------------------------------------------------------------------------
    _suppress_warnings() do
        sim2 = dgp_ssvarma(; T=180, Phi=[0.4 0.05; -0.05 0.35],
                           Theta=[-0.5 0.0; 1.0 -2.0], C=[0.6 1.0; 0.8 0.4],
                           dist=:student_t, nu=5.0, seed=90801)
        m_ssv_ml = estimate_ssvarma(sim2.Y, 1, 1; method=:ml, distribution=:student_t,
                                    n_starts=2, max_iter=120,
                                    varnames=["Output", "Prices"],
                                    shock_names=["Supply", "Demand"], seed=90802)
        m_ssv_gmm = estimate_ssvarma(sim2.Y, 1, 1; method=:gmm, max_iter=120,
                                     varnames=["Output", "Prices"],
                                     shock_names=["Supply", "Demand"], seed=90803)

        sim3 = dgp_ssvarma(; T=160, Phi=0.35 * Matrix{Float64}(I, 3, 3),
                           Theta=diagm([-0.4, -1.6, 0.3]), C=[1.0 0.2 0.0; -0.2 0.9 0.15; 0.1 -0.15 0.85],
                           dist=:student_t, nu=5.0, seed=90804)
        var3 = estimate_var(sim3.Y, 1; varnames=["y1", "y2", "y3"])
        lms3 = estimate_lms_svar(var3; distributions=[:student_t, :skewed_t, :ged],
                                 method=:three_step, n_starts=2, max_iter=150,
                                 shock_names=["s1", "s2", "s3"], seed=90805)
        diag3 = diagnose_nongaussian_svar(lms3; lags=3)
        ht_wald = test_svar_hypothesis(lms3, recursiveness_hypothesis(3); method=:wald)
        ht_boot = test_svar_hypothesis(lms3, impact_zero_hypothesis(3, [(1, 2)]);
                                       method=:lr, bootstrap=:iid, n_boot=19, seed=90806)

        sim_gmm = dgp_ssvarma(; T=240, Phi=[0.4 0.05; -0.05 0.35], Theta=zeros(2, 2, 0),
                              C=[1.2 -0.35; 0.30 0.95], dist=:student_t, nu=5.0, seed=90807)
        res_ll = estimate_lanne_luoto_gmm(sim_gmm.Y, 1; moment_selection=:msc_rmsc,
                                          n_starts=2, max_iter=120,
                                          varnames=["GDP", "Inflation"],
                                          shock_names=["Tech", "Policy"], seed=90808)
        sel_ll = res_ll.selection

        var2_for_fund = estimate_var(sim2.Y, 2)
        res_fund = test_fundamentalness(var2_for_fund; max_lag=4, n_boot=99, seed=90809)

        rng = Xoshiro(90810)
        X_extra = randn(rng, size(sim2.Y, 1), 8)
        X_panel = hcat(sim2.Y, X_extra)
        res_suff = test_sufficient_information(var2_for_fund, X_panel; r_max=3, p_f=2)

        # ---------------------------------------------------------------------
        # 1. SVARMAModel{T}
        # ---------------------------------------------------------------------
        @testset "1. SVARMAModel" begin
            for vw in (:regime_map, :shock_densities, :roots)
                p = plot_result(m_ssv_ml; view=vw)
                check_plot(p); assert_all_json_valid(p)
            end
            @test occursin("GMR Fig. 6", plot_result(m_ssv_ml; view=:regime_map).html)
            @test occursin("Fitted", plot_result(m_ssv_ml; view=:shock_densities).html)
            @test occursin("ellipse", plot_result(m_ssv_ml; view=:roots).html)

            # GMM (nonparametric) shock densities
            p_gmm_d = plot_result(m_ssv_gmm; view=:shock_densities)
            check_plot(p_gmm_d); assert_all_json_valid(p_gmm_d)
            @test "Empirical KDE" in series_names(p_gmm_d.html)
            @test "N(0,1)" in series_names(p_gmm_d.html)

            # Shock selection by Int and String
            p_s1 = plot_result(m_ssv_ml; view=:shock_densities, shock=1)
            p_s2 = plot_result(m_ssv_ml; view=:shock_densities, shock="Demand")
            check_plot(p_s1); assert_all_json_valid(p_s1)
            check_plot(p_s2); assert_all_json_valid(p_s2)
            @test length(panel_titles(p_s1.html)) == 1
            @test occursin("Demand", panel_titles(p_s2.html)[1])

            # Error paths (view + shock rules)
            @test_throws ArgumentError plot_result(m_ssv_ml; view=:bogus)
            @test_throws ArgumentError plot_result(m_ssv_ml; view=:regime_map, shock=1)
            @test_throws ArgumentError plot_result(m_ssv_ml; view=:roots, shock="Supply")
            @test_throws ArgumentError plot_result(m_ssv_ml; view=:shock_densities, shock=99)
            @test_throws ArgumentError plot_result(m_ssv_ml; view=:shock_densities, shock="Missing")

            # n != 2 and q == 0 fallback checks
            m_3d = SVARMAModel(sim3.Y, sim3.Phi, sim3.Theta, sim3.C;
                               distributions=[:student_t, :pearson_iv, :skew_normal],
                               dist_params=[[5.0], [0.2, 4.5], [1.0]])
            for vw in (:regime_map, :shock_densities, :roots)
                p3 = plot_result(m_3d; view=vw)
                check_plot(p3); assert_all_json_valid(p3)
            end
            m_1d = SVARMAModel(sim2.Y[:, 1:1], [0.3;;], [-1.8;;], [1.0;;])
            for vw in (:regime_map, :shock_densities, :roots)
                p1 = plot_result(m_1d; view=vw)
                check_plot(p1); assert_all_json_valid(p1)
            end
            m_q0 = SVARMAModel(sim2.Y, sim2.Phi, nothing, sim2.C)
            for vw in (:regime_map, :shock_densities, :roots)
                pq0 = plot_result(m_q0; view=vw)
                check_plot(pq0); assert_all_json_valid(pq0)
            end

            # NaN handling & hostile escaping
            raw_ssv = Any[getfield(m_ssv_ml, f) for f in fieldnames(SVARMAModel)]
            idx_re = findfirst(==(:ma_eig_re), fieldnames(SVARMAModel))
            idx_sn = findfirst(==(:shock_names), fieldnames(SVARMAModel))
            raw_nan = copy(raw_ssv)
            raw_nan[idx_re] = copy(m_ssv_ml.ma_eig_re); raw_nan[idx_re][1] = NaN
            m_nan = SVARMAModel{Float64}(raw_nan...)
            assert_nan_becomes_null(plot_result(m_nan; view=:regime_map))
            assert_nan_becomes_null(plot_result(m_nan; view=:roots))

            raw_esc = copy(raw_ssv)
            raw_esc[idx_sn] = [HOSTILE_NAME, "Demand"]
            m_esc = SVARMAModel{Float64}(raw_esc...)
            assert_escapes(plot_result(m_esc; view=:shock_densities))

            # save_path & title override
            tmp = tempname() * ".html"
            pt = plot_result(m_ssv_ml; title="Custom SSVARMA", ncols=2, save_path=tmp)
            @test isfile(tmp) && startswith(strip(read(tmp, String)), "<!DOCTYPE html>")
            @test occursin("Custom SSVARMA", pt.html)
            rm(tmp; force=true)
        end

        # ---------------------------------------------------------------------
        # 2. LMSSVARResult{T}
        # ---------------------------------------------------------------------
        @testset "2. LMSSVARResult" begin
            for vw in (:qq, :densities)
                p = plot_result(lms3; view=vw)
                check_plot(p); assert_all_json_valid(p)
                @test length(panel_titles(p.html)) == 3
            end
            p_d1 = plot_result(lms3; view=:densities, shock=1)
            p_q2 = plot_result(lms3; view=:qq, shock="s2")
            check_plot(p_d1); assert_all_json_valid(p_d1)
            check_plot(p_q2); assert_all_json_valid(p_q2)
            @test length(panel_titles(p_q2.html)) == 1
            @test "Empirical KDE" in series_names(p_d1.html)
            @test "N(0,1)" in series_names(p_d1.html)

            @test_throws ArgumentError plot_result(lms3; view=:invalid)
            @test_throws ArgumentError plot_result(lms3; shock=99)
            @test_throws ArgumentError plot_result(lms3; shock="missing")

            # NaN & escaping
            raw_lms = Any[getfield(lms3, f) for f in fieldnames(LMSSVARResult)]
            idx_sh = findfirst(==(:shocks), fieldnames(LMSSVARResult))
            idx_sn = findfirst(==(:shock_names), fieldnames(LMSSVARResult))
            raw_nan = copy(raw_lms)
            raw_nan[idx_sh] = copy(lms3.shocks); raw_nan[idx_sh][1, 1] = NaN
            lms_nan = LMSSVARResult{Float64}(raw_nan...)
            assert_nan_becomes_null(plot_result(lms_nan; view=:qq))

            raw_allnan = copy(raw_lms)
            raw_allnan[idx_sh] = fill(NaN, size(lms3.shocks))
            lms_allnan = LMSSVARResult{Float64}(raw_allnan...)
            assert_nan_becomes_null(plot_result(lms_allnan; view=:densities))

            raw_esc = copy(raw_lms)
            raw_esc[idx_sn] = [HOSTILE_NAME, "s2", "s3"]
            lms_esc = LMSSVARResult{Float64}(raw_esc...)
            assert_escapes(plot_result(lms_esc; view=:qq))
            assert_escapes(plot_result(lms_esc; view=:densities))
        end

        # ---------------------------------------------------------------------
        # 3. LanneLuotoGMMResult{T}
        # ---------------------------------------------------------------------
        @testset "3. LanneLuotoGMMResult" begin
            for vw in (:qq, :densities)
                p = plot_result(res_ll; view=vw)
                check_plot(p); assert_all_json_valid(p)
                @test length(panel_titles(p.html)) == 2
            end
            p_q = plot_result(res_ll; view=:qq, shock="Policy")
            p_d = plot_result(res_ll; view=:densities, shock=1)
            check_plot(p_q); assert_all_json_valid(p_q)
            check_plot(p_d); assert_all_json_valid(p_d)
            @test "Histogram" in series_names(p_d.html)

            @test_throws ArgumentError plot_result(res_ll; view=:bad)
            @test_throws ArgumentError plot_result(res_ll; shock=0)
            @test_throws ArgumentError plot_result(res_ll; shock="NoSuch")

            raw_ll = Any[getfield(res_ll, f) for f in fieldnames(LanneLuotoGMMResult)]
            idx_sh = findfirst(==(:shocks), fieldnames(LanneLuotoGMMResult))
            idx_sn = findfirst(==(:shock_names), fieldnames(LanneLuotoGMMResult))
            raw_nan = copy(raw_ll)
            raw_nan[idx_sh] = copy(res_ll.shocks); raw_nan[idx_sh][1, 1] = NaN
            ll_nan = LanneLuotoGMMResult{Float64}(raw_nan...)
            assert_nan_becomes_null(plot_result(ll_nan; view=:qq))

            raw_esc = copy(raw_ll)
            raw_esc[idx_sn] = [HOSTILE_NAME, "Policy"]
            ll_esc = LanneLuotoGMMResult{Float64}(raw_esc...)
            assert_escapes(plot_result(ll_esc; view=:qq))
            assert_escapes(plot_result(ll_esc; view=:densities))
        end

        # ---------------------------------------------------------------------
        # 4. NonGaussianDiagnostics{T}
        # ---------------------------------------------------------------------
        @testset "4. NonGaussianDiagnostics" begin
            p = plot_result(diag3)
            check_plot(p); assert_all_json_valid(p)
            @test length(panel_titles(p.html)) == 3
            for lvl in (1, 10)
                pl = plot_result(diag3; level=lvl)
                check_plot(pl); assert_all_json_valid(pl)
            end
            @test_throws ArgumentError plot_result(diag3; level=0)
            @test_throws ArgumentError plot_result(diag3; level=100)

            raw_d = Any[getfield(diag3, f) for f in fieldnames(NonGaussianDiagnostics)]
            idx_bds = findfirst(==(:bds_pvals), fieldnames(NonGaussianDiagnostics))
            idx_sn = findfirst(==(:shock_names), fieldnames(NonGaussianDiagnostics))
            idx_wk = findfirst(==(:weak_id_warning), fieldnames(NonGaussianDiagnostics))
            raw_nan = copy(raw_d)
            raw_nan[idx_bds] = copy(diag3.bds_pvals); raw_nan[idx_bds][1] = NaN
            raw_nan[idx_wk] = true
            d_nan = NonGaussianDiagnostics{Float64}(raw_nan...)
            pn = plot_result(d_nan)
            assert_nan_becomes_null(pn)
            @test occursin("WARNING", pn.html)

            raw_esc = copy(raw_d)
            raw_esc[idx_sn] = [HOSTILE_NAME, "s2", "s3"]
            assert_escapes(plot_result(NonGaussianDiagnostics{Float64}(raw_esc...)))
        end

        # ---------------------------------------------------------------------
        # 5. SVARHypothesisTest{T}
        # ---------------------------------------------------------------------
        @testset "5. SVARHypothesisTest" begin
            for ht in (ht_wald, ht_boot)
                p = plot_result(ht)
                check_plot(p); assert_all_json_valid(p)
                @test length(panel_titles(p.html)) == 3
            end
            pb = plot_result(ht_boot; level=1)
            check_plot(pb); assert_all_json_valid(pb)
            @test occursin("Bootstrap", pb.html)
            @test_throws ArgumentError plot_result(ht_wald; level=0)

            # df == 1 finite-grid guard + NaN + escaping
            raw_h = Any[getfield(ht_boot, f) for f in fieldnames(SVARHypothesisTest)]
            idx_stat = findfirst(==(:statistic), fieldnames(SVARHypothesisTest))
            idx_desc = findfirst(==(:description), fieldnames(SVARHypothesisTest))
            raw_nan = copy(raw_h); raw_nan[idx_stat] = NaN
            assert_nan_becomes_null(plot_result(SVARHypothesisTest{Float64}(raw_nan...)))

            raw_esc = copy(raw_h); raw_esc[idx_desc] = HOSTILE_NAME
            assert_escapes(plot_result(SVARHypothesisTest{Float64}(raw_esc...)))
        end

        # ---------------------------------------------------------------------
        # 6. GMMMomentSelectionResult{T}
        # ---------------------------------------------------------------------
        @testset "6. GMMMomentSelectionResult" begin
            p = plot_result(sel_ll)
            check_plot(p); assert_all_json_valid(p)
            @test length(panel_titles(p.html)) == 3
            @test_throws ArgumentError plot_result(sel_ll; max_candidates=0)

            # Cap note when candidates exceed max_candidates (C7)
            if length(sel_ll.candidates) > 1
                pcap = plot_result(sel_ll; max_candidates=1)
                check_plot(pcap); assert_all_json_valid(pcap)
                @test occursin("Showing 1 of $(length(sel_ll.candidates)) candidates", pcap.html)
            end

            # Empty candidates fallback + NaN candidate
            raw_ms = Any[getfield(sel_ll, f) for f in fieldnames(GMMMomentSelectionResult)]
            idx_c = findfirst(==(:candidates), fieldnames(GMMMomentSelectionResult))
            raw_empty = copy(raw_ms); raw_empty[idx_c] = Any[]
            p_empty = plot_result(GMMMomentSelectionResult{Float64}(raw_empty...))
            check_plot(p_empty); assert_all_json_valid(p_empty)

            if !isempty(sel_ll.candidates)
                c1 = sel_ll.candidates[1]
                c_nan = (asym_pairs=c1.asym_pairs, sym_pairs=c1.sym_pairs,
                         n_moments=c1.n_moments, df=c1.df, J_stat=NaN,
                         J_pvalue=NaN, msc=NaN, rmsc=NaN, rank_ok=c1.rank_ok)
                raw_nan = copy(raw_ms); raw_nan[idx_c] = [c_nan]
                assert_nan_becomes_null(plot_result(GMMMomentSelectionResult{Float64}(raw_nan...)))
            end
        end

        # ---------------------------------------------------------------------
        # 7. FundamentalnessTestResult{T}
        # ---------------------------------------------------------------------
        @testset "7. FundamentalnessTestResult" begin
            p = plot_result(res_fund)
            check_plot(p); assert_all_json_valid(p)
            @test length(panel_titles(p.html)) == 3
            @test occursin("d3.format('d')", p.html)
            @test_throws ArgumentError plot_result(res_fund; level=105)

            raw_f = Any[getfield(res_fund, f) for f in fieldnames(FundamentalnessTestResult)]
            idx_lp = findfirst(==(:lag_pvalues), fieldnames(FundamentalnessTestResult))
            raw_nan = copy(raw_f)
            raw_nan[idx_lp] = copy(res_fund.lag_pvalues); raw_nan[idx_lp][1] = NaN
            assert_nan_becomes_null(plot_result(FundamentalnessTestResult{Float64}(raw_nan...)))
        end

        # ---------------------------------------------------------------------
        # 8. SufficientInformationTestResult{T}
        # ---------------------------------------------------------------------
        @testset "8. SufficientInformationTestResult" begin
            p = plot_result(res_suff)
            check_plot(p); assert_all_json_valid(p)
            @test length(panel_titles(p.html)) == 3
            @test "F-test p-value" in series_names(p.html)
            @test "HAC Wald p-value" in series_names(p.html)
            @test_throws ArgumentError plot_result(res_suff; level=0)

            res_suff_1 = test_sufficient_information(var2_for_fund, X_panel; r_max=1, p_f=2)
            p1 = plot_result(res_suff_1; level=10)
            check_plot(p1); assert_all_json_valid(p1)

            raw_s = Any[getfield(res_suff, f) for f in fieldnames(SufficientInformationTestResult)]
            idx_fp = findfirst(==(:f_pvalues), fieldnames(SufficientInformationTestResult))
            raw_nan = copy(raw_s)
            raw_nan[idx_fp] = copy(res_suff.f_pvalues); raw_nan[idx_fp][1] = NaN
            assert_nan_becomes_null(plot_result(SufficientInformationTestResult{Float64}(raw_nan...)))
        end

        # ---------------------------------------------------------------------
        # 9. Single positional argument + ambiguity hygiene
        # ---------------------------------------------------------------------
        @testset "9. Method Signature & Ambiguity Hygiene" begin
            @test isempty(Test.detect_ambiguities(_MN))
            for Ttype in (SVARMAModel, LMSSVARResult, LanneLuotoGMMResult,
                          NonGaussianDiagnostics, SVARHypothesisTest,
                          GMMMomentSelectionResult, FundamentalnessTestResult,
                          SufficientInformationTestResult)
                ms = [m for m in methods(plot_result, Tuple{Ttype})]
                @test length(ms) == 1
                @test ms[1].nargs == 2  # (plot_result, x) -> 1 positional argument
            end
        end
    end

end
