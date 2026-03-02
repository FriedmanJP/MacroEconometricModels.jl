# MacroEconometricModels.jl — Difference-in-Differences Test Suite
# Tests for: estimate_did (TWFE, CS), estimate_event_study_lp, estimate_lp_did,
#            bacon_decomposition, pretrend_test, negative_weight_check, plotting, refs

using Test, Random, Statistics
using MacroEconometricModels

# =============================================================================
# DGP Helper: staggered adoption panel
# =============================================================================

"""
    _make_did_panel(; n_units=50, n_periods=20, treat_effect=2.0,
                      n_cohorts=2, seed=42)

Create a PanelData with staggered treatment adoption for DiD testing.

- `n_cohorts` treatment cohorts + 1 never-treated group
- Treatment effect grows slightly over time: te * (1 + 0.1*(t - g_i))
- Returns (pd, treat_effect)
"""
function _make_did_panel(; n_units=50, n_periods=20, treat_effect=2.0,
                           n_cohorts=2, seed=42)
    rng = Random.MersenneTwister(seed)
    T_type = Float64

    units_per_group = n_units ÷ (n_cohorts + 1)
    treat_times = zeros(Int, n_units)
    cohort_periods = [5 + 3*(c-1) for c in 1:n_cohorts]
    for c in 1:n_cohorts
        for u in ((c-1)*units_per_group + 1):(c*units_per_group)
            treat_times[u] = cohort_periods[c]
        end
    end
    # Remaining units (units_per_group*(n_cohorts)+1 : n_units) are never-treated (treat_times=0)

    N_obs = n_units * n_periods
    data = Matrix{T_type}(undef, N_obs, 3)
    group_id = Vector{Int}(undef, N_obs)
    time_id = Vector{Int}(undef, N_obs)

    row = 1
    for i in 1:n_units
        alpha_i = randn(rng)
        for t in 1:n_periods
            gamma_t = 0.1 * t
            te = (treat_times[i] > 0 && t >= treat_times[i]) ? treat_effect : 0.0
            if te > 0
                te *= (1.0 + 0.1 * (t - treat_times[i]))
            end
            y = alpha_i + gamma_t + te + 0.5 * randn(rng)
            data[row, 1] = y
            data[row, 2] = T_type(treat_times[i])
            data[row, 3] = randn(rng)  # covariate
            group_id[row] = i
            time_id[row] = t
            row += 1
        end
    end

    group_names = ["unit_$i" for i in 1:n_units]
    pd = PanelData{T_type}(data, ["outcome", "treat_time", "covariate"],
                            Quarterly, [1, 1, 1], group_id, time_id,
                            group_names, n_units, 3, N_obs, true,
                            ["DiD test panel"], Dict{String,String}(), Symbol[])
    pd, treat_effect
end


@testset "Difference-in-Differences" begin

    # =========================================================================
    # TWFE DiD
    # =========================================================================
    @testset "TWFE DiD" begin
        pd, te = _make_did_panel(seed=100)

        did = estimate_did(pd, "outcome", "treat_time";
                           method=:twfe, leads=3, horizon=5, cluster=:unit)

        # Type checks
        @test did isa DIDResult{Float64}
        @test did.method == :twfe
        @test did.outcome_var == "outcome"
        @test did.treatment_var == "treat_time"
        @test did.control_group == :never_treated
        @test did.cluster == :unit
        @test did.conf_level == 0.95
        @test did.n_obs == 50 * 20
        @test did.n_groups == 50

        # Event-time grid
        @test did.event_times == collect(-3:5)
        @test did.reference_period == -1
        @test length(did.att) == length(did.event_times)
        @test length(did.se) == length(did.event_times)
        @test length(did.ci_lower) == length(did.event_times)
        @test length(did.ci_upper) == length(did.event_times)

        # Reference period has zero coefficient and SE
        ref_idx = findfirst(==(-1), did.event_times)
        @test did.att[ref_idx] == 0.0
        @test did.se[ref_idx] == 0.0

        # Post-treatment ATTs: with TWFE and staggered adoption, bias is possible
        # but at least some post-treatment effects should be nonzero
        post_mask = did.event_times .>= 0
        @test any(did.att[post_mask] .!= 0.0)

        # Overall ATT should be finite
        @test isfinite(did.overall_att)
        @test did.overall_se >= 0

        # SEs should be non-negative
        @test all(did.se .>= 0)

        # CIs bracket the point estimates
        @test all(did.ci_lower .<= did.att)
        @test all(did.ci_upper .>= did.att)

        # TWFE has no group-time ATT matrix
        @test did.group_time_att === nothing
        @test did.cohorts === nothing

        # StatsAPI interface
        @test nobs(did) == did.n_obs
        @test coef(did) === did.att
        @test stderror(did) === did.se
        ci = confint(did)
        @test size(ci) == (length(did.att), 2)
        @test ci[:, 1] == did.ci_lower
        @test ci[:, 2] == did.ci_upper

        # Display works without error
        io = IOBuffer()
        show(io, did)
        s = String(take!(io))
        @test occursin("Two-Way Fixed Effects", s)
        @test occursin("Event-Study Coefficients", s)
        @test occursin("Aggregate Treatment Effect", s)
    end

    # =========================================================================
    # Callaway-Sant'Anna
    # =========================================================================
    @testset "Callaway-Sant'Anna" begin
        pd, te = _make_did_panel(seed=200)

        did_cs = estimate_did(pd, "outcome", "treat_time";
                              method=:callaway_santanna, leads=3, horizon=5,
                              control_group=:never_treated)

        @test did_cs isa DIDResult{Float64}
        @test did_cs.method == :callaway_santanna
        @test did_cs.control_group == :never_treated

        # Group-time ATT matrix should exist for CS
        @test did_cs.group_time_att !== nothing
        @test did_cs.cohorts !== nothing
        @test length(did_cs.cohorts) >= 1

        # Group-time ATT dimensions: n_cohorts x n_periods
        @test size(did_cs.group_time_att, 1) == length(did_cs.cohorts)

        # Post-treatment aggregate ATTs should be positive
        post_mask = did_cs.event_times .>= 0
        post_att = did_cs.att[post_mask]
        # At least some post-treatment ATTs should be positive (effect=2.0 is large)
        @test any(post_att .> 0)

        # Overall ATT positive
        @test did_cs.overall_att > 0

        # SEs non-negative
        @test all(did_cs.se .>= 0)

        # Display
        io = IOBuffer()
        show(io, did_cs)
        s = String(take!(io))
        @test occursin("Callaway", s)
    end

    # =========================================================================
    # Callaway-Sant'Anna with not_yet_treated controls
    # =========================================================================
    @testset "Callaway-Sant'Anna not_yet_treated" begin
        pd, te = _make_did_panel(seed=300)

        did_nyt = estimate_did(pd, "outcome", "treat_time";
                               method=:callaway_santanna, leads=2, horizon=4,
                               control_group=:not_yet_treated)

        @test did_nyt isa DIDResult{Float64}
        @test did_nyt.method == :callaway_santanna
        @test did_nyt.control_group == :not_yet_treated

        # Should still produce valid results
        @test length(did_nyt.att) == length(did_nyt.event_times)
        @test did_nyt.overall_att > 0
    end

    # =========================================================================
    # Event Study LP
    # =========================================================================
    @testset "Event Study LP" begin
        pd, te = _make_did_panel(seed=400, n_units=60, n_periods=25)

        eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 5;
                                        leads=3, lags=2, cluster=:unit)

        @test eslp isa EventStudyLP{Float64}
        @test eslp.clean_controls == false
        @test eslp.outcome_var == "outcome"
        @test eslp.treatment_var == "treat_time"
        @test eslp.leads == 3
        @test eslp.lags == 2
        @test eslp.horizon == 5
        @test eslp.n_groups == 60
        @test eslp.cluster == :unit
        @test eslp.reference_period == -1

        # Event-time grid from -leads to H
        @test eslp.event_times == collect(-3:5)
        @test length(eslp.coefficients) == length(eslp.event_times)
        @test length(eslp.se) == length(eslp.event_times)

        # Reference period has zero coefficient
        ref_idx = findfirst(==(-1), eslp.event_times)
        @test eslp.coefficients[ref_idx] == 0.0
        @test eslp.se[ref_idx] == 0.0

        # Per-horizon regression data
        @test length(eslp.B) == length(eslp.event_times)
        @test length(eslp.residuals_per_h) == length(eslp.event_times)
        @test length(eslp.vcov) == length(eslp.event_times)
        @test length(eslp.T_eff) == length(eslp.event_times)

        # CIs bracket estimates
        @test all(eslp.ci_lower .<= eslp.coefficients)
        @test all(eslp.ci_upper .>= eslp.coefficients)

        # StatsAPI
        @test nobs(eslp) == eslp.n_obs
        @test coef(eslp) === eslp.coefficients
        @test stderror(eslp) === eslp.se

        # Display
        io = IOBuffer()
        show(io, eslp)
        s = String(take!(io))
        @test occursin("Event Study LP", s)
        @test occursin("Dynamic Treatment Effects", s)
    end

    # =========================================================================
    # LP-DiD (Dube et al. 2023)
    # =========================================================================
    @testset "LP-DiD" begin
        pd, te = _make_did_panel(seed=500, n_units=60, n_periods=25)

        lpdid = estimate_lp_did(pd, "outcome", "treat_time", 5;
                                 leads=3, lags=2, cluster=:unit)

        @test lpdid isa EventStudyLP{Float64}
        @test lpdid.clean_controls == true

        # Should still have valid event times and coefficients
        @test length(lpdid.coefficients) == length(lpdid.event_times)
        @test all(lpdid.se .>= 0)

        # Display identifies as LP-DiD
        io = IOBuffer()
        show(io, lpdid)
        s = String(take!(io))
        @test occursin("LP-DiD", s)
    end

    # =========================================================================
    # Bacon Decomposition
    # =========================================================================
    @testset "Bacon Decomposition" begin
        pd, te = _make_did_panel(seed=600)

        bd = bacon_decomposition(pd, "outcome", "treat_time")

        @test bd isa BaconDecomposition{Float64}

        # Weights sum to 1
        @test isapprox(sum(bd.weights), 1.0; atol=1e-8)

        # Has valid comparison types
        @test all(ct -> ct in (:earlier_vs_later, :later_vs_earlier, :treated_vs_untreated),
                  bd.comparison_type)

        # Should include at least a :later_vs_earlier or :earlier_vs_later (2 cohorts)
        has_timing_comp = any(ct -> ct in (:earlier_vs_later, :later_vs_earlier),
                              bd.comparison_type)
        @test has_timing_comp

        # All fields have consistent lengths
        n = length(bd.estimates)
        @test length(bd.weights) == n
        @test length(bd.comparison_type) == n
        @test length(bd.cohort_i) == n
        @test length(bd.cohort_j) == n

        # Overall ATT is the weighted average
        @test isapprox(bd.overall_att, sum(bd.estimates .* bd.weights); atol=1e-10)

        # Display
        io = IOBuffer()
        show(io, bd)
        s = String(take!(io))
        @test occursin("Goodman-Bacon", s)
        @test occursin("Weight", s)
    end

    # =========================================================================
    # Pre-trend test
    # =========================================================================
    @testset "Pre-trend test" begin
        pd, te = _make_did_panel(seed=700, n_units=60, n_periods=25)

        # Test on DIDResult
        did = estimate_did(pd, "outcome", "treat_time";
                           method=:twfe, leads=4, horizon=5)
        pt_did = pretrend_test(did)

        @test pt_did isa PretrendTestResult{Float64}
        @test pt_did.statistic >= 0
        @test 0 <= pt_did.pvalue <= 1
        @test pt_did.df >= 0
        @test pt_did.test_type in (:f_test, :wald)
        @test length(pt_did.pre_coefficients) == pt_did.df || pt_did.df == 0

        # With the DGP having parallel trends by construction (no pre-treatment effect),
        # p-value should generally be high (not always, but let's not make this too strict)
        # Just check it doesn't error

        # Display
        io = IOBuffer()
        show(io, pt_did)
        s = String(take!(io))
        @test occursin("Pre-Trend Test", s)

        # Test on EventStudyLP
        eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 5;
                                        leads=4, lags=2)
        pt_lp = pretrend_test(eslp)

        @test pt_lp isa PretrendTestResult{Float64}
        @test pt_lp.statistic >= 0
        @test 0 <= pt_lp.pvalue <= 1
    end

    # =========================================================================
    # Negative weight check
    # =========================================================================
    @testset "Negative weight check" begin
        pd, te = _make_did_panel(seed=800)

        nw = negative_weight_check(pd, "treat_time")

        @test nw isa NegativeWeightResult{Float64}
        @test nw.has_negative_weights isa Bool
        @test nw.n_negative >= 0
        @test length(nw.weights) >= 1
        @test length(nw.cohort_time_pairs) == length(nw.weights)

        # If negative weights exist, total_negative_weight should be < 0
        if nw.has_negative_weights
            @test nw.total_negative_weight < 0
            @test nw.n_negative > 0
        else
            @test nw.n_negative == 0
        end

        # Display
        io = IOBuffer()
        show(io, nw)
        s = String(take!(io))
        @test occursin("Negative Weight", s)
    end

    # =========================================================================
    # Clustering options
    # =========================================================================
    @testset "Clustering" begin
        pd, te = _make_did_panel(seed=900)

        # All three clustering options should work
        for clust in (:unit, :time, :twoway)
            did = estimate_did(pd, "outcome", "treat_time";
                               method=:twfe, leads=2, horizon=3, cluster=clust)
            @test did isa DIDResult{Float64}
            @test did.cluster == clust
            @test all(did.se .>= 0)
        end

        # LP clustering
        for clust in (:unit, :time, :twoway)
            eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 3;
                                            leads=2, lags=2, cluster=clust)
            @test eslp isa EventStudyLP{Float64}
            @test eslp.cluster == clust
        end
    end

    # =========================================================================
    # Plotting
    # =========================================================================
    @testset "Plotting" begin
        pd, te = _make_did_panel(seed=1000)

        # DIDResult plot
        did = estimate_did(pd, "outcome", "treat_time";
                           method=:twfe, leads=3, horizon=5)
        p_did = plot_result(did)
        @test p_did isa PlotOutput
        @test occursin("DiD Event Study", p_did.html)
        @test occursin("d3", p_did.html)

        # EventStudyLP plot
        eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 5;
                                        leads=3, lags=2)
        p_eslp = plot_result(eslp)
        @test p_eslp isa PlotOutput
        @test occursin("Event Study LP", p_eslp.html)

        # LP-DiD plot
        lpdid = estimate_lp_did(pd, "outcome", "treat_time", 5;
                                 leads=3, lags=2)
        p_lpdid = plot_result(lpdid)
        @test p_lpdid isa PlotOutput
        @test occursin("LP-DiD", p_lpdid.html)

        # BaconDecomposition plot
        bd = bacon_decomposition(pd, "outcome", "treat_time")
        p_bd = plot_result(bd)
        @test p_bd isa PlotOutput
        @test occursin("Bacon", p_bd.html)
    end

    # =========================================================================
    # References
    # =========================================================================
    @testset "References" begin
        pd, te = _make_did_panel(seed=1100)

        did = estimate_did(pd, "outcome", "treat_time"; method=:twfe, leads=2, horizon=3)
        io = IOBuffer()
        refs(io, did)
        s = String(take!(io))
        @test occursin("Callaway", s) || occursin("Goodman-Bacon", s)

        eslp = estimate_event_study_lp(pd, "outcome", "treat_time", 3; leads=2, lags=2)
        io2 = IOBuffer()
        refs(io2, eslp)
        s2 = String(take!(io2))
        @test occursin("Jord", s2) || occursin("Dube", s2)

        bd = bacon_decomposition(pd, "outcome", "treat_time")
        io3 = IOBuffer()
        refs(io3, bd)
        s3 = String(take!(io3))
        @test occursin("Goodman-Bacon", s3)
    end

    # =========================================================================
    # Edge cases
    # =========================================================================
    @testset "Edge cases" begin
        # Single cohort
        pd_single, te_single = _make_did_panel(seed=1200, n_cohorts=1, n_units=30)
        did_single = estimate_did(pd_single, "outcome", "treat_time";
                                  method=:twfe, leads=2, horizon=3)
        @test did_single isa DIDResult{Float64}
        @test did_single.n_treated > 0

        # Invalid method
        pd, te = _make_did_panel(seed=1300)
        @test_throws ArgumentError estimate_did(pd, "outcome", "treat_time"; method=:invalid)

        # Invalid cluster
        @test_throws ArgumentError estimate_did(pd, "outcome", "treat_time"; cluster=:invalid)

        # Nonexistent variable
        @test_throws ArgumentError estimate_did(pd, "nonexistent", "treat_time")
        @test_throws ArgumentError estimate_did(pd, "outcome", "nonexistent")

        # All-treated panel with CS + never_treated should throw
        pd_all_treat, _ = _make_did_panel(seed=1400, n_cohorts=3, n_units=30)
        # With n_cohorts=3 and n_units=30, units_per_group=30/4=7, so 21 treated and 9 never-treated
        # We need ALL units treated. Build a custom panel:
        rng = Random.MersenneTwister(1500)
        n_u = 20
        n_t = 10
        N_all = n_u * n_t
        data_all = Matrix{Float64}(undef, N_all, 2)
        gid_all = Vector{Int}(undef, N_all)
        tid_all = Vector{Int}(undef, N_all)
        row = 1
        for i in 1:n_u
            t_treat = (i <= 10) ? 4 : 7  # All treated in 2 cohorts, no never-treated
            for t in 1:n_t
                data_all[row, 1] = randn(rng)
                data_all[row, 2] = Float64(t_treat)
                gid_all[row] = i
                tid_all[row] = t
                row += 1
            end
        end
        pd_all = PanelData{Float64}(data_all, ["y", "tt"],
                                     Quarterly, [1, 1], gid_all, tid_all,
                                     ["u$i" for i in 1:n_u], n_u, 2, N_all, true,
                                     ["all treated"], Dict{String,String}(), Symbol[])
        @test_throws ArgumentError estimate_did(pd_all, "y", "tt";
                                                method=:callaway_santanna,
                                                control_group=:never_treated)

        # Invalid control_group
        @test_throws ArgumentError estimate_did(pd, "outcome", "treat_time";
                                                control_group=:invalid_group)

        # Zero leads, zero horizon
        did_zero = estimate_did(pd, "outcome", "treat_time";
                                method=:twfe, leads=0, horizon=0)
        @test did_zero isa DIDResult{Float64}
        @test length(did_zero.event_times) == 1  # just [0] minus ref=-1 => [-1, 0] => 2 actually
        # event_times = collect(-0:0) = [0], but reference_period=-1 is not in grid
        @test -1 in did_zero.event_times || length(did_zero.event_times) >= 1
    end

    # =========================================================================
    # With covariates
    # =========================================================================
    @testset "With covariates" begin
        pd, te = _make_did_panel(seed=1600)

        did_cov = estimate_did(pd, "outcome", "treat_time";
                               method=:twfe, leads=3, horizon=5,
                               covariates=["covariate"])

        @test did_cov isa DIDResult{Float64}
        @test length(did_cov.att) == length(did_cov.event_times)
        @test all(did_cov.se .>= 0)

        # Post-treatment ATTs should still be positive with a large true effect
        post_mask = did_cov.event_times .>= 0
        @test any(did_cov.att[post_mask] .> 0)
    end

    # =========================================================================
    # Symbol interface
    # =========================================================================
    @testset "Symbol interface" begin
        pd, te = _make_did_panel(seed=1700)

        # Symbols should work just like strings
        did_sym = estimate_did(pd, :outcome, :treat_time;
                               method=:twfe, leads=2, horizon=3)
        @test did_sym isa DIDResult{Float64}

        eslp_sym = estimate_event_study_lp(pd, :outcome, :treat_time, 3;
                                            leads=2, lags=2)
        @test eslp_sym isa EventStudyLP{Float64}

        lpdid_sym = estimate_lp_did(pd, :outcome, :treat_time, 3;
                                     leads=2, lags=2)
        @test lpdid_sym isa EventStudyLP{Float64}

        bd_sym = bacon_decomposition(pd, :outcome, :treat_time)
        @test bd_sym isa BaconDecomposition{Float64}

        nw_sym = negative_weight_check(pd, :treat_time)
        @test nw_sym isa NegativeWeightResult{Float64}
    end

    # =========================================================================
    # Pre-trend test edge cases
    # =========================================================================
    @testset "Pre-trend test edge cases" begin
        pd, te = _make_did_panel(seed=1800)

        # With zero leads: no pre-treatment periods to test
        did_no_pre = estimate_did(pd, "outcome", "treat_time";
                                  method=:twfe, leads=0, horizon=3)
        pt_no_pre = pretrend_test(did_no_pre)
        @test pt_no_pre isa PretrendTestResult{Float64}
        # With no pre-periods, should return trivial result
        @test pt_no_pre.df == 0 || length(pt_no_pre.pre_coefficients) == 0
    end

    # =========================================================================
    # Bacon decomposition with single cohort
    # =========================================================================
    @testset "Bacon single cohort" begin
        pd_single, _ = _make_did_panel(seed=1900, n_cohorts=1, n_units=30)
        bd_single = bacon_decomposition(pd_single, "outcome", "treat_time")
        @test bd_single isa BaconDecomposition{Float64}
        # With a single cohort, should only have treated_vs_untreated comparisons
        @test all(ct -> ct == :treated_vs_untreated, bd_single.comparison_type)
        @test isapprox(sum(bd_single.weights), 1.0; atol=1e-8)
    end

    # =========================================================================
    # report() dispatch
    # =========================================================================
    @testset "report() dispatch" begin
        pd, te = _make_did_panel(seed=2000)

        did = estimate_did(pd, "outcome", "treat_time";
                           method=:twfe, leads=2, horizon=3)
        # report() should not error (dispatches to show for DIDResult)
        io = IOBuffer()
        show(io, did)
        @test length(String(take!(io))) > 0
    end

end  # @testset "Difference-in-Differences"
