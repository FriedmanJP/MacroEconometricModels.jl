# MacroEconometricModels.jl — LP-DiD Test Suite
# Tests for: estimate_lp_did (absorbing, non-absorbing, PMD, reweighting, pooled),
#            panel_lag/lead/diff, DDCG dataset, plotting, refs

using Test, Random, Statistics, DataFrames
using MacroEconometricModels

# =============================================================================
# DGP Helper: simple staggered panel for LP-DiD
# =============================================================================

"""
Create a clean staggered-adoption panel for LP-DiD tests.
- `n_treated` units adopt at `treat_period`, rest are never-treated.
- Treatment effect = `te` (constant post-treatment).
"""
function _make_lpdid_panel(; n_units=60, n_periods=30, treat_effect=2.0,
                              treat_period=15, n_treated=20, seed=42)
    rng = Random.MersenneTwister(seed)
    N_obs = n_units * n_periods
    data = Matrix{Float64}(undef, N_obs, 2)  # y, treat
    group_id = Vector{Int}(undef, N_obs)
    time_id = Vector{Int}(undef, N_obs)

    row = 1
    for i in 1:n_units
        alpha_i = randn(rng) * 0.5
        for t in 1:n_periods
            gamma_t = 0.05 * t
            treated = (i <= n_treated && t >= treat_period)
            te = treated ? treat_effect : 0.0
            y = alpha_i + gamma_t + te + randn(rng) * 0.5
            treat_val = treated ? 1.0 : 0.0
            data[row, 1] = y
            data[row, 2] = treat_val
            group_id[row] = i
            time_id[row] = t
            row += 1
        end
    end

    pd = PanelData{Float64}(data, ["y", "treat"], Other, [1, 1],
                            group_id, time_id, nothing,
                            [string(i) for i in 1:n_units],
                            n_units, 2, N_obs, true,
                            ["Test panel"], Dict{String,String}(), Symbol[])
    return pd, treat_effect
end

"""
Create a staggered panel with multiple cohorts (for more complex tests).
"""
function _make_lpdid_staggered(; n_units=60, n_periods=30, seed=42)
    rng = Random.MersenneTwister(seed)
    N_obs = n_units * n_periods
    data = Matrix{Float64}(undef, N_obs, 2)  # y, treat
    group_id = Vector{Int}(undef, N_obs)
    time_id = Vector{Int}(undef, N_obs)

    # 3 cohorts: units 1-15 treated at t=10, units 16-30 treated at t=18, rest never-treated
    treat_times = zeros(Int, n_units)
    for i in 1:15
        treat_times[i] = 10
    end
    for i in 16:30
        treat_times[i] = 18
    end

    row = 1
    for i in 1:n_units
        alpha_i = randn(rng) * 0.5
        for t in 1:n_periods
            gamma_t = 0.05 * t
            g_i = treat_times[i]
            treated = (g_i > 0 && t >= g_i)
            te = treated ? 1.5 * (1 + 0.1 * (t - g_i)) : 0.0
            y = alpha_i + gamma_t + te + randn(rng) * 0.5
            treat_val = treated ? 1.0 : 0.0
            data[row, 1] = y
            data[row, 2] = treat_val
            group_id[row] = i
            time_id[row] = t
            row += 1
        end
    end

    pd = PanelData{Float64}(data, ["y", "treat"], Other, [1, 1],
                            group_id, time_id, nothing,
                            [string(i) for i in 1:n_units],
                            n_units, 2, N_obs, true,
                            ["Test staggered panel"], Dict{String,String}(), Symbol[])
    return pd
end


@testset "LP-DiD Engine" begin

    # =========================================================================
    # Basic absorbing treatment
    # =========================================================================
    @testset "Absorbing treatment" begin
        pd, te = _make_lpdid_panel(seed=100)

        r = estimate_lp_did(pd, :y, :treat, 5; pre_window=3, ylags=2)
        @test r isa LPDiDResult{Float64}
        @test r.specification == :absorbing
        @test r.outcome_var == "y"
        @test r.treatment_var == "treat"
        @test length(r.coefficients) == 9  # -3 to 5
        @test r.event_times == collect(-3:5)
        @test r.reference_period == -1
        @test r.ylags == 2
        @test r.dylags == 0
        @test r.pre_window == 3
        @test r.post_window == 5
        @test r.cluster == :unit
        @test r.reweight == false
        @test r.nocomp == false
        @test r.pmd === nothing

        # Reference period has zero coefficient
        ref_idx = findfirst(==(-1), r.event_times)
        @test r.coefficients[ref_idx] == 0.0
        @test r.se[ref_idx] == 0.0

        # Post-treatment coefficients should be positive (treatment effect ~2.0)
        for h in 0:5
            hi = findfirst(==(h), r.event_times)
            hi === nothing && continue
            @test r.coefficients[hi] > 0 || isnan(r.coefficients[hi])
        end

        # CIs bracket estimates
        valid = .!isnan.(r.coefficients)
        @test all(r.ci_lower[valid] .<= r.coefficients[valid])
        @test all(r.ci_upper[valid] .>= r.coefficients[valid])

        # Per-horizon nobs (excluding reference period which has nobs=0)
        @test length(r.nobs_per_horizon) == length(r.event_times)
        non_ref = r.event_times .!= r.reference_period
        active = valid .& non_ref
        @test all(r.nobs_per_horizon[active] .> 0)

        # StatsAPI
        @test nobs(r) == pd.T_obs
        @test coef(r) === r.coefficients
        @test stderror(r) === r.se
        ci = confint(r)
        @test size(ci) == (length(r.coefficients), 2)
    end

    # =========================================================================
    # String vs Symbol interface
    # =========================================================================
    @testset "String vs Symbol interface" begin
        pd, _ = _make_lpdid_panel(seed=101)

        r_str = estimate_lp_did(pd, "y", "treat", 3; pre_window=2)
        r_sym = estimate_lp_did(pd, :y, :treat, 3; pre_window=2)

        @test r_str isa LPDiDResult{Float64}
        @test r_sym isa LPDiDResult{Float64}
        @test r_str.coefficients == r_sym.coefficients
    end

    # =========================================================================
    # Outcome lags and diff lags
    # =========================================================================
    @testset "Outcome lags and diff lags" begin
        pd, _ = _make_lpdid_panel(seed=102)

        # ylags only
        r1 = estimate_lp_did(pd, :y, :treat, 3; ylags=3)
        @test r1 isa LPDiDResult{Float64}
        @test r1.ylags == 3
        @test r1.dylags == 0

        # dylags only
        r2 = estimate_lp_did(pd, :y, :treat, 3; dylags=2)
        @test r2 isa LPDiDResult{Float64}
        @test r2.ylags == 0
        @test r2.dylags == 2

        # Both
        r3 = estimate_lp_did(pd, :y, :treat, 3; ylags=1, dylags=1)
        @test r3 isa LPDiDResult{Float64}
        @test r3.ylags == 1
        @test r3.dylags == 1
    end

    # =========================================================================
    # Clustering options
    # =========================================================================
    @testset "Clustering options" begin
        pd, _ = _make_lpdid_panel(seed=103)

        for clust in (:unit, :time, :twoway)
            r = estimate_lp_did(pd, :y, :treat, 3; cluster=clust)
            @test r isa LPDiDResult{Float64}
            @test r.cluster == clust
            @test all(r.se .>= 0)
        end
    end

    # =========================================================================
    # Non-absorbing treatment
    # =========================================================================
    @testset "Non-absorbing treatment" begin
        pd, _ = _make_lpdid_panel(seed=104)

        r = estimate_lp_did(pd, :y, :treat, 3; nonabsorbing=5)
        @test r isa LPDiDResult{Float64}
        @test r.specification == :nonabsorbing
    end

    # =========================================================================
    # One-off treatment
    # =========================================================================
    @testset "One-off treatment" begin
        pd, _ = _make_lpdid_panel(seed=105)

        r = estimate_lp_did(pd, :y, :treat, 3; nonabsorbing=3, oneoff=true)
        @test r isa LPDiDResult{Float64}
        @test r.specification == :oneoff
    end

    # =========================================================================
    # Control group restrictions
    # =========================================================================
    @testset "Control group restrictions" begin
        pd = _make_lpdid_staggered(seed=106)

        # Not-yet-treated
        r_nyt = estimate_lp_did(pd, :y, :treat, 3; notyet=true)
        @test r_nyt isa LPDiDResult{Float64}

        # Never-treated
        r_nt = estimate_lp_did(pd, :y, :treat, 3; nevertreated=true)
        @test r_nt isa LPDiDResult{Float64}

        # First-treat
        r_ft = estimate_lp_did(pd, :y, :treat, 3; firsttreat=true)
        @test r_ft isa LPDiDResult{Float64}
    end

    # =========================================================================
    # PMD (pre-mean differencing)
    # =========================================================================
    @testset "PMD (pre-mean differencing)" begin
        pd, _ = _make_lpdid_panel(seed=107)

        # pmd=:max
        r_max = estimate_lp_did(pd, :y, :treat, 3; pmd=:max)
        @test r_max isa LPDiDResult{Float64}
        @test r_max.pmd == :max

        # pmd=integer
        r_k = estimate_lp_did(pd, :y, :treat, 3; pmd=3)
        @test r_k isa LPDiDResult{Float64}
        @test r_k.pmd == 3
    end

    # =========================================================================
    # Nocomp restriction
    # =========================================================================
    @testset "Nocomp restriction" begin
        pd, _ = _make_lpdid_panel(seed=108)

        r = estimate_lp_did(pd, :y, :treat, 3; nocomp=true)
        @test r isa LPDiDResult{Float64}
        @test r.nocomp == true
    end

    # =========================================================================
    # Reweighting
    # =========================================================================
    @testset "Reweighting" begin
        pd, _ = _make_lpdid_panel(seed=109)

        r = estimate_lp_did(pd, :y, :treat, 3; reweight=true)
        @test r isa LPDiDResult{Float64}
        @test r.reweight == true
    end

    # =========================================================================
    # Pooled regressions
    # =========================================================================
    @testset "Pooled regressions" begin
        pd, te = _make_lpdid_panel(seed=110)

        # Post pooled
        r = estimate_lp_did(pd, :y, :treat, 5; post_pooled=(0, 3))
        @test r isa LPDiDResult{Float64}
        @test r.pooled_post !== nothing
        @test haskey(r.pooled_post, :coef)
        @test haskey(r.pooled_post, :se)
        @test haskey(r.pooled_post, :nobs)
        @test r.pooled_post.coef > 0  # positive treatment effect

        # Pre pooled
        r2 = estimate_lp_did(pd, :y, :treat, 5; pre_pooled=(1, 3))
        @test r2 isa LPDiDResult{Float64}
        @test r2.pooled_pre !== nothing

        # Both
        r3 = estimate_lp_did(pd, :y, :treat, 5; post_pooled=(0, 3), pre_pooled=(1, 3))
        @test r3.pooled_post !== nothing
        @test r3.pooled_pre !== nothing

        # only_event skips pooled
        r4 = estimate_lp_did(pd, :y, :treat, 5; only_event=true, post_pooled=(0, 3))
        @test r4.pooled_post === nothing

        # only_pooled skips event
        r5 = estimate_lp_did(pd, :y, :treat, 5; only_pooled=true, post_pooled=(0, 3))
        @test r5 isa LPDiDResult{Float64}
        # All event-study coefficients should be zero (skipped)
        @test all(r5.coefficients .== 0.0)
    end

    # =========================================================================
    # Input validation
    # =========================================================================
    @testset "Input validation" begin
        pd, _ = _make_lpdid_panel(seed=111)

        # Cannot combine notyet and nevertreated
        @test_throws ArgumentError estimate_lp_did(pd, :y, :treat, 3;
                                                    notyet=true, nevertreated=true)

        # oneoff requires nonabsorbing
        @test_throws ArgumentError estimate_lp_did(pd, :y, :treat, 3; oneoff=true)

        # Cannot combine only_pooled and only_event
        @test_throws ArgumentError estimate_lp_did(pd, :y, :treat, 3;
                                                    only_pooled=true, only_event=true)
    end

    # =========================================================================
    # Timing column (treat_time as year values)
    # =========================================================================
    @testset "Timing column treatment" begin
        rng = Random.MersenneTwister(112)
        n_units, n_periods = 40, 25
        N_obs = n_units * n_periods
        data = Matrix{Float64}(undef, N_obs, 2)
        group_id = Vector{Int}(undef, N_obs)
        time_id = Vector{Int}(undef, N_obs)

        # Units 1-15 treated at period 12, rest never-treated (0)
        row = 1
        for i in 1:n_units
            alpha_i = randn(rng) * 0.5
            for t in 1:n_periods
                treat_time = (i <= 15) ? 12.0 : 0.0
                treated = (i <= 15 && t >= 12)
                y = alpha_i + 0.05*t + (treated ? 1.5 : 0.0) + randn(rng) * 0.5
                data[row, 1] = y
                data[row, 2] = treat_time
                group_id[row] = i
                time_id[row] = t
                row += 1
            end
        end

        pd = PanelData{Float64}(data, ["y", "treat_time"], Other, [1, 1],
                                group_id, time_id, nothing,
                                [string(i) for i in 1:n_units],
                                n_units, 2, N_obs, true,
                                ["Timing panel"], Dict{String,String}(), Symbol[])

        r = estimate_lp_did(pd, :y, :treat_time, 5; pre_window=3)
        @test r isa LPDiDResult{Float64}
        @test length(r.coefficients) > 0
    end

    # =========================================================================
    # Display
    # =========================================================================
    @testset "Display" begin
        pd, _ = _make_lpdid_panel(seed=113)

        r = estimate_lp_did(pd, :y, :treat, 3; pre_window=2, ylags=1)

        io = IOBuffer()
        show(io, r)
        s = String(take!(io))
        @test occursin("LP-DiD", s)
        @test occursin("Absorbing", s)
        @test occursin("Dynamic Treatment Effects", s)
    end

    # =========================================================================
    # Plotting
    # =========================================================================
    @testset "Plotting" begin
        pd, _ = _make_lpdid_panel(seed=114)

        r = estimate_lp_did(pd, :y, :treat, 3; pre_window=2)
        p = plot_result(r)
        @test p isa PlotOutput
        @test occursin("LP-DiD", p.html)
        @test occursin("d3", p.html)
    end

    # =========================================================================
    # References
    # =========================================================================
    @testset "References" begin
        pd, _ = _make_lpdid_panel(seed=115)
        r = estimate_lp_did(pd, :y, :treat, 3)

        io = IOBuffer()
        refs(io, r)
        s = String(take!(io))
        @test occursin("Dube", s) || occursin("dube", s) || occursin("Jordà", s) || occursin("jorda", s)
    end

    # =========================================================================
    # Panel lag/lead/diff utilities
    # =========================================================================
    @testset "Panel lag/lead/diff" begin
        pd, _ = _make_lpdid_panel(seed=200, n_units=5, n_periods=10)

        # panel_lag
        l1 = panel_lag(pd, :y, 1)
        @test length(l1) == pd.T_obs
        # First observation per group should be NaN (no lag available)
        for g in 1:pd.n_groups
            first_row = findfirst(i -> pd.group_id[i] == g, 1:pd.T_obs)
            @test isnan(l1[first_row])
        end
        # Non-first observations should have some non-NaN values
        non_nan_count = count(!isnan, l1)
        @test non_nan_count > 0

        # panel_lead
        f1 = panel_lead(pd, :y, 1)
        @test length(f1) == pd.T_obs

        # panel_diff
        d1 = panel_diff(pd, :y)
        @test length(d1) == pd.T_obs

        # add_panel_lag returns new PanelData with extra column
        pd2 = add_panel_lag(pd, :y, 1)
        @test pd2.n_vars == pd.n_vars + 1
        @test "lag1_y" in pd2.varnames

        # add_panel_lead
        pd3 = add_panel_lead(pd, :y, 1)
        @test pd3.n_vars == pd.n_vars + 1
        @test "lead1_y" in pd3.varnames

        # add_panel_diff
        pd4 = add_panel_diff(pd, :y)
        @test pd4.n_vars == pd.n_vars + 1
        @test "d_y" in pd4.varnames
    end

    # =========================================================================
    # DDCG dataset loading
    # =========================================================================
    @testset "DDCG dataset" begin
        ddcg = load_example(:ddcg)
        @test ddcg isa PanelData{Float64}
        @test ddcg.n_groups > 100   # 184 countries
        @test "y" in ddcg.varnames
        @test "dem" in ddcg.varnames
        @test ddcg.T_obs > 5000

        # Should be able to run LP-DiD on it
        r = estimate_lp_did(ddcg, :y, :dem, 5; pre_window=3, ylags=1)
        @test r isa LPDiDResult{Float64}
        @test length(r.coefficients) == 9  # -3 to 5
    end

    # =========================================================================
    # Staggered adoption (multiple cohorts)
    # =========================================================================
    @testset "Staggered adoption" begin
        pd = _make_lpdid_staggered(seed=300)

        r = estimate_lp_did(pd, :y, :treat, 5; pre_window=3, ylags=1)
        @test r isa LPDiDResult{Float64}
        @test length(r.coefficients) == 9

        # Post-treatment should be positive
        h0_idx = findfirst(==(0), r.event_times)
        @test r.coefficients[h0_idx] > 0 || isnan(r.coefficients[h0_idx])
    end

    # =========================================================================
    # Combined options
    # =========================================================================
    @testset "Combined options" begin
        pd = _make_lpdid_staggered(seed=301)

        # PMD + reweighting + nocomp + notyet
        r = estimate_lp_did(pd, :y, :treat, 3;
                             pmd=:max, reweight=true, nocomp=true, notyet=true,
                             ylags=1, dylags=1)
        @test r isa LPDiDResult{Float64}
        @test r.pmd == :max
        @test r.reweight == true
        @test r.nocomp == true
    end

end
