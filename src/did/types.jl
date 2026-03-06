# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for Difference-in-Differences and Event Study LP methods.

Implements types for:
- TWFE DiD and Callaway-Sant'Anna (2021)
- Event Study LP (Jorda 2005 + panel FE)
- LP-DiD (Dube et al. 2025)
- Bacon decomposition (Goodman-Bacon 2021)
- Pre-trend testing and negative weight diagnostics
"""

using LinearAlgebra, StatsAPI

# =============================================================================
# DIDResult — unified across all DiD estimators
# =============================================================================

"""
    DIDResult{T} <: AbstractFrequentistResult

Difference-in-Differences estimation result.

Stores event-study ATT coefficients, confidence intervals, and optionally
group-time ATTs for Callaway-Sant'Anna. All DiD methods return this type.

# Fields
- `att::Vector{T}` — ATT by event-time
- `se::Vector{T}` — standard errors
- `ci_lower::Vector{T}` — lower CI bounds
- `ci_upper::Vector{T}` — upper CI bounds
- `event_times::Vector{Int}` — event-time grid (e.g., [-3,-2,-1,0,1,...,H])
- `reference_period::Int` — omitted period (typically -1)
- `group_time_att::Union{Matrix{T}, Nothing}` — n_cohorts x n_periods (CS only)
- `cohorts::Union{Vector{Int}, Nothing}` — treatment cohort identifiers
- `overall_att::T` — single aggregate ATT
- `overall_se::T` — SE of aggregate ATT
- `n_obs::Int` — total observations
- `n_groups::Int` — number of panel units
- `n_treated::Int` — number of ever-treated units
- `n_control::Int` — number of never-treated units
- `method::Symbol` — :twfe or :callaway_santanna
- `outcome_var::String` — outcome variable name
- `treatment_var::String` — treatment variable name
- `control_group::Symbol` — :never_treated or :not_yet_treated
- `cluster::Symbol` — clustering level (:unit, :time, :twoway)
- `conf_level::T` — confidence level

# References
- Callaway, B. & Sant'Anna, P. H. C. (2021). *JoE* 225(2), 200-230.
- Goodman-Bacon, A. (2021). *JoE* 225(2), 254-277.
"""
struct DIDResult{T<:AbstractFloat} <: AbstractFrequentistResult
    att::Vector{T}
    se::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    event_times::Vector{Int}
    reference_period::Int
    group_time_att::Union{Matrix{T}, Nothing}
    cohorts::Union{Vector{Int}, Nothing}
    overall_att::T
    overall_se::T
    n_obs::Int
    n_groups::Int
    n_treated::Int
    n_control::Int
    method::Symbol
    outcome_var::String
    treatment_var::String
    control_group::Symbol
    cluster::Symbol
    conf_level::T
end

# StatsAPI interface
StatsAPI.nobs(r::DIDResult) = r.n_obs
StatsAPI.coef(r::DIDResult) = r.att
StatsAPI.stderror(r::DIDResult) = r.se
StatsAPI.confint(r::DIDResult) = hcat(r.ci_lower, r.ci_upper)

# =============================================================================
# EventStudyLP — LP-based event study with panel FE
# =============================================================================

"""
    EventStudyLP{T<:AbstractFloat}

Event Study Local Projection result with panel fixed effects.

Stores dynamic treatment effect coefficients at each event-time horizon,
along with per-horizon regression details for diagnostics.

# Fields
- `coefficients::Vector{T}` — beta_h for h = -leads, ..., 0, ..., H
- `se::Vector{T}` — standard errors
- `ci_lower::Vector{T}` — lower CI bounds
- `ci_upper::Vector{T}` — upper CI bounds
- `event_times::Vector{Int}` — event-time grid
- `reference_period::Int` — omitted period (typically -1)
- `B::Vector{Matrix{T}}` — full coefficient matrices per horizon
- `residuals_per_h::Vector{Matrix{T}}` — residuals per horizon
- `vcov::Vector{Matrix{T}}` — VCV matrices per horizon
- `T_eff::Vector{Int}` — effective obs per horizon
- `outcome_var::String` — outcome variable name
- `treatment_var::String` — treatment variable name
- `n_obs::Int` — total observations
- `n_groups::Int` — number of panel units
- `lags::Int` — number of control lags
- `leads::Int` — number of pre-treatment leads
- `horizon::Int` — post-treatment horizon H
- `clean_controls::Bool` — LP-DiD flag (true = only not-yet-treated controls)
- `cluster::Symbol` — clustering level
- `conf_level::T` — confidence level
- `data::PanelData{T}` — original data

# References
- Jorda, O. (2005). *AER* 95(1), 161-182.
- Dube, A. et al. (2025). *JAE*.
"""
struct EventStudyLP{T<:AbstractFloat}
    coefficients::Vector{T}
    se::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    event_times::Vector{Int}
    reference_period::Int
    B::Vector{Matrix{T}}
    residuals_per_h::Vector{Matrix{T}}
    vcov::Vector{Matrix{T}}
    T_eff::Vector{Int}
    outcome_var::String
    treatment_var::String
    n_obs::Int
    n_groups::Int
    lags::Int
    leads::Int
    horizon::Int
    clean_controls::Bool
    cluster::Symbol
    conf_level::T
    data::PanelData{T}
end

# StatsAPI interface
StatsAPI.nobs(r::EventStudyLP) = r.n_obs
StatsAPI.coef(r::EventStudyLP) = r.coefficients
StatsAPI.stderror(r::EventStudyLP) = r.se
StatsAPI.confint(r::EventStudyLP) = hcat(r.ci_lower, r.ci_upper)

# =============================================================================
# LPDiDResult — LP-DiD with full Dube et al. (2025) specification
# =============================================================================

"""
    LPDiDResult{T<:AbstractFloat}

LP-DiD estimation result with full Dube, Girardi, Jordà & Taylor (2025) specification.

# Fields
## Event Study
- `coefficients::Vector{T}` — treatment effects β_h at each event-time
- `se::Vector{T}` — standard errors
- `ci_lower::Vector{T}` — lower CI bounds
- `ci_upper::Vector{T}` — upper CI bounds
- `event_times::Vector{Int}` — event-time grid
- `reference_period::Int` — omitted period (typically -1)
- `nobs_per_horizon::Vector{Int}` — effective observations per horizon

## Pooled Estimates
- `pooled_post` — NamedTuple (coef, se, ci_lower, ci_upper, nobs) or nothing
- `pooled_pre` — NamedTuple (coef, se, ci_lower, ci_upper, nobs) or nothing

## Regression Details
- `vcov::Vector{Matrix{T}}` — VCV matrices per horizon

## Metadata
- `outcome_var::String` — outcome variable name
- `treatment_var::String` — treatment variable name
- `T_obs::Int` — total observations in panel
- `n_groups::Int` — number of panel units
- `specification::Symbol` — :absorbing, :nonabsorbing, or :oneoff
- `pmd::Union{Nothing,Symbol,Int}` — PMD specification
- `reweight::Bool` — whether IPW reweighting was used
- `nocomp::Bool` — whether composition changes were ruled out
- `ylags::Int` — number of outcome lags as controls
- `dylags::Int` — number of ΔY lags as controls
- `pre_window::Int` — pre-treatment event window
- `post_window::Int` — post-treatment event window
- `cluster::Symbol` — clustering level
- `conf_level::T` — confidence level
- `data::PanelData{T}` — original panel data

# References
- Dube, A., Girardi, D., Jordà, Ò. & Taylor, A.M. (2025). *JAE*.
- Acemoglu, D. et al. (2019). *JPE* 127(1), 47-100.
"""
struct LPDiDResult{T<:AbstractFloat}
    coefficients::Vector{T}
    se::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    event_times::Vector{Int}
    reference_period::Int
    nobs_per_horizon::Vector{Int}
    pooled_post::Union{Nothing, @NamedTuple{coef::T, se::T, ci_lower::T, ci_upper::T, nobs::Int}}
    pooled_pre::Union{Nothing, @NamedTuple{coef::T, se::T, ci_lower::T, ci_upper::T, nobs::Int}}
    vcov::Vector{Matrix{T}}
    outcome_var::String
    treatment_var::String
    T_obs::Int
    n_groups::Int
    specification::Symbol
    pmd::Union{Nothing,Symbol,Int}
    reweight::Bool
    nocomp::Bool
    ylags::Int
    dylags::Int
    pre_window::Int
    post_window::Int
    cluster::Symbol
    conf_level::T
    data::PanelData{T}
end

StatsAPI.nobs(r::LPDiDResult) = r.T_obs
StatsAPI.coef(r::LPDiDResult) = r.coefficients
StatsAPI.stderror(r::LPDiDResult) = r.se
StatsAPI.confint(r::LPDiDResult) = hcat(r.ci_lower, r.ci_upper)

# =============================================================================
# BaconDecomposition
# =============================================================================

"""
    BaconDecomposition{T<:AbstractFloat}

Goodman-Bacon (2021) decomposition of the TWFE DiD estimator.

Decomposes the TWFE estimate into a weighted average of all possible 2x2 DiD
comparisons. Weights reflect sample size and variance of treatment.

# Fields
- `estimates::Vector{T}` — 2x2 DiD estimates
- `weights::Vector{T}` — corresponding weights (sum to 1)
- `comparison_type::Vector{Symbol}` — :earlier_vs_later, :later_vs_earlier, :treated_vs_untreated
- `cohort_i::Vector{Int}` — first cohort in each 2x2
- `cohort_j::Vector{Int}` — second cohort (or 0 for never-treated)
- `overall_att::T` — weighted average = TWFE estimate

# Reference
Goodman-Bacon, A. (2021). *JoE* 225(2), 254-277.
"""
struct BaconDecomposition{T<:AbstractFloat}
    estimates::Vector{T}
    weights::Vector{T}
    comparison_type::Vector{Symbol}
    cohort_i::Vector{Int}
    cohort_j::Vector{Int}
    overall_att::T
end

# =============================================================================
# Diagnostic result types
# =============================================================================

"""
    PretrendTestResult{T<:AbstractFloat}

Joint F-test result for parallel trends assumption.

Tests H0: all pre-treatment event-time coefficients are jointly zero.
A rejection suggests violation of parallel trends.

# Fields
- `statistic::T` — F-statistic (or Wald chi-squared)
- `pvalue::T` — p-value
- `df::Int` — degrees of freedom (number of pre-treatment periods)
- `pre_coefficients::Vector{T}` — pre-treatment coefficients
- `pre_se::Vector{T}` — SEs of pre-treatment coefficients
- `test_type::Symbol` — :f_test or :wald
"""
struct PretrendTestResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    df::Int
    pre_coefficients::Vector{T}
    pre_se::Vector{T}
    test_type::Symbol
end

"""
    NegativeWeightResult{T<:AbstractFloat}

de Chaisemartin & D'Haultfoeuille (2020) negative weight diagnostic.

Checks whether the TWFE estimator places negative weights on some
group-time ATTs, which can cause sign reversal of the overall estimate.

# Fields
- `has_negative_weights::Bool` — true if any weights are negative
- `n_negative::Int` — count of negative weights
- `total_negative_weight::T` — sum of negative weights
- `weights::Vector{T}` — all weights
- `cohort_time_pairs::Vector{Tuple{Int,Int}}` — (cohort, time) for each weight
"""
struct NegativeWeightResult{T<:AbstractFloat}
    has_negative_weights::Bool
    n_negative::Int
    total_negative_weight::T
    weights::Vector{T}
    cohort_time_pairs::Vector{Tuple{Int,Int}}
end

# =============================================================================
# HonestDiD Sensitivity Analysis
# =============================================================================

"""
    HonestDiDResult{T<:AbstractFloat}

Rambachan & Roth (2023) HonestDiD sensitivity analysis result.

Provides robust confidence sets under bounded violations of parallel trends.
The `Mbar` parameter controls the maximum allowed violation magnitude.

# Fields
- `Mbar::T` — violation bound used
- `robust_ci_lower::Vector{T}` — robust CI lower bounds per post-period
- `robust_ci_upper::Vector{T}` — robust CI upper bounds per post-period
- `original_ci_lower::Vector{T}` — original CIs for comparison
- `original_ci_upper::Vector{T}`
- `breakdown_value::T` — smallest Mbar making result insignificant
- `post_event_times::Vector{Int}` — post-treatment event-time grid
- `post_att::Vector{T}` — post-treatment point estimates
- `conf_level::T`

# Reference
Rambachan, A. & Roth, J. (2023). *RES* 90(5), 2555-2591.
"""
struct HonestDiDResult{T<:AbstractFloat}
    Mbar::T
    robust_ci_lower::Vector{T}
    robust_ci_upper::Vector{T}
    original_ci_lower::Vector{T}
    original_ci_upper::Vector{T}
    breakdown_value::T
    post_event_times::Vector{Int}
    post_att::Vector{T}
    conf_level::T
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, r::DIDResult{T}) where {T}
    method_str = r.method == :twfe ? "Two-Way Fixed Effects" :
                 r.method == :callaway_santanna ? "Callaway-Sant'Anna (2021)" :
                 r.method == :sun_abraham ? "Sun-Abraham (2021)" :
                 r.method == :bjs ? "Borusyak-Jaravel-Spiess (2024)" :
                 r.method == :did_multiplegt ? "de Chaisemartin-D'Haultfoeuille (2020)" :
                 string(r.method)
    cluster_str = r.cluster == :twoway ? "Two-way (unit + time)" :
                  r.cluster == :unit ? "Unit-clustered" : "Time-clustered"

    spec = Any[
        "Method"          method_str;
        "Outcome"         r.outcome_var;
        "Treatment"       r.treatment_var;
        "Clustering"      cluster_str;
        "Control group"   string(r.control_group);
        "Groups"          r.n_groups;
        "Treated units"   r.n_treated;
        "Control units"   r.n_control;
        "Observations"    r.n_obs;
    ]

    _pretty_table(io, spec;
        title = "Difference-in-Differences -- $method_str",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    # Event-study coefficient table
    n_evt = length(r.event_times)
    evt_names = ["e=" * string(e) for e in r.event_times]
    _coef_table(io, "Event-Study Coefficients", evt_names,
                T.(r.att), T.(r.se); dist=:z)

    # Overall ATT
    println(io)
    z_ov = r.overall_se > 0 ? r.overall_att / r.overall_se : T(NaN)
    p_ov = isnan(z_ov) ? T(NaN) : 2 * (1 - cdf(Normal(), abs(z_ov)))
    overall = Any[
        "Overall ATT"  _fmt(r.overall_att);
        "SE"           _fmt(r.overall_se);
        "z"            _fmt(z_ov);
        "P>|z|"        _format_pvalue(p_ov);
    ]
    _pretty_table(io, overall;
        title = "Aggregate Treatment Effect",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::EventStudyLP{T}) where {T}
    lp_str = r.clean_controls ? "LP-DiD (Dube et al. 2025)" : "Event Study LP"
    cluster_str = r.cluster == :twoway ? "Two-way (unit + time)" :
                  r.cluster == :unit ? "Unit-clustered" : "Time-clustered"

    spec = Any[
        "Method"          lp_str;
        "Outcome"         r.outcome_var;
        "Treatment"       r.treatment_var;
        "Clustering"      cluster_str;
        "Leads"           r.leads;
        "Lags"            r.lags;
        "Horizon"         r.horizon;
        "Groups"          r.n_groups;
        "Observations"    r.n_obs;
    ]

    _pretty_table(io, spec;
        title = "$lp_str Estimation",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    evt_names = ["h=" * string(e) for e in r.event_times]
    _coef_table(io, "Dynamic Treatment Effects", evt_names,
                T.(r.coefficients), T.(r.se); dist=:z)
end

function Base.show(io::IO, r::LPDiDResult{T}) where {T}
    spec_str = r.specification == :absorbing ? "Absorbing" :
               r.specification == :nonabsorbing ? "Non-absorbing" :
               r.specification == :oneoff ? "One-off" : string(r.specification)
    pmd_str = r.pmd === nothing ? "Standard (long diff)" :
              r.pmd == :max ? "PMD (all pre-treatment)" :
              "PMD (MA k=$(r.pmd))"
    cluster_str = r.cluster == :twoway ? "Two-way (unit + time)" :
                  r.cluster == :unit ? "Unit-clustered" : "Time-clustered"

    spec = Any[
        "Method"          "LP-DiD (Dube et al. 2025)";
        "Treatment"       spec_str;
        "LHS"             pmd_str;
        "Outcome"         r.outcome_var;
        "Treatment var"   r.treatment_var;
        "Clustering"      cluster_str;
        "Reweighted"      r.reweight ? "Yes (IPW)" : "No";
        "Y lags"          r.ylags;
        "ΔY lags"         r.dylags;
        "Pre-window"      r.pre_window;
        "Post-window"     r.post_window;
        "Groups"          r.n_groups;
        "Observations"    r.T_obs;
    ]

    _pretty_table(io, spec;
        title = "LP-DiD Estimation -- Dube, Girardi, Jorda & Taylor (2025)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    evt_names = ["h=" * string(e) for e in r.event_times]
    _coef_table(io, "Dynamic Treatment Effects", evt_names,
                T.(r.coefficients), T.(r.se); dist=:z)

    if r.pooled_post !== nothing || r.pooled_pre !== nothing
        println(io)
        pooled_names = String[]
        pooled_coefs = T[]
        pooled_ses = T[]
        if r.pooled_pre !== nothing
            push!(pooled_names, "Pre-pooled")
            push!(pooled_coefs, r.pooled_pre.coef)
            push!(pooled_ses, r.pooled_pre.se)
        end
        if r.pooled_post !== nothing
            push!(pooled_names, "Post-pooled")
            push!(pooled_coefs, r.pooled_post.coef)
            push!(pooled_ses, r.pooled_post.se)
        end
        _coef_table(io, "Pooled Estimates", pooled_names,
                    pooled_coefs, pooled_ses; dist=:z)
    end
end

function Base.show(io::IO, r::BaconDecomposition{T}) where {T}
    n = length(r.estimates)
    data = Matrix{Any}(undef, n, 4)
    for i in 1:n
        type_str = r.comparison_type[i] == :treated_vs_untreated ? "Treated vs Untreated" :
                   r.comparison_type[i] == :earlier_vs_later ? "Earlier vs Later" :
                   r.comparison_type[i] == :later_vs_earlier ? "Later vs Earlier" :
                   string(r.comparison_type[i])
        data[i, 1] = type_str
        data[i, 2] = "$(r.cohort_i[i]) vs $(r.cohort_j[i])"
        data[i, 3] = _fmt(r.estimates[i])
        data[i, 4] = _fmt(r.weights[i])
    end
    _pretty_table(io, data;
        title = "Goodman-Bacon Decomposition (Overall ATT = $(_fmt(r.overall_att)))",
        column_labels = ["Type", "Comparison", "Estimate", "Weight"],
        alignment = [:l, :l, :r, :r],
    )
end

function Base.show(io::IO, r::PretrendTestResult{T}) where {T}
    data = Any[
        "Test"           r.test_type == :f_test ? "Joint F-test" : "Wald test";
        "Statistic"      _fmt(r.statistic);
        "P-value"        _format_pvalue(r.pvalue);
        "DF"             r.df;
        "Pre-periods"    length(r.pre_coefficients);
    ]
    _pretty_table(io, data;
        title = "Pre-Trend Test",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::NegativeWeightResult{T}) where {T}
    status = r.has_negative_weights ? "NEGATIVE WEIGHTS DETECTED" : "No negative weights"
    data = Any[
        "Status"               status;
        "Negative weights"     r.n_negative;
        "Total negative wt"    _fmt(r.total_negative_weight);
        "Total weights"        length(r.weights);
    ]
    _pretty_table(io, data;
        title = "Negative Weight Diagnostic (de Chaisemartin-D'Haultfoeuille 2020)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::HonestDiDResult{T}) where {T}
    n = length(r.post_event_times)
    data = Matrix{Any}(undef, n, 5)
    for i in 1:n
        data[i, 1] = "e=$(r.post_event_times[i])"
        data[i, 2] = _fmt(r.post_att[i])
        data[i, 3] = "[$(_fmt(r.original_ci_lower[i])), $(_fmt(r.original_ci_upper[i]))]"
        data[i, 4] = "[$(_fmt(r.robust_ci_lower[i])), $(_fmt(r.robust_ci_upper[i]))]"
        data[i, 5] = (r.robust_ci_lower[i] > 0 || r.robust_ci_upper[i] < 0) ? "***" : ""
    end
    _pretty_table(io, data;
        title = "HonestDiD Sensitivity Analysis (Mbar = $(_fmt(r.Mbar)))",
        column_labels = ["Period", "ATT", "Original CI", "Robust CI", "Sig"],
        alignment = [:l, :r, :r, :r, :c],
    )
    println(io)
    summary_data = Any[
        "Mbar"              _fmt(r.Mbar);
        "Breakdown value"   _fmt(r.breakdown_value);
        "Conf. level"       _fmt(r.conf_level);
    ]
    _pretty_table(io, summary_data;
        title = "Sensitivity Summary",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end
