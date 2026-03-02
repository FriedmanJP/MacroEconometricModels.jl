# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Type definitions for Difference-in-Differences and Event Study LP methods.

Implements types for:
- TWFE DiD and Callaway-Sant'Anna (2021)
- Event Study LP (Jorda 2005 + panel FE)
- LP-DiD (Dube et al. 2023)
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
- Dube, A. et al. (2023). NBER WP 31184.
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
# Display
# =============================================================================

function Base.show(io::IO, r::DIDResult{T}) where {T}
    method_str = r.method == :twfe ? "Two-Way Fixed Effects" :
                 r.method == :callaway_santanna ? "Callaway-Sant'Anna (2021)" :
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
    lp_str = r.clean_controls ? "LP-DiD (Dube et al. 2023)" : "Event Study LP"
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
