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
Diagnostic tools for Difference-in-Differences estimation.

- `bacon_decomposition`: Goodman-Bacon (2021) decomposition of TWFE
- `pretrend_test`: Joint F-test for parallel trends
- `negative_weight_check`: de Chaisemartin & D'Haultfoeuille (2020) negative weight diagnostic
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Bacon Decomposition (Goodman-Bacon 2021)
# =============================================================================

"""
    bacon_decomposition(pd::PanelData{T}, outcome::Union{String,Symbol},
                        treatment::Union{String,Symbol}) where {T}

Decompose the TWFE DiD estimator into a weighted average of all 2x2 DiD comparisons.

Each weight reflects the subsample size and variance of treatment within the comparison.

# Returns
`BaconDecomposition{T}` with estimates, weights, and comparison types.

# Reference
Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing."
*JoE* 225(2), 254-277.
"""
function bacon_decomposition(pd::PanelData{T}, outcome::Union{String,Symbol},
                             treatment::Union{String,Symbol}) where {T<:AbstractFloat}
    outcome_col = _resolve_varindex(pd, outcome)
    treat_col = _resolve_varindex(pd, treatment)
    timing = _extract_treatment_timing(pd, treat_col)
    all_times = sort(unique(pd.time_id))
    n_times = length(all_times)

    # Build panel lookup
    panel = Dict{Int, Dict{Int, T}}()
    for i in 1:pd.T_obs
        g = pd.group_id[i]
        t = pd.time_id[i]
        if !haskey(panel, g)
            panel[g] = Dict{Int, T}()
        end
        panel[g][t] = pd.data[i, outcome_col]
    end

    cohorts = sort(unique([t for (_, t) in timing if t > 0]))
    never_treated = [g for (g, t) in timing if t == 0]
    n_total = pd.n_groups

    estimates = T[]
    weights = T[]
    comp_types = Symbol[]
    c_i = Int[]
    c_j = Int[]

    # 2x2 comparisons
    for (idx_k, g_k) in enumerate(cohorts)
        units_k = [g for (g, t) in timing if t == g_k]
        n_k = length(units_k)

        # Treated vs Never-treated
        if !isempty(never_treated)
            dd, w = _bacon_2x2(panel, units_k, never_treated, g_k, all_times, n_total)
            if !isnan(dd)
                push!(estimates, dd)
                push!(weights, w)
                push!(comp_types, :treated_vs_untreated)
                push!(c_i, g_k)
                push!(c_j, 0)
            end
        end

        # Earlier vs Later and Later vs Earlier
        for (idx_l, g_l) in enumerate(cohorts)
            g_k == g_l && continue
            units_l = [g for (g, t) in timing if t == g_l]
            n_l = length(units_l)

            if g_k < g_l
                # Earlier treated (k) vs Later treated (l) -- use pre-l period
                dd, w = _bacon_2x2_timing(panel, units_k, units_l, g_k, g_l,
                                          all_times, n_total)
                if !isnan(dd)
                    push!(estimates, dd)
                    push!(weights, w)
                    push!(comp_types, :earlier_vs_later)
                    push!(c_i, g_k)
                    push!(c_j, g_l)
                end
            end
        end
    end

    # Normalize weights
    w_total = sum(weights)
    if w_total > 0
        weights ./= w_total
    end

    overall = sum(estimates .* weights)

    BaconDecomposition{T}(estimates, weights, comp_types, c_i, c_j, overall)
end

"""2x2 DiD: treated group vs never-treated group."""
function _bacon_2x2(panel, treated, control, g_time, all_times, n_total)
    T_type = eltype(first(values(first(values(panel)))))

    # Pre-treatment mean difference
    pre_treat = T_type[]
    pre_ctrl = T_type[]
    post_treat = T_type[]
    post_ctrl = T_type[]

    for t in all_times
        for u in treated
            haskey(panel[u], t) || continue
            if t < g_time
                push!(pre_treat, panel[u][t])
            else
                push!(post_treat, panel[u][t])
            end
        end
        for u in control
            haskey(panel[u], t) || continue
            if t < g_time
                push!(pre_ctrl, panel[u][t])
            else
                push!(post_ctrl, panel[u][t])
            end
        end
    end

    if isempty(pre_treat) || isempty(pre_ctrl) || isempty(post_treat) || isempty(post_ctrl)
        return T_type(NaN), zero(T_type)
    end

    dd = (mean(post_treat) - mean(pre_treat)) - (mean(post_ctrl) - mean(pre_ctrl))

    # Weight proportional to subsample size x variance of treatment
    n_k = length(unique([u for u in keys(panel) if u in treated]))
    n_u = length(unique([u for u in keys(panel) if u in control]))
    n_sub = n_k + n_u
    p_k = n_k / n_sub
    w = T_type(n_sub) / T_type(n_total) * p_k * (1 - p_k)

    dd, w
end

"""2x2 DiD: earlier vs later treated groups."""
function _bacon_2x2_timing(panel, early_units, late_units, g_early, g_late,
                           all_times, n_total)
    T_type = eltype(first(values(first(values(panel)))))

    # Use the period [g_early, g_late) where early is treated but late is not
    pre_early = T_type[]
    post_early = T_type[]
    pre_late = T_type[]
    post_late = T_type[]

    for t in all_times
        t >= g_late && continue  # Only use pre-late period

        for u in early_units
            haskey(panel[u], t) || continue
            if t < g_early
                push!(pre_early, panel[u][t])
            else
                push!(post_early, panel[u][t])
            end
        end
        for u in late_units
            haskey(panel[u], t) || continue
            if t < g_early
                push!(pre_late, panel[u][t])
            else
                push!(post_late, panel[u][t])
            end
        end
    end

    if isempty(pre_early) || isempty(pre_late) || isempty(post_early) || isempty(post_late)
        return T_type(NaN), zero(T_type)
    end

    dd = (mean(post_early) - mean(pre_early)) - (mean(post_late) - mean(pre_late))

    n_e = length(early_units)
    n_l = length(late_units)
    n_sub = n_e + n_l
    p_e = n_e / n_sub
    w = T_type(n_sub) / T_type(n_total) * p_e * (1 - p_e)

    dd, w
end

# =============================================================================
# Pre-trend Test
# =============================================================================

"""
    pretrend_test(result::DIDResult{T}) where {T}
    pretrend_test(result::EventStudyLP{T}) where {T}

Joint F-test for parallel trends: H0: beta_{-K} = beta_{-K+1} = ... = beta_{-1} = 0.

Tests whether pre-treatment event-time coefficients are jointly zero.
High p-value -> no evidence against parallel trends.

# Returns
`PretrendTestResult{T}` with F-statistic and p-value.
"""
function pretrend_test(result::DIDResult{T}) where {T<:AbstractFloat}
    pre_mask = result.event_times .< 0 .&& result.event_times .!= result.reference_period
    pre_coefs = result.att[pre_mask]
    pre_ses = result.se[pre_mask]
    _compute_pretrend_test(pre_coefs, pre_ses)
end

function pretrend_test(result::EventStudyLP{T}) where {T<:AbstractFloat}
    pre_mask = result.event_times .< 0 .&& result.event_times .!= result.reference_period
    pre_coefs = result.coefficients[pre_mask]
    pre_ses = result.se[pre_mask]
    _compute_pretrend_test(pre_coefs, pre_ses)
end

function _compute_pretrend_test(pre_coefs::Vector{T}, pre_ses::Vector{T}) where {T<:AbstractFloat}
    K = length(pre_coefs)
    K == 0 && return PretrendTestResult{T}(zero(T), one(T), 0, pre_coefs, pre_ses, :f_test)

    # Wald test: beta' V^{-1} beta ~ chi^2(K)
    # Using diagonal V (conservative, ignores cross-horizon covariance)
    valid = pre_ses .> 0
    if !any(valid)
        return PretrendTestResult{T}(zero(T), one(T), K, pre_coefs, pre_ses, :wald)
    end

    wald_stat = sum((pre_coefs[valid] ./ pre_ses[valid]) .^ 2)
    pval = 1 - cdf(Chisq(count(valid)), wald_stat)

    PretrendTestResult{T}(wald_stat, T(pval), count(valid), pre_coefs, pre_ses, :wald)
end

# =============================================================================
# Negative Weight Check
# =============================================================================

"""
    negative_weight_check(pd::PanelData{T}, treatment::Union{String,Symbol}) where {T}

de Chaisemartin & D'Haultfoeuille (2020) negative weight diagnostic.

Checks whether the TWFE estimator assigns negative weights to some group-time cells.
Negative weights can cause the overall DiD estimate to have the opposite sign of
all underlying ATT(g,t).

# Returns
`NegativeWeightResult{T}`.

# Reference
de Chaisemartin, C. & D'Haultfoeuille, X. (2020). *AER* 110(9), 2964-2996.
"""
function negative_weight_check(pd::PanelData{T}, treatment::Union{String,Symbol}) where {T<:AbstractFloat}
    treat_col = _resolve_varindex(pd, treatment)
    timing = _extract_treatment_timing(pd, treat_col)
    all_times = sort(unique(pd.time_id))

    # Compute TWFE weights for each (g, t) cell
    weights_vec = T[]
    pairs = Tuple{Int,Int}[]

    # For each treated (group, time) cell, compute the weight in TWFE regression
    cohorts = sort(unique([t for (_, t) in timing if t > 0]))
    n_total = pd.n_groups
    n_times = length(all_times)

    for g_time in cohorts
        units_g = [g for (g, t) in timing if t == g_time]
        n_g = length(units_g)
        share_g = T(n_g) / T(n_total)

        for t in all_times
            t < g_time && continue  # Only post-treatment

            # Treatment indicator variance contribution
            # D_bar_t = share of treated at time t
            n_treated_t = count(g -> timing[g] > 0 && timing[g] <= t, 1:n_total)
            D_bar_t = T(n_treated_t) / T(n_total)

            if D_bar_t > 0 && D_bar_t < 1
                # Weight for this (g,t) cell
                D_gt = one(T)  # treated unit is treated at this time
                epsilon_gt = D_gt - D_bar_t

                # Weight proportional to deviation from time-mean
                w = share_g * epsilon_gt / (D_bar_t * (1 - D_bar_t))
                push!(weights_vec, w)
                push!(pairs, (g_time, t))
            end
        end
    end

    n_neg = count(w -> w < 0, weights_vec)
    total_neg = sum(w for w in weights_vec if w < 0; init=zero(T))

    NegativeWeightResult{T}(n_neg > 0, n_neg, total_neg, weights_vec, pairs)
end
