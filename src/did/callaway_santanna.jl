# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Callaway & Sant'Anna (2021) group-time ATT estimator.

Estimates ATT(g,t) for each cohort g and time t using outcome regression,
then aggregates to event-time ATTs with cohort-size weights.
"""

using LinearAlgebra, Statistics, Distributions

"""
    _estimate_callaway_santanna(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                                leads::Int=0, horizon::Int=5,
                                control_group::Symbol=:never_treated,
                                cluster::Symbol=:unit,
                                conf_level::Real=0.95,
                                base_period::Symbol=:varying) where {T}

Internal Callaway-Sant'Anna estimator.

Algorithm:
1. Identify cohorts G = unique treatment times
2. For each (g, t): compute ATT(g,t) = E[ΔY | G=g] - E[ΔY | C]
   - `:varying` base: pre-treatment uses ΔY = Y_t - Y_{t-1}; post uses ΔY = Y_t - Y_{g-1}
   - `:universal` base: always ΔY = Y_t - Y_{g-1}
3. Aggregate: ATT(e) = sum_g w_g * ATT(g, g+e) with w_g proportional to cohort size
4. SEs via analytical variance (delta method)
"""
function _estimate_callaway_santanna(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                                     leads::Int=0, horizon::Int=5,
                                     control_group::Symbol=:never_treated,
                                     cluster::Symbol=:unit,
                                     conf_level::Real=0.95,
                                     base_period::Symbol=:varying) where {T<:AbstractFloat}

    base_period in (:varying, :universal) ||
        throw(ArgumentError("base_period must be :varying or :universal, got :$base_period"))

    timing = _extract_treatment_timing(pd, treat_col)

    # Override timing with cohort_id if present
    if pd.cohort_id !== nothing
        for g in 1:pd.n_groups
            mask = pd.group_id .== g
            timing[g] = pd.cohort_id[findfirst(mask)]
        end
    end

    all_times = sort(unique(pd.time_id))
    n_times = length(all_times)

    # Identify cohorts and control groups
    cohorts = sort(unique([t for (_, t) in timing if t > 0]))
    never_treated = [g for (g, t) in timing if t == 0]
    n_cohorts = length(cohorts)
    n_control = length(never_treated)

    if control_group == :never_treated && n_control == 0
        throw(ArgumentError("No never-treated units found. Use control_group=:not_yet_treated"))
    end

    # Build panel: for each unit, extract outcome time series
    # Store as Dict{Int, Dict{Int, T}} -- group -> time -> outcome
    panel = Dict{Int, Dict{Int, T}}()
    for i in 1:pd.T_obs
        g = pd.group_id[i]
        t = pd.time_id[i]
        if !haskey(panel, g)
            panel[g] = Dict{Int, T}()
        end
        panel[g][t] = pd.data[i, outcome_col]
    end

    # Compute group-time ATTs
    group_time_att = fill(T(NaN), n_cohorts, n_times)
    group_time_se = fill(T(NaN), n_cohorts, n_times)

    for (ci, g_time) in enumerate(cohorts)
        # Units in this cohort
        cohort_units = [g for (g, t) in timing if t == g_time]
        n_g = length(cohort_units)

        # Universal base period: g_time - 1 (last pre-treatment period)
        universal_base = g_time - 1
        universal_base < minimum(all_times) && continue

        for (ti, t) in enumerate(all_times)
            # Determine base period for this (g, t) cell
            if base_period == :varying && t < g_time
                # Pre-treatment with varying base: compare adjacent periods
                ti <= 1 && continue  # need a preceding period
                base_t = all_times[ti - 1]
            else
                # Post-treatment (both modes) or universal pre-treatment
                base_t = universal_base
            end

            # Control group selection
            # Threshold = max(t, base_t): exclude units treated at or before the base period
            ctrl_threshold = max(t, base_t)
            if control_group == :never_treated
                ctrl_units = never_treated
            else  # :not_yet_treated
                ctrl_units = [u for (u, ut) in timing if ut == 0 || ut > ctrl_threshold]
                # Exclude the current cohort from controls
                ctrl_units = filter(u -> !(u in cohort_units), ctrl_units)
            end
            isempty(ctrl_units) && continue

            # Compute ATT(g,t) = mean(DeltaY_treated) - mean(DeltaY_control)
            # where DeltaY = Y_t - Y_{base}
            dy_treated = T[]
            for u in cohort_units
                haskey(panel[u], t) && haskey(panel[u], base_t) &&
                    push!(dy_treated, panel[u][t] - panel[u][base_t])
            end
            isempty(dy_treated) && continue

            dy_control = T[]
            for u in ctrl_units
                haskey(panel[u], t) && haskey(panel[u], base_t) &&
                    push!(dy_control, panel[u][t] - panel[u][base_t])
            end
            isempty(dy_control) && continue

            att_gt = mean(dy_treated) - mean(dy_control)
            # SE: sqrt(var_treated/n_treated + var_control/n_control)
            # Guard single-observation case where var() returns NaN (Bessel n-1=0)
            v_treat = length(dy_treated) > 1 ? var(dy_treated) / length(dy_treated) : zero(T)
            v_ctrl = length(dy_control) > 1 ? var(dy_control) / length(dy_control) : zero(T)
            se_gt = sqrt(v_treat + v_ctrl)

            group_time_att[ci, ti] = att_gt
            group_time_se[ci, ti] = se_gt
        end
    end

    # Aggregate to event-time
    reference_period = -1
    event_times_all = collect(-leads:horizon)
    n_evt = length(event_times_all)
    att_agg = zeros(T, n_evt)
    se_agg = zeros(T, n_evt)

    for (ei, e) in enumerate(event_times_all)
        # With universal base, reference period e=-1 is zero by construction
        if base_period == :universal && e == reference_period
            att_agg[ei] = zero(T)
            se_agg[ei] = zero(T)
            continue
        end

        # Aggregate across cohorts for event-time e
        att_vals = T[]
        se_vals = T[]
        weights_e = T[]

        for (ci, g_time) in enumerate(cohorts)
            t_target = g_time + e
            ti = findfirst(==(t_target), all_times)
            ti === nothing && continue
            isnan(group_time_att[ci, ti]) && continue

            # Weight = cohort size
            n_g = T(count(g -> timing[g] == g_time, keys(timing)))
            push!(att_vals, group_time_att[ci, ti])
            push!(se_vals, group_time_se[ci, ti])
            push!(weights_e, n_g)
        end

        if !isempty(att_vals)
            w_total = sum(weights_e)
            w_norm = weights_e ./ w_total
            att_agg[ei] = sum(w_norm .* att_vals)
            # SE of weighted average (assuming independence across cohorts)
            se_agg[ei] = sqrt(sum((w_norm .^ 2) .* (se_vals .^ 2)))
        end
    end

    # CIs
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = att_agg .- z .* se_agg
    ci_upper = att_agg .+ z .* se_agg

    # Overall ATT
    post_mask = event_times_all .>= 0
    post_att = att_agg[post_mask]
    post_se = se_agg[post_mask]
    nonzero_post = post_se .> 0
    if any(nonzero_post)
        n_post = count(nonzero_post)
        overall_att = mean(post_att[nonzero_post])
        overall_se = sqrt(sum(post_se[nonzero_post] .^ 2)) / n_post
    else
        overall_att = zero(T)
        overall_se = zero(T)
    end

    n_treated = length([g for (g, t) in timing if t > 0])

    DIDResult{T}(att_agg, se_agg, ci_lower, ci_upper, event_times_all,
                 reference_period, group_time_att, cohorts,
                 overall_att, overall_se,
                 pd.T_obs, pd.n_groups, n_treated, n_control,
                 :callaway_santanna,
                 pd.varnames[outcome_col], pd.varnames[treat_col],
                 control_group, cluster, T(conf_level))
end
