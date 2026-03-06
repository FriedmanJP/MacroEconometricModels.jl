# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
de Chaisemartin & D'Haultfoeuille (2020) `did_multiplegt` estimator for staggered DiD.

For binary absorbing treatment, computes cohort-specific DiD effects and aggregates
with cohort-size weights. SEs via unit-level block bootstrap.

# Reference
de Chaisemartin, C. & D'Haultfoeuille, X. (2020). *AER* 110(9), 2964-2996.
"""

using LinearAlgebra, Statistics, Distributions, Random

# =============================================================================
# Point estimate helper
# =============================================================================

"""
    _dcdh_point_estimate(panel, timing, cohorts, never_treated, event_times_all,
                         reference_period, control_group, all_times, n_groups)

Compute de Chaisemartin-D'Haultfoeuille point estimates for all event times.

For each cohort g (switching at time g):
    DID_g(e) = mean(Y_{g+e} - Y_{g-1} | cohort=g) - mean(Y_{g+e} - Y_{g-1} | control)

Aggregates with cohort-size weights: ATT(e) = sum_g w_g * DID_g(e).

Returns `(att, group_time_effects)` where `group_time_effects` is n_cohorts x n_events.
"""
function _dcdh_point_estimate(panel::Dict{Int, Dict{Int, V}},
                              timing::Dict{Int, Int},
                              cohorts::Vector{Int},
                              never_treated::Vector{Int},
                              event_times_all::Vector{Int},
                              reference_period::Int,
                              control_group::Symbol,
                              all_times::Vector{Int},
                              n_groups::Int) where {V}
    # Infer float type from panel values
    T_type = if !isempty(panel)
        first_unit = first(panel)
        if !isempty(first_unit.second)
            typeof(first(values(first_unit.second)))
        else
            Float64
        end
    else
        Float64
    end

    n_cohorts = length(cohorts)
    n_evt = length(event_times_all)

    group_time_effects = fill(T_type(NaN), n_cohorts, n_evt)

    # Cohort sizes for weighting
    cohort_sizes = zeros(T_type, n_cohorts)
    cohort_unit_map = Dict{Int, Vector{Int}}()  # g_time -> [unit_ids]

    for (ci, g_time) in enumerate(cohorts)
        units = [g for (g, t) in timing if t == g_time]
        cohort_unit_map[g_time] = units
        cohort_sizes[ci] = T_type(length(units))
    end

    for (ci, g_time) in enumerate(cohorts)
        cohort_units = cohort_unit_map[g_time]
        base_t = g_time - 1

        # Skip if base period is before available data
        base_t < minimum(all_times) && continue

        for (ei, e) in enumerate(event_times_all)
            # Reference period: set to zero
            if e == reference_period
                group_time_effects[ci, ei] = zero(T_type)
                continue
            end

            t_target = g_time + e

            # Skip if target time outside data range
            t_target < minimum(all_times) && continue
            t_target > maximum(all_times) && continue

            # Treated group: change in outcome
            dy_treat = T_type[]
            for u in cohort_units
                if haskey(panel, u) && haskey(panel[u], t_target) && haskey(panel[u], base_t)
                    push!(dy_treat, panel[u][t_target] - panel[u][base_t])
                end
            end
            isempty(dy_treat) && continue

            # Control group selection
            if control_group == :never_treated
                ctrl_units = never_treated
            else  # :not_yet_treated
                ctrl_units = [u for (u, ut) in timing if (ut == 0 || ut > t_target)]
                # Exclude the current cohort from controls
                ctrl_units = filter(u -> !(u in cohort_units), ctrl_units)
            end
            isempty(ctrl_units) && continue

            # Control group: change in outcome
            dy_ctrl = T_type[]
            for u in ctrl_units
                if haskey(panel, u) && haskey(panel[u], t_target) && haskey(panel[u], base_t)
                    push!(dy_ctrl, panel[u][t_target] - panel[u][base_t])
                end
            end
            isempty(dy_ctrl) && continue

            group_time_effects[ci, ei] = mean(dy_treat) - mean(dy_ctrl)
        end
    end

    # Aggregate across cohorts for each event time
    att = zeros(T_type, n_evt)

    for (ei, e) in enumerate(event_times_all)
        if e == reference_period
            att[ei] = zero(T_type)
            continue
        end

        vals = T_type[]
        weights_e = T_type[]

        for ci in 1:n_cohorts
            isnan(group_time_effects[ci, ei]) && continue
            push!(vals, group_time_effects[ci, ei])
            push!(weights_e, cohort_sizes[ci])
        end

        if !isempty(vals)
            w_total = sum(weights_e)
            w_norm = weights_e ./ w_total
            att[ei] = sum(w_norm .* vals)
        end
    end

    return (att, group_time_effects)
end

# =============================================================================
# Main estimator
# =============================================================================

"""
    _estimate_did_multiplegt(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                              leads::Int=0, horizon::Int=5,
                              control_group::Symbol=:never_treated,
                              cluster::Symbol=:unit,
                              conf_level::Real=0.95,
                              n_boot::Int=200) where {T}

de Chaisemartin & D'Haultfoeuille (2020) estimator for staggered DiD with
binary absorbing treatment.

# Algorithm
1. For each cohort g (switching at time g): compute
       DID_g(e) = mean(Y_{g+e} - Y_{g-1} | cohort=g) - mean(Y_{g+e} - Y_{g-1} | untreated at g+e)
2. Aggregate: ATT(e) = sum_g w_g * DID_g(e) with w_g = n_g / sum n_g (cohort-size weights)
3. SEs via unit-level block bootstrap (resample entire unit time series)

# Arguments
- `pd`: Panel data with outcome and treatment timing columns
- `outcome_col`: Column index of the outcome variable
- `treat_col`: Column index of the treatment timing variable
- `leads`: Number of pre-treatment periods (default: 0)
- `horizon`: Post-treatment horizon (default: 5)
- `control_group`: `:never_treated` (default) or `:not_yet_treated`
- `cluster`: SE clustering (default: `:unit`)
- `conf_level`: Confidence level (default: 0.95)
- `n_boot`: Number of bootstrap replications (default: 200)

# Returns
`DIDResult{T}` with event-study ATTs, bootstrap SEs, and CIs.

# Reference
de Chaisemartin, C. & D'Haultfoeuille, X. (2020). *AER* 110(9), 2964-2996.
"""
function _estimate_did_multiplegt(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                                   leads::Int=0, horizon::Int=5,
                                   control_group::Symbol=:never_treated,
                                   cluster::Symbol=:unit,
                                   conf_level::Real=0.95,
                                   n_boot::Int=200) where {T<:AbstractFloat}

    timing = _extract_treatment_timing(pd, treat_col)

    # Override timing with cohort_id if present
    if pd.cohort_id !== nothing
        for g in 1:pd.n_groups
            mask = pd.group_id .== g
            timing[g] = pd.cohort_id[findfirst(mask)]
        end
    end

    all_times = sort(unique(pd.time_id))

    # Identify cohorts and control groups
    cohorts = sort(unique([t for (_, t) in timing if t > 0]))
    never_treated = [g for (g, t) in timing if t == 0]
    n_treated = length([g for (g, t) in timing if t > 0])
    n_control = length(never_treated)

    if control_group == :never_treated && n_control == 0
        throw(ArgumentError("No never-treated units found. Use control_group=:not_yet_treated"))
    end

    # Event-time grid
    reference_period = -1
    event_times_all = collect(-leads:horizon)
    n_evt = length(event_times_all)

    # Build panel lookup: group_id -> time -> outcome value
    panel = Dict{Int, Dict{Int, T}}()
    for i in 1:pd.T_obs
        g = pd.group_id[i]
        t = pd.time_id[i]
        if !haskey(panel, g)
            panel[g] = Dict{Int, T}()
        end
        panel[g][t] = pd.data[i, outcome_col]
    end

    # -----------------------------------------------------------------
    # Point estimates
    # -----------------------------------------------------------------
    att, group_time_att = _dcdh_point_estimate(panel, timing, cohorts, never_treated,
                                               event_times_all, reference_period,
                                               control_group, all_times, pd.n_groups)

    # -----------------------------------------------------------------
    # Bootstrap SEs
    # -----------------------------------------------------------------
    rng = Random.MersenneTwister(1234)
    unit_ids = collect(keys(timing))
    n_units = length(unit_ids)
    boot_atts = Vector{Vector{T}}()

    for b in 1:n_boot
        # Resample unit IDs with replacement (block bootstrap)
        boot_ids = [unit_ids[rand(rng, 1:n_units)] for _ in 1:n_units]

        # Build bootstrap panel and timing
        boot_panel = Dict{Int, Dict{Int, T}}()
        boot_timing = Dict{Int, Int}()
        for (new_id, orig_id) in enumerate(boot_ids)
            boot_timing[new_id] = timing[orig_id]
            if haskey(panel, orig_id)
                boot_panel[new_id] = panel[orig_id]
            end
        end

        # Identify bootstrap cohorts and controls
        boot_cohorts = sort(unique([t for (_, t) in boot_timing if t > 0]))
        boot_never = [g for (g, t) in boot_timing if t == 0]

        # Skip replications where no cohorts or no controls exist
        isempty(boot_cohorts) && continue
        if control_group == :never_treated && isempty(boot_never)
            continue
        end

        boot_att, _ = _dcdh_point_estimate(boot_panel, boot_timing, boot_cohorts,
                                            boot_never, event_times_all, reference_period,
                                            control_group, all_times, n_units)

        push!(boot_atts, boot_att)
    end

    # Compute SEs from bootstrap distribution
    n_valid_boot = length(boot_atts)
    se = zeros(T, n_evt)
    if n_valid_boot > 1
        for ei in 1:n_evt
            boot_vals = T[boot_atts[b][ei] for b in 1:n_valid_boot]
            se[ei] = std(boot_vals)
        end
    end

    # CIs
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = att .- z .* se
    ci_upper = att .+ z .* se

    # Overall ATT: average of post-treatment coefficients
    post_mask = event_times_all .>= 0
    post_att = att[post_mask]
    post_se = se[post_mask]
    nonzero_post = post_se .> 0
    if any(nonzero_post)
        n_post = count(nonzero_post)
        overall_att = mean(post_att[nonzero_post])
        overall_se = sqrt(sum(post_se[nonzero_post] .^ 2)) / n_post
    else
        overall_att = isempty(post_att) ? zero(T) : mean(post_att)
        overall_se = zero(T)
    end

    DIDResult{T}(att, se, ci_lower, ci_upper, event_times_all, reference_period,
                 group_time_att, cohorts, overall_att, overall_se,
                 pd.T_obs, pd.n_groups, n_treated, n_control,
                 :did_multiplegt, pd.varnames[outcome_col], pd.varnames[treat_col],
                 control_group, cluster, T(conf_level))
end
