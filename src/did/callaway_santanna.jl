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

    # Compute group-time ATTs + per-cell unit influence-function contributions.
    # ATT(g,t) = mean(ΔY_treated) − mean(ΔY_control) is a difference in means, whose exact
    # (unit-clustered) variance comes from per-unit scores c_gt(i): a treated unit i∈g
    # contributes (ΔY_i − mean_g)/n_g and a control unit i∈C contributes −(ΔY_i − mean_c)/n_c,
    # with Σ_i c_gt(i) = 0 by construction. Cross-(g,t) covariance is non-zero because cells
    # SHARE the same control units (and, within a cohort, the same treated units), so the
    # diagonal-only se_agg = sqrt(Σ w²se²) understates event-time and overall SEs.
    group_time_att = fill(T(NaN), n_cohorts, n_times)
    group_time_se = fill(T(NaN), n_cohorts, n_times)
    n_units = pd.n_groups
    # (ci,ti) -> (unit_ids, per-unit influence contributions c_gt(i))
    cell_infl = Dict{Tuple{Int,Int}, Tuple{Vector{Int}, Vector{T}}}()

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

            # Compute ATT(g,t) = mean(DeltaY_treated) - mean(DeltaY_control), DeltaY = Y_t - Y_base,
            # tracking unit ids so per-unit influence scores can be formed.
            t_units = Int[]; dy_treated = T[]
            for u in cohort_units
                if haskey(panel[u], t) && haskey(panel[u], base_t)
                    push!(t_units, u); push!(dy_treated, panel[u][t] - panel[u][base_t])
                end
            end
            isempty(dy_treated) && continue

            c_units = Int[]; dy_control = T[]
            for u in ctrl_units
                if haskey(panel[u], t) && haskey(panel[u], base_t)
                    push!(c_units, u); push!(dy_control, panel[u][t] - panel[u][base_t])
                end
            end
            isempty(dy_control) && continue

            mean_t = mean(dy_treated); mean_c = mean(dy_control)
            att_gt = mean_t - mean_c
            n_gt = length(dy_treated); n_ct = length(dy_control)

            # Per-unit influence scores (ATT̂ − ATT = Σ_i c_gt(i)); treated ∪ control are disjoint.
            infl_units = Vector{Int}(undef, n_gt + n_ct)
            infl_vals = Vector{T}(undef, n_gt + n_ct)
            @inbounds for j in 1:n_gt
                infl_units[j] = t_units[j]
                infl_vals[j] = (dy_treated[j] - mean_t) / n_gt
            end
            @inbounds for j in 1:n_ct
                infl_units[n_gt + j] = c_units[j]
                infl_vals[n_gt + j] = -(dy_control[j] - mean_c) / n_ct
            end

            group_time_att[ci, ti] = att_gt
            group_time_se[ci, ti] = sqrt(max(sum(abs2, infl_vals), zero(T)))
            cell_infl[(ci, ti)] = (infl_units, infl_vals)
        end
    end

    # Aggregate to event-time, accumulating the weighted per-unit influence into Φ.
    reference_period = -1
    event_times_all = collect(-leads:horizon)
    n_evt = length(event_times_all)
    att_agg = zeros(T, n_evt)
    Phi = zeros(T, n_units, n_evt)   # Φ[i,ei] = per-unit influence of the event-time-e ATT

    for (ei, e) in enumerate(event_times_all)
        # With universal base, reference period e=-1 is zero by construction
        if base_period == :universal && e == reference_period
            att_agg[ei] = zero(T)
            continue                                     # Φ column stays 0 ⇒ se = 0
        end

        # Aggregate across cohorts for event-time e (cohort-size weights, unchanged point est.)
        att_vals = T[]
        weights_e = T[]
        cells_e = Tuple{Int,Int}[]

        for (ci, g_time) in enumerate(cohorts)
            t_target = g_time + e
            ti = findfirst(==(t_target), all_times)
            ti === nothing && continue
            isnan(group_time_att[ci, ti]) && continue

            n_g = T(count(g -> timing[g] == g_time, keys(timing)))
            push!(att_vals, group_time_att[ci, ti])
            push!(weights_e, n_g)
            push!(cells_e, (ci, ti))
        end

        if !isempty(att_vals)
            w_norm = weights_e ./ sum(weights_e)
            att_agg[ei] = sum(w_norm .* att_vals)
            @inbounds for (m, cell) in enumerate(cells_e)
                us, vs = cell_infl[cell]
                a = w_norm[m]
                for (u, v) in zip(us, vs)
                    Phi[u, ei] += a * v
                end
            end
        end
    end

    # Event-time ATT covariance V_evt = Φ'Φ (unit-clustered by construction). Coarser
    # clustering is non-standard for CS (cross-sectional asymptotics) → warn, use unit level.
    if cluster != :unit
        @warn "Callaway-Sant'Anna variance uses unit-level (cross-sectional) influence-function " *
              "clustering; cluster=:$cluster is treated as unit-level." maxlog=1
    end
    att_vcov = Phi' * Phi
    att_vcov = (att_vcov .+ att_vcov') ./ 2               # symmetrize FP asymmetry
    se_agg = sqrt.(max.(diag(att_vcov), zero(T)))

    # CIs
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = att_agg .- z .* se_agg
    ci_upper = att_agg .+ z .* se_agg

    # Overall ATT: equal-weighted mean of post ATTs with nonzero SE; SE via the full
    # covariance sub-block sqrt(w'V_post w), w = 1/n_post (T068).
    post_idx = findall(>=(0), event_times_all)
    valid_post = post_idx[se_agg[post_idx] .> 0]
    if !isempty(valid_post)
        n_post = length(valid_post)
        overall_att = mean(att_agg[valid_post])
        w = fill(one(T) / n_post, n_post)
        Vp = att_vcov[valid_post, valid_post]
        overall_se = sqrt(max(dot(w, Vp * w), zero(T)))
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
                 control_group, cluster, T(conf_level), att_vcov)
end
