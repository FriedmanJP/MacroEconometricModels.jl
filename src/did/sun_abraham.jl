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
Sun & Abraham (2021) interaction-weighted estimator for staggered DiD.

Estimates cohort-specific effects via per-cohort TWFE regressions (each cohort
vs control group) with dummies for ALL event times, then aggregates to
event-time ATTs with cohort-size weights. Reports the requested window.

# Reference
Sun, L. & Abraham, S. (2021). *JoE* 225(2), 175-199.
"""

using LinearAlgebra, Statistics, Distributions

"""
    _estimate_sun_abraham(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                          leads::Int=0, horizon::Int=5,
                          control_group::Symbol=:never_treated,
                          cluster::Symbol=:unit,
                          conf_level::Real=0.95) where {T<:AbstractFloat}

Internal Sun & Abraham (2021) interaction-weighted estimator.

# Algorithm
1. For each cohort g, subset to cohort g + controls
2. Include event-time dummies for ALL relative time periods (not just [-leads, horizon])
3. OLS with double-demeaned FE → cohort-specific β_{g,e} and SEs
4. Aggregate: ATT(e) = Σ_g w_g β_{g,e}, w_g = n_g / Σ n_g
5. Report only the requested [-leads, horizon] window

# Returns
`DIDResult{T}` with interaction-weighted event-time ATTs.

# Reference
Sun, L. & Abraham, S. (2021). *JoE* 225(2), 175-199.
"""
function _estimate_sun_abraham(pd::PanelData{T}, outcome_col::Int, treat_col::Int;
                                leads::Int=0, horizon::Int=5,
                                control_group::Symbol=:never_treated,
                                cluster::Symbol=:unit,
                                conf_level::Real=0.95) where {T<:AbstractFloat}
    timing = _extract_treatment_timing(pd, treat_col)
    all_times = sort(unique(pd.time_id))

    # Identify cohorts and control groups
    cohorts = sort(unique([t for (_, t) in timing if t > 0]))
    n_cohorts = length(cohorts)

    never_treated = Set(g for (g, t) in timing if t == 0)
    if control_group == :never_treated
        isempty(never_treated) &&
            throw(ArgumentError("No never-treated units found. Use control_group=:not_yet_treated"))
    end

    treated_groups = [g for (g, t) in timing if t > 0]
    n_treated = length(treated_groups)
    n_control = length(never_treated)

    # Reporting window
    reference_period = -1
    event_times_all = collect(-leads:horizon)
    n_evt_all = length(event_times_all)

    # Per-cohort regressions
    # Store cohort-specific betas and SEs for the REPORTING window
    beta_mat = zeros(T, n_cohorts, n_evt_all)
    se_mat = zeros(T, n_cohorts, n_evt_all)
    cohort_sizes = zeros(T, n_cohorts)

    for (ci, g_time) in enumerate(cohorts)
        cohort_units = Set(g for (g, t) in timing if t == g_time)
        cohort_sizes[ci] = T(length(cohort_units))

        # Determine control group
        if control_group == :not_yet_treated
            ctrl = Set(g for (g, t) in timing if t == 0 || t > g_time)
        else
            ctrl = never_treated
        end
        isempty(ctrl) && continue

        # Build subsample
        sub_idx = Int[]
        for i in 1:pd.T_obs
            if pd.group_id[i] in cohort_units || pd.group_id[i] in ctrl
                push!(sub_idx, i)
            end
        end
        n_sub = length(sub_idx)

        # Compute ALL possible event times for this cohort
        min_evt = minimum(all_times) - g_time
        max_evt = maximum(all_times) - g_time
        full_evt = collect(min_evt:max_evt)
        full_evt_est = filter(!=(reference_period), full_evt)
        n_full_est = length(full_evt_est)

        # Build event-time dummies for ALL relative times
        D = zeros(T, n_sub, n_full_est)
        y_sub = zeros(T, n_sub)
        gid_sub = zeros(Int, n_sub)
        tid_sub = zeros(Int, n_sub)

        for (si, idx) in enumerate(sub_idx)
            g = pd.group_id[idx]
            t = pd.time_id[idx]
            y_sub[si] = pd.data[idx, outcome_col]
            gid_sub[si] = g
            tid_sub[si] = t

            if g in cohort_units
                evt = t - g_time
                if evt != reference_period
                    ei = findfirst(==(evt), full_evt_est)
                    if ei !== nothing
                        D[si, ei] = one(T)
                    end
                end
            end
        end

        # Double-demean
        y_dm = _double_demean(y_sub, gid_sub, tid_sub)
        D_dm = _double_demean_matrix(D, gid_sub, tid_sub)

        # OLS
        DtD_inv = robust_inv(D_dm' * D_dm)
        beta_g = DtD_inv * (D_dm' * y_dm)
        resid = y_dm - D_dm * beta_g

        # Clustered VCV
        V_g = _did_vcov(D_dm, resid, gid_sub, tid_sub, cluster)

        # Extract only the reporting-window coefficients
        for (ri, e) in enumerate(event_times_all)
            if e == reference_period
                # Reference period = 0
                beta_mat[ci, ri] = zero(T)
                se_mat[ci, ri] = zero(T)
            else
                ei = findfirst(==(e), full_evt_est)
                if ei !== nothing
                    beta_mat[ci, ri] = beta_g[ei]
                    se_mat[ci, ri] = sqrt(max(V_g[ei, ei], zero(T)))
                end
            end
        end
    end

    # Cohort-size weights (only over treated cohorts)
    w_total = sum(cohort_sizes)
    w = cohort_sizes ./ w_total

    # Aggregate to event-time ATTs
    att = zeros(T, n_evt_all)
    se = zeros(T, n_evt_all)

    for ri in 1:n_evt_all
        att[ri] = sum(w .* beta_mat[:, ri])
        se[ri] = sqrt(sum(w .^ 2 .* se_mat[:, ri] .^ 2))
    end

    # Group-time ATT matrix for output
    group_time_att = copy(beta_mat)

    # Confidence intervals
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = att .- z .* se
    ci_upper = att .+ z .* se

    # Overall ATT: mean of post-treatment ATTs
    post_mask = event_times_all .>= 0
    post_att = att[post_mask]
    post_se = se[post_mask]
    n_post = count(post_mask)
    overall_att = mean(post_att)
    overall_se = sqrt(sum(post_se .^ 2)) / n_post

    DIDResult{T}(att, se, ci_lower, ci_upper, event_times_all, reference_period,
                 group_time_att, cohorts, overall_att, overall_se,
                 pd.T_obs, pd.n_groups, n_treated, n_control,
                 :sun_abraham, pd.varnames[outcome_col], pd.varnames[treat_col],
                 control_group, cluster, T(conf_level))
end
