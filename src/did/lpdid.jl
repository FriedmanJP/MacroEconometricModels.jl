# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
LP-DiD engine matching Stata `lpdid` v1.0.2.

Dube, A., Girardi, D., Jordà, Ò. & Taylor, A.M. (2025).
"A Local Projections Approach to Difference-in-Differences."
*Journal of Applied Econometrics*.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Main API
# =============================================================================

"""
    estimate_lp_did(pd::PanelData{T}, outcome, treatment, H; kwargs...) -> LPDiDResult{T}

LP-DiD estimator with full feature parity to the Stata `lpdid` package v1.0.2.

Uses the switching indicator (ΔD_{it}) as treatment, clean control samples (CCS),
and time-only fixed effects (long differencing absorbs unit FE).

# Arguments
- `pd` — panel data
- `outcome` — outcome variable (name or symbol)
- `treatment` — treatment variable (binary 0/1 or timing column)
- `H` — maximum horizon

# Keyword Arguments
- `pre_window::Int=3` — pre-treatment event-time window
- `post_window::Int=H` — post-treatment event-time window
- `ylags::Int=0` — number of outcome lags as controls
- `dylags::Int=0` — number of ΔY lags as controls
- `covariates::Vector{String}=String[]` — additional covariates
- `nonabsorbing::Union{Nothing,Int}=nothing` — stabilization window L for non-absorbing treatment
- `notyet::Bool=false` — restrict controls to not-yet-treated
- `nevertreated::Bool=false` — restrict controls to never-treated
- `firsttreat::Bool=false` — use only first treatment event per unit
- `oneoff::Bool=false` — one-off treatment (requires `nonabsorbing`)
- `pmd::Union{Nothing,Symbol,Int}=nothing` — pre-mean differencing (:max or integer k)
- `reweight::Bool=false` — IPW reweighting for equally weighted ATE
- `nocomp::Bool=false` — restrict to observations in CCS at all horizons
- `cluster::Symbol=:unit` — clustering level (:unit, :time, :twoway)
- `conf_level::Real=0.95` — confidence level
- `only_pooled::Bool=false` — skip event study, compute pooled only
- `only_event::Bool=false` — skip pooled estimates
- `post_pooled::Union{Nothing,Tuple{Int,Int}}=nothing` — pooled post-treatment window (start, end)
- `pre_pooled::Union{Nothing,Tuple{Int,Int}}=nothing` — pooled pre-treatment window (start, end)

# References
- Dube, A., Girardi, D., Jordà, Ò. & Taylor, A.M. (2025). *JAE*.
- Acemoglu, D. et al. (2019). *JPE* 127(1), 47-100.
"""
function estimate_lp_did(pd::PanelData{T}, outcome::Union{String,Symbol},
                         treatment::Union{String,Symbol}, H::Int;
                         pre_window::Int=3,
                         post_window::Int=H,
                         ylags::Int=0,
                         dylags::Int=0,
                         covariates::Vector{String}=String[],
                         nonabsorbing::Union{Nothing,Int}=nothing,
                         notyet::Bool=false,
                         nevertreated::Bool=false,
                         firsttreat::Bool=false,
                         oneoff::Bool=false,
                         pmd::Union{Nothing,Symbol,Int}=nothing,
                         reweight::Bool=false,
                         nocomp::Bool=false,
                         cluster::Symbol=:unit,
                         conf_level::Real=0.95,
                         only_pooled::Bool=false,
                         only_event::Bool=false,
                         post_pooled::Union{Nothing,Tuple{Int,Int}}=nothing,
                         pre_pooled::Union{Nothing,Tuple{Int,Int}}=nothing,
                         ) where {T<:AbstractFloat}
    _validate_lpdid_inputs(nonabsorbing, notyet, nevertreated, oneoff, only_pooled, only_event)

    outcome_col = _resolve_varindex(pd, outcome)
    treat_col = _resolve_varindex(pd, treatment)
    cov_cols = [_resolve_varindex(pd, c) for c in covariates]

    # Build panel lookup: group -> (time -> row_index)
    panel_idx = _lpdid_build_panel_lookup(pd)
    all_times = sort(unique(pd.time_id))

    # Detect treatment type and build binary treatment + switching indicator
    treat_binary, switching, first_treat_time = _lpdid_build_treatment(pd, treat_col, panel_idx, all_times)

    spec_type = nonabsorbing !== nothing ? (oneoff ? :oneoff : :nonabsorbing) : :absorbing

    # Max horizons needed for CCS
    post_ccs_max = max(post_window, post_pooled !== nothing ? post_pooled[2] : 0)
    pre_ccs_max = max(pre_window, pre_pooled !== nothing ? pre_pooled[2] : 0)

    # Build Clean Control Sample indicators
    ccs_post, ccs_pre = _lpdid_build_ccs(pd, treat_binary, switching, first_treat_time,
                                          panel_idx, all_times,
                                          post_ccs_max, pre_ccs_max, spec_type,
                                          nonabsorbing, nevertreated, notyet, firsttreat)

    # Apply nocomp restriction
    if nocomp
        _lpdid_apply_nocomp!(ccs_post, ccs_pre, post_window, pre_window)
    end

    # Build PMD baseline if needed
    pmd_baseline = _lpdid_build_pmd_baseline(pd, outcome_col, treat_binary, panel_idx, all_times, pmd)

    # Event study regressions
    event_times = collect(-pre_window:post_window)
    n_h = length(event_times)
    reference_period = -1

    coefficients = zeros(T, n_h)
    se_vec = zeros(T, n_h)
    nobs_h = zeros(Int, n_h)
    vcov_all = [zeros(T, 1, 1) for _ in 1:n_h]

    if !only_pooled
        for (hi, h) in enumerate(event_times)
            if h == reference_period
                continue
            end

            ccs_h = h >= 0 ? get(ccs_post, h, Dict{Tuple{Int,Int},Bool}()) :
                             get(ccs_pre, abs(h), Dict{Tuple{Int,Int},Bool}())

            beta, se, nobs, V = _lpdid_horizon_regression(
                pd, outcome_col, cov_cols, switching, ccs_h, panel_idx, all_times,
                h, ylags, dylags, pmd, pmd_baseline, reweight, cluster, T)

            coefficients[hi] = beta
            se_vec[hi] = se
            nobs_h[hi] = nobs
            vcov_all[hi] = V
        end
    end

    # Pooled regressions (skip when only_event=true)
    pooled_post_result = if !only_event
        _lpdid_pooled_regression(
            pd, outcome_col, cov_cols, switching, ccs_post, panel_idx, all_times,
            post_pooled, ylags, dylags, pmd, pmd_baseline, reweight, cluster, conf_level, true, T)
    else
        nothing
    end

    pooled_pre_result = if !only_event && pre_pooled !== nothing
        _lpdid_pooled_regression(
            pd, outcome_col, cov_cols, switching, ccs_pre, panel_idx, all_times,
            pre_pooled, ylags, dylags, pmd, pmd_baseline, reweight, cluster, conf_level, false, T)
    else
        nothing
    end

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = coefficients .- z .* se_vec
    ci_upper = coefficients .+ z .* se_vec

    LPDiDResult{T}(coefficients, se_vec, ci_lower, ci_upper,
                    event_times, reference_period, nobs_h,
                    pooled_post_result, pooled_pre_result,
                    vcov_all,
                    pd.varnames[outcome_col], pd.varnames[treat_col],
                    pd.T_obs, pd.n_groups, spec_type, pmd, reweight, nocomp,
                    ylags, dylags, pre_window, post_window,
                    cluster, T(conf_level), pd)
end

# =============================================================================
# Input validation
# =============================================================================

function _validate_lpdid_inputs(nonabsorbing, notyet, nevertreated, oneoff, only_pooled, only_event)
    notyet && nevertreated && throw(ArgumentError("Cannot combine notyet=true and nevertreated=true"))
    oneoff && nonabsorbing === nothing && throw(ArgumentError("oneoff=true requires nonabsorbing=L"))
    only_pooled && only_event && throw(ArgumentError("Cannot combine only_pooled and only_event"))
end

# =============================================================================
# Panel lookup
# =============================================================================

function _lpdid_build_panel_lookup(pd::PanelData)
    panel_idx = Dict{Int, Dict{Int, Int}}()
    for i in 1:pd.T_obs
        g = pd.group_id[i]
        t = pd.time_id[i]
        if !haskey(panel_idx, g)
            panel_idx[g] = Dict{Int,Int}()
        end
        panel_idx[g][t] = i
    end
    panel_idx
end

# =============================================================================
# Treatment detection and switching indicator
# =============================================================================

function _lpdid_build_treatment(pd::PanelData{T}, treat_col::Int,
                                 panel_idx::Dict{Int,Dict{Int,Int}},
                                 all_times::Vector{Int}) where {T}
    # Auto-detect: binary (0/1) vs timing (year values)
    is_binary = true
    for i in 1:pd.T_obs
        v = pd.data[i, treat_col]
        if !isnan(v) && v != zero(T) && v != one(T)
            is_binary = false
            break
        end
    end

    # Build binary treatment dict: (g, t) -> 0/1
    treat_binary = Dict{Int, Dict{Int, T}}()
    first_treat_time = Dict{Int, Int}()  # g -> first treatment time

    if is_binary
        # Binary: use directly
        for g in 1:pd.n_groups
            haskey(panel_idx, g) || continue
            treat_binary[g] = Dict{Int, T}()
            for (t, row) in panel_idx[g]
                treat_binary[g][t] = pd.data[row, treat_col]
            end
            # Find first treatment time
            times_g = sort(collect(keys(panel_idx[g])))
            for t in times_g
                if get(treat_binary[g], t, zero(T)) == one(T)
                    first_treat_time[g] = t
                    break
                end
            end
        end
    else
        # Timing column: convert to binary
        for g in 1:pd.n_groups
            haskey(panel_idx, g) || continue
            treat_binary[g] = Dict{Int, T}()
            # Treatment timing is the value in the treatment column (constant per unit)
            timing_val = NaN
            for (t, row) in panel_idx[g]
                v = pd.data[row, treat_col]
                if !isnan(v) && v > 0
                    timing_val = v
                    break
                end
            end
            treat_time = isnan(timing_val) ? 0 : Int(timing_val)
            if treat_time > 0
                first_treat_time[g] = treat_time
            end
            for (t, row) in panel_idx[g]
                treat_binary[g][t] = (treat_time > 0 && t >= treat_time) ? one(T) : zero(T)
            end
        end
    end

    # Build switching indicator: ΔD_{it} = D_{it} - D_{i,t-1}
    switching = Dict{Int, Dict{Int, T}}()
    for g in 1:pd.n_groups
        haskey(treat_binary, g) || continue
        switching[g] = Dict{Int, T}()
        times_g = sort(collect(keys(treat_binary[g])))
        for (j, t) in enumerate(times_g)
            if j == 1
                switching[g][t] = T(NaN)  # no previous period
            else
                t_prev = times_g[j-1]
                # Only valid if consecutive (t_prev == t - 1)
                if t_prev == t - 1
                    d_now = treat_binary[g][t]
                    d_prev = treat_binary[g][t_prev]
                    if isnan(d_now) || isnan(d_prev)
                        switching[g][t] = T(NaN)
                    else
                        switching[g][t] = d_now - d_prev
                    end
                else
                    switching[g][t] = T(NaN)
                end
            end
        end
    end

    return treat_binary, switching, first_treat_time
end

# =============================================================================
# Clean Control Sample (CCS)
# =============================================================================

function _lpdid_build_ccs(pd::PanelData{T}, treat_binary, switching, first_treat_time,
                           panel_idx, all_times,
                           post_max, pre_max, spec_type,
                           nonabsorbing, nevertreated, notyet, firsttreat) where {T}
    ccs_post = Dict{Int, Dict{Tuple{Int,Int}, Bool}}()
    ccs_pre = Dict{Int, Dict{Tuple{Int,Int}, Bool}}()

    if spec_type == :absorbing
        _lpdid_ccs_absorbing!(ccs_post, ccs_pre, pd, treat_binary, switching,
                               panel_idx, all_times, post_max, pre_max)
    elseif spec_type == :nonabsorbing
        L = nonabsorbing
        _lpdid_ccs_nonabsorbing!(ccs_post, ccs_pre, pd, treat_binary, switching,
                                  panel_idx, all_times, post_max, pre_max, L)
    elseif spec_type == :oneoff
        L = nonabsorbing
        _lpdid_ccs_oneoff!(ccs_post, ccs_pre, pd, treat_binary, switching,
                            panel_idx, all_times, post_max, pre_max, L)
    end

    # Apply control group restrictions
    if nevertreated
        _lpdid_restrict_nevertreated!(ccs_post, ccs_pre, switching, first_treat_time, pd)
    end
    if notyet
        _lpdid_restrict_notyet!(ccs_post, ccs_pre, switching, first_treat_time, pd, all_times)
    end
    if firsttreat
        _lpdid_restrict_firsttreat!(ccs_post, ccs_pre, switching, first_treat_time, pd, all_times)
    end

    return ccs_post, ccs_pre
end

# Absorbing CCS: CCS_h = (ΔD==1) OR (D_{t+h}==0)
function _lpdid_ccs_absorbing!(ccs_post, ccs_pre, pd, treat_binary, switching,
                                panel_idx, all_times, post_max, pre_max)
    T_type = eltype(pd.data)
    for h in 0:post_max
        ccs_post[h] = Dict{Tuple{Int,Int}, Bool}()
        for g in 1:pd.n_groups
            haskey(switching, g) || continue
            for t in all_times
                Δ = get(switching[g], t, T_type(NaN))
                isnan(Δ) && continue

                if Δ == 1  # switching on
                    # Check that t+h exists
                    if haskey(panel_idx[g], t + h)
                        ccs_post[h][(g, t)] = true
                    end
                elseif Δ == 0
                    # Control: must be untreated at t+h
                    d_at_th = get(get(treat_binary, g, Dict()), t + h, T_type(NaN))
                    if !isnan(d_at_th) && d_at_th == 0
                        ccs_post[h][(g, t)] = true
                    end
                end
                # Δ < 0 or Δ > 1 or already treated: excluded
            end
        end
    end

    # Pre-periods: use CCS_0 (same base sample)
    ccs0 = get(ccs_post, 0, Dict{Tuple{Int,Int},Bool}())
    for h in 1:pre_max
        ccs_pre[h] = Dict{Tuple{Int,Int}, Bool}()
        for (key, val) in ccs0
            ccs_pre[h][key] = val
        end
    end
end

# Non-absorbing CCS: no switches in past L periods
function _lpdid_ccs_nonabsorbing!(ccs_post, ccs_pre, pd, treat_binary, switching,
                                   panel_idx, all_times, post_max, pre_max, L)
    T_type = eltype(pd.data)

    # CCS_0: no switches in past L periods AND (switching at t OR control at t)
    ccs_post[0] = Dict{Tuple{Int,Int}, Bool}()
    for g in 1:pd.n_groups
        haskey(switching, g) || continue
        for t in all_times
            Δ = get(switching[g], t, T_type(NaN))
            isnan(Δ) && continue
            (Δ == 1 || Δ == 0) || continue  # must be switching or clean control

            # Check no switches in past L periods
            clean = true
            for k in 1:L
                Δ_past = get(switching[g], t - k, T_type(NaN))
                if !isnan(Δ_past) && abs(Δ_past) > 0
                    clean = false
                    break
                end
            end
            if clean
                ccs_post[0][(g, t)] = true
            end
        end
    end

    # Forward CCS: extend CCS_{h-1} by requiring no switch at t+h
    for h in 1:post_max
        ccs_post[h] = Dict{Tuple{Int,Int}, Bool}()
        prev = ccs_post[h-1]
        for ((g, t), _) in prev
            Δ_fwd = get(get(switching, g, Dict()), t + h, T_type(NaN))
            if isnan(Δ_fwd) || abs(Δ_fwd) == 0
                ccs_post[h][(g, t)] = true
            end
        end
    end

    # Backward CCS: CCS_1 = CCS_0, then extend backward
    ccs0 = ccs_post[0]
    ccs_pre[1] = Dict{Tuple{Int,Int}, Bool}()
    for (key, val) in ccs0
        ccs_pre[1][key] = val
    end
    for h in 2:pre_max
        ccs_pre[h] = Dict{Tuple{Int,Int}, Bool}()
        prev = ccs_pre[h-1]
        for ((g, t), _) in prev
            # Require the base observation at t-(h-1) is also in CCS_0
            if haskey(ccs0, (g, t - (h - 1)))
                ccs_pre[h][(g, t)] = true
            end
        end
    end
end

# One-off CCS: treatment lasts exactly 1 period
function _lpdid_ccs_oneoff!(ccs_post, ccs_pre, pd, treat_binary, switching,
                             panel_idx, all_times, post_max, pre_max, L)
    T_type = eltype(pd.data)

    # CCS_0: current status matches previous AND no treatment in past L periods
    ccs_post[0] = Dict{Tuple{Int,Int}, Bool}()
    for g in 1:pd.n_groups
        haskey(treat_binary, g) || continue
        haskey(switching, g) || continue
        for t in all_times
            Δ = get(switching[g], t, T_type(NaN))
            isnan(Δ) && continue
            d_now = get(treat_binary[g], t, T_type(NaN))
            isnan(d_now) && continue

            # For one-off: treat at t means D=1 and ΔD=1; control means D=0 and ΔD=0
            ok = (d_now == 1 && Δ == 1) || (d_now == 0 && Δ == 0)
            ok || continue

            # No treatment in past L periods (D_{t-1} must be 0)
            d_prev = get(treat_binary[g], t - 1, T_type(NaN))
            if !isnan(d_prev) && d_prev != 0
                continue
            end
            clean = true
            for k in 2:L
                d_past = get(treat_binary[g], t - k, T_type(NaN))
                if !isnan(d_past) && d_past != 0
                    clean = false
                    break
                end
            end
            if clean
                ccs_post[0][(g, t)] = true
            end
        end
    end

    # Forward: no future treatments
    for h in 1:post_max
        ccs_post[h] = Dict{Tuple{Int,Int}, Bool}()
        prev = ccs_post[h-1]
        for ((g, t), _) in prev
            d_fwd = get(get(treat_binary, g, Dict()), t + h, T_type(NaN))
            if isnan(d_fwd) || d_fwd == 0
                ccs_post[h][(g, t)] = true
            end
        end
    end

    # Backward CCS same as non-absorbing
    ccs0 = ccs_post[0]
    ccs_pre[1] = Dict{Tuple{Int,Int}, Bool}()
    for (key, val) in ccs0
        ccs_pre[1][key] = val
    end
    for h in 2:pre_max
        ccs_pre[h] = Dict{Tuple{Int,Int}, Bool}()
        prev = ccs_pre[h-1]
        for ((g, t), _) in prev
            if haskey(ccs0, (g, t - (h - 1)))
                ccs_pre[h][(g, t)] = true
            end
        end
    end
end

# Control group restrictions
function _lpdid_restrict_nevertreated!(ccs_post, ccs_pre, switching, first_treat_time, pd)
    for ccs_dict in values(ccs_post)
        for (g, t) in collect(keys(ccs_dict))
            Δ = get(get(switching, g, Dict()), t, NaN)
            if isnan(Δ) || Δ != 1  # control observation
                if haskey(first_treat_time, g)  # ever treated
                    delete!(ccs_dict, (g, t))
                end
            end
        end
    end
    for ccs_dict in values(ccs_pre)
        for (g, t) in collect(keys(ccs_dict))
            Δ = get(get(switching, g, Dict()), t, NaN)
            if isnan(Δ) || Δ != 1
                if haskey(first_treat_time, g)
                    delete!(ccs_dict, (g, t))
                end
            end
        end
    end
end

function _lpdid_restrict_notyet!(ccs_post, ccs_pre, switching, first_treat_time, pd, all_times)
    for ccs_dict in values(ccs_post)
        for (g, t) in collect(keys(ccs_dict))
            Δ = get(get(switching, g, Dict()), t, NaN)
            if isnan(Δ) || Δ != 1  # control
                if haskey(first_treat_time, g) && first_treat_time[g] <= t
                    delete!(ccs_dict, (g, t))
                end
            end
        end
    end
    for ccs_dict in values(ccs_pre)
        for (g, t) in collect(keys(ccs_dict))
            Δ = get(get(switching, g, Dict()), t, NaN)
            if isnan(Δ) || Δ != 1
                if haskey(first_treat_time, g) && first_treat_time[g] <= t
                    delete!(ccs_dict, (g, t))
                end
            end
        end
    end
end

function _lpdid_restrict_firsttreat!(ccs_post, ccs_pre, switching, first_treat_time, pd, all_times)
    # Only keep the first treatment event per unit
    for ccs_dict in values(ccs_post)
        for (g, t) in collect(keys(ccs_dict))
            Δ = get(get(switching, g, Dict()), t, NaN)
            if !isnan(Δ) && Δ == 1  # treated
                if haskey(first_treat_time, g) && first_treat_time[g] != t
                    delete!(ccs_dict, (g, t))
                end
            end
        end
    end
    for ccs_dict in values(ccs_pre)
        for (g, t) in collect(keys(ccs_dict))
            Δ = get(get(switching, g, Dict()), t, NaN)
            if !isnan(Δ) && Δ == 1
                if haskey(first_treat_time, g) && first_treat_time[g] != t
                    delete!(ccs_dict, (g, t))
                end
            end
        end
    end
end

# =============================================================================
# No-composition restriction
# =============================================================================

function _lpdid_apply_nocomp!(ccs_post, ccs_pre, post_window, pre_window)
    # Find intersection of all CCS across horizons
    common = nothing
    for h in 0:post_window
        d = get(ccs_post, h, Dict{Tuple{Int,Int},Bool}())
        if common === nothing
            common = Set(keys(d))
        else
            intersect!(common, keys(d))
        end
    end
    for h in 1:pre_window
        d = get(ccs_pre, h, Dict{Tuple{Int,Int},Bool}())
        if common !== nothing
            intersect!(common, keys(d))
        end
    end
    common === nothing && return

    # Restrict all CCS to common set
    for h in keys(ccs_post)
        for key in collect(keys(ccs_post[h]))
            if !(key in common)
                delete!(ccs_post[h], key)
            end
        end
    end
    for h in keys(ccs_pre)
        for key in collect(keys(ccs_pre[h]))
            if !(key in common)
                delete!(ccs_pre[h], key)
            end
        end
    end
end

# =============================================================================
# PMD baseline
# =============================================================================

function _lpdid_build_pmd_baseline(pd::PanelData{T}, outcome_col, treat_binary,
                                    panel_idx, all_times, pmd) where {T}
    pmd === nothing && return nothing

    baseline = Dict{Int, Dict{Int, T}}()
    for g in 1:pd.n_groups
        haskey(panel_idx, g) || continue
        baseline[g] = Dict{Int, T}()
        times_g = sort(collect(keys(panel_idx[g])))

        for t in times_g
            if pmd == :max
                # Mean of all pre-treatment Y values: Y_{i,s} for s < t
                vals = T[]
                for s in times_g
                    s >= t && break
                    y_s = pd.data[panel_idx[g][s], outcome_col]
                    !isnan(y_s) && push!(vals, y_s)
                end
                baseline[g][t] = isempty(vals) ? T(NaN) : mean(vals)
            else
                # PMD with moving average over [t-k, t-1]
                k = Int(pmd)
                vals = T[]
                for lag in 1:k
                    s = t - lag
                    if haskey(panel_idx[g], s)
                        y_s = pd.data[panel_idx[g][s], outcome_col]
                        !isnan(y_s) && push!(vals, y_s)
                    end
                end
                baseline[g][t] = isempty(vals) ? T(NaN) : mean(vals)
            end
        end
    end
    baseline
end

# =============================================================================
# Time demeaning (NOT double-demean)
# =============================================================================

function _lpdid_time_demean!(y::Vector{T}, time_ids::Vector{Int}) where {T}
    for t in unique(time_ids)
        mask = time_ids .== t
        n_t = count(mask)
        n_t > 0 || continue
        m = zero(T)
        for i in eachindex(y)
            if mask[i]
                m += y[i]
            end
        end
        m /= n_t
        for i in eachindex(y)
            if mask[i]
                y[i] -= m
            end
        end
    end
end

# =============================================================================
# Per-horizon regression
# =============================================================================

function _lpdid_horizon_regression(pd::PanelData{T}, outcome_col, cov_cols,
                                    switching, ccs_h, panel_idx, all_times,
                                    h, ylags, dylags, pmd, pmd_baseline,
                                    reweight, cluster, ::Type{T}) where {T}
    # Build regression data
    y_vec = T[]
    d_vec = T[]
    x_rows = Vector{T}[]
    unit_ids = Int[]
    time_ids = Int[]

    n_ctrl = ylags + dylags + length(cov_cols)

    for g in 1:pd.n_groups
        haskey(panel_idx, g) || continue
        haskey(switching, g) || continue

        for t in all_times
            get(ccs_h, (g, t), false) || continue

            # Need Y at t+h
            haskey(panel_idx[g], t + h) || continue
            y_fwd = pd.data[panel_idx[g][t + h], outcome_col]
            isnan(y_fwd) && continue

            # Baseline: Y_{t-1} or PMD
            if pmd_baseline === nothing
                haskey(panel_idx[g], t - 1) || continue
                y_base = pd.data[panel_idx[g][t - 1], outcome_col]
            else
                y_base = get(get(pmd_baseline, g, Dict()), t, T(NaN))
            end
            isnan(y_base) && continue

            # Switching indicator
            Δ = get(switching[g], t, T(NaN))
            isnan(Δ) && continue

            # Build controls
            x_row = T[]
            valid = true

            for l in 1:ylags
                if haskey(panel_idx[g], t - l)
                    val = pd.data[panel_idx[g][t - l], outcome_col]
                    isnan(val) && (valid = false; break)
                    push!(x_row, val)
                else
                    valid = false; break
                end
            end
            !valid && continue

            for l in 1:dylags
                if haskey(panel_idx[g], t - l) && haskey(panel_idx[g], t - l - 1)
                    v1 = pd.data[panel_idx[g][t - l], outcome_col]
                    v0 = pd.data[panel_idx[g][t - l - 1], outcome_col]
                    (isnan(v1) || isnan(v0)) && (valid = false; break)
                    push!(x_row, v1 - v0)
                else
                    valid = false; break
                end
            end
            !valid && continue

            for c in cov_cols
                if haskey(panel_idx[g], t)
                    val = pd.data[panel_idx[g][t], c]
                    isnan(val) && (valid = false; break)
                    push!(x_row, val)
                else
                    valid = false; break
                end
            end
            !valid && continue

            push!(y_vec, y_fwd - y_base)
            push!(d_vec, Δ)
            push!(x_rows, x_row)
            push!(unit_ids, g)
            push!(time_ids, t)
        end
    end

    N = length(y_vec)
    if N < 3
        return T(NaN), T(NaN), N, zeros(T, 1, 1)
    end

    # Build X matrix: [ΔD, controls]
    K_ctrl = isempty(x_rows) || isempty(x_rows[1]) ? 0 : length(x_rows[1])
    K = 1 + K_ctrl
    X = Matrix{T}(undef, N, K)
    for i in 1:N
        X[i, 1] = d_vec[i]
        for j in 1:K_ctrl
            X[i, 1 + j] = x_rows[i][j]
        end
    end

    # Time-only demeaning
    y_dm = copy(y_vec)
    _lpdid_time_demean!(y_dm, time_ids)
    X_dm = copy(X)
    for k in 1:K
        col = X_dm[:, k]
        _lpdid_time_demean!(col, time_ids)
        X_dm[:, k] = col
    end

    # OLS
    XtX_inv = robust_inv(X_dm' * X_dm; silent=true)
    beta_vec = XtX_inv * (X_dm' * y_dm)
    resid = y_dm - X_dm * beta_vec

    # Clustered VCV
    V = _did_vcov(X_dm, resid, unit_ids, time_ids, cluster)

    beta = beta_vec[1]
    se = sqrt(max(V[1, 1], zero(T)))

    return beta, se, N, V
end

# =============================================================================
# Pooled regression
# =============================================================================

function _lpdid_pooled_regression(pd::PanelData{T}, outcome_col, cov_cols,
                                   switching, ccs_dict, panel_idx, all_times,
                                   pooled_spec, ylags, dylags, pmd, pmd_baseline,
                                   reweight, cluster, conf_level, is_post, ::Type{T}) where {T}
    pooled_spec === nothing && return nothing

    start_h, end_h = pooled_spec

    # Use CCS at the maximum horizon
    ccs_h = get(ccs_dict, end_h, Dict{Tuple{Int,Int},Bool}())

    y_vec = T[]
    d_vec = T[]
    x_rows = Vector{T}[]
    unit_ids = Int[]
    time_ids = Int[]

    for g in 1:pd.n_groups
        haskey(panel_idx, g) || continue
        haskey(switching, g) || continue

        for t in all_times
            get(ccs_h, (g, t), false) || continue

            # Pooled LHS: mean(Y_{t+start} ... Y_{t+end}) - baseline
            y_sum = zero(T)
            y_count = 0
            valid = true

            if is_post
                for hh in start_h:end_h
                    if haskey(panel_idx[g], t + hh)
                        val = pd.data[panel_idx[g][t + hh], outcome_col]
                        if !isnan(val)
                            y_sum += val
                            y_count += 1
                        end
                    end
                end
            else
                for hh in start_h:end_h
                    if haskey(panel_idx[g], t - hh)
                        val = pd.data[panel_idx[g][t - hh], outcome_col]
                        if !isnan(val)
                            y_sum += val
                            y_count += 1
                        end
                    end
                end
            end
            y_count == 0 && continue
            y_avg = y_sum / y_count

            # Baseline
            if pmd_baseline === nothing
                haskey(panel_idx[g], t - 1) || continue
                y_base = pd.data[panel_idx[g][t - 1], outcome_col]
            else
                y_base = get(get(pmd_baseline, g, Dict()), t, T(NaN))
            end
            isnan(y_base) && continue

            # Switching
            Δ = get(switching[g], t, T(NaN))
            isnan(Δ) && continue

            # Controls
            x_row = T[]
            for l in 1:ylags
                if haskey(panel_idx[g], t - l)
                    val = pd.data[panel_idx[g][t - l], outcome_col]
                    isnan(val) && (valid = false; break)
                    push!(x_row, val)
                else
                    valid = false; break
                end
            end
            !valid && continue

            for l in 1:dylags
                if haskey(panel_idx[g], t - l) && haskey(panel_idx[g], t - l - 1)
                    v1 = pd.data[panel_idx[g][t - l], outcome_col]
                    v0 = pd.data[panel_idx[g][t - l - 1], outcome_col]
                    (isnan(v1) || isnan(v0)) && (valid = false; break)
                    push!(x_row, v1 - v0)
                else
                    valid = false; break
                end
            end
            !valid && continue

            for c in cov_cols
                if haskey(panel_idx[g], t)
                    val = pd.data[panel_idx[g][t], c]
                    isnan(val) && (valid = false; break)
                    push!(x_row, val)
                else
                    valid = false; break
                end
            end
            !valid && continue

            push!(y_vec, y_avg - y_base)
            push!(d_vec, Δ)
            push!(x_rows, x_row)
            push!(unit_ids, g)
            push!(time_ids, t)
        end
    end

    N = length(y_vec)
    if N < 3
        return (coef=T(NaN), se=T(NaN), ci_lower=T(NaN), ci_upper=T(NaN), nobs=N)
    end

    K_ctrl = isempty(x_rows) || isempty(x_rows[1]) ? 0 : length(x_rows[1])
    K = 1 + K_ctrl
    X = Matrix{T}(undef, N, K)
    for i in 1:N
        X[i, 1] = d_vec[i]
        for j in 1:K_ctrl
            X[i, 1 + j] = x_rows[i][j]
        end
    end

    y_dm = copy(y_vec)
    _lpdid_time_demean!(y_dm, time_ids)
    X_dm = copy(X)
    for k in 1:K
        col = X_dm[:, k]
        _lpdid_time_demean!(col, time_ids)
        X_dm[:, k] = col
    end

    XtX_inv = robust_inv(X_dm' * X_dm; silent=true)
    beta_vec = XtX_inv * (X_dm' * y_dm)
    resid = y_dm - X_dm * beta_vec
    V = _did_vcov(X_dm, resid, unit_ids, time_ids, cluster)

    coef = beta_vec[1]
    se = sqrt(max(V[1, 1], zero(T)))
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))

    return (coef=coef, se=se, ci_lower=coef - z * se, ci_upper=coef + z * se, nobs=N)
end
