# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Event Study Local Projections with panel fixed effects.

Implements `estimate_event_study_lp`: Panel event study LP (Jordà 2005, ANRR 2019).

Uses the switching indicator ΔD_{it} and time-only FE (long differencing absorbs unit FE):

    Y_{i,t+h} - Y_{i,t-1} = γ_t^h + β_h ΔD_{it} + X_{it}'δ^h + ε_{i,t+h}

For LP-DiD with clean control samples, see `estimate_lp_did` in `lpdid.jl`.
"""

using LinearAlgebra, Statistics, Distributions

"""
    estimate_event_study_lp(pd::PanelData{T}, outcome::Union{String,Symbol},
                            treatment::Union{String,Symbol}, H::Int;
                            leads::Int=3, lags::Int=4,
                            covariates::Vector{String}=String[],
                            cluster::Symbol=:unit,
                            conf_level::Real=0.95) where {T}

Estimate event study impulse responses using local projections with panel FE.

Uses the switching indicator ΔD_{it} (= 1 only at the treatment onset period)
and time-only FE. Already-treated observations are excluded from the sample.

# Arguments
- `pd`: Panel data
- `outcome`: Outcome variable name
- `treatment`: Treatment variable (binary 0/1 or timing column)
- `H`: Maximum post-treatment horizon
- `leads`: Number of pre-treatment leads to estimate (for pre-trends)
- `lags`: Number of outcome lags as controls
- `covariates`: Additional control variable names
- `cluster`: Clustering level (:unit, :time, :twoway)
- `conf_level`: Confidence level (default: 0.95)

# Returns
`EventStudyLP{T}` with dynamic treatment effect coefficients.

# References
- Jordà, Ò. (2005). *AER* 95(1), 161-182.
- Acemoglu, D. et al. (2019). *JPE* 127(1), 47-100.
"""
function estimate_event_study_lp(pd::PanelData{T}, outcome::Union{String,Symbol},
                                 treatment::Union{String,Symbol}, H::Int;
                                 leads::Int=3, lags::Int=4,
                                 covariates::Vector{String}=String[],
                                 cluster::Symbol=:unit,
                                 conf_level::Real=0.95) where {T<:AbstractFloat}
    outcome_col = _resolve_varindex(pd, outcome)
    treat_col = _resolve_varindex(pd, treatment)
    cov_cols = [_resolve_varindex(pd, c) for c in covariates]

    # Build panel lookup
    panel_idx = Dict{Int, Dict{Int, Int}}()
    for i in 1:pd.T_obs
        g = pd.group_id[i]
        t = pd.time_id[i]
        if !haskey(panel_idx, g)
            panel_idx[g] = Dict{Int, Int}()
        end
        panel_idx[g][t] = i
    end
    all_times = sort(unique(pd.time_id))

    # Build treatment info using the LP-DiD infrastructure
    treat_binary, switching, first_treat_time = _lpdid_build_treatment(pd, treat_col, panel_idx, all_times)

    # Override timing with cohort_id if present
    if pd.cohort_id !== nothing
        # Rebuild first_treat_time from cohort_id
        empty!(first_treat_time)
        for g in 1:pd.n_groups
            mask = pd.group_id .== g
            cid = pd.cohort_id[findfirst(mask)]
            if cid > 0
                first_treat_time[g] = cid
            end
        end
    end

    # Event-time grid
    reference_period = -1
    event_times_all = collect(-leads:H)
    n_horizons = length(event_times_all)

    coefficients = zeros(T, n_horizons)
    se_vec = zeros(T, n_horizons)
    B_all = Vector{Matrix{T}}(undef, n_horizons)
    resid_all = Vector{Matrix{T}}(undef, n_horizons)
    vcov_all = Vector{Matrix{T}}(undef, n_horizons)
    T_eff_all = Vector{Int}(undef, n_horizons)

    for (hi, h) in enumerate(event_times_all)
        if h == reference_period
            coefficients[hi] = zero(T)
            se_vec[hi] = zero(T)
            B_all[hi] = zeros(T, 1, 1)
            resid_all[hi] = zeros(T, 0, 1)
            vcov_all[hi] = zeros(T, 1, 1)
            T_eff_all[hi] = 0
            continue
        end

        y_h = T[]
        D_h = T[]
        X_h_rows = Vector{Vector{T}}()
        unit_ids_h = Int[]
        time_ids_h = Int[]

        for g in 1:pd.n_groups
            haskey(panel_idx, g) || continue
            haskey(switching, g) || continue

            for t in all_times
                # Need Y at t+h and Y at t-1
                haskey(panel_idx[g], t + h) || continue
                haskey(panel_idx[g], t - 1) || continue

                # Get switching indicator
                Δ = get(switching[g], t, T(NaN))
                isnan(Δ) && continue

                # Keep only: switching obs (ΔD=1) and control obs (ΔD=0)
                # Exclude already-treated (ΔD < 0 or unit treated before t)
                if Δ != 1 && Δ != 0
                    continue
                end
                # If ΔD=0 but unit is currently treated (D=1), exclude (already treated)
                if Δ == 0
                    d_now = get(get(treat_binary, g, Dict()), t, T(NaN))
                    if !isnan(d_now) && d_now == 1
                        continue
                    end
                end

                row_h = panel_idx[g][t + h]
                row_base = panel_idx[g][t - 1]

                y_fwd = pd.data[row_h, outcome_col]
                y_base = pd.data[row_base, outcome_col]
                (isnan(y_fwd) || isnan(y_base)) && continue

                dy = y_fwd - y_base

                # Control lags: Y_{i,t-l}
                x_row = T[]
                valid_lags = true
                for l in 1:lags
                    if haskey(panel_idx[g], t - l)
                        val = pd.data[panel_idx[g][t - l], outcome_col]
                        isnan(val) && (valid_lags = false; break)
                        push!(x_row, val)
                    else
                        valid_lags = false
                        break
                    end
                end
                !valid_lags && continue

                # Covariates
                if !isempty(cov_cols)
                    row_t = get(panel_idx[g], t, nothing)
                    row_t === nothing && continue
                    for c in cov_cols
                        val = pd.data[row_t, c]
                        isnan(val) && (valid_lags = false; break)
                        push!(x_row, val)
                    end
                    !valid_lags && continue
                end

                push!(y_h, dy)
                push!(D_h, Δ)
                push!(X_h_rows, x_row)
                push!(unit_ids_h, g)
                push!(time_ids_h, t)
            end
        end

        N_h = length(y_h)
        if N_h < 3
            coefficients[hi] = T(NaN)
            se_vec[hi] = T(NaN)
            B_all[hi] = zeros(T, 1, 1)
            resid_all[hi] = zeros(T, 0, 1)
            vcov_all[hi] = zeros(T, 1, 1)
            T_eff_all[hi] = N_h
            continue
        end

        # Assemble X: [ΔD, controls]
        K_ctrl = isempty(X_h_rows) || isempty(X_h_rows[1]) ? 0 : length(X_h_rows[1])
        X_full = Matrix{T}(undef, N_h, 1 + K_ctrl)
        for i in 1:N_h
            X_full[i, 1] = D_h[i]
            for j in 1:K_ctrl
                X_full[i, 1 + j] = X_h_rows[i][j]
            end
        end
        y_vec = Vector{T}(y_h)

        # Time-only demeaning (long differencing absorbs unit FE)
        y_dm = copy(y_vec)
        _lpdid_time_demean!(y_dm, time_ids_h)
        X_dm = copy(X_full)
        for k in 1:size(X_dm, 2)
            col = X_dm[:, k]
            _lpdid_time_demean!(col, time_ids_h)
            X_dm[:, k] = col
        end

        # OLS
        XtX_inv = robust_inv(X_dm' * X_dm; silent=true)
        beta = XtX_inv * (X_dm' * y_dm)
        resid = y_dm - X_dm * beta

        # Clustered SEs
        V = _did_vcov(X_dm, resid, unit_ids_h, time_ids_h, cluster)

        coefficients[hi] = beta[1]
        se_vec[hi] = sqrt(max(V[1, 1], zero(T)))
        B_all[hi] = reshape(beta, :, 1)
        resid_all[hi] = reshape(resid, :, 1)
        vcov_all[hi] = V
        T_eff_all[hi] = N_h
    end

    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = coefficients .- z .* se_vec
    ci_upper = coefficients .+ z .* se_vec

    EventStudyLP{T}(coefficients, se_vec, ci_lower, ci_upper,
                    event_times_all, reference_period,
                    B_all, resid_all, vcov_all, T_eff_all,
                    pd.varnames[outcome_col], pd.varnames[treat_col],
                    pd.T_obs, pd.n_groups, lags, leads, H,
                    false, cluster, T(conf_level), pd)
end
