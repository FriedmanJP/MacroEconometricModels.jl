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
Event Study Local Projections with panel fixed effects.

Implements:
- `estimate_event_study_lp`: Standard panel event study LP (Jorda 2005 + panel FE)
- `estimate_lp_did`: LP-DiD with clean controls (Dube et al. 2023)

Both estimate horizon-by-horizon regressions:
    Y_{i,t+h} - Y_{i,t-1} = alpha_i^h + gamma_t^h + beta_h D_{it} + X_{it}'delta^h + epsilon_{i,t+h}
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

# Arguments
- `pd`: Panel data
- `outcome`: Outcome variable name
- `treatment`: Treatment timing variable name (contains period of first treatment)
- `H`: Maximum post-treatment horizon
- `leads`: Number of pre-treatment leads to estimate (for pre-trends)
- `lags`: Number of control lags
- `covariates`: Additional control variable names
- `cluster`: Clustering level (:unit, :time, :twoway)
- `conf_level`: Confidence level (default: 0.95)

# Returns
`EventStudyLP{T}` with dynamic treatment effect coefficients.
"""
function estimate_event_study_lp(pd::PanelData{T}, outcome::Union{String,Symbol},
                                 treatment::Union{String,Symbol}, H::Int;
                                 leads::Int=3, lags::Int=4,
                                 covariates::Vector{String}=String[],
                                 cluster::Symbol=:unit,
                                 conf_level::Real=0.95) where {T<:AbstractFloat}
    _event_study_lp_internal(pd, outcome, treatment, H;
                             leads=leads, lags=lags, covariates=covariates,
                             cluster=cluster, conf_level=conf_level,
                             clean_controls=false)
end

"""
    estimate_lp_did(pd::PanelData{T}, outcome::Union{String,Symbol},
                    treatment::Union{String,Symbol}, H::Int;
                    leads::Int=3, lags::Int=4,
                    covariates::Vector{String}=String[],
                    cluster::Symbol=:unit,
                    conf_level::Real=0.95) where {T}

LP-DiD estimator (Dube et al. 2023) with clean controls.

Same as `estimate_event_study_lp` but restricts the control group at each horizon h
to units that are never-treated or not-yet-treated at t+h. This avoids contamination
bias from already-treated units serving as controls.
"""
function estimate_lp_did(pd::PanelData{T}, outcome::Union{String,Symbol},
                         treatment::Union{String,Symbol}, H::Int;
                         leads::Int=3, lags::Int=4,
                         covariates::Vector{String}=String[],
                         cluster::Symbol=:unit,
                         conf_level::Real=0.95) where {T<:AbstractFloat}
    _event_study_lp_internal(pd, outcome, treatment, H;
                             leads=leads, lags=lags, covariates=covariates,
                             cluster=cluster, conf_level=conf_level,
                             clean_controls=true)
end

"""
    _event_study_lp_internal(pd, outcome, treatment, H; leads, lags, covariates,
                             cluster, conf_level, clean_controls)

Internal implementation shared by estimate_event_study_lp and estimate_lp_did.
"""
function _event_study_lp_internal(pd::PanelData{T}, outcome::Union{String,Symbol},
                                  treatment::Union{String,Symbol}, H::Int;
                                  leads::Int=3, lags::Int=4,
                                  covariates::Vector{String}=String[],
                                  cluster::Symbol=:unit,
                                  conf_level::Real=0.95,
                                  clean_controls::Bool=false) where {T<:AbstractFloat}
    outcome_col = _resolve_varindex(pd, outcome)
    treat_col = _resolve_varindex(pd, treatment)
    cov_cols = [_resolve_varindex(pd, c) for c in covariates]

    timing = _extract_treatment_timing(pd, treat_col)

    # Override timing with cohort_id if present
    if pd.cohort_id !== nothing
        for g in 1:pd.n_groups
            mask = pd.group_id .== g
            timing[g] = pd.cohort_id[findfirst(mask)]
        end
    end

    all_times = sort(unique(pd.time_id))

    # Build panel lookup: group -> time -> row_index
    panel_idx = Dict{Int, Dict{Int, Int}}()
    for i in 1:pd.T_obs
        g = pd.group_id[i]
        t = pd.time_id[i]
        if !haskey(panel_idx, g)
            panel_idx[g] = Dict{Int, Int}()
        end
        panel_idx[g][t] = i
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
            # Reference period: coefficient = 0
            coefficients[hi] = zero(T)
            se_vec[hi] = zero(T)
            B_all[hi] = zeros(T, 1, 1)
            resid_all[hi] = zeros(T, 0, 1)
            vcov_all[hi] = zeros(T, 1, 1)
            T_eff_all[hi] = 0
            continue
        end

        # Build regression data for this horizon
        y_h = T[]
        D_h = T[]           # treatment indicator
        X_h_rows = Vector{Vector{T}}()
        unit_ids_h = Int[]
        time_ids_h = Int[]

        for g in 1:pd.n_groups
            g_time = timing[g]
            haskey(panel_idx, g) || continue

            for t in all_times
                # Need Y at t+h and Y at t-1
                haskey(panel_idx[g], t + h) || continue
                haskey(panel_idx[g], t - 1) || continue

                # For LP-DiD (Dube et al. 2023): exclude contaminated controls.
                # A unit is a contaminated control if it becomes treated between
                # t and t+h (not-yet-treated at t, but treated at t+h).
                if clean_controls && g_time > 0 && g_time > t && g_time <= t + h
                    continue
                end

                row_h = panel_idx[g][t + h]
                row_base = panel_idx[g][t - 1]

                # Dependent variable: Y_{i,t+h} - Y_{i,t-1}
                dy = pd.data[row_h, outcome_col] - pd.data[row_base, outcome_col]

                # Treatment dummy
                d_val = (g_time > 0 && t >= g_time) ? one(T) : zero(T)

                # Control lags: Y_{i,t-l} for l=1,...,lags
                x_row = T[one(T)]  # intercept
                valid_lags = true
                for l in 1:lags
                    if haskey(panel_idx[g], t - l)
                        row_l = panel_idx[g][t - l]
                        push!(x_row, pd.data[row_l, outcome_col])
                    else
                        valid_lags = false
                        break
                    end
                end
                !valid_lags && continue

                # Covariates at time t
                if !isempty(cov_cols)
                    row_t = get(panel_idx[g], t, nothing)
                    row_t === nothing && continue
                    for c in cov_cols
                        push!(x_row, pd.data[row_t, c])
                    end
                end

                push!(y_h, dy)
                push!(D_h, d_val)
                push!(X_h_rows, x_row)
                push!(unit_ids_h, g)
                push!(time_ids_h, t)
            end
        end

        N_h = length(y_h)
        if N_h < 3
            # Not enough observations
            coefficients[hi] = T(NaN)
            se_vec[hi] = T(NaN)
            B_all[hi] = zeros(T, 1, 1)
            resid_all[hi] = zeros(T, 0, 1)
            vcov_all[hi] = zeros(T, 1, 1)
            T_eff_all[hi] = N_h
            continue
        end

        # Assemble matrices
        K_ctrl = length(X_h_rows[1])
        X_full = Matrix{T}(undef, N_h, K_ctrl + 1)  # [D, controls]
        for i in 1:N_h
            X_full[i, 1] = D_h[i]
            X_full[i, 2:end] = X_h_rows[i]
        end
        y_vec = Vector{T}(y_h)

        # Double-demean for unit + time FE
        y_dm = _double_demean(y_vec, unit_ids_h, time_ids_h)
        X_dm = _double_demean_matrix(X_full, unit_ids_h, time_ids_h)

        # OLS
        XtX_inv = robust_inv(X_dm' * X_dm; silent=true)
        beta = XtX_inv * (X_dm' * y_dm)
        resid = y_dm - X_dm * beta

        # Clustered SEs
        V = _did_vcov(X_dm, resid, unit_ids_h, time_ids_h, cluster)

        coefficients[hi] = beta[1]  # treatment coefficient
        se_vec[hi] = sqrt(max(V[1, 1], zero(T)))
        B_all[hi] = reshape(beta, :, 1)
        resid_all[hi] = reshape(resid, :, 1)
        vcov_all[hi] = V
        T_eff_all[hi] = N_h
    end

    # CIs
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = coefficients .- z .* se_vec
    ci_upper = coefficients .+ z .* se_vec

    EventStudyLP{T}(coefficients, se_vec, ci_lower, ci_upper,
                    event_times_all, reference_period,
                    B_all, resid_all, vcov_all, T_eff_all,
                    pd.varnames[outcome_col], pd.varnames[treat_col],
                    pd.T_obs, pd.n_groups, lags, leads, H,
                    clean_controls, cluster, T(conf_level), pd)
end
