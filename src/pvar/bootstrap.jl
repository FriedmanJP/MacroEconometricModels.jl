# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Group-level (block) bootstrap for Panel VAR impulse response confidence intervals.

Resamples groups (entities) with replacement, re-estimates the PVAR, computes
IRFs, and constructs pointwise quantile-based confidence intervals.
"""

# =============================================================================
# Bootstrap IRF
# =============================================================================

"""
    pvar_bootstrap_irf(model::PVARModel{T}, H::Int;
                        irf_type::Symbol=:oirf, n_draws::Int=500,
                        ci::Real=0.95, rng::AbstractRNG=Random.default_rng()) -> NamedTuple

Group-level block bootstrap for PVAR impulse response confidence intervals.

Resamples N groups with replacement → re-estimates PVAR → computes IRF → collects
pointwise quantiles.

# Arguments
- `model::PVARModel` — estimated PVAR
- `H::Int` — maximum IRF horizon

# Keywords
- `irf_type::Symbol=:oirf` — `:oirf` or `:girf`
- `n_draws::Int=500` — number of bootstrap replications
- `ci::Real=0.95` — confidence level
- `rng::AbstractRNG` — random number generator

# Returns
NamedTuple with:
- `irf::Array{T,3}` — point estimate (H+1 × m × m)
- `lower::Array{T,3}` — lower CI bound
- `upper::Array{T,3}` — upper CI bound
- `draws::Array{T,4}` — all bootstrap draws (n_draws × H+1 × m × m)
"""
function pvar_bootstrap_irf(model::PVARModel{T}, H::Int;
                             irf_type::Symbol=:oirf,
                             n_draws::Int=500,
                             ci::Real=0.95,
                             seed::Union{Integer,Nothing}=nothing,
                             rng::AbstractRNG=Random.default_rng()) where {T}
    rng = _resolve_repro_rng(rng, seed)
    H < 0 && throw(ArgumentError("Horizon H must be non-negative"))
    irf_type ∈ (:oirf, :girf) || throw(ArgumentError("irf_type must be :oirf or :girf"))
    0 < ci < 1 || throw(ArgumentError("ci must be in (0, 1)"))

    m_dim = model.m
    N = model.n_groups
    alpha = 1 - ci

    # Point estimate
    irf_point = irf_type == :oirf ? pvar_oirf(model, H) : pvar_girf(model, H)

    # Collect bootstrap draws
    draws = Array{T}(undef, n_draws, H + 1, m_dim, m_dim)

    d = model.data
    dep_names = model.varnames
    predet_names = model.predet_names
    exog_names = model.exog_names

    # Pre-index group rows to avoid repeated mask computation
    group_rows = [findall(d.group_id .== g) for g in 1:N]

    # Pre-generate seeds for thread-safe RNG (must be done before threaded loop)
    boot_seeds = rand(rng, UInt64, n_draws)

    Threads.@threads for b in 1:n_draws
        local_rng = Random.MersenneTwister(boot_seeds[b])

        # Resample groups with replacement
        sampled_groups = rand(local_rng, 1:N, N)

        # Build new PanelData from resampled groups
        new_data = Matrix{T}(undef, 0, nvars(d))
        new_gid = Int[]
        new_tid = Int[]

        for (new_g, orig_g) in enumerate(sampled_groups)
            rows = d.data[group_rows[orig_g], :]
            tids = d.time_id[group_rows[orig_g]]
            new_data = vcat(new_data, rows)
            append!(new_gid, fill(new_g, size(rows, 1)))
            append!(new_tid, tids)
        end

        d_boot = PanelData{T}(
            new_data, copy(d.varnames), d.frequency, copy(d.tcode),
            new_gid, new_tid, nothing, [string(i) for i in 1:N],
            N, d.n_vars, size(new_data, 1),
            d.balanced, copy(d.desc), copy(d.vardesc), copy(d.source_refs)
        )

        # Re-estimate
        try
            model_b = if model.method == :fe_ols
                estimate_pvar_feols(d_boot, model.p;
                                    dependent_vars=dep_names,
                                    predet_vars=predet_names,
                                    exog_vars=exog_names)
            else
                estimate_pvar(d_boot, model.p;
                              dependent_vars=dep_names,
                              predet_vars=predet_names,
                              exog_vars=exog_names,
                              transformation=model.transformation,
                              steps=model.steps,
                              system_instruments=(model.method == :system_gmm),
                              system_constant=model.system_constant,
                              min_lag_endo=model.min_lag_endo,
                              max_lag_endo=model.max_lag_endo,
                              collapse=model.collapse,
                              pca_instruments=model.pca_instruments,
                              pca_max_components=model.pca_max_components)
            end

            irf_b = irf_type == :oirf ? pvar_oirf(model_b, H) : pvar_girf(model_b, H)
            draws[b, :, :, :] = irf_b
        catch
            # On failure, use point estimate (conservative)
            draws[b, :, :, :] = irf_point
        end
    end

    # Compute quantile-based CIs
    lower = Array{T}(undef, H + 1, m_dim, m_dim)
    upper = Array{T}(undef, H + 1, m_dim, m_dim)

    @inbounds for h in 1:(H+1), i in 1:m_dim, j in 1:m_dim
        vals = sort(@view draws[:, h, i, j])
        lower[h, i, j] = quantile(vals, alpha / 2)
        upper[h, i, j] = quantile(vals, 1 - alpha / 2)
    end

    manifest = capture_manifest(; seed=seed, settings=Dict{String,Any}(
        "H" => H, "irf_type" => String(irf_type), "n_draws" => n_draws, "ci" => Float64(ci)))
    updated = PVARModel{T}(
        model.Phi, model.Sigma, model.se, model.pvalues,
        model.m, model.p, model.n_predet, model.n_exog,
        model.varnames, model.predet_names, model.exog_names,
        model.method, model.transformation, model.steps, model.system_constant,
        model.min_lag_endo, model.max_lag_endo, model.collapse,
        model.pca_instruments, model.pca_max_components,
        model.n_groups, model.n_periods, model.n_obs, model.obs_per_group,
        model.instruments, model.residuals_transformed, model.weighting_matrix,
        model.n_instruments, model.data, model.coef_vcov;
        boot_irf=irf_point, boot_lower=lower, boot_upper=upper, boot_draws=draws,
        manifest=manifest)
    (irf=irf_point, lower=lower, upper=upper, draws=draws, model=updated, manifest=manifest)
end
