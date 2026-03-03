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
IRF, FEVD, and Historical Decomposition dispatch for FAVAR via VAR conversion,
plus panel-wide IRF and forecast mapping through factor loadings.

All structural analysis methods work automatically through `to_var()`,
following the same pattern as VECM delegation (see `vecm/analysis.jl`).
"""

# =============================================================================
# Structural Analysis Delegation (via to_var)
# =============================================================================

"""
    irf(favar::FAVARModel, horizon; kwargs...) -> ImpulseResponse

Compute IRFs for a FAVAR by converting to VAR representation.
All identification methods (Cholesky, sign, narrative, etc.) are supported.
"""
function irf(favar::FAVARModel{T}, horizon::Int; kwargs...) where {T}
    irf(to_var(favar), horizon; kwargs...)
end

"""
    fevd(favar::FAVARModel, horizon; kwargs...) -> FEVD

Compute FEVD for a FAVAR by converting to VAR representation.
"""
function fevd(favar::FAVARModel{T}, horizon::Int; kwargs...) where {T}
    fevd(to_var(favar), horizon; kwargs...)
end

"""
    historical_decomposition(favar::FAVARModel, horizon; kwargs...) -> HistoricalDecomposition

Compute historical decomposition for a FAVAR by converting to VAR representation.
"""
function historical_decomposition(favar::FAVARModel{T}, horizon::Int=effective_nobs(favar); kwargs...) where {T}
    historical_decomposition(to_var(favar), horizon; kwargs...)
end

# =============================================================================
# Bayesian FAVAR Structural Analysis (via BVARPosterior delegation)
# =============================================================================

"""
    _to_bvar_posterior(bfavar::BayesianFAVAR) -> BVARPosterior

Convert a BayesianFAVAR to a BVARPosterior for delegation to existing
Bayesian structural analysis methods (IRF, FEVD, HD).
"""
function _to_bvar_posterior(bfavar::BayesianFAVAR{T}) where {T}
    BVARPosterior{T}(
        bfavar.B_draws,
        bfavar.Sigma_draws,
        size(bfavar.B_draws, 1),   # n_draws
        bfavar.p,
        bfavar.n,
        bfavar.data,
        :normal,                    # prior (placeholder)
        :gibbs,                     # sampler
        bfavar.varnames
    )
end

"""
    irf(bfavar::BayesianFAVAR, horizon; kwargs...) -> BayesianImpulseResponse

Compute Bayesian IRFs for a Bayesian FAVAR by converting to BVARPosterior
and delegating to the existing Bayesian IRF infrastructure.
"""
function irf(bfavar::BayesianFAVAR{T}, horizon::Int; kwargs...) where {T}
    irf(_to_bvar_posterior(bfavar), horizon; kwargs...)
end

"""
    fevd(bfavar::BayesianFAVAR, horizon; kwargs...) -> BayesianFEVD

Compute Bayesian FEVD for a Bayesian FAVAR.
"""
function fevd(bfavar::BayesianFAVAR{T}, horizon::Int; kwargs...) where {T}
    fevd(_to_bvar_posterior(bfavar), horizon; kwargs...)
end

"""
    historical_decomposition(bfavar::BayesianFAVAR; kwargs...) -> BayesianHistoricalDecomposition

Compute Bayesian historical decomposition for a Bayesian FAVAR.
"""
function historical_decomposition(bfavar::BayesianFAVAR{T}, horizon::Int=0; kwargs...) where {T}
    historical_decomposition(_to_bvar_posterior(bfavar), horizon; kwargs...)
end

# =============================================================================
# Panel-Wide IRF Mapping
# =============================================================================

"""
    favar_panel_irf(favar::FAVARModel, irf_result::ImpulseResponse) -> ImpulseResponse

Map factor-space IRFs to all N panel variables using the factor loadings.

For each panel variable i and shock j:
    panel_irf[h, i, j] = sum_k Lambda[i, k] * irf_result.values[h, k, j]

where Lambda is the N x r loading matrix from PCA.

Key variables (those in `Y_key_indices`) use their direct VAR IRF responses
instead of the factor mapping, providing exact impulse responses for the
variables that enter the FAVAR directly.

# Arguments
- `favar`: Estimated FAVAR model
- `irf_result`: IRF computed on the FAVAR's augmented VAR system

# Returns
`ImpulseResponse{T}` with N panel variables as the response dimension.

# Example
```julia
favar = estimate_favar(X, [1, 5], 3, 2)
irf_aug = irf(favar, 20)             # IRF in factor space (r + n_key vars)
irf_panel = favar_panel_irf(favar, irf_aug)  # IRF for all N panel vars
```
"""
function favar_panel_irf(favar::FAVARModel{T}, irf_result::ImpulseResponse{T}) where {T}
    r = favar.n_factors
    n_key = favar.n_key
    N = size(favar.X_panel, 2)
    n_aug = r + n_key  # number of VAR variables
    H = irf_result.horizon
    Lambda = favar.loadings  # N x r

    # Validate dimensions
    n_shocks = size(irf_result.values, 3)
    n_shocks == n_aug || throw(ArgumentError(
        "IRF has $n_shocks shocks but FAVAR has $n_aug VAR variables"))

    # Map factor IRFs to panel variables
    panel_values = zeros(T, H, N, n_shocks)

    for h in 1:H
        for j in 1:n_shocks
            # Factor contribution: Lambda * [factor IRFs at horizon h for shock j]
            factor_irfs_h = @view irf_result.values[h, 1:r, j]
            panel_values[h, :, j] = Lambda * factor_irfs_h
        end
    end

    # Override key variables with direct VAR IRF
    if !isempty(favar.Y_key_indices)
        for (k_idx, panel_idx) in enumerate(favar.Y_key_indices)
            if 1 <= panel_idx <= N
                var_idx = r + k_idx  # position of key var in augmented VAR
                for h in 1:H, j in 1:n_shocks
                    panel_values[h, panel_idx, j] = irf_result.values[h, var_idx, j]
                end
            end
        end
    end

    # Map confidence intervals if available
    has_ci = irf_result.ci_type != :none
    if has_ci
        panel_ci_lower = zeros(T, H, N, n_shocks)
        panel_ci_upper = zeros(T, H, N, n_shocks)

        for h in 1:H, j in 1:n_shocks
            factor_lo = @view irf_result.ci_lower[h, 1:r, j]
            factor_hi = @view irf_result.ci_upper[h, 1:r, j]
            panel_ci_lower[h, :, j] = Lambda * factor_lo
            panel_ci_upper[h, :, j] = Lambda * factor_hi
        end

        # Override key variable CIs
        if !isempty(favar.Y_key_indices)
            for (k_idx, panel_idx) in enumerate(favar.Y_key_indices)
                if 1 <= panel_idx <= N
                    var_idx = r + k_idx
                    for h in 1:H, j in 1:n_shocks
                        panel_ci_lower[h, panel_idx, j] = irf_result.ci_lower[h, var_idx, j]
                        panel_ci_upper[h, panel_idx, j] = irf_result.ci_upper[h, var_idx, j]
                    end
                end
            end
        end
    else
        panel_ci_lower = zeros(T, H, N, n_shocks)
        panel_ci_upper = zeros(T, H, N, n_shocks)
    end

    # Build variable and shock names for panel IRF
    panel_var_names = copy(favar.panel_varnames)
    shock_names = irf_result.shocks

    ImpulseResponse{T}(
        panel_values,
        panel_ci_lower,
        panel_ci_upper,
        H,
        panel_var_names,
        shock_names,
        irf_result.ci_type,
        nothing,
        irf_result._conf_level
    )
end

"""
    favar_panel_irf(bfavar::BayesianFAVAR, irf_result::BayesianImpulseResponse) -> BayesianImpulseResponse

Map Bayesian factor-space IRFs to all N panel variables using posterior mean loadings.

For each panel variable i and shock j:
    panel_irf[h, i, j] = sum_k Lambda_mean[i, k] * irf_result.point_estimate[h, k, j]

Key variables use their direct VAR IRF responses.

# Arguments
- `bfavar`: Estimated Bayesian FAVAR model
- `irf_result`: Bayesian IRF computed on the FAVAR's augmented VAR system

# Returns
`BayesianImpulseResponse{T}` with N panel variables as the response dimension.
"""
function favar_panel_irf(bfavar::BayesianFAVAR{T}, irf_result::BayesianImpulseResponse{T}) where {T}
    r = bfavar.n_factors
    n_key = bfavar.n_key
    N = size(bfavar.X_panel, 2)
    n_aug = r + n_key
    H = irf_result.horizon
    n_q = length(irf_result.quantile_levels)

    # Use posterior mean loadings for the mapping
    Lambda = dropdims(mean(bfavar.loadings_draws, dims=1), dims=1)  # N x r

    # Validate dimensions
    n_shocks = size(irf_result.point_estimate, 3)
    n_shocks == n_aug || throw(ArgumentError(
        "IRF has $n_shocks shocks but Bayesian FAVAR has $n_aug VAR variables"))

    # Map point estimate
    panel_pe = zeros(T, H, N, n_shocks)
    for h in 1:H, j in 1:n_shocks
        factor_irfs_h = @view irf_result.point_estimate[h, 1:r, j]
        panel_pe[h, :, j] = Lambda * factor_irfs_h
    end

    # Map quantiles
    panel_q = zeros(T, H, N, n_shocks, n_q)
    for qi in 1:n_q, h in 1:H, j in 1:n_shocks
        factor_q_h = @view irf_result.quantiles[h, 1:r, j, qi]
        panel_q[h, :, j, qi] = Lambda * factor_q_h
    end

    # Override key variables with direct VAR IRF
    if !isempty(bfavar.Y_key_indices)
        for (k_idx, panel_idx) in enumerate(bfavar.Y_key_indices)
            if 1 <= panel_idx <= N
                var_idx = r + k_idx
                for h in 1:H, j in 1:n_shocks
                    panel_pe[h, panel_idx, j] = irf_result.point_estimate[h, var_idx, j]
                    for qi in 1:n_q
                        panel_q[h, panel_idx, j, qi] = irf_result.quantiles[h, var_idx, j, qi]
                    end
                end
            end
        end
    end

    panel_var_names = copy(bfavar.panel_varnames)
    shock_names = irf_result.shocks

    BayesianImpulseResponse{T}(
        panel_q,
        panel_pe,
        H,
        panel_var_names,
        shock_names,
        irf_result.quantile_levels,
        nothing  # no raw draws for panel mapping
    )
end

# =============================================================================
# Panel-Wide Forecast Mapping
# =============================================================================

"""
    favar_panel_forecast(favar::FAVARModel, fc::VARForecast) -> VARForecast

Map factor-space forecasts to all N panel variables using the factor loadings.

Key variables use their direct VAR forecast; other variables are mapped
through Lambda * F_forecast.

# Arguments
- `favar`: Estimated FAVAR model
- `fc`: Forecast from the augmented VAR (via `forecast(favar, h)`)

# Returns
`VARForecast{T}` with N panel variable forecasts.
"""
function favar_panel_forecast(favar::FAVARModel{T}, fc::VARForecast{T}) where {T}
    r = favar.n_factors
    n_key = favar.n_key
    N = size(favar.X_panel, 2)
    h = fc.horizon
    Lambda = favar.loadings  # N x r

    # Map factor forecasts to panel
    panel_fc = zeros(T, h, N)
    panel_lo = zeros(T, h, N)
    panel_hi = zeros(T, h, N)

    for step in 1:h
        factor_fc = @view fc.forecast[step, 1:r]
        panel_fc[step, :] = Lambda * factor_fc

        factor_lo = @view fc.ci_lower[step, 1:r]
        factor_hi = @view fc.ci_upper[step, 1:r]
        panel_lo[step, :] = Lambda * factor_lo
        panel_hi[step, :] = Lambda * factor_hi
    end

    # Override key variables with direct forecasts
    if !isempty(favar.Y_key_indices)
        for (k_idx, panel_idx) in enumerate(favar.Y_key_indices)
            if 1 <= panel_idx <= N
                var_idx = r + k_idx
                for step in 1:h
                    panel_fc[step, panel_idx] = fc.forecast[step, var_idx]
                    panel_lo[step, panel_idx] = fc.ci_lower[step, var_idx]
                    panel_hi[step, panel_idx] = fc.ci_upper[step, var_idx]
                end
            end
        end
    end

    VARForecast{T}(
        panel_fc,
        panel_lo,
        panel_hi,
        h,
        fc.ci_method,
        fc.conf_level,
        copy(favar.panel_varnames)
    )
end

# =============================================================================
# Structural DFM — IRF and FEVD Dispatch
# =============================================================================

"""
    irf(sdfm::StructuralDFM, horizon; kwargs...) -> ImpulseResponse

Return pre-computed panel-wide structural IRFs from a Structural DFM.

The structural IRFs map identified factor shocks to all N panel variables
through the time-domain loadings Lambda.

Dimensions: (H x N x q) where N = panel variables, q = structural shocks.

If `horizon` exceeds the stored horizon, returns IRFs up to the stored horizon.
"""
function irf(sdfm::StructuralDFM{T}, horizon::Int; kwargs...) where {T}
    H_stored = size(sdfm.structural_irf, 1)
    H = min(horizon, H_stored)
    N = size(sdfm.structural_irf, 2)
    q = size(sdfm.structural_irf, 3)

    values = sdfm.structural_irf[1:H, :, :]

    ci_lo = zeros(T, H, N, q)
    ci_hi = zeros(T, H, N, q)

    panel_names = ["Var $i" for i in 1:N]

    ImpulseResponse{T}(values, ci_lo, ci_hi, H, panel_names,
        sdfm.shock_names, :none, nothing, zero(T))
end

"""
    fevd(sdfm::StructuralDFM, horizon; kwargs...) -> FEVD

Compute FEVD for the factor VAR underlying a Structural DFM.

This delegates to the standard FEVD computation on the q-variable factor VAR,
providing the forecast error variance decomposition among structural shocks
in the factor space.
"""
function fevd(sdfm::StructuralDFM{T}, horizon::Int; kwargs...) where {T}
    fevd(sdfm.factor_var, horizon; kwargs...)
end
