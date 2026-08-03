# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
    Lambda_y = favar.Lambda_y

    # Validate dimensions
    n_shocks = size(irf_result.values, 3)
    n_shocks == n_aug || throw(ArgumentError(
        "IRF has $n_shocks shocks but FAVAR has $n_aug VAR variables"))

    # Map factor IRFs to panel variables (point estimate)
    panel_values = zeros(T, H, N, n_shocks)
    for h in 1:H, j in 1:n_shocks
        factor_irfs_h = @view irf_result.values[h, 1:r, j]
        y_irfs_h = @view irf_result.values[h, (r + 1):(r + n_key), j]
        panel_values[h, :, j] = Lambda * factor_irfs_h + Lambda_y * y_irfs_h
    end
    _favar_override_key_irf!(panel_values, irf_result.values, favar.Y_key_indices, r, N)

    # Confidence intervals: push DRAWS through Λ and take quantiles in panel space (#524).
    # Never map interval endpoints — a linear map with negative loadings inverts bounds.
    has_ci = irf_result.ci_type != :none
    panel_draws = nothing
    if has_ci && irf_result._draws !== nothing
        draws = irf_result._draws  # (reps, H, n_aug, n_shocks)
        n_reps = size(draws, 1)
        panel_draws = zeros(T, n_reps, H, N, n_shocks)
        for rep in 1:n_reps, h in 1:H, j in 1:n_shocks
            factor_d = @view draws[rep, h, 1:r, j]
            y_d = @view draws[rep, h, (r + 1):(r + n_key), j]
            panel_draws[rep, h, :, j] = Lambda * factor_d + Lambda_y * y_d
        end
        _favar_override_key_draws!(panel_draws, draws, favar.Y_key_indices, r, N)
        alpha = (one(T) - irf_result._conf_level) / 2
        panel_ci_lower = zeros(T, H, N, n_shocks)
        panel_ci_upper = zeros(T, H, N, n_shocks)
        @inbounds for h in 1:H, v in 1:N, s in 1:n_shocks
            d = @view panel_draws[:, h, v, s]
            panel_ci_lower[h, v, s] = quantile(d, alpha)
            panel_ci_upper[h, v, s] = quantile(d, one(T) - alpha)
        end
    elseif has_ci
        # Fallback when raw draws are unavailable: map endpoints then order with min/max
        # so inverted intervals cannot leak into the result. This is an APPROXIMATION —
        # it assumes the augmented components are comonotonic, which understates the
        # band whenever loadings mix components of opposite sign. The bootstrap path
        # above (ci_type=:bootstrap keeps raw draws) is the supported route (#524);
        # the IRF coefficient covariance needed for an exact analytic mapping is not
        # retained on ImpulseResponse.
        panel_ci_lower = zeros(T, H, N, n_shocks)
        panel_ci_upper = zeros(T, H, N, n_shocks)
        for h in 1:H, j in 1:n_shocks
            a = Lambda * (@view irf_result.ci_lower[h, 1:r, j]) +
                Lambda_y * (@view irf_result.ci_lower[h, (r + 1):(r + n_key), j])
            b = Lambda * (@view irf_result.ci_upper[h, 1:r, j]) +
                Lambda_y * (@view irf_result.ci_upper[h, (r + 1):(r + n_key), j])
            panel_ci_lower[h, :, j] = min.(a, b)
            panel_ci_upper[h, :, j] = max.(a, b)
        end
        _favar_override_key_irf!(panel_ci_lower, irf_result.ci_lower, favar.Y_key_indices, r, N)
        _favar_override_key_irf!(panel_ci_upper, irf_result.ci_upper, favar.Y_key_indices, r, N)
    else
        panel_ci_lower = zeros(T, H, N, n_shocks)
        panel_ci_upper = zeros(T, H, N, n_shocks)
    end

    ImpulseResponse{T}(
        panel_values,
        panel_ci_lower,
        panel_ci_upper,
        H,
        copy(favar.panel_varnames),
        irf_result.shocks,
        irf_result.ci_type,
        panel_draws,
        irf_result._conf_level
    )
end

"""Override key-variable slices of a panel (H × N × n_shocks) array with direct VAR values."""
function _favar_override_key_irf!(panel::AbstractArray{T,3}, src::AbstractArray{T,3},
                                  key_indices::Vector{Int}, r::Int, N::Int) where {T}
    isempty(key_indices) && return panel
    H = size(panel, 1)
    n_shocks = size(panel, 3)
    for (k_idx, panel_idx) in enumerate(key_indices)
        (1 <= panel_idx <= N) || continue
        var_idx = r + k_idx
        for h in 1:H, j in 1:n_shocks
            panel[h, panel_idx, j] = src[h, var_idx, j]
        end
    end
    panel
end

"""Override key-variable slices of a draws array (reps × H × N × n_shocks)."""
function _favar_override_key_draws!(panel::AbstractArray{T,4}, src::AbstractArray{T,4},
                                    key_indices::Vector{Int}, r::Int, N::Int) where {T}
    isempty(key_indices) && return panel
    n_reps, H = size(panel, 1), size(panel, 2)
    n_shocks = size(panel, 4)
    for (k_idx, panel_idx) in enumerate(key_indices)
        (1 <= panel_idx <= N) || continue
        var_idx = r + k_idx
        for rep in 1:n_reps, h in 1:H, j in 1:n_shocks
            panel[rep, h, panel_idx, j] = src[rep, h, var_idx, j]
        end
    end
    panel
end

"""
    favar_panel_irf(bfavar::BayesianFAVAR, irf_result::BayesianImpulseResponse) -> BayesianImpulseResponse

Map Bayesian factor-space IRFs to all N panel variables using posterior mean loadings.

For each panel variable i and shock j:
    panel_irf[h, i, j] = Λ[i,:]·factor_irf[h,:,j] + Λ_y[i,:]·y_irf[h,:,j]

Since the #528 fix the Gibbs measurement equation is BBE (2005) eq. 3,
`X = F Λ' + Y_key Λ_y' + e`, so the factors are purged of the key variables' own
variation and the direct channel `Λ_y · y_irf` must be added back — omitting it
would understate the response of `Y_key`-loading panel series. (Before #528 the
factors absorbed the entire `Y_key` co-movement and adding `Λ_y` would have
double-counted, which was the original #525 resolution; the model changed.)

Key variables use their direct VAR IRF responses. When raw posterior draws are available
on `irf_result`, they are pushed through the loadings and quantiles are recomputed in
panel space (#524) — never map quantile endpoints through Λ.

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

    # Posterior mean loadings (common factors + direct key-variable channel, #528)
    Lambda = dropdims(mean(bfavar.loadings_draws, dims=1), dims=1)     # N x r
    Lambda_y = dropdims(mean(bfavar.lambda_y_draws, dims=1), dims=1)   # N x n_key

    # Validate dimensions
    n_shocks = size(irf_result.point_estimate, 3)
    n_shocks == n_aug || throw(ArgumentError(
        "IRF has $n_shocks shocks but Bayesian FAVAR has $n_aug VAR variables"))

    # Map point estimate through the factor loadings + the direct Λ_y channel
    panel_pe = zeros(T, H, N, n_shocks)
    for h in 1:H, j in 1:n_shocks
        factor_irfs_h = @view irf_result.point_estimate[h, 1:r, j]
        y_irfs_h = @view irf_result.point_estimate[h, (r+1):n_aug, j]
        panel_pe[h, :, j] = Lambda * factor_irfs_h + Lambda_y * y_irfs_h
    end
    _favar_override_key_irf!(panel_pe, irf_result.point_estimate, bfavar.Y_key_indices, r, N)

    panel_draws = nothing
    panel_q = zeros(T, H, N, n_shocks, n_q)
    if irf_result._draws !== nothing
        # Push draws through loadings; recompute quantiles in panel space (#524)
        draws = irf_result._draws  # (samples, H, n_aug, n_shocks)
        n_reps = size(draws, 1)
        panel_draws = zeros(T, n_reps, H, N, n_shocks)
        for rep in 1:n_reps, h in 1:H, j in 1:n_shocks
            factor_d = @view draws[rep, h, 1:r, j]
            y_d = @view draws[rep, h, (r+1):n_aug, j]
            panel_draws[rep, h, :, j] = Lambda * factor_d + Lambda_y * y_d
        end
        _favar_override_key_draws!(panel_draws, draws, bfavar.Y_key_indices, r, N)
        @inbounds for qi in 1:n_q, h in 1:H, v in 1:N, s in 1:n_shocks
            panel_q[h, v, s, qi] = quantile(@view(panel_draws[:, h, v, s]),
                                            irf_result.quantile_levels[qi])
        end
    else
        # Fallback: map quantile endpoints then order (cannot invert without draws)
        for qi in 1:n_q, h in 1:H, j in 1:n_shocks
            factor_q_h = @view irf_result.quantiles[h, 1:r, j, qi]
            y_q_h = @view irf_result.quantiles[h, (r+1):n_aug, j, qi]
            panel_q[h, :, j, qi] = Lambda * factor_q_h + Lambda_y * y_q_h
        end
        if !isempty(bfavar.Y_key_indices)
            for (k_idx, panel_idx) in enumerate(bfavar.Y_key_indices)
                (1 <= panel_idx <= N) || continue
                var_idx = r + k_idx
                for h in 1:H, j in 1:n_shocks, qi in 1:n_q
                    panel_q[h, panel_idx, j, qi] = irf_result.quantiles[h, var_idx, j, qi]
                end
            end
        end
        # Ensure lower ≤ upper across quantile levels for each (h,v,s)
        if n_q >= 2
            @inbounds for h in 1:H, v in 1:N, s in 1:n_shocks
                lo, hi = panel_q[h, v, s, 1], panel_q[h, v, s, n_q]
                if lo > hi
                    panel_q[h, v, s, 1] = hi
                    panel_q[h, v, s, n_q] = lo
                end
            end
        end
    end

    BayesianImpulseResponse{T}(
        panel_q,
        panel_pe,
        H,
        copy(bfavar.panel_varnames),
        irf_result.shocks,
        irf_result.quantile_levels,
        panel_draws,
        irf_result.n_requested,
        irf_result.n_effective,
        irf_result.n_failed
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
    Lambda_y = favar.Lambda_y

    # Map factor forecasts to panel (point)
    panel_fc = zeros(T, h, N)
    for step in 1:h
        factor_fc = @view fc.forecast[step, 1:r]
        y_fc = @view fc.forecast[step, (r + 1):(r + n_key)]
        panel_fc[step, :] = Lambda * factor_fc + Lambda_y * y_fc
    end

    # CI: push bootstrap draws through Λ when available (#524); else order mapped endpoints.
    panel_lo = zeros(T, h, N)
    panel_hi = zeros(T, h, N)
    panel_draws = nothing
    if fc.ci_method != :none && fc._draws !== nothing
        draws = fc._draws  # (reps, h, n_aug)
        n_reps = size(draws, 1)
        panel_draws = zeros(T, n_reps, h, N)
        for rep in 1:n_reps, step in 1:h
            factor_d = @view draws[rep, step, 1:r]
            y_d = @view draws[rep, step, (r + 1):(r + n_key)]
            panel_draws[rep, step, :] = Lambda * factor_d + Lambda_y * y_d
        end
        # Override key variables with direct forecast draws
        if !isempty(favar.Y_key_indices)
            for (k_idx, panel_idx) in enumerate(favar.Y_key_indices)
                (1 <= panel_idx <= N) || continue
                var_idx = r + k_idx
                for rep in 1:n_reps, step in 1:h
                    panel_draws[rep, step, panel_idx] = draws[rep, step, var_idx]
                end
            end
        end
        alpha = (one(T) - fc.conf_level) / 2
        @inbounds for step in 1:h, j in 1:N
            d = @view panel_draws[:, step, j]
            panel_lo[step, j] = quantile(d, alpha)
            panel_hi[step, j] = quantile(d, one(T) - alpha)
        end
    elseif fc.ci_method != :none
        # Analytic panel bands (#524). Mapping interval ENDPOINTS through Λ
        # assumes the augmented variables are comonotonic and produces invalid
        # (sometimes near-zero-width) bands. The correct band uses the full
        # forecast-error covariance of the augmented VAR:
        #   Var(x̂_i(h)) = [Λ_aug · MSE(h) · Λ_aug']_{ii},
        # with MSE(h) = Σ_{s<h} Φ_s Σ Φ_s' (Lütkepohl §3.5) and Λ_aug = [Λ Λ_y].
        n_aug = r + n_key
        A = extract_ar_coefficients(favar.B, n_aug, favar.p)
        Phi = Vector{Matrix{T}}(undef, h)
        Phi[1] = Matrix{T}(I, n_aug, n_aug)
        for i in 1:(h - 1)
            acc = zeros(T, n_aug, n_aug)
            for j in 1:min(i, favar.p)
                acc .+= Phi[i - j + 1] * A[j]
            end
            Phi[i + 1] = acc
        end
        Lambda_aug = hcat(Lambda, Lambda_y)          # N × n_aug
        z = T(quantile(Normal(), 1 - (1 - fc.conf_level) / 2))
        mse = zeros(T, n_aug, n_aug)
        for step in 1:h
            mse .+= Phi[step] * favar.Sigma * Phi[step]'
            pv = Lambda_aug * mse * Lambda_aug'
            for j in 1:N
                se = sqrt(max(pv[j, j], zero(T)))
                panel_lo[step, j] = panel_fc[step, j] - z * se
                panel_hi[step, j] = panel_fc[step, j] + z * se
            end
        end
    end

    # Override key variables with direct forecasts (point + CI when no draws path)
    if !isempty(favar.Y_key_indices)
        for (k_idx, panel_idx) in enumerate(favar.Y_key_indices)
            (1 <= panel_idx <= N) || continue
            var_idx = r + k_idx
            for step in 1:h
                panel_fc[step, panel_idx] = fc.forecast[step, var_idx]
                if fc.ci_method != :none && fc._draws === nothing
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
        copy(favar.panel_varnames),
        panel_draws
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

    panel_names = copy(sdfm.varnames)

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

# =============================================================================
# sdfm_panel_irf — Project structural factor IRFs to panel observables
# =============================================================================

"""
    sdfm_panel_irf(sdfm::StructuralDFM, H::Int) -> ImpulseResponse{T}

Compute panel-wide structural IRFs by projecting factor-space IRFs to all N
observable variables via the time-domain loading matrix.

Internally computes factor IRFs from the factor VAR using the stored
identification matrix Q, then applies `Λ * factor_irf` at each horizon.

# Arguments
- `sdfm`: Estimated Structural DFM
- `H`: IRF horizon

# Returns
`ImpulseResponse{T}` with dimensions (H, N, q) — N observable responses to q structural shocks.
"""
function sdfm_panel_irf(sdfm::StructuralDFM{T}, H::Int) where {T}
    H >= 1 || throw(ArgumentError("horizon H must be >= 1"))

    q = sdfm.gdfm.q
    factor_irf = compute_irf(sdfm.factor_var, sdfm.Q, H)  # H x q x q

    _sdfm_project_irf(sdfm, factor_irf, H)
end

"""
    sdfm_panel_irf(sdfm::StructuralDFM, irf_result::ImpulseResponse) -> ImpulseResponse{T}

Project an existing factor-space `ImpulseResponse` to all N observable panel
variables via the time-domain loading matrix.

Analogous to `favar_panel_irf`: takes factor-space IRFs (from the factor VAR
with any identification) and maps them to observable space.

# Arguments
- `sdfm`: Estimated Structural DFM (provides loadings)
- `irf_result`: Factor-space IRF (H x q x q)

# Returns
`ImpulseResponse{T}` with dimensions (H, N, q).
"""
function sdfm_panel_irf(sdfm::StructuralDFM{T}, irf_result::ImpulseResponse{T}) where {T}
    q = sdfm.gdfm.q
    n_vars = size(irf_result.values, 2)
    n_shocks = size(irf_result.values, 3)
    H = irf_result.horizon

    n_vars == q || throw(ArgumentError(
        "IRF has $n_vars variables but StructuralDFM has $q factors"))
    n_shocks == q || throw(ArgumentError(
        "IRF has $n_shocks shocks but StructuralDFM has $q factors"))

    _sdfm_project_irf(sdfm, irf_result.values, H)
end

"""Project factor-space IRF array (H x q x q) to panel space (H x N x q)."""
function _sdfm_project_irf(sdfm::StructuralDFM{T}, factor_irf::AbstractArray{T,3}, H::Int) where {T}
    q = sdfm.gdfm.q
    N = size(sdfm.loadings_td, 1)
    Lambda = sdfm.loadings_td  # N x q

    panel_values = zeros(T, H, N, q)
    for h in 1:H
        for j in 1:q
            factor_irfs_h = @view factor_irf[h, :, j]
            panel_values[h, :, j] = Lambda * factor_irfs_h
        end
    end

    panel_names = copy(sdfm.varnames)
    ci_lo = zeros(T, H, N, q)
    ci_hi = zeros(T, H, N, q)

    ImpulseResponse{T}(panel_values, ci_lo, ci_hi, H, panel_names,
        sdfm.shock_names, :none, nothing, zero(T))
end
