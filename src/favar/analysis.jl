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

"""Reject identification-method keywords: the rotation is fixed at estimation (#697 / SDFM-01)."""
function _reject_sdfm_ident_kwargs(kwargs, fn::AbstractString)
    isempty(kwargs) && return nothing
    k = first(keys(kwargs))
    if k === :method || k === :check_func || k === :narrative_check
        throw(ArgumentError("$fn does not accept `$k`; identification is fixed at estimation"))
    end
    throw(ArgumentError("$fn does not accept `$k`"))
end

"""
    irf(sdfm::StructuralDFM, horizon; kwargs...) -> ImpulseResponse

Panel-wide structural IRFs from a Structural DFM, computed on demand for the
requested horizon from the stored rotation `Q` and time-domain loadings.

Dimensions: (H x N x q) where N = panel variables, q = structural shocks.

Identification is fixed at estimation — `method` / `check_func` / `narrative_check`
are rejected. Alias of [`sdfm_panel_irf`](@ref).
"""
function irf(sdfm::StructuralDFM{T}, horizon::Int;
    point::Symbol=:auto,
    ci_type::Symbol=:none,
    reps::Int=200,
    conf_level::Real=0.95,
    bootstrap::Symbol=:iid,
    block_length::Int=0,
    wild_dist::Symbol=:rademacher,
    stationary_only::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
    kwargs...) where {T}
    _reject_sdfm_ident_kwargs(kwargs, "irf(::StructuralDFM)")
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    ci_type in (:none, :bootstrap) || throw(ArgumentError(
        "ci_type must be :none or :bootstrap, got :$ci_type"))
    if ci_type === :bootstrap
        rng = _resolve_repro_rng(rng, seed)
        return _sdfm_bootstrap_irf(sdfm, horizon; reps=reps, conf_level=T(conf_level),
            bootstrap=bootstrap, block_length=block_length, wild_dist=wild_dist,
            stationary_only=stationary_only, rng=rng, seed=seed)
    end
    if sdfm.identified_set !== nothing && point !== :first
        return _sdfm_set_irf(sdfm, horizon)
    end
    sdfm_panel_irf(sdfm, horizon)
end

"""Residual bootstrap of panel IRFs: resample factor-VAR residuals, re-estimate
the VAR and `K`, re-apply the stored identification rule, project through `Λ`,
then take pointwise quantiles. Sign-normalise each draw on the first ordered
observable's impact so bands do not straddle zero from label switching."""
function _sdfm_bootstrap_irf(sdfm::StructuralDFM{T}, horizon::Int;
    reps::Int, conf_level::T, bootstrap::Symbol, block_length::Int,
    wild_dist::Symbol, stationary_only::Bool, rng::AbstractRNG,
    seed::Union{Integer,Nothing}) where {T<:AbstractFloat}
    reps < 1 && throw(ArgumentError("reps must be ≥ 1"))
    bootstrap in (:iid, :wild, :block) || throw(ArgumentError(
        "bootstrap must be :iid, :wild, or :block, got :$bootstrap"))
    point = sdfm_panel_irf(sdfm, horizon)
    fv = sdfm.factor_var
    U = fv.U
    T_eff = size(U, 1)
    p = fv.p
    F_init = fv.Y[1:p, :]
    Lambda = sdfm.method === :fglr ? sdfm.loadings_static : sdfm.loadings_td
    q = sdfm.gdfm.q
    N = size(Lambda, 1)
    order = sdfm.id_order
    length(order) == q || (order = collect(1:q))
    i_star = clamp(order[1], 1, N)
    X = sdfm.gdfm.X
    draws = zeros(T, reps, horizon, N, q)
    seeds = rand(rng, UInt64, reps)
    n_kept = 0
    max_try = stationary_only ? 10 * reps : reps
    attempt = 0
    while n_kept < reps && attempt < max_try
        attempt += 1
        local_rng = Random.MersenneTwister(attempt <= reps ? seeds[attempt] : rand(rng, UInt64))
        panel = _sdfm_one_boot_irf(sdfm, fv, U, F_init, p, T_eff, Lambda, q, N, order,
            horizon, bootstrap, block_length, wild_dist, local_rng)
        if stationary_only
            # modulus of the re-estimated VAR is checked inside; skip NaNs
            any(!isfinite, panel) && continue
        end
        for j in 1:q
            if panel[1, i_star, j] * point.values[1, i_star, j] < 0
                panel[:, :, j] .*= -one(T)
            end
        end
        n_kept += 1
        draws[n_kept, :, :, :] = panel
    end
    n_kept < reps && @warn "Only $n_kept/$reps SDFM bootstrap IRF draws obtained"
    n_use = max(n_kept, 1)
    sim = draws[1:n_use, :, :, :]
    alpha = (one(T) - conf_level) / 2
    lo = similar(point.values)
    hi = similar(point.values)
    @inbounds for h in 1:horizon, v in 1:N, s in 1:q
        d = @view sim[:, h, v, s]
        qlo = quantile(d, alpha)
        qhi = quantile(d, one(T) - alpha)
        pv = point.values[h, v, s]
        lo[h, v, s] = min(qlo, pv)
        hi[h, v, s] = max(qhi, pv)
    end
    manifest = capture_manifest(; seed=seed, settings=Dict{String,Any}(
        "ci_type" => "bootstrap", "reps" => n_use,
        "bootstrap" => String(bootstrap), "block_length" => block_length))
    ImpulseResponse{T}(point.values, lo, hi, horizon, copy(sdfm.varnames),
        copy(sdfm.shock_names), :bootstrap, sim, conf_level; manifest=manifest)
end

function _sdfm_one_boot_irf(sdfm::StructuralDFM{T}, fv::VARModel{T}, U, F_init, p, T_eff,
    Lambda, q, N, order, horizon, bootstrap, block_length, wild_dist, rng) where {T<:AbstractFloat}
    z = sdfm.instrument
    if z !== nothing && sdfm.identification === :proxy
        U_boot, z_boot = _resample_residuals_with_z(U, z, bootstrap, rng;
            block_length=block_length, wild_dist=wild_dist)
    else
        U_boot = _resample_residuals(U, bootstrap, rng; block_length=block_length, wild_dist=wild_dist)
        z_boot = nothing
    end
    F_boot = _simulate_var(F_init, fv.B, U_boot, T_eff + p)
    mstar = estimate_var(F_boot, p; check_stability=false)
    if sdfm.method === :fglr
        Kstar, _ = _rank_q_reduction(mstar.Sigma, q)
        if z_boot !== nothing
            Hstar, _, _ = _sdfm_proxy_H(Lambda, Kstar, mstar, z_boot, (order[1], 1.0), sdfm.varnames)
            B0star = Kstar * Hstar
        else
            C0 = Lambda * Kstar
            P = C0[order, :]
            Hstar = Matrix{T}(P \ safe_cholesky(P * P'))
            B0star = Kstar * Hstar
        end
        return _fglr_panel_irf(mstar, Lambda, B0star, horizon;
            X=sdfm.gdfm.X, standardized=sdfm.gdfm.standardized, units=sdfm.units)
    end
    Qstar = sdfm.identification === :cholesky ? Matrix{T}(I, q, q) : sdfm.Q
    firf = compute_irf(mstar, Qstar, horizon)
    panel = zeros(T, horizon, N, q)
    @inbounds for h in 1:horizon, j in 1:q
        panel[h, :, j] = Lambda * @view(firf[h, :, j])
    end
    panel
end

"""Pointwise median and 16/84 bands over the stored `SignIdentifiedSet`."""
function _sdfm_set_irf(sdfm::StructuralDFM{T}, horizon::Int) where {T}
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    s = sdfm.identified_set
    H_stored = size(s.irf_draws, 2)
    N = size(s.irf_draws, 3)
    q = size(s.irf_draws, 4)
    if horizon == H_stored
        draws = s.irf_draws
    else
        draws = zeros(T, s.n_accepted, horizon, N, q)
        for (i, Q) in enumerate(s.Q_draws)
            draws[i, :, :, :] = _sdfm_panel_from_Q(sdfm, Q, horizon)
        end
    end
    tmp = SignIdentifiedSet{T}(s.Q_draws, draws, s.n_accepted, s.n_total,
                               s.acceptance_rate, s.variables, s.shocks)
    med = irf_median(tmp)
    lo, hi = irf_bounds(tmp)
    ImpulseResponse{T}(med, lo, hi, horizon, copy(sdfm.varnames), copy(sdfm.shock_names),
                       :sign_set, draws, T(0.68))
end

"""Panel IRFs for one rotation `Q` at an arbitrary horizon."""
function _sdfm_panel_from_Q(sdfm::StructuralDFM{T}, Q::AbstractMatrix{T}, horizon::Int) where {T}
    if sdfm.method === :fglr
        B0 = sdfm.K * Q
        return _fglr_panel_irf(sdfm.factor_var, sdfm.loadings_static, B0, horizon;
                               X=sdfm.gdfm.X, standardized=sdfm.gdfm.standardized,
                               units=sdfm.units)
    end
    factor_irf = compute_irf(sdfm.factor_var, Q, horizon)
    N = size(sdfm.loadings_td, 1)
    q = size(factor_irf, 3)
    panel = zeros(T, horizon, N, q)
    @inbounds for h in 1:horizon, j in 1:q
        panel[h, :, j] = sdfm.loadings_td * @view(factor_irf[h, :, j])
    end
    panel
end

"""
    fevd(sdfm::StructuralDFM, horizon; space=:factor, include_idiosyncratic=false, idio_model=:white) -> FEVD

Factor-space (`space=:factor`) or panel (`space=:panel`) FEVD using the
identification stored at estimation. Panel FEVD optionally appends an
`"Idiosyncratic"` column (`include_idiosyncratic=true`).

Identification is fixed at estimation — `method` / `check_func` / `narrative_check`
are rejected.
"""
function fevd(sdfm::StructuralDFM{T}, horizon::Int;
    space::Symbol=:factor, include_idiosyncratic::Bool=false,
    idio_model::Symbol=:white, kwargs...) where {T}
    _reject_sdfm_ident_kwargs(kwargs, "fevd(::StructuralDFM)")
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    space in (:factor, :panel) || throw(ArgumentError("space must be :factor or :panel"))
    idio_model in (:white, :ar1) || throw(ArgumentError("idio_model must be :white or :ar1"))
    if space === :factor
        include_idiosyncratic && throw(ArgumentError(
            "include_idiosyncratic is only defined for space=:panel"))
        irf_vals = _sdfm_factor_structural_irf(sdfm, horizon)
        n_var = size(irf_vals, 2)
        n_shock = size(irf_vals, 3)
        decomp, props = _compute_fevd_rect(irf_vals, n_var, n_shock, horizon)
        return FEVD{T}(decomp, props, sdfm.factor_var.varnames, sdfm.shock_names)
    end
    panel = sdfm_panel_irf(sdfm, horizon).values
    N, q = size(panel, 2), size(panel, 3)
    decomp_c, props_c = _compute_fevd_rect(panel, N, q, horizon)
    if !include_idiosyncratic
        return FEVD{T}(decomp_c, props_c, copy(sdfm.varnames), copy(sdfm.shock_names))
    end
    ξ = residuals(sdfm)
    idio_var = vec(var(ξ; dims=1))
    n_s = q + 1
    decomp = zeros(T, N, n_s, horizon)
    props = zeros(T, N, n_s, horizon)
    ρ = idio_model === :ar1 ? [_ar1_coef(@view ξ[:, i]) for i in 1:N] : fill(zero(T), N)
    @inbounds for h in 1:horizon
        for i in 1:N
            decomp[i, 1:q, h] = decomp_c[i, :, h]
            if idio_model === :white
                decomp[i, n_s, h] = idio_var[i]
            else
                acc = zero(T)
                ρ2 = ρ[i]^2
                term = one(T)
                for _ in 1:h
                    acc += term
                    term *= ρ2
                end
                decomp[i, n_s, h] = idio_var[i] * acc
            end
            tot = sum(@view decomp[i, :, h])
            tot > 0 && (props[i, :, h] = decomp[i, :, h] ./ tot)
        end
    end
    shocks = vcat(sdfm.shock_names, ["Idiosyncratic"])
    FEVD{T}(decomp, props, copy(sdfm.varnames), shocks)
end

function _ar1_coef(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    n < 3 && return zero(T)
    x1 = @view x[1:n-1]
    x2 = @view x[2:n]
    d = dot(x1, x1)
    d < T(1e-12) && return zero(T)
    T(dot(x1, x2) / d)
end

# =============================================================================
# Structural DFM — historical decomposition
# =============================================================================

"""
    historical_decomposition(sdfm::StructuralDFM, horizon=effective_nobs(sdfm.factor_var);
                             space=:panel, include_idiosyncratic=true) -> HistoricalDecomposition

Panel (`space=:panel`) or factor (`space=:factor`) HD using the **stored**
rotation `Q` / `B0`. Identification kwargs are rejected.

Panel contributions are `T_eff × N × (q+1)` with an `"Idiosyncratic"` column
when `include_idiosyncratic=true`.
"""
function historical_decomposition(sdfm::StructuralDFM{T},
    horizon::Int=effective_nobs(sdfm.factor_var);
    space::Symbol=:panel, include_idiosyncratic::Bool=true, kwargs...) where {T<:AbstractFloat}
    _reject_sdfm_ident_kwargs(kwargs, "historical_decomposition(::StructuralDFM)")
    space in (:panel, :factor) || throw(ArgumentError("space must be :panel or :factor"))
    fv = sdfm.factor_var
    T_eff = effective_nobs(fv)
    horizon = min(horizon, T_eff)
    shocks = _sdfm_structural_shocks(sdfm)          # T_eff × q
    Theta = _sdfm_structural_ma(sdfm, horizon)      # Vector of r×q
    contrib_f = _compute_hd_contributions(shocks[1:horizon, :], Theta)
    actual_f = fv.Y[(fv.p + 1):end, :]
    actual_f = actual_f[1:horizon, :]
    init_f = _compute_initial_conditions(actual_f, contrib_f)
    if space === :factor
        return HistoricalDecomposition{T}(contrib_f, init_f, actual_f, shocks[1:horizon, :],
            horizon, copy(fv.varnames), copy(sdfm.shock_names), sdfm.identification)
    end
    _project_hd_to_panel(sdfm, contrib_f, init_f, shocks[1:horizon, :], horizon,
                         include_idiosyncratic)
end

"""Structural shocks ε_t from factor-VAR residuals and stored B0 (`T_eff × q`)."""
function _sdfm_structural_shocks(sdfm::StructuralDFM{T}) where {T<:AbstractFloat}
    U = sdfm.factor_var.U                 # T_eff × r
    B0 = sdfm.B0                          # r × q
    r, q = size(B0)
    if r == q
        return Matrix{T}((B0 \ U')')
    end
    Matrix{T}((robust_inv(B0' * B0) * (B0' * U'))')
end

"""Structural MA Θ_s = Ψ_s B0 (r × q) for s = 1..horizon (Ψ_1 = I)."""
function _sdfm_structural_ma(sdfm::StructuralDFM{T}, horizon::Int) where {T<:AbstractFloat}
    Psi = _ma_array(sdfm.factor_var, horizon)
    B0 = sdfm.B0
    [Matrix{T}(@view(Psi[s, :, :]) * B0) for s in 1:horizon]
end

function _project_hd_to_panel(sdfm::StructuralDFM{T}, contrib_f::Array{T,3},
    init_f::Matrix{T}, shocks::Matrix{T}, T_hd::Int, include_idiosyncratic::Bool) where {T<:AbstractFloat}
    Lambda = sdfm.method === :fglr ? sdfm.loadings_static : sdfm.loadings_td
    N = size(Lambda, 1)
    q = size(contrib_f, 3)
    n_s = include_idiosyncratic ? q + 1 : q
    contrib = zeros(T, T_hd, N, n_s)
    @inbounds for t in 1:T_hd, j in 1:q
        contrib[t, :, j] = Lambda * @view(contrib_f[t, :, j])
    end
    init = init_f * Lambda'
    X = sdfm.gdfm.X
    p = sdfm.p_var
    actual = X[(p + 1):(p + T_hd), :]
    F_eff = sdfm.factor_var.Y[(p + 1):(p + T_hd), :]
    fitted = F_eff * Lambda'                 # T × N in the PCA/GDFM factor scale
    if sdfm.method === :fglr && sdfm.gdfm.standardized && sdfm.units === :raw
        σ = max.(vec(std(X; dims=1)), T(1e-10))
        μ = vec(mean(X; dims=1))
        contrib .*= reshape(σ, 1, N, 1)
        init .*= σ'
        init .+= μ'
        fitted = fitted .* σ' .+ μ'
    end
    sn = copy(sdfm.shock_names)
    if include_idiosyncratic
        contrib[:, :, n_s] = actual - fitted
        push!(sn, "Idiosyncratic")
    end
    HistoricalDecomposition{T}(contrib, init, actual, shocks, T_hd,
        copy(sdfm.varnames), sn, sdfm.identification)
end

# =============================================================================
# Structural DFM — forecast
# =============================================================================

"""
    forecast(sdfm::StructuralDFM, h; ci_method=:none, reps=200, conf_level=0.95, rng) -> FactorForecast

Forecast the factor VAR and map to the panel through `Λ`. Bootstrap bands (when
requested) are pointwise quantiles of draws pushed through `Λ`, never mapped
interval endpoints. Identification is not re-run.
"""
function forecast(sdfm::StructuralDFM{T}, h::Int;
    ci_method::Symbol=:none, reps::Int=200, conf_level::Real=0.95,
    rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("h must be ≥ 1"))
    ci_method in (:none, :bootstrap) || throw(ArgumentError(
        "ci_method must be :none or :bootstrap, got :$ci_method"))
    Lambda = sdfm.method === :fglr ? sdfm.loadings_static : sdfm.loadings_td
    N = size(Lambda, 1)
    r = size(Lambda, 2)
    fc_f = forecast(sdfm.factor_var, h; ci_method=ci_method, reps=reps,
                    conf_level=conf_level, rng=rng)
    F_fc = fc_f.forecast
    X_fc = F_fc * Lambda'
    F_lo = zeros(T, h, r)
    F_hi = zeros(T, h, r)
    X_lo = zeros(T, h, N)
    X_hi = zeros(T, h, N)
    F_se = zeros(T, h, r)
    X_se = zeros(T, h, N)
    if ci_method === :bootstrap && fc_f._draws !== nothing
        draws = fc_f._draws                    # reps × h × r
        n_reps = size(draws, 1)
        Xd = zeros(T, n_reps, h, N)
        @inbounds for b in 1:n_reps
            Xd[b, :, :] = draws[b, :, :] * Lambda'
        end
        alpha = (one(T) - T(conf_level)) / 2
        @inbounds for hi in 1:h, j in 1:r
            d = @view draws[:, hi, j]
            F_lo[hi, j] = quantile(d, alpha)
            F_hi[hi, j] = quantile(d, one(T) - alpha)
            F_se[hi, j] = std(d)
        end
        @inbounds for hi in 1:h, j in 1:N
            d = @view Xd[:, hi, j]
            X_lo[hi, j] = quantile(d, alpha)
            X_hi[hi, j] = quantile(d, one(T) - alpha)
            X_se[hi, j] = std(d)
        end
    end
    if sdfm.gdfm.standardized && sdfm.units === :raw
        X = sdfm.gdfm.X
        μ = vec(mean(X; dims=1))
        σ = max.(vec(std(X; dims=1)), T(1e-10))
        X_fc .= X_fc .* σ' .+ μ'
        if ci_method === :bootstrap
            X_lo .= X_lo .* σ' .+ μ'
            X_hi .= X_hi .* σ' .+ μ'
            X_se .= X_se .* σ'
        end
    end
    _build_factor_forecast(F_fc, X_fc, F_lo, F_hi, X_lo, X_hi, F_se, X_se,
        h, T(conf_level), ci_method)
end

# =============================================================================
# sdfm_panel_irf — Project structural factor IRFs to panel observables
# =============================================================================

"""
    sdfm_panel_irf(sdfm::StructuralDFM, H::Int) -> ImpulseResponse{T}

Compute panel-wide structural IRFs by projecting factor-space IRFs to all N
observable variables via the time-domain loading matrix.

`irf(::StructuralDFM, H)` is this same path: factor IRFs from the stored
rotation `Q`, then `Λ * factor_irf` at each horizon.

# Arguments
- `sdfm`: Estimated Structural DFM
- `H`: IRF horizon

# Returns
`ImpulseResponse{T}` with dimensions (H, N, q) — N observable responses to q structural shocks.
"""
function sdfm_panel_irf(sdfm::StructuralDFM{T}, H::Int) where {T}
    H >= 1 || throw(ArgumentError("horizon H must be >= 1"))
    factor_irf = _sdfm_factor_structural_irf(sdfm, H)
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
- `irf_result`: Factor-space IRF (`H × r × q` under `:fglr`, `H × q × q` under `:gdfm_var`)

# Returns
`ImpulseResponse{T}` with dimensions (H, N, q).
"""
function sdfm_panel_irf(sdfm::StructuralDFM{T}, irf_result::ImpulseResponse{T}) where {T}
    n_fac = sdfm.method === :fglr ? sdfm.r : sdfm.gdfm.q
    q = sdfm.gdfm.q
    n_vars = size(irf_result.values, 2)
    n_shocks = size(irf_result.values, 3)
    H = irf_result.horizon

    n_vars == n_fac || throw(ArgumentError(
        "IRF has $n_vars variables but StructuralDFM has $n_fac factors"))
    n_shocks == q || throw(ArgumentError(
        "IRF has $n_shocks shocks but StructuralDFM has $q shocks"))

    _sdfm_project_irf(sdfm, irf_result.values, H;
        draws=irf_result._draws, ci_type=irf_result.ci_type,
        conf_level=irf_result._conf_level)
end

"""Factor-space structural IRFs: `H × r × q` (FGLR) or `H × q × q` (legacy)."""
function _sdfm_factor_structural_irf(sdfm::StructuralDFM{T}, horizon::Int) where {T}
    if sdfm.method === :fglr
        Psi = _ma_array(sdfm.factor_var, horizon)
        r, q = size(sdfm.B0)
        irf = zeros(T, horizon, r, q)
        @inbounds for h in 1:horizon
            irf[h, :, :] = @view(Psi[h, :, :]) * sdfm.B0
        end
        return irf
    end
    compute_irf(sdfm.factor_var, sdfm.Q, horizon)
end

"""Project factor-space IRF array to panel space (H x N x q)."""
function _sdfm_project_irf(sdfm::StructuralDFM{T}, factor_irf::AbstractArray{T,3}, H::Int;
    draws=nothing, ci_type::Symbol=:none, conf_level::T=zero(T)) where {T}
    q = size(factor_irf, 3)
    Lambda = sdfm.method === :fglr ? sdfm.loadings_static : sdfm.loadings_td
    N = size(Lambda, 1)
    n_fac = size(Lambda, 2)
    size(factor_irf, 2) == n_fac || throw(ArgumentError(
        "factor IRF has $(size(factor_irf, 2)) variables but loadings have $n_fac columns"))

    panel_values = zeros(T, H, N, q)
    for h in 1:H
        for j in 1:q
            panel_values[h, :, j] = Lambda * @view(factor_irf[h, :, j])
        end
    end
    scale = nothing
    if sdfm.method === :fglr && sdfm.units === :raw
        X = sdfm.gdfm.X
        # PCA loadings are on standardized data when gdfm.standardized; scale to raw
        if sdfm.gdfm.standardized
            scale = max.(vec(std(X; dims=1)), T(1e-10))
            panel_values .*= reshape(scale, 1, N, 1)
        end
    end

    panel_names = copy(sdfm.varnames)
    ci_lo = zeros(T, H, N, q)
    ci_hi = zeros(T, H, N, q)
    panel_draws = nothing
    ctype = :none
    cl = zero(T)
    if draws !== nothing && ci_type != :none
        n_reps = size(draws, 1)
        panel_draws = zeros(T, n_reps, H, N, q)
        @inbounds for rep in 1:n_reps, h in 1:H, j in 1:q
            panel_draws[rep, h, :, j] = Lambda * @view(draws[rep, h, :, j])
        end
        if scale !== nothing
            panel_draws .*= reshape(scale, 1, 1, N, 1)
        end
        alpha = (one(T) - T(conf_level)) / 2
        @inbounds for h in 1:H, v in 1:N, s in 1:q
            d = @view panel_draws[:, h, v, s]
            ci_lo[h, v, s] = quantile(d, alpha)
            ci_hi[h, v, s] = quantile(d, one(T) - alpha)
        end
        ctype = ci_type
        cl = T(conf_level)
    end

    ImpulseResponse{T}(panel_values, ci_lo, ci_hi, H, panel_names,
        sdfm.shock_names, ctype, panel_draws, cl)
end
