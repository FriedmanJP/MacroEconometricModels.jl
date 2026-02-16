# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
LP-based Forecast Error Variance Decomposition (Gorodnichenko & Lee 2019).

Implements the R²-based FEVD estimator: regress estimated LP forecast errors
on structural shocks to measure variance shares. Includes LP-A and LP-B
alternative estimators, VAR-based bootstrap bias correction, and CIs.

Reference:
Gorodnichenko, Y. & Lee, B. (2019). "Forecast Error Variance Decompositions
with Local Projections." *JBES*, 38(4), 921–933.
"""

using LinearAlgebra, Statistics, Random

# =============================================================================
# Main Entry Point
# =============================================================================

"""
    lp_fevd(slp::StructuralLP{T}, horizon::Int; kwargs...) -> LPFEVD{T}

Compute LP-based FEVD using the R²-based estimator of Gorodnichenko & Lee (2019).

At each horizon h, the share of variable i's forecast error variance due to
shock j is estimated by regressing LP forecast error residuals on structural
shock leads z_{t+h}, z_{t+h-1}, ..., z_t and computing R².

# Arguments
- `slp`: Structural LP result from `structural_lp()`
- `horizon`: Maximum FEVD horizon (capped at IRF horizon)

# Keyword Arguments
- `method`: Estimator (:r2, :lp_a, :lp_b). Default: :r2
- `bias_correct`: Apply VAR-based bootstrap bias correction. Default: true
- `n_boot`: Number of bootstrap replications. Default: 500
- `conf_level`: Confidence level for CIs. Default: 0.95
- `var_lags`: VAR lag order for bias correction (default: HQIC-selected)

# Returns
`LPFEVD{T}` with raw proportions, bias-corrected values, SEs, and CIs.

# Reference
Gorodnichenko, Y. & Lee, B. (2019). "Forecast Error Variance Decompositions
with Local Projections." *JBES*, 38(4), 921–933.
"""
function lp_fevd(slp::StructuralLP{T}, horizon::Int;
                 method::Symbol=:r2,
                 bias_correct::Bool=true,
                 n_boot::Int=500,
                 conf_level::Real=0.95,
                 var_lags::Union{Nothing,Int}=nothing) where {T<:AbstractFloat}

    @assert method ∈ (:r2, :lp_a, :lp_b) "method must be :r2, :lp_a, or :lp_b"

    n = nvars(slp)
    H = min(horizon, size(slp.irf.values, 1))
    eps_mat = slp.structural_shocks  # T_eff × n

    # Response data from underlying VAR
    p = slp.var_model.p
    Y_eff = slp.var_model.Y[(p+1):end, :]  # T_eff × n

    proportions = zeros(T, n, n, H)
    bias_corrected = zeros(T, n, n, H)
    se_arr = zeros(T, n, n, H)
    ci_lower = zeros(T, n, n, H)
    ci_upper = zeros(T, n, n, H)

    for shock in 1:n
        lp_model = slp.lp_models[shock]
        shock_eps = eps_mat[:, shock]

        for resp in 1:n
            # Step 1: Compute raw FEVD at each horizon
            raw_vals = zeros(T, H)
            for h in 1:H
                raw_vals[h] = _compute_lp_fevd_h(lp_model, shock_eps, resp, h, method)
            end
            proportions[resp, shock, :] = raw_vals

            # Step 2: Bootstrap bias correction and CIs
            if n_boot > 0
                bc, se_h, ci_lo, ci_hi = _lp_fevd_bootstrap(
                    shock_eps, Y_eff[:, resp], H, lp_model.lags,
                    raw_vals, n_boot, T(conf_level), var_lags)

                bias_corrected[resp, shock, :] = bias_correct ? bc : raw_vals
                se_arr[resp, shock, :] = se_h
                ci_lower[resp, shock, :] = ci_lo
                ci_upper[resp, shock, :] = ci_hi
            else
                bias_corrected[resp, shock, :] = raw_vals
            end
        end
    end

    LPFEVD{T}(proportions, bias_corrected, se_arr, ci_lower, ci_upper,
              method, H, n_boot, T(conf_level), bias_correct)
end

"""
    fevd(slp::StructuralLP{T}, horizon::Int; kwargs...) -> LPFEVD{T}

Compute LP-based FEVD for structural LP results. Dispatches to `lp_fevd`.

See `lp_fevd` for full documentation.
"""
fevd(slp::StructuralLP{T}, horizon::Int; kwargs...) where {T} =
    lp_fevd(slp, horizon; kwargs...)

# =============================================================================
# Per-Horizon FEVD Computation
# =============================================================================

"""Dispatch to R², LP-A, or LP-B estimator at horizon h."""
function _compute_lp_fevd_h(lp_model::LPModel{T}, shock_eps::Vector{T},
                             resp_idx::Int, h::Int, method::Symbol) where {T}
    f_hat = lp_model.residuals[h+1][:, resp_idx]
    t_start = lp_model.lags + 1

    if method == :r2
        return _lp_fevd_r2_h(f_hat, shock_eps, t_start, h)
    elseif method == :lp_a
        return _lp_fevd_lpa_h(lp_model, shock_eps, resp_idx, h)
    else  # :lp_b
        return _lp_fevd_lpb_h(f_hat, shock_eps, lp_model, resp_idx, t_start, h)
    end
end

# =============================================================================
# R² Estimator (Eq. 6 in GL2019)
# =============================================================================

"""
    _lp_fevd_r2_h(f_hat, shock, t_start, h) -> T

R² FEVD at horizon h: regress forecast errors f̂ on shock leads [z_{t+h},...,z_t].
"""
function _lp_fevd_r2_h(f_hat::Vector{T}, shock::Vector{T},
                        t_start::Int, h::Int) where {T}
    return _r2_on_shock_leads(f_hat, shock, t_start, h)
end

# =============================================================================
# LP-A Estimator (Eq. 9 in GL2019)
# =============================================================================

"""
    _lp_fevd_lpa_h(lp_model, shock, resp_idx, h) -> T

LP-A FEVD: ŝ_h = (Σ_{i=0}^{h} β̂₀^{i,LP}² σ̂_z²) / Var(f̂_{t+h|t-1}).
Uses IRF coefficients directly — no R² regression needed.
"""
function _lp_fevd_lpa_h(lp_model::LPModel{T}, shock::Vector{T},
                         resp_idx::Int, h::Int) where {T}
    # β̂₀^{i,LP} = shock coefficient at horizon i (position 2 in X)
    irf_sum_sq = zero(T)
    for i in 0:h
        irf_sum_sq += lp_model.B[i+1][2, resp_idx]^2
    end

    sigma_z_sq = var(shock)
    numerator = irf_sum_sq * sigma_z_sq

    fe_var = var(lp_model.residuals[h+1][:, resp_idx])
    fe_var < eps(T) && return zero(T)

    return clamp(numerator / fe_var, zero(T), one(T))
end

# =============================================================================
# LP-B Estimator (Eq. 10 in GL2019)
# =============================================================================

"""
    _lp_fevd_lpb_h(f_hat, shock, lp_model, resp_idx, t_start, h) -> T

LP-B FEVD: ŝ_h = numerator / (numerator + Var(ṽ_{t+h|t-1})),
where ṽ is the residual from the R²-regression (Eq. 6).
"""
function _lp_fevd_lpb_h(f_hat::Vector{T}, shock::Vector{T},
                          lp_model::LPModel{T}, resp_idx::Int,
                          t_start::Int, h::Int) where {T}
    # Numerator: same as LP-A
    irf_sum_sq = zero(T)
    for i in 0:h
        irf_sum_sq += lp_model.B[i+1][2, resp_idx]^2
    end
    sigma_z_sq = var(shock)
    numerator = irf_sum_sq * sigma_z_sq

    # Denominator: numerator + Var(ṽ) from R² regression residuals
    vtilde_var = _r2_residual_variance(f_hat, shock, t_start, h)

    denom = numerator + vtilde_var
    denom < eps(T) && return zero(T)

    return clamp(numerator / denom, zero(T), one(T))
end

# =============================================================================
# Shared R² Regression Helpers
# =============================================================================

"""
Regress f_hat on [1, z_{t+h}, z_{t+h-1}, ..., z_t] and return R².
"""
function _r2_on_shock_leads(f_hat::Vector{T}, shock::Vector{T},
                             t_start::Int, h::Int) where {T}
    T_lp = length(f_hat)

    # Guard: need more observations than regressors
    n_regressors = h + 2  # intercept + (h+1) shock values
    if T_lp <= n_regressors
        return zero(T)
    end

    # Build Z = [1, z_{t+h}, z_{t+h-1}, ..., z_t]
    Z = Matrix{T}(undef, T_lp, n_regressors)
    @inbounds for i in 1:T_lp
        t = t_start + i - 1
        Z[i, 1] = one(T)
        for j in 0:h
            Z[i, j + 2] = shock[t + h - j]
        end
    end

    # OLS
    ZtZ = Z' * Z
    ZtZ_inv = try; robust_inv(ZtZ); catch; return zero(T); end
    β = ZtZ_inv * (Z' * f_hat)
    f_fitted = Z * β

    SS_res = sum(abs2, f_hat - f_fitted)
    f_mean = mean(f_hat)
    SS_tot = sum(abs2, f_hat .- f_mean)

    SS_tot < eps(T) * T_lp && return zero(T)

    R2 = one(T) - SS_res / SS_tot
    return clamp(R2, zero(T), one(T))
end

"""
Regress f_hat on shock leads and return Var(residuals).
"""
function _r2_residual_variance(f_hat::Vector{T}, shock::Vector{T},
                                t_start::Int, h::Int) where {T}
    T_lp = length(f_hat)
    n_regressors = h + 2
    if T_lp <= n_regressors
        return var(f_hat)
    end

    Z = Matrix{T}(undef, T_lp, n_regressors)
    @inbounds for i in 1:T_lp
        t = t_start + i - 1
        Z[i, 1] = one(T)
        for j in 0:h
            Z[i, j + 2] = shock[t + h - j]
        end
    end

    ZtZ_inv = try; robust_inv(Z' * Z); catch; return var(f_hat); end
    β = ZtZ_inv * (Z' * f_hat)
    vtilde = f_hat - Z * β

    return var(vtilde)
end

# =============================================================================
# VAR-Based Bootstrap Bias Correction (Section 3.4 in GL2019)
# =============================================================================

"""
Bootstrap bias correction and CIs for LP-FEVD.

1. Fit bivariate VAR(L) on (z, y) with HQIC lag selection
2. Compute 'true' FEVD from VAR (theoretical benchmark)
3. Simulate B samples from VAR, compute LP-FEVD for each
4. Bias = mean(bootstrap FEVD) - true FEVD
5. Bias-corrected = raw - bias
6. CIs from centered bootstrap distribution (Kilian 1998)
"""
function _lp_fevd_bootstrap(shock::Vector{T}, response::Vector{T},
                             H::Int, lp_lags::Int,
                             raw_vals::Vector{T},
                             n_boot::Int, conf_level::T,
                             var_lags_opt::Union{Nothing,Int}) where {T}
    T_obs = length(shock)

    # 1. Fit bivariate VAR on w = (z, y)
    W = hcat(shock, response)

    # Select VAR lag order with HQIC
    max_p = isnothing(var_lags_opt) ? min(12, max(1, floor(Int, T_obs / 10))) : var_lags_opt
    p_var = max_p
    if isnothing(var_lags_opt)
        try
            lag_info = select_lag_order(W, max_p)
            p_var = lag_info.hqic
        catch
            p_var = min(4, max_p)
        end
    end
    p_var = max(1, p_var)

    var_model = estimate_var(W, p_var)

    # 2. Compute "true" FEVD from VAR (Cholesky, variable 2 w.r.t. shock 1)
    true_fevd = _var_theoretical_fevd(var_model, H)

    # 3. Bootstrap: simulate from VAR, compute LP-FEVD
    boot_vals = fill(T(NaN), n_boot, H)

    for b in 1:n_boot
        try
            _suppress_warnings() do
                W_sim = _simulate_from_var(var_model, T_obs)
                z_sim = W_sim[:, 1]
                y_sim = W_sim[:, 2]

                for h in 1:H
                    boot_vals[b, h] = _scalar_lp_fevd_r2(z_sim, y_sim, h, lp_lags)
                end
            end
        catch
            # Failed bootstrap draw — leave as NaN, will be filtered
            continue
        end
    end

    # 4. Compute bias, SEs, CIs
    bc = zeros(T, H)
    se_arr = zeros(T, H)
    ci_lo = zeros(T, H)
    ci_hi = zeros(T, H)
    alpha = (1 - conf_level) / 2

    for h in 1:H
        valid = filter(!isnan, @view(boot_vals[:, h]))
        if length(valid) < 10
            bc[h] = raw_vals[h]
            continue
        end

        mean_boot = mean(valid)
        bias = mean_boot - true_fevd[h]
        bc[h] = clamp(raw_vals[h] - bias, zero(T), one(T))

        se_arr[h] = std(valid)

        # Centered bootstrap CIs (Kilian 1998)
        delta = valid .- mean_boot
        q_lo = T(quantile(delta, alpha))
        q_hi = T(quantile(delta, 1 - alpha))
        ci_lo[h] = clamp(bc[h] + q_lo, zero(T), one(T))
        ci_hi[h] = clamp(bc[h] + q_hi, zero(T), one(T))
    end

    bc, se_arr, ci_lo, ci_hi
end

# =============================================================================
# VAR Simulation Helper
# =============================================================================

"""Simulate T_sim observations from a VAR model with burn-in."""
function _simulate_from_var(model::VARModel{T}, T_sim::Int;
                             burn::Int=100) where {T}
    n = nvars(model)
    p = model.p
    B = model.B       # (1+n*p) × n
    Σ = model.Sigma

    L = cholesky(Symmetric(Σ) + T(1e-10) * I(n)).L

    T_total = T_sim + burn + p
    Y = zeros(T, T_total, n)

    @inbounds for t in (p+1):T_total
        x = zeros(T, 1 + n * p)
        x[1] = one(T)
        for l in 1:p
            for v in 1:n
                x[(l-1)*n + v + 1] = Y[t-l, v]
            end
        end
        noise = L * randn(T, n)
        for v in 1:n
            Y[t, v] = dot(@view(B[:, v]), x) + noise[v]
        end
    end

    Y[(burn+p+1):end, :]
end

# =============================================================================
# Theoretical FEVD from Bivariate VAR
# =============================================================================

"""Compute theoretical FEVD of variable 2 w.r.t. shock 1 from bivariate VAR."""
function _var_theoretical_fevd(var_model::VARModel{T}, H::Int) where {T}
    irf_result = irf(var_model, H; method=:cholesky)
    _, props = _compute_fevd(irf_result.values, nvars(var_model), H)

    true_fevd = zeros(T, H)
    for h in 1:H
        true_fevd[h] = props[2, 1, h]
    end
    true_fevd
end

# =============================================================================
# Scalar LP-FEVD R² (for bootstrap on bivariate data)
# =============================================================================

"""
Compute R²-based FEVD for scalar (z, y) pair at horizon h.
Used in bootstrap bias correction.
"""
function _scalar_lp_fevd_r2(z::Vector{T}, y::Vector{T},
                              h::Int, lags::Int) where {T}
    T_obs = length(z)
    t_start = lags + 1
    t_end = T_obs - h
    T_eff = t_end - t_start + 1

    T_eff < max(lags + 2, 10) && return zero(T)

    # LP regression: y_{t+h} on [1, z_t, z_{t-1},...,z_{t-L}, y_{t-1},...,y_{t-L}]
    k = 2 + 2 * lags
    X = Matrix{T}(undef, T_eff, k)
    Y_h = Vector{T}(undef, T_eff)

    @inbounds for i in 1:T_eff
        t = t_start + i - 1
        Y_h[i] = y[t + h]
        X[i, 1] = one(T)
        X[i, 2] = z[t]
        for l in 1:lags
            X[i, 2 + l] = z[t - l]
            X[i, 2 + lags + l] = y[t - l]
        end
    end

    # OLS
    XtX = X' * X
    XtX_inv = try; robust_inv(XtX); catch; return zero(T); end
    B_lp = XtX_inv * (X' * Y_h)
    f_hat = Y_h - X * B_lp

    # R² regression on shock leads
    return _r2_on_shock_leads(f_hat, z, t_start, h)
end
