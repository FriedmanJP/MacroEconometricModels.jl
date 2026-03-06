# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Gregory-Hansen cointegration test with structural break (Gregory & Hansen 1996).
"""

using LinearAlgebra, Statistics

"""
    gregory_hansen_test(Y; model=:C, lags=:aic, max_lags=nothing, trim=0.15) -> GregoryHansenResult

Gregory-Hansen cointegration test with structural break (Gregory & Hansen 1996).

Tests H₀: no cointegration against H₁: cointegration with a structural break.

# Arguments
- `Y`: T×n matrix where Y[:,1] is the dependent variable and Y[:,2:end] are regressors
- `model`: `:C` (level shift), `:CT` (level + trend shift), `:CS` (regime shift)
- `lags`: Number of augmenting lags for ADF, or `:aic`/`:bic`
- `max_lags`: Maximum lags for IC selection
- `trim`: Trimming fraction (default 0.15)

# References
- Gregory, A. W., & Hansen, B. E. (1996). Residual-based tests for cointegration
  in models with regime shifts. Journal of Econometrics, 70(1), 99-126.
"""
function gregory_hansen_test(Y::AbstractMatrix{T};
                              model::Symbol=:C,
                              lags::Union{Int,Symbol}=:aic,
                              max_lags::Union{Int,Nothing}=nothing,
                              trim::Real=0.15) where {T<:AbstractFloat}
    model ∈ (:C, :CT, :CS) || throw(ArgumentError("model must be :C, :CT, or :CS"))

    n_obs, n_vars = size(Y)
    n_vars >= 2 || throw(ArgumentError("Need at least 2 columns (dependent + regressor)"))
    n_obs < 50 && throw(ArgumentError("Need at least 50 observations, got $n_obs"))

    y = Y[:, 1]
    X_regs = Y[:, 2:end]
    m = n_vars - 1

    max_p = isnothing(max_lags) ? floor(Int, 12 * (n_obs / 100)^0.25) : max_lags

    start_idx = max(2, ceil(Int, trim * n_obs))
    end_idx = min(n_obs - 1, floor(Int, (1 - trim) * n_obs))

    min_adf = T(Inf)
    min_zt = T(Inf)
    min_za = T(Inf)
    best_adf_break = start_idx
    best_zt_break = start_idx
    best_za_break = start_idx

    for tb in start_idx:end_idx
        DU = T.((1:n_obs) .> tb)

        Z = if model == :C
            hcat(ones(T, n_obs), DU, X_regs)
        elseif model == :CT
            hcat(ones(T, n_obs), T.(1:n_obs), DU, X_regs)
        else  # :CS
            X_shift = X_regs .* DU
            hcat(ones(T, n_obs), DU, X_regs, X_shift)
        end

        beta = Z \ y
        resid_vec = y - Z * beta

        adf_stat, _ = _gh_adf_on_residuals(resid_vec, n_obs, max_p, lags, T)

        zt_stat, za_stat = _gh_pp_on_residuals(resid_vec, n_obs, T)

        if adf_stat < min_adf
            min_adf = adf_stat
            best_adf_break = tb
        end
        if zt_stat < min_zt
            min_zt = zt_stat
            best_zt_break = tb
        end
        if za_stat < min_za
            min_za = za_stat
            best_za_break = tb
        end
    end

    m_clamped = clamp(m, 1, 4)
    cv_key = (model, m_clamped)
    if haskey(GREGORY_HANSEN_CV, cv_key)
        adf_cv = Dict{Int,T}(k => T(v) for (k, v) in GREGORY_HANSEN_CV[cv_key][:ADF])
        za_cv = Dict{Int,T}(k => T(v) for (k, v) in GREGORY_HANSEN_CV[cv_key][:Za])
    else
        adf_cv = Dict{Int,T}(k => T(v) for (k, v) in GREGORY_HANSEN_CV[(model, 1)][:ADF])
        za_cv = Dict{Int,T}(k => T(v) for (k, v) in GREGORY_HANSEN_CV[(model, 1)][:Za])
    end

    adf_pval = _gh_pvalue(min_adf, adf_cv)
    zt_pval = _gh_pvalue(min_zt, adf_cv)
    za_pval = _gh_pvalue(min_za, za_cv)

    GregoryHansenResult(min_adf, adf_pval, min_zt, zt_pval, min_za, za_pval,
                        best_adf_break, best_zt_break, best_za_break,
                        model, m, adf_cv, za_cv, n_obs)
end

function _gh_adf_on_residuals(resid_vec::AbstractVector{T}, n::Int, max_p::Int,
                               lags_spec::Union{Int,Symbol}, ::Type{T}) where {T}
    dr = diff(resid_vec)

    p_range = lags_spec isa Int ? [lags_spec] : collect(0:max_p)
    best_ic = T(Inf)
    best_stat = T(Inf)
    best_p = 0

    for p in p_range
        nobs = n - 1 - p
        nobs < 10 && continue

        Y_dep = dr[(p+1):end]
        r_lag = resid_vec[(p+1):(n-1)]

        if p > 0
            dr_lags = Matrix{T}(undef, nobs, p)
            for j in 1:p
                dr_lags[:, j] = dr[(p+1-j):(n-1-j)]
            end
            X = hcat(r_lag, dr_lags)
        else
            X = reshape(r_lag, :, 1)
        end

        XtX = X'X
        cond(XtX) > 1e12 && continue

        B = XtX \ (X'Y_dep)
        e = Y_dep - X * B
        sigma2 = sum(e.^2) / (nobs - size(X, 2))

        if lags_spec isa Symbol
            log_ssr = log(sum(e.^2) / nobs)
            k_params = size(X, 2)
            ic = lags_spec == :aic ? log_ssr + 2*k_params/nobs : log_ssr + log(nobs)*k_params/nobs

            if ic < best_ic
                best_ic = ic
                se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
                best_stat = se[1] > zero(T) ? B[1] / se[1] : T(Inf)
                best_p = p
            end
        else
            se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
            best_stat = se[1] > zero(T) ? B[1] / se[1] : T(Inf)
            best_p = p
        end
    end

    return (best_stat, best_p)
end

function _gh_pp_on_residuals(resid_vec::AbstractVector{T}, n::Int, ::Type{T}) where {T}
    e_lag = resid_vec[1:n-1]
    e_curr = resid_vec[2:n]
    nobs = n - 1

    rho = dot(e_lag, e_curr) / dot(e_lag, e_lag)
    u = e_curr - rho * e_lag
    sigma2 = sum(u.^2) / nobs

    bw = floor(Int, 4 * (nobs / 100)^0.25)
    gamma0 = sum(u.^2) / nobs
    lrv = gamma0
    for j in 1:bw
        w = 1 - j / (bw + 1)
        gamma_j = sum(u[j+1:nobs] .* u[1:nobs-j]) / nobs
        lrv += 2 * w * gamma_j
    end

    se_rho = sqrt(sigma2 / dot(e_lag, e_lag))
    t_rho = (rho - 1) / se_rho
    zt = sqrt(sigma2 / lrv) * t_rho - (lrv - sigma2) / (2 * se_rho * sqrt(lrv) * sqrt(T(nobs)))

    za = nobs * (rho - 1) - (lrv - sigma2) * nobs^2 / (2 * dot(e_lag, e_lag))

    return (zt, za)
end

function _gh_pvalue(stat::T, cv::Dict{Int,T}) where {T}
    if stat <= cv[1]; T(0.001)
    elseif stat <= cv[5]; T(0.01) + (stat - cv[1]) / (cv[5] - cv[1]) * T(0.04)
    elseif stat <= cv[10]; T(0.05) + (stat - cv[5]) / (cv[10] - cv[5]) * T(0.05)
    else; T(0.20)
    end
end

gregory_hansen_test(Y::AbstractMatrix; kwargs...) = gregory_hansen_test(Float64.(Y); kwargs...)
