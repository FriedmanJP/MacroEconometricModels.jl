# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
ADF unit root test with two structural breaks (Narayan & Popp 2010).
"""

using LinearAlgebra, Statistics

"""
    adf_2break_test(y; model=:level, lags=:aic, max_lags=nothing, trim=0.10) -> ADF2BreakResult

ADF unit root test with two structural breaks (Narayan & Popp 2010).

Tests H₀: unit root with two breaks against H₁: stationary with two breaks.

# Arguments
- `y`: Time series vector
- `model`: `:level` (level shifts only) or `:both` (level + trend shifts)
- `lags`: Number of augmenting lags, or `:aic`/`:bic`
- `max_lags`: Maximum lags for IC selection
- `trim`: Trimming fraction (default 0.10)

# References
- Narayan, P. K., & Popp, S. (2010). A new unit root test with two structural
  breaks in level and slope at unknown time. Journal of Applied Statistics,
  22(2), 206-233.
"""
function adf_2break_test(y::AbstractVector{T};
                          model::Symbol=:level,
                          lags::Union{Int,Symbol}=:aic,
                          max_lags::Union{Int,Nothing}=nothing,
                          trim::Real=0.10) where {T<:AbstractFloat}
    model in (:level, :both) || throw(ArgumentError("model must be :level or :both"))

    n = length(y)
    n < 50 && throw(ArgumentError("Need at least 50 observations, got $n"))

    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags

    start_idx = max(2, ceil(Int, trim * n))
    end_idx = min(n - 1, floor(Int, (1 - trim) * n))
    min_gap = model == :level ? 2 : 3

    dy = diff(y)

    min_stat = T(Inf)
    best_tb1 = start_idx
    best_tb2 = start_idx + min_gap
    best_lags_used = 0

    for tb1 in start_idx:end_idx
        for tb2 in (tb1 + min_gap):end_idx
            stat, p_used = _adf_2break_at(y, dy, tb1, tb2, n, max_p, lags, model, T)
            if stat < min_stat
                min_stat = stat
                best_tb1 = tb1
                best_tb2 = tb2
                best_lags_used = p_used
            end
        end
    end

    nobs = n - 1 - best_lags_used
    cv = _narayan_popp_cv(model, n, T)

    pval = if min_stat <= cv[1]; T(0.001)
    elseif min_stat <= cv[5]; T(0.01) + (min_stat - cv[1]) / (cv[5] - cv[1]) * T(0.04)
    elseif min_stat <= cv[10]; T(0.05) + (min_stat - cv[5]) / (cv[10] - cv[5]) * T(0.05)
    else; T(0.20)
    end

    ADF2BreakResult(min_stat, pval, best_tb1, best_tb2,
                    T(best_tb1) / n, T(best_tb2) / n, best_lags_used, model, cv, nobs)
end

function _adf_2break_at(y::AbstractVector{T}, dy::AbstractVector{T}, tb1::Int, tb2::Int,
                         n::Int, max_p::Int, lags_spec::Union{Int,Symbol},
                         model::Symbol, ::Type{T}) where {T}
    p_range = lags_spec isa Int ? [lags_spec] : collect(0:max_p)

    best_ic = T(Inf)
    best_stat = T(Inf)
    best_p = 0

    for p in p_range
        nobs = n - 1 - p
        nobs < 10 && continue

        Y = dy[(p+1):end]

        ones_col = ones(T, nobs)
        trend = T.(1:nobs)
        y_lag = y[(p+1):(n-1)]

        DU1 = T.((p+1:n-1) .>= tb1)
        DU2 = T.((p+1:n-1) .>= tb2)

        X_base = if model == :level
            hcat(ones_col, trend, DU1, DU2, y_lag)
        else
            DT1 = T.(max.(0, (p+1:n-1) .- tb1 .+ 1))
            DT2 = T.(max.(0, (p+1:n-1) .- tb2 .+ 1))
            hcat(ones_col, trend, DU1, DU2, DT1, DT2, y_lag)
        end

        if p > 0
            dy_lags = Matrix{T}(undef, nobs, p)
            for j in 1:p
                dy_lags[:, j] = dy[(p+1-j):(n-1-j)]
            end
            X = hcat(X_base, dy_lags)
        else
            X = X_base
        end

        XtX = X'X
        cond(XtX) > 1e12 && continue

        B = XtX \ (X'Y)
        resid_vec = Y - X * B
        sigma2 = sum(resid_vec.^2) / (nobs - size(X, 2))

        gamma_idx = model == :level ? 5 : 7

        if lags_spec isa Symbol
            log_ssr = log(sum(resid_vec.^2) / nobs)
            k_params = size(X, 2)
            ic = lags_spec == :aic ? log_ssr + 2 * k_params / nobs : log_ssr + log(nobs) * k_params / nobs

            if ic < best_ic
                best_ic = ic
                se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
                best_stat = se[gamma_idx] > zero(T) ? B[gamma_idx] / se[gamma_idx] : T(Inf)
                best_p = p
            end
        else
            se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
            best_stat = se[gamma_idx] > zero(T) ? B[gamma_idx] / se[gamma_idx] : T(Inf)
            best_p = p
        end
    end

    return (best_stat, best_p)
end

adf_2break_test(y::AbstractVector; kwargs...) = adf_2break_test(Float64.(y); kwargs...)
