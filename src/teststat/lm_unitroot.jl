# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
LM unit root test with 0, 1, or 2 structural breaks.
"""

using LinearAlgebra, Statistics

"""
    lm_unitroot_test(y; breaks=0, regression=:level, lags=:aic, max_lags=nothing, trim=0.15) -> LMUnitRootResult

LM unit root test (Schmidt-Phillips 1992; Lee-Strazicich 2003, 2013).

Tests H₀: unit root (with breaks under H₀) against H₁: stationary.

# Arguments
- `y`: Time series vector
- `breaks`: Number of structural breaks (0, 1, or 2)
- `regression`: `:level` for intercept break, `:both` for intercept + trend break
- `lags`: Number of augmenting lags, or `:aic`/`:bic`
- `max_lags`: Maximum lags for IC selection
- `trim`: Trimming fraction (default 0.15)

# References
- Schmidt, P., & Phillips, P. C. B. (1992). LM tests for a unit root in the
  presence of deterministic trends. Oxford Bulletin of Economics and Statistics,
  54(3), 257-287.
- Lee, J., & Strazicich, M. C. (2003). Minimum Lagrange multiplier unit root
  test with two structural breaks. Review of Economics and Statistics, 85(4),
  1082-1089.
- Lee, J., & Strazicich, M. C. (2013). Minimum LM unit root test with one
  structural break. Economics Bulletin, 33(4), 2483-2492.
"""
function lm_unitroot_test(y::AbstractVector{T};
                           breaks::Int=0,
                           regression::Symbol=:level,
                           lags::Union{Int,Symbol}=:aic,
                           max_lags::Union{Int,Nothing}=nothing,
                           trim::Real=0.15) where {T<:AbstractFloat}
    0 <= breaks <= 2 || throw(ArgumentError("breaks must be 0, 1, or 2"))
    regression ∈ (:level, :both) || throw(ArgumentError("regression must be :level or :both"))

    n = length(y)
    n < 50 && throw(ArgumentError("Need at least 50 observations, got $n"))

    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags

    if breaks == 0
        return _lm_unitroot_nobreak(y, n, max_p, lags, regression, T)
    elseif breaks == 1
        return _lm_unitroot_1break(y, n, max_p, lags, regression, trim, T)
    else
        return _lm_unitroot_2break(y, n, max_p, lags, regression, trim, T)
    end
end

function _lm_unitroot_nobreak(y::AbstractVector{T}, n::Int, max_p::Int,
                               lags_spec::Union{Int,Symbol}, regression::Symbol, ::Type{T}) where {T}
    if regression == :level
        Z = ones(T, n)
        delta = sum(y) / n
        S_tilde = y .- delta
    else
        Z = hcat(ones(T, n), T.(1:n))
        delta = Z \ y
        S_tilde = y - Z * delta
    end

    dS = diff(S_tilde)

    best_ic = T(Inf)
    best_stat = T(Inf)
    best_p = 0

    p_range = lags_spec isa Int ? [lags_spec] : collect(0:max_p)

    for p in p_range
        nobs = n - 1 - p
        nobs < 10 && continue

        Y = dS[(p+1):end]
        S_lag = S_tilde[(p+1):(n-1)]

        if p > 0
            dS_lags = Matrix{T}(undef, nobs, p)
            for j in 1:p
                dS_lags[:, j] = dS[(p+1-j):(n-1-j)]
            end
            X = hcat(S_lag, dS_lags)
        else
            X = reshape(S_lag, :, 1)
        end

        XtX = X'X
        cond(XtX) > 1e12 && continue

        B = XtX \ (X'Y)
        resid_vec = Y - X * B
        sigma2 = sum(resid_vec.^2) / (nobs - size(X, 2))

        if lags_spec isa Symbol
            log_ssr = log(sum(resid_vec.^2) / nobs)
            k_params = size(X, 2)
            ic = lags_spec == :aic ? log_ssr + 2 * k_params / nobs : log_ssr + log(nobs) * k_params / nobs

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

    nobs = n - 1 - best_p
    cv = _lm_unitroot_critical_values(0, nobs, best_p, T)

    pval = if best_stat <= cv[1]; T(0.001)
    elseif best_stat <= cv[5]; T(0.01) + (best_stat - cv[1]) / (cv[5] - cv[1]) * T(0.04)
    elseif best_stat <= cv[10]; T(0.05) + (best_stat - cv[5]) / (cv[10] - cv[5]) * T(0.05)
    else; T(0.20)
    end

    LMUnitRootResult(best_stat, pval, 0, Int[], T[], best_p, regression, cv, nobs)
end

function _lm_unitroot_1break(y::AbstractVector{T}, n::Int, max_p::Int,
                              lags_spec::Union{Int,Symbol}, regression::Symbol, trim::Real, ::Type{T}) where {T}
    start_idx = max(2, ceil(Int, trim * n))
    end_idx = min(n - 1, floor(Int, (1 - trim) * n))

    min_stat = T(Inf)
    best_break = start_idx
    best_lags = 0

    for tb in start_idx:end_idx
        DU = T.((1:n) .> tb)

        if regression == :level
            Z = hcat(ones(T, n), T.(1:n), DU)
        else
            DT = T.(max.(0, (1:n) .- tb))
            Z = hcat(ones(T, n), T.(1:n), DU, DT)
        end

        delta = Z \ y
        S_tilde = y - Z * delta

        dS = diff(S_tilde)
        p_range = lags_spec isa Int ? [lags_spec] : collect(0:max_p)

        best_ic_tb = T(Inf)
        best_stat_tb = T(Inf)
        best_p_tb = 0

        for p in p_range
            nobs = n - 1 - p
            nobs < 10 && continue

            Y = dS[(p+1):end]
            S_lag = S_tilde[(p+1):(n-1)]

            if p > 0
                dS_lags = Matrix{T}(undef, nobs, p)
                for j in 1:p
                    dS_lags[:, j] = dS[(p+1-j):(n-1-j)]
                end
                X = hcat(S_lag, dS_lags)
            else
                X = reshape(S_lag, :, 1)
            end

            XtX = X'X
            cond(XtX) > 1e12 && continue

            B = XtX \ (X'Y)
            resid_vec = Y - X * B
            sigma2 = sum(resid_vec.^2) / (nobs - size(X, 2))

            if lags_spec isa Symbol
                log_ssr = log(sum(resid_vec.^2) / nobs)
                k_params = size(X, 2)
                ic = lags_spec == :aic ? log_ssr + 2 * k_params / nobs : log_ssr + log(nobs) * k_params / nobs

                if ic < best_ic_tb
                    best_ic_tb = ic
                    se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
                    best_stat_tb = se[1] > zero(T) ? B[1] / se[1] : T(Inf)
                    best_p_tb = p
                end
            else
                se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
                best_stat_tb = se[1] > zero(T) ? B[1] / se[1] : T(Inf)
                best_p_tb = p
            end
        end

        if best_stat_tb < min_stat
            min_stat = best_stat_tb
            best_break = tb
            best_lags = best_p_tb
        end
    end

    nobs = n - 1 - best_lags
    cv = _lm_unitroot_critical_values(1, nobs, best_lags, T)
    break_frac = T(best_break) / n

    pval = if min_stat <= cv[1]; T(0.001)
    elseif min_stat <= cv[5]; T(0.01) + (min_stat - cv[1]) / (cv[5] - cv[1]) * T(0.04)
    elseif min_stat <= cv[10]; T(0.05) + (min_stat - cv[5]) / (cv[10] - cv[5]) * T(0.05)
    else; T(0.20)
    end

    LMUnitRootResult(min_stat, pval, 1, [best_break], [break_frac], best_lags, regression, cv, nobs)
end

function _lm_unitroot_2break(y::AbstractVector{T}, n::Int, max_p::Int,
                              lags_spec::Union{Int,Symbol}, regression::Symbol, trim::Real, ::Type{T}) where {T}
    start_idx = max(2, ceil(Int, trim * n))
    end_idx = min(n - 1, floor(Int, (1 - trim) * n))
    min_gap = max(2, ceil(Int, trim * n))

    min_stat = T(Inf)
    best_tb1 = start_idx
    best_tb2 = start_idx + min_gap
    best_lags = 0

    for tb1 in start_idx:end_idx
        for tb2 in (tb1 + min_gap):end_idx
            DU1 = T.((1:n) .> tb1)
            DU2 = T.((1:n) .> tb2)

            if regression == :level
                Z = hcat(ones(T, n), T.(1:n), DU1, DU2)
            else
                DT1 = T.(max.(0, (1:n) .- tb1))
                DT2 = T.(max.(0, (1:n) .- tb2))
                Z = hcat(ones(T, n), T.(1:n), DU1, DU2, DT1, DT2)
            end

            delta = Z \ y
            S_tilde = y - Z * delta
            dS = diff(S_tilde)

            p_range = lags_spec isa Int ? [lags_spec] : collect(0:max_p)
            best_ic_tb = T(Inf)
            best_stat_tb = T(Inf)
            best_p_tb = 0

            for p in p_range
                nobs = n - 1 - p
                nobs < 10 && continue

                Y = dS[(p+1):end]
                S_lag = S_tilde[(p+1):(n-1)]

                if p > 0
                    dS_lags = Matrix{T}(undef, nobs, p)
                    for j in 1:p
                        dS_lags[:, j] = dS[(p+1-j):(n-1-j)]
                    end
                    X = hcat(S_lag, dS_lags)
                else
                    X = reshape(S_lag, :, 1)
                end

                XtX = X'X
                cond(XtX) > 1e12 && continue

                B = XtX \ (X'Y)
                resid_vec = Y - X * B
                sigma2 = sum(resid_vec.^2) / (nobs - size(X, 2))

                if lags_spec isa Symbol
                    log_ssr = log(sum(resid_vec.^2) / nobs)
                    k_params = size(X, 2)
                    ic = lags_spec == :aic ? log_ssr + 2 * k_params / nobs : log_ssr + log(nobs) * k_params / nobs

                    if ic < best_ic_tb
                        best_ic_tb = ic
                        se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
                        best_stat_tb = se[1] > zero(T) ? B[1] / se[1] : T(Inf)
                        best_p_tb = p
                    end
                else
                    se = sqrt.(max.(sigma2 * diag(inv(XtX)), zero(T)))
                    best_stat_tb = se[1] > zero(T) ? B[1] / se[1] : T(Inf)
                    best_p_tb = p
                end
            end

            if best_stat_tb < min_stat
                min_stat = best_stat_tb
                best_tb1 = tb1
                best_tb2 = tb2
                best_lags = best_p_tb
            end
        end
    end

    nobs = n - 1 - best_lags
    cv = _lm_unitroot_critical_values(2, nobs, best_lags, T)

    pval = if min_stat <= cv[1]; T(0.001)
    elseif min_stat <= cv[5]; T(0.01) + (min_stat - cv[1]) / (cv[5] - cv[1]) * T(0.04)
    elseif min_stat <= cv[10]; T(0.05) + (min_stat - cv[5]) / (cv[10] - cv[5]) * T(0.05)
    else; T(0.20)
    end

    LMUnitRootResult(min_stat, pval, 2, [best_tb1, best_tb2],
                     [T(best_tb1)/n, T(best_tb2)/n], best_lags, regression, cv, nobs)
end

lm_unitroot_test(y::AbstractVector; kwargs...) = lm_unitroot_test(Float64.(y); kwargs...)
