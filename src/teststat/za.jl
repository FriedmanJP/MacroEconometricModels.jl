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
Zivot-Andrews unit root test with endogenous structural break.
"""

using LinearAlgebra, Statistics

"""
    za_test(y; regression=:both, trim=0.15, lags=:aic, max_lags=nothing, outlier=:io) -> ZAResult

Zivot-Andrews test for unit root with endogenous structural break.

Tests H₀: y has a unit root without break against H₁: y is stationary with break.

# Arguments
- `y`: Time series vector
- `regression`: Type of break - :constant (intercept), :trend (slope), or :both
- `trim`: Trimming fraction for break search (default 0.15)
- `lags`: Number of augmenting lags, or :aic/:bic for automatic selection
- `max_lags`: Maximum lags for selection
- `outlier`: Outlier model - :io (innovational outlier) or :ao (additive outlier)

# Returns
`ZAResult` containing minimum t-statistic, break point, p-value, etc.

# Example
```julia
# Series with structural break
y = vcat(randn(100), randn(100) .+ 2)
result = za_test(y; regression=:constant)

# Additive outlier model
result_ao = za_test(y; regression=:constant, outlier=:ao)
```

# References
- Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the great crash,
  the oil-price shock, and the unit-root hypothesis. JBES, 10(3), 251-270.
"""
function za_test(y::AbstractVector{T};
                 regression::Symbol=:both,
                 trim::Real=0.15,
                 lags::Union{Int,Symbol}=:aic,
                 max_lags::Union{Int,Nothing}=nothing,
                 outlier::Symbol=:io) where {T<:AbstractFloat}

    regression ∈ (:constant, :trend, :both) ||
        throw(ArgumentError("regression must be :constant, :trend, or :both"))
    outlier ∈ (:io, :ao) ||
        throw(ArgumentError("outlier must be :io or :ao"))
    0 < trim < 0.5 || throw(ArgumentError("trim must be between 0 and 0.5"))

    n = length(y)
    n < 50 && throw(ArgumentError("Time series too short for ZA test (n=$n), need at least 50"))

    # Maximum lags
    max_p = isnothing(max_lags) ? floor(Int, 12 * (n / 100)^0.25) : max_lags

    # Trimming bounds
    start_idx = max(2, ceil(Int, trim * n))
    end_idx = min(n - 1, floor(Int, (1 - trim) * n))

    # First differences
    dy = diff(y)

    min_stat = T(Inf)
    best_break = start_idx
    best_lags = 0

    for tb in start_idx:end_idx
        if outlier == :io
            stat, p_used = _za_io_at_break(y, dy, tb, n, max_p, lags, regression, T)
        else
            stat, p_used = _za_ao_at_break(y, dy, tb, n, max_p, lags, regression, T)
        end

        if stat < min_stat
            min_stat = stat
            best_break = tb
            best_lags = p_used
        end
    end

    # Critical values and p-value
    cv = Dict{Int,T}(k => T(v) for (k, v) in ZA_CRITICAL_VALUES[regression])
    pval = za_pvalue(min_stat, regression)
    break_frac = T(best_break) / n

    ZAResult(min_stat, pval, best_break, break_frac, regression, cv, best_lags, n - 1 - best_lags)
end

"""
IO (innovational outlier) model for a single break candidate.

Regression: Δy_t = μ + β·t + θ·DU_t(tb) [+ δ·DT_t(tb)] + γ·y_{t-1} + Σδ_j·Δy_{t-j} + e_t
where DU_t = 1{t >= tb} and DT_t = max(0, t - tb + 1).

Returns (t_stat_on_gamma, lags_used).
"""
function _za_io_at_break(y::AbstractVector{T}, dy::AbstractVector{T}, tb::Int, n::Int,
                          max_p::Int, lags_spec::Union{Int,Symbol}, regression::Symbol, ::Type{T}) where {T}
    p_range = lags_spec isa Int ? [lags_spec] : collect(0:max_p)

    best_ic = T(Inf)
    best_stat = T(Inf)
    best_p = 0

    for p in p_range
        nobs = n - 1 - p
        nobs < 10 && continue

        # Dependent variable: Δy from index p+1 to end
        Y = dy[(p+1):end]

        # Regressors
        ones_col = ones(T, nobs)
        trend = T.(1:nobs)
        y_lag = y[(p+1):(n-1)]

        # Break dummies (adjusted for offset from lagged differences)
        DU = T.((p+1:n-1) .>= tb)
        DT = T.(max.(0, (p+1:n-1) .- tb .+ 1))

        X_base = if regression == :constant
            hcat(ones_col, trend, DU, y_lag)
        elseif regression == :trend
            hcat(ones_col, trend, DT, y_lag)
        else  # :both
            hcat(ones_col, trend, DU, DT, y_lag)
        end

        # Add lagged differences
        if p > 0
            dy_lags = Matrix{T}(undef, nobs, p)
            for j in 1:p
                dy_lags[:, j] = dy[(p+1-j):(n-1-j)]
            end
            X = hcat(X_base, dy_lags)
        else
            X = X_base
        end

        # OLS
        XtX = X'X
        cond(XtX) > 1e12 && continue

        B = XtX \ (X'Y)
        resid_vec = Y - X * B

        sigma2 = sum(resid_vec.^2) / (nobs - size(X, 2))

        # Index of gamma (coefficient on y_{t-1})
        gamma_idx = regression == :both ? 5 : 4

        # IC for lag selection
        if lags_spec isa Symbol
            log_ssr = log(sum(resid_vec.^2) / nobs)
            k_params = size(X, 2)
            ic = if lags_spec == :aic
                log_ssr + 2 * k_params / nobs
            else  # :bic
                log_ssr + log(nobs) * k_params / nobs
            end

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

"""
AO (additive outlier) model for a single break candidate.

Step 1: Detrend y via OLS on deterministics with break dummies → residuals e.
Step 2: ADF regression on e with pulse dummy D(TB)_t = 1{t == tb} and its lags.

Returns (t_stat_on_e_lag, lags_used).
"""
function _za_ao_at_break(y::AbstractVector{T}, dy::AbstractVector{T}, tb::Int, n::Int,
                          max_p::Int, lags_spec::Union{Int,Symbol}, regression::Symbol, ::Type{T}) where {T}
    # Step 1: Detrend y with break dummies
    ones_col = ones(T, n)
    trend = T.(1:n)
    DU = T.((1:n) .>= tb)
    DT = T.(max.(0, (1:n) .- tb .+ 1))

    Z = if regression == :constant
        hcat(ones_col, trend, DU)
    elseif regression == :trend
        hcat(ones_col, trend, DT)
    else  # :both
        hcat(ones_col, trend, DU, DT)
    end

    beta_detrend = Z \ y
    e = y - Z * beta_detrend
    de = diff(e)

    # Step 2: ADF on residuals with pulse dummy
    # D(TB)_t = 1 if t == tb, 0 otherwise
    p_range = lags_spec isa Int ? [lags_spec] : collect(0:max_p)

    best_ic = T(Inf)
    best_stat = T(Inf)
    best_p = 0

    for p in p_range
        nobs = n - 1 - p
        nobs < 10 && continue

        Y = de[(p+1):end]
        e_lag = e[(p+1):(n-1)]

        # Pulse dummy on the differenced residuals
        pulse = zeros(T, n - 1)
        if 1 <= tb - 1 <= n - 1
            pulse[tb-1] = one(T)
        end

        X_parts = [reshape(e_lag, :, 1)]

        # Add pulse dummy and its lags (p+1 pulse columns total: lag 0 through lag p)
        for j in 0:p
            idx_range = (p+1-j):(n-1-j)
            if all(1 .<= idx_range .<= length(pulse))
                push!(X_parts, reshape(pulse[idx_range], :, 1))
            end
        end

        # Add lagged differences of e
        if p > 0
            de_lags = Matrix{T}(undef, nobs, p)
            for j in 1:p
                de_lags[:, j] = de[(p+1-j):(n-1-j)]
            end
            push!(X_parts, de_lags)
        end

        X = hcat(X_parts...)

        # OLS
        XtX = X'X
        cond(XtX) > 1e12 && continue

        B = XtX \ (X'Y)
        resid_vec = Y - X * B
        sigma2 = sum(resid_vec.^2) / (nobs - size(X, 2))

        # t-stat on e_{t-1} (first column)
        if lags_spec isa Symbol
            log_ssr = log(sum(resid_vec.^2) / nobs)
            k_params = size(X, 2)
            ic = if lags_spec == :aic
                log_ssr + 2 * k_params / nobs
            else  # :bic
                log_ssr + log(nobs) * k_params / nobs
            end

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

za_test(y::AbstractVector; kwargs...) = za_test(Float64.(y); kwargs...)
