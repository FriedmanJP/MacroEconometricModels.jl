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
Bridge equation combination nowcasting (Bańbura et al. 2023).

Combines multiple OLS bridge regressions (using pairs of monthly indicators
to predict quarterly GDP) via median combination.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1, lagY=1) -> NowcastBridge{T}

Estimate bridge equation combination nowcast.

Each bridge equation uses a pair of monthly indicators (aggregated to quarterly)
plus optional lagged quarterly variables and autoregressive terms to predict
the target quarterly variable (last column).

# Arguments
- `Y::AbstractMatrix` — T_obs × N data matrix (NaN for missing)
- `nM::Int` — number of monthly variables
- `nQ::Int` — number of quarterly variables (last `nQ` columns)

# Keyword Arguments
- `lagM::Int=1` — lags for monthly indicators (after quarterly aggregation)
- `lagQ::Int=1` — lags for quarterly indicators
- `lagY::Int=1` — autoregressive lags for target variable

# Returns
`NowcastBridge{T}` with combined and individual nowcasts.

# References
- Bańbura, M., Belousova, I., Bodnár, K. & Tóth, M. B. (2023).
  Nowcasting Employment in the Euro Area. ECB Working Paper No 2815.
"""
function nowcast_bridge(Y::AbstractMatrix, nM::Int, nQ::Int;
                        lagM::Int=1, lagQ::Int=1, lagY::Int=1)
    T_obs, N = size(Y)
    N == nM + nQ || throw(ArgumentError("nM ($nM) + nQ ($nQ) must equal N ($N)"))
    nQ >= 1 || throw(ArgumentError("nQ must be >= 1"))
    lagM >= 0 || throw(ArgumentError("lagM must be >= 0"))

    Tf = eltype(Y) <: AbstractFloat ? eltype(Y) : Float64
    Ymat = Matrix{Tf}(Y)

    # Fill missing monthly values via forward-fill + interpolation
    X_sm = _bridge_fill_monthly(Ymat, nM)

    # Aggregate monthly to quarterly (3-month averages)
    n_quarters = T_obs ÷ 3
    Xm_q = _bridge_m2q(X_sm[:, 1:nM], n_quarters)  # n_quarters × nM

    # Quarterly variables (sampled every 3rd month)
    Xq = zeros(Tf, n_quarters, nQ)
    for q in 1:n_quarters
        t_end = q * 3
        if t_end <= T_obs
            Xq[q, :] = X_sm[t_end, (nM + 1):N]
        end
    end

    # Target variable: last quarterly variable
    Y_target = Xq[:, nQ]

    # Generate bridge equation combinations
    # Each equation: pair of monthly + optional quarterly regressors
    equations = _bridge_combinations(nM, nQ - 1)  # exclude target from quarterly
    n_equations = size(equations, 1)

    # Estimate each bridge equation
    max_lag = max(lagM, lagQ, lagY)
    t_start = max_lag + 1
    t_end_est = n_quarters

    # Find last non-NaN quarter for target
    while t_end_est >= t_start && isnan(Y_target[t_end_est])
        t_end_est -= 1
    end

    coefficients = Vector{Vector{Tf}}(undef, n_equations)
    Y_individual = fill(Tf(NaN), n_quarters, n_equations)

    for eq in 1:n_equations
        m1, m2 = equations[eq, 1], equations[eq, 2]

        # Build regressor matrix for this equation
        X_eq, y_eq, valid_range = _bridge_build_equation(
            Xm_q, Xq, Y_target, m1, m2, nQ, lagM, lagQ, lagY,
            t_start, t_end_est)

        if length(valid_range) < 3 || size(X_eq, 1) < size(X_eq, 2) + 1
            coefficients[eq] = zeros(Tf, 0)
            continue
        end

        # OLS with robust inversion to handle near-singular designs
        XtX = X_eq' * X_eq
        b = try
            XtX_reg = XtX + Tf(1e-6) * I(size(X_eq, 2))
            XtX_reg \ (X_eq' * y_eq)
        catch
            # Fall back to robust_inv for numerically difficult cases
            robust_inv(XtX + Tf(1e-4) * I(size(X_eq, 2))) * (X_eq' * y_eq)
        end
        coefficients[eq] = b

        # In-sample and out-of-sample predictions
        for q in t_start:n_quarters
            X_q_pred, _, _ = _bridge_build_equation(
                Xm_q, Xq, Y_target, m1, m2, nQ, lagM, lagQ, lagY,
                q, q)
            if !isempty(X_q_pred) && size(X_q_pred, 2) == length(b)
                Y_individual[q, eq] = dot(X_q_pred[1, :], b)
            end
        end
    end

    # Combine via median across equations (robust to outliers)
    Y_nowcast = zeros(Tf, n_quarters)
    for q in 1:n_quarters
        preds = filter(!isnan, Y_individual[q, :])
        Y_nowcast[q] = isempty(preds) ? Tf(NaN) : median(preds)
    end

    NowcastBridge{Tf}(X_sm, Y_nowcast, Y_individual, n_equations,
                      coefficients, nM, nQ, lagM, lagQ, lagY, Ymat)
end

# =============================================================================
# Helpers
# =============================================================================

"""Fill missing monthly values via forward-fill and linear interpolation."""
function _bridge_fill_monthly(Y::Matrix{T}, nM::Int) where {T<:AbstractFloat}
    X = copy(Y)
    N = size(Y, 2)
    T_obs = size(Y, 1)

    for j in 1:N
        # Linear interpolation for interior gaps
        for i in 1:T_obs
            if isnan(X[i, j])
                lo = i - 1
                while lo >= 1 && isnan(X[lo, j])
                    lo -= 1
                end
                hi = i + 1
                while hi <= T_obs && isnan(X[hi, j])
                    hi += 1
                end
                if lo >= 1 && hi <= T_obs
                    X[i, j] = X[lo, j] + (X[hi, j] - X[lo, j]) * T(i - lo) / T(hi - lo)
                elseif lo >= 1
                    X[i, j] = X[lo, j]  # forward fill
                elseif hi <= T_obs
                    X[i, j] = X[hi, j]  # backward fill
                else
                    valid = filter(!isnan, Y[:, j])
                    X[i, j] = isempty(valid) ? zero(T) : mean(valid)
                end
            end
        end
    end
    return X
end

"""Aggregate monthly data to quarterly (3-month average)."""
function _bridge_m2q(Xm::Matrix{T}, n_quarters::Int) where {T<:AbstractFloat}
    nM = size(Xm, 2)
    Xq = zeros(T, n_quarters, nM)
    for q in 1:n_quarters
        t_start = (q - 1) * 3 + 1
        t_end = min(q * 3, size(Xm, 1))
        n_months = t_end - t_start + 1
        Xq[q, :] = vec(mean(Xm[t_start:t_end, :], dims=1))
    end
    return Xq
end

"""Generate bridge equation combinations: all pairs of monthly indicators."""
function _bridge_combinations(nM::Int, nQ_other::Int)
    combos = Matrix{Int}(undef, 0, 2)
    # All pairs of monthly indicators
    for i in 1:nM
        for j in (i + 1):nM
            combos = vcat(combos, [i j])
        end
    end
    # Also univariate equations
    for i in 1:nM
        combos = vcat(combos, [i i])
    end
    return combos
end

"""Build regressor matrix for a single bridge equation."""
function _bridge_build_equation(Xm_q::Matrix{T}, Xq::Matrix{T}, Y_target::Vector{T},
                                 m1::Int, m2::Int, nQ::Int,
                                 lagM::Int, lagQ::Int, lagY::Int,
                                 t_start::Int, t_end::Int) where {T<:AbstractFloat}
    n_quarters = length(Y_target)
    valid = collect(t_start:t_end)

    # Filter out quarters with NaN target
    valid = filter(t -> t <= n_quarters && !isnan(Y_target[t]), valid)
    isempty(valid) && return Matrix{T}(undef, 0, 0), T[], Int[]

    n_obs = length(valid)
    regressors = ones(T, n_obs, 1)  # intercept

    # Monthly indicator lags
    nM_cols = size(Xm_q, 2)
    for lag in 0:lagM
        for m in [m1, m2]
            if m <= nM_cols
                col = zeros(T, n_obs)
                for (i, t) in enumerate(valid)
                    t_lag = t - lag
                    if 1 <= t_lag <= size(Xm_q, 1)
                        col[i] = Xm_q[t_lag, m]
                    end
                end
                regressors = hcat(regressors, col)
            end
        end
    end

    # Quarterly indicator lags (excluding target)
    nQ_other = nQ - 1
    for lag in 1:lagQ
        for q in 1:nQ_other
            col = zeros(T, n_obs)
            for (i, t) in enumerate(valid)
                t_lag = t - lag
                if 1 <= t_lag <= size(Xq, 1)
                    col[i] = Xq[t_lag, q]
                end
            end
            regressors = hcat(regressors, col)
        end
    end

    # Autoregressive lags
    for lag in 1:lagY
        col = zeros(T, n_obs)
        for (i, t) in enumerate(valid)
            t_lag = t - lag
            if 1 <= t_lag <= n_quarters && !isnan(Y_target[t_lag])
                col[i] = Y_target[t_lag]
            end
        end
        regressors = hcat(regressors, col)
    end

    y = T[Y_target[t] for t in valid]
    return regressors, y, valid
end
