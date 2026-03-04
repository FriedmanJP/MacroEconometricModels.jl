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
Johansen cointegration test for VAR systems.
"""

using LinearAlgebra

"""
    johansen_test(Y, p; deterministic=:constant) -> JohansenResult

Johansen cointegration test for VAR system.

Tests for the number of cointegrating relationships among variables using
trace and maximum eigenvalue tests.

# Arguments
- `Y`: Data matrix (T × n)
- `p`: Number of lags in the VECM representation
- `deterministic`: Specification for deterministic terms
  - :none - No deterministic terms
  - :constant - Constant in cointegrating relation (default)
  - :trend - Linear trend in levels

# Returns
`JohansenResult` containing trace and max-eigenvalue statistics, cointegrating
vectors, adjustment coefficients, and estimated rank.

# Example
```julia
# Generate cointegrated system
n, T = 3, 200
Y = randn(T, n)
Y[:, 2] = Y[:, 1] + 0.1 * randn(T)  # Y2 cointegrated with Y1

result = johansen_test(Y, 2)
result.rank  # Should detect 1 or 2 cointegrating relations
```

# References
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration
  vectors in Gaussian vector autoregressive models. Econometrica, 59(6), 1551-1580.
- Osterwald-Lenum, M. (1992). A note with quantiles of the asymptotic
  distribution of the ML cointegration rank test statistics. Oxford BEJM.
"""
function johansen_test(Y::AbstractMatrix{T}, p::Int;
                       deterministic::Symbol=:constant) where {T<:AbstractFloat}

    deterministic ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("deterministic must be :none, :constant, or :trend"))

    T_obs, n = size(Y)
    T_obs < n + p + 10 && throw(ArgumentError("Not enough observations for Johansen test"))
    p < 1 && throw(ArgumentError("Number of lags p must be at least 1"))

    # VECM representation: ΔYₜ = αβ'Yₜ₋₁ + Σᵢ Γᵢ ΔYₜ₋ᵢ + det + εₜ
    # Johansen (1991) Cases:
    #   :none    = Case 1: no deterministic terms
    #   :constant = Case 2: restricted constant in cointegrating relation
    #   :trend   = Case 4: restricted trend + unrestricted constant

    # Construct matrices
    dY = diff(Y, dims=1)  # ΔY: (T-1) × n
    Y_lag = Y[p:end-1, :]  # Y_{t-1}: (T-p) × n

    # Lagged differences
    T_eff = T_obs - p
    dY_lags = if p > 1
        hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...)
    else
        Matrix{eltype(Y)}(undef, T_eff, 0)
    end

    # Dependent variable
    dY_eff = dY[p:end, :]

    # Deterministic terms and augmented Y_lag
    # Case 1 (:none): Z = lagged diffs only; Y_lag unaugmented
    # Case 2 (:constant): constant restricted to cointegrating space (augment Y_lag)
    # Case 4 (:trend): trend restricted, constant unrestricted in Z
    # Use T_float for the element type to avoid confusion with dimensions
    T_float = T

    if deterministic == :none
        Z = dY_lags
        Y_lag_aug = Y_lag
    elseif deterministic == :constant
        # Case 2: restrict constant to cointegrating relation
        # Augment Y_lag with ones so constant enters β'[Y_{t-1}; 1]
        Z = dY_lags  # no unrestricted deterministic terms
        Y_lag_aug = hcat(Y_lag, ones(T_float, T_eff))
    else  # :trend
        # Case 4: restrict trend, keep constant unrestricted
        Z = if isempty(dY_lags)
            reshape(ones(T_float, T_eff), :, 1)
        else
            hcat(ones(T_float, T_eff), dY_lags)
        end
        Y_lag_aug = hcat(Y_lag, T_float.(1:T_eff))
    end

    # Concentrate out short-run dynamics via least squares projection
    if size(Z, 2) > 0
        R0 = dY_eff - Z * (Z \ dY_eff)
        R1 = Y_lag_aug - Z * (Z \ Y_lag_aug)
    else
        R0 = dY_eff
        R1 = Y_lag_aug
    end

    # Dimension of the augmented system (n + number of restricted deterministic terms)
    n_aug = size(Y_lag_aug, 2)

    # Moment matrices
    S00 = (R0'R0) / T_eff
    S11 = (R1'R1) / T_eff
    S01 = (R0'R1) / T_eff
    S10 = S01'

    # Solve generalized eigenvalue problem
    # |λS₁₁ - S₁₀S₀₀⁻¹S₀₁| = 0
    S00_inv = robust_inv(S00)
    A = S11 \ (S10 * S00_inv * S01)

    # Eigendecomposition (n_aug × n_aug for augmented systems)
    eig = eigen(A)
    idx = sortperm(real.(eig.values), rev=true)
    eigenvalues_all = real.(eig.values[idx])
    eigenvectors_all = real.(eig.vectors[:, idx])

    # Use only the first n eigenvalues for test statistics
    eigenvalues = clamp.(eigenvalues_all[1:n], 0, 1 - eps(T))

    # Test statistics
    trace_stats = Vector{T}(undef, n)
    max_eigen_stats = Vector{T}(undef, n)

    for r in 0:(n-1)
        # Trace statistic: -T Σᵢ₌ᵣ₊₁ⁿ ln(1 - λᵢ)
        trace_stats[r+1] = -T_eff * sum(log.(1 .- eigenvalues[(r+1):n]))
        # Max eigenvalue statistic: -T ln(1 - λᵣ₊₁)
        max_eigen_stats[r+1] = -T_eff * log(1 - eigenvalues[r+1])
    end

    # Select critical values based on deterministic specification
    cv_trace_tbl, cv_max_tbl = if deterministic == :none
        JOHANSEN_TRACE_CV_NONE, JOHANSEN_MAX_CV_NONE
    elseif deterministic == :constant
        JOHANSEN_TRACE_CV_CONSTANT, JOHANSEN_MAX_CV_CONSTANT
    else  # :trend
        JOHANSEN_TRACE_CV_TREND, JOHANSEN_MAX_CV_TREND
    end

    cv_trace = Matrix{T}(undef, n, 3)
    cv_max = Matrix{T}(undef, n, 3)

    for r in 0:(n-1)
        n_minus_r = n - r
        if haskey(cv_trace_tbl, n_minus_r)
            cv_trace[r+1, :] = T.(cv_trace_tbl[n_minus_r])
            cv_max[r+1, :] = T.(cv_max_tbl[n_minus_r])
        else
            # Extrapolate for large systems (approximate)
            cv_trace[r+1, :] = T.([6.5 + 10*n_minus_r, 8.18 + 10*n_minus_r, 11.65 + 12*n_minus_r])
            cv_max[r+1, :] = T.([6.5 + 6*n_minus_r, 8.18 + 6*n_minus_r, 11.65 + 7*n_minus_r])
        end
    end

    # P-values (approximate, based on critical value interpolation)
    trace_pvalues = Vector{T}(undef, n)
    max_pvalues = Vector{T}(undef, n)

    for r in 1:n
        # Trace test p-value
        stat = trace_stats[r]
        cv = cv_trace[r, :]
        if stat >= cv[3]
            trace_pvalues[r] = T(0.01)
        elseif stat >= cv[2]
            trace_pvalues[r] = T(0.01 + 0.04 * (cv[3] - stat) / (cv[3] - cv[2]))
        elseif stat >= cv[1]
            trace_pvalues[r] = T(0.05 + 0.05 * (cv[2] - stat) / (cv[2] - cv[1]))
        else
            trace_pvalues[r] = T(min(1.0, 0.10 + 0.40 * (cv[1] - stat) / cv[1]))
        end

        # Max eigenvalue p-value
        stat = max_eigen_stats[r]
        cv = cv_max[r, :]
        if stat >= cv[3]
            max_pvalues[r] = T(0.01)
        elseif stat >= cv[2]
            max_pvalues[r] = T(0.01 + 0.04 * (cv[3] - stat) / (cv[3] - cv[2]))
        elseif stat >= cv[1]
            max_pvalues[r] = T(0.05 + 0.05 * (cv[2] - stat) / (cv[2] - cv[1]))
        else
            max_pvalues[r] = T(min(1.0, 0.10 + 0.40 * (cv[1] - stat) / cv[1]))
        end
    end

    # Determine rank (using trace test at 5% level)
    rank = 0
    for r in 0:(n-1)
        if trace_stats[r+1] > cv_trace[r+1, 2]  # 5% critical value
            rank = r
        else
            break
        end
    end

    # Cointegrating vectors and adjustment coefficients
    # For augmented systems, extract only the n variable rows from eigenvectors
    r_eff = max(1, rank)
    beta_aug = eigenvectors_all[:, 1:r_eff]  # full augmented eigenvectors
    beta = beta_aug[1:n, :]  # β: cointegrating vectors (n × r)
    alpha = S01 * beta_aug * robust_inv(beta_aug' * S11 * beta_aug)  # α: adjustment (n × r)

    JohansenResult(
        trace_stats, trace_pvalues,
        max_eigen_stats, max_pvalues,
        rank, beta, alpha, eigenvalues,
        cv_trace, cv_max,
        deterministic, p, T_eff
    )
end

johansen_test(Y::AbstractMatrix, p::Int; kwargs...) = johansen_test(Float64.(Y), p; kwargs...)
