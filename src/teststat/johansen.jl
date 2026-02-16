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

    # VECM representation: ΔYₜ = Π Yₜ₋₁ + Σᵢ Γᵢ ΔYₜ₋ᵢ + μ + εₜ
    # where Π = αβ' is the long-run matrix

    # Construct matrices
    dY = diff(Y, dims=1)  # ΔY: (T-1) × n
    Y_lag = Y[p:end-1, :]  # Y_{t-1}: (T-p) × n

    # Lagged differences
    T_eff = T_obs - p
    dY_lags = if p > 1
        hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...)
    else
        Matrix{T}(undef, T_eff, 0)
    end

    # Dependent variable
    dY_eff = dY[p:end, :]

    # Deterministic terms
    if deterministic == :none
        Z = dY_lags
    elseif deterministic == :constant
        Z = isempty(dY_lags) ? ones(T, T_eff, 1) : hcat(ones(T, T_eff), dY_lags)
    else  # :trend
        trend = T.(1:T_eff)
        Z = isempty(dY_lags) ? hcat(ones(T, T_eff), trend) : hcat(ones(T, T_eff), trend, dY_lags)
    end

    # Concentrate out short-run dynamics
    if size(Z, 2) > 0
        M = I - Z * ((Z'Z) \ Z')
        R0 = M * dY_eff   # Residuals from regressing ΔY on Z
        R1 = M * Y_lag    # Residuals from regressing Y_{t-1} on Z
    else
        R0 = dY_eff
        R1 = Y_lag
    end

    # Moment matrices
    S00 = (R0'R0) / T_eff
    S11 = (R1'R1) / T_eff
    S01 = (R0'R1) / T_eff
    S10 = S01'

    # Solve generalized eigenvalue problem
    # |λS₁₁ - S₁₀S₀₀⁻¹S₀₁| = 0
    S00_inv = inv(S00)
    A = S11 \ (S10 * S00_inv * S01)

    # Eigendecomposition
    eig = eigen(A)
    idx = sortperm(real.(eig.values), rev=true)
    eigenvalues = real.(eig.values[idx])
    eigenvectors = real.(eig.vectors[:, idx])

    # Ensure eigenvalues are in [0, 1]
    eigenvalues = clamp.(eigenvalues, 0, 1 - eps(T))

    # Test statistics
    trace_stats = Vector{T}(undef, n)
    max_eigen_stats = Vector{T}(undef, n)

    for r in 0:(n-1)
        # Trace statistic: -T Σᵢ₌ᵣ₊₁ⁿ ln(1 - λᵢ)
        trace_stats[r+1] = -T_eff * sum(log.(1 .- eigenvalues[(r+1):n]))
        # Max eigenvalue statistic: -T ln(1 - λᵣ₊₁)
        max_eigen_stats[r+1] = -T_eff * log(1 - eigenvalues[r+1])
    end

    # Critical values
    cv_trace = Matrix{T}(undef, n, 3)
    cv_max = Matrix{T}(undef, n, 3)

    for r in 0:(n-1)
        n_minus_r = n - r
        if haskey(JOHANSEN_TRACE_CV_CONSTANT, n_minus_r)
            cv_trace[r+1, :] = T.(JOHANSEN_TRACE_CV_CONSTANT[n_minus_r])
            cv_max[r+1, :] = T.(JOHANSEN_MAX_CV_CONSTANT[n_minus_r])
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
    beta = eigenvectors[:, 1:max(1, rank)]  # β: cointegrating vectors
    alpha = S01 * beta * inv(beta' * S11 * beta)  # α: adjustment coefficients

    JohansenResult(
        trace_stats, trace_pvalues,
        max_eigen_stats, max_pvalues,
        rank, beta, alpha, eigenvalues,
        cv_trace, cv_max,
        deterministic, p, T_eff
    )
end

johansen_test(Y::AbstractMatrix, p::Int; kwargs...) = johansen_test(Float64.(Y), p; kwargs...)
