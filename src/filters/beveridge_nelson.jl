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
Beveridge-Nelson (1981) trend-cycle decomposition.

Decomposes an I(1) process into a permanent (random walk + drift) component
and a stationary transitory component using the ARIMA representation.

Reference: Beveridge, Stephen, and Charles R. Nelson. 1981.
"A New Approach to Decomposition of Economic Time Series into Permanent and
Transitory Components with Particular Attention to Measurement of the
'Business Cycle'." *Journal of Monetary Economics* 7 (2): 151–174.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    beveridge_nelson(y::AbstractVector; p=:auto, q=:auto, max_terms=500) -> BeveridgeNelsonResult

Compute the Beveridge-Nelson decomposition of time series `y`.

Assumes `y` is I(1) and decomposes it into:
```math
y_t = \\tau_t + c_t
```
where ``\\tau_t`` is a random walk with drift (permanent component)
and ``c_t`` is a stationary transitory component.

The decomposition uses the MA(∞) representation of ``\\Delta y_t``:
```math
\\Delta y_t = \\mu + \\psi(L) \\varepsilon_t
```
where the long-run multiplier ``\\psi(1) = 1 + \\sum_{j=1}^\\infty \\psi_j``.

# Arguments
- `y::AbstractVector`: Time series data (assumed I(1), length ≥ 10)

# Keywords
- `p`: AR order for ARMA model of Δy. `:auto` (default) uses `auto_arima` on Δy
- `q`: MA order for ARMA model of Δy. `:auto` (default) uses `auto_arima` on Δy
- `max_terms::Int=500`: Maximum number of ψ-weights for MA(∞) truncation

# Returns
- `BeveridgeNelsonResult{T}` with fields `permanent`, `transitory`, `drift`,
  `long_run_multiplier`, `arima_order`, `T_obs`

# Examples
```julia
# Random walk with stationary cycle
y = cumsum(randn(200)) + 0.3 * sin.(2π * (1:200) / 20)
result = beveridge_nelson(y)
result = beveridge_nelson(y; p=2, q=1)  # manual ARMA order
```

# References
- Beveridge, S., & Nelson, C. R. (1981). *JME* 7(2): 151–174.
"""
function beveridge_nelson(y::AbstractVector{T}; p=:auto, q=:auto,
                          max_terms::Int=500) where {T<:AbstractFloat}
    T_obs = length(y)
    T_obs < 10 && throw(ArgumentError("BN decomposition requires at least 10 observations, got $T_obs"))
    max_terms < 1 && throw(ArgumentError("max_terms must be positive, got $max_terms"))

    yv = Vector{T}(y)

    # First differences
    dy = diff(yv)
    drift = mean(dy)

    # Determine ARMA order for Δy (already stationary, so max_d=0)
    if p === :auto || q === :auto
        sel = auto_arima(dy; max_p=6, max_q=6, max_d=0)
        p_use = p === :auto ? ar_order(sel) : Int(p)
        q_use = q === :auto ? ma_order(sel) : Int(q)
    else
        p_use = Int(p)
        q_use = Int(q)
    end

    # Handle pure white noise case (p=0, q=0)
    if p_use == 0 && q_use == 0
        # psi(1) = 1, transitory = 0
        permanent = copy(yv)
        transitory = zeros(T, T_obs)
        return BeveridgeNelsonResult(permanent, transitory, drift, one(T),
                                     (0, 1, 0), T_obs)
    end

    # Fit ARMA model to Δy
    if p_use > 0 && q_use > 0
        arma_model = estimate_arma(dy, p_use, q_use)
        phi = arma_model.phi
        theta = arma_model.theta
        resid = arma_model.residuals
    elseif p_use > 0
        ar_model = estimate_ar(dy, p_use)
        phi = ar_model.phi
        theta = T[]
        resid = ar_model.residuals
    else
        ma_model = estimate_ma(dy, q_use)
        phi = T[]
        theta = ma_model.theta
        resid = ma_model.residuals
    end

    # Compute ψ-weights: MA(∞) representation coefficients
    psi = _compute_psi_weights(phi, theta, max_terms)

    # Long-run multiplier: ψ(1) = 1 + Σ ψ_j
    long_run = one(T) + sum(psi)

    # Cumulative ψ-star weights: ψ*_j = Σ_{i=j}^∞ ψ_i
    # (truncated at max_terms)
    psi_star = zeros(T, max_terms)
    psi_star[max_terms] = psi[max_terms]
    @inbounds for j in (max_terms - 1):-1:1
        psi_star[j] = psi[j] + psi_star[j + 1]
    end

    # Transitory component: c_t = -Σ_{j=1}^∞ ψ*_j ε_{t-j+1}
    # We only have residuals from p_use+1:end (length of resid)
    n_resid = length(resid)

    # Align residuals with the original series
    # The ARMA residuals correspond to Δy, which starts at index 2 of y
    # With p lags, residuals start at index p_use+1 of Δy = index p_use+2 of y
    transitory = zeros(T, T_obs)
    start_idx = T_obs - n_resid + 1  # index in y where residuals start

    @inbounds for t in start_idx:T_obs
        ct = zero(T)
        for j in 1:min(max_terms, t - start_idx + 1)
            resid_idx = (t - start_idx + 1) - j + 1
            if resid_idx >= 1 && resid_idx <= n_resid
                ct -= psi_star[j] * resid[resid_idx]
            end
        end
        transitory[t] = ct
    end

    # Permanent component: τ_t = y_t - c_t
    permanent = yv .- transitory

    BeveridgeNelsonResult(permanent, transitory, drift, long_run,
                           (p_use, 1, q_use), T_obs)
end

# Float64 fallback for non-float input
beveridge_nelson(y::AbstractVector; kwargs...) = beveridge_nelson(Float64.(y); kwargs...)
