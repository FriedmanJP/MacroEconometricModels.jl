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
VECM Granger causality tests: short-run, long-run, and strong (joint).
"""

using LinearAlgebra, Statistics, Distributions

"""
    granger_causality_vecm(vecm, cause, effect) -> VECMGrangerResult

Test Granger causality from variable `cause` to variable `effect` in a VECM.

Three tests are computed:
1. **Short-run**: Wald test on Γ coefficients of the `cause` variable in the `effect` equation
2. **Long-run**: Wald test on α coefficients in the `effect` equation (error correction channel)
3. **Strong**: Joint test of both short-run and long-run

# Arguments
- `vecm`: Estimated `VECMModel`
- `cause`: Index of the causing variable
- `effect`: Index of the effect variable

# Returns
`VECMGrangerResult` with test statistics, p-values, and degrees of freedom.
"""
function granger_causality_vecm(vecm::VECMModel{T}, cause::Int, effect::Int) where {T}
    n = nvars(vecm)
    r = vecm.rank
    p = vecm.p

    (1 <= cause <= n) || throw(ArgumentError("cause must be in 1:$n"))
    (1 <= effect <= n) || throw(ArgumentError("effect must be in 1:$n"))
    cause == effect && throw(ArgumentError("cause and effect must be different variables"))

    T_eff = effective_nobs(vecm)

    # === Short-run test: Γ coefficients of `cause` in `effect` equation ===
    # Under H₀: all Γᵢ[effect, cause] = 0 for i = 1, ..., p-1
    n_gamma = p - 1  # number of Gamma matrices

    if n_gamma > 0
        gamma_vals = T[vecm.Gamma[i][effect, cause] for i in 1:n_gamma]

        # Reconstruct the OLS system to get the covariance of coefficients
        # Build RHS matrix (same as in estimation)
        dY = diff(vecm.Y, dims=1)
        Y_lag = vecm.Y[p:end-1, :]
        dY_lags = hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...)
        dY_eff = dY[p:end, :]

        ecm = vecm.rank > 0 ? Y_lag * vecm.beta : Matrix{T}(undef, T_eff, 0)

        has_const = vecm.deterministic ∈ (:constant, :trend)
        has_trend = vecm.deterministic == :trend

        RHS = ecm
        if size(dY_lags, 2) > 0
            RHS = hcat(RHS, dY_lags)
        end
        if has_const
            RHS = hcat(RHS, ones(T, T_eff))
        end
        if has_trend
            RHS = hcat(RHS, T.(1:T_eff))
        end

        # Covariance of OLS coefficients for `effect` equation
        RtR_inv = robust_inv(RHS'RHS)
        sigma_ee = vecm.Sigma[effect, effect]

        # Indices of Gamma[i][effect, cause] in the RHS columns:
        # RHS layout: [ecm (r cols) | dY_lag1 (n cols) | dY_lag2 (n cols) | ... | const | trend]
        gamma_indices = Int[]
        for i in 1:n_gamma
            # Column for cause variable in i-th lagged difference block
            col_idx = r + (i - 1) * n + cause
            push!(gamma_indices, col_idx)
        end

        # Wald statistic: γ' * (Σ_γγ)⁻¹ * γ ~ χ²(n_gamma)
        V_gamma = sigma_ee * RtR_inv[gamma_indices, gamma_indices]
        short_run_stat = T(gamma_vals' * robust_inv(V_gamma) * gamma_vals)
        short_run_df = n_gamma
        short_run_pvalue = T(1 - cdf(Chisq(short_run_df), max(short_run_stat, zero(T))))
    else
        short_run_stat = zero(T)
        short_run_pvalue = one(T)
        short_run_df = 0
    end

    # === Long-run test: α coefficients in `effect` equation ===
    if r > 0
        alpha_vals = vecm.alpha[effect, :]  # r values

        # Reconstruct RHS for covariance
        if !@isdefined(RHS)
            dY = diff(vecm.Y, dims=1)
            Y_lag = vecm.Y[p:end-1, :]
            dY_lags = p > 1 ? hcat([dY[(p-j):(end-j), :] for j in 1:(p-1)]...) : Matrix{T}(undef, T_eff, 0)

            ecm = Y_lag * vecm.beta
            has_const = vecm.deterministic ∈ (:constant, :trend)
            has_trend = vecm.deterministic == :trend

            RHS = ecm
            if size(dY_lags, 2) > 0
                RHS = hcat(RHS, dY_lags)
            end
            if has_const
                RHS = hcat(RHS, ones(T, T_eff))
            end
            if has_trend
                RHS = hcat(RHS, T.(1:T_eff))
            end

            RtR_inv = robust_inv(RHS'RHS)
            sigma_ee = vecm.Sigma[effect, effect]
        end

        # Alpha indices are the first r columns
        alpha_indices = collect(1:r)
        V_alpha = sigma_ee * RtR_inv[alpha_indices, alpha_indices]
        long_run_stat = T(alpha_vals' * robust_inv(V_alpha) * alpha_vals)
        long_run_df = r
        long_run_pvalue = T(1 - cdf(Chisq(long_run_df), max(long_run_stat, zero(T))))
    else
        long_run_stat = zero(T)
        long_run_pvalue = one(T)
        long_run_df = 0
    end

    # === Strong test: joint test of both ===
    strong_df = short_run_df + long_run_df
    if strong_df > 0
        if n_gamma > 0 && r > 0
            all_indices = vcat(alpha_indices, gamma_indices)
            all_vals = vcat(alpha_vals, gamma_vals)
            V_all = sigma_ee * RtR_inv[all_indices, all_indices]
            strong_stat = T(all_vals' * robust_inv(V_all) * all_vals)
        else
            strong_stat = short_run_stat + long_run_stat
        end
        strong_pvalue = T(1 - cdf(Chisq(strong_df), max(strong_stat, zero(T))))
    else
        strong_stat = zero(T)
        strong_pvalue = one(T)
    end

    VECMGrangerResult{T}(
        short_run_stat, short_run_pvalue, short_run_df,
        long_run_stat, long_run_pvalue, long_run_df,
        strong_stat, strong_pvalue, strong_df,
        cause, effect
    )
end
