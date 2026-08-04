# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Hansen (1982) J-test for overidentifying restrictions in Panel VAR.
"""

"""
    pvar_hansen_j(model::PVARModel{T}) -> PVARTestResult{T}

Hansen (1982) J-test for overidentifying restrictions.

Estimation is equation-by-equation GMM with a shared instrument matrix `Z`
(`n_instruments` columns). For each equation the Hansen statistic is

    J_eq = N · ḡ' W_opt ḡ,   ḡ = N⁻¹ Σ_i Z_i' e_{i,eq},   W_opt = (N⁻¹ Σ_i g_i g_i')⁻¹

Under the null and treating residual cross-equation correlation as negligible
(the equation-by-equation weighting already ignores it), the sum

    J = Σ_eq J_eq  ~  χ²(m · (n_instruments − K))

where `K` is the per-equation parameter count and `m` is the number of
equations. This matches Stata `xtabond2` for the single-equation (`m = 1`)
case (`df = n_instruments − K`) and extends it to multi-equation PVAR under
equation-independence. The reported `n_instruments` and `n_params` fields are
both **system** counts (`m · q` and `m · K`), so `df == n_instruments − n_params`.

!!! warning "Multi-equation reference distribution"
    For `m > 1` the χ²(m(q − K)) reference treats the per-equation J statistics as
    independent. Residuals are correlated across equations in almost any PVAR, and
    that correlation makes `Σ_eq J_eq` more variable than a χ² with `m(q − K)` df:
    the test **over-rejects** instrument validity. Read a rejection with `m > 1` as
    suggestive, or test one equation at a time.

H0: All moment conditions are valid.
H1: Some moment conditions are invalid.

# Examples
```julia
j = pvar_hansen_j(model)
j.pvalue > 0.05  # fail to reject → instruments valid
```
"""
function pvar_hansen_j(model::PVARModel{T}) where {T}
    model.method == :fe_ols && throw(ArgumentError("Hansen J-test not applicable to FE-OLS"))

    N = model.n_groups
    m_dim = model.m
    K = size(model.Phi, 2)
    n_inst = model.n_instruments

    # Compute J-statistic per equation using optimal weighting from residuals
    # J_eq = g_bar' W_opt g_bar where W_opt = inv(D_e/N), D_e = Σ_i (Z_i e_i)(Z_i e_i)'
    J_total = zero(T)
    n_eq_valid = 0

    for eq in 1:m_dim
        g_bar = zeros(T, n_inst)
        D_e = zeros(T, n_inst, n_inst)
        n_valid = 0
        for g in 1:N
            Z_g = model.instruments[g]
            E_g = model.residuals_transformed[g]
            if size(E_g, 2) >= eq && size(Z_g, 1) == size(E_g, 1) && size(Z_g, 2) == n_inst
                Ze = Z_g' * E_g[:, eq]
                g_bar .+= Ze
                D_e .+= Ze * Ze'
                n_valid += 1
            end
        end
        n_valid == 0 && continue

        # Average
        g_bar ./= n_valid
        D_e ./= n_valid

        # Optimal weighting for J-test: W = inv(D_e)
        W_opt = Matrix{T}(robust_inv(Hermitian((D_e + D_e') / 2)))
        J_eq = n_valid * (g_bar' * W_opt * g_bar)
        J_total += max(J_eq, zero(T))  # ensure non-negative
        n_eq_valid += 1
    end

    # System overidentification: sum of equation J-statistics under independence.
    # df = m * (n_instruments − K_per_eq). Matches xtabond2 when m = 1. Both reported
    # counts are system-wide so that df == n_instruments − n_params (#552).
    n_eq = max(n_eq_valid, 1)
    n_inst_system = n_inst * n_eq
    n_params_system = K * n_eq
    df = max(n_eq_valid * (n_inst - K), 0)

    pval = df > 0 ? T(1 - cdf(Chisq(df), J_total)) : one(T)

    PVARTestResult{T}("Hansen J-test", J_total, pval, df, n_inst_system, n_params_system)
end
