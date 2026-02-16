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
Hansen (1982) J-test for overidentifying restrictions in Panel VAR.
"""

"""
    pvar_hansen_j(model::PVARModel{T}) -> PVARTestResult{T}

Hansen (1982) J-test for overidentifying restrictions.

J = (Σ_i Z_i' e_i)' W (Σ_i Z_i' e_i) ~ χ²(q - k)

where q = number of instruments, k = number of parameters per equation.

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
    end

    J_avg = J_total / m_dim

    # Degrees of freedom: (instruments - parameters) per equation
    df = max(n_inst - K, 0)

    pval = df > 0 ? T(1 - cdf(Chisq(df), J_avg)) : one(T)

    PVARTestResult{T}("Hansen J-test", J_avg, pval, df, n_inst, K)
end
