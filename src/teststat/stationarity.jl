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
VAR model stationarity check via companion matrix eigenvalues.
"""

using LinearAlgebra

"""
    is_stationary(model::VARModel) -> VARStationarityResult

Check if estimated VAR model is stationary.

A VAR(p) is stationary if and only if all eigenvalues of the companion matrix
have modulus strictly less than 1.

# Returns
`VARStationarityResult` with:
- `is_stationary`: Boolean indicating stationarity
- `eigenvalues`: Complex eigenvalues of companion matrix
- `max_modulus`: Maximum eigenvalue modulus
- `companion_matrix`: The (np × np) companion form matrix

# Example
```julia
model = estimate_var(Y, 2)
result = is_stationary(model)
if !result.is_stationary
    println("Warning: VAR is non-stationary, max modulus = ", result.max_modulus)
end
```
"""
function is_stationary(model::VARModel{T}) where {T}
    F = companion_matrix(model.B, nvars(model), model.p)
    eigs = eigvals(F)
    max_mod = T(maximum(abs.(eigs)))
    VARStationarityResult(max_mod < one(T), eigs, max_mod, F)
end
