# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
