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
Lag order selection for Panel VAR via Andrews-Lu MMSC criteria.
"""

"""
    pvar_lag_selection(d::PanelData{T}, max_p::Int; kwargs...) -> NamedTuple

Select optimal PVAR lag order by estimating models for p = 1, ..., max_p
and comparing Andrews-Lu MMSC criteria.

# Returns
NamedTuple with:
- `table::Matrix` — comparison table (max_p × 4: p, BIC, AIC, HQIC)
- `best_bic::Int`, `best_aic::Int`, `best_hqic::Int` — optimal lag orders
- `models::Vector{PVARModel}` — estimated models

# Examples
```julia
sel = pvar_lag_selection(pd, 4)
sel.best_bic  # optimal lag by BIC
```
"""
function pvar_lag_selection(d::PanelData{T}, max_p::Int;
                            dependent_vars::Union{Vector{String},Nothing}=nothing,
                            kwargs...) where {T}
    max_p < 1 && throw(ArgumentError("max_p must be at least 1"))

    models = Vector{PVARModel{T}}(undef, max_p)
    criteria = Matrix{T}(undef, max_p, 3)  # BIC, AIC, HQIC

    for p_try in 1:max_p
        try
            m = estimate_pvar(d, p_try; dependent_vars=dependent_vars, kwargs...)
            models[p_try] = m
            mmsc = pvar_mmsc(m)
            criteria[p_try, 1] = mmsc.bic
            criteria[p_try, 2] = mmsc.aic
            criteria[p_try, 3] = mmsc.hqic
        catch e
            # If estimation fails for this lag, set criteria to Inf
            criteria[p_try, :] .= T(Inf)
        end
    end

    # Find best
    valid = [isfinite(criteria[p, 1]) for p in 1:max_p]
    best_bic = any(valid) ? argmin(criteria[:, 1]) : 1
    best_aic = any(valid) ? argmin(criteria[:, 2]) : 1
    best_hqic = any(valid) ? argmin(criteria[:, 3]) : 1

    # Build table
    tbl = Matrix{Any}(undef, max_p, 4)
    for p_try in 1:max_p
        tbl[p_try, 1] = p_try
        tbl[p_try, 2] = isfinite(criteria[p_try, 1]) ? _fmt(criteria[p_try, 1]) : "—"
        tbl[p_try, 3] = isfinite(criteria[p_try, 2]) ? _fmt(criteria[p_try, 2]) : "—"
        tbl[p_try, 4] = isfinite(criteria[p_try, 3]) ? _fmt(criteria[p_try, 3]) : "—"
    end

    (table=tbl, best_bic=best_bic, best_aic=best_aic, best_hqic=best_hqic,
     models=models)
end
