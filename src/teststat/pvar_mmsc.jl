# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Andrews-Lu (2001) Model and Moment Selection Criteria for Panel VAR.
"""

"""
    pvar_mmsc(model::PVARModel{T}; hq_criterion::Real=2.1) -> NamedTuple

Andrews-Lu (2001) Model and Moment Selection Criteria based on Hansen J-statistic.

MMSC_BIC  = J - (c - b) × log(n)
MMSC_AIC  = J - (c - b) × 2
MMSC_HQIC = J - Q(c - b) × log(log(n))

Lower values are preferred.

# Returns
NamedTuple `(bic, aic, hqic)` of MMSC values.

# Examples
```julia
mmsc = pvar_mmsc(model)
mmsc.bic   # MMSC-BIC value
```
"""
function pvar_mmsc(model::PVARModel{T}; hq_criterion::Real=2.1) where {T}
    j = pvar_hansen_j(model)
    K = size(model.Phi, 2)
    andrews_lu_mmsc(j.statistic, model.n_instruments, K, model.n_obs;
                    hq_criterion=hq_criterion)
end
