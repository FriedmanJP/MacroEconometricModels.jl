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
Unified `estimate_did` dispatcher for Difference-in-Differences estimation.

Routes to internal implementations based on `method` keyword.
"""

"""
    estimate_did(pd::PanelData{T}, outcome::Union{String,Symbol},
                 treatment::Union{String,Symbol};
                 method::Symbol=:twfe,
                 leads::Int=0, horizon::Int=5,
                 covariates::Vector{String}=String[],
                 control_group::Symbol=:never_treated,
                 cluster::Symbol=:unit,
                 conf_level::Real=0.95) where {T}

Estimate a Difference-in-Differences model.

# Arguments
- `pd`: Panel data with outcome and treatment timing columns
- `outcome`: Name of outcome variable
- `treatment`: Name of treatment timing variable (contains period of first treatment;
  0 or NaN for never-treated units)
- `method`: Estimation method
  - `:twfe` -- Two-Way Fixed Effects event-study regression (default)
  - `:callaway_santanna` -- Callaway & Sant'Anna (2021) group-time ATT

# Keyword Arguments
- `leads`: Number of pre-treatment periods to estimate (default: 0)
- `horizon`: Post-treatment horizon (default: 5)
- `covariates`: Additional control variable names
- `control_group`: `:never_treated` (default) or `:not_yet_treated`
- `cluster`: SE clustering: `:unit` (default), `:time`, `:twoway`
- `conf_level`: Confidence level (default: 0.95)

# Returns
`DIDResult{T}` -- unified result type for all methods.

# Examples
```julia
# TWFE event study
did = estimate_did(pd, :gdp, :reform_year; method=:twfe, leads=3, horizon=5)
report(did)

# Callaway-Sant'Anna
did_cs = estimate_did(pd, :gdp, :reform_year; method=:callaway_santanna, leads=3, horizon=5)
plot_result(did_cs)
```

# References
- Callaway, B. & Sant'Anna, P. H. C. (2021). *JoE* 225(2), 200-230.
"""
function estimate_did(pd::PanelData{T}, outcome::Union{String,Symbol},
                      treatment::Union{String,Symbol};
                      method::Symbol=:twfe,
                      leads::Int=0, horizon::Int=5,
                      covariates::Vector{String}=String[],
                      control_group::Symbol=:never_treated,
                      cluster::Symbol=:unit,
                      conf_level::Real=0.95) where {T<:AbstractFloat}
    # Validate inputs
    outcome_col = _resolve_varindex(pd, outcome)
    treat_col = _resolve_varindex(pd, treatment)
    cov_cols = [_resolve_varindex(pd, c) for c in covariates]

    cluster in (:unit, :time, :twoway) ||
        throw(ArgumentError("cluster must be :unit, :time, or :twoway, got :$cluster"))
    control_group in (:never_treated, :not_yet_treated) ||
        throw(ArgumentError("control_group must be :never_treated or :not_yet_treated"))
    horizon >= 0 || throw(ArgumentError("horizon must be non-negative"))
    leads >= 0 || throw(ArgumentError("leads must be non-negative"))

    if method == :twfe
        _estimate_twfe(pd, outcome_col, treat_col;
                       leads=leads, horizon=horizon,
                       covariate_cols=cov_cols,
                       control_group=control_group,
                       cluster=cluster, conf_level=conf_level)
    elseif method == :callaway_santanna
        _estimate_callaway_santanna(pd, outcome_col, treat_col;
                                    leads=leads, horizon=horizon,
                                    control_group=control_group,
                                    cluster=cluster, conf_level=conf_level)
    else
        throw(ArgumentError("Unknown DiD method :$method. Available: :twfe, :callaway_santanna"))
    end
end


