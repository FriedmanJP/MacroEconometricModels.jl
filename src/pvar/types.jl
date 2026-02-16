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
Panel VAR types for MacroEconometricModels.jl — PVARModel, PVARStability, PVARTestResult.
"""

# =============================================================================
# PVARModel
# =============================================================================

"""
    PVARModel{T} <: StatsAPI.RegressionModel

Panel VAR model estimated via GMM or FE-OLS.

Stores coefficient matrices, robust standard errors, GMM internals for specification
tests, and panel descriptors. GMM internals (instruments, weighting matrix, residuals)
are retained to support Hansen J-test, bootstrap, and Andrews-Lu MMSC without
re-estimation.

# Fields
- `Phi::Matrix{T}` — m × K coefficient matrix (rows = equations, K = m*p + n_predet + n_exog [+ 1])
- `Sigma::Matrix{T}` — m × m residual covariance (from level residuals)
- `se::Matrix{T}` — same shape as Phi (robust SEs, Windmeijer-corrected for 2-step)
- `pvalues::Matrix{T}` — same shape as Phi
- `m::Int` — number of endogenous variables
- `p::Int` — number of lags
- `n_predet::Int` — number of predetermined variables
- `n_exog::Int` — number of strictly exogenous variables
- `varnames::Vector{String}` — endogenous variable names
- `predet_names::Vector{String}` — predetermined variable names
- `exog_names::Vector{String}` — exogenous variable names
- `method::Symbol` — :fd_gmm, :system_gmm, :fe_ols
- `transformation::Symbol` — :fd, :fod, :demean
- `steps::Symbol` — :onestep, :twostep, :mstep
- `system_constant::Bool` — whether level equation includes a constant
- `n_groups::Int` — number of panel groups
- `n_periods::Int` — number of time periods (max)
- `n_obs::Int` — total effective observations
- `obs_per_group::NamedTuple{(:min,:avg,:max), Tuple{Int,Float64,Int}}`
- `instruments::Vector{Matrix{T}}` — per-group instrument matrices
- `residuals_transformed::Vector{Matrix{T}}` — per-group transformed residuals
- `weighting_matrix::Matrix{T}` — final weighting matrix W
- `n_instruments::Int` — number of moment conditions
- `data::PanelData{T}` — original panel data

# References
- Holtz-Eakin, Newey & Rosen (1988), Econometrica 56(6), 1371-1395.
- Arellano & Bond (1991), Review of Economic Studies 58(2), 277-297.
- Blundell & Bond (1998), Journal of Econometrics 87(1), 115-143.
"""
struct PVARModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    # Coefficient matrices
    Phi::Matrix{T}              # m × K
    Sigma::Matrix{T}            # m × m residual covariance
    se::Matrix{T}               # same shape as Phi
    pvalues::Matrix{T}          # same shape as Phi

    # Model specification
    m::Int                      # number of endogenous variables
    p::Int                      # number of lags
    n_predet::Int               # number of predetermined variables
    n_exog::Int                 # number of strictly exogenous variables
    varnames::Vector{String}
    predet_names::Vector{String}
    exog_names::Vector{String}

    # Estimation details
    method::Symbol              # :fd_gmm, :system_gmm, :fe_ols
    transformation::Symbol      # :fd, :fod, :demean
    steps::Symbol               # :onestep, :twostep, :mstep
    system_constant::Bool

    # Panel descriptors
    n_groups::Int
    n_periods::Int
    n_obs::Int
    obs_per_group::NamedTuple{(:min,:avg,:max), Tuple{Int,Float64,Int}}

    # GMM internals
    instruments::Vector{Matrix{T}}
    residuals_transformed::Vector{Matrix{T}}
    weighting_matrix::Matrix{T}
    n_instruments::Int

    # Original data
    data::PanelData{T}
end

# StatsAPI interface
StatsAPI.coef(m::PVARModel) = vec(m.Phi)
StatsAPI.vcov(m::PVARModel) = Diagonal(vec(m.se).^2)  # diagonal approx
StatsAPI.nobs(m::PVARModel) = m.n_obs
StatsAPI.dof(m::PVARModel) = m.m * size(m.Phi, 2)
StatsAPI.stderror(m::PVARModel) = vec(m.se)

# =============================================================================
# PVARStability
# =============================================================================

"""
    PVARStability{T} <: Any

Eigenvalue stability analysis for a PVARModel.

# Fields
- `eigenvalues::Vector{Complex{T}}` — eigenvalues of companion matrix
- `moduli::Vector{T}` — moduli |λ|
- `is_stable::Bool` — true if all |λ| < 1
"""
struct PVARStability{T<:AbstractFloat}
    eigenvalues::Vector{Complex{T}}
    moduli::Vector{T}
    is_stable::Bool
end

# =============================================================================
# PVARTestResult
# =============================================================================

"""
    PVARTestResult{T} <: StatsAPI.HypothesisTest

Specification test result for Panel VAR (Hansen J-test, etc.).

# Fields
- `test_name::String` — e.g. "Hansen J-test"
- `statistic::T` — test statistic value
- `pvalue::T` — p-value
- `df::Int` — degrees of freedom
- `n_instruments::Int` — number of instruments (moment conditions)
- `n_params::Int` — number of estimated parameters
"""
struct PVARTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    test_name::String
    statistic::T
    pvalue::T
    df::Int
    n_instruments::Int
    n_params::Int
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, model::PVARModel{T}) where {T}
    K = size(model.Phi, 2)
    method_str = model.method == :fd_gmm ? "FD-GMM" :
                 model.method == :system_gmm ? "System GMM" : "FE-OLS"
    trans_str = model.transformation == :fd ? "First-Difference" :
                model.transformation == :fod ? "Forward Orthogonal Deviations" : "Within Demeaning"
    steps_str = model.method == :fe_ols ? "—" : string(model.steps)

    spec = Any[
        "Method"          method_str;
        "Transformation"  trans_str;
        "Steps"           steps_str;
        "Endogenous vars" model.m;
        "Lags"            model.p;
        "Parameters"      model.m * K;
        "Groups"          model.n_groups;
        "Obs/group"       "$(model.obs_per_group.min)/$(round(model.obs_per_group.avg, digits=1))/$(model.obs_per_group.max)";
        "Observations"    model.n_obs;
    ]
    if model.method != :fe_ols
        spec = vcat(spec, Any["Instruments" model.n_instruments])
    end

    _pretty_table(io, spec;
        title = "Panel VAR($( model.p)) — $method_str",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    # Coefficient table per equation
    regressor_names = String[]
    for l in 1:model.p
        for v in model.varnames
            push!(regressor_names, "L$l.$v")
        end
    end
    for v in model.predet_names; push!(regressor_names, v); end
    for v in model.exog_names; push!(regressor_names, v); end
    if model.system_constant && K > model.m * model.p + model.n_predet + model.n_exog
        push!(regressor_names, "_cons")
    end
    # Pad if needed
    while length(regressor_names) < K
        push!(regressor_names, "x$(length(regressor_names)+1)")
    end

    for eq in 1:model.m
        coef_vals = T.(model.Phi[eq, :])
        se_vals = T.(model.se[eq, :])
        _coef_table(io, "Equation: $(model.varnames[eq])", regressor_names, coef_vals, se_vals; dist=:z)
    end
end

function Base.show(io::IO, s::PVARStability{T}) where {T}
    n = length(s.moduli)
    data = Matrix{Any}(undef, n, 3)
    for i in 1:n
        data[i, 1] = i
        data[i, 2] = _fmt(s.moduli[i])
        data[i, 3] = s.moduli[i] < 1 ? "Inside" : "Outside"
    end
    stable_str = s.is_stable ? "STABLE" : "UNSTABLE"
    _pretty_table(io, data;
        title = "PVAR Stability ($stable_str)",
        column_labels = ["#", "|λ|", "Unit Circle"],
        alignment = [:r, :r, :l],
    )
end

function Base.show(io::IO, t::PVARTestResult{T}) where {T}
    data = Any[
        "Test"          t.test_name;
        "Statistic"     _fmt(t.statistic);
        "P-value"       _format_pvalue(t.pvalue);
        "DF"            t.df;
        "Instruments"   t.n_instruments;
        "Parameters"    t.n_params;
    ]
    _pretty_table(io, data;
        title = t.test_name,
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end
