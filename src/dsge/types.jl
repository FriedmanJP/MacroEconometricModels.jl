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
Type definitions for DSGE models — specification, linearized form, solution, and estimation.
"""

using LinearAlgebra

# =============================================================================
# DSGESpec — parsed model specification from @dsge macro
# =============================================================================

"""
    DSGESpec{T}

Parsed DSGE model specification. Created by the `@dsge` macro.

Fields:
- `endog::Vector{Symbol}` — endogenous variable names
- `exog::Vector{Symbol}` — exogenous shock names
- `params::Vector{Symbol}` — parameter names
- `param_values::Dict{Symbol,T}` — calibrated parameter values
- `equations::Vector{Expr}` — raw Julia equation expressions
- `residual_fns::Vector{Function}` — callable `f(y_t, y_lag, y_lead, ε, θ) → scalar`
- `n_endog::Int` — number of endogenous variables
- `n_exog::Int` — number of exogenous shocks
- `n_params::Int` — number of parameters
- `n_expect::Int` — number of expectation errors (forward-looking variables)
- `forward_indices::Vector{Int}` — indices of equations with `[t+1]` terms
- `steady_state::Vector{T}` — steady state values
- `varnames::Vector{String}` — display names
- `ss_fn::Union{Nothing, Function}` — optional analytical steady-state function `θ → y_ss`
"""
struct DSGESpec{T<:AbstractFloat}
    endog::Vector{Symbol}
    exog::Vector{Symbol}
    params::Vector{Symbol}
    param_values::Dict{Symbol,T}
    equations::Vector{Expr}
    residual_fns::Vector{Function}
    n_endog::Int
    n_exog::Int
    n_params::Int
    n_expect::Int
    forward_indices::Vector{Int}
    steady_state::Vector{T}
    varnames::Vector{String}
    ss_fn::Union{Nothing, Function}

    function DSGESpec{T}(endog, exog, params, param_values, equations, residual_fns,
                         n_expect, forward_indices, steady_state,
                         ss_fn::Union{Nothing, Function}=nothing) where {T<:AbstractFloat}
        n_endog = length(endog)
        n_exog = length(exog)
        n_params = length(params)
        @assert length(equations) == n_endog "Need as many equations as endogenous variables"
        @assert length(residual_fns) == n_endog
        @assert length(forward_indices) == n_expect
        varnames = [string(s) for s in endog]
        new{T}(endog, exog, params, param_values, equations, residual_fns,
               n_endog, n_exog, n_params, n_expect, forward_indices, steady_state, varnames, ss_fn)
    end
end

function Base.show(io::IO, spec::DSGESpec{T}) where {T}
    spec_data = Any[
        "Endogenous"    spec.n_endog;
        "Exogenous"     spec.n_exog;
        "Parameters"    spec.n_params;
        "Equations"     length(spec.equations);
        "Forward-looking" spec.n_expect;
        "Steady state"  isempty(spec.steady_state) ? "Not computed" : "Computed"
    ]
    _pretty_table(io, spec_data;
        title = "DSGE Model Specification",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# LinearDSGE — canonical form Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t
# =============================================================================

"""
    LinearDSGE{T}

Linearized DSGE in Sims canonical form: `Γ₀·y_t = Γ₁·y_{t-1} + C + Ψ·ε_t + Π·η_t`.

Fields:
- `Gamma0::Matrix{T}` — n × n coefficient on y_t
- `Gamma1::Matrix{T}` — n × n coefficient on y_{t-1}
- `C::Vector{T}` — n × 1 constants
- `Psi::Matrix{T}` — n × n_shocks shock loading
- `Pi::Matrix{T}` — n × n_expect expectation error selection
- `spec::DSGESpec{T}` — back-reference to specification
"""
struct LinearDSGE{T<:AbstractFloat}
    Gamma0::Matrix{T}
    Gamma1::Matrix{T}
    C::Vector{T}
    Psi::Matrix{T}
    Pi::Matrix{T}
    spec::DSGESpec{T}

    function LinearDSGE{T}(Gamma0, Gamma1, C, Psi, Pi, spec) where {T<:AbstractFloat}
        n = spec.n_endog
        @assert size(Gamma0) == (n, n) "Gamma0 must be n×n"
        @assert size(Gamma1) == (n, n) "Gamma1 must be n×n"
        @assert length(C) == n "C must be length n"
        @assert size(Psi, 1) == n "Psi must have n rows"
        @assert size(Pi, 1) == n "Pi must have n rows"
        new{T}(Gamma0, Gamma1, C, Psi, Pi, spec)
    end
end

function Base.show(io::IO, ld::LinearDSGE{T}) where {T}
    n = ld.spec.n_endog
    spec_data = Any[
        "State dimension"   n;
        "Shocks"            size(ld.Psi, 2);
        "Expectation errors" size(ld.Pi, 2);
        "rank(Γ₀)"         rank(ld.Gamma0);
    ]
    _pretty_table(io, spec_data;
        title = "Linearized DSGE — Canonical Form",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# DSGESolution — RE solution y_t = G1·y_{t-1} + impact·ε_t + C_sol
# =============================================================================

"""
    DSGESolution{T}

Rational expectations solution: `y_t = G1·y_{t-1} + impact·ε_t + C_sol`.

Fields:
- `G1::Matrix{T}` — n × n state transition matrix
- `impact::Matrix{T}` — n × n_shocks impact matrix
- `C_sol::Vector{T}` — n × 1 constants
- `eu::Vector{Int}` — [existence, uniqueness]: 1=yes, 0=no, -1=indeterminate
- `method::Symbol` — `:gensys` or `:blanchard_kahn`
- `eigenvalues::Vector{ComplexF64}` — generalized eigenvalues from QZ
- `spec::DSGESpec{T}` — model specification
- `linear::LinearDSGE{T}` — linearized form
"""
struct DSGESolution{T<:AbstractFloat}
    G1::Matrix{T}
    impact::Matrix{T}
    C_sol::Vector{T}
    eu::Vector{Int}
    method::Symbol
    eigenvalues::Vector{ComplexF64}
    spec::DSGESpec{T}
    linear::LinearDSGE{T}
end

# Accessors
nvars(sol::DSGESolution) = sol.spec.n_endog
nshocks(sol::DSGESolution) = sol.spec.n_exog
is_determined(sol::DSGESolution) = sol.eu[1] == 1 && sol.eu[2] == 1
is_stable(sol::DSGESolution) = maximum(abs.(eigvals(sol.G1))) < 1.0

function Base.show(io::IO, sol::DSGESolution{T}) where {T}
    n = nvars(sol)
    n_stable = count(x -> abs(x) < 1.0, sol.eigenvalues)
    n_unstable = length(sol.eigenvalues) - n_stable
    exist_str = sol.eu[1] == 1 ? "Yes" : "No"
    unique_str = sol.eu[2] == 1 ? "Yes" : "No"
    max_eig = maximum(abs.(eigvals(sol.G1)))

    spec_data = Any[
        "Variables"        n;
        "Shocks"           nshocks(sol);
        "Method"           string(sol.method);
        "Existence"        exist_str;
        "Uniqueness"       unique_str;
        "Stable eigenvalues"   n_stable;
        "Unstable eigenvalues" n_unstable;
        "Max |eigenvalue(G1)|" _fmt(max_eig);
    ]
    _pretty_table(io, spec_data;
        title = "DSGE Solution",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# PerfectForesightPath
# =============================================================================

"""
    PerfectForesightPath{T}

Deterministic perfect foresight path.

Fields:
- `path::Matrix{T}` — T_periods × n_endog level values
- `deviations::Matrix{T}` — T_periods × n_endog deviations from SS
- `converged::Bool` — Newton convergence flag
- `iterations::Int` — Newton iterations used
- `spec::DSGESpec{T}` — model specification
"""
struct PerfectForesightPath{T<:AbstractFloat}
    path::Matrix{T}
    deviations::Matrix{T}
    converged::Bool
    iterations::Int
    spec::DSGESpec{T}
end

function Base.show(io::IO, pf::PerfectForesightPath{T}) where {T}
    spec_data = Any[
        "Variables"   pf.spec.n_endog;
        "Periods"     size(pf.path, 1);
        "Converged"   pf.converged ? "Yes" : "No";
        "Iterations"  pf.iterations;
    ]
    _pretty_table(io, spec_data;
        title = "Perfect Foresight Path",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# DSGEEstimation — GMM estimation result
# =============================================================================

"""
    DSGEEstimation{T} <: AbstractDSGEModel

DSGE model estimated via GMM (IRF matching or Euler equation moments).

Fields:
- `theta::Vector{T}` — estimated deep parameters
- `vcov::Matrix{T}` — asymptotic covariance matrix
- `param_names::Vector{Symbol}` — names of estimated parameters
- `method::Symbol` — `:irf_matching` or `:euler_gmm`
- `J_stat::T` — Hansen J-test statistic
- `J_pvalue::T` — J-test p-value
- `solution::DSGESolution{T}` — solution at estimated parameters
- `converged::Bool` — optimization convergence
- `spec::DSGESpec{T}` — model specification
"""
struct DSGEEstimation{T<:AbstractFloat} <: AbstractDSGEModel
    theta::Vector{T}
    vcov::Matrix{T}
    param_names::Vector{Symbol}
    method::Symbol
    J_stat::T
    J_pvalue::T
    solution::DSGESolution{T}
    converged::Bool
    spec::DSGESpec{T}

    function DSGEEstimation{T}(theta, vcov, param_names, method, J_stat, J_pvalue,
                                solution, converged, spec) where {T<:AbstractFloat}
        @assert length(theta) == length(param_names)
        @assert size(vcov) == (length(theta), length(theta))
        @assert method ∈ (:irf_matching, :euler_gmm, :smm, :analytical_gmm)
        new{T}(theta, vcov, param_names, method, J_stat, J_pvalue, solution, converged, spec)
    end
end

# StatsAPI interface
StatsAPI.coef(m::DSGEEstimation) = m.theta
StatsAPI.vcov(m::DSGEEstimation) = m.vcov
StatsAPI.dof(m::DSGEEstimation) = length(m.theta)
StatsAPI.islinear(::DSGEEstimation) = false
StatsAPI.stderror(m::DSGEEstimation) = sqrt.(max.(diag(m.vcov), zero(eltype(m.theta))))

function Base.show(io::IO, est::DSGEEstimation{T}) where {T}
    spec_data = Any[
        "Parameters"    length(est.theta);
        "Method"        string(est.method);
        "J-statistic"   _fmt(est.J_stat);
        "J p-value"     _format_pvalue(est.J_pvalue);
        "Converged"     est.converged ? "Yes" : "No";
        "Determined"    is_determined(est.solution) ? "Yes" : "No";
    ]
    _pretty_table(io, spec_data;
        title = "DSGE Estimation — GMM",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    # Coefficient table
    se = stderror(est)
    pnames = [string(s) for s in est.param_names]
    _coef_table(io, "Estimated Parameters", pnames, est.theta, se; dist=:z)
    _sig_legend(io)
end

# =============================================================================
# OccBin types — occasionally binding constraint solver
# =============================================================================

"""
    OccBinConstraint{T}

A single occasionally binding constraint for OccBin piecewise-linear solution.

Fields:
- `expr::Expr` — full constraint expression (e.g., `:(R >= 0)`)
- `variable::Symbol` — constrained variable name
- `bound::T` — constraint bound value
- `direction::Symbol` — `:geq` or `:leq`
- `bind_expr::Expr` — expression substituted when the constraint binds
"""
struct OccBinConstraint{T<:AbstractFloat}
    expr::Expr
    variable::Symbol
    bound::T
    direction::Symbol
    bind_expr::Expr
end

"""
    OccBinRegime{T}

Linearized coefficient matrices for one regime (binding or slack).

Fields:
- `A::Matrix{T}` — coefficient on `y[t+1]` (expectation terms)
- `B::Matrix{T}` — coefficient on `y[t]` (contemporaneous terms)
- `C::Matrix{T}` — coefficient on `y[t-1]` (lagged terms)
- `D::Matrix{T}` — coefficient on `ε[t]` (shock impact)
"""
struct OccBinRegime{T<:AbstractFloat}
    A::Matrix{T}
    B::Matrix{T}
    C::Matrix{T}
    D::Matrix{T}
end

"""
    OccBinSolution{T}

Piecewise-linear solution from the OccBin algorithm (Guerrieri & Iacoviello 2015).

Fields:
- `linear_path::Matrix{T}` — T_periods × n_endog unconstrained linear path
- `piecewise_path::Matrix{T}` — T_periods × n_endog piecewise-linear path
- `steady_state::Vector{T}` — steady state values
- `regime_history::Matrix{Int}` — T_periods × n_constraints regime indicators (0 = slack, 1+ = binding)
- `converged::Bool` — convergence flag
- `iterations::Int` — number of guess-and-verify iterations
- `spec::DSGESpec{T}` — model specification
- `varnames::Vector{String}` — variable display names
"""
struct OccBinSolution{T<:AbstractFloat}
    linear_path::Matrix{T}
    piecewise_path::Matrix{T}
    steady_state::Vector{T}
    regime_history::Matrix{Int}
    converged::Bool
    iterations::Int
    spec::DSGESpec{T}
    varnames::Vector{String}
end

function Base.show(io::IO, sol::OccBinSolution{T}) where {T}
    n_constraints = size(sol.regime_history, 2)
    binding_periods = sum(sol.regime_history .> 0)
    spec_data = Any[
        "Variables"       sol.spec.n_endog;
        "Periods"         size(sol.piecewise_path, 1);
        "Constraints"     n_constraints;
        "Binding periods" binding_periods;
        "Converged"       sol.converged ? "Yes" : "No";
        "Iterations"      sol.iterations;
    ]
    _pretty_table(io, spec_data;
        title = "OccBin Piecewise-Linear Solution",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

report(sol::OccBinSolution) = show(stdout, sol)

"""
    OccBinIRF{T}

Impulse response comparison between unconstrained linear and OccBin piecewise-linear paths.

Fields:
- `linear::Matrix{T}` — H × n_endog linear IRF
- `piecewise::Matrix{T}` — H × n_endog piecewise-linear IRF
- `regime_history::Matrix{Int}` — H × n_constraints regime indicators
- `varnames::Vector{String}` — variable display names
- `shock_name::String` — name of the shocked variable
"""
struct OccBinIRF{T<:AbstractFloat}
    linear::Matrix{T}
    piecewise::Matrix{T}
    regime_history::Matrix{Int}
    varnames::Vector{String}
    shock_name::String
end

function Base.show(io::IO, oirf::OccBinIRF{T}) where {T}
    binding_periods = sum(oirf.regime_history .> 0)
    max_dev = maximum(abs.(oirf.piecewise .- oirf.linear))
    spec_data = Any[
        "Shock"           oirf.shock_name;
        "Variables"       size(oirf.piecewise, 2);
        "Horizon"         size(oirf.piecewise, 1);
        "Binding periods" binding_periods;
        "Max deviation"   round(max_dev; digits=6);
    ]
    _pretty_table(io, spec_data;
        title = "OccBin IRF Comparison",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

report(oirf::OccBinIRF) = show(stdout, oirf)

