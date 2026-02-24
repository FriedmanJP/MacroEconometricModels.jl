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
- `endog::Vector{Symbol}` — endogenous variable names (possibly augmented)
- `exog::Vector{Symbol}` — exogenous shock names
- `params::Vector{Symbol}` — parameter names
- `param_values::Dict{Symbol,T}` — calibrated parameter values
- `equations::Vector{Expr}` — raw Julia equation expressions (possibly augmented)
- `residual_fns::Vector{Function}` — callable `f(y_t, y_lag, y_lead, ε, θ) → scalar`
- `n_endog::Int` — number of endogenous variables (including auxiliaries)
- `n_exog::Int` — number of exogenous shocks
- `n_params::Int` — number of parameters
- `n_expect::Int` — number of expectation errors (forward-looking variables)
- `forward_indices::Vector{Int}` — indices of equations with `[t+1]` terms
- `steady_state::Vector{T}` — steady state values
- `varnames::Vector{String}` — display names
- `ss_fn::Union{Nothing, Function}` — optional analytical steady-state function `θ → y_ss`
- `original_endog::Vector{Symbol}` — pre-augmentation endogenous variable names
- `original_equations::Vector{Expr}` — pre-augmentation equation expressions
- `n_original_endog::Int` — number of original endogenous variables
- `n_original_eq::Int` — number of original equations
- `augmented::Bool` — whether model was augmented with auxiliary variables
- `max_lag::Int` — maximum lag order in the model (1 for standard, >1 if augmented)
- `max_lead::Int` — maximum lead order in the model (1 for standard, >1 if augmented)
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
    original_endog::Vector{Symbol}
    original_equations::Vector{Expr}
    n_original_endog::Int
    n_original_eq::Int
    augmented::Bool
    max_lag::Int
    max_lead::Int

    function DSGESpec{T}(endog, exog, params, param_values, equations, residual_fns,
                         n_expect, forward_indices, steady_state,
                         ss_fn::Union{Nothing, Function}=nothing;
                         original_endog::Vector{Symbol}=endog,
                         original_equations::Vector{Expr}=equations,
                         augmented::Bool=false,
                         max_lag::Int=1,
                         max_lead::Int=1) where {T<:AbstractFloat}
        n_endog = length(endog)
        n_exog = length(exog)
        n_params = length(params)
        n_original_endog = length(original_endog)
        n_original_eq = length(original_equations)
        @assert length(equations) == n_endog "Need as many equations as endogenous variables"
        @assert length(residual_fns) == n_endog
        @assert length(forward_indices) == n_expect
        varnames = [string(s) for s in endog]
        new{T}(endog, exog, params, param_values, equations, residual_fns,
               n_endog, n_exog, n_params, n_expect, forward_indices, steady_state,
               varnames, ss_fn, original_endog, original_equations,
               n_original_endog, n_original_eq, augmented, max_lag, max_lead)
    end
end

"""
    _original_var_indices(spec::DSGESpec) → Vector{Int}

Return indices of original endogenous variables in the (possibly augmented) state vector.
"""
function _original_var_indices(spec::DSGESpec)
    if !spec.augmented
        return collect(1:spec.n_endog)
    end
    return [findfirst(==(v), spec.endog) for v in spec.original_endog]
end

# show(io, DSGESpec) is defined in dsge/display.jl

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
# PerturbationSolution — higher-order perturbation with pruning
# =============================================================================

"""
    PerturbationSolution{T}

Higher-order perturbation solution with Kim et al. (2008) pruning.

For order k, the decision rule is:
- Order 1: `z_t = z̄ + g_x·x̂_t`
- Order 2: `+ (1/2)·g_xx·(x̂_t ⊗ x̂_t) + (1/2)·g_σσ·σ²`
- Order 3: `+ (1/6)·g_xxx·(x̂_t ⊗ x̂_t ⊗ x̂_t) + (3/6)·g_σσx·σ²·x̂_t`

Fields:
- `order` — perturbation order (1, 2, or 3)
- `gx, hx` — first-order coefficients (controls: ny×nv, states: nx×nv)
- `gxx, hxx, gσσ, hσσ` — second-order (nothing if order < 2)
- `gxxx, hxxx, gσσx, hσσx, gσσσ, hσσσ` — third-order (nothing if order < 3)
- `eta` — shock loading matrix (nv × nu)
- `steady_state` — full steady state vector
- `state_indices, control_indices` — variable partition
- `eu` — [existence, uniqueness] from first-order
- `method` — `:perturbation`
- `spec` — model specification
- `linear` — linearized form
"""
struct PerturbationSolution{T<:AbstractFloat}
    order::Int

    # First-order (always present) — in terms of v = [x; ε]
    gx::Matrix{T}                         # ny × nv
    hx::Matrix{T}                         # nx × nv

    # Second-order (order ≥ 2)
    gxx::Union{Nothing, Matrix{T}}        # ny × nv² (flattened tensor)
    hxx::Union{Nothing, Matrix{T}}        # nx × nv² (flattened tensor)
    gσσ::Union{Nothing, Vector{T}}        # ny
    hσσ::Union{Nothing, Vector{T}}        # nx

    # Third-order (order == 3)
    gxxx::Union{Nothing, Matrix{T}}       # ny × nv³ (flattened tensor)
    hxxx::Union{Nothing, Matrix{T}}       # nx × nv³ (flattened tensor)
    gσσx::Union{Nothing, Matrix{T}}       # ny × nv
    hσσx::Union{Nothing, Matrix{T}}       # nx × nv
    gσσσ::Union{Nothing, Vector{T}}       # ny
    hσσσ::Union{Nothing, Vector{T}}       # nx

    # Shock loading & metadata
    eta::Matrix{T}                        # nv × nu — [0; I] block
    steady_state::Vector{T}
    state_indices::Vector{Int}
    control_indices::Vector{Int}

    eu::Vector{Int}
    method::Symbol
    spec::DSGESpec{T}
    linear::LinearDSGE{T}
end

# Accessors
nvars(sol::PerturbationSolution) = sol.spec.n_endog
nshocks(sol::PerturbationSolution) = sol.spec.n_exog
nstates(sol::PerturbationSolution) = length(sol.state_indices)
ncontrols(sol::PerturbationSolution) = length(sol.control_indices)
is_determined(sol::PerturbationSolution) = sol.eu[1] == 1 && sol.eu[2] == 1
function is_stable(sol::PerturbationSolution{T}) where {T}
    nx = nstates(sol)
    nx == 0 && return true
    hx_state = sol.hx[:, 1:nx]  # state-to-state block
    maximum(abs.(eigvals(hx_state))) < one(T)
end

function Base.show(io::IO, sol::PerturbationSolution{T}) where {T}
    nx = nstates(sol)
    ny = ncontrols(sol)
    exist_str = sol.eu[1] == 1 ? "Yes" : "No"
    unique_str = sol.eu[2] == 1 ? "Yes" : "No"
    stable_str = is_stable(sol) ? "Yes" : "No"

    spec_data = Any[
        "Variables"     nvars(sol);
        "States"        nx;
        "Controls"      ny;
        "Shocks"        nshocks(sol);
        "Order"         sol.order;
        "Existence"     exist_str;
        "Uniqueness"    unique_str;
        "Stable"        stable_str;
    ]
    _pretty_table(io, spec_data;
        title = "DSGE Perturbation Solution (order $(sol.order))",
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

