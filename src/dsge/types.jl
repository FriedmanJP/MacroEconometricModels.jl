# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for DSGE models — specification, linearized form, solution, and estimation.
"""

using LinearAlgebra

# =============================================================================
# ModelSpec (IR + compiled residuals) lives in dsge/ir.jl.
# DSGESpec / HADSGESpec were removed in the v0.9.0 ModelSpec overhaul (#634).
# =============================================================================

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
- `spec::ModelSpec{T,NoAgents}` — back-reference to specification
"""
struct LinearDSGE{T<:AbstractFloat}
    Gamma0::Matrix{T}
    Gamma1::Matrix{T}
    C::Vector{T}
    Psi::Matrix{T}
    Pi::Matrix{T}
    spec::ModelSpec{T,NoAgents}

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
- `eu::Vector{Int}` — [existence, uniqueness]: 1=yes, 0=no
- `method::Symbol` — `:gensys`, `:blanchard_kahn`, or `:klein`
- `eigenvalues::Vector{ComplexF64}` — eigenvalues of G1 (the state-transition matrix)
- `spec::ModelSpec{T,NoAgents}` — model specification
- `linear::LinearDSGE{T}` — linearized form
"""
struct DSGESolution{T<:AbstractFloat}
    G1::Matrix{T}
    impact::Matrix{T}
    C_sol::Vector{T}
    eu::Vector{Int}
    method::Symbol
    eigenvalues::Vector{ComplexF64}
    spec::ModelSpec{T,NoAgents}
    linear::LinearDSGE{T}
end

# Accessors
nvars(sol::DSGESolution) = sol.spec.n_endog

"""
    nshocks(sol) -> Int

Number of structural shocks (exogenous driving processes) in a DSGE solution.
"""
nshocks(sol::DSGESolution) = sol.spec.n_exog

"""
    is_determined(sol) -> Bool

Return `true` if the linear RE solution is unique (Blanchard–Kahn / Sims `eu == [1,1]`).
"""
is_determined(sol::DSGESolution) = sol.eu[1] == 1 && sol.eu[2] == 1

# Reuse the cached eigenvalues (all ctors store eigvals(G1)) instead of recomputing (S-17 / #224).
# The comment must stay ABOVE the docstring: a comment between a docstring and the
# definition makes Julia discard the docstring silently (#590).
"""
    is_stable(sol) -> Bool

Return `true` if all eigenvalues of the state transition `G1` lie inside the unit circle.
"""
is_stable(sol::DSGESolution) = maximum(abs.(sol.eigenvalues); init=0.0) < 1.0

"""Render a variable→steady-state two-column table (S4/T168). Skips silently if empty."""
function _ss_table(io::IO, varnames::Vector{String}, ss::AbstractVector; title::String="Steady State")
    n = min(length(varnames), length(ss))
    n == 0 && return
    data = Matrix{Any}(undef, n, 2)
    for i in 1:n
        data[i, 1] = varnames[i]
        data[i, 2] = _fmt(ss[i]; digits=6)
    end
    _pretty_table(io, data; title = title, column_labels = ["", ""], alignment = [:l, :r])
end

"""Steady state for a DSGESolution: `spec.steady_state`, or the linear observation offset
`(I - G1)\\C_sol` for pre-linearized (linear=true) models whose spec SS auto-zeros."""
function _dsge_solution_ss(sol::DSGESolution)
    ss = sol.spec.steady_state
    (!isempty(ss) && !all(iszero, ss)) && return ss
    if !isempty(sol.C_sol) && size(sol.G1, 1) == length(sol.C_sol)
        try
            return (I - sol.G1) \ sol.C_sol
        catch
        end
    end
    return ss
end

function Base.show(io::IO, sol::DSGESolution{T}) where {T}
    n = nvars(sol)
    n_stable = count(x -> abs(x) < 1.0, sol.eigenvalues)
    n_unstable = length(sol.eigenvalues) - n_stable
    exist_str = sol.eu[1] == 1 ? "Yes" : "No"
    unique_str = sol.eu[2] == 1 ? "Yes" : "No"
    max_eig = maximum(abs.(sol.eigenvalues); init=0.0)

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
    _ss_table(io, sol.spec.varnames, _dsge_solution_ss(sol))
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
    spec::ModelSpec{T,NoAgents}
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
    _ss_table(io, sol.spec.varnames, sol.steady_state)
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
- `spec::ModelSpec{T,NoAgents}` — model specification
"""
struct PerfectForesightPath{T<:AbstractFloat}
    path::Matrix{T}
    deviations::Matrix{T}
    converged::Bool
    iterations::Int
    spec::ModelSpec{T,NoAgents}
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
    # Path summary: steady state, terminal value, and max deviation per variable. (S4/T168)
    ss = pf.spec.steady_state
    vn = pf.spec.varnames
    np = min(length(vn), size(pf.path, 2))
    if np > 0
        pdata = Matrix{Any}(undef, np, 4)
        for i in 1:np
            s = (!isempty(ss) && i <= length(ss)) ? ss[i] : zero(T)
            pdata[i, 1] = vn[i]
            pdata[i, 2] = _fmt(s; digits=6)
            pdata[i, 3] = _fmt(pf.path[end, i]; digits=6)
            pdata[i, 4] = _fmt(maximum(abs.(pf.path[:, i] .- s)); digits=6)
        end
        _pretty_table(io, pdata; title = "Path Summary",
            column_labels = ["", "Steady state", "Terminal", "Max |dev|"],
            alignment = [:l, :r, :r, :r])
    end
end

# =============================================================================
# ProjectionSolution — Chebyshev collocation global policy approximation
# =============================================================================

"""
    ProjectionSolution{T}

Global policy function approximation via Chebyshev collocation.

The policy function is `y = Σ_k coefficients[k] * T_k(x_scaled)` where T_k are
Chebyshev polynomials evaluated at states mapped to [-1,1].

Fields:
- `coefficients` — `n_vars × n_basis` Chebyshev coefficients
- `state_bounds` — `nx × 2` domain bounds `[lower upper]` per state
- `grid_type` — `:tensor` or `:smolyak`
- `degree` — polynomial degree (tensor) or Smolyak level μ
- `collocation_nodes` — `n_nodes × nx` grid points
- `residual_norm` — final `||R||`
- `n_basis` — number of basis functions
- `multi_indices` — `n_basis × nx` multi-index matrix
- `quadrature` — `:gauss_hermite` or `:monomial`
- `spec` — model specification
- `linear` — linearized form
- `impact` — first-order shock-impact matrix (companion-QZ), cached so `irf`/`simulate` need
  not re-solve the first-order system on every call
- `steady_state` — cached steady state vector
- `state_indices, control_indices` — variable partition
- `converged` — Newton convergence flag
- `iterations` — Newton iterations used
- `method` — `:projection`
- `euler_error` — max Euler error achieved on the adaptive test set; `NaN` when the solver did
  not measure it (i.e. `adaptive=false`). Call `max_euler_error(sol)` to measure it on demand.
- `smolyak_levels` — `n_blocks × nx` admissible Smolyak level multi-index set; `0×0` for tensor
  grids. `vec(maximum(smolyak_levels; dims=1))` gives the resolution reached in each state,
  which is where an anisotropic or adaptively refined grid differs from an isotropic one.
- `refinements` — adaptive refinement rounds performed (`0` when `adaptive=false`)
- `value_fn` — `n_nodes × 1` value function on the collocation nodes; empty unless
  `method === :vfi`
- `value_coefficients` — Chebyshev coefficients of `V`; empty unless `method === :vfi`
"""
struct ProjectionSolution{T<:AbstractFloat}
    coefficients::Matrix{T}         # n_vars × n_basis
    state_bounds::Matrix{T}         # nx × 2 ([lower upper] per state)
    grid_type::Symbol               # :tensor or :smolyak
    degree::Int                     # polynomial degree (tensor) or Smolyak level μ
    collocation_nodes::Matrix{T}    # n_nodes × nx
    residual_norm::T                # final ||R||
    n_basis::Int
    multi_indices::Matrix{Int}      # n_basis × nx
    quadrature::Symbol              # :gauss_hermite or :monomial
    spec::ModelSpec{T,NoAgents}
    linear::LinearDSGE{T}
    impact::Matrix{T}               # first-order shock-impact matrix (companion-QZ), cached
    steady_state::Vector{T}
    state_indices::Vector{Int}
    control_indices::Vector{Int}
    converged::Bool
    iterations::Int
    method::Symbol                  # :projection
    euler_error::T                  # achieved max Euler error (NaN when not measured)
    smolyak_levels::Matrix{Int}     # n_blocks × nx level set (0×0 for tensor grids)
    refinements::Int                # adaptive refinement rounds performed
    value_fn::Matrix{T}             # n_nodes × 1 Bellman value (empty if not VFI)
    value_coefficients::Vector{T}   # Chebyshev coefficients of V (empty if not VFI)

    # The accuracy / value-function fields are trailing keywords with defaults so every
    # existing 18-positional construction site (collocation/PFI) keeps working unchanged.
    function ProjectionSolution{T}(coefficients, state_bounds, grid_type, degree,
                                   collocation_nodes, residual_norm, n_basis, multi_indices,
                                   quadrature, spec, linear, impact, steady_state,
                                   state_indices, control_indices, converged, iterations,
                                   method;
                                   euler_error::Real=T(NaN),
                                   smolyak_levels::Matrix{Int}=zeros(Int, 0, 0),
                                   refinements::Int=0,
                                   value_fn::AbstractMatrix{<:Real}=zeros(T, 0, 0),
                                   value_coefficients::AbstractVector{<:Real}=T[]) where {T<:AbstractFloat}
        new{T}(coefficients, state_bounds, grid_type, degree, collocation_nodes,
               residual_norm, n_basis, multi_indices, quadrature, spec, linear, impact,
               steady_state, state_indices, control_indices, converged, iterations, method,
               T(euler_error), smolyak_levels, refinements,
               Matrix{T}(value_fn), Vector{T}(value_coefficients))
    end
end

nvars(sol::ProjectionSolution) = sol.spec.n_endog
nshocks(sol::ProjectionSolution) = sol.spec.n_exog
nstates(sol::ProjectionSolution) = length(sol.state_indices)
ncontrols(sol::ProjectionSolution) = length(sol.control_indices)
is_determined(sol::ProjectionSolution) = sol.converged
is_stable(sol::ProjectionSolution) = sol.converged

function Base.show(io::IO, sol::ProjectionSolution{T}) where {T}
    nx = nstates(sol)
    ny = ncontrols(sol)
    conv_str = sol.converged ? "Yes" : "No"

    spec_data = Any[
        "Variables"       nvars(sol);
        "States"          nx;
        "Controls"        ny;
        "Shocks"          nshocks(sol);
        "Grid type"       sol.grid_type;
        "Degree"          sol.degree;
        "Basis functions" sol.n_basis;
        "Grid points"     size(sol.collocation_nodes, 1);
        "Quadrature"      sol.quadrature;
        "Residual norm"   string(round(sol.residual_norm; sigdigits=3));
        "Converged"       conv_str;
        "Iterations"      sol.iterations;
    ]
    if !isempty(sol.smolyak_levels) && size(sol.smolyak_levels, 1) > 0
        lv = vec(maximum(sol.smolyak_levels; dims=1))
        spec_data = vcat(spec_data, Any["Levels per state" string(lv)])
    end
    if isfinite(sol.euler_error)
        spec_data = vcat(spec_data, Any["Max Euler error" string(round(sol.euler_error; sigdigits=3))])
    end
    if sol.refinements > 0
        spec_data = vcat(spec_data, Any["Refinements" sol.refinements])
    end
    if !isempty(sol.value_fn)
        spec_data = vcat(spec_data, Any["Value function" "stored"])
    end
    title = sol.method === :vfi ? "DSGE VFI Solution (Bellman / Chebyshev)" :
            "DSGE Projection Solution (Chebyshev Collocation)"
    _pretty_table(io, spec_data;
        title = title,
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _ss_table(io, sol.spec.varnames, sol.steady_state)
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
- `solution::Union{DSGESolution{T}, PerturbationSolution{T}}` — solution at estimated parameters
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
    solution::Union{DSGESolution{T}, PerturbationSolution{T}, ProjectionSolution{T}}
    converged::Bool
    spec::ModelSpec{T,NoAgents}

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
- `spec::ModelSpec{T,NoAgents}` — model specification
- `varnames::Vector{String}` — variable display names
- `constraints::Vector{OccBinConstraint{T}}` — constraint(s) used in the solve
"""
struct OccBinSolution{T<:AbstractFloat}
    linear_path::Matrix{T}
    piecewise_path::Matrix{T}
    steady_state::Vector{T}
    regime_history::Matrix{Int}
    converged::Bool
    iterations::Int
    spec::ModelSpec{T,NoAgents}
    varnames::Vector{String}
    constraints::Vector{OccBinConstraint{T}}
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
    _ss_table(io, sol.varnames, sol.steady_state)
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

# =============================================================================
# Exception taxonomy for Bayesian DSGE solve/likelihood failures
# =============================================================================

"""
    DSGESolveError <: Exception

Internal DSGE solve/estimation failure that legitimately implies zero posterior
mass (indeterminacy, no-solution, steady-state non-convergence) — as opposed to a
code bug. Likelihood closures catch this (and the concrete numeric-failure types in
`_benign_solve_error`) and return `-Inf`, counting the draw as failed;
everything else is rethrown.
"""
struct DSGESolveError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DSGESolveError) = print(io, "DSGESolveError: ", e.msg)

"""
    _benign_solve_error(e) -> Bool

True when `e` is a legitimate per-θ numeric failure that should be caught and
counted as a failed draw (return `-Inf`), rather than a bug that must propagate.
"""
_benign_solve_error(e) = e isa DSGESolveError ||
                         e isa DomainError ||
                         e isa LinearAlgebra.SingularException ||
                         e isa LinearAlgebra.PosDefException ||
                         e isa LinearAlgebra.LAPACKException

