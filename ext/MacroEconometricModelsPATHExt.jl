module MacroEconometricModelsPATHExt

using MacroEconometricModels
using JuMP
using PATHSolver

# Extension loaded when JuMP + PATHSolver are available.
# Adds PATH-based MCP solvers for constrained SS and PF.

# =============================================================================
# Helper: extract variable bounds as vectors
# =============================================================================

function _extract_bounds(spec, constraints)
    FT = eltype(spec.steady_state)
    if isempty(spec.steady_state)
        FT = Float64
    end
    n = spec.n_endog
    lower = fill(FT(-Inf), n)
    upper = fill(FT(Inf), n)
    for c in constraints
        if c isa MacroEconometricModels.VariableBound
            idx = findfirst(==(c.var_name), spec.endog)
            c.lower !== nothing && (lower[idx] = FT(c.lower))
            c.upper !== nothing && (upper[idx] = FT(c.upper))
        end
    end
    return lower, upper
end

# =============================================================================
# Helper: build ForwardDiff-compatible residual wrappers
# =============================================================================

"""
Build a callable for a single residual equation i at steady state.
Takes n scalar args (the SS values). Returns scalar residual.
"""
function _build_ss_residual_i(residual_fn, n_ε, θ)
    function ss_res(args::S...) where {S<:Real}
        y = collect(args)
        ε_z = zeros(S, n_ε)
        try
            return residual_fn(y, y, y, ε_z, θ)
        catch e
            (e isa DomainError || e isa InexactError) && return S(NaN)
            rethrow(e)
        end
    end
    return ss_res
end

"""
Build a callable for one equilibrium equation in the perfect foresight system.
Takes 3n + n_ε scalar args: [y_t; y_lag; y_lead; ε_t].
"""
function _build_pf_equation(fn, n, n_ε, θ)
    function pf_eq(args::S...) where {S<:Real}
        a = collect(args)
        y_t    = a[1:n]
        y_lag  = a[n+1:2n]
        y_lead = a[2n+1:3n]
        ε_t    = a[3n+1:3n+n_ε]
        try
            return fn(y_t, y_lag, y_lead, ε_t, θ)
        catch e
            (e isa DomainError || e isa InexactError) && return S(NaN)
            rethrow(e)
        end
    end
    return pf_eq
end

# =============================================================================
# MCP Steady State via JuMP + PATHSolver
# =============================================================================

function MacroEconometricModels._path_compute_steady_state(
        spec::MacroEconometricModels.DSGESpec{T},
        constraints::Vector;
        initial_guess::Union{Nothing,AbstractVector}=nothing) where {T}

    n = spec.n_endog
    n_ε = spec.n_exog
    θ = spec.param_values
    lower, upper = _extract_bounds(spec, constraints)

    model = JuMP.Model(PATHSolver.Optimizer)
    JuMP.set_silent(model)

    # Variables with bounds
    @variable(model, lower[i] <= x[i=1:n] <= upper[i])

    # Initial guess
    ss_guess = if initial_guess !== nothing
        T.(initial_guess)
    elseif !isempty(spec.steady_state)
        spec.steady_state
    else
        ones(T, n)
    end
    for i in 1:n
        JuMP.set_start_value(x[i], ss_guess[i])
    end

    # Complementarity: residual_i(x, x, x, 0, θ) ⟂ x[i]
    for i in 1:n
        wrapper = _build_ss_residual_i(spec.residual_fns[i], n_ε, θ)
        op = JuMP.add_nonlinear_operator(model, n, wrapper;
                  name=Symbol(:mcp_ss_eq_, i))
        @constraint(model, op(x...) ⟂ x[i])
    end

    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    if status != JuMP.LOCALLY_SOLVED && status != JuMP.OPTIMAL
        @warn "MCP steady state: solver status = $status"
    end

    return T.(JuMP.value.(x))
end

# =============================================================================
# MCP Perfect Foresight via JuMP + PATHSolver
# =============================================================================

function MacroEconometricModels._path_perfect_foresight(
        spec::MacroEconometricModels.DSGESpec{FT},
        T_periods::Int,
        shocks::Matrix{FT},
        constraints::Vector) where {FT}

    n = spec.n_endog
    n_ε = spec.n_exog
    θ = spec.param_values
    y_ss = spec.steady_state
    lower, upper = _extract_bounds(spec, constraints)

    # Warn about PATH free-tier size limit
    total_vars = n * T_periods
    if total_vars > 300
        @warn "PATH free-tier limit: 300 variables, this problem has $total_vars. " *
              "Obtain a PATH license for larger problems." maxlog=1
    end

    model = JuMP.Model(PATHSolver.Optimizer)
    JuMP.set_silent(model)

    # Stacked variables: x[var, period] for t=1..T
    @variable(model, lower[i] <= x[i=1:n, t=1:T_periods] <= upper[i])

    # Initial guess: all periods at SS
    for t in 1:T_periods, i in 1:n
        JuMP.set_start_value(x[i, t], y_ss[i])
    end

    # Build one operator per (equation, period) pair
    n_args = 3 * n + n_ε

    for t in 1:T_periods
        for eq in 1:n
            wrapper = _build_pf_equation(spec.residual_fns[eq], n, n_ε, θ)
            op = JuMP.add_nonlinear_operator(model, n_args, wrapper;
                      name=Symbol(:mcp_pf_eq_, eq, :_t, t))

            # Boundary conditions: t=0 → y_ss, t=T+1 → y_ss
            if t == 1 && t == T_periods
                @constraint(model,
                    op(x[:, t]..., y_ss..., y_ss..., shocks[t, :]...) ⟂ x[eq, t])
            elseif t == 1
                @constraint(model,
                    op(x[:, t]..., y_ss..., x[:, t+1]..., shocks[t, :]...) ⟂ x[eq, t])
            elseif t == T_periods
                @constraint(model,
                    op(x[:, t]..., x[:, t-1]..., y_ss..., shocks[t, :]...) ⟂ x[eq, t])
            else
                @constraint(model,
                    op(x[:, t]..., x[:, t-1]..., x[:, t+1]..., shocks[t, :]...) ⟂ x[eq, t])
            end
        end
    end

    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    converged = (status == JuMP.LOCALLY_SOLVED || status == JuMP.OPTIMAL)
    if !converged
        @warn "MCP perfect foresight: solver status = $status"
    end

    # Extract solution path
    path_full = zeros(FT, T_periods, n)
    for t in 1:T_periods, i in 1:n
        path_full[t, i] = FT(JuMP.value(x[i, t]))
    end
    deviations_full = path_full .- y_ss'

    # Filter to original variables if augmented
    if spec.augmented
        orig_idx = MacroEconometricModels._original_var_indices(spec)
        path = Matrix{FT}(path_full[:, orig_idx])
        deviations = Matrix{FT}(deviations_full[:, orig_idx])
    else
        path = Matrix{FT}(path_full)
        deviations = Matrix{FT}(deviations_full)
    end

    iter = try
        Int(JuMP.solve_time(model) > 0)
    catch
        0
    end

    MacroEconometricModels.PerfectForesightPath{FT}(path, deviations, converged, iter, spec)
end

end # module
