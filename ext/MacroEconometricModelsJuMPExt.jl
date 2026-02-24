module MacroEconometricModelsJuMPExt

using MacroEconometricModels
using JuMP
using Ipopt

function __init__()
    MacroEconometricModels._JUMP_LOADED[] = true
end

# =============================================================================
# Helper: build a ForwardDiff-compatible steady-state objective
# =============================================================================

"""
Build a callable for the steady-state objective that works with ForwardDiff Dual numbers.
JuMP passes Dual numbers for AD — we must avoid collect(T, ...) and use the input type.
"""
function _build_ss_objective(residual_fns, n_ε, θ)
    function ss_obj(args::S...) where {S<:Real}
        y = collect(args)
        ε_z = zeros(S, n_ε)
        total = zero(S)
        for fn in residual_fns
            try
                r = fn(y, y, y, ε_z, θ)
                total += r^2
            catch e
                (e isa DomainError || e isa InexactError) && return S(1e20)
                rethrow(e)
            end
        end
        return total
    end
    return ss_obj
end

"""
Build a callable for a nonlinear constraint at steady state.
"""
function _build_ss_nlcon(cfn, n_ε, θ)
    function ss_nlcon(args::S...) where {S<:Real}
        y = collect(args)
        ε_z = zeros(S, n_ε)
        cfn(y, y, y, ε_z, θ)
    end
    return ss_nlcon
end

# =============================================================================
# Helper: build ForwardDiff-compatible equation and constraint wrappers for PF
# =============================================================================

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
            (e isa DomainError || e isa InexactError) && return S(1e20)
            rethrow(e)
        end
    end
    return pf_eq
end

"""
Build a callable for a nonlinear inequality constraint in the perfect foresight system.
"""
function _build_pf_nlcon(cfn, n, n_ε, θ)
    function pf_nlcon(args::S...) where {S<:Real}
        a = collect(args)
        y_t    = a[1:n]
        y_lag  = a[n+1:2n]
        y_lead = a[2n+1:3n]
        ε_t    = a[3n+1:3n+n_ε]
        cfn(y_t, y_lag, y_lead, ε_t, θ)
    end
    return pf_nlcon
end

# =============================================================================
# Constrained Steady State via JuMP + Ipopt
# =============================================================================

function MacroEconometricModels._jump_compute_steady_state(
        spec::MacroEconometricModels.DSGESpec{T},
        constraints::Vector;
        initial_guess::Union{Nothing,AbstractVector}=nothing) where {T}

    n = spec.n_endog
    n_ε = spec.n_exog
    θ = spec.param_values

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)

    # Variables
    @variable(model, x[1:n])

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

    # Apply variable bounds
    for c in constraints
        if c isa MacroEconometricModels.VariableBound
            idx = findfirst(==(c.var_name), spec.endog)
            c.lower !== nothing && JuMP.set_lower_bound(x[idx], T(c.lower))
            c.upper !== nothing && JuMP.set_upper_bound(x[idx], T(c.upper))
        end
    end

    # Objective: minimize sum of squared residuals at steady state
    ss_obj = _build_ss_objective(spec.residual_fns, n_ε, θ)
    op_obj = JuMP.add_nonlinear_operator(model, n, ss_obj; name=:ss_objective)
    @objective(model, Min, op_obj(x...))

    # Nonlinear inequality constraints
    nl_idx = 0
    for c in constraints
        if c isa MacroEconometricModels.NonlinearConstraint
            nl_idx += 1
            wrapper = _build_ss_nlcon(c.fn, n_ε, θ)
            op_c = JuMP.add_nonlinear_operator(model, n, wrapper;
                        name=Symbol(:nlcon_, nl_idx))
            @constraint(model, op_c(x...) <= 0)
        end
    end

    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    if status != JuMP.LOCALLY_SOLVED && status != JuMP.OPTIMAL
        @warn "Constrained steady state: solver status = $status"
    end

    return T.(JuMP.value.(x))
end

# =============================================================================
# Constrained Perfect Foresight via JuMP + Ipopt
# =============================================================================

function MacroEconometricModels._jump_perfect_foresight(
        spec::MacroEconometricModels.DSGESpec{FT},
        T_periods::Int,
        shocks::Matrix{FT},
        constraints::Vector) where {FT}

    n = spec.n_endog
    n_ε = spec.n_exog
    θ = spec.param_values
    y_ss = spec.steady_state

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_optimizer_attribute(model, "max_iter", 3000)

    # Variables: x[var, period] for t = 0, 1, ..., T+1
    # t=0 and t=T+1 are boundary conditions (fixed to SS)
    @variable(model, x[1:n, 0:T_periods+1])

    # Fix boundary conditions
    for i in 1:n
        JuMP.fix(x[i, 0], y_ss[i]; force=true)
        JuMP.fix(x[i, T_periods + 1], y_ss[i]; force=true)
    end

    # Initial guess: all periods at SS
    for t in 1:T_periods, i in 1:n
        JuMP.set_start_value(x[i, t], y_ss[i])
    end

    # Apply variable bounds at each interior period
    for c in constraints
        if c isa MacroEconometricModels.VariableBound
            idx = findfirst(==(c.var_name), spec.endog)
            for t in 1:T_periods
                c.lower !== nothing && JuMP.set_lower_bound(x[idx, t], FT(c.lower))
                c.upper !== nothing && JuMP.set_upper_bound(x[idx, t], FT(c.upper))
            end
        end
    end

    # Register one operator per equation
    # Each takes 3n + n_ε scalar args: [y_t(1:n), y_lag(n+1:2n), y_lead(2n+1:3n), ε(3n+1:end)]
    n_args = 3 * n + n_ε
    eq_ops = Vector{Any}(undef, n)

    for eq in 1:n
        wrapper = _build_pf_equation(spec.residual_fns[eq], n, n_ε, θ)
        eq_ops[eq] = JuMP.add_nonlinear_operator(model, n_args, wrapper;
                          name=Symbol(:pf_eq_, eq))
    end

    # Equilibrium constraints for each period
    for t in 1:T_periods
        for eq in 1:n
            @constraint(model,
                eq_ops[eq](x[:, t]..., x[:, t-1]..., x[:, t+1]..., shocks[t, :]...) == 0)
        end
    end

    # Nonlinear inequality constraints at each period
    nl_idx = 0
    for c in constraints
        if c isa MacroEconometricModels.NonlinearConstraint
            nl_idx += 1
            wrapper = _build_pf_nlcon(c.fn, n, n_ε, θ)
            op_c = JuMP.add_nonlinear_operator(model, n_args, wrapper;
                        name=Symbol(:pf_nlcon_, nl_idx))
            for t in 1:T_periods
                @constraint(model,
                    op_c(x[:, t]..., x[:, t-1]..., x[:, t+1]..., shocks[t, :]...) <= 0)
            end
        end
    end

    # Feasibility problem
    @objective(model, Min, 0)

    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    converged = (status == JuMP.LOCALLY_SOLVED || status == JuMP.OPTIMAL)
    if !converged
        @warn "Constrained perfect foresight: solver status = $status"
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

    iter = 0
    try
        iter = JuMP.barrier_iterations_count(model)
    catch
    end

    MacroEconometricModels.PerfectForesightPath{FT}(path, deviations, converged, iter, spec)
end

end # module
