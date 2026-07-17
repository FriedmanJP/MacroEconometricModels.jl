# Optim.jl + NLopt.jl Fallback for Constrained DSGE — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Optim.jl and NLopt.jl as always-available backends for constrained DSGE solving, removing the JuMP+Ipopt requirement for box-constrained and nonlinear-inequality-constrained problems.

**Architecture:** Four new solver functions (`_optim_steady_state`, `_nlopt_steady_state`, `_projected_newton_pf`, `_nlopt_perfect_foresight`) added inline to existing files. Adapter wrappers bridge the splatted-scalar closure API (designed for JuMP) to vector interfaces. Dispatch logic updated so NonlinearConstraint defaults to NLopt when JuMP is absent, and PF escalation uses projected Newton instead of JuMP.

**Tech Stack:** Julia, Optim.jl (existing dep), NLopt.jl (new dep), ForwardDiff.jl (new dep), NonlinearSolve.jl (existing), SparseArrays (existing)

**Spec:** `docs/superpowers/specs/2026-03-18-optim-nlopt-fallback-design.md`

---

## File Map

| File | Responsibility | Action |
|------|---------------|--------|
| `Project.toml` | Package dependencies and compat | Modify: add NLopt + ForwardDiff |
| `src/MacroEconometricModels.jl` | Top-level imports | Modify: add `import NLopt, ForwardDiff` |
| `src/dsge/constraints.jl` | Constraint types, dispatch, helpers, adapter wrappers | Modify: update `_select_solver`, error msgs, validation, add adapters |
| `src/dsge/steady_state.jl` | Steady-state computation | Modify: add `_optim_steady_state`, `_nlopt_steady_state`, wire dispatch + escalation |
| `src/dsge/perfect_foresight.jl` | Perfect foresight solver | Modify: add `_projected_newton_pf`, `_nlopt_perfect_foresight`, update escalation |
| `test/dsge/test_dsge.jl` | DSGE test suite | Modify: add ~9 new tests |

---

### Task 1: Add Dependencies to Project.toml

**Files:**
- Modify: `Project.toml` (lines 6–18 for `[deps]`, lines 29–43 for `[compat]`)

- [ ] **Step 1: Add NLopt and ForwardDiff to `[deps]`**

In `Project.toml`, add these two lines to the `[deps]` section (alphabetical order):

```toml
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
NLopt = "76087f3c-5699-56af-9a33-bf431cd00edd"
```

- [ ] **Step 2: Add compat entries**

In `Project.toml` `[compat]` section, add:

```toml
ForwardDiff = "0.10"
NLopt = "1"
```

- [ ] **Step 3: Add imports to main module**

In `src/MacroEconometricModels.jl`, after `import Optim` (line 71), add:

```julia
import NLopt
import ForwardDiff
```

- [ ] **Step 4: Resolve and verify**

Run:
```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); using MacroEconometricModels; println("OK")'
```
Expected: `OK` — package loads with new deps.

- [ ] **Step 5: Commit**

```bash
git add Project.toml src/MacroEconometricModels.jl
git commit -m "deps: add NLopt and ForwardDiff for constrained DSGE fallback (#91)"
```

---

### Task 2: Update Dispatch Logic and Add Adapter Wrappers in constraints.jl

**Files:**
- Modify: `src/dsge/constraints.jl`

- [ ] **Step 1: Update `_select_solver` to default to `:nlopt` when JuMP absent**

Replace the `_select_solver` function (lines 130–135) with:

```julia
function _select_solver(constraints::Vector, solver_override::Union{Nothing,Symbol})
    solver_override !== nothing && return solver_override
    has_nlcon = any(c -> c isa NonlinearConstraint, constraints)
    if has_nlcon
        # Prefer Ipopt when JuMP extension is loaded (more robust for large NLP)
        if hasmethod(_jump_compute_steady_state, Tuple{DSGESpec, Vector})
            return :ipopt
        end
        return :nlopt
    end
    return :nonlinearsolve
end
```

- [ ] **Step 2: Update `_JUMP_INSTALL_MSG` to mention NLopt (line 101)**

Replace the `_JUMP_INSTALL_MSG` constant:

```julia
const _JUMP_INSTALL_MSG = "JuMP + Ipopt not loaded. NLopt handles most constrained problems by default.\n" *
    "For explicit JuMP/Ipopt use, install with:\n" *
    "  using Pkg; Pkg.add(\"JuMP\"); Pkg.add(\"Ipopt\")\n" *
    "Then load: import JuMP, Ipopt"
```

- [ ] **Step 3: Add adapter wrappers after `_extract_bounds` (after line 277)**

Append to end of `constraints.jl`:

```julia
# =============================================================================
# Adapter wrappers: splatted-scalar closures → vector interfaces
# =============================================================================

"""Wrap a splatted-scalar closure `f(args::Real...)` as `f(x::AbstractVector)` for Optim.jl."""
_vec_wrap(f) = x -> f(x...)

"""
Wrap a splatted-scalar closure for NLopt's `(x::Vector, grad::Vector)` callback.
Computes gradient in-place via ForwardDiff when `length(grad) > 0`.
"""
function _nlopt_wrap(f)
    function nlopt_cb(x::Vector, grad::Vector)
        if length(grad) > 0
            ForwardDiff.gradient!(grad, z -> f(z...), x)
        end
        return f(x...)
    end
    return nlopt_cb
end

"""
Wrap a PF equation/constraint for NLopt. Each constraint depends on variables at
periods (t-1, t, t+1), so only a `3n + n_ε` slice of the full `T*n` stacked vector
matters. This adapter extracts the relevant slice, computes the local gradient via
ForwardDiff, and scatters it into the full-length gradient vector.

Arguments:
- `f` — splatted-scalar closure from `_build_pf_equation` or `_build_pf_nlcon`
- `t` — period index (1-based)
- `n` — number of endogenous variables
- `n_ε` — number of exogenous shocks
- `T_periods` — total periods
- `y_ss` — steady state vector (for boundary conditions)
- `shocks` — T_periods × n_ε shock matrix
"""
function _pf_nlopt_wrap(f, t::Int, n::Int, n_ε::Int, T_periods::Int,
                         y_ss::Vector, shocks::Matrix)
    function pf_nlopt_cb(x::Vector, grad::Vector)
        # Extract y_t from stacked vector
        y_t = x[(t-1)*n+1 : t*n]

        # y_{t-1}: boundary at t=1
        y_lag = t == 1 ? y_ss : x[(t-2)*n+1 : (t-1)*n]

        # y_{t+1}: boundary at t=T
        y_lead = t == T_periods ? y_ss : x[t*n+1 : (t+1)*n]

        ε_t = shocks[t, :]

        # Build local args for the splatted closure: [y_t; y_lag; y_lead; ε_t]
        local_args = vcat(y_t, y_lag, y_lead, ε_t)
        val = f(local_args...)

        if length(grad) > 0
            fill!(grad, 0.0)
            # Compute local gradient w.r.t. local_args
            local_grad = ForwardDiff.gradient(z -> f(z...), local_args)

            # Scatter: first n entries → ∂f/∂y_t → grad[(t-1)*n+1 : t*n]
            grad[(t-1)*n+1 : t*n] .= local_grad[1:n]

            # Next n entries → ∂f/∂y_{t-1} → grad[(t-2)*n+1 : (t-1)*n] (if t > 1)
            if t > 1
                grad[(t-2)*n+1 : (t-1)*n] .= local_grad[n+1:2n]
            end

            # Next n entries → ∂f/∂y_{t+1} → grad[t*n+1 : (t+1)*n] (if t < T)
            if t < T_periods
                grad[t*n+1 : (t+1)*n] .= local_grad[2n+1:3n]
            end
            # ε_t entries are not decision variables — no scatter needed
        end

        return val
    end
    return pf_nlopt_cb
end
```

- [ ] **Step 4: Verify load**

```bash
julia --project=. -e 'using MacroEconometricModels; println("OK")'
```

- [ ] **Step 5: Commit**

```bash
git add src/dsge/constraints.jl
git commit -m "feat(dsge): update _select_solver and add Optim/NLopt adapter wrappers (#91)"
```

---

### Task 3: Add `_optim_steady_state` and `_nlopt_steady_state`

**Files:**
- Modify: `src/dsge/steady_state.jl`

- [ ] **Step 1: Add `_optim_steady_state` function**

Insert before the `_update_steady_state` function (before line 173):

```julia
"""
    _optim_steady_state(spec, lower, upper; initial_guess=nothing, algorithm=nothing)

Box-constrained steady state via Optim.jl Fminbox(LBFGS()).
Minimizes sum-of-squared-residuals subject to variable bounds.
"""
function _optim_steady_state(spec::DSGESpec{T}, lower::Vector{T}, upper::Vector{T};
        initial_guess::Union{Nothing,AbstractVector}=nothing,
        algorithm=nothing) where {T<:AbstractFloat}

    n = spec.n_endog
    if initial_guess !== nothing
        y0 = T.(initial_guess)
    elseif !isempty(spec.steady_state)
        y0 = T.(spec.steady_state)
    else
        y0 = ones(T, n)
    end
    @assert length(y0) == n "initial_guess must have length $n"

    # Clamp initial guess to bounds
    for i in 1:n
        y0[i] = clamp(y0[i], isfinite(lower[i]) ? lower[i] : T(-1e10),
                               isfinite(upper[i]) ? upper[i] : T(1e10))
    end

    # Build objective: sum of squared residuals (vector interface via _vec_wrap)
    ss_obj_splat = _build_ss_objective(spec.residual_fns, spec.n_exog, spec.param_values)
    obj = _vec_wrap(ss_obj_splat)

    alg = algorithm !== nothing ? algorithm : Optim.Fminbox(Optim.LBFGS())
    result = Optim.optimize(obj, lower, upper, y0, alg,
                Optim.Options(iterations=5000, g_tol=T(1e-12), show_trace=false);
                autodiff=:forward)

    if !Optim.converged(result)
        @warn "Optim steady state solver did not converge. " *
              "Try solver=:ipopt with JuMP + Ipopt for large-scale NLP."
    end

    y_ss = Vector{T}(Optim.minimizer(result))

    # Verify residuals are near zero
    final_obj = obj(y_ss)
    if final_obj > T(1e-6)
        @warn "Optim converged to non-zero residual (||F||² = $(final_obj)). " *
              "Steady state may not be accurate."
    end

    return y_ss
end
```

- [ ] **Step 2: Add `_nlopt_steady_state` function**

Insert right after `_optim_steady_state`:

```julia
"""
    _nlopt_steady_state(spec, constraints; initial_guess=nothing, algorithm=nothing)

Constrained steady state via NLopt LD_SLSQP.
Handles both VariableBound (box) and NonlinearConstraint (inequality).
"""
function _nlopt_steady_state(spec::DSGESpec{T}, constraints::Vector;
        initial_guess::Union{Nothing,AbstractVector}=nothing,
        algorithm=nothing) where {T<:AbstractFloat}

    n = spec.n_endog
    # NLopt.jl only supports Float64 — convert everything, then convert back at return
    if initial_guess !== nothing
        y0 = Float64.(initial_guess)
    elseif !isempty(spec.steady_state)
        y0 = Float64.(spec.steady_state)
    else
        y0 = ones(Float64, n)
    end
    @assert length(y0) == n "initial_guess must have length $n"

    # Build objective: sum of squared residuals
    ss_obj_splat = _build_ss_objective(spec.residual_fns, spec.n_exog, spec.param_values)
    nlopt_obj = _nlopt_wrap(ss_obj_splat)

    # Choose algorithm
    alg_sym = algorithm !== nothing ? algorithm : :LD_SLSQP
    opt = NLopt.Opt(alg_sym, n)

    # Set objective
    NLopt.min_objective!(opt, nlopt_obj)

    # Box bounds
    lower, upper = _extract_bounds(spec, constraints)
    NLopt.lower_bounds!(opt, Float64.(lower))
    NLopt.upper_bounds!(opt, Float64.(upper))

    # Nonlinear inequality constraints: fn(y, y, y, 0, θ) <= 0
    for c in constraints
        if c isa NonlinearConstraint
            ss_nlcon_splat = _build_ss_nlcon(c.fn, spec.n_exog, spec.param_values)
            nlopt_con = _nlopt_wrap(ss_nlcon_splat)
            NLopt.inequality_constraint!(opt, nlopt_con, 1e-8)
        end
    end

    # Tolerances
    NLopt.xtol_rel!(opt, 1e-12)
    NLopt.ftol_rel!(opt, 1e-12)
    NLopt.maxeval!(opt, 5000)

    # Clamp initial guess
    lo_f = Float64.(lower)
    hi_f = Float64.(upper)
    for i in 1:n
        y0[i] = clamp(y0[i], isfinite(lo_f[i]) ? lo_f[i] : -1e10,
                               isfinite(hi_f[i]) ? hi_f[i] : 1e10)
    end

    (min_val, min_x, ret) = NLopt.optimize(opt, y0)

    if ret ∉ (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED, :STOPVAL_REACHED)
        throw(ErrorException(
            "NLopt steady state solver did not converge (return code: $ret). " *
            "Try solver=:ipopt with JuMP + Ipopt for large-scale NLP."))
    end

    # Verify residuals are near zero
    if min_val > 1e-10
        throw(ErrorException(
            "NLopt converged to a non-zero residual (||F||² = $(min_val)). " *
            "Steady state not found. Try solver=:ipopt or a different initial_guess."))
    end

    return Vector{T}(min_x)
end
```

- [ ] **Step 3: Update dispatch in `compute_steady_state`**

Replace the constraint dispatch block (lines 39–67) with:

```julia
    if !isempty(constraints)
        _validate_constraints(spec, constraints)
        chosen = _select_solver(constraints, solver)
        chosen ∉ (:path, :ipopt, :nonlinearsolve, :optim, :nlopt) &&
            throw(ArgumentError("Unknown solver :$chosen. " *
                "Valid options: :nonlinearsolve, :optim, :nlopt, :path, :ipopt"))

        if chosen == :nonlinearsolve
            lower, upper = _extract_bounds(spec, constraints)
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "NonlinearSolve solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt (default) or solver=:ipopt for nonlinear inequality constraints."))
            y_ss = _nonlinearsolve_steady_state(spec, lower, upper;
                        initial_guess=initial_guess, algorithm=algorithm)
            # Check if bounds are violated — escalate to Optim if needed
            bounds_ok = all(i -> (!isfinite(lower[i]) || y_ss[i] >= lower[i] - T(1e-6)) &&
                                  (!isfinite(upper[i]) || y_ss[i] <= upper[i] + T(1e-6)),
                            1:n)
            if !bounds_ok
                y_ss = _optim_steady_state(spec, lower, upper;
                            initial_guess=initial_guess, algorithm=nothing)
            end
        elseif chosen == :optim
            lower, upper = _extract_bounds(spec, constraints)
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "Optim solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt or solver=:ipopt."))
            y_ss = _optim_steady_state(spec, lower, upper;
                        initial_guess=initial_guess, algorithm=algorithm)
        elseif chosen == :nlopt
            y_ss = _nlopt_steady_state(spec, constraints;
                        initial_guess=initial_guess, algorithm=algorithm)
        elseif chosen == :path
            _check_jump_loaded()
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "PATH solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt or solver=:ipopt."))
            y_ss = _path_compute_steady_state(spec, constraints;
                        initial_guess=initial_guess)
        else  # :ipopt
            _check_jump_loaded()
            y_ss = _jump_compute_steady_state(spec, constraints;
                        initial_guess=initial_guess)
        end
        return _update_steady_state(spec, Vector{T}(y_ss))
    end
```

- [ ] **Step 4: Verify load**

```bash
julia --project=. -e 'using MacroEconometricModels; println("OK")'
```

- [ ] **Step 5: Commit**

```bash
git add src/dsge/steady_state.jl
git commit -m "feat(dsge): add Optim.jl and NLopt.jl steady state solvers (#91)"
```

---

### Task 4: Add `_projected_newton_pf` and `_nlopt_perfect_foresight`

**Files:**
- Modify: `src/dsge/perfect_foresight.jl`

- [ ] **Step 1: Add `_projected_newton_pf` function**

Insert after `_nonlinearsolve_perfect_foresight` (after line 201), before the residual section:

```julia
# =============================================================================
# Projected Newton solver for box-constrained perfect foresight
# =============================================================================

"""
    _projected_newton_pf(spec, T_periods, shocks, lower, upper;
                          max_iter=100, tol=1e-8)

Box-constrained perfect foresight via projected Newton with Armijo backtracking.

Uses the existing block-tridiagonal sparse Jacobian (`_pf_jacobian`) for the
Newton step, then clamps to bounds. Preserves O(T·n) sparsity structure.
"""
function _projected_newton_pf(spec::DSGESpec{FT}, T_periods::Int,
        shocks::Matrix{FT}, lower::Vector{FT}, upper::Vector{FT};
        max_iter::Int=100, tol::Real=1e-8) where {FT<:AbstractFloat}

    n = spec.n_endog
    N = T_periods * n

    # Stack bounds: repeat per-variable bounds across all periods
    lower_stacked = repeat(lower, T_periods)
    upper_stacked = repeat(upper, T_periods)

    # Initial guess: steady state, clamped to bounds
    x = repeat(spec.steady_state, T_periods)
    x .= clamp.(x, lower_stacked, upper_stacked)

    F = zeros(FT, N)
    _pf_residual!(F, x, spec, shocks, T_periods)
    merit = dot(F, F)

    converged = false
    iter = 0
    c_armijo = FT(1e-4)

    for k in 1:max_iter
        iter = k

        # Build sparse Jacobian and compute Newton direction
        J = _pf_jacobian(x, spec, shocks, T_periods)
        d = -(J \ F)  # Newton step (sparse LU)

        # Armijo backtracking line search on merit = ||F(x)||^2
        # Directional derivative: ∇merit · d = 2 * F' * J * d = -2 * F' * F (at Newton step)
        dir_deriv = FT(2) * dot(F, J * d)
        α = FT(1.0)
        x_trial = similar(x)
        F_trial = similar(F)

        for _ls in 1:20
            x_trial .= clamp.(x .+ α .* d, lower_stacked, upper_stacked)
            _pf_residual!(F_trial, x_trial, spec, shocks, T_periods)
            merit_trial = dot(F_trial, F_trial)
            if merit_trial <= merit + c_armijo * α * dir_deriv
                break
            end
            α *= FT(0.5)
        end

        x .= clamp.(x .+ α .* d, lower_stacked, upper_stacked)
        _pf_residual!(F, x, spec, shocks, T_periods)
        merit = dot(F, F)

        if sqrt(merit) < FT(tol)
            converged = true
            break
        end
    end

    if !converged
        throw(ErrorException(
            "Projected Newton PF did not converge after $max_iter iterations " *
            "(||F|| = $(sqrt(merit))). Try solver=:ipopt with JuMP + Ipopt."))
    end

    # Reshape solution
    path_full = reshape(copy(x), n, T_periods)'
    deviations_full = path_full .- spec.steady_state'

    if spec.augmented
        orig_idx = _original_var_indices(spec)
        path = Matrix{FT}(path_full[:, orig_idx])
        deviations = Matrix{FT}(deviations_full[:, orig_idx])
    else
        path = Matrix{FT}(path_full)
        deviations = Matrix{FT}(deviations_full)
    end

    PerfectForesightPath{FT}(path, deviations, converged, iter, spec)
end
```

- [ ] **Step 2: Add `_nlopt_perfect_foresight` function**

Insert right after `_projected_newton_pf`:

```julia
# =============================================================================
# NLopt solver for nonlinear-constrained perfect foresight
# =============================================================================

"""
    _nlopt_perfect_foresight(spec, T_periods, shocks, constraints; algorithm=nothing)

Perfect foresight with nonlinear inequality constraints via NLopt LD_SLSQP.

Formulates as a feasibility problem with equality constraints (model equations)
and inequality constraints (NonlinearConstraint). Box bounds from VariableBound.
"""
function _nlopt_perfect_foresight(spec::DSGESpec{FT}, T_periods::Int,
        shocks::Matrix{FT}, constraints::Vector;
        algorithm=nothing) where {FT<:AbstractFloat}

    n = spec.n_endog
    n_ε = spec.n_exog
    N = T_periods * n
    θ = spec.param_values
    y_ss = Float64.(spec.steady_state)
    shocks_f = Float64.(shocks)

    # Warn for large problems
    if N > 1000
        @warn "NLopt PF with $N decision variables may be slow. " *
              "Consider solver=:ipopt with JuMP + Ipopt for large problems."
    end

    alg_sym = algorithm !== nothing ? algorithm : :LD_SLSQP
    opt = NLopt.Opt(alg_sym, N)

    # Objective: constant zero (feasibility problem)
    NLopt.min_objective!(opt, (x, grad) -> begin
        if length(grad) > 0
            fill!(grad, 0.0)
        end
        return 0.0
    end)

    # Box bounds: stack per-variable bounds across all periods
    lower, upper = _extract_bounds(spec, constraints)
    NLopt.lower_bounds!(opt, repeat(Float64.(lower), T_periods))
    NLopt.upper_bounds!(opt, repeat(Float64.(upper), T_periods))

    # Equality constraints: model equations f_i(y_t, y_{t-1}, y_{t+1}, ε_t, θ) = 0
    for t in 1:T_periods
        for i in 1:n
            pf_eq = _build_pf_equation(spec.residual_fns[i], n, n_ε, θ)
            cb = _pf_nlopt_wrap(pf_eq, t, n, n_ε, T_periods, y_ss, shocks_f)
            NLopt.equality_constraint!(opt, cb, 1e-8)
        end
    end

    # Inequality constraints: NonlinearConstraint fn(...) <= 0
    for c in constraints
        if c isa NonlinearConstraint
            for t in 1:T_periods
                pf_nlcon = _build_pf_nlcon(c.fn, n, n_ε, θ)
                cb = _pf_nlopt_wrap(pf_nlcon, t, n, n_ε, T_periods, y_ss, shocks_f)
                NLopt.inequality_constraint!(opt, cb, 1e-8)
            end
        end
    end

    # Tolerances
    NLopt.xtol_rel!(opt, 1e-10)
    NLopt.ftol_rel!(opt, 1e-10)
    NLopt.maxeval!(opt, 3000)

    # Initial guess: steady state, clamped to bounds
    x0 = repeat(y_ss, T_periods)
    lo_stacked = repeat(Float64.(lower), T_periods)
    hi_stacked = repeat(Float64.(upper), T_periods)
    x0 .= clamp.(x0, lo_stacked, hi_stacked)

    (min_val, min_x, ret) = NLopt.optimize(opt, x0)

    if ret ∉ (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED, :STOPVAL_REACHED)
        throw(ErrorException(
            "NLopt PF solver did not converge (return code: $ret). " *
            "Try solver=:ipopt with JuMP + Ipopt for large-scale NLP."))
    end

    converged = true
    x = Vector{FT}(min_x)

    # Reshape solution
    path_full = reshape(copy(x), n, T_periods)'
    deviations_full = path_full .- spec.steady_state'

    if spec.augmented
        orig_idx = _original_var_indices(spec)
        path = Matrix{FT}(path_full[:, orig_idx])
        deviations = Matrix{FT}(deviations_full[:, orig_idx])
    else
        path = Matrix{FT}(path_full)
        deviations = Matrix{FT}(deviations_full)
    end

    PerfectForesightPath{FT}(path, deviations, converged, 0, spec)
end
```

- [ ] **Step 3: Update PF dispatch logic**

Replace the constraint dispatch block in `perfect_foresight` (lines 67–112) with:

```julia
    if !isempty(constraints)
        _validate_constraints(spec, constraints)
        chosen = _select_solver(constraints, solver)

        if chosen == :nonlinearsolve
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "NonlinearSolve solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt (default) or solver=:ipopt for nonlinear inequality constraints."))
            lower, upper = _extract_bounds(spec, constraints)
            pf = _nonlinearsolve_perfect_foresight(spec, T_periods, shocks;
                        max_iter=max_iter, tol=tol, algorithm=algorithm)
            # Check if bounds are violated in the unconstrained solution
            bounds_ok = true
            for t in 1:T_periods, i in 1:n
                v = pf.path[t, i]
                if (isfinite(lower[i]) && v < lower[i] - FT(1e-6)) ||
                   (isfinite(upper[i]) && v > upper[i] + FT(1e-6))
                    bounds_ok = false
                    break
                end
            end
            bounds_ok && return pf
            # Escalate to projected Newton (always available)
            return _projected_newton_pf(spec, T_periods, shocks, lower, upper;
                        max_iter=max_iter, tol=tol)
        elseif chosen == :nlopt
            return _nlopt_perfect_foresight(spec, T_periods, shocks, constraints;
                        algorithm=algorithm)
        elseif chosen == :path
            _check_jump_loaded()
            any(c -> c isa NonlinearConstraint, constraints) &&
                throw(ArgumentError(
                    "PATH solver cannot handle NonlinearConstraint. " *
                    "Use solver=:nlopt or solver=:ipopt."))
            return _path_perfect_foresight(spec, T_periods, shocks, constraints)
        elseif chosen == :ipopt
            _check_jump_loaded()
            return _jump_perfect_foresight(spec, T_periods, shocks, constraints)
        else
            throw(ArgumentError("Unknown solver :$chosen. " *
                "Valid options: :nonlinearsolve, :nlopt, :path, :ipopt"))
        end
    end
```

- [ ] **Step 4: Verify load**

```bash
julia --project=. -e 'using MacroEconometricModels; println("OK")'
```

- [ ] **Step 5: Commit**

```bash
git add src/dsge/perfect_foresight.jl
git commit -m "feat(dsge): add projected Newton PF and NLopt PF solvers (#91)"
```

---

### Task 5: Add Tests

**Files:**
- Modify: `test/dsge/test_dsge.jl`

- [ ] **Step 1: Add Optim/NLopt SS and PF tests**

Insert in the "DSGE Constraint Types" testset, after the "Backward compatibility" testset (after line 5067) and before the "Solver auto-detection" testset:

```julia
@testset "Box-constrained SS via Optim.jl" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    # SS is 0; lower=0.5 forces Optim to find constrained minimum
    spec_c = compute_steady_state(spec;
        constraints=[variable_bound(:y, lower=0.5)], solver=:optim)
    @test spec_c.steady_state[1] >= 0.5 - 1e-4
end

@testset "Box-constrained SS via NLopt" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    spec_c = compute_steady_state(spec;
        constraints=[variable_bound(:y, lower=0.5)], solver=:nlopt)
    @test spec_c.steady_state[1] >= 0.5 - 1e-4
end

@testset "Nonlinear-constrained SS via NLopt" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    # Box bound + nonlinear constraint: y >= 0.3 AND (y - 0.5) <= 0 → y <= 0.5
    constraints = [
        variable_bound(:y, lower=0.3),
        nonlinear_constraint((y, yl, yld, e, th) -> y[1] - 0.5; label="upper_nl")
    ]
    spec_c = compute_steady_state(spec; constraints=constraints, solver=:nlopt)
    @test spec_c.steady_state[1] >= 0.3 - 1e-4
    @test spec_c.steady_state[1] <= 0.5 + 1e-4
end

@testset "SS escalation: NonlinearSolve → Optim.jl" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    # SS is 0; lower=0.5 should trigger escalation from NonlinearSolve to Optim
    spec_c = compute_steady_state(spec;
        constraints=[variable_bound(:y, lower=0.5)], solver=:nonlinearsolve)
    @test spec_c.steady_state[1] >= 0.5 - 1e-3
end

@testset "Box-constrained PF via projected Newton" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    shocks = zeros(30, 1)
    shocks[1, 1] = -3.0
    # Binding: unconstrained goes below 0; projected Newton should enforce y >= 0
    pf = solve(spec; method=:perfect_foresight, T_periods=30, shock_path=shocks,
               constraints=[variable_bound(:y, lower=0.0)])
    @test pf isa PerfectForesightPath
    @test pf.converged
    @test all(pf.path[:, 1] .>= -1e-4)
end

@testset "PF escalation: NonlinearSolve → projected Newton" begin
    _suppress_warnings() do
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    shocks = zeros(20, 1)
    shocks[1, 1] = -3.0
    # Escalation path: NonlinearSolve unconstrained → bounds violated → projected Newton
    pf = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks,
               constraints=[variable_bound(:y, lower=0.0)], solver=:nonlinearsolve)
    @test pf isa PerfectForesightPath
    @test pf.converged
    @test all(pf.path[:, 1] .>= -1e-4)
    end
end

@testset "NLopt PF with mixed constraints" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 1.0
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    shocks = zeros(15, 1)
    shocks[1, 1] = -2.0
    # Box bound + nonlinear constraint
    constraints = [
        variable_bound(:y, lower=-0.5),
        nonlinear_constraint((y, yl, yld, e, th) -> y[1] - 0.5; label="cap_y")
    ]
    pf = solve(spec; method=:perfect_foresight, T_periods=15, shock_path=shocks,
               constraints=constraints, solver=:nlopt)
    @test pf isa PerfectForesightPath
    @test all(pf.path[:, 1] .>= -0.5 - 1e-4)
    @test all(pf.path[:, 1] .<= 0.5 + 1e-4)
end

@testset "Explicit solver=:ipopt/:path errors without JuMP" begin
    spec = @dsge begin
        parameters: ρ = 0.9, σ = 0.01
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state: [0.0]
    end
    spec = compute_steady_state(spec)
    constraints = [variable_bound(:y, lower=0.0)]
    # These should error if JuMP is not loaded (in CI they may be loaded)
    if !hasmethod(MacroEconometricModels._jump_compute_steady_state, Tuple{DSGESpec, Vector})
        @test_throws ArgumentError compute_steady_state(spec;
            constraints=constraints, solver=:ipopt)
    end
end
```

- [ ] **Step 2: Update the "Solver auto-detection" testset**

Replace the existing testset (lines 5069–5083) to account for the new behavior:

```julia
@testset "Solver auto-detection" begin
    bounds_only = [variable_bound(:y, lower=0.0)]
    mixed = [variable_bound(:y, lower=0.0),
             nonlinear_constraint((y, yl, yld, e, th) -> y[1] - 1.0; label="test")]

    # Bounds-only → :nonlinearsolve (default)
    @test MacroEconometricModels._select_solver(bounds_only, nothing) == :nonlinearsolve

    # With NonlinearConstraints → :nlopt when JuMP not loaded, :ipopt when loaded
    result = MacroEconometricModels._select_solver(mixed, nothing)
    @test result ∈ (:nlopt, :ipopt)

    # User override always wins
    @test MacroEconometricModels._select_solver(bounds_only, :ipopt) == :ipopt
    @test MacroEconometricModels._select_solver(mixed, :path) == :path
    @test MacroEconometricModels._select_solver(bounds_only, :optim) == :optim
    @test MacroEconometricModels._select_solver(mixed, :nlopt) == :nlopt
end
```

- [ ] **Step 3: Run DSGE tests**

Per CLAUDE.md: run only the relevant test file, not the full suite.

```bash
julia --project=. test/dsge/test_dsge.jl
```

Expected: All existing tests pass, plus 9 new tests pass.

- [ ] **Step 4: Commit**

```bash
git add test/dsge/test_dsge.jl
git commit -m "test(dsge): add Optim.jl/NLopt.jl constrained DSGE solver tests (#91)"
```

---

### Task 6: Final Verification

- [ ] **Step 1: Run full DSGE test suite**

```bash
julia --project=. test/dsge/test_dsge.jl
```

Expected: All tests pass (existing + new).

- [ ] **Step 2: Verify package loads cleanly**

```bash
julia --project=. -e '
using MacroEconometricModels
spec = @dsge begin
    parameters: ρ = 0.9, σ = 1.0
    endogenous: y
    exogenous: ε
    y[t] = ρ * y[t-1] + σ * ε[t]
    steady_state: [0.0]
end
spec = compute_steady_state(spec)

# Test Optim SS
spec_c = compute_steady_state(spec; constraints=[variable_bound(:y, lower=0.5)], solver=:optim)
println("Optim SS: y = $(spec_c.steady_state[1]) (expected ≥ 0.5)")

# Test NLopt SS
spec_n = compute_steady_state(spec; constraints=[variable_bound(:y, lower=0.5)], solver=:nlopt)
println("NLopt SS: y = $(spec_n.steady_state[1]) (expected ≥ 0.5)")

# Test projected Newton PF
shocks = zeros(20, 1); shocks[1,1] = -3.0
pf = solve(spec; method=:perfect_foresight, T_periods=20, shock_path=shocks,
           constraints=[variable_bound(:y, lower=0.0)])
println("PF converged: $(pf.converged), min y = $(minimum(pf.path[:,1])) (expected ≥ 0)")
println("All OK!")
'
```

- [ ] **Step 3: Commit everything if any unstaged changes**

```bash
git status
# If clean, no commit needed. Otherwise stage specific changed files:
# git add <specific files> && git commit -m "fix(dsge): address any remaining issues from constrained solver tests (#91)"
```
