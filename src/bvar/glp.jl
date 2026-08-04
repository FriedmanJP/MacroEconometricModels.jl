# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Giannone, Lenza & Primiceri (2015) hierarchical hyperparameter selection (T252 / #351).

GLP treat the Minnesota hyperparameters as random with hyperpriors and choose them by
maximizing the **posterior** `p(γ|Y) ∝ p(Y|γ)p(γ)` — the closed-form conjugate marginal
likelihood times Gamma hyperpriors — rather than grid-searching the overall tightness alone
and leaving the rest at defaults. A one-dimensional grid over `tau` cannot trade overall
tightness off against the sum-of-coefficients and dummy-initial-observation priors, which is
how an optimizer ends up reporting an extreme, unconverged shrinkage value.

Provides:
- `GLPHyperparameters` — optimized hyperparameters plus the diagnostics that say whether to
  trust them
- `optimize_hyperparameters_glp` — the joint optimizer

References:
- Giannone, D., Lenza, M. & Primiceri, G. E. (2015). Prior Selection for Vector
  Autoregressions. *Review of Economics and Statistics*, 97(2), 436-451.
"""

using LinearAlgebra
using Statistics
using Distributions

# =============================================================================
# Hyperprior parameterization
# =============================================================================

"""
    _gamma_from_mode_sd(mode, sd) -> (shape, scale)

Gamma parameters implied by a mode/standard-deviation pair, the parameterization GLP use
for their hyperpriors. For `Gamma(k, θ)` the mode is `(k-1)θ` and the variance `kθ²`, so
`θ` solves `θ² + mode·θ − sd² = 0` and `k = mode/θ + 1`.
"""
function _gamma_from_mode_sd(mode::T, sd::T) where {T<:AbstractFloat}
    mode > 0 || throw(ArgumentError("hyperprior mode must be positive, got $mode"))
    sd > 0 || throw(ArgumentError("hyperprior sd must be positive, got $sd"))
    theta = (-mode + sqrt(mode^2 + 4 * sd^2)) / 2
    shape = mode / theta + one(T)
    return (shape, theta)
end

"""
GLP (2015) hyperprior modes and standard deviations, in this package's hyperparameter
names. `tau` is GLP's overall shrinkage `λ`, `lambda` their sum-of-coefficients `μ`, and
`mu` their dummy-initial-observation `δ`.
"""
const _GLP_HYPERPRIORS = (
    tau    = (mode = 0.2, sd = 0.4),
    lambda = (mode = 1.0, sd = 1.0),
    mu     = (mode = 1.0, sd = 1.0),
)

"""Sensible optimization box; a hyperparameter pinned to an edge is reported, not hidden."""
const _GLP_BOUNDS = (
    tau    = (0.005, 50.0),
    lambda = (0.01, 100.0),
    mu     = (0.01, 100.0),
)

# =============================================================================
# Result container
# =============================================================================

"""
    GLPHyperparameters{T}

Result of the Giannone-Lenza-Primiceri hierarchical hyperparameter optimization.

# Fields
- `hyper::MinnesotaHyperparameters{T}` — the optimized hyperparameters
- `log_ml::T` — log marginal likelihood at the optimum
- `log_posterior::T` — `log_ml` plus the log hyperprior density, the maximized objective
- `converged::Bool` — the optimizer converged **and** no hyperparameter sits on a bound
- `at_bound::Bool` — some hyperparameter is pinned to the edge of the search box
- `iterations::Int` — optimizer iterations
- `log_ml_default::T` — log marginal likelihood at the package defaults, for comparison

A result with `converged == false` must not be treated as an estimate: that is exactly how
a wildly out-of-range shrinkage value gets reported downstream as if it were selected.
"""
struct GLPHyperparameters{T<:AbstractFloat}
    hyper::MinnesotaHyperparameters{T}
    log_ml::T
    log_posterior::T
    converged::Bool
    at_bound::Bool
    iterations::Int
    log_ml_default::T
end

function Base.show(io::IO, r::GLPHyperparameters{T}) where {T}
    h = r.hyper
    data = Any[
        "τ (overall tightness)"        _fmt(h.tau; digits=4);
        "λ (sum-of-coefficients)"      _fmt(h.lambda; digits=4);
        "μ (dummy initial obs.)"       _fmt(h.mu; digits=4);
        "decay (lag)"                  _fmt(h.decay; digits=4);
        "ω (covariance)"               _fmt(h.omega; digits=4);
        "log marginal likelihood"      _fmt(r.log_ml; digits=3);
        "log posterior (with priors)"  _fmt(r.log_posterior; digits=3);
        "log ML at defaults"           _fmt(r.log_ml_default; digits=3);
        "Iterations"                   r.iterations;
        "Converged"                    r.converged ? "Yes" : "No";
        "On a bound"                   r.at_bound ? "Yes" : "No"
    ]
    _pretty_table(io, data;
        title = "GLP (2015) Hyperparameter Optimization",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    r.converged || println(io, "Optimization did NOT converge" *
                               (r.at_bound ? " (a hyperparameter is pinned to a bound)" : "") *
                               ". Treat these values as unselected — supply `hyper=` " *
                               "explicitly or widen the sample.")
    return nothing
end

# =============================================================================
# Objective
# =============================================================================

"""
    _glp_log_hyperprior(tau, lambda, mu, ::Type{T}) -> T

Sum of the GLP Gamma log-hyperprior densities at the three optimized hyperparameters.
"""
function _glp_log_hyperprior(tau::T, lambda::T, mu::T, ::Type{T}) where {T<:AbstractFloat}
    lp = zero(T)
    for (val, spec) in ((tau, _GLP_HYPERPRIORS.tau),
                        (lambda, _GLP_HYPERPRIORS.lambda),
                        (mu, _GLP_HYPERPRIORS.mu))
        val > 0 || return T(-Inf)
        k, theta = _gamma_from_mode_sd(T(spec.mode), T(spec.sd))
        lp += T(logpdf(Gamma(k, theta), val))
    end
    return lp
end

"""
    _glp_objective(x, Y, p, decay, omega, ::Type{T}) -> T

Negative log posterior of the hyperparameters at `x = log.([tau, lambda, mu])`. Optimizing
in log space keeps the hyperparameters positive without a constrained solver; the box is
enforced by returning a large penalty outside it, so a pinned optimum is visible rather than
silently clipped.
"""
function _glp_objective(x::Vector{T}, Y::Matrix{T}, p::Int,
                        decay::T, omega::T, ::Type{T}) where {T<:AbstractFloat}
    penalty = T(1e12)
    tau, lambda, mu = exp(x[1]), exp(x[2]), exp(x[3])
    for (val, b) in ((tau, _GLP_BOUNDS.tau), (lambda, _GLP_BOUNDS.lambda),
                     (mu, _GLP_BOUNDS.mu))
        (b[1] <= val <= b[2]) || return penalty
    end
    h = MinnesotaHyperparameters(; tau=tau, decay=decay, lambda=lambda, mu=mu, omega=omega)
    ml = try
        log_marginal_likelihood(Y, p, h)
    catch err
        err isa ArgumentError && return penalty
        rethrow(err)
    end
    isfinite(ml) || return penalty
    lp = _glp_log_hyperprior(tau, lambda, mu, T)
    isfinite(lp) || return penalty
    return -(ml + lp)
end

# =============================================================================
# Optimizer
# =============================================================================

"""
    optimize_hyperparameters_glp(Y, p; kwargs...) -> GLPHyperparameters

Joint Giannone-Lenza-Primiceri (2015) hyperparameter selection: maximize the log marginal
likelihood of the conjugate Minnesota BVAR plus GLP's Gamma hyperpriors, over the overall
tightness `tau`, the sum-of-coefficients tightness `lambda`, and the dummy-initial-
observation tightness `mu`, jointly.

```math
\\hat\\gamma = \\arg\\max_\\gamma\\ \\log p(Y\\mid\\gamma) + \\log p(\\gamma)
```

where:
- ``p(Y\\mid\\gamma)`` is the closed-form conjugate marginal likelihood of the
  dummy-observation representation ([`log_marginal_likelihood`](@ref))
- ``p(\\gamma)`` are Gamma hyperpriors in GLP's mode/standard-deviation parameterization:
  `tau` mode 0.2 sd 0.4, `lambda` mode 1 sd 1, `mu` mode 1 sd 1

This differs from [`optimize_hyperparameters`](@ref), which grid-searches `tau` alone and
leaves the others at their defaults — a search that cannot trade overall tightness against
the sum-of-coefficients and initial-observation priors, and so can drive `tau` to an
extreme value that is then reported as if it had been selected.

Optimization runs in log space (keeping every hyperparameter positive without a constrained
solver) with a derivative-free Nelder-Mead, restarted from several dispersed starting
points because the marginal-likelihood surface is not concave in the hyperparameters.
`converged` is `true` only when the optimizer converged **and** no hyperparameter sits on a
bound.

# Keywords
- `decay::Real=0.5` / `omega::Real=1.0` — held fixed (GLP fix the lag decay rather than
  estimate it); pass different values to shift them
- `starts::Int=4` — dispersed restarts of the optimizer
- `max_iter::Int=500` — iterations per restart
- `f_reltol::Real=1e-8` — relative objective tolerance
- `verbose::Bool=true` — warn on non-convergence

# Returns
[`GLPHyperparameters`](@ref).

# References
- Giannone, D., Lenza, M. & Primiceri, G. E. (2015). *Review of Economics and Statistics*,
  97(2), 436-451.
"""
function optimize_hyperparameters_glp(Y::AbstractMatrix{T}, p::Int;
                                      decay::Real=0.5, omega::Real=1.0,
                                      starts::Int=4, max_iter::Int=500,
                                      f_reltol::Real=1e-8,
                                      verbose::Bool=true) where {T<:AbstractFloat}
    p >= 1 || throw(ArgumentError("p must be at least 1, got $p"))
    starts >= 1 || throw(ArgumentError("starts must be positive"))
    Ym = Matrix{T}(Y)
    dec, om = T(decay), T(omega)

    obj = x -> _glp_objective(x, Ym, p, dec, om, T)

    # Dispersed starting points: the GLP prior modes plus tighter/looser variants. The
    # marginal-likelihood surface is not concave in the hyperparameters, so a single start
    # can settle in a local optimum.
    base = T[_GLP_HYPERPRIORS.tau.mode, _GLP_HYPERPRIORS.lambda.mode, _GLP_HYPERPRIORS.mu.mode]
    scales = T[1.0, 0.25, 4.0, 16.0]
    inits = [log.(clamp.(base .* scales[min(i, length(scales))],
                         T(1e-3), T(50))) for i in 1:starts]

    best_x = inits[1]
    best_f = T(Inf)
    best_conv = false
    best_iters = 0
    for x0 in inits
        res = Optim.optimize(obj, x0, Optim.NelderMead(),
                             Optim.Options(iterations=max_iter, f_reltol=f_reltol,
                                           show_trace=false))
        f = T(Optim.minimum(res))
        if f < best_f
            best_f = f
            best_x = T.(Optim.minimizer(res))
            best_conv = Optim.converged(res)
            best_iters = Optim.iterations(res)
        end
    end

    tau, lambda, mu = exp(best_x[1]), exp(best_x[2]), exp(best_x[3])
    at_bound = any(((v, b),) -> v <= b[1] * (1 + 1e-6) || v >= b[2] * (1 - 1e-6),
                   ((tau, _GLP_BOUNDS.tau), (lambda, _GLP_BOUNDS.lambda),
                    (mu, _GLP_BOUNDS.mu)))
    converged = best_conv && !at_bound && isfinite(best_f) && best_f < T(1e11)

    hyper = MinnesotaHyperparameters(; tau=tau, decay=dec, lambda=lambda, mu=mu, omega=om)
    log_ml = try
        T(log_marginal_likelihood(Ym, p, hyper))
    catch
        T(-Inf)
    end
    log_post = converged || isfinite(best_f) ? -best_f : T(-Inf)
    log_ml_default = try
        T(log_marginal_likelihood(Ym, p, MinnesotaHyperparameters(; decay=dec, omega=om)))
    catch
        T(-Inf)
    end

    if verbose && !converged
        @warn "GLP hyperparameter optimization did not converge" tau lambda mu at_bound
    end
    return GLPHyperparameters{T}(hyper, log_ml, log_post, converged, at_bound,
                                 best_iters, log_ml_default)
end

@float_fallback optimize_hyperparameters_glp Y
