# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Public API for Bayesian DSGE estimation and posterior analysis functions.

Provides:
- `estimate_dsge_bayes` — main estimation entry point (SMC, SMC², RWMH)
- `posterior_mode` — posterior mode + Laplace marginal likelihood + RWMH proposal seed
- `posterior_summary` — posterior mean, median, std, credible intervals
- `marginal_likelihood` — log marginal likelihood accessor
- `bridge_sampling_ml` — bridge sampling log marginal likelihood (Meng & Wong 1996)
- `bayes_factor` — log Bayes factor between two models
- `prior_posterior_table` — prior vs posterior comparison
- `posterior_predictive` — simulate from posterior draws

References:
- Herbst, E. & Schorfheide, F. (2014). Sequential Monte Carlo Sampling for DSGE Models.
  Journal of Applied Econometrics, 29(7), 1073-1098.
- An, S. & Schorfheide, F. (2007). Bayesian Analysis of DSGE Models.
  Econometric Reviews, 26(2-4), 113-172.
"""

using LinearAlgebra
using Random
using Statistics
using Distributions

# =============================================================================
# Helper: infer bounds from prior distribution support
# =============================================================================

"""
    _infer_prior_bounds(d::Distribution) → (lower::Float64, upper::Float64)

Infer parameter bounds from the support of a prior distribution.
- Beta: [0, 1]
- InverseGamma, Gamma: [0, +Inf]
- Other bounded: use minimum/maximum if finite
- Default: [-Inf, +Inf]
"""
function _infer_prior_bounds(d::Distribution)
    if d isa Beta
        return (0.0, 1.0)
    elseif d isa InverseGamma || d isa Gamma
        return (0.0, Inf)
    else
        lo = try
            m = minimum(d)
            isfinite(m) ? Float64(m) : -Inf
        catch
            -Inf
        end
        hi = try
            m = maximum(d)
            isfinite(m) ? Float64(m) : Inf
        catch
            Inf
        end
        return (lo, hi)
    end
end

"""
    _build_bayes_prior(priors::Dict{Symbol,<:Distribution}) → DSGEPrior

Build a `DSGEPrior` from a priors dict, auto-inferring parameter bounds from
each distribution's support via `_infer_prior_bounds`.
"""
function _build_bayes_prior(priors::Dict{Symbol,<:Distribution})
    lower_dict = Dict{Symbol,Float64}()
    upper_dict = Dict{Symbol,Float64}()
    for (pn, d) in priors
        lo, hi = _infer_prior_bounds(d)
        lower_dict[pn] = lo
        upper_dict[pn] = hi
    end
    return DSGEPrior(priors; lower=lower_dict, upper=upper_dict)
end

"""
    _orient_bayes_data(data, n_obs, ::Type{T}) → Matrix{T}

Convert `data` to an `n_obs × T_obs` matrix (each column one time period), the
orientation the Kalman/particle filters expect. Transposes when the column
count matches `n_obs`, or best-guesses (more rows than columns → transpose)
when neither dimension matches.
"""
function _orient_bayes_data(data::AbstractMatrix, n_obs::Int, ::Type{T}) where {T<:AbstractFloat}
    data_mat = Matrix{T}(data)
    nrows, ncols = size(data_mat)
    if nrows != n_obs && ncols == n_obs
        data_mat = Matrix{T}(data_mat')
    elseif nrows != n_obs && ncols != n_obs
        if nrows > ncols
            data_mat = Matrix{T}(data_mat')
        end
    end
    return data_mat
end

# =============================================================================
# Main public API
# =============================================================================

"""
    estimate_dsge_bayes(spec, data, θ0; priors, method=:smc, ...) → BayesianDSGE

Bayesian estimation of DSGE model parameters via Sequential Monte Carlo (SMC),
SMC², or Random-Walk Metropolis-Hastings (RWMH).

# Arguments
- `spec::DSGESpec{T}` — model specification from `@dsge` macro
- `data::AbstractMatrix` — observed data (T_obs × n_obs or n_obs × T_obs, auto-detected)
- `θ0::AbstractVector{<:Real}` — initial parameter guess (length must match number of priors)

# Keywords
- `priors::Dict{Symbol,<:Distribution}` — prior distributions keyed by parameter name
- `method::Symbol=:smc` — estimation method: `:smc`, `:smc2`, or `:mh`
- `observables::Vector{Symbol}=Symbol[]` — which endogenous variables are observed
  (default: all `spec.endog`)
- `n_smc::Int=5000` — number of SMC particles (for `:smc` and `:smc2`)
- `n_particles::Int=500` — number of PF particles (for `:smc2` only)
- `n_mh_steps::Int=1` — MH mutation steps per SMC stage
- `n_draws::Int=10000` — total draws for `:mh` (including burnin)
- `burnin::Int=5000` — burn-in draws discarded from `:mh` output; the posterior uses `n_draws - burnin`
- `keep_burnin::Bool=false` — if true, retain the full `:mh` chain (e.g. for trace plots)
- `ess_target::Float64=0.5` — target ESS fraction for adaptive tempering
- `measurement_error::Union{Nothing,Vector{<:Real}}=nothing` — measurement error SDs
- `likelihood::Symbol=:auto` — likelihood evaluation method (currently auto = Kalman)
- `solver::Symbol=:gensys` — DSGE solver method
- `solver_kwargs::NamedTuple=NamedTuple()` — additional solver keyword arguments
- `delayed_acceptance::Bool=false` — enable two-stage delayed acceptance MH for `:smc2`
  (Christen & Fox 2005). Pre-screens proposals with a cheap bootstrap PF to avoid
  expensive CSMC evaluations on proposals that would be rejected. Exact posterior.
- `n_screen::Int=200` — particles for screening PF (only used when `delayed_acceptance=true`)
- `proposal::Symbol=:adaptive` — RWMH proposal initialization (`:mh` only). `:adaptive`
  keeps the current behavior (identity proposal, adapted along the chain). `:mode` first
  runs [`posterior_mode`](@ref) and seeds the chain at the mode with proposal covariance
  `c²·H⁻¹` where `c = 2.38/√d` (Roberts & Rosenthal 2001 optimal scaling) and `H` is the
  negative-log-posterior Hessian at the mode.
- `transform::Bool=true` — RWMH walks in the prior-transformed unconstrained space
  (`:mh` only): log for positive supports, logit for bounded intervals, with the
  log-Jacobian added to the target so the θ-space posterior is preserved. No proposal
  is ever wasted outside the parameter support. `transform=false` restores the
  natural-space walk.
- `rng::AbstractRNG=Random.default_rng()` — random number generator

# Returns
`BayesianDSGE{T}` containing posterior draws, log posterior, marginal likelihood, etc.

# References
- Herbst, E. & Schorfheide, F. (2014). Sequential Monte Carlo Sampling for DSGE Models.
  *Journal of Applied Econometrics*, 29(7), 1073-1098.
- An, S. & Schorfheide, F. (2007). Bayesian Analysis of DSGE Models.
  *Econometric Reviews*, 26(2-4), 113-172.
"""
function estimate_dsge_bayes(spec::DSGESpec{T}, data::AbstractMatrix,
                              theta0::AbstractVector{<:Real};
                              priors::Dict{Symbol,<:Distribution},
                              method::Symbol=:smc,
                              observables::Vector{Symbol}=Symbol[],
                              n_smc::Int=5000,
                              n_particles::Int=500,
                              n_mh_steps::Int=1,
                              n_draws::Int=10000,
                              burnin::Int=5000,
                              keep_burnin::Bool=false,
                              ess_target::Float64=0.5,
                              measurement_error::Union{Nothing,Vector{<:Real}}=nothing,
                              likelihood::Symbol=:auto,
                              solver::Symbol=:gensys,
                              solver_kwargs::NamedTuple=NamedTuple(),
                              delayed_acceptance::Bool=false,
                              n_screen::Int=200,
                              proposal::Symbol=:adaptive,
                              transform::Bool=true,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}

    # ── 1. Build DSGEPrior from priors dict (bounds inferred from support) ─
    prior = _build_bayes_prior(priors)

    # ── 2. Sort param_names to match DSGEPrior ordering ──────────────────
    param_names = prior.param_names  # already sorted by DSGEPrior constructor

    # theta0 is provided in the same order as the sorted prior keys
    theta0_sorted = T.(theta0)

    # ── 3. Handle observables ────────────────────────────────────────────
    if isempty(observables)
        observables = copy(spec.endog)
    end
    n_obs = length(observables)

    # ── 4. Data handling: ensure n_obs × T_obs format ────────────────────
    # The Kalman filter expects n_obs × T_obs (each column is one time period);
    # simulate() returns T_periods × n_endog.
    data_mat = _orient_bayes_data(data, n_obs, T)

    # ── 5. Validate method / proposal ────────────────────────────────────
    if method ∉ (:smc, :smc2, :mh)
        throw(ArgumentError("method must be :smc, :smc2, or :mh, got :$method"))
    end
    if proposal ∉ (:adaptive, :mode)
        throw(ArgumentError("proposal must be :adaptive or :mode, got :$proposal"))
    end

    # ── 6. Dispatch to sampler ───────────────────────────────────────────
    if method == :smc
        state = _smc_sample(spec, data_mat, param_names, prior, theta0_sorted;
                             n_smc=n_smc, n_mh_steps=n_mh_steps,
                             ess_target=ess_target, observables=observables,
                             measurement_error=measurement_error,
                             solver=solver, solver_kwargs=solver_kwargs, rng=rng)
        return _smc_state_to_bayesian_dsge(state, prior, param_names, spec, :smc,
                                            observables, measurement_error,
                                            solver, solver_kwargs, data_mat)

    elseif method == :smc2
        state = _smc2_sample(spec, data_mat, param_names, prior, theta0_sorted;
                              n_smc=n_smc, n_particles=n_particles,
                              n_mh_steps=n_mh_steps, ess_target=ess_target,
                              observables=observables,
                              measurement_error=measurement_error,
                              solver=solver, solver_kwargs=solver_kwargs,
                              delayed_acceptance=delayed_acceptance,
                              n_screen=n_screen, rng=rng)
        return _smc_state_to_bayesian_dsge(state, prior, param_names, spec, :smc2,
                                            observables, measurement_error,
                                            solver, solver_kwargs, data_mat)

    else  # :mh
        theta_init = theta0_sorted
        init_proposal_cov = nothing
        if proposal == :mode
            pm = posterior_mode(spec, data_mat, theta0_sorted;
                                priors=priors, observables=observables,
                                measurement_error=measurement_error,
                                solver=solver, solver_kwargs=solver_kwargs)
            c2 = T(2.38)^2 / T(length(param_names))
            init_proposal_cov = c2 .* pm.inv_hessian
            theta_init = pm.mode
            if transform
                # The walk runs in y-space: map the θ-space covariance through
                # the inverse Jacobian at the mode, Σ_y = D⁻¹ Σ_θ D⁻¹
                pt = ParameterTransform(prior.lower, prior.upper)
                y_mode = to_unconstrained(pt, pm.mode)
                D = transform_jacobian(pt, y_mode)
                Dinv = Diagonal([D[i, i] != 0 ? 1 / D[i, i] : one(T)
                                 for i in 1:size(D, 1)])
                init_proposal_cov = Matrix{T}(Dinv * init_proposal_cov * Dinv)
            end
        end
        draws, log_posterior, acceptance_rate = _mh_sample(
            spec, data_mat, param_names, prior, theta_init;
            n_draws=n_draws, burnin=burnin,
            observables=observables,
            measurement_error=measurement_error,
            solver=solver, solver_kwargs=solver_kwargs,
            init_proposal_cov=init_proposal_cov, transform=transform, rng=rng)
        # Discard burn-in so posterior summaries exclude the transient. The burnin kwarg was
        # previously a no-op for :mh (all draws stored). keep_burnin=true retains the full
        # chain (e.g. for trace plots) without affecting the default summaries (E-03 / #122).
        if !keep_burnin && burnin > 0
            keep = min(burnin, n_draws - 1)          # always retain ≥1 posterior draw
            draws = draws[keep+1:end, :]
            log_posterior = log_posterior[keep+1:end]
        end
        return _mh_to_bayesian_dsge(draws, log_posterior, acceptance_rate,
                                     prior, param_names, spec,
                                     observables, measurement_error,
                                     solver, solver_kwargs, data_mat)
    end
end

# =============================================================================
# Posterior mode finding + Laplace marginal likelihood (Dynare-style workflow)
# =============================================================================

"""
    posterior_mode(spec, data, θ0; priors, kwargs...) → PosteriorMode

Find the posterior mode of a DSGE model by numerically maximizing
`log p(θ|Y) ∝ log L(θ) + log π(θ)`, and compute the Laplace approximation of
the log marginal likelihood plus the inverse Hessian at the mode (the
asymptotic posterior covariance, usable as an RWMH proposal via
`estimate_dsge_bayes(...; method=:mh, proposal=:mode)`).

Optimization runs in a prior-transformed unconstrained space by default
(log for positive supports, logit for bounded intervals, via
[`ParameterTransform`](@ref)), so bounded parameters never hit their
boundaries; the returned mode is in the natural parameter space. The
transform only reparameterizes the optimizer — the objective is the
natural-space log posterior, so the maximizer is the natural-space mode.

The Laplace approximation is
```math
\\log \\hat p(Y) = \\log L(\\theta^*) + \\log \\pi(\\theta^*)
    + \\tfrac{d}{2}\\log(2\\pi) - \\tfrac{1}{2}\\log\\det H,
```
where `H` is the Hessian of the negative log posterior at the mode `θ*` and
`d` the number of estimated parameters (Tierney & Kadane 1986). If `H` is not
finite and positive definite, `laplace_log_ml` is `NaN`, a warning is emitted,
and `inv_hessian` falls back to a diagonal matrix.

# Arguments
- `spec::DSGESpec{T}` — model specification from `@dsge`
- `data::AbstractMatrix` — observed data (`T_obs × n_obs` or `n_obs × T_obs`, auto-detected)
- `θ0::AbstractVector{<:Real}` — initial guess, ordered like the sorted prior keys

# Keywords
- `priors::Dict{Symbol,<:Distribution}` — prior distributions keyed by parameter name
- `observables::Vector{Symbol}=Symbol[]` — observed endogenous variables (default: all)
- `measurement_error=nothing` — measurement error SDs
- `solver::Symbol=:gensys` — DSGE solver method
- `solver_kwargs::NamedTuple=NamedTuple()` — additional solver kwargs
- `transform::Bool=true` — optimize in the unconstrained (prior-transformed) space
- `optimizer=Optim.LBFGS()` — any `Optim.jl` first-order optimizer
- `f_reltol::Real=1e-8` — relative objective tolerance (`Optim.Options` `f_reltol`)
- `max_iter::Int=500` — maximum optimizer iterations

# Returns
[`PosteriorMode`](@ref) carrying the mode, inverse Hessian, log posterior at
the mode, and the Laplace log marginal likelihood.

# References
- Tierney, L. & Kadane, J. B. (1986). Accurate Approximations for Posterior
  Moments and Marginal Densities. *JASA*, 81(393), 82-86.
- Roberts, G. O. & Rosenthal, J. S. (2001). Optimal Scaling for Various
  Metropolis-Hastings Algorithms. *Statistical Science*, 16(4), 351-367.
"""
function posterior_mode(spec::DSGESpec{T}, data::AbstractMatrix,
                        theta0::AbstractVector{<:Real};
                        priors::Dict{Symbol,<:Distribution},
                        observables::Vector{Symbol}=Symbol[],
                        measurement_error=nothing,
                        solver::Symbol=:gensys,
                        solver_kwargs::NamedTuple=NamedTuple(),
                        transform::Bool=true,
                        optimizer=Optim.LBFGS(),
                        f_reltol::Real=1e-8,
                        max_iter::Int=500) where {T<:AbstractFloat}
    prior = _build_bayes_prior(priors)
    param_names = prior.param_names
    d = length(param_names)
    length(theta0) == d ||
        throw(ArgumentError("theta0 has length $(length(theta0)), expected $d (one per prior)"))

    if isempty(observables)
        observables = copy(spec.endog)
    end
    data_mat = _orient_bayes_data(data, length(observables), T)

    # Same likelihood closure the samplers use, so mode and chain see identical evaluations
    ll_fn = _build_likelihood_fn(spec, param_names, data_mat, observables,
                                 measurement_error, solver, solver_kwargs)

    penalty = T(1e10)
    function logpost(theta::Vector{T})
        lp = _log_prior(theta, prior)
        isfinite(lp) || return -penalty
        ll = ll_fn(theta)
        isfinite(ll) || return -penalty
        return ll + lp
    end

    # Nudge θ0 strictly inside the prior support so the transform is finite
    theta_start = Vector{T}(theta0)
    for i in 1:d
        lo, hi = prior.lower[i], prior.upper[i]
        span = isfinite(lo) && isfinite(hi) ? hi - lo : one(T)
        margin = T(1e-8) * span
        if isfinite(lo) && theta_start[i] <= lo
            theta_start[i] = lo + margin
        end
        if isfinite(hi) && theta_start[i] >= hi
            theta_start[i] = hi - margin
        end
    end

    opts = Optim.Options(f_reltol=T(f_reltol), iterations=max_iter)
    local res, theta_star
    if transform
        pt = ParameterTransform(prior.lower, prior.upper)
        phi0 = to_unconstrained(pt, theta_start)
        obj_phi = phi -> -logpost(to_constrained(pt, phi))
        res = Optim.optimize(obj_phi, phi0, optimizer, opts)
        theta_star = to_constrained(pt, Optim.minimizer(res))
    else
        obj = theta -> -logpost(Vector{T}(theta))
        res = Optim.optimize(obj, theta_start, optimizer, opts)
        theta_star = Vector{T}(Optim.minimizer(res))
    end

    ll_star = ll_fn(theta_star)
    lpost_star = logpost(theta_star)

    # Hessian of the NEGATIVE log posterior at the mode, in the natural space.
    # ForwardDiff first (fast if the likelihood path is Dual-compatible), falling
    # back to central finite differences (the QZ-based solvers are not).
    negpost = theta -> -logpost(Vector{T}(theta))
    H = try
        ForwardDiff.hessian(negpost, theta_star)
    catch
        _numerical_hessian(negpost, theta_star; eps_step=T(1e-4))
    end
    H = Matrix{T}((H + H') / 2)

    Hh = Hermitian(H)
    hessian_ok = all(isfinite, H) && isposdef(Hh)
    local inv_H, laplace_log_ml
    if hessian_ok
        inv_H = Matrix{T}(robust_inv(Hh))
        laplace_log_ml = lpost_star + T(d) / 2 * log(T(2) * T(pi)) - logdet_safe(H) / 2
    else
        @warn "posterior_mode: Hessian at the mode is not finite positive definite; " *
              "Laplace log-ML set to NaN and inv_hessian falls back to a diagonal matrix"
        diag_fallback = ones(T, d)
        for i in 1:d
            hii = H[i, i]
            if isfinite(hii) && hii > 0
                diag_fallback[i] = one(T) / hii
            end
        end
        inv_H = Matrix{T}(Diagonal(diag_fallback))
        laplace_log_ml = T(NaN)
    end

    return PosteriorMode{T}(theta_star, inv_H, H, lpost_star, ll_star,
                            laplace_log_ml, param_names,
                            Optim.converged(res), Optim.iterations(res))
end

function Base.show(io::IO, pm::PosteriorMode{T}) where {T}
    d = length(pm.param_names)
    spec_data = Any[
        "Parameters"           d;
        "Log posterior"        round(pm.log_posterior; digits=4);
        "Log likelihood"       round(pm.log_likelihood; digits=4);
        "Laplace log-ML"       round(pm.laplace_log_ml; digits=4);
        "Converged"            pm.converged;
        "Iterations"           pm.n_iterations;
    ]
    _pretty_table(io, spec_data;
        title="DSGE Posterior Mode (Laplace)",
        column_labels=["", ""],
        alignment=[:l, :r])

    sds = [pm.inv_hessian[i, i] > 0 ? sqrt(pm.inv_hessian[i, i]) : T(NaN) for i in 1:d]
    mode_data = hcat([string(p) for p in pm.param_names],
                     round.(pm.mode; digits=4), round.(sds; digits=4))
    _pretty_table(io, mode_data;
        title="Posterior Mode",
        column_labels=["Parameter", "Mode", "Std (Laplace)"],
        alignment=[:l, :r, :r])
end

# =============================================================================
# Internal: convert SMCState to BayesianDSGE
# =============================================================================

"""
    _smc_state_to_bayesian_dsge(state, prior, param_names, spec, method_sym,
                                  observables, measurement_error, solver, solver_kwargs)

Convert an `SMCState{T}` into a `BayesianDSGE{T}` result container.
"""
function _smc_state_to_bayesian_dsge(state::SMCState{T}, prior::DSGEPrior{T},
                                       param_names::Vector{Symbol},
                                       spec::DSGESpec{T}, method_sym::Symbol,
                                       observables::Vector{Symbol},
                                       measurement_error,
                                       solver::Symbol,
                                       solver_kwargs::NamedTuple,
                                       data::Matrix{T}=zeros(T, 0, 0)) where {T<:AbstractFloat}
    # theta_draws: transpose from n_params × N_smc to N_smc × n_params
    theta_draws = Matrix{T}(state.theta_particles')

    # log_posterior = log_likelihoods + log_priors
    log_posterior = state.log_likelihoods .+ state.log_priors

    # Acceptance rate: last element of acceptance_rates, or 0
    acceptance_rate = isempty(state.acceptance_rates) ? zero(T) : last(state.acceptance_rates)

    # Build solution at posterior mean
    n_params = length(param_names)
    N = size(theta_draws, 1)

    # Compute weighted posterior mean
    w = exp.(state.log_weights .- _logsumexp(state.log_weights))
    theta_mean = zeros(T, n_params)
    for j in 1:N
        for i in 1:n_params
            theta_mean[i] += w[j] * theta_draws[j, i]
        end
    end

    sol, ss = _build_solution_at_theta(spec, param_names, theta_mean,
                                        observables, measurement_error,
                                        solver, solver_kwargs)

    BayesianDSGE{T}(
        theta_draws, log_posterior, param_names, prior,
        state.log_marginal_likelihood, method_sym, acceptance_rate,
        state.ess_history, state.phi_schedule,
        spec, sol, ss,
        data, observables, measurement_error, solver, solver_kwargs
    )
end

# =============================================================================
# Internal: convert MH output to BayesianDSGE
# =============================================================================

"""
    _mh_to_bayesian_dsge(draws, log_posterior, acceptance_rate,
                           prior, param_names, spec,
                           observables, measurement_error, solver, solver_kwargs)

Convert MH sampler output into a `BayesianDSGE{T}` result container.
"""
function _mh_to_bayesian_dsge(draws::Matrix{T}, log_posterior::Vector{T},
                                acceptance_rate::T,
                                prior::DSGEPrior{T},
                                param_names::Vector{Symbol},
                                spec::DSGESpec{T},
                                observables::Vector{Symbol},
                                measurement_error,
                                solver::Symbol,
                                solver_kwargs::NamedTuple,
                                data::Matrix{T}=zeros(T, 0, 0)) where {T<:AbstractFloat}
    # Posterior mean from draws
    theta_mean = vec(mean(draws; dims=1))

    sol, ss = _build_solution_at_theta(spec, param_names, theta_mean,
                                        observables, measurement_error,
                                        solver, solver_kwargs)

    # Approximate log marginal likelihood via harmonic mean estimator
    # log p(Y) ≈ -log(1/n * sum(1/p(Y|θ_i))) = -log(mean(exp(-log_lik)))
    # This is a simple approximation; SMC gives better estimates
    finite_lp = filter(isfinite, log_posterior)
    log_ml = if length(finite_lp) > 0
        # Harmonic mean: 1/p(Y) ≈ E[1/L(θ)] under posterior
        # Use log_posterior which includes prior, so extract log_lik estimate
        # Simple approximation: max log posterior
        T(maximum(finite_lp))
    else
        T(-Inf)
    end

    BayesianDSGE{T}(
        draws, log_posterior, param_names, prior,
        log_ml, :rwmh, acceptance_rate,
        T[], T[],  # no ESS history or phi schedule for MH
        spec, sol, ss,
        data, observables, measurement_error, solver, solver_kwargs
    )
end

# =============================================================================
# Internal: build solution and state space at a given parameter vector
# =============================================================================

"""
    _build_solution_at_theta(spec, param_names, theta, observables,
                               measurement_error, solver, solver_kwargs)

Build the DSGE solution and state space at a given parameter vector.
Returns `(solution, state_space)`.
"""
function _build_solution_at_theta(spec::DSGESpec{T}, param_names::Vector{Symbol},
                                    theta::Vector{T},
                                    observables::Vector{Symbol},
                                    measurement_error,
                                    solver::Symbol,
                                    solver_kwargs::NamedTuple) where {T<:AbstractFloat}
    # Update spec parameters
    new_pv = copy(spec.param_values)
    for (i, pn) in enumerate(param_names)
        new_pv[pn] = theta[i]
    end

    new_spec = _respec(spec, new_pv)

    new_spec = compute_steady_state(new_spec)
    sol = solve(new_spec; method=solver, solver_kwargs...)

    # Build observation equation and state space
    Z, d, H = _build_observation_equation(new_spec, observables, measurement_error)

    # Dispatch on solution type for state space construction
    if sol isa PerturbationSolution
        ss = _build_nonlinear_state_space(sol, Z, d, H)
    elseif sol isa ProjectionSolution
        ss = _build_projection_state_space(sol, Z, d, H)
    else
        ss = _build_state_space(sol, Z, d, H)
    end

    return sol, ss
end

# =============================================================================
# Posterior analysis functions
# =============================================================================

"""
    posterior_summary(result::BayesianDSGE{T}; min_ess::Real=400) → Dict{Symbol, Dict{Symbol, T}}

Compute posterior summary statistics for each estimated parameter.

Returns a dictionary keyed by parameter name, each containing:
- `:mean` — posterior mean
- `:median` — posterior median (50th percentile)
- `:std` — posterior standard deviation
- `:ci_lower` — 2.5th percentile (lower bound of 95% credible interval)
- `:ci_upper` — 97.5th percentile (upper bound of 95% credible interval)

For RWMH chains (`method == :rwmh`), each entry additionally carries
- `:ess_bulk` — rank-normalized bulk effective sample size (Vehtari et al. 2021)
- `:low_ess` — `1.0` when `ess_bulk < min_ess` (default 400, the Vehtari et al.
  recommendation), else `0.0`

and a warning names the offending parameters — credible intervals from a chain
with low ESS are not reliable. Pass a smaller `min_ess` to relax the check.
"""
function posterior_summary(result::BayesianDSGE{T}; min_ess::Real=400) where {T<:AbstractFloat}
    n_params = length(result.param_names)
    summary = Dict{Symbol, Dict{Symbol, T}}()

    for i in 1:n_params
        pn = result.param_names[i]
        draws_i = result.theta_draws[:, i]
        sorted = sort(draws_i)
        n = length(sorted)

        # Quantile indices
        idx_025 = max(1, round(Int, 0.025 * n))
        idx_50 = max(1, round(Int, 0.50 * n))
        idx_975 = min(n, round(Int, 0.975 * n))

        summary[pn] = Dict{Symbol, T}(
            :mean => mean(draws_i),
            :median => sorted[idx_50],
            :std => std(draws_i),
            :ci_lower => sorted[idx_025],
            :ci_upper => sorted[idx_975]
        )
    end

    # ESS reliability check — MCMC chains only (SMC draws are weighted particle
    # systems where autocorrelation-based ESS does not apply)
    if result.method == :rwmh
        low = Symbol[]
        for i in 1:n_params
            pn = result.param_names[i]
            ess_i = _ess_bulk(result.theta_draws[:, i])
            summary[pn][:ess_bulk] = ess_i
            is_low = isfinite(ess_i) && ess_i < min_ess
            summary[pn][:low_ess] = is_low ? one(T) : zero(T)
            is_low && push!(low, pn)
        end
        if !isempty(low)
            @warn "posterior_summary: bulk ESS below $min_ess for " *
                  join(string.(low), ", ") *
                  " — credible intervals may be unreliable; run a longer chain, " *
                  "seed the proposal via proposal=:mode, or use method=:smc"
        end
    end

    return summary
end

"""
    marginal_likelihood(result::BayesianDSGE) → T

Return the log marginal likelihood (model evidence) from Bayesian estimation.

For SMC methods, this is the normalizing constant estimate from the
adaptive tempering schedule. For RWMH, this is an approximation.
"""
function marginal_likelihood(result::BayesianDSGE)
    return result.log_marginal_likelihood
end

"""
    bayes_factor(r1::BayesianDSGE, r2::BayesianDSGE) → T

Compute the log Bayes factor comparing model 1 to model 2:
`log BF₁₂ = log p(Y|M₁) - log p(Y|M₂)`.

A positive value favors model 1; negative favors model 2.
Following Kass & Raftery (1995), `2 * log BF > 6` is strong evidence.

# References
- Kass, R. E. & Raftery, A. E. (1995). Bayes Factors.
  *Journal of the American Statistical Association*, 90(430), 773-795.
"""
function bayes_factor(r1::BayesianDSGE, r2::BayesianDSGE)
    return r1.log_marginal_likelihood - r2.log_marginal_likelihood
end

"""
    bridge_sampling_ml(result::BayesianDSGE; proposal=:normal, df=5,
                       n_proposal=0, max_iter=1000, tol=1e-10,
                       rng=Random.default_rng()) → T

Bridge sampling estimate of the log marginal likelihood from stored posterior
draws (Meng & Wong 1996), using the optimal bridge function with the iterative
implementation of Gronau et al. (2017).

The estimator works in the **prior-transformed unconstrained space** (same
[`ParameterTransform`](@ref) as [`posterior_mode`](@ref)): the posterior draws
are mapped through the transform, a multivariate normal (or Student-t) proposal
`g` is fitted to the first half of the transformed draws, and the bridge
recursion

```math
r_{t+1} = \\frac{\\tfrac{1}{N_2}\\sum_j \\ell_{2,j} / (s_1 \\ell_{2,j} + s_2 r_t)}
               {\\tfrac{1}{N_1}\\sum_i 1 / (s_1 \\ell_{1,i} + s_2 r_t)},
\\qquad \\ell = \\frac{p(Y|\\theta)\\,\\pi(\\theta)\\,|J(\\phi)|}{g(\\phi)}
```

is iterated to convergence on the second half (`ℓ₁`) and fresh draws from `g`
(`ℓ₂`); `log p̂(Y) = log r^*`. All averages run in shifted log space to avoid
overflow. Bridge sampling is markedly more stable than the modified harmonic
mean, whose importance weights can have infinite variance.

Returns `NaN` with a warning when the chain is too short, the result carries no
estimation data, too few proposal draws yield a finite posterior kernel, or the
recursion fails to converge — never a silently wrong number. The additive
constant convention matches the SMC tempering-path estimate, so the two are
comparable via [`bayes_factor`](@ref).

# Keywords
- `proposal::Symbol=:normal` — proposal family: `:normal` or `:t` (Student-t)
- `df::Real=5` — degrees of freedom for the `:t` proposal
- `n_proposal::Int=0` — number of proposal draws (`0` → same as bridge half)
- `max_iter::Int=1000` — maximum bridge iterations
- `tol::Real=1e-10` — relative convergence tolerance on `r`
- `rng::AbstractRNG` — random number generator for proposal draws

# References
- Meng, X.-L. & Wong, W. H. (1996). Simulating Ratios of Normalizing Constants
  via a Simple Identity. *Statistica Sinica*, 6, 831-860.
- Gronau, Q. F. et al. (2017). A Tutorial on Bridge Sampling.
  *Journal of Mathematical Psychology*, 81, 80-97.
"""
function bridge_sampling_ml(result::BayesianDSGE{T};
                            proposal::Symbol=:normal,
                            df::Real=5,
                            n_proposal::Int=0,
                            max_iter::Int=1000,
                            tol::Real=1e-10,
                            rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    proposal in (:normal, :t) ||
        throw(ArgumentError("proposal must be :normal or :t, got :$proposal"))
    N, d = size(result.theta_draws)

    if isempty(result.data)
        @warn "bridge_sampling_ml: result carries no estimation data " *
              "(hand-constructed result?); returning NaN"
        return T(NaN)
    end
    if N < max(20, 4 * d)
        @warn "bridge_sampling_ml: chain too short (N=$N draws for d=$d " *
              "parameters); returning NaN"
        return T(NaN)
    end

    prior = result.priors
    pt = ParameterTransform(prior.lower, prior.upper)

    # Transform all draws to the unconstrained space
    phi = zeros(T, N, d)
    for i in 1:N
        phi[i, :] = to_unconstrained(pt, result.theta_draws[i, :])
    end
    finite_rows = [all(isfinite, @view phi[i, :]) for i in 1:N]
    if !all(finite_rows)
        keep = findall(finite_rows)
        length(keep) < max(20, 4 * d) &&
            (@warn "bridge_sampling_ml: too few interior draws after transform"; return T(NaN))
        phi = phi[keep, :]
        N = length(keep)
    end

    # First half fits the proposal, second half enters the bridge (Gronau et al.)
    n_fit = N ÷ 2
    idx_bridge = (n_fit+1):N
    n1 = length(idx_bridge)
    n2 = n_proposal > 0 ? n_proposal : n1

    mu_g = vec(mean(@view(phi[1:n_fit, :]); dims=1))
    Sigma_g = cov(@view(phi[1:n_fit, :]))
    Sigma_g = Matrix{T}((Sigma_g + Sigma_g') / 2)
    L = _suppress_warnings() do
        safe_cholesky(Sigma_g)
    end
    Sigma_pd = Matrix{T}(L * L')
    g = proposal == :normal ? MvNormal(mu_g, Sigma_pd) :
                              MvTDist(T(df), mu_g, Sigma_pd)

    # log |J(φ)| — Jacobian of the unconstrained → constrained map (shared,
    # numerically stable implementation)
    log_jac(phiv) = log_jacobian(pt, phiv)

    # ℓ₁: posterior-kernel over proposal at the bridge half (kernel values stored)
    l1 = zeros(T, n1)
    for (k, i) in enumerate(idx_bridge)
        phiv = phi[i, :]
        l1[k] = result.log_posterior[i] + log_jac(phiv) - T(logpdf(g, phiv))
    end

    # ℓ₂: posterior-kernel over proposal at fresh g draws — evaluated through the
    # same likelihood closure the samplers used
    ll_fn = _build_likelihood_fn(result.spec, result.param_names, result.data,
                                 result.observables, result.measurement_error,
                                 result.solver, result.solver_kwargs)
    phi_g = rand(rng, g, n2)                       # d × n2
    l2 = fill(T(-Inf), n2)
    for j in 1:n2
        phiv = Vector{T}(phi_g[:, j])
        theta_j = to_constrained(pt, phiv)
        lp = _log_prior(theta_j, prior)
        isfinite(lp) || continue
        ll = ll_fn(theta_j)
        isfinite(ll) || continue
        l2[j] = ll + lp + log_jac(phiv) - T(logpdf(g, phiv))
    end
    n2_finite = count(isfinite, l2)
    if n2_finite < max(10, n2 ÷ 20)
        @warn "bridge_sampling_ml: only $n2_finite of $n2 proposal draws yield a " *
              "finite posterior kernel; proposal too diffuse — returning NaN"
        return T(NaN)
    end

    # Bridge recursion in shifted log space
    lstar = median(filter(isfinite, l1))
    e1 = [isfinite(v) ? exp(v - lstar) : zero(T) for v in l1]
    e2 = [isfinite(v) ? exp(v - lstar) : zero(T) for v in l2]
    s1 = T(n1) / (n1 + n2)
    s2 = T(n2) / (n1 + n2)

    r = one(T)
    converged = false
    for _ in 1:max_iter
        num = mean(e2 ./ (s1 .* e2 .+ s2 * r))
        den = mean(one(T) ./ (s1 .* e1 .+ s2 * r))
        r_new = num / den
        if !isfinite(r_new) || r_new <= 0
            break
        end
        if abs(r_new - r) <= tol * abs(r_new)
            r = r_new
            converged = true
            break
        end
        r = r_new
    end

    if !converged || !isfinite(r) || r <= 0
        @warn "bridge_sampling_ml: bridge recursion did not converge in " *
              "$max_iter iterations; returning NaN"
        return T(NaN)
    end

    return log(r) + lstar
end

"""
    prior_posterior_table(result::BayesianDSGE{T}) → Vector{NamedTuple}

Generate rows for a prior-posterior comparison table.

Each row contains:
- `param` — parameter name
- `prior_dist` — prior distribution type
- `prior_mean` — prior mean
- `prior_std` — prior standard deviation
- `post_mean` — posterior mean
- `post_std` — posterior standard deviation
- `ci_lower` — posterior 2.5th percentile
- `ci_upper` — posterior 97.5th percentile
- `low_ess` — `true` when the parameter's bulk ESS falls below `min_ess`
  (RWMH chains only; always `false` for SMC results)
"""
function prior_posterior_table(result::BayesianDSGE{T}; min_ess::Real=400) where {T<:AbstractFloat}
    ps = posterior_summary(result; min_ess=min_ess)
    n_params = length(result.param_names)

    rows = NamedTuple[]
    for i in 1:n_params
        pn = result.param_names[i]
        d = result.priors.distributions[i]
        row = (
            param = pn,
            prior_dist = string(typeof(d).name.name),
            prior_mean = T(mean(d)),
            prior_std = T(std(d)),
            post_mean = ps[pn][:mean],
            post_std = ps[pn][:std],
            ci_lower = ps[pn][:ci_lower],
            ci_upper = ps[pn][:ci_upper],
            low_ess = get(ps[pn], :low_ess, zero(T)) == one(T)
        )
        push!(rows, row)
    end

    return rows
end

"""
    posterior_predictive(result::BayesianDSGE{T}, n_sim::Int;
                          T_periods::Int=100,
                          rng::AbstractRNG=Random.default_rng()) → Array{T,3}

Simulate from the posterior predictive distribution.

For `n_sim` randomly selected posterior draws, solves the model at those
parameter values and simulates forward `T_periods` periods.

# Arguments
- `result` — Bayesian estimation result
- `n_sim` — number of simulated paths

# Keywords
- `T_periods::Int=100` — number of periods per simulation
- `rng::AbstractRNG` — random number generator

# Returns
`n_sim × T_periods × n_vars` array of simulated paths.
"""
function posterior_predictive(result::BayesianDSGE{T}, n_sim::Int;
                               T_periods::Int=100,
                               rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n_draws_total = size(result.theta_draws, 1)
    n_params = length(result.param_names)
    spec = result.spec

    # Determine number of variables from spec
    n_vars = spec.augmented ? length(spec.original_endog) : spec.n_endog

    paths = zeros(T, n_sim, T_periods, n_vars)

    for s in 1:n_sim
        # Randomly select a posterior draw
        draw_idx = rand(rng, 1:n_draws_total)
        theta = result.theta_draws[draw_idx, :]

        # Update spec parameters
        new_pv = copy(spec.param_values)
        for (i, pn) in enumerate(result.param_names)
            new_pv[pn] = theta[i]
        end

        new_spec = _respec(spec, new_pv)

        try
            new_spec = compute_steady_state(new_spec)
            sol = solve(new_spec; method=:gensys)
            if is_determined(sol)
                sim = simulate(sol, T_periods; rng=rng)
                # simulate returns T_periods × n_vars
                paths[s, :, :] = sim
            end
        catch
            # Leave as zeros for failed simulations
            continue
        end
    end

    return paths
end

# =============================================================================
# Prior predictive & posterior predictive checks (Gelman, Meng & Stern 1996)
# =============================================================================

"""
    PriorPredictiveResult{T}

Prior predictive distribution of summary statistics.

Fields:
- `stat_names::Vector{String}` — statistic labels
- `stats::Matrix{T}` — `n_effective × n_stats` draw-level statistics
- `n_draws::Int` — requested prior draws
- `n_effective::Int` — draws that solved and simulated successfully
- `T_periods::Int` — simulated sample length per draw
"""
struct PriorPredictiveResult{T<:AbstractFloat}
    stat_names::Vector{String}
    stats::Matrix{T}
    n_draws::Int
    n_effective::Int
    T_periods::Int
end

"""
    PosteriorPredictiveCheck{T}

Posterior predictive check result.

Fields:
- `stat_names::Vector{String}` — statistic labels
- `observed::Vector{T}` — statistics computed on the observed data
- `replicated::Matrix{T}` — `n_effective × n_stats` statistics on replicated datasets
- `p_values::Vector{T}` — posterior predictive p-values `P(T(y_rep) ≥ T(y_obs))`
- `n_draws::Int` — requested posterior draws
- `n_effective::Int` — draws that solved and simulated successfully
"""
struct PosteriorPredictiveCheck{T<:AbstractFloat}
    stat_names::Vector{String}
    observed::Vector{T}
    replicated::Matrix{T}
    p_values::Vector{T}
    n_draws::Int
    n_effective::Int
end

"""
    _make_default_stats(observables) → Function

Default predictive summary statistics: mean, variance, and first-order
autocorrelation of each observable, plus pairwise cross-correlations.
"""
function _make_default_stats(observables::Vector{Symbol})
    return function (Y::AbstractMatrix)
        names = String[]
        vals = Float64[]
        n = size(Y, 2)
        for (j, o) in enumerate(observables)
            yj = @view Y[:, j]
            push!(names, "mean_$o");
            push!(vals, mean(yj))
            push!(names, "var_$o")
            push!(vals, var(yj))
            v = var(yj)
            r1 = v > 0 ? cor(@view(Y[2:end, j]), @view(Y[1:end-1, j])) : 0.0
            push!(names, "ar1_$o")
            push!(vals, r1)
        end
        for i in 1:n, j in (i+1):n
            push!(names, "corr_$(observables[i])_$(observables[j])")
            push!(vals, cor(@view(Y[:, i]), @view(Y[:, j])))
        end
        return (names, vals)
    end
end

"""
    _stat_pairs(out) → (names::Vector{String}, values::Vector{Float64})

Normalize a stats-function return value: accepts a `(names, values)` tuple or a
`NamedTuple` of scalars.
"""
_stat_pairs(out::Tuple{<:AbstractVector,<:AbstractVector}) =
    (String.(out[1]), Float64.(out[2]))
_stat_pairs(out::NamedTuple) = (String.(collect(keys(out))), Float64.(collect(values(out))))

"""
    _predictive_stats_loop(spec, thetas, param_names, observables, T_periods,
                           stats_fn, solver, solver_kwargs, rng)
        → (stat_names, stats_matrix, n_effective)

Shared loop for predictive simulation: for each parameter draw (row of
`thetas`), solve the model, simulate `T_periods`, restrict to the observables,
and compute the summary statistics. Failed solves/simulations are **dropped and
counted**, never zero-filled.
"""
function _predictive_stats_loop(spec::DSGESpec{T}, thetas::AbstractMatrix{T},
                                param_names::Vector{Symbol},
                                observables::Vector{Symbol}, T_periods::Int,
                                stats_fn, solver::Symbol,
                                solver_kwargs::NamedTuple,
                                rng::AbstractRNG) where {T<:AbstractFloat}
    var_syms = spec.augmented ? spec.original_endog : spec.endog
    obs_idx = [findfirst(==(o), var_syms) for o in observables]
    any(isnothing, obs_idx) &&
        throw(ArgumentError("observables must be endogenous variables of the spec"))
    obs_idx = Vector{Int}(obs_idx)

    n = size(thetas, 1)
    stat_names = String[]
    rows = Vector{Vector{Float64}}()
    for s in 1:n
        theta = Vector{T}(thetas[s, :])
        try
            new_pv = copy(spec.param_values)
            for (i, pn) in enumerate(param_names)
                new_pv[pn] = theta[i]
            end
            new_spec = _respec(spec, new_pv)
            new_spec = compute_steady_state(new_spec)
            sol = solve(new_spec; method=solver, solver_kwargs...)
            if sol isa DSGESolution && !is_determined(sol)
                continue
            end
            sim = simulate(sol, T_periods; rng=rng)      # T_periods × n_vars
            Y = sim[:, obs_idx]
            all(isfinite, Y) || continue
            names, vals = _stat_pairs(stats_fn(Y))
            if isempty(stat_names)
                append!(stat_names, names)
            end
            length(vals) == length(stat_names) || continue
            push!(rows, vals)
        catch
            continue
        end
    end

    n_eff = length(rows)
    stats_mat = n_eff == 0 ? zeros(T, 0, length(stat_names)) :
                Matrix{T}(reduce(vcat, (permutedims(r) for r in rows)))
    return stat_names, stats_mat, n_eff
end

"""
    prior_predictive(spec, priors; n_draws=500, T_periods=200,
                     observables=Symbol[], stats=nothing, solver=:gensys,
                     solver_kwargs=NamedTuple(), rng=Random.default_rng())
        → PriorPredictiveResult

Prior predictive analysis (Geweke 2005): draw parameters from the prior, solve
the model, simulate `T_periods` of observables, and record summary statistics
per draw — the implied *prior predictive distribution* of the statistics. Use
this to check whether a prior implies economically absurd data **before**
estimating.

Draws where the model fails to solve are dropped and counted (`n_effective`);
a warning reports the skipped fraction when it exceeds 10%.

# Keywords
- `n_draws::Int=500` — prior draws
- `T_periods::Int=200` — simulated periods per draw
- `observables::Vector{Symbol}=Symbol[]` — observables (default: all endogenous)
- `stats=nothing` — statistic function `Y::Matrix → (names, values)` or
  `NamedTuple`; default: mean/variance/AR(1) per observable + cross-correlations
- `solver`, `solver_kwargs`, `rng` — as in [`estimate_dsge_bayes`](@ref)
"""
function prior_predictive(spec::DSGESpec{T},
                          priors::Dict{Symbol,<:Distribution};
                          n_draws::Int=500,
                          T_periods::Int=200,
                          observables::Vector{Symbol}=Symbol[],
                          stats=nothing,
                          solver::Symbol=:gensys,
                          solver_kwargs::NamedTuple=NamedTuple(),
                          rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    prior = _build_bayes_prior(priors)
    param_names = prior.param_names
    d = length(param_names)
    if isempty(observables)
        observables = spec.augmented ? copy(spec.original_endog) : copy(spec.endog)
    end
    stats_fn = stats === nothing ? _make_default_stats(observables) : stats

    thetas = zeros(T, n_draws, d)
    for s in 1:n_draws, i in 1:d
        for attempt in 1:100
            draw = T(rand(rng, prior.distributions[i]))
            if prior.lower[i] <= draw <= prior.upper[i]
                thetas[s, i] = draw
                break
            end
            if attempt == 100
                lo = isfinite(prior.lower[i]) ? prior.lower[i] : T(0)
                hi = isfinite(prior.upper[i]) ? prior.upper[i] : T(1)
                thetas[s, i] = (lo + hi) / 2
            end
        end
    end

    stat_names, stats_mat, n_eff = _predictive_stats_loop(
        spec, thetas, param_names, observables, T_periods, stats_fn,
        solver, solver_kwargs, rng)

    skipped = n_draws - n_eff
    if skipped > n_draws ÷ 10
        @warn "prior_predictive: $skipped of $n_draws prior draws failed to solve " *
              "(indeterminate/explosive) and were dropped — the prior puts substantial " *
              "mass outside the determinacy region"
    end
    n_eff == 0 && @warn "prior_predictive: no prior draw produced a valid simulation"

    return PriorPredictiveResult{T}(stat_names, stats_mat, n_draws, n_eff, T_periods)
end

"""
    posterior_predictive_check(result::BayesianDSGE; data=nothing, n_draws=200,
                               stats=nothing, rng=Random.default_rng())
        → PosteriorPredictiveCheck

Posterior predictive checks (Gelman, Meng & Stern 1996): draw parameter vectors
from the posterior, simulate replicated datasets of the observed sample length,
compute summary statistics on each replication and on the observed data, and
report the **posterior predictive p-value** per statistic,

```math
p_j = \\Pr\\big(T_j(y^{\\mathrm{rep}}) \\geq T_j(y^{\\mathrm{obs}})\\big).
```

Interior p-values (say 0.05–0.95) indicate the model reproduces that feature of
the data; extreme p-values flag misspecification along that dimension. Failed
draws are dropped and counted (`n_effective`), never zero-filled.

# Keywords
- `data=nothing` — observed data; default: the sample stored on the result
- `n_draws::Int=200` — posterior draws to replicate
- `stats=nothing` — statistic function (same contract as [`prior_predictive`](@ref))
- `rng::AbstractRNG` — random number generator
"""
function posterior_predictive_check(result::BayesianDSGE{T};
                                    data::Union{Nothing,AbstractMatrix}=nothing,
                                    n_draws::Int=200,
                                    stats=nothing,
                                    rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    observables = isempty(result.observables) ?
        (result.spec.augmented ? copy(result.spec.original_endog) : copy(result.spec.endog)) :
        result.observables
    if data === nothing
        isempty(result.data) &&
            throw(ArgumentError("result carries no stored data; pass data=... explicitly"))
        data = result.data
    end
    # Observed data → T_obs × n_obs orientation for the stats function
    data_no = _orient_bayes_data(data, length(observables), T)
    Y_obs = Matrix{T}(data_no')
    T_obs = size(Y_obs, 1)

    stats_fn = stats === nothing ? _make_default_stats(observables) : stats
    obs_names, obs_vals = _stat_pairs(stats_fn(Y_obs))

    n_total = size(result.theta_draws, 1)
    n_use = min(n_draws, n_total)
    idx = n_use == n_total ? collect(1:n_total) : randperm(rng, n_total)[1:n_use]
    thetas = result.theta_draws[idx, :]

    stat_names, rep_mat, n_eff = _predictive_stats_loop(
        result.spec, thetas, result.param_names, observables, T_obs, stats_fn,
        result.solver, result.solver_kwargs, rng)

    skipped = n_use - n_eff
    skipped > n_use ÷ 10 &&
        @warn "posterior_predictive_check: $skipped of $n_use posterior draws failed " *
              "to solve and were dropped"
    n_eff == 0 &&
        throw(ArgumentError("no posterior draw produced a valid replication"))

    n_stats = length(obs_names)
    pvals = zeros(T, n_stats)
    for j in 1:n_stats
        pvals[j] = mean(rep_mat[:, j] .>= obs_vals[j])
    end

    return PosteriorPredictiveCheck{T}(obs_names, Vector{T}(obs_vals), rep_mat,
                                       pvals, n_use, n_eff)
end

function Base.show(io::IO, ppr::PriorPredictiveResult{T}) where {T}
    header = Any[
        "Prior draws"      ppr.n_draws;
        "Effective draws"  ppr.n_effective;
        "Periods"          ppr.T_periods;
        "Statistics"       length(ppr.stat_names);
    ]
    _pretty_table(io, header;
        title="Prior Predictive Analysis",
        column_labels=["", ""],
        alignment=[:l, :r])
    ppr.n_effective == 0 && return
    n_stats = length(ppr.stat_names)
    data = Matrix{Any}(undef, n_stats, 6)
    for j in 1:n_stats
        col = ppr.stats[:, j]
        q = quantile(col, [0.05, 0.5, 0.95])
        data[j, 1] = ppr.stat_names[j]
        data[j, 2] = round(mean(col); sigdigits=4)
        data[j, 3] = round(std(col); sigdigits=4)
        data[j, 4] = round(q[1]; sigdigits=4)
        data[j, 5] = round(q[2]; sigdigits=4)
        data[j, 6] = round(q[3]; sigdigits=4)
    end
    _pretty_table(io, data;
        title="Prior Predictive Distribution of Statistics",
        column_labels=["Statistic", "Mean", "Std", "5%", "50%", "95%"],
        alignment=[:l, :r, :r, :r, :r, :r])
end

function Base.show(io::IO, ppc::PosteriorPredictiveCheck{T}) where {T}
    header = Any[
        "Posterior draws"  ppc.n_draws;
        "Effective draws"  ppc.n_effective;
        "Statistics"       length(ppc.stat_names);
    ]
    _pretty_table(io, header;
        title="Posterior Predictive Check",
        column_labels=["", ""],
        alignment=[:l, :r])
    n_stats = length(ppc.stat_names)
    data = Matrix{Any}(undef, n_stats, 6)
    for j in 1:n_stats
        col = ppc.replicated[:, j]
        q = quantile(col, [0.05, 0.95])
        p = ppc.p_values[j]
        data[j, 1] = ppc.stat_names[j]
        data[j, 2] = round(ppc.observed[j]; sigdigits=4)
        data[j, 3] = round(mean(col); sigdigits=4)
        data[j, 4] = round(q[1]; sigdigits=4)
        data[j, 5] = round(q[2]; sigdigits=4)
        data[j, 6] = string(round(p; digits=3), (p < 0.05 || p > 0.95) ? " *" : "")
    end
    _pretty_table(io, data;
        title="Observed vs Replicated (p = P(rep ≥ obs); * = extreme)",
        column_labels=["Statistic", "Observed", "Rep. mean", "Rep. 5%", "Rep. 95%", "p-value"],
        alignment=[:l, :r, :r, :r, :r, :r])
end

# =============================================================================
# Bayesian DSGE IRF — posterior credible bands
# =============================================================================

"""
    irf(result::BayesianDSGE{T}, horizon::Int;
        n_draws::Int=200,
        quantiles::Vector{<:Real}=[0.05, 0.16, 0.84, 0.95],
        solver::Symbol=:gensys,
        solver_kwargs::NamedTuple=NamedTuple(),
        rng::AbstractRNG=Random.default_rng()) → BayesianImpulseResponse{T}

Compute Bayesian impulse responses with posterior credible bands.

For each of `n_draws` randomly selected posterior parameter draws, re-solves
the DSGE model and computes analytical IRFs. Reports pointwise posterior median
and quantile bands.

Default quantiles give dual 68% (16th–84th) and 90% (5th–95th) credible bands.

# References
- Herbst, E. & Schorfheide, F. (2015). *Bayesian Estimation of DSGE Models*.
  Princeton University Press.
"""
function irf(result::BayesianDSGE{T}, horizon::Int;
             n_draws::Int=200,
             quantiles::Vector{<:Real}=T[0.05, 0.16, 0.84, 0.95],
             solver::Symbol=:gensys,
             solver_kwargs::NamedTuple=NamedTuple(),
             rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    spec = result.spec
    n_draws_total = size(result.theta_draws, 1)
    n_sim = min(n_draws, n_draws_total)

    # Determine output dimensions from a reference IRF at the posterior mode solution
    ref_irf = irf(result.solution, horizon)
    n_vars = length(ref_irf.variables)
    n_shocks = length(ref_irf.shocks)
    var_names = ref_irf.variables
    shock_names = ref_irf.shocks

    # Collect IRF arrays from posterior draws
    all_irfs = Vector{Array{T,3}}()
    draw_indices = randperm(rng, n_draws_total)[1:n_sim]

    for idx in draw_indices
        theta = Vector{T}(result.theta_draws[idx, :])
        try
            _suppress_warnings() do
                sol, _ = _build_solution_at_theta(spec, result.param_names, theta,
                                                   spec.endog, nothing, solver, solver_kwargs)
                if sol isa DSGESolution && (!is_determined(sol))
                    return  # skip indeterminate
                end
                irf_i = irf(sol, horizon)
                push!(all_irfs, irf_i.values)
            end
        catch
            continue
        end
    end

    n_valid = length(all_irfs)
    n_valid == 0 && error("All posterior draws failed to produce valid IRFs")

    # Stack into (n_valid x H x n_vars x n_shocks) array
    stacked = zeros(T, n_valid, horizon, n_vars, n_shocks)
    for (s, arr) in enumerate(all_irfs)
        stacked[s, :, :, :] = arr
    end

    # Compute quantiles
    q_vec = T.(quantiles)
    irf_q, irf_m = compute_posterior_quantiles(stacked, q_vec)

    BayesianImpulseResponse{T}(irf_q, irf_m, horizon, var_names, shock_names, q_vec, stacked)
end

# =============================================================================
# Bayesian DSGE FEVD — posterior credible bands
# =============================================================================

"""
    fevd(result::BayesianDSGE{T}, horizon::Int;
         n_draws::Int=200,
         quantiles::Vector{<:Real}=[0.05, 0.16, 0.84, 0.95],
         solver::Symbol=:gensys,
         solver_kwargs::NamedTuple=NamedTuple(),
         rng::AbstractRNG=Random.default_rng()) → BayesianFEVD{T}

Compute Bayesian forecast error variance decomposition with posterior credible bands.

For each of `n_draws` randomly selected posterior parameter draws, re-solves
the DSGE model and computes FEVD proportions. Reports pointwise posterior median
and quantile bands.

Default quantiles give dual 68% (16th–84th) and 90% (5th–95th) credible bands.
"""
function fevd(result::BayesianDSGE{T}, horizon::Int;
              n_draws::Int=200,
              quantiles::Vector{<:Real}=T[0.05, 0.16, 0.84, 0.95],
              solver::Symbol=:gensys,
              solver_kwargs::NamedTuple=NamedTuple(),
              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    spec = result.spec
    n_draws_total = size(result.theta_draws, 1)
    n_sim = min(n_draws, n_draws_total)

    # Reference dimensions from posterior mode
    ref_fevd = fevd(result.solution, horizon)
    n_vars = length(ref_fevd.variables)
    n_shocks = length(ref_fevd.shocks)
    var_names = ref_fevd.variables
    shock_names = ref_fevd.shocks

    # Collect FEVD proportions from posterior draws
    all_fevds = Vector{Array{T,3}}()
    draw_indices = randperm(rng, n_draws_total)[1:n_sim]

    for idx in draw_indices
        theta = Vector{T}(result.theta_draws[idx, :])
        try
            _suppress_warnings() do
                sol, _ = _build_solution_at_theta(spec, result.param_names, theta,
                                                   spec.endog, nothing, solver, solver_kwargs)
                if sol isa DSGESolution && (!is_determined(sol))
                    return
                end
                fevd_i = fevd(sol, horizon)
                push!(all_fevds, fevd_i.proportions)
            end
        catch
            continue
        end
    end

    n_valid = length(all_fevds)
    n_valid == 0 && error("All posterior draws failed to produce valid FEVDs")

    # Stack: FEVD proportions are (n_vars x n_shocks x horizon)
    # Rearrange to (n_valid x horizon x n_vars x n_shocks) for quantile computation
    stacked = zeros(T, n_valid, horizon, n_vars, n_shocks)
    for (s, arr) in enumerate(all_fevds)
        for h in 1:horizon, v in 1:n_vars, sh in 1:n_shocks
            stacked[s, h, v, sh] = arr[v, sh, h]
        end
    end

    q_vec = T.(quantiles)
    fevd_q, fevd_m = compute_posterior_quantiles(stacked, q_vec)

    BayesianFEVD{T}(fevd_q, fevd_m, horizon, var_names, shock_names, q_vec)
end

# =============================================================================
# Bayesian DSGE Simulation — posterior predictive bands
# =============================================================================

"""
    simulate(result::BayesianDSGE{T}, T_periods::Int;
             n_draws::Int=200,
             quantiles::Vector{<:Real}=[0.05, 0.16, 0.84, 0.95],
             solver::Symbol=:gensys,
             solver_kwargs::NamedTuple=NamedTuple(),
             rng::AbstractRNG=Random.default_rng()) → BayesianDSGESimulation{T}

Simulate from the posterior predictive distribution with credible bands.

For each of `n_draws` randomly selected posterior parameter draws, re-solves
the DSGE model and simulates `T_periods` forward. Reports pointwise posterior
median and quantile bands.

Default quantiles give dual 68% (16th–84th) and 90% (5th–95th) credible bands.
"""
function simulate(result::BayesianDSGE{T}, T_periods::Int;
                   n_draws::Int=200,
                   quantiles::Vector{<:Real}=T[0.05, 0.16, 0.84, 0.95],
                   solver::Symbol=:gensys,
                   solver_kwargs::NamedTuple=NamedTuple(),
                   rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    spec = result.spec
    n_draws_total = size(result.theta_draws, 1)
    n_sim = min(n_draws, n_draws_total)

    # Determine n_vars from spec
    n_vars = spec.augmented ? length(spec.original_endog) : spec.n_endog
    var_names = spec.augmented ? [string(s) for s in spec.original_endog] :
                                 [string(s) for s in spec.endog]

    # Collect simulation paths
    all_paths_list = Vector{Matrix{T}}()
    draw_indices = randperm(rng, n_draws_total)[1:n_sim]

    for idx in draw_indices
        theta = Vector{T}(result.theta_draws[idx, :])
        try
            _suppress_warnings() do
                sol, _ = _build_solution_at_theta(spec, result.param_names, theta,
                                                   spec.endog, nothing, solver, solver_kwargs)
                if sol isa DSGESolution && (!is_determined(sol))
                    return
                end
                sim = simulate(sol, T_periods; rng=rng)
                push!(all_paths_list, sim)
            end
        catch
            continue
        end
    end

    n_valid = length(all_paths_list)
    n_valid == 0 && error("All posterior draws failed to produce valid simulations")

    # Stack into (n_valid x T_periods x n_vars)
    stacked = zeros(T, n_valid, T_periods, n_vars)
    for (s, path) in enumerate(all_paths_list)
        stacked[s, :, :] = path
    end

    q_vec = T.(quantiles)
    sim_q, sim_m = compute_posterior_quantiles(stacked, q_vec)

    BayesianDSGESimulation{T}(sim_q, sim_m, T_periods, var_names, q_vec, stacked)
end

# =============================================================================
# StatsAPI
# =============================================================================

StatsAPI.coef(m::BayesianDSGE) = vec(mean(m.theta_draws, dims=1))
StatsAPI.islinear(::BayesianDSGE) = false

# =============================================================================
# Show
# =============================================================================

function Base.show(io::IO, result::BayesianDSGE{T}) where {T}
    n_draws, n_params = size(result.theta_draws)
    ps = posterior_summary(result)

    # Summary table
    spec_data = Any[
        "Method"                string(result.method);
        "Parameters"            n_params;
        "Posterior draws"       n_draws;
        "Log marginal lik."    round(result.log_marginal_likelihood; digits=4);
        "Acceptance rate"      round(result.acceptance_rate; digits=4);
        "Tempering stages"     length(result.phi_schedule);
    ]
    _pretty_table(io, spec_data;
        title="Bayesian DSGE Estimation",
        column_labels=["", ""],
        alignment=[:l, :r])

    # Posterior summary table — mark low-ESS parameters (RWMH only) with †
    pnames = [string(s) * (get(ps[s], :low_ess, zero(T)) == one(T) ? " †" : "")
              for s in result.param_names]
    means = [ps[n][:mean] for n in result.param_names]
    stds = [ps[n][:std] for n in result.param_names]
    medians = [ps[n][:median] for n in result.param_names]
    ci_lo = [ps[n][:ci_lower] for n in result.param_names]
    ci_hi = [ps[n][:ci_upper] for n in result.param_names]

    post_data = hcat(pnames, round.(means; digits=4), round.(stds; digits=4),
                     round.(medians; digits=4), round.(ci_lo; digits=4),
                     round.(ci_hi; digits=4))
    _pretty_table(io, post_data;
        title="Posterior Summary",
        column_labels=["Parameter", "Mean", "Std", "Median", "2.5%", "97.5%"],
        alignment=[:l, :r, :r, :r, :r, :r])

    if any(get(ps[s], :low_ess, zero(T)) == one(T) for s in result.param_names)
        println(io, "† bulk ESS below threshold — credible intervals may be unreliable.")
    end

    # Prior vs posterior table (min_ess=0: the ESS warning already fired above)
    pt = prior_posterior_table(result; min_ess=0)
    if !isempty(pt)
        n_rows = length(pt)
        pp_data = Matrix{Any}(undef, n_rows, 8)
        for (ri, row) in enumerate(pt)
            pp_data[ri, 1] = string(row.param)
            pp_data[ri, 2] = row.prior_dist
            pp_data[ri, 3] = round(row.prior_mean; digits=4)
            pp_data[ri, 4] = round(row.prior_std; digits=4)
            pp_data[ri, 5] = round(row.post_mean; digits=4)
            pp_data[ri, 6] = round(row.post_std; digits=4)
            pp_data[ri, 7] = round(row.ci_lower; digits=4)
            pp_data[ri, 8] = round(row.ci_upper; digits=4)
        end
        _pretty_table(io, pp_data;
            title="Prior vs Posterior",
            column_labels=["Parameter", "Prior", "Prior Mean", "Prior Std",
                          "Post Mean", "Post Std", "2.5%", "97.5%"],
            alignment=[:l, :l, :r, :r, :r, :r, :r, :r])
    end
end

function Base.show(io::IO, result::BayesianDSGESimulation{T}) where {T}
    println(io, "BayesianDSGESimulation{$T}")
    println(io, "  Periods:    $(result.T_periods)")
    println(io, "  Variables:  $(length(result.variables))")
    println(io, "  Draws:      $(size(result.all_paths, 1))")
    println(io, "  Quantiles:  $(result.quantile_levels)")
end
