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
        end
        draws, log_posterior, acceptance_rate = _mh_sample(
            spec, data_mat, param_names, prior, theta_init;
            n_draws=n_draws, burnin=burnin,
            observables=observables,
            measurement_error=measurement_error,
            solver=solver, solver_kwargs=solver_kwargs,
            init_proposal_cov=init_proposal_cov, rng=rng)
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

    # log |J(φ)| — Jacobian of the unconstrained → constrained map
    log_jac(phiv) = sum(log(abs(transform_jacobian(pt, phiv)[k, k])) for k in 1:d)

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
