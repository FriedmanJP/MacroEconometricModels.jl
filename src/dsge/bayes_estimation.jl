# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Public API for Bayesian DSGE estimation and posterior analysis functions.

Provides:
- `estimate_dsge_bayes` — main estimation entry point (SMC, SMC², RWMH)
- `posterior_summary` — posterior mean, median, std, credible intervals
- `marginal_likelihood` — log marginal likelihood accessor
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

# =============================================================================
# Helper: resolve theta0 (positional / Dict / NamedTuple) onto sorted params
# =============================================================================

"""
    _resolve_theta0(theta0, param_names, ::Type{T}) -> Vector{T}

Resolve the initial parameter vector against the (alphabetically sorted) prior
parameter names (E-12 / H-12 / #136).

- `Dict{Symbol}` / `NamedTuple`: build the internal vector by looking up each name
  in `param_names`, so the input is **order-independent**. Errors if any prior
  parameter is missing from `theta0`, or if `theta0` names a parameter not in the
  priors.
- `AbstractVector`: used positionally and **must be in sorted prior-key order**;
  its length must equal `length(param_names)`, else an `ArgumentError` naming the
  expected parameters is thrown (previously a wrong length failed opaquely much
  later, and a mis-ordered vector was silently permuted onto the wrong parameters).
"""
function _resolve_theta0(theta0, param_names::Vector{Symbol}, ::Type{T}) where {T}
    if theta0 isa AbstractDict || theta0 isa NamedTuple
        for k in keys(theta0)
            k in param_names || throw(ArgumentError(
                "theta0 names unknown parameter :$k; the estimated parameters " *
                "(sorted) are $(param_names)."))
        end
        out = Vector{T}(undef, length(param_names))
        for (i, pn) in enumerate(param_names)
            haskey(theta0, pn) || throw(ArgumentError(
                "theta0 is missing parameter :$pn; provide all of $(param_names)."))
            out[i] = T(theta0[pn])
        end
        return out
    else
        v = collect(theta0)
        length(v) == length(param_names) || throw(ArgumentError(
            "theta0 has length $(length(v)) but the model has $(length(param_names)) " *
            "estimated parameters. Pass a positional vector in sorted prior-key order " *
            "$(param_names), or a Dict/NamedTuple (order-independent)."))
        return T.(v)
    end
end

"""
    _orient_data(data, n_obs, ::Type{T}) -> Matrix{T}

Resolve data orientation by matching a dimension to `n_obs` (the number of
observables), NOT by comparing the two dimensions (E-18 / #142). The public
convention is `T×n` (time in rows, variables in columns); the Kalman/PF routines
expect `n_obs × T_obs` internally, so:

- `size(data, 2) == n_obs`  → `T×n`, transpose to `n_obs × T_obs`;
- `size(data, 1) == n_obs`  → already `n_obs × T_obs`, use as-is;
- otherwise                 → `ArgumentError` naming `n_obs` and the received shape.

The only genuinely ambiguous case is `T == n_obs` (both dimensions equal `n_obs`);
it is resolved as `T×n` (rows = time), consistent with the package convention.
"""
function _orient_data(data::AbstractMatrix, n_obs::Int, ::Type{T}) where {T}
    dm = Matrix{T}(data)
    nrows, ncols = size(dm)
    if ncols == n_obs
        return Matrix{T}(dm')          # T×n (convention) → n_obs × T_obs
    elseif nrows == n_obs
        return dm                       # already n_obs × T_obs
    else
        throw(ArgumentError(
            "data has shape $(size(data)) but neither dimension equals the number of " *
            "observables n_obs=$n_obs. Pass data as T×n (time in rows, variables in columns)."))
    end
end

"""
    _resolve_measurement_error(measurement_error, data_mat, observables) -> Union{Nothing, Vector}

Resolve the `measurement_error` keyword before it reaches the observation-equation
builders. `nothing` passes through (zero ME). `:auto` scales measurement error to
each observable's own variance — √(0.1·var(yᵢ)) per series (DSGE.jl/Dynare heuristic),
which fixes the HA magnitude problem where aggregate observables span different scales
— and emits a warning. A vector passes through as-is. Any other symbol errors.
`data_mat` is n_obs × T_obs.
"""
function _resolve_measurement_error(measurement_error, data_mat::AbstractMatrix{T},
                                     observables::Vector{Symbol}) where {T<:AbstractFloat}
    measurement_error === nothing && return nothing
    if measurement_error isa Symbol
        measurement_error === :auto || throw(ArgumentError(
            "measurement_error Symbol must be :auto, got :$measurement_error"))
        n = size(data_mat, 1)
        me = Vector{T}(undef, n)
        for i in 1:n
            me[i] = sqrt(T(0.1) * var(view(data_mat, i, :)))
        end
        @warn "measurement_error=:auto: added per-observable measurement error at 10% of " *
              "each series' variance" observables sds=me
        return me
    end
    return Vector{T}(measurement_error)
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
- `data::AbstractMatrix` — observed data in `T×n` (time in rows, variables in columns),
  the package convention; orientation is resolved by matching a dimension to the number
  of observables (an `n×T` matrix is accepted and transposed internally).
- `θ0` — initial parameter guess. Preferred: a `Dict{Symbol}`/`NamedTuple` keyed by
  parameter name (order-independent). A positional `AbstractVector` is also accepted but
  **must be in sorted (alphabetical) prior-key order** and its length must equal the
  number of estimated parameters (otherwise an informative `ArgumentError` is thrown).

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
- `measurement_error::Union{Nothing,Symbol,Vector{<:Real}}=nothing` — measurement error SDs;
  `nothing` means zero ME (requires `n_obs ≤ n_shocks`, else a `StochasticSingularityError`
  is thrown); `:auto` adds per-observable ME at 10% of each series' variance with a warning
- `likelihood::Symbol=:auto` — likelihood evaluation method (currently auto = Kalman)
- `solver::Symbol=:gensys` — DSGE solver method
- `solver_kwargs::NamedTuple=NamedTuple()` — additional solver keyword arguments
- `delayed_acceptance::Bool=false` — enable two-stage delayed acceptance MH for `:smc2`
  (Christen & Fox 2005). Pre-screens proposals with a cheap bootstrap PF to avoid
  expensive CSMC evaluations on proposals that would be rejected. Exact posterior.
- `n_screen::Int=200` — particles for screening PF (only used when `delayed_acceptance=true`)
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
                              theta0::Union{AbstractVector{<:Real},
                                            AbstractDict{Symbol,<:Real},NamedTuple};
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
                              measurement_error::Union{Nothing,Symbol,Vector{<:Real}}=nothing,
                              likelihood::Symbol=:auto,
                              solver::Symbol=:gensys,
                              solver_kwargs::NamedTuple=NamedTuple(),
                              delayed_acceptance::Bool=false,
                              n_screen::Int=200,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}

    # ── 1. Build DSGEPrior from priors dict ──────────────────────────────
    # Auto-infer bounds from distribution support
    lower_dict = Dict{Symbol,Float64}()
    upper_dict = Dict{Symbol,Float64}()
    for (pn, d) in priors
        lo, hi = _infer_prior_bounds(d)
        lower_dict[pn] = lo
        upper_dict[pn] = hi
    end
    prior = DSGEPrior(priors; lower=lower_dict, upper=upper_dict)

    # ── 2. Sort param_names to match DSGEPrior ordering ──────────────────
    param_names = prior.param_names  # already sorted by DSGEPrior constructor

    # Resolve theta0: Dict/NamedTuple by name (order-independent), or a positional
    # vector in sorted prior-key order with length validation (E-12 / #136).
    theta0_sorted = _resolve_theta0(theta0, param_names, T)

    # ── 3. Handle observables ────────────────────────────────────────────
    if isempty(observables)
        observables = copy(spec.endog)
    end
    n_obs = length(observables)

    # ── 4. Data handling: resolve orientation by matching n_obs (E-18 / #142) ─
    # Public convention is T×n; Kalman/PF expect n_obs × T_obs internally.
    data_mat = _orient_data(data, n_obs, T)

    # Resolve :auto measurement error against data variance (per-observable). (#141/T042)
    measurement_error = _resolve_measurement_error(measurement_error, data_mat, observables)

    # Stochastic-singularity guard: with no measurement error and more observables than
    # structural shocks, the model-implied observation covariance is singular. Checked
    # eagerly here (spec.n_exog is exact and free) because the sampler closures would
    # otherwise swallow the builder-thrown error via their per-θ catch. (#141/T042)
    if measurement_error === nothing && n_obs > spec.n_exog
        throw(StochasticSingularityError(
            "$n_obs observables exceed $(spec.n_exog) structural shocks; the model-implied " *
            "observation covariance is singular and the likelihood is ill-defined. " *
            "Add measurement error (measurement_error=:auto or a vector of SDs) or " *
            "reduce the number of observables."))
    end

    # ── 5. Validate method ───────────────────────────────────────────────
    if method ∉ (:smc, :smc2, :mh)
        throw(ArgumentError("method must be :smc, :smc2, or :mh, got :$method"))
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
                                            solver, solver_kwargs)

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
                                            solver, solver_kwargs)

    else  # :mh
        draws, log_posterior, acceptance_rate, _mh_diag = _mh_sample(
            spec, data_mat, param_names, prior, theta0_sorted;
            n_draws=n_draws, burnin=burnin,
            observables=observables,
            measurement_error=measurement_error,
            solver=solver, solver_kwargs=solver_kwargs, rng=rng)
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
                                     solver, solver_kwargs,
                                     _mh_diag.n_failed, _mh_diag.n_evals)
    end
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
                                       solver_kwargs::NamedTuple) where {T<:AbstractFloat}
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
        state.n_lik_failures, state.n_lik_evals
    )
end

# =============================================================================
# Internal: Geweke (1999) modified harmonic mean marginal-likelihood estimator
# =============================================================================

"""
    _geweke_mhm(draws, log_post_kernel; p=0.5) -> T

Geweke (1999) modified harmonic mean (MHM) estimator of the **log marginal
likelihood** from posterior draws — the Dynare-standard estimator for MCMC output.

Arguments
- `draws::AbstractMatrix{T}` — `S×d` matrix of post-burn-in posterior draws
  (rows = draws, columns = parameters). Relies on burn-in already being
  discarded (see [T023]).
- `log_post_kernel::AbstractVector{T}` — per-draw log posterior *kernel*
  `log L(θ⁽ˢ⁾) + log π(θ⁽ˢ⁾)` (log likelihood + log prior). Any additive
  constant in the likelihood cancels in the ratio and shifts the estimate by
  that same constant, so the estimate is on the **same additive scale as the
  SMC tempering-path estimator** (the Kalman log-likelihood carries its own
  normalizing constant in both paths) — `:mh` and `:smc` marginal likelihoods,
  and Bayes factors between them, are therefore comparable.

Method. With posterior sample mean `θ̄` and covariance `Σ`, the weighting
density `f` is a truncated multivariate normal on the `p`-probability ellipsoid
`{θ : (θ−θ̄)'Σ⁻¹(θ−θ̄) ≤ χ²_{d,p}}`, normalized by the truncation mass `τ = p`:

    log f(θ) = −log p − (d/2) log(2π) − ½ log|Σ| − ½ (θ−θ̄)'Σ⁻¹(θ−θ̄)   (inside)
             = −∞                                                        (outside)

and the estimator is

    log p̂(y) = log S − logsumexp_s( log f(θ⁽ˢ⁾) − log_post_kernel(θ⁽ˢ⁾) ).

Returns `NaN` (with a `@warn`) when the effective (finite-kernel) chain is too
short (`S < 10·d`) or when no draw falls inside the truncation region.

Reference: Geweke (1999), *Econometric Reviews* 18(1), 1–73.
"""
function _geweke_mhm(draws::AbstractMatrix{T}, log_post_kernel::AbstractVector{T};
                     p::Real=0.5) where {T<:AbstractFloat}
    d = size(draws, 2)
    finite = findall(isfinite, log_post_kernel)
    if length(finite) < 10 * d
        @warn "Geweke MHM: effective post-burn-in chain too short " *
              "(finite draws = $(length(finite)) < 10·d = $(10*d)); returning NaN. " *
              "Increase n_draws or reduce burnin for a usable marginal likelihood."
        return T(NaN)
    end
    D = Matrix{T}(draws[finite, :])
    L = Vector{T}(log_post_kernel[finite])
    S = length(finite)

    # Posterior mean and (unbiased) covariance of the sampled draws.
    θbar = vec(mean(D; dims=1))
    Σ = d == 1 ? reshape([var(vec(D))], 1, 1) : cov(D)
    Σsym = Symmetric(Matrix{T}(Σ))
    Σinv = Matrix{T}(robust_inv(Σsym; silent=true))
    logdetΣ = logdet_safe(Σsym)

    # Truncated-normal weighting density constants; τ = p is the truncation mass.
    thresh = T(quantile(Chisq(d), p))
    log_const = -log(T(p)) - T(d) / 2 * log(T(2π)) - logdetΣ / 2

    terms = fill(T(-Inf), S)
    n_inside = 0
    @inbounds for s in 1:S
        δ = @view(D[s, :]) .- θbar
        q = dot(δ, Σinv, δ)
        if q <= thresh
            terms[s] = (log_const - q / 2) - L[s]   # log f(θ) − log kernel(θ)
            n_inside += 1
        end
    end
    if n_inside == 0
        @warn "Geweke MHM: no draws fell inside the p=$(p) truncation region; " *
              "returning NaN (degenerate or extremely diffuse posterior sample)."
        return T(NaN)
    end
    return log(T(S)) - _logsumexp(terms)
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
                                n_failed::Int=0,
                                n_evals::Int=0) where {T<:AbstractFloat}
    # Posterior mean from draws
    theta_mean = vec(mean(draws; dims=1))

    sol, ss = _build_solution_at_theta(spec, param_names, theta_mean,
                                        observables, measurement_error,
                                        solver, solver_kwargs)

    # Log marginal likelihood via Geweke (1999) modified harmonic mean (E-04 / #130).
    # `log_posterior` stores the per-draw kernel log L(θ) + log π(θ), on the same
    # additive scale as the SMC tempering-path estimator, so :mh and :smc marginal
    # likelihoods (and Bayes factors between them) are comparable. Returns NaN + @warn
    # when the post-burn-in chain is too short.
    log_ml = _geweke_mhm(draws, log_posterior)

    BayesianDSGE{T}(
        draws, log_posterior, param_names, prior,
        log_ml, :rwmh, acceptance_rate,
        T[], T[],  # no ESS history or phi schedule for MH
        spec, sol, ss,
        n_failed, n_evals
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
    posterior_summary(result::BayesianDSGE{T}) → Dict{Symbol, Dict{Symbol, T}}

Compute posterior summary statistics for each estimated parameter.

Returns a dictionary keyed by parameter name, each containing:
- `:mean` — posterior mean
- `:median` — posterior median (50th percentile)
- `:std` — posterior standard deviation
- `:ci_lower` — 2.5th percentile (lower bound of 95% credible interval)
- `:ci_upper` — 97.5th percentile (upper bound of 95% credible interval)
"""
function posterior_summary(result::BayesianDSGE{T}) where {T<:AbstractFloat}
    n_params = length(result.param_names)
    summary = Dict{Symbol, Dict{Symbol, T}}()

    for i in 1:n_params
        pn = result.param_names[i]
        draws_i = result.theta_draws[:, i]

        # Interpolated quantiles via Statistics.quantile rather than ad-hoc order-statistic
        # indexing (E-20 / #144). The stored draws are unweighted after the SMC terminal
        # resample (E-09 / #132), so a plain quantile is correct.
        summary[pn] = Dict{Symbol, T}(
            :mean => mean(draws_i),
            :median => quantile(draws_i, 0.5),
            :std => std(draws_i),
            :ci_lower => quantile(draws_i, 0.025),
            :ci_upper => quantile(draws_i, 0.975)
        )
    end

    return summary
end

"""
    marginal_likelihood(result::BayesianDSGE) → T

Return the log marginal likelihood (model evidence) from Bayesian estimation.

For SMC methods, this is the normalizing-constant estimate from the adaptive
tempering schedule (preferred when available). For RWMH (`:mh`), it is the
Geweke (1999) modified harmonic mean (MHM) estimate computed from the
post-burn-in draws — the Dynare standard — on the same additive scale as the
SMC estimate. The MHM returns `NaN` (with a warning) when the chain is too
short to form a usable estimate.

# References
- Geweke, J. (1999). Using simulation methods for Bayesian econometric models.
  *Econometric Reviews*, 18(1), 1-73.
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

Both marginal likelihoods must be on the same additive scale: SMC (tempering
path) and `:mh` (Geweke MHM) estimates are comparable by construction, but
compare estimators consistently and prefer SMC when available.

# References
- Kass, R. E. & Raftery, A. E. (1995). Bayes Factors.
  *Journal of the American Statistical Association*, 90(430), 773-795.
"""
function bayes_factor(r1::BayesianDSGE, r2::BayesianDSGE)
    return r1.log_marginal_likelihood - r2.log_marginal_likelihood
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
"""
function prior_posterior_table(result::BayesianDSGE{T}) where {T<:AbstractFloat}
    ps = posterior_summary(result)
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
            ci_upper = ps[pn][:ci_upper]
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

    # Collect one path per SUCCESSFUL draw; failed draws are DROPPED (not zero-filled,
    # which would bias the predictive distribution toward zero).
    paths_list = Vector{Matrix{T}}()

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
                sim = simulate(sol, T_periods; rng=rng)  # T_periods × n_vars
                push!(paths_list, Matrix{T}(sim))
            end
        catch e
            _benign_solve_error(e) || rethrow(e)
            continue
        end
    end

    n_valid = length(paths_list)
    n_dropped = n_sim - n_valid
    n_dropped > 0 && @warn "posterior_predictive: dropped $(n_dropped)/$(n_sim) draws " *
        "($(round(100*n_dropped/n_sim; digits=1))%) that failed to solve."

    # Stack into (n_valid × T_periods × n_vars); first dim is successful draws only.
    paths = Array{T,3}(undef, n_valid, T_periods, n_vars)
    for s in 1:n_valid
        paths[s, :, :] = paths_list[s]
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
        catch e
            _benign_solve_error(e) || rethrow(e)
            continue
        end
    end

    n_valid = length(all_irfs)
    n_skipped = n_sim - n_valid
    n_skipped > 0 && @warn "Bayesian IRF: skipped $(n_skipped)/$(n_sim) posterior draws " *
        "($(round(100*n_skipped/n_sim; digits=1))%) that failed to solve; bands conditioned on $(n_valid) draws."
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
        catch e
            _benign_solve_error(e) || rethrow(e)
            continue
        end
    end

    n_valid = length(all_fevds)
    n_skipped = n_sim - n_valid
    n_skipped > 0 && @warn "Bayesian FEVD: skipped $(n_skipped)/$(n_sim) posterior draws " *
        "($(round(100*n_skipped/n_sim; digits=1))%) that failed to solve; bands conditioned on $(n_valid) draws."
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
        catch e
            _benign_solve_error(e) || rethrow(e)
            continue
        end
    end

    n_valid = length(all_paths_list)
    n_skipped = n_sim - n_valid
    n_skipped > 0 && @warn "Bayesian simulate: skipped $(n_skipped)/$(n_sim) posterior draws " *
        "($(round(100*n_skipped/n_sim; digits=1))%) that failed to solve; bands conditioned on $(n_valid) draws."
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
    fail_share = result.n_lik_evals > 0 ?
        round(100 * result.n_failed_draws / result.n_lik_evals; digits=1) : 0.0
    spec_data = Any[
        "Method"                string(result.method);
        "Parameters"            n_params;
        "Posterior draws"       n_draws;
        "Log marginal lik."    round(result.log_marginal_likelihood; digits=4);
        "Acceptance rate"      round(result.acceptance_rate; digits=4);
        "Tempering stages"     length(result.phi_schedule);
        "Failed lik. evals"    string(result.n_failed_draws, " / ", result.n_lik_evals,
                                        " (", fail_share, "%)");
    ]
    _pretty_table(io, spec_data;
        title="Bayesian DSGE Estimation",
        column_labels=["", ""],
        alignment=[:l, :r])

    # Posterior summary table
    pnames = [string(s) for s in result.param_names]
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

    # Prior vs posterior table
    pt = prior_posterior_table(result)
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
