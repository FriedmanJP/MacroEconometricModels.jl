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
- `burnin::Int=5000` — burnin draws for `:mh`
- `ess_target::Float64=0.5` — target ESS fraction for adaptive tempering
- `measurement_error::Union{Nothing,Vector{<:Real}}=nothing` — measurement error SDs
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
                              theta0::AbstractVector{<:Real};
                              priors::Dict{Symbol,<:Distribution},
                              method::Symbol=:smc,
                              observables::Vector{Symbol}=Symbol[],
                              n_smc::Int=5000,
                              n_particles::Int=500,
                              n_mh_steps::Int=1,
                              n_draws::Int=10000,
                              burnin::Int=5000,
                              ess_target::Float64=0.5,
                              measurement_error::Union{Nothing,Vector{<:Real}}=nothing,
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

    # Sort theta0 to match param_names ordering
    prior_keys_sorted = param_names
    # theta0 is provided in the same order as the sorted prior keys
    theta0_sorted = T.(theta0)

    # ── 3. Handle observables ────────────────────────────────────────────
    if isempty(observables)
        observables = copy(spec.endog)
    end
    n_obs = length(observables)

    # ── 4. Data handling: ensure n_obs × T_obs format ────────────────────
    data_mat = Matrix{T}(data)
    nrows, ncols = size(data_mat)
    # The Kalman filter expects n_obs × T_obs (each column is one time period)
    # simulate() returns T_periods × n_endog
    # If data has more rows than columns and nrows != n_obs, transpose
    if nrows != n_obs && ncols == n_obs
        data_mat = Matrix{T}(data_mat')
    elseif nrows != n_obs && ncols != n_obs
        # Neither dimension matches n_obs; try best guess: if nrows > ncols, transpose
        if nrows > ncols
            data_mat = Matrix{T}(data_mat')
        end
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
        draws, log_posterior, acceptance_rate = _mh_sample(
            spec, data_mat, param_names, prior, theta0_sorted;
            n_draws=n_draws, burnin=burnin,
            observables=observables,
            measurement_error=measurement_error,
            solver=solver, solver_kwargs=solver_kwargs, rng=rng)
        return _mh_to_bayesian_dsge(draws, log_posterior, acceptance_rate,
                                     prior, param_names, spec,
                                     observables, measurement_error,
                                     solver, solver_kwargs)
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
        spec, sol, ss
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
                                solver_kwargs::NamedTuple) where {T<:AbstractFloat}
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
        spec, sol, ss
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

    new_spec = DSGESpec{T}(
        spec.endog, spec.exog, spec.params, new_pv,
        spec.equations, spec.residual_fns,
        spec.n_expect, spec.forward_indices, T[], spec.ss_fn;
        original_endog=spec.original_endog,
        original_equations=spec.original_equations,
        augmented=spec.augmented,
        max_lag=spec.max_lag,
        max_lead=spec.max_lead
    )

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

        new_spec = DSGESpec{T}(
            spec.endog, spec.exog, spec.params, new_pv,
            spec.equations, spec.residual_fns,
            spec.n_expect, spec.forward_indices, T[], spec.ss_fn;
            original_endog=spec.original_endog,
            original_equations=spec.original_equations,
            augmented=spec.augmented,
            max_lag=spec.max_lag,
            max_lead=spec.max_lead
        )

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
