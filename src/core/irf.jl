# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Impulse Response Functions for frequentist and Bayesian VAR models.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Frequentist IRF
# =============================================================================

"""
    irf(model, horizon; method=:cholesky, ci_type=:none, reps=200, conf_level=0.95, ...)

Compute IRFs with optional confidence intervals.

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

# CI types
- `:none`
- `:bootstrap` --- nonparametric residual (recursive-design) bootstrap: resamples the
  estimated residuals, regenerates data from the estimated `B`, and re-estimates the VAR
  per replication. With `stationary_only=true`, draws whose companion matrix has
  `|λmax| ≥ 1` are rejected and redrawn. This is **not** the Kilian (1998) bias-corrected
  bootstrap — `B̂` is not bias-adjusted; the bias-corrected and wild bootstraps are tracked
  as roadmap task T271.
- `:theoretical` --- asymptotic (delta-method) confidence intervals.
"""
function irf(model::VARModel{T}, horizon::Int;
    method::Symbol=:cholesky, check_func=nothing, narrative_check=nothing,
    ci_type::Symbol=:none, reps::Int=200, conf_level::Real=0.95,
    stationary_only::Bool=false,
    shock_names::Union{Nothing,Vector{String}}=nothing,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing,
    seed::Union{Integer,Nothing}=nothing,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}

    # Reproducibility (T246/#345): a `seed` owns the RNG so bootstrap bands can be
    # reproduced bit-for-bit (the per-replication sub-seeding is thread-invariant).
    rng = _resolve_repro_rng(rng, seed)
    _validate_data(model.Sigma, "Sigma")
    _validate_data(model.B, "B")
    n = nvars(model)
    Q = compute_Q(model, method, horizon, check_func, narrative_check;
                  transition_var=transition_var, regime_indicator=regime_indicator, rng=rng)
    point_irf = compute_irf(model, Q, horizon)

    ci_lower, ci_upper = zeros(T, horizon, n, n), zeros(T, horizon, n, n)
    sim_irfs = nothing
    if ci_type != :none
        sim_irfs = _simulate_irfs(model, method, horizon, check_func, narrative_check, ci_type, reps;
                                  stationary_only=stationary_only,
                                  transition_var=transition_var, regime_indicator=regime_indicator,
                                  rng=rng)
        alpha = (1 - T(conf_level)) / 2
        @inbounds for h in 1:horizon, v in 1:n, s in 1:n
            d = @view sim_irfs[:, h, v, s]
            ci_lower[h, v, s], ci_upper[h, v, s] = quantile(d, alpha), quantile(d, 1 - alpha)
        end
    end

    snames = isnothing(shock_names) ? model.varnames : shock_names
    cl = ci_type == :none ? zero(T) : T(conf_level)
    # Attach a reproducibility manifest whenever the bands consume randomness.
    manifest = ci_type == :none ? nothing :
        capture_manifest(; seed=seed, settings=Dict{String,Any}(
            "method" => String(method), "ci_type" => String(ci_type),
            "reps" => reps, "stationary_only" => stationary_only))
    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                       model.varnames, snames, ci_type, sim_irfs, cl; manifest=manifest)
end

"""Simulate IRFs for confidence intervals (bootstrap or asymptotic)."""
function _simulate_irfs(model::VARModel{T}, method::Symbol, horizon::Int,
    check_func, narrative_check, ci_type::Symbol, reps::Int;
    stationary_only::Bool=false,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}
    n, p = nvars(model), model.p

    if ci_type == :bootstrap
        U, T_eff = model.U, size(model.U, 1)
        Y_init = model.Y[1:p, :]

        if stationary_only
            # Rejection sampling with DETERMINISTIC slot assignment (C-02): seed each of the
            # 1:max_iter iterations by its index, write each passing draw into a staging buffer
            # at its iteration index, then keep the first `reps` passing draws IN INDEX ORDER.
            # Result is invariant to thread scheduling / JULIA_NUM_THREADS.
            max_iter = 10 * reps
            staging = zeros(T, max_iter, horizon, n, n)
            passed = fill(false, max_iter)
            seeds = rand(rng, UInt64, max_iter)
            Threads.@threads for it in 1:max_iter
                local_rng = Random.MersenneTwister(seeds[it])
                _suppress_warnings() do
                    U_boot = U[rand(local_rng, 1:T_eff, T_eff), :]
                    Y_boot = _simulate_var(Y_init, model.B, U_boot, T_eff + p)
                    m = estimate_var(Y_boot, p; check_stability=false)
                    F = companion_matrix(m.B, n, p)
                    maximum(abs.(eigvals(F))) >= one(T) && return  # reject non-stationary draw
                    Q = compute_Q(m, method, horizon, check_func, narrative_check;
                                  transition_var=transition_var, regime_indicator=regime_indicator, rng=local_rng)
                    staging[it, :, :, :] = compute_irf(m, Q, horizon)
                    passed[it] = true
                end
            end
            kept = findall(passed)
            n_valid = min(length(kept), reps)
            n_valid < reps && @warn "Only $n_valid/$reps stationary bootstrap draws obtained after $max_iter iterations"
            sim_irfs = zeros(T, max(n_valid, 1), horizon, n, n)
            @inbounds for j in 1:n_valid
                sim_irfs[j, :, :, :] = staging[kept[j], :, :, :]
            end
            return sim_irfs
        else
            sim_irfs = zeros(T, reps, horizon, n, n)
            seeds = rand(rng, UInt64, reps)
            Threads.@threads for r in 1:reps
                local_rng = Random.MersenneTwister(seeds[r])
                _suppress_warnings() do
                    U_boot = U[rand(local_rng, 1:T_eff, T_eff), :]
                    Y_boot = _simulate_var(Y_init, model.B, U_boot, T_eff + p)
                    m = estimate_var(Y_boot, p; check_stability=false)
                    Q = compute_Q(m, method, horizon, check_func, narrative_check;
                                  transition_var=transition_var, regime_indicator=regime_indicator, rng=local_rng)
                    sim_irfs[r, :, :, :] = compute_irf(m, Q, horizon)
                end
            end
            return sim_irfs
        end
    elseif ci_type == :theoretical
        _, X = construct_var_matrices(model.Y, p)
        L_V, L_S = safe_cholesky(robust_inv(X'X)), safe_cholesky(model.Sigma)
        k = ncoefs(model)

        if stationary_only
            max_iter = 10 * reps
            staging = zeros(T, max_iter, horizon, n, n)
            passed = fill(false, max_iter)
            seeds = rand(rng, UInt64, max_iter)
            Threads.@threads for it in 1:max_iter
                local_rng = Random.MersenneTwister(seeds[it])
                _suppress_warnings() do
                    B_star = model.B + L_V * randn(local_rng, T, k, n) * L_S'
                    F = companion_matrix(B_star, n, p)
                    maximum(abs.(eigvals(F))) >= one(T) && return  # reject non-stationary draw
                    m = VARModel(zeros(T, 0, n), p, B_star, zeros(T, 0, n), model.Sigma, zero(T), zero(T), zero(T))
                    Q = compute_Q(m, method, horizon, check_func, narrative_check;
                                  transition_var=transition_var, regime_indicator=regime_indicator, rng=local_rng)
                    staging[it, :, :, :] = compute_irf(m, Q, horizon)
                    passed[it] = true
                end
            end
            kept = findall(passed)
            n_valid = min(length(kept), reps)
            n_valid < reps && @warn "Only $n_valid/$reps stationary theoretical draws obtained after $max_iter iterations"
            sim_irfs = zeros(T, max(n_valid, 1), horizon, n, n)
            @inbounds for j in 1:n_valid
                sim_irfs[j, :, :, :] = staging[kept[j], :, :, :]
            end
            return sim_irfs
        else
            sim_irfs = zeros(T, reps, horizon, n, n)
            seeds = rand(rng, UInt64, reps)
            Threads.@threads for r in 1:reps
                local_rng = Random.MersenneTwister(seeds[r])
                _suppress_warnings() do
                    B_star = model.B + L_V * randn(local_rng, T, k, n) * L_S'
                    m = VARModel(zeros(T, 0, n), p, B_star, zeros(T, 0, n), model.Sigma, zero(T), zero(T), zero(T))
                    Q = compute_Q(m, method, horizon, check_func, narrative_check;
                                  transition_var=transition_var, regime_indicator=regime_indicator, rng=local_rng)
                    sim_irfs[r, :, :, :] = compute_irf(m, Q, horizon)
                end
            end
            return sim_irfs
        end
    end
    zeros(T, reps, horizon, n, n)
end

"""Simulate VAR data from initial conditions and innovations."""
function _simulate_var(Y_init::AbstractMatrix{T}, B::AbstractMatrix{T},
                       U::AbstractMatrix{T}, T_total::Int) where {T<:AbstractFloat}
    p, n = size(Y_init)
    Y = zeros(T, T_total, n)
    Y[1:p, :] = Y_init

    A = extract_ar_coefficients(B, n, p)
    intercept = @view B[1, :]

    @inbounds for t in (p+1):T_total
        Y[t, :] = intercept
        for i in 1:p
            Y[t, :] .+= A[i] * @view(Y[t-i, :])
        end
        Y[t, :] .+= @view(U[t-p, :])
    end
    Y
end

# =============================================================================
# Bayesian IRF
# =============================================================================

"""
    irf(post::BVARPosterior, horizon; method=:cholesky, quantiles=[0.16, 0.5, 0.84], point_estimate=:mean, ...)

Compute Bayesian IRFs from posterior draws with posterior quantiles.
Uses posterior mean as central tendency by default (pass `point_estimate=:median` for median).

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

Uses `process_posterior_samples` and `compute_posterior_quantiles` from bayesian_utils.jl.
"""
function irf(post::BVARPosterior, horizon::Int;
    method::Symbol=:cholesky, data::AbstractMatrix=Matrix{Float64}(undef, 0, 0),
    check_func=nothing, narrative_check=nothing, quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
    threaded::Bool=false, point_estimate::Symbol=:mean,
    shock_names::Union{Nothing,Vector{String}}=nothing,
    max_draws::Int=1000,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing
)
    use_data = isempty(data) ? post.data : data
    _validate_narrative_data(method, use_data)

    n = post.n
    ET = eltype(use_data)

    # Process posterior samples using shared utility
    results, samples = process_posterior_samples(post,
        (m, Q, h) -> compute_irf(m, Q, h);
        data=use_data, method=method, horizon=horizon,
        check_func=check_func, narrative_check=narrative_check,
        max_draws=max_draws,
        transition_var=transition_var, regime_indicator=regime_indicator
    )

    # Stack results into single array
    all_irfs = stack_posterior_results(results, (horizon, n, n), ET)

    # Compute quantiles using shared utility (threaded for large arrays)
    q_vec = ET.(quantiles)
    use_threaded = threaded || (samples * horizon * n * n > 100000)
    irf_q, irf_m = compute_posterior_quantiles(all_irfs, q_vec; threaded=use_threaded, central=point_estimate)

    snames = isnothing(shock_names) ? post.varnames : shock_names
    # MC honesty (#244): process_posterior_samples drops non-stationary / unidentified draws.
    n_req = post.n_draws
    BayesianImpulseResponse{ET}(irf_q, irf_m, horizon, post.varnames, snames, q_vec, all_irfs,
                                n_req, samples, n_req - samples)
end

# Deprecated wrapper for old (chain, p, n, horizon) signature
function irf(post::BVARPosterior, p::Int, n::Int, horizon::Int; kwargs...)
    irf(post, horizon; kwargs...)
end

# =============================================================================
# Structural LP IRF Accessor
# =============================================================================

"""
    irf(slp::StructuralLP) -> ImpulseResponse

Extract the impulse response object from a structural LP result.
"""
irf(slp::StructuralLP) = slp.irf

# =============================================================================
# Local Projection IRF
# =============================================================================

"""
    lp_irf(model::LPModel{T}; conf_level::Real=0.95) -> LPImpulseResponse{T}

Extract impulse response function with confidence intervals from LP model.
"""
function lp_irf(model::LPModel{T}; conf_level::Real=0.95) where {T<:AbstractFloat}
    irf_data = extract_shock_irf(model.B, model.vcov, model.response_vars, 2;
                                  conf_level=conf_level)

    response_names = model.varnames[model.response_vars]
    shock_name = model.varnames[model.shock_var]
    cov_type_sym = model.cov_estimator isa NeweyWestEstimator ? :newey_west : :white

    LPImpulseResponse{T}(irf_data.values, irf_data.ci_lower, irf_data.ci_upper,
                         irf_data.se, model.horizon, response_names, shock_name,
                         cov_type_sym, T(conf_level))
end

"""
    lp_irf(Y::AbstractMatrix, shock_var::Int, horizon::Int; kwargs...) -> LPImpulseResponse

Convenience function: estimate LP and extract IRF in one call.
"""
function lp_irf(Y::AbstractMatrix, shock_var::Int, horizon::Int; conf_level::Real=0.95, kwargs...)
    model = estimate_lp(Y, shock_var, horizon; kwargs...)
    lp_irf(model; conf_level=conf_level)
end

# =============================================================================
# Cumulative IRF
# =============================================================================

"""
    cumulative_irf(irf::LPImpulseResponse{T}) -> LPImpulseResponse{T}

Compute cumulative impulse response: Σₛ₌₀ʰ β_s.
"""
function cumulative_irf(irf::LPImpulseResponse{T}) where {T<:AbstractFloat}
    cum_values = cumsum(irf.values, dims=1)
    cum_se = sqrt.(cumsum(irf.se.^2, dims=1))

    z = T(quantile(Normal(), 1 - (1 - irf.conf_level) / 2))
    cum_ci_lower = cum_values .- z .* cum_se
    cum_ci_upper = cum_values .+ z .* cum_se

    LPImpulseResponse{T}(cum_values, cum_ci_lower, cum_ci_upper, cum_se, irf.horizon,
                         irf.response_vars, irf.shock_var, irf.cov_type, irf.conf_level)
end

"""
    cumulative_irf(irf_result::ImpulseResponse{T}) -> ImpulseResponse{T}

Compute cumulative impulse response for VAR models: Σₛ₌₀ʰ IRF_s.

When raw bootstrap/simulation draws are available, cumulates each draw first
then extracts quantiles — the statistically correct approach since quantiles
are NOT additive: Q_α(A+B) ≠ Q_α(A) + Q_α(B).
"""
function cumulative_irf(irf_result::ImpulseResponse{T}) where {T<:AbstractFloat}
    cum_values = cumsum(irf_result.values, dims=1)

    if irf_result._draws !== nothing && irf_result._conf_level > zero(T)
        # Correct approach: cumulate each draw, then extract quantiles
        cum_draws = cumsum(irf_result._draws, dims=2)
        alpha = (one(T) - irf_result._conf_level) / 2
        horizon, nv, ns = size(cum_values)
        cum_lower = zeros(T, horizon, nv, ns)
        cum_upper = zeros(T, horizon, nv, ns)
        @inbounds for h in 1:horizon, v in 1:nv, s in 1:ns
            d = @view cum_draws[:, h, v, s]
            cum_lower[h, v, s] = quantile(d, alpha)
            cum_upper[h, v, s] = quantile(d, 1 - alpha)
        end
    else
        # Fallback for no-CI case (ci_lower/ci_upper are zeros)
        cum_lower = cumsum(irf_result.ci_lower, dims=1)
        cum_upper = cumsum(irf_result.ci_upper, dims=1)
    end

    ImpulseResponse{T}(cum_values, cum_lower, cum_upper, irf_result.horizon,
                       irf_result.variables, irf_result.shocks, irf_result.ci_type)
end

"""
    cumulative_irf(irf_result::BayesianImpulseResponse{T}) -> BayesianImpulseResponse{T}

Compute cumulative Bayesian impulse response: Σₛ₌₀ʰ IRF_s.

When raw posterior draws are available, cumulates each draw first then
extracts quantiles — the statistically correct approach.
"""
function cumulative_irf(irf_result::BayesianImpulseResponse{T}) where {T<:AbstractFloat}
    cum_pe = cumsum(irf_result.point_estimate, dims=1)

    if irf_result._draws !== nothing
        # Correct approach: cumulate each draw, then extract quantiles
        cum_draws = cumsum(irf_result._draws, dims=2)
        q_vec = irf_result.quantile_levels
        horizon, nv, ns = size(cum_pe)
        nq = length(q_vec)
        cum_quantiles = zeros(T, horizon, nv, ns, nq)
        @inbounds for h in 1:horizon, v in 1:nv, s in 1:ns
            d = @view cum_draws[:, h, v, s]
            for (qi, q) in enumerate(q_vec)
                cum_quantiles[h, v, s, qi] = quantile(d, q)
            end
        end
    else
        cum_quantiles = cumsum(irf_result.quantiles, dims=1)
    end

    # Cumulation is a deterministic transform of the same draws — propagate the MC counts.
    BayesianImpulseResponse{T}(cum_quantiles, cum_pe, irf_result.horizon,
                               irf_result.variables, irf_result.shocks, irf_result.quantile_levels,
                               nothing, irf_result.n_requested, irf_result.n_effective, irf_result.n_failed)
end

# =============================================================================
# Reproducibility (T246 / #345)
# =============================================================================

"""
    reproduce(ir::ImpulseResponse, model::VARModel) -> ReproReport

Re-run the bootstrap that produced `ir`'s confidence bands from the manifest's
recorded seed and settings and check the point IRF and bands match bit-for-bit.
The source `model` is passed explicitly — a large object deliberately not retained
on the IRF result. Requires `ir` to have been produced by
`irf(model, H; ci_type=:bootstrap, seed=N)`.

```julia
model = estimate_var(Y, 2)
ir = irf(model, 20; ci_type=:bootstrap, reps=200, seed=20260717)
reproduce(ir, model)   # ReproReport: PASS
```
"""
function reproduce(ir::ImpulseResponse, model::VARModel)
    m = ir.manifest
    m === nothing && return _no_manifest_report("ImpulseResponse")
    m.seed === nothing && return _no_seed_report(m, "irf(model, H; ci_type=:bootstrap, seed=N)")
    s = m.settings
    fresh = irf(model, ir.horizon;
                method = Symbol(get(s, "method", "cholesky")),
                ci_type = Symbol(get(s, "ci_type", "bootstrap")),
                reps = Int(get(s, "reps", ir._draws === nothing ? 0 : size(ir._draws, 1))),
                conf_level = ir._conf_level,
                stationary_only = Bool(get(s, "stationary_only", false)),
                seed = m.seed)
    diffs = [_repro_field_diff("values", ir.values, fresh.values),
             _repro_field_diff("ci_lower", ir.ci_lower, fresh.ci_lower),
             _repro_field_diff("ci_upper", ir.ci_upper, fresh.ci_upper)]
    return _finalize_repro(diffs, m)
end

# Single-argument form: the source model is required (not retained on the result).
reproduce(::ImpulseResponse) =
    _needs_source_report("bootstrap ImpulseResponse", "reproduce(ir, model)")
