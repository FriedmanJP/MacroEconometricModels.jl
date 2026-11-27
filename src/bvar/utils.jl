# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Bayesian Processing Utilities for MCMC Chain Analysis.

Provides shared utilities for processing posterior samples across different
Bayesian analysis functions (IRF, FEVD, Historical Decomposition).

The main functions are:
- `process_posterior_samples`: Generic loop over MCMC chain samples
- `compute_posterior_quantiles!`: Compute quantiles and means from samples (in-place)
- `compute_posterior_quantiles_threaded!`: Threaded version for large arrays
"""

using LinearAlgebra, Statistics

# =============================================================================
# Posterior Sample Processing
# =============================================================================

# Explicit allowlist of per-draw BVAR identification methods safe for outer-loop
# `Threads.@threads :dynamic` parallelization (#945). Methods that call non-thread-safe
# C libraries (NLopt, e.g., :robust_bayes) or already spawn heavy internal threading
# (:arias, :uhlig, :markov_switching) stay outer-serial.
const _PARALLEL_SAFE_BVAR_ID_METHODS = (
    :cholesky, :sign, :narrative, :long_run,
    :fastica, :jade, :sobi, :dcov, :hsic, :cvm,
    :student_t, :mixture_normal, :mixture_normal_asym, :pml, :skew_normal,
    :lms, :gmm_moments, :lewis_tvv, :sv_em,
    :garch, :smooth_transition, :external_volatility,
)

@inline _is_parallel_safe_bvar_id_method(method::Symbol) = method in _PARALLEL_SAFE_BVAR_ID_METHODS

"""
    process_posterior_samples(post::BVARPosterior, compute_func::Function;
                              data, method, horizon, check_func, narrative_check,
                              max_draws, transition_var, regime_indicator,
                              threaded=true, rng=Random.default_rng()) -> (Vector, Int)

Generic framework for processing posterior samples from BVARPosterior.

# Process
1. Extract chain parameters using `extract_chain_parameters`
2. Loop over samples (in parallel via `Threads.@threads :dynamic` with pre-generated
   per-draw `Xoshiro` seeds when `threaded=true && Threads.nthreads() > 1` and `method`
   is parallel-safe, producing bit-identical results across thread counts)
3. Skip non-stationary draws (counted separately from identification failures)
4. Compute identification matrix Q using specified method
5. Apply `compute_func(model, Q, horizon)` to get result for each sample
6. Catch [`IdentificationError`](@ref) per draw (skip + count); throw only if every draw fails

# Arguments
- `post::BVARPosterior`: Posterior draws from `estimate_bvar`
- `compute_func::Function`: Function taking (model, Q, horizon) -> result

# Keyword Arguments
- `data::AbstractMatrix`: Override data (defaults to post.data)
- `method::Symbol`: Identification method (see `compute_Q` for full list)
- `horizon::Int`: IRF/computation horizon
- `check_func`: Sign restriction check function (for method=:sign or :narrative)
- `narrative_check`: Narrative restriction check function (for method=:narrative)
- `max_draws::Int`: Maximum draws for sign/narrative identification
- `transition_var`: Transition variable (for method=:smooth_transition)
- `regime_indicator`: Regime indicator (for method=:external_volatility)
- `threaded::Bool=true`: Parallelize across posterior draws when `Threads.nthreads() > 1`
  and `method` is thread-safe. For optimal multi-core scaling on small `n × n` matrix loops,
  pair `Threads.nthreads() > 1` with `BLAS.set_num_threads(1)` (or `OPENBLAS_NUM_THREADS=1`);
  this routine also scopes single-thread BLAS automatically around its parallel region.
- `rng::AbstractRNG=Random.default_rng()`: RNG used to generate thread-count-invariant
  per-draw `Xoshiro` seeds

# Returns
- `results::Vector`: Vector of results from `compute_func` for each valid sample; the element type is
  the concrete type returned by `compute_func` (inferred from the first valid draw, not `Any`)
- `n_samples::Int`: Number of valid samples (`n_requested - n_unidentified - n_nonstationary`).
  `n_failed` on IRF/FEVD/HD is this gap — unidentified and non-stationary are counted separately
  in the all-fail error and in the unidentified-fraction warning.

# Example
```julia
# Compute IRF for each posterior sample
post = estimate_bvar(Y, 2; n_draws=500)
results, n_samples = process_posterior_samples(post,
    (m, Q, h) -> compute_irf(m, Q, h);
    horizon=20, method=:cholesky
)
```
"""
function process_posterior_samples(post::BVARPosterior, compute_func::Function;
    data::AbstractMatrix=Matrix{Float64}(undef, 0, 0),
    method::Symbol=:cholesky, horizon::Int=20,
    check_func=nothing, narrative_check=nothing, max_draws::Int=5000,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing,
    restrictions=nothing,
    normalize=nothing, shock_size=nothing,  # peeled: never forwarded to compute_Q
    threaded::Bool=true,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...
)
    use_data = isempty(data) ? post.data : data
    method == :narrative && isempty(use_data) &&
        throw(ArgumentError("Narrative method requires data"))

    samples = post.n_draws
    p, n = post.p, post.n
    b_vecs, sigmas = extract_chain_parameters(post)
    T = eltype(use_data)

    # Match statistical-ID columns to the posterior-mean impact (SID-04). Set-ID
    # methods (:sign, :narrative), recursive/long-run, and :proxy are left unmatched.
    # If the mean model is unidentified, warn and fall back to the first
    # successfully identified draw as P_ref (re-matching that draw is a no-op).
    match_columns = _should_match_columns(method)
    P_ref = nothing
    if match_columns
        m_mean = posterior_mean_model(post; data=use_data)
        try
            Q_mean = compute_Q(m_mean, method; horizon=horizon, check_func=check_func,
                               narrative_check=narrative_check, restrictions=restrictions,
                               max_draws=max_draws, transition_var=transition_var,
                               regime_indicator=regime_indicator, rng=rng, kwargs...)
            P_ref = Matrix{T}(safe_cholesky(m_mean.Sigma) * Q_mean)
        catch e
            e isa IdentificationError || rethrow(e)
            @warn "Column matching skipped because the reference Q could not be identified from the posterior-mean model; using first identified draw as P_ref"
        end
    end

    # Pre-generate per-draw seeds sequentially before the loop so serial (threaded=false)
    # and multi-threaded (threaded=true, any thread count) runs are 100% bit-identical (#945).
    seeds = rand(rng, UInt64, samples)
    # Status per draw: 0 = ok, 1 = non-stationary, 2 = unidentified
    statuses = fill(Int8(2), samples)
    slots = Vector{Any}(undef, samples)

    use_parallel = threaded && Threads.nthreads() > 1 && samples > 1 &&
                   _is_parallel_safe_bvar_id_method(method) &&
                   !(match_columns && P_ref === nothing)

    if use_parallel
        P_ref_fixed = P_ref
        _with_single_blas_thread() do
            Threads.@threads :dynamic for s in 1:samples
                local_rng = Random.Xoshiro(seeds[s])
                m = parameters_to_model(b_vecs[s, :], sigmas[s, :], p, n, use_data)
                if !is_stationary(m).is_stationary
                    statuses[s] = Int8(1)
                    continue
                end
                try
                    Q = compute_Q(m, method; horizon=horizon, check_func=check_func,
                                  narrative_check=narrative_check, restrictions=restrictions,
                                  max_draws=max_draws, transition_var=transition_var,
                                  regime_indicator=regime_indicator, rng=local_rng, kwargs...)
                    Q, _ = _maybe_match_Q(Q, m, P_ref_fixed)
                    slots[s] = compute_func(m, Q, horizon)
                    statuses[s] = Int8(0)
                catch e
                    e isa IdentificationError || rethrow(e)
                    statuses[s] = Int8(2)
                end
            end
        end
    else
        for s in 1:samples
            local_rng = Random.Xoshiro(seeds[s])
            m = parameters_to_model(b_vecs[s, :], sigmas[s, :], p, n, use_data)
            if !is_stationary(m).is_stationary
                statuses[s] = Int8(1)
                continue
            end
            try
                Q = compute_Q(m, method; horizon=horizon, check_func=check_func,
                              narrative_check=narrative_check, restrictions=restrictions,
                              max_draws=max_draws, transition_var=transition_var,
                              regime_indicator=regime_indicator, rng=local_rng, kwargs...)
                if P_ref === nothing && match_columns
                    P_ref = Matrix{T}(safe_cholesky(m.Sigma) * Q)
                end
                Q, _ = _maybe_match_Q(Q, m, P_ref)
                slots[s] = compute_func(m, Q, horizon)
                statuses[s] = Int8(0)
            catch e
                e isa IdentificationError || rethrow(e)
                statuses[s] = Int8(2)
            end
        end
    end

    valid_idx = findall(==(Int8(0)), statuses)
    valid_count = length(valid_idx)
    n_nonstationary = count(==(Int8(1)), statuses)
    n_unidentified = count(==(Int8(2)), statuses)

    valid_count == 0 && throw(IdentificationError(
        "All posterior draws failed identification or were non-stationary " *
        "(unidentified=$n_unidentified, non-stationary=$n_nonstationary)"))
    n_nonstationary > samples ÷ 2 && @warn "$n_nonstationary/$samples posterior draws non-stationary, skipped"
    frac_u = T(n_unidentified) / T(samples)
    frac_u > T(0.5) && @warn "$n_unidentified/$samples posterior draws unidentified ($n_nonstationary non-stationary)"

    # Concrete-typed result container inferred from first valid draw (#210 box A)
    first_res = slots[valid_idx[1]]
    results = Vector{typeof(first_res)}(undef, valid_count)
    @inbounds for (k, s) in enumerate(valid_idx)
        results[k] = slots[s]
    end

    results, valid_count
end

# Deprecated wrapper for old (chain, p, n) signature
function process_posterior_samples(post::BVARPosterior, p::Int, n::Int, compute_func::Function; kwargs...)
    process_posterior_samples(post, compute_func; kwargs...)
end

# =============================================================================
# Quantile Computation
# =============================================================================

"""
    compute_posterior_quantiles!(quantile_out::AbstractArray{T}, mean_out::AbstractArray{T},
                                  samples::AbstractArray{T}, q_vec::AbstractVector{T}) where {T}

Compute quantiles and means from posterior samples (in-place).

Operates over the first dimension (samples) and computes quantiles/means for all other dimensions.

# Arguments
- `quantile_out`: Output array for quantiles, size = (other_dims..., n_quantiles)
- `mean_out`: Output array for means, size = (other_dims...)
- `samples`: Input samples array, size = (n_samples, other_dims...)
- `q_vec`: Vector of quantile levels (e.g., [0.16, 0.5, 0.84])

# Example
```julia
samples = randn(1000, 20, 3, 3)  # 1000 samples of 20×3×3 IRF
q_out = zeros(20, 3, 3, 3)      # 3 quantiles
m_out = zeros(20, 3, 3)
compute_posterior_quantiles!(q_out, m_out, samples, [0.16, 0.5, 0.84])
```
"""
function compute_posterior_quantiles!(quantile_out::AbstractArray{T}, mean_out::AbstractArray{T},
                                       samples::AbstractArray{T}, q_vec::AbstractVector;
                                       central::Symbol=:mean) where {T<:AbstractFloat}
    other_dims = size(samples)[2:end]
    n_q = length(q_vec)

    @assert size(quantile_out) == (other_dims..., n_q) "quantile_out size mismatch"
    @assert size(mean_out) == other_dims "mean_out size mismatch"

    @inbounds for idx in CartesianIndices(other_dims)
        d = @view samples[:, idx]
        mean_out[idx] = central == :median ? median(d) : mean(d)
        for (qi, q) in enumerate(q_vec)
            quantile_out[idx, qi] = quantile(d, q)
        end
    end

    nothing
end

"""
    compute_posterior_quantiles_threaded!(quantile_out::AbstractArray{T}, mean_out::AbstractArray{T},
                                           samples::AbstractArray{T}, q_vec::AbstractVector{T}) where {T}

Threaded version of quantile computation for large arrays.

Uses `Threads.@threads` to parallelize over the index space.
Recommended when `prod(size(samples)[2:end]) > 1000`.
"""
function compute_posterior_quantiles_threaded!(quantile_out::AbstractArray{T}, mean_out::AbstractArray{T},
                                                samples::AbstractArray{T}, q_vec::AbstractVector;
                                                central::Symbol=:mean) where {T<:AbstractFloat}
    other_dims = size(samples)[2:end]
    n_q = length(q_vec)

    @assert size(quantile_out) == (other_dims..., n_q) "quantile_out size mismatch"
    @assert size(mean_out) == other_dims "mean_out size mismatch"

    # Flatten to linear indices for threading
    indices = collect(CartesianIndices(other_dims))

    Threads.@threads for idx in indices
        @inbounds begin
            d = @view samples[:, idx]
            mean_out[idx] = central == :median ? median(d) : mean(d)
            for (qi, q) in enumerate(q_vec)
                quantile_out[idx, qi] = quantile(d, q)
            end
        end
    end

    nothing
end

"""
    compute_posterior_quantiles(samples::AbstractArray{T}, q_vec::AbstractVector;
                                 threaded::Bool=false) where {T}

Compute quantiles and means from posterior samples (allocating version).

# Arguments
- `samples`: Input samples array, size = (n_samples, other_dims...)
- `q_vec`: Vector of quantile levels
- `threaded`: Use threaded version for large arrays

# Returns
- `quantiles`: Array of shape (other_dims..., n_quantiles)
- `means`: Array of shape (other_dims...)
"""
function compute_posterior_quantiles(samples::AbstractArray{T}, q_vec::AbstractVector;
                                      threaded::Bool=false,
                                      central::Symbol=:mean) where {T<:AbstractFloat}
    other_dims = size(samples)[2:end]
    n_q = length(q_vec)
    q_vec_T = T.(q_vec)

    quantile_out = zeros(T, other_dims..., n_q)
    mean_out = zeros(T, other_dims...)

    if threaded && prod(other_dims) > 1000
        compute_posterior_quantiles_threaded!(quantile_out, mean_out, samples, q_vec_T; central=central)
    else
        compute_posterior_quantiles!(quantile_out, mean_out, samples, q_vec_T; central=central)
    end

    quantile_out, mean_out
end

# =============================================================================
# Weighted Quantile Computation (for Arias et al. identification)
# =============================================================================

"""
    compute_weighted_quantiles!(quantile_out::AbstractArray{T}, mean_out::AbstractArray{T},
                                 samples::AbstractArray{T}, weights::AbstractVector{T},
                                 q_vec::AbstractVector) where {T}

Compute weighted quantiles and weighted means from posterior samples (in-place).

Used for importance-weighted posterior inference (e.g., Arias et al. 2018 SVAR).

# Arguments
- `quantile_out`: Output array for quantiles
- `mean_out`: Output array for weighted means
- `samples`: Input samples array, size = (n_samples, other_dims...)
- `weights`: Importance weights, normalized to sum to 1
- `q_vec`: Vector of quantile levels
"""
function compute_weighted_quantiles!(quantile_out::AbstractArray{T}, mean_out::AbstractArray{T},
                                      samples::AbstractArray{T}, weights::AbstractVector{T},
                                      q_vec::AbstractVector) where {T<:AbstractFloat}
    other_dims = size(samples)[2:end]
    n_q = length(q_vec)

    @assert size(quantile_out) == (other_dims..., n_q) "quantile_out size mismatch"
    @assert size(mean_out) == other_dims "mean_out size mismatch"
    @assert length(weights) == size(samples, 1) "weights length must match n_samples"

    @inbounds for idx in CartesianIndices(other_dims)
        d = @view samples[:, idx]
        mean_out[idx] = sum(weights .* d)
        for (qi, q) in enumerate(q_vec)
            quantile_out[idx, qi] = _weighted_quantile(d, weights, q)
        end
    end

    nothing
end

"""
    compute_weighted_quantiles_threaded!(quantile_out::AbstractArray{T}, mean_out::AbstractArray{T},
                                          samples::AbstractArray{T}, weights::AbstractVector{T},
                                          q_vec::AbstractVector) where {T}

Threaded version of weighted quantile computation for large arrays.
"""
function compute_weighted_quantiles_threaded!(quantile_out::AbstractArray{T}, mean_out::AbstractArray{T},
                                               samples::AbstractArray{T}, weights::AbstractVector{T},
                                               q_vec::AbstractVector) where {T<:AbstractFloat}
    other_dims = size(samples)[2:end]
    n_q = length(q_vec)

    @assert size(quantile_out) == (other_dims..., n_q) "quantile_out size mismatch"
    @assert size(mean_out) == other_dims "mean_out size mismatch"
    @assert length(weights) == size(samples, 1) "weights length must match n_samples"

    indices = collect(CartesianIndices(other_dims))

    Threads.@threads for idx in indices
        @inbounds begin
            d = @view samples[:, idx]
            mean_out[idx] = sum(weights .* d)
            for (qi, q) in enumerate(q_vec)
                quantile_out[idx, qi] = _weighted_quantile(d, weights, q)
            end
        end
    end

    nothing
end

# =============================================================================
# Helper: Convert IRF/FEVD/HD array to standard format for quantile computation
# =============================================================================

"""
    stack_posterior_results(results::Vector, result_size::Tuple, ::Type{T}=Float64) where {T}

Stack vector of results into single array for quantile computation.

# Arguments
- `results`: Vector of arrays from posterior samples
- `result_size`: Expected size of each result array
- `T`: Element type

# Returns
Array of size (n_samples, result_size...)
"""
function stack_posterior_results(results::Vector, result_size::Tuple, ::Type{T}=Float64) where {T}
    n_samples = length(results)
    stacked = zeros(T, n_samples, result_size...)

    @inbounds for s in 1:n_samples
        stacked[s, axes(stacked)[2:end]...] = results[s]
    end

    stacked
end
