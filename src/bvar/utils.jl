"""
Bayesian Processing Utilities for MCMC Chain Analysis.

Provides shared utilities for processing posterior samples across different
Bayesian analysis functions (IRF, FEVD, Historical Decomposition).

The main functions are:
- `process_posterior_samples`: Generic loop over MCMC chain samples
- `compute_posterior_quantiles!`: Compute quantiles and means from samples (in-place)
- `compute_posterior_quantiles_threaded!`: Threaded version for large arrays
"""

using LinearAlgebra, Statistics, MCMCChains

# =============================================================================
# Posterior Sample Processing
# =============================================================================

"""
    process_posterior_samples(chain::Chains, p::Int, n::Int, compute_func::Function;
                              data, method, horizon, check_func, narrative_check,
                              max_draws, transition_var, regime_indicator) -> (Vector{Any}, Int)

Generic framework for processing posterior samples from MCMC chain.

# Process
1. Extract chain parameters using `extract_chain_parameters`
2. Loop over samples, reconstructing VARModel for each
3. Compute identification matrix Q using specified method
4. Apply `compute_func(model, Q, horizon)` to get result for each sample

# Arguments
- `chain::Chains`: MCMC chain from `estimate_bvar`
- `p::Int`: Number of VAR lags
- `n::Int`: Number of variables
- `compute_func::Function`: Function taking (model, Q, horizon) -> result

# Keyword Arguments
- `data::AbstractMatrix`: Original data (required for narrative method and residual computation)
- `method::Symbol`: Identification method (see `compute_Q` for full list)
- `horizon::Int`: IRF/computation horizon
- `check_func`: Sign restriction check function (for method=:sign or :narrative)
- `narrative_check`: Narrative restriction check function (for method=:narrative)
- `max_draws::Int`: Maximum draws for sign/narrative identification
- `transition_var`: Transition variable (for method=:smooth_transition)
- `regime_indicator`: Regime indicator (for method=:external_volatility)

# Returns
- `results::Vector{Any}`: Vector of results from `compute_func` for each sample
- `n_samples::Int`: Number of samples processed

# Example
```julia
# Compute IRF for each posterior sample
results, n_samples = process_posterior_samples(chain, p, n,
    (m, Q, h) -> compute_irf(m, Q, h);
    horizon=20, method=:cholesky
)
```
"""
function process_posterior_samples(chain::Chains, p::Int, n::Int, compute_func::Function;
    data::AbstractMatrix=Matrix{Float64}(undef, 0, 0),
    method::Symbol=:cholesky, horizon::Int=20,
    check_func=nothing, narrative_check=nothing, max_draws::Int=100,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing
)
    method == :narrative && isempty(data) &&
        throw(ArgumentError("Narrative method requires data"))

    samples = size(chain, 1)
    b_vecs, sigmas = extract_chain_parameters(chain)

    results = Vector{Any}(undef, samples)

    for s in 1:samples
        m = parameters_to_model(b_vecs[s, :], sigmas[s, :], p, n, data)
        Q = compute_Q(m, method, horizon, check_func, narrative_check;
                      max_draws=max_draws, transition_var=transition_var, regime_indicator=regime_indicator)
        results[s] = compute_func(m, Q, horizon)
    end

    results, samples
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
                                       samples::AbstractArray{T}, q_vec::AbstractVector) where {T<:AbstractFloat}
    other_dims = size(samples)[2:end]
    n_q = length(q_vec)

    @assert size(quantile_out) == (other_dims..., n_q) "quantile_out size mismatch"
    @assert size(mean_out) == other_dims "mean_out size mismatch"

    @inbounds for idx in CartesianIndices(other_dims)
        d = @view samples[:, idx]
        mean_out[idx] = mean(d)
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
                                                samples::AbstractArray{T}, q_vec::AbstractVector) where {T<:AbstractFloat}
    other_dims = size(samples)[2:end]
    n_q = length(q_vec)

    @assert size(quantile_out) == (other_dims..., n_q) "quantile_out size mismatch"
    @assert size(mean_out) == other_dims "mean_out size mismatch"

    # Flatten to linear indices for threading
    indices = collect(CartesianIndices(other_dims))

    Threads.@threads for idx in indices
        @inbounds begin
            d = @view samples[:, idx]
            mean_out[idx] = mean(d)
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
                                      threaded::Bool=false) where {T<:AbstractFloat}
    other_dims = size(samples)[2:end]
    n_q = length(q_vec)
    q_vec_T = T.(q_vec)

    quantile_out = zeros(T, other_dims..., n_q)
    mean_out = zeros(T, other_dims...)

    if threaded && prod(other_dims) > 1000
        compute_posterior_quantiles_threaded!(quantile_out, mean_out, samples, q_vec_T)
    else
        compute_posterior_quantiles!(quantile_out, mean_out, samples, q_vec_T)
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
