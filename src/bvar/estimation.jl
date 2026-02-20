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
Bayesian VAR estimation via conjugate Normal-Inverse-Wishart posterior.

Two samplers:
- `:direct` (default) — i.i.d. draws from the analytical posterior (no burnin needed)
- `:gibbs` — Two-block Gibbs sampler with burnin/thinning (for diagnostics/extensions)
"""

# =============================================================================
# Main Estimation
# =============================================================================

"""
    estimate_bvar(Y, p; n_draws=1000, sampler=:direct, burnin=0, thin=1,
                  prior=:normal, hyper=nothing) -> BVARPosterior

Estimate Bayesian VAR via conjugate Normal-Inverse-Wishart posterior.

# Model
    Y = X B + E,    E ~ MN(0, Σ, I_T)
    Prior: Σ ~ IW(ν₀, S₀),  vec(B)|Σ ~ N(b₀, Σ ⊗ V₀)

# Samplers
- `:direct` (default) — i.i.d. draws from analytical posterior. `burnin` and `thin` are ignored.
- `:gibbs` — Standard two-block Gibbs sampler. `burnin` defaults to 200 if not specified.

# Arguments
- `Y::AbstractMatrix`: T × n data matrix
- `p::Int`: Number of lags

# Keyword Arguments
- `n_draws::Int=1000`: Number of posterior draws to keep
- `sampler::Symbol=:direct`: Sampling algorithm (`:direct` or `:gibbs`)
- `burnin::Int=0`: Burnin period (only for `:gibbs`; defaults to 200 when sampler=:gibbs and burnin=0)
- `thin::Int=1`: Thinning interval (only for `:gibbs`)
- `prior::Symbol=:normal`: Prior type (`:normal` or `:minnesota`)
- `hyper::Union{Nothing,MinnesotaHyperparameters}=nothing`: Minnesota hyperparameters.
  When `prior=:minnesota` and `hyper=nothing`, tau is automatically optimized via
  marginal likelihood maximization (Giannone, Lenza & Primiceri 2015). Pass an explicit
  `MinnesotaHyperparameters(...)` to use fixed values instead.

# Returns
`BVARPosterior{T}` containing coefficient and covariance draws.

# Example
```julia
Y = randn(200, 3)
post = estimate_bvar(Y, 2; n_draws=1000)
post_mn = estimate_bvar(Y, 2; prior=:minnesota, n_draws=500)
```
"""
function estimate_bvar(Y::AbstractMatrix{T}, p::Int;
    n_draws::Int=1000, sampler::Symbol=:direct,
    burnin::Int=0, thin::Int=1,
    prior::Symbol=:normal,
    hyper::Union{Nothing,MinnesotaHyperparameters}=nothing,
    varnames::Union{Vector{String},Nothing}=nothing
) where {T<:AbstractFloat}

    _validate_data(Y, "Y")
    T_obs, n = size(Y)
    validate_var_inputs(T_obs, n, p)

    Y_eff, X = construct_var_matrices(Y, p)
    T_eff = size(Y_eff, 1)
    k = size(X, 2)  # 1 + n*p

    # Apply Minnesota prior augmentation if requested
    Y_data, X_data = if prior == :minnesota
        h = isnothing(hyper) ? optimize_hyperparameters(Y_eff, p) : hyper
        Y_d, X_d = gen_dummy_obs(Y, p, h)
        (vcat(Y_eff, Y_d), vcat(X, X_d))
    else
        (Y_eff, X)
    end

    T_data = size(Y_data, 1)

    # Set up prior hyperparameters
    # Prior: vec(B) | Σ ~ N(b₀, Σ ⊗ V₀), Σ ~ IW(ν₀, S₀)
    # Diffuse prior: V₀ = κ·I (large κ), ν₀ = n+2, S₀ = I
    κ = T(100.0)
    V0_inv = (one(T) / κ) * Matrix{T}(I, k, k)
    B0 = zeros(T, k, n)
    ν0 = n + 2
    S0 = Matrix{T}(I, n, n)

    # Posterior parameters (conjugate update)
    XtX = X_data' * X_data
    XtY = X_data' * Y_data
    V_post_inv = XtX + V0_inv
    V_post = robust_inv(V_post_inv)
    V_post = T(0.5) * (V_post + V_post')  # Ensure symmetry
    B_post = V_post * (XtY + V0_inv * B0)
    ν_post = ν0 + T_data
    S_post = S0 + Y_data' * Y_data + B0' * V0_inv * B0 - B_post' * V_post_inv * B_post
    S_post = T(0.5) * (S_post + S_post')  # Ensure symmetry

    vn = something(varnames, ["y$i" for i in 1:n])

    if sampler == :direct
        return _sample_direct(Y, p, n, k, n_draws, B_post, V_post, ν_post, S_post, prior, vn)
    elseif sampler == :gibbs
        eff_burnin = burnin == 0 ? 200 : burnin
        return _sample_gibbs(Y, p, n, k, n_draws, eff_burnin, thin,
                             Y_data, X_data, V0_inv, B0, ν0, S0, prior, vn)
    else
        throw(ArgumentError("Unknown sampler: $sampler. Use :direct or :gibbs"))
    end
end

@float_fallback estimate_bvar Y

# =============================================================================
# Direct (i.i.d.) Sampler
# =============================================================================

"""Draw i.i.d. samples from the conjugate Normal-Inverse-Wishart posterior."""
function _sample_direct(Y::Matrix{T}, p::Int, n::Int, k::Int, n_draws::Int,
                        B_post::Matrix{T}, V_post::Matrix{T},
                        ν_post::Int, S_post::Matrix{T}, prior::Symbol,
                        varnames::Vector{String}) where {T<:AbstractFloat}

    B_draws = Array{T,3}(undef, n_draws, k, n)
    Sigma_draws = Array{T,3}(undef, n_draws, n, n)

    # Cholesky of V_post for efficient sampling
    L_V = safe_cholesky(V_post)
    Z_buf = Matrix{T}(undef, k, n)

    for s in 1:n_draws
        # Step 1: Draw Σ ~ IW(ν_post, S_post)
        Sigma = _draw_inverse_wishart(ν_post, S_post)

        # Step 2: Draw B | Σ ~ MN(B_post, V_post, Σ)
        #   vec(B) ~ N(vec(B_post), Σ ⊗ V_post)
        #   Efficient: B = B_post + L_V * Z * L_Σ' where Z ~ N(0, I_{k×n})
        L_Sigma = safe_cholesky(Sigma)
        randn!(Z_buf)
        B = B_post + L_V * Z_buf * L_Sigma'

        B_draws[s, :, :] = B
        Sigma_draws[s, :, :] = Sigma
    end

    BVARPosterior{T}(B_draws, Sigma_draws, n_draws, p, n, Matrix{T}(Y), prior, :direct, varnames)
end

# =============================================================================
# Gibbs Sampler
# =============================================================================

"""Two-block Gibbs sampler for Normal-Inverse-Wishart posterior."""
function _sample_gibbs(Y::Matrix{T}, p::Int, n::Int, k::Int,
                       n_draws::Int, burnin::Int, thin::Int,
                       Y_data::Matrix{T}, X_data::Matrix{T},
                       V0_inv::Matrix{T}, B0::Matrix{T},
                       ν0::Int, S0::Matrix{T}, prior::Symbol,
                       varnames::Vector{String}) where {T<:AbstractFloat}

    B_draws = Array{T,3}(undef, n_draws, k, n)
    Sigma_draws = Array{T,3}(undef, n_draws, n, n)

    # Initialize from OLS
    B_curr = robust_inv(X_data' * X_data) * (X_data' * Y_data)
    resid = Y_data - X_data * B_curr
    Sigma_curr = (resid' * resid) / size(Y_data, 1)
    Sigma_curr = T(0.5) * (Sigma_curr + Sigma_curr')

    # Posterior for B | Σ, Y — depends only on data (XtX, V0_inv, B0), not Σ
    XtX = X_data' * X_data
    V_post_inv = XtX + V0_inv
    V_post = robust_inv(V_post_inv)
    V_post = T(0.5) * (V_post + V_post')
    B_post = V_post * (X_data' * Y_data + V0_inv * B0)
    L_V = safe_cholesky(V_post)

    # Pre-allocate MCMC workspace
    Z_buf = Matrix{T}(undef, k, n)

    total_iters = burnin + n_draws * thin
    draw_idx = 0

    for s in 1:total_iters
        # Block 1: Draw B | Σ, Y
        L_Sigma = safe_cholesky(Sigma_curr)
        randn!(Z_buf)
        B_curr = B_post + L_V * Z_buf * L_Sigma'

        # Block 2: Draw Σ | B, Y
        resid = Y_data - X_data * B_curr
        S_post = S0 + resid' * resid
        S_post = T(0.5) * (S_post + S_post')
        ν_post = ν0 + size(Y_data, 1)
        Sigma_curr = _draw_inverse_wishart(ν_post, S_post)

        # Store after burnin, with thinning
        if s > burnin && (s - burnin - 1) % thin == 0
            draw_idx += 1
            draw_idx > n_draws && break
            B_draws[draw_idx, :, :] = B_curr
            Sigma_draws[draw_idx, :, :] = Sigma_curr
        end
    end

    BVARPosterior{T}(B_draws, Sigma_draws, n_draws, p, n, Matrix{T}(Y), prior, :gibbs, varnames)
end

# =============================================================================
# Inverse-Wishart Sampler
# =============================================================================

"""
Draw from Inverse-Wishart(ν, S) distribution.
Uses the Bartlett decomposition: if X ~ W(ν, S⁻¹), then X⁻¹ ~ IW(ν, S).
"""
function _draw_inverse_wishart(ν::Int, S::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(S, 1)
    L_S_inv = safe_cholesky(robust_inv(S))

    # Bartlett decomposition of Wishart
    A = zeros(T, n, n)
    for i in 1:n
        A[i, i] = sqrt(rand(Chisq(T(ν - i + 1))))
        for j in 1:(i-1)
            A[i, j] = randn(T)
        end
    end

    # W = L * A * A' * L'  where L = chol(S⁻¹)
    LA = L_S_inv * A
    W = LA * LA'

    # Σ = W⁻¹
    Sigma = robust_inv(W)
    T(0.5) * (Sigma + Sigma')
end

# =============================================================================
# Chain Parameter Extraction (BVARPosterior interface)
# =============================================================================

"""
    extract_chain_parameters(post::BVARPosterior) -> (b_vecs, sigmas)

Extract coefficient vectors and covariance matrices from posterior draws.
Returns matrices compatible with downstream `parameters_to_model` calls.

- `b_vecs`: n_draws × (k*n) matrix of vectorized coefficients
- `sigmas`: n_draws × (n*n) matrix of vectorized covariance matrices
"""
function extract_chain_parameters(post::BVARPosterior{T}) where {T}
    n_draws = post.n_draws
    k, n = size(post.B_draws, 2), post.n

    b_vecs = Matrix{T}(undef, n_draws, k * n)
    sigmas = Matrix{T}(undef, n_draws, n * n)

    @inbounds for s in 1:n_draws
        b_vecs[s, :] = vec(post.B_draws[s, :, :])
        sigmas[s, :] = vec(post.Sigma_draws[s, :, :])
    end

    (b_vecs, sigmas)
end

"""Convert chain parameters to VARModel. Provide `data` for residual computation."""
function parameters_to_model(b_vec::AbstractVector{T}, sigma_vec::AbstractVector{T},
                             p::Int, n::Int, data::AbstractMatrix{T}=Matrix{T}(undef, 0, 0);
                             varnames::Vector{String}=["y$i" for i in 1:n]) where {T<:AbstractFloat}
    k = 1 + n * p
    B, Sigma = reshape(b_vec, k, n), reshape(sigma_vec, n, n)

    U = if !isempty(data) && size(data, 1) > p
        Y_eff, X = construct_var_matrices(data, p)
        Y_eff - X * B
    else
        Matrix{T}(undef, 0, n)
    end

    VARModel(isempty(data) ? zeros(T, 0, n) : data, p, B, U, Sigma, zero(T), zero(T), zero(T), varnames)
end

function parameters_to_model(b_vec, sigma_vec, p::Int, n::Int, data::AbstractMatrix=Matrix{Float64}(undef, 0, 0);
                             varnames::Vector{String}=["y$i" for i in 1:n])
    T = promote_type(eltype(b_vec), eltype(sigma_vec))
    parameters_to_model(Vector{T}(b_vec), Vector{T}(sigma_vec), p, n, Matrix{T}(data); varnames=varnames)
end

# =============================================================================
# Posterior Summary
# =============================================================================

"""VARModel with posterior mean parameters."""
function posterior_mean_model(post::BVARPosterior{T}; data::AbstractMatrix=Matrix{T}(undef, 0, 0)) where {T}
    use_data = isempty(data) ? post.data : Matrix{T}(data)
    b, s = extract_chain_parameters(post)
    parameters_to_model(vec(mean(b, dims=1)), vec(mean(s, dims=1)), post.p, post.n, use_data;
                        varnames=post.varnames)
end

"""VARModel with posterior median parameters."""
function posterior_median_model(post::BVARPosterior{T}; data::AbstractMatrix=Matrix{T}(undef, 0, 0)) where {T}
    use_data = isempty(data) ? post.data : Matrix{T}(data)
    b, s = extract_chain_parameters(post)
    parameters_to_model(vec(median(b, dims=1)), vec(median(s, dims=1)), post.p, post.n, use_data;
                        varnames=post.varnames)
end

# Deprecated wrappers (old Chains-based signatures)
function posterior_mean_model(post::BVARPosterior, p::Int, n::Int; data::AbstractMatrix=Matrix{Float64}(undef, 0, 0))
    posterior_mean_model(post; data=data)
end

function posterior_median_model(post::BVARPosterior, p::Int, n::Int; data::AbstractMatrix=Matrix{Float64}(undef, 0, 0))
    posterior_median_model(post; data=data)
end

# Accessor
varnames(post::BVARPosterior) = post.varnames

# =============================================================================
# BVAR Forecasting
# =============================================================================

"""
    forecast(post::BVARPosterior, h; reps=nothing, conf_level=0.95, point_estimate=:median) -> BVARForecast{T}

Forecast from Bayesian VAR using posterior draws.

For each posterior draw, iterates the VAR recursion forward `h` steps.
Returns posterior median forecast with credible intervals from the
distribution of forecast paths across draws.

# Arguments
- `post`: BVAR posterior
- `h`: Forecast horizon

# Keyword Arguments
- `reps`: Number of posterior draws to use (default: all)
- `conf_level`: Credible interval level (default: 0.95)
- `point_estimate`: `:median` (default) or `:mean` for central tendency

# Returns
`BVARForecast{T}` with posterior mean forecast and credible intervals.

# Example
```julia
post = estimate_bvar(Y, 4; n_draws=1000)
fc = forecast(post, 12)
```
"""
function forecast(post::BVARPosterior{T}, h::Int;
                  reps::Union{Nothing,Int}=nothing,
                  conf_level::Real=0.95,
                  point_estimate::Symbol=:median) where {T}
    h < 1 && throw(ArgumentError("Forecast horizon must be positive"))

    n, p = post.n, post.p
    n_use = reps === nothing ? post.n_draws : min(reps, post.n_draws)

    sim = Array{T,3}(undef, n_use, h, n)
    valid = 0

    for s in 1:n_use
        B_s = post.B_draws[s, :, :]   # k × n
        Sigma_s = post.Sigma_draws[s, :, :]  # n × n

        # Skip non-stationary draws
        A_list = extract_ar_coefficients(B_s, n, p)
        companion = zeros(T, n * p, n * p)
        for lag in 1:p
            companion[1:n, (lag-1)*n+1:lag*n] = A_list[lag]
        end
        if p > 1
            companion[n+1:end, 1:n*(p-1)] = Matrix{T}(I, n*(p-1), n*(p-1))
        end
        max_eig = maximum(abs.(eigvals(companion)))
        max_eig >= one(T) && continue

        valid += 1
        intercept = @view B_s[1, :]
        history = copy(post.data[(end - p + 1):end, :])

        # Draw shocks from posterior covariance
        L = safe_cholesky(Sigma_s)

        @inbounds for step in 1:h
            y_hat = copy(intercept)
            for lag in 1:p
                y_hat .+= A_list[lag] * @view(history[end - lag + 1, :])
            end
            y_hat .+= L * randn(T, n)
            sim[valid, step, :] = y_hat
            history = vcat(@view(history[2:end, :]), y_hat')
        end
    end

    valid == 0 && error("All posterior draws are non-stationary")
    valid < n_use ÷ 2 && @warn "$(n_use - valid)/$n_use posterior draws non-stationary, skipped"

    # Compute posterior central tendency and credible intervals from valid draws
    sim_valid = @view sim[1:valid, :, :]
    alpha_half = (1 - T(conf_level)) / 2
    _central = point_estimate == :median ? median : mean

    point = Matrix{T}(undef, h, n)
    ci_lower = Matrix{T}(undef, h, n)
    ci_upper = Matrix{T}(undef, h, n)

    @inbounds for hi in 1:h, j in 1:n
        d = @view sim_valid[:, hi, j]
        point[hi, j] = _central(d)
        ci_lower[hi, j] = quantile(d, alpha_half)
        ci_upper[hi, j] = quantile(d, 1 - alpha_half)
    end

    BVARForecast{T}(point, ci_lower, ci_upper, h, T(conf_level), point_estimate, post.varnames)
end
