# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Two-step FAVAR estimation following Bernanke, Boivin & Eliasz (2005).

Step 1: Extract r factors from the large panel X via PCA.
Step 2: Remove double-counting by regressing factors on Y_key, keeping residuals
         as "slow-moving" factors F_tilde.
Step 3: Estimate VAR(p) on [F_tilde, Y_key].

References:
- Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the Effects of
  Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach.
  Quarterly Journal of Economics, 120(1), 387-422.
"""

# =============================================================================
# Two-Step FAVAR Estimation
# =============================================================================

"""
    estimate_favar(X, Y_key, r, p; method=:two_step, panel_varnames=nothing) -> FAVARModel
    estimate_favar(X, key_indices::Vector{Int}, r, p; ...) -> FAVARModel

Estimate a Factor-Augmented VAR (Bernanke, Boivin & Eliasz, 2005).

# Two-Step Algorithm (BBE 2005)
1. Extract `r` factors from panel `X` via PCA (`estimate_factors`)
2. Remove double-counting: regress each factor on `Y_key`, use residuals
   as slow-moving factors `F_tilde`
3. Estimate VAR(p) on augmented system `[F_tilde, Y_key]`

# Arguments
- `X`: Panel data matrix (T x N) — large cross-section of macroeconomic variables
- `Y_key`: Key observed variables (T x n_key) or column indices into X
- `r`: Number of latent factors to extract
- `p`: VAR lag order

# Keyword Arguments
- `method::Symbol=:two_step`: Estimation method (currently only `:two_step`)
- `panel_varnames::Union{Nothing, Vector{String}}=nothing`: Names for the N panel variables
- `n_draws::Int=5000`: Number of posterior draws (for Bayesian, reserved for future use)
- `burnin::Int=1000`: Burn-in draws (for Bayesian, reserved for future use)

# Returns
`FAVARModel{T}` with estimated VAR on [F_tilde, Y_key], factor loadings,
and panel variable metadata for panel-wide IRF mapping.

# Example
```julia
X = randn(200, 50)    # 200 obs, 50 variables
Y_key = X[:, [1, 5]]  # 2 key variables
favar = estimate_favar(X, Y_key, 3, 2)

# Or using column indices:
favar = estimate_favar(X, [1, 5], 3, 2)

# IRFs (via to_var delegation):
irf_result = irf(favar, 20)

# Panel-wide IRFs (all 50 variables):
panel_irfs = favar_panel_irf(favar, irf_result)
```
"""
function estimate_favar(X::AbstractMatrix{T}, Y_key::AbstractMatrix{T}, r::Int, p::Int;
    method::Symbol=:two_step,
    panel_varnames::Union{Nothing, Vector{String}}=nothing,
    n_draws::Int=5000,
    burnin::Int=1000) where {T<:AbstractFloat}

    # --- Validate inputs ---
    _validate_data(X, "X")
    _validate_data(Y_key, "Y_key")

    T_obs, N = size(X)
    T_key, n_key = size(Y_key)

    T_obs == T_key || throw(ArgumentError(
        "X has $T_obs rows but Y_key has $T_key rows; they must match"))
    r >= 1 || throw(ArgumentError("Number of factors r must be >= 1"))
    r <= N || throw(ArgumentError(
        "Number of factors r=$r exceeds number of panel variables N=$N"))
    p >= 1 || throw(ArgumentError("VAR lag order p must be >= 1"))

    n_aug = r + n_key
    T_obs > p + n_aug || throw(ArgumentError(
        "Not enough observations (T=$T_obs) for p=$p lags with $n_aug variables"))

    method in (:two_step, :bayesian) || throw(ArgumentError(
        "method must be :two_step or :bayesian, got :$method"))

    # --- Determine panel variable names ---
    pvn = something(panel_varnames, ["X$i" for i in 1:N])
    length(pvn) == N || throw(ArgumentError(
        "panel_varnames has $(length(pvn)) entries but X has $N columns"))

    # --- Try to identify Y_key columns in X ---
    Y_key_indices = _find_key_indices(X, Y_key)

    # --- Build augmented variable names ---
    factor_names = ["F$i" for i in 1:r]
    key_names = if !isempty(Y_key_indices) && all(idx -> 1 <= idx <= N, Y_key_indices)
        [pvn[idx] for idx in Y_key_indices]
    else
        ["Y$i" for i in 1:n_key]
    end
    aug_varnames = vcat(factor_names, key_names)

    if method == :bayesian
        return _estimate_favar_bayesian(Matrix{T}(X), Matrix{T}(Y_key), r, p,
            n_draws, burnin, pvn, Y_key_indices, aug_varnames)
    end

    # --- Two-Step Estimation ---

    # Step 1: Extract factors via PCA
    fm = estimate_factors(X, r; standardize=true)
    F_raw = fm.factors           # T_obs x r
    Lambda = fm.loadings         # N x r

    # Step 2: Remove double-counting (slow-moving factors)
    F_tilde = _remove_double_counting(F_raw, Y_key)

    # Step 3: Estimate VAR on [F_tilde, Y_key]
    Y_aug = hcat(F_tilde, Y_key)

    var_model = estimate_var(Y_aug, p; check_stability=false, varnames=aug_varnames)

    # Compute log-likelihood
    T_eff = effective_nobs(var_model)
    n_var = n_aug
    Sigma_ml = (var_model.U'var_model.U) / T_eff
    log_det = logdet_safe(Sigma_ml)
    loglik = -T_eff / 2 * (n_var * log(T(2pi)) + log_det + n_var)

    FAVARModel{T}(
        Y_aug,
        p,
        var_model.B,
        var_model.U,
        var_model.Sigma,
        aug_varnames,
        Matrix{T}(X),
        pvn,
        Y_key_indices,
        r,
        n_key,
        F_tilde,
        Lambda,
        fm,
        var_model.aic,
        var_model.bic,
        loglik
    )
end

# =============================================================================
# Column Index Dispatch
# =============================================================================

"""
    estimate_favar(X, key_indices::Vector{Int}, r, p; kwargs...) -> FAVARModel or BayesianFAVAR

Estimate FAVAR where key variables are specified as column indices of X.
"""
function estimate_favar(X::AbstractMatrix{T}, key_indices::Vector{Int}, r::Int, p::Int;
    kwargs...) where {T<:AbstractFloat}

    N = size(X, 2)
    for idx in key_indices
        1 <= idx <= N || throw(ArgumentError(
            "Key variable index $idx is out of range [1, $N]"))
    end

    Y_key = X[:, key_indices]
    # Pass panel_varnames if given in kwargs, and store key_indices
    result = estimate_favar(X, Y_key, r, p; kwargs...)

    # For BayesianFAVAR, Y_key_indices are already set correctly from the main function
    if result isa BayesianFAVAR
        # Override Y_key_indices with the explicit indices
        return BayesianFAVAR{T}(
            result.B_draws,
            result.Sigma_draws,
            result.factor_draws,
            result.loadings_draws,
            result.X_panel,
            result.panel_varnames,
            key_indices,
            result.n_factors,
            result.n_key,
            result.n,
            result.p,
            result.data,
            result.varnames
        )
    end

    # Override Y_key_indices with the explicit indices for FAVARModel
    FAVARModel{T}(
        result.Y,
        result.p,
        result.B,
        result.U,
        result.Sigma,
        result.varnames,
        result.X_panel,
        result.panel_varnames,
        key_indices,
        result.n_factors,
        result.n_key,
        result.factors,
        result.loadings,
        result.factor_model,
        result.aic,
        result.bic,
        result.loglik
    )
end

# Manual float fallback for two AbstractMatrix arguments
# (@float_fallback only handles a single AbstractMatrix argument)
function estimate_favar(X::AbstractMatrix, Y_key::AbstractMatrix, r::Int, p::Int; kwargs...)
    estimate_favar(Float64.(X), Float64.(Y_key), r, p; kwargs...)
end

# =============================================================================
# Internal Helpers
# =============================================================================

"""
    _find_key_indices(X, Y_key) -> Vector{Int}

Try to identify which columns of X correspond to Y_key by exact column matching.
Returns empty vector if no match is found.
"""
function _find_key_indices(X::AbstractMatrix{T}, Y_key::AbstractMatrix{T}) where {T}
    N = size(X, 2)
    n_key = size(Y_key, 2)
    indices = Int[]

    for j in 1:n_key
        y_col = @view Y_key[:, j]
        found = false
        for i in 1:N
            x_col = @view X[:, i]
            if x_col == y_col
                push!(indices, i)
                found = true
                break
            end
        end
        if !found
            push!(indices, 0)
        end
    end

    # Only return valid indices if all were found
    all(idx -> idx > 0, indices) ? indices : Int[]
end

"""
    _remove_double_counting(F_raw, Y_key) -> Matrix{T}

Remove the component of F_raw that is spanned by Y_key (BBE 2005 Step 2).

For each factor column, regress it on Y_key via OLS and keep the residual.
This yields "slow-moving" factors that are orthogonal to the key variables.
"""
function _remove_double_counting(F_raw::Matrix{T}, Y_key::AbstractMatrix{T}) where {T}
    T_obs, r = size(F_raw)
    n_key = size(Y_key, 2)

    # Design matrix: [1, Y_key]
    Z = hcat(ones(T, T_obs), Y_key)
    ZtZ_inv = Matrix{T}(robust_inv(Z'Z))

    F_tilde = Matrix{T}(undef, T_obs, r)
    for j in 1:r
        f_j = @view F_raw[:, j]
        beta = ZtZ_inv * (Z' * f_j)
        F_tilde[:, j] = f_j - Z * beta
    end

    F_tilde
end

# =============================================================================
# Bayesian FAVAR Estimation (BBE 2005, Section IV)
# =============================================================================

"""
    _estimate_favar_bayesian(X, Y_key, r, p, n_draws, burnin, pvn, Y_key_indices, aug_varnames) -> BayesianFAVAR

Bayesian FAVAR via Gibbs sampling (Bernanke, Boivin & Eliasz 2005, Section IV).

Algorithm:
1. Initialize factors via PCA
2. Gibbs sampler iterates:
   - Draw (B, Σ) | F, Y_key from the VAR posterior
   - Draw Λ | F, X equation-by-equation
   - Draw F | Λ, B, Σ, X, Y_key via regression posterior
3. Store post-burnin draws
"""
function _estimate_favar_bayesian(X::Matrix{T}, Y_key::Matrix{T}, r::Int, p::Int,
    n_draws::Int, burnin::Int, pvn::Vector{String},
    Y_key_indices::Vector{Int}, aug_varnames::Vector{String}) where {T<:AbstractFloat}

    T_obs, N = size(X)
    n_key = size(Y_key, 2)
    n_var = r + n_key  # total VAR dimension

    eff_burnin = burnin == 0 ? 200 : burnin
    total_iters = n_draws + eff_burnin

    # --- Step 1: Initialize factors via PCA ---
    fm = estimate_factors(X, r; standardize=true)
    F_curr = Matrix{T}(fm.factors)       # T_obs x r
    Lambda_curr = Matrix{T}(fm.loadings) # N x r

    # Standardize X for idiosyncratic variance estimation
    X_std = Matrix{T}(undef, T_obs, N)
    X_means = vec(mean(X, dims=1))
    X_stds = vec(std(X, dims=1))
    for j in 1:N
        s = max(X_stds[j], T(1e-10))
        @inbounds for t in 1:T_obs
            X_std[t, j] = (X[t, j] - X_means[j]) / s
        end
    end

    # Initialize idiosyncratic variances from PCA residuals
    resid_X = X_std - F_curr * Lambda_curr'
    sigma2_e = Vector{T}(undef, N)
    for j in 1:N
        sigma2_e[j] = max(var(@view(resid_X[:, j])), T(1e-10))
    end

    # Pre-allocate storage for posterior draws
    k = 1 + n_var * p  # number of coefficients per equation in VAR
    B_draws = Array{T,3}(undef, n_draws, k, n_var)
    Sigma_draws = Array{T,3}(undef, n_draws, n_var, n_var)
    factor_draws = Array{T,3}(undef, n_draws, T_obs, r)
    loadings_draws = Array{T,3}(undef, n_draws, N, r)

    # Pre-allocate workspace
    Y_aug = Matrix{T}(undef, T_obs, n_var)
    draw_idx = 0

    for iter in 1:total_iters
        _suppress_warnings() do
            # === Block 1: Draw (B, Σ) | F, Y_key ===
            # Build augmented system [F, Y_key]
            Y_aug[:, 1:r] = F_curr
            Y_aug[:, (r+1):n_var] = Y_key

            # Estimate VAR on augmented system
            var_model = estimate_var(Y_aug, p; check_stability=false, varnames=aug_varnames)

            B_curr = Matrix{T}(var_model.B)
            Sigma_curr = Matrix{T}(var_model.Sigma)

            # Add small jitter to ensure positive definiteness
            Sigma_curr = T(0.5) * (Sigma_curr + Sigma_curr')
            for i in 1:n_var
                Sigma_curr[i, i] = max(Sigma_curr[i, i], T(1e-10))
            end

            # === Block 2: Draw Λ | F, X ===
            # Equation-by-equation: X_i = F * λ_i + e_i
            # Posterior: λ_i ~ N(β_hat_i, σ²_i * (F'F)^{-1})
            FtF = F_curr' * F_curr
            FtF_inv = Matrix{T}(robust_inv(FtF))

            # Ensure positive definiteness
            FtF_inv = T(0.5) * (FtF_inv + FtF_inv')
            for i in 1:r
                FtF_inv[i, i] = max(FtF_inv[i, i], T(1e-10))
            end

            L_FtF_inv = safe_cholesky(FtF_inv)

            for j in 1:N
                x_j = @view X_std[:, j]
                beta_hat = FtF_inv * (F_curr' * x_j)

                # Update idiosyncratic variance estimate
                resid_j = x_j - F_curr * beta_hat
                sigma2_j = max(dot(resid_j, resid_j) / T(T_obs), T(1e-10))
                sigma2_e[j] = sigma2_j

                # Draw λ_i from posterior
                Lambda_curr[j, :] = beta_hat + sqrt(sigma2_j) * L_FtF_inv * randn(T, r)
            end

            # === Block 3: Draw F | Λ, B, Σ, X, Y_key ===
            # Simplified regression approach:
            # Observation: X_std[t,:] = Λ * F[t,:] + e[t,:]
            # Posterior for F[t,:]:
            #   Precision = Λ' Σ_e^{-1} Λ + I
            #   Mean = Precision^{-1} * Λ' Σ_e^{-1} X_std[t,:]
            Sigma_e_inv_diag = Vector{T}(undef, N)
            for j in 1:N
                Sigma_e_inv_diag[j] = one(T) / max(sigma2_e[j], T(1e-10))
            end

            # Λ' Σ_e^{-1} Λ (r x r)
            LtSinvL = zeros(T, r, r)
            for j in 1:N
                w = Sigma_e_inv_diag[j]
                lam_j = @view Lambda_curr[j, :]
                for a in 1:r, b in 1:r
                    LtSinvL[a, b] += w * lam_j[a] * lam_j[b]
                end
            end

            # Posterior precision and covariance
            F_precision = LtSinvL + Matrix{T}(I, r, r)
            F_precision = T(0.5) * (F_precision + F_precision')
            F_cov = Matrix{T}(robust_inv(F_precision))
            F_cov = T(0.5) * (F_cov + F_cov')
            for i in 1:r
                F_cov[i, i] = max(F_cov[i, i], T(1e-10))
            end
            L_F_cov = safe_cholesky(F_cov)

            # Λ' Σ_e^{-1} (r x N)
            LtSinv = zeros(T, r, N)
            for j in 1:N
                w = Sigma_e_inv_diag[j]
                for a in 1:r
                    LtSinv[a, j] = w * Lambda_curr[j, a]
                end
            end

            # Draw F[t,:] for each t
            for t in 1:T_obs
                x_t = @view X_std[t, :]
                f_mean = F_cov * (LtSinv * x_t)
                F_curr[t, :] = f_mean + L_F_cov * randn(T, r)
            end

            # === Store draws after burnin ===
            if iter > eff_burnin
                draw_idx += 1
                B_draws[draw_idx, :, :] = B_curr
                Sigma_draws[draw_idx, :, :] = Sigma_curr
                factor_draws[draw_idx, :, :] = F_curr
                loadings_draws[draw_idx, :, :] = Lambda_curr
            end
        end  # _suppress_warnings
    end  # iter

    # Build augmented data from final factor draws (use posterior mean factors)
    F_final = dropdims(mean(factor_draws, dims=1), dims=1)
    data_aug = hcat(F_final, Y_key)

    BayesianFAVAR{T}(
        B_draws,
        Sigma_draws,
        factor_draws,
        loadings_draws,
        Matrix{T}(X),
        pvn,
        Y_key_indices,
        r,
        n_key,
        n_var,
        p,
        data_aug,
        aug_varnames
    )
end
