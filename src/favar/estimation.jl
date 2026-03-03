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

    method == :two_step || throw(ArgumentError(
        "Only :two_step method is currently supported, got :$method"))

    # --- Determine panel variable names ---
    pvn = something(panel_varnames, ["X$i" for i in 1:N])
    length(pvn) == N || throw(ArgumentError(
        "panel_varnames has $(length(pvn)) entries but X has $N columns"))

    # --- Try to identify Y_key columns in X ---
    Y_key_indices = _find_key_indices(X, Y_key)

    # --- Step 1: Extract factors via PCA ---
    fm = estimate_factors(X, r; standardize=true)
    F_raw = fm.factors           # T_obs x r
    Lambda = fm.loadings         # N x r

    # --- Step 2: Remove double-counting (slow-moving factors) ---
    # Regress each factor on Y_key, keep residuals as F_tilde
    F_tilde = _remove_double_counting(F_raw, Y_key)

    # --- Step 3: Estimate VAR on [F_tilde, Y_key] ---
    Y_aug = hcat(F_tilde, Y_key)

    # Build variable names for the augmented VAR
    factor_names = ["F$i" for i in 1:r]
    key_names = if !isempty(Y_key_indices) && all(idx -> 1 <= idx <= N, Y_key_indices)
        [pvn[idx] for idx in Y_key_indices]
    else
        ["Y$i" for i in 1:n_key]
    end
    aug_varnames = vcat(factor_names, key_names)

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
    estimate_favar(X, key_indices::Vector{Int}, r, p; kwargs...) -> FAVARModel

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

    # Override Y_key_indices with the explicit indices
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
