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

# Bayesian Algorithm (`method=:bayesian`, BBE 2005 Section IV)
Gibbs sampler over `(B, Σ)`, the loadings `Λ`, and the factor path `F`, with the factors
initialized by PCA and drawn by Carter–Kohn FFBS. The measurement equation is BBE's

    X_t = Λ F_t + Λ_y Y_t^key + e_t

The direct loading `Λ_y` is the Bayesian counterpart of the two-step path's
double-counting removal: it stops the factors from absorbing the key variables' own
variation, which would otherwise leave `F` nearly collinear with `Y_key` and make the VAR
coefficient posterior explode (#528). Its draws are stored as `lambda_y_draws`
(rescaled to the VAR's original `Y_key` units) and `favar_panel_irf` adds the
direct channel `Λ_y · y_irf` to the panel mapping.

Identification uses the BBE normalization `Λ[anchor, :] = I_r`, `Λ_y[anchor, :] = 0`,
where `anchor` is the first `r` columns of `X` that are not key variables. **The ordering
of `X` therefore matters**: put `r` slow-moving series that are informative about, and not
collinear in, the factors first, and read factor `i` as the common component of anchor
series `i`. Estimation throws an `ArgumentError` if that loading block is numerically
singular, or if fewer than `r` non-key columns are available.

# Arguments
- `X`: Panel data matrix (T x N) — large cross-section of macroeconomic variables
- `Y_key`: Key observed variables (T x n_key) or column indices into X
- `r`: Number of latent factors to extract
- `p`: VAR lag order

# Keyword Arguments
- `method::Symbol=:two_step`: Estimation method (currently only `:two_step`)
- `panel_varnames::Union{Nothing, Vector{String}}=nothing`: Names for the N panel variables
- `n_draws::Int=5000`: Number of posterior draws (Bayesian Gibbs path)
- `burnin::Int=1000`: Burn-in draws (Bayesian Gibbs path)

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
    burnin::Int=1000,
    seed::Union{Integer,Nothing}=nothing,
    rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)

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
        result = _estimate_favar_bayesian(Matrix{T}(X), Matrix{T}(Y_key), r, p,
            n_draws, burnin, pvn, Y_key_indices, aug_varnames, rng)
        return _with_manifest(result, capture_manifest(; seed=seed,
            settings=Dict{String,Any}("n_draws" => n_draws, "burnin" => burnin,
                                      "method" => "bayesian")))
    end

    # --- Two-Step Estimation ---

    # Step 1: Extract factors via PCA
    fm = estimate_factors(X, r; standardize=true)
    F_raw = fm.factors           # T_obs x r
    Lambda = fm.loadings         # N x r

    # Step 2: Remove double-counting (slow-moving factors); B_y are the Y_key slopes per factor
    F_tilde, B_y = _remove_double_counting(F_raw, Y_key)
    Lambda_y = Lambda * B_y'      # N x n_key implied direct panel loadings on Y_key

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
        Lambda_y,
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
            result.lambda_y_draws,
            result.X_panel,
            result.panel_varnames,
            key_indices,
            result.n_factors,
            result.n_key,
            result.n,
            result.p,
            result.data,
            result.varnames;
            manifest=result.manifest
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
        result.Lambda_y,
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
    B_y = Matrix{T}(undef, n_key, r)          # per-factor Y_key slopes (intercept dropped)
    for j in 1:r
        f_j = @view F_raw[:, j]
        beta = ZtZ_inv * (Z' * f_j)
        F_tilde[:, j] = f_j - Z * beta
        B_y[:, j] = beta[2:end]
    end

    (F_tilde, B_y)
end

# =============================================================================
# Bayesian FAVAR Estimation (BBE 2005, Section IV)
# =============================================================================

# PSD-robust matrix square root L (L*L' = PSD projection of M) for the FFBS conditional-
# covariance draws. Conditional covariances are PSD in exact arithmetic but a poor Gibbs draw
# (near-singular Σ_ηη or a near-explosive VAR) can make them slightly indefinite; clipping the
# eigenvalues at zero keeps the sampler from throwing on a single bad sweep.
function _favar_state_chol(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    Ms = T(0.5) * (M + M')
    E = eigen(Symmetric(Ms))
    E.vectors * Diagonal(sqrt.(max.(E.values, zero(T))))
end

"""
    _favar_ffbs(X_std, Lambda, A_lags, Sigma_eta, sigma2_e, r, p) -> Matrix

Carter–Kohn forward-filter / backward-sample of the FAVAR factor path. Casts the factors in
companion state-space form — state `s_t = [F_t; …; F_{t-p+1}]`, transition from the F→F VAR
blocks `A_lags` with innovation covariance `Sigma_eta` (rank `r`) plus a known per-t `drift`
(the intercept and lagged-Y_key feedback of the augmented VAR), observation
`X_std[t] = Lambda·F_t + e_t`, `e_t ~ N(0, diag(sigma2_e))` — runs the Kalman filter, then
draws the path backward. Because the companion state noise is singular (only the top `r`
rows carry innovations), each backward step conditions on the top-`r` block `F_{t+1}` of the
next state only (Kim–Nelson). Returns the sampled factors (T_obs × r).
"""
function _favar_ffbs(X_std::AbstractMatrix{T}, Lambda::AbstractMatrix{T},
                     A_lags::Vector{Matrix{T}}, Sigma_eta::AbstractMatrix{T},
                     sigma2_e::AbstractVector{T}, r::Int, p::Int,
                     drift::AbstractMatrix{T},
                     rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    T_obs, N = size(X_std)
    sd = r * p

    # Companion state-space matrices
    Z = zeros(T, N, sd); Z[:, 1:r] = Lambda
    T_mat = zeros(T, sd, sd)
    for l in 1:p
        T_mat[1:r, ((l-1)*r+1):(l*r)] = A_lags[l]
    end
    p > 1 && (T_mat[(r+1):sd, 1:(sd-r)] = Matrix{T}(I, sd - r, sd - r))
    Q = zeros(T, sd, sd); Q[1:r, 1:r] = Sigma_eta

    # Forward Kalman filter (store filtered mean/cov)
    a_filt = zeros(T, T_obs, sd)
    P_filt = Array{T,3}(undef, T_obs, sd, sd)
    a_t = zeros(T, sd)
    P0 = _compute_unconditional_covariance(T_mat, Q, sd)
    P_t = all(isfinite, P0) ? Matrix{T}(P0) : Matrix{T}(I, sd, sd) * T(1e6)
    for t in 1:T_obs
        a_pred = T_mat * a_t
        @inbounds for i in 1:r
            a_pred[i] += drift[t, i]      # known intercept + Y_key feedback in the top-r rows
        end
        P_pred = T_mat * P_t * T_mat' + Q
        ZP = Z * P_pred                       # N × sd
        F_obs = ZP * Z'                       # N × N
        @inbounds for j in 1:N
            F_obs[j, j] += sigma2_e[j]
        end
        F_obs = T(0.5) * (F_obs + F_obs')
        F_inv = robust_inv(F_obs)
        K = P_pred * Z' * F_inv               # sd × N
        v = @view(X_std[t, :]) .- Z * a_pred
        a_t = a_pred + K * v
        P_t = P_pred - K * ZP
        P_t = T(0.5) * (P_t + P_t')
        a_filt[t, :] = a_t
        P_filt[t, :, :] = P_t
    end

    # Backward sampling: F_t | F_{t+1}, data_{1:t}, conditioning on the top-r block only
    F_samp = Matrix{T}(undef, T_obs, r)
    Fstar = Matrix{T}(T_mat[1:r, :])          # r × sd (top rows of the transition)
    CT = Matrix{T}(P_filt[T_obs, 1:r, 1:r])
    # Reused per-step standard-normal shock buffer; `randn!` draws from the same global stream
    # as `randn(T, r)`, so the sampled path is unchanged. (#210 box D)
    z_state = Vector{T}(undef, r)
    randn!(rng, z_state)
    F_samp[T_obs, :] = a_filt[T_obs, 1:r] + _favar_state_chol(CT) * z_state
    for t in (T_obs - 1):-1:1
        af = a_filt[t, :]
        Pf = Matrix{T}(P_filt[t, :, :])
        FPf = Fstar * Pf                      # r × sd
        Fbk = FPf * Fstar' + Sigma_eta        # r × r  (= top-r block of P_pred[t+1])
        Fbk = T(0.5) * (Fbk + Fbk')
        gain = Pf * Fstar' * robust_inv(Fbk)  # sd × r
        # predicted top-r mean of s_{t+1} includes the known drift at t+1
        m_full = af + gain * (F_samp[t+1, :] - (drift[t+1, :] + Fstar * af))
        C_full = Pf - gain * FPf              # sd × sd
        CF = Matrix{T}(C_full[1:r, 1:r])
        randn!(rng, z_state)
        F_samp[t, :] = m_full[1:r] + _favar_state_chol(CF) * z_state
    end
    F_samp
end

"""
    _estimate_favar_bayesian(X, Y_key, r, p, n_draws, burnin, pvn, Y_key_indices, aug_varnames) -> BayesianFAVAR

Bayesian FAVAR via Gibbs sampling (Bernanke, Boivin & Eliasz 2005, Section IV).

Measurement equation (BBE 2005, eq. 3), on standardized data:

    X_t = Λ F_t + Λ_y Y_t^key + e_t,    e_t ~ N(0, diag(σ²_e))

Algorithm:
1. Initialize F via PCA and (Λ, Λ_y) by regressing X on [F, Y_key], then rotate to the
   normalization below
2. Gibbs sampler iterates:
   - Draw (B, Σ) | F, Y_key from the flat-prior NIW VAR posterior
   - Draw (Λ, Λ_y) | F, X, Y_key equation-by-equation, holding the anchor rows fixed
   - Draw F | Λ, Λ_y, B, Σ, X, Y_key via Carter–Kohn FFBS
3. Store post-burnin draws

Identification (#528). The `anchor` rows — the first `r` panel columns that are not key
variables — are restricted to `Λ[anchor, :] = I_r` and `Λ_y[anchor, :] = 0`. The identity
block fixes the r² free elements of the (F, Λ) rotation/scale indeterminacy, which the
likelihood does not pin down; the zero block is BBE's "slow-moving" restriction. The `Λ_y`
term is what keeps `F` from absorbing the key variables' own variation: without it `F`
comes out nearly collinear with `Y_key` (correlation ≈ 0.9 on the docs panel), the VAR
design matrix on `[F, Y_key]` is near-singular, and the coefficient posterior explodes.
`Λ_y` draws are stored on `BayesianFAVAR.lambda_y_draws`, rescaled per draw to the
VAR's original `Y_key` units so panel mappings can apply them to VAR responses
directly.
"""
function _estimate_favar_bayesian(X::Matrix{T}, Y_key::Matrix{T}, r::Int, p::Int,
    n_draws::Int, burnin::Int, pvn::Vector{String},
    Y_key_indices::Vector{Int}, aug_varnames::Vector{String},
    rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}

    T_obs, N = size(X)
    n_key = size(Y_key, 2)
    n_var = r + n_key  # total VAR dimension

    eff_burnin = burnin == 0 ? 200 : burnin
    total_iters = n_draws + eff_burnin

    # --- Anchor rows for the BBE identification (see below) ---
    # The anchors must be informational series, never the key variables themselves: pinning
    # X_a = F_a would make a factor a noisy copy of a variable the VAR already contains.
    anchor = setdiff(1:N, Y_key_indices)
    length(anchor) >= r || throw(ArgumentError(
        "Bayesian FAVAR needs at least r=$r panel columns that are not key variables to " *
        "anchor the factors, but X has only $(length(anchor)) (N=$N, of which " *
        "$(length(Y_key_indices)) are key). Supply a larger panel or reduce r."))
    anchor = anchor[1:r]
    anchor_pos = zeros(Int, N)          # panel row -> anchored factor index (0 = free row)
    for (i, a) in enumerate(anchor)
        anchor_pos[a] = i
    end

    # --- Standardize the panel and the key variables ---
    # X_std feeds both the measurement equation and the idiosyncratic variances; Y_std enters
    # the measurement equation only (the VAR block keeps Y_key on its original scale).
    X_std = Matrix{T}(undef, T_obs, N)
    X_means = vec(mean(X, dims=1))
    X_stds = vec(std(X, dims=1))
    for j in 1:N
        s = max(X_stds[j], T(1e-10))
        @inbounds for t in 1:T_obs
            X_std[t, j] = (X[t, j] - X_means[j]) / s
        end
    end
    Y_std = Matrix{T}(undef, T_obs, n_key)
    Y_means = vec(mean(Y_key, dims=1))
    Y_stds = vec(std(Y_key, dims=1))
    for j in 1:n_key
        s = max(Y_stds[j], T(1e-10))
        @inbounds for t in 1:T_obs
            Y_std[t, j] = (Y_key[t, j] - Y_means[j]) / s
        end
    end

    # --- Step 1: Initialize factors via PCA ---
    fm = estimate_factors(X, r; standardize=true)
    F_curr = Matrix{T}(fm.factors)       # T_obs x r

    # --- Step 2: Initialize (Λ, Λ_y) from the joint regression X = F Λ' + Y Λ_y' + e ---
    # The measurement equation carries a direct Y_key loading Λ_y, as in BBE (2005) eq. (3).
    # Without it the factors have to absorb the key variables' own variation, so F comes out
    # nearly collinear with Y_key; the VAR on [F, Y_key] is then near-singular and its
    # coefficient posterior explodes (#528). Λ_y is the Bayesian counterpart of the two-step
    # path's `_remove_double_counting`; it identifies F, and its draws are stored
    # (rescaled to Y-units) for the panel mapping's direct channel.
    W_init = hcat(F_curr, Y_std)
    coef_init = Matrix{T}(robust_inv(W_init' * W_init)) * (W_init' * X_std)  # (r+n_key) x N
    Lambda_curr = Matrix{T}(permutedims(coef_init[1:r, :]))                  # N x r
    Lambda_y = Matrix{T}(permutedims(coef_init[(r+1):(r+n_key), :]))         # N x n_key

    # --- Step 3: BBE (2005, §IV) normalization: Λ[anchor, :] = I_r and Λ_y[anchor, :] = 0 ---
    # The likelihood pins down only the column space of Λ: any invertible r × r rotation G
    # leaves F·Λ' = (F G')·(Λ G⁻¹)' unchanged, so the sampler's factor scale and rotation
    # would wander from sweep to sweep. Anchoring r informational series to unit loadings on
    # one factor each removes exactly the r² free elements of that indeterminacy. Λ_y = 0 on
    # the same rows is BBE's "slow-moving" restriction: the anchor series respond to the key
    # variables only through the factors.
    G = Matrix{T}(Lambda_curr[anchor, :])
    cond_G = cond(G)
    isfinite(cond_G) && cond_G < T(1e10) || throw(ArgumentError(
        "Panel columns $anchor do not identify the $r factors: their loading block is " *
        "numerically singular (cond = $cond_G). The Bayesian FAVAR anchors the factor scale " *
        "on the first $r non-key columns of X (Bernanke, Boivin & Eliasz 2005), so those " *
        "series must load on r distinct factors. Reorder the columns of X so that its " *
        "leading non-key series are informative about, and not collinear in, the factors, " *
        "or reduce r."))
    F_curr = F_curr * G'
    Lambda_curr = Lambda_curr / G
    # Impose the restrictions exactly (the solve is accurate only to rounding).
    Lambda_curr[anchor, :] = Matrix{T}(I, r, r)
    Lambda_y[anchor, :] .= zero(T)

    # Initialize idiosyncratic variances from the normalized measurement residuals
    resid_X = X_std - F_curr * Lambda_curr' - Y_std * Lambda_y'
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
    # Direct key-variable loadings, stored per draw RESCALED from standardized-Y to the
    # VAR's original Y units (divide column k by sd(Y_k)), so favar_panel_irf can apply
    # them straight to the augmented VAR's Y_key responses (#528 follow-up).
    lambda_y_draws = Array{T,3}(undef, n_draws, N, n_key)
    Y_sds_c = max.(Y_stds, T(1e-10))

    # Pre-allocate workspace
    Y_aug = Matrix{T}(undef, T_obs, n_var)
    # Reused per-sweep standard-normal shock buffers; `randn!` draws from the same global stream
    # as the previous `randn(T, …)` calls, so every draw is unchanged. (#210 box D)
    Z_Bdraw = Matrix{T}(undef, k, n_var)   # Block 1: (B | Σ) matrix-normal shock
    z_lambda = Vector{T}(undef, r + n_key) # Block 2: per-equation (λ, λ_y) loading shock
    W_lam = Matrix{T}(undef, T_obs, r + n_key)  # Block 2 design [F, Y_std]
    W_lam[:, (r+1):(r+n_key)] = Y_std           # the Y_key block never changes
    X_obs = Matrix{T}(undef, T_obs, N)      # Block 3 observation net of the Λ_y term
    draw_idx = 0

    for iter in 1:total_iters
        _suppress_warnings() do
            # === Block 1: Draw (B, Σ) | F, Y_key ===
            # Build augmented system [F, Y_key]
            Y_aug[:, 1:r] = F_curr
            Y_aug[:, (r+1):n_var] = Y_key

            # Flat-prior NIW posterior. Deliberately NOT Minnesota-augmented (#528):
            # `gen_dummy_obs` would recompute its scale moments from the *current
            # draw* of Y_aug every sweep — a state-dependent prior with no fixed
            # invariant distribution — and the shrinking dummies would track the
            # factor scale rather than anchor it. The explosive draws that motivated
            # the dummies were a factor-scale/rotation identification problem; it is
            # fixed by the Λ[1:r, 1:r] = I normalization imposed at initialization and
            # held fixed in Block 2, which leaves this prior free to stay flat.
            Y_data, X_data = construct_var_matrices(Y_aug, p)
            XtX = X_data' * X_data
            XtX_inv = Matrix{T}(robust_inv(XtX))
            XtX_inv = T(0.5) * (XtX_inv + XtX_inv')
            B_hat = XtX_inv * (X_data' * Y_data)
            U_aug = Y_data - X_data * B_hat
            S_post = Matrix{T}(U_aug' * U_aug)
            S_post = T(0.5) * (S_post + S_post')
            T_data = size(Y_data, 1)
            # Degrees of freedom after integrating B: T_data - k (augmented sample length).
            nu_sigma = max(T_data - k, n_var + 2)
            Sigma_curr = _draw_inverse_wishart(nu_sigma, S_post, rng)
            randn!(rng, Z_Bdraw)
            B_curr = B_hat + safe_cholesky(XtX_inv) * Z_Bdraw * safe_cholesky(Sigma_curr)'

            # === Block 2: Draw (Λ, Λ_y) | F, X, Y_key ===
            # Equation-by-equation on the BBE measurement equation
            #   X_i = F λ_i + Y_std λ_y,i + e_i,
            # posterior (λ_i, λ_y,i) ~ N(β_hat_i, σ²_i (W'W)^{-1}) with W = [F, Y_std].
            # The anchor rows are NOT drawn: they stay pinned at λ = e_a, λ_y = 0, which is
            # what fixes the factor scale and rotation (#528). Their idiosyncratic variances
            # are still updated, from the residual X_a − F_a.
            W_lam[:, 1:r] = F_curr
            WtW_inv = Matrix{T}(robust_inv(W_lam' * W_lam))

            # Ensure positive definiteness
            WtW_inv = T(0.5) * (WtW_inv + WtW_inv')
            for i in 1:(r + n_key)
                WtW_inv[i, i] = max(WtW_inv[i, i], T(1e-10))
            end

            L_WtW_inv = safe_cholesky(WtW_inv)

            for j in 1:N
                x_j = @view X_std[:, j]

                a = anchor_pos[j]
                if a > 0
                    # Anchored row: λ_j = e_a and λ_y,j = 0, so the residual is x_j − F[:, a].
                    resid_j = x_j - @view(F_curr[:, a])
                    sigma2_e[j] = max(dot(resid_j, resid_j) / T(T_obs), T(1e-10))
                    continue
                end

                beta_hat = WtW_inv * (W_lam' * x_j)

                # Update idiosyncratic variance estimate
                resid_j = x_j - W_lam * beta_hat
                sigma2_j = max(dot(resid_j, resid_j) / T(T_obs), T(1e-10))
                sigma2_e[j] = sigma2_j

                # Draw (λ_i, λ_y,i) from posterior
                randn!(rng, z_lambda)
                lam_j = beta_hat + sqrt(sigma2_j) * L_WtW_inv * z_lambda
                Lambda_curr[j, :] = @view lam_j[1:r]
                Lambda_y[j, :] = @view lam_j[(r+1):(r+n_key)]
            end

            # === Block 3: Draw F | Λ, B, Σ, X_std via Carter–Kohn FFBS ===
            # Extract from B_curr (rows: 1 = intercept, then n_var per lag; columns = equations):
            #  - the F→F companion blocks A_lags[l][a,b] = coef of F_{t-l}[b] in equation a;
            #  - the F←Y_key feedback blocks + the intercept, which enter the state equation as a
            #    KNOWN deterministic drift (Y_key is observed) — omitting it would make the factor
            #    draw inconsistent with the augmented VAR just drawn in Block 1 and bias F.
            A_lags = Vector{Matrix{T}}(undef, p)
            AFY = Vector{Matrix{T}}(undef, p)
            for l in 1:p
                r0 = 2 + (l - 1) * n_var
                A_lags[l] = permutedims(Matrix{T}(B_curr[r0:(r0 + r - 1), 1:r]))            # r × r
                AFY[l]    = permutedims(Matrix{T}(B_curr[(r0 + r):(r0 + n_var - 1), 1:r]))   # r × n_key
            end
            c_F = Vector{T}(B_curr[1, 1:r])
            drift = Matrix{T}(undef, T_obs, r)
            for t in 1:T_obs
                u = copy(c_F)
                for l in 1:p
                    t - l >= 1 && (u .+= AFY[l] * @view(Y_key[t - l, :]))
                end
                drift[t, :] = u
            end
            Sigma_eta = Matrix{T}(Sigma_curr[1:r, 1:r])
            Sigma_eta = T(0.5) * (Sigma_eta + Sigma_eta')
            # Net the known Λ_y term out of the observation, leaving X_obs = Λ F_t + e_t.
            mul!(X_obs, Y_std, Lambda_y')
            X_obs .= X_std .- X_obs
            F_curr = _favar_ffbs(X_obs, Lambda_curr, A_lags, Sigma_eta, sigma2_e, r, p, drift, rng)

            # === Store draws after burnin ===
            if iter > eff_burnin
                draw_idx += 1
                B_draws[draw_idx, :, :] = B_curr
                Sigma_draws[draw_idx, :, :] = Sigma_curr
                factor_draws[draw_idx, :, :] = F_curr
                loadings_draws[draw_idx, :, :] = Lambda_curr
                lambda_y_draws[draw_idx, :, :] = Lambda_y ./ Y_sds_c'
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
        lambda_y_draws,
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
