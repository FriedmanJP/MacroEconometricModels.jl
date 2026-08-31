# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Heteroskedasticity-based SVAR identification (Lewis 2025, Section 3).

Exploits time-varying second moments (changes in the volatility regime) to identify
structural shocks without distributional assumptions. Methods: Markov-switching,
GARCH, smooth transition, external volatility instruments.

References:
- Lewis, D. J. (2025). "Identification based on higher moments in macroeconometrics."
- Lewis, D. J. (2021). "Identifying shocks via time-varying volatility."
- Sentana, E. & Fiorentini, G. (2001). "Identification, estimation and testing of conditionally heteroskedastic factor models."
- Rigobon, R. (2003). "Identification through heteroskedasticity."
- Lanne, M. & Lütkepohl, H. (2008). "Identifying monetary policy shocks via changes in volatility."
- Normandin, M. & Phaneuf, L. (2004). "Monetary policy shocks."
- Lütkepohl, H. & Netšunajev, A. (2017). "Structural vector autoregressions with smooth transition in variances."
"""

using LinearAlgebra, Statistics, Distributions
import Optim
import ForwardDiff

# =============================================================================
# Result Types
# =============================================================================

"""
    MarkovSwitchingSVARResult{T} <: AbstractNonGaussianSVAR

Result from Markov-switching heteroskedasticity SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix
- `Q::Matrix{T}` — rotation matrix
- `Sigma_regimes::Vector{Matrix{T}}` — covariance per regime
- `Lambda::Vector{Vector{T}}` — relative variances per regime
- `regime_probs::Matrix{T}` — smoothed regime probabilities (T × K)
- `transition_matrix::Matrix{T}` — Markov transition probabilities (K × K)
- `loglik::T`
- `converged::Bool`
- `iterations::Int`
- `n_regimes::Int`
"""
struct MarkovSwitchingSVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    Sigma_regimes::Vector{Matrix{T}}
    Lambda::Vector{Vector{T}}
    regime_probs::Matrix{T}
    transition_matrix::Matrix{T}
    loglik::T
    converged::Bool
    iterations::Int
    n_regimes::Int
end

function Base.show(io::IO, r::MarkovSwitchingSVARResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Variables"  n;
        "Regimes"    r.n_regimes;
        "Log-likelihood" _fmt(r.loglik; digits=4);
        "Converged"  r.converged ? "Yes" : "No";
        "Iterations" r.iterations
    ]
    _pretty_table(io, spec;
        title = "Markov-Switching SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _matrix_table(io, r.B0, "Structural Impact Matrix (B₀)";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Shock $j" for j in 1:n])
end

"""
    GARCHSVARResult{T} <: AbstractNonGaussianSVAR

Result from GARCH-based SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix
- `Q::Matrix{T}` — rotation matrix
- `garch_params::Matrix{T}` — (n × 3): [ω, α, β] per shock
- `cond_var::Matrix{T}` — (T_eff × n) conditional variances
- `shocks::Matrix{T}` — structural shocks
- `loglik::T`
- `converged::Bool`
- `iterations::Int`
"""
struct GARCHSVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    garch_params::Matrix{T}
    cond_var::Matrix{T}
    shocks::Matrix{T}
    loglik::T
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, r::GARCHSVARResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Variables"      n;
        "Log-likelihood" _fmt(r.loglik; digits=4);
        "Converged"      r.converged ? "Yes" : "No";
        "Iterations"     r.iterations
    ]
    _pretty_table(io, spec;
        title = "GARCH-SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    # GARCH parameters table
    garch_data = Matrix{Any}(undef, n, 4)
    for i in 1:n
        garch_data[i, 1] = "Shock $i"
        garch_data[i, 2] = _fmt(r.garch_params[i, 1])
        garch_data[i, 3] = _fmt(r.garch_params[i, 2])
        garch_data[i, 4] = _fmt(r.garch_params[i, 3])
    end
    _pretty_table(io, garch_data;
        title = "GARCH Parameters",
        column_labels = ["", "ω", "α", "β"],
        alignment = [:l, :r, :r, :r],
    )
    _matrix_table(io, r.B0, "Structural Impact Matrix (B₀)";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Shock $j" for j in 1:n])
end

"""
    SmoothTransitionSVARResult{T} <: AbstractNonGaussianSVAR

Result from smooth-transition heteroskedasticity SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix
- `Q::Matrix{T}` — rotation matrix
- `Sigma_regimes::Vector{Matrix{T}}` — covariance matrices for extreme regimes
- `Lambda::Vector{Vector{T}}` — relative variances per regime
- `gamma::T` — transition speed parameter
- `threshold::T` — transition location parameter
- `transition_var::Vector{T}` — transition variable values
- `G_values::Vector{T}` — transition function G(s_t) values
- `loglik::T`
- `converged::Bool`
- `iterations::Int`
- `se::Union{Nothing,Vector{T}}` — parameter SEs (nothing until SID-10)
- `vcov::Union{Nothing,Matrix{T}}` — parameter covariance (nothing until SID-10)
"""
struct SmoothTransitionSVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    Sigma_regimes::Vector{Matrix{T}}
    Lambda::Vector{Vector{T}}
    gamma::T
    threshold::T
    transition_var::Vector{T}
    G_values::Vector{T}
    loglik::T
    converged::Bool
    iterations::Int
    se::Union{Nothing,Vector{T}}
    vcov::Union{Nothing,Matrix{T}}
end

# Back-compat: 11-arg positional calls default se/vcov to nothing until SID-10.
function SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                       transition_var, G_values, loglik, converged,
                                       iterations) where {T<:AbstractFloat}
    SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                  transition_var, G_values, loglik, converged,
                                  iterations, nothing, nothing)
end

function SmoothTransitionSVARResult(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                    transition_var, G_values, loglik, converged,
                                    iterations)
    T = eltype(B0)
    SmoothTransitionSVARResult{T}(B0, Q, Sigma_regimes, Lambda, gamma, threshold,
                                  transition_var, G_values, loglik, converged,
                                  iterations, nothing, nothing)
end

function Base.show(io::IO, r::SmoothTransitionSVARResult{T}) where {T}
    n = size(r.B0, 1)
    se_γ = r.se === nothing ? "—" : _fmt(r.se[1]; digits=2)
    se_c = r.se === nothing ? "—" : _fmt(r.se[2]; digits=4)
    spec = Any[
        "Variables"      n;
        "γ (speed)"      _fmt(r.gamma; digits=2);
        "γ SE"           se_γ;
        "Threshold"      _fmt(r.threshold; digits=4);
        "Threshold SE"   se_c;
        "Log-likelihood" _fmt(r.loglik; digits=4);
        "Converged"      r.converged ? "Yes" : "No";
        "Iterations"     r.iterations
    ]
    _pretty_table(io, spec;
        title = "Smooth-Transition SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _matrix_table(io, r.B0, "Structural Impact Matrix (B₀)";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Shock $j" for j in 1:n])
end

"""
    ExternalVolatilitySVARResult{T} <: AbstractNonGaussianSVAR

Result from external volatility instrument SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix
- `Q::Matrix{T}` — rotation matrix
- `Sigma_regimes::Vector{Matrix{T}}` — covariance per regime
- `Lambda::Vector{Vector{T}}` — relative variances per regime
- `regime_indices::Vector{Vector{Int}}` — observation indices per regime
- `loglik::T`
"""
struct ExternalVolatilitySVARResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    Sigma_regimes::Vector{Matrix{T}}
    Lambda::Vector{Vector{T}}
    regime_indices::Vector{Vector{Int}}
    loglik::T
end

function Base.show(io::IO, r::ExternalVolatilitySVARResult{T}) where {T}
    n = size(r.B0, 1)
    K = length(r.Sigma_regimes)
    spec = Any[
        "Variables"      n;
        "Regimes"        K;
        "Log-likelihood" _fmt(r.loglik; digits=4)
    ]
    _pretty_table(io, spec;
        title = "External Volatility SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _matrix_table(io, r.B0, "Structural Impact Matrix (B₀)";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Shock $j" for j in 1:n])
end

# =============================================================================
# Hamilton Filter/Smoother
# =============================================================================

"""Hamilton (1989) forward filter for Markov-switching model."""
function _hamilton_filter(U::Matrix{T}, Sigma_regimes::Vector{Matrix{T}},
                          transition_matrix::Matrix{T}) where {T<:AbstractFloat}
    T_obs, n = size(U)
    K = length(Sigma_regimes)

    # Precompute
    Sigma_invs = [robust_inv(S) for S in Sigma_regimes]
    logdet_Sigmas = [logdet_safe(S) for S in Sigma_regimes]

    filtered_probs = zeros(T, T_obs, K)
    predicted_probs = zeros(T, T_obs, K)
    loglik = zero(T)

    # Initial probabilities: ergodic distribution
    P = transition_matrix
    A = [P' - I; ones(T, 1, K)]
    b = [zeros(T, K); one(T)]
    xi_0 = try
        (A' * A) \ (A' * b)
    catch
        fill(one(T) / K, K)
    end

    predicted_probs[1, :] = xi_0

    for t in 1:T_obs
        u = @view U[t, :]
        eta = zeros(T, K)

        for k in 1:K
            eta[k] = exp(-T(0.5) * (n * log(T(2π)) + logdet_Sigmas[k] +
                     dot(u, Sigma_invs[k] * u)))
        end

        xi_pred = t == 1 ? xi_0 : predicted_probs[t, :]
        joint = xi_pred .* eta
        margin = sum(joint)
        margin = max(margin, eps(T))
        loglik += log(margin)

        filtered_probs[t, :] = joint / margin

        if t < T_obs
            predicted_probs[t + 1, :] = P' * filtered_probs[t, :]
        end
    end

    (filtered_probs, predicted_probs, loglik)
end

"""Kim (1994) backward smoother for Markov-switching model."""
function _hamilton_smoother(filtered_probs::Matrix{T}, predicted_probs::Matrix{T},
                            transition_matrix::Matrix{T}) where {T<:AbstractFloat}
    T_obs, K = size(filtered_probs)
    smoothed = zeros(T, T_obs, K)
    smoothed[T_obs, :] = filtered_probs[T_obs, :]

    P = transition_matrix
    for t in (T_obs-1):-1:1
        for i in 1:K
            s = zero(T)
            for j in 1:K
                pred_j = max(predicted_probs[t + 1, j], eps(T))
                s += P[i, j] * smoothed[t + 1, j] / pred_j
            end
            smoothed[t, i] = filtered_probs[t, i] * s
        end
        # Normalize
        total = sum(smoothed[t, :])
        if total > 0
            smoothed[t, :] /= total
        end
    end
    smoothed
end

"""EM step for Markov-switching covariances and transition matrix.

Uses Kim (1994) joint smoothed probabilities for the transition matrix update:
  ξ_{t-1,t|T}(i,j) = ξ_{t|T}(j) · P[i,j] · ξ_{t-1|t-1}(i) / ξ_{t|t-1}(j)
"""
function _ms_em_step(U::Matrix{T}, smoothed::Matrix{T}, filtered::Matrix{T},
                      predicted::Matrix{T}, P::Matrix{T}, K::Int) where {T<:AbstractFloat}
    T_obs, n = size(U)

    # Update covariance matrices (uses smoothed probabilities as weights)
    Sigma_new = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        w = max.(smoothed[:, k], eps(T))
        w_sum = sum(w)
        Sigma_k = zeros(T, n, n)
        for t in 1:T_obs
            u = @view U[t, :]
            Sigma_k .+= w[t] * (u * u')
        end
        Sigma_new[k] = Symmetric(Sigma_k / w_sum)
        # Regularize
        Sigma_new[k] = Sigma_new[k] + eps(T) * I
    end

    # Update transition matrix using Kim (1994) / Hamilton (1994 Ch.22)
    # joint smoothed probabilities instead of marginal products
    P_new = zeros(T, K, K)
    for t in 2:T_obs
        for i in 1:K
            for j in 1:K
                pred_j = max(predicted[t, j], eps(T))
                joint_ij = smoothed[t, j] * P[i, j] * filtered[t-1, i] / pred_j
                P_new[i, j] += joint_ij
            end
        end
    end
    # Normalize rows
    for i in 1:K
        row_sum = sum(P_new[i, :])
        if row_sum > 0
            P_new[i, :] /= row_sum
        else
            P_new[i, :] .= one(T) / K
        end
    end

    (Sigma_new, P_new)
end

# =============================================================================
# Public API: Markov-Switching
# =============================================================================

"""
    identify_markov_switching(model::VARModel; n_regimes=2, max_iter=500, tol=1e-6) -> MarkovSwitchingSVARResult

Identify SVAR via Markov-switching heteroskedasticity (Lanne & Lütkepohl 2008).

Estimates regime-specific covariance matrices Σ₁, Σ₂, ..., Σ_K via EM algorithm,
then identifies B₀ from the eigendecomposition of Σ₁⁻¹ Σ₂.

Identification requires that the relative variance ratios (eigenvalues) are distinct.

**Reference**: Lanne & Lütkepohl (2008), Rigobon (2003)
"""
function identify_markov_switching(model::VARModel{T}; n_regimes::Int=2,
                                    max_iter::Int=500,
                                    tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    K = n_regimes
    T_obs = size(model.U, 1)

    # Initialize: K-means-like initialization
    Sigma_regimes = Vector{Matrix{T}}(undef, K)
    chunk = T_obs ÷ K
    for k in 1:K
        idx_start = (k - 1) * chunk + 1
        idx_end = k == K ? T_obs : k * chunk
        U_k = model.U[idx_start:idx_end, :]
        Sigma_regimes[k] = cov(U_k) + eps(T) * I
    end

    P = zeros(T, K, K)
    for i in 1:K
        for j in 1:K
            P[i, j] = i == j ? T(0.9) : T(0.1) / (K - 1)
        end
    end

    loglik_old = T(-Inf)
    converged = false
    iter = 0

    for it in 1:max_iter
        iter = it

        # E-step: Hamilton filter + smoother
        filtered, predicted, loglik = _hamilton_filter(model.U, Sigma_regimes, P)
        smoothed = _hamilton_smoother(filtered, predicted, P)

        # Check convergence
        if abs(loglik - loglik_old) < tol * abs(loglik_old + one(T))
            converged = true
            break
        end
        loglik_old = loglik

        # M-step
        Sigma_regimes, P = _ms_em_step(model.U, smoothed, filtered, predicted, P, K)
    end

    # Final filter for smoothed probabilities
    filtered, predicted, loglik = _hamilton_filter(model.U, Sigma_regimes, P)
    smoothed = _hamilton_smoother(filtered, predicted, P)

    # Identify B₀ from regime covariances
    B0, Q, Lambda = _eigendecomposition_id(Sigma_regimes[1], Sigma_regimes[2])

    # Compute Lambda vectors for each regime
    Lambda_vecs = Vector{Vector{T}}(undef, K)
    B0_inv = robust_inv(B0)
    for k in 1:K
        D_k = diag(B0_inv * Sigma_regimes[k] * B0_inv')
        Lambda_vecs[k] = D_k
    end

    MarkovSwitchingSVARResult{T}(B0, Q, Sigma_regimes, Lambda_vecs, smoothed, P,
                                  loglik, converged, iter, K)
end

# =============================================================================
# GARCH(1,1) Helpers
# =============================================================================

"""GARCH(1,1) conditional variance filter: h_t = ω + α ε²_{t-1} + β h_{t-1}."""
function _garch11_filter(omega::T, alpha::T, beta::T,
                          epsilon_sq::Vector{T}) where {T<:AbstractFloat}
    T_obs = length(epsilon_sq)
    h = Vector{T}(undef, T_obs)
    h[1] = omega / max(one(T) - alpha - beta, eps(T))  # unconditional variance
    for t in 2:T_obs
        h[t] = omega + alpha * epsilon_sq[t-1] + beta * h[t-1]
        h[t] = max(h[t], eps(T))
    end
    h
end

"""GARCH(1,1) log-likelihood (negative, for minimization)."""
function _garch11_loglik(params::Vector{T}, epsilon_sq::Vector{T}) where {T<:AbstractFloat}
    omega = exp(params[1])
    alpha = one(T) / (one(T) + exp(-params[2])) * T(0.5)  # constrain to (0, 0.5)
    beta = one(T) / (one(T) + exp(-params[3])) * T(0.99)   # constrain to (0, 0.99)

    # Ensure stationarity
    if alpha + beta >= one(T)
        return T(Inf)
    end

    h = _garch11_filter(omega, alpha, beta, epsilon_sq)
    T_obs = length(epsilon_sq)

    loglik = zero(T)
    for t in 1:T_obs
        loglik -= T(0.5) * (log(T(2π)) + log(h[t]) + epsilon_sq[t] / h[t])
    end
    -loglik  # negative for minimization
end

"""Estimate GARCH(1,1) parameters for a single series."""
function _estimate_garch11(epsilon_sq::Vector{T}) where {T<:AbstractFloat}
    # Stationary GARCH(1,1) start: α≈0.05, β≈0.90 ⇒ α+β=0.95 < 1.
    # Inverse of α = 0.5/(1+exp(-p2)) ⇒ p2 = -log(0.5/α - 1); same for β with cap 0.99.
    # Previous init [log(var*0.05), 0, 2] mapped to (α,β)=(0.25,0.872) with α+β>1 (Inf loglik).
    α0, β0 = T(0.05), T(0.90)
    p2_0 = -log(T(0.5) / α0 - one(T))          # logit for α-map
    p3_0 = -log(T(0.99) / β0 - one(T))          # logit for β-map
    var_eps = max(var(epsilon_sq), eps(T))
    params0 = [log(var_eps * (one(T) - α0 - β0)), p2_0, p3_0]

    obj = p -> begin
        ll = _garch11_loglik(p, epsilon_sq)
        ifelse(isfinite(ll), ll, T(1e20))  # reject Inf / NaN starts inside Nelder-Mead
    end
    result = Optim.optimize(obj, params0,
                            Optim.NelderMead(),
                            Optim.Options(iterations=500))

    p = Optim.minimizer(result)
    omega = exp(p[1])
    alpha = one(T) / (one(T) + exp(-p[2])) * T(0.5)
    beta = one(T) / (one(T) + exp(-p[3])) * T(0.99)
    # Final stationarity clip (should already hold via the barrier)
    if alpha + beta >= one(T)
        s = alpha + beta
        alpha *= T(0.99) / s
        beta  *= T(0.99) / s
    end

    h = _garch11_filter(omega, alpha, beta, epsilon_sq)
    (omega, alpha, beta, h)
end

# =============================================================================
# Public API: GARCH
# =============================================================================

"""
    identify_garch(model::VARModel; max_iter=500, tol=1e-6) -> GARCHSVARResult

Identify SVAR via GARCH-based heteroskedasticity (Normandin & Phaneuf 2004).

Iterative procedure:
1. Start with Cholesky B₀
2. Compute structural shocks ε_t = B₀⁻¹ u_t
3. Fit GARCH(1,1) to each ε_j,t
4. Use conditional covariances to re-estimate B₀
5. Repeat until convergence

**Reference**: Normandin & Phaneuf (2004)
"""
function identify_garch(model::VARModel{T}; max_iter::Int=500,
                         tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)

    # Initialize with Cholesky: B₀ = L * Q(θ), start at Q = I (θ = 0)
    L = safe_cholesky(model.Sigma)
    L_mat = Matrix(L)
    n_angles = n * (n - 1) ÷ 2
    angles = zeros(T, n_angles)
    log_det_B0_inv = -sum(log(L_mat[i, i]) for i in 1:n)

    # Precompute whitened residuals: Z = U * L⁻ᵀ, so shocks = Z * Q
    L_inv_t = Matrix{T}(robust_inv(L_mat)')
    Z = model.U * L_inv_t

    garch_params = zeros(T, n, 3)
    cond_var = ones(T, T_obs, n)
    loglik_old = T(-Inf)
    converged = false
    iter = 0

    for it in 1:max_iter
        iter = it

        # Build B₀ from current Givens angles, compute structural shocks
        Q = _givens_to_orthogonal(angles, n)
        shocks = Z * Q

        # Fit GARCH(1,1) to each structural shock series
        loglik = zero(T)
        for j in 1:n
            resid_sq = shocks[:, j] .^ 2
            omega, alpha, beta, h = _estimate_garch11(resid_sq)
            garch_params[j, :] = [omega, alpha, beta]
            cond_var[:, j] = h

            for t in 1:T_obs
                loglik -= T(0.5) * (log(T(2π)) + log(h[t]) + resid_sq[t] / h[t])
            end
        end
        loglik += T_obs * log_det_B0_inv

        # Check convergence
        if abs(loglik - loglik_old) < tol * abs(loglik_old + one(T))
            converged = true
            break
        end
        loglik_old = loglik

        # Re-estimate B₀ by optimizing Givens angles with fixed GARCH variances.
        # Since |det(Q)| = 1, the log-det term is constant w.r.t. θ.
        # The log(h) terms are also fixed. Only the weighted residual sum varies.
        if n_angles > 0
            obj = theta -> begin
                Q_t = _givens_to_orthogonal(theta, n)
                shocks_t = Z * Q_t
                val = zero(T)
                for t in 1:T_obs
                    for j in 1:n
                        val += shocks_t[t, j]^2 / cond_var[t, j]
                    end
                end
                val
            end
            result_opt = Optim.optimize(obj, angles, Optim.NelderMead(),
                                        Optim.Options(iterations=200))
            angles = Optim.minimizer(result_opt)
        end
    end

    # Final B₀ and Q with sign normalization
    Q = _givens_to_orthogonal(angles, n)
    B0 = L_mat * Q
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] *= -one(T)
            Q[:, j] *= -one(T)
        end
    end
    B0_inv = robust_inv(B0)
    shocks = (B0_inv * model.U')'

    GARCHSVARResult{T}(B0, Q, garch_params, cond_var, shocks, loglik_old, converged, iter)
end

# =============================================================================
# Smooth Transition
# =============================================================================

"""Logistic transition: G(s) = 1 / (1 + exp(-γ(s - c)/σs)). Dual-friendly (no AbstractFloat bound)."""
_logistic_transition(s, gamma, c) = _logistic_transition(s, gamma, c, one(gamma))
function _logistic_transition(s, gamma, c, sigma_s)
    z = gamma * (s - c) / sigma_s
    one(z) / (one(z) + exp(-z))
end

"""Pack lower-triangular L as `[log diag(L); strictly-lower column-major]`."""
function _st_pack_L(L::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(L, 1)
    v = Vector{T}(undef, n * (n + 1) ÷ 2)
    k = 1
    for i in 1:n
        v[k] = log(max(L[i, i], T(1e-12)))
        k += 1
    end
    for j in 1:(n - 1)
        for i in (j + 1):n
            v[k] = L[i, j]
            k += 1
        end
    end
    v
end

"""Unpack a free Cholesky factor. Dual-friendly."""
function _st_unpack_L(v, n::Int)
    Tθ = eltype(v)
    L = zeros(Tθ, n, n)
    k = 1
    for i in 1:n
        L[i, i] = exp(v[k])
        k += 1
    end
    for j in 1:(n - 1)
        for i in (j + 1):n
            L[i, j] = v[k]
            k += 1
        end
    end
    L
end

"""Forward substitution Lz = u for lower-triangular L. Dual-friendly."""
function _st_forward_sub(L, u)
    n = length(u)
    Tθ = promote_type(eltype(L), eltype(u))
    z = Vector{Tθ}(undef, n)
    @inbounds for i in 1:n
        acc = Tθ(u[i])
        for j in 1:(i - 1)
            acc -= L[i, j] * z[j]
        end
        z[i] = acc / L[i, i]
    end
    z
end

"""Negative Gaussian log-likelihood of the smooth-transition SVAR. Dual-friendly."""
function _smooth_transition_nll(params, U::AbstractMatrix, s::AbstractVector,
                                sigma_s, n::Int, logγ_lo, logγ_hi)
    n_L = n * (n + 1) ÷ 2
    n_angles = n * (n - 1) ÷ 2
    L = _st_unpack_L(view(params, 1:n_L), n)
    θ = view(params, (n_L + 1):(n_L + n_angles))
    logΛ = view(params, (n_L + n_angles + 1):(n_L + n_angles + n))
    xγ = params[n_L + n_angles + n + 1]
    cc = params[n_L + n_angles + n + 2]
    Q = _givens_to_orthogonal(θ, n)
    Tθ = eltype(params)
    half = Tθ(0.5)
    log2π = log(Tθ(2) * Tθ(π))
    floor_d = Tθ(1e-12)
    logdetL = zero(Tθ)
    for i in 1:n
        logdetL += log(L[i, i])
    end
    γ = _st_gamma_from_x(xγ, logγ_lo, logγ_hi)
    nll = zero(Tθ)
    T_obs = size(U, 1)
    for t in 1:T_obs
        G_t = _logistic_transition(s[t], γ, cc, sigma_s)
        z = _st_forward_sub(L, view(U, t, :))
        acc = zero(Tθ)
        for j in 1:n
            λj = exp(logΛ[j])
            d_j = max(one(Tθ) + G_t * (λj - one(Tθ)), floor_d)
            εj = zero(Tθ)
            for i in 1:n
                εj += Q[i, j] * z[i]
            end
            acc += log(d_j) + εj^2 / d_j
        end
        nll += half * (Tθ(n) * log2π + Tθ(2) * logdetL + acc)
    end
    nll
end

"""Map the unbounded γ-score to a bounded γ ∈ [γ_lo, γ_hi]."""
function _st_gamma_from_x(xγ, logγ_lo, logγ_hi)
    logγ = logγ_lo + (logγ_hi - logγ_lo) / (one(xγ) + exp(-xγ))
    exp(logγ)
end

function _st_x_from_gamma(γ, logγ_lo, logγ_hi)
    logγ = log(γ)
    p = (logγ - logγ_lo) / (logγ_hi - logγ_lo)
    p = clamp(p, oftype(p, 1e-8), oftype(p, 1 - 1e-8))
    log(p / (one(p) - p))
end

"""
    identify_smooth_transition(model::VARModel, transition_var::AbstractVector;
                               max_iter=500, tol=1e-6) -> SmoothTransitionSVARResult

Identify SVAR via smooth-transition heteroskedasticity (Lütkepohl & Netšunajev 2017).

Joint ML over `(θ_Givens, Λ, γ, c)` with
```math
\\Sigma_t = B_0 [I + G(s_t)(\\Lambda - I)] B_0', \\qquad B_0 = L Q(\\theta)
```
where ``G(s_t) = 1/(1 + \\exp(-\\gamma(s_t - c)/\\mathrm{std}(s)))``. `L` is
initialized from the median-split G=0 Cholesky and estimated jointly with
`(θ, Λ, γ, c)` so `B₀B₀'` is the G=0 pole, not the unconditional covariance.
`log γ` is bounded so `γ ∈ [10^{-3}, 20]/std(s)`.

Arguments:
- `transition_var` — the transition variable s_t (e.g., a lagged endogenous variable)

**Reference**: Lütkepohl & Netšunajev (2017)
"""
function identify_smooth_transition(model::VARModel{T}, transition_var::AbstractVector;
                                     max_iter::Int=500,
                                     tol::T=T(1e-6)) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)
    s = Vector{T}(transition_var[1:T_obs])
    sigma_s = std(s)
    sigma_s > zero(T) || throw(ArgumentError(
        "transition variable has zero variance; cannot scale γ."))

    # Split kernel: starting point, not the estimator.
    gamma_init = one(T) / sigma_s
    c_init = median(s)
    G_vals = [_logistic_transition(s[t], gamma_init, c_init, sigma_s) for t in 1:T_obs]
    low_idx = findall(g -> g < T(0.5), G_vals)
    high_idx = findall(g -> g >= T(0.5), G_vals)
    Sigma1 = isempty(low_idx) ? cov(model.U) : cov(model.U[low_idx, :])
    Sigma2 = isempty(high_idx) ? cov(model.U) : cov(model.U[high_idx, :])
    Sigma1 = Matrix{T}(Symmetric(Sigma1 + eps(T) * I))
    Sigma2 = Matrix{T}(Symmetric(Sigma2 + eps(T) * I))

    B0_split, Q_split, Lambda_raw = _eigendecomposition_id(Sigma1, Sigma2)
    Lambda_init = max.(Lambda_raw, T(1e-8))

    # B₀ = L Q(θ). L starts at chol(Σ_{G=0}) from the split kernel and is
    # estimated jointly so the G=0 pole is not frozen at the split/unconditional Σ.
    L_init = Matrix{T}(safe_cholesky(Sigma1))
    Q_init = Q_split
    if det(Q_init) < 0
        Q_init = copy(Q_init)
        Q_init[:, n] .*= -one(T)
    end
    θ_init = _orthogonal_to_givens(Q_init, n)

    γ_lo = T(1e-3) / sigma_s
    γ_hi = T(20) / sigma_s
    logγ_lo = log(γ_lo)
    logγ_hi = log(γ_hi)
    xγ_init = _st_x_from_gamma(gamma_init, logγ_lo, logγ_hi)

    n_L = n * (n + 1) ÷ 2
    n_angles = n * (n - 1) ÷ 2
    params0 = vcat(_st_pack_L(L_init), θ_init, log.(Lambda_init), xγ_init, c_init)

    obj = p -> _smooth_transition_nll(p, model.U, s, sigma_s, n, logγ_lo, logγ_hi)
    g! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
    result = Optim.optimize(obj, g!, params0, Optim.LBFGS(),
                            Optim.Options(iterations=max_iter, g_tol=min(tol, T(1e-8)),
                                          f_reltol=T(1e-12), allow_f_increases=true))

    p_opt = Optim.minimizer(result)
    L_mat = Matrix{T}(_st_unpack_L(view(p_opt, 1:n_L), n))
    θ_opt = p_opt[(n_L + 1):(n_L + n_angles)]
    Λ_opt = exp.(p_opt[(n_L + n_angles + 1):(n_L + n_angles + n)])
    gamma_opt = _st_gamma_from_x(p_opt[n_L + n_angles + n + 1], logγ_lo, logγ_hi)
    c_opt = p_opt[n_L + n_angles + n + 2]
    Q = _givens_to_orthogonal(θ_opt, n)
    B0 = L_mat * Q

    idx = sortperm(Λ_opt)
    Λ_opt = Λ_opt[idx]
    Q = Q[:, idx]
    B0 = B0[:, idx]
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] .*= -one(T)
            Q[:, j] .*= -one(T)
        end
    end

    Sigma_G0 = B0 * B0'
    Sigma_G1 = B0 * Diagonal(Λ_opt) * B0'
    G_final = [_logistic_transition(s[t], gamma_opt, c_opt, sigma_s) for t in 1:T_obs]
    loglik_final = -obj(p_opt)

    SmoothTransitionSVARResult{T}(B0, Q, [Matrix{T}(Sigma_G0), Matrix{T}(Sigma_G1)],
                                   [ones(T, n), Vector{T}(Λ_opt)],
                                   T(gamma_opt), T(c_opt), s, G_final, T(loglik_final),
                                   Optim.converged(result), Optim.iterations(result),
                                   nothing, nothing)
end

# =============================================================================
# Public API: External Volatility
# =============================================================================

"""
    identify_external_volatility(model::VARModel, regime_indicator::AbstractVector{Int};
                                 regimes=2) -> ExternalVolatilitySVARResult

Identify SVAR via externally specified volatility regimes (Rigobon 2003).

Uses a known regime indicator (e.g., NBER recessions, financial crises) to split
the sample and estimate regime-specific covariance matrices.

Arguments:
- `regime_indicator` — integer vector of regime labels (1, 2, ..., K)
- `regimes` — number of distinct regimes (default: 2)

A regime with fewer than `n + 1` observations throws `ArgumentError`. Until SID-10
only the first two regimes enter the kernel (`K ≥ 3` warns).

**Reference**: Rigobon (2003)
"""
function identify_external_volatility(model::VARModel{T},
                                       regime_indicator::AbstractVector{Int};
                                       regimes::Int=2) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)
    K = regimes

    @assert length(regime_indicator) >= T_obs "regime_indicator must have length ≥ T_obs"
    K >= 2 || throw(ArgumentError("regimes must be ≥ 2, got $K"))
    if K >= 3
        @warn "identify_external_volatility uses only two regimes for identification until SID-10; K-regime joint ML is not yet applied"
    end

    # Split sample by regime
    regime_indices = [findall(regime_indicator[1:T_obs] .== k) for k in 1:K]

    Sigma_regimes = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        idx = regime_indices[k]
        if length(idx) < n + 1
            throw(ArgumentError(
                "regime $k has $(length(idx)) observations; need at least $(n + 1)"))
        end
        Sigma_regimes[k] = cov(model.U[idx, :]) + eps(T) * I
    end

    # Identify from regime 1 and 2
    B0, Q, Lambda = _eigendecomposition_id(Sigma_regimes[1], Sigma_regimes[2])

    # Compute all Lambda vectors
    Lambda_vecs = Vector{Vector{T}}(undef, K)
    B0_inv = robust_inv(B0)
    for k in 1:K
        Lambda_vecs[k] = diag(B0_inv * Sigma_regimes[k] * B0_inv')
    end

    # Log-likelihood
    loglik = zero(T)
    for k in 1:K
        idx = regime_indices[k]
        ld = logdet_safe(Sigma_regimes[k])
        Sigma_inv = robust_inv(Sigma_regimes[k])
        for t in idx
            u = @view model.U[t, :]
            loglik -= T(0.5) * (n * log(T(2π)) + ld + dot(u, Sigma_inv * u))
        end
    end

    ExternalVolatilitySVARResult{T}(B0, Q, Sigma_regimes, Lambda_vecs, regime_indices, loglik)
end
