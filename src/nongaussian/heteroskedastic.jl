"""
Heteroskedasticity-based SVAR identification.

Exploits changes in the volatility regime to identify structural shocks without
distributional assumptions. Methods: Markov-switching, GARCH, smooth transition,
external volatility instruments.

References:
- Rigobon, R. (2003). "Identification through heteroskedasticity."
- Lanne, M. & Lütkepohl, H. (2008). "Identifying monetary policy shocks via changes in volatility."
- Normandin, M. & Phaneuf, L. (2004). "Monetary policy shocks."
- Lütkepohl, H. & Netšunajev, A. (2017). "Structural vector autoregressions with smooth transition in variances."
"""

using LinearAlgebra, Statistics, Distributions
import Optim

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
end

function Base.show(io::IO, r::SmoothTransitionSVARResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Variables"      n;
        "γ (speed)"      _fmt(r.gamma; digits=2);
        "Threshold"      _fmt(r.threshold; digits=4);
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
# Core Identification: Eigendecomposition
# =============================================================================

"""
Identify B₀ from two covariance matrices via eigendecomposition.

Given Σ₁, Σ₂:
  Σ₁⁻¹ Σ₂ has eigendecomposition V D V⁻¹
  B₀ = Σ₁^{1/2} V (normalized so B₀ B₀' = Σ₁)

Returns (B₀, Λ) where Λ = diag(D) are relative variance ratios.
Identification requires distinct eigenvalues.
"""
function _eigendecomposition_id(Sigma1::Matrix{T}, Sigma2::Matrix{T}) where {T<:AbstractFloat}
    n = size(Sigma1, 1)
    S1_inv = robust_inv(Sigma1)
    M = S1_inv * Sigma2

    E = eigen(M)
    D = real.(E.values)
    V = real.(E.vectors)

    # Sort by eigenvalue magnitude for consistent ordering
    idx = sortperm(D)
    D = D[idx]
    V = V[:, idx]

    # B₀ = chol(Σ₁) * V, normalized so columns have unit norm
    L1 = safe_cholesky(Sigma1)
    B0 = Matrix(L1) * V

    # Normalize columns
    for j in 1:n
        B0[:, j] /= norm(B0[:, j])
    end

    # Scale so B₀ B₀' ≈ Σ₁
    # B₀ = L₁ * Q where Q is orthogonal
    Q_raw = robust_inv(Matrix(L1)) * B0
    F = svd(Q_raw)
    Q = F.U * F.Vt
    B0 = Matrix(L1) * Q

    Lambda = D

    # Sign convention: positive diagonal
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] *= -one(T)
            Q[:, j] *= -one(T)
        end
    end

    (B0, Q, Lambda)
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

"""EM step for Markov-switching covariances and transition matrix."""
function _ms_em_step(U::Matrix{T}, probs::Matrix{T}, K::Int) where {T<:AbstractFloat}
    T_obs, n = size(U)

    # Update covariance matrices
    Sigma_new = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        w = max.(probs[:, k], eps(T))
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

    # Update transition matrix
    P_new = zeros(T, K, K)
    for i in 1:K, j in 1:K
        num = zero(T)
        for t in 2:T_obs
            num += probs[t-1, i] * probs[t, j]
        end
        P_new[i, j] = num
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
        Sigma_regimes, P = _ms_em_step(model.U, smoothed, K)
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
    params0 = [log(var(epsilon_sq) * T(0.05)), T(0.0), T(2.0)]  # omega, alpha, beta init

    result = Optim.optimize(p -> _garch11_loglik(p, epsilon_sq), params0,
                            Optim.NelderMead(),
                            Optim.Options(iterations=500))

    p = Optim.minimizer(result)
    omega = exp(p[1])
    alpha = one(T) / (one(T) + exp(-p[2])) * T(0.5)
    beta = one(T) / (one(T) + exp(-p[3])) * T(0.99)

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

    # Initialize with Cholesky
    L = safe_cholesky(model.Sigma)
    B0 = Matrix(L)
    Q = Matrix{T}(I, n, n)

    garch_params = zeros(T, n, 3)
    cond_var = ones(T, T_obs, n)
    loglik_old = T(-Inf)
    converged = false
    iter = 0

    for it in 1:max_iter
        iter = it

        # Compute structural shocks
        B0_inv = robust_inv(B0)
        shocks = (B0_inv * model.U')'

        # Fit GARCH(1,1) to each shock
        loglik = zero(T)
        for j in 1:n
            eps_sq = shocks[:, j] .^ 2
            omega, alpha, beta, h = _estimate_garch11(eps_sq)
            garch_params[j, :] = [omega, alpha, beta]
            cond_var[:, j] = h

            # Contribution to log-likelihood
            for t in 1:T_obs
                loglik -= T(0.5) * (log(T(2π)) + log(h[t]) + eps_sq[t] / h[t])
            end
        end
        loglik += T_obs * log(max(abs(det(B0_inv)), eps(T)))

        # Check convergence
        if abs(loglik - loglik_old) < tol * abs(loglik_old + one(T))
            converged = true
            break
        end
        loglik_old = loglik

        # Re-estimate B₀ using weighted covariances
        # Use time-varying Σ_t = B₀ diag(h_t) B₀'
        # Simplified: use two sub-periods with different volatilities
        mid = T_obs ÷ 2
        Sigma1 = zeros(T, n, n)
        Sigma2 = zeros(T, n, n)
        for t in 1:mid
            u = @view model.U[t, :]
            Sigma1 .+= u * u'
        end
        for t in (mid+1):T_obs
            u = @view model.U[t, :]
            Sigma2 .+= u * u'
        end
        Sigma1 ./= mid
        Sigma2 ./= (T_obs - mid)

        B0_new, Q_new, _ = _eigendecomposition_id(Sigma1, Sigma2)
        B0 = B0_new
        Q = Q_new
    end

    # Final shocks
    B0_inv = robust_inv(B0)
    shocks = (B0_inv * model.U')'

    GARCHSVARResult{T}(B0, Q, garch_params, cond_var, shocks, loglik_old, converged, iter)
end

# =============================================================================
# Smooth Transition
# =============================================================================

"""Logistic transition function: G(s) = 1 / (1 + exp(-γ(s - c)))."""
function _logistic_transition(s::T, gamma::T, c::T) where {T<:AbstractFloat}
    one(T) / (one(T) + exp(-gamma * (s - c)))
end

"""
    identify_smooth_transition(model::VARModel, transition_var::AbstractVector;
                               max_iter=500, tol=1e-6) -> SmoothTransitionSVARResult

Identify SVAR via smooth-transition heteroskedasticity (Lütkepohl & Netšunajev 2017).

The covariance matrix varies smoothly between two regimes:
```math
\\Sigma_t = B_0 [I + G(s_t)(\\Lambda - I)] B_0'
```
where G(s_t) = 1/(1 + exp(-γ(s_t - c))) is the logistic transition function.

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

    # Initialize
    gamma_init = T(1.0) / std(s)
    c_init = median(s)

    G_vals = [_logistic_transition(s[t], gamma_init, c_init) for t in 1:T_obs]

    # Split into low-G and high-G periods for initial covariances
    low_idx = findall(g -> g < T(0.5), G_vals)
    high_idx = findall(g -> g >= T(0.5), G_vals)

    Sigma1 = isempty(low_idx) ? cov(model.U) : cov(model.U[low_idx, :])
    Sigma2 = isempty(high_idx) ? cov(model.U) : cov(model.U[high_idx, :])
    Sigma1 += eps(T) * I
    Sigma2 += eps(T) * I

    # Identify B₀
    B0, Q, Lambda_raw = _eigendecomposition_id(Sigma1, Sigma2)
    Lambda = max.(Lambda_raw, eps(T))

    # Optimize gamma and c
    function st_loglik(params::Vector{T2}) where {T2}
        gam = exp(params[1])
        cc = params[2]

        ll = zero(T2)
        for t in 1:T_obs
            G_t = _logistic_transition(s[t], gam, cc)
            D_t = Diagonal(ones(T2, n) .+ G_t .* (Lambda .- one(T2)))
            Sigma_t = B0 * D_t * B0'
            Sigma_t = Symmetric(Sigma_t)

            ld = logdet_safe(Sigma_t)
            u = @view model.U[t, :]
            Sigma_t_inv = robust_inv(Matrix(Sigma_t))
            ll -= T2(0.5) * (n * log(T2(2π)) + ld + dot(u, Sigma_t_inv * u))
        end
        -ll
    end

    params0 = [log(gamma_init), c_init]
    result = Optim.optimize(st_loglik, params0, Optim.NelderMead(),
                            Optim.Options(iterations=max_iter))

    p_opt = Optim.minimizer(result)
    gamma_opt = exp(p_opt[1])
    c_opt = p_opt[2]
    G_final = [_logistic_transition(s[t], gamma_opt, c_opt) for t in 1:T_obs]
    loglik_final = -Optim.minimum(result)

    SmoothTransitionSVARResult{T}(B0, Q, [Sigma1, Sigma2], [ones(T, n), Lambda],
                                   gamma_opt, c_opt, s, G_final, loglik_final,
                                   Optim.converged(result), Optim.iterations(result))
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

**Reference**: Rigobon (2003)
"""
function identify_external_volatility(model::VARModel{T},
                                       regime_indicator::AbstractVector{Int};
                                       regimes::Int=2) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)
    K = regimes

    @assert length(regime_indicator) >= T_obs "regime_indicator must have length ≥ T_obs"

    # Split sample by regime
    regime_indices = [findall(regime_indicator[1:T_obs] .== k) for k in 1:K]

    Sigma_regimes = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        idx = regime_indices[k]
        if length(idx) < n + 1
            Sigma_regimes[k] = cov(model.U) + eps(T) * I
        else
            Sigma_regimes[k] = cov(model.U[idx, :]) + eps(T) * I
        end
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
