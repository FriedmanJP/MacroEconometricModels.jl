"""
Non-Gaussian maximum likelihood SVAR identification.

Estimates B₀ and distribution parameters jointly by maximizing the log-likelihood
under non-Gaussian shock distributions: Student-t, mixture of normals, PML (Pearson Type IV),
skew-normal, or a unified dispatcher.

References:
- Lanne, M., Meitz, M. & Saikkonen, P. (2017). "Identification and estimation of non-Gaussian SVAR."
- Lanne, M. & Lütkepohl, H. (2010). "Structural VAR analysis in a data-rich environment."
- Herwartz, H. (2018). "Hodges-Lehmann detection of structural shocks."
- Azzalini, A. (1985). "A class of distributions which includes the normal ones."
"""

using LinearAlgebra, Statistics, Distributions
import Optim

# =============================================================================
# Result Type
# =============================================================================

"""
    NonGaussianMLResult{T} <: AbstractNonGaussianSVAR

Result from non-Gaussian maximum likelihood SVAR identification.

Fields:
- `B0::Matrix{T}` — structural impact matrix (n × n)
- `Q::Matrix{T}` — rotation matrix
- `shocks::Matrix{T}` — structural shocks (T_eff × n)
- `distribution::Symbol` — `:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`
- `loglik::T` — log-likelihood at MLE
- `loglik_gaussian::T` — Gaussian log-likelihood (for LR test)
- `dist_params::Dict{Symbol, Any}` — distribution parameters
- `vcov::Matrix{T}` — asymptotic covariance of B₀ elements
- `se::Matrix{T}` — standard errors for B₀
- `converged::Bool`
- `iterations::Int`
- `aic::T`
- `bic::T`
"""
struct NonGaussianMLResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    shocks::Matrix{T}
    distribution::Symbol
    loglik::T
    loglik_gaussian::T
    dist_params::Dict{Symbol, Any}
    vcov::Matrix{T}
    se::Matrix{T}
    converged::Bool
    iterations::Int
    aic::T
    bic::T
end

function Base.show(io::IO, r::NonGaussianMLResult{T}) where {T}
    n = size(r.B0, 1)
    lr_stat = T(2) * (r.loglik - r.loglik_gaussian)
    lr_stat = max(lr_stat, zero(T))
    # Degrees of freedom: number of additional distribution parameters
    n_dist_params = length(get(r.dist_params, :all_params, T[]))
    lr_df = max(n_dist_params, 1)
    lr_pval = lr_stat > 0 ? T(1) - cdf(Chisq(lr_df), lr_stat) : one(T)
    lr_stars = _significance_stars(lr_pval)

    spec = Any[
        "Distribution"   string(r.distribution);
        "Variables"      n;
        "Log-likelihood" _fmt(r.loglik; digits=4);
        "Log-lik (Gaussian)" _fmt(r.loglik_gaussian; digits=4);
        "AIC"            _fmt(r.aic; digits=2);
        "BIC"            _fmt(r.bic; digits=2);
        "Converged"      r.converged ? "Yes" : "No";
        "Iterations"     r.iterations
    ]
    _pretty_table(io, spec;
        title = "Non-Gaussian ML Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    # LR test against Gaussian
    lr_data = Any[
        "LR statistic" string(_fmt(lr_stat; digits=2), " ", lr_stars);
        "P-value"      _format_pvalue(lr_pval);
        "H₀"           "Gaussian shocks (standard SVAR)"
    ]
    _pretty_table(io, lr_data;
        title = "LR Test: Non-Gaussian vs. Gaussian",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _matrix_table(io, r.B0, "Structural Impact Matrix (B₀)";
        row_labels=["Var $i" for i in 1:n],
        col_labels=["Shock $j" for j in 1:n])
end

# =============================================================================
# Log-pdf Functions
# =============================================================================

"""Student-t log-pdf (standardized to unit variance)."""
function _student_t_logpdf(x::T, nu::T) where {T<:AbstractFloat}
    # Standardized so Var = 1: scale by sqrt(nu/(nu-2))
    nu_safe = max(nu, T(2.01))
    s = sqrt(nu_safe / (nu_safe - T(2)))
    z = x * s
    logpdf(TDist(nu_safe), z) + log(s)
end

"""Mixture of two normals log-pdf: p*N(0,σ₁²) + (1-p)*N(0,σ₂²) with unit variance constraint."""
function _mixture_normal_logpdf(x::T, p_mix::T, sigma1::T, sigma2::T) where {T<:AbstractFloat}
    # Enforce unit variance: p σ₁² + (1-p) σ₂² = 1
    d1 = p_mix * pdf(Normal(zero(T), sigma1), x)
    d2 = (one(T) - p_mix) * pdf(Normal(zero(T), sigma2), x)
    log(max(d1 + d2, eps(T)))
end

"""Skew-normal log-pdf: f(x) = 2 φ(x) Φ(αx)."""
function _skew_normal_logpdf(x::T, alpha::T) where {T<:AbstractFloat}
    log(T(2)) + logpdf(Normal(), x) + logcdf(Normal(), alpha * x)
end

"""Pearson Type IV log-pdf for PML (approximation via scaled-t with skewness)."""
function _pearson_iv_logpdf(x::T, kappa::T, nu::T) where {T<:AbstractFloat}
    # Simplified Pearson IV: use Student-t with additional kurtosis parameter
    nu_safe = max(nu, T(2.01))
    _student_t_logpdf(x, nu_safe) + kappa * (x^3 / T(6))  # skewness correction
end

# =============================================================================
# Gaussian Reference Log-Likelihood
# =============================================================================

"""Compute Gaussian log-likelihood for a VAR model (for LR test comparison)."""
function _gaussian_loglik(model::VARModel{T}) where {T<:AbstractFloat}
    T_obs = size(model.U, 1)
    n = nvars(model)
    ld = logdet_safe(model.Sigma)
    -T_obs / 2 * (n * log(T(2π)) + ld) - T_obs / 2 * n  # trace term = n under MLE
end

# =============================================================================
# Unified Non-Gaussian Log-Likelihood
# =============================================================================

"""
Full non-Gaussian log-likelihood:
ℓ(θ) = Σ_t [log|det(B₀⁻¹)| + Σ_j log f_j(ε_{j,t}; θ_j)]
where ε_t = B₀⁻¹ u_t.
"""
function _nongaussian_loglik(angles::AbstractVector{T}, dist_params_vec::AbstractVector{T},
                             U::Matrix{T}, L::LowerTriangular{T, Matrix{T}}, n::Int;
                             distribution::Symbol=:student_t) where {T<:AbstractFloat}
    Q = _givens_to_orthogonal(angles, n)
    B0 = Matrix(L) * Q
    B0_inv = robust_inv(B0)
    shocks = (B0_inv * U')'

    T_obs = size(U, 1)
    loglik = T_obs * log(max(abs(det(B0_inv)), eps(T)))

    if distribution == :student_t
        for j in 1:n
            nu = max(exp(dist_params_vec[j]) + T(2.01), T(2.01))  # ensure nu > 2
            for t in 1:T_obs
                loglik += _student_t_logpdf(shocks[t, j], nu)
            end
        end
    elseif distribution == :mixture_normal
        for j in 1:n
            idx = (j - 1) * 3
            p_mix = one(T) / (one(T) + exp(-dist_params_vec[idx + 1]))  # logistic transform
            log_s1 = dist_params_vec[idx + 2]
            sigma1 = exp(log_s1)
            # Enforce unit variance: p σ₁² + (1-p) σ₂² = 1
            sigma2_sq = (one(T) - p_mix * sigma1^2) / max(one(T) - p_mix, eps(T))
            sigma2 = sigma2_sq > zero(T) ? sqrt(sigma2_sq) : eps(T)
            for t in 1:T_obs
                loglik += _mixture_normal_logpdf(shocks[t, j], p_mix, sigma1, sigma2)
            end
        end
    elseif distribution == :pml
        for j in 1:n
            idx = (j - 1) * 2
            kappa = dist_params_vec[idx + 1]
            nu = max(exp(dist_params_vec[idx + 2]) + T(2.01), T(2.01))
            for t in 1:T_obs
                loglik += _pearson_iv_logpdf(shocks[t, j], kappa, nu)
            end
        end
    elseif distribution == :skew_normal
        for j in 1:n
            alpha = dist_params_vec[j]
            for t in 1:T_obs
                loglik += _skew_normal_logpdf(shocks[t, j], alpha)
            end
        end
    end

    loglik
end

# =============================================================================
# Parameter Packing/Unpacking
# =============================================================================

"""Pack rotation angles and distribution parameters into single vector."""
function _pack_nongaussian_params(angles::Vector{T}, dist_params::Vector{T}) where {T}
    vcat(angles, dist_params)
end

"""Unpack single vector into rotation angles and distribution parameters."""
function _unpack_nongaussian_params(params::Vector{T}, n::Int, distribution::Symbol) where {T}
    n_angles = n * (n - 1) ÷ 2
    angles = params[1:n_angles]
    dist_params = params[n_angles+1:end]
    (angles, dist_params)
end

"""Determine number of distribution parameters per shock."""
function _n_dist_params(distribution::Symbol)
    distribution == :student_t ? 1 :
    distribution == :mixture_normal ? 3 :
    distribution == :pml ? 2 :
    distribution == :skew_normal ? 1 :
    throw(ArgumentError("Unknown distribution: $distribution"))
end

# =============================================================================
# Numerical Hessian for Standard Errors
# =============================================================================

"""Compute numerical Hessian via central finite differences."""
function _numerical_hessian(f::Function, x::Vector{T}; eps_step::T=T(1e-5)) where {T<:AbstractFloat}
    n = length(x)
    H = zeros(T, n, n)
    f0 = f(x)

    for i in 1:n
        ei = zeros(T, n)
        ei[i] = eps_step
        for j in i:n
            ej = zeros(T, n)
            ej[j] = eps_step
            fpp = f(x + ei + ej)
            fpm = f(x + ei - ej)
            fmp = f(x - ei + ej)
            fmm = f(x - ei - ej)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps_step^2)
            H[j, i] = H[i, j]
        end
    end
    H
end

"""Compute asymptotic covariance and standard errors from Hessian."""
function _nongaussian_vcov(params::Vector{T}, U::Matrix{T}, L::LowerTriangular{T, Matrix{T}},
                           n::Int, distribution::Symbol) where {T<:AbstractFloat}
    n_angles = n * (n - 1) ÷ 2
    n_dp = _n_dist_params(distribution) * n
    n_total = n_angles + n_dp

    obj = p -> begin
        angles, dp = _unpack_nongaussian_params(p, n, distribution)
        -_nongaussian_loglik(angles, dp, U, L, n; distribution=distribution)
    end

    H = _numerical_hessian(obj, params)

    # Try to invert Hessian for vcov
    vcov_full = try
        robust_inv(H)
    catch
        zeros(T, n_total, n_total)
    end

    # Extract B₀ standard errors (from angle parameters only)
    vcov_angles = vcov_full[1:n_angles, 1:n_angles]
    se_B0 = zeros(T, n, n)

    # Approximate: propagate angle uncertainty to B₀ via numerical Jacobian
    angles_opt = params[1:n_angles]
    Q0 = _givens_to_orthogonal(angles_opt, n)
    B0 = Matrix(L) * Q0

    for idx in 1:n_angles
        angles_p = copy(angles_opt)
        angles_p[idx] += T(1e-6)
        Q_p = _givens_to_orthogonal(angles_p, n)
        B0_p = Matrix(L) * Q_p
        dB = (B0_p - B0) / T(1e-6)
        for i in 1:n, j in 1:n
            se_B0[i, j] += dB[i, j]^2 * max(vcov_angles[idx, idx], zero(T))
        end
    end
    se_B0 .= sqrt.(se_B0)

    (vcov_full, se_B0)
end

# =============================================================================
# Core Estimation
# =============================================================================

"""Internal: estimate non-Gaussian ML SVAR for a given distribution."""
function _estimate_nongaussian_ml(model::VARModel{T}, distribution::Symbol;
                                  max_iter::Int=500, tol::T=T(1e-6),
                                  dist_init::Union{Nothing, Vector{T}}=nothing) where {T<:AbstractFloat}
    n = nvars(model)
    L = safe_cholesky(model.Sigma)

    n_angles = n * (n - 1) ÷ 2
    n_dp = _n_dist_params(distribution)
    n_dp_total = n_dp * n

    # Initialize
    angles0 = zeros(T, n_angles)
    if isnothing(dist_init)
        if distribution == :student_t
            dist_params0 = fill(log(T(5.0) - T(2.01)), n)  # ν ≈ 5
        elseif distribution == :mixture_normal
            dist_params0 = repeat([T(0.0), T(0.0), T(0.0)], n)  # logistic(0)=0.5, log(1)=0
        elseif distribution == :pml
            dist_params0 = repeat([T(0.0), log(T(5.0) - T(2.01))], n)  # κ=0, ν≈5
        elseif distribution == :skew_normal
            dist_params0 = zeros(T, n)  # α=0 (no skewness initially)
        end
    else
        dist_params0 = dist_init
    end

    params0 = _pack_nongaussian_params(angles0, dist_params0)

    obj = p -> begin
        angles, dp = _unpack_nongaussian_params(p, n, distribution)
        -_nongaussian_loglik(angles, dp, model.U, L, n; distribution=distribution)
    end

    result = Optim.optimize(obj, params0, Optim.NelderMead(),
                            Optim.Options(iterations=max_iter, g_tol=tol,
                                          show_trace=false))

    params_opt = Optim.minimizer(result)
    angles_opt, dp_opt = _unpack_nongaussian_params(params_opt, n, distribution)

    Q = _givens_to_orthogonal(angles_opt, n)
    B0 = Matrix(L) * Q
    B0_inv = robust_inv(B0)
    shocks = (B0_inv * model.U')'

    loglik = -Optim.minimum(result)
    loglik_g = _gaussian_loglik(model)

    # Standard errors
    vcov_mat, se_mat = _nongaussian_vcov(params_opt, model.U, L, n, distribution)

    # Distribution parameters dict
    dp_dict = Dict{Symbol, Any}()
    T_obs = size(model.U, 1)
    if distribution == :student_t
        dp_dict[:nu] = [max(exp(dp_opt[j]) + T(2.01), T(2.01)) for j in 1:n]
    elseif distribution == :mixture_normal
        dp_dict[:p_mix] = [one(T) / (one(T) + exp(-dp_opt[(j-1)*3+1])) for j in 1:n]
        dp_dict[:sigma1] = [exp(dp_opt[(j-1)*3+2]) for j in 1:n]
    elseif distribution == :pml
        dp_dict[:kappa] = [dp_opt[(j-1)*2+1] for j in 1:n]
        dp_dict[:nu] = [max(exp(dp_opt[(j-1)*2+2]) + T(2.01), T(2.01)) for j in 1:n]
    elseif distribution == :skew_normal
        dp_dict[:alpha] = [dp_opt[j] for j in 1:n]
    end

    n_params = length(params_opt)
    aic_val = -T(2) * loglik + T(2) * n_params
    bic_val = -T(2) * loglik + log(T(T_obs)) * n_params

    # Normalize signs
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] *= -one(T)
            Q[:, j] *= -one(T)
            shocks[:, j] *= -one(T)
        end
    end

    NonGaussianMLResult{T}(B0, Q, shocks, distribution, loglik, loglik_g, dp_dict,
                           vcov_mat, se_mat, Optim.converged(result),
                           Optim.iterations(result), aic_val, bic_val)
end

# =============================================================================
# Public API
# =============================================================================

"""
    identify_student_t(model::VARModel; max_iter=500, tol=1e-6) -> NonGaussianMLResult

Identify SVAR assuming Student-t distributed structural shocks.

Each shock εⱼ ~ t(νⱼ) (standardized to unit variance). Identification is achieved
when at most one νⱼ = ∞ (Gaussian).

**Reference**: Lanne, Meitz & Saikkonen (2017)
"""
function identify_student_t(model::VARModel{T}; max_iter::Int=500,
                            tol::T=T(1e-6)) where {T<:AbstractFloat}
    _estimate_nongaussian_ml(model, :student_t; max_iter=max_iter, tol=tol)
end

"""
    identify_mixture_normal(model::VARModel; n_components=2, max_iter=500, tol=1e-6) -> NonGaussianMLResult

Identify SVAR assuming mixture-of-normals distributed structural shocks.

Each shock εⱼ ~ p_j N(0,σ₁ⱼ²) + (1-p_j) N(0,σ₂ⱼ²) with unit variance constraint.

**Reference**: Lanne & Lütkepohl (2010)
"""
function identify_mixture_normal(model::VARModel{T}; n_components::Int=2,
                                 max_iter::Int=500, tol::T=T(1e-6)) where {T<:AbstractFloat}
    _estimate_nongaussian_ml(model, :mixture_normal; max_iter=max_iter, tol=tol)
end

"""
    identify_pml(model::VARModel; max_iter=500, tol=1e-6) -> NonGaussianMLResult

Identify SVAR via Pseudo Maximum Likelihood using Pearson Type IV distributions.

Allows both skewness and excess kurtosis in the structural shocks.

**Reference**: Herwartz (2018)
"""
function identify_pml(model::VARModel{T}; max_iter::Int=500,
                      tol::T=T(1e-6)) where {T<:AbstractFloat}
    _estimate_nongaussian_ml(model, :pml; max_iter=max_iter, tol=tol)
end

"""
    identify_skew_normal(model::VARModel; max_iter=500, tol=1e-6) -> NonGaussianMLResult

Identify SVAR assuming skew-normal distributed structural shocks.

Each shock εⱼ has pdf f(x) = 2 φ(x) Φ(αⱼ x), where αⱼ controls skewness.

**Reference**: Azzalini (1985)
"""
function identify_skew_normal(model::VARModel{T}; max_iter::Int=500,
                              tol::T=T(1e-6)) where {T<:AbstractFloat}
    _estimate_nongaussian_ml(model, :skew_normal; max_iter=max_iter, tol=tol)
end

"""
    identify_nongaussian_ml(model::VARModel; distribution=:student_t,
                            max_iter=500, tol=1e-6) -> NonGaussianMLResult

Unified non-Gaussian ML SVAR identification dispatcher.

Supported distributions:
- `:student_t` — independent Student-t shocks (Lanne, Meitz & Saikkonen 2017)
- `:mixture_normal` — mixture of two normals (Lanne & Lütkepohl 2010)
- `:pml` — Pearson Type IV / Pseudo-ML (Herwartz 2018)
- `:skew_normal` — skew-normal (Azzalini 1985)
"""
function identify_nongaussian_ml(model::VARModel{T}; distribution::Symbol=:student_t,
                                 max_iter::Int=500, tol::T=T(1e-6)) where {T<:AbstractFloat}
    _estimate_nongaussian_ml(model, distribution; max_iter=max_iter, tol=tol)
end
