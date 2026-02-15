"""
Likelihood Ratio (LR) and Lagrange Multiplier (LM/Score) tests for nested model comparison.

The classical "trinity" of specification tests (Wald, LR, LM) provides asymptotically
equivalent tests under the null hypothesis. This module implements:

- **LR test**: Generic test for any model pair with `loglikelihood`, `dof`, `nobs`.
  LR = -2(ℓ_R - ℓ_U) ~ χ²(df) where df = dof_U - dof_R.

- **LM test**: Score-based test evaluated at the restricted model. Model-family specific
  (ARIMA, VAR, ARCH/GARCH). Uses numerical score and Hessian computation.
  LM = s'(-H)⁻¹s ~ χ²(df).

# References
- Wilks (1938). "The Large-Sample Distribution of the Likelihood Ratio."
- Neyman & Pearson (1933). "On the Problem of the Most Efficient Tests."
- Rao (1948). "Large Sample Tests of Statistical Hypotheses."
- Silvey (1959). "The Lagrangian Multiplier Test."
"""

# =============================================================================
# Result Types
# =============================================================================

"""
    LRTestResult{T} <: StatsAPI.HypothesisTest

Result from a likelihood ratio test comparing nested models.

# Fields
- `statistic::T`: LR statistic = -2(ℓ_R - ℓ_U)
- `pvalue::T`: p-value from χ²(df) distribution
- `df::Int`: Degrees of freedom (dof_U - dof_R)
- `loglik_restricted::T`: Log-likelihood of restricted model
- `loglik_unrestricted::T`: Log-likelihood of unrestricted model
- `dof_restricted::Int`: Parameters in restricted model
- `dof_unrestricted::Int`: Parameters in unrestricted model
- `nobs_restricted::Int`: Observations in restricted model
- `nobs_unrestricted::Int`: Observations in unrestricted model
"""
struct LRTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    df::Int
    loglik_restricted::T
    loglik_unrestricted::T
    dof_restricted::Int
    dof_unrestricted::Int
    nobs_restricted::Int
    nobs_unrestricted::Int
end

"""
    LMTestResult{T} <: StatsAPI.HypothesisTest

Result from a Lagrange multiplier (score) test comparing nested models.

# Fields
- `statistic::T`: LM statistic = s'(-H)⁻¹s
- `pvalue::T`: p-value from χ²(df) distribution
- `df::Int`: Degrees of freedom (dof_U - dof_R)
- `nobs::Int`: Number of observations
- `score_norm::T`: ‖s‖₂ diagnostic (Euclidean norm of score vector)
"""
struct LMTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    df::Int
    nobs::Int
    score_norm::T
end

# StatsAPI interface
StatsAPI.nobs(r::LRTestResult) = r.nobs_unrestricted
StatsAPI.nobs(r::LMTestResult) = r.nobs
StatsAPI.dof(r::LRTestResult) = r.df
StatsAPI.dof(r::LMTestResult) = r.df

# =============================================================================
# Numerical Score (Central Differences)
# =============================================================================

"""Compute score vector s = -∇(negloglik) via central finite differences."""
function _numerical_score(f::Function, x::Vector{T}; eps_step::T=T(1e-5)) where {T<:AbstractFloat}
    n = length(x)
    g = Vector{T}(undef, n)
    @inbounds for i in 1:n
        ei = zeros(T, n)
        ei[i] = eps_step
        g[i] = (f(x + ei) - f(x - ei)) / (2 * eps_step)
    end
    -g  # negate: score = -∇(negloglik)
end

# =============================================================================
# LR Test — Generic
# =============================================================================

"""
    lr_test(m1, m2) -> LRTestResult

Likelihood ratio test for nested models.

Computes LR = -2(ℓ_R - ℓ_U) where the restricted model has fewer parameters.
Under H₀ (restricted model is correct), LR ~ χ²(df) where df = dof_U - dof_R.

Works for **any** two models implementing `loglikelihood`, `dof`, and `nobs`
from StatsAPI. Automatically determines which model is restricted by comparing
degrees of freedom.

# Arguments
- `m1`, `m2`: Two fitted models (order does not matter)

# Returns
[`LRTestResult`](@ref) with test statistic, p-value, and model details.

# Example
```julia
ar2 = estimate_ar(y, 2; method=:mle)
ar4 = estimate_ar(y, 4; method=:mle)
result = lr_test(ar2, ar4)
```
"""
function lr_test(m1, m2)
    ll1, ll2 = loglikelihood(m1), loglikelihood(m2)
    d1, d2 = dof(m1), dof(m2)
    n1, n2 = nobs(m1), nobs(m2)

    T = promote_type(typeof(ll1), typeof(ll2))

    d1 == d2 && throw(ArgumentError("Models have the same number of parameters (dof=$d1); cannot determine nesting"))

    # Determine restricted vs unrestricted by dof
    if d1 < d2
        ll_R, ll_U = T(ll1), T(ll2)
        dof_R, dof_U = d1, d2
        n_R, n_U = n1, n2
    else
        ll_R, ll_U = T(ll2), T(ll1)
        dof_R, dof_U = d2, d1
        n_R, n_U = n2, n1
    end

    n_R != n_U && @warn "Models have different number of observations ($n_R vs $n_U); LR test may be unreliable"

    df = dof_U - dof_R
    LR = -2 * (ll_R - ll_U)

    if LR < zero(T)
        @warn "Negative LR statistic ($(_fmt(LR))); clamping to 0. The unrestricted model has lower likelihood than the restricted model."
        LR = zero(T)
    end

    pval = ccdf(Chisq(df), LR)

    LRTestResult{T}(LR, T(pval), df, ll_R, ll_U, dof_R, dof_U, n_R, n_U)
end

# =============================================================================
# LM Test — Internal Engine
# =============================================================================

"""Core LM test computation: score and Hessian at restricted parameters."""
function _lm_test_core(negloglik::Function, theta_embedded::Vector{T}, df::Int, n::Int) where {T<:AbstractFloat}
    # Compute score at restricted (embedded) parameters
    s = _numerical_score(negloglik, theta_embedded)
    score_norm = norm(s)

    # Compute Hessian at restricted parameters
    H = _numerical_hessian(negloglik, theta_embedded)

    # LM = s' * (-H)^{-1} * s  (H is Hessian of negloglik, so -H is neg. Hessian of loglik)
    # Since negloglik Hessian H ≈ Fisher info, we want s' * H^{-1} * s
    Hinv = robust_inv(H)
    LM = dot(s, Hinv * s)
    LM = max(LM, zero(T))

    pval = ccdf(Chisq(df), LM)

    LMTestResult{T}(LM, T(pval), df, n, score_norm)
end

# =============================================================================
# LM Test — ARIMA Family
# =============================================================================

"""Extract optimization parameters from an ARIMA-class model."""
function _arima_to_optim_params(m::AbstractARIMAModel)
    T = eltype(m.y)
    phi = m isa ARModel ? m.phi : (m isa MAModel ? T[] : m.phi)
    theta = m isa ARModel ? T[] : (m isa MAModel ? m.theta : m.theta)
    _pack_arma_params(m.c, phi, theta; include_intercept=true, log_sigma2=log(m.sigma2))
end

"""Embed restricted ARIMA parameters into unrestricted parameter space (zero-pad)."""
function _embed_arima_params(theta_R::Vector{T}, p_R::Int, q_R::Int, p_U::Int, q_U::Int) where {T}
    # Layout: [c, φ₁..φ_pR, θ₁..θ_qR, log(σ²)]
    c = theta_R[1]
    phi_R = theta_R[2:1+p_R]
    theta_R_ma = theta_R[2+p_R:1+p_R+q_R]
    log_s2 = theta_R[end]

    # Zero-pad to unrestricted dimensions
    phi_U = vcat(phi_R, zeros(T, p_U - p_R))
    theta_U = vcat(theta_R_ma, zeros(T, q_U - q_R))

    vcat(c, phi_U, theta_U, log_s2)
end

"""LM test for ARIMA-class model pairs."""
function _lm_arima(m_R::AbstractARIMAModel, m_U::AbstractARIMAModel)
    T = eltype(m_R.y)
    p_R, q_R = ar_order(m_R), ma_order(m_R)
    p_U, q_U = ar_order(m_U), ma_order(m_U)
    d_R, d_U = diff_order(m_R), diff_order(m_U)

    d_R != d_U && throw(ArgumentError("ARIMA models must have the same differencing order d (got d=$d_R vs d=$d_U)"))

    # Get working series (differenced if ARIMA)
    y = m_R isa ARIMAModel ? m_R.y_diff : m_R.y

    # Verify same data
    y_U = m_U isa ARIMAModel ? m_U.y_diff : m_U.y
    length(y) == length(y_U) && y == y_U || throw(ArgumentError("Models must be estimated on the same data"))

    # Embed restricted params into unrestricted space
    theta_R = _arima_to_optim_params(m_R)
    theta_embedded = _embed_arima_params(theta_R, p_R, q_R, p_U, q_U)

    # Negative log-likelihood in unrestricted parameterization
    negloglik = params -> _arma_negloglik(params, y, p_U, q_U; include_intercept=true)

    df = dof(m_U) - dof(m_R)
    _lm_test_core(negloglik, theta_embedded, df, length(y))
end

# =============================================================================
# LM Test — VAR Family
# =============================================================================

"""LM test for VAR model pairs (different lag orders)."""
function _lm_var(m_R::VARModel{T}, m_U::VARModel{T}) where {T}
    # Verify same underlying data
    m_R.Y == m_U.Y || throw(ArgumentError("VAR models must be estimated on the same data matrix Y"))

    n = nvars(m_U)
    p_U = m_U.p

    # Construct unrestricted design matrices
    Y_eff, X = construct_var_matrices(m_U.Y, p_U)
    T_eff = size(Y_eff, 1)
    k = size(X, 2)  # 1 + n * p_U

    # Embed restricted B into unrestricted space (zero-pad extra lag coefficients)
    B_embedded = zeros(T, k, n)
    k_R = size(m_R.B, 1)  # 1 + n * p_R
    B_embedded[1:k_R, :] .= m_R.B

    # Concentrated Gaussian negloglik: negloglik(vec(B))
    function negloglik(b_vec::Vector{T2}) where {T2}
        B_mat = reshape(b_vec, k, n)
        U = Y_eff - X * B_mat
        Sigma_ml = (U' * U) / T_eff
        ld = logdet_safe(Sigma_ml)
        T2(T_eff * n / 2) * log(T2(2π)) + T2(T_eff / 2) * ld + T2(T_eff * n / 2)
    end

    theta_embedded = vec(B_embedded)
    df = dof(m_U) - dof(m_R)
    _lm_test_core(negloglik, theta_embedded, df, T_eff)
end

# =============================================================================
# LM Test — Volatility Family
# =============================================================================

"""Convert ARCH model parameters to optimization space [μ, log(ω), log(α)...]."""
function _arch_to_optim_params(m::ARCHModel{T}) where {T}
    vcat(m.mu, log(m.omega), log.(m.alpha))
end

"""Convert GARCH model parameters to optimization space [μ, log(ω), log(α)..., log(β)...]."""
function _garch_to_optim_params(m::GARCHModel{T}) where {T}
    vcat(m.mu, log(m.omega), log.(m.alpha), log.(m.beta))
end

"""Convert EGARCH model parameters to optimization space [μ, ω, α..., γ..., β...] (unconstrained)."""
function _egarch_to_optim_params(m::EGARCHModel{T}) where {T}
    vcat(m.mu, m.omega, m.alpha, m.gamma, m.beta)
end

"""Convert GJR-GARCH model parameters to optimization space [μ, log(ω), log(α)..., log(γ)..., log(β)...]."""
function _gjr_to_optim_params(m::GJRGARCHModel{T}) where {T}
    vcat(m.mu, log(m.omega), log.(m.alpha), log.(m.gamma), log.(m.beta))
end

"""Set up LM test for volatility model pairs. Returns (negloglik, theta_embedded, df)."""
function _setup_volatility_lm(m_R::AbstractVolatilityModel, m_U::AbstractVolatilityModel, y::Vector{T}) where {T}
    # ARCH → ARCH (different q)
    if m_R isa ARCHModel && m_U isa ARCHModel
        q_R, q_U = m_R.q, m_U.q
        theta_R = _arch_to_optim_params(m_R)
        # Pad alpha with T(-30) ≈ exp(-30) ≈ 0
        theta_embedded = vcat(theta_R[1:2], theta_R[3:end], fill(T(-30), q_U - q_R))
        negloglik = params -> _arch_negloglik(params, y, q_U)
        df = dof(m_U) - dof(m_R)
        return negloglik, theta_embedded, df

    # GARCH → GARCH (different p or q)
    elseif m_R isa GARCHModel && m_U isa GARCHModel
        p_R, q_R = m_R.p, m_R.q
        p_U, q_U = m_U.p, m_U.q
        theta_R = _garch_to_optim_params(m_R)
        # Layout: [mu, log(omega), log(alpha_1..q_R), log(beta_1..p_R)]
        mu_logomega = theta_R[1:2]
        log_alpha_R = theta_R[3:2+q_R]
        log_beta_R = theta_R[3+q_R:end]
        log_alpha_U = vcat(log_alpha_R, fill(T(-30), q_U - q_R))
        log_beta_U = vcat(log_beta_R, fill(T(-30), p_U - p_R))
        theta_embedded = vcat(mu_logomega, log_alpha_U, log_beta_U)
        negloglik = params -> _garch_negloglik(params, y, p_U, q_U)
        df = dof(m_U) - dof(m_R)
        return negloglik, theta_embedded, df

    # ARCH → GARCH (ARCH is GARCH with p=0)
    elseif m_R isa ARCHModel && m_U isa GARCHModel
        q_R = m_R.q
        p_U, q_U = m_U.p, m_U.q
        theta_R = _arch_to_optim_params(m_R)
        mu_logomega = theta_R[1:2]
        log_alpha_R = theta_R[3:end]
        log_alpha_U = vcat(log_alpha_R, fill(T(-30), q_U - q_R))
        log_beta_U = fill(T(-30), p_U)
        theta_embedded = vcat(mu_logomega, log_alpha_U, log_beta_U)
        negloglik = params -> _garch_negloglik(params, y, p_U, q_U)
        df = dof(m_U) - dof(m_R)
        return negloglik, theta_embedded, df

    # EGARCH → EGARCH (different p or q)
    elseif m_R isa EGARCHModel && m_U isa EGARCHModel
        p_R, q_R = m_R.p, m_R.q
        p_U, q_U = m_U.p, m_U.q
        theta_R = _egarch_to_optim_params(m_R)
        # Layout: [mu, omega, alpha_1..q_R, gamma_1..q_R, beta_1..p_R]
        mu_omega = theta_R[1:2]
        alpha_R = theta_R[3:2+q_R]
        gamma_R = theta_R[3+q_R:2+2q_R]
        beta_R = theta_R[3+2q_R:end]
        alpha_U = vcat(alpha_R, zeros(T, q_U - q_R))
        gamma_U = vcat(gamma_R, zeros(T, q_U - q_R))
        beta_U = vcat(beta_R, zeros(T, p_U - p_R))
        theta_embedded = vcat(mu_omega, alpha_U, gamma_U, beta_U)
        negloglik = params -> _egarch_negloglik(params, y, p_U, q_U)
        df = dof(m_U) - dof(m_R)
        return negloglik, theta_embedded, df

    # GJR-GARCH → GJR-GARCH (different p or q)
    elseif m_R isa GJRGARCHModel && m_U isa GJRGARCHModel
        p_R, q_R = m_R.p, m_R.q
        p_U, q_U = m_U.p, m_U.q
        theta_R = _gjr_to_optim_params(m_R)
        # Layout: [mu, log(omega), log(alpha_1..q_R), log(gamma_1..q_R), log(beta_1..p_R)]
        mu_logomega = theta_R[1:2]
        log_alpha_R = theta_R[3:2+q_R]
        log_gamma_R = theta_R[3+q_R:2+2q_R]
        log_beta_R = theta_R[3+2q_R:end]
        log_alpha_U = vcat(log_alpha_R, fill(T(-30), q_U - q_R))
        log_gamma_U = vcat(log_gamma_R, fill(T(-30), q_U - q_R))
        log_beta_U = vcat(log_beta_R, fill(T(-30), p_U - p_R))
        theta_embedded = vcat(mu_logomega, log_alpha_U, log_gamma_U, log_beta_U)
        negloglik = params -> _gjr_negloglik(params, y, p_U, q_U)
        df = dof(m_U) - dof(m_R)
        return negloglik, theta_embedded, df

    else
        throw(ArgumentError("LM test not supported for $(nameof(typeof(m_R))) × $(nameof(typeof(m_U))). " *
            "Supported: ARCH×ARCH, GARCH×GARCH, ARCH×GARCH, EGARCH×EGARCH, GJR×GJR."))
    end
end

"""LM test for volatility model pairs."""
function _lm_volatility(m_R::AbstractVolatilityModel, m_U::AbstractVolatilityModel)
    T = eltype(m_R.y)
    m_R.y == m_U.y || throw(ArgumentError("Models must be estimated on the same data"))
    negloglik, theta_embedded, df = _setup_volatility_lm(m_R, m_U, m_R.y)
    _lm_test_core(negloglik, theta_embedded, df, length(m_R.y))
end

# =============================================================================
# LM Test — Public Interface
# =============================================================================

"""
    lm_test(m1, m2) -> LMTestResult

Lagrange multiplier (score) test for nested models.

Evaluates the score of the unrestricted log-likelihood at the restricted parameter
estimates. Under H₀, LM = s'(-H)⁻¹s ~ χ²(df) where df = dof_U - dof_R.

Automatically determines restricted/unrestricted by comparing `dof`. Dispatches
to model-family specific implementations for ARIMA, VAR, and volatility models.

# Supported model pairs
- `AbstractARIMAModel` × `AbstractARIMAModel` (same differencing order `d`)
- `VARModel` × `VARModel` (different lag orders, same data)
- `ARCHModel` × `ARCHModel`, `GARCHModel` × `GARCHModel`
- `ARCHModel` × `GARCHModel` (cross-type nesting)
- `EGARCHModel` × `EGARCHModel`, `GJRGARCHModel` × `GJRGARCHModel`

# Arguments
- `m1`, `m2`: Two fitted models from the same family (order does not matter)

# Returns
[`LMTestResult`](@ref) with test statistic, p-value, and score norm diagnostic.

# Example
```julia
ar2 = estimate_ar(y, 2; method=:mle)
ar4 = estimate_ar(y, 4; method=:mle)
result = lm_test(ar2, ar4)
```
"""
function lm_test(m1, m2)
    d1, d2 = dof(m1), dof(m2)
    d1 == d2 && throw(ArgumentError("Models have the same number of parameters (dof=$d1); cannot determine nesting"))

    # Order: restricted has fewer dof
    m_R, m_U = d1 < d2 ? (m1, m2) : (m2, m1)

    # Dispatch by model family
    if m_R isa AbstractARIMAModel && m_U isa AbstractARIMAModel
        return _lm_arima(m_R, m_U)
    elseif m_R isa VARModel && m_U isa VARModel
        return _lm_var(m_R, m_U)
    elseif m_R isa AbstractVolatilityModel && m_U isa AbstractVolatilityModel
        return _lm_volatility(m_R, m_U)
    else
        throw(ArgumentError(
            "LM test not supported for $(nameof(typeof(m_R))) × $(nameof(typeof(m_U))). " *
            "Supported families: ARIMA, VAR, ARCH/GARCH/EGARCH/GJR-GARCH."
        ))
    end
end

# =============================================================================
# Display Methods
# =============================================================================

function Base.show(io::IO, r::LRTestResult)
    stars = _significance_stars(r.pvalue)
    spec_data = Any[
        "H₀"  "Restricted model is adequate";
        "H₁"  "Unrestricted model fits significantly better";
        "Restricted dof"    r.dof_restricted;
        "Unrestricted dof"  r.dof_unrestricted;
        "Restricted nobs"   r.nobs_restricted;
        "Unrestricted nobs" r.nobs_unrestricted
    ]
    _pretty_table(io, spec_data;
        title = "Likelihood Ratio Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    results_data = Any[
        "Log-likelihood (restricted)"    _fmt(r.loglik_restricted; digits=4);
        "Log-likelihood (unrestricted)"  _fmt(r.loglik_unrestricted; digits=4);
        "LR statistic"    string(_fmt(r.statistic), " ", stars);
        "Degrees of freedom"  r.df;
        "P-value"          _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )

    conclusion = if r.pvalue < 0.01
        "Reject H₀ at 1% level — unrestricted model preferred"
    elseif r.pvalue < 0.05
        "Reject H₀ at 5% level — unrestricted model preferred"
    elseif r.pvalue < 0.10
        "Reject H₀ at 10% level — unrestricted model preferred"
    else
        "Fail to reject H₀ — restricted model is adequate"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::LMTestResult)
    stars = _significance_stars(r.pvalue)
    spec_data = Any[
        "H₀"  "Restricted model is adequate";
        "H₁"  "Unrestricted model fits significantly better";
        "Observations"  r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Lagrange Multiplier (Score) Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    results_data = Any[
        "LM statistic"       string(_fmt(r.statistic), " ", stars);
        "Degrees of freedom"  r.df;
        "P-value"             _format_pvalue(r.pvalue);
        "Score norm (‖s‖₂)"  _fmt(r.score_norm)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )

    conclusion = if r.pvalue < 0.01
        "Reject H₀ at 1% level — unrestricted model preferred"
    elseif r.pvalue < 0.05
        "Reject H₀ at 5% level — unrestricted model preferred"
    elseif r.pvalue < 0.10
        "Reject H₀ at 10% level — unrestricted model preferred"
    else
        "Fail to reject H₀ — restricted model is adequate"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end
