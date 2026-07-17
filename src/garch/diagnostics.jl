# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
News impact curve for GARCH-family models.
"""

"""
    news_impact_curve(m; range=(-3,3), n_points=200)

Compute the news impact curve: how a shock εₜ₋₁ maps to σ²ₜ, holding all else at unconditional values.

Returns named tuple `(shocks, variance)` where both are vectors of length `n_points`.

# Supported models
- `GARCHModel`: Symmetric parabola
- `GJRGARCHModel`: Asymmetric parabola (steeper for negative shocks)
- `EGARCHModel`: Asymmetric exponential curve
"""
function news_impact_curve(m::GARCHModel{T}; range::Tuple{Real,Real}=(-3.0,3.0), n_points::Int=200) where {T}
    sigma = sqrt(unconditional_variance(m))
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))

    # For GARCH(1,1): h_t = ω + α₁ε²_{t-1} + β₁h̄
    # where h̄ = unconditional variance
    h_bar = unconditional_variance(m)
    h_bar = isfinite(h_bar) ? h_bar : var(m.y)

    variance = map(shocks) do e
        ht = m.omega
        ht += m.alpha[1] * e^2
        for j in 1:m.p
            ht += m.beta[j] * h_bar
        end
        max(ht, eps(T))
    end

    (shocks=shocks, variance=variance)
end

function news_impact_curve(m::GJRGARCHModel{T}; range::Tuple{Real,Real}=(-3.0,3.0), n_points::Int=200) where {T}
    sigma = sqrt(unconditional_variance(m))
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))

    h_bar = unconditional_variance(m)
    h_bar = isfinite(h_bar) ? h_bar : var(m.y)

    variance = map(shocks) do e
        ht = m.omega
        indicator = e < zero(T) ? one(T) : zero(T)
        ht += (m.alpha[1] + m.gamma[1] * indicator) * e^2
        for j in 1:m.p
            ht += m.beta[j] * h_bar
        end
        max(ht, eps(T))
    end

    (shocks=shocks, variance=variance)
end

function news_impact_curve(m::EGARCHModel{T}; range::Tuple{Real,Real}=(-3.0,3.0), n_points::Int=200) where {T}
    sigma = sqrt(unconditional_variance(m))
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))
    E_abs_z = sqrt(T(2) / T(π))

    h_bar = unconditional_variance(m)
    h_bar = isfinite(h_bar) ? h_bar : var(m.y)
    log_h_bar = log(h_bar)

    variance = map(shocks) do e
        z = e / sigma
        log_ht = m.omega
        log_ht += m.alpha[1] * (abs(z) - E_abs_z) + m.gamma[1] * z
        for j in 1:m.p
            log_ht += m.beta[j] * log_h_bar
        end
        exp(clamp(log_ht, T(-50), T(50)))
    end

    (shocks=shocks, variance=variance)
end

# =============================================================================
# News impact curves — IGARCH / APARCH / Component-GARCH (EV-15, #423)
# =============================================================================

function news_impact_curve(m::IGARCHModel{T}; range::Tuple{Real,Real}=(-3.0,3.0), n_points::Int=200) where {T}
    # IGARCH has divergent unconditional variance — anchor on the sample variance.
    h_bar = var(m.y)
    sigma = sqrt(h_bar)
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))
    variance = map(shocks) do e
        ht = m.omega + m.alpha[1] * e^2
        for j in 1:m.p
            ht += m.beta[j] * h_bar
        end
        max(ht, eps(T))
    end
    (shocks=shocks, variance=variance)
end

function news_impact_curve(m::APARCHModel{T}; range::Tuple{Real,Real}=(-3.0,3.0), n_points::Int=200) where {T}
    uv = unconditional_variance(m)
    hbar = isfinite(uv) ? uv : var(m.y)
    sigbar = sqrt(hbar)
    sdelta_bar = sigbar^m.delta
    shocks = collect(LinRange(T(range[1]) * sigbar, T(range[2]) * sigbar, n_points))
    variance = map(shocks) do e
        st = m.omega + m.alpha[1] * (abs(e) - m.gamma[1] * e)^m.delta
        for j in 1:m.p
            st += m.beta[j] * sdelta_bar
        end
        max(st, eps(T))^(2 / m.delta)
    end
    (shocks=shocks, variance=variance)
end

function news_impact_curve(m::CGARCHModel{T}; range::Tuple{Real,Real}=(-3.0,3.0), n_points::Int=200) where {T}
    # Anchor permanent q̄ = ω and σ̄² = ω ⇒ h(e) = ω + α(e² − ω).
    h_bar = m.omega
    sigma = sqrt(h_bar)
    shocks = collect(LinRange(T(range[1]) * sigma, T(range[2]) * sigma, n_points))
    variance = map(shocks) do e
        ht = m.omega + m.alpha * (e^2 - m.omega)
        max(ht, eps(T))
    end
    (shocks=shocks, variance=variance)
end

# =============================================================================
# Engle–Ng (1993) sign-bias / size-bias test
# =============================================================================

"""
    sign_bias_test(z_or_model) -> NamedTuple

Engle & Ng (1993) sign-bias and size-bias diagnostic for asymmetry left in a
fitted volatility model. Regresses the squared standardized residuals `ẑ²ₜ` on a
constant and three asymmetry regressors built from the lagged standardized shock
`ẑₜ₋₁`:

    ẑ²ₜ = b₀ + b₁ S⁻ₜ₋₁ + b₂ S⁻ₜ₋₁ ẑₜ₋₁ + b₃ S⁺ₜ₋₁ ẑₜ₋₁ + uₜ

where `S⁻ₜ₋₁ = 1(ẑₜ₋₁ < 0)`, `S⁺ₜ₋₁ = 1 − S⁻ₜ₋₁`. `b₁` is the **sign bias**, `b₂`
the **negative size bias**, `b₃` the **positive size bias**. The joint test
statistic is `(n−1)·R² ~ χ²(3)` under H₀ (no remaining asymmetry). Individual
`t`-statistics use the OLS covariance.

# Arguments
- `z_or_model`: standardized-residual vector, or any `AbstractVolatilityModel`
  (uses `m.standardized_residuals`).

# Returns
Named tuple with `sign_bias`, `sign_bias_t`, `sign_bias_p`, `neg_size_t`,
`neg_size_p`, `pos_size_t`, `pos_size_p`, `joint_statistic`, `joint_pvalue`, `dof`.

# References
- Engle & Ng (1993). *Journal of Finance* 48(5), 1749–1778.
"""
function sign_bias_test(z::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(z)
    n < 10 && throw(ArgumentError("Need at least 10 observations for sign-bias test"))
    zsq = z .^ 2
    y = zsq[2:n]
    zlag = z[1:n-1]
    neff = n - 1
    Sneg = T.(zlag .< zero(T))
    Spos = one(T) .- Sneg
    X = hcat(ones(T, neff), Sneg, Sneg .* zlag, Spos .* zlag)

    XtX_inv = robust_inv(X' * X)
    b = XtX_inv * (X' * y)
    resid = y .- X * b
    ss_res = sum(abs2, resid)
    ss_tot = sum(abs2, y .- mean(y))
    r2 = one(T) - ss_res / ss_tot

    joint = T(neff) * r2
    pj = one(T) - cdf(Chisq(3), joint)

    kX = size(X, 2)
    sigma2 = ss_res / (neff - kX)
    V = sigma2 * XtX_inv
    se = sqrt.(max.(diag(V), zero(T)))
    tstat = b ./ se
    tdist = TDist(neff - kX)
    pind = [2 * (one(T) - cdf(tdist, abs(tstat[i]))) for i in 1:kX]

    (sign_bias=b[2], sign_bias_t=tstat[2], sign_bias_p=pind[2],
     neg_size_t=tstat[3], neg_size_p=pind[3],
     pos_size_t=tstat[4], pos_size_p=pind[4],
     joint_statistic=joint, joint_pvalue=pj, dof=3)
end

sign_bias_test(m::AbstractVolatilityModel) = sign_bias_test(m.standardized_residuals)
sign_bias_test(z::AbstractVector) = sign_bias_test(Float64.(z))

# =============================================================================
# Nyblom (1989) / Hansen (1992) parameter-stability test
# =============================================================================

# Hansen (1992, JBES) Table 1 asymptotic 5% critical values for the joint L_C
# statistic, indexed by the number of parameters (1..20).
const _NYBLOM_JOINT_CV5 = (0.470, 0.749, 1.01, 1.24, 1.47, 1.68, 1.90, 2.11, 2.32,
                           2.54, 2.75, 2.96, 3.15, 3.34, 3.54, 3.75, 3.95, 4.14,
                           4.33, 4.52)
_nyblom_joint_cv5(k::Int) = _NYBLOM_JOINT_CV5[clamp(k, 1, length(_NYBLOM_JOINT_CV5))]

# Per-observation score matrix (n×k) at the MLE, in each model's free transform space.
function _volatility_nyblom_scores(m::GARCHModel)
    free = vcat(m.mu, log(m.omega), log.(m.alpha), log.(m.beta))
    ForwardDiff.jacobian(θ -> _garch_loglik_contribs(θ, m.y, m.p, m.q), free)
end
function _volatility_nyblom_scores(m::EGARCHModel)
    free = vcat(m.mu, m.omega, m.alpha, m.gamma, m.beta)
    ForwardDiff.jacobian(θ -> _egarch_loglik_contribs(θ, m.y, m.p, m.q), free)
end
function _volatility_nyblom_scores(m::GJRGARCHModel)
    free = vcat(m.mu, log(m.omega), log.(m.alpha), log.(m.gamma), log.(m.beta))
    ForwardDiff.jacobian(θ -> _gjr_loglik_contribs(θ, m.y, m.p, m.q), free)
end
function _volatility_nyblom_scores(m::IGARCHModel)
    free = _igarch_reconstruct_free(m)
    ForwardDiff.jacobian(θ -> _igarch_loglik_contribs(θ, m.y, m.p, m.q), free)
end
function _volatility_nyblom_scores(m::CGARCHModel)
    free = _cgarch_reconstruct_free(m)
    ForwardDiff.jacobian(θ -> _cgarch_loglik_contribs(θ, m.y), free)
end
function _volatility_nyblom_scores(m::APARCHModel)
    lay = _APARCHLayout(m.q, m.p,
        m.fixed_delta ? Float64(m.delta) : nothing,
        m.fixed_gamma ? Float64(m.gamma[1]) : nothing)
    free = _aparch_reconstruct_free(m, lay)
    ForwardDiff.jacobian(θ -> _aparch_loglik_contribs(θ, m.y, lay), free)
end

"""
    nyblom_test(m::AbstractVolatilityModel) -> NamedTuple

Nyblom (1989) / Hansen (1992) test for parameter stability of a fitted volatility
model against the alternative that parameters follow a martingale (random-walk)
process. Using the per-observation score matrix `sₜ = ∇_θ ℓₜ` evaluated at the MLE
(in the model's free transform space):

    Lᵢ = (1/n) Σₜ (Σ_{j≤t} s̃_{j,i})² / Vᵢᵢ ,   L_C = (1/n) Σₜ Ŝₜ' V⁻¹ Ŝₜ

with `s̃` the centered scores, `Ŝₜ = Σ_{j≤t} s̃ⱼ` the cumulative score, and
`V = Σₜ s̃ₜ s̃ₜ'`. Large `Lᵢ`/`L_C` reject stability. Critical values are the
Hansen (1992) asymptotic values (individual 5% ≈ 0.470; joint from the Table 1
`L_C` column, indexed by the number of parameters).

# Returns
Named tuple with `individual` (per-parameter `Lᵢ`), `joint` (`L_C`), `k`,
`cv_individual` (0.470), `cv_joint` (5% Hansen value for `k`), and `param_names`.

# References
- Nyblom (1989). *Journal of the American Statistical Association* 84(405), 223–230.
- Hansen (1992). *Journal of Business & Economic Statistics* 10(3), 321–335.
"""
function nyblom_test(m::AbstractVolatilityModel)
    T = eltype(m.y)
    S = _volatility_nyblom_scores(m)::Matrix
    S = Matrix{T}(S)
    n, k = size(S)
    Sc = S .- (sum(S, dims=1) ./ n)         # center the scores
    V = Sc' * Sc                             # k×k outer-product information
    cum = cumsum(Sc, dims=1)                 # n×k cumulative scores

    Li = Vector{T}(undef, k)
    for i in 1:k
        vii = V[i, i]
        Li[i] = vii > zero(T) ? sum(abs2, @view cum[:, i]) / (T(n) * vii) : T(NaN)
    end

    Vinv = robust_inv(V)
    Lc = zero(T)
    @inbounds for t in 1:n
        ct = @view cum[t, :]
        Lc += dot(ct, Vinv * ct)
    end
    Lc /= T(n)

    param_names = _volatility_param_names(m)
    (individual=Li, joint=Lc, k=k, cv_individual=T(0.470),
     cv_joint=T(_nyblom_joint_cv5(k)), param_names=param_names)
end

# Free-space parameter labels (match the score-matrix column order).
_volatility_param_names(m::GARCHModel) =
    vcat("μ", "ω", ["α$i" for i in 1:m.q], ["β$j" for j in 1:m.p])
_volatility_param_names(m::EGARCHModel) =
    vcat("μ", "ω", ["α$i" for i in 1:m.q], ["γ$i" for i in 1:m.q], ["β$j" for j in 1:m.p])
_volatility_param_names(m::GJRGARCHModel) =
    vcat("μ", "ω", ["α$i" for i in 1:m.q], ["γ$i" for i in 1:m.q], ["β$j" for j in 1:m.p])
_volatility_param_names(m::IGARCHModel) =
    vcat("μ", "ω", ["w$i" for i in 1:(m.q + m.p - 1)])
_volatility_param_names(::CGARCHModel) = ["μ", "ω", "ρ", "φ", "α", "β"]
function _volatility_param_names(m::APARCHModel)
    v = ["μ", "ω"]
    append!(v, ["α$i" for i in 1:m.q])
    m.fixed_gamma || append!(v, ["γ$i" for i in 1:m.q])
    append!(v, ["β$j" for j in 1:m.p])
    m.fixed_delta || push!(v, "δ")
    v
end
