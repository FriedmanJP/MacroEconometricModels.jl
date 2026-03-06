# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Marginal effects and odds ratios for binary response models (Logit/Probit).

Implements average marginal effects (AME), marginal effects at the mean (MEM),
marginal effects at representative values (MER), and odds ratios with
delta-method standard errors.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Link Function Helpers
# =============================================================================

"""
    _logit_pdf(eta) -> f(eta)

Logistic PDF: f(eta) = exp(-eta) / (1 + exp(-eta))^2 = p(1-p).
"""
function _logit_pdf(eta::T) where {T<:AbstractFloat}
    p = one(T) / (one(T) + exp(-eta))
    p = clamp(p, T(1e-10), one(T) - T(1e-10))
    p * (one(T) - p)
end

"""
    _logit_pdf_deriv(eta) -> f'(eta)

Derivative of the logistic PDF: f'(eta) = p(1-p)(1-2p).
"""
function _logit_pdf_deriv(eta::T) where {T<:AbstractFloat}
    p = one(T) / (one(T) + exp(-eta))
    p = clamp(p, T(1e-10), one(T) - T(1e-10))
    p * (one(T) - p) * (one(T) - 2 * p)
end

"""
    _probit_pdf(eta) -> phi(eta)

Standard normal PDF evaluated at eta.
"""
function _probit_pdf(eta::T) where {T<:AbstractFloat}
    T(pdf(Normal(zero(T), one(T)), eta))
end

"""
    _probit_pdf_deriv(eta) -> phi'(eta)

Derivative of the standard normal PDF: phi'(eta) = -eta * phi(eta).
"""
function _probit_pdf_deriv(eta::T) where {T<:AbstractFloat}
    -eta * _probit_pdf(eta)
end

# =============================================================================
# Shared Implementation
# =============================================================================

"""
    _marginal_effects_impl(m, link, type, at, conf_level) -> MarginalEffects{T}

Shared implementation for computing marginal effects from binary response models.

# Arguments
- `m` — LogitModel or ProbitModel
- `link::Symbol` — :logit or :probit
- `type::Symbol` — :ame (average), :mem (at-mean), :mer (at-representative)
- `at::Union{Nothing,Dict}` — evaluation points for :mer (Dict mapping column index to value)
- `conf_level::Real` — confidence level for CIs

# Formulas
- AME: (1/N) sum_i f(X_i beta) * beta_j
- MEM: f(X_bar beta) * beta_j
- MER: f(x_0 beta) * beta_j at user-specified x_0

Delta method: Var(ME) = G' Var(beta) G where G = d(marginal effect)/d(beta).
"""
function _marginal_effects_impl(m, link::Symbol, type::Symbol,
                                 at::Union{Nothing,Dict}, conf_level::Real)
    T = eltype(m.beta)
    X = m.X
    beta = m.beta
    V = m.vcov_mat
    n, k = size(X)

    # Select PDF and its derivative based on link
    f_pdf = link == :logit ? _logit_pdf : _probit_pdf
    f_pdf_deriv = link == :logit ? _logit_pdf_deriv : _probit_pdf_deriv

    # ---- Compute marginal effects and Jacobian G ----
    me = Vector{T}(undef, k)
    G = Matrix{T}(undef, k, k)

    if type == :ame
        # Average Marginal Effects: (1/N) sum_i f(eta_i) * beta_j
        # G[j,l] = (1/N) sum_i [(j==l ? f_i : 0) + f'(eta_i) * beta_j * x_il]

        me .= zero(T)
        G .= zero(T)

        @inbounds for i in 1:n
            xi = @view X[i, :]
            eta_i = dot(xi, beta)
            f_i = f_pdf(eta_i)
            fp_i = f_pdf_deriv(eta_i)

            for j in 1:k
                me[j] += f_i * beta[j]
                for l in 1:k
                    G[j, l] += (j == l ? f_i : zero(T)) + fp_i * beta[j] * xi[l]
                end
            end
        end

        me ./= T(n)
        G ./= T(n)

    elseif type == :mem
        # Marginal Effects at the Mean: f(X_bar beta) * beta_j
        x_bar = vec(mean(X, dims=1))
        eta_bar = dot(x_bar, beta)
        f_bar = f_pdf(eta_bar)
        fp_bar = f_pdf_deriv(eta_bar)

        @inbounds for j in 1:k
            me[j] = f_bar * beta[j]
            for l in 1:k
                G[j, l] = (j == l ? f_bar : zero(T)) + fp_bar * beta[j] * x_bar[l]
            end
        end

    elseif type == :mer
        # Marginal Effects at Representative values
        at === nothing && throw(ArgumentError("at must be provided for type=:mer"))

        # Start with sample mean, then override specified values
        x_0 = vec(mean(X, dims=1))
        for (col_idx, val) in at
            1 <= col_idx <= k || throw(ArgumentError("at key $col_idx out of range [1, $k]"))
            x_0[col_idx] = T(val)
        end

        eta_0 = dot(x_0, beta)
        f_0 = f_pdf(eta_0)
        fp_0 = f_pdf_deriv(eta_0)

        @inbounds for j in 1:k
            me[j] = f_0 * beta[j]
            for l in 1:k
                G[j, l] = (j == l ? f_0 : zero(T)) + fp_0 * beta[j] * x_0[l]
            end
        end
    else
        throw(ArgumentError("type must be :ame, :mem, or :mer; got :$type"))
    end

    # ---- Delta-method standard errors ----
    # Var(ME) = G V G'
    var_me = G * V * G'
    se = sqrt.(max.(diag(var_me), zero(T)))

    # ---- z-statistics and p-values ----
    z_stat = Vector{T}(undef, k)
    p_values = Vector{T}(undef, k)
    @inbounds for j in 1:k
        z_stat[j] = se[j] > zero(T) ? me[j] / se[j] : zero(T)
        p_values[j] = T(2 * (1 - cdf(Normal(), abs(z_stat[j]))))
    end

    # ---- Confidence intervals ----
    z_crit = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = me .- z_crit .* se
    ci_upper = me .+ z_crit .* se

    MarginalEffects{T}(
        me, se, z_stat, p_values,
        ci_lower, ci_upper,
        copy(m.varnames), type, T(conf_level)
    )
end

# =============================================================================
# Public API — Marginal Effects
# =============================================================================

"""
    marginal_effects(m::LogitModel{T}; type=:ame, at=nothing, conf_level=0.95) -> MarginalEffects{T}

Compute marginal effects from a logistic regression model with delta-method SEs.

# Types
- `:ame` — Average Marginal Effects: (1/N) sum_i f(X_i beta) * beta_j
- `:mem` — Marginal Effects at the Mean: f(X_bar beta) * beta_j
- `:mer` — Marginal Effects at Representative values: f(x_0 beta) * beta_j

where f(eta) = logistic PDF = p(1-p) and p = 1/(1+exp(-eta)).

# Arguments
- `m::LogitModel{T}` — estimated logit model
- `type::Symbol` — :ame (default), :mem, or :mer
- `at::Union{Nothing,Dict}` — for :mer, Dict mapping column index => value
- `conf_level::Real` — confidence level (default 0.95)

# Returns
`MarginalEffects{T}` with effects, delta-method SEs, z-stats, p-values, and CIs.

# Examples
```julia
m = estimate_logit(y, X)
me_ame = marginal_effects(m)                          # AME
me_mem = marginal_effects(m; type=:mem)                # MEM
me_mer = marginal_effects(m; type=:mer, at=Dict(2=>0.0))  # MER at x2=0
```

# References
- Cameron, A. C. & Trivedi, P. K. (2005). *Microeconometrics*. Cambridge University Press, ch. 14.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
"""
function marginal_effects(m::LogitModel{T};
                          type::Symbol=:ame,
                          at::Union{Nothing,Dict}=nothing,
                          conf_level::Real=0.95) where {T<:AbstractFloat}
    _marginal_effects_impl(m, :logit, type, at, conf_level)
end

"""
    marginal_effects(m::ProbitModel{T}; type=:ame, at=nothing, conf_level=0.95) -> MarginalEffects{T}

Compute marginal effects from a probit regression model with delta-method SEs.

# Types
- `:ame` — Average Marginal Effects: (1/N) sum_i phi(X_i beta) * beta_j
- `:mem` — Marginal Effects at the Mean: phi(X_bar beta) * beta_j
- `:mer` — Marginal Effects at Representative values: phi(x_0 beta) * beta_j

where phi(eta) = standard normal PDF.

# Arguments
- `m::ProbitModel{T}` — estimated probit model
- `type::Symbol` — :ame (default), :mem, or :mer
- `at::Union{Nothing,Dict}` — for :mer, Dict mapping column index => value
- `conf_level::Real` — confidence level (default 0.95)

# Returns
`MarginalEffects{T}` with effects, delta-method SEs, z-stats, p-values, and CIs.

# Examples
```julia
m = estimate_probit(y, X)
me_ame = marginal_effects(m)                          # AME
me_mem = marginal_effects(m; type=:mem)                # MEM
me_mer = marginal_effects(m; type=:mer, at=Dict(2=>0.0))  # MER at x2=0
```

# References
- Cameron, A. C. & Trivedi, P. K. (2005). *Microeconometrics*. Cambridge University Press, ch. 14.
"""
function marginal_effects(m::ProbitModel{T};
                          type::Symbol=:ame,
                          at::Union{Nothing,Dict}=nothing,
                          conf_level::Real=0.95) where {T<:AbstractFloat}
    _marginal_effects_impl(m, :probit, type, at, conf_level)
end

# =============================================================================
# Odds Ratios (Logit Only)
# =============================================================================

"""
    odds_ratio(m::LogitModel{T}; conf_level=0.95) -> NamedTuple

Compute odds ratios from a logistic regression model with delta-method CIs.

OR_j = exp(beta_j). Standard error via delta method: SE(OR) = OR * SE(beta).
Confidence intervals on log scale: exp(beta +/- z * SE(beta)).

# Arguments
- `m::LogitModel{T}` — estimated logit model
- `conf_level::Real` — confidence level (default 0.95)

# Returns
Named tuple with fields:
- `or::Vector{T}` — odds ratios exp(beta)
- `se::Vector{T}` — delta-method standard errors
- `ci_lower::Vector{T}` — lower CI bounds
- `ci_upper::Vector{T}` — upper CI bounds
- `varnames::Vector{String}` — variable names

# Examples
```julia
m = estimate_logit(y, X)
result = odds_ratio(m)
result.or       # odds ratios
result.ci_lower # lower 95% CI
```

# References
- Agresti, A. (2002). *Categorical Data Analysis*. 2nd ed. Wiley, ch. 5.
"""
function odds_ratio(m::LogitModel{T}; conf_level::Real=0.95) where {T<:AbstractFloat}
    beta = m.beta
    se_beta = sqrt.(max.(diag(m.vcov_mat), zero(T)))
    k = length(beta)

    or = exp.(beta)

    # Delta method: SE(OR_j) = OR_j * SE(beta_j)
    se_or = or .* se_beta

    # CIs on log scale: exp(beta +/- z * SE(beta))
    z_crit = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = exp.(beta .- z_crit .* se_beta)
    ci_upper = exp.(beta .+ z_crit .* se_beta)

    (or=or, se=se_or, ci_lower=ci_lower, ci_upper=ci_upper,
     varnames=copy(m.varnames))
end
