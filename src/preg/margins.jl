# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel marginal effects for PanelLogitModel and PanelProbitModel.

Computes average marginal effects (AME) with delta-method standard errors,
handling pooled, FE, RE, and CRE specifications appropriately.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Panel Marginal Effects — Logit
# =============================================================================

"""
    marginal_effects(m::PanelLogitModel{T}; conf_level=0.95, n_quadrature=12) -> MarginalEffects{T}

Compute average marginal effects from a panel logit model with delta-method SEs.

# Formulas by model type
- **Pooled**: AME_j = (1/n) sum_i Lambda(x_i'beta)(1-Lambda(x_i'beta)) * beta_j
- **FE (conditional)**: AME_j = (1/n) sum_i Lambda(x_i'beta)(1-Lambda(x_i'beta)) * beta_j
  (evaluated at observed X without entity effects)
- **RE**: AME_j = (1/n) sum_i integral Lambda(x_i'beta + sigma_u*sqrt(2)*u)(1-Lambda(...)) * beta_j * w(u) du
  (integrated over random effect using Gauss-Hermite quadrature)
- **CRE**: Same as RE but only reports AMEs for original variables (not group-mean augmented ones)

# Arguments
- `m::PanelLogitModel{T}` -- estimated panel logit model
- `conf_level::Real` -- confidence level for CIs (default 0.95)
- `n_quadrature::Int` -- number of Gauss-Hermite quadrature points for RE/CRE (default 12)

# Returns
`MarginalEffects{T}` with AMEs, delta-method SEs, z-stats, p-values, and CIs.

# References
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. 2nd ed. MIT Press.
- Cameron, A. C. & Trivedi, P. K. (2005). *Microeconometrics*. Cambridge University Press.
"""
function marginal_effects(m::PanelLogitModel{T};
                          conf_level::Real=0.95,
                          n_quadrature::Int=12) where {T<:AbstractFloat}
    if m.method == :pooled
        return _panel_logit_ame_pooled(m, T(conf_level))
    elseif m.method == :fe
        return _panel_logit_ame_fe(m, T(conf_level))
    elseif m.method == :re
        return _panel_logit_ame_re(m, T(conf_level), n_quadrature)
    elseif m.method == :cre
        return _panel_logit_ame_cre(m, T(conf_level), n_quadrature)
    else
        throw(ArgumentError("Unsupported method: $(m.method)"))
    end
end

# =============================================================================
# Panel Marginal Effects — Probit
# =============================================================================

"""
    marginal_effects(m::PanelProbitModel{T}; conf_level=0.95, n_quadrature=12) -> MarginalEffects{T}

Compute average marginal effects from a panel probit model with delta-method SEs.

# Formulas by model type
- **Pooled**: AME_j = (1/n) sum_i phi(x_i'beta) * beta_j
- **RE**: AME_j = (1/n) sum_i integral phi(x_i'beta + sigma_u*sqrt(2)*u) * beta_j * w(u) du
- **CRE**: Same as RE but only reports AMEs for original variables

# Arguments
- `m::PanelProbitModel{T}` -- estimated panel probit model
- `conf_level::Real` -- confidence level for CIs (default 0.95)
- `n_quadrature::Int` -- Gauss-Hermite quadrature points for RE/CRE (default 12)

# Returns
`MarginalEffects{T}` with AMEs, delta-method SEs, z-stats, p-values, and CIs.
"""
function marginal_effects(m::PanelProbitModel{T};
                          conf_level::Real=0.95,
                          n_quadrature::Int=12) where {T<:AbstractFloat}
    if m.method == :pooled
        return _panel_probit_ame_pooled(m, T(conf_level))
    elseif m.method == :re
        return _panel_probit_ame_re(m, T(conf_level), n_quadrature)
    elseif m.method == :cre
        return _panel_probit_ame_cre(m, T(conf_level), n_quadrature)
    else
        throw(ArgumentError("Unsupported method: $(m.method)"))
    end
end

# =============================================================================
# Pooled Logit AME
# =============================================================================

function _panel_logit_ame_pooled(m::PanelLogitModel{T}, conf_level::T) where {T}
    X = m.X
    beta = m.beta
    V = m.vcov_mat
    n, k = size(X)

    me = zeros(T, k)
    G = zeros(T, k, k)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        eta_i = dot(xi, beta)
        f_i = _logit_pdf(eta_i)
        fp_i = _logit_pdf_deriv(eta_i)

        for j in 1:k
            me[j] += f_i * beta[j]
            for l in 1:k
                G[j, l] += (j == l ? f_i : zero(T)) + fp_i * beta[j] * xi[l]
            end
        end
    end

    me ./= T(n)
    G ./= T(n)

    _panel_me_finalize(me, G, V, m.varnames, conf_level)
end

# =============================================================================
# FE Logit AME (conditional)
# =============================================================================

function _panel_logit_ame_fe(m::PanelLogitModel{T}, conf_level::T) where {T}
    X = m.X   # no intercept for FE
    beta = m.beta
    V = m.vcov_mat
    n, k = size(X)

    me = zeros(T, k)
    G = zeros(T, k, k)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        eta_i = dot(xi, beta)
        f_i = _logit_pdf(eta_i)
        fp_i = _logit_pdf_deriv(eta_i)

        for j in 1:k
            me[j] += f_i * beta[j]
            for l in 1:k
                G[j, l] += (j == l ? f_i : zero(T)) + fp_i * beta[j] * xi[l]
            end
        end
    end

    me ./= T(n)
    G ./= T(n)

    _panel_me_finalize(me, G, V, m.varnames, conf_level)
end

# =============================================================================
# RE Logit AME — Gauss-Hermite integration over random effect
# =============================================================================

function _panel_logit_ame_re(m::PanelLogitModel{T}, conf_level::T, n_quad::Int) where {T}
    X = m.X
    beta = m.beta
    V = m.vcov_mat
    sigma_u = m.sigma_u
    n, k = size(X)

    nodes, weights = _gauss_hermite_nodes_weights(n_quad)

    me = zeros(T, k)
    G = zeros(T, k, k)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        eta_i = dot(xi, beta)

        for q in 1:n_quad
            alpha = sqrt(T(2)) * sigma_u * T(nodes[q])
            wq = T(weights[q]) / sqrt(T(pi))
            eta_q = eta_i + alpha

            f_q = _logit_pdf(eta_q)
            fp_q = _logit_pdf_deriv(eta_q)

            for j in 1:k
                me[j] += wq * f_q * beta[j]
                for l in 1:k
                    G[j, l] += wq * ((j == l ? f_q : zero(T)) + fp_q * beta[j] * xi[l])
                end
            end
        end
    end

    me ./= T(n)
    G ./= T(n)

    _panel_me_finalize(me, G, V, m.varnames, conf_level)
end

# =============================================================================
# CRE Logit AME — RE with augmented X, report only original vars
# =============================================================================

function _panel_logit_ame_cre(m::PanelLogitModel{T}, conf_level::T, n_quad::Int) where {T}
    X = m.X
    beta = m.beta
    V = m.vcov_mat
    sigma_u = m.sigma_u
    n, k_full = size(X)

    nodes, weights = _gauss_hermite_nodes_weights(n_quad)

    # Compute full AMEs (all k_full variables)
    me_full = zeros(T, k_full)
    G_full = zeros(T, k_full, k_full)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        eta_i = dot(xi, beta)

        for q in 1:n_quad
            alpha = sqrt(T(2)) * sigma_u * T(nodes[q])
            wq = T(weights[q]) / sqrt(T(pi))
            eta_q = eta_i + alpha

            f_q = _logit_pdf(eta_q)
            fp_q = _logit_pdf_deriv(eta_q)

            for j in 1:k_full
                me_full[j] += wq * f_q * beta[j]
                for l in 1:k_full
                    G_full[j, l] += wq * ((j == l ? f_q : zero(T)) + fp_q * beta[j] * xi[l])
                end
            end
        end
    end

    me_full ./= T(n)
    G_full ./= T(n)

    # Identify original variable indices: varnames that do NOT end with "_mean"
    # CRE varnames = ["_cons", orig1, ..., origK, orig1_mean, ..., origK_mean]
    orig_idx = Int[]
    for (j, vn) in enumerate(m.varnames)
        if !endswith(vn, "_mean") && vn != "_cons"
            push!(orig_idx, j)
        end
    end

    # Extract sub-effects and propagate delta-method through full Jacobian
    me = me_full[orig_idx]
    G_sub = G_full[orig_idx, :]  # K_orig x K_full

    var_me = G_sub * V * G_sub'
    se = sqrt.(max.(diag(var_me), zero(T)))

    vn_orig = m.varnames[orig_idx]
    _panel_me_stats(me, se, vn_orig, conf_level)
end

# =============================================================================
# Pooled Probit AME
# =============================================================================

function _panel_probit_ame_pooled(m::PanelProbitModel{T}, conf_level::T) where {T}
    X = m.X
    beta = m.beta
    V = m.vcov_mat
    n, k = size(X)

    me = zeros(T, k)
    G = zeros(T, k, k)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        eta_i = dot(xi, beta)
        f_i = _probit_pdf(eta_i)
        fp_i = _probit_pdf_deriv(eta_i)

        for j in 1:k
            me[j] += f_i * beta[j]
            for l in 1:k
                G[j, l] += (j == l ? f_i : zero(T)) + fp_i * beta[j] * xi[l]
            end
        end
    end

    me ./= T(n)
    G ./= T(n)

    _panel_me_finalize(me, G, V, m.varnames, conf_level)
end

# =============================================================================
# RE Probit AME — Gauss-Hermite integration over random effect
# =============================================================================

function _panel_probit_ame_re(m::PanelProbitModel{T}, conf_level::T, n_quad::Int) where {T}
    X = m.X
    beta = m.beta
    V = m.vcov_mat
    sigma_u = m.sigma_u
    n, k = size(X)

    nodes, weights = _gauss_hermite_nodes_weights(n_quad)

    me = zeros(T, k)
    G = zeros(T, k, k)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        eta_i = dot(xi, beta)

        for q in 1:n_quad
            alpha = sqrt(T(2)) * sigma_u * T(nodes[q])
            wq = T(weights[q]) / sqrt(T(pi))
            eta_q = eta_i + alpha

            f_q = _probit_pdf(eta_q)
            fp_q = _probit_pdf_deriv(eta_q)

            for j in 1:k
                me[j] += wq * f_q * beta[j]
                for l in 1:k
                    G[j, l] += wq * ((j == l ? f_q : zero(T)) + fp_q * beta[j] * xi[l])
                end
            end
        end
    end

    me ./= T(n)
    G ./= T(n)

    _panel_me_finalize(me, G, V, m.varnames, conf_level)
end

# =============================================================================
# CRE Probit AME — RE with augmented X, report only original vars
# =============================================================================

function _panel_probit_ame_cre(m::PanelProbitModel{T}, conf_level::T, n_quad::Int) where {T}
    X = m.X
    beta = m.beta
    V = m.vcov_mat
    sigma_u = m.sigma_u
    n, k_full = size(X)

    nodes, weights = _gauss_hermite_nodes_weights(n_quad)

    me_full = zeros(T, k_full)
    G_full = zeros(T, k_full, k_full)

    @inbounds for i in 1:n
        xi = @view X[i, :]
        eta_i = dot(xi, beta)

        for q in 1:n_quad
            alpha = sqrt(T(2)) * sigma_u * T(nodes[q])
            wq = T(weights[q]) / sqrt(T(pi))
            eta_q = eta_i + alpha

            f_q = _probit_pdf(eta_q)
            fp_q = _probit_pdf_deriv(eta_q)

            for j in 1:k_full
                me_full[j] += wq * f_q * beta[j]
                for l in 1:k_full
                    G_full[j, l] += wq * ((j == l ? f_q : zero(T)) + fp_q * beta[j] * xi[l])
                end
            end
        end
    end

    me_full ./= T(n)
    G_full ./= T(n)

    # Identify original variable indices
    orig_idx = Int[]
    for (j, vn) in enumerate(m.varnames)
        if !endswith(vn, "_mean") && vn != "_cons"
            push!(orig_idx, j)
        end
    end

    me = me_full[orig_idx]
    G_sub = G_full[orig_idx, :]

    var_me = G_sub * V * G_sub'
    se = sqrt.(max.(diag(var_me), zero(T)))

    vn_orig = m.varnames[orig_idx]
    _panel_me_stats(me, se, vn_orig, conf_level)
end

# =============================================================================
# Shared Finalization Helpers
# =============================================================================

"""Compute SE from full Jacobian G and vcov V, then build MarginalEffects."""
function _panel_me_finalize(me::Vector{T}, G::Matrix{T}, V::Matrix{T},
                             varnames::Vector{String}, conf_level::T) where {T}
    var_me = G * V * G'
    se = sqrt.(max.(diag(var_me), zero(T)))
    _panel_me_stats(me, se, varnames, conf_level)
end

"""Build MarginalEffects from AME vector and SE vector."""
function _panel_me_stats(me::Vector{T}, se::Vector{T},
                          varnames::Vector{String}, conf_level::T) where {T}
    k = length(me)
    z_stat = Vector{T}(undef, k)
    p_values = Vector{T}(undef, k)
    @inbounds for j in 1:k
        z_stat[j] = se[j] > zero(T) ? me[j] / se[j] : zero(T)
        p_values[j] = T(2 * (1 - cdf(Normal(), abs(z_stat[j]))))
    end

    z_crit = T(quantile(Normal(), 1 - (1 - conf_level) / 2))
    ci_lower = me .- z_crit .* se
    ci_upper = me .+ z_crit .* se

    MarginalEffects{T}(
        me, se, z_stat, p_values,
        ci_lower, ci_upper,
        copy(varnames), :ame, conf_level
    )
end
