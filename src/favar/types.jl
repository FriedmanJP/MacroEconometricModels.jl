# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Type definitions for Factor-Augmented VAR (FAVAR) models.

FAVAR augments a standard VAR with latent factors extracted from a large
panel of macroeconomic variables, following Bernanke, Boivin & Eliasz (2005).

References:
- Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the Effects of
  Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach.
  Quarterly Journal of Economics, 120(1), 387-422.
"""

# =============================================================================
# FAVARModel
# =============================================================================

"""
    FAVARModel{T} <: AbstractVARModel

Factor-Augmented VAR model (Bernanke, Boivin & Eliasz, 2005).

The FAVAR system estimates a VAR on [F_t, Y_key,t] where F_t are latent
factors extracted from a large panel X, and Y_key are observed key variables
(e.g., the federal funds rate).

# Fields
- `Y::Matrix{T}`: Augmented data [F, Y_key] used in VAR (T_eff × (r+n_key))
- `p::Int`: VAR lag order
- `B::Matrix{T}`: VAR coefficient matrix ((1+p*(r+n_key)) × (r+n_key))
- `U::Matrix{T}`: VAR residuals
- `Sigma::Matrix{T}`: Residual covariance
- `varnames::Vector{String}`: Variable names in VAR (["F1","F2",...,"Y1","Y2",...])
- `X_panel::Matrix{T}`: Original panel data (T_obs × N)
- `panel_varnames::Vector{String}`: Panel variable names (length N)
- `Y_key_indices::Vector{Int}`: Column indices of key variables in X_panel
- `n_factors::Int`: Number of latent factors r
- `n_key::Int`: Number of key observed variables
- `factors::Matrix{T}`: Extracted factors (T_obs × r)
- `loadings::Matrix{T}`: Factor loadings (N × r)
- `factor_model::FactorModel{T}`: Underlying factor model from PCA
- `aic::T`: Akaike information criterion
- `bic::T`: Bayesian information criterion
- `loglik::T`: Log-likelihood
"""
struct FAVARModel{T<:AbstractFloat} <: AbstractVARModel
    Y::Matrix{T}
    p::Int
    B::Matrix{T}
    U::Matrix{T}
    Sigma::Matrix{T}
    varnames::Vector{String}
    X_panel::Matrix{T}
    panel_varnames::Vector{String}
    Y_key_indices::Vector{Int}
    n_factors::Int
    n_key::Int
    factors::Matrix{T}
    loadings::Matrix{T}
    factor_model::FactorModel{T}
    aic::T
    bic::T
    loglik::T
end

# =============================================================================
# BayesianFAVAR
# =============================================================================

"""
    BayesianFAVAR{T}

Bayesian FAVAR with posterior draws of coefficients, covariances,
factors, and loadings.

# Fields
- `B_draws::Array{T,3}`: Coefficient draws (n_draws × k × n_var)
- `Sigma_draws::Array{T,3}`: Covariance draws (n_draws × n_var × n_var)
- `factor_draws::Array{T,3}`: Factor draws (n_draws × T_obs × r)
- `loadings_draws::Array{T,3}`: Loading draws (n_draws × N × r)
- `X_panel::Matrix{T}`: Original panel data (T_obs × N)
- `panel_varnames::Vector{String}`: Panel variable names
- `Y_key_indices::Vector{Int}`: Key variable column indices
- `n_factors::Int`: Number of factors
- `n_key::Int`: Number of key variables
- `n::Int`: Total VAR variables (r + n_key)
- `p::Int`: VAR lag order
- `data::Matrix{T}`: Augmented VAR data [F, Y_key]
- `varnames::Vector{String}`: VAR variable names
"""
struct BayesianFAVAR{T<:AbstractFloat}
    B_draws::Array{T,3}
    Sigma_draws::Array{T,3}
    factor_draws::Array{T,3}
    loadings_draws::Array{T,3}
    X_panel::Matrix{T}
    panel_varnames::Vector{String}
    Y_key_indices::Vector{Int}
    n_factors::Int
    n_key::Int
    n::Int
    p::Int
    data::Matrix{T}
    varnames::Vector{String}
end

# =============================================================================
# Accessors
# =============================================================================

nvars(m::FAVARModel) = m.n_factors + m.n_key
nlags(m::FAVARModel) = m.p
ncoefs(m::FAVARModel) = 1 + nvars(m) * m.p
effective_nobs(m::FAVARModel) = size(m.Y, 1) - m.p
varnames(m::FAVARModel) = m.varnames

# =============================================================================
# to_var — Convert FAVAR to VARModel for structural analysis dispatch
# =============================================================================

"""
    to_var(favar::FAVARModel) -> VARModel

Convert a FAVAR to a standard VARModel for use with `irf`, `fevd`,
`historical_decomposition`, and other structural analysis methods.

This follows the same delegation pattern as `to_var(vecm::VECMModel)`.
"""
function to_var(favar::FAVARModel{T}) where {T}
    n = nvars(favar)
    p = favar.p

    # Re-use stored B, U, Sigma from the underlying VAR estimation
    Y_aug = favar.Y
    B = favar.B
    U = favar.U
    Sigma = favar.Sigma

    # Compute information criteria from the VAR representation
    T_eff = size(U, 1)
    k = size(B, 1)
    Sigma_ml = (U'U) / T_eff
    log_det = logdet_safe(Sigma_ml)
    aic_val = log_det + 2 * k / T_eff
    bic_val = log_det + k * log(T_eff) / T_eff
    hqic_val = log_det + 2 * k * log(log(T_eff)) / T_eff

    VARModel(Y_aug, p, B, U, Sigma, aic_val, bic_val, hqic_val, favar.varnames)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::FAVARModel{T}) where {T}
    n = nvars(m)
    N = size(m.X_panel, 2)
    T_obs = size(m.X_panel, 1)
    spec = Any[
        "Factors (r)"      m.n_factors;
        "Key variables"     m.n_key;
        "VAR variables"     n;
        "Panel variables"   N;
        "VAR lags (p)"      m.p;
        "Observations"      T_obs;
        "Effective obs"     effective_nobs(m);
        "AIC"               _fmt(m.aic; digits=2);
        "BIC"               _fmt(m.bic; digits=2);
        "Log-lik"           _fmt(m.loglik; digits=4)
    ]
    _pretty_table(io, spec;
        title = "FAVAR($(m.p)) — $(m.n_factors) factors, $(m.n_key) key vars",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Show factor variance explained
    fm = m.factor_model
    n_show = min(m.n_factors, 5)
    var_data = Matrix{Any}(undef, n_show, 3)
    for i in 1:n_show
        var_data[i, 1] = "Factor $i"
        var_data[i, 2] = _fmt_pct(fm.explained_variance[i])
        var_data[i, 3] = _fmt_pct(fm.cumulative_variance[i])
    end
    _pretty_table(io, var_data;
        title = "Variance Explained (PCA)",
        column_labels = ["", "Variance", "Cumulative"],
        alignment = [:l, :r, :r],
    )
end

function Base.show(io::IO, post::BayesianFAVAR{T}) where {T}
    k = size(post.B_draws, 2)
    N = size(post.X_panel, 2)
    n_draws = size(post.B_draws, 1)
    spec = Any[
        "Factors"           post.n_factors;
        "Key variables"     post.n_key;
        "VAR variables"     post.n;
        "Panel variables"   N;
        "Lags"              post.p;
        "Draws"             n_draws;
        "Parameters/eq"     k
    ]
    _pretty_table(io, spec;
        title = "Bayesian FAVAR($(post.p)) — $(post.n_factors) factors, $(post.n_key) key vars",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Per-equation posterior summary (like BVARPosterior)
    coef_names = String["const"]
    for l in 1:post.p
        for v in 1:post.n
            push!(coef_names, "$(post.varnames[v]).L$l")
        end
    end
    while length(coef_names) < k
        push!(coef_names, "x$(length(coef_names)+1)")
    end

    for eq in 1:post.n
        draws_eq = post.B_draws[:, :, eq]
        n_show = min(k, size(draws_eq, 2))
        data = Matrix{Any}(undef, n_show, 6)
        for i in 1:n_show
            col_draws = @view draws_eq[:, i]
            m = mean(col_draws)
            s = std(col_draws)
            q025 = T(quantile(col_draws, 0.025))
            q500 = T(quantile(col_draws, 0.50))
            q975 = T(quantile(col_draws, 0.975))
            data[i, 1] = coef_names[i]
            data[i, 2] = _fmt(m)
            data[i, 3] = _fmt(s)
            data[i, 4] = _fmt(q025)
            data[i, 5] = _fmt(q500)
            data[i, 6] = _fmt(q975)
        end
        _pretty_table(io, data;
            title = "Equation: $(post.varnames[eq])",
            column_labels = ["", "Mean", "Std", "2.5%", "50%", "97.5%"],
            alignment = [:l, :r, :r, :r, :r, :r],
        )
    end
end
