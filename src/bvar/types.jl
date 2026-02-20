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
Type definitions for Bayesian VAR posterior results.
"""

# =============================================================================
# Bayesian VAR Posterior
# =============================================================================

"""
    BVARPosterior{T} <: Any

Posterior draws from Bayesian VAR estimation.

Replaces MCMCChains.Chains — stores i.i.d. or Gibbs draws from the
Normal-Inverse-Wishart posterior directly.

# Fields
- `B_draws::Array{T,3}`: Coefficient draws (n_draws × k × n)
- `Sigma_draws::Array{T,3}`: Covariance draws (n_draws × n × n)
- `n_draws::Int`: Number of posterior draws
- `p::Int`: Number of VAR lags
- `n::Int`: Number of variables
- `data::Matrix{T}`: Original Y matrix (for residual computation downstream)
- `prior::Symbol`: Prior used (:normal or :minnesota)
- `sampler::Symbol`: Sampler used (:direct or :gibbs)
- `varnames::Vector{String}`: Variable names
"""
struct BVARPosterior{T<:AbstractFloat}
    B_draws::Array{T,3}       # n_draws × k × n
    Sigma_draws::Array{T,3}   # n_draws × n × n
    n_draws::Int
    p::Int
    n::Int
    data::Matrix{T}
    prior::Symbol
    sampler::Symbol
    varnames::Vector{String}
end

"""
    BVARForecast{T} <: AbstractForecastResult{T}

Bayesian VAR forecast with posterior credible intervals.

Fields: forecast (h×n), ci_lower (h×n), ci_upper (h×n), horizon, conf_level, point_estimate, varnames.
"""
struct BVARForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    forecast::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    conf_level::T
    point_estimate::Symbol
    varnames::Vector{String}
end

function Base.show(io::IO, fc::BVARForecast{T}) where {T}
    n_vars = length(fc.varnames)
    ci_pct = round(Int, 100 * fc.conf_level)

    spec = Any[
        "Horizon"     fc.horizon;
        "Variables"   n_vars;
        "Credibility" "$(ci_pct)%"
    ]
    _pretty_table(io, spec;
        title = "Bayesian VAR Forecast",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Per-variable forecast table
    lo_pct = round(Int, 100 * (1 - fc.conf_level) / 2)
    hi_pct = 100 - lo_pct
    for vi in 1:n_vars
        data = Matrix{Any}(undef, fc.horizon, 4)
        for h in 1:fc.horizon
            data[h, 1] = h
            data[h, 2] = _fmt(fc.forecast[h, vi])
            data[h, 3] = _fmt(fc.ci_lower[h, vi])
            data[h, 4] = _fmt(fc.ci_upper[h, vi])
        end
        _pretty_table(io, data;
            title = "$(fc.varnames[vi])",
            column_labels = ["h", fc.point_estimate == :median ? "Post. Median" : "Post. Mean", "$(lo_pct)%", "$(hi_pct)%"],
            alignment = [:r, :r, :r, :r],
        )
    end
end

Base.size(post::BVARPosterior, dim::Int) = dim == 1 ? post.n_draws : error("BVARPosterior has 1 dimension (n_draws)")
Base.length(post::BVARPosterior) = post.n_draws

function Base.show(io::IO, post::BVARPosterior{T}) where {T}
    k = size(post.B_draws, 2)  # parameters per equation

    # Specification table
    spec_data = Any[
        "Variables"     post.n;
        "Lags"          post.p;
        "Draws"         post.n_draws;
        "Prior"         string(post.prior);
        "Sampler"       string(post.sampler);
        "Parameters/eq" k
    ]
    _pretty_table(io, spec_data;
        title = "Bayesian VAR — BVAR($(post.p))",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Build coefficient names
    coef_names = String["const"]
    for l in 1:post.p
        for v in 1:post.n
            push!(coef_names, "Var$(v).L$l")
        end
    end
    while length(coef_names) < k
        push!(coef_names, "x$(length(coef_names)+1)")
    end

    # Per-equation posterior summary
    vn = post.varnames
    for eq in 1:post.n
        draws_eq = post.B_draws[:, :, eq]  # n_draws × k
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
            title = "Equation: $(vn[eq])",
            column_labels = ["", "Mean", "Std", "2.5%", "50%", "97.5%"],
            alignment = [:l, :r, :r, :r, :r, :r],
        )
    end

    # Posterior mean of Σ
    Sigma_mean = dropdims(mean(post.Sigma_draws; dims=1); dims=1)
    _matrix_table(io, Sigma_mean, "Posterior Mean Σ";
        row_labels=vn,
        col_labels=vn)
end
