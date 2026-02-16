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
Type definitions for Vector Error Correction Models (VECM).
"""

# =============================================================================
# VECM Model
# =============================================================================

"""
    VECMModel{T} <: AbstractVARModel

Vector Error Correction Model estimated via Johansen MLE or Engle-Granger two-step.

The VECM representation:
    ΔYₜ = αβ'Yₜ₋₁ + Γ₁ΔYₜ₋₁ + ... + Γₚ₋₁ΔYₜ₋ₚ₊₁ + μ + uₜ

where Π = αβ' is the long-run matrix (n × n, rank r), α (n × r) are adjustment
speeds, and β (n × r) are cointegrating vectors.

# Fields
- `Y`: Original data in levels (T_obs × n)
- `p`: Underlying VAR order (VECM has p-1 lagged differences)
- `rank`: Cointegrating rank r
- `alpha`: Adjustment coefficients (n × r)
- `beta`: Cointegrating vectors (n × r), Phillips-normalized
- `Pi`: Long-run matrix αβ' (n × n)
- `Gamma`: Short-run dynamics [Γ₁, ..., Γₚ₋₁]
- `mu`: Intercept (n)
- `U`: Residuals (T_eff × n)
- `Sigma`: Residual covariance (n × n)
- `aic`, `bic`, `hqic`: Information criteria
- `loglik`: Log-likelihood
- `deterministic`: `:none`, `:constant`, or `:trend`
- `method`: `:johansen` or `:engle_granger`
- `johansen_result`: Johansen test result (if applicable)
"""
struct VECMModel{T<:AbstractFloat} <: AbstractVARModel
    Y::Matrix{T}
    p::Int
    rank::Int
    alpha::Matrix{T}
    beta::Matrix{T}
    Pi::Matrix{T}
    Gamma::Vector{Matrix{T}}
    mu::Vector{T}
    U::Matrix{T}
    Sigma::Matrix{T}
    aic::T
    bic::T
    hqic::T
    loglik::T
    deterministic::Symbol
    method::Symbol
    johansen_result::Union{Nothing,JohansenResult{T}}
end

# =============================================================================
# Accessors
# =============================================================================

nvars(m::VECMModel) = size(m.Y, 2)
nlags(m::VECMModel) = m.p
effective_nobs(m::VECMModel) = size(m.U, 1)
ncoefs(m::VECMModel) = m.rank + nvars(m) * (m.p - 1) + (m.deterministic != :none ? 1 : 0) + (m.deterministic == :trend ? 1 : 0)

"""
    cointegrating_rank(m::VECMModel) -> Int

Return the cointegrating rank r of the VECM.
"""
cointegrating_rank(m::VECMModel) = m.rank

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::VECMModel{T}) where {T}
    n = nvars(m)
    p_diff = m.p - 1
    spec = Any[
        "Variables"             n;
        "VAR order (p)"         m.p;
        "Lagged differences"    p_diff;
        "Cointegrating rank"    m.rank;
        "Observations"          size(m.Y, 1);
        "Effective obs"         effective_nobs(m);
        "Deterministic"         string(m.deterministic);
        "Method"                string(m.method);
        "AIC"                   _fmt(m.aic; digits=2);
        "BIC"                   _fmt(m.bic; digits=2)
    ]
    _pretty_table(io, spec;
        title = "VECM($(p_diff)) — Rank $(m.rank)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# VECMForecast
# =============================================================================

"""
    VECMForecast{T}

Forecast result from a VECM, preserving cointegrating relationships.

# Fields
- `levels`: Forecasts in levels (h × n)
- `differences`: Forecasts in first differences (h × n)
- `ci_lower`, `ci_upper`: Confidence interval bounds in levels (h × n)
- `horizon`: Forecast horizon
- `ci_method`: CI method used (`:none`, `:bootstrap`, `:simulation`)
"""
struct VECMForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    levels::Matrix{T}
    differences::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    ci_method::Symbol
end

function Base.show(io::IO, f::VECMForecast{T}) where {T}
    h, n = size(f.levels)
    horizons = _select_horizons(h)
    data = Matrix{Any}(undef, length(horizons), n + 1)
    for (row, hi) in enumerate(horizons)
        data[row, 1] = "h=$hi"
        for j in 1:n
            data[row, j+1] = _fmt(f.levels[hi, j])
        end
    end
    _pretty_table(io, data;
        title = "VECM Forecast (levels, $(f.ci_method) CI)",
        column_labels = vcat(["Horizon"], ["Var $j" for j in 1:n]),
        alignment = vcat([:l], fill(:r, n)),
    )
end

# =============================================================================
# VECMGrangerResult
# =============================================================================

"""
    VECMGrangerResult{T}

VECM Granger causality test result with short-run, long-run, and strong (joint) tests.

# Fields
- `short_run_stat`, `short_run_pvalue`, `short_run_df`: Wald test on Γ coefficients
- `long_run_stat`, `long_run_pvalue`, `long_run_df`: Wald test on α (error correction)
- `strong_stat`, `strong_pvalue`, `strong_df`: Joint test
- `cause_var`, `effect_var`: Variable indices
"""
struct VECMGrangerResult{T<:AbstractFloat}
    short_run_stat::T
    short_run_pvalue::T
    short_run_df::Int
    long_run_stat::T
    long_run_pvalue::T
    long_run_df::Int
    strong_stat::T
    strong_pvalue::T
    strong_df::Int
    cause_var::Int
    effect_var::Int
end

function Base.show(io::IO, g::VECMGrangerResult{T}) where {T}
    data = Any[
        "Short-run (Γ)"  _fmt(g.short_run_stat)  g.short_run_df  _format_pvalue(g.short_run_pvalue)  _significance_stars(g.short_run_pvalue);
        "Long-run (α)"   _fmt(g.long_run_stat)   g.long_run_df   _format_pvalue(g.long_run_pvalue)   _significance_stars(g.long_run_pvalue);
        "Strong (joint)"  _fmt(g.strong_stat)     g.strong_df     _format_pvalue(g.strong_pvalue)     _significance_stars(g.strong_pvalue)
    ]
    _pretty_table(io, data;
        title = "VECM Granger Causality: Var $(g.cause_var) → Var $(g.effect_var)",
        column_labels = ["Test", "Wald χ²", "df", "P-value", ""],
        alignment = [:l, :r, :r, :r, :l],
    )
end
