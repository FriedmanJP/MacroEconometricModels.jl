# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Concrete type definitions for VAR models, IRF, FEVD, and priors.
"""

# =============================================================================
# VAR Models
# =============================================================================

"""
    VARModel{T} <: AbstractVARModel

VAR model estimated via OLS.

Fields: Y (data), p (lags), B (coefficients), U (residuals), Sigma (covariance), aic, bic, hqic, varnames.
"""
struct VARModel{T<:AbstractFloat} <: AbstractVARModel
    Y::Matrix{T}
    p::Int
    B::Matrix{T}
    U::Matrix{T}
    Sigma::Matrix{T}
    aic::T
    bic::T
    hqic::T
    varnames::Vector{String}

    function VARModel(Y::Matrix{T}, p::Int, B::Matrix{T}, U::Matrix{T},
                      Sigma::Matrix{T}, aic::T, bic::T, hqic::T,
                      varnames::Vector{String}=["y$i" for i in 1:size(Y,2)]) where {T<:AbstractFloat}
        n = size(Y, 2)
        @assert size(B, 1) == 1 + n*p && size(B, 2) == n "B dimensions mismatch"
        @assert size(Sigma) == (n, n) "Sigma must be n × n"
        new{T}(Y, p, B, U, Sigma, aic, bic, hqic, varnames)
    end
end

# Convenience constructor with type promotion
function VARModel(Y::AbstractMatrix, p::Int, B::AbstractMatrix, U::AbstractMatrix,
                  Sigma::AbstractMatrix, aic::Real, bic::Real, hqic::Real,
                  varnames::Vector{String}=["y$i" for i in 1:size(Y,2)])
    T = promote_type(eltype(Y), eltype(B), eltype(U), eltype(Sigma), typeof(aic))
    VARModel(Matrix{T}(Y), p, Matrix{T}(B), Matrix{T}(U), Matrix{T}(Sigma),
             T(aic), T(bic), T(hqic), varnames)
end

# Accessors
nvars(model::VARModel) = size(model.Y, 2)

"""
    nlags(model::VARModel) -> Int

Number of lags ``p`` in the fitted VAR.
"""
nlags(model::VARModel) = model.p

"""
    ncoefs(model::VARModel) -> Int

Number of estimated coefficients per equation, ``1 + n p`` (intercept plus ``p`` lags of ``n`` variables).
"""
ncoefs(model::VARModel) = 1 + nvars(model) * model.p
effective_nobs(model::VARModel) = size(model.Y, 1) - model.p
varnames(model::VARModel) = model.varnames

"""
    VARForecast{T} <: AbstractForecastResult{T}

VAR model forecast with optional bootstrap confidence intervals.

Fields: forecast (h×n), ci_lower (h×n), ci_upper (h×n), horizon, ci_method, conf_level, varnames.
"""
struct VARForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    forecast::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    ci_method::Symbol
    conf_level::T
    varnames::Vector{String}
end

function Base.show(io::IO, fc::VARForecast{T}) where {T}
    n_vars = length(fc.varnames)
    has_ci = fc.ci_method != :none
    ci_pct = has_ci ? round(Int, 100 * fc.conf_level) : 0

    spec = Any[
        "Horizon"     fc.horizon;
        "Variables"   n_vars;
        "CI method"   string(fc.ci_method);
        "Conf. level" has_ci ? "$(ci_pct)%" : "—"
    ]
    _pretty_table(io, spec;
        title = "VAR Forecast",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Per-variable forecast table
    for vi in 1:n_vars
        data = Matrix{Any}(undef, fc.horizon, has_ci ? 4 : 2)
        for h in 1:fc.horizon
            data[h, 1] = h
            data[h, 2] = _fmt(fc.forecast[h, vi])
            if has_ci
                data[h, 3] = _fmt(fc.ci_lower[h, vi])
                data[h, 4] = _fmt(fc.ci_upper[h, vi])
            end
        end
        labels = has_ci ?
            ["h", "Forecast", "Lower $(ci_pct)%", "Upper $(ci_pct)%"] :
            ["h", "Forecast"]
        align = has_ci ? [:r, :r, :r, :r] : [:r, :r]
        _pretty_table(io, data;
            title = "$(fc.varnames[vi])",
            column_labels = labels,
            alignment = align,
        )
    end
end

function Base.show(io::IO, m::VARModel{T}) where {T}
    n = nvars(m)
    spec = Any[
        "Variables"    n;
        "Lags"         m.p;
        "Observations" size(m.Y, 1);
        "AIC"          _fmt(m.aic; digits=2);
        "BIC"          _fmt(m.bic; digits=2);
        "HQIC"         _fmt(m.hqic; digits=2)
    ]
    _pretty_table(io, spec;
        title = "VAR($(m.p)) Model",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# Impulse Response Functions
# =============================================================================

"""
    ImpulseResponse{T} <: AbstractImpulseResponse

IRF results with optional confidence intervals.

Fields: values (H×n×n), ci_lower, ci_upper, horizon, variables, shocks, ci_type.
Internal: _draws (raw bootstrap/simulation draws for correct cumulative IRF), _conf_level.
"""
struct ImpulseResponse{T<:AbstractFloat} <: AbstractImpulseResponse
    values::Array{T,3}
    ci_lower::Array{T,3}
    ci_upper::Array{T,3}
    horizon::Int
    variables::Vector{String}
    shocks::Vector{String}
    ci_type::Symbol
    _draws::Union{Nothing, Array{T,4}}
    _conf_level::T
    # Reproducibility manifest (T246/#345): populated by the bootstrap-IRF path,
    # `nothing` for deterministic/analytic IRFs. Optional trailing field so every
    # existing positional call site is unchanged.
    manifest::Union{ReproManifest,Nothing}
end

# Backward-compatible constructors. The 9-arg form defaults the manifest to
# `nothing` (covers every existing positional call site); the 7-arg form
# additionally defaults draws/conf-level. Only the bootstrap path (core/irf.jl)
# passes `manifest=`.
ImpulseResponse{T}(values, ci_lower, ci_upper, horizon, variables, shocks, ci_type,
                   _draws, _conf_level; manifest=nothing) where {T} =
    ImpulseResponse{T}(values, ci_lower, ci_upper, horizon, variables, shocks, ci_type,
                       _draws, _conf_level, manifest)
ImpulseResponse{T}(values, ci_lower, ci_upper, horizon, variables, shocks, ci_type) where {T} =
    ImpulseResponse{T}(values, ci_lower, ci_upper, horizon, variables, shocks, ci_type, nothing, zero(T))

"""
    BayesianImpulseResponse{T} <: AbstractImpulseResponse

Bayesian IRF with posterior quantiles.

Fields: quantiles (H×n×n×q), point_estimate (H×n×n), horizon, variables, shocks, quantile_levels.
Internal: _draws (raw posterior draws for correct cumulative IRF).
"""
struct BayesianImpulseResponse{T<:AbstractFloat} <: AbstractImpulseResponse
    quantiles::Array{T,4}
    point_estimate::Array{T,3}
    horizon::Int
    variables::Vector{String}
    shocks::Vector{String}
    quantile_levels::Vector{T}
    _draws::Union{Nothing, Array{T,4}}
    # MC honesty counts (#244): posterior draws requested, usable, and dropped (failed to
    # solve / non-stationary / identification-rejected).
    n_requested::Int
    n_effective::Int
    n_failed::Int
end

# Backward-compatible constructors (pre-#244, no MC counts). Default to "no dropped draws":
# n_requested = n_effective = number of stacked draws (0 when draws absent), n_failed = 0.
_bir_ndraws(draws) = draws === nothing ? 0 : size(draws, 1)
BayesianImpulseResponse{T}(quantiles, point_estimate, horizon, variables, shocks, quantile_levels, draws) where {T} =
    BayesianImpulseResponse{T}(quantiles, point_estimate, horizon, variables, shocks, quantile_levels, draws,
                               _bir_ndraws(draws), _bir_ndraws(draws), 0)
BayesianImpulseResponse{T}(quantiles, point_estimate, horizon, variables, shocks, quantile_levels) where {T} =
    BayesianImpulseResponse{T}(quantiles, point_estimate, horizon, variables, shocks, quantile_levels, nothing)

# =============================================================================
# FEVD
# =============================================================================

"""FEVD results: decomposition (n×n×H), proportions, variable/shock names."""
struct FEVD{T<:AbstractFloat} <: AbstractFEVD
    decomposition::Array{T,3}
    proportions::Array{T,3}
    variables::Vector{String}
    shocks::Vector{String}
end

"""Bayesian FEVD with posterior quantiles."""
struct BayesianFEVD{T<:AbstractFloat} <: AbstractFEVD
    quantiles::Array{T,4}
    point_estimate::Array{T,3}
    horizon::Int
    variables::Vector{String}
    shocks::Vector{String}
    quantile_levels::Vector{T}
    # MC honesty counts (#244): posterior draws requested, usable, and dropped.
    n_requested::Int
    n_effective::Int
    n_failed::Int
end

# Backward-compatible constructor (pre-#244, no MC counts ⇒ untracked, no dropped draws).
BayesianFEVD{T}(quantiles, point_estimate, horizon, variables, shocks, quantile_levels) where {T} =
    BayesianFEVD{T}(quantiles, point_estimate, horizon, variables, shocks, quantile_levels, 0, 0, 0)

# =============================================================================
# Priors
# =============================================================================

"""
    MinnesotaHyperparameters{T} <: AbstractPrior

Minnesota prior hyperparameters (Bańbura–Giannone–Reichlin stacked-dummy parameterization).

- `tau`    — overall shrinkage, as **inverse tightness**: dummy observations are divided by `tau`,
             so LARGER `tau` ⇒ LOOSER prior. (Opposite direction to the reference `BVAR_`'s
             `mnprior.tight`, which multiplies.)
- `decay`  — lag decay; higher lags shrink toward zero faster (scaling `lag^decay`).
- `lambda` — weight on the **sum-of-coefficients** prior (shrinks toward Σₗ Aₗ = I).
- `mu`     — weight on the **co-persistence / dummy-initial-observation** prior.
- `omega`  — weight on the prior for the residual covariance.

Reference-naming caveat (audit F-03): in Ferroni–Canova `BVAR_`/`rfvar3`, `lambda` is
co-persistence and `mu` is own/sum-of-coefficients — i.e. our `lambda`/`mu` roles are SWAPPED
relative to that toolbox. The defaults `lambda=5, mu=2` reuse the reference's numeric guidance
under our role assignment.
"""
struct MinnesotaHyperparameters{T<:AbstractFloat} <: AbstractPrior
    tau::T
    decay::T
    lambda::T
    mu::T
    omega::T
end

function MinnesotaHyperparameters(; tau::Real=3.0, decay::Real=0.5,
                                   lambda::Real=5.0, mu::Real=2.0, omega::Real=2.0)
    T = promote_type(typeof(tau), typeof(decay), typeof(lambda), typeof(mu), typeof(omega))
    MinnesotaHyperparameters{T}(T(tau), T(decay), T(lambda), T(mu), T(omega))
end

# =============================================================================
# Sign-Identified Set (Baumeister & Hamilton 2015)
# =============================================================================

"""
    SignIdentifiedSet{T} <: AbstractAnalysisResult

Full identified set from sign-restricted SVAR identification.

Stores all accepted rotation matrices and corresponding IRFs, enabling
characterization of the identified set (Baumeister & Hamilton, 2015).

Fields:
- `Q_draws::Vector{Matrix{T}}` — accepted rotation matrices
- `irf_draws::Array{T,4}` — stacked IRFs (n_accepted × horizon × n × n)
- `n_accepted::Int` — number of accepted draws
- `n_total::Int` — total draws attempted
- `acceptance_rate::T` — fraction accepted
- `variables::Vector{String}` — variable names
- `shocks::Vector{String}` — shock names
"""
struct SignIdentifiedSet{T<:AbstractFloat} <: AbstractAnalysisResult
    Q_draws::Vector{Matrix{T}}
    irf_draws::Array{T,4}
    n_accepted::Int
    n_total::Int
    acceptance_rate::T
    variables::Vector{String}
    shocks::Vector{String}
end

function Base.show(io::IO, s::SignIdentifiedSet{T}) where {T}
    println(io, "Sign-Identified Set")
    println(io, "  Accepted draws: $(s.n_accepted) / $(s.n_total) ($(round(s.acceptance_rate * 100, digits=1))%)")
    println(io, "  Variables: $(length(s.variables))")
    if s.n_accepted > 0
        H = size(s.irf_draws, 2)
        println(io, "  IRF horizon: $H")
    end
end
