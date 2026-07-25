# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Observation handling for trending data in DSGE estimation (T240 / #339).

DSGE observables are stationary deviations from steady state, but macro data
(GDP, consumption, investment) trend. This file provides the two standard ways
of reconciling the two, matching Dynare's `prefilter` and `observation_trends`:

- [`apply_prefilter`](@ref) / [`invert_prefilter`](@ref) — built-in observable
  transforms (`:demean`, `:first_difference`, `:linear_detrend`, `:hp`) recorded
  on a [`PrefilterSpec`](@ref) so forecasts can be mapped back to the observed scale.
- [`ObservationTrends`](@ref) — deterministic constant/linear/quadratic terms in the
  measurement equation `yₜ^obs = d + Z·sₜ + trendₜ + vₜ`, with coefficients either
  fixed numbers or model-parameter symbols (hence estimable like any other parameter).
- [`detect_trend`](@ref) — the guidance warning emitted when obviously trending data
  is fed to a stationary model with neither of the above.

References:
- Dynare Reference Manual, `prefilter` and `observation_trends`.
- Fernández-Villaverde, J., Rubio-Ramírez, J. F. & Schorfheide, F. (2016).
  "Solution and Estimation Methods for DSGE Models", *Handbook of Macroeconomics* 2A.
"""

using LinearAlgebra
using Statistics

const _PREFILTER_TRANSFORMS = (:none, :demean, :first_difference, :linear_detrend, :hp)

# =============================================================================
# PrefilterSpec — the record of what was removed, so forecasts can be inverted
# =============================================================================

"""
    PrefilterSpec{T}

Record of the observable transform applied before estimation, sufficient to invert
filtered quantities back to the observed scale via [`invert_prefilter`](@ref).

# Fields
- `transform::Symbol` — one of `:none`, `:demean`, `:first_difference`, `:linear_detrend`, `:hp`
- `observables::Vector{Symbol}` — observable names, in row order
- `intercepts::Vector{T}` — removed constant per observable (`:demean` mean, `:linear_detrend` intercept)
- `slopes::Vector{T}` — removed linear slope per observable (`:linear_detrend`)
- `initial_levels::Vector{T}` — first observed level per observable (`:first_difference` anchor)
- `final_levels::Vector{T}` — last observed level per observable (forecast anchor)
- `removed::Matrix{T}` — `n_obs × T_kept` component subtracted in sample (zeros for `:none`)
- `lambda::T` — HP smoothing parameter (`:hp` only)
- `n_dropped::Int` — leading observations dropped by the transform (1 for `:first_difference`)
"""
struct PrefilterSpec{T<:AbstractFloat}
    transform::Symbol
    observables::Vector{Symbol}
    intercepts::Vector{T}
    slopes::Vector{T}
    initial_levels::Vector{T}
    final_levels::Vector{T}
    removed::Matrix{T}
    lambda::T
    n_dropped::Int

    function PrefilterSpec{T}(transform, observables, intercepts, slopes,
                              initial_levels, final_levels, removed, lambda,
                              n_dropped) where {T<:AbstractFloat}
        transform in _PREFILTER_TRANSFORMS || throw(ArgumentError(
            "prefilter transform must be one of $(_PREFILTER_TRANSFORMS), got :$transform"))
        new{T}(transform, Vector{Symbol}(observables), Vector{T}(intercepts),
               Vector{T}(slopes), Vector{T}(initial_levels), Vector{T}(final_levels),
               Matrix{T}(removed), T(lambda), Int(n_dropped))
    end
end

Base.show(io::IO, pf::PrefilterSpec) =
    print(io, "PrefilterSpec(:", pf.transform, ", n_obs=", length(pf.observables),
          ", n_dropped=", pf.n_dropped, ")")

# =============================================================================
# apply_prefilter / invert_prefilter
# =============================================================================

"""
    apply_prefilter(data, transform; observables=Symbol[], lambda=1600) -> (filtered, spec)

Apply a built-in observable transform to `data` (an `n_obs × T_obs` matrix, i.e. the
internal Kalman orientation with **time in columns**) before estimation.

# Transforms
- `:none` — pass through unchanged.
- `:demean` — subtract each observable's sample mean.
- `:first_difference` — `Δyₜ = yₜ − yₜ₋₁`; drops the first observation.
- `:linear_detrend` — subtract the OLS fit on `[1, t]` per observable.
- `:hp` — subtract the Hodrick–Prescott trend (keeps the cycle); `lambda` is the
  HP smoothing parameter (1600 quarterly, 129600 monthly, 6.25 annual).

Returns the filtered data (same orientation) and the [`PrefilterSpec`](@ref) recording
what was removed.
"""
function apply_prefilter(data::AbstractMatrix{T}, transform::Symbol;
                         observables::Vector{Symbol}=Symbol[],
                         lambda::Real=1600) where {T<:AbstractFloat}
    transform in _PREFILTER_TRANSFORMS || throw(ArgumentError(
        "prefilter must be one of $(_PREFILTER_TRANSFORMS), got :$transform"))
    n_obs, T_obs = size(data)
    n_obs == 0 && throw(ArgumentError("data must have at least one observable"))
    obs = isempty(observables) ? [Symbol("y", i) for i in 1:n_obs] : copy(observables)
    length(obs) == n_obs || throw(ArgumentError(
        "observables length ($(length(obs))) must match data rows ($n_obs)"))

    initial_levels = T[data[i, 1] for i in 1:n_obs]
    final_levels = T[data[i, end] for i in 1:n_obs]
    intercepts = zeros(T, n_obs)
    slopes = zeros(T, n_obs)
    lam = T(lambda)

    if transform === :none
        removed = zeros(T, n_obs, T_obs)
        filtered = Matrix{T}(data)

    elseif transform === :demean
        removed = zeros(T, n_obs, T_obs)
        filtered = Matrix{T}(undef, n_obs, T_obs)
        for i in 1:n_obs
            mu = mean(view(data, i, :))
            intercepts[i] = mu
            @views filtered[i, :] .= data[i, :] .- mu
            @views removed[i, :] .= mu
        end

    elseif transform === :first_difference
        T_obs >= 2 || throw(ArgumentError(
            ":first_difference requires at least 2 observations, got $T_obs"))
        filtered = Matrix{T}(undef, n_obs, T_obs - 1)
        for i in 1:n_obs, t in 2:T_obs
            filtered[i, t-1] = data[i, t] - data[i, t-1]
        end
        removed = zeros(T, n_obs, T_obs - 1)

    elseif transform === :linear_detrend
        T_obs >= 3 || throw(ArgumentError(
            ":linear_detrend requires at least 3 observations, got $T_obs"))
        tt = T[T(t) for t in 1:T_obs]
        X = hcat(ones(T, T_obs), tt)
        XtXinv = inv(Symmetric(X' * X))
        filtered = Matrix{T}(undef, n_obs, T_obs)
        removed = Matrix{T}(undef, n_obs, T_obs)
        for i in 1:n_obs
            yi = Vector{T}(view(data, i, :))
            b = XtXinv * (X' * yi)
            intercepts[i] = b[1]
            slopes[i] = b[2]
            fit = X * b
            @views removed[i, :] .= fit
            @views filtered[i, :] .= yi .- fit
        end

    else  # :hp
        T_obs >= 3 || throw(ArgumentError(
            ":hp requires at least 3 observations, got $T_obs"))
        filtered = Matrix{T}(undef, n_obs, T_obs)
        removed = Matrix{T}(undef, n_obs, T_obs)
        for i in 1:n_obs
            res = hp_filter(Vector{T}(view(data, i, :)); lambda=lam)
            @views filtered[i, :] .= res.cycle
            @views removed[i, :] .= res.trend
        end
    end

    n_dropped = transform === :first_difference ? 1 : 0
    spec = PrefilterSpec{T}(transform, obs, intercepts, slopes, initial_levels,
                            final_levels, removed, lam, n_dropped)
    return filtered, spec
end

"""
    invert_prefilter(pf::PrefilterSpec, y; time_offset=0, level0=nothing) -> Matrix

Map filtered quantities `y` (an `n_obs × H` matrix in the model/filtered scale) back to
the observed scale implied by `pf`.

`time_offset` is the number of periods between the start of the estimation sample and
the first column of `y`; pass `0` for in-sample values and the estimation sample length
for forecasts. `level0` overrides the level anchor used by `:first_difference`
(defaults to the last observed level).

Inversion rules:
- `:none` — identity.
- `:demean` — add back the sample mean.
- `:linear_detrend` — add back `intercept + slope·t`, extrapolating `t` past the sample.
- `:hp` — add back the in-sample HP trend where it exists; beyond the sample the trend is
  linearly extrapolated from its last two points (the HP trend is not defined out of sample).
- `:first_difference` — cumulate the differences onto `level0`.
"""
function invert_prefilter(pf::PrefilterSpec{T}, y::AbstractMatrix;
                          time_offset::Int=0,
                          level0::Union{Nothing,AbstractVector}=nothing) where {T<:AbstractFloat}
    ym = Matrix{T}(y)
    n_obs, H = size(ym)
    n_obs == length(pf.observables) || throw(ArgumentError(
        "y has $n_obs rows but the prefilter covers $(length(pf.observables)) observables"))
    time_offset >= 0 || throw(ArgumentError("time_offset must be non-negative"))

    if pf.transform === :none
        return ym

    elseif pf.transform === :demean
        return ym .+ pf.intercepts

    elseif pf.transform === :linear_detrend
        out = similar(ym)
        for i in 1:n_obs, h in 1:H
            t = T(time_offset + h)
            out[i, h] = ym[i, h] + pf.intercepts[i] + pf.slopes[i] * t
        end
        return out

    elseif pf.transform === :hp
        T_in = size(pf.removed, 2)
        out = similar(ym)
        for i in 1:n_obs
            # Slope for extrapolation beyond the estimation sample
            slope = T_in >= 2 ? pf.removed[i, T_in] - pf.removed[i, T_in-1] : zero(T)
            for h in 1:H
                t = time_offset + h
                trend = t <= T_in ? pf.removed[i, t] :
                        pf.removed[i, T_in] + slope * T(t - T_in)
                out[i, h] = ym[i, h] + trend
            end
        end
        return out

    else  # :first_difference
        anchor = level0 === nothing ? pf.final_levels : Vector{T}(level0)
        length(anchor) == n_obs || throw(ArgumentError(
            "level0 length ($(length(anchor))) must match observables ($n_obs)"))
        out = similar(ym)
        for i in 1:n_obs
            acc = anchor[i]
            for h in 1:H
                acc += ym[i, h]
                out[i, h] = acc
            end
        end
        return out
    end
end

# =============================================================================
# ObservationTrends — deterministic trends in the measurement equation
# =============================================================================

"""
    ObservationTrends{T}

Deterministic trend terms in the DSGE measurement equation

```math
y_t^{obs} = d + Z s_t + \\underbrace{(c_0 + c_1 t + c_2 t^2)}_{trend_t} + v_t
```

with one `(c₀, c₁, c₂)` triple per observable. Each coefficient is either a fixed
number or a `Symbol` naming a **model parameter** — in which case it is estimated
like any other parameter simply by giving it a prior (Dynare's `observation_trends`
semantics, where the trend is written in terms of declared model parameters).

Build one with [`observation_trends`](@ref) rather than calling the constructor directly.
"""
struct ObservationTrends{T<:AbstractFloat}
    observables::Vector{Symbol}
    constants::Vector{Union{T,Symbol}}
    linears::Vector{Union{T,Symbol}}
    quadratics::Vector{Union{T,Symbol}}

    function ObservationTrends{T}(observables, constants, linears,
                                  quadratics) where {T<:AbstractFloat}
        n = length(observables)
        (length(constants) == n && length(linears) == n && length(quadratics) == n) ||
            throw(ArgumentError("trend coefficient vectors must all have length $n"))
        new{T}(Vector{Symbol}(observables),
               Vector{Union{T,Symbol}}(constants),
               Vector{Union{T,Symbol}}(linears),
               Vector{Union{T,Symbol}}(quadratics))
    end
end

function Base.show(io::IO, tr::ObservationTrends{T}) where {T}
    print(io, "ObservationTrends{", T, "} over ", length(tr.observables), " observable(s)")
    for i in eachindex(tr.observables)
        print(io, "\n  ", tr.observables[i], ": c0=", tr.constants[i],
              ", c1=", tr.linears[i], ", c2=", tr.quadratics[i])
    end
end

"""
    observation_trends(userspec, observables, ::Type{T}=Float64) -> ObservationTrends{T}

Build an [`ObservationTrends`](@ref) for `observables` from a user specification. This is
the same conversion `estimate_dsge_bayes(...; observation_trends=...)` applies internally,
exposed so trends can be built and inspected ahead of estimation.

Accepted per-observable values in a `Dict{Symbol,<:Any}`:
- a `Real` or `Symbol` — the **linear** (growth) term, matching Dynare's convention;
- a `NamedTuple` with any of `constant`, `linear`, `quadratic`;
- a `Tuple`/`AbstractVector` of length ≤ 3 read as `(constant, linear, quadratic)`.

Observables absent from the dict get a zero trend. Passing an `ObservationTrends`
re-validates it against `observables`.
"""
observation_trends(userspec, observables::Vector{Symbol}, ::Type{T}=Float64) where {T<:AbstractFloat} =
    _build_observation_trends(userspec, observables, T)

function _build_observation_trends(userspec, observables::Vector{Symbol},
                                   ::Type{T}) where {T<:AbstractFloat}
    if userspec isa ObservationTrends
        userspec.observables == observables || throw(ArgumentError(
            "ObservationTrends observables $(userspec.observables) do not match the " *
            "estimation observables $observables"))
        return ObservationTrends{T}(userspec.observables, userspec.constants,
                                    userspec.linears, userspec.quadratics)
    end
    userspec isa AbstractDict || throw(ArgumentError(
        "observation_trends must be a Dict{Symbol,...} (or an ObservationTrends), " *
        "got $(typeof(userspec))"))

    for k in keys(userspec)
        k in observables || throw(ArgumentError(
            "observation_trends names :$k, which is not among the observables $observables"))
    end

    n = length(observables)
    c0 = Vector{Union{T,Symbol}}(undef, n)
    c1 = Vector{Union{T,Symbol}}(undef, n)
    c2 = Vector{Union{T,Symbol}}(undef, n)
    for (i, obs) in enumerate(observables)
        c0[i], c1[i], c2[i] = _parse_trend_entry(get(userspec, obs, nothing), obs, T)
    end
    return ObservationTrends{T}(observables, c0, c1, c2)
end

_coerce_trend_term(v, obs::Symbol, ::Type{T}) where {T} =
    v isa Symbol ? v :
    v isa Real ? T(v) :
    throw(ArgumentError("observation_trends[:$obs] terms must be Real or Symbol, got $(typeof(v))"))

function _parse_trend_entry(entry, obs::Symbol, ::Type{T}) where {T}
    z = zero(T)
    entry === nothing && return (z, z, z)
    if entry isa Symbol || entry isa Real
        return (z, _coerce_trend_term(entry, obs, T), z)
    elseif entry isa NamedTuple
        for k in keys(entry)
            k in (:constant, :linear, :quadratic) || throw(ArgumentError(
                "observation_trends[:$obs] NamedTuple keys must be :constant/:linear/" *
                ":quadratic, got :$k"))
        end
        return (_coerce_trend_term(get(entry, :constant, z), obs, T),
                _coerce_trend_term(get(entry, :linear, z), obs, T),
                _coerce_trend_term(get(entry, :quadratic, z), obs, T))
    elseif entry isa Tuple || entry isa AbstractVector
        length(entry) <= 3 || throw(ArgumentError(
            "observation_trends[:$obs] must have at most 3 terms (constant, linear, " *
            "quadratic), got $(length(entry))"))
        terms = Any[z, z, z]
        for (j, v) in enumerate(entry)
            terms[j] = _coerce_trend_term(v, obs, T)
        end
        return (terms[1], terms[2], terms[3])
    end
    throw(ArgumentError(
        "observation_trends[:$obs] must be a Real, Symbol, NamedTuple, Tuple or Vector, " *
        "got $(typeof(entry))"))
end

"""
    _trend_param_symbols(tr::ObservationTrends) -> Vector{Symbol}

Distinct model-parameter symbols referenced by the trend coefficients.
"""
function _trend_param_symbols(tr::ObservationTrends)
    out = Symbol[]
    for v in Iterators.flatten((tr.constants, tr.linears, tr.quadratics))
        v isa Symbol && !(v in out) && push!(out, v)
    end
    return out
end

"""
    _has_estimated_terms(tr) -> Bool

`true` when any trend coefficient is a parameter symbol (so the trend is θ-dependent
and must be recomputed at every likelihood evaluation).
"""
_has_estimated_terms(tr::ObservationTrends) = !isempty(_trend_param_symbols(tr))
_has_estimated_terms(::Nothing) = false

"""
    _validate_trend_params(tr, spec)

Every parameter symbol used by a trend must be a declared model parameter, so that
`_respec` carries it and the likelihood can read its current value.
"""
function _validate_trend_params(tr::ObservationTrends, spec::DSGESpec)
    for s in _trend_param_symbols(tr)
        haskey(spec.param_values, s) || throw(ArgumentError(
            "observation_trends references :$s, which is not a declared model parameter " *
            "$(sort(collect(keys(spec.param_values)))). Declare it in the @dsge params " *
            "block (it need not appear in any equation) so it can be fixed or estimated."))
    end
    return nothing
end
_validate_trend_params(::Nothing, ::DSGESpec) = nothing

_trend_value(v::Symbol, pv) = pv[v]
_trend_value(v::Real, pv) = v

"""
    _trend_matrix(tr, param_values, T_obs, ::Type{T}) -> Matrix{T}

Evaluate the deterministic trend `c₀ + c₁·t + c₂·t²` for `t = 1:T_obs`, resolving
symbolic coefficients against `param_values`. Returns an `n_obs × T_obs` matrix.
"""
function _trend_matrix(tr::ObservationTrends{T}, param_values, T_obs::Int,
                       ::Type{T}) where {T<:AbstractFloat}
    n = length(tr.observables)
    M = Matrix{T}(undef, n, T_obs)
    @inbounds for i in 1:n
        a = T(_trend_value(tr.constants[i], param_values))
        b = T(_trend_value(tr.linears[i], param_values))
        c = T(_trend_value(tr.quadratics[i], param_values))
        for t in 1:T_obs
            tt = T(t)
            M[i, t] = a + b * tt + c * tt * tt
        end
    end
    return M
end
_trend_matrix(::Nothing, param_values, T_obs::Int, ::Type{T}) where {T} = zeros(T, 0, T_obs)

# =============================================================================
# detect_trend — guidance warning at estimation entry
# =============================================================================

"""
    detect_trend(y::AbstractVector; tstat_threshold=4.0) -> NamedTuple

Test a single series for an obvious deterministic trend by OLS on `[1, t]`, using a
**Newey–West HAC** standard error for the slope with automatic bandwidth.

The HAC correction matters here: DSGE observables are highly persistent, and a
classical OLS trend t-statistic on a persistent-but-stationary series is spuriously
large (Granger & Newbold 1974), which would make the estimation-entry guidance fire
on perfectly well-specified inputs.

Returns `(slope, tstat, trending)` where `trending` is `true` when `|t|` on the slope
exceeds `tstat_threshold`. This is a cheap guidance heuristic, **not** a unit-root test
— use [`adf_test`](@ref) for inference.
"""
function detect_trend(y::AbstractVector{<:Real}; tstat_threshold::Real=4.0)
    T_obs = length(y)
    yv = float.(collect(y))
    F = eltype(yv)
    T_obs >= 8 || return (slope=zero(F), tstat=zero(F), trending=false)
    tt = F[F(t) for t in 1:T_obs]
    X = hcat(ones(F, T_obs), tt)
    XtXinv = inv(Symmetric(X' * X))
    b = XtXinv * (X' * yv)
    resid = yv .- X * b
    V = newey_west(X, resid; XtX_inv=Matrix{F}(XtXinv))
    se = sqrt(max(V[2, 2], zero(F)))
    tstat = se > 0 && isfinite(se) ? b[2] / se : zero(F)
    return (slope=b[2], tstat=tstat, trending=abs(tstat) > F(tstat_threshold))
end

"""
    detect_trend(data::AbstractMatrix; names=Symbol[], tstat_threshold=4.0, warn=false)

Column-wise trend detection over a `T×n` matrix (package convention: time in rows).
Returns a `Vector{Bool}`, one flag per column. With `warn=true`, emits the estimation
guidance warning naming the trending series.
"""
function detect_trend(data::AbstractMatrix{<:Real}; names::Vector{Symbol}=Symbol[],
                      tstat_threshold::Real=4.0, warn::Bool=false)
    n = size(data, 2)
    flags = [detect_trend(view(data, :, j); tstat_threshold=tstat_threshold).trending
             for j in 1:n]
    if warn && any(flags)
        labels = isempty(names) ? [Symbol("y", j) for j in 1:n] : names
        _warn_trending(labels[flags])
    end
    return flags
end

function _warn_trending(trending::Vector{Symbol})
    @warn "Observables $(trending) show a strong deterministic trend, but the model is " *
          "estimated on stationary deviations from steady state with no prefilter and no " *
          "observation trends. Pass `prefilter=:demean/:first_difference/:linear_detrend/:hp` " *
          "or `observation_trends=Dict(...)` to reconcile the data with the model."
    return nothing
end

"""
    _warn_untransformed_trends(data_mat, observables; tstat_threshold=4.0)

Estimation-entry guidance check. `data_mat` is `n_obs × T_obs` (Kalman orientation).
"""
function _warn_untransformed_trends(data_mat::AbstractMatrix, observables::Vector{Symbol};
                                    tstat_threshold::Real=4.0)
    n_obs = size(data_mat, 1)
    flags = [detect_trend(view(data_mat, i, :); tstat_threshold=tstat_threshold).trending
             for i in 1:n_obs]
    any(flags) && _warn_trending(observables[flags])
    return flags
end
