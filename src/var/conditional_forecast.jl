# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Conditional forecasting / scenario analysis for VAR and BVAR models (T241 / #340).

Implements Waggoner & Zha (1999): constrain selected variables to specified paths over
the forecast horizon, and draw the future **structural** shocks from the conditional
distribution those constraints imply. The remaining (unconstrained) shock directions
stay random, so the conditional forecast carries genuine uncertainty rather than a
single deterministic path.

Provides:
- `ForecastCondition` / `forecast_condition` — one constraint on (variable, horizon)
- `ConditionalForecast` — the result container (an `AbstractForecastResult`)
- `conditional_forecast(::VARModel, ...)` / `conditional_forecast(::BVARPosterior, ...)`

References:
- Waggoner, D. F. & Zha, T. (1999). Conditional Forecasts in Dynamic Multivariate Models.
  *Review of Economics and Statistics*, 81(4), 639-651.
- Antolín-Díaz, J., Petrella, I. & Rubio-Ramírez, J. F. (2021). Structural Scenario
  Analysis with SVARs. *Journal of Monetary Economics*, 117, 798-815.
"""

using LinearAlgebra
using Statistics
using Random

# =============================================================================
# ForecastCondition
# =============================================================================

"""
    ForecastCondition{T}

One conditioning restriction: `variable` takes `value` at forecast `horizon`.

`sd == 0` is a **hard** condition (the path is hit exactly). `sd > 0` is a **soft**
condition — the restriction is treated as a noisy observation with standard deviation
`sd`, so the conditional shock distribution shrinks toward the target instead of
pinning it, and `sd → 0` recovers the hard case.

Build with [`forecast_condition`](@ref).
"""
struct ForecastCondition{T<:AbstractFloat}
    variable::Union{Int,String,Symbol}
    horizon::Int
    value::T
    sd::T

    function ForecastCondition{T}(variable, horizon::Integer, value::Real,
                                  sd::Real=zero(T)) where {T<:AbstractFloat}
        horizon >= 1 || throw(ArgumentError("condition horizon must be ≥ 1, got $horizon"))
        sd >= 0 || throw(ArgumentError("condition sd must be non-negative, got $sd"))
        new{T}(variable, Int(horizon), T(value), T(sd))
    end
end

"""
    forecast_condition(variable, horizon, value; sd=0.0) -> ForecastCondition

Build one conditioning restriction for [`conditional_forecast`](@ref).

`variable` is a column index, name `String`, or `Symbol`. `sd=0` (the default) makes the
condition hard; a positive `sd` makes it soft (a target with tolerance).

```julia
conds = [forecast_condition("INFL", 1, 2.0),
         forecast_condition("INFL", 2, 2.0; sd=0.25)]
```
"""
forecast_condition(variable::Union{Int,String,Symbol}, horizon::Integer, value::Real;
                   sd::Real=0.0) =
    ForecastCondition{Float64}(variable, horizon, value, sd)

Base.show(io::IO, c::ForecastCondition) =
    print(io, "ForecastCondition(", repr(c.variable), ", h=", c.horizon,
          ", value=", c.value, c.sd > 0 ? ", sd=$(c.sd)" : "", ")")

# =============================================================================
# ConditionalForecast
# =============================================================================

"""
    ConditionalForecast{T} <: AbstractForecastResult{T}

Waggoner-Zha conditional forecast.

Fields:
- `forecast::Matrix{T}` — `h × n` conditional mean path
- `ci_lower::Matrix{T}` / `ci_upper::Matrix{T}` — `h × n` band bounds
- `horizon::Int` — forecast horizon
- `conf_level::T` — band coverage
- `varnames::Vector{String}` — variable names
- `conditions::Vector{ForecastCondition{T}}` — the restrictions imposed (indices resolved)
- `unconditional::Matrix{T}` — `h × n` unconditional forecast, for comparison
- `shocks::Matrix{T}` — `h × n_shocks` mean structural shocks implied by the conditions
- `identification::Symbol` — `:cholesky` or `:custom`
- `n_draws::Int` — draws used for the bands

The standard accessors `point_forecast`, `lower_bound`, `upper_bound` and
`forecast_horizon` work on this type.
"""
struct ConditionalForecast{T<:AbstractFloat} <: AbstractForecastResult{T}
    forecast::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    conf_level::T
    varnames::Vector{String}
    conditions::Vector{ForecastCondition{T}}
    unconditional::Matrix{T}
    shocks::Matrix{T}
    identification::Symbol
    n_draws::Int
end

function Base.show(io::IO, fc::ConditionalForecast{T}) where {T}
    n_vars = length(fc.varnames)
    ci_pct = round(Int, 100 * fc.conf_level)
    n_hard = count(c -> c.sd == 0, fc.conditions)

    spec = Any[
        "Horizon"        fc.horizon;
        "Variables"      n_vars;
        "Conditions"     "$(length(fc.conditions)) ($(n_hard) hard)";
        "Identification" string(fc.identification);
        "Draws"          fc.n_draws;
        "Conf. level"    "$(ci_pct)%"
    ]
    _pretty_table(io, spec;
        title = "Conditional Forecast (Waggoner-Zha)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    cond = Matrix{Any}(undef, length(fc.conditions), 4)
    for (i, c) in enumerate(fc.conditions)
        cond[i, 1] = fc.varnames[c.variable]
        cond[i, 2] = c.horizon
        cond[i, 3] = _fmt(c.value)
        cond[i, 4] = c.sd == 0 ? "hard" : "soft (sd=$(_fmt(c.sd)))"
    end
    isempty(fc.conditions) || _pretty_table(io, cond;
        title = "Conditions",
        column_labels = ["Variable", "Horizon", "Value", "Type"],
        alignment = [:l, :r, :r, :l],
    )

    lo_label = _fmt_pct((1 - fc.conf_level) / 2)
    hi_label = _fmt_pct((1 + fc.conf_level) / 2)
    for vi in 1:n_vars
        data = Matrix{Any}(undef, fc.horizon, 5)
        for h in 1:fc.horizon
            data[h, 1] = h
            data[h, 2] = _fmt(fc.forecast[h, vi])
            data[h, 3] = _fmt(fc.ci_lower[h, vi])
            data[h, 4] = _fmt(fc.ci_upper[h, vi])
            data[h, 5] = _fmt(fc.unconditional[h, vi])
        end
        _pretty_table(io, data;
            title = "$(fc.varnames[vi])",
            column_labels = ["h", "Conditional", lo_label, hi_label, "Unconditional"],
            alignment = [:r, :r, :r, :r, :r],
        )
    end
    return nothing
end

# =============================================================================
# Internal helpers
# =============================================================================

"""
    _cf_resolve_var(v, varnames) -> Int

Resolve a condition's variable (index, `String`, or `Symbol`) to a column index.
"""
function _cf_resolve_var(v::Union{Int,String,Symbol}, varnames::Vector{String})
    if v isa Int
        1 <= v <= length(varnames) || throw(ArgumentError(
            "condition variable index $v out of range 1:$(length(varnames))"))
        return v
    end
    name = string(v)
    idx = findfirst(==(name), varnames)
    idx === nothing && throw(ArgumentError(
        "condition variable '$name' not found. Available: $varnames"))
    return idx
end

"""
    _cf_normalize_conditions(conditions, varnames, h, ::Type{T}) -> Vector{ForecastCondition{T}}

Accept either a vector of [`ForecastCondition`](@ref) or a `Dict` keyed by
`(variable, horizon)` whose values are `value` or `(value, sd)`, and return the
resolved, horizon-validated, horizon-sorted condition list.
"""
function _cf_normalize_conditions(conditions, varnames::Vector{String}, h::Int,
                                  ::Type{T}) where {T<:AbstractFloat}
    out = ForecastCondition{T}[]
    if conditions isa AbstractDict
        for (k, v) in conditions
            (k isa Tuple && length(k) == 2) || throw(ArgumentError(
                "condition Dict keys must be (variable, horizon) tuples, got $(repr(k))"))
            var, hh = k
            val, sd = if v isa Tuple || v isa AbstractVector
                length(v) == 2 || throw(ArgumentError(
                    "condition value must be `value` or `(value, sd)`, got $(repr(v))"))
                (v[1], v[2])
            else
                (v, zero(T))
            end
            push!(out, ForecastCondition{T}(var, hh, val, sd))
        end
    elseif conditions isa AbstractVector
        for c in conditions
            c isa ForecastCondition || throw(ArgumentError(
                "condition vectors must hold ForecastCondition values, got $(typeof(c))"))
            push!(out, ForecastCondition{T}(c.variable, c.horizon, c.value, c.sd))
        end
    else
        throw(ArgumentError(
            "conditions must be a Vector{ForecastCondition} or a Dict keyed by " *
            "(variable, horizon), got $(typeof(conditions))"))
    end
    isempty(out) && throw(ArgumentError("conditions must be non-empty"))

    resolved = ForecastCondition{T}[]
    for c in out
        c.horizon <= h || throw(ArgumentError(
            "condition horizon $(c.horizon) exceeds the forecast horizon $h"))
        push!(resolved, ForecastCondition{T}(_cf_resolve_var(c.variable, varnames),
                                             c.horizon, c.value, c.sd))
    end
    # Deterministic order (horizon, then variable) so R is reproducible across runs
    sort!(resolved; by=c -> (c.horizon, c.variable::Int))
    return resolved
end

"""
    _cf_uncond_path(intercept, A, history, h) -> Matrix

Iterate the VAR forward `h` periods from `history` (the last `p` observations, oldest
first) with zero shocks. Returns an `h × n` matrix.
"""
function _cf_uncond_path(intercept::AbstractVector{T}, A::Vector{Matrix{T}},
                         history::AbstractMatrix{T}, h::Int) where {T<:AbstractFloat}
    p = length(A)
    n = length(intercept)
    hist = Matrix{T}(history)
    out = Matrix{T}(undef, h, n)
    y_hat = Vector{T}(undef, n)
    @inbounds for step in 1:h
        copyto!(y_hat, intercept)
        for lag in 1:p
            y_hat .+= A[lag] * @view(hist[end-lag+1, :])
        end
        out[step, :] = y_hat
        for r in 1:(p-1)
            @views hist[r, :] .= hist[r+1, :]
        end
        @views hist[p, :] .= y_hat
    end
    return out
end

"""
    _cf_structural_ma(A, P, h) -> Array{T,3}

Structural moving-average coefficients `Ψ_s = Φ_s · P` for `s = 0 … h-1`, returned as an
`h × n × n_shocks` array with `Ψ[s+1]` at index `s+1` (so `Ψ[1] == P`). This is the same
recursion `compute_irf` runs, expressed on raw `(A, P)` so it can be reused per posterior
draw without building a `VARModel`.
"""
function _cf_structural_ma(A::Vector{Matrix{T}}, P::AbstractMatrix{T},
                           h::Int) where {T<:AbstractFloat}
    n = size(P, 1)
    p = length(A)
    Psi = zeros(T, h, n, size(P, 2))
    Phi = [zeros(T, n, n) for _ in 1:h]
    copyto!(Phi[1], I(n))
    Psi[1, :, :] = P
    temp = zeros(T, n, n)
    scratch = zeros(T, n, n)
    @inbounds for s in 2:h
        fill!(temp, zero(T))
        for j in 1:min(p, s - 1)
            mul!(scratch, A[j], Phi[s-j])
            temp .+= scratch
        end
        Phi[s] .= temp
        Psi[s, :, :] = temp * P
    end
    return Psi
end

"""
    _cf_build_R(conds, Psi, uncond, h, n_shocks) -> (R, r, Omega)

Stack the Waggoner-Zha restriction system `R ε = r` over the vectorized future shocks
`ε = (ε_{T+1}', …, ε_{T+h}')'`.

Condition `(i, k, v)` contributes the row whose block `j ≤ k` is `Ψ_{k-j}[i, :]`, and the
right-hand side `v − ŷ_{T+k,i}` — the gap between the target and the unconditional path.
`Omega` is the diagonal soft-condition variance (zero for hard conditions).
"""
function _cf_build_R(conds::Vector{ForecastCondition{T}}, Psi::Array{T,3},
                     uncond::AbstractMatrix{T}, h::Int,
                     n_shocks::Int) where {T<:AbstractFloat}
    m = length(conds)
    R = zeros(T, m, n_shocks * h)
    r = Vector{T}(undef, m)
    om = Vector{T}(undef, m)
    @inbounds for (row, c) in enumerate(conds)
        i, k = c.variable::Int, c.horizon
        for j in 1:k
            s = k - j + 1                     # Ψ_{k-j} lives at index k-j+1
            for q in 1:n_shocks
                R[row, (j-1)*n_shocks+q] = Psi[s, i, q]
            end
        end
        r[row] = c.value - uncond[k, i]
        om[row] = c.sd^2
    end
    return R, r, om
end

"""
    _cf_shock_distribution(R, r, om) -> (mu, sqrtV)

Waggoner-Zha conditional shock distribution `ε ~ N(μ, V)` with

    μ = R'(RR' + Ω)⁻¹ r,        V = I − R'(RR' + Ω)⁻¹ R.

With hard conditions (`Ω = 0`) `V` is the orthogonal projector onto `null(R)`. The
symmetric PSD square root is taken by eigendecomposition with negative eigenvalues
clipped, which handles the rank-deficient hard case and the full-rank soft case
identically. `robust_inv` guards the (possibly ill-conditioned) `RR' + Ω` inverse.
"""
function _cf_shock_distribution(R::Matrix{T}, r::Vector{T},
                                om::Vector{T}) where {T<:AbstractFloat}
    M = R * R'
    @inbounds for i in eachindex(om)
        M[i, i] += om[i]
    end
    Minv = robust_inv(Symmetric(M))
    mu = R' * (Minv * r)
    V = Symmetric(-R' * Minv * R + I)
    E = eigen(V)
    lam = max.(E.values, zero(T))
    sqrtV = E.vectors * Diagonal(sqrt.(lam)) * E.vectors'
    return mu, sqrtV
end

"""
    _cf_path(uncond, Psi, shocks, h) -> Matrix

Map a drawn shock path back through the structural MA representation:
`y_{T+s} = ŷ_{T+s} + Σ_{j=1}^{s} Ψ_{s-j} ε_{T+j}`.
"""
function _cf_path(uncond::AbstractMatrix{T}, Psi::Array{T,3}, shocks::AbstractMatrix{T},
                  h::Int) where {T<:AbstractFloat}
    n = size(uncond, 2)
    out = Matrix{T}(uncond)
    @inbounds for s in 1:h, j in 1:s
        idx = s - j + 1
        for i in 1:n
            acc = zero(T)
            for q in 1:size(shocks, 2)
                acc += Psi[idx, i, q] * shocks[j, q]
            end
            out[s, i] += acc
        end
    end
    return out
end

_cf_reshape_shocks(v::AbstractVector{T}, h::Int, n_shocks::Int) where {T} =
    Matrix{T}(reshape(v, n_shocks, h)')

function _cf_bands(sim::Array{T,3}, conf_level::Real) where {T<:AbstractFloat}
    n_draws, h, n = size(sim)
    alpha = (1 - T(conf_level)) / 2
    lo = Matrix{T}(undef, h, n)
    hi = Matrix{T}(undef, h, n)
    @inbounds for v in 1:n, s in 1:h
        col = @view sim[:, s, v]
        lo[s, v] = quantile(col, alpha)
        hi[s, v] = quantile(col, 1 - alpha)
    end
    return lo, hi
end

# =============================================================================
# Public API — VAR
# =============================================================================

"""
    conditional_forecast(model::VARModel, conditions, h; kwargs...) -> ConditionalForecast

Waggoner-Zha (1999) conditional forecast: project the VAR `h` periods ahead subject to
`conditions` on the paths of selected variables.

Write the forecast as the unconditional path plus the structural moving average of the
future shocks,

```math
y_{T+s} = \\hat y_{T+s} + \\sum_{j=1}^{s} \\Psi_{s-j}\\, \\varepsilon_{T+j},
\\qquad \\Psi_s = \\Phi_s P,
```

where ``P`` is the structural impact matrix. Each condition is linear in
``\\varepsilon``, so stacking them gives ``R\\varepsilon = r`` and the shocks are drawn
from the implied conditional distribution

```math
\\varepsilon \\sim N\\big(R'(RR')^{-1}r,\\; I - R'(RR')^{-1}R\\big),
```

the minimum-norm mean plus randomness in the null space of the restrictions. The point
forecast uses the conditional mean; the bands come from `reps` draws.

!!! note "Identification and conditional forecasts"
    For conditions on **observable paths**, the conditional forecast is *invariant* to the
    rotation ``Q``: replacing ``P = L Q`` by ``L`` leaves ``R'(RR')^{-1}r`` mapped through
    the moving average unchanged, because ``Q`` cancels between the restriction matrix and
    the impact matrix. Identification therefore does **not** change the forecast; it
    changes only the *interpretation* of the implied shocks in `result.shocks`, which
    rotate as ``\\varepsilon_L = Q\\,\\varepsilon_{LQ}``. Pass `Q` when the implied
    structural shocks are themselves of interest, or as the basis for restrictions on
    shocks rather than observables (structural scenario analysis, Antolín-Díaz, Petrella
    & Rubio-Ramírez 2021). The default is a Cholesky factorization in variable order.

# Arguments
- `model::VARModel{T}` — estimated VAR
- `conditions` — a `Vector{ForecastCondition}` (see [`forecast_condition`](@ref)) or a
  `Dict` keyed by `(variable, horizon)` with values `value` or `(value, sd)`
- `h::Int` — forecast horizon

# Keywords
- `Q::Union{Nothing,AbstractMatrix}=nothing` — rotation matrix; `nothing` = Cholesky. Only
  affects the reported `shocks` (the forecast itself is rotation-invariant, see above)
- `reps::Int=1000` — draws used for the bands
- `conf_level::Real=0.95` — band coverage
- `rng::AbstractRNG=Random.default_rng()` — random number generator

# Returns
[`ConditionalForecast`](@ref).

# References
- Waggoner, D. F. & Zha, T. (1999). Conditional Forecasts in Dynamic Multivariate Models.
  *Review of Economics and Statistics*, 81(4), 639-651.
"""
function conditional_forecast(model::VARModel{T}, conditions, h::Int;
                              Q::Union{Nothing,AbstractMatrix}=nothing,
                              reps::Int=1000,
                              conf_level::Real=0.95,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon must be positive"))
    reps < 1 && throw(ArgumentError("reps must be positive"))
    (0 < conf_level < 1) || throw(ArgumentError("conf_level must be in (0, 1)"))

    n, p = nvars(model), model.p
    varnames = model.varnames
    conds = _cf_normalize_conditions(conditions, varnames, h, T)

    Qm = Q === nothing ? Matrix{T}(I, n, n) : Matrix{T}(Q)
    size(Qm) == (n, n) || throw(ArgumentError(
        "Q must be $n×$n, got $(size(Qm))"))
    ident = Q === nothing ? :cholesky : :custom

    P = safe_cholesky(model.Sigma) * Qm
    A = extract_ar_coefficients(model.B, n, p)
    intercept = Vector{T}(@view model.B[1, :])
    history = Matrix{T}(model.Y[(end-p+1):end, :])

    uncond = _cf_uncond_path(intercept, A, history, h)
    Psi = _cf_structural_ma(A, P, h)
    R, r, om = _cf_build_R(conds, Psi, uncond, h, n)
    mu, sqrtV = _cf_shock_distribution(R, r, om)

    shock_mean = _cf_reshape_shocks(mu, h, n)
    point = _cf_path(uncond, Psi, shock_mean, h)

    sim = Array{T,3}(undef, reps, h, n)
    z = Vector{T}(undef, length(mu))
    @inbounds for rep in 1:reps
        randn!(rng, z)
        eps_draw = mu .+ sqrtV * z
        sim[rep, :, :] = _cf_path(uncond, Psi, _cf_reshape_shocks(eps_draw, h, n), h)
    end
    lo, hi = _cf_bands(sim, conf_level)

    return ConditionalForecast{T}(point, lo, hi, h, T(conf_level), varnames, conds,
                                  uncond, shock_mean, ident, reps)
end

# =============================================================================
# Public API — BVAR
# =============================================================================

"""
    conditional_forecast(post::BVARPosterior, conditions, h; kwargs...) -> ConditionalForecast

Waggoner-Zha conditional forecast integrating over the BVAR posterior. Each posterior
draw supplies its own coefficients, structural impact matrix, unconditional path and
restriction system, and contributes one conditional shock draw — so the bands reflect
both parameter and shock uncertainty, unlike the `VARModel` method which conditions on
the point estimate.

Non-stationary posterior draws are skipped (as in `forecast(::BVARPosterior, h)`).

# Keywords
- `Q::Union{Nothing,AbstractMatrix}=nothing` — rotation matrix; `nothing` = Cholesky. Only
  affects the reported `shocks` (the forecast itself is rotation-invariant, see above)
- `reps::Union{Nothing,Int}=nothing` — posterior draws used (default: all)
- `conf_level::Real=0.95` — credible-band coverage
- `point_estimate::Symbol=:mean` — `:mean` or `:median` across draws
- `rng::AbstractRNG=Random.default_rng()` — random number generator
"""
function conditional_forecast(post::BVARPosterior{T}, conditions, h::Int;
                              Q::Union{Nothing,AbstractMatrix}=nothing,
                              reps::Union{Nothing,Int}=nothing,
                              conf_level::Real=0.95,
                              point_estimate::Symbol=:mean,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    h < 1 && throw(ArgumentError("Forecast horizon must be positive"))
    (0 < conf_level < 1) || throw(ArgumentError("conf_level must be in (0, 1)"))
    point_estimate ∈ (:mean, :median) ||
        throw(ArgumentError("point_estimate must be :mean or :median"))

    n, p = post.n, post.p
    varnames = post.varnames
    conds = _cf_normalize_conditions(conditions, varnames, h, T)

    Qm = Q === nothing ? Matrix{T}(I, n, n) : Matrix{T}(Q)
    size(Qm) == (n, n) || throw(ArgumentError("Q must be $n×$n, got $(size(Qm))"))
    ident = Q === nothing ? :cholesky : :custom

    n_use = reps === nothing ? post.n_draws : min(reps, post.n_draws)
    history0 = Matrix{T}(post.data[(end-p+1):end, :])

    sim = Array{T,3}(undef, n_use, h, n)
    uncond_acc = zeros(T, h, n)
    shock_acc = zeros(T, h, n)
    valid = 0

    companion = zeros(T, n * p, n * p)
    if p > 1
        companion[n+1:end, 1:n*(p-1)] = Matrix{T}(I, n * (p - 1), n * (p - 1))
    end

    for s in 1:n_use
        B_s = post.B_draws[s, :, :]
        Sigma_s = post.Sigma_draws[s, :, :]
        A = extract_ar_coefficients(B_s, n, p)
        for lag in 1:p
            companion[1:n, (lag-1)*n+1:lag*n] = A[lag]
        end
        maximum(abs.(eigvals(companion))) >= one(T) && continue

        valid += 1
        intercept = Vector{T}(@view B_s[1, :])
        uncond = _cf_uncond_path(intercept, A, history0, h)
        Psi = _cf_structural_ma(A, safe_cholesky(Sigma_s) * Qm, h)
        R, r, om = _cf_build_R(conds, Psi, uncond, h, n)
        mu, sqrtV = _cf_shock_distribution(R, r, om)

        eps_draw = mu .+ sqrtV * randn(rng, T, length(mu))
        sim[valid, :, :] = _cf_path(uncond, Psi, _cf_reshape_shocks(eps_draw, h, n), h)
        uncond_acc .+= uncond
        shock_acc .+= _cf_reshape_shocks(mu, h, n)
    end

    valid == 0 && error("All posterior draws are non-stationary")
    valid < n_use ÷ 2 && @warn "$(n_use - valid)/$n_use posterior draws non-stationary, skipped"

    sim = sim[1:valid, :, :]
    point = Matrix{T}(undef, h, n)
    @inbounds for v in 1:n, s in 1:h
        col = @view sim[:, s, v]
        point[s, v] = point_estimate == :median ? median(col) : mean(col)
    end
    lo, hi = _cf_bands(sim, conf_level)

    return ConditionalForecast{T}(point, lo, hi, h, T(conf_level), varnames, conds,
                                  uncond_acc ./ valid, shock_acc ./ valid, ident, valid)
end
