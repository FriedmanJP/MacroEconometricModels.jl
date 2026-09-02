# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Arias, Rubio-Ramírez, Waggoner (2018) - Zero + Sign Restrictions
# =============================================================================

# Errors that legitimately indicate a *rejectable* random draw / failed optimization
# (numerical degeneracy), as opposed to a genuine bug (BoundsError, MethodError,
# DimensionMismatch, ...) which must propagate rather than be silently swallowed.
# Shared by the Arias and Uhlig identification loops.
_is_rejectable_draw_error(err) =
    err isa LinearAlgebra.SingularException ||
    err isa LinearAlgebra.PosDefException ||
    err isa LinearAlgebra.LAPACKException ||
    err isa DomainError

"""Abstract super-type of every SVAR restriction (linear zero or rejection)."""
abstract type AbstractSVARRestriction end

"""True when `r` is a linear equality on a column of `Q` (null-space row)."""
is_linear_zero(::AbstractSVARRestriction) = false

"""Zero restriction: variable does not respond to shock at a finite horizon."""
struct ZeroRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
    horizon::Int
end
is_linear_zero(::ZeroRestriction) = true

"""Long-run zero: ``e_v' C(1) L q_s = 0`` with ``C(1) = (I - \\sum A_i)^{-1}``."""
struct LongRunZeroRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
end
is_linear_zero(::LongRunZeroRestriction) = true

"""Zero on ``A_0[\\mathrm{equation},\\mathrm{shock}]`` in the RWZ form ``A_0 = L^{-T} Q`` (``y'A_0``)."""
struct A0ZeroRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
end
is_linear_zero(::A0ZeroRestriction) = true

"""Zero restriction on an entry of ``A_+ = B A_0`` (lag `0` is the intercept row)."""
struct AplusZeroRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
    lag::Int
end
is_linear_zero(::AplusZeroRestriction) = true

"""Sign restriction: variable response to shock has required sign at horizon."""
struct SignRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
    horizon::Int
    sign::Int  # +1 or -1
end

"""Sign on ``A_0[\\mathrm{equation},\\mathrm{shock}]`` in the RWZ form ``A_0 = L^{-T} Q`` (``y'A_0``)."""
struct A0SignRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
    sign::Int
end

"""Sign restriction on an entry of ``A_+``."""
struct AplusSignRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
    lag::Int
    sign::Int
end

"""Kilian–Murphy elasticity bound: ``\\mathrm{IRF}_{\\mathrm{num}} / \\mathrm{IRF}_{\\mathrm{den}} \\in [\\mathrm{lower}, \\mathrm{upper}]``."""
struct ElasticityBound <: AbstractSVARRestriction
    numerator_var::Int
    denominator_var::Int
    shock::Int
    horizon::Int
    lower::Float64
    upper::Float64
end

"""Magnitude bound on an IRF entry."""
struct MagnitudeBound <: AbstractSVARRestriction
    variable::Int
    shock::Int
    horizon::Int
    lower::Float64
    upper::Float64
end

"""FEVD-share restriction (Ben Zeev / Volpicella): shock's share of MSE in ``[\\mathrm{lower}, \\mathrm{upper}]``."""
struct FEVDShareRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
    horizon::Int
    lower::Float64
    upper::Float64
end

"""Cumulative-IRF sign restriction on ``\\sum_{h \\in \\mathrm{horizons}} \\Theta_h``."""
struct CumulativeRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
    horizons::UnitRange{Int}
    sign::Int
end

"""Narrative shock-sign restriction: structural `ε[date, shock]` has the given sign."""
struct NarrativeShockRestriction <: AbstractSVARRestriction
    shock::Int
    dates::Vector{Int}
    sign::Int
end

"""
Narrative contribution restriction.

ADRR Type A (`kind=:most_important`): ``|H_j| > \\max_{k \\neq j}|H_k|``.
Type B (`kind=:overwhelming`): ``|H_j| > \\sum_{k \\neq j}|H_k|``.
`kind=:least_important`: ``|H_j| < \\min_{k \\neq j}|H_k|``.
"""
struct NarrativeContributionRestriction <: AbstractSVARRestriction
    variable::Int
    shock::Int
    window::UnitRange{Int}
    kind::Symbol
end
NarrativeContributionRestriction(variable::Int, shock::Int, window::AbstractUnitRange{<:Integer}) =
    NarrativeContributionRestriction(variable, shock, UnitRange{Int}(window), :most_important)

const _NARRATIVE_CONTRIBUTION_KINDS = (:most_important, :overwhelming, :least_important)

function _check_contribution_kind(kind::Symbol)
    kind in _NARRATIVE_CONTRIBUTION_KINDS || throw(ArgumentError(
        "kind must be :most_important (Type A), :overwhelming (Type B), or :least_important"))
    nothing
end

"""
    SVARRestrictions(n_vars; zeros=[], signs=[])

Typed restriction container. `zeros` holds linear equalities (null-space rows of
`Q`); `signs` holds rejection restrictions. Nested vectors from
`sign_restriction(...; horizons=0:K)` are flattened. FEVD is computed only when
an [`FEVDShareRestriction`](@ref) is present.
"""
struct SVARRestrictions
    zeros::Vector{AbstractSVARRestriction}
    signs::Vector{AbstractSVARRestriction}
    n_vars::Int
    n_shocks::Int
    function SVARRestrictions(zeros, signs, n_vars::Int, n_shocks::Int)
        all_r = AbstractSVARRestriction[]
        append!(all_r, _flatten_restrictions(zeros))
        append!(all_r, _flatten_restrictions(signs))
        z = AbstractSVARRestriction[r for r in all_r if is_linear_zero(r)]
        s = AbstractSVARRestriction[r for r in all_r if !is_linear_zero(r)]
        new(z, s, n_vars, n_shocks)
    end
end

function _flatten_restrictions(rs)
    out = AbstractSVARRestriction[]
    if rs isa AbstractSVARRestriction
        push!(out, rs)
        return out
    end
    for item in rs
        if item isa AbstractVector
            append!(out, _flatten_restrictions(item))
        else
            push!(out, item)
        end
    end
    out
end

function SVARRestrictions(n_vars::Int; zeros=ZeroRestriction[], signs=SignRestriction[])
    n_vars > 0 || throw(ArgumentError("n_vars must be positive"))
    z = _flatten_restrictions(zeros)
    s = _flatten_restrictions(signs)
    for r in Iterators.flatten((z, s))
        _validate_restriction(r, n_vars)
    end
    SVARRestrictions(z, s, n_vars, n_vars)
end

_parse_sign(sign::Symbol) =
    sign === :positive ? 1 : sign === :negative ? -1 :
        throw(ArgumentError("sign must be :positive or :negative"))

function _check_index(i::Int, n::Int, label::AbstractString)
    (1 <= i <= n) || throw(ArgumentError("$label $i out of 1:$n"))
end

function _check_finite_bounds(lower, upper)
    (isfinite(lower) && isfinite(upper)) ||
        throw(ArgumentError("bounds must be finite"))
    lower < upper || throw(ArgumentError("lower bound must be < upper bound"))
    nothing
end

function _check_horizon_range(horizons::AbstractUnitRange{<:Integer})
    first(horizons) >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    nothing
end

function _validate_restriction(r::ZeroRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
    r.horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
end
function _validate_restriction(r::LongRunZeroRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
end
function _validate_restriction(r::A0ZeroRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
end
function _validate_restriction(r::AplusZeroRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
    _check_aplus_lag_equation(r.lag, r.variable)
end
function _validate_restriction(r::SignRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
    r.horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    r.sign in (-1, 1) || throw(ArgumentError("sign must be -1 or +1"))
end
function _validate_restriction(r::A0SignRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
    r.sign in (-1, 1) || throw(ArgumentError("sign must be -1 or +1"))
end
function _validate_restriction(r::AplusSignRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
    _check_aplus_lag_equation(r.lag, r.variable)
    r.sign in (-1, 1) || throw(ArgumentError("sign must be -1 or +1"))
end
function _validate_restriction(r::ElasticityBound, n_vars::Int)
    _check_index(r.numerator_var, n_vars, "numerator variable")
    _check_index(r.denominator_var, n_vars, "denominator variable")
    _check_index(r.shock, n_vars, "shock")
    r.horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    _check_finite_bounds(r.lower, r.upper)
end
function _validate_restriction(r::MagnitudeBound, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
    r.horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    _check_finite_bounds(r.lower, r.upper)
end
function _validate_restriction(r::FEVDShareRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
    r.horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    _check_finite_bounds(r.lower, r.upper)
    (0 <= r.lower && r.upper <= 1) ||
        throw(ArgumentError("FEVD share bounds must lie in [0, 1]"))
end
function _validate_restriction(r::CumulativeRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
    _check_horizon_range(r.horizons)
    r.sign in (-1, 1) || throw(ArgumentError("sign must be -1 or +1"))
end
function _validate_restriction(r::NarrativeShockRestriction, n_vars::Int)
    _check_index(r.shock, n_vars, "shock")
    r.sign in (-1, 1) || throw(ArgumentError("sign must be -1 or +1"))
    isempty(r.dates) && throw(ArgumentError("narrative dates must be nonempty"))
    all(d -> d >= 1, r.dates) || throw(ArgumentError("narrative dates must be ≥ 1"))
end
function _validate_restriction(r::NarrativeContributionRestriction, n_vars::Int)
    _check_index(r.variable, n_vars, "variable")
    _check_index(r.shock, n_vars, "shock")
    isempty(r.window) && throw(ArgumentError("narrative contribution window must be nonempty"))
    first(r.window) >= 1 || throw(ArgumentError("narrative dates must be ≥ 1"))
    _check_contribution_kind(r.kind)
end
_validate_restriction(::AbstractSVARRestriction, ::Int) = nothing

_restriction_horizon(r::ZeroRestriction) = r.horizon
_restriction_horizon(::LongRunZeroRestriction) = 0
_restriction_horizon(::A0ZeroRestriction) = 0
_restriction_horizon(::AplusZeroRestriction) = 0
_restriction_horizon(r::SignRestriction) = r.horizon
_restriction_horizon(r::ElasticityBound) = r.horizon
_restriction_horizon(r::MagnitudeBound) = r.horizon
_restriction_horizon(r::FEVDShareRestriction) = r.horizon
_restriction_horizon(r::CumulativeRestriction) = last(r.horizons)
_restriction_horizon(r::NarrativeContributionRestriction) = max(0, last(r.window) - first(r.window))
_restriction_horizon(::AbstractSVARRestriction) = 0

function _check_aplus_lag_equation(lag::Int, equation::Int)
    lag >= 0 || throw(ArgumentError("A+ lag must be ≥ 0"))
    if lag == 0 && equation != 1
        throw(ArgumentError(
            "A+ intercept (lag=0) is the first row of B; equation must be 1, got $equation"))
    end
    nothing
end

_aplus_row(lag::Int, variable::Int, n::Int) = lag <= 0 ? 1 : 1 + (lag - 1) * n + variable
_aplus_row(r::AplusZeroRestriction, n::Int) = _aplus_row(r.lag, r.variable, n)
_aplus_row(r::AplusSignRestriction, n::Int) = _aplus_row(r.lag, r.variable, n)

function _C1_from_B(B::AbstractMatrix{T}, n::Int, p::Int) where {T}
    A_sum = sum(extract_ar_coefficients(B, n, p))
    robust_inv(Matrix{T}(I(n) - A_sum); silent=true)
end

"""
Result from Arias et al. (2018) identification.

`ess` is Kish's effective sample size of the importance weights and
`ess_fraction` is `ess / length(weights)`. Under pure sign restrictions the
weights are uniform and `ess_fraction == 1`; with zero or narrative restrictions
a fraction far below 1 means a handful of draws carry most of the posterior mass
and the weighted summaries rest on far fewer effective draws than `n_draws`
suggests. `n_narrative_sims` is the Monte Carlo size used for ADRR ``ω̂``
(0 when no narrative restriction is present). `elapsed` is wall-clock seconds of
the accept/weight loop; `weights_elapsed` is the summed time in the volume
element (zero when `compute_weights=false` or there are no zeros).
"""
struct AriasSVARResult{T<:AbstractFloat}
    Q_draws::Vector{Matrix{T}}
    irf_draws::Array{T,4}
    weights::Vector{T}
    acceptance_rate::T
    restrictions::SVARRestrictions
    ess::T
    ess_fraction::T
    varnames::Vector{String}
    n_degenerate_weights::Int
    n_narrative_sims::Int
    elapsed::T
    weights_elapsed::T
end

# Back-compatible arities: pre-ESS / pre-varnames / pre-degenerate construction sites.
function AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate,
                            restrictions) where {T<:AbstractFloat}
    ess = T(_effective_sample_size(weights))
    n = length(weights)
    nv = restrictions.n_vars
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions,
                       ess, n > 0 ? ess / T(n) : zero(T),
                       ["var$i" for i in 1:nv], 0)
end

function AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate,
                            restrictions, ess, ess_fraction) where {T<:AbstractFloat}
    nv = restrictions.n_vars
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions,
                       ess, ess_fraction, ["var$i" for i in 1:nv], 0)
end

function AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate,
                            restrictions, ess, ess_fraction, varnames) where {T<:AbstractFloat}
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions,
                       ess, ess_fraction, varnames, 0, 0)
end

function AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate,
                            restrictions, ess, ess_fraction, varnames,
                            n_degenerate_weights::Int) where {T<:AbstractFloat}
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions,
                       ess, ess_fraction, varnames, n_degenerate_weights, 0)
end

function AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate,
                            restrictions, ess, ess_fraction, varnames,
                            n_degenerate_weights::Int, n_narrative_sims::Int) where {T<:AbstractFloat}
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions,
                       ess, ess_fraction, varnames, n_degenerate_weights, n_narrative_sims,
                       zero(T), zero(T))
end

"""
Bayesian set-identified SVAR (Arias et al. 2018 applied to posterior draws).

`total_accepted` is `size(irf_draws, 1)`. `n_degenerate_weights` counts draws
skipped because the importance log-weight was non-finite. `n_narrative_sims`
is the ADRR ``ω̂`` Monte Carlo size (0 when narrative restrictions are absent).
"""
struct BayesianSetIdentifiedSVAR{T<:AbstractFloat}
    Q_draws::Vector{Matrix{T}}
    irf_draws::Array{T,4}
    weights::Vector{T}
    ess::T
    restrictions::SVARRestrictions
    varnames::Vector{String}
    n_unidentified::Int
    n_degenerate_weights::Int
    irf_quantiles::Array{T,4}
    irf_mean::Array{T,3}
    acceptance_rates::Vector{T}
    ess_fraction::T
    n_narrative_sims::Int
end

function BayesianSetIdentifiedSVAR{T}(Q_draws, irf_draws, weights, ess, restrictions,
        varnames, n_unidentified, n_degenerate_weights, irf_quantiles, irf_mean,
        acceptance_rates, ess_fraction) where {T<:AbstractFloat}
    BayesianSetIdentifiedSVAR{T}(Q_draws, irf_draws, weights, ess, restrictions,
        varnames, n_unidentified, n_degenerate_weights, irf_quantiles, irf_mean,
        acceptance_rates, ess_fraction, 0)
end

function Base.getproperty(r::BayesianSetIdentifiedSVAR, s::Symbol)
    s === :total_accepted && return size(getfield(r, :irf_draws), 1)
    getfield(r, s)
end

function Base.propertynames(::BayesianSetIdentifiedSVAR, private::Bool=false)
    (fieldnames(BayesianSetIdentifiedSVAR)..., :total_accepted)
end

Base.haskey(::BayesianSetIdentifiedSVAR, s::Symbol) =
    s === :total_accepted || hasfield(BayesianSetIdentifiedSVAR, s)
Base.getindex(r::BayesianSetIdentifiedSVAR, s::Symbol) = getproperty(r, s)

"""
Fraction of the nominal draw count below which importance weights are reported
as degenerate. Matches the conventional SMC resampling trigger.
"""
const _ARIAS_ESS_WARN_FRACTION = 0.1

"""
    _warn_low_ess(ess, ess_fraction, n, label)

Emit the degeneracy warning when the effective sample size has collapsed. Silent
when the weights are uniform (`ess_fraction == 1`), which is the sign-only case.
"""
function _warn_low_ess(ess::Real, ess_fraction::Real, n::Integer, label::AbstractString)
    (n > 0 && ess_fraction < _ARIAS_ESS_WARN_FRACTION) || return nothing
    @warn "$label: importance weights are degenerate — effective sample size " *
          "$(round(ess; digits=1)) of $n draws " *
          "($(round(100 * ess_fraction; digits=1))% < " *
          "$(round(Int, 100 * _ARIAS_ESS_WARN_FRACTION))%). A few draws carry almost all " *
          "the posterior mass, so weighted IRF summaries are far less precise than the " *
          "draw count suggests. Consider loosening the zero/sign/narrative restrictions or raising n_draws."
    nothing
end

# --- MA Coefficients (wrappers around the shared kernels in irf.jl) ---

"""Compute MA coefficients Φ_0, ..., Φ_horizon (length horizon+1)."""
function _compute_ma_coefficients(model::VARModel{T}, horizon::Int) where {T<:AbstractFloat}
    ma_coefficients(model.B, nvars(model), model.p, horizon + 1)
end

"""Compute structural IRF for rotation Q."""
function _compute_irf_for_Q(model::VARModel{T}, Q::Matrix{T}, Phi::Vector{Matrix{T}},
                            L::LowerTriangular{T,Matrix{T}}, horizon::Int) where {T<:AbstractFloat}
    structural_irf(Phi, L, Q, horizon)
end

# --- Restriction Checking ---

"""
    check(r, irf, A0, Aplus, fevd, shocks=nothing, hd=nothing) -> Bool

Rejection test for restriction `r`. Linear zeros are enforced as null-space rows
on `Q` rather than here. Narrative types evaluate on structural shocks ``ε``
and historical-decomposition contributions over the restricted dates.
"""
check(::AbstractSVARRestriction, irf, A0, Aplus, fevd) = true
check(r::AbstractSVARRestriction, irf, A0, Aplus, fevd, shocks, hd=nothing) =
    check(r, irf, A0, Aplus, fevd)

function check(r::SignRestriction, irf, A0, Aplus, fevd)
    val = irf[r.horizon + 1, r.variable, r.shock]
    r.sign > 0 ? val > 0 : val < 0
end

function check(r::A0SignRestriction, irf, A0, Aplus, fevd)
    A0 === nothing && return false
    val = A0[r.variable, r.shock]
    r.sign > 0 ? val > 0 : val < 0
end

function check(r::AplusSignRestriction, irf, A0, Aplus, fevd)
    Aplus === nothing && return false
    n = size(Aplus, 2)
    val = Aplus[_aplus_row(r, n), r.shock]
    r.sign > 0 ? val > 0 : val < 0
end

function check(r::ElasticityBound, irf, A0, Aplus, fevd)
    T = eltype(irf)
    num = irf[r.horizon + 1, r.numerator_var, r.shock]
    den = irf[r.horizon + 1, r.denominator_var, r.shock]
    abs(den) <= eps(T) && return false
    e = num / den
    T(r.lower) <= e <= T(r.upper)
end

function check(r::MagnitudeBound, irf, A0, Aplus, fevd)
    T = eltype(irf)
    val = irf[r.horizon + 1, r.variable, r.shock]
    T(r.lower) <= val <= T(r.upper)
end

function check(r::FEVDShareRestriction, irf, A0, Aplus, fevd)
    fevd === nothing && return false
    T = eltype(irf)
    share = fevd[r.variable, r.shock, r.horizon + 1]
    T(r.lower) <= share <= T(r.upper)
end

function check(r::CumulativeRestriction, irf, A0, Aplus, fevd)
    s = zero(eltype(irf))
    @inbounds for h in r.horizons
        s += irf[h + 1, r.variable, r.shock]
    end
    r.sign > 0 ? s > 0 : s < 0
end

check(::NarrativeShockRestriction, irf, A0, Aplus, fevd) = false
check(::NarrativeContributionRestriction, irf, A0, Aplus, fevd) = false

function check(r::NarrativeShockRestriction, irf, A0, Aplus, fevd, shocks, hd=nothing)
    shocks === nothing && return false
    T_eff, n = size(shocks)
    (1 <= r.shock <= n) || return false
    @inbounds for d in r.dates
        (1 <= d <= T_eff) || return false
        val = shocks[d, r.shock]
        r.sign > 0 ? (val > 0 || return false) : (val < 0 || return false)
    end
    true
end

function check(r::NarrativeContributionRestriction, irf, A0, Aplus, fevd, shocks, hd=nothing)
    H = if hd isa AbstractVector
        hd
    elseif irf !== nothing && shocks !== nothing
        _narrative_window_contributions(irf, shocks, r.variable, r.window)
    else
        return false
    end
    H === nothing && return false
    (1 <= r.shock <= length(H)) || return false
    _is_leading_contributor(H, r.shock, r.kind)
end

_is_narrative(::NarrativeShockRestriction) = true
_is_narrative(::NarrativeContributionRestriction) = true
_is_narrative(::AbstractSVARRestriction) = false
_has_narrative(r::SVARRestrictions) = any(_is_narrative, r.signs)

"""ADRR contribution of each shock to variable `i` over `window` (eq. 6–8)."""
function _narrative_window_contributions(irf::AbstractArray{T,3}, shocks::AbstractMatrix,
                                         variable::Int, window::UnitRange{Int}) where {T}
    t0, t1 = first(window), last(window)
    h = t1 - t0
    n = size(irf, 3)
    (1 <= t0 && t1 <= size(shocks, 1) && 0 <= h < size(irf, 1) &&
     1 <= variable <= size(irf, 2)) || return nothing
    H = zeros(T, n)
    @inbounds for j in 1:n
        s = zero(T)
        for ℓ in 0:h
            s += irf[ℓ + 1, variable, j] * shocks[t1 - ℓ, j]
        end
        H[j] = s
    end
    H
end

function _is_leading_contributor(H::AbstractVector{T}, shock::Int, kind::Symbol) where {T}
    abs_j = abs(H[shock])
    if kind === :overwhelming
        others = zero(T)
        @inbounds for k in eachindex(H)
            k == shock && continue
            others += abs(H[k])
        end
        return abs_j > others
    elseif kind === :least_important
        @inbounds for k in eachindex(H)
            k == shock && continue
            abs(H[k]) <= abs_j && return false
        end
        return true
    end
    @inbounds for k in eachindex(H)
        k == shock && continue
        abs(H[k]) >= abs_j && return false
    end
    true
end

function _narrative_dates(restrictions::SVARRestrictions)
    dates = Int[]
    for r in restrictions.signs
        if r isa NarrativeShockRestriction
            append!(dates, r.dates)
        elseif r isa NarrativeContributionRestriction
            append!(dates, collect(r.window))
        end
    end
    unique!(sort!(dates))
end

function _narrative_restrictions_hold(restrictions::SVARRestrictions, irf, shocks)
    all(restrictions.signs) do r
        _is_narrative(r) || return true
        check(r, irf, nothing, nothing, nothing, shocks, nothing)
    end
end

"""Monte Carlo estimate of ADRR ``ω̂``: fraction of ``ε* ~ N(0,I)`` paths that satisfy the narrative restrictions."""
function _omega_hat(restrictions::SVARRestrictions, irf::AbstractArray{T,3}, n::Int, ::Type{T};
                    n_sims::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    dates = _narrative_dates(restrictions)
    isempty(dates) && return one(T)
    tmax = last(dates)
    ε = zeros(T, tmax, n)
    n_ok = 0
    @inbounds for _ in 1:n_sims
        for t in dates, j in 1:n
            ε[t, j] = randn(rng, T)
        end
        _narrative_restrictions_hold(restrictions, irf, ε) && (n_ok += 1)
    end
    T(n_ok) / T(n_sims)
end

function _validate_narrative_sample(restrictions::SVARRestrictions, T_eff::Int)
    T_eff >= 1 || throw(ArgumentError("narrative restrictions require VAR residuals"))
    for r in restrictions.signs
        if r isa NarrativeShockRestriction
            for d in r.dates
                (1 <= d <= T_eff) || throw(ArgumentError(
                    "narrative date $d out of residual sample 1:$T_eff"))
            end
        elseif r isa NarrativeContributionRestriction
            w = r.window
            (1 <= first(w) && last(w) <= T_eff) || throw(ArgumentError(
                "narrative window $w out of residual sample 1:$T_eff"))
        end
    end
    nothing
end

"""Check finite-horizon IRF zeros (long-run / A0 / A+ zeros are enforced in the null space)."""
_check_zero_restrictions(irf::Array{T,3}, r::SVARRestrictions; tol::T=T(1e-10)) where {T} =
    all(r.zeros) do zr
        zr isa ZeroRestriction || return true
        abs(irf[zr.horizon + 1, zr.variable, zr.shock]) <= tol
    end

"""Check IRF sign restrictions only (pure-sign containers)."""
_check_sign_restrictions(irf::Array{T,3}, r::SVARRestrictions) where {T} =
    all(r.signs) do sr
        sr isa SignRestriction || return true
        sr.sign > 0 ? irf[sr.horizon + 1, sr.variable, sr.shock] > 0 :
                      irf[sr.horizon + 1, sr.variable, sr.shock] < 0
    end

_check_rejections(r::SVARRestrictions, irf, A0, Aplus, fevd) =
    all(r.signs) do sr
        _is_narrative(sr) && return true
        check(sr, irf, A0, Aplus, fevd)
    end

"""
    sign_check(r::SVARRestrictions) -> Function

Return `irf -> Bool` for `identify_sign`, SDFM, and counterfactuals. Computes FEVD
only when an FEVD-share restriction is present.

Throws `ArgumentError` if `r` contains [`A0SignRestriction`](@ref) or
[`AplusSignRestriction`](@ref): those `check` methods need the structural
matrices `A0`/`A₊`, which this IRF-only closure does not have. Use
[`identify_arias`](@ref) instead.
"""
function sign_check(r::SVARRestrictions)
    if any(s -> s isa Union{A0SignRestriction, AplusSignRestriction}, r.signs)
        throw(ArgumentError(
            "sign_check cannot evaluate A0/A+ sign restrictions (needs A0/A₊); " *
            "use identify_arias"))
    end
    if any(_is_narrative, r.signs)
        throw(ArgumentError(
            "sign_check cannot evaluate narrative restrictions (needs shocks); " *
            "use identify_arias"))
    end
    has_fevd = any(s -> s isa FEVDShareRestriction, r.signs)
    function (irf)
        fevd = nothing
        if has_fevd
            fevd = _compute_fevd(irf, size(irf, 2), size(irf, 1))[2]
        end
        _check_rejections(r, irf, nothing, nothing, fevd)
    end
end

# --- Zero Restriction Algorithm ---

"""Build constraint matrix for zero restrictions on shock j."""
function _build_zero_constraint_matrix(r::SVARRestrictions, shock::Int, Phi::Vector{Matrix{T}},
                                       L::LowerTriangular{T,Matrix{T}};
                                       B=nothing, C1=nothing) where {T<:AbstractFloat}
    ZF = _compute_ZF(r, Phi, L, shock; B=B, C1=C1)
    [Vector{T}(ZF[i, :]) for i in axes(ZF, 1)]
end

"""Draw unit vector from null space of constraints."""
function _draw_null_space_vector(constraints::Vector{Vector{T}}, n::Int; rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    isempty(constraints) && return (x = randn(rng, T, n); x / norm(x))

    F = reduce(vcat, [c' for c in constraints])
    svd_result = svd(F, full=true)
    V = transpose(svd_result.Vt)
    tol = max(size(F)...) * eps(T) * (isempty(svd_result.S) ? one(T) : maximum(svd_result.S))
    rank_F = sum(svd_result.S .> tol)
    null_dim = n - rank_F
    null_dim <= 0 && throw(IdentificationError("Zero restrictions over-constrain shock"))

    N = V[:, (rank_F + 1):n]
    z = randn(rng, T, null_dim)
    q = N * z
    q / norm(q)
end

"""Draw orthogonal Q satisfying zero restrictions (Algorithm 2, Arias et al. 2018)."""
function _draw_Q_with_zero_restrictions(r::SVARRestrictions, Phi::Vector{Matrix{T}},
                                         L::LowerTriangular{T,Matrix{T}};
                                         rng::AbstractRNG=Random.default_rng(),
                                         B=nothing, C1=nothing) where {T<:AbstractFloat}
    n = r.n_vars
    Q = zeros(T, n, n)
    for j in 1:n
        zero_constraints = _build_zero_constraint_matrix(r, j, Phi, L; B=B, C1=C1)
        ortho_constraints = [Vector{T}(Q[:, k]) for k in 1:j-1]
        Q[:, j] = _draw_null_space_vector(vcat(zero_constraints, ortho_constraints), n; rng=rng)
    end
    @assert norm(Q' * Q - I) < 1e-10 "Q not orthogonal"
    Q
end

# --- Arias et al. (2018) Setup & Volume Element Helpers ---

"""Precomputed auxiliary data for Arias et al. (2018) importance weight computation."""
struct _AriasSVARSetup{T<:AbstractFloat}
    W::Vector{Matrix{T}}         # Auxiliary matrices W_j (s_j × n), j=1..n
    zeros_per_shock::Vector{Int}  # z_j per shock
    sphere_dims::Vector{Int}      # s_j = n - (j-1) - z_j per shock
    dim::Int                      # total sphere dimension = Σ s_j
end

function _AriasSVARSetup(restrictions::SVARRestrictions, n::Int, ::Type{T};
                        rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    zeros_per_shock = zeros(Int, n)
    for zr in restrictions.zeros
        zeros_per_shock[zr.shock] += 1
    end
    sphere_dims = [n - (j - 1) - zeros_per_shock[j] for j in 1:n]
    W = [randn(rng, T, s, n) for s in sphere_dims]
    _AriasSVARSetup{T}(W, zeros_per_shock, sphere_dims, sum(sphere_dims))
end

"""Safe log|det(A)| — returns -Inf for singular matrices."""
function _log_abs_det(A::AbstractMatrix{T}) where {T}
    lad = logabsdet(A)
    isfinite(lad[1]) ? lad[1] : T(-Inf)
end

"""Central finite-difference Jacobian of f at x."""
function _numerical_jacobian(f, x::AbstractVector{T}; fd_eps::T=T(1e-7)) where {T}
    n = length(x)
    f0 = f(x)
    m = length(f0)
    J = zeros(T, m, n)
    xp = copy(x)
    xm = copy(x)
    for i in 1:n
        xp[i] = x[i] + fd_eps
        xm[i] = x[i] - fd_eps
        fp = f(xp)
        fm = f(xm)
        J[:, i] = (fp - fm) / (2 * fd_eps)
        xp[i] = x[i]
        xm[i] = x[i]
    end
    J
end

"""
Log volume element of f restricted to manifold {h(x)=0}.
Computes 0.5 * log|det(N'*N)| where N = Df * nullspace(Dh).
"""
function _volume_from_jacobians(Df::AbstractMatrix, Dh::AbstractMatrix, ::Type{T}) where {T}
    Ns = nullspace(Dh)
    size(Ns, 2) == 0 && return T(-Inf)
    Nproj = Df * Ns
    G = Nproj' * Nproj
    T(0.5) * _log_abs_det(G)
end

function _log_volume_element(f, x::AbstractVector{T}, h) where {T}
    Df = ForwardDiff.jacobian(f, x)
    Dh = ForwardDiff.jacobian(h, x)
    _volume_from_jacobians(Df, Dh, T)
end

"""Central-difference volume element (FD pin / SID-27 comparison)."""
function _log_volume_element_fd(f, x::AbstractVector{T}, h) where {T}
    Df = _numerical_jacobian(f, x)
    Dh = _numerical_jacobian(h, x)
    _volume_from_jacobians(Df, Dh, T)
end

"""Pack (A0, Aplus) into a single vector."""
_pack_structural(A0::Matrix{T}, Aplus::Matrix{T}) where {T} = vcat(vec(A0), vec(Aplus))

"""Unpack vector into (A0 n×n, Aplus m×n)."""
function _unpack_structural(x::AbstractVector{T}, n::Int, m::Int) where {T}
    A0 = reshape(x[1:n*n], n, n)
    Aplus = reshape(x[n*n+1:end], m, n)
    (A0, Aplus)
end

# --- Structural ↔ Reduced-Form Mappings ---

"""Structural → reduced-form: (A0, Aplus) → (B, Σ)."""
function _struct_to_rf(A0::AbstractMatrix, Aplus::AbstractMatrix)
    A0m = A0 isa Matrix ? A0 : Matrix(A0)
    A0_inv = robust_inv(A0m)
    B = Aplus * A0_inv
    Sigma = A0_inv * A0_inv'
    (B, Sigma)
end

"""Reduced-form → structural: (B, L, Q) → (A0, Aplus).

RWZ row convention ``y_t' A_0``: ``A_0 = L^{-T} Q``. The column-convention
impact matrix is ``A_0^{-1} = L Q``, the transpose-inverse of this `A0`.
"""
function _rf_to_struct(B::Matrix{T}, L::LowerTriangular{T,Matrix{T}}, Q::Matrix{T}) where {T}
    A0 = Matrix{T}(L') \ Q   # A0 = inv(L') * Q  (y'A0, not A0^{-1}=LQ)
    Aplus = B * A0
    (A0, Aplus)
end

# --- Sphere Coordinate Mappings ---

"""Compute zero-restriction constraint rows for shock j.

Finite-horizon zeros use ``e_v' Φ_h L``. Long-run zeros use ``e_v' C(1) L``
with ``C(1) = (I - \\sum A_i)^{-1}``. ``A_0`` zeros use ``e_v' L^{-T}``
(``A_0 = L^{-T} Q``). ``A_+`` zeros use the corresponding row of ``B L^{-T}``.
"""
function _compute_ZF(restrictions::SVARRestrictions, Phi::Vector{<:AbstractMatrix},
                     L::LowerTriangular, shock_j::Int;
                     B::Union{Nothing,AbstractMatrix}=nothing,
                     C1::Union{Nothing,AbstractMatrix}=nothing)
    T = eltype(L)
    n = size(L, 1)
    rows = Vector{Vector{T}}()
    C1_local = C1
    LinvT = nothing
    for zr in restrictions.zeros
        zr.shock == shock_j || continue
        if zr isa ZeroRestriction
            push!(rows, Vector{T}((Phi[zr.horizon + 1] * L)[zr.variable, :]))
        elseif zr isa LongRunZeroRestriction
            if C1_local === nothing
                B === nothing && error("long-run zero restrictions require the reduced-form B")
                p = (size(B, 1) - 1) ÷ n
                C1_local = _C1_from_B(B, n, p)
            end
            push!(rows, Vector{T}((C1_local * L)[zr.variable, :]))
        elseif zr isa A0ZeroRestriction
            e = zeros(T, n)
            e[zr.variable] = one(T)
            push!(rows, Vector{T}(L \ e))
        elseif zr isa AplusZeroRestriction
            B === nothing && error("A+ zero restrictions require the reduced-form B")
            if LinvT === nothing
                LinvT = Matrix{T}(L' \ I(n))
            end
            row_idx = _aplus_row(zr, n)
            (1 <= row_idx <= size(B, 1)) || throw(ArgumentError(
                "A+ restriction lag=$(zr.lag) equation=$(zr.variable) is outside B ($(size(B, 1)) rows)"))
            push!(rows, Vector{T}((B * LinvT)[row_idx, :]))
        end
    end
    isempty(rows) ? zeros(T, 0, n) : reduce(vcat, [rw' for rw in rows])
end

# =============================================================================
# RWZ rank / order condition (Rubio-Ramírez, Waggoner & Zha 2010, Theorem 1)
# =============================================================================

"""
    IdentificationStatus

Rubio-Ramírez, Waggoner & Zha (2010) rank/order diagnostic for linear zero
restrictions. `status` is `:exact`, `:over`, `:under`, or `:set`.

# Fields
- `status::Symbol`: classification of the restriction pattern
- `ranks::Vector{Int}`: ``\\mathrm{rank}(M_j)`` for shock ``j = 1:n``, where
  ``M_j`` stacks the linear zero-restriction rows of shock ``j``
- `orders::Vector{Int}`: number of linear zeros on shock ``j``
- `n_overidentifying::Int`: ``\\sum_j \\max(\\mathrm{rank}_j - (n-j), 0)``
"""
struct IdentificationStatus
    status::Symbol
    ranks::Vector{Int}
    orders::Vector{Int}
    n_overidentifying::Int
    function IdentificationStatus(status::Symbol, ranks::Vector{Int},
                                  orders::Vector{Int}, n_overidentifying::Int)
        status in (:exact, :over, :under, :set) || throw(ArgumentError(
            "status must be :exact, :over, :under, or :set, got :$status"))
        length(ranks) == length(orders) ||
            throw(ArgumentError("ranks and orders must have the same length"))
        n_overidentifying >= 0 || throw(ArgumentError("n_overidentifying must be ≥ 0"))
        new(status, ranks, orders, n_overidentifying)
    end
end

function _zeros_per_shock(restrictions::SVARRestrictions, n::Int)
    orders = zeros(Int, n)
    for zr in restrictions.zeros
        (1 <= zr.shock <= n) && (orders[zr.shock] += 1)
    end
    orders
end

function _matrix_rank(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    (size(A, 1) == 0 || size(A, 2) == 0) && return 0
    svals = svdvals(A)
    isempty(svals) && return 0
    tol = max(size(A)...) * Base.eps(T) * maximum(svals)
    count(>(tol), svals)
end

function _rwz_phi_length(restrictions::SVARRestrictions)
    h = 0
    for zr in restrictions.zeros
        h = max(h, _restriction_horizon(zr))
    end
    h + 1
end

function _rwz_classify(ranks::Vector{Int}, orders::Vector{Int}, n::Int, has_signs::Bool)
    short = false
    extra = false
    n_over = 0
    @inbounds for j in 1:n
        need = n - j
        ranks[j] < need && (short = true)
        ranks[j] > need && (extra = true)
        n_over += max(ranks[j] - need, 0)
    end
    status = if !short && !extra
        :exact
    elseif !short && extra
        :over
    else
        drawable = true
        @inbounds for j in 1:n
            if orders[j] > n - j
                drawable = false
                break
            end
        end
        (has_signs && drawable) ? :set : :under
    end
    (status, n_over)
end

function _draw_generic_structural(n::Int, m::Int, ::Type{T};
                                  rng::AbstractRNG) where {T<:AbstractFloat}
    A0 = randn(rng, T, n, n)
    @inbounds for i in 1:n
        A0[i, i] += T(2n)
    end
    Aplus = randn(rng, T, m, n)
    (A0, Aplus)
end

function _accumulate_zf_ranks!(ranks::Vector{Int}, restrictions::SVARRestrictions,
                               B::AbstractMatrix{T}, Sigma::AbstractMatrix{T},
                               n::Int, p::Int, phi_len::Int) where {T<:AbstractFloat}
    L = safe_cholesky(Matrix{T}(Sigma))
    Phi = ma_coefficients(Matrix{T}(B), n, p, phi_len)
    C1 = any(z -> z isa LongRunZeroRestriction, restrictions.zeros) ?
         _C1_from_B(B, n, p) : nothing
    for j in 1:n
        ZF = _compute_ZF(restrictions, Phi, L, j; B=B, C1=C1)
        ranks[j] = max(ranks[j], _matrix_rank(ZF))
    end
    nothing
end

"""
    check_identification(restrictions::SVARRestrictions, model::VARModel;
                         n_points=10, rng) -> IdentificationStatus
    check_identification(restrictions::SVARRestrictions, n::Int) -> IdentificationStatus

Rubio-Ramírez, Waggoner & Zha (2010) Theorem 1: with zeros ordered so shock
``j`` has ``q_j`` linear restrictions, the SVAR is globally identified almost
everywhere if and only if ``\\mathrm{rank}(M_j(f(A_0, A_+))) = n - j`` for
``j = 1,\\ldots,n`` at a generic point. ``M_j`` is the stack of SID-14
zero-restriction rows for shock ``j``.

The `model` method evaluates those ranks at the estimated reduced form and at
`n_points` random ``(A_0, A_+)``, taking the maximum (generic rank). The `n`
method checks the order condition only, treating the zeros as independent.

Sign (rejection) restrictions do not enter the rank: a shortfall of zeros with
a drawable null space and at least one sign is classified `:set`. `:under`
means the equalities cannot identify the rotation. `:over` means some shock
has more independent zeros than the null space admits; `identify_arias` and
`identify_uhlig` throw [`IdentificationError`](@ref) for both `:under` and
`:over`.

# Returns
[`IdentificationStatus`](@ref) with `status ∈ (:exact, :over, :under, :set)`.
"""
function check_identification(restrictions::SVARRestrictions, model::VARModel{T};
                              n_points::Int=10,
                              rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(model)
    n_points >= 1 || throw(ArgumentError("n_points must be ≥ 1, got $n_points"))
    restrictions.n_vars == n || throw(ArgumentError(
        "Restriction dimension ($(restrictions.n_vars)) must match model ($n)"))
    orders = _zeros_per_shock(restrictions, n)
    ranks = zeros(Int, n)
    phi_len = _rwz_phi_length(restrictions)
    p = model.p
    mrow = size(model.B, 1)
    try
        _accumulate_zf_ranks!(ranks, restrictions, model.B, model.Sigma, n, p, phi_len)
    catch err
        _is_rejectable_draw_error(err) || rethrow(err)
    end
    _suppress_warnings() do
        for _ in 1:n_points
            try
                A0, Aplus = _draw_generic_structural(n, mrow, T; rng=rng)
                B_rf, Sigma_rf = _struct_to_rf(A0, Aplus)
                _accumulate_zf_ranks!(ranks, restrictions, B_rf, Sigma_rf, n, p, phi_len)
            catch err
                _is_rejectable_draw_error(err) || rethrow(err)
                continue
            end
        end
    end
    status, n_over = _rwz_classify(ranks, orders, n, !isempty(restrictions.signs))
    IdentificationStatus(status, ranks, orders, n_over)
end

function check_identification(restrictions::SVARRestrictions, n::Int)
    n >= 1 || throw(ArgumentError("n must be positive"))
    restrictions.n_vars == n || throw(ArgumentError(
        "Restriction dimension ($(restrictions.n_vars)) must match n=$n"))
    orders = _zeros_per_shock(restrictions, n)
    ranks = copy(orders)
    status, n_over = _rwz_classify(ranks, orders, n, !isempty(restrictions.signs))
    IdentificationStatus(status, ranks, orders, n_over)
end

function _assert_rwz_identified(restrictions::SVARRestrictions, model::VARModel;
                                rng::AbstractRNG=Random.default_rng())
    # Copy so generic (A₀, A₊) probes do not advance the caller's Q-draw stream.
    st = check_identification(restrictions, model; rng=copy(rng))
    if st.status === :under || st.status === :over
        n = nvars(model)
        needed = [n - j for j in 1:n]
        kind = st.status === :over ? "overidentified" : "underidentified"
        throw(IdentificationError(
            "SVAR is $kind by the RWZ rank/order condition " *
            "(status=:$(st.status); ranks=$(st.ranks), orders=$(st.orders), " *
            "need rank(M_j)=$needed)."))
    end
    st
end

"""
Compute QR sign patterns for each shock's M_j matrix.
Returns Vector{Vector{Int}} where signs[j][col] is +1 or -1.
Used to fix the sign convention across finite-difference perturbations.
"""
function _compute_qr_signs(Q::AbstractMatrix{T}, setup::_AriasSVARSetup,
                           restrictions::SVARRestrictions,
                           Phi::Vector{<:AbstractMatrix}, L::LowerTriangular;
                           B=nothing, C1=nothing) where {T}
    n = size(Q, 1)
    signs = Vector{Vector{Int}}(undef, n)

    for j in 1:n
        parts = Matrix{T}[]
        j > 1 && push!(parts, Matrix{T}(Q[:, 1:j-1]'))
        ZF_j = _compute_ZF(restrictions, Phi, L, j; B=B, C1=C1)
        size(ZF_j, 1) > 0 && push!(parts, Matrix{T}(ZF_j))
        push!(parts, Matrix{T}(setup.W[j]))

        M_j = vcat(parts...)
        F = qr(M_j')
        R_diag = diag(F.R)
        signs[j] = [R_diag[col] < zero(T) ? -1 : 1 for col in 1:length(R_diag)]
    end

    signs
end

"""Convert orthogonal matrix Q to sphere coordinates w using setup's W matrices.

When `ref_signs` is provided, uses the fixed sign convention from a reference point
instead of re-evaluating `sign(diag(R))`. This eliminates discontinuities in the
QR sign correction that cause unreliable finite-difference Jacobians.
"""
function _Q_to_spheres(Q::AbstractMatrix, setup::_AriasSVARSetup, restrictions::SVARRestrictions,
                       Phi::Vector{<:AbstractMatrix}, L::LowerTriangular;
                       ref_signs::Union{Nothing, Vector{Vector{Int}}}=nothing,
                       B=nothing, C1=nothing)
    T = eltype(Q)
    n = size(Q, 1)
    w_parts = Vector{Vector{T}}()

    for j in 1:n
        s_j = setup.sphere_dims[j]
        s_j <= 0 && continue

        # Build M_j: stack [Q[:,1:j-1]'; ZF_j; W_j]
        parts = Matrix{T}[]
        j > 1 && push!(parts, Matrix{T}(Q[:, 1:j-1]'))
        ZF_j = _compute_ZF(restrictions, Phi, L, j; B=B, C1=C1)
        size(ZF_j, 1) > 0 && push!(parts, Matrix{T}(ZF_j))
        push!(parts, Matrix{T}(setup.W[j]))

        M_j = vcat(parts...)
        # QR of M_j' with sign correction
        F = qr(M_j')
        K = Matrix{T}(F.Q)
        R_diag = diag(F.R)
        if ref_signs !== nothing
            # Use reference signs to avoid discontinuity across finite differences
            for col in 1:size(K, 2)
                ref_signs[j][col] < 0 && (K[:, col] = -K[:, col])
            end
        else
            for col in 1:size(K, 2)
                R_diag[col] < 0 && (K[:, col] = -K[:, col])
            end
        end

        # Last s_j columns form null-space basis
        K_j = K[:, end-s_j+1:end]
        w_j = K_j' * Q[:, j]
        push!(w_parts, w_j)
    end

    vcat(w_parts...)
end

"""Convert sphere coordinates w back to orthogonal matrix Q."""
function _spheres_to_Q(w::AbstractVector{T}, setup::_AriasSVARSetup, restrictions::SVARRestrictions,
                       Phi::Vector{<:AbstractMatrix}, L::LowerTriangular;
                       B=nothing, C1=nothing) where {T}
    n = length(setup.sphere_dims)
    Q = zeros(T, n, n)
    offset = 0

    for j in 1:n
        s_j = setup.sphere_dims[j]
        if s_j <= 0
            throw(IdentificationError("Zero restrictions over-constrain shock $j"))
        end

        w_j = w[offset+1:offset+s_j]
        offset += s_j

        # Build same M_j as in _Q_to_spheres
        parts = Matrix{T}[]
        j > 1 && push!(parts, Matrix{T}(Q[:, 1:j-1]'))
        ZF_j = _compute_ZF(restrictions, Phi, L, j; B=B, C1=C1)
        size(ZF_j, 1) > 0 && push!(parts, Matrix{T}(ZF_j))
        push!(parts, Matrix{T}(setup.W[j]))

        M_j = vcat(parts...)
        F = qr(M_j')
        K = Matrix{T}(F.Q)
        R_diag = diag(F.R)
        for col in 1:size(K, 2)
            R_diag[col] < 0 && (K[:, col] = -K[:, col])
        end

        K_j = K[:, end-s_j+1:end]
        Q[:, j] = K_j * w_j
    end

    Q
end

"""Draw w from product of unit spheres S^{s_1-1} × ... × S^{s_n-1}."""
function _draw_w(setup::_AriasSVARSetup{T}; rng::AbstractRNG=Random.default_rng()) where {T}
    w_parts = Vector{T}()
    for s_j in setup.sphere_dims
        x = randn(rng, T, s_j)
        append!(w_parts, x / norm(x))
    end
    Vector{T}(w_parts)
end

# --- Volume Element Closures ---

"""
Build closure ff_h: structural_vec → (B, Σ, w) for volume element computation.
Maps structural parameters to reduced-form parameters + sphere coordinates.

QR column signs use `R_diag < 0` on the current input. For `ForwardDiff.Dual`
that comparison is on the primal value, so the sign pattern is frozen at the
seed point (the Issue #37 FD discontinuity does not arise under AD). The
closure is stateless and thread-safe.
"""
function _build_ff_h(setup::_AriasSVARSetup, restrictions::SVARRestrictions,
                     n::Int, m::Int, p::Int, max_h::Int)
    function ff_h(x::AbstractVector)
        A0, Aplus = _unpack_structural(x, n, m)
        A0m = A0 isa Matrix ? A0 : Matrix(A0)
        Aplusm = Aplus isa Matrix ? Aplus : Matrix(Aplus)
        B_rf, Sigma_rf = _struct_to_rf(A0m, Aplusm)

        L_rf = safe_cholesky(Sigma_rf)
        Q_rf = Matrix(L_rf') * A0m

        Phi_rf = ma_coefficients(B_rf isa Matrix ? B_rf : Matrix(B_rf), n, p, max_h + 1)

        w_rf = _Q_to_spheres(Q_rf, setup, restrictions, Phi_rf, L_rf; B=B_rf)

        vcat(vec(B_rf), _vech(Sigma_rf), w_rf)
    end
    ff_h
end

"""FD companion of `_build_ff_h` with QR signs frozen at the first (primal) call."""
function _build_ff_h_fd(setup::_AriasSVARSetup{T}, restrictions::SVARRestrictions,
                        n::Int, m::Int, p::Int, max_h::Int) where {T}
    ref_signs_storage = Ref{Union{Nothing, Vector{Vector{Int}}}}(nothing)
    first_call = Ref(true)
    function ff_h(x::AbstractVector)
        A0, Aplus = _unpack_structural(x, n, m)
        B_rf, Sigma_rf = _struct_to_rf(Matrix{T}(A0), Matrix{T}(Aplus))
        L_rf = safe_cholesky(Sigma_rf)
        Q_rf = Matrix{T}(L_rf') * A0
        Phi_rf = ma_coefficients(Matrix{T}(B_rf), n, p, max_h + 1)
        if first_call[]
            ref_signs_storage[] = _compute_qr_signs(Q_rf, setup, restrictions, Phi_rf, L_rf; B=B_rf)
            first_call[] = false
        end
        w_rf = _Q_to_spheres(Q_rf, setup, restrictions, Phi_rf, L_rf;
                              ref_signs=ref_signs_storage[], B=B_rf)
        vcat(vec(B_rf), _vech(Sigma_rf), w_rf)
    end
    ff_h
end

"""Extract lower triangle of symmetric matrix (vectorize unique elements)."""
function _vech(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    v = Vector{T}(undef, n * (n + 1) ÷ 2)
    k = 0
    for j in 1:n, i in j:n
        k += 1
        v[k] = A[i, j]
    end
    v
end

"""Build closure for zero restriction evaluation at structural params."""
function _build_zero_restrictions_fn(restrictions::SVARRestrictions, n::Int, m::Int, p::Int, max_h::Int,
                                      ::Type{T}=Float64) where {T}
    isempty(restrictions.zeros) && return x -> T[]

    function zero_fn(x::AbstractVector)
        Tx = eltype(x)
        A0, Aplus = _unpack_structural(x, n, m)
        A0m = A0 isa Matrix ? A0 : Matrix(A0)
        Aplusm = Aplus isa Matrix ? Aplus : Matrix(Aplus)
        A0_inv = robust_inv(A0m)
        B_rf = Aplusm * A0_inv

        Phi = ma_coefficients(B_rf, n, p, max(max_h + 1, 1))

        vals = Vector{Tx}(undef, length(restrictions.zeros))
        for (idx, zr) in enumerate(restrictions.zeros)
            vals[idx] = _zero_residual(zr, Phi, A0m, Aplusm, A0_inv, B_rf, n, p)
        end
        vals
    end
    zero_fn
end

_zero_residual(zr::ZeroRestriction, Phi, A0, Aplus, A0_inv, B_rf, n, p) =
    (Phi[zr.horizon + 1] * A0_inv)[zr.variable, zr.shock]

function _zero_residual(zr::LongRunZeroRestriction, Phi, A0, Aplus, A0_inv, B_rf, n, p)
    C1 = _C1_from_B(B_rf, n, p)
    (C1 * A0_inv)[zr.variable, zr.shock]
end

_zero_residual(zr::A0ZeroRestriction, Phi, A0, Aplus, A0_inv, B_rf, n, p) =
    A0[zr.variable, zr.shock]

_zero_residual(zr::AplusZeroRestriction, Phi, A0, Aplus, A0_inv, B_rf, n, p) =
    Aplus[_aplus_row(zr, n), zr.shock]

# --- Importance Weights (Proposition 4, Arias et al. 2018) ---

"""
Compute draw-dependent importance weight for Q.

For zero+sign restrictions, the weight corrects for the non-uniform proposal
distribution on Q induced by the zero-restriction constraint manifold.
Uses the volume element formula from Proposition 4 of Arias et al. (2018, Econometrica).

The weight is: w = exp(log|v_e(f_h)| - log|v_e(ff_h|Z=0)|)
where f_h is the structural-to-reduced-form map and ff_h includes sphere coordinates.
"""
function _compute_importance_weight(Q::Matrix{T}, model::VARModel{T},
                                     setup::_AriasSVARSetup{T}, restrictions::SVARRestrictions,
                                     Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}},
                                     ff_h, zero_fn) where {T}
    isempty(restrictions.zeros) && return one(T)

    n = nvars(model)
    m = size(model.B, 1)  # 1 + n*p

    A0, Aplus = _rf_to_struct(model.B, L, Q)
    structpara = _pack_structural(A0, Aplus)

    # Analytical numerator: log|v_e(f_h)|
    # From Proposition 4: n(n+1)/2 * log(2) - (2n + m + 1) * log|det(A0)|
    log_ve_fh = T(n * (n + 1)) / 2 * log(T(2)) - T(2n + m + 1) * _log_abs_det(A0)

    log_ve_gfhZ = _log_volume_element(ff_h, structpara, zero_fn)

    # Guard against numerical issues: caller skips non-finite log-weights
    # (do not substitute unit weight — SID-19).
    log_w = log_ve_fh - log_ve_gfhZ
    isfinite(log_w) || return T(NaN)

    exp(log_w)
end

function _compute_importance_weight(Q::Matrix{T}, model::VARModel{T},
                                     setup::_AriasSVARSetup{T}, restrictions::SVARRestrictions,
                                     Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}}) where {T}
    isempty(restrictions.zeros) && return one(T)
    n = nvars(model)
    p = model.p
    m = size(model.B, 1)
    max_h = maximum(_restriction_horizon(zr) for zr in restrictions.zeros)
    ff_h = _build_ff_h(setup, restrictions, n, m, p, max_h)
    zero_fn = _build_zero_restrictions_fn(restrictions, n, m, p, max_h, T)
    _compute_importance_weight(Q, model, setup, restrictions, Phi, L, ff_h, zero_fn)
end

"""FD importance weight (SID-27 pin): freeze-QR finite differences, new closures."""
function _compute_importance_weight_fd(Q::Matrix{T}, model::VARModel{T},
                                       setup::_AriasSVARSetup{T}, restrictions::SVARRestrictions,
                                       Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}}) where {T}
    isempty(restrictions.zeros) && return one(T)
    n = nvars(model)
    p = model.p
    m = size(model.B, 1)
    max_h = maximum(_restriction_horizon(zr) for zr in restrictions.zeros)
    A0, Aplus = _rf_to_struct(model.B, L, Q)
    structpara = _pack_structural(A0, Aplus)
    log_ve_fh = T(n * (n + 1)) / 2 * log(T(2)) - T(2n + m + 1) * _log_abs_det(A0)
    ff_h = _build_ff_h_fd(setup, restrictions, n, m, p, max_h)
    zero_fn = _build_zero_restrictions_fn(restrictions, n, m, p, max_h, T)
    log_ve_gfhZ = _log_volume_element_fd(ff_h, structpara, zero_fn)
    log_w = log_ve_fh - log_ve_gfhZ
    isfinite(log_w) || return T(NaN)
    exp(log_w)
end

# Backward-compatible signature for pure sign restrictions (no setup needed)
function _compute_importance_weight(Q::Matrix{T}, r::SVARRestrictions,
                                     Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}}) where {T}
    isempty(r.zeros) && return one(T)
    # This path should not be reached for zero restrictions in the new code,
    # but kept for backward compatibility
    one(T)
end

# --- Main Arias Identification ---

"""
    identify_arias(model, restrictions, horizon; n_draws=1000, n_rotations=1000, compute_weights=true) -> AriasSVARResult

Identify SVAR using Arias et al. (2018) with zero and sign restrictions.

Uses importance sampling with draw-dependent weights (Proposition 4) for zero+sign restriction
combinations. For pure sign restrictions, draws uniformly from O(n) with unit weights.

Calls [`check_identification`](@ref) first. An `:under` or non-drawable `:over`
RWZ rank/order diagnosis throws [`IdentificationError`](@ref). Sign-only
patterns are `:set` and sampled as usual.

The result reports Kish's **effective sample size** of those weights (`ess`,
`ess_fraction`). Uneven weights mean the weighted IRF summaries rest on fewer
effective draws than `n_draws`; below 10% of the nominal count the sampler is
reported as degenerate and a warning is emitted.

# Keywords
- `n_draws::Int=1000`: Target number of accepted draws
- `n_rotations::Int=1000`: Maximum attempts per target draw
- `compute_weights::Bool=true`: Compute importance weights (set false for faster exploratory analysis)
- `normalize_weights::Bool=true`: Scale the stored weights to sum to 1. Pass `false`
  to keep them on the raw volume-element scale, which is required when pooling
  across draws of `(B, Σ)` — see [`identify_arias_bayesian`](@ref). `ess` is
  scale-invariant and unaffected either way.
- `rng::AbstractRNG`: Random number generator (thread through for reproducible `ess`)
- `n_narrative_sims::Int=1000`: Monte Carlo draws of ``ε* ~ N(0,I)`` used to
  estimate ADRR ``ω̂`` when narrative restrictions are present. The importance
  weight is multiplied by ``1/ω̂``. Unused (and stored as 0) when they are absent.

Draws are assigned to pre-seeded slots (`seeds = rand(rng, UInt64, n_draws)`,
per-slot `MersenneTwister`) and filled with `Threads.@threads`, so the accepted
rotations are invariant to `JULIA_NUM_THREADS`. The result stores wall-clock
`elapsed` for the accept loop and `weights_elapsed` as the summed time spent
in the volume-element weight.
"""
function identify_arias(model::VARModel{T}, restrictions::SVARRestrictions, horizon::Int;
                        n_draws::Int=1000, n_rotations::Int=1000,
                        compute_weights::Bool=true,
                        normalize_weights::Bool=true,
                        setup::Union{Nothing,_AriasSVARSetup}=nothing,
                        rng::AbstractRNG=Random.default_rng(),
                        seed::Union{Integer,Nothing}=nothing,
                        check_id::Bool=true,
                        n_narrative_sims::Int=1000) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    n = nvars(model)
    @assert restrictions.n_vars == n "Restriction dimension must match model"
    check_id && _assert_rwz_identified(restrictions, model; rng=rng)

    has_narrative = _has_narrative(restrictions)
    n_nar_sims = 0
    if has_narrative
        n_narrative_sims >= 1 || throw(ArgumentError("n_narrative_sims must be ≥ 1"))
        _validate_narrative_sample(restrictions, size(model.U, 1))
        n_nar_sims = n_narrative_sims
    end

    max_h = max(horizon,
        isempty(restrictions.zeros) ? 0 : maximum(_restriction_horizon(zr) for zr in restrictions.zeros) + 1,
        isempty(restrictions.signs) ? 0 : maximum(_restriction_horizon(sr) for sr in restrictions.signs) + 1)

    Phi, L = _compute_ma_coefficients(model, max_h), safe_cholesky(model.Sigma)
    has_zeros = !isempty(restrictions.zeros)
    has_fevd = any(s -> s isa FEVDShareRestriction, restrictions.signs)
    needs_struct = any(s -> s isa Union{A0SignRestriction, AplusSignRestriction}, restrictions.signs)
    C1 = any(z -> z isa LongRunZeroRestriction, restrictions.zeros) ?
         first(_long_run_multiplier(model.B, model.Sigma, n, model.p)) : nothing

    # Create setup once for zero restrictions (W matrices fixed for all draws),
    # or reuse a caller-supplied setup (Bayesian pooling).
    if has_zeros && setup === nothing
        setup = _AriasSVARSetup(restrictions, n, T; rng=rng)
    end

    max_h_zeros = isempty(restrictions.zeros) ? 0 :
                  maximum(_restriction_horizon(zr) for zr in restrictions.zeros)
    mrow = size(model.B, 1)
    ff_h = has_zeros && compute_weights ?
           _build_ff_h(setup, restrictions, n, mrow, model.p, max_h_zeros) : nothing
    zero_fn = has_zeros && compute_weights ?
              _build_zero_restrictions_fn(restrictions, n, mrow, model.p, max_h_zeros, T) : nothing

    seeds = rand(rng, UInt64, n_draws)
    Q_store = Vector{Matrix{T}}(undef, n_draws)
    irf_store = Vector{Array{T,3}}(undef, n_draws)
    w_store = fill(T(NaN), n_draws)
    accepted = fill(false, n_draws)
    n_att_slot = zeros(Int, n_draws)
    n_degen_slot = zeros(Int, n_draws)
    last_err_slot = Vector{Any}(nothing, n_draws)
    fatal_slot = Vector{Any}(nothing, n_draws)
    wtime_slot = zeros(Float64, n_draws)

    t0 = time_ns()
    Threads.@threads for d in 1:n_draws
        local_rng = Random.MersenneTwister(seeds[d])
        try
            for _rot in 1:n_rotations
                n_att_slot[d] += 1
                try
                    Q = if has_zeros
                        _draw_Q_with_zero_restrictions(restrictions, Phi, L; rng=local_rng,
                                                       B=model.B, C1=C1)
                    else
                        haar_orthogonal(n, T; rng=local_rng)
                    end
                    irf_full = _compute_irf_for_Q(model, Q, Phi, L, max_h)
                    A0 = Aplus = nothing
                    fevd_props = nothing
                    if needs_struct
                        A0, Aplus = _rf_to_struct(model.B, L, Q)
                    end
                    if has_fevd
                        fevd_props = _compute_fevd(irf_full, n, size(irf_full, 1))[2]
                    end
                    (has_zeros && !_check_zero_restrictions(irf_full, restrictions)) && continue
                    !_check_rejections(restrictions, irf_full, A0, Aplus, fevd_props) && continue
                    if has_narrative
                        shocks = compute_structural_shocks(model, Q)
                        !_narrative_restrictions_hold(restrictions, irf_full, shocks) && continue
                    end
                    irf = irf_full[1:horizon, :, :]

                    w = one(T)
                    if has_zeros && compute_weights
                        tw = time_ns()
                        w = _compute_importance_weight(Q, model, setup, restrictions, Phi, L,
                                                       ff_h, zero_fn)
                        wtime_slot[d] += (time_ns() - tw) / 1e9
                    end
                    if has_narrative
                        ω = _omega_hat(restrictions, irf_full, n, T; n_sims=n_nar_sims, rng=local_rng)
                        if !(ω > 0) || !isfinite(ω)
                            n_degen_slot[d] += 1
                            continue
                        end
                        w *= 1 / ω
                    end
                    if !isfinite(w)
                        n_degen_slot[d] += 1
                        continue
                    end

                    Q_store[d] = Q
                    irf_store[d] = irf
                    w_store[d] = w
                    accepted[d] = true
                    break
                catch err
                    _is_rejectable_draw_error(err) || rethrow(err)
                    last_err_slot[d] = err
                    continue
                end
            end
        catch err
            fatal_slot[d] = err
        end
    end
    elapsed = T((time_ns() - t0) / 1e9)
    weights_elapsed = T(sum(wtime_slot))

    for d in 1:n_draws
        fatal_slot[d] === nothing && continue
        throw(fatal_slot[d])
    end

    kept = findall(accepted)
    n_attempts = sum(n_att_slot)
    n_degenerate = sum(n_degen_slot)
    last_err = nothing
    for d in n_draws:-1:1
        if last_err_slot[d] !== nothing
            last_err = last_err_slot[d]
            break
        end
    end

    isempty(kept) && throw(IdentificationError("No valid identification after $n_attempts attempts" *
        (last_err === nothing ? "" :
         "; last rejectable failure: $(typeof(last_err)): $(sprint(showerror, last_err))")))

    n_acc = length(kept)
    Q_draws = Q_store[kept]
    weights = w_store[kept]
    irf_array = zeros(T, n_acc, horizon, n, n)
    for (i, idx) in enumerate(kept)
        irf_array[i, :, :, :] = irf_store[idx]
    end

    n_degenerate > 0 && @warn "identify_arias: $n_degenerate draw(s) skipped because the importance log-weight was non-finite"

    # ESS is scale-invariant, so it is identical before and after normalization.
    ess = T(_effective_sample_size(weights))
    ess_frac = ess / T(n_acc)
    _warn_low_ess(ess, ess_frac, n_acc, "identify_arias")

    w_out = normalize_weights ? weights ./ sum(weights) : weights
    vnames = copy(model.varnames)
    AriasSVARResult{T}(Q_draws, irf_array, w_out, T(n_acc / n_attempts), restrictions,
                       ess, ess_frac, vnames, n_degenerate, n_nar_sims,
                       elapsed, weights_elapsed)
end

# --- Bayesian Integration ---

"""
    identify_arias_bayesian(post::BVARPosterior, restrictions, horizon; data=nothing, n_rotations=100, quantiles=[0.16,0.5,0.84], compute_weights=true, n_narrative_sims=1000)

Apply Arias identification to each posterior draw.

Creates the `_AriasSVARSetup` once (W matrices fixed across all posterior draws) for consistency.
The RWZ restriction-pattern check also runs once on the first posterior draw.

Importance weights are pooled across posterior draws on the **raw volume-element
scale** and normalized once at the end. Each per-draw call accepts a single
rotation, so normalizing there would force every weight to 1 and reduce the
weighted summaries to unweighted ones.

# Returns
[`BayesianSetIdentifiedSVAR`](@ref) with draws, weights, ESS, `n_unidentified`,
`n_degenerate_weights`, and `n_narrative_sims`. Property `total_accepted` is
`size(irf_draws, 1)`. `irf_quantiles` / `irf_mean` / `acceptance_rates` are stored
for back-compat with the previous NamedTuple return.
"""
function identify_arias_bayesian(post::BVARPosterior, restrictions::SVARRestrictions, horizon::Int;
    data::Union{Nothing,AbstractMatrix}=nothing, n_rotations::Int=100,
    quantiles::Vector{Float64}=[0.16, 0.5, 0.84], compute_weights::Bool=true,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
    n_narrative_sims::Int=1000)
    rng = _resolve_repro_rng(rng, seed)

    use_data = isnothing(data) ? (isempty(post.data) ? nothing : post.data) : data
    p, n = post.p, post.n
    T = eltype(post.Sigma_draws)
    b_vecs, sigmas = extract_chain_parameters(post)
    n_samples = size(b_vecs, 1)
    all_irfs, all_weights = Vector{Array{T,3}}(), T[]
    all_Qs = Matrix{T}[]
    acc_rates = zeros(T, n_samples)
    n_unidentified = 0
    n_degenerate = 0

    has_zeros = !isempty(restrictions.zeros)
    setup = has_zeros ? _AriasSVARSetup(restrictions, n, T; rng=rng) : nothing

    # Restriction-pattern diagnosis is a property of the zeros, not of each
    # posterior reduced form. Probe once on the first draw; inner identify_arias
    # calls skip the generic-rank RNG consumption.
    n_samples >= 1 && _assert_rwz_identified(restrictions,
        parameters_to_model(b_vecs[1, :], sigmas[1, :], p, n, use_data); rng=rng)

    for s in 1:n_samples
        m = parameters_to_model(b_vecs[s,:], sigmas[s,:], p, n, use_data)
        try
            # normalize_weights=false is essential: each inner call accepts a
            # SINGLE draw, so normalizing there would set every weight to 1 and
            # silently discard the importance correction. Pooling happens once,
            # below, across all posterior draws on the common volume-element scale.
            result = identify_arias(m, restrictions, horizon;
                n_draws=1, n_rotations=n_rotations, compute_weights=compute_weights,
                normalize_weights=false, setup=setup, rng=rng, check_id=false,
                n_narrative_sims=n_narrative_sims)
            n_degenerate += result.n_degenerate_weights
            for (i, w) in enumerate(result.weights)
                push!(all_irfs, result.irf_draws[i, :, :, :])
                push!(all_weights, w)
                push!(all_Qs, result.Q_draws[i])
            end
            acc_rates[s] = result.acceptance_rate
        catch err
            if err isa IdentificationError
                n_unidentified += 1
                acc_rates[s] = 0
                continue
            end
            _is_rejectable_draw_error(err) || rethrow(err)
            acc_rates[s] = 0
        end
    end

    isempty(all_irfs) && throw(IdentificationError(
        "No valid identifications across posterior " *
        "(unidentified=$n_unidentified, n_samples=$n_samples)"))
    frac_u = n_unidentified / n_samples
    frac_u > 0.5 && @warn "$n_unidentified/$n_samples posterior draws unidentified"
    n_degenerate > 0 && @warn "identify_arias_bayesian: $n_degenerate draw(s) skipped because the importance log-weight was non-finite"

    n_acc = length(all_irfs)
    irf_array = zeros(T, n_acc, horizon, n, n)
    for (i, irf) in enumerate(all_irfs)
        irf_array[i, :, :, :] = irf
    end
    ess = T(_effective_sample_size(all_weights))
    ess_frac = ess / T(n_acc)
    _warn_low_ess(ess, ess_frac, n_acc, "identify_arias_bayesian")
    w_norm = all_weights ./ sum(all_weights)

    irf_q = zeros(T, horizon, n, n, length(quantiles))
    irf_m = zeros(T, horizon, n, n)
    for h in 1:horizon, i in 1:n, j in 1:n
        vals = irf_array[:, h, i, j]
        irf_m[h, i, j] = sum(w_norm .* vals)
        for (qi, q) in enumerate(quantiles)
            irf_q[h, i, j, qi] = _weighted_quantile(vals, w_norm, q)
        end
    end

    vnames = copy(post.varnames)
    n_nar = _has_narrative(restrictions) ? n_narrative_sims : 0
    BayesianSetIdentifiedSVAR{T}(all_Qs, irf_array, Vector{T}(w_norm), ess, restrictions,
                                 vnames, n_unidentified, n_degenerate,
                                 irf_q, irf_m, acc_rates, ess_frac, n_nar)
end

# Deprecated wrapper for old (chain, p, n, ...) signature
function identify_arias_bayesian(post::BVARPosterior, p::Int, n::Int, restrictions::SVARRestrictions, horizon::Int; kwargs...)
    identify_arias_bayesian(post, restrictions, horizon; kwargs...)
end

"""Parse Arias kwargs from `irf`/`fevd` on `BVARPosterior` (extra keys ignored)."""
function _arias_posterior_kwargs(max_draws::Int; rng::AbstractRNG=Random.default_rng(),
                                 n_rotations::Union{Nothing,Integer}=nothing,
                                 compute_weights::Bool=true,
                                 n_narrative_sims::Int=1000, kwargs...)
    (rng, Int(something(n_rotations, max_draws)), compute_weights, Int(n_narrative_sims))
end

"""Peel `n_draws`/`n_rotations` before splatting into `identify_arias` (frequentist `irf`/`fevd`/`hd`/`compute_Q`).

`default_n_draws` is used when the caller does not pass `n_draws` (`compute_Q` wants 1).
"""
function _arias_freq_kwargs(max_draws::Int; rng::AbstractRNG=Random.default_rng(),
                            n_rotations::Union{Nothing,Integer}=nothing,
                            n_draws::Union{Nothing,Integer}=nothing,
                            default_n_draws::Integer=max_draws,
                            kwargs...)
    (; n_draws=Int(something(n_draws, default_n_draws)),
       n_rotations=Int(something(n_rotations, max_draws)),
       rng, kwargs...)
end

"""Weighted Arias identified-set from a BVAR posterior (`irf`/`fevd` `method=:arias`)."""
function _arias_from_bvar_posterior(post::BVARPosterior, restrictions, horizon;
                                    data=nothing, n_rotations::Int=100,
                                    quantiles::AbstractVector=[0.16, 0.5, 0.84],
                                    rng::AbstractRNG=Random.default_rng(),
                                    compute_weights::Bool=true,
                                    n_narrative_sims::Int=1000)
    isnothing(restrictions) && throw(ArgumentError("arias requires restrictions"))
    restrictions isa SVARRestrictions || throw(ArgumentError(
        ":arias requires restrictions::SVARRestrictions"))
    identify_arias_bayesian(post, restrictions, horizon;
        data=data, n_rotations=n_rotations, quantiles=collect(Float64.(quantiles)),
        compute_weights=compute_weights, rng=rng, n_narrative_sims=n_narrative_sims)
end

# `_weighted_quantile` lives in `src/core/utils.jl` (shared with SignIdentifiedSet summaries).

# --- Convenience Functions ---

"""Create zero restriction: variable doesn't respond to shock at horizon.

Pass `horizon=:long_run` for a Blanchard–Quah long-run zero (``e_v' C(1) L q_s = 0``).
"""
function zero_restriction(variable::Int, shock::Int; horizon::Union{Int,Symbol}=0)
    if horizon === :long_run
        return LongRunZeroRestriction(variable, shock)
    end
    horizon isa Integer || throw(ArgumentError("horizon must be ≥ 0 or :long_run"))
    h = Int(horizon)
    h >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    ZeroRestriction(variable, shock, h)
end

"""Create sign restriction: variable response has given sign (`:positive`/`:negative`).

`horizons::UnitRange` expands to `length(horizons)` `SignRestriction`s.
"""
function sign_restriction(variable::Int, shock::Int, sign::Symbol;
                          horizon::Int=0,
                          horizons::Union{Nothing,AbstractUnitRange{<:Integer}}=nothing)
    s = _parse_sign(sign)
    if horizons !== nothing
        _check_horizon_range(horizons)
        return [SignRestriction(variable, shock, Int(h), s) for h in horizons]
    end
    horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    SignRestriction(variable, shock, horizon, s)
end

"""
    a0_zero_restriction(equation, shock)

Zero restriction on `A0[equation, shock]` in the Rubio-Ramírez–Waggoner–Zha
row convention ``y_t' A_0 = \\ldots``, where the identified matrix is
``A_0 = L^{-T} Q``.

`equation` is the row (the variable entering that structural equation) and
`shock` is the column. This is **not** an entry of the column-convention
impact matrix ``A_0^{-1} = L Q``: `a0_zero_restriction(i, j)` zeros
`(L^{-T} Q)[i, j]`, not `(L Q)[i, j]`.
"""
a0_zero_restriction(equation::Int, shock::Int) = A0ZeroRestriction(equation, shock)

"""
    a0_sign_restriction(equation, shock, sign)

Sign restriction on `A0[equation, shock]` in the same RWZ ``y'A_0`` convention
as [`a0_zero_restriction`](@ref) (`A_0 = L^{-T} Q`, not ``A_0^{-1} = L Q``).
`sign` is `:positive` or `:negative`.
"""
a0_sign_restriction(equation::Int, shock::Int, sign::Symbol) =
    A0SignRestriction(equation, shock, _parse_sign(sign))

"""
    aplus_zero_restriction(equation, shock; lag=1)

Zero restriction on the `lag`-th block of ``A_+ = B A_0``. `lag=0` is the
intercept row of `B` (a single row); `equation` must be 1.
"""
function aplus_zero_restriction(equation::Int, shock::Int; lag::Int=1)
    _check_aplus_lag_equation(lag, equation)
    AplusZeroRestriction(equation, shock, lag)
end

"""
    aplus_sign_restriction(equation, shock, sign; lag=1)

Sign restriction on the `lag`-th block of ``A_+``. `lag=0` is the intercept
row of `B`; `equation` must be 1.
"""
function aplus_sign_restriction(equation::Int, shock::Int, sign::Symbol; lag::Int=1)
    _check_aplus_lag_equation(lag, equation)
    AplusSignRestriction(equation, shock, lag, _parse_sign(sign))
end

"""Kilian–Murphy elasticity bound ``\\mathrm{IRF}_{num}/\\mathrm{IRF}_{den} \\in [\\mathrm{lower}, \\mathrm{upper}]``."""
function elasticity_bound(numerator_var::Int, denominator_var::Int, shock::Int;
                          horizon::Int=0, lower::Real, upper::Real)
    horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    _check_finite_bounds(lower, upper)
    ElasticityBound(numerator_var, denominator_var, shock, horizon, Float64(lower), Float64(upper))
end

"""Magnitude bound on an IRF entry."""
function magnitude_bound(variable::Int, shock::Int; horizon::Int=0, lower::Real, upper::Real)
    horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    _check_finite_bounds(lower, upper)
    MagnitudeBound(variable, shock, horizon, Float64(lower), Float64(upper))
end

"""FEVD-share restriction: shock's contribution to MSE of `variable` at `horizon`."""
function fevd_share_restriction(variable::Int, shock::Int;
                                horizon::Int=0, lower::Real, upper::Real)
    horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    _check_finite_bounds(lower, upper)
    (0 <= lower && upper <= 1) ||
        throw(ArgumentError("FEVD share bounds must lie in [0, 1]"))
    FEVDShareRestriction(variable, shock, horizon, Float64(lower), Float64(upper))
end

"""Cumulative IRF sign restriction over `horizons`."""
function cumulative_restriction(variable::Int, shock::Int, sign::Symbol;
                                horizons::AbstractUnitRange{<:Integer})
    _check_horizon_range(horizons)
    CumulativeRestriction(variable, shock, UnitRange{Int}(horizons), _parse_sign(sign))
end

"""Narrative shock-sign restriction: structural shock `shock` has `sign` at `dates`."""
function narrative_shock_restriction(shock::Int, dates, sign::Symbol)
    ds = collect(Int, dates)
    isempty(ds) && throw(ArgumentError("narrative dates must be nonempty"))
    all(d -> d >= 1, ds) || throw(ArgumentError("narrative dates must be ≥ 1"))
    NarrativeShockRestriction(shock, ds, _parse_sign(sign))
end

"""
    narrative_contribution_restriction(variable, shock, window; kind=:most_important)

ADRR historical-decomposition restriction over `window`.

- `kind=:most_important` (Type A, default): ``|H_j| > \\max_{k \\neq j}|H_k|``
- `kind=:overwhelming` (Type B): ``|H_j| > \\sum_{k \\neq j}|H_k|``
- `kind=:least_important`: ``|H_j| < \\min_{k \\neq j}|H_k|``
"""
function narrative_contribution_restriction(variable::Int, shock::Int,
                                            window::AbstractUnitRange{<:Integer};
                                            kind::Symbol=:most_important)
    _check_contribution_kind(kind)
    w = UnitRange{Int}(window)
    isempty(w) && throw(ArgumentError("narrative contribution window must be nonempty"))
    first(w) >= 1 || throw(ArgumentError("narrative dates must be ≥ 1"))
    NarrativeContributionRestriction(variable, shock, w, kind)
end

"""
    identify_narrative(model, restrictions::SVARRestrictions, horizon; kwargs...)

Thin wrapper around [`identify_arias`](@ref) for ADRR narrative restrictions.
Closure-based `identify_narrative(model, horizon, sign_check, narrative_check)`
remains the set-aware path used by `compute_Q(:narrative)`.
"""
identify_narrative(model::VARModel, restrictions::SVARRestrictions, horizon::Int; kwargs...) =
    identify_arias(model, restrictions, horizon; kwargs...)

"""Weighted identified-set bounds from an `AriasSVARResult` (Kish-weighted quantiles)."""
function irf_bounds(r::AriasSVARResult{T}; quantiles=[0.16, 0.84]) where {T<:AbstractFloat}
    qv = Float64[quantiles[1], quantiles[2]]
    pct = irf_percentiles(r; quantiles=qv)
    (pct[:, :, :, 1], pct[:, :, :, 2])
end

"""Compute weighted IRF percentiles from AriasSVARResult."""
function irf_percentiles(result::AriasSVARResult{T}; quantiles::Vector{Float64}=[0.16, 0.5, 0.84]) where {T}
    _irf_percentiles_from_draws(result.irf_draws, result.weights, quantiles; uniform_unweighted=false)
end

"""Compute weighted mean IRF from AriasSVARResult."""
function irf_mean(result::AriasSVARResult{T}) where {T}
    n_draws, horizon, n_vars, n_shocks = size(result.irf_draws)
    mean_irf = zeros(T, horizon, n_vars, n_shocks)
    for h in 1:horizon, i in 1:n_vars, j in 1:n_shocks
        mean_irf[h, i, j] = sum(result.weights .* result.irf_draws[:, h, i, j])
    end
    mean_irf
end
