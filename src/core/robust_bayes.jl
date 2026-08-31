# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Giacomini–Kitagawa (2021) robust Bayes for set-identified SVARs
# =============================================================================

"""
    RobustBayesResult{T} <: AbstractAnalysisResult

Giacomini–Kitagawa (2021) prior-robust inference for a set-identified SVAR.

For each reduced-form draw `(B, Σ)` the identified set of every IRF entry is
the interval `[ℓ(B,Σ), u(B,Σ)]` over admissible rotations `Q`. Arrays are
`(horizon × n × n)` with `IRF[h, i, j]` the response of variable `i` to shock
`j` at horizon `h-1`.

# Fields
- `lower`, `upper`: posterior means of the identified-set bounds (set of posterior means)
- `robust_lower`, `robust_upper`: smallest interval containing `level` of the
  draws' *sets* (robust credible region). Not post-hoc expanded to nest the
  Haar interval: GK guarantees `Π_Haar(η ∈ CR) ≥ level`, not quantile-interval
  nesting.
- `single_prior_lower`, `single_prior_upper`: Haar / uniform-on-`O(n)` equal-tailed
  credible interval from [`identify_arias_bayesian`](@ref) (all nonempty posterior
  draws, `compute_weights=true`, pooled importance weights).
- `informativeness`: Giacomini–Kitagawa (2021, §2.6) diagnostic
  `κ = 1 − |C_single| / |C_robust|`, averaged over IRF entries with positive
  robust width and clipped to `[0, 1]`. `κ = 0` when the rotation prior is
  uninformative (robust = single-prior); `κ = 1` when the single-prior interval
  is fully prior-driven.
- `empty_set_prob`: posterior probability that the identified set is empty
  (GK's `Π(∅)`)
- `level`: credibility level of the robust region (default `0.68`)
"""
struct RobustBayesResult{T<:AbstractFloat} <: AbstractAnalysisResult
    lower::Array{T,3}
    upper::Array{T,3}
    robust_lower::Array{T,3}
    robust_upper::Array{T,3}
    single_prior_lower::Array{T,3}
    single_prior_upper::Array{T,3}
    informativeness::T
    empty_set_prob::T
    level::T
end

function Base.show(io::IO, r::RobustBayesResult{T}) where {T}
    H, n, _ = size(r.lower)
    _show_spec_table(io, "Giacomini–Kitagawa Robust Bayes",
        ["Horizon" => H,
         "Variables" => n,
         "Level" => _fmt(r.level; digits=2),
         "Empty-set prob" => _fmt(r.empty_set_prob; digits=4),
         "Informativeness" => _fmt(r.informativeness; digits=4)])
    if any(isfinite, r.lower)
        _irf_points_table(io, r.lower[:, :, 1], ["var$i" for i in 1:n],
                          "Posterior-mean lower bound (shock 1)")
        _irf_points_table(io, r.upper[:, :, 1], ["var$i" for i in 1:n],
                          "Posterior-mean upper bound (shock 1)")
    end
end

# Linear (in a column of Q) restriction types — n=2 closed form applies.
_gk_is_linear_zero(::ZeroRestriction) = true
_gk_is_linear_zero(::LongRunZeroRestriction) = true
_gk_is_linear_zero(::A0ZeroRestriction) = true
_gk_is_linear_zero(::AplusZeroRestriction) = true
_gk_is_linear_zero(::AbstractSVARRestriction) = false

_gk_is_linear_ineq(::SignRestriction) = true
_gk_is_linear_ineq(::A0SignRestriction) = true
_gk_is_linear_ineq(::AplusSignRestriction) = true
_gk_is_linear_ineq(::CumulativeRestriction) = true
_gk_is_linear_ineq(::MagnitudeBound) = true
_gk_is_linear_ineq(::AbstractSVARRestriction) = false

function _gk_linear_in_Q(restrictions::SVARRestrictions)
    all(_gk_is_linear_zero, restrictions.zeros) && all(_gk_is_linear_ineq, restrictions.signs)
end

function _gk_phi_length(restrictions::SVARRestrictions, horizon::Int)
    mh = horizon
    for zr in restrictions.zeros
        mh = max(mh, _restriction_horizon(zr) + 1)
    end
    for sr in restrictions.signs
        mh = max(mh, _restriction_horizon(sr) + 1)
    end
    mh
end

"""
    identified_set_bounds(model, restrictions, H; solver=:draws, n_draws=1000, rng)

Identified-set envelope `[ℓ, u]` of each IRF entry over admissible `Q ∈ O(n)`.

`solver=:draws` is the accepted-draw envelope (inner approximation, consistent
as `n_draws → ∞`). `solver=:optimize` maximises/minimises each entry subject to
the restrictions: one-shock `n = 2` uses the Gafarov–Meier–Montiel Olea (2018)
closed form (no NLopt); larger systems use NLopt on [`_spheres_to_Q`](@ref)
coordinates (Giacomini–Kitagawa 2021, Algorithm 1).

Returns `(lower, upper)` arrays of size `(H, n, n)`. Throws
[`IdentificationError`](@ref) when the identified set is empty.
"""
function identified_set_bounds(model::VARModel{T}, restrictions::SVARRestrictions, horizon::Int;
                               solver::Symbol=:draws, n_draws::Int=1000,
                               rng::AbstractRNG=Random.default_rng(),
                               n_starts::Int=4, n_rotations::Int=200,
                               threaded::Bool=true) where {T<:AbstractFloat}
    horizon >= 1 || throw(ArgumentError("horizon must be ≥ 1"))
    n_draws >= 1 || throw(ArgumentError("n_draws must be ≥ 1"))
    solver === :draws || solver === :optimize ||
        throw(ArgumentError("solver must be :draws or :optimize, got :$solver"))
    n = nvars(model)
    restrictions.n_vars == n || throw(ArgumentError("Restriction dimension must match model"))

    if solver === :optimize && n == 2 && _gk_linear_in_Q(restrictions)
        return _n2_identified_set_bounds(model, restrictions, horizon)
    elseif solver === :optimize
        return _nlopt_identified_set_bounds(model, restrictions, horizon;
                                            rng=rng, n_starts=n_starts,
                                            n_rotations=n_rotations)
    end
    _draws_identified_set_bounds(model, restrictions, horizon;
                                 n_draws=n_draws, rng=rng, threaded=threaded)
end

# -----------------------------------------------------------------------------
# n=2 closed form (GMMO 2018)
# -----------------------------------------------------------------------------

# Q(θ, s) = [cosθ  -s sinθ; sinθ  s cosθ], s ∈ {+1, −1}.
# Shock-1 column is (cosθ, sinθ); shock-2 column is s (−sinθ, cosθ).

function _n2_row(r::SignRestriction, Phi, L, B, C1, LinvT, n)
    m = (Phi[r.horizon + 1] * L)[r.variable, :]
    (r.shock, Vector(m), false, _gk_sign_lbub(r.sign, eltype(L)))
end
function _n2_row(r::A0SignRestriction, Phi, L, B, C1, LinvT, n)
    m = LinvT[r.variable, :]
    (r.shock, Vector(m), false, _gk_sign_lbub(r.sign, eltype(L)))
end
function _n2_row(r::AplusSignRestriction, Phi, L, B, C1, LinvT, n)
    m = (B * LinvT)[_aplus_row(r, n), :]
    (r.shock, Vector(m), false, _gk_sign_lbub(r.sign, eltype(L)))
end
function _n2_row(r::CumulativeRestriction, Phi, L, B, C1, LinvT, n)
    T = eltype(L)
    m = zeros(T, n)
    for h in r.horizons
        m .+= (Phi[h + 1] * L)[r.variable, :]
    end
    (r.shock, m, false, _gk_sign_lbub(r.sign, T))
end
function _n2_row(r::MagnitudeBound, Phi, L, B, C1, LinvT, n)
    T = eltype(L)
    m = (Phi[r.horizon + 1] * L)[r.variable, :]
    (r.shock, Vector(m), false, (T(r.lower), T(r.upper)))
end
function _n2_row(r::ZeroRestriction, Phi, L, B, C1, LinvT, n)
    T = eltype(L)
    m = (Phi[r.horizon + 1] * L)[r.variable, :]
    (r.shock, Vector(m), true, (zero(T), zero(T)))
end
function _n2_row(r::LongRunZeroRestriction, Phi, L, B, C1, LinvT, n)
    T = eltype(L)
    m = (C1 * L)[r.variable, :]
    (r.shock, Vector(m), true, (zero(T), zero(T)))
end
function _n2_row(r::A0ZeroRestriction, Phi, L, B, C1, LinvT, n)
    T = eltype(L)
    m = LinvT[r.variable, :]
    (r.shock, Vector(m), true, (zero(T), zero(T)))
end
function _n2_row(r::AplusZeroRestriction, Phi, L, B, C1, LinvT, n)
    T = eltype(L)
    m = (B * LinvT)[_aplus_row(r, n), :]
    (r.shock, Vector(m), true, (zero(T), zero(T)))
end

_gk_sign_lbub(sign::Int, ::Type{T}) where {T} =
    sign > 0 ? (zero(T), T(Inf)) : (T(-Inf), zero(T))

# Map a row `m` on Q[:, shock] to coefficients of (cosθ, sinθ), before the
# reflection sign s on shock 2.
function _n2_ab(shock::Int, m::AbstractVector{T}) where {T}
    shock == 1 ? (m[1], m[2]) : (m[2], -m[1])
end

function _n2_value(a::T, b::T, θ::T, shock::Int, s::Int) where {T}
    raw = a * cos(θ) + b * sin(θ)
    shock == 1 ? raw : T(s) * raw
end

function _n2_identified_set_bounds(model::VARModel{T}, restrictions::SVARRestrictions,
                                   horizon::Int) where {T<:AbstractFloat}
    n = 2
    max_h = _gk_phi_length(restrictions, horizon)
    Phi = ma_coefficients(model.B, n, model.p, max_h)
    L = safe_cholesky(model.Sigma)
    C1 = any(z -> z isa LongRunZeroRestriction, restrictions.zeros) ?
         _C1_from_B(model.B, n, model.p) : nothing
    needs_Linv = any(z -> z isa Union{A0ZeroRestriction, AplusZeroRestriction}, restrictions.zeros) ||
                 any(s -> s isa Union{A0SignRestriction, AplusSignRestriction}, restrictions.signs)
    LinvT = needs_Linv ? Matrix{T}(L' \ I(n)) : nothing

    forms = Tuple{Int,T,T,Bool,T,T}[]
    for zr in restrictions.zeros
        shock, m, eq, (lb, ub) = _n2_row(zr, Phi, L, model.B, C1, LinvT, n)
        a, b = _n2_ab(shock, m)
        push!(forms, (shock, a, b, eq, lb, ub))
    end
    for sr in restrictions.signs
        shock, m, eq, (lb, ub) = _n2_row(sr, Phi, L, model.B, C1, LinvT, n)
        a, b = _n2_ab(shock, m)
        push!(forms, (shock, a, b, eq, lb, ub))
    end

    ML = [Phi[h] * L for h in 1:horizon]
    lower = fill(T(Inf), horizon, n, n)
    upper = fill(T(-Inf), horizon, n, n)
    found = false
    ang_tol = sqrt(eps(T)) * 10

    for s in (1, -1)
        eqs = Tuple{Int,T,T}[]
        ineqs = Tuple{Int,T,T,T,T}[]
        for (shock, a, b, eq, lb, ub) in forms
            if eq
                push!(eqs, (shock, a, b))
            else
                push!(ineqs, (shock, a, b, lb, ub))
            end
        end
        candidates = T[]
        if !isempty(eqs)
            # Each equality has two roots on the circle; keep those that satisfy all.
            shock0, a0, b0 = eqs[1]
            R0 = hypot(a0, b0)
            R0 < ang_tol && continue
            α0 = atan(b0, a0)
            for θ in (α0 + T(π) / 2, α0 + 3T(π) / 2)
                θn = mod(θ, T(2π))
                _n2_point_feasible(θn, s, eqs, ineqs, ang_tol) && push!(candidates, θn)
            end
            isempty(candidates) && continue
            for θ in candidates
                _n2_accumulate_irf!(lower, upper, ML, θ, s)
                found = true
            end
        else
            arcs = _n2_feasible_arcs(s, ineqs, ang_tol)
            isempty(arcs) && continue
            found = true
            for (θ0, θ1) in arcs
                _n2_accumulate_irf!(lower, upper, ML, θ0, s)
                _n2_accumulate_irf!(lower, upper, ML, θ1, s)
                for h in 1:horizon, i in 1:n, j in 1:n
                    a, b = _n2_irf_ab_h(ML[h], i, j)
                    vmin, vmax = _n2_trig_extrema(a, b, θ0, θ1, j, s)
                    lower[h, i, j] = min(lower[h, i, j], vmin)
                    upper[h, i, j] = max(upper[h, i, j], vmax)
                end
            end
        end
    end

    found || throw(IdentificationError("Identified set is empty"))
    (lower, upper)
end

function _n2_irf_ab_h(ML::AbstractMatrix{T}, i::Int, j::Int) where {T}
    m = ML[i, :]
    j == 1 ? (m[1], m[2]) : (m[2], -m[1])
end

function _n2_point_feasible(θ::T, s::Int, eqs, ineqs, tol::T) where {T}
    for (shock, a, b) in eqs
        v = _n2_value(a, b, θ, shock, s)
        abs(v) > tol && return false
    end
    for (shock, a, b, lb, ub) in ineqs
        v = _n2_value(a, b, θ, shock, s)
        (v < lb - tol || v > ub + tol) && return false
    end
    true
end

function _n2_feasible_arcs(s::Int, ineqs, tol::T) where {T}
    angs = T[zero(T), T(2π)]
    for (shock, a, b, lb, ub) in ineqs
        # Boundary of a half-circle / slab: a_eff cos + b_eff sin = bound
        for bnd in (lb, ub)
            isfinite(bnd) || continue
            a_eff, b_eff = _n2_signed_ab(shock, s, a, b)
            # a_eff cos + b_eff sin = bnd
            R = hypot(a_eff, b_eff)
            R < tol && continue
            # R cos(θ − α) = bnd, α = atan(b_eff, a_eff)
            c = bnd / R
            abs(c) > 1 + 10tol && continue
            c = clamp(c, -one(T), one(T))
            α = atan(b_eff, a_eff)
            δ = acos(c)
            push!(angs, mod(α + δ, T(2π)))
            push!(angs, mod(α - δ, T(2π)))
        end
    end
    sort!(angs)
    uniq = T[angs[1]]
    for θ in angs
        abs(θ - uniq[end]) > 100 * eps(T) && push!(uniq, θ)
    end
    if abs(uniq[end] - T(2π)) > 100 * eps(T)
        push!(uniq, T(2π))
    end
    if abs(uniq[1]) > 100 * eps(T)
        pushfirst!(uniq, zero(T))
    end
    arcs = Tuple{T,T}[]
    for i in 1:length(uniq)-1
        θ0, θ1 = uniq[i], uniq[i+1]
        abs(θ1 - θ0) < 100 * eps(T) && continue
        θm = (θ0 + θ1) / 2
        _n2_point_feasible(θm, s, Tuple{Int,T,T}[], ineqs, tol) && push!(arcs, (θ0, θ1))
    end
    arcs
end

function _n2_signed_ab(shock::Int, s::Int, a::T, b::T) where {T}
    shock == 1 ? (a, b) : (T(s) * a, T(s) * b)
end

function _n2_trig_extrema(a::T, b::T, θ0::T, θ1::T, shock::Int, s::Int) where {T}
    f(θ) = _n2_value(a, b, θ, shock, s)
    vmin = min(f(θ0), f(θ1))
    vmax = max(f(θ0), f(θ1))
    R = hypot(a, b)
    R < 100 * eps(T) && return (vmin, vmax)
    α = atan(b, a)
    # shock 2 multiplies by s: peak of s * R cos(θ−α) is α if s>0 else α+π
    peak = s > 0 || shock == 1 ? α : α + T(π)
    twoπ = T(2π)
    for k in -2:2
        for θstar in (peak + k * twoπ, peak + T(π) + k * twoπ)
            if θ0 - 100 * eps(T) <= θstar <= θ1 + 100 * eps(T)
                v = f(θstar)
                vmin = min(vmin, v)
                vmax = max(vmax, v)
            end
        end
    end
    (vmin, vmax)
end

function _n2_accumulate_irf!(lower, upper, ML, θ::T, s::Int) where {T}
    c, sn = cos(θ), sin(θ)
    for h in eachindex(ML)
        M = ML[h]
        for i in 1:2
            v1 = M[i, 1] * c + M[i, 2] * sn
            v2 = T(s) * (-M[i, 1] * sn + M[i, 2] * c)
            lower[h, i, 1] = min(lower[h, i, 1], v1)
            upper[h, i, 1] = max(upper[h, i, 1], v1)
            lower[h, i, 2] = min(lower[h, i, 2], v2)
            upper[h, i, 2] = max(upper[h, i, 2], v2)
        end
    end
    nothing
end

# -----------------------------------------------------------------------------
# Draw envelope
# -----------------------------------------------------------------------------

function _gk_check_Q(model::VARModel{T}, Q::Matrix{T}, restrictions::SVARRestrictions,
                     Phi, L, max_h) where {T<:AbstractFloat}
    irf_full = _compute_irf_for_Q(model, Q, Phi, L, max_h)
    needs_struct = any(s -> s isa Union{A0SignRestriction, AplusSignRestriction}, restrictions.signs)
    has_fevd = any(s -> s isa FEVDShareRestriction, restrictions.signs)
    A0 = Aplus = nothing
    fevd_props = nothing
    if needs_struct
        A0, Aplus = _rf_to_struct(model.B, L, Q)
    end
    if has_fevd
        fevd_props = _compute_fevd(irf_full, nvars(model), size(irf_full, 1))[2]
    end
    has_zeros = !isempty(restrictions.zeros)
    (has_zeros && !_check_zero_restrictions(irf_full, restrictions)) && return nothing
    !_check_rejections(restrictions, irf_full, A0, Aplus, fevd_props) && return nothing
    if _has_narrative(restrictions)
        shocks = compute_structural_shocks(model, Q)
        !_narrative_restrictions_hold(restrictions, irf_full, shocks) && return nothing
    end
    irf_full
end

function _draws_identified_set_bounds(model::VARModel{T}, restrictions::SVARRestrictions,
                                      horizon::Int; n_draws::Int, rng::AbstractRNG,
                                      threaded::Bool) where {T<:AbstractFloat}
    n = nvars(model)
    max_h = _gk_phi_length(restrictions, horizon)
    Phi = ma_coefficients(model.B, n, model.p, max_h)
    L = safe_cholesky(model.Sigma)
    C1 = any(z -> z isa LongRunZeroRestriction, restrictions.zeros) ?
         _C1_from_B(model.B, n, model.p) : nothing
    has_zeros = !isempty(restrictions.zeros)
    if _has_narrative(restrictions)
        _validate_narrative_sample(restrictions, size(model.U, 1))
    end

    seeds = rand(rng, UInt64, n_draws)
    irf_store = fill(T(NaN), n_draws, horizon, n, n)
    accepted = fill(false, n_draws)

    draw_one = function (d::Int)
        local_rng = Random.MersenneTwister(seeds[d])
        try
            Q = if has_zeros
                _draw_Q_with_zero_restrictions(restrictions, Phi, L; rng=local_rng,
                                               B=model.B, C1=C1)
            else
                haar_orthogonal(n, T; rng=local_rng)
            end
            irf_full = _gk_check_Q(model, Q, restrictions, Phi, L, max_h)
            irf_full === nothing && return
            irf_store[d, :, :, :] = irf_full[1:horizon, :, :]
            accepted[d] = true
        catch err
            _is_rejectable_draw_error(err) || (err isa IdentificationError) || rethrow(err)
        end
        nothing
    end

    if threaded && n_draws > 1 && Threads.nthreads() > 1
        Threads.@threads for d in 1:n_draws
            draw_one(d)
        end
    else
        for d in 1:n_draws
            draw_one(d)
        end
    end

    any(accepted) || throw(IdentificationError(
        "Identified set is empty (no admissible Q in $n_draws draws)"))
    lower = fill(T(Inf), horizon, n, n)
    upper = fill(T(-Inf), horizon, n, n)
    @inbounds for d in 1:n_draws
        accepted[d] || continue
        for h in 1:horizon, i in 1:n, j in 1:n
            v = irf_store[d, h, i, j]
            lower[h, i, j] = min(lower[h, i, j], v)
            upper[h, i, j] = max(upper[h, i, j], v)
        end
    end
    (lower, upper)
end

# -----------------------------------------------------------------------------
# NLopt on sphere coordinates (n > 2, or n = 2 with nonlinear restrictions)
# -----------------------------------------------------------------------------

function _normalize_sphere_blocks(z::AbstractVector{T}, dims::Vector{Int}) where {T}
    w = Vector{T}(undef, length(z))
    offset = 0
    for s_j in dims
        zj = view(z, offset+1:offset+s_j)
        nrm = norm(zj)
        nrm < 100 * eps(T) && return nothing
        w[offset+1:offset+s_j] = zj ./ nrm
        offset += s_j
    end
    w
end

# Continuous IRF slacks g(Q) ≤ 0 for NLopt (signed distance to the restriction).
_gk_n_ineq(::SignRestriction) = 1
_gk_n_ineq(::A0SignRestriction) = 1
_gk_n_ineq(::AplusSignRestriction) = 1
_gk_n_ineq(::CumulativeRestriction) = 1
_gk_n_ineq(::MagnitudeBound) = 2
_gk_n_ineq(::ElasticityBound) = 2
_gk_n_ineq(::FEVDShareRestriction) = 2
_gk_n_ineq(::AbstractSVARRestriction) = 0

_gk_sign_slack(val::Float64, sign::Int) = sign > 0 ? -val : val

function _gk_ineq_slack(sr::SignRestriction, k::Int, irf, A0, Aplus, fevd)
    _gk_sign_slack(Float64(irf[sr.horizon + 1, sr.variable, sr.shock]), sr.sign)
end
function _gk_ineq_slack(sr::A0SignRestriction, k::Int, irf, A0, Aplus, fevd)
    A0 === nothing && return 1.0
    _gk_sign_slack(Float64(A0[sr.variable, sr.shock]), sr.sign)
end
function _gk_ineq_slack(sr::AplusSignRestriction, k::Int, irf, A0, Aplus, fevd)
    Aplus === nothing && return 1.0
    _gk_sign_slack(Float64(Aplus[_aplus_row(sr, size(Aplus, 2)), sr.shock]), sr.sign)
end
function _gk_ineq_slack(sr::CumulativeRestriction, k::Int, irf, A0, Aplus, fevd)
    s = 0.0
    @inbounds for h in sr.horizons
        s += Float64(irf[h + 1, sr.variable, sr.shock])
    end
    _gk_sign_slack(s, sr.sign)
end
function _gk_ineq_slack(sr::MagnitudeBound, k::Int, irf, A0, Aplus, fevd)
    val = Float64(irf[sr.horizon + 1, sr.variable, sr.shock])
    k == 1 ? Float64(sr.lower) - val : val - Float64(sr.upper)
end
function _gk_ineq_slack(sr::ElasticityBound, k::Int, irf, A0, Aplus, fevd)
    num = Float64(irf[sr.horizon + 1, sr.numerator_var, sr.shock])
    den = Float64(irf[sr.horizon + 1, sr.denominator_var, sr.shock])
    abs(den) <= eps(Float64) && return 1.0
    e = num / den
    k == 1 ? Float64(sr.lower) - e : e - Float64(sr.upper)
end
function _gk_ineq_slack(sr::FEVDShareRestriction, k::Int, irf, A0, Aplus, fevd)
    fevd === nothing && return 1.0
    share = Float64(fevd[sr.variable, sr.shock, sr.horizon + 1])
    k == 1 ? Float64(sr.lower) - share : share - Float64(sr.upper)
end
_gk_ineq_slack(::AbstractSVARRestriction, k::Int, irf, A0, Aplus, fevd) = 1.0

function _nlopt_identified_set_bounds(model::VARModel{T}, restrictions::SVARRestrictions,
                                      horizon::Int; rng::AbstractRNG, n_starts::Int,
                                      n_rotations::Int) where {T<:AbstractFloat}
    n = nvars(model)
    max_h = _gk_phi_length(restrictions, horizon)
    Phi = ma_coefficients(model.B, n, model.p, max_h)
    L = safe_cholesky(model.Sigma)
    C1 = any(z -> z isa LongRunZeroRestriction, restrictions.zeros) ?
         _C1_from_B(model.B, n, model.p) : nothing

    # Haar envelope first (same generator as :draws) so :optimize is an expansion
    # of the same-seed draw envelope. `_AriasSVARSetup` consumes RNG for W and
    # must not run before this call.
    n_haar = max(n_starts, n_rotations, 32)
    lower, upper = _draws_identified_set_bounds(model, restrictions, horizon;
                                                n_draws=n_haar, rng=rng, threaded=false)

    setup = _AriasSVARSetup(restrictions, n, T; rng=rng)
    dim = setup.dim
    dim < 1 && throw(IdentificationError("Zero restrictions over-constrain Q"))

    function Q_from_z(z::AbstractVector)
        w = _normalize_sphere_blocks(T.(z), setup.sphere_dims)
        w === nothing && return nothing
        try
            return _spheres_to_Q(w, setup, restrictions, Phi, L; B=model.B, C1=C1)
        catch err
            (err isa IdentificationError || _is_rejectable_draw_error(err)) && return nothing
            rethrow(err)
        end
    end

    needs_struct = any(s -> s isa Union{A0SignRestriction, AplusSignRestriction}, restrictions.signs)
    has_fevd = any(s -> s isa FEVDShareRestriction, restrictions.signs)

    # Seed from Haar-feasible accepted `_spheres_to_Q` points, not unconstrained `_draw_w`.
    z0_pool = Vector{Vector{Float64}}()
    n_want = max(n_starts, 1)
    n_try = max(n_haar, n_want * 20)
    for _ in 1:n_try
        length(z0_pool) >= n_want && break
        w = _draw_w(setup; rng=rng)
        Q = Q_from_z(w)
        Q === nothing && continue
        irf = _gk_check_Q(model, Q, restrictions, Phi, L, max_h)
        irf === nothing && continue
        push!(z0_pool, Float64.(w))
        @inbounds for hh in 1:horizon, ii in 1:n, jj in 1:n
            v = T(irf[hh, ii, jj])
            lower[hh, ii, jj] = min(lower[hh, ii, jj], v)
            upper[hh, ii, jj] = max(upper[hh, ii, jj], v)
        end
    end
    isempty(z0_pool) && return (lower, upper)

    ineq_rest = AbstractSVARRestriction[sr for sr in restrictions.signs
                                        if !_is_narrative(sr) && _gk_n_ineq(sr) > 0]

    for h in 1:horizon, i in 1:n, j in 1:n, sense in (1.0, -1.0)
        opt = NLopt.Opt(:LN_COBYLA, dim)
        packed = let last_z = fill(NaN, dim),
                     last_irf = Ref{Any}(nothing),
                     last_A0 = Ref{Any}(nothing),
                     last_Aplus = Ref{Any}(nothing),
                     last_fevd = Ref{Any}(nothing)
            function (z)
                same = length(last_z) == length(z)
                if same
                    @inbounds for t in eachindex(z)
                        z[t] == last_z[t] || (same = false; break)
                    end
                end
                same && return (last_irf[], last_A0[], last_Aplus[], last_fevd[])
                copyto!(last_z, z)
                Q = Q_from_z(z)
                if Q === nothing
                    last_irf[] = last_A0[] = last_Aplus[] = last_fevd[] = nothing
                    return (nothing, nothing, nothing, nothing)
                end
                # Continuous objective: IRF at Q even if signs are slightly violated;
                # inequality slacks restore feasibility.
                irf = _compute_irf_for_Q(model, Q, Phi, L, max_h)
                last_irf[] = irf
                if needs_struct
                    last_A0[], last_Aplus[] = _rf_to_struct(model.B, L, Q)
                else
                    last_A0[] = last_Aplus[] = nothing
                end
                last_fevd[] = has_fevd ? _compute_fevd(irf, n, size(irf, 1))[2] : nothing
                (last_irf[], last_A0[], last_Aplus[], last_fevd[])
            end
        end
        NLopt.min_objective!(opt, (z, g) -> begin
            irf, _, _, _ = packed(z)
            irf === nothing && return 1e20
            Float64(sense * irf[h, i, j])
        end)
        for sr in ineq_rest
            nsl = _gk_n_ineq(sr)
            for k in 1:nsl
                let sr = sr, k = k
                    NLopt.inequality_constraint!(opt, (z, g) -> begin
                        irf, A0, Aplus, fevd = packed(z)
                        irf === nothing && return 1.0
                        _gk_ineq_slack(sr, k, irf, A0, Aplus, fevd)
                    end, 1e-8)
                end
            end
        end
        NLopt.maxeval!(opt, 400)
        NLopt.xtol_rel!(opt, 1e-8)
        NLopt.ftol_rel!(opt, 1e-10)
        for z0 in z0_pool
            try
                (_, zopt, _) = NLopt.optimize(opt, copy(z0))
                irf, A0, Aplus, fevd = packed(zopt)
                irf === nothing && continue
                feasible = true
                for sr in ineq_rest
                    for k in 1:_gk_n_ineq(sr)
                        if _gk_ineq_slack(sr, k, irf, A0, Aplus, fevd) > 1e-6
                            feasible = false
                            break
                        end
                    end
                    feasible || break
                end
                feasible || continue
                v = T(irf[h, i, j])
                lower[h, i, j] = min(lower[h, i, j], v)
                upper[h, i, j] = max(upper[h, i, j], v)
            catch err
                inner = err isa Base.CapturedException ? err.ex : err
                (inner isa IdentificationError || _is_rejectable_draw_error(inner)) && continue
                rethrow(err)
            end
        end
    end
    (lower, upper)
end

# -----------------------------------------------------------------------------
# Robust credible region and informativeness
# -----------------------------------------------------------------------------

"""
Smallest interval `[c_l, c_u]` containing at least `level` of the intervals
`[ℓ_s, u_s]` (Giacomini–Kitagawa 2021 robust credible region).
"""
function _smallest_covering_interval(lows::AbstractVector{T}, highs::AbstractVector{T},
                                     level::T) where {T<:AbstractFloat}
    N = length(lows)
    N == 0 && return (T(NaN), T(NaN))
    k = max(1, ceil(Int, Float64(level) * N))
    k > N && (k = N)
    perm = sortperm(lows)
    best_len = T(Inf)
    cl_best = lows[perm[1]]
    cu_best = maximum(highs)
    @inbounds for i in 1:N
        n_elig = N - i + 1
        n_elig < k && break
        cl = lows[perm[i]]
        eh = Vector{T}(undef, n_elig)
        for t in 1:n_elig
            eh[t] = highs[perm[i + t - 1]]
        end
        sort!(eh)
        cu = eh[k]
        len = cu - cl
        if len < best_len
            best_len = len
            cl_best = cl
            cu_best = cu
        end
    end
    (cl_best, cu_best)
end

function _gk_informativeness(single_lo::Array{T,3}, single_hi::Array{T,3},
                             rob_lo::Array{T,3}, rob_hi::Array{T,3}) where {T<:AbstractFloat}
    num = zero(T)
    den = zero(T)
    n_used = 0
    @inbounds for i in eachindex(single_lo)
        w_r = rob_hi[i] - rob_lo[i]
        w_s = single_hi[i] - single_lo[i]
        (isfinite(w_r) && isfinite(w_s) && w_r > 100 * eps(T)) || continue
        num += w_s
        den += w_r
        n_used += 1
    end
    n_used == 0 && return zero(T)
    κ = one(T) - num / den
    clamp(κ, zero(T), one(T))
end

"""
    identify_robust_bayes(post, restrictions, H; level=0.68, solver=:optimize, n_draws=200, rng)

Giacomini–Kitagawa (2021) robust Bayes for a set-identified SVAR.

Computes identified-set bounds at each posterior draw of `(B, Σ)`, the set of
posterior means, a robust credible region (smallest interval containing `level`
of the draws' sets), the Haar single-prior interval from
[`identify_arias_bayesian`](@ref), the prior-informativeness diagnostic, and
`Π(∅)`.
"""
function identify_robust_bayes(post::BVARPosterior, restrictions::SVARRestrictions, horizon::Int;
                               level::Real=0.68, solver::Symbol=:optimize,
                               n_draws::Int=200, n_rotations::Int=100,
                               rng::AbstractRNG=Random.default_rng(),
                               data::Union{Nothing,AbstractMatrix}=nothing)
    (0 < level < 1) || throw(ArgumentError("level must lie in (0, 1)"))
    horizon >= 1 || throw(ArgumentError("horizon must be ≥ 1"))
    solver === :draws || solver === :optimize ||
        throw(ArgumentError("solver must be :draws or :optimize, got :$solver"))

    use_data = isnothing(data) ? (isempty(post.data) ? nothing : post.data) : data
    p, n = post.p, post.n
    T = eltype(post.Sigma_draws)
    α = T(level)
    b_vecs, sigmas = extract_chain_parameters(post)
    n_samples = size(b_vecs, 1)
    n_samples >= 1 || throw(ArgumentError("posterior contains no draws"))

    lo_store = fill(T(NaN), n_samples, horizon, n, n)
    hi_store = fill(T(NaN), n_samples, horizon, n, n)
    empty = fill(true, n_samples)
    seeds = rand(rng, UInt64, n_samples)
    threaded = solver === :optimize && n == 2 && _gk_linear_in_Q(restrictions)

    process_draw = function (s::Int)
        local_rng = Random.MersenneTwister(seeds[s])
        m = parameters_to_model(b_vecs[s, :], sigmas[s, :], p, n, use_data;
                                varnames=post.varnames)
        try
            lo, hi = identified_set_bounds(m, restrictions, horizon;
                                           solver=solver, n_draws=n_draws,
                                           rng=local_rng, n_rotations=n_rotations,
                                           threaded=false)
            lo_store[s, :, :, :] = lo
            hi_store[s, :, :, :] = hi
            empty[s] = false
        catch err
            if err isa IdentificationError || _is_rejectable_draw_error(err)
                return
            end
            rethrow(err)
        end
        nothing
    end

    if threaded && n_samples > 1 && Threads.nthreads() > 1
        Threads.@threads for s in 1:n_samples
            process_draw(s)
        end
    else
        for s in 1:n_samples
            process_draw(s)
        end
    end

    n_empty = count(empty)
    empty_prob = T(n_empty) / T(n_samples)
    n_ok = n_samples - n_empty

    lower = fill(T(NaN), horizon, n, n)
    upper = fill(T(NaN), horizon, n, n)
    rob_lo = fill(T(NaN), horizon, n, n)
    rob_hi = fill(T(NaN), horizon, n, n)

    if n_ok > 0
        @inbounds for h in 1:horizon, i in 1:n, j in 1:n
            ℓs = Vector{T}(undef, n_ok)
            us = Vector{T}(undef, n_ok)
            t = 0
            for s in 1:n_samples
                empty[s] && continue
                t += 1
                ℓs[t] = lo_store[s, h, i, j]
                us[t] = hi_store[s, h, i, j]
            end
            lower[h, i, j] = mean(ℓs)
            upper[h, i, j] = mean(us)
            cl, cu = _smallest_covering_interval(ℓs, us, α)
            rob_lo[h, i, j] = cl
            rob_hi[h, i, j] = cu
        end
    end

    sp_lo = fill(T(NaN), horizon, n, n)
    sp_hi = fill(T(NaN), horizon, n, n)
    q_lo = (1 - Float64(level)) / 2
    q_hi = 1 - q_lo
    try
        arias = identify_arias_bayesian(post, restrictions, horizon;
                                        data=use_data, n_rotations=n_rotations,
                                        quantiles=[q_lo, 0.5, q_hi],
                                        compute_weights=true, rng=rng)
        nq = size(arias.irf_quantiles, 4)
        sp_lo .= arias.irf_quantiles[:, :, :, 1]
        sp_hi .= arias.irf_quantiles[:, :, :, nq]
    catch err
        err isa IdentificationError || rethrow(err)
    end

    κ = _gk_informativeness(sp_lo, sp_hi, rob_lo, rob_hi)
    RobustBayesResult{T}(lower, upper, rob_lo, rob_hi, sp_lo, sp_hi, κ, empty_prob, α)
end
