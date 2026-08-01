# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Weak-instrument-robust inference for LP-IV (T245 / #344).

External instruments for macro shocks — narrative series, high-frequency surprises — are
routinely weak, which invalidates the horizon-wise 2SLS confidence bands `lp_iv_irf`
reports. Two tools address this:

- [`montiel_olea_pflueger_f`](@ref) — the effective first-stage `F`, the correct weak-
  instrument diagnostic under heteroskedasticity and autocorrelation
- [`lp_iv_ar_band`](@ref) — an Anderson-Rubin confidence set at every horizon, computed
  with HAR covariance whose lag length scales with the horizon, and therefore valid no
  matter how weak the instrument is

References:
- Montiel Olea, J. L. & Pflueger, C. (2013). A Robust Test for Weak Instruments.
  *Journal of Business & Economic Statistics*, 31(3), 358-369.
- Andrews, I., Stock, J. H. & Sun, L. (2019). Weak Instruments in Instrumental Variables
  Regression: Theory and Practice. *Annual Review of Economics*, 11, 727-753.
- Stock, J. H. & Watson, M. W. (2018). Identification and Estimation of Dynamic Causal
  Effects in Macroeconomics Using External Instruments. *Economic Journal*, 128, 917-948.
"""

using LinearAlgebra
using Statistics
using Distributions

# =============================================================================
# Montiel Olea-Pflueger effective F
# =============================================================================

"""
Montiel Olea-Pflueger **simplified** (nuisance-parameter-free) critical values for the
effective `F`, by worst-case relative bias of 2SLS, as tabulated in Andrews, Stock & Sun
(2019, Table 1). These are conservative upper bounds: the exact MOP critical values depend
on the estimated covariance structure and are weakly smaller.
"""
const _MOP_SIMPLIFIED_CV = Dict(0.05 => 37.42, 0.10 => 23.11, 0.20 => 15.06, 0.30 => 12.04)

"""
    MontielOleaPfluegerF{T}

Effective first-stage `F` for LP-IV.

# Fields
- `f_effective::T` — the effective `F` statistic
- `critical_value::T` — MOP simplified critical value at the requested bias target
- `tau::T` — the worst-case relative-bias target (`0.05`, `0.10`, `0.20`, `0.30`)
- `weak::Bool` — `true` when `f_effective < critical_value`
- `n_instruments::Int` — number of excluded instruments `q`
- `bandwidth::Int` — HAC lag length used for the first-stage covariance
- `f_naive::T` — the model's stored first-stage `F` at horizon 0, for comparison
"""
struct MontielOleaPfluegerF{T<:AbstractFloat}
    f_effective::T
    critical_value::T
    tau::T
    weak::Bool
    n_instruments::Int
    bandwidth::Int
    f_naive::T
end

function Base.show(io::IO, r::MontielOleaPfluegerF{T}) where {T}
    data = Any[
        "Effective F"            _fmt(r.f_effective; digits=2);
        "MOP critical value"     _fmt(r.critical_value; digits=2);
        "Worst-case bias target" "$(round(Int, 100 * r.tau))%";
        "Verdict"                r.weak ? "WEAK instruments" : "instruments adequate";
        "Excluded instruments"   r.n_instruments;
        "HAC bandwidth"          r.bandwidth;
        "First-stage F (h=0)"    _fmt(r.f_naive; digits=2)
    ]
    _pretty_table(io, data;
        title = "Montiel Olea-Pflueger Effective F",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    r.weak && println(io, "The effective F is below the critical value: 2SLS bands are " *
                          "unreliable. Use `lp_iv_ar_band(model)` for weak-instrument-" *
                          "robust bands.")
    return nothing
end

"""
    _lp_iv_horizon_pieces(model, h) -> (Y_h, endog, Z, W)

Rebuild the horizon-`h` estimation arrays — response, endogenous shock, instruments and
controls — exactly as `estimate_lp_iv` constructed them, so the weak-IV routines condition
on the same sample and the same controls the point estimates did.
"""
function _lp_iv_horizon_pieces(model::LPIVModel{T}, h::Int) where {T<:AbstractFloat}
    T_obs, nvar = size(model.Y)
    t_start, t_end = compute_horizon_bounds(T_obs, h, model.lags)
    Y_h = build_response_matrix(model.Y, h, t_start, t_end, model.response_vars)
    endog = T[model.Y[t, model.shock_var] for t in t_start:t_end]
    Z = Matrix{T}(model.instruments[t_start:t_end, :])
    W = ones(T, t_end - t_start + 1, 1 + nvar * model.lags)
    build_control_columns!(W, model.Y, t_start, t_end, model.lags, 2)
    return Y_h, endog, Z, W
end

"""
    montiel_olea_pflueger_f(model::LPIVModel; tau=0.10, bandwidth=0) -> MontielOleaPfluegerF

Montiel Olea & Pflueger (2013) **effective first-stage F** for an LP-IV model.

The classical first-stage `F` is only a valid weak-instrument diagnostic under
homoskedastic, serially uncorrelated errors — precisely what macro data violate. The
effective `F` replaces the homoskedastic scale with a HAR one:

```math
F_{\\text{eff}} = \\frac{\\tilde Y_2' P_{\\tilde Z} \\tilde Y_2}
                        {\\operatorname{tr}\\!\\big(\\hat V_{\\hat\\Pi}\\,\\tilde Z'\\tilde Z\\big)}
```

where:
- ``\\tilde Z = M_W Z`` are the excluded instruments with the LP controls partialled out
- ``\\tilde Y_2 = M_W x`` is the residualized endogenous shock
- ``\\hat V_{\\hat\\Pi}`` is the Newey-West sandwich covariance of the first-stage coefficients

Under homoskedasticity ``\\hat V_{\\hat\\Pi} = \\hat\\sigma_v^2 (\\tilde Z'\\tilde Z)^{-1}``, the
denominator collapses to ``q\\hat\\sigma_v^2`` and ``F_{\\text{eff}}`` **is** the classical
first-stage `F` — the reduction Montiel Olea and Pflueger establish.

The LP first stage is the same regression at every horizon (only the sample window
shrinks), so the effective `F` is computed once, at horizon 0.

# Keywords
- `tau::Real=0.10` — worst-case relative bias of 2SLS to tolerate (`0.05`, `0.10`, `0.20`, `0.30`)
- `bandwidth::Int=0` — HAC lag length; `0` selects it from the data

# Returns
[`MontielOleaPfluegerF`](@ref).
"""
function montiel_olea_pflueger_f(model::LPIVModel{T}; tau::Real=0.10,
                                 bandwidth::Int=0) where {T<:AbstractFloat}
    haskey(_MOP_SIMPLIFIED_CV, Float64(tau)) || throw(ArgumentError(
        "tau must be one of $(sort(collect(keys(_MOP_SIMPLIFIED_CV)))), got $tau"))

    _, endog, Z, W = _lp_iv_horizon_pieces(model, 0)
    Zt = _partial_out(Z, W)
    xt = _partial_out(endog, W)
    q = size(Z, 2)

    ZtZ = Zt' * Zt
    pi_hat = robust_inv(Symmetric(Matrix{T}(ZtZ))) * (Zt' * xt)
    resid = xt .- Zt * pi_hat

    bw = bandwidth > 0 ? bandwidth : optimal_bandwidth_nw(resid)
    V_pi = newey_west(Matrix{T}(Zt), resid; bandwidth=bw)

    signal = dot(xt, Zt * pi_hat)                       # x̃'P_Z̃ x̃
    noise = tr(Matrix{T}(V_pi) * Matrix{T}(ZtZ))
    f_eff = noise > 0 ? T(signal / noise) : T(NaN)

    crit = T(_MOP_SIMPLIFIED_CV[Float64(tau)])
    return MontielOleaPfluegerF{T}(f_eff, crit, T(tau), isfinite(f_eff) && f_eff < crit,
                                   q, bw, model.first_stage_F[1])
end

# =============================================================================
# Horizon-wise Anderson-Rubin bands
# =============================================================================

"""
    LPIVARBand{T}

Anderson-Rubin confidence sets for the LP-IV impulse response, one per horizon and
response variable.

Because an AR set need not be an interval, the full component list is kept in `sets`;
`lower`/`upper` give the plottable envelope, carrying `±Inf` where a set is unbounded and
`NaN` where it is empty.

# Fields
- `lower::Matrix{T}` / `upper::Matrix{T}` — `(H+1) × n_resp` envelope of the AR set
- `sets::Matrix{Vector{Tuple{T,T}}}` — the connected components at each horizon/response
- `bounded::Matrix{Bool}` / `is_empty::Matrix{Bool}` — set shape flags
- `wald_lower::Matrix{T}` / `wald_upper::Matrix{T}` — the 2SLS Wald band, for comparison
- `point::Matrix{T}` — the LP-IV point estimates
- `bandwidths::Matrix{Int}` — HAC lag length used at each horizon/response (scales with `h`)
- `horizon::Int`, `level::T`, `critical_value::T`, `df1::Int`
- `response_names::Vector{String}`, `shock_name::String`
"""
struct LPIVARBand{T<:AbstractFloat}
    lower::Matrix{T}
    upper::Matrix{T}
    sets::Matrix{Vector{Tuple{T,T}}}
    bounded::Matrix{Bool}
    is_empty::Matrix{Bool}
    wald_lower::Matrix{T}
    wald_upper::Matrix{T}
    point::Matrix{T}
    bandwidths::Matrix{Int}
    horizon::Int
    level::T
    critical_value::T
    df1::Int
    response_names::Vector{String}
    shock_name::String
end

function Base.show(io::IO, b::LPIVARBand{T}) where {T}
    pct = round(Int, 100 * b.level)
    n_unb = count(!, b.bounded)
    spec = Any[
        "Shock"                 b.shock_name;
        "Horizon"               b.horizon;
        "Responses"             length(b.response_names);
        "Level"                 "$(pct)%";
        "Excluded instruments"  b.df1;
        "Critical value"        _fmt(b.critical_value);
        "Unbounded cells"       "$(n_unb) / $(length(b.bounded))"
    ]
    _pretty_table(io, spec;
        title = "LP-IV Anderson-Rubin Bands (weak-instrument robust)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    for (j, name) in enumerate(b.response_names)
        data = Matrix{Any}(undef, b.horizon + 1, 6)
        for h in 0:b.horizon
            i = h + 1
            data[i, 1] = h
            data[i, 2] = _fmt(b.point[i, j])
            data[i, 3] = b.is_empty[i, j] ? "∅" : (isfinite(b.lower[i, j]) ? _fmt(b.lower[i, j]) : "-∞")
            data[i, 4] = b.is_empty[i, j] ? "∅" : (isfinite(b.upper[i, j]) ? _fmt(b.upper[i, j]) : "∞")
            data[i, 5] = _fmt(b.wald_lower[i, j])
            data[i, 6] = _fmt(b.wald_upper[i, j])
        end
        _pretty_table(io, data;
            title = "$(name)",
            column_labels = ["h", "IRF", "AR lo", "AR hi", "Wald lo", "Wald hi"],
            alignment = [:r, :r, :r, :r, :r, :r],
        )
    end
    n_unb > 0 && println(io, "Unbounded AR cells indicate horizons at which the " *
                             "instrument is too weak to bound the response; the Wald band " *
                             "there is over-confident.")
    return nothing
end

"""
    lp_iv_ar_band(model::LPIVModel; level=0.95, n_grid=401, span=20, bandwidth=0,
                  responses=nothing) -> LPIVARBand

Anderson-Rubin confidence sets for the LP-IV impulse response, horizon by horizon.

At each horizon `h` and response `j`, the AR test of `H₀: θ_{h,j} = θ₀` regresses
`Y_{t+h,j} − θ₀ x_t` on the controls and instruments and tests the instruments' joint
irrelevance. Because that restriction holds under `H₀` whatever the first stage looks like,
the resulting set has correct coverage no matter how weak the instrument is.

The covariance uses **Newey-West with a lag length that scales with the horizon** —
`max(data-driven bandwidth, h+1)`, the same rule `estimate_lp_iv` applies to its own
standard errors — because the horizon-`h` LP residuals are MA(`h`) correlated by
construction. The data-driven part is taken from each cell's own LP-IV residuals, and the
bandwidth actually used is returned in `bandwidths` (`(H+1) × n_resp`).

Sets that reach the edge of the search range are reported as unbounded rather than
truncated; the search range spans `θ̂ ± span·se` around the 2SLS estimate at each cell.

# Keywords
- `level::Real=0.95` — nominal coverage
- `n_grid::Int=401` — grid points per cell
- `span::Real=20` — half-width of the search range, in 2SLS standard errors
- `bandwidth::Int=0` — fixed HAC lag length; `0` uses `max(auto, h+1)`
- `responses::Union{Nothing,Vector{Int}}` — subset of `model.response_vars` positions to compute

# Returns
[`LPIVARBand`](@ref).
"""
function lp_iv_ar_band(model::LPIVModel{T}; level::Real=0.95, n_grid::Int=401,
                       span::Real=20, bandwidth::Int=0,
                       responses::Union{Nothing,Vector{Int}}=nothing) where {T<:AbstractFloat}
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1)"))
    n_grid >= 5 || throw(ArgumentError("n_grid must be at least 5"))
    H = model.horizon
    n_resp_all = length(model.response_vars)
    resp_idx = responses === nothing ? collect(1:n_resp_all) : responses
    all(1 .<= resp_idx .<= n_resp_all) || throw(ArgumentError(
        "responses must index into 1:$n_resp_all"))
    nr = length(resp_idx)

    wald = lp_iv_irf(model; conf_level=level)
    q = n_instruments(model)
    crit = T(quantile(Chisq(q), level) / q)
    z = T(quantile(Normal(), (1 + level) / 2))

    lower = Matrix{T}(undef, H + 1, nr)
    upper = Matrix{T}(undef, H + 1, nr)
    sets = Matrix{Vector{Tuple{T,T}}}(undef, H + 1, nr)
    bounded = Matrix{Bool}(undef, H + 1, nr)
    empty_f = Matrix{Bool}(undef, H + 1, nr)
    wlo = Matrix{T}(undef, H + 1, nr)
    whi = Matrix{T}(undef, H + 1, nr)
    point = Matrix{T}(undef, H + 1, nr)
    bws = Matrix{Int}(undef, H + 1, nr)

    for h in 0:H
        i = h + 1
        Y_h, endog, Z, W = _lp_iv_horizon_pieces(model, h)
        Zb, _, qh = _ar_excluded_basis(hcat(endog, W), hcat(Z, W), [1])
        Xen = reshape(endog, :, 1)
        resid_h = model.residuals[i]

        for (jj, j) in enumerate(resp_idx)
            yv = Vector{T}(view(Y_h, :, j))

            # Horizon-scaled HAC lag, chosen PER RESPONSE from that equation's own LP-IV
            # residuals: the horizon-h residual is MA(h) by construction, so the h+1 floor
            # applies, and using the cell's own residuals makes a subset call agree
            # exactly with the corresponding column of a full call.
            bw_auto = optimal_bandwidth_nw(Vector{T}(view(resid_h, :, j)))
            bw = bandwidth > 0 ? bandwidth : max(bw_auto, h + 1)
            bws[i, jj] = bw
            theta = T(wald.values[i, j])
            se = T(wald.se[i, j])
            point[i, jj] = theta
            wlo[i, jj] = theta - z * se
            whi[i, jj] = theta + z * se

            arstat(t0) = _ar_stat(yv, Xen, W, Zb, T[t0], :hac, size(Z, 2) + size(W, 2);
                                  bandwidth=bw)[1]

            half = isfinite(se) && se > 0 ? T(span) * se : T(span)
            gvec = collect(range(theta - half, theta + half; length=n_grid))
            inside = [arstat(g) <= crit for g in gvec]

            if !any(inside)
                sets[i, jj] = Tuple{T,T}[]
                empty_f[i, jj] = true
                bounded[i, jj] = true
                lower[i, jj] = T(NaN); upper[i, jj] = T(NaN)
                continue
            end
            empty_f[i, jj] = false

            function refine(a, b)
                for _ in 1:35
                    mid = (a + b) / 2
                    arstat(mid) <= crit ? (a = mid) : (b = mid)
                end
                return a
            end

            comps = Tuple{T,T}[]
            p = 1
            N = length(gvec)
            while p <= N
                if inside[p]
                    f_in = p
                    while p < N && inside[p+1]
                        p += 1
                    end
                    l_in = p
                    lo = f_in == 1 ? T(-Inf) : refine(gvec[f_in], gvec[f_in-1])
                    hi = l_in == N ? T(Inf) : refine(gvec[l_in], gvec[l_in+1])
                    push!(comps, (lo, hi))
                end
                p += 1
            end
            sets[i, jj] = comps
            bounded[i, jj] = all(t -> isfinite(t[1]) && isfinite(t[2]), comps)
            lower[i, jj] = minimum(t[1] for t in comps)
            upper[i, jj] = maximum(t[2] for t in comps)
        end
    end

    return LPIVARBand{T}(lower, upper, sets, bounded, empty_f, wlo, whi, point, bws,
                         H, T(level), crit, q,
                         String[wald.response_vars[j] for j in resp_idx],
                         wald.shock_var)
end
