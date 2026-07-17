# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Panel cointegrating regression — panel FMOLS / DOLS (EV-22, #430)
#
# A cross-unit AGGREGATION layer on top of EV-10's single-equation cointegrating
# regression (`estimate_cointreg`). Each unit i is estimated by `estimate_cointreg`
# (FMOLS/DOLS), and only the panel aggregation is added here:
#
#   • group-mean (between-dimension, Pedroni 2001 / Mark–Sul 2003):
#         β̄ = N⁻¹ Σᵢ β̂ᵢ,   reported t = N^{-1/2} Σᵢ tᵢ   (tᵢ = β̂ᵢ/se(β̂ᵢ))
#   • pooled  (within-dimension):
#         – FMOLS (Pedroni 2000): pool L̂⁻²_{11i}-weighted corrected moments into
#           one common β with Var(β̂) = (Σᵢ L̂⁻²_{11i} S_{xx,i})⁻¹.
#         – DOLS (Kao–Chiang 2000): stacked within-demeaned lead/lag-augmented
#           regression with unit fixed effects and unit-specific short-run dynamics.
#
# References
#   Pedroni, P. (2000). Fully Modified OLS for Heterogeneous Cointegrated Panels.
#     Advances in Econometrics 15, 93–130.
#   Pedroni, P. (2001). Purchasing Power Parity Tests in Cointegrated Panels.
#     Review of Economics and Statistics 83(4), 727–731.
#   Kao, C. & Chiang, M.-H. (2000). On the Estimation and Inference of a
#     Cointegrated Regression in Panel Data. Advances in Econometrics 15, 179–222.
#   Mark, N. C. & Sul, D. (2003). Cointegration Vector Estimation by Panel DOLS
#     and Long-Run Money Demand. OBES 65(5), 655–680.
# =============================================================================

"""
    _cointreg_xnames_named(xnames, k) -> Vector{String}

Slope names: the user-supplied `xnames` if it has length `k`, else the generic
`_cointreg_xnames(k)` fallback (`"x"`/`"x1"…"xk"`).
"""
_cointreg_xnames_named(xnames::Vector{String}, k::Int) =
    length(xnames) == k ? copy(xnames) : _cointreg_xnames(k)

"""
    _xtcointreg_units_panel(pd, y, xs) -> (units, xnames)

Extract per-unit `(yᵢ::Vector, Xᵢ::Matrix)` from a `PanelData`, ordered by group id with
observations sorted by time within each group. Unbalanced panels are allowed (each unit may
have its own `Tᵢ`). Also returns the regressor names.
"""
function _xtcointreg_units_panel(pd::PanelData{T}, y::Union{Symbol,String},
                                 xs::Tuple) where {T<:AbstractFloat}
    isempty(xs) && throw(ArgumentError("At least one regressor is required"))
    yc = _panel_varindex(pd, y)
    xcs = Int[_panel_varindex(pd, x) for x in xs]
    N = pd.n_groups
    units = Vector{Tuple{Vector{T},Matrix{T}}}(undef, N)
    for g in 1:N
        gd = group_data(pd, g)
        ord = sortperm(gd.time_index)
        yg = collect(T, gd.data[ord, yc])
        Xg = Matrix{T}(gd.data[ord, xcs])
        units[g] = (yg, Xg)
    end
    xnames = String[pd.varnames[c] for c in xcs]
    return units, xnames
end

"""
    _xtcointreg_units_long(y, X, id, time) -> (units, order)

Group a long-format `(y, X)` by `id` (units in first-appearance order) with observations
sorted by `time` within each unit. Returns per-unit `(yᵢ, Xᵢ)`.
"""
function _xtcointreg_units_long(y::AbstractVector, X::AbstractMatrix,
                                id::AbstractVector, time::AbstractVector)
    T = float(eltype(y))
    n = length(y)
    (size(X, 1) == n == length(id) == length(time)) ||
        throw(DimensionMismatch("y, X rows, id, and time must have equal length"))
    order = Vector{eltype(id)}()
    idx = Dict{eltype(id),Vector{Int}}()
    for i in 1:n
        g = id[i]
        if !haskey(idx, g)
            idx[g] = Int[]
            push!(order, g)
        end
        push!(idx[g], i)
    end
    units = Vector{Tuple{Vector{T},Matrix{T}}}(undef, length(order))
    for (u, g) in enumerate(order)
        rows = idx[g]
        ord = sortperm(time[rows])
        rr = rows[ord]
        units[u] = (collect(T, @view y[rr]), Matrix{T}(@view X[rr, :]))
    end
    return units, order
end

# =============================================================================
# Public API
# =============================================================================

"""
    estimate_xtcointreg(pd::PanelData, y, xs...; method=:fmols, pooling=:group,
                        trend=:const, kernel=:bartlett, bandwidth=:andrews,
                        leads=:auto, lags=:auto, ic=:aic, dols_se=:lrv)
        -> PanelCointRegModel

Panel cointegrating regression of `y` on the `I(1)` regressors `xs` across the `N` units of
a [`PanelData`](@ref). Each unit is estimated by the single-equation
[`estimate_cointreg`](@ref) and the results are aggregated into a
[`PanelCointRegModel`](@ref).

A long-format method is also available:

    estimate_xtcointreg(y::AbstractVector, X::AbstractVecOrMat,
                        id::AbstractVector, time::AbstractVector; kwargs...)

# Keywords
- `method`: `:fmols` (Phillips–Hansen fully-modified OLS, default) or `:dols` (dynamic OLS).
- `pooling`: `:group` (group-mean / between-dimension, Pedroni 2001, default) or `:pooled`
  (within-dimension — Pedroni 2000 FMOLS / Kao–Chiang 2000 DOLS).
- `trend`: per-unit deterministics — `:none`, `:const` (default), or `:linear`.
- `kernel`, `bandwidth`: HAC kernel / bandwidth forwarded to the per-unit long-run
  covariances (see [`estimate_cointreg`](@ref)).
- `leads`, `lags`, `ic`: DOLS lead/lag order (per unit; `:auto` selects by `ic`).
- `dols_se`: per-unit DOLS standard errors (`:lrv` default or `:robust`).

# Group-mean vs pooled
`:group` reports `β̄ = N⁻¹ Σᵢ β̂ᵢ` over the full per-unit coefficient vector
(`[deterministics; slopes]`) with Pedroni's between-dimension `t = N^{-1/2} Σᵢ tᵢ`.
`:pooled` reports the common **slopes only**, with fixed effects (and, for DOLS,
unit-specific short-run dynamics) partialled out.

# Examples
```julia
pd = xtset(df, :country, :year)
m  = estimate_xtcointreg(pd, :lc, :ly; method=:fmols, pooling=:group)
report(m)
coef(m)
```

# References
- Pedroni (2000), *Adv. in Econometrics* 15; Pedroni (2001), *REStat* 83(4).
- Kao & Chiang (2000), *Adv. in Econometrics* 15; Mark & Sul (2003), *OBES* 65(5).
"""
function estimate_xtcointreg(pd::PanelData, y::Union{Symbol,String},
                             xs::Union{Symbol,String}...;
                             method::Symbol=:fmols, pooling::Symbol=:group,
                             trend::Symbol=:const, kernel::Symbol=:bartlett,
                             bandwidth=:andrews, leads=:auto, lags=:auto,
                             ic::Symbol=:aic, dols_se::Symbol=:lrv)
    units, xnames = _xtcointreg_units_panel(pd, y, xs)
    return _xtcointreg_core(units, xnames; method=method, pooling=pooling, trend=trend,
                            kernel=kernel, bandwidth=bandwidth, leads=leads, lags=lags,
                            ic=ic, dols_se=dols_se)
end

function estimate_xtcointreg(y::AbstractVector, X::AbstractVecOrMat,
                             id::AbstractVector, time::AbstractVector;
                             method::Symbol=:fmols, pooling::Symbol=:group,
                             trend::Symbol=:const, kernel::Symbol=:bartlett,
                             bandwidth=:andrews, leads=:auto, lags=:auto,
                             ic::Symbol=:aic, dols_se::Symbol=:lrv,
                             xnames::Union{Nothing,Vector{String}}=nothing)
    Xm = X isa AbstractVector ? reshape(collect(X), :, 1) : Matrix(X)
    units, _ = _xtcointreg_units_long(y, Xm, id, time)
    names = xnames === nothing ? _cointreg_xnames(size(Xm, 2)) : xnames
    return _xtcointreg_core(units, names; method=method, pooling=pooling, trend=trend,
                            kernel=kernel, bandwidth=bandwidth, leads=leads, lags=lags,
                            ic=ic, dols_se=dols_se)
end

# =============================================================================
# Core dispatcher
# =============================================================================

function _xtcointreg_core(units::Vector{Tuple{Vector{T},Matrix{T}}}, xnames::Vector{String};
                          method::Symbol, pooling::Symbol, trend::Symbol, kernel::Symbol,
                          bandwidth, leads, lags, ic::Symbol,
                          dols_se::Symbol) where {T<:AbstractFloat}
    method ∈ (:fmols, :dols) ||
        throw(ArgumentError("method must be :fmols or :dols; got :$method"))
    pooling ∈ (:group, :pooled) ||
        throw(ArgumentError("pooling must be :group or :pooled; got :$pooling"))
    trend ∈ (:none, :const, :linear) ||
        throw(ArgumentError("trend must be :none, :const, or :linear; got :$trend"))
    N = length(units)
    N ≥ 1 || throw(ArgumentError("need at least one unit"))
    k = size(units[1][2], 2)
    for (i, (yi, Xi)) in enumerate(units)
        size(Xi, 2) == k ||
            throw(DimensionMismatch("unit $i has $(size(Xi,2)) regressors; expected $k"))
        length(yi) == size(Xi, 1) ||
            throw(DimensionMismatch("unit $i: length(y)=$(length(yi)) ≠ size(X,1)=$(size(Xi,1))"))
    end
    d = trend === :none ? 0 : trend === :const ? 1 : 2

    # Per-unit single-equation fits (reused everywhere) — EV-10 estimate_cointreg.
    unit_models = CointRegModel{T}[
        estimate_cointreg(yi, Xi; method=method, trend=trend, kernel=kernel,
                          bandwidth=bandwidth, leads=leads, lags=lags, ic=ic,
                          dols_se=dols_se) for (yi, Xi) in units
    ]
    T_i = Int[length(yi) for (yi, _) in units]
    balanced = all(==(first(T_i)), T_i)

    if pooling === :group
        return _xtcointreg_group(unit_models, method, trend, kernel, xnames,
                                 k, d, T_i, balanced)
    elseif method === :fmols
        return _xtcointreg_pooled_fmols(units, unit_models, trend, kernel, bandwidth,
                                        xnames, k, d, T_i, balanced)
    else
        return _xtcointreg_pooled_dols(units, unit_models, trend, kernel, bandwidth,
                                       leads, lags, ic, xnames, k, d, T_i, balanced)
    end
end

# =============================================================================
# Group-mean (between-dimension) — Pedroni (2001) / Mark–Sul (2003)
# =============================================================================

function _xtcointreg_group(unit_models::Vector{CointRegModel{T}}, method::Symbol,
                           trend::Symbol, kernel::Symbol, xnames::Vector{String},
                           k::Int, d::Int, T_i::Vector{Int}, balanced::Bool) where {T<:AbstractFloat}
    N = length(unit_models)
    p = length(unit_models[1].coef)              # d + k
    C = Matrix{T}(undef, p, N)                    # per-unit coef vectors
    Tm = Matrix{T}(undef, p, N)                   # per-unit t-ratios
    for i in 1:N
        C[:, i] = unit_models[i].coef
        Tm[:, i] = unit_models[i].coef ./ stderror(unit_models[i])
    end

    coef = vec(sum(C; dims=2)) ./ T(N)            # β̄ = N⁻¹ Σ β̂ᵢ  (exact arithmetic mean)
    # Pedroni between-dimension group-mean t-statistic: t = N^{-1/2} Σᵢ tᵢ.
    tstats = vec(sum(Tm; dims=2)) ./ sqrt(T(N))
    # Back out a display standard error consistent with the reported t (coef / t). Where the
    # signs of β̄ and t disagree (degenerate), keep a positive SE; the authoritative statistic
    # is `tstats`, exposed directly.
    se = similar(coef)
    @inbounds for j in eachindex(coef)
        s = tstats[j] == 0 ? T(Inf) : abs(coef[j] / tstats[j])
        se[j] = (isfinite(s) && s > 0) ? s : T(Inf)
    end
    pvalues = T(2) .* ccdf.(Normal(), abs.(tstats))
    vcov = Matrix{T}(Diagonal(se .^ 2))

    dnames = _cointreg_deter(1, trend, T)[2]
    varnames = vcat(dnames, _cointreg_xnames_named(xnames, k))
    unit_coefs = C[(d + 1):(d + k), :]            # slopes only, for diagnostics

    return PanelCointRegModel{T}(method, :group, trend, kernel, coef, vcov, se, tstats,
                                 pvalues, varnames, Matrix{T}(unit_coefs), unit_models,
                                 N, T_i, sum(T_i), k, d, balanced)
end

# =============================================================================
# Pooled FMOLS (within-dimension) — Pedroni (2000)
# =============================================================================

"""
    _fmols_pooled_pieces(y, X, m, trend, kernel) -> (Sxx, Sxyp, w)

Per-unit within-demeaned FMOLS moment contributions, reusing the stacked `(u, Δx)` long-run
covariance already stored on the per-unit fit `m` (EV-10). Returns the fixed-effect-removed
regressor cross-product `S_{xx} = X'M_D X`, the endogeneity/serial-correlation-corrected
cross-moment `S_{xy⁺} = X'M_D y⁺ − T γ̂`, and the Pedroni weight `w = L̂⁻²_{11} = ω̂⁻¹_{u·Δx}`.
"""
function _fmols_pooled_pieces(y::Vector{T}, X::Matrix{T}, m::CointRegModel{T},
                              trend::Symbol) where {T<:AbstractFloat}
    n, k = size(X)
    D, _ = _cointreg_deter(n, trend, T)
    d = size(D, 2)
    Ω = m.Omega
    Δcr = Matrix{T}(m.Lambda')                    # one-sided Δ = Λ' (matches _estimate_fmols)
    vidx = 2:(k + 1)
    Ω_vu = reshape(Ω[vidx, 1], k, 1)
    Ω_vv = Ω[vidx, vidx]
    Δ_vu = reshape(Δcr[vidx, 1], k, 1)
    Δ_vv = Δcr[vidx, vidx]
    Ωvv_inv_vu = robust_inv(Ω_vv) * Ω_vu          # k×1
    Δ_vuplus = Δ_vu .- Δ_vv * Ωvv_inv_vu          # γ̂  (k×1)

    Xdelta = diff(X; dims=1)                       # (n-1)×k
    y_plus = y[2:n] .- vec(Xdelta * Ωvv_inv_vu)    # endogeneity-corrected (aligned t=2:n)
    Xfm = X[2:n, :]
    if d == 0
        MX = Xfm
        My = y_plus
    else
        Dfm = D[2:n, :]
        P = Dfm * robust_inv(Symmetric(Dfm' * Dfm)) * Dfm'
        MX = Xfm .- P * Xfm                        # M_D X
        My = y_plus .- P * y_plus                  # M_D y⁺
    end
    Sxx = Matrix{T}(MX' * MX)
    Sxyp = vec(MX' * My) .- T(n) .* vec(Δ_vuplus)  # serial-correlation correction (n = m.nobs)
    w = one(T) / m.omega_uv
    return Sxx, Sxyp, w
end

function _xtcointreg_pooled_fmols(units::Vector{Tuple{Vector{T},Matrix{T}}},
                                  unit_models::Vector{CointRegModel{T}}, trend::Symbol,
                                  kernel::Symbol, bandwidth, xnames::Vector{String},
                                  k::Int, d::Int, T_i::Vector{Int},
                                  balanced::Bool) where {T<:AbstractFloat}
    N = length(units)
    A = zeros(T, k, k)                             # Σ wᵢ S_xx,i
    b = zeros(T, k)                                # Σ wᵢ S_xy⁺,i
    unit_slopes = Matrix{T}(undef, k, N)
    for i in 1:N
        yi, Xi = units[i]
        Sxx, Sxyp, w = _fmols_pooled_pieces(yi, Xi, unit_models[i], trend)
        A .+= w .* Sxx
        b .+= w .* Sxyp
        unit_slopes[:, i] = unit_models[i].coef[(d + 1):(d + k)]
    end
    Ainv = Matrix{T}(robust_inv(Symmetric(A)))
    coef = Ainv * b
    vcov = Ainv                                    # Var(β̂) = (Σ wᵢ S_xx,i)⁻¹
    se = sqrt.(max.(diag(vcov), zero(T)))
    tstats = coef ./ se
    pvalues = T(2) .* ccdf.(Normal(), abs.(tstats))
    varnames = _cointreg_xnames_named(xnames, k)

    return PanelCointRegModel{T}(:fmols, :pooled, trend, kernel, coef, vcov, se, tstats,
                                 pvalues, varnames, unit_slopes, unit_models,
                                 N, T_i, sum(T_i), k, d, balanced)
end

# =============================================================================
# Pooled DOLS (within-dimension) — Kao–Chiang (2000) / Mark–Sul (2003)
# =============================================================================

function _xtcointreg_pooled_dols(units::Vector{Tuple{Vector{T},Matrix{T}}},
                                 unit_models::Vector{CointRegModel{T}}, trend::Symbol,
                                 kernel::Symbol, bandwidth, leads, lags, ic::Symbol,
                                 xnames::Vector{String}, k::Int, d::Int,
                                 T_i::Vector{Int}, balanced::Bool) where {T<:AbstractFloat}
    N = length(units)
    Xc_blocks = Matrix{T}[]                         # common-slope level regressors per unit
    yt_blocks = Vector{T}[]
    nuis_blocks = Matrix{T}[]                       # unit-specific [D, lead/lag(Δx)]
    unit_slopes = Matrix{T}(undef, k, N)
    for i in 1:N
        yi, Xi = units[i]
        ni = length(yi)
        D, _ = _cointreg_deter(ni, trend, T)
        Z = hcat(D, Xi)
        Xdelta = diff(Xi; dims=1)
        # Lead/lag order — same selection path as EV-10 _estimate_dols (per unit).
        nlag = lags === :auto ? -1 : Int(lags)
        nlead = leads === :auto ? -1 : Int(leads)
        if nlag < 0 || nlead < 0
            snlag, snlead = _dols_select_leadlag(yi, Z, Xdelta, k, ic)
            nlag < 0 && (nlag = snlag)
            nlead < 0 && (nlead = snlead)
        end
        (nlag ≥ 0 && nlead ≥ 0) || throw(ArgumentError("leads/lags must be ≥ 0"))
        Xfull, yt = _dols_design(yi, Z, Xdelta, nlag, nlead)  # cols: [D(d) | X(k) | leadlag]
        Dtrunc = Xfull[:, 1:d]
        Xtrunc = Xfull[:, (d + 1):(d + k)]          # shared β block
        LL = Xfull[:, (d + k + 1):end]              # unit-specific short-run dynamics
        push!(Xc_blocks, Xtrunc)
        push!(yt_blocks, yt)
        push!(nuis_blocks, hcat(Dtrunc, LL))
        unit_slopes[:, i] = unit_models[i].coef[(d + 1):(d + k)]
    end

    Xc = reduce(vcat, Xc_blocks)                    # (Σmᵢ)×k common slope columns
    yt = reduce(vcat, yt_blocks)
    total = length(yt)
    ncols_nuis = sum(size(bl, 2) for bl in nuis_blocks)
    Nz = zeros(T, total, ncols_nuis)               # block-diagonal nuisance (unit FE + dynamics)
    roff = 0
    coff = 0
    row_ranges = UnitRange{Int}[]
    for bl in nuis_blocks
        r = size(bl, 1)
        c = size(bl, 2)
        Nz[(roff + 1):(roff + r), (coff + 1):(coff + c)] = bl
        push!(row_ranges, (roff + 1):(roff + r))
        roff += r
        coff += c
    end
    Xdesign = hcat(Xc, Nz)
    XtX = Symmetric(Xdesign' * Xdesign)
    XtXinv = Matrix{T}(robust_inv(XtX))
    theta = XtXinv * (Xdesign' * yt)
    coef = theta[1:k]
    resid = yt .- Xdesign * theta

    # Kao–Chiang long-run-variance-corrected covariance: average per-unit residual LRV
    # times the β-block of (X'X)⁻¹. For N=1 this reduces to EV-10's DOLS slope covariance.
    omega_bar = zero(T)
    for (i, rr) in enumerate(row_ranges)
        seg = resid[rr]
        bw = bandwidth isa Symbol ?
            T(_resolve_bandwidth(reshape(seg, :, 1), _lrv_kernel(kernel), bandwidth)) :
            T(bandwidth)
        omega_bar += lrvar(seg; kernel=kernel, bandwidth=bw, demean=false)
    end
    omega_bar /= T(N)
    vcov = Matrix{T}(omega_bar .* XtXinv[1:k, 1:k])
    se = sqrt.(max.(diag(vcov), zero(T)))
    tstats = coef ./ se
    pvalues = T(2) .* ccdf.(Normal(), abs.(tstats))
    varnames = _cointreg_xnames_named(xnames, k)

    return PanelCointRegModel{T}(:dols, :pooled, trend, kernel, coef, vcov, se, tstats,
                                 pvalues, varnames, unit_slopes, unit_models,
                                 N, T_i, sum(T_i), k, d, balanced)
end
