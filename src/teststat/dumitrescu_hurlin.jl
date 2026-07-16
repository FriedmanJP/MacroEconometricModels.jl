# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Dumitrescu-Hurlin (2012) heterogeneous-panel Granger non-causality test.

For each cross-sectional unit `i`, regress `y_it` on an intercept, `p` lags of
`y_i`, and `p` lags of `x_i`, and compute the individual Granger-Wald statistic
`W_i` for `H0: all p coefficients on lagged x_i are zero`. The `W_i` are averaged
to `W̄ = N⁻¹ Σ W_i` and standardized two ways:

- **Asymptotic** (T→∞ then N→∞): `Z̄ = √(N/(2p)) (W̄ − p) → N(0,1)`.
- **Small-T (semi-asymptotic)** `Z̃ = √N (W̄ − E[W_i]) / √Var[W_i] → N(0,1)`, using
  the exact finite-`T` moments (Dumitrescu-Hurlin 2012, eqs. 26-27)

      E[W_i]   = p (T−2p−1) / (T−2p−3)
      Var[W_i] = 2p (T−2p−1)² (T−p−3) / [ (T−2p−3)² (T−2p−5) ]

  averaged per unit with its own `T_i` for unbalanced panels. Both statistics are
  **right-tailed** (large `W̄` ⇒ reject non-causality).

**Convention.** `W_i` is the χ²(p) Wald form `W = θ'V⁻¹θ` with `V = ŝ²(X'X)⁻¹`
and `ŝ² = RSS/(T−2p−1)` (the unbiased unrestricted OLS variance), so that
`W_i = p · F_i` where `F_i` is the classical joint-significance F statistic
(`F ~ F(p, T−2p−1)` under Gaussian errors). R's `plm::pgrangertest` reports the
**F-based** `W_i` (i.e. `F_i = W_i/p`); to cross-check, rescale by `p`.

The `Z̃` moments require `T > 2p+5` (else `Var[W_i]` is undefined) — units below
this are skipped, and the whole test errors if no unit qualifies.

References:
- Dumitrescu, E.-I. and Hurlin, C. (2012). "Testing for Granger Non-causality in
  Heterogeneous Panels." Economic Modelling 29(4): 1450-1460.
- Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models
  and Cross-spectral Methods." Econometrica 37(3): 424-438.
"""

# =============================================================================
# Exact finite-T moments of the individual Wald statistic (DH 2012, eqs. 26-27).
# `Tobs` is the per-unit regression sample (rows in the design after building p
# lags); residual df = Tobs − (2p+1). Requires Tobs > 2p+5.
# =============================================================================

"""Exact mean `E[W_i] = p(T−2p−1)/(T−2p−3)` (DH 2012, eq. 26)."""
function _dh_ew(Tobs::Int, p::Int)
    return p * (Tobs - 2p - 1) / (Tobs - 2p - 3)
end

"""Exact variance `Var[W_i] = 2p(T−2p−1)²(T−p−3)/[(T−2p−3)²(T−2p−5)]` (DH 2012, eq. 27)."""
function _dh_varw(Tobs::Int, p::Int)
    return 2p * (Tobs - 2p - 1)^2 * (Tobs - p - 3) /
           ((Tobs - 2p - 3)^2 * (Tobs - 2p - 5))
end

# =============================================================================
# Per-unit Granger-Wald statistic
# =============================================================================

"""
Individual DH Wald statistic for one unit. Regresses `yv` on an intercept, `p`
lags of `yv`, and `p` lags of `xv`; returns the χ²(p) Wald statistic
`W = θ'V⁻¹θ` (`θ` = the `p` lagged-`x` coefficients, `V = ŝ²(X'X)⁻¹`,
`ŝ² = RSS/(m−k)`), which equals `p · F_i`. Also returns the effective regression
sample `m = length(yv) − p`. Returns `(NaN, m)` when `m ≤ 2p+1` (rank-deficient).
"""
function _dh_unit_wald(yv::AbstractVector{T}, xv::AbstractVector{T}, p::Int) where {T<:AbstractFloat}
    Traw = length(yv)
    m = Traw - p                       # regression rows
    k = 2p + 1                         # intercept + p y-lags + p x-lags
    (m <= k) && return (T(NaN), m)
    yd = yv[(p+1):Traw]                # dependent, length m
    X = Matrix{T}(undef, m, k)
    @inbounds for t in 1:m
        X[t, 1] = one(T)
        for l in 1:p
            X[t, 1 + l]      = yv[p + t - l]   # y_{t-l}
            X[t, 1 + p + l]  = xv[p + t - l]   # x_{t-l}
        end
    end
    XtX_inv = robust_inv(X'X)
    beta = XtX_inv * (X'yd)
    resid = yd - X * beta
    rss = dot(resid, resid)
    s2 = rss / (m - k)
    restr = (p + 2):(2p + 1)           # the p lagged-x coefficient columns
    theta = beta[restr]
    Vsub = s2 .* XtX_inv[restr, restr]
    W = dot(theta, robust_inv(Vsub) * theta)
    return (max(W, zero(T)), m)
end

"""
Restricted (null: x does not cause y) OLS fit for one unit: regress `yv` on an
intercept and `p` own lags. Returns `(fitted, resid, Xr)` where `Xr` is the
`m×(p+1)` restricted design (used by the block bootstrap).
"""
function _dh_unit_restricted(yv::AbstractVector{T}, p::Int) where {T<:AbstractFloat}
    Traw = length(yv)
    m = Traw - p
    kr = p + 1
    yd = yv[(p+1):Traw]
    Xr = Matrix{T}(undef, m, kr)
    @inbounds for t in 1:m
        Xr[t, 1] = one(T)
        for l in 1:p
            Xr[t, 1 + l] = yv[p + t - l]
        end
    end
    br = robust_inv(Xr'Xr) * (Xr'yd)
    fitted = Xr * br
    resid = yd - fitted
    return (fitted, resid, Xr)
end

# =============================================================================
# Standardization
# =============================================================================

"""Standardize `W̄` to `(Z̄, Z̃)` from the averaged exact moments (`Ebar`, `Vbar`)."""
function _dh_standardize(Wbar::T, Ebar::T, Vbar::T, N::Int, p::Int) where {T<:AbstractFloat}
    Zbar = sqrt(T(N) / (2p)) * (Wbar - p)
    Ztilde = sqrt(T(N)) * (Wbar - Ebar) / sqrt(Vbar)
    return (Zbar, Ztilde)
end

# =============================================================================
# dh_causality_test
# =============================================================================

"""
    dh_causality_test(pd::PanelData, x::Symbol, y::Symbol; p::Int=1,
                      bootstrap::Int=0, seed::Int=1234) -> DumitrescuHurlinResult

Dumitrescu-Hurlin (2012) test of whether `x` Granger-causes `y` in a
heterogeneous panel (unit-specific coefficients).

- `H0`: `x` does not Granger-cause `y` for **any** unit.
- `H1`: `x` Granger-causes `y` for **some** units.

For each unit `i`, `y_it` is regressed on an intercept, `p` lags of `y_i` and `p`
lags of `x_i`; the individual Wald statistic `W_i` (χ²(p) convention, `df = p`) is
averaged to `W̄` and standardized to the asymptotic `Z̄` and small-`T` `Z̃`
(right-tailed). Units with an effective regression sample `T_i ≤ 2p+5` are skipped
(the `Z̃` moments are then undefined); the call errors if no unit qualifies.

# Arguments
- `pd`: a [`PanelData`](@ref) (build with `xtset`).
- `x`, `y`: cause and effect variable names (resolved via `pd.varnames`).

# Keyword Arguments
- `p`: lag order (default 1).
- `bootstrap`: number of block-bootstrap replications for a cross-sectional-
  dependence-robust p-value on `Z̄` (default 0 = none). The bootstrap resamples
  time blocks of the restricted-model residuals **jointly across units** (when the
  panel is balanced) to preserve cross-sectional dependence under the null.
- `seed`: RNG seed for the bootstrap (stored for reproducibility).

# Example
```julia
pd = xtset(df, :country, :year)
res = dh_causality_test(pd, :credit, :gdp; p=2)
res.Ztilde, res.Ztilde_pvalue
```

# References
- Dumitrescu & Hurlin (2012), Economic Modelling 29(4).
"""
function dh_causality_test(pd::PanelData{TT}, x::Symbol, y::Symbol;
                           p::Int=1, bootstrap::Int=0, seed::Int=1234) where {TT}
    p >= 1 || throw(ArgumentError("lag order p must be ≥ 1, got $p"))
    bootstrap >= 0 || throw(ArgumentError("bootstrap must be ≥ 0, got $bootstrap"))
    T = float(TT)
    xc = _panel_varindex(pd, x)
    yc = _panel_varindex(pd, y)
    xc == yc && throw(ArgumentError("x and y must be different variables"))
    Nall = pd.n_groups

    yvecs = Vector{Vector{T}}(undef, Nall)
    xvecs = Vector{Vector{T}}(undef, Nall)
    for g in 1:Nall
        gd = group_data(pd, g)
        ord = sortperm(gd.time_index)
        yvecs[g] = T.(gd.data[ord, yc])
        xvecs[g] = T.(gd.data[ord, xc])
    end

    Wvals = T[]
    Evals = T[]
    Vvals = T[]
    Tvals = Int[]
    kept = Int[]                       # original unit indices retained
    n_skipped = 0
    for g in 1:Nall
        Traw = length(yvecs[g])
        m = Traw - p                   # effective regression sample T_i
        if m <= 2p + 5                 # Var[W_i] undefined below this
            n_skipped += 1
            continue
        end
        Wi, mm = _dh_unit_wald(yvecs[g], xvecs[g], p)
        if !isfinite(Wi)
            n_skipped += 1
            continue
        end
        push!(Wvals, Wi)
        push!(Evals, T(_dh_ew(mm, p)))
        push!(Vvals, T(_dh_varw(mm, p)))
        push!(Tvals, mm)
        push!(kept, g)
    end

    N = length(Wvals)
    N == 0 && throw(ArgumentError(
        "No unit satisfies the small-T guard T > 2p+5 (need effective sample > $(2p+5) " *
        "after building $p lags); with p=$p every unit has too few observations."))

    Wbar = sum(Wvals) / N
    Ebar = sum(Evals) / N
    Vbar = sum(Vvals) / N
    Zbar, Ztilde = _dh_standardize(Wbar, Ebar, Vbar, N, p)
    # Right-tailed: large W̄ ⇒ evidence of causality.
    Zbar_p = T(ccdf(Normal(), Zbar))
    Ztilde_p = T(ccdf(Normal(), Ztilde))

    boot_p = T(NaN)
    if bootstrap > 0
        boot_p = _dh_bootstrap(yvecs[kept], xvecs[kept], p, N, bootstrap, seed, Zbar, T)
    end

    Tbar = round(Int, sum(Tvals) / N)
    return DumitrescuHurlinResult{T}(Wbar, Zbar, Zbar_p, Ztilde, Ztilde_p,
                                     Wvals, p, N, Tbar, n_skipped, bootstrap,
                                     seed, boot_p, x, y)
end

# String-name convenience.
function dh_causality_test(pd::PanelData, x::AbstractString, y::AbstractString; kwargs...)
    dh_causality_test(pd, Symbol(x), Symbol(y); kwargs...)
end

# =============================================================================
# Block bootstrap (CSD-robust p-value under H0)
# =============================================================================

"""
CSD-robust block bootstrap p-value for `Z̄`. Imposes the non-causality null by
resampling time blocks of the **restricted-model** residuals (`y` on its own
lags) and regenerating `y*` with the original design fixed. When all units share
a common effective sample the block draws are shared across units (joint resample)
to preserve cross-sectional dependence; otherwise each unit is resampled with its
own block draws. Returns the upper-tail fraction `#{Z̄* ≥ Z̄_obs}/B`.
"""
function _dh_bootstrap(yvecs::Vector{Vector{T}}, xvecs::Vector{Vector{T}},
                       p::Int, N::Int, B::Int, seed::Int, Zbar_obs::T,
                       ::Type{T}) where {T<:AbstractFloat}
    # Restricted fits + residuals per unit.
    fits = Vector{Vector{T}}(undef, N)
    resids = Vector{Vector{T}}(undef, N)
    Xrs = Vector{Matrix{T}}(undef, N)
    ms = Int[]
    for i in 1:N
        f, r, Xr = _dh_unit_restricted(yvecs[i], p)
        fits[i] = f; resids[i] = r; Xrs[i] = Xr
        push!(ms, length(r))
    end
    balanced = all(==(ms[1]), ms)
    L = max(1, round(Int, ms[1]^(1/3)))     # block length ~ m^{1/3}

    rng = Random.MersenneTwister(seed)
    count_ge = 0
    Evec = T[T(_dh_ew(ms[i], p)) for i in 1:N]
    Vvec = T[T(_dh_varw(ms[i], p)) for i in 1:N]

    for _ in 1:B
        # Build a resampled residual-index sequence. Joint (shared) across units
        # when balanced; per-unit otherwise.
        Wboot = Vector{T}(undef, N)
        idx_shared = balanced ? _block_indices(rng, ms[1], L) : Int[]
        for i in 1:N
            mi = ms[i]
            idx = balanced ? idx_shared : _block_indices(rng, mi, L)
            rstar = resids[i][idx]
            ystar = fits[i] .+ rstar         # y*_dep under H0 (design fixed)
            Wboot[i] = _dh_wald_fixed_design(ystar, yvecs[i], xvecs[i], p)
        end
        Wbar_b = sum(Wboot) / N
        Ebar_b = sum(Evec) / N
        Zbar_b = sqrt(T(N) / (2p)) * (Wbar_b - p)
        if Zbar_b >= Zbar_obs
            count_ge += 1
        end
    end
    return T(count_ge / B)
end

"""Draw a length-`m` sequence of time indices by concatenating random blocks of
length `L` (circular block bootstrap)."""
function _block_indices(rng::Random.AbstractRNG, m::Int, L::Int)
    idx = Vector{Int}(undef, m)
    filled = 0
    while filled < m
        start = rand(rng, 1:m)
        for j in 0:(L-1)
            filled += 1
            filled > m && break
            idx[filled] = ((start + j - 1) % m) + 1
        end
    end
    return idx
end

"""Unrestricted Wald on a bootstrap dependent `ystar` (already the `m`-vector of
dependent values) with the design built from the ORIGINAL `yv`/`xv` lags."""
function _dh_wald_fixed_design(ystar::AbstractVector{T}, yv::AbstractVector{T},
                               xv::AbstractVector{T}, p::Int) where {T<:AbstractFloat}
    Traw = length(yv)
    m = Traw - p
    k = 2p + 1
    X = Matrix{T}(undef, m, k)
    @inbounds for t in 1:m
        X[t, 1] = one(T)
        for l in 1:p
            X[t, 1 + l]     = yv[p + t - l]
            X[t, 1 + p + l] = xv[p + t - l]
        end
    end
    XtX_inv = robust_inv(X'X)
    beta = XtX_inv * (X'ystar)
    resid = ystar - X * beta
    s2 = dot(resid, resid) / (m - k)
    restr = (p + 2):(2p + 1)
    theta = beta[restr]
    Vsub = s2 .* XtX_inv[restr, restr]
    W = dot(theta, robust_inv(Vsub) * theta)
    return max(W, zero(T))
end
