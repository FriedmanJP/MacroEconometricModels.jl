# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Pedroni (1999, 2004) residual-based panel cointegration tests.

Seven statistics split into a *within-dimension* (panel) family that pools the
numerators and denominators across units before taking the ratio, and a
*between-dimension* (group) family that averages the per-unit ratios. All seven
are asymptotically N(0,1) after subtracting `μ√N` and dividing by `√v`, with the
(μ, v) adjustment moments taken from Pedroni (1999, Table 2, p. 666) — the same
values reproduced in Pedroni (2004). **The panel-v statistic is RIGHT-tailed**
(large positive rejects the no-cointegration null); the other six are
left-tailed.

References:
- Pedroni, P. (1999). Critical Values for Cointegration Tests in Heterogeneous
  Panels with Multiple Regressors. Oxford Bulletin of Economics and Statistics,
  61(S1), 653-670.
- Pedroni, P. (2004). Panel Cointegration: Asymptotic and Finite Sample
  Properties of Pooled Time Series Tests with an Application to the PPP
  Hypothesis. Econometric Theory, 20(3), 597-625.
"""

# =============================================================================
# Shared helpers for the panel-cointegration tests (EV-21).
# Included first (before kao/westerlund/fisher_johansen), so these are visible
# to the sibling files.
# =============================================================================

"""
Stata/EViews-convention Bartlett long-run variance: uncentered autocovariances
scaled by `1/n` (NOT `1/(n-j)`) and `nodemean` (assumes `x` already centered).
Distinct from `_long_run_variance` (which centers via `var(;corrected=false)`);
the residual-based panel-cointegration statistics require this exact convention
to reproduce Pedroni's `pco`/Stata `xtcointtest`/`xtwest` reference values.
"""
function _stata_lrv(x::AbstractVector{T}, maxlag::Int) where {T<:AbstractFloat}
    n = length(x)
    n == 0 && return T(NaN)
    g0 = zero(T)
    @inbounds for t in 1:n
        g0 += x[t]^2
    end
    g0 /= n
    maxlag <= 0 && return g0
    sw = zero(T)
    for j in 1:maxlag
        j >= n && break
        gj = zero(T)
        @inbounds for t in (j+1):n
            gj += x[t] * x[t-j]
        end
        gj /= n
        sw += (1 - j / (maxlag + 1)) * gj
    end
    g0 + 2 * sw
end

"""One-sided Bartlett autocovariance sum `(1/n)Σₛ(1−s/(k+1))Σₜ xₜxₜ₋ₛ` — Pedroni's
`nw`/λ̂ building block (Pedroni 1999; matches the `pco` R package)."""
function _ped_nw_oneside(x::AbstractVector{T}, k::Int) where {T<:AbstractFloat}
    n = length(x)
    s_out = zero(T)
    for s in 1:k
        s >= n && break
        acc = zero(T)
        @inbounds for t in (s+1):n
            acc += x[t] * x[t-s]
        end
        s_out += (1 - s / (k + 1)) * acc
    end
    s_out / n
end

"""
Extract balanced per-unit `(Y, X)` from a `PanelData` for the cointegration
tests. Returns `Y::Matrix{T}` (`T×N`) and `X::Array{T,3}` (`T×N×k`), with units
ordered by group id and observations sorted by time within each group. Errors on
an unbalanced panel (the residual/ECM statistics assume common `T`).
"""
function _panel_coint_data(pd::PanelData{T}, y::Union{Symbol,String},
                           xs::Tuple) where {T<:AbstractFloat}
    isempty(xs) && throw(ArgumentError("At least one regressor is required"))
    yc = _panel_varindex(pd, y)
    xcs = Int[_panel_varindex(pd, x) for x in xs]
    k = length(xcs)
    N = pd.n_groups
    Ycols = Vector{Vector{T}}(undef, N)
    Xcols = Vector{Matrix{T}}(undef, N)
    Tg = -1
    for g in 1:N
        gd = group_data(pd, g)
        ti = gd.time_index
        ord = sortperm(ti)
        yg = T.(gd.data[ord, yc])
        Xg = T.(gd.data[ord, xcs])
        if Tg < 0
            Tg = length(yg)
        elseif length(yg) != Tg
            throw(ArgumentError(
                "Panel cointegration tests require a balanced panel; group $g has " *
                "$(length(yg)) obs vs $Tg. Balance the panel first."))
        end
        Ycols[g] = yg
        Xcols[g] = Xg
    end
    Tg < 4 && throw(ArgumentError("Time dimension T=$Tg too small for a cointegration test"))
    Y = Matrix{T}(undef, Tg, N)
    X = Array{T,3}(undef, Tg, N, k)
    for g in 1:N
        Y[:, g] = Ycols[g]
        X[:, g, :] = Xcols[g]
    end
    Y, X
end

"""OLS residuals of `y` on `Z` (columns already include any deterministics)."""
function _resid_ols(y::AbstractVector{T}, Z::AbstractMatrix{T}) where {T<:AbstractFloat}
    size(Z, 2) == 0 && return copy(y)
    b = Z \ y
    y - Z * b
end

# Deterministic block for the per-unit cointegrating regression.
#   :none      -> no column
#   :constant  -> intercept
#   :trend     -> intercept + linear trend
function _coint_det(::Type{T}, trend::Symbol, n::Int) where {T<:AbstractFloat}
    if trend == :none
        return Matrix{T}(undef, n, 0)
    elseif trend == :constant
        return reshape(ones(T, n), n, 1)
    else # :trend
        return hcat(ones(T, n), T.(1:n))
    end
end

# =============================================================================
# Pedroni (1999) Table 2 (p. 666) adjustment moments.
# Transcribed VERBATIM from the `pco` R package (`pedroni99`/`pedroni99m`,
# arrays `stamm`/`stavv`), which "strictly follows the text" of Pedroni (1999).
# Third index = number of regressors k ∈ 1:6 (the bivariate `pedroni99`, k=1,
# uses slice 1); second index = deterministic case (:none,:constant,:trend).
# Row order of the 7 statistics: panel-v, panel-ρ, panel-t, panel-ADF-t,
# group-ρ, group-t, group-ADF-t.
# =============================================================================

const _PEDRONI_STATS = ("panel-v", "panel-rho", "panel-t", "panel-adf",
                        "group-rho", "group-t", "group-adf")

# mean μ : Dict k => (none, constant, trend) each a length-7 vector.
const _PEDRONI_MU = Dict{Int,NTuple{3,Vector{Float64}}}(
    1 => ([6.982, -6.388, -1.662, -1.662, -9.889, -1.992, -1.992],
          [11.754, -9.495, -2.177, -2.177, -12.938, -2.453, -2.453],
          [21.162, -14.011, -2.648, -2.648, -17.359, -2.872, -2.872]),
    2 => ([10.402, -10.191, -2.156, -2.156, -13.865, -2.44, -2.44],
          [15.197, -13.256, -2.567, -2.567, -16.888, -2.827, -2.827],
          [24.556, -17.6, -2.967, -2.967, -21.116, -3.179, -3.179]),
    3 => ([14.254, -14.136, -2.571, -2.571, -17.834, -2.819, -2.819],
          [18.91, -17.163, -2.93, -2.93, -20.841, -3.157, -3.157],
          [28.046, -21.287, -3.262, -3.262, -24.93, -3.464, -3.464]),
    4 => ([18.198, -18.042, -2.926, -2.926, -21.805, -3.151, -3.151],
          [22.715, -21.013, -3.241, -3.241, -24.775, -3.452, -3.452],
          [31.738, -25.13, -3.545, -3.545, -28.849, -3.737, -3.737]),
    5 => ([22.169, -21.985, -3.244, -3.244, -25.75, -3.45, -3.45],
          [26.603, -24.944, -3.531, -3.531, -28.72, -3.726, -3.726],
          [35.537, -28.981, -3.806, -3.806, -32.716, -3.986, -3.986]),
    6 => ([26.12, -25.889, -3.533, -3.533, -29.627, -3.723, -3.723],
          [30.457, -28.795, -3.795, -3.795, -32.538, -3.976, -3.976],
          [39.231, -32.756, -4.047, -4.047, -36.494, -4.217, -4.217]),
)

# variance v : Dict k => (none, constant, trend).
const _PEDRONI_V = Dict{Int,NTuple{3,Vector{Float64}}}(
    1 => ([81.145, 64.288, 1.559, 1.559, 41.943, 0.649, 0.649],
          [104.546, 57.61, 0.964, 0.964, 51.49, 0.618, 0.618],
          [160.249, 64.219, 0.69, 0.69, 66.387, 0.555, 0.555]),
    2 => ([140.804, 89.962, 1.286, 1.286, 57.801, 0.6, 0.6],
          [151.094, 81.772, 0.923, 0.923, 67.123, 0.585, 0.585],
          [198.167, 83.815, 0.686, 0.686, 81.832, 0.548, 0.548]),
    3 => ([182.45, 103.176, 1.028, 1.028, 72.097, 0.567, 0.567],
          [190.661, 99.331, 0.843, 0.843, 81.835, 0.56, 0.56],
          [239.425, 103.905, 0.688, 0.688, 97.362, 0.543, 0.543]),
    4 => ([217.784, 120.787, 0.928, 0.928, 88.611, 0.559, 0.559],
          [231.864, 119.546, 0.8, 0.8, 98.278, 0.553, 0.553],
          [276.997, 124.613, 0.686, 0.686, 113.145, 0.538, 0.538]),
    5 => ([256.53, 132.499, 0.82, 0.82, 103.371, 0.544, 0.544],
          [270.451, 134.341, 0.75, 0.75, 113.131, 0.542, 0.542],
          [310.982, 138.227, 0.654, 0.654, 127.989, 0.53, 0.53]),
    6 => ([277.429, 143.561, 0.75, 0.75, 117.059, 0.53, 0.53],
          [293.431, 144.615, 0.685, 0.685, 126.059, 0.525, 0.525],
          [348.217, 154.378, 0.638, 0.638, 140.756, 0.518, 0.518]),
)

"""Look up the Pedroni (1999) Table 2 (μ, v) length-7 vectors for `k` regressors
and deterministic case `trend`."""
function _pedroni_moments(k::Int, trend::Symbol)
    1 <= k <= 6 || throw(ArgumentError(
        "Pedroni moments tabulated for 1..6 regressors, got k=$k"))
    ci = trend == :none ? 1 : trend == :constant ? 2 : 3
    (_PEDRONI_MU[k][ci], _PEDRONI_V[k][ci])
end

# Pedroni's parametric-ADF residualization (matches `pco::adfl`): regress
# ê_t on ê_{t-1} and (lags-1) constructed differences ê_t − ê_{t−(lags−j)},
# no intercept, return residuals.
function _ped_adf_resid(ee::AbstractVector{T}, lags::Int) where {T<:AbstractFloat}
    nn = length(ee)
    m = nn - lags                       # effective rows
    z = ee[(lags+1):nn]                 # ê_t, length m
    zl = ee[lags:(nn-1)]                # ê_{t-1}, length m
    ncol = 1 + max(0, lags - 1)
    W = Matrix{T}(undef, m, ncol)
    @inbounds for t in 1:m
        W[t, 1] = zl[t]
        for j in 2:lags
            # pco: zd[t,j] = z[t] - ee[t+lags-j]; keep j = 2..lags
            W[t, j] = z[t] - ee[t + lags - j]
        end
    end
    _resid_ols(z, W)
end

# =============================================================================
# pedroni_test
# =============================================================================

"""
    pedroni_test(pd::PanelData, y::Symbol, xs::Symbol...; trend=:constant,
                 lags=:auto, adf_lags=2) -> PedroniResult

Pedroni (1999, 2004) residual-based panel cointegration test. Runs the per-unit
cointegrating regression of `y` on the deterministics and `xs`, then forms the
seven pooled/group statistics from the residuals.

# Keyword Arguments
- `trend`: deterministic case of the cointegrating regression — `:none`,
  `:constant` (default, an intercept), or `:trend` (intercept + linear trend).
- `lags`: Newey-West bandwidth for the residual long-run variances. `:auto`
  (default) uses `round(4·(T/100)^{2/9})` (the `pco`/Kao rule); or an integer.
- `adf_lags`: augmentation order for the parametric (ADF) statistics (default 2).

# Statistics (all standardized to N(0,1) under H0: no cointegration)
Within-dimension (panel): `panel-v` (variance ratio, **RIGHT-tailed**), `panel-ρ`,
`panel-t` (nonparametric), `panel-ADF-t` (parametric). Between-dimension (group):
`group-ρ`, `group-t`, `group-ADF-t`. All but `panel-v` are left-tailed.

# Example
```julia
pd = xtset(df, :country, :year)
res = pedroni_test(pd, :lny, :lnx; trend=:constant)
res.pvalues        # one per statistic
```

# References
- Pedroni (1999), OBES 61(S1); Pedroni (2004), Econometric Theory 20(3).
"""
function pedroni_test(pd::PanelData{TT}, y::Symbol, xs::Symbol...;
                      trend::Symbol=:constant,
                      lags::Union{Int,Symbol}=:auto,
                      adf_lags::Int=2) where {TT}
    trend in (:none, :constant, :trend) || throw(ArgumentError(
        "trend must be :none, :constant, or :trend, got :$trend"))
    isempty(xs) && throw(ArgumentError("pedroni_test needs at least one regressor"))
    T = float(TT)
    Y, X = _panel_coint_data(pd, y, xs)
    return _pedroni_core(Y, X, trend, lags, adf_lags)
end

function _pedroni_core(Y::Matrix{T}, X::Array{T,3}, trend::Symbol,
                       lags::Union{Int,Symbol}, adf_lags::Int) where {T<:AbstractFloat}
    Tobs, N = size(Y)
    k = size(X, 3)
    N < 2 && throw(ArgumentError("Pedroni test needs at least N=2 units, got N=$N"))
    adf_lags < 2 && throw(ArgumentError("adf_lags must be ≥ 2, got $adf_lags"))

    nw = lags === :auto ? max(1, round(Int, 4 * (Tobs / 100)^(2/9))) : (lags::Int)

    # Per-unit residuals and building blocks.
    e = Matrix{T}(undef, Tobs, N)              # cointegrating-regression residuals
    L11 = Vector{T}(undef, N)                   # L̂²_11i (LRV of differenced-regression resid)
    lambda = Vector{T}(undef, N)                # λ̂_i
    shat2 = Vector{T}(undef, N)                 # ŝ²_i (short-run var of AR resid)
    sigmahat2 = Vector{T}(undef, N)             # σ̂²_i = ŝ²_i + 2λ̂_i
    shatstar2 = Vector{T}(undef, N)             # ŝ*²_i (ADF resid var)

    for i in 1:N
        yi = @view Y[:, i]
        Xi = @view X[:, i, :]
        det = _coint_det(T, trend, Tobs)
        Z = size(det, 2) == 0 ? Matrix{T}(Xi) : hcat(det, Xi)
        ei = _resid_ols(collect(yi), Z)
        e[:, i] = ei

        # differenced-regression residuals η̂ (no intercept): ΔY on ΔX
        dY = diff(collect(yi))
        dX = diff(Matrix(Xi); dims=1)
        eta = _resid_ols(dY, dX)
        L11[i] = _stata_lrv(eta, nw)

        # AR(1) residuals μ̂: e_t on e_{t-1} (no intercept)
        el = ei[1:(Tobs-1)]
        ec = ei[2:Tobs]
        rho = dot(el, ec) / dot(el, el)
        mu = ec .- rho .* el
        lambda[i] = _ped_nw_oneside(mu, nw)
        shat2[i] = dot(mu, mu) / length(mu)
        sigmahat2[i] = shat2[i] + 2 * lambda[i]

        # parametric ADF residuals
        mustar = _ped_adf_resid(ei, adf_lags)
        shatstar2[i] = dot(mustar, mustar) / length(mustar)
    end

    stildestar2 = sum(shatstar2) / N
    sigmatilde2 = sum(i -> L11[i]^(-2) * sigmahat2[i], 1:N) / N

    # pooled numerator/denominator building blocks
    nipa = zero(T)                              # Σᵢ L11⁻² Σₜ ê²_{t-1}
    lel = zero(T)                               # Σᵢ L11⁻² Σₜ (ê_{t-1}Δê_t − λ̂ᵢ)
    denom_par = zero(T)                         # Σᵢ L11⁻² Σₜ ê²_{t-1}   (parametric denom)
    num_par = zero(T)                           # Σᵢ L11⁻² Σₜ ê_{t-1}Δê_t (parametric num)
    rhogroup = zero(T)
    tgroupnp = zero(T)
    tgrouppar = zero(T)

    for i in 1:N
        ei = @view e[:, i]
        el = ei[1:(Tobs-1)]                     # ê_{t-1}
        de = diff(collect(ei))                  # Δê_t  (length T-1, aligned with el)
        w = L11[i]^(-2)
        sum_ell2 = dot(el, el)
        sum_num = dot(el, de) - lambda[i] * (Tobs - 1)   # Σ (ê_{t-1}Δê_t − λ̂ᵢ)
        sum_num_par = dot(el, de)                          # Σ ê_{t-1}Δê_t (no λ̂ for ADF)

        nipa += w * sum_ell2
        lel += w * sum_num
        denom_par += w * sum_ell2
        num_par += w * sum_num_par

        rhogroup += sum_num / sum_ell2
        tgroupnp += sum_num / sqrt(sigmahat2[i] * sum_ell2)
        tgrouppar += sum_num_par / sqrt(shat2[i] * sum_ell2)
    end

    raw = Vector{T}(undef, 7)
    raw[1] = Tobs^2 * N^(T(3)/2) / nipa                              # panel-v
    raw[2] = Tobs * sqrt(T(N)) / nipa * lel                          # panel-ρ
    raw[3] = lel / sqrt(sigmatilde2 * nipa)                          # panel-t (np)
    raw[4] = num_par / sqrt(stildestar2 * denom_par)                 # panel-ADF-t
    raw[5] = Tobs / sqrt(T(N)) * rhogroup                            # group-ρ
    raw[6] = tgroupnp / sqrt(T(N))                                   # group-t (np)
    raw[7] = tgrouppar / sqrt(T(N))                                  # group-ADF-t

    mu, vv = _pedroni_moments(k, trend)
    std = Vector{T}(undef, 7)
    pv = Vector{T}(undef, 7)
    for s in 1:7
        std[s] = (raw[s] - T(mu[s]) * sqrt(T(N))) / sqrt(T(vv[s]))
        # panel-v (s==1) is right-tailed; the other six are left-tailed.
        pv[s] = s == 1 ? T(ccdf(Normal(), std[s])) : T(cdf(Normal(), std[s]))
    end

    PedroniResult{T}(collect(_PEDRONI_STATS), raw, std, pv,
                     T.(mu), T.(vv), trend, k, nw, adf_lags, Tobs, N)
end
