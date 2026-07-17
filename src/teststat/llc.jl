# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Levin-Lin-Chu (2002) first-generation panel unit root test (unit-root null).

Assumes cross-sectional independence and a common autoregressive parameter. The
bias-adjusted pooled t-statistic `t*_δ` is asymptotically N(0,1). Route genuine
cross-sectional dependence to the second-generation tests
([`pesaran_cips_test`](@ref), [`panic_test`](@ref)) instead.

References:
- Levin, A., Lin, C.-F., & Chu, C.-S. J. (2002). Unit root tests in panel data:
  asymptotic and finite-sample properties. Journal of Econometrics, 108(1), 1-24.
"""

# =============================================================================
# LLC (2002) Table 2 mean/std adjustments (μ*_T̃, σ*_T̃)
# Transcribed VERBATIM from the reference implementation `plm::purtest`
# (`adj.levinlin`, array dim c(13,2,3), dimnames T × {mu,sigma} × {none,intercept,
# trend}); these reproduce LLC (2002, Table 2). Column-major fill order verified
# against the raw `v <- c(...)` literal in plm/R/test_uroot.R.
# =============================================================================

const _LLC_T_GRID = Float64[25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 250, 500]

# (μ*, σ*) per deterministic case, aligned to _LLC_T_GRID.
const _LLC_ADJ = Dict{Symbol,NTuple{2,Vector{Float64}}}(
    :none => (
        Float64[0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        Float64[1.049, 1.035, 1.027, 1.021, 1.017, 1.014, 1.011, 1.008, 1.007, 1.006, 1.005, 1.001, 1.000],
    ),
    :constant => (
        Float64[-0.554, -0.546, -0.541, -0.537, -0.533, -0.531, -0.527, -0.524, -0.521, -0.520, -0.518, -0.509, -0.500],
        Float64[0.919, 0.889, 0.867, 0.850, 0.837, 0.826, 0.810, 0.798, 0.789, 0.782, 0.776, 0.742, 0.707],
    ),
    :trend => (
        Float64[-0.703, -0.674, -0.653, -0.637, -0.624, -0.614, -0.598, -0.587, -0.578, -0.571, -0.566, -0.533, -0.500],
        Float64[1.003, 0.949, 0.906, 0.871, 0.842, 0.818, 0.780, 0.751, 0.728, 0.710, 0.695, 0.603, 0.500],
    ),
)

"""Linear interpolation of `y` (aligned to sorted `xgrid`) at `x`, clamped at ends."""
function _interp_clamped(x::Real, xgrid::AbstractVector, y::AbstractVector)
    x <= xgrid[1] && return float(y[1])
    x >= xgrid[end] && return float(y[end])
    k = searchsortedfirst(xgrid, x)          # first index with xgrid[k] >= x
    xgrid[k] == x && return float(y[k])
    x0, x1 = xgrid[k-1], xgrid[k]
    y0, y1 = y[k-1], y[k]
    float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
end

"""Look up LLC (μ*_T̃, σ*_T̃) at adjusted sample T̃ for a deterministic case."""
function _llc_adjustment(T_tilde::Real, deterministic::Symbol)
    haskey(_LLC_ADJ, deterministic) ||
        throw(ArgumentError("LLC deterministic must be :none, :constant, or :trend"))
    mu_grid, sig_grid = _LLC_ADJ[deterministic]
    mu = _interp_clamped(T_tilde, _LLC_T_GRID, mu_grid)
    sig = _interp_clamped(T_tilde, _LLC_T_GRID, sig_grid)
    (mu, sig)
end

# =============================================================================
# LLC test
# =============================================================================

"""
    llc_test(X::AbstractMatrix{T}; deterministic=:constant, lags=:auto,
             max_lags=nothing, criterion=:aic, cs_demean=false) -> LLCResult{T}

Levin-Lin-Chu (2002) panel unit root test. `X` is `T×N` (time in rows, units in
columns); a `PanelData` method is also provided.

Follows the three-step procedure of LLC (2002) as documented in the Stata
`xtunitroot llc` Methods and Formulas: (1) per-unit ADF regressions produce
orthogonalized residuals `ẽ`, `ṽ` normalized by `σ̂_εi`; (2) the ratio of
long-run to short-run standard deviations `S̄_N` is formed from a Bartlett-kernel
long-run variance (bandwidth `int(3.21·T^{1/3})`); (3) a pooled OLS of `ẽ` on `ṽ`
gives `t_δ`, adjusted to `t*_δ = (t_δ − N·T̃·S̄_N·se(δ̂)·μ*_T̃)/σ*_T̃ ~ N(0,1)`.

# Keyword Arguments
- `deterministic`: `:none`, `:constant` (default), or `:trend`
- `lags`: common integer lag `p` for all units, or `:auto` for per-unit IC selection
- `max_lags`: cap for `:auto` selection (default `floor(12·(T/100)^{1/4})`)
- `criterion`: `:aic` (default), `:bic`, or `:hqic` for `:auto` selection
- `cs_demean`: subtract the cross-sectional mean at each `t` before testing — a
  crude cross-sectional-dependence mitigation. This alters the null distribution;
  for genuine CSD prefer [`pesaran_cips_test`](@ref) / [`panic_test`](@ref).

# Example
```julia
X = cumsum(randn(60, 20); dims=1)   # random-walk panel
result = llc_test(X; deterministic=:constant)
result.pvalue      # large ⇒ fail to reject the panel unit root
```

# References
- Levin, Lin & Chu (2002). Journal of Econometrics, 108(1), 1-24.
"""
function llc_test(X::AbstractMatrix{T};
                  deterministic::Symbol=:constant,
                  lags::Union{Int,Symbol}=:auto,
                  max_lags::Union{Int,Nothing}=nothing,
                  criterion::Symbol=:aic,
                  cs_demean::Bool=false) where {T<:AbstractFloat}
    deterministic in (:none, :constant, :trend) || throw(ArgumentError(
        "deterministic must be :none, :constant, or :trend, got :$deterministic"))
    Xw = cs_demean ? _cs_demean(X) : X
    T_obs, N = size(Xw)
    T_obs < 10 && throw(ArgumentError("Time dimension T=$T_obs too small for LLC"))
    N < 2 && throw(ArgumentError("LLC needs at least N=2 panel units, got N=$N"))

    reg = deterministic                      # matches adf regression symbol
    mpx = isnothing(max_lags) ? max(1, floor(Int, 12 * (T_obs / 100)^0.25)) : max_lags

    m = max(1, floor(Int, 3.21 * T_obs^(1/3)))   # Bartlett bandwidth (LLC rule)

    e_pool = T[]                              # ẽ_it stacked over all units
    v_pool = T[]                              # ṽ_{i,t-1} stacked
    s_ratios = T[]                            # ŝ_i = σ̂_yi / σ̂_εi
    p_used = Int[]

    for i in 1:N
        y = @view Xw[:, i]
        p = lags === :auto ? adf_select_lags(collect(y), mpx, reg, criterion) : (lags::Int)
        push!(p_used, p)

        dy = diff(y)                         # T_obs-1
        n = length(dy) - p                   # effective rows
        n < (p + 4) && continue              # too few obs for this unit

        # Regressor block W = deterministics + lagged differences (NOT y_{t-1}).
        W = _llc_det_block(T, reg, n)
        if p > 0
            L = Matrix{T}(undef, n, p)
            for j in 1:p
                L[:, j] = dy[(p+1-j):(length(dy)-j)]
            end
            W = hcat(W, L)
        end
        dY = dy[(p+1):end]                    # Δy_it aligned, length n
        y_lag = y[(p+1):(length(dy))]         # y_{i,t-1}, length n

        WtW_inv = robust_inv(W'W)
        e_hat = dY - W * (WtW_inv * (W'dY))       # (8) residualized Δy
        v_hat = y_lag - W * (WtW_inv * (W'y_lag))  # (9) residualized y_{t-1}

        # δ̂_i and short-run σ̂_εi (Stata: /(T - p - 1))
        svv = dot(v_hat, v_hat)
        svv <= zero(T) && continue
        delta_i = dot(v_hat, e_hat) / svv
        res_i = e_hat .- delta_i .* v_hat
        sig2_eps = dot(res_i, res_i) / max(T_obs - p - 1, 1)
        sig_eps = sqrt(max(sig2_eps, T(1e-30)))

        # Long-run variance of Δy (Bartlett); constant/trend ⇒ demean Δy.
        dyc = (reg == :none) ? dy : (dy .- mean(dy))
        sig2_y = _long_run_variance(collect(dyc), m)
        sig2_y <= zero(T) && continue
        push!(s_ratios, sqrt(sig2_y) / sig_eps)

        append!(e_pool, e_hat ./ sig_eps)
        append!(v_pool, v_hat ./ sig_eps)
    end

    (isempty(s_ratios) || isempty(v_pool)) &&
        throw(ArgumentError("LLC: no panel unit yielded a usable ADF regression"))

    S_N = mean(s_ratios)
    p_bar = mean(p_used)
    T_tilde = T_obs - p_bar - 1

    # Step 3: pooled regression ẽ = δ ṽ + ε̃.
    Svv = dot(v_pool, v_pool)
    delta = dot(v_pool, e_pool) / Svv
    res_pool = e_pool .- delta .* v_pool
    NT = length(e_pool)
    sig2_etil = dot(res_pool, res_pool) / NT      # (1/(N T̃)) ΣΣ (ẽ - δ ṽ)²
    sig_etil = sqrt(max(sig2_etil, T(1e-30)))
    se_delta = sig_etil / sqrt(Svv)
    t_delta = delta / se_delta

    mu_star, sigma_star = _llc_adjustment(T_tilde, deterministic)
    t_star = (t_delta - N * T_tilde * S_N * se_delta * mu_star) / sigma_star

    pval = T(cdf(Normal(), t_star))               # left-tailed unit-root null

    LLCResult{T}(t_star, pval, t_delta, delta, S_N, T(mu_star), T(sigma_star),
                 T(T_tilde), p_used, deterministic, T_obs, N)
end

llc_test(X::AbstractMatrix; kwargs...) = llc_test(Float64.(X); kwargs...)
llc_test(pd::PanelData; kwargs...) = llc_test(_panel_to_matrix(pd); kwargs...)

# Deterministic block for the LLC auxiliary regressions.
function _llc_det_block(::Type{T}, reg::Symbol, n::Int) where {T<:AbstractFloat}
    if reg == :none
        return Matrix{T}(undef, n, 0)
    elseif reg == :constant
        return reshape(ones(T, n), n, 1)
    else  # :trend
        return hcat(ones(T, n), T.(1:n))
    end
end

"""Subtract the cross-sectional mean at each time period (crude CSD mitigation)."""
function _cs_demean(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    X .- mean(X; dims=2)
end
_cs_demean(X::AbstractMatrix) = _cs_demean(Float64.(X))
