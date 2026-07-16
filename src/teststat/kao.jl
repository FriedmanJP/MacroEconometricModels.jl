# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Kao (1999) residual-based panel cointegration tests (homogeneous cointegrating
vector). Estimates panel-specific fixed effects and a single common slope, then
tests the null of no cointegration (a unit root, ρ = 1, in the pooled residuals)
via five Dickey-Fuller-type statistics: DFρ, DFt, DF*ρ, DF*t, and ADF. All five
are asymptotically N(0,1) and **left-tailed** (large negative rejects).

The DF/DFt variants use closed-form N(0,1) standardizations (constants from Kao
1999); the modified variants (DF*ρ, DF*t, ADF) correct for endogeneity /
serial-correlation via the short-run σ̂²_ν and long-run ω̂²_ν conditional variances
of the DF residual given Δx.

References:
- Kao, C. (1999). Spurious Regression and Residual-Based Tests for Cointegration
  in Panel Data. Journal of Econometrics, 90(1), 1-44.
- StataCorp, xtcointtest (Kao tests), Methods and Formulas.
"""

# =============================================================================
# kao_test
# =============================================================================

"""
    kao_test(pd::PanelData, y::Symbol, xs::Symbol...; lags=:auto, kernel_lags=:auto)
        -> KaoResult

Kao (1999) residual-based panel cointegration test. Within-demeans `y` and `xs`
(panel-specific fixed effects), fits a pooled cointegrating regression with a
common slope, and reports five DF-type statistics.

# Keyword Arguments
- `lags`: lag order `p` of the pooled ADF regression. `:auto` (default) uses
  `round(4·(T/100)^{2/9})`; or an integer.
- `kernel_lags`: Bartlett bandwidth for the long-run variances ω̂²_ν. `:auto`
  (default) uses the same rule; or an integer.

# Statistics (all N(0,1), left-tailed, H0: no cointegration)
`DFrho` and `DFt` assume strict exogeneity (unadjusted); `DFrho_star` (modified
DFρ) and `DFt_star` (modified DFt) and `ADF` add the endogeneity correction.

# Example
```julia
pd = xtset(df, :country, :year)
res = kao_test(pd, :lny, :lnx)
res.pvalues
```

# References
- Kao (1999), Journal of Econometrics 90(1).
"""
function kao_test(pd::PanelData{TT}, y::Symbol, xs::Symbol...;
                  lags::Union{Int,Symbol}=:auto,
                  kernel_lags::Union{Int,Symbol}=:auto) where {TT}
    isempty(xs) && throw(ArgumentError("kao_test needs at least one regressor"))
    T = float(TT)
    Y, X = _panel_coint_data(pd, y, xs)
    return _kao_core(Y, X, lags, kernel_lags)
end

function _kao_core(Y::Matrix{T}, X::Array{T,3}, lags::Union{Int,Symbol},
                   kernel_lags::Union{Int,Symbol}) where {T<:AbstractFloat}
    Tobs, N = size(Y)
    k = size(X, 3)
    N < 2 && throw(ArgumentError("Kao test needs at least N=2 units, got N=$N"))

    p = lags === :auto ? max(1, round(Int, 4 * (Tobs / 100)^(2/9))) : (lags::Int)
    Kb = kernel_lags === :auto ? max(1, round(Int, 4 * (Tobs / 100)^(2/9))) : (kernel_lags::Int)
    p >= 0 || throw(ArgumentError("lags must be ≥ 0, got $p"))

    # --- Step 1: within-demean and pooled common-slope regression ---
    yv = Vector{T}(undef, Tobs * N)
    Xv = Matrix{T}(undef, Tobs * N, k)
    r = 0
    for i in 1:N
        yi = @view Y[:, i]
        Xi = @view X[:, i, :]
        ybar = mean(yi)
        xbar = vec(mean(Xi; dims=1))
        for t in 1:Tobs
            r += 1
            yv[r] = yi[t] - ybar
            @inbounds for j in 1:k
                Xv[r, j] = Xi[t, j] - xbar[j]
            end
        end
    end
    beta = Xv \ yv                                    # common slope (no intercept)
    ehat = reshape(yv - Xv * beta, Tobs, N)           # residuals, T×N

    # --- Step 2: pooled DF regression ê_it = ρ ê_{i,t-1} + u_it ---
    num = zero(T); den = zero(T)
    for i in 1:N, t in 2:Tobs
        num += ehat[t-1, i] * ehat[t, i]
        den += ehat[t-1, i]^2
    end
    rho = num / den
    # pooled DF residuals u and short-run pieces
    ssr = zero(T); nobs_df = N * (Tobs - 1)
    U = Matrix{T}(undef, Tobs - 1, N)                 # u_it aligned to t=2..T
    for i in 1:N, t in 2:Tobs
        u = ehat[t, i] - rho * ehat[t-1, i]
        U[t-1, i] = u
        ssr += u^2
    end
    s2 = ssr / (nobs_df - 1)
    se_rho = sqrt(s2 / den)
    t_rho = (rho - 1) / se_rho

    # --- Step 3: pooled ADF regression for t_ADF ---
    t_adf = _kao_adf_trho(ehat, p)

    # --- Step 4: short-run Σ̂ and long-run Ω̂ conditional variances ---
    # w_it = (u_it, Δx_it) stacked; Δx aligned to t=2..T.
    dX = Array{T,3}(undef, Tobs - 1, N, k)
    for i in 1:N, t in 2:Tobs, j in 1:k
        dX[t-1, i, j] = X[t, i, j] - X[t-1, i, j]
    end
    sig2_v = _kao_cond_var(U, dX, 0)                  # contemporaneous (short-run)
    om2_v = _kao_cond_var(U, dX, Kb)                  # long-run (Bartlett)
    sig2_v = max(sig2_v, T(1e-30))
    om2_v = max(om2_v, T(1e-30))
    sig_v = sqrt(sig2_v); om_v = sqrt(om2_v)

    # --- Step 5: the five statistics (all N(0,1), left-tailed) ---
    sqrtN = sqrt(T(N))
    # Unadjusted (strict-exogeneity) forms.
    dfrho = (sqrtN * Tobs * (rho - 1) + 3 * sqrtN) / sqrt(T(51) / 5)     # 51/5 = 10.2
    dft = sqrt(T(5) / 4) * t_rho + sqrt(T(15) / 8 * N)                    # √1.25 tρ + √(1.875 N)
    # Endogeneity-corrected forms.
    dfrho_star = (sqrtN * Tobs * (rho - 1) + 3 * sqrtN * sig2_v / om2_v) /
                 sqrt(3 + 36 * sig2_v^2 / (5 * om2_v^2))
    denom_t = sqrt(om2_v / (2 * sig2_v) + 3 * sig2_v / (10 * om2_v))
    dft_star = (t_rho + sqrt(T(6) * N) * sig_v / (2 * om_v)) / denom_t
    adf = (t_adf + sqrt(T(6) * N) * sig_v / (2 * om_v)) / denom_t

    names = ["DFrho", "DFt", "DFrho_star", "DFt_star", "ADF"]
    stats = T[dfrho, dft, dfrho_star, dft_star, adf]
    pvals = T[cdf(Normal(), s) for s in stats]        # all left-tailed

    KaoResult{T}(names, stats, pvals, T(rho), T(t_rho), T(t_adf),
                 sig2_v, om2_v, p, Kb, k, Tobs, N)
end

# Pooled ADF t-statistic for H0: ρ = 1 with p lagged differences (per-unit
# design stacked, common coefficients).
function _kao_adf_trho(ehat::Matrix{T}, p::Int) where {T<:AbstractFloat}
    Tobs, N = size(ehat)
    rows = Vector{T}[]
    dep = T[]
    for i in 1:N
        ei = @view ehat[:, i]
        de = diff(collect(ei))                        # Δê, length Tobs-1 (index t=2..T)
        # rows t = p+2 .. Tobs (need ê_{t-1} and p lagged diffs)
        for t in (p+2):Tobs
            reg = T[ei[t-1]]
            for j in 1:p
                push!(reg, de[(t-1) - j])              # Δê_{t-j}
            end
            push!(rows, reg)
            push!(dep, ei[t])
        end
    end
    isempty(dep) && return T(NaN)
    W = permutedims(hcat(rows...))                    # (nobs) × (1+p)
    b = W \ dep
    resid = dep - W * b
    dof = length(dep) - size(W, 2)
    s2 = dot(resid, resid) / max(dof, 1)
    WtW_inv = robust_inv(W'W)
    se_rho = sqrt(s2 * WtW_inv[1, 1])
    (b[1] - 1) / se_rho
end

# Conditional variance of u given ε from the (short-run if K=0, else Bartlett
# long-run) covariance of w = (u, ε): σ²_ν = Σ_uu − Σ_uε Σ_εε⁻¹ Σ_εu.
# Per-unit covariances are averaged across units (Kao pools homogeneously).
function _kao_cond_var(U::Matrix{T}, dX::Array{T,3}, K::Int) where {T<:AbstractFloat}
    m, N = size(U)
    k = size(dX, 3)
    d = 1 + k
    Ω = zeros(T, d, d)
    for i in 1:N
        W = Matrix{T}(undef, m, d)
        @inbounds for t in 1:m
            W[t, 1] = U[t, i]
            for j in 1:k
                W[t, 1+j] = dX[t, i, j]
            end
        end
        Ω .+= _bartlett_cov(W, K)
    end
    Ω ./= N
    ouu = Ω[1, 1]
    if k == 0
        return ouu
    end
    oue = Ω[1, 2:d]
    oee = Ω[2:d, 2:d]
    ouu - dot(oue, robust_inv(oee) * oue)
end

# (1/m)-scaled Bartlett long-run covariance matrix of the columns of W (already
# treated as mean-zero innovations). K=0 returns the contemporaneous covariance.
function _bartlett_cov(W::AbstractMatrix{T}, K::Int) where {T<:AbstractFloat}
    m = size(W, 1)
    Γ0 = (W' * W) ./ m
    K <= 0 && return Γ0
    Ω = copy(Γ0)
    for s in 1:K
        s >= m && break
        Γs = (W[(s+1):m, :]' * W[1:(m-s), :]) ./ m
        w = 1 - s / (K + 1)
        Ω .+= w .* (Γs .+ Γs')
    end
    Ω
end
