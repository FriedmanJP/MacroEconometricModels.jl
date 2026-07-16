# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# DOLS — Dynamic OLS, Saikkonen (1991) / Stock–Watson (1993)
# =============================================================================

# The levels regression y_t = D_t'δ + x_t'β + u_t is augmented with the contemporaneous
# value plus `nlag` lags and `nlead` leads of Δx_t; the level-regressor coefficients are the
# efficient long-run estimates. Standard errors are long-run-variance corrected: ω̂ (the HAC
# long-run variance of the DOLS residual) replaces σ² in the OLS covariance (Stock–Watson
# 1993), or a Newey–West sandwich is used when `dols_se=:robust`.

"""
    _make_leadlag(Xdelta, nlag, nlead) -> Matrix

Stack `[Δx_t, Δx_{t−1..t−nlag}, Δx_{t+1..t+nlead}]` column-wise (zeros outside the sample),
matching cointReg's `makeLeadLagMatrix`. `Xdelta` has one row per `t = 2:T`.
"""
function _make_leadlag(Xdelta::Matrix{T}, nlag::Int, nlead::Int) where {T<:AbstractFloat}
    m, k = size(Xdelta)
    blocks = Matrix{T}[Xdelta]
    for i in 1:nlag
        L = zeros(T, m, k)
        @inbounds for r in (i + 1):m
            @views L[r, :] .= Xdelta[r - i, :]
        end
        push!(blocks, L)
    end
    for i in 1:nlead
        L = zeros(T, m, k)
        @inbounds for r in 1:(m - i)
            @views L[r, :] .= Xdelta[r + i, :]
        end
        push!(blocks, L)
    end
    return reduce(hcat, blocks)
end

"""
    _dols_design(y, Z, Xdelta, nlag, nlead) -> (Xfull, yt)

Build the truncated DOLS design `Xfull = [Z, leads/lags of Δx]` and aligned response `yt`
over the common sample, replicating cointReg's `getModD`.
"""
function _dols_design(y::Vector{T}, Z::Matrix{T}, Xdelta::Matrix{T},
                      nlag::Int, nlead::Int) where {T<:AbstractFloat}
    n = length(y)
    if nlag + nlead == 0
        return copy(Z), copy(y)
    end
    dx_all = _make_leadlag(Xdelta, nlag, nlead)      # (T-1)×(k·(1+nlag+nlead))
    Zs = Z[2:n, :]
    all_untrunc = hcat(Zs, dx_all)                   # (T-1) rows
    T1 = size(all_untrunc, 1)                        # = T-1
    rows = (nlag + 1):(T1 - nlead)
    Xfull = all_untrunc[rows, :]
    ys = y[2:n]
    yt = ys[rows]
    return Xfull, yt
end

"""
    _dols_ssr(y, Z, Xdelta, nlag, nlead) -> (SSR, act_samp)

Residual sum of squares and effective sample size of the DOLS regression, for lead/lag
selection (cointReg `getLeadLag` objective).
"""
function _dols_ssr(y::Vector{T}, Z::Matrix{T}, Xdelta::Matrix{T},
                   nlag::Int, nlead::Int) where {T<:AbstractFloat}
    Xfull, yt = _dols_design(y, Z, Xdelta, nlag, nlead)
    theta = robust_inv(Symmetric(Xfull' * Xfull)) * (Xfull' * yt)
    resid = yt .- Xfull * theta
    return sum(abs2, resid), length(yt)
end

"""
    _dols_select_leadlag(y, Z, Xdelta, k, ic) -> (nlag, nlead)

Independent AIC/BIC selection of DOLS leads/lags over `0:kmax`, `kmax = ⌊4(T/100)^{1/4}⌋`,
matching cointReg's `getLeadLag` (`symmet=FALSE`, `kmax="k4"`).
"""
function _dols_select_leadlag(y::Vector{T}, Z::Matrix{T}, Xdelta::Matrix{T},
                              k::Int, ic::Symbol) where {T<:AbstractFloat}
    n = length(y)
    kmax = floor(Int, 4 * (n / 100)^(1 / 4))
    kmax = max(kmax, 0)
    best_ic = T(Inf)
    best = (0, 0)
    for nlag in 0:kmax, nlead in 0:kmax
        act = (nlag + nlead == 0) ? n : n - 1 - nlag - nlead
        act <= k * (nlag + nlead + 2) + 2 && continue
        SSR, _ = _dols_ssr(y, Z, Xdelta, nlag, nlead)
        SSR <= 0 && continue
        npar = k * (nlag + nlead + 2) + 2
        pen = ic === :bic ? log(T(act)) : T(2)
        crit = T(act) * log(SSR / act) + pen * npar
        if crit < best_ic
            best_ic = crit
            best = (nlag, nlead)
        end
    end
    return best
end

function _estimate_dols(y::Vector{T}, X::Matrix{T}, trend::Symbol, kernel::Symbol,
                        bandwidth, leads, lags, ic::Symbol, dols_se::Symbol) where {T<:AbstractFloat}
    dols_se ∈ (:lrv, :robust) ||
        throw(ArgumentError("dols_se must be :lrv or :robust; got :$dols_se"))
    ic ∈ (:aic, :bic) || throw(ArgumentError("ic must be :aic or :bic; got :$ic"))
    n, k = size(X)
    D, dnames = _cointreg_deter(n, trend, T)
    d = size(D, 2)
    Z = hcat(D, X)
    Xdelta = diff(X; dims=1)                          # (T-1)×k

    # Lead/lag selection.
    nlag = lags === :auto ? -1 : Int(lags)
    nlead = leads === :auto ? -1 : Int(leads)
    if nlag < 0 || nlead < 0
        snlag, snlead = _dols_select_leadlag(y, Z, Xdelta, k, ic)
        nlag < 0 && (nlag = snlag)
        nlead < 0 && (nlead = snlead)
    end
    (nlag ≥ 0 && nlead ≥ 0) || throw(ArgumentError("leads/lags must be ≥ 0"))

    # Augmented regression.
    Xfull, yt = _dols_design(y, Z, Xdelta, nlag, nlead)
    XtX = Symmetric(Xfull' * Xfull)
    XtX_inv = robust_inv(XtX)
    theta_all = Matrix{T}(XtX_inv) * (Xfull' * yt)
    u_dols = yt .- Xfull * theta_all                 # DOLS residuals
    theta_lr = theta_all[1:(d + k)]                  # long-run (level) coefficients

    # Bandwidth: cointReg resolves it from the univariate DOLS residual.
    ures = reshape(u_dols, :, 1)
    bw_num = bandwidth isa Symbol ?
        T(_resolve_bandwidth(ures, _lrv_kernel(kernel), bandwidth)) : T(bandwidth)

    # Long-run variance of the DOLS residual → LRV-corrected OLS covariance.
    ω_resid = lrvar(u_dols; kernel=kernel, bandwidth=bw_num, demean=false)
    vcov_full = dols_se === :robust ?
        newey_west(Xfull, u_dols; bandwidth=round(Int, bw_num), kernel=kernel) :
        ω_resid .* Matrix{T}(XtX_inv)
    vcov = Matrix{T}(vcov_full[1:(d + k), 1:(d + k)])

    # Stored stacked (u, Δx) long-run covariance for downstream tests (EV-11/EV-22).
    if nlag + nlead == 0
        u4 = u_dols[2:end]
        Δx_use = Xdelta
    else
        u4 = u_dols
        Δx_use = Xdelta[(nlag + 1):(size(Xdelta, 1) - nlead), :]
    end
    Ustore = hcat(u4, Δx_use)
    Ω, Λ, Σ = _cointreg_lrv(Ustore; kernel=kernel, bandwidth=bw_num)
    vidx = 2:(k + 1)
    Ω_uv = reshape(Ω[1, vidx], 1, k)
    Ω_vu = reshape(Ω[vidx, 1], k, 1)
    ω_uv = Ω[1, 1] - (Ω_uv*robust_inv(Ω[vidx, vidx])*Ω_vu)[1, 1]

    fitted = Z * theta_lr
    resid = y .- fitted
    varnames = vcat(dnames, _cointreg_xnames(k))

    return CointRegModel{T}(y, X, :dols, trend, kernel, bw_num, theta_lr, vcov,
                            resid, fitted, varnames, n, nlead, nlag, Ω, Λ, Σ, ω_uv, d, k)
end
