# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Multi-step conditional covariance forecasting for multivariate GARCH models.

- **CCC** — per-margin analytic GARCH variance forecasts with the fixed correlation `R`.
- **DCC** — margin variance forecasts combined with the Engle–Sheppard (2001) linear
  correlation approximation `E[Q_{T+k}] ≈ (1-(a+b)^{k-1})R̄ + (a+b)^{k-1}Q_{T+1}`.
- **BEKK** — deterministic iteration of the covariance recursion (`E[εε'] = H`).
"""

# =============================================================================
# Per-margin analytic GARCH variance forecast
# =============================================================================

"""
    _garch_var_forecast(g::GARCHModel, h) -> Vector

Deterministic h-step-ahead conditional-variance forecast for a univariate GARCH(p,q),
iterating `h_{T+k} = ω + Σαᵢ E[ε²_{T+k-i}] + Σβⱼ E[h_{T+k-j}]` with `E[ε²] = E[h]` for
future indices and actual values for indices ≤ T.
"""
function _garch_var_forecast(g::GARCHModel{T}, h::Int) where {T}
    q, p = g.q, g.p
    resid_sq = g.residuals .^ 2
    hist_h = g.conditional_variance
    n = length(hist_h)
    fc = Vector{T}(undef, h)
    for k in 1:h
        ht = g.omega
        for i in 1:q
            idx = n + k - i
            e2 = idx <= n ? resid_sq[idx] : fc[idx-n]  # future ε² ← forecast variance
            ht += g.alpha[i] * e2
        end
        for j in 1:p
            idx = n + k - j
            hh = idx <= n ? hist_h[idx] : fc[idx-n]
            ht += g.beta[j] * hh
        end
        fc[k] = max(ht, eps(T))
    end
    fc
end

# One-step-ahead DCC Q matrix Q_{T+1} (needed to seed the correlation forecast).
function _dcc_last_Q(z::AbstractMatrix{T}, a::T, b::T, Qbar::AbstractMatrix{T},
                     correction::Symbol) where {T}
    Tn, n = size(z)
    Q = Matrix{T}(Qbar)
    om = (one(T) - a - b)
    @inbounds for t in 1:Tn
        zt = @view z[t, :]
        if correction === :aielli
            qs = [sqrt(max(Q[i, i], eps(T))) * zt[i] for i in 1:n]
            outer = qs * qs'
        else
            outer = Vector{T}(zt) * Vector{T}(zt)'
        end
        Q = om .* Qbar .+ a .* outer .+ b .* Q
        Q = (Q + Q') ./ 2
    end
    Q
end

# =============================================================================
# forecast(::MGARCHModel, h)
# =============================================================================

"""
    forecast(m::MGARCHModel, h::Int) -> Array{T,3}

`h`-step-ahead conditional covariance forecasts, returned as an n×n×h array where
`forecast(m, h)[:,:,k]` is the predicted covariance at horizon `k`.

- CCC: analytic per-margin variance forecasts with the fixed correlation.
- DCC: margin variance forecasts with the standard linear correlation-forecast recursion.
- BEKK: deterministic covariance-recursion iteration (`E[εₜεₜ'] = Hₜ`).
"""
function forecast(m::MGARCHModel{T}, h::Int) where {T}
    h >= 1 || throw(ArgumentError("forecast horizon must be ≥ 1"))
    n = m.n
    Hf = Array{T,3}(undef, n, n, h)

    if m.kind === :bekk
        Sigbar = zeros(T, n, n)
        # Recover Σ̄ from the (constant) unconditional relationship: use sample cov of resid.
        resid = m.Y .- m.mu'
        Sigbar = Matrix{T}(cov(resid; corrected=false))
        Sigbar = (Sigbar + Sigbar') ./ 2
        Hlast = Matrix{T}(m.H[:, :, end])
        elast = Vector{T}(resid[end, :])
        if m.bekk_kind === :scalar
            a, b = m.params[1], m.params[2]
            for k in 1:h
                if k == 1
                    Hn = (one(T) - a - b) .* Sigbar .+ a .* (elast * elast') .+ b .* Hlast
                else
                    Hn = (one(T) - a - b) .* Sigbar .+ (a + b) .* Matrix{T}(Hf[:, :, k-1])
                end
                Hn = (Hn + Hn') ./ 2
                Hf[:, :, k] .= Hn
            end
        else
            avec = m.params[1:n]; bvec = m.params[n+1:2n]
            Ctil = Sigbar .* (ones(T, n, n) .- avec * avec' .- bvec * bvec')
            Ctil = (Ctil + Ctil') ./ 2
            for k in 1:h
                Hprev = k == 1 ? Hlast : Matrix{T}(Hf[:, :, k-1])
                if k == 1
                    Hn = Ctil .+ (avec .* elast) * (avec .* elast)' .+ (bvec * bvec') .* Hprev
                else
                    # E[A εε' A] = A Hprev A elementwise = (aa') ∘ Hprev
                    Hn = Ctil .+ (avec * avec') .* Hprev .+ (bvec * bvec') .* Hprev
                end
                Hn = (Hn + Hn') ./ 2
                Hf[:, :, k] .= Hn
            end
        end
        return Hf
    end

    # CCC / DCC: margin variance forecasts
    Dfc = Matrix{T}(undef, h, n)  # forecasted conditional sd
    for i in 1:n
        vfc = _garch_var_forecast(m.margins[i], h)
        @inbounds for k in 1:h
            Dfc[k, i] = sqrt(vfc[k])
        end
    end

    if m.kind === :ccc
        Rc = m.R isa Matrix ? m.R::Matrix{T} : Matrix{T}(m.R[:, :, end])
        for k in 1:h
            @inbounds for j in 1:n, i in 1:n
                Hf[i, j, k] = Dfc[k, i] * Rc[i, j] * Dfc[k, j]
            end
        end
        return Hf
    end

    # DCC
    z = Matrix{T}(undef, size(m.Y, 1), n)
    for i in 1:n
        z[:, i] .= m.margins[i].standardized_residuals
    end
    Qbar = _uncentered_Q(z)
    a, b = m.params[1], m.params[2]
    Q1 = _dcc_last_Q(z, a, b, Qbar, m.correction)  # Q_{T+1}
    ab = a + b
    Rbar = m.Rbar
    for k in 1:h
        if k == 1
            Qk = Q1
        else
            w = ab^(k - 1)
            Qk = (one(T) - w) .* Rbar .+ w .* Q1
        end
        Rk = _corr_from_cov(Qk)
        @inbounds for j in 1:n, i in 1:n
            Hf[i, j, k] = Dfc[k, i] * Rk[i, j] * Dfc[k, j]
        end
    end
    Hf
end
