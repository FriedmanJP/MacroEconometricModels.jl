# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
GARCH-MIDAS estimation (Engle, Ghysels & Sohn 2013): a mixed-frequency
volatility model splitting the conditional variance into a short-run unit-mean
GARCH(1,1) component and a long-run MIDAS-filtered component driven by a
low-frequency macro or realized-variance series.

Reuses the shared volatility helpers (`_volatility_negloglik`,
`_volatility_loglik_contribs`, `_qmle_sandwich_cov`, `_numerical_hessian`,
`_compute_aic_bic`, `_sanitize_init_params`) and EV-01's Beta MIDAS weight
helper `_midas_weights(θ, K, :beta2)` (imported — not re-derived).
"""

import Optim

# =============================================================================
# MIDAS design: build the retained-sample lagged-regressor matrix Z (n_ret × K)
# =============================================================================

"""
    _garch_midas_design(r, x_lf, K, m_freq, rv, span) -> (Z, ret_idx, n_blocks)

Build the MIDAS regressor design for a GARCH-MIDAS fit. Each **retained**
high-frequency observation `i` gets a length-`K` row `Z[i,:] = [X_{lag1}, …,
X_{lagK}]` (most-recent-first) so that `τ = exp(m + θ·(Z·φ))`.

- `span=:fixed` — one long-run value per calendar block: `X_t` is `x_lf[t]`
  (`rv=:macro`) or the block realized variance `Σ_{i∈block t} r_i²`
  (`rv=:realized`). Block `t` uses lags `X_{t-1},…,X_{t-K}`; the first `K` blocks
  are dropped (ragged edge).
- `span=:rolling` — long-run varies every HF obs via a trailing rolling realized
  variance `RV_i = Σ_{j=0}^{m_freq-1} r_{i-j}²`, with MIDAS lags spaced by
  `m_freq`: `Z[i,:] = [RV_{i-m_freq},…,RV_{i-K·m_freq}]`. The first `(K+1)·m_freq`
  HF obs are dropped.

Returns the `Float64` design `Z`, the retained HF indices `ret_idx` (into `r`),
and the number of complete blocks `n_blocks`.
"""
function _garch_midas_design(r::Vector{Float64}, x_lf::Vector{Float64}, K::Int,
                             m_freq::Int, rv::Symbol, span::Symbol)
    n_hf = length(r)
    n_blocks = fld(n_hf, m_freq)
    n_blocks > K + 1 ||
        throw(ArgumentError("need > K+1 = $(K+1) complete blocks of length m_freq=$m_freq; " *
                            "got $n_blocks (n_hf=$n_hf)"))

    if span === :fixed
        # per-block low-frequency driver X_t, t = 1..n_blocks
        X = Vector{Float64}(undef, n_blocks)
        if rv === :macro
            length(x_lf) >= n_blocks ||
                throw(ArgumentError("rv=:macro needs length(x_lf) ≥ n_blocks=$n_blocks, got $(length(x_lf))"))
            @inbounds for t in 1:n_blocks
                X[t] = x_lf[t]
            end
        elseif rv === :realized
            @inbounds for t in 1:n_blocks
                s = 0.0
                for i in ((t-1)*m_freq+1):(t*m_freq)
                    s += r[i]^2
                end
                X[t] = s
            end
        else
            throw(ArgumentError("rv must be :macro or :realized, got :$rv"))
        end

        ret_idx = Int[]
        rows = Vector{Vector{Float64}}()
        @inbounds for t in (K+1):n_blocks
            xlag = [X[t-k] for k in 1:K]           # most-recent-first
            for i in ((t-1)*m_freq+1):(t*m_freq)
                push!(ret_idx, i)
                push!(rows, xlag)
            end
        end
        Z = Matrix{Float64}(undef, length(ret_idx), K)
        @inbounds for (row, xr) in enumerate(rows)
            Z[row, :] .= xr
        end
        return Z, ret_idx, n_blocks

    elseif span === :rolling
        # trailing rolling realized variance at each HF obs
        RV = fill(NaN, n_hf)
        @inbounds for i in m_freq:n_hf
            s = 0.0
            for j in 0:(m_freq-1)
                s += r[i-j]^2
            end
            RV[i] = s
        end
        start = (K + 1) * m_freq          # smallest i with all K lags (RV_{i-K·m_freq}) valid
        start < n_hf ||
            throw(ArgumentError("rolling span needs n_hf > (K+1)·m_freq = $start, got $n_hf"))
        ret_idx = collect((start + 1):n_hf)
        Z = Matrix{Float64}(undef, length(ret_idx), K)
        @inbounds for (row, i) in enumerate(ret_idx)
            for k in 1:K
                Z[row, k] = RV[i - k*m_freq]
            end
        end
        return Z, ret_idx, n_blocks
    else
        throw(ArgumentError("span must be :fixed or :rolling, got :$span"))
    end
end

# =============================================================================
# Core filter (generic in T so ForwardDiff Duals propagate)
# =============================================================================

"""
    _garch_midas_filter(mu, alpha, beta, m, theta, w, r_ret, Z, K)
        -> (tau, g, sigma2)

Compute the long-run `τ`, short-run unit-mean `g`, and total variance `σ² = τ·g`
over the retained sample `r_ret` given the natural parameters and MIDAS design
`Z` (`n_ret × K`). The short-run recursion divides the lagged squared residual by
`τ` (the `√τ` innovation scaling), initialising `g[1]=1`.
"""
function _garch_midas_filter(mu::T, alpha::T, beta::T, m::T, theta::T, w::T,
                             r_ret::AbstractVector, Z::AbstractMatrix, K::Int) where {T}
    n = length(r_ret)
    phi = _midas_weights([one(T), w], K, :beta2)        # length K, sums to 1
    mp = Z * phi                                         # Σ_k φ_k X_{lag k}
    z = m .+ theta .* mp
    z = clamp.(z, T(-50), T(50))                         # guard exp overflow
    tau = exp.(z)
    resid = T.(r_ret) .- mu
    g = Vector{T}(undef, n)
    g[1] = one(T)
    omega_g = one(T) - alpha - beta
    @inbounds for i in 2:n
        g[i] = omega_g + alpha * resid[i-1]^2 / tau[i-1] + beta * g[i-1]
    end
    sigma2 = tau .* g
    return tau, g, sigma2
end

# =============================================================================
# (Negative) log-likelihood
# =============================================================================

# optimization-space params: [μ, log α, log β, m, θ, w̃]  with  w = 1 + exp(w̃)
function _garch_midas_unpack(p::AbstractVector{T}) where {T}
    mu    = p[1]
    alpha = exp(p[2])
    beta  = exp(p[3])
    m     = p[4]
    theta = p[5]
    w     = one(T) + exp(p[6])
    return mu, alpha, beta, m, theta, w
end

function _garch_midas_negloglik(p::AbstractVector{T}, r_ret::Vector{T},
                                Z::Matrix{T}, K::Int) where {T}
    mu, alpha, beta, m, theta, w = _garch_midas_unpack(p)
    (alpha + beta) >= one(T) && return T(1e10)          # short-run stationarity
    _, _, sigma2 = _garch_midas_filter(mu, alpha, beta, m, theta, w, r_ret, Z, K)
    resid = r_ret .- mu
    _volatility_negloglik(sigma2, resid .^ 2, length(r_ret))
end

# per-observation Gaussian log-lik contributions (no stationarity guard — evaluated
# at the stationary optimum). Generic so ForwardDiff.jacobian gives the n×6 score S.
function _garch_midas_loglik_contribs(p, r_ret, Z, K::Int)
    mu, alpha, beta, m, theta, w = _garch_midas_unpack(p)
    _, _, sigma2 = _garch_midas_filter(mu, alpha, beta, m, theta, w, r_ret, Z, K)
    resid = r_ret .- mu
    _volatility_loglik_contribs(sigma2, resid .^ 2)
end

# =============================================================================
# Estimation
# =============================================================================

"""
    estimate_garch_midas(r, x_lf; K=12, m_freq, rv=:macro, span=:fixed) -> GarchMidasModel

Estimate a GARCH-MIDAS model (Engle, Ghysels & Sohn 2013) by Gaussian QMLE.

The conditional variance is `σ²_{i,t} = τ_t · g_{i,t}` with a unit-mean short-run
GARCH(1,1) `g` on the τ-standardized return and a long-run MIDAS component
`τ_t = exp(m + θ Σ_{k=1}^K φ_k(w) X_{t-k})` (Beta weights, monotone decaying).

# Arguments
- `r`: high-frequency return series.
- `x_lf`: low-frequency driver, one value per block (`rv=:macro`). Ignored — pass
  `Float64[]` — for `rv=:realized`.

# Keywords
- `K`: number of MIDAS lags (default 12).
- `m_freq`: high-to-low frequency ratio, i.e. HF observations per LF block (required).
- `rv`: `:macro` (exogenous `x_lf`) or `:realized` (realized variance from `r`).
- `span`: `:fixed` (calendar-block τ) or `:rolling` (rolling-RV τ, `rv` ignored).

# Example
```julia
m = estimate_garch_midas(r, x_lf; K=12, m_freq=22)
report(m)
m.variance_ratio            # long-run share of total variance variation
```
"""
function estimate_garch_midas(r::AbstractVector, x_lf::AbstractVector;
                              K::Int=12, m_freq::Int,
                              rv::Symbol=:macro, span::Symbol=:fixed)
    K >= 2 || throw(ArgumentError("K must be ≥ 2 (Beta MIDAS weights), got K=$K"))
    m_freq >= 1 || throw(ArgumentError("m_freq must be ≥ 1, got $m_freq"))
    r_full = Vector{Float64}(r)
    x_full = Vector{Float64}(x_lf)
    length(r_full) >= 20 || throw(ArgumentError("need at least 20 observations, got $(length(r_full))"))

    Z, ret_idx, n_blocks = _garch_midas_design(r_full, x_full, K, m_freq, rv, span)
    r_ret = r_full[ret_idx]
    n_ret = length(r_ret)

    # --- initial parameters (optimization space) --------------------------------
    mu0 = mean(r_ret)
    v0 = var(r_ret .- mu0; corrected=false)
    a0, b0 = 0.05, 0.85
    m0 = log(max(v0, eps()))       # θ starts at 0 ⇒ τ ≈ exp(m0) ≈ variance level
    th0 = 0.0
    wt0 = log(2.0)                 # w = 1 + exp(w̃) = 3 (moderate decay)
    p0 = _sanitize_init_params([mu0, log(a0), log(b0), m0, th0, wt0])

    obj = p -> _garch_midas_negloglik(p, r_ret, Z, K)
    res1 = Optim.optimize(obj, p0, Optim.NelderMead(),
                          Optim.Options(iterations=3000, show_trace=false))
    # The long-run MIDAS weight `w` sits on a near-flat ridge, so LBFGS's gradient
    # criterion can stall just shy of tolerance; accept an f/x-based stationary point
    # too so a genuinely-flat direction still counts as converged.
    lbfgs_opts = Optim.Options(iterations=1000, g_tol=1e-7,
                               f_reltol=1e-11, x_reltol=1e-10, show_trace=false)
    res2 = Optim.optimize(obj, Optim.minimizer(res1), Optim.LBFGS(), lbfgs_opts)
    res = Optim.optimize(obj, Optim.minimizer(res2), Optim.LBFGS(), lbfgs_opts)
    converged = Optim.converged(res) || Optim.converged(res2)

    p_opt = Optim.minimizer(res)
    mu, alpha, beta, m_const, theta, w = _garch_midas_unpack(p_opt)

    tau, g, sigma2 = _garch_midas_filter(mu, alpha, beta, m_const, theta, w, r_ret, Z, K)
    phi = _midas_weights([1.0, w], K, :beta2)
    resid = r_ret .- mu
    std_resid = resid ./ sqrt.(sigma2)

    # variance ratio VR = Var(log τ) / Var(log σ²)
    log_tau = log.(tau)
    log_sig = log.(sigma2)
    vr = var(log_sig; corrected=false) > 0 ?
         var(log_tau; corrected=false) / var(log_sig; corrected=false) : 0.0

    loglik = -Optim.minimum(res)
    aic_val, bic_val = _compute_aic_bic(loglik, 6, n_ret)

    # QMLE sandwich covariance in optimization space (evaluated at the optimum)
    param_vcov = try
        H = _numerical_hessian(obj, p_opt)
        S = ForwardDiff.jacobian(θ_ -> _garch_midas_loglik_contribs(θ_, r_ret, Z, K), p_opt)
        Matrix{Float64}(_qmle_sandwich_cov(H, S))
    catch
        fill(NaN, 6, 6)
    end

    GarchMidasModel{Float64}(r_full, x_full, mu, alpha, beta, m_const, theta, w,
                             phi, tau, g, sigma2, ret_idx, resid, std_resid, vr,
                             K, m_freq, n_blocks, rv, span, loglik, aic_val, bic_val,
                             :qmle, converged, Optim.iterations(res), param_vcov)
end

# convenience: realized-variance driver (no exogenous x_lf needed)
"""
    estimate_garch_midas(r; K=12, m_freq, rv=:realized, span=:fixed) -> GarchMidasModel

Convenience method with the long-run component driven by realized variance
computed from `r` itself (no exogenous low-frequency series).
"""
function estimate_garch_midas(r::AbstractVector; K::Int=12, m_freq::Int,
                              rv::Symbol=:realized, span::Symbol=:fixed)
    estimate_garch_midas(r, Float64[]; K=K, m_freq=m_freq, rv=rv, span=span)
end

# =============================================================================
# Standard errors (delta method: elementwise transform → natural)
# =============================================================================

"""
    StatsAPI.stderror(m::GarchMidasModel{T}; cov_type=:robust) -> Vector{T}

Standard errors for `coef(m) = [μ, α, β, m, θ, w]` via the delta method on the
elementwise optimization→natural transform (`α=exp·`, `β=exp·`, `w=1+exp·`).

`cov_type`: `:robust` (default) uses the cached Bollerslev–Wooldridge QMLE
sandwich `H⁻¹(S'S)H⁻¹`; `:hessian` uses inverse observed information `H⁻¹`.

!!! note
    Until backlog #173 lands a first-class Bollerslev–Wooldridge routine, the
    `:robust` covariance is the numerically-differentiated sandwich computed at
    estimation time.
"""
function StatsAPI.stderror(m::GarchMidasModel{T}; cov_type::Symbol=:robust) where {T}
    cov_type in (:robust, :qmle, :sandwich, :bw, :hessian, :opg_hessian) ||
        throw(ArgumentError("cov_type must be :robust or :hessian, got :$cov_type"))

    p_opt = [m.mu, log(m.alpha), log(m.beta), m.m_const, m.theta, log(m.w - one(T))]

    C_opt = if cov_type in (:robust, :qmle, :sandwich, :bw) && all(isfinite, m.param_vcov)
        m.param_vcov
    else
        Z, ret_idx, _ = _garch_midas_design(Vector{Float64}(m.y), Vector{Float64}(m.x_lf),
                                             m.K, m.m_freq, m.rv, m.span)
        r_ret = Vector{Float64}(m.y[ret_idx])
        obj = p -> _garch_midas_negloglik(p, r_ret, Z, m.K)
        try
            H = _numerical_hessian(obj, p_opt)
            if cov_type in (:robust, :qmle, :sandwich, :bw)
                S = ForwardDiff.jacobian(θ_ -> _garch_midas_loglik_contribs(θ_, r_ret, Z, m.K), p_opt)
                _qmle_sandwich_cov(H, S)
            else
                robust_inv(H)
            end
        catch
            return fill(T(NaN), 6)
        end
    end

    d = sqrt.(max.(diag(C_opt), zero(T)))          # SEs in optimization space
    # elementwise Jacobian |∂natural/∂transformed|
    se = Vector{T}(undef, 6)
    se[1] = d[1]                       # μ  (identity)
    se[2] = m.alpha * d[2]             # α = exp(·)
    se[3] = m.beta  * d[3]             # β = exp(·)
    se[4] = d[4]                       # m  (identity)
    se[5] = d[5]                       # θ  (identity)
    se[6] = (m.w - one(T)) * d[6]      # w = 1 + exp(·)
    se
end

# =============================================================================
# Forecasting
# =============================================================================

"""
    forecast(m::GarchMidasModel, h) -> NamedTuple

Iterate the GARCH-MIDAS conditional variance `h` steps ahead. The short-run `g`
mean-reverts to 1 at rate `α+β`; the long-run `τ` is held at its last fitted
low-frequency value (extrapolating the macro block forward). Returns
`(total, long_run, short_run, horizon)` with the total (`τ·g`) and long-run
variance paths.
"""
function forecast(m::GarchMidasModel{T}, h::Int) where {T}
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))
    tau_last = m.tau[end]
    g_last = m.g[end]
    resid_last = m.residuals[end]
    ab = m.alpha + m.beta
    omega_g = one(T) - ab

    total = Vector{T}(undef, h)
    long_run = fill(tau_last, h)
    short_run = Vector{T}(undef, h)

    # one-step-ahead uses the realized last shock; thereafter g mean-reverts to 1
    g_prev = g_last
    g1 = omega_g + m.alpha * resid_last^2 / tau_last + m.beta * g_last
    for k in 1:h
        gk = k == 1 ? g1 : (one(T) + ab * (g_prev - one(T)))
        short_run[k] = gk
        total[k] = tau_last * gk
        g_prev = gk
    end
    (total=total, long_run=long_run, short_run=short_run, horizon=h)
end

StatsAPI.predict(m::GarchMidasModel, h::Int) = forecast(m, h).total

# =============================================================================
# Display
# =============================================================================

function _show_garch_midas(io::IO, m::GarchMidasModel{T}) where {T}
    se = try
        stderror(m)
    catch
        fill(T(NaN), 6)
    end
    # short-run block: μ, α, β
    sr_names = String["μ (mean)", "α (ARCH)", "β (GARCH)"]
    sr_vals = T[m.mu, m.alpha, m.beta]
    _coef_table(io, "GARCH-MIDAS ($(m.rv), span=$(m.span), K=$(m.K), m=$(m.m_freq))",
                sr_names, sr_vals, se[1:3]; dist=:z)
    # long-run / MIDAS block: m, θ, w
    lr_names = String["m (LR const)", "θ (MIDAS)", "w (Beta shape)"]
    lr_vals = T[m.m_const, m.theta, m.w]
    _coef_table(io, "Long-run (MIDAS) parameters", lr_names, lr_vals, se[4:6]; dist=:z)

    fit_data = Any[
        "HF observations"  length(m.y);
        "Retained obs"     length(m.ret_idx);
        "LF blocks"        m.n_blocks;
        "Persistence α+β"  _fmt(persistence(m));
        "Variance ratio"   _fmt(m.variance_ratio);
        "Log-likelihood"   _fmt(m.loglik; digits=4);
        "AIC"              _fmt(m.aic; digits=4);
        "BIC"              _fmt(m.bic; digits=4);
        "Converged"        _yesno(m.converged)
    ]
    _pretty_table(io, fit_data; column_labels=["Fit", "Value"], alignment=[:l, :r])
    _sig_legend(io)
end
