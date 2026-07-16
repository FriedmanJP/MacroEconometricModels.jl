# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
MIDAS estimation: frequency alignment (`_align_hf`) and the NLS/OLS estimator
`estimate_midas`.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Frequency alignment
# =============================================================================

"""
    _align_hf(y_lf, X_hf; m, K) -> (Xlags, retained)

Build, for each low-frequency period `t`, the stacked high-frequency regressor
block `[x_{t,m}, x_{t,m−1}, …, x_{t,m−K+1}]` (most-recent-first). The last
low-frequency observation is anchored to the last high-frequency observation, so
any leading ragged edge (incomplete early blocks) is dropped automatically.

Returns the `n×K` lag matrix `Xlags` and the vector of retained low-frequency
indices `retained` (`1 ≤ t ≤ length(y_lf)`).
"""
function _align_hf(y_lf::AbstractVector{T}, X_hf::AbstractVector{T};
                   m::Int, K::Int) where {T<:AbstractFloat}
    m >= 1 || throw(ArgumentError("m must be ≥ 1 (got m=$m)"))
    K >= 1 || throw(ArgumentError("K must be ≥ 1 (got K=$K)"))
    T_lf = length(y_lf)
    len_hf = length(X_hf)
    retained = Int[]
    rows = Vector{Vector{T}}()
    for t in 1:T_lf
        hi = len_hf - (T_lf - t) * m          # HF index of most-recent obs in LF period t
        lo = hi - K + 1
        (lo >= 1 && hi <= len_hf) || continue
        push!(retained, t)
        push!(rows, collect(X_hf[hi:-1:lo]))  # most-recent-first
    end
    isempty(retained) && throw(ArgumentError(
        "no complete high-frequency blocks: need ≥ $K HF obs before a low-frequency period"))
    Xlags = Matrix{T}(undef, length(retained), K)
    @inbounds for i in eachindex(rows)
        Xlags[i, :] = rows[i]
    end
    return Xlags, retained
end

# =============================================================================
# AR-lag block
# =============================================================================

"""
    _ar_block(y_lf, retained, p_ar) -> (Wlin, keep, y_used)

Assemble the linear block `[1, y_{t-1}, …, y_{t-p_ar}]` for the retained
low-frequency periods that also have `p_ar` available own-lags. Returns the
`n×(1+p_ar)` matrix, the sub-index into `retained` that was kept, and the
aligned target vector.
"""
function _ar_block(y_lf::AbstractVector{T}, retained::Vector{Int}, p_ar::Int) where {T<:AbstractFloat}
    keep = Int[]
    for (i, t) in enumerate(retained)
        t - p_ar >= 1 && push!(keep, i)
    end
    isempty(keep) && throw(ArgumentError("no periods with $p_ar autoregressive lags available"))
    n = length(keep)
    Wlin = Matrix{T}(undef, n, 1 + p_ar)
    y_used = Vector{T}(undef, n)
    @inbounds for (r, i) in enumerate(keep)
        t = retained[i]
        Wlin[r, 1] = one(T)
        for j in 1:p_ar
            Wlin[r, 1 + j] = y_lf[t - j]
        end
        y_used[r] = y_lf[t]
    end
    return Wlin, keep, y_used
end

# =============================================================================
# Profiled SSR objective (concentrated linear params)
# =============================================================================

# Given θ, build s = Xlags·w(θ), design M = [Wlin  s], solve β̂ = M \ y by OLS.
# Returns (ssr, beta, resid, s, w).
function _midas_profile(theta::AbstractVector{T}, Xlags::Matrix{T}, Wlin::Matrix{T},
                        y::Vector{T}, K::Int, kind::Symbol) where {T<:AbstractFloat}
    w = _midas_weights(theta, K, kind)
    s = Xlags * w
    # design: intercept & AR lags in Wlin (col 1 = const), s inserted as β₁ column.
    # Ordering: [const, s, AR…] to match varnames [β₀, β₁, ρ…].
    M = hcat(Wlin[:, 1], s, Wlin[:, 2:end])
    beta = M \ y
    resid = y - M * beta
    ssr = dot(resid, resid)
    return ssr, beta, resid, s, w, M
end

# =============================================================================
# Estimator
# =============================================================================

"""
    estimate_midas(y_lf, X_hf; m, K, weights=:expalmon, p_ar=0, poly_degree=2,
                   h=1, max_iter=500) -> MidasModel{T}

Estimate a (restricted) MIDAS / ADL-MIDAS regression of the low-frequency target
`y_lf` on `K` high-frequency lags of the single indicator `X_hf`, aggregated
through the weight function `weights`.

# Arguments
- `y_lf::AbstractVector` — low-frequency target.
- `X_hf::AbstractVector` — high-frequency indicator (chronological; the last obs
  aligns to the last target period). Multi-indicator MIDAS is future work.

# Keywords
- `m::Int` — high/low frequency ratio (3 = monthly→quarterly, ≈66 = daily→quarterly).
- `K::Int` — number of high-frequency lags.
- `weights::Symbol` — `:expalmon` (default), `:beta2`, `:beta3`, `:almon`, `:umidas`.
- `p_ar::Int` — autoregressive lags of the target (ADL-MIDAS).
- `poly_degree::Int` — polynomial degree for `:almon`.
- `h::Int` — direct forecast horizon the model targets (stored for `forecast`).
- `max_iter::Int` — LBFGS iteration cap per start.

# Method
Restricted MIDAS is nonlinear least squares: the linear coefficients `(β₀,β₁,ρ)`
are concentrated out by OLS given `θ`, and the profiled SSR is minimized over
`θ` with `Optim.LBFGS` from a documented multi-start grid (the profiled surface
has flat ridges). `:umidas` is plain OLS on the `K` stacked lags
(Foroni–Marcellino–Schumacher 2015). Standard errors use the Gauss–Newton
sandwich `robust_inv(Hermitian(J'J))·σ̂²`.

# References
- Ghysels, Sinko & Valkanov (2007). *Econometric Reviews* 26(1), 53-90.
- Foroni, Marcellino & Schumacher (2015). *JRSS-A* 178(1), 57-82.
"""
function estimate_midas(y_lf::AbstractVector, X_hf::AbstractVector;
                        m::Int, K::Int, weights::Symbol=:expalmon,
                        p_ar::Int=0, poly_degree::Int=2, h::Int=1,
                        max_iter::Int=500)
    T = float(promote_type(eltype(y_lf), eltype(X_hf), Float64))
    yv = convert(Vector{T}, collect(y_lf))
    xv = convert(Vector{T}, collect(X_hf))
    K >= 1 || throw(ArgumentError("K must be ≥ 1"))
    p_ar >= 0 || throw(ArgumentError("p_ar must be ≥ 0"))

    Xlags_all, retained = _align_hf(yv, xv; m=m, K=K)
    Wlin, keep, y = _ar_block(yv, retained, p_ar)
    Xlags = Xlags_all[keep, :]
    n = length(y)

    if weights === :umidas
        return _estimate_umidas(y, Xlags, Wlin, m, K, p_ar, h)
    end

    # ---- restricted MIDAS: NLS by profiling ----
    starts = _midas_theta_starts(weights, poly_degree)
    best_ssr = T(Inf)
    best_theta = convert(Vector{T}, starts[1])
    best_conv = false

    for st in starts
        theta0 = convert(Vector{T}, st)
        f = θ -> _midas_profile(θ, Xlags, Wlin, y, K, weights)[1]
        function g!(G, θ)
            ssr, beta, resid, s, w, M = _midas_profile(θ, Xlags, Wlin, y, K, weights)
            Jw = _midas_weights_jac(θ, K, weights)      # K×p
            b1 = beta[2]                                 # β₁ multiplies the s-column
            XJ = Xlags * Jw                              # n×p : ∂s/∂θ
            # ∂SSR/∂θ = -2 β₁ · rᵀ (Xlags·Jw)
            G .= (-T(2) * b1) .* (XJ' * resid)
            return G
        end
        local res
        try
            res = Optim.optimize(f, g!, theta0, Optim.LBFGS(),
                                 Optim.Options(iterations=max_iter, g_tol=1e-8,
                                               show_trace=false))
        catch
            continue
        end
        ssr_st = Optim.minimum(res)
        if isfinite(ssr_st) && ssr_st < best_ssr
            best_ssr = ssr_st
            best_theta = convert(Vector{T}, Optim.minimizer(res))
            best_conv = Optim.converged(res)
        end
    end

    isfinite(best_ssr) || throw(ErrorException("MIDAS NLS failed to converge from any start"))

    ssr, beta, resid, s, w, M = _midas_profile(best_theta, Xlags, Wlin, y, K, weights)
    theta = best_theta

    # ---- Gauss-Newton SEs on full parameter vector [β; θ] ----
    # f_i(φ) = M_i·β ; ∂f/∂β = M, ∂f/∂θ_l = β₁·(Xlags·Jw)[:,l]
    Jw = _midas_weights_jac(theta, K, weights)
    Jtheta = beta[2] .* (Xlags * Jw)              # n×p
    Jfull = hcat(M, Jtheta)                        # n×(p_lin+p_theta)
    p = size(Jfull, 2)
    dofres = max(n - p, 1)
    sigma2 = ssr / T(dofres)
    vcov = Matrix{T}(robust_inv(Hermitian(Jfull' * Jfull)) .* sigma2)

    fitted = M * beta
    return _finalize_midas(y, Xlags, Wlin, theta, beta, vcov, weights, m, K,
                           p_ar, poly_degree, h, w, fitted, resid, ssr, sigma2,
                           best_conv, n, p)
end

# --- U-MIDAS: plain OLS on the K stacked lags (+ AR) --------------------------
function _estimate_umidas(y::Vector{T}, Xlags::Matrix{T}, Wlin::Matrix{T},
                          m::Int, K::Int, p_ar::Int, h::Int) where {T<:AbstractFloat}
    n = length(y)
    # design: [const, x_lag1…x_lagK, AR…]
    M = hcat(Wlin[:, 1], Xlags, Wlin[:, 2:end])
    beta = M \ y
    resid = y - M * beta
    ssr = dot(resid, resid)
    p = size(M, 2)
    dofres = max(n - p, 1)
    sigma2 = ssr / T(dofres)
    vcov = Matrix{T}(robust_inv(Hermitian(M' * M)) .* sigma2)
    fitted = M * beta
    # realized "weight curve" = the K unrestricted lag coefficients
    w = beta[2:(1 + K)]
    return _finalize_midas(y, Xlags, Wlin, T[], beta, vcov, :umidas, m, K,
                           p_ar, 0, h, w, fitted, resid, ssr, sigma2, true, n, p)
end

# --- shared finalizer: names, R², IC, construct MidasModel --------------------
function _finalize_midas(y::Vector{T}, Xlags::Matrix{T}, Wlin::Matrix{T},
                         theta::Vector{T}, beta::Vector{T}, vcov::Matrix{T},
                         kind::Symbol, m::Int, K::Int, p_ar::Int, poly_degree::Int,
                         h::Int, w::Vector{T}, fitted::Vector{T}, resid::Vector{T},
                         ssr::T, sigma2::T, converged::Bool, n::Int, p::Int) where {T<:AbstractFloat}
    ybar = mean(y)
    tss = sum(abs2, y .- ybar)
    r2 = tss > 0 ? one(T) - ssr / tss : zero(T)
    adj_r2 = n > p ? one(T) - (one(T) - r2) * T(n - 1) / T(n - p) : r2
    loglik = -T(0.5) * n * (log(T(2π)) + log(ssr / n) + one(T))
    aic = T(2) * p - T(2) * loglik
    bic = T(log(n)) * p - T(2) * loglik

    varnames = String[]
    if kind === :umidas
        push!(varnames, "const")
        for k in 1:K
            push!(varnames, "HF lag $k")
        end
        for j in 1:p_ar
            push!(varnames, "AR($j)")
        end
    else
        push!(varnames, "const")
        push!(varnames, "β₁ (HF loading)")
        for j in 1:p_ar
            push!(varnames, "AR($j)")
        end
        for l in 1:length(theta)
            push!(varnames, "θ$l")
        end
    end

    return MidasModel{T}(y, Xlags, Wlin, theta, beta, vcov, kind, m, K, p_ar,
                         poly_degree, h, w, fitted, resid, ssr, sigma2, r2,
                         adj_r2, loglik, aic, bic, varnames, converged)
end

# =============================================================================
# Display
# =============================================================================

const _MIDAS_KIND_LABEL = Dict(
    :expalmon => "Exponential Almon",
    :beta2    => "Beta (2-param)",
    :beta3    => "Beta (3-param)",
    :almon    => "Polynomial Almon",
    :umidas   => "U-MIDAS (unrestricted)",
)

function Base.show(io::IO, m::MidasModel{T}) where {T}
    n = nobs(m)
    p_lin = length(m.beta)
    p = dof(m)
    kind_str = get(_MIDAS_KIND_LABEL, m.weights_kind, string(m.weights_kind))

    spec = Any[
        "Weight scheme"  kind_str;
        "Frequency (m)"  m.m;
        "HF lags (K)"    m.K;
        "AR lags"        m.p_ar;
        "Horizon (h)"    m.h;
        "Observations"   n;
        "Parameters"     p;
        "R-squared"      _fmt(m.r2);
        "Adj. R-sq."     _fmt(m.adj_r2);
        "SSR"            _fmt(m.ssr; digits=4);
        "AIC"            _fmt(m.aic; digits=2);
        "BIC"            _fmt(m.bic; digits=2);
        "Converged"      (m.converged ? "Yes" : "No")
    ]
    _pretty_table(io, spec;
        title = "MIDAS Regression",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    se = stderror(m)
    dofr = dof_residual(m)

    lin_names = m.varnames[1:p_lin]
    _coef_table(io, "Coefficients", lin_names, m.beta, se[1:p_lin];
                dist = :t, dof_r = max(dofr, 1))

    if !isempty(m.theta)
        th_names = m.varnames[(p_lin + 1):end]
        _coef_table(io, "Weight parameters (θ)", th_names, m.theta,
                    se[(p_lin + 1):end]; dist = :t, dof_r = max(dofr, 1))
    end
    _sig_legend(io)
end

report(m::MidasModel) = show(stdout, m)
