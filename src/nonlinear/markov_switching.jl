# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Markov-switching regression and mean-switching autoregression (Hamilton 1989),
estimated by the scaled Hamilton forward filter, the Kim (1994) backward
smoother, EM (Baum–Welch), and an `Optim` maximum-likelihood polish with
delta-method standard errors (EV-07 / #415).

Two public entry points:

- [`estimate_ms`](@ref) — a K-state switching regression `yₜ = xₜ'β_{sₜ} + εₜ`,
  `εₜ ~ N(0, σ²_{sₜ})`, filtered on `K` states.
- [`estimate_ms_ar`](@ref) — the Hamilton (1989) *mean-switching* AR
  `(yₜ − μ_{sₜ}) = Σⱼ φⱼ (y_{t−j} − μ_{s_{t−j}}) + εₜ`, filtered on the `Kᵖ⁺¹`
  expanded regime-path state space and marginalised back to `sₜ`.

This is a DISTINCT code path from `src/nongaussian/heteroskedastic.jl`
(`_hamilton_filter`, variance-regime SVAR identification) and from
`src/filters/hamilton.jl` (`hamilton_filter`, the 2018 detrending filter): here
the mean/level switches. The filter helper is named `_ms_hamilton_filter` to
avoid any symbol collision.

References
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle. *Econometrica* 57(2), 357–384.
- Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of
  Econometrics* 60(1–2), 1–22.
- Hamilton, J. D. (1994). *Time Series Analysis*, Ch. 22. Princeton Univ. Press.
"""

# =============================================================================
# Scaled Hamilton (1989) forward filter — generic over the number of states M
# =============================================================================

"""
    _ms_hamilton_filter(eta, P, xi0) -> (filtered, predicted, loglik)

Scaled Hamilton (1989) forward filter over `M` latent states. `eta` is the
`n × M` matrix of state-conditional densities `η_{t,m} = f(yₜ | sₜ=m, ℱ_{t-1})`,
`P` the `M × M` transition matrix (rows sum to 1), and `xi0` the initial state
distribution (typically the ergodic vector). Returns the filtered probabilities
`Pr(sₜ=m | ℱₜ)`, the one-step predicted probabilities `Pr(sₜ=m | ℱ_{t-1})`, and
the exact log-likelihood `Σₜ log Σₘ ξ_{t|t-1,m} η_{t,m}`.

Element-type-generic (works with `ForwardDiff.Dual` for gradient/Hessian passes).
"""
function _ms_hamilton_filter(eta::AbstractMatrix, P::AbstractMatrix,
                             xi0::AbstractVector)
    n, M = size(eta)
    S = promote_type(eltype(eta), eltype(P), eltype(xi0))
    filtered = Matrix{S}(undef, n, M)
    predicted = Matrix{S}(undef, n, M)
    loglik = zero(S)
    Pt = permutedims(P)                      # column k of Pt = row k of P
    xi_pred = Vector{S}(xi0)
    for t in 1:n
        predicted[t, :] = xi_pred
        joint = xi_pred .* @view eta[t, :]
        margin = sum(joint) + S(1e-300)      # guard against total underflow
        loglik += log(margin)
        f = joint ./ margin
        filtered[t, :] = f
        xi_pred = Pt * f
    end
    return filtered, predicted, loglik
end

"""
    _ms_kim_smoother(filtered, predicted, P) -> smoothed

Kim (1994) fixed-interval backward smoother. Given the Hamilton-filter output,
returns `Pr(sₜ=m | ℱ_T)` via
`ξ_{t|T} = ξ_{t|t} ⊙ Pᵀ (ξ_{t+1|T} ⊘ ξ_{t+1|t})`, row-normalised for numerical
safety. Run at the estimated parameters only (concrete float type).
"""
function _ms_kim_smoother(filtered::AbstractMatrix{T}, predicted::AbstractMatrix{T},
                          P::AbstractMatrix{T}) where {T<:AbstractFloat}
    n, M = size(filtered)
    smoothed = similar(filtered)
    smoothed[n, :] = filtered[n, :]
    for t in (n - 1):-1:1
        for i in 1:M
            s = zero(T)
            for j in 1:M
                pj = max(predicted[t + 1, j], eps(T))
                s += P[i, j] * smoothed[t + 1, j] / pj
            end
            smoothed[t, i] = filtered[t, i] * s
        end
        tot = sum(@view smoothed[t, :])
        tot > 0 && (smoothed[t, :] ./= tot)
    end
    return smoothed
end

"""
    _ms_ergodic(P) -> Vector

Ergodic (stationary) distribution `π` of the transition matrix `P` (`π'P = π'`,
`Σπ = 1`), computed as the least-squares solution of the constrained linear
system `[P'−I; 1'] π = [0; 1]`, projected onto the simplex. Element-type-generic.
"""
function _ms_ergodic(P::AbstractMatrix)
    S = eltype(P)
    K = size(P, 1)
    A = vcat(permutedims(P) - Matrix{S}(I, K, K), ones(S, 1, K))
    b = vcat(zeros(S, K), one(S))
    xi = try
        (permutedims(A) * A) \ (permutedims(A) * b)
    catch
        fill(one(S) / K, K)
    end
    xi = max.(xi, zero(S))
    tot = sum(xi)
    return tot > 0 ? xi ./ tot : fill(one(S) / K, K)
end

# =============================================================================
# Transition-matrix parameterisation (row-wise softmax, last column = reference)
# =============================================================================

"""
    _logits_to_P(l, K) -> Matrix

Map `K·(K−1)` unconstrained logits to a `K × K` row-stochastic transition matrix.
Row `i` uses logits `l_{i,1..K-1}` with the last column as the reference:
`P[i,j] = exp(l_{ij}) / (1 + Σ exp(l))` for `j<K`, `P[i,K] = 1/(1 + Σ exp(l))`.
"""
function _logits_to_P(l::AbstractVector, K::Int)
    S = eltype(l)
    P = Matrix{S}(undef, K, K)
    for i in 1:K
        row = @view l[((i - 1) * (K - 1) + 1):(i * (K - 1))]
        ex = exp.(row)
        denom = one(S) + sum(ex)
        for j in 1:(K - 1)
            P[i, j] = ex[j] / denom
        end
        P[i, K] = one(S) / denom
    end
    return P
end

"""
    _P_to_logits(P, K) -> Vector

Inverse of [`_logits_to_P`](@ref): `l_{ij} = log(P[i,j] / P[i,K])`, with a small
floor on the reference column for stability.
"""
function _P_to_logits(P::AbstractMatrix{T}, K::Int) where {T}
    l = Vector{T}(undef, K * (K - 1))
    for i in 1:K
        ref = max(P[i, K], eps(T))
        for j in 1:(K - 1)
            l[(i - 1) * (K - 1) + j] = log(max(P[i, j], eps(T)) / ref)
        end
    end
    return l
end

# =============================================================================
# Expanded regime-path state space for the mean-switching AR (Hamilton 1989)
# =============================================================================

"""
    _expand_states(K, p) -> (states, allowed)

Enumerate the `M = Kᵖ⁺¹` regime paths `(sₜ, s_{t-1}, …, s_{t-p})`.

- `states::Matrix{Int}` is `M × (p+1)`; `states[a, d]` is the regime in the
  `d`-th path slot (slot 1 = current `sₜ`), with slot-1 varying fastest.
- `allowed::Matrix{Bool}` is `M × M`; `allowed[a,b]` is `true` when path `a` can
  transition to path `b` (the history shifts: `states[b, 2:end] == states[a, 1:end-1]`),
  in which case the expanded transition probability is `P[states[a,1], states[b,1]]`.
"""
function _expand_states(K::Int, p::Int)
    M = K^(p + 1)
    states = Matrix{Int}(undef, M, p + 1)
    for a in 0:(M - 1)
        for d in 1:(p + 1)
            states[a + 1, d] = (a ÷ K^(d - 1)) % K + 1
        end
    end
    allowed = falses(M, M)
    for a in 1:M, b in 1:M
        ok = true
        for d in 1:p
            if states[b, d + 1] != states[a, d]
                ok = false
                break
            end
        end
        allowed[a, b] = ok
    end
    return states, allowed
end

"""
    _expanded_P(P, states, allowed) -> Matrix

Lift the `K × K` regime transition matrix `P` to the `M × M` transition matrix of
the expanded regime-path chain: `Pexp[a,b] = P[states[a,1], states[b,1]]` when
`allowed[a,b]`, else `0`. Element-type-generic.
"""
function _expanded_P(P::AbstractMatrix, states::Matrix{Int}, allowed::BitMatrix)
    S = eltype(P)
    M = size(states, 1)
    Pexp = zeros(S, M, M)
    for a in 1:M, b in 1:M
        if allowed[a, b]
            Pexp[a, b] = P[states[a, 1], states[b, 1]]
        end
    end
    return Pexp
end

# =============================================================================
# Delta-method standard errors
# =============================================================================

"""
    _ms_delta_ses(nll, natmap, theta_hat) -> se

Delta-method standard errors of the natural parameters. `nll` is the negative
log-likelihood in the unconstrained parameter `θ`; `natmap` maps `θ` to the
vector of natural parameters. Returns `sqrt.(diag(J · H⁻¹ · Jᵀ))` where `H` is the
Hessian of `nll` at `θ̂` and `J = ∂natmap/∂θ`.
"""
function _ms_delta_ses(nll, natmap, theta_hat::Vector{T}) where {T}
    H = ForwardDiff.hessian(nll, theta_hat)
    cov_t = Matrix{T}(robust_inv(Hermitian(H); silent=true))
    J = ForwardDiff.jacobian(natmap, theta_hat)
    cov_nat = J * cov_t * permutedims(J)
    return sqrt.(max.(diag(cov_nat), zero(T)))
end

# =============================================================================
# estimate_ms — K-state switching regression (EM + ML polish)
# =============================================================================

"""
    estimate_ms(y, X; k_regimes=2, switching_variance=true, max_iter=500,
                tol=1e-8, xnames=nothing) -> MSRegModel
    estimate_ms(y; kwargs...) -> MSRegModel   # intercept-only design

Estimate a K-state Markov-switching regression
`yₜ = xₜ'β_{sₜ} + εₜ`, `εₜ ~ N(0, σ²_{sₜ})`, where every coefficient switches with
the latent regime `sₜ` (a `K`-state first-order Markov chain with transition
matrix `P`) and, when `switching_variance=true`, the variance switches too.

**Estimation.** EM (Baum–Welch): the E-step runs the scaled Hamilton (1989)
forward filter and the Kim (1994) smoother; the M-step re-fits each regime's
coefficients by weighted least squares (smoothed probabilities as weights),
updates the variances, and updates `P` from the Kim joint smoothed
probabilities. The EM optimum is polished by `Optim` (LBFGS on the exact
log-likelihood over the unconstrained parameterisation `[β; log σ²; transition
logits]`) and standard errors are the delta-method SEs from the ML Hessian.

**Labelling.** Regimes are ordered by increasing average fitted mean, so regime 1
is the lowest-mean state — deterministic across RNG seeds (defeats
label-switching).

# Arguments
- `y::AbstractVector`: dependent variable.
- `X::AbstractMatrix`: regressor matrix (`n × kx`); include a constant column for
  a switching intercept. The single-argument form uses `X = ones(n, 1)`.

# Keywords
- `k_regimes::Int`: number of regimes `K` (default `2`).
- `switching_variance::Bool`: switch `σ²` across regimes (default `true`).
- `max_iter::Int`, `tol`: EM iteration cap and log-likelihood tolerance.
- `xnames`: regressor labels.

Returns an [`MSRegModel`](@ref) with `model_type = :regression`.
"""
function estimate_ms(y::AbstractVector, X::AbstractMatrix; k_regimes::Int=2,
                     switching_variance::Bool=true, max_iter::Int=500,
                     tol::Real=1e-8, xnames=nothing)
    k_regimes >= 2 || throw(ArgumentError("k_regimes must be ≥ 2; got $k_regimes."))
    T = float(eltype(X))
    yv = Vector{T}(y)
    Xm = Matrix{T}(X)
    n, kx = size(Xm)
    length(yv) == n || throw(DimensionMismatch(
        "length(y)=$(length(yv)) must equal size(X,1)=$n."))
    K = k_regimes
    nσ = switching_variance ? K : 1
    (n > K * kx + nσ + K * (K - 1)) || throw(ArgumentError(
        "sample too small for a $K-regime switching regression with $kx regressors."))

    # --- EM initialisation: quantile bins of y for regime assignment ---
    B = Matrix{T}(undef, kx, K)
    sig2 = Vector{T}(undef, K)
    order = sortperm(yv)
    binsz = cld(n, K)
    for k in 1:K
        idx = order[((k - 1) * binsz + 1):min(k * binsz, n)]
        Xk = Xm[idx, :]; yk = yv[idx]
        bk = Matrix{T}(robust_inv(Hermitian(Xk' * Xk); silent=true)) * (Xk' * yk)
        rk = yk .- Xk * bk
        B[:, k] = bk
        sig2[k] = max(var(rk), T(1e-4))
    end
    P = fill(T(0.2) / (K - 1), K, K)
    for i in 1:K
        P[i, i] = T(0.8)
    end

    # --- EM loop ---
    ll_old = T(-Inf)
    converged = false
    iters = 0
    filtered = predicted = smoothed = Matrix{T}(undef, n, K)
    for it in 1:max_iter
        iters = it
        eta = _ms_reg_eta(yv, Xm, B, sig2, K)
        filtered, predicted, ll = _ms_hamilton_filter(eta, P, _ms_ergodic(P))
        smoothed = _ms_kim_smoother(filtered, predicted, P)
        if abs(ll - ll_old) < tol * (abs(ll_old) + one(T))
            converged = true
            break
        end
        ll_old = ll
        # M-step: WLS coefficients + variances.
        pooled_num = zero(T); pooled_den = zero(T)
        for k in 1:K
            w = max.(smoothed[:, k], eps(T))
            W = Diagonal(w)
            XtWX = Hermitian(Xm' * W * Xm)
            bk = Matrix{T}(robust_inv(XtWX; silent=true)) * (Xm' * (w .* yv))
            B[:, k] = bk
            rk = yv .- Xm * bk
            wr2 = sum(w .* rk .^ 2)
            sig2[k] = max(wr2 / sum(w), T(1e-8))
            pooled_num += wr2; pooled_den += sum(w)
        end
        if !switching_variance
            sig2 .= max(pooled_num / pooled_den, T(1e-8))
        end
        P = _ms_update_P(smoothed, filtered, predicted, P, K)
    end

    # --- ML polish over unconstrained θ = [vec(B); logσ²(nσ); logits] ---
    theta0 = vcat(vec(B), log.(switching_variance ? sig2 : sig2[1:1]),
                  _P_to_logits(P, K))
    nll = θ -> _ms_reg_nll(θ, yv, Xm, K, nσ, kx)
    theta_hat = theta0
    try
        g! = (G, θ) -> ForwardDiff.gradient!(G, nll, θ)
        res = Optim.optimize(nll, g!, theta0, Optim.LBFGS(),
                             Optim.Options(f_reltol=1e-10, g_tol=1e-7, iterations=500))
        cand = Optim.minimizer(res)
        nll(cand) <= nll(theta0) && (theta_hat = cand; converged = converged || Optim.converged(res))
    catch err
        err isa InterruptException && rethrow()
    end

    # --- Unpack, filter/smooth at θ̂, order regimes by mean ---
    B = reshape(theta_hat[1:(kx * K)], kx, K)
    off = kx * K
    sig2 = switching_variance ? exp.(theta_hat[(off + 1):(off + K)]) :
                                fill(exp(theta_hat[off + 1]), K)
    off += nσ
    P = _logits_to_P(theta_hat[(off + 1):end], K)

    means = T[mean(Xm * B[:, k]) for k in 1:K]
    perm = sortperm(means)
    B = B[:, perm]; sig2 = sig2[perm]; means = means[perm]
    P = P[perm, perm]

    eta = _ms_reg_eta(yv, Xm, B, sig2, K)
    filtered, predicted, loglik = _ms_hamilton_filter(eta, P, _ms_ergodic(P))
    smoothed = _ms_kim_smoother(filtered, predicted, P)

    # --- Delta-method SEs (compute in θ̂ order, then apply perm) ---
    natmap = θ -> _ms_reg_nat(θ, K, nσ, kx)
    se_nat = _ms_delta_ses(nll, natmap, Vector{T}(theta_hat))
    se_B = reshape(se_nat[1:(kx * K)], kx, K)[:, perm]
    se_sig2 = (switching_variance ? se_nat[(kx * K + 1):(kx * K + K)] :
               fill(se_nat[kx * K + 1], K))[perm]

    # --- Diagnostics ---
    fitmean = [dot(smoothed[t, :], [dot(Xm[t, :], B[:, k]) for k in 1:K]) for t in 1:n]
    resid = yv .- fitmean
    ergodic = _ms_ergodic(P)
    durations = T[one(T) / max(one(T) - P[k, k], eps(T)) for k in 1:K]
    n_params = K * kx + nσ + K * (K - 1)
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + log(T(n)) * n_params
    xnms = xnames === nothing ? _ms_default_xnames(kx) : collect(String.(xnames))

    return MSRegModel{T}(:regression, yv, Xm, K, 0, means, B, se_B, T[], T[],
        Vector{T}(sig2), Vector{T}(se_sig2), P, ergodic, durations,
        filtered, smoothed, resid, loglik, aic, bic, n, n_params,
        switching_variance, false, converged, iters, xnms, "y")
end

estimate_ms(y::AbstractVector; kwargs...) =
    estimate_ms(y, ones(float(eltype(y)), length(y), 1); xnames=["const"], kwargs...)

function _ms_default_xnames(kx::Int)
    kx == 1 && return ["const"]
    return vcat("const", ["x$i" for i in 1:(kx - 1)])
end

# State-conditional Gaussian densities for the regression model (n × K).
function _ms_reg_eta(y::AbstractVector, X::AbstractMatrix, B::AbstractMatrix,
                     sig2::AbstractVector, K::Int)
    S = promote_type(eltype(B), eltype(sig2))
    n = length(y)
    eta = Matrix{S}(undef, n, K)
    for k in 1:K
        r = y .- X * @view B[:, k]
        s2 = sig2[k]
        @. eta[:, k] = exp(-S(0.5) * (log(S(2π)) + log(s2) + r^2 / s2))
    end
    return eta
end

# Negative log-likelihood for the switching regression (ForwardDiff-friendly).
function _ms_reg_nll(theta::AbstractVector, y, X, K::Int, nσ::Int, kx::Int)
    S = eltype(theta)
    B = reshape(theta[1:(kx * K)], kx, K)
    off = kx * K
    sig2 = nσ == K ? exp.(theta[(off + 1):(off + K)]) :
                     fill(exp(theta[off + 1]), K)
    off += nσ
    P = _logits_to_P(theta[(off + 1):end], K)
    eta = _ms_reg_eta(y, X, B, sig2, K)
    _, _, ll = _ms_hamilton_filter(eta, P, _ms_ergodic(P))
    return -ll
end

# Natural-parameter map for the regression model: [vec(B); σ²(K); vec(P)].
function _ms_reg_nat(theta::AbstractVector, K::Int, nσ::Int, kx::Int)
    B = theta[1:(kx * K)]
    off = kx * K
    sig2 = nσ == K ? exp.(theta[(off + 1):(off + K)]) :
                     fill(exp(theta[off + 1]), K)
    off += nσ
    P = _logits_to_P(theta[(off + 1):end], K)
    return vcat(B, sig2, vec(P))
end

# Kim (1994) transition-matrix EM update from joint smoothed probabilities.
function _ms_update_P(smoothed::AbstractMatrix{T}, filtered::AbstractMatrix{T},
                      predicted::AbstractMatrix{T}, P::AbstractMatrix{T},
                      K::Int) where {T}
    n = size(smoothed, 1)
    Pn = zeros(T, K, K)
    for t in 2:n, i in 1:K, j in 1:K
        pj = max(predicted[t, j], eps(T))
        Pn[i, j] += smoothed[t, j] * P[i, j] * filtered[t - 1, i] / pj
    end
    for i in 1:K
        rs = sum(@view Pn[i, :])
        rs > 0 ? (Pn[i, :] ./= rs) : (Pn[i, :] .= one(T) / K)
    end
    return Pn
end

# =============================================================================
# estimate_ms_ar — Hamilton (1989) mean-switching autoregression
# =============================================================================

"""
    estimate_ms_ar(y, p; k_regimes=2, switching_variance=false, max_iter=1000,
                   yname="y") -> MSRegModel

Estimate the Hamilton (1989) mean-switching Markov-switching autoregression

```math
(y_t - μ_{s_t}) = \\sum_{j=1}^{p} φ_j (y_{t-j} - μ_{s_{t-j}}) + ε_t,
\\qquad ε_t \\sim N(0, σ²_{s_t}),
```

where only the level `μ` switches with the latent `K`-state Markov chain `sₜ`; the
autoregressive coefficients `φ` are common across regimes, and the variance
switches only when `switching_variance=true`.

**Estimation.** Because the conditional density depends on the regime *path*
`(sₜ, s_{t-1}, …, s_{t-p})`, the model is filtered on the `Kᵖ⁺¹` expanded state
space (Hamilton 1989). Parameters `[μ; φ; log σ²; transition logits]` are
estimated by maximum likelihood (`Optim` LBFGS with a `ForwardDiff` gradient),
initialised from a linear AR(`p`) fit and quantile-spread means. Standard errors
are the delta-method SEs from the ML Hessian. Filtered/smoothed *regime*
probabilities are the marginals of the expanded-state probabilities over `sₜ`.

**Labelling.** Regimes are ordered by increasing `μ`, so regime 1 is the
low-growth (recession) state and regime `K` the high-growth (expansion) state.

# Arguments
- `y::AbstractVector`: the series.
- `p::Int`: autoregressive order.

# Keywords
- `k_regimes::Int`: number of regimes `K` (default `2`).
- `switching_variance::Bool`: switch `σ²` across regimes (default `false`, the
  Hamilton 1989 form).
- `max_iter::Int`: optimiser iteration cap.
- `yname`: dependent-variable label.

Returns an [`MSRegModel`](@ref) with `model_type = :ms_ar`.

# Example (Hamilton 1989 GNP)
```julia
ts = load_example(:gnp_hamilton)
m  = estimate_ms_ar(vec(ts.data), 4)     # μ̂ ≈ (-0.36, 1.16), p₁₁≈0.75, p₂₂≈0.90
report(m)
```
"""
function estimate_ms_ar(y::AbstractVector, p::Int; k_regimes::Int=2,
                        switching_variance::Bool=false, max_iter::Int=1000,
                        yname::String="y")
    p >= 1 || throw(ArgumentError("AR order p must be ≥ 1; got $p."))
    k_regimes >= 2 || throw(ArgumentError("k_regimes must be ≥ 2; got $k_regimes."))
    T = float(eltype(y))
    yv = Vector{T}(y)
    n_full = length(yv)
    K = k_regimes
    nσ = switching_variance ? K : 1
    n = n_full - p                                   # effective sample
    n > K + p + nσ + K * (K - 1) + 1 || throw(ArgumentError(
        "series too short for a $K-regime MS-AR($p)."))

    states, allowed = _expand_states(K, p)
    allowed_bm = BitMatrix(allowed)

    # --- Initialisation: linear AR(p) + quantile-spread regime means ---
    Xlin = Matrix{T}(undef, n, p + 1)
    Xlin[:, 1] .= one(T)
    ylin = yv[(p + 1):n_full]
    for j in 1:p
        Xlin[:, j + 1] = yv[(p + 1 - j):(n_full - j)]
    end
    blin = Matrix{T}(robust_inv(Hermitian(Xlin' * Xlin); silent=true)) * (Xlin' * ylin)
    phi0 = blin[2:end]
    rlin = ylin .- Xlin * blin
    s2_0 = max(var(rlin), T(1e-3))
    qs = T.(quantile(yv, range(0.1, 0.9, length=K)))
    mu0 = collect(qs)
    P0 = fill(T(0.15) / (K - 1), K, K)
    for i in 1:K
        P0[i, i] = T(0.85)
    end
    theta0 = vcat(mu0, phi0, log.(switching_variance ? fill(s2_0, K) : [s2_0]),
                  _P_to_logits(P0, K))

    nll = θ -> _ms_ar_nll(θ, yv, p, K, nσ, states, allowed_bm)
    theta_hat = theta0
    converged = false
    iters = 0
    try
        g! = (G, θ) -> ForwardDiff.gradient!(G, nll, θ)
        res = Optim.optimize(nll, g!, theta0, Optim.LBFGS(),
                             Optim.Options(f_reltol=1e-11, g_tol=1e-7, iterations=max_iter))
        cand = Optim.minimizer(res)
        iters = Optim.iterations(res)
        if nll(cand) <= nll(theta0)
            theta_hat = cand
            converged = Optim.converged(res)
        end
    catch err
        err isa InterruptException && rethrow()
    end

    # --- Unpack ---
    mu = theta_hat[1:K]
    phi = theta_hat[(K + 1):(K + p)]
    off = K + p
    sig2 = switching_variance ? exp.(theta_hat[(off + 1):(off + K)]) :
                                fill(exp(theta_hat[off + 1]), K)
    off += nσ
    P = _logits_to_P(theta_hat[(off + 1):end], K)

    # --- Order regimes by increasing μ ---
    perm = sortperm(mu)
    mu = mu[perm]; sig2 = sig2[perm]; P = P[perm, perm]

    # --- Filter/smooth at θ̂ (expanded state), marginalise to regimes ---
    eta = _ms_ar_eta(yv, mu, phi, sig2, p, K, states)
    Pexp = _expanded_P(P, states, allowed_bm)
    xi0 = _ms_ergodic(Pexp)
    filtered_e, predicted_e, loglik = _ms_hamilton_filter(eta, Pexp, xi0)
    smoothed_e = _ms_kim_smoother(filtered_e, predicted_e, Pexp)
    filtered = _ms_marginalise(filtered_e, states, K)
    smoothed = _ms_marginalise(smoothed_e, states, K)

    # --- Delta-method SEs (θ̂ order, then permute the switching pieces) ---
    natmap = θ -> _ms_ar_nat(θ, K, p, nσ)
    se_nat = _ms_delta_ses(nll, natmap, Vector{T}(theta_hat))
    se_mu = se_nat[1:K][perm]
    se_phi = se_nat[(K + 1):(K + p)]
    se_sig2 = (switching_variance ? se_nat[(K + p + 1):(K + p + K)] :
               fill(se_nat[K + p + 1], K))[perm]

    # --- Diagnostics ---
    # Smoothed conditional mean at each t (over expanded paths).
    fitmean = Vector{T}(undef, n)
    for tau in 1:n
        t = p + tau
        m_t = zero(T)
        for a in 1:size(states, 1)
            i0 = states[a, 1]
            mval = mu[i0]                     # NB: perm applied to mu already
            # reconstruct using ORIGINAL-order state means requires care; use
            # the ordered μ consistently since we re-filtered with ordered P/μ.
            for j in 1:p
                mval += phi[j] * (yv[t - j] - mu[states[a, j + 1]])
            end
            m_t += smoothed_e[tau, a] * mval
        end
        fitmean[tau] = m_t
    end
    # smoothed_e was computed with ORDERED μ/P, and `states` index regimes in the
    # ordered labelling — consistent, so fitmean is correct.
    resid = yv[(p + 1):n_full] .- fitmean
    ergodic = _ms_ergodic(P)
    durations = T[one(T) / max(one(T) - P[k, k], eps(T)) for k in 1:K]
    n_params = K + p + nσ + K * (K - 1)
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + log(T(n)) * n_params

    X_eff = Xlin                              # [1, y_{t-1}, …, y_{t-p}]
    xnms = vcat("const", ["y[t-$i]" for i in 1:p])
    coefs = reshape(Vector{T}(mu), 1, K)
    se_coefs = reshape(Vector{T}(se_mu), 1, K)

    return MSRegModel{T}(:ms_ar, yv[(p + 1):n_full], X_eff, K, p, Vector{T}(mu),
        coefs, se_coefs, Vector{T}(phi), Vector{T}(se_phi),
        Vector{T}(sig2), Vector{T}(se_sig2), P, ergodic, durations,
        filtered, smoothed, resid, loglik, aic, bic, n, n_params,
        switching_variance, false, converged, iters, xnms, yname)
end

# Marginalise expanded-state probabilities (n × M) to regime marginals (n × K).
function _ms_marginalise(prob_e::AbstractMatrix{T}, states::Matrix{Int}, K::Int) where {T}
    n, M = size(prob_e)
    out = zeros(T, n, K)
    for t in 1:n, a in 1:M
        out[t, states[a, 1]] += prob_e[t, a]
    end
    return out
end

# Expanded-state Gaussian densities for the mean-switching AR (n_eff × M).
function _ms_ar_eta(y::AbstractVector, mu::AbstractVector, phi::AbstractVector,
                    sig2::AbstractVector, p::Int, K::Int, states::Matrix{Int})
    S = promote_type(eltype(mu), eltype(phi), eltype(sig2))
    n_full = length(y)
    n = n_full - p
    M = size(states, 1)
    eta = Matrix{S}(undef, n, M)
    c = log(S(2π))
    for tau in 1:n
        t = p + tau
        for a in 1:M
            i0 = states[a, 1]
            dev = y[t] - mu[i0]
            for j in 1:p
                dev -= phi[j] * (y[t - j] - mu[states[a, j + 1]])
            end
            s2 = sig2[i0]
            eta[tau, a] = exp(-S(0.5) * (c + log(s2) + dev^2 / s2))
        end
    end
    return eta
end

# Negative log-likelihood for the mean-switching AR (ForwardDiff-friendly).
function _ms_ar_nll(theta::AbstractVector, y, p::Int, K::Int, nσ::Int,
                    states::Matrix{Int}, allowed::BitMatrix)
    mu = theta[1:K]
    phi = theta[(K + 1):(K + p)]
    off = K + p
    sig2 = nσ == K ? exp.(theta[(off + 1):(off + K)]) :
                     fill(exp(theta[off + 1]), K)
    off += nσ
    P = _logits_to_P(theta[(off + 1):end], K)
    eta = _ms_ar_eta(y, mu, phi, sig2, p, K, states)
    Pexp = _expanded_P(P, states, allowed)
    xi0 = _ms_ergodic(Pexp)
    _, _, ll = _ms_hamilton_filter(eta, Pexp, xi0)
    return -ll
end

# Natural-parameter map for the mean-switching AR: [μ(K); φ(p); σ²(K); vec(P)].
function _ms_ar_nat(theta::AbstractVector, K::Int, p::Int, nσ::Int)
    mu = theta[1:K]
    phi = theta[(K + 1):(K + p)]
    off = K + p
    sig2 = nσ == K ? exp.(theta[(off + 1):(off + K)]) :
                     fill(exp(theta[off + 1]), K)
    off += nσ
    P = _logits_to_P(theta[(off + 1):end], K)
    return vcat(mu, phi, sig2, vec(P))
end
