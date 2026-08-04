# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Time-varying-parameter VAR with stochastic volatility (T250 / #349).

Primiceri (2005) writes the VAR with drifting coefficients and drifting shock variances

```math
y_t = X_t' B_t + A_t^{-1}\\Sigma_t\\varepsilon_t,\\qquad \\varepsilon_t\\sim N(0, I_n)
```

with `B_t`, the free elements of the lower-triangular `A_t`, and `log σ²_{i,t}` each
following independent random walks. Setting `tvp=false` fixes `B_t` and `A_t` and leaves
only the volatilities drifting — the Cogley & Sargent (2005) constant-coefficient SV-BVAR.

The sampler reuses the in-repo Kim-Shephard-Chib mixture machinery (`src/sv/`) for the
volatility block and `_draw_inverse_wishart` for the state-innovation covariances; the
coefficient blocks use a Carter-Kohn forward-filter/backward-sample written for the
random-walk state with a controlled initial-state prior.

Provides:
- `TVPVARPosterior` — posterior draws of `B_t`, `A_t`, `log σ²_t` and the state covariances
- `estimate_tvpvar` — the Gibbs sampler
- `irf(::TVPVARPosterior, H; t)` — the impulse response *at a chosen date*

References:
- Primiceri, G. E. (2005). Time Varying Structural Vector Autoregressions and Monetary
  Policy. *Review of Economic Studies*, 72(3), 821-852.
- Del Negro, M. & Primiceri, G. E. (2015). Time Varying Structural Vector Autoregressions
  and Monetary Policy: A Corrigendum. *Review of Economic Studies*, 82(4), 1342-1345.
- Cogley, T. & Sargent, T. J. (2005). Drifts and Volatilities: Monetary Policies and
  Outcomes in the Post WWII US. *Review of Economic Dynamics*, 8(2), 262-302.
- Kim, S., Shephard, N. & Chib, S. (1998). Stochastic Volatility: Likelihood Inference and
  Comparison with ARCH Models. *Review of Economic Studies*, 65(3), 361-393.
"""

using LinearAlgebra
using Random
using Statistics
using Distributions

# =============================================================================
# Posterior container
# =============================================================================

"""
    TVPVARPosterior{T}

Posterior draws from a time-varying-parameter VAR with stochastic volatility.

# Fields
- `B_draws::Array{T,3}` — `n_draws × T_eff × k` drifting VAR coefficients, `k = n(1+np)`
- `A_draws::Array{T,3}` — `n_draws × T_eff × n_a` free elements of `A_t`, `n_a = n(n-1)/2`,
  stacked row-wise (`a₂₁, a₃₁, a₃₂, …`)
- `H_draws::Array{T,3}` — `n_draws × T_eff × n` log-**variances** `log σ²_{i,t}`. This is
  the Kim-Shephard-Chib state convention (`y* = log(e²) = h + log ε²`), so the standard
  deviation is `exp(h/2)`; [`volatility_path`](@ref) returns it directly
- `Q_draws::Array{T,3}` — `n_draws × k × k` coefficient random-walk covariance
- `S_draws::Array{T,3}` — `n_draws × n_a × n_a` block-diagonal `A_t` random-walk covariance
- `W_draws::Matrix{T}` — `n_draws × n` log-volatility random-walk variances
- `Y::Matrix{T}` — the original data
- `p::Int`, `n::Int`, `T_eff::Int` — lag order, variables, effective sample
- `n_train::Int` — training-sample length used to calibrate the priors
- `tvp::Bool` / `sv::Bool` — whether coefficients drift / volatilities drift
- `varnames::Vector{String}` — variable names
"""
struct TVPVARPosterior{T<:AbstractFloat}
    B_draws::Array{T,3}
    A_draws::Array{T,3}
    H_draws::Array{T,3}
    Q_draws::Array{T,3}
    S_draws::Array{T,3}
    W_draws::Matrix{T}
    Y::Matrix{T}
    p::Int
    n::Int
    T_eff::Int
    n_train::Int
    tvp::Bool
    sv::Bool
    varnames::Vector{String}
end

n_draws(post::TVPVARPosterior) = size(post.B_draws, 1)

function Base.show(io::IO, post::TVPVARPosterior{T}) where {T}
    label = post.tvp && post.sv ? "TVP-VAR with stochastic volatility (Primiceri 2005)" :
            !post.tvp && post.sv ? "SV-BVAR, constant coefficients (Cogley-Sargent 2005)" :
            post.tvp ? "TVP-VAR, homoskedastic" : "Constant-coefficient VAR (Gibbs)"
    spec = Any[
        "Model"                 label;
        "Variables"             post.n;
        "Lags"                  post.p;
        "Effective obs."        post.T_eff;
        "Training sample"       post.n_train;
        "Posterior draws"       n_draws(post);
        "Drifting coefficients" post.tvp ? "Yes" : "No";
        "Stochastic volatility" post.sv ? "Yes" : "No"
    ]
    _pretty_table(io, spec;
        title = "Time-Varying Parameter VAR",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Posterior mean volatility path at the sample endpoints, the headline TVP object
    vol = dropdims(mean(exp.(post.H_draws ./ 2); dims=1); dims=1)   # σ = exp(h/2), T_eff × n
    idx = unique(clamp.([1, post.T_eff ÷ 4, post.T_eff ÷ 2, 3 * post.T_eff ÷ 4, post.T_eff],
                        1, post.T_eff))
    data = Matrix{Any}(undef, length(idx), post.n + 1)
    for (r, t) in enumerate(idx)
        data[r, 1] = t
        for j in 1:post.n
            data[r, j+1] = _fmt(vol[t, j])
        end
    end
    _pretty_table(io, data;
        title = "Posterior mean σ_{i,t}",
        column_labels = vcat("t", post.varnames),
        alignment = vcat(:r, fill(:r, post.n)),
    )

    # Random-walk innovation scale — how much drift the data actually support
    rw = Any[
        "mean diag(Q)"  _fmt(mean(mean(post.Q_draws[d, i, i] for i in 1:size(post.Q_draws, 2))
                                  for d in 1:n_draws(post)); digits=6);
        "mean diag(W)"  _fmt(mean(post.W_draws); digits=6)
    ]
    _pretty_table(io, rw;
        title = "State-innovation variances",
        column_labels = ["", "Posterior mean"],
        alignment = [:l, :r],
    )
    return nothing
end

# =============================================================================
# Design construction
# =============================================================================

"""
    _tvp_design(Y, p) -> (y_eff, Xt, T_eff, k)

Build the per-period design matrices. Row `t` of the effective sample has
`y_t = X_t' B_t + u_t` with `X_t' = I_n ⊗ z_t'` where `z_t = [1, y_{t-1}', …, y_{t-p}']`,
so `X_t'` is `n × k` with `k = n(1 + np)`.
"""
function _tvp_design(Y::Matrix{T}, p::Int) where {T<:AbstractFloat}
    T_obs, n = size(Y)
    T_eff = T_obs - p
    m = 1 + n * p
    k = n * m
    y_eff = Matrix{T}(Y[(p+1):end, :])
    Xt = Vector{Matrix{T}}(undef, T_eff)
    z = Vector{T}(undef, m)
    for t in 1:T_eff
        z[1] = one(T)
        col = 2
        for lag in 1:p, j in 1:n
            z[col] = Y[p + t - lag, j]
            col += 1
        end
        Xi = zeros(T, n, k)
        for i in 1:n
            @views Xi[i, ((i-1)*m+1):(i*m)] .= z
        end
        Xt[t] = Xi
    end
    return y_eff, Xt, T_eff, k
end

# =============================================================================
# Carter-Kohn FFBS for a random-walk state
# =============================================================================

"""
    _tvp_ffbs_rw(y, Xt, Rt, Q, b0, P0, rng) -> Matrix

Carter-Kohn forward-filter / backward-sample of a random-walk coefficient path:

```
    y_t = X_t b_t + e_t,   e_t ~ N(0, R_t)
    b_t = b_{t-1} + v_t,   v_t ~ N(0, Q)
```

`R_t` is allowed to differ every period — that is how the stochastic-volatility block
feeds into the coefficient block. Returns the sampled path as a `T × k` matrix.

The backward pass exploits the identity transition: given `b_{t+1}`, the conditional mean
is `b_{t|t} + G(b_{t+1} - b_{t|t})` with `G = P_{t|t}P_{t+1|t}^{-1}` and covariance
`P_{t|t} - G P_{t+1|t} G'`. Draws use an eigenvalue-clipped symmetric square root so a
marginally indefinite conditional covariance (possible after a poor sweep) does not abort
the chain.
"""
function _tvp_ffbs_rw(y::Matrix{T}, Xt::Vector{Matrix{T}}, Rt::Vector{Matrix{T}},
                      Q::Matrix{T}, b0::Vector{T}, P0::Matrix{T},
                      rng::AbstractRNG) where {T<:AbstractFloat}
    T_eff = length(Xt)
    k = length(b0)
    b_filt = Matrix{T}(undef, T_eff, k)
    P_filt = Array{T,3}(undef, T_eff, k, k)
    P_pred_store = Array{T,3}(undef, T_eff, k, k)

    b_prev = copy(b0)
    P_prev = Matrix{T}(P0)

    for t in 1:T_eff
        b_pred = b_prev                       # random walk ⇒ prediction is the previous state
        P_pred = P_prev + Q
        P_pred = T(0.5) * (P_pred + P_pred')
        P_pred_store[t, :, :] = P_pred

        X = Xt[t]
        PXt = P_pred * X'
        F = X * PXt + Rt[t]
        F = T(0.5) * (F + F')
        Finv = robust_inv(Symmetric(F))
        K = PXt * Finv
        innov = @view(y[t, :])
        v = innov .- X * b_pred

        b_prev = b_pred .+ K * v
        P_prev = P_pred .- K * (X * P_pred)
        P_prev = T(0.5) * (P_prev + P_prev')

        b_filt[t, :] = b_prev
        P_filt[t, :, :] = P_prev
    end

    out = Matrix{T}(undef, T_eff, k)
    out[T_eff, :] = @views b_filt[T_eff, :] .+
                    _psd_sqrt(Matrix{T}(P_filt[T_eff, :, :])) * randn(rng, T, k)
    for t in (T_eff-1):-1:1
        Pf = Matrix{T}(@view P_filt[t, :, :])
        Pp = Matrix{T}(@view P_pred_store[t+1, :, :])
        G = Pf * robust_inv(Symmetric(Pp))
        mu = @views b_filt[t, :] .+ G * (out[t+1, :] .- b_filt[t, :])
        # Symmetric form P_{t|t} - G P_{t+1|t} G' rather than P_{t|t} - G P_{t|t}: the two
        # are equal in exact arithmetic (G = P_{t|t}P_{t+1|t}^{-1}), but the symmetric one
        # is far better conditioned when the filtered covariance is large, as it is under a
        # diffuse initial prior.
        Pc = Pf .- G * Pp * G'
        Pc = T(0.5) * (Pc + Pc')
        out[t, :] = mu .+ _psd_sqrt(Pc) * randn(rng, T, k)
    end
    return out
end

"""Symmetric PSD square root with negative eigenvalues clipped to zero."""
function _psd_sqrt(M::Matrix{T}) where {T<:AbstractFloat}
    Ms = T(0.5) * (M + M')
    E = eigen(Symmetric(Ms))
    return E.vectors * Diagonal(sqrt.(max.(E.values, zero(T)))) * E.vectors'
end

# =============================================================================
# Random-walk stochastic-volatility block (KSC mixture)
# =============================================================================

"""
    _tvp_sv_ffbs!(h_out, h_filt, P_filt, y_star, s, sigma2_eta, h0_mean, h0_var, rng)

FFBS for a **random-walk** log-volatility conditional on the KSC mixture indicators:

```
    y*_t = h_t + ξ_t,   ξ_t ~ N(m_{s_t}, v²_{s_t})     (KSC 10-component mixture)
    h_t  = h_{t-1} + η_t,  η_t ~ N(0, σ²_η)
```

This differs from `_ksc_ffbs!` only in the state prior: that routine initializes from the
stationary distribution of an AR(1), which is undefined at `φ = 1`. Primiceri calibrates
`h_0 ~ N(log σ̂²_OLS, 4)` from a training sample instead, which is what `h0_mean`/`h0_var`
carry. The mixture constants and the indicator draw are reused from `src/sv/`.
"""
function _tvp_sv_ffbs!(h_out::Vector{T}, h_filt::Vector{T}, P_filt::Vector{T},
                       y_star::Vector{T}, s::Vector{Int}, sigma2_eta::T,
                       h0_mean::T, h0_var::T, rng::AbstractRNG) where {T<:AbstractFloat}
    n = length(y_star)
    h_pred = h0_mean
    P_pred = h0_var + sigma2_eta
    P_pred_store = Vector{T}(undef, n)

    for t in 1:n
        P_pred_store[t] = P_pred
        m_st = T(_KSC_MEANS[s[t]])
        v_st = T(_KSC_VARIANCES[s[t]])
        F_t = P_pred + v_st
        K_t = P_pred / F_t
        h_filt[t] = h_pred + K_t * (y_star[t] - m_st - h_pred)
        P_filt[t] = P_pred * (one(T) - K_t)
        if t < n
            h_pred = h_filt[t]
            P_pred = P_filt[t] + sigma2_eta
        end
    end

    h_out[n] = h_filt[n] + sqrt(max(P_filt[n], T(1e-12))) * randn(rng, T)
    for t in (n-1):-1:1
        G = P_filt[t] / P_pred_store[t+1]
        mu = h_filt[t] + G * (h_out[t+1] - h_filt[t])
        Pc = max(P_filt[t] - G * P_filt[t], T(1e-12))
        h_out[t] = mu + sqrt(Pc) * randn(rng, T)
    end
    return h_out
end

# =============================================================================
# Training-sample prior (Primiceri 2005 §4.1)
# =============================================================================

"""
    _tvp_training_prior(Y, p, n_train) -> NamedTuple

Calibrate the initial-state priors and the state-covariance priors on a training sample,
following Primiceri (2005 §4.1): a fixed-coefficient VAR is fitted to the first `n_train`
observations, and

- `B_0 ~ N(B̂_OLS, 4 V(B̂_OLS))`
- `A_0 ~ N(Â_OLS, 4 V(Â_OLS))`
- `log σ²_0 ~ N(log σ̂²_OLS, 4)`
- `Q ~ IW(k_Q² · τ · V(B̂_OLS), τ)`
- `S_i ~ IW(k_S² · (1+i-1) · V(Â_{i,OLS}), 1+i-1)` per equation block
- `W_i ~ IG((1+1)/2, k_W² · (1+1)/2)`

with `τ` the training length. The `k` constants control how much drift the prior permits;
Primiceri's values (`k_Q = 0.01`, `k_S = 0.1`, `k_W = 0.01`) are the defaults.
"""
function _tvp_training_prior(Y::Matrix{T}, p::Int, n_train::Int;
                             k_Q::Real=0.01, k_S::Real=0.1,
                             k_W::Real=0.01) where {T<:AbstractFloat}
    n = size(Y, 2)
    Y_tr = Matrix{T}(Y[1:n_train, :])
    y_tr, Xt_tr, T_tr, k = _tvp_design(Y_tr, p)
    m = 1 + n * p

    # Pooled OLS on the training sample: stack the per-period designs
    XtX = zeros(T, k, k)
    Xty = zeros(T, k)
    for t in 1:T_tr
        X = Xt_tr[t]
        XtX .+= X' * X
        Xty .+= X' * @view(y_tr[t, :])
    end
    XtXinv = Matrix{T}(robust_inv(Symmetric(XtX)))
    B_ols = XtXinv * Xty

    resid = Matrix{T}(undef, T_tr, n)
    for t in 1:T_tr
        resid[t, :] = @view(y_tr[t, :]) .- Xt_tr[t] * B_ols
    end
    Sigma_ols = (resid' * resid) / T(max(T_tr - m, 1))
    Sigma_ols = T(0.5) * (Sigma_ols + Sigma_ols')
    V_B = Matrix{T}(_psd_sqrt(XtXinv * mean(diag(Sigma_ols)))^2)   # scale (X'X)⁻¹ by σ̄²
    V_B = T(0.5) * (V_B + V_B')

    # Contemporaneous matrix from a Cholesky of the training residual covariance:
    # A Σ Σ' A' = residual covariance ⇒ A = inv(chol lower), free elements below the diagonal.
    L = Matrix{T}(safe_cholesky(Sigma_ols))
    A_full = Matrix{T}(robust_inv(LowerTriangular(L)))
    A_full = A_full ./ diag(A_full)          # normalize the unit diagonal
    n_a = n * (n - 1) ÷ 2
    A_ols = Vector{T}(undef, n_a)
    idx = 1
    for i in 2:n, j in 1:(i-1)
        A_ols[idx] = A_full[i, j]
        idx += 1
    end
    log_var_ols = T[log(max(Sigma_ols[i, i], T(1e-10))) for i in 1:n]   # h₀ = log σ̂²

    # Per-equation prior variance for the A block: the OLS variance of regressing the i-th
    # residual on the preceding ones, which is the regression A_t implicitly runs.
    V_A_blocks = Vector{Matrix{T}}(undef, n - 1)
    for i in 2:n
        Xi = -Matrix{T}(resid[:, 1:(i-1)])
        yi = Vector{T}(resid[:, i])
        XtXi = robust_inv(Symmetric(Xi' * Xi))
        bi = XtXi * (Xi' * yi)
        ri = yi .- Xi * bi
        s2 = dot(ri, ri) / T(max(T_tr - (i - 1), 1))
        V_A_blocks[i-1] = Matrix{T}(Symmetric(XtXi .* s2))
    end
    V_A = zeros(T, n_a, n_a)
    off = 0
    for i in 2:n
        d = i - 1
        V_A[(off+1):(off+d), (off+1):(off+d)] = V_A_blocks[d]
        off += d
    end

    tau = T_tr
    Q_scale = Matrix{T}(Symmetric(T(k_Q)^2 * T(tau) * V_B))
    Q_df = max(tau, k + 1)
    S_scales = [Matrix{T}(Symmetric(T(k_S)^2 * T(i) * V_A_blocks[i-1])) for i in 2:n]
    S_dfs = [i for i in 2:n]                      # 1 + (i-1) degrees of freedom
    W_shape = T(0.5) * T(2)
    W_scale = T(0.5) * T(k_W)^2 * T(2)

    return (B_ols=B_ols, V_B=V_B, A_ols=A_ols, V_A=V_A, V_A_blocks=V_A_blocks,
            log_var_ols=log_var_ols, Sigma_ols=Sigma_ols,
            Q_scale=Q_scale, Q_df=Q_df, S_scales=S_scales, S_dfs=S_dfs,
            W_shape=W_shape, W_scale=W_scale, k=k, n_a=n_a)
end

"""Rebuild the lower-triangular `A_t` (unit diagonal) from its stacked free elements."""
function _tvp_A_matrix(a::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    A = Matrix{T}(I, n, n)
    idx = 1
    for i in 2:n, j in 1:(i-1)
        A[i, j] = a[idx]
        idx += 1
    end
    return A
end

# =============================================================================
# Gibbs sampler
# =============================================================================

"""
    estimate_tvpvar(Y, p; kwargs...) -> TVPVARPosterior

Estimate a time-varying-parameter VAR with stochastic volatility (Primiceri 2005) by Gibbs
sampling, or its constant-coefficient special case (Cogley & Sargent 2005) with `tvp=false`.

# Model

```math
y_t = X_t' B_t + A_t^{-1}\\Sigma_t\\varepsilon_t,\\qquad
B_t = B_{t-1} + \\nu_t,\\quad a_t = a_{t-1} + \\zeta_t,\\quad
\\log\\sigma^2_{i,t} = \\log\\sigma^2_{i,t-1} + \\eta_{i,t}
```

where:
- ``X_t' = I_n \\otimes [1, y_{t-1}', \\ldots, y_{t-p}']`` so ``B_t`` stacks all equations
- ``A_t`` is lower triangular with a unit diagonal; its free elements are the structural
  contemporaneous coefficients
- ``\\Sigma_t = \\mathrm{diag}(\\sigma_{1,t}, \\ldots, \\sigma_{n,t})``
- ``\\nu_t \\sim N(0,Q)``, ``\\zeta_t \\sim N(0,S)`` (block diagonal by equation),
  ``\\eta_t \\sim N(0,W)`` (diagonal)

# Gibbs blocks

Each sweep draws, in the Del Negro & Primiceri (2015) **corrected** order:

1. `B_{1:T}` by Carter-Kohn, with per-period observation covariance ``A_t^{-1}\\Sigma_t^2 A_t^{-1\\prime}``
2. `a_{1:T}` equation by equation, each a univariate Carter-Kohn on the VAR residuals
3. the mixture indicators `s_t` given the **current** volatilities, then `log σ²_{1:T}` given
   `s` — the corrigendum's point is that `s` must be drawn before, not after, the volatility
   update, and that the coefficient blocks are drawn without conditioning on `s`
4. `Q`, `S`, `W` from inverse-Wishart / inverse-gamma conjugate updates

# Arguments
- `Y::AbstractMatrix` — `T × n` data, time in rows
- `p::Int` — lag order

# Keywords
- `tvp::Bool=true` — let `B_t` and `a_t` drift; `false` gives the Cogley-Sargent SV-BVAR
- `sv::Bool=true` — let the volatilities drift; `false` fixes them at their training values
- `n_draws::Int=2000` — retained posterior draws
- `n_burn::Int=1000` — burn-in sweeps discarded
- `thin::Int=1` — keep every `thin`-th post-burn-in sweep
- `n_train::Int=0` — training-sample length for the prior; `0` uses `max(4p+n+2, T÷4)`
- `k_Q::Real=0.01`, `k_S::Real=0.1`, `k_W::Real=0.01` — Primiceri's prior-drift constants
- `varnames::Vector{String}` — variable names
- `rng::AbstractRNG=Random.default_rng()` — random number generator

# Returns
[`TVPVARPosterior`](@ref).
"""
function estimate_tvpvar(Y::AbstractMatrix{T}, p::Int;
                         tvp::Bool=true, sv::Bool=true,
                         n_draws::Int=2000, n_burn::Int=1000, thin::Int=1,
                         n_train::Int=0,
                         k_Q::Real=0.01, k_S::Real=0.1, k_W::Real=0.01,
                         varnames::Vector{String}=String[],
                         rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    _validate_data(Y, "Y")
    p >= 1 || throw(ArgumentError("p must be at least 1, got $p"))
    n_draws >= 1 || throw(ArgumentError("n_draws must be positive"))
    n_burn >= 0 || throw(ArgumentError("n_burn must be non-negative"))
    thin >= 1 || throw(ArgumentError("thin must be at least 1"))

    Ym = Matrix{T}(Y)
    T_obs, n = size(Ym)
    n >= 2 || throw(ArgumentError("TVP-VAR requires at least 2 variables, got $n"))
    vn = isempty(varnames) ? ["y$i" for i in 1:n] : copy(varnames)
    length(vn) == n || throw(ArgumentError("varnames must have length $n"))

    ntr = n_train > 0 ? n_train : max(4 * p + n + 2, T_obs ÷ 4)
    ntr < T_obs || throw(ArgumentError(
        "training sample ($ntr) must be shorter than the data ($T_obs)"))
    ntr > p + n + 1 || throw(ArgumentError(
        "training sample ($ntr) is too short for $n variables and $p lags"))

    pri = _tvp_training_prior(Ym, p, ntr; k_Q=k_Q, k_S=k_S, k_W=k_W)
    k, n_a = pri.k, pri.n_a

    # Estimation sample starts after the training block (Primiceri's convention)
    Y_est = Matrix{T}(Ym[(ntr - p + 1):end, :])
    y, Xt, T_eff, _ = _tvp_design(Y_est, p)

    # ── Initial states ───────────────────────────────────────────────────
    B = repeat(reshape(pri.B_ols, 1, :), T_eff, 1)              # T_eff × k
    A = repeat(reshape(pri.A_ols, 1, :), T_eff, 1)              # T_eff × n_a
    H = repeat(reshape(pri.log_var_ols, 1, :), T_eff, 1)        # T_eff × n (log σ²)
    Q = Matrix{T}(pri.Q_scale ./ max(pri.Q_df, 1))
    Sblocks = [Matrix{T}(pri.S_scales[i] ./ max(pri.S_dfs[i], 1)) for i in 1:(n-1)]
    W = fill(T(k_W)^2, n)

    P0_B = Matrix{T}(Symmetric(4 * pri.V_B))
    P0_A_blocks = [Matrix{T}(Symmetric(4 * pri.V_A_blocks[i])) for i in 1:(n-1)]
    h0_var = T(4)

    n_keep = n_draws
    B_out = Array{T,3}(undef, n_keep, T_eff, k)
    A_out = Array{T,3}(undef, n_keep, T_eff, n_a)
    H_out = Array{T,3}(undef, n_keep, T_eff, n)
    Q_out = Array{T,3}(undef, n_keep, k, k)
    S_out = Array{T,3}(undef, n_keep, n_a, n_a)
    W_out = Matrix{T}(undef, n_keep, n)

    # Workspaces
    Rt = Vector{Matrix{T}}(undef, T_eff)
    resid = Matrix{T}(undef, T_eff, n)
    y_star = Vector{T}(undef, T_eff)
    s_idx = Vector{Int}(undef, T_eff)
    log_probs = Vector{T}(undef, 10)
    h_filt = Vector{T}(undef, T_eff)
    P_filt = Vector{T}(undef, T_eff)
    h_new = Vector{T}(undef, T_eff)
    h_col = Vector{T}(undef, T_eff)     # _ksc_draw_indicators! takes a Vector, not a view
    c_off = T(1e-6)                                    # log-offset, as in estimate_sv

    total = n_burn + n_draws * thin
    kept = 0
    for sweep in 1:total
        # ── Block 1: B_{1:T} | A, Σ ──────────────────────────────────────
        for t in 1:T_eff
            At = _tvp_A_matrix(@view(A[t, :]), n)
            Ainv = Matrix{T}(robust_inv(LowerTriangular(At)))
            # h is log σ², so the variance matrix is diag(exp(h)) directly
            Rt[t] = Matrix{T}(Symmetric(Ainv * Diagonal(exp.(@view H[t, :])) * Ainv'))
        end
        if tvp
            B = _tvp_ffbs_rw(y, Xt, Rt, Q, pri.B_ols, P0_B, rng)
        else
            # Constant coefficients: one GLS draw pooling every period
            XtX = zeros(T, k, k); Xty = zeros(T, k)
            for t in 1:T_eff
                Rinv = robust_inv(Symmetric(Rt[t]))
                XR = Xt[t]' * Rinv
                XtX .+= XR * Xt[t]
                Xty .+= XR * @view(y[t, :])
            end
            V0inv = robust_inv(Symmetric(P0_B))
            Vpost = Matrix{T}(robust_inv(Symmetric(XtX + V0inv)))
            bpost = Vpost * (Xty + V0inv * pri.B_ols)
            bdraw = bpost .+ _psd_sqrt(Vpost) * randn(rng, T, k)
            B = repeat(reshape(bdraw, 1, :), T_eff, 1)
        end

        for t in 1:T_eff
            resid[t, :] = @view(y[t, :]) .- Xt[t] * @view(B[t, :])
        end

        # ── Block 2: a_{1:T} | B, Σ, equation by equation ────────────────
        if n > 1
            off = 0
            for i in 2:n
                d = i - 1
                yi = reshape(Vector{T}(@view resid[:, i]), T_eff, 1)
                Xi = Vector{Matrix{T}}(undef, T_eff)
                Ri = Vector{Matrix{T}}(undef, T_eff)
                for t in 1:T_eff
                    Xi[t] = reshape(T[-resid[t, j] for j in 1:d], 1, d)
                    Ri[t] = fill(exp(H[t, i]), 1, 1)
                end
                a0 = Vector{T}(pri.A_ols[(off+1):(off+d)])
                if tvp
                    ai = _tvp_ffbs_rw(yi, Xi, Ri, Sblocks[d], a0, P0_A_blocks[d], rng)
                else
                    XtX = zeros(T, d, d); Xty = zeros(T, d)
                    for t in 1:T_eff
                        w = one(T) / Ri[t][1, 1]
                        XtX .+= w .* (Xi[t]' * Xi[t])
                        Xty .+= w .* vec(Xi[t]' * yi[t, :])
                    end
                    V0inv = robust_inv(Symmetric(P0_A_blocks[d]))
                    Vp = Matrix{T}(robust_inv(Symmetric(XtX + V0inv)))
                    bp = Vp * (Xty + V0inv * a0)
                    adraw = bp .+ _psd_sqrt(Vp) * randn(rng, T, d)
                    ai = repeat(reshape(adraw, 1, :), T_eff, 1)
                end
                A[:, (off+1):(off+d)] = ai
                off += d
            end
        end

        # ── Block 3: volatilities (Del Negro-Primiceri corrected order) ──
        # Orthogonalized residuals ŷ*_t = A_t û_t have independent components.
        for i in 1:n
            for t in 1:T_eff
                At = _tvp_A_matrix(@view(A[t, :]), n)
                e = dot(@view(At[i, :]), @view(resid[t, :]))
                y_star[t] = log(e^2 + c_off)
            end
            if sv
                # s_t is drawn conditional on the CURRENT h (not the updated one), then h
                # given s — the Del Negro-Primiceri (2015) corrigendum ordering.
                copyto!(h_col, @view H[:, i])
                _ksc_draw_indicators!(s_idx, log_probs, y_star, h_col, rng)
                _tvp_sv_ffbs!(h_new, h_filt, P_filt, y_star, s_idx, W[i],
                              pri.log_var_ols[i], h0_var, rng)
                H[:, i] = h_new
            end
        end

        # ── Block 4: state covariances ───────────────────────────────────
        if tvp
            dB = diff(B; dims=1)
            SQ = Matrix{T}(pri.Q_scale + dB' * dB)
            Q = Matrix{T}(_draw_inverse_wishart(pri.Q_df + T_eff - 1, Symmetric(SQ), rng))

            off = 0
            for i in 2:n
                d = i - 1
                dA = diff(@view(A[:, (off+1):(off+d)]); dims=1)
                SS = Matrix{T}(pri.S_scales[d] + dA' * dA)
                Sblocks[d] = Matrix{T}(_draw_inverse_wishart(pri.S_dfs[d] + T_eff - 1,
                                                             Symmetric(SS), rng))
                off += d
            end
        end
        if sv
            for i in 1:n
                dh = diff(@view(H[:, i]))
                shape = pri.W_shape + T(T_eff - 1) / 2
                scale = pri.W_scale + dot(dh, dh) / 2
                W[i] = T(rand(rng, InverseGamma(shape, scale)))
            end
        end

        # ── Store ────────────────────────────────────────────────────────
        if sweep > n_burn && (sweep - n_burn) % thin == 0 && kept < n_keep
            kept += 1
            B_out[kept, :, :] = B
            A_out[kept, :, :] = A
            H_out[kept, :, :] = H
            Q_out[kept, :, :] = Q
            off = 0
            Sfull = zeros(T, n_a, n_a)
            for i in 2:n
                d = i - 1
                Sfull[(off+1):(off+d), (off+1):(off+d)] = Sblocks[d]
                off += d
            end
            S_out[kept, :, :] = Sfull
            W_out[kept, :] = W
        end
    end

    kept == n_keep || throw(ErrorException(
        "TVP-VAR Gibbs retained $kept of $n_keep draws — increase n_draws or reduce thin"))

    return TVPVARPosterior{T}(B_out, A_out, H_out, Q_out, S_out, W_out,
                              Ym, p, n, T_eff, ntr, tvp, sv, vn)
end

estimate_tvpvar(Y::AbstractMatrix, p::Int; kwargs...) =
    estimate_tvpvar(Float64.(Y), p; kwargs...)

# =============================================================================
# Time-varying impulse responses
# =============================================================================

"""
    irf(post::TVPVARPosterior, horizon; t=post.T_eff, n_draws=500, quantile_levels=..., rng=...)
        -> BayesianImpulseResponse

Impulse response **at date `t`** of the effective sample, integrating over posterior draws.

At each draw the coefficients `B_t` give the VAR companion form and the structural impact
matrix is `A_t^{-1}Σ_t` — the same recursive identification Primiceri's `A_t` encodes, but
with both the propagation and the impact matrix taken at date `t`. Comparing `irf(post, H;
t=t₁)` with `irf(post, H; t=t₂)` is how time variation in the transmission mechanism is
read off.

Draws whose companion matrix is explosive are dropped and counted (reported on the result
as `n_failed`), matching the BVAR IRF convention.

# Keywords
- `t::Int=post.T_eff` — date within the effective sample (1 = first post-training period)
- `n_draws::Int=500` — posterior draws used (capped at the number available)
- `quantile_levels::Vector{<:Real}=[0.05, 0.16, 0.84, 0.95]` — credible-band levels
- `stationary_only::Bool=true` — drop explosive draws
"""
function irf(post::TVPVARPosterior{T}, horizon::Int;
             t::Int=post.T_eff, n_draws::Int=500,
             quantile_levels::Vector{<:Real}=[0.05, 0.16, 0.84, 0.95],
             stationary_only::Bool=true) where {T<:AbstractFloat}
    horizon >= 1 || throw(ArgumentError("horizon must be positive"))
    1 <= t <= post.T_eff || throw(ArgumentError(
        "t must be in 1:$(post.T_eff), got $t"))
    n, p = post.n, post.p
    m = 1 + n * p
    N_avail = size(post.B_draws, 1)
    N = min(n_draws, N_avail)
    ql = T.(quantile_levels)

    sims = Array{T,4}(undef, N, horizon, n, n)
    valid = 0
    failed = 0
    companion = zeros(T, n * p, n * p)
    p > 1 && (companion[(n+1):end, 1:(n*(p-1))] = Matrix{T}(I, n * (p - 1), n * (p - 1)))

    for d in 1:N
        b = @view post.B_draws[d, t, :]
        # Unstack: equation i occupies rows ((i-1)m+1):(i m), first element the intercept
        A_lags = [zeros(T, n, n) for _ in 1:p]
        for i in 1:n, lag in 1:p, j in 1:n
            A_lags[lag][i, j] = b[(i-1)*m + 1 + (lag-1)*n + j]
        end
        for lag in 1:p
            companion[1:n, ((lag-1)*n+1):(lag*n)] = A_lags[lag]
        end
        if stationary_only && maximum(abs.(eigvals(companion))) >= one(T)
            failed += 1
            continue
        end

        At = _tvp_A_matrix(@view(post.A_draws[d, t, :]), n)
        P = Matrix{T}(robust_inv(LowerTriangular(At))) *
            Diagonal(exp.(@view(post.H_draws[d, t, :]) ./ 2))     # Σ_t = diag(exp(h/2))

        valid += 1
        Phi = [zeros(T, n, n) for _ in 1:horizon]
        copyto!(Phi[1], I(n))
        sims[valid, 1, :, :] = P
        for hh in 2:horizon
            acc = zeros(T, n, n)
            for lag in 1:min(p, hh - 1)
                acc .+= A_lags[lag] * Phi[hh-lag]
            end
            Phi[hh] .= acc
            sims[valid, hh, :, :] = acc * P
        end
    end
    valid == 0 && error("All posterior draws were explosive at t=$t")

    sims = sims[1:valid, :, :, :]
    point = dropdims(mean(sims; dims=1); dims=1)
    quant = Array{T,4}(undef, horizon, n, n, length(ql))
    for q in eachindex(ql), j in 1:n, i in 1:n, hh in 1:horizon
        quant[hh, i, j, q] = quantile(@view(sims[:, hh, i, j]), ql[q])
    end

    shocks = ["$(nm) shock" for nm in post.varnames]
    return BayesianImpulseResponse{T}(quant, point, horizon, copy(post.varnames), shocks,
                                       ql, sims, N, valid, failed)
end

"""
    volatility_path(post::TVPVARPosterior; quantile_levels=[0.16, 0.5, 0.84])
        -> (mean, quantiles)

Posterior mean and credible bands of the **standard-deviation** path
`σ_{i,t} = exp(h_{i,t}/2)`, where `h` is the stored log-variance state.
Returns a `T_eff × n` mean matrix and a `T_eff × n × n_q` quantile array.
"""
function volatility_path(post::TVPVARPosterior{T};
                         quantile_levels::Vector{<:Real}=[0.16, 0.5, 0.84]) where {T<:AbstractFloat}
    ql = T.(quantile_levels)
    vol = exp.(post.H_draws ./ 2)                   # σ = exp(h/2); n_draws × T_eff × n
    mu = dropdims(mean(vol; dims=1); dims=1)
    qs = Array{T,3}(undef, post.T_eff, post.n, length(ql))
    for q in eachindex(ql), j in 1:post.n, t in 1:post.T_eff
        qs[t, j, q] = quantile(@view(vol[:, t, j]), ql[q])
    end
    return mu, qs
end
