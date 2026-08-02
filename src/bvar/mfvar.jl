# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Mixed-frequency Bayesian VAR (T251 / #350).

Schorfheide & Song (2015) cast the VAR at the **high** frequency and treat the
low-frequency series as latent high-frequency processes observed only at reference
periods, linked to the latent path by a deterministic temporal-aggregation rule. That is a
linear Gaussian state-space model with a periodic observation matrix and missing rows in
between, so a two-block Gibbs sampler alternates between

1. the VAR parameters `(B, Σ)` given the completed high-frequency path — the same conjugate
   Normal-Inverse-Wishart draw the conjugate BVAR uses, Minnesota dummies included; and
2. the latent high-frequency path given `(B, Σ)` — a Durbin-Koopman simulation smoother on
   the companion state with the observation rows present at each date.

Provides:
- `MFVARPosterior` — parameter draws plus the latent high-frequency paths
- `estimate_mfvar` — the Gibbs sampler
- `latent_path` — posterior mean and bands of the interpolated low-frequency series

References:
- Schorfheide, F. & Song, D. (2015). Real-Time Forecasting with a Mixed-Frequency VAR.
  *Journal of Business & Economic Statistics*, 33(3), 366-380.
- Mariano, R. S. & Murasawa, Y. (2003). A New Coincident Index of Business Cycles Based on
  Monthly and Quarterly Series. *Journal of Applied Econometrics*, 18(4), 427-443.
- Durbin, J. & Koopman, S. J. (2002). A Simple and Efficient Simulation Smoother for
  State Space Time Series Analysis. *Biometrika*, 89(3), 603-616.
"""

using LinearAlgebra
using Random
using Statistics

# =============================================================================
# Temporal aggregation
# =============================================================================

"""
    _mf_agg_weights(kind, m, ::Type{T}) -> Vector{T}

Temporal-aggregation weights linking a latent high-frequency series to its observed
low-frequency counterpart at a frequency ratio `m`. The observation at a reference date `t`
is `Σⱼ w[j]·z_{t-j+1}`.

- `:stock` — `[1]`; the low-frequency observation is the end-of-period level.
- `:flow` — `ones(m)`; the observation is the sum over the reference period.
- `:average` — `ones(m)/m`; the observation is the within-period mean.
- `:growth` — the Mariano-Murasawa (2003) triangular weights `[1, 2, …, m, …, 2, 1]/m` over
  `2m-1` high-frequency lags. This is the standard approximation for the low-frequency
  growth rate of a variable whose high-frequency log-level growth is `z`: the quarterly
  growth of a quarterly average is, to first order, that triangular filter of monthly
  growth rates.
"""
function _mf_agg_weights(kind::Symbol, m::Int, ::Type{T}) where {T<:AbstractFloat}
    m >= 1 || throw(ArgumentError("frequency ratio must be ≥ 1, got $m"))
    if kind === :stock
        return T[one(T)]
    elseif kind === :flow
        return ones(T, m)
    elseif kind === :average
        return fill(one(T) / T(m), m)
    elseif kind === :growth
        w = Vector{T}(undef, 2m - 1)
        for j in 1:(2m-1)
            w[j] = T(min(j, 2m - j)) / T(m)
        end
        return w
    end
    throw(ArgumentError(
        "aggregation must be :stock, :flow, :average or :growth, got :$kind"))
end

# =============================================================================
# Posterior container
# =============================================================================

"""
    MFVARPosterior{T}

Posterior draws from a mixed-frequency Bayesian VAR.

# Fields
- `B_draws::Array{T,3}` — `n_draws × k × n` VAR coefficients, `k = 1 + np`
- `Sigma_draws::Array{T,3}` — `n_draws × n × n` innovation covariances
- `Z_draws::Array{T,3}` — `n_draws × T_hf × n` latent **high-frequency** paths; for the
  high-frequency series these reproduce the data, for the low-frequency ones they are the
  interpolated series
- `data::Matrix{T}` — the input `T_hf × n` panel, `NaN` where unobserved
- `p::Int`, `n::Int`, `T_hf::Int` — lag order, variables, high-frequency sample length
- `low_freq::Vector{Int}` — column indices treated as low frequency
- `freq_ratio::Int` — high-frequency periods per low-frequency period
- `aggregation::Vector{Symbol}` — aggregation rule per low-frequency series
- `varnames::Vector{String}` — variable names
"""
struct MFVARPosterior{T<:AbstractFloat}
    B_draws::Array{T,3}
    Sigma_draws::Array{T,3}
    Z_draws::Array{T,3}
    data::Matrix{T}
    p::Int
    n::Int
    T_hf::Int
    low_freq::Vector{Int}
    freq_ratio::Int
    aggregation::Vector{Symbol}
    varnames::Vector{String}
end

n_draws(post::MFVARPosterior) = size(post.B_draws, 1)

function Base.show(io::IO, post::MFVARPosterior{T}) where {T}
    n_lf = length(post.low_freq)
    spec = Any[
        "Variables"             post.n;
        "Lags"                  post.p;
        "High-freq. obs."       post.T_hf;
        "Low-freq. series"      n_lf == 0 ? "none (single frequency)" :
                                join(post.varnames[post.low_freq], ", ");
        "Frequency ratio"       post.freq_ratio;
        "Aggregation"           n_lf == 0 ? "—" : join(string.(post.aggregation), ", ");
        "Posterior draws"       n_draws(post)
    ]
    _pretty_table(io, spec;
        title = "Mixed-Frequency VAR (Schorfheide-Song)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    if n_lf > 0
        mu, qs = latent_path(post)
        idx = unique(clamp.([1, post.T_hf ÷ 3, 2 * post.T_hf ÷ 3, post.T_hf], 1, post.T_hf))
        data = Matrix{Any}(undef, length(idx), 1 + 3 * n_lf)
        for (r, t) in enumerate(idx)
            data[r, 1] = t
            for (c, j) in enumerate(post.low_freq)
                data[r, 1 + 3(c-1) + 1] = _fmt(qs[t, j, 1])
                data[r, 1 + 3(c-1) + 2] = _fmt(mu[t, j])
                data[r, 1 + 3(c-1) + 3] = _fmt(qs[t, j, 3])
            end
        end
        labels = ["t"]
        for j in post.low_freq
            push!(labels, "$(post.varnames[j]) lo", "$(post.varnames[j])", "$(post.varnames[j]) hi")
        end
        _pretty_table(io, data;
            title = "Interpolated high-frequency path (16%/mean/84%)",
            column_labels = labels,
            alignment = vcat(:r, fill(:r, 3 * n_lf)),
        )
    end
    return nothing
end

# =============================================================================
# Observation-equation construction
# =============================================================================

"""
    _mf_obs_rows(t, data, low_freq, weights, n, L) -> (Z_t, y_t)

Observation matrix and vector at high-frequency date `t`. High-frequency series contribute
a unit row on `z_t`; a low-frequency series contributes a row only at its reference dates,
carrying its aggregation weights across the state's lag blocks. Dates at which nothing is
observed return an empty system, which the filter handles by propagating the prediction.
"""
function _mf_obs_rows(t::Int, data::Matrix{T}, is_low::Vector{Bool},
                      weights::Vector{Vector{T}}, n::Int, L::Int) where {T<:AbstractFloat}
    rows = Int[]
    for i in 1:n
        isnan(data[t, i]) && continue
        push!(rows, i)
    end
    isempty(rows) && return (zeros(T, 0, n * L), zeros(T, 0))

    Z = zeros(T, length(rows), n * L)
    yv = Vector{T}(undef, length(rows))
    for (r, i) in enumerate(rows)
        yv[r] = data[t, i]
        if is_low[i]
            w = weights[i]
            for (j, wj) in enumerate(w)
                # lag j-1 lives in state block j, variable i
                Z[r, (j-1)*n + i] = wj
            end
        else
            Z[r, i] = one(T)
        end
    end
    return Z, yv
end

# =============================================================================
# Carter-Kohn FFBS on the VAR companion state
# =============================================================================

"""
    _mf_companion(B, Sigma, n, p, L) -> (Tm, drift, Q)

Companion-form transition for the state `s_t = [z_t; z_{t-1}; …; z_{t-L+1}]`, with the VAR
intercept carried as a known drift and the singular state-noise covariance (only the top
`n` rows carry innovations).
"""
function _mf_companion(B::Matrix{T}, Sigma::Matrix{T}, n::Int, p::Int,
                       L::Int) where {T<:AbstractFloat}
    sd = n * L
    Tm = zeros(T, sd, sd)
    for lag in 1:p
        Tm[1:n, ((lag-1)*n+1):(lag*n)] = transpose(@view B[(2 + (lag-1)*n):(1 + lag*n), :])
    end
    L > 1 && (Tm[(n+1):sd, 1:(sd-n)] = Matrix{T}(I, sd - n, sd - n))
    drift = zeros(T, sd)
    drift[1:n] = @view B[1, :]
    Q = zeros(T, sd, sd)
    Q[1:n, 1:n] = Sigma
    return Tm, drift, Q
end

"""
    _mf_smooth_mean(data, Tm, Q, is_low, weights, n, L, P0; jitter) -> Matrix

Kalman filter plus Rauch-Tung-Striebel mean smoother on the companion state, with the
observation rows present at each date (none at non-reference dates) and **no** drift — the
routine is used on a differenced series in which the drift cancels. Returns `E[s_t | y_{1:T}]`
as a `T × sd` matrix.

Only the smoothed mean is needed: the Durbin-Koopman construction supplies the sampling
variability separately, which is why no backward covariance recursion appears here.
"""
function _mf_smooth_mean(data::Matrix{T}, Tm::Matrix{T}, Q::Matrix{T},
                         is_low::Vector{Bool}, weights::Vector{Vector{T}},
                         n::Int, L::Int, P0::Matrix{T};
                         jitter::T=T(1e-8)) where {T<:AbstractFloat}
    T_hf = size(data, 1)
    sd = n * L
    a_filt = Matrix{T}(undef, T_hf, sd)
    P_filt = Array{T,3}(undef, T_hf, sd, sd)
    a_pred = Matrix{T}(undef, T_hf, sd)
    P_pred = Array{T,3}(undef, T_hf, sd, sd)

    a = zeros(T, sd)
    P = Matrix{T}(P0)
    for t in 1:T_hf
        ap = Tm * a
        Pp = Tm * P * Tm' + Q
        Pp = T(0.5) * (Pp + Pp')
        a_pred[t, :] = ap
        P_pred[t, :, :] = Pp

        Z, yv = _mf_obs_rows(t, data, is_low, weights, n, L)
        if isempty(yv)
            a = ap; P = Pp
        else
            PZ = Pp * Z'
            F = Z * PZ
            @inbounds for i in 1:size(F, 1)
                F[i, i] += jitter
            end
            F = T(0.5) * (F + F')
            K = PZ * robust_inv(Symmetric(F))
            a = ap .+ K * (yv .- Z * ap)
            P = Pp .- K * (Z * Pp)
            P = T(0.5) * (P + P')
        end
        a_filt[t, :] = a
        P_filt[t, :, :] = P
    end

    out = Matrix{T}(undef, T_hf, sd)
    out[T_hf, :] = @view a_filt[T_hf, :]
    for t in (T_hf-1):-1:1
        Pf = Matrix{T}(@view P_filt[t, :, :])
        Pp = Matrix{T}(@view P_pred[t+1, :, :])
        @inbounds for i in 1:sd
            Pp[i, i] += jitter
        end
        J = (Pf * Tm') * robust_inv(Symmetric(Pp))
        out[t, :] = a_filt[t, :] .+ J * (out[t+1, :] .- a_pred[t+1, :])
    end
    return out
end

"""
    _mf_draw_states(data, B, Sigma, is_low, weights, n, p, L, rng; jitter) -> Matrix

Draw the latent high-frequency path by the **Durbin & Koopman (2002) simulation smoother**.

A backward sampler on this model is not straightforward: the companion state noise is
singular, so the Kim-Nelson backward step conditions only on the top block `z_{t+1}` — but
a temporal-aggregation row links `z_t, z_{t-1}, …` across several lag blocks, and lags
redrawn at successive backward steps are then mutually inconsistent with that constraint.
The aggregation identity fails as a result.

Durbin-Koopman sidesteps this entirely. Simulate an unconditional path `s⁺` and the
observations `y⁺` it implies, smooth the difference `y − y⁺` (a linear operation, hence the
drift cancels and no intercept is needed), and set `s̃ = ŝ(y − y⁺) + s⁺`. Because the
temporal aggregation is noiseless, applying the observation map to `s̃` returns
`(y − y⁺) + y⁺ = y` **exactly** at every reference date, so the identity holds by
construction rather than approximately.
"""
function _mf_draw_states(data::Matrix{T}, B::Matrix{T}, Sigma::Matrix{T},
                         is_low::Vector{Bool}, weights::Vector{Vector{T}},
                         n::Int, p::Int, L::Int, rng::AbstractRNG;
                         jitter::T=T(1e-8)) where {T<:AbstractFloat}
    T_hf = size(data, 1)
    sd = n * L
    Tm, drift, Q = _mf_companion(B, Sigma, n, p, L)
    P0 = Matrix{T}(I, sd, sd) * T(10)

    # 1. Unconditional draw of the state path (with drift) and the observations it implies
    L_S = safe_cholesky(Symmetric(Sigma))
    s = _psd_sqrt(P0) * randn(rng, T, sd)
    s_plus = Matrix{T}(undef, T_hf, sd)
    y_plus = fill(T(NaN), T_hf, n)
    for t in 1:T_hf
        u = zeros(T, sd)
        u[1:n] = L_S * randn(rng, T, n)
        s = Tm * s .+ drift .+ u
        s_plus[t, :] = s
        for i in 1:n
            isnan(data[t, i]) && continue
            if is_low[i]
                w = weights[i]
                acc = zero(T)
                for (j, wj) in enumerate(w)
                    acc += wj * s[(j-1)*n + i]
                end
                y_plus[t, i] = acc
            else
                y_plus[t, i] = s[i]
            end
        end
    end

    # 2. Smooth the difference (drift-free by construction) and recombine
    y_diff = data .- y_plus
    s_hat = _mf_smooth_mean(y_diff, Tm, Q, is_low, weights, n, L, P0; jitter=jitter)
    return Matrix{T}(s_hat[:, 1:n] .+ s_plus[:, 1:n])
end

# =============================================================================
# Conditional NIW parameter draw
# =============================================================================

"""
    _mf_draw_niw(Z, p, prior, hyper, rng) -> (B, Sigma)

One conjugate Normal-Inverse-Wishart draw of `(B, Σ)` given a completed high-frequency
path, using the same diffuse prior and optional Minnesota dummy observations as
`estimate_bvar`.
"""
function _mf_draw_niw(Z::Matrix{T}, p::Int, prior::Symbol,
                      hyper::Union{Nothing,MinnesotaHyperparameters},
                      rng::AbstractRNG) where {T<:AbstractFloat}
    n = size(Z, 2)
    Y_eff, X = construct_var_matrices(Z, p)
    k = size(X, 2)

    Y_data, X_data = if prior === :minnesota
        h = hyper === nothing ? optimize_hyperparameters(Z, p) : hyper
        Yd, Xd = gen_dummy_obs(Z, p, h)
        (vcat(Y_eff, Yd), vcat(X, Xd))
    else
        (Y_eff, X)
    end

    kappa = T(100)
    V0_inv = (one(T) / kappa) * Matrix{T}(I, k, k)
    B0 = zeros(T, k, n)
    nu0 = n + 2
    S0 = Matrix{T}(I, n, n)

    V_post_inv = X_data' * X_data + V0_inv
    V_post = Matrix{T}(robust_inv(Symmetric(V_post_inv)))
    V_post = T(0.5) * (V_post + V_post')
    B_post = V_post * (X_data' * Y_data + V0_inv * B0)
    nu_post = nu0 + size(Y_data, 1)
    S_post = S0 + Y_data' * Y_data + B0' * V0_inv * B0 - B_post' * V_post_inv * B_post
    S_post = Matrix{T}(Symmetric(T(0.5) * (S_post + S_post')))

    Sigma = Matrix{T}(_draw_inverse_wishart(nu_post, S_post, rng))
    L_V = safe_cholesky(Symmetric(V_post))
    L_S = safe_cholesky(Symmetric(Sigma))
    B = B_post + L_V * randn(rng, T, k, n) * L_S'
    return B, Sigma
end

# =============================================================================
# Initialization
# =============================================================================

"""
    _mf_initial_path(data, is_low, weights) -> Matrix

Seed the latent path: high-frequency columns pass through, low-frequency columns are filled
by linear interpolation between reference dates, rescaled so the aggregation identity holds
approximately at the observed dates (dividing by the weight sum). Only the starting point
of the chain depends on this.
"""
function _mf_initial_path(data::Matrix{T}, is_low::Vector{Bool},
                          weights::Vector{Vector{T}}) where {T<:AbstractFloat}
    T_hf, n = size(data)
    Z = Matrix{T}(undef, T_hf, n)
    for i in 1:n
        col = @view data[:, i]
        obs = findall(!isnan, col)
        if isempty(obs)
            Z[:, i] .= zero(T)
            continue
        end
        scale = is_low[i] ? sum(weights[i]) : one(T)
        for t in 1:T_hf
            if !isnan(col[t])
                Z[t, i] = col[t] / scale
            else
                # nearest observed neighbours, linearly interpolated
                lo = findlast(o -> o < t, obs)
                hi = findfirst(o -> o > t, obs)
                if lo === nothing
                    Z[t, i] = col[obs[hi]] / scale
                elseif hi === nothing
                    Z[t, i] = col[obs[lo]] / scale
                else
                    t0, t1 = obs[lo], obs[hi]
                    w = T(t - t0) / T(t1 - t0)
                    Z[t, i] = ((1 - w) * col[t0] + w * col[t1]) / scale
                end
            end
        end
    end
    return Z
end

# =============================================================================
# Public API
# =============================================================================

"""
    estimate_mfvar(data, p; low_freq=Int[], freq_ratio=3, aggregation=:growth, kwargs...)
        -> MFVARPosterior

Mixed-frequency Bayesian VAR (Schorfheide & Song 2015).

`data` is a **high-frequency** `T × n` panel with `NaN` wherever a series is not observed:
a quarterly series in a monthly panel carries a value every third row and `NaN` elsewhere.
The VAR itself lives entirely at the high frequency; the low-frequency observations are
temporal aggregates of the latent high-frequency path,

```math
y^{lo}_{i,t} = \\sum_{j} w_j\\, z_{i,t-j+1}
```

with weights set by `aggregation` (see `_mf_agg_weights` for the exact rules;
`:growth` uses the Mariano-Murasawa triangular filter).

A two-block Gibbs sampler alternates between the conjugate NIW draw of `(B, Σ)` given the
completed path and a Durbin-Koopman simulation-smoother draw of the path given `(B, Σ)`. With `low_freq` empty the
first block is the only source of randomness in the parameters and the sampler reduces to
the conjugate BVAR.

# Arguments
- `data::AbstractMatrix` — `T × n` high-frequency panel, `NaN` where unobserved
- `p::Int` — lag order at the high frequency

# Keywords
- `low_freq::Vector{Int}=Int[]` — column indices observed at the low frequency
- `freq_ratio::Int=3` — high-frequency periods per low-frequency period (3 = monthly/quarterly)
- `aggregation::Union{Symbol,Vector{Symbol}}=:growth` — `:growth`, `:flow`, `:average` or
  `:stock`, either one rule for all low-frequency series or one per series
- `n_draws::Int=1000` — retained posterior draws
- `n_burn::Int=500` — burn-in sweeps
- `prior::Symbol=:minnesota` — `:minnesota` (dummy observations) or `:diffuse`
- `hyper::Union{Nothing,MinnesotaHyperparameters}=nothing` — fixed Minnesota hyperparameters;
  `nothing` optimizes them once on the initial path
- `varnames::Vector{String}` — variable names
- `rng::AbstractRNG=Random.default_rng()` — random number generator

# Returns
[`MFVARPosterior`](@ref).

# References
- Schorfheide, F. & Song, D. (2015). *Journal of Business & Economic Statistics*, 33(3), 366-380.
"""
function estimate_mfvar(data::AbstractMatrix{T}, p::Int;
                        low_freq::Vector{Int}=Int[],
                        freq_ratio::Int=3,
                        aggregation::Union{Symbol,Vector{Symbol}}=:growth,
                        n_draws::Int=1000, n_burn::Int=500,
                        prior::Symbol=:minnesota,
                        hyper::Union{Nothing,MinnesotaHyperparameters}=nothing,
                        varnames::Vector{String}=String[],
                        rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    p >= 1 || throw(ArgumentError("p must be at least 1, got $p"))
    n_draws >= 1 || throw(ArgumentError("n_draws must be positive"))
    n_burn >= 0 || throw(ArgumentError("n_burn must be non-negative"))
    prior in (:minnesota, :diffuse) ||
        throw(ArgumentError("prior must be :minnesota or :diffuse, got :$prior"))
    freq_ratio >= 1 || throw(ArgumentError("freq_ratio must be ≥ 1, got $freq_ratio"))

    dm = Matrix{T}(data)
    T_hf, n = size(dm)
    n >= 1 || throw(ArgumentError("data must have at least one column"))
    all(1 .<= low_freq .<= n) || throw(ArgumentError(
        "low_freq indices must be in 1:$n, got $low_freq"))
    length(unique(low_freq)) == length(low_freq) ||
        throw(ArgumentError("low_freq must not repeat an index"))
    vn = isempty(varnames) ? ["y$i" for i in 1:n] : copy(varnames)
    length(vn) == n || throw(ArgumentError("varnames must have length $n"))

    aggs = aggregation isa Symbol ? fill(aggregation, length(low_freq)) : copy(aggregation)
    length(aggs) == length(low_freq) || throw(ArgumentError(
        "aggregation must be a Symbol or one Symbol per low_freq series " *
        "($(length(low_freq)) expected, got $(length(aggs)))"))

    is_low = fill(false, n)
    weights = [T[one(T)] for _ in 1:n]
    for (c, j) in enumerate(low_freq)
        is_low[j] = true
        weights[j] = _mf_agg_weights(aggs[c], freq_ratio, T)
    end

    # High-frequency columns must be fully observed; low-frequency ones must have some data
    for i in 1:n
        if is_low[i]
            any(!isnan, @view dm[:, i]) || throw(ArgumentError(
                "low-frequency series $(vn[i]) has no observations"))
        else
            any(isnan, @view dm[:, i]) && throw(ArgumentError(
                "high-frequency series $(vn[i]) contains NaN; either list it in low_freq " *
                "or supply a complete series"))
        end
    end

    L = max(p, maximum(length(w) for w in weights))
    T_hf > L + 1 || throw(ArgumentError(
        "need more than $(L + 1) high-frequency observations, got $T_hf"))

    Z = _mf_initial_path(dm, is_low, weights)
    hyper_fixed = hyper
    if prior === :minnesota && hyper_fixed === nothing
        # Optimize once on the initial path rather than every sweep: re-optimizing inside
        # the loop would make the prior itself a function of the current state draw.
        hyper_fixed = optimize_hyperparameters(Z, p)
    end

    k = 1 + n * p
    B_out = Array{T,3}(undef, n_draws, k, n)
    S_out = Array{T,3}(undef, n_draws, n, n)
    Z_out = Array{T,3}(undef, n_draws, T_hf, n)

    any_low = !isempty(low_freq)
    kept = 0
    for sweep in 1:(n_burn + n_draws)
        B, Sigma = _mf_draw_niw(Z, p, prior, hyper_fixed, rng)
        if any_low
            Z = _mf_draw_states(dm, B, Sigma, is_low, weights, n, p, L, rng)
            # The observed high-frequency series are data, not states: restore them exactly
            # so filter roundoff cannot drift them away from their observations.
            for i in 1:n
                is_low[i] || (Z[:, i] = @view dm[:, i])
            end
        end
        if sweep > n_burn
            kept += 1
            B_out[kept, :, :] = B
            S_out[kept, :, :] = Sigma
            Z_out[kept, :, :] = Z
        end
    end

    return MFVARPosterior{T}(B_out, S_out, Z_out, dm, p, n, T_hf,
                             copy(low_freq), freq_ratio, aggs, vn)
end

estimate_mfvar(data::AbstractMatrix, p::Int; kwargs...) =
    estimate_mfvar(Float64.(data), p; kwargs...)

"""
    latent_path(post::MFVARPosterior; quantile_levels=[0.16, 0.5, 0.84]) -> (mean, quantiles)

Posterior mean and credible bands of the latent high-frequency paths. Returns a
`T_hf × n` mean matrix and a `T_hf × n × n_q` quantile array. For high-frequency series the
path reproduces the data exactly, so its bands are degenerate.
"""
function latent_path(post::MFVARPosterior{T};
                     quantile_levels::Vector{<:Real}=[0.16, 0.5, 0.84]) where {T<:AbstractFloat}
    ql = T.(quantile_levels)
    mu = dropdims(mean(post.Z_draws; dims=1); dims=1)
    qs = Array{T,3}(undef, post.T_hf, post.n, length(ql))
    for q in eachindex(ql), j in 1:post.n, t in 1:post.T_hf
        qs[t, j, q] = quantile(@view(post.Z_draws[:, t, j]), ql[q])
    end
    return mu, qs
end

"""
    _mf_as_bvar(post::MFVARPosterior) -> BVARPosterior

View the parameter draws as a `BVARPosterior` so the existing BVAR `irf`, `fevd`,
`historical_decomposition` and `forecast` dispatches apply unchanged. The data attached to
the view is the posterior-mean latent high-frequency path, which is the series those
routines should condition on.
"""
function _mf_as_bvar(post::MFVARPosterior{T}) where {T<:AbstractFloat}
    Zbar = dropdims(mean(post.Z_draws; dims=1); dims=1)
    return BVARPosterior{T}(post.B_draws, post.Sigma_draws, n_draws(post), post.p, post.n,
                            Matrix{T}(Zbar), :minnesota, :mfvar, copy(post.varnames))
end

"""
    irf(post::MFVARPosterior, horizon; kwargs...)

Impulse responses at the **high** frequency, computed from the parameter draws exactly as
for a conjugate BVAR (see `irf(::BVARPosterior, ...)`).
"""
irf(post::MFVARPosterior, horizon::Int; kwargs...) = irf(_mf_as_bvar(post), horizon; kwargs...)

"""
    forecast(post::MFVARPosterior, h; kwargs...)

High-frequency forecasts with posterior credible bands, iterating each parameter draw
forward from the posterior-mean latent path (see `forecast(::BVARPosterior, ...)`).
"""
forecast(post::MFVARPosterior, h::Int; kwargs...) = forecast(_mf_as_bvar(post), h; kwargs...)
