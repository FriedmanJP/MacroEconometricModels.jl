# =============================================================================
# Consolidated Parametric Kalman Kernel (T147 / #246)
# =============================================================================
# One correct forward filter shared by the domain-specific wrappers (ARIMA, BN,
# factor DFM, nowcast, DSGE). Predict-then-update, Joseph-stabilized covariance,
# `safe_cholesky` + triangular solves for the gain with the log-det taken from the
# SAME factor, an always-add likelihood (a degenerate innovation covariance is
# regularized + warned, never silently dropped), NaN entries treated as missing
# (row-dropped per step), and a scalar / rank-1 specialization for single-obs series.
#
# System (time-invariant matrices):
#     x_t = b + Tt x_{t-1} + η_t ,   Cov(η) = RQR   (RQR = R Q R', pre-formed by caller)
#     y_t = d + Z  x_t      + ν_t ,   Cov(ν) = Hobs
#
# Naming trap (R1): the core `_kalman_update(x_pred,P_pred,y,H,R)` names the
# OBSERVATION matrix `H` and the OBSERVATION-noise covariance `R`. Here `Z` is the
# observation matrix and `Hobs` the observation-noise covariance (Harvey convention).
# The kernel's inlined update is numerically identical to
# `_kalman_update(x_pred, P_pred, y-d, Z, Hobs)` plus the log-det from the same factor.

"""
    _kalman_init(mode, Tt, RQR, n_state; a0=nothing, P0=nothing, kappa=1e6, stationary_tol=1e-6)
        -> (a0::Vector{T}, P0::Matrix{T})

Build the forward filter's initial `(mean, covariance)`. Modes:
- `:explicit`   — pass `a0`,`P0` through (caller-supplied, e.g. nowcast).
- `:stationary` — `a0=0`; `P0` solves `P = Tt P Tt' + RQR` via `_dlyap_doubling`. Errors if `Tt` is not stable.
- `:diffuse`    — `a0=0`; `P0 = _diffuse_initial_covariance(Tt, RQR; kappa, tol)` (κ on the unit-root subspace, finite Lyapunov on the stationary complement). Collapses to `:stationary` when `Tt` is fully stable.
- `:kappa`      — `a0=0`; `P0 = kappa·I` (flat large-variance prior; the bit-compat bridge for the old `1e6·I` / `10·I` / `P[1,1]=1e6` fallbacks).
"""
function _kalman_init(mode::Symbol, Tt::AbstractMatrix{T}, RQR::AbstractMatrix{T},
                      n_state::Integer; a0=nothing, P0=nothing,
                      kappa::Real=1e6, stationary_tol::Real=1e-6) where {T<:AbstractFloat}
    if mode === :explicit
        (a0 === nothing || P0 === nothing) &&
            throw(ArgumentError("Kalman init :explicit requires both a0 and P0"))
        return Vector{T}(a0), Matrix{T}(P0)
    end
    a_init = a0 === nothing ? zeros(T, n_state) : Vector{T}(a0)
    if mode === :stationary
        rho = maximum(abs, eigvals(Matrix(Tt)))
        rho >= one(T) - T(stationary_tol) &&
            throw(ArgumentError("Kalman init :stationary but the transition is not stable " *
                "(max|eig| = $(rho) ≥ 1); use :diffuse or :kappa."))
        return a_init, Matrix{T}(_dlyap_doubling(Tt, RQR))
    elseif mode === :diffuse
        return a_init, Matrix{T}(_diffuse_initial_covariance(Tt, RQR; kappa=T(kappa), tol=T(stationary_tol)))
    elseif mode === :kappa
        return a_init, Matrix{T}(T(kappa) * Matrix{T}(I, n_state, n_state))
    end
    throw(ArgumentError("Unknown Kalman init mode :$mode " *
        "(expected :explicit, :stationary, :diffuse, or :kappa)"))
end

"""
    KalmanFilterStore{T}(n_state, T_obs; innovations=false)

Time-last (`[:,:,t]`) storage sink for the forward filter's predicted/filtered
moments — `a_pred`/`P_pred` (`a_{t|t-1}`, `P_{t|t-1}`) and `a_filt`/`P_filt`
(`a_{t|t}`, `P_{t|t}`) — consumed by `_rts_smoother`. When `innovations=true` it
also keeps per-step innovation `v`, innovation covariance `F`, and gain `K`
(ragged over the missing-data mask). Pass `store=nothing` to `_kalman_filter!`
for a log-likelihood-only run with zero storage allocation.
"""
mutable struct KalmanFilterStore{T<:AbstractFloat}
    a_pred::Matrix{T}
    P_pred::Array{T,3}
    a_filt::Matrix{T}
    P_filt::Array{T,3}
    v::Union{Nothing,Vector{Vector{T}}}
    F::Union{Nothing,Vector{Matrix{T}}}
    K::Union{Nothing,Vector{Matrix{T}}}
end
function KalmanFilterStore{T}(n_state::Integer, T_obs::Integer;
                             innovations::Bool=false) where {T<:AbstractFloat}
    KalmanFilterStore{T}(
        zeros(T, n_state, T_obs), zeros(T, n_state, n_state, T_obs),
        zeros(T, n_state, T_obs), zeros(T, n_state, n_state, T_obs),
        innovations ? Vector{Vector{T}}(undef, T_obs) : nothing,
        innovations ? Vector{Matrix{T}}(undef, T_obs) : nothing,
        innovations ? Vector{Matrix{T}}(undef, T_obs) : nothing,
    )
end

"""
    _kalman_filter!(store, y, Z, Tt, RQR, Hobs; d=nothing, b=nothing, a0, P0,
                    scalar=(size(y,1)==1)) -> loglik::T

Forward prediction-error-decomposition filter for
`x_t = b + Tt x_{t-1} + η_t` (`Cov η = RQR`), `y_t = d + Z x_t + ν_t` (`Cov ν = Hobs`),
with `NaN` entries of `y` treated as missing (row-dropped per step). `y` is
`n_obs × T_obs` (columns = periods). Returns `Σ_t log p(y_t | y_{1:t-1})` and, when
`store !== nothing`, fills the `KalmanFilterStore`. `scalar` forces the rank-1
observation specialization (auto for `n_obs == 1`).
"""
function _kalman_filter!(store, y::AbstractMatrix{T}, Z::AbstractMatrix{T},
                         Tt::AbstractMatrix{T}, RQR::AbstractMatrix{T}, Hobs::AbstractMatrix{T};
                         d=nothing, b=nothing, a0::AbstractVector{T}, P0::AbstractMatrix{T},
                         scalar::Bool=(size(y, 1) == 1)) where {T<:AbstractFloat}
    n_obs, T_obs = size(y)
    x = Vector{T}(a0)
    P = Matrix{T}(P0)
    d_vec = d === nothing ? nothing : Vector{T}(d)
    b_vec = b === nothing ? nothing : Vector{T}(b)
    ll = zero(T)
    log2pi = T(log(2 * pi))
    keep_innov = store !== nothing && store.v !== nothing
    for t in 1:T_obs
        # --- predict ---
        x_pred = b_vec === nothing ? Tt * x : b_vec + Tt * x
        P_pred = Tt * P * Tt' + RQR
        P_pred = (P_pred + P_pred') / 2
        yt = view(y, :, t)
        if scalar
            yi = @inbounds yt[1]
            if isnan(yi)
                x = x_pred; P = P_pred
            else
                Zv = vec(Z)                                  # n_state
                di = d_vec === nothing ? zero(T) : @inbounds d_vec[1]
                PZv = P_pred * Zv
                f = dot(Zv, PZv) + @inbounds Hobs[1, 1]
                if f <= zero(T)
                    @warn "Kalman kernel: non-positive scalar innovation variance; regularized." maxlog = 3
                    f = max(f, eps(T))
                end
                vi = yi - di - dot(Zv, x_pred)
                Kv = PZv ./ f
                x = x_pred + Kv * vi
                IKH = I - Kv * Zv'
                P = IKH * P_pred * IKH' + (@inbounds Hobs[1, 1]) * (Kv * Kv')
                P = (P + P') / 2
                ll += -T(0.5) * (log2pi + log(f) + vi^2 / f)
                if keep_innov
                    store.v[t] = T[vi]
                    store.F[t] = fill(T(f), 1, 1)
                    store.K[t] = reshape(Kv, :, 1)
                end
            end
        else
            mask = .!isnan.(yt)
            m = count(mask)
            if m == 0
                x = x_pred; P = P_pred
            else
                Zt = Z[mask, :]
                Ht = Hobs[mask, mask]
                yobs = yt[mask]
                dt = d_vec === nothing ? zeros(T, m) : d_vec[mask]
                v = yobs - dt - Zt * x_pred
                S = Zt * P_pred * Zt' + Ht
                S = (S + S') / 2
                L = safe_cholesky(Hermitian(S))              # lower factor, S = L L'
                logdetS = 2 * sum(log, diag(L))
                Kt = Matrix{T}((L' \ (L \ (Zt * P_pred')))')  # P_pred Zt' S^{-1} via tri-solves
                x = x_pred + Kt * v
                IKH = I - Kt * Zt
                P = IKH * P_pred * IKH' + Kt * Ht * Kt'
                P = (P + P') / 2
                w = L \ v
                ll += -T(0.5) * (m * log2pi + logdetS + dot(w, w))
                if keep_innov
                    store.v[t] = v
                    store.F[t] = S
                    store.K[t] = Kt
                end
            end
        end
        if store !== nothing
            @inbounds store.a_pred[:, t] = x_pred
            @inbounds store.P_pred[:, :, t] = P_pred
            @inbounds store.a_filt[:, t] = x
            @inbounds store.P_filt[:, :, t] = P
        end
    end
    return ll
end
