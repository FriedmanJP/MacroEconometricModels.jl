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
- `:stationary` — `a0=0`; `P0` solves `P = Tt P Tt' + RQR` via `_dlyap` (doubling, residual-checked, Bartels-Stewart fallback). Errors if `Tt` is not stable.
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
        return a_init, Matrix{T}(_dlyap(Matrix{T}(Tt), Matrix{T}(RQR);
                                        warn_label="Kalman :stationary init"))
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

# =============================================================================
# Steady-State Kalman Filter
# =============================================================================

"""
    SteadyStateKalman{T<:AbstractFloat}

Steady-state Kalman filter solution obtained by solving the discrete-time algebraic
Riccati equation (DARE) for the one-step-ahead predicted state covariance.

# Fields
- `P_pred::Matrix{T}` — steady-state predicted state covariance `P_{∞|∞-1}` (`n_state × n_state`)
- `P_filt::Matrix{T}` — steady-state filtered state covariance `P_{∞|∞}` (`n_state × n_state`)
- `K_gain::Matrix{T}` — steady-state filtered Kalman gain `P_pred Z' F_innov^{-1}` (`n_state × n_obs`)
- `F_innov::Matrix{T}` — steady-state innovation covariance `Z P_pred Z' + H` (`n_obs × n_obs`)
- `closed_loop_eigs::Vector{Complex{T}}` — closed-loop transition eigenvalues `eigvals(Tt - K_dare * Z)`
"""
struct SteadyStateKalman{T<:AbstractFloat}
    P_pred::Matrix{T}
    P_filt::Matrix{T}
    K_gain::Matrix{T}
    F_innov::Matrix{T}
    closed_loop_eigs::Vector{Complex{T}}
end

function Base.show(io::IO, ssk::SteadyStateKalman{T}) where {T}
    n_state, n_obs = size(ssk.K_gain)
    rho = isempty(ssk.closed_loop_eigs) ? zero(T) : maximum(abs, ssk.closed_loop_eigs)
    print(io, "SteadyStateKalman{$T}(n_state=$(n_state), n_obs=$(n_obs), max|eig|=$(round(rho, digits=4)))")
end

"""
    steady_state_kalman(Z, H, Tt, RQR; G=nothing) -> SteadyStateKalman

Compute the steady-state Kalman filter for the time-invariant linear-Gaussian system
(Durbin & Koopman 2012, §4.11; Anderson & Moore 1979, §5.5):

    α_{t+1} = b + Tt α_t + η_t,   y_t = d + Z α_t + ν_t,   Cov(η_t, ν_t) = G

Maps directly to `_solve_dare` with `(A=Tt, Q=RQR, C=Z, R=H, S=G)`, returning
the steady-state predicted covariance `P_pred`, filtered covariance `P_filt`, filtered
Kalman gain `K_gain = P_pred * Z' * F_innov^{-1}` (matching `store.K[end]` in
`_kalman_filter!`), innovation covariance `F_innov = Z * P_pred * Z' + H`, and closed-loop
eigenvalues of `Tt - K_dare * Z` where `K_dare = (Tt * P_pred * Z' + G) * F_innov^{-1}`.
"""
function steady_state_kalman(Z::AbstractMatrix, H::AbstractMatrix,
                             Tt::AbstractMatrix, RQR::AbstractMatrix;
                             G::Union{Nothing,AbstractMatrix}=nothing)
    n_obs, n_state = size(Z)
    size(H) == (n_obs, n_obs) || throw(ArgumentError("H must be $n_obs×$n_obs, got $(size(H))"))
    size(Tt) == (n_state, n_state) || throw(ArgumentError("Tt must be $n_state×$n_state, got $(size(Tt))"))
    size(RQR) == (n_state, n_state) || throw(ArgumentError("RQR must be $n_state×$n_state, got $(size(RQR))"))
    if G !== nothing
        size(G) == (n_state, n_obs) || throw(ArgumentError("G must be $n_state×$n_obs, got $(size(G))"))
    end

    T = G === nothing ?
        float(promote_type(eltype(Z), eltype(H), eltype(Tt), eltype(RQR))) :
        float(promote_type(eltype(Z), eltype(H), eltype(Tt), eltype(RQR), eltype(G)))

    Zm = Matrix{T}(Z)
    Hraw = Matrix{T}(H)
    Hm = Matrix{T}((Hraw + Hraw') / 2)
    Tm = Matrix{T}(Tt)
    RQRraw = Matrix{T}(RQR)
    RQRm = Matrix{T}((RQRraw + RQRraw') / 2)
    Gm = G === nothing ? nothing : Matrix{T}(G)

    P_pred, K_dare, F_innov, _ = _solve_dare(Tm, RQRm, Zm, Hm; S=Gm)

    K_gain = if n_obs == 0
        zeros(T, n_state, 0)
    else
        try
            cf = cholesky(Hermitian(F_innov))
            Matrix{T}((cf.L' \ (cf.L \ (Zm * P_pred')))')
        catch
            luF = lu(F_innov; check=false)
            if issuccess(luF)
                Matrix{T}((luF \ (Zm * P_pred'))')
            else
                Matrix{T}(P_pred * Zm' * Matrix{T}(robust_inv(Hermitian(F_innov); silent=true)))
            end
        end
    end

    IKZ = Matrix{T}(I, n_state, n_state) - K_gain * Zm
    P_filt_raw = IKZ * P_pred * IKZ' + K_gain * Hm * K_gain'
    P_filt = Matrix{T}((P_filt_raw + P_filt_raw') / 2)
    closed_loop_eigs = Vector{Complex{T}}(eigvals(Tm - K_dare * Zm))

    return SteadyStateKalman{T}(P_pred, P_filt, K_gain, F_innov, closed_loop_eigs)
end

"""
    _kalman_filter!(store, y, Z, Tt, RQR, Hobs; d=nothing, b=nothing, a0, P0,
                    scalar=(size(y,1)==1), predict_first=true, G=nothing) -> loglik::T

Forward prediction-error-decomposition filter for
`x_{t+1} = b + Tt x_t + η_t` (`Cov η = RQR`), `y_t = d + Z x_t + ν_t` (`Cov ν = Hobs`),
with optional cross-covariance `G = Cov(η_t, ν_t)` (Durbin & Koopman 2012, §4.11;
Anderson & Moore 1979, §5.5), and `NaN` entries of `y` treated as missing (row-dropped per
step). `y` is `n_obs × T_obs` (columns = periods). Returns `Σ_t log p(y_t | y_{1:t-1})`
and, when `store !== nothing`, fills the `KalmanFilterStore`. `scalar` forces the rank-1
observation specialization (auto for `n_obs == 1`). When `G === nothing`, executes the
uncorrelated fast path bit-for-bit.
"""
function _kalman_filter!(store, y::AbstractMatrix{T}, Z::AbstractMatrix{T},
                         Tt::AbstractMatrix{T}, RQR::AbstractMatrix{T}, Hobs::AbstractMatrix{T};
                         d=nothing, b=nothing, a0::AbstractVector{T}, P0::AbstractMatrix{T},
                         scalar::Bool=(size(y, 1) == 1),
                         predict_first::Bool=true,
                         G::Union{Nothing,AbstractMatrix}=nothing) where {T<:AbstractFloat}
    n_obs, T_obs = size(y)
    x = Vector{T}(a0)
    P = Matrix{T}(P0)
    d_vec = d === nothing ? nothing : Vector{T}(d)
    b_vec = b === nothing ? nothing : Vector{T}(b)
    ll = zero(T)
    log2pi = T(log(2 * pi))
    keep_innov = store !== nothing && store.v !== nothing

    if G === nothing
        for t in 1:T_obs
            # --- predict (skipped at t=1 when predict_first=false: a0/P0 are the prior
            #     a_{1|0}/P_{1|0} directly, e.g. filters that seed the state at the first obs) ---
            if t == 1 && !predict_first
                x_pred = copy(x)
                P_pred = (P + P') / 2
            else
                x_pred = b_vec === nothing ? Tt * x : b_vec + Tt * x
                P_pred = Tt * P * Tt' + RQR
                P_pred = (P_pred + P_pred') / 2
            end
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

    # Correlated state-measurement noise path (G = Cov(η_t, ν_t), DK §4.11 / AM §5.5)
    n_state = size(Tt, 1)
    Gm = Matrix{T}(G)
    size(Gm) == (n_state, n_obs) ||
        throw(ArgumentError("G must be n_state×n_obs = $((n_state, n_obs)), got $(size(Gm))"))

    if predict_first
        x_pred = b_vec === nothing ? Tt * x : b_vec + Tt * x
        P_pred = Tt * P * Tt' + RQR
        P_pred = (P_pred + P_pred') / 2
    else
        x_pred = copy(x)
        P_pred = (P + P') / 2
    end

    for t in 1:T_obs
        yt = view(y, :, t)
        x_pred_next = x_pred
        P_pred_next = P_pred
        if scalar
            yi = @inbounds yt[1]
            if isnan(yi)
                x_filt = x_pred
                P_filt = P_pred
                if t < T_obs
                    x_pred_next = b_vec === nothing ? Tt * x_pred : b_vec + Tt * x_pred
                    P_pred_next = Tt * P_pred * Tt' + RQR
                    P_pred_next = (P_pred_next + P_pred_next') / 2
                end
            else
                Zv = vec(Z)
                Gv = vec(Gm)
                di = d_vec === nothing ? zero(T) : @inbounds d_vec[1]
                h11 = @inbounds Hobs[1, 1]
                PZv = P_pred * Zv
                f = dot(Zv, PZv) + h11
                if f <= zero(T)
                    @warn "Kalman kernel: non-positive scalar innovation variance; regularized." maxlog = 3
                    f = max(f, eps(T))
                end
                vi = yi - di - dot(Zv, x_pred)
                Kv = PZv ./ f
                x_filt = x_pred + Kv * vi
                IKH = I - Kv * Zv'
                P_filt = IKH * P_pred * IKH' + h11 * (Kv * Kv')
                P_filt = (P_filt + P_filt') / 2
                ll += -T(0.5) * (log2pi + log(f) + vi^2 / f)
                if keep_innov
                    store.v[t] = T[vi]
                    store.F[t] = fill(T(f), 1, 1)
                    store.K[t] = reshape(Kv, :, 1)
                end
                if t < T_obs
                    Kpv = (Tt * PZv + Gv) ./ f
                    x_pred_next = (b_vec === nothing ? Tt * x_pred : b_vec + Tt * x_pred) + Kpv * vi
                    IKZp = Tt - Kpv * Zv'
                    P_pred_next = IKZp * P_pred * IKZp' + RQR + h11 * (Kpv * Kpv') - Gv * Kpv' - Kpv * Gv'
                    P_pred_next = (P_pred_next + P_pred_next') / 2
                end
            end
        else
            mask = .!isnan.(yt)
            m = count(mask)
            if m == 0
                x_filt = x_pred
                P_filt = P_pred
                if t < T_obs
                    x_pred_next = b_vec === nothing ? Tt * x_pred : b_vec + Tt * x_pred
                    P_pred_next = Tt * P_pred * Tt' + RQR
                    P_pred_next = (P_pred_next + P_pred_next') / 2
                end
            else
                Zt = Z[mask, :]
                Ht = Hobs[mask, mask]
                Gt = Gm[:, mask]
                yobs = yt[mask]
                dt = d_vec === nothing ? zeros(T, m) : d_vec[mask]
                v = yobs - dt - Zt * x_pred
                S = Zt * P_pred * Zt' + Ht
                S = (S + S') / 2
                L = safe_cholesky(Hermitian(S))
                logdetS = 2 * sum(log, diag(L))
                Kt = Matrix{T}((L' \ (L \ (Zt * P_pred')))')
                x_filt = x_pred + Kt * v
                IKH = I - Kt * Zt
                P_filt = IKH * P_pred * IKH' + Kt * Ht * Kt'
                P_filt = (P_filt + P_filt') / 2
                w = L \ v
                ll += -T(0.5) * (m * log2pi + logdetS + dot(w, w))
                if keep_innov
                    store.v[t] = v
                    store.F[t] = S
                    store.K[t] = Kt
                end
                if t < T_obs
                    TPZt_G = Tt * P_pred * Zt' + Gt
                    K_pred = Matrix{T}((L' \ (L \ TPZt_G'))')
                    x_pred_next = (b_vec === nothing ? Tt * x_pred : b_vec + Tt * x_pred) + K_pred * v
                    IKZp = Tt - K_pred * Zt
                    P_pred_next = IKZp * P_pred * IKZp' + RQR + K_pred * Ht * K_pred' - Gt * K_pred' - K_pred * Gt'
                    P_pred_next = (P_pred_next + P_pred_next') / 2
                end
            end
        end
        if store !== nothing
            @inbounds store.a_pred[:, t] = x_pred
            @inbounds store.P_pred[:, :, t] = P_pred
            @inbounds store.a_filt[:, t] = x_filt
            @inbounds store.P_filt[:, :, t] = P_filt
        end
        if t < T_obs
            x_pred = x_pred_next
            P_pred = P_pred_next
        end
    end
    return ll
end

"""
    _rts_smoother(store, Tt; nlag=0, G=nothing, y=nothing) -> (a_smooth, P_smooth, Plag)

Rauch-Tung-Striebel fixed-interval smoother consuming a `KalmanFilterStore` filled by
`_kalman_filter!` (needs `a_pred`/`P_pred`/`a_filt`/`P_filt`). Time-last `[:,:,t]`
layout. The stored `a_pred` already includes any state intercept `b` (the kernel added
it during prediction), so no intercept is threaded here:

    a_{t|T} = a_{t|t} + J_t (a_{t+1|T} - a_{t+1|t}),   J_t = P_{t|t} Tt' P_{t+1|t}^{-1}
    P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t'

When `G !== nothing` (correlated noise `Cov(η_t, ν_t) = G`), `store` must be constructed
with `innovations=true` (so `store.K` is populated), and `J_t = (P_{t|t} Tt' - K_t^{filt} G_t') P_{t+1|t}^{-1}`
(pass `y` if partial `NaN` missingness occurs).

`nlag > 0` additionally returns lag-`j` smoothed cross-covariances `Plag[j]` with
`Plag[j][:,:,t] = Cov(x_t, x_{t-j} | Y_T)`, via the backward-Markov recursion
`Plag[1][:,:,t] = P_{t|T} J_{t-1}'`, `Plag[j][:,:,t] = Plag[j-1][:,:,t] J_{t-j}'`
(consumed e.g. by the DFM EM M-step and nowcast news decomposition). `Plag` is empty
when `nlag == 0`.
"""
function _rts_smoother(store::KalmanFilterStore{T}, Tt::AbstractMatrix{T};
                       nlag::Int=0,
                       G::Union{Nothing,AbstractMatrix}=nothing,
                       y::Union{Nothing,AbstractMatrix}=nothing) where {T<:AbstractFloat}
    n_state, T_obs = size(store.a_filt)
    a_smooth = similar(store.a_filt)
    P_smooth = similar(store.P_filt)
    J = nlag > 0 ? Array{T,3}(undef, n_state, n_state, T_obs) : Array{T,3}(undef, 0, 0, 0)
    @inbounds a_smooth[:, T_obs] = store.a_filt[:, T_obs]
    @inbounds P_smooth[:, :, T_obs] = store.P_filt[:, :, T_obs]
    if G === nothing
        for t in (T_obs-1):-1:1
            Pf = Matrix{T}(@view store.P_filt[:, :, t])
            Jt = _rts_smoother_gain(Pf, Tt, store.P_pred[:, :, t+1])
            nlag > 0 && (@inbounds J[:, :, t] = Jt)
            @inbounds a_smooth[:, t] = store.a_filt[:, t] + Jt * (a_smooth[:, t+1] - store.a_pred[:, t+1])
            Ps = Pf + Jt * (P_smooth[:, :, t+1] - store.P_pred[:, :, t+1]) * Jt'
            @inbounds P_smooth[:, :, t] = (Ps + Ps') / 2
        end
    else
        store.K === nothing &&
            throw(ArgumentError("_rts_smoother with G !== nothing requires KalmanFilterStore(..., innovations=true)"))
        Gm = Matrix{T}(G)
        size(Gm, 1) == n_state ||
            throw(ArgumentError("G must have n_state=$n_state rows, got $(size(Gm, 1))"))
        n_obs_G = size(Gm, 2)
        for t in (T_obs-1):-1:1
            Pf = Matrix{T}(@view store.P_filt[:, :, t])
            if !isassigned(store.K, t)
                Jt = _rts_smoother_gain(Pf, Tt, store.P_pred[:, :, t+1])
            else
                Kt = store.K[t]
                Gt = if size(Kt, 2) == n_obs_G
                    Gm
                else
                    y === nothing &&
                        throw(ArgumentError("_rts_smoother with partial NaNs and G !== nothing requires y keyword argument"))
                    mask = .!isnan.(view(y, :, t))
                    Gm[:, mask]
                end
                P_pred_inv = Matrix{T}(robust_inv(Hermitian(store.P_pred[:, :, t+1])))
                Jt = (Pf * Tt' - Kt * Gt') * P_pred_inv
            end
            nlag > 0 && (@inbounds J[:, :, t] = Jt)
            @inbounds a_smooth[:, t] = store.a_filt[:, t] + Jt * (a_smooth[:, t+1] - store.a_pred[:, t+1])
            Ps = Pf + Jt * (P_smooth[:, :, t+1] - store.P_pred[:, :, t+1]) * Jt'
            @inbounds P_smooth[:, :, t] = (Ps + Ps') / 2
        end
    end
    Plag = Vector{Array{T,3}}(undef, nlag)
    for j in 1:nlag
        Plag[j] = zeros(T, n_state, n_state, T_obs)
    end
    if nlag >= 1
        @inbounds for t in 2:T_obs
            Plag[1][:, :, t] = P_smooth[:, :, t] * J[:, :, t-1]'
        end
        @inbounds for j in 2:nlag, t in (j+1):T_obs
            Plag[j][:, :, t] = Plag[j-1][:, :, t] * J[:, :, t-j]'
        end
    end
    return a_smooth, P_smooth, Plag
end

"""
    _simulation_smoother(rng, y, Z, Tt, RQR, Hobs; d=nothing, b=nothing, a0, P0, G=nothing)
        -> draw::Matrix{T}

Durbin–Koopman (2002) simulation smoother: one draw `α̃ ~ p(α | y)` from the
smoothing distribution, reusing the consolidated kernel (no new filter math).
Simulate an unconditional path `α⁺` (from the same `(a0, P0)` prior) and the
observations `y⁺` it implies, RTS-smooth both `y` and `y⁺` with identical linear
operations, and mean-correct: `α̃ = E[α|y] + (α⁺ − E[α⁺|y⁺])`. `y` is
`n_obs × T_obs` (columns = periods, `NaN` = missing); the draw is
`n_state × T_obs`. When `G !== nothing`, `(η_t, ν_t)` are drawn jointly from
`[RQR G; G' Hobs]` via `_psd_sqrt` and passed to the correlated filter and RTS smoother.
"""
function _simulation_smoother(rng::AbstractRNG, y::AbstractMatrix{T}, Z::AbstractMatrix{T},
                              Tt::AbstractMatrix{T}, RQR::AbstractMatrix{T},
                              Hobs::AbstractMatrix{T};
                              d=nothing, b=nothing,
                              a0::AbstractVector{T}, P0::AbstractMatrix{T},
                              G::Union{Nothing,AbstractMatrix}=nothing) where {T<:AbstractFloat}
    n_obs, T_obs = size(y)
    n_state = length(a0)
    Zm, Tm = Matrix{T}(Z), Matrix{T}(Tt)
    L0 = _psd_sqrt(Matrix{T}(P0))
    d_vec = d === nothing ? zeros(T, n_obs) : Vector{T}(d)
    b_vec = b === nothing ? zeros(T, n_state) : Vector{T}(b)

    use_corr = false
    Gm = zeros(T, n_state, n_obs)
    if G !== nothing
        Gm = Matrix{T}(G)
        size(Gm) == (n_state, n_obs) ||
            throw(ArgumentError("G must be n_state×n_obs = $((n_state, n_obs)), got $(size(Gm))"))
        use_corr = !iszero(Gm)
    end

    RQRm = Matrix{T}(RQR)
    Hm = Matrix{T}(Hobs)

    if !use_corr
        Lq = try
            Matrix{T}(cholesky(Symmetric(Matrix{T}((RQRm + RQRm') / 2))).L)
        catch
            _psd_sqrt(RQRm)
        end
        Lh = try
            Matrix{T}(cholesky(Symmetric(Matrix{T}((Hm + Hm') / 2))).L)
        catch
            _psd_sqrt(Hm)
        end
        # --- 1. unconditional forward simulation from the same prior (predict_first=true) ---
        a_plus = Matrix{T}(undef, n_state, T_obs)
        y_plus = Matrix{T}(undef, n_obs, T_obs)
        a = b_vec + Tm * (Vector{T}(a0) + L0 * randn(rng, T, n_state)) + Lq * randn(rng, T, n_state)
        for t in 1:T_obs
            @inbounds a_plus[:, t] = a
            @inbounds y_plus[:, t] = d_vec + Zm * a + Lh * randn(rng, T, n_obs)
            if t < T_obs
                a = b_vec + Tm * a + Lq * randn(rng, T, n_state)
            end
        end
        if any(isnan, y)
            y_plus[isnan.(y)] .= T(NaN)
        end
        # --- 2. smooth y and y⁺ with identical linear operations ---
        store = KalmanFilterStore{T}(n_state, T_obs)
        _kalman_filter!(store, Matrix{T}(y), Zm, Tm, RQRm, Hm;
                        d=d === nothing ? nothing : d_vec, b=b === nothing ? nothing : b_vec,
                        a0=Vector{T}(a0), P0=Matrix{T}(P0))
        a_smooth, _, _ = _rts_smoother(store, Tm)
        store_plus = KalmanFilterStore{T}(n_state, T_obs)
        _kalman_filter!(store_plus, y_plus, Zm, Tm, RQRm, Hm;
                        d=d === nothing ? nothing : d_vec, b=b === nothing ? nothing : b_vec,
                        a0=Vector{T}(a0), P0=Matrix{T}(P0))
        a_plus_smooth, _, _ = _rts_smoother(store_plus, Tm)
        # --- 3. mean correction: the draw inherits Cov(α|y) exactly in expectation ---
        return a_smooth + (a_plus - a_plus_smooth)
    end

    Jnt = [RQRm Gm; Gm' Hm]
    L_joint = _psd_sqrt(Matrix{T}((Jnt + Jnt') / 2))

    a_plus = Matrix{T}(undef, n_state, T_obs)
    y_plus = Matrix{T}(undef, n_obs, T_obs)
    w0 = L_joint * randn(rng, T, n_state + n_obs)
    a = b_vec + Tm * (Vector{T}(a0) + L0 * randn(rng, T, n_state)) + w0[1:n_state]
    for t in 1:T_obs
        w = L_joint * randn(rng, T, n_state + n_obs)
        @inbounds a_plus[:, t] = a
        @inbounds y_plus[:, t] = d_vec + Zm * a + view(w, (n_state+1):(n_state+n_obs))
        a = b_vec + Tm * a + view(w, 1:n_state)
    end
    if any(isnan, y)
        y_plus[isnan.(y)] .= T(NaN)
    end

    ym = Matrix{T}(y)
    d_arg = d === nothing ? nothing : d_vec
    b_arg = b === nothing ? nothing : b_vec
    a0_arg = Vector{T}(a0)
    P0_arg = Matrix{T}(P0)

    store = KalmanFilterStore{T}(n_state, T_obs; innovations=true)
    _kalman_filter!(store, ym, Zm, Tm, RQRm, Hm;
                    d=d_arg, b=b_arg, a0=a0_arg, P0=P0_arg, G=Gm)
    a_smooth, _, _ = _rts_smoother(store, Tm; G=Gm, y=ym)

    store_plus = KalmanFilterStore{T}(n_state, T_obs; innovations=true)
    _kalman_filter!(store_plus, y_plus, Zm, Tm, RQRm, Hm;
                    d=d_arg, b=b_arg, a0=a0_arg, P0=P0_arg, G=Gm)
    a_plus_smooth, _, _ = _rts_smoother(store_plus, Tm; G=Gm, y=y_plus)

    return a_smooth + (a_plus - a_plus_smooth)
end
