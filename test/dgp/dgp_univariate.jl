# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790) / DGP-09 (#798) / DGP-12 (#801): univariate, trend-cycle,
# state-space and hypothesis-test-pair simulators.

using Random, LinearAlgebra, Distributions

"""
    dgp_arima(rng; phi, theta, d, Phi, Theta, s, c, sigma, T, burn)
        -> NamedTuple

Seasonal ARIMA: ARMA(`phi`, `theta`) plus `d` differences and seasonal
AR/MA (`Phi`, `Theta` at lag `s`), `e_t ~ N(0, sigma²)`.
Returns `(y, phi, theta, d, Phi, Theta, s, c, sigma)`.
"""
function dgp_arima(rng::AbstractRNG; phi=Float64[], theta=Float64[], d::Int=0,
                   Phi=Float64[], Theta=Float64[], s::Int=0,
                   c::Float64=0.0, sigma::Float64=1.0,
                   T::Int=500, burn::Int=200)
    ph, th = Vector{Float64}(phi), Vector{Float64}(theta)
    PH, TH = Vector{Float64}(Phi), Vector{Float64}(Theta)
    p, q = length(ph), length(th)
    P, Q = length(PH), length(TH)
    maxlag = max(p, q, s * max(P, Q, 1))
    N = T + burn
    e = sigma .* randn(rng, N + maxlag)
    xb = zeros(maxlag)  # pre-sample buffer (burn-in discards its influence)
    x = zeros(N)
    for t in 1:N
        v = c + e[maxlag + t]
        for i in 1:p
            v += ph[i] * (t - i >= 1 ? x[t - i] : xb[maxlag + t - i])
        end
        for j in 1:q
            v += th[j] * e[maxlag + t - j]
        end
        for k in 1:P
            v += PH[k] * (t - k * s >= 1 ? x[t - k * s] : xb[maxlag + t - k * s])
        end
        for k in 1:Q
            v += TH[k] * e[maxlag + t - k * s]
        end
        x[t] = v
    end
    y = copy(x)
    for _ in 1:d
        y = cumsum(y)
    end
    return (y=y[(burn + 1):end], phi=ph, theta=th, d=d, Phi=PH, Theta=TH, s=s,
            c=c, sigma=sigma)
end

"""
    dgp_trend_cycle(rng; trend, drift, period, rho, sigma_trend, sigma_cycle,
                    sigma_noise, T) -> NamedTuple

Trend + cycle: random-walk-with-drift (`trend = :rw`) or linear (`:linear`)
trend plus an AR(2) cycle with complex roots (`period` = 16, damping `rho` =
0.9 by default: `φ₁ = 2ρcos(2π/period)`, `φ₂ = −ρ²`). Returns
`(y, trend, cycle)`.
"""
function dgp_trend_cycle(rng::AbstractRNG; trend::Symbol=:rw, drift::Float64=0.1,
                         period::Float64=16.0, rho::Float64=0.9,
                         sigma_trend::Float64=0.3, sigma_cycle::Float64=1.0,
                         sigma_noise::Float64=0.2, T::Int=400)
    phi1 = 2 * rho * cos(2 * pi / period)
    phi2 = -rho^2
    tr = zeros(T)
    if trend === :rw
        for t in 2:T
            tr[t] = tr[t - 1] + drift + sigma_trend * randn(rng)
        end
    elseif trend === :linear
        tr = @. drift * (1:T)
    else
        throw(ArgumentError("unknown trend :$trend (rw|linear)"))
    end
    cy = zeros(T)
    cy[1] = sigma_cycle * randn(rng)
    t2 = 2 <= T ? 2 : 1
    cy[t2] = phi1 * cy[1] + sigma_cycle * randn(rng)
    for t in 3:T
        cy[t] = phi1 * cy[t - 1] + phi2 * cy[t - 2] + sigma_cycle * randn(rng)
    end
    y = tr + cy + sigma_noise .* randn(rng, T)
    return (y=y, trend=tr, cycle=cy, phi=[phi1, phi2])
end

"""
    dgp_ar2_peak(rng; period, modulus, sigma, T) -> NamedTuple

AR(2) with spectral peak at `1/period` (`modulus` < 1 controls sharpness).
Returns `(y, phi, spectrum)` with the analytic spectrum
`S(ω) = σ²/2π / |1 − φ₁e^{−iω} − φ₂e^{−i2ω}|²` on a 256 grid.
"""
function dgp_ar2_peak(rng::AbstractRNG; period::Float64=8.0, modulus::Float64=0.9,
                      sigma::Float64=1.0, T::Int=1000, burn::Int=200)
    w0 = 2 * pi / period
    phi = [2 * modulus * cos(w0), -modulus^2]
    d = dgp_arima(rng; phi=phi, sigma=sigma, T=T, burn=burn)
    grid = range(0, pi; length=256)
    spec = @. sigma^2 / (2 * pi) /
        abs2(1 - phi[1] * exp(-im * grid) - phi[2] * exp(-im * 2 * grid))
    return (y=d.y, phi=phi, freqs=collect(grid), spectrum=collect(spec))
end

"""
    dgp_lagged_pair(rng; d, gain, T, burn) -> NamedTuple

`y_t = gain * x_{t-d} + noise` with AR(1) `x`: known phase `−d·ω` and gain.
Returns `(x, y, d, gain)`.
"""
function dgp_lagged_pair(rng::AbstractRNG; d::Int=3, gain::Float64=2.0,
                         T::Int=1000, burn::Int=200)
    N = T + burn
    x = zeros(N)
    for t in 2:N
        x[t] = 0.6 * x[t - 1] + randn(rng)
    end
    y = zeros(N)
    for t in (d + 1):N
        y[t] = gain * x[t - d] + 0.5 * randn(rng)
    end
    keep = (burn + 1):N
    return (x=x[keep], y=y[keep], d=d, gain=gain)
end

"""
    dgp_state_space(rng; F, H, Q, R, x0, T) -> NamedTuple

Linear Gaussian state space `x_{t+1} = F x_t + w_t`, `y_t = H x_t + v_t`,
`w ~ N(0, Q)`, `v ~ N(0, R)`. Multivariate-capable. Returns
`(y, x)` with the true state path.
"""
function dgp_state_space(rng::AbstractRNG; F=[0.9;;], H=[1.0;;],
                         Q=[0.5;;], R=[0.5;;], x0=nothing, T::Int=300)
    Fm, Hm = Matrix{Float64}(F), Matrix{Float64}(H)
    Qm, Rm = Matrix{Float64}(Q), Matrix{Float64}(R)
    k, n = size(Fm, 1), size(Hm, 1)
    Lq = cholesky(Symmetric(Qm)).L
    Lr = cholesky(Symmetric(Rm)).L
    x = x0 === nothing ? zeros(k) : Vector{Float64}(x0)
    X = zeros(T, k)
    Y = zeros(T, n)
    for t in 1:T
        x = Fm * x + Lq * randn(rng, k)
        X[t, :] .= x
        Y[t, :] .= Hm * x + Lr * randn(rng, n)
    end
    return (y=Y, x=X, F=Fm, H=Hm, Q=Qm, R=Rm)
end

"""
    dgp_unit_root_pair(rng; kind, T, phi, break_at, deterministic, rho, N, csd)
        -> NamedTuple

H0/H1 pair `(h0, h1, truth)` per hypothesis-test family (DGP-12):
kinds `:adf` (RW vs AR), `:kpss` (reversed), `:trend` (drift-RW vs
trend-stationary), `:break_level`/`:break_trend` (unit root with break vs
stationary around broken mean/trend), `:seasonal`, `:fourier` (smooth break
`2sin(2πt/200)`), `:explosive` (RW vs bubble window), `:cointegrated_pair`
(with `β`, optional regime shift), `:granger` (`y₂ ← y₁` vs independent),
`:panel_ur` (ρ = 0.9 alternative, common-factor `csd` option),
`:nongaussian` (Gaussian vs t₃), `:heteroskedastic_groups`.
`deterministic ∈ :none|:constant|:trend` is matched to the DGP.
"""
function dgp_unit_root_pair(rng::AbstractRNG; kind::Symbol=:adf, T::Int=200,
                            phi::Float64=0.5, break_at::Float64=0.5,
                            deterministic::Symbol=:constant, rho::Float64=0.9,
                            N::Int=20, csd::Bool=false)
    br = round(Int, break_at * T)
    det = t -> deterministic === :trend ? 0.1 * t :
               deterministic === :constant ? 1.0 : 0.0
    rw(n) = cumsum(randn(rng, n))
    ar1(n, p) = (x = zeros(n); for t in 2:n
        x[t] = p * x[t - 1] + randn(rng); end; x)
    truth = (; kind=kind, break_at=br)
    if kind === :adf
        h0 = rw(T) .+ det.(1:T)
        h1 = ar1(T, phi) .+ det.(1:T)
    elseif kind === :kpss
        h1 = rw(T) .+ det.(1:T)
        h0 = ar1(T, phi) .+ det.(1:T)
    elseif kind === :trend
        h0 = cumsum(ones(T) .+ randn(rng, T))
        h1 = det.(1:T) .+ ar1(T, phi)
    elseif kind === :break_level
        h0 = rw(T); h0[(br + 1):end] .+= 3.0
        h1 = ar1(T, phi); h1[(br + 1):end] .+= 3.0
    elseif kind === :break_trend
        h0 = rw(T) + 0.05 * (1:T)
        h1 = 0.05 * (1:T) + ar1(T, phi)
        step = collect((br + 1):T) .- br  # (: binds tighter than .-; parenthesise)
        h1[(br + 1):end] .+= 0.1 .* step
    elseif kind === :seasonal
        seas = @. 2 * sin(2 * pi * (1:T) / 4)
        h0 = cumsum(randn(rng, T)) + seas
        h1 = seas + 0.5 * randn(rng, T)
    elseif kind === :fourier
        sm = @. 2 * sin(2 * pi * (1:T) / 200)
        h0 = rw(T)
        h1 = sm + ar1(T, phi)
    elseif kind === :explosive
        h0 = rw(T)
        h1 = rw(T); seg = br:min(T, br + round(Int, 0.2T))
        h1[seg] = h1[br] .+ cumsum(1.02 .^ (1:length(seg)) .* randn(rng, length(seg)))
    elseif kind === :cointegrated_pair
        x = rw(T)
        h0 = (x, rw(T))
        h1 = (x, x + 0.5 * randn(rng, T))
        truth = (; kind=kind, break_at=br, beta=[1.0, -1.0])
    elseif kind === :granger
        y1 = ar1(T, 0.6)
        h0 = (y1, ar1(T, 0.6))
        y2 = zeros(T)
        for t in 2:T
            y2[t] = 0.5 * y2[t - 1] + 0.7 * y1[t - 1] + randn(rng)
        end
        h1 = (y1, y2)
    elseif kind === :panel_ur
        F = csd ? randn(rng, T) : zeros(T)
        h0 = cumsum(randn(rng, T, N), dims=1) .+ F
        h1 = hcat([ar1(T, rho) for _ in 1:N]...) .+ 0.2 .* F
    elseif kind === :nongaussian
        h0 = randn(rng, T)
        h1 = rand(rng, TDist(3), T) ./ sqrt(3)
    elseif kind === :heteroskedastic_groups
        h0 = (randn(rng, T), randn(rng, T))
        h1 = (randn(rng, T), 2 .* randn(rng, T))
    else
        throw(ArgumentError("unknown kind :$kind"))
    end
    return (h0=h0, h1=h1, truth=truth)
end
