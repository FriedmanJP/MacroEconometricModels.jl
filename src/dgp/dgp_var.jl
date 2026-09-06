# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790): shared truth-returning VAR simulators.
#
# Contract for every public simulator (see docs/src/simulation.md): rng::AbstractRNG first positional
# argument (never touches the global RNG), burn-in for dynamic processes
# (default burn = 200), and a NamedTuple return with everything a recovery
# assertion needs (coefficients, covariances, shocks, latent paths).


"""
    dgp_var(rng; A, B0, Sigma, c, T, burn) -> NamedTuple

Stationary VAR(p): `Y_t = c + A_1 Y_{t-1} + … + A_p Y_{t-p} + u_t`,
`u_t = B0 * ε_t`, `ε_t ~ N(0, I)`.

- `A`: coefficient matrix (VAR(1)) or vector of `p` matrices. Default is the
  DGP-02 reference design with non-diagonal dynamics and a non-identity impact
  matrix, so shock orderings and rotations matter.
- `B0` / `Sigma`: impact matrix (default lower-triangular) or error
  covariance (`Sigma = B0 * B0'`). Pass exactly one: `Sigma` alone takes the
  lower Cholesky factor; passing both throws.
- Returns `(Y, eps, A, Sigma, B0, c)`: the `T×n` sample (post-burn-in), the
  `T×n` structural shocks, the coefficient list, `Sigma`, `B0`, intercept.
"""
const _DGP_VAR_B0 = [1.0 0.0 0.0; 0.5 1.0 0.0; 0.3 0.2 1.0]

function dgp_var(rng::AbstractRNG;
                 A=[0.5 0.1 0.0; 0.2 0.4 0.1; 0.0 0.1 0.3],
                 B0=nothing,
                 Sigma=nothing, c=nothing, T::Int=500, burn::Int=200)
    As = A isa AbstractMatrix ? [Matrix{Float64}(A)] : [Matrix{Float64}(a) for a in A]
    n, p = size(As[1], 1), length(As)
    B0 !== nothing && Sigma !== nothing &&
        throw(ArgumentError("dgp_var: pass exactly one of B0 and Sigma"))
    L = B0 !== nothing ? Matrix{Float64}(B0) :
        Sigma !== nothing ? Matrix{Float64}(cholesky(Symmetric(Matrix{Float64}(Sigma))).L) :
        Matrix{Float64}(_DGP_VAR_B0)
    S = L * L'
    cc = c === nothing ? zeros(n) : Vector{Float64}(c)
    Eps = randn(rng, T + burn, n)
    Y = zeros(T + burn, n)
    hist = [zeros(n) for _ in 1:p]
    for t in 1:(T + burn)
        y = copy(cc) + L * Eps[t, :]
        for i in 1:p
            y += As[i] * hist[i]
        end
        for i in p:-1:2
            hist[i] .= hist[i - 1]
        end
        hist[1] .= y
        Y[t, :] .= y
    end
    keep = (burn + 1):(T + burn)
    return (Y=Y[keep, :], eps=Eps[keep, :], A=As, Sigma=S, B0=L, c=cc)
end

"""
    lyapunov_gamma0(A, Sigma) -> Matrix

Stationary covariance `Γ₀` of a VAR(1) with coefficient `A` and innovation
covariance `Sigma`: `vec(Γ₀) = (I − A⊗A)⁻¹ vec(Sigma)`.
"""
function lyapunov_gamma0(A::AbstractMatrix, Sigma::AbstractMatrix)
    k = size(A, 1)
    G = (Matrix{Float64}(I, k * k, k * k) - kron(Matrix{Float64}(A), Matrix{Float64}(A))) \
        vec(Matrix{Float64}(Sigma))
    return reshape(G, k, k)
end

# Companion-form VAR(1) coefficient of a VAR(p): stacked [A_1 … A_p; I 0].
function _companion(As::AbstractVector)
    p = length(As)
    p == 1 && return As[1]
    n = size(As[1], 1)
    C = zeros(n * p, n * p)
    C[1:n, :] = hcat(As...)
    C[(n + 1):end, 1:(n * (p - 1))] = Matrix{Float64}(I, n * (p - 1), n * (p - 1))
    return C
end

"""
    var_irf(A, B0, H) -> Array{Float64,3}

Structural moving-average matrices `Θ_h = Φ_h * B0`, `h = 0…H`, where
`Φ_0 = I`, `Φ_h = Σ_{i=1}^{min(h,p)} A_i Φ_{h-i}`. Returns `(H+1)×n×n`
with `out[h+1, :, :] = Θ_h`.
"""
function var_irf(A, B0::AbstractMatrix, H::Int)
    As = A isa AbstractMatrix ? [Matrix{Float64}(A)] : [Matrix{Float64}(a) for a in A]
    n, p = size(As[1], 1), length(As)
    B = Matrix{Float64}(B0)
    Phi = [zeros(n, n) for _ in 0:H]
    Phi[1] = Matrix{Float64}(I, n, n)
    for h in 1:H
        M = zeros(n, n)
        for i in 1:min(h, p)
            M += As[i] * Phi[h - i + 1]
        end
        Phi[h + 1] = M
    end
    out = zeros(H + 1, n, n)
    for h in 0:H
        out[h + 1, :, :] = Phi[h + 1] * B
    end
    return out
end

"""
    var_fevd(A, B0, H) -> Array{Float64,3}

Forecast-error variance decomposition shares `(H+1)×n×n`:
`out[h+1, i, j] = Σ_{l=0}^{h} Θ_l[i,j]² / Σ_k Σ_{l=0}^{h} Θ_l[i,k]²`.
Rows (`i`) sum to 1 at every horizon by construction.
"""
function var_fevd(A, B0::AbstractMatrix, H::Int)
    irf = var_irf(A, B0, H)
    _, n, _ = size(irf)
    out = zeros(H + 1, n, n)
    acc = zeros(n, n)
    for h in 0:H
        acc += irf[h + 1, :, :] .^ 2
        tot = sum(acc, dims=2)
        out[h + 1, :, :] = acc ./ tot
    end
    return out
end

"""
    var_hd(A, B0, eps; c=zeros(n)) -> Array{Float64,3}

Historical shock contributions `T×n×n`: `out[t, i, j]` is shock `j`'s
contribution to variable `i` at time `t` (MA filter of the true shocks from
a zero initial deviation). Summing over `j` reproduces `Y_t − μ` exactly
when the sample starts at the stationary mean (e.g. `dgp_var` with
`burn = 0` and `c = 0`, whose zero pre-sample history is `μ = 0`); with
burn-in the early observations additionally carry the initial-condition
term `Φ_t (Y_0 − μ)`.
"""
function var_hd(A, B0::AbstractMatrix, eps::AbstractMatrix; c=nothing)
    As = A isa AbstractMatrix ? [Matrix{Float64}(A)] : [Matrix{Float64}(a) for a in A]
    n, p = size(As[1], 1), length(As)
    T = size(eps, 1)
    B = Matrix{Float64}(B0)
    E = Matrix{Float64}(eps)
    Phi = [zeros(n, n) for _ in 0:(T - 1)]
    Phi[1] = Matrix{Float64}(I, n, n)
    for h in 1:(T - 1)
        M = zeros(n, n)
        for i in 1:min(h, p)
            M += As[i] * Phi[h - i + 1]
        end
        Phi[h + 1] = M
    end
    out = zeros(T, n, n)
    for t in 1:T
        for j in 1:n
            acc = zeros(n)
            for l in 0:(t - 1)
                acc += (Phi[l + 1] * B)[:, j] * E[t - l, j]
            end
            out[t, :, j] = acc
        end
    end
    return out
end
