# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Shared infrastructure for time-varying-volatility SVAR identification.

Supports the nonparametric Lewis (2021) TVV-ID estimator (`identify_lewis_tvv`)
and the parametric Bertsche–Braun (2022) SV-SVAR estimator (`identify_sv_svar`):
SV/TVV shock simulation, permutation/sign-invariant Q comparison for tests,
and an adjustable-tolerance orthogonality check.

Deliberately reuses existing machinery instead of duplicating it: GMM weighting
comes from the `estimate_gmm` kernel (`src/gmm/gmm.jl`), Q-distance delegates
to `_procrustes_distance` (`src/nongaussian/tests.jl`), and permutation
enumeration reuses `_permutations` (same file). `src/sv/` holds only the
KSC-Gibbs SV estimator/forecaster — no unconditional SV path simulator — so
lightweight AR(1) log-vol simulation lives here.

References:
- Lewis, D. J. (2021). "Identifying Shocks via Time-Varying Volatility."
  Review of Economic Studies 88(6), 3086–3124.
- Bertsche, D. & Braun, R. (2022). "Identification of Structural Vector
  Autoregressions by Stochastic Volatility." Journal of Business & Economic
  Statistics 40(1), 328–341.
"""

using LinearAlgebra, Statistics, Random

# =============================================================================
# SV / TVV shock simulation
# =============================================================================

"""
    simulate_sv_shocks(rng, Tobs, n; mus, rhos, sigmas) -> (E, H)

Simulate `Tobs × n` structural shocks `E` with independent AR(1) log-volatility
`H`: `h_it = μ_i + ρ_i (h_{i,t-1} − μ_i) + σ_i η_it`, `e_it = exp(h_it/2) z_it`
with `η_it, z_it ~ N(0, 1)`. `H[1, :]` is drawn from the stationary distribution.
Defaults give distinct, estimation-grade persistence (`ρ` from 0.97 down to
0.85, `σ` from 0.25 down to 0.20) so each shock's volatility dynamics are
strong and linearly independent. `σ_i = 0` is allowed and yields a
homoskedastic shock (constant `H[:, i]`), useful for partial-ID fixtures.
"""
function simulate_sv_shocks(rng::AbstractRNG, Tobs::Int, n::Int;
        mus::AbstractVector{<:Real}=zeros(n),
        rhos::AbstractVector{<:Real}=collect(range(0.97, stop=0.85, length=n)),
        sigmas::AbstractVector{<:Real}=collect(range(0.25, stop=0.20, length=n)))
    Tobs >= 1 || throw(ArgumentError("Tobs must be ≥ 1, got $Tobs"))
    n >= 1 || throw(ArgumentError("n must be ≥ 1, got $n"))
    length(mus) == n && length(rhos) == n && length(sigmas) == n ||
        throw(ArgumentError("mus/rhos/sigmas must have length n=$n"))
    all(r -> abs(r) < 1, rhos) ||
        throw(ArgumentError("simulate_sv_shocks requires |rhos| < 1 (stationary AR(1) log-vol)"))
    all(s -> s >= 0, sigmas) ||
        throw(ArgumentError("simulate_sv_shocks requires sigmas ≥ 0"))
    mu = Vector{Float64}(mus)
    rho = Vector{Float64}(rhos)
    sig = Vector{Float64}(sigmas)
    H = Matrix{Float64}(undef, Tobs, n)
    stat_sd = sig ./ sqrt.(1 .- rho .^ 2)
    H[1, :] = mu .+ stat_sd .* randn(rng, n)
    for t in 2:Tobs
        H[t, :] = mu .+ rho .* (@view(H[t - 1, :]) .- mu) .+ sig .* randn(rng, n)
    end
    E = exp.(H ./ 2) .* randn(rng, Tobs, n)
    return E, H
end

"""Two-state Markov variance shocks: per-shock states `{1, s_hi}` with symmetric
stay probability `stay[i]`. Distinct `stay` gives independent volatility
dynamics across shocks without any parametric SV law."""
function _simulate_markov_vol_shocks(rng::AbstractRNG, Tobs::Int, n::Int;
        s_hi::Real=3.0, stay::AbstractVector{<:Real}=collect(range(0.98, stop=0.90, length=n)))
    s_hi > 0 || throw(ArgumentError("s_hi must be > 0, got $s_hi"))
    length(stay) == n || throw(ArgumentError("stay must have length n=$n"))
    all(s -> 0 < s < 1, stay) ||
        throw(ArgumentError("stay probabilities must lie in (0, 1)"))
    s = Float64(s_hi)
    pstay = Vector{Float64}(stay)
    S = Matrix{Float64}(undef, Tobs, n)
    state = rand(rng, Bool, n)  # true = high-variance state
    for t in 1:Tobs
        S[t, :] = ifelse.(state, s, 1.0)
        state = xor.(state, rand(rng, n) .> pstay)
    end
    E = sqrt.(S) .* randn(rng, Tobs, n)
    return E, S
end

"""
    simulate_tvv_dgp(rng, n, p, Tobs; kind=:sv, A=nothing, B0=nothing, burn=50,
                     mus=nothing, rhos=nothing, sigmas=nothing, s_hi=3.0,
                     stay=nothing) -> (Y, A_true, B0_true, U_true)

Simulate a VAR(p) with time-varying-volatility structural shocks.

- `kind=:sv` — Gaussian shocks with AR(1) log-volatility
  (`simulate_sv_shocks`; `mus`/`rhos`/`sigmas` forwarded when given).
- `kind=:markov` — Gaussian shocks with two-state Markov variances
  (`s_hi` high-state variance, `stay` per-shock stay probabilities).

Shocks are standardized to unit variance (volatility-dependence shape is
preserved), so `B0_true` is the impact matrix for unit-variance shocks.
Defaults: `A_l = (0.5/l) I` (stable), `B0 = I + 0.3 (11' − I)`. The VAR
recursion mirrors `simulate_svar` (test/var/id_dgps.jl) with `burn` discarded.
"""
function simulate_tvv_dgp(rng::AbstractRNG, n::Int, p::Int, Tobs::Int;
        kind::Symbol=:sv, A=nothing, B0=nothing, burn::Int=50,
        mus=nothing, rhos=nothing, sigmas=nothing,
        s_hi::Real=3.0, stay=nothing)
    kind === :sv || kind === :markov ||
        throw(ArgumentError("kind must be :sv or :markov, got :$kind"))
    n >= 1 || throw(ArgumentError("n must be ≥ 1, got $n"))
    p >= 1 || throw(ArgumentError("p must be ≥ 1, got $p"))
    Tobs >= 1 || throw(ArgumentError("Tobs must be ≥ 1, got $Tobs"))
    burn >= 0 || throw(ArgumentError("burn must be ≥ 0, got $burn"))
    A_true = if A === nothing
        [Matrix{Float64}((0.5 / l) * I, n, n) for l in 1:p]
    else
        length(A) == p || throw(ArgumentError("A must hold p=$p matrices"))
        [Matrix{Float64}(a) for a in A]
    end
    for a in A_true
        size(a) == (n, n) || throw(ArgumentError("each A[lag] must be $n×$n"))
    end
    B0_true = B0 === nothing ?
        Matrix{Float64}(I, n, n) + 0.3 * (ones(n, n) - I) :
        Matrix{Float64}(B0)
    size(B0_true) == (n, n) || throw(ArgumentError("B0 must be $n×$n"))
    ntot = Tobs + p + burn
    E_full = if kind === :sv
        E, _ = simulate_sv_shocks(rng, ntot, n;
            mus=mus === nothing ? zeros(n) : mus,
            rhos=rhos === nothing ? collect(range(0.97, stop=0.85, length=n)) : rhos,
            sigmas=sigmas === nothing ? collect(range(0.25, stop=0.20, length=n)) : sigmas)
        E
    else
        E, _ = _simulate_markov_vol_shocks(rng, ntot, n; s_hi=s_hi,
            stay=stay === nothing ? collect(range(0.98, stop=0.90, length=n)) : stay)
        E
    end
    E_full ./= sqrt.(mean(E_full .^ 2; dims=1))  # unit-variance shocks
    U_full = E_full * B0_true'
    Y_full = zeros(ntot, n)
    for t in (p + 1):ntot
        yt = U_full[t, :]
        for lag in 1:p
            yt = yt + A_true[lag] * Y_full[t - lag, :]
        end
        Y_full[t, :] = yt
    end
    sel = (ntot - Tobs + 1):ntot
    return Y_full[sel, :], A_true, B0_true, U_full[sel, :]
end

# =============================================================================
# Permutation/sign-invariant Q comparison (test support)
# =============================================================================

"""
    align_Q(Qhat, Q0) -> (Q_aligned, perm, signs)

Align columns of square `Q0` to `Qhat` by the signed permutation minimizing
`‖Qhat − Q0[:, perm] .* signs'‖_F`. Returns the aligned matrix, the column
permutation, and the sign vector. Exhaustive for `n ≤ 5` (reuses
`_permutations`); greedy correlation matching above (mirrors
`_procrustes_distance`).
"""
function align_Q(Qhat::AbstractMatrix, Q0::AbstractMatrix)
    size(Qhat) == size(Q0) ||
        throw(DimensionMismatch("align_Q needs equally sized Qhat and Q0"))
    n = size(Qhat, 1)
    n >= 1 && size(Qhat, 2) == n ||
        throw(DimensionMismatch("align_Q needs nonempty square matrices"))
    T = float(promote_type(eltype(Qhat), eltype(Q0)))
    A = Matrix{T}(Qhat)
    B = Matrix{T}(Q0)
    if n <= 5
        best_d = T(Inf)
        best_perm = collect(1:n)
        best_s = ones(T, n)
        for perm in _permutations(n)
            Bp = B[:, perm]
            for signs in Iterators.product(fill((one(T), -one(T)), n)...)
                s = collect(signs)
                d = norm(A - Bp .* s')
                if d < best_d
                    best_d = d
                    best_perm = perm
                    best_s = s
                end
            end
        end
        return B[:, best_perm] .* best_s', best_perm, best_s
    else
        # Greedy matching by column correlation (mirrors _procrustes_distance)
        perm = zeros(Int, n)
        s = ones(T, n)
        used = falses(n)
        for j in 1:n
            best_k, best_c = 0, T(-Inf)
            for k in 1:n
                used[k] && continue
                c = abs(dot(view(A, :, j), view(B, :, k)))
                if c > best_c
                    best_k, best_c = k, c
                end
            end
            perm[j] = best_k
            used[best_k] = true
            sj = sign(dot(view(A, :, j), view(B, :, best_k)))
            s[j] = sj == 0 ? one(T) : sj
        end
        return B[:, perm] .* s', perm, s
    end
end

"""
    q_distance(Qhat, Q0) -> Real

Permutation/sign-invariant Frobenius distance between orthogonal matrices:
delegates to `_procrustes_distance` (same signed-permutation minimum). Use
`align_Q` when the aligned matrix or the `(perm, signs)` map is needed.
"""
function q_distance(Qhat::AbstractMatrix, Q0::AbstractMatrix)
    size(Qhat) == size(Q0) ||
        throw(DimensionMismatch("q_distance needs equally sized Qhat and Q0"))
    T = float(promote_type(eltype(Qhat), eltype(Q0)))
    _procrustes_distance(Matrix{T}(Qhat), Matrix{T}(Q0))
end

"""
    check_orthogonal(Q; atol=1e-8) -> Bool

True when `Q` is square and `‖Q'Q − I‖ ≤ atol` (same Frobenius expression as
`_q_is_orthogonal`, with adjustable tolerance).
"""
function check_orthogonal(Q::AbstractMatrix; atol::Real=1e-8)
    size(Q, 1) == size(Q, 2) || return false
    atol >= 0 || throw(ArgumentError("atol must be ≥ 0, got $atol"))
    return norm(Q' * Q - I(size(Q, 1))) <= atol
end
