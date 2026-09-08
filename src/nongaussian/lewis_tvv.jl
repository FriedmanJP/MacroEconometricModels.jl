# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Identification of SVAR shocks via time-varying volatility (Lewis 2021).

Parameterises ``B₀ = L Q(θ)`` with Givens angles (as in `identify_gmm_moments`)
and estimates ``θ`` from the autocovariance structure of squared shocks —
``E[(e²_it − 1)(e²_{j,t−k} − 1)] = 0`` for ``i ≠ j`` — using the package GMM
kernel. No variance law is assumed: the squared candidate shocks serve as their
own volatility proxy. Identification requires persistent, shock-specific
volatility dynamics; a portmanteau diagnostic on squared whitened residuals
flags weak identification instead of returning a silent wrong ``B₀``.

References:
- Lewis, D. J. (2021). "Identifying Shocks via Time-Varying Volatility."
  Review of Economic Studies 88(6), 3086–3124.
"""

using LinearAlgebra, Statistics, Random

# =============================================================================
# Result Type
# =============================================================================

"""
    LewisTVVResult{T} <: AbstractNonGaussianSVAR

Result from Lewis (2021) TVV-ID estimation.

Fields:
- `B0::Matrix{T}` — structural impact matrix (n × n)
- `Q::Matrix{T}` — rotation matrix (`B₀ = L Q`)
- `theta::Vector{T}` — Givens angles
- `vcov::Matrix{T}` — sandwich covariance of `theta`
- `se::Matrix{T}` — delta-method standard errors for `B₀`
- `J::T` — Hansen J statistic
- `J_pvalue::T` — J-test p-value
- `K::Int` — default lag count (informational when `lags` is given explicitly)
- `lags::Vector{Int}` — lags in the TVV moment block
- `weighting::Symbol` — `:one_step`, `:two_step`, or `:cue`
- `converged::Bool` — GMM optimizer convergence (not identification strength)
- `iters::Int` — total GMM iterations at the best start
- `weak_id::Bool` — true when the TVV strength diagnostic finds no serial
  dependence in squared whitened residuals (estimate untrustworthy)
- `id_strength::T` — portmanteau statistic relative to its 99.9% null value
  (> 1 means identified; see `_tvv_strength`)
- `message::String`
- `shocks::Matrix{T}` — structural shocks (T_eff × n)
- `varnames::Vector{String}`
- `shock_names::Vector{String}`
"""
struct LewisTVVResult{T<:AbstractFloat} <: AbstractNonGaussianSVAR
    B0::Matrix{T}
    Q::Matrix{T}
    theta::Vector{T}
    vcov::Matrix{T}
    se::Matrix{T}
    J::T
    J_pvalue::T
    K::Int
    lags::Vector{Int}
    weighting::Symbol
    converged::Bool
    iters::Int
    weak_id::Bool
    id_strength::T
    message::String
    shocks::Matrix{T}
    varnames::Vector{String}
    shock_names::Vector{String}
end

function Base.show(io::IO, r::LewisTVVResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Variables"    n;
        "TVV lags"     join(r.lags, ",");
        "Weighting"    string(r.weighting);
        "J-statistic"  _fmt(r.J);
        "J p-value"    _format_pvalue(r.J_pvalue);
        "ID strength"  _fmt(r.id_strength);
        "Weak ID"      string(r.weak_id);
        "Givens angles" length(r.theta)
    ]
    _pretty_table(io, spec;
        title = "Lewis TVV-ID Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _show_B0_ses(io, r.B0, r.se)
end

# =============================================================================
# TVV moment conditions (Lewis 2021)
# =============================================================================

"""Number of TVV moment conditions: ordered `i ≠ j` pairs × lags."""
_lewis_n_moments(n::Int, lags) = n * (n - 1) * length(lags)

"""
Per-observation TVV moments.

For candidate shocks `E = Z Q(θ)` with centered squares `S = E² − 1`, row `t`
holds `(S[t, i] · S[t−k, j])` for every ordered `i ≠ j` and every lag `k`.
Rows `1:Lmax` are dropped so every entry is well-defined. `k = 0` is allowed
(contemporaneous independence moment); otherwise `k ≥ 1` autocovariance moments.
"""
function _lewis_tvv_moments(theta::AbstractVector{T}, data) where {T<:AbstractFloat}
    Z = data.Z
    n = data.n
    lags = data.lags
    Q = _givens_to_orthogonal(theta, n)
    E = Z * Q
    S = E .^ 2 .- one(T)
    Lmax = maximum(lags)
    Tobs = size(E, 1)
    rows = (Lmax + 1):Tobs
    M = Matrix{T}(undef, length(rows), _lewis_n_moments(n, lags))
    col = 1
    for k in lags, i in 1:n, j in 1:n
        i == j && continue
        @inbounds for (r, t) in enumerate(rows)
            M[r, col] = S[t, i] * S[t - k, j]
        end
        col += 1
    end
    M
end

"""
Portmanteau TVV strength diagnostic on whitened residuals `Z`.

Statistic: `T_eff · Σ_{i,k≥1} ĉ_{i,k}²` over own-autocorrelations `ĉ` of
centered squared residuals. Under homoskedastic iid shocks this is ≈ χ² with
`n·(#lags≥1)` degrees of freedom; persistent shock-specific volatility pushes
it far above. Returns `(strength, weak)` where `strength` is the statistic
relative to the 99.9% null value (Wilson–Hilferty) and `weak = strength < 1`.
With no `k ≥ 1` lags the diagnostic is vacuous: returns `(1, false)` and the
caller records that it was skipped.
"""
function _tvv_strength(Z::AbstractMatrix{T}, lags) where {T<:AbstractFloat}
    klags = filter(k -> k >= 1, lags)
    isempty(klags) && return one(T), false
    S = Z .^ 2 .- one(T)
    Teff = size(S, 1)
    n = size(S, 2)
    denom = vec(mean(S .^ 2; dims=1))
    acc = zero(T)
    for i in 1:n
        denom[i] > 0 || continue
        for k in klags
            k >= Teff && continue
            c = mean(@view(S[(k + 1):Teff, i]) .* @view(S[1:(Teff - k), i])) / denom[i]
            acc += c * c
        end
    end
    stat = Teff * acc
    df = n * length(klags)
    # Wilson–Hilferty χ²_{df, 0.999}: df·(1 − 2/9df + 3.09·√(2/9df))³
    w = 1 - 2 / (9 * df) + 3.09 * sqrt(2 / (9 * df))
    thresh = df * w^3
    strength = stat / thresh
    return strength, strength < 1
end

# =============================================================================
# Estimation
# =============================================================================

"""
    identify_lewis_tvv(model::VARModel; K=5, lags=1:K, weighting=:two_step,
                       hac=true, bandwidth=0, n_starts=10, max_iter=100,
                       tol=1e-8, rng=Random.default_rng()) -> LewisTVVResult

`J_pvalue` is `NaN` under `weighting=:one_step`: identity weighting is
inefficient, so Hansen's χ² limit fails and the GMM kernel reports no
p-value (same convention as `estimate_gmm`).

Identify `B₀ = L Q(θ)` from time-varying volatility (Lewis 2021) without
assuming any variance law.

The GMM moment block is the cross-autocovariance of squared shocks:
`E[(e²_it − 1)(e²_{j,t−k} − 1)] = 0` for all ordered `i ≠ j` and `k ∈ lags`,
estimated through the `estimate_gmm` kernel (`:one_step` → identity weighting,
`:two_step` → Hansen two-step, `:cue` → iterated/CUE). Q-space is non-convex,
so estimation multi-starts (`n_starts`, first start at `θ = 0`) and keeps the
lowest J-statistic. `lags` overrides `K` when given explicitly.

`weak_id` is true when squared whitened residuals show no serial dependence at
the requested lags — the estimate is then untrustworthy (proportional or
homoskedastic variances carry no identifying information). This flag, not
`converged` (which reports optimizer convergence only), is the identification
check. Always inspect it.

# Examples
```julia
rng = Xoshiro(11)
B0 = [1.0 0.4; -0.2 1.0]
E = vcat(0.5 .* randn(rng, 750, 2), 2.0 .* randn(rng, 750, 2))
Y = Matrix((B0 * E')')
res = identify_lewis_tvv(estimate_var(Y, 1); rng=Xoshiro(12))
res.B0  # ≈ B0 up to signed permutation; check res.weak_id == false
Q = MacroEconometricModels.compute_Q(estimate_var(Y, 1), :lewis_tvv;
                                     rng=Xoshiro(12))
```
"""
function identify_lewis_tvv(model::VARModel{T};
        K::Int=5,
        lags::AbstractVector{<:Integer}=collect(1:K),
        weighting::Symbol=:two_step,
        hac::Bool=true,
        bandwidth::Int=0,
        n_starts::Int=10,
        max_iter::Int=100,
        tol::T=T(1e-8),
        rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    weighting in (:one_step, :two_step, :cue) || throw(ArgumentError(
        "weighting must be :one_step, :two_step, or :cue, got :$weighting"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be ≥ 1"))
    isempty(lags) && throw(ArgumentError("lags must be nonempty"))
    all(l -> l >= 0, lags) || throw(ArgumentError("lags must be ≥ 0"))
    n = nvars(model)
    n >= 2 || throw(ArgumentError("Lewis TVV-ID requires n ≥ 2 variables"))
    n_angles = n * (n - 1) ÷ 2
    q = _lewis_n_moments(n, lags)
    q >= n_angles || throw(ArgumentError(
        "lags supply $q conditions for $n_angles Givens angles"))

    L = Matrix{T}(safe_cholesky(model.Sigma))
    Z = Matrix{T}(model.U / L')
    Teff = size(Z, 1)
    Lmax = maximum(lags)
    Teff - Lmax >= 100 || throw(ArgumentError(
        "need ≥ 100 usable observations after max lag (got T_eff=$Teff, max lag=$Lmax)"))
    lagvec = Vector{Int}(lags)
    data = (Z=Z, n=n, lags=lagvec)

    strength, weak = _tvv_strength(Z, lagvec)
    w_kernel = weighting === :cue ? :iterated :
        weighting === :one_step ? :identity : :two_step

    starts = Vector{Vector{T}}(undef, n_starts)
    starts[1] = zeros(T, n_angles)
    for s in 2:n_starts
        starts[s] = T(π) .* (rand(rng, T, n_angles) .- T(0.5))
    end

    best = nothing
    for θ0 in starts
        gmm = estimate_gmm(_lewis_tvv_moments, θ0, data;
            weighting=w_kernel, hac=hac, bandwidth=bandwidth,
            max_iter=max_iter, tol=tol)
        if best === nothing || gmm.J_stat < best.J_stat
            best = gmm
        end
    end

    theta = best.theta
    Q = _givens_to_orthogonal(theta, n)
    B0 = L * Q
    shocks = Z * Q
    for j in 1:n
        if B0[j, j] < 0
            B0[:, j] .*= -one(T)
            Q[:, j] .*= -one(T)
            shocks[:, j] .*= -one(T)
        end
    end

    se_B0, _ = _delta_B0_se(theta, best.vcov, ϑ -> L * _givens_to_orthogonal(ϑ, n), n)

    message = if weak
        "weak identification: TVV strength $(round(strength; digits=3)) < 1 " *
        "(no serial dependence in squared residuals; estimate untrustworthy)"
    elseif !best.converged
        "GMM optimization did not converge"
    else
        "converged"
    end

    varnames = copy(model.varnames)
    shock_names = ["Shock $j" for j in 1:n]
    LewisTVVResult{T}(B0, Q, theta, Matrix{T}(best.vcov), se_B0,
        T(best.J_stat), T(best.J_pvalue), K, lagvec, weighting,
        best.converged, best.iterations, weak, T(strength), message,
        shocks, varnames, shock_names)
end
