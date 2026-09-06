# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Brock-Dechert-Scheinkman-LeBaron (BDS) test for iid/independence (EV-28, #436).

After fitting an ARIMA or GARCH model, the standard residual check is Ljung-Box,
which detects only *linear* dependence. The BDS test detects *any* remaining
departure from iid — nonlinear structure, neglected conditional heteroskedasticity,
or chaos — and is the canonical post-GARCH adequacy check.

The engine is a vectorised correlation-integral computation over a symmetric
`T×T` `BitMatrix` `Θ[i,j] = (|yᵢ − yⱼ| < ε)`. For embedding dimension `m` the
`m`-history indicator is built incrementally:

    Θ⁽¹⁾[s,t] = Θ[s,t]
    Θ⁽ᵐ⁾[s,t] = Θ⁽ᵐ⁻¹⁾[s,t]  &  Θ[s+m−1, t+m−1]          (s,t ∈ 1..T−m+1)

so `Θ⁽ᵐ⁾` is obtained from `Θ⁽ᵐ⁻¹⁾` by one shifted `.&`, and the m-fold product is
NEVER recomputed from scratch. The correlation integral is

    C_m(ε) = (2 / (Tₘ(Tₘ−1))) Σ_{s<t} Θ⁽ᵐ⁾[s,t],     Tₘ = T − m + 1.

References:
- Brock, W. A., Dechert, W. D., Scheinkman, J. A. & LeBaron, B. (1996). "A test
  for independence based on the correlation dimension." Econometric Reviews 15(3).
- Brock, W. A., Hsieh, D. A. & LeBaron, B. (1991). Nonlinear Dynamics, Chaos, and
  Instability. MIT Press.
- Kanzler, L. (1999). "Very Fast and Correctly Sized Estimation of the BDS
  Statistic." Oxford working paper.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Correlation-integral engine
# =============================================================================

"""
Build the symmetric `T×T` `BitMatrix` `Θ[i,j] = (|yᵢ − yⱼ| < ε)` (strict `<`).
`Θ[i,i] = 1` (distance 0). O(T²) memory — for very long series subsample first.
"""
function _bds_theta(y::AbstractVector{T}, eps::T) where {T<:AbstractFloat}
    n = length(y)
    Θ = falses(n, n)
    @inbounds for j in 1:n
        yj = y[j]
        for i in 1:n
            Θ[i, j] = abs(y[i] - yj) < eps
        end
    end
    return Θ
end

"""
Dimension-1 correlation integral `C = C_1(ε)` and the `K` estimator from a
pre-built base indicator `Θ` (full sample of `T` points).

- `k_i` = number of points within `ε` of point `i` (`j ≠ i`) = (row sum of `Θ`) − 1.
- `C = C_1(ε) = (2 / (T(T−1))) Σ_{i<j} Θ[i,j] = (Σ kᵢ) / (T(T−1))`.
- `K̂ = Σ_i kᵢ(kᵢ−1) / (T(T−1)(T−2))`  — the U-statistic estimate of the
  probability that two point-pairs sharing a common point are both within `ε`
  (Kanzler 1999; equivalently `(6/(T(T−1)(T−2))) Σ_{t<s<r} hₜₛᵣ` with the
  symmetrised triple indicator `hₜₛᵣ`). This is the documented failure point of
  BDS variance code, so it is written out explicitly.
"""
function _bds_c1_k(Θ::BitMatrix, ::Type{T}) where {T<:AbstractFloat}
    n = size(Θ, 1)
    sumk = 0          # Σ kᵢ
    sumkk = 0         # Σ kᵢ(kᵢ−1)
    @inbounds for i in 1:n
        ki = 0
        for j in 1:n
            ki += Θ[i, j]
        end
        ki -= 1       # drop the diagonal (self) term
        sumk += ki
        sumkk += ki * (ki - 1)
    end
    C = T(sumk) / (T(n) * T(n - 1))
    K = T(sumkk) / (T(n) * T(n - 1) * T(n - 2))
    return C, K
end

"""
Correlation integrals `C_m(ε)` for every `m` in `ms`, from a pre-built base
indicator `Θ`. Builds the embedded indicator incrementally (see file header):
a single `T×T` `BitMatrix` `emb` starts as `Θ` and, at step `m`, is updated by
`emb[1:Tₘ,1:Tₘ] .&= Θ[m:T, m:T]` (the `(m−1)`-shifted base). `C_m` is then the
off-diagonal upper-triangle count of `emb` restricted to `1..Tₘ`, normalised.

Returns a `Vector{T}` aligned with `ms` (entry is `NaN` when `Tₘ < 2`).
"""
function _bds_cm(Θ::BitMatrix, ms::AbstractVector{Int}, ::Type{T}) where {T<:AbstractFloat}
    n = size(Θ, 1)
    mmax = maximum(ms)
    emb = copy(Θ)                      # Θ⁽¹⁾
    out = Dict{Int,T}()
    # C_1 requested directly:
    if 1 in ms
        # off-diagonal upper-triangle count of Θ
        s = 0
        @inbounds for i in 1:n, j in (i+1):n
            s += Θ[i, j]
        end
        out[1] = T(2 * s) / (T(n) * T(n - 1))
    end
    for m in 2:mmax
        Tm = n - m + 1
        Tm < 1 && break
        # Θ⁽ᵐ⁾[s,t] = Θ⁽ᵐ⁻¹⁾[s,t] & Θ[s+m−1, t+m−1]; the shift is Θ[m:T, m:T].
        @inbounds @views emb[1:Tm, 1:Tm] .&= Θ[m:n, m:n]
        if m in ms
            if Tm < 2
                out[m] = T(NaN)
            else
                total = 0
                @inbounds for i in 1:Tm, j in (i+1):Tm
                    total += emb[i, j]
                end
                out[m] = T(2 * total) / (T(Tm) * T(Tm - 1))
            end
        end
    end
    return T[get(out, m, T(NaN)) for m in ms]
end

"""
Asymptotic variance `σ²_m(ε)` of the BDS statistic (Brock et al. 1996):

    σ²_m = 4 [ Kᵐ + 2 Σ_{j=1}^{m−1} K^{m−j} C^{2j} + (m−1)² C^{2m} − m² K C^{2m−2} ]

with `C = C_1(ε)` and the `K` estimator above. The `K−C²` cross terms (here the
`−m² K C^{2m−2}` and the `Σ K^{m−j} C^{2j}` blocks) are where BDS variance code
usually goes wrong, so each block is written separately. For `m=2` this collapses
to the classic `σ²_2 = 4(K − C²)²`.
"""
function _bds_sigma2(C::T, K::T, m::Int) where {T<:AbstractFloat}
    term_Km   = K^m                                   # Kᵐ
    term_sum  = zero(T)                               # 2 Σ_{j=1}^{m−1} K^{m−j} C^{2j}
    @inbounds for j in 1:(m - 1)
        term_sum += K^(m - j) * C^(2j)
    end
    term_sum *= 2
    term_end  = T(m - 1)^2 * C^(2m)                   # (m−1)² C^{2m}
    term_cross = T(m)^2 * K * C^(2m - 2)              # m² K C^{2m−2}
    return 4 * (term_Km + term_sum + term_end - term_cross)
end

# =============================================================================
# Statistic assembly
# =============================================================================

"""
BDS `w_m` statistics for a series `y` over embedding dims `ms` and thresholds
`epsvals`. Returns `(W, C)` matrices of shape `length(ms) × length(epsvals)`,

    w_m = √T (C_m(ε) − C_1(ε)^m) / σ_m(ε),

using the FULL-sample `C_1` and `K` (Brock et al. 1996 convention). Cells with a
non-positive/degenerate variance (e.g. `ε` larger than every pairwise distance)
are `NaN`. `C[i,j]` holds the raw correlation integral `C_{ms[i]}(epsvals[j])`.
"""
function _bds_stats(y::AbstractVector{T}, ms::AbstractVector{Int},
                    epsvals::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(y)
    nm = length(ms)
    ne = length(epsvals)
    W = Matrix{T}(undef, nm, ne)
    Cmat = Matrix{T}(undef, nm, ne)
    for (je, eps) in enumerate(epsvals)
        Θ = _bds_theta(y, eps)
        C1, K = _bds_c1_k(Θ, T)
        Cm = _bds_cm(Θ, ms, T)
        for (im, m) in enumerate(ms)
            cm = Cm[im]
            Cmat[im, je] = cm
            Tm = n - m + 1
            if !isfinite(cm) || Tm < 2
                W[im, je] = T(NaN)
                continue
            end
            σ2 = _bds_sigma2(C1, K, m)
            if !(σ2 > 0) || !isfinite(σ2)
                W[im, je] = T(NaN)
            else
                W[im, je] = sqrt(T(n)) * (cm - C1^m) / sqrt(σ2)
            end
        end
    end
    return W, Cmat
end

# =============================================================================
# Public API
# =============================================================================

"""
    bds_test(y::AbstractVector; m=2:6, eps_frac=0.7, bootstrap=0, seed=1234) -> BDSResult
    bds_test(model::AbstractARIMAModel; kwargs...)      # tests model residuals
    bds_test(model::AbstractVolatilityModel; kwargs...) # tests standardized residuals

Brock-Dechert-Scheinkman-LeBaron test for iid/independence of a series.

- `H₀`: the observations are independent and identically distributed.
- `H₁`: the observations are not iid (nonlinear dependence, neglected conditional
  heteroskedasticity, or deterministic chaos).

The statistic `w_m = √T (C_m(ε) − C_1(ε)^m) / σ_m(ε) → N(0,1)` is computed for
each embedding dimension `m` and threshold `ε`; the (two-sided) N(0,1) p-value is
reported per `(m, ε)` cell. Large `|w_m|` rejects iid.

# Arguments
- `y`: the series (raw vector, or supply a fitted model to test its residuals).

# Keyword arguments
- `m`: embedding dimensions (default `2:6`). Values `m ≥ T` are dropped.
- `eps_frac`: threshold multiplier(s) of the sample sd; `ε = eps_frac · std(y)`.
  Pass a `Real` (default `0.7`) or a vector (e.g. `[0.5, 1.0, 1.5, 2.0]`) for a
  multi-`ε` table.
- `bootstrap`: number of permutation replications for an iid-null p-value
  (default `0` = asymptotic only; use `≥ 500` when requested). Each replication
  permutes `y` (destroying dependence under H₀) and recomputes `w_m`; the
  bootstrap p-value is the fraction of `|w*| ≥ |w_obs|`.
- `seed`: RNG seed for the permutation bootstrap.

A small-sample warning is emitted when `T < 200` (the asymptotic N(0,1) is then
unreliable — prefer `bootstrap`).

# Model dispatches
For `AbstractARIMAModel` the raw residuals are tested. For volatility models
(`GARCHModel` and relatives) the **standardized** residuals `εₜ/σ̂ₜ` are tested —
running BDS on raw returns would merely re-detect the volatility clustering the
model already removed.

# Example
```julia
r = bds_test(randn(500))
r = bds_test(estimate_garch(returns, 1, 1))   # post-GARCH adequacy check
```

# References
- Brock, Dechert, Scheinkman & LeBaron (1996), Econometric Reviews 15(3).
- Brock, Hsieh & LeBaron (1991), Nonlinear Dynamics, Chaos, and Instability.
"""
function bds_test(y::AbstractVector{<:Real};
                  m=2:6, eps_frac::Union{Real,AbstractVector{<:Real}}=0.7,
                  bootstrap::Int=0, seed::Int=1234)
    yv = float.(collect(y))
    T = eltype(yv)
    n = length(yv)
    bootstrap >= 0 || throw(ArgumentError("bootstrap must be ≥ 0, got $bootstrap"))

    ms = collect(Int, m)
    ms = sort(unique(filter(mm -> mm >= 1, ms)))
    isempty(ms) && throw(ArgumentError("no valid embedding dimension in m=$m"))
    valid = filter(mm -> n - mm + 1 >= 2, ms)
    isempty(valid) && throw(ArgumentError(
        "series length T=$n too short for any embedding dimension in $ms " *
        "(need T − m + 1 ≥ 2)"))
    ms = valid

    sd = T(std(yv))
    sd > 0 || throw(ArgumentError("series has zero variance; ε = eps_frac·sd is 0"))
    fracs = eps_frac isa Real ? T[T(eps_frac)] : T[T(f) for f in eps_frac]
    all(f -> f > 0, fracs) || throw(ArgumentError("eps_frac values must be > 0"))
    epsvals = sd .* fracs

    W, Cmat = _bds_stats(yv, ms, epsvals)
    pval = T[isfinite(w) ? T(2 * ccdf(Normal(), abs(w))) : T(NaN) for w in W]
    pval = reshape(pval, size(W))

    boot_p = fill(T(NaN), size(W))
    if bootstrap > 0
        boot_p = _bds_bootstrap(yv, ms, epsvals, W, bootstrap, seed, T)
    end

    small = n < 200
    if small
        @warn "BDS test: T=$n < 200; the asymptotic N(0,1) approximation is " *
              "unreliable — consider the permutation bootstrap (bootstrap=…)."
    end

    return BDSResult{T}(ms, epsvals, fracs, sd, W, pval, boot_p, Cmat,
                        n, small, bootstrap, seed)
end

"""Permutation-bootstrap p-values under H₀ (iid). Permutes `y` each replication,
recomputes `w_m`, and returns the fraction of `|w*| ≥ |w_obs|` per `(m, ε)`."""
function _bds_bootstrap(y::AbstractVector{T}, ms::Vector{Int}, epsvals::Vector{T},
                        Wobs::Matrix{T}, B::Int, seed::Int, ::Type{T}) where {T<:AbstractFloat}
    rng = Random.Xoshiro(seed)
    n = length(y)
    ge = zeros(Int, size(Wobs))
    absobs = abs.(Wobs)
    perm = collect(1:n)
    for _ in 1:B
        Random.shuffle!(rng, perm)
        yb = y[perm]
        Wb, _ = _bds_stats(yb, ms, epsvals)
        @inbounds for k in eachindex(Wb)
            (isfinite(Wb[k]) && isfinite(absobs[k]) && abs(Wb[k]) >= absobs[k]) && (ge[k] += 1)
        end
    end
    return T[isfinite(absobs[k]) ? T(ge[k] / B) : T(NaN) for k in eachindex(absobs)] |>
           v -> reshape(v, size(Wobs))
end

# --- Model dispatches ---

"""Test the residuals of a fitted ARIMA-family model for remaining dependence."""
bds_test(model::AbstractARIMAModel; kwargs...) = bds_test(StatsAPI.residuals(model); kwargs...)

"""Test the **standardized** residuals `εₜ/σ̂ₜ` of a fitted volatility model."""
bds_test(model::AbstractVolatilityModel; kwargs...) =
    bds_test(model.standardized_residuals; kwargs...)
