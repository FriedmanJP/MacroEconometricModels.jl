# network.jl — network / granularity statistics for IO tables

"""
    NetworkStatsResult

Output of [`network_stats`](@ref): concentration, multiplier dispersion, average
propagation lengths, and degree structure of an IO table.

# Fields
- `domar` — Domar weights ``\\lambda_i = x_i / \\mathrm{GDP}``
- `herfindahl` — Herfindahl index of Domar weights ``\\sum_i \\lambda_i^2``
  (Gabaix granular-residual concentration ingredient)
- `multipliers` — Type I output multipliers (column sums of ``L``)
- `multiplier_dispersion` — standard deviation of output multipliers
- `apl` — ``n \\times n`` average-propagation-length matrix (Dietzenbacher et al. 2005)
- `in_degree` — weighted in-degrees of ``A`` (row sums: intermediate sales intensity)
- `out_degree` — weighted out-degrees of ``A`` (column sums: intermediate purchase intensity)
- `upstreamness` — Antràs-style upstreamness ``\\sum_j L_{ij}`` (row sums of ``L``)
- `downstreamness` — column sums of ``L`` (equals output multipliers under Type I)
- `sectors` — sector labels
"""
struct NetworkStatsResult
    domar::Vector{Float64}
    herfindahl::Float64
    multipliers::Vector{Float64}
    multiplier_dispersion::Float64
    apl::Matrix{Float64}
    in_degree::Vector{Float64}
    out_degree::Vector{Float64}
    upstreamness::Vector{Float64}
    downstreamness::Vector{Float64}
    sectors::Vector{String}
end

"""
    network_stats(io) -> NetworkStatsResult

Network and granularity statistics for a classical IO table (column orientation).

Computes:
1. **Domar weights** and their **Herfindahl** concentration
   ``H = \\sum_i \\lambda_i^2`` — the key ingredient of the Gabaix (2011)
   granular residual (idiosyncratic shocks reweighted by Domar shares).
2. **Output-multiplier dispersion** (std of column sums of ``L``).
3. **Average propagation length (APL)** matrix of Dietzenbacher, Romero &
   Bosma (2005):

```math
H = L(L - I), \\qquad
v_{ij} = \\frac{H_{ij}}{L_{ij} - \\delta_{ij}}
\\quad (0 \\text{ when the denominator vanishes})
```

4. Weighted **in/out-degree** of the technical-coefficients matrix ``A``.
5. **Upstreamness / downstreamness** (row / column sums of ``L``), matching
   [`baqaee_farhi`](@ref) — not recomputed by a different formula.

# Orientation
Column orientation: ``A = Z\\hat{x}^{-1}``, ``L = (I - A)^{-1}``.
APL is identical under the Ghosh dual (Dietzenbacher et al. 2005, §3).
"""
function network_stats(io::IOData)
    λ = Float64.(domar_weights(io))
    HHI = sum(abs2, λ)

    L = leontief_inverse(io)
    mult = vec(sum(L, dims=1))                     # Type I output multipliers
    mult_sd = length(mult) > 1 ? std(Float64.(mult); corrected=true) : 0.0

    apl = _apl_matrix(Matrix{Float64}(L))

    A = technical_coefficients(io)
    in_deg = vec(sum(A, dims=2))                   # intermediate sales intensity
    out_deg = vec(sum(A, dims=1))                  # intermediate purchase intensity

    # Reuse the same definitions as baqaee_farhi (do not invent alternatives).
    up = vec(sum(L, dims=2))
    down = vec(sum(L, dims=1))

    NetworkStatsResult(λ, Float64(HHI), Float64.(mult), Float64(mult_sd),
                       apl, Float64.(in_deg), Float64.(out_deg),
                       Float64.(up), Float64.(down), copy(io.sectors))
end

"""
    _apl_matrix(L) -> Matrix

Average propagation length matrix from a Leontief inverse
``V_{ij} = [L(L-I)]_{ij} / (L_{ij} - \\delta_{ij})``
(Dietzenbacher, Romero & Bosma 2005). Zero when the linkage is null.
"""
function _apl_matrix(L::AbstractMatrix{<:Real})
    n = size(L, 1)
    Lf = Matrix{Float64}(L)
    H = Lf * (Lf - I)                              # Σ_k k A^k = L(L − I)
    V = zeros(Float64, n, n)
    @inbounds for j in 1:n, i in 1:n
        denom = Lf[i, j] - (i == j ? 1.0 : 0.0)
        V[i, j] = abs(denom) > 1e-14 ? H[i, j] / denom : 0.0
    end
    return V
end
