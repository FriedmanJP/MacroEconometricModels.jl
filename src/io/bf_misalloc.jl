# bf_misalloc.jl — Baqaee–Farhi (2020) Proposition 5 Harberger triangle / H_μ
#
# Orientation: ROW (B&F). Hessian of log Y in log μ (same sign as BFLocal.second_order).
# Distance L = log(Y*/Y) ≈ −½ Δlogμ' H_μ Δlogμ at the efficient point.

"""
    BFMisallocation{T<:AbstractFloat}

Baqaee–Farhi (2020) Proposition 5 Harberger-style misallocation distance and
the Hessian of log real output in log markups (`H_μ`).

# Fields
- `distance` — exact `L = log(Y*/Y)` from `bf_equilibrium`
- `first_order` — Theorem 1 piece `−λ̃'Δlogμ − Λ̃'ΔlogΛ` at `point` (0 at `:efficient`)
- `second_order` — `−½ Δlogμ' H Δlogμ`
- `H_mu` — `n×n` Hessian of log Y in log μ; `0×0` if `hessian=:none`
- `delta_logmu` — log μ of real-sector outers (0 if μ≡1)
- `point` — `:efficient` | `:observed`
- `lambda` — evaluation-point Domar on outers (length `n`)
- `mu` — sectoral markups (length `n`)
- `sectors` — real-sector labels
"""
struct BFMisallocation{T<:AbstractFloat}
    distance::T                 # exact L = log(Y*/Y) from bf_equilibrium
    first_order::T              # Theorem 1 piece −λ̃'Δlogμ − Λ̃'ΔlogΛ at `point` (0 at :efficient)
    second_order::T             # −½ Δlogμ' H Δlogμ
    H_mu::Matrix{T}             # n×n Hessian of log Y in log μ; 0×0 if hessian=:none
    delta_logmu::Vector{T}      # log μ of real-sector outers (0 if μ≡1)
    point::Symbol               # :efficient | :observed
    lambda::Vector{T}           # evaluation-point Domar on outers (length n)
    mu::Vector{T}               # sectoral markups (length n)
    sectors::Vector{String}
end

"""
    bf_misallocation(net; point=:efficient, hessian=:auto) -> BFMisallocation

Proposition 5 Harberger distance and markup Hessian for a
[`ProductionNetwork`](@ref). Not yet implemented.
"""
function bf_misallocation(net::ProductionNetwork{T}; point::Symbol=:efficient,
                          hessian::Symbol=:auto) where {T<:AbstractFloat}
    error("bf_misallocation not implemented")
end
