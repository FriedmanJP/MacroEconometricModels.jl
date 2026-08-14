# bf_wedges.jl — Baqaee–Farhi (2020) wedges, cost vs revenue Domar, Theorem 1
#
# Orientation: ROW (B&F). Cost shares Ω̃; revenue shares Ω = diag(1/μ) Ω̃ on
# producer rows. Theorem 1 (QJE 2020, eq. 4):
#   d log Y = λ̃' d log A  −  λ̃' d log μ  −  Λ̃' d log Λ  +  Λ̃' d log L
#             |__technology__|  |________allocative efficiency________|  |_factor supply_|

"""
    BFWedgeDecomp{T<:AbstractFloat}

Baqaee–Farhi (2020) Theorem 1 first-order decomposition of `d log Y` into a
pure technology term and an allocative-efficiency term, evaluated at a
[`ProductionNetwork`](@ref) with (optional) markup/productivity shocks.

# Theorem 1 (B&F 2020, eq. 4) plus the Hulten factor-supply term

```
d log Y = λ̃' d log A  −  λ̃' d log μ  −  Λ̃' d log Λ  +  Λ̃' d log L
          |__ΔTechnology__|  |______ΔAllocative efficiency______|  |_factor supply_|
```

Theorem 1 itself is stated for productivity and markup shocks (`d log A`,
`d log μ`). Factor-supply shocks enter at first order as `Λ̃' d log L`
(Hulten); they are stored separately as `factor_supply` and are **not** part
of the technology / allocative split. `λ̃` are **cost-based** Domar weights,
`Λ̃` cost-based factor shares (`Σ_f Λ̃_f = 1`), and `d log Λ` is the change
in **revenue-based** factor income shares. When `μ ≡ 1` the allocative term
is zero to first order (Hulten / Corollary 1).

# Fields
- `dlogY` — exact (or first-order) change in log real output.
- `technology` — `Σ_k λ̃_k d log A_k`.
- `allocative` — `−Σ_k λ̃_k d log μ_k − Σ_f Λ̃_f d log Λ_f`.
- `allocative_mu` — markup part `−Σ_k λ̃_k d log μ_k`.
- `allocative_factor` — factor-reallocation part `−Σ_f Λ̃_f d log Λ_f`.
- `factor_supply` — Hulten factor-supply term `Σ_f Λ̃_f d log L_f`.
- `lambda_cost` — base cost-based Domar on real sectors (length `n`).
- `lambda_rev` — base revenue-based Domar on real sectors (length `n`).
- `Lambda_cost`, `Lambda_rev` — base factor shares (length `F`).
- `mu` — sectoral markups length `n` (outer-node markups).
- `dlogA`, `dlogmu` — shocks used (length `n`).
- `dlogL` — factor-supply shocks used (length `F`).
- `dlog_Lambda` — change in revenue factor shares (length `F`).
- `sectors` — real-sector labels.
"""
struct BFWedgeDecomp{T<:AbstractFloat}
    dlogY::T
    technology::T
    allocative::T
    allocative_mu::T
    allocative_factor::T
    factor_supply::T
    lambda_cost::Vector{T}
    lambda_rev::Vector{T}
    Lambda_cost::Vector{T}
    Lambda_rev::Vector{T}
    mu::Vector{T}
    dlogA::Vector{T}
    dlogmu::Vector{T}
    dlogL::Vector{T}
    dlog_Lambda::Vector{T}
    sectors::Vector{String}
end

"""
    bf_wedge_decomp(net::ProductionNetwork; dlogA=0, dlogmu=0, dlogL=0,
                    kwargs...) -> BFWedgeDecomp

Solve the exact equilibrium under the given shocks via [`bf_equilibrium`](@ref)
and return the B&F (2020) Theorem 1 decomposition.

The technology and allocative terms are the **first-order** formula of Theorem 1
evaluated with base cost-based Domars and the equilibrium change in revenue
factor shares. `factor_supply = Λ̃' dlogL` is the Hulten factor-supply term
(`dlogL` is forwarded to the solver; it is outside Theorem 1). `dlogY` is the
exact nonlinear change. For infinitesimal shocks
`dlogY ≈ technology + allocative + factor_supply`.

# Orientation
Row (B&F). See [`BFWedgeDecomp`](@ref).
"""
function bf_wedge_decomp(net::ProductionNetwork{T};
                         dlogA=nothing,
                         dlogmu=nothing,
                         dlogL=nothing,
                         kwargs...) where {T<:AbstractFloat}
    eq = bf_equilibrium(net; dlogA=dlogA, dlogL=dlogL, dlogmu=dlogmu, kwargs...)
    n, M, F = net.n, net.M, net.F
    λ̃_out = T[net.lambda[g] for g in net.outer_nodes]
    λ_rev_out = T[net.lambda_rev[g] for g in net.outer_nodes]
    μ_out = T[net.mu[g - 1] for g in net.outer_nodes]
    Λ̃ = net.lambda[M+2:M+1+F]
    Λ_rev0 = net.lambda_rev[M+2:M+1+F]
    dA = eq.dlogA
    dμ = eq.dlogmu
    dL = eq.dlogL
    dlog_Λ = log.(max.(eq.Lambda, eps(T))) .- log.(max.(Λ_rev0, eps(T)))
    alloc_mu = -dot(λ̃_out, dμ)
    alloc_fac = -dot(Λ̃, dlog_Λ)
    fac_sup = dot(Λ̃, dL)
    BFWedgeDecomp{T}(eq.dlogY, eq.technology, eq.allocative,
                     alloc_mu, alloc_fac, fac_sup,
                     λ̃_out, λ_rev_out, Vector{T}(Λ̃), Vector{T}(Λ_rev0),
                     μ_out, dA, dμ, dL, dlog_Λ, String.(net.io.sectors))
end

"""
    cost_based_domar(net::ProductionNetwork) -> Vector

Cost-based Domar weights `λ̃` on real-sector outer nodes (length `n`).
"""
cost_based_domar(net::ProductionNetwork) =
    [net.lambda[g] for g in net.outer_nodes]

"""
    revenue_based_domar(net::ProductionNetwork) -> Vector

Revenue-based Domar weights `λ` on real-sector outer nodes (length `n`).
Equals [`cost_based_domar`](@ref) when `μ ≡ 1`.
"""
revenue_based_domar(net::ProductionNetwork) =
    [net.lambda_rev[g] for g in net.outer_nodes]

"""
Horizontal-economy closed form (B&F 2020 §2.4, eqs. 5–6).

Producers buy only labor; household CES elasticity `σ`. Cost-based = revenue-based
Domar on goods (`λ̃ = λ`); `Λ̃_L = 1`.

```
dlogY/dlogA_k = λ_k − λ_k (σ−1) (μ_k^{-1}/Σ_j λ_j μ_j^{-1} − 1)
dlogY/dlogμ_k = λ_k · σ · (μ_k^{-1}/Σ_i λ_i μ_i^{-1} − 1)
```
"""
function _bf_horizontal_elasticities(λ::AbstractVector{T}, μ::AbstractVector{T},
                                     σ::T) where {T<:AbstractFloat}
    n = length(λ)
    length(μ) == n || throw(ArgumentError("λ and μ length mismatch"))
    s = zero(T)
    @inbounds for i in 1:n
        s += λ[i] / μ[i]
    end
    s > 0 || throw(ArgumentError("Σ λ/μ must be positive"))
    dY_dA = Vector{T}(undef, n)
    dY_dμ = Vector{T}(undef, n)
    @inbounds for k in 1:n
        gap = (one(T) / μ[k]) / s - one(T)
        dY_dA[k] = λ[k] - λ[k] * (σ - one(T)) * gap
        dY_dμ[k] = λ[k] * σ * gap
    end
    return dY_dA, dY_dμ
end
