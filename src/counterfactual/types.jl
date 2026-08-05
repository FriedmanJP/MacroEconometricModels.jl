# Counterfactual module — core types (CF-01, #381)
#
# Shared sufficient-statistic containers for the policy-counterfactual methods of
# Barnichon & Mesters (2023, AER), McKay & Wolf (2023, ECMA), and Caravello,
# McKay & Wolf (2025). Downstream engines (rule counterfactuals, optimal policy,
# OPP, model averaging) consume a `PolicyCausalEffects` container plus a
# `PolicyRule` or `PolicyLoss` specification.

"""
    AbstractCounterfactual

Supertype for all result and container types of the policy-counterfactual module
(McKay–Wolf rule counterfactuals, Barnichon–Mesters optimal policy perturbations,
Caravello–McKay–Wolf model averaging).
"""
abstract type AbstractCounterfactual end

# Provenance tags for the causal-effect container. Adapters (CF-04..CF-08) and
# the pooling layer (CF-17) each stamp their own tag.
const _CF_SOURCES = (:var, :bvar, :lp, :sign_set, :dsge, :hank, :pooled, :manual)

"""
    PolicyCausalEffects{T<:AbstractFloat} <: AbstractCounterfactual

Sufficient-statistic container for policy counterfactuals: the causal effects of
identified policy (news) shocks on outcomes and policy instruments over a
truncation horizon `H`, in deviations from steady state / target gaps.

# Fields
- `outcomes::Vector{Symbol}`: names of the `n_x ≥ 1` outcome variables.
- `instruments::Vector{Symbol}`: names of the `n_z ≥ 0` policy instruments
  (may be empty, e.g. for a pure OPP optimality test).
- `Theta_x::Vector{Matrix{T}}`: `n_x` matrices, each `H × n_s`; column `k` of
  `Theta_x[i]` is the IRF of outcome `i` to identified policy shock `k`.
- `Theta_z::Vector{Matrix{T}}`: `n_z` matrices, each `H × n_s`, likewise for the
  instruments.
- `Theta_x_draws::Union{Nothing,Vector{Array{T,3}}}`: optional uncertainty draws,
  `n_x` arrays of size `H × n_s × n_draws` (a single draw is the contiguous slice
  `[:, :, d]`).
- `Theta_z_draws::Union{Nothing,Vector{Array{T,3}}}`: likewise for instruments.
- `H::Int`: truncation horizon.
- `shock_labels::Vector{String}`: `n_s` labels for the identified shocks.
- `source::Symbol`: provenance, one of `$( _CF_SOURCES )`.

A container is **square** when `n_s == H` (full news menu, one column per
announcement horizon — the model-implied case) and **thin** when `n_s < H`
(empirically identified shocks); see [`is_square`](@ref). Columns of a thin
container are whatever shocks the user identified — downstream code must never
assume they are horizon-ordered.

Construct via the keyword constructor:

    PolicyCausalEffects(; outcomes, Theta_x, instruments=Symbol[], Theta_z=nothing,
                        Theta_x_draws=nothing, Theta_z_draws=nothing,
                        shock_labels=nothing, source=:manual)

which accepts any `AbstractMatrix`/`AbstractArray` inputs, converts them to
concrete `Matrix{T}`/`Array{T,3}`, derives `H` and `n_s` from `Theta_x[1]`, and
validates every dimension (throwing a descriptive `ArgumentError` on any
inconsistency). When `shock_labels` is omitted, labels default to
`"shock 1", …, "shock n_s"`.
"""
struct PolicyCausalEffects{T<:AbstractFloat} <: AbstractCounterfactual
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    Theta_x::Vector{Matrix{T}}
    Theta_z::Vector{Matrix{T}}
    Theta_x_draws::Union{Nothing,Vector{Array{T,3}}}
    Theta_z_draws::Union{Nothing,Vector{Array{T,3}}}
    H::Int
    shock_labels::Vector{String}
    source::Symbol

    function PolicyCausalEffects{T}(outcomes, instruments, Theta_x, Theta_z,
                                    Theta_x_draws, Theta_z_draws, H, shock_labels,
                                    source) where {T<:AbstractFloat}
        isempty(outcomes) && throw(ArgumentError(
            "outcomes: expected at least 1 outcome, got 0"))
        length(Theta_x) == length(outcomes) || throw(ArgumentError(
            "Theta_x: expected $(length(outcomes)) matrices (one per outcome), got $(length(Theta_x))"))
        length(Theta_z) == length(instruments) || throw(ArgumentError(
            "Theta_z: expected $(length(instruments)) matrices (one per instrument), got $(length(Theta_z))"))
        H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
        size(Theta_x[1], 1) == H || throw(ArgumentError(
            "Theta_x[1]: expected $H rows (= H), got $(size(Theta_x[1], 1))"))
        n_s = size(Theta_x[1], 2)
        n_s >= 1 || throw(ArgumentError(
            "Theta_x[1]: expected at least 1 shock column, got $n_s"))
        for (i, Th) in enumerate(Theta_x)
            size(Th) == (H, n_s) || throw(ArgumentError(
                "Theta_x[$i]: expected size (H, n_s) = ($H, $n_s), got $(size(Th))"))
        end
        for (k, Th) in enumerate(Theta_z)
            size(Th) == (H, n_s) || throw(ArgumentError(
                "Theta_z[$k]: expected size (H, n_s) = ($H, $n_s), got $(size(Th))"))
        end
        length(shock_labels) == n_s || throw(ArgumentError(
            "shock_labels: expected $n_s labels (= n_s), got $(length(shock_labels))"))
        source in _CF_SOURCES || throw(ArgumentError(
            "source: expected one of $(_CF_SOURCES), got :$source"))
        nd = 0
        if Theta_x_draws !== nothing
            length(Theta_x_draws) == length(outcomes) || throw(ArgumentError(
                "Theta_x_draws: expected $(length(outcomes)) arrays (one per outcome), got $(length(Theta_x_draws))"))
            nd = size(Theta_x_draws[1], 3)
            nd >= 1 || throw(ArgumentError(
                "Theta_x_draws[1]: expected n_draws >= 1, got $nd"))
            for (i, D) in enumerate(Theta_x_draws)
                size(D) == (H, n_s, nd) || throw(ArgumentError(
                    "Theta_x_draws[$i]: expected size (H, n_s, n_draws) = ($H, $n_s, $nd), got $(size(D))"))
            end
        end
        if Theta_z_draws !== nothing
            length(Theta_z_draws) == length(instruments) || throw(ArgumentError(
                "Theta_z_draws: expected $(length(instruments)) arrays (one per instrument), got $(length(Theta_z_draws))"))
            for (k, D) in enumerate(Theta_z_draws)
                nd_k = nd > 0 ? nd : size(Theta_z_draws[1], 3)
                size(D) == (H, n_s, nd_k) || throw(ArgumentError(
                    "Theta_z_draws[$k]: expected size (H, n_s, n_draws) = ($H, $n_s, $nd_k), got $(size(D))"))
            end
        end
        new{T}(outcomes, instruments, Theta_x, Theta_z, Theta_x_draws, Theta_z_draws,
               H, shock_labels, source)
    end
end

# Promote the element types of all provided arrays to a common AbstractFloat.
function _cf_promote_eltype(collections...)
    TT = Bool
    for c in collections
        c === nothing && continue
        for a in c
            TT = promote_type(TT, eltype(a))
        end
    end
    T = float(TT)
    T <: AbstractFloat || throw(ArgumentError(
        "element type: expected inputs convertible to an AbstractFloat, got $TT"))
    return T
end

function PolicyCausalEffects(; outcomes::AbstractVector{Symbol},
                             Theta_x::AbstractVector{<:AbstractMatrix},
                             instruments::AbstractVector{Symbol}=Symbol[],
                             Theta_z::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                             Theta_x_draws::Union{Nothing,AbstractVector{<:AbstractArray{<:Real,3}}}=nothing,
                             Theta_z_draws::Union{Nothing,AbstractVector{<:AbstractArray{<:Real,3}}}=nothing,
                             shock_labels::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                             source::Symbol=:manual)
    isempty(Theta_x) && throw(ArgumentError(
        "Theta_x: expected at least 1 matrix, got 0"))
    T = _cf_promote_eltype(Theta_x, Theta_z, Theta_x_draws, Theta_z_draws)
    H = size(first(Theta_x), 1)
    n_s = size(first(Theta_x), 2)
    labels = shock_labels === nothing ? [string("shock ", k) for k in 1:n_s] :
             String.(collect(shock_labels))
    PolicyCausalEffects{T}(collect(Symbol, outcomes), collect(Symbol, instruments),
                           [Matrix{T}(M) for M in Theta_x],
                           Theta_z === nothing ? Matrix{T}[] : [Matrix{T}(M) for M in Theta_z],
                           Theta_x_draws === nothing ? nothing : [Array{T,3}(A) for A in Theta_x_draws],
                           Theta_z_draws === nothing ? nothing : [Array{T,3}(A) for A in Theta_z_draws],
                           H, labels, source)
end

"""
    PolicyRule{T<:AbstractFloat}

Linear policy-rule specification over a truncation horizon `H`:

`Σᵢ A_x[i] · xᵢ + Σₖ A_z[k] · zₖ = wedge`

where `xᵢ` and `zₖ` are the `H`-period paths of outcome `i` and instrument `k`.

# Fields
- `outcomes::Vector{Symbol}`, `instruments::Vector{Symbol}`: variable names.
- `A_x::Vector{Matrix{T}}`: one `H × H` coefficient matrix per outcome.
- `A_z::Vector{Matrix{T}}`: one `H × H` coefficient matrix per instrument.
- `wedge::Vector{T}`: length-`H` initial conditions / exogenous rule terms.
- `name::String`: human-readable rule label.

Construct via the keyword constructor (template constructors arrive with CF-02):

    PolicyRule(; outcomes, instruments, A_x, A_z, wedge=zeros(H), name="custom")

All matrices must be square `H × H` and `wedge` of length `H`; inputs are
converted to concrete `Matrix{T}`/`Vector{T}` and validated with descriptive
`ArgumentError`s.
"""
struct PolicyRule{T<:AbstractFloat}
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    A_x::Vector{Matrix{T}}
    A_z::Vector{Matrix{T}}
    wedge::Vector{T}
    name::String

    function PolicyRule{T}(outcomes, instruments, A_x, A_z, wedge, name) where {T<:AbstractFloat}
        length(A_x) == length(outcomes) || throw(ArgumentError(
            "A_x: expected $(length(outcomes)) matrices (one per outcome), got $(length(A_x))"))
        length(A_z) == length(instruments) || throw(ArgumentError(
            "A_z: expected $(length(instruments)) matrices (one per instrument), got $(length(A_z))"))
        (isempty(A_x) && isempty(A_z)) && throw(ArgumentError(
            "PolicyRule: expected at least one A_x or A_z matrix to derive H, got none"))
        H = isempty(A_x) ? size(A_z[1], 1) : size(A_x[1], 1)
        H >= 1 || throw(ArgumentError("PolicyRule: expected H >= 1, got $H"))
        for (i, A) in enumerate(A_x)
            size(A) == (H, H) || throw(ArgumentError(
                "A_x[$i]: expected square size (H, H) = ($H, $H), got $(size(A))"))
        end
        for (k, A) in enumerate(A_z)
            size(A) == (H, H) || throw(ArgumentError(
                "A_z[$k]: expected square size (H, H) = ($H, $H), got $(size(A))"))
        end
        length(wedge) == H || throw(ArgumentError(
            "wedge: expected length H = $H, got $(length(wedge))"))
        new{T}(outcomes, instruments, A_x, A_z, wedge, name)
    end
end

function PolicyRule(; outcomes::AbstractVector{Symbol},
                    instruments::AbstractVector{Symbol},
                    A_x::AbstractVector{<:AbstractMatrix},
                    A_z::AbstractVector{<:AbstractMatrix},
                    wedge::Union{Nothing,AbstractVector{<:Real}}=nothing,
                    name::AbstractString="custom")
    (isempty(A_x) && isempty(A_z)) && throw(ArgumentError(
        "PolicyRule: expected at least one A_x or A_z matrix to derive H, got none"))
    T = _cf_promote_eltype(A_x, A_z, wedge === nothing ? nothing : (wedge,))
    H = isempty(A_x) ? size(first(A_z), 1) : size(first(A_x), 1)
    w = wedge === nothing ? zeros(T, H) : Vector{T}(wedge)
    PolicyRule{T}(collect(Symbol, outcomes), collect(Symbol, instruments),
                  [Matrix{T}(A) for A in A_x], [Matrix{T}(A) for A in A_z],
                  w, String(name))
end

"""
    PolicyLoss{T<:AbstractFloat}

Quadratic policy-loss specification over a truncation horizon `H`:

`L = Σᵢ xᵢ' W_x[i] xᵢ + Σₖ zₖ' W_z[k] zₖ`

with the outcome weights `λᵢ` and the discounting `βʰ` already folded into the
`W` matrices.

# Fields
- `outcomes::Vector{Symbol}`, `instruments::Vector{Symbol}`: variable names.
- `W_x::Vector{Matrix{T}}`: one `H × H` weight matrix per outcome.
- `W_z::Union{Nothing,Vector{Matrix{T}}}`: optional instrument penalties
  (e.g. Δz smoothing), one `H × H` matrix per instrument.
- `lambda::Vector{T}`: the per-outcome loss weights `λᵢ` (recorded for reporting;
  already folded into `W_x`).
- `beta::T`: discount factor in `(0, 1]` (recorded; already folded into `W`).
- `name::String`: human-readable loss label.

Construct via the keyword constructor (template builders arrive with CF-02):

    PolicyLoss(; outcomes, W_x, instruments=Symbol[], W_z=nothing,
               lambda=ones(n_x), beta=1.0, name="custom")

The `W` matrices are deliberately NOT numerically verified to be positive
semidefinite: legitimate loss weights are often PSD-singular by construction
(e.g. average-inflation-targeting weights), so a tolerance-based eigenvalue
check would falsely reject them.
"""
struct PolicyLoss{T<:AbstractFloat}
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    W_x::Vector{Matrix{T}}
    W_z::Union{Nothing,Vector{Matrix{T}}}
    lambda::Vector{T}
    beta::T
    name::String

    function PolicyLoss{T}(outcomes, instruments, W_x, W_z, lambda, beta, name) where {T<:AbstractFloat}
        isempty(outcomes) && throw(ArgumentError(
            "outcomes: expected at least 1 outcome, got 0"))
        length(W_x) == length(outcomes) || throw(ArgumentError(
            "W_x: expected $(length(outcomes)) matrices (one per outcome), got $(length(W_x))"))
        H = size(W_x[1], 1)
        H >= 1 || throw(ArgumentError("PolicyLoss: expected H >= 1, got $H"))
        for (i, W) in enumerate(W_x)
            size(W) == (H, H) || throw(ArgumentError(
                "W_x[$i]: expected square size (H, H) = ($H, $H), got $(size(W))"))
        end
        if W_z !== nothing
            length(W_z) == length(instruments) || throw(ArgumentError(
                "W_z: expected $(length(instruments)) matrices (one per instrument), got $(length(W_z))"))
            for (k, W) in enumerate(W_z)
                size(W) == (H, H) || throw(ArgumentError(
                    "W_z[$k]: expected square size (H, H) = ($H, $H), got $(size(W))"))
            end
        end
        length(lambda) == length(outcomes) || throw(ArgumentError(
            "lambda: expected $(length(outcomes)) weights (one per outcome), got $(length(lambda))"))
        zero(beta) < beta <= one(beta) || throw(ArgumentError(
            "beta: expected 0 < beta <= 1, got $beta"))
        new{T}(outcomes, instruments, W_x, W_z, lambda, beta, name)
    end
end

function PolicyLoss(; outcomes::AbstractVector{Symbol},
                    W_x::AbstractVector{<:AbstractMatrix},
                    instruments::AbstractVector{Symbol}=Symbol[],
                    W_z::Union{Nothing,AbstractVector{<:AbstractMatrix}}=nothing,
                    lambda::Union{Nothing,AbstractVector{<:Real}}=nothing,
                    beta::Real=1.0,
                    name::AbstractString="custom")
    T = _cf_promote_eltype(W_x, W_z, lambda === nothing ? nothing : (lambda,), (beta,))
    lam = lambda === nothing ? ones(T, length(outcomes)) : Vector{T}(lambda)
    PolicyLoss{T}(collect(Symbol, outcomes), collect(Symbol, instruments),
                  [Matrix{T}(W) for W in W_x],
                  W_z === nothing ? nothing : [Matrix{T}(W) for W in W_z],
                  lam, T(beta), String(name))
end

"""
    BaselinePath{T<:AbstractFloat} <: AbstractCounterfactual

The baseline the counterfactual acts on: the IRF of every outcome/instrument to
ONE non-policy shock (or a forecast path, CF-05), over `H` periods.

# Fields
- `outcomes::Vector{Symbol}`, `instruments::Vector{Symbol}`: variable names.
- `x::Vector{Vector{T}}`, `z::Vector{Vector{T}}`: one length-`H` path per
  outcome / instrument.
- `x_draws::Union{Nothing,Vector{Matrix{T}}}`, `z_draws`: optional uncertainty
  draws, one `H × n_draws` matrix per variable.
- `H::Int`: truncation horizon.
- `label::String`: what the baseline is (e.g. the non-policy shock name).

Built by [`baseline_path`](@ref) (CF-04) and the forecast adapters (CF-05).
"""
struct BaselinePath{T<:AbstractFloat} <: AbstractCounterfactual
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    x::Vector{Vector{T}}
    z::Vector{Vector{T}}
    x_draws::Union{Nothing,Vector{Matrix{T}}}
    z_draws::Union{Nothing,Vector{Matrix{T}}}
    H::Int
    label::String

    function BaselinePath{T}(outcomes, instruments, x, z, x_draws, z_draws, H,
                             label) where {T<:AbstractFloat}
        length(x) == length(outcomes) || throw(ArgumentError(
            "x: expected $(length(outcomes)) paths (one per outcome), got $(length(x))"))
        length(z) == length(instruments) || throw(ArgumentError(
            "z: expected $(length(instruments)) paths (one per instrument), got $(length(z))"))
        H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
        for (i, v) in enumerate(x)
            length(v) == H || throw(ArgumentError(
                "x[$i]: expected length H = $H, got $(length(v))"))
        end
        for (k, v) in enumerate(z)
            length(v) == H || throw(ArgumentError(
                "z[$k]: expected length H = $H, got $(length(v))"))
        end
        nd = 0
        if x_draws !== nothing
            length(x_draws) == length(outcomes) || throw(ArgumentError(
                "x_draws: expected $(length(outcomes)) matrices (one per outcome), got $(length(x_draws))"))
            nd = isempty(x_draws) ? 0 : size(x_draws[1], 2)
            for (i, D) in enumerate(x_draws)
                size(D) == (H, nd) || throw(ArgumentError(
                    "x_draws[$i]: expected size (H, n_draws) = ($H, $nd), got $(size(D))"))
            end
        end
        if z_draws !== nothing
            length(z_draws) == length(instruments) || throw(ArgumentError(
                "z_draws: expected $(length(instruments)) matrices (one per instrument), got $(length(z_draws))"))
            for (k, D) in enumerate(z_draws)
                nd_k = nd > 0 ? nd : size(z_draws[1], 2)
                size(D) == (H, nd_k) || throw(ArgumentError(
                    "z_draws[$k]: expected size (H, n_draws) = ($H, $nd_k), got $(size(D))"))
            end
        end
        new{T}(outcomes, instruments, x, z, x_draws, z_draws, H, label)
    end
end

"""
    PolicyForecast{T<:AbstractFloat} <: AbstractCounterfactual

Baseline forecast of the objective **gaps** `E_t Y_t⁰` — one of the two
sufficient statistics of the Barnichon–Mesters OPP (CF-13/14).

# Fields
- `outcomes::Vector{Symbol}`: gap variable names.
- `values::Vector{Vector{T}}`: one length-`H` **gap** path per outcome
  (deviations from target, NOT levels — feeding levels makes the OPP drive
  levels to zero).
- `draws::Union{Nothing,Vector{Matrix{T}}}`: optional forecast-uncertainty
  draws, one `H × n_draws` matrix per outcome (semantics aligned with
  [`BaselinePath`](@ref) — OPP consumes both interchangeably).
- `H::Int`: forecast horizon.
- `origin::String`: forecast origin label (e.g. `"2008M4"`, `"2021Q2"`).

Built by [`policy_forecast`](@ref). The forecast must be conditional on the
**baseline** policy rule: when OPP recommendations are adopted repeatedly,
each subsequent forecast must still be constructed under the *old* rule
(Barnichon–Mesters, web appendix S0.5).
"""
struct PolicyForecast{T<:AbstractFloat} <: AbstractCounterfactual
    outcomes::Vector{Symbol}
    values::Vector{Vector{T}}
    draws::Union{Nothing,Vector{Matrix{T}}}
    H::Int
    origin::String

    function PolicyForecast{T}(outcomes, values, draws, H, origin) where {T<:AbstractFloat}
        isempty(outcomes) && throw(ArgumentError(
            "outcomes: expected at least 1 outcome, got 0"))
        length(values) == length(outcomes) || throw(ArgumentError(
            "values: expected $(length(outcomes)) paths (one per outcome), got $(length(values))"))
        H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
        for (i, v) in enumerate(values)
            length(v) == H || throw(ArgumentError(
                "values[$i]: expected length H = $H, got $(length(v))"))
        end
        if draws !== nothing
            length(draws) == length(outcomes) || throw(ArgumentError(
                "draws: expected $(length(outcomes)) matrices (one per outcome), got $(length(draws))"))
            nd = isempty(draws) ? 0 : size(draws[1], 2)
            for (i, D) in enumerate(draws)
                size(D) == (H, nd) || throw(ArgumentError(
                    "draws[$i]: expected size (H, n_draws) = ($H, $nd), got $(size(D))"))
            end
        end
        new{T}(outcomes, values, draws, H, origin)
    end
end

"""
    WoldRepresentation{T<:AbstractFloat} <: AbstractCounterfactual

Orthonormalized Wold (structural MA) representation of a reduced-form VAR:
`Theta[h, i, j]` is the response of variable `i` at horizon `h − 1` to
orthogonalized innovation `j` (`Theta[1, :, :]` is the impact matrix).

# Fields
- `Theta::Array{T,3}`: `H × n_y × n_y` Wold IRFs.
- `Sigma_u::Matrix{T}`: innovation covariance (pre-orthogonalization).
- `varnames::Vector{String}`: variable names.
- `draws::Union{Nothing,Array{T,4}}`: optional `H × n_y × n_y × n_draws`
  posterior draws.

Any orthogonalization works: second-moment and historical counterfactuals are
invariant to the rotation (CMW 2025, App. A.2 — an orthogonal `P` cancels), so
the Cholesky ordering used by [`wold_representation`](@ref) carries no
identification content.
"""
struct WoldRepresentation{T<:AbstractFloat} <: AbstractCounterfactual
    Theta::Array{T,3}
    Sigma_u::Matrix{T}
    varnames::Vector{String}
    draws::Union{Nothing,Array{T,4}}

    function WoldRepresentation{T}(Theta, Sigma_u, varnames, draws) where {T<:AbstractFloat}
        H, n1, n2 = size(Theta)
        n1 == n2 || throw(ArgumentError(
            "Theta: expected square variable dimensions, got ($n1, $n2)"))
        H >= 1 || throw(ArgumentError("Theta: expected H >= 1, got $H"))
        size(Sigma_u) == (n1, n1) || throw(ArgumentError(
            "Sigma_u: expected size ($n1, $n1), got $(size(Sigma_u))"))
        length(varnames) == n1 || throw(ArgumentError(
            "varnames: expected $n1 names, got $(length(varnames))"))
        if draws !== nothing
            size(draws)[1:3] == (H, n1, n1) || throw(ArgumentError(
                "draws: expected leading size (H, n, n) = ($H, $n1, $n1), got $(size(draws)[1:3])"))
        end
        new{T}(Theta, Sigma_u, varnames, draws)
    end
end

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

"""
    is_square(ce::PolicyCausalEffects) -> Bool

`true` when the causal-effect container carries a full news menu (`n_s == H`,
one shock column per announcement horizon — the model-implied case), `false`
for a thin container of empirically identified shocks (`n_s < H`).
"""
is_square(ce::PolicyCausalEffects) = size(ce.Theta_x[1], 2) == ce.H

n_shocks(ce::PolicyCausalEffects) = size(ce.Theta_x[1], 2)

function n_draws(ce::PolicyCausalEffects)
    ce.Theta_x_draws !== nothing && return size(ce.Theta_x_draws[1], 3)
    if ce.Theta_z_draws !== nothing && !isempty(ce.Theta_z_draws)
        return size(ce.Theta_z_draws[1], 3)
    end
    return 0
end

Base.eltype(::Type{PolicyCausalEffects{T}}) where {T} = T
Base.eltype(::Type{PolicyRule{T}}) where {T} = T
Base.eltype(::Type{PolicyLoss{T}}) where {T} = T

function n_draws(bp::BaselinePath)
    bp.x_draws !== nothing && !isempty(bp.x_draws) && return size(bp.x_draws[1], 2)
    bp.z_draws !== nothing && !isempty(bp.z_draws) && return size(bp.z_draws[1], 2)
    return 0
end
n_draws(w::WoldRepresentation) = w.draws === nothing ? 0 : size(w.draws, 4)
n_draws(pf::PolicyForecast) =
    (pf.draws === nothing || isempty(pf.draws)) ? 0 : size(pf.draws[1], 2)

Base.eltype(::Type{BaselinePath{T}}) where {T} = T
Base.eltype(::Type{WoldRepresentation{T}}) where {T} = T
Base.eltype(::Type{PolicyForecast{T}}) where {T} = T

_rule_horizon(r::PolicyRule) = isempty(r.A_x) ? size(r.A_z[1], 1) : size(r.A_x[1], 1)
_loss_horizon(l::PolicyLoss) = size(l.W_x[1], 1)

_cf_plural(n::Int) = n == 1 ? "" : "s"

function Base.show(io::IO, ce::PolicyCausalEffects{T}) where {T}
    n_x = length(ce.outcomes)
    n_z = length(ce.instruments)
    shape = is_square(ce) ? "square" : "thin"
    print(io, "PolicyCausalEffects{$T}: $n_x outcome$(_cf_plural(n_x)), ",
          "$n_z instrument$(_cf_plural(n_z)), H=$(ce.H), n_s=$(n_shocks(ce)) ($shape), ",
          "$(n_draws(ce)) draws, source=:$(ce.source)")
end

function Base.show(io::IO, r::PolicyRule{T}) where {T}
    n_x = length(r.outcomes)
    n_z = length(r.instruments)
    print(io, "PolicyRule{$T} \"$(r.name)\": $n_x outcome$(_cf_plural(n_x)), ",
          "$n_z instrument$(_cf_plural(n_z)), H=$(_rule_horizon(r))")
end

function Base.show(io::IO, l::PolicyLoss{T}) where {T}
    n_x = length(l.outcomes)
    n_z = length(l.instruments)
    print(io, "PolicyLoss{$T} \"$(l.name)\": $n_x outcome$(_cf_plural(n_x)), ",
          "$n_z instrument$(_cf_plural(n_z)), H=$(_loss_horizon(l)), beta=$(l.beta)")
end

function Base.show(io::IO, bp::BaselinePath{T}) where {T}
    n_x = length(bp.outcomes)
    n_z = length(bp.instruments)
    print(io, "BaselinePath{$T} \"$(bp.label)\": $n_x outcome$(_cf_plural(n_x)), ",
          "$n_z instrument$(_cf_plural(n_z)), H=$(bp.H), $(n_draws(bp)) draws")
end

function Base.show(io::IO, w::WoldRepresentation{T}) where {T}
    H, n, _ = size(w.Theta)
    print(io, "WoldRepresentation{$T}: $n variable$(_cf_plural(n)), H=$H, ",
          "$(n_draws(w)) draws")
end

function Base.show(io::IO, pf::PolicyForecast{T}) where {T}
    n_x = length(pf.outcomes)
    origin = isempty(pf.origin) ? "" : ", origin=$(pf.origin)"
    print(io, "PolicyForecast{$T}: $n_x outcome$(_cf_plural(n_x)) (gaps), ",
          "H=$(pf.H), $(n_draws(pf)) draws$origin")
end
