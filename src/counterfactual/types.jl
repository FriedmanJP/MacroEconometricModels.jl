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

"""
    PolicyCounterfactual{T<:AbstractFloat} <: AbstractCounterfactual

Result of a McKay–Wolf rule counterfactual ([`policy_counterfactual`](@ref)):
baseline and counterfactual paths, the enforcing date-0 policy-shock vector
`ν*`, uncertainty bands, and the honesty diagnostics.

# Fields
- `outcomes` / `instruments`: variable names (the causal-effect container's).
- `x_base`/`z_base`, `x_cf`/`z_cf`: length-`H` baseline and counterfactual
  paths per variable.
- `x_bands`/`z_bands`: optional `H × n_quantiles` band matrices per variable.
- `nu::Vector{T}`, `shock_labels`: the enforcing shock vector and its labels.
- `error_path::Vector{T}`, `rel_residual::T`: the implementation-error path
  `M·ν* + b` and its relative norm — the honesty signal; a large value means
  the rule is NOT enforceable with the supplied shocks.
- `rel_residual_bands::Union{Nothing,Vector{T}}`: draw quantiles of the
  relative residual.
- `spanned::Bool`: `rel_residual < spanned_tol` (reporting convenience — the
  honest object is `rel_residual` itself).
- `rule_name::String`, `H::Int`, `quantile_levels::Vector{T}`.
- `n_draws_used::Int`, `n_draws_failed::Int`: draw-propagation honesty counts.
- `loss_base::T`, `loss_cf::T`, `foc_norm::T`: loss accounting and the
  first-order-condition norm — populated by [`optimal_policy`](@ref) (CF-11),
  `NaN` for plain rule counterfactuals.
"""
struct PolicyCounterfactual{T<:AbstractFloat} <: AbstractCounterfactual
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    x_base::Vector{Vector{T}}
    z_base::Vector{Vector{T}}
    x_cf::Vector{Vector{T}}
    z_cf::Vector{Vector{T}}
    x_bands::Union{Nothing,Vector{Matrix{T}}}
    z_bands::Union{Nothing,Vector{Matrix{T}}}
    nu::Vector{T}
    shock_labels::Vector{String}
    error_path::Vector{T}
    rel_residual::T
    rel_residual_bands::Union{Nothing,Vector{T}}
    spanned::Bool
    rule_name::String
    H::Int
    quantile_levels::Vector{T}
    n_draws_used::Int
    n_draws_failed::Int
    loss_base::T
    loss_cf::T
    foc_norm::T
end

# Backward-compatible constructor (CF-10 rule counterfactuals: no loss accounting)
PolicyCounterfactual{T}(outcomes, instruments, x_base, z_base, x_cf, z_cf,
                        x_bands, z_bands, nu, shock_labels, error_path,
                        rel_residual, rel_residual_bands, spanned, rule_name,
                        H, quantile_levels, n_draws_used, n_draws_failed) where {T<:AbstractFloat} =
    PolicyCounterfactual{T}(outcomes, instruments, x_base, z_base, x_cf, z_cf,
                            x_bands, z_bands, nu, shock_labels, error_path,
                            rel_residual, rel_residual_bands, spanned, rule_name,
                            H, quantile_levels, n_draws_used, n_draws_failed,
                            T(NaN), T(NaN), T(NaN))

"""
    CounterfactualMoments{T<:AbstractFloat} <: AbstractCounterfactual

Second-moment (Wold) counterfactual ([`counterfactual_moments`](@ref)): the
unconditional covariance of the mapped variables under the baseline and under
an alternative rule/loss, from re-solving every orthonormalized Wold column.

# Fields
- `varnames::Vector{Symbol}`: mapped variables (outcomes then instruments).
- `Sigma_base`/`Sigma_cf`, `sd_base`/`sd_cf`, `corr_base`/`corr_cf`: VMA-sum
  covariances (`Σ_h Θ_h Θ_h'`), standard deviations, correlations — band-
  limited when `freq_band` is set.
- `sd_cf_bands::Union{Nothing,Matrix{T}}`: draw quantiles of `sd_cf`
  (`n_vars × n_quantiles`).
- `Theta_cf::Array{T,3}`: `H × n_vars × n_innovations` counterfactual Wold
  IRFs.
- `policy_name::String`, `H::Int`.
- `tail_share::T`: `‖Θ_cf[last 10 rows]‖/‖Θ_cf‖` — VMA convergence
  diagnostic; > 1% means `H` must grow.
- `freq_band::Union{Nothing,Tuple{T,T}}`: `(ω_lo, ω_hi)` when band-limited.
"""
struct CounterfactualMoments{T<:AbstractFloat} <: AbstractCounterfactual
    varnames::Vector{Symbol}
    Sigma_base::Matrix{T}
    Sigma_cf::Matrix{T}
    sd_base::Vector{T}
    sd_cf::Vector{T}
    corr_base::Matrix{T}
    corr_cf::Matrix{T}
    sd_cf_bands::Union{Nothing,Matrix{T}}
    Theta_cf::Array{T,3}
    policy_name::String
    H::Int
    tail_share::T
    freq_band::Union{Nothing,Tuple{T,T}}
end

"""
    OPPResult{T<:AbstractFloat} <: AbstractCounterfactual

Barnichon–Mesters Optimal Policy Perturbation ([`opp`](@ref)): is the
announced policy optimal, and if not, by how much to adjust?

# Fields
- `delta::Vector{T}`: the OPP `δ* = −(R'WR)⁻¹R'W·E_tY⁰` — the recommended
  perturbation per identified shock direction, in that shock's instrument
  units (length `n_s`). After `estimate_opp` this is the draw **median** (BM
  convention); `delta_plugin` keeps the plug-in point.
- `delta_plugin::Vector{T}`: the plug-in point estimate (equals `delta` on
  the point version).
- `shock_labels::Vector{String}`.
- `gradient::Vector{T}`: `R_y'W·E_tY⁰ (+ instrument terms)` — the FOC
  statistic; optimality ⟺ `gradient = 0` ⟺ `δ* = 0`.
- `loss_base::T`, `loss_opp::T`: loss at the announced policy and after the
  perturbation (`loss_opp ≤ loss_base` by construction).
- `Y_base`/`Y_opp`: objective gap paths before/after, per loss outcome.
- `P_base`/`P_opp`: announced/recommended instrument paths (`nothing` for the
  pure optimality test — the recommendation needs them, the test does not).
- `outcomes`/`instruments`, `H`, `origin`.
- `delta_draws`, `bands`, `reject`, `n_failed`: two-source simulation output —
  filled by `estimate_opp` (CF-14), `nothing`/0 on the point version. Bands
  are equal-tailed at the BM levels (default 60/75/90%): **rejection at LOWER
  levels is the conservative choice for a policymaker averse to running
  non-optimal policy** (the reversed test polarity, BM §5.1).
"""
struct OPPResult{T<:AbstractFloat} <: AbstractCounterfactual
    delta::Vector{T}
    delta_plugin::Vector{T}
    shock_labels::Vector{String}
    gradient::Vector{T}
    loss_base::T
    loss_opp::T
    Y_base::Vector{Vector{T}}
    Y_opp::Vector{Vector{T}}
    P_base::Union{Nothing,Vector{Vector{T}}}
    P_opp::Union{Nothing,Vector{Vector{T}}}
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    H::Int
    origin::String
    delta_draws::Union{Nothing,Matrix{T}}
    bands::Union{Nothing,Dict{T,Matrix{T}}}
    reject::Union{Nothing,Dict{T,Vector{Bool}}}
    n_failed::Int
end

"""
    OPPSequence{T<:AbstractFloat} <: AbstractCounterfactual

A policy-evaluation history ([`opp_sequence`](@ref)): the OPP at every
decision date, with the time-consistency decomposition of its revisions.

# Fields
- `dates::Vector{String}`: opaque date labels.
- `delta::Matrix{T}`: `n_s × n_dates` OPPs (draw medians when simulated;
  `NaN` columns for missing/failed dates).
- `delta_tc::Matrix{T}`: the time-consistent OPP — revisions purged of the
  preference/operator part (`delta − pref_part`).
- `news_part`/`pref_part`/`aging_part`: the exact revision decomposition
  `δ_t − δ_{t−1} = news + pref + aging` (see [`opp_sequence`](@ref); the
  aging term is the mechanical horizon roll that BM's infinite-horizon
  stationary operator hides — under finite-`H` truncation it is real).
- `bands`/`reject`: per-date CF-14 output (`level => n_s×2×n_dates` /
  `level => n_s×n_dates`), `nothing` without simulation.
- `shock_labels`, `loss_name`.
"""
struct OPPSequence{T<:AbstractFloat} <: AbstractCounterfactual
    dates::Vector{String}
    delta::Matrix{T}
    delta_tc::Matrix{T}
    news_part::Matrix{T}
    pref_part::Matrix{T}
    aging_part::Matrix{T}
    bands::Union{Nothing,Dict{T,Array{T,3}}}
    reject::Union{Nothing,Dict{T,Matrix{Bool}}}
    shock_labels::Vector{String}
    loss_name::String
end

"""
    ModelBankMember{T<:AbstractFloat} <: AbstractCounterfactual

One structural model estimated by limited-information IRF matching
([`irf_match`](@ref), CMW 2025 §4): the RWMH posterior over its parameters,
the Geweke log marginal likelihood, and the thinned posterior menus.

# Fields
- `name::String`, `param_names::Vector{Symbol}`.
- `theta_draws::Matrix{T}`: kept (thinned) posterior draws, `n_kept × n_params`.
- `log_post::Vector{T}`: per-kept-draw log posterior kernel.
- `log_marglik::T`: Geweke modified-harmonic-mean log marginal likelihood.
- `menu_draws::Vector{PolicyCausalEffects{T}}`: menus rebuilt at the kept
  draws (horizon-truncated to the `T_store` kwarg — memory honesty).
- `acceptance_rate::T`, `H_news::Int`.
"""
struct ModelBankMember{T<:AbstractFloat} <: AbstractCounterfactual
    name::String
    param_names::Vector{Symbol}
    theta_draws::Matrix{T}
    log_post::Vector{T}
    log_marglik::T
    menu_draws::Vector{PolicyCausalEffects{T}}
    acceptance_rate::T
    H_news::Int
end

"""
    CounterfactualHistory{T<:AbstractFloat} <: AbstractCounterfactual

Historical-evolution counterfactual ([`counterfactual_history`](@ref)): the
realized panel next to the path the economy would have followed had the
alternative policy been in place from the first date of the window — built
from forecast REVISIONS only, never from identified structural shocks.

# Fields
- `dates::Vector{String}`, `varnames::Vector{Symbol}` (outcomes then
  instruments).
- `realized::Matrix{T}`, `cf::Matrix{T}`: `n_dates × n_vars` panels.
- `cf_bands::Union{Nothing,Array{T,3}}`: `n_dates × n_vars × n_quantiles`.
- `nu::Matrix{T}`: per-date enforcing shock vectors (`n_s × n_dates`).
- `rel_residual::Vector{T}`: per-date implementation-error residuals.
- `policy_name::String`, `H::Int`, `quantile_levels::Vector{T}`,
  `n_draws_used::Int`, `n_draws_failed::Int`.
"""
struct CounterfactualHistory{T<:AbstractFloat} <: AbstractCounterfactual
    dates::Vector{String}
    varnames::Vector{Symbol}
    realized::Matrix{T}
    cf::Matrix{T}
    cf_bands::Union{Nothing,Array{T,3}}
    nu::Matrix{T}
    rel_residual::Vector{T}
    policy_name::String
    H::Int
    quantile_levels::Vector{T}
    n_draws_used::Int
    n_draws_failed::Int
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

function Base.show(io::IO, pc::PolicyCounterfactual{T}) where {T}
    n_x = length(pc.outcomes)
    n_z = length(pc.instruments)
    print(io, "PolicyCounterfactual{$T} \"$(pc.rule_name)\": ",
          "$n_x outcome$(_cf_plural(n_x)), $n_z instrument$(_cf_plural(n_z)), ",
          "H=$(pc.H), spanned=$(pc.spanned) (rel_residual=$(round(pc.rel_residual, sigdigits=3))), ",
          "$(pc.n_draws_used) draws used ($(pc.n_draws_failed) failed)")
    if !isnan(pc.loss_cf)
        print(io, ", loss $(round(pc.loss_base, sigdigits=4)) -> $(round(pc.loss_cf, sigdigits=4)) ",
              "(‖FOC‖=$(round(pc.foc_norm, sigdigits=3)))")
    end
end

function Base.show(io::IO, r::OPPResult{T}) where {T}
    n_s = length(r.delta)
    origin = isempty(r.origin) ? "" : ", origin=$(r.origin)"
    print(io, "OPPResult{$T}: δ* = $(round.(r.delta, sigdigits=4)) ",
          "($n_s shock direction$(_cf_plural(n_s))), loss ",
          "$(round(r.loss_base, sigdigits=4)) -> $(round(r.loss_opp, sigdigits=4)), ",
          "H=$(r.H)$origin",
          r.delta_draws === nothing ? "" :
          ", $(size(r.delta_draws, 2)) sims ($(r.n_failed) failed)")
end

function Base.show(io::IO, ch::CounterfactualHistory{T}) where {T}
    n_d, n_v = size(ch.realized)
    print(io, "CounterfactualHistory{$T} \"$(ch.policy_name)\": $n_d date$(_cf_plural(n_d)), ",
          "$n_v variable$(_cf_plural(n_v)), max rel_residual = ",
          "$(round(maximum(ch.rel_residual; init=zero(T)), sigdigits=3)), ",
          "$(ch.n_draws_used) draws used ($(ch.n_draws_failed) failed)")
end

function Base.show(io::IO, mb::ModelBankMember{T}) where {T}
    print(io, "ModelBankMember{$T} \"$(mb.name)\": $(size(mb.theta_draws, 1)) kept draws over ",
          "$(length(mb.param_names)) parameter$(_cf_plural(length(mb.param_names))), ",
          "log-ML = $(round(mb.log_marglik, sigdigits=6)), ",
          "acc = $(round(mb.acceptance_rate, sigdigits=3)), H_news = $(mb.H_news)")
end

function Base.show(io::IO, sq::OPPSequence{T}) where {T}
    n_s, n_d = size(sq.delta)
    n_ok = count(t -> all(isfinite, @view(sq.delta[:, t])), 1:n_d)
    print(io, "OPPSequence{$T} \"$(sq.loss_name)\": $n_d date$(_cf_plural(n_d)) ",
          "($n_ok valid), $n_s shock direction$(_cf_plural(n_s))",
          sq.bands === nothing ? "" : ", with bands")
end

function Base.show(io::IO, cm::CounterfactualMoments{T}) where {T}
    n = length(cm.varnames)
    band = cm.freq_band === nothing ? "" :
           ", band=($(round(cm.freq_band[1], sigdigits=3)), $(round(cm.freq_band[2], sigdigits=3)))"
    print(io, "CounterfactualMoments{$T} \"$(cm.policy_name)\": $n variable$(_cf_plural(n)), ",
          "H=$(cm.H), tail_share=$(round(cm.tail_share, sigdigits=3))$band")
end
