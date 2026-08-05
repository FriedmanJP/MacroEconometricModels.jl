# Counterfactual module — empirical causal-effect adapters (CF-04, #384)
#
# Lossless extractors from the package's IRF machinery into the CF containers.
# Nothing is re-estimated, resampled or thinned here. The top silent-bug risk
# is the draw-layout permutation: every package draw array is
# (draw, horizon, variable, shock); the CF containers store (H, n_s, n_draws).

# --- selector resolution: Int index or String name -> Int index -------------

function _cf_resolve(sel::Integer, names::Vector{String}, what::String)
    1 <= sel <= length(names) || throw(ArgumentError(
        "$what index $sel out of bounds 1:$(length(names))"))
    return Int(sel)
end

function _cf_resolve(sel::AbstractString, names::Vector{String}, what::String)
    idx = findfirst(==(String(sel)), names)
    idx === nothing && throw(ArgumentError(
        "$what \"$sel\" not found among $(names)"))
    return idx
end

_cf_resolve(sel, names::Vector{String}, what::String) = throw(ArgumentError(
    "$what selector must be an Int or String, got $(typeof(sel))"))

# --- draw-layout permutation (the ONE place package draws are re-laid-out) ---

# Package layout (draw, horizon, variable, shock) -> container (H, n_s, n_draws)
# for a single variable index vi and selected shock columns.
_pce_draws_slice(D::AbstractArray{T,4}, H::Int, vi::Int, shock_idx::Vector{Int}) where {T} =
    permutedims(D[:, 1:H, vi, shock_idx], (2, 3, 1))

# Package layout (draw, horizon, variable, shock) -> (H, n_draws) for one
# (variable, shock) pair.
_bp_draws_slice(D::AbstractArray{T,4}, H::Int, vi::Int, si::Int) where {T} =
    permutedims(D[:, 1:H, vi, si], (2, 1))

function _cf_check_horizon(H::Int, avail_H::Int)
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    H <= avail_H || throw(ArgumentError(
        "H = $H exceeds the available IRF horizon $avail_H; re-run irf with a longer horizon — the solve horizon must exceed the reporting horizon (McKay–Wolf use H = 100)"))
    return nothing
end

# --- normalization: rescale so the FIRST instrument's impact response is +1 --

function _pce_normalize!(Theta_x::Vector{Matrix{T}}, Theta_z::Vector{Matrix{T}},
                         Dx::Union{Nothing,Vector{Array{T,3}}},
                         Dz::Union{Nothing,Vector{Array{T,3}}},
                         labels::Vector{String}, instruments::Vector{Symbol}) where {T}
    isempty(Theta_z) && throw(ArgumentError(
        "normalize = :instrument_impact requires at least one instrument"))
    n_s = size(Theta_z[1], 2)
    guard = T(1e-10)
    # point normalization
    for k in 1:n_s
        cscale = Theta_z[1][1, k]
        abs(cscale) >= guard || throw(ArgumentError(
            "cannot normalize: point impact response of :$(instruments[1]) to shock \"$(labels[k])\" is $cscale (|·| < 1e-10)"))
        for M in Theta_x
            M[:, k] ./= cscale
        end
        for M in Theta_z
            M[:, k] ./= cscale
        end
    end
    # per-draw normalization
    (Dx === nothing && Dz === nothing) && return (Dx, Dz)
    Dz === nothing && throw(ArgumentError(
        "normalize = :instrument_impact with draws requires instrument draws (Theta_z_draws) for the per-draw rescaling"))
    nd = size(Dz[1], 3)
    keep = trues(nd)
    for d in 1:nd, k in 1:n_s
        abs(Dz[1][1, k, d]) < guard && (keep[d] = false)
    end
    n_drop = nd - count(keep)
    if n_drop > 0
        @warn "instrument-impact normalization dropped $n_drop of $nd draws with near-zero impact response (|·| < 1e-10) of :$(instruments[1])"
    end
    kept = findall(keep)
    scale_of = (d, k) -> Dz[1][1, k, d]
    function _renorm(arrs)
        arrs === nothing && return nothing
        out = Vector{Array{T,3}}(undef, length(arrs))
        for (i, A) in enumerate(arrs)
            B = A[:, :, kept]
            for (jd, d) in enumerate(kept), k in 1:n_s
                B[:, k, jd] ./= scale_of(d, k)
            end
            out[i] = B
        end
        return out
    end
    # NOTE: rescale Dx with the ORIGINAL Dz scales, so renormalize Dx first.
    Dx2 = _renorm(Dx)
    Dz2 = _renorm(Dz)
    return (Dx2, Dz2)
end

# --- shared builder for (values, package-layout draws) sources ---------------

function _pce_build(values::AbstractArray{T,3}, draws4::Union{Nothing,AbstractArray{T,4}},
                    variables::Vector{String}, shocknames::Vector{String},
                    shocks::AbstractVector, outcomes::AbstractVector{<:Pair},
                    instruments::AbstractVector{<:Pair};
                    H::Int, normalize::Symbol, source::Symbol) where {T<:AbstractFloat}
    normalize in (:none, :instrument_impact) || throw(ArgumentError(
        "normalize: expected :none or :instrument_impact, got :$normalize"))
    _cf_check_horizon(H, size(values, 1))
    isempty(shocks) && throw(ArgumentError("shocks: expected at least one policy shock"))
    shock_idx = Int[_cf_resolve(s, shocknames, "shock") for s in shocks]
    out_syms = Symbol[first(p) for p in outcomes]
    out_idx = Int[_cf_resolve(last(p), variables, "variable") for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    ins_idx = Int[_cf_resolve(last(p), variables, "variable") for p in instruments]

    Theta_x = [Matrix{T}(values[1:H, vi, shock_idx]) for vi in out_idx]
    Theta_z = [Matrix{T}(values[1:H, vi, shock_idx]) for vi in ins_idx]
    Dx = draws4 === nothing ? nothing :
         Array{T,3}[_pce_draws_slice(draws4, H, vi, shock_idx) for vi in out_idx]
    Dz = draws4 === nothing ? nothing :
         Array{T,3}[_pce_draws_slice(draws4, H, vi, shock_idx) for vi in ins_idx]
    labels = shocknames[shock_idx]

    if normalize == :instrument_impact
        Dx, Dz = _pce_normalize!(Theta_x, Theta_z, Dx, Dz, labels, ins_syms)
    end
    PolicyCausalEffects{T}(out_syms, ins_syms, Theta_x, Theta_z, Dx, Dz, H, labels, source)
end

"""
    policy_causal_effects(ir::ImpulseResponse, shocks, outcomes, instruments=[];
                          H=ir.horizon, normalize=:none, source=:var)

Extract a [`PolicyCausalEffects`](@ref) container from a frequentist
[`ImpulseResponse`](@ref) — the causal effects of the identified **policy**
shocks on the chosen outcomes and instruments.

- `shocks`: vector of `Int` indices or `String` names selecting the policy
  shock columns of the IRF.
- `outcomes` / `instruments`: vectors of `Pair{Symbol,Union{Int,String}}`
  mapping module symbols to IRF variables (e.g. `[:infl => "infl", :ygap => 2]`)
  — matching is by symbol downstream, never by position.
- `H`: truncation horizon, at most `ir.horizon` (re-run `irf` with a longer
  horizon otherwise; the solve horizon must exceed the reporting horizon).
- `normalize = :instrument_impact` rescales every shock column so the FIRST
  instrument's impact response is `+1` (per-draw renormalization; draws with
  near-zero impact are dropped with a warning). `:none` keeps the estimator's
  scale.

Bootstrap draws in `ir._draws` (layout `reps × horizon × variable × shock`)
are carried over losslessly into the container layout `H × n_s × n_draws`.
A `VECMModel` works for free via `to_var(vecm)` followed by `irf` — no
separate dispatch is needed.
"""
policy_causal_effects(ir::ImpulseResponse{T}, shocks::AbstractVector,
                      outcomes::AbstractVector{<:Pair},
                      instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                      H::Int=ir.horizon, normalize::Symbol=:none,
                      source::Symbol=:var) where {T<:AbstractFloat} =
    _pce_build(ir.values, ir._draws, ir.variables, ir.shocks,
               shocks, outcomes, instruments; H=H, normalize=normalize, source=source)

"""
    policy_causal_effects(bir::BayesianImpulseResponse, shocks, outcomes,
                          instruments=[]; H=bir.horizon, normalize=:none,
                          source=:bvar)

Posterior variant: point matrices from `bir.point_estimate`, uncertainty from
the stored posterior draws (`bir._draws`, layout
`n_draws × horizon × variable × shock`). Errors when the result carries no
draws. See the [`ImpulseResponse`](@ref) method for the argument conventions.
"""
function policy_causal_effects(bir::BayesianImpulseResponse{T}, shocks::AbstractVector,
                               outcomes::AbstractVector{<:Pair},
                               instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                               H::Int=bir.horizon, normalize::Symbol=:none,
                               source::Symbol=:bvar) where {T<:AbstractFloat}
    bir._draws === nothing && throw(ArgumentError(
        "BayesianImpulseResponse carries no stored posterior draws; re-run irf(post, horizon) on the BVARPosterior (draws are stored on the main path)"))
    _pce_build(bir.point_estimate, bir._draws, bir.variables, bir.shocks,
               shocks, outcomes, instruments; H=H, normalize=normalize, source=source)
end

"""
    policy_causal_effects(s::SignIdentifiedSet, shocks, outcomes, instruments=[];
                          H=..., normalize=:none)

Set-identification variant: point matrices from [`irf_median`](@ref), draws
from the accepted rotations (`s.irf_draws`). Tagged `source = :sign_set`. See
the [`ImpulseResponse`](@ref) method for the argument conventions.
"""
policy_causal_effects(s::SignIdentifiedSet{T}, shocks::AbstractVector,
                      outcomes::AbstractVector{<:Pair},
                      instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                      H::Int=size(s.irf_draws, 2), normalize::Symbol=:none) where {T<:AbstractFloat} =
    _pce_build(irf_median(s), s.irf_draws, s.variables, s.shocks,
               shocks, outcomes, instruments; H=H, normalize=normalize, source=:sign_set)

"""
    policy_causal_effects(slp::StructuralLP, shocks, outcomes, instruments=[];
                          H=..., normalize=:none, n_draws=500, rng=Random.default_rng())

Structural-LP variant: point matrices from `slp.irf`, uncertainty draws
sampled as `N(value, se)` per `(h, variable, shock)` from `slp.se`.

!!! note
    LP draws are an **independent-normal approximation** — no cross-horizon or
    cross-variable correlation. Fine for pointwise bands; NOT a joint
    posterior.
"""
function policy_causal_effects(slp::StructuralLP{T}, shocks::AbstractVector,
                               outcomes::AbstractVector{<:Pair},
                               instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                               H::Int=slp.irf.horizon, normalize::Symbol=:none,
                               n_draws::Int=500,
                               rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    ir = slp.irf
    draws4 = _lp_normal_draws(ir.values, slp.se, n_draws, rng)
    _pce_build(ir.values, draws4, ir.variables, ir.shocks,
               shocks, outcomes, instruments; H=H, normalize=normalize, source=:lp)
end

# Independent-normal approximation draws in package layout (draw, h, var, shock).
function _lp_normal_draws(values::AbstractArray{T,3}, se::AbstractArray{T,3},
                          n_draws::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    n_draws >= 1 || throw(ArgumentError("n_draws: expected n_draws >= 1, got $n_draws"))
    size(values) == size(se) || throw(ArgumentError(
        "se: expected size $(size(values)) (matching the IRF values), got $(size(se))"))
    Hh, nv, ns = size(values)
    D = Array{T,4}(undef, n_draws, Hh, nv, ns)
    for s in 1:ns, v in 1:nv, h in 1:Hh
        mu = values[h, v, s]
        sd = se[h, v, s]
        for d in 1:n_draws
            D[d, h, v, s] = mu + sd * randn(rng, T)
        end
    end
    return D
end

"""
    policy_causal_effects(lpr::LPImpulseResponse, outcomes, instruments=[];
                          H=..., normalize=:none, n_draws=500, rng=Random.default_rng())

Single-shock LP variant (`n_s = 1`): point from `lpr.values`, draws sampled as
`N(value, se)` (independent-normal approximation — see the
[`StructuralLP`](@ref) method note). Convenience dispatches on `LPModel` /
`LPIVModel` route through `lp_irf` / `lp_iv_irf`. Tagged `source = :lp`.
"""
function policy_causal_effects(lpr::LPImpulseResponse{T},
                               outcomes::AbstractVector{<:Pair},
                               instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                               H::Int=size(lpr.values, 1), normalize::Symbol=:none,
                               n_draws::Int=500,
                               rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    # (H+1) × n_resp single-shock layout -> (h, var, shock) with n_s = 1
    Hh, nv = size(lpr.values)
    values = reshape(lpr.values, Hh, nv, 1)
    se = reshape(lpr.se, Hh, nv, 1)
    draws4 = _lp_normal_draws(values, se, n_draws, rng)
    _pce_build(values, draws4, lpr.response_vars, [lpr.shock_var],
               [1], outcomes, instruments; H=H, normalize=normalize, source=:lp)
end

policy_causal_effects(m::LPModel, outcomes::AbstractVector{<:Pair},
                      instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[]; kwargs...) =
    policy_causal_effects(lp_irf(m), outcomes, instruments; kwargs...)

policy_causal_effects(m::LPIVModel, outcomes::AbstractVector{<:Pair},
                      instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[]; kwargs...) =
    policy_causal_effects(lp_iv_irf(m), outcomes, instruments; kwargs...)

# --- baseline paths -----------------------------------------------------------

"""
    baseline_path(ir, nonpolicy_shock, outcomes, instruments=[]; H=..., negate=false)

Extract a [`BaselinePath`](@ref): the IRF of every outcome/instrument to ONE
non-policy shock — the disturbance the counterfactual policy responds to.
Works on [`ImpulseResponse`](@ref) (bootstrap draws when present) and
[`BayesianImpulseResponse`](@ref) (posterior draws). `negate = true` flips the
sign (e.g. the contractionary version of an identified expansionary shock —
McKay–Wolf negate their investment shock).
"""
function baseline_path(ir::ImpulseResponse{T}, nonpolicy_shock,
                       outcomes::AbstractVector{<:Pair},
                       instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                       H::Int=ir.horizon, negate::Bool=false) where {T<:AbstractFloat}
    _bp_build(ir.values, ir._draws, ir.variables, ir.shocks, nonpolicy_shock,
              outcomes, instruments; H=H, negate=negate)
end

function baseline_path(bir::BayesianImpulseResponse{T}, nonpolicy_shock,
                       outcomes::AbstractVector{<:Pair},
                       instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                       H::Int=bir.horizon, negate::Bool=false) where {T<:AbstractFloat}
    _bp_build(bir.point_estimate, bir._draws, bir.variables, bir.shocks, nonpolicy_shock,
              outcomes, instruments; H=H, negate=negate)
end

function _bp_build(values::AbstractArray{T,3}, draws4::Union{Nothing,AbstractArray{T,4}},
                   variables::Vector{String}, shocknames::Vector{String},
                   nonpolicy_shock, outcomes::AbstractVector{<:Pair},
                   instruments::AbstractVector{<:Pair};
                   H::Int, negate::Bool) where {T<:AbstractFloat}
    _cf_check_horizon(H, size(values, 1))
    si = _cf_resolve(nonpolicy_shock, shocknames, "shock")
    out_syms = Symbol[first(p) for p in outcomes]
    out_idx = Int[_cf_resolve(last(p), variables, "variable") for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    ins_idx = Int[_cf_resolve(last(p), variables, "variable") for p in instruments]
    sgn = negate ? -one(T) : one(T)
    x = [sgn .* Vector{T}(values[1:H, vi, si]) for vi in out_idx]
    z = [sgn .* Vector{T}(values[1:H, vi, si]) for vi in ins_idx]
    x_draws = draws4 === nothing ? nothing :
              Matrix{T}[sgn .* _bp_draws_slice(draws4, H, vi, si) for vi in out_idx]
    z_draws = draws4 === nothing ? nothing :
              Matrix{T}[sgn .* _bp_draws_slice(draws4, H, vi, si) for vi in ins_idx]
    label = shocknames[si] * (negate ? " (negated)" : "")
    BaselinePath{T}(out_syms, ins_syms, x, z, x_draws, z_draws, H, label)
end

# --- Wold representation --------------------------------------------------------

"""
    wold_representation(m::VARModel; H, orthogonalize=:cholesky)

Orthonormalized Wold representation of the estimated VAR:
`Theta[h, :, :] = Ψ_{h−1} · chol(Σ)` with `Ψ` the reduced-form MA coefficients
(`Theta[1, :, :]` is the impact matrix).

**Rotation invariance**: any orthogonalization works — second-moment and
historical counterfactuals are invariant to the rotation (CMW 2025, App. A.2:
an orthogonal `P` cancels in `Σ_h Θ_h Θ_h'`). The Cholesky ordering here
carries no identification content and is NOT exposed as a modeling choice;
`orthogonalize = :none` returns the reduced-form `Ψ_h` instead.
"""
function wold_representation(m::VARModel{T}; H::Int,
                             orthogonalize::Symbol=:cholesky) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    orthogonalize in (:cholesky, :none) || throw(ArgumentError(
        "orthogonalize: expected :cholesky or :none, got :$orthogonalize"))
    n = nvars(m)
    A = extract_ar_coefficients(m.B, n, m.p)
    P = orthogonalize == :cholesky ? Matrix{T}(safe_cholesky(m.Sigma)) : Matrix{T}(I, n, n)
    Theta = _cf_structural_ma(A, P, H)
    WoldRepresentation{T}(Theta, copy(m.Sigma), copy(m.varnames), nothing)
end

"""
    wold_representation(post::BVARPosterior; H, orthogonalize=:cholesky,
                        max_draws=post.n_draws)

Posterior variant: the point representation is computed at the posterior means
of `(B, Σ)`; per-draw representations are stacked into `draws`
(`H × n × n × n_draws`, capped at `max_draws`). Rotation invariance as in the
`VARModel` method.
"""
function wold_representation(post::BVARPosterior{T}; H::Int,
                             orthogonalize::Symbol=:cholesky,
                             max_draws::Int=post.n_draws) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    orthogonalize in (:cholesky, :none) || throw(ArgumentError(
        "orthogonalize: expected :cholesky or :none, got :$orthogonalize"))
    max_draws >= 1 || throw(ArgumentError("max_draws: expected max_draws >= 1, got $max_draws"))
    n = post.n
    p = post.p
    B_bar = dropdims(mean(post.B_draws, dims=1), dims=1)
    Sigma_bar = Matrix{T}((dropdims(mean(post.Sigma_draws, dims=1), dims=1) .+
                           dropdims(mean(post.Sigma_draws, dims=1), dims=1)') ./ 2)
    A_bar = extract_ar_coefficients(B_bar, n, p)
    P_bar = orthogonalize == :cholesky ? Matrix{T}(safe_cholesky(Sigma_bar)) : Matrix{T}(I, n, n)
    Theta = _cf_structural_ma(A_bar, P_bar, H)

    nd = min(max_draws, post.n_draws)
    draws = Array{T,4}(undef, H, n, n, nd)
    for d in 1:nd
        Bd = Matrix{T}(post.B_draws[d, :, :])
        Sd = Matrix{T}(post.Sigma_draws[d, :, :])
        Ad = extract_ar_coefficients(Bd, n, p)
        Pd = orthogonalize == :cholesky ? Matrix{T}(safe_cholesky(Sd)) : Matrix{T}(I, n, n)
        draws[:, :, :, d] = _cf_structural_ma(Ad, Pd, H)
    end
    WoldRepresentation{T}(Theta, Sigma_bar, copy(post.varnames), draws)
end

# --- variable-major stacking (shared with OPP CF-13 and IRF matching CF-06) ---

"""
    _stack(blocks, order) -> stacked

Variable-major stacking `[x₁; x₂; …]` of length-`H` path blocks (vectors) or
`H × n_s` matrix blocks, in the given symbol order. One implementation shared
by the OPP statistic (CF-13) and the IRF-matching target (CF-06) so the two
never disagree on block order.
"""
function _stack(blocks::AbstractDict{Symbol,<:AbstractVecOrMat}, order::AbstractVector{Symbol})
    isempty(order) && throw(ArgumentError("_stack: expected at least one block symbol"))
    for s in order
        haskey(blocks, s) || throw(ArgumentError(
            "_stack: block :$s missing from $(sort!(collect(keys(blocks))))"))
    end
    return reduce(vcat, (blocks[s] for s in order))
end
