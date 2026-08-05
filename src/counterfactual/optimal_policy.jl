# Counterfactual module — optimal-policy projection (CF-11, #391)
#
# Optimal policy in the sufficient-statistics framework is a PROJECTION: the
# reachable outcome set is {x_base + Theta_x·nu}, so minimizing a quadratic
# loss over it is the CF-03 normal-equation solve with loss weights in place
# of rule matrices (McKay & Wolf 2023 Prop. 2, eqs. 16-17/26-27; CMW §5 adds
# the Delta-z smoothing penalty and its wedge initial condition). By certainty
# equivalence the projection on EXPECTED paths solves the stochastic LQ
# problem — no simulation needed (MW eq. 27).
#
# CF-13 NOTE: this function IS the OPP kernel with a different base — the BM
# OPP calls `_optimal_projection` with a PolicyForecast in place of a shock
# IRF (MW optimal policy ≡ BM OPP is a theorem-level identity, encoded as
# code reuse here and tested as an identity in CF-23).

# One weighted stacked solve shared by optimal_policy (CF-11) and the OPP
# (CF-13): blocks are (Theta_j, W_j, base_j); c is the optional linear term.
function _optimal_projection(Theta_blocks::Vector{<:AbstractMatrix{T}},
                             W_blocks::Vector{<:AbstractMatrix{T}},
                             base_blocks::Vector{<:AbstractVector{T}},
                             c::Union{Nothing,Vector{T}}=nothing;
                             Cs::Union{Nothing,Vector{Matrix{T}}}=nothing) where {T<:AbstractFloat}
    isempty(Theta_blocks) && throw(ArgumentError(
        "_optimal_projection: expected at least one (Theta, W, base) block"))
    length(Theta_blocks) == length(W_blocks) == length(base_blocks) || throw(ArgumentError(
        "_optimal_projection: block counts differ ($(length(Theta_blocks)), $(length(W_blocks)), $(length(base_blocks)))"))
    Cw = Cs === nothing ? [_pp_weight_factor(Matrix{T}(W)) for W in W_blocks] : Cs
    A = reduce(vcat, (Cw[j] * Theta_blocks[j] for j in eachindex(Theta_blocks)))
    btil = reduce(vcat, (Cw[j] * base_blocks[j] for j in eachindex(base_blocks)))
    return _policy_projection(Matrix{T}(A), Vector{T}(btil); c=c, method=:ls)
end

# Quadratic loss value ½·(Σ x'W_x x + Σ z'W_z z) − Σ wedge'z, up to the
# additive constant of the smoothing penalty (identical for base and cf, so
# comparisons are exact).
function _policy_loss_value(loss::PolicyLoss{T}, x_paths::Vector{Vector{T}},
                            z_paths::Vector{Vector{T}};
                            z_wedge::Union{Nothing,Vector{Vector{T}}}=nothing) where {T<:AbstractFloat}
    L = zero(T)
    for i in eachindex(loss.outcomes)
        L += x_paths[i]' * loss.W_x[i] * x_paths[i] / 2
    end
    for k in eachindex(loss.instruments)
        loss.W_z !== nothing && (L += z_paths[k]' * loss.W_z[k] * z_paths[k] / 2)
        z_wedge !== nothing && (L -= z_wedge[k]' * z_paths[k])
    end
    return L
end

"""
    optimal_policy(base::BaselinePath, ce::PolicyCausalEffects, loss::PolicyLoss;
                   z_wedge=nothing, draws=:auto, baseline_draws=:fixed,
                   quantiles=(0.16, 0.5, 0.84)) -> PolicyCounterfactual

Optimal-policy projection (McKay–Wolf Prop. 2): choose the date-0 policy-shock
vector minimizing the quadratic `loss` over the reachable set
`{x_base + Θ_x·ν}`,

`ν* = −(Σᵢ Θ_x'W_xΘ_x + Σₖ Θ_z'W_zΘ_z)⁻¹ (Σᵢ Θ_x'W_x·x_base + Σₖ Θ_z'W_z·z_base + c)`,

solved through the whitened projection kernel (never forming `Θ'WΘ` by hand).
Instruments enter **only** when `loss.W_z !== nothing` — the MW loss does not
penalize the instrument. `z_wedge` supplies the smoothing-penalty
initial-condition vectors (one per loss instrument, from
[`smoothing_penalty`](@ref)); it contributes the linear term
`c = −Σₖ Θ_z[k]'·wedge_k`.

The result reuses [`PolicyCounterfactual`](@ref) (`rule_name = loss.name`)
with `loss_base`/`loss_cf` accounting (½-quadratic convention, up to the
smoothing constant common to both) and `foc_norm` — the norm of the MW
optimality condition `Σᵢ Θ_x'W_x·x_cf (+ z-terms)`, ≈ 0 at the optimum ("IRFs
⊥ optimal forecasts", eq. 26). By certainty equivalence this projection on
expected paths solves the stochastic LQ problem. Draw propagation as in
[`policy_counterfactual`](@ref).
"""
function optimal_policy(base::BaselinePath{T}, ce::PolicyCausalEffects{T},
                        loss::PolicyLoss{T};
                        z_wedge::Union{Nothing,AbstractVector{<:AbstractVector{<:Real}}}=nothing,
                        draws::Symbol=:auto,
                        baseline_draws::Symbol=:fixed,
                        quantiles::Union{Tuple,AbstractVector}=(0.16, 0.5, 0.84)) where {T<:AbstractFloat}
    H = ce.H
    base.H == H || throw(ArgumentError(
        "baseline H = $(base.H) does not match the container H = $H"))
    _loss_horizon(loss) == H || throw(ArgumentError(
        "loss H = $(_loss_horizon(loss)) does not match the container H = $H"))
    draws in (:auto, :on, :off) || throw(ArgumentError(
        "draws: expected :auto, :on or :off, got :$draws"))
    for sym in loss.outcomes
        sym in ce.outcomes || throw(ArgumentError(
            "loss outcome :$sym not found in the container outcomes $(ce.outcomes)"))
        sym in base.outcomes || throw(ArgumentError(
            "loss outcome :$sym not found in the baseline outcomes $(base.outcomes)"))
    end
    use_z = loss.W_z !== nothing || z_wedge !== nothing
    if use_z
        isempty(loss.instruments) && throw(ArgumentError(
            "loss has instrument penalties (W_z / z_wedge) but no instruments"))
        for sym in loss.instruments
            sym in ce.instruments || throw(ArgumentError(
                "loss instrument :$sym not found in the container instruments $(ce.instruments)"))
            sym in base.instruments || throw(ArgumentError(
                "loss instrument :$sym not found in the baseline instruments $(base.instruments)"))
        end
        z_wedge === nothing || length(z_wedge) == length(loss.instruments) || throw(ArgumentError(
            "z_wedge: expected $(length(loss.instruments)) vectors (one per loss instrument), got $(length(z_wedge))"))
    end
    for sym in ce.outcomes
        sym in base.outcomes || throw(ArgumentError(
            "container outcome :$sym has no baseline path in $(base.outcomes)"))
    end
    for sym in ce.instruments
        sym in base.instruments || throw(ArgumentError(
            "container instrument :$sym has no baseline path in $(base.instruments)"))
    end

    xb = [Vector{T}(base.x[findfirst(==(s), base.outcomes)]) for s in ce.outcomes]
    zb = [Vector{T}(base.z[findfirst(==(s), base.instruments)]) for s in ce.instruments]
    wz = z_wedge === nothing ? nothing : [Vector{T}(w) for w in z_wedge]

    ix = [findfirst(==(s), ce.outcomes) for s in loss.outcomes]
    iz = use_z ? [findfirst(==(s), ce.instruments) for s in loss.instruments] : Int[]

    # loss-referenced blocks (Theta, W, base) + optional linear term
    function _blocks(Tx, Tz, xb_a, zb_a)
        Th = Vector{Matrix{T}}()
        Wb = Vector{Matrix{T}}()
        Bb = Vector{Vector{T}}()
        for (j, sym) in enumerate(loss.outcomes)
            push!(Th, Tx[ix[j]])
            push!(Wb, loss.W_x[j])
            push!(Bb, xb_a[ix[j]])
        end
        if loss.W_z !== nothing
            for (j, sym) in enumerate(loss.instruments)
                push!(Th, Tz[iz[j]])
                push!(Wb, loss.W_z[j])
                push!(Bb, zb_a[iz[j]])
            end
        end
        c = nothing
        if wz !== nothing
            cv = zeros(T, size(Tx[1], 2))
            for (j, _) in enumerate(loss.instruments)
                cv .-= Tz[iz[j]]' * wz[j]
            end
            c = cv
        end
        return Th, Wb, Bb, c
    end

    Th0, Wb0, Bb0, c0 = _blocks(ce.Theta_x, ce.Theta_z, xb, zb)
    Cs = [_pp_weight_factor(W) for W in Wb0]     # factor the (draw-invariant) weights once
    res = _optimal_projection(Th0, Wb0, Bb0, c0; Cs=Cs)
    nu = res.nu
    x_cf = [xb[i] + ce.Theta_x[i] * nu for i in eachindex(ce.outcomes)]
    z_cf = [zb[k] + ce.Theta_z[k] * nu for k in eachindex(ce.instruments)]

    # loss accounting on the loss-referenced paths
    lx_base = [xb[ix[j]] for j in eachindex(loss.outcomes)]
    lz_base = [zb[iz[j]] for j in eachindex(iz)]
    lx_cf = [x_cf[ix[j]] for j in eachindex(loss.outcomes)]
    lz_cf = [z_cf[iz[j]] for j in eachindex(iz)]
    L_base = _policy_loss_value(loss, lx_base, lz_base; z_wedge=wz)
    L_cf = _policy_loss_value(loss, lx_cf, lz_cf; z_wedge=wz)
    L_cf > L_base + sqrt(eps(T)) * max(one(T), abs(L_base)) &&
        @warn "optimal_policy: loss increased ($(L_base) -> $(L_cf)) — this indicates a kernel/sign bug, please report"

    # FOC (MW eq. 26): Σ Θ_x'W_x x_cf + Σ Θ_z'(W_z z_cf − wedge) ≈ 0
    g = zeros(T, size(ce.Theta_x[1], 2))
    for (j, _) in enumerate(loss.outcomes)
        g .+= ce.Theta_x[ix[j]]' * (loss.W_x[j] * lx_cf[j])
    end
    for (j, _) in enumerate(iz)
        loss.W_z !== nothing && (g .+= ce.Theta_z[iz[j]]' * (loss.W_z[j] * lz_cf[j]))
        wz !== nothing && (g .-= ce.Theta_z[iz[j]]' * wz[j])
    end

    nd = n_draws(ce)
    use_draws = draws == :on || (draws == :auto && nd > 0)
    use_draws && nd == 0 && throw(ArgumentError(
        "draws = :on requires a draws-bearing container"))
    qlev = collect(T, quantiles)
    x_bands = nothing
    z_bands = nothing
    rr_bands = nothing
    n_used = 0
    n_failed = 0
    if use_draws
        solve_d = (Tx_d, Tz_d, xb_d, zb_d) -> begin
            Th_d, _, Bb_d, c_d = _blocks(Tx_d, Tz_d, xb_d, zb_d)
            _optimal_projection(Th_d, Wb0, Bb_d, c_d; Cs=Cs)
        end
        x_bands, z_bands, rr_bands, n_used, n_failed =
            _cf_draw_bands(ce, base, baseline_draws, xb, zb, qlev, solve_d)
    end

    PolicyCounterfactual{T}(copy(ce.outcomes), copy(ce.instruments),
                            xb, zb, x_cf, z_cf, x_bands, z_bands,
                            nu, copy(ce.shock_labels),
                            res.error_path, res.rel_residual, rr_bands,
                            res.rel_residual < T(0.05),
                            loss.name, H, qlev, n_used, n_failed,
                            L_base, L_cf, norm(g))
end

"""
    optimal_rule(ce::PolicyCausalEffects, loss::PolicyLoss; z_wedge=nothing)
        -> PolicyRule

The implied optimal **targeting rule** (square containers only; MW eqs.
16–17): the first-order condition of [`optimal_policy`](@ref) as a rule CF-10
can enforce —

`Σᵢ (Θ_x[i]'W_x[i])·xᵢ + Σₖ (Θ_z[k]'W_z[k])·zₖ = Σₖ Θ_z[k]'·wedge_k`.

Handing this rule to [`policy_counterfactual`](@ref) with the same
`(base, ce)` reproduces the `optimal_policy` paths — the MW "optimal policy is
itself a rule" circle, asserted end-to-end in the CF-23 oracle suite.
"""
function optimal_rule(ce::PolicyCausalEffects{T}, loss::PolicyLoss{T};
                      z_wedge::Union{Nothing,AbstractVector{<:AbstractVector{<:Real}}}=nothing) where {T<:AbstractFloat}
    is_square(ce) || throw(ArgumentError(
        "optimal_rule requires a square (model-implied) container: the targeting rule needs the full news menu"))
    H = ce.H
    for sym in loss.outcomes
        sym in ce.outcomes || throw(ArgumentError(
            "loss outcome :$sym not found in the container outcomes $(ce.outcomes)"))
    end
    _loss_horizon(loss) == H || throw(ArgumentError(
        "loss H = $(_loss_horizon(loss)) does not match the container H = $H"))
    A_x = Vector{Matrix{T}}()
    for (j, sym) in enumerate(loss.outcomes)
        i = findfirst(==(sym), ce.outcomes)
        push!(A_x, Matrix{T}(ce.Theta_x[i]' * loss.W_x[j]))
    end
    A_z = Vector{Matrix{T}}()
    wedge = zeros(T, H)
    for (j, sym) in enumerate(loss.instruments)
        k = findfirst(==(sym), ce.instruments)
        k === nothing && throw(ArgumentError(
            "loss instrument :$sym not found in the container instruments $(ce.instruments)"))
        push!(A_z, loss.W_z === nothing ? zeros(T, H, H) :
                   Matrix{T}(ce.Theta_z[k]' * loss.W_z[j]))
        z_wedge !== nothing && (wedge .+= ce.Theta_z[k]' * Vector{T}(z_wedge[j]))
    end
    PolicyRule{T}(copy(loss.outcomes), copy(loss.instruments), A_x, A_z, wedge,
                  "optimal targeting rule ($(loss.name))")
end
