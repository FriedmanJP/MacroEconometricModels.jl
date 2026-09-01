# Counterfactual module — constrained OPP (CF-15, #395)
#
# With constraints (ZLB on the recommended path, pre-commitment pledges) the
# OPP has no closed form: min_delta (EY + R_y d)'W(EY + R_y d) s.t. g(d) >= 0,
# warm-started at the unconstrained OPP (BM §4.3, web appendix S4). Solver:
# NLopt SLSQP per the package's constrained-solver conventions (same pattern
# as src/did/honest_did.jl) — no new dependencies, no QP package.

"""
    OPPConstraint

Supertype of the constrained-OPP constraint DSL: [`PathFloorConstraint`](@ref)
(linear path floors, e.g. the ZLB via [`zlb_constraint`](@ref)) and
[`FunctionConstraint`](@ref) (user `g(δ, paths) ≥ 0`, e.g. BM's state-
contingent liftoff pledge — products of paths, nonconvex).
"""
abstract type OPPConstraint end

"""
    PathFloorConstraint(instrument, floor, horizons)

Linear floor on a recommended instrument path: `P⁰ + R_p·δ ≥ floor` on the
given `horizons` (clamped to `1:H`). Build the ZLB case with
[`zlb_constraint`](@ref).
"""
struct PathFloorConstraint{T<:AbstractFloat} <: OPPConstraint
    instrument::Symbol
    floor::T
    horizons::UnitRange{Int}
end

"""
    FunctionConstraint(g, n_out, name)

User constraint `g(δ, paths) ≥ 0` (elementwise over `n_out` outputs), where
`paths` is a `Dict{Symbol,Vector}` of the recommended instrument paths
`P⁰ + R_p·δ` (ForwardDiff-compatible). Nonconvex pledges (e.g. BM's
three-part Sept-2020 FOMC liftoff pledge, products of paths) belong here —
use `multistart > 1`.
"""
struct FunctionConstraint <: OPPConstraint
    g::Function
    n_out::Int
    name::String
end

"""
    zlb_constraint(; floor=0.0, instrument=:rate, horizons=1:typemax(Int))

Zero-lower-bound [`PathFloorConstraint`](@ref): the recommended `instrument`
path must satisfy `P ≥ floor` on `horizons`.
"""
zlb_constraint(; floor::Real=0.0, instrument::Symbol=:rate,
               horizons::UnitRange{Int}=1:typemax(Int)) =
    PathFloorConstraint{Float64}(instrument, Float64(floor), horizons)

_cf_constraint_name(cn::PathFloorConstraint) =
    "PathFloorConstraint(:$(cn.instrument) ≥ $(cn.floor))"
_cf_constraint_name(cn::FunctionConstraint) = "FunctionConstraint($(cn.name))"

"""
    constrained_opp(fc, ce, loss, constraints; instrument_path, z_wedge=nothing,
                    method=:auto, delta0=nothing, multistart=1,
                    rng=Random.default_rng(), n_sim=0, levels=(0.6, 0.75, 0.9),
                    independent=true)
        -> (; result::OPPResult, method_used, binding, kkt_residual,
             warm_start_feasible)

Constrained OPP (BM §4.3): minimize the quadratic loss over `δ` subject to
the [`OPPConstraint`](@ref)s, warm-started at the unconstrained OPP.

- `method = :auto`/`:slsqp`: NLopt SLSQP with the analytic objective gradient
  `R'W(EY + Rδ)`; `FunctionConstraint` derivatives via `ForwardDiff`.
  `multistart > 1` re-solves from perturbed starts and keeps the best
  feasible point — pledge constraints are nonconvex and single-start SLSQP
  can land in poor local optima.
- `method = :projection`: the BM sequence-script fallback — zero the
  negative `δ` components and re-solve on the remaining columns. Explicitly
  NOT the constrained optimum (warned, tagged `method_used = :projection`);
  floor constraints only.
- `n_sim > 0` reruns the constrained solve per CF-14 simulation draw
  (two-source resampling; expensive, off by default); `delta` is then the
  draw median, `delta_plugin` the unconstrained point.

Diagnostics: `binding` lists constraints with |slack| < 1e-6,
`kkt_residual` is `‖∇L − Aₐ'max(λ,0)‖` with least-squares multipliers on the
active set (≈ 0 at a KKT point), `warm_start_feasible` reports whether the
unconstrained OPP already satisfied the constraints. Infeasible problems
error naming the most-violated constraint.
"""
function constrained_opp(fc::PolicyForecast{T}, ce::PolicyCausalEffects{T},
                         loss::PolicyLoss{T},
                         constraints::AbstractVector{<:OPPConstraint};
                         instrument_path::Union{Nothing,AbstractVector{<:Pair}}=nothing,
                         z_wedge::Union{Nothing,AbstractVector{<:AbstractVector{<:Real}}}=nothing,
                         method::Symbol=:auto,
                         delta0::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         multistart::Int=1,
                         seed::Union{Integer,Nothing}=nothing,
                         rng::AbstractRNG=Random.default_rng(),
                         n_sim::Int=0,
                         levels::Union{Tuple,AbstractVector}=(0.6, 0.75, 0.9),
                         independent::Bool=true) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    method in (:auto, :slsqp, :projection) || throw(ArgumentError(
        "method: expected :auto, :slsqp or :projection, got :$method"))
    isempty(constraints) && throw(ArgumentError(
        "constraints: expected at least one OPPConstraint (use opp/estimate_opp for the unconstrained problem)"))
    multistart >= 1 || throw(ArgumentError("multistart: expected >= 1, got $multistart"))
    instrument_path === nothing && throw(ArgumentError(
        "constrained_opp requires the announced instrument paths (instrument_path) — constraints act on the recommended paths"))

    # unconstrained warm start (validates all CF-13 inputs)
    r_u = _suppress_warnings() do
        opp(fc, ce, loss; instrument_path=instrument_path, z_wedge=z_wedge)
    end
    H = ce.H
    n_s = n_shocks(ce)
    p_syms = Symbol[first(p) for p in instrument_path]
    P0 = [Vector{T}(last(p)) for p in instrument_path]
    Rp = [ce.Theta_z[findfirst(==(s), ce.instruments)] for s in p_syms]
    for cn in constraints
        cn isa PathFloorConstraint && !(cn.instrument in p_syms) && throw(ArgumentError(
            "$( _cf_constraint_name(cn)): instrument :$(cn.instrument) has no announced path in instrument_path"))
    end

    # stacked objective: L(δ) = ½‖Aδ + b̃‖² + c'δ  (same blocks as opp)
    wz = z_wedge === nothing ? nothing : [Vector{T}(w) for w in z_wedge]
    use_z = loss.W_z !== nothing || wz !== nothing
    ix_ce = [findfirst(==(s), ce.outcomes) for s in loss.outcomes]
    ix_fc = [findfirst(==(s), fc.outcomes) for s in loss.outcomes]
    iz_ce = use_z ? [findfirst(==(s), ce.instruments) for s in loss.instruments] : Int[]
    iz_p = use_z ? [findfirst(==(s), p_syms) for s in loss.instruments] : Int[]
    Th, Wb, Bb, cterm = _opp_blocks(loss,
                                    [ce.Theta_x[i] for i in ix_ce],
                                    [Vector{T}(fc.values[i]) for i in ix_fc],
                                    use_z ? [ce.Theta_z[k] for k in iz_ce] : Matrix{T}[],
                                    use_z ? [P0[k] for k in iz_p] : Vector{T}[],
                                    wz)
    Cs = [_pp_weight_factor(W) for W in Wb]
    A64 = Float64.(reduce(vcat, (Cs[j] * Th[j] for j in eachindex(Th))))
    b64 = Float64.(reduce(vcat, (Cs[j] * Bb[j] for j in eachindex(Bb))))
    c64 = cterm === nothing ? zeros(Float64, n_s) : Float64.(cterm)
    P0_64 = [Float64.(v) for v in P0]
    Rp_64 = [Float64.(M) for M in Rp]
    obj(x) = 0.5 * sum(abs2, A64 * x + b64) + dot(c64, x)
    grad!(g, x) = (g .= A64' * (A64 * x + b64) .+ c64)

    paths_of(x) = Dict{Symbol,Any}(s => P0_64[k] + Rp_64[k] * x for (k, s) in enumerate(p_syms))

    # constraint evaluation: vector of (value ≥ 0 feasible, description)
    function _violations(x::Vector{Float64})
        v = Float64[]
        who = String[]
        for cn in constraints
            if cn isa PathFloorConstraint
                k = findfirst(==(cn.instrument), p_syms)
                hs = intersect(cn.horizons, 1:H)
                path = P0_64[k] + Rp_64[k] * x
                for h in hs
                    push!(v, path[h] - Float64(cn.floor))
                    push!(who, _cf_constraint_name(cn) * " @h=$h")
                end
            else
                gv = cn.g(x, paths_of(x))
                for (i, gi) in enumerate(gv)
                    push!(v, Float64(gi))
                    push!(who, _cf_constraint_name(cn) * "[$i]")
                end
            end
        end
        return v, who
    end

    delta_u64 = Float64.(r_u.delta)
    v_u, _ = _violations(delta_u64)
    warm_feasible = all(v_u .>= -1e-8)

    local delta_c::Vector{Float64}
    method_used = method == :projection ? :projection : :slsqp
    if method == :projection
        any(cn isa FunctionConstraint for cn in constraints) && throw(ArgumentError(
            "method = :projection supports floor constraints only (the BM crude fallback); use :slsqp for FunctionConstraints"))
        @warn "constrained_opp: method = :projection is the BM sequence-script fast approximation (zero negative components, re-solve on the rest) — NOT the constrained optimum"
        keep = trues(n_s)
        delta_c = copy(delta_u64)
        for _ in 1:n_s
            neg = findall(k -> keep[k] && delta_c[k] < 0, 1:n_s)
            isempty(neg) && break
            keep[neg] .= false
            if !any(keep)
                delta_c = zeros(Float64, n_s)
                break
            end
            Asub = A64[:, keep]
            sub = _suppress_warnings() do
                _policy_projection(Asub, b64; c=(cterm === nothing ? nothing : c64[keep]), method=:ls)
            end
            delta_c = zeros(Float64, n_s)
            delta_c[keep] = sub.nu
        end
    else
        delta_c = _copp_slsqp(obj, grad!, _violations, constraints, p_syms, P0_64,
                              Rp_64, H, paths_of, n_s,
                              delta0 === nothing ? delta_u64 : Float64.(delta0),
                              multistart, rng)
    end

    # feasibility + diagnostics
    v_c, who = _violations(delta_c)
    if !isempty(v_c) && minimum(v_c) < -1e-6
        iworst = argmin(v_c)
        throw(ArgumentError(
            "constrained_opp: no feasible solution found — most violated: $(who[iworst]) (slack $(round(v_c[iworst], sigdigits=3))). The constraint set may be infeasible (e.g. a floor above every reachable path)."))
    end
    binding = [who[i] for i in eachindex(v_c) if abs(v_c[i]) < 1e-6]
    gL = A64' * (A64 * delta_c + b64) .+ c64
    kkt = if isempty(binding)
        norm(gL)
    else
        bidx = [i for i in eachindex(v_c) if abs(v_c[i]) < 1e-6]
        Aact = _violation_jacobian(delta_c, constraints, p_syms, P0_64, Rp_64, H, paths_of)[bidx, :]
        lam = (Aact * Aact' + 1e-12 * I) \ (Aact * gL)
        norm(gL - Aact' * max.(lam, 0.0))
    end

    delta_T = T.(delta_c)
    dd = nothing
    bands = nothing
    reject = nothing
    n_failed = 0
    if n_sim > 0
        dd, bands, reject, n_failed =
            _copp_simulate(fc, ce, loss, constraints, instrument_path, wz,
                           method_used, delta_T, multistart, levels, n_sim,
                           independent, rng)
    end
    delta_rep = dd === nothing ? delta_T :
                [T(median(@view(dd[k, :]))) for k in 1:n_s]

    res = _opp_package(fc, ce, loss, delta_rep, r_u.delta, r_u.gradient,
                       P0, p_syms, wz, ix_ce, ix_fc, iz_p,
                       dd, bands, reject, n_failed)
    return (result=res, method_used=method_used, binding=binding,
            kkt_residual=T(kkt), warm_start_feasible=warm_feasible)
end

# Jacobian of the stacked constraint values (rows follow _violations order).
function _violation_jacobian(x::Vector{Float64}, constraints, p_syms, P0_64,
                             Rp_64, H, paths_of)
    rows = Vector{Vector{Float64}}()
    for cn in constraints
        if cn isa PathFloorConstraint
            k = findfirst(==(cn.instrument), p_syms)
            for h in intersect(cn.horizons, 1:H)
                push!(rows, Vector{Float64}(Rp_64[k][h, :]))
            end
        else
            for i in 1:cn.n_out
                gi = ForwardDiff.gradient(z -> Float64(cn.g(z, paths_of(z))[i]), x)
                push!(rows, gi)
            end
        end
    end
    return isempty(rows) ? zeros(0, length(x)) : Matrix(reduce(hcat, rows)')
end

# SLSQP core with multistart; returns the best feasible point.
function _copp_slsqp(obj, grad!, _violations, constraints, p_syms, P0_64,
                     Rp_64, H, paths_of, n_s::Int, x0::Vector{Float64},
                     multistart::Int, rng::AbstractRNG)
    starts = [copy(x0)]
    scale = 0.5 * max(norm(x0), 1.0)
    for _ in 2:multistart
        push!(starts, x0 .+ scale .* randn(rng, n_s))
    end
    best = nothing
    best_f = Inf
    for s0 in starts
        opt = NLopt.Opt(:LD_SLSQP, n_s)
        NLopt.min_objective!(opt, (x, g) -> begin
            length(g) > 0 && grad!(g, x)
            obj(x)
        end)
        for cn in constraints
            if cn isa PathFloorConstraint
                k = findfirst(==(cn.instrument), p_syms)
                for h in intersect(cn.horizons, 1:H)
                    let k = k, h = h, fl = Float64(cn.floor)
                        NLopt.inequality_constraint!(opt, (x, g) -> begin
                            if length(g) > 0
                                g .= -@view(Rp_64[k][h, :])
                            end
                            fl - P0_64[k][h] - dot(@view(Rp_64[k][h, :]), x)
                        end, 1e-10)
                    end
                end
            else
                for i in 1:cn.n_out
                    let cn = cn, i = i
                        NLopt.inequality_constraint!(opt, (x, g) -> begin
                            if length(g) > 0
                                g .= -ForwardDiff.gradient(
                                    z -> Float64(cn.g(z, paths_of(z))[i]), x)
                            end
                            -Float64(cn.g(x, paths_of(x))[i])
                        end, 1e-10)
                    end
                end
            end
        end
        NLopt.xtol_rel!(opt, 1e-12)
        NLopt.maxeval!(opt, 3000)
        fval, xopt, ret = try
            NLopt.optimize(opt, s0)
        catch
            continue
        end
        v, _ = _violations(Vector{Float64}(xopt))
        feasible = isempty(v) || minimum(v) >= -1e-6
        if feasible && fval < best_f
            best = Vector{Float64}(xopt)
            best_f = fval
        end
    end
    return best === nothing ? x0 : best
end

# CF-14-style simulation with the constrained solve per draw.
function _copp_simulate(fc::PolicyForecast{T}, ce::PolicyCausalEffects{T},
                        loss::PolicyLoss{T}, constraints, instrument_path, wz,
                        method_used::Symbol, delta_pt::Vector{T},
                        multistart::Int, levels, n_sim::Int,
                        independent::Bool, rng::AbstractRNG) where {T<:AbstractFloat}
    nd_R = n_draws(ce)
    nd_Y = n_draws(fc)
    if independent
        nd_R == 0 && nd_Y == 0 && throw(ArgumentError(
            "n_sim > 0 requires draws on at least one source"))
    else
        (nd_R == nd_Y && nd_R > 0) || throw(ArgumentError(
            "independent = false requires equal positive draw counts, got IRF $nd_R vs forecast $nd_Y"))
        n_sim = nd_R
    end
    n_s = length(delta_pt)
    dd = Matrix{T}(undef, n_s, n_sim)
    keep = falses(n_sim)
    _suppress_warnings() do
        for s in 1:n_sim
            dR = independent ? (nd_R > 0 ? rand(rng, 1:nd_R) : 0) : s
            dY = independent ? (nd_Y > 0 ? rand(rng, 1:nd_Y) : 0) : s
            ce_s = dR == 0 ? ce :
                   PolicyCausalEffects{T}(ce.outcomes, ce.instruments,
                                          [Matrix{T}(ce.Theta_x_draws[i][:, :, dR]) for i in eachindex(ce.outcomes)],
                                          [Matrix{T}(ce.Theta_z_draws[k][:, :, dR]) for k in eachindex(ce.instruments)],
                                          nothing, nothing, ce.H, ce.shock_labels, ce.source)
            fc_s = dY == 0 ? fc :
                   PolicyForecast{T}(fc.outcomes,
                                     [Vector{T}(fc.draws[i][:, dY]) for i in eachindex(fc.outcomes)],
                                     nothing, fc.H, fc.origin)
            try
                out = constrained_opp(fc_s, ce_s, loss, constraints;
                                      instrument_path=instrument_path,
                                      z_wedge=wz,
                                      method=(method_used == :projection ? :projection : :slsqp),
                                      delta0=delta_pt, multistart=multistart,
                                      rng=rng, n_sim=0)
                if all(isfinite, out.result.delta)
                    dd[:, s] = out.result.delta
                    keep[s] = true
                end
            catch
                keep[s] = false
            end
        end
    end
    used = findall(keep)
    n_failed = n_sim - length(used)
    n_failed > 0 && @warn "constrained_opp: $n_failed of $n_sim simulation draws failed and were dropped"
    isempty(used) && throw(ArgumentError(
        "constrained_opp: every simulation draw failed"))
    delta_draws = dd[:, used]
    qb = Dict{T,Matrix{T}}()
    rj = Dict{T,Vector{Bool}}()
    for l in levels
        lv = T(l)
        B = Matrix{T}(undef, n_s, 2)
        r = Vector{Bool}(undef, n_s)
        for k in 1:n_s
            lo = T(quantile(@view(delta_draws[k, :]), (1 - lv) / 2))
            hi = T(quantile(@view(delta_draws[k, :]), (1 + lv) / 2))
            B[k, 1] = lo
            B[k, 2] = hi
            r[k] = !(lo <= zero(T) <= hi)
        end
        qb[lv] = B
        rj[lv] = r
    end
    return delta_draws, qb, rj, n_failed
end
