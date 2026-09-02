# Counterfactual module — Barnichon-Mesters OPP statistic (CF-13, #393)
#
# The Optimal Policy Perturbation evaluates whether an announced policy path
# is optimal (BM 2023, AER, eq. 24): with the gap forecast E_tY⁰ and the
# policy-shock IRFs R_y as the two sufficient statistics,
#     δ* = −(R_y'W R_y)⁻¹ R_y'W E_tY⁰
# — literally MINUS the WLS regression coefficient of the forecast on the
# IRFs (dropping the sign WORSENS the loss). The empirically feasible form is
# the subset OPP (thin R_y): a nonzero subset-OPP still rejects optimality
# (BM Corollary 1). This IS CF-11's optimal-policy projection with a forecast
# in place of a shock-IRF baseline — one implementation (_optimal_projection),
# two entry points; the identity is asserted in the CF-23 oracle suite.

"""
    opp(fc::PolicyForecast, ce::PolicyCausalEffects, loss::PolicyLoss;
        instrument_path=nothing, z_wedge=nothing) -> OPPResult

Barnichon–Mesters OPP: the recommended policy perturbation

`δ* = −(R_y'W R_y)⁻¹ R_y'W·E_tY⁰`

with `R_y` the variable-major stack of the loss outcomes' policy-shock IRFs
and `E_tY⁰` the stacked **gap** forecast (gaps-not-levels is enforced by the
[`PolicyForecast`](@ref) type — CF-05's container is gap-typed by
construction). Optimality of the announced policy ⟺ `gradient = R_y'W·E_tY⁰
= 0` ⟺ `δ* = 0` — the model-free targeting rule "at the optimum, forecasts
are orthogonal to IRFs".

- `instrument_path`: announced `E_tP⁰` paths as `Pair`s keyed by instrument
  symbol (e.g. `[:rate => path]`) — required for the *recommendation*
  `P_opp = P_base + R_p·δ*` and for instrument penalties, NOT for the test
  (instruments may be absent entirely).
- Instrument penalties (`loss.W_z`, `z_wedge`) are supported because CMW's
  smoothing loss uses them; the plain BM loss does not penalize the
  instrument.

# Interpretation limits (BM)
- `δ_k` is the perturbation in the direction of identified shock `k`, in that
  shock's instrument units; the subset OPP improves but need not reach the
  full optimum.
- The OPP cannot separate a bad rule from exogenous mistakes (BM eq. 26).
- Lucas caveat: the *improvement* claim requires rule coefficients not to
  feed the non-policy block (breaks under regime-switching credibility);
  the *detection* of non-optimality is robust more broadly (BM S2).
"""
function opp(fc::PolicyForecast{T}, ce::PolicyCausalEffects{T}, loss::PolicyLoss{T};
             instrument_path::Union{Nothing,AbstractVector{<:Pair}}=nothing,
             z_wedge::Union{Nothing,AbstractVector{<:AbstractVector{<:Real}}}=nothing) where {T<:AbstractFloat}
    H = ce.H
    fc.H == H || throw(ArgumentError(
        "forecast H = $(fc.H) does not match the IRF container H = $H; re-build the two on a common horizon"))
    _loss_horizon(loss) == H || throw(ArgumentError(
        "loss H = $(_loss_horizon(loss)) does not match H = $H"))
    for sym in loss.outcomes
        sym in ce.outcomes || throw(ArgumentError(
            "loss outcome :$sym not found in the container outcomes $(ce.outcomes)"))
        sym in fc.outcomes || throw(ArgumentError(
            "loss outcome :$sym not found in the forecast outcomes $(fc.outcomes)"))
    end
    is_square(ce) && @warn "opp: the container is square (n_s = H); the full-menu OPP is ill-posed as columns grow — BM design the method around thin shock subsets (select a subset of identified shocks)"

    # announced instrument paths, keyed by symbol
    P_base = nothing
    p_syms = Symbol[]
    if instrument_path !== nothing
        p_syms = Symbol[first(p) for p in instrument_path]
        for s in p_syms
            s in ce.instruments || throw(ArgumentError(
                "instrument_path: :$s not found in the container instruments $(ce.instruments)"))
        end
        P_base = [Vector{T}(last(p)) for p in instrument_path]
        for (s, v) in zip(p_syms, P_base)
            length(v) == H || throw(ArgumentError(
                "instrument_path[:$s]: expected length H = $H, got $(length(v))"))
        end
    end
    use_z = loss.W_z !== nothing || z_wedge !== nothing
    if use_z
        isempty(loss.instruments) && throw(ArgumentError(
            "loss has instrument penalties (W_z / z_wedge) but no instruments"))
        instrument_path === nothing && throw(ArgumentError(
            "instrument penalties require the announced instrument paths (instrument_path)"))
        for s in loss.instruments
            s in p_syms || throw(ArgumentError(
                "loss instrument :$s has no announced path in instrument_path"))
            s in ce.instruments || throw(ArgumentError(
                "loss instrument :$s not found in the container instruments $(ce.instruments)"))
        end
        z_wedge === nothing || length(z_wedge) == length(loss.instruments) || throw(ArgumentError(
            "z_wedge: expected $(length(loss.instruments)) vectors (one per loss instrument), got $(length(z_wedge))"))
    end
    wz = z_wedge === nothing ? nothing : [Vector{T}(w) for w in z_wedge]

    ix_ce = [findfirst(==(s), ce.outcomes) for s in loss.outcomes]
    ix_fc = [findfirst(==(s), fc.outcomes) for s in loss.outcomes]
    iz_ce = use_z ? [findfirst(==(s), ce.instruments) for s in loss.instruments] : Int[]
    iz_p = use_z ? [findfirst(==(s), p_syms) for s in loss.instruments] : Int[]

    Tx0 = [ce.Theta_x[i] for i in ix_ce]
    Bx0 = [Vector{T}(fc.values[i]) for i in ix_fc]
    Tz0 = use_z ? [ce.Theta_z[k] for k in iz_ce] : Matrix{T}[]
    Bz0 = use_z ? [P_base[k] for k in iz_p] : Vector{T}[]

    Th, Wb, Bb, c = _opp_blocks(loss, Tx0, Bx0, Tz0, Bz0, wz)
    res = _optimal_projection(Th, Wb, Bb, c)
    delta = res.nu

    # FOC statistic at the ANNOUNCED policy: R'W·base (+ linear term)
    g = _opp_gradient(Th, Wb, Bb, c)

    out = _opp_package(fc, ce, loss, delta, delta, g, P_base, p_syms, wz,
                       ix_ce, ix_fc, iz_p, nothing, nothing, nothing, 0)
    return out
end

# Assemble the (Theta, W, base, c) blocks of the OPP/optimal-policy solve.
# Tx/Bx aligned to loss.outcomes; Tz/Bz aligned to loss.instruments (only
# consulted when W_z or a wedge is present).
function _opp_blocks(loss::PolicyLoss{T}, Tx::Vector{<:AbstractMatrix{T}},
                     Bx::Vector{<:AbstractVector{T}},
                     Tz::Vector{<:AbstractMatrix{T}},
                     Bz::Vector{<:AbstractVector{T}},
                     wz::Union{Nothing,Vector{Vector{T}}}) where {T<:AbstractFloat}
    Th = Vector{Matrix{T}}(Tx)
    Wb = Vector{Matrix{T}}(loss.W_x)
    Bb = Vector{Vector{T}}(Bx)
    if loss.W_z !== nothing
        append!(Th, Tz)
        append!(Wb, loss.W_z)
        append!(Bb, Bz)
    end
    c = nothing
    if wz !== nothing
        cv = zeros(T, size(Tx[1], 2))
        for j in eachindex(wz)
            cv .-= Tz[j]' * wz[j]
        end
        c = cv
    end
    return Th, Wb, Bb, c
end

function _opp_gradient(Th, Wb, Bb, c)
    g = zeros(eltype(Bb[1]), size(Th[1], 2))
    for j in eachindex(Th)
        g .+= Th[j]' * (Wb[j] * Bb[j])
    end
    c !== nothing && (g .+= c)
    return g
end

# Paths, loss accounting and result packaging shared by opp/estimate_opp.
function _opp_package(fc::PolicyForecast{T}, ce::PolicyCausalEffects{T},
                      loss::PolicyLoss{T}, delta::Vector{T},
                      delta_plugin::Vector{T}, g::Vector{T},
                      P_base, p_syms::Vector{Symbol},
                      wz::Union{Nothing,Vector{Vector{T}}},
                      ix_ce::Vector{Int}, ix_fc::Vector{Int}, iz_p::Vector{Int},
                      delta_draws, bands, reject, n_failed::Int) where {T<:AbstractFloat}
    use_z = loss.W_z !== nothing || wz !== nothing
    Y_base = [Vector{T}(fc.values[i]) for i in ix_fc]
    Y_opp = [Y_base[j] + ce.Theta_x[ix_ce[j]] * delta for j in eachindex(ix_ce)]
    P_opp = nothing
    if P_base !== nothing
        P_opp = [P_base[k] + ce.Theta_z[findfirst(==(s), ce.instruments)] * delta
                 for (k, s) in enumerate(p_syms)]
    end
    lz_base = use_z ? [P_base[k] for k in iz_p] : Vector{T}[]
    lz_opp = use_z ? [P_opp[k] for k in iz_p] : Vector{T}[]
    L_base = _policy_loss_value(loss, Y_base, lz_base; z_wedge=wz)
    L_opp = _policy_loss_value(loss, Y_opp, lz_opp; z_wedge=wz)
    delta == delta_plugin && L_opp > L_base + sqrt(eps(T)) * max(one(T), abs(L_base)) &&
        @warn "opp: loss increased ($(L_base) -> $(L_opp)) — this indicates a kernel/sign bug, please report"
    OPPResult{T}(delta, delta_plugin, copy(ce.shock_labels), g, L_base, L_opp,
                 Y_base, Y_opp, P_base, P_opp,
                 copy(loss.outcomes), copy(p_syms), fc.H, fc.origin,
                 delta_draws, bands, reject, n_failed)
end

"""
    estimate_opp(fc::PolicyForecast, ce::PolicyCausalEffects, loss::PolicyLoss;
                 instrument_path=nothing, z_wedge=nothing, independent=true,
                 levels=(0.60, 0.75, 0.90), n_sim=2000,
                 rng=Random.default_rng()) -> OPPResult

Two-source OPP uncertainty (BM §5, web appendix S0.3–S0.4): simulate the OPP
distribution over IRF estimation uncertainty (`ce` draws) and forecast
uncertainty (`fc` draws).

- `independent = true` (REQUIRED when the two come from different sources —
  the empirical norm): each simulation independently resamples an IRF draw
  and a forecast draw with replacement. A source without draws is held fixed
  (an `@info` names which source the bands then reflect; degenerate SEP
  dispersions are CF-05's `min_sd` job).
- `independent = false`: matched pairing `d = 1…n_draws` for a joint
  posterior; requires equal draw counts.
- Rank-deficient/non-finite simulation draws are skipped and counted in
  `n_failed`.

`delta` becomes the per-dimension draw **median** (BM convention; the plug-in
point is kept in `delta_plugin`); `bands[ℓ]` are equal-tailed `n_s × 2`
quantiles and `reject[ℓ][k] = !(lo ≤ 0 ≤ hi)`.

**Reversed polarity**: the bands are 60/75/90% — a policymaker averse to
running non-optimal policy prefers rejecting at the LOWER level, which
rejects more readily (BM §5.1). Only per-dimension bands are reported — BM
test per shock dimension and report the vector; no joint statistic is
invented here.
"""
function estimate_opp(fc::PolicyForecast{T}, ce::PolicyCausalEffects{T},
                      loss::PolicyLoss{T};
                      instrument_path::Union{Nothing,AbstractVector{<:Pair}}=nothing,
                      z_wedge::Union{Nothing,AbstractVector{<:AbstractVector{<:Real}}}=nothing,
                      independent::Bool=true,
                      levels::Union{Tuple,AbstractVector}=(0.60, 0.75, 0.90),
                      n_sim::Int=2000,
                      seed::Union{Integer,Nothing}=nothing,
                      rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    all(l -> 0 < l < 1, levels) || throw(ArgumentError(
        "levels: expected band levels strictly between 0 and 1, got $levels"))
    n_sim >= 2 || throw(ArgumentError("n_sim: expected >= 2, got $n_sim"))

    # plug-in point (also runs all CF-13 validation)
    r0 = _suppress_warnings() do
        opp(fc, ce, loss; instrument_path=instrument_path, z_wedge=z_wedge)
    end
    is_square(ce) && @warn "estimate_opp: the container is square (n_s = H); BM design the OPP around thin shock subsets"

    nd_R = n_draws(ce)
    nd_Y = n_draws(fc)
    if independent
        nd_R == 0 && nd_Y == 0 && throw(ArgumentError(
            "estimate_opp requires draws on at least one source (IRF container or forecast)"))
        nd_R == 0 && @info "estimate_opp: the IRF container has no draws — bands reflect forecast uncertainty only"
        nd_Y == 0 && @info "estimate_opp: the forecast has no draws — bands reflect IRF estimation uncertainty only"
    else
        (nd_R == nd_Y && nd_R > 0) || throw(ArgumentError(
            "independent = false pairs IRF draw d with forecast draw d (joint-posterior semantics) and requires equal positive draw counts, got IRF $nd_R vs forecast $nd_Y"))
        n_sim = nd_R
    end

    # index maps (validated inside opp already)
    p_syms = instrument_path === nothing ? Symbol[] :
             Symbol[first(p) for p in instrument_path]
    P_base = instrument_path === nothing ? nothing :
             [Vector{T}(last(p)) for p in instrument_path]
    wz = z_wedge === nothing ? nothing : [Vector{T}(w) for w in z_wedge]
    use_z = loss.W_z !== nothing || wz !== nothing
    ix_ce = [findfirst(==(s), ce.outcomes) for s in loss.outcomes]
    ix_fc = [findfirst(==(s), fc.outcomes) for s in loss.outcomes]
    iz_ce = use_z ? [findfirst(==(s), ce.instruments) for s in loss.instruments] : Int[]
    iz_p = use_z ? [findfirst(==(s), p_syms) for s in loss.instruments] : Int[]
    Bz0 = use_z ? [P_base[k] for k in iz_p] : Vector{T}[]

    Cs = [_pp_weight_factor(W) for W in
          (loss.W_z === nothing ? loss.W_x : vcat(loss.W_x, loss.W_z))]

    n_s = n_shocks(ce)
    dd = Matrix{T}(undef, n_s, n_sim)
    keep = falses(n_sim)
    _suppress_warnings() do
        for s in 1:n_sim
            dR = independent ? (nd_R > 0 ? rand(rng, 1:nd_R) : 0) : s
            dY = independent ? (nd_Y > 0 ? rand(rng, 1:nd_Y) : 0) : s
            Tx_s = dR == 0 ? [ce.Theta_x[i] for i in ix_ce] :
                   [Matrix{T}(ce.Theta_x_draws[i][:, :, dR]) for i in ix_ce]
            Tz_s = !use_z ? Matrix{T}[] :
                   (dR == 0 ? [ce.Theta_z[k] for k in iz_ce] :
                    [Matrix{T}(ce.Theta_z_draws[k][:, :, dR]) for k in iz_ce])
            Bx_s = dY == 0 ? [Vector{T}(fc.values[i]) for i in ix_fc] :
                   [Vector{T}(fc.draws[i][:, dY]) for i in ix_fc]
            try
                Th_s, Wb_s, Bb_s, c_s = _opp_blocks(loss, Tx_s, Bx_s, Tz_s, Bz0, wz)
                r_s = _optimal_projection(Th_s, Wb_s, Bb_s, c_s; Cs=Cs)
                if !r_s.deficient && all(isfinite, r_s.nu)
                    dd[:, s] = r_s.nu
                    keep[s] = true
                end
            catch
                keep[s] = false
            end
        end
    end
    used = findall(keep)
    n_failed = n_sim - length(used)
    n_failed > 0 && @warn "estimate_opp: $n_failed of $n_sim simulation draws failed (rank-deficient or non-finite) and were dropped"
    isempty(used) && throw(ArgumentError(
        "estimate_opp: every simulation draw failed — the shock columns are collinear in all draws"))

    delta_draws = dd[:, used]
    delta_med = [T(median(@view(delta_draws[k, :]))) for k in 1:n_s]
    bands = Dict{T,Matrix{T}}()
    reject = Dict{T,Vector{Bool}}()
    for l in levels
        lv = T(l)
        B = Matrix{T}(undef, n_s, 2)
        rj = Vector{Bool}(undef, n_s)
        for k in 1:n_s
            lo = T(quantile(@view(delta_draws[k, :]), (1 - lv) / 2))
            hi = T(quantile(@view(delta_draws[k, :]), (1 + lv) / 2))
            B[k, 1] = lo
            B[k, 2] = hi
            rj[k] = !(lo <= zero(T) <= hi)
        end
        bands[lv] = B
        reject[lv] = rj
    end

    result = _opp_package(fc, ce, loss, delta_med, r0.delta_plugin, r0.gradient,
                 P_base, p_syms, wz, ix_ce, ix_fc, iz_p,
                 delta_draws, bands, reject, n_failed)
    return _with_manifest(result, capture_manifest(; seed=seed,
        settings=Dict{String,Any}(
            "n_sim" => n_sim, "independent" => independent,
            "levels" => collect(Float64, levels),
            "fc" => fc, "ce" => ce, "loss" => loss)))
end
