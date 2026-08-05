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

    # blocks: loss outcomes (forecast base), then penalized instruments
    Th = Vector{Matrix{T}}()
    Wb = Vector{Matrix{T}}()
    Bb = Vector{Vector{T}}()
    for (j, sym) in enumerate(loss.outcomes)
        push!(Th, ce.Theta_x[findfirst(==(sym), ce.outcomes)])
        push!(Wb, loss.W_x[j])
        push!(Bb, Vector{T}(fc.values[findfirst(==(sym), fc.outcomes)]))
    end
    if loss.W_z !== nothing
        for (j, sym) in enumerate(loss.instruments)
            push!(Th, ce.Theta_z[findfirst(==(sym), ce.instruments)])
            push!(Wb, loss.W_z[j])
            push!(Bb, P_base[findfirst(==(sym), p_syms)])
        end
    end
    c = nothing
    if wz !== nothing
        cv = zeros(T, n_shocks(ce))
        for (j, sym) in enumerate(loss.instruments)
            cv .-= ce.Theta_z[findfirst(==(sym), ce.instruments)]' * wz[j]
        end
        c = cv
    end

    res = _optimal_projection(Th, Wb, Bb, c)
    delta = res.nu

    # FOC statistic at the ANNOUNCED policy: R'W·base (+ linear term)
    g = zeros(T, n_shocks(ce))
    for j in eachindex(Th)
        g .+= Th[j]' * (Wb[j] * Bb[j])
    end
    c !== nothing && (g .+= c)

    # paths and loss accounting
    Y_base = [Vector{T}(fc.values[findfirst(==(sym), fc.outcomes)]) for sym in loss.outcomes]
    Y_opp = [Y_base[j] + ce.Theta_x[findfirst(==(sym), ce.outcomes)] * delta
             for (j, sym) in enumerate(loss.outcomes)]
    P_opp = nothing
    if P_base !== nothing
        P_opp = [P_base[k] + ce.Theta_z[findfirst(==(s), ce.instruments)] * delta
                 for (k, s) in enumerate(p_syms)]
    end
    lz_base = use_z ? [P_base[findfirst(==(s), p_syms)] for s in loss.instruments] : Vector{T}[]
    lz_opp = use_z ? [P_opp[findfirst(==(s), p_syms)] for s in loss.instruments] : Vector{T}[]
    L_base = _policy_loss_value(loss, Y_base, lz_base; z_wedge=wz)
    L_opp = _policy_loss_value(loss, Y_opp, lz_opp; z_wedge=wz)
    L_opp > L_base + sqrt(eps(T)) * max(one(T), abs(L_base)) &&
        @warn "opp: loss increased ($(L_base) -> $(L_opp)) — this indicates a kernel/sign bug, please report"

    OPPResult{T}(delta, copy(ce.shock_labels), g, L_base, L_opp,
                 Y_base, Y_opp, P_base, P_opp,
                 copy(loss.outcomes), copy(p_syms), H, fc.origin,
                 nothing, nothing, nothing)
end
