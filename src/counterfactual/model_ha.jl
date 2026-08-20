# Counterfactual module — HA sequence-space policy effects (CF-08, #388)
#
# SPEC RECONCILIATION (#388 was written against v0.6.5): v0.7.x already
# shipped the public SSJ block system (HetBlock / SimpleBlock / combine_blocks
# / block_jacobian / ssj_jacobian -> SSJGEJacobian with public H_U/H_Z/
# H_U_fact / ssj_irf). The issue's `_assemble_clearing_system` refactor is
# therefore moot — the Huggett GE closure ALREADY routes through those public
# blocks (#352/T253), and `src/dsge/heterogeneous/` is left untouched here.
# Scope per the issue's own timebox: one-asset only; the production (KS) class
# is NOT GE-closed anywhere in the SSJ path on v0.7.3, so the `:market`
# closure ships for the one GE-closed class (:huggett) and the administered-
# rate closure ships for every one-asset class.
#
# Wedge sign convention (fixed here, tested): `+m` ADDS to the rate households
# face — a positive wedge is a rate HIKE (contractionary); expansionary = −m.

function _cf_require_one_asset(spec::ModelSpec)
    (_hh(spec).grid.n_dims == 1 && _hh(spec).individual.n_asset_dims == 1) || throw(ArgumentError(
        "one-asset heterogeneous-agent models only (got a $(_hh(spec).grid.n_dims)-dimensional asset grid); two-asset models are out of scope — refusing to approximate"))
    return nothing
end

"""
    sequence_jacobian(spec::ModelSpec, ss::HASteadyState, input, output;
                      T_horizon=300, dx=1e-4) -> Matrix

Public sequence-space (fake-news) Jacobian of the heterogeneous household
block: `J[t, s] = ∂O_t/∂I_s` — the response of aggregate output `output` at
date `t` to a unit change of `input` announced at date 1 and occurring at date
`s` (Auclert, Bardóczy, Rognlie & Straub 2021, *Econometrica*). Entries with
`t < s` are **anticipation effects** — `J` is dense, not lower-triangular.

- `input ∈ (:r, :w)`; `output` is any aggregate the household block recognizes
  (`:A`/`:K`/`:assets`, `:C`/`:consumption`, `:N`/`:hours`, …).
- One-asset models only; two-asset specs error.
- `dx` is the numerical-differentiation step of the backward EGM sweep; on
  coarse asset grids a too-small `dx` amplifies interpolation noise — keep it
  of the order of the default (`1e-4`) unless the grid is fine.

Thin wrapper over the block machinery (`HetBlock` + `block_jacobian`), so it
is behavior-identical to the internal `_ssj_jacobian`.
"""
function sequence_jacobian(spec::ModelSpec{T}, ss::HASteadyState{T},
                           input::Symbol, output::Symbol;
                           T_horizon::Int=300, dx::Real=1e-4) where {T<:AbstractFloat}
    _cf_require_one_asset(spec)
    input in (:r, :w) || throw(ArgumentError(
        "input: expected :r or :w, got :$input"))
    T_horizon >= 1 || throw(ArgumentError("T_horizon: expected >= 1, got $T_horizon"))
    b = HetBlock(spec, ss; inputs=[input], outputs=[output], name=:cf_household, dx=T(dx))
    return block_jacobian(b, T_horizon)[(output, input)]
end

"""
    policy_causal_effects(spec::ModelSpec, ss::HASteadyState; outcomes,
                          instruments=[:rate => :r], H=100, T_horizon=300,
                          rule_closure=:administered, dx=1e-4)
        -> PolicyCausalEffects

Square (`n_s = H`) GE news maps of a **rate wedge** `m` for one-asset
heterogeneous-agent models (`source = :hank`): column `s` is the response to a
wedge announced at date 1 and applied at date `s`. Sign convention: `+m` adds
to the rate households face (a hike; expansionary = `−m`).

Two closures (`rule_closure`):
- `:administered` (default, any one-asset class): the policy rate follows the
  wedge one-for-one (`Θ_z = I`) and the intermediary absorbs the induced asset
  flows — the sequence-space analogue of an administered-rate/Taylor-rule
  disturbance. Outcome maps are the household fake-news Jacobians truncated to
  `H`, anticipation effects included.
- `:market` (`:huggett` only — the single GE-closed class in the v0.7.x SSJ
  path): the bond market clears in zero net supply, so the market rate offsets
  the wedge one-for-one (`∂F/∂m = H_U` ⟹ `dr_market = −m`) and the wedge is
  **neutral**: effective-rate and outcome responses are ≈ 0. This exact
  neutrality is the analytic GE-closure check (and the reason a real
  endowment economy is not an interesting policy laboratory).

`T_horizon` is the internal Jacobian length; require `T_horizon ≥ H` and keep
`T_horizon ≥ H + 50` to avoid truncation-edge bias (warned otherwise; same
tradeoff as SSJ estimation, T049).
"""
function policy_causal_effects(spec::ModelSpec{T}, ss::HASteadyState{T};
                               outcomes::AbstractVector{<:Pair{Symbol,Symbol}},
                               instruments::AbstractVector{<:Pair{Symbol,Symbol}}=[:rate => :r],
                               H::Int=100, T_horizon::Int=300,
                               rule_closure::Symbol=:administered,
                               dx::Real=1e-4) where {T<:AbstractFloat}
    _cf_require_one_asset(spec)
    rule_closure in (:administered, :market) || throw(ArgumentError(
        "rule_closure: expected :administered or :market, got :$rule_closure"))
    rule_closure == :market && _hh(spec).model !== :huggett && throw(ArgumentError(
        ":market closure is GE-closed only for the :huggett class (the v0.7.x SSJ path ships no production-side clearing); use rule_closure = :administered"))
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    T_horizon >= H || throw(ArgumentError(
        "T_horizon = $T_horizon must be >= H = $H"))
    T_horizon < H + 50 && @warn "T_horizon = $T_horizon < H + 50 = $(H + 50): truncation-edge bias in the news columns near H — increase T_horizon"
    isempty(outcomes) && throw(ArgumentError("outcomes: expected at least one outcome"))
    for p in instruments
        last(p) == :r || throw(ArgumentError(
            "instruments: only the policy rate :r is supported, got :$(last(p))"))
    end

    out_syms = Symbol[first(p) for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    out_vars = unique([last(p) for p in outcomes])
    b = HetBlock(spec, ss; inputs=[:r], outputs=out_vars, name=:cf_household, dx=T(dx))
    Jd = block_jacobian(b, T_horizon)

    local dRe::Union{Nothing,Matrix{T}} = nothing
    if rule_closure == :market
        household = HetBlock(spec, ss; inputs=[:r, :w], outputs=[:A], name=:household, dx=T(dx))
        bond_market = SimpleBlock(x -> [x[1]]; inputs=[:A], outputs=[:bond_mkt],
                                  ss_inputs=Dict(:A => household.ss_outputs[:A]),
                                  name=:bond_market)
        dag = combine_blocks(household, bond_market; name=:huggett_cf)
        gej = ssj_jacobian(dag; unknowns=[:r], targets=[:bond_mkt], shocks=[:w],
                           T_horizon=T_horizon, target_tol=Inf)
        # F(r_mkt; m) = 0 with households facing r_mkt + m: ∂F/∂m = H_U, so
        # dr_mkt = −H_U⁻¹H_U·m = −m and the faced rate moves dRe = I + dr ≈ 0.
        dMarket = -(gej.H_U_fact \ Matrix{T}(gej.H_U))
        dRe = Matrix{T}(I, T_horizon, T_horizon) + dMarket
    end

    Theta_x = Vector{Matrix{T}}(undef, length(outcomes))
    for (i, p) in enumerate(outcomes)
        J = Jd[(last(p), :r)]
        Theta_x[i] = dRe === nothing ? Matrix{T}(J[1:H, 1:H]) :
                     Matrix{T}((J * dRe)[1:H, 1:H])
    end
    Theta_z = [dRe === nothing ? Matrix{T}(I, H, H) : Matrix{T}(dRe[1:H, 1:H])
               for _ in instruments]

    labels = [s == 1 ? "m (s=0)" : "m news (s=$(s-1))" for s in 1:H]
    PolicyCausalEffects{T}(out_syms, ins_syms, Theta_x, Theta_z,
                           nothing, nothing, H, labels, :hank)
end
