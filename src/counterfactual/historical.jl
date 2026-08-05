# Counterfactual module — historical-evolution and conditional-forecast
# counterfactuals via forecast revisions (CF-18, #398)
#
# CMW (2025) App. A.2(ii)-A.3: the per-date input is the forecast REVISION
# (E_t − E_{t−1})[y_{t+h}] — what date-t news implied — re-solved under the
# alternative policy and rolled forward:  path ← [path[2:end]; 0] + d̃_t.
# This reconstructs the counterfactual without ever identifying individual
# structural shocks; only forecast revisions are needed (this is where
# invertibility / forecast-sufficiency earns its keep). Raw forecasts would
# DOUBLE-COUNT — only revisions enter the per-date solve (CMW subtlety #9);
# the end-to-end linear oracle in the tests catches any violation.

# H-step point forecast path from the last p rows of `hist` under coefficient
# matrix B ((1+np)×n, row 1 = intercept).
function _var_forecast_path(B::AbstractMatrix{T}, hist::AbstractMatrix{T},
                            p::Int, H::Int) where {T<:AbstractFloat}
    n = size(B, 2)
    A = extract_ar_coefficients(Matrix{T}(B), n, p)
    c = Vector{T}(B[1, :])
    buf = Matrix{T}(undef, p + H, n)
    buf[1:p, :] = hist[end-p+1:end, :]
    for h in 1:H
        y = copy(c)
        for l in 1:p
            y .+= A[l] * @view(buf[p+h-l, :])
        end
        buf[p+h, :] = y
    end
    return buf[p+1:end, :]
end

# Forecast-revision matrix at origin t (rows r = 1…H, columns = variables):
# row 1 = y_t − ŷ_{t|t−1} (the period-t innovation content), row r ≥ 2 =
# ŷ_{t+r−1|t} − ŷ_{t+r−1|t−1}. The t−1 forecast is advanced one period so
# horizons refer to the same calendar dates (the classic off-by-one).
# Deterministic components cancel by construction (tested on a trended AR(1)).
function _forecast_revisions(B::AbstractMatrix{T}, data::AbstractMatrix{T},
                             t::Int, p::Int, H::Int) where {T<:AbstractFloat}
    t > p || throw(ArgumentError(
        "forecast revisions need t > p (origin t = $t, lags p = $p)"))
    f_prev = _var_forecast_path(B, @view(data[1:t-1, :]), p, H)   # ŷ_{t-1+r|t-1}
    d = Matrix{T}(undef, H, size(data, 2))
    d[1, :] = data[t, :] - f_prev[1, :]
    if H > 1
        f_now = _var_forecast_path(B, @view(data[1:t, :]), p, H - 1)  # ŷ_{t+h|t}
        for r in 2:H
            d[r, :] = f_now[r-1, :] - f_prev[r, :]
        end
    end
    return d
end

_cf18_point_B(m::VARModel) = m.B
_cf18_point_B(post::BVARPosterior) = dropdims(mean(post.B_draws, dims=1), dims=1)
_cf18_p(m::VARModel) = m.p
_cf18_p(post::BVARPosterior) = post.p
_cf18_varnames(m::VARModel) = m.varnames
_cf18_varnames(post::BVARPosterior) = post.varnames

# Build the revision BaselinePath (+ the carry-over forecast for add-back).
function _revision_baseline(m, data::AbstractMatrix{T}, t::Int, H::Int,
                            outcomes, instruments, label::String) where {T<:AbstractFloat}
    vn = _cf18_varnames(m)
    p = _cf18_p(m)
    B = Matrix{T}(_cf18_point_B(m))
    d = _forecast_revisions(B, data, t, p, H)
    out_syms = Symbol[first(q) for q in outcomes]
    out_idx = Int[_cf_resolve(last(q), vn, "variable") for q in outcomes]
    ins_syms = Symbol[first(q) for q in instruments]
    ins_idx = Int[_cf_resolve(last(q), vn, "variable") for q in instruments]
    base = BaselinePath{T}(out_syms, ins_syms,
                           [Vector{T}(d[:, j]) for j in out_idx],
                           [Vector{T}(d[:, j]) for j in ins_idx],
                           nothing, nothing, H, label)
    carry = _var_forecast_path(B, @view(data[1:t-1, :]), p, H)   # ŷ_{t-1+r|t-1}
    return base, carry, out_idx, ins_idx
end

_cf18_solve(base, ce, policy::PolicyRule; draws, z_wedge) =
    policy_counterfactual(base, ce, policy; draws=draws)
_cf18_solve(base, ce, policy::PolicyLoss; draws, z_wedge) =
    optimal_policy(base, ce, policy; draws=draws, z_wedge=z_wedge)

"""
    counterfactual_forecast(m, data_through_tstar, ce, policy;
                            outcomes, instruments=[], H=ce.H,
                            wedge_init=nothing, draws=:auto,
                            add_back=:forecast) -> PolicyCounterfactual

Conditional-forecast counterfactual at one decision date `t*` (the last row
of `data_through_tstar`): the date-`t*` forecast **revision** is re-solved
under `policy` (a `PolicyRule` via [`policy_counterfactual`](@ref), a
`PolicyLoss` via [`optimal_policy`](@ref)) and, with `add_back = :forecast`,
added back onto the pre-existing trajectory `E_{t*−1}[y]` — CMW's
`E_{t*}[ỹ] = Θ̃·ε_{t*} + ỹ*`. `add_back = :none` returns the pure revision
response. `m` is a `VARModel` or `BVARPosterior` (posterior mean
coefficients; CMW condition on the full-sample estimate and vary only the
conditioning data). `wedge_init` supplies a smoothing-penalty `z_wedge`
vector set (losses only).
"""
function counterfactual_forecast(m::Union{VARModel{T},BVARPosterior{T}},
                                 data_through_tstar::AbstractMatrix{<:Real},
                                 ce::PolicyCausalEffects{T},
                                 policy::Union{PolicyRule{T},PolicyLoss{T}};
                                 outcomes::AbstractVector{<:Pair},
                                 instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[],
                                 H::Int=ce.H,
                                 wedge_init::Union{Nothing,AbstractVector{<:AbstractVector{<:Real}}}=nothing,
                                 draws::Symbol=:auto,
                                 add_back::Symbol=:forecast) where {T<:AbstractFloat}
    add_back in (:forecast, :none) || throw(ArgumentError(
        "add_back: expected :forecast or :none, got :$add_back"))
    H == ce.H || throw(ArgumentError(
        "H = $H must equal the container H = $(ce.H)"))
    data = Matrix{T}(data_through_tstar)
    t = size(data, 1)
    base, carry, out_idx, ins_idx = _revision_baseline(m, data, t, H,
                                                       outcomes, instruments,
                                                       "forecast revision @t*=$t")
    pc = _cf18_solve(base, ce, policy; draws=draws,
                     z_wedge=(policy isa PolicyLoss ? wedge_init : nothing))
    add_back == :none && return pc

    # shift baseline/counterfactual (and bands) by the pre-existing trajectory
    shift_x = [Vector{T}(carry[:, j]) for j in out_idx]
    shift_z = [Vector{T}(carry[:, j]) for j in ins_idx]
    xb = [pc.x_base[i] + shift_x[i] for i in eachindex(shift_x)]
    zb = [pc.z_base[k] + shift_z[k] for k in eachindex(shift_z)]
    x_cf = [pc.x_cf[i] + shift_x[i] for i in eachindex(shift_x)]
    z_cf = [pc.z_cf[k] + shift_z[k] for k in eachindex(shift_z)]
    xbands = pc.x_bands === nothing ? nothing :
             [pc.x_bands[i] .+ shift_x[i] for i in eachindex(shift_x)]
    zbands = pc.z_bands === nothing ? nothing :
             [pc.z_bands[k] .+ shift_z[k] for k in eachindex(shift_z)]
    PolicyCounterfactual{T}(pc.outcomes, pc.instruments, xb, zb, x_cf, z_cf,
                            xbands, zbands, pc.nu, pc.shock_labels,
                            pc.error_path, pc.rel_residual, pc.rel_residual_bands,
                            pc.spanned, pc.rule_name, pc.H, pc.quantile_levels,
                            pc.n_draws_used, pc.n_draws_failed,
                            pc.loss_base, pc.loss_cf, pc.foc_norm)
end

"""
    counterfactual_history(m, data, t_range, ce, policy;
                           outcomes, instruments=[], H=ce.H, dates=nothing,
                           wedge_builder=nothing, draws=:auto,
                           quantiles=(0.16, 0.5, 0.84)) -> CounterfactualHistory

Historical-evolution counterfactual (CMW `get_historical_evol.m`): had
`policy` been in place from `first(t_range)` through `last(t_range)`, what
path would the mapped variables have followed? Per date `t ∈ t_range`: the
forecast revision is computed from the full-sample coefficients, re-solved
under the policy, and rolled forward (`path ← [path[2:end]; 0] + d̃_t`) for
BOTH the raw and the counterfactual revisions; the counterfactual level is
`data_t − rolled_raw₁ + rolled_cf₁`. When the policy equals the
data-generating rule the two rolls coincide and the counterfactual IS the
realized data.

- `t_range` holds integer row indices into `data` (each must exceed the lag
  order); `dates` are optional display labels.
- `wedge_builder`: for smoothing-penalty losses, a closure
  `z_lag -> wedge_term::Vector` (e.g. built from [`smoothing_penalty`](@ref))
  — the per-date `z_lag` is threaded automatically as the PREVIOUS date's
  counterfactual instrument level from the rolled path, never realized data.
  (Deviation from the issue's `wedge_init=:auto` literal: a builder closure
  is required because λ/β cannot be recovered from a general `W_z`.)
- `draws = :auto`: with a draws-bearing `ce`, each container draw is held
  fixed across the whole window (paired accumulation) and quantile bands on
  the counterfactual levels are reported; failed draws are dropped whole and
  counted. Window length must not exceed `H − 1` (the roll would truncate).

ZLB-era wedge adjustments are a documentation-level pattern: use
`wedge_builder` and per-date rule wedges rather than any special-casing.
"""
function counterfactual_history(m::Union{VARModel{T},BVARPosterior{T}},
                                data::AbstractMatrix{<:Real},
                                t_range::AbstractUnitRange{Int},
                                ce::PolicyCausalEffects{T},
                                policy::Union{PolicyRule{T},PolicyLoss{T}};
                                outcomes::AbstractVector{<:Pair},
                                instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[],
                                H::Int=ce.H,
                                dates::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                                wedge_builder::Union{Nothing,Function}=nothing,
                                draws::Symbol=:auto,
                                quantiles::Union{Tuple,AbstractVector}=(0.16, 0.5, 0.84)) where {T<:AbstractFloat}
    H == ce.H || throw(ArgumentError(
        "H = $H must equal the container H = $(ce.H)"))
    dat = Matrix{T}(data)
    n_d = length(t_range)
    n_d >= 1 || throw(ArgumentError("t_range: expected at least one date"))
    last(t_range) <= size(dat, 1) || throw(ArgumentError(
        "t_range exceeds the data length $(size(dat, 1))"))
    n_d <= H - 1 || throw(ArgumentError(
        "window length $(n_d) exceeds H − 1 = $(H - 1): the revision roll would truncate — increase H"))
    dts = dates === nothing ? [string("t", t) for t in t_range] : String.(collect(dates))
    length(dts) == n_d || throw(ArgumentError(
        "dates: expected $n_d labels, got $(length(dts))"))
    wedge_builder !== nothing && !(policy isa PolicyLoss) && throw(ArgumentError(
        "wedge_builder applies to PolicyLoss policies only"))
    (wedge_builder !== nothing && length(ce.instruments) != 1) && throw(ArgumentError(
        "wedge_builder threading requires exactly one instrument"))
    draws in (:auto, :on, :off) || throw(ArgumentError(
        "draws: expected :auto, :on or :off, got :$draws"))
    nd = n_draws(ce)
    use_draws = draws == :on || (draws == :auto && nd > 0)
    use_draws && nd == 0 && throw(ArgumentError(
        "draws = :on requires a draws-bearing container"))

    vars = vcat(ce.outcomes, ce.instruments)
    n_v = length(vars)
    n_x = length(ce.outcomes)
    n_s = n_shocks(ce)
    qlev = collect(T, quantiles)

    # one pass of the recursion for a fixed container (point or one draw)
    function _run(ce_use::PolicyCausalEffects{T}; record::Bool=false)
        rolled_raw = zeros(T, H, n_v)
        rolled_cf = zeros(T, H, n_v)
        cf = Matrix{T}(undef, n_d, n_v)
        realized = Matrix{T}(undef, n_d, n_v)
        nus = record ? Matrix{T}(undef, n_s, n_d) : Matrix{T}(undef, 0, 0)
        rrs = record ? Vector{T}(undef, n_d) : T[]
        z_lag = zero(T)
        out_idx = Int[]
        ins_idx = Int[]
        for (j, t) in enumerate(t_range)
            base, _, oi, ii = _revision_baseline(m, dat, t, H, outcomes,
                                                 instruments, "revision @t=$t")
            out_idx, ins_idx = oi, ii
            zw = wedge_builder === nothing ? nothing : [Vector{T}(wedge_builder(z_lag))]
            pc = _suppress_warnings() do
                _cf18_solve(base, ce_use, policy; draws=:off, z_wedge=zw)
            end
            # roll both accumulators forward and add this date's revisions
            for M in (rolled_raw, rolled_cf)
                M[1:end-1, :] = M[2:end, :]
                M[end, :] .= zero(T)
            end
            for (i, _) in enumerate(ce_use.outcomes)
                rolled_raw[:, i] .+= base.x[i]
                rolled_cf[:, i] .+= pc.x_cf[i]
            end
            for (k, _) in enumerate(ce_use.instruments)
                rolled_raw[:, n_x+k] .+= base.z[k]
                rolled_cf[:, n_x+k] .+= pc.z_cf[k]
            end
            for (v, col) in enumerate(vcat(out_idx, ins_idx))
                realized[j, v] = dat[t, col]
                cf[j, v] = dat[t, col] - rolled_raw[1, v] + rolled_cf[1, v]
            end
            if record
                nus[:, j] = pc.nu
                rrs[j] = pc.rel_residual
            end
            if wedge_builder !== nothing
                z_lag = cf[j, n_x+1]     # previous COUNTERFACTUAL instrument level
            end
        end
        return realized, cf, nus, rrs
    end

    realized, cf, nus, rrs = _run(ce; record=true)

    cf_bands = nothing
    n_used = 0
    n_failed = 0
    if use_draws
        sims = Array{T,3}(undef, n_d, n_v, nd)
        keep = falses(nd)
        for d in 1:nd
            ce_d = PolicyCausalEffects{T}(ce.outcomes, ce.instruments,
                                          [Matrix{T}(ce.Theta_x_draws[i][:, :, d]) for i in eachindex(ce.outcomes)],
                                          [Matrix{T}(ce.Theta_z_draws[k][:, :, d]) for k in eachindex(ce.instruments)],
                                          nothing, nothing, H, ce.shock_labels, ce.source)
            try
                _, cf_d, _, _ = _run(ce_d)
                if all(isfinite, cf_d)
                    sims[:, :, d] = cf_d
                    keep[d] = true
                end
            catch
                keep[d] = false
            end
        end
        used = findall(keep)
        n_used = length(used)
        n_failed = nd - n_used
        n_failed > 0 && @warn "counterfactual_history: $n_failed of $nd draws failed and were dropped"
        if n_used > 0
            cf_bands = Array{T,3}(undef, n_d, n_v, length(qlev))
            for j in 1:n_d, v in 1:n_v, (qi, q) in enumerate(qlev)
                cf_bands[j, v, qi] = quantile(@view(sims[j, v, used]), q)
            end
        end
    end

    CounterfactualHistory{T}(dts, vars, realized, cf, cf_bands, nus, rrs,
                             policy.name, H, qlev, n_used, n_failed)
end
