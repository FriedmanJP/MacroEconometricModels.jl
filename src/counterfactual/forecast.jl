# Counterfactual module — forecast adapters (CF-05, #385)
#
# The Barnichon–Mesters OPP consumes the baseline forecast of objective GAPS
# E_t Y_t⁰ with uncertainty. Two sources: (a) package VAR/BVAR forecasts
# (draws retained via forecast(...; store_draws=true) on the BVAR side; the
# VAR bootstrap path always stores them, #524), and (b) external SEP/Greenbook
# style point + dispersion forecasts with AR(1)-correlated simulated errors.
# Unconditional forecasts only — conditional-on-policy-path forecasts are
# CF-18's forecast-revision machinery.

# Resolve a per-outcome target (scalar or length-H path) into a path.
function _pf_target_path(::Type{T}, tgt, H::Int, sym::Symbol) where {T<:AbstractFloat}
    tgt isa Real && return fill(T(tgt), H)
    tgt isa AbstractVector || throw(ArgumentError(
        "targets[:$sym]: expected a Real or a length-$H vector, got $(typeof(tgt))"))
    length(tgt) == H || throw(ArgumentError(
        "targets[:$sym]: expected length H = $H, got $(length(tgt))"))
    return Vector{T}(tgt)
end

"""
    policy_forecast(fc::Union{VARForecast,BVARForecast}, outcomes;
                    targets=[], H=fc.horizon, origin="")

Build a [`PolicyForecast`](@ref) of objective **gaps** from a package forecast.

- `outcomes`: `Pair{Symbol,Union{Int,String}}` maps from gap symbol to forecast
  column (as in [`policy_causal_effects`](@ref)).
- `targets`: per-outcome targets subtracted from the forecast to produce gaps —
  scalars or length-`H` paths, e.g. `[:infl => 2.0, :ygap => 0.0]` (omitted
  outcomes default to a zero target).

!!! warning "Gaps, not levels"
    Forecasts must enter as **gaps from target** — otherwise the OPP drives
    *levels* to zero. And the forecast must be conditional on the **baseline**
    policy rule: if OPP recommendations are adopted repeatedly, each subsequent
    forecast must still be constructed under the *old* rule (Barnichon–Mesters
    S0.5).

Forecast draws are carried through (gap-transformed) when present:
`forecast(::VARModel; ci_method=:bootstrap)` always stores them;
`forecast(::BVARPosterior)` stores them when called with `store_draws=true`.
"""
function policy_forecast(fc::Union{VARForecast{T},BVARForecast{T}},
                         outcomes::AbstractVector{<:Pair};
                         targets::AbstractVector{<:Pair}=Pair{Symbol,Float64}[],
                         H::Int=fc.horizon,
                         origin::AbstractString="") where {T<:AbstractFloat}
    _cf_check_horizon(H, fc.horizon; what="forecast")
    isempty(outcomes) && throw(ArgumentError("outcomes: expected at least one outcome"))
    out_syms = Symbol[first(p) for p in outcomes]
    out_idx = Int[_cf_resolve(last(p), fc.varnames, "variable") for p in outcomes]
    tgt_map = Dict{Symbol,Any}(first(p) => last(p) for p in targets)
    for k in keys(tgt_map)
        k in out_syms || throw(ArgumentError(
            "targets: :$k is not among the requested outcomes $(out_syms)"))
    end

    values = Vector{Vector{T}}(undef, length(out_syms))
    draws = fc._draws === nothing ? nothing : Vector{Matrix{T}}(undef, length(out_syms))
    for (i, (sym, vi)) in enumerate(zip(out_syms, out_idx))
        tgt = _pf_target_path(T, get(tgt_map, sym, zero(T)), H, sym)
        values[i] = Vector{T}(fc.forecast[1:H, vi]) .- tgt
        if fc._draws !== nothing
            D = permutedims(fc._draws[:, 1:H, vi], (2, 1))   # (H, n_draws)
            draws[i] = D .- tgt
        end
    end
    PolicyForecast{T}(out_syms, values, draws, H, String(origin))
end

"""
    policy_forecast(outcomes, values; sd, rho=0.9, n_draws=1000,
                    rng=Random.default_rng(), H=..., cross_corr=:independent,
                    min_sd=0.0, origin="")

External (SEP/Greenbook-style) route: point **gap** forecasts plus dispersion.

- `values::Vector{<:Vector}`: one length-`H` point gap path per outcome.
- `sd::Vector{<:Vector}`: matching per-horizon standard deviations.
- Draws are simulated from `N(values[i], Σᵢ)` with the Barnichon–Mesters
  cross-horizon damping `Σᵢ[j,k] = sd[j]·sd[k]·rho^{|j−k|}`; outcomes are
  independent under `cross_corr = :independent` (BM default). Pass a full
  `(n_x·H) × (n_x·H)` covariance (variable-major block order = `outcomes`
  order) as `cross_corr` to override the constructed one — `sd`/`rho` are then
  ignored.
- `min_sd` floors the dispersions and warns when it binds — degenerate inputs
  (e.g. long-run inflation pinned at target with zero sd) should be filled by
  the caller.

Gaps-not-levels and baseline-rule warnings as in the package-forecast method.
"""
function policy_forecast(outcomes::AbstractVector{Symbol},
                         values::AbstractVector{<:AbstractVector{<:Real}};
                         sd::Union{Nothing,AbstractVector{<:AbstractVector{<:Real}}}=nothing,
                         rho::Real=0.9,
                         n_draws::Int=1000,
                         rng::AbstractRNG=Random.default_rng(),
                         H::Int=isempty(values) ? 0 : length(first(values)),
                         cross_corr::Union{Symbol,AbstractMatrix}=:independent,
                         min_sd::Real=0.0,
                         origin::AbstractString="")
    isempty(outcomes) && throw(ArgumentError("outcomes: expected at least one outcome"))
    length(values) == length(outcomes) || throw(ArgumentError(
        "values: expected $(length(outcomes)) paths (one per outcome), got $(length(values))"))
    n_draws >= 1 || throw(ArgumentError("n_draws: expected n_draws >= 1, got $n_draws"))
    (cross_corr isa Symbol && cross_corr != :independent) && throw(ArgumentError(
        "cross_corr: expected :independent or a covariance matrix, got :$cross_corr"))
    T = float(promote_type(mapreduce(eltype, promote_type, values), typeof(rho)))
    n_x = length(outcomes)
    vals = [Vector{T}(v) for v in values]
    for (i, v) in enumerate(vals)
        length(v) == H || throw(ArgumentError(
            "values[$i]: expected length H = $H, got $(length(v))"))
    end

    if cross_corr isa AbstractMatrix
        size(cross_corr) == (n_x * H, n_x * H) || throw(ArgumentError(
            "cross_corr: expected size ($(n_x * H), $(n_x * H)) (variable-major stacking), got $(size(cross_corr))"))
        C = safe_cholesky(Matrix{T}(cross_corr))
        draws = [Matrix{T}(undef, H, n_draws) for _ in 1:n_x]
        stacked_mean = reduce(vcat, vals)
        for d in 1:n_draws
            y = stacked_mean + C * randn(rng, T, n_x * H)
            for i in 1:n_x
                draws[i][:, d] = y[(i-1)*H+1:i*H]
            end
        end
        return PolicyForecast{T}(collect(Symbol, outcomes), vals, draws, H, String(origin))
    end

    sd === nothing && throw(ArgumentError(
        "sd: per-horizon standard deviations are required unless a full cross_corr covariance is given"))
    length(sd) == n_x || throw(ArgumentError(
        "sd: expected $n_x paths (one per outcome), got $(length(sd))"))
    sds = [Vector{T}(s) for s in sd]
    for (i, s) in enumerate(sds)
        length(s) == H || throw(ArgumentError(
            "sd[$i]: expected length H = $H, got $(length(s))"))
        any(s .< 0) && throw(ArgumentError(
            "sd[$i]: expected nonnegative standard deviations"))
    end
    if min_sd > 0
        n_floored = 0
        for s in sds
            for j in eachindex(s)
                if s[j] < T(min_sd)
                    s[j] = T(min_sd)
                    n_floored += 1
                end
            end
        end
        n_floored > 0 && @warn "policy_forecast: min_sd = $min_sd floored $n_floored dispersion entries (degenerate SEP-style inputs — consider supplying a positive sd instead)"
    end

    draws = [Matrix{T}(undef, H, n_draws) for _ in 1:n_x]
    for i in 1:n_x
        Sigma_i = [sds[i][j] * sds[i][k] * T(rho)^abs(j - k) for j in 1:H, k in 1:H]
        C = safe_cholesky(Sigma_i)
        for d in 1:n_draws
            draws[i][:, d] = vals[i] + C * randn(rng, T, H)
        end
    end
    PolicyForecast{T}(collect(Symbol, outcomes), vals, draws, H, String(origin))
end

"""
    interp_to_quarterly(annual, H) -> Vector{Float64}

Linear interpolation of annual forecast points onto a quarterly grid, endpoints
held (the role of Barnichon–Mesters' `naninterp`): annual value `k` is anchored
at quarter `4k`, quarters before the first anchor hold the first value,
quarters past the last anchor hold the last. Cosmetic convenience for
SEP-style inputs — no statistical content.
"""
function interp_to_quarterly(annual::AbstractVector{<:Real}, H::Int)
    isempty(annual) && throw(ArgumentError("annual: expected at least one point"))
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    a = Float64.(annual)
    out = Vector{Float64}(undef, H)
    for q in 1:H
        t = q / 4                          # anchor positions: year k at quarter 4k
        if t <= 1
            out[q] = a[1]                  # endpoints held
        elseif t >= length(a)
            out[q] = a[end]
        else
            k0 = floor(Int, t)
            w = t - k0
            out[q] = (1 - w) * a[k0] + w * a[k0+1]
        end
    end
    return out
end
