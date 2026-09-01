# Counterfactual module — OPP sequences, time consistency, λ sensitivity
# (CF-16, #396)
#
# Running the OPP at every decision date produces a policy-evaluation history
# (BM §4.4, Fig. 1). Revisions decompose EXACTLY (finite-H form) into
#     δ_t − δ_{t−1} = news + pref + aging,
#     news_t  = D_t·(EY_t − EY_{t−1→t})          (forecast revision)
#     pref_t  = (D_t − D_{t−1})·EY_{t−1→t}       (operator/preference change)
#     aging_t = D_{t−1}·(EY_{t−1→t} − EY_{t−1})  (mechanical horizon roll)
# with D = −(R'WR)⁻¹R'W and EY_{t−1→t} the date-(t−1) forecast shifted one
# period forward onto date-t's calendar (the classic off-by-one). BM's
# eq. (32) has two terms because their stationary infinite-horizon operator
# absorbs the aging term; under truncation it is real, so we surface it
# (deviation from the issue's two-term statement — the identity is exact).
# The time-consistent OPP revises on news (and aging) only:
# δ_tc = δ − pref. BM find time-inconsistency empirically negligible.

# Date-(t−1) forecast shifted one period forward onto date-t's calendar:
# drop the first period, append 0 (values only — draws are not needed).
function _shift_forecast(fc::PolicyForecast{T}) where {T<:AbstractFloat}
    vals = [vcat(v[2:end], zero(T)) for v in fc.values]
    PolicyForecast{T}(copy(fc.outcomes), vals, nothing, fc.H, fc.origin)
end

# The linear OPP operator D = −(M)⁻¹K on the stacked loss-outcome forecast,
# with M = ΣΘ'WΘ and K = [Θ_1'W_1 … Θ_n'W_n]. Instrument-penalty terms are
# date-constant offsets and drop out of revision differences; the
# decomposition is defined on the forecast part only.
function _opp_operator(ce::PolicyCausalEffects{T}, loss::PolicyLoss{T}) where {T<:AbstractFloat}
    H = ce.H
    n_s = n_shocks(ce)
    M = zeros(T, n_s, n_s)
    Ks = Vector{Matrix{T}}()
    for (j, sym) in enumerate(loss.outcomes)
        Th = ce.Theta_x[findfirst(==(sym), ce.outcomes)]
        TW = Th' * loss.W_x[j]
        M .+= TW * Th
        push!(Ks, TW)
    end
    K = reduce(hcat, Ks)
    return -Matrix{T}(robust_inv(Hermitian(M))) * K
end

_stack_loss_forecast(fc::PolicyForecast{T}, loss::PolicyLoss{T}) where {T} =
    reduce(vcat, (Vector{T}(fc.values[findfirst(==(s), fc.outcomes)]) for s in loss.outcomes))

"""
    opp_sequence(forecasts, ce, loss; dates=nothing, ce_by_date=nothing,
                 instrument_paths=nothing, constraints=OPPConstraint[],
                 z_wedge=nothing, n_sim=0, levels=(0.6, 0.75, 0.9),
                 independent=true, rng=Random.default_rng()) -> OPPSequence

Run the OPP at every decision date (BM §4.4). `forecasts` is a vector of
[`PolicyForecast`](@ref)s (entries may be `missing` — real forecast panels
have gaps; missing/failed dates become `NaN` columns with a summary warning).
`ce_by_date` supplies per-date IRF containers (default: the single `ce`, the
BM convention); `instrument_paths` per-date `instrument_path` vectors;
nonempty `constraints` route each date through [`constrained_opp`](@ref);
`n_sim > 0` adds CF-14 bands per date (quantiles stored, not raw draws).

The revision decomposition (exact, three terms — see the file header) is
computed from the plug-in linear operator `D_t`; with constraints active it
describes the *unconstrained* revisions. `delta_tc = delta − pref_part`.
"""
function opp_sequence(forecasts::AbstractVector, ce::PolicyCausalEffects{T},
                      loss::PolicyLoss{T};
                      dates::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                      ce_by_date::Union{Nothing,AbstractVector}=nothing,
                      instrument_paths::Union{Nothing,AbstractVector}=nothing,
                      constraints::AbstractVector{<:OPPConstraint}=OPPConstraint[],
                      z_wedge::Union{Nothing,AbstractVector{<:AbstractVector{<:Real}}}=nothing,
                      n_sim::Int=0,
                      levels::Union{Tuple,AbstractVector}=(0.6, 0.75, 0.9),
                      independent::Bool=true,
                      seed::Union{Integer,Nothing}=nothing,
                      rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    n_d = length(forecasts)
    n_d >= 1 || throw(ArgumentError("forecasts: expected at least one date"))
    dts = dates === nothing ? [string("t", t) for t in 1:n_d] : String.(collect(dates))
    length(dts) == n_d || throw(ArgumentError(
        "dates: expected $n_d labels, got $(length(dts))"))
    ce_by_date === nothing || length(ce_by_date) == n_d || throw(ArgumentError(
        "ce_by_date: expected $n_d containers, got $(length(ce_by_date))"))
    instrument_paths === nothing || length(instrument_paths) == n_d || throw(ArgumentError(
        "instrument_paths: expected $n_d entries, got $(length(instrument_paths))"))

    n_s = n_shocks(ce)
    delta = fill(T(NaN), n_s, n_d)
    qlev = collect(T, levels)
    bands = n_sim > 0 ? Dict{T,Array{T,3}}(l => fill(T(NaN), n_s, 2, n_d) for l in qlev) : nothing
    reject = n_sim > 0 ? Dict{T,Matrix{Bool}}(l => falses(n_s, n_d) for l in qlev) : nothing
    n_skipped = 0
    for t in 1:n_d
        fc_t = forecasts[t]
        if fc_t === missing || fc_t === nothing
            n_skipped += 1
            continue
        end
        ce_t = ce_by_date === nothing ? ce : ce_by_date[t]
        ip_t = instrument_paths === nothing ? nothing : instrument_paths[t]
        try
            r = _suppress_warnings() do
                if !isempty(constraints)
                    constrained_opp(fc_t, ce_t, loss, constraints;
                                    instrument_path=ip_t, z_wedge=z_wedge,
                                    n_sim=n_sim, levels=levels,
                                    independent=independent, rng=rng).result
                elseif n_sim > 0
                    estimate_opp(fc_t, ce_t, loss; instrument_path=ip_t,
                                 z_wedge=z_wedge, levels=levels, n_sim=n_sim,
                                 independent=independent, rng=rng)
                else
                    opp(fc_t, ce_t, loss; instrument_path=ip_t, z_wedge=z_wedge)
                end
            end
            delta[:, t] = r.delta
            if n_sim > 0 && r.bands !== nothing
                for l in qlev
                    bands[l][:, :, t] = r.bands[l]
                    reject[l][:, t] = r.reject[l]
                end
            end
        catch
            n_skipped += 1
        end
    end
    n_skipped > 0 && @warn "opp_sequence: $n_skipped of $n_d dates skipped (missing forecasts or failed solves) — NaN columns"

    # exact three-part revision decomposition on the plug-in operator
    news = zeros(T, n_s, n_d)
    pref = zeros(T, n_s, n_d)
    aging = zeros(T, n_s, n_d)
    D_prev = nothing
    fc_prev = nothing
    for t in 1:n_d
        fc_t = forecasts[t]
        if fc_t === missing || fc_t === nothing
            D_prev = nothing
            fc_prev = nothing
            continue
        end
        ce_t = ce_by_date === nothing ? ce : ce_by_date[t]
        D_t = _opp_operator(ce_t, loss)
        if D_prev !== nothing
            ey_t = _stack_loss_forecast(fc_t, loss)
            ey_prev = _stack_loss_forecast(fc_prev, loss)
            ey_shift = _stack_loss_forecast(_shift_forecast(fc_prev), loss)
            news[:, t] = D_t * (ey_t - ey_shift)
            pref[:, t] = (D_t - D_prev) * ey_shift
            aging[:, t] = D_prev * (ey_shift - ey_prev)
        end
        D_prev = D_t
        fc_prev = fc_t
    end
    delta_tc = delta .- pref

    result = OPPSequence{T}(dts, delta, delta_tc, news, pref, aging, bands, reject,
                            copy(ce.shock_labels), loss.name)
    return _with_manifest(result, capture_manifest(; seed=seed,
        settings=Dict{String,Any}("n_sim" => n_sim, "independent" => independent)))
end

"""
    opp_sensitivity(forecasts, ce, H; lambda_grid, build_loss, kwargs...)
        -> Vector{OPPSequence}

Preference-weight sensitivity (BM §4.4): rerun the [`opp_sequence`](@ref)
over `lambda_grid`, mapping each `λ` through the user's
`build_loss(λ) -> PolicyLoss` (BM: `λ ∈ 0.2…2` on unemployment). Returns one
sequence per grid point; dates are independent across the grid, so callers
can thread over `lambda_grid` if needed.
"""
function opp_sensitivity(forecasts::AbstractVector, ce::PolicyCausalEffects{T}, H::Int;
                         lambda_grid::AbstractVector{<:Real},
                         build_loss::Function, kwargs...) where {T<:AbstractFloat}
    H == ce.H || throw(ArgumentError("H = $H does not match the container H = $(ce.H)"))
    isempty(lambda_grid) && throw(ArgumentError("lambda_grid: expected at least one value"))
    return [opp_sequence(forecasts, ce, build_loss(l); kwargs...) for l in lambda_grid]
end

"""
    robust_weights(seq_builder, theta_grid) -> (theta_hat, criterion_path)

BM's robust-preferences extraction (web appendix S5): pick the loss weight
that makes past policy look most optimal,
`θ̂ = argmin_θ ‖(1/n)Σ_t δ_t(θ)‖²`. `seq_builder(θ)` must return an
[`OPPSequence`](@ref) (or a `n_s × n_dates` δ-matrix); `NaN` columns are
skipped. Returns the argmin and the criterion evaluated on the grid.
"""
function robust_weights(seq_builder::Function, theta_grid::AbstractVector{<:Real})
    isempty(theta_grid) && throw(ArgumentError("theta_grid: expected at least one value"))
    crit = Vector{Float64}(undef, length(theta_grid))
    for (i, th) in enumerate(theta_grid)
        out = seq_builder(th)
        dl = out isa OPPSequence ? out.delta : out
        ok = [t for t in 1:size(dl, 2) if all(isfinite, @view(dl[:, t]))]
        isempty(ok) && throw(ArgumentError(
            "robust_weights: seq_builder($(th)) produced no valid dates"))
        m = vec(sum(dl[:, ok]; dims=2)) ./ length(ok)
        crit[i] = sum(abs2, m)
    end
    return (theta_hat=theta_grid[argmin(crit)], criterion_path=crit)
end
