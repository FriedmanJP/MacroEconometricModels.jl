# Counterfactual module — second-moment (Wold) counterfactuals (CF-12, #392)
#
# "How volatile would the economy have been under rule A-tilde?" Under
# invertibility (more precisely forecast-sufficiency, CF-19), every
# orthonormalized Wold-innovation column is a non-policy baseline that can be
# re-ruled/re-optimized with the CF-10/CF-11 machinery; the counterfactual
# covariance is the VMA sum over re-solved columns (MW §2.3; CMW §5.1). The
# orthogonalizing rotation is immaterial: an orthogonal P cancels in
# Σ_h Θ_h Θ_h' (CMW App. A.2).

# Σ_h Θ[h,:,:]·Θ[h,:,:]' over the whole array.
function _vma_covariance(Theta::AbstractArray{T,3}) where {T<:AbstractFloat}
    H, n, _ = size(Theta)
    S = zeros(T, n, n)
    for h in 1:H
        Th = @view Theta[h, :, :]
        S .+= Th * Th'
    end
    return S
end

# Band-limited VMA covariance: (1/π)·∫_{ω_lo}^{ω_hi} Re(Ψ(e^{-iω})Ψ(e^{-iω})*) dω
# on a midpoint Riemann grid (512 points — simpler than reusing the spectral
# module's grid and exact enough for a smooth integrand; the full band
# [0, π] reproduces the time-domain sum by Parseval).
function _vma_band_covariance(Theta::AbstractArray{T,3}, w_lo::T, w_hi::T;
                              n_grid::Int=512) where {T<:AbstractFloat}
    H, n, _ = size(Theta)
    S = zeros(T, n, n)
    dw = (w_hi - w_lo) / n_grid
    for g in 1:n_grid
        w = w_lo + (g - T(0.5)) * dw
        Psi = zeros(Complex{T}, n, size(Theta, 3))
        for h in 1:H
            Psi .+= @view(Theta[h, :, :]) .* cis(-w * (h - 1))
        end
        S .+= real.(Psi * Psi') .* dw
    end
    return S ./ T(pi)
end

function _cf_resolve_band(frequencies, ::Type{T}) where {T<:AbstractFloat}
    frequencies === :none && return nothing
    if frequencies === :business_cycle
        return (T(2pi / 32), T(2pi / 6))
    end
    frequencies isa Tuple{<:Real,<:Real} || throw(ArgumentError(
        "frequencies: expected :none, :business_cycle or an (ω_lo, ω_hi) tuple, got $frequencies"))
    lo, hi = T(frequencies[1]), T(frequencies[2])
    (0 <= lo < hi <= T(pi)) || throw(ArgumentError(
        "frequencies: expected 0 <= ω_lo < ω_hi <= π, got ($lo, $hi)"))
    return (lo, hi)
end

"""
    counterfactual_moments(wold::WoldRepresentation, ce::PolicyCausalEffects,
                           policy::Union{PolicyRule,PolicyLoss};
                           outcomes, instruments=[], draws=:auto,
                           draw_source=:ce, quantiles=(0.16, 0.5, 0.84),
                           frequencies=:none, warn_invertibility=true)
        -> CounterfactualMoments

Second-moment counterfactual: re-solve every orthonormalized Wold column of
`wold` under the alternative `policy` (a `PolicyRule` via
[`policy_counterfactual`](@ref) or a `PolicyLoss` via
[`optimal_policy`](@ref)) and sum the re-solved VMA:
`Σ^cf = Σ_h Θ̃_h Θ̃_h'`.

- `outcomes`/`instruments`: `Pair` maps from the container's symbols to Wold
  rows (`Int` index or `String` name); they must cover exactly the
  container's variables.
- `frequencies = :business_cycle` band-limits the variances to
  `ω ∈ [2π/32, 2π/6]` (CMW `freq_var_fn`); a custom `(ω_lo, ω_hi)` tuple is
  accepted. Both are computed on a 512-point Riemann grid of the finite-VMA
  spectral density.
- `draws`: `:auto` propagates uncertainty from `draw_source`
  (`:ce`, `:wold`, or `:both` — `:both` pairs the two sources and requires
  equal draw counts), yielding quantile bands on `sd_cf`.
- `tail_share` (last-10-rows share of `‖Θ̃‖`) warns above 1% — the VMA sum has
  not converged and `H` must grow (persistent counterfactual rules push mass
  into the tail; MW use `H = 100`).

!!! warning "Invertibility"
    This object ADDITIONALLY assumes the Wold innovations span the structural
    shocks (or forecast-sufficiency, CF-19). Level counterfactuals (CF-10/11)
    do NOT need this assumption — the two capabilities are deliberately kept
    separate. Suppress the reminder with `warn_invertibility=false`.
"""
function counterfactual_moments(wold::WoldRepresentation{T}, ce::PolicyCausalEffects{T},
                                policy::Union{PolicyRule{T},PolicyLoss{T}};
                                outcomes::AbstractVector{<:Pair},
                                instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[],
                                draws::Symbol=:auto,
                                draw_source::Symbol=:ce,
                                quantiles::Union{Tuple,AbstractVector}=(0.16, 0.5, 0.84),
                                frequencies=:none,
                                warn_invertibility::Bool=true) where {T<:AbstractFloat}
    H = ce.H
    Hw = size(wold.Theta, 1)
    Hw >= H || throw(ArgumentError(
        "Wold horizon $Hw is shorter than the container H = $H; re-run wold_representation with H >= $H"))
    draws in (:auto, :on, :off) || throw(ArgumentError(
        "draws: expected :auto, :on or :off, got :$draws"))
    draw_source in (:ce, :wold, :both) || throw(ArgumentError(
        "draw_source: expected :ce, :wold or :both, got :$draw_source"))
    band = _cf_resolve_band(frequencies, T)

    out_syms = Symbol[first(p) for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    sort(out_syms) == sort(ce.outcomes) || throw(ArgumentError(
        "outcomes must map exactly the container outcomes $(ce.outcomes), got $(out_syms)"))
    sort(ins_syms) == sort(ce.instruments) || throw(ArgumentError(
        "instruments must map exactly the container instruments $(ce.instruments), got $(ins_syms)"))
    row_of = Dict{Symbol,Int}()
    for p in vcat(collect(outcomes), collect(instruments))
        row_of[first(p)] = _cf_resolve(last(p), wold.varnames, "Wold variable")
    end

    warn_invertibility &&
        @warn "counterfactual_moments assumes the Wold innovations span the structural shocks (invertibility / forecast-sufficiency, CF-19); level counterfactuals do not need this. Pass warn_invertibility=false to silence." maxlog = 1

    n_innov = size(wold.Theta, 3)
    vars = vcat(ce.outcomes, ce.instruments)
    n_vars = length(vars)
    n_x = length(ce.outcomes)

    # one column = one baseline; solve under the policy with draws off
    function _solve_columns(Theta_w::AbstractArray{T,3}, ce_use::PolicyCausalEffects{T})
        Tcf = Array{T,3}(undef, H, n_vars, n_innov)
        _suppress_warnings() do
            for j in 1:n_innov
                x = [Vector{T}(Theta_w[1:H, row_of[s], j]) for s in ce_use.outcomes]
                z = [Vector{T}(Theta_w[1:H, row_of[s], j]) for s in ce_use.instruments]
                base_j = BaselinePath{T}(copy(ce_use.outcomes), copy(ce_use.instruments),
                                         x, z, nothing, nothing, H, "wold innovation $j")
                pc = policy isa PolicyRule ?
                     policy_counterfactual(base_j, ce_use, policy; draws=:off) :
                     optimal_policy(base_j, ce_use, policy; draws=:off)
                for (v, _) in enumerate(ce_use.outcomes)
                    Tcf[:, v, j] = pc.x_cf[v]
                end
                for (k, _) in enumerate(ce_use.instruments)
                    Tcf[:, n_x+k, j] = pc.z_cf[k]
                end
            end
        end
        return Tcf
    end

    Theta_cf = _solve_columns(wold.Theta, ce)
    Theta_base = Array{T,3}(undef, H, n_vars, n_innov)
    for (v, s) in enumerate(vars), j in 1:n_innov
        Theta_base[:, v, j] = wold.Theta[1:H, row_of[s], j]
    end

    _mom(Th) = band === nothing ? _vma_covariance(Th) :
               _vma_band_covariance(Th, band[1], band[2])
    Sigma_base = _mom(Theta_base)
    Sigma_cf = _mom(Theta_cf)
    sd_base = sqrt.(max.(diag(Sigma_base), zero(T)))
    sd_cf = sqrt.(max.(diag(Sigma_cf), zero(T)))
    _corr(S, sd) = [sd[i] > 0 && sd[j] > 0 ? S[i, j] / (sd[i] * sd[j]) : zero(T)
                    for i in 1:n_vars, j in 1:n_vars]
    corr_base = _corr(Sigma_base, sd_base)
    corr_cf = _corr(Sigma_cf, sd_cf)

    tail_len = min(10, H)
    denom = norm(Theta_cf)
    tail_share = denom > 0 ? T(norm(@view(Theta_cf[H-tail_len+1:H, :, :])) / denom) : zero(T)
    tail_share > T(0.01) &&
        @warn "counterfactual_moments: tail_share = $(round(tail_share, sigdigits=3)) > 1% — the VMA sum has not converged; increase H (and the Wold horizon)"

    # --- draw propagation on sd_cf ---
    nd_ce = n_draws(ce)
    nd_w = n_draws(wold)
    nd = draw_source == :ce ? nd_ce : draw_source == :wold ? nd_w : min(nd_ce, nd_w)
    if draw_source == :both && nd_ce != nd_w && (nd_ce > 0 || nd_w > 0)
        throw(ArgumentError(
            "draw_source = :both requires matching draw counts, got ce $nd_ce vs wold $nd_w"))
    end
    use_draws = draws == :on || (draws == :auto && nd > 0)
    use_draws && nd == 0 && throw(ArgumentError(
        "draws = :on requires draws on the selected draw_source (:$draw_source)"))
    qlev = collect(T, quantiles)
    sd_cf_bands = nothing
    if use_draws
        sds = Matrix{T}(undef, n_vars, nd)
        keep = falses(nd)
        for d in 1:nd
            ce_d = draw_source == :wold ? ce :
                   PolicyCausalEffects{T}(ce.outcomes, ce.instruments,
                                          [Matrix{T}(ce.Theta_x_draws[i][:, :, d]) for i in eachindex(ce.outcomes)],
                                          [Matrix{T}(ce.Theta_z_draws[k][:, :, d]) for k in eachindex(ce.instruments)],
                                          nothing, nothing, H, ce.shock_labels, ce.source)
            Tw_d = draw_source == :ce ? wold.Theta : wold.draws[:, :, :, d]
            try
                Tcf_d = _solve_columns(Tw_d, ce_d)
                s = sqrt.(max.(diag(_mom(Tcf_d)), zero(T)))
                if all(isfinite, s)
                    sds[:, d] = s
                    keep[d] = true
                end
            catch
                keep[d] = false
            end
        end
        used = findall(keep)
        n_failed = nd - length(used)
        n_failed > 0 && @warn "counterfactual_moments: $n_failed of $nd draws failed and were dropped"
        if !isempty(used)
            sd_cf_bands = Matrix{T}([quantile(@view(sds[v, used]), q) for v in 1:n_vars, q in qlev])
        end
    end

    CounterfactualMoments{T}(vars, Sigma_base, Sigma_cf, sd_base, sd_cf,
                             corr_base, corr_cf, sd_cf_bands, Theta_cf,
                             policy.name, H, tail_share, band)
end
