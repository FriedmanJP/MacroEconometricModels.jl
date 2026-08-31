# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Impulse Response Functions for frequentist and Bayesian VAR models.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Shared MA / structural-IRF kernels (SID-19)
# =============================================================================

"""
    ma_coefficients(B, n, p, H) -> Vector{Matrix{T}}

Reduced-form moving-average coefficients `Φ_0 = I, Φ_1, …, Φ_{H-1}` of a VAR(p).
"""
function ma_coefficients(B::AbstractMatrix{T}, n::Int, p::Int, H::Int) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("H must be positive, got $H"))
    A = extract_ar_coefficients(B, n, p)
    Phi = [zeros(T, n, n) for _ in 1:H]
    copyto!(Phi[1], I(n))
    scratch = zeros(T, n, n)
    @inbounds for h in 2:H
        for j in 1:min(p, h - 1)
            mul!(scratch, A[j], Phi[h-j])
            Phi[h] .+= scratch
        end
    end
    Phi
end

ma_coefficients(model::VARModel, H::Int) = ma_coefficients(model.B, nvars(model), model.p, H)

"""Stacked MA coefficients as an `H × n × n` array (`Φ_0` in slice 1)."""
function _ma_array(model::VARModel{T}, H::Int) where {T<:AbstractFloat}
    Phi = ma_coefficients(model, H)
    n = nvars(model)
    out = zeros(T, H, n, n)
    @inbounds for h in 1:H
        out[h, :, :] = Phi[h]
    end
    out
end

"""
    structural_irf(Phi, L, Q, H) -> Array{T,3}

Structural IRFs `Θ_h = Φ_h L Q` for `h = 0,…,H-1`. `Phi` is length ≥ `H`.
"""
function structural_irf(Phi::Vector{<:AbstractMatrix{T}}, L::AbstractMatrix{T},
                        Q::AbstractMatrix{T}, H::Int) where {T<:AbstractFloat}
    n = size(L, 1)
    P = L * Q
    irf = zeros(T, H, n, n)
    @inbounds for h in 1:H
        irf[h, :, :] = Phi[h] * P
    end
    irf
end

# =============================================================================
# Frequentist IRF
# =============================================================================

"""
    irf(model, horizon; method=:cholesky, ci_type=:none, reps=200, conf_level=0.95, ...)

Compute IRFs with optional confidence intervals.

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

# CI types
- `:none`
- `:bootstrap` --- residual (recursive-design) bootstrap: resamples the estimated residuals,
  regenerates data from `B`, and re-estimates the VAR per replication. With
  `stationary_only=true`, draws whose companion matrix has `|λmax| ≥ 1` are rejected and
  redrawn. For `:sign`/`:narrative` this mixes Haar rotations with sampling uncertainty
  and throws unless `set_inference=:bootstrap_x_rotations`.
- `:theoretical` --- asymptotic (delta-method) confidence intervals. Throws for
  `:sign`/`:narrative`: identified-set methods do not have theoretical residual-free
  coefficient bands; use the default identified-set bands or
  `set_inference=:bootstrap_x_rotations`.
- `:identified_set` --- returned automatically for `:sign`/`:narrative`: pointwise median
  over accepted Haar rotations, with quantile bands over that set.

# Set-ID options (SID-05)
- `max_draws::Int=1000` --- Haar draws for `:sign`/`:narrative` (forwarded to `compute_Q`).
- `set_inference::Symbol=:none` --- pass `:bootstrap_x_rotations` to opt into
  bootstrap×rotation bands with `ci_type=:bootstrap`.

# Bootstrap options ([T271])
- `bootstrap::Symbol=:iid` --- residual resampling scheme. `:iid` resamples rows with
  replacement (the historical default); `:wild` multiplies each residual ROW by one scalar
  draw, preserving the contemporaneous cross-equation covariance while randomising the
  conditional variance (Gonçalves & Kilian 2004); `:block` concatenates contiguous blocks,
  retaining serial dependence (Brüggemann et al. 2016).
- `block_length::Int=0` --- moving-block length; `0` selects `⌈T^{1/3}⌉`.
- `wild_dist::Symbol=:rademacher` --- `:rademacher` (±1) or `:mammen` (matches the third
  moment as well).
- `bias_correct::Bool=false` --- Kilian (1998) bootstrap-after-bootstrap. An inner bootstrap
  estimates the small-sample bias `Ψ = E[B*] − B̂`; the DGP is re-centred at `B̂ − δΨ` with
  Kilian's stationarity shrinkage, each outer draw is corrected by the same `Ψ` before its
  IRF is computed, **and the reported point IRF is computed from the bias-corrected
  coefficients** (`B̂ − δΨ`). Off by default, so existing bands/points are unchanged.
- `bias_reps::Int=reps` --- replications for the inner bias bootstrap.
"""
function irf(model::VARModel{T}, horizon::Int;
    method::Symbol=:cholesky, check_func=nothing, narrative_check=nothing,
    ci_type::Symbol=:none, reps::Int=200, conf_level::Real=0.95,
    stationary_only::Bool=false,
    bootstrap::Symbol=:iid, block_length::Int=0, wild_dist::Symbol=:rademacher,
    bias_correct::Bool=false, bias_reps::Int=0,
    shock_names::Union{Nothing,Vector{String}}=nothing,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing,
    restrictions=nothing,
    max_draws::Int=1000,
    set_inference::Symbol=:none,
    seed::Union{Integer,Nothing}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...
) where {T<:AbstractFloat}

    # Reproducibility (T246/#345): a `seed` owns the RNG so bootstrap bands can be
    # reproduced bit-for-bit (the per-replication sub-seeding is thread-invariant).
    rng = _resolve_repro_rng(rng, seed)
    _validate_data(model.Sigma, "Sigma")
    _validate_data(model.B, "B")
    # Kilian bias correction needs the bootstrap machinery to estimate Psi; with
    # any other ci_type it used to be silently ignored, returning the uncorrected
    # point under a kwarg that promised correction (#564).
    if bias_correct && ci_type != :bootstrap
        throw(ArgumentError(
            "bias_correct=true requires ci_type=:bootstrap (Kilian 1998 estimates " *
            "the coefficient bias by bootstrap); got ci_type=:$ci_type"))
    end
    # SID-05/19: set-identified methods return the pointwise median IRF with
    # identified-set bands. Bootstrap×rotations requires an explicit opt-in.
    if _is_set_identified(method)
        ci_type === :bootstrap && set_inference !== :bootstrap_x_rotations &&
            throw(ArgumentError("ci_type=:bootstrap with method=:$method mixes rotations; " *
                                "pass set_inference=:bootstrap_x_rotations to opt in"))
        ci_type === :theoretical &&
            throw(ArgumentError("ci_type=:theoretical with method=:$method: identified-set methods " *
                                "do not have theoretical residual-free coefficient bands; " *
                                "use the default identified-set bands or set_inference=:bootstrap_x_rotations"))
        if ci_type !== :bootstrap
            snames = isnothing(shock_names) ? model.varnames : shock_names
            if method === :arias
                isnothing(restrictions) && throw(ArgumentError("arias requires restrictions"))
                s = identify_arias(model, restrictions, horizon;
                                   _arias_freq_kwargs(max_draws; rng=rng, kwargs...)...)
                alpha = (1 - T(conf_level)) / 2
                pct = irf_percentiles(s; quantiles=Float64[alpha, 0.5, 1 - alpha])
                med = pct[:, :, :, 2]
                lo, hi = pct[:, :, :, 1], pct[:, :, :, 3]
                return ImpulseResponse{T}(med, lo, hi, horizon, model.varnames, snames,
                                          :identified_set, s.irf_draws, T(conf_level);
                                          manifest=capture_manifest(; seed=seed, settings=Dict{String,Any}(
                                              "method" => String(method), "ci_type" => "identified_set",
                                              "max_draws" => max_draws, "acceptance_rate" => s.acceptance_rate)))
            end
            isnothing(check_func) && throw(ArgumentError(
                method === :narrative ? "Need check_func and narrative_check for narrative" :
                                        "Need check_func for sign"))
            method === :narrative && isnothing(narrative_check) &&
                throw(ArgumentError("Need check_func and narrative_check for narrative"))
            s = method === :sign ?
                identify_sign(model, horizon, check_func; max_draws=max_draws, store_all=true, rng=rng) :
                identify_narrative(model, horizon, check_func, narrative_check; max_draws=max_draws, store_all=true, rng=rng)
            med = irf_median(s)
            lo, hi = irf_bounds(s; quantiles=((1 - conf_level)/2, 1 - (1 - conf_level)/2))
            return ImpulseResponse{T}(med, lo, hi, horizon, model.varnames, snames,
                                      :identified_set, s.irf_draws, T(conf_level);
                                      manifest=capture_manifest(; seed=seed, settings=Dict{String,Any}(
                                          "method" => String(method), "ci_type" => "identified_set",
                                          "max_draws" => max_draws, "acceptance_rate" => s.acceptance_rate)))
        end
    end
    if ci_type === :theoretical && _needs_residuals(method)
        throw(ArgumentError(
            "ci_type=:theoretical draws only the VAR coefficients; method=:$method " *
            "identifies from residuals — use ci_type=:bootstrap"))
    end
    n = nvars(model)
    p = model.p
    Q = compute_Q(model, method; horizon=horizon, check_func=check_func,
                  narrative_check=narrative_check, restrictions=restrictions,
                  max_draws=max_draws, transition_var=transition_var,
                  regime_indicator=regime_indicator, rng=rng, kwargs...)

    # Kilian (1998) bias-corrects the coefficient estimate itself, so the reported
    # point IRF must also use the bias-corrected B when bias_correct=true (#564).
    # Psi is estimated once here and reused for the outer bootstrap DGP/corrections.
    model_point = model
    Psi_point = nothing
    if bias_correct && ci_type == :bootstrap
        nb = bias_reps > 0 ? bias_reps : reps
        Psi_point = _estimate_var_bias(model, nb, bootstrap, rng;
                                       block_length=block_length, wild_dist=wild_dist)
        B_bc, _ = _kilian_bias_correction(model.B, Psi_point, n, p)
        model_point = VARModel(model.Y, p, B_bc, model.U, model.Sigma,
                               model.aic, model.bic, model.hqic, model.varnames)
        # Re-identify at the bias-corrected coefficients so the point Q matches.
        Q = compute_Q(model_point, method; horizon=horizon, check_func=check_func,
                      narrative_check=narrative_check, restrictions=restrictions,
                      max_draws=max_draws, transition_var=transition_var,
                      regime_indicator=regime_indicator, rng=rng, kwargs...)
    end
    point_irf = compute_irf(model_point, Q, horizon)

    ci_lower, ci_upper = zeros(T, horizon, n, n), zeros(T, horizon, n, n)
    sim_irfs = nothing
    if ci_type != :none
        sim_irfs, relabeled_frac = _simulate_irfs(model, method, horizon, check_func, narrative_check, ci_type, reps;
                                  stationary_only=stationary_only,
                                  transition_var=transition_var, regime_indicator=regime_indicator,
                                  restrictions=restrictions,
                                  rng=rng, bootstrap=bootstrap, block_length=block_length,
                                  wild_dist=wild_dist, bias_correct=bias_correct,
                                  bias_reps=bias_reps, Psi_precomputed=Psi_point, Q_ref=Q,
                                  max_draws=max_draws, kwargs...)
        alpha = (1 - T(conf_level)) / 2
        @inbounds for h in 1:horizon, v in 1:n, s in 1:n
            d = @view sim_irfs[:, h, v, s]
            ci_lower[h, v, s], ci_upper[h, v, s] = quantile(d, alpha), quantile(d, 1 - alpha)
        end
    end

    snames = isnothing(shock_names) ? model.varnames : shock_names
    cl = ci_type == :none ? zero(T) : T(conf_level)
    # Attach a reproducibility manifest whenever the bands consume randomness.
    manifest = ci_type == :none ? nothing :
        capture_manifest(; seed=seed, settings=Dict{String,Any}(
            "method" => String(method), "ci_type" => String(ci_type),
            "reps" => reps, "stationary_only" => stationary_only,
            "bootstrap" => String(bootstrap), "block_length" => block_length,
            "wild_dist" => String(wild_dist), "bias_correct" => bias_correct,
            "bias_reps" => bias_reps, "relabeled_fraction" => relabeled_frac))
    ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                       model.varnames, snames, ci_type, sim_irfs, cl; manifest=manifest)
end

"""Simulate IRFs for confidence intervals (bootstrap or asymptotic).

`Psi_precomputed` optionally supplies a bias matrix already estimated by the caller
(so the point IRF and bands share the same Ψ under `bias_correct=true`; #564).
`Q_ref` is the point-estimate rotation; statistical-ID columns are matched to
`chol(Σ) * Q_ref` before `compute_irf` (SID-04). Returns `(sim_irfs, relabeled_fraction)`.
"""
function _simulate_irfs(model::VARModel{T}, method::Symbol, horizon::Int,
    check_func, narrative_check, ci_type::Symbol, reps::Int;
    stationary_only::Bool=false,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing,
    restrictions=nothing,
    rng::AbstractRNG=Random.default_rng(),
    bootstrap::Symbol=:iid, block_length::Int=0, wild_dist::Symbol=:rademacher,
    bias_correct::Bool=false, bias_reps::Int=0,
    Psi_precomputed::Union{Nothing,AbstractMatrix}=nothing,
    Q_ref::Union{Nothing,AbstractMatrix}=nothing,
    max_draws::Int=1000,
    kwargs...
) where {T<:AbstractFloat}
    n, p = nvars(model), model.p
    bootstrap in (:iid, :wild, :block) || throw(ArgumentError(
        "bootstrap must be :iid, :wild, or :block; got :$bootstrap"))
    P_ref = (Q_ref !== nothing && _should_match_columns(method)) ?
        Matrix{T}(safe_cholesky(model.Sigma) * Q_ref) : nothing

    if ci_type == :bootstrap
        U, T_eff = model.U, size(model.U, 1)
        Y_init = model.Y[1:p, :]

        # Kilian (1998) bootstrap-after-bootstrap: estimate the bias once, re-centre the DGP
        # at the corrected coefficients, and correct every outer draw by the same Ψ.
        B_dgp = model.B
        Psi = zeros(T, size(model.B))
        if bias_correct
            if Psi_precomputed !== nothing
                Psi = Matrix{T}(Psi_precomputed)
            else
                nb = bias_reps > 0 ? bias_reps : reps
                Psi = _estimate_var_bias(model, nb, bootstrap, rng;
                                         block_length=block_length, wild_dist=wild_dist)
            end
            B_dgp, _ = _kilian_bias_correction(model.B, Psi, n, p)
        end

        if stationary_only
            # Rejection sampling with DETERMINISTIC slot assignment (C-02): seed each of the
            # 1:max_iter iterations by its index, write each passing draw into a staging buffer
            # at its iteration index, then keep the first `reps` passing draws IN INDEX ORDER.
            # Result is invariant to thread scheduling / JULIA_NUM_THREADS.
            max_iter = 10 * reps
            staging = zeros(T, max_iter, horizon, n, n)
            passed = fill(false, max_iter)
            relabeled = fill(false, max_iter)
            seeds = rand(rng, UInt64, max_iter)
            Threads.@threads for it in 1:max_iter
                local_rng = Random.MersenneTwister(seeds[it])
                _suppress_warnings() do
                    U_boot = _resample_residuals(U, bootstrap, local_rng;
                                                 block_length=block_length, wild_dist=wild_dist)
                    Y_boot = _simulate_var(Y_init, B_dgp, U_boot, T_eff + p)
                    m = estimate_var(Y_boot, p; check_stability=false)
                    m = _apply_boot_bias_correction(m, Psi, n, p, bias_correct)
                    F = companion_matrix(m.B, n, p)
                    maximum(abs.(eigvals(F))) >= one(T) && return  # reject non-stationary draw
                    Q = compute_Q(m, method; horizon=horizon, check_func=check_func,
                                  narrative_check=narrative_check, restrictions=restrictions,
                                  max_draws=max_draws, transition_var=transition_var,
                                  regime_indicator=regime_indicator, rng=local_rng, kwargs...)
                    Q, was_relabeled = _maybe_match_Q(Q, m, P_ref)
                    staging[it, :, :, :] = compute_irf(m, Q, horizon)
                    relabeled[it] = was_relabeled
                    passed[it] = true
                end
            end
            kept = findall(passed)
            n_valid = min(length(kept), reps)
            n_valid < reps && @warn "Only $n_valid/$reps stationary bootstrap draws obtained after $max_iter iterations"
            sim_irfs = zeros(T, max(n_valid, 1), horizon, n, n)
            @inbounds for j in 1:n_valid
                sim_irfs[j, :, :, :] = staging[kept[j], :, :, :]
            end
            frac = n_valid == 0 ? zero(T) : T(count(j -> relabeled[kept[j]], 1:n_valid)) / T(n_valid)
            return sim_irfs, frac
        else
            sim_irfs = zeros(T, reps, horizon, n, n)
            relabeled = fill(false, reps)
            seeds = rand(rng, UInt64, reps)
            Threads.@threads for r in 1:reps
                local_rng = Random.MersenneTwister(seeds[r])
                _suppress_warnings() do
                    U_boot = _resample_residuals(U, bootstrap, local_rng;
                                                 block_length=block_length, wild_dist=wild_dist)
                    Y_boot = _simulate_var(Y_init, B_dgp, U_boot, T_eff + p)
                    m = estimate_var(Y_boot, p; check_stability=false)
                    m = _apply_boot_bias_correction(m, Psi, n, p, bias_correct)
                    Q = compute_Q(m, method; horizon=horizon, check_func=check_func,
                                  narrative_check=narrative_check, restrictions=restrictions,
                                  max_draws=max_draws, transition_var=transition_var,
                                  regime_indicator=regime_indicator, rng=local_rng, kwargs...)
                    Q, was_relabeled = _maybe_match_Q(Q, m, P_ref)
                    sim_irfs[r, :, :, :] = compute_irf(m, Q, horizon)
                    relabeled[r] = was_relabeled
                end
            end
            return sim_irfs, T(count(relabeled)) / T(reps)
        end
    elseif ci_type == :theoretical
        _, X = construct_var_matrices(model.Y, p)
        L_V, L_S = safe_cholesky(robust_inv(X'X)), safe_cholesky(model.Sigma)
        k = ncoefs(model)

        if stationary_only
            max_iter = 10 * reps
            staging = zeros(T, max_iter, horizon, n, n)
            passed = fill(false, max_iter)
            relabeled = fill(false, max_iter)
            seeds = rand(rng, UInt64, max_iter)
            Threads.@threads for it in 1:max_iter
                local_rng = Random.MersenneTwister(seeds[it])
                _suppress_warnings() do
                    B_star = model.B + L_V * randn(local_rng, T, k, n) * L_S'
                    F = companion_matrix(B_star, n, p)
                    maximum(abs.(eigvals(F))) >= one(T) && return  # reject non-stationary draw
                    m = VARModel(zeros(T, 0, n), p, B_star, zeros(T, 0, n), model.Sigma, zero(T), zero(T), zero(T))
                    Q = compute_Q(m, method; horizon=horizon, check_func=check_func,
                                  narrative_check=narrative_check, restrictions=restrictions,
                                  max_draws=max_draws, transition_var=transition_var,
                                  regime_indicator=regime_indicator, rng=local_rng, kwargs...)
                    Q, was_relabeled = _maybe_match_Q(Q, m, P_ref)
                    staging[it, :, :, :] = compute_irf(m, Q, horizon)
                    relabeled[it] = was_relabeled
                    passed[it] = true
                end
            end
            kept = findall(passed)
            n_valid = min(length(kept), reps)
            n_valid < reps && @warn "Only $n_valid/$reps stationary theoretical draws obtained after $max_iter iterations"
            sim_irfs = zeros(T, max(n_valid, 1), horizon, n, n)
            @inbounds for j in 1:n_valid
                sim_irfs[j, :, :, :] = staging[kept[j], :, :, :]
            end
            frac = n_valid == 0 ? zero(T) : T(count(j -> relabeled[kept[j]], 1:n_valid)) / T(n_valid)
            return sim_irfs, frac
        else
            sim_irfs = zeros(T, reps, horizon, n, n)
            relabeled = fill(false, reps)
            seeds = rand(rng, UInt64, reps)
            Threads.@threads for r in 1:reps
                local_rng = Random.MersenneTwister(seeds[r])
                _suppress_warnings() do
                    B_star = model.B + L_V * randn(local_rng, T, k, n) * L_S'
                    m = VARModel(zeros(T, 0, n), p, B_star, zeros(T, 0, n), model.Sigma, zero(T), zero(T), zero(T))
                    Q = compute_Q(m, method; horizon=horizon, check_func=check_func,
                                  narrative_check=narrative_check, restrictions=restrictions,
                                  max_draws=max_draws, transition_var=transition_var,
                                  regime_indicator=regime_indicator, rng=local_rng, kwargs...)
                    Q, was_relabeled = _maybe_match_Q(Q, m, P_ref)
                    sim_irfs[r, :, :, :] = compute_irf(m, Q, horizon)
                    relabeled[r] = was_relabeled
                end
            end
            return sim_irfs, T(count(relabeled)) / T(reps)
        end
    end
    zeros(T, reps, horizon, n, n), zero(T)
end

"""Simulate VAR data from initial conditions and innovations."""
function _simulate_var(Y_init::AbstractMatrix{T}, B::AbstractMatrix{T},
                       U::AbstractMatrix{T}, T_total::Int) where {T<:AbstractFloat}
    p, n = size(Y_init)
    Y = zeros(T, T_total, n)
    Y[1:p, :] = Y_init

    A = extract_ar_coefficients(B, n, p)
    intercept = @view B[1, :]

    @inbounds for t in (p+1):T_total
        Y[t, :] = intercept
        for i in 1:p
            Y[t, :] .+= A[i] * @view(Y[t-i, :])
        end
        Y[t, :] .+= @view(U[t-p, :])
    end
    Y
end

# =============================================================================
# Bayesian IRF
# =============================================================================

"""
    irf(post::BVARPosterior, horizon; method=:cholesky, quantiles=[0.16, 0.5, 0.84], point_estimate=:mean, ...)

Compute Bayesian IRFs from posterior draws with posterior quantiles.
Uses posterior mean as central tendency by default (pass `point_estimate=:median` for median).

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

Uses `process_posterior_samples` and `compute_posterior_quantiles` from bayesian_utils.jl.
"""
function irf(post::BVARPosterior, horizon::Int;
    method::Symbol=:cholesky, data::AbstractMatrix=Matrix{Float64}(undef, 0, 0),
    check_func=nothing, narrative_check=nothing, quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
    threaded::Bool=false, point_estimate::Symbol=:mean,
    shock_names::Union{Nothing,Vector{String}}=nothing,
    max_draws::Int=1000,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing,
    restrictions=nothing,
    kwargs...
)
    use_data = isempty(data) ? post.data : data
    _validate_narrative_data(method, use_data)

    n = post.n
    ET = eltype(use_data)

    # Weighted identified-set summary; do not fall through to compute_Q (n_draws=1, unweighted).
    if method === :arias
        rng, n_rot, cw = _arias_posterior_kwargs(max_draws; kwargs...)
        use_data_a = isempty(data) ? nothing : data
        rset = _arias_from_bvar_posterior(post, restrictions, horizon;
            data=use_data_a, n_rotations=n_rot, quantiles=quantiles,
            rng=rng, compute_weights=cw)
        bir = irf(rset; quantiles=collect(quantiles))
        shock_names === nothing && return bir
        return BayesianImpulseResponse{ET}(bir.quantiles, bir.point_estimate, bir.horizon,
            bir.variables, shock_names, bir.quantile_levels, bir._draws,
            bir.n_requested, bir.n_effective, bir.n_failed)
    end

    # Process posterior samples using shared utility
    results, samples = process_posterior_samples(post,
        (m, Q, h) -> compute_irf(m, Q, h);
        data=use_data, method=method, horizon=horizon,
        check_func=check_func, narrative_check=narrative_check,
        max_draws=max_draws,
        transition_var=transition_var, regime_indicator=regime_indicator,
        restrictions=restrictions, kwargs...
    )

    # Stack results into single array
    all_irfs = stack_posterior_results(results, (horizon, n, n), ET)

    # Compute quantiles using shared utility (threaded for large arrays)
    q_vec = ET.(quantiles)
    use_threaded = threaded || (samples * horizon * n * n > 100000)
    irf_q, irf_m = compute_posterior_quantiles(all_irfs, q_vec; threaded=use_threaded, central=point_estimate)

    snames = isnothing(shock_names) ? post.varnames : shock_names
    # MC honesty (#244): process_posterior_samples drops non-stationary / unidentified draws.
    n_req = post.n_draws
    BayesianImpulseResponse{ET}(irf_q, irf_m, horizon, post.varnames, snames, q_vec, all_irfs,
                                n_req, samples, n_req - samples)
end

# Deprecated wrapper for old (chain, p, n, horizon) signature
function irf(post::BVARPosterior, p::Int, n::Int, horizon::Int; kwargs...)
    irf(post, horizon; kwargs...)
end

"""IRF view of a [`BayesianSetIdentifiedSVAR`](@ref).

Quantiles are always recomputed from draws and importance weights (same as
[`fevd`](@ref)), so requested levels are honored even when they differ from the
stored `irf_quantiles` (default `[0.16, 0.5, 0.84]`).
"""
function irf(r::BayesianSetIdentifiedSVAR{T}; quantiles::Vector{<:Real}=[0.16, 0.5, 0.84]) where {T<:AbstractFloat}
    H = size(r.irf_draws, 2)
    n_acc = size(r.irf_draws, 1)
    q_vec = T.(quantiles)
    n1, n2 = size(r.irf_draws, 3), size(r.irf_draws, 4)
    qarr = zeros(T, H, n1, n2, length(q_vec))
    mean_irf = zeros(T, H, n1, n2)
    w = r.weights
    for h in 1:H, i in axes(r.irf_draws, 3), j in axes(r.irf_draws, 4)
        vals = @view r.irf_draws[:, h, i, j]
        mean_irf[h, i, j] = sum(w .* vals)
        for (qi, q) in enumerate(q_vec)
            qarr[h, i, j, qi] = _weighted_quantile(vals, w, q)
        end
    end
    n_req = n_acc + r.n_unidentified
    BayesianImpulseResponse{T}(qarr, mean_irf, H, r.varnames, r.varnames, q_vec, r.irf_draws,
                               n_req, n_acc, r.n_unidentified)
end

# =============================================================================
# Structural LP IRF Accessor
# =============================================================================

"""
    irf(slp::StructuralLP) -> ImpulseResponse

Extract the impulse response object from a structural LP result.
"""
irf(slp::StructuralLP) = slp.irf

# =============================================================================
# Local Projection IRF
# =============================================================================

"""
    lp_irf(model::LPModel{T}; conf_level=0.95, ci_type=:analytical, bootstrap=:wild, …)
        -> LPImpulseResponse{T}

Extract impulse response functions with confidence intervals from an LP model.

# Confidence intervals
- `ci_type=:analytical` (default) — HAC (Newey-West or White) standard errors from the
  estimated `vcov`, unchanged from before.
- `ci_type=:bootstrap` — percentile bands from a **fixed-design residual bootstrap** ([T271]).
  At each horizon the regressor matrix is held fixed and only the errors are resampled:
  `y* = X_h·β̂_h + u*`, refit by OLS. Holding the design fixed is what makes this valid for
  LP, where the regressors are predetermined but the errors are MA(h)-correlated by
  construction.

# Bootstrap options
- `bootstrap::Symbol=:wild` — `:wild` (default here, unlike the VAR, because LP residuals are
  serially correlated *and* often heteroskedastic), `:block`, or `:iid`.
- `block_length::Int=0` — moving-block length; `0` selects `⌈T^{1/3}⌉`.
- `wild_dist::Symbol=:rademacher` — or `:mammen`.
- `reps::Int=500`, `rng`/`seed` for reproducibility.

The point estimates and standard errors are the analytical ones in both cases; only the
bands change, so switching `ci_type` never moves the reported IRF.
"""
function lp_irf(model::LPModel{T}; conf_level::Real=0.95,
                ci_type::Symbol=:analytical, bootstrap::Symbol=:wild,
                block_length::Int=0, wild_dist::Symbol=:rademacher,
                reps::Int=500, seed::Union{Integer,Nothing}=nothing,
                rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    ci_type in (:analytical, :bootstrap) || throw(ArgumentError(
        "ci_type must be :analytical or :bootstrap; got :$ci_type"))
    # Validate BEFORE the threaded bootstrap loop: an ArgumentError raised inside
    # `Threads.@threads` surfaces as a TaskFailedException, which is a worse diagnostic.
    bootstrap in (:iid, :wild, :block) || throw(ArgumentError(
        "bootstrap must be :iid, :wild, or :block; got :$bootstrap"))
    wild_dist in (:rademacher, :mammen) || throw(ArgumentError(
        "wild_dist must be :rademacher or :mammen; got :$wild_dist"))
    irf_data = extract_shock_irf(model.B, model.vcov, model.response_vars, 2;
                                  conf_level=conf_level)

    ci_lower, ci_upper = irf_data.ci_lower, irf_data.ci_upper
    if ci_type == :bootstrap
        rng = _resolve_repro_rng(rng, seed)
        ci_lower, ci_upper = _lp_irf_bootstrap_bands(model, T(conf_level), reps, bootstrap,
                                                     block_length, wild_dist, rng)
    end

    response_names = model.varnames[model.response_vars]
    shock_name = model.varnames[model.shock_var]
    cov_type_sym = model.cov_estimator isa NeweyWestEstimator ? :newey_west : :white

    LPImpulseResponse{T}(irf_data.values, ci_lower, ci_upper,
                         irf_data.se, model.horizon, response_names, shock_name,
                         cov_type_sym, T(conf_level))
end

"""
    _lp_irf_bootstrap_bands(model, conf_level, reps, scheme, block_length, wild_dist, rng)
        → (ci_lower, ci_upper)

Percentile bands for the LP shock coefficient by a fixed-design residual bootstrap, run
horizon by horizon.

The design `X_h` is rebuilt from the model's own data and reused across replications, so the
only randomness is in the resampled errors. Replication seeds are drawn up front and each
result written to its own slot, making the bands invariant to thread scheduling ([T144]).
"""
function _lp_irf_bootstrap_bands(model::LPModel{T}, conf_level::T, reps::Int,
                                 scheme::Symbol, block_length::Int, wild_dist::Symbol,
                                 rng::AbstractRNG) where {T<:AbstractFloat}
    H = model.horizon
    nr = length(model.response_vars)
    ci_lower = Matrix{T}(undef, H + 1, nr)
    ci_upper = Matrix{T}(undef, H + 1, nr)
    alpha = (1 - conf_level) / 2

    for h in 0:H
        _, X_h, _ = construct_lp_matrices(model.Y, model.shock_var, h, model.lags;
                                          response_vars=model.response_vars)
        B_h = model.B[h+1]
        U_h = model.residuals[h+1]
        fitted = X_h * B_h
        XtX_inv = robust_inv(X_h' * X_h)
        draws = zeros(T, reps, nr)
        seeds = rand(rng, UInt64, reps)
        Threads.@threads for r in 1:reps
            local_rng = Random.MersenneTwister(seeds[r])
            U_star = _resample_residuals(U_h, scheme, local_rng;
                                         block_length=block_length, wild_dist=wild_dist)
            B_star = XtX_inv * (X_h' * (fitted + U_star))
            @inbounds for j in 1:nr
                draws[r, j] = B_star[2, j]     # row 2 is the shock coefficient
            end
        end
        @inbounds for j in 1:nr
            d = @view draws[:, j]
            ci_lower[h+1, j] = quantile(d, alpha)
            ci_upper[h+1, j] = quantile(d, 1 - alpha)
        end
    end
    return (ci_lower, ci_upper)
end

"""
    lp_irf(Y::AbstractMatrix, shock_var::Int, horizon::Int; kwargs...) -> LPImpulseResponse

Convenience function: estimate LP and extract IRF in one call.
"""
function lp_irf(Y::AbstractMatrix, shock_var::Int, horizon::Int; conf_level::Real=0.95,
                ci_type::Symbol=:analytical, bootstrap::Symbol=:wild, block_length::Int=0,
                wild_dist::Symbol=:rademacher, reps::Int=500,
                seed::Union{Integer,Nothing}=nothing, kwargs...)
    model = estimate_lp(Y, shock_var, horizon; kwargs...)
    lp_irf(model; conf_level=conf_level, ci_type=ci_type, bootstrap=bootstrap,
           block_length=block_length, wild_dist=wild_dist, reps=reps, seed=seed)
end

# =============================================================================
# Cumulative IRF
# =============================================================================

"""
    cumulative_irf(irf::LPImpulseResponse{T}) -> LPImpulseResponse{T}

Compute cumulative impulse response: Σₛ₌₀ʰ β_s.
"""
function cumulative_irf(irf::LPImpulseResponse{T}) where {T<:AbstractFloat}
    cum_values = cumsum(irf.values, dims=1)
    cum_se = sqrt.(cumsum(irf.se.^2, dims=1))

    z = T(quantile(Normal(), 1 - (1 - irf.conf_level) / 2))
    cum_ci_lower = cum_values .- z .* cum_se
    cum_ci_upper = cum_values .+ z .* cum_se

    LPImpulseResponse{T}(cum_values, cum_ci_lower, cum_ci_upper, cum_se, irf.horizon,
                         irf.response_vars, irf.shock_var, irf.cov_type, irf.conf_level)
end

"""
    cumulative_irf(irf_result::ImpulseResponse{T}) -> ImpulseResponse{T}

Compute cumulative impulse response for VAR models: Σₛ₌₀ʰ IRF_s.

When raw bootstrap/simulation draws are available, cumulates each draw first
then extracts quantiles — the statistically correct approach since quantiles
are NOT additive: Q_α(A+B) ≠ Q_α(A) + Q_α(B).
"""
function cumulative_irf(irf_result::ImpulseResponse{T}) where {T<:AbstractFloat}
    cum_values = cumsum(irf_result.values, dims=1)

    if irf_result._draws !== nothing && irf_result._conf_level > zero(T)
        # Correct approach: cumulate each draw, then extract quantiles
        cum_draws = cumsum(irf_result._draws, dims=2)
        alpha = (one(T) - irf_result._conf_level) / 2
        horizon, nv, ns = size(cum_values)
        cum_lower = zeros(T, horizon, nv, ns)
        cum_upper = zeros(T, horizon, nv, ns)
        @inbounds for h in 1:horizon, v in 1:nv, s in 1:ns
            d = @view cum_draws[:, h, v, s]
            cum_lower[h, v, s] = quantile(d, alpha)
            cum_upper[h, v, s] = quantile(d, 1 - alpha)
        end
    else
        # Fallback for no-CI case (ci_lower/ci_upper are zeros)
        cum_lower = cumsum(irf_result.ci_lower, dims=1)
        cum_upper = cumsum(irf_result.ci_upper, dims=1)
    end

    ImpulseResponse{T}(cum_values, cum_lower, cum_upper, irf_result.horizon,
                       irf_result.variables, irf_result.shocks, irf_result.ci_type)
end

"""
    cumulative_irf(irf_result::BayesianImpulseResponse{T}) -> BayesianImpulseResponse{T}

Compute cumulative Bayesian impulse response: Σₛ₌₀ʰ IRF_s.

When raw posterior draws are available, cumulates each draw first then
extracts quantiles — the statistically correct approach.
"""
function cumulative_irf(irf_result::BayesianImpulseResponse{T}) where {T<:AbstractFloat}
    cum_pe = cumsum(irf_result.point_estimate, dims=1)

    if irf_result._draws !== nothing
        # Correct approach: cumulate each draw, then extract quantiles
        cum_draws = cumsum(irf_result._draws, dims=2)
        q_vec = irf_result.quantile_levels
        horizon, nv, ns = size(cum_pe)
        nq = length(q_vec)
        cum_quantiles = zeros(T, horizon, nv, ns, nq)
        @inbounds for h in 1:horizon, v in 1:nv, s in 1:ns
            d = @view cum_draws[:, h, v, s]
            for (qi, q) in enumerate(q_vec)
                cum_quantiles[h, v, s, qi] = quantile(d, q)
            end
        end
    else
        cum_quantiles = cumsum(irf_result.quantiles, dims=1)
    end

    # Cumulation is a deterministic transform of the same draws — propagate the MC counts.
    BayesianImpulseResponse{T}(cum_quantiles, cum_pe, irf_result.horizon,
                               irf_result.variables, irf_result.shocks, irf_result.quantile_levels,
                               nothing, irf_result.n_requested, irf_result.n_effective, irf_result.n_failed)
end

# =============================================================================
# Reproducibility (T246 / #345)
# =============================================================================

"""
    reproduce(ir::ImpulseResponse, model::VARModel) -> ReproReport

Re-run the bootstrap that produced `ir`'s confidence bands from the manifest's
recorded seed and settings and check the point IRF and bands match bit-for-bit.
The source `model` is passed explicitly — a large object deliberately not retained
on the IRF result. Requires `ir` to have been produced by
`irf(model, H; ci_type=:bootstrap, seed=N)`.

```julia
model = estimate_var(Y, 2)
ir = irf(model, 20; ci_type=:bootstrap, reps=200, seed=20260717)
reproduce(ir, model)   # ReproReport: PASS
```
"""
function reproduce(ir::ImpulseResponse, model::VARModel)
    m = ir.manifest
    m === nothing && return _no_manifest_report("ImpulseResponse")
    m.seed === nothing && return _no_seed_report(m, "irf(model, H; ci_type=:bootstrap, seed=N)")
    s = m.settings
    fresh = irf(model, ir.horizon;
                method = Symbol(get(s, "method", "cholesky")),
                ci_type = Symbol(get(s, "ci_type", "bootstrap")),
                reps = Int(get(s, "reps", ir._draws === nothing ? 0 : size(ir._draws, 1))),
                conf_level = ir._conf_level,
                stationary_only = Bool(get(s, "stationary_only", false)),
                seed = m.seed)
    diffs = [_repro_field_diff("values", ir.values, fresh.values),
             _repro_field_diff("ci_lower", ir.ci_lower, fresh.ci_lower),
             _repro_field_diff("ci_upper", ir.ci_upper, fresh.ci_upper)]
    return _finalize_repro(diffs, m)
end

# Single-argument form: the source model is required (not retained on the result).
reproduce(::ImpulseResponse) =
    _needs_source_report("bootstrap ImpulseResponse", "reproduce(ir, model)")
