# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Forecast Error Variance Decomposition for frequentist and Bayesian VAR models.
"""

using LinearAlgebra, Statistics

# =============================================================================
# Frequentist FEVD
# =============================================================================

"""
    fevd(model, horizon; method=:cholesky, ...) -> FEVD

Compute FEVD showing proportion of h-step forecast error variance attributable to each shock.

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`, `:proxy`, `:ab`, `:max_share`, `:svec`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.
      `:proxy` requires `instruments` and is partial when `k < n`.
      `:ab` requires `pattern::SVARPattern`.
      `:max_share` requires `target` and is partial; pass `horizons` or `band`.
      `:svec` is the structural-VECM route (`identify_svec` on a `VECMModel`).

For `:sign`/`:narrative`, each accepted rotation gets its own FEVD; the reported
decomposition and proportions are the pointwise median. `n_effective` is the
number of accepted rotations.
"""
function fevd(model::VARModel{T}, horizon::Int;
    method::Symbol=:cholesky, check_func=nothing, narrative_check=nothing,
    shock_names::Union{Nothing,Vector{String}}=nothing,
    transition_var::Union{Nothing,AbstractVector}=nothing,
    regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing,
    restrictions=nothing,
    max_draws::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...
) where {T<:AbstractFloat}
    _validate_data(model.Sigma, "Sigma")
    _validate_data(model.B, "B")
    snames = isnothing(shock_names) ? model.varnames : shock_names
    # SID-05/19: FEVD of each accepted rotation, then pointwise median. The median
    # IRF is not Σ-orthonormal, so this is not FEVD-of-the-median-IRF.
    if _is_set_identified(method)
        irf_draws, n_acc, wts = if method === :arias
            isnothing(restrictions) && throw(ArgumentError("arias requires restrictions"))
            s = identify_arias(model, restrictions, horizon;
                               _arias_freq_kwargs(max_draws; rng=rng, kwargs...)...)
            s.irf_draws, length(s.Q_draws), s.weights
        else
            isnothing(check_func) && throw(ArgumentError(
                method === :narrative ? "Need check_func and narrative_check for narrative" :
                                        "Need check_func for sign"))
            method === :narrative && isnothing(narrative_check) &&
                throw(ArgumentError("Need check_func and narrative_check for narrative"))
            s = method === :sign ?
                identify_sign(model, horizon, check_func; max_draws=max_draws, store_all=true, rng=rng) :
                identify_narrative(model, horizon, check_func, narrative_check;
                                   max_draws=max_draws, store_all=true, rng=rng)
            s.irf_draws, s.n_accepted, nothing
        end
        n = nvars(model)
        _check_fevd_orthogonality(@view(irf_draws[1, 1, :, :]), model.Sigma; method=method)
        decomp_draws = zeros(T, n_acc, n, n, horizon)
        props_draws = zeros(T, n_acc, n, n, horizon)
        for i in 1:n_acc
            d, p = _compute_fevd(irf_draws[i, :, :, :], n, horizon)
            decomp_draws[i, :, :, :] = d
            props_draws[i, :, :, :] = p
        end
        decomp = zeros(T, n, n, horizon)
        props = zeros(T, n, n, horizon)
        @inbounds for v in 1:n, sh in 1:n, h in 1:horizon
            if wts === nothing
                decomp[v, sh, h] = quantile(@view(decomp_draws[:, v, sh, h]), T(0.5))
                props[v, sh, h] = quantile(@view(props_draws[:, v, sh, h]), T(0.5))
            else
                decomp[v, sh, h] = _weighted_quantile(decomp_draws[:, v, sh, h], wts, T(0.5))
                props[v, sh, h] = _weighted_quantile(props_draws[:, v, sh, h], wts, T(0.5))
            end
        end
        return FEVD{T}(decomp, props, model.varnames, snames, n_acc)
    end
    irf_result = irf(model, horizon; method, check_func, narrative_check,
                     transition_var=transition_var, regime_indicator=regime_indicator,
                     restrictions=restrictions, rng=rng, max_draws=max_draws, kwargs...)
    # The impact matrix P = IRF[1,:,:] = chol(Σ)·Q; the squared-IRF FEVD accumulation is a
    # proper variance decomposition only when P is Σ-orthonormal (P*P' = Σ ⇔ Q*Q' = I).
    if _is_partial(method)
        @warn "fevd: method=:$method is partially identified; FEVD shares of unidentified shocks are not identified." maxlog = 1
    end
    _check_fevd_orthogonality(@view(irf_result.values[1, :, :]), model.Sigma; method=method)
    decomp, props = _compute_fevd(irf_result.values, nvars(model), horizon)
    FEVD{T}(decomp, props, model.varnames, snames)
end

"""Compute FEVD from IRF array: decomposition[i,j,h] = cumulative MSE contribution."""
function _compute_fevd(irfs::Array{T,3}, n::Int, horizon::Int) where {T<:AbstractFloat}
    decomp, props = zeros(T, n, n, horizon), zeros(T, n, n, horizon)
    mse = zeros(T, n, horizon)

    @inbounds for h in 1:horizon
        for i in 1:n
            total = zero(T)
            for j in 1:n
                prev = h == 1 ? zero(T) : decomp[i, j, h-1]
                decomp[i, j, h] = prev + irfs[h, i, j]^2
                total += decomp[i, j, h]
            end
            mse[i, h] = total
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end
    decomp, props
end

"""FEVD from a possibly rectangular IRF array (n_var × n_shock), e.g. FGLR `r×q`."""
function _compute_fevd_rect(irfs::Array{T,3}, n_var::Int, n_shock::Int, horizon::Int) where {T<:AbstractFloat}
    decomp, props = zeros(T, n_var, n_shock, horizon), zeros(T, n_var, n_shock, horizon)
    @inbounds for h in 1:horizon
        for i in 1:n_var
            total = zero(T)
            for j in 1:n_shock
                prev = h == 1 ? zero(T) : decomp[i, j, h - 1]
                decomp[i, j, h] = prev + irfs[h, i, j]^2
                total += decomp[i, j, h]
            end
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end
    decomp, props
end

"""
    _check_fevd_orthogonality(P, Sigma; method=:cholesky) -> Bool

Check that the impact matrix `P` is Σ-orthonormal (`P*P' ≈ Σ`). The squared-IRF FEVD
accumulation is a proper forecast-error-variance decomposition only when the structural
shocks are orthonormal. Some statistical-identification methods (ICA / heteroskedasticity)
may return a rotation `Q` that is not exactly orthonormal, in which case the returned
proportions do not sum to one across shocks. Warns (once) when the invariant fails;
use a generalized (Pesaran-Shin 1998) FEVD for genuinely non-orthogonal identifications.
"""
function _check_fevd_orthogonality(P::AbstractMatrix{T}, Sigma::AbstractMatrix{T};
                                   method::Symbol=:cholesky) where {T<:AbstractFloat}
    n = size(P, 1)
    resid = norm(P * P' - Sigma)
    tol = sqrt(eps(T)) * n * norm(Sigma)
    ok = resid <= tol
    ok || @warn string("fevd: impact matrix for method=:", method,
        " is not orthonormal in the Σ-metric (‖PP′−Σ‖=", resid, " > tol=", tol,
        "); the returned proportions are NOT a proper forecast-error-variance ",
        "decomposition. Use a generalized (Pesaran–Shin 1998) FEVD instead.") maxlog = 1
    return ok
end

# =============================================================================
# Bayesian FEVD
# =============================================================================

"""
    fevd(post::BVARPosterior, horizon; quantiles=[0.16, 0.5, 0.84], ...) -> BayesianFEVD

Compute Bayesian FEVD from posterior draws with posterior quantiles.

# Methods
`:cholesky`, `:sign`, `:narrative`, `:long_run`,
`:fastica`, `:jade`, `:sobi`, `:dcov`, `:hsic`,
`:student_t`, `:mixture_normal`, `:pml`, `:skew_normal`, `:nongaussian_ml`,
`:markov_switching`, `:garch`, `:smooth_transition`, `:external_volatility`

Note: `:smooth_transition` requires `transition_var` kwarg.
      `:external_volatility` requires `regime_indicator` kwarg.

Uses `process_posterior_samples` and `compute_posterior_quantiles` from bayesian_utils.jl.
"""
# =============================================================================
# Structural LP FEVD — see lp_fevd.jl (Gorodnichenko & Lee 2019)
# =============================================================================

function fevd(post::BVARPosterior, horizon::Int;
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
        rng, n_rot, cw, n_nar = _arias_posterior_kwargs(max_draws; kwargs...)
        use_data_a = isempty(data) ? nothing : data
        rset = _arias_from_bvar_posterior(post, restrictions, horizon;
            data=use_data_a, n_rotations=n_rot, quantiles=quantiles,
            rng=rng, compute_weights=cw, n_narrative_sims=n_nar)
        fv = fevd(rset; quantiles=collect(quantiles))
        shock_names === nothing && return fv
        return BayesianFEVD{ET}(fv.quantiles, fv.point_estimate, fv.horizon,
            fv.variables, shock_names, fv.quantile_levels,
            fv.n_requested, fv.n_effective, fv.n_failed)
    end

    # Process posterior samples - compute FEVD proportions for each
    results, samples = process_posterior_samples(post,
        (m, Q, h) -> begin
            irf_vals = compute_irf(m, Q, h)
            _, props = _compute_fevd(irf_vals, nvars(m), h)
            props  # Returns (n, n, horizon)
        end;
        data=use_data, method=method, horizon=horizon,
        check_func=check_func, narrative_check=narrative_check,
        max_draws=max_draws,
        transition_var=transition_var, regime_indicator=regime_indicator,
        restrictions=restrictions, kwargs...
    )

    # Stack results in FEVD axis order (variable, shock, horizon) — unified with
    # FEVD.proportions / LPFEVD (#527). samples is the leading MC axis for quantiles.
    all_fevds = zeros(ET, samples, n, n, horizon)
    @inbounds for s in 1:samples
        for h in 1:horizon, v in 1:n, sh in 1:n
            all_fevds[s, v, sh, h] = results[s][v, sh, h]
        end
    end

    # Compute quantiles using shared utility → (n, n, horizon, nq) and (n, n, horizon)
    q_vec = ET.(quantiles)
    use_threaded = threaded || (samples * horizon * n * n > 100000)
    fevd_q, fevd_m = compute_posterior_quantiles(all_fevds, q_vec; threaded=use_threaded, central=point_estimate)

    snames = isnothing(shock_names) ? post.varnames : shock_names
    # MC honesty (#244): process_posterior_samples drops non-stationary / unidentified draws.
    n_req = post.n_draws
    BayesianFEVD{ET}(fevd_q, fevd_m, horizon, post.varnames, snames, q_vec,
                     n_req, samples, n_req - samples)
end

# Deprecated wrapper for old (chain, p, n, horizon) signature
function fevd(post::BVARPosterior, p::Int, n::Int, horizon::Int; kwargs...)
    fevd(post, horizon; kwargs...)
end

"""FEVD of a [`BayesianSetIdentifiedSVAR`](@ref) (weighted per-draw decompositions)."""
function fevd(r::BayesianSetIdentifiedSVAR{T}; quantiles::Vector{<:Real}=[0.16, 0.5, 0.84]) where {T<:AbstractFloat}
    n_acc, H, n, _ = size(r.irf_draws)
    all_props = zeros(T, n_acc, n, n, H)
    for i in 1:n_acc
        _, props = _compute_fevd(r.irf_draws[i, :, :, :], n, H)
        all_props[i, :, :, :] = props
    end
    q_vec = T.(quantiles)
    nq = length(q_vec)
    fevd_q = zeros(T, n, n, H, nq)
    fevd_m = zeros(T, n, n, H)
    w = r.weights
    for v in 1:n, sh in 1:n, h in 1:H
        vals = @view all_props[:, v, sh, h]
        fevd_m[v, sh, h] = sum(w .* vals)
        for (qi, q) in enumerate(q_vec)
            fevd_q[v, sh, h, qi] = _weighted_quantile(vals, w, q)
        end
    end
    n_req = n_acc + r.n_unidentified
    BayesianFEVD{T}(fevd_q, fevd_m, H, r.varnames, r.varnames, q_vec,
                    n_req, n_acc, r.n_unidentified)
end

# =============================================================================
# Generalized FEVD — Pesaran & Shin (1998)
# =============================================================================

# Reduced-form MA: `ma_coefficients` in irf.jl (SID-19 kernel merge).

"""
    _generalized_fevd_arrays(Phi, Sigma, horizon; normalize) → (decomp, props, rowsums)

Core Pesaran-Shin accumulation. For variable `i` and shock-variable `j`,

```math
gFEVD_{ij}(H) = \\frac{\\sigma_{jj}^{-1}\\sum_{h=0}^{H-1}\\left(e_i' \\Phi_h \\Sigma e_j\\right)^2}
                      {\\sum_{h=0}^{H-1} e_i' \\Phi_h \\Sigma \\Phi_h' e_i}
```

The denominator is the h-step forecast error variance of variable `i`; the numerator is the
part attributable to a unit shock to variable `j`, scaled by that variable's own error
variance. Nothing is orthogonalized, so the result does not depend on the variable ordering.
"""
function _generalized_fevd_arrays(Phi::Vector{Matrix{T}}, Sigma::AbstractMatrix{T},
                                  horizon::Int; normalize::Bool=false) where {T<:AbstractFloat}
    n = size(Sigma, 1)
    decomp = zeros(T, n, n, horizon)
    props = zeros(T, n, n, horizon)
    rowsums = zeros(T, n, horizon)

    num = zeros(T, n, n)          # running Σ_h (e_i' Φ_h Σ e_j)^2
    den = zeros(T, n)             # running Σ_h e_i' Φ_h Σ Φ_h' e_i
    sig_jj = T[Sigma[j, j] for j in 1:n]

    @inbounds for h in 1:horizon
        PS = Phi[h] * Sigma                       # n×n, row i = e_i' Φ_h Σ
        for i in 1:n
            for j in 1:n
                num[i, j] += PS[i, j]^2
            end
            den[i] += dot(@view(PS[i, :]), @view(Phi[h][i, :]))   # e_i' Φ_h Σ Φ_h' e_i
        end
        for i in 1:n
            for j in 1:n
                decomp[i, j, h] = num[i, j] / sig_jj[j]
            end
            total_var = den[i]
            if total_var > 0
                for j in 1:n
                    props[i, j, h] = decomp[i, j, h] / total_var
                end
            end
            rowsums[i, h] = sum(@view props[i, :, h])
        end
    end

    if normalize
        @inbounds for h in 1:horizon, i in 1:n
            s = rowsums[i, h]
            s > 0 && (props[i, :, h] ./= s)
        end
    end
    return decomp, props, rowsums
end

"""
    generalized_fevd(model::VARModel, horizon; normalize=false, shock_names=nothing) → FEVD

Generalized forecast-error variance decomposition (Pesaran & Shin 1998).

The structural [`fevd`](@ref) accumulates squared *orthogonalized* IRFs, so it is a proper
variance decomposition only when the impact matrix satisfies `PP' = Σ`, and under a Cholesky
identification it depends on the variable ordering. The generalized decomposition instead
uses the reduced-form `Σ` directly: it needs no orthogonalization and is **invariant to the
variable ordering**, which makes it the standard choice when the ordering is arbitrary or a
structural identification is unavailable.

The price is that the generalized shocks are correlated, so the shares of a given variable
**do not sum to one**.

# Keyword Arguments
- `normalize::Bool = false` — rescale each row to sum to one. This is the common applied
  convention (and what the Diebold-Yilmaz connectedness literature uses), but it is a
  **convention, not an identity**: the raw shares genuinely do not decompose the variance
  into exclusive parts. The unnormalized row sums are recoverable from the returned
  `decomposition` and the model's own forecast error variance.
- `shock_names` — labels for the shock dimension (defaults to the variable names, since a
  "shock" here is a shock to a *variable*, not to an orthogonalized structural factor)

# Returns
An [`FEVD`](@ref); `proportions[i, j, h]` is the share of the `h`-step forecast error
variance of variable `i` attributable to a shock to variable `j`.

# Examples
```julia
g = generalized_fevd(model, 20)                  # raw shares, rows need not sum to 1
gn = generalized_fevd(model, 20; normalize=true) # rows sum to 1 by construction
```

See also [`fevd`](@ref).

# References
- Pesaran, H. H., & Shin, Y. (1998). Generalized impulse response analysis in linear
  multivariate models. *Economics Letters*, 58(1), 17–29.
"""
function generalized_fevd(model::VARModel{T}, horizon::Int;
                          normalize::Bool=false,
                          shock_names::Union{Nothing,Vector{String}}=nothing) where {T<:AbstractFloat}
    horizon > 0 || throw(ArgumentError("horizon must be positive, got $horizon"))
    _validate_data(model.Sigma, "Sigma")
    _validate_data(model.B, "B")
    n = nvars(model)
    Phi = ma_coefficients(model.B, n, model.p, horizon)
    decomp, props, _ = _generalized_fevd_arrays(Phi, model.Sigma, horizon; normalize=normalize)
    snames = isnothing(shock_names) ? model.varnames : shock_names
    return FEVD{T}(decomp, props, model.varnames, snames)
end

"""
    generalized_fevd(post::BVARPosterior, horizon; normalize=false, quantiles=[0.16,0.5,0.84],
                     point_estimate=:mean, shock_names=nothing, max_draws=1000,
                     threaded=false) → BayesianFEVD

Bayesian generalized FEVD (Pesaran & Shin 1998) with posterior credible bands.

Each posterior draw of `(B, Σ)` gives one generalized decomposition; the reported bands are
posterior quantiles across draws. Because the decomposition needs no orthogonalization, the
`method`/`check_func` machinery of the structural [`fevd`](@ref) does not apply — every draw
contributes, and there is no rotation to accept or reject.

See [`generalized_fevd(::VARModel, ::Int)`](@ref) for the estimand and for what `normalize`
does and does not mean.
"""
function generalized_fevd(post::BVARPosterior, horizon::Int;
                          normalize::Bool=false,
                          quantiles::Vector{<:Real}=[0.16, 0.5, 0.84],
                          point_estimate::Symbol=:mean,
                          shock_names::Union{Nothing,Vector{String}}=nothing,
                          max_draws::Int=1000, threaded::Bool=false)
    horizon > 0 || throw(ArgumentError("horizon must be positive, got $horizon"))
    n = post.n
    p = post.p
    ET = eltype(post.Sigma_draws)
    n_draws = size(post.B_draws, 1)
    n_use = min(n_draws, max_draws)

    # Stack in (variable, shock, horizon) order — unified with FEVD (#527)
    all_fevds = zeros(ET, n_use, n, n, horizon)
    @inbounds for s in 1:n_use
        # Posterior draws are stored with the DRAW on the first axis:
        # `B_draws[s, :, :]` is (1+n*p) x n, matching `VARModel.B`, which is what
        # `extract_ar_coefficients` expects. No transpose.
        B_s = Matrix{ET}(@view post.B_draws[s, :, :])
        S_s = Matrix{ET}(@view post.Sigma_draws[s, :, :])
        Phi = ma_coefficients(B_s, n, p, horizon)
        _, props, _ = _generalized_fevd_arrays(Phi, S_s, horizon; normalize=normalize)
        for h in 1:horizon, v in 1:n, sh in 1:n
            all_fevds[s, v, sh, h] = props[v, sh, h]
        end
    end

    q_vec = ET.(quantiles)
    use_threaded = threaded || (n_use * horizon * n * n > 100000)
    fevd_q, fevd_m = compute_posterior_quantiles(all_fevds, q_vec;
                                                 threaded=use_threaded, central=point_estimate)
    snames = isnothing(shock_names) ? post.varnames : shock_names
    return BayesianFEVD{ET}(fevd_q, fevd_m, horizon, post.varnames, snames, q_vec,
                            n_draws, n_use, n_draws - n_use)
end
