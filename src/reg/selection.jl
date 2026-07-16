# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Automated variable selection on top of the OLS regression stack (EV-04, #412):
stepwise (forward / backward / bidirectional) by p-value or AIC/BIC, exhaustive
best-subset for small candidate sets, and the LSE general-to-specific (GETS /
Autometrics-style) multi-path backward reduction with a misspecification gate.

All routines wrap [`estimate_reg`](@ref); the final selected model is refit via
`estimate_reg` so downstream `report`/`predict`/`refs` work unchanged.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Internal OLS kernel — lightweight refit for the search (no vcov machinery)
# =============================================================================

# Plain OLS fit of `y` on the columns `cols` of `X`. Returns the SSR, the
# residuals, the coefficient vector, and the (classical) coefficient p-values.
# A refit-per-move is intentional: correctness lives in the *selected set*, not
# in shaving flops (QR-updating is an optional optimisation — issue #412 note).
function _sel_ols(y::Vector{T}, X::Matrix{T}, cols::Vector{Int}) where {T<:AbstractFloat}
    Xs = X[:, cols]
    n = length(y)
    k = length(cols)
    XtXinv = robust_inv(Xs' * Xs)
    beta = XtXinv * (Xs' * y)
    resid = y .- Xs * beta
    ssr = dot(resid, resid)
    # Classical coefficient p-values (two-sided t): needed for the :pvalue rule
    # and for the GETS least-significant deletion order.
    dfr = n - k
    pvals = fill(one(T), k)
    if dfr >= 1 && ssr > zero(T)
        sigma2 = ssr / T(dfr)
        se = sqrt.(max.(sigma2 .* diag(XtXinv), zero(T)))
        for j in 1:k
            if se[j] > zero(T)
                tstat = abs(beta[j] / se[j])
                pvals[j] = T(2) * (one(T) - cdf(TDist(dfr), tstat))
            end
        end
    end
    (ssr = ssr, resid = resid, beta = beta, pvals = pvals, k = k, n = n)
end

# AIC / BIC matching `estimate_reg`'s Gaussian-MLE convention exactly (so the
# search IC and the final `RegModel`'s IC fields agree, and best-subset ties
# resolve identically). +1 df for the estimated σ².
function _sel_ic(ssr::T, n::Int, k::Int, which::Symbol) where {T<:AbstractFloat}
    sigma2 = max(ssr / T(n), T(1e-300))
    loglik = -T(n) / 2 * log(T(2) * T(pi)) - T(n) / 2 * log(sigma2) - T(n) / 2
    which === :bic ? -2 * loglik + log(T(n)) * T(k + 1) :
                     -2 * loglik + 2 * T(k + 1)          # :aic (also default)
end

# Columns of `X` that are (numerically) constant — treated as intercepts and
# always forced into the model. Mirrors the `vif`/`_nonintercept_cols` rule.
function _intercept_cols(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    k = size(X, 2)
    findall(j -> all(==(X[1, j]), @view X[:, j]), 1:k)
end

# Misspecification gate for GETS: a model passes when it shows neither
# significant residual serial correlation (Breusch–Godfrey LM, EV-31) nor
# significant non-normality (Jarque–Bera). Both diagnostic p-values must exceed
# `level` (fail to reject a well-specified model).
function _sel_passes_diag(resid::Vector{T}, X::Matrix{T}, cols::Vector{Int},
                          level::Real; bg_lags::Int=1) where {T<:AbstractFloat}
    n = length(resid)
    # Breusch–Godfrey serial-correlation LM (consumes EV-31's implementation).
    bg_ok = true
    if bg_lags >= 1 && bg_lags < n
        bg = breusch_godfrey_test(resid, X[:, cols]; lags=bg_lags)
        bg_ok = bg.pvalue > T(level)
    end
    # Jarque–Bera normality on the residual series (teststat/normality.jl).
    jb = jarque_bera_test(reshape(resid, :, 1); method=:component)
    jb_ok = jb.pvalue > level
    bg_ok && jb_ok
end

# =============================================================================
# Stepwise search
# =============================================================================

# Forward selection. `keep` are always-in columns; `pool` the addable candidates.
function _forward_select(y, X, keep, pool, criterion, p_enter, path)
    T = eltype(y)
    cur = copy(keep)
    remaining = copy(pool)
    base_ic = _sel_ic(_sel_ols(y, X, cur).ssr, length(y), length(cur), criterion)
    while !isempty(remaining)
        best_j = 0
        best_val = criterion === :pvalue ? T(Inf) : base_ic
        for j in remaining
            fit = _sel_ols(y, X, vcat(cur, j))
            if criterion === :pvalue
                # p-value of the just-added candidate (last coefficient).
                pj = fit.pvals[end]
                if pj < best_val
                    best_val = pj; best_j = j
                end
            else
                ic = _sel_ic(fit.ssr, fit.n, fit.k, criterion)
                if ic < best_val - eps(T)
                    best_val = ic; best_j = j
                end
            end
        end
        if criterion === :pvalue
            (best_j == 0 || best_val >= p_enter) && break
        else
            best_j == 0 && break
            base_ic = best_val
        end
        push!(cur, best_j)
        deleteat!(remaining, findfirst(==(best_j), remaining))
        push!(path, (:enter, best_j, T(best_val)))
    end
    sort!(cur)
    cur
end

# Backward elimination. Starts from the full model (keep ∪ pool).
function _backward_select(y, X, keep, pool, criterion, p_remove, path)
    T = eltype(y)
    cur = sort(vcat(keep, pool))
    base_ic = _sel_ic(_sel_ols(y, X, cur).ssr, length(y), length(cur), criterion)
    while true
        removable = setdiff(cur, keep)
        isempty(removable) && break
        if criterion === :pvalue
            fit = _sel_ols(y, X, cur)
            # p-values line up with `cur`; consider only removable columns.
            worst_j = 0; worst_p = zero(T)
            for (idx, c) in enumerate(cur)
                c in keep && continue
                if fit.pvals[idx] > worst_p
                    worst_p = fit.pvals[idx]; worst_j = c
                end
            end
            (worst_j == 0 || worst_p <= p_remove) && break
            deleteat!(cur, findfirst(==(worst_j), cur))
            push!(path, (:remove, worst_j, T(worst_p)))
        else
            best_j = 0; best_val = base_ic
            for j in removable
                trial = filter(!=(j), cur)
                fit = _sel_ols(y, X, trial)
                ic = _sel_ic(fit.ssr, fit.n, fit.k, criterion)
                if ic < best_val - eps(T)
                    best_val = ic; best_j = j
                end
            end
            best_j == 0 && break
            base_ic = best_val
            deleteat!(cur, findfirst(==(best_j), cur))
            push!(path, (:remove, best_j, T(best_val)))
        end
    end
    sort!(cur)
    cur
end

# Bidirectional: alternate one add and one drop, taking the single best move at
# each pass. For :pvalue we require p_remove ≥ p_enter (a just-entered variable
# has p < p_enter ≤ p_remove, so it cannot be immediately removed) — this, plus
# the iteration cap, rules out cycling. For IC every accepted move strictly
# lowers the criterion, so termination is guaranteed at a local optimum.
function _bidirectional_select(y, X, keep, pool, criterion, p_enter, p_remove, path)
    T = eltype(y)
    cur = copy(keep)
    maxiter = 4 * (length(pool) + 1) + 10
    for _ in 1:maxiter
        changed = false
        remaining = setdiff(pool, cur)
        # ---- forward move ----
        if !isempty(remaining)
            base_ic = _sel_ic(_sel_ols(y, X, cur).ssr, length(y), length(cur), criterion)
            best_j = 0; best_val = criterion === :pvalue ? T(Inf) : base_ic
            for j in remaining
                fit = _sel_ols(y, X, vcat(cur, j))
                if criterion === :pvalue
                    pj = fit.pvals[end]
                    (pj < best_val) && (best_val = pj; best_j = j)
                else
                    ic = _sel_ic(fit.ssr, fit.n, fit.k, criterion)
                    (ic < best_val - eps(T)) && (best_val = ic; best_j = j)
                end
            end
            accept = criterion === :pvalue ? (best_j != 0 && best_val < p_enter) : (best_j != 0)
            if accept
                push!(cur, best_j); sort!(cur)
                push!(path, (:enter, best_j, T(best_val)))
                changed = true
            end
        end
        # ---- backward move ----
        removable = setdiff(cur, keep)
        if !isempty(removable)
            fit = _sel_ols(y, X, cur)
            if criterion === :pvalue
                worst_j = 0; worst_p = zero(T)
                for (idx, c) in enumerate(cur)
                    c in keep && continue
                    (fit.pvals[idx] > worst_p) && (worst_p = fit.pvals[idx]; worst_j = c)
                end
                if worst_j != 0 && worst_p > p_remove
                    deleteat!(cur, findfirst(==(worst_j), cur))
                    push!(path, (:remove, worst_j, T(worst_p)))
                    changed = true
                end
            else
                base_ic = _sel_ic(fit.ssr, fit.n, fit.k, criterion)
                best_j = 0; best_val = base_ic
                for j in removable
                    f2 = _sel_ols(y, X, filter(!=(j), cur))
                    ic = _sel_ic(f2.ssr, f2.n, f2.k, criterion)
                    (ic < best_val - eps(T)) && (best_val = ic; best_j = j)
                end
                if best_j != 0
                    deleteat!(cur, findfirst(==(best_j), cur))
                    push!(path, (:remove, best_j, T(best_val)))
                    changed = true
                end
            end
        end
        changed || break
    end
    sort!(cur)
    cur
end

# =============================================================================
# Best-subset (exhaustive) — small candidate sets only
# =============================================================================

function _best_subset(y, X, keep, pool, criterion, path)
    T = eltype(y)
    length(pool) <= 20 ||
        throw(ArgumentError("best-subset is exhaustive; restrict to ≤ 20 candidate columns (got $(length(pool)))"))
    best_cols = sort(copy(keep))
    best_ic = _sel_ic(_sel_ols(y, X, best_cols).ssr, length(y), length(best_cols), criterion)
    m = length(pool)
    for mask in 0:(2^m - 1)
        subset = Int[pool[b] for b in 1:m if (mask >> (b - 1)) & 1 == 1]
        cols = sort(vcat(keep, subset))
        length(cols) < length(y) || continue
        fit = _sel_ols(y, X, cols)
        ic = _sel_ic(fit.ssr, fit.n, fit.k, criterion)
        if ic < best_ic - eps(T)
            best_ic = ic; best_cols = cols
        end
    end
    push!(path, (:best_subset, 0, T(best_ic)))
    best_cols
end

# =============================================================================
# GETS — LSE general-to-specific multi-path backward reduction
# =============================================================================

# Single-path backward reduction from `start_cols`, deleting the least
# significant removable term at each node provided the resulting model still
# passes the misspecification gate. Returns the terminal (irreducible,
# diagnostic-passing) column set.
function _gets_path(y, X, keep, start_cols, p_gets, diag_level, bg_lags)
    T = eltype(y)
    cols = sort(copy(start_cols))
    while true
        removable = setdiff(cols, keep)
        isempty(removable) && break
        fit = _sel_ols(y, X, cols)
        # Deletion candidates: insignificant terms, least-significant first.
        order = sort(collect(enumerate(cols)); by = ic -> fit.pvals[ic[1]], rev = true)
        removed = false
        for (idx, c) in order
            c in keep && continue
            fit.pvals[idx] > p_gets || break   # remaining are all significant
            trial = filter(!=(c), cols)
            tf = _sel_ols(y, X, trial)
            if _sel_passes_diag(tf.resid, X, trial, diag_level; bg_lags=bg_lags)
                cols = trial; removed = true; break
            end
        end
        removed || break
    end
    sort!(cols)
    cols
end

function _gets_select(y, X, keep, pool, criterion, p_gets, diag_level, bg_lags, path)
    T = eltype(y)
    gum = sort(vcat(keep, pool))
    gum_fit = _sel_ols(y, X, gum)
    # Insignificant regressors in the GUM seed the distinct deletion paths.
    insig = Int[]
    for (idx, c) in enumerate(gum)
        (c in keep) && continue
        gum_fit.pvals[idx] > p_gets && push!(insig, c)
    end
    terminals = Vector{Vector{Int}}()
    if isempty(insig)
        push!(terminals, gum)                       # GUM already parsimonious
    else
        for r0 in insig
            trial = filter(!=(r0), gum)
            tf = _sel_ols(y, X, trial)
            start = _sel_passes_diag(tf.resid, X, trial, diag_level; bg_lags=bg_lags) ?
                    trial : gum
            term = _gets_path(y, X, keep, start, p_gets, diag_level, bg_lags)
            (term in terminals) || push!(terminals, term)
        end
    end
    isempty(terminals) && push!(terminals, gum)
    # Final model: the terminal with the best (smallest) information criterion,
    # ties broken toward the more parsimonious model (getsm `tie.breaking` = IC).
    best = terminals[1]
    best_ic = _sel_ic(_sel_ols(y, X, best).ssr, length(y), length(best), criterion)
    for term in terminals[2:end]
        ic = _sel_ic(_sel_ols(y, X, term).ssr, length(y), length(term), criterion)
        if ic < best_ic - eps(T) || (abs(ic - best_ic) <= eps(T) && length(term) < length(best))
            best = term; best_ic = ic
        end
    end
    for c in setdiff(best, keep)
        push!(path, (:retain, c, T(best_ic)))
    end
    (sort(best), terminals, gum)
end

# =============================================================================
# Public API
# =============================================================================

"""
    select_variables(y, X; method=:bidirectional, criterion=:pvalue, kwargs...) -> SelectionResult{T}

Automated regressor selection on top of OLS. Searches over the columns of the
general unrestricted model (GUM) `X` and returns a [`SelectionResult`](@ref)
holding the search path, the final selected column set, and a refit
[`RegModel`](@ref) (so `report`/`predict`/`refs` work on `result.final`).

# Methods
- `:forward` — greedily add the candidate that most improves the criterion.
- `:backward` — start from the GUM and greedily drop the weakest term.
- `:bidirectional` (default) — alternate add/drop moves to a local optimum.
- `:best_subset` — exhaustive search over all subsets (requires an IC criterion;
  `≤ 20` candidate columns).
- `:gets` — LSE general-to-specific multi-path backward reduction (Hoover &
  Perez 1999; Hendry & Krolzig 2005). Each path seeds on a different
  insignificant GUM regressor and deletes the least-significant term at each
  node, subject to a misspecification gate (Breusch–Godfrey serial-correlation
  LM + Jarque–Bera normality). The final model is the diagnostic-passing
  terminal with the best information criterion; a parsimonious-encompassing
  F-test of the selection against the GUM is stored.

# Arguments
- `criterion::Symbol` — `:pvalue` (default), `:aic`, or `:bic`. `:pvalue` adds
  candidates with a coefficient p-value below `p_enter` and drops those above
  `p_remove`; `:aic`/`:bic` move in the direction of best information-criterion
  improvement. `:best_subset` and `:gets` ignore `:pvalue` for their final
  tie-break and use `:bic` by default when `criterion` is `:pvalue`.
- `p_enter::Real=0.05`, `p_remove::Real=0.10` — stepwise entry/removal levels
  (require `p_remove ≥ p_enter` for `:bidirectional` to preclude cycling).
- `p_gets::Real=0.05` — GETS deletion significance level.
- `diag_level::Real=0.05` — GETS misspecification-gate level (a model passes
  when both diagnostic p-values exceed it).
- `bg_lags::Int=1` — Breusch–Godfrey lag order for the GETS gate.
- `keep` — extra column indices to force into every model (intercept columns,
  detected as numerically constant, are always kept).
- `varnames` — names for the GUM columns (auto `x1…xk` otherwise).

# Post-selection inference
Standard errors, t-statistics, and p-values on `result.final` are **conditional
on the selected specification** and ignore the search that produced it; they are
not valid unconditional inference. For a shrinkage alternative with a single
tuning path see [`estimate_lasso`](@ref) (EV-03).

# References
- Hoover, K. D. & Perez, S. J. (1999). *Econometrics Journal* 2(2), 167-191.
- Hendry, D. F. & Krolzig, H.-M. (2005). *Economic Journal* 115(502), C32-C61.
- Pretis, F., Reade, J. J. & Sucarrat, G. (2018). *J. Statistical Software* 86(3).
"""
function select_variables(y::AbstractVector{T}, X::AbstractMatrix{T};
                          method::Symbol=:bidirectional,
                          criterion::Symbol=:pvalue,
                          p_enter::Real=0.05, p_remove::Real=0.10,
                          p_gets::Real=0.05, diag_level::Real=0.05, bg_lags::Int=1,
                          keep::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                          varnames::Union{Nothing,Vector{String}}=nothing) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_data(X, "X")
    n, k = size(X)
    length(y) == n || throw(ArgumentError("X must have $(length(y)) rows (got $n)"))
    method in (:forward, :backward, :bidirectional, :best_subset, :gets) ||
        throw(ArgumentError("method must be :forward, :backward, :bidirectional, :best_subset, or :gets; got :$method"))
    criterion in (:pvalue, :aic, :bic) ||
        throw(ArgumentError("criterion must be :pvalue, :aic, or :bic; got :$criterion"))

    ym = Vector{T}(y)
    Xm = Matrix{T}(X)
    vn = something(varnames, ["x$i" for i in 1:k])
    length(vn) == k || throw(ArgumentError("varnames must have length $k"))

    # Forced-keep set: detected intercepts ∪ user `keep`.
    keepset = _intercept_cols(Xm)
    if keep !== nothing
        all(c -> 1 <= c <= k, keep) || throw(ArgumentError("keep indices must be in 1:$k"))
        keepset = sort(union(keepset, Int.(keep)))
    end
    pool = setdiff(1:k, keepset)

    # best_subset / gets tie-break on an IC — default to :bic if the caller left
    # criterion at :pvalue (p-values do not define a subset ordering).
    ic_crit = criterion === :pvalue ? :bic : criterion

    path = Vector{Tuple{Symbol,Int,T}}()
    terminals = Vector{Vector{Int}}()
    enc_f = nothing; enc_p = nothing; enc_df = nothing
    gum = sort(vcat(keepset, pool))

    if method === :forward
        selected = _forward_select(ym, Xm, keepset, pool, criterion, T(p_enter), path)
    elseif method === :backward
        selected = _backward_select(ym, Xm, keepset, pool, criterion, T(p_remove), path)
    elseif method === :bidirectional
        criterion === :pvalue && p_remove < p_enter &&
            throw(ArgumentError("bidirectional :pvalue search requires p_remove ≥ p_enter"))
        selected = _bidirectional_select(ym, Xm, keepset, pool, criterion, T(p_enter), T(p_remove), path)
    elseif method === :best_subset
        selected = _best_subset(ym, Xm, keepset, pool, ic_crit, path)
    else  # :gets
        selected, terminals, gum = _gets_select(ym, Xm, keepset, pool, ic_crit,
                                                 T(p_gets), T(diag_level), bg_lags, path)
    end

    # Parsimonious-encompassing F-test of the selected model against the GUM
    # (nested: selected ⊆ GUM). Stored for :gets; also computed for stepwise.
    if length(selected) < length(gum)
        sf = _sel_ols(ym, Xm, selected)
        gf = _sel_ols(ym, Xm, gum)
        q = length(gum) - length(selected)
        df2 = n - length(gum)
        if df2 >= 1 && gf.ssr > zero(T)
            fstat = ((sf.ssr - gf.ssr) / T(q)) / (gf.ssr / T(df2))
            enc_f = fstat
            enc_p = T(1 - cdf(FDist(q, df2), fstat))
            enc_df = (q, df2)
        end
    end

    final = estimate_reg(ym, Xm[:, selected]; cov_type=:ols, varnames=vn[selected])

    SelectionResult{T}(method, criterion, selected, keepset, vn, path, terminals,
                       enc_f, enc_p, enc_df, final, length(gum))
end

# Float fallback.
function select_variables(y::AbstractVector, X::AbstractMatrix; kwargs...)
    select_variables(Float64.(y), Float64.(X); kwargs...)
end
