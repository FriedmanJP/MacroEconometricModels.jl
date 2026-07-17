# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Nonlinear ARDL (NARDL) of Shin, Yu & Greenwood-Nimmo (2014). Each asymmetric
regressor is decomposed into positive/negative partial sums, the enlarged design
is estimated by the reused EV-08 `estimate_ardl` machinery, and asymmetric
long-run coefficients (θ⁺/θ⁻), long- and short-run symmetry Wald tests, and
cumulative dynamic multipliers (m⁺_h/m⁻_h with a recursive-design residual
bootstrap) are recovered. The Pesaran–Shin–Smith bounds test is reused verbatim
on the enlarged specification — an asymmetric regressor contributes **two**
columns to the bounds-table `k`.
"""

using LinearAlgebra, Statistics, Random

# =============================================================================
# Partial-sum decomposition
# =============================================================================

"""
    _partial_sums(x) -> (xpos, xneg)

Positive/negative partial sums of a level series `x`:
`x⁺_t = Σ_{s≤t} max(Δx_s, 0)`, `x⁻_t = Σ_{s≤t} min(Δx_s, 0)`, with
`x⁺_1 = x⁻_1 = 0`. By construction `x_t = x_1 + x⁺_t + x⁻_t` (exact to machine
precision), so the partial sums are cumulated **levels** — they are I(1) like `x`
and must enter the ARDL undifferenced.
"""
function _partial_sums(x::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(x)
    xpos = zeros(T, N)
    xneg = zeros(T, N)
    @inbounds for t in 2:N
        dx = x[t] - x[t-1]
        xpos[t] = xpos[t-1] + max(dx, zero(T))
        xneg[t] = xneg[t-1] + min(dx, zero(T))
    end
    (xpos, xneg)
end

# =============================================================================
# Public estimator
# =============================================================================

"""
    estimate_nardl(y, X; asymmetric=:all, p=:auto, q=:auto, max_p=4, max_q=4,
                   ic=:aic, case=3, xnames=nothing, yname="y") -> NARDLModel{T}

Estimate the nonlinear ARDL (NARDL) of Shin, Yu & Greenwood-Nimmo (2014).

Each regressor selected by `asymmetric` is split into its positive/negative
partial sums `x⁺, x⁻` (levels, cumulated from `Δx`; see [`_partial_sums`](@ref)),
and the pair replaces the original column in the ARDL design. The enlarged design
is handed to the reused EV-08 [`estimate_ardl`](@ref); lag orders, long-run
coefficients, the ECM re-parameterisation and the [`bounds_test`](@ref) then all
operate on the split regressors.

# Arguments
- `y::AbstractVector` — dependent variable (length `N`).
- `X::AbstractMatrix` — `N × k₀` distributed-lag regressors (no intercept column);
  a plain `AbstractVector` is treated as a single regressor.

# Keywords
- `asymmetric=:all` — `:all` splits every column; a vector of column indices splits
  only those (symmetric columns pass through unchanged).
- `p`, `q`, `max_p`, `max_q`, `ic`, `case` — forwarded to [`estimate_ardl`](@ref) on the
  **enlarged** design. IC lag selection therefore searches over the split regressors.
- `xnames`, `yname` — labels; enlarged labels append `"_POS"` / `"_NEG"`.

# Returns
`NARDLModel{T}` (`T = float(eltype(y))`) wrapping the ARDL fit, the enlarged-`k`
bounds test, the partial-sum design, and the split bookkeeping.

# References
- Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). *Modelling Asymmetric Cointegration
  and Dynamic Multipliers in a Nonlinear ARDL Framework.* Springer.
"""
function estimate_nardl(y::AbstractVector{T}, X::AbstractMatrix{T};
                        asymmetric::Union{Symbol,AbstractVector{<:Integer}}=:all,
                        p::Union{Symbol,Integer}=:auto,
                        q::Union{Symbol,Integer,AbstractVector}=:auto,
                        max_p::Int=4, max_q::Int=4, ic::Symbol=:aic,
                        case::Int=3,
                        xnames::Union{Nothing,Vector{String}}=nothing,
                        yname::AbstractString="y") where {T<:AbstractFloat}
    N, k0 = size(X)
    length(y) == N || throw(ArgumentError("y has length $(length(y)); X has $N rows"))
    vnames = xnames === nothing ? ["x$j" for j in 1:k0] : xnames
    length(vnames) == k0 || throw(ArgumentError("xnames must have length $k0"))

    # ---- resolve which columns are asymmetric ----
    if asymmetric === :all
        asym = collect(1:k0)
    else
        asym = sort(unique(collect(Int, asymmetric)))
        all(j -> 1 <= j <= k0, asym) ||
            throw(ArgumentError("asymmetric indices must lie in 1:$k0; got $asym"))
    end
    isempty(asym) &&
        throw(ArgumentError("NARDL needs at least one asymmetric regressor; use estimate_ardl for a symmetric model"))

    # ---- build the enlarged (split) design ----
    cols = Vector{Vector{T}}()
    meta = Tuple{Int,Symbol}[]
    enames = String[]
    for j in 1:k0
        if j in asym
            xp, xn = _partial_sums(@view X[:, j])
            push!(cols, xp); push!(meta, (j, :pos)); push!(enames, vnames[j] * "_POS")
            push!(cols, xn); push!(meta, (j, :neg)); push!(enames, vnames[j] * "_NEG")
        else
            push!(cols, T.(collect(@view X[:, j]))); push!(meta, (j, :sym))
            push!(enames, vnames[j])
        end
    end
    Xsplit = reduce(hcat, cols)
    kk = size(Xsplit, 2)

    # ---- reuse EV-08 estimation on the enlarged design ----
    ardl = estimate_ardl(y, Xsplit; p=p, q=q, max_p=max_p, max_q=max_q,
                         ic=ic, case=case, xnames=enames, yname=String(yname))

    # ---- bounds test: k = enlarged count (asymmetric regressor counted twice) ----
    bt = bounds_test(ardl; case=case)

    NARDLModel{T}(ardl, bt, T.(collect(y)), T.(collect(X)), Xsplit, asym, meta,
                  k0, kk, vnames, enames, String(yname))
end

estimate_nardl(y::AbstractVector, X::AbstractMatrix; kwargs...) =
    estimate_nardl(float.(collect(y)), float.(collect(X)); kwargs...)
estimate_nardl(y::AbstractVector, x::AbstractVector; kwargs...) =
    estimate_nardl(y, reshape(collect(x), :, 1); kwargs...)

# =============================================================================
# Enlarged-column lookup helpers
# =============================================================================

"""Return the enlarged-regressor index (into `m.ardl.q` / `m.meta`) of original
regressor `orig` with the given `kind ∈ (:pos, :neg, :sym)`, or `0` if absent."""
function _enlarged_index(m::NARDLModel, orig::Int, kind::Symbol)
    for (e, (o, k)) in enumerate(m.meta)
        (o == orig && k == kind) && return e
    end
    0
end

# =============================================================================
# Long-run asymmetry
# =============================================================================

"""
    long_run(m::NARDLModel) -> ARDLLongRun

Long-run (level) multipliers of the NARDL on the **enlarged** regressor set: one
entry per partial sum, so an asymmetric regressor yields both `θ⁺_j` (label
`…_POS`) and `θ⁻_j` (label `…_NEG`), each with a delta-method standard error.
"""
long_run(m::NARDLModel) = long_run(m.ardl)

"""
    bounds_test(m::NARDLModel) -> ARDLBoundsTest

The cached Pesaran–Shin–Smith bounds test on the enlarged NARDL specification. The
regressor count `k` counts each partial sum separately (an asymmetric regressor
contributes 2), so the tabulated I(0)/I(1) bounds are read at the enlarged `k`.
"""
bounds_test(m::NARDLModel) = m.bounds

# =============================================================================
# Symmetry Wald tests
# =============================================================================

"""Long-run symmetry Wald `χ²(1)` for asymmetric regressor `orig`: `H₀: θ⁺=θ⁻`,
tested by the delta method on `θ⁺−θ⁻ = (Σβ⁺−Σβ⁻)/(1−Σφ)`."""
function _longrun_symmetry_wald(m::NARDLModel{T}, orig::Int) where {T<:AbstractFloat}
    a = m.ardl
    b = a.coef; V = a.vcov; K = a.K
    ep = _enlarged_index(m, orig, :pos)
    en = _enlarged_index(m, orig, :neg)
    (ep == 0 || en == 0) && throw(ArgumentError("regressor $orig is not asymmetric"))
    denom = one(T) - sum(@view b[a.ar_idx])              # 1 − Σφ
    Sp = sum(@view b[a.x_idx[ep]])                       # Σβ⁺
    Sn = sum(@view b[a.x_idx[en]])                       # Σβ⁻
    theta_p = Sp / denom
    theta_n = Sn / denom
    diff = theta_p - theta_n
    # Jacobian g = ∂diff/∂b
    g = zeros(T, K)
    for c in a.x_idx[ep]; g[c] += one(T) / denom; end     # ∂/∂β⁺
    for c in a.x_idx[en]; g[c] -= one(T) / denom; end     # ∂/∂β⁻
    for c in a.ar_idx;    g[c] += (Sp - Sn) / denom^2; end # ∂/∂φ
    var = dot(g, V * g)
    stat = var > zero(T) ? diff^2 / var : zero(T)
    (T(stat), theta_p, theta_n)
end

"""Short-run symmetry Wald `χ²(1)` for asymmetric regressor `orig`:
`H₀: Σ_ℓ π⁺_ℓ = Σ_ℓ π⁻_ℓ` on the ECM differenced-term coefficients. The sum of the
Δ-coefficients of a distributed lag `Σ_{ℓ=0}^q β_ℓ x_{t-ℓ}` equals `−Σ_{ℓ=1}^q ℓ·β_ℓ`
(the `B(1)` term of the `A(L)=A(1)+(1−L)B(L)` decomposition), so the restriction is
linear in the levels coefficients: `Σ_ℓ ℓ·β⁺_ℓ − Σ_ℓ ℓ·β⁻_ℓ = 0`."""
function _shortrun_symmetry_wald(m::NARDLModel{T}, orig::Int) where {T<:AbstractFloat}
    a = m.ardl
    b = a.coef; V = a.vcov; K = a.K
    ep = _enlarged_index(m, orig, :pos)
    en = _enlarged_index(m, orig, :neg)
    r = zeros(T, K)
    # x_idx[ep] holds columns for lags ℓ = 0..q⁺ in order; weight ℓ.
    for (ℓ, c) in enumerate(a.x_idx[ep]); r[c] += T(ℓ - 1); end   # ℓ = 0,1,...
    for (ℓ, c) in enumerate(a.x_idx[en]); r[c] -= T(ℓ - 1); end
    val = dot(r, b)
    var = dot(r, V * r)
    stat = var > zero(T) ? val^2 / var : zero(T)
    T(stat)
end

"""
    symmetry_test(m::NARDLModel) -> NARDLSymmetryTest

Long- and short-run symmetry Wald tests for every asymmetric regressor of a
[`NARDLModel`](@ref).

- **Long-run** `H₀: θ⁺_j = θ⁻_j` — delta-method Wald on the long-run coefficients
  (`θ⁺−θ⁻ = (Σβ⁺−Σβ⁻)/(1−Σφ̂)`; the Jacobian carries the ECM denominator).
- **Short-run** `H₀: Σ_ℓ π⁺_{jℓ} = Σ_ℓ π⁻_{jℓ}` — linear Wald on the ECM
  differenced-term coefficients (`Σ_ℓ ℓ·β⁺_{jℓ} = Σ_ℓ ℓ·β⁻_{jℓ}`).

Each single-restriction statistic is reported as both a `χ²(1)` and an `F(1, n−K)`
with the matching p-value. Rejecting `H₀` is evidence of asymmetric adjustment.
"""
function symmetry_test(m::NARDLModel{T}) where {T<:AbstractFloat}
    asym = m.asym
    na = length(asym)
    lr_stat = zeros(T, na); lr_pc = zeros(T, na); lr_pf = zeros(T, na)
    sr_stat = zeros(T, na); sr_pc = zeros(T, na); sr_pf = zeros(T, na)
    tp = zeros(T, na); tn = zeros(T, na)
    names = String[]
    dof_r = max(m.ardl.n - m.ardl.K, 1)
    for (i, orig) in enumerate(asym)
        push!(names, m.xnames[orig])
        s, thp, thn = _longrun_symmetry_wald(m, orig)
        lr_stat[i] = s; tp[i] = thp; tn[i] = thn
        lr_pc[i] = T(ccdf(Chisq(1), s))
        lr_pf[i] = T(ccdf(FDist(1, dof_r), s))          # F = χ²/1
        ss = _shortrun_symmetry_wald(m, orig)
        sr_stat[i] = ss
        sr_pc[i] = T(ccdf(Chisq(1), ss))
        sr_pf[i] = T(ccdf(FDist(1, dof_r), ss))
    end
    NARDLSymmetryTest{T}(copy(asym), names, lr_stat, lr_pc, lr_pf,
                        sr_stat, sr_pc, sr_pf, tp, tn, 1, dof_r)
end

# =============================================================================
# Cumulative dynamic multipliers
# =============================================================================

"""
    _iterate_multiplier(phi, beta, H) -> g

Response `g_h = ∂y_{t+h}/∂(unit step in x)` of the ARDL difference equation
`y_t = Σφ_i y_{t-i} + Σ_ℓ β_ℓ x_{t-ℓ}` to a unit **permanent** (step) change in a
regressor with lag coefficients `beta` (`0..q`, `beta[ℓ+1]`). `g_h` is the
cumulative dynamic multiplier and converges to `Σβ / (1−Σφ)` as `h → ∞`.
"""
function _iterate_multiplier(phi::AbstractVector{T}, beta::AbstractVector{T},
                             H::Int) where {T<:AbstractFloat}
    p = length(phi)
    q = length(beta) - 1
    g = zeros(T, H + 1)
    @inbounds for h in 0:H
        val = zero(T)
        for i in 1:p
            (h - i) >= 0 && (val += phi[i] * g[h - i + 1])
        end
        for ℓ in 0:q
            h >= ℓ && (val += beta[ℓ + 1])
        end
        g[h + 1] = val
    end
    g
end

"""Extract `(phi, beta⁺, beta⁻)` from an ARDL fit for the enlarged columns of `orig`."""
function _mult_coefs(m::NARDLModel{T}, ardl::ARDLModel{T}, orig::Int) where {T}
    ep = _enlarged_index(m, orig, :pos)
    en = _enlarged_index(m, orig, :neg)
    phi = T[ardl.coef[c] for c in ardl.ar_idx]
    bp = T[ardl.coef[c] for c in ardl.x_idx[ep]]
    bn = T[ardl.coef[c] for c in ardl.x_idx[en]]
    (phi, bp, bn)
end

"""
    _bootstrap_multipliers(m, H, nreps, level, rng) -> bands...

Recursive-design (condition-on-`x`) residual bootstrap for the dynamic-multiplier
bands. Centered residuals are resampled, `y*` is rebuilt recursively from the
fitted dynamics **holding the (split) regressors fixed**, the ARDL is re-estimated
at the **selected** lag orders (`p`, `q` fixed — no re-selection), and the
multipliers are recomputed. Wrapped in `_suppress_warnings`.
"""
function _bootstrap_multipliers(m::NARDLModel{T}, H::Int, nreps::Int,
                                level::Real, rng::AbstractRNG) where {T<:AbstractFloat}
    a = m.ardl
    N = length(m.y)
    L = max(a.p, maximum(a.q))
    row_start = L + 1
    n = a.n
    na = length(m.asym)

    # non-y contribution of each effective-sample row: deterministics + x-lag columns.
    Xd = a.X                                    # n × K levels design
    coef = a.coef
    base = zeros(T, n)
    @inbounds for r in 1:n
        acc = zero(T)
        for c in 1:a.K
            (c in a.ar_idx) && continue
            acc += Xd[r, c] * coef[c]
        end
        base[r] = acc
    end
    phi = T[coef[c] for c in a.ar_idx]
    resid_c = a.residuals .- mean(a.residuals)

    Hp1 = H + 1
    boot_pos = [Matrix{T}(undef, nreps, Hp1) for _ in 1:na]
    boot_neg = [Matrix{T}(undef, nreps, Hp1) for _ in 1:na]
    boot_dif = [Matrix{T}(undef, nreps, Hp1) for _ in 1:na]

    _suppress_warnings() do
        for rep in 1:nreps
            ystar = copy(m.y)
            @inbounds for (r, t) in enumerate(row_start:N)
                acc = base[r] + resid_c[rand(rng, 1:n)]
                for (i, c) in enumerate(a.ar_idx)
                    acc += phi[i] * ystar[t - i]
                end
                ystar[t] = acc
            end
            mb = estimate_ardl(ystar, m.Xsplit; p=a.p, q=a.q, case=a.case,
                               xnames=m.enames, yname=m.yname)
            for (i, orig) in enumerate(m.asym)
                phib, bpb, bnb = _mult_coefs(m, mb, orig)
                gp = _iterate_multiplier(phib, bpb, H)
                gn = _iterate_multiplier(phib, bnb, H)
                @views boot_pos[i][rep, :] .= gp
                @views boot_neg[i][rep, :] .= gn
                @views boot_dif[i][rep, :] .= gp .- gn
            end
        end
    end

    alpha = (one(T) - T(level)) / 2
    pos_lo = zeros(T, na, Hp1); pos_hi = zeros(T, na, Hp1)
    neg_lo = zeros(T, na, Hp1); neg_hi = zeros(T, na, Hp1)
    dif_lo = zeros(T, na, Hp1); dif_hi = zeros(T, na, Hp1)
    for i in 1:na
        for h in 1:Hp1
            pos_lo[i, h] = quantile(@view(boot_pos[i][:, h]), alpha)
            pos_hi[i, h] = quantile(@view(boot_pos[i][:, h]), one(T) - alpha)
            neg_lo[i, h] = quantile(@view(boot_neg[i][:, h]), alpha)
            neg_hi[i, h] = quantile(@view(boot_neg[i][:, h]), one(T) - alpha)
            dif_lo[i, h] = quantile(@view(boot_dif[i][:, h]), alpha)
            dif_hi[i, h] = quantile(@view(boot_dif[i][:, h]), one(T) - alpha)
        end
    end
    (pos_lo, pos_hi, neg_lo, neg_hi, dif_lo, dif_hi)
end

"""
    dynamic_multipliers(m::NARDLModel, H; bootstrap=true, nreps=500, level=0.95,
                        rng=Random.default_rng()) -> NARDLMultipliers

Cumulative dynamic multipliers `m⁺_{j,h}` / `m⁻_{j,h}` for `h = 0…H`: the response
of `y` to a unit permanent change in each asymmetric regressor's positive / negative
partial sum, obtained by recursively iterating the estimated ARDL difference
equation to the shock (holding everything else fixed). They converge to the
long-run θ⁺_j / θ⁻_j.

# Keywords
- `bootstrap::Bool=true` — attach pointwise percentile bands from a recursive-design
  (condition-on-`x`) residual bootstrap.
- `nreps::Int=500` — bootstrap replications (≥ 500 recommended).
- `level::Real=0.95` — band coverage.
- `rng::AbstractRNG` — RNG for the bootstrap (pass a seeded RNG for reproducibility).

# References
- Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014).
"""
function dynamic_multipliers(m::NARDLModel{T}, H::Int; bootstrap::Bool=true,
                             nreps::Int=500, level::Real=0.95,
                             rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    H >= 0 || throw(ArgumentError("H must be ≥ 0; got $H"))
    na = length(m.asym)
    Hp1 = H + 1
    m_pos = zeros(T, na, Hp1); m_neg = zeros(T, na, Hp1); m_dif = zeros(T, na, Hp1)
    tp = zeros(T, na); tn = zeros(T, na)
    lr = long_run(m.ardl)
    for (i, orig) in enumerate(m.asym)
        phi, bp, bn = _mult_coefs(m, m.ardl, orig)
        gp = _iterate_multiplier(phi, bp, H)
        gn = _iterate_multiplier(phi, bn, H)
        m_pos[i, :] .= gp; m_neg[i, :] .= gn; m_dif[i, :] .= gp .- gn
        tp[i] = lr.theta[_enlarged_index(m, orig, :pos)]
        tn[i] = lr.theta[_enlarged_index(m, orig, :neg)]
    end

    if bootstrap && nreps > 0
        pl, ph, nl, nh, dl, dh = _bootstrap_multipliers(m, H, nreps, level, rng)
    else
        z = Matrix{T}(undef, 0, 0)
        pl = ph = nl = nh = dl = dh = z
        nreps = 0
    end

    names = [m.xnames[o] for o in m.asym]
    NARDLMultipliers{T}(collect(0:H), copy(m.asym), names, m_pos, m_neg, m_dif,
                       pl, ph, nl, nh, dl, dh, tp, tn, nreps, T(level))
end

# =============================================================================
# StatsAPI passthroughs
# =============================================================================

StatsAPI.coef(m::NARDLModel) = coef(m.ardl)
StatsAPI.vcov(m::NARDLModel) = vcov(m.ardl)
StatsAPI.residuals(m::NARDLModel) = residuals(m.ardl)
StatsAPI.nobs(m::NARDLModel) = m.ardl.n
StatsAPI.dof(m::NARDLModel) = m.ardl.K
StatsAPI.loglikelihood(m::NARDLModel) = m.ardl.loglik
StatsAPI.aic(m::NARDLModel) = m.ardl.aic
StatsAPI.bic(m::NARDLModel) = m.ardl.bic

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::NARDLModel{T}) where {T}
    a = m.ardl
    qstr = join(a.q, ", ")
    spec = Any[
        "Model"          "NARDL($(a.p); $(qstr))";
        "Dependent"      m.yname;
        "Regressors"     "$(m.k_orig) → $(m.k) (split)";
        "Asymmetric"     join([m.xnames[o] for o in m.asym], ", ");
        "Case"           get(_ARDL_CASE_DESC, a.case, string(a.case));
        "Selection"      a.selected ? uppercase(string(a.ic)) * " grid" : "fixed";
        "Observations"   a.n;
        "Coefficients"   a.K;
        "σ̂²"             _fmt(a.sigma2);
        "AIC"            _fmt(a.aic; digits=2);
        "BIC"            _fmt(a.bic; digits=2)
    ]
    _pretty_table(io, spec; title="Nonlinear ARDL (NARDL)",
                  column_labels=["Specification", ""], alignment=[:l, :r])

    _coef_table(io, "Coefficients (levels form, split regressors)",
                a.coefnames, a.coef, stderror(a); dist=:t, dof_r=max(a.n - a.K, 1))

    lr = long_run(a)
    _coef_table(io, "Asymmetric long-run coefficients (θ⁺ / θ⁻)",
                lr.varnames, lr.theta, lr.se; dist=:z)

    # symmetry tests
    st = symmetry_test(m)
    sym = Matrix{Any}(undef, length(st.reg_names), 6)
    for i in eachindex(st.reg_names)
        sym[i, 1] = st.reg_names[i]
        sym[i, 2] = _fmt(st.lr_stat[i]; digits=3)
        sym[i, 3] = _format_pvalue(st.lr_p_chi2[i])
        sym[i, 4] = _fmt(st.sr_stat[i]; digits=3)
        sym[i, 5] = _format_pvalue(st.sr_p_chi2[i])
        sym[i, 6] = st.lr_p_chi2[i] < 0.10 ? "asymmetric" : "symmetric"
    end
    _pretty_table(io, sym; title="Symmetry Wald tests (χ², H₀: symmetry)",
                  column_labels=["Regressor", "LR χ²", "LR p", "SR χ²", "SR p", "Long-run"],
                  alignment=[:l, :r, :r, :r, :r, :l])

    # bounds test with degenerate-case caveat
    show(io, m.bounds)
    _show_note(io, "NARDL bounds test uses the ENLARGED k=$(m.bounds.k) (each " *
               "asymmetric regressor counts twice). Degenerate-case caveat " *
               "(Shin et al. 2014 / PSS 2001): if the F-test and the t-test on " *
               "the lagged y level disagree — F significant but t insignificant, " *
               "or vice versa — the bounds F can be degenerate; apply the PSS/" *
               "Banerjee check on both the F and the t outcome before concluding.")
    _sig_legend(io)
end

report(m::NARDLModel) = show(stdout, m)
report(io::IO, m::NARDLModel) = show(io, m)

function Base.show(io::IO, st::NARDLSymmetryTest{T}) where {T}
    tbl = Matrix{Any}(undef, length(st.reg_names), 8)
    for i in eachindex(st.reg_names)
        tbl[i, 1] = st.reg_names[i]
        tbl[i, 2] = _fmt(st.theta_pos[i])
        tbl[i, 3] = _fmt(st.theta_neg[i])
        tbl[i, 4] = _fmt(st.lr_stat[i]; digits=3)
        tbl[i, 5] = _format_pvalue(st.lr_p_chi2[i])
        tbl[i, 6] = _fmt(st.sr_stat[i]; digits=3)
        tbl[i, 7] = _format_pvalue(st.sr_p_chi2[i])
        tbl[i, 8] = _significance_stars(min(st.lr_p_chi2[i], st.sr_p_chi2[i]))
    end
    _pretty_table(io, tbl; title="NARDL Symmetry Tests (χ²(1); F(1,$(st.dof_resid)) p-values available)",
                  column_labels=["Regressor", "θ⁺", "θ⁻", "LR Wald", "LR p", "SR Wald", "SR p", ""],
                  alignment=[:l, :r, :r, :r, :r, :r, :r, :l])
    _show_note(io, "H₀ (long-run): θ⁺ = θ⁻. H₀ (short-run): Σπ⁺ = Σπ⁻ " *
               "(sum of ECM differenced-term coefficients). Single-restriction " *
               "Wald: χ²(1), equivalently F(1, $(st.dof_resid)). Reject ⇒ asymmetry.")
    _sig_legend(io)
end

report(st::NARDLSymmetryTest) = show(stdout, st)
report(io::IO, st::NARDLSymmetryTest) = show(io, st)

function Base.show(io::IO, mm::NARDLMultipliers{T}) where {T}
    H = last(mm.horizons)
    for i in eachindex(mm.reg_names)
        title = "Cumulative dynamic multipliers: $(mm.reg_names[i]) " *
                "(θ⁺=$(_fmt(mm.theta_pos[i])), θ⁻=$(_fmt(mm.theta_neg[i])))"
        has_ci = mm.nreps > 0
        ncol = has_ci ? 7 : 4
        tbl = Matrix{Any}(undef, length(mm.horizons), ncol)
        for (h, hz) in enumerate(mm.horizons)
            tbl[h, 1] = hz
            tbl[h, 2] = _fmt(mm.m_pos[i, h])
            tbl[h, 3] = _fmt(mm.m_neg[i, h])
            tbl[h, 4] = _fmt(mm.m_diff[i, h])
            if has_ci
                tbl[h, 5] = "[$(_fmt(mm.m_pos_lo[i,h])), $(_fmt(mm.m_pos_hi[i,h]))]"
                tbl[h, 6] = "[$(_fmt(mm.m_neg_lo[i,h])), $(_fmt(mm.m_neg_hi[i,h]))]"
                tbl[h, 7] = "[$(_fmt(mm.m_diff_lo[i,h])), $(_fmt(mm.m_diff_hi[i,h]))]"
            end
        end
        labels = has_ci ?
            ["h", "m⁺", "m⁻", "m⁺−m⁻", "m⁺ CI", "m⁻ CI", "asym CI"] :
            ["h", "m⁺", "m⁻", "m⁺−m⁻"]
        align = has_ci ? [:r, :r, :r, :r, :r, :r, :r] : [:r, :r, :r, :r]
        _pretty_table(io, tbl; title=title, column_labels=labels, alignment=align)
    end
    if mm.nreps > 0
        _show_note(io, "Bands: $(round(Int, 100*mm.level))% pointwise percentile " *
                   "from a recursive-design (condition-on-x) residual bootstrap " *
                   "($(mm.nreps) reps). Multipliers converge to θ⁺/θ⁻ as h→∞ (H=$H).")
    end
end

report(mm::NARDLMultipliers) = show(stdout, mm)
report(io::IO, mm::NARDLMultipliers) = show(io, mm)
