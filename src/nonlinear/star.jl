# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Smooth-transition autoregression (STAR / LSTR1 / LSTR2 / ESTR): NLS estimation,
the Luukkonen–Saikkonen–Teräsvirta (1988) LM3 linearity test, and Teräsvirta's
(1994) sequential model-selection procedure.

References
- Luukkonen, R., Saikkonen, P. & Teräsvirta, T. (1988). Testing linearity against
  smooth transition autoregressive models. *Biometrika* 75(3), 491–499.
- Teräsvirta, T. (1994). Specification, estimation, and evaluation of smooth
  transition autoregressive models. *JASA* 89(425), 208–218.
- van Dijk, D., Teräsvirta, T. & Franses, P. H. (2002). Smooth transition
  autoregressive models — a survey of recent developments. *Econometric Reviews*
  21(1), 1–47.

The transition slope `γ` is scaled by `1/σ̂_s` (Teräsvirta 1994) so it is
dimension-free; this is essential — without it the optimiser stalls on the flat
region of `G`. Estimation mirrors the numerics of `src/lp/state.jl`'s
`logistic_transition` (a 2-D `(γ, c)` grid start with the linear coefficients
concentrated out, then a smooth refinement) but is re-derived here and not
coupled to the state-dependent LP code.
"""

# =============================================================================
# Transition functions G(s; γ, c) with the 1/σ̂_s slope scaling
# =============================================================================

"""
    _star_transition(s, gamma, c, sigma_s, ttype) -> Vector

Smooth-transition weight `G(sₜ; γ, c) ∈ [0,1]` with the Teräsvirta (1994) slope
scaling folded in (`γ/σ̂_s` for LSTR1, `γ/σ̂_s²` for the quadratic transitions):

- `:lstr1` `G = 1/(1 + exp(-(γ/σ_s)(s - c₁)))`;
- `:lstr2` `G = 1/(1 + exp(-(γ/σ_s²)(s - c₁)(s - c₂)))`;
- `:estr`  `G = 1 - exp(-(γ/σ_s²)(s - c₁)²)`.

Written generically in the element type so it is ForwardDiff-differentiable.
"""
function _star_transition(s::AbstractVector, gamma, c::AbstractVector, sigma_s, ttype::Symbol)
    if ttype === :lstr1
        a = gamma / sigma_s
        return @. one(a) / (one(a) + exp(-a * (s - c[1])))
    elseif ttype === :lstr2
        a = gamma / sigma_s^2
        return @. one(a) / (one(a) + exp(-a * (s - c[1]) * (s - c[2])))
    elseif ttype === :estr
        a = gamma / sigma_s^2
        return @. one(a) - exp(-a * (s - c[1])^2)
    else
        throw(ArgumentError("Unknown STAR transition :$ttype; use :lstr1, :lstr2, or :estr."))
    end
end

_star_ncloc(ttype::Symbol) = ttype === :lstr2 ? 2 : 1

# =============================================================================
# Design matrix and effective sample
# =============================================================================

"""
    _star_design(y, p, d, s_ext) -> (y_eff, z, s, sname, m0)

Build the STAR regression design. Dependent `yₜ`, regressors
`zₜ = [1, y_{t-1}, …, y_{t-p}]`, and transition variable `sₜ = y_{t-d}` (or the
user-supplied `s_ext`, aligned with `y`). The effective sample starts at
`t = m0 + 1` with `m0 = max(p, d)` for a self-exciting `s`, or `m0 = p` when `s`
is supplied externally.
"""
function _star_design(y::AbstractVector{T}, p::Int, d::Int,
                      s_ext::Union{Nothing,AbstractVector}) where {T<:AbstractFloat}
    n = length(y)
    m0 = s_ext === nothing ? max(p, d) : p
    idx = (m0 + 1):n
    m = length(idx)
    y_eff = y[idx]
    z = Matrix{T}(undef, m, p + 1)
    z[:, 1] .= one(T)
    for j in 1:p
        z[:, j + 1] .= y[idx .- j]
    end
    if s_ext === nothing
        s = y[idx .- d]
        sname = "y[t-$d]"
    else
        length(s_ext) == n || throw(DimensionMismatch(
            "external transition variable s must have the same length as y ($n); got $(length(s_ext))."))
        s = T.(s_ext[idx])
        sname = "s"
    end
    return y_eff, z, s, sname, m0
end

# =============================================================================
# Concentrated NLS objective (linear φ concentrated out at fixed (γ, c))
# =============================================================================

"""
    _star_conc(y, z, s, gamma, c, sigma_s, ttype) -> (ssr, phi, G)

Concentrated fit at a fixed transition `(γ, c)`: form the weighted design
`W = [z⊙(1-G)  z⊙G]`, solve for the linear coefficients `φ = [φ₁; φ₂]` by OLS,
and return the SSR, `φ`, and the transition weights `G`.
"""
function _star_conc(y::AbstractVector{T}, z::AbstractMatrix{T}, s::AbstractVector{T},
                    gamma::T, c::AbstractVector{T}, sigma_s::T,
                    ttype::Symbol) where {T<:AbstractFloat}
    G = _star_transition(s, gamma, c, sigma_s, ttype)
    W = hcat(z .* (one(T) .- G), z .* G)
    phi, _, ssr, _ = _ols_fit(y, W)
    return ssr, phi, G
end

# =============================================================================
# Full-parameter residual vector (for LBFGS refinement + Gauss–Newton SEs)
# =============================================================================

"""
    _star_resid(theta, y, z, s, sigma_s, ttype, k, nc) -> Vector

Residual vector `uₜ = yₜ − φ₁'zₜ(1−G) − φ₂'zₜ G` as a smooth, ForwardDiff-
differentiable function of the full parameter vector
`θ = [φ₁ (k); φ₂ (k); log γ; c (nc)]`. `γ = exp(θ[2k+1])` keeps the slope
positive without a box constraint.
"""
function _star_resid(theta::AbstractVector, y::AbstractVector, z::AbstractMatrix,
                     s::AbstractVector, sigma_s, ttype::Symbol, k::Int, nc::Int)
    phi1 = @view theta[1:k]
    phi2 = @view theta[(k + 1):(2k)]
    gamma = exp(theta[2k + 1])
    c = @view theta[(2k + 2):(2k + 1 + nc)]
    G = _star_transition(s, gamma, c, sigma_s, ttype)
    fitted = (z * phi1) .* (one(eltype(G)) .- G) .+ (z * phi2) .* G
    return y .- fitted
end

# =============================================================================
# 2-D grid start over (γ, c)
# =============================================================================

"""
    _star_grid_start(y, z, s, sigma_s, ttype; n_gamma, n_c) -> (gamma0, c0, phi0, ssr0)

Grid the concentrated SSR over a log-spaced γ grid (`n_gamma` nodes on
`[0.5, 50]`, the scaled slope) and locations `c` on the sample quantiles of `s`
(`n_c` nodes on `[0.1, 0.9]`), concentrating the linear coefficients out by OLS
at each node. For LSTR2 the two locations range over ordered quantile pairs.
Returns the SSR-minimising start values.

The default grid resolution (`n_gamma=15`, `n_c=15`; ≈ 225 LSTR1/ESTR nodes,
≈ `15·105` ordered LSTR2 pairs) is what makes the multimodal STAR NLS reliable.
"""
function _star_grid_start(y::AbstractVector{T}, z::AbstractMatrix{T}, s::AbstractVector{T},
                          sigma_s::T, ttype::Symbol; n_gamma::Int=15,
                          n_c::Int=15) where {T<:AbstractFloat}
    gamma_grid = T.(exp.(range(log(0.5), log(50.0), length=n_gamma)))
    c_grid = T.(quantile(s, range(0.1, 0.9, length=n_c)))
    nc = _star_ncloc(ttype)

    best_ssr = T(Inf)
    best_gamma = gamma_grid[1]
    best_c = nc == 1 ? T[c_grid[cld(n_c, 2)]] : T[c_grid[1], c_grid[end]]
    best_phi = zeros(T, 2 * size(z, 2))

    if nc == 1
        for g in gamma_grid, cc in c_grid
            ssr, phi, _ = _star_conc(y, z, s, g, T[cc], sigma_s, ttype)
            if isfinite(ssr) && ssr < best_ssr
                best_ssr, best_gamma, best_c, best_phi = ssr, g, T[cc], phi
            end
        end
    else
        # Ordered pairs c₁ < c₂ from the same quantile grid.
        for g in gamma_grid, i in 1:n_c, j in (i + 1):n_c
            cvec = T[c_grid[i], c_grid[j]]
            ssr, phi, _ = _star_conc(y, z, s, g, cvec, sigma_s, ttype)
            if isfinite(ssr) && ssr < best_ssr
                best_ssr, best_gamma, best_c, best_phi = ssr, g, cvec, phi
            end
        end
    end
    return best_gamma, best_c, best_phi, best_ssr
end

# =============================================================================
# estimate_star
# =============================================================================

"""
    estimate_star(y, p; s=nothing, d=1, type=:auto, n_gamma=15, n_c=15) -> STARModel

Estimate a two-regime smooth-transition autoregression (STAR) by nonlinear least
squares (Teräsvirta 1994).

The model is `yₜ = φ₁'zₜ·(1−G(sₜ;γ,c)) + φ₂'zₜ·G(sₜ;γ,c) + uₜ` with
`zₜ = [1, y_{t-1}, …, y_{t-p}]` and transition variable `sₜ = y_{t-d}` (or the
supplied `s`). The transition shape `G` is LSTR1, LSTR2, or ESTR.

**Estimation.** The slope `γ` is scaled by `1/σ̂_s` (dimension-free identification;
Teräsvirta 1994). Starting values come from a 2-D grid over `(γ, c)` — `γ`
log-spaced, `c` on sample quantiles of `s` — with the linear `(φ₁, φ₂)`
concentrated out by OLS at each node; the best node is refined with
`Optim.optimize` (LBFGS, ForwardDiff gradient over `θ = [φ; log γ; c]`). Standard
errors are the Gauss–Newton delta-method SEs `robust_inv(Hermitian(J'J))·σ̂²`.

# Arguments
- `y::AbstractVector`: the series.
- `p::Int`: autoregressive order (per regime).

# Keywords
- `s`: transition variable (a vector aligned with `y`); default `nothing` ⇒ `y_{t-d}`.
- `d::Int`: transition delay for the self-exciting case (default `1`).
- `type::Symbol`: `:lstr1`, `:lstr2`, `:estr`, or `:auto` (default) to run the
  Teräsvirta (1994) sequential-test selection and store its three p-values.
- `n_gamma`, `n_c`: grid resolution for the start values.

Returns a [`STARModel`](@ref).
"""
function estimate_star(y::AbstractVector, p::Int; s=nothing, d::Int=1,
                       type::Symbol=:auto, n_gamma::Int=15, n_c::Int=15)
    p >= 1 || throw(ArgumentError("STAR order p must be ≥ 1; got $p."))
    d >= 1 || throw(ArgumentError("delay d must be ≥ 1; got $d."))
    T = float(eltype(y))
    yv = Vector{T}(y)

    y_eff, z, sv, sname, _ = _star_design(yv, p, d, s)
    n = length(y_eff)
    k = p + 1
    sigma_s = std(sv)
    sigma_s > 0 || throw(ArgumentError("transition variable has zero variance; cannot scale γ."))

    # Transition-type selection (or the supplied fixed type).
    sel_pvalues = nothing
    ttype = type
    if type === :auto
        sel = _terasvirta_select(y_eff, z, sv, p)
        ttype = sel.type
        sel_pvalues = sel.pvalues
    end
    ttype in (:lstr1, :lstr2, :estr) ||
        throw(ArgumentError("type must be :lstr1, :lstr2, :estr, or :auto; got :$type."))
    nc = _star_ncloc(ttype)
    (n > 2k + 1 + nc) || throw(ArgumentError(
        "sample too small: STAR($p, $(ttype)) needs more than $(2k + 1 + nc) effective observations, has $n."))

    # 2-D grid start, then LBFGS refinement over θ = [φ; log γ; c].
    g0, c0, phi0, _ = _star_grid_start(y_eff, z, sv, sigma_s, ttype;
                                       n_gamma=n_gamma, n_c=n_c)
    theta0 = vcat(phi0, log(g0), c0)

    obj(theta) = sum(abs2, _star_resid(theta, y_eff, z, sv, sigma_s, ttype, k, nc))
    g!(G, theta) = ForwardDiff.gradient!(G, obj, theta)

    converged = false
    theta_hat = theta0
    try
        res = Optim.optimize(obj, g!, theta0, Optim.LBFGS(),
                             Optim.Options(f_reltol=1e-10, g_tol=1e-8, iterations=1000))
        theta_hat = Optim.minimizer(res)
        converged = Optim.converged(res)
    catch err
        err isa InterruptException && rethrow()
        theta_hat = theta0   # fall back to the grid start
    end
    # Guard against a worse refinement (LBFGS can wander on the flat γ ridge).
    if obj(theta_hat) > obj(theta0)
        theta_hat = theta0
    end

    phi1 = theta_hat[1:k]
    phi2 = theta_hat[(k + 1):(2k)]
    gamma = exp(theta_hat[2k + 1])
    c = theta_hat[(2k + 2):(2k + 1 + nc)]
    G = _star_transition(sv, gamma, c, sigma_s, ttype)
    resid = _star_resid(theta_hat, y_eff, z, sv, sigma_s, ttype, k, nc)
    ssr = dot(resid, resid)
    npar = 2k + 1 + nc
    sigma2 = ssr / max(n - npar, 1)

    # Gauss–Newton delta-method covariance: robust_inv(Hermitian(J'J))·σ̂².
    J = ForwardDiff.jacobian(th -> _star_resid(th, y_eff, z, sv, sigma_s, ttype, k, nc),
                             theta_hat)
    covm = Matrix{T}(robust_inv(Hermitian(J' * J); silent=true)) .* sigma2
    se_all = sqrt.(max.(diag(covm), zero(T)))
    se_phi1 = se_all[1:k]
    se_phi2 = se_all[(k + 1):(2k)]
    # θ carries log γ, so SE(γ) = γ·SE(log γ) by the delta method.
    se_gamma = gamma * se_all[2k + 1]
    se_c = se_all[(2k + 2):(2k + 1 + nc)]

    # Information criteria (Gaussian log-likelihood at σ̂²ₘₗₑ = SSR/n).
    s2_mle = ssr / n
    loglik = -T(n) / 2 * (log(T(2π)) + log(s2_mle) + one(T))
    aic = -2 * loglik + 2 * npar
    bic = -2 * loglik + log(T(n)) * npar

    # LM3 linearity test on the same design (χ² and F forms).
    lm = _star_lm3(y_eff, z, sv, p)

    znames = vcat("const", ["y[t-$i]" for i in 1:p])

    return STARModel{T}(y_eff, z, sv, phi1, phi2, se_phi1, se_phi2, gamma, c,
        se_gamma, se_c, G, ttype, resid, ssr, sigma2, n, p, d, k, sigma_s,
        aic, bic, znames, sname, lm.stat, lm.pvalue, lm.fstat, lm.fpvalue,
        sel_pvalues, converged)
end

# =============================================================================
# LM3 auxiliary regression (shared by the test and the selection sequence)
# =============================================================================

"""
    _star_aux_ssr(resid0, z, s, p) -> (ssr0, ssr1, ssr2, ssr3, n, k_base)

Fit the Luukkonen–Saikkonen–Teräsvirta auxiliary regressions of the linear-AR
residuals `ê` on `zₜ` augmented with the interaction blocks `z̃ₜ·sₜ^j`
(`j = 1,2,3`), where `z̃ₜ = (y_{t-1}, …, y_{t-p})` are the AR lags (constant
excluded). Returns the SSRs of the nested regressions:

- `ssr0` = SSR of `ê ~ z` (no interaction; equals Σê² since `ê ⟂ z`);
- `ssr1` = SSR of `ê ~ [z, z̃·s]`;
- `ssr2` = SSR of `ê ~ [z, z̃·s, z̃·s²]`;
- `ssr3` = SSR of `ê ~ [z, z̃·s, z̃·s², z̃·s³]` (the full LM3 auxiliary regression).
"""
function _star_aux_ssr(resid0::AbstractVector{T}, z::AbstractMatrix{T},
                       s::AbstractVector{T}, p::Int) where {T<:AbstractFloat}
    n = length(resid0)
    zt = z[:, 2:end]                       # lags only (p columns)
    s1 = zt .* s
    s2 = zt .* (s .^ 2)
    s3 = zt .* (s .^ 3)

    _, _, ssr0, _ = _ols_fit(resid0, z)
    _, _, ssr1, _ = _ols_fit(resid0, hcat(z, s1))
    _, _, ssr2, _ = _ols_fit(resid0, hcat(z, s1, s2))
    _, _, ssr3, _ = _ols_fit(resid0, hcat(z, s1, s2, s3))
    return ssr0, ssr1, ssr2, ssr3, n, size(z, 2)
end

"""
    _star_lm3(y, z, s, p) -> (stat, pvalue, fstat, fpvalue)

Luukkonen–Saikkonen–Teräsvirta (1988) LM3 linearity test: the `nR²` χ²(3p)
statistic from the auxiliary regression of the linear-AR residuals on the cubic
Taylor expansion of `G`, plus the F-form `F(3p, n−4p−1)` (better small-sample
size).
"""
function _star_lm3(y::AbstractVector{T}, z::AbstractMatrix{T}, s::AbstractVector{T},
                   p::Int) where {T<:AbstractFloat}
    _, resid0, _, _ = _ols_fit(y, z)
    ssr0, _, _, ssr3, n, k_base = _star_aux_ssr(resid0, z, s, p)
    df = 3 * p                              # restrictions tested
    k_full = k_base + df                    # regressors in the full auxiliary regression
    r2 = ssr0 > 0 ? (ssr0 - ssr3) / ssr0 : zero(T)
    r2 = clamp(r2, zero(T), one(T))
    stat = T(n) * r2
    pvalue = T(ccdf(Chisq(df), stat))
    df2 = max(n - k_full, 1)
    fstat = ((ssr0 - ssr3) / df) / (ssr3 / df2)
    fstat = max(fstat, zero(T))
    fpvalue = T(ccdf(FDist(df, df2), fstat))
    return (stat=stat, pvalue=pvalue, fstat=fstat, fpvalue=fpvalue)
end

# =============================================================================
# star_linearity_test (public)
# =============================================================================

"""
    star_linearity_test(y, p; s=nothing, d=1) -> (; stat, pvalue, fstat, fpvalue, df)

Luukkonen–Saikkonen–Teräsvirta (1988) LM3 test of linearity against a smooth-
transition alternative. The residuals `ê` of the linear AR(`p`) are regressed on
`zₜ` augmented with `z̃ₜ·sₜ`, `z̃ₜ·sₜ²`, `z̃ₜ·sₜ³` (the third-order Taylor
expansion of `G` around `γ = 0`), where `z̃ₜ = (y_{t-1}, …, y_{t-p})`. The LM
statistic `n·R² ∼ χ²(3p)` and its F-form `F(3p, n−4p−1)` (better small-sample
size) are returned.

# Arguments
- `y::AbstractVector`, `p::Int`: series and AR order.

# Keywords
- `s`: transition variable aligned with `y` (default `y_{t-d}`).
- `d::Int`: transition delay (default `1`).

Returns a `NamedTuple` `(stat, pvalue, fstat, fpvalue, df)`.
"""
function star_linearity_test(y::AbstractVector, p::Int; s=nothing, d::Int=1)
    p >= 1 || throw(ArgumentError("AR order p must be ≥ 1; got $p."))
    d >= 1 || throw(ArgumentError("delay d must be ≥ 1; got $d."))
    T = float(eltype(y))
    yv = Vector{T}(y)
    y_eff, z, sv, _, _ = _star_design(yv, p, d, s)
    lm = _star_lm3(y_eff, z, sv, p)
    return (stat=lm.stat, pvalue=lm.pvalue, fstat=lm.fstat, fpvalue=lm.fpvalue, df=3 * p)
end

# =============================================================================
# Teräsvirta (1994) sequential model-selection procedure
# =============================================================================

"""
    _terasvirta_select(y, z, s, p) -> (; type, pvalues)

Teräsvirta's (1994) sequential F-test on the LM3 auxiliary regression, selecting
the transition shape from the nested hypotheses on the cubic/quadratic/linear
interaction blocks:

- `F₄` tests `H₀₄: β₃ = 0` (the `s³` block);
- `F₃` tests `H₀₃: β₂ = 0 ∣ β₃ = 0` (the `s²` block, imposing `β₃ = 0`);
- `F₂` tests `H₀₂: β₁ = 0 ∣ β₂ = β₃ = 0` (the `s` block).

Decision rule (van Dijk–Teräsvirta–Franses 2002): if the p-value of `F₃` is the
smallest of the three, choose the symmetric **ESTR** transition; otherwise choose
**LSTR1**. Returns the selected `type` and the p-value triple `(F₄, F₃, F₂)`.
"""
function _terasvirta_select(y::AbstractVector{T}, z::AbstractMatrix{T},
                            s::AbstractVector{T}, p::Int) where {T<:AbstractFloat}
    _, resid0, _, _ = _ols_fit(y, z)
    ssr0, ssr1, ssr2, ssr3, n, k_base = _star_aux_ssr(resid0, z, s, p)
    df = p                                  # each interaction block has p terms

    # F₄: full [z,s,s²,s³] vs [z,s,s²].  denom dof = n − (k_base + 3p).
    d4 = max(n - (k_base + 3p), 1)
    F4 = ((ssr2 - ssr3) / df) / (ssr3 / d4)
    # F₃: [z,s,s²] vs [z,s]  (imposing β₃=0).  denom dof = n − (k_base + 2p).
    d3 = max(n - (k_base + 2p), 1)
    F3 = ((ssr1 - ssr2) / df) / (ssr2 / d3)
    # F₂: [z,s] vs [z]  (imposing β₂=β₃=0).  denom dof = n − (k_base + p).
    d2 = max(n - (k_base + p), 1)
    F2 = ((ssr0 - ssr1) / df) / (ssr1 / d2)

    p4 = T(ccdf(FDist(df, d4), max(F4, zero(T))))
    p3 = T(ccdf(FDist(df, d3), max(F3, zero(T))))
    p2 = T(ccdf(FDist(df, d2), max(F2, zero(T))))

    # If H₀₃ is the most strongly rejected, the transition is symmetric ⇒ ESTR;
    # otherwise LSTR1 (Teräsvirta 1994; van Dijk et al. 2002).
    sel = (p3 < p4 && p3 < p2) ? :estr : :lstr1
    return (type=sel, pvalues=(p4, p3, p2))
end

# =============================================================================
# Forecast (bootstrap simulation of a STAR model)
# =============================================================================

"""
    forecast(m::STARModel, h; reps=1000, level=0.90, rng=Random.default_rng())

Multi-step bootstrap-simulation forecast of a fitted [`STARModel`](@ref).

Iterates the smooth-transition recursion forward `h` steps over `reps` simulated
paths, drawing residuals with replacement from the fitted residuals at each step.
Only self-exciting models (`sₜ = y_{t-d}`) are supported; a model fit with an
external transition variable would require its future path. Returns a
[`STARForecast`](@ref) with the mean path, per-horizon standard deviations, and
central `level` percentile bands.
"""
function forecast(m::STARModel{T}, h::Int; reps::Int=1000, level::Real=0.90,
                  rng::Random.AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    startswith(m.sname, "y[t-") || throw(ArgumentError(
        "forecast is only defined for self-exciting STAR models (sₜ = y_{t-d})."))
    h >= 1 || throw(ArgumentError("horizon h must be ≥ 1."))
    (0 < level < 1) || throw(ArgumentError("level must satisfy 0 < level < 1."))

    p = m.p; d = m.d
    # Reconstruct the observed history: z[:,1] is the constant, z[1, 2:end] are the
    # p earliest lags (reversed) that precede m.y.
    hist0 = vcat(m.z[1, 2:end][end:-1:1], m.y)   # length = p + n
    resids = m.residuals

    paths = Matrix{T}(undef, reps, h)
    for r in 1:reps
        hist = copy(hist0)
        for step in 1:h
            L = length(hist)
            sval = hist[L + 1 - d]
            Gt = _star_transition(T[sval], m.gamma, m.c, m.sigma_s, m.trans_type)[1]
            zt = Vector{T}(undef, p + 1)
            zt[1] = one(T)
            for j in 1:p
                zt[j + 1] = hist[L + 1 - j]
            end
            yhat = dot(zt, m.phi1) * (one(T) - Gt) + dot(zt, m.phi2) * Gt +
                   resids[rand(rng, 1:length(resids))]
            push!(hist, yhat)
            paths[r, step] = yhat
        end
    end

    alpha = (1 - level) / 2
    fmean = vec(mean(paths; dims=1))
    fse = vec(std(paths; dims=1))
    lo = T[Statistics.quantile(view(paths, :, s), alpha) for s in 1:h]
    hi = T[Statistics.quantile(view(paths, :, s), 1 - alpha) for s in 1:h]
    return STARForecast{T}(fmean, lo, hi, fse, h, T(level), reps)
end
