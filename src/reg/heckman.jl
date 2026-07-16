# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Heckman (1979) sample-selection (incidental-truncation) model — EV-18 (#426).

The continuous outcome `y` is observed only where a binary selection indicator `d == 1`
(wages for labor-force participants, export prices for exporters, loan rates for approved
applicants). Two estimators are provided:

  * **Two-step (Heckit)** — probit of `d` on `Z` (via `estimate_probit`), form the inverse-Mills
    ratio `λ̂ = φ(z'γ̂)/Φ(z'γ̂)` with the EV-17 helper [`_mills`](@ref), then OLS of `y` on
    `[X λ̂]` over the selected subsample. The coefficient on `λ̂` is `ρσ`; `σ` is recovered from
    the `δ̂`-corrected residual variance and the covariance is the Greene corrected two-step
    covariance (generated-regressor + selection-induced heteroskedasticity).
  * **Full-information ML** — the bivariate-normal selection log-likelihood, with `ρ = tanh(a)`
    and `σ = exp(s)` for unconstrained optimization (`Optim.optimize`), started at the two-step
    estimates, with delta-method standard errors.

References: Heckman (1979) *Econometrica*; Greene, *Econometric Analysis* (sample-selection
chapter); data (Mroz 1987, *Econometrica*, via `load_example(:mroz)`).
"""

using LinearAlgebra, Statistics, Distributions, StatsAPI

# =============================================================================
# Log-likelihood (bivariate-normal selection model), shared by :mle and the
# two-step profile evaluation.
# =============================================================================

# Negative log-likelihood in the unconstrained parameterization
#   par = [γ (pg); β (kb); s=log σ; a=atanh ρ].
# `Zsel`, `Xsel`, `ysel` cover the SELECTED subsample; `Zuns` the unselected rows.
function _heckman_negll(par::AbstractVector{S}, Zsel::Matrix{T}, Xsel::Matrix{T},
                        ysel::Vector{T}, Zuns::Matrix{T}) where {S,T<:AbstractFloat}
    pg = size(Zsel, 2); kb = size(Xsel, 2)
    γ = view(par, 1:pg)
    β = view(par, pg+1:pg+kb)
    s = par[pg+kb+1]; a = par[pg+kb+2]
    σ = exp(s); ρ = tanh(a)
    N = Normal(zero(S), one(S))
    ll = zero(S)
    # Unselected: log Φ(−z'γ)
    if !isempty(Zuns)
        zgu = Zuns * γ
        @inbounds for i in eachindex(zgu)
            ll += logcdf(N, -zgu[i])
        end
    end
    # Selected: log[ φ(u/σ)/σ · Φ( (z'γ + (ρ/σ) u) / √(1−ρ²) ) ], u = y − x'β
    zgs = Zsel * γ
    resid = ysel .- Xsel * β
    denom = sqrt(one(S) - ρ^2)
    @inbounds for i in eachindex(ysel)
        u = resid[i]
        arg = (zgs[i] + (ρ / σ) * u) / denom
        ll += logpdf(N, u / σ) - log(σ) + logcdf(N, arg)
    end
    -ll
end

# =============================================================================
# estimate_heckman
# =============================================================================

"""
    estimate_heckman(y, X, d, Z; method=:twostep, outcome_names=nothing,
                     select_names=nothing, maxiter=1000, tol=1e-10) -> HeckmanModel{T}

Estimate a Heckman (1979) sample-selection model. `y` is the (full-length) outcome, observed
only where the binary selection indicator `d == 1` (entries of `y` for `d == 0` are ignored and
may be `NaN`). `X` is the outcome-equation design matrix and `Z` the selection-equation design
matrix, both full-length and **including their own intercept columns**. For identification `Z`
should contain at least one regressor excluded from `X` (an exclusion restriction); if `Z` lies
entirely in the column space of `X`, identification rests solely on the nonlinearity of the
Mills ratio and a warning is emitted.

`method`:
  * `:twostep` (default) — Heckit: probit selection (`estimate_probit`) → inverse-Mills-augmented
    OLS, with the Greene corrected two-step covariance.
  * `:mle` — full-information bivariate-normal ML, started at the two-step estimates.

# Examples
```julia
using MacroEconometricModels
d = load_example(:mroz)
inlf = d[:, "inlf"]
lwage = d[:, "lwage"]
X = hcat(ones(753), d[:, "educ"], d[:, "exper"], d[:, "expersq"])
Z = hcat(ones(753), d[:, "nwifeinc"], d[:, "educ"], d[:, "exper"],
         d[:, "expersq"], d[:, "age"], d[:, "kidslt6"], d[:, "kidsge6"])
m = estimate_heckman(lwage, X, inlf, Z; method=:twostep)
report(m)
```

# References
- Heckman, J. J. (1979). *Econometrica* 47(1), 153-161.
- Greene, W. H. *Econometric Analysis* (sample-selection chapter).
"""
function estimate_heckman(y::AbstractVector{T}, X::AbstractMatrix{T},
                          d::AbstractVector, Z::AbstractMatrix{T};
                          method::Symbol=:twostep,
                          outcome_names::Union{Nothing,Vector{String}}=nothing,
                          select_names::Union{Nothing,Vector{String}}=nothing,
                          maxiter::Int=1000, tol::T=T(1e-10)) where {T<:AbstractFloat}
    method in (:twostep, :mle) ||
        throw(ArgumentError("method must be :twostep or :mle; got :$method"))
    n = length(y)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    size(Z, 1) == n || throw(ArgumentError("Z must have $n rows (got $(size(Z, 1)))"))
    length(d) == n || throw(ArgumentError("d must have length $n (got $(length(d)))"))
    k = size(X, 2); p = size(Z, 2)

    Xm = Matrix{T}(X); Zm = Matrix{T}(Z)
    dv = Vector{T}(float.(d))
    all(v -> v == zero(T) || v == one(T), dv) ||
        throw(ArgumentError("selection indicator d must be binary (0/1)"))
    on = something(outcome_names, ["x$i" for i in 1:k])
    sn = something(select_names, ["z$i" for i in 1:p])
    length(on) == k || throw(ArgumentError("outcome_names must have length $k"))
    length(sn) == p || throw(ArgumentError("select_names must have length $p"))

    sel = findall(==(one(T)), dv)
    n1 = length(sel)
    n1 > k + 1 || throw(ArgumentError("need more selected obs than outcome params (n_sel=$n1, k=$k)"))
    ys = Vector{T}(y[sel])
    all(isfinite, ys) || throw(ArgumentError("y has non-finite values among selected (d==1) observations"))

    # ---- Exclusion-restriction diagnostic: does span(X) already contain all of Z? ----
    _warn_no_exclusion(Xm, Zm)

    # ---- Step 1: probit selection ----
    probit = estimate_probit(dv, Zm; varnames=sn)
    γ = probit.beta
    Vγ = Matrix{T}(probit.vcov_mat)

    zg = Zm * γ                       # z'γ for all obs
    mills_all = _mills.(zg)           # inverse Mills ratio φ/Φ (EV-17 helper)
    Xs = Xm[sel, :]
    Zs = Zm[sel, :]
    mills_s = mills_all[sel]
    zg_s = zg[sel]

    # ---- Step 2: OLS of y on [X  λ̂] over the selected subsample ----
    W = hcat(Xs, mills_s)             # n1 × (k+1)
    WtW = Symmetric(W' * W)
    WtW_inv = robust_inv(WtW)
    βstar = WtW_inv * (W' * ys)
    β = βstar[1:k]
    βλ = βstar[k+1]
    resid = ys .- W * βstar

    # σ̂² recovery with the δ̂-correction; ρ̂ = β_λ / σ̂
    δ = mills_s .* (mills_s .+ zg_s)          # 0 < δ_i < 1
    δbar = mean(δ)
    σ2 = dot(resid, resid) / n1 + δbar * βλ^2
    σ = sqrt(max(σ2, sqrt(eps(T))))
    ρ = βλ / σ
    ρ = clamp(ρ, -one(T) + sqrt(eps(T)), one(T) - sqrt(eps(T)))
    λ = βλ

    # ---- Greene corrected two-step covariance of β* ----
    # V = σ̂²(W'W)⁻¹[ W'(I − ρ̂²Δ)W + Q ](W'W)⁻¹,  Δ = diag(δ),
    #   Q = ρ̂² (W'Δ Z_s) V̂_γ (Z_s'Δ W)   (generated-regressor feedback from step 1).
    # Greene's coefficient on the feedback term is ρ̂²: the feedback variance is
    # β_λ²(W'ΔZ)V_γ(Z'ΔW) with β_λ = ρσ, so after factoring σ̂² out front it is ρ̂²
    # (not ρ̂⁴). Under-weighting by ρ² makes two-step SEs anti-conservative at high ρ.
    Δ = Diagonal(δ)
    WtDW = W' * Δ * W
    WtDZ = W' * Δ * Zs                        # (k+1) × p
    Q = (ρ^2) * (WtDZ * Vγ * WtDZ')
    mid = Symmetric(Matrix(WtW) .- (ρ^2) .* WtDW .+ Q)
    Vstar = Symmetric(σ2 .* (WtW_inv * mid * WtW_inv))
    Vfull = Matrix{T}(Vstar)
    vcov_beta = Vfull[1:k, 1:k]
    λ_se = sqrt(max(Vfull[k+1, k+1], zero(T)))

    if method === :twostep
        par = vcat(γ, β, log(σ), atanh(ρ))
        Zuns = Zm[dv .== zero(T), :]
        loglik = -_heckman_negll(par, Zs, Xs, ys, Zuns)
        npar = p + k + 2
        aic = -2 * loglik + 2 * T(npar)
        bic = -2 * loglik + log(T(n)) * T(npar)
        return HeckmanModel{T}(β, vcov_beta, on, γ, Vγ, sn, ρ, σ, λ,
                               T(NaN), T(NaN), λ_se, mills_s, :twostep,
                               loglik, aic, bic, n1, n, ys, Xs, true)
    end

    # ---- Full-information ML, started at the two-step estimates ----
    Zuns = Zm[dv .== zero(T), :]
    par0 = vcat(γ, β, log(σ), atanh(ρ))
    obj = par -> _heckman_negll(par, Zs, Xs, ys, Zuns)
    g! = (G, par) -> ForwardDiff.gradient!(G, obj, par)
    # The selection likelihood flattens as ρ → 0 (the two-step is then near-singular), so a
    # strict gradient tolerance can fail to flag convergence at a genuine optimum; 1e-6 is ample
    # for delta-method SEs and lets the flat-ρ case converge.
    opts = Optim.Options(iterations=maxiter, g_tol=T(1e-6))
    res = Optim.optimize(obj, g!, par0, Optim.BFGS(), opts)
    p̂ = Optim.minimizer(res)
    converged = Optim.converged(res)

    γ̂ = p̂[1:p]; β̂ = p̂[p+1:p+k]; ŝ = p̂[p+k+1]; â = p̂[p+k+2]
    σ̂ = exp(ŝ); ρ̂ = tanh(â); λ̂ = ρ̂ * σ̂

    H = ForwardDiff.hessian(obj, p̂)
    Vp = Matrix{T}(robust_inv(Symmetric(H)))
    vcov_gamma_ml = Vp[1:p, 1:p]
    vcov_beta_ml = Vp[p+1:p+k, p+1:p+k]
    # Delta-method SEs for (σ, ρ, λ) from the (s, a) block.
    Vsa = Vp[p+k+1:p+k+2, p+k+1:p+k+2]
    σ_se = σ̂ * sqrt(max(Vsa[1, 1], zero(T)))          # ∂σ/∂s = σ
    dρ = one(T) - ρ̂^2                                  # ∂ρ/∂a
    ρ_se = dρ * sqrt(max(Vsa[2, 2], zero(T)))
    gλ = [λ̂, σ̂ * dρ]                                   # ∂λ/∂s = λ, ∂λ/∂a = σ(1−ρ²)
    λ_se_ml = sqrt(max(dot(gλ, Vsa * gλ), zero(T)))

    loglik = -Optim.minimum(res)
    mills_ml = _mills.(Zs * γ̂)
    npar = p + k + 2
    aic = -2 * loglik + 2 * T(npar)
    bic = -2 * loglik + log(T(n)) * T(npar)
    return HeckmanModel{T}(β̂, vcov_beta_ml, on, γ̂, vcov_gamma_ml, sn, ρ̂, σ̂, λ̂,
                           ρ_se, σ_se, λ_se_ml, mills_ml, :mle,
                           loglik, aic, bic, n1, n, ys, Xs, converged)
end

# Float fallback
function estimate_heckman(y::AbstractVector, X::AbstractMatrix, d::AbstractVector,
                          Z::AbstractMatrix; kwargs...)
    estimate_heckman(Float64.(y), Float64.(X), d, Float64.(Z); kwargs...)
end

# Warn when Z adds no exclusion restriction beyond X (Z ⊆ span(X)).
function _warn_no_exclusion(X::Matrix{T}, Z::Matrix{T}) where {T<:AbstractFloat}
    tol = sqrt(eps(T))
    XtX_inv = robust_inv(Symmetric(X' * X); silent=true)
    P = X * XtX_inv * X'                       # projector onto span(X)
    spanned = true
    for j in axes(Z, 2)
        zj = @view Z[:, j]
        r = zj .- P * zj
        nz = norm(zj)
        if norm(r) > tol * max(nz, one(T))
            spanned = false
            break
        end
    end
    spanned && @warn("Heckman: selection regressors Z lie in the column space of the outcome " *
                     "regressors X (no exclusion restriction). Identification then rests solely " *
                     "on the nonlinearity of the inverse-Mills ratio and is fragile.")
    nothing
end

# =============================================================================
# StatsAPI interface
# =============================================================================

StatsAPI.coef(m::HeckmanModel) = m.beta
StatsAPI.vcov(m::HeckmanModel) = m.vcov_beta
StatsAPI.stderror(m::HeckmanModel) =
    sqrt.(max.(diag(m.vcov_beta), zero(eltype(m.beta))))
StatsAPI.nobs(m::HeckmanModel) = m.n_selected
StatsAPI.dof(m::HeckmanModel) = length(m.beta) + length(m.gamma) + 2
StatsAPI.dof_residual(m::HeckmanModel) = m.n_selected - length(m.beta)
StatsAPI.loglikelihood(m::HeckmanModel) = m.loglik
StatsAPI.aic(m::HeckmanModel) = m.aic
StatsAPI.bic(m::HeckmanModel) = m.bic
StatsAPI.islinear(::HeckmanModel) = false
StatsAPI.predict(m::HeckmanModel) = m.X * m.beta
StatsAPI.residuals(m::HeckmanModel) = m.y .- m.X * m.beta

function StatsAPI.confint(m::HeckmanModel{T}; level::Real=0.95) where {T}
    se = stderror(m)
    crit = T(quantile(Normal(), 1 - (1 - level) / 2))
    hcat(m.beta .- crit .* se, m.beta .+ crit .* se)
end

# Selection-equation (probit) standard errors.
_selection_stderror(m::HeckmanModel{T}) where {T} =
    sqrt.(max.(diag(m.vcov_gamma), zero(T)))

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::HeckmanModel{T}) where {T}
    meth = m.method === :twostep ? "Two-step (Heckit)" : "Maximum likelihood"
    spec = Any[
        "Model"          "Heckman selection";
        "Method"         meth;
        "Total obs."     m.n_total;
        "Selected obs."  m.n_selected;
        "Censored obs."  m.n_total - m.n_selected;
        "rho"            _fmt(m.rho);
        "sigma"          _fmt(m.sigma);
        "lambda (rho*sigma)" _fmt(m.lambda);
        "Log-lik."       _fmt(m.loglik; digits=2);
        "AIC"            _fmt(m.aic; digits=2);
        "BIC"            _fmt(m.bic; digits=2);
        "Converged"      m.converged ? "Yes" : "No"
    ]
    _pretty_table(io, spec; title="Heckman Selection Model",
                  column_labels=["Specification", ""], alignment=[:l, :r])
    _coef_table(io, "Selection equation (probit)", m.select_names, m.gamma,
                _selection_stderror(m); dist=:z)
    _coef_table(io, "Outcome equation", m.outcome_names, m.beta, stderror(m); dist=:z)

    # Selection footer: ρ/σ/λ and the H₀: no selection test.
    if m.method === :mle && isfinite(m.rho_se) && m.rho_se > 0
        w = m.rho / m.rho_se
        pw = 2 * (1 - cdf(Normal(), abs(w)))
        println(io, "Selection: rho = ", _fmt(m.rho), " (se ", _fmt(m.rho_se),
                "), Wald H0: rho=0  z = ", _fmt(w), ", p = ", _format_pvalue(T(pw)))
    elseif isfinite(m.lambda_se) && m.lambda_se > 0
        tl = m.lambda / m.lambda_se
        pl = 2 * (1 - cdf(Normal(), abs(tl)))
        println(io, "Selection: lambda = ", _fmt(m.lambda), " (se ", _fmt(m.lambda_se),
                "), H0: no selection  z = ", _fmt(tl), ", p = ", _format_pvalue(T(pl)))
    end
    _sig_legend(io)
end
