# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Dynamic heterogeneous-panel ARDL: Pooled Mean Group (PMG), Mean Group (MG), and
Dynamic Fixed Effects (DFE), plus a generalized Hausman selection test. Extends the
EV-08 `src/ardl/` scaffold: the MG estimator reuses [`estimate_ardl`](@ref) per unit,
DFE reuses `_panel_cluster_vcov`, and the Hausman quadratic form reuses
`_hausman_quadratic_form` from `src/preg/tests.jl`.

All three methods build each unit's ARDL(p, q) in the conditional error-correction form

    Δy_it = φ_i (y_{i,t-1} − θ' x_{i,t-1}) + Σ_{j=1}^{p-1} ξ_ij Δy_{i,t-j}
            + Σ_{j=0}^{q-1} ψ_ij' Δx_{i,t-j} + deterministics_i + ε_it .

References:
- Pesaran, Shin & Smith (1999, JASA 94:621–634) — PMG.
- Pesaran & Smith (1995, J.Econometrics 68:79–113) — MG.
- Blackburne & Frank (2007, Stata Journal 7:197–208) — Stata `xtpmg`.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# trend ↔ PSS case mapping (for the per-unit MG ARDL call)
# =============================================================================

# The short-run deterministics of the EC form: :none → nothing; :constant → unit
# intercept; :trend → unit intercept + linear trend. The MG per-unit ARDL is fit in
# levels form with the matching PSS (2001) case so its long-run θ_i is comparable.
function _pmg_trend_case(trend::Symbol)
    trend === :none && return 1
    trend === :constant && return 3
    trend === :trend && return 5
    throw(ArgumentError("trend must be :none, :constant, or :trend; got :$trend"))
end

# =============================================================================
# Per-unit error-correction design
# =============================================================================

"""
    _pmg_unit_design(y, X, p, q, trend) -> (dy, ylag, Xlag, W, srnames)

Build one unit's EC-form pieces over the effective sample `t = L+1 … T_i`,
`L = max(p, q)`:

- `dy`      : `Δy_t`                          (`n`-vector, the dependent variable)
- `ylag`    : `y_{t-1}`                        (`n`-vector, EC level of y)
- `Xlag`    : `x_{t-1}`                        (`n×k`, EC levels of the regressors)
- `W`       : short-run design `[deterministics  Δy lags  Δx lags]` (`n×m`)
- `srnames` : column labels of `W`

The EC regressor for a given `θ` is `ξ = ylag − Xlag*θ`; it enters with coefficient
`φ_i`. `W`'s column space plus `{ylag, Xlag}` spans exactly the levels-ARDL design, so
the fit is identical to [`estimate_ardl`](@ref) on the same sample.
"""
function _pmg_unit_design(y::AbstractVector{T}, X::AbstractMatrix{T}, p::Int, q::Int,
                          trend::Symbol; yname::String,
                          xnames::Vector{String}) where {T<:AbstractFloat}
    Ti = length(y)
    k = size(X, 2)
    L = max(p, q)
    rows = (L + 1):Ti
    n = length(rows)

    dy = T[y[t] - y[t-1] for t in rows]
    ylag = T[y[t-1] for t in rows]
    Xlag = Matrix{T}(undef, n, k)
    for j in 1:k, (r, t) in enumerate(rows)
        Xlag[r, j] = X[t-1, j]
    end

    cols = Vector{Vector{T}}()
    srnames = String[]
    if trend === :constant || trend === :trend
        push!(cols, ones(T, n)); push!(srnames, "(Intercept)")
    end
    if trend === :trend
        push!(cols, T.(collect(rows))); push!(srnames, "trend")
    end
    # Δy lags 1 … p-1
    for j in 1:(p-1)
        push!(cols, T[y[t-j] - y[t-j-1] for t in rows]); push!(srnames, "L$(j).D.$(yname)")
    end
    # Δx lags 0 … q-1, per regressor
    for jx in 1:k, l in 0:(q-1)
        push!(cols, T[X[t-l, jx] - X[t-l-1, jx] for t in rows])
        push!(srnames, l == 0 ? "D.$(xnames[jx])" : "L$(l).D.$(xnames[jx])")
    end

    W = isempty(cols) ? Matrix{T}(undef, n, 0) : reduce(hcat, cols)
    (dy, ylag, Xlag, W, srnames)
end

# OLS of `y` on `Z` → (coef, resid, XtXinv). robust_inv guards near-singular Z'Z.
function _pmg_ols(Z::AbstractMatrix{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
    ZtZ = Symmetric(Z' * Z)
    ZtZinv = Matrix{T}(robust_inv(ZtZ))
    b = ZtZinv * (Z' * y)
    resid = y .- Z * b
    (b, resid, ZtZinv)
end

# =============================================================================
# Public estimator
# =============================================================================

"""
    estimate_pmg(pd::PanelData, y::Symbol, xs::Symbol...;
                 p=1, q=1, method=:pmg, trend=:constant, maxiter=100, tol=1e-8)
    estimate_pmg(y, X, id, time; kwargs...)

Estimate a dynamic heterogeneous-panel ARDL(`p`, `q`) in error-correction form.
`method` selects the estimator:

- `:pmg` — Pooled Mean Group (common long-run `θ`, heterogeneous short-run + `φ_i`).
- `:mg`  — Mean Group (per-unit unrestricted ARDL, averaged).
- `:dfe` — Dynamic Fixed Effects (pooled, unit intercepts, clustered SEs).

# Arguments
- `pd::PanelData` — panel from [`xtset`](@ref); `y` is the dependent variable symbol and
  `xs...` the long-run regressor symbols. Alternatively pass `y::AbstractVector`,
  `X::AbstractMatrix`, `id`, and `time` vectors directly.

# Keywords
- `p::Int=1` — autoregressive order (`≥ 1`); `q::Int=1` — distributed-lag order (`≥ 0`).
- `method::Symbol=:pmg`; `trend::Symbol=:constant` (`:none`/`:constant`/`:trend`).
- `maxiter::Int=100`, `tol=1e-8` — PMG outer-loop controls.

# Returns
`PMGModel{T}` with the long-run block (`theta`/`theta_se`/`theta_vcov`), per-unit
error-correction speeds `phi_i`, the averaged short-run block, per-unit variances, the
log-likelihood, and convergence diagnostics.

# Notes on the PMG algorithm
The concentrated (profile) likelihood is maximised by **block coordinate ascent**: given
`θ`, each unit's `(φ_i, short-run, σ²_i)` is a closed-form OLS update; given the unit
parameters, `θ` is a closed-form pooled-GLS update
`θ = [Σ_i (φ_i²/σ²_i) Xlag_i'Xlag_i]⁻¹ Σ_i (−φ_i/σ²_i) Xlag_i'r_i` with
`r_i = Δy_i − W_iγ_i − φ_i y_{i,-1}`. The two steps alternate until `‖Δθ‖∞ < tol`. (An
`Optim.optimize` outer step on `θ` with forward-diff gradients is an equivalent
alternative; the closed-form GLS update is used here because it yields the inverse-
information covariance of `θ` directly.)

# References
- Pesaran, M. H., Shin, Y. & Smith, R. P. (1999). *JASA* 94(446), 621–634.
- Pesaran, M. H. & Smith, R. (1995). *Journal of Econometrics* 68(1), 79–113.
- Blackburne, E. F. & Frank, M. W. (2007). *Stata Journal* 7(2), 197–208.
"""
function estimate_pmg(pd::PanelData{T}, y::Symbol, xs::Symbol...;
                      p::Int=1, q::Int=1, method::Symbol=:pmg,
                      trend::Symbol=:constant, maxiter::Int=100,
                      tol::Real=1e-8) where {T<:AbstractFloat}
    isempty(xs) && throw(ArgumentError("at least one long-run regressor is required"))
    y_idx = findfirst(==(String(y)), pd.varnames)
    y_idx === nothing && throw(ArgumentError("Variable :$y not found. Available: $(pd.varnames)"))
    x_idxs = Int[]
    for v in xs
        idx = findfirst(==(String(v)), pd.varnames)
        idx === nothing && throw(ArgumentError("Variable :$v not found. Available: $(pd.varnames)"))
        push!(x_idxs, idx)
    end

    # Split into per-unit series, sorted within group by time.
    ug = sort(unique(pd.group_id))
    ys = Vector{Vector{T}}()
    Xs = Vector{Matrix{T}}()
    for g in ug
        gi = findall(==(g), pd.group_id)
        ord = sortperm(pd.time_id[gi])
        gi = gi[ord]
        push!(ys, T.(pd.data[gi, y_idx]))
        push!(Xs, T.(pd.data[gi, x_idxs]))
    end
    _estimate_pmg_core(ys, Xs, String(y), [String(v) for v in xs];
                       p=p, q=q, method=method, trend=trend, maxiter=maxiter, tol=tol)
end

# Long-matrix + id/time convenience.
function estimate_pmg(y::AbstractVector, X::AbstractMatrix, id::AbstractVector,
                      time::AbstractVector; xnames::Union{Nothing,Vector{String}}=nothing,
                      yname::AbstractString="y", kwargs...)
    T = float(eltype(collect(y)))
    yv = T.(collect(y)); Xm = T.(collect(X))
    length(yv) == size(Xm, 1) == length(id) == length(time) ||
        throw(ArgumentError("y, X, id, time must share the same number of rows"))
    k = size(Xm, 2)
    vnames = xnames === nothing ? ["x$j" for j in 1:k] : xnames
    ug = sort(unique(id))
    ys = Vector{Vector{T}}(); Xs = Vector{Matrix{T}}()
    for g in ug
        gi = findall(==(g), id)
        ord = sortperm(collect(time)[gi])
        gi = gi[ord]
        push!(ys, yv[gi]); push!(Xs, Xm[gi, :])
    end
    _estimate_pmg_core(ys, Xs, String(yname), vnames; kwargs...)
end

# =============================================================================
# Core (operates on per-unit series lists)
# =============================================================================

function _estimate_pmg_core(ys::Vector{Vector{T}}, Xs::Vector{Matrix{T}},
                            yname::String, xnames::Vector{String};
                            p::Int=1, q::Int=1, method::Symbol=:pmg,
                            trend::Symbol=:constant, maxiter::Int=100,
                            tol::Real=1e-8) where {T<:AbstractFloat}
    method in (:pmg, :mg, :dfe) ||
        throw(ArgumentError("method must be :pmg, :mg, or :dfe; got :$method"))
    p >= 1 || throw(ArgumentError("p must be ≥ 1; got $p"))
    q >= 0 || throw(ArgumentError("q must be ≥ 0; got $q"))
    N = length(ys)
    N >= 2 || throw(ArgumentError("need at least 2 units; got $N"))
    k = length(xnames)

    # Per-unit EC design.
    L = max(p, q)
    dys = Vector{Vector{T}}(undef, N); ylags = Vector{Vector{T}}(undef, N)
    Xlags = Vector{Matrix{T}}(undef, N); Ws = Vector{Matrix{T}}(undef, N)
    srnames = String[]
    Ti = zeros(Int, N)
    for i in 1:N
        length(ys[i]) > L + k + (trend === :none ? 0 : 1) + (p - 1) + k * q ||
            throw(ArgumentError("unit $i: sample too short for ARDL($p,$q)"))
        dy, ylag, Xlag, W, srn = _pmg_unit_design(ys[i], Xs[i], p, q, trend;
                                                   yname=yname, xnames=xnames)
        dys[i] = dy; ylags[i] = ylag; Xlags[i] = Xlag; Ws[i] = W
        Ti[i] = length(dy)
        i == 1 && (srnames = srn)
    end
    m_sr = size(Ws[1], 2)

    if method === :mg
        return _pmg_mean_group(ys, Xs, dys, ylags, Xlags, Ws, srnames, Ti,
                               yname, xnames, p, q, trend)
    elseif method === :dfe
        return _pmg_dfe(dys, ylags, Xlags, Ws, srnames, Ti, yname, xnames, p, q)
    else
        return _pmg_pooled_mg(dys, ylags, Xlags, Ws, srnames, Ti, yname, xnames,
                              p, q, maxiter, tol)
    end
end

# =============================================================================
# MG (Pesaran–Smith 1995): per-unit unrestricted ARDL, averaged
# =============================================================================

function _pmg_mean_group(ys, Xs, dys, ylags, Xlags, Ws, srnames, Ti,
                         yname::String, xnames::Vector{String}, p::Int, q::Int,
                         trend::Symbol)
    T = eltype(dys[1])
    N = length(ys)
    k = length(xnames)
    m_sr = size(Ws[1], 2)
    case = _pmg_trend_case(trend)

    theta_i = Matrix{T}(undef, N, k)
    phi_i = zeros(T, N)
    sr_i = Matrix{T}(undef, N, m_sr)
    sigma2_i = zeros(T, N)

    for i in 1:N
        # Long-run θ_i from the EV-08 per-unit ARDL (the MG oracle path).
        mi = estimate_ardl(ys[i], Xs[i]; p=p, q=fill(q, k), case=case,
                           xnames=xnames, yname=yname)
        lri = long_run(mi)
        theta_i[i, :] .= lri.theta
        # φ_i and the short-run block: OLS of the EC form given this unit's θ_i.
        xi = ylags[i] .- Xlags[i] * lri.theta
        Z = hcat(xi, Ws[i])
        b, resid, _ = _pmg_ols(Z, dys[i])
        phi_i[i] = b[1]
        sr_i[i, :] .= b[2:end]
        sigma2_i[i] = dot(resid, resid) / Ti[i]
    end

    theta = vec(mean(theta_i; dims=1))
    # Swamy between-unit covariance: (N(N-1))⁻¹ Σ (θ_i−θ̄)(θ_i−θ̄)'; diag SE = std/√N.
    V = zeros(T, k, k)
    for i in 1:N
        d = theta_i[i, :] .- theta
        V .+= d * d'
    end
    V ./= T(N * (N - 1))
    theta_se = sqrt.(max.(diag(V), zero(T)))

    phi = mean(phi_i)
    phi_se = std(phi_i; corrected=true) / sqrt(T(N))
    sr = vec(mean(sr_i; dims=1))
    sr_se = [std(@view(sr_i[:, j]); corrected=true) / sqrt(T(N)) for j in 1:m_sr]

    loglik = _pmg_loglik(sigma2_i, Ti)
    n_nonconv = count(>=(zero(T)), phi_i)
    PMGModel{T}(:mg, yname, xnames, srnames, theta, theta_se, V, theta_i, phi_i,
                phi, phi_se, sr, sr_se, sr_i, sigma2_i, loglik, N, Ti, p, q,
                n_nonconv, true, 0)
end

# =============================================================================
# PMG (Pesaran–Shin–Smith 1999): common long-run θ via concentrated ML
# =============================================================================

function _pmg_pooled_mg(dys, ylags, Xlags, Ws, srnames, Ti,
                        yname::String, xnames::Vector{String}, p::Int, q::Int,
                        maxiter::Int, tol::Real)
    T = eltype(dys[1])
    N = length(dys)
    k = length(xnames)
    m_sr = size(Ws[1], 2)

    # --- initialise θ at the MG mean (per-unit unrestricted OLS long-run) ---
    theta = zeros(T, k)
    for i in 1:N
        Zi = hcat(ylags[i], Xlags[i], Ws[i])       # y_{-1}, x_{-1}, short-run
        b, _, _ = _pmg_ols(Zi, dys[i])
        phi0 = b[1]
        beta0 = @view b[2:(k+1)]
        theta .+= (-beta0 ./ phi0)                  # θ_i = −β_i/φ_i
    end
    theta ./= T(N)

    phi_i = zeros(T, N); sigma2_i = ones(T, N)
    sr_i = Matrix{T}(undef, N, m_sr)
    converged = false; iters = 0
    for it in 1:maxiter
        iters = it
        # --- inner: given θ, per-unit OLS of Δy on [ξ_i(θ)  W_i] ---
        for i in 1:N
            xi = ylags[i] .- Xlags[i] * theta
            Z = hcat(xi, Ws[i])
            b, resid, _ = _pmg_ols(Z, dys[i])
            phi_i[i] = b[1]
            sr_i[i, :] .= b[2:end]
            sigma2_i[i] = max(dot(resid, resid) / Ti[i], floatmin(T))
        end
        # --- outer: pooled-GLS update of θ ---
        #   r_i = Δy_i − W_iγ_i − φ_i y_{i,-1} = −φ_i Xlag_i θ + ε_i
        A = zeros(T, k, k); bvec = zeros(T, k)
        for i in 1:N
            w = phi_i[i] / sigma2_i[i]
            ri = dys[i] .- Ws[i] * sr_i[i, :] .- phi_i[i] .* ylags[i]
            A .+= (phi_i[i] * w) .* (Xlags[i]' * Xlags[i])
            bvec .+= (-w) .* (Xlags[i]' * ri)
        end
        theta_new = Matrix{T}(robust_inv(Symmetric(A))) * bvec
        Δ = maximum(abs.(theta_new .- theta))
        theta = theta_new
        if Δ < T(tol)
            converged = true
            break
        end
    end

    # --- final inner pass at converged θ (refresh unit params) ---
    for i in 1:N
        xi = ylags[i] .- Xlags[i] * theta
        Z = hcat(xi, Ws[i])
        b, resid, _ = _pmg_ols(Z, dys[i])
        phi_i[i] = b[1]
        sr_i[i, :] .= b[2:end]
        sigma2_i[i] = max(dot(resid, resid) / Ti[i], floatmin(T))
    end

    # --- covariance of θ̂: [Σ_i (φ_i²/σ²_i) Xlag_i' N_i Xlag_i]⁻¹ ---
    Info = zeros(T, k, k)
    for i in 1:N
        xi = ylags[i] .- Xlags[i] * theta
        R = hcat(xi, Ws[i])                          # EC + short-run block
        RtRinv = Matrix{T}(robust_inv(Symmetric(R' * R)))
        # N_i Xlag = Xlag − R (R'R)⁻¹ R' Xlag
        RtX = R' * Xlags[i]
        NX = Xlags[i] .- R * (RtRinv * RtX)
        Info .+= (phi_i[i]^2 / sigma2_i[i]) .* (Xlags[i]' * NX)
    end
    V = Matrix{T}(robust_inv(Symmetric(Info)))
    V = (V .+ V') ./ 2
    theta_se = sqrt.(max.(diag(V), zero(T)))

    phi = mean(phi_i)
    phi_se = std(phi_i; corrected=true) / sqrt(T(N))
    sr = vec(mean(sr_i; dims=1))
    sr_se = [std(@view(sr_i[:, j]); corrected=true) / sqrt(T(N)) for j in 1:m_sr]

    loglik = _pmg_loglik(sigma2_i, Ti)
    n_nonconv = count(>=(zero(T)), phi_i)
    PMGModel{T}(:pmg, yname, xnames, srnames, theta, theta_se, V,
                Matrix{T}(undef, 0, k), phi_i, phi, phi_se, sr, sr_se, sr_i,
                sigma2_i, loglik, N, Ti, p, q, n_nonconv, converged, iters)
end

# =============================================================================
# DFE: pooled within-transformed EC regression with clustered SEs
# =============================================================================

function _pmg_dfe(dys, ylags, Xlags, Ws, srnames, Ti,
                  yname::String, xnames::Vector{String}, p::Int, q::Int)
    T = eltype(dys[1])
    N = length(dys)
    k = length(xnames)
    m_sr = size(Ws[1], 2)

    # Stack the pooled EC regression: Δy on [y_{-1}  x_{-1}  short-run], unit FE.
    #   linear form: Δy = φ y_{-1} + β' x_{-1} + γ' W + α_i ,  θ = −β/φ.
    dy_all = vcat(dys...)
    # Design excluding the deterministic intercept (absorbed by unit FE); keep a
    # trend column if present. Column 1 = y_{-1}; cols 2..k+1 = x_{-1}; then short-run
    # (dropping the "(Intercept)" column, which the within transform removes).
    has_int = !isempty(srnames) && srnames[1] == "(Intercept)"
    sr_keep = has_int ? (2:m_sr) : (1:m_sr)
    sr_kept_names = String[srnames[j] for j in sr_keep]

    blocks = Vector{Matrix{T}}(undef, N)
    grp = Int[]
    for i in 1:N
        Zi = hcat(ylags[i], Xlags[i], @view(Ws[i][:, sr_keep]))
        blocks[i] = Zi
        append!(grp, fill(i, Ti[i]))
    end
    Z_all = vcat(blocks...)
    n = size(Z_all, 1)
    kk = size(Z_all, 2)

    # Within (unit) demeaning to absorb α_i.
    Zw = copy(Z_all); yw = copy(dy_all)
    row = 0
    for i in 1:N
        rng = (row + 1):(row + Ti[i])
        yw[rng] .-= mean(@view yw[rng])
        for c in 1:kk
            Zw[rng, c] .-= mean(@view Zw[rng, c])
        end
        row += Ti[i]
    end

    ZtZinv = Matrix{T}(robust_inv(Symmetric(Zw' * Zw)))
    b = ZtZinv * (Zw' * yw)
    resid = yw .- Zw * b
    # Cluster-robust (by unit) vcov via the existing panel machinery.
    Vc = _panel_cluster_vcov(Zw, resid, ZtZinv, grp; n_absorbed=N)

    phi = b[1]
    beta = b[2:(k+1)]
    sr_kept = b[(k+2):end]

    # θ = −β/φ with delta-method covariance from Vc (indices 1=φ, 2..k+1=β).
    theta = -beta ./ phi
    Vtheta = zeros(T, k, k)
    for a in 1:k, bcol in 1:k
        # Jacobian of θ_a wrt (φ, β): ∂θ_a/∂φ = β_a/φ²; ∂θ_a/∂β_a = −1/φ.
        ga = zeros(T, kk); gb = zeros(T, kk)
        ga[1] = beta[a] / phi^2; ga[a+1] = -one(T) / phi
        gb[1] = beta[bcol] / phi^2; gb[bcol+1] = -one(T) / phi
        Vtheta[a, bcol] = dot(ga, Vc * gb)
    end
    Vtheta = (Vtheta .+ Vtheta') ./ 2
    theta_se = sqrt.(max.(diag(Vtheta), zero(T)))

    phi_se = sqrt(max(Vc[1, 1], zero(T)))
    phi_i = fill(phi, N)

    # Short-run block: keep the pooled slopes (common across units for DFE).
    sr = sr_kept
    sr_se = sqrt.(max.(diag(Vc)[(k+2):end], zero(T)))
    sigma2 = dot(resid, resid) / T(max(n - kk - N, 1))
    sigma2_i = fill(sigma2, N)
    loglik = -T(n) / 2 * (log(2 * T(π)) + log(max(sigma2, floatmin(T))) + one(T))

    n_nonconv = phi >= zero(T) ? N : 0
    PMGModel{T}(:dfe, yname, xnames, sr_kept_names, theta, theta_se, Vtheta,
                Matrix{T}(undef, 0, k), phi_i, phi, phi_se, sr, sr_se,
                Matrix{T}(undef, 0, length(sr)), sigma2_i, loglik, N, Ti, p, q,
                n_nonconv, true, 0)
end

# Concentrated Gaussian log-likelihood from per-unit ML variances.
function _pmg_loglik(sigma2_i::Vector{T}, Ti::Vector{Int}) where {T<:AbstractFloat}
    ll = zero(T)
    for i in eachindex(sigma2_i)
        ll -= T(Ti[i]) / 2 * (log(2 * T(π)) + log(max(sigma2_i[i], floatmin(T))) + one(T))
    end
    ll
end

# =============================================================================
# Hausman selection test (PMG-vs-MG, DFE-vs-MG)
# =============================================================================

"""
    hausman_test(efficient::PMGModel, consistent::PMGModel) -> PanelTestResult

Generalized Hausman test between two panel-ARDL estimators. `consistent` must be the
always-consistent Mean Group model; `efficient` is the estimator efficient under
`H0` (PMG for long-run homogeneity, or DFE). The statistic is the quadratic form on the
common long-run coefficients

```math
H = (\\hat\\theta_e - \\hat\\theta_c)'\\,(V_c - V_e)^{-1}\\,(\\hat\\theta_e - \\hat\\theta_c)
    \\;\\sim\\; \\chi^2_r,
```

computed via the same generalized-inverse machinery as the FE-vs-RE Hausman test
(`_hausman_quadratic_form`). For PMG-vs-MG, `H0` is long-run homogeneity: failing to
reject supports the pooled (PMG) long-run vector.

# References
- Hausman, J. A. (1978). *Econometrica* 46(6), 1251–1271.
- Pesaran, Shin & Smith (1999); Blackburne & Frank (2007).
"""
function hausman_test(efficient::PMGModel{T}, consistent::PMGModel{T}) where {T}
    consistent.method === :mg ||
        throw(ArgumentError("second argument must be the Mean Group model (:mg); got :$(consistent.method)"))
    length(efficient.theta) == length(consistent.theta) ||
        throw(ArgumentError("the two models must share the same long-run dimension"))
    db = efficient.theta .- consistent.theta
    dV = consistent.theta_vcov .- efficient.theta_vcov
    chi2, df, nonpsd = _hausman_quadratic_form(db, dV)
    pval = chi2 > zero(T) ? T(1 - cdf(Chisq(df), chi2)) : one(T)
    name = "Hausman test ($(uppercase(string(efficient.method))) vs MG)"
    h0 = efficient.method === :pmg ? "long-run homogeneity" : "$(uppercase(string(efficient.method))) consistent"
    desc = (nonpsd ? "[non-PSD dV] " : "") * (pval < T(0.05) ?
        "Reject H0 ($h0): use MG (p=$(_format_pvalue(pval)))" :
        "Fail to reject H0 ($h0): $(uppercase(string(efficient.method))) preferred (p=$(_format_pvalue(pval)))")
    PanelTestResult{T}(name, chi2, pval, df, desc)
end

# =============================================================================
# StatsAPI interface
# =============================================================================

StatsAPI.coef(m::PMGModel) = m.theta
StatsAPI.vcov(m::PMGModel) = m.theta_vcov
StatsAPI.stderror(m::PMGModel) = m.theta_se
StatsAPI.nobs(m::PMGModel) = sum(m.T_i)
StatsAPI.loglikelihood(m::PMGModel) = m.loglik
StatsAPI.islinear(::PMGModel) = true

# =============================================================================
# Display
# =============================================================================

const _PMG_METHOD_DESC = Dict(
    :pmg => "Pooled Mean Group (common long-run)",
    :mg  => "Mean Group (per-unit averaged)",
    :dfe => "Dynamic Fixed Effects (pooled)",
)

function Base.show(io::IO, m::PMGModel{T}) where {T}
    spec = Any[
        "Estimator"       get(_PMG_METHOD_DESC, m.method, string(m.method));
        "Dependent"       m.yname;
        "Model"           "ARDL($(m.p), $(m.q)) — EC form";
        "Units (N)"       m.N;
        "Obs. (Σ Tᵢ)"     sum(m.T_i);
        "Avg. Tᵢ"         _fmt(mean(m.T_i); digits=1);
        "Log-lik."        _fmt(m.loglik; digits=2);
        "φ ≥ 0 units"     m.n_nonconv
    ]
    _pretty_table(io, spec; title="Panel ARDL (PMG/MG/DFE)",
                  column_labels=["Specification", ""], alignment=[:l, :r])

    _coef_table(io, "Long-run coefficients (θ)", m.xnames, m.theta, m.theta_se; dist=:z)

    # Error-correction speed φ.
    ec = Any[
        "Speed of adj. φ"  _fmt(m.phi);
        "  Std.Err."       _fmt(m.phi_se);
        "  t-ratio"        _fmt(m.phi / m.phi_se)
    ]
    _pretty_table(io, ec; title="Error-correction speed" *
                  (m.method === :dfe ? " (common)" : " (mean)"),
                  column_labels=["", "Value"], alignment=[:l, :r])

    if !isempty(m.srnames)
        _coef_table(io, "Short-run coefficients" *
                    (m.method === :dfe ? " (common)" : " (mean)"),
                    m.srnames, m.sr, m.sr_se; dist=:z)
    end

    if m.n_nonconv > 0
        println(io, "Note: $(m.n_nonconv) unit(s) have φ_i ≥ 0 (non-error-correcting).")
    end
    if m.method === :pmg && !m.converged
        println(io, "Warning: PMG outer loop did not converge in $(m.iters) iterations.")
    end
    _sig_legend(io)
end

"""
    report(m::PMGModel)

Print the panel-ARDL specification, the long-run coefficient block `θ`, the
error-correction speed `φ`, and the averaged (or common, for DFE) short-run block. Units
with `φ_i ≥ 0` (non-error-correcting) are flagged.
"""
report(m::PMGModel) = show(stdout, m)
report(io::IO, m::PMGModel) = show(io, m)
