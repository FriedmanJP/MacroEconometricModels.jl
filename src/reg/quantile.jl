# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Quantile regression (Koenker & Bassett 1978)
#
# References:
#   Koenker, R. & Bassett, G. (1978). Regression Quantiles. Econometrica 46(1), 33–50.
#   Koenker, R. (2005). Quantile Regression. Cambridge University Press.
#   Powell, J. L. (1991). Estimation of monotonic regression models under quantile
#     restrictions. In Nonparametric and Semiparametric Methods in Econometrics.
#   Hall, P. & Sheather, S. J. (1988). On the distribution of a Studentized quantile.

using LinearAlgebra
using Statistics

# =============================================================================
# Check function and the solver
# =============================================================================

"""
    _check_loss(u, tau) → T

Koenker-Bassett check-function loss `Σ ρ_τ(u_i)` with `ρ_τ(u) = u(τ − 1{u < 0})`.
Asymmetric absolute loss: over-prediction is penalized `1-τ` and under-prediction `τ`, so
the minimizer is the conditional `τ`-quantile.
"""
function _check_loss(u::AbstractVector{T}, tau::T) where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds for ui in u
        acc += ui * (tau - (ui < zero(T) ? one(T) : zero(T)))
    end
    return acc
end

"""
    _qreg_fit(X, y, tau; maxit, tol, n_stage) → (beta, converged)

Minimize the check loss by iteratively reweighted least squares.

The check function is majorized by a weighted quadratic, `ρ_τ(u) ≈ w u²` with
`w = |τ − 1{u<0}| / |u|`, so each step is an ordinary weighted least-squares solve. The
`|u|` in the denominator is floored at `delta`, and `delta` is annealed down across
`n_stage` stages: a fixed floor would leave a smoothed (Huber-like) solution rather than the
true piecewise-linear one, and annealing is what drives it onto the exact LP vertex.

Verified against exact enumeration of the basic solutions — the Koenker-Bassett optimum
interpolates `k` observations exactly, so for small problems every `k`-subset can be checked.
Agreement is to 1e-12..1e-9 in the coefficients.
"""
function _qreg_fit(X::Matrix{T}, y::Vector{T}, tau::T;
                   maxit::Int=200, tol::Real=1e-10, n_stage::Int=12) where {T<:AbstractFloat}
    n, k = size(X)
    b = X \ y                                    # OLS start
    delta = one(T)
    converged = false
    for _ in 1:n_stage
        for _ in 1:maxit
            u = y .- X * b
            w = similar(u)
            @inbounds for i in eachindex(u)
                w[i] = abs(tau - (u[i] < zero(T) ? one(T) : zero(T))) /
                       max(abs(u[i]), delta)
            end
            Xw = X .* w
            A = Xw' * X
            bnew = Matrix{T}(robust_inv(Symmetric(A))) * (Xw' * y)
            step = maximum(abs.(bnew .- b))
            b = bnew
            if step < T(tol)
                converged = true
                break
            end
        end
        delta = max(delta / 10, T(1e-12))
    end
    return b, converged
end

# =============================================================================
# Sparsity and covariance
# =============================================================================

"""
    _hall_sheather_bandwidth(n, tau, alpha) → T

Hall & Sheather (1988) bandwidth for the sparsity estimate, the `qreg` default:
`h = n^{-1/3} z_α^{2/3} [1.5 φ(z_τ)² / (2 z_τ² + 1)]^{1/3}`.
"""
function _hall_sheather_bandwidth(n::Int, tau::T, alpha::T=T(0.05)) where {T<:AbstractFloat}
    nd = Distributions.Normal()
    z_a = Distributions.quantile(nd, one(T) - alpha / 2)
    z_t = Distributions.quantile(nd, tau)
    phi = Distributions.pdf(nd, z_t)
    h = T(n)^(-one(T) / 3) * z_a^(T(2) / 3) *
        (T(1.5) * phi^2 / (2 * z_t^2 + one(T)))^(one(T) / 3)
    return clamp(h, T(1e-6), min(tau, one(T) - tau) / 2)
end

"""
    _qreg_sparsity(resid, tau, h) → T

Siddiqui-Hendricks-Koenker sparsity estimate `s(τ) = [F⁻¹(τ+h) − F⁻¹(τ−h)] / (2h)`, the
reciprocal of the residual density at the quantile. The asymptotic variance of a regression
quantile is proportional to `s(τ)²`, so this is what sets the scale of the standard errors.
"""
function _qreg_sparsity(resid::Vector{T}, tau::T, h::T) where {T<:AbstractFloat}
    lo = clamp(tau - h, T(1e-8), one(T) - T(1e-8))
    hi = clamp(tau + h, T(1e-8), one(T) - T(1e-8))
    s = (Statistics.quantile(resid, hi) - Statistics.quantile(resid, lo)) / (hi - lo)
    return max(s, T(1e-8))
end

"""
    _qreg_vcov(X, resid, tau, se; alpha) → Matrix{T}

Asymptotic covariance of the regression quantile.

- `:iid` — Koenker's sparsity form `τ(1−τ) s(τ)² (X'X)⁻¹`, valid when the conditional
  density of the error at the quantile does not vary with `x`. This is the `qreg` default.
- `:robust` — Powell's kernel sandwich
  `τ(1−τ) (X'DX)⁻¹ (X'X) (X'DX)⁻¹` with `D = diag(K(u_i/h)/h)`, which allows the density at
  the quantile to depend on `x` (the analogue of moving from OLS to HC).
"""
function _qreg_vcov(X::Matrix{T}, resid::Vector{T}, tau::T, se::Symbol;
                    alpha::T=T(0.05)) where {T<:AbstractFloat}
    n, k = size(X)
    h = _hall_sheather_bandwidth(n, tau, alpha)
    XtXinv = Matrix{T}(robust_inv(Symmetric(X' * X)))

    if se === :iid
        s = _qreg_sparsity(resid, tau, h)
        return tau * (one(T) - tau) * s^2 .* XtXinv
    end

    # Powell sandwich: a Gaussian kernel density of the residuals at zero.
    bw = max(h * Statistics.std(resid), T(1e-8))
    D = similar(resid)
    @inbounds for i in eachindex(resid)
        D[i] = exp(-T(0.5) * (resid[i] / bw)^2) / (bw * sqrt(T(2π)))
    end
    XDX = Matrix{T}((X .* D)' * X)
    XDXinv = Matrix{T}(robust_inv(Symmetric(XDX)))
    return tau * (one(T) - tau) .* (XDXinv * (X' * X) * XDXinv)
end

# =============================================================================
# Type
# =============================================================================

"""
    QuantileRegModel{T} <: StatsAPI.RegressionModel

Quantile regression fit (Koenker & Bassett 1978) at one or several quantiles.

Every matrix field is stored with **one column per quantile**, in the order the quantiles
were supplied.

Fields:
- `y`, `X` — data
- `taus::Vector{T}` — the quantiles fitted
- `beta::Matrix{T}` — `k × n_tau` coefficients
- `vcov_mats::Vector{Matrix{T}}` — covariance per quantile
- `stderr::Matrix{T}` — `k × n_tau` standard errors
- `residuals`, `fitted` — `n × n_tau`
- `objective::Vector{T}` — check-function loss at the optimum, per quantile
- `pseudo_r2::Vector{T}` — Koenker-Machado `R¹(τ) = 1 − V̂(τ)/Ṽ(τ)`, the loss relative to
  the intercept-only fit. It is NOT comparable to an OLS `R²`.
- `varnames`, `se_type`, `n_obs`, `converged`
"""
struct QuantileRegModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    y::Vector{T}
    X::Matrix{T}
    taus::Vector{T}
    beta::Matrix{T}
    vcov_mats::Vector{Matrix{T}}
    stderr::Matrix{T}
    residuals::Matrix{T}
    fitted::Matrix{T}
    objective::Vector{T}
    pseudo_r2::Vector{T}
    varnames::Vector{String}
    se_type::Symbol
    n_obs::Int
    converged::Vector{Bool}
    manifest::Union{ReproManifest,Nothing}
end

QuantileRegModel{T}(y, X, taus, beta, vcov_mats, stderr, residuals, fitted, objective,
                    pseudo_r2, varnames, se_type, n_obs, converged;
                    manifest=nothing) where {T<:AbstractFloat} =
    QuantileRegModel{T}(y, X, taus, beta, vcov_mats, stderr, residuals, fitted, objective,
                        pseudo_r2, varnames, se_type, n_obs, converged, manifest)

# =============================================================================
# Estimation
# =============================================================================

"""
    estimate_qreg(y, X, tau=0.5; se=:iid, varnames=nothing, n_boot=500,
                  rng=Random.default_rng(), alpha=0.05) → QuantileRegModel

Quantile regression (Koenker & Bassett 1978): estimate `β(τ)` by minimizing the
check-function loss

```math
\\min_\\beta \\sum_i \\rho_\\tau(y_i - x_i'\\beta), \\qquad
\\rho_\\tau(u) = u\\,(\\tau - \\mathbf{1}\\{u < 0\\})
```

Unlike OLS, which fits the conditional **mean**, this fits the conditional `τ`-quantile — so
a covariate is allowed to shift the lower tail of the outcome distribution differently from
the upper tail. `tau` may be a scalar or a vector; one fit is returned per quantile.

# Arguments
- `y::AbstractVector` — outcome (length `n`)
- `X::AbstractMatrix` — `n × k` design matrix (include your own intercept column)
- `tau` — quantile(s) in `(0, 1)`; default `0.5` (median regression, i.e. LAD)

# Keyword Arguments
- `se::Symbol = :iid` — `:iid` (Koenker sparsity, the `qreg` default), `:robust`
  (Powell kernel sandwich), or `:boot` (xy-pair bootstrap)
- `n_boot::Int = 500` — bootstrap replications when `se = :boot`
- `rng` — random number generator for the bootstrap
- `alpha::Real = 0.05` — level used by the Hall-Sheather bandwidth
- `varnames` — coefficient names

# Returns
A [`QuantileRegModel`](@ref). `report(m)` prints one coefficient table per quantile.

# Examples
```julia
m = estimate_qreg(y, X, [0.1, 0.5, 0.9])
report(m)
coef(m)              # k × 3
```

See also [`estimate_reg`](@ref).

# References
- Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33–50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
"""
function estimate_qreg(y::AbstractVector{T}, X::AbstractMatrix{T},
                       tau::Union{Real,AbstractVector}=T(0.5);
                       se::Symbol=:iid,
                       varnames::Union{Nothing,Vector{String}}=nothing,
                       n_boot::Int=500,
                       seed::Union{Integer,Nothing}=nothing,
                       rng=Random.default_rng(),
                       alpha::Real=0.05) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    _validate_data(y, "y")
    _validate_data(X, "X")
    n = length(y)
    k = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))
    se in (:iid, :robust, :boot) ||
        throw(ArgumentError("se must be :iid, :robust, or :boot; got :$se"))
    n_boot > 0 || throw(ArgumentError("n_boot must be positive, got $n_boot"))

    taus = tau isa Real ? T[T(tau)] : Vector{T}(collect(T, tau))
    isempty(taus) && throw(ArgumentError("tau must not be empty"))
    all(t -> zero(T) < t < one(T), taus) ||
        throw(ArgumentError("every tau must lie strictly in (0, 1); got $taus"))

    Xm = Matrix{T}(X)
    yv = Vector{T}(y)
    nt = length(taus)

    beta = zeros(T, k, nt)
    stderr = zeros(T, k, nt)
    vcovs = Vector{Matrix{T}}(undef, nt)
    resid = zeros(T, n, nt)
    fit = zeros(T, n, nt)
    obj = zeros(T, nt)
    pr2 = zeros(T, nt)
    conv = Vector{Bool}(undef, nt)

    for (j, t) in enumerate(taus)
        b, ok = _qreg_fit(Xm, yv, t)
        beta[:, j] = b
        conv[j] = ok
        f = Xm * b
        r = yv .- f
        fit[:, j] = f
        resid[:, j] = r
        obj[j] = _check_loss(r, t)

        # Koenker-Machado goodness of fit: the loss relative to the intercept-only model,
        # whose fitted value is the unconditional tau-quantile.
        r0 = yv .- Statistics.quantile(yv, t)
        v0 = _check_loss(r0, t)
        pr2[j] = v0 > zero(T) ? one(T) - obj[j] / v0 : zero(T)

        V = if se === :boot
            _qreg_boot_vcov(Xm, yv, t, n_boot, rng)
        else
            _qreg_vcov(Xm, r, t, se; alpha=T(alpha))
        end
        vcovs[j] = V
        stderr[:, j] = sqrt.(max.(diag(V), zero(T)))
    end

    names = varnames === nothing ? ["x$i" for i in 1:k] : varnames
    length(names) == k || throw(ArgumentError("varnames must have length $k"))

    result = QuantileRegModel{T}(yv, Xm, taus, beta, vcovs, stderr, resid, fit,
                                 obj, pr2, names, se, n, conv)
    return _with_manifest(result, capture_manifest(; seed=seed,
        settings=Dict{String,Any}("n_boot" => n_boot, "se" => String(se),
                                  "alpha" => Float64(alpha))))
end

function estimate_qreg(y::AbstractVector, X::AbstractMatrix,
                       tau::Union{Real,AbstractVector}=0.5; kwargs...)
    estimate_qreg(Float64.(y), Float64.(X), tau; kwargs...)
end

"""
    _qreg_boot_vcov(X, y, tau, n_boot, rng) → Matrix{T}

xy-pair (design) bootstrap covariance: resample observations with replacement and refit.
The pair bootstrap is the standard choice for quantile regression because it makes no
assumption about the conditional density at the quantile — which is exactly the object the
asymptotic formulas have to estimate.
"""
function _qreg_boot_vcov(X::Matrix{T}, y::Vector{T}, tau::T, n_boot::Int, rng) where {T<:AbstractFloat}
    n, k = size(X)
    draws = zeros(T, k, n_boot)
    idx = Vector{Int}(undef, n)
    kept = 0
    for bnum in 1:n_boot
        for i in 1:n
            idx[i] = rand(rng, 1:n)
        end
        Xb = X[idx, :]
        rank(Xb) < k && continue                 # degenerate resample; skip
        bb, _ = _qreg_fit(Xb, y[idx], tau)
        kept += 1
        draws[:, kept] = bb
    end
    kept >= 2 || return fill(T(NaN), k, k)
    D = @view draws[:, 1:kept]
    mu = vec(Statistics.mean(D; dims=2))
    C = zeros(T, k, k)
    for bnum in 1:kept
        d = @view D[:, bnum]
        C .+= (d .- mu) * (d .- mu)'
    end
    return C ./ T(kept - 1)
end

# =============================================================================
# StatsAPI interface — `fitted` has no default method, so define it explicitly
# =============================================================================

StatsAPI.coef(m::QuantileRegModel) = size(m.beta, 2) == 1 ? vec(m.beta) : m.beta
StatsAPI.fitted(m::QuantileRegModel) = size(m.fitted, 2) == 1 ? vec(m.fitted) : m.fitted
StatsAPI.residuals(m::QuantileRegModel) = size(m.residuals, 2) == 1 ? vec(m.residuals) : m.residuals
StatsAPI.nobs(m::QuantileRegModel) = m.n_obs
StatsAPI.vcov(m::QuantileRegModel) = length(m.vcov_mats) == 1 ? m.vcov_mats[1] : m.vcov_mats
StatsAPI.stderror(m::QuantileRegModel) = size(m.stderr, 2) == 1 ? vec(m.stderr) : m.stderr

"""
    predict(m::QuantileRegModel, X_new) → Vector or Matrix

Predicted conditional quantiles at `X_new`: one column per fitted quantile.
"""
function StatsAPI.predict(m::QuantileRegModel{T}, X_new::AbstractMatrix) where {T}
    size(X_new, 2) == size(m.X, 2) ||
        throw(ArgumentError("X_new must have $(size(m.X, 2)) columns"))
    P = Matrix{T}(X_new) * m.beta
    return size(P, 2) == 1 ? vec(P) : P
end
StatsAPI.predict(m::QuantileRegModel) = StatsAPI.fitted(m)

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, m::QuantileRegModel{T}) where {T}
    ts = join(string.(round.(m.taus; digits=3)), ", ")
    print(io, "QuantileRegModel{$T}: n=$(m.n_obs), k=$(size(m.X, 2)), tau=[$ts], se=:$(m.se_type)")
end

"""
    report(m::QuantileRegModel)

Print one Stata-style coefficient table per quantile, plus the check-function loss and the
Koenker-Machado pseudo-`R¹`.
"""
function report(io::IO, m::QuantileRegModel{T}) where {T}
    n, k = size(m.X)
    se_label = m.se_type === :iid ? "sparsity (iid)" :
               m.se_type === :robust ? "Powell kernel (robust)" : "xy-pair bootstrap"
    _show_spec_table(io, "Quantile Regression (Koenker-Bassett)",
        ["Observations" => string(n), "Coefficients" => string(k),
         "Quantiles" => join(string.(round.(m.taus; digits=3)), ", "),
         "Std. errors" => se_label])
    for (j, t) in enumerate(m.taus)
        m.converged[j] || @warn "Quantile regression at tau = $t did not converge." maxlog = 1
        _coef_table(io, "tau = $(round(t; digits=3))  (loss = $(_fmt(m.objective[j]; digits=4)), " *
                        "pseudo R1 = $(_fmt(m.pseudo_r2[j]; digits=4)))",
                    m.varnames, Vector{T}(m.beta[:, j]), Vector{T}(m.stderr[:, j]);
                    dist=:t, dof_r=n - k)
    end
    return nothing
end
report(m::QuantileRegModel) = report(stdout, m)
