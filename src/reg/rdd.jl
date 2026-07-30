# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Regression discontinuity (Calonico-Cattaneo-Titiunik)
#
# References:
#   Calonico, S., Cattaneo, M. D. & Titiunik, R. (2014). Robust Nonparametric Confidence
#     Intervals for Regression-Discontinuity Designs. Econometrica 82(6), 2295–2326.
#   Calonico, S., Cattaneo, M. D., Farrell, M. H. & Titiunik, R. (2017). rdrobust:
#     Software for Regression-Discontinuity Designs. Stata Journal 17(2), 372–404.
#   Cattaneo, M. D., Idrobo, N. & Titiunik, R. (2020). A Practical Introduction to
#     Regression Discontinuity Designs. Cambridge University Press.

using LinearAlgebra
using Statistics

# =============================================================================
# Kernels
# =============================================================================

"""
    _rd_kernel(u, kernel) → T

Kernel weight at scaled distance `u = (x − c)/h`, zero outside `|u| ≤ 1`.

- `:triangular` — `1 − |u|`, the `rdrobust` default and MSE-optimal for boundary estimation
- `:epanechnikov` — `0.75(1 − u²)`
- `:uniform` — `1`
"""
@inline function _rd_kernel(u::T, kernel::Symbol) where {T<:AbstractFloat}
    au = abs(u)
    au > one(T) && return zero(T)
    if kernel === :triangular
        return one(T) - au
    elseif kernel === :epanechnikov
        return T(0.75) * (one(T) - u^2)
    else
        return one(T)
    end
end

# =============================================================================
# One-sided local polynomial pieces
# =============================================================================

"""
    _rd_side_weights(x, c, h, p, kernel, right) → (idx, W, Xd)

Observations inside the window on one side of the cutoff, their kernel weights, and the
polynomial design `[1, (x−c), …, (x−c)^p]`.

`right = true` takes `x ≥ c`, otherwise `x < c`. Returns empty arrays when the side has no
observations in the window.
"""
function _rd_side_weights(x::Vector{T}, c::T, h::T, p::Int, kernel::Symbol,
                          right::Bool) where {T<:AbstractFloat}
    idx = Int[]
    @inbounds for i in eachindex(x)
        d = x[i] - c
        inside = right ? (d >= zero(T)) : (d < zero(T))
        inside && abs(d) <= h && push!(idx, i)
    end
    m = length(idx)
    W = zeros(T, m)
    Xd = zeros(T, m, p + 1)
    @inbounds for (r, i) in enumerate(idx)
        d = x[i] - c
        W[r] = _rd_kernel(d / h, kernel)
        acc = one(T)
        for j in 0:p
            Xd[r, j + 1] = acc
            acc *= d
        end
    end
    return idx, W, Xd
end

"""
    _rd_side_linear(x, c, h, b, p, q, kernel, right, n) → (w_conv, w_bc)

Weight vectors (length `n`) making the one-sided intercept estimators **linear in `y`**:
`μ̂(c) = w_conv' y` and its bias-corrected counterpart `μ̂_bc(c) = w_bc' y`.

The conventional piece is the usual local-polynomial intercept
`e₁' Γ_p⁻¹ X_p' W_p`. Calonico, Cattaneo & Titiunik correct it by the leading bias term,
whose curvature is estimated by an order-`q = p+1` fit at the pilot bandwidth `b`:

```math
\\hat\\mu_{bc}(c) = e_1' \\Gamma_p^{-1}\\left[X_p' W_p - \\Lambda_p \\, e_{p+2}' \\Gamma_q^{-1} X_q' W_q\\right] y
```

with `Λ_p = X_p' W_p (x−c)^{p+1}`.

Writing both as explicit weights on `y` is what makes the **robust** variance exact: it is
just `w_bc' diag(σ²) w_bc`, with the sampling variability of the estimated bias already
inside `w_bc`. That is the entire point of CCT — the conventional interval is too narrow at
the MSE-optimal bandwidth precisely because it ignores that term.
"""
function _rd_side_linear(x::Vector{T}, c::T, h::T, b::T, p::Int, q::Int,
                         kernel::Symbol, right::Bool, n::Int) where {T<:AbstractFloat}
    idx_p, Wp, Xp = _rd_side_weights(x, c, h, p, kernel, right)
    length(idx_p) > p || return (zeros(T, n), zeros(T, n), false)

    Gp = Matrix{T}((Xp .* Wp)' * Xp)
    Gpi = Matrix{T}(robust_inv(Symmetric(Gp)))
    # e1' Γ_p^{-1} X_p' W_p  →  a length-m row of weights
    row_p = (Gpi[1, :]' * (Xp .* Wp)')          # 1 × m
    w_conv = zeros(T, n)
    @inbounds for (r, i) in enumerate(idx_p)
        w_conv[i] += row_p[r]
    end

    # Bias term: Λ_p = X_p' W_p (x−c)^{p+1}
    d_p = T[x[i] - c for i in idx_p]
    Lam = Matrix{T}((Xp .* Wp)') * (d_p .^ (p + 1))     # (p+1) vector
    scal = dot(Gpi[1, :], Lam)                          # e1' Γ_p^{-1} Λ_p  (a scalar)

    idx_q, Wq, Xq = _rd_side_weights(x, c, b, q, kernel, right)
    w_bc = copy(w_conv)
    ok = true
    if length(idx_q) > q
        Gq = Matrix{T}((Xq .* Wq)' * Xq)
        Gqi = Matrix{T}(robust_inv(Symmetric(Gq)))
        # e_{p+2}' Γ_q^{-1} X_q' W_q  — the (p+1)-th derivative coefficient, times (p+1)!
        row_q = (Gqi[p + 2, :]' * (Xq .* Wq)')
        @inbounds for (r, i) in enumerate(idx_q)
            w_bc[i] -= scal * row_q[r]
        end
    else
        ok = false                                       # pilot window too thin
    end
    return w_conv, w_bc, ok
end

"""
    _rd_sigma2(y, x, c, h, p, kernel) → Vector{T}

Heteroskedasticity-robust residual variances `σ̂²_i`, one per observation, from the
one-sided local-polynomial fits at bandwidth `h`. Observations outside every window get
zero (they receive zero weight anyway).
"""
function _rd_sigma2(y::Vector{T}, x::Vector{T}, c::T, h::T, p::Int,
                    kernel::Symbol) where {T<:AbstractFloat}
    n = length(y)
    s2 = zeros(T, n)
    for right in (false, true)
        idx, W, Xd = _rd_side_weights(x, c, h, p, kernel, right)
        length(idx) > p + 1 || continue
        ys = y[idx]
        G = Matrix{T}((Xd .* W)' * Xd)
        beta = Matrix{T}(robust_inv(Symmetric(G))) * ((Xd .* W)' * ys)
        r = ys .- Xd * beta
        @inbounds for (rr, i) in enumerate(idx)
            s2[i] = r[rr]^2
        end
    end
    return s2
end

# =============================================================================
# Bandwidth selection
# =============================================================================

"""
    _rd_bandwidth(y, x, c, p, kernel) → (h, b)

MSE-optimal bandwidth and its bias-estimation pilot, by the Calonico-Cattaneo-Titiunik
plug-in rule.

The asymptotic MSE of the RD estimator balances a variance term of order `1/(nh)` against a
squared-bias term of order `h^{2(p+1)}`, giving

```math
h_{MSE} = \\left[\\frac{(2p+1)\\,V}{2(p+1)\\,n\\,B^2}\\right]^{1/(2p+3)}
```

The constants are estimated from **global** polynomial pre-fits, as `rdbwselect` does: `B`
from the `(p+1)`-th derivative of an order-`(p+2)` fit on each side (the curvature that
drives the bias), and `V` from the residual variance near the cutoff scaled by the density
of the running variable there. The pilot `b` uses the same rule one order up.

The result is clamped to the observed support so a degenerate pre-fit cannot return a
bandwidth that empties or swallows the sample.
"""
function _rd_bandwidth(y::Vector{T}, x::Vector{T}, c::T, p::Int,
                       kernel::Symbol) where {T<:AbstractFloat}
    n = length(y)
    d = x .- c
    left = d .< zero(T)
    right = .!left
    range_l = maximum(abs, d[left]; init=zero(T))
    range_r = maximum(abs, d[right]; init=zero(T))
    span = max(min(range_l, range_r), T(1e-8))

    # Curvature: the (p+1)-th derivative from an order-(p+2) global fit on each side.
    function curvature(mask, order)
        m = count(mask)
        m > order + 2 || return zero(T)
        dm = d[mask]; ym = y[mask]
        Xg = hcat((dm .^ j for j in 0:order)...)
        bg = Matrix{T}(robust_inv(Symmetric(Matrix{T}(Xg' * Xg)))) * (Xg' * ym)
        # coefficient on d^(p+1) is the (p+1)-th derivative over (p+1)!
        return abs(bg[p + 2])
    end
    curv = max(curvature(left, p + 2), curvature(right, p + 2))
    curv = max(curv, T(1e-10))

    # Variance scale: residual variance near the cutoff, and the density of x there.
    near = abs.(d) .<= max(span / 4, T(1e-8))
    v_hat = count(near) > 2 ? Statistics.var(y[near]) : Statistics.var(y)
    v_hat = max(v_hat, T(1e-12))
    f_hat = max(count(near) / (T(n) * 2 * max(span / 4, T(1e-8))), T(1e-8))

    Vc = v_hat / f_hat
    Bc = curv
    h = (T(2 * p + 1) * Vc / (T(2 * (p + 1)) * T(n) * Bc^2))^(one(T) / T(2 * p + 3))
    h = clamp(h, span / T(50), span)
    # Pilot for the bias: one polynomial order up, hence a wider window.
    bpilot = clamp(h * T(n)^(one(T) / T((2 * p + 3) * (2 * p + 5))) * T(1.5), h, span)
    return h, bpilot
end

# =============================================================================
# Result type
# =============================================================================

"""
    RDDResult{T}

Regression-discontinuity estimate (Calonico, Cattaneo & Titiunik 2014).

Three estimates are reported, and the difference between them is the point of the method:

| Name | Estimate | Standard error |
|---|---|---|
| conventional | local polynomial of order `p` at `h` | its own variance |
| bias-corrected | conventional minus the estimated leading bias | conventional SE |
| **robust** | bias-corrected | variance that also accounts for estimating the bias |

The conventional interval under-covers at the MSE-optimal bandwidth, because that bandwidth
is deliberately large enough for the bias to matter. **Report the robust interval.**

Fields:
- `tau_conventional`, `tau_bias_corrected` — point estimates of the jump
- `se_conventional`, `se_robust` — standard errors
- `ci_conventional`, `ci_robust` — `(lower, upper)` at `level`
- `pvalue_robust`, `z_robust`
- `h`, `b` — main and pilot bandwidths
- `n_left`, `n_right` — effective observations inside the window on each side
- `cutoff`, `p`, `kernel`, `level`, `design` (`:sharp` or `:fuzzy`)
- `first_stage` — the treatment jump for a fuzzy design (`nothing` when sharp)
"""
struct RDDResult{T<:AbstractFloat}
    tau_conventional::T
    tau_bias_corrected::T
    se_conventional::T
    se_robust::T
    ci_conventional::Tuple{T,T}
    ci_robust::Tuple{T,T}
    pvalue_robust::T
    z_robust::T
    h::T
    b::T
    n_left::Int
    n_right::Int
    cutoff::T
    p::Int
    kernel::Symbol
    level::T
    design::Symbol
    first_stage::Union{Nothing,T}
end

# =============================================================================
# Estimation
# =============================================================================

"""
    estimate_rdd(y, running; cutoff=0.0, fuzzy=nothing, kernel=:triangular, p=1,
                 h=nothing, b=nothing, level=0.95) → RDDResult

Sharp or fuzzy regression discontinuity by local polynomial regression, with
Calonico-Cattaneo-Titiunik (2014) robust bias-corrected inference.

Units just above and just below a cutoff in a **running variable** are comparable, so the
jump in the conditional mean of the outcome at that cutoff identifies a local treatment
effect. The estimator fits a weighted polynomial on each side and takes the difference of
the two intercepts.

Passing `fuzzy` (a treatment indicator that only *jumps* in probability at the cutoff rather
than switching deterministically) gives the **fuzzy** design: the estimate is the local Wald
ratio, the outcome jump divided by the treatment jump.

# Arguments
- `y::AbstractVector` — outcome
- `running::AbstractVector` — running (forcing) variable

# Keyword Arguments
- `cutoff::Real = 0.0` — threshold in the running variable
- `fuzzy` — treatment indicator for a fuzzy design; `nothing` for sharp
- `kernel::Symbol = :triangular` — `:triangular`, `:epanechnikov`, or `:uniform`
- `p::Int = 1` — polynomial order (1 = local linear, the standard choice)
- `h`, `b` — main and pilot bandwidths; omit for the MSE-optimal plug-in
- `level::Real = 0.95` — confidence level

# Returns
An [`RDDResult`](@ref). **Use `ci_robust`**, not `ci_conventional` — see the type's docstring.

# Examples
```julia
rd = estimate_rdd(y, x; cutoff=0.0)
report(rd)
rd.ci_robust
```

# References
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust Nonparametric Confidence
  Intervals for Regression-Discontinuity Designs. *Econometrica*, 82(6), 2295–2326.
"""
function estimate_rdd(y::AbstractVector{T}, running::AbstractVector{T};
                      cutoff::Real=zero(T),
                      fuzzy::Union{Nothing,AbstractVector}=nothing,
                      kernel::Symbol=:triangular, p::Int=1,
                      h::Union{Nothing,Real}=nothing,
                      b::Union{Nothing,Real}=nothing,
                      level::Real=0.95) where {T<:AbstractFloat}
    _validate_data(y, "y")
    _validate_data(running, "running")
    n = length(y)
    length(running) == n || throw(ArgumentError("running must have length $n"))
    kernel in (:triangular, :epanechnikov, :uniform) ||
        throw(ArgumentError("kernel must be :triangular, :epanechnikov, or :uniform; got :$kernel"))
    p >= 1 || throw(ArgumentError("p must be >= 1, got $p"))
    zero(T) < T(level) < one(T) || throw(ArgumentError("level must lie in (0,1), got $level"))
    if fuzzy !== nothing
        length(fuzzy) == n || throw(ArgumentError("fuzzy must have length $n"))
    end

    yv = Vector{T}(y)
    xv = Vector{T}(running)
    c = T(cutoff)
    q = p + 1

    any(xv .< c) || throw(ArgumentError("no observations below the cutoff $cutoff"))
    any(xv .>= c) || throw(ArgumentError("no observations at or above the cutoff $cutoff"))

    h_use, b_auto = _rd_bandwidth(yv, xv, c, p, kernel)
    h_use = h === nothing ? h_use : T(h)
    b_use = b === nothing ? max(b_auto, h_use) : T(b)
    h_use > 0 || throw(ArgumentError("bandwidth h must be positive"))

    # Linear-in-y weights for each side, conventional and bias-corrected.
    wl_c, wl_bc, okl = _rd_side_linear(xv, c, h_use, b_use, p, q, kernel, false, n)
    wr_c, wr_bc, okr = _rd_side_linear(xv, c, h_use, b_use, p, q, kernel, true, n)
    (okl && okr) || @warn "estimate_rdd: the pilot window contains too few observations " *
                          "for the bias estimate on at least one side; the bias-corrected " *
                          "and robust results fall back to the conventional ones." maxlog = 1

    w_conv = wr_c .- wl_c
    w_bc = wr_bc .- wl_bc

    s2 = _rd_sigma2(yv, xv, c, h_use, p, kernel)
    tau_conv = dot(w_conv, yv)
    tau_bc = dot(w_bc, yv)
    var_conv = sum(w_conv .^ 2 .* s2)
    var_rob = sum(w_bc .^ 2 .* s2)

    first_stage = nothing
    if fuzzy !== nothing
        dv = Vector{T}(fuzzy)
        fs_conv = dot(w_conv, dv)
        fs_bc = dot(w_bc, dv)
        abs(fs_bc) > T(1e-10) || throw(ArgumentError(
            "fuzzy RDD: the treatment jump at the cutoff is numerically zero " *
            "($fs_bc); the local Wald ratio is not identified."))
        # Delta method on the ratio, treating the first stage as the denominator.
        s2d = _rd_sigma2(dv, xv, c, h_use, p, kernel)
        var_num_c = var_conv
        var_den_c = sum(w_conv .^ 2 .* s2d)
        cov_c = sum(w_conv .^ 2 .* sqrt.(s2 .* s2d))
        ratio_c = tau_conv / fs_conv
        var_conv = (var_num_c - 2 * ratio_c * cov_c + ratio_c^2 * var_den_c) / fs_conv^2

        var_num_r = var_rob
        var_den_r = sum(w_bc .^ 2 .* s2d)
        cov_r = sum(w_bc .^ 2 .* sqrt.(s2 .* s2d))
        ratio_r = tau_bc / fs_bc
        var_rob = (var_num_r - 2 * ratio_r * cov_r + ratio_r^2 * var_den_r) / fs_bc^2

        first_stage = fs_bc
        tau_conv = ratio_c
        tau_bc = ratio_r
    end

    se_conv = sqrt(max(var_conv, zero(T)))
    se_rob = sqrt(max(var_rob, zero(T)))
    z = Distributions.quantile(Distributions.Normal(), (one(T) + T(level)) / 2)
    ci_c = (tau_conv - z * se_conv, tau_conv + z * se_conv)
    ci_r = (tau_bc - z * se_rob, tau_bc + z * se_rob)
    zstat = se_rob > zero(T) ? tau_bc / se_rob : T(NaN)
    pval = se_rob > zero(T) ?
           2 * (one(T) - Distributions.cdf(Distributions.Normal(), abs(zstat))) : T(NaN)

    nl = count(i -> xv[i] < c && abs(xv[i] - c) <= h_use, 1:n)
    nr = count(i -> xv[i] >= c && abs(xv[i] - c) <= h_use, 1:n)

    return RDDResult{T}(tau_conv, tau_bc, se_conv, se_rob, ci_c, ci_r, pval, zstat,
                        h_use, b_use, nl, nr, c, p, kernel, T(level),
                        fuzzy === nothing ? :sharp : :fuzzy, first_stage)
end

function estimate_rdd(y::AbstractVector, running::AbstractVector; kwargs...)
    estimate_rdd(Float64.(y), Float64.(running); kwargs...)
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, r::RDDResult{T}) where {T}
    print(io, "RDDResult{$T}: $(r.design) RD, tau=$(round(r.tau_bias_corrected; digits=4)), ",
              "robust SE=$(round(r.se_robust; digits=4)), h=$(round(r.h; digits=4))")
end

"""
    report(r::RDDResult)

Print the conventional and robust bias-corrected estimates side by side, with the
bandwidths and effective sample sizes.
"""
function report(io::IO, r::RDDResult{T}) where {T}
    rows = ["Design" => string(r.design),
            "Cutoff" => _fmt(r.cutoff; digits=4),
            "Polynomial order" => string(r.p),
            "Kernel" => string(r.kernel),
            "Bandwidth h" => _fmt(r.h; digits=5),
            "Pilot bandwidth b" => _fmt(r.b; digits=5),
            "Eff. obs (left)" => string(r.n_left),
            "Eff. obs (right)" => string(r.n_right)]
    r.first_stage === nothing ||
        push!(rows, "First-stage jump" => _fmt(r.first_stage; digits=5))
    _show_spec_table(io, "Regression Discontinuity (Calonico-Cattaneo-Titiunik)", rows)
    _coef_table(io, "RD treatment effect",
                ["Conventional", "Robust (bias-corrected)"],
                T[r.tau_conventional, r.tau_bias_corrected],
                T[r.se_conventional, r.se_robust];
                dist=:z, level=r.level)
    return nothing
end
report(r::RDDResult) = report(stdout, r)
