# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Winberry (2018) parametric-family representation of the cross-sectional
distribution.

The Young (2010) histogram carries one state per grid node, so a linearized
heterogeneous-agent model has `n_a × n_e` distribution states.  Winberry
instead approximates the asset density *within each income state* by an
exponential family pinned down by a handful of moments,

    g_j(a) ∝ exp( λ_{j,1} z + Σ_{i=2}^{n} λ_{j,i} (z^i − μ_{j,i}) ),
    z = (a − m_{j,1}) / sqrt(m_{j,2}),

so the distribution state collapses to the `n_e × n_moments` moment vector
`m`.  The `λ` that match a given moment vector solve a small strictly convex
problem, and the stationary distribution is the fixed point of the induced
law of motion on `m` rather than of the `N × N` histogram transition.

# References
- Winberry, T. (2018). A method for solving and estimating heterogeneous agent
  macro models. *Quantitative Economics*, 9(3), 1123–1151.
- Young, E. R. (2010). Solving the incomplete markets model with aggregate
  uncertainty using the Krusell–Smith algorithm and non-stochastic simulations.
  *Journal of Economic Dynamics and Control*, 34(1), 36–41.
"""

# =============================================================================
# Quadrature — composite Gauss–Legendre on the reference grid
# =============================================================================

"""
    _gauss_legendre(T, n) → (x, w)

`n`-point Gauss–Legendre nodes and weights on `[-1, 1]`, via the Golub–Welsch
eigendecomposition of the Legendre Jacobi matrix.  Exact for polynomials of
degree `≤ 2n − 1`.
"""
function _gauss_legendre(::Type{T}, n::Int) where {T<:AbstractFloat}
    @assert n >= 1 "Gauss–Legendre needs at least one node"
    n == 1 && return T[0], T[2]
    beta = T[k / sqrt(T(4 * k^2 - 1)) for k in 1:(n - 1)]
    F = eigen(SymTridiagonal(zeros(T, n), beta))
    x = F.values
    w = T(2) .* vec(F.vectors[1, :]) .^ 2
    return x, w
end

"""
    _composite_quadrature(edges, k) → (nodes, weights)

Composite Gauss–Legendre rule: a `k`-point rule on every subinterval
`[edges[s], edges[s+1]]`.  Using the *asset grid nodes* as the edges makes the
rule exact on the piecewise-linear savings policy and inherits the grid's own
curvature near the borrowing constraint, where the density is steepest.
"""
function _composite_quadrature(edges::AbstractVector{T}, k::Int) where {T<:AbstractFloat}
    @assert length(edges) >= 2 "Need at least two edges"
    @assert k >= 1 "Need at least one node per segment"
    x0, w0 = _gauss_legendre(T, k)
    m = length(edges) - 1
    nodes = Vector{T}(undef, m * k)
    weights = Vector{T}(undef, m * k)
    @inbounds for s in 1:m
        lo = edges[s]
        hi = edges[s + 1]
        half = (hi - lo) / T(2)
        mid = (hi + lo) / T(2)
        for q in 1:k
            idx = (s - 1) * k + q
            nodes[idx] = mid + half * x0[q]
            weights[idx] = half * w0[q]
        end
    end
    return nodes, weights
end

"""
    winberry_quadrature(grid; n_quad=4) → (nodes, weights)

Reference-grid quadrature for the parametric family on a one-asset
[`HAGrid`](@ref): a composite `n_quad`-point Gauss–Legendre rule whose segments
are the asset grid intervals.  Returns nodes and weights in *asset units*, so
`sum(weights)` is the width of the asset domain.
"""
function winberry_quadrature(grid::HAGrid{T}; n_quad::Int=4) where {T<:AbstractFloat}
    grid.n_dims == 1 || throw(ArgumentError(
        "winberry_quadrature: the parametric family is implemented for one-asset " *
        "grids only (got n_dims = $(grid.n_dims))."))
    return _composite_quadrature(grid.grids[1], n_quad)
end

# =============================================================================
# _winberry_basis — standardized polynomial basis
# =============================================================================

"""
    _winberry_basis(nodes, center, scale, mu) → B

Design matrix `B[k, i] = z_k^i − μ_i` with `z_k = (nodes[k] − center) / scale`.

The basis is *standardized* (location `center = m_1`, scale `sqrt(m_2)`) and
*centered on its own targets*, so the fitted `λ` are O(1) and the gradient of
the fit objective is exactly the vector of moment residuals.
"""
function _winberry_basis(nodes::AbstractVector{T}, center::T, scale::T,
                          mu::AbstractVector{T}) where {T<:AbstractFloat}
    nq = length(nodes)
    n = length(mu)
    B = Matrix{T}(undef, nq, n)
    @inbounds for k in 1:nq
        z = (nodes[k] - center) / scale
        zp = one(T)
        for i in 1:n
            zp *= z
            B[k, i] = zp - mu[i]
        end
    end
    return B
end

"""
    _standardized_targets(moments) → (center, scale, mu)

Split a centered-moment vector `m = (m_1, m_2, …, m_n)` — mean, variance, then
higher *central* moments — into the location `m_1`, the scale `sqrt(m_2)` and
the standardized targets `μ_i = m_i / m_2^{i/2}` (so `μ_1 = 0`, `μ_2 = 1`).
"""
function _standardized_targets(moments::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(moments)
    n >= 2 || throw(ArgumentError(
        "Winberry parametric family: need at least 2 moments (mean and variance), " *
        "got n_moments = $n."))
    moments[2] > zero(T) || throw(ArgumentError(
        "Winberry parametric family: the second moment must be a positive " *
        "variance (got $(moments[2]))."))
    center = moments[1]
    scale = sqrt(moments[2])
    mu = Vector{T}(undef, n)
    mu[1] = zero(T)
    mu[2] = one(T)
    @inbounds for i in 3:n
        mu[i] = moments[i] / scale^i
    end
    return center, scale, mu
end

# =============================================================================
# fit_parametric_density — convex moment solve for λ
# =============================================================================

"""
    fit_parametric_density(moments; bounds=nothing, nodes=nothing, weights=nothing,
                           n_segments=64, n_quad=5, max_iter=100, tol=1e-10,
                           lambda_init=nothing) → ParametricDensity{T}

Fit the Winberry (2018) exponential family whose moments equal `moments`.

`moments` is the vector of *centered* moments `(m_1, m_2, …, m_n)`: the mean,
the variance, and then the higher central moments.  At least two are required.
The fitted density on the reference interval is

    g(a) = exp( Σ_i λ_i (z^i − μ_i) − log_norm ),   z = (a − m_1) / sqrt(m_2),

which integrates to one and reproduces `moments` exactly at the solution.

# Algorithm
`λ` minimizes the strictly convex log-normalizer
`F(λ) = log ∫ exp(Σ_i λ_i (z^i − μ_i)) da`, whose gradient is
`∇_i F = E_g[z^i] − μ_i` — the moment residual itself — and whose Hessian is
`Cov_g(z^i, z^j)`.  Both are available in closed form for an exponential family,
so the Newton step uses the *exact* derivatives rather than an AD
approximation; `test_ha_dsge.jl` cross-checks them against
`ForwardDiff.gradient`/`ForwardDiff.hessian`.  Backtracking on `F` makes the
iteration globally convergent from the Gaussian warm start `λ = (0, −1/2, 0…)`.
The normalizer is evaluated in log space with the exponent maximum subtracted,
so a family that is sharply peaked at the borrowing constraint never overflows.

# Arguments
- `moments::AbstractVector` — target centered moments, length `n ≥ 2`

# Keyword Arguments
- `bounds::Union{Nothing,Tuple{Real,Real}}` — reference interval, used to build a
  default quadrature when `nodes`/`weights` are not supplied
- `nodes`, `weights` — explicit quadrature in asset units (e.g. from
  [`winberry_quadrature`](@ref)); both must be given together
- `n_segments::Int` — subintervals of the default rule built from `bounds` (default 64)
- `n_quad::Int` — Gauss–Legendre nodes per subinterval (default 5)
- `max_iter::Int` — maximum Newton iterations (default 100)
- `tol::Real` — convergence tolerance on the largest standardized moment residual,
  **relative to the target scale** `max(1, maximum(abs, μ))` (default 1e-10). The
  test is relative because an absolute one demands more precision the larger the
  moments are, which made `converged` platform-dependent on the ill-conditioned
  four-moment basis (#514). A fit that reaches a stationary point the line search
  cannot improve on is also accepted, provided its residual is below
  `sqrt(eps(T))` relative — an infeasible fit stalls at an `O(1)` residual and is
  still reported as not converged.
- `lambda_init` — warm start for `λ` (default: the Gaussian `(0, −1/2, 0, …)`)

# Returns
A [`ParametricDensity`](@ref) carrying `lambda`, the target `moments`, the
location/scale, `log_norm`, and a `converged` flag with the attained `residual`
(#356/T257 — the `λ` solve never reports success silently).

# References
- Winberry, T. (2018). A method for solving and estimating heterogeneous agent
  macro models. *Quantitative Economics*, 9(3), 1123–1151.
"""
function fit_parametric_density(moments::AbstractVector{<:Real};
                                 bounds::Union{Nothing,Tuple{Real,Real}}=nothing,
                                 nodes::Union{Nothing,AbstractVector}=nothing,
                                 weights::Union{Nothing,AbstractVector}=nothing,
                                 n_segments::Int=64, n_quad::Int=5,
                                 max_iter::Int=100, tol::Real=1e-10,
                                 lambda_init::Union{Nothing,AbstractVector}=nothing)
    T = float(eltype(moments))
    m = collect(T, moments)
    if nodes === nothing || weights === nothing
        bounds === nothing && throw(ArgumentError(
            "fit_parametric_density: supply either `bounds` or both `nodes` and " *
            "`weights` for the reference quadrature."))
        edges = collect(range(T(bounds[1]), T(bounds[2]); length=n_segments + 1))
        nd, wt = _composite_quadrature(edges, n_quad)
    else
        nd = collect(T, nodes)
        wt = collect(T, weights)
        length(nd) == length(wt) || throw(ArgumentError(
            "fit_parametric_density: `nodes` and `weights` must have equal length."))
    end
    return _fit_parametric_density(m, nd, wt; max_iter=max_iter, tol=T(tol),
                                   lambda_init=lambda_init)
end

# Core Newton solve. Kept separate from the keyword front end so the hot paths
# (the stationary fixed point and the linearization) can pass a pre-built
# quadrature and a warm start without re-allocating either.
function _fit_parametric_density(moments::Vector{T}, nodes::Vector{T},
                                  weights::Vector{T};
                                  max_iter::Int=100, tol::T=T(1e-10),
                                  lambda_init::Union{Nothing,AbstractVector}=nothing,
                                  n_recondition::Int=4) where {T<:AbstractFloat}
    center, scale, mu = _standardized_targets(moments)
    n = length(mu)
    B0 = _winberry_basis(nodes, center, scale, mu)

    lambda = if lambda_init === nothing
        lam = zeros(T, n)
        lam[2] = -T(0.5)          # Gaussian warm start: exp(−z²/2)
        lam
    else
        collect(T, lambda_init)
    end
    length(lambda) == n || throw(ArgumentError(
        "fit_parametric_density: lambda_init has length $(length(lambda)), expected $n."))

    # `resid` is the largest moment mismatch in standardized units, so testing it
    # against an ABSOLUTE `tol` implicitly demands more relative precision the larger
    # the targets are — and the targets grow exactly where the problem is hardest
    # (max|μ| = 1 for two or three moments, but 9 for the four-moment exponential
    # basis whose Hessian reaches cond ~1e8). That made `converged` depend on the
    # platform's arithmetic rather than on the fit: 4.1e-15 on 1.12/arm64 and
    # 6.5e-15 on 1.10/arm64 against a 1e-12 request, but above it on Windows (#514).
    # `tol` is therefore relative to the target scale.
    resid_scale = max(one(T), maximum(abs, mu))
    tol_eff = T(tol) * resid_scale
    # Floor for the stagnation test below: the best a gradient computed in floating
    # point can be expected to reach.
    floor_tol = sqrt(eps(T)) * resid_scale

    u = B0 * lambda
    Fval, p = _log_normalizer(u, weights)
    resid = maximum(abs, B0' * p)
    converged = resid <= tol_eff
    iters = 0

    for _ in 1:max(n_recondition, 1)
        converged && break
        # Whiten the monomial basis under the CURRENT density. The raw Hessian
        # Cov_g(z^i, z^j) is catastrophically conditioned once four moments are
        # carried — cond ≈ 1e8 on a calibrated Krusell–Smith grid, whose top asset
        # node sits 33 standard deviations above the mean, so z^4 spans 10^6 — and a
        # Newton step solved in those coordinates loses eight digits. `qr(√p ⊙ B).R`
        # gives `B̃ = B R⁻¹` orthonormal under `g`, so the Hessian starts at the
        # identity; going through the QR (not a Cholesky of the Gram matrix) never
        # squares the condition number. `λ = R⁻¹ λ̃` is an exact linear
        # reparameterization, so the fitted density is unchanged (#356/T257).
        Rf = _whitening_factor(B0, p, n)
        B, Rf = try
            Bt = Matrix{T}(B0 / Rf)
            all(isfinite, Bt) ? (Bt, Rf) : (copy(B0), Matrix{T}(I, n, n))
        catch
            (copy(B0), Matrix{T}(I, n, n))
        end
        lam = Rf * lambda
        u = B * lam
        Fval, p = _log_normalizer(u, weights)
        grad = B' * p
        lam_new = similar(lam)

        for _ in 1:max_iter
            iters += 1
            # Hessian = Cov_g(b̃): PSD by construction, so a jittered Cholesky solve
            # is both cheaper and better conditioned than a general inverse.
            H = B' * (B .* p)
            H .-= grad * grad'
            step = _psd_solve(H, grad)
            # The Newton direction must descend; fall back to steepest descent if a
            # numerically semidefinite Hessian produces a non-descent step.
            slope = -dot(grad, step)
            if !(slope < zero(T)) || !all(isfinite, step)
                step = copy(grad)
                slope = -dot(grad, step)
            end
            t = one(T)
            accepted = false
            local F_new, p_new
            for _ in 1:60
                @. lam_new = lam - t * step
                u = B * lam_new
                F_new, p_new = _log_normalizer(u, weights)
                if isfinite(F_new) && F_new <= Fval + T(1e-4) * t * slope
                    accepted = true
                    break
                end
                t /= T(2)
            end
            # The line search failing means a stationary point of a strictly convex
            # objective: no further descent exists at this precision. Judged against
            # the arithmetic floor after the loop, together with the iteration-limit
            # exit, which lands in exactly the same place.
            accepted || break
            lam .= lam_new
            Fval = F_new
            p = p_new
            grad = B' * p
            # Report the residual in the ORIGINAL standardized-moment units
            # (∇ = Rᵀ ∇̃), so `tol` always means "largest moment mismatch".
            resid = maximum(abs, transpose(Rf) * grad)
            if resid <= tol_eff
                converged = true
                break
            end
        end
        lambda = Rf \ lam
    end

    # Arithmetic floor. Below `sqrt(eps)` relative to the target scale the residual
    # is floating-point noise in the gradient, not a mismatch the solve could act
    # on, so `tol` is not meaningful there and demanding it tests the hardware. A
    # fit that reaches this floor is converged however the loop ended — line-search
    # stall or iteration limit. An infeasible or under-iterated fit stalls at an
    # O(1) residual and still reports `converged = false` (#514).
    if !converged && resid <= floor_tol
        converged = true
    end

    # log_norm is in ASSET units: g(a) = exp(Σ λ_i (z^i − μ_i) − log_norm) with
    # ∫ g da = 1 over the reference interval.
    u = B0 * lambda
    umax = maximum(u)
    log_norm = umax + log(sum(weights .* exp.(u .- umax)))
    return ParametricDensity{T}(lambda, copy(moments), center, scale, log_norm,
                                converged, iters, resid)
end

"""
    _whitening_factor(B, p, n) → R

Upper-triangular `R` such that `B R⁻¹` is orthonormal under the node
probabilities `p`, from the QR factorization of the weighted design matrix
`√p ⊙ B`.  Falls back to the identity when the factor is singular or
non-finite (a degenerate density), which simply leaves the basis unwhitened.
"""
function _whitening_factor(B::AbstractMatrix{T}, p::AbstractVector{T},
                            n::Int) where {T<:AbstractFloat}
    try
        R = Matrix{T}(qr(sqrt.(p) .* B).R)
        d = abs.(diag(R))
        (all(isfinite, R) && minimum(d) > eps(T) * max(maximum(d), one(T))) ||
            return Matrix{T}(I, n, n)
        return R
    catch
        return Matrix{T}(I, n, n)
    end
end

"""
    _log_normalizer(u, weights) → (F, p)

Stable log-normalizer `F = log Σ_k w_k e^{u_k}` and the induced node
probabilities `p_k = w_k e^{u_k} / Σ w e^u` (which sum to one).  The exponent
maximum is subtracted before exponentiating, so an exponential family peaked at
the borrowing constraint never overflows.
"""
function _log_normalizer(u::AbstractVector{T}, weights::AbstractVector{T}) where {T<:AbstractFloat}
    umax = maximum(u)
    p = weights .* exp.(u .- umax)
    s = sum(p)
    F = umax + log(s)
    s > zero(T) && (p ./= s)
    return F, p
end

"""
    _psd_solve(H, g) → H \\ g

Solve a positive-semidefinite system by Cholesky with automatic jitter,
falling back to `robust_inv` when even the jittered factorization fails.  The
Winberry Hessian `Cov_g(z^i, z^j)` is badly conditioned whenever the reference
interval is wide relative to the dispersion of the density.
"""
function _psd_solve(H::AbstractMatrix{T}, g::AbstractVector{T}) where {T<:AbstractFloat}
    try
        L, _ = safe_cholesky_jitter(H; silent=true)
        return L' \ (L \ g)
    catch
        return Matrix{T}(robust_inv(Symmetric(H); silent=true)) * g
    end
end

# =============================================================================
# Density evaluation and the inverse (λ → moments) map
# =============================================================================

"""
    parametric_density(pd::ParametricDensity, a) → T

Evaluate the fitted density at asset level `a`.  Values outside the reference
interval used for the fit are *not* clipped — the caller is responsible for
staying inside `bounds`, exactly as the underlying quadrature is.
"""
function parametric_density(pd::ParametricDensity{T}, a::Real) where {T<:AbstractFloat}
    z = (T(a) - pd.center) / pd.scale
    n = length(pd.lambda)
    mu = _implied_mu(pd)
    u = zero(T)
    zp = one(T)
    @inbounds for i in 1:n
        zp *= z
        u += pd.lambda[i] * (zp - mu[i])
    end
    return exp(u - pd.log_norm)
end

# Standardized targets implied by the stored (moments, scale) pair.
function _implied_mu(pd::ParametricDensity{T}) where {T<:AbstractFloat}
    n = length(pd.lambda)
    mu = Vector{T}(undef, n)
    mu[1] = zero(T)
    mu[2] = one(T)
    @inbounds for i in 3:n
        mu[i] = pd.moments[i] / pd.scale^i
    end
    return mu
end

"""
    parametric_moments(pd::ParametricDensity, nodes, weights) → Vector{T}

Centered moments *implied by* the fitted `λ` — the inverse of the
[`fit_parametric_density`](@ref) map.  Computed in two passes (mean first, then
central moments about it) so no raw power sums are formed and no cancellation
occurs at the scale of `E[a^4]`.

Returns `(m_1, m_2, …, m_n)` on the same convention as the fit targets; at a
converged fit it reproduces `pd.moments` to the fit tolerance.
"""
function parametric_moments(pd::ParametricDensity{T}, nodes::AbstractVector{T},
                             weights::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(pd.lambda)
    mu = _implied_mu(pd)
    nq = length(nodes)
    u = Vector{T}(undef, nq)
    @inbounds for k in 1:nq
        z = (nodes[k] - pd.center) / pd.scale
        acc = zero(T)
        zp = one(T)
        for i in 1:n
            zp *= z
            acc += pd.lambda[i] * (zp - mu[i])
        end
        u[k] = acc
    end
    _, p = _log_normalizer(u, weights)
    return _central_moments(nodes, p, n)
end

"""
    _central_moments(x, p, n) → Vector{T}

Mean and central moments 2…`n` of the discrete measure `p` on support `x`
(`p` sums to one).  Two passes: the mean, then the powers of the deviations.
"""
function _central_moments(x::AbstractVector{T}, p::AbstractVector{T},
                           n::Int) where {T<:AbstractFloat}
    out = zeros(T, n)
    m1 = zero(T)
    @inbounds for k in eachindex(x)
        m1 += p[k] * x[k]
    end
    out[1] = m1
    @inbounds for k in eachindex(x)
        dev = x[k] - m1
        dp = dev
        for i in 2:n
            dp *= dev
            out[i] += p[k] * dp
        end
    end
    return out
end

# =============================================================================
# Histogram ↔ moments
# =============================================================================

"""
    winberry_moments(distribution, grid; n_moments=3) → (moments, mass)

Per-income-state centered moments of a Young (2010) histogram.

`distribution` is the `n_a × n_e` stationary histogram (or its flattened
`n_a·n_e` vector).  Returns an `n_e × n_moments` matrix whose row `j` holds
`(m_1, m_2, …, m_n)` of the asset density *conditional* on income state `j`,
together with the vector of income-state masses.
"""
function winberry_moments(distribution::AbstractArray{T}, grid::HAGrid{T};
                           n_moments::Int=3) where {T<:AbstractFloat}
    n_moments >= 2 || throw(ArgumentError(
        "winberry_moments: need at least 2 moments (mean and variance), got $n_moments."))
    a_grid = grid.grids[1]
    n_a = length(a_grid)
    d = reshape(collect(T, vec(distribution)), n_a, :)
    n_e = size(d, 2)
    M = zeros(T, n_e, n_moments)
    mass = zeros(T, n_e)
    @inbounds for j in 1:n_e
        mj = sum(view(d, :, j))
        mass[j] = mj
        if mj <= zero(T)
            M[j, 2] = one(T)          # degenerate state: unit-variance placeholder
            continue
        end
        p = view(d, :, j) ./ mj
        M[j, :] .= _central_moments(a_grid, p, n_moments)
        # A histogram concentrated on a single node has zero variance, which the
        # exponential family cannot represent. Floor it at the finest grid step so
        # the fit stays well posed instead of dividing by zero in the standardization.
        if M[j, 2] <= zero(T)
            M[j, 2] = (a_grid[2] - a_grid[1])^2
        end
    end
    return M, mass
end

"""
    winberry_histogram(family::WinberryFamily, grid::HAGrid) → Vector{T}

Histogram representation of a fitted parametric family: the mass the family
assigns to each `(asset, income)` node of `grid`, on the same
`(j-1)·n_a + i` index convention as the Young distribution.

Cell boundaries are the midpoints between adjacent asset nodes (clipped to the
grid bounds), and each cell mass is an exact composite Gauss–Legendre integral
of the density over its cell, so the result sums to one by construction.  This
is what makes a parametric solution plottable and comparable, node by node,
against a Young histogram.
"""
function winberry_histogram(family::WinberryFamily{T}, grid::HAGrid{T};
                             n_quad::Int=4) where {T<:AbstractFloat}
    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = length(family.densities)
    edges = _cell_edges(a_grid, family.bounds)
    nodes, weights = _composite_quadrature(edges, n_quad)
    out = zeros(T, n_a * n_e)
    @inbounds for j in 1:n_e
        pd = family.densities[j]
        cell = zeros(T, n_a)
        for s in 1:n_a
            acc = zero(T)
            for q in 1:n_quad
                idx = (s - 1) * n_quad + q
                acc += weights[idx] * parametric_density(pd, nodes[idx])
            end
            cell[s] = max(acc, zero(T))
        end
        tot = sum(cell)
        tot > zero(T) && (cell .*= family.mass[j] / tot)
        out[((j - 1) * n_a + 1):(j * n_a)] .= cell
    end
    s = sum(out)
    s > zero(T) && (out ./= s)
    return out
end

# Midpoint cell edges for the histogram representation: n_a cells, edges[1] and
# edges[end] pinned to the reference bounds so no mass escapes the domain.
function _cell_edges(a_grid::AbstractVector{T}, bounds::Tuple{T,T}) where {T<:AbstractFloat}
    n_a = length(a_grid)
    edges = Vector{T}(undef, n_a + 1)
    edges[1] = bounds[1]
    edges[end] = bounds[2]
    @inbounds for i in 1:(n_a - 1)
        edges[i + 1] = (a_grid[i] + a_grid[i + 1]) / T(2)
    end
    return edges
end

# =============================================================================
# fit_winberry — fit the family to a histogram or a moment matrix
# =============================================================================

"""
    fit_winberry(distribution, grid; n_moments=3, n_quad=4, kwargs...) → WinberryFamily{T}
    fit_winberry(ss::HASteadyState; n_moments=3, kwargs...) → WinberryFamily{T}

Fit the Winberry (2018) parametric family to a Young histogram — one
exponential-family density per income state, matching that state's first
`n_moments` centered moments.

# Keyword Arguments
- `n_moments::Int` — moments carried per income state (default 3: mean,
  variance, third central moment)
- `n_quad::Int` — Gauss–Legendre nodes per asset-grid interval (default 4)
- `max_iter`, `tol` — forwarded to [`fit_parametric_density`](@ref)

The returned family's `converged` field is `true` only if *every* income
state's `λ` solve converged.
"""
function fit_winberry(distribution::AbstractArray{T}, grid::HAGrid{T};
                       n_moments::Int=3, n_quad::Int=4,
                       max_iter::Int=100, tol::Real=1e-10) where {T<:AbstractFloat}
    M, mass = winberry_moments(distribution, grid; n_moments=n_moments)
    nodes, weights = winberry_quadrature(grid; n_quad=n_quad)
    return _build_family(M, mass, nodes, weights, grid;
                         max_iter=max_iter, tol=T(tol))
end

fit_winberry(ss::HASteadyState{T}; kwargs...) where {T<:AbstractFloat} =
    fit_winberry(ss.distribution, ss.grid; kwargs...)

# Assemble a WinberryFamily from a moment matrix (n_e × n_moments), fitting each
# income state's λ. `lambda_warm` re-uses the previous solve as a starting point,
# which is what makes the stationary fixed point and the finite-difference
# Jacobian affordable.
function _build_family(M::AbstractMatrix{T}, mass::AbstractVector{T},
                        nodes::Vector{T}, weights::Vector{T}, grid::HAGrid{T};
                        max_iter::Int=100, tol::T=T(1e-10),
                        lambda_warm::Union{Nothing,Vector{Vector{T}}}=nothing
                        ) where {T<:AbstractFloat}
    n_e = size(M, 1)
    dens = Vector{ParametricDensity{T}}(undef, n_e)
    all_ok = true
    for j in 1:n_e
        warm = lambda_warm === nothing ? nothing : lambda_warm[j]
        pd = _fit_parametric_density(collect(T, view(M, j, :)), nodes, weights;
                                     max_iter=max_iter, tol=tol, lambda_init=warm)
        dens[j] = pd
        all_ok &= pd.converged
    end
    bounds = (grid.grids[1][1], grid.grids[1][end])
    return WinberryFamily{T}(dens, collect(T, mass), size(M, 2), nodes, weights,
                             bounds, all_ok)
end

# Node probabilities of a fitted density on the family's own quadrature.
function _density_weights(pd::ParametricDensity{T}, nodes::Vector{T},
                           weights::Vector{T}) where {T<:AbstractFloat}
    n = length(pd.lambda)
    mu = _implied_mu(pd)
    u = Vector{T}(undef, length(nodes))
    @inbounds for k in eachindex(nodes)
        z = (nodes[k] - pd.center) / pd.scale
        acc = zero(T)
        zp = one(T)
        for i in 1:n
            zp *= z
            acc += pd.lambda[i] * (zp - mu[i])
        end
        u[k] = acc
    end
    _, p = _log_normalizer(u, weights)
    return p
end

# =============================================================================
# _winberry_forward — one step of the moment law of motion
# =============================================================================

"""
    _winberry_forward(M, mass, a_pol, grid, income, nodes, weights; lambda_warm, …)
        → (M_next, lambdas, converged)

One period of the Winberry law of motion on the moment state.

Given moments `M` (`n_e × n_moments`) and income-state masses, fit the family,
push every quadrature node through the savings policy `a'(a, e_j)`, mix across
the income transition, and read off the next period's centered moments:

    m'_{j',1} = Σ_j Π_{jj'} mass_j E_j[a'] / mass'_{j'}
    m'_{j',i} = Σ_j Π_{jj'} mass_j E_j[(a' − m'_{j',1})^i] / mass'_{j'},  i ≥ 2

The mean is computed first and the central moments in a second pass, so no raw
power sums are formed.  The policy is evaluated by the same flat-extrapolating
linear interpolation the Young transition uses and clamped to the grid, keeping
the two representations comparable.
"""
function _winberry_forward(M::AbstractMatrix{T}, mass::AbstractVector{T},
                            a_pol::AbstractMatrix{T}, grid::HAGrid{T},
                            income::IncomeProcess{T}, nodes::Vector{T},
                            weights::Vector{T};
                            lambda_warm::Union{Nothing,Vector{Vector{T}}}=nothing,
                            max_iter::Int=100, tol::T=T(1e-11)) where {T<:AbstractFloat}
    a_grid = grid.grids[1]
    lo, hi = a_grid[1], a_grid[end]
    n_e, n_m = size(M)
    nq = length(nodes)
    Pi = income.transition

    P = Matrix{T}(undef, nq, n_e)      # node probabilities per income state
    A = Matrix{T}(undef, nq, n_e)      # next-period assets per node/state
    lambdas = Vector{Vector{T}}(undef, n_e)
    ok = true

    for j in 1:n_e
        warm = lambda_warm === nothing ? nothing : lambda_warm[j]
        pd = _fit_parametric_density(collect(T, view(M, j, :)), nodes, weights;
                                     max_iter=max_iter, tol=tol, lambda_init=warm)
        ok &= pd.converged
        lambdas[j] = pd.lambda
        P[:, j] .= _density_weights(pd, nodes, weights)
        pol = view(a_pol, :, j)
        @inbounds for k in 1:nq
            A[k, j] = clamp(_linear_interp(a_grid, pol, nodes[k]), lo, hi)
        end
    end

    mass_next = transpose(Pi) * mass
    M_next = zeros(T, n_e, n_m)

    # Pass 1 — the mean of the mixed next-period density in each income state.
    @inbounds for jp in 1:n_e
        mass_next[jp] <= zero(T) && continue
        acc = zero(T)
        for j in 1:n_e
            wj = Pi[j, jp] * mass[j]
            wj <= zero(T) && continue
            s = zero(T)
            for k in 1:nq
                s += P[k, j] * A[k, j]
            end
            acc += wj * s
        end
        M_next[jp, 1] = acc / mass_next[jp]
    end

    # Pass 2 — central moments about that mean.
    @inbounds for jp in 1:n_e
        mass_next[jp] <= zero(T) && (M_next[jp, 2] = one(T); continue)
        m1 = M_next[jp, 1]
        for j in 1:n_e
            wj = Pi[j, jp] * mass[j]
            wj <= zero(T) && continue
            for k in 1:nq
                dev = A[k, j] - m1
                dp = dev
                pk = wj * P[k, j]
                for i in 2:n_m
                    dp *= dev
                    M_next[jp, i] += pk * dp
                end
            end
        end
        for i in 2:n_m
            M_next[jp, i] /= mass_next[jp]
        end
        M_next[jp, 2] <= zero(T) && (M_next[jp, 2] = eps(T))
    end

    return M_next, lambdas, ok
end

# =============================================================================
# _winberry_stationary — fixed point of the moment law of motion
# =============================================================================

"""
    _winberry_stationary(a_pol, grid, income; n_moments=3, n_quad=4, M_init=nothing,
                         n_picard=30, newton_rounds=12, tol=1e-10)
        → (; moments, mass, lambdas, converged, iterations, jacobian)

Stationary distribution in *moment space*: the fixed point of
[`_winberry_forward`](@ref), `M = F(M)`, at a fixed savings policy.  This is
the parametric analogue of `_stationary_dist_young`, and is computed
independently of the histogram — so comparing the two is a genuine accuracy
test of the reduction, not a tautology.

Solved in two phases: `n_picard` plain sweeps to reach the basin and warm the
`λ` solves, then Newton on the residual `F(M) − M` with a finite-difference
Jacobian (see the inline note — the Picard rate is the income persistence, so
iteration alone is not viable).

Income-state masses are the ergodic distribution of the income chain, which is
invariant because the idiosyncratic transition matrix does not move with
aggregates.  Convergence is measured on *standardized* moments
`(m_1, sd, skew, …)`, which share a scale; raw central moments do not.  The
default `tol = 1e-9` sits above the map's own noise floor: `λ` is recovered to
about `1e-12` in standardized moment units and the pushforward amplifies that
to `~1e-10`, so a tighter target would report failure on a converged solve.
"""
function _winberry_stationary(a_pol::AbstractMatrix{T}, grid::HAGrid{T},
                               income::IncomeProcess{T};
                               n_moments::Int=3, n_quad::Int=4,
                               M_init::Union{Nothing,AbstractMatrix{T}}=nothing,
                               mass::Union{Nothing,AbstractVector{T}}=nothing,
                               n_picard::Int=30, newton_rounds::Int=40,
                               newton_max_step::Real=1.0, tol::Real=1e-9,
                               fit_tol::Real=1e-11,
                               J_warm::Union{Nothing,AbstractMatrix{T}}=nothing
                               ) where {T<:AbstractFloat}
    nodes, weights = winberry_quadrature(grid; n_quad=n_quad)
    n_e = length(income.states)
    mass_v = mass === nothing ? _ergodic_income_mass(income) : collect(T, mass)

    # Starting guess. Absent an explicit `M_init`, take the moments of the Young
    # histogram for this policy — one sparse solve. This matters: the parametric map
    # has spurious fixed points well outside the ergodic set, and a start chosen
    # without reference to the model finds them. A diffuse family covering the
    # reference interval converges at two moments but strands at three; seeding each
    # income state at its own deterministic policy fixed point collapses the
    # conditional densities (stationary standard deviations of 0.03 against a true
    # 29); continuation from a lower moment order fixes four moments but breaks
    # three. The histogram guess works at every order tested (#356/T257).
    # It is only a guess: the returned point is the fixed point of the MOMENT map,
    # verified stationary to `tol`, and is a different object from the histogram it
    # started from.
    M = if M_init === nothing
        Lambda0 = _build_transition_matrix(collect(T, a_pol), grid, income)
        d0, _ = _stationary_dist_young(Lambda0)
        first(winberry_moments(d0, grid; n_moments=n_moments))
    else
        collect(T, M_init)
    end

    tol_T = T(tol)
    fwd(Mx, warm) = _winberry_forward(Mx, mass_v, a_pol, grid, income, nodes,
                                       weights; lambda_warm=warm, tol=T(fit_tol))

    # ── Phase 1: Picard, to reach the basin and warm every λ solve ───────────
    lambdas = nothing
    converged = false
    iters = 0
    fits_ok = true
    for it in 1:n_picard
        iters = it
        M_new, lam, ok = fwd(M, lambdas)
        fits_ok = ok
        err = _moment_distance(M, M_new)
        lambdas = lam
        M = M_new
        if err < tol_T
            return (moments=M, mass=mass_v, lambdas=lambdas, converged=fits_ok,
                    iterations=iters, jacobian=J_warm)
        end
    end

    # ── Phase 2: Newton on the fixed-point residual ──────────────────────────
    # Plain iteration on the moment map inherits the persistence of the income
    # process: on the calibrated Krusell–Smith example (ρ_e = 0.966) successive
    # Picard errors shrink by a factor of ~0.97, so ten digits would take ~700
    # sweeps and the λ warm-start chain has that many chances to drift. The moment
    # state is small (n_e · n_moments ≤ 40), so one finite-difference Jacobian of
    # `x ↦ state(F(M(x)))` costs less than 100 Picard sweeps and converges
    # quadratically instead (#356/T257).
    n_dist = n_e * n_moments
    Mp = similar(M)
    radius = T(newton_max_step)
    # Chord/Newton hybrid. Re-differencing the map costs 2·n_dist forward
    # evaluations, which dominates everything else, and the Jacobian barely moves
    # between nearby iterates — or between successive trial rates of an outer
    # market-clearing bisection. So reuse `J` until a step it proposes fails, and
    # only then refresh (#356/T257).
    J = J_warm === nothing ? nothing : copy(J_warm)
    J_is_stale = J !== nothing
    stalls = 0
    for _ in 1:newton_rounds
        s = _winberry_scales(M)
        M_img, lam, ok = fwd(M, lambdas)
        lambdas = lam
        fits_ok = ok
        r = _winberry_to_state(M_img .- M, s)
        if maximum(abs, r) < tol_T
            converged = true
            break
        end

        if J === nothing
            J = zeros(T, n_dist, n_dist)
            ok_jac = true
            for i in 1:n_moments, j in 1:n_e
                step = T(1e-4) * s[j]^i
                copyto!(Mp, M); Mp[j, i] += step
                M_up, _, _ = fwd(Mp, lambdas)
                copyto!(Mp, M); Mp[j, i] -= step
                M_dn, _, _ = fwd(Mp, lambdas)
                col = _win_index(j, i, n_e)
                J[:, col] .= _winberry_to_state((M_up .- M_dn) .* (s[j]^i / (T(2) * step)), s)
                ok_jac &= all(isfinite, view(J, :, col))
            end
            J_is_stale = false
            ok_jac || break
        end

        A = J - Matrix{T}(I, n_dist, n_dist)
        dx = try
            A \ (-r)
        catch
            -(Matrix{T}(robust_inv(A; silent=true)) * r)
        end
        all(isfinite, dx) || break
        # Adaptive trust region. The moment map has a near-unit eigenvalue
        # (ρ(J) ≈ 0.992 on the calibrated Krusell–Smith example — it inherits the
        # income persistence), so I − J is nearly singular along the slow mode and an
        # unrestrained Newton step can jump clean out of the ergodic region into a
        # *different*, economically meaningless fixed point of the parametric map
        # (measured: mean assets 371 against a true 42). The radius therefore starts
        # at `newton_max_step` in standardized units and doubles after every accepted
        # step, so the guard binds only while the iterate is still far away
        # (#356/T257).
        big = maximum(abs, dx)
        big > radius && (dx .*= radius / big)

        # Backtrack on the EUCLIDEAN residual norm with an Armijo-style sufficient
        # decrease. The ∞-norm is the wrong merit here: it is attained by a single
        # moment coordinate, and a Newton step that improves the state overall while
        # temporarily worsening that one coordinate gets rejected, after which the
        # shrinking trust region strands the solve short of its tolerance (measured:
        # stalls at a residual of 4e-4 on the two-moment Krusell–Smith fixed point).
        # A full step can also leave the feasible moment cone (a negative variance),
        # which the fit would reject outright — hence the explicit guard.
        base = norm(r)
        t = one(T)
        accepted = false
        for _ in 1:40
            copyto!(Mp, M)
            for i in 1:n_moments, j in 1:n_e
                Mp[j, i] += t * dx[_win_index(j, i, n_e)] * s[j]^i
            end
            if all(k -> Mp[k, 2] > zero(T), 1:n_e)
                M_try, lam_try, _ = fwd(Mp, lambdas)
                r_try = _winberry_to_state(M_try .- Mp, _winberry_scales(Mp))
                if all(isfinite, r_try) && norm(r_try) <= (one(T) - T(1e-4) * t) * base
                    copyto!(M, Mp)
                    lambdas = lam_try
                    accepted = true
                    break
                end
            end
            t /= T(2)
        end
        iters += 1
        if accepted
            radius = min(radius * T(2), T(1e4))
        elseif J_is_stale
            J = nothing              # the reused Jacobian is no longer good enough
            J_is_stale = false
        else
            radius /= T(4)
            if radius < T(1e-8)
                # Neither a fresh Jacobian nor a short step helps: fall back to plain
                # Picard sweeps, which are globally convergent (just slow), then
                # re-linearize from wherever they land. Giving up here instead would
                # return a point that is not stationary at all.
                stalls += 1
                stalls > 6 && break
                for _ in 1:100
                    M_new, lam, ok = fwd(M, lambdas)
                    lambdas = lam; fits_ok = ok; M = M_new
                end
                J = nothing
                radius = T(newton_max_step)
            end
        end
    end

    if !converged
        M_img, lam, ok = fwd(M, lambdas)
        lambdas = lam
        fits_ok = ok
        converged = maximum(abs, _winberry_to_state(M_img .- M,
                                                    _winberry_scales(M))) < tol_T
    end
    return (moments=M, mass=mass_v, lambdas=lambdas,
            converged=(converged && fits_ok), iterations=iters, jacobian=J)
end

"""
    _moment_distance(M1, M2) → T

Distance between two moment matrices in *standardized* units — mean, standard
deviation, then the standardized higher moments `m_i / m_2^{i/2}` — each scaled
by `max(1, |value|)`.  Raw central moments span many orders of magnitude
(`m_1 ≈ 40`, `m_4 ≈ 10^7` on a calibrated Krusell–Smith grid), so a convergence
test on them is dominated by the highest moment alone.
"""
function _moment_distance(M1::AbstractMatrix{T}, M2::AbstractMatrix{T}) where {T<:AbstractFloat}
    n_e, n_m = size(M1)
    worst = zero(T)
    @inbounds for j in 1:n_e
        s1 = sqrt(max(M1[j, 2], zero(T)))
        s2 = sqrt(max(M2[j, 2], zero(T)))
        worst = max(worst, abs(M1[j, 1] - M2[j, 1]) / max(one(T), abs(M1[j, 1])))
        worst = max(worst, abs(s1 - s2) / max(one(T), s1))
        for i in 3:n_m
            v1 = s1 > zero(T) ? M1[j, i] / s1^i : zero(T)
            v2 = s2 > zero(T) ? M2[j, i] / s2^i : zero(T)
            worst = max(worst, abs(v1 - v2) / max(one(T), abs(v1)))
        end
    end
    return worst
end

"""
    _ergodic_income_mass(income) → Vector{T}

Ergodic distribution of the (row-stochastic) idiosyncratic income chain, i.e.
the left eigenvector of `income.transition` for eigenvalue one, normalized to
sum to one.
"""
function _ergodic_income_mass(income::IncomeProcess{T}) where {T<:AbstractFloat}
    Pi = income.transition
    n = size(Pi, 1)
    A = Matrix{T}(transpose(Pi)) - Matrix{T}(I, n, n)
    A[n, :] .= one(T)
    b = zeros(T, n)
    b[n] = one(T)
    local m
    try
        m = A \ b
        all(isfinite, m) || error("non-finite")
    catch
        m = fill(one(T) / n, n)
        for _ in 1:10_000
            m = vec(transpose(Pi) * m)
        end
    end
    @inbounds for i in eachindex(m)
        m[i] < zero(T) && (m[i] = zero(T))
    end
    s = sum(m)
    s > zero(T) && (m ./= s)
    return m
end

# =============================================================================
# _winberry_state_scales — standardized state coordinates
# =============================================================================

"""
    _winberry_scales(M) → s

Per-income-state standard deviations `s_j = sqrt(M[j,2])`, used to express the
moment state in *standardized* coordinates

    x_{j,1} = (m_{j,1} − m̄_{j,1}) / s_j,   x_{j,i} = (m_{j,i} − m̄_{j,i}) / s_j^i.

Raw central moments span many orders of magnitude on a calibrated grid
(`m_1 ≈ 40`, `m_3 ≈ 10^5`), so a linear system written in them is badly scaled
and its eigenvalues are numerically meaningless.  Every Jacobian below is
therefore assembled directly in `x`.
"""
function _winberry_scales(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    n_e = size(M, 1)
    s = Vector{T}(undef, n_e)
    @inbounds for j in 1:n_e
        s[j] = sqrt(max(M[j, 2], eps(T)))
    end
    return s
end

# Flatten/unflatten a moment deviation between the n_e × n_m matrix layout and
# the standardized state vector (column-major: income fastest, moment slowest).
@inline _win_index(j::Int, i::Int, n_e::Int) = (i - 1) * n_e + j

function _winberry_to_state(dM::AbstractMatrix{T}, s::AbstractVector{T}) where {T<:AbstractFloat}
    n_e, n_m = size(dM)
    x = Vector{T}(undef, n_e * n_m)
    @inbounds for i in 1:n_m, j in 1:n_e
        x[_win_index(j, i, n_e)] = dM[j, i] / s[j]^i
    end
    return x
end

# =============================================================================
# _winberry_linearize — moment-space linearization (Winberry 2018 + Reiter GE)
# =============================================================================

"""
    _winberry_linearize(ss, ip, grid, income; n_moments=3, n_quad=4, model=:aiyagari,
                        het_params=Dict(), n_sim=200, rng=nothing)
        → (G1, impact, n_dist, explained, basis)

Linearize a heterogeneous agent model whose distribution is carried by the
Winberry (2018) parametric family instead of the Young histogram.

The distribution state is the standardized moment vector — `n_income ×
n_moments` coordinates rather than `n_a × n_income`.  Its law of motion is
[`_winberry_forward`](@ref), differentiated by central finite differences in
standardized units; the general-equilibrium closure is identical to the one
`_reiter_linearize` uses, because only the *representation* of the
distribution changes, not the economics:

- **Aiyagari** — state `[x_t; K_t; Z_t]`, prices from the firm FOC at a
  predetermined `K` (see [`_aiyagari_foc_derivatives`](@ref)).
- **Huggett** — state `[x_t; w_t]`, the rate pinned every period by bond market
  clearing `∫a′ = 0`.

Aggregate capital is *exactly* linear in the state, `K = Σ_j mass_j m_{j,1}`,
so — unlike the SVD reduction — the aggregator carries no approximation error
at all; the whole error sits in the density's shape.

# Returns
- `G1::Matrix{T}` — `(n_dist + n_agg)²` transition
- `impact::Matrix{T}` — `(n_dist + n_agg) × 1` shock loading
- `n_dist::Int` — number of distribution states, `n_income × n_moments`
- `explained::T` — fraction of the aggregate-capital response to random
  histogram perturbations that the moment state reproduces
- `basis::Matrix{T}` — `N × n_dist` map from moment deviations back to a
  histogram deviation (`N = n_a·n_e`), so `distribution_irf` works unchanged

# References
- Winberry, T. (2018). A method for solving and estimating heterogeneous agent
  macro models. *Quantitative Economics*, 9(3), 1123–1151.
"""
function _winberry_linearize(ss::HASteadyState{T}, ip::IndividualProblem{T},
                              grid::HAGrid{T}, income::IncomeProcess{T};
                              n_moments::Int=3, n_quad::Int=4,
                              model::Symbol=:aiyagari,
                              het_params::Dict{Symbol,T}=Dict{Symbol,T}(),
                              dm_step::Real=1e-4, dr_step::Real=1e-5,
                              dw_step::Real=1e-5, n_sim::Int=200,
                              rng::Union{Nothing,AbstractRNG}=nothing) where {T<:AbstractFloat}
    grid.n_dims == 1 || throw(ArgumentError(
        "Winberry linearization requires a one-asset grid (got n_dims = $(grid.n_dims))."))
    ip.n_asset_dims == 1 || throw(ArgumentError(
        "Winberry linearization requires a one-asset individual problem."))

    a_pol_ss = ss.policies[:savings]
    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = grid.n_income
    N = n_a * n_e
    nodes, weights = winberry_quadrature(grid; n_quad=n_quad)

    # ── Step 1: moment steady state ──────────────────────────────────────────
    # The linearization point must be a fixed point of the MOMENT map, not merely
    # the moments of the Young histogram: linearizing around a non-stationary point
    # leaves a constant drift in every IRF. Warm-start from whatever the steady
    # state already carries, then re-converge.
    fam0 = ss.parametric
    M_init = if fam0 !== nothing && fam0.n_moments == n_moments
        reduce(vcat, transpose(pd.moments) for pd in fam0.densities)
    else
        first(winberry_moments(ss.distribution, grid; n_moments=n_moments))
    end
    stat = _winberry_stationary(a_pol_ss, grid, income; n_moments=n_moments,
                                 n_quad=n_quad, M_init=M_init)
    M_ss, mass, lam_ss = stat.moments, stat.mass, stat.lambdas
    stat.converged ||
        @warn "Winberry linearization: the moment steady state did not fully " *
              "converge; the linearization point carries residual drift." maxlog = 1

    s_scale = _winberry_scales(M_ss)
    n_dist = n_e * n_moments

    # Baseline forward image (equal to M_ss up to the fixed-point tolerance; using
    # the COMPUTED image rather than M_ss itself keeps every finite difference a
    # difference of two evaluations of the same map, cancelling that residual).
    M_base, _, _ = _winberry_forward(M_ss, mass, a_pol_ss, grid, income, nodes,
                                     weights; lambda_warm=lam_ss)

    # ── Step 2: ∂M′/∂M by central differences in standardized units ──────────
    h = T(dm_step)
    J_M = zeros(T, n_dist, n_dist)
    Mp = similar(M_ss)
    for i in 1:n_moments, j in 1:n_e
        col = _win_index(j, i, n_e)
        step = h * s_scale[j]^i
        copyto!(Mp, M_ss); Mp[j, i] += step
        M_up, _, _ = _winberry_forward(Mp, mass, a_pol_ss, grid, income, nodes,
                                        weights; lambda_warm=lam_ss)
        copyto!(Mp, M_ss); Mp[j, i] -= step
        M_dn, _, _ = _winberry_forward(Mp, mass, a_pol_ss, grid, income, nodes,
                                        weights; lambda_warm=lam_ss)
        # (M_up − M_dn)/(2·step) is ∂M′/∂m_{j,i}; multiplying by the input scale
        # s_j^i converts the DENOMINATOR to standardized units, and
        # _winberry_to_state converts the NUMERATOR. Both are needed.
        J_M[:, col] .= _winberry_to_state((M_up .- M_dn) .* (s_scale[j]^i /
                                          (T(2) * step)), s_scale)
    end

    # ── Step 3: ∂M′/∂prices via re-solved policies ───────────────────────────
    prices_r = copy(ss.prices); prices_r[:r] = ss.prices[:r] + T(dr_step)
    _, a_pol_r = _egm_solve(ip, grid, income, prices_r; max_iter=1000, tol=T(1e-10))
    M_r, _, _ = _winberry_forward(M_ss, mass, a_pol_r, grid, income, nodes, weights;
                                   lambda_warm=lam_ss)

    prices_w = copy(ss.prices); prices_w[:w] = ss.prices[:w] + T(dw_step)
    _, a_pol_w = _egm_solve(ip, grid, income, prices_w; max_iter=1000, tol=T(1e-10))
    M_w, _, _ = _winberry_forward(M_ss, mass, a_pol_w, grid, income, nodes, weights;
                                   lambda_warm=lam_ss)

    J_r = _winberry_to_state((M_r .- M_base) ./ T(dr_step), s_scale)
    J_w = _winberry_to_state((M_w .- M_base) ./ T(dw_step), s_scale)

    # ── Step 4: aggregate loading (exact) ────────────────────────────────────
    # K = Σ_j mass_j m_{j,1} = Σ_j mass_j (m̄_{j,1} + s_j x_{j,1}).
    K_load = zeros(T, n_dist)
    @inbounds for j in 1:n_e
        K_load[_win_index(j, 1, n_e)] = mass[j] * s_scale[j]
    end

    # ── Step 5: histogram basis (for distribution_irf) ───────────────────────
    basis = _winberry_basis_matrix(M_ss, mass, grid, nodes, weights, s_scale;
                                    dm_step=h, lambda_warm=lam_ss)

    # ── Step 6: accuracy of the moment reduction ─────────────────────────────
    explained = _winberry_explained(ss, grid, income, M_ss, mass, s_scale, J_M,
                                     K_load, a_pol_ss; n_sim=n_sim, rng=rng)

    # ── Step 7: general-equilibrium closure ──────────────────────────────────
    if model === :huggett
        rho = T(get(het_params, :rho_e, 0.9))
        A_r = dot(K_load, J_r)
        A_w = dot(K_load, J_w)
        sav_load = transpose(J_M) * K_load          # ∂(∫a′)/∂x at fixed prices

        abs(A_r) > eps(T) || error("Winberry Reiter (Huggett): ∂(asset demand)/∂r ≈ 0 " *
                                   "(cannot pin the rate).")

        channel_w = J_w .- J_r .* (A_w / A_r)
        n_total = n_dist + 1
        G1 = zeros(T, n_total, n_total)
        G1[1:n_dist, 1:n_dist] .= J_M .- (J_r ./ A_r) * transpose(sav_load)
        G1[1:n_dist, n_dist + 1] .= channel_w
        G1[n_dist + 1, n_dist + 1] = rho

        impact_vec = zeros(T, n_total, 1)
        impact_vec[n_dist + 1, 1] = one(T)
        impact_vec[1:n_dist, 1] .= channel_w

        _reiter_warn_unstable(G1, "Huggett/Winberry")
        return G1, impact_vec, n_dist, explained, basis
    end

    for k in (:alpha, :delta, :rho_z)
        haskey(het_params, k) ||
            error("Winberry (Aiyagari) linearization requires parameter :$k in spec params")
    end
    alpha_val = T(het_params[:alpha])
    delta_val = T(het_params[:delta])
    Z_val     = T(get(het_params, :Z, one(T)))
    rho_z     = T(het_params[:rho_z])
    K_ss      = ss.aggregates[:K]

    dr_dK, dw_dK, dr_dZ, dw_dZ =
        _aiyagari_foc_derivatives(ss.prices[:r], ss.prices[:w], K_ss,
                                  alpha_val, delta_val, Z_val)

    K_column = J_r .* dr_dK .+ J_w .* dw_dK
    Z_column = J_r .* dr_dZ .+ J_w .* dw_dZ

    n_total = n_dist + 2
    G1 = zeros(T, n_total, n_total)
    G1[1:n_dist, 1:n_dist] .= J_M
    G1[1:n_dist, n_dist + 1] .= K_column
    G1[1:n_dist, n_dist + 2] .= Z_column
    G1[n_dist + 1, 1:n_dist] .= vec(transpose(J_M) * K_load)
    G1[n_dist + 1, n_dist + 1] = dot(K_load, K_column)
    G1[n_dist + 1, n_dist + 2] = dot(K_load, Z_column)
    G1[n_dist + 2, n_dist + 2] = rho_z

    impact_vec = zeros(T, n_total, 1)
    impact_vec[n_dist + 2, 1] = one(T)
    impact_vec[1:n_dist, 1] .= Z_column
    impact_vec[n_dist + 1, 1] = dot(K_load, Z_column)

    _reiter_warn_unstable(G1, "Aiyagari/Winberry")
    return G1, impact_vec, n_dist, explained, basis
end

"""
    _winberry_basis_matrix(M_ss, mass, grid, nodes, weights, s_scale; dm_step, lambda_warm)
        → Matrix{T}

`N × n_dist` map from a standardized moment deviation to the corresponding
histogram deviation, `d_dev = basis · x`.  Column `(j, i)` is a central finite
difference of [`winberry_histogram`](@ref) in moment coordinate `(j, i)`.

This is the parametric counterpart of the Reiter SVD basis `U_k`, and having it
in the same `N × n_reduced` shape is what lets `distribution_irf` and
`inequality_irf` work on a Winberry solution without a special case.
"""
function _winberry_basis_matrix(M_ss::AbstractMatrix{T}, mass::AbstractVector{T},
                                 grid::HAGrid{T}, nodes::Vector{T}, weights::Vector{T},
                                 s_scale::AbstractVector{T};
                                 dm_step::T=T(1e-4),
                                 lambda_warm::Union{Nothing,Vector{Vector{T}}}=nothing
                                 ) where {T<:AbstractFloat}
    n_e, n_m = size(M_ss)
    n_a = grid.n_points[1]
    basis = zeros(T, n_a * n_e, n_e * n_m)
    Mp = similar(M_ss)
    for i in 1:n_m, j in 1:n_e
        step = dm_step * s_scale[j]^i
        copyto!(Mp, M_ss); Mp[j, i] += step
        h_up = winberry_histogram(_build_family(Mp, mass, nodes, weights, grid;
                                                lambda_warm=lambda_warm), grid)
        copyto!(Mp, M_ss); Mp[j, i] -= step
        h_dn = winberry_histogram(_build_family(Mp, mass, nodes, weights, grid;
                                                lambda_warm=lambda_warm), grid)
        # Denominator in standardized units: Δx = step / s_j^i = dm_step.
        basis[:, _win_index(j, i, n_e)] .= (h_up .- h_dn) ./ (T(2) * dm_step)
    end
    return basis
end

"""
    _winberry_explained(ss, grid, income, M_ss, mass, s_scale, J_M, K_load, a_pol; n_sim, rng)
        → T

Fraction of the aggregate-capital response that the moment state reproduces.

For random, per-income-state mass-preserving histogram perturbations `δ`, the
full model's one-period capital response is `a' Λ_ss δ`, while the Winberry
system only sees the induced moment deviation `x = P δ` and predicts
`K_load' J_M x`.  The reported number is `1 − Var(residual)/Var(full)`, the same
form `_reiter_linearize` reports for the SVD basis.

The probe directions are *mass-weighted* (`δ ∝ d_ss ⊙ noise`) rather than
uniform.  Uniform noise puts as much probe mass on an empty cell at the top of
the asset grid as on the mode, and the higher-moment rows of `P` scale like
`((a − μ)/σ)^i` — of order `10^6` at `a_max` on the calibrated Krusell–Smith
grid — so the statistic ends up reporting the reduction's response to
perturbations the model can never experience (measured: 0.40 at three moments
against 0.9999 at two, while the three-moment IRFs are the *more* accurate of
the pair).  Mass weighting confines the probes to where the stationary
distribution actually lives.

`P = ∂x/∂d` is exact, not a finite difference: for a histogram with mass
`mass_j` and mean `μ_j`,

    ∂μ_j/∂d_{jk}   = (a_k − μ_j) / mass_j
    ∂m_{j,i}/∂d_{jk} = [(a_k − μ_j)^i − i·m_{j,i−1}(a_k − μ_j) − m_{j,i}] / mass_j

with `m_{j,1} ≡ 0` (the first *central* moment).
"""
function _winberry_explained(ss::HASteadyState{T}, grid::HAGrid{T},
                              income::IncomeProcess{T}, M_ss::AbstractMatrix{T},
                              mass::AbstractVector{T}, s_scale::AbstractVector{T},
                              J_M::AbstractMatrix{T}, K_load::AbstractVector{T},
                              a_pol::AbstractMatrix{T};
                              n_sim::Int=200,
                              rng::Union{Nothing,AbstractRNG}=nothing) where {T<:AbstractFloat}
    a_grid = grid.grids[1]
    n_a = length(a_grid)
    n_e = grid.n_income
    N = n_a * n_e
    n_e_m = size(M_ss, 2)
    rng_actual = isnothing(rng) ? Random.MersenneTwister(1234) : rng

    Lambda = _build_transition_matrix(collect(T, a_pol), grid, income)
    a_vec = zeros(T, N)
    @inbounds for j in 1:n_e, i in 1:n_a
        a_vec[(j - 1) * n_a + i] = a_grid[i]
    end

    # P = ∂x/∂d, exact (see docstring).
    P = zeros(T, n_e * n_e_m, N)
    @inbounds for j in 1:n_e
        mass[j] <= zero(T) && continue
        mu = M_ss[j, 1]
        for k in 1:n_a
            dev = a_grid[k] - mu
            col = (j - 1) * n_a + k
            P[_win_index(j, 1, n_e), col] = dev / (mass[j] * s_scale[j])
            dp = dev
            for i in 2:n_e_m
                dp *= dev
                cm_prev = i == 2 ? zero(T) : M_ss[j, i - 1]
                raw = (dp - T(i) * cm_prev * dev - M_ss[j, i]) / mass[j]
                P[_win_index(j, i, n_e), col] = raw / s_scale[j]^i
            end
        end
    end

    var_full = zero(T)
    var_resid = zero(T)
    n_test = min(n_sim, 100)
    d_ss = vec(ss.distribution)
    for _ in 1:n_test
        delta = randn(rng_actual, T, N) .* d_ss
        # The Winberry state holds income-state masses fixed (the idiosyncratic
        # chain is exogenous), so probe only mass-preserving directions.
        @inbounds for j in 1:n_e
            blk = view(delta, ((j - 1) * n_a + 1):(j * n_a))
            blk .-= mean(blk)
        end
        nrm = maximum(abs, delta)
        nrm > zero(T) && (delta .*= T(1e-6) / nrm)
        dK_full = dot(a_vec, Lambda * delta)
        dK_red = dot(K_load, J_M * (P * delta))
        var_full += dK_full^2
        var_resid += (dK_full - dK_red)^2
    end
    explained = var_full > zero(T) ? one(T) - var_resid / var_full : one(T)
    return clamp(explained, zero(T), one(T))
end
