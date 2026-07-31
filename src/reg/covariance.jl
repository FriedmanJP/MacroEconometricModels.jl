# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Heteroskedasticity-consistent and cluster-robust covariance estimators for
cross-sectional regression models.

Implements HC0 (White 1980), HC1, HC2, HC3 (MacKinnon & White 1985),
and cluster-robust (Arellano 1987) covariance matrices.
"""

using LinearAlgebra

# =============================================================================
# Hat Matrix Diagonal
# =============================================================================

"""
    _hat_diag(X::Matrix{T}, XtXinv::Matrix{T}; weights=nothing) -> Vector{T}

Compute the diagonal of the hat (projection) matrix H = X (X'X)^{-1} X'.

Element h_ii measures the leverage of observation i. Used by HC2 and HC3.

For GLM/IRLS fits pass the working weights `w` (and `XtXinv = (X'WX)^{-1}`), giving
the GLM leverage h_ii = w_i x_i'(X'WX)^{-1} x_i, the diagonal of
H = W^{1/2} X (X'WX)^{-1} X' W^{1/2} (McCullagh & Nelder 1989).
"""
function _hat_diag(X::Matrix{T}, XtXinv::Matrix{T};
                   weights::Union{Nothing,Vector{T}}=nothing) where {T<:AbstractFloat}
    n = size(X, 1)
    h = Vector{T}(undef, n)
    @inbounds for i in 1:n
        xi = @view X[i, :]
        h[i] = dot(xi, XtXinv * xi)
        if weights !== nothing
            h[i] *= weights[i]
        end
        # Clamp to [0, 1) for numerical safety
        h[i] = clamp(h[i], zero(T), one(T) - T(1e-10))
    end
    h
end

# =============================================================================
# Cluster-Robust Covariance
# =============================================================================

"""
    _cluster_vcov(X, resid, XtXinv, clusters) -> Matrix{T}

Cluster-robust (Arellano 1987) covariance estimator.

V = (X'X)^{-1} B (X'X)^{-1}, where B = sum_g (X_g' e_g)(X_g' e_g)'.
Includes the standard G/(G-1) * n/(n-k) finite-sample correction.

# Arguments
- `X::Matrix{T}` — regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `clusters::Vector` — cluster assignment for each observation

# References
- Arellano, M. (1987). *Oxford Bulletin of Economics and Statistics* 49(4), 431-434.
- Cameron, A. C. & Miller, D. L. (2015). *JPE* 50, 327-372.
"""
function _cluster_vcov(X::Matrix{T}, resid::Vector{T},
                       XtXinv::Matrix{T}, clusters::AbstractVector) where {T<:AbstractFloat}
    n, k = size(X)
    length(clusters) == n || throw(ArgumentError("clusters must have length n=$n"))

    unique_clusters = unique(clusters)
    G = length(unique_clusters)
    G < 2 && throw(ArgumentError("Need at least 2 clusters for cluster-robust SE"))

    # B = sum_g (X_g' e_g)(X_g' e_g)'
    B = zeros(T, k, k)
    for g in unique_clusters
        idx = findall(==(g), clusters)
        X_g = @view X[idx, :]
        e_g = @view resid[idx]
        score_g = X_g' * e_g   # k x 1
        B .+= score_g * score_g'
    end

    # Finite-sample correction: G/(G-1) * (n-1)/(n-k)
    correction = T(G) / T(G - 1) * T(n - 1) / T(n - k)
    B .*= correction

    XtXinv * B * XtXinv
end

# =============================================================================
# Main Covariance Dispatch
# =============================================================================

"""
    _reg_vcov(X, resid, cov_type, XtXinv; clusters=nothing) -> Matrix{T}

Compute the variance-covariance matrix of OLS/WLS coefficients.

# Supported covariance types
- `:ols` — classical homoskedastic: sigma^2 (X'X)^{-1}
- `:hc0` — White (1980): (X'X)^{-1} (sum e_i^2 x_i x_i') (X'X)^{-1}
- `:hc1` — HC0 with n/(n-k) finite-sample correction
- `:hc2` — HC0 with 1/(1-h_ii) leverage correction
- `:hc3` — HC0 with 1/(1-h_ii)^2 jackknife-like correction
- `:cluster` — cluster-robust (requires `clusters` argument)

# Arguments
- `X::Matrix{T}` — regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `cov_type::Symbol` — covariance estimator type
- `XtXinv::Matrix{T}` — precomputed (X'X)^{-1}
- `clusters` — cluster assignments (required if `cov_type == :cluster`)
- `weights` — GLM/IRLS working weights for the HC2/HC3 leverage (h_ii = w_i x_i'(X'WX)^{-1}x_i);
  leave `nothing` for OLS/WLS on (transformed) designs

# References
- White, H. (1980). *Econometrica* 48(4), 817-838.
- MacKinnon, J. G. & White, H. (1985). *JBES* 3(3), 305-314.
"""
function _reg_vcov(X::Matrix{T}, resid::Vector{T}, cov_type::Symbol,
                   XtXinv::Matrix{T};
                   clusters::Union{Nothing,AbstractVector}=nothing,
                   weights::Union{Nothing,Vector{T}}=nothing,
                   coords::Union{Nothing,AbstractMatrix}=nothing,
                   cutoff::Real=0.0, conley_kernel::Symbol=:bartlett,
                   conley_metric::Symbol=:euclidean,
                   time::Union{Nothing,AbstractVector}=nothing,
                   time_cutoff::Int=0, conley_psd::Bool=true) where {T<:AbstractFloat}
    n, k = size(X)

    if cov_type == :conley
        coords === nothing && throw(ArgumentError("coords required for :conley cov_type"))
        V, _ = _conley_vcov(X, resid, XtXinv, Matrix{T}(coords);
                            cutoff=cutoff, kernel=conley_kernel, metric=conley_metric,
                            time=time, time_cutoff=time_cutoff, psd=conley_psd)
        return V
    end

    if cov_type == :ols
        sigma2 = dot(resid, resid) / T(n - k)
        return sigma2 .* XtXinv
    end

    if cov_type == :cluster
        clusters === nothing && throw(ArgumentError("clusters required for :cluster cov_type"))
        return _cluster_vcov(X, resid, XtXinv, clusters)
    end

    # Heteroskedasticity-consistent estimators: V = (X'X)^{-1} S (X'X)^{-1}
    # where S = X' diag(omega_i) X
    cov_type in (:hc0, :hc1, :hc2, :hc3) ||
        throw(ArgumentError("cov_type must be :ols, :hc0, :hc1, :hc2, :hc3, :cluster, or " *
                            ":conley; got :$cov_type"))

    # Compute leverage if needed
    h = (cov_type == :hc2 || cov_type == :hc3) ? _hat_diag(X, XtXinv; weights=weights) : nothing

    # Build the meat matrix S = X' Omega X
    S = zeros(T, k, k)
    @inbounds for i in 1:n
        xi = @view X[i, :]
        ei = resid[i]

        omega_i = if cov_type == :hc0
            ei^2
        elseif cov_type == :hc1
            ei^2
        elseif cov_type == :hc2
            ei^2 / (one(T) - h[i])
        else  # :hc3
            ei^2 / (one(T) - h[i])^2
        end

        S .+= omega_i .* (xi * xi')
    end

    # HC1 finite-sample correction
    if cov_type == :hc1
        S .*= T(n) / T(n - k)
    end

    XtXinv * S * XtXinv
end

# =============================================================================
# Conley (1999) spatial HAC covariance
# =============================================================================

"""
    _great_circle(lat1, lon1, lat2, lon2) → T

Great-circle (haversine) distance in kilometres between two lat/lon points in degrees.
Uses the mean Earth radius 6371 km.
"""
@inline function _great_circle(lat1::T, lon1::T, lat2::T, lon2::T) where {T<:AbstractFloat}
    R = T(6371)
    φ1 = deg2rad(lat1); φ2 = deg2rad(lat2)
    dφ = φ2 - φ1
    dλ = deg2rad(lon2 - lon1)
    a = sin(dφ / 2)^2 + cos(φ1) * cos(φ2) * sin(dλ / 2)^2
    return 2 * R * asin(min(one(T), sqrt(max(a, zero(T)))))
end

"""
    _conley_distance(coords, i, j, metric) → T

Distance between observations `i` and `j`. `metric = :haversine` treats the first two
columns of `coords` as latitude and longitude in degrees; `:euclidean` uses the ordinary
Euclidean norm over all columns.
"""
@inline function _conley_distance(coords::Matrix{T}, i::Int, j::Int, metric::Symbol) where {T<:AbstractFloat}
    if metric === :haversine
        return _great_circle(coords[i, 1], coords[i, 2], coords[j, 1], coords[j, 2])
    else
        acc = zero(T)
        @inbounds for d in axes(coords, 2)
            acc += (coords[i, d] - coords[j, d])^2
        end
        return sqrt(acc)
    end
end

"""
    _conley_kernel(d, cutoff, kernel) → T

Kernel weight as a function of distance. Both kernels are zero beyond `cutoff`, which is
what makes the double sum truncatable.

- `:uniform` — 1 inside the cutoff, 0 outside
- `:bartlett` — `1 - d/cutoff`, decaying linearly to 0 at the cutoff

A `cutoff <= 0` means "no spatial correlation": only the own-observation term survives, so
the meat collapses to the White/HC0 outer-product sum.
"""
@inline function _conley_kernel(d::T, cutoff::T, kernel::Symbol) where {T<:AbstractFloat}
    cutoff <= zero(T) && return d == zero(T) ? one(T) : zero(T)
    d > cutoff && return zero(T)
    return kernel === :uniform ? one(T) : one(T) - d / cutoff
end

"""
    _conley_vcov(X, resid, XtXinv, coords; cutoff, kernel, metric, time, time_cutoff,
                 time_kernel, psd) → (V, adjusted)

Conley (1999) spatial-HAC sandwich `V = (X'X)⁻¹ S (X'X)⁻¹` with meat

```math
S = \\sum_i \\sum_j K_s(d_{ij}) \\, K_t(|t_i - t_j|) \\, x_i u_i (x_j u_j)'
```

`K_s` decays to zero beyond the spatial `cutoff`; `K_t` is a Bartlett weight in the time
dimension with bandwidth `time_cutoff` (a spatial panel — Conley crossed with Newey-West).
With no `time` supplied, `K_t ≡ 1` and the estimator is purely spatial.

Following the package HAC convention the meat carries no `1/n`.

Conley's `S` need not be positive semi-definite in finite samples. With `psd = true` the
eigenvalues of the symmetrized `S` are clipped at zero and a warning is emitted, as `acreg`
does; the second return value reports whether any clipping occurred.
"""
function _conley_vcov(X::Matrix{T}, resid::Vector{T}, XtXinv::Matrix{T},
                      coords::Matrix{T};
                      cutoff::Real, kernel::Symbol=:bartlett, metric::Symbol=:euclidean,
                      time::Union{Nothing,AbstractVector}=nothing,
                      time_cutoff::Int=0, time_kernel::Symbol=:bartlett,
                      psd::Bool=true) where {T<:AbstractFloat}
    n, k = size(X)
    size(coords, 1) == n || throw(ArgumentError(
        "coords must have $n rows (got $(size(coords, 1)))"))
    kernel in (:bartlett, :uniform) || throw(ArgumentError(
        "kernel must be :bartlett or :uniform; got :$kernel"))
    metric in (:euclidean, :haversine) || throw(ArgumentError(
        "metric must be :euclidean or :haversine; got :$metric"))
    metric === :haversine && size(coords, 2) < 2 && throw(ArgumentError(
        "metric = :haversine needs coords with 2 columns (latitude, longitude)"))
    time_cutoff >= 0 || throw(ArgumentError("time_cutoff must be >= 0, got $time_cutoff"))
    if time !== nothing
        length(time) == n || throw(ArgumentError("time must have length $n"))
    end
    cut = T(cutoff)

    # x_i · u_i, the score contributions the meat is built from.
    Xu = similar(X)
    @inbounds for i in 1:n, c in 1:k
        Xu[i, c] = X[i, c] * resid[i]
    end
    tvec = time === nothing ? nothing : collect(time)

    S = zeros(T, k, k)
    @inbounds for i in 1:n
        xi = @view Xu[i, :]
        for j in 1:n
            w = _conley_kernel(_conley_distance(coords, i, j, metric), cut, kernel)
            w == zero(T) && continue
            if tvec !== nothing
                lag = abs(tvec[i] - tvec[j])
                # Non-integer or out-of-range time gaps get zero weight, matching the
                # compact support of the Bartlett kernel.
                lag_i = try
                    Int(lag)
                catch
                    time_cutoff + 1
                end
                wt = lag_i == 0 ? one(T) :
                     kernel_weight(lag_i, time_cutoff, time_kernel, T)
                wt == zero(T) && continue
                w *= wt
            end
            xj = @view Xu[j, :]
            for c2 in 1:k, c1 in 1:k
                S[c1, c2] += w * xi[c1] * xj[c2]
            end
        end
    end

    S = (S .+ S') ./ 2                       # symmetrize against roundoff
    adjusted = false
    if psd
        F = eigen(Symmetric(S))
        if any(<(zero(T)), F.values)
            adjusted = true
            @warn "Conley covariance: the spatial-HAC meat was not positive semi-definite " *
                  "(min eigenvalue $(minimum(F.values))); clipping negative eigenvalues to " *
                  "zero. This is a known finite-sample property of the Conley (1999) " *
                  "estimator — see `acreg`. Pass `psd=false` to keep the raw matrix." maxlog = 1
            S = Matrix{T}(F.vectors * Diagonal(max.(F.values, zero(T))) * F.vectors')
            S = (S .+ S') ./ 2
        end
    end

    return XtXinv * S * XtXinv, adjusted
end

"""
    conley_se(m::RegModel; coords, cutoff, kernel=:bartlett, metric=:euclidean,
              time=nothing, time_cutoff=0, time_kernel=:bartlett, psd=true)
        → (vcov, se, adjusted)

Conley (1999) spatial-HAC covariance and standard errors for a fitted regression.

Errors of geographically close observations are correlated, and neither HC nor clustering
addresses that: HC assumes independence and clustering assumes correlation *within* groups
and none across them. Conley's estimator instead weights every pair by a kernel in their
distance, so correlation decays smoothly and vanishes beyond `cutoff`.

# Keyword Arguments
- `coords::AbstractMatrix` — `n × d` observation positions; with `metric = :haversine` the
  first two columns are latitude and longitude **in degrees**
- `cutoff::Real` — distance beyond which errors are treated as uncorrelated (kilometres for
  `:haversine`, the coordinates' own units otherwise). `cutoff <= 0` reduces the estimator
  to HC0.
- `kernel::Symbol = :bartlett` — `:bartlett` (linear decay) or `:uniform`
- `metric::Symbol = :euclidean` — `:haversine` for lat/lon
- `time`, `time_cutoff`, `time_kernel` — optional spatial-panel extension: a Newey-West
  weight in `|t_i - t_j|` multiplies the spatial weight
- `psd::Bool = true` — clip negative eigenvalues of the meat (with a warning)

# Returns
A named tuple `(vcov, se, adjusted)`; `adjusted` records whether the PSD clipping fired.

!!! note "Cost"
    The estimator is `O(n²)` in the number of observations because every pair must be
    weighted. The kernel is zero beyond `cutoff`, so distant pairs are skipped as soon as
    the distance is known, but the distance itself is still computed for each pair.

# Examples
```julia
m = estimate_reg(y, X)
c = conley_se(m; coords=[lat lon], cutoff=100.0, metric=:haversine)
c.se
```

See also [`estimate_reg`](@ref).

# References
- Conley, T. G. (1999). GMM estimation with cross sectional dependence.
  *Journal of Econometrics*, 92(1), 1–45.
- Colella, F., Lalive, R., Sakalli, S. O., & Thoenig, M. (2019). Inference with arbitrary
  clustering. IZA Discussion Paper 12584 (the `acreg` conventions).
"""
function conley_se(m::RegModel{T}; coords::AbstractMatrix, cutoff::Real,
                   kernel::Symbol=:bartlett, metric::Symbol=:euclidean,
                   time::Union{Nothing,AbstractVector}=nothing,
                   time_cutoff::Int=0, time_kernel::Symbol=:bartlett,
                   psd::Bool=true) where {T<:AbstractFloat}
    X = m.X
    XtXinv = robust_inv(Symmetric(X' * X))
    V, adjusted = _conley_vcov(X, m.residuals, Matrix{T}(XtXinv), Matrix{T}(coords);
                               cutoff=cutoff, kernel=kernel, metric=metric,
                               time=time, time_cutoff=time_cutoff,
                               time_kernel=time_kernel, psd=psd)
    return (vcov=V, se=sqrt.(max.(diag(V), zero(T))), adjusted=adjusted)
end
