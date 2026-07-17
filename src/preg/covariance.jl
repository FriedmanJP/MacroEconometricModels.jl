# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Panel covariance estimators: entity-cluster, time-cluster, two-way cluster
(Cameron-Gelbach-Miller 2011), and Driscoll-Kraay (1998) HAC.
"""

using LinearAlgebra

# =============================================================================
# Entity Cluster-Robust Covariance
# =============================================================================

"""
    _panel_cluster_vcov(X, resid, XtXinv, groups; n_absorbed=0) -> Matrix{T}

Entity cluster-robust variance-covariance estimator for panel data.

V = (X'X)^{-1} B (X'X)^{-1}, where B = sum_g (X_g' e_g)(X_g' e_g)'.
Includes G/(G-1) * (n-1)/(n-k-n_absorbed) finite-sample correction.

`n_absorbed` counts fixed-effect parameters absorbed by the within-transformation
that are not columns of `X`. Following the reghdfe convention, absorbed FE nested
within the clustering dimension (entity FE clustered on entity — the default panel
setup) are excluded, so the default `n_absorbed=0` reproduces the standard
correction; pass the absorbed count only for non-nested dimensions.

# Arguments
- `X::Matrix{T}` — demeaned regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `groups::AbstractVector{Int}` — group assignment for each observation
- `n_absorbed::Int` — absorbed (non-nested) FE parameters for the dof correction

# References
- Arellano, M. (1987). *Oxford Bulletin of Economics and Statistics* 49(4), 431-434.
- Cameron, A. C. & Miller, D. L. (2015). *JPE* 50, 327-372.
"""
function _panel_cluster_vcov(X::Matrix{T}, resid::Vector{T},
                             XtXinv::Matrix{T}, groups::AbstractVector{Int};
                             n_absorbed::Int=0) where {T<:AbstractFloat}
    n, k = size(X)
    unique_groups = unique(groups)
    G = length(unique_groups)
    G < 2 && throw(ArgumentError("Need at least 2 clusters for cluster-robust SE"))

    B = zeros(T, k, k)
    for g in unique_groups
        idx = findall(==(g), groups)
        X_g = @view X[idx, :]
        e_g = @view resid[idx]
        score_g = X_g' * e_g
        B .+= score_g * score_g'
    end

    correction = T(G) / T(G - 1) * T(n - 1) / T(max(n - k - n_absorbed, 1))
    B .*= correction

    XtXinv * B * XtXinv
end

# =============================================================================
# Time Cluster-Robust Covariance
# =============================================================================

"""
    _panel_time_cluster_vcov(X, resid, XtXinv, time_ids) -> Matrix{T}

Time cluster-robust variance-covariance estimator for panel data.

Clusters observations by time period rather than entity.
Same formula as entity-cluster but grouping by time.

# Arguments
- `X::Matrix{T}` — demeaned regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `time_ids::AbstractVector{Int}` — time period for each observation

# References
- Cameron, A. C. & Miller, D. L. (2015). *JPE* 50, 327-372.
"""
function _panel_time_cluster_vcov(X::Matrix{T}, resid::Vector{T},
                                  XtXinv::Matrix{T}, time_ids::AbstractVector{Int};
                                  n_absorbed::Int=0) where {T<:AbstractFloat}
    n, k = size(X)
    unique_times = unique(time_ids)
    G = length(unique_times)
    G < 2 && throw(ArgumentError("Need at least 2 time clusters for time-cluster SE"))

    B = zeros(T, k, k)
    for t in unique_times
        idx = findall(==(t), time_ids)
        X_t = @view X[idx, :]
        e_t = @view resid[idx]
        score_t = X_t' * e_t
        B .+= score_t * score_t'
    end

    correction = T(G) / T(G - 1) * T(n - 1) / T(max(n - k - n_absorbed, 1))
    B .*= correction

    XtXinv * B * XtXinv
end

# =============================================================================
# Two-Way Cluster-Robust Covariance (Cameron-Gelbach-Miller 2011)
# =============================================================================

"""
    _panel_twoway_vcov(X, resid, XtXinv, groups, time_ids) -> Matrix{T}

Two-way cluster-robust variance-covariance estimator.

V = V_entity + V_time - V_entity_x_time

where V_entity_x_time clusters on the intersection of entity and time
(which, for typical panels with unique (i,t) pairs, equals the
HC1 heteroskedasticity-robust estimator).

# Arguments
- `X::Matrix{T}` — demeaned regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `groups::AbstractVector{Int}` — entity group for each observation
- `time_ids::AbstractVector{Int}` — time period for each observation

# References
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).
  *JBES* 29(2), 238-249.
"""
function _panel_twoway_vcov(X::Matrix{T}, resid::Vector{T},
                            XtXinv::Matrix{T}, groups::AbstractVector{Int},
                            time_ids::AbstractVector{Int}) where {T<:AbstractFloat}
    V_entity = _panel_cluster_vcov(X, resid, XtXinv, groups)
    V_time = _panel_time_cluster_vcov(X, resid, XtXinv, time_ids)

    # Intersection clustering: unique (entity, time) pairs
    # For balanced panels with unique (i,t), each obs is its own cluster -> HC1
    n, k = size(X)
    interaction = groups .* maximum(time_ids) .+ time_ids
    V_interaction = _panel_cluster_vcov(X, resid, XtXinv, interaction)

    V = V_entity .+ V_time .- V_interaction

    # Ensure positive semi-definiteness
    V = (V .+ V') ./ 2
    V
end

# =============================================================================
# Driscoll-Kraay (1998) HAC Covariance
# =============================================================================

"""
    _panel_driscoll_kraay_vcov(X, resid, XtXinv, groups, time_ids; bandwidth=nothing) -> Matrix{T}

Driscoll-Kraay (1998) heteroskedasticity- and autocorrelation-consistent
covariance estimator for panel data.

Averages moment conditions cross-sectionally at each time period, then
applies Newey-West HAC kernel to the resulting time series of averages.

# Arguments
- `X::Matrix{T}` — demeaned regressor matrix (n x k)
- `resid::Vector{T}` — residuals (n)
- `XtXinv::Matrix{T}` — (X'X)^{-1}
- `groups::AbstractVector{Int}` — entity group for each observation
- `time_ids::AbstractVector{Int}` — time period for each observation
- `bandwidth::Union{Nothing,Int}` — Newey-West bandwidth (default: floor(T^(1/4)))

# References
- Driscoll, J. C. & Kraay, A. C. (1998). *Review of Economics and Statistics*
  80(4), 549-560.
"""
function _panel_driscoll_kraay_vcov(X::Matrix{T}, resid::Vector{T},
                                    XtXinv::Matrix{T}, groups::AbstractVector{Int},
                                    time_ids::AbstractVector{Int};
                                    bandwidth::Union{Nothing,Int}=nothing) where {T<:AbstractFloat}
    n, k = size(X)
    unique_times = sort(unique(time_ids))
    n_times = length(unique_times)

    # Default bandwidth: floor(T^(1/4))
    bw = something(bandwidth, max(1, floor(Int, n_times^(1/4))))

    # Cross-sectional average of moment conditions at each time period
    # h_t = (1/N) sum_i x_{it} e_{it}  (but we use sum, not average, for sandwich)
    H = zeros(T, n_times, k)
    for (j, t) in enumerate(unique_times)
        idx = findall(==(t), time_ids)
        X_t = @view X[idx, :]
        e_t = @view resid[idx]
        # Sum of x_{it} * e_{it} at time t
        for i in eachindex(idx)
            for p in 1:k
                H[j, p] += X_t[i, p] * e_t[i]
            end
        end
    end

    # Newey-West HAC on the time series of moment sums
    S = zeros(T, k, k)

    # Lag 0: S_0 = sum_t h_t h_t'
    for j in 1:n_times
        h_t = @view H[j, :]
        S .+= h_t * h_t'
    end

    # Lags 1..bw: Bartlett kernel w(l) = 1 - l/(bw+1)
    for l in 1:bw
        weight = one(T) - T(l) / T(bw + 1)
        Gamma_l = zeros(T, k, k)
        for j in (l+1):n_times
            h_t = @view H[j, :]
            h_lag = @view H[j-l, :]
            Gamma_l .+= h_t * h_lag'
        end
        S .+= weight .* (Gamma_l .+ Gamma_l')
    end

    XtXinv * S * XtXinv
end

# =============================================================================
# Beck-Katz (1995) Panel-Corrected Standard Errors (PCSE)  [EV-25, #433]
# =============================================================================

"""
    _panel_pcse_vcov(X, resid, XtXinv, groups, time_ids; unbalanced=:casewise) -> Matrix{T}

Beck & Katz (1995) panel-corrected standard errors for time-series-cross-section
(TSCS) data with contemporaneous cross-section correlation.

Forms the `N×N` contemporaneous residual covariance
`Σ̂_ij = (Σ_t ê_it ê_jt) / T` and returns the sandwich

    V = (X'X)⁻¹ [ Σ_t X_t' Σ̂ X_t ] (X'X)⁻¹,

where `X_t` is the block of regressor rows observed at time `t`. The meat
`Σ_t X_t' Σ̂ X_t` is **accumulated time-by-time** — the `NT×NT` Kronecker
`Σ̂ ⊗ I` is **never materialized** (it is the memory trap on wide panels).

# Arguments
- `X::Matrix{T}` — regressor matrix used to form `XtXinv` (n × k); for FE this is
  the within-demeaned design, matching the residuals `resid`.
- `resid::Vector{T}` — residuals aligned with the rows of `X` (n)
- `XtXinv::Matrix{T}` — `(X'X)⁻¹`
- `groups::AbstractVector{Int}` — entity (cross-section) id per observation
- `time_ids::AbstractVector{Int}` — time period per observation

# Keyword Arguments
- `unbalanced::Symbol` — `:casewise` (only fully-observed periods enter `Σ̂`;
  Beck-Katz default) or `:pairwise` (`Σ̂_ij` over the overlapping periods of `i`
  and `j`).

`Σ̂` is never inverted, so the sandwich is well-defined even when `Σ̂` is
rank-deficient; a rank-deficient casewise `Σ̂` (fewer fully-observed periods than
units, `T < N`) triggers a warning rather than a garbage inverse.

# References
- Beck, N. & Katz, J. N. (1995). *American Political Science Review* 89(3), 634-647.
"""
function _panel_pcse_vcov(X::Matrix{T}, resid::Vector{T}, XtXinv::Matrix{T},
                          groups::AbstractVector{Int}, time_ids::AbstractVector{Int};
                          unbalanced::Symbol=:casewise) where {T<:AbstractFloat}
    unbalanced in (:casewise, :pairwise) ||
        throw(ArgumentError("unbalanced must be :casewise or :pairwise; got :$unbalanced"))
    n, k = size(X)

    unit_ids = sort(unique(groups))
    Ncs = length(unit_ids)
    time_vals = sort(unique(time_ids))
    Tn = length(time_vals)
    unit_pos = Dict(g => i for (i, g) in enumerate(unit_ids))
    time_pos = Dict(t => j for (j, t) in enumerate(time_vals))

    # Residual panel E (Ncs × Tn) + presence mask.
    E = zeros(T, Ncs, Tn)
    present = falses(Ncs, Tn)
    unit_of = Vector{Int}(undef, n)       # cached unit position per obs row
    for r in 1:n
        i = unit_pos[groups[r]]
        j = time_pos[time_ids[r]]
        E[i, j] = resid[r]
        present[i, j] = true
        unit_of[r] = i
    end

    # ---- Contemporaneous residual covariance Σ̂ (Ncs × Ncs) ----
    Sigma = zeros(T, Ncs, Ncs)
    if unbalanced == :casewise
        full_cols = [j for j in 1:Tn if all(@view present[:, j])]
        Tf = length(full_cols)
        Tf == 0 && throw(ArgumentError(
            "PCSE casewise: no fully-observed time period exists; use unbalanced=:pairwise"))
        Tf < Ncs && @warn "PCSE casewise Σ̂ is rank-deficient (fully-observed periods " *
            "T_full=$Tf < N=$Ncs units): the contemporaneous covariance is singular. " *
            "Beck & Katz (1995) require T ≥ N; consider unbalanced=:pairwise or a longer panel."
        Ef = @view E[:, full_cols]
        Sigma = (Ef * Ef') ./ T(Tf)
    else  # :pairwise
        @inbounds for i in 1:Ncs, jj in i:Ncs
            s = zero(T)
            cnt = 0
            for c in 1:Tn
                if present[i, c] && present[jj, c]
                    s += E[i, c] * E[jj, c]
                    cnt += 1
                end
            end
            val = cnt > 0 ? s / T(cnt) : zero(T)
            Sigma[i, jj] = val
            Sigma[jj, i] = val
        end
    end

    # ---- Sandwich meat: accumulate Σ_t X_t' Σ̂_t X_t time-by-time ----
    # (Σ̂_t is the submatrix of Σ̂ for units present at t; NEVER form Σ̂ ⊗ I.)
    tmap = Dict{Int,Vector{Int}}()
    for r in 1:n
        push!(get!(() -> Int[], tmap, time_ids[r]), r)
    end
    meat = zeros(T, k, k)
    for t in time_vals
        rows = tmap[t]
        m_t = length(rows)
        Xt = @view X[rows, :]                       # m_t × k
        Sig_t = Matrix{T}(undef, m_t, m_t)
        @inbounds for a in 1:m_t, b in 1:m_t
            Sig_t[a, b] = Sigma[unit_of[rows[a]], unit_of[rows[b]]]
        end
        meat .+= Xt' * (Sig_t * Xt)
    end

    XtXinv * meat * XtXinv
end

# =============================================================================
# Panel Covariance Dispatch
# =============================================================================

"""
    _panel_vcov(X, resid, XtXinv, groups, time_ids, cov_type;
                bandwidth=nothing, n_absorbed=0) -> Matrix{T}

Dispatch to the appropriate panel covariance estimator.

# Supported types
- `:ols` — classical homoskedastic: sigma^2 (X'X)^{-1}
- `:cluster` — entity cluster-robust (default)
- `:twoway` — two-way cluster (Cameron-Gelbach-Miller 2011)
- `:driscoll_kraay` — Driscoll-Kraay (1998) HAC
- `:pcse` — Beck-Katz (1995) panel-corrected SE (`pcse_unbalanced` ∈ `:casewise`/`:pairwise`)

`n_absorbed` is forwarded to the `:cluster` small-sample dof correction — count
only absorbed FE parameters NOT nested within the entity clustering dimension
(reghdfe convention); entity FE clustered on entity contribute 0.
"""
function _panel_vcov(X::Matrix{T}, resid::Vector{T}, XtXinv::Matrix{T},
                     groups::AbstractVector{Int}, time_ids::AbstractVector{Int},
                     cov_type::Symbol;
                     bandwidth::Union{Nothing,Int}=nothing,
                     n_absorbed::Int=0,
                     pcse_unbalanced::Symbol=:casewise) where {T<:AbstractFloat}
    n, k = size(X)

    if cov_type == :ols
        sigma2 = dot(resid, resid) / T(n - k)
        return sigma2 .* XtXinv
    elseif cov_type == :cluster
        return _panel_cluster_vcov(X, resid, XtXinv, groups; n_absorbed=n_absorbed)
    elseif cov_type == :twoway
        return _panel_twoway_vcov(X, resid, XtXinv, groups, time_ids)
    elseif cov_type == :driscoll_kraay
        return _panel_driscoll_kraay_vcov(X, resid, XtXinv, groups, time_ids;
                                          bandwidth=bandwidth)
    elseif cov_type == :pcse
        return _panel_pcse_vcov(X, resid, XtXinv, groups, time_ids;
                                unbalanced=pcse_unbalanced)
    else
        throw(ArgumentError("cov_type must be :ols, :cluster, :twoway, :driscoll_kraay, or :pcse; got :$cov_type"))
    end
end
