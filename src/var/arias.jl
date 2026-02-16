# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

# =============================================================================
# Arias, Rubio-Ramírez, Waggoner (2018) - Zero + Sign Restrictions
# =============================================================================

"""Zero restriction: variable doesn't respond to shock at horizon."""
struct ZeroRestriction
    variable::Int
    shock::Int
    horizon::Int
end

"""Sign restriction: variable response to shock has required sign at horizon."""
struct SignRestriction
    variable::Int
    shock::Int
    horizon::Int
    sign::Int  # +1 or -1
end

"""Container for SVAR restrictions."""
struct SVARRestrictions
    zeros::Vector{ZeroRestriction}
    signs::Vector{SignRestriction}
    n_vars::Int
    n_shocks::Int
end

SVARRestrictions(n_vars::Int; zeros=ZeroRestriction[], signs=SignRestriction[]) =
    SVARRestrictions(zeros, signs, n_vars, n_vars)

"""Result from Arias et al. (2018) identification."""
struct AriasSVARResult{T<:AbstractFloat}
    Q_draws::Vector{Matrix{T}}
    irf_draws::Array{T,4}
    weights::Vector{T}
    acceptance_rate::T
    restrictions::SVARRestrictions
end

# --- MA Coefficients ---

"""Compute MA coefficients Φ_0, ..., Φ_horizon."""
function _compute_ma_coefficients(model::VARModel{T}, horizon::Int) where {T<:AbstractFloat}
    n, p = nvars(model), model.p
    A = extract_ar_coefficients(model.B, n, p)
    Phi = Vector{Matrix{T}}(undef, horizon + 1)
    Phi[1] = Matrix{T}(I, n, n)
    for h in 1:horizon
        Phi[h + 1] = sum(A[j] * Phi[h - j + 1] for j in 1:min(p, h); init=zeros(T, n, n))
    end
    Phi
end

"""Draw uniformly from O(n) via QR decomposition."""
function _draw_uniform_orthogonal(n::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    X = randn(T, n, n)
    F = qr(X)
    Q = Matrix(F.Q)
    R_diag = diag(F.R)
    for j in 1:n
        R_diag[j] < 0 && (Q[:, j] = -Q[:, j])
    end
    Q
end

"""Compute structural IRF for rotation Q."""
function _compute_irf_for_Q(model::VARModel{T}, Q::Matrix{T}, Phi::Vector{Matrix{T}},
                            L::LowerTriangular{T,Matrix{T}}, horizon::Int) where {T<:AbstractFloat}
    n = nvars(model)
    A0_inv = L * Q
    irf = zeros(T, horizon, n, n)
    for h in 1:horizon
        irf[h, :, :] = Phi[h] * A0_inv
    end
    irf
end

# --- Restriction Checking ---

"""Check if all zero restrictions are satisfied."""
_check_zero_restrictions(irf::Array{T,3}, r::SVARRestrictions; tol::T=T(1e-10)) where {T} =
    all(abs(irf[zr.horizon + 1, zr.variable, zr.shock]) <= tol for zr in r.zeros)

"""Check if all sign restrictions are satisfied."""
_check_sign_restrictions(irf::Array{T,3}, r::SVARRestrictions) where {T} =
    all(sr.sign > 0 ? irf[sr.horizon + 1, sr.variable, sr.shock] > 0 :
                      irf[sr.horizon + 1, sr.variable, sr.shock] < 0 for sr in r.signs)

# --- Zero Restriction Algorithm ---

"""Build constraint matrix for zero restrictions on shock j."""
_build_zero_constraint_matrix(r::SVARRestrictions, shock::Int, Phi::Vector{Matrix{T}},
                               L::LowerTriangular{T,Matrix{T}}) where {T} =
    [Vector{T}((Phi[zr.horizon + 1] * L)[zr.variable, :]) for zr in r.zeros if zr.shock == shock]

"""Draw unit vector from null space of constraints."""
function _draw_null_space_vector(constraints::Vector{Vector{T}}, n::Int) where {T<:AbstractFloat}
    isempty(constraints) && return (x = randn(T, n); x / norm(x))

    F = reduce(vcat, [c' for c in constraints])
    svd_result = svd(F, full=true)
    V = transpose(svd_result.Vt)
    tol = max(size(F)...) * eps(T) * (isempty(svd_result.S) ? one(T) : maximum(svd_result.S))
    rank_F = sum(svd_result.S .> tol)
    null_dim = n - rank_F
    null_dim <= 0 && error("Zero restrictions over-constrain shock")

    N = V[:, (rank_F + 1):n]
    z = randn(T, null_dim)
    q = N * z
    q / norm(q)
end

"""Draw orthogonal Q satisfying zero restrictions (Algorithm 2, Arias et al. 2018)."""
function _draw_Q_with_zero_restrictions(r::SVARRestrictions, Phi::Vector{Matrix{T}},
                                         L::LowerTriangular{T,Matrix{T}}) where {T<:AbstractFloat}
    n = r.n_vars
    Q = zeros(T, n, n)
    for j in 1:n
        zero_constraints = _build_zero_constraint_matrix(r, j, Phi, L)
        ortho_constraints = [Vector{T}(Q[:, k]) for k in 1:j-1]
        Q[:, j] = _draw_null_space_vector(vcat(zero_constraints, ortho_constraints), n)
    end
    @assert norm(Q' * Q - I) < 1e-10 "Q not orthogonal"
    Q
end

# --- Arias et al. (2018) Setup & Volume Element Helpers ---

"""Precomputed auxiliary data for Arias et al. (2018) importance weight computation."""
struct _AriasSVARSetup{T<:AbstractFloat}
    W::Vector{Matrix{T}}         # Auxiliary matrices W_j (s_j × n), j=1..n
    zeros_per_shock::Vector{Int}  # z_j per shock
    sphere_dims::Vector{Int}      # s_j = n - (j-1) - z_j per shock
    dim::Int                      # total sphere dimension = Σ s_j
end

function _AriasSVARSetup(restrictions::SVARRestrictions, n::Int, ::Type{T}) where {T<:AbstractFloat}
    zeros_per_shock = zeros(Int, n)
    for zr in restrictions.zeros
        zeros_per_shock[zr.shock] += 1
    end
    sphere_dims = [n - (j - 1) - zeros_per_shock[j] for j in 1:n]
    W = [randn(T, s, n) for s in sphere_dims]
    _AriasSVARSetup{T}(W, zeros_per_shock, sphere_dims, sum(sphere_dims))
end

"""Safe log|det(A)| — returns -Inf for singular matrices."""
function _log_abs_det(A::AbstractMatrix{T}) where {T}
    lad = logabsdet(A)
    isfinite(lad[1]) ? lad[1] : T(-Inf)
end

"""Central finite-difference Jacobian of f at x."""
function _numerical_jacobian(f, x::AbstractVector{T}; fd_eps::T=T(1e-7)) where {T}
    n = length(x)
    f0 = f(x)
    m = length(f0)
    J = zeros(T, m, n)
    xp = copy(x)
    xm = copy(x)
    for i in 1:n
        xp[i] = x[i] + fd_eps
        xm[i] = x[i] - fd_eps
        fp = f(xp)
        fm = f(xm)
        J[:, i] = (fp - fm) / (2 * fd_eps)
        xp[i] = x[i]
        xm[i] = x[i]
    end
    J
end

"""
Log volume element of f restricted to manifold {h(x)=0}.
Computes 0.5 * log|det(N'*N)| where N = Df * nullspace(Dh).
"""
function _log_volume_element(f, x::AbstractVector{T}, h) where {T}
    Df = _numerical_jacobian(f, x)
    Dh = _numerical_jacobian(h, x)

    # Null space of Dh (tangent space of constraint manifold)
    Ns = nullspace(Dh)
    size(Ns, 2) == 0 && return T(-Inf)

    # Project Jacobian onto tangent space
    N = Df * Ns
    G = N' * N
    T(0.5) * _log_abs_det(G)
end

"""Pack (A0, Aplus) into a single vector."""
_pack_structural(A0::Matrix{T}, Aplus::Matrix{T}) where {T} = vcat(vec(A0), vec(Aplus))

"""Unpack vector into (A0 n×n, Aplus m×n)."""
function _unpack_structural(x::AbstractVector{T}, n::Int, m::Int) where {T}
    A0 = reshape(x[1:n*n], n, n)
    Aplus = reshape(x[n*n+1:end], m, n)
    (A0, Aplus)
end

# --- Structural ↔ Reduced-Form Mappings ---

"""Structural → reduced-form: (A0, Aplus) → (B, Σ)."""
function _struct_to_rf(A0::Matrix{T}, Aplus::Matrix{T}) where {T}
    A0_inv = robust_inv(A0)
    B = Aplus * A0_inv
    Sigma = A0_inv * A0_inv'
    (B, Sigma)
end

"""Reduced-form → structural: (B, L, Q) → (A0, Aplus)."""
function _rf_to_struct(B::Matrix{T}, L::LowerTriangular{T,Matrix{T}}, Q::Matrix{T}) where {T}
    A0 = Matrix{T}(L') \ Q   # A0 = inv(L') * Q
    Aplus = B * A0
    (A0, Aplus)
end

# --- Sphere Coordinate Mappings ---

"""Compute zero-restriction constraint rows for shock j."""
function _compute_ZF(restrictions::SVARRestrictions, Phi::Vector{<:AbstractMatrix},
                     L::LowerTriangular, shock_j::Int)
    T = eltype(L)
    rows = Vector{Vector{T}}()
    for zr in restrictions.zeros
        zr.shock == shock_j || continue
        push!(rows, Vector{T}((Phi[zr.horizon + 1] * L)[zr.variable, :]))
    end
    isempty(rows) ? zeros(T, 0, size(L, 1)) : reduce(vcat, [r' for r in rows])
end

"""Convert orthogonal matrix Q to sphere coordinates w using setup's W matrices."""
function _Q_to_spheres(Q::AbstractMatrix, setup::_AriasSVARSetup, restrictions::SVARRestrictions,
                       Phi::Vector{<:AbstractMatrix}, L::LowerTriangular)
    T = eltype(Q)
    n = size(Q, 1)
    w_parts = Vector{Vector{T}}()

    for j in 1:n
        s_j = setup.sphere_dims[j]
        s_j <= 0 && continue

        # Build M_j: stack [Q[:,1:j-1]'; ZF_j; W_j]
        parts = Matrix{T}[]
        j > 1 && push!(parts, Matrix{T}(Q[:, 1:j-1]'))
        ZF_j = _compute_ZF(restrictions, Phi, L, j)
        size(ZF_j, 1) > 0 && push!(parts, Matrix{T}(ZF_j))
        push!(parts, Matrix{T}(setup.W[j]))

        M_j = vcat(parts...)
        # QR of M_j' with sign correction
        F = qr(M_j')
        K = Matrix{T}(F.Q)
        R_diag = diag(F.R)
        for col in 1:size(K, 2)
            R_diag[col] < 0 && (K[:, col] = -K[:, col])
        end

        # Last s_j columns form null-space basis
        K_j = K[:, end-s_j+1:end]
        w_j = K_j' * Q[:, j]
        push!(w_parts, w_j)
    end

    vcat(w_parts...)
end

"""Convert sphere coordinates w back to orthogonal matrix Q."""
function _spheres_to_Q(w::AbstractVector{T}, setup::_AriasSVARSetup, restrictions::SVARRestrictions,
                       Phi::Vector{<:AbstractMatrix}, L::LowerTriangular) where {T}
    n = length(setup.sphere_dims)
    Q = zeros(T, n, n)
    offset = 0

    for j in 1:n
        s_j = setup.sphere_dims[j]
        if s_j <= 0
            error("Zero restrictions over-constrain shock $j")
        end

        w_j = w[offset+1:offset+s_j]
        offset += s_j

        # Build same M_j as in _Q_to_spheres
        parts = Matrix{T}[]
        j > 1 && push!(parts, Matrix{T}(Q[:, 1:j-1]'))
        ZF_j = _compute_ZF(restrictions, Phi, L, j)
        size(ZF_j, 1) > 0 && push!(parts, Matrix{T}(ZF_j))
        push!(parts, Matrix{T}(setup.W[j]))

        M_j = vcat(parts...)
        F = qr(M_j')
        K = Matrix{T}(F.Q)
        R_diag = diag(F.R)
        for col in 1:size(K, 2)
            R_diag[col] < 0 && (K[:, col] = -K[:, col])
        end

        K_j = K[:, end-s_j+1:end]
        Q[:, j] = K_j * w_j
    end

    Q
end

"""Draw w from product of unit spheres S^{s_1-1} × ... × S^{s_n-1}."""
function _draw_w(setup::_AriasSVARSetup{T}) where {T}
    w_parts = Vector{T}()
    for s_j in setup.sphere_dims
        x = randn(T, s_j)
        append!(w_parts, x / norm(x))
    end
    Vector{T}(w_parts)
end

# --- Volume Element Closures ---

"""Compute MA coefficients from coefficient matrix B (non-VARModel version)."""
function _compute_ma_from_B(B::Matrix{T}, n::Int, p::Int, max_h::Int) where {T}
    A = extract_ar_coefficients(B, n, p)
    Phi = Vector{Matrix{T}}(undef, max_h + 1)
    Phi[1] = Matrix{T}(I, n, n)
    for h in 1:max_h
        Phi[h + 1] = sum(A[j] * Phi[h - j + 1] for j in 1:min(p, h); init=zeros(T, n, n))
    end
    Phi
end

"""
Build closure ff_h: structural_vec → (B, Σ, w) for volume element computation.
Maps structural parameters to reduced-form parameters + sphere coordinates.
"""
function _build_ff_h(setup::_AriasSVARSetup{T}, restrictions::SVARRestrictions,
                     n::Int, m::Int, p::Int, max_h::Int) where {T}
    function ff_h(x::AbstractVector)
        A0, Aplus = _unpack_structural(x, n, m)
        B_rf, Sigma_rf = _struct_to_rf(Matrix{T}(A0), Matrix{T}(Aplus))

        L_rf = safe_cholesky(Sigma_rf)
        Q_rf = Matrix{T}(L_rf') * A0

        Phi_rf = _compute_ma_from_B(Matrix{T}(B_rf), n, p, max_h)
        w_rf = _Q_to_spheres(Q_rf, setup, restrictions, Phi_rf, L_rf)

        vcat(vec(B_rf), _vech(Sigma_rf), w_rf)
    end
    ff_h
end

"""Extract lower triangle of symmetric matrix (vectorize unique elements)."""
function _vech(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    v = Vector{T}(undef, n * (n + 1) ÷ 2)
    k = 0
    for j in 1:n, i in j:n
        k += 1
        v[k] = A[i, j]
    end
    v
end

"""Build closure for zero restriction evaluation at structural params."""
function _build_zero_restrictions_fn(restrictions::SVARRestrictions, n::Int, m::Int, p::Int, max_h::Int,
                                      ::Type{T}=Float64) where {T}
    isempty(restrictions.zeros) && return x -> T[]

    function zero_fn(x::AbstractVector)
        A0, Aplus = _unpack_structural(x, n, m)
        A0_inv = robust_inv(Matrix{T}(A0))
        B_rf = Matrix{T}(Aplus) * A0_inv

        Phi = _compute_ma_from_B(B_rf, n, p, max_h)

        vals = Vector{T}(undef, length(restrictions.zeros))
        for (idx, zr) in enumerate(restrictions.zeros)
            vals[idx] = (Phi[zr.horizon + 1] * A0_inv)[zr.variable, zr.shock]
        end
        vals
    end
    zero_fn
end

# --- Importance Weights (Proposition 4, Arias et al. 2018) ---

"""
Compute draw-dependent importance weight for Q.

For zero+sign restrictions, the weight corrects for the non-uniform proposal
distribution on Q induced by the zero-restriction constraint manifold.
Uses the volume element formula from Proposition 4 of Arias et al. (2018, Econometrica).

The weight is: w = exp(log|v_e(f_h)| - log|v_e(ff_h|Z=0)|)
where f_h is the structural-to-reduced-form map and ff_h includes sphere coordinates.
"""
function _compute_importance_weight(Q::Matrix{T}, model::VARModel{T},
                                     setup::_AriasSVARSetup{T}, restrictions::SVARRestrictions,
                                     Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}}) where {T}
    isempty(restrictions.zeros) && return one(T)

    n = nvars(model)
    p = model.p
    m = size(model.B, 1)  # 1 + n*p

    # Structural params for this Q
    A0, Aplus = _rf_to_struct(model.B, L, Q)
    structpara = _pack_structural(A0, Aplus)

    max_h = isempty(restrictions.zeros) ? 0 : maximum(zr.horizon for zr in restrictions.zeros)

    # Analytical numerator: log|v_e(f_h)|
    # From Proposition 4: n(n+1)/2 * log(2) - (2n + m + 1) * log|det(A0)|
    log_ve_fh = T(n * (n + 1)) / 2 * log(T(2)) - T(2n + m + 1) * _log_abs_det(A0)

    # Numerical denominator: log|v_e(ff_h | Z=0)|
    ff_h = _build_ff_h(setup, restrictions, n, m, p, max_h)
    zero_fn = _build_zero_restrictions_fn(restrictions, n, m, p, max_h, T)
    log_ve_gfhZ = _log_volume_element(ff_h, structpara, zero_fn)

    # Guard against numerical issues
    log_w = log_ve_fh - log_ve_gfhZ
    isfinite(log_w) || return one(T)  # fallback for degenerate cases

    exp(log_w)
end

# Backward-compatible signature for pure sign restrictions (no setup needed)
function _compute_importance_weight(Q::Matrix{T}, r::SVARRestrictions,
                                     Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}}) where {T}
    isempty(r.zeros) && return one(T)
    # This path should not be reached for zero restrictions in the new code,
    # but kept for backward compatibility
    one(T)
end

# --- Main Arias Identification ---

"""
    identify_arias(model, restrictions, horizon; n_draws=1000, n_rotations=1000, compute_weights=true) -> AriasSVARResult

Identify SVAR using Arias et al. (2018) with zero and sign restrictions.

Uses importance sampling with draw-dependent weights (Proposition 4) for zero+sign restriction
combinations. For pure sign restrictions, draws uniformly from O(n) with unit weights.

# Keywords
- `n_draws::Int=1000`: Target number of accepted draws
- `n_rotations::Int=1000`: Maximum attempts per target draw
- `compute_weights::Bool=true`: Compute importance weights (set false for faster exploratory analysis)
"""
function identify_arias(model::VARModel{T}, restrictions::SVARRestrictions, horizon::Int;
                        n_draws::Int=1000, n_rotations::Int=1000,
                        compute_weights::Bool=true) where {T<:AbstractFloat}
    n = nvars(model)
    @assert restrictions.n_vars == n "Restriction dimension must match model"

    max_h = max(horizon,
        isempty(restrictions.zeros) ? 0 : maximum(zr.horizon for zr in restrictions.zeros) + 1,
        isempty(restrictions.signs) ? 0 : maximum(sr.horizon for sr in restrictions.signs) + 1)

    Phi, L = _compute_ma_coefficients(model, max_h), safe_cholesky(model.Sigma)
    Q_draws, irf_draws, weights = Matrix{T}[], Array{T,3}[], T[]
    has_zeros = !isempty(restrictions.zeros)
    n_attempts = 0

    # Create setup once for zero restrictions (W matrices fixed for all draws)
    setup = has_zeros ? _AriasSVARSetup(restrictions, n, T) : nothing

    while length(Q_draws) < n_draws && n_attempts < n_draws * n_rotations
        n_attempts += 1
        try
            if has_zeros
                Q = _draw_Q_with_zero_restrictions(restrictions, Phi, L)
            else
                Q = _draw_uniform_orthogonal(n, T)
            end
            irf = _compute_irf_for_Q(model, Q, Phi, L, horizon)

            (has_zeros && !_check_zero_restrictions(irf, restrictions)) && continue
            !_check_sign_restrictions(irf, restrictions) && continue

            push!(Q_draws, Q)
            push!(irf_draws, irf)
            if has_zeros && compute_weights
                push!(weights, _compute_importance_weight(Q, model, setup, restrictions, Phi, L))
            else
                push!(weights, one(T))
            end
        catch; continue; end
    end

    isempty(Q_draws) && error("No valid identification after $n_attempts attempts")

    n_acc = length(Q_draws)
    irf_array = zeros(T, n_acc, horizon, n, n)
    for (i, irf) in enumerate(irf_draws)
        irf_array[i, :, :, :] = irf
    end

    AriasSVARResult{T}(Q_draws, irf_array, weights ./ sum(weights), T(n_acc / n_attempts), restrictions)
end

# --- Bayesian Integration ---

"""
    identify_arias_bayesian(post::BVARPosterior, restrictions, horizon; data=nothing, n_rotations=100, quantiles=[0.16,0.5,0.84], compute_weights=true)

Apply Arias identification to each posterior draw. Returns IRF quantiles, mean, acceptance rates.

Creates the `_AriasSVARSetup` once (W matrices fixed across all posterior draws) for consistency.
"""
function identify_arias_bayesian(post::BVARPosterior, restrictions::SVARRestrictions, horizon::Int;
    data::Union{Nothing,AbstractMatrix}=nothing, n_rotations::Int=100,
    quantiles::Vector{Float64}=[0.16, 0.5, 0.84], compute_weights::Bool=true)

    use_data = isnothing(data) ? (isempty(post.data) ? nothing : post.data) : data
    p, n = post.p, post.n
    b_vecs, sigmas = extract_chain_parameters(post)
    n_samples = size(b_vecs, 1)
    all_irfs, all_weights = Vector{Array{Float64,3}}(), Float64[]
    acc_rates = zeros(n_samples)

    for s in 1:n_samples
        m = parameters_to_model(b_vecs[s,:], sigmas[s,:], p, n, use_data)
        try
            result = identify_arias(m, restrictions, horizon;
                n_draws=1, n_rotations=n_rotations, compute_weights=compute_weights)
            for (i, w) in enumerate(result.weights)
                push!(all_irfs, result.irf_draws[i, :, :, :])
                push!(all_weights, w)
            end
            acc_rates[s] = result.acceptance_rate
        catch
            acc_rates[s] = 0.0
        end
    end

    isempty(all_irfs) && error("No valid identifications across posterior")

    n_acc = length(all_irfs)
    irf_array = zeros(n_acc, horizon, n, n)
    for (i, irf) in enumerate(all_irfs)
        irf_array[i, :, :, :] = irf
    end
    w_norm = all_weights ./ sum(all_weights)

    irf_q = zeros(horizon, n, n, length(quantiles))
    irf_m = zeros(horizon, n, n)
    for h in 1:horizon, i in 1:n, j in 1:n
        vals = irf_array[:, h, i, j]
        irf_m[h, i, j] = sum(w_norm .* vals)
        for (qi, q) in enumerate(quantiles)
            irf_q[h, i, j, qi] = _weighted_quantile(vals, w_norm, q)
        end
    end

    (irf_quantiles=irf_q, irf_mean=irf_m, acceptance_rates=acc_rates, total_accepted=n_acc, weights=w_norm)
end

# Deprecated wrapper for old (chain, p, n, ...) signature
function identify_arias_bayesian(post::BVARPosterior, p::Int, n::Int, restrictions::SVARRestrictions, horizon::Int; kwargs...)
    identify_arias_bayesian(post, restrictions, horizon; kwargs...)
end

"""Weighted quantile via linear interpolation."""
function _weighted_quantile(vals::AbstractVector{T}, weights::AbstractVector{S}, q::Real) where {T,S}
    perm = sortperm(vals)
    sv, sw = vals[perm], weights[perm]
    cw = cumsum(sw)
    cw ./= cw[end]
    idx = searchsortedfirst(cw, q)
    idx == 1 && return sv[1]
    idx > length(sv) && return sv[end]
    t = (q - cw[idx-1]) / (cw[idx] - cw[idx-1] + eps())
    (1 - t) * sv[idx-1] + t * sv[idx]
end

# --- Convenience Functions ---

"""Create zero restriction: variable doesn't respond to shock at horizon."""
zero_restriction(variable::Int, shock::Int; horizon::Int=0) = ZeroRestriction(variable, shock, horizon)

"""Create sign restriction: variable response has given sign (:positive/:negative) at horizon."""
sign_restriction(variable::Int, shock::Int, sign::Symbol; horizon::Int=0) =
    SignRestriction(variable, shock, horizon, sign == :positive ? 1 : -1)

"""Compute weighted IRF percentiles from AriasSVARResult."""
function irf_percentiles(result::AriasSVARResult{T}; quantiles::Vector{Float64}=[0.16, 0.5, 0.84]) where {T}
    n_draws, horizon, n_vars, n_shocks = size(result.irf_draws)
    pct = zeros(T, horizon, n_vars, n_shocks, length(quantiles))
    for h in 1:horizon, i in 1:n_vars, j in 1:n_shocks
        for (pi, p) in enumerate(quantiles)
            pct[h, i, j, pi] = _weighted_quantile(result.irf_draws[:, h, i, j], result.weights, p)
        end
    end
    pct
end

"""Compute weighted mean IRF from AriasSVARResult."""
function irf_mean(result::AriasSVARResult{T}) where {T}
    n_draws, horizon, n_vars, n_shocks = size(result.irf_draws)
    mean_irf = zeros(T, horizon, n_vars, n_shocks)
    for h in 1:horizon, i in 1:n_vars, j in 1:n_shocks
        mean_irf[h, i, j] = sum(result.weights .* result.irf_draws[:, h, i, j])
    end
    mean_irf
end
