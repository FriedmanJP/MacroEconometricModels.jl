# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Arias, Rubio-Ramírez, Waggoner (2018) - Zero + Sign Restrictions
# =============================================================================

# Errors that legitimately indicate a *rejectable* random draw / failed optimization
# (numerical degeneracy), as opposed to a genuine bug (BoundsError, MethodError,
# DimensionMismatch, ...) which must propagate rather than be silently swallowed.
# Shared by the Arias and Uhlig identification loops.
_is_rejectable_draw_error(err) =
    err isa LinearAlgebra.SingularException ||
    err isa LinearAlgebra.PosDefException ||
    err isa LinearAlgebra.LAPACKException ||
    err isa DomainError

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

function SVARRestrictions(n_vars::Int; zeros=ZeroRestriction[], signs=SignRestriction[])
    n_vars > 0 || throw(ArgumentError("n_vars must be positive"))
    for zr in zeros
        (1 <= zr.variable <= n_vars && 1 <= zr.shock <= n_vars) ||
            throw(ArgumentError("zero restriction (var=$(zr.variable), shock=$(zr.shock)) out of 1:$n_vars"))
        zr.horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    end
    for sr in signs
        (1 <= sr.variable <= n_vars && 1 <= sr.shock <= n_vars) ||
            throw(ArgumentError("sign restriction (var=$(sr.variable), shock=$(sr.shock)) out of 1:$n_vars"))
        sr.horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
        sr.sign in (-1, 1) || throw(ArgumentError("sign must be -1 or +1"))
    end
    SVARRestrictions(zeros, signs, n_vars, n_vars)
end

"""
Result from Arias et al. (2018) identification.

`ess` is Kish's effective sample size of the importance weights and
`ess_fraction` is `ess / length(weights)`. Under pure sign restrictions the
weights are uniform and `ess_fraction == 1`; with zero restrictions a fraction
far below 1 means a handful of draws carry most of the posterior mass and the
weighted summaries rest on far fewer effective draws than `n_draws` suggests.
"""
struct AriasSVARResult{T<:AbstractFloat}
    Q_draws::Vector{Matrix{T}}
    irf_draws::Array{T,4}
    weights::Vector{T}
    acceptance_rate::T
    restrictions::SVARRestrictions
    ess::T
    ess_fraction::T
    varnames::Vector{String}
    n_degenerate_weights::Int
end

# Back-compatible arities: pre-ESS / pre-varnames / pre-degenerate construction sites.
function AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate,
                            restrictions) where {T<:AbstractFloat}
    ess = T(_effective_sample_size(weights))
    n = length(weights)
    nv = restrictions.n_vars
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions,
                       ess, n > 0 ? ess / T(n) : zero(T),
                       ["var$i" for i in 1:nv], 0)
end

function AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate,
                            restrictions, ess, ess_fraction) where {T<:AbstractFloat}
    nv = restrictions.n_vars
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions,
                       ess, ess_fraction, ["var$i" for i in 1:nv], 0)
end

function AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate,
                            restrictions, ess, ess_fraction, varnames) where {T<:AbstractFloat}
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions,
                       ess, ess_fraction, varnames, 0)
end

"""
Bayesian set-identified SVAR (Arias et al. 2018 applied to posterior draws).

`total_accepted` is `size(irf_draws, 1)`. `n_degenerate_weights` counts draws
skipped because the importance log-weight was non-finite.
"""
struct BayesianSetIdentifiedSVAR{T<:AbstractFloat}
    Q_draws::Vector{Matrix{T}}
    irf_draws::Array{T,4}
    weights::Vector{T}
    ess::T
    restrictions::SVARRestrictions
    varnames::Vector{String}
    n_unidentified::Int
    n_degenerate_weights::Int
    irf_quantiles::Array{T,4}
    irf_mean::Array{T,3}
    acceptance_rates::Vector{T}
    ess_fraction::T
end

function Base.getproperty(r::BayesianSetIdentifiedSVAR, s::Symbol)
    s === :total_accepted && return size(getfield(r, :irf_draws), 1)
    getfield(r, s)
end

function Base.propertynames(::BayesianSetIdentifiedSVAR, private::Bool=false)
    (fieldnames(BayesianSetIdentifiedSVAR)..., :total_accepted)
end

Base.haskey(::BayesianSetIdentifiedSVAR, s::Symbol) =
    s === :total_accepted || hasfield(BayesianSetIdentifiedSVAR, s)
Base.getindex(r::BayesianSetIdentifiedSVAR, s::Symbol) = getproperty(r, s)

"""
Fraction of the nominal draw count below which importance weights are reported
as degenerate. Matches the conventional SMC resampling trigger.
"""
const _ARIAS_ESS_WARN_FRACTION = 0.1

"""
    _warn_low_ess(ess, ess_fraction, n, label)

Emit the degeneracy warning when the effective sample size has collapsed. Silent
when the weights are uniform (`ess_fraction == 1`), which is the sign-only case.
"""
function _warn_low_ess(ess::Real, ess_fraction::Real, n::Integer, label::AbstractString)
    (n > 0 && ess_fraction < _ARIAS_ESS_WARN_FRACTION) || return nothing
    @warn "$label: importance weights are degenerate — effective sample size " *
          "$(round(ess; digits=1)) of $n draws " *
          "($(round(100 * ess_fraction; digits=1))% < " *
          "$(round(Int, 100 * _ARIAS_ESS_WARN_FRACTION))%). A few draws carry almost all " *
          "the posterior mass, so weighted IRF summaries are far less precise than the " *
          "draw count suggests. Consider loosening the zero/sign restrictions or raising n_draws."
    nothing
end

# --- MA Coefficients (wrappers around the shared kernels in irf.jl) ---

"""Compute MA coefficients Φ_0, ..., Φ_horizon (length horizon+1)."""
function _compute_ma_coefficients(model::VARModel{T}, horizon::Int) where {T<:AbstractFloat}
    ma_coefficients(model.B, nvars(model), model.p, horizon + 1)
end

"""Compute structural IRF for rotation Q."""
function _compute_irf_for_Q(model::VARModel{T}, Q::Matrix{T}, Phi::Vector{Matrix{T}},
                            L::LowerTriangular{T,Matrix{T}}, horizon::Int) where {T<:AbstractFloat}
    structural_irf(Phi, L, Q, horizon)
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
function _draw_null_space_vector(constraints::Vector{Vector{T}}, n::Int; rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    isempty(constraints) && return (x = randn(rng, T, n); x / norm(x))

    F = reduce(vcat, [c' for c in constraints])
    svd_result = svd(F, full=true)
    V = transpose(svd_result.Vt)
    tol = max(size(F)...) * eps(T) * (isempty(svd_result.S) ? one(T) : maximum(svd_result.S))
    rank_F = sum(svd_result.S .> tol)
    null_dim = n - rank_F
    null_dim <= 0 && error("Zero restrictions over-constrain shock")

    N = V[:, (rank_F + 1):n]
    z = randn(rng, T, null_dim)
    q = N * z
    q / norm(q)
end

"""Draw orthogonal Q satisfying zero restrictions (Algorithm 2, Arias et al. 2018)."""
function _draw_Q_with_zero_restrictions(r::SVARRestrictions, Phi::Vector{Matrix{T}},
                                         L::LowerTriangular{T,Matrix{T}};
                                         rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = r.n_vars
    Q = zeros(T, n, n)
    for j in 1:n
        zero_constraints = _build_zero_constraint_matrix(r, j, Phi, L)
        ortho_constraints = [Vector{T}(Q[:, k]) for k in 1:j-1]
        Q[:, j] = _draw_null_space_vector(vcat(zero_constraints, ortho_constraints), n; rng=rng)
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

function _AriasSVARSetup(restrictions::SVARRestrictions, n::Int, ::Type{T};
                        rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    zeros_per_shock = zeros(Int, n)
    for zr in restrictions.zeros
        zeros_per_shock[zr.shock] += 1
    end
    sphere_dims = [n - (j - 1) - zeros_per_shock[j] for j in 1:n]
    W = [randn(rng, T, s, n) for s in sphere_dims]
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

"""
Compute QR sign patterns for each shock's M_j matrix.
Returns Vector{Vector{Int}} where signs[j][col] is +1 or -1.
Used to fix the sign convention across finite-difference perturbations.
"""
function _compute_qr_signs(Q::AbstractMatrix{T}, setup::_AriasSVARSetup,
                           restrictions::SVARRestrictions,
                           Phi::Vector{<:AbstractMatrix}, L::LowerTriangular) where {T}
    n = size(Q, 1)
    signs = Vector{Vector{Int}}(undef, n)

    for j in 1:n
        parts = Matrix{T}[]
        j > 1 && push!(parts, Matrix{T}(Q[:, 1:j-1]'))
        ZF_j = _compute_ZF(restrictions, Phi, L, j)
        size(ZF_j, 1) > 0 && push!(parts, Matrix{T}(ZF_j))
        push!(parts, Matrix{T}(setup.W[j]))

        M_j = vcat(parts...)
        F = qr(M_j')
        R_diag = diag(F.R)
        signs[j] = [R_diag[col] < zero(T) ? -1 : 1 for col in 1:length(R_diag)]
    end

    signs
end

"""Convert orthogonal matrix Q to sphere coordinates w using setup's W matrices.

When `ref_signs` is provided, uses the fixed sign convention from a reference point
instead of re-evaluating `sign(diag(R))`. This eliminates discontinuities in the
QR sign correction that cause unreliable finite-difference Jacobians.
"""
function _Q_to_spheres(Q::AbstractMatrix, setup::_AriasSVARSetup, restrictions::SVARRestrictions,
                       Phi::Vector{<:AbstractMatrix}, L::LowerTriangular;
                       ref_signs::Union{Nothing, Vector{Vector{Int}}}=nothing)
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
        if ref_signs !== nothing
            # Use reference signs to avoid discontinuity across finite differences
            for col in 1:size(K, 2)
                ref_signs[j][col] < 0 && (K[:, col] = -K[:, col])
            end
        else
            for col in 1:size(K, 2)
                R_diag[col] < 0 && (K[:, col] = -K[:, col])
            end
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
function _draw_w(setup::_AriasSVARSetup{T}; rng::AbstractRNG=Random.default_rng()) where {T}
    w_parts = Vector{T}()
    for s_j in setup.sphere_dims
        x = randn(rng, T, s_j)
        append!(w_parts, x / norm(x))
    end
    Vector{T}(w_parts)
end

# --- Volume Element Closures ---

"""
Build closure ff_h: structural_vec → (B, Σ, w) for volume element computation.
Maps structural parameters to reduced-form parameters + sphere coordinates.

Captures reference QR sign patterns on first evaluation to ensure the function
is smooth for numerical differentiation. Without this, the QR sign correction
`R_diag[col] < 0 → flip` creates discontinuities that make finite-difference
Jacobians unreliable (Issue #37).
"""
function _build_ff_h(setup::_AriasSVARSetup{T}, restrictions::SVARRestrictions,
                     n::Int, m::Int, p::Int, max_h::Int) where {T}
    ref_signs_storage = Ref{Union{Nothing, Vector{Vector{Int}}}}(nothing)
    first_call = Ref(true)

    function ff_h(x::AbstractVector)
        A0, Aplus = _unpack_structural(x, n, m)
        B_rf, Sigma_rf = _struct_to_rf(Matrix{T}(A0), Matrix{T}(Aplus))

        L_rf = safe_cholesky(Sigma_rf)
        Q_rf = Matrix{T}(L_rf') * A0

        Phi_rf = ma_coefficients(Matrix{T}(B_rf), n, p, max_h + 1)

        if first_call[]
            # Record the sign pattern at the reference point
            ref_signs_storage[] = _compute_qr_signs(Q_rf, setup, restrictions, Phi_rf, L_rf)
            first_call[] = false
        end

        w_rf = _Q_to_spheres(Q_rf, setup, restrictions, Phi_rf, L_rf;
                              ref_signs=ref_signs_storage[])

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

        Phi = ma_coefficients(B_rf, n, p, max_h + 1)

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

    # Guard against numerical issues: caller skips non-finite log-weights
    # (do not substitute unit weight — SID-19).
    log_w = log_ve_fh - log_ve_gfhZ
    isfinite(log_w) || return T(NaN)

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

The result reports Kish's **effective sample size** of those weights (`ess`,
`ess_fraction`). Uneven weights mean the weighted IRF summaries rest on fewer
effective draws than `n_draws`; below 10% of the nominal count the sampler is
reported as degenerate and a warning is emitted.

# Keywords
- `n_draws::Int=1000`: Target number of accepted draws
- `n_rotations::Int=1000`: Maximum attempts per target draw
- `compute_weights::Bool=true`: Compute importance weights (set false for faster exploratory analysis)
- `normalize_weights::Bool=true`: Scale the stored weights to sum to 1. Pass `false`
  to keep them on the raw volume-element scale, which is required when pooling
  across draws of `(B, Σ)` — see [`identify_arias_bayesian`](@ref). `ess` is
  scale-invariant and unaffected either way.
- `rng::AbstractRNG`: Random number generator (thread through for reproducible `ess`)
"""
function identify_arias(model::VARModel{T}, restrictions::SVARRestrictions, horizon::Int;
                        n_draws::Int=1000, n_rotations::Int=1000,
                        compute_weights::Bool=true,
                        normalize_weights::Bool=true,
                        setup::Union{Nothing,_AriasSVARSetup}=nothing,
                        rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(model)
    @assert restrictions.n_vars == n "Restriction dimension must match model"

    max_h = max(horizon,
        isempty(restrictions.zeros) ? 0 : maximum(zr.horizon for zr in restrictions.zeros) + 1,
        isempty(restrictions.signs) ? 0 : maximum(sr.horizon for sr in restrictions.signs) + 1)

    Phi, L = _compute_ma_coefficients(model, max_h), safe_cholesky(model.Sigma)
    Q_draws, irf_draws, weights = Matrix{T}[], Array{T,3}[], T[]
    has_zeros = !isempty(restrictions.zeros)
    n_attempts = 0
    n_degenerate = 0
    last_err = nothing

    # Create setup once for zero restrictions (W matrices fixed for all draws),
    # or reuse a caller-supplied setup (Bayesian pooling).
    if has_zeros && setup === nothing
        setup = _AriasSVARSetup(restrictions, n, T; rng=rng)
    end

    while length(Q_draws) < n_draws && n_attempts < n_draws * n_rotations
        n_attempts += 1
        try
            if has_zeros
                Q = _draw_Q_with_zero_restrictions(restrictions, Phi, L; rng=rng)
            else
                Q = haar_orthogonal(n, T; rng=rng)
            end
            irf_full = _compute_irf_for_Q(model, Q, Phi, L, max_h)
            (has_zeros && !_check_zero_restrictions(irf_full, restrictions)) && continue
            !_check_sign_restrictions(irf_full, restrictions) && continue
            irf = irf_full[1:horizon, :, :]

            w = if has_zeros && compute_weights
                _compute_importance_weight(Q, model, setup, restrictions, Phi, L)
            else
                one(T)
            end
            if !isfinite(w)
                n_degenerate += 1
                continue
            end

            push!(Q_draws, Q)
            push!(irf_draws, irf)
            push!(weights, w)
        catch err
            _is_rejectable_draw_error(err) || rethrow(err)
            last_err = err
            continue
        end
    end

    isempty(Q_draws) && throw(IdentificationError("No valid identification after $n_attempts attempts" *
        (last_err === nothing ? "" :
         "; last rejectable failure: $(typeof(last_err)): $(sprint(showerror, last_err))")))

    n_acc = length(Q_draws)
    irf_array = zeros(T, n_acc, horizon, n, n)
    for (i, irf) in enumerate(irf_draws)
        irf_array[i, :, :, :] = irf
    end

    n_degenerate > 0 && @warn "identify_arias: $n_degenerate draw(s) skipped because the importance log-weight was non-finite"

    # ESS is scale-invariant, so it is identical before and after normalization.
    ess = T(_effective_sample_size(weights))
    ess_frac = ess / T(n_acc)
    _warn_low_ess(ess, ess_frac, n_acc, "identify_arias")

    w_out = normalize_weights ? weights ./ sum(weights) : weights
    vnames = copy(model.varnames)
    AriasSVARResult{T}(Q_draws, irf_array, w_out, T(n_acc / n_attempts), restrictions,
                       ess, ess_frac, vnames, n_degenerate)
end

# --- Bayesian Integration ---

"""
    identify_arias_bayesian(post::BVARPosterior, restrictions, horizon; data=nothing, n_rotations=100, quantiles=[0.16,0.5,0.84], compute_weights=true)

Apply Arias identification to each posterior draw.

Creates the `_AriasSVARSetup` once (W matrices fixed across all posterior draws) for consistency.

Importance weights are pooled across posterior draws on the **raw volume-element
scale** and normalized once at the end. Each per-draw call accepts a single
rotation, so normalizing there would force every weight to 1 and reduce the
weighted summaries to unweighted ones.

# Returns
[`BayesianSetIdentifiedSVAR`](@ref) with draws, weights, ESS, `n_unidentified`,
and `n_degenerate_weights`. Property `total_accepted` is `size(irf_draws, 1)`.
`irf_quantiles` / `irf_mean` / `acceptance_rates` are stored for back-compat
with the previous NamedTuple return.
"""
function identify_arias_bayesian(post::BVARPosterior, restrictions::SVARRestrictions, horizon::Int;
    data::Union{Nothing,AbstractMatrix}=nothing, n_rotations::Int=100,
    quantiles::Vector{Float64}=[0.16, 0.5, 0.84], compute_weights::Bool=true,
    rng::AbstractRNG=Random.default_rng())

    use_data = isnothing(data) ? (isempty(post.data) ? nothing : post.data) : data
    p, n = post.p, post.n
    T = eltype(post.Sigma_draws)
    b_vecs, sigmas = extract_chain_parameters(post)
    n_samples = size(b_vecs, 1)
    all_irfs, all_weights = Vector{Array{T,3}}(), T[]
    all_Qs = Matrix{T}[]
    acc_rates = zeros(T, n_samples)
    n_unidentified = 0
    n_degenerate = 0

    has_zeros = !isempty(restrictions.zeros)
    setup = has_zeros ? _AriasSVARSetup(restrictions, n, T; rng=rng) : nothing

    for s in 1:n_samples
        m = parameters_to_model(b_vecs[s,:], sigmas[s,:], p, n, use_data)
        try
            # normalize_weights=false is essential: each inner call accepts a
            # SINGLE draw, so normalizing there would set every weight to 1 and
            # silently discard the importance correction. Pooling happens once,
            # below, across all posterior draws on the common volume-element scale.
            result = identify_arias(m, restrictions, horizon;
                n_draws=1, n_rotations=n_rotations, compute_weights=compute_weights,
                normalize_weights=false, setup=setup, rng=rng)
            n_degenerate += result.n_degenerate_weights
            for (i, w) in enumerate(result.weights)
                push!(all_irfs, result.irf_draws[i, :, :, :])
                push!(all_weights, w)
                push!(all_Qs, result.Q_draws[i])
            end
            acc_rates[s] = result.acceptance_rate
        catch err
            if err isa IdentificationError
                n_unidentified += 1
                acc_rates[s] = 0
                continue
            end
            _is_rejectable_draw_error(err) || rethrow(err)
            acc_rates[s] = 0
        end
    end

    isempty(all_irfs) && throw(IdentificationError(
        "No valid identifications across posterior " *
        "(unidentified=$n_unidentified, n_samples=$n_samples)"))
    frac_u = n_unidentified / n_samples
    frac_u > 0.5 && @warn "$n_unidentified/$n_samples posterior draws unidentified"
    n_degenerate > 0 && @warn "identify_arias_bayesian: $n_degenerate draw(s) skipped because the importance log-weight was non-finite"

    n_acc = length(all_irfs)
    irf_array = zeros(T, n_acc, horizon, n, n)
    for (i, irf) in enumerate(all_irfs)
        irf_array[i, :, :, :] = irf
    end
    ess = T(_effective_sample_size(all_weights))
    ess_frac = ess / T(n_acc)
    _warn_low_ess(ess, ess_frac, n_acc, "identify_arias_bayesian")
    w_norm = all_weights ./ sum(all_weights)

    irf_q = zeros(T, horizon, n, n, length(quantiles))
    irf_m = zeros(T, horizon, n, n)
    for h in 1:horizon, i in 1:n, j in 1:n
        vals = irf_array[:, h, i, j]
        irf_m[h, i, j] = sum(w_norm .* vals)
        for (qi, q) in enumerate(quantiles)
            irf_q[h, i, j, qi] = _weighted_quantile(vals, w_norm, q)
        end
    end

    vnames = copy(post.varnames)
    BayesianSetIdentifiedSVAR{T}(all_Qs, irf_array, Vector{T}(w_norm), ess, restrictions,
                                 vnames, n_unidentified, n_degenerate,
                                 irf_q, irf_m, acc_rates, ess_frac)
end

# Deprecated wrapper for old (chain, p, n, ...) signature
function identify_arias_bayesian(post::BVARPosterior, p::Int, n::Int, restrictions::SVARRestrictions, horizon::Int; kwargs...)
    identify_arias_bayesian(post, restrictions, horizon; kwargs...)
end

"""Parse Arias kwargs from `irf`/`fevd` on `BVARPosterior` (extra keys ignored)."""
function _arias_posterior_kwargs(max_draws::Int; rng::AbstractRNG=Random.default_rng(),
                                 n_rotations::Union{Nothing,Integer}=nothing,
                                 compute_weights::Bool=true, kwargs...)
    (rng, Int(something(n_rotations, max_draws)), compute_weights)
end

"""Peel `n_draws`/`n_rotations` before splatting into `identify_arias` (frequentist `irf`/`fevd`/`hd`/`compute_Q`).

`default_n_draws` is used when the caller does not pass `n_draws` (`compute_Q` wants 1).
"""
function _arias_freq_kwargs(max_draws::Int; rng::AbstractRNG=Random.default_rng(),
                            n_rotations::Union{Nothing,Integer}=nothing,
                            n_draws::Union{Nothing,Integer}=nothing,
                            default_n_draws::Integer=max_draws,
                            kwargs...)
    (; n_draws=Int(something(n_draws, default_n_draws)),
       n_rotations=Int(something(n_rotations, max_draws)),
       rng, kwargs...)
end

"""Weighted Arias identified-set from a BVAR posterior (`irf`/`fevd` `method=:arias`)."""
function _arias_from_bvar_posterior(post::BVARPosterior, restrictions, horizon;
                                    data=nothing, n_rotations::Int=100,
                                    quantiles::AbstractVector=[0.16, 0.5, 0.84],
                                    rng::AbstractRNG=Random.default_rng(),
                                    compute_weights::Bool=true)
    isnothing(restrictions) && throw(ArgumentError("arias requires restrictions"))
    restrictions isa SVARRestrictions || throw(ArgumentError(
        ":arias requires restrictions::SVARRestrictions"))
    identify_arias_bayesian(post, restrictions, horizon;
        data=data, n_rotations=n_rotations, quantiles=collect(Float64.(quantiles)),
        compute_weights=compute_weights, rng=rng)
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
    t = (q - cw[idx-1]) / (cw[idx] - cw[idx-1] + eps(T))
    (1 - t) * sv[idx-1] + t * sv[idx]
end

# --- Convenience Functions ---

"""Create zero restriction: variable doesn't respond to shock at horizon."""
function zero_restriction(variable::Int, shock::Int; horizon::Int=0)
    horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    ZeroRestriction(variable, shock, horizon)
end

"""Create sign restriction: variable response has given sign (:positive/:negative) at horizon."""
function sign_restriction(variable::Int, shock::Int, sign::Symbol; horizon::Int=0)
    horizon >= 0 || throw(ArgumentError("restriction horizon must be ≥ 0"))
    s = sign === :positive ? 1 : sign === :negative ? -1 :
        throw(ArgumentError("sign must be :positive or :negative"))
    SignRestriction(variable, shock, horizon, s)
end

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
