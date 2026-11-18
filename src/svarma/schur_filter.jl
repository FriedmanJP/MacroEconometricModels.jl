# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Real Schur forward/backward shock filter and exact sample-average log-Jacobian
for Strong Structural VARMA(p, q) models (Gouriéroux, Monfort & Renne 2020, §4.1.2, §4.1.4).
"""

using LinearAlgebra

@inline _svarma_logabsdet(M::AbstractMatrix{T}) where {T<:AbstractFloat} = T(first(logabsdet(M)))
@inline _svarma_logabsdet(M::AbstractMatrix{<:Real}) = log(abs(det(M)))

function _svarma_check_C_nonsingular(C::AbstractMatrix)
    n = size(C, 1)
    size(C, 2) == n || throw(ArgumentError("C must be square, got $(size(C))"))
    T_r = typeof(float(real(zero(eltype(C)))))
    det_C = det(C)
    nrm = try
        opnorm(C)
    catch
        one(T_r)
    end
    scale_ref = isfinite(nrm) ? max(nrm^n, one(T_r)) : one(T_r)
    if !isfinite(det_C) || abs(det_C) <= T_r(1e-13) * scale_ref
        throw(SingularSystemError("Structural impact matrix C is singular or near-singular (|det(C)| = $(abs(det_C)))"))
    end
    return nothing
end

function _svarma_schur_partition(Theta_3d::Array{T,3}, tol::Real) where {T<:AbstractFloat}
    n, _, q = size(Theta_3d)
    F_ma = _svarma_companion(Theta_3d)
    F_schur = schur(Matrix{T}(F_ma))
    eigs = Complex{T}.(F_schur.values)
    tol_T = max(T(tol), T(16) * eps(T))
    for lam in eigs
        if abs(abs(lam) - one(T)) <= tol_T
            throw(IdentificationError(
                "MA companion matrix has eigenvalue on or near the unit circle " *
                "(||λ| - 1| = $(abs(abs(lam) - one(T))) <= tol=$tol_T); " *
                "GMR (2020) Assumption A.5 requires no unit roots."))
        end
    end
    sel = abs.(eigs) .< one(T)
    F_ord = ordschur(F_schur, sel)
    Z = Matrix{T}(F_ord.Z)
    T_mat = Matrix{T}(F_ord.T)
    eigs_ord = Complex{T}.(F_ord.values)
    m1 = count(sel)
    m2 = n * q - m1
    return (Z=Z, T_mat=T_mat, eigs=eigs_ord, m1=m1, m2=m2, n=n, q=q)
end

function _svarma_filter_core(U::AbstractMatrix, Theta_3d::Array{<:Real,3},
                             C::AbstractMatrix; tol::Real=1e-8)
    T_eff, n = size(U)
    T_eff >= 1 || throw(ArgumentError("Need at least 1 effective residual observation, got $T_eff"))
    size(Theta_3d, 1) == n && size(Theta_3d, 2) == n ||
        throw(ArgumentError("Theta dimensions $(size(Theta_3d)) incompatible with n=$n"))
    _svarma_check_C_nonsingular(C)

    T_ma = float(eltype(Theta_3d))
    T_work = float(promote_type(eltype(U), T_ma, eltype(C)))
    q = size(Theta_3d, 3)

    C_inv = robust_inv(C; silent=true)
    ld_C = T_work(_svarma_logabsdet(C))

    if q == 0
        eps_mat = Matrix{T_work}(U)
        eta = eps_mat * transpose(C_inv)
        return (
            shocks = eta,
            errors = eps_mat,
            residuals = Matrix{T_work}(U),
            log_jacobian = ld_C,
            log_det_C = ld_C,
            log_det_T22 = zero(T_work),
            eigenvalues = Complex{T_ma}[],
            n_fundamental = 0,
            n_nonfundamental = 0,
        )
    end

    part = _svarma_schur_partition(Array{T_ma,3}(Theta_3d), tol)
    Z, T_schur, eigs, m1, m2 = part.Z, part.T_mat, part.eigs, part.m1, part.m2
    nq = n * q

    Z1_top = @view Z[1:n, 1:m1]
    Z2_top = @view Z[1:n, (m1 + 1):nq]
    T11 = @view T_schur[1:m1, 1:m1]
    T12 = @view T_schur[1:m1, (m1 + 1):nq]
    T22 = T_schur[(m1 + 1):nq, (m1 + 1):nq]

    Ut1 = U * Z1_top
    Ut2 = U * Z2_top

    # Block 2: backward recursion from terminal condition ε̃_{2, T_eff} = 0
    E2 = zeros(T_work, m2, T_eff)
    E2_0 = zeros(T_work, m2)
    ld_T22 = zero(T_work)
    if m2 > 0
        T22_inv = Matrix{T_ma}(robust_inv(T22; silent=true))
        @inbounds for t in (T_eff - 1):-1:1
            E2[:, t] = T22_inv * (@view(E2[:, t + 1]) .- @view(Ut2[t + 1, :]))
        end
        E2_0 = T22_inv * (@view(E2[:, 1]) .- @view(Ut2[1, :]))
        ld_T22 = T_work(_svarma_logabsdet(T22))
    end

    # Block 1: forward recursion from initial condition ε̃_{1, 0} = 0
    E1 = zeros(T_work, m1, T_eff)
    if m1 > 0
        e1_prev = zeros(T_work, m1)
        @inbounds for t in 1:T_eff
            e1_curr = @view(Ut1[t, :]) .+ T11 * e1_prev
            if m2 > 0
                e2_prev = t == 1 ? E2_0 : @view(E2[:, t - 1])
                e1_curr = e1_curr .+ T12 * e2_prev
            end
            E1[:, t] = e1_curr
            e1_prev = e1_curr
        end
    end

    eps_mat = zeros(T_work, T_eff, n)
    if m1 > 0
        eps_mat .+= transpose(E1) * transpose(Z1_top)
    end
    if m2 > 0
        eps_mat .+= transpose(E2) * transpose(Z2_top)
    end
    eta = eps_mat * transpose(C_inv)

    return (
        shocks = eta,
        errors = eps_mat,
        residuals = Matrix{T_work}(U),
        log_jacobian = ld_C + ld_T22,
        log_det_C = ld_C,
        log_det_T22 = ld_T22,
        eigenvalues = eigs,
        n_fundamental = m1,
        n_nonfundamental = m2,
    )
end

# Exact finite-sample augmented Schur bijection:
# maps (z_init, U, z_term) ∈ ℝ^{m1} × ℝ^{T_eff × n} × ℝ^{m2} bijectively to
# (eps_pre, shocks) ∈ ℝ^{q × n} × ℝ^{T_eff × n} where Ẽ_{1,0} = z_init and Ẽ_{2,T_eff} = z_term.
function _svarma_filter_augmented(U::AbstractMatrix, Theta_3d::Array{<:Real,3},
                                  C::AbstractMatrix, z_init::AbstractVector,
                                  z_term::AbstractVector; tol::Real=1e-8)
    T_eff, n = size(U)
    q = size(Theta_3d, 3)
    T_ma = float(eltype(Theta_3d))
    T_work = float(promote_type(eltype(U), T_ma, eltype(C), eltype(z_init), eltype(z_term)))
    C_inv = robust_inv(C; silent=true)

    if q == 0
        eta = Matrix{T_work}(U) * transpose(C_inv)
        return (eps_pre=zeros(T_work, 0, n), shocks=eta)
    end

    part = _svarma_schur_partition(Array{T_ma,3}(Theta_3d), tol)
    Z, T_schur, m1, m2 = part.Z, part.T_mat, part.m1, part.m2
    nq = n * q
    length(z_init) == m1 || throw(ArgumentError("z_init must have length $m1"))
    length(z_term) == m2 || throw(ArgumentError("z_term must have length $m2"))

    Z1_top = @view Z[1:n, 1:m1]
    Z2_top = @view Z[1:n, (m1 + 1):nq]
    T11 = @view T_schur[1:m1, 1:m1]
    T12 = @view T_schur[1:m1, (m1 + 1):nq]
    T22 = T_schur[(m1 + 1):nq, (m1 + 1):nq]

    Ut1 = U * Z1_top
    Ut2 = U * Z2_top

    E2 = zeros(T_work, m2, T_eff)
    E2_0 = zeros(T_work, m2)
    if m2 > 0
        T22_inv = Matrix{T_ma}(robust_inv(T22; silent=true))
        E2[:, T_eff] .= z_term
        @inbounds for t in (T_eff - 1):-1:1
            E2[:, t] = T22_inv * (@view(E2[:, t + 1]) .- @view(Ut2[t + 1, :]))
        end
        E2_0 = T22_inv * (@view(E2[:, 1]) .- @view(Ut2[1, :]))
    end

    E1 = zeros(T_work, m1, T_eff)
    if m1 > 0
        e1_prev = Vector{T_work}(z_init)
        @inbounds for t in 1:T_eff
            e1_curr = @view(Ut1[t, :]) .+ T11 * e1_prev
            if m2 > 0
                e2_prev = t == 1 ? E2_0 : @view(E2[:, t - 1])
                e1_curr = e1_curr .+ T12 * e2_prev
            end
            E1[:, t] = e1_curr
            e1_prev = e1_curr
        end
    end

    E0_vec = zeros(T_work, nq)
    if m1 > 0
        E0_vec .+= @view(Z[:, 1:m1]) * Vector{T_work}(z_init)
    end
    if m2 > 0
        E0_vec .+= @view(Z[:, (m1 + 1):nq]) * E2_0
    end
    eps_pre = Matrix{T_work}(undef, q, n)
    @inbounds for j in 1:q
        # E0 = [ε_0; ε_{-1}; ...; ε_{-q+1}] -> store chronologically [-q+1 .. 0]
        src_start = (q - j) * n + 1
        eps_pre[j, :] .= @view(E0_vec[src_start:(src_start + n - 1)])
    end

    eps_mat = zeros(T_work, T_eff, n)
    if m1 > 0
        eps_mat .+= transpose(E1) * transpose(Z1_top)
    end
    if m2 > 0
        eps_mat .+= transpose(E2) * transpose(Z2_top)
    end
    eta = eps_mat * transpose(C_inv)
    return (eps_pre=eps_pre, shocks=eta)
end

@inline function _svarma_lag_type(M, ::Type{T_fallback}) where {T_fallback}
    if M === nothing || (M isa AbstractVector && isempty(M))
        return T_fallback <: AbstractFloat ? T_fallback : Float64
    end
    return float(_svarma_eltype(M))
end

"""
    ssvarma_filter_shocks(Y, mu, Phi, Theta, C; tol=1e-8) -> NamedTuple
    ssvarma_filter_shocks(Y, Phi, Theta, C; mu=nothing, tol=1e-8) -> NamedTuple
    ssvarma_filter_shocks(U, Theta, C; tol=1e-8) -> NamedTuple
    ssvarma_filter_shocks(m::SVARMAModel; tol=1e-8) -> NamedTuple

Recover structural shocks ``\\hat{\\eta}_t`` and sample-average log-Jacobian across
arbitrary fundamental, mixed, and purely non-fundamental SSVARMA(p, q) regimes via the
real Schur forward/backward algorithm of Gouriéroux, Monfort & Renne (2020, §4.1.2, §4.1.4).

Given AR-filtered reduced-form residuals ``u_t = Y_t - \\mu - \\sum_{k=1}^p \\Phi_k Y_{t-k}``
satisfying the moving-average system ``u_t = \\varepsilon_t + \\sum_{j=1}^q \\Theta_j \\varepsilon_{t-j}``
with ``\\varepsilon_t = C \\eta_t``, the companion matrix ``F_{\\text{ma}}`` (``-\\Theta_1``
for ``q = 1``) is factored via real Schur decomposition and reordered via `ordschur`:

```math
F_{\\text{ma}} = Z T Z', \\qquad T = \\begin{bmatrix} T_{11} & T_{12} \\\\ 0 & T_{22} \\end{bmatrix}, \\qquad \\tilde{u}_t = Z_{1:n, :}' u_t
```

where:
- `F_{\\text{ma}}` is the ``nq \\times nq`` moving-average companion matrix
- `Z` is the ``nq \\times nq`` orthogonal Schur basis matrix partitioned into causal (`|\\lambda| < 1`) and anti-causal (`|\\lambda| > 1`) blocks, with top ``n`` rows `Z_{1:n, :}`
- `T` is the ``nq \\times nq`` reordered real Schur form
- `T_{11}` is the ``m_1 \\times m_1`` upper quasi-triangular causal block with all eigenvalues strictly inside the unit circle
- `T_{22}` is the ``m_2 \\times m_2`` upper quasi-triangular anti-causal block with all eigenvalues strictly outside the unit circle
- `T_{12}` is the ``m_1 \\times m_2`` cross-coupling block
- `u_t` is the ``n \\times 1`` AR-filtered reduced-form residual at time ``t``
- `\\tilde{u}_t` is the rotated residual vector in Schur coordinates

Block 2 is solved backward from terminal condition ``\\tilde{\\varepsilon}_{2, T_{\\text{eff}}} = 0``,
block 1 is solved forward from ``\\tilde{\\varepsilon}_{1, 0} = 0``, and structural shocks are
recovered as ``\\hat{\\eta}_t = C^{-1} (Z \\tilde{\\varepsilon}_t)_{1:n}``.
Per GMR (2020, Prop. 2), the exact sample-average log-Jacobian is:

```math
T_{\\text{eff}}^{-1} \\log |\\det(\\partial u / \\partial \\eta)| = \\log |\\det(C)| + \\sum_{| \\lambda_j | > 1} \\log | \\lambda_j |
```

where:
- `T_{\\text{eff}}` is the effective sample size (`T_obs - p`)
- `u` and `\\eta` denote the stacked reduced-form residuals and structural shocks
- `C` is the ``n \\times n`` structural impact matrix
- `\\lambda_j` are the eigenvalues of the MA companion matrix ``F_{\\text{ma}}``

Returns a `NamedTuple` `(shocks, errors, residuals, log_jacobian, log_det_C, log_det_T22, eigenvalues, n_fundamental, n_nonfundamental)`.
"""
function ssvarma_filter_shocks(U::AbstractMatrix, Theta, C::AbstractMatrix; tol::Real=1e-8)
    n = size(U, 2)
    T_fb = float(promote_type(eltype(U), eltype(C)))
    T_th = _svarma_lag_type(Theta, T_fb)
    Theta_3d = _svarma_to_3d(Theta, n, T_th)
    return _svarma_filter_core(U, Theta_3d, C; tol=tol)
end

function ssvarma_filter_shocks(Y::AbstractMatrix, mu::AbstractVector, Phi, Theta,
                               C::AbstractMatrix; tol::Real=1e-8)
    n = size(Y, 2)
    T_fb = float(promote_type(eltype(Y), eltype(mu), eltype(C)))
    T_ph = _svarma_lag_type(Phi, T_fb)
    T_th = _svarma_lag_type(Theta, T_fb)
    Phi_3d = _svarma_to_3d(Phi, n, T_ph)
    Theta_3d = _svarma_to_3d(Theta, n, T_th)
    U = _svarma_ar_residuals(Y, mu, Phi_3d)
    return _svarma_filter_core(U, Theta_3d, C; tol=tol)
end

function ssvarma_filter_shocks(Y::AbstractMatrix, Phi, Theta, C::AbstractMatrix;
                               mu=nothing, tol::Real=1e-8)
    n = size(Y, 2)
    T_fb = float(promote_type(eltype(Y), _svarma_eltype(Phi), _svarma_eltype(Theta), eltype(C)))
    T_mu = T_fb <: AbstractFloat ? T_fb : Float64
    mu_vec = mu === nothing ? zeros(T_mu, n) : Vector{T_mu}(mu)
    return ssvarma_filter_shocks(Y, mu_vec, Phi, Theta, C; tol=tol)
end

function ssvarma_filter_shocks(m::SVARMAModel; tol::Real=1e-8)
    return ssvarma_filter_shocks(m.Y, m.mu, m.Phi, m.Theta, m.C; tol=tol)
end
