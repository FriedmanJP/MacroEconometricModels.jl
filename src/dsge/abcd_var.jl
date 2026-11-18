# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Shock recoverability (Sims 2012), exact population VAR(p) projection, and three-way
DSGE-vs-SVAR(p) IRF distortion decomposition (Ravenna 2007) for canonical ABCD models.

References:
- Fernández-Villaverde, J., Rubio-Ramírez, J. F., Sargent, T. J., & Watson, M. W. (2007).
  ABCs (and Ds) of Understanding VARs. *American Economic Review*, 97(3), 1021–1026.
- Ravenna, F. (2007). Vector autoregressions and reduced form representations of DSGE models.
  *Journal of Monetary Economics*, 54(7), 2048–2064.
- Sims, E. R. (2012). News, non-invertibility, and structural VARs.
  *Advances in Econometrics*, 28, 81–135.
"""

# =============================================================================
# Shared internal helpers (_abcd_var_*)
# =============================================================================

"""Subset an `ABCDSystem` by observable selector if provided."""
function _abcd_var_subset(sys::ABCDSystem{T}, observables) where {T<:AbstractFloat}
    observables === nothing && return sys
    obs_vec = observables isa AbstractVector ? observables : [observables]
    isempty(obs_vec) && return sys

    all_syms = Symbol.(sys.obs_names)
    obs_idx, obs_names = _resolve_abcd_observables(all_syms, all_syms, obs_vec)
    return ABCDSystem{T}(
        copy(sys.A),
        copy(sys.B),
        Matrix{T}(sys.C[obs_idx, :]),
        Matrix{T}(sys.D[obs_idx, :]),
        copy(sys.c_state),
        Vector{T}(sys.d_obs[obs_idx]),
        copy(sys.state_names),
        obs_names,
        copy(sys.shock_names),
    )
end

function _abcd_var_to_abcd(sys::ABCDSystem; observables=nothing, kwargs...)
    return _abcd_var_subset(sys, observables)
end

function _abcd_var_to_abcd(
    sol::Union{DSGESolution,PerturbationSolution,HADSGESolution};
    observables=nothing,
    measurement_error=nothing,
    reduction::Symbol=:minimal,
    abcd_tol::Real=1e-10,
)
    obs_arg = observables === nothing ? Symbol[] :
        (observables isa AbstractVector ? observables : [observables])
    return abcd_representation(sol, obs_arg;
                               measurement_error=measurement_error,
                               reduction=reduction,
                               tol=abcd_tol)
end

function _abcd_var_to_abcd(
    ss::StateSpaceModel;
    observables=nothing,
    reduction::Symbol=:none,
    abcd_tol::Real=1e-10,
)
    obs_arg = observables === nothing ? Symbol[] :
        (observables isa AbstractVector ? observables : [observables])
    return abcd_representation(ss, obs_arg; reduction=reduction, tol=abcd_tol)
end

"""Symmetric Moore–Penrose pseudoinverse of a PSD covariance matrix with relative eigval cutoff."""
function _abcd_var_pinv_sym(S::AbstractMatrix{WT}; rcond_tol::WT) where {WT<:AbstractFloat}
    k = size(S, 1)
    k == 0 && return zeros(WT, 0, 0)
    S_sym = Matrix{WT}((S + S') / 2)
    F = eigen(Hermitian(S_sym))
    lmax = maximum(F.values; init=zero(WT))
    lmax <= zero(WT) && return zeros(WT, k, k)
    thresh = max(rcond_tol, eps(WT) * WT(max(k, 1)) * WT(10)) * lmax
    inv_vals = WT[v > thresh ? one(WT) / v : zero(WT) for v in F.values]
    return Matrix{WT}(F.vectors * Diagonal(inv_vals) * F.vectors')
end

"""Symmetric PSD square root `L` such that `L * L' ≈ S` (tolerates singular PSD matrices)."""
function _abcd_var_psd_sqrt(S::AbstractMatrix{WT}) where {WT<:AbstractFloat}
    k = size(S, 1)
    k == 0 && return zeros(WT, 0, 0)
    S_sym = Matrix{WT}((S + S') / 2)
    F = eigen(Hermitian(S_sym))
    vals = max.(F.values, zero(WT))
    return Matrix{WT}(F.vectors * Diagonal(sqrt.(vals)))
end

"""Clip tiny negative numerical eigenvalues on a symmetrized PSD matrix."""
function _abcd_var_clip_psd(S::AbstractMatrix{WT}) where {WT<:AbstractFloat}
    k = size(S, 1)
    k == 0 && return zeros(WT, 0, 0)
    S_sym = Matrix{WT}((S + S') / 2)
    F = eigen(Hermitian(S_sym))
    min_ev = minimum(F.values)
    min_ev >= zero(WT) && return S_sym
    vals = max.(F.values, zero(WT))
    S_clean = Matrix{WT}(F.vectors * Diagonal(vals) * F.vectors')
    return Matrix{WT}((S_clean + S_clean') / 2)
end

"""Apply gentle stationary damping to `A` if unit-root eigenvalues appear; reject explosive `A`."""
function _abcd_var_damped_A(A_w::Matrix{WT}; unit_root_damping::Real=1e-5) where {WT<:AbstractFloat}
    n = size(A_w, 1)
    n == 0 && return copy(A_w)
    damping = WT(unit_root_damping)
    (zero(WT) < damping < one(WT)) ||
        throw(ArgumentError("unit_root_damping must lie in (0, 1); got $unit_root_damping"))

    max_eig = maximum(abs, eigvals(A_w); init=zero(WT))
    max_eig > one(WT) + WT(1e-4) && throw(ArgumentError(
        "ABCDSystem state transition A is explosive (max |eigenvalue| = $max_eig > 1); " *
        "stationary population moments do not exist"))
    if max_eig >= one(WT) - min(damping / WT(10), WT(1e-7))
        scale = (one(WT) - damping) / max_eig
        return Matrix{WT}(A_w * scale)
    end
    return copy(A_w)
end

"""Deterministic, zero-RNG Weyl-equidistributed white noise matrix (`T_eff × k`) with `(E'E)/T_eff ≈ I_k`."""
function _abcd_var_deterministic_noise(::Type{WT}, T_eff::Int, k::Int;
                                       offset::Int=0) where {WT<:AbstractFloat}
    (T_eff == 0 || k == 0) && return zeros(WT, T_eff, k)
    primes = Int[]
    cand = 2
    while length(primes) < k
        is_p = true
        for p_prev in primes
            p_prev * p_prev > cand && break
            if cand % p_prev == 0
                is_p = false
                break
            end
        end
        is_p && push!(primes, cand)
        cand += 1
    end

    E0 = Matrix{WT}(undef, T_eff, k)
    @inbounds for j in 1:k
        alpha = sqrt(WT(primes[j]))
        for t in 1:T_eff
            idx = t + offset
            u = mod(WT(0.5) * WT(idx) * WT(idx + 1) * alpha, one(WT))
            u_clamped = clamp(u, WT(1e-6), one(WT) - WT(1e-6))
            E0[t, j] = WT(quantile(Normal(), Float64(u_clamped)))
        end
    end
    E0 .-= mean(E0; dims=1)
    if T_eff > k
        C_E = Matrix{WT}((E0' * E0) / WT(T_eff))
        L_E = safe_cholesky(Hermitian(C_E); silent=true)
        L_E_inv = Matrix{WT}(robust_inv(L_E; silent=true))
        return Matrix{WT}(E0 * transpose(L_E_inv))
    end
    return E0
end

# =============================================================================
# ShockRecoverabilityResult{T} & shock_recoverability (Sims 2012)
# =============================================================================

"""
    ShockRecoverabilityResult{T<:AbstractFloat}

Sims (2012) shock-by-shock fundamentalness and two-sided recoverability measure
computed from the steady-state Wold innovations representation of an [`ABCDSystem`](@ref):

```math
R_j^2(\\ell) = \\mathrm{Var}\\!\\left(\\mathbb{E}[w_{j,t} \\mid y^{t+\\ell}]\\right) = \\sum_{s=0}^{\\ell} \\left[H_s' \\Sigma_u^{+} H_s\\right]_{jj} \\in [0, 1]
```

where:
- ``w_{j,t}`` is the ``j``-th unit-variance structural shock at time ``t``
- ``y^{t+\\ell}`` is the history of observables up to lead ``t + \\ell``
- ``H_0 = D`` and ``H_s = C (A - K C)^{s-1} (B - K D)`` for ``s \\ge 1`` are the shock-to-innovation cross-covariances
- ``\\Sigma_u^{+}`` is the symmetric Moore–Penrose pseudoinverse of the Wold innovation covariance ``\\Sigma_u = C P C' + D D'``
- ``\\ell \\in \\{0, \\dots, \\ell_{\\max}\\}`` is the future observable lead horizon

# Fields
- `r2_matrix::Matrix{T}` — `(max_lead + 1) × m` matrix of `R²_j(ℓ)` across leads `ℓ = 0..max_lead`
- `r2_contemp::Vector{T}` — length-`m` contemporaneous recoverability vector `R²_j(0)`
- `shock_names::Vector{String}` — length-`m` structural shock labels
- `obs_names::Vector{String}` — length-`k` observable labels
- `is_invertible::Bool` — `true` when the underlying `ABCDSystem` satisfies shock invertibility
- `max_lead::Int` — maximum future lead horizon `ℓ_max`
"""
struct ShockRecoverabilityResult{T<:AbstractFloat}
    r2_matrix::Matrix{T}
    r2_contemp::Vector{T}
    shock_names::Vector{String}
    obs_names::Vector{String}
    is_invertible::Bool
    max_lead::Int

    function ShockRecoverabilityResult{T}(
        r2_matrix::AbstractMatrix,
        r2_contemp::AbstractVector,
        shock_names::AbstractVector,
        obs_names::AbstractVector,
        is_invertible::Bool,
        max_lead::Int,
    ) where {T<:AbstractFloat}
        max_lead >= 0 || throw(ArgumentError("max_lead must be non-negative; got $max_lead"))
        r2m = Matrix{T}(r2_matrix)
        r2c = Vector{T}(r2_contemp)
        sn = String[string(s) for s in shock_names]
        on = String[string(s) for s in obs_names]
        m = length(sn)
        size(r2m) == (max_lead + 1, m) || throw(ArgumentError(
            "r2_matrix must be (max_lead+1)×m = ($(max_lead + 1), $m), got $(size(r2m))"))
        length(r2c) == m || throw(ArgumentError(
            "r2_contemp must have length m=$m, got $(length(r2c))"))
        new{T}(r2m, r2c, sn, on, is_invertible, max_lead)
    end
end

StatsAPI.r2(r::ShockRecoverabilityResult) = r.r2_contemp
is_invertible(r::ShockRecoverabilityResult) = r.is_invertible

function Base.show(io::IO, res::ShockRecoverabilityResult{T}) where {T}
    m = length(res.shock_names)
    k = length(res.obs_names)
    min_r2 = isempty(res.r2_contemp) ? "—" : _fmt(minimum(res.r2_contemp); digits=6)
    mean_r2 = isempty(res.r2_contemp) ? "—" : _fmt(mean(res.r2_contemp); digits=6)
    max_r2_end = isempty(res.r2_matrix) ? "—" : _fmt(minimum(@view res.r2_matrix[end, :]); digits=6)

    spec_data = Any[
        "Shocks (m)"            m;
        "Observables (k)"       k;
        "Max lead (ℓ)"          res.max_lead;
        "Invertible"            _yesno(res.is_invertible);
        "Min R²(0)"             min_r2;
        "Mean R²(0)"            mean_r2;
        "Min R²($(res.max_lead))" max_r2_end;
        "Observable labels"     isempty(res.obs_names) ? "—" : join(res.obs_names, ", ");
    ]
    _pretty_table(io, spec_data;
        title = "Sims (2012) Shock Recoverability Diagnostics",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

"""
    report([io::IO], res::ShockRecoverabilityResult)

Print a summary table of [`ShockRecoverabilityResult`](@ref) and the per-shock
recoverability profile `R²_j(ℓ)` across representative future observable leads `ℓ`.
"""
function report(io::IO, res::ShockRecoverabilityResult{T}) where {T}
    show(io, res)
    m = length(res.shock_names)
    if m > 0
        println(io)
        leads = res.max_lead == 0 ? [0] :
            unique!(sort(vcat([0], filter(<=(res.max_lead), [1, 2, 4, 8, 12, 24]), [res.max_lead])))
        tbl = Matrix{Any}(undef, m, length(leads) + 1)
        for j in 1:m
            tbl[j, 1] = res.shock_names[j]
            for (c, ell) in enumerate(leads)
                tbl[j, c + 1] = _fmt(res.r2_matrix[ell + 1, j]; digits=6)
            end
        end
        _pretty_table(io, tbl;
            title = "Shock Recoverability R²_j(ℓ) = Var(E[w_{j,t} | y^{t+ℓ}])",
            column_labels = vcat(["Shock"], ["ℓ=$ell" for ell in leads]),
            alignment = vcat([:l], fill(:r, length(leads))),
        )
    end
    return nothing
end

report(res::ShockRecoverabilityResult) = report(stdout, res)

"""
    shock_recoverability(sys::ABCDSystem{T}; max_lead=12, observables=nothing, rcond_tol=1e-12) -> ShockRecoverabilityResult{T}
    shock_recoverability(sol; max_lead=12, observables=nothing, measurement_error=nothing, reduction=:predetermined, abcd_tol=1e-10, rcond_tol=1e-12) -> ShockRecoverabilityResult
    shock_recoverability(sol, observables; max_lead=12, ...) -> ShockRecoverabilityResult

Compute Sims (2012) shock-by-shock fundamentalness and ``\\ell``-lead smoothed
recoverability ``R_j^2(\\ell) = \\mathrm{Var}(\\mathbb{E}[w_{j,t} \\mid y^{t+\\ell}]) \\in [0, 1]``
for each structural shock ``j = 1 \\dots m`` and lead ``\\ell = 0 \\dots \\ell_{\\max}``:

```math
R_j^2(\\ell) = \\sum_{s=0}^{\\ell} \\left[H_s' \\Sigma_u^{+} H_s\\right]_{jj}, \\qquad
H_0 = D, \\quad H_s = C (A - K C)^{s-1} (B - K D) \\; (s \\ge 1)
```

where:
- ``A, B, C, D`` are the canonical state-space matrices
- ``K`` and ``\\Sigma_u`` are the steady-state Kalman gain and Wold innovation covariance from [`innovations_representation`](@ref)
- ``\\Sigma_u^{+}`` is the symmetric pseudoinverse of ``\\Sigma_u``
- ``w_{j,t}`` is the ``j``-th unit-variance structural shock and ``y^{t+\\ell}`` is the observable history through ``t+\\ell``

# References
- Sims, E. R. (2012). News, non-invertibility, and structural VARs. *Advances in Econometrics*, 28, 81–135.
- Gouriéroux, C., Monfort, A., & Renne, J.-P. (2020). Identification and estimation in non-fundamental structural VARMA models. *Review of Economic Studies*, 87(4), 1915–1953.
"""
function shock_recoverability(sys::ABCDSystem{T};
                              max_lead::Int=12,
                              observables=nothing,
                              rcond_tol::Real=1e-12) where {T<:AbstractFloat}
    max_lead >= 0 || throw(ArgumentError("max_lead must be non-negative; got $max_lead"))
    rcond_tol > 0 || throw(ArgumentError("rcond_tol must be positive; got $rcond_tol"))

    abcd = _abcd_var_subset(sys, observables)
    n = nstates(abcd)
    k = nobservables(abcd)
    m = nshocks(abcd)

    innov = innovations_representation(abcd; rcond_tol=rcond_tol)
    WT = promote_type(T, Float64)
    r2_w = zeros(WT, max_lead + 1, m)

    if k > 0 && m > 0
        A_w = Matrix{WT}(abcd.A)
        B_w = Matrix{WT}(abcd.B)
        C_w = Matrix{WT}(abcd.C)
        D_w = Matrix{WT}(abcd.D)
        K_w = Matrix{WT}(innov.K)
        Sigma_u_w = Matrix{WT}(innov.Sigma_u)
        Sigma_u_pinv = _abcd_var_pinv_sym(Sigma_u_w; rcond_tol=WT(rcond_tol))

        F_w = n == 0 ? zeros(WT, 0, 0) : Matrix{WT}(A_w - K_w * C_w)
        G_w = n == 0 ? zeros(WT, 0, m) : Matrix{WT}(B_w - K_w * D_w)

        cum_cov = Matrix{WT}(D_w' * Sigma_u_pinv * D_w)
        @inbounds for j in 1:m
            r2_w[1, j] = clamp(cum_cov[j, j], zero(WT), one(WT))
        end

        if max_lead >= 1 && n > 0
            F_pow_G = copy(G_w)
            for ell in 1:max_lead
                H_ell = C_w * F_pow_G
                cum_cov .+= H_ell' * Sigma_u_pinv * H_ell
                @inbounds for j in 1:m
                    val = clamp(cum_cov[j, j], zero(WT), one(WT))
                    r2_w[ell + 1, j] = max(r2_w[ell, j], val)
                end
                if ell < max_lead
                    F_pow_G = F_w * F_pow_G
                end
            end
        elseif max_lead >= 1
            for ell in 1:max_lead
                @inbounds for j in 1:m
                    r2_w[ell + 1, j] = r2_w[ell, j]
                end
            end
        end
    end

    r2_mat = Matrix{T}(r2_w)
    r2_c = m == 0 ? T[] : Vector{T}(@view r2_mat[1, :])
    return ShockRecoverabilityResult{T}(
        r2_mat,
        r2_c,
        copy(abcd.shock_names),
        copy(abcd.obs_names),
        innov.invertible,
        max_lead,
    )
end

function shock_recoverability(
    sol::Union{DSGESolution,PerturbationSolution,HADSGESolution};
    max_lead::Int=12,
    observables=nothing,
    measurement_error=nothing,
    reduction::Symbol=:minimal,
    abcd_tol::Real=1e-10,
    rcond_tol::Real=1e-12,
)
    sys = _abcd_var_to_abcd(sol;
                            observables=observables,
                            measurement_error=measurement_error,
                            reduction=reduction,
                            abcd_tol=abcd_tol)
    return shock_recoverability(sys; max_lead=max_lead, rcond_tol=rcond_tol)
end

function shock_recoverability(
    sol::Union{DSGESolution,PerturbationSolution,HADSGESolution},
    observables::AbstractVector;
    max_lead::Int=12,
    measurement_error=nothing,
    reduction::Symbol=:minimal,
    abcd_tol::Real=1e-10,
    rcond_tol::Real=1e-12,
)
    return shock_recoverability(sol;
                                max_lead=max_lead,
                                observables=observables,
                                measurement_error=measurement_error,
                                reduction=reduction,
                                abcd_tol=abcd_tol,
                                rcond_tol=rcond_tol)
end

function shock_recoverability(
    ss::StateSpaceModel;
    max_lead::Int=12,
    observables=nothing,
    reduction::Symbol=:none,
    abcd_tol::Real=1e-10,
    rcond_tol::Real=1e-12,
)
    sys = _abcd_var_to_abcd(ss; observables=observables, reduction=reduction, abcd_tol=abcd_tol)
    return shock_recoverability(sys; max_lead=max_lead, rcond_tol=rcond_tol)
end

function shock_recoverability(
    ss::StateSpaceModel,
    observables::AbstractVector;
    max_lead::Int=12,
    reduction::Symbol=:none,
    abcd_tol::Real=1e-10,
    rcond_tol::Real=1e-12,
)
    return shock_recoverability(ss;
                                max_lead=max_lead,
                                observables=observables,
                                reduction=reduction,
                                abcd_tol=abcd_tol,
                                rcond_tol=rcond_tol)
end

# =============================================================================
# Exact Population VAR(p) Projection (population_var)
# =============================================================================

"""Compute exact theoretical autocovariances `Γ_y(0..p)` and unconditional mean `μ_y`."""
function _abcd_var_autocovariances(abcd::ABCDSystem{T}, p::Int;
                                   unit_root_damping::Real=1e-5) where {T<:AbstractFloat}
    WT = promote_type(T, Float64)
    n = nstates(abcd)
    k = nobservables(abcd)

    A_w = Matrix{WT}(abcd.A)
    B_w = Matrix{WT}(abcd.B)
    C_w = Matrix{WT}(abcd.C)
    D_w = Matrix{WT}(abcd.D)

    A_work = _abcd_var_damped_A(A_w; unit_root_damping=unit_root_damping)
    Sigma_x = n == 0 ? zeros(WT, 0, 0) : Matrix{WT}(solve_lyapunov(A_work, B_w))

    G0_raw = C_w * Sigma_x * C_w' + D_w * D_w'
    Gamma0 = Matrix{WT}((G0_raw + G0_raw') / 2)

    Gammas = Vector{Matrix{WT}}(undef, p)
    if p > 0
        M_cross = A_work * Sigma_x * C_w' + B_w * D_w'
        A_pow = Matrix{WT}(I, n, n)
        for h in 1:p
            Gammas[h] = Matrix{WT}(C_w * (A_pow * M_cross))
            if h < p
                A_pow = A_pow * A_work
            end
        end
    end

    c_s = Vector{WT}(abcd.c_state)
    d_o = Vector{WT}(abcd.d_obs)
    mu_x = (n == 0 || iszero(c_s)) ? zeros(WT, n) :
        Vector{WT}(robust_inv(Matrix{WT}(I, n, n) - A_work; silent=true, rcond_tol=WT(1e-12)) * c_s)
    mu_y = Vector{WT}(d_o + (n == 0 ? zeros(WT, k) : C_w * mu_x))

    return Gamma0, Gammas, mu_y
end

"""
    population_var(sys::ABCDSystem{T}, p::Int; observables=nothing, n_obs_effective=1000, unit_root_damping=1e-5) -> VARModel{T}
    population_var(sol, p::Int; observables=nothing, n_obs_effective=1000, ...) -> VARModel
    population_var(sol, observables, p::Int; n_obs_effective=1000, ...) -> VARModel

Construct the exact population `VAR(p)` projection of an [`ABCDSystem`](@ref) (or solved
DSGE / Perturbation / HA-DSGE / StateSpace model) by solving the block-Toeplitz Yule–Walker
equations on theoretical autocovariances ``\\Gamma_y(0), \\dots, \\Gamma_y(p)``:

```math
\\Gamma_{yx} = \\Phi^{(p)} \\Gamma_{xx}, \\qquad
\\Sigma_u^{(p)} = \\Gamma_y(0) - \\Phi^{(p)} \\Gamma_{yx}'
```

where:
- ``\\Gamma_y(0) = C \\Sigma_x C' + D D'`` and ``\\Gamma_y(h) = C A^{h-1}(A \\Sigma_x C' + B D')`` are exact observable autocovariances
- ``\\Gamma_{xx}`` is the ``kp \\times kp`` symmetric block-Toeplitz matrix with ``(i, j)`` block ``\\Gamma_y(j - i)``
- ``\\Gamma_{yx} = [\\Gamma_y(1) \\;\\; \\cdots \\;\\; \\Gamma_y(p)]`` is the ``k \\times kp`` lag-1..p cross-covariance block
- ``\\Phi^{(p)} = [\\Phi_1^{(p)} \\;\\; \\cdots \\;\\; \\Phi_p^{(p)}]`` are the population VAR slope matrices
- ``\\Sigma_u^{(p)}`` is the population `VAR(p)` prediction-error covariance

When ``A`` contains unit-root eigenvalues, gentle stationary damping (`unit_root_damping`)
stabilizes the Lyapunov equation before forming ``\\Gamma_y``. The returned [`VARModel`](@ref)
carries a deterministic population-moment surrogate sample of effective length
`n_obs_effective` and works directly with [`irf`](@ref), [`fevd`](@ref), [`compute_Q`](@ref),
and all structural identification schemes.

# References
- Fernández-Villaverde, J., Rubio-Ramírez, J. F., Sargent, T. J., & Watson, M. W. (2007).
  ABCs (and Ds) of Understanding VARs. *American Economic Review*, 97(3), 1021–1026.
- Ravenna, F. (2007). Vector autoregressions and reduced form representations of DSGE models.
  *Journal of Monetary Economics*, 54(7), 2048–2064.
"""
function population_var(sys::ABCDSystem{T}, p::Int;
                        observables=nothing,
                        n_obs_effective::Int=1000,
                        unit_root_damping::Real=1e-5) where {T<:AbstractFloat}
    p >= 1 || throw(ArgumentError("Lag order p must be >= 1; got $p"))
    n_obs_effective >= 1 || throw(ArgumentError(
        "n_obs_effective must be >= 1; got $n_obs_effective"))

    abcd = _abcd_var_subset(sys, observables)
    k = nobservables(abcd)
    k >= 1 || throw(ArgumentError("ABCDSystem must have at least 1 observable for population_var"))

    WT = promote_type(T, Float64)
    Gamma0, Gammas, mu_y = _abcd_var_autocovariances(abcd, p; unit_root_damping=unit_root_damping)

    kp = k * p
    Gamma_xx = zeros(WT, kp, kp)
    Gamma_yx = zeros(WT, k, kp)
    @inbounds for j in 1:p
        cols = ((j - 1) * k + 1):(j * k)
        Gamma_yx[:, cols] .= Gammas[j]
        for i in 1:p
            rows = ((i - 1) * k + 1):(i * k)
            if i == j
                Gamma_xx[rows, cols] .= Gamma0
            elseif j > i
                Gamma_xx[rows, cols] .= Gammas[j - i]
            else
                Gamma_xx[rows, cols] .= transpose(Gammas[i - j])
            end
        end
    end
    Gamma_xx = Matrix{WT}((Gamma_xx + Gamma_xx') / 2)

    Gamma_xx_inv = Matrix{WT}(robust_inv(Gamma_xx; silent=true, rcond_tol=WT(1e-12)))
    Phi_T = Matrix{WT}(Gamma_xx_inv * transpose(Gamma_yx))
    Phi = transpose(Phi_T)

    Sigma_raw = Gamma0 - Phi * transpose(Gamma_yx)
    Sigma_u_w = _abcd_var_clip_psd(Sigma_raw)

    c_vec = copy(mu_y)
    Phi_blocks = Matrix{WT}[Matrix{WT}(Phi[:, ((l - 1) * k + 1):(l * k)]) for l in 1:p]
    for l in 1:p
        c_vec .-= Phi_blocks[l] * mu_y
    end
    B_w = vcat(transpose(c_vec), Phi_T)

    # Construct deterministic sample consistent with (B_w, Sigma_u_w) for downstream consumers
    T_eff = n_obs_effective
    E = _abcd_var_deterministic_noise(WT, T_eff, k)
    L_u = _abcd_var_psd_sqrt(Sigma_u_w)
    U_w = Matrix{WT}(E * transpose(L_u))

    E_init = _abcd_var_deterministic_noise(WT, p, k; offset=T_eff)
    L_0 = _abcd_var_psd_sqrt(Gamma0)
    Y_w = Matrix{WT}(undef, T_eff + p, k)
    Y_w[1:p, :] .= transpose(mu_y) .+ E_init * transpose(L_0)

    y_buf = Vector{WT}(undef, k)
    @inbounds for t in (p + 1):(T_eff + p)
        copyto!(y_buf, c_vec)
        for l in 1:p
            mul!(y_buf, Phi_blocks[l], @view(Y_w[t - l, :]), one(WT), one(WT))
        end
        for j in 1:k
            y_buf[j] += U_w[t - p, j]
        end
        Y_w[t, :] .= y_buf
    end

    Sigma_T = Matrix{T}(Sigma_u_w)
    log_det = _suppress_warnings() do
        logdet_safe(Sigma_T)
    end
    k_sys = k * (1 + kp)
    aic = T(log_det + 2 * k_sys / T_eff)
    bic = T(log_det + k_sys * log(T_eff) / T_eff)
    hqic = T_eff > 1 ? T(log_det + 2 * k_sys * log(max(log(T_eff), 1e-12)) / T_eff) : aic

    return VARModel(
        Matrix{T}(Y_w),
        p,
        Matrix{T}(B_w),
        Matrix{T}(U_w),
        Sigma_T,
        aic,
        bic,
        hqic,
        copy(abcd.obs_names),
    )
end

function population_var(
    sol::Union{DSGESolution,PerturbationSolution,HADSGESolution},
    p::Int;
    observables=nothing,
    measurement_error=nothing,
    reduction::Symbol=:minimal,
    abcd_tol::Real=1e-10,
    n_obs_effective::Int=1000,
    unit_root_damping::Real=1e-5,
)
    sys = _abcd_var_to_abcd(sol;
                            observables=observables,
                            measurement_error=measurement_error,
                            reduction=reduction,
                            abcd_tol=abcd_tol)
    return population_var(sys, p;
                          n_obs_effective=n_obs_effective,
                          unit_root_damping=unit_root_damping)
end

function population_var(
    sol::Union{DSGESolution,PerturbationSolution,HADSGESolution},
    observables::AbstractVector,
    p::Int;
    measurement_error=nothing,
    reduction::Symbol=:minimal,
    abcd_tol::Real=1e-10,
    n_obs_effective::Int=1000,
    unit_root_damping::Real=1e-5,
)
    return population_var(sol, p;
                          observables=observables,
                          measurement_error=measurement_error,
                          reduction=reduction,
                          abcd_tol=abcd_tol,
                          n_obs_effective=n_obs_effective,
                          unit_root_damping=unit_root_damping)
end

function population_var(
    ss::StateSpaceModel,
    p::Int;
    observables=nothing,
    reduction::Symbol=:none,
    abcd_tol::Real=1e-10,
    n_obs_effective::Int=1000,
    unit_root_damping::Real=1e-5,
)
    sys = _abcd_var_to_abcd(ss; observables=observables, reduction=reduction, abcd_tol=abcd_tol)
    return population_var(sys, p;
                          n_obs_effective=n_obs_effective,
                          unit_root_damping=unit_root_damping)
end

function population_var(
    ss::StateSpaceModel,
    observables::AbstractVector,
    p::Int;
    reduction::Symbol=:none,
    abcd_tol::Real=1e-10,
    n_obs_effective::Int=1000,
    unit_root_damping::Real=1e-5,
)
    return population_var(ss, p;
                          observables=observables,
                          reduction=reduction,
                          abcd_tol=abcd_tol,
                          n_obs_effective=n_obs_effective,
                          unit_root_damping=unit_root_damping)
end

# =============================================================================
# DSGEVARIRFAudit{T} & compare_dsge_svar_irf (Ravenna 2007 3-Way Decomposition)
# =============================================================================

"""
    DSGEVARIRFAudit{T<:AbstractFloat}

Ravenna (2007) three-way DSGE vs. population `SVAR(p)` impulse-response distortion
decomposition separating non-invertibility, finite-lag truncation, and identification
rotation wedges:

```math
\\Theta^{\\text{SVAR}(p)} - \\Theta^{\\text{DSGE}} =
\\underbrace{\\left(\\Theta^{\\text{SVAR}(p)} - \\Theta^{\\text{Wold},\\text{id}}\\right)}_{\\text{Lag-truncation wedge}} +
\\underbrace{\\left(\\Theta^{\\text{Wold},\\text{id}} - \\Theta^{\\text{Wold},*}\\right)}_{\\text{Identification wedge}} +
\\underbrace{\\left(\\Theta^{\\text{Wold},*} - \\Theta^{\\text{DSGE}}\\right)}_{\\text{Non-invertibility wedge}}
```

where:
- ``\\Theta^{\\text{DSGE}}`` is the true ``H \\times k \\times m`` DSGE impulse response array
- ``\\Theta^{\\text{Wold},*}`` is the `VAR(∞)` Wold IRF rotated by Procrustes-optimal ``Q^*``
- ``\\Theta^{\\text{Wold},\\text{id}}`` is the `VAR(∞)` Wold IRF rotated by the chosen identification scheme
- ``\\Theta^{\\text{SVAR}(p)}`` is the population `SVAR(p)` IRF under the chosen identification scheme

# Fields
- `irf_dsge::Array{T,3}` — `(horizon × k × m)` true DSGE IRFs
- `irf_wold_best::Array{T,3}` — `(horizon × k × m)` best-rotated `VAR(∞)` Wold IRFs
- `irf_wold_id::Array{T,3}` — `(horizon × k × m)` identified `VAR(∞)` Wold IRFs
- `irf_svar::Array{T,3}` — `(horizon × k × m)` identified population `SVAR(p)` IRFs
- `noninvertibility_wedge::T` — RMSE of `irf_wold_best - irf_dsge`
- `truncation_wedge::T` — RMSE of `irf_svar - irf_wold_id`
- `identification_wedge::T` — RMSE of `irf_wold_id - irf_wold_best`
- `total_rmse::T` — RMSE of `irf_svar - irf_dsge`
- `rmse_by_horizon::Vector{T}` — length-`horizon` per-horizon total RMSE
- `noninv_by_horizon::Vector{T}` — length-`horizon` per-horizon non-invertibility RMSE
- `trunc_by_horizon::Vector{T}` — length-`horizon` per-horizon lag-truncation RMSE
- `id_by_horizon::Vector{T}` — length-`horizon` per-horizon identification RMSE
- `Q_best::Matrix{T}` — `k × m` Procrustes optimal rotation `Q^*`
- `Q_id_inf::Matrix{T}` — `k × m` identification rotation for `VAR(∞)`
- `Q_id_p::Matrix{T}` — `k × m` identification rotation for `SVAR(p)`
- `p::Int` — population VAR lag order
- `horizon::Int` — IRF horizon
- `identification::Symbol` — identification scheme (`:best_rotation`, `:cholesky`, `:long_run`, `:sign`, ...)
- `is_invertible::Bool` — invertibility flag of the underlying `ABCDSystem`
- `obs_names::Vector{String}` — length-`k` observable labels
- `shock_names::Vector{String}` — length-`m` structural shock labels
"""
struct DSGEVARIRFAudit{T<:AbstractFloat}
    irf_dsge::Array{T,3}
    irf_wold_best::Array{T,3}
    irf_wold_id::Array{T,3}
    irf_svar::Array{T,3}
    noninvertibility_wedge::T
    truncation_wedge::T
    identification_wedge::T
    total_rmse::T
    rmse_by_horizon::Vector{T}
    noninv_by_horizon::Vector{T}
    trunc_by_horizon::Vector{T}
    id_by_horizon::Vector{T}
    Q_best::Matrix{T}
    Q_id_inf::Matrix{T}
    Q_id_p::Matrix{T}
    p::Int
    horizon::Int
    identification::Symbol
    is_invertible::Bool
    obs_names::Vector{String}
    shock_names::Vector{String}

    function DSGEVARIRFAudit{T}(
        irf_dsge::AbstractArray{<:Real,3},
        irf_wold_best::AbstractArray{<:Real,3},
        irf_wold_id::AbstractArray{<:Real,3},
        irf_svar::AbstractArray{<:Real,3},
        noninvertibility_wedge::Real,
        truncation_wedge::Real,
        identification_wedge::Real,
        total_rmse::Real,
        rmse_by_horizon::AbstractVector,
        noninv_by_horizon::AbstractVector,
        trunc_by_horizon::AbstractVector,
        id_by_horizon::AbstractVector,
        Q_best::AbstractMatrix,
        Q_id_inf::AbstractMatrix,
        Q_id_p::AbstractMatrix,
        p::Int,
        horizon::Int,
        identification::Symbol,
        is_invertible::Bool,
        obs_names::AbstractVector,
        shock_names::AbstractVector,
    ) where {T<:AbstractFloat}
        p >= 1 || throw(ArgumentError("p must be >= 1; got $p"))
        horizon >= 1 || throw(ArgumentError("horizon must be >= 1; got $horizon"))
        on = String[string(s) for s in obs_names]
        sn = String[string(s) for s in shock_names]
        k = length(on)
        m = length(sn)
        exp_sz = (horizon, k, m)
        id = Array{T,3}(irf_dsge)
        iwb = Array{T,3}(irf_wold_best)
        iwi = Array{T,3}(irf_wold_id)
        isv = Array{T,3}(irf_svar)
        size(id) == exp_sz || throw(ArgumentError("irf_dsge size $(size(id)) != $exp_sz"))
        size(iwb) == exp_sz || throw(ArgumentError("irf_wold_best size $(size(iwb)) != $exp_sz"))
        size(iwi) == exp_sz || throw(ArgumentError("irf_wold_id size $(size(iwi)) != $exp_sz"))
        size(isv) == exp_sz || throw(ArgumentError("irf_svar size $(size(isv)) != $exp_sz"))
        rbh = Vector{T}(rmse_by_horizon)
        nbh = Vector{T}(noninv_by_horizon)
        tbh = Vector{T}(trunc_by_horizon)
        ibh = Vector{T}(id_by_horizon)
        length(rbh) == horizon || throw(ArgumentError("rmse_by_horizon length $(length(rbh)) != $horizon"))
        length(nbh) == horizon || throw(ArgumentError("noninv_by_horizon length $(length(nbh)) != $horizon"))
        length(tbh) == horizon || throw(ArgumentError("trunc_by_horizon length $(length(tbh)) != $horizon"))
        length(ibh) == horizon || throw(ArgumentError("id_by_horizon length $(length(ibh)) != $horizon"))
        Qb = Matrix{T}(Q_best)
        Qi = Matrix{T}(Q_id_inf)
        Qp = Matrix{T}(Q_id_p)
        size(Qb) == (k, m) || throw(ArgumentError("Q_best size $(size(Qb)) != ($k, $m)"))
        size(Qi) == (k, m) || throw(ArgumentError("Q_id_inf size $(size(Qi)) != ($k, $m)"))
        size(Qp) == (k, m) || throw(ArgumentError("Q_id_p size $(size(Qp)) != ($k, $m)"))
        new{T}(
            id, iwb, iwi, isv,
            T(noninvertibility_wedge), T(truncation_wedge),
            T(identification_wedge), T(total_rmse),
            rbh, nbh, tbh, ibh,
            Qb, Qi, Qp,
            p, horizon, identification, is_invertible, on, sn,
        )
    end
end

function Base.getproperty(a::DSGEVARIRFAudit, s::Symbol)
    if s === :irf_wold
        return getfield(a, :irf_wold_id)
    elseif s === :rmse
        return getfield(a, :total_rmse)
    end
    return getfield(a, s)
end

function Base.propertynames(a::DSGEVARIRFAudit, private::Bool=false)
    return (fieldnames(typeof(a))..., :irf_wold, :rmse)
end

is_invertible(a::DSGEVARIRFAudit) = a.is_invertible

function Base.show(io::IO, audit::DSGEVARIRFAudit{T}) where {T}
    spec_data = Any[
        "VAR lag order (p)"            audit.p;
        "IRF horizon (H)"              audit.horizon;
        "Identification"               string(audit.identification);
        "Invertible"                   _yesno(audit.is_invertible);
        "Non-invertibility wedge"      _fmt(audit.noninvertibility_wedge; digits=6);
        "Lag-truncation wedge"         _fmt(audit.truncation_wedge; digits=6);
        "Identification wedge"         _fmt(audit.identification_wedge; digits=6);
        "Total RMSE"                   _fmt(audit.total_rmse; digits=6);
        "Observables"                  isempty(audit.obs_names) ? "—" : join(audit.obs_names, ", ");
        "Shocks"                       isempty(audit.shock_names) ? "—" : join(audit.shock_names, ", ");
    ]
    _pretty_table(io, spec_data;
        title = "Ravenna (2007) DSGE-vs-SVAR(p) IRF Distortion Audit",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

"""
    report([io::IO], audit::DSGEVARIRFAudit)

Print a structured summary of a [`DSGEVARIRFAudit`](@ref) including the three-way
RMSE distortion wedges and a horizon-by-horizon error profile.
"""
function report(io::IO, audit::DSGEVARIRFAudit{T}) where {T}
    show(io, audit)
    println(io)
    hs = _select_horizons(audit.horizon)
    tbl = Matrix{Any}(undef, length(hs), 5)
    for (idx, h) in enumerate(hs)
        tbl[idx, 1] = h - 1
        tbl[idx, 2] = _fmt(audit.noninv_by_horizon[h]; digits=6)
        tbl[idx, 3] = _fmt(audit.trunc_by_horizon[h]; digits=6)
        tbl[idx, 4] = _fmt(audit.id_by_horizon[h]; digits=6)
        tbl[idx, 5] = _fmt(audit.rmse_by_horizon[h]; digits=6)
    end
    _pretty_table(io, tbl;
        title = "IRF Distortion Wedges by Horizon (RMSE)",
        column_labels = ["h", "Non-Inv.", "Trunc.", "Ident.", "Total"],
        alignment = [:r, :r, :r, :r, :r],
    )
    return nothing
end

report(audit::DSGEVARIRFAudit) = report(stdout, audit)

"""Orthogonal Procrustes rotation `Q ∈ ℝ^{k×m}` minimizing `∑_h ||Ψ_h Q - Θ_h||_F²`."""
function _abcd_var_procrustes(Psi::Array{WT,3}, Theta::Array{WT,3}) where {WT<:AbstractFloat}
    H, k, k2 = size(Psi)
    _, _, m = size(Theta)
    M = zeros(WT, k2, m)
    @inbounds for h in 1:H
        M .+= transpose(@view Psi[h, :, :]) * (@view Theta[h, :, :])
    end
    if norm(M) <= eps(WT) * WT(10)
        Q_id = zeros(WT, k2, m)
        for i in 1:min(k2, m)
            Q_id[i, i] = one(WT)
        end
        return Q_id
    end
    sv = svd(M; full=false)
    return Matrix{WT}(sv.U * sv.Vt)
end

"""Rotate `(H × k × k)` Wold IRF tensor by `Q (k × m)` into `(H × k × m)`."""
function _abcd_var_rotate_irf(Psi::Array{WT,3}, Q::Matrix{WT}) where {WT<:AbstractFloat}
    H, k, _ = size(Psi)
    m = size(Q, 2)
    out = zeros(WT, H, k, m)
    @inbounds for h in 1:H
        out[h, :, :] = (@view Psi[h, :, :]) * Q
    end
    return out
end

"""Blanchard–Quah long-run rotation `Q` on the `VAR(∞)` Wold innovations representation."""
function _abcd_var_long_run_Q(innov::InnovationsRepresentation{T};
                              unit_root_damping::Real=1e-5) where {T<:AbstractFloat}
    WT = promote_type(T, Float64)
    k = size(innov.C, 1)
    n = size(innov.A, 1)
    k == 0 && return zeros(WT, 0, 0)

    A_work = _abcd_var_damped_A(Matrix{WT}(innov.A); unit_root_damping=unit_root_damping)
    C_w = Matrix{WT}(innov.C)
    K_w = Matrix{WT}(innov.K)
    Sigma_u_w = Matrix{WT}((innov.Sigma_u + innov.Sigma_u') / 2)

    if n == 0
        C_u1 = Matrix{WT}(I, k, k)
    else
        I_minus_A_inv = Matrix{WT}(robust_inv(Matrix{WT}(I, n, n) - A_work;
                                              silent=true, rcond_tol=WT(1e-12)))
        C_u1 = Matrix{WT}(I, k, k) + C_w * I_minus_A_inv * K_w
    end

    V_LR_raw = C_u1 * Sigma_u_w * transpose(C_u1)
    V_LR = Matrix{WT}((V_LR_raw + V_LR_raw') / 2)
    L_LR = Matrix{WT}(safe_cholesky(V_LR; silent=true))
    @inbounds for j in 1:k
        L_LR[j, j] < zero(WT) && (@views L_LR[:, j] .*= -one(WT))
    end
    D_hat = Matrix{WT}(safe_cholesky(Sigma_u_w; silent=true))
    Q = Matrix{WT}(robust_inv(D_hat; silent=true) *
                   (robust_inv(C_u1; silent=true) * L_LR))
    ortho = norm(Q' * Q - I(k))
    ortho < WT(1e-6) || throw(IdentificationError(
        "_abcd_var_long_run_Q: long-run multiplier is singular or Q is not orthogonal (‖Q'Q−I‖=$ortho)"))
    return Q
end

"""Sign-restriction draw directly on a `(H × k × k)` Cholesky IRF tensor."""
function _abcd_var_sign_Q(Psi::Array{WT,3}, check_func::Function,
                          max_draws::Int, rng::AbstractRNG) where {WT<:AbstractFloat}
    k = size(Psi, 2)
    for _ in 1:max_draws
        Q = generate_Q(k, WT; rng=rng)
        irf_rot = _abcd_var_rotate_irf(Psi, Q)
        check_func(irf_rot) && return Q
    end
    throw(IdentificationError("No valid Q found after $max_draws sign-restriction draws"))
end

function _abcd_var_rmse(A::Array{WT,3}) where {WT<:AbstractFloat}
    isempty(A) && return zero(WT)
    return sqrt(sum(abs2, A) / WT(length(A)))
end

function _abcd_var_rmse_by_h(A::Array{WT,3}) where {WT<:AbstractFloat}
    H, k, m = size(A)
    denom = WT(max(k * m, 1))
    return WT[sqrt(sum(abs2, @view(A[h, :, :])) / denom) for h in 1:H]
end

"""
    compare_dsge_svar_irf(sys::ABCDSystem{T}, p::Int=4, horizon::Int=20; identification=:best_rotation, observables=nothing, ...) -> DSGEVARIRFAudit{T}
    compare_dsge_svar_irf(sol, p::Int=4, horizon::Int=20; identification=:best_rotation, observables=nothing, ...) -> DSGEVARIRFAudit
    compare_dsge_svar_irf(sol, observables::AbstractVector, p::Int=4, horizon::Int=20; ...) -> DSGEVARIRFAudit

Decompose the discrepancy between true DSGE impulse responses and population ``\\text{SVAR}(p)``
impulse responses into Ravenna's (2007) three orthogonal/structural wedges:

```math
\\Theta^{\\text{SVAR}(p)} - \\Theta^{\\text{DSGE}} =
\\underbrace{\\left(\\Theta^{\\text{SVAR}(p)} - \\Theta^{\\text{Wold},\\text{id}}\\right)}_{\\text{Lag-truncation}} +
\\underbrace{\\left(\\Theta^{\\text{Wold},\\text{id}} - \\Theta^{\\text{Wold},*}\\right)}_{\\text{Identification}} +
\\underbrace{\\left(\\Theta^{\\text{Wold},*} - \\Theta^{\\text{DSGE}}\\right)}_{\\text{Non-invertibility}}
```

where:
- ``\\Theta^{\\text{DSGE}}`` is the true structural impulse response from the `ABCDSystem`
- ``\\Theta^{\\text{Wold},*}`` is the `VAR(∞)` Wold IRF rotated by Procrustes-optimal orthogonal ``Q^*``
- ``\\Theta^{\\text{Wold},\\text{id}}`` is the `VAR(∞)` Wold IRF under `identification` (`:best_rotation`, `:cholesky`, `:long_run`, `:sign`, or any residual-free `compute_Q` method)
- ``\\Theta^{\\text{SVAR}(p)}`` is the population `SVAR(p)` IRF from [`population_var`](@ref) under `identification`

# Keyword Arguments
- `identification::Symbol=:best_rotation`: structural rotation scheme (`:best_rotation`, `:cholesky`, `:long_run`, `:sign`, or residual-free `compute_Q` scheme)
- `observables=nothing`: optional observable subset selector
- `n_obs_effective::Int=1000`: effective sample size carried by the population `VAR(p)` model
- `unit_root_damping::Real=1e-5`: gentle stationary damping applied when `A` has unit roots
- `rcond_tol::Real=1e-12`: conditioning threshold passed to [`innovations_representation`](@ref)
- `check_func=nothing`: sign-restriction check function required when `identification=:sign`
- `max_draws::Int=1000`: maximum Haar rotation draws for `:sign` / `:arias`
- `rng::AbstractRNG=Random.default_rng()`: random number generator for stochastic identification schemes
- `seed::Union{Integer,Nothing}=nothing`: optional seed for reproducible stochastic identification

# References
- Ravenna, F. (2007). Vector autoregressions and reduced form representations of DSGE models.
  *Journal of Monetary Economics*, 54(7), 2048–2064.
- Fernández-Villaverde, J., Rubio-Ramírez, J. F., Sargent, T. J., & Watson, M. W. (2007).
  ABCs (and Ds) of Understanding VARs. *American Economic Review*, 97(3), 1021–1026.
"""
function compare_dsge_svar_irf(
    sys::ABCDSystem{T},
    p::Int=4,
    horizon::Int=20;
    identification::Symbol=:best_rotation,
    observables=nothing,
    n_obs_effective::Int=1000,
    unit_root_damping::Real=1e-5,
    rcond_tol::Real=1e-12,
    check_func=nothing,
    max_draws::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
    kwargs...,
) where {T<:AbstractFloat}
    p >= 1 || throw(ArgumentError("Lag order p must be >= 1; got $p"))
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1; got $horizon"))
    rng = _resolve_repro_rng(rng, seed)

    abcd = _abcd_var_subset(sys, observables)
    k = nobservables(abcd)
    m = nshocks(abcd)
    (k >= 1 && m >= 1) || throw(ArgumentError(
        "ABCDSystem must have at least 1 observable and 1 shock; got k=$k, m=$m"))

    WT = promote_type(T, Float64)
    Theta_dsge = Array{WT,3}(irf(abcd, horizon).values)

    innov = innovations_representation(abcd; rcond_tol=rcond_tol)
    Psi_wold = Array{WT,3}(irf(innov, horizon).values)

    Q_best = _abcd_var_procrustes(Psi_wold, Theta_dsge)
    Theta_wold_best = _abcd_var_rotate_irf(Psi_wold, Q_best)

    var_p = population_var(abcd, p;
                           n_obs_effective=n_obs_effective,
                           unit_root_damping=unit_root_damping)
    L_p = Matrix{WT}(safe_cholesky(var_p.Sigma; silent=true))
    Phi_p = ma_coefficients(Matrix{WT}(var_p.B), k, p, horizon)
    Psi_p = zeros(WT, horizon, k, k)
    @inbounds for h in 1:horizon
        Psi_p[h, :, :] = Phi_p[h] * L_p
    end

    if identification === :best_rotation
        Q_id_inf = copy(Q_best)
        Theta_wold_id = copy(Theta_wold_best)
        Q_id_p = _abcd_var_procrustes(Psi_p, Theta_wold_best)
        Theta_svar = _abcd_var_rotate_irf(Psi_p, Q_id_p)
    else
        k == m || throw(ArgumentError(
            "Identification scheme :$identification requires a square system (k == m), " *
            "got k=$k, m=$m; use identification=:best_rotation for rectangular systems"))

        if identification === :cholesky
            Q_id_inf = Matrix{WT}(I, k, k)
            Q_id_p = Matrix{WT}(I, k, k)
        elseif identification === :long_run
            Q_id_inf = _abcd_var_long_run_Q(innov; unit_root_damping=unit_root_damping)
            Q_id_p = Matrix{WT}(identify_long_run(var_p))
        elseif identification === :sign
            isnothing(check_func) && throw(ArgumentError("identification=:sign requires check_func"))
            rng_inf = copy(rng)
            rng_p = copy(rng)
            Q_id_inf = _abcd_var_sign_Q(Psi_wold, check_func, max_draws, rng_inf)
            Q_id_p = _abcd_var_sign_Q(Psi_p, check_func, max_draws, rng_p)
        else
            haskey(IDENTIFICATION_REGISTRY, identification) || throw(ArgumentError(
                "Unknown identification scheme :$identification"))
            _needs_residuals(identification) && throw(ArgumentError(
                "Identification scheme :$identification requires residual-level observations " *
                "and is not supported on population VAR moments"))
            var_inf = population_var(to_abcd(innov), max(p + 20, 40);
                                     n_obs_effective=max(n_obs_effective, 1000),
                                     unit_root_damping=unit_root_damping)
            rng_inf = copy(rng)
            rng_p = copy(rng)
            Q_id_inf = Matrix{WT}(compute_Q(var_inf, identification;
                                            horizon=horizon, check_func=check_func,
                                            max_draws=max_draws, rng=rng_inf, kwargs...))
            Q_id_p = Matrix{WT}(compute_Q(var_p, identification;
                                          horizon=horizon, check_func=check_func,
                                          max_draws=max_draws, rng=rng_p, kwargs...))
        end

        Theta_wold_id = _abcd_var_rotate_irf(Psi_wold, Q_id_inf)
        Theta_svar = _abcd_var_rotate_irf(Psi_p, Q_id_p)
    end

    diff_noninv = Theta_wold_best .- Theta_dsge
    diff_id = Theta_wold_id .- Theta_wold_best
    diff_trunc = Theta_svar .- Theta_wold_id
    diff_total = Theta_svar .- Theta_dsge

    return DSGEVARIRFAudit{T}(
        Array{T,3}(Theta_dsge),
        Array{T,3}(Theta_wold_best),
        Array{T,3}(Theta_wold_id),
        Array{T,3}(Theta_svar),
        T(_abcd_var_rmse(diff_noninv)),
        T(_abcd_var_rmse(diff_trunc)),
        T(_abcd_var_rmse(diff_id)),
        T(_abcd_var_rmse(diff_total)),
        Vector{T}(_abcd_var_rmse_by_h(diff_total)),
        Vector{T}(_abcd_var_rmse_by_h(diff_noninv)),
        Vector{T}(_abcd_var_rmse_by_h(diff_trunc)),
        Vector{T}(_abcd_var_rmse_by_h(diff_id)),
        Matrix{T}(Q_best),
        Matrix{T}(Q_id_inf),
        Matrix{T}(Q_id_p),
        p,
        horizon,
        identification,
        innov.invertible,
        copy(abcd.obs_names),
        copy(abcd.shock_names),
    )
end

function compare_dsge_svar_irf(
    sol::Union{DSGESolution,PerturbationSolution,HADSGESolution},
    p::Int=4,
    horizon::Int=20;
    identification::Symbol=:best_rotation,
    observables=nothing,
    measurement_error=nothing,
    reduction::Symbol=:minimal,
    abcd_tol::Real=1e-10,
    n_obs_effective::Int=1000,
    unit_root_damping::Real=1e-5,
    rcond_tol::Real=1e-12,
    check_func=nothing,
    max_draws::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
    kwargs...,
)
    sys = _abcd_var_to_abcd(sol;
                            observables=observables,
                            measurement_error=measurement_error,
                            reduction=reduction,
                            abcd_tol=abcd_tol)
    return compare_dsge_svar_irf(sys, p, horizon;
                                 identification=identification,
                                 n_obs_effective=n_obs_effective,
                                 unit_root_damping=unit_root_damping,
                                 rcond_tol=rcond_tol,
                                 check_func=check_func,
                                 max_draws=max_draws,
                                 rng=rng,
                                 seed=seed,
                                 kwargs...)
end

function compare_dsge_svar_irf(
    sol::Union{DSGESolution,PerturbationSolution,HADSGESolution},
    observables::AbstractVector,
    p::Int=4,
    horizon::Int=20;
    identification::Symbol=:best_rotation,
    measurement_error=nothing,
    reduction::Symbol=:minimal,
    abcd_tol::Real=1e-10,
    n_obs_effective::Int=1000,
    unit_root_damping::Real=1e-5,
    rcond_tol::Real=1e-12,
    check_func=nothing,
    max_draws::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
    kwargs...,
)
    return compare_dsge_svar_irf(sol, p, horizon;
                                 identification=identification,
                                 observables=observables,
                                 measurement_error=measurement_error,
                                 reduction=reduction,
                                 abcd_tol=abcd_tol,
                                 n_obs_effective=n_obs_effective,
                                 unit_root_damping=unit_root_damping,
                                 rcond_tol=rcond_tol,
                                 check_func=check_func,
                                 max_draws=max_draws,
                                 rng=rng,
                                 seed=seed,
                                 kwargs...)
end

function compare_dsge_svar_irf(
    ss::StateSpaceModel,
    p::Int=4,
    horizon::Int=20;
    identification::Symbol=:best_rotation,
    observables=nothing,
    reduction::Symbol=:none,
    abcd_tol::Real=1e-10,
    n_obs_effective::Int=1000,
    unit_root_damping::Real=1e-5,
    rcond_tol::Real=1e-12,
    check_func=nothing,
    max_draws::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
    kwargs...,
)
    sys = _abcd_var_to_abcd(ss; observables=observables, reduction=reduction, abcd_tol=abcd_tol)
    return compare_dsge_svar_irf(sys, p, horizon;
                                 identification=identification,
                                 n_obs_effective=n_obs_effective,
                                 unit_root_damping=unit_root_damping,
                                 rcond_tol=rcond_tol,
                                 check_func=check_func,
                                 max_draws=max_draws,
                                 rng=rng,
                                 seed=seed,
                                 kwargs...)
end

function compare_dsge_svar_irf(
    ss::StateSpaceModel,
    observables::AbstractVector,
    p::Int=4,
    horizon::Int=20;
    identification::Symbol=:best_rotation,
    reduction::Symbol=:none,
    abcd_tol::Real=1e-10,
    n_obs_effective::Int=1000,
    unit_root_damping::Real=1e-5,
    rcond_tol::Real=1e-12,
    check_func=nothing,
    max_draws::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
    seed::Union{Integer,Nothing}=nothing,
    kwargs...,
)
    return compare_dsge_svar_irf(ss, p, horizon;
                                 identification=identification,
                                 observables=observables,
                                 reduction=reduction,
                                 abcd_tol=abcd_tol,
                                 n_obs_effective=n_obs_effective,
                                 unit_root_damping=unit_root_damping,
                                 rcond_tol=rcond_tol,
                                 check_func=check_func,
                                 max_draws=max_draws,
                                 rng=rng,
                                 seed=seed,
                                 kwargs...)
end
