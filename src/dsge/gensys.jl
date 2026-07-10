# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Sims (2002) gensys solver for linear rational expectations models.

Solves: Gamma0 * y_t = Gamma1 * y_{t-1} + C + Psi * eps_t + Pi * eta_t
Returns: y_t = G1 * y_{t-1} + impact * eps_t + C_sol

Solution approach in `solve(:gensys)`: undetermined coefficients (primary G1/impact,
robust to many static variables), with the companion-QZ core (`_solve_qz_quadratic`)
supplying determinacy (`eu`) and a fallback G1/impact when UC does not converge.
"""

"""
    gensys(Gamma0, Gamma1, C, Psi, Pi; div=1.0+1e-8) -> NamedTuple

Solve the linear RE system via QZ decomposition (Sims 2002).

Returns `(G1, impact, C_sol, eu, eigenvalues)` where:
- `eu = [exist, unique]`: 1=yes, 0=no
- `div` is the dividing line between stable (|lambda| < div) and unstable eigenvalues.
"""
function gensys(Gamma0::AbstractMatrix{T}, Gamma1::AbstractMatrix{T},
                C::AbstractVector{T}, Psi::AbstractMatrix{T}, Pi::AbstractMatrix{T};
                div::Real=1.0 + 1e-8, cluster_tol::Real=1e-6,
                f_lead::Union{Nothing,AbstractMatrix{T}}=nothing) where {T<:AbstractFloat}
    n = size(Gamma0, 1)
    eu = [0, 0]

    # QZ (generalized Schur) decomposition for eigenvalue analysis
    F = schur(complex(Gamma0), complex(Gamma1))
    S = F.S; TT = F.T

    gev_mag = zeros(n)
    for i in 1:n
        gev_mag[i] = abs(S[i,i]) > eps(T) ? abs(TT[i,i] / S[i,i]) : Inf
    end

    divhat = _place_divhat(gev_mag, div, cluster_tol)   # exact |λ|=1 → unstable (shared with QZ core)
    stable_select = BitVector(gev_mag .< divhat)
    F_ordered = ordschur(F, stable_select)
    S = F_ordered.S; TT = F_ordered.T
    Q = F_ordered.Q; Z = F_ordered.Z
    nstab = count(stable_select)
    nunstab = n - nstab

    eigenvalues = Vector{ComplexF64}(undef, n)
    for i in 1:n
        eigenvalues[i] = abs(S[i,i]) > eps(T) ? TT[i,i] / S[i,i] : complex(T(Inf))
    end

    Qp = Q'

    # Existence/uniqueness via QZ rank conditions
    n_fwd = size(Pi, 2)
    if n_fwd > 0 && nunstab > 0
        Q2Pi = Qp[nstab+1:n, :] * complex(Pi)
        Q2Psi = Qp[nstab+1:n, :] * complex(Psi)
        sv = svd(Q2Pi)
        rank_Q2Pi = count(s -> s > eps(T) * 1e6 * maximum(sv.S), sv.S)

        # Check consistency: range(Q2*Psi) ⊆ range(Q2*Pi)
        # Project Q2Psi onto null space of (Q2Pi)'
        proj_null = Q2Psi - Q2Pi * pinv(Q2Pi) * Q2Psi
        consistent = maximum(abs.(proj_null)) < T(1e-8)

        if consistent || rank_Q2Pi >= nunstab
            eu[1] = 1
        end

        # Count finite unstable eigenvalues for BK condition (same adaptive boundary)
        n_finite_unstable = count(i -> !isinf(abs(eigenvalues[i])) && abs(eigenvalues[i]) >= divhat,
                                   1:n)
        if n_finite_unstable == n_fwd || (consistent && n_finite_unstable <= n_fwd)
            eu[2] = 1
        end
    else
        if nunstab == 0
            eu = [1, 1]
        end
    end

    # Build solution via QZ stable block
    if nstab > 0
        Z1 = Z[:, 1:nstab]
        S11 = S[1:nstab, 1:nstab]
        T11 = TT[1:nstab, 1:nstab]
        Q1 = Qp[1:nstab, :]

        G1_c = Z1 * (S11 \ T11) * Z1'

        if nunstab > 0 && n_fwd > 0
            Q2 = Qp[nstab+1:n, :]
            Q2Pi = Q2 * complex(Pi)
            Q1Pi = Q1 * complex(Pi)
            Q2Pi_pinv = pinv(Q2Pi)
            Q1_adj = Q1 - Q1Pi * Q2Pi_pinv * Q2
            impact_c = Z1 * (S11 \ (Q1_adj * complex(Psi)))
        else
            impact_c = Z1 * (S11 \ (Q1 * complex(Psi)))
        end

        G1 = Matrix{T}(real(G1_c))
        impact = Matrix{T}(real(impact_c))
    else
        G1 = zeros(T, n, n)
        impact = zeros(T, n, size(Psi, 2))
    end

    # Compute solution constant: y_t = G1*y_{t-1} + impact*eps + C_sol
    # Static SS relation is (f₀+f₁+f_lead)·y = C. In the pencil Γ₀-Γ₁ = f₀+f₁; f_lead was
    # folded into Π, so add it back when supplied (audit S-06 / #114) — without it the SS
    # constant is wrong for forward-looking models with a constant.
    if norm(C) > eps(T)
        A_ss = f_lead === nothing ? (complex(Gamma0) - complex(Gamma1)) :
                                    (complex(Gamma0) - complex(Gamma1) + complex(f_lead))
        y_bar = real(Vector{T}(A_ss \ complex(C)))
        C_sol = (I - G1) * y_bar
    else
        C_sol = zeros(T, n)
    end

    (G1=G1, impact=impact, C_sol=C_sol, eu=eu, eigenvalues=eigenvalues)
end

"""
    _solve_undetermined_coefficients(spec::DSGESpec{T}) -> (G1, impact, eigenvalues)

Solve the first-order DSGE system via iterative undetermined coefficients.

From the linearized system: f₀·ŷ_t + f₁·ŷ_{t-1} + f_lead·E_t[ŷ_{t+1}] + f_ε·ε_t = 0

Guessing ŷ_t = G1·ŷ_{t-1} + M·ε_t and matching coefficients:
- G1 satisfies: (f₀ + f_lead·G1)·G1 + f₁ = 0 (quadratic matrix equation)
- M satisfies: (f₀ + f_lead·G1)·M + f_ε = 0

The iteration G1_{k+1} = -(f₀ + f_lead·G1_k)⁻¹·f₁ converges to the unique stable
solution. This method is robust to models with many static variables.
"""
function _solve_undetermined_coefficients(spec::DSGESpec{T};
        maxiter::Int=10000, tol::Real=1e-13) where {T<:AbstractFloat}
    y_ss = spec.steady_state
    f_0 = _dsge_jacobian(spec, y_ss, :current)
    f_1 = _dsge_jacobian(spec, y_ss, :lag)
    f_lead = _dsge_jacobian(spec, y_ss, :lead)
    f_eps = _dsge_jacobian_shocks(spec, y_ss)
    n = spec.n_endog

    # Iterative solution: G1_{k+1} = -(f_0 + f_lead * G1_k)^{-1} * f_1
    G1 = zeros(T, n, n)
    converged = false
    for iter in 1:maxiter
        A = f_0 + f_lead * G1
        G1_new = -(A \ f_1)
        diff = maximum(abs.(G1_new - G1))
        G1 = G1_new
        if diff < tol
            converged = true
            break
        end
    end

    # Impact: M = -(f_0 + f_lead * G1)^{-1} * f_eps
    A = f_0 + f_lead * G1
    impact = -(A \ f_eps)

    ev = eigvals(G1)
    (G1=G1, impact=impact, eigenvalues=ev, converged=converged)
end

"""
    solve(spec::DSGESpec{T}; method=:gensys, kwargs...) -> DSGESolution or PerfectForesightPath or PerturbationSolution

Solve a DSGE model.

# Methods
- `:gensys` -- Sims (2002) QZ decomposition (default)
- `:blanchard_kahn` -- Blanchard-Kahn (1980) eigenvalue counting
- `:klein` -- Klein (2000) generalized Schur decomposition
- `:perturbation` -- Higher-order perturbation (Schmitt-Grohe & Uribe 2004); pass `order=2` for second-order
- `:projection` -- Chebyshev collocation (Judd 1998); pass `degree=5` for polynomial degree
- `:pfi` -- Policy Function Iteration / Time Iteration (Coleman 1990); pass `degree=5`, `damping=1.0`
- `:vfi` -- Euler-equation time iteration (Coleman 1990), equivalent to `:pfi` (the name is historical, not value-function iteration); pass `degree=5`, `howard_steps=0`
- `:perfect_foresight` -- deterministic Newton solver
"""
function solve(spec::DSGESpec{T}; method::Symbol=:gensys, kwargs...) where {T<:AbstractFloat}
    if isempty(spec.steady_state)
        if spec.linear
            # Linear models: steady state is all zeros (variables are deviations)
            spec = _update_steady_state(spec, zeros(T, spec.n_endog))
        else
            spec = compute_steady_state(spec)
        end
    end

    if method == :gensys
        ld = linearize(spec)

        # Recover f_0,f_1,f_ε from the canonical form (as klein/bk do); f_lead is
        # not stored losslessly in ld.Pi, so compute it directly.
        f_0 = ld.Gamma0
        f_1 = -ld.Gamma1
        f_ε = -ld.Psi
        f_lead = _dsge_jacobian(spec, spec.steady_state, :lead)

        div = T(get(kwargs, :div, 1 + 1e-8))
        cluster_tol = T(get(kwargs, :cluster_tol, 1e-6))

        # Companion-QZ for correct determinacy + a robust solution fallback
        qz_core = _solve_qz_quadratic(f_0, f_1, f_lead, f_ε; div=div, cluster_tol=cluster_tol)

        # Primary solution via undetermined coefficients (robust to many static vars). Accept
        # it only if the residual/convergence AND stability hold: a converged-but-explosive
        # solvent (max|eigvals(G1)| ≥ div) must fall back to the stable QZ solvent (#213).
        uc_ok = false
        local uc_result
        try
            uc_result = _solve_undetermined_coefficients(spec)
            resid = (f_0 + f_lead * uc_result.G1) * uc_result.G1 + f_1
            uc_ok = maximum(abs.(resid)) < T(1e-8) && uc_result.converged &&
                    maximum(abs.(eigvals(uc_result.G1)); init=zero(T)) < div
        catch
            # UC failed (SingularException or non-convergence) — fall through to qz_core
        end

        G1 = uc_ok ? uc_result.G1 : qz_core.G
        impact = uc_ok ? uc_result.impact : qz_core.impact

        # Constants: C_sol = (I - G1)·y_ss, y_ss = (f₀+f₁+f_lead)⁻¹·C. Include the lead
        # block (Γ0-Γ1 = f₀+f₁ omits it) or the SS is wrong for forward models (S-06 / #114).
        if norm(ld.C) > eps(T)
            y_bar = real(Vector{T}(complex(f_0 + f_1 + f_lead) \ complex(ld.C)))
            C_sol = (I - G1) * y_bar
        else
            C_sol = zeros(T, spec.n_endog)
        end

        eigenvalues = Vector{ComplexF64}(eigvals(G1))
        return DSGESolution{T}(
            G1, impact, Vector{T}(C_sol), qz_core.eu,
            :gensys, eigenvalues, spec, ld
        )
    elseif method == :blanchard_kahn
        ld = linearize(spec)
        return blanchard_kahn(ld, spec; div=get(kwargs, :div, 1.0 + 1e-8),
                              cluster_tol=get(kwargs, :cluster_tol, 1e-6))
    elseif method == :perfect_foresight
        return perfect_foresight(spec; kwargs...)
    elseif method == :klein
        ld = linearize(spec)
        return klein(ld, spec; div=get(kwargs, :div, 1.0 + 1e-8),
                     cluster_tol=get(kwargs, :cluster_tol, 1e-6))
    elseif method == :perturbation
        order = get(kwargs, :order, 2)
        return perturbation_solver(spec; order=order)
    elseif method == :projection
        return collocation_solver(spec; kwargs...)
    elseif method == :pfi
        return pfi_solver(spec; kwargs...)
    elseif method == :vfi
        return vfi_solver(spec; kwargs...)
    else
        throw(ArgumentError("method must be :gensys, :blanchard_kahn, :klein, :perturbation, :projection, :pfi, :vfi, or :perfect_foresight"))
    end
end
