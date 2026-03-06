# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Second- and third-order perturbation solver for DSGE models.

Implements the Schmitt-Grohe & Uribe (2004) method for computing second-order
approximations to the policy functions of DSGE models. The decision rules take
the form:

    z_t = z_ss + f_v * v_t + (1/2) * f_vv * (v_t kron v_t) + (1/2) * f_ss * sigma^2

where v_t = [x_{t-1}; epsilon_t] is the innovations vector containing lagged states
and current shocks, and f_v, f_vv, f_ss are the first-order, second-order, and
variance correction coefficients respectively.

References:
- Schmitt-Grohe, S. & Uribe, M. (2004). "Solving Dynamic General Equilibrium Models
  Using a Second-Order Approximation to the Policy Function." Journal of Economic
  Dynamics and Control, 28(4), 755-775.
- Kim, J., Kim, S., Schaumburg, E., & Sims, C. A. (2008). "Calculating and Using
  Second-Order Accurate Solutions of Discrete Time Dynamic Equilibrium Models."
  Journal of Economic Dynamics and Control, 32(11), 3397-3414.
"""

# =============================================================================
# perturbation_solver — main entry point
# =============================================================================

"""
    perturbation_solver(spec::DSGESpec{T}; order::Int=2, method::Symbol=:gensys) → PerturbationSolution{T}

Compute a perturbation approximation to the policy functions of a DSGE model.

# Arguments
- `spec` — DSGE model specification (must have computed steady state)

# Keywords
- `order::Int=2` — perturbation order (1, 2, or 3)
- `method::Symbol=:gensys` — first-order solver (`:gensys` or `:blanchard_kahn`)

# Returns
A `PerturbationSolution{T}` containing the policy function coefficients:
- Order 1: `gx` (ny x nv), `hx` (nx x nv) where nv = nx + n_epsilon
- Order 2: additionally `gxx`, `hxx` (nv^2 columns), `g_sigma_sigma`, `h_sigma_sigma`
- Order 3: additionally `gxxx`, `hxxx` (nv^3 columns), `gσσx`, `hσσx`, `gσσσ`, `hσσσ`

# Algorithm
1. Solve the first-order system via gensys/BK to get G1, impact
2. Partition variables into states (x) and controls (y) via Gamma1
3. Build the innovations vector v = [x_{t-1}; epsilon_t] and mapping matrices
4. For order >= 2: compute Hessians, assemble the Kronecker system, solve for f_vv
5. Solve the sigma^2 correction: f_sigma_sigma

# References
- Schmitt-Grohe & Uribe (2004), JEDC 28(4), 755-775.
- Kim, Kim, Schaumburg & Sims (2008), JEDC 32(11), 3397-3414.
"""
function perturbation_solver(spec::DSGESpec{T};
                              order::Int=2,
                              method::Symbol=:gensys) where {T<:AbstractFloat}
    order in (1, 2, 3) || throw(ArgumentError("order must be 1, 2, or 3; got $order"))
    isempty(spec.steady_state) &&
        throw(ArgumentError("Must compute steady state first (call compute_steady_state)"))

    # -------------------------------------------------------------------------
    # Step 1: Solve first-order system
    # -------------------------------------------------------------------------
    ld = linearize(spec)

    first_order = if method == :gensys
        gensys(ld.Gamma0, ld.Gamma1, ld.C, ld.Psi, ld.Pi)
    elseif method == :blanchard_kahn
        # blanchard_kahn returns a DSGESolution; extract the fields we need
        bk_sol = blanchard_kahn(ld, spec)
        (G1=bk_sol.G1, impact=bk_sol.impact, C_sol=bk_sol.C_sol,
         eu=bk_sol.eu, eigenvalues=bk_sol.eigenvalues)
    else
        throw(ArgumentError("method must be :gensys or :blanchard_kahn; got $method"))
    end

    G1 = first_order.G1         # n x n
    impact = first_order.impact # n x n_epsilon
    eu = first_order.eu

    n = spec.n_endog
    n_eps = spec.n_exog

    # -------------------------------------------------------------------------
    # Step 2: Partition into states and controls
    # -------------------------------------------------------------------------
    state_idx, control_idx = _state_control_indices(ld)
    nx = length(state_idx)
    ny = length(control_idx)
    nv = nx + n_eps  # dimension of innovations vector v = [x_{t-1}; epsilon_t]

    # -------------------------------------------------------------------------
    # Step 3: Build first-order coefficient matrices in v-space
    # -------------------------------------------------------------------------
    # From the first-order solution y_t = G1 * y_{t-1} + impact * eps_t,
    # extract the partitioned form:
    #   x_t = hx * v_t  (state transition in v-space)
    #   y_t = gx * v_t  (control decision rule in v-space)
    # where v_t = [x_{t-1}; eps_t]

    # hx: state part (nx x nv)
    hx = zeros(T, nx, nv)
    if nx > 0
        hx[:, 1:nx] = G1[state_idx, state_idx]      # state-to-state
        hx[:, nx+1:nv] = impact[state_idx, :]        # shock-to-state
    end

    # gx: control part (ny x nv)
    gx = zeros(T, ny, nv)
    if ny > 0
        gx[:, 1:nx] = G1[control_idx, state_idx]     # state-to-control
        gx[:, nx+1:nv] = impact[control_idx, :]       # shock-to-control
    end

    # fv: full first-order in v-space (n x nv), ordered [state; control]
    # Reorder so rows correspond to [state_idx; control_idx]
    fv = zeros(T, n, nv)
    for (row, si) in enumerate(state_idx)
        fv[si, :] = hx[row, :]
    end
    for (row, ci) in enumerate(control_idx)
        fv[ci, :] = gx[row, :]
    end

    # eta: shock loading in v-space (nv x n_eps)
    # v_t = [x_{t-1}; eps_t], so shocks load into the bottom block
    eta = zeros(T, nv, n_eps)
    if n_eps > 0
        eta[nx+1:nv, :] = Matrix{T}(I, n_eps, n_eps)
    end

    # M: augmented transition matrix (nv x nv)
    # v_{t+1} = M * v_t + eta * eps_{t+1}
    # The state part of v_{t+1} is x_t = hx * v_t
    # The shock part of v_{t+1} is eps_{t+1} (independent, zero in expectation)
    M = zeros(T, nv, nv)
    if nx > 0
        M[1:nx, :] = hx   # state transition
    end
    # Bottom block (shock rows) is zero: E[eps_{t+1}|v_t] = 0

    # -------------------------------------------------------------------------
    # Build first-order solution and return if order == 1
    # -------------------------------------------------------------------------
    if order == 1
        return PerturbationSolution{T}(
            1,
            gx, hx,
            nothing, nothing, nothing, nothing,          # second-order
            nothing, nothing, nothing, nothing, nothing, nothing,  # third-order
            eta, spec.steady_state, state_idx, control_idx,
            eu, :perturbation, spec, ld
        )
    end

    # -------------------------------------------------------------------------
    # Step 4: Second-order — compute Hessians and RHS of Kronecker system
    # -------------------------------------------------------------------------
    y_ss = spec.steady_state
    hessians = _compute_all_hessians(spec, y_ss)

    # Build argument-slot-to-v mapping matrices:
    #   :lag     -> L_v (n x nv): lag variables come from x-part of v
    #   :current -> C_v = fv (n x nv): current z_t = f_v * v_t
    #   :lead    -> F_v = fv * M (n x nv): z_{t+1} = f_v * v_{t+1} = f_v * M * v_t
    #   :shock   -> E_v (n_eps x nv): shocks are eps-part of v

    L_v = zeros(T, n, nv)
    for (local_j, si) in enumerate(state_idx)
        L_v[si, local_j] = one(T)   # lag var si maps to x_{local_j} in v
    end

    C_v = fv                         # n x nv
    F_v = fv * M                     # n x nv
    E_v = zeros(T, n_eps, nv)
    if n_eps > 0
        E_v[:, nx+1:nv] = Matrix{T}(I, n_eps, n_eps)
    end

    # Map from slot symbol to derivative-argument-space-to-v mapping
    slot_to_Dv = Dict{Symbol, Matrix{T}}(
        :lag     => L_v,
        :current => C_v,
        :lead    => F_v,
        :shock   => E_v,
    )

    # Canonical ordering of slots (same as in _compute_all_hessians)
    slots = [:current, :lag, :lead, :shock]

    # Accumulate RHS: n x nv^2 matrix
    # RHS[i, (p-1)*nv + q] = sum over all (a,b) pairs of H_{a,b}[i,j,k] * Da[j,p] * Db[k,q]
    RHS = zeros(T, n, nv * nv)

    for (ai, a) in enumerate(slots)
        Da = slot_to_Dv[a]   # dim_a x nv
        dim_a = size(Da, 1)
        for (bi, b) in enumerate(slots)
            Db = slot_to_Dv[b]   # dim_b x nv
            dim_b = size(Db, 1)

            # Look up the Hessian tensor: try (a,b) then (b,a) with transposition
            H = _lookup_hessian(hessians, a, b)
            if H === nothing
                continue  # no second-order terms for this pair
            end

            # Accumulate: for each equation i, sum over derivative indices j,k
            _accumulate_kronecker_rhs!(RHS, H, Da, Db, n, dim_a, dim_b, nv)
        end
    end

    # -------------------------------------------------------------------------
    # Step 5: Solve the second-order Kronecker system
    # -------------------------------------------------------------------------
    # The system is: [I_nv^2 kron f_c + kron(M,M)' kron f_f] * vec(f_vv) = -vec(RHS)
    #
    # where f_c = df/dz_t and f_f = df/dz_{t+1} are the Jacobians of the residual
    # with respect to current and lead variables.
    #
    # In Sims notation:
    #   Gamma0 * z_t = Gamma1 * z_{t-1} + Psi * eps_t + Pi * eta_t
    #   => f(z_{t+1}, z_t, z_{t-1}, eps_t) = 0
    #
    # The Jacobians we need: f_c = -(Gamma0^{-1} style), but more precisely,
    # f_c is the derivative of the residual w.r.t. z_t (= f_0 from linearize)
    # and f_f is the derivative w.r.t. z_{t+1} (= f_lead from linearize).

    f_c = _dsge_jacobian(spec, y_ss, :current)    # n x n  (df/dz_t)
    f_f = _dsge_jacobian(spec, y_ss, :lead)        # n x n  (df/dz_{t+1})

    # LHS coefficient matrix: size (n * nv^2) x (n * nv^2)
    # [I_{nv^2} kron f_c + kron(M,M) kron f_f] * vec(fvv) = -vec(RHS)
    #
    # More carefully: fvv is n x nv^2. The system stacks all n equations.
    # For each equation i and each (p,q) pair:
    #   sum_r f_c[i,r] * fvv[r, (p-1)*nv+q]
    #   + sum_r f_f[i,r] * sum_{s,t} fvv[r, (s-1)*nv+t] * M[s,p] * M[t,q]
    #   = -RHS[i, (p-1)*nv+q]
    #
    # In Kronecker form: (I_{nv^2} kron f_c + kron(M',M') kron f_f) vec(fvv) = -vec(RHS)
    # Since kron(M,M) maps (v kron v)_{t+1} to (v kron v)_t, the correct form is:
    # (I_{nv^2} kron f_c + kron(M,M)^T kron f_f) * vec(fvv) = -vec(RHS)
    #
    # Actually, following SGU(2004) more carefully:
    # The equation for the second-order terms is:
    #   f_c * fvv + f_f * fvv * kron(M, M) = -RHS
    # where fvv is n x nv^2, RHS is n x nv^2.
    #
    # Vectorizing: vec(A * X * B) = (B' kron A) * vec(X)
    # vec(f_c * fvv) = (I_{nv^2} kron f_c) * vec(fvv)
    # vec(f_f * fvv * kron(M,M)) = (kron(M,M)' kron f_f) * vec(fvv)
    #
    # So: [(I_{nv^2} kron f_c) + (kron(M,M)' kron f_f)] * vec(fvv) = -vec(RHS)

    nv2 = nv * nv

    if nv > 0
        MkM = kron(M, M)  # nv^2 x nv^2
        LHS = kron(Matrix{T}(I, nv2, nv2), f_c) + kron(MkM', f_f)
        fvv_vec = LHS \ (-vec(RHS))
        fvv = reshape(fvv_vec, n, nv2)
    else
        fvv = zeros(T, n, 0)
    end

    # -------------------------------------------------------------------------
    # Step 6: Solve the sigma^2 correction
    # -------------------------------------------------------------------------
    # (f_c + f_f) * f_sigma_sigma = -f_f * fvv * vec(eta * eta')
    #
    # eta * eta' is nv x nv. Its vec is nv^2 x 1.
    # fvv * vec(eta * eta') gives an n x 1 vector.

    eta_outer = eta * eta'   # nv x nv
    if nv > 0
        rhs_sigma = -f_f * fvv * vec(eta_outer)
    else
        rhs_sigma = zeros(T, n)
    end

    fc_plus_ff = f_c + f_f
    f_sigma_sigma = fc_plus_ff \ rhs_sigma   # n x 1

    # -------------------------------------------------------------------------
    # Step 7: Partition fvv and f_sigma_sigma into state/control blocks
    # -------------------------------------------------------------------------
    hxx = fvv[state_idx, :]       # nx x nv^2
    gxx = fvv[control_idx, :]     # ny x nv^2
    h_sigma_sigma = f_sigma_sigma[state_idx]    # nx
    g_sigma_sigma = f_sigma_sigma[control_idx]  # ny

    # Handle edge case: if nx == 0, produce empty matrices
    if nx == 0
        hxx = zeros(T, 0, nv2)
        h_sigma_sigma = zeros(T, 0)
    end
    if ny == 0
        gxx = zeros(T, 0, nv2)
        g_sigma_sigma = zeros(T, 0)
    end

    # -------------------------------------------------------------------------
    # Return for order == 2
    # -------------------------------------------------------------------------
    if order == 2
        return PerturbationSolution{T}(
            2,
            gx, hx,
            gxx, hxx, g_sigma_sigma, h_sigma_sigma,         # second-order
            nothing, nothing, nothing, nothing, nothing, nothing,  # third-order
            eta, spec.steady_state, state_idx, control_idx,
            eu, :perturbation, spec, ld
        )
    end

    # =====================================================================
    # Third-order perturbation (order == 3)
    # =====================================================================

    # -------------------------------------------------------------------------
    # Step 8: Compute third derivatives and build RHS_3
    # -------------------------------------------------------------------------
    third_derivs = _compute_all_third_derivatives(spec, y_ss)

    nv3 = nv * nv * nv
    RHS_3 = zeros(T, n, nv3)

    # (A) Pure third-derivative contributions:
    #     RHS_3[i, ((p-1)*nv + q-1)*nv + r] += D3_{a,b,c}[i,j,k,l] * Da[j,p] * Db[k,q] * Dc[l,r]
    for (ai, a) in enumerate(slots)
        Da = slot_to_Dv[a]
        dim_a = size(Da, 1)
        for (bi, b) in enumerate(slots)
            Db = slot_to_Dv[b]
            dim_b = size(Db, 1)
            for (ci, c) in enumerate(slots)
                Dc = slot_to_Dv[c]
                dim_c = size(Dc, 1)

                D3 = _lookup_third_derivative(third_derivs, a, b, c)
                if D3 === nothing
                    continue
                end

                _accumulate_third_order_rhs!(RHS_3, D3, Da, Db, Dc, n, dim_a, dim_b, dim_c, nv)
            end
        end
    end

    # (B) Mixed Hessian x second-order interaction terms:
    #     3 permutations of D2_{a,b}[i,j,k] * fvv[j, (s-1)*nv + t] * Da_2nd[s,p] * Db[k,q] * Dc_perm[t,r]
    # where Da_2nd maps the fvv second-order result back through the appropriate slot.
    #
    # The second-order terms interact through:
    #   current slot: fvv (the second-order polynomial) → D2_current = fvv  (n × nv²)
    #   lead slot:    fvv * kron(M,M) (propagated through transition) → D2_lead
    #   lag slot:     no second-order (lag is x_{t-1}, exogenous to the policy)
    #   shock slot:   no second-order (shocks are exogenous)

    D2_current = fvv                    # n × nv²
    D2_lead = fvv * MkM                 # n × nv² (fvv * kron(M,M))

    # For each Hessian H_{a,b}, the mixed contribution has the form:
    # H_{a,b}[i,j,k] * fvv_a[j, (s-1)*nv + t] * mapping_for_first_index[s, p] * Db[k, q]
    # where fvv_a is the appropriate second-order applied to slot a.
    # This needs to be accumulated for all 3 cyclic permutations of (result_col, Da, Db).

    for (ai, a) in enumerate(slots)
        # Only :current and :lead have second-order expansions
        D2_a = if a == :current
            D2_current
        elseif a == :lead
            D2_lead
        else
            nothing
        end
        D2_a === nothing && continue

        dim_a = size(slot_to_Dv[a], 1)

        for (bi, b) in enumerate(slots)
            Db = slot_to_Dv[b]
            dim_b = size(Db, 1)

            H = _lookup_hessian(hessians, a, b)
            if H === nothing
                continue
            end

            # For each column_index r = ((p-1)*nv + q-1)*nv + s in nv³ layout,
            # the mixed term contracts H[i,j,k] with D2_a[j, col_vv] and Db[k, ...]
            # across all 3 permutations of v⊗v⊗v indices.
            _accumulate_mixed_rhs!(RHS_3, H, D2_a, Db, slot_to_Dv[a], n, dim_a, dim_b, nv)
        end
    end

    # Also need second-order sigma correction interaction:
    # H_{a,b}[i,j,k] * f_sigma_sigma_a[j] * Db[k, r] (contributes to RHS_σσv, handled below)

    # -------------------------------------------------------------------------
    # Step 9: Solve the third-order Kronecker system for fvvv
    # -------------------------------------------------------------------------
    # f_c * fvvv + f_f * fvvv * kron(M,M,M) = -RHS_3
    # Vectorizing: [(I_{nv³} ⊗ f_c) + (kron(M,M,M)' ⊗ f_f)] * vec(fvvv) = -vec(RHS_3)

    if nv > 0
        MkMkM = kron(MkM, M)  # nv³ × nv³
        LHS_3 = kron(Matrix{T}(I, nv3, nv3), f_c) + kron(MkMkM', f_f)
        fvvv_vec = LHS_3 \ (-vec(RHS_3))
        fvvv = reshape(fvvv_vec, n, nv3)
    else
        fvvv = zeros(T, n, 0)
    end

    # -------------------------------------------------------------------------
    # Step 10: σ²v correction (f_σσv)
    # -------------------------------------------------------------------------
    # [I_nv ⊗ f_c + M' ⊗ f_f] vec(f_σσv) = -vec(RHS_σσv)
    # where RHS_σσv[i, p] = Σ_{q,r} fvvv[i, ((q-1)*nv+r-1)*nv+p] * eta_outer[q,r]
    #                      + Hessian correction from f_sigma_sigma interaction

    RHS_sigma_v = zeros(T, n, nv)

    # Contraction of fvvv with eta_outer: sum over q,r of fvvv[i, ((q-1)*nv+r-1)*nv+p] * eta_outer[q,r]
    if nv > 0
        for p in 1:nv
            for q in 1:nv
                for r in 1:nv
                    col3 = ((q - 1) * nv + (r - 1)) * nv + p
                    eo_qr = eta_outer[q, r]
                    iszero(eo_qr) && continue
                    for i in 1:n
                        RHS_sigma_v[i, p] += fvvv[i, col3] * eo_qr
                    end
                end
            end
        end
    end

    # Hessian correction: H_{a,b}[i,j,k] * f_sigma_sigma_a[j] * Db[k,p]
    # where f_sigma_sigma_a is the sigma² correction applied to slot a
    # Only :current and :lead have sigma corrections
    f_sigma_sigma_current = f_sigma_sigma          # n-vector
    f_sigma_sigma_lead = f_sigma_sigma              # same at steady state (deterministic SS)

    for (ai, a) in enumerate(slots)
        fss_a = if a == :current
            f_sigma_sigma_current
        elseif a == :lead
            f_sigma_sigma_lead
        else
            nothing
        end
        fss_a === nothing && continue

        dim_a = size(slot_to_Dv[a], 1)

        for (bi, b) in enumerate(slots)
            Db = slot_to_Dv[b]
            dim_b = size(Db, 1)

            H = _lookup_hessian(hessians, a, b)
            H === nothing && continue

            # H[i,j,k] * fss_a[j] * Db[k,p]
            for p in 1:nv
                for i in 1:n
                    s = zero(T)
                    for j in 1:dim_a
                        fss_j = fss_a[j]
                        iszero(fss_j) && continue
                        for k in 1:dim_b
                            Db_kp = Db[k, p]
                            iszero(Db_kp) && continue
                            s += H[i, j, k] * fss_j * Db_kp
                        end
                    end
                    RHS_sigma_v[i, p] += s
                end
            end
        end
    end

    # Solve: [I_nv ⊗ f_c + M' ⊗ f_f] vec(f_σσv) = -vec(RHS_σσv)
    if nv > 0
        LHS_sv = kron(Matrix{T}(I, nv, nv), f_c) + kron(M', f_f)
        f_sigma_v_vec = LHS_sv \ (-vec(RHS_sigma_v))
        f_sigma_v = reshape(f_sigma_v_vec, n, nv)
    else
        f_sigma_v = zeros(T, n, 0)
    end

    # -------------------------------------------------------------------------
    # Step 11: σ³ correction — zero for Gaussian shocks
    # -------------------------------------------------------------------------
    f_sigma3 = zeros(T, n)

    # -------------------------------------------------------------------------
    # Step 12: Partition into state/control blocks
    # -------------------------------------------------------------------------
    hxxx = fvvv[state_idx, :]            # nx × nv³
    gxxx = fvvv[control_idx, :]          # ny × nv³
    h_sigma_x = f_sigma_v[state_idx, :]  # nx × nv
    g_sigma_x = f_sigma_v[control_idx, :] # ny × nv
    h_sigma3 = f_sigma3[state_idx]       # nx
    g_sigma3 = f_sigma3[control_idx]     # ny

    # Handle edge cases
    if nx == 0
        hxxx = zeros(T, 0, nv3)
        h_sigma_x = zeros(T, 0, nv)
        h_sigma3 = zeros(T, 0)
    end
    if ny == 0
        gxxx = zeros(T, 0, nv3)
        g_sigma_x = zeros(T, 0, nv)
        g_sigma3 = zeros(T, 0)
    end

    return PerturbationSolution{T}(
        3,
        gx, hx,
        gxx, hxx, g_sigma_sigma, h_sigma_sigma,         # second-order
        gxxx, hxxx, g_sigma_x, h_sigma_x, g_sigma3, h_sigma3,  # third-order
        eta, spec.steady_state, state_idx, control_idx,
        eu, :perturbation, spec, ld
    )
end


# =============================================================================
# Helper: look up Hessian with automatic transposition
# =============================================================================

"""
    _lookup_hessian(hessians, a, b) → Union{Nothing, Array{T,3}}

Look up Hessian H_{a,b} from the dictionary computed by `_compute_all_hessians`.
If (a,b) is stored directly, return it. If (b,a) is stored, return the transposed
tensor (swap axes 2 and 3). Returns `nothing` if neither is found.
"""
function _lookup_hessian(hessians::Dict{Tuple{Symbol,Symbol}, Array{T,3}},
                          a::Symbol, b::Symbol) where {T}
    if haskey(hessians, (a, b))
        return hessians[(a, b)]
    elseif haskey(hessians, (b, a))
        # H_{b,a}[i,j,k] is stored; H_{a,b}[i,j,k] = H_{b,a}[i,k,j]
        H_ba = hessians[(b, a)]
        return permutedims(H_ba, (1, 3, 2))
    else
        return nothing
    end
end


# =============================================================================
# Helper: accumulate Kronecker RHS contribution from one Hessian block
# =============================================================================

"""
    _accumulate_kronecker_rhs!(RHS, H, Da, Db, n, dim_a, dim_b, nv)

Accumulate the contribution of one Hessian block H (n x dim_a x dim_b) and
its associated mapping matrices Da (dim_a x nv), Db (dim_b x nv) into the
RHS matrix (n x nv^2).

For each equation i and each (p,q) pair in the innovations-space Kronecker product:
    RHS[i, (p-1)*nv + q] += sum_{j,k} H[i,j,k] * Da[j,p] * Db[k,q]

This is equivalent to: RHS += H_matricized * kron(Da, Db) but computed without
forming the full Kronecker product.
"""
function _accumulate_kronecker_rhs!(RHS::Matrix{T},
                                     H::Array{T,3},
                                     Da::Matrix{T}, Db::Matrix{T},
                                     n::Int, dim_a::Int, dim_b::Int,
                                     nv::Int) where {T}
    # Efficient implementation: contract H with Da and Db
    # Step 1: Contract H with Da along axis 2: tmp[i,p,k] = sum_j H[i,j,k] * Da[j,p]
    # Step 2: Contract tmp with Db along axis 3: result[i,p,q] = sum_k tmp[i,p,k] * Db[k,q]
    # Step 3: Map (p,q) -> column index (p-1)*nv + q

    # For small models, direct loop is efficient and avoids large temporary allocations.
    # For medium models, we use a matricized approach.

    if nv == 0
        return
    end

    # Matricized approach: reshape H to (n, dim_a * dim_b), compute
    # H_mat * kron(Da, Db) which gives (n, nv * nv)
    # But kron(Da, Db) is (dim_a * dim_b) x (nv * nv), which can be large.
    # Instead, use the two-step contraction.

    # Step 1: H_Da[i, :, k] = H[i, :, k]' * Da[:, :] for each k
    # => H_Da is n x nv x dim_b
    # Reshape H to (n * dim_b, dim_a) and multiply by Da to get (n * dim_b, nv)
    # Then reshape to (n, dim_b, nv) and permute.

    # More directly: H is n x dim_a x dim_b
    # H_reshaped = reshape(H, n, dim_a, dim_b)
    # Contract axis 2 with Da:
    #   tmp[i, p, k] = sum_j H[i, j, k] * Da[j, p]

    # Using matrix multiplication: for each k, tmp[:, :, k] = H[:, :, k] * Da
    # But this requires n * dim_b matrix multiplications.

    # Better: reshape H to (n, dim_a * dim_b), multiply by kron(I_dim_b, Da)
    # That's expensive. Let's use the loop approach for small/medium models.

    # Threshold: if nv <= 30, use direct loops; otherwise use batched matmul
    if nv <= 30 || (dim_a * dim_b <= 400)
        # Direct accumulation with loops
        @inbounds for p in 1:nv
            for q in 1:nv
                col = (p - 1) * nv + q
                for i in 1:n
                    s = zero(T)
                    for j in 1:dim_a
                        Da_jp = Da[j, p]
                        iszero(Da_jp) && continue
                        for k in 1:dim_b
                            Db_kq = Db[k, q]
                            iszero(Db_kq) && continue
                            s += H[i, j, k] * Da_jp * Db_kq
                        end
                    end
                    RHS[i, col] += s
                end
            end
        end
    else
        # Batched approach for larger models:
        # tmp = H matricized * kron(Da, Db)
        # H_mat is n x (dim_a * dim_b), kron(Da, Db) is (dim_a * dim_b) x nv^2
        H_mat = reshape(H, n, dim_a * dim_b)
        DaDb = kron(Da, Db)  # (dim_a * dim_b) x nv^2
        RHS .+= H_mat * DaDb
    end

    return nothing
end


# =============================================================================
# Third-order helpers
# =============================================================================

"""
    _compute_all_third_derivatives(spec::DSGESpec{T}, y_ss::Vector{T})
        → Dict{Tuple{Symbol,Symbol,Symbol}, Array{T,4}}

Compute all 20 unique third-derivative tensors for the 4 argument slots
{current, lag, lead, shock}. Only canonical orderings (a ≤ b ≤ c) are computed;
other orderings can be obtained by axis permutation via `_lookup_third_derivative`.
"""
function _compute_all_third_derivatives(spec::DSGESpec{T}, y_ss::Vector{T}) where {T}
    slots = [:current, :lag, :lead, :shock]
    result = Dict{Tuple{Symbol,Symbol,Symbol}, Array{T,4}}()

    for a in 1:length(slots)
        for b in a:length(slots)
            for c in b:length(slots)
                s1, s2, s3 = slots[a], slots[b], slots[c]
                result[(s1, s2, s3)] = _third_derivative(spec, y_ss, s1, s2, s3)
            end
        end
    end

    result
end

"""
    _lookup_third_derivative(derivs, a, b, c) → Union{Nothing, Array{T,4}}

Look up third derivative D3_{a,b,c} from the dictionary of canonical triples.
If the canonical ordering differs, permute axes 2-4 accordingly.
Returns `nothing` if not found (should not happen for valid slot combinations).
"""
function _lookup_third_derivative(derivs::Dict{Tuple{Symbol,Symbol,Symbol}, Array{T,4}},
                                   a::Symbol, b::Symbol, c::Symbol) where {T}
    # Try all 6 permutations and find the canonical one
    slots_order = Dict(:current => 1, :lag => 2, :lead => 3, :shock => 4)
    oa, ob, oc = slots_order[a], slots_order[b], slots_order[c]

    # Sort to find canonical key
    sorted = sort([(oa, a, 1), (ob, b, 2), (oc, c, 3)], by=x->x[1])
    canonical_key = (sorted[1][2], sorted[2][2], sorted[3][2])

    if !haskey(derivs, canonical_key)
        return nothing
    end

    D3_canonical = derivs[canonical_key]

    # Determine the permutation from canonical → requested
    # canonical axes are (1=eq, 2=sorted[1], 3=sorted[2], 4=sorted[3])
    # requested axes are (1=eq, 2=a, 3=b, 4=c)
    # We need: perm such that canonical[perm[i]] = requested[i] for axes 2,3,4

    # Build the inverse mapping: where does each requested axis come from in canonical?
    orig_positions = [sorted[1][3], sorted[2][3], sorted[3][3]]  # original positions of sorted elements
    # orig_positions[k] tells us which original axis (1=a, 2=b, 3=c) ended up at canonical position k+1

    # We want perm such that D3_requested[i,j,k,l] = D3_canonical[i, perm_map...]
    # If original axis `m` ended up at canonical position `p+1`, then
    # to get requested axis m, we read from canonical axis p+1.

    # Map from requested axis (1=a,2=b,3=c) to canonical axis position (2,3,4)
    perm = zeros(Int, 3)
    for k in 1:3
        # canonical position k+1 corresponds to original axis orig_positions[k]
        perm[orig_positions[k]] = k + 1
    end

    if perm == [2, 3, 4]
        return D3_canonical  # no permutation needed
    else
        return permutedims(D3_canonical, (1, perm[1], perm[2], perm[3]))
    end
end

"""
    _accumulate_third_order_rhs!(RHS_3, D3, Da, Db, Dc, n, dim_a, dim_b, dim_c, nv)

Accumulate the contribution of one third-derivative block D3 (n × dim_a × dim_b × dim_c)
and its associated mapping matrices Da, Db, Dc into RHS_3 (n × nv³).

For each equation i and each (p,q,r) triple:
    RHS_3[i, ((p-1)*nv + q-1)*nv + r] += Σ_{j,k,l} D3[i,j,k,l] * Da[j,p] * Db[k,q] * Dc[l,r]
"""
function _accumulate_third_order_rhs!(RHS_3::Matrix{T},
                                       D3::Array{T,4},
                                       Da::Matrix{T}, Db::Matrix{T}, Dc::Matrix{T},
                                       n::Int, dim_a::Int, dim_b::Int, dim_c::Int,
                                       nv::Int) where {T}
    nv == 0 && return nothing

    # Direct loop approach (efficient for small/medium DSGE models)
    @inbounds for p in 1:nv
        for q in 1:nv
            pq_base = ((p - 1) * nv + (q - 1)) * nv
            for r in 1:nv
                col = pq_base + r
                for i in 1:n
                    s = zero(T)
                    for j in 1:dim_a
                        Da_jp = Da[j, p]
                        iszero(Da_jp) && continue
                        for k in 1:dim_b
                            Db_kq = Db[k, q]
                            iszero(Db_kq) && continue
                            Da_Db = Da_jp * Db_kq
                            for l in 1:dim_c
                                Dc_lr = Dc[l, r]
                                iszero(Dc_lr) && continue
                                s += D3[i, j, k, l] * Da_Db * Dc_lr
                            end
                        end
                    end
                    RHS_3[i, col] += s
                end
            end
        end
    end

    return nothing
end

"""
    _accumulate_mixed_rhs!(RHS_3, H, D2_a, Db, Da_map, n, dim_a, dim_b, nv)

Accumulate mixed Hessian × second-order interaction terms into RHS_3 (n × nv³).

The mixed term arises from the chain rule: H_{a,b}[i,j,k] contracts with
the second-order expansion D2_a[j, (s-1)*nv+t] and the first-order mapping Db[k, ...].

Three permutations of the (p,q,r) indices are accumulated:
  - (s,t,r): D2_a on first two v-indices, Db on third
  - (s,r,t): D2_a straddling first and third, Db on second
  - (r,s,t): D2_a on last two v-indices, Db on first
"""
function _accumulate_mixed_rhs!(RHS_3::Matrix{T},
                                 H::Array{T,3},
                                 D2_a::Matrix{T},
                                 Db::Matrix{T},
                                 Da_map::Matrix{T},
                                 n::Int, dim_a::Int, dim_b::Int,
                                 nv::Int) where {T}
    nv == 0 && return nothing

    # Pre-contract: tmp[i, col_vv] = Σ_j H[i,j,k] * ... for each k
    # But simpler: for each k, compute H_contracted[i, (s-1)*nv+t] = Σ_j H[i,j,k] * D2_a[j, (s-1)*nv+t]
    # Then multiply by Db[k, r] and accumulate into the 3 permutation slots.

    # For efficiency, pre-compute the contraction of H with D2_a:
    # contracted[i, k, st] = Σ_j H[i,j,k] * D2_a[j, st]
    # This is equivalent to: for each k, contracted[:, k, :] = H[:, :, k] * D2_a (if H were reshaped)

    nv2 = nv * nv

    # Compute H_D2[i, k, st] = Σ_j H[i,j,k] * D2_a[j,st]
    H_D2 = zeros(T, n, dim_b, nv2)
    @inbounds for st in 1:nv2
        for k in 1:dim_b
            for i in 1:n
                s_val = zero(T)
                for j in 1:dim_a
                    s_val += H[i, j, k] * D2_a[j, st]
                end
                H_D2[i, k, st] = s_val
            end
        end
    end

    # Now accumulate the 3 permutations into RHS_3
    @inbounds for s in 1:nv
        for t in 1:nv
            st = (s - 1) * nv + t
            for r in 1:nv
                # Permutation 1: (s, t, r) — D2 covers first two indices
                col_str = ((s - 1) * nv + (t - 1)) * nv + r
                # Permutation 2: (s, r, t) — D2 covers first and third
                col_srt = ((s - 1) * nv + (r - 1)) * nv + t
                # Permutation 3: (r, s, t) — D2 covers second and third
                col_rst = ((r - 1) * nv + (s - 1)) * nv + t

                for k in 1:dim_b
                    Db_kr = Db[k, r]
                    iszero(Db_kr) && continue
                    for i in 1:n
                        val = H_D2[i, k, st] * Db_kr
                        RHS_3[i, col_str] += val
                        RHS_3[i, col_srt] += val
                        RHS_3[i, col_rst] += val
                    end
                end
            end
        end
    end

    return nothing
end
