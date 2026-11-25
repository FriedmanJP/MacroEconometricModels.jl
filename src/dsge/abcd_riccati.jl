# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Hansen–Sargent / Fernández-Villaverde, Rubio-Ramírez, Sargent & Watson (2007)
Wold innovations representation via the generalized Schur / Van Dooren Discrete
Algebraic Riccati Equation (DARE) solver with correlated state and observation
disturbances `S = B * D'`.

References:
- Fernández-Villaverde, J., Rubio-Ramírez, J. F., Sargent, T. J., & Watson, M. W. (2007).
  ABCs (and Ds) of Understanding VARs. *American Economic Review*, 97(3), 1021–1026 (eqs. 7–16).
- Hansen, L. P., & Sargent, T. J. (2013). *Recursive Models of Dynamic Linear Economies*.
  Princeton University Press, Ch. 8–9.
- Pappas, T., Laub, A. J., & Sandell, N. R. (1980). On the numerical solution of the
  discrete-time algebraic Riccati equation. *IEEE TAC*, 25(4), 631–641.
"""

# =============================================================================
# InnovationsRepresentation{T}
# =============================================================================

"""
    InnovationsRepresentation{T<:AbstractFloat}

Steady-state Wold innovations representation of an [`ABCDSystem`](@ref) obtained by
solving the filtering Discrete Algebraic Riccati Equation (DARE) with correlated
state-observation shocks (FVRSW 2007, eqs. 7–12):

```math
\\hat{x}_{t+1} = A \\hat{x}_t + K u_{t+1}, \\qquad y_{t+1} = C \\hat{x}_t + u_{t+1}, \\qquad
E[u_t u_t'] = \\Sigma_u = \\hat{D} \\hat{D}'
```

where:
- ``\\hat{x}_t = E[x_t \\mid y^t]`` is the ``n \\times 1`` filtered state estimate
- ``y_{t+1}`` is the ``k \\times 1`` vector of observables
- ``u_{t+1} = y_{t+1} - E[y_{t+1} \\mid y^t]`` is the ``k \\times 1`` Wold innovation vector
- ``A`` is the ``n \\times n`` state transition matrix
- ``K`` is the ``n \\times k`` steady-state Kalman gain
- ``C`` is the ``k \\times n`` observation matrix
- ``\\Sigma_u`` is the ``k \\times k`` symmetric Wold innovation covariance matrix
- ``\\hat{D}`` is the ``k \\times k`` lower-triangular Cholesky factor of ``\\Sigma_u``

Derived aliases `Sigma_state` (`P`), `Omega_innov` (`Sigma_u`), `D_hat` (`Sigma_u_chol`),
`B_hat` (`K * Sigma_u_chol`), and `M_innov` (`A - K * C`) are exposed via `getproperty`.

# Fields
- `A::Matrix{T}` — `n × n` state transition matrix
- `K::Matrix{T}` — `n × k` steady-state Kalman predictor gain (closed-loop `A - K * C`)
- `C::Matrix{T}` — `k × n` observation matrix
- `Sigma_u::Matrix{T}` — `k × k` Wold innovation covariance `C * P * C' + D * D'`
- `Sigma_u_chol::Matrix{T}` — `k × k` lower-triangular Cholesky factor of `Sigma_u`
- `P::Matrix{T}` — `n × n` steady-state state uncertainty covariance `Var(x_t | y^t)`
- `dare_residual::T` — relative Frobenius residual of the Riccati equation
- `invertible::Bool` — `true` when the underlying `ABCDSystem` satisfies shock invertibility
- `information_loss::Vector{T}` — per-observable excess-variance ratio `diag(C P C') ./ diag(Σ_u)`
- `state_names::Vector{String}` — length-`n` state labels
- `obs_names::Vector{String}` — length-`k` observable labels
"""
struct InnovationsRepresentation{T<:AbstractFloat}
    A::Matrix{T}
    K::Matrix{T}
    C::Matrix{T}
    Sigma_u::Matrix{T}
    Sigma_u_chol::Matrix{T}
    P::Matrix{T}
    dare_residual::T
    invertible::Bool
    information_loss::Vector{T}
    state_names::Vector{String}
    obs_names::Vector{String}

    function InnovationsRepresentation{T}(
        A::AbstractMatrix,
        K::AbstractMatrix,
        C::AbstractMatrix,
        Sigma_u::AbstractMatrix,
        Sigma_u_chol::AbstractMatrix,
        P::AbstractMatrix,
        dare_residual::Real,
        invertible::Bool,
        information_loss::AbstractVector,
        state_names::AbstractVector,
        obs_names::AbstractVector,
    ) where {T<:AbstractFloat}
        Am = Matrix{T}(A)
        Km = Matrix{T}(K)
        Cm = Matrix{T}(C)
        Su = Matrix{T}(Sigma_u)
        Suc = Matrix{T}(Sigma_u_chol)
        Pm = Matrix{T}(P)
        il = Vector{T}(information_loss)
        sn = String[string(s) for s in state_names]
        on = String[string(s) for s in obs_names]

        n = size(Am, 1)
        size(Am, 2) == n || throw(ArgumentError("A must be square, got $(size(Am))"))
        k = size(Cm, 1)
        size(Cm, 2) == n || throw(ArgumentError("C must have n=$n columns, got $(size(Cm, 2))"))
        size(Km) == (n, k) || throw(ArgumentError("K must be n×k = ($n, $k), got $(size(Km))"))
        size(Su) == (k, k) || throw(ArgumentError("Sigma_u must be k×k = ($k, $k), got $(size(Su))"))
        size(Suc) == (k, k) || throw(ArgumentError("Sigma_u_chol must be k×k = ($k, $k), got $(size(Suc))"))
        size(Pm) == (n, n) || throw(ArgumentError("P must be n×n = ($n, $n), got $(size(Pm))"))
        length(il) == k || throw(ArgumentError("information_loss must have length k=$k, got $(length(il))"))
        length(sn) == n || throw(ArgumentError("state_names must have length n=$n, got $(length(sn))"))
        length(on) == k || throw(ArgumentError("obs_names must have length k=$k, got $(length(on))"))

        new{T}(Am, Km, Cm, Su, Suc, Pm, T(dare_residual), invertible, il, sn, on)
    end
end

function Base.getproperty(innov::InnovationsRepresentation, s::Symbol)
    if s === :Sigma_state
        return getfield(innov, :P)
    elseif s === :Omega_innov
        return getfield(innov, :Sigma_u)
    elseif s === :D_hat
        return getfield(innov, :Sigma_u_chol)
    elseif s === :B_hat
        return getfield(innov, :K) * getfield(innov, :Sigma_u_chol)
    elseif s === :M_innov
        return getfield(innov, :A) - getfield(innov, :K) * getfield(innov, :C)
    end
    return getfield(innov, s)
end

function Base.propertynames(innov::InnovationsRepresentation, private::Bool=false)
    return (fieldnames(typeof(innov))..., :Sigma_state, :Omega_innov, :D_hat, :B_hat, :M_innov)
end

# =============================================================================
# Display & Reporting
# =============================================================================

function Base.show(io::IO, innov::InnovationsRepresentation{T}) where {T}
    n = size(innov.A, 1)
    k = size(innov.C, 1)
    max_cl = n == 0 ? zero(T) : maximum(abs, eigvals(innov.A - innov.K * innov.C))
    max_loss = isempty(innov.information_loss) ? zero(T) : maximum(innov.information_loss)

    spec_data = Any[
        "States (n)"           n;
        "Observables (k)"      k;
        "Invertible"           _yesno(innov.invertible);
        "Max |eig(A - K C)|"   _fmt(max_cl; digits=6);
        "Max information loss" _fmt(max_loss; digits=6);
        "DARE residual"        _fmt(innov.dare_residual; digits=4);
        "Observable labels"    isempty(innov.obs_names) ? "—" : join(innov.obs_names, ", ");
    ]
    _pretty_table(io, spec_data;
        title = "Hansen–Sargent Wold Innovations Representation",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

"""
    report([io::IO], innov::InnovationsRepresentation)

Print a comprehensive summary table of an [`InnovationsRepresentation`](@ref),
including per-observable information loss, Wold innovation covariance `Σ_u`,
Kalman gain `K`, and state uncertainty covariance `P`.
"""
function report(io::IO, innov::InnovationsRepresentation{T}) where {T}
    show(io, innov)
    n = size(innov.A, 1)
    k = size(innov.C, 1)
    if k > 0
        println(io)
        loss_tbl = Matrix{Any}(undef, k, 3)
        for i in 1:k
            loss_tbl[i, 1] = innov.obs_names[i]
            loss_tbl[i, 2] = _fmt(innov.Sigma_u[i, i]; digits=6)
            loss_tbl[i, 3] = _fmt(innov.information_loss[i]; digits=6)
        end
        _pretty_table(io, loss_tbl;
            title = "Information Loss by Observable",
            column_labels = ["Observable", "Var(u_t)", "Info Loss Share"],
            alignment = [:l, :r, :r],
        )
    end
    if 0 < k <= 12
        println(io)
        _matrix_table(io, innov.Sigma_u, "Wold Innovation Covariance Σ_u";
            row_labels=innov.obs_names, col_labels=innov.obs_names)
    end
    if 0 < n <= 12 && 0 < k <= 12
        println(io)
        _matrix_table(io, innov.K, "Steady-State Kalman Gain K";
            row_labels=innov.state_names, col_labels=innov.obs_names)
    end
    if 0 < n <= 12
        println(io)
        _matrix_table(io, innov.P, "State Uncertainty Covariance P = Var(x_t | y^t)";
            row_labels=innov.state_names, col_labels=innov.state_names)
    end
    return nothing
end

report(innov::InnovationsRepresentation) = report(stdout, innov)

# =============================================================================
# Innovations Representation Solver & Bridges
# =============================================================================

"""
    innovations_representation(abcd::ABCDSystem{T}; rcond_tol=1e-12) -> InnovationsRepresentation{T}
    innovations_representation(sol, observables=Symbol[]; kwargs...) -> InnovationsRepresentation

Compute the Wold innovations representation of an [`ABCDSystem`](@ref) (or solved DSGE,
Perturbation, HA-DSGE, or StateSpace model) by solving the steady-state Discrete
Algebraic Riccati Equation with correlated state-observation disturbances `S = B * D'`:

```math
P = A P A' + B B' - (A P C' + B D')(C P C' + D D')^{-1}(A P C' + B D')'
```
```math
K = (A P C' + B D') \\Sigma_u^{-1}, \\qquad \\Sigma_u = C P C' + D D'
```

where:
- ``P = \\mathrm{Var}(x_t \\mid y^t)`` is the ``n \\times n`` stabilizing state error covariance
- ``A``, ``B``, ``C``, ``D`` are the canonical state-space matrices
- ``K`` is the ``n \\times k`` steady-state Kalman gain ensuring ``\\rho(A - K C) \\le 1``
- ``\\Sigma_u`` is the ``k \\times k`` Wold one-step-ahead innovation covariance

Solves the DARE via `_solve_dare` using a `2n × 2n` generalized real Schur
symplectic pencil or Van Dooren QR-compressed `(2n+k)` extended pencil when `D * D'`
is singular or ill-conditioned.

# References
- Fernández-Villaverde, J., Rubio-Ramírez, J. F., Sargent, T. J., & Watson, M. W. (2007).
  ABCs (and Ds) of Understanding VARs. *American Economic Review*, 97(3), 1021–1026.
- Hansen, L. P., & Sargent, T. J. (2013). *Recursive Models of Dynamic Linear Economies*.
  Princeton University Press, Ch. 8–9.
"""
function innovations_representation(abcd::ABCDSystem{T};
                                    rcond_tol::Real=1e-12) where {T<:AbstractFloat}
    rcond_tol > 0 || throw(ArgumentError("rcond_tol must be positive; got $rcond_tol"))

    n = nstates(abcd)
    k = nobservables(abcd)

    WT = promote_type(T, Float64)
    A_w = Matrix{WT}(abcd.A)
    B_w = Matrix{WT}(abcd.B)
    C_w = Matrix{WT}(abcd.C)
    D_w = Matrix{WT}(abcd.D)

    Q_raw = B_w * B_w'
    Q_w = Matrix{WT}((Q_raw + Q_raw') / 2)
    R_raw = D_w * D_w'
    R_w = Matrix{WT}((R_raw + R_raw') / 2)
    S_w = Matrix{WT}(B_w * D_w')

    P_w, K_w, F_innov_w, res_w = _solve_dare(
        A_w, Q_w, C_w, R_w; S=S_w, rcond_tol=WT(rcond_tol),
    )

    P_sym = n == 0 ? zeros(WT, 0, 0) : Matrix{WT}((P_w + P_w') / 2)
    Sigma_u_w = k == 0 ? zeros(WT, 0, 0) : Matrix{WT}((F_innov_w + F_innov_w') / 2)
    Sigma_u_chol_w = k == 0 ? zeros(WT, 0, 0) : Matrix{WT}(safe_cholesky(Hermitian(Sigma_u_w)))

    if n > 0
        max_cl = maximum(abs, eigvals(A_w - K_w * C_w))
        max_cl <= one(WT) + WT(1e-8) || throw(ArgumentError(
            "Wold innovations closed-loop matrix A - K*C is unstable (spectral radius $max_cl > 1)"))
    end

    if k == 0
        info_loss_w = WT[]
    else
        CPC_diag = n == 0 ? zeros(WT, k) : max.(diag(C_w * P_sym * C_w'), zero(WT))
        info_loss_w = CPC_diag ./ max.(diag(Sigma_u_w), WT(eps(T)))
    end

    inv_flag = is_invertible(abcd)

    return InnovationsRepresentation{T}(
        abcd.A,
        Matrix{T}(K_w),
        abcd.C,
        Matrix{T}(Sigma_u_w),
        Matrix{T}(Sigma_u_chol_w),
        Matrix{T}(P_sym),
        T(res_w),
        inv_flag,
        Vector{T}(info_loss_w),
        abcd.state_names,
        abcd.obs_names,
    )
end

function innovations_representation(
    sol::Union{DSGESolution,PerturbationSolution,HADSGESolution},
    observables::AbstractVector=Symbol[];
    measurement_error=nothing,
    reduction::Symbol=:predetermined,
    abcd_tol::Real=1e-10,
    rcond_tol::Real=1e-12,
)
    sys = abcd_representation(sol, observables;
                              measurement_error=measurement_error,
                              reduction=reduction,
                              tol=abcd_tol)
    return innovations_representation(sys; rcond_tol=rcond_tol)
end

function innovations_representation(
    ss::StateSpaceModel,
    observables::AbstractVector=Symbol[];
    reduction::Symbol=:none,
    abcd_tol::Real=1e-10,
    rcond_tol::Real=1e-12,
)
    sys = abcd_representation(ss, observables; reduction=reduction, tol=abcd_tol)
    return innovations_representation(sys; rcond_tol=rcond_tol)
end

"""
    to_abcd(innov::InnovationsRepresentation{T}) -> ABCDSystem{T}

Convert an [`InnovationsRepresentation`](@ref) into its Wold-fundamental canonical
[`ABCDSystem`](@ref):

```math
\\hat{x}_{t+1} = A \\hat{x}_t + K \\hat{D} \\varepsilon_{t+1}, \\qquad
y_{t+1} = C \\hat{x}_t + \\hat{D} \\varepsilon_{t+1}, \\qquad
E[\\varepsilon_t \\varepsilon_t'] = I_k
```

where:
- ``\\hat{x}_t`` is the ``n \\times 1`` filtered state vector
- ``y_{t+1}`` is the ``k \\times 1`` observable vector
- ``\\varepsilon_{t+1} = \\hat{D}^{-1} u_{t+1}`` are orthonormalized Wold innovations
- ``A``, ``K``, ``C`` are the state transition, Kalman gain, and observation matrices
- ``\\hat{D}`` is `Sigma_u_chol`, the lower-triangular Cholesky factor of ``\\Sigma_u``
"""
function to_abcd(innov::InnovationsRepresentation{T}) where {T<:AbstractFloat}
    n = size(innov.A, 1)
    k = size(innov.C, 1)
    B_hat = Matrix{T}(innov.K * innov.Sigma_u_chol)
    shock_names = ["u_$j" for j in 1:k]
    return ABCDSystem{T}(
        copy(innov.A),
        B_hat,
        copy(innov.C),
        copy(innov.Sigma_u_chol),
        zeros(T, n),
        zeros(T, k),
        copy(innov.state_names),
        copy(innov.obs_names),
        shock_names,
    )
end

"""
    irf(innov::InnovationsRepresentation{T}, horizon::Int=20; ci_type::Symbol=:none) -> ImpulseResponse{T}

Compute analytical Wold-innovation impulse responses of `y_{t+1}` to orthonormalized
innovations `ε_{t+1}` by delegating `irf(to_abcd(innov), horizon; ci_type=ci_type)`.
"""
function irf(innov::InnovationsRepresentation{T}, horizon::Int=20;
             ci_type::Symbol=:none) where {T<:AbstractFloat}
    return irf(to_abcd(innov), horizon; ci_type=ci_type)
end
