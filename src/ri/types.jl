# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Type definitions and display methods for steady-state LQG Rational Inattention
models (Sims 2003; Afrouzi & Yang 2021; Miao, Wu & Young 2022).
"""

# =============================================================================
# Helper conversion utilities
# =============================================================================

function _ri_eltype(x)
    x isa Real && return typeof(x)
    return eltype(x)
end

function _ri_to_square_mat(::Type{T}, A::Real) where {T<:AbstractFloat}
    return fill(T(A), 1, 1)
end

function _ri_to_square_mat(::Type{T}, A::AbstractVector{<:Real}) where {T<:AbstractFloat}
    length(A) == 1 || throw(ArgumentError(
        "A must be a square n×n matrix or scalar, got vector of length $(length(A))"))
    return fill(T(A[1]), 1, 1)
end

function _ri_to_square_mat(::Type{T}, A::AbstractMatrix{<:Real}) where {T<:AbstractFloat}
    return Matrix{T}(A)
end

function _ri_to_colmat(::Type{T}, X::Real) where {T<:AbstractFloat}
    return fill(T(X), 1, 1)
end

function _ri_to_colmat(::Type{T}, X::AbstractVector{<:Real}) where {T<:AbstractFloat}
    return reshape(Vector{T}(X), :, 1)
end

function _ri_to_colmat(::Type{T}, X::AbstractMatrix{<:Real}) where {T<:AbstractFloat}
    return Matrix{T}(X)
end

function _ri_to_names(names::Union{Nothing,AbstractVector}, prefix::String, count::Int, label::String)
    if names === nothing
        return ["$(prefix)$(i)" for i in 1:count]
    end
    length(names) == count || throw(ArgumentError(
        "$label must have length $count, got $(length(names))"))
    return String[string(s) for s in names]
end

# =============================================================================
# RationalInattentionProblem
# =============================================================================

"""
    RationalInattentionProblem{T<:AbstractFloat}

Steady-state linear-quadratic-Gaussian (LQG) Rational Inattention tracking problem
in the sense of Sims (2003):

```math
x_t = A x_{t-1} + B_u u_{t-1} + B w_t, \\qquad w_t \\sim \\mathcal{N}(0, I_m)
```

with target action ``u_t^* = H' x_t \\in \\mathbb{R}^d`` and per-period quadratic
tracking loss ``\\mathbb{E}_t[(u_t - H' x_t)' M (u_t - H' x_t)]`` under either a
Shannon mutual-information capacity constraint ``I(x_t; s^t \\mid s^{t-1}) \\le \\kappa``
or a marginal information shadow cost ``\\omega > 0`` per nat.

# Fields
- `A::Matrix{T}`: `n × n` state transition matrix.
- `B::Matrix{T}`: `n × m` shock loading matrix (`Q = B * B'`).
- `H::Matrix{T}`: `n × d` target action loading (`u_t^* = H' * x_t`).
- `M::Matrix{T}`: `d × d` symmetric positive semidefinite/definite payoff matrix.
- `B_u::Union{Nothing,Matrix{T}}`: optional `n × d` control feedback matrix.
- `beta::T`: discount factor `β ∈ (0, 1]`.
- `capacity::Union{Nothing,T}`: capacity `κ > 0` in `units` (mutually exclusive with `omega`).
- `omega::Union{Nothing,T}`: marginal cost `ω > 0` per nat (mutually exclusive with `capacity`).
- `units::Symbol`: `:bits` or `:nats` (`1 bit = ln(2) nats`).
- `state_names::Vector{String}`: labels for the `n` state variables.
- `action_names::Vector{String}`: labels for the `d` action variables.
- `shock_names::Vector{String}`: labels for the `m` shocks.

# Constructor
    RationalInattentionProblem(A, B, H; M=nothing, B_u=nothing, beta=1.0,
                               capacity=nothing, omega=nothing, units=:bits,
                               state_names=nothing, action_names=nothing, shock_names=nothing)
"""
struct RationalInattentionProblem{T<:AbstractFloat}
    A::Matrix{T}
    B::Matrix{T}
    H::Matrix{T}
    M::Matrix{T}
    B_u::Union{Nothing,Matrix{T}}
    beta::T
    capacity::Union{Nothing,T}
    omega::Union{Nothing,T}
    units::Symbol
    state_names::Vector{String}
    action_names::Vector{String}
    shock_names::Vector{String}

    function RationalInattentionProblem{T}(
        A::Matrix{T},
        B::Matrix{T},
        H::Matrix{T},
        M::Matrix{T},
        B_u::Union{Nothing,Matrix{T}},
        beta::T,
        capacity::Union{Nothing,T},
        omega::Union{Nothing,T},
        units::Symbol,
        state_names::Vector{String},
        action_names::Vector{String},
        shock_names::Vector{String},
    ) where {T<:AbstractFloat}
        n = size(A, 1)
        n >= 1 || throw(ArgumentError("State dimension n must be >= 1, got $n"))
        size(A, 2) == n || throw(ArgumentError(
            "A must be square n×n, got $(size(A))"))
        all(isfinite, A) || throw(ArgumentError("A contains non-finite values"))

        size(B, 1) == n || throw(ArgumentError(
            "B must have $n rows (matching A), got $(size(B))"))
        m = size(B, 2)
        m >= 1 || throw(ArgumentError("B must have at least 1 shock column, got $m"))
        all(isfinite, B) || throw(ArgumentError("B contains non-finite values"))

        size(H, 1) == n || throw(ArgumentError(
            "H must have $n rows (matching A), got $(size(H))"))
        d = size(H, 2)
        d >= 1 || throw(ArgumentError("H must have at least 1 action column, got $d"))
        all(isfinite, H) || throw(ArgumentError("H contains non-finite values"))

        size(M) == (d, d) || throw(ArgumentError(
            "M must be $d×$d (matching action dimension d), got $(size(M))"))
        all(isfinite, M) || throw(ArgumentError("M contains non-finite values"))
        sym_tol = sqrt(eps(T)) * max(opnorm(M, Inf), one(T))
        opnorm(M - M', Inf) <= sym_tol || throw(ArgumentError(
            "M must be symmetric (asymmetry norm $(opnorm(M - M', Inf)) > $sym_tol)"))
        M_sym = Matrix{T}((M + M') / 2)
        min_ev_M = minimum(eigvals(Symmetric(M_sym)))
        psd_tol = sqrt(eps(T)) * max(opnorm(M_sym, Inf), one(T))
        min_ev_M >= -psd_tol || throw(ArgumentError(
            "M must be positive semidefinite (minimum eigenvalue $min_ev_M < -$psd_tol)"))

        if B_u !== nothing
            size(B_u) == (n, d) || throw(ArgumentError(
                "B_u must be $n×$d (n×d), got $(size(B_u))"))
            all(isfinite, B_u) || throw(ArgumentError("B_u contains non-finite values"))
        end

        (isfinite(beta) && zero(T) < beta <= one(T)) || throw(ArgumentError(
            "beta must lie in (0, 1], got $beta"))

        (capacity === nothing) != (omega === nothing) || throw(ArgumentError(
            "Specify exactly one of `capacity` or `omega` (mutually exclusive)"))
        if capacity !== nothing
            (isfinite(capacity) && capacity > zero(T)) || throw(ArgumentError(
                "capacity must be finite and > 0, got $capacity"))
        end
        if omega !== nothing
            (isfinite(omega) && omega > zero(T)) || throw(ArgumentError(
                "omega must be finite and > 0, got $omega"))
        end

        units in (:bits, :nats) || throw(ArgumentError(
            "units must be :bits or :nats, got :$units"))

        length(state_names) == n || throw(ArgumentError(
            "state_names must have length $n, got $(length(state_names))"))
        length(action_names) == d || throw(ArgumentError(
            "action_names must have length $d, got $(length(action_names))"))
        length(shock_names) == m || throw(ArgumentError(
            "shock_names must have length $m, got $(length(shock_names))"))

        return new{T}(
            A, B, H, M_sym, B_u, beta, capacity, omega, units,
            state_names, action_names, shock_names,
        )
    end
end

function RationalInattentionProblem(
    A::Union{Real,AbstractVecOrMat{<:Real}},
    B::Union{Real,AbstractVecOrMat{<:Real}},
    H::Union{Real,AbstractVecOrMat{<:Real}};
    M::Union{Nothing,Real,AbstractMatrix{<:Real}}=nothing,
    B_u::Union{Nothing,Real,AbstractVecOrMat{<:Real}}=nothing,
    beta::Real=1.0,
    capacity::Union{Nothing,Real}=nothing,
    omega::Union{Nothing,Real}=nothing,
    units::Symbol=:bits,
    state_names::Union{Nothing,AbstractVector}=nothing,
    action_names::Union{Nothing,AbstractVector}=nothing,
    shock_names::Union{Nothing,AbstractVector}=nothing,
)
    T = float(promote_type(
        _ri_eltype(A),
        _ri_eltype(B),
        _ri_eltype(H),
        M === nothing ? Bool : _ri_eltype(M),
        B_u === nothing ? Bool : _ri_eltype(B_u),
    ))

    A_mat = _ri_to_square_mat(T, A)
    B_mat = _ri_to_colmat(T, B)
    H_mat = _ri_to_colmat(T, H)

    n = size(A_mat, 1)
    m = size(B_mat, 2)
    d = size(H_mat, 2)

    M_mat = if M === nothing
        Matrix{T}(I, d, d)
    elseif M isa Real
        d == 1 || throw(ArgumentError(
            "Scalar M requires d = 1 action dimension, got d = $d"))
        fill(T(M), 1, 1)
    else
        Matrix{T}(M)
    end

    Bu_mat = B_u === nothing ? nothing : _ri_to_colmat(T, B_u)
    beta_T = T(beta)
    cap_T = capacity === nothing ? nothing : T(capacity)
    om_T = omega === nothing ? nothing : T(omega)

    s_names = _ri_to_names(state_names, "x", n, "state_names")
    a_names = _ri_to_names(action_names, "u", d, "action_names")
    w_names = _ri_to_names(shock_names, "w", m, "shock_names")

    return RationalInattentionProblem{T}(
        A_mat, B_mat, H_mat, M_mat, Bu_mat, beta_T, cap_T, om_T, units,
        s_names, a_names, w_names,
    )
end

# =============================================================================
# SimsKalmanSolution
# =============================================================================

"""
    SimsKalmanSolution{T<:AbstractFloat}

Solution of a steady-state LQG Rational Inattention Kalman filter problem
(Sims 2003; Afrouzi & Yang 2021; Miao, Wu & Young 2022).

# Fields
- `problem::RationalInattentionProblem{T}`: original problem specification.
- `Sigma_prior::Matrix{T}`: steady-state prior error covariance `Σ_{t|t-1}` (`n × n`).
- `Sigma_post::Matrix{T}`: steady-state posterior error covariance `Σ_{t|t}` (`n × n`).
- `Omega_benefit::Matrix{T}`: dynamic effective payoff matrix `Ω_eff` (`n × n`).
- `omega::T`: shadow cost `ω` per nat of mutual information.
- `capacity_bits::T`: equilibrium mutual information rate in bits (`I / ln(2)`).
- `capacity_nats::T`: equilibrium mutual information rate in nats.
- `signal_rank::Int`: endogenous signal dimension `r_s = rank(Σ_prior - Σ_post)`.
- `Y_signal::Matrix{T}`: `n × r_s` orthonormal signal loadings (`s_t = Y_signal' x_t + ψ_t`).
- `Sigma_psi::Matrix{T}`: `r_s × r_s` diagonal SPD signal noise covariance.
- `K_gain::Matrix{T}`: `n × r_s` steady-state Sims Kalman gain.
- `F_policy::Matrix{T}`: `d × n` optimal action policy (`u_t = F_policy * x̂_{t|t}`).
- `expected_loss::T`: steady-state tracking loss `tr(Ω_base * Σ_post)`.
- `converged::Bool`: whether the solver converged within tolerance.
- `iterations::Int`: iteration count.
- `method::Symbol`: solver backend (`:water_filling` or `:logdet_barrier`).
"""
struct SimsKalmanSolution{T<:AbstractFloat}
    problem::RationalInattentionProblem{T}
    Sigma_prior::Matrix{T}
    Sigma_post::Matrix{T}
    Omega_benefit::Matrix{T}
    omega::T
    capacity_bits::T
    capacity_nats::T
    signal_rank::Int
    Y_signal::Matrix{T}
    Sigma_psi::Matrix{T}
    K_gain::Matrix{T}
    F_policy::Matrix{T}
    expected_loss::T
    converged::Bool
    iterations::Int
    method::Symbol
end

# =============================================================================
# Display & Reporting
# =============================================================================

function Base.show(io::IO, prob::RationalInattentionProblem{T}) where {T}
    n, m = size(prob.B)
    d = size(prob.H, 2)
    mode_label = prob.capacity !== nothing ? "Capacity κ" : "Marginal cost ω"
    mode_val = prob.capacity !== nothing ?
        "$(_fmt(prob.capacity)) $(prob.units)" :
        "$(_fmt(prob.omega)) / nat"
    bu_str = prob.B_u === nothing ? "None" : "$(size(prob.B_u, 1))×$(size(prob.B_u, 2))"
    spec = Any[
        "State dimension n"    n;
        "Shock dimension m"    m;
        "Action dimension d"   d;
        "Discount factor β"    _fmt(prob.beta);
        mode_label             mode_val;
        "Control feedback B_u" bu_str;
        "Precision"            string(T);
    ]
    _pretty_table(io, spec;
        title = "Rational Inattention Problem (LQG)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

function report(io::IO, prob::RationalInattentionProblem{T}) where {T}
    show(io, prob)
    n = size(prob.A, 1)
    d = size(prob.H, 2)
    if n <= 12
        println(io)
        _matrix_table(io, prob.A, "State Transition A";
            row_labels=prob.state_names, col_labels=prob.state_names)
    end
    if n <= 12 && d <= 12
        println(io)
        _matrix_table(io, Matrix{T}(prob.H'), "Target Action Loading H'";
            row_labels=prob.action_names, col_labels=prob.state_names)
    end
    return nothing
end
report(prob::RationalInattentionProblem) = report(stdout, prob)

function Base.show(io::IO, sol::SimsKalmanSolution{T}) where {T}
    n = size(sol.Sigma_prior, 1)
    method_str = sol.method === :water_filling ? "Water-filling" : "Log-det barrier"
    spec = Any[
        "Solver method"        method_str;
        "Converged"            "$(_yesno(sol.converged)) ($(sol.iterations) iter)";
        "Shadow cost ω (/nat)" _fmt(sol.omega);
        "Mutual info (bits)"   _fmt(sol.capacity_bits);
        "Mutual info (nats)"   _fmt(sol.capacity_nats);
        "Signal rank r_s"      "$(sol.signal_rank) / $(n)";
        "Expected loss"        _fmt(sol.expected_loss);
    ]
    _pretty_table(io, spec;
        title = "Sims (2003) Rational Inattention Kalman Solution",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

function report(io::IO, sol::SimsKalmanSolution{T}) where {T}
    show(io, sol)
    prob = sol.problem
    n = size(sol.Sigma_prior, 1)
    d = size(sol.F_policy, 1)
    if n <= 12
        println(io)
        _matrix_table(io, sol.Sigma_prior, "Prior Covariance Σ_{t|t-1}";
            row_labels=prob.state_names, col_labels=prob.state_names)
        println(io)
        _matrix_table(io, sol.Sigma_post, "Posterior Covariance Σ_{t|t}";
            row_labels=prob.state_names, col_labels=prob.state_names)
    end
    if sol.signal_rank > 0 && n <= 12
        sig_labels = ["s$(j)" for j in 1:sol.signal_rank]
        println(io)
        _matrix_table(io, sol.Y_signal, "Endogenous Signal Loadings Y_signal";
            row_labels=prob.state_names, col_labels=sig_labels)
        println(io)
        _matrix_table(io, sol.K_gain, "Sims Kalman Gain K_gain";
            row_labels=prob.state_names, col_labels=sig_labels)
        noise_vars = diag(sol.Sigma_psi)
        noise_tbl = Matrix{Any}(undef, sol.signal_rank, 2)
        for j in 1:sol.signal_rank
            noise_tbl[j, 1] = sig_labels[j]
            noise_tbl[j, 2] = _fmt(noise_vars[j])
        end
        println(io)
        _pretty_table(io, noise_tbl;
            title = "Signal Noise Variances diag(Σ_ψ)",
            column_labels = ["Signal", "Variance"],
            alignment = [:l, :r],
        )
    end
    if n <= 12 && d <= 12
        println(io)
        _matrix_table(io, sol.F_policy, "Policy Rule F_policy (u_t = F * x̂_{t|t})";
            row_labels=prob.action_names, col_labels=prob.state_names)
    end
    return nothing
end
report(sol::SimsKalmanSolution) = report(stdout, sol)
