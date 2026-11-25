# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Afrouzi & Yang (2021) dynamic Rational Inattention transition path solver.

Computes the finite-horizon transition path of prior covariances
``\\Sigma_{t|t-1}``, posterior covariances ``\\Sigma_{t|t}``, shadow values
of prior uncertainty ``\\Xi_t``, time-varying Sims Kalman gains
``G_t = K_t Y_t'``, and endogenous information acquisition ``I_t`` (or shadow
prices ``\\omega_t`` under fixed capacity) via forward–backward shooting
anchored at the infinite-horizon steady state `SimsKalmanSolution`.
"""

# =============================================================================
# DynamicRISolution
# =============================================================================

"""
    DynamicRISolution{T<:AbstractFloat}

Transition path solution for a dynamic LQG Rational Inattention problem
(Afrouzi & Yang 2021, §2–3) initialized at ``\\Sigma_{1|0} = \\Sigma_{\\text{init}}``
and anchored at the terminal steady state ``\\Xi_{T_{\\text{path}}+1} = \\Xi_{ss}``.

# Fields
- `Sigma_prior_path::Array{T,3}`: prior covariances ``\\Sigma_{t|t-1}`` (`n × n × T_path`).
- `Sigma_post_path::Array{T,3}`: posterior covariances ``\\Sigma_{t|t}`` (`n × n × T_path`).
- `Xi_path::Array{T,3}`: dynamic marginal value of prior uncertainty ``\\Xi_t`` (`n × n × T_path`).
- `K_gain_path::Array{T,3}`: Kalman gain projection matrices ``G_t = K_t Y_t'`` (`n × n × T_path`).
- `Y_terminal::Matrix{T}`: terminal signal loading matrix `Y_signal` (`n × r_{s,T}`).
- `Sigma_psi_terminal::Matrix{T}`: terminal signal noise covariance ``\\Sigma_{\\psi,T}`` (`r_{s,T} × r_{s,T}`).
- `omega_path::Vector{T}`: shadow information cost path ``\\omega_t`` (`T_path`).
- `capacity_bits_path::Vector{T}`: mutual information rate path in bits (`T_path`).
- `capacity_nats_path::Vector{T}`: mutual information rate path in nats (`T_path`).
- `signal_rank_path::Vector{Int}`: active endogenous signal dimension ``r_{s,t}`` (`T_path`).
- `Sigma_init::Matrix{T}`: initial prior covariance ``\\Sigma_{1|0}`` (`n × n`).
- `T_path::Int`: transition horizon length.
- `converged::Bool`: whether forward–backward shooting converged within `tol`.
- `iterations::Int`: number of shooting iterations performed.
- `residual::T`: maximum relative forward–backward fixed-point residual at termination.
- `state_names::Vector{String}`: labels for the `n` state variables.
- `action_names::Vector{String}`: labels for the `d` action variables.
- `shock_names::Vector{String}`: labels for the `m` fundamental shocks.
"""
struct DynamicRISolution{T<:AbstractFloat}
    Sigma_prior_path::Array{T,3}
    Sigma_post_path::Array{T,3}
    Xi_path::Array{T,3}
    K_gain_path::Array{T,3}
    Y_terminal::Matrix{T}
    Sigma_psi_terminal::Matrix{T}
    omega_path::Vector{T}
    capacity_bits_path::Vector{T}
    capacity_nats_path::Vector{T}
    signal_rank_path::Vector{Int}
    Sigma_init::Matrix{T}
    T_path::Int
    converged::Bool
    iterations::Int
    residual::T
    state_names::Vector{String}
    action_names::Vector{String}
    shock_names::Vector{String}
end

# =============================================================================
# Internal Helpers (_ridyn_*)
# =============================================================================

@inline _ridyn_symmetrize(M::AbstractMatrix{T}) where {T<:AbstractFloat} =
    Matrix{T}((M + M') / 2)

function _ridyn_validate_sigma_init(::Type{TW}, Sigma_init::AbstractMatrix{<:Real}, n::Int) where {TW<:AbstractFloat}
    size(Sigma_init) == (n, n) || throw(ArgumentError(
        "Sigma_init must be $n×$n (matching state dimension n = $n), got $(size(Sigma_init))"))
    all(isfinite, Sigma_init) || throw(ArgumentError(
        "Sigma_init contains non-finite values"))
    S = Matrix{TW}(Sigma_init)
    sym_tol = sqrt(eps(TW)) * max(opnorm(S, Inf), one(TW))
    asym = opnorm(S - S', Inf)
    asym <= sym_tol || throw(ArgumentError(
        "Sigma_init must be symmetric (asymmetry norm $asym > $sym_tol)"))
    S_sym = _ridyn_symmetrize(S)
    min_ev = minimum(eigvals(Symmetric(S_sym)))
    psd_tol = sqrt(eps(TW)) * max(opnorm(S_sym, Inf), one(TW))
    min_ev >= -psd_tol || throw(ArgumentError(
        "Sigma_init must be positive semidefinite (minimum eigenvalue $min_ev < -$psd_tol)"))
    S_psd, _ = _ri_project_psd(S_sym)
    return S_psd
end

function _ridyn_resolve_path(
    ::Type{TW},
    prob::RationalInattentionProblem{T},
    T_path::Int,
    omega_path,
    capacity_path,
) where {T<:AbstractFloat, TW<:AbstractFloat}
    (omega_path !== nothing && capacity_path !== nothing) && throw(ArgumentError(
        "Specify at most one of `omega_path` or `capacity_path` (mutually exclusive)"))

    if omega_path !== nothing
        om_vec = if omega_path isa Real
            fill(TW(omega_path), T_path)
        elseif omega_path isa AbstractVector{<:Real}
            length(omega_path) == T_path || throw(ArgumentError(
                "omega_path must have length T_path = $T_path, got $(length(omega_path))"))
            Vector{TW}(omega_path)
        else
            throw(ArgumentError(
                "omega_path must be a Real scalar or AbstractVector{<:Real}, got $(typeof(omega_path))"))
        end
        all(x -> isfinite(x) && x > zero(TW), om_vec) || throw(ArgumentError(
            "All entries of omega_path must be finite and strictly positive"))
        return (:omega, om_vec)
    elseif capacity_path !== nothing
        cap_vec = if capacity_path isa Real
            fill(TW(capacity_path), T_path)
        elseif capacity_path isa AbstractVector{<:Real}
            length(capacity_path) == T_path || throw(ArgumentError(
                "capacity_path must have length T_path = $T_path, got $(length(capacity_path))"))
            Vector{TW}(capacity_path)
        else
            throw(ArgumentError(
                "capacity_path must be a Real scalar or AbstractVector{<:Real}, got $(typeof(capacity_path))"))
        end
        all(x -> isfinite(x) && x >= zero(TW), cap_vec) || throw(ArgumentError(
            "All entries of capacity_path must be finite and non-negative"))
        cap_nats = prob.units === :bits ? cap_vec .* log(TW(2)) : cap_vec
        return (:capacity, cap_nats)
    elseif prob.omega !== nothing
        return (:omega, fill(TW(prob.omega), T_path))
    else
        cap_val = prob.units === :bits ? TW(prob.capacity) * log(TW(2)) : TW(prob.capacity)
        return (:capacity, fill(cap_val, T_path))
    end
end

function _ridyn_step_fixed_cap(
    Sp::Matrix{TW},
    Oeff::Matrix{TW},
    cap_nats::TW,
    brent_tol::TW,
) where {TW<:AbstractFloat}
    _, _, d, _ = _ri_whitened_modes(Sp, Oeff)
    if cap_nats == zero(TW)
        om_zero = isempty(d) ? zero(TW) : TW(2) * max(d[1], zero(TW))
        return copy(Sp), copy(Oeff), zero(TW), om_zero
    end
    if isempty(d) || d[1] <= TW(1e-30)
        throw(ArgumentError(
            "Cannot allocate positive Shannon capacity κ = $cap_nats nats when prior " *
            "uncertainty or effective payoff is degenerate."))
    end

    om_hi = TW(2) * d[1]
    f_hi = -cap_nats

    function _excess_info(om::TW)
        s = zero(TW)
        @inbounds for i in eachindex(d)
            two_d = TW(2) * d[i]
            if two_d > om
                s += TW(0.5) * log(two_d / om)
            else
                break
            end
        end
        return s - cap_nats
    end

    om_lo = om_hi * exp(-TW(2) * cap_nats) * TW(0.9)
    om_lo = max(om_lo, floatmin(TW) * TW(1e4))
    f_lo = _excess_info(om_lo)
    for _ in 1:60
        f_lo >= zero(TW) && break
        om_hi = om_lo
        f_hi = f_lo
        om_lo *= TW(0.1)
        f_lo = _excess_info(om_lo)
    end
    f_lo >= zero(TW) || throw(ArgumentError(
        "Failed to bracket transition capacity κ = $cap_nats nats from below."))

    om_star, _, _ = _ri_brent_root(_excess_info, om_lo, om_hi, f_lo, f_hi; tol=brent_tol, max_eval=100)
    Spost_t, Xi_t, Inats_t = _ri_wf_step(Sp, Oeff, om_star)
    return Spost_t, Xi_t, Inats_t, om_star
end

function _ridyn_step(
    Sp::Matrix{TW},
    Oeff::Matrix{TW},
    mode::Symbol,
    target_val::TW,
    brent_tol::TW,
) where {TW<:AbstractFloat}
    if mode === :omega
        Spost_t, Xi_t, Inats_t = _ri_wf_step(Sp, Oeff, target_val)
        return Spost_t, Xi_t, Inats_t, target_val
    else
        return _ridyn_step_fixed_cap(Sp, Oeff, target_val, brent_tol)
    end
end

function _ridyn_promote_problem(::Type{TW}, prob::RationalInattentionProblem{T}) where {TW<:AbstractFloat, T<:AbstractFloat}
    TW === T && return prob
    return RationalInattentionProblem{TW}(
        Matrix{TW}(prob.A),
        Matrix{TW}(prob.B),
        Matrix{TW}(prob.H),
        Matrix{TW}(prob.M),
        prob.B_u === nothing ? nothing : Matrix{TW}(prob.B_u),
        TW(prob.beta),
        prob.capacity === nothing ? nothing : TW(prob.capacity),
        prob.omega === nothing ? nothing : TW(prob.omega),
        prob.units,
        copy(prob.state_names),
        copy(prob.action_names),
        copy(prob.shock_names),
    )
end

function _solve_ri_transition_impl(
    ::Type{TW},
    prob::RationalInattentionProblem{T},
    Sigma_init::AbstractMatrix{<:Real},
    T_path::Int,
    omega_path,
    capacity_path,
    tol_w::TW,
    max_iter::Int,
    damp::TW,
) where {T<:AbstractFloat, TW<:AbstractFloat}
    prob_w = _ridyn_promote_problem(TW, prob)
    A = Matrix{TW}(prob_w.A)
    B = Matrix{TW}(prob_w.B)
    H = Matrix{TW}(prob_w.H)
    M = Matrix{TW}(prob_w.M)
    beta = TW(prob_w.beta)
    n = size(A, 1)

    S0 = _ridyn_validate_sigma_init(TW, Sigma_init, n)
    mode, targets = _ridyn_resolve_path(TW, prob_w, T_path, omega_path, capacity_path)

    Obase = _ridyn_symmetrize(H * M * H')
    Q = _ridyn_symmetrize(B * B')

    # Terminal steady-state anchor Ξ_{T_path + 1} = ss.Xi
    ss_tol = min(tol_w * TW(0.05), TW(1e-12))
    ss_w = solve_ri_kalman(prob_w; method=:water_filling, tol=ss_tol, max_iter=2000)
    _, Xi_ss, _ = _ri_wf_step(ss_w.Sigma_prior, ss_w.Omega_benefit, ss_w.omega)

    brent_tol = min(tol_w * TW(0.01), TW(1e-13))

    Sigma_prior_path = Array{TW,3}(undef, n, n, T_path)
    Sigma_post_fwd = Array{TW,3}(undef, n, n, T_path)
    Sigma_post_bwd = Array{TW,3}(undef, n, n, T_path)
    Xi_cur = Array{TW,3}(undef, n, n, T_path)
    Xi_new = Array{TW,3}(undef, n, n, T_path)
    om_cur = Vector{TW}(undef, T_path)
    inats_cur = Vector{TW}(undef, T_path)

    @inbounds for t in 1:T_path
        Xi_cur[:, :, t] .= Xi_ss
    end

    converged = false
    iters_done = 0
    residual_val = TW(Inf)

    for iter in 1:max_iter
        iters_done = iter

        # Forward shooting pass: Σ_{t|t-1} -> Σ_{t|t} -> Σ_{t+1|t}
        Sp_t = copy(S0)
        for t in 1:T_path
            Sigma_prior_path[:, :, t] .= Sp_t
            Xi_next = t < T_path ? @view(Xi_cur[:, :, t + 1]) : Xi_ss
            Oeff_t = _ridyn_symmetrize(Obase + beta * (A' * Xi_next * A))
            Spost_t, _, _, _ = _ridyn_step(Sp_t, Oeff_t, mode, targets[t], brent_tol)
            Sigma_post_fwd[:, :, t] .= Spost_t
            if t < T_path
                Sp_t = _ridyn_symmetrize(A * Spost_t * A' + Q)
            end
        end

        # Backward shooting pass: Ξ_{T+1} = Ξ_ss -> Ξ_t given Σ_{t|t-1}
        Xi_next = copy(Xi_ss)
        for t in T_path:-1:1
            Sp_t = Matrix{TW}(@view Sigma_prior_path[:, :, t])
            Oeff_t = _ridyn_symmetrize(Obase + beta * (A' * Xi_next * A))
            Spost_t, Xi_t, Inats_t, om_t = _ridyn_step(Sp_t, Oeff_t, mode, targets[t], brent_tol)
            Sigma_post_bwd[:, :, t] .= Spost_t
            Xi_new[:, :, t] .= Xi_t
            om_cur[t] = om_t
            inats_cur[t] = Inats_t
            Xi_next = Xi_t
        end

        # Maximum relative forward-backward discrepancy across the full path
        err = zero(TW)
        for t in 1:T_path
            scale_s = max(norm(@view Sigma_post_bwd[:, :, t]), norm(Q), one(TW))
            scale_x = max(norm(@view Xi_new[:, :, t]), norm(Obase), one(TW))
            es = norm(@view(Sigma_post_bwd[:, :, t]) .- @view(Sigma_post_fwd[:, :, t])) / scale_s
            ex = norm(@view(Xi_new[:, :, t]) .- @view(Xi_cur[:, :, t])) / scale_x
            err = max(err, es, ex)
        end
        residual_val = err

        if err < tol_w
            converged = true
            @inbounds for t in 1:T_path
                Xi_cur[:, :, t] .= @view Xi_new[:, :, t]
            end
            break
        end

        one_minus_damp = one(TW) - damp
        @inbounds for t in 1:T_path
            Xi_cur[:, :, t] .= _ridyn_symmetrize(
                one_minus_damp .* @view(Xi_cur[:, :, t]) .+ damp .* @view(Xi_new[:, :, t])
            )
        end
    end

    # Final consistent forward sweep and per-period signal/gain extraction
    Sigma_post_final = Array{TW,3}(undef, n, n, T_path)
    K_gain_path = Array{TW,3}(undef, n, n, T_path)
    cap_bits_path = Vector{TW}(undef, T_path)
    cap_nats_path = Vector{TW}(undef, T_path)
    signal_rank_path = Vector{Int}(undef, T_path)

    zeros_L = zeros(TW, n, n)
    Y_term_w = zeros(TW, n, 0)
    Spsi_term_w = zeros(TW, 0, 0)

    Sp_t = copy(S0)
    for t in 1:T_path
        Sigma_prior_path[:, :, t] .= Sp_t
        Xi_next = t < T_path ? @view(Xi_cur[:, :, t + 1]) : Xi_ss
        Oeff_t = _ridyn_symmetrize(Obase + beta * (A' * Xi_next * A))
        Spost_t, _, _, om_t = _ridyn_step(Sp_t, Oeff_t, mode, targets[t], brent_tol)
        sig_t = _ri_extract_signals(Sp_t, Spost_t, Oeff_t, om_t, :water_filling, zeros_L)

        Sigma_post_final[:, :, t] .= sig_t.Sigma_post_out
        om_cur[t] = om_t
        cap_nats_path[t] = sig_t.capacity_nats
        cap_bits_path[t] = sig_t.capacity_bits
        signal_rank_path[t] = sig_t.signal_rank
        K_gain_path[:, :, t] .= sig_t.signal_rank == 0 ? zeros_L : sig_t.K_gain * sig_t.Y_signal'

        if t == T_path
            Y_term_w = sig_t.Y_signal
            Spsi_term_w = sig_t.Sigma_psi
        else
            Sp_t = _ridyn_symmetrize(A * sig_t.Sigma_post_out * A' + Q)
        end
    end

    return DynamicRISolution{T}(
        Array{T,3}(Sigma_prior_path),
        Array{T,3}(Sigma_post_final),
        Array{T,3}(Xi_cur),
        Array{T,3}(K_gain_path),
        Matrix{T}(Y_term_w),
        Matrix{T}(Spsi_term_w),
        Vector{T}(om_cur),
        Vector{T}(cap_bits_path),
        Vector{T}(cap_nats_path),
        signal_rank_path,
        Matrix{T}(S0),
        T_path,
        converged,
        iters_done,
        T(residual_val),
        copy(prob.state_names),
        copy(prob.action_names),
        copy(prob.shock_names),
    )
end

# =============================================================================
# Public API: solve_ri_transition
# =============================================================================

"""
    solve_ri_transition(prob::RationalInattentionProblem{T},
                        Sigma_init::AbstractMatrix{<:Real};
                        T_path::Int=100,
                        omega_path=nothing,
                        capacity_path=nothing,
                        tol::Real=1e-10,
                        max_iter::Int=500,
                        damping::Real=0.7) where {T<:AbstractFloat} -> DynamicRISolution{T}

Solve the finite-horizon dynamic Rational Inattention transition path (Afrouzi & Yang 2021)
from initial prior covariance ``\\Sigma_{1|0} = \\Sigma_{\\text{init}}`` to the steady-state
anchor ``\\Xi_{T_{\\text{path}}+1} = \\Xi_{ss}`` computed via [`solve_ri_kalman`](@ref):

```math
\\Sigma_{t|t-1} = A \\Sigma_{t-1|t-1} A' + B B', \\qquad \\Omega_{\\text{eff},t} = H M H' + \\beta A' \\Xi_{t+1} A
```

where:
- ``\\Sigma_{t|t-1}`` is the ``n \\times n`` prior state error covariance at period ``t``
- ``\\Sigma_{t-1|t-1}`` is the ``n \\times n`` posterior state error covariance at period ``t-1``
- ``A`` is the ``n \\times n`` state transition matrix
- ``B`` is the ``n \\times m`` fundamental shock loading matrix
- ``\\Omega_{\\text{eff},t}`` is the ``n \\times n`` dynamic marginal benefit matrix of precision
- ``H`` is the ``n \\times d`` target action loading matrix
- ``M`` is the ``d \\times d`` tracking loss weighting matrix
- ``\\beta`` is the discount factor in ``(0, 1]``
- ``\\Xi_{t+1}`` is the ``n \\times n`` dynamic continuation value of prior uncertainty at ``t+1``

# Arguments
- `prob::RationalInattentionProblem{T}`: steady-state LQG RI problem specification.
- `Sigma_init::AbstractMatrix{<:Real}`: symmetric PSD ``n \\times n`` initial prior covariance ``\\Sigma_{1|0}``.

# Keyword Arguments
- `T_path::Int=100`: number of transition periods.
- `omega_path=nothing`: optional scalar or length-`T_path` vector of shadow costs ``\\omega_t > 0`` (mutually exclusive with `capacity_path`).
- `capacity_path=nothing`: optional scalar or length-`T_path` vector of per-period capacities ``\\kappa_t \\ge 0`` in `prob.units` (mutually exclusive with `omega_path`).
- `tol::Real=1e-10`: convergence tolerance on the relative forward–backward fixed-point residual.
- `max_iter::Int=500`: maximum forward–backward shooting iterations.
- `damping::Real=0.7`: update mixing weight in `(0, 1]` applied to ``\\Xi_t`` (`1.0` = undamped).
"""
function solve_ri_transition(
    prob::RationalInattentionProblem{T},
    Sigma_init::AbstractMatrix{<:Real};
    T_path::Int=100,
    omega_path=nothing,
    capacity_path=nothing,
    tol::Real=1e-10,
    max_iter::Int=500,
    damping::Real=0.7,
) where {T<:AbstractFloat}
    T_path >= 1 || throw(ArgumentError("T_path must be >= 1, got $T_path"))
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be >= 1, got $max_iter"))
    (isfinite(damping) && 0 < damping <= 1) || throw(ArgumentError(
        "damping must lie in (0, 1], got $damping"))

    if T <: Union{Float16, Float32}
        return _solve_ri_transition_impl(
            Float64, prob, Sigma_init, T_path, omega_path, capacity_path,
            max(Float64(tol), 10 * eps(Float64)), max_iter, Float64(damping),
        )
    else
        return _solve_ri_transition_impl(
            T, prob, Sigma_init, T_path, omega_path, capacity_path,
            max(T(tol), T(10) * eps(T)), max_iter, T(damping),
        )
    end
end

# =============================================================================
# Display & Reporting
# =============================================================================

function Base.show(io::IO, sol::DynamicRISolution{T}) where {T}
    n = size(sol.Sigma_init, 1)
    spec = Any[
        "Horizon T_path"            sol.T_path;
        "State dimension n"         n;
        "Converged"                 "$(_yesno(sol.converged)) ($(sol.iterations) iter)";
        "Residual"                  _fmt(sol.residual);
        "Initial info flow (bits)"  _fmt(sol.capacity_bits_path[1]);
        "Terminal info flow (bits)" _fmt(sol.capacity_bits_path[end]);
        "Initial shadow cost ω₁"    _fmt(sol.omega_path[1]);
        "Terminal shadow cost ω_T"  _fmt(sol.omega_path[end]);
        "Signal rank (t=1 → t=T)"   "$(sol.signal_rank_path[1]) → $(sol.signal_rank_path[end])";
        "Precision"                 string(T);
    ]
    _pretty_table(io, spec;
        title = "Afrouzi–Yang (2021) Dynamic RI Transition Solution",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

function report(io::IO, sol::DynamicRISolution{T}) where {T}
    show(io, sol)
    Tp = sol.T_path
    n = size(sol.Sigma_init, 1)

    t_idx = _select_horizons(Tp)
    traj = Matrix{Any}(undef, length(t_idx), 5)
    for (row, t) in enumerate(t_idx)
        traj[row, 1] = t
        traj[row, 2] = _fmt(sol.capacity_bits_path[t])
        traj[row, 3] = _fmt(sol.capacity_nats_path[t])
        traj[row, 4] = _fmt(sol.omega_path[t])
        traj[row, 5] = sol.signal_rank_path[t]
    end
    println(io)
    _pretty_table(io, traj;
        title = "Transition Path Summary",
        column_labels = ["t", "Info (bits)", "Info (nats)", "Shadow ω_t", "Rank r_s"],
        alignment = [:r, :r, :r, :r, :r],
    )

    if n <= 12
        println(io)
        _matrix_table(io, sol.Sigma_init, "Initial Prior Covariance Σ_{1|0}";
            row_labels=sol.state_names, col_labels=sol.state_names)
        println(io)
        _matrix_table(io, sol.Sigma_post_path[:, :, 1], "Initial Posterior Covariance Σ_{1|1}";
            row_labels=sol.state_names, col_labels=sol.state_names)
        println(io)
        _matrix_table(io, sol.Sigma_post_path[:, :, end], "Terminal Posterior Covariance Σ_{T|T}";
            row_labels=sol.state_names, col_labels=sol.state_names)
        r_term = size(sol.Y_terminal, 2)
        if r_term > 0
            sig_labels = ["s$(j)" for j in 1:r_term]
            println(io)
            _matrix_table(io, sol.Y_terminal, "Terminal Signal Loadings Y_terminal";
                row_labels=sol.state_names, col_labels=sig_labels)
        end
    end
    return nothing
end
report(sol::DynamicRISolution) = report(stdout, sol)
