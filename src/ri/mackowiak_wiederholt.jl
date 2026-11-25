# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Maćkowiak–Wiederholt (2009, 2015) multi-block Rational Inattention solver.

Computes the optimal allocation of Shannon capacity ``\\kappa = \\kappa_{\\text{agg}} +
\\kappa_{\\text{idio}}`` across independent aggregate and idiosyncratic price-setting
blocks via water-filling at common shadow cost ``\\omega``, jointly with the
general-equilibrium fixed point for the endogenous aggregate price level ``P_t``
under strategic complementarity ``r \\in [0, 1)``.
"""

# =============================================================================
# Result Type
# =============================================================================

"""
    MWAttentionResult{T<:AbstractFloat}

Equilibrium solution to the Maćkowiak & Wiederholt (2009, 2015) multi-block
Rational Inattention attention-allocation problem with general-equilibrium
strategic complementarity.

Given total Shannon capacity ``\\kappa`` (or shadow information cost ``\\omega``)
and strategic complementarity ``r \\in [0, 1)``, firms optimally track the
aggregate profit-maximizing target ``\\Delta_t = (1 - r) q_t + r p_t`` and the
idiosyncratic target ``z_t``, equating marginal information value across blocks:

```math
\\kappa = \\kappa_{\\text{agg}}(\\omega) + \\kappa_{\\text{idio}}(\\omega), \\qquad p_t = \\bar{\\mathbb{E}}_t\\bigl[(1 - r) q_t + r p_t\\bigr], \\qquad y_t = q_t - p_t
```

where:
- ``\\kappa`` is total Shannon mutual information allocated across aggregate and idiosyncratic signals
- ``\\kappa_{\\text{agg}}(\\omega)`` and ``\\kappa_{\\text{idio}}(\\omega)`` are the water-filling capacities allocated at common shadow cost ``\\omega``
- ``r \\in [0, 1)`` is the degree of strategic complementarity in price setting
- ``q_t`` and ``p_t`` are nominal aggregate demand and the consensus aggregate price level
- ``y_t = q_t - p_t`` is the equilibrium real output response to aggregate nominal shocks

# Fields
- `kappa_agg::T`: equilibrium Shannon capacity allocated to the aggregate block (in `units`).
- `kappa_idio::T`: equilibrium Shannon capacity allocated to the idiosyncratic block (in `units`).
- `total_kappa::T`: total equilibrium capacity `kappa_agg + kappa_idio` (in `units`).
- `omega::T`: equilibrium Lagrange multiplier (marginal value of information) equalized across blocks.
- `strategic_complementarity::T`: strategic complementarity parameter ``r \\in [0, 1)``.
- `units::Symbol`: information units (`:bits` or `:nats`).
- `agg_price_irf::Vector{T}`: length-`horizon` consensus aggregate price impulse response ``p_h`` to a 1-s.d. aggregate shock.
- `idio_price_irf::Vector{T}`: length-`horizon` idiosyncratic price impulse response to a 1-s.d. idiosyncratic shock.
- `real_output_irf::Vector{T}`: length-`horizon` real output impulse response ``y_h = q_h - p_h`` to a 1-s.d. aggregate shock.
- `agg_solution::SimsKalmanSolution{T}`: converged steady-state Sims Kalman solution for the aggregate equilibrium tracking problem.
- `idio_solution::SimsKalmanSolution{T}`: converged steady-state Sims Kalman solution for the idiosyncratic block.
- `converged::Bool`: whether the fixed-point and water-filling solver converged within `tol`.
- `iterations::Int`: number of general-equilibrium fixed-point iterations executed.
"""
struct MWAttentionResult{T<:AbstractFloat}
    kappa_agg::T
    kappa_idio::T
    total_kappa::T
    omega::T
    strategic_complementarity::T
    units::Symbol
    agg_price_irf::Vector{T}
    idio_price_irf::Vector{T}
    real_output_irf::Vector{T}
    agg_solution::SimsKalmanSolution{T}
    idio_solution::SimsKalmanSolution{T}
    converged::Bool
    iterations::Int
end

# =============================================================================
# Internal Helpers (_rimw_*)
# =============================================================================

@inline _rimw_symmetrize(M::AbstractMatrix{T}) where {T<:AbstractFloat} =
    Matrix{T}((M + M') / 2)

function _rimw_to_mat(::Type{TW}, X) where {TW<:AbstractFloat}
    if X isa Real
        return fill(TW(X), 1, 1)
    elseif X isa AbstractVector
        return reshape(TW[TW(v) for v in X], length(X), 1)
    elseif X isa AbstractMatrix
        return Matrix{TW}(X)
    else
        throw(ArgumentError("Expected scalar, vector, or matrix in block specification, got $(typeof(X))"))
    end
end

function _rimw_psd_sqrt(::Type{TW}, Q::AbstractMatrix{<:Real}) where {TW<:AbstractFloat}
    Qm = _rimw_symmetrize(Matrix{TW}(Q))
    n = size(Qm, 1)
    size(Qm, 2) == n || throw(DimensionMismatch("Q must be square, got $(size(Qm))"))
    if n == 1
        val = Qm[1, 1]
        val >= -TW(1e-12) || throw(ArgumentError("Q must be positive semi-definite, got $val"))
        return fill(sqrt(max(val, zero(TW))), 1, 1)
    end
    F = eigen(Symmetric(Qm))
    tol_ev = TW(1e-12) * max(maximum(abs, F.values), one(TW))
    minimum(F.values) >= -tol_ev || throw(ArgumentError(
        "Q must be positive semi-definite, minimum eigenvalue $(minimum(F.values))"))
    vals = max.(F.values, zero(TW))
    return Matrix{TW}(F.vectors * Diagonal(sqrt.(vals)))
end

function _rimw_parse_block(
    ::Type{TW},
    block,
    label::String;
    default_units::Symbol=:bits,
) where {TW<:AbstractFloat}
    if block isa RationalInattentionProblem
        size(block.H, 2) == 1 || throw(ArgumentError(
            "$label block must have a scalar tracking target (size(H, 2) == 1), got $(size(block.H, 2))"))
        block.B_u === nothing || throw(ArgumentError(
            "$label block in solve_mw_attention cannot have endogenous state feedback B_u !== nothing"))
        A = Matrix{TW}(block.A)
        B = Matrix{TW}(block.B)
        H = Matrix{TW}(block.H)
        M = Matrix{TW}(block.M)
        beta = TW(block.beta)
        units = block.units
        return (A=A, B=B, H=H, M=M, beta=beta, units=units,
                state_names=copy(block.state_names),
                action_names=copy(block.action_names),
                shock_names=copy(block.shock_names))
    elseif block isa Tuple
        if length(block) in (2, 3) &&
           (block[1] isa Real || block[1] isa Tuple || block[1] isa AbstractVector) &&
           block[2] isa Real
            w = length(block) == 3 ? block[3] : one(TW)
            w isa Real || throw(ArgumentError(
                "$label tuple weight must be Real, got $(typeof(w))"))
            return _rimw_parse_block(TW, (rho=block[1], sigma=block[2], weight=w), label;
                                     default_units=default_units)
        elseif length(block) in (3, 4)
            nt = length(block) == 3 ?
                (A=block[1], B=block[2], H=block[3]) :
                (A=block[1], B=block[2], H=block[3], M=block[4])
            return _rimw_parse_block(TW, nt, label; default_units=default_units)
        else
            throw(ArgumentError(
                "$label Tuple block must be (rho, sigma[, weight]) or (A, B, H[, M]), got length $(length(block))"))
        end
    elseif block isa NamedTuple
        beta = haskey(block, :beta) ? TW(block.beta) : TW(0.99)
        (zero(TW) < beta <= one(TW)) || throw(ArgumentError(
            "$label beta must lie in (0, 1], got $beta"))
        units = haskey(block, :units) ? Symbol(block.units) : default_units
        units in (:bits, :nats) || throw(ArgumentError(
            "$label units must be :bits or :nats, got :$units"))

        if haskey(block, :A)
            A = _rimw_to_mat(TW, block.A)
            n = LinearAlgebra.checksquare(A)
            H_raw = haskey(block, :H) ? block.H : throw(ArgumentError(
                "$label matrix block must include :H"))
            H = H_raw isa AbstractVector ? reshape(TW[TW(x) for x in H_raw], length(H_raw), 1) :
                _rimw_to_mat(TW, H_raw)
            size(H, 1) == n || throw(DimensionMismatch(
                "$label H row count $(size(H, 1)) must match state dimension $n"))
            size(H, 2) == 1 || throw(ArgumentError(
                "$label block must have scalar action (size(H, 2) == 1), got $(size(H, 2))"))

            B = if haskey(block, :B)
                B_in = _rimw_to_mat(TW, block.B)
                size(B_in, 1) == n || throw(DimensionMismatch(
                    "$label B row count $(size(B_in, 1)) must match state dimension $n"))
                B_in
            elseif haskey(block, :Q)
                L_Q = _rimw_psd_sqrt(TW, _rimw_to_mat(TW, block.Q))
                if haskey(block, :G)
                    Gmat = _rimw_to_mat(TW, block.G)
                    size(Gmat, 1) == n || throw(DimensionMismatch(
                        "$label G row count $(size(Gmat, 1)) must match state dimension $n"))
                    size(Gmat, 2) == size(L_Q, 1) || throw(DimensionMismatch(
                        "$label G columns $(size(Gmat, 2)) must match Q dimension $(size(L_Q, 1))"))
                    Gmat * L_Q
                else
                    size(L_Q, 1) == n || throw(DimensionMismatch(
                        "$label Q dimension $(size(L_Q, 1)) must match state dimension $n when G is omitted"))
                    L_Q
                end
            else
                throw(ArgumentError("$label matrix block must specify :B or :Q (with optional :G)"))
            end

            M = if haskey(block, :M)
                _rimw_to_mat(TW, block.M)
            elseif haskey(block, :R)
                _rimw_to_mat(TW, block.R)
            elseif haskey(block, :weight)
                fill(TW(block.weight), 1, 1)
            elseif haskey(block, :payoff_weight)
                fill(TW(block.payoff_weight), 1, 1)
            else
                ones(TW, 1, 1)
            end
            size(M) == (1, 1) || throw(DimensionMismatch(
                "$label payoff matrix M/R must be 1×1, got $(size(M))"))
            M[1, 1] > zero(TW) || throw(ArgumentError(
                "$label payoff weight must be positive, got $(M[1, 1])"))

            snames = ["x$(i)" for i in 1:n]
            m_sh = size(B, 2)
            shnames = ["w$(j)" for j in 1:m_sh]
            anames = [label == "agg" ? "p_agg" : "p_idio"]
            return (A=A, B=B, H=H, M=M, beta=beta, units=units,
                    state_names=snames, action_names=anames, shock_names=shnames)
        elseif haskey(block, :rho) && haskey(block, :sigma)
            sig = TW(block.sigma)
            (isfinite(sig) && sig > zero(TW)) || throw(ArgumentError(
                "$label sigma must be finite and > 0, got $sig"))
            weight = haskey(block, :weight) ? TW(block.weight) :
                     (haskey(block, :payoff_weight) ? TW(block.payoff_weight) : one(TW))
            (isfinite(weight) && weight > zero(TW)) || throw(ArgumentError(
                "$label weight must be finite and > 0, got $weight"))
            proc = haskey(block, :process) ? Symbol(block.process) : :level
            proc in (:level, :growth) || throw(ArgumentError(
                "$label process must be :level or :growth, got :$proc"))

            rho_raw = block.rho
            if rho_raw isa Real
                rho = TW(rho_raw)
                isfinite(rho) || throw(ArgumentError("$label rho must be finite, got $rho"))
                if proc === :level
                    A = fill(rho, 1, 1)
                    B = fill(sig, 1, 1)
                    H = ones(TW, 1, 1)
                    M = fill(weight, 1, 1)
                    snames = ["x1"]
                else
                    A = TW[one(TW) rho; zero(TW) rho]
                    B = reshape(TW[sig, sig], 2, 1)
                    H = reshape(TW[one(TW), zero(TW)], 2, 1)
                    M = fill(weight, 1, 1)
                    snames = ["level", "growth"]
                end
            elseif rho_raw isa Union{AbstractVector,Tuple}
                length(rho_raw) == 2 || throw(ArgumentError(
                    "$label vector/tuple rho only supports length 2 AR(2) roots (ρ₁, ρ₂), got length $(length(rho_raw))"))
                r1, r2 = TW(rho_raw[1]), TW(rho_raw[2])
                (isfinite(r1) && isfinite(r2)) || throw(ArgumentError(
                    "$label AR(2) roots must be finite"))
                A = TW[r1 + r2  -r1 * r2; one(TW) zero(TW)]
                B = reshape(TW[sig, zero(TW)], 2, 1)
                H = reshape(TW[one(TW), zero(TW)], 2, 1)
                M = fill(weight, 1, 1)
                snames = ["x_t", "x_lag1"]
            else
                throw(ArgumentError("$label rho must be Real or 2-element Tuple/Vector, got $(typeof(rho_raw))"))
            end
            return (A=A, B=B, H=H, M=M, beta=beta, units=units,
                    state_names=snames,
                    action_names=[label == "agg" ? "p_agg" : "p_idio"],
                    shock_names=["w1"])
        else
            throw(ArgumentError(
                "$label NamedTuple block must provide either (:rho, :sigma) or (:A, :B/:Q, :H)"))
        end
    else
        throw(ArgumentError(
            "$label block must be a RationalInattentionProblem, NamedTuple, or Tuple; got $(typeof(block))"))
    end
end

function _rimw_nominal_irf(
    A::Matrix{TW},
    B::Matrix{TW},
    H::Matrix{TW},
    L::Int,
) where {TW<:AbstractFloat}
    n_a, m_a = size(B)
    q_irf = Matrix{TW}(undef, L, m_a)
    for j in 1:m_a
        x = copy(@view B[:, j])
        q_irf[1, j] = dot(@view(H[:, 1]), x)
        for h in 2:L
            x = A * x
            q_irf[h, j] = dot(@view(H[:, 1]), x)
        end
    end
    return q_irf
end

function _rimw_build_agg_aug_static(spec, L::Int, r::TW) where {TW<:AbstractFloat}
    r == zero(TW) && return (A_aug=spec.A, B_aug=spec.B, use_aug=false)
    n_a, m_a = size(spec.B)
    n_w = m_a * L
    n_aug = n_a + n_w

    A_aug = zeros(TW, n_aug, n_aug)
    @views A_aug[1:n_a, 1:n_a] .= spec.A
    for l in 2:L
        r_idx = n_a + (l - 1) * m_a
        c_idx = n_a + (l - 2) * m_a
        for j in 1:m_a
            A_aug[r_idx + j, c_idx + j] = one(TW)
        end
    end

    B_aug = zeros(TW, n_aug, m_a)
    @views B_aug[1:n_a, :] .= spec.B
    for j in 1:m_a
        B_aug[n_a + j, j] = one(TW)
    end
    return (A_aug=A_aug, B_aug=B_aug, use_aug=true)
end

function _rimw_make_H_aug(
    spec,
    static_aug,
    y_irf::Matrix{TW},
    r::TW,
    L::Int,
) where {TW<:AbstractFloat}
    !static_aug.use_aug && return spec.H
    n_a, m_a = size(spec.B)
    n_aug = size(static_aug.A_aug, 1)
    H_aug = zeros(TW, n_aug, 1)
    @views H_aug[1:n_a, 1] .= spec.H[:, 1]
    # Δ_t = (1 - r) q_t + r p_t = q_t - r y_t, invariant under uniform B scaling
    for l in 1:L
        base = n_a + (l - 1) * m_a
        for j in 1:m_a
            H_aug[base + j, 1] = -r * y_irf[l, j]
        end
    end
    return H_aug
end

function _rimw_make_agg_prob(
    spec,
    static_aug,
    y_irf::Matrix{TW},
    r::TW,
    L::Int;
    omega::TW,
    scale_B::TW=one(TW),
) where {TW<:AbstractFloat}
    B_scaled = static_aug.B_aug .* scale_B
    H_aug = _rimw_make_H_aug(spec, static_aug, y_irf, r, L)
    if !static_aug.use_aug
        return RationalInattentionProblem(
            spec.A, B_scaled, H_aug;
            M=spec.M, beta=spec.beta, omega=omega, units=spec.units,
            state_names=spec.state_names, action_names=spec.action_names, shock_names=spec.shock_names,
        )
    end
    n_a, m_a = size(spec.B)
    snames = vcat(spec.state_names, ["w_lag$(l)_$(j)" for l in 1:L for j in 1:m_a])
    return RationalInattentionProblem(
        static_aug.A_aug, B_scaled, H_aug;
        M=spec.M, beta=spec.beta, omega=omega, units=spec.units,
        state_names=snames, action_names=spec.action_names, shock_names=spec.shock_names,
    )
end

function _rimw_consensus_irf_mats(
    A::Matrix{TW},
    B_unit::AbstractMatrix{TW},
    H_col::AbstractMatrix{TW},
    K_gain::Matrix{TW},
    Y_signal::Matrix{TW},
    L::Int,
) where {TW<:AbstractFloat}
    m = size(B_unit, 2)
    n = size(A, 1)
    KY = size(Y_signal, 2) == 0 ? zeros(TW, n, n) : K_gain * Matrix{TW}(Y_signal')
    F = Matrix{TW}(H_col')

    out = Matrix{TW}(undef, L, m)
    for j in 1:m
        x = Matrix{TW}(B_unit)[:, j]
        xbar = KY * x
        out[1, j] = (F * xbar)[1]
        for h in 2:L
            x = A * x
            xbar_prior = A * xbar
            xbar = xbar_prior + KY * (x - xbar_prior)
            out[h, j] = (F * xbar)[1]
        end
    end
    return out
end

function _rimw_consensus_irf(
    sol::SimsKalmanSolution{TW},
    B_unit::AbstractMatrix{TW},
    L::Int,
) where {TW<:AbstractFloat}
    prob = sol.problem
    return _rimw_consensus_irf_mats(
        Matrix{TW}(prob.A), B_unit, Matrix{TW}(prob.H),
        Matrix{TW}(sol.K_gain), Matrix{TW}(sol.Y_signal), L,
    )
end

function _rimw_nats_to_units(Inats::TW, units::Symbol) where {TW<:AbstractFloat}
    return units === :bits ? Inats / log(TW(2)) : Inats
end

function _rimw_try_fp(
    A::Matrix{TW},
    B::Matrix{TW},
    Q::Matrix{TW},
    Obase::Matrix{TW},
    beta::TW,
    omega::TW,
    tol_inner::TW;
    Sigma_uncond::Union{Nothing,Matrix{TW}}=nothing,
    Xi_zero::Union{Nothing,Matrix{TW}}=nothing,
    warm_ref::Ref{Any}=Ref{Any}(nothing),
) where {TW<:AbstractFloat}
    try
        r = _ri_solve_fixed_omega(
            A, B, Q, Obase, beta, omega, :water_filling, tol_inner, 2000;
            Sigma_uncond=Sigma_uncond, Xi_zero=Xi_zero, warm_state=warm_ref[],
        )
        if r.converged
            warm_ref[] = (copy(r.Sigma_post), copy(r.Xi), nothing)
        end
        return r
    catch e
        if e isa ArgumentError && occursin("diverged", e.msg)
            return nothing
        end
        rethrow()
    end
end

function _rimw_solve_given_y(
    A_agg::Matrix{TW},
    B_agg_scaled::Matrix{TW},
    Q_agg_scaled::Matrix{TW},
    beta_agg::TW,
    H_aug::Matrix{TW},
    M_agg::Matrix{TW},
    Sigma_uncond_agg::Union{Nothing,Matrix{TW}},
    Xi_zero_agg::Union{Nothing,Matrix{TW}},
    warm_agg::Ref{Any},
    A_id::Matrix{TW},
    B_id_scaled::Matrix{TW},
    Q_id_scaled::Matrix{TW},
    beta_id::TW,
    Obase_id::Matrix{TW},
    Sigma_uncond_id::Union{Nothing,Matrix{TW}},
    Xi_zero_id::Union{Nothing,Matrix{TW}},
    warm_id::Ref{Any},
    units::Symbol,
    scale_B::TW,
    target_cap::Union{Nothing,TW},
    fixed_omega::Union{Nothing,TW},
    om_hint::TW,
    tol_inner::TW,
    brent_tol::TW;
    warm_bracket::Bool=false,
) where {TW<:AbstractFloat}
    Obase_agg = _rimw_symmetrize(H_aug * M_agg * H_aug')

    if fixed_omega !== nothing
        om_s = fixed_omega * (scale_B^2)
        r_agg = _ri_solve_fixed_omega(
            A_agg, B_agg_scaled, Q_agg_scaled, Obase_agg, beta_agg, om_s,
            :water_filling, tol_inner, 2000;
            Sigma_uncond=Sigma_uncond_agg, Xi_zero=Xi_zero_agg, warm_state=warm_agg[],
        )
        r_agg.converged && (warm_agg[] = (copy(r_agg.Sigma_post), copy(r_agg.Xi), nothing))
        r_id = _ri_solve_fixed_omega(
            A_id, B_id_scaled, Q_id_scaled, Obase_id, beta_id, om_s,
            :water_filling, tol_inner, 2000;
            Sigma_uncond=Sigma_uncond_id, Xi_zero=Xi_zero_id, warm_state=warm_id[],
        )
        r_id.converged && (warm_id[] = (copy(r_id.Sigma_post), copy(r_id.Xi), nothing))
        return (omega_scaled=om_s, fp_agg=r_agg, fp_id=r_id, Obase_agg=Obase_agg)
    end

    cap_t = target_cap::TW
    cache_fps = Ref{Any}(nothing)
    cache_om = Ref{TW}(TW(NaN))

    function eval_excess(om_s::TW)
        r_agg = _rimw_try_fp(
            A_agg, B_agg_scaled, Q_agg_scaled, Obase_agg, beta_agg, om_s, tol_inner;
            Sigma_uncond=Sigma_uncond_agg, Xi_zero=Xi_zero_agg, warm_ref=warm_agg,
        )
        r_agg === nothing && return TW(NaN)

        r_id = _rimw_try_fp(
            A_id, B_id_scaled, Q_id_scaled, Obase_id, beta_id, om_s, tol_inner;
            Sigma_uncond=Sigma_uncond_id, Xi_zero=Xi_zero_id, warm_ref=warm_id,
        )
        r_id === nothing && return TW(NaN)

        cache_fps[] = (r_agg, r_id)
        cache_om[] = om_s
        return (_rimw_nats_to_units(r_agg.Inats, units) + _rimw_nats_to_units(r_id.Inats, units)) - cap_t
    end

    om_c = om_hint
    f_c = eval_excess(om_c)
    for _ in 1:40
        isfinite(f_c) && break
        om_c *= TW(0.25)
        f_c = eval_excess(om_c)
    end
    isfinite(f_c) || throw(ArgumentError(
        "Failed to locate a finite steady-state attention equilibrium in solve_mw_attention."))

    if abs(f_c) <= brent_tol
        r_agg, r_id = cache_fps[]
        return (omega_scaled=om_c, fp_agg=r_agg, fp_id=r_id, Obase_agg=Obase_agg)
    end

    local om_lo::TW
    local om_hi::TW
    local f_lo::TW
    local f_hi::TW

    if f_c > zero(TW)
        om_lo = om_c
        f_lo = f_c
        om_hi = om_c
        f_hi = f_c
        step = warm_bracket ? TW(1.06) : TW(1.6)
        found = false
        for _ in 1:60
            om_try = om_hi * step
            f_try = eval_excess(om_try)
            if isfinite(f_try)
                if f_try <= zero(TW)
                    om_hi = om_try
                    f_hi = f_try
                    found = true
                    break
                else
                    om_lo = om_try
                    f_lo = f_try
                    om_hi = om_try
                    step = warm_bracket ? min(step * TW(1.5), TW(2.0)) : min(step * TW(1.3), TW(3.0))
                end
            else
                step = sqrt(step)
                step > one(TW) + TW(1e-4) || break
            end
        end
        found || throw(ArgumentError(
            "Insufficient upper omega bracket to bring total capacity below $cap_t $(units)."))
    else
        om_hi = om_c
        f_hi = f_c
        om_lo = om_c
        f_lo = f_c
        step = warm_bracket ? TW(0.94) : TW(0.5)
        found = false
        for _ in 1:60
            om_try = om_lo * step
            f_try = eval_excess(om_try)
            if isfinite(f_try)
                if f_try >= zero(TW)
                    om_lo = om_try
                    f_lo = f_try
                    found = true
                    break
                else
                    om_hi = om_try
                    f_hi = f_try
                    om_lo = om_try
                    warm_bracket && (step = max(step^2, TW(0.5)))
                end
            else
                om_lo = om_try
            end
        end
        found || throw(ArgumentError(
            "Failed to bracket lower shadow cost ω for total capacity = $cap_t $(units)."))
    end

    om_star, _, _ = _ri_brent_root(
        eval_excess, om_lo, om_hi, f_lo, f_hi;
        tol=brent_tol * max(om_lo, TW(1e-12)), max_eval=60,
    )
    if abs(cache_om[] - om_star) > eps(TW) * max(one(TW), om_star)
        eval_excess(om_star)
    end
    r_agg, r_id = cache_fps[]
    return (omega_scaled=om_star, fp_agg=r_agg, fp_id=r_id, Obase_agg=Obase_agg)
end

# =============================================================================
# Public API: solve_mw_attention
# =============================================================================

"""
    solve_mw_attention(agg_block, idio_block;
                       capacity=nothing,
                       omega=nothing,
                       strategic_complementarity::Real=0.85,
                       horizon::Int=40,
                       tol::Real=1e-8,
                       maxiter::Int=200) -> MWAttentionResult

Solve the Maćkowiak & Wiederholt (2009, 2015) multi-block Rational Inattention
problem, jointly determining:

1. **Water-filling attention split** across the aggregate block and idiosyncratic
   block at common marginal information cost ``\\omega``:
   ``\\kappa = \\kappa_{\\text{agg}}(\\omega) + \\kappa_{\\text{idio}}(\\omega)``.
2. **General-equilibrium price fixed point** under strategic complementarity
   ``r \\in [0, 1)``, where firms track ``\\Delta_t = (1 - r) q_t + r p_t`` and
   consensus pricing implies ``p_t = \\bar{\\mathbb{E}}_t[\\Delta_t]`` and real
   output gap ``y_t = q_t - p_t``.

```math
\\min_{\\{\\kappa_{\\text{agg}}, \\kappa_{\\text{idio}}\\}} \\; \\omega\\bigl(\\kappa_{\\text{agg}} + \\kappa_{\\text{idio}}\\bigr) \\quad \\text{s.t.} \\quad \\kappa_{\\text{agg}}(\\omega) + \\kappa_{\\text{idio}}(\\omega) = \\kappa, \\qquad p_t = \\bar{\\mathbb{E}}_t\\bigl[q_t - r y_t\\bigr]
```

where:
- ``\\kappa`` is total available Shannon capacity (`capacity`)
- ``\\omega`` is the common shadow cost of attention (`omega`)
- ``r`` is `strategic_complementarity` in ``[0, 1)``
- ``q_t`` is the aggregate nominal target defined by `agg_block`
- ``p_t`` is the endogenous consensus aggregate price level
- ``y_t = q_t - p_t`` is real output driven by aggregate shocks

# Arguments
- `agg_block`: aggregate block specified as a [`RationalInattentionProblem`](@ref), a `NamedTuple` (`(; rho, sigma, weight=1.0, process=:level, beta=0.99, units=:bits)` with scalar or 2-root AR(2) `rho`, or `(; A, B, H, M)` / `(; A, G, Q, H, R)`), or a `Tuple` (`(rho, sigma[, weight])` or `(A, B, H[, M])`). Note `process=:growth` constructs ``\\Delta q_t = \\rho \\Delta q_{t-1} + \\sigma w_t``.
- `idio_block`: idiosyncratic block specified in any of the same formats as `agg_block`.

# Keyword Arguments
- `capacity=nothing`: total Shannon capacity ``\\kappa > 0`` to allocate across blocks (mutually exclusive with `omega`).
- `omega=nothing`: common Lagrange multiplier ``\\omega > 0`` on Shannon mutual information (mutually exclusive with `capacity`).
- `strategic_complementarity::Real=0.85`: pricing complementarity ``r \\in [0, 1)`` (higher ``r`` induces stronger aggregate real rigidity).
- `horizon::Int=40`: impulse-response horizon (and equilibrium lag truncation horizon) returned in `MWAttentionResult`.
- `tol::Real=1e-8`: convergence tolerance on equilibrium price IRFs and capacity shares.
- `maxiter::Int=200`: maximum fixed-point iterations for the general-equilibrium aggregate price loop.
"""
function solve_mw_attention(
    agg_block,
    idio_block;
    capacity::Union{Nothing,Real}=nothing,
    omega::Union{Nothing,Real}=nothing,
    strategic_complementarity::Real=0.85,
    horizon::Int=40,
    tol::Real=1e-8,
    maxiter::Int=200,
)
    (capacity === nothing) != (omega === nothing) || throw(ArgumentError(
        "Exactly one of capacity or omega must be provided (mutually exclusive)."))
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1, got $horizon"))
    (isfinite(tol) && tol > 0) || throw(ArgumentError("tol must be finite and > 0, got $tol"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1, got $maxiter"))
    (isfinite(strategic_complementarity) && 0 <= strategic_complementarity < 1) ||
        throw(ArgumentError(
            "strategic_complementarity must satisfy 0 <= r < 1, got $strategic_complementarity"))

    if capacity !== nothing
        (isfinite(capacity) && capacity > 0) || throw(ArgumentError(
            "capacity must be finite and > 0, got $capacity"))
    end
    if omega !== nothing
        (isfinite(omega) && omega > 0) || throw(ArgumentError(
            "omega must be finite and > 0, got $omega"))
    end

    # Infer floating-point precision from inputs
    T = Float64
    if agg_block isa RationalInattentionProblem
        T = promote_type(T, eltype(agg_block.A))
    elseif idio_block isa RationalInattentionProblem
        T = promote_type(T, eltype(idio_block.A))
    end
    if agg_block isa RationalInattentionProblem && eltype(agg_block.A) === Float32 &&
       (!(idio_block isa RationalInattentionProblem) || eltype(idio_block.A) === Float32)
        T = Float32
    elseif idio_block isa RationalInattentionProblem && eltype(idio_block.A) === Float32 &&
           !(agg_block isa RationalInattentionProblem)
        T = Float32
    end
    TW = T <: Union{Float16, Float32} ? Float64 : T

    agg_spec = _rimw_parse_block(TW, agg_block, "agg")
    idio_spec = _rimw_parse_block(TW, idio_block, "idio"; default_units=agg_spec.units)
    agg_spec.units === idio_spec.units || throw(ArgumentError(
        "agg_block units (.$(agg_spec.units)) must match idio_block units (.$(idio_spec.units))"))
    units = agg_spec.units

    r = TW(strategic_complementarity)
    cap_target = capacity === nothing ? nothing : TW(capacity)
    fixed_om = omega === nothing ? nothing : TW(omega)

    # Normalize B scale internally so small/large innovation volatilities solve cleanly
    b_scale_ref = max(norm(agg_spec.B), norm(idio_spec.B))
    b_scale_ref > zero(TW) || throw(ArgumentError(
        "Both agg_block and idio_block have zero shock matrices."))
    scale_B = one(TW) / b_scale_ref

    idio_prob_scaled = RationalInattentionProblem(
        idio_spec.A, idio_spec.B .* scale_B, idio_spec.H;
        M=idio_spec.M, beta=idio_spec.beta, omega=one(TW), units=units,
        state_names=idio_spec.state_names, action_names=idio_spec.action_names,
        shock_names=idio_spec.shock_names,
    )

    L_lag = r == zero(TW) ? horizon : max(horizon, 40)
    static_pe = _rimw_build_agg_aug_static(agg_spec, L_lag, zero(TW))
    static_aug = _rimw_build_agg_aug_static(agg_spec, L_lag, r)
    q_irf = _rimw_nominal_irf(agg_spec.A, agg_spec.B, agg_spec.H, L_lag)
    scale_q = max(maximum(abs, q_irf), one(TW))

    solve_tol = min(TW(tol) * TW(0.05), TW(1e-9))
    brent_tol = min(TW(tol) * TW(0.05), TW(1e-8))

    # Precompute fixed idiosyncratic matrices once
    B_id_scaled = idio_prob_scaled.B
    Q_id_scaled = _rimw_symmetrize(B_id_scaled * B_id_scaled')
    Obase_id = _rimw_symmetrize(idio_spec.H * idio_spec.M * idio_spec.H')
    is_stable_id = maximum(abs, eigvals(idio_spec.A); init=zero(TW)) < one(TW)
    is_stable_id || _ri_check_detectable(idio_spec.A, B_id_scaled, Obase_id)
    Sigma_uncond_id = is_stable_id ? _rimw_symmetrize(solve_lyapunov(idio_spec.A, B_id_scaled)) : nothing
    Xi_zero_id = is_stable_id ?
        _rimw_symmetrize(_dlyap(sqrt(idio_spec.beta) * Matrix{TW}(idio_spec.A'), Obase_id;
                                tol=TW(1e-13), warn_label="solve_mw_attention")) :
        nothing
    warm_id = Ref{Any}(nothing)

    # Smart initial shadow-cost hint from idiosyncratic block
    om_hint = if fixed_om !== nothing
        fixed_om * (scale_B^2)
    else
        cap_guess = clamp(cap_target * TW(0.8), TW(0.05), cap_target)
        p_hint = RationalInattentionProblem(
            idio_prob_scaled.A, idio_prob_scaled.B, idio_prob_scaled.H;
            M=idio_prob_scaled.M, beta=idio_prob_scaled.beta,
            capacity=cap_guess, units=units,
        )
        try
            solve_ri_kalman(p_hint; method=:water_filling, tol=TW(1e-7), max_iter=1000).omega
        catch
            one(TW)
        end
    end

    # Partial equilibrium initialization on non-augmented agg_spec
    B_pe_scaled = static_pe.B_aug .* scale_B
    Q_pe_scaled = _rimw_symmetrize(B_pe_scaled * B_pe_scaled')
    Obase_pe = _rimw_symmetrize(agg_spec.H * agg_spec.M * agg_spec.H')
    is_stable_agg = maximum(abs, eigvals(agg_spec.A); init=zero(TW)) < one(TW)
    is_stable_agg || _ri_check_detectable(agg_spec.A, B_pe_scaled, Obase_pe)
    Sigma_uncond_pe = is_stable_agg ? _rimw_symmetrize(solve_lyapunov(agg_spec.A, B_pe_scaled)) : nothing
    Xi_zero_pe = is_stable_agg ?
        _rimw_symmetrize(_dlyap(sqrt(agg_spec.beta) * Matrix{TW}(agg_spec.A'), Obase_pe;
                                tol=TW(1e-13), warn_label="solve_mw_attention")) :
        nothing
    warm_pe = Ref{Any}(nothing)

    m_a = size(agg_spec.B, 2)
    pe_res = _rimw_solve_given_y(
        agg_spec.A, B_pe_scaled, Q_pe_scaled, agg_spec.beta, agg_spec.H, agg_spec.M,
        Sigma_uncond_pe, Xi_zero_pe, warm_pe,
        idio_spec.A, B_id_scaled, Q_id_scaled, idio_spec.beta, Obase_id,
        Sigma_uncond_id, Xi_zero_id, warm_id,
        units, scale_B, cap_target, fixed_om, om_hint, solve_tol, brent_tol,
    )
    om_cur = pe_res.omega_scaled
    k_agg_cur = _rimw_nats_to_units(pe_res.fp_agg.Inats, units)
    sig_pe = _ri_extract_signals(
        pe_res.fp_agg.Sigma_prior, pe_res.fp_agg.Sigma_post, pe_res.fp_agg.Omega_eff,
        om_cur, :water_filling, pe_res.fp_agg.L_env,
    )
    p_cur = _rimw_consensus_irf_mats(
        agg_spec.A, static_pe.B_aug, agg_spec.H, sig_pe.K_gain, sig_pe.Y_signal, L_lag,
    )
    y_cur = q_irf .- p_cur

    converged = pe_res.fp_agg.converged && pe_res.fp_id.converged
    iters = 1

    warm_agg = Ref{Any}(nothing)
    if r > zero(TW)
        B_aug_scaled = static_aug.B_aug .* scale_B
        Q_aug_scaled = _rimw_symmetrize(B_aug_scaled * B_aug_scaled')
        Sigma_uncond_aug = is_stable_agg ?
            _rimw_symmetrize(solve_lyapunov(static_aug.A_aug, B_aug_scaled)) : nothing

        p_hist = Vector{TW}[]
        p_resids = Vector{TW}[]
        prev_res_norm = TW(Inf)
        err_prev = TW(1.0)
        damp = TW(0.85)
        converged = false
        iters = 0

        for iter in 1:maxiter
            iters = iter
            tol_k = max(solve_tol, min(TW(1e-5), err_prev * TW(1e-3)))
            btol_k = max(brent_tol, min(TW(1e-5), err_prev * TW(1e-3)))

            H_aug_k = _rimw_make_H_aug(agg_spec, static_aug, y_cur, r, L_lag)
            Obase_aug_k = _rimw_symmetrize(H_aug_k * agg_spec.M * H_aug_k')
            is_stable_agg || _ri_check_detectable(static_aug.A_aug, B_aug_scaled, Obase_aug_k)
            Xi_zero_aug_k = is_stable_agg ?
                _rimw_symmetrize(_dlyap(sqrt(agg_spec.beta) * Matrix{TW}(static_aug.A_aug'), Obase_aug_k;
                                        tol=TW(1e-13), warn_label="solve_mw_attention")) :
                nothing

            res_k = _rimw_solve_given_y(
                static_aug.A_aug, B_aug_scaled, Q_aug_scaled, agg_spec.beta, H_aug_k, agg_spec.M,
                Sigma_uncond_aug, Xi_zero_aug_k, warm_agg,
                idio_spec.A, B_id_scaled, Q_id_scaled, idio_spec.beta, Obase_id,
                Sigma_uncond_id, Xi_zero_id, warm_id,
                units, scale_B, cap_target, fixed_om, om_cur, tol_k, btol_k;
                warm_bracket=(iter > 1),
            )
            om_new = res_k.omega_scaled
            k_agg_new = _rimw_nats_to_units(res_k.fp_agg.Inats, units)
            k_id_new = _rimw_nats_to_units(res_k.fp_id.Inats, units)
            tot_k = max(k_agg_new + k_id_new, one(TW))

            sig_k = _ri_extract_signals(
                res_k.fp_agg.Sigma_prior, res_k.fp_agg.Sigma_post, res_k.fp_agg.Omega_eff,
                om_new, :water_filling, res_k.fp_agg.L_env,
            )
            p_new = _rimw_consensus_irf_mats(
                static_aug.A_aug, static_aug.B_aug, H_aug_k, sig_k.K_gain, sig_k.Y_signal, L_lag,
            )

            err_p = maximum(abs, p_new .- p_cur) / scale_q
            err_k = abs(k_agg_new - k_agg_cur) / tot_k
            err_cur = max(err_p, err_k)

            om_cur = om_new
            k_agg_cur = k_agg_new

            if err_cur < TW(tol) && tol_k <= solve_tol * TW(1.1)
                p_cur = p_new
                y_cur = q_irf .- p_cur
                converged = res_k.fp_agg.converged && res_k.fp_id.converged
                break
            end

            xk = vec(copy(p_cur))
            rk = vec(p_new .- p_cur)
            rn = norm(rk)
            if isfinite(prev_res_norm) && rn > TW(2) * prev_res_norm && length(p_hist) >= 2
                empty!(p_hist)
                empty!(p_resids)
            end
            prev_res_norm = rn
            push!(p_hist, xk)
            push!(p_resids, rk)
            if length(p_hist) > 6
                popfirst!(p_hist)
                popfirst!(p_resids)
            end

            p_next = if length(p_hist) >= 2
                aa = _anderson_step(p_hist, p_resids, 5)
                all(isfinite, aa) ? reshape(aa, L_lag, m_a) :
                    (one(TW) - damp) .* p_cur .+ damp .* p_new
            else
                (one(TW) - damp) .* p_cur .+ damp .* p_new
            end

            p_cur = p_next
            y_cur = q_irf .- p_cur
            err_prev = err_cur
        end
    end

    # Recover unscaled omega and unscaled SimsKalmanSolutions (reuse converged warm states)
    omega_true = om_cur / (scale_B^2)
    final_tol = min(TW(tol) * TW(0.05), TW(1e-11))
    agg_prob_final = _rimw_make_agg_prob(
        agg_spec, static_aug, y_cur, r, L_lag; omega=omega_true, scale_B=one(TW),
    )
    warm_agg_final = if r > zero(TW) && warm_agg[] !== nothing
        (warm_agg[][1] ./ (scale_B^2), warm_agg[][2], nothing)
    elseif r == zero(TW) && warm_pe[] !== nothing
        (warm_pe[][1] ./ (scale_B^2), warm_pe[][2], nothing)
    else
        nothing
    end
    warm_id_final = warm_id[] === nothing ? nothing :
        (warm_id[][1] ./ (scale_B^2), warm_id[][2], nothing)

    agg_prob_out = T === TW ? agg_prob_final : RationalInattentionProblem(
        Matrix{T}(agg_prob_final.A), Matrix{T}(agg_prob_final.B), Matrix{T}(agg_prob_final.H);
        M=Matrix{T}(agg_prob_final.M), beta=T(agg_prob_final.beta),
        omega=T(omega_true), units=units,
        state_names=agg_prob_final.state_names,
        action_names=agg_prob_final.action_names,
        shock_names=agg_prob_final.shock_names,
    )
    agg_sol_out = _solve_ri_kalman_impl(
        TW, agg_prob_out, :water_filling, max(final_tol, TW(10) * eps(TW)), 2000;
        warm_state=warm_agg_final,
    )

    idio_prob_final = RationalInattentionProblem(
        idio_spec.A, idio_spec.B, idio_spec.H;
        M=idio_spec.M, beta=idio_spec.beta, omega=omega_true, units=units,
        state_names=idio_spec.state_names, action_names=idio_spec.action_names,
        shock_names=idio_spec.shock_names,
    )
    idio_prob_out = T === TW ? idio_prob_final : RationalInattentionProblem(
        Matrix{T}(idio_prob_final.A), Matrix{T}(idio_prob_final.B), Matrix{T}(idio_prob_final.H);
        M=Matrix{T}(idio_prob_final.M), beta=T(idio_prob_final.beta),
        omega=T(omega_true), units=units,
        state_names=idio_prob_final.state_names,
        action_names=idio_prob_final.action_names,
        shock_names=idio_prob_final.shock_names,
    )
    idio_sol_out = _solve_ri_kalman_impl(
        TW, idio_prob_out, :water_filling, max(final_tol, TW(10) * eps(TW)), 2000;
        warm_state=warm_id_final,
    )

    p_final = _rimw_consensus_irf_mats(
        Matrix{TW}(agg_prob_final.A), static_aug.B_aug, Matrix{TW}(agg_prob_final.H),
        Matrix{TW}(agg_sol_out.K_gain), Matrix{TW}(agg_sol_out.Y_signal), horizon,
    )[:, 1]
    y_final = q_irf[1:horizon, 1] .- p_final
    p_idio_final = _rimw_consensus_irf_mats(
        Matrix{TW}(idio_prob_final.A), idio_spec.B, Matrix{TW}(idio_prob_final.H),
        Matrix{TW}(idio_sol_out.K_gain), Matrix{TW}(idio_sol_out.Y_signal), horizon,
    )[:, 1]

    k_agg_val = units === :bits ? TW(agg_sol_out.capacity_bits) : TW(agg_sol_out.capacity_nats)
    k_id_val = units === :bits ? TW(idio_sol_out.capacity_bits) : TW(idio_sol_out.capacity_nats)
    k_tot_val = k_agg_val + k_id_val

    return MWAttentionResult{T}(
        T(k_agg_val),
        T(k_id_val),
        T(k_tot_val),
        T(omega_true),
        T(r),
        units,
        Vector{T}(p_final),
        Vector{T}(p_idio_final),
        Vector{T}(y_final),
        agg_sol_out,
        idio_sol_out,
        converged,
        iters,
    )
end

# =============================================================================
# Display & Reporting
# =============================================================================

function Base.show(io::IO, r::MWAttentionResult{T}) where {T}
    share_agg = r.total_kappa > zero(T) ? T(100) * r.kappa_agg / r.total_kappa : zero(T)
    share_id = r.total_kappa > zero(T) ? T(100) * r.kappa_idio / r.total_kappa : zero(T)
    peak_val, peak_idx = findmax(abs, r.real_output_irf)
    u_lbl = string(r.units)

    spec = Any[
        "Total capacity κ ($(u_lbl))"    _fmt(r.total_kappa);
        "  Aggregate κ_agg ($(_fmt(share_agg; digits=1))%)"   _fmt(r.kappa_agg);
        "  Idiosyncratic κ_idio ($(_fmt(share_id; digits=1))%)" _fmt(r.kappa_idio);
        "Common shadow cost ω"           _fmt(r.omega);
        "Strategic complementarity r"      _fmt(r.strategic_complementarity);
        "Peak |real output IRF|"         "$(_fmt(peak_val)) (h = $(peak_idx))";
        "Horizon"                        length(r.real_output_irf);
        "Converged"                      "$(_yesno(r.converged)) ($(r.iterations) iters)";
        "Precision"                      string(T);
    ]
    _pretty_table(io, spec;
        title = "Maćkowiak–Wiederholt (2009) Multi-Block RI Equilibrium",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

function report(io::IO, r::MWAttentionResult{T}) where {T}
    show(io, r)
    H = length(r.agg_price_irf)
    h_idx = _select_horizons(H)
    tbl = Matrix{Any}(undef, length(h_idx), 4)
    for (row, h) in enumerate(h_idx)
        tbl[row, 1] = h
        tbl[row, 2] = _fmt(r.agg_price_irf[h])
        tbl[row, 3] = _fmt(r.real_output_irf[h])
        tbl[row, 4] = _fmt(r.idio_price_irf[h])
    end
    println(io)
    _pretty_table(io, tbl;
        title = "Equilibrium Impulse Responses (1-s.d. Shocks)",
        column_labels = ["Horizon h", "Agg. Price p_h", "Real Output y_h", "Idio. Price p_h^idio"],
        alignment = [:r, :r, :r, :r],
    )
    return nothing
end
report(r::MWAttentionResult) = report(stdout, r)
