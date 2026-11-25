# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

# Built-in GPL-3.0-compatible interior-point constrained DSGE solvers via
# MadNLP (MIT) + NLPModels (MPL-2.0) + LDLFactorizations (LGPL-3.0+) / SuiteSparse (GPL-2.0+).
#
# Replaces Ipopt (EPL-2.0) / MUMPS_seq (CeCILL-C) as the default large-scale
# constrained NLP backend (issue #930). Formulates constrained steady state and
# perfect foresight directly as sparse `NLPModels.AbstractNLPModel` instances
# with exact ForwardDiff Jacobians and block-pentadiagonal Lagrangian Hessians,
# eliminating JuMP operator construction and C-callback trampoline overhead.

# =============================================================================
# Steady-State NLPModel
# =============================================================================

struct _DSGESSNLPModel{FObj, FCons} <: NLPModels.AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
    obj_fn::FObj
    con_fns::FCons
    n::Int
    m::Int
end

function _build_ss_vec_objective(residual_fns, n_ε::Int, θ)
    function ss_obj(y::AbstractVector{S}) where {S<:Real}
        ε_z = zeros(S, n_ε)
        total = zero(S)
        for fn in residual_fns
            try
                r = fn(y, y, y, ε_z, θ)
                total += r * r
            catch e
                (e isa DomainError || e isa InexactError) && return S(NaN)
                rethrow(e)
            end
        end
        return total
    end
    return ss_obj
end

function _build_ss_vec_nlcon(cfn, n_ε::Int, θ)
    function ss_con(y::AbstractVector{S}) where {S<:Real}
        ε_z = zeros(S, n_ε)
        try
            return cfn(y, y, y, ε_z, θ)
        catch e
            (e isa DomainError || e isa InexactError) && return S(NaN)
            rethrow(e)
        end
    end
    return ss_con
end

function _DSGESSNLPModel(spec::ModelSpec, constraints::Vector, x0::Vector{Float64},
                         lvar::Vector{Float64}, uvar::Vector{Float64})
    n = spec.n_endog
    n_ε = spec.n_exog
    θ = spec.param_values
    obj_fn = _build_ss_vec_objective(spec.residual_fns, n_ε, θ)
    con_fns = Function[_build_ss_vec_nlcon(c.fn, n_ε, θ)
                       for c in constraints if c isa NonlinearConstraint]
    m = length(con_fns)
    lcon = fill(-Inf, m)
    ucon = zeros(Float64, m)
    nnzj = m * n
    nnzh = (n * (n + 1)) ÷ 2
    meta = NLPModels.NLPModelMeta(
        n;
        x0=x0,
        lvar=lvar,
        uvar=uvar,
        ncon=m,
        lcon=lcon,
        ucon=ucon,
        y0=zeros(Float64, m),
        nnzj=nnzj,
        nnzh=nnzh,
        minimize=true,
    )
    return _DSGESSNLPModel(meta, NLPModels.Counters(), obj_fn, con_fns, n, m)
end

function NLPModels.obj(nlp::_DSGESSNLPModel, x::AbstractVector)
    return Float64(nlp.obj_fn(x))
end

function NLPModels.grad!(nlp::_DSGESSNLPModel, x::AbstractVector, g::AbstractVector)
    try
        ForwardDiff.gradient!(g, nlp.obj_fn, x)
    catch e
        (e isa DomainError || e isa InexactError) ? fill!(g, NaN) : rethrow(e)
    end
    return g
end

function NLPModels.cons!(nlp::_DSGESSNLPModel, x::AbstractVector, c::AbstractVector)
    for k in 1:nlp.m
        c[k] = Float64(nlp.con_fns[k](x))
    end
    return c
end

function NLPModels.jac_structure!(nlp::_DSGESSNLPModel, rows::AbstractVector{<:Integer},
                                  cols::AbstractVector{<:Integer})
    idx = 0
    for j in 1:nlp.n, i in 1:nlp.m
        idx += 1
        rows[idx] = i
        cols[idx] = j
    end
    return rows, cols
end

function NLPModels.jac_coord!(nlp::_DSGESSNLPModel, x::AbstractVector, vals::AbstractVector)
    nlp.m == 0 && return vals
    try
        J = ForwardDiff.jacobian(z -> [nlp.con_fns[k](z) for k in 1:nlp.m], x)
        idx = 0
        for j in 1:nlp.n, i in 1:nlp.m
            idx += 1
            vals[idx] = Float64(J[i, j])
        end
    catch e
        (e isa DomainError || e isa InexactError) ? fill!(vals, NaN) : rethrow(e)
    end
    return vals
end

function NLPModels.hess_structure!(nlp::_DSGESSNLPModel, rows::AbstractVector{<:Integer},
                                   cols::AbstractVector{<:Integer})
    idx = 0
    for j in 1:nlp.n, i in j:nlp.n
        idx += 1
        rows[idx] = i
        cols[idx] = j
    end
    return rows, cols
end

function NLPModels.hess_coord!(nlp::_DSGESSNLPModel, x::AbstractVector, y::AbstractVector,
                               vals::AbstractVector; obj_weight::Real=1.0)
    ow = Float64(obj_weight)
    lag(z) = begin
        val = ow * nlp.obj_fn(z)
        for k in 1:nlp.m
            yk = y[k]
            yk != 0.0 && (val += yk * nlp.con_fns[k](z))
        end
        return val
    end
    try
        H = ForwardDiff.hessian(lag, x)
        idx = 0
        for j in 1:nlp.n, i in j:nlp.n
            idx += 1
            vals[idx] = Float64(H[i, j])
        end
    catch e
        (e isa DomainError || e isa InexactError) ? fill!(vals, NaN) : rethrow(e)
    end
    return vals
end

function NLPModels.hess_coord!(nlp::_DSGESSNLPModel, x::AbstractVector, vals::AbstractVector;
                               obj_weight::Real=1.0)
    return NLPModels.hess_coord!(nlp, x, Float64[], vals; obj_weight=obj_weight)
end

# =============================================================================
# Constrained Steady State via MadNLP
# =============================================================================

function _madnlp_compute_steady_state(
        spec::ModelSpec{T},
        constraints::Vector;
        initial_guess::Union{Nothing,AbstractVector}=nothing) where {T}

    n = spec.n_endog
    lower, upper = _extract_bounds(spec, constraints)
    lvar = Float64.(lower)
    uvar = Float64.(upper)

    x0 = if initial_guess !== nothing
        Float64.(initial_guess)
    elseif !isempty(spec.steady_state)
        Float64.(spec.steady_state)
    else
        ones(Float64, n)
    end
    @assert length(x0) == n "initial_guess must have length $n"
    x0 .= clamp.(x0, lvar, uvar)

    nlp = _DSGESSNLPModel(spec, constraints, x0, lvar, uvar)
    stats = MadNLP.madnlp(nlp;
        linear_solver=MadNLP.UmfpackSolver,
        print_level=MadNLP.ERROR,
        max_iter=3000,
        tol=1e-12,
        acceptable_tol=1e-10,
        rethrow_error=false,
    )

    if stats.status != MadNLP.SOLVE_SUCCEEDED &&
       stats.status != MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        @warn "Constrained steady state (MadNLP): solver status = $(stats.status)"
    end

    return Vector{T}(stats.solution)
end

# =============================================================================
# Perfect-Foresight Sparse NLPModel
# =============================================================================

struct _DSGEPFNLPModel{Spec} <: NLPModels.AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
    spec::Spec
    T_periods::Int
    n::Int
    n_ε::Int
    m_nl::Int
    y_ss::Vector{Float64}
    shocks::Matrix{Float64}
    nl_fns::Vector{Function}
    # Per-period active block lists (subset of [t-1, t, t+1] in ascending order)
    blocks::Vector{Vector{Int}}
    jac_period_off::Vector{Int}
    jac_rows::Vector{Int}
    jac_cols::Vector{Int}
    # Lower-triangular block-pentadiagonal Hessian COO layout
    diag_off::Vector{Int}
    sub1_off::Vector{Int}
    sub2_off::Vector{Int}
    hess_rows::Vector{Int}
    hess_cols::Vector{Int}
end

function _DSGEPFNLPModel(spec::ModelSpec, T_periods::Int, shocks::Matrix{Float64},
                         constraints::Vector)
    n = spec.n_endog
    n_ε = spec.n_exog
    N = T_periods * n
    y_ss = Float64.(spec.steady_state)

    nl_fns = Function[c.fn for c in constraints if c isa NonlinearConstraint]
    m_nl = length(nl_fns)
    n_con_t = n + m_nl
    M = T_periods * n_con_t

    lower, upper = _extract_bounds(spec, constraints)
    lvar = repeat(Float64.(lower), T_periods)
    uvar = repeat(Float64.(upper), T_periods)
    x0 = clamp.(repeat(y_ss, T_periods), lvar, uvar)

    # First T*n rows: equilibrium equalities (= 0); next T*m_nl rows: inequalities (<= 0)
    lcon = Vector{Float64}(undef, M)
    ucon = zeros(Float64, M)
    lcon[1:T_periods*n] .= 0.0
    if m_nl > 0
        lcon[T_periods*n+1:M] .= -Inf
    end

    # Precompute Jacobian COO sparsity (exact block-tridiagonal structure)
    blocks = Vector{Vector{Int}}(undef, T_periods)
    jac_period_off = Vector{Int}(undef, T_periods)
    jac_rows = Int[]
    jac_cols = Int[]
    for t in 1:T_periods
        bt = Int[]
        t > 1 && push!(bt, t - 1)
        push!(bt, t)
        t < T_periods && push!(bt, t + 1)
        blocks[t] = bt
        jac_period_off[t] = length(jac_rows)
        # Match column-major ordering of ForwardDiff.jacobian (size n_con_t × (length(bt)*n))
        for b in bt
            col_base = (b - 1) * n
            for j in 1:n
                col = col_base + j
                eq_base = (t - 1) * n
                for i in 1:n
                    push!(jac_rows, eq_base + i)
                    push!(jac_cols, col)
                end
                if m_nl > 0
                    nl_base = T_periods * n + (t - 1) * m_nl
                    for k in 1:m_nl
                        push!(jac_rows, nl_base + k)
                        push!(jac_cols, col)
                    end
                end
            end
        end
    end
    nnzj = length(jac_rows)

    # Precompute unique lower-triangular block-pentadiagonal Hessian COO structure:
    # diagonal blocks (t, t), sub-diagonal (t+1, t), and second sub-diagonal (t+2, t)
    # (period t+1 couples y_t and y_{t+2} through its lag and lead arguments).
    hess_rows = Int[]
    hess_cols = Int[]
    diag_off = Vector{Int}(undef, T_periods)
    sub1_off = Vector{Int}(undef, max(0, T_periods - 1))
    sub2_off = Vector{Int}(undef, max(0, T_periods - 2))

    for t in 1:T_periods
        diag_off[t] = length(hess_rows)
        base = (t - 1) * n
        for j in 1:n, i in j:n
            push!(hess_rows, base + i)
            push!(hess_cols, base + j)
        end
    end
    for t in 1:(T_periods - 1)
        sub1_off[t] = length(hess_rows)
        rbase = t * n
        cbase = (t - 1) * n
        for j in 1:n, i in 1:n
            push!(hess_rows, rbase + i)
            push!(hess_cols, cbase + j)
        end
    end
    for t in 1:(T_periods - 2)
        sub2_off[t] = length(hess_rows)
        rbase = (t + 1) * n
        cbase = (t - 1) * n
        for j in 1:n, i in 1:n
            push!(hess_rows, rbase + i)
            push!(hess_cols, cbase + j)
        end
    end
    nnzh = length(hess_rows)

    meta = NLPModels.NLPModelMeta(
        N;
        x0=x0,
        lvar=lvar,
        uvar=uvar,
        ncon=M,
        lcon=lcon,
        ucon=ucon,
        y0=zeros(Float64, M),
        nnzj=nnzj,
        nnzh=nnzh,
        minimize=true,
    )
    return _DSGEPFNLPModel(
        meta, NLPModels.Counters(), spec, T_periods, n, n_ε, m_nl,
        y_ss, shocks, nl_fns, blocks, jac_period_off, jac_rows, jac_cols,
        diag_off, sub1_off, sub2_off, hess_rows, hess_cols,
    )
end

# Extract (y_t, y_lag, y_lead) views from a period's packed local vector z
@inline function _pf_unpack_local(z::AbstractVector{S}, t::Int, Tp::Int, n::Int,
                                  y_ss::Vector{Float64}) where {S<:Real}
    if Tp == 1
        y_t = @view z[1:n]
        y_lag = S.(y_ss)
        y_lead = S.(y_ss)
    elseif t == 1
        y_t = @view z[1:n]
        y_lag = S.(y_ss)
        y_lead = @view z[n+1:2n]
    elseif t == Tp
        y_lag = @view z[1:n]
        y_t = @view z[n+1:2n]
        y_lead = S.(y_ss)
    else
        y_lag = @view z[1:n]
        y_t = @view z[n+1:2n]
        y_lead = @view z[2n+1:3n]
    end
    return y_t, y_lag, y_lead
end

function _pf_pack_local(x::AbstractVector, bt::Vector{Int}, n::Int)
    z = Vector{Float64}(undef, length(bt) * n)
    for (p, b) in enumerate(bt)
        src_off = (b - 1) * n
        dst_off = (p - 1) * n
        @inbounds for j in 1:n
            z[dst_off + j] = Float64(x[src_off + j])
        end
    end
    return z
end

function _pf_eval_period_cons(nlp::_DSGEPFNLPModel, t::Int, z::AbstractVector{S}) where {S<:Real}
    n = nlp.n
    m_nl = nlp.m_nl
    θ = nlp.spec.param_values
    fns = nlp.spec.residual_fns
    ε_t = @view nlp.shocks[t, :]
    y_t, y_lag, y_lead = _pf_unpack_local(z, t, nlp.T_periods, n, nlp.y_ss)
    out = Vector{S}(undef, n + m_nl)
    for i in 1:n
        out[i] = try
            S(fns[i](y_t, y_lag, y_lead, ε_t, θ))
        catch e
            (e isa DomainError || e isa InexactError) ? S(NaN) : rethrow(e)
        end
    end
    for k in 1:m_nl
        out[n + k] = try
            S(nlp.nl_fns[k](y_t, y_lag, y_lead, ε_t, θ))
        catch e
            (e isa DomainError || e isa InexactError) ? S(NaN) : rethrow(e)
        end
    end
    return out
end

function _pf_eval_period_lag(nlp::_DSGEPFNLPModel, t::Int, z::AbstractVector{S},
                             y::AbstractVector) where {S<:Real}
    n = nlp.n
    m_nl = nlp.m_nl
    Tp = nlp.T_periods
    θ = nlp.spec.param_values
    fns = nlp.spec.residual_fns
    ε_t = @view nlp.shocks[t, :]
    y_t, y_lag, y_lead = _pf_unpack_local(z, t, Tp, n, nlp.y_ss)
    acc = zero(S)
    eq_base = (t - 1) * n
    for i in 1:n
        yi = y[eq_base + i]
        yi == 0.0 && continue
        r = try
            S(fns[i](y_t, y_lag, y_lead, ε_t, θ))
        catch e
            (e isa DomainError || e isa InexactError) && return S(NaN)
            rethrow(e)
        end
        acc += yi * r
    end
    if m_nl > 0
        nl_base = Tp * n + (t - 1) * m_nl
        for k in 1:m_nl
            yk = y[nl_base + k]
            yk == 0.0 && continue
            g = try
                S(nlp.nl_fns[k](y_t, y_lag, y_lead, ε_t, θ))
            catch e
                (e isa DomainError || e isa InexactError) && return S(NaN)
                rethrow(e)
            end
            acc += yk * g
        end
    end
    return acc
end

NLPModels.obj(::_DSGEPFNLPModel, ::AbstractVector) = 0.0

function NLPModels.grad!(::_DSGEPFNLPModel, ::AbstractVector, g::AbstractVector)
    fill!(g, 0.0)
    return g
end

function NLPModels.cons!(nlp::_DSGEPFNLPModel, x::AbstractVector, c::AbstractVector)
    n = nlp.n
    m_nl = nlp.m_nl
    Tp = nlp.T_periods
    θ = nlp.spec.param_values
    fns = nlp.spec.residual_fns
    for t in 1:Tp
        y_t = @view x[(t - 1) * n + 1 : t * n]
        y_lag = t == 1 ? nlp.y_ss : @view x[(t - 2) * n + 1 : (t - 1) * n]
        y_lead = t == Tp ? nlp.y_ss : @view x[t * n + 1 : (t + 1) * n]
        ε_t = @view nlp.shocks[t, :]
        eq_base = (t - 1) * n
        for i in 1:n
            c[eq_base + i] = try
                Float64(fns[i](y_t, y_lag, y_lead, ε_t, θ))
            catch e
                (e isa DomainError || e isa InexactError) ? NaN : rethrow(e)
            end
        end
        if m_nl > 0
            nl_base = Tp * n + (t - 1) * m_nl
            for k in 1:m_nl
                c[nl_base + k] = try
                    Float64(nlp.nl_fns[k](y_t, y_lag, y_lead, ε_t, θ))
                catch e
                    (e isa DomainError || e isa InexactError) ? NaN : rethrow(e)
                end
            end
        end
    end
    return c
end

function NLPModels.jac_structure!(nlp::_DSGEPFNLPModel, rows::AbstractVector{<:Integer},
                                  cols::AbstractVector{<:Integer})
    copyto!(rows, nlp.jac_rows)
    copyto!(cols, nlp.jac_cols)
    return rows, cols
end

function NLPModels.jac_coord!(nlp::_DSGEPFNLPModel, x::AbstractVector, vals::AbstractVector)
    n = nlp.n
    n_con_t = n + nlp.m_nl
    for t in 1:nlp.T_periods
        bt = nlp.blocks[t]
        z_t = _pf_pack_local(x, bt, n)
        off = nlp.jac_period_off[t]
        n_cols_t = length(bt) * n
        try
            J_loc = ForwardDiff.jacobian(z -> _pf_eval_period_cons(nlp, t, z), z_t)
            k = off
            @inbounds for j in 1:n_cols_t, i in 1:n_con_t
                k += 1
                vals[k] = Float64(J_loc[i, j])
            end
        catch e
            if e isa DomainError || e isa InexactError
                vals[off + 1 : off + n_con_t * n_cols_t] .= NaN
            else
                rethrow(e)
            end
        end
    end
    return vals
end

function NLPModels.hess_structure!(nlp::_DSGEPFNLPModel, rows::AbstractVector{<:Integer},
                                   cols::AbstractVector{<:Integer})
    copyto!(rows, nlp.hess_rows)
    copyto!(cols, nlp.hess_cols)
    return rows, cols
end

function NLPModels.hess_coord!(nlp::_DSGEPFNLPModel, x::AbstractVector, y::AbstractVector,
                               vals::AbstractVector; obj_weight::Real=1.0)
    fill!(vals, 0.0)
    n = nlp.n
    m_nl = nlp.m_nl
    Tp = nlp.T_periods
    for t in 1:Tp
        # Skip period if all associated multipliers are zero
        eq_base = (t - 1) * n
        nl_base = Tp * n + (t - 1) * m_nl
        has_nz = false
        @inbounds for i in 1:n
            if y[eq_base + i] != 0.0
                has_nz = true
                break
            end
        end
        if !has_nz && m_nl > 0
            @inbounds for k in 1:m_nl
                if y[nl_base + k] != 0.0
                    has_nz = true
                    break
                end
            end
        end
        has_nz || continue

        bt = nlp.blocks[t]
        z_t = _pf_pack_local(x, bt, n)
        H_loc = try
            ForwardDiff.hessian(z -> _pf_eval_period_lag(nlp, t, z, y), z_t)
        catch e
            if e isa DomainError || e isa InexactError
                fill!(vals, NaN)
                return vals
            end
            rethrow(e)
        end

        nb = length(bt)
        @inbounds for pc in 1:nb, pr in pc:nb
            bc = bt[pc]
            br = bt[pr]
            r0 = (pr - 1) * n
            c0 = (pc - 1) * n
            if br == bc
                idx = nlp.diag_off[br]
                for j in 1:n, i in j:n
                    idx += 1
                    vals[idx] += Float64(H_loc[r0 + i, c0 + j])
                end
            elseif br == bc + 1
                idx = nlp.sub1_off[bc]
                for j in 1:n, i in 1:n
                    idx += 1
                    vals[idx] += Float64(H_loc[r0 + i, c0 + j])
                end
            elseif br == bc + 2
                idx = nlp.sub2_off[bc]
                for j in 1:n, i in 1:n
                    idx += 1
                    vals[idx] += Float64(H_loc[r0 + i, c0 + j])
                end
            end
        end
    end
    return vals
end

function NLPModels.hess_coord!(nlp::_DSGEPFNLPModel, x::AbstractVector, vals::AbstractVector;
                               obj_weight::Real=1.0)
    fill!(vals, 0.0)
    return vals
end

# =============================================================================
# Constrained Perfect Foresight via MadNLP
# =============================================================================

function _madnlp_perfect_foresight(
        spec::ModelSpec{FT},
        T_periods::Int,
        shocks::Matrix{FT},
        constraints::Vector) where {FT}

    n = spec.n_endog
    y_ss = spec.steady_state
    shocks_f = Matrix{Float64}(shocks)

    nlp = _DSGEPFNLPModel(spec, T_periods, shocks_f, constraints)
    stats = MadNLP.madnlp(nlp;
        linear_solver=MadNLP.UmfpackSolver,
        print_level=MadNLP.ERROR,
        max_iter=3000,
        tol=1e-8,
        acceptable_tol=1e-6,
        rethrow_error=false,
    )

    converged = (stats.status == MadNLP.SOLVE_SUCCEEDED ||
                 stats.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)
    if !converged
        @warn "Constrained perfect foresight (MadNLP): solver status = $(stats.status)"
    end

    # Extract solution path (reshape n × T_periods -> T_periods × n)
    sol_vec = stats.solution
    path_full = Matrix{FT}(undef, T_periods, n)
    @inbounds for t in 1:T_periods, i in 1:n
        path_full[t, i] = FT(sol_vec[(t - 1) * n + i])
    end
    deviations_full = path_full .- y_ss'

    if spec.augmented
        orig_idx = _original_var_indices(spec)
        path = Matrix{FT}(path_full[:, orig_idx])
        deviations = Matrix{FT}(deviations_full[:, orig_idx])
    else
        path = Matrix{FT}(path_full)
        deviations = Matrix{FT}(deviations_full)
    end

    iter = Int(stats.iter)
    return PerfectForesightPath{FT}(path, deviations, converged, iter, spec)
end
