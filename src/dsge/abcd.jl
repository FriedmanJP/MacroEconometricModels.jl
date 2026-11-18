# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Canonical ABCD linear state-space representation (Fernandez-Villaverde, Rubio-Ramirez,
Sargent & Watson 2007) and minimal predetermined-state extraction for DSGE, perturbation,
heterogeneous-agent DSGE, and general state-space models.

References:
- Fernández-Villaverde, J., Rubio-Ramírez, J. F., Sargent, T. J., & Watson, M. W. (2007).
  ABCs (and Ds) of Understanding VARs. *American Economic Review*, 97(3), 1021–1026.
"""

# =============================================================================
# ABCDSystem{T} — canonical FVRSW (2007) state-space representation
# =============================================================================

"""
    ABCDSystem{T<:AbstractFloat}

Canonical linear state-space system in Fernández-Villaverde, Rubio-Ramírez, Sargent, and
Watson (2007, eqs. 1–2) ABCD form:

```math
x_{t+1} = c_{\\text{state}} + A x_t + B w_{t+1}, \\qquad E[w_t w_t'] = I_m
```
```math
y_{t+1} = d_{\\text{obs}} + C x_t + D w_{t+1}
```

# Fields
- `A::Matrix{T}` — `n × n` state transition matrix
- `B::Matrix{T}` — `n × m` shock-to-state impact matrix
- `C::Matrix{T}` — `k × n` state-to-observable observation matrix
- `D::Matrix{T}` — `k × m` shock-to-observable impact matrix
- `c_state::Vector{T}` — length-`n` state intercept
- `d_obs::Vector{T}` — length-`k` observable intercept
- `state_names::Vector{String}` — length-`n` state labels
- `obs_names::Vector{String}` — length-`k` observable labels
- `shock_names::Vector{String}` — length-`m` shock labels
"""
struct ABCDSystem{T<:AbstractFloat}
    A::Matrix{T}
    B::Matrix{T}
    C::Matrix{T}
    D::Matrix{T}
    c_state::Vector{T}
    d_obs::Vector{T}
    state_names::Vector{String}
    obs_names::Vector{String}
    shock_names::Vector{String}

    function ABCDSystem{T}(A::AbstractMatrix, B::AbstractMatrix,
                            C::AbstractMatrix, D::AbstractMatrix,
                            c_state::AbstractVector, d_obs::AbstractVector,
                            state_names::AbstractVector,
                            obs_names::AbstractVector,
                            shock_names::AbstractVector) where {T<:AbstractFloat}
        Am = Matrix{T}(A)
        Bm = Matrix{T}(B)
        Cm = Matrix{T}(C)
        Dm = Matrix{T}(D)
        cs = Vector{T}(c_state)
        ds = Vector{T}(d_obs)
        sn = String[string(s) for s in state_names]
        on = String[string(s) for s in obs_names]
        wn = String[string(s) for s in shock_names]

        n = size(Am, 1)
        size(Am, 2) == n || throw(ArgumentError("A must be square, got $(size(Am))"))
        size(Bm, 1) == n || throw(ArgumentError("B must have n=$n rows, got $(size(Bm, 1))"))
        m = size(Bm, 2)
        k = size(Cm, 1)
        size(Cm, 2) == n || throw(ArgumentError("C must have n=$n columns, got $(size(Cm, 2))"))
        size(Dm) == (k, m) || throw(ArgumentError("D must be k×m = ($k, $m), got $(size(Dm))"))
        length(cs) == n || throw(ArgumentError("c_state must have length n=$n, got $(length(cs))"))
        length(ds) == k || throw(ArgumentError("d_obs must have length k=$k, got $(length(ds))"))
        length(sn) == n || throw(ArgumentError("state_names must have length n=$n, got $(length(sn))"))
        length(on) == k || throw(ArgumentError("obs_names must have length k=$k, got $(length(on))"))
        length(wn) == m || throw(ArgumentError("shock_names must have length m=$m, got $(length(wn))"))

        new{T}(Am, Bm, Cm, Dm, cs, ds, sn, on, wn)
    end
end

function ABCDSystem(A::AbstractMatrix, B::AbstractMatrix,
                    C::AbstractMatrix, D::AbstractMatrix,
                    c_state::AbstractVector, d_obs::AbstractVector,
                    state_names::AbstractVector,
                    obs_names::AbstractVector,
                    shock_names::AbstractVector)
    T = float(promote_type(eltype(A), eltype(B), eltype(C), eltype(D),
                           eltype(c_state), eltype(d_obs)))
    return ABCDSystem{T}(A, B, C, D, c_state, d_obs, state_names, obs_names, shock_names)
end

function ABCDSystem{T}(A::AbstractMatrix, B::AbstractMatrix,
                       C::AbstractMatrix, D::AbstractMatrix;
                       c_state::Union{Nothing,AbstractVector}=nothing,
                       d_obs::Union{Nothing,AbstractVector}=nothing,
                       state_names::Union{Nothing,AbstractVector}=nothing,
                       obs_names::Union{Nothing,AbstractVector}=nothing,
                       shock_names::Union{Nothing,AbstractVector}=nothing) where {T<:AbstractFloat}
    n = size(A, 1)
    m = size(B, 2)
    k = size(C, 1)
    cs = c_state === nothing ? zeros(T, n) : c_state
    ds = d_obs === nothing ? zeros(T, k) : d_obs
    sn = state_names === nothing ? ["x_$i" for i in 1:n] : state_names
    on = obs_names === nothing ? ["y_$i" for i in 1:k] : obs_names
    wn = shock_names === nothing ? ["w_$j" for j in 1:m] : shock_names
    return ABCDSystem{T}(A, B, C, D, cs, ds, sn, on, wn)
end

"""
    ABCDSystem(A, B, C, D; c_state=nothing, d_obs=nothing,
               state_names=nothing, obs_names=nothing, shock_names=nothing)

Construct an `ABCDSystem{T}` with automatic floating-point type promotion and default zero
intercepts and `"x_i"`, `"y_i"`, `"w_j"` labels.
"""
function ABCDSystem(A::AbstractMatrix, B::AbstractMatrix,
                    C::AbstractMatrix, D::AbstractMatrix;
                    c_state::Union{Nothing,AbstractVector}=nothing,
                    d_obs::Union{Nothing,AbstractVector}=nothing,
                    state_names::Union{Nothing,AbstractVector}=nothing,
                    obs_names::Union{Nothing,AbstractVector}=nothing,
                    shock_names::Union{Nothing,AbstractVector}=nothing)
    T = float(promote_type(
        eltype(A), eltype(B), eltype(C), eltype(D),
        c_state === nothing ? Bool : eltype(c_state),
        d_obs === nothing ? Bool : eltype(d_obs),
    ))
    return ABCDSystem{T}(A, B, C, D; c_state=c_state, d_obs=d_obs,
                         state_names=state_names, obs_names=obs_names, shock_names=shock_names)
end

# =============================================================================
# Queries & structural methods
# =============================================================================

"""
    nstates(sys::ABCDSystem) -> Int

Number of state variables (`size(sys.A, 1)`) in an `ABCDSystem`.
"""
nstates(sys::ABCDSystem) = size(sys.A, 1)

"""
    nobservables(sys::ABCDSystem) -> Int

Number of observable variables (`size(sys.C, 1)`) in an `ABCDSystem`.
"""
nobservables(sys::ABCDSystem) = size(sys.C, 1)

"""
    nshocks(sys::ABCDSystem) -> Int

Number of structural/innovation shocks (`size(sys.B, 2)`) in an `ABCDSystem`.
"""
nshocks(sys::ABCDSystem) = size(sys.B, 2)

"""
    nvars(sys::ABCDSystem) -> Int

Number of observable variables (`nobservables(sys)`) in an `ABCDSystem`.
"""
nvars(sys::ABCDSystem) = nobservables(sys)

"""
    is_square(sys::ABCDSystem) -> Bool

Return `true` if `nobservables(sys) == nshocks(sys)` (square ABCD system).
"""
is_square(sys::ABCDSystem) = size(sys.C, 1) == size(sys.B, 2)

"""
    is_stable(sys::ABCDSystem{T}) -> Bool

Return `true` if `nstates(sys) == 0` or all eigenvalues of `sys.A` lie strictly inside
the unit circle (`maximum(abs.(eigvals(sys.A))) < 1`).
"""
is_stable(sys::ABCDSystem{T}) where {T} =
    size(sys.A, 1) == 0 || maximum(abs.(eigvals(sys.A))) < one(T)

"""
    irf(sys::ABCDSystem{T}, horizon::Int; ci_type::Symbol=:none) -> ImpulseResponse{T}

Compute analytical impulse responses from an `ABCDSystem`:
`Θ_1 = D` at impact (`h = 1`), and `Θ_h = C * A^(h - 2) * B` for `h ≥ 2`.
"""
function irf(sys::ABCDSystem{T}, horizon::Int; ci_type::Symbol=:none) where {T<:AbstractFloat}
    horizon >= 1 || throw(ArgumentError("horizon must be positive; got $horizon"))
    n = nstates(sys)
    k = nobservables(sys)
    m = nshocks(sys)

    point_irf = zeros(T, horizon, k, m)
    point_irf[1, :, :] = sys.D
    if horizon >= 2
        A_pow = Matrix{T}(I, n, n)
        for h in 2:horizon
            point_irf[h, :, :] = sys.C * (A_pow * sys.B)
            A_pow = A_pow * sys.A
        end
    end

    ci_lower = zeros(T, horizon, k, m)
    ci_upper = zeros(T, horizon, k, m)
    return ImpulseResponse{T}(point_irf, ci_lower, ci_upper, horizon,
                              copy(sys.obs_names), copy(sys.shock_names), ci_type)
end

"""
    fevd(sys::ABCDSystem{T}, horizon::Int) -> FEVD{T}

Compute analytical forecast error variance decomposition from an `ABCDSystem`.
"""
function fevd(sys::ABCDSystem{T}, horizon::Int) where {T<:AbstractFloat}
    horizon >= 1 || throw(ArgumentError("horizon must be positive; got $horizon"))
    ir = irf(sys, horizon)
    decomp, props = _fevd_accumulate(ir)
    return FEVD{T}(decomp, props, ir.variables, ir.shocks)
end

"""
    analytical_moments(sys::ABCDSystem{T}; lags=1, format=:vector)

Compute closed-form unconditional autocovariance moments of `y` in an `ABCDSystem`:
`Sigma_x = solve_lyapunov(A, B)`, `Gamma_y(0) = C * Sigma_x * C' + D * D'`, and
`Gamma_y(h) = C * A^(h - 1) * (A * Sigma_x * C' + B * D')` for `h ≥ 1`.

When `format === :vector` (default), returns the upper triangle of `Gamma_y(0)` followed by
`diag(Gamma_y(l))` for each positive lag in `lags`. When `format === :matrices`, returns
`[Gamma_y(0), Gamma_y(1), ..., Gamma_y(max_lag)]`.
"""
function analytical_moments(
    sys::ABCDSystem{T};
    lags::Union{Integer,AbstractVector{<:Integer}}=1,
    format::Symbol=:vector,
) where {T<:AbstractFloat}
    if lags isa Integer
        lags >= 0 || throw(ArgumentError("lags must be non-negative; got $lags"))
        lag_vec = lags > 0 ? collect(1:lags) : Int[]
    else
        all(>=(0), lags) || throw(ArgumentError("all lags must be non-negative"))
        lag_vec = collect(Int, lags)
    end
    format in (:vector, :matrices) ||
        throw(ArgumentError("format must be :vector or :matrices; got :$format"))

    n = nstates(sys)
    k = nobservables(sys)
    max_lag = isempty(lag_vec) ? 0 : maximum(lag_vec)

    Sigma_x = n == 0 ? zeros(T, 0, 0) : solve_lyapunov(sys.A, sys.B)
    G0_raw = sys.C * Sigma_x * sys.C' + sys.D * sys.D'
    Gamma0 = Matrix{T}((G0_raw + G0_raw') / 2)

    Gammas = Vector{Matrix{T}}(undef, max_lag)
    if max_lag > 0
        M_cross = sys.A * Sigma_x * sys.C' + sys.B * sys.D'
        A_pow = Matrix{T}(I, n, n)
        for h in 1:max_lag
            Gammas[h] = Matrix{T}(sys.C * (A_pow * M_cross))
            A_pow = A_pow * sys.A
        end
    end

    format === :matrices && return vcat(Matrix{T}[Gamma0], Gammas)

    moments = T[]
    for i in 1:k, j in i:k
        push!(moments, Gamma0[i, j])
    end
    for lag in lag_vec
        if lag > 0
            Gh = Gammas[lag]
            for i in 1:k
                push!(moments, Gh[i, i])
            end
        end
    end
    return moments
end

function Base.show(io::IO, sys::ABCDSystem{T}) where {T}
    print(io, "ABCDSystem{$T}(n_states=$(nstates(sys)), n_obs=$(nobservables(sys)), ",
          "n_shocks=$(nshocks(sys)), square=$(is_square(sys)), stable=$(is_stable(sys)))")
end

"""
    report([io::IO], sys::ABCDSystem)

Print a structured summary table of an `ABCDSystem`.
"""
function report(io::IO, sys::ABCDSystem{T}) where {T}
    n = nstates(sys)
    k = nobservables(sys)
    m = nshocks(sys)
    max_eig = n == 0 ? zero(T) : maximum(abs, eigvals(sys.A))

    spec_data = Any[
        "States (n)"          n;
        "Observables (k)"     k;
        "Shocks (m)"          m;
        "Square (k == m)"     is_square(sys) ? "Yes" : "No";
        "Stable"              is_stable(sys) ? "Yes" : "No";
        "Max |eigenvalue(A)|" _fmt(max_eig; digits=6);
        "State labels"        isempty(sys.state_names) ? "—" : join(sys.state_names, ", ");
        "Observable labels"   isempty(sys.obs_names) ? "—" : join(sys.obs_names, ", ");
        "Shock labels"        isempty(sys.shock_names) ? "—" : join(sys.shock_names, ", ");
    ]
    _pretty_table(io, spec_data;
        title = "ABCD State-Space System",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    return nothing
end

report(sys::ABCDSystem) = report(stdout, sys)

# =============================================================================
# Orthogonal Kalman Staircase Minimal Realization
# =============================================================================

"""
    _controllable_subspace(A, B; tol=1e-10) -> Matrix{T}

Compute an orthonormal basis `U` (`n × r`) for the controllable subspace of `(A, B)` via
rank-revealing orthogonal staircase expansion.
"""
function _controllable_subspace(A::Matrix{T}, B::Matrix{T}; tol::Real=1e-10) where {T<:AbstractFloat}
    n = size(A, 1)
    (n == 0 || size(B, 2) == 0) && return zeros(T, n, 0)

    scale = max(opnorm(B), opnorm(A), one(T))
    thresh = T(tol) * scale

    F0 = svd(B)
    r0 = count(>(thresh), F0.S)
    r0 == 0 && return zeros(T, n, 0)

    U = Matrix{T}(F0.U[:, 1:r0])
    U_new = U

    for _ in 1:n
        size(U, 2) >= n && break
        W = A * U_new
        # Double classical Gram-Schmidt against existing basis avoids orthogonality loss
        W .-= U * (U' * W)
        W .-= U * (U' * W)

        Fw = svd(W)
        rw = min(count(>(thresh), Fw.S), n - size(U, 2))
        rw == 0 && break

        U_next = Matrix{T}(Fw.U[:, 1:rw])
        U_next .-= U * (U' * U_next)
        F_re = svd(U_next)
        rw_re = min(count(>(T(0.5)), F_re.S), n - size(U, 2))
        rw_re == 0 && break

        U_new = Matrix{T}(F_re.U[:, 1:rw_re])
        U = hcat(U, U_new)
    end

    return U
end

"""
    _abcd_minimal_realization(A, B, C, state_names; tol=1e-10)
        -> (A_m, B_m, C_m, names_m, U_tot)

Reduce `(A, B, C)` to a minimal (controllable and observable) realization by orthogonal
controllability and observability staircase deflation.
"""
function _abcd_minimal_realization(A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                   C::AbstractMatrix{T}, state_names::AbstractVector;
                                   tol::Real=1e-10) where {T<:AbstractFloat}
    Am0 = Matrix{T}(A)
    Bm0 = Matrix{T}(B)
    Cm0 = Matrix{T}(C)
    n = size(Am0, 1)
    m = size(Bm0, 2)
    k = size(Cm0, 1)

    U_c = _controllable_subspace(Am0, Bm0; tol=tol)
    if size(U_c, 2) == 0
        return zeros(T, 0, 0), zeros(T, 0, m), zeros(T, k, 0), String[], zeros(T, n, 0)
    end

    A_c = Matrix{T}(U_c' * Am0 * U_c)
    B_c = Matrix{T}(U_c' * Bm0)
    C_c = Matrix{T}(Cm0 * U_c)

    U_o = _controllable_subspace(Matrix{T}(A_c'), Matrix{T}(C_c'); tol=tol)
    if size(U_o, 2) == 0
        return zeros(T, 0, 0), zeros(T, 0, m), zeros(T, k, 0), String[], zeros(T, n, 0)
    end

    A_m = Matrix{T}(U_o' * A_c * U_o)
    B_m = Matrix{T}(U_o' * B_c)
    C_m = Matrix{T}(C_c * U_o)
    U_tot = Matrix{T}(U_c * U_o)
    names_m = ["x_$i" for i in 1:size(A_m, 1)]

    return A_m, B_m, C_m, names_m, U_tot
end

# =============================================================================
# Shared internal helpers for observables & measurement-error augmentation
# =============================================================================

function _resolve_abcd_observables(all_names::Vector{Symbol},
                                   default_names::Vector{Symbol},
                                   observables::AbstractVector)
    if !isempty(observables) && all(x -> x isa Integer, observables)
        n_all = length(all_names)
        obs_idx = Int[]
        for raw_idx in observables
            idx = Int(raw_idx)
            (1 <= idx <= n_all) || throw(BoundsError(1:n_all, idx))
            push!(obs_idx, idx)
        end
        return obs_idx, [string(all_names[i]) for i in obs_idx]
    elseif isempty(observables) || all(x -> x isa Symbol || x isa AbstractString, observables)
        obs_syms = isempty(observables) ? default_names : Symbol.(observables)
        obs_idx = Int[]
        for s in obs_syms
            idx = findfirst(==(s), all_names)
            idx === nothing && throw(ArgumentError(
                "Observable :$s not found in available variables: $all_names"))
            push!(obs_idx, idx)
        end
        return obs_idx, [string(s) for s in obs_syms]
    else
        throw(ArgumentError(
            "observables must be Symbols, Strings, or Integer indices; got $(typeof(observables))"))
    end
end

function _augment_abcd_measurement_error(B::Matrix{T}, D::Matrix{T},
                                         shock_names::Vector{String},
                                         obs_names::Vector{String},
                                         measurement_error) where {T<:AbstractFloat}
    n = size(B, 1)
    k = size(D, 1)

    if measurement_error isa AbstractVector
        length(measurement_error) == k || throw(ArgumentError(
            "measurement_error length ($(length(measurement_error))) must match observables ($k)"))
        me = Vector{T}(measurement_error)
        all(>=(zero(T)), me) || throw(ArgumentError(
            "measurement_error standard deviations must be non-negative"))
        D_me = Matrix{T}(Diagonal(me))
    elseif measurement_error isa AbstractMatrix
        size(measurement_error) == (k, k) || throw(ArgumentError(
            "measurement_error matrix must be $k×$k, got $(size(measurement_error))"))
        if k == 0
            D_me = zeros(T, 0, 0)
        else
            M_me = Matrix{T}(measurement_error)
            sym_tol = sqrt(eps(T)) * max(norm(M_me), one(T))
            norm(M_me - M_me') <= sym_tol || throw(ArgumentError(
                "measurement_error covariance matrix must be symmetric"))
            M_sym = Matrix{T}((M_me + M_me') / 2)
            D_me = Matrix{T}(safe_cholesky(M_sym))
        end
    else
        throw(ArgumentError(
            "measurement_error must be a vector of standard deviations, a covariance matrix, " *
            "or nothing; got $(typeof(measurement_error))"))
    end

    B_aug = hcat(B, zeros(T, n, k))
    D_aug = hcat(D, D_me)
    me_names = ["me_" * nm for nm in obs_names]
    return B_aug, D_aug, vcat(shock_names, me_names)
end

function _validate_abcd_reduction(reduction::Symbol, valid::Tuple{Vararg{Symbol}})
    reduction in valid || throw(ArgumentError(
        "reduction must be one of $valid; got :$reduction"))
end

function _abcd_shock_names(exog::Vector{Symbol}, m::Int)
    length(exog) == m && return [string(s) for s in exog]
    return ["w_$j" for j in 1:m]
end

"""Optionally apply minimal staircase deflation and measurement-error augmentation, then build `ABCDSystem{T}`."""
function _finalize_abcd(A::Matrix{T}, B::Matrix{T}, C::Matrix{T}, D::Matrix{T},
                        c_state::Vector{T}, d_obs::Vector{T},
                        snames::Vector{String}, obs_names::Vector{String},
                        shock_names::Vector{String};
                        reduction::Symbol,
                        measurement_error=nothing,
                        tol::Real=1e-10) where {T<:AbstractFloat}
    if reduction === :minimal
        A, B, C, snames, U_tot = _abcd_minimal_realization(A, B, C, snames; tol=tol)
        c_state = Vector{T}(U_tot' * c_state)
    end
    if measurement_error !== nothing
        B, D, shock_names = _augment_abcd_measurement_error(B, D, shock_names, obs_names, measurement_error)
    end
    return ABCDSystem{T}(A, B, C, D, c_state, d_obs, snames, obs_names, shock_names)
end

# =============================================================================
# abcd_representation dispatches
# =============================================================================

"""
    abcd_representation(sol, observables=Symbol[]; kwargs...) -> ABCDSystem

Extract a canonical [`ABCDSystem`](@ref) from a solved model (`DSGESolution`,
`PerturbationSolution`, `HADSGESolution`, or `StateSpaceModel`).

# Keyword Arguments
- `observables`: subset of observable variables (`Symbol`, `String`, or integer indices;
  empty selects all model observables)
- `measurement_error=nothing`: vector of measurement-error standard deviations or `k × k`
  covariance matrix (excluded for `StateSpaceModel`, whose `H` already encodes observation noise)
- `reduction::Symbol`: `:predetermined` (default for DSGE/Perturbation/HA-DSGE) drops zero-column
  jump variables; `:minimal` applies orthogonal controllability/observability Kalman staircase
  deflation; `:none` (default for `StateSpaceModel`) keeps all state coordinates
- `tol::Real=1e-10`: tolerance for predetermined-column detection and staircase SVD rank cutoff
"""
function abcd_representation(sol::DSGESolution{T},
                             observables::AbstractVector=Symbol[];
                             measurement_error=nothing,
                             reduction::Symbol=:predetermined,
                             tol::Real=1e-10) where {T<:AbstractFloat}
    _validate_abcd_reduction(reduction, (:predetermined, :minimal, :none))

    n_endog = sol.spec.n_endog
    default_obs = sol.spec.augmented ? sol.spec.original_endog : sol.spec.endog
    obs_idx, obs_names = _resolve_abcd_observables(sol.spec.endog, default_obs, observables)
    k = length(obs_idx)

    S_obs = zeros(T, k, n_endog)
    for (i, j) in enumerate(obs_idx)
        S_obs[i, j] = one(T)
    end

    idx = if reduction === :none
        collect(1:n_endog)
    else
        findall(j -> norm(@view sol.G1[:, j]) > T(tol), 1:n_endog)
    end

    A = Matrix{T}(sol.G1[idx, idx])
    B = Matrix{T}(sol.impact[idx, :])
    C = Matrix{T}(S_obs * sol.G1[:, idx])
    D = Matrix{T}(S_obs * sol.impact)

    c_full = isempty(sol.C_sol) ? zeros(T, n_endog) : Vector{T}(sol.C_sol)
    ss_full = isempty(sol.spec.steady_state) ? zeros(T, n_endog) : Vector{T}(sol.spec.steady_state)
    c_state = Vector{T}(c_full[idx])
    d_obs = Vector{T}(S_obs * ss_full + S_obs * c_full)

    snames = Vector{String}(sol.spec.varnames[idx])
    shock_names = _abcd_shock_names(sol.spec.exog, size(B, 2))

    return _finalize_abcd(A, B, C, D, c_state, d_obs, snames, obs_names, shock_names;
                          reduction=reduction, measurement_error=measurement_error, tol=tol)
end

function abcd_representation(sol::PerturbationSolution{T},
                             observables::AbstractVector=Symbol[];
                             measurement_error=nothing,
                             reduction::Symbol=:predetermined,
                             tol::Real=1e-10) where {T<:AbstractFloat}
    _validate_abcd_reduction(reduction, (:predetermined, :minimal, :none))

    pss = pruned_state_space(sol)
    n_endog = pss.n
    nx = pss.nx
    n_eps = pss.n_eps

    default_obs = sol.spec.augmented ? sol.spec.original_endog : sol.spec.endog
    obs_idx, obs_names = _resolve_abcd_observables(sol.spec.endog, default_obs, observables)

    C_all = zeros(T, n_endog, nx)
    D_all = zeros(T, n_endog, n_eps)
    if nx > 0
        C_all[sol.state_indices, :] = pss.hx_state
        C_all[sol.control_indices, :] = pss.gx_state
        D_all[sol.state_indices, :] = pss.eta_x
    end
    D_all[sol.control_indices, :] = pss.eta_y

    if reduction === :none
        A = zeros(T, n_endog, n_endog)
        B = zeros(T, n_endog, n_eps)
        if nx > 0
            A[sol.state_indices, sol.state_indices] = pss.hx_state
            A[sol.control_indices, sol.state_indices] = pss.gx_state
            B[sol.state_indices, :] = pss.eta_x
        end
        B[sol.control_indices, :] = pss.eta_y
        C = Matrix{T}(A[obs_idx, :])
        D = Matrix{T}(D_all[obs_idx, :])
        snames = Vector{String}(sol.spec.varnames)
    else
        A = copy(pss.hx_state)
        B = copy(pss.eta_x)
        C = Matrix{T}(C_all[obs_idx, :])
        D = Matrix{T}(D_all[obs_idx, :])
        snames = Vector{String}(sol.spec.varnames[sol.state_indices])
    end

    c_state = zeros(T, size(A, 1))
    ss_full = isempty(sol.steady_state) ?
        (isempty(sol.spec.steady_state) ? zeros(T, n_endog) : Vector{T}(sol.spec.steady_state)) :
        Vector{T}(sol.steady_state)
    d_obs = Vector{T}(ss_full[obs_idx])
    shock_names = _abcd_shock_names(sol.spec.exog, n_eps)

    return _finalize_abcd(A, B, C, D, c_state, d_obs, snames, obs_names, shock_names;
                          reduction=reduction, measurement_error=measurement_error, tol=tol)
end

function abcd_representation(sol::HADSGESolution{T},
                             observables::AbstractVector=Symbol[];
                             measurement_error=nothing,
                             reduction::Symbol=:predetermined,
                             tol::Real=1e-10) where {T<:AbstractFloat}
    _validate_abcd_reduction(reduction, (:predetermined, :minimal, :none))

    lsol = sol.linear_solution
    G1 = lsol.G1
    impact = lsol.impact
    n_sys = size(G1, 1)

    agg_names = _ha_obs_names(sol)
    agg_syms = Symbol.(agg_names)

    use_reduced_endog = false
    if isempty(observables)
        obs_idx = collect(1:length(agg_syms))
        obs_names = copy(agg_names)
        C_raw = Matrix{T}(sol.C_obs)
        D_raw = Matrix{T}(sol.D_obs)
    elseif all(x -> x isa Integer, observables)
        obs_idx, obs_names = _resolve_abcd_observables(agg_syms, agg_syms, observables)
        C_raw = Matrix{T}(sol.C_obs[obs_idx, :])
        D_raw = Matrix{T}(sol.D_obs[obs_idx, :])
    elseif all(x -> x isa Symbol || x isa AbstractString, observables)
        req_syms = Symbol.(observables)
        if all(s -> s in agg_syms, req_syms)
            obs_idx, obs_names = _resolve_abcd_observables(agg_syms, agg_syms, req_syms)
            C_raw = Matrix{T}(sol.C_obs[obs_idx, :])
            D_raw = Matrix{T}(sol.D_obs[obs_idx, :])
        elseif all(s -> s in lsol.spec.endog, req_syms)
            use_reduced_endog = true
            obs_idx, obs_names = _resolve_abcd_observables(lsol.spec.endog, lsol.spec.endog, req_syms)
            C_raw = Matrix{T}(G1[obs_idx, :])
            D_raw = Matrix{T}(impact[obs_idx, :])
        else
            throw(ArgumentError(
                "Observables $req_syms not found in HA outputs $agg_syms or reduced states $(lsol.spec.endog)"))
        end
    else
        throw(ArgumentError(
            "observables must be Symbols, Strings, or Integer indices; got $(typeof(observables))"))
    end

    c_full = isempty(lsol.C_sol) ? zeros(T, n_sys) : Vector{T}(lsol.C_sol)
    if use_reduced_endog
        ss_full = isempty(lsol.spec.steady_state) ? zeros(T, n_sys) : Vector{T}(lsol.spec.steady_state)
        d_obs = Vector{T}(ss_full[obs_idx] .+ c_full[obs_idx])
    else
        d_obs = zeros(T, length(obs_names))
        for (i, nm) in enumerate(obs_names)
            sym = Symbol(nm)
            if haskey(sol.steady_state.aggregates, sym)
                d_obs[i] = T(sol.steady_state.aggregates[sym])
            elseif haskey(sol.steady_state.prices, sym)
                d_obs[i] = T(sol.steady_state.prices[sym])
            end
        end
    end

    # Keep states that enter either G1 or C_raw so lag-1 feedthrough via C_obs is preserved
    idx = if reduction === :none
        collect(1:n_sys)
    else
        findall(j -> norm(@view G1[:, j]) > T(tol) || norm(@view C_raw[:, j]) > T(tol), 1:n_sys)
    end

    A = Matrix{T}(G1[idx, idx])
    B = Matrix{T}(impact[idx, :])
    C = Matrix{T}(C_raw[:, idx])
    D = Matrix{T}(D_raw)

    c_state = Vector{T}(c_full[idx])
    snames = Vector{String}(lsol.spec.varnames[idx])
    shock_names = _abcd_shock_names(lsol.spec.exog, size(B, 2))

    return _finalize_abcd(A, B, C, D, c_state, d_obs, snames, obs_names, shock_names;
                          reduction=reduction, measurement_error=measurement_error, tol=tol)
end

function abcd_representation(ss::StateSpaceModel{T},
                             observables::AbstractVector=Symbol[];
                             reduction::Symbol=:none,
                             tol::Real=1e-10) where {T<:AbstractFloat}
    _validate_abcd_reduction(reduction, (:none, :predetermined, :minimal))

    all_obs_syms = [Symbol("obs_$i") for i in 1:ss.n_obs]
    obs_idx, obs_names = _resolve_abcd_observables(all_obs_syms, all_obs_syms, observables)
    k = length(obs_idx)
    n = ss.n_state

    Z_sub = Matrix{T}(ss.Z[obs_idx, :])
    d_obs = Vector{T}(ss.d[obs_idx])
    H_sub = Matrix{T}(ss.H[obs_idx, obs_idx])

    RQR = Matrix{T}(ss.R * ss.Q * ss.R')
    G_eff = ss.G === nothing ? zeros(T, n, k) : Matrix{T}((ss.R * ss.G)[:, obs_idx])

    M_raw = [RQR G_eff; G_eff' H_sub]
    M_sym = Matrix{T}((M_raw + M_raw') / 2)

    if isempty(M_sym)
        B = zeros(T, n, 0)
        D = zeros(T, k, 0)
    else
        F = eigen(Hermitian(M_sym))
        max_ev = maximum(abs, F.values; init=zero(T))
        min_ev = minimum(F.values)
        neg_tol = max(T(tol), eps(T) * T(100)) * max(max_ev, one(T))
        min_ev < -neg_tol && throw(ArgumentError(
            "Joint innovation covariance [R*Q*R' R*G; (R*G)' H] is not positive semidefinite " *
            "(min eigenvalue = $min_ev)"))

        rel_cut = max(T(tol), eps(T) * T(size(M_sym, 1)) * T(10))
        eig_tol = max_ev > zero(T) ? rel_cut * max_ev : zero(T)
        pos = findall(>(eig_tol), F.values)

        if isempty(pos)
            B = zeros(T, n, 0)
            D = zeros(T, k, 0)
        else
            BD = F.vectors[:, pos] * Diagonal(sqrt.(F.values[pos]))
            B = Matrix{T}(BD[1:n, :])
            D = Matrix{T}(BD[n+1:end, :])
        end
    end

    idx = if reduction === :predetermined
        findall(j -> norm(@view ss.Tt[:, j]) > T(tol) || norm(@view Z_sub[:, j]) > T(tol), 1:n)
    else
        collect(1:n)
    end

    A = Matrix{T}(ss.Tt[idx, idx])
    B = Matrix{T}(B[idx, :])
    C = Matrix{T}(Z_sub[:, idx])
    c_state = Vector{T}(ss.c[idx])
    snames = ["x_$i" for i in idx]
    shock_names = ["w_$j" for j in 1:size(B, 2)]

    return _finalize_abcd(A, B, C, D, c_state, d_obs, snames, obs_names, shock_names;
                          reduction=reduction, tol=tol)
end
