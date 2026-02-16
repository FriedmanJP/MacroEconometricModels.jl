# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
DFM-based nowcasting via EM algorithm (Bańbura & Modugno 2014).

Handles mixed-frequency data with arbitrary missing patterns. Quarterly
variables use the Mariano-Murasawa [1 2 3 2 1] temporal aggregation.
"""

# =============================================================================
# Public API
# =============================================================================

"""
    nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1, blocks=nothing,
                max_iter=100, thresh=1e-4) -> NowcastDFM{T}

Estimate a dynamic factor model on mixed-frequency data with missing values.

The first `nM` columns of `Y` are monthly variables; the next `nQ` columns
are quarterly variables (observed every 3rd month, NaN otherwise).

# Arguments
- `Y::AbstractMatrix` — T_obs × N data matrix (NaN for missing)
- `nM::Int` — number of monthly variables
- `nQ::Int` — number of quarterly variables

# Keyword Arguments
- `r::Int=2` — number of factors
- `p::Int=1` — VAR lags in factor dynamics
- `idio::Symbol=:ar1` — idiosyncratic component (`:ar1` or `:iid`)
- `blocks::Union{Matrix{Int},Nothing}=nothing` — block structure (N × n_blocks)
- `max_iter::Int=100` — maximum EM iterations
- `thresh::Real=1e-4` — convergence threshold (relative log-likelihood change)

# Returns
`NowcastDFM{T}` with smoothed data, factors, and state-space parameters.

# References
- Bańbura, M. & Modugno, M. (2014). Maximum Likelihood Estimation of Factor
  Models on Datasets with Arbitrary Pattern of Missing Data.
"""
function nowcast_dfm(Y::AbstractMatrix, nM::Int, nQ::Int;
                     r::Int=2, p::Int=1, idio::Symbol=:ar1,
                     blocks::Union{Matrix{Int},Nothing}=nothing,
                     max_iter::Int=100, thresh::Real=1e-4)
    T_obs, N = size(Y)
    N == nM + nQ || throw(ArgumentError("nM ($nM) + nQ ($nQ) must equal number of columns ($N)"))
    r >= 1 || throw(ArgumentError("r must be >= 1, got $r"))
    p >= 1 || throw(ArgumentError("p must be >= 1, got $p"))
    idio ∈ (:ar1, :iid) || throw(ArgumentError("idio must be :ar1 or :iid, got $idio"))

    Tf = eltype(Y) <: AbstractFloat ? eltype(Y) : Float64
    Ymat = Matrix{Tf}(Y)

    # Default block structure: single block for all variables
    if blocks === nothing
        blocks = ones(Int, N, 1)
    end
    n_blocks = size(blocks, 2)
    size(blocks, 1) == N || throw(ArgumentError("blocks must have $N rows, got $(size(blocks, 1))"))

    # Standardize data (compute on non-NaN entries)
    Mx = zeros(Tf, N)
    Wx = ones(Tf, N)
    for j in 1:N
        valid = filter(!isnan, Ymat[:, j])
        if !isempty(valid)
            Mx[j] = mean(valid)
            s = std(valid)
            Wx[j] = s > Tf(1e-10) ? s : one(Tf)
        end
    end
    xNaN = (Ymat .- Mx') ./ Wx'

    # Indices for idiosyncratic components
    i_idio_M = idio == :ar1 ? collect(1:nM) : Int[]
    i_idio_Q = nM < N ? collect((nM + 1):N) : Int[]
    i_idio = vcat(i_idio_M, i_idio_Q)

    # Initialize state-space matrices
    A, C, Q, R, Z_0, V_0 = _dfm_init_cond(xNaN, r, p, blocks, nQ, i_idio, idio)

    # EM iterations
    prev_loglik = Tf(-Inf)
    loglik = Tf(-Inf)
    n_iter = 0

    for iter in 1:max_iter
        # E-step: Kalman smoother
        # M-step: update parameters
        A, C, Q, R, Z_0, V_0, loglik_new = _dfm_em_step(
            xNaN, A, C, Q, R, Z_0, V_0, r, p, nQ, i_idio, blocks, idio)

        # Check convergence
        if iter > 1 && _em_converged(loglik_new, prev_loglik, Tf(thresh))
            loglik = loglik_new
            n_iter = iter
            break
        end
        prev_loglik = loglik_new
        loglik = loglik_new
        n_iter = iter
    end

    # Final smoother run for smoothed states and data
    state_dim = size(A, 1)
    y_t = xNaN'  # N × T_obs
    x_smooth, _, _, _ = _kalman_smoother_missing(y_t, A, C, Q, R, Z_0, V_0)

    # Extract factors (first r*p states are factors)
    F = x_smooth[1:min(r * p, state_dim), :]'  # T_obs × r*p

    # Reconstruct smoothed data: X_sm = C * x_smooth (in standardized space)
    X_sm_std = (C * x_smooth)'  # T_obs × N

    # Unstandardize
    X_sm = X_sm_std .* Wx' .+ Mx'

    # For observed values, keep originals
    for t in 1:T_obs, j in 1:N
        if !isnan(Ymat[t, j])
            X_sm[t, j] = Ymat[t, j]
        end
    end

    NowcastDFM{Tf}(X_sm, F, C, A, Q, R, Mx, Wx, Z_0, V_0,
                   r, p, blocks, loglik, n_iter, nM, nQ, idio, Ymat)
end

# =============================================================================
# EM Convergence Check
# =============================================================================

"""Check EM convergence via relative log-likelihood change."""
function _em_converged(loglik::T, prev_loglik::T, thresh::T) where {T}
    prev_loglik == T(-Inf) && return false
    denom = abs(prev_loglik)
    denom < T(1e-10) && (denom = one(T))
    return abs((loglik - prev_loglik) / denom) < thresh
end

# =============================================================================
# Initial Conditions via PCA
# =============================================================================

"""
    _dfm_init_cond(xNaN, r, p, blocks, nQ, i_idio, idio)

Initialize state-space matrices via PCA on balanced panel.

Uses spline interpolation to fill NaN for initial PCA, then sets up
monthly loadings, quarterly loadings with [1 2 3 2 1] constraint,
and idiosyncratic AR(1) initialization.
"""
function _dfm_init_cond(xNaN::Matrix{T}, r::Int, p::Int, blocks::Matrix{Int},
                        nQ::Int, i_idio::Vector{Int}, idio::Symbol) where {T<:AbstractFloat}
    T_obs, N = size(xNaN)
    nM = N - nQ
    n_blocks = size(blocks, 2)

    # Fill NaN for initial PCA via column mean interpolation
    xBal = copy(xNaN)
    for j in 1:N
        col = xBal[:, j]
        valid = filter(!isnan, col)
        m = isempty(valid) ? zero(T) : mean(valid)
        for i in 1:T_obs
            isnan(col[i]) && (xBal[i, j] = m)
        end
    end

    # PCA for initial factors
    n_factors_total = r * n_blocks
    Xc = xBal .- mean(xBal, dims=1)
    Cx = Xc' * Xc / T_obs
    eig = eigen(Symmetric(Cx), sortby = x -> -x)
    n_eig = min(n_factors_total, length(eig.values))
    F0 = Xc * eig.vectors[:, 1:n_eig]  # T_obs × n_eig
    Lambda0 = eig.vectors[:, 1:n_eig]  # N × n_eig

    # Build state dimension
    # State vector: [block_factors (r*n_blocks*p), idio_M (if ar1), idio_Q (5*nQ)]
    n_idio_M = idio == :ar1 ? nM : 0
    n_idio_Q = 5 * nQ  # 5 states for quarterly temporal aggregation
    state_dim = r * n_blocks * p + n_idio_M + n_idio_Q

    # State transition matrix
    A = zeros(T, state_dim, state_dim)

    # Factor VAR: estimate from initial factors
    if n_eig >= 1
        n_f = min(n_eig, r * n_blocks)
        if T_obs > p + 1
            # Simple VAR(p) on initial factors
            Y_f = F0[(p + 1):end, 1:n_f]
            X_f = ones(T, size(Y_f, 1), 1)
            for lag in 1:p
                X_f = hcat(X_f, F0[(p + 1 - lag):(end - lag), 1:n_f])
            end
            B_f = (X_f' * X_f) \ (X_f' * Y_f)
            for lag in 1:p
                row_start = (lag - 1) * n_f + 1
                row_end = lag * n_f
                A[1:n_f, row_start:row_end] = B_f[(1 + (lag - 1) * n_f + 1):(1 + lag * n_f), :]'
            end
            # Companion form for lagged factors
            if p > 1
                A[(n_f + 1):(n_f * p), 1:(n_f * (p - 1))] = Matrix{T}(I, n_f * (p - 1), n_f * (p - 1))
            end
        end
    end

    n_f = min(n_eig, r * n_blocks)
    factor_block_end = n_f * p

    # Idiosyncratic AR(1) for monthly variables
    if idio == :ar1
        for i in 1:nM
            idx = factor_block_end + i
            # OLS AR(1) on residuals
            n_use = min(n_f, size(F0, 2), size(Lambda0, 2))
            resid_i = xBal[:, i] - F0[:, 1:n_use] * Lambda0[i, 1:n_use]
            if length(resid_i) > 2
                rho = dot(resid_i[2:end], resid_i[1:end-1]) / max(dot(resid_i[1:end-1], resid_i[1:end-1]), T(1e-10))
                A[idx, idx] = clamp(rho, T(-0.99), T(0.99))
            end
        end
    end

    # Quarterly idiosyncratic: 5-state companion for [1 2 3 2 1] aggregation
    for q in 1:nQ
        base = factor_block_end + n_idio_M + (q - 1) * 5
        # Shift register for temporal aggregation
        if base + 4 <= state_dim
            for s in 2:5
                A[base + s, base + s - 1] = one(T)
            end
        end
    end

    # Observation matrix
    C = zeros(T, N, state_dim)

    # Monthly loadings: direct mapping to factors
    for i in 1:nM
        for b in 1:n_blocks
            if blocks[i, b] == 1
                col_start = (b - 1) * r + 1
                col_end = b * r
                if col_end <= n_f
                    C[i, col_start:col_end] = Lambda0[i, col_start:col_end]
                elseif col_start <= n_f
                    C[i, col_start:n_f] = Lambda0[i, col_start:n_f]
                end
            end
        end
        # Idiosyncratic loading
        if idio == :ar1
            C[i, factor_block_end + i] = one(T)
        end
    end

    # Quarterly loadings with [1 2 3 2 1] temporal aggregation
    for q in 1:nQ
        i = nM + q
        base = factor_block_end + n_idio_M + (q - 1) * 5

        # Factor loadings via temporal aggregation
        for b in 1:n_blocks
            if blocks[i, b] == 1
                col_start = (b - 1) * r + 1
                col_end = min(b * r, n_f)
                if col_start <= n_f
                    for c in col_start:col_end
                        load = Lambda0[i, c]
                        C[i, c] = T(3) * load  # Peak weight at current month
                    end
                end
            end
        end

        # Quarterly idiosyncratic with [1 2 3 2 1] weights
        if base + 4 <= state_dim
            C[i, base + 1] = one(T)
            C[i, base + 2] = T(2)
            C[i, base + 3] = T(3)
            C[i, base + 4] = T(2)
            C[i, base + 5] = one(T)
        end
    end

    # State noise covariance
    Q_mat = zeros(T, state_dim, state_dim)
    for i in 1:min(n_f, state_dim)
        Q_mat[i, i] = one(T)
    end
    # Idiosyncratic noise
    if idio == :ar1
        for i in 1:nM
            idx = factor_block_end + i
            Q_mat[idx, idx] = one(T)
        end
    end
    for q in 1:nQ
        base = factor_block_end + n_idio_M + (q - 1) * 5
        if base + 1 <= state_dim
            Q_mat[base + 1, base + 1] = one(T)
        end
    end

    # Observation noise covariance (diagonal)
    R_mat = Matrix{T}(T(0.5) * I(N))

    # Initial state
    Z_0 = zeros(T, state_dim)
    V_0 = Matrix{T}(I(state_dim))

    return A, C, Q_mat, R_mat, Z_0, V_0
end

# =============================================================================
# EM Step
# =============================================================================

"""
    _dfm_em_step(xNaN, A, C, Q, R, Z_0, V_0, r, p, nQ, i_idio, blocks, idio)

One EM iteration: E-step (Kalman smoother) + M-step (parameter updates).

Returns updated (A, C, Q, R, Z_0, V_0, loglik).
"""
function _dfm_em_step(xNaN::Matrix{T}, A::Matrix{T}, C::Matrix{T},
                      Q::Matrix{T}, R::Matrix{T}, Z_0::Vector{T},
                      V_0::Matrix{T}, r::Int, p::Int, nQ::Int,
                      i_idio::Vector{Int}, blocks::Matrix{Int},
                      idio::Symbol) where {T<:AbstractFloat}
    T_obs, N = size(xNaN)
    nM = N - nQ
    state_dim = size(A, 1)
    n_blocks = size(blocks, 2)
    n_f = min(r * n_blocks, state_dim)

    # E-step: Kalman smoother
    y_t = xNaN'  # N × T_obs
    x_smooth, P_smooth, PP_smooth, loglik = _kalman_smoother_missing(
        y_t, A, C, Q, R, Z_0, V_0)

    # Sufficient statistics
    # EZZ = sum_t E[z_t z_t']  (state × state)
    # EZZ_lag = sum_t E[z_t z_{t-1}']
    EZZ = zeros(T, state_dim, state_dim)
    EZZ_lag = zeros(T, state_dim, state_dim)
    EZZ_prev = zeros(T, state_dim, state_dim)

    for t in 1:T_obs
        EZZ += @view(P_smooth[:, :, t]) + @view(x_smooth[:, t]) * @view(x_smooth[:, t])'
        if t > 1
            EZZ_lag += @view(PP_smooth[:, :, t])
            EZZ_prev += @view(P_smooth[:, :, t - 1]) + @view(x_smooth[:, t - 1]) * @view(x_smooth[:, t - 1])'
        end
    end

    # M-step: Update transition matrix A for factor block
    if n_f >= 1 && T_obs > 1
        factor_end = n_f * p
        if factor_end <= state_dim
            # Update factor VAR coefficients
            fi = 1:n_f
            fi_all = 1:factor_end
            EZZ_lag_f = EZZ_lag[fi, fi_all]
            EZZ_prev_f = EZZ_prev[fi_all, fi_all]
            EZZ_prev_reg = EZZ_prev_f + T(1e-8) * I(length(fi_all))
            A_new_f = EZZ_lag_f / EZZ_prev_reg
            A[fi, fi_all] = A_new_f
        end
    end

    # M-step: Update idiosyncratic AR(1) coefficients
    n_idio_M = idio == :ar1 ? nM : 0
    factor_block_end = n_f * p
    if idio == :ar1
        for i in 1:nM
            idx = factor_block_end + i
            if idx <= state_dim
                num = EZZ_lag[idx, idx]
                den = EZZ_prev[idx, idx] + T(1e-10)
                A[idx, idx] = clamp(num / den, T(-0.99), T(0.99))
            end
        end
    end

    # M-step: Update factor noise covariance Q
    if n_f >= 1
        fi = 1:n_f
        Q_new = (EZZ[fi, fi] - A[fi, 1:(n_f * p)] * EZZ_lag[fi, 1:(n_f * p)]') / T(T_obs)
        Q_new = (Q_new + Q_new') / T(2)  # Symmetrize
        for i in fi
            Q[i, i] = max(Q_new[i, i], T(1e-6))
        end
        # Off-diagonal
        for i in fi, j in fi
            if i != j
                Q[i, j] = Q_new[i, j]
            end
        end
    end

    # Update idiosyncratic noise in Q
    if idio == :ar1
        for i in 1:nM
            idx = factor_block_end + i
            if idx <= state_dim
                q_idio = (EZZ[idx, idx] - A[idx, idx]^2 * EZZ_prev[idx, idx]) / T(T_obs)
                Q[idx, idx] = max(q_idio, T(1e-6))
            end
        end
    end
    for q_idx in 1:nQ
        base = factor_block_end + n_idio_M + (q_idx - 1) * 5
        idx = base + 1
        if idx <= state_dim
            Q[idx, idx] = max(EZZ[idx, idx] / T(T_obs), T(1e-6))
        end
    end

    # M-step: Update observation loadings C (per-variable OLS)
    for i in 1:N
        # Find observed time points for variable i
        obs_t = findall(!isnan, xNaN[:, i])
        isempty(obs_t) && continue

        if i <= nM
            # Monthly variable: C[i,:] loads on factor block + idio
            # Determine which states this variable loads on
            load_idx = Int[]
            for b in 1:n_blocks
                if blocks[i, b] == 1
                    append!(load_idx, ((b - 1) * r + 1):min(b * r, n_f))
                end
            end
            if idio == :ar1
                push!(load_idx, factor_block_end + i)
            end
            isempty(load_idx) && continue

            # OLS: C[i, load_idx] = (sum x*z') / (sum z*z')
            xz = zeros(T, length(load_idx))
            zz = zeros(T, length(load_idx), length(load_idx))
            for t in obs_t
                z_t = x_smooth[load_idx, t]
                xz += xNaN[t, i] * z_t
                zz += P_smooth[load_idx, load_idx, t] + z_t * z_t'
            end
            zz_reg = zz + T(1e-8) * I(length(load_idx))
            C[i, load_idx] = zz_reg \ xz
        else
            # Quarterly variable: loadings constrained by temporal aggregation
            # Only update factor loadings, keep [1 2 3 2 1] structure fixed
            q_num = i - nM
            load_idx = Int[]
            for b in 1:n_blocks
                if blocks[i, b] == 1
                    append!(load_idx, ((b - 1) * r + 1):min(b * r, n_f))
                end
            end
            isempty(load_idx) && continue

            xz = zeros(T, length(load_idx))
            zz = zeros(T, length(load_idx), length(load_idx))
            for t in obs_t
                z_t = x_smooth[load_idx, t]
                xz += xNaN[t, i] * z_t
                zz += P_smooth[load_idx, load_idx, t] + z_t * z_t'
            end
            zz_reg = zz + T(1e-8) * I(length(load_idx))
            c_new = zz_reg \ xz
            C[i, load_idx] = c_new
        end
    end

    # M-step: Update observation noise R (diagonal)
    for i in 1:N
        obs_t = findall(!isnan, xNaN[:, i])
        isempty(obs_t) && continue

        sse = zero(T)
        for t in obs_t
            resid = xNaN[t, i] - dot(C[i, :], x_smooth[:, t])
            sse += resid^2 + dot(C[i, :], P_smooth[:, :, t] * C[i, :])
        end
        R[i, i] = max(sse / T(length(obs_t)), T(1e-4))
    end

    # Update initial conditions
    Z_0_new = x_smooth[:, 1]
    V_0_new = P_smooth[:, :, 1]
    V_0_new = (V_0_new + V_0_new') / T(2)
    for i in 1:state_dim
        V_0_new[i, i] = max(V_0_new[i, i], T(1e-6))
    end

    return A, C, Q, R, Z_0_new, V_0_new, loglik
end
