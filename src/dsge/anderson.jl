# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# MacroEconometricModels.jl — Anderson Acceleration for Fixed-Point Iterations
#
# References:
#   Walker & Ni (2011), Anderson Acceleration for Fixed-Point Iterations,
#     SIAM J. Numer. Anal. 49(4): 1715–1735

"""
    _anderson_step(history, residuals, m) -> Vector{T}

Compute an Anderson-accelerated iterate from the last `m` iterates.

Given iterates `x_k` and residuals `r_k = g(x_k) - x_k`, solve:

    min ‖Σ αᵢ rᵢ‖²  s.t. Σ αᵢ = 1

and return the mixed iterate `x_new = Σ αᵢ (xᵢ + rᵢ)`.

# Arguments
- `history::Vector{Vector{T}}`: previous iterates x_k
- `residuals::Vector{Vector{T}}`: corresponding residuals r_k = g(x_k) - x_k
- `m::Int`: mixing depth (use last m entries)
"""
function _anderson_step(history::Vector{Vector{T}}, residuals::Vector{Vector{T}},
                         m::Int) where {T<:AbstractFloat}
    n_hist = length(history)
    m_eff = min(m, n_hist)

    if m_eff <= 1
        return history[end] .+ residuals[end]
    end

    start_idx = n_hist - m_eff + 1
    R = hcat(residuals[start_idx:end]...)

    r_last = residuals[end]

    if m_eff == 2
        dr = R[:, 2] .- R[:, 1]
        dr_norm_sq = dot(dr, dr)
        if dr_norm_sq < eps(T)
            return history[end] .+ r_last
        end
        gamma1 = dot(dr, r_last) / dr_norm_sq
        alpha = [gamma1, one(T) - gamma1]
    else
        n_col = m_eff - 1
        n_dim = length(r_last)
        DR = zeros(T, n_dim, n_col)
        for j in 1:n_col
            DR[:, j] = R[:, j + 1] .- R[:, j]
        end

        DRtDR = DR' * DR
        DRtr = DR' * r_last

        reg = max(eps(T) * T(1e4), eps(T) * norm(DRtDR))
        for i in 1:n_col
            DRtDR[i, i] += reg
        end

        gamma = DRtDR \ DRtr

        alpha = zeros(T, m_eff)
        alpha[1] = gamma[1]
        for j in 2:n_col
            alpha[j] = gamma[j] - gamma[j - 1]
        end
        alpha[m_eff] = one(T) - gamma[n_col]
    end

    x_new = zeros(T, length(r_last))
    for j in 1:m_eff
        idx = start_idx + j - 1
        x_new .+= alpha[j] .* (history[idx] .+ residuals[idx])
    end

    return x_new
end
