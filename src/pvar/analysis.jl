# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
Structural analysis for Panel VAR — OIRF, GIRF, FEVD, stability.

Reuses `companion_matrix` from `src/core/utils.jl` and `safe_cholesky` for
Cholesky-based orthogonalized impulse responses.
"""

# =============================================================================
# Helper: extract endogenous coefficient block and build companion form
# =============================================================================

"""
    _pvar_companion(model::PVARModel{T}) -> Matrix{T}

Build companion matrix from PVAR coefficient matrices.
Extracts the m × (m*p) endogenous block from Phi, reshapes to match
`companion_matrix(B, n, p)` from core/utils.jl.
"""
function _pvar_companion(model::PVARModel{T}) where {T}
    m_dim = model.m
    p = model.p
    # Extract A_1, ..., A_p from Phi (first m*p columns)
    # Phi is m × K where first m*p columns are lag coefficients
    A_block = model.Phi[:, 1:(m_dim*p)]

    # Build companion matrix directly: F is (m*p) × (m*p)
    np = m_dim * p
    F = zeros(T, np, np)
    for l in 1:p
        cols = ((l-1)*m_dim+1):(l*m_dim)
        F[1:m_dim, cols] .= A_block[:, cols]
    end
    if p > 1
        for i in 1:(p-1)
            F[(i*m_dim+1):((i+1)*m_dim), ((i-1)*m_dim+1):(i*m_dim)] .= I(m_dim)
        end
    end
    F
end

"""
    _pvar_ma_coefficients(model::PVARModel{T}, H::Int) -> Vector{Matrix{T}}

Compute MA(∞) coefficient matrices Φ_0, Φ_1, ..., Φ_H from companion form.
Φ_h = J A^h J' where J = [I_m | 0 ... 0] is m × (m*p) selector.
"""
function _pvar_ma_coefficients(model::PVARModel{T}, H::Int) where {T}
    m_dim = model.m
    p = model.p
    F = _pvar_companion(model)
    np = m_dim * p

    # Selector matrix
    J = zeros(T, m_dim, np)
    J[1:m_dim, 1:m_dim] .= I(m_dim)

    Phi_h = Vector{Matrix{T}}(undef, H + 1)
    Fh = Matrix{T}(I, np, np)
    for h in 0:H
        Phi_h[h+1] = J * Fh * J'
        Fh = Fh * F
    end
    Phi_h
end

# =============================================================================
# OIRF — Orthogonalized Impulse Response
# =============================================================================

"""
    pvar_oirf(model::PVARModel{T}, H::Int) -> Array{T, 3}

Orthogonalized impulse response functions via Cholesky decomposition.

Ψ_h = Φ_h P where P = chol(Σ) is the lower Cholesky factor.

Returns H+1 × m × m array: `oirf[h+1, response, shock]`.

# Arguments
- `model::PVARModel` — estimated PVAR
- `H::Int` — maximum horizon

# Examples
```julia
oirf = pvar_oirf(model, 10)
oirf[1, :, :]  # impact response (h=0)
```
"""
function pvar_oirf(model::PVARModel{T}, H::Int) where {T}
    H < 0 && throw(ArgumentError("Horizon H must be non-negative"))
    m_dim = model.m
    P = safe_cholesky(model.Sigma)

    Phi_h = _pvar_ma_coefficients(model, H)

    oirf = Array{T}(undef, H + 1, m_dim, m_dim)
    for h in 0:H
        oirf[h+1, :, :] = Phi_h[h+1] * P
    end
    oirf
end

# =============================================================================
# GIRF — Generalized Impulse Response (Pesaran & Shin, 1998)
# =============================================================================

"""
    pvar_girf(model::PVARModel{T}, H::Int) -> Array{T, 3}

Generalized impulse response functions (Pesaran & Shin, 1998).

GIRF_h(j) = Φ_h Σ e_j / √σ_jj

where e_j is the j-th unit vector and σ_jj = Σ[j,j].

Returns H+1 × m × m array: `girf[h+1, response, shock]`.
"""
function pvar_girf(model::PVARModel{T}, H::Int) where {T}
    H < 0 && throw(ArgumentError("Horizon H must be non-negative"))
    m_dim = model.m
    Sigma = model.Sigma

    Phi_h = _pvar_ma_coefficients(model, H)

    girf = Array{T}(undef, H + 1, m_dim, m_dim)
    for h in 0:H
        for j in 1:m_dim
            scale = sqrt(max(Sigma[j, j], T(1e-16)))
            girf[h+1, :, j] = Phi_h[h+1] * Sigma[:, j] ./ scale
        end
    end
    girf
end

# =============================================================================
# FEVD — Forecast Error Variance Decomposition
# =============================================================================

"""
    pvar_fevd(model::PVARModel{T}, H::Int) -> Array{T, 3}

Forecast error variance decomposition based on OIRF.

Ω[l, k, h] = Σ_{j=0}^{h} (Ψ_j[l,k])² / MSE_h[l,l]

where MSE_h = Σ_{j=0}^{h} Φ_j Σ Φ_j'.

Returns H+1 × m × m array: `fevd[h+1, variable, shock]`. Each row sums to 1.

# Examples
```julia
fv = pvar_fevd(model, 10)
sum(fv[11, 1, :])  # ≈ 1.0 (all shocks for var 1 at h=10)
```
"""
function pvar_fevd(model::PVARModel{T}, H::Int) where {T}
    H < 0 && throw(ArgumentError("Horizon H must be non-negative"))
    m_dim = model.m

    oirf = pvar_oirf(model, H)
    Phi_h = _pvar_ma_coefficients(model, H)
    Sigma = model.Sigma

    fevd_arr = Array{T}(undef, H + 1, m_dim, m_dim)

    for h in 0:H
        # MSE at horizon h
        mse = zeros(T, m_dim, m_dim)
        for j in 0:h
            mse .+= Phi_h[j+1] * Sigma * Phi_h[j+1]'
        end

        # Cumulative squared OIRF contributions
        for l in 1:m_dim
            mse_ll = max(mse[l, l], T(1e-16))
            for k in 1:m_dim
                contrib = zero(T)
                for j in 0:h
                    contrib += oirf[j+1, l, k]^2
                end
                fevd_arr[h+1, l, k] = contrib / mse_ll
            end
        end
    end
    fevd_arr
end

# =============================================================================
# Stability analysis
# =============================================================================

"""
    pvar_stability(model::PVARModel{T}) -> PVARStability{T}

Check stability of the PVAR by computing eigenvalues of the companion matrix.
The system is stable if all eigenvalue moduli are strictly less than 1.

# Examples
```julia
stab = pvar_stability(model)
stab.is_stable  # true if all |λ| < 1
stab.moduli     # sorted eigenvalue moduli
```
"""
function pvar_stability(model::PVARModel{T}) where {T}
    F = _pvar_companion(model)
    eigenvals = eigvals(F)
    moduli = abs.(eigenvals)

    # Sort by modulus (descending)
    perm = sortperm(moduli, rev=true)
    eigenvals = eigenvals[perm]
    moduli = moduli[perm]

    is_stable = all(m -> m < one(T), moduli)
    PVARStability{T}(eigenvals, moduli, is_stable)
end
