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
Structural identification: Cholesky, sign restrictions, narrative, long-run, Arias et al. (2018).
"""

using LinearAlgebra, Random, Statistics
using Distributions: loggamma

# =============================================================================
# Cholesky Identification
# =============================================================================

"""Identify via Cholesky decomposition (recursive ordering). Returns L where Σ = LL'."""
identify_cholesky(model::VARModel{T}) where {T<:AbstractFloat} = safe_cholesky(model.Sigma)

# =============================================================================
# Random Orthogonal Matrix
# =============================================================================

"""Generate random orthogonal matrix via QR decomposition (Haar measure)."""
function generate_Q(n::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    X = randn(T, n, n)
    Q, R = qr(X)
    Matrix(Q) * Diagonal(sign.(diag(R)))
end

# =============================================================================
# IRF Computation
# =============================================================================

"""
    compute_irf(model, Q, horizon) -> Array{T,3}

Compute IRFs for rotation matrix Q. Returns (horizon × n × n) array.
IRF[h, i, j] = response of variable i to shock j at horizon h-1.
"""
function compute_irf(model::VARModel{T}, Q::AbstractMatrix{T}, horizon::Int) where {T<:AbstractFloat}
    n, p = nvars(model), model.p
    P = safe_cholesky(model.Sigma) * Q

    IRF, Phi = zeros(T, horizon, n, n), zeros(T, horizon, n, n)
    Phi[1, :, :], IRF[1, :, :] = I(n), P

    A = extract_ar_coefficients(model.B, n, p)
    @inbounds for h in 2:horizon
        temp = zeros(T, n, n)
        for j in 1:min(p, h-1)
            temp .+= A[j] * @view(Phi[h-j, :, :])
        end
        Phi[h, :, :], IRF[h, :, :] = temp, temp * P
    end
    IRF
end

"""Compute structural shocks: εₜ = Q'L⁻¹uₜ."""
function compute_structural_shocks(model::VARModel{T}, Q::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = safe_cholesky(model.Sigma)
    (Q' * robust_inv(Matrix(L)) * model.U')'
end

# =============================================================================
# Sign Restrictions
# =============================================================================

"""
    identify_sign(model, horizon, check_func; max_draws=1000) -> (Q, irf)

Find Q satisfying sign restrictions via random draws.
"""
function identify_sign(model::VARModel{T}, horizon::Int, check_func::Function;
                       max_draws::Int=1000) where {T<:AbstractFloat}
    n = nvars(model)
    for _ in 1:max_draws
        Q = generate_Q(n, T)
        irf = compute_irf(model, Q, horizon)
        check_func(irf) && return Q, irf
    end
    error("No valid Q found after $max_draws draws")
end

# =============================================================================
# Narrative Restrictions
# =============================================================================

"""
    identify_narrative(model, horizon, sign_check, narrative_check; max_draws=1000)

Combine sign and narrative restrictions. Returns (Q, irf, shocks).
"""
function identify_narrative(model::VARModel{T}, horizon::Int, sign_check::Function,
                            narrative_check::Function; max_draws::Int=1000) where {T<:AbstractFloat}
    n = nvars(model)
    for _ in 1:max_draws
        Q = generate_Q(n, T)
        irf = compute_irf(model, Q, horizon)
        if sign_check(irf)
            shocks = compute_structural_shocks(model, Q)
            narrative_check(shocks) && return Q, irf, shocks
        end
    end
    error("No valid Q found after $max_draws draws")
end

# =============================================================================
# Long-Run Restrictions (Blanchard-Quah)
# =============================================================================

"""Identify via long-run restrictions: long-run cumulative impact matrix is lower triangular."""
function identify_long_run(model::VARModel{T}) where {T<:AbstractFloat}
    n, p = nvars(model), model.p
    A_sum = sum(extract_ar_coefficients(model.B, n, p))
    inv_lag = robust_inv(I(n) - A_sum)
    V_LR = inv_lag * model.Sigma * inv_lag'
    D = safe_cholesky(V_LR)
    P = (I(n) - A_sum) * D
    robust_inv(Matrix(safe_cholesky(model.Sigma))) * P
end

# =============================================================================
# Unified Interface
# =============================================================================

"""
    compute_Q(model, method, horizon, check_func, narrative_check;
              max_draws=100, transition_var=nothing, regime_indicator=nothing)

Compute identification matrix Q for structural VAR analysis.

# Methods
- `:cholesky` — Cholesky decomposition (recursive ordering)
- `:sign` — Sign restrictions (requires `check_func`)
- `:narrative` — Narrative restrictions (requires `check_func` and `narrative_check`)
- `:long_run` — Long-run restrictions (Blanchard-Quah)
- `:fastica` — FastICA (Hyvärinen 1999)
- `:jade` — JADE (Cardoso 1999)
- `:sobi` — SOBI (Belouchrani et al. 1997)
- `:dcov` — Distance covariance ICA (Matteson & Tsay 2017)
- `:hsic` — HSIC independence ICA (Gretton et al. 2005)
- `:student_t` — Student-t ML (Lanne et al. 2017)
- `:mixture_normal` — Mixture of normals ML (Lanne et al. 2017)
- `:pml` — Pseudo-ML (Gouriéroux et al. 2017)
- `:skew_normal` — Skew-normal ML (Lanne & Luoto 2020)
- `:nongaussian_ml` — Unified non-Gaussian ML dispatcher (default: Student-t)
- `:markov_switching` — Markov-switching heteroskedasticity (Lütkepohl & Netšunajev 2017)
- `:garch` — GARCH-based heteroskedasticity (Normandin & Phaneuf 2004)
- `:smooth_transition` — Smooth-transition heteroskedasticity (requires `transition_var`)
- `:external_volatility` — External volatility regimes (requires `regime_indicator`)

# Keyword Arguments
- `max_draws::Int=100`: Maximum draws for sign/narrative identification
- `transition_var::Union{Nothing,AbstractVector}=nothing`: Transition variable for `:smooth_transition`
- `regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing`: Regime indicator for `:external_volatility`
"""
function compute_Q(model::VARModel{T}, method::Symbol, horizon::Int, check_func, narrative_check;
                   max_draws::Int=100,
                   transition_var::Union{Nothing,AbstractVector}=nothing,
                   regime_indicator::Union{Nothing,AbstractVector{Int}}=nothing) where {T<:AbstractFloat}
    n = nvars(model)
    method == :cholesky && return Matrix{T}(I, n, n)
    method == :sign && (isnothing(check_func) && throw(ArgumentError("Need check_func for sign"));
                        return identify_sign(model, horizon, check_func; max_draws)[1])
    method == :narrative && (isnothing(check_func) || isnothing(narrative_check)) &&
        throw(ArgumentError("Need check_func and narrative_check for narrative"))
    method == :narrative && return identify_narrative(model, horizon, check_func, narrative_check; max_draws)[1]
    method == :long_run && return identify_long_run(model)

    # Non-Gaussian ICA methods (defined in nongaussian_ica.jl, loaded after this file)
    method == :fastica       && return identify_fastica(model).Q
    method == :jade          && return identify_jade(model).Q
    method == :sobi          && return identify_sobi(model).Q
    method == :dcov          && return identify_dcov(model).Q
    method == :hsic          && return identify_hsic(model).Q

    # Non-Gaussian ML methods (defined in nongaussian_ml.jl)
    method == :student_t      && return identify_student_t(model).Q
    method == :mixture_normal && return identify_mixture_normal(model).Q
    method == :pml            && return identify_pml(model).Q
    method == :skew_normal    && return identify_skew_normal(model).Q
    method == :nongaussian_ml && return identify_nongaussian_ml(model).Q

    # Heteroskedasticity methods (defined in heteroskedastic_id.jl)
    method == :markov_switching && return identify_markov_switching(model).Q
    method == :garch            && return identify_garch(model).Q
    method == :smooth_transition && (isnothing(transition_var) &&
        throw(ArgumentError("smooth_transition requires transition_var kwarg"));
        return identify_smooth_transition(model, transition_var).Q)
    method == :external_volatility && (isnothing(regime_indicator) &&
        throw(ArgumentError("external_volatility requires regime_indicator kwarg"));
        return identify_external_volatility(model, regime_indicator).Q)

    throw(ArgumentError("Unknown method: $method"))
end


# Arias et al. (2018) identification — extracted to arias.jl
include("arias.jl")
