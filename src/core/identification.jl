# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
    # Sign-normalize columns by the QR pivots. Use an explicit ±1 map (not `sign`, whose
    # sign(0.0)=0.0 would zero an entire rotation column when a pivot is exactly 0).
    d = [r < zero(T) ? -one(T) : one(T) for r in diag(R)]
    Matrix(Q) * Diagonal(d)
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
    A = extract_ar_coefficients(model.B, n, p)      # Vector{Matrix{T}} (contiguous n×n)

    IRF = zeros(T, horizon, n, n)
    Phi = [zeros(T, n, n) for _ in 1:horizon]       # per-horizon contiguous buffers
    temp = zeros(T, n, n)
    scratch = zeros(T, n, n)
    copyto!(Phi[1], I(n))
    IRF[1, :, :] = P

    @inbounds for h in 2:horizon
        fill!(temp, zero(T))
        for j in 1:min(p, h-1)
            mul!(scratch, A[j], Phi[h-j])           # in-place gemm, no A[j]*view alloc
            temp .+= scratch
        end
        Phi[h] .= temp
        mul!(scratch, temp, P)
        IRF[h, :, :] = scratch
    end
    IRF
end

"""Compute structural shocks: εₜ = Q'L⁻¹uₜ."""
function compute_structural_shocks(model::VARModel{T}, Q::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = safe_cholesky(model.Sigma)          # lower-triangular Cholesky factor
    (Q' * (L \ model.U'))'                    # L \ U' = L⁻¹U' via triangular backsolve
end

# =============================================================================
# Sign Restrictions
# =============================================================================

"""
    identify_sign(model, horizon, check_func; max_draws=1000, store_all=false)

Find Q satisfying sign restrictions via random draws.

With `store_all=false` (default), returns `(Q, irf)` — the first valid rotation.
With `store_all=true`, returns a `SignIdentifiedSet` containing ALL accepted
rotations and their IRFs (Baumeister & Hamilton, 2015).
"""
function identify_sign(model::VARModel{T}, horizon::Int, check_func::Function;
                       max_draws::Int=1000, store_all::Bool=false,
                       shock_names::Union{Nothing,Vector{String}}=nothing) where {T<:AbstractFloat}
    n = nvars(model)

    if !store_all
        # Original behavior: return first valid Q
        for _ in 1:max_draws
            Q = generate_Q(n, T)
            irf_result = compute_irf(model, Q, horizon)
            check_func(irf_result) && return Q, irf_result
        end
        error("No valid Q found after $max_draws draws")
    end

    # Full identified set: collect ALL valid rotations
    accepted_Q = Matrix{T}[]
    accepted_irf_list = Array{T,3}[]

    for _ in 1:max_draws
        Q = generate_Q(n, T)
        irf_result = compute_irf(model, Q, horizon)
        if check_func(irf_result)
            push!(accepted_Q, Q)
            push!(accepted_irf_list, irf_result)
        end
    end

    n_accepted = length(accepted_Q)
    n_accepted == 0 && error("No valid Q found after $max_draws draws")

    # Stack IRFs into 4D array (n_accepted × horizon × n × n)
    irf_draws = zeros(T, n_accepted, horizon, n, n)
    for (i, irf_i) in enumerate(accepted_irf_list)
        irf_draws[i, :, :, :] = irf_i
    end

    acceptance_rate = T(n_accepted) / T(max_draws)

    snames = isnothing(shock_names) ? model.varnames : shock_names
    SignIdentifiedSet{T}(accepted_Q, irf_draws, n_accepted, max_draws, acceptance_rate,
                         model.varnames, snames)
end

"""
    irf_bounds(s::SignIdentifiedSet{T}; quantiles=[0.16, 0.84]) -> (lower, upper)

Compute pointwise bounds (or quantile bands) over the identified set.
"""
function irf_bounds(s::SignIdentifiedSet{T}; quantiles::Vector{<:Real}=T[0.16, 0.84]) where {T}
    q = T.(quantiles)
    H, n = size(s.irf_draws, 2), size(s.irf_draws, 3)
    lower = zeros(T, H, n, n)
    upper = zeros(T, H, n, n)
    for h in 1:H, i in 1:n, j in 1:n
        d = @view s.irf_draws[:, h, i, j]
        lower[h, i, j] = quantile(d, q[1])
        upper[h, i, j] = quantile(d, q[2])
    end
    (lower, upper)
end

"""
    irf_median(s::SignIdentifiedSet{T}) -> Array{T,3}

Compute pointwise median IRF over the identified set.
"""
function irf_median(s::SignIdentifiedSet{T}) where {T}
    H, n = size(s.irf_draws, 2), size(s.irf_draws, 3)
    med = zeros(T, H, n, n)
    for h in 1:H, i in 1:n, j in 1:n
        d = @view s.irf_draws[:, h, i, j]
        med[h, i, j] = quantile(d, T(0.5))
    end
    med
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

"""
Identify via long-run restrictions (Blanchard–Quah): the long-run cumulative impact matrix is
lower triangular. Shocks are sign-normalized so each permanent shock has a non-negative long-run
effect on its own variable (a positive diagonal of the long-run cumulative impact matrix) — the
standard BQ normalization, applied explicitly so shock signs are deterministic (audit F-05; the
reference `iresponse_longrun.m` normalizes only the impact sign of the first shock).
"""
function identify_long_run(model::VARModel{T}) where {T<:AbstractFloat}
    n, p = nvars(model), model.p
    A_sum = sum(extract_ar_coefficients(model.B, n, p))
    M = Matrix{T}(I(n) - A_sum)
    cM = cond(M)
    cM > one(T) / sqrt(eps(T)) && @warn "identify_long_run: (I − ΣAᵢ) is near-singular (cond ≈ $(cM)); the VAR is near a unit root, so the long-run impact matrix is numerically unstable." maxlog = 1
    inv_lag = robust_inv(M; silent=true)
    V_LR = inv_lag * model.Sigma * inv_lag'
    D = Matrix(safe_cholesky(V_LR))   # lower-triangular; D == long-run cumulative impact matrix
    # Sign-normalize: long-run own-effect (diag of D) non-negative for every shock.
    @inbounds for j in 1:n
        D[j, j] < zero(T) && (@views D[:, j] .*= -one(T))
    end
    P = M * D
    safe_cholesky(model.Sigma) \ P     # L⁻¹P via triangular backsolve
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
