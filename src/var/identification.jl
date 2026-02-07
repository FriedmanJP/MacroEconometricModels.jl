"""
Structural identification: Cholesky, sign restrictions, narrative, long-run, Arias et al. (2018).
"""

using LinearAlgebra, Random, Statistics
using SpecialFunctions: loggamma

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

# =============================================================================
# Arias, Rubio-Ramírez, Waggoner (2018) - Zero + Sign Restrictions
# =============================================================================

"""Zero restriction: variable doesn't respond to shock at horizon."""
struct ZeroRestriction
    variable::Int
    shock::Int
    horizon::Int
end

"""Sign restriction: variable response to shock has required sign at horizon."""
struct SignRestriction
    variable::Int
    shock::Int
    horizon::Int
    sign::Int  # +1 or -1
end

"""Container for SVAR restrictions."""
struct SVARRestrictions
    zeros::Vector{ZeroRestriction}
    signs::Vector{SignRestriction}
    n_vars::Int
    n_shocks::Int
end

SVARRestrictions(n_vars::Int; zeros=ZeroRestriction[], signs=SignRestriction[]) =
    SVARRestrictions(zeros, signs, n_vars, n_vars)

"""Result from Arias et al. (2018) identification."""
struct AriasSVARResult{T<:AbstractFloat}
    Q_draws::Vector{Matrix{T}}
    irf_draws::Array{T,4}
    weights::Vector{T}
    acceptance_rate::T
    restrictions::SVARRestrictions
end

# --- MA Coefficients ---

"""Compute MA coefficients Φ_0, ..., Φ_horizon."""
function _compute_ma_coefficients(model::VARModel{T}, horizon::Int) where {T<:AbstractFloat}
    n, p = nvars(model), model.p
    A = extract_ar_coefficients(model.B, n, p)
    Phi = Vector{Matrix{T}}(undef, horizon + 1)
    Phi[1] = Matrix{T}(I, n, n)
    for h in 1:horizon
        Phi[h + 1] = sum(A[j] * Phi[h - j + 1] for j in 1:min(p, h); init=zeros(T, n, n))
    end
    Phi
end

"""Draw uniformly from O(n) via QR decomposition."""
function _draw_uniform_orthogonal(n::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    X = randn(T, n, n)
    F = qr(X)
    Q = Matrix(F.Q)
    R_diag = diag(F.R)
    for j in 1:n
        R_diag[j] < 0 && (Q[:, j] = -Q[:, j])
    end
    Q
end

"""Compute structural IRF for rotation Q."""
function _compute_irf_for_Q(model::VARModel{T}, Q::Matrix{T}, Phi::Vector{Matrix{T}},
                            L::LowerTriangular{T,Matrix{T}}, horizon::Int) where {T<:AbstractFloat}
    n = nvars(model)
    A0_inv = L * Q
    irf = zeros(T, horizon, n, n)
    for h in 1:horizon
        irf[h, :, :] = Phi[h] * A0_inv
    end
    irf
end

# --- Restriction Checking ---

"""Check if all zero restrictions are satisfied."""
_check_zero_restrictions(irf::Array{T,3}, r::SVARRestrictions; tol::T=T(1e-10)) where {T} =
    all(abs(irf[zr.horizon + 1, zr.variable, zr.shock]) <= tol for zr in r.zeros)

"""Check if all sign restrictions are satisfied."""
_check_sign_restrictions(irf::Array{T,3}, r::SVARRestrictions) where {T} =
    all(sr.sign > 0 ? irf[sr.horizon + 1, sr.variable, sr.shock] > 0 :
                      irf[sr.horizon + 1, sr.variable, sr.shock] < 0 for sr in r.signs)

# --- Zero Restriction Algorithm ---

"""Build constraint matrix for zero restrictions on shock j."""
_build_zero_constraint_matrix(r::SVARRestrictions, shock::Int, Phi::Vector{Matrix{T}},
                               L::LowerTriangular{T,Matrix{T}}) where {T} =
    [Vector{T}((Phi[zr.horizon + 1] * L)[zr.variable, :]) for zr in r.zeros if zr.shock == shock]

"""Draw unit vector from null space of constraints."""
function _draw_null_space_vector(constraints::Vector{Vector{T}}, n::Int) where {T<:AbstractFloat}
    isempty(constraints) && return (x = randn(T, n); x / norm(x))

    F = reduce(vcat, [c' for c in constraints])
    svd_result = svd(F, full=true)
    V = transpose(svd_result.Vt)
    tol = max(size(F)...) * eps(T) * (isempty(svd_result.S) ? one(T) : maximum(svd_result.S))
    rank_F = sum(svd_result.S .> tol)
    null_dim = n - rank_F
    null_dim <= 0 && error("Zero restrictions over-constrain shock")

    N = V[:, (rank_F + 1):n]
    z = randn(T, null_dim)
    q = N * z
    q / norm(q)
end

"""Draw orthogonal Q satisfying zero restrictions (Algorithm 2, Arias et al. 2018)."""
function _draw_Q_with_zero_restrictions(r::SVARRestrictions, Phi::Vector{Matrix{T}},
                                         L::LowerTriangular{T,Matrix{T}}) where {T<:AbstractFloat}
    n = r.n_vars
    Q = zeros(T, n, n)
    for j in 1:n
        zero_constraints = _build_zero_constraint_matrix(r, j, Phi, L)
        ortho_constraints = [Vector{T}(Q[:, k]) for k in 1:j-1]
        Q[:, j] = _draw_null_space_vector(vcat(zero_constraints, ortho_constraints), n)
    end
    @assert norm(Q' * Q - I) < 1e-10 "Q not orthogonal"
    Q
end

# --- Importance Weights ---

"""Compute importance weight for Q (corrects non-uniform prior from zero restrictions)."""
function _compute_importance_weight(Q::Matrix{T}, r::SVARRestrictions,
                                     Phi::Vector{Matrix{T}}, L::LowerTriangular{T,Matrix{T}}) where {T}
    isempty(r.zeros) && return one(T)

    n = size(Q, 1)
    zeros_per_shock = zeros(Int, n)
    for zr in r.zeros
        zeros_per_shock[zr.shock] += 1
    end

    log_weight = zero(T)
    for j in 1:n
        dim_free = n - zeros_per_shock[j] - (j - 1)
        if dim_free > 0
            log_weight += (dim_free - 1) / 2 * log(T(π)) - loggamma(T(dim_free) / 2)
            log_weight -= (n - j) / 2 * log(T(π)) - loggamma(T(n - j + 1) / 2)
        end
    end
    exp(log_weight)
end

# --- Main Arias Identification ---

"""
    identify_arias(model, restrictions, horizon; n_draws=1000, n_rotations=1000) -> AriasSVARResult

Identify SVAR using Arias et al. (2018) with zero and sign restrictions.
"""
function identify_arias(model::VARModel{T}, restrictions::SVARRestrictions, horizon::Int;
                        n_draws::Int=1000, n_rotations::Int=1000) where {T<:AbstractFloat}
    n = nvars(model)
    @assert restrictions.n_vars == n "Restriction dimension must match model"

    max_h = max(horizon,
        isempty(restrictions.zeros) ? 0 : maximum(zr.horizon for zr in restrictions.zeros) + 1,
        isempty(restrictions.signs) ? 0 : maximum(sr.horizon for sr in restrictions.signs) + 1)

    Phi, L = _compute_ma_coefficients(model, max_h), safe_cholesky(model.Sigma)
    Q_draws, irf_draws, weights = Matrix{T}[], Array{T,3}[], T[]
    has_zeros, n_attempts = !isempty(restrictions.zeros), 0

    while length(Q_draws) < n_draws && n_attempts < n_draws * n_rotations
        n_attempts += 1
        try
            Q = has_zeros ? _draw_Q_with_zero_restrictions(restrictions, Phi, L) :
                            _draw_uniform_orthogonal(n, T)
            irf = _compute_irf_for_Q(model, Q, Phi, L, horizon)

            (has_zeros && !_check_zero_restrictions(irf, restrictions)) && continue
            !_check_sign_restrictions(irf, restrictions) && continue

            push!(Q_draws, Q)
            push!(irf_draws, irf)
            push!(weights, _compute_importance_weight(Q, restrictions, Phi, L))
        catch; continue; end
    end

    isempty(Q_draws) && error("No valid identification after $n_attempts attempts")

    n_acc = length(Q_draws)
    irf_array = zeros(T, n_acc, horizon, n, n)
    for (i, irf) in enumerate(irf_draws)
        irf_array[i, :, :, :] = irf
    end

    AriasSVARResult{T}(Q_draws, irf_array, weights ./ sum(weights), T(n_acc / n_attempts), restrictions)
end

# --- Bayesian Integration ---

"""
    identify_arias_bayesian(chain, p, n, restrictions, horizon; data=nothing, n_rotations=100, quantiles=[0.16,0.5,0.84])

Apply Arias identification to each posterior draw. Returns IRF quantiles, mean, acceptance rates.
"""
function identify_arias_bayesian(chain, p::Int, n::Int, restrictions::SVARRestrictions, horizon::Int;
    data::Union{Nothing,AbstractMatrix}=nothing, n_rotations::Int=100, quantiles::Vector{Float64}=[0.16, 0.5, 0.84])

    b_vecs, sigmas = extract_chain_parameters(chain)
    n_samples = size(b_vecs, 1)
    all_irfs, all_weights = Vector{Array{Float64,3}}(), Float64[]
    acc_rates = zeros(n_samples)

    for s in 1:n_samples
        m = parameters_to_model(b_vecs[s,:], sigmas[s,:], p, n, data)
        try
            result = identify_arias(m, restrictions, horizon; n_draws=1, n_rotations=n_rotations)
            for (i, w) in enumerate(result.weights)
                push!(all_irfs, result.irf_draws[i, :, :, :])
                push!(all_weights, w)
            end
            acc_rates[s] = result.acceptance_rate
        catch
            acc_rates[s] = 0.0
        end
    end

    isempty(all_irfs) && error("No valid identifications across posterior")

    n_acc = length(all_irfs)
    irf_array = zeros(n_acc, horizon, n, n)
    for (i, irf) in enumerate(all_irfs)
        irf_array[i, :, :, :] = irf
    end
    w_norm = all_weights ./ sum(all_weights)

    irf_q = zeros(horizon, n, n, length(quantiles))
    irf_m = zeros(horizon, n, n)
    for h in 1:horizon, i in 1:n, j in 1:n
        vals = irf_array[:, h, i, j]
        irf_m[h, i, j] = sum(w_norm .* vals)
        for (qi, q) in enumerate(quantiles)
            irf_q[h, i, j, qi] = _weighted_quantile(vals, w_norm, q)
        end
    end

    (irf_quantiles=irf_q, irf_mean=irf_m, acceptance_rates=acc_rates, total_accepted=n_acc, weights=w_norm)
end

"""Weighted quantile via linear interpolation."""
function _weighted_quantile(vals::AbstractVector{T}, weights::AbstractVector{S}, q::Real) where {T,S}
    perm = sortperm(vals)
    sv, sw = vals[perm], weights[perm]
    cw = cumsum(sw)
    cw ./= cw[end]
    idx = searchsortedfirst(cw, q)
    idx == 1 && return sv[1]
    idx > length(sv) && return sv[end]
    t = (q - cw[idx-1]) / (cw[idx] - cw[idx-1] + eps())
    (1 - t) * sv[idx-1] + t * sv[idx]
end

# --- Convenience Functions ---

"""Create zero restriction: variable doesn't respond to shock at horizon."""
zero_restriction(variable::Int, shock::Int; horizon::Int=0) = ZeroRestriction(variable, shock, horizon)

"""Create sign restriction: variable response has given sign (:positive/:negative) at horizon."""
sign_restriction(variable::Int, shock::Int, sign::Symbol; horizon::Int=0) =
    SignRestriction(variable, shock, horizon, sign == :positive ? 1 : -1)

"""Compute weighted IRF percentiles from AriasSVARResult."""
function irf_percentiles(result::AriasSVARResult{T}; probs::Vector{Float64}=[0.16, 0.5, 0.84]) where {T}
    n_draws, horizon, n_vars, n_shocks = size(result.irf_draws)
    pct = zeros(T, horizon, n_vars, n_shocks, length(probs))
    for h in 1:horizon, i in 1:n_vars, j in 1:n_shocks
        for (pi, p) in enumerate(probs)
            pct[h, i, j, pi] = _weighted_quantile(result.irf_draws[:, h, i, j], result.weights, p)
        end
    end
    pct
end

"""Compute weighted mean IRF from AriasSVARResult."""
function irf_mean(result::AriasSVARResult{T}) where {T}
    n_draws, horizon, n_vars, n_shocks = size(result.irf_draws)
    mean_irf = zeros(T, horizon, n_vars, n_shocks)
    for h in 1:horizon, i in 1:n_vars, j in 1:n_shocks
        mean_irf[h, i, j] = sum(result.weights .* result.irf_draws[:, h, i, j])
    end
    mean_irf
end
