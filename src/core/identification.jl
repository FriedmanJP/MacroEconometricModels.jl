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

"""Identify via Cholesky decomposition (recursive ordering). Returns Q = I."""
identify_cholesky(model::VARModel{T}) where {T<:AbstractFloat} =
    Matrix{T}(I, nvars(model), nvars(model))

"""Lower-triangular Cholesky factor L of `model.Sigma` (Σ = LL')."""
cholesky_factor(model::VARModel{T}) where {T<:AbstractFloat} = safe_cholesky(model.Sigma)

# =============================================================================
# External-instrument (proxy) identification
# =============================================================================

"""
    identify_proxy(model::VARModel, z; normalize=1, normalize_value=1) -> NamedTuple

External-instrument identification of the first structural shock (Stock & Watson
2012; Mertens & Ravn 2013). `z` is aligned to the VAR residual sample (`T_eff`
rows, or the full `T` with the first `p` observations dropped). Missing/`NaN`
rows are dropped pairwise.

Returns `(Q, b1, first_stage_F, z_eff)` where `Q` is an orthogonal rotation with
the first column identified, `b1` is the impact column (`Σ û z` normalised so
variable `normalize` responds by `normalize_value`), and `first_stage_F` is the
OLS F-statistic of the residual projection on `z`. Warns when that F is below 10.
"""
function identify_proxy(model::VARModel{T}, z::AbstractVector;
                        normalize::Int=1, normalize_value::Real=one(T)) where {T<:AbstractFloat}
    n = nvars(model)
    (1 <= normalize <= n) || throw(ArgumentError("normalize must be in 1:$n"))
    U = model.U
    T_eff = size(U, 1)
    z_eff = _align_instrument(z, size(model.Y, 1), model.p, T_eff)
    mask = [isfinite(z_eff[t]) && all(isfinite, @view U[t, :]) for t in 1:T_eff]
    count(mask) < 8 && throw(ArgumentError("instrument has too few finite observations"))
    Uu = U[mask, :]
    zz = z_eff[mask]
    zc = zz .- mean(zz)
    Uc = Uu .- mean(Uu; dims=1)
    nobs = length(zz)
    Suz = vec(Uc' * zc) / T(nobs - 1)
    P = safe_cholesky(model.Sigma)
    denom = Suz[normalize]
    abs(denom) < T(1e-12) && throw(ArgumentError(
        "instrument is uncorrelated with residual $normalize; cannot normalise"))
    b1 = Suz ./ denom .* T(normalize_value)
    Q = _complete_proxy_Q(P, b1, n, T)
    B0 = P * Q
    scale = B0[normalize, 1]
    abs(scale) < T(1e-12) && throw(ArgumentError("proxy impact on the normalising variable is zero"))
    Q[:, 1] .*= T(normalize_value) / scale
    y = Uc * (Suz ./ (norm(Suz) + T(1e-12)))
    Fstat = _ols_first_stage_F(y, zc)
    if Fstat < T(10)
        @warn "Weak instrument: first-stage F = $(round(Fstat, digits=2)) < 10 (Montiel Olea, Stock & Watson 2021)"
    end
    (Q=Q, b1=Vector{T}(b1), first_stage_F=T(Fstat), z_eff=Vector{T}(z_eff))
end

function _align_instrument(z::AbstractVector, T_obs::Int, p::Int, T_eff::Int)
    n = length(z)
    n == T_eff && return float.(z)
    n == T_obs && return float.(z[(p + 1):end])
    throw(ArgumentError(
        "instrument length $n must equal residual sample $T_eff or full sample $T_obs"))
end

function _ols_first_stage_F(y::AbstractVector{T}, zc::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(y)
    n < 3 && return zero(T)
    denom = dot(zc, zc)
    denom <= zero(T) && return zero(T)
    b = dot(zc, y) / denom
    fitted = b .* zc
    yc = y .- mean(y)
    sse = dot(yc - fitted, yc - fitted)
    ssr = dot(fitted, fitted)
    sse <= zero(T) && return T(Inf)
    T((ssr / 1) / (sse / (n - 2)))
end

function _complete_proxy_Q(P::AbstractMatrix{T}, b1::AbstractVector{T}, n::Int, ::Type{T}) where {T}
    q1 = P \ b1
    nrm = norm(q1)
    nrm < T(1e-12) && throw(ArgumentError("proxy impact column is numerically zero"))
    M = Matrix{T}(I, n, n)
    M[:, 1] = q1 ./ nrm
    Fq = qr(M)
    Q = Matrix{T}(Fq.Q)
    s = [Fq.R[i, i] < zero(T) ? -one(T) : one(T) for i in 1:n]
    Q * Diagonal(s)
end

# =============================================================================
# Random Orthogonal Matrix
# =============================================================================

"""Generate random orthogonal matrix via QR decomposition (Haar measure)."""
function generate_Q(n::Int, ::Type{T}=Float64; rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    X = randn(rng, T, n, n)
    Q, R = qr(X)
    # Sign-normalize columns by the QR pivots. Use an explicit ±1 map (not `sign`, whose
    # sign(0.0)=0.0 would zero an entire rotation column when a pivot is exactly 0).
    d = [r < zero(T) ? -one(T) : one(T) for r in diag(R)]
    Matrix(Q) * Diagonal(d)
end

"""Haar-uniform draw from O(n); alias of [`generate_Q`](@ref)."""
const haar_orthogonal = generate_Q

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
                       shock_names::Union{Nothing,Vector{String}}=nothing,
                       rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(model)

    if !store_all
        # Original behavior: return first valid Q
        for _ in 1:max_draws
            Q = generate_Q(n, T; rng=rng)
            irf_result = compute_irf(model, Q, horizon)
            check_func(irf_result) && return Q, irf_result
        end
        throw(IdentificationError("No valid Q found after $max_draws draws"))
    end

    # Full identified set: collect ALL valid rotations
    accepted_Q = Matrix{T}[]
    accepted_irf_list = Array{T,3}[]

    for _ in 1:max_draws
        Q = generate_Q(n, T; rng=rng)
        irf_result = compute_irf(model, Q, horizon)
        if check_func(irf_result)
            push!(accepted_Q, Q)
            push!(accepted_irf_list, irf_result)
        end
    end

    n_accepted = length(accepted_Q)
    n_accepted == 0 && throw(IdentificationError("No valid Q found after $max_draws draws"))

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
function irf_bounds(s::SignIdentifiedSet{T}; quantiles=[0.16, 0.84]) where {T}
    q = T.(quantiles)
    H, n_var, n_shock = size(s.irf_draws, 2), size(s.irf_draws, 3), size(s.irf_draws, 4)
    lower = zeros(T, H, n_var, n_shock)
    upper = zeros(T, H, n_var, n_shock)
    for h in 1:H, i in 1:n_var, j in 1:n_shock
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
    H, n_var, n_shock = size(s.irf_draws, 2), size(s.irf_draws, 3), size(s.irf_draws, 4)
    med = zeros(T, H, n_var, n_shock)
    for h in 1:H, i in 1:n_var, j in 1:n_shock
        d = @view s.irf_draws[:, h, i, j]
        med[h, i, j] = quantile(d, T(0.5))
    end
    med
end

# =============================================================================
# Narrative Restrictions
# =============================================================================

"""
    identify_narrative(model, horizon, sign_check, narrative_check; max_draws=1000, store_all=false)

Combine sign and narrative restrictions.

With `store_all=false` (default), returns `(Q, irf, shocks)` — the first valid rotation.
With `store_all=true`, returns a `SignIdentifiedSet` of every accepted rotation (same shape
as [`identify_sign`](@ref)).
"""
function identify_narrative(model::VARModel{T}, horizon::Int, sign_check::Function,
                            narrative_check::Function; max_draws::Int=1000,
                            store_all::Bool=false,
                            shock_names::Union{Nothing,Vector{String}}=nothing,
                            rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(model)

    if !store_all
        for _ in 1:max_draws
            Q = generate_Q(n, T; rng=rng)
            irf_result = compute_irf(model, Q, horizon)
            if sign_check(irf_result)
                shocks = compute_structural_shocks(model, Q)
                narrative_check(shocks) && return Q, irf_result, shocks
            end
        end
        throw(IdentificationError("No valid Q found after $max_draws draws"))
    end

    accepted_Q = Matrix{T}[]
    accepted_irf_list = Array{T,3}[]
    for _ in 1:max_draws
        Q = generate_Q(n, T; rng=rng)
        irf_result = compute_irf(model, Q, horizon)
        if sign_check(irf_result)
            shocks = compute_structural_shocks(model, Q)
            if narrative_check(shocks)
                push!(accepted_Q, Q)
                push!(accepted_irf_list, irf_result)
            end
        end
    end

    n_accepted = length(accepted_Q)
    n_accepted == 0 && throw(IdentificationError("No valid Q found after $max_draws draws"))

    irf_draws = zeros(T, n_accepted, horizon, n, n)
    for (i, irf_i) in enumerate(accepted_irf_list)
        irf_draws[i, :, :, :] = irf_i
    end

    acceptance_rate = T(n_accepted) / T(max_draws)
    snames = isnothing(shock_names) ? model.varnames : shock_names
    SignIdentifiedSet{T}(accepted_Q, irf_draws, n_accepted, max_draws, acceptance_rate,
                         model.varnames, snames)
end

# =============================================================================
# Long-Run Restrictions (Blanchard-Quah)
# =============================================================================

"""Long-run multiplier ``C(1) = (I - \\sum A_i)^{-1}`` and the BQ rotation `Q`.

Throws [`IdentificationError`](@ref) when ``Q`` would be non-orthogonal (SID-08).
"""
function _long_run_multiplier(B::AbstractMatrix{T}, Sigma::AbstractMatrix{T},
                              n::Int, p::Int) where {T<:AbstractFloat}
    A_sum = sum(extract_ar_coefficients(B, n, p))
    M = Matrix{T}(I(n) - A_sum)
    cM = cond(M)
    C1 = robust_inv(M; silent=true)
    V_LR = C1 * Sigma * C1'
    D = Matrix(safe_cholesky(V_LR))
    @inbounds for j in 1:n
        D[j, j] < zero(T) && (@views D[:, j] .*= -one(T))
    end
    P = M * D
    Q = Matrix(safe_cholesky(Sigma) \ P)
    ortho = norm(Q' * Q - I(n))
    if ortho >= T(1e-8)
        throw(IdentificationError(
            "identify_long_run: (I − ΣAᵢ) is singular or Q is not orthogonal " *
            "(cond=$(cM), ‖Q'Q−I‖=$ortho). Use a structural VECM (identify_svec) " *
            "or difference the data."))
    end
    cM > one(T) / sqrt(eps(T)) && @warn "identify_long_run: (I − ΣAᵢ) is near-singular (cond ≈ $(cM)); the VAR is near a unit root, so the long-run impact matrix is numerically unstable." maxlog = 1
    C1, Q
end

"""
Identify via long-run restrictions (Blanchard–Quah): the long-run cumulative impact matrix is
lower triangular. Shocks are sign-normalized so each permanent shock has a non-negative long-run
effect on its own variable (a positive diagonal of the long-run cumulative impact matrix) — the
standard BQ normalization, applied explicitly so shock signs are deterministic (audit F-05; the
reference `iresponse_longrun.m` normalizes only the impact sign of the first shock).
"""
function identify_long_run(model::VARModel{T}) where {T<:AbstractFloat}
    n, p = nvars(model), model.p
    _, Q = _long_run_multiplier(model.B, model.Sigma, n, p)
    Q
end

# Identification registry (SID-19). Flags used by SID-04/05/06 and FEVD.
struct IdentificationMethod
    name::Symbol
    needs_residuals::Bool
    is_set_identified::Bool
    is_partial::Bool
end

const IDENTIFICATION_REGISTRY = Dict{Symbol,IdentificationMethod}()

register_identification!(m::IdentificationMethod) = (IDENTIFICATION_REGISTRY[m.name] = m)

function _identification_method(method::Symbol)
    get(IDENTIFICATION_REGISTRY, method) do
        throw(ArgumentError("Unknown method: $method"))
    end
end

_needs_residuals(method::Symbol) = _identification_method(method).needs_residuals
_is_set_identified(method::Symbol) = _identification_method(method).is_set_identified
_is_partial(method::Symbol) = _identification_method(method).is_partial

# Statistical-ID Q is identified only up to signed permutation. Match bootstrap /
# posterior columns to a point-estimate impact. Skip recursive/long-run and set-ID.
_should_match_columns(method::Symbol) = _needs_residuals(method) && !_is_set_identified(method)

"""
    _match_columns(P_ref, P_b) -> (perm, signs)

Signed permutation aligning impact columns of `P_b` to `P_ref`. Exhaustive for
`n ≤ 8` (maximising `Σᵢ |corr(P_ref[:,i], P_b[:,perm[i]])|`); greedy otherwise.
Signs are `sign.(diag(P_ref' P_b[:, perm]))` with zeros mapped to `+1`.
"""
function _match_columns(P_ref::AbstractMatrix{T}, P_b::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(P_ref, 2)
    size(P_b, 2) == n || throw(ArgumentError("_match_columns: column counts differ"))
    # |corr| = |⟨ref_i, b_k⟩| / (‖ref_i‖ ‖b_k‖). Copy norms first so row/col
    # scaling does not compound (the naive in-place loop would double-divide).
    ref_norms = [norm(view(P_ref, :, j)) + eps(T) for j in 1:n]
    b_norms   = [norm(view(P_b, :, j)) + eps(T) for j in 1:n]
    S = abs.(P_ref' * P_b)
    @inbounds for k in 1:n
        S[:, k] ./= b_norms[k]
    end
    @inbounds for i in 1:n
        S[i, :] ./= ref_norms[i]
    end
    best_perm = collect(1:n)
    best_score = -T(Inf)
    if n <= 8
        for perm in _permutations(n)
            sc = zero(T)
            @inbounds for i in 1:n
                sc += S[i, perm[i]]
            end
            if sc > best_score
                best_score = sc
                best_perm = collect(perm)
            end
        end
    else
        used = falses(n)
        best_perm = zeros(Int, n)
        for i in 1:n
            k = argmax(j -> used[j] ? -T(Inf) : S[i, j], 1:n)
            best_perm[i] = k
            used[k] = true
        end
    end
    aligned = P_b[:, best_perm]
    signs = [dot(view(P_ref, :, i), view(aligned, :, i)) >= 0 ? 1 : -1 for i in 1:n]
    (best_perm, signs)
end

"""Apply `_match_columns` to a rotation `Q`. Returns `(Q_aligned, relabeled)`."""
function _maybe_match_Q(Q::AbstractMatrix{T}, m::VARModel{T},
                        P_ref::Union{Nothing,AbstractMatrix{T}}) where {T<:AbstractFloat}
    P_ref === nothing && return Q, false
    P_b = safe_cholesky(m.Sigma) * Q
    perm, signs = _match_columns(P_ref, P_b)
    Qm = Q[:, perm]
    n = size(Qm, 2)
    @inbounds for j in 1:n
        Qm[:, j] .*= signs[j]
    end
    Qm, !(perm == 1:n && all(==(1), signs))
end

# =============================================================================
# Unified Interface
# =============================================================================

function _assert_orthogonal(Q::AbstractMatrix, method::Symbol)
    _is_partial(method) && return Q
    n = size(Q, 1)
    size(Q, 2) == n || return Q
    d = norm(Q' * Q - I(n))
    d < 1e-6 || throw(IdentificationError("compute_Q(:$method) returned non-orthogonal Q (‖Q'Q−I‖=$d)"))
    Q
end

"""
    compute_Q(model, method; horizon=1, restrictions=nothing, check_func=nothing,
              narrative_check=nothing, max_draws=1000, rng, ...)

Compute identification matrix Q for structural VAR analysis.

# Methods
- `:cholesky` — recursive ordering (`Q = I`; impact is `cholesky_factor(model)`)
- `:sign` — Sign restrictions (requires `check_func`)
- `:narrative` — Narrative restrictions (requires `check_func` and `narrative_check`)
- `:long_run` — Long-run restrictions (Blanchard-Quah)
- `:arias` — Arias, Rubio-Ramírez & Waggoner (2018) (requires `restrictions`)
- `:uhlig` — Mountford & Uhlig (2009) penalty function (requires `restrictions`)
- `:fastica` — FastICA (Hyvärinen 1999)
- `:jade` — JADE (Cardoso & Souloumiac 1993)
- `:sobi` — SOBI (Belouchrani et al. 1997)
- `:dcov` — Distance covariance ICA (Matteson & Tsay 2017)
- `:hsic` — HSIC independence ICA (Gretton et al. 2005)
- `:student_t` — Student-t ML (Lanne, Meitz & Saikkonen 2017)
- `:mixture_normal` — Mixture of normals ML (Lanne & Lütkepohl 2010)
- `:pml` — Pseudo-ML with Pearson Type IV (Gouriéroux, Monfort & Renne 2017)
- `:skew_normal` — Skew-normal ML (Azzalini 1985 density)
- `:nongaussian_ml` — Unified non-Gaussian ML dispatcher (default: Student-t)
- `:markov_switching` — Markov-switching heteroskedasticity (Lanne, Lütkepohl & Maciejowska 2010)
- `:garch` — GARCH-based heteroskedasticity (Normandin & Phaneuf 2004)
- `:smooth_transition` — Smooth-transition heteroskedasticity (requires `transition_var`)
- `:external_volatility` — External volatility regimes (requires `regime_indicator`)

# Keyword Arguments
- `horizon::Int=1`: IRF horizon for sign/narrative/Arias/Uhlig
- `restrictions`: `SVARRestrictions` for `:arias` / `:uhlig`
- `max_draws::Int=1000`: Maximum draws for sign/narrative/Arias
- `transition_var`: Transition variable for `:smooth_transition`
- `regime_indicator`: Regime indicator for `:external_volatility`
"""
function compute_Q(model::VARModel{T}, method::Symbol;
                   horizon::Int=1, restrictions=nothing, check_func=nothing,
                   narrative_check=nothing, instruments=nothing, pattern=nothing,
                   target=nothing, max_draws::Int=1000,
                   transition_var=nothing, regime_indicator=nothing,
                   rng::AbstractRNG=Random.default_rng(),
                   kwargs...) where {T<:AbstractFloat}
    haskey(IDENTIFICATION_REGISTRY, method) || throw(ArgumentError("Unknown method: $method"))
    n = nvars(model)
    Q = if method == :cholesky
        identify_cholesky(model)
    elseif method == :sign
        isnothing(check_func) && throw(ArgumentError("Need check_func for sign"))
        identify_sign(model, horizon, check_func; max_draws, rng)[1]
    elseif method == :narrative
        (isnothing(check_func) || isnothing(narrative_check)) &&
            throw(ArgumentError("Need check_func and narrative_check for narrative"))
        identify_narrative(model, horizon, check_func, narrative_check; max_draws, rng)[1]
    elseif method == :long_run
        identify_long_run(model)
    elseif method == :arias
        isnothing(restrictions) && throw(ArgumentError("arias requires restrictions"))
        identify_arias(model, restrictions, horizon;
                       _arias_freq_kwargs(max_draws; default_n_draws=1, rng=rng, kwargs...)...).Q_draws[1]
    elseif method == :uhlig
        isnothing(restrictions) && throw(ArgumentError("uhlig requires restrictions"))
        identify_uhlig(model, restrictions, horizon; rng=rng, kwargs...).Q
    # Non-Gaussian ICA methods (defined in nongaussian_ica.jl, loaded after this file)
    # :fastica is the only statistical-identification method that draws (random init);
    # jade/sobi/dcov/hsic + all ML/heteroskedastic methods are deterministic (#243).
    elseif method == :fastica
        identify_fastica(model; rng=rng).Q
    elseif method == :jade
        identify_jade(model).Q
    elseif method == :sobi
        identify_sobi(model).Q
    elseif method == :dcov
        identify_dcov(model).Q
    elseif method == :hsic
        identify_hsic(model).Q
    # Non-Gaussian ML methods (defined in nongaussian_ml.jl)
    elseif method == :student_t
        identify_student_t(model).Q
    elseif method == :mixture_normal
        identify_mixture_normal(model).Q
    elseif method == :pml
        identify_pml(model).Q
    elseif method == :skew_normal
        identify_skew_normal(model).Q
    elseif method == :nongaussian_ml
        identify_nongaussian_ml(model).Q
    # Heteroskedasticity methods (defined in heteroskedastic_id.jl)
    elseif method == :markov_switching
        identify_markov_switching(model).Q
    elseif method == :garch
        identify_garch(model).Q
    elseif method == :smooth_transition
        isnothing(transition_var) &&
            throw(ArgumentError("smooth_transition requires transition_var kwarg"))
        identify_smooth_transition(model, transition_var).Q
    elseif method == :external_volatility
        isnothing(regime_indicator) &&
            throw(ArgumentError("external_volatility requires regime_indicator kwarg"))
        identify_external_volatility(model, regime_indicator).Q
    else
        throw(ArgumentError("Unknown method: $method"))
    end
    _assert_orthogonal(Q, method)
end

function compute_Q(model, method, horizon, check_func, narrative_check; kwargs...)
    Base.depwarn("positional compute_Q(model, method, horizon, check_func, narrative_check) is deprecated; use keyword form", :compute_Q)
    compute_Q(model, method; horizon, check_func, narrative_check, kwargs...)
end

function _register_builtin_identification!()
    specs = (
        (:cholesky, false, false, false),
        (:long_run, false, false, false),
        (:sign, false, true, false),
        (:narrative, true, true, false),
        (:fastica, true, false, false),
        (:jade, true, false, false),
        (:sobi, true, false, false),
        (:dcov, true, false, false),
        (:hsic, true, false, false),
        (:student_t, true, false, false),
        (:mixture_normal, true, false, false),
        (:pml, true, false, false),
        (:skew_normal, true, false, false),
        (:nongaussian_ml, true, false, false),
        (:markov_switching, true, false, false),
        (:garch, true, false, false),
        (:smooth_transition, true, false, false),
        (:external_volatility, true, false, false),
        (:arias, false, true, false),
        (:uhlig, false, false, false),
    )
    for (name, needs_resid, set_id, partial) in specs
        register_identification!(IdentificationMethod(name, needs_resid, set_id, partial))
    end
    nothing
end
_register_builtin_identification!()


# Arias et al. (2018) identification — extracted to arias.jl
include("arias.jl")
