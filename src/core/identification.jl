# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Structural identification: Cholesky, sign restrictions, narrative, long-run, proxy, Arias et al. (2018).
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
    ProxySVARResult{T} <: AbstractAnalysisResult

Proxy / external-instrument SVAR identification (Mertens & Ravn 2013; Stock &
Watson 2018). The first `k` columns of `Q` (after `shocks` placement) are
identified; remaining columns are an orthogonal complement (`is_partial=true`
when `k < n`).
"""
struct ProxySVARResult{T<:AbstractFloat} <: AbstractAnalysisResult
    Q::Matrix{T}
    B0::Matrix{T}
    k::Int
    first_stage_F::T
    reliability::T
    instruments_names::Vector{String}
    varnames::Vector{String}
    shock_names::Vector{String}
    is_partial::Bool
end

function Base.show(io::IO, r::ProxySVARResult{T}) where {T}
    n = size(r.B0, 1)
    spec = Any[
        "Variables"        n;
        "Instruments (k)"  r.k;
        "Partial"          r.is_partial ? "Yes" : "No";
        "First-stage F"    _fmt(r.first_stage_F; digits=2);
        "Reliability"      _fmt(r.reliability; digits=3);
        "Instruments"      join(r.instruments_names, ", ")
    ]
    _pretty_table(io, spec;
        title = "Proxy SVAR Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _matrix_table(io, r.B0, "Structural Impact Matrix (B₀)";
        row_labels=r.varnames,
        col_labels=r.shock_names)
    r.first_stage_F < T(10) && println(io,
        "Weak instrument: first-stage F < 10 (Montiel Olea, Stock & Watson 2021).")
    r.is_partial && println(io,
        "Partial identification: FEVD/HD of unidentified shocks are not identified.")
end

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

"""
    identify_proxy(model::VARModel, Z::AbstractMatrix; shocks=1:k,
                   normalize=:unit_effect, align=true) -> ProxySVARResult

Proxy SVAR with `k = size(Z, 2)` instruments for `k` shocks (Mertens & Ravn 2013,
Appendix A). `k = 1` is the closed form; `k > 1` identifies the column space of
the instrumented impacts up to a `k × k` rotation. Missing/`NaN` rows are dropped
pairwise when `align=true`.

`normalize = :unit_effect` scales identified column `j` so variable
`normalize_var` (k=1) or variable `j` (k>1) responds by one on impact.
`:unit_variance` returns an orthogonal `Q` with `B₀ = chol(Σ) Q`.
"""
function identify_proxy(model::VARModel{T}, Z::AbstractMatrix;
                        shocks=1:size(Z, 2),
                        normalize::Symbol=:unit_effect,
                        align::Bool=true,
                        normalize_var::Int=1,
                        instrument_names::Union{Nothing,Vector{String}}=nothing,
                        shock_names::Union{Nothing,Vector{String}}=nothing,
                        kwargs...) where {T<:AbstractFloat}
    n = nvars(model)
    k = size(Z, 2)
    k < 1 && throw(ArgumentError("need at least one instrument"))
    k > n && throw(ArgumentError("k=$k instruments exceeds n=$n variables"))
    normalize in (:unit_effect, :unit_variance) || throw(ArgumentError(
        "normalize must be :unit_effect or :unit_variance, got :$normalize"))
    sh = collect(Int, shocks)
    length(sh) == k || throw(ArgumentError("shocks must have length k=$k"))
    allunique(sh) || throw(ArgumentError("shocks must be unique"))
    all(1 .<= sh .<= n) || throw(ArgumentError("shocks must be in 1:$n"))
    (1 <= normalize_var <= n) || throw(ArgumentError("normalize_var must be in 1:$n"))

    Uu, ZZ, _, nobs = _proxy_complete_cases(model, Z; align=align)
    Uc = Uu .- mean(Uu; dims=1)
    Zc = ZZ .- mean(ZZ; dims=1)
    df = T(nobs - 1)
    Suz = Matrix{T}((Uc' * Zc) / df)
    Sigmaz = Matrix{T}((Zc' * Zc) / df)
    P = safe_cholesky(model.Sigma)

    if normalize === :unit_variance
        Qid = _complete_proxy_Q(P, Suz, n, T)
    elseif k == 1
        denom = Suz[normalize_var, 1]
        abs(denom) < T(1e-12) && throw(ArgumentError(
            "instrument is uncorrelated with residual $normalize_var; cannot normalise"))
        b1 = vec(Suz) ./ denom
        Qid = _complete_proxy_Q(P, b1, n, T)
    else
        S1 = Suz[1:k, :]
        S1inv = Matrix{T}(robust_inv(S1))
        srel = Suz[(k + 1):n, :] * S1inv
        R = vcat(Matrix{T}(I, k, k), srel)
        Qid = _complete_proxy_Q(P, R, n, T)
    end

    Btmp = P * Qid
    if normalize === :unit_effect
        for j in 1:k
            i = k == 1 ? normalize_var : j
            sc = Btmp[i, j]
            abs(sc) < T(1e-12) && throw(ArgumentError(
                "proxy impact on the normalising variable is zero"))
            Qid[:, j] .*= one(T) / sc
        end
    else
        for j in 1:k
            i = k == 1 ? normalize_var : j
            Btmp[i, j] < 0 && (Qid[:, j] .*= -one(T))
        end
    end

    Q = _place_proxy_columns(Qid, sh, n, k)
    B0 = P * Q

    iF = k == 1 ? normalize_var : 1
    Fstat = _proxy_first_stage_F(Uu, ZZ, iF)
    if Fstat < T(10)
        @warn "Weak instrument: first-stage F = $(round(Fstat, digits=2)) < 10 (Montiel Olea, Stock & Watson 2021)"
    end
    rel = _proxy_reliability(Suz, model.Sigma, Sigmaz)

    inames = something(instrument_names, ["z$i" for i in 1:k])
    length(inames) == k || throw(ArgumentError("instrument_names must have length k=$k"))
    snames = something(shock_names, _default_proxy_shock_names(n, k, sh))
    length(snames) == n || throw(ArgumentError("shock_names must have length n=$n"))

    ProxySVARResult{T}(Matrix{T}(Q), Matrix{T}(B0), k, T(Fstat), T(rel),
                       Vector{String}(inames), copy(model.varnames),
                       Vector{String}(snames), k < n)
end

function _default_proxy_shock_names(n::Int, k::Int, sh::Vector{Int})
    sn = Vector{String}(undef, n)
    fill!(sn, "")
    for j in 1:k
        sn[sh[j]] = k == 1 ? "Proxy" : "Proxy $j"
    end
    u = 1
    for i in 1:n
        if isempty(sn[i])
            sn[i] = "Unidentified $u"
            u += 1
        end
    end
    sn
end

function _proxy_complete_cases(model::VARModel{T}, Z::AbstractMatrix;
                               align::Bool=true) where {T<:AbstractFloat}
    U = model.U
    T_eff = size(U, 1)
    T_obs = size(model.Y, 1)
    p = model.p
    Z_eff = if align
        _align_instrument(Z, T_obs, p, T_eff)
    else
        size(Z, 1) == T_eff || throw(ArgumentError(
            "align=false requires instruments with $T_eff rows (residual sample); got $(size(Z, 1))"))
        float.(Z)
    end
    Z_eff = Matrix{T}(Z_eff)
    mask = [all(isfinite, @view Z_eff[t, :]) && all(isfinite, @view U[t, :]) for t in 1:T_eff]
    count(mask) < 8 && throw(ArgumentError("instrument has too few finite observations"))
    Uu = U[mask, :]
    ZZ = Z_eff[mask, :]
    Uu, ZZ, Z_eff, size(ZZ, 1)
end

function _proxy_first_stage_F(Uc::AbstractMatrix{T}, Zc::AbstractMatrix{T},
                              i_norm::Int) where {T<:AbstractFloat}
    y = Vector{T}(@view Uc[:, i_norm])
    controls = Matrix{T}(undef, length(y), 0)
    first_stage_regression(y, Zc, controls).F_stat
end

function _proxy_reliability(Suz::AbstractMatrix{T}, Sigma::AbstractMatrix{T},
                            Sigmaz::AbstractMatrix{T}) where {T<:AbstractFloat}
    Sinv = Matrix{T}(robust_inv(Hermitian(Matrix{T}(Sigma))))
    G = Suz' * Sinv * Suz
    Zinv = Matrix{T}(robust_inv(Hermitian(Matrix{T}(Sigmaz))))
    Rmat = G * Zinv
    k = size(Rmat, 1)
    val = tr(Rmat) / T(k)
    T(clamp(real(val), zero(T), one(T)))
end

function _place_proxy_columns(Q::AbstractMatrix{T}, shocks::Vector{Int},
                              n::Int, k::Int) where {T}
    shocks == collect(1:k) && return Q
    Qout = similar(Q)
    used = falses(n)
    for j in 1:k
        Qout[:, shocks[j]] = Q[:, j]
        used[shocks[j]] = true
    end
    c = k + 1
    for i in 1:n
        if !used[i]
            Qout[:, i] = Q[:, c]
            c += 1
        end
    end
    Qout
end

function _align_instrument(z::AbstractVector, T_obs::Int, p::Int, T_eff::Int)
    n = length(z)
    n == T_eff && return float.(z)
    n == T_obs && return float.(z[(p + 1):end])
    throw(ArgumentError(
        "instrument length $n must equal residual sample $T_eff or full sample $T_obs"))
end

function _align_instrument(Z::AbstractMatrix, T_obs::Int, p::Int, T_eff::Int)
    nrows = size(Z, 1)
    nrows == T_eff && return float.(Z)
    nrows == T_obs && return float.(Z[(p + 1):end, :])
    throw(ArgumentError(
        "instrument rows $nrows must equal residual sample $T_eff or full sample $T_obs"))
end

"""
    proxy_ar_band(model, z; horizon=0, normalize_var=1, level=0.95, ...) -> LPIVARBand

Weak-instrument-robust Anderson–Rubin bands for a `k = 1` proxy SVAR (Montiel
Olea, Stock & Watson 2021). Builds the observationally equivalent LP-IV
(Plagborg-Møller & Wolf 2021) with the VAR lag length as controls and reuses
[`lp_iv_ar_band`](@ref).
"""
function proxy_ar_band(model::VARModel{T}, z;
                       horizon::Int=0, normalize_var::Int=1, level::Real=0.95,
                       n_grid::Int=401, span::Real=20, bandwidth::Int=0,
                       responses=nothing) where {T<:AbstractFloat}
    Z = z isa AbstractVector ? reshape(collect(float.(z)), :, 1) : Matrix{T}(float.(z))
    size(Z, 2) == 1 || throw(ArgumentError("Anderson-Rubin bands require k=1 instrument"))
    Y = model.Y
    T_obs = size(Y, 1)
    T_eff = T_obs - model.p
    Z_eff = _align_instrument(Z, T_obs, model.p, T_eff)
    any(!isfinite, Z_eff) && throw(ArgumentError(
        "proxy_ar_band requires a fully observed instrument; drop missing rows first"))
    any(!isfinite, Y) && throw(ArgumentError("proxy_ar_band requires fully observed Y"))
    (1 <= normalize_var <= nvars(model)) || throw(ArgumentError(
        "normalize_var must be in 1:$(nvars(model))"))
    # LP-IV needs length T; leading p rows are unused (`t_start = lags + 1`).
    Z_full = vcat(fill(T(NaN), model.p, 1), Matrix{T}(Z_eff))
    lp = estimate_lp_iv(Y, normalize_var, Z_full, horizon; lags=model.p, varnames=model.varnames)
    lp_iv_ar_band(lp; level=level, n_grid=n_grid, span=span, bandwidth=bandwidth,
                  responses=responses)
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

function _complete_proxy_Q(P::AbstractMatrix{T}, Bcols::AbstractMatrix{T}, n::Int, ::Type{T}) where {T}
    k = size(Bcols, 2)
    k == 1 && return _complete_proxy_Q(P, vec(Bcols), n, T)
    size(Bcols, 1) == n || throw(ArgumentError("proxy impact has incompatible size"))
    (1 <= k <= n) || throw(ArgumentError("proxy impact has incompatible size"))
    Qk = P \ Bcols
    Fk = qr(Qk)
    Qon = Matrix{T}(Fk.Q)
    sk = [i <= k && Fk.R[i, i] < zero(T) ? -one(T) : one(T) for i in 1:k]
    M = Matrix{T}(I, n, n)
    M[:, 1:k] = Qon[:, 1:k] * Diagonal(sk)
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

"""
    compute_structural_shocks(model, Q) -> Matrix

Recover structural shocks from reduced-form residuals `u`. When `Q` is orthogonal
(`Q'Q = I`; Cholesky, sign, Arias) this is `ε = Q' L⁻¹ u` via a triangular
backsolve. When `Q` is not orthogonal (proxy unit-effect scaling of identified
columns) it is `ε = (L Q)⁻¹ u`, i.e. `B₀ \\ u`.
"""
function compute_structural_shocks(model::VARModel{T}, Q::AbstractMatrix{T}) where {T<:AbstractFloat}
    L = safe_cholesky(model.Sigma)
    U = model.U
    if _q_is_orthogonal(Q)
        (Q' * (L \ U'))'                      # orthogonal: Q' L⁻¹ u
    else
        ((L * Q) \ U')'                       # B₀ \ u (unit-effect proxy)
    end
end

"""True when `Q` is square and `‖Q'Q − I‖ < 1e-6` (same tolerance as `_assert_orthogonal`)."""
function _q_is_orthogonal(Q::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(Q, 1)
    size(Q, 2) == n || return false
    norm(Q' * Q - I(n)) < T(1e-6)
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
- `:proxy` — External instruments (requires `instruments`)

# Keyword Arguments
- `horizon::Int=1`: IRF horizon for sign/narrative/Arias/Uhlig
- `restrictions`: `SVARRestrictions` for `:arias` / `:uhlig`
- `max_draws::Int=1000`: Maximum draws for sign/narrative/Arias
- `transition_var`: Transition variable for `:smooth_transition`
- `regime_indicator`: Regime indicator for `:external_volatility`
- `instruments`: Instrument vector/matrix for `:proxy`
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
    elseif method == :proxy
        isnothing(instruments) &&
            throw(ArgumentError("proxy requires instruments"))
        Z = instruments isa AbstractVector ? reshape(instruments, :, 1) : instruments
        identify_proxy(model, Z; kwargs...).Q
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
        (:proxy, true, false, true),
    )
    for (name, needs_resid, set_id, partial) in specs
        register_identification!(IdentificationMethod(name, needs_resid, set_id, partial))
    end
    nothing
end
_register_builtin_identification!()


# Arias et al. (2018) identification — extracted to arias.jl
include("arias.jl")
