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
function generate_Q(n::Int, ::Type{T}=Float64; rng::AbstractRNG=Random.default_rng(),
                    seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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
                       seed::Union{Integer,Nothing}=nothing,
                       rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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

"""Pointwise IRF quantiles from a draw stack. Uniform weights use `quantile` so
the result matches the pre-SID-17 unweighted identified set; otherwise Kish-weighted."""
function _irf_percentiles_from_draws(irf_draws::Array{T,4}, weights::AbstractVector,
                                     quantiles::AbstractVector;
                                     uniform_unweighted::Bool=true) where {T<:AbstractFloat}
    n_draws, H, nv, ns = size(irf_draws)
    length(weights) == n_draws || throw(ArgumentError(
        "weights length ($(length(weights))) must match n_draws ($n_draws)"))
    pct = zeros(T, H, nv, ns, length(quantiles))
    use_unw = uniform_unweighted && _weights_are_uniform(weights)
    @inbounds for h in 1:H, i in 1:nv, j in 1:ns
        vals = @view irf_draws[:, h, i, j]
        for (pi, p) in enumerate(quantiles)
            pct[h, i, j, pi] = use_unw ? quantile(vals, T(p)) : _weighted_quantile(vals, weights, p)
        end
    end
    pct
end

"""Cell quantile of an identified-set draw vector (unweighted `quantile` if weights are uniform)."""
function _setid_quantile(vals::AbstractVector{T}, weights::AbstractVector, q::Real;
                         uniform_unweighted::Bool=true) where {T<:AbstractFloat}
    (uniform_unweighted && _weights_are_uniform(weights)) ? quantile(vals, T(q)) :
        _weighted_quantile(vals, weights, q)
end

"""Identified-set standard deviation per cell. Uniform weights use `Statistics.std`."""
function _setid_std(vals::AbstractVector{T}, weights::AbstractVector;
                    uniform_unweighted::Bool=true) where {T<:AbstractFloat}
    n = length(vals)
    n <= 1 && return zero(T)
    if uniform_unweighted && _weights_are_uniform(weights)
        return std(vals)
    end
    s = sum(weights)
    s <= 0 && return zero(T)
    μ = zero(T)
    @inbounds for i in eachindex(vals)
        μ += T(weights[i]) * vals[i]
    end
    μ /= T(s)
    v = zero(T)
    @inbounds for i in eachindex(vals)
        δ = vals[i] - μ
        v += T(weights[i]) * δ * δ
    end
    sqrt(v / T(s))
end

"""
    irf_percentiles(s::SignIdentifiedSet; quantiles=[0.16, 0.5, 0.84]) -> Array{T,4}

Pointwise IRF quantiles over the identified set (horizon × n × n × nq).
Uniform weights reproduce `Statistics.quantile`; non-uniform weights use
the shared weighted-quantile helper.
"""
function irf_percentiles(s::SignIdentifiedSet{T}; quantiles::Vector{Float64}=[0.16, 0.5, 0.84]) where {T}
    _irf_percentiles_from_draws(s.irf_draws, s.weights, quantiles; uniform_unweighted=true)
end

"""
    irf_bounds(s::SignIdentifiedSet{T}; quantiles=[0.16, 0.84]) -> (lower, upper)

Compute pointwise bounds (or quantile bands) over the identified set.
"""
function irf_bounds(s::SignIdentifiedSet{T}; quantiles=[0.16, 0.84]) where {T}
    pct = irf_percentiles(s; quantiles=Float64[quantiles[1], quantiles[2]])
    (pct[:, :, :, 1], pct[:, :, :, 2])
end

"""
    irf_median(s::SignIdentifiedSet{T}) -> Array{T,3}

Compute pointwise median IRF over the identified set.
"""
function irf_median(s::SignIdentifiedSet{T}) where {T}
    irf_percentiles(s; quantiles=[0.5])[:, :, :, 1]
end

# =============================================================================
# Narrative Restrictions
# =============================================================================

"""
    identify_narrative(model, horizon, sign_check, narrative_check; max_draws=1000, store_all=false)

Combine sign and narrative restrictions via closures (set-aware `compute_Q(:narrative)`).

With `store_all=false` (default), returns `(Q, irf, shocks)` — the first valid rotation.
With `store_all=true`, returns a `SignIdentifiedSet` of every accepted rotation (same shape
as [`identify_sign`](@ref)).

Typed ADRR restrictions belong on [`identify_arias`](@ref) / the
`identify_narrative(model, restrictions::SVARRestrictions, horizon)` wrapper.
"""
function identify_narrative(model::VARModel{T}, horizon::Int, sign_check::Function,
                            narrative_check::Function; max_draws::Int=1000,
                            store_all::Bool=false,
                            shock_names::Union{Nothing,Vector{String}}=nothing,
                            seed::Union{Integer,Nothing}=nothing,
                            rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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
"""
    IdentificationMethod(name, needs_residuals, is_set_identified, is_partial)

Registry entry for a `compute_Q` / `irf` identification scheme. `needs_residuals`
is true when the kernel reads `model.U`; `is_set_identified` marks sign / narrative
/ Arias; `is_partial` marks proxy and max-share (some columns unidentified).
"""
struct IdentificationMethod
    name::Symbol
    needs_residuals::Bool
    is_set_identified::Bool
    is_partial::Bool
end

const IDENTIFICATION_REGISTRY = Dict{Symbol,IdentificationMethod}()

"""
    register_identification!(m::IdentificationMethod)

Insert `m` into `IDENTIFICATION_REGISTRY`. Built-in schemes are registered at
load time; user schemes use the same hook.
"""
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
# posterior columns to a point-estimate impact. Skip recursive/long-run, set-ID,
# :proxy, and :max_share (identified columns are already labeled).
_should_match_columns(method::Symbol) =
    _needs_residuals(method) && !_is_set_identified(method) &&
    method !== :proxy && method !== :max_share

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
    # Overidentified AB-models have Σ_r ≠ Σ, so Q = L^{-1} A^{-1} B need not
    # be orthogonal. IRFs still use B₀ = L Q = A^{-1} B.
    (_is_partial(method) || method === :ab || method === :svec) && return Q
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
- `:ab` — AB-model ML (requires `pattern::SVARPattern`)
- `:max_share` — Max-share / news-shock (requires `target`; `horizons` or `band`)
- `:svec` — structural VECM (KPSW); pass `_svec_Q` or call `irf(vecm; method=:svec)`

# Keyword Arguments
- `horizon::Int=1`: IRF horizon for sign/narrative/Arias/Uhlig
- `restrictions`: `SVARRestrictions` for `:arias` / `:uhlig`
- `max_draws::Int=1000`: Maximum draws for sign/narrative/Arias
- `transition_var`: Transition variable for `:smooth_transition`
- `regime_indicator`: Regime indicator for `:external_volatility`
- `instruments`: Instrument vector/matrix for `:proxy`
- `pattern`: `SVARPattern` for `:ab`
- `target`: Variable index or name for `:max_share`
- `horizons`: Time-domain window for `:max_share` (default `0:20`)
- `band`: Frequency band `(ω₁, ω₂)` for `:max_share`
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
    elseif method == :gmm_moments
        identify_gmm_moments(model; kwargs...).Q
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
    elseif method == :ab
        isnothing(pattern) && throw(ArgumentError("ab requires pattern"))
        estimate_svar(model, pattern; rng=rng, kwargs...).Q
    elseif method == :max_share
        isnothing(target) && throw(ArgumentError("max_share requires target"))
        identify_max_share(model; target=target, kwargs...).Q
    elseif method == :svec
        Qs = get(kwargs, :_svec_Q, nothing)
        Qs === nothing && throw(ArgumentError(
            "method=:svec requires a VECMModel; use irf(vecm; method=:svec) or identify_svec"))
        Matrix{T}(Qs)
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
        (:gmm_moments, true, false, false),
        (:markov_switching, true, false, false),
        (:garch, true, false, false),
        (:smooth_transition, true, false, false),
        (:external_volatility, true, false, false),
        (:arias, false, true, false),
        (:uhlig, false, false, false),
        (:proxy, true, false, true),
        (:ab, false, false, false),
        (:max_share, false, false, true),
        (:svec, false, false, false),
    )
    for (name, needs_resid, set_id, partial) in specs
        register_identification!(IdentificationMethod(name, needs_resid, set_id, partial))
    end
    nothing
end
_register_builtin_identification!()


# Arias et al. (2018) identification — extracted to arias.jl
include("arias.jl")
include("robust_bayes.jl")   # SID-18 Giacomini–Kitagawa robust Bayes

# =============================================================================
# Set-identification summaries (Fry–Pagan / Inoue–Kilian; SID-17)
# =============================================================================

const _SetIdentifiedSVAR{T} = Union{SignIdentifiedSet{T}, AriasSVARResult{T}}

function _setid_uniform_unweighted(::SignIdentifiedSet)
    true
end
_setid_uniform_unweighted(::AriasSVARResult) = false

function _median_irf(s::_SetIdentifiedSVAR{T}) where {T<:AbstractFloat}
    _irf_percentiles_from_draws(s.irf_draws, s.weights, [0.5];
                                uniform_unweighted=_setid_uniform_unweighted(s))[:, :, :, 1]
end

"""Fry–Pagan (2011) standardised L2 distance of each draw to the pointwise median."""
function _median_target_distances(irf_draws::Array{T,4}, med::Array{T,3}, σ::Array{T,3}) where {T<:AbstractFloat}
    n_d = size(irf_draws, 1)
    H, nv, ns = size(med)
    dists = zeros(T, n_d)
    Threads.@threads for d in 1:n_d
        acc = zero(T)
        @inbounds for h in 1:H, i in 1:nv, j in 1:ns
            s = σ[h, i, j]
            δ = irf_draws[d, h, i, j] - med[h, i, j]
            acc += s > 0 ? (δ / s)^2 : δ * δ
        end
        dists[d] = acc
    end
    dists
end

"""
    median_target(s) -> (Q, irf, index)

Fry–Pagan (2011) median-target rotation: the single admissible `Q` in `s.Q_draws`
whose IRF is closest to the pointwise median in standardised Euclidean distance
across `(h, i, j)`. Location and scale are weighted when `s.weights` are
non-uniform.
"""
function median_target(s::_SetIdentifiedSVAR{T}) where {T<:AbstractFloat}
    n_d = size(s.irf_draws, 1)
    n_d == 0 && throw(ArgumentError("identified set has no draws"))
    uw = _setid_uniform_unweighted(s)
    med = _median_irf(s)
    n_d == 1 && return (Q=s.Q_draws[1], irf=s.irf_draws[1, :, :, :], index=1)
    H, nv, ns = size(med)
    σ = similar(med)
    @inbounds for h in 1:H, i in 1:nv, j in 1:ns
        σ[h, i, j] = _setid_std(view(s.irf_draws, :, h, i, j), s.weights; uniform_unweighted=uw)
    end
    idx = argmin(_median_target_distances(s.irf_draws, med, σ))
    (Q=s.Q_draws[idx], irf=s.irf_draws[idx, :, :, :], index=idx)
end

function _default_kde_bandwidth(X::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(X, 1)
    n <= 1 && return one(T)
    n_s = min(n, 32)
    acc = zero(T)
    cnt = 0
    @inbounds for i in 1:(n_s - 1), j in (i + 1):n_s
        d2 = zero(T)
        for p in axes(X, 2)
            δ = X[i, p] - X[j, p]
            d2 += δ * δ
        end
        acc += sqrt(d2)
        cnt += 1
    end
    med = cnt > 0 ? acc / T(cnt) : one(T)
    med > 0 ? med : one(T)
end

"""Inoue–Kilian kernel density evaluated at each draw (Gaussian kernel, weighted)."""
function _kde_at_draws(X::AbstractMatrix{T}, weights::AbstractVector, h::T) where {T<:AbstractFloat}
    n = size(X, 1)
    dens = zeros(T, n)
    inv2h2 = one(T) / (T(2) * h * h)
    Threads.@threads for i in 1:n
        s = zero(T)
        @inbounds for j in 1:n
            d2 = zero(T)
            for p in axes(X, 2)
                δ = X[i, p] - X[j, p]
                d2 += δ * δ
            end
            s += T(weights[j]) * exp(-d2 * inv2h2)
        end
        dens[i] = s
    end
    dens
end

"""
    modal_model(s; bandwidth=nothing) -> (Q, irf, index)

Inoue–Kilian (2013) modal model: the stored draw at which a Gaussian KDE over
vectorised, cell-standardised IRFs attains its maximum (weighted by `s.weights`).
Default `bandwidth` is the mean pairwise Euclidean distance on a 32-draw subsample.
"""
function modal_model(s::_SetIdentifiedSVAR{T}; bandwidth::Union{Nothing,Real}=nothing) where {T<:AbstractFloat}
    n_d = size(s.irf_draws, 1)
    n_d == 0 && throw(ArgumentError("identified set has no draws"))
    n_d == 1 && return (Q=s.Q_draws[1], irf=s.irf_draws[1, :, :, :], index=1)
    H, nv, ns = size(s.irf_draws, 2), size(s.irf_draws, 3), size(s.irf_draws, 4)
    k = H * nv * ns
    X = Matrix{T}(undef, n_d, k)
    @inbounds for d in 1:n_d
        X[d, :] = vec(s.irf_draws[d, :, :, :])
    end
    uw = _setid_uniform_unweighted(s)
    @inbounds for p in 1:k
        σp = _setid_std(view(X, :, p), s.weights; uniform_unweighted=uw)
        if σp > 0
            X[:, p] ./= σp
        end
    end
    h = bandwidth === nothing ? _default_kde_bandwidth(X) : T(bandwidth)
    h > 0 || throw(ArgumentError("bandwidth must be positive, got $h"))
    idx = argmax(_kde_at_draws(X, s.weights, h))
    (Q=s.Q_draws[idx], irf=s.irf_draws[idx, :, :, :], index=idx)
end

"""
    joint_band(s; level=0.68, loss=:absolute) -> (lower, upper)

Inoue–Kilian (2022) joint credible set: the componentwise envelope of the
lowest-loss draws whose (positive) weights sum to at least `level`, always
including the median-target IRF. Draws with non-positive weight are skipped
when accumulating the HPD set. A draw is inside the band iff every `(h, i, j)`
entry lies in `[lower, upper]`. `loss=:absolute` is the L1 distance to the
pointwise median.
"""
function joint_band(s::_SetIdentifiedSVAR{T}; level::Real=0.68,
                    loss::Symbol=:absolute) where {T<:AbstractFloat}
    loss === :absolute || throw(ArgumentError("loss must be :absolute, got :$loss"))
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1), got $level"))
    n_d = size(s.irf_draws, 1)
    n_d == 0 && throw(ArgumentError("identified set has no draws"))
    med = _median_irf(s)
    H, nv, ns = size(med)
    n_d == 1 && return (s.irf_draws[1, :, :, :], s.irf_draws[1, :, :, :])
    losses = zeros(T, n_d)
    Threads.@threads for d in 1:n_d
        acc = zero(T)
        @inbounds for h in 1:H, i in 1:nv, j in 1:ns
            acc += abs(s.irf_draws[d, h, i, j] - med[h, i, j])
        end
        losses[d] = acc
    end
    mt = median_target(s)
    order = sortperm(losses)
    wsum = sum(s.weights)
    target = T(level) * T(wsum)
    kept = falses(n_d)
    cum = zero(T)
    @inbounds for idx in order
        kept[idx] && continue
        w = T(s.weights[idx])
        w <= 0 && continue
        kept[idx] = true
        cum += w
        cum >= target && break
    end
    kept[mt.index] = true
    lower = fill(T(Inf), H, nv, ns)
    upper = fill(T(-Inf), H, nv, ns)
    @inbounds for d in 1:n_d
        kept[d] || continue
        for h in 1:H, i in 1:nv, j in 1:ns
            v = s.irf_draws[d, h, i, j]
            lower[h, i, j] = min(lower[h, i, j], v)
            upper[h, i, j] = max(upper[h, i, j], v)
        end
    end
    (lower, upper)
end

"""
    sup_t_band(s; level=0.68) -> (lower, upper)

Montiel Olea–Plagborg-Møller (2019) sup-t simultaneous band for the identified
set: `median ± c · σ`, where `c` is the `level` quantile of
`max_{h,i,j} |IRF − median| / σ` across draws.
"""
function sup_t_band(s::_SetIdentifiedSVAR{T}; level::Real=0.68) where {T<:AbstractFloat}
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1), got $level"))
    n_d = size(s.irf_draws, 1)
    n_d == 0 && throw(ArgumentError("identified set has no draws"))
    med = _median_irf(s)
    n_d == 1 && return (s.irf_draws[1, :, :, :], s.irf_draws[1, :, :, :])
    uw = _setid_uniform_unweighted(s)
    H, nv, ns = size(med)
    σ = similar(med)
    @inbounds for h in 1:H, i in 1:nv, j in 1:ns
        σ[h, i, j] = _setid_std(view(s.irf_draws, :, h, i, j), s.weights; uniform_unweighted=uw)
    end
    supt = zeros(T, n_d)
    Threads.@threads for d in 1:n_d
        m = zero(T)
        @inbounds for h in 1:H, i in 1:nv, j in 1:ns
            s_ij = σ[h, i, j]
            t = s_ij > 0 ? abs(s.irf_draws[d, h, i, j] - med[h, i, j]) / s_ij : zero(T)
            m = max(m, t)
        end
        supt[d] = m
    end
    c = _setid_quantile(supt, s.weights, level; uniform_unweighted=uw)
    (med .- c .* σ, med .+ c .* σ)
end

function _setid_irfs_at_horizon(model::VARModel{T}, Q_draws, irf_draws::Array{T,4},
                                H::Int) where {T<:AbstractFloat}
    H >= 1 || throw(ArgumentError("horizon must be ≥ 1, got $H"))
    H_stored = size(irf_draws, 2)
    n_acc = length(Q_draws)
    n = nvars(model)
    if H == H_stored && size(irf_draws, 1) == n_acc && size(irf_draws, 3) == n
        return irf_draws
    elseif H < H_stored && size(irf_draws, 1) == n_acc && size(irf_draws, 3) == n
        return irf_draws[:, 1:H, :, :]
    end
    draws = zeros(T, n_acc, H, n, n)
    Threads.@threads for i in 1:n_acc
        draws[i, :, :, :] = compute_irf(model, Q_draws[i], H)
    end
    draws
end

"""Weighted-quantile summary of structural shocks across identified-set draws."""
function _structural_shocks_from_Qs(model::VARModel{T}, Q_draws, weights;
                                    quantiles::Vector{Float64}=[0.16, 0.5, 0.84],
                                    uniform_unweighted::Bool=true) where {T<:AbstractFloat}
    n_d = length(Q_draws)
    n_d == 0 && throw(ArgumentError("identified set has no draws"))
    s1 = compute_structural_shocks(model, Q_draws[1])
    T_eff, n = size(s1)
    all_shocks = zeros(T, n_d, T_eff, n)
    all_shocks[1, :, :] = s1
    Threads.@threads for i in 2:n_d
        all_shocks[i, :, :] = compute_structural_shocks(model, Q_draws[i])
    end
    qv = T.(quantiles)
    nq = length(qv)
    qarr = zeros(T, T_eff, n, nq)
    @inbounds for t in 1:T_eff, j in 1:n
        vals = @view all_shocks[:, t, j]
        for (qi, q) in enumerate(qv)
            qarr[t, j, qi] = _setid_quantile(vals, weights, q; uniform_unweighted=uniform_unweighted)
        end
    end
    imid = findfirst(q -> q == T(0.5) || q == 0.5, qv)
    med = imid === nothing ? [_setid_quantile(view(all_shocks, :, t, j), weights, T(0.5);
                                              uniform_unweighted=uniform_unweighted)
                              for t in 1:T_eff, j in 1:n] : qarr[:, :, imid]
    (median=med, lower=qarr[:, :, 1], upper=qarr[:, :, nq], quantiles=qarr)
end

"""
    structural_shocks(model, s::SignIdentifiedSet; quantiles=[0.16, 0.5, 0.84])

Weighted-quantile summary of structural shocks recovered at each stored rotation.
Returns `(median, lower, upper, quantiles)`.
"""
function structural_shocks(model::VARModel{T}, s::SignIdentifiedSet{T};
                           quantiles::Vector{Float64}=[0.16, 0.5, 0.84]) where {T<:AbstractFloat}
    _structural_shocks_from_Qs(model, s.Q_draws, s.weights; quantiles=quantiles,
                               uniform_unweighted=true)
end

function structural_shocks(model::VARModel{T}, s::AriasSVARResult{T};
                           quantiles::Vector{Float64}=[0.16, 0.5, 0.84]) where {T<:AbstractFloat}
    _structural_shocks_from_Qs(model, s.Q_draws, s.weights; quantiles=quantiles,
                               uniform_unweighted=false)
end

"""Point structural shocks `ε = B₀ \\ u` from a rotation `Q` (`B₀ = L Q`)."""
structural_shocks(model::VARModel, Q::AbstractMatrix) = compute_structural_shocks(model, Q)

"""
    structural_shocks(result)

Stored structural shocks of a statistical identification result. Types without a
`shocks` series (Markov-switching, external volatility) throw; recover them with
`structural_shocks(model, result.Q)`.
"""
function structural_shocks(result::AbstractNonGaussianSVAR)
    if hasfield(typeof(result), :shocks)
        sh = getfield(result, :shocks)
        !isempty(sh) && return sh
    end
    if hasfield(typeof(result), :residuals)
        U = getfield(result, :residuals)
        if !isempty(U)
            return (result.B0 \ U')'
        end
    end
    throw(ArgumentError(
        "structural_shocks(result): $(typeof(result).name.name) does not store shocks; " *
        "use structural_shocks(model, result.Q)"))
end

_default_shock_names(n::Int) = ["Shock $j" for j in 1:n]

# =============================================================================
# SID-20: shock labels (signed permutation of B₀)
# =============================================================================

"""
    label_shocks(result; by=:restrictions, restrictions=nothing, B_ref=nothing,
                 variables=nothing, shock_names=nothing,
                 sign_convention=:positive_diagonal) -> typeof(result)

Relabel statistically identified shocks by a signed permutation of the columns
of `B₀` (and of `Q`, stored shocks, and column-tied `Λ`).

- `by=:restrictions` — maximise satisfied impact-sign restrictions (`n ≤ 8`
  exhaustive; greedy otherwise). `restrictions` is a sign matrix (`±1`/`0`),
  `SVARRestrictions`, or a vector of `SignRestriction`s (horizon 0).
- `by=:max_impact` — column `j` gets the shock with largest `|B₀[var_j, ·]|`
  divided by the column norm. `variables` defaults to `1:n`.
- `by=:reference` — `_match_columns(B_ref, result.B0)`.

`sign_convention=:positive_diagonal` (default) makes `B₀[j,j] > 0`;
`:unit_effect` makes the impact on `variables[j]` (or variable `j`) positive.
Restriction labelling keeps the signs that maximise the score.
`by=:reference` keeps `_match_columns` signs (the convention is not applied).
"""
function label_shocks(result::AbstractNonGaussianSVAR;
                      by::Symbol=:restrictions,
                      restrictions=nothing,
                      B_ref=nothing,
                      variables=nothing,
                      shock_names::Union{Nothing,Vector{String}}=nothing,
                      sign_convention::Symbol=:positive_diagonal)
    B0 = result.B0
    n = size(B0, 2)
    size(B0, 1) == n || throw(ArgumentError("label_shocks: B₀ must be square"))
    by in (:restrictions, :max_impact, :reference) || throw(ArgumentError(
        "by must be :restrictions, :max_impact, or :reference, got :$by"))
    sign_convention in (:positive_diagonal, :unit_effect, :none) || throw(ArgumentError(
        "sign_convention must be :positive_diagonal, :unit_effect, or :none, got :$sign_convention"))

    vars = _label_variables(variables, result, n)
    perm, signs = if by === :restrictions
        restrictions === nothing && throw(ArgumentError(
            "by=:restrictions requires restrictions (sign matrix or SVARRestrictions)"))
        _label_by_restrictions(B0, restrictions)
    elseif by === :max_impact
        _label_by_max_impact(B0, vars)
    else
        B_ref === nothing && throw(ArgumentError("by=:reference requires B_ref"))
        size(B_ref) == size(B0) || throw(ArgumentError("B_ref size must match B₀"))
        _match_columns(Matrix{eltype(B0)}(B_ref), Matrix{eltype(B0)}(B0))
    end

    # `:reference` signs come from `_match_columns`; do not overwrite a negative
    # own-effect that the reference impact itself carries.
    if by === :max_impact && sign_convention !== :none
        signs = _apply_sign_convention(B0, perm, signs, sign_convention, vars)
    end

    names = if shock_names !== nothing
        length(shock_names) == n || throw(ArgumentError(
            "shock_names must have length $n, got $(length(shock_names))"))
        Vector{String}(shock_names)
    elseif by === :max_impact && variables isa AbstractVector &&
           !isempty(variables) && variables[1] isa AbstractString
        String[String(v) for v in variables]
    else
        nothing
    end
    _signed_perm_result(result, perm, signs; shock_names=names)
end

function _label_variables(variables, result, n::Int)
    variables === nothing && return collect(1:n)
    length(variables) == n || throw(ArgumentError(
        "variables must have length $n, got $(length(variables))"))
    if eltype(variables) <: Integer
        v = Int[Int(i) for i in variables]
        all(1 .<= v .<= n) || throw(ArgumentError("variables must be in 1:$n"))
        return v
    end
    names = hasfield(typeof(result), :varnames) ? getfield(result, :varnames) :
            String["Var $i" for i in 1:n]
    [_resolve_var_index(v, names, n) for v in variables]
end

function _resolve_var_index(var, names::Vector{String}, n::Int)
    if var isa Integer
        (1 <= Int(var) <= n) || throw(ArgumentError("variable index $var not in 1:$n"))
        return Int(var)
    elseif var isa AbstractString || var isa Symbol
        s = String(var)
        idx = findfirst(==(s), names)
        idx === nothing && throw(ArgumentError("variable \"$s\" not in $names"))
        return idx
    end
    throw(ArgumentError("variable must be an index or name, got $(typeof(var))"))
end

function _restriction_sign_matrix(restrictions::AbstractMatrix, n::Int)
    size(restrictions) == (n, n) || throw(ArgumentError(
        "restriction matrix must be $n × $n, got $(size(restrictions))"))
    Int.(sign.(restrictions))
end

function _restriction_sign_matrix(restrictions::SVARRestrictions, n::Int)
    n == restrictions.n_vars || throw(ArgumentError(
        "SVARRestrictions.n_vars=$(restrictions.n_vars) does not match n=$n"))
    S = zeros(Int, n, n)
    for r in restrictions.signs
        if r isa SignRestriction && r.horizon == 0
            S[r.variable, r.shock] = r.sign
        end
    end
    S
end

function _restriction_sign_matrix(restrictions, n::Int)
    S = zeros(Int, n, n)
    rs = restrictions isa AbstractSVARRestriction ? (restrictions,) : restrictions
    for item in rs
        r = item
        if r isa AbstractVector
            for rr in r
                if rr isa SignRestriction && rr.horizon == 0
                    S[rr.variable, rr.shock] = rr.sign
                end
            end
        elseif r isa SignRestriction && r.horizon == 0
            S[r.variable, r.shock] = r.sign
        end
    end
    S
end

function _col_restriction_score(Bcol::AbstractVector, j::Int, S::AbstractMatrix)
    sc = 0
    @inbounds for i in eachindex(Bcol)
        sij = S[i, j]
        sij == 0 && continue
        sc += Int(sign(Bcol[i]) == sij)
    end
    sc
end

function _label_by_restrictions(B0::AbstractMatrix{T}, restrictions) where {T<:AbstractFloat}
    n = size(B0, 2)
    S = _restriction_sign_matrix(restrictions, n)
    n_restr = count(!iszero, S)
    n_restr == 0 && throw(ArgumentError("restrictions contain no horizon-0 sign restrictions"))
    best_perm = collect(1:n)
    best_signs = ones(Int, n)
    best_score = -1
    n_ties = 0
    if n <= 8
        for perm in _permutations(n)
            signs = ones(Int, n)
            sc = 0
            @inbounds for j in 1:n
                col = view(B0, :, perm[j])
                s_plus = _col_restriction_score(col, j, S)
                s_minus = _col_restriction_score(-col, j, S)
                if s_minus > s_plus
                    signs[j] = -1
                    sc += s_minus
                else
                    signs[j] = 1
                    sc += s_plus
                end
            end
            if sc > best_score
                best_score = sc
                best_perm = collect(perm)
                best_signs = signs
                n_ties = 1
            elseif sc == best_score
                n_ties += 1
            end
        end
    else
        used = falses(n)
        best_perm = zeros(Int, n)
        best_signs = ones(Int, n)
        best_score = 0
        for j in 1:n
            bk, bs, bsc = 0, 1, -1
            for k in 1:n
                used[k] && continue
                col = view(B0, :, k)
                s_plus = _col_restriction_score(col, j, S)
                s_minus = _col_restriction_score(-col, j, S)
                if s_minus > s_plus && s_minus > bsc
                    bk, bs, bsc = k, -1, s_minus
                elseif s_plus >= bsc
                    bk, bs, bsc = k, 1, s_plus
                end
            end
            best_perm[j] = bk
            best_signs[j] = bs
            used[bk] = true
            best_score += bsc
        end
        n_ties = 1
    end
    if best_score < n_restr
        @warn "label_shocks: $(n_restr - best_score) of $n_restr sign restrictions remain unsatisfied"
    end
    n_ties > 1 && @warn "label_shocks: $n_ties signed permutations attain the same restriction score; labels are ambiguous"
    (best_perm, best_signs)
end

function _label_by_max_impact(B0::AbstractMatrix{T}, variables::Vector{Int}) where {T<:AbstractFloat}
    n = size(B0, 2)
    coln = [norm(view(B0, :, k)) + eps(T) for k in 1:n]
    S = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, k in 1:n
        S[j, k] = abs(B0[variables[j], k]) / coln[k]
    end
    best_perm = collect(1:n)
    best_score = -T(Inf)
    n_ties = 0
    if n <= 8
        for perm in _permutations(n)
            sc = zero(T)
            @inbounds for j in 1:n
                sc += S[j, perm[j]]
            end
            if sc > best_score + T(10) * eps(T)
                best_score = sc
                best_perm = collect(perm)
                n_ties = 1
            elseif abs(sc - best_score) <= T(10) * eps(T)
                n_ties += 1
            end
        end
    else
        used = falses(n)
        best_perm = zeros(Int, n)
        for j in 1:n
            k = argmax(i -> used[i] ? -T(Inf) : S[j, i], 1:n)
            best_perm[j] = k
            used[k] = true
        end
        n_ties = 1
    end
    n_ties > 1 && @warn "label_shocks: max-impact assignment is ambiguous"
    signs = ones(Int, n)
    (best_perm, signs)
end

function _apply_sign_convention(B0::AbstractMatrix{T}, perm::Vector{Int}, signs::Vector{Int},
                                 convention::Symbol, variables::Vector{Int}) where {T<:AbstractFloat}
    n = length(perm)
    out = copy(signs)
    @inbounds for j in 1:n
        col = view(B0, :, perm[j])
        i = convention === :unit_effect ? variables[j] : j
        s = out[j]
        impact = s * col[i]
        if impact < 0
            out[j] = -s
        end
    end
    out
end

function _col_signed(M::AbstractMatrix{T}, perm::Vector{Int}, signs::Vector{Int}) where {T}
    out = Matrix{T}(M[:, perm])
    @inbounds for j in eachindex(perm)
        signs[j] < 0 && (out[:, j] .*= -one(T))
    end
    out
end

function _row_signed(M::AbstractMatrix{T}, perm::Vector{Int}, signs::Vector{Int}) where {T}
    out = Matrix{T}(M[perm, :])
    @inbounds for j in eachindex(perm)
        signs[j] < 0 && (out[j, :] .*= -one(T))
    end
    out
end

function _permute_dist_params(d::Dict, perm::Vector{Int}, signs::Vector{Int}, n::Int)
    out = Dict{Symbol,Any}()
    for (k, v) in d
        if v isa AbstractVector && length(v) == n
            vv = v[perm]
            if k === :alpha || k === :kappa
                vv = copy(vv)
                @inbounds for j in 1:n
                    signs[j] < 0 && (vv[j] = -vv[j])
                end
            end
            out[k] = vv
        elseif k === :all_params && v isa AbstractVector && !isempty(v) && length(v) % n == 0
            nblk = length(v) ÷ n
            parts = [v[((b - 1) * n + 1):(b * n)][perm] for b in 1:nblk]
            out[k] = vcat(parts...)
        else
            out[k] = v
        end
    end
    out
end

function _permute_statid_field(f::Symbol, v, perm::Vector{Int}, signs::Vector{Int},
                                n::Int, shock_names, Q_new, theta_from_q::Bool)
    f === :B0 && return _col_signed(v, perm, signs)
    f === :Q && return Q_new === nothing ? _col_signed(v, perm, signs) : Q_new
    f === :W && return _row_signed(v, perm, signs)
    f === :se && return (v isa AbstractMatrix && size(v) == (n, n)) ?
        _col_signed(v, perm, ones(Int, n)) : v
    f === :shocks && return (v isa AbstractMatrix && size(v, 2) == n) ?
        _col_signed(v, perm, signs) : v
    f === :cond_var && return (v isa AbstractMatrix && size(v, 2) == n) ? v[:, perm] : v
    f === :garch_params && return (v isa AbstractMatrix && size(v, 1) == n) ? v[perm, :] : v
    if f === :Lambda && v isa AbstractVector
        return [λ isa AbstractVector && length(λ) == n ? λ[perm] : λ for λ in v]
    end
    f === :shock_names && return shock_names
    if f === :theta
        Q_new === nothing && return v
        theta_from_q && return _orthogonal_to_givens(Q_new, n)
        # Givens angles parameterize SO(n); an improper rotation is not reconstructed.
        return fill(eltype(v)(NaN), length(v))
    end
    f === :dist_params && return v isa Dict ? _permute_dist_params(v, perm, signs, n) : v
    v
end

function _signed_perm_result(result::R, perm::Vector{Int}, signs::Vector{Int};
                             shock_names::Union{Nothing,Vector{String}}=nothing) where {R}
    n = size(result.B0, 2)
    Q_new = hasfield(R, :Q) ? _col_signed(getfield(result, :Q), perm, signs) : nothing
    theta_from_q = Q_new !== nothing && det(Q_new) > 0
    if hasfield(R, :theta) && Q_new !== nothing && !theta_from_q
        @warn "label_shocks: det(Q) < 0 after signed permutation; Givens angles (theta) " *
              "parameterize SO(n) and are set to NaN. Use Q as the source of truth."
    end
    names_new = if shock_names !== nothing
        Vector{String}(shock_names)
    elseif hasfield(R, :shock_names)
        getfield(result, :shock_names)[perm]
    else
        _default_shock_names(n)
    end
    vals = ntuple(fieldcount(R)) do i
        f = fieldname(R, i)
        v = getfield(result, f)
        _permute_statid_field(f, v, perm, signs, n, names_new, Q_new, theta_from_q)
    end
    R(vals...)
end

# =============================================================================
# SID-20: unit-effect IRF scaling of Q
# =============================================================================

"""Scale rotation columns so selected impact entries equal `value`."""
function _unit_effect_specs(Q::AbstractMatrix{T}, model::VARModel{T},
                            shock_size) where {T<:AbstractFloat}
    n = size(Q, 2)
    L = safe_cholesky(model.Sigma)
    B0 = Matrix{T}(L * Q)
    if shock_size === nothing
        return Tuple{Int,Int,T}[(j, j, one(T)) for j in 1:n]
    end
    shock_size isa Pair || throw(ArgumentError(
        "shock_size must be a Pair (variable => value), got $(typeof(shock_size))"))
    k = _resolve_var_index(shock_size.first, model.varnames, n)
    value = T(shock_size.second)
    j = Int(argmax(abs.(view(B0, k, :))))
    Tuple{Int,Int,T}[(k, j, value)]
end

function _apply_irf_scale(Q::AbstractMatrix{T}, model::VARModel{T},
                          specs) where {T<:AbstractFloat}
    specs === nothing && return Q
    Qs = Matrix{T}(Q)
    L = safe_cholesky(model.Sigma)
    for (k, j, value) in specs
        b = (L * Qs)[k, j]
        abs(b) < T(100) * eps(T) && throw(ArgumentError(
            "unit-effect normalisation: impact of shock $j on variable $k is zero"))
        Qs[:, j] .*= value / b
    end
    Qs
end

function _scale_irf_values!(vals::AbstractArray{T,3}, specs) where {T<:AbstractFloat}
    specs === nothing && return vals
    for (k, j, value) in specs
        b = vals[1, k, j]
        abs(b) < T(100) * eps(T) && continue
        vals[:, :, j] .*= value / b
    end
    vals
end

function _scale_irf_draws!(draws::AbstractArray{T,4}, specs) where {T<:AbstractFloat}
    specs === nothing && return draws
    n_acc = size(draws, 1)
    @inbounds for i in 1:n_acc
        for (k, j, value) in specs
            b = draws[i, 1, k, j]
            abs(b) < T(100) * eps(T) && continue
            draws[i, :, :, j] .*= value / b
        end
    end
    draws
end

function _unit_effect_requested(normalize)
    eff = normalize === nothing ? :unit_variance : normalize
    eff in (:unit_variance, :unit_effect) || throw(ArgumentError(
        "normalize must be :unit_variance or :unit_effect, got :$eff"))
    eff === :unit_effect
end

function _unit_effect_specs_from_irf(irfvals::AbstractArray{T,3}, varnames::Vector{String},
                                     shock_size) where {T<:AbstractFloat}
    n = size(irfvals, 3)
    if shock_size === nothing
        return Tuple{Int,Int,T}[(j, j, one(T)) for j in 1:n]
    end
    shock_size isa Pair || throw(ArgumentError(
        "shock_size must be a Pair (variable => value), got $(typeof(shock_size))"))
    k = _resolve_var_index(shock_size.first, varnames, n)
    value = T(shock_size.second)
    j = Int(argmax(abs.(view(irfvals, 1, k, :))))
    Tuple{Int,Int,T}[(k, j, value)]
end
_unit_effect_specs_from_irf(irfvals::AbstractArray{T,3}, model::VARModel{T},
                            shock_size) where {T<:AbstractFloat} =
    _unit_effect_specs_from_irf(irfvals, model.varnames, shock_size)

function _scale_setid_irf(med, lo, hi, draws, specs)
    specs === nothing && return med, lo, hi, draws
    medc = copy(med)
    loc = copy(lo)
    hic = copy(hi)
    dc = draws === nothing ? nothing : copy(draws)
    T = eltype(medc)
    for (k, j, value) in specs
        b = medc[1, k, j]
        abs(b) < T(100) * eps(T) && continue
        sc = value / b
        medc[:, :, j] .*= sc
        loc[:, :, j] .*= sc
        hic[:, :, j] .*= sc
        if sc < 0
            tmp = copy(loc[:, :, j])
            loc[:, :, j] = hic[:, :, j]
            hic[:, :, j] = tmp
        end
        dc !== nothing && (dc[:, :, :, j] .*= sc)
    end
    medc, loc, hic, dc
end
