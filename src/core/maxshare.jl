# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Max-share identification (Uhlig 2003; Francis, Owyang, Roush & DiCecio 2014;
Barsky & Sims 2011; Angeletos, Collard & Dellas 2020).
"""

# =============================================================================
# Result
# =============================================================================

"""
    MaxShareResult{T} <: AbstractAnalysisResult

Max-share (forecast-error-variance / spectral-share) identification of one or
more structural shocks. The first `n_shocks` columns of `Q` are identified;
remaining columns are an orthogonal complement (`is_partial=true` when that
complement is nonempty).

# Fields
- `Q`: orthogonal rotation; identified columns occupy the leading positions
- `q`: first identified column of `Q`
- `target`: index of the target variable
- `horizons`: time-domain window (`nothing` under frequency-domain identification)
- `band`: frequency band `(ω₁, ω₂)` (`nothing` under time-domain identification)
- `share`: leading eigenvalue of `S` divided by `tr(S)`
- `eigvals`: eigenvalues of `S` in descending order
- `varnames`, `shock_names`
- `is_partial`: `true` when unidentified columns remain
"""
struct MaxShareResult{T<:AbstractFloat} <: AbstractAnalysisResult
    Q::Matrix{T}
    q::Vector{T}
    target::Int
    horizons::Union{UnitRange{Int},Nothing}
    band::Union{Tuple{T,T},Nothing}
    share::T
    eigvals::Vector{T}
    varnames::Vector{String}
    shock_names::Vector{String}
    is_partial::Bool
end

function Base.show(io::IO, r::MaxShareResult{T}) where {T}
    n = size(r.Q, 1)
    window = if r.band !== nothing
        "band ($(round(r.band[1]; digits=3)), $(round(r.band[2]; digits=3)))"
    else
        "horizons $(r.horizons)"
    end
    spec = Any[
        "Variables"     n;
        "Target"        "$(r.varnames[r.target]) ($(r.target))";
        "Window"        window;
        "FEV share"     _fmt(r.share; digits=3);
        "Partial"       r.is_partial ? "Yes" : "No";
    ]
    _pretty_table(io, spec;
        title = "Max-Share Identification Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    _matrix_table(io, r.Q, "Rotation Matrix (Q)";
        row_labels=r.varnames,
        col_labels=r.shock_names)
    r.is_partial && println(io,
        "Partial identification: FEVD/HD of unidentified shocks are not identified.")
end

# =============================================================================
# Public entry
# =============================================================================

"""
    identify_max_share(model; target, horizons=0:20, band=nothing,
                       cumulative=false, n_shocks=1, previous=nothing) -> MaxShareResult

Identify the shock (or `n_shocks` shocks) that maximises the forecast-error
variance of `target` over `horizons`, or the spectral mass of `target` on
`band=(ω₁, ω₂)`.

Time domain (Francis, Owyang, Roush & DiCecio 2014; Barsky & Sims 2011):

```math
S = \\sum_{h \\in \\mathcal{H}} (e_i' \\Phi_h L)'(e_i' \\Phi_h L), \\qquad
q_1 = \\arg\\max_{q'q=1} q'Sq
```

Frequency domain (Angeletos, Collard & Dellas 2020): Gauss–Legendre quadrature
of the same quadratic form with the VAR transfer function on `band`.

`previous` (a vector or `n × k` matrix of earlier rotation columns) restricts
the maximisation to the orthogonal complement — the Barsky–Sims news-shock
construction after a surprise-TFP column. Sign-normalise so the target impact
is positive. Complete `Q` with an orthogonal complement; remaining columns are
unidentified.
"""
function identify_max_share(model::VARModel{T};
                           target::Union{Int,AbstractString},
                           horizons::Union{Nothing,UnitRange{Int},AbstractVector{<:Integer}}=nothing,
                           band::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                           cumulative::Bool=false,
                           n_shocks::Int=1,
                           previous::Union{Nothing,AbstractVector,AbstractMatrix}=nothing,
                           n_quad::Int=32,
                           shock_names::Union{Nothing,Vector{String}}=nothing,
                           kwargs...) where {T<:AbstractFloat}
    n = nvars(model)
    tgt = _maxshare_resolve_target(target, model.varnames, n)
    n_shocks >= 1 || throw(ArgumentError("n_shocks must be ≥ 1"))
    n_quad >= 1 || throw(ArgumentError("n_quad must be ≥ 1"))

    Qprev = previous === nothing ? nothing : _maxshare_orthonormal_previous(previous, n, T)
    kprev = Qprev === nothing ? 0 : size(Qprev, 2)
    n_shocks + kprev <= n || throw(ArgumentError(
        "n_shocks=$n_shocks plus $(kprev) previous columns exceeds n=$n"))

    if band !== nothing && horizons !== nothing
        throw(ArgumentError("specify either horizons or band, not both"))
    end

    L = cholesky_factor(model)
    A = extract_ar_coefficients(model.B, n, model.p)

    hs_store, band_store, S = if band !== nothing
        ω1, ω2 = T(band[1]), T(band[2])
        ω2 > ω1 || throw(ArgumentError("band requires ω₂ > ω₁, got ($ω1, $ω2)"))
        Sω = _maxshare_S_frequency(A, L, tgt, ω1, ω2, n_quad, T)
        nothing, (ω1, ω2), Sω
    else
        hs = _maxshare_horizons(horizons)
        Hmax = maximum(hs)
        Phi = ma_coefficients(model, Hmax + 1)
        St = _maxshare_S_time(Phi, L, tgt, hs, cumulative, T)
        _maxshare_unitrange(hs), nothing, St
    end

    S = Matrix{T}((S + S') / T(2))
    if Qprev !== nothing
        P = Matrix{T}(I, n, n) - Qprev * Qprev'
        S = Matrix{T}((P * S * P + (P * S * P)') / T(2))
    end
    trS = tr(S)
    trS > T(1e-14) || throw(IdentificationError(
        "max-share criterion matrix S is numerically zero for the chosen window"))

    evals, evecs = _maxshare_leading(S, n_shocks, T)
    for j in 1:n_shocks
        _maxshare_sign_normalize!(view(evecs, :, j), L, tgt)
    end
    Q = _complete_maxshare_Q(evecs, n)
    share = evals[1] / trS
    share = T(clamp(share, zero(T), one(T)))

    snames = something(shock_names, _maxshare_shock_names(n, n_shocks))
    length(snames) == n || throw(ArgumentError("shock_names must have length n=$n"))

    MaxShareResult{T}(Q, Vector{T}(Q[:, 1]), tgt, hs_store, band_store,
                      T(share), evals, copy(model.varnames), Vector{String}(snames),
                      n_shocks < n)
end

# =============================================================================
# Internals
# =============================================================================

function _maxshare_resolve_target(target::Integer, varnames::Vector{String}, n::Int)
    (1 <= target <= n) || throw(ArgumentError("target must be in 1:$n"))
    Int(target)
end

function _maxshare_resolve_target(target::AbstractString, varnames::Vector{String}, n::Int)
    idx = findfirst(==(String(target)), varnames)
    idx === nothing && throw(ArgumentError(
        "target \"$target\" is not in varnames $(varnames)"))
    idx
end

function _maxshare_horizons(horizons::Nothing)
    0:20
end

function _maxshare_horizons(horizons::UnitRange{Int})
    first(horizons) >= 0 || throw(ArgumentError("horizons must be ≥ 0"))
    isempty(horizons) && throw(ArgumentError("horizons must be nonempty"))
    horizons
end

function _maxshare_horizons(horizons::AbstractVector{<:Integer})
    isempty(horizons) && throw(ArgumentError("horizons must be nonempty"))
    hs = sort!(unique(Int.(horizons)))
    first(hs) >= 0 || throw(ArgumentError("horizons must be ≥ 0"))
    hs
end

function _maxshare_unitrange(hs::UnitRange{Int})
    hs
end

function _maxshare_unitrange(hs::AbstractVector{Int})
    r = first(hs):last(hs)
    length(r) == length(hs) || throw(ArgumentError(
        "horizons must be a contiguous unit range (got a gapped vector)"))
    r
end

function _maxshare_orthonormal_previous(previous, n::Int, ::Type{T}) where {T<:AbstractFloat}
    M = previous isa AbstractVector ? reshape(Vector{T}(previous), :, 1) : Matrix{T}(previous)
    size(M, 1) == n || throw(ArgumentError("previous must have $n rows, got $(size(M, 1))"))
    k = size(M, 2)
    k < 1 && throw(ArgumentError("previous has no columns"))
    k < n || throw(ArgumentError("previous has too many columns"))
    F = qr(M)
    Q = Matrix{T}(F.Q)
    s = [i <= k && F.R[i, i] < zero(T) ? -one(T) : one(T) for i in 1:k]
    Q[:, 1:k] * Diagonal(s)
end

function _maxshare_S_time(Phi::Vector{<:AbstractMatrix{T}}, L::AbstractMatrix{T},
                          target::Int, horizons, cumulative::Bool,
                          ::Type{T}) where {T<:AbstractFloat}
    n = size(L, 1)
    S = zeros(T, n, n)
    v = zeros(T, n)
    if cumulative
        PsiL = zeros(T, n, n)
        hmax = maximum(horizons)
        @inbounds for h in 0:hmax
            mul!(PsiL, Phi[h + 1], L, one(T), one(T))
            if h in horizons
                v .= @view PsiL[target, :]
                S .+= v * v'
            end
        end
    else
        scratch = zeros(T, n, n)
        @inbounds for h in horizons
            mul!(scratch, Phi[h + 1], L)
            v .= @view scratch[target, :]
            S .+= v * v'
        end
    end
    S
end

function _maxshare_S_frequency(A::Vector{<:AbstractMatrix{T}}, L::AbstractMatrix{T},
                               target::Int, ω1::T, ω2::T, n_quad::Int,
                               ::Type{T}) where {T<:AbstractFloat}
    n = size(L, 1)
    p = length(A)
    nodes, weights = _gauss_legendre_interval(n_quad, ω1, ω2)
    S = zeros(T, n, n)
    Lc = Matrix{Complex{T}}(L)
    @inbounds for i in eachindex(nodes)
        ω = nodes[i]
        w = weights[i]
        M = Matrix{Complex{T}}(I, n, n)
        zk = one(Complex{T})
        z = cis(-ω)
        for j in 1:p
            zk *= z
            M .-= A[j] .* zk
        end
        CL = M \ Lc
        r = CL[target, :]
        S .+= w .* real.(r * r')
    end
    S
end

function _maxshare_leading(S::AbstractMatrix{T}, n_shocks::Int,
                           ::Type{T}) where {T<:AbstractFloat}
    F = eigen(Symmetric(S))
    idx = sortperm(F.values; rev=true)
    evals = Vector{T}(F.values[idx])
    evecs = Matrix{T}(F.vectors[:, idx[1:n_shocks]])
    for j in 1:n_shocks
        nrm = norm(view(evecs, :, j))
        nrm < T(1e-12) && throw(IdentificationError(
            "max-share eigenvector $j is numerically zero"))
        evecs[:, j] ./= nrm
    end
    evals, evecs
end

function _maxshare_sign_normalize!(q::AbstractVector{T}, L::AbstractMatrix{T},
                                   target::Int) where {T<:AbstractFloat}
    impact = dot(view(L, target, :), q)
    if impact < 0
        q .*= -one(T)
    elseif abs(impact) < T(1e-14)
        k = findfirst(x -> abs(x) > T(1e-12), q)
        k !== nothing && q[k] < 0 && (q .*= -one(T))
    end
    q
end

function _complete_maxshare_Q(Qid::AbstractMatrix{T}, n::Int) where {T<:AbstractFloat}
    k = size(Qid, 2)
    k == n && return Matrix{T}(Qid)
    M = Matrix{T}(I, n, n)
    M[:, 1:k] = Qid
    Fq = qr(M)
    Q = Matrix{T}(Fq.Q)
    s = [Fq.R[i, i] < zero(T) ? -one(T) : one(T) for i in 1:n]
    Q = Q * Diagonal(s)
    @inbounds for j in 1:k
        if dot(view(Q, :, j), view(Qid, :, j)) < 0
            Q[:, j] .*= -one(T)
        end
    end
    Q
end

function _maxshare_shock_names(n::Int, n_shocks::Int)
    sn = Vector{String}(undef, n)
    @inbounds for j in 1:n_shocks
        sn[j] = n_shocks == 1 ? "Max-share" : "Max-share $j"
    end
    u = 1
    @inbounds for i in (n_shocks + 1):n
        sn[i] = "Unidentified $u"
        u += 1
    end
    sn
end
