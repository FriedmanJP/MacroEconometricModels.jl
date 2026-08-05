# Counterfactual module — behavioral-expectations operators (CF-09, #389)
#
# Behavioral frictions in sequence space are linear operators transforming a
# rational-expectations Jacobian J into a behavioral one, column-by-column in
# the ANTICIPATION dimension (Gabaix 2020 cognitive discounting; Auclert,
# Rognlie & Straub 2020 sticky expectations; CMW 2025 App. C.2.3,
# add_cognitive_disc.m / add_sticky.m).
#
# These operators apply to PRIVATE-BLOCK expectations: CMW apply them to block
# Jacobians BEFORE the GE closure (household m_d = 1, firm m_f = 0.65 in their
# headline runs — apply selectively per block to `sequence_jacobian` outputs).
# Applying them to an already-GE-closed menu (`behavioral(ce)`) is an
# approximation; the docstring says so.
#
# Column convention (matches CMW 1-based indexing): s = 1 is the UNANTICIPATED
# shock (announcement = arrival, weight 1); column s carries news s−1 periods
# ahead and the Gabaix weight is m^(s−1).

"""
    _fake_news_of(J) -> F

Fake-news decomposition of a sequence-space Jacobian: `F[t, s] = J[t, s] −
J[t−1, s−1]` (zero padding on the first row/column), inverting the SSJ
identity `J[t, s] = J[t−1, s−1] + F[t, s]` (Auclert et al. 2021).
"""
function _fake_news_of(J::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = LinearAlgebra.checksquare(J)
    F = copyto!(Matrix{T}(undef, n, n), J)
    for s in n:-1:2, t in n:-1:2
        F[t, s] -= J[t-1, s-1]
    end
    return F
end

# Rebuild J from a (possibly reweighted) fake-news matrix.
function _rebuild_from_fake_news(F::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = LinearAlgebra.checksquare(F)
    J = copyto!(Matrix{T}(undef, n, n), F)
    for s in 2:n, t in 2:n
        J[t, s] += J[t-1, s-1]
    end
    return J
end

"""
    cognitive_discounting(J, m) -> Matrix

Gabaix (2020) cognitive discounting of a `T × T` sequence-space Jacobian:
news about a date `s − 1` periods ahead is down-weighted by `m^(s−1)`
(`s = 1` unanticipated, weight 1). Implemented through the fake-news
decomposition — `F` columns are reweighted and `J` is rebuilt — which is
algebraically identical to CMW's direct recursion (`add_cognitive_disc.m`,
ported as `_cognitive_disc_cmw` and cross-checked in the tests).

`m = 1` returns an exact copy; `m = 0` kills all anticipation (every column
becomes the time-shifted unanticipated response). On an anticipation-free
(backward-looking) Jacobian the operator is the identity for every `m`.
"""
function cognitive_discounting(J::AbstractMatrix{T}, m::Real) where {T<:AbstractFloat}
    n = LinearAlgebra.checksquare(J)
    0 <= m <= 1 || throw(ArgumentError("m: expected 0 <= m <= 1 (Gabaix weight), got $m"))
    m == 1 && return copy(Matrix{T}(J))
    F = _fake_news_of(J)
    for s in 2:n
        F[:, s] .*= T(m)^(s - 1)
    end
    return _rebuild_from_fake_news(F)
end

# Direct port of CMW add_cognitive_disc.m (solve_model.m:666) — kept as the
# cross-check implementation; tests assert 1e-12 agreement with the primary.
function _cognitive_disc_cmw(J::AbstractMatrix{T}, m::Real) where {T<:AbstractFloat}
    n = LinearAlgebra.checksquare(J)
    Jcd = zeros(T, n, n)
    Jcd[:, 1] = J[:, 1]
    for j in 2:n
        Jcd[2:end, j] = T(m)^(j - 1) .* (J[2:end, j] .- J[1:end-1, j-1]) .+ Jcd[1:end-1, j-1]
        Jcd[1, j] = T(m)^(j - 1) * J[1, j]
    end
    return Jcd
end

"""
    sticky_expectations(J, theta) -> Matrix

Auclert–Rognlie–Straub sticky expectations on a `T × T` sequence-space
Jacobian: each period agents update their information set with probability
`1 − theta`. Exact port of CMW's `add_sticky.m` recursion (O(T²) running
accumulator):

    J_st[:, 1] = J[:, 1]
    J_st[1, s] = (1 − θ)·J[1, s]                    (s ≥ 2)
    J_st[t, s] = θ·J_st[t−1, s−1] + (1 − θ)·J[t, s] (t, s ≥ 2)

`theta = 0` returns an exact copy. On an anticipation-free Jacobian the
operator is the identity for every `theta` (nothing to be inattentive about);
on anticipation entries it damps by the informed share (e.g. `(1 − θ^t)` for a
one-equation forward model — derived in the tests).
"""
function sticky_expectations(J::AbstractMatrix{T}, theta::Real) where {T<:AbstractFloat}
    n = LinearAlgebra.checksquare(J)
    0 <= theta <= 1 || throw(ArgumentError(
        "theta: expected 0 <= theta <= 1 (per-period non-update probability), got $theta"))
    theta == 0 && return copy(Matrix{T}(J))
    th = T(theta)
    Jst = zeros(T, n, n)
    Jst[:, 1] = J[:, 1]
    for s in 2:n
        Jst[1, s] = (1 - th) * J[1, s]
    end
    for s in 2:n, t in 2:n
        Jst[t, s] = th * Jst[t-1, s-1] + (1 - th) * J[t, s]
    end
    return Jst
end

"""
    behavioral(ce::PolicyCausalEffects; m=1.0, theta=0.0) -> PolicyCausalEffects

Apply [`cognitive_discounting`](@ref) then [`sticky_expectations`](@ref) to
every block (and draw) of a **square** model-implied container. Labels gain a
`(m=…, θ=…)` tag.

!!! warning
    Thin empirical containers must NOT be behavioralized — they are already
    data (an informative error is thrown). And applying the operators to a
    GE-closed menu is an approximation: CMW apply them to *block* Jacobians
    before the GE closure, selectively per block (household `m_d = 1`, firm
    `m_f = 0.65`) — for that, apply the matrix operators directly to
    [`sequence_jacobian`](@ref) outputs and re-close.
"""
function behavioral(ce::PolicyCausalEffects{T}; m::Real=1.0, theta::Real=0.0) where {T<:AbstractFloat}
    is_square(ce) || throw(ArgumentError(
        "behavioral operators apply to model-implied (square) menus; this container is thin (n_s = $(n_shocks(ce)) < H = $(ce.H)) — empirically identified causal effects are already data and must not be behavioralized"))
    f(J) = sticky_expectations(cognitive_discounting(J, m), theta)
    fx = [f(M) for M in ce.Theta_x]
    fz = [f(M) for M in ce.Theta_z]
    fdx = ce.Theta_x_draws === nothing ? nothing :
          [begin
               D2 = similar(D)
               for d in 1:size(D, 3)
                   D2[:, :, d] = f(D[:, :, d])
               end
               D2
           end for D in ce.Theta_x_draws]
    fdz = ce.Theta_z_draws === nothing ? nothing :
          [begin
               D2 = similar(D)
               for d in 1:size(D, 3)
                   D2[:, :, d] = f(D[:, :, d])
               end
               D2
           end for D in ce.Theta_z_draws]
    labels = [l * " (m=$(m), θ=$(theta))" for l in ce.shock_labels]
    PolicyCausalEffects{T}(copy(ce.outcomes), copy(ce.instruments), fx, fz,
                           fdx, fdz, ce.H, labels, ce.source)
end
