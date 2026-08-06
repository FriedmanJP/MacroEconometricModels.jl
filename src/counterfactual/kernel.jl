# Counterfactual module — weighted policy-projection kernel (CF-03, #383)
#
# Every counterfactual in the series (McKay-Wolf rule counterfactuals CF-10,
# optimal policy CF-11, Barnichon-Mesters OPP CF-13, CMW model averaging CF-17)
# reduces to this single weighted projection with a different (M, b, W, c).
# Matrices in, NamedTuple out: no RNG, no globals, no CF-01 type dependencies.

"""
    _policy_projection(M, b; W=nothing, c=nothing, method=:auto, rank_rtol=0.0)
        -> (; nu, error_path, rel_residual, rank, deficient, method_used)

Choose the policy-shock vector `ν ∈ R^{n_s}` minimizing

`(Mν + b)' W (Mν + b)`   (with `W = I` when `W === nothing`),

or, when the linear term `c` is supplied (the instrument-smoothing
initial-condition `wedge_term` of CF-02),

`ν'(M'WM)ν/2 + (M'Wb + c)'ν`.

`M` (`m × n_s`) maps shocks into rule violations (or stacked target responses)
and `b` (`m`) is the baseline violation (or stacked baseline/forecast path).

**Sign convention**: the returned `nu` INCLUDES the minus —
`ν* = −(M'WM)⁻¹M'W·b` (and `−(M'WM)⁻¹(M'Wb + c)` with a linear term). This is
the Barnichon–Mesters pitfall: the OPP is **minus** the WLS coefficient of the
gap forecast on the shock IRFs; dropping the sign *worsens* the loss. `b` is
built with CF-02's wedge convention: `b = Σᵢ A_x[i]·x_base[i] + Σₖ
A_z[k]·z_base[k] − wedge`.

# Regimes
- **`:ls`** — thin over-determined `n_s ≪ m` (the empirical case) or forced via
  `method = :ls`: whitened column-pivoted QR least squares. `(M'M)⁻¹M'` is never
  formed literally.
- **`:exact`** — `method = :exact`, or `:auto` with square `M` and no `c`:
  direct solve `ν = M \\ (−b)` (the rule then holds exactly and the solution is
  `W`-independent). Guarded by a condition estimate: `cond(M, 1) >
  1/sqrt(eps(T))` warns (ill-conditioning near the truncation edge — CMW cap
  their news columns for exactly this reason) and falls back to `:ls`.
- **`:ls_minnorm`** — rank-deficient `M` (near-collinear identified shocks —
  this WILL occur on real data): warns naming the deficient columns, then
  returns the minimum-norm solution (SVD, equivalent to `pinv`), `deficient =
  true`.

# Weights
`W` may be PSD-singular (pure-AIT weights are, by construction): a plain
Cholesky is tried first and, on failure, `W` is eigen-reduced with eigenvalues
below `maximum(λ)·m·eps(T)` dropped. (`safe_cholesky` is deliberately NOT used
here — its auto-jitter would silently assign ~eps weight to the null
directions of a singular `W`.)

# Diagnostics — the honesty signal
`error_path = M·ν* + b` is the implementation error: a large path means the
counterfactual is NOT enforceable with the available shocks (McKay–Wolf's
failed rate-peg case). It is always computed on the ORIGINAL (unwhitened)
`(M, b)` and must be surfaced by every caller, never hidden.
`rel_residual = ‖error_path‖ / max(‖b‖, floatmin(T)^{1/4})`, defined as 0 when
`‖b‖ ≈ 0`.
"""
function _policy_projection(M::AbstractMatrix{T}, b::AbstractVector{T};
                            W::Union{Nothing,AbstractMatrix{T}}=nothing,
                            c::Union{Nothing,AbstractVector{T}}=nothing,
                            method::Symbol=:auto,
                            rank_rtol::Real=0.0) where {T<:AbstractFloat}
    m, n_s = size(M)
    length(b) == m || throw(ArgumentError(
        "b: expected length $m (= rows of M), got $(length(b))"))
    W === nothing || size(W) == (m, m) || throw(ArgumentError(
        "W: expected size ($m, $m), got $(size(W))"))
    c === nothing || length(c) == n_s || throw(ArgumentError(
        "c: expected length $n_s (= columns of M), got $(length(c))"))
    method in (:auto, :ls, :exact) || throw(ArgumentError(
        "method: expected :auto, :ls or :exact, got :$method"))
    if method == :exact
        c === nothing || throw(ArgumentError(
            "method = :exact is incompatible with a linear term c (the exact solve ignores W and c); use :auto or :ls"))
        m == n_s || throw(ArgumentError(
            "method = :exact requires a square M, got ($m, $n_s)"))
    end

    # --- exact square solve (W-independent: M ν = −b holds exactly) ---
    if c === nothing && (method == :exact || (method == :auto && m == n_s))
        kappa = try
            cond(M, 1)
        catch
            T(Inf)
        end
        if isfinite(kappa) && kappa <= 1 / sqrt(eps(T))
            nu = M \ (-b)
            error_path = M * nu + b
            return (nu=nu, error_path=error_path,
                    rel_residual=_pp_rel_residual(error_path, b),
                    rank=n_s, deficient=false, method_used=:exact)
        else
            @warn "policy projection: square M is ill-conditioned (cond ≈ $(round(kappa, sigdigits=3)) > 1/sqrt(eps)); falling back to least squares. Consider capping the news-shock horizon (truncation-edge columns are the usual culprit)."
        end
    end

    # --- whiten by W = C'C ---
    A, btil = W === nothing ? (Matrix{T}(M), Vector{T}(b)) :
              begin
                  C = _pp_weight_factor(W)
                  (C * M, C * b)
              end

    # --- rank detection via column-pivoted QR ---
    F = qr(A, ColumnNorm())   # ColumnNorm() is the pivoted form on Julia >= 1.10
    Rdiag = abs.(diag(F.R))
    dmax = isempty(Rdiag) ? zero(T) : maximum(Rdiag)
    tol = (rank_rtol > 0 ? T(rank_rtol) : T(maximum(size(A))) * eps(T)) * dmax
    r = dmax == zero(T) ? 0 : count(>(tol), Rdiag)

    local nu::Vector{T}
    if r == n_s
        # full column rank
        if c === nothing
            nu = F \ (-btil)
        else
            # FOC (A'A) ν = −(A'b̃ + c) through the R factor: A'A = Π R'R Π'
            rhs2 = -(A' * btil + c)
            y = LowerTriangular(F.R') \ rhs2[F.p]
            v = UpperTriangular(F.R) \ y
            nu = zeros(T, n_s)
            nu[F.p] = v
        end
        method_used = :ls
        deficient = false
    else
        deficient_cols = sort(F.p[r+1:end])
        @warn "policy projection: M is rank-deficient (rank $r < $n_s columns; near-collinear shock columns $(deficient_cols)); returning the minimum-norm solution. Near-collinear identified shocks (e.g. two transitory policy-shock paths) are the usual cause."
        SV = svd(A)
        smax = isempty(SV.S) ? zero(T) : SV.S[1]
        stol = (rank_rtol > 0 ? T(rank_rtol) : T(maximum(size(A))) * eps(T)) * smax
        keep = findall(>(stol), SV.S)
        r = length(keep)
        if isempty(keep)
            nu = zeros(T, n_s)
        elseif c === nothing
            nu = SV.V[:, keep] * ((SV.U[:, keep]' * (-btil)) ./ SV.S[keep])
        else
            rhs2 = -(A' * btil + c)
            Vk = SV.V[:, keep]
            nu = Vk * ((Vk' * rhs2) ./ (SV.S[keep] .^ 2))
        end
        method_used = :ls_minnorm
        deficient = true
    end

    error_path = M * nu + b
    return (nu=nu, error_path=error_path,
            rel_residual=_pp_rel_residual(error_path, b),
            rank=r, deficient=deficient, method_used=method_used)
end

# Relative implementation-error residual; 0 when the baseline violation is ~0.
function _pp_rel_residual(error_path::AbstractVector{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    nb = norm(b)
    nb <= eps(T) && return zero(T)
    return T(norm(error_path) / max(nb, floatmin(T)^(1 // 4)))
end

# Factor W = C'C. Plain Cholesky when PD; eigen-reduction when PSD-singular
# (pure-AIT weights). Deliberately not safe_cholesky: its auto-jitter would
# silently give ~eps weight to the null directions of a singular W.
function _pp_weight_factor(W::AbstractMatrix{T}) where {T<:AbstractFloat}
    S = Symmetric(Matrix{T}(W))
    F = cholesky(S; check=false)
    issuccess(F) && return Matrix{T}(F.U)
    E = eigen(S)
    lmax = maximum(E.values)
    lmax > 0 || throw(ArgumentError(
        "W: expected a PSD weight matrix with at least one positive eigenvalue, largest eigenvalue is $lmax"))
    keep = findall(>(lmax * size(W, 1) * eps(T)), E.values)
    return Matrix{T}(Diagonal(sqrt.(E.values[keep])) * E.vectors[:, keep]')
end
