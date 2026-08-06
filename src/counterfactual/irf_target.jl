# Counterfactual module — stacked IRF-matching targets + CTW covariance (CF-06, #386)
#
# CMW's model bank is estimated by limited-information IRF matching: stack the
# empirical policy-shock IRFs into a target vector theta_hat with covariance
# V_bar, then evaluate a Gaussian quasi-likelihood at model-implied IRFs.
# Non-diagonal weighting is load-bearing: a diagonal V makes posterior model
# probabilities artificially decisive (CMW's explicit warning).

"""
    stacked_irf_target(ce::PolicyCausalEffects; order=:shock_major,
                       scale=Dict{Symbol,Float64}(), drop=[], inflate=[])
        -> (; theta_hat, V_bar, index)

Stack the causal-effect container into an IRF-matching target vector
`theta_hat` with draw covariance `V_bar` (CMW 2025, App. B.1).

- Stacking runs per shock, then per variable (outcomes first, then
  instruments), then horizon `1…H` (`order = :shock_major`, the CMW order
  `[Π^s1; Y^s1; R^s1; Π^s2; …]`; `:variable_major` swaps the outer loops).
  `index` records `(var, shock, h)` per element — CF-17 aligns the model side
  against it.
- `theta_hat` from the point matrices; `V_bar` is the sample covariance across
  the container's draws (draws are required).
- `scale`: per-variable multiplicative factors applied to point and draws
  (annualization bookkeeping — CMW divide the π and i blocks by 4; a mismatch
  here silently biases the model match, so the applied factors are logged).
- `drop`: `(variable, shock_index, horizon)` tuples removed from the target.
- `inflate`: like `drop` but keeps the element and assigns it a huge variance
  (`1e6`, off-diagonals zeroed) so the match ignores it — CMW do this to the
  Romer–Romer impact responses.
"""
function stacked_irf_target(ce::PolicyCausalEffects{T};
                            order::Symbol=:shock_major,
                            scale::AbstractDict{Symbol,<:Real}=Dict{Symbol,Float64}(),
                            drop::AbstractVector{<:Tuple}=Tuple{Symbol,Int,Int}[],
                            inflate::AbstractVector{<:Tuple}=Tuple{Symbol,Int,Int}[]) where {T<:AbstractFloat}
    order in (:shock_major, :variable_major) || throw(ArgumentError(
        "order: expected :shock_major or :variable_major, got :$order"))
    nd = n_draws(ce)
    nd >= 2 || throw(ArgumentError(
        "stacked_irf_target requires a draws-bearing container (n_draws >= 2, got $nd); build the PolicyCausalEffects from an estimator with uncertainty draws"))
    (isempty(ce.instruments) || ce.Theta_z_draws !== nothing) || throw(ArgumentError(
        "stacked_irf_target: instrument draws (Theta_z_draws) are missing"))

    vars = vcat(ce.outcomes, ce.instruments)
    n_x = length(ce.outcomes)
    for k in keys(scale)
        k in vars || throw(ArgumentError(
            "scale: :$k is not among the container variables $(vars)"))
    end
    isempty(scale) || @info "stacked_irf_target: applying per-variable scale factors $(Dict(scale))"

    point_of(v) = v <= n_x ? ce.Theta_x[v] : ce.Theta_z[v-n_x]
    draws_of(v) = v <= n_x ? ce.Theta_x_draws[v] : ce.Theta_z_draws[v-n_x]
    scale_of(v) = T(get(scale, vars[v], one(T)))

    n_s = n_shocks(ce)
    H = ce.H
    dropset = Set{Tuple{Symbol,Int,Int}}((Symbol(t[1]), Int(t[2]), Int(t[3])) for t in drop)
    for t in dropset
        (t[1] in vars && 1 <= t[2] <= n_s && 1 <= t[3] <= H) || throw(ArgumentError(
            "drop: entry $t does not match any (variable, shock, horizon) in the container"))
    end

    idx = NamedTuple{(:var, :shock, :h),Tuple{Symbol,Int,Int}}[]
    outer = order == :shock_major ? (1:n_s) : (1:length(vars))
    for a in outer
        inner = order == :shock_major ? (1:length(vars)) : (1:n_s)
        for b in inner
            k = order == :shock_major ? a : b
            v = order == :shock_major ? b : a
            for h in 1:H
                (vars[v], k, h) in dropset && continue
                push!(idx, (var=vars[v], shock=k, h=h))
            end
        end
    end
    m = length(idx)
    m >= 1 || throw(ArgumentError("stacked_irf_target: every element was dropped"))

    theta_hat = Vector{T}(undef, m)
    Dmat = Matrix{T}(undef, m, nd)
    for (i, e) in enumerate(idx)
        v = findfirst(==(e.var), vars)
        s = scale_of(v)
        theta_hat[i] = s * point_of(v)[e.h, e.shock]
        Dmat[i, :] = s .* draws_of(v)[e.h, e.shock, :]
    end
    V_bar = Matrix{T}(cov(Dmat, dims=2))

    for t in inflate
        e = (Symbol(t[1]), Int(t[2]), Int(t[3]))
        i = findfirst(x -> (x.var, x.shock, x.h) == e, idx)
        i === nothing && throw(ArgumentError(
            "inflate: entry $t does not match any element of the stacked target"))
        V_bar[i, :] .= zero(T)
        V_bar[:, i] .= zero(T)
        V_bar[i, i] = T(1e6)
    end

    return (theta_hat=theta_hat, V_bar=V_bar, index=idx)
end

"""
    ctw_covariance(V_bar, block_len; bandwidth=8, eta=1.0) -> (; V, repair)

Christiano–Trabandt–Walentin (2010) triangular-kernel damping of a stacked IRF
covariance, followed by a PSD repair:

- `V_bar` is partitioned into `block_len × block_len` blocks (one block per
  `(variable, shock)` pair; `block_len = H`).
- Diagonal *elements* of `V_bar` are kept exactly.
- Every off-diagonal element at horizon distance `d = |iₕ − jₕ|` (within- and
  cross-block alike) is multiplied by `max(0, 1 − d/bandwidth)^eta`; everything
  with `d ≥ bandwidth` is zeroed — CMW zero cross-covariances beyond the
  bandwidth entirely. Same-horizon cross-block elements (`d = 0`) are
  untouched.
- The kernel transform does not preserve PSD-ness in general, so the result is
  symmetrized and eigen-clipped; `repair` is the largest applied eigenvalue
  clip (0 when no repair was needed) so CF-17 can report it.

This transform is a *statistical regularization* of a noisy draw covariance,
not an estimator of the true covariance (Christiano, Trabandt & Walentin 2010;
CMW use `bandwidth = 8, eta = 1`).
"""
function ctw_covariance(V_bar::AbstractMatrix{T}, block_len::Int;
                        bandwidth::Int=8, eta::Real=1.0) where {T<:AbstractFloat}
    m = size(V_bar, 1)
    size(V_bar, 2) == m || throw(ArgumentError(
        "V_bar: expected a square matrix, got $(size(V_bar))"))
    block_len >= 1 || throw(ArgumentError("block_len: expected >= 1, got $block_len"))
    m % block_len == 0 || throw(ArgumentError(
        "V_bar: size $m is not a multiple of block_len = $block_len"))
    bandwidth >= 1 || throw(ArgumentError("bandwidth: expected >= 1, got $bandwidth"))

    V = Matrix{T}(V_bar)
    for j in 1:m, i in 1:m
        i == j && continue
        hi = mod1(i, block_len)
        hj = mod1(j, block_len)
        d = abs(hi - hj)
        kern = d >= bandwidth ? zero(T) : (one(T) - T(d) / T(bandwidth))^T(eta)
        V[i, j] *= kern
    end

    # PSD repair: symmetrize + eigen-clip (the kernel does not preserve PSD-ness).
    S = Symmetric((V + V') / 2)
    E = eigen(S)
    lmax = maximum(abs, E.values)
    tiny = lmax > 0 ? lmax * m * eps(T) : zero(T)
    repair = zero(T)
    lam = copy(E.values)
    for i in eachindex(lam)
        if lam[i] < tiny
            repair = max(repair, tiny - lam[i])
            lam[i] = tiny
        end
    end
    Vp = repair > 0 ? Matrix{T}(E.vectors * Diagonal(lam) * E.vectors') : Matrix{T}(S)
    return (V=(Vp + Vp') / 2, repair=repair)
end

"""
    precision_of(V) -> (; precision, logdet)

Inverse and log-determinant of a (repaired) target covariance from one
factorization: Cholesky when positive definite, `robust_inv` + `logdet_safe`
fallback otherwise. CF-17's marginal likelihood needs the log-det on exactly
the same matrix as the precision (CMW fixed a bug on this normalization).
"""
function precision_of(V::AbstractMatrix{T}) where {T<:AbstractFloat}
    Vh = Hermitian(Matrix{T}(V))
    F = cholesky(Vh; check=false)
    if issuccess(F)
        return (precision=Matrix{T}(inv(F)), logdet=T(logdet(F)))
    end
    return (precision=Matrix{T}(robust_inv(Vh)), logdet=T(logdet_safe(Matrix{T}(Vh))))
end
