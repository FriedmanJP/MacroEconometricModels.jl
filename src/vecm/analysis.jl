# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
IRF, FEVD, and Historical Decomposition dispatch for VECM via VAR conversion.

All structural analysis methods work automatically through `to_var()`.
"""

"""
    irf(vecm::VECMModel, horizon; kwargs...) -> ImpulseResponse

Compute IRFs for a VECM by converting to VAR representation.
All identification methods (Cholesky, sign, narrative, etc.) are supported.
"""
function irf(vecm::VECMModel{T}, horizon::Int; kwargs...) where {T}
    irf(to_var(vecm), horizon; kwargs...)
end

"""
    fevd(vecm::VECMModel, horizon; kwargs...) -> FEVD

Compute FEVD for a VECM by converting to VAR representation.
"""
function fevd(vecm::VECMModel{T}, horizon::Int; kwargs...) where {T}
    fevd(to_var(vecm), horizon; kwargs...)
end

"""
    historical_decomposition(vecm::VECMModel, horizon; kwargs...) -> HistoricalDecomposition

Compute historical decomposition for a VECM by converting to VAR representation.
"""
function historical_decomposition(vecm::VECMModel{T}, horizon::Int=effective_nobs(vecm); kwargs...) where {T}
    historical_decomposition(to_var(vecm), horizon; kwargs...)
end

# =============================================================================
# VECM restriction testing — Johansen LR on the cointegrating structure
# (EV-38 / #446). References: Johansen (1991, 1995); Johansen & Juselius (1990).
# =============================================================================

using LinearAlgebra, Distributions

"""
    _grr_eigen(Saa, Sab, Sbb) -> (values, vectors)

Solve the generalized reduced-rank eigenproblem `|λ Saa − Sab Sbb⁻¹ Sba| = 0`
(with `Sba = Sab'`) by whitening `Saa` with a `safe_cholesky` factor and solving
the resulting symmetric eigenproblem. Returns real eigenvalues sorted descending
(clamped to `[0, 1)`) and their eigenvectors expressed in the `Saa` coordinate
system. Imaginary parts are numerically zero for the correct pencil.
"""
function _grr_eigen(Saa::AbstractMatrix{T}, Sab::AbstractMatrix{T},
                    Sbb::AbstractMatrix{T}) where {T<:AbstractFloat}
    Sba = Matrix{T}(Sab')
    Mm = Sab * (robust_inv(Sbb) * Sba)
    Mm = (Mm + Mm') / 2
    L = safe_cholesky(Matrix{T}(Saa); silent=true)   # Saa ≈ L L'
    Linv = robust_inv(Matrix{T}(L))
    W = Linv * Mm * Linv'
    W = (W + W') / 2
    e = eigen(Symmetric(W))
    idx = sortperm(e.values, rev=true)
    vals = clamp.(e.values[idx], zero(T), one(T) - eps(T))
    vecs = Matrix{T}(Linv' * e.vectors[:, idx])       # back to Saa-space
    (vals, vecs)
end

# Phillips normalization: first r rows form the identity (matches estimate_vecm).
_phillips_normalize(B::AbstractMatrix{T}, r::Int) where {T} = B * robust_inv(Matrix{T}(B[1:r, :]))

# LR statistic  T Σ_{i=1}^r ln[(1−λ*_i)/(1−λ_i)]  (paired descending; clamped ≥ 0).
function _vecm_lr(lamstar::AbstractVector{T}, lam::AbstractVector{T}, r::Int, Teff::Int) where {T}
    s = sum(log((one(T) - lamstar[i]) / (one(T) - lam[i])) for i in 1:r)
    max(T(Teff) * s, zero(T))
end

function _chisq_pvalue(lr::T, df::Int) where {T<:AbstractFloat}
    df <= 0 && return one(T)
    T(ccdf(Chisq(df), max(Float64(lr), 0.0)))
end

# Unrestricted reduced-rank eigenvalues/β-vectors and the moment matrices.
function _vecm_unrestricted_eigen(m::VECMModel{T}) where {T}
    mom = _johansen_moments(m)
    S10 = Matrix{T}(mom.S01')
    vals, vecs = _grr_eigen(mom.S11, S10, mom.S00)
    (vals, vecs, mom)
end

# Restricted eigenvalues for α = Aψ (β free) via the CONDITIONAL (partial-system)
# reduced-rank regression: the A_⊥ innovations A_⊥'R0 have no error-correction, so
# condition the A-block A'R0 (and R1) on them and solve the reduced-rank problem in
# the conditional moments. Closed-form and identical to the switching MLE.
function _alpha_restricted_eigen(mom, A::AbstractMatrix{T}, r::Int) where {T}
    R0 = mom.R0; R1 = mom.R1; Teff = mom.T_eff
    Aperp = nullspace(Matrix{T}(A'))                       # n × (n-a)
    if size(Aperp, 2) == 0                                 # a = n: no restriction
        S10 = Matrix{T}(mom.S01')
        vals, _ = _grr_eigen(mom.S11, S10, mom.S00)
        return vals
    end
    w = R0 * Aperp                                         # Teff × (n-a): A_⊥'R0
    Pw = w * (robust_inv(w'w) * w')
    R0a = (R0 * A) - Pw * (R0 * A)                         # Teff × a, conditioned
    R1e = R1 - Pw * R1                                     # Teff × n, conditioned
    Skk = (R0a'R0a) / Teff
    S1k = (R1e'R0a) / Teff                                 # n × a
    S11e = (R1e'R1e) / Teff
    vals, _ = _grr_eigen(S11e, S1k, Skk)                   # |λ S11.w − S1k Skk⁻¹ Sk1|
    vals
end

# Resolve variable references (Int / String / Symbol, or vectors thereof) to indices.
function _vecm_varindices(m::VECMModel, vars)
    vv = vars isa Union{AbstractVector,Tuple} ? collect(vars) : [vars]
    idx = Int[]
    for v in vv
        if v isa Integer
            push!(idx, Int(v))
        else
            name = String(v)
            j = findfirst(==(name), m.varnames)
            j === nothing && throw(ArgumentError("Variable '$name' not found. Available: $(m.varnames)"))
            push!(idx, j)
        end
    end
    idx
end

# Rebuild a VECMModel with β fixed to `beta_r`; α, Γ, μ, Σ re-estimated by OLS.
# If `alpha_basis` A is given, α is restricted to span(A) (projection) and the
# short-run/deterministic block is refit holding α fixed.
function _refit_vecm_beta(m::VECMModel{T}, beta_r::AbstractMatrix{T};
                          alpha_basis::Union{Nothing,AbstractMatrix{T}}=nothing) where {T}
    n = nvars(m); r = m.rank; p = m.p
    mom = _johansen_moments(m)
    Y_lag = mom.Y_lag; dY_lags = mom.dY_lags; dY_eff = mom.dY_eff; T_eff = mom.T_eff
    br = Matrix{T}(beta_r)
    ecm = Y_lag * br                                   # T_eff × r

    has_const = m.deterministic ∈ (:constant, :trend)
    has_trend = m.deterministic == :trend

    # short-run + deterministic design (no ECM)
    Xsr = dY_lags
    has_const && (Xsr = size(Xsr, 2) > 0 ? hcat(Xsr, ones(T, T_eff)) : ones(T, T_eff, 1))
    has_trend && (Xsr = hcat(Xsr, T.(1:T_eff)))

    RHS = size(Xsr, 2) > 0 ? hcat(ecm, Xsr) : ecm
    B_full = robust_inv(RHS'RHS) * (RHS' * dY_eff)
    alpha = Matrix{T}(B_full[1:r, :]')                 # n × r

    if alpha_basis !== nothing
        A = Matrix{T}(alpha_basis)
        alpha = A * (robust_inv(A'A) * (A' * alpha))   # project α onto span(A)
        target = dY_eff - ecm * alpha'
        if size(Xsr, 2) > 0
            Bsr = robust_inv(Xsr'Xsr) * (Xsr' * target)
            U = target - Xsr * Bsr
            Gamma, mu = _split_shortrun(Bsr, n, p, has_const)
        else
            U = target; Gamma = Matrix{T}[]; mu = zeros(T, n)
        end
    else
        U = dY_eff - RHS * B_full
        Bsr = size(Xsr, 2) > 0 ? B_full[r+1:end, :] : zeros(T, 0, n)
        Gamma, mu = _split_shortrun(Bsr, n, p, has_const)
    end

    Pi = alpha * br'
    Sigma = (U'U) / T_eff
    log_det = logdet_safe(Sigma)
    k_total = r * n + n * n * (p - 1) + (has_const ? n : 0) + (has_trend ? n : 0)
    loglik = -T(T_eff * n / 2) * log(T(2π)) - T(T_eff / 2) * log_det - T(T_eff * n / 2)
    aic_val = log_det + 2 * k_total / T_eff
    bic_val = log_det + k_total * log(T_eff) / T_eff
    hqic_val = log_det + 2 * k_total * log(log(T_eff)) / T_eff

    VECMModel{T}(m.Y, p, r, alpha, br, Pi, Gamma, mu, U, Sigma,
                 aic_val, bic_val, hqic_val, loglik,
                 m.deterministic, m.method, m.johansen_result, m.varnames)
end

# Split a short-run+deterministic OLS coefficient block into (Γ list, μ).
function _split_shortrun(Bsr::AbstractMatrix{T}, n::Int, p::Int, has_const::Bool) where {T}
    Gamma = Matrix{T}[]
    c = 1
    for _ in 1:(p-1)
        push!(Gamma, Matrix{T}(Bsr[c:c+n-1, :]'))
        c += n
    end
    mu = has_const ? vec(Bsr[c, :]) : zeros(T, n)
    (Gamma, mu)
end

"""
    test_beta_restriction(m::VECMModel, H) -> VECMRestrictionTest

Johansen LR test that every cointegrating vector lies in a known space,
`β = Hφ`, where `H` is `p × s` (`p = nvars(m)`, `s ≥ r = cointegrating_rank(m)`).

The restricted eigenvalues `λ*_i` solve the transformed eigenproblem
`|λ H′S₁₁H − H′S₁₀ S₀₀⁻¹ S₀₁H| = 0`; the statistic is
`LR = T Σ_{i=1}^r ln[(1−λ*_i)/(1−λ_i)] ∼ χ²(df)` with **`df = r(p − s)`**.
The non-binding case `H = Iₚ` (`s = p`) returns `LR ≈ 0`, `df = 0`.

The returned test carries a re-estimated `restricted_model` whose `β` lies in
`span(H)`, ready for `irf`/`fevd`/`historical_decomposition`.
"""
function test_beta_restriction(m::VECMModel{T}, H::AbstractMatrix) where {T<:AbstractFloat}
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("β restriction test requires cointegrating rank ≥ 1 (got r=$r)"))
    Hc = Matrix{T}(H)
    size(Hc, 1) == n || throw(DimensionMismatch("H must have $n rows (nvars), got $(size(Hc,1))"))
    s = size(Hc, 2)
    s >= r || throw(ArgumentError("H must have at least r=$r columns, got s=$s"))
    lam, _, mom = _vecm_unrestricted_eigen(m)
    S10 = Matrix{T}(mom.S01')
    lamstar, phivecs = _grr_eigen(Hc' * mom.S11 * Hc, Hc' * S10, mom.S00)
    df = r * (n - s)
    lr = _vecm_lr(lamstar, lam, r, mom.T_eff)
    pval = _chisq_pvalue(lr, df)
    beta_r = _phillips_normalize(Hc * phivecs[:, 1:r], r)
    rmodel = _refit_vecm_beta(m, beta_r)
    VECMRestrictionTest{T}(:beta, lr, df, pval, r,
        "β = Hφ (restricted to span(H), s=$s)", beta_r, m.beta,
        lamstar[1:r], lam[1:r], true, rmodel)
end

"""
    test_alpha_restriction(m::VECMModel, A) -> VECMRestrictionTest

Johansen LR test that all adjustment vectors lie in a known space, `α = Aψ`,
where `A` is `p × a` (`a ≥ r`). Solved on the dual reduced-rank device
`|λ A′S₀₀A − A′S₀₁ S₁₁⁻¹ S₁₀A| = 0`; `LR = T Σ_{i=1}^r ln[(1−λ*_i)/(1−λ_i)]
∼ χ²(df)` with **`df = r(p − a)`**.

The `restricted_model` keeps the (superconsistent) unrestricted `β` and projects
`α` onto `span(A)`.
"""
function test_alpha_restriction(m::VECMModel{T}, A::AbstractMatrix) where {T<:AbstractFloat}
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("α restriction test requires cointegrating rank ≥ 1 (got r=$r)"))
    Ac = Matrix{T}(A)
    size(Ac, 1) == n || throw(DimensionMismatch("A must have $n rows (nvars), got $(size(Ac,1))"))
    a = size(Ac, 2)
    a >= r || throw(ArgumentError("A must have at least r=$r columns, got a=$a"))
    lam, _, mom = _vecm_unrestricted_eigen(m)
    lamstar = _alpha_restricted_eigen(mom, Ac, r)
    df = r * (n - a)
    lr = _vecm_lr(lamstar, lam, r, mom.T_eff)
    pval = _chisq_pvalue(lr, df)
    rmodel = _refit_vecm_beta(m, m.beta; alpha_basis=Ac)
    VECMRestrictionTest{T}(:alpha, lr, df, pval, r,
        "α = Aψ (restricted to span(A), a=$a)", m.beta, m.beta,
        lamstar[1:r], lam[1:r], true, rmodel)
end

"""
    test_weak_exogeneity(m::VECMModel, vars) -> VECMRestrictionTest

Test weak exogeneity of the named `vars` (indices, names, or a vector thereof)
for the long-run parameters — the headline central-bank question, e.g. *is the
policy rate weakly exogenous for the cointegrating relations?* It is the leading
`α`-restriction special case in which the `α` rows of `vars` are zero; `A` selects
the complementary (error-correcting) rows. With `m = length(vars)`,
**`df = r·m`**.
"""
function test_weak_exogeneity(m::VECMModel{T}, vars) where {T<:AbstractFloat}
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("weak-exogeneity test requires cointegrating rank ≥ 1 (got r=$r)"))
    ex_idx = _vecm_varindices(m, vars)
    all(1 .<= ex_idx .<= n) || throw(ArgumentError("variable index out of range 1:$n"))
    keep = setdiff(1:n, ex_idx)                        # error-correcting rows
    isempty(keep) && throw(ArgumentError("cannot make all variables weakly exogenous"))
    A = zeros(T, n, length(keep))
    for (j, k) in enumerate(keep); A[k, j] = one(T); end
    base = test_alpha_restriction(m, A)
    labels = join([m.varnames[i] for i in ex_idx], ", ")
    VECMRestrictionTest{T}(:weak_exogeneity, base.lr_stat, base.df, base.pvalue, r,
        "Weak exogeneity of {$labels} (α rows = 0, df = r·m)",
        base.beta_restricted, base.beta_unrestricted,
        base.eigenvalues_restricted, base.eigenvalues_unrestricted,
        base.converged, base.restricted_model)
end

"""
    test_known_beta(m::VECMModel, b) -> VECMRestrictionTest

Johansen LR test of a fully specified cointegrating space `β = b`, where `b` is
`p × r`. Uses the transformed eigenproblem with `H = b` (degenerate `s = r`
case); **`df = r(p − r)`**. Setting `b = m.beta` (the estimated β) returns
`LR ≈ 0`.
"""
function test_known_beta(m::VECMModel{T}, b::AbstractMatrix) where {T<:AbstractFloat}
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("known-β test requires cointegrating rank ≥ 1 (got r=$r)"))
    bc = Matrix{T}(b)
    size(bc, 1) == n || throw(DimensionMismatch("b must have $n rows (nvars), got $(size(bc,1))"))
    size(bc, 2) == r || throw(DimensionMismatch("b must have exactly r=$r columns, got $(size(bc,2))"))
    lam, _, mom = _vecm_unrestricted_eigen(m)
    S10 = Matrix{T}(mom.S01')
    lamstar, _ = _grr_eigen(bc' * mom.S11 * bc, bc' * S10, mom.S00)
    df = r * (n - r)
    lr = _vecm_lr(lamstar, lam, r, mom.T_eff)
    pval = _chisq_pvalue(lr, df)
    beta_r = _phillips_normalize(bc, r)
    rmodel = _refit_vecm_beta(m, beta_r)
    VECMRestrictionTest{T}(:known_beta, lr, df, pval, r,
        "β = b (fully known cointegrating space)", beta_r, m.beta,
        lamstar[1:r], lam[1:r], true, rmodel)
end

"""
    test_joint_restriction(m::VECMModel, H, A; maxiter=1000, tol=1e-8) -> VECMRestrictionTest

Joint LR test of `β = Hφ` **and** `α = Aψ`, estimated by the Johansen–Juselius
switching algorithm (alternate: β given α, then α given β, until the restricted
log-likelihood change `< tol`). The statistic compares the restricted and
unrestricted Gaussian log-likelihoods, `LR = T (ln|Ω̂_r| − ln|Ω̂_u|) ∼ χ²(df)`
with **`df = r(p − s) + r(p − a)`**. Non-convergence within `maxiter` sets
`converged = false` and emits a warning.
"""
function test_joint_restriction(m::VECMModel{T}, H::AbstractMatrix, A::AbstractMatrix;
                                maxiter::Int=1000, tol::Real=1e-8) where {T<:AbstractFloat}
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("joint restriction test requires cointegrating rank ≥ 1 (got r=$r)"))
    Hc = Matrix{T}(H); Ac = Matrix{T}(A)
    size(Hc, 1) == n || throw(DimensionMismatch("H must have $n rows, got $(size(Hc,1))"))
    size(Ac, 1) == n || throw(DimensionMismatch("A must have $n rows, got $(size(Ac,1))"))
    s = size(Hc, 2); a = size(Ac, 2)
    (s >= r && a >= r) || throw(ArgumentError("need s ≥ r and a ≥ r (got s=$s, a=$a, r=$r)"))

    lam, betavecs, mom = _vecm_unrestricted_eigen(m)
    S00 = mom.S00; S11 = mom.S11; S01 = mom.S01; S10 = Matrix{T}(S01'); Teff = mom.T_eff
    logdet_u = logdet_safe(S00) + sum(log(one(T) - lam[i]) for i in 1:r)

    # initialize β by projecting the unrestricted β onto span(H)
    beta = _phillips_normalize(Hc * (robust_inv(Hc'Hc) * (Hc' * betavecs[:, 1:r])), r)
    Omega = Matrix{T}(S00)
    ll_prev = T(-Inf)
    converged = false
    local alpha = zeros(T, n, r)

    for _ in 1:maxiter
        # α = Aψ given (β, Ω)
        Oi = robust_inv(Omega)
        bSb = beta' * S11 * beta
        rhs_a = Ac' * Oi * (S01 * beta)
        psi = reshape(robust_inv(kron(bSb, Ac' * Oi * Ac)) * vec(rhs_a), a, r)
        alpha = Ac * psi
        Omega = S00 - alpha * (beta' * S10) - (S01 * beta) * alpha' + alpha * bSb * alpha'
        Omega = (Omega + Omega') / 2

        # β = Hφ given (α, Ω)
        Oi = robust_inv(Omega)
        aOa = alpha' * Oi * alpha
        rhs_b = Hc' * S10 * Oi * alpha
        phi = reshape(robust_inv(kron(aOa, Hc' * S11 * Hc)) * vec(rhs_b), s, r)
        beta = _phillips_normalize(Hc * phi, r)

        Omega = S00 - alpha * (beta' * S10) - (S01 * beta) * alpha' + alpha * (beta' * S11 * beta) * alpha'
        Omega = (Omega + Omega') / 2
        ll = -logdet_safe(Omega)
        if abs(ll - ll_prev) < tol
            converged = true
            break
        end
        ll_prev = ll
    end
    converged || @warn "test_joint_restriction: switching algorithm did not converge in $maxiter iterations"

    lr = max(T(Teff) * (logdet_safe(Omega) - logdet_u), zero(T))
    df = r * (n - s) + r * (n - a)
    pval = _chisq_pvalue(lr, df)
    beta_r = _phillips_normalize(beta, r)
    rmodel = _refit_vecm_beta(m, beta_r; alpha_basis=Ac)
    VECMRestrictionTest{T}(:joint, lr, df, pval, r,
        "Joint β=Hφ (s=$s), α=Aψ (a=$a) via switching", beta_r, m.beta,
        T[], lam[1:r], converged, rmodel)
end
