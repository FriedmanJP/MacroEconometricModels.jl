# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Identifiability and specification tests for statistical SVAR identification.

Tests whether non-Gaussian / heteroskedasticity-based identification conditions hold,
whether recovered shocks are non-Gaussian and independent, and model specification tests.

Note: Weak identification is an important concern when variances change little or
deviations from Gaussianity are small (Lewis 2022). Standard Wald tests may have
poor size properties in such cases.

References:
- Lewis, D. J. (2025). "Identification based on higher moments in macroeconometrics."
- Lewis, D. J. (2022). "Robust inference in models identified via heteroskedasticity."
- Lanne, M., Meitz, M. & Saikkonen, P. (2017). "Identification and estimation of non-Gaussian SVAR."
- Herwartz, H. & Plödt, M. (2016). "The macroeconomic effects of oil price shocks."
"""

using LinearAlgebra, Statistics, Distributions, Random

# =============================================================================
# Result Type
# =============================================================================

"""
    IdentifiabilityTestResult{T}

Result from an identifiability or specification test.

Fields:
- `test_name::Symbol` — test identifier
- `statistic::T` — test statistic
- `pvalue::T` — p-value
- `identified::Bool` — whether identification appears to hold
- `details::Dict{Symbol, Any}` — method-specific details
"""
struct IdentifiabilityTestResult{T<:AbstractFloat}
    test_name::Symbol
    statistic::T
    pvalue::T
    identified::Bool
    details::Dict{Symbol, Any}
end

function Base.show(io::IO, r::IdentifiabilityTestResult{T}) where {T}
    stars = _significance_stars(r.pvalue)
    status_str = r.identified ? "Identified" : "Not identified"
    data = Any[
        "H₀"            "Structural shocks are not identified";
        "H₁"            "Structural shocks are identified";
        "Test Statistic" string(_fmt(r.statistic), " ", stars);
        "P-value"        _format_pvalue(r.pvalue);
        "Status"         status_str
    ]
    _pretty_table(io, data;
        title = "Identifiability Test: $(r.test_name)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    conc_data = Any["Conclusion" (r.identified ? "Evidence supports identification" : "Identification conditions may not hold");
                    "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# Internal Helpers
# =============================================================================

"""Procrustes distance: minimum ||P B₁ - B₂||_F over signed permutations P."""
function _procrustes_distance(B1::Matrix{T}, B2::Matrix{T}) where {T<:AbstractFloat}
    n = size(B1, 1)

    # Try all column permutations (feasible for small n)
    # For n > 5, use Hungarian algorithm approximation
    if n <= 5
        min_dist = T(Inf)
        for perm in _permutations(n)
            for signs in Iterators.product(fill([-1, 1], n)...)
                B1_perm = B1[:, perm] .* collect(signs)'
                d = norm(B1_perm - B2)
                min_dist = min(min_dist, d)
            end
        end
        return min_dist
    else
        # Greedy matching by column correlation
        B1_matched = copy(B1)
        used = Set{Int}()
        for j in 1:n
            best_k, best_corr = 0, T(-Inf)
            for k in 1:n
                k in used && continue
                c = abs(dot(@view(B1[:, k]), @view(B2[:, j])))
                if c > best_corr
                    best_k, best_corr = k, c
                end
            end
            push!(used, best_k)
            s = sign(dot(@view(B1[:, best_k]), @view(B2[:, j])))
            B1_matched[:, j] = s * @view(B1[:, best_k])
        end
        return norm(B1_matched - B2)
    end
end

"""Generate all permutations of 1:n."""
function _permutations(n::Int)
    if n == 1
        return [[1]]
    end
    result = Vector{Int}[]
    for p in _permutations(n - 1)
        for i in 1:n
            new_p = copy(p)
            insert!(new_p, i, n)
            push!(result, new_p)
        end
    end
    result
end

"""Cross-correlation test for independence of shock series."""
function _cross_correlation_test(shocks::Matrix{T}, max_lag::Int) where {T<:AbstractFloat}
    T_obs, n = size(shocks)
    stat = zero(T)

    for i in 1:n-1, j in (i+1):n
        for lag in 0:max_lag
            if lag == 0
                r = cor(@view(shocks[:, i]), @view(shocks[:, j]))
            else
                r = cor(@view(shocks[lag+1:end, i]), @view(shocks[1:end-lag, j]))
            end
            stat += T_obs * r^2
        end
    end

    df = n * (n - 1) ÷ 2 * (max_lag + 1)
    pval = 1.0 - cdf(Chisq(df), stat)
    (stat, pval, df)
end

"""Distance covariance independence test on all shock pairs."""
function _dcov_independence_test(shocks::Matrix{T};
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    T_obs, n = size(shocks)
    stat = zero(T)

    for i in 1:n-1, j in (i+1):n
        dcov = _distance_covariance(@view(shocks[:, i]), @view(shocks[:, j]))
        stat += T_obs * dcov
    end

    # Approximate p-value via permutation (seeded for reproducibility)
    n_perm = 199
    count_ge = 0
    for _ in 1:n_perm
        shocks_perm = copy(shocks)
        for j in 2:n
            shocks_perm[:, j] = shocks_perm[randperm(rng, T_obs), j]
        end
        stat_perm = zero(T)
        for i in 1:n-1, j in (i+1):n
            dcov = _distance_covariance(@view(shocks_perm[:, i]), @view(shocks_perm[:, j]))
            stat_perm += T_obs * dcov
        end
        stat_perm >= stat && (count_ge += 1)
    end

    pval = (count_ge + 1) / (n_perm + 1)
    (stat, T(pval))
end

# =============================================================================
# Public API: Test Identification Strength
# =============================================================================

"""
    test_identification_strength(model::VARModel; method=:fastica,
                                 n_bootstrap=999) -> IdentifiabilityTestResult

Test the strength of non-Gaussian identification via bootstrap.

Resamples residuals with replacement, re-estimates B₀, and computes the Procrustes
distance between bootstrap and original B₀. Small distances indicate strong identification.

Returns: test statistic = median Procrustes distance, p-value from distribution.
"""
function test_identification_strength(model::VARModel{T}; method::Symbol=:fastica,
                                       n_bootstrap::Int=999,
                                       rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)

    # Get reference B₀
    ref_result = if method == :fastica
        identify_fastica(model; rng=rng)
    elseif method == :jade
        identify_jade(model)
    elseif method == :sobi
        identify_sobi(model)
    else
        identify_fastica(model; rng=rng)
    end
    B0_ref = ref_result.B0

    # Bootstrap
    distances = T[]
    for _ in 1:n_bootstrap
        idx = rand(rng, 1:T_obs, T_obs)
        U_boot = model.U[idx, :]
        Sigma_boot = cov(U_boot)

        # Create bootstrap model
        boot_model = VARModel(model.Y, model.p, model.B, U_boot, Sigma_boot,
                              model.aic, model.bic, model.hqic)

        try
            boot_result = if method == :fastica
                identify_fastica(boot_model; rng=rng)
            elseif method == :jade
                identify_jade(boot_model)
            elseif method == :sobi
                identify_sobi(boot_model)
            else
                identify_fastica(boot_model; rng=rng)
            end
            push!(distances, _procrustes_distance(boot_result.B0, B0_ref))
        catch
            continue
        end
    end

    if isempty(distances)
        return IdentifiabilityTestResult{T}(:identification_strength, T(NaN), T(NaN), false,
                                             Dict{Symbol, Any}(:method => method, :n_bootstrap => 0))
    end

    med_dist = median(distances)
    # Identification is "strong" if median distance is small relative to ||B₀||
    normalized_dist = med_dist / norm(B0_ref)
    identified = normalized_dist < T(0.5)

    # p-value: fraction of bootstrap distances exceeding threshold
    threshold = T(0.5) * norm(B0_ref)
    pval = mean(distances .> threshold)

    IdentifiabilityTestResult{T}(:identification_strength, med_dist, T(pval), identified,
                                  Dict{Symbol, Any}(:method => method,
                                                     :n_bootstrap => length(distances),
                                                     :normalized_distance => normalized_dist,
                                                     :distances => distances))
end

# =============================================================================
# Public API: Test Shock Gaussianity
# =============================================================================

"""
    test_shock_gaussianity(result::ICASVARResult) -> IdentifiabilityTestResult
    test_shock_gaussianity(result::NonGaussianMLResult) -> IdentifiabilityTestResult

Test whether recovered structural shocks are non-Gaussian using univariate JB tests.

Non-Gaussian identification requires at most one shock to be Gaussian. This test
checks each shock individually and reports the joint result.

At most one Gaussian shock → identification holds.
"""
function test_shock_gaussianity(result::ICASVARResult{T}) where {T<:AbstractFloat}
    _test_shock_gaussianity_impl(result.shocks, result.method)
end

function test_shock_gaussianity(result::NonGaussianMLResult{T}) where {T<:AbstractFloat}
    _test_shock_gaussianity_impl(result.shocks, result.distribution)
end

function _test_shock_gaussianity_impl(shocks::Matrix{T}, method::Symbol) where {T<:AbstractFloat}
    T_obs, n = size(shocks)
    jb_stats = T[]
    jb_pvals = T[]

    for j in 1:n
        s = @view shocks[:, j]
        s_std = (s .- mean(s)) / std(s)
        skew = mean(s_std .^ 3)
        kurt = mean(s_std .^ 4) - T(3)
        jb = T_obs * (skew^2 / T(6) + kurt^2 / T(24))
        pval = 1.0 - cdf(Chisq(2), jb)
        push!(jb_stats, jb)
        push!(jb_pvals, T(pval))
    end

    # Count how many shocks fail to reject Gaussianity at 5%
    n_gaussian = sum(jb_pvals .>= T(0.05))
    identified = n_gaussian <= 1  # At most one Gaussian is OK

    # Joint statistic
    joint_stat = sum(jb_stats)
    joint_pval = 1.0 - cdf(Chisq(2n), joint_stat)

    IdentifiabilityTestResult{T}(:shock_gaussianity, joint_stat, T(joint_pval), identified,
                                  Dict{Symbol, Any}(:jb_stats => jb_stats,
                                                     :jb_pvals => jb_pvals,
                                                     :n_gaussian => n_gaussian,
                                                     :method => method))
end

# =============================================================================
# Public API: Gaussian vs Non-Gaussian LR Test
# =============================================================================

"""
    test_gaussian_vs_nongaussian(model::VARModel; distribution=:student_t) -> IdentifiabilityTestResult

Likelihood ratio test: H₀ Gaussian vs H₁ non-Gaussian structural shocks.

Under H₀, the LR statistic LR = 2(ℓ₁ - ℓ₀) ~ χ²(n_extra_params).
"""
function test_gaussian_vs_nongaussian(model::VARModel{T};
                                       distribution::Symbol=:student_t) where {T<:AbstractFloat}
    result = identify_nongaussian_ml(model; distribution=distribution)

    LR = T(2) * (result.loglik - result.loglik_gaussian)
    LR = max(LR, zero(T))

    n = nvars(model)
    n_extra = _n_dist_params(distribution) * n
    pval = 1.0 - cdf(Chisq(n_extra), LR)
    identified = pval < T(0.05)

    IdentifiabilityTestResult{T}(:gaussian_vs_nongaussian, LR, T(pval), identified,
                                  Dict{Symbol, Any}(:distribution => distribution,
                                                     :loglik_nongaussian => result.loglik,
                                                     :loglik_gaussian => result.loglik_gaussian,
                                                     :df => n_extra))
end

# =============================================================================
# Public API: Shock Independence Test
# =============================================================================

"""
    test_shock_independence(result::ICASVARResult; max_lag=10) -> IdentifiabilityTestResult
    test_shock_independence(result::NonGaussianMLResult; max_lag=10) -> IdentifiabilityTestResult

Test independence of recovered structural shocks.

Uses both cross-correlation (portmanteau) and distance covariance tests.
Independence is a necessary condition for valid identification.
"""
function test_shock_independence(result::ICASVARResult{T}; max_lag::Int=10,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    _test_independence_impl(result.shocks, max_lag; rng=rng)
end

function test_shock_independence(result::NonGaussianMLResult{T}; max_lag::Int=10,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    _test_independence_impl(result.shocks, max_lag; rng=rng)
end

function _test_independence_impl(shocks::Matrix{T}, max_lag::Int;
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    # Cross-correlation test
    cc_stat, cc_pval, cc_df = _cross_correlation_test(shocks, max_lag)

    # Distance covariance test (on subset for speed)
    T_obs = size(shocks, 1)
    if T_obs > 500
        idx = randperm(rng, T_obs)[1:500]
        shocks_sub = shocks[idx, :]
    else
        shocks_sub = shocks
    end
    dcov_stat, dcov_pval = _dcov_independence_test(shocks_sub; rng=rng)

    # Combined: use Fisher's method
    # χ² = -2 Σ log(pᵢ)
    pvals = [max(cc_pval, eps()), max(Float64(dcov_pval), eps())]
    fisher_stat = -2.0 * sum(log, pvals)
    fisher_pval = 1.0 - cdf(Chisq(2 * length(pvals)), fisher_stat)
    identified = fisher_pval >= 0.05  # fail to reject independence

    IdentifiabilityTestResult{T}(:shock_independence, T(fisher_stat), T(fisher_pval), identified,
                                  Dict{Symbol, Any}(:cc_statistic => cc_stat,
                                                     :cc_pvalue => cc_pval,
                                                     :cc_df => cc_df,
                                                     :dcov_statistic => dcov_stat,
                                                     :dcov_pvalue => dcov_pval,
                                                     :max_lag => max_lag))
end

# =============================================================================
# Public API: Overidentification Test
# =============================================================================

"""
    test_overidentification(model::VARModel, result::AbstractNonGaussianSVAR;
                            restrictions=nothing, n_bootstrap=499) -> IdentifiabilityTestResult

Test overidentifying restrictions for non-Gaussian SVAR.

When additional restrictions beyond non-Gaussianity are imposed (e.g., zero restrictions
on B₀), this test checks whether those restrictions are consistent with the data.

Uses a bootstrap approach: compares the restricted log-likelihood to bootstrap distribution.
"""
function test_overidentification(model::VARModel{T}, result::AbstractNonGaussianSVAR;
                                  restrictions::Union{Nothing, Function}=nothing,
                                  n_bootstrap::Int=499,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)

    B0 = result.B0
    Q = result.Q

    # Compute residual from Σ = B₀ B₀'
    Sigma_model = B0 * B0'
    discrepancy = norm(Sigma_model - model.Sigma) / norm(model.Sigma)

    # Check orthogonality of Q
    orth_err = norm(Q' * Q - I)

    # Optional user restrictions (e.g. zero pattern on B₀). When absent the
    # test reduces to a pure covariance-fit + orthogonality check, which is
    # exactly zero for any just-identified B₀ = L·Q with Q orthogonal.
    restr_err = zero(T)
    if restrictions !== nothing
        restr_err = T(restrictions(B0, Q))
    end

    # Common statistic used for both the sample and the bootstrap.
    _oid_stat(disc, orth, rerr) = disc + orth + rerr
    stat = _oid_stat(discrepancy, orth_err, restr_err)

    # Just-identified B₀ with no extra restrictions: the statistic is
    # machine-epsilon by construction and the test has no content.
    just_id = restrictions === nothing && discrepancy < T(1e-10) && orth_err < T(1e-10)
    if just_id
        @warn "test_overidentification: B₀ is just-identified (Σ ≈ B₀B₀', Q orthogonal) " *
              "with no extra restrictions; the test has no power. " *
              "Pass `restrictions=f` that returns a non-negative discrepancy, " *
              "or use an overidentified estimator."
        return IdentifiabilityTestResult{T}(:overidentification, stat, one(T), true,
                                            Dict{Symbol, Any}(:discrepancy => discrepancy,
                                                               :orthogonality_error => orth_err,
                                                               :restriction_error => restr_err,
                                                               :n_bootstrap => 0,
                                                               :just_identified => true))
    end

    # Bootstrap the SAME statistic under residual resampling with fixed Q.
    boot_stats = Vector{T}(undef, n_bootstrap)
    rerr_boots = Vector{T}(undef, n_bootstrap)
    for b in 1:n_bootstrap
        idx = rand(rng, 1:T_obs, T_obs)
        U_boot = model.U[idx, :]
        Sigma_boot = cov(U_boot; corrected=true)
        # Symmetrize / regularize
        Sigma_boot = (Sigma_boot + Sigma_boot') / 2 + T(1e-12) * I

        L_boot = safe_cholesky(Sigma_boot)
        B0_boot = Matrix(L_boot) * Q
        Sigma_model_boot = B0_boot * B0_boot'
        disc_boot = norm(Sigma_model_boot - Sigma_boot) / max(norm(Sigma_boot), eps(T))
        orth_boot = norm(Q' * Q - I)  # Q fixed ⇒ same orth_err
        rerr_boots[b] = restrictions !== nothing ? T(restrictions(B0_boot, Q)) : zero(T)
        boot_stats[b] = _oid_stat(disc_boot, orth_boot, rerr_boots[b])
    end

    # p-value (#568). With Q held at the sample estimate the bootstrap draws are
    # centred on the SAMPLE restriction value, so the uncentred `boot ≥ stat`
    # returned p ≈ 0.5 no matter how badly the restriction was violated. When a
    # restriction is supplied, test ONLY its component, centred at the sample
    # value (under H₀ the restriction discrepancy is 0): the disc/orth terms
    # have no bootstrap analogue — `L_boot·Q` reconstructs `Σ_boot` exactly, so
    # their bootstrap counterparts are degenerate at 0 and including them makes
    # the comparison mechanical, not statistical.
    pval = if restrictions === nothing
        mean(boot_stats .>= stat)
    else
        mean(abs.(rerr_boots .- restr_err) .>= restr_err)
    end
    identified = pval >= T(0.05)  # fail to reject → restrictions OK

    IdentifiabilityTestResult{T}(:overidentification, stat, T(pval), identified,
                                  Dict{Symbol, Any}(:discrepancy => discrepancy,
                                                     :orthogonality_error => orth_err,
                                                     :restriction_error => restr_err,
                                                     :n_bootstrap => n_bootstrap,
                                                     :just_identified => false))
end

# =============================================================================
# Heteroskedastic identification: λ-distinctness and B₀ restriction tests
# =============================================================================

function _zero_mask(restrictions, n::Int)
    size(restrictions) == (n, n) || throw(ArgumentError(
        "restrictions must be $n×$n, got $(size(restrictions))"))
    if eltype(restrictions) <: Bool
        return BitMatrix(restrictions)
    end
    mask = falses(n, n)
    @inbounds for i in 1:n, j in 1:n
        v = restrictions[i, j]
        if v isa Number && iszero(v) && !isnan(v)
            mask[i, j] = true
        end
    end
    mask
end

function _align_to_zeros(B::AbstractMatrix{T}, mask::BitMatrix) where {T<:AbstractFloat}
    n = size(B, 1)
    best = T(Inf)
    best_perm = collect(1:n)
    best_signs = ones(T, n)
    for perm in _permutations(n)
        for signs in Iterators.product(ntuple(_ -> (-one(T), one(T)), n)...)
            B2 = B[:, perm] .* collect(signs)'
            s = zero(T)
            @inbounds for i in 1:n, j in 1:n
                mask[i, j] && (s += B2[i, j]^2)
            end
            if s < best
                best = s
                best_perm = copy(perm)
                best_signs = collect(T, signs)
            end
        end
    end
    best_perm, best_signs
end

"""True if `V` can support a Wald contrast (nonempty, not all-zero, some positive var)."""
function _vcov_wald_ok(V::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(V, 1)
    (n > 0 && size(V, 2) == n) || return false
    any(x -> isfinite(x) && abs(x) > zero(T), V) || return false
    any(d -> isfinite(d) && d > T(1e-14), diag(V)) || return false
end

"""Delta-method vcov of `λ = exp(α)` from the log-λ block of `V`, or `nothing`."""
function _loglambda_delta_vcov(V::AbstractMatrix, λ::AbstractVector{T},
                                i1::Int, i2::Int) where {T<:AbstractFloat}
    nλ = length(λ)
    (i1 >= 1 && i2 == i1 + nλ - 1 && size(V, 1) >= i2 && size(V, 2) >= i2) || return nothing
    Vα = Matrix{T}(V[i1:i2, i1:i2])
    _vcov_wald_ok(Vα) || return nothing
    D = Diagonal(λ)
    Vλ = Matrix{T}(D * Vα * D)
    _vcov_wald_ok(Vλ) ? Vλ : nothing
end

function _push_lambda_block!(blocks, λ::AbstractVector{T}, Vλ) where {T<:AbstractFloat}
    Vλ === nothing && return
    push!(blocks, (Vector{T}(λ), Vλ))
    nothing
end

"""Return usable `(λ, Vλ)` blocks across regimes k=2…K (or the GARCH/ST analogue)."""
function _lambda_and_vcov(r::Union{MarkovSwitchingSVARResult{T},
                                    ExternalVolatilitySVARResult{T}}) where {T<:AbstractFloat}
    n = size(r.B0, 1)
    nB = n * n
    K = length(r.Lambda)
    K >= 2 || throw(ArgumentError("result has no relative-variance vector"))
    blocks = Tuple{Vector{T}, Matrix{T}}[]
    for k in 2:K
        λ = Vector{T}(r.Lambda[k])
        i1 = nB + (k - 2) * n + 1
        i2 = nB + (k - 1) * n
        _push_lambda_block!(blocks, λ, _loglambda_delta_vcov(r.vcov, λ, i1, i2))
    end
    isempty(blocks) && throw(ArgumentError(
        "result has no parameter covariance usable for a Wald test of λ"))
    blocks
end

function _lambda_and_vcov(r::SmoothTransitionSVARResult{T}) where {T<:AbstractFloat}
    n = size(r.B0, 1)
    n_L = n * (n + 1) ÷ 2
    n_angles = n * (n - 1) ÷ 2
    i1 = n_L + n_angles + 1
    i2 = n_L + n_angles + n
    length(r.Lambda) >= 2 || throw(ArgumentError("result has no relative-variance vector"))
    λ = Vector{T}(r.Lambda[2])
    Vλ = _loglambda_delta_vcov(r.vcov, λ, i1, i2)
    Vλ === nothing && throw(ArgumentError(
        "result has no parameter covariance usable for a Wald test of λ"))
    [(λ, Vλ)]
end

function _lambda_and_vcov(r::GARCHSVARResult{T}) where {T<:AbstractFloat}
    h = r.cond_var
    Tobs, n = size(h)
    Tobs >= 2 || throw(ArgumentError("result has no conditional variances for a Wald test of λ"))
    μ = vec(mean(h; dims=1))
    λ = μ ./ μ[1]
    Hc = h .- μ'
    Ω = (Hc' * Hc) / T(Tobs)
    Vμ = Ω / T(Tobs)
    J = zeros(T, n, n)
    @inbounds for j in 1:n
        J[j, j] = one(T) / μ[1]
        J[j, 1] -= μ[j] / μ[1]^2
    end
    Vλ = Matrix{T}(J * Vμ * J')
    _vcov_wald_ok(Vλ) || throw(ArgumentError(
        "result has no parameter covariance usable for a Wald test of λ"))
    [(λ, Vλ)]
end

"""
    test_lambda_distinct(result; pairs=:all) -> NamedTuple

Wald test of `H₀: λ_i = λ_j` for relative shock variances (LLM 2010).

`pairs=:all` tests every pair `i < j`. For K>2 discrete-regime results each pair
uses the most separating regime `k=2…K`. Bonferroni-adjusted p-values are
returned alongside the raw Wald statistics. Empty, all-zero, or not-SPD vcov
throws `ArgumentError` (or yields `NaN` stats if a contrast variance is not
positive).
"""
function test_lambda_distinct(result; pairs=:all)
    blocks = _lambda_and_vcov(result)
    Tλ = eltype(blocks[1][1])
    n = length(blocks[1][1])
    pair_list = pairs === :all ?
                [(i, j) for i in 1:(n - 1) for j in (i + 1):n] :
                collect(pairs)
    n_pairs = length(pair_list)
    n_pairs >= 1 || throw(ArgumentError("pairs must be non-empty"))
    stats = Vector{Tλ}(undef, n_pairs)
    pvals = Vector{Tλ}(undef, n_pairs)
    pbonf = Vector{Tλ}(undef, n_pairs)
    for (idx, pr) in enumerate(pair_list)
        i, j = pr
        (1 <= i <= n && 1 <= j <= n && i != j) || throw(ArgumentError(
            "pair ($i, $j) is not a valid shock pair in 1:$n"))
        best_W = Tλ(NaN)
        for (λ, Vλ) in blocks
            d = λ[i] - λ[j]
            v = Vλ[i, i] + Vλ[j, j] - 2 * Vλ[i, j]
            (isfinite(v) && v > zero(Tλ)) || continue
            W = d^2 / v
            isfinite(W) || continue
            if !isfinite(best_W) || W > best_W
                best_W = W
            end
        end
        stats[idx] = best_W
        if !isfinite(best_W)
            pvals[idx] = Tλ(NaN)
            pbonf[idx] = Tλ(NaN)
        else
            pvals[idx] = Tλ(1) - Tλ(cdf(Chisq(1), best_W))
            pbonf[idx] = min(one(Tλ), Tλ(n_pairs) * pvals[idx])
        end
    end
    (statistic=stats, pvalue=pvals, pvalue_bonferroni=pbonf, pairs=pair_list)
end

function _restriction_lr(ℓ_u::T, ℓ_r::T, df::Int) where {T<:AbstractFloat}
    LR = max(zero(T), T(2) * (ℓ_u - ℓ_r))
    df_eff = max(df, 1)
    pval = T(1) - T(cdf(Chisq(df_eff), LR))
    (statistic=LR, pvalue=pval, df=df)
end

function _aligned_B_Lambda(B0, Lambda, mask)
    perm, signs = _align_to_zeros(B0, mask)
    B_al = B0[:, perm] .* signs'
    Λ_al = [λ[perm] for λ in Lambda]
    B_al, Λ_al
end

"""
    test_restrictions(result, restrictions) -> NamedTuple

LR test of zero restrictions on `B₀`. `restrictions` is an n×n mask:
`0` = restricted to zero, `NaN`/missing = free. A `BitMatrix` treats `true`
as restricted. Columns are aligned by signed permutation before re-estimation.
"""
function test_restrictions(result::Union{MarkovSwitchingSVARResult{T},
                                          ExternalVolatilitySVARResult{T}},
                            restrictions) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    mask = _zero_mask(restrictions, n)
    Tks = if result isa ExternalVolatilitySVARResult
        T[T(length(idx)) for idx in result.regime_indices]
    else
        vec(sum(result.regime_probs; dims=1))
    end
    B_al, Λ_al = _aligned_B_Lambda(result.B0, result.Lambda, mask)
    ℓ_u = _k_regime_conc_ll(result.B0, result.Lambda, Tks, result.Sigma_regimes)
    ℓ_r, _ = _k_regime_restricted_ml(B_al, Λ_al, mask, result.Sigma_regimes, Tks)
    _restriction_lr(ℓ_u, ℓ_r, count(mask))
end

function test_restrictions(result::SmoothTransitionSVARResult{T},
                            restrictions) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    U = result.residuals
    size(U, 1) >= n || throw(ArgumentError(
        "result has no residuals; re-estimate with current identify_smooth_transition"))
    mask = _zero_mask(restrictions, n)
    free = .!mask
    n_free = count(free)
    B_al, Λ_al = _aligned_B_Lambda(result.B0, result.Lambda, mask)
    B_al[mask] .= zero(T)
    s = result.transition_var
    sigma_s = std(s)
    logγ_lo = log(T(1e-3) / sigma_s)
    logγ_hi = log(T(20) / sigma_s)
    xγ = _st_x_from_gamma(result.gamma, logγ_lo, logγ_hi)
    p0 = vcat(B_al[free], log.(max.(Λ_al[2], T(1e-8))), xγ, result.threshold)
    obj = p -> _st_restricted_nll(p, U, s, sigma_s, n, logγ_lo, logγ_hi, free, n_free)
    g! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
    res = Optim.optimize(obj, g!, p0, Optim.LBFGS(),
                         Optim.Options(iterations=200, g_tol=T(1e-8),
                                       allow_f_increases=true))
    ℓ_r = -T(Optim.minimum(res))
    _restriction_lr(result.loglik, ℓ_r, n_free == 0 ? n * n : count(mask))
end

function test_restrictions(result::GARCHSVARResult{T}, restrictions) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    U = result.shocks * result.B0'
    mask = _zero_mask(restrictions, n)
    free = .!mask
    n_free = count(free)
    perm, signs = _align_to_zeros(result.B0, mask)
    B_al = result.B0[:, perm] .* signs'
    B_al[mask] .= zero(T)
    p0 = B_al[free]
    h_al = result.cond_var[:, perm]
    obj = p -> _garch_restricted_nll(p, U, h_al, free, n_free, n)
    g! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
    res = Optim.optimize(obj, g!, p0, Optim.LBFGS(),
                         Optim.Options(iterations=200, g_tol=T(1e-8),
                                       allow_f_increases=true))
    ℓ_r = -T(Optim.minimum(res))
    _restriction_lr(result.loglik, ℓ_r, count(mask))
end
