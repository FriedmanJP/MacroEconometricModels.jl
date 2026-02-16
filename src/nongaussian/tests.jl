# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

"""
Identifiability and specification tests for non-Gaussian SVAR.

Tests whether non-Gaussian identification conditions hold, whether recovered shocks
are non-Gaussian and independent, and model specification tests.

References:
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
function _dcov_independence_test(shocks::Matrix{T}) where {T<:AbstractFloat}
    T_obs, n = size(shocks)
    stat = zero(T)

    for i in 1:n-1, j in (i+1):n
        dcov = _distance_covariance(@view(shocks[:, i]), @view(shocks[:, j]))
        stat += T_obs * dcov
    end

    # Approximate p-value via permutation
    n_perm = 199
    count_ge = 0
    for _ in 1:n_perm
        shocks_perm = copy(shocks)
        for j in 2:n
            shocks_perm[:, j] = shocks_perm[randperm(T_obs), j]
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
                                       n_bootstrap::Int=999) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)

    # Get reference B₀
    ref_result = if method == :fastica
        identify_fastica(model)
    elseif method == :jade
        identify_jade(model)
    elseif method == :sobi
        identify_sobi(model)
    else
        identify_fastica(model)
    end
    B0_ref = ref_result.B0

    # Bootstrap
    distances = T[]
    for _ in 1:n_bootstrap
        idx = rand(1:T_obs, T_obs)
        U_boot = model.U[idx, :]
        Sigma_boot = cov(U_boot)

        # Create bootstrap model
        boot_model = VARModel(model.Y, model.p, model.B, U_boot, Sigma_boot,
                              model.aic, model.bic, model.hqic)

        try
            boot_result = if method == :fastica
                identify_fastica(boot_model)
            elseif method == :jade
                identify_jade(boot_model)
            elseif method == :sobi
                identify_sobi(boot_model)
            else
                identify_fastica(boot_model)
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
function test_shock_independence(result::ICASVARResult{T}; max_lag::Int=10) where {T<:AbstractFloat}
    _test_independence_impl(result.shocks, max_lag)
end

function test_shock_independence(result::NonGaussianMLResult{T}; max_lag::Int=10) where {T<:AbstractFloat}
    _test_independence_impl(result.shocks, max_lag)
end

function _test_independence_impl(shocks::Matrix{T}, max_lag::Int) where {T<:AbstractFloat}
    # Cross-correlation test
    cc_stat, cc_pval, cc_df = _cross_correlation_test(shocks, max_lag)

    # Distance covariance test (on subset for speed)
    T_obs = size(shocks, 1)
    if T_obs > 500
        idx = randperm(T_obs)[1:500]
        shocks_sub = shocks[idx, :]
    else
        shocks_sub = shocks
    end
    dcov_stat, dcov_pval = _dcov_independence_test(shocks_sub)

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
                                  n_bootstrap::Int=499) where {T<:AbstractFloat}
    n = nvars(model)
    T_obs = size(model.U, 1)

    B0 = result.B0
    Q = result.Q

    # Compute residual from Σ = B₀ B₀'
    L = safe_cholesky(model.Sigma)
    Sigma_model = B0 * B0'
    discrepancy = norm(Sigma_model - model.Sigma) / norm(model.Sigma)

    # Check orthogonality of Q
    orth_err = norm(Q' * Q - I)

    stat = discrepancy + orth_err

    # Bootstrap under the null
    boot_stats = T[]
    for _ in 1:n_bootstrap
        idx = rand(1:T_obs, T_obs)
        U_boot = model.U[idx, :]
        Sigma_boot = cov(U_boot) + eps(T) * I

        L_boot = safe_cholesky(Sigma_boot)
        B0_boot = Matrix(L_boot) * Q
        Sigma_model_boot = B0_boot * B0_boot'
        disc_boot = norm(Sigma_model_boot - Sigma_boot) / norm(Sigma_boot)
        push!(boot_stats, disc_boot)
    end

    pval = mean(boot_stats .>= stat)
    identified = pval >= T(0.05)  # fail to reject → restrictions OK

    IdentifiabilityTestResult{T}(:overidentification, stat, T(pval), identified,
                                  Dict{Symbol, Any}(:discrepancy => discrepancy,
                                                     :orthogonality_error => orth_err,
                                                     :n_bootstrap => n_bootstrap))
end
