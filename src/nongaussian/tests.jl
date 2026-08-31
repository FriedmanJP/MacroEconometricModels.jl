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
    no_pval = isnan(r.pvalue)
    stars = no_pval ? "" : _significance_stars(r.pvalue)
    is_label = r.test_name === :label_stability ||
               get(r.details, :fallback, nothing) === :label_stability
    status_str = if is_label
        r.identified ? "Labels stable" : "Labels unstable"
    else
        r.identified ? "Identified" : "Not identified"
    end
    h0 = if is_label
        "Shock column labels are unstable under resampling"
    elseif r.test_name === :overidentification
        "Overidentifying restrictions hold"
    else
        "Structural shocks are not identified"
    end
    h1 = if is_label
        "Shock column labels match the identity permutation"
    elseif r.test_name === :overidentification
        "Overidentifying restrictions are violated"
    else
        "Structural shocks are identified"
    end
    stat_label = is_label ? "Match fraction" : "Test Statistic"
    data = Any[
        "H₀"            h0;
        "H₁"            h1;
        stat_label      string(_fmt(r.statistic), stars == "" ? "" : " $stars");
        "P-value"        no_pval ? "— (not a test)" : _format_pvalue(r.pvalue);
        "Status"         status_str
    ]
    title = if get(r.details, :fallback, nothing) === :label_stability
        "Identifiability Test: $(r.test_name) (ICA fallback: label-stability)"
    else
        "Identifiability Test: $(r.test_name)"
    end
    _pretty_table(io, data;
        title = title,
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    conc = if is_label
        r.identified ? "Column assignment is stable across residual-bootstrap VAR re-estimates" :
                       "Column assignment is unstable; labels should not be treated as identified"
    else
        r.identified ? "Evidence supports identification" : "Identification conditions may not hold"
    end
    note = no_pval ? "Label-stability reports a match fraction; it has no p-value" :
                     "*** p<0.01, ** p<0.05, * p<0.10"
    conc_data = Any["Conclusion" conc; "Note" note]
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
# Internal helpers: shocks, Holm, method dispatch
# =============================================================================

"""Holm (1979) adjusted p-values: ``\\tilde p_{(k)} = \\max_{j\\le k}\\min(1,(m-j+1)p_{(j)})``."""
function _holm_adjust(pvals::AbstractVector{T}) where {T<:AbstractFloat}
    m = length(pvals)
    adj = similar(pvals)
    m == 0 && return adj
    order = sortperm(pvals)
    running = zero(T)
    @inbounds for k in 1:m
        idx = order[k]
        raw = T(m - k + 1) * pvals[idx]
        running = max(running, min(one(T), raw))
        adj[idx] = running
    end
    adj
end

function _result_method(r::ICASVARResult)
    r.method
end
_result_method(r::NonGaussianMLResult) = r.distribution
_result_method(::NonGaussianGMMResult) = :gmm_moments
_result_method(::GARCHSVARResult) = :garch
_result_method(::MarkovSwitchingSVARResult) = :markov_switching
_result_method(::SmoothTransitionSVARResult) = :smooth_transition
_result_method(::ExternalVolatilitySVARResult) = :external_volatility
_result_method(::AbstractNonGaussianSVAR) = :unknown

function _result_shocks(r)
    if hasfield(typeof(r), :shocks)
        s = getfield(r, :shocks)
        if s isa AbstractMatrix && size(s, 1) > 0
            return s
        end
    end
    if hasfield(typeof(r), :residuals)
        U = getfield(r, :residuals)
        if U isa AbstractMatrix && size(U, 1) > 0
            return (robust_inv(r.B0) * U')'
        end
    end
    throw(ArgumentError(
        "$(typeof(r).name.name) does not store shocks; pass a result from the current identifier"))
end

const _ICA_METHODS = (:fastica, :jade, :sobi, :dcov, :hsic)
const _HETEROSKEDASTIC_METHODS = (:markov_switching, :garch, :smooth_transition,
                                  :external_volatility)

function _identify_for_method(model::VARModel, method::Symbol;
                              rng::AbstractRNG=Random.default_rng(),
                              transition_var=nothing,
                              regime_indicator=nothing)
    method === :fastica && return identify_fastica(model; rng=rng)
    method === :jade && return identify_jade(model)
    method === :sobi && return identify_sobi(model)
    method === :dcov && return identify_dcov(model)
    method === :hsic && return identify_hsic(model)
    method === :student_t && return identify_student_t(model)
    method === :mixture_normal && return identify_mixture_normal(model)
    method === :pml && return identify_pml(model)
    method === :skew_normal && return identify_skew_normal(model)
    method === :nongaussian_ml && return identify_nongaussian_ml(model)
    method === :gmm_moments && return identify_gmm_moments(model)
    method === :markov_switching && return identify_markov_switching(model; rng=rng)
    method === :garch && return identify_garch(model)
    if method === :smooth_transition
        transition_var === nothing && throw(ArgumentError(
            "method=:smooth_transition requires transition_var"))
        return identify_smooth_transition(model, transition_var)
    end
    if method === :external_volatility
        regime_indicator === nothing && throw(ArgumentError(
            "method=:external_volatility requires regime_indicator"))
        return identify_external_volatility(model, regime_indicator)
    end
    throw(ArgumentError("unknown identification method :$method"))
end

function _jb_kurtosis_from_shocks(shocks::Matrix{T}) where {T<:AbstractFloat}
    T_obs, n = size(shocks)
    jb_stats = Vector{T}(undef, n)
    jb_pvals = Vector{T}(undef, n)
    kurt_stats = Vector{T}(undef, n)
    kurt_pvals = Vector{T}(undef, n)
    for j in 1:n
        s = @view shocks[:, j]
        σ = std(s)
        σ > zero(T) || throw(ArgumentError("shock $j has zero variance"))
        s_std = (s .- mean(s)) / σ
        skew = mean(s_std .^ 3)
        kurt = mean(s_std .^ 4) - T(3)
        jb = T_obs * (skew^2 / T(6) + kurt^2 / T(24))
        jb_stats[j] = jb
        jb_pvals[j] = T(1) - T(cdf(Chisq(2), jb))
        se_k = sqrt(T(24) / T(T_obs))
        z_k = kurt / se_k
        kurt_stats[j] = z_k
        kurt_pvals[j] = T(2) * (T(1) - T(cdf(Normal(), abs(z_k))))
    end
    (jb_stats, jb_pvals, kurt_stats, kurt_pvals)
end

# =============================================================================
# Public API: Label-stability bootstrap
# =============================================================================

"""
    test_label_stability(model::VARModel; method=:fastica, n_bootstrap=999,
                         rng) -> IdentifiabilityTestResult

Residual-bootstrap diagnostic of shock **label stability**.

Each replication resamples residuals, rebuilds a pseudo-sample from the estimated
VAR, re-estimates the VAR, re-identifies ``B_0``, and matches columns to the
original estimate with [`_match_columns`](@ref). The statistic is the fraction of
replications whose signed permutation is the **identity** (column order unchanged).
There is **no p-value**: this is a descriptive match-fraction, not a hypothesis test.

`identified` is `true` when the match fraction is at least 1/2.
"""
function test_label_stability(model::VARModel{T}; method::Symbol=:fastica,
                              n_bootstrap::Int=999,
                              rng::AbstractRNG=Random.default_rng(),
                              transition_var=nothing,
                              regime_indicator=nothing) where {T<:AbstractFloat}
    n = nvars(model)
    p = model.p
    T_eff = size(model.U, 1)
    ref = _identify_for_method(model, method; rng=rng, transition_var=transition_var,
                               regime_indicator=regime_indicator)
    B0_ref = Matrix{T}(ref.B0)
    n_id = 0
    n_ok = 0
    Y_init = model.Y[1:p, :]
    for _ in 1:n_bootstrap
        try
            U_boot = _resample_residuals(model.U, :iid, rng)
            Y_boot = _simulate_var(Y_init, model.B, U_boot, T_eff + p)
            m_star = estimate_var(Y_boot, p; check_stability=false)
            boot = _identify_for_method(m_star, method; rng=rng,
                                        transition_var=transition_var,
                                        regime_indicator=regime_indicator)
            perm, _ = _match_columns(B0_ref, Matrix{T}(boot.B0))
            n_ok += 1
            perm == 1:n && (n_id += 1)
        catch
            continue
        end
    end
    n_ok == 0 && return IdentifiabilityTestResult{T}(
        :label_stability, T(NaN), T(NaN), false,
        Dict{Symbol, Any}(:method => method, :n_bootstrap => 0, :n_identity => 0,
                          :match_fraction => T(NaN)))
    frac = T(n_id) / T(n_ok)
    IdentifiabilityTestResult{T}(:label_stability, frac, T(NaN), frac >= T(0.5),
                                  Dict{Symbol, Any}(:method => method,
                                                     :n_bootstrap => n_ok,
                                                     :n_identity => n_id,
                                                     :match_fraction => frac))
end

# =============================================================================
# Public API: Test Identification Strength (deprecated wrapper)
# =============================================================================

const _STRENGTH_DEPWARN =
    "test_identification_strength is deprecated; use test_lambda_distinct " *
    "for heteroskedastic identification, test_gaussian_shock_count for " *
    "non-Gaussian identification, or test_label_stability for column-label stability"

"""
    test_identification_strength(model::VARModel; method=:fastica, n_bootstrap=999)
    test_identification_strength(result)

Deprecated wrapper (one release) around the principled diagnostics:

- heteroskedastic results → [`test_lambda_distinct`](@ref)
- non-Gaussian results → [`test_gaussian_shock_count`](@ref) (per-shock JB/kurtosis, Holm)
- `VARModel` with an ICA `method` → [`test_label_stability`](@ref) (match fraction, **no p-value**)
"""
function test_identification_strength(model::VARModel{T}; method::Symbol=:fastica,
                                       n_bootstrap::Int=999,
                                       rng::AbstractRNG=Random.default_rng(),
                                       transition_var=nothing,
                                       regime_indicator=nothing) where {T<:AbstractFloat}
    Base.depwarn(_STRENGTH_DEPWARN, :test_identification_strength)
    if method in _ICA_METHODS
        return test_label_stability(model; method=method, n_bootstrap=n_bootstrap, rng=rng)
    elseif method in _HETEROSKEDASTIC_METHODS
        r = _identify_for_method(model, method; rng=rng, transition_var=transition_var,
                                 regime_indicator=regime_indicator)
        return _wrap_lambda_distinct(r)
    else
        r = _identify_for_method(model, method; rng=rng, transition_var=transition_var,
                                 regime_indicator=regime_indicator)
        return test_gaussian_shock_count(r)
    end
end

function test_identification_strength(result::Union{MarkovSwitchingSVARResult,
                                                     GARCHSVARResult,
                                                     SmoothTransitionSVARResult,
                                                     ExternalVolatilitySVARResult};
                                       kwargs...)
    Base.depwarn(_STRENGTH_DEPWARN, :test_identification_strength)
    _wrap_lambda_distinct(result; kwargs...)
end

function test_identification_strength(result::Union{ICASVARResult,
                                                     NonGaussianMLResult,
                                                     NonGaussianGMMResult};
                                       kwargs...)
    Base.depwarn(_STRENGTH_DEPWARN, :test_identification_strength)
    test_gaussian_shock_count(result; kwargs...)
end

function _wrap_lambda_distinct(result; pairs=:all)
    w = test_lambda_distinct(result; pairs=pairs)
    Tλ = eltype(w.statistic)
    finite_p = Tλ[p for p in w.pvalue_bonferroni if isfinite(p)]
    identified = !isempty(finite_p) && all(<(Tλ(0.05)), finite_p)
    finite_stats = Tλ[x for x in w.statistic if isfinite(x)]
    stat = isempty(finite_stats) ? Tλ(NaN) : maximum(finite_stats)
    pval = isempty(finite_p) ? Tλ(NaN) : minimum(finite_p)
    IdentifiabilityTestResult{Tλ}(:lambda_distinct, stat, pval, identified,
                                  Dict{Symbol, Any}(:pairs => w.pairs,
                                                     :pvalue_bonferroni => w.pvalue_bonferroni,
                                                     :statistics => w.statistic,
                                                     :pvalues => w.pvalue))
end

# =============================================================================
# Public API: Test Shock Gaussianity
# =============================================================================

"""
    test_shock_gaussianity(result) -> IdentifiabilityTestResult

Test whether recovered structural shocks are non-Gaussian using univariate JB
tests. Dispatches on every identification result that stores (or can form)
shocks.

Non-Gaussian identification requires at most one shock to be Gaussian. The
`identified` flag is `true` when at most one shock fails to reject Gaussianity
at 5% after Holm adjustment of the per-shock JB p-values.
"""
function test_shock_gaussianity(result)
    s = _result_shocks(result)
    _test_shock_gaussianity_impl(Matrix{eltype(s)}(s), _result_method(result))
end

# =============================================================================
# Public API: Gaussian-shock count (Keweloh 2021; LMS 2017)
# =============================================================================

"""
    test_gaussian_shock_count(result; alpha=0.05) -> IdentifiabilityTestResult

Sequential count of Gaussian structural shocks (Keweloh 2021; Lanne, Meitz &
Saikkonen 2017). Each recovered shock is tested for zero third and fourth
cumulants (Jarque–Bera) and for excess kurtosis. Per-shock p-values are
Holm-adjusted. Identification of ``B₀`` requires at most one Gaussian shock;
two or more failures to reject Gaussianity at `alpha` flag non-identification.

`details` stores `:n_gaussian`, `:jb_stats`, `:jb_pvals`, `:jb_pvals_holm`,
`:kurt_stats`, `:kurt_pvals`, `:kurt_pvals_holm`, and `:alpha`.
"""
function test_gaussian_shock_count(result; alpha::Real=0.05)
    s = _result_shocks(result)
    Tsh = eltype(s)
    α = Tsh(alpha)
    jb_stats, jb_pvals, kurt_stats, kurt_pvals = _jb_kurtosis_from_shocks(Matrix{Tsh}(s))
    jb_holm = _holm_adjust(jb_pvals)
    kurt_holm = _holm_adjust(kurt_pvals)
    n = length(jb_pvals)
    n_gaussian = count(p -> p >= α, jb_holm)
    identified = n_gaussian <= 1
    jb_sorted = sort(jb_stats)
    stat = n >= 2 ? jb_sorted[2] : jb_sorted[1]
    pval = Tsh(1) - Tsh(cdf(Chisq(2), stat))
    details = Dict{Symbol, Any}(:jb_stats => jb_stats,
                                :jb_pvals => jb_pvals,
                                :jb_pvals_holm => jb_holm,
                                :kurt_stats => kurt_stats,
                                :kurt_pvals => kurt_pvals,
                                :kurt_pvals_holm => kurt_holm,
                                :n_gaussian => n_gaussian,
                                :alpha => α)
    hasfield(typeof(result), :moments) && (details[:moments] = getfield(result, :moments))
    IdentifiabilityTestResult{Tsh}(:gaussian_shock_count, stat, pval, identified, details)
end

function _test_shock_gaussianity_impl(shocks::Matrix{T}, method::Symbol) where {T<:AbstractFloat}
    jb_stats, jb_pvals, kurt_stats, kurt_pvals = _jb_kurtosis_from_shocks(shocks)
    n = length(jb_stats)
    jb_holm = _holm_adjust(jb_pvals)
    n_gaussian = count(p -> p >= T(0.05), jb_holm)
    identified = n_gaussian <= 1
    joint_stat = sum(jb_stats)
    joint_pval = T(1) - T(cdf(Chisq(2n), joint_stat))
    IdentifiabilityTestResult{T}(:shock_gaussianity, joint_stat, joint_pval, identified,
                                  Dict{Symbol, Any}(:jb_stats => jb_stats,
                                                     :jb_pvals => jb_pvals,
                                                     :jb_pvals_holm => jb_holm,
                                                     :kurt_stats => kurt_stats,
                                                     :kurt_pvals => kurt_pvals,
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
    test_shock_independence(result; max_lag=10) -> IdentifiabilityTestResult

Test independence of recovered structural shocks.

Uses both cross-correlation (portmanteau) and distance covariance tests.
Independence is a necessary condition for valid identification. Dispatches on
every result that stores (or can form) shocks, including Markov-switching and
external-volatility identification.
"""
function test_shock_independence(result; max_lag::Int=10,
                                  rng::AbstractRNG=Random.default_rng())
    s = _result_shocks(result)
    _test_independence_impl(Matrix{eltype(s)}(s), max_lag; rng=rng)
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

function _just_id_overid(::Type{T}; note::AbstractString="") where {T<:AbstractFloat}
    IdentifiabilityTestResult{T}(:overidentification, zero(T), one(T), true,
                                  Dict{Symbol, Any}(:just_identified => true,
                                                     :method => :just_identified,
                                                     :note => note))
end

function _overid_from_lr(lr; extra=Dict{Symbol,Any}())
    Tlr = typeof(lr.statistic)
    details = Dict{Symbol, Any}(:method => :lr, :df => lr.df, :just_identified => false)
    merge!(details, extra)
    IdentifiabilityTestResult{Tlr}(:overidentification, lr.statistic, lr.pvalue,
                                    lr.pvalue >= Tlr(0.05), details)
end

"""Wald of ``R\\mathrm{vec}(B_0)=0``: ``RVR'`` when `vcov_B` is stored, else independence."""
function _wald_B0_zeros(B0::AbstractMatrix{T}, se, mask::BitMatrix;
                        vcov_B=nothing) where {T<:AbstractFloat}
    n = size(B0, 1)
    idx = Int[]
    @inbounds for j in 1:n, i in 1:n
        mask[i, j] && push!(idx, i + (j - 1) * n)
    end
    n_r = length(idx)
    n_r == 0 && return nothing
    b = vec(Matrix{T}(B0))[idx]
    if vcov_B isa AbstractMatrix && size(vcov_B) == (n * n, n * n)
        w = _wald_rvr(b, Matrix{T}(vcov_B[idx, idx]))
        w !== nothing && return merge(w, (approximation=:rvr,))
    end
    (se isa AbstractMatrix && size(se) == size(B0)) || return nothing
    W = zero(T)
    @inbounds for i in 1:n, j in 1:n
        mask[i, j] || continue
        s = se[i, j]
        (isfinite(s) && s > zero(T)) || return nothing
        W += (B0[i, j] / s)^2
    end
    pval = T(1) - T(cdf(Chisq(n_r), W))
    (statistic=W, pvalue=pval, df=n_r, approximation=:independence)
end

function _wald_rvr(b::AbstractVector{T}, V::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = length(b)
    (size(V, 1) == n && size(V, 2) == n) || return nothing
    _vcov_wald_ok(V) || return nothing
    W = try
        T(dot(b, Matrix{T}(robust_inv(Hermitian(V); silent=true)) * b))
    catch
        return nothing
    end
    isfinite(W) || return nothing
    W = max(W, zero(T))
    pval = T(1) - T(cdf(Chisq(n), W))
    (statistic=W, pvalue=pval, df=n)
end

function _with_wald!(extra::Dict{Symbol,Any}, wald)
    wald === nothing && return extra
    extra[:wald_statistic] = wald.statistic
    extra[:wald_pvalue] = wald.pvalue
    extra[:wald_df] = wald.df
    extra[:wald_approximation] = wald.approximation
    extra
end

function _align_vec_B0_vcov(V, perm::AbstractVector, signs, n::Int)
    V isa AbstractMatrix && size(V) == (n * n, n * n) || return nothing
    T = eltype(V)
    P = zeros(T, n * n, n * n)
    @inbounds for k in 1:n
        j = perm[k]
        s = T(signs[k])
        for i in 1:n
            P[i + (k - 1) * n, i + (j - 1) * n] = s
        end
    end
    Matrix{T}(P * V * P')
end

function _Q_givens_angles(Q::AbstractMatrix{T}, n::Int) where {T<:AbstractFloat}
    n <= 1 && return T[]
    Qp = Matrix{T}(Q)
    if det(Qp) < 0
        Qp[:, n] .*= -one(T)
    end
    _orthogonal_to_givens(Qp, n)
end

function _givens_vec_B0_vcov(θ::Vector{T}, Vθ::AbstractMatrix{T}, Lmat::AbstractMatrix{T},
                             n::Int) where {T<:AbstractFloat}
    n_angles = length(θ)
    n_angles == 0 && return nothing
    size(Vθ) == (n_angles, n_angles) || return nothing
    _vcov_wald_ok(Vθ) || return nothing
    VB = try
        _, VBm = _delta_B0_se(θ, Vθ, ϑ -> Lmat * _givens_to_orthogonal(ϑ, n), n)
        VBm
    catch
        return nothing
    end
    size(VB) == (n * n, n * n) && _vcov_wald_ok(VB) ? Matrix{T}(VB) : nothing
end

function _vec_B0_vcov(result::NonGaussianMLResult{T},
                      model::VARModel{T}) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    n_angles = n * (n - 1) ÷ 2
    V = result.vcov
    (V isa AbstractMatrix && size(V, 1) >= n_angles && size(V, 2) >= n_angles) ||
        return nothing
    n_angles == 0 && return nothing
    Vθ = Matrix{T}(V[1:n_angles, 1:n_angles])
    Lmat = Matrix{T}(safe_cholesky(model.Sigma))
    _givens_vec_B0_vcov(_Q_givens_angles(result.Q, n), Vθ, Lmat, n)
end

function _vec_B0_vcov(result::NonGaussianGMMResult{T},
                      model::VARModel{T}) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    Lmat = Matrix{T}(safe_cholesky(model.Sigma))
    _givens_vec_B0_vcov(Vector{T}(result.theta), Matrix{T}(result.vcov), Lmat, n)
end

function _vec_B0_vcov(result::Union{MarkovSwitchingSVARResult{T},
                                    ExternalVolatilitySVARResult{T}},
                      ::Any) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    nn = n * n
    V = result.vcov
    (V isa AbstractMatrix && size(V, 1) >= nn && size(V, 2) >= nn) || return nothing
    VB = Matrix{T}(V[1:nn, 1:nn])
    _vcov_wald_ok(VB) ? VB : nothing
end

function _vec_B0_vcov(result::GARCHSVARResult{T},
                      model::VARModel{T}) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    n_angles = n * (n - 1) ÷ 2
    V = result.vcov
    (V isa AbstractMatrix && size(V, 1) >= n_angles && size(V, 2) >= n_angles) ||
        return nothing
    n_angles == 0 && return nothing
    Vθ = Matrix{T}(V[1:n_angles, 1:n_angles])
    Lmat = Matrix{T}(safe_cholesky(model.Sigma))
    _givens_vec_B0_vcov(_Q_givens_angles(result.Q, n), Vθ, Lmat, n)
end

function _vec_B0_vcov(result::SmoothTransitionSVARResult{T}, ::Any) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    V = result.vcov
    _vcov_wald_ok(V) || return nothing
    n_L = n * (n + 1) ÷ 2
    n_angles = n * (n - 1) ÷ 2
    size(V, 1) == n_L + n_angles + n + 2 || return nothing
    L_mat = Matrix{T}(result.B0 * result.Q')
    θ = _Q_givens_angles(result.Q, n)
    s = result.transition_var
    length(s) >= 2 || return nothing
    sigma_s = std(s)
    sigma_s > zero(T) || return nothing
    logγ_lo = log(T(1e-3) / sigma_s)
    logγ_hi = log(T(20) / sigma_s)
    xγ = _st_x_from_gamma(result.gamma, logγ_lo, logγ_hi)
    p = vcat(_st_pack_L(L_mat), θ, log.(max.(result.Lambda[2], T(1e-12))), xγ,
             result.threshold)
    length(p) == size(V, 1) || return nothing
    B0_fn = z -> begin
        Lp = _st_unpack_L(view(z, 1:n_L), n)
        Qp = n_angles == 0 ? Matrix{eltype(z)}(I, n, n) :
             _givens_to_orthogonal(z[(n_L + 1):(n_L + n_angles)], n)
        Lp * Qp
    end
    VB = try
        _, VBm = _delta_B0_se(p, Matrix{T}(V), B0_fn, n)
        VBm
    catch
        return nothing
    end
    size(VB) == (n * n, n * n) && _vcov_wald_ok(VB) ? Matrix{T}(VB) : nothing
end

function _vec_B0_vcov(result::SVARModel{T}, ::Any) where {T<:AbstractFloat}
    result.vcov === nothing && return nothing
    n = size(result.B, 1)
    θ = _ab_pack(result.A, result.B, result.pattern)
    isempty(θ) && return nothing
    size(result.vcov) == (length(θ), length(θ)) || return nothing
    _vcov_wald_ok(result.vcov) || return nothing
    B0_fn = p -> begin
        A, B = _ab_unpack(p, result.pattern)
        A \ B
    end
    VB = try
        _, VBm = _delta_B0_se(θ, result.vcov, B0_fn, n)
        VBm
    catch
        return nothing
    end
    size(VB) == (n * n, n * n) && _vcov_wald_ok(VB) ? Matrix{T}(VB) : nothing
end

function _aligned_overid_wald(result, model, B0, se, mask)
    perm, signs = _align_to_zeros(B0, mask)
    B_al = B0[:, perm] .* signs'
    se_al = (se isa AbstractMatrix && size(se) == size(B0)) ? se[:, perm] : se
    n = size(B0, 1)
    VB_al = _align_vec_B0_vcov(_vec_B0_vcov(result, model), perm, signs, n)
    wald = _wald_B0_zeros(B_al, se_al, mask; vcov_B=VB_al)
    (wald, perm, signs, B_al, se_al)
end

function _ml_dist_init(result::NonGaussianMLResult{T}, perm::Vector{Int}) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    d = result.distribution
    dp = result.dist_params
    if d === :student_t
        ν = dp[:nu][perm]
        return T[log(max(ν[j] - T(2.01), T(1e-8))) for j in 1:n]
    elseif d === :skew_normal
        return Vector{T}(dp[:alpha][perm])
    elseif d === :mixture_normal
        p_mix = dp[:p_mix][perm]
        σ1 = dp[:sigma1][perm]
        out = Vector{T}(undef, 2n)
        @inbounds for j in 1:n
            pj = clamp(p_mix[j], T(1e-6), T(1) - T(1e-6))
            out[2j - 1] = log(pj / (1 - pj))
            # invert σ₁² = (1/p) * sigmoid(raw)  ⇒  sigmoid(raw) = p σ₁²
            s2 = σ1[j]^2
            sig = clamp(pj * s2, T(1e-8), T(1) - T(1e-8))
            out[2j] = log(sig / (1 - sig))
        end
        return out
    elseif d === :pml
        κ = dp[:kappa][perm]
        m = dp[:nu][perm]
        out = Vector{T}(undef, 2n)
        @inbounds for j in 1:n
            out[2j - 1] = κ[j]
            out[2j] = log(max(m[j] - T(2.05), T(1e-8)))
        end
        return out
    end
    zeros(T, _n_dist_params(d) * n)
end

"""Solve Givens angles so ``(LQ)[\\mathrm{mask}]\\approx 0`` (drop those rotations)."""
function _solve_givens_for_zeros(Lmat::AbstractMatrix{T}, mask::BitMatrix,
                                 θ0::Vector{T}) where {T<:AbstractFloat}
    n = size(Lmat, 1)
    n_angles = n * (n - 1) ÷ 2
    n_angles == 0 && return (T[], zero(T))
    obj = θ -> begin
        Tθ = eltype(θ)
        B0 = Lmat * _givens_to_orthogonal(θ, n)
        s = zero(Tθ)
        @inbounds for i in 1:n, j in 1:n
            mask[i, j] && (s += abs2(B0[i, j]))
        end
        s
    end
    best_θ = Vector{T}(θ0)
    best_f = T(obj(θ0))
    for start in (copy(θ0), zeros(T, n_angles))
        r = try
            g! = (G, x) -> ForwardDiff.gradient!(G, obj, x)
            Optim.optimize(obj, g!, start, Optim.LBFGS(),
                           Optim.Options(iterations=200, g_tol=T(1e-12),
                                         allow_f_increases=true))
        catch
            Optim.optimize(obj, start, Optim.NelderMead(),
                           Optim.Options(iterations=400, g_tol=T(1e-12)))
        end
        f = T(Optim.minimum(r))
        if f < best_f
            best_f = f
            best_θ = Vector{T}(Optim.minimizer(r))
        end
    end
    (best_θ, best_f)
end

"""Constrained non-Gaussian ML: Givens + dist with ``(LQ)[\\mathrm{mask}]=0`` (NLopt SLSQP)."""
function _nongaussian_slsqp_zeros(U::Matrix{T}, Lchol, Lmat::Matrix{T}, n::Int,
                                  dist::Symbol, mask::BitMatrix,
                                  θ0::Vector{T}, dp0::Vector{T}) where {T<:AbstractFloat}
    n_angles = length(θ0)
    n_dp = length(dp0)
    n_p = n_angles + n_dp
    n_p == 0 && return T(-Inf)
    zeros_ij = Tuple{Int,Int}[(i, j) for j in 1:n for i in 1:n if mask[i, j]]
    p0 = Float64.(vcat(θ0, dp0))
    L64 = Matrix{Float64}(Lmat)
    nll64 = function (x::Vector{Float64})
        p = T.(x)
        ang = n_angles == 0 ? T[] : Vector{T}(p[1:n_angles])
        dp = Vector{T}(p[(n_angles + 1):end])
        Float64(-_nongaussian_loglik(ang, dp, U, Lchol, n; distribution=dist))
    end
    opt = NLopt.Opt(:LD_SLSQP, n_p)
    NLopt.min_objective!(opt, (x, g) -> begin
        val = nll64(x)
        if length(g) > 0
            ε = 1e-6
            @inbounds for i in 1:n_p
                xi = x[i]
                x[i] = xi + ε
                g[i] = (nll64(x) - val) / ε
                x[i] = xi
            end
        end
        val
    end)
    for (i, j) in zeros_ij
        let i = i, j = j
            NLopt.equality_constraint!(opt, (x, g) -> begin
                ang = n_angles == 0 ? Float64[] : x[1:n_angles]
                Q = n_angles == 0 ? Matrix{Float64}(I, n, n) :
                    _givens_to_orthogonal(ang, n)
                val = (L64 * Q)[i, j]
                if length(g) > 0
                    fill!(g, 0.0)
                    if n_angles > 0
                        gj = ForwardDiff.gradient(
                            θ -> (L64 * _givens_to_orthogonal(θ, n))[i, j], ang)
                        copyto!(view(g, 1:n_angles), gj)
                    end
                end
                val
            end, 1e-8)
        end
    end
    NLopt.xtol_rel!(opt, 1e-10)
    NLopt.maxeval!(opt, 400)
    minx = try
        _, xopt, _ = NLopt.optimize(opt, p0)
        Vector{Float64}(xopt)
    catch
        return T(-Inf)
    end
    pstar = T.(minx)
    θs = n_angles == 0 ? T[] : Vector{T}(pstar[1:n_angles])
    dps = Vector{T}(pstar[(n_angles + 1):end])
    _nongaussian_loglik(θs, dps, U, Lchol, n; distribution=dist)
end

"""Restricted non-Gaussian ML: nested LR with zeros imposed on ``B_0 = LQ``."""
function _nongaussian_restricted_lr(model::VARModel{T}, result::NonGaussianMLResult{T},
                                    mask::BitMatrix) where {T<:AbstractFloat}
    n = nvars(model)
    Lchol = safe_cholesky(model.Sigma)
    Lmat = Matrix{T}(Lchol)
    perm, signs = _align_to_zeros(result.B0, mask)
    Q_al = result.Q[:, perm] .* signs'
    if det(Q_al) < 0
        Q_al = copy(Q_al)
        Q_al[:, n] .*= -one(T)
    end
    θ0 = n == 1 ? T[] : _orthogonal_to_givens(Q_al, n)
    dp0 = _ml_dist_init(result, perm)
    dist = result.distribution
    n_angles = n * (n - 1) ÷ 2
    q = count(mask)
    θ_star, resid = _solve_givens_for_zeros(Lmat, mask, θ0)
    pinned = resid <= T(1e-12) && n_angles <= q
    ℓ_r = T(-Inf)
    if pinned || n_angles == 0
        obj = dp -> -_nongaussian_loglik(θ_star, dp, model.U, Lchol, n;
                                         distribution=dist)
        res = Optim.optimize(obj, dp0, Optim.NelderMead(),
                             Optim.Options(iterations=400, g_tol=T(1e-8),
                                           allow_f_increases=true))
        dp_star = Vector{T}(Optim.minimizer(res))
        ℓ_r = _nongaussian_loglik(θ_star, dp_star, model.U, Lchol, n;
                                  distribution=dist)
    end
    if !pinned && n_angles > 0
        ℓ_s = _nongaussian_slsqp_zeros(model.U, Lchol, Lmat, n, dist, mask,
                                       θ_star, dp0)
        ℓ_r = max(ℓ_r, ℓ_s)
    end
    lr = _restriction_lr(result.loglik, ℓ_r, q)
    merge(lr, (zero_residual=resid, givens_dropped=min(q, n_angles)))
end

"""
    test_overidentification(model, result; restrictions=nothing, n_bootstrap=999,
                            rng) -> IdentifiabilityTestResult

Overidentification test for statistical and parametric SVARs.

- **Parametric ML** (`NonGaussianMLResult`): nested LR of extra zeros on ``B_0``.
  Those zeros are imposed by dropping the corresponding Givens rotations
  (``Q`` is solved so ``(LQ)[\\mathrm{mask}]=0``) and re-optimizing the remaining
  parameters, so ``\\mathrm{LR}\\sim\\chi^2(q)`` is a constrained MLE. A companion
  Wald uses ``RVR'`` from the stored ``\\mathrm{vec}(B_0)`` covariance when that
  matrix is available; otherwise `details[:wald_approximation] = :independence`.
- **AB-model** (`SVARModel`): the stored concentrated LR when `restrictions` is
  omitted. A supplied mask is **re-estimated** as an AB B-model; calling the
  `SVARModel`-only method with a mask throws `ArgumentError`.
- **SVEC** (`SVECResult`): LR via the AB B-model when `restrictions` is a zero
  mask; just-identified KPSW otherwise.
- **Heteroskedastic ML**: [`test_restrictions`](@ref) LR, plus a Wald on ``B_0``.
- **GMM**: Hansen ``J`` when no extra zeros are supplied.
- **ICA**: there is no parametric likelihood. The test **falls back to
  label-stability** and records `details[:fallback] = :label_stability`.

With no extra restrictions a just-identified parametric fit returns p-value 1.
"""
function test_overidentification(model::VARModel{T}, result::ICASVARResult{T};
                                  restrictions=nothing,
                                  n_bootstrap::Int=999,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    stab = test_label_stability(model; method=result.method, n_bootstrap=n_bootstrap, rng=rng)
    details = Dict{Symbol, Any}(stab.details)
    details[:fallback] = :label_stability
    details[:just_identified] = false
    details[:note] = "ICA has no parametric overidentification statistic; " *
                     "falling back to label-stability (match fraction, no p-value)"
    IdentifiabilityTestResult{T}(:overidentification, stab.statistic, T(NaN),
                                  stab.identified, details)
end

function test_overidentification(model::VARModel{T}, result::NonGaussianMLResult{T};
                                  restrictions=nothing,
                                  n_bootstrap::Int=999,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    if restrictions === nothing
        return _just_id_overid(T; note="Non-Gaussian ML B₀ = LQ is just-identified " *
                                       "without extra zeros on B₀")
    end
    n = size(result.B0, 1)
    mask = _zero_mask(restrictions, n)
    lr = _nongaussian_restricted_lr(model, result, mask)
    extra = Dict{Symbol,Any}(:zero_residual => lr.zero_residual,
                             :givens_dropped => lr.givens_dropped)
    wald, _, _, _, _ = _aligned_overid_wald(result, model, result.B0, result.se, mask)
    _with_wald!(extra, wald)
    _overid_from_lr(lr; extra=extra)
end

function test_overidentification(model::VARModel{T}, result::NonGaussianGMMResult{T};
                                  restrictions=nothing,
                                  n_bootstrap::Int=999,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    if restrictions === nothing
        identified = isnan(result.J_pvalue) ? true : result.J_pvalue >= T(0.05)
        return IdentifiabilityTestResult{T}(:overidentification, result.J, result.J_pvalue,
                                            identified,
                                            Dict{Symbol, Any}(:method => :hansen_j,
                                                               :moments => result.moments,
                                                               :just_identified => false))
    end
    n = size(result.B0, 1)
    mask = _zero_mask(restrictions, n)
    wald, _, _, _, _ = _aligned_overid_wald(result, model, result.B0, result.se, mask)
    wald === nothing && return _just_id_overid(T; note="GMM result has no usable SEs or vcov for a Wald test of B₀")
    extra = Dict{Symbol, Any}(:method => :wald, :df => wald.df, :just_identified => false)
    _with_wald!(extra, wald)
    IdentifiabilityTestResult{T}(:overidentification, wald.statistic, wald.pvalue,
                                  wald.pvalue >= T(0.05), extra)
end

function test_overidentification(model::VARModel{T},
                                  result::Union{MarkovSwitchingSVARResult{T},
                                                GARCHSVARResult{T},
                                                SmoothTransitionSVARResult{T},
                                                ExternalVolatilitySVARResult{T}};
                                  restrictions=nothing,
                                  n_bootstrap::Int=999,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    if restrictions === nothing
        return _just_id_overid(T; note="Heteroskedastic B₀ is just-identified without extra zeros")
    end
    n = size(result.B0, 1)
    mask = _zero_mask(restrictions, n)
    lr = test_restrictions(result, restrictions)
    extra = Dict{Symbol,Any}()
    wald, _, _, _, _ = _aligned_overid_wald(result, model, result.B0, result.se, mask)
    _with_wald!(extra, wald)
    _overid_from_lr(lr; extra=extra)
end

function test_overidentification(model::VARModel{T}, result::SVARModel{T};
                                  restrictions=nothing,
                                  n_bootstrap::Int=999,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    _ab_overidentification(model, result; restrictions=restrictions, rng=rng)
end

function test_overidentification(model::VARModel{T}, result::SVECResult{T};
                                  restrictions=nothing,
                                  n_bootstrap::Int=999,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    if restrictions === nothing
        df = result.identification.n_overidentifying
        df == 0 && return _just_id_overid(T; note="Default KPSW SVEC is just-identified")
        return IdentifiabilityTestResult{T}(:overidentification, T(NaN), T(NaN), df == 0,
                                            Dict{Symbol, Any}(:method => :svec,
                                                               :df => df,
                                                               :just_identified => false,
                                                               :note => "SVECResult does not store the AB LR; pass restrictions to re-estimate"))
    end
    n = size(result.B0, 1)
    mask = _zero_mask(restrictions, n)
    Bpat = fill(T(NaN), n, n)
    Bpat[mask] .= zero(T)
    svar = estimate_svar(to_var(result.vecm), b_model_pattern(Bpat);
                         rng=rng, long_run_matrix=result.Xi)
    extra = Dict{Symbol,Any}()
    seB = svar.se === nothing ? nothing : svar.se[:, (n + 1):(2n)]
    wald = _wald_B0_zeros(svar.B, seB, mask; vcov_B=_vec_B0_vcov(svar, nothing))
    _with_wald!(extra, wald)
    _overid_from_lr((statistic=svar.lr_stat, pvalue=svar.lr_pvalue, df=svar.lr_df);
                    extra=extra)
end

function test_overidentification(result::SVARModel{T};
                                  restrictions=nothing,
                                  n_bootstrap::Int=999,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    _ab_overidentification(nothing, result; restrictions=restrictions, rng=rng)
end

function _ab_overidentification(model, result::SVARModel{T};
                                restrictions=nothing,
                                rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    if restrictions === nothing
        extra = Dict{Symbol,Any}(:lr_df => result.lr_df, :pattern => :stored)
        return _overid_from_lr((statistic=result.lr_stat, pvalue=result.lr_pvalue,
                                df=result.lr_df); extra=extra)
    end
    model === nothing && throw(ArgumentError(
        "AB overidentification of a supplied mask requires the VAR model so the " *
        "pattern can be re-estimated; call test_overidentification(model, svar; " *
        "restrictions=mask), or omit restrictions to use the stored pattern's LR"))
    n = size(result.B, 1)
    mask = _zero_mask(restrictions, n)
    Bpat = fill(T(NaN), n, n)
    Bpat[mask] .= zero(T)
    svar = estimate_svar(model, b_model_pattern(Bpat); rng=rng)
    extra = Dict{Symbol,Any}(:lr_df => svar.lr_df, :pattern => :reestimated)
    B0 = result.A \ result.B
    seB = result.se === nothing ? nothing : result.se[:, (n + 1):(2n)]
    wald, _, _, _, _ = _aligned_overid_wald(result, model, B0, seB, mask)
    _with_wald!(extra, wald)
    _overid_from_lr((statistic=svar.lr_stat, pvalue=svar.lr_pvalue, df=svar.lr_df);
                    extra=extra)
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
