# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

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
    method === :student_t && return identify_student_t(model; rng=rng)
    method === :mixture_normal && return identify_mixture_normal(model; rng=rng)
    method === :mixture_normal_asym && return identify_mixture_normal(model; asymmetric=true, rng=rng)
    method === :pml && return identify_pml(model; rng=rng)
    method === :skew_normal && return identify_skew_normal(model; rng=rng)
    method === :nongaussian_ml && return identify_nongaussian_ml(model; rng=rng)
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
original estimate by signed column matching. The statistic is the fraction of
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
                                       seed::Union{Integer,Nothing}=nothing,
                                       rng::AbstractRNG=Random.default_rng(),
                                       transition_var=nothing,
                                       regime_indicator=nothing) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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
                                  seed::Union{Integer,Nothing}=nothing,
                                  rng::AbstractRNG=Random.default_rng())
    rng = _resolve_repro_rng(rng, seed)
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
                             n::Int; flip_last::Bool=false,
                             post_diag::Union{Nothing,Diagonal{T,Vector{T}}}=nothing) where {T<:AbstractFloat}
    n_angles = length(θ)
    n_angles == 0 && return nothing
    size(Vθ) == (n_angles, n_angles) || return nothing
    _vcov_wald_ok(Vθ) || return nothing
    VB = try
        B0_fn = if post_diag !== nothing
            ϑ -> Lmat * _givens_to_orthogonal(ϑ, n) * post_diag
        elseif flip_last
            S_last = Diagonal(T[j < n ? one(T) : -one(T) for j in 1:n])
            ϑ -> Lmat * _givens_to_orthogonal(ϑ, n) * S_last
        else
            ϑ -> Lmat * _givens_to_orthogonal(ϑ, n)
        end
        _, VBm = _delta_B0_se(θ, Vθ, B0_fn, n)
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
    _givens_vec_B0_vcov(_Q_givens_angles(result.Q, n), Vθ, Lmat, n;
                         flip_last=det(result.Q) < zero(T))
end

function _vec_B0_vcov(result::Union{NonGaussianGMMResult{T}, LewisTVVResult{T}},
                      model::VARModel{T}) where {T<:AbstractFloat}
    n = size(result.B0, 1)
    Lmat = Matrix{T}(safe_cholesky(model.Sigma))
    n_angles = n * (n - 1) ÷ 2
    if length(result.theta) == n_angles && all(isfinite, result.theta)
        Q_th = n_angles == 0 ? Matrix{T}(I, n, n) : _givens_to_orthogonal(Vector{T}(result.theta), n)
        D_s = Diagonal(T[dot(@view(Q_th[:, j]), @view(result.Q[:, j])) < zero(T) ? -one(T) : one(T) for j in 1:n])
        if maximum(abs.(Q_th * D_s - result.Q)) < T(1e-8)
            return _givens_vec_B0_vcov(Vector{T}(result.theta), Matrix{T}(result.vcov), Lmat, n;
                                       post_diag=all(==(one(T)), D_s.diag) ? nothing : D_s)
        end
    end
    _givens_vec_B0_vcov(_Q_givens_angles(result.Q, n), Matrix{T}(result.vcov), Lmat, n;
                         flip_last=det(result.Q) < zero(T))
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
    _givens_vec_B0_vcov(_Q_givens_angles(result.Q, n), Vθ, Lmat, n;
                         flip_last=det(result.Q) < zero(T))
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
    flip_last = n_angles > 0 && det(result.Q) < zero(T)
    S_last = flip_last ? Diagonal(T[j < n ? one(T) : -one(T) for j in 1:n]) : I
    B0_fn = z -> begin
        Lp = _st_unpack_L(view(z, 1:n_L), n)
        Qp = n_angles == 0 ? Matrix{eltype(z)}(I, n, n) :
             _givens_to_orthogonal(z[(n_L + 1):(n_L + n_angles)], n) * S_last
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
        eps_p = T(1e-4)
        eps_v = T(1e-4)
        @inbounds for j in 1:n
            pj = clamp(p_mix[j], T(2e-4), T(1) - T(2e-4))
            sig_p = clamp((pj - eps_p) / (one(T) - T(2) * eps_p), T(1e-6), one(T) - T(1e-6))
            out[2j - 1] = log(sig_p / (one(T) - sig_p))
            w = clamp(pj * σ1[j]^2, T(2e-4), T(1) - T(2e-4))
            sig_v = clamp((w - eps_v) / (one(T) - T(2) * eps_v), T(1e-6), one(T) - T(1e-6))
            out[2j] = log(sig_v / (one(T) - sig_v))
        end
        return out
    elseif d === :mixture_normal_asym
        p_mix = dp[:p_mix][perm]
        μ1 = dp[:mu1][perm]
        σ1 = dp[:sigma1][perm]
        out = Vector{T}(undef, 3n)
        eps_p = T(1e-4)
        eps_b = T(1e-4)
        eps_v = T(1e-4)
        @inbounds for j in 1:n
            pj = clamp(p_mix[j], T(2e-4), T(1) - T(2e-4))
            sig_p = clamp((pj - eps_p) / (one(T) - T(2) * eps_p), T(1e-6), one(T) - T(1e-6))
            out[3j - 2] = log(sig_p / (one(T) - sig_p))
            r_mu = μ1[j] * sqrt(pj / (one(T) - pj))
            t_u2 = clamp(r_mu / (one(T) - eps_b), -one(T) + T(1e-6), one(T) - T(1e-6))
            out[3j - 1] = atanh(t_u2)
            B = max(one(T) - r_mu^2, T(1e-8))
            w = clamp(pj * σ1[j]^2 / B, T(2e-4), T(1) - T(2e-4))
            sig_v = clamp((w - eps_v) / (one(T) - T(2) * eps_v), T(1e-6), one(T) - T(1e-6))
            out[3j] = log(sig_v / (one(T) - sig_v))
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

"""Build n×n skew-symmetric matrix from strictly upper-triangular vector `k`."""
function _skew_from_vec(k::AbstractVector{T}, n::Int) where {T<:Real}
    K = zeros(T, n, n)
    idx = 0
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        idx += 1
        v = k[idx]
        K[i, j] = v
        K[j, i] = -v
    end
    K
end

"""Cayley map ``C(k) = (I + K)^{-1}(I - K) \\in SO(n)``, Dual- and Float32-compatible."""
function _cayley_orthogonal(k::AbstractVector{T}, n::Int) where {T<:Real}
    n <= 1 && return Matrix{T}(I, n, n)
    K = _skew_from_vec(k, n)
    (I + K) \ (I - K)
end

"""Project matrix onto ``SO(n)`` via SVD polar retraction."""
function _project_so(Q::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(Q, 1)
    n <= 1 && return Matrix{T}(I, n, n)
    F = svd(Matrix{T}(Q))
    Qp = F.U * F.Vt
    if det(Qp) < zero(T)
        U = copy(F.U)
        U[:, end] .*= -one(T)
        Qp = U * F.Vt
    end
    Qp
end

"""Sum of squared entries of ``LQ`` on `mask`."""
function _mask_zero_residual(Lmat::AbstractMatrix{T}, Q::AbstractMatrix,
                             mask::BitMatrix) where {T<:AbstractFloat}
    B0 = Lmat * Q
    n = size(Lmat, 1)
    s = zero(T)
    @inbounds for j in 1:n, i in 1:n
        mask[i, j] && (s += abs2(T(B0[i, j])))
    end
    s
end

"""Solve ``Q \\in SO(n)`` so ``(LQ)[\\mathrm{mask}] \\approx 0`` via re-centered Cayley steps."""
function _solve_cayley_for_zeros(Lmat::AbstractMatrix{T}, mask::BitMatrix,
                                 Q_init::AbstractMatrix{T};
                                 max_recenter::Int=8,
                                 tol::T=T(1e-24)) where {T<:AbstractFloat}
    n = size(Lmat, 1)
    n_angles = n * (n - 1) ÷ 2
    Q_cur = _project_so(Q_init)
    f_cur = _mask_zero_residual(Lmat, Q_cur, mask)
    n_angles == 0 && return (Q_cur, f_cur)
    tol_eff = max(tol, T(16) * eps(T)^2)
    f_cur <= tol_eff && return (Q_cur, f_cur)

    for _ in 1:max_recenter
        Q_base = Q_cur
        obj = k -> begin
            Tk = eltype(k)
            B0 = Lmat * (Q_base * _cayley_orthogonal(k, n))
            s = zero(Tk)
            @inbounds for j in 1:n, i in 1:n
                mask[i, j] && (s += abs2(B0[i, j]))
            end
            s
        end
        k0 = zeros(T, n_angles)
        r = try
            cfg = ForwardDiff.GradientConfig(obj, k0)
            g! = (G, x) -> ForwardDiff.gradient!(G, obj, x, cfg)
            Optim.optimize(obj, g!, k0, Optim.LBFGS(),
                           Optim.Options(iterations=200, g_tol=eps(T),
                                         f_reltol=zero(T), allow_f_increases=true))
        catch
            try
                Optim.optimize(obj, k0, Optim.NelderMead(),
                               Optim.Options(iterations=400, g_tol=T(1e-12)))
            catch
                nothing
            end
        end
        r === nothing && break
        k_star = Vector{T}(Optim.minimizer(r))
        Q_new = _project_so(Q_base * _cayley_orthogonal(k_star, n))
        f_new = _mask_zero_residual(Lmat, Q_new, mask)
        if f_new < f_cur
            Q_cur = Q_new
            f_cur = f_new
        else
            break
        end
        (f_cur <= tol_eff || maximum(abs, k_star) <= T(1e-10)) && break
    end
    (Q_cur, f_cur)
end

"""Canonicalize column signs so largest-magnitude entry per column is non-negative and `det(Q) > 0`."""
function _canonical_col_signs!(B::AbstractMatrix{T}, Q::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(Q, 2)
    @inbounds for j in 1:n
        i_max = 1
        v_max = abs(B[1, j])
        for i in 2:n
            v = abs(B[i, j])
            if v > v_max
                v_max = v
                i_max = i
            end
        end
        if B[i_max, j] < zero(T)
            B[:, j] .*= -one(T)
            Q[:, j] .*= -one(T)
        end
    end
    if n > 0 && det(Q) < zero(T)
        B[:, n] .*= -one(T)
        Q[:, n] .*= -one(T)
    end
    Q
end

"""Canonical column alignment of `(B0, Q)` to `mask_can`, invariant to input column permutation/signs."""
function _canonical_align_to_mask(B0::AbstractMatrix{T}, Q::AbstractMatrix{T},
                                  mask_can::BitMatrix) where {T<:AbstractFloat}
    n = size(B0, 1)
    n <= 1 && return (Matrix{T}(I, n, n), collect(1:n))

    B_can = copy(B0)
    Q_can = copy(Q)
    @inbounds for j in 1:n
        i_max = 1
        v_max = abs(B_can[1, j])
        for i in 2:n
            v = abs(B_can[i, j])
            if v > v_max
                v_max = v
                i_max = i
            end
        end
        if B_can[i_max, j] < zero(T)
            B_can[:, j] .*= -one(T)
            Q_can[:, j] .*= -one(T)
        end
    end
    col_order = sortperm(1:n; by=k -> Tuple(@view(B_can[:, k])))
    B_ord = B_can[:, col_order]
    Q_ord = Q_can[:, col_order]

    best_perm = collect(1:n)
    if n <= 7
        best = T(Inf)
        for perm in _permutations(n)
            s = zero(T)
            @inbounds for j in 1:n, i in 1:n
                mask_can[i, j] && (s += abs2(B_ord[i, perm[j]]))
            end
            if s < best
                best = s
                best_perm = copy(perm)
            end
        end
    else
        used = falses(n)
        col_order_mask = sortperm(1:n; by=j -> -count(@view(mask_can[:, j])))
        for j in col_order_mask
            best_c = 1
            best_s = T(Inf)
            for c in 1:n
                used[c] && continue
                s = zero(T)
                @inbounds for i in 1:n
                    mask_can[i, j] && (s += abs2(B_ord[i, c]))
                end
                if s < best_s
                    best_s = s
                    best_c = c
                end
            end
            used[best_c] = true
            best_perm[j] = best_c
        end
    end

    B_al = B_ord[:, best_perm]
    Q_al = Q_ord[:, best_perm]
    _canonical_col_signs!(B_al, Q_al)
    orig_idx = col_order[best_perm]
    (Q_al, orig_idx)
end

"""Flip asymmetry/skewness parameters in `dp` whenever `Q_new[:, j]` flips relative to `Q_ref[:, j]`."""
function _align_dp_signs!(dp::Vector{T}, Q_new::AbstractMatrix{T},
                          Q_ref::AbstractMatrix{T}, dist::Symbol,
                          n::Int) where {T<:AbstractFloat}
    @inbounds for j in 1:n
        if dot(@view(Q_new[:, j]), @view(Q_ref[:, j])) < zero(T)
            if dist === :skew_normal
                dp[j] = -dp[j]
            elseif dist === :pml
                dp[(j - 1) * 2 + 1] = -dp[(j - 1) * 2 + 1]
            elseif dist === :mixture_normal_asym
                dp[(j - 1) * 3 + 2] = -dp[(j - 1) * 3 + 2]
            end
        end
    end
    dp
end

"""Separable per-shock multi-start optimization of distribution parameters `dp` at fixed `Q`."""
function _optimize_dp_fixed_Q(Z_chol::Matrix{T}, logdet_inv::T, n::Int, dist::Symbol,
                              Q_fixed::Matrix{T},
                              dp_warm::Union{Nothing,Vector{T}}=nothing) where {T<:AbstractFloat}
    shocks = Z_chol * Q_fixed
    T_obs = size(Z_chol, 1)
    k_dp = _n_dist_params(dist)
    dp_best = Vector{T}(undef, n * k_dp)
    n_starts_dp = dist === :mixture_normal_asym ? 5 : 2
    ll_total = T(T_obs) * logdet_inv

    for j in 1:n
        s_j = @view(shocks[:, j])
        idx = (j - 1) * k_dp
        obj_j = dpj -> -_nongaussian_shock_loglik_1d(s_j, dpj, dist)

        cands = Vector{T}[]
        if dp_warm !== nothing && length(dp_warm) == n * k_dp
            push!(cands, Vector{T}(dp_warm[(idx + 1):(idx + k_dp)]))
        end
        for s in 1:n_starts_dp
            push!(cands, _default_dist_start(T, 1, dist, s))
        end
        if dist === :mixture_normal_asym
            push!(cands, T[1.1, -0.5, 0.5], T[1.1, -0.5, -0.5], T[1.1, 0.5, 0.5], T[0.0, -0.5, -0.8])
        end

        best_dp_j = copy(cands[1])
        best_nll_j = T(Inf)
        for dp0_j in cands
            res = try
                cfg = ForwardDiff.GradientConfig(obj_j, dp0_j)
                g! = (G, x) -> ForwardDiff.gradient!(G, obj_j, x, cfg)
                Optim.optimize(obj_j, g!, dp0_j, Optim.LBFGS(),
                               Optim.Options(iterations=400, g_tol=max(T(1e-10), eps(T)),
                                             f_reltol=zero(T), allow_f_increases=true))
            catch
                try
                    Optim.optimize(obj_j, dp0_j, Optim.NelderMead(),
                                   Optim.Options(iterations=400, g_tol=T(1e-8),
                                                 allow_f_increases=true))
                catch
                    nothing
                end
            end
            res === nothing && continue
            v = T(Optim.minimum(res))
            if isfinite(v) && v < best_nll_j
                best_nll_j = v
                best_dp_j = Vector{T}(Optim.minimizer(res))
            end
        end
        if !isfinite(best_nll_j)
            best_nll_j = T(obj_j(best_dp_j))
        end
        dp_best[(idx + 1):(idx + k_dp)] .= best_dp_j
        ll_total -= best_nll_j
    end
    (T(ll_total), dp_best)
end

"""Re-centered Cayley SLSQP constrained non-Gaussian ML for ``(LQ)[\\mathrm{mask}]=0``."""
function _nongaussian_cayley_slsqp_zeros(Z_chol::Matrix{T}, logdet_inv::T,
                                         Lmat::Matrix{T}, n::Int, dist::Symbol,
                                         mask::BitMatrix, Q0_in::Matrix{T},
                                         dp0_in::Vector{T};
                                         max_recenter::Int=3) where {T<:AbstractFloat}
    n_angles = n * (n - 1) ÷ 2
    n_dp = length(dp0_in)
    n_p = n_angles + n_dp
    zeros_ij = Tuple{Int,Int}[(i, j) for j in 1:n for i in 1:n if mask[i, j]]
    q = length(zeros_ij)
    (q == 0 || n_angles == 0) && return (T(-Inf), copy(Q0_in), copy(dp0_in), T(Inf))

    Z64 = Matrix{Float64}(Z_chol)
    L64 = Matrix{Float64}(Lmat)
    ld64 = Float64(logdet_inv)
    Q_cur = Matrix{Float64}(_project_so(Q0_in))
    dp_cur = Float64.(dp0_in)
    best_ll = T(-Inf)
    best_Q = copy(Q0_in)
    best_dp = copy(dp0_in)
    best_cmax = T(Inf)

    lb = vcat(fill(-4.0, n_angles), fill(-Inf, n_dp))
    ub = vcat(fill(4.0, n_angles), fill(Inf, n_dp))
    tol_vec = fill(1e-9, q)

    for _ in 1:max_recenter
        Q_base = Q_cur
        nll_fn = x -> begin
            k = view(x, 1:n_angles)
            dp = view(x, (n_angles + 1):n_p)
            Q = Q_base * _cayley_orthogonal(k, n)
            -_nongaussian_loglik_from_Q(Q, dp, Z64, ld64, n, dist)
        end
        con_fn = k -> begin
            B0 = L64 * (Q_base * _cayley_orthogonal(k, n))
            [B0[i, j] for (i, j) in zeros_ij]
        end

        opt = NLopt.Opt(:LD_SLSQP, n_p)
        NLopt.lower_bounds!(opt, lb)
        NLopt.upper_bounds!(opt, ub)
        res_obj = ForwardDiff.DiffResults.GradientResult(zeros(Float64, n_p))
        cfg_obj = ForwardDiff.GradientConfig(nll_fn, zeros(Float64, n_p))
        NLopt.min_objective!(opt, (x, g) -> begin
            if length(g) > 0
                ForwardDiff.gradient!(res_obj, nll_fn, x, cfg_obj)
                copyto!(g, ForwardDiff.DiffResults.gradient(res_obj))
                val = ForwardDiff.DiffResults.value(res_obj)
            else
                val = nll_fn(x)
            end
            isfinite(val) ? Float64(val) : 1e30
        end)

        out_con = zeros(Float64, q)
        res_con = ForwardDiff.DiffResults.JacobianResult(out_con, zeros(Float64, n_angles))
        cfg_con = ForwardDiff.JacobianConfig(con_fn, zeros(Float64, n_angles))
        NLopt.equality_constraint!(opt, (res, x, g) -> begin
            k = x[1:n_angles]
            if length(g) > 0
                ForwardDiff.jacobian!(res_con, con_fn, k, cfg_con)
                copyto!(res, ForwardDiff.DiffResults.value(res_con))
                J = ForwardDiff.DiffResults.jacobian(res_con)
                fill!(g, 0.0)
                @inbounds for c in 1:q, a in 1:n_angles
                    g[a, c] = J[c, a]
                end
            else
                copyto!(res, con_fn(k))
            end
            nothing
        end, tol_vec)

        NLopt.xtol_rel!(opt, 1e-12)
        NLopt.ftol_rel!(opt, 1e-12)
        NLopt.maxeval!(opt, 400)

        p0 = vcat(zeros(Float64, n_angles), dp_cur)
        minx = try
            _, xopt, _ = NLopt.optimize(opt, p0)
            Vector{Float64}(xopt)
        catch
            break
        end

        k_star = minx[1:n_angles]
        dp_cur = minx[(n_angles + 1):n_p]
        Q_cand = _project_so(Q_base * _cayley_orthogonal(k_star, n))
        B_cand = L64 * Q_cand
        cmax = maximum(abs(B_cand[i, j]) for (i, j) in zeros_ij)
        if cmax > 1e-7
            Q_proj, _ = _solve_cayley_for_zeros(L64, mask, Q_cand; max_recenter=4, tol=1e-24)
            B_proj = L64 * Q_proj
            cmax_proj = maximum(abs(B_proj[i, j]) for (i, j) in zeros_ij)
            if cmax_proj < cmax
                Q_cand = Q_proj
                cmax = cmax_proj
            end
        end
        Q_cur = Q_cand

        if cmax <= 1e-5
            QT = Matrix{T}(Q_cur)
            dpT = Vector{T}(dp_cur)
            ll_cur = T(_nongaussian_loglik_from_Q(QT, dpT, Z_chol, logdet_inv, n, dist))
            if isfinite(ll_cur) && ll_cur > best_ll
                best_ll = ll_cur
                best_Q = QT
                best_dp = dpT
                best_cmax = T(cmax)
            end
        end
        (maximum(abs, k_star) <= 0.20 && cmax <= 1e-8) && break
    end
    (best_ll, best_Q, best_dp, best_cmax)
end

"""Re-centered Cayley Augmented Lagrangian fallback for rank-deficient (`rk < q`) masks."""
function _nongaussian_auglag_zeros(Z_chol::Matrix{T}, logdet_inv::T,
                                   Lmat::Matrix{T}, n::Int, dist::Symbol,
                                   mask::BitMatrix, Q0_in::Matrix{T},
                                   dp0_in::Vector{T}) where {T<:AbstractFloat}
    n_angles = n * (n - 1) ÷ 2
    n_dp = length(dp0_in)
    n_p = n_angles + n_dp
    zeros_ij = Tuple{Int,Int}[(i, j) for j in 1:n for i in 1:n if mask[i, j]]
    q = length(zeros_ij)
    (q == 0 || n_angles == 0) && return (T(-Inf), copy(Q0_in), copy(dp0_in), T(Inf))

    Z64 = Matrix{Float64}(Z_chol)
    L64 = Matrix{Float64}(Lmat)
    ld64 = Float64(logdet_inv)
    Q_cur = Matrix{Float64}(_project_so(Q0_in))
    dp_cur = Float64.(dp0_in)
    λ = zeros(Float64, q)
    ρ = 100.0
    best_ll = T(-Inf)
    best_Q = copy(Q0_in)
    best_dp = copy(dp0_in)
    best_cmax = T(Inf)

    for _ in 1:6
        Q_base = Q_cur
        obj_aug = x -> begin
            Tx = eltype(x)
            k = view(x, 1:n_angles)
            dp = view(x, (n_angles + 1):n_p)
            Q = Q_base * _cayley_orthogonal(k, n)
            nll = -_nongaussian_loglik_from_Q(Q, dp, Z64, ld64, n, dist)
            B0 = L64 * Q
            aug = zero(Tx)
            @inbounds for c in 1:q
                i, j = zeros_ij[c]
                cij = B0[i, j]
                aug += Tx(λ[c]) * cij + Tx(0.5 * ρ) * abs2(cij)
            end
            nll + aug
        end
        p0 = vcat(zeros(Float64, n_angles), dp_cur)
        res = try
            cfg = ForwardDiff.GradientConfig(obj_aug, p0)
            g! = (G, x) -> ForwardDiff.gradient!(G, obj_aug, x, cfg)
            Optim.optimize(obj_aug, g!, p0, Optim.LBFGS(),
                           Optim.Options(iterations=250, g_tol=1e-9,
                                         f_reltol=0.0, allow_f_increases=true))
        catch
            nothing
        end
        res === nothing && break
        xstar = Vector{Float64}(Optim.minimizer(res))
        k_star = xstar[1:n_angles]
        dp_cur = xstar[(n_angles + 1):n_p]
        Q_cur = _project_so(Q_base * _cayley_orthogonal(k_star, n))
        B_cur = L64 * Q_cur
        cvec = Float64[B_cur[i, j] for (i, j) in zeros_ij]
        cmax = maximum(abs, cvec)
        λ .+= ρ .* cvec
        ρ = min(ρ * 10.0, 1e8)

        Q_eval = Q_cur
        cmax_eval = cmax
        if cmax_eval > 1e-7
            Q_proj, _ = _solve_cayley_for_zeros(L64, mask, Q_cur; max_recenter=4, tol=1e-24)
            B_proj = L64 * Q_proj
            cmax_proj = maximum(abs(B_proj[i, j]) for (i, j) in zeros_ij)
            if cmax_proj < cmax_eval
                Q_eval = Q_proj
                cmax_eval = cmax_proj
            end
        end
        if cmax_eval <= 1e-5
            QT = Matrix{T}(Q_eval)
            dpT = Vector{T}(dp_cur)
            ll_cur = T(_nongaussian_loglik_from_Q(QT, dpT, Z_chol, logdet_inv, n, dist))
            if isfinite(ll_cur) && ll_cur > best_ll
                best_ll = ll_cur
                best_Q = QT
                best_dp = dpT
                best_cmax = T(cmax_eval)
            end
        end
        (cmax <= 1e-8 && maximum(abs, k_star) <= 0.15) && break
    end
    (best_ll, best_Q, best_dp, best_cmax)
end

"""Restricted non-Gaussian ML: nested LR with zeros imposed on ``B_0 = LQ``."""
function _nongaussian_restricted_lr(model::VARModel{T}, result::NonGaussianMLResult{T},
                                    mask::BitMatrix) where {T<:AbstractFloat}
    n = nvars(model)
    n_angles = n * (n - 1) ÷ 2
    q = count(mask)
    if q == 0
        lr0 = _restriction_lr(result.loglik, result.loglik, 0)
        return merge(lr0, (zero_residual=zero(T), givens_dropped=0))
    end

    Lchol = safe_cholesky(model.Sigma)
    Lmat = Matrix{T}(Lchol)
    Z_chol = Matrix{T}(transpose(Lchol \ transpose(model.U)))
    logdet_inv = -sum(i -> log(max(abs(Lchol[i, i]), eps(T))), 1:n)
    dist = result.distribution

    π_mask = sortperm(1:n; by=j -> Tuple(@view(mask[:, j])))
    mask_can = mask[:, π_mask]

    Q_al, orig_idx = _canonical_align_to_mask(result.B0, result.Q, mask_can)
    dp0_warm = _ml_dist_init(result, orig_idx)
    _align_dp_signs!(dp0_warm, Q_al, result.Q[:, orig_idx], dist, n)

    if n_angles == 0
        Q1 = Matrix{T}(I, 1, 1)
        resid1 = _mask_zero_residual(Lmat, Q1, mask_can)
        if resid1 > T(1e-10)
            lr_unfeas = _restriction_lr(result.loglik, T(-Inf), q)
            return merge(lr_unfeas, (zero_residual=resid1, givens_dropped=0))
        end
        ll1, _ = _optimize_dp_fixed_Q(Z_chol, logdet_inv, n, dist, Q1, dp0_warm)
        lr1 = _restriction_lr(result.loglik, T(ll1), q)
        return merge(lr1, (zero_residual=resid1, givens_dropped=0))
    end

    rng_det = Xoshiro(901_000 + n)
    raw_starts = Matrix{T}[Q_al, Matrix{T}(I, n, n)]
    for _ in 1:8
        push!(raw_starts, _project_so(generate_Q(n, T; rng=rng_det)))
    end

    tol_feas = T === Float32 ? T(1e-8) : T(1e-10)
    feasible_Qs = Matrix{T}[]
    feasible_resids = T[]
    best_unfeas_f = T(Inf)

    for Qs in raw_starts
        Qp, fp = _solve_cayley_for_zeros(Lmat, mask_can, Qs)
        best_unfeas_f = min(best_unfeas_f, fp)
        fp <= tol_feas || continue
        _canonical_col_signs!(Lmat * Qp, Qp)
        is_dup = false
        for Qe in feasible_Qs
            if maximum(abs.(abs.(Qe' * Qp) .- Matrix{T}(I, n, n))) < T(1e-3)
                is_dup = true
                break
            end
        end
        is_dup && continue
        push!(feasible_Qs, Qp)
        push!(feasible_resids, fp)
    end

    if isempty(feasible_Qs)
        lr_unfeas = _restriction_lr(result.loglik, T(-Inf), q)
        return merge(lr_unfeas, (zero_residual=best_unfeas_f, givens_dropped=min(q, n_angles)))
    end

    zeros_ij = Tuple{Int,Int}[(i, j) for j in 1:n for i in 1:n if mask_can[i, j]]
    L64 = Matrix{Float64}(Lmat)
    ℓ_r = T(-Inf)
    min_resid = minimum(feasible_resids)
    opt_candidates = Tuple{T, Matrix{T}, Vector{T}, T}[]

    for (idx_q, Q0) in enumerate(feasible_Qs)
        Q0_64 = Matrix{Float64}(Q0)
        con_fn0 = k -> begin
            B0 = L64 * (Q0_64 * _cayley_orthogonal(k, n))
            [B0[i, j] for (i, j) in zeros_ij]
        end
        Jc = ForwardDiff.jacobian(con_fn0, zeros(Float64, n_angles))
        sv = svdvals(Jc)
        sv_max = isempty(sv) ? 0.0 : sv[1]
        rk = count(>(1e-7 * max(1.0, sv_max)), sv)

        dp0_q = copy(dp0_warm)
        _align_dp_signs!(dp0_q, Q0, Q_al, dist, n)

        if rk >= n_angles
            ll_pin, _ = _optimize_dp_fixed_Q(Z_chol, logdet_inv, n, dist, Q0, dp0_q)
            if isfinite(ll_pin) && ll_pin > ℓ_r
                ℓ_r = T(ll_pin)
            end
            min_resid = min(min_resid, feasible_resids[idx_q])
            continue
        end

        ll_q0, dp_q0 = _optimize_dp_fixed_Q(Z_chol, logdet_inv, n, dist, Q0, dp0_q)
        if isfinite(ll_q0)
            push!(opt_candidates, (T(ll_q0), Q0, dp_q0, feasible_resids[idx_q]))
        end

        dp_cands = Vector{T}[dp_q0]
        if maximum(abs.(dp0_q - dp_q0)) > T(1e-3)
            push!(dp_cands, dp0_q)
        end
        n_extra_dp = dist === :mixture_normal_asym ? 3 : 2
        for s in 1:n_extra_dp
            ds = _default_dist_start(T, n, dist, s)
            if all(c -> maximum(abs.(ds - c)) > T(1e-3), dp_cands)
                push!(dp_cands, ds)
            end
        end

        for dp_start in dp_cands
            ll_s, Q_s, dp_s, cmax_s = _nongaussian_cayley_slsqp_zeros(
                Z_chol, logdet_inv, Lmat, n, dist, mask_can, Q0, dp_start)
            if rk < q || !isfinite(ll_s)
                ll_a, Q_a, dp_a, cmax_a = _nongaussian_auglag_zeros(
                    Z_chol, logdet_inv, Lmat, n, dist, mask_can, Q0, dp_start)
                if isfinite(ll_a) && (!isfinite(ll_s) || ll_a > ll_s)
                    ll_s, Q_s, dp_s, cmax_s = ll_a, Q_a, dp_a, cmax_a
                end
            end
            if isfinite(ll_s)
                _canonical_col_signs!(Lmat * Q_s, Q_s)
                push!(opt_candidates, (T(ll_s), Q_s, dp_s, T(cmax_s)))
            end
        end
    end

    if !isempty(opt_candidates)
        sort!(opt_candidates; by=x -> x[1], rev=true)
        polished_Qs = Matrix{T}[]
        for (ll_c, Q_c, dp_c, cmax_c) in opt_candidates
            is_dup = false
            for Qp in polished_Qs
                if maximum(abs.(abs.(Qp' * Q_c) .- Matrix{T}(I, n, n))) < T(1e-3)
                    is_dup = true
                    break
                end
            end
            is_dup && continue
            push!(polished_Qs, Q_c)
            dp_pol_in = copy(dp_c)
            _align_dp_signs!(dp_pol_in, Q_c, Q_c, dist, n)
            ll_pol, _ = _optimize_dp_fixed_Q(Z_chol, logdet_inv, n, dist, Q_c, dp_pol_in)
            ll_best_c = max(ll_c, T(ll_pol))
            if isfinite(ll_best_c) && ll_best_c > ℓ_r
                ℓ_r = T(ll_best_c)
            end
            min_resid = min(min_resid, _mask_zero_residual(Lmat, Q_c, mask_can))
            length(polished_Qs) >= 2 && break
        end
    end

    lr = _restriction_lr(result.loglik, T(ℓ_r), q)
    merge(lr, (zero_residual=min_resid, givens_dropped=min(q, n_angles)))
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
                                  seed::Union{Integer,Nothing}=nothing,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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
                                  seed::Union{Integer,Nothing}=nothing,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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
                                  seed::Union{Integer,Nothing}=nothing,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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

function test_overidentification(model::VARModel{T}, result::LewisTVVResult{T};
                                  restrictions=nothing,
                                  n_bootstrap::Int=999,
                                  seed::Union{Integer,Nothing}=nothing,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    if restrictions === nothing
        identified = isnan(result.J_pvalue) ? true : result.J_pvalue >= T(0.05)
        return IdentifiabilityTestResult{T}(:overidentification, result.J, result.J_pvalue,
                                            identified,
                                            Dict{Symbol, Any}(:method => :hansen_j,
                                                               :lags => result.lags,
                                                               :just_identified => false))
    end
    n = size(result.B0, 1)
    mask = _zero_mask(restrictions, n)
    wald, _, _, _, _ = _aligned_overid_wald(result, model, result.B0, result.se, mask)
    wald === nothing && return _just_id_overid(T; note="Lewis TVV result has no usable SEs or vcov for a Wald test of B₀")
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
                                  seed::Union{Integer,Nothing}=nothing,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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
                                  seed::Union{Integer,Nothing}=nothing,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
    _ab_overidentification(model, result; restrictions=restrictions, rng=rng)
end

function test_overidentification(model::VARModel{T}, result::SVECResult{T};
                                  restrictions=nothing,
                                  n_bootstrap::Int=999,
                                  seed::Union{Integer,Nothing}=nothing,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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
                                  seed::Union{Integer,Nothing}=nothing,
                                  rng::AbstractRNG=Random.default_rng()) where {T<:AbstractFloat}
    rng = _resolve_repro_rng(rng, seed)
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

# =============================================================================
# Parameter-space covariance transformation under label_shocks
# =============================================================================

function _givens_chart_M(Q_old::AbstractMatrix{T}, Q_new::AbstractMatrix{T},
                         perm::Vector{Int}, signs::Vector{Int}, n::Int) where {T<:AbstractFloat}
    P_signed = zeros(T, n, n)
    @inbounds for j in 1:n
        P_signed[perm[j], j] = T(signs[j])
    end
    S_old = det(Q_old) < zero(T) ? Diagonal(T[j < n ? one(T) : -one(T) for j in 1:n]) : I
    S_new = det(Q_new) < zero(T) ? Diagonal(T[j < n ? one(T) : -one(T) for j in 1:n]) : I
    Matrix{T}(S_old * P_signed * S_new)
end

function _givens_chart_jacobian(θ_old::AbstractVector{<:Real}, M::AbstractMatrix{T},
                                θ_new::AbstractVector{<:Real}, n::Int) where {T<:AbstractFloat}
    n_angles = n * (n - 1) ÷ 2
    n_angles == 0 && return Matrix{T}(I, 0, 0)
    J = try
        ForwardDiff.jacobian(
            θ -> _orthogonal_to_givens(_givens_to_orthogonal(θ, n) * M, n),
            Vector{T}(θ_old))
    catch
        fill(T(NaN), n_angles, n_angles)
    end
    if size(J) == (n_angles, n_angles) && all(isfinite, J)
        return Matrix{T}(J)
    end
    J_new = ForwardDiff.jacobian(θ -> vec(_givens_to_orthogonal(θ, n)), Vector{T}(θ_new))
    J_old = ForwardDiff.jacobian(θ -> vec(_givens_to_orthogonal(θ, n) * M), Vector{T}(θ_old))
    Matrix{T}(pinv(J_new) * J_old)
end

function _transform_statid_vcov(result::NonGaussianMLResult{T}, perm::Vector{Int},
                                signs::Vector{Int}, n::Int, Q_new) where {T<:AbstractFloat}
    V = result.vcov
    (perm == 1:n && all(==(1), signs)) && return V
    (isempty(V) || Q_new === nothing) && return V
    n_angles = n * (n - 1) ÷ 2
    k_dp = try
        _n_dist_params(result.distribution)
    catch
        return V
    end
    n_total = n_angles + n * k_dp
    size(V) == (n_total, n_total) || return V
    B = zeros(T, n_total, n_total)
    if n_angles > 0
        θ_old = _Q_givens_angles(result.Q, n)
        M = _givens_chart_M(result.Q, Q_new, perm, signs, n)
        θ_new = _Q_givens_angles(Q_new, n)
        B[1:n_angles, 1:n_angles] .= _givens_chart_jacobian(θ_old, M, θ_new, n)
    end
    dist = result.distribution
    @inbounds for j_new in 1:n
        j_old = perm[j_new]
        flip = signs[j_new] < 0
        for r in 1:k_dp
            sgn = one(T)
            if flip && ((dist === :skew_normal && r == 1) ||
                        (dist === :pml && r == 1) ||
                        (dist === :mixture_normal_asym && r == 2))
                sgn = -one(T)
            end
            B[n_angles + (j_new - 1) * k_dp + r,
              n_angles + (j_old - 1) * k_dp + r] = sgn
        end
    end
    Vn = B * V * B'
    Matrix{T}((Vn + Vn') / 2)
end

function _transform_statid_vcov(result::GARCHSVARResult{T}, perm::Vector{Int},
                                signs::Vector{Int}, n::Int, Q_new) where {T<:AbstractFloat}
    V = result.vcov
    (perm == 1:n && all(==(1), signs)) && return V
    (isempty(V) || Q_new === nothing) && return V
    n_angles = n * (n - 1) ÷ 2
    n_total = n_angles + 3 * n
    if size(V) == (n_total, n_total)
        B = zeros(T, n_total, n_total)
        if n_angles > 0
            θ_old = _Q_givens_angles(result.Q, n)
            M = _givens_chart_M(result.Q, Q_new, perm, signs, n)
            θ_new = _Q_givens_angles(Q_new, n)
            B[1:n_angles, 1:n_angles] .= _givens_chart_jacobian(θ_old, M, θ_new, n)
        end
        @inbounds for j_new in 1:n
            j_old = perm[j_new]
            for r in 1:3
                B[n_angles + (j_new - 1) * 3 + r,
                  n_angles + (j_old - 1) * 3 + r] = one(T)
            end
        end
        Vn = B * V * B'
        return Matrix{T}((Vn + Vn') / 2)
    elseif n_angles > 0 && size(V) == (n_angles, n_angles)
        θ_old = _Q_givens_angles(result.Q, n)
        M = _givens_chart_M(result.Q, Q_new, perm, signs, n)
        θ_new = _Q_givens_angles(Q_new, n)
        J = _givens_chart_jacobian(θ_old, M, θ_new, n)
        Vn = J * V * J'
        return Matrix{T}((Vn + Vn') / 2)
    end
    V
end

function _transform_statid_vcov(result::Union{NonGaussianGMMResult{T}, LewisTVVResult{T}},
                                perm::Vector{Int}, signs::Vector{Int}, n::Int,
                                Q_new) where {T<:AbstractFloat}
    V = result.vcov
    (isempty(V) || Q_new === nothing) && return V
    n_angles = n * (n - 1) ÷ 2
    (n_angles > 0 && size(V) == (n_angles, n_angles)) || return V
    Q_old = result.Q
    has_raw_theta = length(result.theta) == n_angles && all(isfinite, result.theta)
    if perm == 1:n && all(==(1), signs)
        has_raw_theta || return V
        Q_raw_id = _givens_to_orthogonal(Vector{T}(result.theta), n)
        maximum(abs.(Q_raw_id - Q_new)) <= T(1e-10) && return V
    end
    θ_old, M = if has_raw_theta
        Q_raw = _givens_to_orthogonal(Vector{T}(result.theta), n)
        S_new = det(Q_new) < zero(T) ? Diagonal(T[j < n ? one(T) : -one(T) for j in 1:n]) : I
        Vector{T}(result.theta), Matrix{T}(Q_raw' * (Q_new * S_new))
    else
        _Q_givens_angles(Q_old, n), _givens_chart_M(Q_old, Q_new, perm, signs, n)
    end
    θ_new = _Q_givens_angles(Q_new, n)
    J = _givens_chart_jacobian(θ_old, M, θ_new, n)
    Vn = J * V * J'
    Matrix{T}((Vn + Vn') / 2)
end

function _transform_statid_vcov(result::SmoothTransitionSVARResult{T}, perm::Vector{Int},
                                signs::Vector{Int}, n::Int, Q_new) where {T<:AbstractFloat}
    V = result.vcov
    (perm == 1:n && all(==(1), signs)) && return V
    (isempty(V) || Q_new === nothing) && return V
    n_L = n * (n + 1) ÷ 2
    n_angles = n * (n - 1) ÷ 2
    pdim = n_L + n_angles + n + 2
    size(V) == (pdim, pdim) || return V
    B = zeros(T, pdim, pdim)
    @inbounds for k in 1:n_L
        B[k, k] = one(T)
    end
    if n_angles > 0
        θ_old = _Q_givens_angles(result.Q, n)
        M = _givens_chart_M(result.Q, Q_new, perm, signs, n)
        θ_new = _Q_givens_angles(Q_new, n)
        B[(n_L + 1):(n_L + n_angles), (n_L + 1):(n_L + n_angles)] .=
            _givens_chart_jacobian(θ_old, M, θ_new, n)
    end
    off_lam = n_L + n_angles
    @inbounds for j_new in 1:n
        B[off_lam + j_new, off_lam + perm[j_new]] = one(T)
    end
    B[pdim - 1, pdim - 1] = one(T)
    B[pdim, pdim] = one(T)
    Vn = B * V * B'
    Matrix{T}((Vn + Vn') / 2)
end

function _transform_statid_vcov(result::Union{MarkovSwitchingSVARResult{T},
                                              ExternalVolatilitySVARResult{T}},
                                perm::Vector{Int}, signs::Vector{Int}, n::Int,
                                ::Any) where {T<:AbstractFloat}
    V = result.vcov
    (perm == 1:n && all(==(1), signs)) && return V
    isempty(V) && return V
    nn = n * n
    pdim = size(V, 1)
    (pdim == size(V, 2) && pdim >= nn && (pdim - nn) % n == 0) || return V
    K_blocks = (pdim - nn) ÷ n
    B = zeros(T, pdim, pdim)
    @inbounds for j_new in 1:n
        j_old = perm[j_new]
        s = T(signs[j_new])
        for i in 1:n
            B[i + (j_new - 1) * n, i + (j_old - 1) * n] = s
        end
    end
    @inbounds for k in 1:K_blocks
        off = nn + (k - 1) * n
        for j_new in 1:n
            B[off + j_new, off + perm[j_new]] = one(T)
        end
    end
    Vn = B * V * B'
    Matrix{T}((Vn + Vn') / 2)
end
