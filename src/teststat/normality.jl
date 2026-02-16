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
Multivariate normality tests for VAR residuals.

Implements Jarque-Bera (multivariate), Mardia skewness/kurtosis,
Doornik-Hansen, Henze-Zirkler, and a convenience suite that runs all tests.

References:
- Jarque, C. M. & Bera, A. K. (1980). "Efficient tests for normality."
- Mardia, K. V. (1970). "Measures of multivariate skewness and kurtosis."
- Doornik, J. A. & Hansen, H. (2008). "An omnibus test for univariate and multivariate normality."
- Henze, N. & Zirkler, B. (1990). "A class of invariant consistent tests for multivariate normality."
- Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Result Types
# =============================================================================

"""
    NormalityTestResult{T} <: AbstractNormalityTest

Result of a multivariate normality test.

Fields:
- `test_name::Symbol` — `:jarque_bera`, `:mardia_skewness`, `:mardia_kurtosis`, `:doornik_hansen`, `:henze_zirkler`
- `statistic::T` — test statistic
- `pvalue::T` — p-value
- `df::Int` — degrees of freedom (for chi-squared tests)
- `n_vars::Int` — number of variables
- `n_obs::Int` — number of observations
- `components::Union{Nothing, Vector{T}}` — per-component statistics (for component-wise tests)
- `component_pvalues::Union{Nothing, Vector{T}}` — per-component p-values
"""
struct NormalityTestResult{T<:AbstractFloat} <: AbstractNormalityTest
    test_name::Symbol
    statistic::T
    pvalue::T
    df::Int
    n_vars::Int
    n_obs::Int
    components::Union{Nothing, Vector{T}}
    component_pvalues::Union{Nothing, Vector{T}}
end

function Base.show(io::IO, r::NormalityTestResult{T}) where {T}
    stars = _significance_stars(r.pvalue)
    reject_5 = r.pvalue < 0.05
    conclusion = reject_5 ? "Reject H₀ (residuals are non-normal)" :
                            "Fail to reject H₀ (no evidence against normality)"
    data = Any[
        "H₀"                "Residuals are multivariate normal";
        "H₁"                "Residuals are non-normal";
        "Test Statistic"     string(_fmt(r.statistic), " ", stars);
        "P-value"            _format_pvalue(r.pvalue);
        "Degrees of freedom" r.df;
        "Variables"          r.n_vars;
        "Observations"       r.n_obs
    ]
    _pretty_table(io, data;
        title = "Normality Test: $(r.test_name)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# StatsAPI interface
StatsAPI.pvalue(r::NormalityTestResult) = r.pvalue

"""
    NormalityTestSuite{T}

Collection of normality test results from `normality_test_suite`.

Fields:
- `results::Vector{NormalityTestResult{T}}` — individual test results
- `residuals::Matrix{T}` — the residual matrix tested
- `n_vars::Int`
- `n_obs::Int`
"""
struct NormalityTestSuite{T<:AbstractFloat}
    results::Vector{NormalityTestResult{T}}
    residuals::Matrix{T}
    n_vars::Int
    n_obs::Int
end

function Base.show(io::IO, s::NormalityTestSuite{T}) where {T}
    data = Matrix{Any}(undef, length(s.results), 4)
    for (i, r) in enumerate(s.results)
        data[i, 1] = string(r.test_name)
        data[i, 2] = _fmt(r.statistic)
        data[i, 3] = _format_pvalue(r.pvalue)
        data[i, 4] = r.pvalue < 0.05 ? "Reject" : "Fail to reject"
    end
    _pretty_table(io, data;
        title = "Multivariate Normality Test Suite (n=$(s.n_obs), k=$(s.n_vars))",
        column_labels = ["Test", "Statistic", "P-value", "H₀ (5%)"],
        alignment = [:l, :r, :r, :l],
    )
end

# =============================================================================
# Internal Helpers
# =============================================================================

"""Standardize residuals: Z = P⁻¹ U where Σ = PP'."""
function _standardize_residuals(U::Matrix{T}, Sigma::Matrix{T}) where {T<:AbstractFloat}
    L = safe_cholesky(Sigma)
    (robust_inv(Matrix(L)) * U')'
end

"""Compute Mahalanobis distances: d²ᵢ = uᵢ' Σ⁻¹ uᵢ."""
function _mahalanobis_distances(U::Matrix{T}, Sigma::Matrix{T}) where {T<:AbstractFloat}
    Sigma_inv = robust_inv(Sigma)
    n = size(U, 1)
    d = Vector{T}(undef, n)
    @inbounds for i in 1:n
        u = @view U[i, :]
        d[i] = dot(u, Sigma_inv * u)
    end
    d
end

"""Multivariate skewness: b₁,ₖ = (1/T²) Σᵢⱼ (uᵢ' Σ⁻¹ uⱼ)³."""
function _multivariate_skewness(U::Matrix{T}, Sigma::Matrix{T}) where {T<:AbstractFloat}
    n = size(U, 1)
    Sigma_inv = robust_inv(Sigma)
    V = U * Sigma_inv  # T × k
    G = V * U'         # T × T: G[i,j] = uᵢ' Σ⁻¹ uⱼ
    sum(G .^ 3) / n^2
end

"""Multivariate kurtosis: b₂,ₖ = (1/T) Σᵢ (uᵢ' Σ⁻¹ uᵢ)²."""
function _multivariate_kurtosis(U::Matrix{T}, Sigma::Matrix{T}) where {T<:AbstractFloat}
    d = _mahalanobis_distances(U, Sigma)
    mean(d .^ 2)
end

"""Doornik-Hansen transformation of skewness and kurtosis (Bowman-Shenton)."""
function _doornik_hansen_transform(skew::T, kurt::T, n_obs::Int) where {T<:AbstractFloat}
    n = T(n_obs)

    # Transform skewness
    beta = T(3) * (n^2 + T(27) * n - T(70)) * (n + T(1)) * (n + T(3)) /
           ((n - T(2)) * (n + T(5)) * (n + T(7)) * (n + T(9)))
    w2 = -T(1) + sqrt(T(2) * (beta - T(1)))
    delta = T(1) / sqrt(log(sqrt(w2)))
    y = skew * sqrt((w2 - T(1)) * (n + T(1)) * (n + T(3)) / (T(12) * (n - T(2))))
    z1 = delta * log(y + sqrt(y^2 + T(1)))

    # Transform kurtosis
    delta_kurt = (n - T(3)) * (n + T(1)) * (n^2 + T(15) * n - T(4))
    a = (n - T(2)) * (n + T(5)) * (n + T(7)) * (n^2 + T(27) * n - T(70)) / (T(6) * delta_kurt)
    c = (n - T(7)) * (n + T(5)) * (n + T(7)) * (n^2 + T(2) * n - T(5)) / (T(6) * delta_kurt)
    alpha = a + c * skew^2
    chi = T(2) * (kurt - T(1) - skew^2) * alpha
    chi = max(chi, eps(T))  # avoid negative
    z2 = (cbrt(chi / (T(2) * alpha)) - T(1) + T(1) / (T(9) * alpha)) * sqrt(T(9) * alpha)

    (z1, z2)
end

# =============================================================================
# Public API: Jarque-Bera Test
# =============================================================================

"""
    jarque_bera_test(model::VARModel; method=:multivariate) -> NormalityTestResult
    jarque_bera_test(U::AbstractMatrix; method=:multivariate)  -> NormalityTestResult

Multivariate Jarque-Bera test for normality of VAR residuals.

Methods:
- `:multivariate` — joint test based on multivariate skewness and kurtosis (Lütkepohl 2005)
- `:component` — component-wise univariate JB tests on standardized residuals

**Reference**: Jarque & Bera (1980), Lütkepohl (2005, §4.5)
"""
function jarque_bera_test(U::AbstractMatrix{<:Real}; method::Symbol=:multivariate)
    Uf = Matrix{Float64}(U)
    T_obs, k = size(Uf)
    Sigma = cov(Uf)

    if method == :component
        Z = _standardize_residuals(Uf, Sigma)
        comp_stats = Float64[]
        comp_pvals = Float64[]
        for j in 1:k
            z = @view Z[:, j]
            s = mean(z .^ 3)
            kk = mean(z .^ 4) - 3.0
            jb = T_obs * (s^2 / 6.0 + kk^2 / 24.0)
            push!(comp_stats, jb)
            push!(comp_pvals, 1.0 - cdf(Chisq(2), jb))
        end
        stat = sum(comp_stats)
        pval = 1.0 - cdf(Chisq(2k), stat)
        return NormalityTestResult{Float64}(:jarque_bera, stat, pval, 2k, k, T_obs,
                                            comp_stats, comp_pvals)
    else
        # Multivariate JB (Lütkepohl 2005, Eq. 4.5.1)
        b1 = _multivariate_skewness(Uf, Sigma)
        b2 = _multivariate_kurtosis(Uf, Sigma)
        lambda_s = T_obs * b1 / 6.0
        lambda_k = T_obs * (b2 - k * (k + 2))^2 / (24.0 * k)
        stat = lambda_s + lambda_k
        df_val = k * (k + 1) ÷ 2 + 1   # skewness dof + kurtosis dof
        # Standard: 2k degrees of freedom (k for skewness, k for kurtosis)
        df_val = 2k
        pval = 1.0 - cdf(Chisq(df_val), stat)
        return NormalityTestResult{Float64}(:jarque_bera, stat, pval, df_val, k, T_obs,
                                            nothing, nothing)
    end
end

function jarque_bera_test(model::VARModel; method::Symbol=:multivariate)
    jarque_bera_test(model.U; method=method)
end

# =============================================================================
# Public API: Mardia Test
# =============================================================================

"""
    mardia_test(model::VARModel; type=:both) -> NormalityTestResult
    mardia_test(U::AbstractMatrix; type=:both) -> NormalityTestResult

Mardia's tests for multivariate normality based on multivariate skewness and kurtosis.

Types:
- `:skewness` — tests multivariate skewness b₁,ₖ
- `:kurtosis` — tests multivariate kurtosis b₂,ₖ
- `:both` — combined test (sum of both statistics)

Under H₀: T·b₁,ₖ/6 ~ χ²(k(k+1)(k+2)/6), (b₂,ₖ - k(k+2)) / √(8k(k+2)/T) ~ N(0,1).

**Reference**: Mardia (1970)
"""
function mardia_test(U::AbstractMatrix{<:Real}; type::Symbol=:both)
    Uf = Matrix{Float64}(U)
    T_obs, k = size(Uf)
    Sigma = cov(Uf)

    b1 = _multivariate_skewness(Uf, Sigma)
    b2 = _multivariate_kurtosis(Uf, Sigma)

    if type == :skewness
        stat = T_obs * b1 / 6.0
        df_val = k * (k + 1) * (k + 2) ÷ 6
        pval = 1.0 - cdf(Chisq(df_val), stat)
        return NormalityTestResult{Float64}(:mardia_skewness, stat, pval, df_val, k, T_obs,
                                            nothing, nothing)
    elseif type == :kurtosis
        z = (b2 - k * (k + 2)) / sqrt(8.0 * k * (k + 2) / T_obs)
        pval = 2.0 * (1.0 - cdf(Normal(), abs(z)))
        return NormalityTestResult{Float64}(:mardia_kurtosis, z, pval, 1, k, T_obs,
                                            nothing, nothing)
    else  # :both
        stat_s = T_obs * b1 / 6.0
        df_s = k * (k + 1) * (k + 2) ÷ 6
        z_k = (b2 - k * (k + 2)) / sqrt(8.0 * k * (k + 2) / T_obs)
        stat = stat_s + z_k^2
        df_val = df_s + 1
        pval = 1.0 - cdf(Chisq(df_val), stat)
        return NormalityTestResult{Float64}(:mardia_both, stat, pval, df_val, k, T_obs,
                                            [stat_s, z_k^2], [1.0 - cdf(Chisq(df_s), stat_s),
                                                               2.0 * (1.0 - cdf(Normal(), abs(z_k)))])
    end
end

function mardia_test(model::VARModel; type::Symbol=:both)
    mardia_test(model.U; type=type)
end

# =============================================================================
# Public API: Doornik-Hansen Test
# =============================================================================

"""
    doornik_hansen_test(model::VARModel) -> NormalityTestResult
    doornik_hansen_test(U::AbstractMatrix)  -> NormalityTestResult

Doornik-Hansen omnibus test for multivariate normality.

Applies the Bowman-Shenton transformation to each component's skewness and kurtosis,
then sums z₁² + z₂² across components. Under H₀: DH ~ χ²(2k).

**Reference**: Doornik & Hansen (2008)
"""
function doornik_hansen_test(U::AbstractMatrix{<:Real})
    Uf = Matrix{Float64}(U)
    T_obs, k = size(Uf)
    Sigma = cov(Uf)
    Z = _standardize_residuals(Uf, Sigma)

    stat = 0.0
    comp_stats = Float64[]
    comp_pvals = Float64[]

    for j in 1:k
        z = @view Z[:, j]
        s = mean(z .^ 3)
        kk = mean(z .^ 4)
        z1, z2 = _doornik_hansen_transform(s, kk, T_obs)
        jj = z1^2 + z2^2
        push!(comp_stats, jj)
        push!(comp_pvals, 1.0 - cdf(Chisq(2), jj))
        stat += jj
    end

    df_val = 2k
    pval = 1.0 - cdf(Chisq(df_val), stat)
    NormalityTestResult{Float64}(:doornik_hansen, stat, pval, df_val, k, T_obs,
                                  comp_stats, comp_pvals)
end

function doornik_hansen_test(model::VARModel)
    doornik_hansen_test(model.U)
end

# =============================================================================
# Public API: Henze-Zirkler Test
# =============================================================================

"""
    henze_zirkler_test(model::VARModel) -> NormalityTestResult
    henze_zirkler_test(U::AbstractMatrix)  -> NormalityTestResult

Henze-Zirkler test for multivariate normality based on the empirical characteristic function.

The test statistic is:
```math
T_{\\beta} = \\frac{1}{n} \\sum_{i,j} e^{-\\beta^2 D_{ij}/2} - 2(1+\\beta^2)^{-k/2} \\sum_i e^{-\\beta^2 d_i^2/(2(1+\\beta^2))} + n(1+2\\beta^2)^{-k/2}
```
where ``D_{ij} = (z_i - z_j)'(z_i - z_j)`` and ``d_i = z_i' z_i``.

**Reference**: Henze & Zirkler (1990)
"""
function henze_zirkler_test(U::AbstractMatrix{<:Real})
    Uf = Matrix{Float64}(U)
    T_obs, k = size(Uf)
    Sigma = cov(Uf)
    Z = _standardize_residuals(Uf, Sigma)

    # Bandwidth parameter (Henze & Zirkler 1990)
    beta = (1.0 / (2k)) * ((2.0 * k + 1.0) * T_obs / 4.0)^(1.0 / (k + 4))

    # Compute test statistic
    b2 = beta^2
    S1 = 0.0
    @inbounds for i in 1:T_obs, j in 1:T_obs
        diff = @view(Z[i, :]) .- @view(Z[j, :])
        D_ij = dot(diff, diff)
        S1 += exp(-b2 * D_ij / 2.0)
    end
    S1 /= T_obs

    S2 = 0.0
    @inbounds for i in 1:T_obs
        di2 = dot(@view(Z[i, :]), @view(Z[i, :]))
        S2 += exp(-b2 * di2 / (2.0 * (1.0 + b2)))
    end
    S2 *= 2.0 * (1.0 + b2)^(-k / 2.0)

    S3 = T_obs * (1.0 + 2.0 * b2)^(-k / 2.0)

    HZ = S1 - S2 + S3

    # Log-normal approximation for p-value (Henze & Zirkler 1990)
    wb = (1.0 + 2.0 * b2)^(-k / 2.0)
    a = 1.0 + 2.0 * b2
    mu = 1.0 - a^(-k / 2.0) * (1.0 + k * b2 / a + k * (k + 2) * b2^2 / (2.0 * a^2))
    si2_term1 = 2.0 * (1.0 + 4.0 * b2)^(-k / 2.0)
    si2_term2 = 2.0 * a^(-k) * (1.0 + 2.0 * k * b2^2 / a^2 + 3.0 * k * (k + 2) * b2^4 / (4.0 * a^4))
    si2_term3 = k * (k + 2) * b2^4 / (a^4)
    si2 = si2_term1 + si2_term2 - 4.0 * wb^2 * (1.0 + k * b2 / (2.0 * a) + k * (k + 2) * b2^2 / (4.0 * a^2))^2

    # Adjust for sample size
    si2 = max(si2, eps()) * 2.0 / T_obs
    mu_log = log(mu / sqrt(1.0 + si2 / mu^2))
    sigma_log = sqrt(log(1.0 + si2 / mu^2))

    if sigma_log > 0
        z = (log(max(HZ, eps())) - mu_log) / sigma_log
        pval = 1.0 - cdf(Normal(), z)
    else
        pval = HZ > mu ? 0.0 : 1.0
    end

    NormalityTestResult{Float64}(:henze_zirkler, HZ, pval, 0, k, T_obs, nothing, nothing)
end

function henze_zirkler_test(model::VARModel)
    henze_zirkler_test(model.U)
end

# =============================================================================
# Public API: Test Suite
# =============================================================================

"""
    normality_test_suite(model::VARModel) -> NormalityTestSuite
    normality_test_suite(U::AbstractMatrix)  -> NormalityTestSuite

Run all available multivariate normality tests and return a `NormalityTestSuite`.

Tests included:
1. Multivariate Jarque-Bera
2. Component-wise Jarque-Bera
3. Mardia skewness
4. Mardia kurtosis
5. Mardia combined
6. Doornik-Hansen
7. Henze-Zirkler
"""
function normality_test_suite(U::AbstractMatrix{<:Real})
    Uf = Matrix{Float64}(U)
    T_obs, k = size(Uf)

    results = NormalityTestResult{Float64}[
        jarque_bera_test(Uf; method=:multivariate),
        jarque_bera_test(Uf; method=:component),
        mardia_test(Uf; type=:skewness),
        mardia_test(Uf; type=:kurtosis),
        mardia_test(Uf; type=:both),
        doornik_hansen_test(Uf),
        henze_zirkler_test(Uf),
    ]

    NormalityTestSuite{Float64}(results, Uf, k, T_obs)
end

function normality_test_suite(model::VARModel)
    normality_test_suite(model.U)
end
