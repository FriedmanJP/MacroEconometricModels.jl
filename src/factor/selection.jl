# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

"""
Factor-number selection tests and estimators for static and dynamic factor models:
- Onatski (2009) dynamic factor test (`onatski_q_test`, `onatski_q_select`, `OnatskiQTestResult`)
- Ahn & Horenstein (2013) static factor ER and GR estimators (`ahn_horenstein`, `AhnHorensteinResult`)
- Onatski (2010) Edge Distribution (ED) estimator (`onatski_ed`, `OnatskiEDResult`)
"""

# =============================================================================
# Tabulated Complex Tracy-Widom (beta = 2) Spacing-Ratio Quantiles
# =============================================================================

# CDF probability grid for Onatski (2009) TW_2 ratio statistic for w = k1 - k0 in 1:15.
const _FSEL_TW2_PGRID = Float64[
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92,
    0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999
]

# High-precision quantile table (15 x 29) for
#   R_w = max_{1 <= i <= w} (mu_i - mu_{i+1}) / (mu_{i+1} - mu_{i+2})
# where mu_1 >= mu_2 >= ... follow the complex Airy_2 (GUE edge, beta = 2) point process.
# Columns 19, 22, 26 correspond to 10%, 5%, and 1% right-tail critical values (p = 0.90, 0.95, 0.99).
const _FSEL_TW2_QTABLE = Float64[
    0.3516 0.4766 0.5826 0.6758 0.7693 0.8627 0.9574 1.0544 1.1593 1.2724 1.3934 1.5324 1.6896 1.8748 2.0992 2.3822 2.7645 3.0853 3.3586 3.7146 4.1920 4.5122 4.9622 5.5708 6.5409 8.4089 10.8693 15.0789 19.5565;
    0.9343 1.0882 1.2077 1.3134 1.4152 1.5172 1.6205 1.7262 1.8398 1.9640 2.0991 2.2500 2.4222 2.6250 2.8785 3.1986 3.6413 3.9953 4.3084 4.7254 5.3155 5.7063 6.2294 6.9590 8.0583 10.4447 13.3303 18.7995 23.7336;
    1.2476 1.3963 1.5175 1.6255 1.7305 1.8350 1.9441 2.0599 2.1807 2.3136 2.4593 2.6226 2.8157 3.0402 3.3168 3.6723 4.1518 4.5578 4.9242 5.3804 6.0327 6.4680 7.0426 7.8129 9.0619 11.7062 14.9385 20.7294 25.9510;
    1.4236 1.5794 1.7028 1.8176 1.9301 2.0456 2.1626 2.2866 2.4176 2.5592 2.7175 2.9000 3.1061 3.3487 3.6474 4.0232 4.5557 5.0028 5.3810 5.8812 6.5808 7.0450 7.6410 8.4674 9.8176 12.6193 16.2057 22.0513 28.0453;
    1.5481 1.7103 1.8450 1.9702 2.0923 2.2132 2.3392 2.4705 2.6087 2.7624 2.9339 3.1241 3.3431 3.6028 3.9191 4.3273 4.9004 5.3607 5.7581 6.2875 7.0256 7.5210 8.1595 9.0560 10.5062 13.5131 17.3119 23.4773 29.3154;
    1.6454 1.8204 1.9651 2.0980 2.2262 2.3573 2.4894 2.6276 2.7781 2.9422 3.1203 3.3207 3.5533 3.8248 4.1582 4.5841 5.1801 5.6630 6.0927 6.6548 7.4163 7.9350 8.6152 9.5342 11.0701 14.1524 18.0704 24.5288 31.0283;
    1.7365 1.9198 2.0723 2.2131 2.3488 2.4858 2.6243 2.7720 2.9325 3.1031 3.2881 3.5010 3.7430 4.0262 4.3777 4.8287 5.4415 5.9483 6.3971 6.9737 7.7576 8.3032 9.0182 10.0019 11.6009 14.8600 18.8620 25.5071 32.4334;
    1.8147 2.0122 2.1721 2.3189 2.4611 2.6031 2.7503 2.9074 3.0709 3.2457 3.4422 3.6611 3.9126 4.2121 4.5713 5.0378 5.6750 6.2044 6.6732 7.2650 8.0763 8.6464 9.3753 10.4094 12.0813 15.4758 19.5960 26.7946 34.1980;
    1.8885 2.0943 2.2615 2.4139 2.5631 2.7106 2.8665 3.0260 3.1958 3.3796 3.5808 3.8081 4.0680 4.3760 4.7535 5.2360 5.8907 6.4488 6.9175 7.5325 8.3793 8.9701 9.7253 10.8062 12.5261 16.0399 20.2642 27.7341 35.3871;
    1.9583 2.1729 2.3450 2.5034 2.6579 2.8140 2.9729 3.1390 3.3135 3.5040 3.7146 3.9472 4.2180 4.5332 4.9231 5.4149 6.0965 6.6687 7.1528 7.7795 8.6625 9.2599 10.0394 11.1652 12.8873 16.5274 20.8478 28.5023 36.0016;
    2.0258 2.2461 2.4255 2.5907 2.7498 2.9108 3.0740 3.2428 3.4238 3.6203 3.8342 4.0760 4.3540 4.6818 5.0774 5.5857 6.2892 6.8700 7.3763 8.0114 8.9148 9.5201 10.3355 11.4890 13.2183 16.9468 21.3576 29.0501 36.6128;
    2.0897 2.3187 2.5027 2.6711 2.8379 3.0014 3.1685 3.3420 3.5269 3.7303 3.9497 4.1994 4.4829 4.8184 5.2272 5.7396 6.4731 7.0626 7.5745 8.2305 9.1589 9.7789 10.6159 11.8001 13.5858 17.3921 21.9109 29.6504 37.2148;
    2.1485 2.3835 2.5747 2.7492 2.9196 3.0882 3.2586 3.4393 3.6302 3.8358 4.0624 4.3179 4.6064 4.9535 5.3700 5.9027 6.6423 7.2444 7.7634 8.4424 9.3824 10.0311 10.8955 12.1073 13.9221 17.7573 22.4064 30.5054 37.8723;
    2.2057 2.4461 2.6431 2.8232 2.9964 3.1697 3.3454 3.5286 3.7278 3.9373 4.1699 4.4293 4.7286 5.0773 5.5054 6.0520 6.8094 7.4244 7.9536 8.6445 9.6103 10.2947 11.1868 12.3794 14.2562 18.2347 22.9873 31.4836 39.5248;
    2.2598 2.5072 2.7085 2.8943 3.0709 3.2475 3.4273 3.6174 3.8174 4.0336 4.2698 4.5353 4.8454 5.2022 5.6350 6.1934 6.9660 7.5909 8.1311 8.8435 9.8387 10.5213 11.4268 12.6525 14.6256 18.5952 23.5316 32.0858 40.1070
]

# =============================================================================
# Result Types
# =============================================================================

"""
    OnatskiQTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest

Result of the Onatski (2009) dynamic factor rank hypothesis test:

```math
H_0: q = k_0 \\quad \\text{vs.} \\quad H_1: k_0 < q \\le k_1
```

where:
- `q` is the true number of pervasive dynamic factors,
- `k_0` is the null factor rank (`k0 >= 0`),
- `k_1` is the alternative upper bound (`k1 > k0`).

# Fields
- `statistic::T`: Test statistic ``\\hat{R}`` (median across frequency blocks in ``(0, \\pi)``)
- `pvalue::T`: Asymptotic right-tail p-value under the complex ``TW_2`` spacing-ratio law
- `crit_vals::Vector{T}`: Critical values `[cv_10, cv_05, cv_01]` at 10%, 5%, and 1% levels
- `alpha_levels::Vector{T}`: Significance levels `[0.10, 0.05, 0.01]`
- `k0::Int`: Null hypothesis number of dynamic factors
- `k1::Int`: Alternative hypothesis maximum number of dynamic factors
- `n_obs::Int`: Number of time observations ``T``
- `n_vars::Int`: Number of cross-sectional series ``N``
- `bandwidth::Int`: Bartlett lag-window truncation parameter ``M``
- `n_blocks::Int`: Number of disjoint frequency blocks in ``(0, \\pi)``
- `rejected_05::Bool`: `true` if `pvalue < 0.05` (`statistic > crit_vals[2]`)
"""
struct OnatskiQTestResult{T<:AbstractFloat} <: StatsAPI.HypothesisTest
    statistic::T
    pvalue::T
    crit_vals::Vector{T}
    alpha_levels::Vector{T}
    k0::Int
    k1::Int
    n_obs::Int
    n_vars::Int
    bandwidth::Int
    n_blocks::Int
    rejected_05::Bool
end

"""
    AhnHorensteinResult{T<:AbstractFloat}

Result of the Ahn & Horenstein (2013) Eigenvalue Ratio (`ER`) and Growth Ratio (`GR`)
estimators for the number of static factors.

# Fields
- `eigenvalues::Vector{T}`: Descending sample covariance eigenvalues ``\\hat{\\mu}_1 \\ge \\dots \\ge \\hat{\\mu}_N``
- `mock_eigenvalue::T`: Mock zero-factor eigenvalue ``\\hat{\\mu}_0 = V(0) / \\ln(\\min(N, T))``
- `er::Vector{T}`: Eigenvalue Ratio criterion ``ER(k)`` for `k = 0:k_max` (length `k_max + 1`)
- `gr::Vector{T}`: Growth Ratio criterion ``GR(k)`` for `k = 0:k_max` (length `k_max + 1`)
- `r_er::Int`: Estimated number of static factors maximizing `ER(k)`
- `r_gr::Int`: Estimated number of static factors maximizing `GR(k)`
- `k_max::Int`: Maximum candidate number of factors considered
- `n_obs::Int`: Number of time observations ``T``
- `n_vars::Int`: Number of cross-sectional series ``N``
"""
struct AhnHorensteinResult{T<:AbstractFloat}
    eigenvalues::Vector{T}
    mock_eigenvalue::T
    er::Vector{T}
    gr::Vector{T}
    r_er::Int
    r_gr::Int
    k_max::Int
    n_obs::Int
    n_vars::Int
end

"""
    OnatskiEDResult{T<:AbstractFloat}

Result of the Onatski (2010) Edge Distribution (`ED`) estimator for the number
of static factors via iterative threshold calibration near the bulk spectral edge.

# Fields
- `eigenvalues::Vector{T}`: Descending sample covariance eigenvalues ``\\hat{\\mu}_1 \\ge \\dots \\ge \\hat{\\mu}_N``
- `r_ed::Int`: Estimated number of static factors at fixed-point convergence
- `delta::T`: Final edge threshold ``\\hat{\\delta} = 2|\\hat{\\beta}|``
- `iterations::Int`: Number of fixed-point iterations executed
- `converged::Bool`: `true` if `j_{t+1} == j_t` within `max_iter` steps
- `k_max::Int`: Maximum candidate number of factors considered
- `n_obs::Int`: Number of time observations ``T``
- `n_vars::Int`: Number of cross-sectional series ``N``
"""
struct OnatskiEDResult{T<:AbstractFloat}
    eigenvalues::Vector{T}
    r_ed::Int
    delta::T
    iterations::Int
    converged::Bool
    k_max::Int
    n_obs::Int
    n_vars::Int
end

# =============================================================================
# StatsAPI Methods
# =============================================================================

StatsAPI.pvalue(r::OnatskiQTestResult) = r.pvalue
StatsAPI.nobs(r::OnatskiQTestResult) = r.n_obs
StatsAPI.dof(r::OnatskiQTestResult) = r.k1 - r.k0
StatsAPI.nobs(r::AhnHorensteinResult) = r.n_obs
StatsAPI.nobs(r::OnatskiEDResult) = r.n_obs

# =============================================================================
# Internal Helpers (Prefixed _fsel_*)
# =============================================================================

"""Preprocess panel matrix: optional column demeaning and standardization."""
function _fsel_preprocess(X::AbstractMatrix{<:AbstractFloat}, demean::Bool, standardize::Bool)
    X64 = Matrix{Float64}(X)
    if demean
        col_means = mean(X64, dims=1)
        X64 = X64 .- col_means
    end
    if standardize
        col_stds = max.(std(X64, dims=1), 1e-10)
        X64 = X64 ./ col_stds
    end
    X64
end

"""Compute descending eigenvalues of `(X_proc' * X_proc) / T_obs`, padded to length `N`."""
function _fsel_cov_eigenvalues(X_proc::Matrix{Float64})
    T_obs, N = size(X_proc)
    s = svdvals(X_proc)
    m = min(N, T_obs)
    evals = zeros(Float64, N)
    @inbounds for i in 1:m
        evals[i] = (s[i] * s[i]) / Float64(T_obs)
    end
    evals
end

"""Lag-window weight matching `_estimate_spectral_density_lagwindow` in `generalized.jl`."""
function _fsel_kernel_weight(kernel::Symbol, lag::Int, M::Int)
    u = Float64(lag) / Float64(max(M, 1))
    if kernel === :bartlett
        1.0 - u
    elseif kernel === :parzen
        u <= 0.5 ? (1.0 - 6.0 * u^2 + 6.0 * u^3) : 2.0 * (1.0 - u)^3
    else
        0.5 * (1.0 + cos(pi * u))
    end
end

"""Compute lag-window smoothed Hermitian spectral density at a single interior frequency `freq`."""
function _fsel_spectral_at_frequency(X_proc::Matrix{Float64}, M::Int,
                                     kernel::Symbol, freq::Float64)
    T_obs, N = size(X_proc)
    M_eff = min(M, T_obs - 1)
    Xc = X_proc .- mean(X_proc; dims=1)
    S = zeros(ComplexF64, N, N)
    Gamma0 = (Xc' * Xc) / Float64(T_obs)
    S .+= Gamma0
    @inbounds for lag in 1:M_eff
        w = _fsel_kernel_weight(kernel, lag, M_eff)
        w == 0.0 && continue
        Gamma_k = ((@view Xc[1:(T_obs - lag), :])' * (@view Xc[(lag + 1):T_obs, :])) / Float64(T_obs)
        e_pos = cis(-lag * freq)
        e_neg = conj(e_pos)
        for j in 1:N, i in 1:N
            S[i, j] += w * (Gamma_k[i, j] * e_pos + Gamma_k[j, i] * e_neg)
        end
    end
    S ./= (2.0 * pi)
    Matrix{ComplexF64}(Hermitian(S))
end

"""Compute descending real eigenvalues across disjoint frequency blocks in `(0, pi)`."""
function _fsel_block_eigenvalues(X_proc::Matrix{Float64}, M::Int,
                                 n_blocks::Union{Int,Nothing},
                                 kernel::Symbol,
                                 freq::Union{Real,Nothing})
    validate_option(kernel, "kernel", (:bartlett, :parzen, :tukey_hanning))
    T_obs, N = size(X_proc)
    if freq !== nothing
        omega = Float64(freq)
        (0.0 < omega < Float64(pi)) ||
            throw(ArgumentError("freq must lie strictly in (0, pi), got $omega"))
        S_omega = _fsel_spectral_at_frequency(X_proc, M, kernel, omega)
        ev = eigvals(Hermitian(S_omega))
        return [sort(real.(ev); rev=true)]
    end

    # Reuse lag-window spectral estimator from src/factor/generalized.jl
    _, spectral_all = _estimate_spectral_density_lagwindow(X_proc, M, kernel)
    # Strictly interior frequencies 2:M lie in (0, pi)
    n_interior = M - 1
    n_interior >= 1 || throw(ArgumentError("Need at least 1 interior frequency in (0, pi); increase bandwidth >= 2"))

    if n_blocks === nothing
        block_spectra = spectral_all[:, :, 2:M]
        evals_mat, _ = _spectral_eigendecomposition(block_spectra)
        return [Vector{Float64}(@view evals_mat[:, b]) for b in 1:n_interior]
    end

    n_blocks >= 1 || throw(ArgumentError("n_blocks must be >= 1, got $n_blocks"))
    B = min(n_blocks, n_interior)
    block_spectra = Array{ComplexF64}(undef, N, N, B)
    # Partition 2:M into B disjoint contiguous blocks and average Hermitian spectra per block
    @inbounds for b in 1:B
        i_start = 2 + div((b - 1) * n_interior, B)
        i_end = 1 + div(b * n_interior, B)
        count = i_end - i_start + 1
        S_b = zeros(ComplexF64, N, N)
        for idx in i_start:i_end
            S_b .+= @view spectral_all[:, :, idx]
        end
        S_b ./= Float64(count)
        block_spectra[:, :, b] .= Matrix{ComplexF64}(Hermitian(S_b))
    end
    evals_mat, _ = _spectral_eigendecomposition(block_spectra)
    [Vector{Float64}(@view evals_mat[:, b]) for b in 1:B]
end

"""Compute Onatski (2009) eigenvalue-difference ratio statistic for one block."""
function _fsel_ratio_stat(evals::Vector{Float64}, k0::Int, k1::Int)
    scale_tol = 1e-12 * max(abs(evals[1]), 1.0)
    r_max = 0.0
    @inbounds for i in (k0 + 1):k1
        num = max(0.0, evals[i] - evals[i + 1])
        den = evals[i + 1] - evals[i + 2]
        val = if num <= scale_tol && den <= scale_tol
            0.0
        elseif den <= scale_tol
            1e12 * (num / max(abs(evals[1]), 1.0))
        else
            num / den
        end
        if val > r_max
            r_max = val
        end
    end
    r_max
end

"""Simulate `n_sim` Complex GUE (beta = 2) Dumitriu-Edelman tridiagonal ratio draws for `w > 15`."""
function _fsel_gue_fallback_draws(w::Int, n_sim::Int, rng::AbstractRNG)
    n_eig = w + 2
    n_mat = max(800, 30 * n_eig)
    m_corner = min(n_mat, max(200, 12 * n_eig))
    eig_lo = m_corner - n_eig + 1
    eig_hi = m_corner

    d = Vector{Float64}(undef, m_corner)
    e = Vector{Float64}(undef, m_corner - 1)
    draws = Vector{Float64}(undef, n_sim)

    @inbounds for s in 1:n_sim
        for i in 1:m_corner
            d[i] = randn(rng)
        end
        for k in 1:(m_corner - 1)
            e[k] = sqrt(rand(rng, Gamma(Float64(n_mat - k), 1.0)))
        end
        ev = eigvals!(SymTridiagonal(d, e), eig_lo:eig_hi)
        r_max = 0.0
        for idx in 1:w
            hi = n_eig + 1 - idx
            num = ev[hi] - ev[hi - 1]
            den = ev[hi - 1] - ev[hi - 2]
            ratio = den > 1e-14 ? (num / den) : 0.0
            if ratio > r_max
                r_max = ratio
            end
        end
        draws[s] = r_max
    end
    sort!(draws)
    draws
end

"""Compute TW_2 ratio critical values `[cv_10, cv_05, cv_01]` and p-value."""
function _fsel_tw2_crit_and_pvalue(stat::Float64, w::Int;
                                   n_sim::Int=5000,
                                   rng::AbstractRNG=Random.default_rng(),
                                   seed::Union{Integer,Nothing}=nothing)
    if 1 <= w <= size(_FSEL_TW2_QTABLE, 1)
        qrow = @view _FSEL_TW2_QTABLE[w, :]
        pgrid = _FSEL_TW2_PGRID
        crit_vals = Float64[qrow[19], qrow[22], qrow[26]]
        pval = if stat <= 0.0
            1.0
        elseif stat < qrow[1]
            1.0 - pgrid[1] * (stat / qrow[1])
        elseif stat >= qrow[end]
            # Pareto upper tail decay for Cauchy-type eigenvalue-gap ratios
            (1.0 - pgrid[end]) * (qrow[end] / stat)
        else
            idx = clamp(searchsortedlast(qrow, stat), 1, length(qrow) - 1)
            frac = (stat - qrow[idx]) / (qrow[idx + 1] - qrow[idx])
            1.0 - (pgrid[idx] + frac * (pgrid[idx + 1] - pgrid[idx]))
        end
        return clamp(pval, 0.0, 1.0), crit_vals
    end

    # Deterministic GUE tridiagonal fallback for w > 15
    eff_seed = (seed === nothing && rng === Random.default_rng()) ? 200909 : seed
    rng_eff = _resolve_repro_rng(rng, eff_seed)
    draws = _fsel_gue_fallback_draws(w, n_sim, rng_eff)
    crit_vals = Float64[
        quantile(draws, 0.90; sorted=true),
        quantile(draws, 0.95; sorted=true),
        quantile(draws, 0.99; sorted=true)
    ]
    n_above = length(draws) - searchsortedlast(draws, stat)
    pval = clamp(Float64(n_above) / Float64(n_sim), 0.0, 1.0)
    pval, crit_vals
end

"""Construct `OnatskiQTestResult{T}` from precomputed block eigenvalues."""
function _fsel_onatski_q_from_evals(::Type{T},
                                    block_evals::Vector{Vector{Float64}},
                                    k0::Int, k1::Int,
                                    T_obs::Int, N::Int,
                                    bandwidth::Int;
                                    n_sim::Int=5000,
                                    rng::AbstractRNG=Random.default_rng(),
                                    seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    n_blocks = length(block_evals)
    r_blocks = Vector{Float64}(undef, n_blocks)
    @inbounds for b in 1:n_blocks
        r_blocks[b] = _fsel_ratio_stat(block_evals[b], k0, k1)
    end
    stat = median(r_blocks)
    w = k1 - k0
    pval, crit = _fsel_tw2_crit_and_pvalue(stat, w; n_sim=n_sim, rng=rng, seed=seed)
    alpha_levels = T[T(0.10), T(0.05), T(0.01)]
    crit_T = T.(crit)
    rejected_05 = stat > crit[2]
    OnatskiQTestResult{T}(
        T(stat), T(pval), crit_T, alpha_levels,
        k0, k1, T_obs, N, bandwidth, n_blocks, rejected_05
    )
end

# =============================================================================
# Public API: Onatski (2009) Dynamic Factor Test & Selector
# =============================================================================

"""
    onatski_q_test(X, k0, k1; bandwidth=nothing, n_blocks=nothing,
                   kernel=:bartlett, freq=nothing, standardize=true,
                   demean=true, n_sim=5000, rng=Random.default_rng(),
                   seed=nothing) -> OnatskiQTestResult

Test the number of pervasive dynamic factors ``q`` in a large panel ``X`` (``T \\times N``)
for ``H_0: q = k_0`` against ``H_1: k_0 < q \\le k_1`` under Onatski (2009) using the
maximum adjacent eigenvalue-gap ratio of smoothed Hermitian spectral density matrices
on disjoint interior Fourier frequency blocks in ``(0, \\pi)``:

```math
\\hat{R}_b(k_0, k_1) = \\max_{k_0 < i \\le k_1} \\frac{\\hat{\\lambda}_{i,b} - \\hat{\\lambda}_{i+1,b}}{\\hat{\\lambda}_{i+1,b} - \\hat{\\lambda}_{i+2,b}}, \\qquad \\hat{R} = \\operatorname{median}_{1 \\le b \\le B} \\hat{R}_b(k_0, k_1)
```

where:
- `\\hat{R}_b(k_0, k_1)` is the block-level maximum eigenvalue spacing ratio on block `b`,
- `\\hat{R}` is the overall Onatski (2009) test statistic (median across blocks),
- `k_0` is the null hypothesis dynamic factor rank (`k0 >= 0`),
- `k_1` is the upper bound under the alternative (`k0 < k1 <= N - 2`),
- `\\hat{\\lambda}_{i,b}` is the `i`-th largest eigenvalue of the lag-window spectral matrix on frequency block `b`,
- `B` is the number of disjoint frequency blocks in `(0, \\pi)`.

Under ``H_0``, ``\\hat{R}_b`` converges in distribution to the complex Tracy-Widom (``TW_2``)
ratio distribution tabulated for ``w = k_1 - k_0 \\in 1:15``, with a seeded Dumitriu-Edelman
GUE tridiagonal simulation fallback for ``w > 15``.

# Arguments
- `X::AbstractMatrix`: Panel data matrix (`T x N`, observations by variables)
- `k0::Int`: Null number of dynamic factors (`>= 0`)
- `k1::Int`: Alternative maximum number of dynamic factors (`> k0`, `<= N - 2`)
- `bandwidth::Union{Int,Nothing}`: Lag-window truncation `M`; defaults to `clamp(round(Int, sqrt(T)), 3, T - 1)`
- `n_blocks::Union{Int,Nothing}`: Optional number of disjoint frequency blocks in `(0, pi)`
- `kernel::Symbol`: Lag-window kernel (`:bartlett`, `:parzen`, `:tukey_hanning`); default `:bartlett`
- `freq::Union{Real,Nothing}`: Optional single frequency `omega in (0, pi)` instead of averaging blocks
- `standardize::Bool`: Standardize columns to unit variance (default `true`)
- `demean::Bool`: Subtract column means prior to spectral estimation (default `true`)
- `n_sim::Int`: GUE Monte Carlo draws when `k1 - k0 > 15` (default `5000`)
- `rng::AbstractRNG`: Random number generator for the `w > 15` fallback
- `seed::Union{Integer,Nothing}`: Optional seed for the `w > 15` fallback

# Returns
- `OnatskiQTestResult{T}` containing test statistic, p-value, critical values, and rejection flag.

# References
- Onatski, A. (2009). Testing hypotheses about the number of factors in large factor models. *Econometrica*, 77(5), 1447-1479.
"""
function onatski_q_test(X::AbstractMatrix{T}, k0::Int, k1::Int;
                        bandwidth::Union{Int,Nothing}=nothing,
                        n_blocks::Union{Int,Nothing}=nothing,
                        kernel::Symbol=:bartlett,
                        freq::Union{Real,Nothing}=nothing,
                        standardize::Bool=true,
                        demean::Bool=true,
                        n_sim::Int=5000,
                        rng::AbstractRNG=Random.default_rng(),
                        seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(X, "X")
    T_obs, N = size(X)
    T_obs >= 4 || throw(ArgumentError("Need at least 4 time observations, got T=$T_obs"))
    k0 >= 0 || throw(ArgumentError("k0 must be >= 0, got $k0"))
    k1 > k0 || throw(ArgumentError("k1 must be strictly greater than k0 ($k0), got $k1"))
    k1 + 2 <= N || throw(ArgumentError("Need k1 + 2 <= n_vars ($N), got k1=$k1"))
    n_sim >= 100 || throw(ArgumentError("n_sim must be >= 100, got $n_sim"))

    M = if bandwidth === nothing
        clamp(round(Int, sqrt(Float64(T_obs))), 3, T_obs - 1)
    else
        1 <= bandwidth < T_obs ||
            throw(ArgumentError("bandwidth must be in 1:(T - 1) ($(T_obs - 1)), got $bandwidth"))
        (freq === nothing && bandwidth < 2) &&
            throw(ArgumentError("bandwidth must be >= 2 when freq is nothing so interior frequencies in (0, pi) exist"))
        bandwidth
    end

    X_proc = _fsel_preprocess(X, demean, standardize)
    block_evals = _fsel_block_eigenvalues(X_proc, M, n_blocks, kernel, freq)
    _fsel_onatski_q_from_evals(T, block_evals, k0, k1, T_obs, N, M;
                               n_sim=n_sim, rng=rng, seed=seed)
end

function onatski_q_test(X::AbstractMatrix{<:Real}, k0::Int, k1::Int; kwargs...)
    onatski_q_test(Matrix{Float64}(X), k0, k1; kwargs...)
end

"""
    onatski_q_select(X; k_max=min(8, size(X, 2) - 2), alpha=0.05, kwargs...)
    onatski_q_select(X, k_max; alpha=0.05, kwargs...)

Select the number of pervasive dynamic factors ``\\hat{q}`` sequentially upward
from ``k_0 = 0`` to ``k_{\\max} - 1`` by testing ``H_0: q = k_0`` vs. ``H_1: k_0 < q \\le k_{\\max}``
with [`onatski_q_test`](@ref) until ``H_0`` cannot be rejected at significance level `alpha`.

# Arguments
- `X::AbstractMatrix`: Panel data matrix (`T x N`, observations by variables)
- `k_max::Int`: Maximum candidate number of dynamic factors (`1 <= k_max <= N - 2`)
- `alpha::Real`: Significance level in `(0, 1)` for sequential testing (default `0.05`)
- `bandwidth::Union{Int,Nothing}`: Lag-window bandwidth `M` passed to `onatski_q_test`
- `n_blocks::Union{Int,Nothing}`: Number of interior frequency blocks in `(0, pi)`
- `kernel::Symbol`: Lag-window kernel (default `:bartlett`)
- `freq::Union{Real,Nothing}`: Optional single frequency `omega in (0, pi)`
- `standardize::Bool`: Standardize columns to unit variance (default `true`)
- `demean::Bool`: Subtract column means (default `true`)
- `n_sim::Int`: GUE draws when `w > 15` (default `5000`)
- `rng::AbstractRNG`: RNG for the `w > 15` fallback
- `seed::Union{Integer,Nothing}`: Seed for the `w > 15` fallback

# Returns
- `(q_hat::Int, tests::Vector{OnatskiQTestResult{T}})` where `q_hat` is the selected
  dynamic factor rank and `tests` contains each sequential test executed.

# References
- Onatski, A. (2009). Testing hypotheses about the number of factors in large factor models. *Econometrica*, 77(5), 1447-1479.
"""
function onatski_q_select(X::AbstractMatrix{T};
                          k_max::Int=min(8, size(X, 2) - 2),
                          alpha::Real=0.05,
                          bandwidth::Union{Int,Nothing}=nothing,
                          n_blocks::Union{Int,Nothing}=nothing,
                          kernel::Symbol=:bartlett,
                          freq::Union{Real,Nothing}=nothing,
                          standardize::Bool=true,
                          demean::Bool=true,
                          n_sim::Int=5000,
                          rng::AbstractRNG=Random.default_rng(),
                          seed::Union{Integer,Nothing}=nothing) where {T<:AbstractFloat}
    _validate_data(X, "X")
    T_obs, N = size(X)
    T_obs >= 4 || throw(ArgumentError("Need at least 4 time observations, got T=$T_obs"))
    1 <= k_max || throw(ArgumentError("k_max must be >= 1, got $k_max"))
    k_max + 2 <= N || throw(ArgumentError("Need k_max + 2 <= n_vars ($N), got k_max=$k_max"))
    alpha_val = Float64(alpha)
    (0.0 < alpha_val < 1.0) || throw(ArgumentError("alpha must lie in (0, 1), got $alpha"))
    n_sim >= 100 || throw(ArgumentError("n_sim must be >= 100, got $n_sim"))

    M = if bandwidth === nothing
        clamp(round(Int, sqrt(Float64(T_obs))), 3, T_obs - 1)
    else
        1 <= bandwidth < T_obs ||
            throw(ArgumentError("bandwidth must be in 1:(T - 1) ($(T_obs - 1)), got $bandwidth"))
        (freq === nothing && bandwidth < 2) &&
            throw(ArgumentError("bandwidth must be >= 2 when freq is nothing so interior frequencies in (0, pi) exist"))
        bandwidth
    end

    X_proc = _fsel_preprocess(X, demean, standardize)
    block_evals = _fsel_block_eigenvalues(X_proc, M, n_blocks, kernel, freq)

    tests = OnatskiQTestResult{T}[]
    sizehint!(tests, k_max)
    q_hat = k_max
    for k0 in 0:(k_max - 1)
        res = _fsel_onatski_q_from_evals(T, block_evals, k0, k_max, T_obs, N, M;
                                         n_sim=n_sim, rng=rng, seed=seed)
        push!(tests, res)
        if Float64(res.pvalue) >= alpha_val
            q_hat = k0
            break
        end
    end
    q_hat, tests
end

function onatski_q_select(X::AbstractMatrix{T}, k_max::Int; kwargs...) where {T<:AbstractFloat}
    onatski_q_select(X; k_max=k_max, kwargs...)
end

function onatski_q_select(X::AbstractMatrix{<:Real}; kwargs...)
    onatski_q_select(Matrix{Float64}(X); kwargs...)
end

function onatski_q_select(X::AbstractMatrix{<:Real}, k_max::Int; kwargs...)
    onatski_q_select(Matrix{Float64}(X); k_max=k_max, kwargs...)
end

# =============================================================================
# Public API: Ahn & Horenstein (2013) ER & GR
# =============================================================================

"""
    ahn_horenstein(X, k_max=min(10, min(size(X)...) - 2);
                   demean=true, standardize=true) -> AhnHorensteinResult

Estimate the number of static factors in a large panel ``X`` (``T \\times N``) using the
Ahn & Horenstein (2013) Eigenvalue Ratio (`ER`) and Growth Ratio (`GR`) criteria:

```math
ER(k) = \\frac{\\hat{\\mu}_k}{\\hat{\\mu}_{k+1}}, \\qquad GR(k) = \\frac{\\ln(1 + \\hat{\\mu}_k / V(k))}{\\ln(1 + \\hat{\\mu}_{k+1} / V(k+1))}, \\qquad k = 0, \\dots, k_{\\max}
```

where:
- `ER(k)` is the Eigenvalue Ratio criterion at candidate rank `k`,
- `GR(k)` is the Growth Ratio criterion at candidate rank `k`,
- `k_{\\max}` is the maximum number of candidate static factors (`1 <= k_{\\max} <= min(N, T) - 2`),
- `\\hat{\\mu}_j` is the `j`-th largest eigenvalue of `(X' X) / T`,
- `\\hat{\\mu}_0 = V(0) / \\ln(\\min(N, T))` is the mock eigenvalue for `r = 0`,
- `V(k) = \\sum_{j=k+1}^{\\min(N, T)} \\hat{\\mu}_j` is the remaining eigenvalue tail sum.

# Arguments
- `X::AbstractMatrix`: Panel data matrix (`T x N`, observations by variables)
- `k_max::Int`: Maximum candidate number of static factors (`1 <= k_max <= min(N, T) - 2`)
- `demean::Bool`: Subtract column means before decomposition (default `true`)
- `standardize::Bool`: Standardize columns to unit variance (default `true`)

# Returns
- `AhnHorensteinResult{T}` with `er`, `gr`, `r_er`, `r_gr`, `eigenvalues`, and `mock_eigenvalue`.

# References
- Ahn, S. C., & Horenstein, A. R. (2013). Eigenvalue ratio test for the number of factors. *Econometrica*, 81(3), 1203-1227.
"""
function ahn_horenstein(X::AbstractMatrix{T},
                        k_max::Int=min(10, min(size(X)...) - 2);
                        demean::Bool=true,
                        standardize::Bool=true) where {T<:AbstractFloat}
    _validate_data(X, "X")
    T_obs, N = size(X)
    m = min(N, T_obs)
    m >= 3 || throw(ArgumentError("Need min(n_vars, n_obs) >= 3, got $m"))
    1 <= k_max <= m - 2 ||
        throw(ArgumentError("k_max must satisfy 1 <= k_max <= min(N, T) - 2 ($(m - 2)), got $k_max"))

    X_proc = _fsel_preprocess(X, demean, standardize)
    evals = _fsel_cov_eigenvalues(X_proc)
    mu = @view evals[1:m]

    # Cumulative tail sums V(k) = sum_{j=k+1}^m mu_j for k = 0:(k_max + 1)
    V = zeros(Float64, k_max + 2)
    total = sum(mu)
    V[1] = total
    @inbounds for k in 1:(k_max + 1)
        V[k + 1] = max(0.0, V[k] - mu[k])
    end

    tol = 1e-12 * max(mu[1], 1e-15)
    mu_0 = total / log(Float64(m))

    er_vals = zeros(Float64, k_max + 1)
    gr_vals = zeros(Float64, k_max + 1)

    if total > tol
        @inbounds for k in 0:k_max
            mu_k = (k == 0) ? mu_0 : mu[k]
            mu_kp1 = mu[k + 1]
            v_k = V[k + 1]
            v_kp1 = V[k + 2]

            if mu_k <= tol && mu_kp1 <= tol
                er_vals[k + 1] = 0.0
                gr_vals[k + 1] = 0.0
            elseif mu_kp1 <= tol
                # Exact low-rank transition: factor eigenvalue mu_k > 0 followed by zero noise eigenvalues
                er_vals[k + 1] = 1e12 * (mu_k / max(mu[1], 1.0))
                gr_vals[k + 1] = 1e12 * (mu_k / max(mu[1], 1.0))
            else
                er_vals[k + 1] = mu_k / mu_kp1
                # When V(k+1) == 0 while mu_{k+1} > 0 (at k = r_true - 1 in a noiseless panel),
                # ln(1 + mu_{k+1}/V(k+1)) -> +Inf so GR(k) -> 0.
                gr_vals[k + 1] = v_kp1 <= tol ? 0.0 : log1p(mu_k / v_k) / log1p(mu_kp1 / v_kp1)
            end
        end
    end

    r_er = total > tol ? (argmax(er_vals) - 1) : 0
    r_gr = total > tol ? (argmax(gr_vals) - 1) : 0

    AhnHorensteinResult{T}(
        T.(evals), T(mu_0), T.(er_vals), T.(gr_vals),
        r_er, r_gr, k_max, T_obs, N
    )
end

function ahn_horenstein(X::AbstractMatrix{<:Real},
                        k_max::Int=min(10, min(size(X)...) - 2);
                        kwargs...)
    ahn_horenstein(Matrix{Float64}(X), k_max; kwargs...)
end

# =============================================================================
# Public API: Onatski (2010) Edge Distribution (ED)
# =============================================================================

"""
    onatski_ed(X, k_max=min(10, min(size(X)...) - 2);
               max_iter=20, standardize=true, demean=true) -> OnatskiEDResult

Estimate the number of static factors in a large panel ``X`` (``T \\times N``) using the
Onatski (2010) Edge Distribution (`ED`) fixed-point procedure:

```math
\\hat{\\delta} = 2|\\hat{\\beta}|, \\qquad \\hat{r}(\\hat{\\delta}) = \\max\\{r \\le k_{\\max} : \\hat{\\mu}_r - \\hat{\\mu}_{r+1} \\ge \\hat{\\delta}\\}
```

where:
- `\\hat{\\beta}` is the OLS slope from regressing consecutive eigenvalues `\\hat{\\mu}_j, \\dots, \\hat{\\mu}_{j+4}` on a constant and `(j - 1)^{2/3}, \\dots, (j + 3)^{2/3}`,
- `\\hat{\\mu}_r` is the `r`-th largest eigenvalue of `(X' X) / T`,
- `\\hat{\\delta}` is the threshold separating spiked factor eigenvalues from the bulk spectral edge,
- `k_{\\max}` is the upper bound on candidate static factors.

Iteration starts from ``j = k_{\\max} + 1`` and updates ``j \\leftarrow \\hat{r}(\\hat{\\delta}) + 1``
until a fixed point is reached or `max_iter` iterations complete.

# Arguments
- `X::AbstractMatrix`: Panel data matrix (`T x N`, observations by variables)
- `k_max::Int`: Maximum candidate number of static factors (`1 <= k_max <= min(N, T) - 2`)
- `max_iter::Int`: Maximum fixed-point iterations (default `20`)
- `standardize::Bool`: Standardize columns to unit variance (default `true`)
- `demean::Bool`: Subtract column means before decomposition (default `true`)

# Returns
- `OnatskiEDResult{T}` with estimated rank `r_ed`, calibrated threshold `delta`,
  iteration count, convergence status, and eigenvalues.

# References
- Onatski, A. (2010). Determining the number of factors from empirical distribution of eigenvalues. *Review of Economics and Statistics*, 92(4), 1004-1016.
"""
function onatski_ed(X::AbstractMatrix{T},
                    k_max::Int=min(10, min(size(X)...) - 2);
                    max_iter::Int=20,
                    standardize::Bool=true,
                    demean::Bool=true) where {T<:AbstractFloat}
    _validate_data(X, "X")
    T_obs, N = size(X)
    m = min(N, T_obs)
    m >= 3 || throw(ArgumentError("Need min(n_vars, n_obs) >= 3, got $m"))
    1 <= k_max <= m - 2 ||
        throw(ArgumentError("k_max must satisfy 1 <= k_max <= min(N, T) - 2 ($(m - 2)), got $k_max"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be >= 1, got $max_iter"))

    X_proc = _fsel_preprocess(X, demean, standardize)
    evals = _fsel_cov_eigenvalues(X_proc)
    mu = @view evals[1:m]

    tol = 1e-12 * max(mu[1], 1.0)
    j = k_max + 1
    r_hat = 0
    delta = 0.0
    iterations = 0
    converged = false

    for iter in 1:max_iter
        iterations = iter
        j_end = min(m, j + 4)
        n_pts = j_end - j + 1

        beta = 0.0
        if n_pts >= 2
            x_sum = 0.0
            y_sum = 0.0
            @inbounds for idx in j:j_end
                x_sum += Float64(idx - 1)^(2.0 / 3.0)
                y_sum += mu[idx]
            end
            x_mean = x_sum / Float64(n_pts)
            y_mean = y_sum / Float64(n_pts)

            cov_xy = 0.0
            var_x = 0.0
            @inbounds for idx in j:j_end
                dx = Float64(idx - 1)^(2.0 / 3.0) - x_mean
                dy = mu[idx] - y_mean
                cov_xy += dx * dy
                var_x += dx * dx
            end
            beta = var_x > 1e-15 ? (cov_xy / var_x) : 0.0
        end

        delta = 2.0 * abs(beta)
        thresh = max(delta, tol)

        r_new = 0
        @inbounds for r in 1:k_max
            if mu[r] - mu[r + 1] >= thresh
                r_new = r
            end
        end

        r_hat = r_new
        j_new = r_new + 1
        if j_new == j
            converged = true
            break
        end
        j = j_new
    end

    OnatskiEDResult{T}(
        T.(evals), r_hat, T(delta), iterations, converged, k_max, T_obs, N
    )
end

function onatski_ed(X::AbstractMatrix{<:Real},
                    k_max::Int=min(10, min(size(X)...) - 2);
                    kwargs...)
    onatski_ed(Matrix{Float64}(X), k_max; kwargs...)
end

# =============================================================================
# Display & Report Methods
# =============================================================================

function Base.show(io::IO, r::OnatskiQTestResult)
    stars = _significance_stars(r.pvalue)
    spec = Any[
        "H0 / H1"        "q = $(r.k0) vs. $(r.k0) < q <= $(r.k1)";
        "Sample (T x N)" "$(r.n_obs) x $(r.n_vars)";
        "Bandwidth (M)"  "$(r.bandwidth)";
        "Freq. Blocks"   "$(r.n_blocks) in (0, pi)";
        "Statistic R"    "$(_fmt(r.statistic; digits=4))$stars";
        "p-value"        _format_pvalue(r.pvalue);
        "Rejected (5%)"  _yesno(r.rejected_05)
    ]
    _pretty_table(io, spec;
        title="Onatski (2009) Dynamic Factor Test",
        column_labels=["Property", "Value"],
        alignment=[:l, :r],
    )

    crit_tbl = Matrix{Any}(undef, length(r.alpha_levels), 3)
    for (i, a) in enumerate(r.alpha_levels)
        crit_tbl[i, 1] = "$(round(Int, 100 * a))%"
        crit_tbl[i, 2] = _fmt(r.crit_vals[i]; digits=4)
        crit_tbl[i, 3] = _yesno(r.statistic > r.crit_vals[i])
    end
    _pretty_table(io, crit_tbl;
        title="Complex Tracy-Widom (TW2) Critical Values (w = $(r.k1 - r.k0))",
        column_labels=["Level", "Critical Value", "Rejected"],
        alignment=[:l, :r, :r],
    )
end

function Base.show(io::IO, r::AhnHorensteinResult)
    spec = Any[
        "Sample (T x N)"  "$(r.n_obs) x $(r.n_vars)";
        "Max Factors"     "$(r.k_max)";
        "Mock Eigenvalue" _fmt(r.mock_eigenvalue; digits=4);
        "Selected r (ER)" "$(r.r_er)";
        "Selected r (GR)" "$(r.r_gr)"
    ]
    _pretty_table(io, spec;
        title="Ahn & Horenstein (2013) Static Factor Selection",
        column_labels=["Property", "Value"],
        alignment=[:l, :r],
    )

    n_rows = length(r.er)
    crit_tbl = Matrix{Any}(undef, n_rows, 4)
    for idx in 1:n_rows
        k = idx - 1
        mu_k = k == 0 ? r.mock_eigenvalue : r.eigenvalues[k]
        crit_tbl[idx, 1] = k
        crit_tbl[idx, 2] = _fmt(mu_k; digits=4)
        crit_tbl[idx, 3] = _fmt(r.er[idx]; digits=4) * (k == r.r_er ? " *" : "")
        crit_tbl[idx, 4] = _fmt(r.gr[idx]; digits=4) * (k == r.r_gr ? " *" : "")
    end
    _pretty_table(io, crit_tbl;
        title="Eigenvalue Ratio (ER) & Growth Ratio (GR) Criteria (* optimal)",
        column_labels=["k", "mu_k", "ER(k)", "GR(k)"],
        alignment=[:r, :r, :r, :r],
    )
end

function Base.show(io::IO, r::OnatskiEDResult)
    spec = Any[
        "Sample (T x N)"  "$(r.n_obs) x $(r.n_vars)";
        "Max Factors"     "$(r.k_max)";
        "Selected r (ED)" "$(r.r_ed)";
        "Threshold delta" _fmt(r.delta; digits=4);
        "Iterations"      "$(r.iterations)";
        "Converged"       _yesno(r.converged)
    ]
    _pretty_table(io, spec;
        title="Onatski (2010) Edge Distribution (ED) Factor Selection",
        column_labels=["Property", "Value"],
        alignment=[:l, :r],
    )

    n_show = min(r.k_max + 1, length(r.eigenvalues) - 1)
    tbl = Matrix{Any}(undef, n_show, 4)
    for k in 1:n_show
        diff_k = r.eigenvalues[k] - r.eigenvalues[k + 1]
        tbl[k, 1] = k
        tbl[k, 2] = _fmt(r.eigenvalues[k]; digits=4)
        tbl[k, 3] = _fmt(diff_k; digits=4)
        tbl[k, 4] = (k == r.r_ed ? "*" : "")
    end
    _pretty_table(io, tbl;
        title="Leading Eigenvalues & Gaps (* selected r_ed = $(r.r_ed))",
        column_labels=["r", "mu_r", "mu_r - mu_{r+1}", "Selected"],
        alignment=[:r, :r, :r, :c],
    )
end

report(io::IO, r::OnatskiQTestResult) = (show(io, r); nothing)
report(r::OnatskiQTestResult) = report(stdout, r)

report(io::IO, r::AhnHorensteinResult) = (show(io, r); nothing)
report(r::AhnHorensteinResult) = report(stdout, r)

report(io::IO, r::OnatskiEDResult) = (show(io, r); nothing)
report(r::OnatskiEDResult) = report(stdout, r)
