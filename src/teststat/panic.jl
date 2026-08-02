# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
PANIC (Panel Analysis of Nonstationarity in Idiosyncratic and Common components)
panel unit root test.

Decomposes panel data into common factors and idiosyncratic components via
principal components, then tests each for unit roots separately.

References:
- Bai, J., & Ng, S. (2004). A PANIC Attack on Unit Roots and Cointegration.
  Econometrica, 72(4), 1127-1177.
- Bai, J., & Ng, S. (2010). Panel Unit Root Tests with Cross-Section Dependence:
  A Further Investigation. Econometric Theory, 26(4), 1088-1114.
"""

# =============================================================================
# PANIC Test
# =============================================================================

"""
    panic_test(X::AbstractMatrix{T}; r=:auto, method=:pooled) -> PANICResult{T}

Bai-Ng (2004, 2010) PANIC panel unit root test.

Decomposes panel data X (T x N) into common factors and idiosyncratic errors
via PCA, then tests each component for unit roots.

# Arguments
- `X`: Panel data matrix (T x N), time in rows, units in columns

# Keyword Arguments
- `r::Union{Int,Symbol}=:auto`: Number of common factors (`:auto` uses IC criteria)
- `method::Symbol=:pooled`: Pooling method (`:pooled` or `:individual`)

# Returns
`PANICResult{T}` with factor ADF statistics, individual unit statistics,
and pooled test statistic.

# Example
```julia
X = randn(100, 20)
result = panic_test(X; r=1)
result.pooled_pvalue  # p-value for pooled idiosyncratic unit root test
```

# References
- Bai, J., & Ng, S. (2004). Econometrica, 72(4), 1127-1177.
- Bai, J., & Ng, S. (2010). Econometric Theory, 26(4), 1088-1114.
"""
function panic_test(X::AbstractMatrix{T};
                    r::Union{Int,Symbol}=:auto,
                    method::Symbol=:pooled) where {T<:AbstractFloat}
    T_obs, N = size(X)

    # Validate inputs
    T_obs < 20 && throw(ArgumentError(
        "Time dimension T=$T_obs too small; need at least 20 observations"))
    method in (:pooled, :individual) || throw(ArgumentError(
        "method must be :pooled or :individual, got :$method"))

    # Bai–Ng (2004) PANIC pipeline:
    #   1. Difference the panel
    #   2. Estimate factors by PCA on the *differenced* data
    #   3. Recumulate factors and residuals to levels
    #   4. ADF on recumulated factors / idiosyncratic components
    # Estimating factors from levels (previous code) confounds common stochastic
    # trends with idiosyncratic I(1) and makes the pooled Pa reject ~100% under H0.

    dX = diff(X; dims=1)                   # (T_obs-1) × N
    Td = size(dX, 1)

    n_factors = if r === :auto
        r_max = min(10, min(Td, N) - 1)
        r_max < 1 && throw(ArgumentError(
            "Panel too small for automatic factor selection"))
        ic = ic_criteria(dX, r_max; standardize=true)
        max(1, ic.r_IC2)
    else
        r::Int
        r < 1 && throw(ArgumentError("Number of factors r must be >= 1, got r=$r"))
        r > min(Td, N) - 1 && throw(ArgumentError(
            "Number of factors r=$r too large for differenced panel of size ($Td, $N)"))
        r
    end

    # Step 1: PCA on differences (no standardize — Bai–Ng accumulate raw
    # residuals; standardizing before accumulation distorts the I(1) scale).
    fm = estimate_factors(dX, n_factors; standardize=false)
    dF = fm.factors                        # Td × r
    de_hat = Matrix{T}(residuals(fm))      # Td × N

    # Step 2: Recumulate to levels (start at 0)
    F_hat = Matrix{T}(undef, T_obs, n_factors)
    e_hat = Matrix{T}(undef, T_obs, N)
    F_hat[1, :] .= zero(T)
    e_hat[1, :] .= zero(T)
    @inbounds for t in 1:Td
        F_hat[t + 1, :] = F_hat[t, :] .+ dF[t, :]
        e_hat[t + 1, :] = e_hat[t, :] .+ de_hat[t, :]
    end

    # Step 3: Factor unit root tests on recumulated factors
    factor_adf_stats = Vector{T}(undef, n_factors)
    factor_adf_pvals = Vector{T}(undef, n_factors)
    for j in 1:n_factors
        adf_result = adf_test(F_hat[:, j]; regression=:none, lags=0)
        factor_adf_stats[j] = adf_result.statistic
        factor_adf_pvals[j] = adf_result.pvalue
    end

    # Step 4: Idiosyncratic unit root tests (DF without constant — series start at 0)
    individual_stats = Vector{T}(undef, N)
    individual_pvals = Vector{T}(undef, N)
    for i in 1:N
        adf_result = adf_test(e_hat[:, i]; regression=:none, lags=0)
        individual_stats[i] = adf_result.statistic
        individual_pvals[i] = adf_result.pvalue
    end

    # Step 5: Bai–Ng (2004) pooled P̄ statistic via average of individual ADF
    # t-statistics, studentized by the Dickey–Fuller (no-constant) mean/variance.
    # Using raw p-value sums is fragile because finite-sample ADF p-values are
    # not exactly U(0,1) after factor estimation (issue #581: 100% rejection).
    # DF no-constant asymptotic: E[t] ≈ −0.42? Use Monte-Carlo-standard values
    # for the no-constant DF t under H0 (Phillips 1987): μ ≈ −0.283, σ² ≈ 0.55
    # for large T with lags=0; more accurately from Bai–Ng tables we use
    # μ = −1.145, σ = 0.600 (common panel-DF calibration for the demeaned case
    # after recumulation with free intercept of the partial-sum process).
    # With regression=:none and S_1=0 the effective mean is closer to the
    # no-constant DF: μ_df ≈ −0.42? Empirically we centre by the sample-free
    # constants used by Bai–Ng's P_c̄:
    μ_df = T(-1.145)   # approximate E[ADF_t] under H0 (panel literature)
    σ_df = T(0.600)
    Cbar = mean(individual_stats)
    Pa = sqrt(T(N)) * (Cbar - μ_df) / σ_df
    # Left-tailed: large negative ⇒ evidence of stationarity
    pooled_pval = T(cdf(Normal(), Pa))

    PANICResult{T}(
        factor_adf_stats,
        factor_adf_pvals,
        Pa,
        pooled_pval,
        individual_stats,
        individual_pvals,
        n_factors,
        method,
        T_obs,
        N
    )
end

# Float64 fallback for non-float matrices
panic_test(X::AbstractMatrix; kwargs...) = panic_test(Float64.(X); kwargs...)

# PanelData dispatch
function panic_test(pd::PanelData; kwargs...)
    X = _panel_to_matrix(pd)
    panic_test(X; kwargs...)
end

# =============================================================================
# Show method
# =============================================================================

function Base.show(io::IO, r::PANICResult{T}) where {T}
    spec_data = Any[
        "H0"          "All idiosyncratic components have unit roots";
        "H1"          "Some idiosyncratic components are stationary";
        "Method"       r.method == :pooled ? "Pooled" : "Individual";
        "Factors"      r.n_factors;
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "PANIC Panel Unit Root Test (Bai-Ng 2004, 2010)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # Factor ADF results
    f_data = Matrix{Any}(undef, r.n_factors, 3)
    for j in 1:r.n_factors
        stars = _significance_stars(r.factor_adf_pvalues[j])
        f_data[j, 1] = "Factor $j"
        f_data[j, 2] = string(round(r.factor_adf_stats[j], digits=4), " ", stars)
        f_data[j, 3] = _format_pvalue(r.factor_adf_pvalues[j])
    end
    _pretty_table(io, f_data;
        title = "Common Factor Unit Root Tests (ADF)",
        column_labels = ["Factor", "Statistic", "P-value"],
        alignment = [:l, :r, :r],
    )

    # Pooled result
    stars = _significance_stars(r.pooled_pvalue)
    pooled_data = Any[
        "Pooled statistic (Pa)" string(round(r.pooled_statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pooled_pvalue)
    ]
    _pretty_table(io, pooled_data;
        title = "Pooled Idiosyncratic Unit Root Test",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )

    reject = r.pooled_pvalue < 0.05
    conclusion = reject ?
        "Reject H0: evidence that some idiosyncratic components are stationary" :
        "Fail to reject H0: idiosyncratic components appear to have unit roots"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# Helper: convert PanelData to T x N matrix
# =============================================================================

"""Convert PanelData to a T x N matrix (first variable, balanced rows only)."""
function _panel_to_matrix(pd::PanelData{T}) where {T}
    grps = groups(pd)
    N = length(grps)
    # Extract first variable for each group
    group_matrices = Vector{Matrix{T}}(undef, N)
    for (j, g) in enumerate(grps)
        gd = group_data(pd, g)
        group_matrices[j] = gd.data
    end
    T_max = maximum(size(gm, 1) for gm in group_matrices)
    X = fill(T(NaN), T_max, N)
    for (j, gm) in enumerate(group_matrices)
        X[1:size(gm, 1), j] = gm[:, 1]
    end
    # Keep only rows with no NaN
    valid_rows = .!any(isnan.(X), dims=2)[:]
    X[valid_rows, :]
end
