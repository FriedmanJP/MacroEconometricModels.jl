# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Moon-Perron (2004) panel unit root test.

Uses factor-adjusted t-statistics for testing the unit root null in panels
with cross-sectional dependence. Reports two modified t-statistics (t_a^*
and t_b^*) with bias and variance corrections.

References:
- Moon, H. R., & Perron, B. (2004). Testing for a unit root in panels with
  dynamic factors. Journal of Econometrics, 122(1), 81-126.
"""

# =============================================================================
# Moon-Perron Test
# =============================================================================

"""
    moon_perron_test(X::AbstractMatrix{T}; r=:auto) -> MoonPerronResult{T}

Moon-Perron (2004) panel unit root test with factor adjustment.

Projects out common factors from panel data, then constructs modified
pooled t-statistics with bias and variance corrections.

# Arguments
- `X`: Panel data matrix (T x N), time in rows, units in columns

# Keyword Arguments
- `r::Union{Int,Symbol}=:auto`: Number of common factors (`:auto` uses IC criteria)

# Returns
`MoonPerronResult{T}` with two modified t-statistics (t_a^*, t_b^*) and
their p-values (standard normal under H0).

# Example
```julia
X = randn(80, 15)
result = moon_perron_test(X; r=1)
result.pvalue_a  # p-value for t_a^* statistic
result.pvalue_b  # p-value for t_b^* statistic
```

# References
- Moon, H. R., & Perron, B. (2004). Journal of Econometrics, 122(1), 81-126.
"""
function moon_perron_test(X::AbstractMatrix{T};
                          r::Union{Int,Symbol}=:auto) where {T<:AbstractFloat}
    T_obs, N = size(X)

    # Validate inputs
    T_obs < 20 && throw(ArgumentError(
        "Time dimension T=$T_obs too small; need at least 20 observations"))

    # Determine number of factors
    n_factors = if r === :auto
        r_max = min(10, min(T_obs, N) - 1)
        r_max < 1 && throw(ArgumentError(
            "Panel too small for automatic factor selection"))
        ic = ic_criteria(X, r_max; standardize=true)
        max(1, ic.r_IC2)
    else
        r::Int
        r < 1 && throw(ArgumentError("Number of factors r must be >= 1, got r=$r"))
        r > min(T_obs, N) - 1 && throw(ArgumentError(
            "Number of factors r=$r too large for panel of size ($T_obs, $N)"))
        r
    end

    # Step 1: Estimate factors and de-factor the data
    fm = estimate_factors(X, n_factors; standardize=true)
    Lambda_hat = fm.loadings  # N x r

    # Projection matrix: Q_perp = I - Lambda (Lambda'Lambda)^{-1} Lambda'
    LtL = Lambda_hat' * Lambda_hat
    LtL_inv = robust_inv(LtL)
    Q_perp = Matrix{T}(I, N, N) - Lambda_hat * LtL_inv * Lambda_hat'

    # De-factored data: X* = X * Q_perp' (project out factor space from each row)
    X_star = X * Q_perp'

    # Step 2: Pooled AR(1) estimation on de-factored data
    # For each unit i: x*_{i,t} = rho_i * x*_{i,t-1} + u_{i,t}
    # Pooled estimator: rho_pool = sum_i(x*_{i,t-1}' x*_{i,t}) / sum_i(x*_{i,t-1}' x*_{i,t-1})

    numerator_rho = zero(T)
    denominator_rho = zero(T)
    numerator_t = zero(T)

    # Per-unit quantities for bias/variance corrections
    sigma2_hat = Vector{T}(undef, N)       # innovation variance per unit
    omega2_hat = Vector{T}(undef, N)       # long-run variance per unit
    phi4_hat = Vector{T}(undef, N)         # fourth moment for variance correction

    for i in 1:N
        xi = X_star[:, i]
        xi_lag = xi[1:end-1]
        xi_cur = xi[2:end]
        T_eff = length(xi_cur)

        # OLS: x_{t} = rho * x_{t-1} + u_t
        sum_xy = dot(xi_lag, xi_cur)
        sum_xx = dot(xi_lag, xi_lag)

        numerator_rho += sum_xy
        denominator_rho += sum_xx

        # Residuals under unit root null (rho = 1)
        ui = xi_cur - xi_lag  # first differences under H0
        sigma2_hat[i] = sum(ui .^ 2) / T_eff

        # Long-run variance via Bartlett kernel
        bw = _nw_bandwidth(ui)
        omega2_hat[i] = _long_run_variance(ui, bw)

        # For t-statistic pooling
        rho_i = sum_xx > T(1e-20) ? sum_xy / sum_xx : one(T)
        resid_i = xi_cur - rho_i * xi_lag
        sig2_i = sum(resid_i .^ 2) / T_eff
        se_rho_i = sqrt(max(sig2_i / max(sum_xx, T(1e-20)), T(1e-20)))
        numerator_t += (rho_i - one(T)) / se_rho_i

        # Fourth moment for variance formula
        phi4_hat[i] = omega2_hat[i]^2
    end

    rho_pool = denominator_rho > T(1e-20) ? numerator_rho / denominator_rho : one(T)
    T_eff = T_obs - 1

    # Step 3: Bias and variance corrections
    # Ratio of long-run to short-run variance
    omega2_mean = mean(omega2_hat)
    sigma2_mean = mean(sigma2_hat)
    phi4_mean = mean(phi4_hat)

    # Bias correction terms (Moon-Perron 2004, Theorem 1)
    # t*_a = (sqrt(N) * T * (rho_pool - 1) - correction_a) / se_a
    # t*_b = (sqrt(N) * t_pool - correction_b) / se_b

    # Correction for t*_a
    correction_a = sqrt(T(N)) * T_eff * (omega2_mean - sigma2_mean) / (T(2) * omega2_mean)

    # Variance for t*_a
    ratio_a = phi4_mean / omega2_mean^2
    se_a = sqrt(max(ratio_a, T(1e-10))) * T_eff

    # Compute t*_a
    t_a_raw = sqrt(T(N)) * T_eff * (rho_pool - one(T))
    t_a_star = se_a > T(1e-20) ? (t_a_raw - correction_a) / se_a : zero(T)

    # For t*_b: pooled t-statistic
    t_pool = numerator_t / N  # average of individual t-statistics

    # Correction for t*_b
    correction_b = sqrt(T(N)) * (omega2_mean - sigma2_mean) / (T(2) * sqrt(omega2_mean * sigma2_mean + T(1e-20)))

    # Variance for t*_b
    se_b = sqrt(max(phi4_mean / (T(4) * omega2_mean * sigma2_mean + T(1e-20)), T(1e-10)))

    # Compute t*_b
    t_b_star = se_b > T(1e-20) ? (sqrt(T(N)) * t_pool - correction_b) / se_b : zero(T)

    # P-values: left-tailed, N(0,1) under H0 (reject for large negative values)
    pvalue_a = T(cdf(Normal(), t_a_star))
    pvalue_b = T(cdf(Normal(), t_b_star))

    MoonPerronResult{T}(
        t_a_star,
        t_b_star,
        pvalue_a,
        pvalue_b,
        n_factors,
        T_obs,
        N
    )
end

# Float64 fallback
moon_perron_test(X::AbstractMatrix; kwargs...) = moon_perron_test(Float64.(X); kwargs...)

# PanelData dispatch
function moon_perron_test(pd::PanelData; kwargs...)
    X = _panel_to_matrix(pd)
    moon_perron_test(X; kwargs...)
end

# =============================================================================
# Show method
# =============================================================================

function Base.show(io::IO, r::MoonPerronResult{T}) where {T}
    spec_data = Any[
        "H0"          "All units have unit roots (panel non-stationary)";
        "H1"          "Some units are stationary";
        "Factors"      r.n_factors;
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Moon-Perron (2004) Panel Unit Root Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    stars_a = _significance_stars(r.pvalue_a)
    stars_b = _significance_stars(r.pvalue_b)
    results_data = Any[
        "t*_a statistic" string(round(r.t_a_statistic, digits=4), " ", stars_a) _format_pvalue(r.pvalue_a);
        "t*_b statistic" string(round(r.t_b_statistic, digits=4), " ", stars_b) _format_pvalue(r.pvalue_b)
    ]
    _pretty_table(io, results_data;
        title = "Modified t-Statistics (N(0,1) under H0)",
        column_labels = ["Statistic", "Value", "P-value"],
        alignment = [:l, :r, :r],
    )

    reject_a = r.pvalue_a < 0.05
    reject_b = r.pvalue_b < 0.05
    conclusion = if reject_a && reject_b
        "Both statistics reject H0: strong evidence against panel unit root"
    elseif reject_a || reject_b
        "One statistic rejects H0: moderate evidence against panel unit root"
    else
        "Fail to reject H0: panel appears non-stationary"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# Convenience: Panel Unit Root Summary
# =============================================================================

"""
    panel_unit_root_summary(X; r=:auto, lags=:auto) -> Nothing

Print a summary of three panel unit root tests: PANIC (Bai-Ng 2004),
Pesaran CIPS (2007), and Moon-Perron (2004).

# Arguments
- `X::AbstractMatrix`: Panel data (T × N)
- `r`: Number of factors for PANIC and Moon-Perron (`:auto` for IC selection)
- `lags`: Number of lags for CIPS (`:auto` for T^{1/3} rule)

# Example
```julia
X = randn(100, 20)
panel_unit_root_summary(X; r=1)
```
"""
function panel_unit_root_summary(X::AbstractMatrix; r::Union{Int,Symbol}=:auto,
                                  lags::Union{Int,Symbol}=:auto)
    panel_unit_root_summary(stdout, X; r=r, lags=lags)
end

function panel_unit_root_summary(io::IO, X::AbstractMatrix; r::Union{Int,Symbol}=:auto,
                                  lags::Union{Int,Symbol}=:auto)
    println(io, "\n", "="^60)
    println(io, "  Panel Unit Root Test Battery")
    println(io, "="^60, "\n")

    # PANIC
    try
        r_panic = panic_test(X; r=r, method=:pooled)
        show(io, r_panic)
        println(io)
    catch e
        println(io, "PANIC test failed: ", sprint(showerror, e))
    end

    # Pesaran CIPS
    try
        r_cips = pesaran_cips_test(X; lags=lags, deterministic=:constant)
        show(io, r_cips)
        println(io)
    catch e
        println(io, "Pesaran CIPS test failed: ", sprint(showerror, e))
    end

    # Moon-Perron
    try
        r_mp = moon_perron_test(X; r=r)
        show(io, r_mp)
        println(io)
    catch e
        println(io, "Moon-Perron test failed: ", sprint(showerror, e))
    end

    nothing
end

# PanelData dispatch
function panel_unit_root_summary(pd::PanelData; kwargs...)
    X = _panel_to_matrix(pd)
    panel_unit_root_summary(X; kwargs...)
end
