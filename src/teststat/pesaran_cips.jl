# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
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
Pesaran (2007) CIPS (Cross-sectionally Augmented IPS) panel unit root test.

Augments individual ADF regressions with cross-section averages to account
for cross-sectional dependence.

References:
- Pesaran, M. H. (2007). A simple panel unit root test in the presence of
  cross-section dependence. Journal of Applied Econometrics, 22(2), 265-312.
"""

using LinearAlgebra, Statistics, Distributions

# =============================================================================
# Pesaran CIPS Test
# =============================================================================

"""
    pesaran_cips_test(X::AbstractMatrix{T}; lags=:auto, deterministic=:constant) -> PesaranCIPSResult{T}

Pesaran (2007) CIPS panel unit root test.

Augments individual ADF regressions with cross-sectional averages to
account for cross-sectional dependence.

# Arguments
- `X`: Panel data matrix (T x N), time in rows, units in columns

# Keyword Arguments
- `lags::Union{Int,Symbol}=:auto`: Number of augmenting lags (`:auto` selects via floor(T^(1/3)))
- `deterministic::Symbol=:constant`: Deterministic terms (`:none`, `:constant`, `:trend`)

# Returns
`PesaranCIPSResult{T}` with CIPS statistic, individual CADF statistics,
critical values, and p-value.

# Example
```julia
X = randn(50, 20)
result = pesaran_cips_test(X; lags=1)
result.pvalue  # p-value for panel unit root test
```

# References
- Pesaran, M. H. (2007). Journal of Applied Econometrics, 22(2), 265-312.
"""
function pesaran_cips_test(X::AbstractMatrix{T};
                           lags::Union{Int,Symbol}=:auto,
                           deterministic::Symbol=:constant) where {T<:AbstractFloat}
    T_obs, N = size(X)

    # Validate inputs
    T_obs < 20 && throw(ArgumentError(
        "Time dimension T=$T_obs too small; need at least 20 observations"))
    deterministic in (:none, :constant, :trend) || throw(ArgumentError(
        "deterministic must be :none, :constant, or :trend, got :$deterministic"))

    # Determine lag length
    p = if lags === :auto
        max(1, floor(Int, T_obs^(1/3)))
    else
        lags::Int
    end

    # Cross-sectional averages
    y_bar = vec(mean(X, dims=2))            # T_obs vector
    dy_bar = diff(y_bar)                     # T_obs-1 vector

    # Lagged cross-sectional average
    y_bar_lag = y_bar[1:end-1]               # T_obs-1 vector

    # Compute individual CADF t-statistics
    individual_cadf = Vector{T}(undef, N)

    for i in 1:N
        yi = X[:, i]
        dyi = diff(yi)

        # Effective sample after lags
        start = p + 1
        n_eff = length(dyi) - p

        # Dependent variable: Delta y_{i,t}
        Y = dyi[start:end]

        # Build regressor matrix for CADF regression:
        #   Delta y_{i,t} = [a_i] + b_i * y_{i,t-1} + c_i * ybar_{t-1} + d_i * Delta ybar_t
        #                   + [lag terms] + epsilon_{i,t}
        yi_lag = yi[start:(end-1)]  # y_{i,t-1}, length = T_obs - 1 - p = n_eff

        # Cross-section average regressors
        yb_lag = y_bar_lag[start:end]   # ybar_{t-1}, length = n_eff
        dyb    = dy_bar[start:end]      # Delta ybar_t, length = n_eff

        # Start building regressor list
        regressors = Matrix{T}(undef, n_eff, 0)

        # Deterministic terms
        if deterministic == :constant
            regressors = hcat(regressors, ones(T, n_eff))
        elseif deterministic == :trend
            regressors = hcat(regressors, ones(T, n_eff), T.(1:n_eff))
        end

        # Key regressors: y_{i,t-1}, ybar_{t-1}, Delta ybar_t
        regressors = hcat(regressors, yi_lag, yb_lag, dyb)

        # Index of b_i coefficient (on y_{i,t-1})
        bi_idx = size(regressors, 2) - 2  # third from last

        # Lagged differences of y_i and ybar
        for lag in 1:p
            if start - lag >= 1 && start - lag + n_eff - 1 <= length(dyi)
                dy_lag_col = dyi[(start-lag):(start-lag+n_eff-1)]
                regressors = hcat(regressors, dy_lag_col)
            end
            if start - lag >= 1 && start - lag + n_eff - 1 <= length(dy_bar)
                dyb_lag_col = dy_bar[(start-lag):(start-lag+n_eff-1)]
                regressors = hcat(regressors, dyb_lag_col)
            end
        end

        # OLS regression
        Xreg = regressors
        k = size(Xreg, 2)

        XtX = Xreg'Xreg
        XtX_inv = robust_inv(XtX)
        beta = XtX_inv * (Xreg'Y)
        resid = Y - Xreg * beta

        sigma2 = sum(resid .^ 2) / (n_eff - k)
        se = sqrt.(max.(sigma2 .* diag(XtX_inv), T(1e-20)))

        # t-statistic on b_i
        individual_cadf[i] = beta[bi_idx] / se[bi_idx]
    end

    # Truncation at +/- 6.19 (Pesaran 2007)
    truncation_bound = T(6.19)
    cadf_truncated = clamp.(individual_cadf, -truncation_bound, truncation_bound)

    # CIPS = mean of truncated CADF statistics
    cips = mean(cadf_truncated)

    # Critical values and p-value via table lookup
    cv, pval = _pesaran_cips_critical_values_and_pvalue(cips, N, T_obs, deterministic, T)

    PesaranCIPSResult{T}(
        cips,
        pval,
        individual_cadf,
        cv,
        p,
        deterministic,
        T_obs,
        N
    )
end

# Float64 fallback
pesaran_cips_test(X::AbstractMatrix; kwargs...) = pesaran_cips_test(Float64.(X); kwargs...)

# PanelData dispatch
function pesaran_cips_test(pd::PanelData; kwargs...)
    X = _panel_to_matrix(pd)
    pesaran_cips_test(X; kwargs...)
end

# =============================================================================
# Critical value lookup and p-value interpolation
# =============================================================================

"""Find nearest (N, T) key in PESARAN_CIPS_CV table and return critical values + p-value."""
function _pesaran_cips_critical_values_and_pvalue(cips::T, N::Int, T_obs::Int,
                                                   deterministic::Symbol,
                                                   ::Type{T}) where {T<:AbstractFloat}
    table = PESARAN_CIPS_CV[deterministic]

    # Available N and T values in table
    N_vals = [10, 20, 30, 50, 100]
    T_vals = [20, 30, 50, 70, 100]

    # Find nearest N and T
    N_near = _nearest_val(N, N_vals)
    T_near = _nearest_val(T_obs, T_vals)

    cv_dict = table[(N_near, T_near)]
    cv = Dict{Int,T}(
        1  => T(cv_dict[1]),
        5  => T(cv_dict[5]),
        10 => T(cv_dict[10])
    )

    # P-value interpolation (more negative CIPS = stronger rejection)
    pval = if cips <= cv[1]
        T(0.001)
    elseif cips <= cv[5]
        T(0.01 + 0.04 * (cips - cv[1]) / (cv[5] - cv[1]))
    elseif cips <= cv[10]
        T(0.05 + 0.05 * (cips - cv[5]) / (cv[10] - cv[5]))
    else
        # Above 10% critical value - use linear extrapolation, capped at 1
        T(min(1.0, 0.10 + 0.30 * (cips - cv[10]) / abs(cv[10])))
    end

    cv, pval
end

"""Find nearest value in sorted array."""
function _nearest_val(x::Int, vals::Vector{Int})
    best = vals[1]
    best_dist = abs(x - best)
    for v in vals[2:end]
        d = abs(x - v)
        if d < best_dist
            best = v
            best_dist = d
        end
    end
    best
end

# =============================================================================
# Show method
# =============================================================================

function Base.show(io::IO, r::PesaranCIPSResult{T}) where {T}
    spec_data = Any[
        "H0"          "All units have unit roots (panel non-stationary)";
        "H1"          "Some units are stationary";
        "Deterministic" _regression_name(r.deterministic);
        "Lags"         r.lags;
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Pesaran (2007) CIPS Panel Unit Root Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "CIPS statistic" string(round(r.cips_statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )

    cv_data = Matrix{Any}(undef, 1, 3)
    cv_data[1, :] = [round(r.critical_values[1], digits=3),
                     round(r.critical_values[5], digits=3),
                     round(r.critical_values[10], digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
    )

    reject = r.cips_statistic < r.critical_values[5]
    conclusion = reject ?
        "Reject H0 at 5% level: evidence that some units are stationary" :
        "Fail to reject H0: panel appears non-stationary"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end
