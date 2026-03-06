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
Bai-Perron (1998, 2003) multiple structural break test.
"""

# =============================================================================
# Main Function
# =============================================================================

"""
    bai_perron_test(y, X; max_breaks=5, trimming=0.15, criterion=:bic) -> BaiPerronResult

Bai-Perron (1998, 2003) test for multiple structural breaks in a linear regression.

Tests for `m` unknown structural break points in the linear model `y = X β + u`,
where regression coefficients are allowed to change at each break date. Uses dynamic
programming to find globally optimal break dates minimizing total sum of squared
residuals. Break number is selected by BIC or LWZ information criteria.

# Arguments
- `y::AbstractVector`: Dependent variable (length n)
- `X::AbstractMatrix`: Regressor matrix (n × k)
- `max_breaks::Int=5`: Maximum number of breaks to consider
- `trimming::Real=0.15`: Minimum fraction of observations per segment
- `criterion::Symbol=:bic`: Information criterion for break selection (`:bic` or `:lwz`)

# Returns
`BaiPerronResult{T}` containing estimated break dates, regime coefficients,
sup-F and sequential test statistics, BIC/LWZ values, and confidence intervals.

# Example
```julia
T = 200
X = ones(T, 1)
y = vcat(ones(100) * 2.0, ones(100) * 5.0) + randn(T) * 0.5
result = bai_perron_test(y, X; max_breaks=3)
result.n_breaks   # Should detect 1 break
result.break_dates # Approximately [100]
```

# References
- Bai, J., & Perron, P. (1998). Estimating and testing linear models with multiple
  structural changes. Econometrica, 66(1), 47-78.
- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change
  models. Journal of Applied Econometrics, 18(1), 1-22.
"""
function bai_perron_test(y::AbstractVector{T}, X::AbstractMatrix{T};
                         max_breaks::Int=5, trimming::Real=0.15,
                         criterion::Symbol=:bic) where {T<:AbstractFloat}

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------
    n = length(y)
    k = size(X, 2)
    n >= 20 || throw(ArgumentError("Need at least 20 observations (got n=$n)"))
    size(X, 1) == n || throw(ArgumentError("y (length $n) and X ($(size(X,1)) rows) must have the same number of observations"))
    criterion in (:bic, :lwz) || throw(ArgumentError("criterion must be :bic or :lwz, got :$criterion"))
    max_breaks >= 1 || throw(ArgumentError("max_breaks must be >= 1"))

    # Minimum segment length: at least k+1 observations per segment, or trimming*n
    h = max(k + 1, ceil(Int, trimming * n))
    # Maximum possible breaks given segment length
    max_possible = floor(Int, n / h) - 1
    max_breaks = min(max_breaks, max_possible)

    if max_breaks < 1
        # Cannot test for breaks — return 0-break result
        ssr_0 = _compute_ssr_segment(y, X, 1, n)
        bic_0 = n * log(ssr_0 / n) + k * log(T(n))
        lwz_0 = n * log(ssr_0 / n) + k * T(0.299) * log(T(n))^T(2.1)
        coefs_0 = _segment_ols(y, X, 1, n)
        ses_0 = _segment_se(y, X, 1, n)
        return BaiPerronResult{T}(
            0, Int[], Tuple{Int,Int}[],
            [coefs_0], [ses_0],
            T[], T[], T[], T[],
            [bic_0], [lwz_0],
            T(trimming), n
        )
    end

    # -------------------------------------------------------------------------
    # Step 1: Compute segment SSR matrix
    # -------------------------------------------------------------------------
    ssr_matrix = _compute_segment_ssr(y, X, n, k, h)

    # -------------------------------------------------------------------------
    # Step 2: Full-sample SSR (0 breaks)
    # -------------------------------------------------------------------------
    ssr_0 = _compute_ssr_segment(y, X, 1, n)

    # -------------------------------------------------------------------------
    # Step 3: Dynamic programming for 1..max_breaks
    # -------------------------------------------------------------------------
    optimal_ssrs = Vector{T}(undef, max_breaks)
    optimal_dates = Vector{Vector{Int}}(undef, max_breaks)

    for m in 1:max_breaks
        ssr_m, dates_m = _dp_optimal_breaks(ssr_matrix, n, m, h)
        optimal_ssrs[m] = ssr_m
        optimal_dates[m] = dates_m
    end

    # -------------------------------------------------------------------------
    # Step 4: Information criteria (BIC and LWZ)
    # -------------------------------------------------------------------------
    bic_values = Vector{T}(undef, max_breaks + 1)
    lwz_values = Vector{T}(undef, max_breaks + 1)

    # 0 breaks
    bic_values[1] = n * log(ssr_0 / n) + k * log(T(n))
    lwz_values[1] = n * log(ssr_0 / n) + k * T(0.299) * log(T(n))^T(2.1)

    for m in 1:max_breaks
        n_params = (m + 1) * k
        bic_values[m + 1] = n * log(optimal_ssrs[m] / n) + n_params * log(T(n))
        lwz_values[m + 1] = n * log(optimal_ssrs[m] / n) + n_params * T(0.299) * log(T(n))^T(2.1)
    end

    # Select number of breaks
    if criterion == :bic
        n_breaks = argmin(bic_values) - 1
    else
        n_breaks = argmin(lwz_values) - 1
    end

    # -------------------------------------------------------------------------
    # Step 5: sup-F(l) statistics
    # -------------------------------------------------------------------------
    supf_stats = Vector{T}(undef, max_breaks)
    supf_pvalues = Vector{T}(undef, max_breaks)

    for l in 1:max_breaks
        # F(l) = [(SSR_0 - SSR_l) / (l * k)] / [SSR_l / (n - (l+1)*k)]
        denom_df = n - (l + 1) * k
        if denom_df > 0 && optimal_ssrs[l] > zero(T)
            f_num = (ssr_0 - optimal_ssrs[l]) / (l * k)
            f_den = optimal_ssrs[l] / denom_df
            supf_stats[l] = max(f_num / f_den, zero(T))
        else
            supf_stats[l] = zero(T)
        end
        supf_pvalues[l] = _baiperron_pvalue(supf_stats[l], l, :supf)
    end

    # -------------------------------------------------------------------------
    # Step 6: Sequential sup-F(l+1|l) statistics
    # -------------------------------------------------------------------------
    sequential_stats = Vector{T}(undef, max(0, max_breaks - 1))
    sequential_pvalues = Vector{T}(undef, max(0, max_breaks - 1))

    for l in 1:(max_breaks - 1)
        # sup-F(l+1 | l): test l+1 breaks vs l breaks
        if optimal_ssrs[l] > zero(T)
            denom_df = n - (l + 2) * k
            if denom_df > 0 && optimal_ssrs[l + 1] > zero(T)
                f_num = (optimal_ssrs[l] - optimal_ssrs[l + 1]) / k
                f_den = optimal_ssrs[l + 1] / denom_df
                sequential_stats[l] = max(f_num / f_den, zero(T))
            else
                sequential_stats[l] = zero(T)
            end
        else
            sequential_stats[l] = zero(T)
        end
        sequential_pvalues[l] = _baiperron_pvalue(sequential_stats[l], l + 1, :seqf)
    end

    # -------------------------------------------------------------------------
    # Step 7: Extract break dates, coefficients, SEs, CIs
    # -------------------------------------------------------------------------
    if n_breaks > 0
        break_dates = optimal_dates[n_breaks]
    else
        break_dates = Int[]
    end

    # Regime coefficients and standard errors
    regime_coefs = Vector{Vector{T}}(undef, n_breaks + 1)
    regime_ses = Vector{Vector{T}}(undef, n_breaks + 1)

    segments = _break_segments(break_dates, n)
    for (i, (s, e)) in enumerate(segments)
        regime_coefs[i] = _segment_ols(y, X, s, e)
        regime_ses[i] = _segment_se(y, X, s, e)
    end

    # Break date confidence intervals (Bai 1997)
    break_cis = Vector{Tuple{Int,Int}}(undef, n_breaks)
    for i in 1:n_breaks
        seg_before = i == 1 ? (1, break_dates[1]) : (break_dates[i-1]+1, break_dates[i])
        seg_after = i == n_breaks ? (break_dates[i]+1, n) : (break_dates[i]+1, break_dates[i+1])

        # Coefficient change at break
        delta_coef = regime_coefs[i+1] - regime_coefs[i]

        # Moment matrix Q = X'X/n in neighborhood of break
        n_before = seg_before[2] - seg_before[1] + 1
        n_after = seg_after[2] - seg_after[1] + 1
        X_before = @view X[seg_before[1]:seg_before[2], :]
        X_after = @view X[seg_after[1]:seg_after[2], :]
        Q_hat = (X_before'X_before / n_before + X_after'X_after / n_after) / 2

        # Residual variance at break
        y_before = @view y[seg_before[1]:seg_before[2]]
        y_after = @view y[seg_after[1]:seg_after[2]]
        r_before = y_before - X_before * regime_coefs[i]
        r_after = y_after - X_after * regime_coefs[i+1]
        sigma2 = (dot(r_before, r_before) + dot(r_after, r_after)) / (n_before + n_after - 2*k)

        # CI width: Bai (1997) formula
        dQd = dot(delta_coef, Q_hat * delta_coef)
        if dQd > zero(T)
            ci_width = ceil(Int, T(1.96)^2 * sigma2 / dQd)
        else
            ci_width = max(1, ceil(Int, 0.10 * n))
        end

        lo = max(1, break_dates[i] - ci_width)
        hi = min(n, break_dates[i] + ci_width)
        break_cis[i] = (lo, hi)
    end

    BaiPerronResult{T}(
        n_breaks, break_dates, break_cis,
        regime_coefs, regime_ses,
        supf_stats, supf_pvalues,
        sequential_stats, sequential_pvalues,
        bic_values, lwz_values,
        T(trimming), n
    )
end

# Float64 fallback
function bai_perron_test(y::AbstractVector, X::AbstractMatrix; kwargs...)
    bai_perron_test(Float64.(y), Float64.(X); kwargs...)
end

# =============================================================================
# Internal Helpers
# =============================================================================

"""
    _compute_segment_ssr(y, X, n, k, h)

Compute the segment SSR matrix using triangular recursion.
`ssr_matrix[i, j]` = SSR from regressing y[i:j] on X[i:j, :].
Only entries where j - i + 1 >= h are computed.
"""
function _compute_segment_ssr(y::AbstractVector{T}, X::AbstractMatrix{T},
                              n::Int, k::Int, h::Int) where {T<:AbstractFloat}
    ssr_matrix = fill(T(Inf), n, n)

    for i in 1:n
        for j in (i + h - 1):n
            ssr_matrix[i, j] = _compute_ssr_segment(y, X, i, j)
        end
    end

    ssr_matrix
end

"""
    _compute_ssr_segment(y, X, s, e)

Compute sum of squared residuals from OLS regression y[s:e] on X[s:e, :].
"""
function _compute_ssr_segment(y::AbstractVector{T}, X::AbstractMatrix{T},
                              s::Int, e::Int) where {T<:AbstractFloat}
    y_seg = @view y[s:e]
    X_seg = @view X[s:e, :]
    n_seg = e - s + 1
    k = size(X, 2)

    if n_seg <= k
        return T(Inf)
    end

    XtX = X_seg' * X_seg
    XtX_inv = robust_inv(Matrix{T}(XtX))
    b = XtX_inv * (X_seg' * y_seg)
    resid = y_seg - X_seg * b
    return sum(resid .^ 2)
end

"""
    _dp_optimal_breaks(ssr_matrix, n, m, h)

Dynamic programming to find m optimal break dates minimizing total SSR.
Returns (best_total_ssr, break_dates).
"""
function _dp_optimal_breaks(ssr_matrix::Matrix{T}, n::Int, m::Int, h::Int) where {T<:AbstractFloat}
    # dp[j, l] = minimum SSR using l breaks with the last segment ending at j
    # We track break dates for backtracking

    # For m breaks, we have m+1 segments.
    # Break dates t_1 < t_2 < ... < t_m define segments:
    #   [1, t_1], [t_1+1, t_2], ..., [t_m+1, n]

    if m == 1
        # Simple case: find best single break
        best_ssr = T(Inf)
        best_date = h
        for t in h:(n - h)
            total = ssr_matrix[1, t] + ssr_matrix[t + 1, n]
            if total < best_ssr
                best_ssr = total
                best_date = t
            end
        end
        return (best_ssr, [best_date])
    end

    # General case: dynamic programming
    # cost[l, j] = min SSR for first l segments ending at observation j
    # date[l, j] = optimal previous break date for backtracking
    cost = fill(T(Inf), m + 1, n)
    date = fill(0, m + 1, n)

    # First segment: cost of segment [1, j]
    for j in h:n
        cost[1, j] = ssr_matrix[1, j]
    end

    # Fill DP table
    for l in 2:(m + 1)
        for j in (l * h):n
            best_c = T(Inf)
            best_d = 0
            # Previous break at position t, current segment [t+1, j]
            lo_t = (l - 1) * h
            hi_t = j - h
            for t in lo_t:hi_t
                candidate = cost[l - 1, t] + ssr_matrix[t + 1, j]
                if candidate < best_c
                    best_c = candidate
                    best_d = t
                end
            end
            cost[l, j] = best_c
            date[l, j] = best_d
        end
    end

    # The total SSR with m breaks is cost[m+1, n]
    best_ssr = cost[m + 1, n]

    # Backtrack to find break dates
    break_dates = Vector{Int}(undef, m)
    pos = n
    for l in (m + 1):-1:2
        break_dates[l - 1] = date[l, pos]
        pos = date[l, pos]
    end

    return (best_ssr, break_dates)
end

"""
    _segment_ols(y, X, s, e)

OLS regression on segment [s, e]. Returns coefficient vector.
"""
function _segment_ols(y::AbstractVector{T}, X::AbstractMatrix{T},
                      s::Int, e::Int) where {T<:AbstractFloat}
    y_seg = @view y[s:e]
    X_seg = @view X[s:e, :]
    XtX = X_seg' * X_seg
    XtX_inv = robust_inv(Matrix{T}(XtX))
    Vector{T}(XtX_inv * (X_seg' * y_seg))
end

"""
    _segment_se(y, X, s, e)

OLS standard errors on segment [s, e]. Returns SE vector.
"""
function _segment_se(y::AbstractVector{T}, X::AbstractMatrix{T},
                     s::Int, e::Int) where {T<:AbstractFloat}
    y_seg = @view y[s:e]
    X_seg = @view X[s:e, :]
    n_seg = e - s + 1
    k = size(X, 2)

    XtX = X_seg' * X_seg
    XtX_inv = robust_inv(Matrix{T}(XtX))
    b = XtX_inv * (X_seg' * y_seg)
    resid = y_seg - X_seg * b
    sigma2 = sum(resid .^ 2) / max(1, n_seg - k)
    se = sqrt.(max.(diag(sigma2 * XtX_inv), zero(T)))
    Vector{T}(se)
end

"""
    _break_segments(break_dates, n)

Convert break dates to segment (start, end) pairs.
"""
function _break_segments(break_dates::Vector{Int}, n::Int)
    m = length(break_dates)
    segments = Vector{Tuple{Int,Int}}(undef, m + 1)
    if m == 0
        segments[1] = (1, n)
    else
        segments[1] = (1, break_dates[1])
        for i in 2:m
            segments[i] = (break_dates[i - 1] + 1, break_dates[i])
        end
        segments[m + 1] = (break_dates[m] + 1, n)
    end
    segments
end

"""
    _baiperron_pvalue(stat, l, test_type)

Approximate p-value for Bai-Perron statistics using critical value tables.
Interpolates between 1%, 5%, 10% levels.
"""
function _baiperron_pvalue(stat::T, l::Int, test_type::Symbol) where {T<:AbstractFloat}
    cv_table = test_type == :supf ? BAIPERRON_SUPF_CV : BAIPERRON_SEQF_CV
    l_clamped = clamp(l, 1, 5)

    if !haskey(cv_table, l_clamped)
        return T(NaN)
    end

    cv = cv_table[l_clamped]
    cv1 = T(cv[1])   # 1% critical value
    cv5 = T(cv[5])   # 5% critical value
    cv10 = T(cv[10]) # 10% critical value

    # Larger stat = more evidence for breaks (reject H₀: no breaks)
    if stat >= cv1
        return T(0.001)
    elseif stat >= cv5
        # Interpolate between 1% and 5%
        return T(0.01 + 0.04 * (cv1 - stat) / (cv1 - cv5))
    elseif stat >= cv10
        # Interpolate between 5% and 10%
        return T(0.05 + 0.05 * (cv5 - stat) / (cv5 - cv10))
    else
        # Below 10% critical value
        return T(min(1.0, 0.10 + 0.90 * max(zero(T), (cv10 - stat) / cv10)))
    end
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, r::BaiPerronResult{T}) where {T}
    spec_data = Any[
        "H₀"                "No structural breaks";
        "H₁"                "Multiple structural breaks";
        "Max breaks tested"  length(r.supf_stats);
        "Trimming fraction"  round(r.trimming, digits=2);
        "Observations"       r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Bai-Perron Multiple Structural Break Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    # sup-F statistics
    if !isempty(r.supf_stats)
        n_stats = length(r.supf_stats)
        supf_data = Matrix{Any}(undef, n_stats, 3)
        for i in 1:n_stats
            stars = _significance_stars(r.supf_pvalues[i])
            supf_data[i, 1] = "sup-F($i)"
            supf_data[i, 2] = string(round(r.supf_stats[i], digits=4), " ", stars)
            supf_data[i, 3] = _format_pvalue(r.supf_pvalues[i])
        end
        _pretty_table(io, supf_data;
            title = "sup-F Tests (l breaks vs. 0 breaks)",
            column_labels = ["Test", "Statistic", "P-value"],
            alignment = [:l, :r, :r],
        )
    end

    # Sequential tests
    if !isempty(r.sequential_stats)
        n_seq = length(r.sequential_stats)
        seq_data = Matrix{Any}(undef, n_seq, 3)
        for i in 1:n_seq
            stars = _significance_stars(r.sequential_pvalues[i])
            seq_data[i, 1] = "sup-F($(i+1)|$i)"
            seq_data[i, 2] = string(round(r.sequential_stats[i], digits=4), " ", stars)
            seq_data[i, 3] = _format_pvalue(r.sequential_pvalues[i])
        end
        _pretty_table(io, seq_data;
            title = "Sequential Tests (l+1 breaks vs. l breaks)",
            column_labels = ["Test", "Statistic", "P-value"],
            alignment = [:l, :r, :r],
        )
    end

    # Information criteria
    if !isempty(r.bic_values)
        n_ic = length(r.bic_values)
        ic_data = Matrix{Any}(undef, n_ic, 3)
        bic_best = argmin(r.bic_values) - 1
        lwz_best = argmin(r.lwz_values) - 1
        for i in 1:n_ic
            m = i - 1
            bic_marker = m == bic_best ? " *" : ""
            lwz_marker = m == lwz_best ? " *" : ""
            ic_data[i, 1] = m
            ic_data[i, 2] = string(round(r.bic_values[i], digits=2), bic_marker)
            ic_data[i, 3] = string(round(r.lwz_values[i], digits=2), lwz_marker)
        end
        _pretty_table(io, ic_data;
            title = "Information Criteria (* = selected)",
            column_labels = ["Breaks", "BIC", "LWZ"],
            alignment = [:r, :r, :r],
        )
    end

    # Break dates and CIs
    if r.n_breaks > 0
        bd_data = Matrix{Any}(undef, r.n_breaks, 3)
        for i in 1:r.n_breaks
            bd_data[i, 1] = i
            bd_data[i, 2] = r.break_dates[i]
            bd_data[i, 3] = string("[", r.break_cis[i][1], ", ", r.break_cis[i][2], "]")
        end
        _pretty_table(io, bd_data;
            title = "Estimated Break Dates",
            column_labels = ["Break", "Date", "95% CI"],
            alignment = [:r, :r, :r],
        )
    end

    # Conclusion
    conclusion = if r.n_breaks == 0
        "No structural breaks detected"
    else
        string("Estimated ", r.n_breaks, " structural break", r.n_breaks > 1 ? "s" : "",
               " at observation", r.n_breaks > 1 ? "s " : " ",
               join(r.break_dates, ", "))
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end
