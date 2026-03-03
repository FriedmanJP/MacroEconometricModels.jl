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
Andrews (1993) / Andrews-Ploberger (1994) structural break tests.
"""

"""
    andrews_test(y, X; test=:supwald, trimming=0.15) -> AndrewsResult

Andrews (1993) / Andrews-Ploberger (1994) structural break test for a single
unknown break point in a linear regression model y = X*beta + u.

Tests H0: no structural break vs H1: single structural break at unknown date.

# Arguments
- `y::AbstractVector`: Dependent variable (n x 1)
- `X::AbstractMatrix`: Regressor matrix (n x k)
- `test::Symbol`: Test variant. One of:
  - `:supwald`, `:suplr`, `:suplm` — supremum (Andrews 1993)
  - `:expwald`, `:explr`, `:explm` — exponential average (Andrews-Ploberger 1994)
  - `:meanwald`, `:meanlr`, `:meanlm` — simple average (Andrews-Ploberger 1994)
- `trimming::Real`: Fraction of sample to trim from each end (default: 0.15)

# Returns
`AndrewsResult{T}` with test statistic, p-value, estimated break date, and
full sequence of statistics across candidate break dates.

# Example
```julia
X = hcat(ones(100), randn(100))
y = X * [1.0, 2.0] + randn(100) * 0.5
y[51:end] .+= X[51:end, 2] .* 3.0  # introduce break at t=50
result = andrews_test(y, X; test=:supwald)
result.pvalue < 0.05  # should reject H0
```

# References
- Andrews, D. W. K. (1993). Tests for parameter instability and structural
  change with unknown change point. Econometrica, 61(4), 821-856.
- Andrews, D. W. K., & Ploberger, W. (1994). Optimal tests when a nuisance
  parameter is present only under the alternative. Econometrica, 62(6), 1383-1414.
- Hansen, B. E. (1997). Approximate asymptotic p values for structural-change
  tests. Journal of Business & Economic Statistics, 15(1), 60-67.
"""
function andrews_test(y::AbstractVector{T}, X::AbstractMatrix{T};
                      test::Symbol=:supwald, trimming::Real=0.15) where {T<:AbstractFloat}

    # --- Validate inputs ---
    valid_tests = (:supwald, :suplr, :suplm, :expwald, :explr, :explm,
                   :meanwald, :meanlr, :meanlm)
    test in valid_tests || throw(ArgumentError(
        "test must be one of $valid_tests, got :$test"))

    n = length(y)
    k = size(X, 2)

    n == size(X, 1) || throw(ArgumentError(
        "y (length $n) and X ($(size(X,1)) rows) must have the same number of observations"))

    n >= 20 || throw(ArgumentError(
        "Time series too short (n=$n), need at least 20 observations"))

    T(0) < T(trimming) < T(0.5) || throw(ArgumentError(
        "trimming must be in (0, 0.5), got $trimming"))

    trimming_T = T(trimming)

    # --- Parse test type into base statistic and functional ---
    test_str = string(test)
    functional = Symbol(test_str[1:3])  # :sup, :exp, :mea(n)
    if functional == :mea
        functional = :mean
    end
    base_stat = Symbol(test_str[(functional == :mean ? 5 : 4):end])  # :wald, :lr, :lm

    # --- Compute trimmed range of candidate break dates ---
    t1 = max(k + 1, ceil(Int, trimming_T * n))
    t2 = min(n - k, floor(Int, (one(T) - trimming_T) * n))

    t1 <= t2 || throw(ArgumentError(
        "No valid candidate break dates after trimming. Increase n or decrease trimming."))

    # --- Full-sample OLS (needed for LR and LM baselines) ---
    XtX_full = X' * X
    XtX_full_inv = robust_inv(XtX_full)
    beta_full = XtX_full_inv * (X' * y)
    resid_full = y - X * beta_full
    SSR_full = dot(resid_full, resid_full)

    # --- Compute statistic sequence across candidate break dates ---
    n_candidates = t2 - t1 + 1
    stat_seq = Vector{T}(undef, n_candidates)

    for (idx, tb) in enumerate(t1:t2)
        stat_seq[idx] = _andrews_base_statistic(y, X, tb, n, k,
                                                 base_stat, SSR_full, resid_full)
    end

    # --- Apply functional to the statistic sequence ---
    statistic, break_idx_in_seq = _andrews_functional(stat_seq, functional)

    # Global break index in the original series
    break_index = t1 + break_idx_in_seq - 1
    break_fraction = T(break_index) / T(n)

    # --- Critical values and p-value ---
    cv = _andrews_critical_values(k, functional, T)
    pval = _andrews_pvalue(statistic, k, functional, T)

    AndrewsResult(statistic, pval, break_index, break_fraction, test,
                  cv, stat_seq, trimming_T, n, k)
end

# Float64 fallback for non-float inputs
function andrews_test(y::AbstractVector, X::AbstractMatrix; kwargs...)
    andrews_test(Float64.(y), Float64.(X); kwargs...)
end

# =============================================================================
# Internal: Base statistic computation at a single candidate break date
# =============================================================================

"""Compute Wald, LR, or LM statistic at a single candidate break date `tb`."""
function _andrews_base_statistic(y::AbstractVector{T}, X::AbstractMatrix{T},
                                  tb::Int, n::Int, k::Int,
                                  base_stat::Symbol,
                                  SSR_full::T,
                                  resid_full::AbstractVector{T}) where {T<:AbstractFloat}
    if base_stat == :wald
        return _andrews_wald(y, X, tb, n, k)
    elseif base_stat == :lr
        return _andrews_lr(y, X, tb, n, k, SSR_full)
    else  # :lm
        return _andrews_lm(X, tb, n, k, resid_full)
    end
end

"""Wald statistic: W = (beta1 - beta2)' * [V1 + V2]^{-1} * (beta1 - beta2)."""
function _andrews_wald(y::AbstractVector{T}, X::AbstractMatrix{T},
                       tb::Int, n::Int, k::Int) where {T<:AbstractFloat}
    # Sub-sample 1: observations 1:tb
    X1 = @view X[1:tb, :]
    y1 = @view y[1:tb]
    XtX1 = X1' * X1
    XtX1_inv = robust_inv(XtX1)
    beta1 = XtX1_inv * (X1' * y1)
    resid1 = y1 - X1 * beta1
    sigma2_1 = dot(resid1, resid1) / T(max(tb - k, 1))
    V1 = sigma2_1 .* XtX1_inv

    # Sub-sample 2: observations (tb+1):n
    X2 = @view X[(tb+1):n, :]
    y2 = @view y[(tb+1):n]
    n2 = n - tb
    XtX2 = X2' * X2
    XtX2_inv = robust_inv(XtX2)
    beta2 = XtX2_inv * (X2' * y2)
    resid2 = y2 - X2 * beta2
    sigma2_2 = dot(resid2, resid2) / T(max(n2 - k, 1))
    V2 = sigma2_2 .* XtX2_inv

    # Wald statistic
    diff_beta = beta1 - beta2
    V_sum_inv = robust_inv(Matrix{T}(V1 + V2))
    W = dot(diff_beta, V_sum_inv * diff_beta)
    return max(W, zero(T))
end

"""LR statistic: n * (log(SSR_full/n) - log(SSR_split/n))."""
function _andrews_lr(y::AbstractVector{T}, X::AbstractMatrix{T},
                     tb::Int, n::Int, k::Int, SSR_full::T) where {T<:AbstractFloat}
    # Sub-sample 1
    X1 = @view X[1:tb, :]
    y1 = @view y[1:tb]
    beta1 = robust_inv(X1' * X1) * (X1' * y1)
    resid1 = y1 - X1 * beta1
    SSR1 = dot(resid1, resid1)

    # Sub-sample 2
    X2 = @view X[(tb+1):n, :]
    y2 = @view y[(tb+1):n]
    beta2 = robust_inv(X2' * X2) * (X2' * y2)
    resid2 = y2 - X2 * beta2
    SSR2 = dot(resid2, resid2)

    SSR_split = SSR1 + SSR2

    # Guard against log of non-positive
    if SSR_split <= zero(T) || SSR_full <= zero(T)
        return zero(T)
    end

    LR = T(n) * (log(SSR_full / T(n)) - log(SSR_split / T(n)))
    return max(LR, zero(T))
end

"""LM statistic: score-based test using full-sample residuals."""
function _andrews_lm(X::AbstractMatrix{T}, tb::Int, n::Int, k::Int,
                     resid_full::AbstractVector{T}) where {T<:AbstractFloat}
    # Build the score contributions: X_i * e_i for each regime indicator
    # Under H0, the score vector for a break at tb is based on:
    # S(tb) = sum_{t=1}^{tb} X_t * e_t  (partial sum of score)
    #
    # LM = S(tb)' * V^{-1} * S(tb) where V is the estimated variance of S

    # Full sample sigma^2
    sigma2 = dot(resid_full, resid_full) / T(n)

    # XtX inverse for the full sample
    XtX_full = X' * X
    XtX_full_inv = robust_inv(XtX_full)

    # Partial score: sum of X_t * e_t for t=1..tb
    Xe = X .* resid_full  # n x k matrix of X_t * e_t
    S = vec(sum(@view(Xe[1:tb, :]), dims=1))  # k x 1

    # Variance of partial score under H0:
    # V = sigma^2 * (tb/n) * (1 - tb/n) * XtX
    # So LM = S' * [sigma^2 * (tb/n)(1-tb/n) * XtX]^{-1} * S
    frac = T(tb) / T(n)
    scale = sigma2 * frac * (one(T) - frac)

    if scale <= zero(T)
        return zero(T)
    end

    # LM = (1/scale) * S' * XtX_full_inv * S
    LM = dot(S, XtX_full_inv * S) / scale
    return max(LM, zero(T))
end

# =============================================================================
# Internal: Functional application (sup, exp, mean)
# =============================================================================

"""Apply the sup/exp/mean functional to the statistic sequence.
Returns (functional_value, index_of_supremum)."""
function _andrews_functional(stat_seq::Vector{T}, functional::Symbol) where {T<:AbstractFloat}
    if functional == :sup
        max_val, max_idx = findmax(stat_seq)
        return (max_val, max_idx)
    elseif functional == :exp
        # log-sum-exp trick for numerical stability:
        # log(mean(exp(s/2))) = log(1/m * sum(exp(s/2)))
        # = -log(m) + logsumexp(s/2)
        half_stats = stat_seq ./ T(2)
        max_half = maximum(half_stats)
        log_mean = -log(T(length(stat_seq))) + max_half + log(sum(exp.(half_stats .- max_half)))
        _, sup_idx = findmax(stat_seq)
        return (log_mean, sup_idx)
    else  # :mean
        mean_val = sum(stat_seq) / T(length(stat_seq))
        _, sup_idx = findmax(stat_seq)
        return (mean_val, sup_idx)
    end
end

# =============================================================================
# Internal: Critical values and p-values
# =============================================================================

"""Look up critical values for Andrews / Andrews-Ploberger tests."""
function _andrews_critical_values(k::Int, functional::Symbol, ::Type{T}) where {T<:AbstractFloat}
    # Clamp k to the range covered by the tables (1-10)
    k_clamped = clamp(k, 1, 10)

    cv_table = if functional == :sup
        HANSEN_ANDREWS_CV
    elseif functional == :exp
        ANDREWS_PLOBERGER_EXP_CV
    else  # :mean
        ANDREWS_PLOBERGER_MEAN_CV
    end

    raw = cv_table[k_clamped]
    Dict{Int,T}(level => T(val) for (level, val) in raw)
end

"""Compute p-value by interpolation from Hansen (1997) critical value tables.
Higher statistic = more significant (chi-squared-like direction)."""
function _andrews_pvalue(stat::T, k::Int, functional::Symbol, ::Type{T2}=T) where {T<:AbstractFloat, T2}
    cv = _andrews_critical_values(k, functional, T)

    if stat >= cv[1]
        # Beyond 1% critical value
        return T(0.005)
    elseif stat >= cv[5]
        # Interpolate between 1% and 5%
        return T(0.01 + 0.04 * (cv[1] - stat) / (cv[1] - cv[5]))
    elseif stat >= cv[10]
        # Interpolate between 5% and 10%
        return T(0.05 + 0.05 * (cv[5] - stat) / (cv[5] - cv[10]))
    else
        # Below 10% critical value — extrapolate toward 1.0
        # Use a smooth extrapolation: the further below cv[10], the closer to 1.0
        if cv[10] <= zero(T)
            return T(0.50)
        end
        excess = (cv[10] - stat) / cv[10]
        return T(min(1.0, 0.10 + 0.90 * min(1.0, excess)))
    end
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, r::AndrewsResult{T}) where {T}
    test_label = Dict(
        :supwald => "Sup-Wald", :suplr => "Sup-LR", :suplm => "Sup-LM",
        :expwald => "Exp-Wald", :explr => "Exp-LR", :explm => "Exp-LM",
        :meanwald => "Mean-Wald", :meanlr => "Mean-LR", :meanlm => "Mean-LM",
    )
    label = get(test_label, r.test_type, string(r.test_type))

    spec_data = Any[
        "H₀"                "No structural break";
        "H₁"                "Single structural break at unknown date";
        "Test type"          label;
        "Parameters tested"  r.n_params;
        "Trimming fraction"  round(r.trimming, digits=2);
        "Observations"       r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Andrews (1993) Structural Break Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )

    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "Test statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue);
        "Break date (index)" r.break_index;
        "Break fraction" round(r.break_fraction, digits=3)
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

    reject = r.pvalue < 0.05
    conclusion = reject ?
        "Reject H₀ at 5% level: evidence of a structural break at observation $(r.break_index)" :
        "Fail to reject H₀: no significant structural break detected"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end
