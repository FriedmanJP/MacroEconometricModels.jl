# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Rambachan & Roth (2023) HonestDiD sensitivity analysis.

Constructs robust confidence sets under bounded violations of parallel trends
using the relative magnitudes restriction. Given a maximum violation parameter
`Mbar`, the worst-case bias at event-time `e >= 0` accumulates linearly as
`Mbar * (e + 1)`, widening the original confidence intervals accordingly.

The breakdown value is the smallest `Mbar*` such that the robust CI for at
least one post-treatment period includes zero — i.e., the amount of trend
violation needed to overturn statistical significance.

# References
- Rambachan, A. & Roth, J. (2023). A More Credible Approach to Parallel Trends.
  *Review of Economic Studies* 90(5), 2555-2591.
"""

using Distributions

# =============================================================================
# Public API — DIDResult dispatch
# =============================================================================

"""
    honest_did(result::DIDResult{T}; Mbar::Real=1.0, conf_level::Real=0.95) where {T}

HonestDiD sensitivity analysis for a Difference-in-Differences result.

Constructs robust confidence intervals that account for bounded violations of
the parallel trends assumption under the relative magnitudes restriction
(Rambachan & Roth 2023).

# Arguments
- `result::DIDResult{T}` — DiD estimation result (from `estimate_did` or `callaway_santanna`)
- `Mbar::Real=1.0` — maximum violation bound (relative magnitudes parameter)
- `conf_level::Real=0.95` — confidence level for the robust CIs

# Returns
`HonestDiDResult{T}` with robust CIs, breakdown value, and post-treatment estimates.

# Example
```julia
did = estimate_did(pd, :y, :treat)
h = honest_did(did; Mbar=1.0)
h.breakdown_value  # smallest Mbar that overturns significance
```
"""
function honest_did(result::DIDResult{T}; Mbar::Real=1.0, conf_level::Real=0.95) where {T<:AbstractFloat}
    return _honest_did_internal(
        result.att, result.se, result.event_times, result.reference_period,
        result.ci_lower, result.ci_upper;
        Mbar=T(Mbar), conf_level=T(conf_level)
    )
end

# =============================================================================
# Public API — EventStudyLP dispatch
# =============================================================================

"""
    honest_did(result::EventStudyLP{T}; Mbar::Real=1.0, conf_level::Real=0.95) where {T}

HonestDiD sensitivity analysis for an Event Study LP result.

Constructs robust confidence intervals that account for bounded violations of
the parallel trends assumption under the relative magnitudes restriction
(Rambachan & Roth 2023).

# Arguments
- `result::EventStudyLP{T}` — event study LP result (from `event_study_lp` or `lp_did`)
- `Mbar::Real=1.0` — maximum violation bound (relative magnitudes parameter)
- `conf_level::Real=0.95` — confidence level for the robust CIs

# Returns
`HonestDiDResult{T}` with robust CIs, breakdown value, and post-treatment estimates.

# Example
```julia
eslp = event_study_lp(pd, :y, :treat, 5; leads=3, lags=2)
h = honest_did(eslp; Mbar=0.5)
h.robust_ci_lower  # lower bounds accounting for trend violations
```
"""
function honest_did(result::EventStudyLP{T}; Mbar::Real=1.0, conf_level::Real=0.95) where {T<:AbstractFloat}
    return _honest_did_internal(
        result.coefficients, result.se, result.event_times, result.reference_period,
        result.ci_lower, result.ci_upper;
        Mbar=T(Mbar), conf_level=T(conf_level)
    )
end

# =============================================================================
# Internal implementation
# =============================================================================

"""
    _honest_did_internal(att, se, event_times, reference_period, ci_lo, ci_hi;
                         Mbar, conf_level) -> HonestDiDResult{T}

Core HonestDiD algorithm implementing the relative magnitudes restriction.

Under the relative magnitudes framework, the worst-case bias at post-treatment
event-time `e` (where `e >= 0`) is `Mbar * (e + 1)`. This linear accumulation
reflects the restriction that violations of parallel trends cannot grow faster
than `Mbar` per period.

The robust confidence interval at event-time `e` is:
    [att_e - Mbar*(e+1) - z*se_e,  att_e + Mbar*(e+1) + z*se_e]

The breakdown value is the smallest `Mbar*` such that the robust CI for any
post-treatment period includes zero:
    Mbar* = min_e { max(|att_e| - z*se_e, 0) / (e+1) }
"""
function _honest_did_internal(att::Vector{T}, se::Vector{T}, event_times::Vector{Int},
                               reference_period::Int, ci_lo::Vector{T}, ci_hi::Vector{T};
                               Mbar::T=one(T), conf_level::T=T(0.95)) where {T<:AbstractFloat}

    # Critical value
    z = T(quantile(Normal(), 1 - (1 - conf_level) / 2))

    # Identify post-treatment periods (event_time >= 0)
    post_mask = event_times .>= 0
    post_idx = findall(post_mask)

    n_post = length(post_idx)
    if n_post == 0
        throw(ArgumentError("No post-treatment periods found (event_times >= 0). " *
                            "Cannot compute HonestDiD sensitivity analysis."))
    end

    post_event_times = event_times[post_idx]
    post_att = att[post_idx]
    post_se = se[post_idx]
    orig_ci_lo = ci_lo[post_idx]
    orig_ci_hi = ci_hi[post_idx]

    # Robust CIs under relative magnitudes restriction
    robust_ci_lo = Vector{T}(undef, n_post)
    robust_ci_hi = Vector{T}(undef, n_post)
    breakdown_per_period = Vector{T}(undef, n_post)

    for i in 1:n_post
        e = post_event_times[i]

        # Sensitivity: worst-case bias accumulates linearly
        sensitivity_e = T(e + 1)
        max_bias = Mbar * sensitivity_e

        # Robust CI: widen by max_bias in both directions
        robust_ci_lo[i] = post_att[i] - max_bias - z * post_se[i]
        robust_ci_hi[i] = post_att[i] + max_bias + z * post_se[i]

        # Breakdown value for this period:
        # smallest Mbar* such that robust CI includes zero
        gap = abs(post_att[i]) - z * post_se[i]
        if gap > zero(T)
            breakdown_per_period[i] = gap / sensitivity_e
        else
            # Original CI already includes zero — breakdown is 0
            breakdown_per_period[i] = zero(T)
        end
    end

    # Overall breakdown: the weakest link (smallest across periods)
    breakdown = minimum(breakdown_per_period)

    return HonestDiDResult{T}(
        Mbar, robust_ci_lo, robust_ci_hi, orig_ci_lo, orig_ci_hi,
        breakdown, post_event_times, post_att, conf_level
    )
end
