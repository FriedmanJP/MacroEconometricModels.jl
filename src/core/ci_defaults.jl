# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.

# =============================================================================
# Central Confidence & Credible Interval Defaults (CI-01 / #936)
# =============================================================================
# Package-wide default confidence and credible interval coverage policy:
#   - Macroeconometrics, time series, dynamic, structural, Bayesian models,
#     and LP-DiD: 68% (`DEFAULT_CI_MACRO = 0.68`, `[0.16, 0.5, 0.84]`).
#   - Microeconometrics, cross-sectional, discrete choice, count, quantile,
#     micro-panel, RDD, and non-LP DiD: 90% (`DEFAULT_CI_MICRO = 0.90`,
#     `[0.05, 0.5, 0.95]`).
#   - Hypothesis test significance thresholds (unit-root, cointegration,
#     stability, bounds, Hausman/Sargan/J tests) and VaR/ES tail probabilities
#     remain at 5% (`0.05`).

"""
    DEFAULT_CI_MACRO = 0.68

Default confidence and credible interval coverage (`0.68`) for macroeconometric,
time-series, dynamic, structural, Bayesian, and LP-DiD estimators, tables,
summaries, and plots.
"""
const DEFAULT_CI_MACRO = 0.68

"""
    DEFAULT_CI_MICRO = 0.90

Default confidence interval coverage (`0.90`) for microeconometric, cross-sectional,
discrete-choice, count, quantile, micro-panel, RDD, GMM/SMM, and non-LP DiD
estimators, tables, summaries, and plots.
"""
const DEFAULT_CI_MICRO = 0.90

"""
    DEFAULT_QUANTILES_MACRO = [0.16, 0.5, 0.84]

Default `[lower, median, upper]` credible-band quantiles corresponding to
[`DEFAULT_CI_MACRO`](@ref) (`68%`).
"""
const DEFAULT_QUANTILES_MACRO = [0.16, 0.5, 0.84]

"""
    DEFAULT_QUANTILES_MICRO = [0.05, 0.5, 0.95]

Default `[lower, median, upper]` quantiles corresponding to
[`DEFAULT_CI_MICRO`](@ref) (`90%`).
"""
const DEFAULT_QUANTILES_MICRO = [0.05, 0.5, 0.95]

"""
    _validate_conf_level(level::Real) -> Float64

Validate that `level` is a finite, non-boolean real number strictly in `(0, 1)`,
returning `Float64(level)`. Throws `ArgumentError` otherwise.
"""
function _validate_conf_level(level::Real)
    if level isa Bool
        throw(ArgumentError("confidence level must be a real number in (0, 1), got Bool ($level)"))
    end
    lv = Float64(level)
    if !isfinite(lv) || !(0.0 < lv < 1.0)
        throw(ArgumentError("confidence level must lie strictly in (0, 1), got $level"))
    end
    return lv
end

"""
    _validate_quantiles(quantiles::AbstractVector{<:Real}) -> Vector{Float64}

Validate that `quantiles` is non-empty and every entry is a finite, non-boolean
real number strictly in `(0, 1)`.
"""
function _validate_quantiles(qs::AbstractVector{<:Real})
    isempty(qs) && throw(ArgumentError("quantiles must not be empty"))
    out = Vector{Float64}(undef, length(qs))
    for (i, q) in enumerate(qs)
        if q isa Bool
            throw(ArgumentError("quantile must be a real number in (0, 1), got Bool ($q)"))
        end
        qf = Float64(q)
        if !isfinite(qf) || !(0.0 < qf < 1.0)
            throw(ArgumentError("every quantile must lie strictly in (0, 1), got $q"))
        end
        out[i] = qf
    end
    return out
end

"""
    _conf_level_to_quantiles(level::Real) -> Vector{Float64}

Map a symmetric confidence/credible interval `level ∈ (0, 1)` to `[q_lo, 0.5, q_hi]`,
rounding cleanly to avoid floating-point noise (e.g., `0.68 -> [0.16, 0.5, 0.84]`,
`0.90 -> [0.05, 0.5, 0.95]`).
"""
function _conf_level_to_quantiles(level::Real)
    lv = _validate_conf_level(level)
    tail = (1.0 - lv) / 2.0
    lo = round(tail; digits=12)
    hi = round(1.0 - tail; digits=12)
    if !(0.0 < lo < 0.5)
        lo = tail
    end
    if !(0.5 < hi < 1.0)
        hi = 1.0 - tail
    end
    return Float64[lo, 0.5, hi]
end

"""
    _conf_z(level::Real) -> Float64
    _conf_z(::Type{T}, level::Real) -> T

Two-sided standard normal critical value `Φ⁻¹(1 - (1 - level)/2)` for validated
`level ∈ (0, 1)`.
"""
_conf_z(level::Real) = quantile(Normal(), 1.0 - (1.0 - _validate_conf_level(level)) / 2.0)
_conf_z(::Type{T}, level::Real) where {T<:AbstractFloat} = T(_conf_z(level))

"""
    _ci_pct(level::Real) -> String

Format a confidence level in `(0, 1)` as a compact percentage string without trailing
zeros (e.g. `0.68 -> "68"`, `0.90 -> "90"`, `0.975 -> "97.5"`).
"""
function _ci_pct(level::Real)
    lv = _validate_conf_level(level)
    pct = 100.0 * lv
    if isapprox(pct, round(pct); atol=1e-9)
        return string(round(Int, pct))
    end
    s = string(round(pct; digits=2))
    s = replace(s, r"0+$" => "")
    s = replace(s, r"\.$" => "")
    return s
end

"""
    _quantile_pct_label(q::Real) -> String

Format a quantile probability in `(0, 1)` as a compact percentage header (e.g.
`0.16 -> "16%"`, `0.5 -> "50%"`, `0.025 -> "2.5%"`).
"""
function _quantile_pct_label(q::Real)
    pct = 100.0 * Float64(q)
    if isapprox(pct, round(pct); atol=1e-9)
        return "$(round(Int, pct))%"
    end
    s = string(round(pct; digits=2))
    s = replace(s, r"0+$" => "")
    s = replace(s, r"\.$" => "")
    return "$(s)%"
end

"""
    default_ci_level(m) -> Float64

Return the default confidence/credible interval coverage level for model or result `m`:
- [`DEFAULT_CI_MACRO`](@ref) (`0.68`) for macroeconometric, time-series, dynamic,
  structural, Bayesian, and LP-DiD models.
- [`DEFAULT_CI_MICRO`](@ref) (`0.90`) for microeconometric, cross-sectional,
  discrete-choice, count, quantile, micro-panel, RDD, GMM/SMM, and non-LP DiD models.

For results that carry a user-chosen estimation-time confidence level (e.g.
[`RDDResult`](@ref), [`DIDResult`](@ref), [`EventStudyLP`](@ref), [`LPDiDResult`](@ref),
[`MarginalEffects`](@ref), [`OddsRatio`](@ref), [`QuantileRegModel`](@ref),
[`ThresholdModel`](@ref)), `default_ci_level(m)` returns that stored level.
"""
default_ci_level(m) = default_ci_level(typeof(m))
default_ci_level(::Type) = DEFAULT_CI_MACRO
