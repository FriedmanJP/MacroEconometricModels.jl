# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Direct MIDAS forecasting: apply the fitted weights and coefficients to a fresh
high-frequency block and return a point forecast with a Gaussian NLS prediction
interval.
"""

using LinearAlgebra, Distributions

"""
    forecast(m::MidasModel, X_new; y_lags=nothing, level=0.95) -> MidasForecast

Direct `h`-step forecast (horizon stored in `m.h`) using the most recent `K`
high-frequency observations in `X_new` (most-recent-first). Autoregressive terms
use `y_lags` (most-recent-first, length `m.p_ar`); if omitted, the most recent
target values retained in-sample are used.

The prediction standard error combines the residual variance with parameter
uncertainty through the Gauss–Newton gradient `x_f`:

    Var(ŷ_f) = σ̂² + x_fᵀ V x_f,

where `V` is the coefficient variance-covariance matrix.
"""
function forecast(m::MidasModel{T}, X_new::AbstractVector;
                  y_lags::Union{Nothing,AbstractVector}=nothing,
                  level::Real=0.95) where {T<:AbstractFloat}
    xn = convert(Vector{T}, collect(X_new))
    length(xn) >= m.K || throw(ArgumentError(
        "X_new needs ≥ K=$(m.K) high-frequency observations (got $(length(xn)))"))
    xblock = xn[1:m.K]                     # most-recent-first

    # autoregressive lags
    if m.p_ar > 0
        if y_lags === nothing
            length(m.y) >= m.p_ar || throw(ArgumentError("cannot infer AR lags: in-sample too short"))
            ylag = reverse(m.y[end - m.p_ar + 1:end])   # most-recent-first
        else
            yl = convert(Vector{T}, collect(y_lags))
            length(yl) >= m.p_ar || throw(ArgumentError("y_lags needs ≥ p_ar=$(m.p_ar) values"))
            ylag = yl[1:m.p_ar]
        end
    else
        ylag = T[]
    end

    p_lin = length(m.beta)
    if m.weights_kind === :umidas
        # design row: [const, x_lag1…x_lagK, AR…]
        xf = vcat(one(T), xblock, ylag)
        point = dot(xf, m.beta)
        xf_full = xf
    else
        w = m.w
        s = dot(xblock, w)
        # linear design row [const, s, AR…]
        xf_lin = vcat(one(T), s, ylag)
        point = dot(xf_lin, m.beta)
        # θ-gradient of the prediction: ∂ŷ/∂θ_l = β₁ · (xblockᵀ ∂w/∂θ_l)
        Jw = _midas_weights_jac(m.theta, m.K, m.weights_kind)   # K×p
        gtheta = m.beta[2] .* (Jw' * xblock)                    # p-vector
        xf_full = vcat(xf_lin, gtheta)
    end

    var_param = dot(xf_full, m.vcov_mat * xf_full)
    se = sqrt(max(m.sigma2 + var_param, zero(T)))
    z = T(quantile(Normal(), 1 - (1 - level) / 2))
    lo = point - z * se
    hi = point + z * se
    return MidasForecast{T}([point], [lo], [hi], [se], m.h, T(level))
end

# =============================================================================
# Display
# =============================================================================

function Base.show(io::IO, f::MidasForecast{T}) where {T}
    hz = length(f.forecast)
    data = Matrix{Any}(undef, hz, 5)
    for i in 1:hz
        data[i, 1] = i
        data[i, 2] = _fmt(f.forecast[i])
        data[i, 3] = _fmt(f.se[i])
        data[i, 4] = _fmt(f.ci_lower[i])
        data[i, 5] = _fmt(f.ci_upper[i])
    end
    pct = round(Int, 100 * f.conf_level)
    _pretty_table(io, data;
        title = "MIDAS Forecast (direct h=$(f.horizon))",
        column_labels = ["Step", "Forecast", "Std.Err.", "$(pct)% lower", "$(pct)% upper"],
        alignment = [:r, :r, :r, :r, :r],
    )
end

report(f::MidasForecast) = show(stdout, f)
