# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-39 (#447): Point forecast-accuracy metrics.
#
# References: Theil (1966) U-statistics & MSE decomposition; Hyndman & Koehler
# (2006, IJF) scale-free MASE.

# Fixed metric column order for ForecastEvaluation.
const _FCEVAL_METRICS = ["ME", "MAE", "RMSE", "MAPE", "sMAPE", "MASE", "U1", "U2"]

"""
    _point_metrics(actual, fc; seasonal_period=1, insample=nothing) -> (vals, decomp)

Compute the point-accuracy metrics for one forecast column. Forecast errors use
the standard convention `e_t = actual_t − fc_t`. Returns the metric vector (in
`_FCEVAL_METRICS` order) and the 3-vector Theil MSE decomposition proportions.
"""
function _point_metrics(actual::AbstractVector{T}, fc::AbstractVector{T};
                        seasonal_period::Int=1,
                        insample::Union{Nothing,AbstractVector{T}}=nothing) where {T<:AbstractFloat}
    n = length(actual)
    n == length(fc) || throw(DimensionMismatch("actual and fc must have equal length"))
    n >= 2 || throw(ArgumentError("need at least 2 observations"))

    e = actual .- fc                       # forecast error
    me  = mean(e)
    mae = mean(abs, e)
    mse = mean(abs2, e)
    rmse = sqrt(mse)

    # MAPE / sMAPE — guard against (near-)zero actuals (spec: "guard against zero").
    tol = sqrt(eps(T))
    mape_terms = T[]
    smape_terms = T[]
    for t in 1:n
        if abs(actual[t]) > tol
            push!(mape_terms, abs(e[t] / actual[t]))
        end
        denom = abs(actual[t]) + abs(fc[t])
        if denom > tol
            push!(smape_terms, 2 * abs(e[t]) / denom)
        end
    end
    mape  = isempty(mape_terms)  ? T(NaN) : 100 * mean(mape_terms)
    smape = isempty(smape_terms) ? T(NaN) : 100 * mean(smape_terms)

    # MASE (Hyndman–Koehler 2006): scale by in-sample MAE of the (seasonal) naive
    # forecast. Falls back to the evaluation actuals when no in-sample series given.
    base = insample === nothing ? actual : insample
    m = seasonal_period
    scale = if length(base) > m
        s = zero(T); c = 0
        @inbounds for t in (m+1):length(base)
            s += abs(base[t] - base[t-m]); c += 1
        end
        c > 0 ? s / c : T(NaN)
    else
        T(NaN)
    end
    mase = (isfinite(scale) && scale > tol) ? mae / scale : T(NaN)

    # Theil U1 (1966): bounded in [0,1]; 0 = perfect.
    denom_u1 = sqrt(mean(abs2, actual)) + sqrt(mean(abs2, fc))
    u1 = denom_u1 > tol ? rmse / denom_u1 : T(NaN)

    # Theil U2 (1966): forecast vs one-step naive (no-change) benchmark. Equals 1
    # exactly for the naive forecast fc_t = actual_{t-1}.
    num = zero(T); den = zero(T)
    @inbounds for t in 2:n
        a_prev = actual[t-1]
        if abs(a_prev) > tol
            num += ((fc[t]     - actual[t]) / a_prev)^2
            den += ((actual[t] - a_prev)    / a_prev)^2
        end
    end
    u2 = den > tol ? sqrt(num / den) : T(NaN)

    # Theil MSE decomposition (proportions sum to 1). Population moments (÷n).
    mf = mean(fc); ma = mean(actual)
    sf = sqrt(mean(abs2, fc .- mf))
    sa = sqrt(mean(abs2, actual .- ma))
    rho = (sf > tol && sa > tol) ? (mean((fc .- mf) .* (actual .- ma)) / (sf * sa)) : zero(T)
    if mse > tol
        bias_prop = (mf - ma)^2 / mse
        var_prop  = (sf - sa)^2 / mse
        cov_prop  = 2 * (1 - rho) * sf * sa / mse
    else
        bias_prop = zero(T); var_prop = zero(T); cov_prop = zero(T)
    end

    vals = T[me, mae, rmse, mape, smape, mase, u1, u2]
    decomp = T[bias_prop, var_prop, cov_prop]
    return vals, decomp
end

"""
    forecast_evaluate(actual, fc; seasonal_period=1, insample=nothing,
                      model_names=nothing) -> ForecastEvaluation

Compute point forecast-accuracy metrics of one or several forecasts against a
common vector of realized values `actual`. Reports ME, MAE, RMSE, MAPE, sMAPE,
MASE, and Theil's `U1`/`U2`, plus the Theil MSE bias/variance/covariance
decomposition (whose three proportions sum to 1).

# Arguments
- `actual::AbstractVector` — realized values (length `T`)
- `fc` — either an `AbstractVector` (single model) or a `T×M` `AbstractMatrix`
  whose columns are competing forecasts
- `seasonal_period::Int=1` — seasonal lag for the MASE naive-forecast scaling
- `insample` — optional in-sample series for MASE scaling (Hyndman–Koehler); by
  default the evaluation `actual` is used
- `model_names` — optional column labels

Forecast errors follow the convention `e_t = actual_t − fc_t`. MAPE/sMAPE skip
observations with (near-)zero denominators.

# Example
```julia
ev = forecast_evaluate(y, hcat(f1, f2); model_names=["AR", "RW"])
report(ev)
```
"""
function forecast_evaluate(actual::AbstractVector{<:Real}, fc::AbstractVector{<:Real};
                           seasonal_period::Int=1,
                           insample::Union{Nothing,AbstractVector}=nothing,
                           model_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing)
    T = float(promote_type(eltype(actual), eltype(fc)))
    a = collect(T, actual)
    f = collect(T, fc)
    ins = insample === nothing ? nothing : collect(T, insample)
    vals, decomp = _point_metrics(a, f; seasonal_period=seasonal_period, insample=ins)
    names = model_names === nothing ? ["Model 1"] : String.(model_names)
    ForecastEvaluation{T}(names, copy(_FCEVAL_METRICS),
                          reshape(vals, 1, :), reshape(decomp, 1, :), length(a))
end

function forecast_evaluate(actual::AbstractVector{<:Real}, fc::AbstractMatrix{<:Real};
                           seasonal_period::Int=1,
                           insample::Union{Nothing,AbstractVector}=nothing,
                           model_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing)
    T = float(promote_type(eltype(actual), eltype(fc)))
    a = collect(T, actual)
    M = size(fc, 2)
    ins = insample === nothing ? nothing : collect(T, insample)
    K = length(_FCEVAL_METRICS)
    vals = Matrix{T}(undef, M, K)
    decomp = Matrix{T}(undef, M, 3)
    for j in 1:M
        v, d = _point_metrics(a, collect(T, @view fc[:, j]);
                              seasonal_period=seasonal_period, insample=ins)
        vals[j, :] = v
        decomp[j, :] = d
    end
    names = model_names === nothing ? ["Model $j" for j in 1:M] : String.(model_names)
    length(names) == M || throw(ArgumentError("model_names must have $M entries"))
    ForecastEvaluation{T}(names, copy(_FCEVAL_METRICS), vals, decomp, length(a))
end

# --- Display -----------------------------------------------------------------

function Base.show(io::IO, ev::ForecastEvaluation{T}) where {T}
    M = length(ev.models)
    K = length(ev.metrics)
    data = Matrix{Any}(undef, M, K + 1)
    for j in 1:M
        data[j, 1] = ev.models[j]
        for k in 1:K
            data[j, k+1] = _fmt(ev.values[j, k])
        end
    end
    _pretty_table(io, data;
        title = "Forecast Evaluation (n=$(ev.n))",
        column_labels = vcat([""], ev.metrics),
        alignment = vcat([:l], fill(:r, K)))

    # MSE decomposition block (proportions sum to 1).
    dec = Matrix{Any}(undef, M, 4)
    for j in 1:M
        dec[j, 1] = ev.models[j]
        dec[j, 2] = _fmt(ev.decomp[j, 1])
        dec[j, 3] = _fmt(ev.decomp[j, 2])
        dec[j, 4] = _fmt(ev.decomp[j, 3])
    end
    _pretty_table(io, dec;
        title = "Theil MSE decomposition (proportions)",
        column_labels = ["", "Bias", "Variance", "Covariance"],
        alignment = [:l, :r, :r, :r])
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", ev::ForecastEvaluation) = show(io, ev)
report(ev::ForecastEvaluation) = show(stdout, ev)
report(io::IO, ev::ForecastEvaluation) = show(io, ev)
