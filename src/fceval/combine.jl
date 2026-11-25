# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-only. See LICENSE for details.
#
# EV-39 (#447): Forecast combination.
#
# References: Bates & Granger (1969, ORQ) inverse-MSE weights;
# Granger & Ramanathan (1984, J. Forecasting) constrained least squares.

"""
    combine_forecasts(F, actual; method=:equal, model_names=nothing) -> ForecastCombination

Combine the columns of a `T×M` forecast matrix `F` into a single series.

# Methods
- `:equal` — simple average, `wᵢ = 1/M` (Bates–Granger 1969 baseline; robust,
  no estimation).
- `:bates_granger` — inverse-MSE weights `wᵢ ∝ 1/MSEᵢ`, normalized to sum to one
  (ignores cross-forecast error correlation).
- `:granger_ramanathan` — constrained least squares minimizing
  `‖actual − F·w‖²` subject to `1'w = 1`, solved in closed form by the KKT
  system. Weights may be **negative** (this is intended; only the sum-to-one
  constraint is imposed and no clamping is applied).

Returns a [`ForecastCombination`](@ref) carrying the weights, the combined
series `F·w`, and the individual-model MSEs.

# Example
```julia
comb = combine_forecasts(hcat(f1, f2, f3), y; method=:bates_granger)
comb.weights
```
"""
function combine_forecasts(F::AbstractMatrix{<:Real}, actual::AbstractVector{<:Real};
                           method::Symbol=:equal,
                           model_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing)
    T = float(promote_type(eltype(F), eltype(actual)))
    n, M = size(F)
    n == length(actual) || throw(DimensionMismatch("F rows must match length(actual)"))
    M >= 1 || throw(ArgumentError("need at least one forecast"))
    Fm = Matrix{T}(F)
    a = collect(T, actual)

    mse = T[mean(abs2, a .- @view Fm[:, j]) for j in 1:M]

    w = if method === :equal
        fill(one(T) / M, M)
    elseif method === :bates_granger
        any(mse .<= 0) && throw(ArgumentError(":bates_granger requires strictly positive MSEs"))
        inv_mse = one(T) ./ mse
        inv_mse ./ sum(inv_mse)
    elseif method === :granger_ramanathan
        # min ‖a − F w‖²  s.t. 1'w = 1. KKT:  [F'F  1; 1'  0][w; λ] = [F'a; 1].
        # Closed form: w = Σ⁻¹c + Σ⁻¹1 (1 − 1'Σ⁻¹c)/(1'Σ⁻¹1), Σ=F'F, c=F'a.
        Sigma = Fm' * Fm
        c = Fm' * a
        Sinv = robust_inv(Matrix{T}(Sigma))
        ones_M = ones(T, M)
        Sinv_c = Sinv * c
        Sinv_1 = Sinv * ones_M
        denom = dot(ones_M, Sinv_1)
        Sinv_c .+ Sinv_1 .* ((one(T) - dot(ones_M, Sinv_c)) / denom)
    else
        throw(ArgumentError("method must be :equal, :bates_granger, or :granger_ramanathan; got :$method"))
    end

    combined = Fm * w
    names = model_names === nothing ? ["Model $j" for j in 1:M] : String.(model_names)
    length(names) == M || throw(ArgumentError("model_names must have $M entries"))
    ForecastCombination{T}(w, combined, method, mse, names)
end

# --- Display -----------------------------------------------------------------

function Base.show(io::IO, c::ForecastCombination{T}) where {T}
    M = length(c.weights)
    data = Matrix{Any}(undef, M, 3)
    for j in 1:M
        data[j, 1] = c.models[j]
        data[j, 2] = _fmt(c.weights[j])
        data[j, 3] = _fmt(c.mse[j])
    end
    _pretty_table(io, data;
        title = "Forecast Combination (method = :$(c.method))",
        column_labels = ["", "Weight", "MSE"],
        alignment = [:l, :r, :r])
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", c::ForecastCombination) = show(io, c)
report(c::ForecastCombination) = show(stdout, c)
report(io::IO, c::ForecastCombination) = show(io, c)
