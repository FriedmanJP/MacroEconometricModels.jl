# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
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
Automatic order selection for ARMA/ARIMA models via information criteria.
"""

using LinearAlgebra

# =============================================================================
# Order Selection
# =============================================================================

"""
    select_arima_order(y, max_p, max_q; criterion=:bic, d=0, method=:css_mle, include_intercept=true)

Automatically select ARMA/ARIMA order via grid search over information criteria.

Searches over p ∈ 0:max_p and q ∈ 0:max_q, fits each model, and selects the
order that minimizes the specified information criterion.

# Arguments
- `y`: Time series vector
- `max_p`: Maximum AR order to consider
- `max_q`: Maximum MA order to consider
- `criterion`: Selection criterion (:aic or :bic, default :bic)
- `d`: Integration order for ARIMA (default 0 = ARMA)
- `method`: Estimation method (:css, :mle, or :css_mle)
- `include_intercept`: Whether to include constant term

# Returns
`ARIMAOrderSelection` with best orders, IC matrices, and fitted models.

# Example
```julia
y = randn(200)
result = select_arima_order(y, 3, 3; criterion=:bic)
println("Best order: p=\$(result.best_p_bic), q=\$(result.best_q_bic)")
best_model = result.best_model_bic
```
"""
function select_arima_order(y::AbstractVector{T}, max_p::Int, max_q::Int;
                            criterion::Symbol=:bic, d::Int=0, method::Symbol=:css_mle,
                            include_intercept::Bool=true) where {T<:AbstractFloat}
    max_p < 0 && throw(ArgumentError("max_p must be non-negative"))
    max_q < 0 && throw(ArgumentError("max_q must be non-negative"))
    d < 0 && throw(ArgumentError("d must be non-negative"))
    criterion ∉ (:aic, :bic) && throw(ArgumentError("criterion must be :aic or :bic"))

    y_vec = Vector{T}(y)
    n = length(y_vec)

    # Difference if needed
    y_work = d > 0 ? _difference(y_vec, d) : y_vec

    # Initialize IC matrices with Inf
    aic_matrix = fill(T(Inf), max_p + 1, max_q + 1)
    bic_matrix = fill(T(Inf), max_p + 1, max_q + 1)

    # Store fitted models
    models = Dict{Tuple{Int,Int}, AbstractARIMAModel}()

    # Grid search
    for p in 0:max_p
        for q in 0:max_q
            try
                # Fit model
                model = if d == 0
                    estimate_arma(y_work, p, q; method=method, include_intercept=include_intercept)
                else
                    estimate_arima(y_vec, p, d, q; method=method, include_intercept=include_intercept)
                end

                # Store IC values (matrix is 1-indexed, p and q are 0-indexed)
                aic_matrix[p + 1, q + 1] = model.aic
                bic_matrix[p + 1, q + 1] = model.bic
                models[(p, q)] = model

            catch e
                # Model failed to fit - leave as Inf
                @debug "Failed to fit ARMA($p,$q): $e"
                continue
            end
        end
    end

    # Find best orders
    aic_idx = argmin(aic_matrix)
    bic_idx = argmin(bic_matrix)

    best_p_aic = aic_idx[1] - 1  # Convert back to 0-indexed
    best_q_aic = aic_idx[2] - 1
    best_p_bic = bic_idx[1] - 1
    best_q_bic = bic_idx[2] - 1

    # Get best models
    best_model_aic = models[(best_p_aic, best_q_aic)]
    best_model_bic = models[(best_p_bic, best_q_bic)]

    ARIMAOrderSelection(best_p_aic, best_q_aic, best_p_bic, best_q_bic,
                        aic_matrix, bic_matrix, best_model_aic, best_model_bic)
end

select_arima_order(y::AbstractVector, max_p::Int, max_q::Int; kwargs...) =
    select_arima_order(Float64.(y), max_p, max_q; kwargs...)

# =============================================================================
# Convenience Functions
# =============================================================================

"""
    auto_arima(y; max_p=5, max_q=5, max_d=2, criterion=:bic, method=:css_mle)

Automatically select and fit the best ARIMA model.

Performs order selection over p, d, and q using the specified criterion.
For integration order d, uses unit root test heuristics.

# Arguments
- `y`: Time series vector
- `max_p`: Maximum AR order (default 5)
- `max_q`: Maximum MA order (default 5)
- `max_d`: Maximum integration order (default 2)
- `criterion`: Selection criterion (:aic or :bic)
- `method`: Estimation method

# Returns
Best fitted ARIMAModel or ARMAModel.

# Example
```julia
y = cumsum(randn(200))
model = auto_arima(y)
println(model)
```
"""
function auto_arima(y::AbstractVector{T}; max_p::Int=5, max_q::Int=5, max_d::Int=2,
                    criterion::Symbol=:bic, method::Symbol=:css_mle,
                    include_intercept::Bool=true) where {T<:AbstractFloat}
    y_vec = Vector{T}(y)

    # Simple heuristic for d: difference until series appears stationary
    # (based on variance reduction)
    d_best = _select_d_heuristic(y_vec, max_d)

    # Grid search for p and q
    result = select_arima_order(y_vec, max_p, max_q; criterion=criterion, d=d_best,
                                method=method, include_intercept=include_intercept)

    criterion == :aic ? result.best_model_aic : result.best_model_bic
end

auto_arima(y::AbstractVector; kwargs...) = auto_arima(Float64.(y); kwargs...)

"""
    _select_d_heuristic(y, max_d) -> Int

Select integration order d using simple variance heuristic.

Differences until variance stops decreasing significantly.
"""
function _select_d_heuristic(y::Vector{T}, max_d::Int) where {T<:AbstractFloat}
    max_d == 0 && return 0

    y_curr = copy(y)
    var_prev = var(y_curr)

    for d in 1:max_d
        y_curr = diff(y_curr)
        var_curr = var(y_curr)

        # If variance increased or decreased by less than 10%, stop
        if var_curr >= var_prev || var_curr > T(0.9) * var_prev
            return d - 1
        end
        var_prev = var_curr
    end

    max_d
end

# =============================================================================
# IC Table Display
# =============================================================================

"""
    ic_table(result::ARIMAOrderSelection; criterion=:bic)

Return a formatted table of IC values for printing.
"""
function ic_table(result::ARIMAOrderSelection{T}; criterion::Symbol=:bic) where {T}
    matrix = criterion == :aic ? result.aic_matrix : result.bic_matrix
    max_p, max_q = size(matrix) .- 1

    # Build table as string
    lines = String[]
    push!(lines, "$(uppercase(string(criterion))) values for ARMA(p,q):")
    push!(lines, "")

    # Helper for formatting
    _fmt(x::Real) = isinf(x) ? "     Inf" : lpad(string(round(x, digits=2)), 8)
    _fmtint(x::Int) = lpad(string(x), 3)

    # Header
    header = "p\\q |" * join([lpad(string(q), 8) for q in 0:max_q], "")
    push!(lines, header)
    push!(lines, "-"^length(header))

    # Rows
    for p in 0:max_p
        row_vals = [_fmt(matrix[p+1, q+1]) for q in 0:max_q]
        push!(lines, _fmtint(p) * " |" * join(row_vals, ""))
    end

    join(lines, "\n")
end

