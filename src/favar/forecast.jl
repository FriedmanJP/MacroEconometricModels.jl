# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Forecasting for FAVAR models via VAR conversion.
"""

# =============================================================================
# FAVAR Forecast (delegates to VARModel)
# =============================================================================

"""
    forecast(favar::FAVARModel, h; kwargs...) -> VARForecast

Forecast the augmented FAVAR system [F, Y_key] h steps ahead.

This delegates to `forecast(to_var(favar), h; kwargs...)`.
Use `favar_panel_forecast` to map factor forecasts to all N panel variables.

# Arguments
- `favar`: Estimated FAVAR model
- `h`: Forecast horizon

# Keyword Arguments
Passed through to `forecast(::VARModel, h; ...)`:
- `ci_method::Symbol=:bootstrap`: CI method (`:none` or `:bootstrap`)
- `reps::Int=500`: Number of bootstrap replications
- `conf_level::Real=0.95`: Confidence level

# Example
```julia
favar = estimate_favar(X, [1, 5], 3, 2)
fc = forecast(favar, 12)                          # forecasts for [F1,F2,F3,Y1,Y5]
panel_fc = favar_panel_forecast(favar, fc)          # forecasts for all 50 panel vars
```
"""
function forecast(favar::FAVARModel{T}, h::Int; kwargs...) where {T}
    forecast(to_var(favar), h; kwargs...)
end
