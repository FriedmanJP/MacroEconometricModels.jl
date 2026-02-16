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
Data conversion utilities and estimation dispatch wrappers for MacroEconometricModels.jl.

Provides `to_matrix`, `to_vector`, and thin dispatch methods so all estimation
functions accept `TimeSeriesData` in addition to raw `Matrix`/`Vector`.
"""

# =============================================================================
# Conversion functions
# =============================================================================

"""
    to_matrix(d::TimeSeriesData) -> Matrix

Return the raw data matrix from a TimeSeriesData container.
"""
to_matrix(d::TimeSeriesData) = d.data

"""
    to_matrix(d::PanelData) -> Matrix

Return the raw stacked data matrix from a PanelData container.
"""
to_matrix(d::PanelData) = d.data

"""
    to_matrix(d::CrossSectionData) -> Matrix

Return the raw data matrix from a CrossSectionData container.
"""
to_matrix(d::CrossSectionData) = d.data

"""
    to_vector(d::TimeSeriesData) -> Vector

Return the data as a vector (requires exactly 1 variable).
"""
function to_vector(d::TimeSeriesData)
    d.n_vars == 1 || throw(ArgumentError(
        "to_vector requires exactly 1 variable, got $(d.n_vars). Use to_vector(d, var) to select a column."))
    d.data[:, 1]
end

"""
    to_vector(d::TimeSeriesData, var::Int) -> Vector

Return a single column by index.
"""
function to_vector(d::TimeSeriesData, var::Int)
    1 <= var <= d.n_vars || throw(BoundsError(d, var))
    d.data[:, var]
end

"""
    to_vector(d::TimeSeriesData, var::String) -> Vector

Return a single column by name.
"""
function to_vector(d::TimeSeriesData, var::String)
    idx = findfirst(==(var), d.varnames)
    idx === nothing && throw(ArgumentError("Variable '$var' not found. Available: $(d.varnames)"))
    d.data[:, idx]
end

# =============================================================================
# Multivariate dispatch wrappers (TimeSeriesData → Matrix)
# =============================================================================

# VAR
estimate_var(d::TimeSeriesData, p::Int; kwargs...) =
    estimate_var(to_matrix(d), p; varnames=d.varnames, kwargs...)

# VECM
estimate_vecm(d::TimeSeriesData, p::Int; kwargs...) =
    estimate_vecm(to_matrix(d), p; kwargs...)

# BVAR
estimate_bvar(d::TimeSeriesData, p::Int; kwargs...) =
    estimate_bvar(to_matrix(d), p; kwargs...)

# Factor models
estimate_factors(d::TimeSeriesData, r::Int; kwargs...) =
    estimate_factors(to_matrix(d), r; kwargs...)

estimate_dynamic_factors(d::TimeSeriesData, r::Int, p::Int; kwargs...) =
    estimate_dynamic_factors(to_matrix(d), r, p; kwargs...)

estimate_gdfm(d::TimeSeriesData, q::Int; kwargs...) =
    estimate_gdfm(to_matrix(d), q; kwargs...)

# LP
estimate_lp(d::TimeSeriesData, shock_var::Int, horizon::Int; kwargs...) =
    estimate_lp(to_matrix(d), shock_var, horizon; kwargs...)

estimate_lp_iv(d::TimeSeriesData, shock_var::Int, instruments::AbstractMatrix, horizon::Int; kwargs...) =
    estimate_lp_iv(to_matrix(d), shock_var, instruments, horizon; kwargs...)

estimate_smooth_lp(d::TimeSeriesData, shock_var::Int, horizon::Int; kwargs...) =
    estimate_smooth_lp(to_matrix(d), shock_var, horizon; kwargs...)

estimate_state_lp(d::TimeSeriesData, shock_var::Int, state_var::AbstractVector, horizon::Int; kwargs...) =
    estimate_state_lp(to_matrix(d), shock_var, state_var, horizon; kwargs...)

structural_lp(d::TimeSeriesData, horizon::Int; kwargs...) =
    structural_lp(to_matrix(d), horizon; kwargs...)

# Johansen test
johansen_test(d::TimeSeriesData, p::Int; kwargs...) =
    johansen_test(to_matrix(d), p; kwargs...)

# =============================================================================
# Univariate dispatch wrappers (TimeSeriesData → Vector)
# =============================================================================

# ARIMA family
estimate_ar(d::TimeSeriesData, p::Int; kwargs...) =
    estimate_ar(to_vector(d), p; kwargs...)

estimate_ma(d::TimeSeriesData, q::Int; kwargs...) =
    estimate_ma(to_vector(d), q; kwargs...)

estimate_arma(d::TimeSeriesData, p::Int, q::Int; kwargs...) =
    estimate_arma(to_vector(d), p, q; kwargs...)

estimate_arima(d::TimeSeriesData, p::Int, dd::Int, q::Int; kwargs...) =
    estimate_arima(to_vector(d), p, dd, q; kwargs...)

# Volatility models
estimate_arch(d::TimeSeriesData, q::Int; kwargs...) =
    estimate_arch(to_vector(d), q; kwargs...)

estimate_garch(d::TimeSeriesData, p::Int=1, q::Int=1; kwargs...) =
    estimate_garch(to_vector(d), p, q; kwargs...)

estimate_egarch(d::TimeSeriesData, p::Int=1, q::Int=1; kwargs...) =
    estimate_egarch(to_vector(d), p, q; kwargs...)

estimate_gjr_garch(d::TimeSeriesData, p::Int=1, q::Int=1; kwargs...) =
    estimate_gjr_garch(to_vector(d), p, q; kwargs...)

estimate_sv(d::TimeSeriesData; kwargs...) =
    estimate_sv(to_vector(d); kwargs...)

# Filters
hp_filter(d::TimeSeriesData; kwargs...) =
    hp_filter(to_vector(d); kwargs...)

hamilton_filter(d::TimeSeriesData; kwargs...) =
    hamilton_filter(to_vector(d); kwargs...)

beveridge_nelson(d::TimeSeriesData; kwargs...) =
    beveridge_nelson(to_vector(d); kwargs...)

baxter_king(d::TimeSeriesData; kwargs...) =
    baxter_king(to_vector(d); kwargs...)

boosted_hp(d::TimeSeriesData; kwargs...) =
    boosted_hp(to_vector(d); kwargs...)

# Unit root tests
adf_test(d::TimeSeriesData; kwargs...) =
    adf_test(to_vector(d); kwargs...)

kpss_test(d::TimeSeriesData; kwargs...) =
    kpss_test(to_vector(d); kwargs...)

pp_test(d::TimeSeriesData; kwargs...) =
    pp_test(to_vector(d); kwargs...)

za_test(d::TimeSeriesData; kwargs...) =
    za_test(to_vector(d); kwargs...)

ngperron_test(d::TimeSeriesData; kwargs...) =
    ngperron_test(to_vector(d); kwargs...)

# =============================================================================
# Nowcast dispatch wrappers (TimeSeriesData → Matrix)
# =============================================================================

nowcast_dfm(d::TimeSeriesData, nM::Int, nQ::Int; kwargs...) =
    nowcast_dfm(to_matrix(d), nM, nQ; kwargs...)

nowcast_bvar(d::TimeSeriesData, nM::Int, nQ::Int; kwargs...) =
    nowcast_bvar(to_matrix(d), nM, nQ; kwargs...)

nowcast_bridge(d::TimeSeriesData, nM::Int, nQ::Int; kwargs...) =
    nowcast_bridge(to_matrix(d), nM, nQ; kwargs...)
