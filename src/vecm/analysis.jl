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
IRF, FEVD, and Historical Decomposition dispatch for VECM via VAR conversion.

All structural analysis methods work automatically through `to_var()`.
"""

"""
    irf(vecm::VECMModel, horizon; kwargs...) -> ImpulseResponse

Compute IRFs for a VECM by converting to VAR representation.
All identification methods (Cholesky, sign, narrative, etc.) are supported.
"""
function irf(vecm::VECMModel{T}, horizon::Int; kwargs...) where {T}
    irf(to_var(vecm), horizon; kwargs...)
end

"""
    fevd(vecm::VECMModel, horizon; kwargs...) -> FEVD

Compute FEVD for a VECM by converting to VAR representation.
"""
function fevd(vecm::VECMModel{T}, horizon::Int; kwargs...) where {T}
    fevd(to_var(vecm), horizon; kwargs...)
end

"""
    historical_decomposition(vecm::VECMModel, horizon; kwargs...) -> HistoricalDecomposition

Compute historical decomposition for a VECM by converting to VAR representation.
"""
function historical_decomposition(vecm::VECMModel{T}, horizon::Int=effective_nobs(vecm); kwargs...) where {T}
    historical_decomposition(to_var(vecm), horizon; kwargs...)
end
