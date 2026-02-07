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
function historical_decomposition(vecm::VECMModel{T}, horizon::Int; kwargs...) where {T}
    historical_decomposition(to_var(vecm), horizon; kwargs...)
end
