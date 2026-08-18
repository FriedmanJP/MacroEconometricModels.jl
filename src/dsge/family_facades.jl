# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Family `irf` / `fevd` / `simulate` façades (#648).

Included after OLG / CT / DCEGM types so the methods can name those payloads.
Returns the same `ImpulseResponse` / `FEVD` types the RA path uses.
"""

function _fevd_from_irf(ir::ImpulseResponse{T}) where {T<:AbstractFloat}
    n_vars = length(ir.variables)
    n_e = length(ir.shocks)
    H = ir.horizon
    decomp = zeros(T, n_vars, n_e, H)
    props  = zeros(T, n_vars, n_e, H)
    @inbounds for h in 1:H
        for i in 1:n_vars
            total = zero(T)
            for j in 1:n_e
                prev = h == 1 ? zero(T) : decomp[i, j, h-1]
                decomp[i, j, h] = prev + ir.values[h, i, j]^2
                total += decomp[i, j, h]
            end
            total > 0 && (props[i, :, h] = decomp[i, :, h] ./ total)
        end
    end
    return FEVD{T}(decomp, props, ir.variables, ir.shocks)
end

function _path_to_irf(values::AbstractMatrix{T}, var_names,
                      shock_name::AbstractString) where {T<:AbstractFloat}
    H, n = size(values)
    point = reshape(Matrix{T}(values), H, n, 1)
    return ImpulseResponse{T}(point, zeros(T, H, n, 1), zeros(T, H, n, 1),
                              H, collect(String, var_names), [String(shock_name)], :none)
end

function _ct_z_path(Zbar::T, horizon::Int, shock_size, persist) where {T<:AbstractFloat}
    horizon >= 2 || throw(ArgumentError("irf/simulate: horizon must be ≥ 2 (MIT path length)"))
    return [Zbar * (one(T) + T(shock_size) * T(persist)^(n - 1)) for n in 1:horizon]
end

"""
    irf(m::CTAiyagari, horizon; ss=nothing, shock_size=0.01, persist=0, dt=0.25, kwargs...)

Wrap [`ct_mit_shock`](@ref) as an [`ImpulseResponse`](@ref) in deviations from the
stationary equilibrium. `persist=0` is a one-period TFP impulse; `0 < persist < 1`
is an AR(1) decay. Extra keywords go to `ct_mit_shock`.
"""
function irf(m::CTAiyagari{T}, horizon::Int;
             ss::Union{Nothing,CTSteadyState{T}}=nothing,
             shock_size::Real=T(0.01), persist::Real=zero(T),
             dt::Real=0.25, kwargs...) where {T<:AbstractFloat}
    ss0 = ss === nothing ? ct_steady_state(m) : ss
    tr = ct_mit_shock(m, ss0, _ct_z_path(m.Z, horizon, shock_size, persist); dt=dt, kwargs...)
    da = ss0.a[2] - ss0.a[1]
    C_ss = sum(ss0.c .* ss0.g) * da
    vals = hcat(tr.K .- ss0.K, tr.r .- ss0.r, tr.w .- ss0.w, tr.C .- C_ss, tr.Z .- m.Z)
    return _path_to_irf(vals, ("K", "r", "w", "C", "Z"), "Z")
end

"""
    irf(m::CTTwoAsset, horizon; ge=nothing, shock_size=0.01, persist=0, dt=0.25, kwargs...)

Wrap [`ct_two_asset_mit`](@ref) as an [`ImpulseResponse`](@ref) in deviations
from the two-asset stationary GE.
"""
function irf(m::CTTwoAsset{T}, horizon::Int;
             ge::Union{Nothing,CTTwoAssetGE{T}}=nothing,
             shock_size::Real=T(0.01), persist::Real=zero(T),
             dt::Real=0.25, kwargs...) where {T<:AbstractFloat}
    ge0 = ge === nothing ? ct_two_asset_ge(m) : ge
    tr = ct_two_asset_mit(m, ge0, _ct_z_path(m.Z, horizon, shock_size, persist); dt=dt, kwargs...)
    vals = hcat(tr.K .- ge0.K, tr.r_a .- ge0.r_a, tr.r_b .- ge0.r_b,
                tr.w .- ge0.w, tr.B .- ge0.B, tr.Z .- m.Z)
    return _path_to_irf(vals, ("K", "r_a", "r_b", "w", "B", "Z"), "Z")
end

fevd(m::CTAiyagari, horizon::Int; kwargs...) = _fevd_from_irf(irf(m, horizon; kwargs...))
fevd(m::CTTwoAsset, horizon::Int; kwargs...) = _fevd_from_irf(irf(m, horizon; kwargs...))

function simulate(m::CTAiyagari{T}, T_periods::Int;
                  ss::Union{Nothing,CTSteadyState{T}}=nothing,
                  shock_size::Real=zero(T), persist::Real=zero(T),
                  dt::Real=0.25, kwargs...) where {T<:AbstractFloat}
    ss0 = ss === nothing ? ct_steady_state(m) : ss
    tr = ct_mit_shock(m, ss0, _ct_z_path(m.Z, T_periods, shock_size, persist); dt=dt, kwargs...)
    return hcat(tr.K, tr.r, tr.w, tr.C, tr.Z)
end

function simulate(m::CTTwoAsset{T}, T_periods::Int;
                  ge::Union{Nothing,CTTwoAssetGE{T}}=nothing,
                  shock_size::Real=zero(T), persist::Real=zero(T),
                  dt::Real=0.25, kwargs...) where {T<:AbstractFloat}
    ge0 = ge === nothing ? ct_two_asset_ge(m) : ge
    tr = ct_two_asset_mit(m, ge0, _ct_z_path(m.Z, T_periods, shock_size, persist); dt=dt, kwargs...)
    return hcat(tr.K, tr.r_a, tr.r_b, tr.w, tr.B, tr.Z)
end

function irf(::DCEGMSolution, ::Int; kwargs...)
    throw(ArgumentError(
        "irf(::DCEGMSolution) is not implemented yet (needs G-11 DCEGM aggregate dynamics, #645)"))
end
function fevd(::DCEGMSolution, ::Int; kwargs...)
    throw(ArgumentError(
        "fevd(::DCEGMSolution) is not implemented yet (needs G-11 DCEGM aggregate dynamics, #645)"))
end
function simulate(::DCEGMSolution, ::Int; kwargs...)
    throw(ArgumentError(
        "simulate(::DCEGMSolution) is not implemented yet (needs G-11 DCEGM aggregate dynamics, #645)"))
end

function irf(::LifeCycleSteadyState, ::Int; kwargs...)
    throw(ArgumentError(
        "irf(::LifeCycleSteadyState) is not implemented yet (needs G-12 life-cycle transition, #646)"))
end
function fevd(::LifeCycleSteadyState, ::Int; kwargs...)
    throw(ArgumentError(
        "fevd(::LifeCycleSteadyState) is not implemented yet (needs G-12 life-cycle transition, #646)"))
end
function simulate(::LifeCycleSteadyState, ::Int; kwargs...)
    throw(ArgumentError(
        "simulate(::LifeCycleSteadyState) is not implemented yet (needs G-12 life-cycle transition, #646)"))
end

"""
    irf(spec::ModelSpec, horizon; kwargs...)

Kind-dispatching façade. Blanchard / RA residuals go through `solve` then the
linear IRF. Continuous-time households wrap the MIT path. DCEGM and life-cycle
throw until G-11 / G-12.
"""
function irf(spec::ModelSpec, horizon::Int; kwargs...)
    if has_kind(spec, DCEGMSystem)
        throw(ArgumentError(
            "irf for DCEGMSystem is not implemented yet (needs G-11 DCEGM aggregate dynamics, #645)"))
    elseif has_kind(spec, LifeCycleSystem)
        throw(ArgumentError(
            "irf for LifeCycleSystem is not implemented yet (needs G-12 life-cycle transition, #646)"))
    elseif has_kind(spec, ContinuousHouseholdSystem)
        return irf(only(agents_of(spec, ContinuousHouseholdSystem)).model, horizon; kwargs...)
    else
        return irf(solve(spec), horizon; kwargs...)
    end
end

function fevd(spec::ModelSpec, horizon::Int; kwargs...)
    if has_kind(spec, DCEGMSystem)
        throw(ArgumentError(
            "fevd for DCEGMSystem is not implemented yet (needs G-11 DCEGM aggregate dynamics, #645)"))
    elseif has_kind(spec, LifeCycleSystem)
        throw(ArgumentError(
            "fevd for LifeCycleSystem is not implemented yet (needs G-12 life-cycle transition, #646)"))
    elseif has_kind(spec, ContinuousHouseholdSystem)
        return fevd(only(agents_of(spec, ContinuousHouseholdSystem)).model, horizon; kwargs...)
    else
        return fevd(solve(spec), horizon)
    end
end

function simulate(spec::ModelSpec, T_periods::Int; kwargs...)
    if has_kind(spec, DCEGMSystem)
        throw(ArgumentError(
            "simulate for DCEGMSystem is not implemented yet (needs G-11 DCEGM aggregate dynamics, #645)"))
    elseif has_kind(spec, LifeCycleSystem)
        throw(ArgumentError(
            "simulate for LifeCycleSystem is not implemented yet (needs G-12 life-cycle transition, #646)"))
    elseif has_kind(spec, ContinuousHouseholdSystem)
        return simulate(only(agents_of(spec, ContinuousHouseholdSystem)).model, T_periods; kwargs...)
    else
        return simulate(solve(spec), T_periods; kwargs...)
    end
end
