# Counterfactual module — policy-rule templates (CF-02, #382)
#
# Rules are encoded as banded H×H matrices acting on stacked H-period paths
# (row t = the rule restriction at horizon t), with the lag operator L realized
# as the subdiagonal shift matrix. Convention (fixed here, used by every later
# CF task): a rule is  Σᵢ A_x[i]·x[i] + Σₖ A_z[k]·z[k] = wedge,  and the
# projection kernel receives  b = Σᵢ A_x[i]·x_base[i] + Σₖ A_z[k]·z_base[k] − wedge.
#
# The framework compares stabilization rules around a FIXED steady state;
# steady-state-changing policies (e.g. a different inflation-target level) are
# out of scope.

"""
    _lag_shift(::Type{T}, H) -> Matrix{T}

The `H × H` subdiagonal-ones lag-shift matrix `L` (`(L·z)ₜ = z_{t-1}`, with
`(L·z)₁ = 0` at the truncation edge).
"""
function _lag_shift(::Type{T}, H::Int) where {T<:AbstractFloat}
    L = zeros(T, H, H)
    for t in 2:H
        L[t, t-1] = one(T)
    end
    return L
end

# Shared validation: a referenced symbol must be present in the given axis;
# returns its index so rule rows match PolicyCausalEffects entries by symbol.
function _cf_require_var(sym::Symbol, axis::AbstractVector{Symbol}, field::String, fn::String)
    idx = findfirst(==(sym), axis)
    idx === nothing && throw(ArgumentError(
        "$fn: $field = :$sym not found in outcomes = $axis"))
    return idx
end

function _cf_single_instrument(instruments, fn::String)
    length(instruments) == 1 || throw(ArgumentError(
        "$fn: expected exactly 1 instrument, got $(length(instruments))"))
    return nothing
end

"""
    rate_peg_rule([T=Float64,] H; outcomes=[:infl, :ygap], instruments=[:rate])

[`PolicyRule`](@ref) pegging the (single) policy instrument at its steady-state
value: `z = 0` in gap space, i.e. `A_z = [I]`, all `A_x = 0`, `wedge = 0`.

The rule convention is `Σᵢ A_x[i]·x[i] + Σₖ A_z[k]·z[k] = wedge`. This template
encodes a stabilization rule around a fixed steady state — a permanently
different instrument *level* is out of scope.
"""
function rate_peg_rule(::Type{T}, H::Int;
                       outcomes::AbstractVector{Symbol}=[:infl, :ygap],
                       instruments::AbstractVector{Symbol}=[:rate]) where {T<:AbstractFloat}
    _cf_single_instrument(instruments, "rate_peg_rule")
    PolicyRule(outcomes=outcomes, instruments=instruments,
               A_x=[zeros(T, H, H) for _ in outcomes],
               A_z=[Matrix{T}(I, H, H)],
               wedge=zeros(T, H), name="rate peg")
end
rate_peg_rule(H::Int; kwargs...) = rate_peg_rule(Float64, H; kwargs...)

"""
    rate_target_rule([T=Float64,] H, path; outcomes=[:infl, :ygap], instruments=[:rate])

[`PolicyRule`](@ref) pegging the (single) policy instrument to an arbitrary
length-`H` gap path: `A_z = [I]`, `wedge = path` (so the rule reads `z = path`).
Stabilization-rule convention as in [`rate_peg_rule`](@ref).
"""
function rate_target_rule(::Type{T}, H::Int, path::AbstractVector{<:Real};
                          outcomes::AbstractVector{Symbol}=[:infl, :ygap],
                          instruments::AbstractVector{Symbol}=[:rate]) where {T<:AbstractFloat}
    _cf_single_instrument(instruments, "rate_target_rule")
    length(path) == H || throw(ArgumentError(
        "rate_target_rule: path: expected length H = $H, got $(length(path))"))
    PolicyRule(outcomes=outcomes, instruments=instruments,
               A_x=[zeros(T, H, H) for _ in outcomes],
               A_z=[Matrix{T}(I, H, H)],
               wedge=Vector{T}(path), name="rate target path")
end
rate_target_rule(H::Int, path::AbstractVector{<:Real}; kwargs...) =
    rate_target_rule(Float64, H, path; kwargs...)

"""
    inflation_target_rule([T=Float64,] H; pi_var=:infl, outcomes=[:infl, :ygap],
                          instruments=[:rate])

[`PolicyRule`](@ref) for strict inflation targeting: `π = 0` period by period
(`A_x[pi_var] = I`, everything else zero). Rule convention and steady-state
caveat as in [`rate_peg_rule`](@ref) — this targets the inflation *gap*, not a
different target level.
"""
function inflation_target_rule(::Type{T}, H::Int; pi_var::Symbol=:infl,
                               outcomes::AbstractVector{Symbol}=[:infl, :ygap],
                               instruments::AbstractVector{Symbol}=[:rate]) where {T<:AbstractFloat}
    i_pi = _cf_require_var(pi_var, outcomes, "pi_var", "inflation_target_rule")
    A_x = [zeros(T, H, H) for _ in outcomes]
    A_x[i_pi] = Matrix{T}(I, H, H)
    PolicyRule(outcomes=outcomes, instruments=instruments,
               A_x=A_x, A_z=[zeros(T, H, H) for _ in instruments],
               wedge=zeros(T, H), name="strict inflation targeting")
end
inflation_target_rule(H::Int; kwargs...) = inflation_target_rule(Float64, H; kwargs...)

"""
    output_gap_rule([T=Float64,] H; y_var=:ygap, outcomes=[:infl, :ygap],
                    instruments=[:rate])

[`PolicyRule`](@ref) for strict output-gap targeting: `y = 0` period by period
(`A_x[y_var] = I`, everything else zero). Rule convention and steady-state
caveat as in [`rate_peg_rule`](@ref).
"""
function output_gap_rule(::Type{T}, H::Int; y_var::Symbol=:ygap,
                         outcomes::AbstractVector{Symbol}=[:infl, :ygap],
                         instruments::AbstractVector{Symbol}=[:rate]) where {T<:AbstractFloat}
    i_y = _cf_require_var(y_var, outcomes, "y_var", "output_gap_rule")
    A_x = [zeros(T, H, H) for _ in outcomes]
    A_x[i_y] = Matrix{T}(I, H, H)
    PolicyRule(outcomes=outcomes, instruments=instruments,
               A_x=A_x, A_z=[zeros(T, H, H) for _ in instruments],
               wedge=zeros(T, H), name="output-gap targeting")
end
output_gap_rule(H::Int; kwargs...) = output_gap_rule(Float64, H; kwargs...)

"""
    ngdp_rule([T=Float64,] H; pi_var=:infl, y_var=:ygap, outcomes=[:infl, :ygap],
              instruments=[:rate])

[`PolicyRule`](@ref) for nominal-GDP (level) targeting: row `t` imposes
`π_t + y_t − y_{t−1} = 0` for `t ≥ 2` and `π_1 + y_1 = 0` at the truncation
edge (`A_x[pi_var] = I`, `A_x[y_var] = I − L`; the first row of `I − L` is
`e₁'` automatically since `L` has an empty first row). Rule convention and
steady-state caveat as in [`rate_peg_rule`](@ref).
"""
function ngdp_rule(::Type{T}, H::Int; pi_var::Symbol=:infl, y_var::Symbol=:ygap,
                   outcomes::AbstractVector{Symbol}=[:infl, :ygap],
                   instruments::AbstractVector{Symbol}=[:rate]) where {T<:AbstractFloat}
    i_pi = _cf_require_var(pi_var, outcomes, "pi_var", "ngdp_rule")
    i_y = _cf_require_var(y_var, outcomes, "y_var", "ngdp_rule")
    i_pi == i_y && throw(ArgumentError("ngdp_rule: pi_var and y_var must differ, both are :$pi_var"))
    A_x = [zeros(T, H, H) for _ in outcomes]
    A_x[i_pi] = Matrix{T}(I, H, H)
    A_x[i_y] = Matrix{T}(I, H, H) - _lag_shift(T, H)
    PolicyRule(outcomes=outcomes, instruments=instruments,
               A_x=A_x, A_z=[zeros(T, H, H) for _ in instruments],
               wedge=zeros(T, H), name="NGDP targeting")
end
ngdp_rule(H::Int; kwargs...) = ngdp_rule(Float64, H; kwargs...)

"""
    taylor_rule([T=Float64,] H; rho=0.5, phi_pi=1.5, phi_y=1.0, z_lag=0.0,
                pi_var=:infl, y_var=:ygap, outcomes=[:infl, :ygap],
                instruments=[:rate])

[`PolicyRule`](@ref) for an inertial Taylor rule

`z_t = ρ·z_{t−1} + (1−ρ)·(φ_π·π_t + φ_y·y_t)`

encoded as `A_z = [I − ρL]`, `A_x[pi_var] = −(1−ρ)·φ_π·I`,
`A_x[y_var] = −(1−ρ)·φ_y·I`, and `wedge = (ρ·z_lag, 0, …, 0)` — the pre-sample
lagged instrument `z_lag` enters period 1 through the wedge. `rho = 0` reduces
to the static rule. The Caravello–McKay–Wolf parameterization is reachable via
`rho=0.85, phi_pi=2.0, phi_y=0.25`. Rule convention and steady-state caveat as
in [`rate_peg_rule`](@ref).
"""
function taylor_rule(::Type{T}, H::Int; rho::Real=0.5, phi_pi::Real=1.5,
                     phi_y::Real=1.0, z_lag::Real=0.0,
                     pi_var::Symbol=:infl, y_var::Symbol=:ygap,
                     outcomes::AbstractVector{Symbol}=[:infl, :ygap],
                     instruments::AbstractVector{Symbol}=[:rate]) where {T<:AbstractFloat}
    _cf_single_instrument(instruments, "taylor_rule")
    i_pi = _cf_require_var(pi_var, outcomes, "pi_var", "taylor_rule")
    i_y = _cf_require_var(y_var, outcomes, "y_var", "taylor_rule")
    i_pi == i_y && throw(ArgumentError("taylor_rule: pi_var and y_var must differ, both are :$pi_var"))
    A_x = [zeros(T, H, H) for _ in outcomes]
    A_x[i_pi] = Matrix{T}(-(1 - T(rho)) * T(phi_pi) * I, H, H)
    A_x[i_y] = Matrix{T}(-(1 - T(rho)) * T(phi_y) * I, H, H)
    A_z = Matrix{T}(I, H, H) - T(rho) * _lag_shift(T, H)
    wedge = zeros(T, H)
    wedge[1] = T(rho) * T(z_lag)
    PolicyRule(outcomes=outcomes, instruments=instruments,
               A_x=A_x, A_z=[A_z], wedge=wedge,
               name="taylor(ρ=$(rho), φπ=$(phi_pi), φy=$(phi_y))")
end
taylor_rule(H::Int; kwargs...) = taylor_rule(Float64, H; kwargs...)

"""
    custom_rule(A_x, A_z; outcomes, instruments, wedge=zeros(H), name="custom")

Raw-matrix escape hatch: build a [`PolicyRule`](@ref) directly from `H × H`
coefficient matrices (one per outcome / instrument, validated by the
`PolicyRule` constructor). Rule convention
`Σᵢ A_x[i]·x[i] + Σₖ A_z[k]·z[k] = wedge` as in [`rate_peg_rule`](@ref).
"""
custom_rule(A_x::AbstractVector{<:AbstractMatrix}, A_z::AbstractVector{<:AbstractMatrix};
            outcomes::AbstractVector{Symbol}, instruments::AbstractVector{Symbol},
            wedge::Union{Nothing,AbstractVector{<:Real}}=nothing,
            name::AbstractString="custom") =
    PolicyRule(outcomes=outcomes, instruments=instruments, A_x=A_x, A_z=A_z,
               wedge=wedge, name=name)
