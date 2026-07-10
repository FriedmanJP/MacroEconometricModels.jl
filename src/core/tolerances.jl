# =============================================================================
# Central Tolerance Constants (T148 / #247)
# =============================================================================
# eps-derived default tolerances for iterative numerical solvers, exposed so
# callers can override the convergence gate via `abstol=`/`reltol=` kwargs.
# `default_abstol(Float64)` returns exactly `1e-8` — the historical package-wide
# convergence default (perfect_foresight, DSGE solvers) — so wiring solver
# defaults through it changes nothing on `Float64`. Other float types scale with
# `sqrt(eps(T))` so lower-precision work loosens automatically. Never hardcodes
# `eps(Float64)` (keeps `eps(T)` genericity).

"""
    default_abstol(::Type{T}=Float64) where {T<:AbstractFloat} -> T
    default_abstol(x::AbstractFloat)

Central default **absolute** convergence tolerance. Returns exactly `1e-8` for
`Float64` — the historical default across the DSGE / perfect-foresight solvers —
and `sqrt(eps(T))` for any other floating-point type, so lower-precision work
loosens automatically. Solvers expose it through an `abstol` keyword, so the gate
can be overridden without editing source.
"""
default_abstol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))
default_abstol(::Type{Float64}) = 1.0e-8
default_abstol(x::AbstractFloat) = default_abstol(typeof(x))
default_abstol() = default_abstol(Float64)

"""
    default_reltol(::Type{T}=Float64) where {T<:AbstractFloat} -> T
    default_reltol(x::AbstractFloat)

Central default **relative** tolerance, `sqrt(eps(T))` for every floating-point
type. There is no dominant historical relative gate in the package (NonlinearSolve
converges on `abstol` only; NLopt pins `xtol_rel`/`ftol_rel` as documented locals),
so this is a documented default for solvers that accept a relative tolerance.
"""
default_reltol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))
default_reltol(x::AbstractFloat) = default_reltol(typeof(x))
default_reltol() = default_reltol(Float64)
