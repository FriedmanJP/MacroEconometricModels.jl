# =============================================================================
# Exception Hierarchy (T146 / #245)
# =============================================================================
# A small typed exception hierarchy so callers can dispatch on failure kind
# (`try …; catch e; e isa ConvergenceError`) instead of string-matching bare
# `ErrorException` messages. `MacroModelError` is the abstract root; the two
# pre-existing narrow types (`DSGESolveError`, `StochasticSingularityError`) are
# defined elsewhere and stay independent.

"""
    MacroModelError <: Exception

Abstract root of the package's typed error hierarchy. Catch it to handle any
model-level failure (`ConvergenceError`, `IdentificationError`,
`SingularSystemError`) uniformly:

```julia
try
    identify_arias(model, restrictions, H)
catch e
    e isa MacroModelError && @warn "identification failed" e
end
```
"""
abstract type MacroModelError <: Exception end

"""
    ConvergenceError(msg; iters=nothing, residual=nothing) <: MacroModelError

An iterative solver failed to converge. Optional `iters`/`residual` carry the
last iteration count and residual for diagnostics.
"""
struct ConvergenceError <: MacroModelError
    msg::String
    iters::Union{Int,Nothing}
    residual::Union{Float64,Nothing}
end
ConvergenceError(msg::AbstractString; iters=nothing, residual=nothing) =
    ConvergenceError(String(msg), iters, residual)

"""
    IdentificationError(msg) <: MacroModelError

A structural model / identification scheme is not identified — no rotation
satisfies the sign or zero restrictions, or the restrictions over-constrain the
system.
"""
struct IdentificationError <: MacroModelError
    msg::String
end
IdentificationError(msg::AbstractString) = IdentificationError(String(msg))

"""
    SingularSystemError(msg; cond=nothing) <: MacroModelError

A linear system / factorization is (numerically) singular. Optional `cond`
carries the offending condition-number estimate.
"""
struct SingularSystemError <: MacroModelError
    msg::String
    cond::Union{Float64,Nothing}
end
SingularSystemError(msg::AbstractString; cond=nothing) =
    SingularSystemError(String(msg), cond)

function Base.showerror(io::IO, e::ConvergenceError)
    print(io, "ConvergenceError: ", e.msg)
    e.iters === nothing || print(io, " (iters=", e.iters, ")")
    e.residual === nothing || print(io, " (residual=", e.residual, ")")
end
Base.showerror(io::IO, e::IdentificationError) =
    print(io, "IdentificationError: ", e.msg)
function Base.showerror(io::IO, e::SingularSystemError)
    print(io, "SingularSystemError: ", e.msg)
    e.cond === nothing || print(io, " (cond≈", e.cond, ")")
end

"""
    _is_recoverable_draw_error(e) -> Bool

True for exceptions that legitimately arise from a single failed Monte-Carlo / bootstrap /
posterior draw — a singular system, a non-convergent solve, an indeterminate or
stochastically-singular DSGE solution. Such draws are caught, counted, and skipped in
resampling loops. Programming errors (`MethodError`, `BoundsError`, `UndefVarError`, …) are
NOT recoverable and must propagate rather than be silently swallowed (T145/#244). The DSGE
types (`DSGESolveError`/`StochasticSingularityError`) are resolved at call time.
"""
_is_recoverable_draw_error(e) =
    e isa MacroModelError ||
    e isa LinearAlgebra.SingularException ||
    e isa LinearAlgebra.PosDefException ||
    e isa LinearAlgebra.LAPACKException ||
    e isa DomainError ||
    e isa DSGESolveError ||
    e isa StochasticSingularityError
