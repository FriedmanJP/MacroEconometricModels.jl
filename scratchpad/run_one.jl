# run_one.jl — run one or more test files standalone with fixtures + FAST loaded.
# Usage: <julia> --project=. scratchpad/run_one.jl test/core/test_covariance.jl [more...]
# Mirrors the runtests.jl environment: defines FAST, loads test/fixtures.jl, then
# includes each target file (each file self-wraps its own @testset).
using Test
using MacroEconometricModels
using Random, LinearAlgebra, Statistics

const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
const _PROJ = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_PROJ, "test", "fixtures.jl"))

for f in ARGS
    path = isabspath(f) ? f : joinpath(_PROJ, f)
    println("\n=========================== RUN: $f ===========================")
    include(path)
end
println("\n=== run_one.jl done ===")
