# Validate all display fixtures: construct, render, time, and dump raw + canonical.
using MacroEconometricModels
using Random, LinearAlgebra, Statistics

const _PROJ = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_PROJ, "test", "fixtures.jl"))
include(joinpath(_PROJ, "test", "display", "display_helpers.jl"))

println("Building fixtures...")
t_build = @elapsed fx = build_display_fixtures()
println("built $(length(fx)) fixtures in $(round(t_build, digits=2))s\n")

for f in fx
    print(rpad(f.name, 12))
    t = @elapsed s = _render(f.obj)
    # invariant probes
    no_omit  = !occursin("omitted", s)
    no_negz  = !occursin("-0.0", s)
    no_exp   = !occursin(r"e[+-]?\d{2,}", s)
    no_trip  = !occursin("\n\n\n", s)
    no_convT = !occursin(r"Converged\s+true", s)
    no_convF = !occursin(r"Converged\s+false", s)
    has_star = occursin("*", s)
    has_dash = occursin("—", s)
    flags = join([
        no_omit ? "" : "OMIT!",
        no_negz ? "" : "NEGZ!",
        no_exp  ? "" : "EXP!",
        no_trip ? "" : "TRIP!",
        (no_convT && no_convF) ? "" : "CONV!",
        f.stars && !has_star ? "NO*!" : "",
        f.ref && !has_dash ? "NOREF—!" : "",
    ], "")
    println("  $(round(t*1000, digits=0))ms  ", isempty(flags) ? "ok" : ">>> $flags")
end

println("\n\n============ RAW + CANONICAL DUMPS ============")
for f in fx
    s = _render(f.obj)
    println("\n\n########## $(f.name)  (stars=$(f.stars) ref=$(f.ref)) ##########")
    println("---- RAW ----")
    println(s)
    println("---- CANONICAL ----")
    println(_canonicalize(s))
end
println("\n=== dump done ===")
