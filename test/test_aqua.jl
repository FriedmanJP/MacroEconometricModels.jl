using Aqua
using MacroEconometricModels

@testset "Aqua.jl" begin
    Aqua.test_all(
        MacroEconometricModels;
        ambiguities=false,       # Skip ambiguity tests (can have false positives with StatsAPI)
        deps_compat=false,       # Skip deps compat (stdlib packages don't need compat)
        persistent_tasks=false,  # Skip persistent tasks (Turing.jl MCMC leaves tasks on macOS)
    )
end
