using Test, MacroEconometricModels, LinearAlgebra

@testset "Impact scenario API" begin
    io = load_example(:wiot)
    L = leontief_inverse(io)

    # Unit final-demand shock to Agriculture → column 1 of L.
    r = impact(io, [1.0, 0.0]; kind=:output)
    @test r.by_sector ≈ L[:, 1] atol=1e-10
    @test r.total ≈ sum(L[:, 1]) atol=1e-10
    @test r.kind === :output
    @test r.type === :I
    @test isempty(r.fixed)

    # Dict API by name.
    rn = impact(io, Dict("Agriculture" => 1.0); kind=:output)
    @test rn.by_sector ≈ r.by_sector atol=1e-12

    # Income impact: h' Δx with h = va/x; Type I income multiplies to 1 on unit FD.
    ri = impact(io, [1.0, 0.0]; kind=:income)
    @test ri.total ≈ 1.0 atol=1e-8
    @test ri.kind === :income

    # Employment requires the satellite account (present on :wiot).
    re = impact(io, [1.0, 0.0]; kind=:employment)
    @test re.total > 0
    @test re.kind === :employment

    # Extension name as kind (CO2 on :wiot).
    rc = impact(io, [1.0, 0.0]; kind=:CO2)
    @test rc.total > 0

    # Type II ≥ Type I for output (household closure adds induced demand).
    r2 = impact(io, [1.0, 0.0]; kind=:output, type=:II)
    @test r2.total >= r.total - 1e-8
    @test r2.type === :II

    # Mixed model: fix Manufacturing output at baseline → Agriculture-only demand
    # shock should not change Manufacturing output.
    x2 = io.x[2]
    rm = impact(io, [10.0, 0.0]; kind=:output, fix=Dict(2 => x2))
    @test rm.type === :mixed
    @test rm.fixed == [2]
    # Manufacturing impact (Δx₂) ≈ 0 under fixed x₂.
    @test abs(rm.by_sector[2]) < 1e-8
    # Agriculture still expands.
    @test rm.by_sector[1] > 0

    @test_throws ArgumentError impact(io, [1.0]; kind=:output)
    @test_throws ArgumentError impact(io, [1.0, 0.0]; type=:III)
end
