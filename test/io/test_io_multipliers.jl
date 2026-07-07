using Test, MacroEconometricModels

@testset "Multipliers" begin
    io = load_example(:wiot)
    om = multipliers(io; kind=:output, type=:I)
    @test om.values ≈ [1.518152, 1.452145] atol=1e-5   # column sums of L

    # Value-added (income) multipliers sum to 1 per sector when VA is the only
    # primary input — a clean accounting identity.
    vm = multipliers(io; kind=:income, type=:I)
    @test vm.values ≈ [1.0, 1.0] atol=1e-8

    em = multipliers(io; kind=:employment, type=:I)
    @test length(em.values) == 2
    @test em.kind == :employment

    # Type II ≥ Type I (household closure adds induced effects)
    om2 = multipliers(io; kind=:output, type=:II)
    @test all(om2.values .>= om.values .- 1e-8)
end
