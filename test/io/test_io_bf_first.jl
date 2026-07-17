using Test, MacroEconometricModels

@testset "Baqaee-Farhi first-order (Hulten)" begin
    io = load_example(:wiot)
    λ = domar_weights(io)
    gdp = sum(io.va)
    @test λ ≈ io.x ./ gdp                              # sales / GDP
    @test sum(λ) ≈ sum(io.x) / gdp

    bf = baqaee_farhi(io)
    # Hulten: first-order output elasticity to productivity = Domar weight
    @test bf.first_order ≈ λ atol=1e-10
    @test length(bf.upstreamness) == 2
    @test length(bf.downstreamness) == 2
    @test length(bf.influence) == 2
end
