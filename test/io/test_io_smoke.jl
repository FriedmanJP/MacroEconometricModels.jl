using Test
using MacroEconometricModels

@testset "IO module loads" begin
    @test isdefined(MacroEconometricModels, :IOData)
end
