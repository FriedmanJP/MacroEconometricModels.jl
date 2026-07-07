using Test, MacroEconometricModels

@testset "load_example(:wiot)" begin
    io = load_example(:wiot)
    @test io isa IOData
    @test io.sectors == ["Agriculture", "Manufacturing"]
    @test io.x ≈ [1000.0, 2000.0]
    @test haskey(io.extensions, "CO2")
    @test haskey(io.extensions, "employment")
    @test technical_coefficients(io) ≈ [0.15 0.25; 0.20 0.05]
    # extension intensities S = F x̂⁻¹ are precomputed
    @test io.extensions["CO2"].S ≈ [100.0/1000 300.0/2000]
end
