using Test, MacroEconometricModels, LinearAlgebra

@testset "Environmental extensions" begin
    io = load_example(:wiot)
    S = intensities(io, "CO2")
    @test S ≈ [100.0/1000 300.0/2000]               # F x̂⁻¹
    M = emission_multipliers(io, "CO2")
    @test M ≈ S * leontief_inverse(io) atol=1e-10
    fp = footprint(io, "CO2")
    # total consumption-based footprint equals total direct emissions
    @test sum(fp.total) ≈ sum(io.extensions["CO2"].F) atol=1e-6
    @test sum(fp.by_sector) ≈ sum(fp.total) atol=1e-6

    add_extension!(io, "water", [10.0 20.0]; stressors=["H2O"], unit=["Ml"])
    @test haskey(io.extensions, "water")
    @test intensities(io, "water") ≈ [10.0/1000 20.0/2000]

    @test_throws ArgumentError intensities(io, "nonexistent")
end
