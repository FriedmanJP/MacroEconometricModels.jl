using Test, MacroEconometricModels

@testset "IO references" begin
    out = sprint(refs, [:baqaee_farhi_2019, :miller_blair_2009])
    @test occursin("Baqaee", out)
    @test occursin("Miller", out)

    # instance dispatch via IO result types
    io = load_example(:wiot)
    @test occursin("Leontief", sprint(refs, io))
    @test occursin("Baqaee", sprint(refs, baqaee_farhi(io)))
    @test occursin("Rasmussen", sprint(refs, linkages(io)))
end
