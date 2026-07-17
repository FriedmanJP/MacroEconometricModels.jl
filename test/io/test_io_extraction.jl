using Test, MacroEconometricModels, LinearAlgebra

@testset "Hypothetical extraction" begin
    io = load_example(:wiot)
    r = hypothetical_extraction(io, 1)
    @test r.extracted == [1]
    @test r.total_loss > 0
    @test r.total_loss ≈ sum(r.sector_loss) atol=1e-8
    # extracting by name matches extracting by index
    rn = hypothetical_extraction(io, "Agriculture")
    @test rn.total_loss ≈ r.total_loss
    # extracting multiple sectors loses at least as much as extracting one
    rall = hypothetical_extraction(io, [1, 2])
    @test rall.total_loss >= r.total_loss - 1e-8
end
