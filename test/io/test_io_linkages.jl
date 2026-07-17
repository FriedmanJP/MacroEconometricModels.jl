using Test, MacroEconometricModels, Statistics

@testset "Linkages & Rasmussen" begin
    io = load_example(:wiot)
    lk = linkages(io)
    @test length(lk.backward) == 2
    # Rasmussen indices are normalized so their mean is ≈ 1
    @test isapprox(mean(lk.Ui), 1.0; atol=1e-8)
    @test isapprox(mean(lk.Uj), 1.0; atol=1e-8)
    @test all(c -> c in (:key, :forward, :backward, :weak), lk.classification)
    @test key_sectors(io) == lk.classification
    @test rasmussen(io).Ui ≈ lk.Ui

    # forward=:leontief uses row sums of L instead of Ghosh G
    lk2 = linkages(io; forward=:leontief)
    @test lk2.forward ≈ vec(sum(leontief_inverse(io), dims=2))
end
