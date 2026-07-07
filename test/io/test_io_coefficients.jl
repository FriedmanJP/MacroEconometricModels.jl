using Test, MacroEconometricModels, LinearAlgebra

@testset "Leontief & Ghosh on Miller & Blair 2-sector table" begin
    Z = [150.0 500.0; 200.0 100.0]
    Y = reshape([350.0, 1700.0], 2, 1)
    io = IOData(Z, Y, [650.0 1400.0]; sectors=["Agr", "Man"])

    A = technical_coefficients(io)
    @test A ≈ [0.15 0.25; 0.20 0.05]

    L = leontief_inverse(io)
    @test L ≈ [0.95 0.25; 0.20 0.85] ./ 0.7575 atol=1e-6

    m = leontief(io)
    @test m.A ≈ A
    @test m.L ≈ L

    # Output reproduces from final demand: x = L * y
    y = vec(sum(io.Y, dims=2))
    @test leontief_inverse(io) * y ≈ io.x atol=1e-8

    B = allocation_coefficients(io)
    @test B ≈ Diagonal(1 ./ io.x) * Z
    G = ghosh_inverse(io)
    @test G ≈ inv(I - B) atol=1e-10
    @test ghosh(io).G ≈ G
    # Ghosh and Leontief inverses are linked: G = x̂⁻¹ L x̂
    @test G ≈ Diagonal(1 ./ io.x) * L * Diagonal(io.x) atol=1e-8
end
