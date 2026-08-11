using Test, MacroEconometricModels, LinearAlgebra

@testset "Price model (Leontief cost-push)" begin
    # Miller & Blair Table 2.3 (load_example :wiot):
    #   A = [0.15 0.25; 0.20 0.05],  v = [0.65, 0.70]
    # Base prices: p = (I − A')⁻¹ v = 1  (value table identity).
    io = load_example(:wiot)
    A = technical_coefficients(io)
    v = vec(sum(io.va, dims=1)) ./ io.x
    @test v ≈ [0.65, 0.70] atol=1e-12

    # Zero shock → zero price change, base p = ones.
    r0 = price_model(io)
    @test r0.dp ≈ zeros(2) atol=1e-12
    @test r0.p ≈ ones(2) atol=1e-12
    @test r0.mode === :leontief

    # Oracle: Δv = [0.10, 0.0]  ⇒  Δp = L' Δv
    # L ≈ [1.2541254  0.3300330;  0.2640264  1.1221122]
    # Δp ≈ [0.12541254, 0.03300330]
    L = leontief_inverse(io)
    dv = [0.10, 0.0]
    r = price_model(io; dva=dv)
    @test r.dv ≈ dv
    @test r.dp ≈ L' * dv atol=1e-10
    @test r.dp ≈ [0.1254125412541254, 0.033003300330033] atol=1e-8
    @test r.p ≈ ones(2) .+ r.dp atol=1e-12

    # Dict / name API for the same shock on Agriculture.
    rn = price_model(io; dva=Dict("Agriculture" => 0.10))
    @test rn.dp ≈ r.dp atol=1e-12

    # Tax shock adds to dva.
    rtax = price_model(io; dva=[0.05, 0.0], dtax=[0.05, 0.0])
    @test rtax.dv ≈ [0.10, 0.0]
    @test rtax.dp ≈ r.dp atol=1e-12

    # Manufacturing VA shock: Δv = [0, 0.1] ⇒ Δp = 0.1 · L[2, :]'
    r2 = price_model(io; dva=[0.0, 0.10])
    @test r2.dp ≈ 0.10 .* vec(L[2, :]) atol=1e-10

    # Ghosh dual runs and differs from Leontief under a non-symmetric A.
    rg = price_model(io; dva=dv, mode=:ghosh)
    @test rg.mode === :ghosh
    @test length(rg.dp) == 2
    @test !(rg.dp ≈ r.dp)

    @test_throws ArgumentError price_model(io; mode=:foo)
    @test_throws ArgumentError price_model(io; dva=[1.0])   # wrong length
end
