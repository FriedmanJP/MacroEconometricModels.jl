using Test, MacroEconometricModels

@testset "SDA decomposition is exact" begin
    io0 = load_example(:wiot)
    # perturb final demand and technology for a second period
    Z1 = io0.Z .* 1.1
    Y1 = io0.Y .* 1.2
    io1 = IOData(Z1, Y1, [330.0 1100.0; 385.0 440.0]; sectors=io0.sectors, check=false)

    r = sda(io0, io1; method=:additive, factors=:LY, average=:two_polar)
    dx = io1.x .- io0.x
    @test r.effects[:L] .+ r.effects[:Y] ≈ dx atol=1e-8     # additive & exact
    @test all(abs.(r.residual) .< 1e-8)
    @test r.total ≈ dx atol=1e-8

    rm = sda(io0, io1; method=:multiplicative)
    @test haskey(rm.effects, :L)
    @test rm.method == :multiplicative
end
