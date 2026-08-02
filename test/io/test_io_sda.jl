using Test, MacroEconometricModels

@testset "SDA decomposition is exact" begin
    io0 = load_example(:wiot)
    # perturb final demand and technology for a second period
    Z1 = io0.Z .* 1.1
    Y1 = io0.Y .* 1.2
    io1 = IOData(Z1, Y1, [330.0 1100.0; 385.0 440.0]; sectors=io0.sectors, check=false)

    r = sda(io0, io1; method=:additive)
    dx = io1.x .- io0.x
    @test r.effects[:L] .+ r.effects[:Y] ≈ dx atol=1e-8     # additive & exact
    @test all(abs.(r.residual) .< 1e-8)
    @test r.total ≈ dx atol=1e-8

    rm = sda(io0, io1; method=:multiplicative)
    @test haskey(rm.effects, :L)
    @test haskey(rm.effects, :Y)
    @test rm.method == :multiplicative
    # Two-polar geometric mean: product of effects recovers the output ratio.
    @test rm.effects[:L] .* rm.effects[:Y] ≈ rm.total rtol=1e-8
end

@testset "SDA multiplicative is genuinely two-polar (#517)" begin
    # Non-proportional demand change: the two polar decompositions differ, so
    # this discriminates the geometric mean from either single polar — and from
    # the old construction Y_eff := ratio ./ L_eff, under which the product
    # identity held for ANY L_eff. (A proportional Δy makes both polars
    # coincide and the old test vacuous.)
    io0 = load_example(:wiot)
    Z1 = io0.Z .* 1.1
    Y1 = copy(io0.Y)
    Y1[1, :] .*= 1.5                        # sector-1 demand only
    io1 = IOData(Z1, Y1, [330.0 1100.0; 385.0 440.0]; sectors=io0.sectors, check=false)

    rm = sda(io0, io1; method=:multiplicative)
    L0 = leontief_inverse(io0); L1 = leontief_inverse(io1)
    y0 = vec(sum(io0.Y, dims=2)); y1 = vec(sum(io1.Y, dims=2))
    x0 = L0 * y0; x1 = L1 * y1
    L_p1 = (L1 * y0) ./ x0
    L_p2 = (L1 * y1) ./ (L0 * y1)
    Y_p1 = (L0 * y1) ./ x0
    Y_p2 = (L1 * y1) ./ (L1 * y0)
    @test rm.effects[:L] ≈ sqrt.(L_p1 .* L_p2) rtol=1e-10
    @test rm.effects[:Y] ≈ sqrt.(Y_p1 .* Y_p2) rtol=1e-10
    @test !(rm.effects[:L] ≈ L_p1)          # ≠ single polar
    @test rm.effects[:L] .* rm.effects[:Y] ≈ rm.total rtol=1e-10
    @test all(abs.(rm.residual) .< 1e-10)
end
