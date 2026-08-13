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
    @test r.on === :output
    @test r.factors == [:L, :Y]

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

@testset "SDA n-factor two-polar residual ≈ 0" begin
    io0 = load_example(:wiot)
    Z1 = io0.Z .* 1.1
    Y1 = io0.Y .* 1.2
    io1 = IOData(Z1, Y1, [330.0 1100.0; 385.0 440.0]; sectors=io0.sectors, check=false)

    # 3-factor level × mix
    r3 = sda(io0, io1; factors=[:technology, :fd_level, :fd_mix])
    @test all(abs.(r3.residual) .< 1e-10)
    @test r3.factors == [:technology, :fd_level, :fd_mix]
    @test sum(r3.effects[f] for f in r3.factors) ≈ r3.total atol=1e-10
    # Proportional FD scale ⇒ pure level, zero mix
    @test all(abs.(r3.effects[:fd_mix]) .< 1e-10)

    # Non-proportional mix
    Y1b = copy(io0.Y)
    Y1b[1, :] .*= 1.5
    io1b = IOData(Z1, Y1b, [330.0 1100.0; 385.0 440.0]; sectors=io0.sectors, check=false)
    r3b = sda(io0, io1b; factors=[:technology, :fd_level, :fd_mix])
    @test all(abs.(r3b.residual) .< 1e-10)
    @test maximum(abs, r3b.effects[:fd_mix]) > 1e-8
end

@testset "SDA 2-factor path matches legacy keys" begin
    io0 = load_example(:wiot)
    Z1 = io0.Z .* 1.1
    Y1 = io0.Y .* 1.2
    io1 = IOData(Z1, Y1, [330.0 1100.0; 385.0 440.0]; sectors=io0.sectors, check=false)

    r_legacy = sda(io0, io1)
    r_named  = sda(io0, io1; factors=[:technology, :final_demand])
    @test r_legacy.effects[:L] ≈ r_named.effects[:technology] rtol=1e-12
    @test r_legacy.effects[:Y] ≈ r_named.effects[:final_demand] rtol=1e-12
    @test r_legacy.total ≈ r_named.total rtol=1e-12
end

@testset "SDA emission (extension) decomposition" begin
    io0 = load_example(:wiot)
    Z1 = io0.Z .* 1.1
    Y1 = io0.Y .* 1.2
    io1 = IOData(Z1, Y1, [330.0 1100.0; 385.0 440.0]; sectors=io0.sectors, check=false)
    add_extension!(io0, "CO2", [10.0 20.0]; stressors=["CO2"], unit=["kt"])
    add_extension!(io1, "CO2", [12.0 18.0]; stressors=["CO2"], unit=["kt"])

    re = sda(io0, io1; on="CO2",
             factors=[:intensity, :technology, :final_demand])
    @test re.on == "CO2"
    @test re.factors == [:intensity, :technology, :final_demand]
    @test length(re.total) == 1
    @test all(abs.(re.residual) .< 1e-10)
    # Explicit total: e = S L y
    e0 = vec(intensities(io0, "CO2") * leontief_inverse(io0) * vec(sum(io0.Y, dims=2)))
    e1 = vec(intensities(io1, "CO2") * leontief_inverse(io1) * vec(sum(io1.Y, dims=2)))
    @test re.total ≈ e1 .- e0 atol=1e-10
end

@testset "SDA fd_destination 4-factor" begin
    io0 = load_example(:wiot)
    Y0 = [300.0 50.0; 1400.0 300.0]
    Y1 = [360.0 60.0; 1600.0 400.0]
    Z1 = io0.Z .* 1.05
    io0b = IOData(io0.Z, Y0, io0.va; sectors=io0.sectors,
                  fd_cats=["C", "G"], va_cats=io0.va_cats, check=false)
    io1b = IOData(Z1, Y1, [330.0 1100.0; 385.0 440.0]; sectors=io0.sectors,
                  fd_cats=["C", "G"], va_cats=io0.va_cats, check=false)
    r = sda(io0b, io1b;
            factors=[:technology, :fd_level, :fd_mix, :fd_destination])
    @test all(abs.(r.residual) .< 1e-10)
    @test length(r.factors) == 4
    @test sum(r.effects[f] for f in r.factors) ≈ r.total atol=1e-10
end

@testset "SDA argument errors" begin
    io = load_example(:wiot)
    @test_throws ArgumentError sda(io, io; method=:bogus)
    @test_throws ArgumentError sda(io, io; factors=[:technology, :fd_level])  # mix missing
    @test_throws ArgumentError sda(io, io; factors=[:final_demand])           # no tech
    @test_throws ArgumentError sda(io, io; factors=[:technology, :final_demand, :fd_level])
    @test_throws ArgumentError sda(io, io; factors=[:intensity, :technology, :final_demand])
    @test_throws ArgumentError sda(io, io; method=:multiplicative,
                                    factors=[:technology, :fd_level, :fd_mix])
end

@testset "SDAResult serialization" begin
    io0 = load_example(:wiot)
    Z1 = io0.Z .* 1.1
    Y1 = io0.Y .* 1.2
    io1 = IOData(Z1, Y1, [330.0 1100.0; 385.0 440.0]; sectors=io0.sectors, check=false)
    r = sda(io0, io1)
    r2 = MacroEconometricModels._reconstruct_from_container(
        MacroEconometricModels._build_container(r))
    @test r2 isa SDAResult
    @test r2.effects[:L] ≈ r.effects[:L]
    @test r2.effects[:Y] ≈ r.effects[:Y]
    @test r2.total ≈ r.total
    @test r2.method === r.method
    @test r2.factors == r.factors
end
