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

    # every IO result type resolves references
    for obj in (leontief(io), ghosh(io), multipliers(io), sda(io, io),
                ras([2.0 1.0; 1.0 2.0], [4.0, 5.0], [3.0, 6.0]),
                hypothetical_extraction(io, 1),
                price_model(io; dva=[0.1, 0.0]),
                impact(io, [1.0, 0.0]),
                network_stats(io),
                footprint(io, "CO2"))
        @test !isempty(sprint(refs, obj))
    end

    # MRIO result types (KWW two-country toy)
    Z = [50.0 50.0; 0.0 0.0]
    Y = [30.0 20.0; 50.0 0.0]
    va = reshape([100.0, 0.0], 1, 2)
    mrio = IOData(Z, Y, va; sectors=["A_g", "B_g"], regions=["A", "B"],
                  fd_cats=["A_fd", "B_fd"], va_cats=["VA"])
    add_extension!(mrio, "CO2", [10.0 0.0]; stressors=["CO2"], unit=["kt"])
    for obj in (export_decomposition(mrio, "A"),
                vertical_specialization(mrio, "B"),
                footprint(mrio, "CO2"; by=:region))
        @test !isempty(sprint(refs, obj))
    end
end
