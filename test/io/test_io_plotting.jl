using Test, MacroEconometricModels

# PLT-39: adopt the shared structural-assertion helper (parses EXTRACTED JSON
# literals, checks self-containment/DOCTYPE) instead of substring-only smoke.
# This file runs in the "Coverage-C + IO" group, which does not otherwise load the
# helper, so self-bootstrap it (dependency-free — plotrule A12).
isdefined(@__MODULE__, :check_plot) ||
    include(joinpath(@__DIR__, "..", "plotting", "plot_test_helpers.jl"))

@testset "IO plotting recipes" begin
    io = load_example(:wiot)

    # IOMultipliers, LinkageResult, LeontiefModel + new IO2-A result types.
    for obj in (multipliers(io), linkages(io), leontief(io),
                hypothetical_extraction(io, 1),
                price_model(io; dva=[0.1, 0.0]),
                impact(io, [1.0, 0.0]),
                network_stats(io),
                sda(io, io),
                ras([2.0 1.0; 1.0 2.0], [4.0, 5.0], [3.0, 6.0]))
        p = plot_result(obj)
        check_plot(p)
        assert_all_json_valid(p)
    end


    # MRIO trade-accounting plots (KWW two-country toy)
    Z = [50.0 50.0; 0.0 0.0]
    Y = [30.0 20.0; 50.0 0.0]
    va = reshape([100.0, 0.0], 1, 2)
    mrio = IOData(Z, Y, va; sectors=["A_g", "B_g"], regions=["A", "B"],
                  fd_cats=["A_fd", "B_fd"], va_cats=["VA"])
    add_extension!(mrio, "CO2", [10.0 0.0]; stressors=["CO2"], unit=["kt"])
    for obj in (export_decomposition(mrio, "A"),
                vertical_specialization(mrio, "B"),
                footprint(mrio, "CO2"; by=:region))
        p = plot_result(obj)
        check_plot(p)
        assert_all_json_valid(p)
    end

    # save_path branch for each recipe (C8): writes a self-contained document.
    d = mktempdir()
    for (obj, name) in ((multipliers(io), "m.html"),
                        (linkages(io), "l.html"),
                        (leontief(io), "h.html"),
                        (price_model(io; dva=[0.1, 0.0]), "p.html"),
                        (impact(io, [1.0, 0.0]), "i.html"),
                        (network_stats(io), "n.html"),
                        (hypothetical_extraction(io, 1), "e.html"))
        p = plot_result(obj; save_path=joinpath(d, name))
        @test p isa PlotOutput
        @test startswith(strip(read(joinpath(d, name), String)), "<!DOCTYPE html>")
    end
end

@testset "BFMisallocation plot" begin
    m = bf_misallocation(production_network(load_example(:wiot); mu=[1.2, 1.1]))
    p = plot_result(m)
    @test p isa PlotOutput
    check_plot(p)
    assert_all_json_valid(p)
end
