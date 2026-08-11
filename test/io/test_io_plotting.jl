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
                network_stats(io))
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
