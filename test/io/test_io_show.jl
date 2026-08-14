using Test, MacroEconometricModels

@testset "show & report smoke" begin
    io = load_example(:wiot)
    objs = Any[io, leontief(io), ghosh(io), multipliers(io), linkages(io),
               sda(io, io), hypothetical_extraction(io, 1),
               price_model(io; dva=[0.1, 0.0]),
               impact(io, [1.0, 0.0]),
               network_stats(io),
               baqaee_farhi(io), footprint(io, "CO2"),
               ras([2.0 1.0; 1.0 2.0], [4.0, 5.0], [3.0, 6.0])]

    for obj in objs
        s = sprint(show, obj)
        @test !isempty(s)
    end
    # report() prints (returns nothing); redirect to a file to keep test output clean
    mktemp() do _path, ioh
        redirect_stdout(ioh) do
            for obj in objs
                report(obj)
            end
        end
    end
    @test report isa Function
end

@testset "BFMisallocation show" begin
    m = bf_misallocation(production_network(load_example(:wiot); mu=[1.2, 1.1]))
    @test occursin("Misallocation", sprint(show, m)) || occursin("frontier", sprint(show, m))
    mktemp() do _path, ioh
        redirect_stdout(ioh) do
            report(m)
        end
    end
end
