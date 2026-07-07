using Test, MacroEconometricModels

@testset "show & report smoke" begin
    io = load_example(:wiot)
    objs = Any[io, leontief(io), ghosh(io), multipliers(io), linkages(io),
               sda(io, io), hypothetical_extraction(io, 1), baqaee_farhi(io),
               footprint(io, "CO2")]
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
