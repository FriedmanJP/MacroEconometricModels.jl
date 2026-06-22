using Test, MacroEconometricModels

@testset "IO plotting recipes" begin
    io = load_example(:wiot)
    p1 = plot_result(multipliers(io))
    @test p1 isa PlotOutput
    @test !isempty(p1.html)
    p2 = plot_result(linkages(io))
    @test p2 isa PlotOutput
    p3 = plot_result(leontief(io))
    @test p3 isa PlotOutput

    # save_path branch for each recipe
    d = mktempdir()
    plot_result(multipliers(io); save_path=joinpath(d, "m.html"))
    plot_result(linkages(io); save_path=joinpath(d, "l.html"))
    plot_result(leontief(io); save_path=joinpath(d, "h.html"))
    @test isfile(joinpath(d, "m.html"))
    @test isfile(joinpath(d, "l.html"))
    @test isfile(joinpath(d, "h.html"))
end
