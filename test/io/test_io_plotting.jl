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
end
