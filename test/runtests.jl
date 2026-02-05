using Test

@testset "MacroEconometricModels Package Tests" begin
    @testset "Aqua" begin
        include("test_aqua.jl")
    end

    @testset "Core VAR" begin
        include("test_core_var.jl")
    end

    @testset "Unit Root Tests" begin
        include("test_unitroot.jl")
    end

    @testset "Bayesian Estimation" begin
        include("test_bayesian.jl")
        include("test_samplers.jl") # Added sampler tests
    end

    @testset "Impulse Response Functions" begin
        include("test_irf.jl")
        include("test_irf_ci.jl") # Added IRF CI tests
    end

    @testset "Minnesota Prior" begin
        include("test_minnesota.jl")
    end

    @testset "BGR Optimization" begin
        include("test_bgr.jl") # Added BGR tests
    end

    @testset "StatsAPI Compatibility" begin
        include("test_statsapi.jl")
    end

    @testset "FEVD" begin
        include("test_fevd.jl")
    end

    @testset "Historical Decomposition" begin
        include("test_hd.jl")
    end

    @testset "Summary Tables" begin
        include("test_summary.jl")
    end

    @testset "Factor Model" begin
        include("test_factormodel.jl")
    end

    @testset "Dynamic Factor Model" begin
        include("test_dynamicfactormodel.jl")
    end

    @testset "Generalized Dynamic Factor Model" begin
        include("test_gdfm.jl")
    end

    @testset "Arias et al. (2018) SVAR Identification" begin
        include("test_arias2018.jl")
    end

    @testset "Local Projections" begin
        include("test_lp.jl")
    end

    @testset "Utility Functions" begin
        include("test_utils.jl")
    end

    @testset "Documentation Examples" begin
        include("test_examples.jl")
    end
end
