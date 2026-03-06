using Test
using MacroEconometricModels
using LinearAlgebra

@testset "Core Quadrature" begin

    @testset "Gauss-Hermite nodes and weights" begin
        # n=1: single node at 0, weight = √π
        nodes, weights = MacroEconometricModels._gauss_hermite_nodes_weights(1)
        @test length(nodes) == 1
        @test nodes[1] ≈ 0.0 atol=1e-14
        @test weights[1] ≈ sqrt(π) atol=1e-14

        # n=3: 3 nodes, weights sum to √π
        nodes3, weights3 = MacroEconometricModels._gauss_hermite_nodes_weights(3)
        @test length(nodes3) == 3
        @test sum(weights3) ≈ sqrt(π) atol=1e-12

        # n=5: symmetry of nodes and weights
        nodes5, weights5 = MacroEconometricModels._gauss_hermite_nodes_weights(5)
        @test length(nodes5) == 5
        for i in 1:2
            @test nodes5[i] ≈ -nodes5[6 - i] atol=1e-12
            @test weights5[i] ≈ weights5[6 - i] atol=1e-12
        end
        @test nodes5[3] ≈ 0.0 atol=1e-14  # middle node at 0

        # Integration accuracy: ∫ x² exp(-x²) dx = √π/2
        # Using GH: Σ wᵢ xᵢ² should equal √π/2
        nodes7, weights7 = MacroEconometricModels._gauss_hermite_nodes_weights(7)
        integral_x2 = sum(weights7 .* nodes7.^2)
        @test integral_x2 ≈ sqrt(π) / 2 atol=1e-10

        # ArgumentError for n=0
        @test_throws ArgumentError MacroEconometricModels._gauss_hermite_nodes_weights(0)
    end

    @testset "Scaled Gauss-Hermite (multivariate)" begin
        # 2D with known sigma matrix, weights sum to 1
        sigma = [1.0 0.3; 0.3 1.0]
        nodes, weights = MacroEconometricModels._gauss_hermite_scaled(3, sigma)
        @test size(nodes, 1) == 9      # 3^2 = 9
        @test size(nodes, 2) == 2      # 2 dimensions
        @test length(weights) == 9
        @test sum(weights) ≈ 1.0 atol=1e-12

        # 1D case: weights should also sum to 1
        sigma1d = reshape([2.0], 1, 1)
        nodes1, weights1 = MacroEconometricModels._gauss_hermite_scaled(5, sigma1d)
        @test size(nodes1, 2) == 1
        @test sum(weights1) ≈ 1.0 atol=1e-12
    end

    @testset "Adaptive Gauss-Hermite (Liu & Pierce 1994)" begin
        # ∫ N(x; 2, 1) dx ≈ 1
        # N(x; mu, sigma²) = 1/(sigma√(2π)) exp(-(x-mu)²/(2σ²))
        g_normal = x -> (1.0 / sqrt(2π)) * exp(-0.5 * (x - 2.0)^2)
        result1 = MacroEconometricModels._adaptive_gauss_hermite(g_normal, 2.0, 1.0, 15)
        @test result1 ≈ 1.0 atol=1e-10

        # ∫ N(x; 0, 0.1²) dx ≈ 1  (narrow Gaussian)
        sigma_narrow = 0.1
        g_narrow = x -> (1.0 / (sigma_narrow * sqrt(2π))) * exp(-0.5 * (x / sigma_narrow)^2)
        result2 = MacroEconometricModels._adaptive_gauss_hermite(g_narrow, 0.0, sigma_narrow, 15)
        @test result2 ≈ 1.0 atol=1e-10

        # E[X] where X ~ N(3, 4) = 3
        # E[X] = ∫ x · N(x; 3, 2²) dx
        sigma_wide = 2.0
        g_mean = x -> x * (1.0 / (sigma_wide * sqrt(2π))) * exp(-0.5 * ((x - 3.0) / sigma_wide)^2)
        result3 = MacroEconometricModels._adaptive_gauss_hermite(g_mean, 3.0, sigma_wide, 15)
        @test result3 ≈ 3.0 atol=1e-10
    end

end
