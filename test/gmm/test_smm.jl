# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics

@testset "SMM Estimation" begin

@testset "Parameter Transforms" begin
    @testset "ParameterTransform construction" begin
        pt = ParameterTransform([0.0, -Inf, 0.0], [1.0, Inf, Inf])
        @test pt.lower == [0.0, -Inf, 0.0]
        @test pt.upper == [1.0, Inf, Inf]
    end

    @testset "Identity transform (unbounded)" begin
        pt = ParameterTransform([-Inf], [Inf])
        @test to_unconstrained(pt, [2.5]) ≈ [2.5]
        @test to_constrained(pt, [2.5]) ≈ [2.5]
    end

    @testset "Exp/log transform (lower bounded)" begin
        pt = ParameterTransform([0.0], [Inf])
        theta = [2.0]
        phi = to_unconstrained(pt, theta)
        @test phi ≈ [log(2.0)]
        @test to_constrained(pt, phi) ≈ theta
    end

    @testset "Negative exp transform (upper bounded)" begin
        pt = ParameterTransform([-Inf], [0.0])
        theta = [-3.0]
        phi = to_unconstrained(pt, theta)
        @test to_constrained(pt, phi) ≈ theta atol=1e-10
    end

    @testset "Logistic transform (bounded interval)" begin
        pt = ParameterTransform([0.0], [1.0])
        theta = [0.5]
        phi = to_unconstrained(pt, theta)
        theta_back = to_constrained(pt, phi)
        @test theta_back ≈ theta atol=1e-10
        # Boundary behavior
        @test to_constrained(pt, [-100.0])[1] >= 0.0
        @test to_constrained(pt, [100.0])[1] <= 1.0
        # Moderate values strictly inside bounds
        @test to_constrained(pt, [-10.0])[1] > 0.0
        @test to_constrained(pt, [10.0])[1] < 1.0
    end

    @testset "Round-trip multiple parameters" begin
        pt = ParameterTransform([0.0, -1.0, 0.0, -Inf], [1.0, 1.0, Inf, Inf])
        theta = [0.3, 0.0, 2.5, -1.0]
        phi = to_unconstrained(pt, theta)
        theta_back = to_constrained(pt, phi)
        @test theta_back ≈ theta atol=1e-10
    end

    @testset "Jacobian diagonal" begin
        pt = ParameterTransform([0.0, -Inf], [1.0, Inf])
        phi = [0.0, 3.0]
        J = transform_jacobian(pt, phi)
        @test size(J) == (2, 2)
        @test J[1, 2] == 0.0  # diagonal
        @test J[2, 1] == 0.0
        @test J[1, 1] > 0.0   # positive for logistic
        @test J[2, 2] == 1.0  # identity for unbounded
    end
end

end  # outer testset
