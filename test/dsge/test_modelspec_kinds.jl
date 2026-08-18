# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels

@testset "Follow-on agent kinds (#631)" begin
    @test DCEGMSystem <: AbstractAgentSystem
    @test LifeCycleSystem <: AbstractAgentSystem
    @test ContinuousHouseholdSystem <: AbstractAgentSystem
    prob = dcegm_retirement_model(; n_a=20, n_periods=4)
    sys = DCEGMSystem(prob)
    @test sys isa DCEGMSystem
    @test sys.problem === prob
end

@testset "G-06: empty endog/exog HA is partial GE (#643)" begin
    spec = @dsge begin
        parameters: beta_hh = 0.99
        heterogeneous: a in [0.0, 10.0], n_grid = 20, utility = log, discount = beta_hh, borrowing = 0.0
        idiosyncratic: e ~ Rouwenhorst(0.9, 0.1, 3)
        aggregation: K = sum(a)
    end
    @test spec isa ModelSpec
    @test spec.n_endog == 0
    @test spec.n_exog == 0
    @test isempty(spec.equations)
    @test MacroEconometricModels.has_kind(spec, HouseholdSystem)
end

@testset "G-06: RA still requires endog/exog (#643)" begin
    @test_throws LoadError eval(:(@dsge begin
        parameters: ρ = 0.9
        y[t] = ρ * y[t-1]
    end))
end

@testset "G-06: HA one-sided empty lists error (#643)" begin
    @test_throws LoadError eval(:(@dsge begin
        parameters: beta_hh = 0.99
        endogenous: K
        heterogeneous: a in [0.0, 10.0], n_grid = 20, utility = log, discount = beta_hh, borrowing = 0.0
        idiosyncratic: e ~ Rouwenhorst(0.9, 0.1, 3)
        aggregation: K = sum(a)
    end))
end
