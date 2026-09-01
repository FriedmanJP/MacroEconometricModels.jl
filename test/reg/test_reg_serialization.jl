# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
@testset "serialization stub: reg" begin
    @test isdefined(Main, :_assert_report_equal) || isdefined(@__MODULE__, :_assert_report_equal)
end
