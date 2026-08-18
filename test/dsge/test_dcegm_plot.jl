# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels
sol = dcegm_solve(dcegm_retirement_model(; n_a=40, n_periods=6))
p = plot_result(sol)
@test p isa PlotOutput
@test occursin("d3", p.html)
p2 = plot_result(sol; view=:threshold)
@test p2 isa PlotOutput
@test_throws ArgumentError plot_result(sol; view=:invalid)
