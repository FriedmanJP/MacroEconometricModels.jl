# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Aqua
using MacroEconometricModels

@testset "Aqua.jl" begin
    Aqua.test_all(
        MacroEconometricModels;
        ambiguities=false,       # Skip ambiguity tests (can have false positives with StatsAPI)
        deps_compat=false,       # Skip deps compat (stdlib packages don't need compat)
        persistent_tasks=false,  # Flaky on macOS ARM (dependency init timers)
    )
end
