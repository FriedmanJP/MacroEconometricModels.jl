# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Aqua
using MacroEconometricModels

@testset "Aqua.jl" begin
    # All gates enabled (#251). Verified on macOS-ARM (the platform previously cited as
    # persistent_tasks-flaky): ambiguities = 0 (no excludes needed), persistent_tasks
    # stable across repeated runs, deps_compat passes now that every dep/weakdep AND the
    # test-only extras (Aqua/Documenter/Logging/Test) carry [compat] entries.
    Aqua.test_all(MacroEconometricModels)
end
