# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Aqua
using MacroEconometricModels

@testset "Aqua.jl" begin
    # All gates enabled (#251) except persistent_tasks on Windows, where Aqua's
    # lingering-task probe false-positives under the threaded CI runner.
    if Sys.iswindows()
        Aqua.test_all(MacroEconometricModels; persistent_tasks=false)
    else
        Aqua.test_all(MacroEconometricModels)
    end
end
