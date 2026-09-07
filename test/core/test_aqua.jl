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
        Aqua.test_all(MacroEconometricModels; persistent_tasks=false)
        # Persistent-tasks probe as its own gate with one retry: it spawns a
        # fresh-env `Pkg.precompile` child that CI load can kill before done.log
        # exists (run 34106005278: "done.log was not created, but precompilation
        # exited"). A genuine lingering task fails deterministically, so a retry
        # only masks infra flakes — and attempt 2 reuses the warm depot.
        @testset "Persistent tasks" begin
            ok = false
            for attempt in 1:2
                ok = !Aqua.has_persistent_tasks(Base.PkgId(MacroEconometricModels))
                ok && break
                attempt == 1 &&
                    @info "Aqua persistent-tasks probe failed; retrying once with warm depot"
            end
            @test ok
        end
    end
end
