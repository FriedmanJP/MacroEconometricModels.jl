# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using Logging

const _MEM = MacroEconometricModels

# Behavioral check helper: run `f` under `with_min_level(_, level)` while a TestLogger
# is the base logger, and return the records that survived the min-level filter.
function _captured(f, level)
    tl = Test.TestLogger(min_level=Logging.BelowMinLevel)
    with_logger(tl) do
        _MEM.with_min_level(f, level)
    end
    return tl.logs
end

@testset "Structured logging facade (T249/#348)" begin

    @testset "with_min_level(Error) hides @warn/@info, surfaces @error" begin
        logs = _captured(() -> (@debug "d"; @info "i"; @warn "w"; @error "e"), Logging.Error)
        @test length(logs) == 1
        @test logs[1].level == Logging.Error
    end

    @testset "with_min_level admits its level and above" begin
        logs = _captured(() -> (@info "i"; @warn "w"; @error "e"), Logging.Warn)
        @test Set(r.level for r in logs) == Set([Logging.Warn, Logging.Error])
        # Debug level admits everything.
        logs2 = _captured(() -> (@debug "d"; @info "i"), Logging.Debug)
        @test length(logs2) == 2
    end

    @testset "Symbol level shorthand" begin
        @test _MEM._log_level(:debug) == Logging.Debug
        @test _MEM._log_level(:info)  == Logging.Info
        @test _MEM._log_level(:warn)  == Logging.Warn
        @test _MEM._log_level(:error) == Logging.Error
        @test _MEM._log_level(Logging.Warn) == Logging.Warn
        @test_throws ArgumentError _MEM._log_level(:bogus)
        # `:warn` shorthand routes @info away but keeps @warn.
        logs = _captured(() -> (@info "drop"; @warn "keep"), :warn)
        @test all(r.level == Logging.Warn for r in logs)
    end

    @testset "_suppress_warnings == with_min_level(_, Error)" begin
        logs = _captured2 = let tl = Test.TestLogger(min_level=Logging.BelowMinLevel)
            with_logger(tl) do
                _MEM._suppress_warnings() do
                    @warn "hidden"
                    @error "still surfaces"
                end
            end
            tl.logs
        end
        @test length(logs) == 1
        @test logs[1].level == Logging.Error
    end

    @testset "set_log_level installs a level and returns it (restore-safe)" begin
        old = global_logger()
        try
            @test set_log_level(:error) == Logging.Error
            @test Logging.min_enabled_level(global_logger()) == Logging.Error
            @test set_log_level(Logging.Info) == Logging.Info
            @test_throws ArgumentError set_log_level(:bogus)
        finally
            global_logger(old)
        end
    end

    @testset "with_min_level returns the wrapped computation's value" begin
        @test _MEM.with_min_level(() -> 42, Logging.Error) == 42
    end
end
