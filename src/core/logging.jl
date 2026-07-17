# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Structured logging facade (T249 / #348)
# =============================================================================
# All library diagnostic output flows through the `Logging` stdlib rather than
# bare `println`/`print` to `stdout`:
#   - `@debug`  — verbose iteration traces (solver residuals, EGM/VFI progress).
#                 OFF by default; opt in with `set_log_level(:debug)` or the
#                 `JULIA_DEBUG` environment variable.
#   - `@info`   — high-level, one-shot progress a user might want.
#   - `@warn`   — recoverable numerical issues.
#   - `@error` / typed exceptions — genuine failures.
# The library is quiet by default: `@debug` is below the default logger's
# threshold, so no progress text reaches the terminal unless the user asks.

"""
Logger that forwards only records at or above `min_level` to `inner`, discarding
lower-severity records. Lets [`with_min_level`](@ref) / `_suppress_warnings` hide
`@warn`/`@info` in noisy Monte-Carlo/bootstrap loops while ALWAYS surfacing
`@error` — unlike a `NullLogger`, which silently swallows errors too.
"""
struct _MinLevelLogger{L<:Base.CoreLogging.AbstractLogger} <: Base.CoreLogging.AbstractLogger
    inner::L
    min_level::Base.CoreLogging.LogLevel
end
Base.CoreLogging.min_enabled_level(l::_MinLevelLogger) = l.min_level
Base.CoreLogging.shouldlog(l::_MinLevelLogger, level, _module, group, id) =
    level >= l.min_level && Base.CoreLogging.shouldlog(l.inner, level, _module, group, id)
Base.CoreLogging.handle_message(l::_MinLevelLogger, level, message, _module, group, id,
                                filepath, line; kwargs...) =
    Base.CoreLogging.handle_message(l.inner, level, message, _module, group, id,
                                    filepath, line; kwargs...)
Base.CoreLogging.catch_exceptions(l::_MinLevelLogger) = Base.CoreLogging.catch_exceptions(l.inner)

# Normalize a level given as a `Symbol` (:debug/:info/:warn/:error) or a raw `LogLevel`.
_log_level(level::Base.CoreLogging.LogLevel) = level
function _log_level(level::Symbol)
    level === :debug ? Logging.Debug :
    level === :info  ? Logging.Info  :
    level === :warn  ? Logging.Warn  :
    level === :error ? Logging.Error :
    throw(ArgumentError("unknown log level :$level; use :debug, :info, :warn, or :error"))
end

"""
    with_min_level(f, level) -> f()

Run the zero-argument function `f` with a logger that forwards only log records at
or above `level` to the current logger, discarding lower-severity records. `level`
is a `Logging.LogLevel` (`Logging.Debug`, `Logging.Info`, `Logging.Warn`,
`Logging.Error`) or the matching `Symbol` (`:debug`, `:info`, `:warn`, `:error`).

This generalizes the internal `_suppress_warnings` (which pins `Logging.Error`):
use it to mute `@debug`/`@info`/`@warn` noise emitted inside a wrapped computation
while ALWAYS surfacing `@error`.

```julia
with_min_level(Logging.Error) do
    # per-draw @warn noise from the bootstrap is hidden; a genuine @error still shows
    bootstrap_irf(model; reps = 1000)
end
```
"""
with_min_level(f, level) =
    Base.CoreLogging.with_logger(f, _MinLevelLogger(Base.CoreLogging.current_logger(), _log_level(level)))

"""Suppress `@warn`/`@info` within `f()` while ALWAYS surfacing `@error`. Used in
bootstrap / Monte-Carlo loops where per-draw warnings are noise but a genuine error must
never be hidden (T145/#244 — the old `NullLogger` swallowed errors too). Thin alias for
`with_min_level(f, Logging.Error)`."""
_suppress_warnings(f) = with_min_level(f, Logging.Error)

"""
    set_log_level(level) -> LogLevel

Set the global minimum log level for the current Julia session and return the
resolved `Logging.LogLevel`. `level` is a `Symbol` (`:debug`, `:info`, `:warn`,
`:error`) or a `Logging.LogLevel`.

The package is quiet by default: solver iteration traces are emitted at `@debug`,
which the default logger hides. `set_log_level(:debug)` turns those traces on;
`set_log_level(:error)` silences `@warn`/`@info` chatter globally. This installs a
fresh `ConsoleLogger(stderr, level)` as the global logger — for a scoped change
that restores the previous logger afterwards, prefer [`with_min_level`](@ref) or
`Logging.with_logger`.

```julia
set_log_level(:debug)                 # show solver iteration traces
collocation_solver(spec; verbose = false)
set_log_level(:info)                  # back to the default verbosity
```
"""
function set_log_level(level)
    lvl = _log_level(level)
    Logging.global_logger(Logging.ConsoleLogger(stderr, lvl))
    return lvl
end
