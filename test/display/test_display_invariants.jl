# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# Display invariants (T176 / #275). Renders every fixture through `_render` and
# asserts the RAW-string properties that the Stage-11 display batch (#260–#274)
# established. Unlike the goldens, these are regex/`occursin` checks — they are
# cross-platform robust and fail loudly the moment a future edit reintroduces a
# cropped table, a `-0.0`, a raw boolean, or a stray blank band. Each invariant
# names the audit defect it locks.

using Test
using MacroEconometricModels

isdefined(@__MODULE__, :build_display_fixtures) || include(joinpath(@__DIR__, "display_helpers.jl"))

# S1 — non-TTY cropping is disabled: no PrettyTables "… columns/rows omitted" footer.
_no_crop(s)     = !occursin("omitted", s)
# S2 — `_fmt` normalizes negative zero: no `-0.0`, `-0.00`, `-0.0000` cell (a value
# like `-0.0283` is fine — only an all-zero-after-the-point token is a defect).
_no_neg_zero(s) = !occursin(r"-0\.0+(?![0-9])", s)
# S2 — no raw un-`_fmt`'d float leaking in scientific form. `_fmt`'s `%.3g` fallback
# for tiny/huge cells caps at 2 mantissa decimals (e.g. `4.46e-12`); a token with ≥3
# mantissa decimals before the exponent is an unformatted `round()`/`string(float)` leak.
_no_raw_exp(s)  = !occursin(r"\.\d{3,}[eE][-+]?\d", s)
# S7 — empty column-label rows are suppressed, so no degenerate blank band. Section
# separators legitimately span up to two blank lines; three or more never occur.
_no_blank_band(s) = !occursin("\n\n\n\n", s)
# S8 — booleans render as Yes/No, never raw `true`/`false`, anywhere in the table.
_no_raw_bool(s) = !occursin(r"\b(?:true|false)\b", s)

@testset "Display invariants (T176/#275)" begin
    fixtures = display_fixtures()
    @test length(fixtures) >= 15

    @testset "$(f.name)" for f in fixtures
        s = _render(f.obj)

        @test !isempty(strip(s))            # the fixture actually rendered something
        @test _no_crop(s)                   # S1
        @test _no_neg_zero(s)               # S2
        @test _no_raw_exp(s)                # S2
        @test _no_blank_band(s)             # S7
        @test _no_raw_bool(s)               # S8

        if f.stars
            # A high-SNR estimate → the coefficient table must carry a significance star.
            @test occursin("*", s)
        end

        if f.ref
            # S6 — the reference row is labelled `(ref)`, carries dashes for the
            # estimate/stat/CI, and never prints a computed significance star.
            @test occursin("—", s)
            reflines = filter(l -> occursin("(ref)", l), split(s, '\n'))
            @test length(reflines) == 1
            @test occursin("—", reflines[1]) && !occursin("*", reflines[1])
        end
    end
end
