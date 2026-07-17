# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# Display goldens (T176 / #275). Snapshots the canonicalized full render of every
# fixture against `goldens/<name>.txt`. Canonicalization (see `_canonicalize`)
# masks the volatile numeric cells and column-padding whitespace, so the golden
# locks the STRUCTURE — titles, column headers, row labels, notes, `%`/integer
# labels, significance stars, `—`/`(ref)` markers, section order — while staying
# immune to last-ulp float drift across BLAS/OS. Regenerate the whole set with
#
#     MACRO_UPDATE_GOLDENS=1 <julia> --project=. scratchpad/run_one.jl \
#         test/display/test_display_goldens.jl
#
# after any Stage-11 sibling reformats output; eyeball the diff, then commit.

using Test
using MacroEconometricModels

isdefined(@__MODULE__, :build_display_fixtures) || include(joinpath(@__DIR__, "display_helpers.jl"))

@testset "Display goldens (T176/#275)" begin
    fixtures = display_fixtures()

    # Golden names are unique (one file per fixture).
    @test allunique(getproperty.(fixtures, :name))

    @testset "$(f.name)" for f in fixtures
        s = _render(f.obj)
        @test _check_golden(f.name, s)
    end

    if _UPDATE_GOLDENS
        @info "goldens (re)written to $_GOLDEN_DIR — eyeball the diff and commit"
    end
end
