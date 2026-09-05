# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# DGP-01 (#790): white-noise lint. Fails when a file under test/ passes a
# bare randn()/rand() array directly to an estimator/test, or draws from the
# global RNG at all — unless the file is on the allowlist
# (test/dgp/ALLOWLIST.md). Per-module issues (DGP-02…18) shrink the
# grandfathered section as they migrate their files to dgp.* simulators.
#
# Single-line static check by design: every bare draw is banned outright, so
# multi-line estimator calls are covered too (their bare draws still fail).

using Test

const _RNG_FIRST_ARG = r"^\s*(rng\b|MersenneTwister|Random\.MersenneTwister|Xoshiro|RandomDevice|TaskLocalRNG)"
const _DRAW_CALL = r"\brandn?!?\s*\("
const _EST_CALL =
    r"(?<![\w.])(estimate_\w+|identify_\w+|nowcast_\w+|forecast|fevd|irf|historical_decomposition|\w+_test)\s*\("

# Strip string literals, then comments, tracking """ blocks across lines.
# Char-based (not byte-based): test sources contain Unicode (yₜ, β, ε).
function _code_part(line::String, in_triple::Bool)
    cs = collect(line)
    out = IOBuffer()
    i, n = 1, length(cs)
    in_str = false
    rest(i, k) = i + k - 1 <= n ? String(cs[i:(i + k - 1)]) : ""
    while i <= n
        c = cs[i]
        if in_triple
            if rest(i, 3) == "\"\"\""
                in_triple = false
                i += 3
            else
                i += 1
            end
        elseif in_str
            if c == '\\'
                i += 2
            elseif c == '"'
                in_str = false
                i += 1
            else
                i += 1
            end
        elseif rest(i, 3) == "\"\"\""
            in_triple = true
            i += 3
        elseif c == '"'
            in_str = true
            i += 1
        elseif c == '#'
            break
        else
            write(out, c)
            i += 1
        end
    end
    return String(take!(out)), in_triple
end

# First argument of the call starting at the byte index open_at.
# Char-based: test sources contain Unicode. open_at comes from a regex
# byte match, so convert it to a char index first.
function _first_arg(line::String, open_at::Int)
    cs = collect(line)
    bytepos = 1
    ci = 1
    for (j, c) in enumerate(cs)
        bytepos > open_at && break
        ci = j
        bytepos += ncodeunits(c)
    end
    depth, i, n = 0, ci, length(cs)
    buf = IOBuffer()
    while i <= n
        c = cs[i]
        if c == '('
            depth += 1
            depth > 1 && write(buf, c)
        elseif c == ')'
            depth -= 1
            depth == 0 && break
        elseif c == ',' && depth == 1
            break
        else
            depth >= 1 && write(buf, c)
        end
        i += 1
    end
    return String(take!(buf))
end

function _bare_draws(code::String)
    found = String[]
    for m in eachmatch(_DRAW_CALL, code)
        open_at = m.offset + length(m.match) - 1
        occursin(_RNG_FIRST_ARG, _first_arg(code, open_at)) || push!(found, m.match)
    end
    return found
end

function _lint_file(path::String)
    bare, direct = 0, 0
    in_triple = false
    for raw in eachline(path)
        code, in_triple = _code_part(raw, in_triple)
        isempty(strip(code)) && continue
        startswith(strip(code), "function ") && continue
        isempty(_bare_draws(code)) || (bare += 1)
        if occursin(_EST_CALL, code) && !isempty(_bare_draws(code))
            direct += 1
        end
    end
    return (bare=bare, direct=direct)
end

function _read_allowlist(path::String)
    allowed = Set{String}()
    for raw in eachline(path)
        m = match(r"^\s*-\s+(test/\S+\.jl)\s*$", raw)
        m !== nothing && push!(allowed, m.captures[1])
    end
    return allowed
end

@testset "white-noise lint (DGP-01 #790)" begin
    testdir = dirname(@__DIR__)
    allowed = _read_allowlist(joinpath(testdir, "dgp", "ALLOWLIST.md"))
    @test !isempty(allowed)  # allowlist must exist and be non-empty
    bad = Dict{String,NamedTuple}()
    for (root, _, files) in walkdir(testdir)
        for f in files
            endswith(f, ".jl") || continue
            rel = relpath(joinpath(root, f), testdir)
            # Allowlist entries carry the test/ prefix; relpaths do not.
            (rel in allowed || joinpath("test", rel) in allowed) && continue
            r = _lint_file(joinpath(root, f))
            (r.bare > 0 || r.direct > 0) && (bad[rel] = r)
        end
    end
    if !isempty(bad)
        println("White-noise lint violations (add to test/dgp/ALLOWLIST.md grandfathered section only via a DGP module issue):")
        for k in sort(collect(keys(bad)))
            println("  ", k, " bare=", bad[k].bare, " direct=", bad[k].direct)
        end
    end
    # Rationale: bare draws make data stream-position-dependent and hide
    # white-noise DGPs; every non-allowlisted file must thread an explicit rng.
    @test isempty(bad)
end
