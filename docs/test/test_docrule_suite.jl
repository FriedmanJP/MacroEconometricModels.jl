#!/usr/bin/env julia
#
# MacroEconometricModels.jl - Documentation Test Suite (Tiers 1-4)
#
# Independent E2E test runner validating all mechanical, structural,
# mathematical, navigation, API surface, and execution contracts from
# docs/docrule.md and PROJECT.md.
#
# Usage:
#   julia --project=docs docs/test/test_docrule_suite.jl           # Tiers 1-3
#   julia --project=docs docs/test/test_docrule_suite.jl --tier 1  # Tier 1 only
#   julia --project=docs docs/test/test_docrule_suite.jl --tier 2  # Tier 2 only
#   julia --project=docs docs/test/test_docrule_suite.jl --tier 3  # Tier 3 only
#   julia --project=docs docs/test/test_docrule_suite.jl --tier 4  # Tier 4 only
#   julia --project=docs docs/test/test_docrule_suite.jl --all     # All Tiers (1-4)
#   julia --project=docs docs/test/test_docrule_suite.jl --audit   # Audit report

using MacroEconometricModels
using Test

const DOCS_DIR = normpath(joinpath(@__DIR__, ".."))
const SRCDOCS = joinpath(DOCS_DIR, "src")
const MAKE_JL = joinpath(DOCS_DIR, "make.jl")
const BANNER = "="^80
const ANCHOR_REGEX = Regex(
    "^#\\s+.*(?:\\(@id\\s+([A-Za-z0-9_\\-\\.]+)\\)|" *
    "\\[.*\\]\\(@id\\s+([A-Za-z0-9_\\-\\.]+)\\))",
)

# -----------------------------------------------------------------------------
# Page Classification & Constants
# -----------------------------------------------------------------------------

const LANDING_PAGES = Set(["index.md"])

const HUB_PAGES = Set([
    "dsge.md",
    "tests.md",
    "innovation_accounting.md",
    "nongaussian.md",
    "io.md",
    "nowcast.md",
    "counterfactual.md",
    "dsge_heterogeneity.md",
])

const REFERENCE_HUB_PAGES = Set(["api.md"])

const REFERENCE_PAGES = Set(["plotting.md"])

const PROJECT_PAGES = Set([
    "notation.md",
    "bibliography.md",
    "changelog.md",
    "citation.md",
    "getting_started.md",
    "method_guide.md",
])

function classify_page(rel::String)
    startswith(rel, "api/") && return :reference
    rel in LANDING_PAGES && return :landing
    rel in HUB_PAGES && return :hub
    rel in REFERENCE_HUB_PAGES && return :reference_hub
    rel in REFERENCE_PAGES && return :reference
    rel in PROJECT_PAGES && return :project
    return :method
end

# -----------------------------------------------------------------------------
# AST and Filesystem Parsers
# -----------------------------------------------------------------------------

function collect_disk_pages(srcdir::String)
    pages = String[]
    for (root, _, files) in walkdir(srcdir)
        for f in files
            endswith(f, ".md") || continue
            push!(pages, relpath(joinpath(root, f), srcdir))
        end
    end
    return sort(pages)
end

function extract_pages_ast(make_file::String)
    content = read(make_file, String)
    ast = Meta.parseall(content)
    function find_pages(ex)
        if ex isa Expr
            if (ex.head === :kw || ex.head === :(=)) && ex.args[1] === :pages
                return ex.args[2]
            end
            for a in ex.args
                res = find_pages(a)
                res !== nothing && return res
            end
        end
        return nothing
    end
    pages_ex = find_pages(ast)
    pages_ex === nothing && error("Could not locate `pages` in $(make_file)")
    return Core.eval(Module(), pages_ex)
end

function extract_registered_files(pages_tree)
    files = String[]
    function traverse(item)
        if item isa Pair
            _, v = item
            if v isa String
                endswith(v, ".md") && push!(files, v)
            elseif v isa Vector
                for sub in v
                    traverse(sub)
                end
            end
        elseif item isa Vector
            for sub in item
                traverse(sub)
            end
        end
        return nothing
    end
    traverse(pages_tree)
    return sort(unique(files))
end

function extract_hub_children(pages_tree)
    hub_children = Dict{String, Vector{String}}()
    function traverse(item, current_hub)
        if item isa Pair
            _, v = item
            if v isa String
                if endswith(v, ".md") && current_hub !== nothing && v != current_hub
                    push!(get!(hub_children, current_hub, String[]), v)
                end
            elseif v isa Vector
                local_hub = current_hub
                for child in v
                    if child isa Pair && child.first == "Overview" &&
                            child.second isa String
                        local_hub = child.second
                        hub_children[local_hub] = String[]
                        break
                    end
                end
                for child in v
                    if child isa Pair && child.first == "Overview" &&
                            child.second == local_hub
                        continue
                    end
                    traverse(child, local_hub)
                end
            end
        elseif item isa Vector
            for sub in item
                traverse(sub, current_hub)
            end
        end
        return nothing
    end
    traverse(pages_tree, nothing)
    return hub_children
end

# -----------------------------------------------------------------------------
# Tier 1: Structural Invariants & Registry Parity
# -----------------------------------------------------------------------------

function run_tier1_tests(audit_mode::Bool=false)
    @testset "Tier 1: Structural Invariants & Registry Parity" begin
        disk_pages = collect_disk_pages(SRCDOCS)
        pages_tree = extract_pages_ast(MAKE_JL)
        make_pages = extract_registered_files(pages_tree)

        @testset "1.1 File Registry Parity (make.jl ↔ docs/src)" begin
            missing_on_disk = setdiff(make_pages, disk_pages)
            omitted_from_make = setdiff(disk_pages, make_pages)

            if !isempty(missing_on_disk)
                println("  [T1.1 DEFECT] In make.jl, missing on disk: ", missing_on_disk)
            end
            if !isempty(omitted_from_make)
                println("  [T1.1 DEFECT] On disk, omitted from make: ", omitted_from_make)
            end

            @test isempty(missing_on_disk)
            @test isempty(omitted_from_make)
            @test length(disk_pages) == length(make_pages)
        end

        @testset "1.2 Single H1 & (@id ...) Anchor Integrity" begin
            h1_re = r"^#\s+(.+)$"
            anchor_re = r"\(@id\s+([A-Za-z0-9_\-\.]+)\)"

            missing_h1 = Tuple{String, Int}[]
            multiple_h1 = Tuple{String, Vector{Int}}[]
            missing_anchor = Tuple{String, Int, String}[]

            for rel in disk_pages
                lines = readlines(joinpath(SRCDOCS, rel))
                in_fence = false
                h1_lines = Int[]
                for (idx, line) in enumerate(lines)
                    trimmed = strip(line)
                    if startswith(trimmed, "```")
                        in_fence = !in_fence
                        continue
                    end
                    in_fence && continue
                    if occursin(h1_re, line)
                        push!(h1_lines, idx)
                    end
                end

                if isempty(h1_lines)
                    push!(missing_h1, (rel, 0))
                elseif length(h1_lines) > 1
                    push!(multiple_h1, (rel, h1_lines))
                else
                    idx = h1_lines[1]
                    h1_text = lines[idx]
                    if match(anchor_re, h1_text) === nothing
                        push!(missing_anchor, (rel, idx, h1_text))
                    end
                end
            end

            if !isempty(missing_h1)
                println("  [T1.2 DEFECT] Pages missing H1 title: ", missing_h1)
            end
            if !isempty(multiple_h1)
                println("  [T1.2 DEFECT] Pages with multiple H1 titles: ", multiple_h1)
            end
            if !isempty(missing_anchor)
                println("  [T1.2 DEFECT] Pages missing (@id ...) on H1:")
                for (f, idx, txt) in missing_anchor
                    println("    - docs/src/$(f):$(idx) -> $(txt)")
                end
            end

            @test isempty(missing_h1)
            @test isempty(multiple_h1)
            @test isempty(missing_anchor)
        end

        @testset "1.3 Heading Depth (0 H4+ Headings)" begin
            h4_violations = Tuple{String, Int, String}[]
            for rel in disk_pages
                lines = readlines(joinpath(SRCDOCS, rel))
                in_fence = false
                for (idx, line) in enumerate(lines)
                    trimmed = strip(line)
                    if startswith(trimmed, "```")
                        in_fence = !in_fence
                        continue
                    end
                    in_fence && continue
                    if occursin(r"^####+\s+\S", line)
                        push!(h4_violations, (rel, idx, line))
                    end
                end
            end

            if !isempty(h4_violations)
                println("  [T1.3 DEFECT] H4+ headings detected (docrule bans H4+):")
                for (f, idx, line) in h4_violations
                    println("    - docs/src/$(f):$(idx) -> $(line)")
                end
            end

            @test isempty(h4_violations)
        end

        @testset "1.4 HTML Directive Safety & Fenced Plot Embeds" begin
            bare_raw_html = Tuple{String, Int, String}[]
            missing_iframes = Tuple{String, Int, String}[]

            for rel in disk_pages
                lines = readlines(joinpath(SRCDOCS, rel))
                in_fence = false
                in_raw = false
                for (idx, line) in enumerate(lines)
                    trimmed = strip(line)
                    if startswith(trimmed, "```")
                        if occursin(r"^```@raw\s+html\b", trimmed)
                            in_raw = true
                        elseif in_raw
                            in_raw = false
                        end
                        in_fence = !in_fence
                        continue
                    end
                    if !in_fence && occursin(r"^@raw\s+html\b", trimmed)
                        push!(bare_raw_html, (rel, idx, line))
                    end
                    if in_raw
                        for m in eachmatch(r"src=\"([^\"]*assets/plots/[^\"]+)\"", line)
                            path = m.captures[1]
                            base = replace(path, r".*assets/plots/" => "")
                            target = joinpath(SRCDOCS, "assets", "plots", base)
                            if !isfile(target)
                                push!(missing_iframes, (rel, idx, base))
                            end
                        end
                    end
                end
            end

            if !isempty(bare_raw_html)
                println("  [T1.4 DEFECT] Bare @raw html directives found: ", bare_raw_html)
            end
            if !isempty(missing_iframes)
                println("  [T1.4 DEFECT] Missing plot iframe assets: ", missing_iframes)
            end

            @test isempty(bare_raw_html)
            @test isempty(missing_iframes)
        end

        @testset "1.5 Terminal References on Method and Hub Pages" begin
            missing_refs = Tuple{String, Symbol}[]
            non_terminal_refs = Tuple{String, Int, String}[]

            for rel in disk_pages
                t = classify_page(rel)
                t in (:method, :hub) || continue
                lines = readlines(joinpath(SRCDOCS, rel))
                ref_idx = 0
                for (idx, line) in enumerate(lines)
                    if occursin(r"^##\s+References\s*$", line)
                        ref_idx = idx
                    elseif ref_idx > 0 && occursin(r"^##\s+\S", line)
                        push!(non_terminal_refs, (rel, idx, line))
                    end
                end
                if ref_idx == 0
                    push!(missing_refs, (rel, t))
                end
            end

            if !isempty(missing_refs)
                println("  [T1.5 DEFECT] Missing `## References`: ", missing_refs)
            end
            if !isempty(non_terminal_refs)
                println("  [T1.5 DEFECT] Non-terminal `## References`: ", non_terminal_refs)
            end

            @test isempty(missing_refs)
            @test isempty(non_terminal_refs)
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Tier 2: Mathematical & Code Block Rigor
# -----------------------------------------------------------------------------

function run_tier2_tests(audit_mode::Bool=false)
    @testset "Tier 2: Mathematical & Code Block Rigor" begin
        disk_pages = collect_disk_pages(SRCDOCS)

        @testset "2.1 Display Math & Symbol Definitions (where: lists)" begin
            missing_where = Tuple{String, Int, String}[]
            total_math = 0

            for rel in disk_pages
                t = classify_page(rel)
                t in (:method, :hub) || continue
                lines = readlines(joinpath(SRCDOCS, rel))
                n = length(lines)
                i = 1
                while i <= n
                    if strip(lines[i]) == "```math"
                        total_math += 1
                        math_start = i
                        i += 1
                        while i <= n && strip(lines[i]) != "```"
                            i += 1
                        end
                        j = i + 1
                        while j <= n && isempty(strip(lines[j]))
                            j += 1
                        end
                        # Allow chained display math blocks
                        if j <= n && strip(lines[j]) == "```math"
                            i += 1
                            continue
                        end
                        if j <= n
                            follower = strip(lines[j])
                            is_where = occursin(
                                r"^(?:where:?|\*\*where\*\*:?|\*where\*:?|Where:?)\b",
                                follower,
                            )
                            if !is_where
                                push!(missing_where, (rel, math_start, follower))
                            end
                        else
                            push!(missing_where, (rel, math_start, "<EOF>"))
                        end
                    end
                    i += 1
                end
            end

            if !isempty(missing_where)
                affected_pages = length(unique([m[1] for m in missing_where]))
                println("  [T2.1 DEFECT] Display math missing `where:` lists: " *
                        "$(length(missing_where)) blocks across $(affected_pages) pages")
                sample_count = min(8, length(missing_where))
                for (f, line, foll) in missing_where[1:sample_count]
                    println("    - docs/src/$(f):$(line) -> \"$(foll)\"")
                end
                if length(missing_where) > 8
                    rem = length(missing_where) - 8
                    println("    ... and $(rem) more blocks.")
                end
            end

            @test total_math > 0
            @test isempty(missing_where)
        end

        @testset "2.2 Placeholder Elimination (TODO, bare ...)" begin
            placeholders = Tuple{String, Int, String}[]

            for rel in disk_pages
                lines = readlines(joinpath(SRCDOCS, rel))
                in_block = false
                for (idx, line) in enumerate(lines)
                    trimmed = strip(line)
                    if startswith(trimmed, "```")
                        in_block = !in_block
                        continue
                    end
                    if in_block
                        if occursin(r"\b(?:TODO|FIXME|XXX)\b", line) ||
                                occursin(r"^\s*\.\.\.\s*$", line)
                            push!(placeholders, (rel, idx, line))
                        end
                    end
                end
            end

            if !isempty(placeholders)
                println("  [T2.2 DEFECT] Placeholders in code/math blocks: ", placeholders)
            end

            @test isempty(placeholders)
        end

        @testset "2.3 Single Named Sandbox Group per Page" begin
            multigroups = Tuple{String, Vector{String}}[]

            for rel in disk_pages
                lines = readlines(joinpath(SRCDOCS, rel))
                groups = String[]
                for line in lines
                    m = match(r"^```@(setup|example)(?:\s+(\S+))?\s*$", line)
                    if m !== nothing
                        grp = m.captures[2] === nothing ? "<anon>" : String(m.captures[2])
                        push!(groups, grp)
                    end
                end
                ug = unique(groups)
                if length(ug) > 1
                    push!(multigroups, (rel, ug))
                end
            end

            if !isempty(multigroups)
                println("  [T2.3 DEFECT] Pages with multiple sandbox group names:")
                for (f, ug) in multigroups
                    println("    - docs/src/$(f): groups = $(join(ug, ", "))")
                end
            end

            @test isempty(multigroups)
        end

        @testset "2.4 Model Summary & Output Integrity (no println for results)" begin
            println_in_example = Tuple{String, Int, String}[]
            report_in_static = Tuple{String, Int, String}[]

            for rel in disk_pages
                lines = readlines(joinpath(SRCDOCS, rel))
                in_example = false
                in_static = false
                for (idx, line) in enumerate(lines)
                    trimmed = strip(line)
                    if startswith(trimmed, "```")
                        if occursin(r"^```@example\b", trimmed)
                            in_example = true
                        elseif trimmed == "```julia"
                            in_static = true
                        else
                            in_example = false
                            in_static = false
                        end
                        continue
                    end
                    if in_example
                        if occursin(r"\bprintln\s*\(", line) &&
                                !occursin(r"#\s*lint:\s*ignore", line)
                            push!(println_in_example, (rel, idx, line))
                        end
                    end
                    if in_static
                        if occursin(r"\breport\s*\(", line)
                            push!(report_in_static, (rel, idx, line))
                        end
                    end
                end
            end

            if !isempty(println_in_example)
                println("  [T2.4 DEFECT] println in @example: ", println_in_example)
            end
            if !isempty(report_in_static)
                println("  [T2.4 DEFECT] report() in static julia (must be @example):")
                for (f, idx, l) in report_in_static
                    println("    - docs/src/$(f):$(idx) -> $(l)")
                end
            end

            @test isempty(println_in_example)
            @test isempty(report_in_static)
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Tier 3: Navigation & API Surface Integrity
# -----------------------------------------------------------------------------

function run_tier3_tests(audit_mode::Bool=false)
    @testset "Tier 3: Navigation & API Surface Integrity" begin
        pages_tree = extract_pages_ast(MAKE_JL)
        hub_children = extract_hub_children(pages_tree)
        disk_pages = collect_disk_pages(SRCDOCS)

        @testset "3.1 Sidebar & Hub Children Limit (<= 6 direct children)" begin
            oversized_sections = Tuple{String, Int}[]
            for item in pages_tree
                if item isa Pair && item.second isa Vector
                    title = item.first
                    kids = item.second
                    if length(kids) > 6
                        push!(oversized_sections, (title, length(kids)))
                    end
                end
            end

            if !isempty(oversized_sections)
                println("  [T3.1 DEFECT] Sidebar sections with > 6 children:")
                for (title, count) in oversized_sections
                    println("    - \"$(title)\": $(count) children")
                end
            end

            @test isempty(oversized_sections)
        end

        @testset "3.2 Hub Complete Child Linkage (## Child Pages)" begin
            child_anchors = Dict{String, String}()
            for rel in disk_pages
                lines = readlines(joinpath(SRCDOCS, rel))
                for l in lines
                    m = match(ANCHOR_REGEX, l)
                    if m !== nothing
                        anc = m.captures[1] !== nothing ? m.captures[1] : m.captures[2]
                        child_anchors[rel] = anc
                        break
                    end
                end
            end

            missing_hub_links = Tuple{String, String}[]
            for (hub, kids) in hub_children
                hub == "api.md" && continue  # api.md is a reference hub
                hub_content = read(joinpath(SRCDOCS, hub), String)
                sec_start = findfirst("## Child Pages", hub_content)
                if sec_start === nothing
                    push!(missing_hub_links, (hub, "<missing ## Child Pages section>"))
                    continue
                end
                rest = hub_content[sec_start[1]:end]
                next_h2 = findnext("\n## ", rest, 4)
                sec_content = next_h2 === nothing ? rest : rest[1:next_h2[1]]

                for kid in kids
                    anc = get(child_anchors, kid, "")
                    has_link = false
                    if !isempty(anc) && occursin(anc, sec_content)
                        has_link = true
                    elseif occursin(kid, sec_content)
                        has_link = true
                    end
                    if !has_link
                        push!(missing_hub_links, (hub, kid))
                    end
                end
            end

            if !isempty(missing_hub_links)
                println("  [T3.2 DEFECT] Hub missing child links in `## Child Pages`:")
                for (hub, kid) in missing_hub_links
                    println("    - docs/src/$(hub) missing child link to $(kid)")
                end
            end

            @test isempty(missing_hub_links)
        end

        @testset "3.3 Child Back-Links to Parent Hub" begin
            hub_anchors = Dict{String, String}()
            for hub in keys(hub_children)
                lines = readlines(joinpath(SRCDOCS, hub))
                for l in lines
                    m = match(ANCHOR_REGEX, l)
                    if m !== nothing
                        anc = m.captures[1] !== nothing ? m.captures[1] : m.captures[2]
                        hub_anchors[hub] = anc
                        break
                    end
                end
            end

            missing_backlinks = Tuple{String, String}[]
            for (hub, kids) in hub_children
                hub == "api.md" && continue
                anc = get(hub_anchors, hub, "")
                for kid in kids
                    content = read(joinpath(SRCDOCS, kid), String)
                    h2_pos = findfirst("##", content)
                    intro = h2_pos === nothing ? content : content[1:(h2_pos[1] - 1)]
                    has_link = false
                    if !isempty(anc) && occursin("@ref " * anc, intro)
                        has_link = true
                    elseif occursin(hub, intro) || (!isempty(anc) && occursin(anc, intro))
                        has_link = true
                    end
                    if !has_link
                        push!(missing_backlinks, (hub, kid))
                    end
                end
            end

            if !isempty(missing_backlinks)
                println("  [T3.3 DEFECT] Child pages missing hub back-link: ",
                        missing_backlinks)
            end

            @test isempty(missing_backlinks)
        end

        @testset "3.4 Exported API Coverage & Docstring Parity" begin
            all_exports = setdiff(
                names(MacroEconometricModels),
                [:MacroEconometricModels],
            )

            missing_docstrings = Symbol[]
            for s in all_exports
                b = Docs.Binding(MacroEconometricModels, s)
                d = Docs.doc(b)
                sprint_d = sprint(show, "text/plain", d)
                if occursin("No documentation found", sprint_d)
                    push!(missing_docstrings, s)
                end
            end

            if !isempty(missing_docstrings)
                println("  [T3.4 DEFECT] Exported symbols missing docstrings: ",
                        missing_docstrings)
            end

            @test isempty(missing_docstrings)

            narrative_docs_pages = Tuple{String, Int, Int}[]
            for rel in disk_pages
                (startswith(rel, "api/") || rel == "api.md") && continue
                lines = readlines(joinpath(SRCDOCS, rel))
                in_docs = false
                docs_start = 0
                sym_count = 0
                for (idx, line) in enumerate(lines)
                    trimmed = strip(line)
                    if occursin(r"^```@docs\b", trimmed)
                        in_docs = true
                        docs_start = idx
                        sym_count = 0
                    elseif in_docs && occursin(r"^```\s*$", trimmed)
                        in_docs = false
                        push!(narrative_docs_pages, (rel, docs_start, sym_count))
                    elseif in_docs && !isempty(trimmed)
                        sym_count += 1
                    end
                end
            end

            if !isempty(narrative_docs_pages)
                println("  [T3.4 DEFECT] Narrative pages with canonical @docs blocks:")
                for (f, line, cnt) in narrative_docs_pages
                    println("    - docs/src/$(f):$(line) ($(cnt) symbols)")
                end
            end

            @test isempty(narrative_docs_pages)

            api_files = filter(f -> startswith(f, "api/"), disk_pages)
            doc_entries = Dict{String, Vector{Tuple{String, Int}}}()
            intra_dups = Tuple{String, Int, String}[]

            for rel in api_files
                lines = readlines(joinpath(SRCDOCS, rel))
                in_docs = false
                seen_in_block = Set{String}()
                for (idx, line) in enumerate(lines)
                    trimmed = strip(line)
                    if occursin(r"^```@docs\b", trimmed)
                        in_docs = true
                        empty!(seen_in_block)
                        continue
                    elseif in_docs && occursin(r"^```\s*$", trimmed)
                        in_docs = false
                        continue
                    end
                    if in_docs && !isempty(trimmed)
                        sym_clean = replace(trimmed, r"\(\)$" => "")
                        if sym_clean in seen_in_block
                            push!(intra_dups, (rel, idx, trimmed))
                        else
                            push!(seen_in_block, sym_clean)
                        end
                        locs = get!(doc_entries, sym_clean, Tuple{String, Int}[])
                        push!(locs, (rel, idx))
                    end
                end
            end

            cross_dups = filter(p -> length(p.second) > 1, doc_entries)
            if !isempty(intra_dups)
                println("  [T3.4 DEFECT] Intra-block duplicate @docs entries: ", intra_dups)
            end
            if !isempty(cross_dups)
                println("  [T3.4 DEFECT] Cross-page duplicate @docs entries:")
                for (sym, locs) in cross_dups
                    println("    - $(sym) -> ", locs)
                end
            end

            @test isempty(intra_dups)
            @test isempty(cross_dups)
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Tier 4: Executable Examples & Full Documenter Build Harness
# -----------------------------------------------------------------------------

function run_tier4_tests()
    @testset "Tier 4: Executable Examples & Full Documenter Build" begin
        @testset "4.1 verify_examples.jl Execution Harness" begin
            verifier = joinpath(DOCS_DIR, "verify_examples.jl")
            cmd = `$(Base.julia_cmd()) --project=$(DOCS_DIR) $(verifier) --all`
            println("  Running: ", cmd)
            proc = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=true)
            @test proc.exitcode == 0
        end

        @testset "4.2 Full Documenter Build (make.jl with DOCS_DRAFT=false)" begin
            cmd = addenv(
                `$(Base.julia_cmd()) --project=$(DOCS_DIR) $(MAKE_JL)`,
                "DOCS_DRAFT" => "false",
            )
            println("  Running: ", cmd)
            proc = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=true)
            @test proc.exitcode == 0
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

function main(args)
    tier1 = false
    tier2 = false
    tier3 = false
    tier4 = false
    audit_mode = false

    if isempty(args)
        tier1 = true
        tier2 = true
        tier3 = true
    else
        i = 1
        while i <= length(args)
            a = args[i]
            if a == "--tier" && i < length(args)
                t = args[i + 1]
                t == "1" && (tier1 = true)
                t == "2" && (tier2 = true)
                t == "3" && (tier3 = true)
                t == "4" && (tier4 = true)
                i += 1
            elseif a == "--tier=1"
                tier1 = true
            elseif a == "--tier=2"
                tier2 = true
            elseif a == "--tier=3"
                tier3 = true
            elseif a == "--tier=4"
                tier4 = true
            elseif a in ("--all", "--all-tiers")
                tier1 = true
                tier2 = true
                tier3 = true
                tier4 = true
            elseif a == "--audit"
                audit_mode = true
                tier1 = true
                tier2 = true
                tier3 = true
            end
            i += 1
        end
    end

    tier_names = String[]
    tier1 && push!(tier_names, "Tier 1 (Structural)")
    tier2 && push!(tier_names, "Tier 2 (Math/Code)")
    tier3 && push!(tier_names, "Tier 3 (Nav/API)")
    tier4 && push!(tier_names, "Tier 4 (Build/Exec)")

    println(BANNER)
    println("MacroEconometricModels.jl - Documentation Test Suite")
    println("Selected Tiers: ", join(tier_names, ", "))
    println("Mode: ", audit_mode ? "AUDIT (Report Only)" : "STRICT (Assert Failures)")
    println(BANNER)

    tier_results = Dict{String, Bool}()

    if tier1
        try
            run_tier1_tests(audit_mode)
            tier_results["Tier 1"] = true
        catch e
            tier_results["Tier 1"] = false
            !audit_mode && println("  Tier 1 encountered test failures.")
        end
    end

    if tier2
        try
            run_tier2_tests(audit_mode)
            tier_results["Tier 2"] = true
        catch e
            tier_results["Tier 2"] = false
            !audit_mode && println("  Tier 2 encountered test failures.")
        end
    end

    if tier3
        try
            run_tier3_tests(audit_mode)
            tier_results["Tier 3"] = true
        catch e
            tier_results["Tier 3"] = false
            !audit_mode && println("  Tier 3 encountered test failures.")
        end
    end

    if tier4
        try
            run_tier4_tests()
            tier_results["Tier 4"] = true
        catch e
            tier_results["Tier 4"] = false
            !audit_mode && println("  Tier 4 encountered test failures.")
        end
    end

    all_passed = all(values(tier_results))
    println("\n" * BANNER)
    println("Test Execution Summary:")
    for (tn, ok) in sort(collect(tier_results))
        println("  $(tn): ", ok ? "PASS ✓" : "FAIL ✗")
    end
    println(BANNER)

    if all_passed
        println("\n✓ All selected documentation tests PASSED.")
        exit(0)
    else
        if audit_mode
            println("\n[AUDIT MODE] Completed with identified defects. (Exit code 0)")
            exit(0)
        else
            println("\n✗ Test suite failed. Fix identified defects above.")
            exit(1)
        end
    end
    return nothing
end

main(ARGS)
