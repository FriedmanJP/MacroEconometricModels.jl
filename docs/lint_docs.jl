#!/usr/bin/env julia
#
# Mechanical docrule linter for MacroEconometricModels.jl documentation.
#
#   julia --project=docs docs/lint_docs.jl            # lint docs/src/**.md
#   julia --project=docs docs/lint_docs.jl file.md    # lint specific pages
#
# Docs are NOT built by CI since 2026-07-10 (Documentation.yml removed), so this
# is a LOCAL pre-deploy gate — a fast, package-load-free companion to
# verify_examples.jl. It scans the markdown corpus for the mechanical rules in
# docs/docrule.md and exits non-zero on any violation, printing each as
# `file:line [RULE-ID] message`.
#
# Checks (docrule references in brackets):
#   [SKELETON]  Method pages carry Quick Start / Complete Example / Common
#               Pitfalls / References; no H4 on narrative pages. Hubs are NOT
#               required to have a Complete Example.            [docrule Page Types]
#   [GROUP]     One @setup/@example group name per page.        [docrule:138 / 228]
#   [PRINTLN]   No `println` for results inside @example blocks. [docrule:189 / 439]
#   [COUNT]     No bare hard-coded plot-dispatch / result-type
#               counts outside an @eval block.                  [docrule Page Types]
#   [IFRAME]    Every iframe src resolves under assets/plots/.   [docrule:414-431]
#   [SAVEPLOT]  save_plot filename matches an iframe src on the
#               same page.                                       [docrule:412]
#   [KWDEFAULT] Keyword-table Default cells match the literal
#               default in the source signature.                [docrule:166 / 326]
#
# Rule-specific opt-outs are documented inline where a page's state is intentional
# (illustrative filenames, hub-child pages that inherit a hub's worked examples).

const SRCDOCS = joinpath(@__DIR__, "src")
const PKGSRC  = normpath(joinpath(@__DIR__, "..", "src"))

struct Violation
    file::String   # path relative to docs/src (e.g. "arima.md" or "api/dsge.md")
    line::Int
    rule::String
    msg::String
end

# ----------------------------------------------------------------- page types
#
# Classification is EXPLICIT (reconciled against the actual corpus, per the v0.6.7
# docs plan) rather than heuristic — the "first child under a section header → Hub"
# guess misclassifies full Method pages. Any page NOT listed below defaults to
# METHOD, so a newly-added page must either carry the full skeleton or be classified
# here on purpose.

const LANDING = Set(["index.md"])

# Section overviews fronting a family of child pages. Hubs are NOT required to have
# a Complete Example (worked examples live on the children).
const HUB = Set(["dsge.md", "tests.md", "innovation_accounting.md",
                 "nongaussian.md", "io.md", "nowcast.md"])

# Auto-generated API/reference pages + the plotting reference: minimal prose, no
# method skeleton. All api/*.md are added dynamically below.
const REFERENCE = Set(["api.md", "plotting.md"])

# Project front-matter / getting-started pages: bespoke structure, no method skeleton.
const PROJECT = Set(["notation.md", "bibliography.md", "changelog.md", "citation.md",
                     "getting_started.md", "method_guide.md"])

# Children of the io.md hub that inherit the hub's worked examples and carry no
# standalone Complete Example — treated as hub children, not standalone methods.
const HUB_CHILDREN = Set(["io_classical.md", "io_environmental.md",
                          "io_baqaee_farhi.md", "io_download.md"])

# save_plot filenames that intentionally differ from the embedded iframe asset
# (illustrative / placeholder names in reference & getting-started material). The
# aligned pages (regression/binary_choice/event_study/x13) are NOT allowlisted, so
# a regression there is still caught.
const SAVEPLOT_ALLOW = Set([
    ("plotting.md", "irf_plot.html"), ("plotting.md", "name.html"),
    ("plotting.md", "path.html"),
    ("filters.md", "hp_filter.html"),
    ("getting_started.md", "irf.html"),
])

function page_type(rel::String)
    startswith(rel, "api/")   && return :reference
    rel in LANDING            && return :landing
    rel in HUB                && return :hub
    rel in REFERENCE          && return :reference
    rel in PROJECT            && return :project
    rel in HUB_CHILDREN       && return :hub_child
    return :method
end

# ----------------------------------------------------------------- block parsing

struct CodeBlock
    kind::Symbol      # :setup | :example | :eval | :static
    name::String      # group name for :setup/:example, "" otherwise
    startline::Int    # line of opening fence (1-based)
    lines::Vector{Tuple{Int,String}}  # (lineno, text) of body
end

function parse_code_blocks(text_lines::Vector{String})
    blocks = CodeBlock[]
    i = 1
    n = length(text_lines)
    while i <= n
        m = match(r"^```@(setup|example)(?:\s+(\S+))?\s*$", text_lines[i])
        ev = match(r"^```@eval\s*$", text_lines[i])
        st = match(r"^```julia\s*$", text_lines[i])
        if m !== nothing
            kind = m.captures[1] == "setup" ? :setup : :example
            name = m.captures[2] === nothing ? "" : String(m.captures[2])
            start = i
            body = Tuple{Int,String}[]
            i += 1
            while i <= n && !occursin(r"^```\s*$", text_lines[i])
                push!(body, (i, text_lines[i])); i += 1
            end
            push!(blocks, CodeBlock(kind, name, start, body))
        elseif ev !== nothing || st !== nothing
            start = i
            body = Tuple{Int,String}[]
            i += 1
            while i <= n && !occursin(r"^```\s*$", text_lines[i])
                push!(body, (i, text_lines[i])); i += 1
            end
            push!(blocks, CodeBlock(ev !== nothing ? :eval : :static, "", start, body))
        end
        i += 1
    end
    return blocks
end

# ----------------------------------------------------------------- checks 1-6

has_h2(lines, title) = any(l -> occursin(Regex("^##\\s+" * title * "\\s*\$", "i"), l), lines)

function check_skeleton!(V, rel, lines)
    t = page_type(rel)
    # No H4 on any narrative page (method/hub); reference/project may use deeper
    # auto-generated structure.
    if t === :method || t === :hub
        for (ln, l) in enumerate(lines)
            if occursin(r"^####\s+\S", l)
                push!(V, Violation(rel, ln, "SKELETON", "H4 heading — flatten to H3 or bold"))
            end
        end
    end
    t === :method || return
    for (title, label) in (("Quick Start", "Quick Start"),
                           ("Complete Example", "Complete Example"),
                           ("Common Pitfalls", "Common Pitfalls"),
                           ("References", "References"))
        if !has_h2(lines, title)
            push!(V, Violation(rel, 1, "SKELETON",
                "Method page missing `## $label` section"))
        end
    end
end

function check_group!(V, rel, blocks)
    names = String[]
    for b in blocks
        (b.kind === :setup || b.kind === :example) && !isempty(b.name) && push!(names, b.name)
    end
    uniq = unique(names)
    if length(uniq) > 1
        push!(V, Violation(rel, 1, "GROUP",
            "multiple @setup/@example group names: " * join(uniq, ", ") *
            " (docrule: one named group per page)"))
    end
end

function check_println!(V, rel, blocks)
    for b in blocks
        b.kind === :example || continue
        for (ln, l) in b.lines
            occursin(r"#\s*lint:\s*ignore", l) && continue
            if occursin(r"\bprintln\s*\(", l)
                push!(V, Violation(rel, ln, "PRINTLN",
                    "println inside @example — use report()/field access (docrule anti-pattern 1)"))
            end
        end
    end
end

# Bare plot-dispatch / result-type counts that should be an @eval-generated figure
# rather than a hardcoded numeral (the recurrently-stale 41/52/53/55/56 class).
const COUNT_RE = r"\b\d+\s+(?:plot[- ]?dispatch|dispatches|result types|plottable|plot_result)\b|\bdispatch(?:es)?\s+on\s+\d+\b"i

function check_counts!(V, rel, lines, blocks)
    evalspans = [(b.startline, isempty(b.lines) ? b.startline : b.lines[end][1] + 1)
                 for b in blocks if b.kind === :eval]
    in_eval(ln) = any(s -> s[1] <= ln <= s[2], evalspans)
    for (ln, l) in enumerate(lines)
        in_eval(ln) && continue
        if occursin(COUNT_RE, l)
            push!(V, Violation(rel, ln, "COUNT",
                "hard-coded plot-dispatch/result-type count — generate via @eval or omit the numeral"))
        end
    end
end

function check_iframes!(V, rel, lines)
    for (ln, l) in enumerate(lines)
        for m in eachmatch(r"src=\"([^\"]*assets/plots/[^\"]+)\"", l)
            path = m.captures[1]
            base = replace(path, r".*assets/plots/" => "")
            if !isfile(joinpath(SRCDOCS, "assets", "plots", base))
                push!(V, Violation(rel, ln, "IFRAME",
                    "iframe src `$base` has no file under docs/src/assets/plots/"))
            end
        end
    end
end

function check_saveplot!(V, rel, lines)
    iframes = Set{String}()
    for l in lines, m in eachmatch(r"src=\"[^\"]*assets/plots/([^\"]+)\"", l)
        push!(iframes, m.captures[1])
    end
    isempty(iframes) && return  # nothing to reconcile against
    for (ln, l) in enumerate(lines)
        for m in eachmatch(r"save_plot\([^,]+,\s*\"([^\"]+\.html)\"", l)
            fname = m.captures[1]
            (rel, fname) in SAVEPLOT_ALLOW && continue
            if !(fname in iframes)
                push!(V, Violation(rel, ln, "SAVEPLOT",
                    "save_plot(\"$fname\") has no matching iframe src on this page"))
            end
        end
    end
end

# ----------------------------------------------------------------- check 7

# Harvest TYPED literal keyword defaults from the package source: a keyword arg
# written `name::Type = <literal>` where <literal> is a number, :symbol, Bool,
# nothing, or a string. Untyped kwargs and expression-valued defaults are skipped
# (not literals — nothing to compare). Maps kw name -> set of normalized literals.
const LIT_RE = r"\b([A-Za-z_][A-Za-z0-9_]*)\s*::\s*[A-Za-z0-9_.{}<:, \}]+?\s*=\s*(:[A-Za-z_][A-Za-z0-9_]*|true|false|nothing|-?\d+\.?\d*(?:[eE]-?\d+)?|\"[^\"]*\")"

# A "simple literal" is a bare Symbol, Bool, nothing, integer/float, or a quoted
# string — the only forms that can be compared like-for-like. Tuples, ranges,
# expressions, and descriptive words ("Other", "varies") are NOT literals.
function is_simple_literal(s::AbstractString)
    s = strip(s)
    (startswith(s, ":") && occursin(r"^:[A-Za-z_][A-Za-z0-9_]*$", s)) && return true
    (s in ("true", "false", "nothing")) && return true
    (startswith(s, "\"") && endswith(s, "\"")) && return true
    tryparse(Int, s) !== nothing && return true
    tryparse(Float64, s) !== nothing && return true
    return false
end

function normalize_literal(s::AbstractString)
    s = strip(s)
    if startswith(s, ":") || s == "true" || s == "false" || s == "nothing" ||
       (startswith(s, "\"") && endswith(s, "\""))
        return s
    end
    v = tryparse(Int, s)
    v !== nothing && return string(v)
    f = tryparse(Float64, s)
    f !== nothing && return string(f)
    return s
end

function harvest_src_defaults()
    d = Dict{String,Set{String}}()
    for (root, _, files) in walkdir(PKGSRC), f in files
        endswith(f, ".jl") || continue
        text = read(joinpath(root, f), String)
        for m in eachmatch(LIT_RE, text)
            kw = m.captures[1]
            get!(d, kw, Set{String}())
            push!(d[kw], normalize_literal(m.captures[2]))
        end
    end
    return d
end

# Doc keyword-table row: | `kw` | `Type` | `default` | Description |
const ROW_RE = r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|[^|]*\|\s*`([^`|]+)`\s*\|"

function check_kwdefaults!(V, rel, lines, srcdefaults)
    for (ln, l) in enumerate(lines)
        m = match(ROW_RE, l)
        m === nothing && continue
        kw = m.captures[1]
        # only compare when the doc cell is itself a simple literal (rejects math /
        # power-of-ten, tuples, ranges, and descriptive words like "Other")
        is_simple_literal(m.captures[2]) || continue
        docdef = normalize_literal(m.captures[2])
        haskey(srcdefaults, kw) || continue
        srcset = srcdefaults[kw]
        # ambiguous when the kw carries multiple distinct literal defaults across
        # the source — can't attribute a single truth, so skip.
        length(srcset) == 1 || continue
        srcdef = first(srcset)
        # only flag when both sides are the SAME literal category (avoid comparing
        # a Symbol doc-cell against a numeric src default, etc.)
        samecat = (startswith(docdef, ":") == startswith(srcdef, ":")) &&
                  ((docdef in ("true", "false", "nothing")) == (srcdef in ("true", "false", "nothing")))
        samecat || continue
        if docdef != srcdef
            push!(V, Violation(rel, ln, "KWDEFAULT",
                "`$kw` Default `$(m.captures[2])` ≠ source default `$srcdef`"))
        end
    end
end

# ----------------------------------------------------------------- driver

function lint_page(rel::String, srcdefaults)
    V = Violation[]
    lines = readlines(joinpath(SRCDOCS, rel))
    blocks = parse_code_blocks(lines)
    check_skeleton!(V, rel, lines)
    check_group!(V, rel, blocks)
    check_println!(V, rel, blocks)
    check_counts!(V, rel, lines, blocks)
    check_iframes!(V, rel, lines)
    check_saveplot!(V, rel, lines)
    check_kwdefaults!(V, rel, lines, srcdefaults)
    return V
end

function collect_pages()
    pages = String[]
    for (root, _, files) in walkdir(SRCDOCS), f in files
        endswith(f, ".md") || continue
        push!(pages, relpath(joinpath(root, f), SRCDOCS))
    end
    return sort(pages)
end

function main(args)
    pages = isempty(args) ? collect_pages() :
            [startswith(a, "docs/src/") ? a[length("docs/src/")+1:end] :
             (startswith(a, SRCDOCS) ? relpath(a, SRCDOCS) : a) for a in args]
    srcdefaults = harvest_src_defaults()
    allV = Violation[]
    for p in pages
        append!(allV, lint_page(p, srcdefaults))
    end
    sort!(allV, by = v -> (v.file, v.line, v.rule))
    for v in allV
        println("docs/src/$(v.file):$(v.line) [$(v.rule)] $(v.msg)")
    end
    println("\nlint_docs: $(length(pages)) pages, $(length(allV)) violation(s)")
    exit(isempty(allV) ? 0 : 1)
end

main(ARGS)
