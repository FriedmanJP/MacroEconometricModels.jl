#!/usr/bin/env julia
#
# Verify that the executable code blocks in documentation .md files run without
# error, matching Documenter's semantics as closely as a standalone tool can.
#
# What it checks (per Documenter):
#   * @setup NAME / @example NAME blocks are EXECUTED. Blocks that share a NAME
#     share one sandbox module and run in document order; anonymous @example
#     blocks each get a fresh module (Documenter's per-group sandboxing).
#   * `# hide` lines are EXECUTED (Documenter hides them from display only, it
#     does NOT skip them) — we strip the marker, run the line, and suppress its
#     output, so program semantics match the rendered build.
#   * Each @example block's final value is pushed through
#     `show(devnull, MIME"text/plain"(), value)` so the display path Documenter
#     uses (and which can throw even when the REPL succeeds) is exercised.
#   * Static ```julia blocks are NOT executed (Documenter renders them verbatim,
#     e.g. plot_result() recipes) — they are listed and skipped.
#
# Reporting is block-by-block: `page.md [group] block N (lines a-b)` with elapsed
# time; failures name the block, its group, its source line range, and the error.
# A per-block timeout (default 120 s, override with DOCS_VERIFY_TIMEOUT) aborts a
# hanging block and attributes it. Note: on a single Julia thread the timeout can
# only fire once a CPU-bound block yields; run with JULIA_NUM_THREADS>=2 (or rely
# on the CI job timeout) for a hard backstop against non-yielding hangs.
#
# Usage:
#   julia --project=docs docs/verify_examples.jl docs/src/manual.md
#   julia --project=docs docs/verify_examples.jl docs/src/manual.md docs/src/bayesian.md
#   julia --project=docs docs/verify_examples.jl --all
#

const BLOCK_TIMEOUT = parse(Float64, get(ENV, "DOCS_VERIFY_TIMEOUT", "120"))

struct DocBlock
    kind::Symbol        # :setup | :example | :static
    group::String       # sandbox name (":static" blocks carry a synthetic group)
    code::String        # block body with `# hide` markers stripped, ready to eval
    startline::Int      # 1-based line of the opening fence
    endline::Int        # 1-based line of the closing fence
end

# Strip a trailing `# hide` marker so the line still EXECUTES (Documenter hides
# output, not execution). Leaves everything else untouched.
_strip_hide(line::AbstractString) = replace(line, r"\s*#\s*hide\s*$" => "")

"""Parse a markdown page into its executable/static code blocks, preserving order."""
function parse_blocks(mdfile::String)
    blocks = DocBlock[]
    lines = readlines(mdfile)
    i = 1
    anon = 0
    while i <= length(lines)
        m = match(r"^```@(setup|example)(?:\s+(\S+))?\s*$", lines[i])
        s = match(r"^```julia\s*$", lines[i])
        if m !== nothing
            kind = m.captures[1] == "setup" ? :setup : :example
            name = m.captures[2]
            if name === nothing
                anon += 1
                group = "__anon_$(anon)"
            else
                group = String(name)
            end
            startline = i
            body = String[]
            i += 1
            while i <= length(lines) && !occursin(r"^```\s*$", lines[i])
                push!(body, _strip_hide(lines[i]))
                i += 1
            end
            push!(blocks, DocBlock(kind, group, join(body, "\n"), startline, i))
        elseif s !== nothing
            startline = i
            body = String[]
            i += 1
            while i <= length(lines) && !occursin(r"^```\s*$", lines[i])
                push!(body, lines[i])
                i += 1
            end
            push!(blocks, DocBlock(:static, "__static", join(body, "\n"), startline, i))
        end
        i += 1
    end
    return blocks
end

# Evaluate `expr` in `mod` with a wall-clock timeout. Returns
# (:ok, value) | (:error, (exception, backtrace)) | (:timeout, nothing).
function eval_with_timeout(mod::Module, expr, timeout::Float64)
    value = Ref{Any}(nothing)
    err = Ref{Any}(nothing)
    done = Ref(false)
    t = Threads.@spawn begin
        try
            value[] = Core.eval(mod, expr)
        catch e
            err[] = (e, catch_backtrace())
        finally
            done[] = true
        end
    end
    status = timedwait(() -> done[], timeout; pollint=0.05)
    if status === :timed_out
        return (:timeout, nothing)
    end
    wait(t)
    err[] !== nothing && return (:error, err[])
    return (:ok, value[])
end

"""Verify one markdown page. Returns (passed::Bool, nfail::Int)."""
function verify_file(mdfile::String)
    blocks = parse_blocks(mdfile)
    executable = count(b -> b.kind != :static, blocks)
    if executable == 0
        println("  (no @setup/@example blocks)")
        return (true, 0)
    end
    sandboxes = Dict{String,Module}()
    nfail = 0
    n = 0
    for b in blocks
        n += 1
        loc = "$(basename(mdfile)) [$(b.group)] block $(n) (lines $(b.startline)-$(b.endline))"
        if b.kind === :static
            println("  ~ $(loc): static (not executed)")
            continue
        end
        mod = get!(sandboxes, b.group) do
            Module(gensym(b.group))
        end
        # Parse the whole block; a parse error is attributed to this block.
        local parsed
        try
            parsed = Meta.parseall(b.code)
            if parsed isa Expr && parsed.head === :toplevel
                for a in parsed.args
                    if a isa Expr && a.head === :error
                        throw(a.args[1])
                    end
                end
            end
        catch pe
            nfail += 1
            println("  ✗ $(loc): PARSE ERROR")
            println("      ", sprint(showerror, pe isa Exception ? pe : ErrorException(string(pe))))
            continue
        end
        t0 = time()
        status, payload = eval_with_timeout(mod, parsed, BLOCK_TIMEOUT)
        elapsed = round(time() - t0; digits=2)
        if status === :timeout
            nfail += 1
            println("  ✗ $(loc): TIMEOUT after $(BLOCK_TIMEOUT)s (possible hang)")
        elseif status === :error
            nfail += 1
            e, bt = payload
            println("  ✗ $(loc): ERROR ($(elapsed)s)")
            println("      ", sprint(showerror, e))
        else
            # Exercise the display path exactly as Documenter would, but only for
            # @example blocks (@setup output is never rendered).
            if b.kind === :example
                try
                    show(devnull, MIME"text/plain"(), payload)
                catch se
                    nfail += 1
                    println("  ✗ $(loc): SHOW/display ERROR ($(elapsed)s)")
                    println("      ", sprint(showerror, se))
                    continue
                end
            end
            println("  ✓ $(loc): OK ($(elapsed)s)")
        end
    end
    return (nfail == 0, nfail)
end

# ------------------------------------------------------------------- main

if isempty(ARGS)
    println("Usage: julia --project=docs docs/verify_examples.jl [--all | file1.md file2.md ...]")
    exit(1)
end

files = if ARGS[1] == "--all"
    sort(filter(f -> endswith(f, ".md"), readdir(joinpath(@__DIR__, "src"); join=true)))
else
    ARGS
end

passed = 0
failed = String[]

for f in files
    println("\n=== ", f, " ===")
    ok, _ = verify_file(f)
    if ok
        global passed += 1
    else
        push!(failed, basename(f))
    end
end

println("\n", passed, "/", length(files), " pages passed")
if !isempty(failed)
    println("Failed: ", join(failed, ", "))
    exit(1)
end
