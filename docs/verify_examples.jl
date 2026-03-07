#!/usr/bin/env julia
#
# Verify that @setup/@example blocks in a documentation .md file run without error.
#
# Usage:
#   julia --project=docs docs/verify_examples.jl docs/src/manual.md
#   julia --project=docs docs/verify_examples.jl docs/src/manual.md docs/src/bayesian.md
#   julia --project=docs docs/verify_examples.jl --all
#

using MacroEconometricModels

function extract_blocks(mdfile::String)
    blocks = String[]
    lines = readlines(mdfile)
    i = 1
    while i <= length(lines)
        m = match(r"^```@(setup|example)\s+\w+", lines[i])
        if m !== nothing
            code = String[]
            i += 1
            while i <= length(lines) && lines[i] != "```"
                # skip `# hide` lines
                endswith(lines[i], "# hide") || push!(code, lines[i])
                i += 1
            end
            push!(blocks, join(code, "\n"))
        end
        i += 1
    end
    return blocks
end

function verify_file(mdfile::String)
    blocks = extract_blocks(mdfile)
    isempty(blocks) && (println("  (no @example blocks)"); return true)

    code = join(blocks, "\n")
    mod = Module(gensym(basename(mdfile)))
    Core.eval(mod, :(using MacroEconometricModels))

    try
        Core.eval(mod, Meta.parse("begin\n$code\nend"))
        return true
    catch e
        println(stderr, "  ERROR: ", sprint(showerror, e))
        return false
    end
end

# --- main ---

if isempty(ARGS)
    println("Usage: julia --project=docs docs/verify_examples.jl [--all | file1.md file2.md ...]")
    exit(1)
end

files = if ARGS[1] == "--all"
    filter(f -> endswith(f, ".md"), readdir("docs/src"; join=true))
else
    ARGS
end

passed = 0
failed = String[]

for f in files
    print(basename(f), " ... ")
    if verify_file(f)
        println("OK")
        global passed += 1
    else
        println("FAIL")
        push!(failed, basename(f))
    end
end

println("\n$(passed)/$(length(files)) passed")
if !isempty(failed)
    println("Failed: ", join(failed, ", "))
    exit(1)
end
