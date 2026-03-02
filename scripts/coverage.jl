#!/usr/bin/env julia
# Usage: julia --project=. --code-coverage=user scripts/coverage.jl
#
# Runs the full test suite in serial mode (single process) so all .cov files
# land in src/. Then merges them into lcov.info and prints a per-file summary.

using Pkg

# Ensure CoverageTools is available
try
    @eval using CoverageTools
catch
    Pkg.add("CoverageTools")
    @eval using CoverageTools
end

# Clean old .cov files
for dir in ["src", "ext"]
    isdir(dir) || continue
    for (root, dirs, files) in walkdir(dir)
        for f in files
            endswith(f, ".cov") && rm(joinpath(root, f))
        end
    end
end
println("Cleaned old .cov files")

# Check that --code-coverage=user was passed
if Base.JLOptions().code_coverage == 0
    error("Must run with --code-coverage=user flag:\n  julia --project=. --code-coverage=user scripts/coverage.jl")
end

# Run tests in serial mode
ENV["MACRO_SERIAL_TESTS"] = "1"
println("Running test suite in serial mode with coverage...\n")
include(joinpath(@__DIR__, "..", "test", "runtests.jl"))

# Process coverage
println("\n\n=== Coverage Report ===\n")
coverage = process_folder("src")
if isdir("ext")
    coverage_ext = process_folder("ext")
    append!(coverage, coverage_ext)
end

# Per-file summary
results = Tuple{String, Int, Int, Float64}[]
total_hit = 0
total_total = 0
for fc in coverage
    hit = count(l -> l > 0, fc.coverage)
    total = count(l -> l !== nothing, fc.coverage)
    total == 0 && continue
    pct = 100.0 * hit / total
    push!(results, (fc.filename, hit, total, pct))
    total_hit += hit
    total_total += total
end

sort!(results, by=x -> x[4])
overall = 100.0 * total_hit / total_total

println("OVERALL: $total_hit / $total_total = $(round(overall, digits=1))%\n")

# Files below 95%
below = filter(r -> r[4] < 95, results)
if !isempty(below)
    println("Files below 95%:")
    for (f, h, t, p) in below
        m = t - h
        println("  $(lpad(round(p, digits=1), 5))%  $h/$t ($m missed)  $f")
    end
    println()
end

# Files 95-99%
mid = filter(r -> 95 <= r[4] < 99, results)
if !isempty(mid)
    println("Files 95-99%: $(length(mid))")
end

above = count(r -> r[4] >= 99, results)
println("Files 99%+: $above")
println("Files 100%: $(count(r -> r[2] == r[3], results))")

# Write lcov.info
LCOV.writefile("lcov.info", coverage)
println("\nWritten lcov.info ($(length(coverage)) files)")
