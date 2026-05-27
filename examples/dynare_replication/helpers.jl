using MAT, Printf

const DYNARE_RESULTS_DIR = joinpath(@__DIR__, "dynare_results")

function load_dynare_results(name::String)
    path = joinpath(DYNARE_RESULTS_DIR, "$(name).mat")
    isfile(path) || error("No Dynare results for '$name'. Run run_dynare_reference.m first.")
    matread(path)
end

function compare_steady_state(julia_ss::Vector, dynare_data::Dict, var_map::Dict{String,Int};
                              atol=1e-6, verbose=true)
    d_ss = vec(dynare_data["steady_state"])
    d_names = dynare_data["endo_names"]
    if d_names isa Matrix
        d_names = vec(d_names)
    end

    all_pass = true
    if verbose
        println("=== Steady State Comparison ===")
        @printf("  %-4s %-20s %15s %15s %12s\n", "", "Variable", "Julia", "Dynare", "Diff")
        println("  ", "-"^70)
    end

    for (d_name, d_idx) in sort(collect(enumerate(d_names)), by=x->x[1])
        dname_str = strip(string(d_name))
        haskey(var_map, dname_str) || continue
        j_idx = var_map[dname_str]
        j_val = julia_ss[j_idx]
        d_val = d_ss[d_idx]
        diff = abs(j_val - d_val)
        match = diff < atol
        status = match ? "✓" : "✗"
        all_pass &= match
        if verbose
            @printf("  %s  %-20s %15.8f %15.8f %12.2e\n",
                    status, dname_str, j_val, d_val, diff)
        end
    end
    return all_pass
end

function compare_irf(julia_irf, dynare_data::Dict, var_map::Dict{String,Int},
                     shock_map::Dict{String,Int}; atol=1e-4, verbose=true)
    haskey(dynare_data, "irfs") || return true

    d_irfs = dynare_data["irfs"]
    all_pass = true

    if verbose
        println("\n=== IRF Comparison ===")
        @printf("  %-4s %-30s %12s %12s\n", "", "Response", "MaxAbsDiff", "Status")
        println("  ", "-"^60)
    end

    for (field, d_vals) in d_irfs
        field_str = string(field)
        d_vec = vec(d_vals)
        H = length(d_vec)
        H == 0 && continue

        parts = split(field_str, "_")
        shock_name = string(parts[end])
        var_name = join(parts[1:end-1], "_")

        haskey(var_map, var_name) && haskey(shock_map, shock_name) || continue
        j_var_idx = var_map[var_name]
        j_shock_idx = shock_map[shock_name]

        H_use = min(H, size(julia_irf.values, 1))
        j_vec = julia_irf.values[1:H_use, j_var_idx, j_shock_idx]
        d_vec_use = d_vec[1:H_use]

        max_diff = maximum(abs.(j_vec .- d_vec_use))
        match = max_diff < atol
        all_pass &= match
        status = match ? "✓ PASS" : "✗ FAIL"

        if verbose
            @printf("  %s  %-30s %12.2e %12s\n",
                    match ? "✓" : "✗", field_str, max_diff, status)
        end
    end
    return all_pass
end

function run_comparison(model_name::String, spec, sol, ir;
                        var_map::Dict{String,Int}, shock_map::Dict{String,Int},
                        ss_atol=1e-6, irf_atol=1e-4)
    println("\n" * "="^60)
    println("  Model: $model_name")
    println("="^60)

    dynare = load_dynare_results(model_name)

    ss_pass = compare_steady_state(sol.spec.steady_state, dynare, var_map; atol=ss_atol)
    irf_pass = compare_irf(ir, dynare, var_map, shock_map; atol=irf_atol)

    println("\n  Overall: SS=$(ss_pass ? "PASS" : "FAIL"), IRF=$(irf_pass ? "PASS" : "FAIL")")
    return ss_pass && irf_pass
end
