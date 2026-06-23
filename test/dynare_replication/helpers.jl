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
                     shock_map::Dict{String,Int};
                     shock_scale::Dict{String,Float64}=Dict{String,Float64}(),
                     shock_remap::Dict{String,String}=Dict{String,String}(),
                     atol=1e-4, verbose=true)
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
        shock_name_raw = string(parts[end])
        var_name = join(parts[1:end-1], "_")

        shock_name = get(shock_remap, shock_name_raw, shock_name_raw)

        haskey(var_map, var_name) && haskey(shock_map, shock_name) || continue
        j_var_idx = var_map[var_name]
        j_shock_idx = shock_map[shock_name]

        H_use = min(H, size(julia_irf.values, 1))
        j_vec = julia_irf.values[1:H_use, j_var_idx, j_shock_idx]

        scale = get(shock_scale, shock_name, 1.0)
        j_vec = j_vec .* scale

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

function _parse_dynare_names(raw)
    if raw isa Matrix
        return vec([strip(string(raw[i,1])) for i in 1:size(raw, 1)])
    else
        return vec([strip(string(x)) for x in raw])
    end
end

function compare_variance_decomposition(julia_fevd, dynare_data::Dict,
                                         var_map::Dict{String,Int},
                                         shock_map::Dict{String,Int};
                                         shock_remap::Dict{String,String}=Dict{String,String}(),
                                         atol=1.0, verbose=true)
    haskey(dynare_data, "variance_decomposition") || return nothing

    d_vd = dynare_data["variance_decomposition"]
    d_names = _parse_dynare_names(dynare_data["endo_names"])
    d_exo_names = _parse_dynare_names(dynare_data["exo_names"])

    d_endo_idx = Dict{String,Int}(n => i for (i, n) in enumerate(d_names))
    d_exo_idx = Dict{String,Int}(n => i for (i, n) in enumerate(d_exo_names))

    j_vd = julia_fevd.proportions[:, :, end] .* 100.0

    all_pass = true
    if verbose
        println("\n=== Variance Decomposition (asymptotic, %) ===")
        @printf("  %-4s %-15s %-15s %10s %10s %10s\n",
                "", "Variable", "Shock", "Julia", "Dynare", "Diff")
        println("  ", "-"^70)
    end

    for (var_name, j_var_idx) in sort(collect(var_map), by=last)
        d_var_i = get(d_endo_idx, var_name, 0)
        d_var_i == 0 && continue
        for (shock_name, j_shock_idx) in sort(collect(shock_map), by=last)
            d_sn = shock_name
            for (dk, dv) in shock_remap
                dv == shock_name && (d_sn = dk)
            end
            d_shock_i = get(d_exo_idx, d_sn, 0)
            d_shock_i == 0 && (d_shock_i = get(d_exo_idx, shock_name, 0))
            d_shock_i == 0 && continue
            j_val = j_vd[j_var_idx, j_shock_idx]
            d_val = d_vd[d_var_i, d_shock_i]
            diff = abs(j_val - d_val)
            match = diff < atol
            all_pass &= match
            if verbose
                @printf("  %s  %-15s %-15s %10.4f %10.4f %10.4f\n",
                        match ? "✓" : "✗", var_name, shock_name, j_val, d_val, diff)
            end
        end
    end
    return all_pass
end

function compare_moments(julia_var_matrix::AbstractMatrix,
                          julia_autocorr::AbstractVector,
                          dynare_data::Dict, var_map::Dict{String,Int};
                          var_atol=0.05, acorr_atol=1e-3, verbose=true)
    d_names = _parse_dynare_names(dynare_data["endo_names"])
    d_endo_idx = Dict{String,Int}(n => i for (i, n) in enumerate(d_names))

    all_pass = true
    sorted_vars = sort(collect(var_map), by=last)

    if haskey(dynare_data, "var_matrix")
        d_var = dynare_data["var_matrix"]
        if verbose
            println("\n=== Variance (diagonal) ===")
            @printf("  %-4s %-15s %14s %14s %10s\n", "", "Variable", "Julia", "Dynare", "Diff")
            println("  ", "-"^60)
        end
        for (vn, j_i) in sorted_vars
            d_i = get(d_endo_idx, vn, 0)
            d_i == 0 && continue
            j_val = julia_var_matrix[j_i, j_i]
            d_val = d_var[d_i, d_i]
            diff = abs(j_val - d_val)
            scale = max(abs(d_val), 1e-10)
            match = diff < var_atol * scale + 1e-8
            all_pass &= match
            if verbose
                @printf("  %s  %-15s %14.8f %14.8f %10.2e\n",
                        match ? "✓" : "✗", vn, j_val, d_val, diff)
            end
        end
    end

    if haskey(dynare_data, "autocorr")
        d_acorr = dynare_data["autocorr"]
        if verbose
            println("\n=== Autocorrelation (lag 1) ===")
            @printf("  %-4s %-15s %14s %14s %10s\n", "", "Variable", "Julia", "Dynare", "Diff")
            println("  ", "-"^60)
        end
        for (vn, j_i) in sorted_vars
            d_i = get(d_endo_idx, vn, 0)
            d_i == 0 && continue
            j_val = julia_autocorr[j_i]
            d_val = d_acorr[d_i, d_i]
            diff = abs(j_val - d_val)
            match = diff < acorr_atol
            all_pass &= match
            if verbose
                @printf("  %s  %-15s %14.8f %14.8f %10.6f\n",
                        match ? "✓" : "✗", vn, j_val, d_val, diff)
            end
        end
    end

    return all_pass
end

function compute_moments(sol, Sigma_e=nothing)
    if Sigma_e !== nothing
        scaled_impact = sol.impact * sqrt(Sigma_e)
    else
        scaled_impact = sol.impact
    end
    Sigma_y = MacroEconometricModels.solve_lyapunov(sol.G1, scaled_impact)
    Gamma_1 = sol.G1 * Sigma_y
    n = size(Sigma_y, 1)
    acorr = zeros(n)
    for i in 1:n
        Sigma_y[i, i] > 0 && (acorr[i] = Gamma_1[i, i] / Sigma_y[i, i])
    end
    return Sigma_y, acorr
end

function run_comparison(model_name::String, spec, sol, ir;
                        var_map::Dict{String,Int}, shock_map::Dict{String,Int},
                        shock_scale::Dict{String,Float64}=Dict{String,Float64}(),
                        shock_remap::Dict{String,String}=Dict{String,String}(),
                        ss_atol=1e-6, irf_atol=1e-4, vd_atol=1.0,
                        Sigma_e=nothing)
    println("\n" * "="^60)
    println("  Model: $model_name")
    println("="^60)

    dynare = load_dynare_results(model_name)

    ss_pass = compare_steady_state(sol.spec.steady_state, dynare, var_map; atol=ss_atol)
    irf_pass = compare_irf(ir, dynare, var_map, shock_map;
                           shock_scale=shock_scale, shock_remap=shock_remap, atol=irf_atol)

    vd_pass = nothing
    if haskey(dynare, "variance_decomposition")
        fv = fevd(sol, 1000)
        vd_pass = compare_variance_decomposition(fv, dynare, var_map, shock_map;
                                                   shock_remap=shock_remap, atol=vd_atol)
    end

    mom_pass = nothing
    if haskey(dynare, "var_matrix") || haskey(dynare, "autocorr")
        Sigma_y, acorr = compute_moments(sol, Sigma_e)
        mom_pass = compare_moments(Sigma_y, acorr, dynare, var_map)
    end

    status = ["SS=$(ss_pass ? "PASS" : "FAIL")", "IRF=$(irf_pass ? "PASS" : "FAIL")"]
    vd_pass !== nothing && push!(status, "VD=$(vd_pass ? "PASS" : "FAIL")")
    mom_pass !== nothing && push!(status, "Moments=$(mom_pass ? "PASS" : "FAIL")")
    println("\n  Overall: ", join(status, ", "))

    results = ss_pass && irf_pass
    vd_pass !== nothing && (results &= vd_pass)
    mom_pass !== nothing && (results &= mom_pass)
    return results
end
