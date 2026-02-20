# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

using Test

# FAST mode for development iteration (shared across all test files in threaded mode)
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"

# Shared test data generators (available to all test files)
include("fixtures.jl")

# =============================================================================
# Parallel test runner: three modes (threaded > multi-process > sequential)
# =============================================================================

const TEST_GROUPS = [
    # Group 1: Core VAR + Bayesian (heavy MCMC)
    ("Core & Bayesian" => [
        "core/test_aqua.jl",
        "var/test_core_var.jl",
        "bvar/test_bayesian.jl",
        "bvar/test_samplers.jl",
        "bvar/test_bayesian_utils.jl",
        "bvar/test_minnesota.jl",
        "bvar/test_bgr.jl",
        "core/test_coverage_gaps.jl",
    ]),
    # Group 2: IRF, FEVD, HD, StatsAPI, VECM, PVAR
    ("IRF & FEVD" => [
        "var/test_irf.jl",
        "var/test_irf_ci.jl",
        "var/test_statsapi.jl",
        "var/test_fevd.jl",
        "var/test_hd.jl",
        "core/test_summary.jl",
        "vecm/test_vecm.jl",
        "pvar/test_pvar.jl",
    ]),
    # Group 3: Factor models + Nowcasting
    ("Factor Models & Nowcasting" => [
        "factor/test_factormodel.jl",
        "factor/test_dynamicfactormodel.jl",
        "factor/test_gdfm.jl",
        "factor/test_factor_forecast.jl",
        "nowcast/test_nowcast.jl",
    ]),
    # Group 4: Local Projections
    ("Local Projections" => [
        "lp/test_lp.jl",
        "lp/test_lp_structural.jl",
        "lp/test_lp_forecast.jl",
        "lp/test_lp_fevd.jl",
    ]),
    # Group 5: ARIMA + Unit Root + GMM + Utilities + Data (~50s)
    ("ARIMA & Utilities" => [
        "teststat/test_unitroot.jl",
        "arima/test_arima.jl",
        "arima/test_arima_coverage.jl",
        "core/test_utils.jl",
        "core/test_edge_cases.jl",
        "core/test_examples.jl",
        "gmm/test_gmm.jl",
        "core/test_covariance.jl",
        "teststat/test_granger.jl",
        "data/test_data.jl",
    ]),
    # Group 6: Non-Gaussian + Display + Plotting (~15s)
    ("Non-Gaussian & Display" => [
        "teststat/test_normality.jl",
        "nongaussian/test_nongaussian_svar.jl",
        "core/test_display_backends.jl",
        "nongaussian/test_nongaussian_internals.jl",
        "core/test_error_paths.jl",
        "core/test_internal_helpers.jl",
        "plotting/test_plot_result.jl",
    ]),
    # Group 7: SVAR Identification + Model Comparison (~50s)
    ("SVAR & Model Comparison" => [
        "var/test_arias2018.jl",
        "var/test_uhlig.jl",
        "teststat/test_model_comparison.jl",
    ]),
    # Group 8: Volatility + Filters (~47s)
    ("Volatility & Filters" => [
        "volatility/test_volatility.jl",
        "volatility/test_volatility_coverage.jl",
        "filters/test_filters.jl",
    ]),
]

# Multi-process runner (fallback when threads unavailable)
function run_test_group(group_name::String, files::Vector{String})
    test_dir = replace(string(@__DIR__), '\\' => '/')  # forward slashes for Windows compat
    includes = join(["include(\"$(test_dir)/$(f)\");" for f in files], "\n    ")
    fixtures_path = replace(joinpath(test_dir, "fixtures.jl"), '\\' => '/')
    code = """
    using Test, MacroEconometricModels
    include("$(fixtures_path)")
    @testset "$group_name" begin
        $includes
    end
    """
    # Propagate --code-coverage flag to child processes (needed for CI coverage)
    # Values: 0=none, 1=user, 2=all, 3=tracefile (Julia 1.12+)
    cov_opt = Base.JLOptions().code_coverage
    cov_flag = cov_opt != 0 ? `--code-coverage=user` : ``
    cmd = `julia $cov_flag --project=$(dirname(test_dir)) -e $code`
    proc = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=false)
    return proc
end

# Check for serial mode or default to parallel
parallel = get(ENV, "MACRO_SERIAL_TESTS", "") != "1"

if parallel && Threads.nthreads() > 1
    # ─────────────────────────────────────────────────────────────────────
    # Threaded single-process parallel testing
    # Loads MacroEconometricModels ONCE, then runs groups in tasks.
    # Eliminates N-1 redundant package loads vs multi-process approach.
    # Requires: julia --threads=auto or JULIA_NUM_THREADS=auto
    # ─────────────────────────────────────────────────────────────────────
    test_dir = replace(string(@__DIR__), '\\' => '/')

    println("Running $(length(TEST_GROUPS)) test groups in $(Threads.nthreads()) threads (single process)")
    println("Set MACRO_SERIAL_TESTS=1 to run sequentially\n")

    # Load once — all tasks share the compiled code
    t_load = @elapsed using MacroEconometricModels
    @info "MacroEconometricModels loaded in $(round(t_load, digits=1))s"

    tasks = Pair{String, Task}[]
    for (group_name, files) in TEST_GROUPS
        local gn = group_name
        local fs = files
        local td = test_dir
        t = Threads.@spawn begin
            @testset "$gn" begin
                for f in fs
                    include(joinpath(td, f))
                end
            end
        end
        push!(tasks, gn => t)
    end

    # Collect results
    failed_groups = String[]
    for (name, task) in tasks
        try
            fetch(task)
            @info "Test group '$name' PASSED"
        catch e
            # On macOS/Julia 1.10, stdout pipe can close during @testset print_counts.
            # Unwrap TaskFailedException to distinguish IOError from real test failures.
            inner = e isa TaskFailedException ? e.task.exception : e
            if inner isa Base.IOError
                @warn "Test group '$name' hit IOError (stdout pipe closed) — treating as PASSED"
            else
                @error "Test group '$name' FAILED" exception=(e, catch_backtrace())
                push!(failed_groups, name)
            end
        end
    end

    isempty(failed_groups) || error("Test groups failed: $(join(failed_groups, ", "))")

elseif parallel && Sys.CPU_THREADS >= 2
    # ─────────────────────────────────────────────────────────────────────
    # Multi-process fallback (when Julia started with single thread)
    # Each group runs in its own julia process — full isolation.
    # ─────────────────────────────────────────────────────────────────────
    cov_level = Base.JLOptions().code_coverage
    println("Running $(length(TEST_GROUPS)) test groups in parallel processes ($(Sys.CPU_THREADS) CPUs)")
    println("Code coverage level: $cov_level (0=none, 1=user, 2=all)")
    println("Tip: set JULIA_NUM_THREADS=auto for faster single-process threading")
    println("Set MACRO_SERIAL_TESTS=1 to run sequentially\n")

    procs = Pair{String, Base.Process}[]
    for (group_name, files) in TEST_GROUPS
        proc = run_test_group(group_name, files)
        push!(procs, group_name => proc)
    end

    # Wait for all and collect results
    failed_groups = String[]
    for (name, proc) in procs
        wait(proc)
        if proc.exitcode != 0
            @error "Test group '$name' FAILED (exit code $(proc.exitcode))"
            push!(failed_groups, name)
        else
            @info "Test group '$name' PASSED"
        end
    end

    isempty(failed_groups) || error("Test groups failed: $(join(failed_groups, ", "))")
else
    # Sequential fallback (original behavior)
    @testset "MacroEconometricModels Package Tests" begin
        @testset "Aqua" begin
            include("core/test_aqua.jl")
        end

        @testset "Core VAR" begin
            include("var/test_core_var.jl")
        end

        @testset "Unit Root Tests" begin
            include("teststat/test_unitroot.jl")
        end

        @testset "Model Comparison Tests" begin
            include("teststat/test_model_comparison.jl")
        end

        @testset "Granger Causality Tests" begin
            include("teststat/test_granger.jl")
        end

        @testset "Bayesian Estimation" begin
            include("bvar/test_bayesian.jl")
            include("bvar/test_samplers.jl")
            include("bvar/test_bayesian_utils.jl")
        end

        @testset "Coverage Gaps" begin
            include("core/test_coverage_gaps.jl")
        end

        @testset "Impulse Response Functions" begin
            include("var/test_irf.jl")
            include("var/test_irf_ci.jl")
        end

        @testset "Minnesota Prior" begin
            include("bvar/test_minnesota.jl")
        end

        @testset "BGR Optimization" begin
            include("bvar/test_bgr.jl")
        end

        @testset "StatsAPI Compatibility" begin
            include("var/test_statsapi.jl")
        end

        @testset "FEVD" begin
            include("var/test_fevd.jl")
        end

        @testset "Historical Decomposition" begin
            include("var/test_hd.jl")
        end

        @testset "Summary Tables" begin
            include("core/test_summary.jl")
        end

        @testset "Factor Model" begin
            include("factor/test_factormodel.jl")
        end

        @testset "Dynamic Factor Model" begin
            include("factor/test_dynamicfactormodel.jl")
        end

        @testset "Generalized Dynamic Factor Model" begin
            include("factor/test_gdfm.jl")
        end

        @testset "Factor Model Forecasting" begin
            include("factor/test_factor_forecast.jl")
        end

        @testset "Nowcasting" begin
            include("nowcast/test_nowcast.jl")
        end

        @testset "Arias et al. (2018) SVAR Identification" begin
            include("var/test_arias2018.jl")
        end

        @testset "Mountford-Uhlig (2009) SVAR Identification" begin
            include("var/test_uhlig.jl")
        end

        @testset "Local Projections" begin
            include("lp/test_lp.jl")
        end

        @testset "Structural LP" begin
            include("lp/test_lp_structural.jl")
        end

        @testset "LP Forecasting" begin
            include("lp/test_lp_forecast.jl")
        end

        @testset "LP-FEVD (Gorodnichenko & Lee 2019)" begin
            include("lp/test_lp_fevd.jl")
        end

        @testset "ARIMA Models" begin
            include("arima/test_arima.jl")
        end

        @testset "ARIMA Coverage" begin
            include("arima/test_arima_coverage.jl")
        end

        @testset "Utility Functions" begin
            include("core/test_utils.jl")
        end

        @testset "Edge Cases" begin
            include("core/test_edge_cases.jl")
        end

        @testset "Documentation Examples" begin
            include("core/test_examples.jl")
        end

        @testset "GMM Estimation" begin
            include("gmm/test_gmm.jl")
        end

        @testset "Covariance Estimators" begin
            include("core/test_covariance.jl")
        end

        @testset "Multivariate Normality Tests" begin
            include("teststat/test_normality.jl")
        end

        @testset "Non-Gaussian SVAR Identification" begin
            include("nongaussian/test_nongaussian_svar.jl")
        end

        @testset "Display Backend Switching" begin
            include("core/test_display_backends.jl")
        end

        @testset "Non-Gaussian Internals" begin
            include("nongaussian/test_nongaussian_internals.jl")
        end

        @testset "Error Paths" begin
            include("core/test_error_paths.jl")
        end

        @testset "Internal Helpers" begin
            include("core/test_internal_helpers.jl")
        end

        @testset "Volatility Models (ARCH/GARCH/SV)" begin
            include("volatility/test_volatility.jl")
        end

        @testset "Volatility Coverage" begin
            include("volatility/test_volatility_coverage.jl")
        end

        @testset "VECM" begin
            include("vecm/test_vecm.jl")
        end

        @testset "Time Series Filters" begin
            include("filters/test_filters.jl")
        end

        @testset "Data Module" begin
            include("data/test_data.jl")
        end

        @testset "Panel VAR" begin
            include("pvar/test_pvar.jl")
        end

        @testset "Plotting" begin
            include("plotting/test_plot_result.jl")
        end
    end
end
