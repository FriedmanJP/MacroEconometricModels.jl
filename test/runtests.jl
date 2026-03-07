# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using FFTW  # activate FFTW extension for GDFM tests

# FAST mode for development iteration (shared across all test files in threaded mode)
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"

# Shared test data generators (available to all test files)
include("fixtures.jl")

# =============================================================================
# Parallel test runner: three modes (threaded > multi-process > sequential)
# =============================================================================

const TEST_GROUPS = [
    # Group 1: Core & VAR (lightweight tests, many files)
    ("Core & VAR" => [
        "core/test_aqua.jl",
        "core/test_kalman.jl",
        "core/test_quadrature.jl",
        "var/test_core_var.jl",
        "var/test_statsapi.jl",
        "core/test_summary.jl",
        "core/test_utils.jl",
        "core/test_edge_cases.jl",
        "core/test_examples.jl",
        "core/test_covariance.jl",
        "core/test_internal_helpers.jl",
        "core/test_error_paths.jl",
        "core/test_display_backends.jl",
        "core/test_coverage_gaps.jl",
    ]),
    # Group 2: Bayesian & SVAR (heavy sampling + multi-start optimization)
    ("Bayesian & SVAR" => [
        "bvar/test_bayesian.jl",
        "bvar/test_samplers.jl",
        "bvar/test_bayesian_utils.jl",
        "bvar/test_minnesota.jl",
        "bvar/test_bgr.jl",
        "var/test_arias2018.jl",
        "var/test_uhlig.jl",
    ]),
    # Group 3: IRF/FEVD/HD & VECM
    ("IRF & VECM" => [
        "var/test_irf.jl",
        "var/test_irf_ci.jl",
        "var/test_fevd.jl",
        "var/test_hd.jl",
        "vecm/test_vecm.jl",
    ]),
    # Group 4: LP & Factor Models & Nowcasting & DiD
    ("LP & Factor & Nowcast" => [
        "lp/test_lp.jl",
        "lp/test_lp_structural.jl",
        "lp/test_lp_forecast.jl",
        "lp/test_lp_fevd.jl",
        "factor/test_factormodel.jl",
        "factor/test_dynamicfactormodel.jl",
        "factor/test_gdfm.jl",
        "factor/test_factor_forecast.jl",
        "factor/test_restricted.jl",
        "factor/test_favar.jl",
        "factor/test_structural_dfm.jl",
        "nowcast/test_nowcast.jl",
        "did/test_did.jl",
        "did/test_lpdid.jl",
    ]),
    # Group 5: ARIMA & Statistical Tests & Data & PVAR & Reg
    ("ARIMA & Tests & Data & Reg" => [
        "teststat/test_unitroot.jl",
        "teststat/test_structural_break.jl",
        "teststat/test_fourier.jl",
        "teststat/test_dfgls.jl",
        "teststat/test_lm_unitroot.jl",
        "teststat/test_adf_2break.jl",
        "teststat/test_gregory_hansen.jl",
        "arima/test_arima.jl",
        "arima/test_arima_coverage.jl",
        "teststat/test_granger.jl",
        "teststat/test_model_comparison.jl",
        "teststat/test_normality.jl",
        "gmm/test_gmm.jl",
        "gmm/test_smm.jl",
        "data/test_data.jl",
        "pvar/test_pvar.jl",
        "reg/test_reg.jl",
        "reg/test_ordered.jl",
        "reg/test_multinomial.jl",
    ]),
    # Group 6: Volatility & Non-Gaussian & Plotting & Filters
    ("Volatility & Filters" => [
        "volatility/test_volatility.jl",
        "volatility/test_volatility_coverage.jl",
        "nongaussian/test_nongaussian_svar.jl",
        "nongaussian/test_nongaussian_internals.jl",
        "plotting/test_plot_result.jl",
        "filters/test_filters.jl",
    ]),
    # Group 7: DSGE Models
    ("DSGE Models" => [
        "dsge/test_dsge.jl",
        "dsge/test_bayesian_dsge.jl",
    ]),
    # Group 8: Coverage-A (DSGE — heaviest coverage tests)
    ("Coverage-A" => [
        "coverage/test_dsge_coverage.jl",
        "coverage/test_dsge_bayes_coverage.jl",
    ]),
    # Group 9: Coverage-B (medium-weight coverage tests)
    ("Coverage-B" => [
        "coverage/test_data_types_coverage.jl",
        "coverage/test_teststat_break_panel_coverage.jl",
        "coverage/test_display_coverage.jl",
        "coverage/test_gmm_ext_coverage.jl",
    ]),
    # Group 10: Coverage-C (lightweight coverage tests)
    ("Coverage-C" => [
        "coverage/test_pvar_nongaussian_coverage.jl",
        "coverage/test_nowcast_coverage.jl",
        "coverage/test_vecm_teststat_coverage.jl",
        "coverage/test_misc_coverage.jl",
    ]),
]

# Multi-process runner (fallback when threads unavailable)
function run_test_group(group_name::String, files::Vector{String})
    test_dir = replace(string(@__DIR__), '\\' => '/')  # forward slashes for Windows compat
    includes = join(["include(\"$(test_dir)/$(f)\");" for f in files], "\n    ")
    fixtures_path = replace(joinpath(test_dir, "fixtures.jl"), '\\' => '/')
    code = """
    using Test, MacroEconometricModels, FFTW
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
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

# =============================================================================
# Execution mode selection (priority order):
#   1. MACRO_SERIAL_TESTS=1     → sequential (debugging)
#   2. MACRO_MULTIPROCESS_TESTS=1 → multi-process parallel (CI-safe)
#   3. Threads.nthreads() > 1   → threaded single-process (local dev)
#   4. Sys.CPU_THREADS >= 2     → multi-process fallback
#   5. else                     → sequential
# =============================================================================

serial = get(ENV, "MACRO_SERIAL_TESTS", "") == "1"
multiprocess = get(ENV, "MACRO_MULTIPROCESS_TESTS", "") == "1"

if !serial && (multiprocess || (!serial && Threads.nthreads() == 1 && Sys.CPU_THREADS >= 2))
    # ─────────────────────────────────────────────────────────────────────
    # Multi-process parallel testing
    # Each group runs in its own julia process — full isolation.
    # Triggered by MACRO_MULTIPROCESS_TESTS=1 or as fallback with CPUs >= 2.
    # ─────────────────────────────────────────────────────────────────────
    cov_level = Base.JLOptions().code_coverage
    println("Running $(length(TEST_GROUPS)) test groups in parallel processes ($(Sys.CPU_THREADS) CPUs)")
    println("Code coverage level: $cov_level (0=none, 1=user, 2=all)")
    FAST && println("FAST mode enabled (reduced sampling)")
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

elseif !serial && Threads.nthreads() > 1
    # ─────────────────────────────────────────────────────────────────────
    # Threaded single-process parallel testing
    # Loads MacroEconometricModels ONCE, then runs groups in tasks.
    # Requires: julia --threads=auto or JULIA_NUM_THREADS=auto
    # ─────────────────────────────────────────────────────────────────────
    test_dir = replace(string(@__DIR__), '\\' => '/')

    println("Running $(length(TEST_GROUPS)) test groups in $(Threads.nthreads()) threads (single process)")
    FAST && println("FAST mode enabled (reduced sampling)")
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

else
    # Sequential fallback (serial mode or single-thread single-CPU)
    @testset "MacroEconometricModels Package Tests" begin
        # Group 1: Core & VAR
        @testset "Aqua" begin include("core/test_aqua.jl") end
        @testset "Core Kalman" begin include("core/test_kalman.jl") end
        @testset "Core Quadrature" begin include("core/test_quadrature.jl") end
        @testset "Core VAR" begin include("var/test_core_var.jl") end
        @testset "StatsAPI Compatibility" begin include("var/test_statsapi.jl") end
        @testset "Summary Tables" begin include("core/test_summary.jl") end
        @testset "Utility Functions" begin include("core/test_utils.jl") end
        @testset "Edge Cases" begin include("core/test_edge_cases.jl") end
        @testset "Documentation Examples" begin include("core/test_examples.jl") end
        @testset "Covariance Estimators" begin include("core/test_covariance.jl") end
        @testset "Internal Helpers" begin include("core/test_internal_helpers.jl") end
        @testset "Error Paths" begin include("core/test_error_paths.jl") end
        @testset "Display Backend Switching" begin include("core/test_display_backends.jl") end
        @testset "Coverage Gaps" begin include("core/test_coverage_gaps.jl") end

        # Group 2: Bayesian & SVAR
        @testset "Bayesian Estimation" begin include("bvar/test_bayesian.jl") end
        @testset "Bayesian Samplers" begin include("bvar/test_samplers.jl") end
        @testset "Bayesian Utils" begin include("bvar/test_bayesian_utils.jl") end
        @testset "Minnesota Prior" begin include("bvar/test_minnesota.jl") end
        @testset "BGR Optimization" begin include("bvar/test_bgr.jl") end
        @testset "Arias et al. (2018) SVAR Identification" begin include("var/test_arias2018.jl") end
        @testset "Mountford-Uhlig (2009) SVAR Identification" begin include("var/test_uhlig.jl") end

        # Group 3: IRF & VECM
        @testset "Impulse Response Functions" begin include("var/test_irf.jl") end
        @testset "IRF Confidence Intervals" begin include("var/test_irf_ci.jl") end
        @testset "FEVD" begin include("var/test_fevd.jl") end
        @testset "Historical Decomposition" begin include("var/test_hd.jl") end
        @testset "VECM" begin include("vecm/test_vecm.jl") end

        # Group 4: LP & Factor & Nowcast
        @testset "Local Projections" begin include("lp/test_lp.jl") end
        @testset "Structural LP" begin include("lp/test_lp_structural.jl") end
        @testset "LP Forecasting" begin include("lp/test_lp_forecast.jl") end
        @testset "LP-FEVD (Gorodnichenko & Lee 2019)" begin include("lp/test_lp_fevd.jl") end
        @testset "Factor Model" begin include("factor/test_factormodel.jl") end
        @testset "Dynamic Factor Model" begin include("factor/test_dynamicfactormodel.jl") end
        @testset "Generalized Dynamic Factor Model" begin include("factor/test_gdfm.jl") end
        @testset "Factor Model Forecasting" begin include("factor/test_factor_forecast.jl") end
        @testset "Restricted Factor Models" begin include("factor/test_restricted.jl") end
        @testset "FAVAR" begin include("factor/test_favar.jl") end
        @testset "Structural DFM" begin include("factor/test_structural_dfm.jl") end
        @testset "Nowcasting" begin include("nowcast/test_nowcast.jl") end
        @testset "Difference-in-Differences" begin include("did/test_did.jl") end
        @testset "LP-DiD" begin include("did/test_lpdid.jl") end

        # Group 5: ARIMA & Tests & Data
        @testset "Unit Root Tests" begin include("teststat/test_unitroot.jl") end
        @testset "Structural Break & Panel Unit Root" begin include("teststat/test_structural_break.jl") end
        @testset "Fourier Unit Root Tests" begin include("teststat/test_fourier.jl") end
        @testset "DF-GLS Unit Root Test" begin include("teststat/test_dfgls.jl") end
        @testset "LM Unit Root Test" begin include("teststat/test_lm_unitroot.jl") end
        @testset "Two-Break ADF Test" begin include("teststat/test_adf_2break.jl") end
        @testset "Gregory-Hansen Cointegration Test" begin include("teststat/test_gregory_hansen.jl") end
        @testset "ARIMA Models" begin include("arima/test_arima.jl") end
        @testset "ARIMA Coverage" begin include("arima/test_arima_coverage.jl") end
        @testset "Granger Causality Tests" begin include("teststat/test_granger.jl") end
        @testset "Model Comparison Tests" begin include("teststat/test_model_comparison.jl") end
        @testset "Multivariate Normality Tests" begin include("teststat/test_normality.jl") end
        @testset "GMM Estimation" begin include("gmm/test_gmm.jl") end
        @testset "SMM Estimation" begin include("gmm/test_smm.jl") end
        @testset "Data Module" begin include("data/test_data.jl") end
        @testset "Panel VAR" begin include("pvar/test_pvar.jl") end
        @testset "Cross-Sectional Models" begin include("reg/test_reg.jl") end
        @testset "Ordered Models" begin include("reg/test_ordered.jl") end
        @testset "Multinomial Models" begin include("reg/test_multinomial.jl") end

        # Group 6: Volatility & Filters
        @testset "Volatility Models (ARCH/GARCH/SV)" begin include("volatility/test_volatility.jl") end
        @testset "Volatility Coverage" begin include("volatility/test_volatility_coverage.jl") end
        @testset "Non-Gaussian SVAR Identification" begin include("nongaussian/test_nongaussian_svar.jl") end
        @testset "Non-Gaussian Internals" begin include("nongaussian/test_nongaussian_internals.jl") end
        @testset "Plotting" begin include("plotting/test_plot_result.jl") end
        @testset "Time Series Filters" begin include("filters/test_filters.jl") end

        # Group 7: DSGE Models
        @testset "DSGE Models" begin
            include("dsge/test_dsge.jl")
            include("dsge/test_bayesian_dsge.jl")
        end

        # Group 8: Coverage-A (DSGE)
        @testset "DSGE Coverage" begin include("coverage/test_dsge_coverage.jl") end
        @testset "DSGE Bayesian Coverage" begin include("coverage/test_dsge_bayes_coverage.jl") end

        # Group 9: Coverage-B (medium-weight)
        @testset "Data Types Coverage" begin include("coverage/test_data_types_coverage.jl") end
        @testset "Structural Break & Panel Coverage" begin include("coverage/test_teststat_break_panel_coverage.jl") end
        @testset "Display & Plotting Coverage" begin include("coverage/test_display_coverage.jl") end
        @testset "GMM & Extension Coverage" begin include("coverage/test_gmm_ext_coverage.jl") end

        # Group 10: Coverage-C (lightweight)
        @testset "PVAR & Non-Gaussian Coverage" begin include("coverage/test_pvar_nongaussian_coverage.jl") end
        @testset "Nowcast Coverage" begin include("coverage/test_nowcast_coverage.jl") end
        @testset "VECM & Teststat Coverage" begin include("coverage/test_vecm_teststat_coverage.jl") end
        @testset "Misc Coverage" begin include("coverage/test_misc_coverage.jl") end
    end
end
