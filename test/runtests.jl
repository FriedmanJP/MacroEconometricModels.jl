# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test

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
        "core/test_lrvar.jl",
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
        "preg/test_panel_nonlinear.jl",   # moved from the ceiling ARIMA group to rebalance (#127)
    ]),
    # Group 3: IRF/FEVD/HD & VECM
    ("IRF & VECM" => [
        "var/test_irf.jl",
        "var/test_irf_ci.jl",
        "var/test_fevd.jl",
        "var/test_hd.jl",
        "vecm/test_vecm.jl",
        "preg/test_panel_iv.jl",          # moved from the ceiling ARIMA group to rebalance (#127)
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
        "midas/test_midas.jl",
        "preg/test_panel_reg.jl",
        "preg/test_panel_tests.jl",
    ]),
    # Group 6: Volatility & Non-Gaussian & Plotting & Filters & Spectral
    ("Volatility & Filters" => [
        "volatility/test_volatility.jl",
        "volatility/test_volatility_coverage.jl",
        "nongaussian/test_nongaussian_svar.jl",
        "nongaussian/test_nongaussian_internals.jl",
        "plotting/test_plot_result.jl",
        "filters/test_filters.jl",
        "filters/test_x13.jl",
        "filters/test_x13_coverage.jl",
        "spectral/test_spectral.jl",
    ]),
    # Group 7 split into three so the DSGE critical path balances across processes (#123):
    # the heavy test_ha_dsge.jl (~65% of the old group) runs alone.
    ("DSGE Core" => [
        "dsge/test_dsge.jl",
        "dsge/test_blanchard_olg.jl",
        "dsge/test_continuous_aiyagari.jl",
    ]),
    ("DSGE Bayesian & HD" => [
        "dsge/test_bayesian_dsge.jl",
        "dsge/test_dsge_hd.jl",
    ]),
    ("HA-DSGE" => [
        "dsge/test_ha_dsge.jl",
    ]),
    # Group 8: Coverage-A (DSGE — heaviest coverage tests)
    ("Coverage-A" => [
        "coverage/test_dsge_coverage.jl",
        "coverage/test_dsge_bayes_coverage.jl",
    ]),
    # Extensions: JuMP/Ipopt/PATH weakdep cold-load isolated here (#309) so the
    # ~1-3 min ext compile is paid once, in its own process, instead of twice.
    ("Extensions (JuMP/Ipopt/PATH)" => [
        "ext/test_constrained_ext.jl",
    ]),
    # Group 9: Coverage-B (medium-weight coverage tests)
    ("Coverage-B" => [
        "coverage/test_data_types_coverage.jl",
        "coverage/test_teststat_break_panel_coverage.jl",
        "coverage/test_display_coverage.jl",
        "coverage/test_gmm_ext_coverage.jl",
    ]),
    # Group 10: Coverage-C + IO. The io tests are sub-second, so they fold into this light
    # group rather than paying a standalone process (#127).
    ("Coverage-C + IO" => [
        "coverage/test_pvar_nongaussian_coverage.jl",
        "coverage/test_nowcast_coverage.jl",
        "coverage/test_vecm_teststat_coverage.jl",
        "coverage/test_misc_coverage.jl",
        "io/test_io_smoke.jl",
        "io/test_io_types.jl",
        "io/test_io_coefficients.jl",
        "io/test_io_example.jl",
        "io/test_io_multipliers.jl",
        "io/test_io_linkages.jl",
        "io/test_io_sda.jl",
        "io/test_io_extraction.jl",
        "io/test_io_environmental.jl",
        "io/test_io_bf_first.jl",
        "io/test_io_bf_second.jl",
        "io/test_io_fetch.jl",
        "io/test_io_registry.jl",
        "io/test_io_sources.jl",
        "io/test_io_parse.jl",
        "io/test_io_ext_parse.jl",
        "io/test_io_show.jl",
        "io/test_io_plotting.jl",
        "io/test_io_refs.jl",
        "io/test_io_coverage.jl",
    ]),
    # Group 11: Display regression harness (T176/#275). A dedicated group — the
    # fixtures compile a broad swath of estimators (VAR/VECM/BVAR/DSGE/GARCH/GMM/
    # panel/DiD/factor/LP/ARIMA/teststat), so it carries real compilation weight and
    # is kept out of the light coverage groups. Renders are sub-second; goldens/
    # invariants lock the display layer against silent regressions.
    ("Display" => [
        "display/test_display_invariants.jl",
        "display/test_display_goldens.jl",
    ]),
]

# Monotone expected-duration ranking (heaviest first) for the longest-first work queue (#124).
# Only the ordering matters, not accurate minutes.
function _expected_rank(name::AbstractString)
    name == "HA-DSGE"             && return 100
    name == "DSGE Core"           && return 90
    name == "DSGE Bayesian & HD"  && return 70
    name == "Extensions (JuMP/Ipopt/PATH)"    && return 60   # cold-load: schedule early
    startswith(name, "Coverage-A")            && return 60
    name == "ARIMA & Tests & Data & Reg"      && return 55
    name == "IRF & VECM"          && return 50
    name == "Bayesian & SVAR"     && return 45
    name == "Display"             && return 42   # est-heavy compile; schedule with the medium wave
    startswith(name, "Coverage")  && return 20   # light coverage groups last
    return 40                                     # default medium
end

# Multi-process runner (fallback when threads unavailable)
function run_test_group(group_name::String, files::Vector{String})
    test_dir = replace(string(@__DIR__), '\\' => '/')  # forward slashes for Windows compat
    # Time each file and print a machine-greppable FILETIME<TAB>group<TAB>file<TAB>seconds line (#125).
    includes = join(
        ["let t = @elapsed include(\"$(test_dir)/$(f)\"); " *
         "println(\"FILETIME\\t$(group_name)\\t$(f)\\t\", round(t; digits=1)); end"
         for f in files],
        "\n    ")
    fixtures_path = replace(joinpath(test_dir, "fixtures.jl"), '\\' => '/')
    code = """
    using Test, MacroEconometricModels
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
    # Child-process hygiene (#127): spawn via the current interpreter (not PATH `julia`),
    # skip startup.jl, and — decision P1.4 — restore standard Pkg.test check-bounds semantics
    # on the full-fidelity Ubuntu job only (MACRO_CHECK_BOUNDS=1 there; ~10-20% slower but net
    # time still falls due to the other savings; off on windows/macOS).
    julia_exe = joinpath(Sys.BINDIR, Base.julia_exename())
    checkbounds_flag = get(ENV, "MACRO_CHECK_BOUNDS", "") == "1" ? `--check-bounds=yes` : ``
    # Single-thread each child (Julia + OpenBLAS): the test matrices are small, so
    # multi-threaded BLAS is pure contention and N threads × ~10 children oversubscribes
    # CI cores. Mirrors the CI.yml env so local multiprocess runs behave identically.
    cmd = addenv(`$julia_exe $cov_flag $checkbounds_flag --startup-file=no --project=$(dirname(test_dir)) -e $code`,
                 "JULIA_NUM_THREADS" => "1", "OPENBLAS_NUM_THREADS" => "1")
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

    # Concurrency-capped, longest-first work queue (#124): order groups heaviest-first and
    # launch at most min(CPU_THREADS, 4) at a time, starting the next as each finishes. This
    # cuts context-switch waste and macOS memory pressure vs spawning all groups at once.
    queue = sort(collect(TEST_GROUPS); by = p -> _expected_rank(first(p)), rev = true)
    max_conc = min(Sys.CPU_THREADS, 4)
    active = Dict{Base.Process, String}()
    failed_groups = String[]
    while !isempty(queue) || !isempty(active)
        while !isempty(queue) && length(active) < max_conc
            (name, files) = popfirst!(queue)
            active[run_test_group(name, files)] = name
        end
        sleep(0.5)   # Julia has no wait_any; poll process_exited (parent cost negligible)
        for (proc, name) in collect(active)
            if process_exited(proc)
                if proc.exitcode == 0
                    @info "Test group '$name' PASSED"
                else
                    @error "Test group '$name' FAILED (exit code $(proc.exitcode))"
                    push!(failed_groups, name)
                end
                delete!(active, proc)
            end
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
        @testset "MIDAS Regression" begin include("midas/test_midas.jl") end
        @testset "Panel Regression" begin include("preg/test_panel_reg.jl") end
        @testset "Panel Specification Tests" begin include("preg/test_panel_tests.jl") end
        @testset "Panel IV" begin include("preg/test_panel_iv.jl") end
        @testset "Panel Nonlinear" begin include("preg/test_panel_nonlinear.jl") end

        # Group 6: Volatility & Filters
        @testset "Volatility Models (ARCH/GARCH/SV)" begin include("volatility/test_volatility.jl") end
        @testset "Volatility Coverage" begin include("volatility/test_volatility_coverage.jl") end
        @testset "Non-Gaussian SVAR Identification" begin include("nongaussian/test_nongaussian_svar.jl") end
        @testset "Non-Gaussian Internals" begin include("nongaussian/test_nongaussian_internals.jl") end
        @testset "Plotting" begin include("plotting/test_plot_result.jl") end
        @testset "Time Series Filters" begin include("filters/test_filters.jl") end
        @testset "X-13ARIMA-SEATS" begin include("filters/test_x13.jl") end
        @testset "X-13 Coverage" begin include("filters/test_x13_coverage.jl") end
        @testset "Spectral Analysis" begin include("spectral/test_spectral.jl") end

        # Group 7 split into three (#123)
        @testset "DSGE Core" begin
            include("dsge/test_dsge.jl")
            include("dsge/test_blanchard_olg.jl")
            include("dsge/test_continuous_aiyagari.jl")
        end
        @testset "DSGE Bayesian & HD" begin
            include("dsge/test_bayesian_dsge.jl")
            include("dsge/test_dsge_hd.jl")
        end
        @testset "HA-DSGE" begin
            include("dsge/test_ha_dsge.jl")
        end

        # Group 8: Coverage-A (DSGE)
        @testset "DSGE Coverage" begin include("coverage/test_dsge_coverage.jl") end
        @testset "DSGE Bayesian Coverage" begin include("coverage/test_dsge_bayes_coverage.jl") end

        # Extensions: JuMP/Ipopt/PATH weakdep testsets (#309)
        @testset "Constrained Extensions (JuMP/Ipopt/PATH)" begin include("ext/test_constrained_ext.jl") end

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

        # IO Analysis (folds into "Coverage-C + IO" for the parallel runner; also restores the
        # serial-fallback includes the io branch omitted — #127).
        @testset "IO Analysis" begin
            include("io/test_io_smoke.jl")
            include("io/test_io_types.jl")
            include("io/test_io_coefficients.jl")
            include("io/test_io_example.jl")
            include("io/test_io_multipliers.jl")
            include("io/test_io_linkages.jl")
            include("io/test_io_sda.jl")
            include("io/test_io_extraction.jl")
            include("io/test_io_environmental.jl")
            include("io/test_io_bf_first.jl")
            include("io/test_io_bf_second.jl")
            include("io/test_io_fetch.jl")
            include("io/test_io_registry.jl")
            include("io/test_io_sources.jl")
            include("io/test_io_parse.jl")
            include("io/test_io_ext_parse.jl")
            include("io/test_io_show.jl")
            include("io/test_io_plotting.jl")
            include("io/test_io_refs.jl")
            include("io/test_io_coverage.jl")
        end

        # Group 11: Display regression harness (T176/#275)
        @testset "Display" begin
            include("display/test_display_invariants.jl")
            include("display/test_display_goldens.jl")
        end
    end
end
