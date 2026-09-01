# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using LinearAlgebra

# FAST mode for development iteration (shared across all test files in threaded mode)
const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
# Ubuntu 1.10 Optim-v1 cell: important numerical tests only (see _numerical_groups).
const NUMERICAL = get(ENV, "MACRO_NUMERICAL_CI", "") == "1"
# CI job split: "dsge" | "empirical" | "serialization" | "" (local full suite).
const SUITE = get(ENV, "MACRO_CI_SUITE", "")

# Shared test data generators (available to all test files)
include("fixtures.jl")
include("runner_helpers.jl")

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
        "core/test_tables.jl",
        "core/test_logging.jl",
        "core/test_repro.jl",
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
        "bvar/test_tvpvar.jl",   # T250 (#349): Primiceri TVP-VAR-SV / Cogley-Sargent SV-BVAR
        "bvar/test_mfvar.jl",    # T251 (#350): Schorfheide-Song mixed-frequency VAR
        "bvar/test_glp.jl",      # T252 (#351): GLP hierarchical hyperparameter optimization
        "bvar/test_issues_523_564.jl",   # PR #597 regression tests (BVAR/FAVAR/IRF fixes)
        "var/test_arias2018.jl",
        "var/test_robust_bayes.jl",   # SID-18 (#747): Giacomini–Kitagawa robust Bayes
        "var/test_uhlig.jl",
        "var/test_ab.jl",                 # SID-13 (#742): AB-model ML
        "var/test_conditional_forecast.jl",   # T241 (#340): Waggoner-Zha conditional forecasts
        "preg/test_panel_nonlinear.jl",   # moved from the ceiling ARIMA group to rebalance (#127)
    ]),
    # Group 3: IRF/FEVD/HD & VECM
    ("IRF & VECM" => [
        "var/test_irf.jl",
        "var/test_irf_ci.jl",
        "var/test_fevd.jl",
        "var/test_hd.jl",
        "var/test_id_recovery.jl",        # SID-01 (#730): heteroskedastic kernel recovery
        "var/test_proxy.jl",              # SID-11 (#740): proxy SVAR / external instruments
        "var/test_maxshare.jl",           # SID-12 (#741): max-share identification
        "vecm/test_vecm.jl",
        "vecm/test_vecm_restrictions.jl", # EV-38 (#446)
        "preg/test_panel_iv.jl",          # moved from the ceiling ARIMA group to rebalance (#127)
    ]),
    # Group 4: LP & Factor Models & Nowcasting & DiD
    ("LP & Factor & Nowcast" => [
        "lp/test_lp.jl",
        "lp/test_lp_structural.jl",
        "lp/test_lp_forecast.jl",
        "lp/test_lp_fevd.jl",
        "lp/test_lp_weak_iv.jl",   # T245 (#344): MOP effective F + LP-IV AR bands
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
        "teststat/test_hegy.jl",   # EV-29 (#437): HEGY seasonal unit roots + ERS point-optimal
        "teststat/test_lm_unitroot.jl",
        "teststat/test_adf_2break.jl",
        "teststat/test_gregory_hansen.jl",
        "teststat/test_panel_unitroot_firstgen.jl",   # EV-20 (#428): LLC/IPS/Breitung/Fisher/Hadri
        "teststat/test_panel_cointegration.jl",       # EV-21 (#429): Pedroni/Kao/Westerlund/Fisher-Johansen
        "arima/test_arima.jl",
        "arima/test_arima_coverage.jl",
        "arima/test_arfima.jl",   # EV-13 (#421): ARFIMA + GPH + local Whittle
        "arima/test_sarima.jl",   # T242 (#341): multiplicative seasonal ARIMA
        "statespace/test_statespace.jl",   # EV-37 (#445): public state-space Kalman MLE + TVP regression
        "teststat/test_granger.jl",
        "teststat/test_dumitrescu_hurlin.jl",   # EV-24 (#432): DH panel Granger non-causality
        "teststat/test_equality.jl",   # EV-34 (#442): equality-of-distribution + rank correlations
        "teststat/test_model_comparison.jl",
        "teststat/test_normality.jl",
        "teststat/test_edf.jl",   # EV-26 (#434): EDF goodness-of-fit battery (KS/Lilliefors/CvM/AD/Watson)
        "teststat/test_bds.jl",   # EV-28 (#436): BDS iid/independence test
        "teststat/test_bubble.jl",   # EV-30 (#438): SADF/GSADF explosive-bubble detection

        "gmm/test_gmm.jl",
        "gmm/test_smm.jl",
        "data/test_data.jl",
        "pvar/test_pvar.jl",
        "reg/test_reg.jl",
        "reg/test_ivkclass.jl",    # EV-36 (#444): IV k-class — LIML / Fuller / generic k-class
        "reg/test_penalized.jl",   # EV-03 (#411): ridge / LASSO / elastic net
        "reg/test_selection.jl",   # EV-04 (#412): stepwise / best-subset / GETS
        "reg/test_tobit.jl",       # EV-17 (#425): Tobit + truncated regression
        "reg/test_heckman.jl",     # EV-18 (#426): Heckman sample-selection (two-step + MLE)
        "reg/test_count.jl",       # EV-19 (#427): Poisson / NegBin2 count-data regression
        "reg/test_robust.jl",      # EV-40 (#448): robust regression — Huber/bisquare M + Yohai MM
        "system/test_system.jl",   # EV-35 (#443): SUR / 3SLS systems estimation
        "reg/test_wildboot.jl",          # T243 (#342): wild cluster bootstrap (boottest-style)
        "reg/test_anderson_rubin.jl",    # T244 (#343): AR weak-IV-robust test + confidence set
        "reg/test_reg_diagnostics.jl",   # EV-31 (#439): White/BP/Glejser/Harvey/BG/RESET
        "reg/test_stability.jl",         # EV-32 (#440): recursive residuals / CUSUM(SQ) / Chow / influence
        "reg/test_ordered.jl",
        "reg/test_multinomial.jl",
        "midas/test_midas.jl",
        "ardl/test_ardl.jl",   # EV-08 (#416): ARDL + PSS bounds test
        "ardl/test_nardl.jl",  # EV-09 (#417): nonlinear ARDL (NARDL) + dynamic multipliers
        "ardl/test_pmg.jl",    # EV-23 (#431): panel ARDL — PMG / MG / DFE + Hausman
        "fceval/test_fceval.jl",   # EV-39 (#447): forecast eval metrics + DM/CW/MZ/encompassing + combination
        "cointreg/test_cointreg.jl",   # EV-10 (#418): FMOLS/CCR/DOLS cointegrating regression
        "teststat/test_cointegration_resid.jl",   # EV-11 (#419): Engle-Granger/Phillips-Ouliaris/Hansen-Lc/Park
        "teststat/test_variance_ratio.jl",   # EV-27 (#435): Lo-MacKinlay/Chow-Denning/Wright/Kim variance-ratio tests
        "cointreg/test_panel_cointreg.jl",   # EV-22 (#430): panel FMOLS/DOLS (group-mean + pooled)
        "preg/test_panel_reg.jl",
        "preg/test_pcse_prais.jl",   # EV-25 (#433): Beck-Katz PCSE + Prais-Winsten AR(1)
        "preg/test_panel_tests.jl",
    ]),
    # Group 6: Volatility & Non-Gaussian & Plotting & Filters & Spectral
    ("Volatility & Filters" => [
        "volatility/test_volatility.jl",
        "volatility/test_volatility_coverage.jl",
        "volatility/test_garch_midas.jl",   # EV-02 (#410): GARCH-MIDAS long/short-run components
        "volatility/test_figarch.jl",       # EV-14 (#422): FIGARCH/FIEGARCH fractionally-integrated volatility
        "volatility/test_garch_family.jl",  # EV-15 (#423): IGARCH/Component-GARCH/APARCH + sign-bias/Nyblom tests
        "mgarch/test_mgarch.jl",            # EV-16 (#424): multivariate GARCH — CCC/DCC/BEKK
        "nongaussian/test_nongaussian_svar.jl",
        "nongaussian/test_nongaussian_internals.jl",
        "filters/test_filters.jl",
        "filters/test_x13.jl",
        "filters/test_x13_coverage.jl",
        "spectral/test_spectral.jl",
    ]),
    # Plotting — consolidated plot_result harness (PLT-39). Split from the old
    # monolith (test_plot_result.jl) into per-domain lanes + the Wave-2 dispatch
    # lanes + renderer-option tests; every file shares the structural-assertion
    # helper test/plotting/plot_test_helpers.jl (parses EXTRACTED JSON literals —
    # check_plot / assert_all_json_valid / assert_escapes / series_count …).
    ("Plotting" => [
        "plotting/test_plot_render.jl",
        "plotting/test_plot_irf_fevd_hd.jl",
        "plotting/test_plot_forecast_filters.jl",
        "plotting/test_plot_models.jl",
        "plotting/test_plot_reg_micro.jl",
        "plotting/test_plot_nowcast.jl",
        "plotting/test_plot_wave2_laneA.jl",
        "plotting/test_plot_wave2_laneB.jl",
        "plotting/test_plot_wave2_laneC.jl",
        "plotting/test_plot_wave2_laneD.jl",
        "plotting/test_plot_wave2_laneE.jl",
        "plotting/test_plot_wave2_laneF.jl",
        "dsge/test_dcegm_plot.jl",
    ]),
    # Nonlinear time series (EV-05 threshold/SETAR; EV-06 STAR & EV-07 Markov
    # switching join this group).
    ("Nonlinear" => [
        "nonlinear/test_threshold.jl",
        "nonlinear/test_star.jl",       # EV-06 smooth-transition (STAR)
        "nonlinear/test_markov_switching.jl",  # EV-07 Markov-switching regression / MS-AR
        "nonparametric/test_nonparametric.jl",  # EV-33 (#441): kernel density / kernel-reg / LOWESS
    ]),
    # Group 7 split into three so the DSGE critical path balances across processes (#123):
    # the heavy test_ha_dsge.jl (~65% of the old group) runs alone.
    ("DSGE Core" => [
        "dsge/test_dsge.jl",
        "dsge/test_perfect_foresight_sparse.jl",
        "dsge/test_dcegm_spec.jl",
        "dsge/test_modelspec_kinds.jl",
        "dsge/test_blanchard_olg.jl",
        "dsge/test_lifecycle_olg.jl",
        "dsge/test_continuous_aiyagari.jl",
    ]),
    ("DSGE Bayesian & HD" => [
        "dsge/test_bayesian_dsge.jl",
        "dsge/test_dsge_hd.jl",
    ]),
    ("HA-DSGE" => [
        "dsge/test_ha_dsge.jl",
    ]),
    ("HA-DSGE Advanced" => [
        "dsge/test_ha_dsge_advanced.jl",
        "dsge/test_modelspec_blocks.jl",
        "dsge/test_modelspec_multipop.jl",
        "dsge/test_firm_system.jl",
        "dsge/test_intermediary_system.jl",
        "dsge/test_ha_occbin.jl",
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
        "io/test_io_ras.jl",
        "io/test_io_extraction.jl",
        "io/test_io_price.jl",
        "io/test_io_impact.jl",
        "io/test_io_network.jl",
        "io/test_io_environmental.jl",
        "io/test_io_mrio.jl",
        "io/test_io_bf_first.jl",
        "io/test_io_bf_second.jl",
        "io/test_io_bf_network.jl",
        "io/test_io_bf_equilibrium.jl",
        "io/test_io_bf_hessian.jl",
        "io/test_io_bf_wedges.jl",
        "io/test_io_bf_misalloc.jl",
        "io/test_io_fetch.jl",
        "io/test_io_registry.jl",
        "io/test_io_sources.jl",
        "io/test_io_parse.jl",
        "io/test_io_ext_parse.jl",
        "io/test_io_source_parse.jl",
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
    # Group 12: Policy counterfactuals (CF-01, #381). Later CF tasks append their
    # test files here. Lightest group; ranked last in _expected_rank.
    ("Counterfactual" => [
        "counterfactual/test_types.jl",
        "counterfactual/test_rules.jl",
        "counterfactual/test_kernel.jl",
        "counterfactual/test_empirical.jl",
        "counterfactual/test_forecast.jl",
        "counterfactual/test_irf_target.jl",
        "counterfactual/test_model_dsge.jl",
        "counterfactual/test_model_ha.jl",
        "counterfactual/test_behavioral.jl",
        "counterfactual/test_counterfactual.jl",
        "counterfactual/test_optimal_policy.jl",
        "counterfactual/test_moments.jl",
        "counterfactual/test_opp.jl",
        "counterfactual/test_opp_inference.jl",
        "counterfactual/test_constrained_opp.jl",
        "counterfactual/test_opp_sequence.jl",
        "counterfactual/test_model_bank.jl",
        "counterfactual/test_historical.jl",
        "counterfactual/test_diagnostics.jl",
        "counterfactual/test_mp_shocks_data.jl",
        "counterfactual/test_show.jl",
        "counterfactual/test_plotting.jl",
        "counterfactual/test_oracles.jl",
    ]),
    # Serialization suite (`MACRO_CI_SUITE=serialization`): round-trip files
    # pulled out of the empirical groups so they do not share a process with
    # display-backend tests or sit on the 60 min job timeout.
    ("Serialization" => [
        "core/test_serialization.jl",
    ]),
]

# Flags every multiprocess child must share so they reuse one compile cache.
# Coverage follows the parent (Pkg.test / julia-runtest). check-bounds is
# only the Ubuntu 1.10 Optim-v1 cell (MACRO_CHECK_BOUNDS=1); 1.12 cells
# keep the faster default (~10-20%, #127 P1.4).
function _child_julia_cmd(code::String; group_name::String="_warmup")
    test_dir = replace(string(@__DIR__), '\\' => '/')
    cov_flag = Base.JLOptions().code_coverage != 0 ? `--code-coverage=user` : ``
    checkbounds_flag = get(ENV, "MACRO_CHECK_BOUNDS", "") == "1" ? `--check-bounds=yes` : ``
    julia_exe = joinpath(Sys.BINDIR, Base.julia_exename())
    addenv(`$julia_exe $cov_flag $checkbounds_flag --startup-file=no --project=$(dirname(test_dir)) -e $code`,
           "JULIA_NUM_THREADS" => "1",
           "OPENBLAS_NUM_THREADS" => string(_blas_threads_for_group(group_name)))
end

# Write compiled/*.ji once before the parallel children start. Four children
# `using MacroEconometricModels` at once race on LinearSolve/PureKLU pidfiles
# (macOS empirical: invalid checksum, LinearSolveForwardDiffExt
# `__precompile__(false)`). The dsge/empirical matrix already has distinct
# julia-actions/cache keys (`include-matrix`); this is the intra-job race.
function _warm_compile_cache()
    t = @elapsed run(pipeline(_child_julia_cmd("using MacroEconometricModels");
                              stdout=stdout, stderr=stderr))
    println("FILETIME\t__runner__\tusing MacroEconometricModels\t", round(t; digits=1))
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
    const NUMERICAL = get(ENV, "MACRO_NUMERICAL_CI", "") == "1"
    include("$(fixtures_path)")
    @testset "$group_name" begin
        $includes
    end
    """
    proc = run(pipeline(_child_julia_cmd(code; group_name=group_name);
                        stdout=stdout, stderr=stderr); wait=false)
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
    run_groups = _ci_suite_groups(_numerical_groups(TEST_GROUPS, NUMERICAL), SUITE)
    println("Running $(length(run_groups)) test groups in parallel processes ($(Sys.CPU_THREADS) CPUs)")
    println("Code coverage level: $cov_level (0=none, 1=user, 2=all)")
    FAST && println("FAST mode enabled (reduced sampling)")
    NUMERICAL && println("NUMERICAL CI profile (important numerical tests only)")
    !isempty(SUITE) && println("CI suite part: $SUITE")
    println("Set MACRO_SERIAL_TESTS=1 to run sequentially\n")

    _warm_compile_cache()

    # Concurrency-capped, longest-first work queue (#124): order groups heaviest-first and
    # launch at most min(CPU_THREADS, 4) at a time, starting the next as each finishes. This
    # cuts context-switch waste and macOS memory pressure vs spawning all groups at once.
    queue = sort(collect(run_groups); by = p -> _expected_rank(first(p)), rev = true)
    max_conc = _runner_max_conc(Sys.CPU_THREADS)
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

    run_groups = _ci_suite_groups(_numerical_groups(TEST_GROUPS, NUMERICAL), SUITE)
    println("Running $(length(run_groups)) test groups in $(Threads.nthreads()) threads (single process, max_conc=$(_runner_max_conc(Threads.nthreads())))")
    FAST && println("FAST mode enabled (reduced sampling)")
    NUMERICAL && println("NUMERICAL CI profile (important numerical tests only)")
    !isempty(SUITE) && println("CI suite part: $SUITE")
    println("Set MACRO_SERIAL_TESTS=1 to run sequentially\n")

    # Load once — all tasks share the compiled code. This is the Windows CI path:
    # NTFS/pkgimage load is ~6 min per process, so 17 child processes dominate the
    # job. One load + a concurrency-capped work queue matches the multiprocess
    # longest-first schedule without paying that tax 17 times.
    t_load = @elapsed using MacroEconometricModels
    @info "MacroEconometricModels loaded in $(round(t_load, digits=1))s"
    println("FILETIME\t__runner__\tusing MacroEconometricModels\t", round(t_load; digits=1))

    max_conc = _runner_max_conc(Threads.nthreads())
    work = _make_work_queue(run_groups)
    failed_groups = String[]
    failed_lock = ReentrantLock()
    @sync for _ in 1:max_conc
        Threads.@spawn begin
            for (gn, fs) in work
                try
                    _with_group_blas(gn) do
                        @testset "$gn" begin
                            for f in fs
                                t = @elapsed include(joinpath(test_dir, f))
                                println("FILETIME\t$(gn)\t$(f)\t", round(t; digits=1))
                            end
                        end
                    end
                    @info "Test group '$gn' PASSED"
                catch e
                    inner = e isa TaskFailedException ? e.task.exception : e
                    if inner isa Base.IOError
                        @warn "Test group '$gn' hit IOError (stdout pipe closed) — treating as PASSED"
                    else
                        @error "Test group '$gn' FAILED" exception=(e, catch_backtrace())
                        lock(failed_lock) do
                            push!(failed_groups, gn)
                        end
                    end
                end
            end
        end
    end

    isempty(failed_groups) || error("Test groups failed: $(join(failed_groups, ", "))")

else
    # Sequential fallback (serial mode or single-thread single-CPU).
    # File list is TEST_GROUPS — do not maintain a second copy (MSR-28).
    @testset "MacroEconometricModels Package Tests" begin
        for (gname, files) in TEST_GROUPS
            @testset "$gname" begin
                for f in files
                    include(f)
                end
            end
        end
    end
end
