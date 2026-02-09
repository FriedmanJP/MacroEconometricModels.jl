using Test

# =============================================================================
# Parallel test runner: spawns independent Julia processes per test group
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
    # Group 2: IRF, FEVD, HD, StatsAPI, VECM
    ("IRF & FEVD" => [
        "var/test_irf.jl",
        "var/test_irf_ci.jl",
        "var/test_statsapi.jl",
        "var/test_fevd.jl",
        "var/test_hd.jl",
        "core/test_summary.jl",
        "vecm/test_vecm.jl",
    ]),
    # Group 3: Factor models
    ("Factor Models" => [
        "factor/test_factormodel.jl",
        "factor/test_dynamicfactormodel.jl",
        "factor/test_gdfm.jl",
        "factor/test_factor_forecast.jl",
    ]),
    # Group 4: Local Projections
    ("Local Projections" => [
        "lp/test_lp.jl",
        "lp/test_lp_structural.jl",
        "lp/test_lp_forecast.jl",
        "lp/test_lp_fevd.jl",
    ]),
    # Group 5: ARIMA + Unit Root + GMM + Utilities + Filters + Model Comparison + Data
    ("ARIMA & Utilities" => [
        "teststat/test_unitroot.jl",
        "arima/test_arima.jl",
        "arima/test_arima_coverage.jl",
        "core/test_utils.jl",
        "core/test_edge_cases.jl",
        "core/test_examples.jl",
        "gmm/test_gmm.jl",
        "core/test_covariance.jl",
        "filters/test_filters.jl",
        "teststat/test_model_comparison.jl",
        "teststat/test_granger.jl",
        "data/test_data.jl",
    ]),
    # Group 6: Non-Gaussian + Display + Misc
    ("Non-Gaussian & Display" => [
        "teststat/test_normality.jl",
        "nongaussian/test_nongaussian_svar.jl",
        "core/test_display_backends.jl",
        "nongaussian/test_nongaussian_internals.jl",
        "core/test_error_paths.jl",
        "core/test_internal_helpers.jl",
        "var/test_arias2018.jl",
    ]),
    # Group 7: Volatility models (ARCH/GARCH/SV — MCMC heavy)
    ("Volatility Models" => [
        "volatility/test_volatility.jl",
        "volatility/test_volatility_coverage.jl",
    ]),
]

function run_test_group(group_name::String, files::Vector{String})
    test_dir = replace(string(@__DIR__), '\\' => '/')  # forward slashes for Windows compat
    includes = join(["include(\"$(test_dir)/$(f)\");" for f in files], "\n    ")
    code = """
    using Test, MacroEconometricModels
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

# Check for PARALLEL_TESTS env var or default to parallel
parallel = get(ENV, "MACRO_SERIAL_TESTS", "") != "1"

if parallel && Sys.CPU_THREADS >= 2
    cov_level = Base.JLOptions().code_coverage
    println("Running $(length(TEST_GROUPS)) test groups in parallel ($(Sys.CPU_THREADS) threads available)")
    println("Code coverage level: $cov_level (0=none, 1=user, 2=all)")
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

        @testset "Arias et al. (2018) SVAR Identification" begin
            include("var/test_arias2018.jl")
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
    end
end
