using Test

# =============================================================================
# Parallel test runner: spawns independent Julia processes per test group
# =============================================================================

const TEST_GROUPS = [
    # Group 1: Core VAR + Bayesian (heavy MCMC)
    ("Core & Bayesian" => [
        "test_aqua.jl",
        "test_core_var.jl",
        "test_bayesian.jl",
        "test_samplers.jl",
        "test_bayesian_utils.jl",
        "test_minnesota.jl",
        "test_bgr.jl",
    ]),
    # Group 2: IRF, FEVD, HD, StatsAPI
    ("IRF & FEVD" => [
        "test_irf.jl",
        "test_irf_ci.jl",
        "test_statsapi.jl",
        "test_fevd.jl",
        "test_hd.jl",
        "test_summary.jl",
    ]),
    # Group 3: Factor models
    ("Factor Models" => [
        "test_factormodel.jl",
        "test_dynamicfactormodel.jl",
        "test_gdfm.jl",
        "test_factor_forecast.jl",
    ]),
    # Group 4: Local Projections
    ("Local Projections" => [
        "test_lp.jl",
        "test_lp_structural.jl",
        "test_lp_forecast.jl",
        "test_lp_fevd.jl",
    ]),
    # Group 5: ARIMA + Unit Root + GMM + Utilities
    ("ARIMA & Utilities" => [
        "test_unitroot.jl",
        "test_arima.jl",
        "test_utils.jl",
        "test_edge_cases.jl",
        "test_examples.jl",
        "test_gmm.jl",
        "test_covariance.jl",
    ]),
    # Group 6: Non-Gaussian + Display + Misc
    ("Non-Gaussian & Display" => [
        "test_normality.jl",
        "test_nongaussian_svar.jl",
        "test_display_backends.jl",
        "test_nongaussian_internals.jl",
        "test_error_paths.jl",
        "test_internal_helpers.jl",
        "test_arias2018.jl",
    ]),
    # Group 7: Volatility models (ARCH/GARCH/SV — MCMC heavy)
    ("Volatility Models" => [
        "test_volatility.jl",
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
    cov_opt = Base.JLOptions().code_coverage
    cov_flag = cov_opt == 1 ? `--code-coverage=user` :
               cov_opt == 2 ? `--code-coverage=all`  :
               ``
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
            include("test_aqua.jl")
        end

        @testset "Core VAR" begin
            include("test_core_var.jl")
        end

        @testset "Unit Root Tests" begin
            include("test_unitroot.jl")
        end

        @testset "Bayesian Estimation" begin
            include("test_bayesian.jl")
            include("test_samplers.jl")
            include("test_bayesian_utils.jl")
        end

        @testset "Impulse Response Functions" begin
            include("test_irf.jl")
            include("test_irf_ci.jl")
        end

        @testset "Minnesota Prior" begin
            include("test_minnesota.jl")
        end

        @testset "BGR Optimization" begin
            include("test_bgr.jl")
        end

        @testset "StatsAPI Compatibility" begin
            include("test_statsapi.jl")
        end

        @testset "FEVD" begin
            include("test_fevd.jl")
        end

        @testset "Historical Decomposition" begin
            include("test_hd.jl")
        end

        @testset "Summary Tables" begin
            include("test_summary.jl")
        end

        @testset "Factor Model" begin
            include("test_factormodel.jl")
        end

        @testset "Dynamic Factor Model" begin
            include("test_dynamicfactormodel.jl")
        end

        @testset "Generalized Dynamic Factor Model" begin
            include("test_gdfm.jl")
        end

        @testset "Factor Model Forecasting" begin
            include("test_factor_forecast.jl")
        end

        @testset "Arias et al. (2018) SVAR Identification" begin
            include("test_arias2018.jl")
        end

        @testset "Local Projections" begin
            include("test_lp.jl")
        end

        @testset "Structural LP" begin
            include("test_lp_structural.jl")
        end

        @testset "LP Forecasting" begin
            include("test_lp_forecast.jl")
        end

        @testset "LP-FEVD (Gorodnichenko & Lee 2019)" begin
            include("test_lp_fevd.jl")
        end

        @testset "ARIMA Models" begin
            include("test_arima.jl")
        end

        @testset "Utility Functions" begin
            include("test_utils.jl")
        end

        @testset "Edge Cases" begin
            include("test_edge_cases.jl")
        end

        @testset "Documentation Examples" begin
            include("test_examples.jl")
        end

        @testset "GMM Estimation" begin
            include("test_gmm.jl")
        end

        @testset "Covariance Estimators" begin
            include("test_covariance.jl")
        end

        @testset "Multivariate Normality Tests" begin
            include("test_normality.jl")
        end

        @testset "Non-Gaussian SVAR Identification" begin
            include("test_nongaussian_svar.jl")
        end

        @testset "Display Backend Switching" begin
            include("test_display_backends.jl")
        end

        @testset "Non-Gaussian Internals" begin
            include("test_nongaussian_internals.jl")
        end

        @testset "Error Paths" begin
            include("test_error_paths.jl")
        end

        @testset "Internal Helpers" begin
            include("test_internal_helpers.jl")
        end

        @testset "Volatility Models (ARCH/GARCH/SV)" begin
            include("test_volatility.jl")
        end
    end
end
