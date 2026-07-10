using MacroEconometricModels
using Documenter

DocMeta.setdocmeta!(MacroEconometricModels, :DocTestSetup, :(using MacroEconometricModels); recursive=true)

makedocs(;
    # Draft mode skips execution of every @setup/@example block. It must be opt-in for
    # local iteration only (DOCS_DRAFT=true) — never committed — so CI and any plain
    # make.jl run execute examples and render real output.
    draft = get(ENV, "DOCS_DRAFT", "false") == "true",
    modules=[MacroEconometricModels],
    authors="Wookyung Chung <chung@friedman.jp>",
    repo="https://github.com/FriedmanJP/MacroEconometricModels.jl/blob/{commit}{path}#{line}",
    sitename="MacroEconometricModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://FriedmanJP.github.io/MacroEconometricModels.jl",
        edit_link="main",
        assets=["assets/custom.css", "assets/theme-toggle.js"],
        size_threshold=900 * 1024,
        mathengine=Documenter.MathJax3(),
        repolink="https://github.com/FriedmanJP/MacroEconometricModels.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Data Management" => "data.md",
        "Univariate Models" => [
            "Time Series Filters" => "filters.md",
            "X-13ARIMA-SEATS" => "x13.md",
            "Spectral Analysis" => "spectral.md",
            "ARIMA" => "arima.md",
            "Volatility Models" => "volatility.md",
        ],
        "Multivariate Models" => [
            "VAR" => "manual.md",
            "Bayesian VAR" => "bayesian.md",
            "VECM" => "vecm.md",
            "Local Projections" => "lp.md",
            "Factor Models" => "factormodels.md",
            "FAVAR" => "favar.md",
        ],
        "Cross-Sectional Models" => [
            "Linear Regression" => "regression.md",
            "Binary Choice Models" => "binary_choice.md",
            "Ordered & Multinomial" => "ordered_multinomial.md",
        ],
        "Panel Models" => [
            "Panel VAR" => "pvar.md",
            "Panel Regression" => "panel_reg.md",
            "Difference-in-Differences" => "did.md",
            "Event Study LP" => "event_study.md",
        ],
        "DSGE Models" => [
            "Overview" => "dsge.md",
            "Linear Solvers" => "dsge_linear.md",
            "Nonlinear Methods" => "dsge_nonlinear.md",
            "Constraints" => "dsge_constraints.md",
            "Estimation" => "dsge_estimation.md",
            "Historical Decomposition" => "dsge_hd.md",
            "Heterogeneous Agents" => "dsge_ha.md",
            "Overlapping Generations" => "dsge_olg.md",
            "Continuous Time" => "dsge_continuous.md",
        ],
        "Input-Output Analysis" => [
            "Overview" => "io.md",
            "Classical Analysis" => "io_classical.md",
            "Environmental Extensions" => "io_environmental.md",
            "Baqaee & Farhi (2019)" => "io_baqaee_farhi.md",
            "Downloading Data" => "io_download.md",
        ],
        "Innovation Accounting" => [
            "Overview" => "innovation_accounting.md",
            "Impulse Responses" => "ia_irf.md",
            "Variance Decomposition" => "ia_fevd.md",
            "Historical Decomposition" => "ia_hd.md",
        ],
        "Structural Identification" => "structural_identification.md",
        "Statistical Identification" => [
            "Overview" => "nongaussian.md",
            "Non-Gaussian Methods" => "id_nongaussian.md",
            "Heteroskedasticity" => "id_heteroskedastic.md",
            "Testing" => "id_testing.md",
        ],
        "Nowcasting" => [
            "Overview" => "nowcast.md",
            "DFM Nowcasting" => "nowcast_dfm.md",
            "BVAR Nowcasting" => "nowcast_bvar.md",
            "Bridge Equations" => "nowcast_bridge.md",
            "News Decomposition" => "nowcast_news.md",
        ],
        "Hypothesis Tests" => [
            "Overview" => "tests.md",
            "Unit Root & Cointegration" => "tests_unitroot.md",
            "Advanced Unit Root" => "tests_unitroot_advanced.md",
            "Structural Breaks" => "tests_breaks.md",
            "Panel Tests" => "tests_panel.md",
            "Model Diagnostics" => "tests_diagnostics.md",
        ],
        "Visualization" => "plotting.md",
        "API Reference" => [
            "Overview" => "api.md",
            "Types" => "api_types.md",
            "Functions" => "api_functions.md",
        ],
    ],
    checkdocs=:exports,
    # example/setup/docs/autodocs block failures must FAIL the build (docrule: "@example
    # blocks MUST run"). :missing_docs stays until the docstring backlog clears ([T202]);
    # :cross_references deferred to the docs-consistency stage (xref burn-down not yet done).
    warnonly=[:missing_docs, :cross_references],
)

# Since v0.6.x docs are NOT built or deployed by CI (Documentation.yml removed
# 2026-07-10). gh-pages is updated manually with docs/deploy_local.jl, which builds
# this site and maintains the version folders/symlinks/versions.js itself — so no
# deploydocs() call here.
