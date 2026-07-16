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
        "Getting Started" => [
            "Installation & First Model" => "getting_started.md",
            "Choosing a Method" => "method_guide.md",
            "Data Management" => "data.md",
        ],
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
            "ARDL & Bounds Test" => "ardl.md",
            "Cointegrating Regression" => "cointreg.md",
            "Local Projections" => "lp.md",
            "Factor Models" => "factormodels.md",
            "FAVAR" => "favar.md",
        ],
        "Nonlinear Time Series" => [
            "Threshold & SETAR" => "nonlinear.md",
        ],
        "Cross-Section & Panel" => [
            "Linear Regression" => "regression.md",
            "Binary Choice Models" => "binary_choice.md",
            "Ordered & Multinomial" => "ordered_multinomial.md",
            "Panel Regression" => "panel_reg.md",
            "Panel VAR" => "pvar.md",
            "Difference-in-Differences" => "did.md",
            "Event Study LP" => "event_study.md",
        ],
        "GMM & SMM" => "gmm.md",
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
        "Structural Analysis" => [
            "Structural Identification" => "structural_identification.md",
            "Statistical Identification" => [
                "Overview" => "nongaussian.md",
                "Non-Gaussian Methods" => "id_nongaussian.md",
                "Heteroskedasticity" => "id_heteroskedastic.md",
                "Testing" => "id_testing.md",
            ],
            "Innovation Accounting" => [
                "Overview" => "innovation_accounting.md",
                "Impulse Responses" => "ia_irf.md",
                "Variance Decomposition" => "ia_fevd.md",
                "Historical Decomposition" => "ia_hd.md",
            ],
        ],
        "Nowcasting" => [
            "Overview" => "nowcast.md",
            "DFM Nowcasting" => "nowcast_dfm.md",
            "BVAR Nowcasting" => "nowcast_bvar.md",
            "Bridge Equations" => "nowcast_bridge.md",
            "News Decomposition" => "nowcast_news.md",
            "MIDAS Regression" => "midas.md",
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
        "Project" => [
            "Notation" => "notation.md",
            "Bibliography" => "bibliography.md",
            "Changelog" => "changelog.md",
            "How to Cite" => "citation.md",
        ],
        "API Reference" => [
            "Overview" => "api.md",
            "Data Management" => "api/data.md",
            "Univariate Models" => "api/univariate.md",
            "Multivariate Models" => "api/multivariate.md",
            "Cross-Sectional Models" => "api/cross_section.md",
            "Panel Models" => "api/panel.md",
            "DSGE Models" => "api/dsge.md",
            "Structural & Statistical Identification" => "api/structural.md",
            "GMM & SMM" => "api/gmm.md",
            "Hypothesis Tests" => "api/tests.md",
            "Nowcasting" => "api/nowcasting.md",
            "Visualization" => "api/visualization.md",
            "Utilities & Display" => "api/utilities.md",
        ],
    ],
    checkdocs=:exports,
    # example/setup/docs/autodocs block failures must FAIL the build (docrule: "@example
    # blocks MUST run"). Every exported docstring is now registered on a reference page
    # ([T202]/[T177]/[T178]), so :missing_docs is a hard error. :cross_references is
    # deferred to the docs-consistency stage (xref burn-down not yet done).
    warnonly=[:cross_references],
)

# Since v0.6.x docs are NOT built or deployed by CI (Documentation.yml removed
# 2026-07-10). gh-pages is updated manually with docs/deploy_local.jl, which builds
# this site and maintains the version folders/symlinks/versions.js itself — so no
# deploydocs() call here.
