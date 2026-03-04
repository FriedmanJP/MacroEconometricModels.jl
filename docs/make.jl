using MacroEconometricModels
using Documenter

DocMeta.setdocmeta!(MacroEconometricModels, :DocTestSetup, :(using MacroEconometricModels); recursive=true)

makedocs(;
    modules=[MacroEconometricModels],
    authors="Wookyung Chung <chung@friedman.jp>",
    repo="https://github.com/FriedmanJP/MacroEconometricModels.jl/blob/{commit}{path}#{line}",
    sitename="MacroEconometricModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://FriedmanJP.github.io/MacroEconometricModels.jl",
        edit_link="main",
        assets=["assets/custom.css", "assets/theme-toggle.js"],
        size_threshold=700 * 1024,
        mathengine=Documenter.MathJax3(),
        repolink="https://github.com/FriedmanJP/MacroEconometricModels.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Data Management" => "data.md",
        "Univariate Models" => [
            "Time Series Filters" => "filters.md",
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
        "Panel Models" => [
            "Panel VAR" => "pvar.md",
            "Difference-in-Differences" => "did.md",
            "Event Study LP" => "event_study.md",
        ],
        "DSGE Models" => [
            "Overview" => "dsge.md",
            "Linear Solvers" => "dsge_linear.md",
            "Nonlinear Methods" => "dsge_nonlinear.md",
            "Constraints" => "dsge_constraints.md",
            "Estimation" => "dsge_estimation.md",
        ],
        "Innovation Accounting" => "innovation_accounting.md",
        "Nowcasting" => "nowcast.md",
        "Statistical Identification" => "nongaussian.md",
        "Hypothesis Tests" => [
            "Overview" => "tests.md",
            "Unit Root & Cointegration" => "tests_unitroot.md",
            "Advanced Unit Root" => "tests_unitroot_advanced.md",
            "Structural Breaks" => "tests_breaks.md",
            "Panel Tests" => "tests_panel.md",
            "Model Diagnostics" => "tests_diagnostics.md",
        ],
        "Visualization" => "plotting.md",
        "Examples" => "examples.md",
        "API Reference" => [
            "Overview" => "api.md",
            "Types" => "api_types.md",
            "Functions" => "api_functions.md",
        ],
    ],
    checkdocs=:exports,
    warnonly=[:missing_docs, :cross_references, :autodocs_block, :docs_block],
)

deploydocs(;
    repo="github.com/FriedmanJP/MacroEconometricModels.jl",
    devbranch="main",
)
