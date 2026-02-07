using MacroEconometricModels
using Documenter

DocMeta.setdocmeta!(MacroEconometricModels, :DocTestSetup, :(using MacroEconometricModels); recursive=true)

makedocs(;
    modules=[MacroEconometricModels],
    authors="Wookyung Chung <mirimtl@protonmail.com>",
    repo="https://github.com/chung9207/MacroEconometricModels.jl/blob/{commit}{path}#{line}",
    sitename="MacroEconometricModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chung9207.github.io/MacroEconometricModels.jl",
        edit_link="main",
        assets=String[],
        size_threshold=350 * 1024,
        mathengine=Documenter.MathJax3(),
        repolink="https://github.com/chung9207/MacroEconometricModels.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Univariate Models" => [
            "ARIMA" => "arima.md",
            "Volatility Models" => "volatility.md",
        ],
        "Frequentist Models" => [
            "VAR" => "manual.md",
            "VECM" => "vecm.md",
            "Local Projections" => "lp.md",
            "Factor Models" => "factormodels.md",
        ],
        "Bayesian Models" => [
            "Bayesian VAR" => "bayesian.md",
        ],
        "Innovation Accounting" => "innovation_accounting.md",
        "Non-Gaussian Structural Identification" => "nongaussian.md",
        "Hypothesis Tests" => [
            "Unit Root & Cointegration" => "hypothesis_tests.md",
        ],
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
    repo="github.com/chung9207/MacroEconometricModels.jl",
    devbranch="main",
)
