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
        mathengine=Documenter.MathJax3(),
        repolink="https://github.com/chung9207/MacroEconometricModels.jl",
    ),
    pages=[
        "Home" => "index.md",
        "by Subject" => [
            "VAR & BVAR" => "manual.md",
            "Local Projections" => "lp.md",
            "Factor Models" => "factormodels.md",
        ],
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:exports,
    warnonly=[:missing_docs, :cross_references, :autodocs_block, :docs_block],
)

deploydocs(;
    repo="github.com/chung9207/MacroEconometricModels.jl",
    devbranch="main",
)
