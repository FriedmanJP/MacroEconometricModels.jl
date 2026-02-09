"""
Example dataset loader stub for MacroEconometricModels.jl.
"""

"""
    load_example(name::Symbol)

Load a built-in example dataset. Currently a stub — example datasets will be
available in a future release.

# Planned Datasets
- `:sw2001` — Stock & Watson (2001) macro dataset
- `:ggr2005` — Giannone, Gambetti & Reichlin (2005) VAR dataset
- `:fred_md` — FRED-MD monthly macroeconomic database format

# Examples
```julia
load_example(:sw2001)  # throws ArgumentError (not yet available)
```
"""
function load_example(name::Symbol)
    throw(ArgumentError("Dataset :$name not yet available. Example datasets coming in a future release."))
end
