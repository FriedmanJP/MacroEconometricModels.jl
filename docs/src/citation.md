# [How to Cite](@id citation)

If you use MacroEconometricModels.jl in published research, please cite the software. Citing the
package supports its continued development and lets readers reproduce your workflow against a specific
version.

---

## Software Citation

Plain-text (AEA style):

> Chung, Wookyung. 2026. *MacroEconometricModels.jl: A Comprehensive Julia Package for
> Macroeconometric Research and Analysis*. Version 0.6.7.
> [https://github.com/FriedmanJP/MacroEconometricModels.jl](https://github.com/FriedmanJP/MacroEconometricModels.jl).

BibTeX:

```bibtex
@misc{chung2026macroeconometricmodels,
  author    = {Chung, Wookyung},
  title     = {{MacroEconometricModels.jl}},
  year      = {2026},
  doi       = {10.5281/zenodo.18439170},
  url       = {https://doi.org/10.5281/zenodo.18439170},
  publisher = {Zenodo}
}
```

The DOI resolves to the Zenodo record for the software; cite the version-specific DOI listed there
for the exact release you used (see [Changelog / What's New](@ref changelog) for the version history).

---

## Citing the Underlying Methods

Each estimator implements one or more published methods. Cite the original papers alongside the
software. The [Bibliography](@ref bibliography) collects every work referenced in the documentation,
and every method page carries its own topic-scoped reference list.

At runtime, `refs(model)` emits the references for the specific methods a fitted model uses, and
`refs(io, :method)` does the same for an Input-Output method. Both accept a `format` keyword returning
`:text`, `:latex`, `:bibtex`, or `:html`:

```julia
refs(model)                    # AEA plain-text (default)
refs(model; format = :bibtex)  # BibTeX entries
refs(io, :multipliers; format = :latex)
```

See the API Reference pages for the full `refs` signature.

---

## Package Metadata

| Field | Value |
|-------|-------|
| Author | Wookyung Chung <chung@friedman.jp> |
| Version | 0.6.7 |
| Repository | [github.com/FriedmanJP/MacroEconometricModels.jl](https://github.com/FriedmanJP/MacroEconometricModels.jl) |
| License | GPL-3.0-or-later |
