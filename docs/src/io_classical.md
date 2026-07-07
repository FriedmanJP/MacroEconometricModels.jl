# Classical IO Analysis

Demand-driven (Leontief) and supply-driven (Ghosh) analysis, multipliers,
linkages, key-sector identification, structural decomposition, and hypothetical
extraction. All examples use the built-in [`load_example`](@ref)`(:wiot)` table.

```@example ioc
using MacroEconometricModels
io = load_example(:wiot)
nothing # hide
```

## Leontief and Ghosh inverses

The **technical-coefficients** matrix is ``A = Z\hat{x}^{-1}`` and the
**Leontief inverse** (total requirements) is ``L = (I-A)^{-1}``. The
supply-side analogues are the **allocation-coefficients** matrix
``B = \hat{x}^{-1}Z`` and the **Ghosh inverse** ``G = (I-B)^{-1}``.

```@example ioc
A = technical_coefficients(io)
L = leontief_inverse(io)
L
```

```@example ioc
m = leontief(io)          # bundles A, L, x
g = ghosh(io)             # bundles B, G, x
g.G
```

The two inverses are linked by ``G = \hat{x}^{-1} L \hat{x}``.

## Multipliers

`multipliers` returns sectoral output, income, or employment multipliers, as
Type I (open model) or Type II (closed with respect to households).

```@example ioc
multipliers(io; kind=:output, type=:I)
```

```@example ioc
multipliers(io; kind=:income, type=:I)    # value-added multipliers
```

Type II multipliers close the model with respect to households (using the
compensation row as household income and final-demand shares as consumption),
adding induced effects:

```@example ioc
multipliers(io; kind=:output, type=:II)
```

## Linkages and key sectors

Backward linkages are column sums of `L`; forward linkages are row sums of the
Ghosh inverse `G` (or of `L` with `forward=:leontief`). The Rasmussen
power-of-dispersion (``U_i``) and sensitivity-of-dispersion (``U_j``) indices
normalize these by their overall average, and the `(U_i, U_j)` quadrants give a
key-sector classification.

```@example ioc
lk = linkages(io)
```

```@example ioc
key_sectors(io)           # :key / :forward / :backward / :weak per sector
```

## Structural decomposition analysis

`sda` decomposes the change in gross output between two periods into a
technology (`ΔL`) effect and a final-demand (`Δy`) effect. The additive
two-polar decomposition is exact (zero residual); a multiplicative variant is
also available.

```@example ioc
io2 = IOData(io.Z .* 1.1, io.Y .* 1.2, [330.0 1100.0; 385.0 440.0];
             sectors=io.sectors, check=false)
r = sda(io, io2; method=:additive)
r.effects
```

## Hypothetical extraction

`hypothetical_extraction` measures the output loss from removing one or more
sectors (by index or by name).

```@example ioc
hypothetical_extraction(io, "Agriculture")
```

## API

```@docs
technical_coefficients
leontief_inverse
allocation_coefficients
ghosh_inverse
leontief
ghosh
multipliers
linkages
rasmussen
key_sectors
sda
hypothetical_extraction
```
