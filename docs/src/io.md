# Input-Output Analysis

The Input-Output (IO) module provides tools for demand- and supply-driven IO
analysis, environmental extensions, the Baqaee & Farhi (2019) nonlinear-IO
decomposition, and automated downloading of public multi-regional IO (MRIO)
databases.

This section is split into five pages:

- **Overview** (this page) — the [`IOData`](@ref) container, the accounting
  identities, and loading the built-in example.
- [Classical Analysis](io_classical.md) — Leontief/Ghosh inverses, multipliers,
  linkages, key sectors, structural decomposition, and hypothetical extraction.
- [Environmental Extensions](io_environmental.md) — satellite accounts,
  intensities, emission multipliers, and footprints.
- [Baqaee & Farhi (2019)](io_baqaee_farhi.md) — Domar weights, Hulten's theorem,
  the second-order term, and network centralities.
- [Downloading Data](io_download.md) — `download_io` and the per-source
  downloaders, plus parsing archives into `IOData`.

## The `IOData` container

An `IOData` table collects the intermediate-flow matrix `Z`, final demand `Y`,
value added `va`, and gross output `x`, together with sector/region labels and
optional satellite accounts. It is multi-region aware (`regions` has length one
for a single-region table).

The fundamental accounting identities are the row balance

```math
x_i = \sum_j Z_{ij} + \sum_k Y_{ik}
```

and the column balance

```math
x_j = \sum_i Z_{ij} + \sum_v va_{vj}.
```

By default the constructor verifies both with a relaxed tolerance; pass
`check=false` to skip validation (e.g. for stylized examples).

## Built-in example

`load_example(:wiot)` returns a small, license-clean teaching table — the
Miller & Blair (2009) hypothetical two-sector economy — with two value-added
categories, one final-demand column, and `employment` and `CO2` satellite
accounts.

```@example io
using MacroEconometricModels

io = load_example(:wiot)
report(io)
```

The technical-coefficients matrix and gross output are then one call away:

```@example io
technical_coefficients(io)
```

```@example io
io.x          # gross output by sector
```

## Constructing an `IOData` directly

```@example io
Z  = [150.0 500.0; 200.0 100.0]
Y  = reshape([350.0, 1700.0], 2, 1)          # one final-demand column
va = [300.0 1000.0; 350.0 400.0]             # two value-added categories
tbl = IOData(Z, Y, va; sectors=["Agriculture", "Manufacturing"],
             fd_cats=["final_demand"], va_cats=["compensation", "other_va"])
tbl.x
```

## API

```@docs
IOData
```
