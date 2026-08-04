# [Input-Output Analysis](@id io_page)

Input-output analysis reads the inter-industry flow table of an economy and turns it into multipliers, linkages, footprints, and productivity elasticities. The package implements the demand-driven model of Leontief (1936) and the supply-driven model of Ghosh (1958) over a single `IOData` container, extends both with satellite accounts for emissions and employment, adds the Baqaee & Farhi (2019) nonlinear production-network decomposition, and downloads the major public multi-regional input-output (MRIO) databases.

- **Container**: `IOData` stores intermediate flows, final demand, value added, gross output, sector and region labels, and satellite accounts, and validates the two accounting identities at construction
- **Classical analysis**: Leontief and Ghosh inverses, Type I and Type II multipliers, Rasmussen linkages and key sectors, structural decomposition, and hypothetical extraction
- **Environmental extensions**: stressor intensities, consumption-based emission multipliers, and footprints for any satellite account
- **Production networks**: Domar weights, Hulten's theorem, the second-order "beyond Hulten" Hessian, and network centralities
- **Data acquisition**: OECD ICIO, WIOD, EXIOBASE 3, EORA26, and GLORIA downloaders, plus CSV, TSV, ZIP, and XLSX parsing
- **Display**: `report()` for every result type and `plot_result()` for multipliers, linkages, and the Leontief inverse

```@setup io_hub
using MacroEconometricModels
```

## Quick Start

**Recipe: Output multipliers from a published table**

```@example io_hub
io = load_example(:wiot)
multipliers(io; kind=:output, type=:I)
```

One additional unit of final demand for agricultural output raises gross output across the whole economy by 1.518 units: the one unit of agriculture itself plus 0.518 units of intermediate production pulled in along the supply chain. Manufacturing's multiplier is 1.452, so agriculture is the more input-intensive of the two sectors even though it is the smaller one. Everything else on the child pages builds on the same table and the same Leontief inverse.

---

## Choosing a Method

| Research question | Function | Page |
|-------------------|----------|------|
| How much output does one unit of final demand require? | `leontief_inverse` | [Classical Analysis](@ref io_classical_page) |
| Which sectors generate the largest output, income, or job effects? | `multipliers` | [Classical Analysis](@ref io_classical_page) |
| Which sectors are key sectors? | `linkages`, `key_sectors` | [Classical Analysis](@ref io_classical_page) |
| What drove the change in output between two years? | `sda` | [Classical Analysis](@ref io_classical_page) |
| How much of the economy depends on one sector? | `hypothetical_extraction` | [Classical Analysis](@ref io_classical_page) |
| Who supplies inputs to whom, in value terms? | `ghosh`, `allocation_coefficients` | [Classical Analysis](@ref io_classical_page) |
| How much CO``_2`` is embodied in final demand? | `footprint` | [Environmental Extensions](@ref io_environmental_page) |
| How emission-intensive is each supply chain? | `emission_multipliers` | [Environmental Extensions](@ref io_environmental_page) |
| How does a sector's productivity shock move GDP? | `domar_weights`, `baqaee_farhi` | [Baqaee & Farhi (2019)](@ref io_baqaee_farhi_page) |
| Where does a real MRIO table come from? | `download_io`, `parse_io` | [Downloading Data](@ref io_download_page) |

Classical analysis and environmental extensions share the Leontief inverse and are linear in final demand. The Baqaee & Farhi decomposition drops that linearity: it treats the same table as the cost-share structure of a general-equilibrium production network in which sectors substitute across inputs.

---

## Child Pages

- [Classical Analysis](@ref io_classical_page) — Leontief and Ghosh inverses, Type I and Type II multipliers, Rasmussen linkages and key sectors, structural decomposition analysis, and hypothetical extraction
- [Environmental Extensions](@ref io_environmental_page) — satellite accounts, per-unit intensities, consumption-based emission multipliers, and footprints
- [Baqaee & Farhi (2019)](@ref io_baqaee_farhi_page) — Domar weights, Hulten's theorem, the second-order "beyond Hulten" term, and network centralities
- [Downloading Data](@ref io_download_page) — the source registry, the per-database downloaders, SHA-256 integrity verification, and parsing archives into `IOData`

---

## The `IOData` Container

Every function on the child pages takes an `IOData`. The container holds four numeric blocks — the intermediate-flow matrix ``Z``, the final-demand matrix ``Y``, the value-added matrix ``V``, and gross output ``x`` — together with the labels that make them interpretable and any satellite accounts attached to them.

The **row balance** states that each sector's gross output is fully absorbed, either as an intermediate input elsewhere or as final demand:

```math
x_i = \sum_{j=1}^{n} Z_{ij} + \sum_{k=1}^{n_{fd}} Y_{ik}
```

The **column balance** states that each sector's gross output is fully paid out, either to its suppliers or to primary factors:

```math
x_j = \sum_{i=1}^{n} Z_{ij} + \sum_{v=1}^{n_{va}} V_{vj}
```

where:
- ``Z_{ij}`` is the flow of sector ``i``'s output used as an intermediate input by sector ``j``
- ``Y_{ik}`` is sector ``i``'s output delivered to final-demand category ``k``
- ``V_{vj}`` is value-added category ``v`` paid by sector ``j``
- ``x_i`` is gross output of sector ``i``
- ``n`` is the number of sectors, ``n_{fd}`` the number of final-demand categories, ``n_{va}`` the number of value-added categories

Summing either identity over all sectors gives the national accounting identity: total final demand equals total value added, and both equal GDP.

!!! note "Technical Note"
    Both balances are checked at construction with the relaxed tolerance
    ``|\text{balance}_i - x_i| \le 10^{-6}\max(1, |x_i|)``, which absorbs the rounding
    that published tables carry. Pass `check=false` for a stylized table that is not
    meant to balance exactly.

### Constructing a Table

Two constructors are available. The first takes value added and derives gross output from the row balance; the second takes gross output and derives a single value-added row from the column balance when `va` is not supplied.

```@example io_hub
Z  = [150.0 500.0; 200.0 100.0]              # intermediate flows
Y  = reshape([350.0, 1700.0], 2, 1)          # one final-demand category
va = [300.0 1000.0; 350.0 400.0]             # compensation and other value added

tbl = IOData(Z, Y, va; sectors=["Agriculture", "Manufacturing"],
             fd_cats=["final_demand"], va_cats=["compensation", "other_va"],
             unit="millions", year=2009)
report(tbl)
```

Gross output follows from the row balance, ``x = [1000, 2000]``, and the column balance holds because the value-added entries were chosen to close it. `size(tbl)` returns `(n, n_fd)`, here `(2, 1)`.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `sectors` | `Vector{String}` | `String[]` | Sector labels; generated as `sector1`, `sector2`, … when empty |
| `regions` | `Vector{String}` | `["total"]` | Region labels; length 1 for a single-region table |
| `fd_cats` | `Vector{String}` | `String[]` | Final-demand category labels; generated as `fd1`, … when empty |
| `va_cats` | `Vector{String}` | `String[]` | Value-added category labels; generated as `va1`, … when empty |
| `unit` | `String` | `""` | Monetary unit of the flows |
| `year` | `Union{Int,Nothing}` | `nothing` | Reference year |
| `source` | `String` | `""` | Provenance string shown by `report()` |
| `meta` | `IOMetaData` | `IOMetaData()` | Download log, populated by the downloaders |
| `check` | `Bool` | `true` | Validate the row and column balances |
| `va` | `AbstractMatrix` | `nothing` | Value added; second constructor only, derived from the column balance when omitted |

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `Z` | `Matrix{T}` | ``n \times n`` intermediate-flow matrix |
| `Y` | `Matrix{T}` | ``n \times n_{fd}`` final demand by category |
| `va` | `Matrix{T}` | ``n_{va} \times n`` value added by category |
| `x` | `Vector{T}` | ``n \times 1`` gross output |
| `sectors` | `Vector{String}` | Row and column labels, length ``n`` |
| `regions` | `Vector{String}` | Region labels |
| `fd_cats` | `Vector{String}` | Final-demand category labels |
| `va_cats` | `Vector{String}` | Value-added category labels |
| `extensions` | `Dict{String,IOExtension{T}}` | Satellite accounts keyed by name |
| `unit` | `String` | Monetary unit |
| `year` | `Union{Int,Nothing}` | Reference year |
| `source` | `String` | Provenance string |
| `meta` | `IOMetaData` | Download provenance log |

### Multi-Region Tables

The container is MRIO-aware. For a table covering ``R`` regions and ``S`` sectors, `regions` has length ``R`` and `sectors` carries all ``n = R \cdot S`` region-sector labels, so ``Z`` is the full ``n \times n`` block matrix of intra- and inter-regional flows. A single-region table simply has `regions = ["total"]`. Every function on the child pages operates on the stacked ``n``-dimensional table without special-casing regions.

---

## The Built-In Example

`load_example(:wiot)` returns the hypothetical two-sector economy of Miller & Blair (2009, Table 2.3) — a small, licence-clean teaching table with two value-added categories, one final-demand column, and `employment` and `CO2` satellite accounts already attached. Every example on the child pages uses it.

```@example io_hub
io = load_example(:wiot)
report(io)
```

```@example io_hub
technical_coefficients(io)
```

Agriculture buys 0.15 units of agricultural output and 0.20 units of manufacturing output per unit of its own output; manufacturing buys 0.25 and 0.05. Column sums of 0.35 and 0.30 leave 65 and 70 percent of each sector's output as value added, which is what makes the column balance close.

---

## API Reference

```@docs
IOData
```

---

## References

- Baqaee, D. R., & Farhi, E. (2019). The Macroeconomic Impact of Microeconomic Shocks: Beyond Hulten's Theorem.
  *Econometrica*, 87(4), 1155--1203. [DOI](https://doi.org/10.3982/ECTA15202)

- Ghosh, A. (1958). Input-Output Approach in an Allocation System.
  *Economica*, 25(97), 58--64. [DOI](https://doi.org/10.2307/2550694)

- Leontief, W. W. (1936). Quantitative Input and Output Relations in the Economic System of the United States.
  *The Review of Economics and Statistics*, 18(3), 105--125. [DOI](https://doi.org/10.2307/1927837)

- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions* (2nd ed.).
  Cambridge University Press. ISBN 978-0-521-51713-3. [DOI](https://doi.org/10.1017/CBO9780511626982)
