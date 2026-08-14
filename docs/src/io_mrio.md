# [MRIO Trade Accounting](@id io_mrio_page)

Multi-regional input-output (MRIO) tables record intermediate and final flows between every pair of regions. Once `IOData` carries more than one region, the same Leontief inverse that powers classical multipliers also identifies **bilateral trade**, the **import content of exports**, and a complete **value-added decomposition of gross exports**. This page implements the Hummels–Ishii–Yi (2001) vertical-specialization measures and the Koopman–Wang–Wei (2014) nine-term export decomposition, together with region/sector aggregation and regional production-versus-consumption footprints. See [Input-Output Analysis](@ref io_page) for the `IOData` container and [Downloading Data](@ref io_download_page) for the public MRIO databases.

- **Aggregation**: `aggregate` collapses regions and sector types while carrying satellite accounts along
- **Block accessors**: `region_block`, `bilateral_trade`, and `gross_exports` read the inter-country blocks of ``Z`` and ``Y``
- **Vertical specialization**: Hummels–Ishii–Yi import content of exports, with the KWW multi-country generalisation
- **Export decomposition**: Koopman–Wang–Wei (2014) DVA / RDV / FVA / PDC breakdown of gross exports
- **Regional footprints**: production-based versus consumption-based stressor accounts by region

```@setup io_mrio
using MacroEconometricModels, LinearAlgebra

# KWW (2014) Section 2.4 Example 1 — hard-coded two-country, one-sector table.
# Country A: GO=150, domestic intermediate 50, VA 100; exports 50 intermediate + 20 final to B.
# Country B: GO=50, imports 50 intermediate from A, adds no VA; re-exports all 50 as final to A.
Z = [50.0 50.0; 0.0 0.0]
Y = [30.0 20.0; 50.0 0.0]
va = reshape([100.0, 0.0], 1, 2)
io = IOData(Z, Y, va;
            sectors=["A_goods", "B_goods"],
            regions=["A", "B"],
            fd_cats=["A_fd", "B_fd"],
            va_cats=["VA"])
```

## Quick Start

**Recipe 1: Bilateral intermediate and final trade**

```@example io_mrio
bilateral_trade(io, "A", "B")
```

**Recipe 2: Vertical specialization (import content of exports)**

```@example io_mrio
vertical_specialization(io, "B")
```

**Recipe 3: KWW (2014) gross-export decomposition**

```@example io_mrio
export_decomposition(io, "A")
```

**Recipe 4: Aggregate two regions into one**

```@example io_mrio
aggregate(io; region_map=Dict("A" => "World", "B" => "World"))
```

**Recipe 5: Regional production vs consumption footprint**

```@example io_mrio
add_extension!(io, "CO2", [10.0 0.0]; stressors=["CO2"], unit=["kt"])
footprint(io, "CO2"; by=:region)
```

---

## Block Layout

An MRIO table with ``G`` regions and ``N`` sectors per region stores ``n = G \cdot N`` industries in **region-major** order: region ``r`` occupies rows and columns

```math
(r-1)N+1,\; \ldots,\; r N.
```

The intermediate-flow matrix ``Z`` is the full ``n \times n`` block matrix whose off-diagonal blocks ``Z^{rs}`` are intermediate exports from ``r`` to ``s``. Final demand ``Y`` is region-blocked by destination whenever ``n_{fd}`` is divisible by ``G``: the columns for destination region ``r`` are

```math
(r-1)n_{fd/r}+1,\; \ldots,\; r\, n_{fd/r},
```

with ``n_{fd/r} = n_{fd}/G``. Under that convention bilateral final trade is identified; otherwise all final demand is treated as domestic and final-export terms in the KWW decomposition are zero.

```@example io_mrio
region_block(io, "A", "B")
```

`region_block(io, r, s)` returns the ``N \times N`` intermediate block from supplying region ``r`` to using region ``s``. Arguments accept region names or 1-based indices.

---

## Bilateral Trade and Gross Exports

`bilateral_trade` sums intermediate and final flows from an exporter to an importer.

```math
E^{rs} = Z^{rs}\mathbf{1} + Y^{r \to s}\mathbf{1}
```

where:
- ``Z^{rs}\mathbf{1}`` is the ``N``-vector of intermediate exports by sector
- ``Y^{r \to s}\mathbf{1}`` is the ``N``-vector of final-goods exports (zero when final demand is not region-blocked)
- the returned scalars `intermediate`, `final`, and `total` are the economy-wide sums

```@example io_mrio
bt = bilateral_trade(io, "A", "B")
(bt.intermediate, bt.final, bt.total)
```

Country A ships 50 of intermediate goods and 20 of final goods to B, for gross bilateral exports of 70 — the numbers of KWW's Example 1.

```@example io_mrio
gross_exports(io, "A")
```

`gross_exports(io, region)` returns the ``N``-vector of sectoral exports to all other regions. For a single-region table it is zero: a closed economy has no exports.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `kind` | `Symbol` | `:total` | `:total`, `:intermediate`, or `:final` |

### Return Values

`bilateral_trade` returns a named tuple:

| Field | Type | Description |
|-------|------|-------------|
| `intermediate` | `Float64` | Intermediate exports |
| `final` | `Float64` | Final-goods exports |
| `total` | `Float64` | Sum of the two |
| `by_sector` | `Vector{Float64}` | ``N``-vector of sectoral gross exports |

---

## Aggregate

`aggregate` collapses an MRIO table over regions and/or sector types. Both maps are optional; unmapped labels keep their names. Satellite accounts are summed along the industry axis (and along destination-region final-demand blocks of `F_Y` when final demand is region-blocked).

```@example io_mrio
world = aggregate(io; region_map=Dict("A" => "World", "B" => "World"))
report(world)
```

Collapsing A and B into a single region zeroes out international trade: the former intermediate export of 50 from A to B becomes a domestic intermediate flow inside World. Gross output is conserved — `sum(world.x) == sum(io.x)`.

```@example io_mrio
sum(world.x), sum(io.x)
```

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `region_map` | `Dict` | `nothing` | Old region name → new region name |
| `sector_map` | `Dict` | `nothing` | Old sector-type name (first-block labels) → new name |

### Return Values

A new [`IOData`](@ref) with the aggregated blocks, labels, and extensions. Accounting identities are re-checked at construction.

---

## Vertical Specialization

Hummels, Ishii & Yi (2001) measure **vertical specialization** as the foreign content of a country's exports — imported intermediates that are re-exported, directly or embodied in further processing. In a multi-country ICIO table the natural generalisation is Koopman–Wang–Wei (2014, eq. 38): foreign value-added plus pure foreign double counting in the region's gross exports.

```math
\mathrm{VS}_s = \sum_{t \neq s} V_t B_{ts} E^{s*}
```

where:
- ``V_t`` is the row vector of direct value-added coefficients of region ``t``
- ``B = (I - A)^{-1}`` is the full multi-regional Leontief inverse
- ``B_{ts}`` is the block of ``B`` from industries of ``t`` to industries of ``s``
- ``E^{s*}`` is the ``N``-vector of sectoral gross exports of ``s``

Domestic content is the complement: ``\mathrm{DC}_s = E^{s*} - \mathrm{VS}_s``. The indirect measure VS1 counts home intermediates embodied in foreign exports,

```math
\mathrm{VS1}_s = V_s \sum_{r \neq s} B_{sr} E^{r*}.
```

```@example io_mrio
vsB = vertical_specialization(io, "B")
report(vsB)
```

Country B's entire export of 50 is foreign content — it adds no domestic value added — so the VS share is 1. Country A's exports contain no foreign intermediates, so its VS share is 0 and its domestic-content share is 1.

```@example io_mrio
vsA = vertical_specialization(io, "A")
(vsA.vs_share, vsA.dc_share, vsA.vs1)
```

A's VS1 of 50 is exactly the intermediate export that B re-exports back as final goods: home intermediates used in foreign exports.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `vs` | `Float64` | Foreign content in exports |
| `vs_share` | `Float64` | ``\mathrm{VS} / E^{s*}`` |
| `vs1` | `Float64` | Indirect VS (home content in foreign exports) |
| `domestic_content` | `Float64` | ``E^{s*} - \mathrm{VS}`` |
| `dc_share` | `Float64` | Domestic-content share |
| `gross_exports` | `Float64` | Total gross exports of the region |
| `region` | `String` | Region name |
| `by_sector` | `Vector{Float64}` | Sectoral foreign content |

---

## KWW Export Decomposition

Koopman, Wang & Wei (2014) break a country's gross exports into nine value-added and double-counted terms that sum exactly to official gross exports. The package reports the four standard aggregates:

| Aggregate | Meaning | KWW (36) terms |
|-----------|---------|----------------|
| **DVA** | Domestic value-added absorbed abroad (value-added exports) | 1–3 |
| **RDV** | Domestic value-added that returns home and is consumed there | 4–5 |
| **FVA** | Foreign value-added embodied in exports | 7–8 |
| **PDC** | Pure double counting from two-way intermediate trade | 6, 9 |

```math
\mathrm{DVA} + \mathrm{RDV} + \mathrm{FVA} + \mathrm{PDC} = E^{s*}
```

The first three DVA terms further split value-added exports by absorption route: final-goods exports, intermediates absorbed by the direct importer, and intermediates re-exported to third countries. RDV is part of the source country's GDP but is double-counted in gross trade statistics; FVA is foreign GDP in the source's exports; PDC vanishes when at least one country does not export intermediates.

```@example io_mrio
edA = export_decomposition(io, "A")
report(edA)
```

In KWW's Example 1, of A's gross exports of 70, twenty units of domestic value-added are absorbed in B (the final-goods export) and fifty return home embedded in B's re-exports. Foreign value-added and pure double counting are both zero because B exports no intermediates.

```@example io_mrio
edB = export_decomposition(io, "B")
report(edB)
```

B's entire export of 50 is foreign value-added — A's intermediate that B re-labels as a final good — so DVA, RDV, and PDC are zero and the VAX ratio is zero.

The Johnson–Noguera VAX ratio is `dva / gross_exports`. It equals the domestic-content share only when RDV and PDC are zero; otherwise it understates the domestic value-added actually present in exports (KWW's central conceptual point).

```@example io_mrio
(edA.vax_ratio, edA.dva / edA.gross_exports, vsA.dc_share)
```

A's VAX ratio of ``20/70 \approx 0.29`` is far below its domestic-content share of 1, because fifty units of domestic value-added return home.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `dva` | `Float64` | Domestic VA absorbed abroad |
| `rdv` | `Float64` | Returned domestic VA |
| `fva` | `Float64` | Foreign VA in exports |
| `pdc` | `Float64` | Pure double counting |
| `gross_exports` | `Float64` | ``E^{s*}`` |
| `vax_ratio` | `Float64` | ``\mathrm{DVA}/E^{s*}`` |
| `region` | `String` | Region name |
| `terms` | `Vector{Float64}` | Nine individual KWW (36) terms |
| `by_sector` | `Matrix{Float64}` | ``N \times 4`` sectoral (DVA, RDV, FVA, PDC) |
| `sectors` | `Vector{String}` | Sector-type labels |

```julia
plot_result(edA)
```

---

## Regional Footprints

`footprint(io, name; by=:region)` reports the territorial (production-based) and consumption-based accounts of a satellite extension for every region.

```math
F^{\mathrm{prod}}_r = F_{\cdot, I_r}\mathbf{1}, \qquad
F^{\mathrm{cb}}_r = M Y^{\cdot r} + F_Y^{\cdot r}
```

where:
- ``I_r`` is the set of industries in region ``r``
- ``M = S L`` is the emission-multiplier matrix
- ``Y^{\cdot r}`` is final demand absorbed in region ``r`` (the destination block of ``Y``)
- ``F_Y^{\cdot r}`` is direct stressor use in final demand of region ``r``

```@example io_mrio
rfp = footprint(io, "CO2"; by=:region)
report(rfp)
```

Production-based emissions equal the columns of ``F`` summed within each region. Consumption-based emissions charge the global supply chain to the consuming region. With ``F_Y = 0`` the two accounts sum to the same global total; they differ region by region precisely when trade carries embodied emissions across borders.

```@example io_mrio
(sum(rfp.production), sum(rfp.consumption))
```

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `production` | `Matrix{Float64}` | Stressor × region territorial emissions |
| `consumption` | `Matrix{Float64}` | Stressor × region consumption-based footprint |
| `stressors` | `Vector{String}` | Stressor labels |
| `regions` | `Vector{String}` | Region labels |
| `name` | `String` | Extension name |

```julia
plot_result(rfp)
```

---

## Complete Example

The complete workflow on KWW's Example 1: inspect blocks, measure vertical specialization, decompose gross exports, and compare regional carbon accounts.

```@example io_mrio
tbl = IOData([50.0 50.0; 0.0 0.0],
             [30.0 20.0; 50.0 0.0],
             reshape([100.0, 0.0], 1, 2);
             sectors=["A_goods", "B_goods"],
             regions=["A", "B"],
             fd_cats=["A_fd", "B_fd"],
             va_cats=["VA"])

region_block(tbl, "A", "B")
```

```@example io_mrio
bilateral_trade(tbl, "A", "B")
```

```@example io_mrio
report(vertical_specialization(tbl, "A"))
```

```@example io_mrio
report(vertical_specialization(tbl, "B"))
```

```@example io_mrio
report(export_decomposition(tbl, "A"))
```

```@example io_mrio
report(export_decomposition(tbl, "B"))
```

```@example io_mrio
add_extension!(tbl, "CO2", [80.0 0.0]; stressors=["CO2"], unit=["kt"])
report(footprint(tbl, "CO2"; by=:region))
```

A produces all 80 kt of CO₂ but consumes only the share embodied in its domestic final demand plus the re-imported final goods from B; B produces nothing yet bears a positive consumption footprint through the intermediates it processes and the finals it absorbs. The two consumption figures sum to 80, matching the global production inventory.

---

## Common Pitfalls

1. **Final demand must be region-blocked for final-export terms.** When `size(Y, 2)` is not divisible by `nregions`, the package treats all final demand as domestic. Intermediate trade is still identified from ``Z``, but DVA in final exports and FVA in final exports are zero by construction. Build ``Y`` with ``G`` destination blocks (as OECD ICIO and WIOD do) to unlock the full decomposition.

2. **Sector labels have length ``G \cdot N``, not ``N``.** `io.sectors` stores one label per industry in the stacked table. `nsectors(io)` returns sectors *per region*. `sector_map` in `aggregate` keys off the first region's labels (the sector types) and applies them uniformly across all regions.

3. **VAX is not domestic content.** The VAX ratio `dva / gross_exports` excludes returned domestic value-added (RDV) and pure double counting. Use `vertical_specialization` for the domestic-content share, or `dva + rdv` for domestic value-added in exports (KWW's ``DV_s``).

4. **Single-region tables have nothing to decompose.** `export_decomposition` and `vertical_specialization` on a one-region table return zeros: there are no foreign blocks. Load an MRIO from [Downloading Data](@ref io_download_page) or construct a multi-region `IOData` as in the examples above.

5. **`aggregate` re-checks accounting identities.** If the source table only balances to a loose tolerance, aggregation can push residuals over the constructor's threshold. Pass a table that already balances, or build the source with `check=false` only when you accept the imbalance.

6. **Regional footprints require the extension to be attached first.** `footprint(io, name; by=:region)` throws if `name` is missing from `io.extensions`, with the same message as the sector-level footprint.

---

## API Reference

```@docs
aggregate
region_block
region_indices
bilateral_trade
gross_exports
vertical_specialization
VerticalSpecialization
export_decomposition
ExportDecomposition
```

---

## References

- Hummels, D., J. Ishii, and K.-M. Yi. 2001. "The Nature and Growth of Vertical Specialization in World Trade." *Journal of International Economics* 54 (1): 75–96. [https://doi.org/10.1016/S0022-1996(00)00093-3](https://doi.org/10.1016/S0022-1996(00)00093-3)
- Koopman, R., Z. Wang, and S.-J. Wei. 2014. "Tracing Value-Added and Double Counting in Gross Exports." *American Economic Review* 104 (2): 459–494. [https://doi.org/10.1257/aer.104.2.459](https://doi.org/10.1257/aer.104.2.459)
- Miller, R. E., and P. D. Blair. 2009. *Input-Output Analysis: Foundations and Extensions*. 2nd ed. Cambridge: Cambridge University Press. [https://doi.org/10.1017/CBO9780511626982](https://doi.org/10.1017/CBO9780511626982)
