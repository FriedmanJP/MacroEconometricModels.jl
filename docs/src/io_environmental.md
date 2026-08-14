# [Environmental Extensions](@id io_environmental_page)

An environmentally extended input-output model attaches physical flows — tonnes of CO``_2``, jobs, cubic metres of water, hectares of land — to the monetary table and pushes them through the Leontief inverse. The result is a **consumption-based** account: emissions are charged to whoever ultimately buys the product rather than to whoever burns the fuel. Leontief (1970) introduced the construction; Kitzes (2013) is a compact modern statement of it. See [Input-Output Analysis](@ref io_page) for the `IOData` container that carries these accounts.

- **Satellite accounts**: attach any stressor matrix to a table with `add_extension!`, keyed by name
- **Intensities**: direct stressor use per unit of gross output
- **Emission multipliers**: total stressor use per unit of final demand, direct plus indirect
- **Footprints**: the consumption-based account by final-demand category and by product

```@setup io_environmental
using MacroEconometricModels
```

## Quick Start

**Recipe 1: Direct emission intensities**

```@example io_environmental
io = load_example(:wiot)      # ships with "employment" and "CO2" accounts
intensities(io, "CO2")
```

**Recipe 2: Consumption-based emission multipliers**

```@example io_environmental
emission_multipliers(io, "CO2")
```

**Recipe 3: The carbon footprint of final demand**

```@example io_environmental
footprint(io, "CO2")
```

**Recipe 4: Attach a new satellite account**

```@example io_environmental
add_extension!(io, "water", [12.0 28.0]; stressors=["H2O"], unit=["Ml"])
intensities(io, "water")
```

**Recipe 5: An account with direct final-demand use**

```@example io_environmental
add_extension!(io, "water_hh", [12.0 28.0]; stressors=["H2O"], unit=["Ml"],
               F_Y=reshape([5.0], 1, 1))       # households draw 5 Ml directly
footprint(io, "water_hh").total
```

---

## Satellite Accounts

A **satellite account** is a block of physical flows measured in the same sector classification as the monetary table but in physical units. `add_extension!` attaches one under a name, checks that it is conformable, and precomputes its intensities.

```math
S = F\hat{x}^{-1}, \qquad S_{sj} = \frac{F_{sj}}{x_j}
```

where:
- ``F`` is the ``n_s \times n`` matrix of stressor flows, with ``F_{sj}`` the physical quantity of stressor ``s`` released or used by sector ``j``
- ``n_s`` is the number of stressors in the account
- ``x_j`` is gross output of sector ``j``, so ``S_{sj}`` is the **direct intensity**: stressor per unit of gross output
- ``F_Y`` is the ``n_s \times n_{fd}`` matrix of stressor flows arising *directly* in final demand — household heating fuel, private vehicle use — which pass through no production process

```@example io_environmental
add_extension!(io, "land", [0.8 0.3]; stressors=["area"], unit=["kha"])
report(io)
```

The account is stored in `io.extensions["land"]` and joins the summary alongside the `CO2` and `employment` accounts that ship with the table and the water accounts attached in the Quick Start. `add_extension!` mutates the table in place and returns it, so accounts accumulate across calls.

!!! warning "Orientation is checked, labels are broadcast"
    `F` must be ``n_s \times n`` — stressors down the rows, sectors across the columns. A
    transposed matrix throws `ArgumentError` naming the column count. `unit` and `stressors`
    each take either a vector with one entry per stressor or a bare string: `unit="kha"`
    applies that unit to every row, and `stressors="area"` is accepted only when `F` has a
    single row. A length that disagrees with ``n_s`` throws `ArgumentError` naming the keyword.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `stressors` | `AbstractVector`/`AbstractString` | — | Required. Stressor labels, one per row of `F`; a bare string only when `F` has one row |
| `unit` | `AbstractVector`/`AbstractString` | — | Required. Physical unit, one per stressor; a bare string applies to every row |
| `F_Y` | `AbstractMatrix` | `nothing` | ``n_s \times n_{fd}`` direct stressor use in final demand; zeros when omitted |

### Return Values

`io.extensions[name]` holds an `IOExtension{T}`:

| Field | Type | Description |
|-------|------|-------------|
| `F` | `Matrix{T}` | ``n_s \times n`` stressor flows by sector |
| `F_Y` | `Matrix{T}` | ``n_s \times n_{fd}`` direct stressor flows in final demand |
| `S` | `Matrix{T}` | ``n_s \times n`` direct intensities ``F\hat{x}^{-1}`` |
| `stressors` | `Vector{String}` | Stressor labels |
| `unit` | `Vector{String}` | Physical unit of each stressor |

---

## Intensities

`intensities` returns the precomputed ``S`` of a named account. Intensities are the physical counterpart of the technical coefficients: they describe technology, not scale, and are the quantities that change when a sector decarbonizes.

```@example io_environmental
intensities(io, "CO2")
```

Agriculture emits 0.10 kt of CO``_2`` per unit of gross output and manufacturing 0.15 — manufacturing is half again as carbon-intensive per unit produced. Because manufacturing is also twice as large, it accounts for 300 of the economy's 400 kt of direct emissions.

```@example io_environmental
intensities(io, "employment")
```

The employment account runs the other way: agriculture uses 0.030 thousand persons per unit of output against manufacturing's 0.020, so agriculture is the labour-intensive and carbon-light sector of the two.

---

## Emission Multipliers

Direct intensities charge every tonne to the sector that emits it. **Emission multipliers** charge it to the final product it ends up in, by routing the intensities through the Leontief inverse.

```math
M = S L, \qquad M_{sj} = \sum_{i=1}^{n} S_{si} L_{ij}
```

where:
- ``M_{sj}`` is the total quantity of stressor ``s`` released across the whole supply chain per unit of final demand for product ``j``
- ``L_{ij}`` is the Leontief inverse, the gross output of ``i`` required per unit of final demand for ``j``
- the sum runs over every supplying sector ``i``, so ``M`` counts direct emissions plus emissions embodied in inputs, inputs of inputs, and so on

```@example io_environmental
emission_multipliers(io, "CO2")
```

One unit of final demand for agricultural output carries 0.165 kt of CO``_2``, against a direct intensity of 0.100 — the extra 0.065 is emitted upstream, mostly by the manufacturing inputs that agriculture buys. Manufacturing's multiplier of 0.201 sits closer to its 0.150 direct intensity because it buys fewer intermediates per unit of output. Multipliers always exceed direct intensities in a table with positive intermediate flows.

```@example io_environmental
emission_multipliers(io, "employment")
```

These are the same numbers as `multipliers(io; kind=:employment, type=:I)` on the [Classical Analysis](@ref io_classical_page) page: a Type I employment multiplier is exactly the emission multiplier of the `employment` satellite account. The two entry points exist because the same arithmetic is read as a jobs multiplier in impact analysis and as a labour footprint in environmental accounting.

---

## Footprints

A **footprint** is the consumption-based account. It answers the question that a production inventory cannot: how much of a stressor is a given final-demand category responsible for, counting every stage of its supply chain?

```math
E = M Y + F_Y, \qquad e_j = M_{\cdot j} \, y_j
```

where:
- ``E`` is the ``n_s \times n_{fd}`` footprint by final-demand category, stored in the field `total`
- ``M Y`` is the stressor embodied in the products bought by each category
- ``F_Y`` adds the stressor released directly by final consumers, bypassing production entirely
- ``e_j`` is the footprint attributable to final demand for product ``j``, stored in the field `by_sector`
- ``y = Y\mathbf{1}`` is total final demand by product

```@example io_environmental
fp = footprint(io, "CO2")
report(fp)
```

```@example io_environmental
fp.by_sector
```

The footprint total of 400 kt equals the economy's direct emissions of 400 kt, as it must: in a single-region table with no imports and no direct household emissions, every tonne emitted is embodied in some final product. What changes is the attribution. Agriculture *emits* 100 kt but only 57.8 kt are embodied in final demand for agricultural products; the other 42.2 kt travel downstream inside the farm goods that manufacturers buy. Manufacturing *emits* 300 kt but carries a footprint of 342.2 kt. A carbon tax levied on emitters and a carbon tariff levied on consumption therefore hit these two sectors in opposite proportions.

### Direct Final-Demand Emissions

`F_Y` breaks the equality between production and consumption totals, because household fuel use is not produced by any sector in the table:

```@example io_environmental
footprint(io, "water_hh").total
```

The water account's supply-chain footprint is 40 Ml and households draw a further 5 Ml directly from the mains, giving a consumption-based total of 45 Ml against 40 Ml of industrial abstraction. In an MRIO setting the same mechanism carries imported emissions, which is why footprints and territorial inventories diverge for open economies (Wiedmann & Lenzen 2018).

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `total` | `Matrix{Float64}` | ``n_s \times n_{fd}`` footprint by final-demand category |
| `by_sector` | `Matrix{Float64}` | ``n_s \times n`` footprint attributed to final demand for each product |
| `stressors` | `Vector{String}` | Stressor labels |
| `name` | `String` | Name of the satellite account |

---

## Complete Example

This example attaches a new satellite account to the built-in table and runs the full environmental workflow on it: intensities, multipliers, footprint, and the comparison against the production-based inventory that motivates consumption-based accounting.

```@example io_environmental
tbl = load_example(:wiot)

# Energy use: agriculture is the smaller but more energy-intensive user per unit
add_extension!(tbl, "energy", [45.0 120.0]; stressors=["TJ"], unit=["terajoules"])
intensities(tbl, "energy")
```

```@example io_environmental
emission_multipliers(tbl, "energy")
```

```@example io_environmental
fp_energy = footprint(tbl, "energy")
report(fp_energy)
```

```@example io_environmental
fp_energy.by_sector
```

```@example io_environmental
sum(tbl.extensions["energy"].F, dims=2)      # production-based inventory
```

Agriculture uses 0.045 TJ per unit of gross output against manufacturing's 0.060, a 33 percent gap. The supply-chain multipliers of 0.0723 and 0.0822 TJ per unit of final demand narrow that gap to 14 percent, because agriculture buys proportionally more from the energy-hungry manufacturing sector than manufacturing buys back. The footprint totals 165 TJ, matching the production-based inventory of 45 + 120 TJ, but splits it 25.3 to 139.7 across the two final products rather than the 45 to 120 of the direct inventory: nearly half of agriculture's direct energy use is ultimately consumed as manufactured goods.

---

## Common Pitfalls

1. **`F` is stressors by sectors, not sectors by stressors.** The matrix must be ``n_s \times n``. A single-stressor account for two sectors is `[12.0 28.0]` — a ``1 \times 2`` row — not a ``2 \times 1`` column. The wrong orientation throws `ArgumentError` naming the column count, which is the good case; a square table would accept it silently.

2. **A bare-string `stressors` is a one-row shorthand, a bare-string `unit` is not.** `unit="kt"` broadcasts to every stressor, so a two-row account labelled `unit="kt"` gets `["kt", "kt"]` — check that both rows really share the unit. `stressors="CO2"` is accepted only when `F` has one row; with more rows it throws `ArgumentError` naming the required length rather than silently labelling one row.

3. **Intensities are frozen at attach time.** `add_extension!` divides `F` by the gross output of the table *as it stands*. Rebuilding or rescaling the table afterwards leaves stale intensities behind. Attach accounts after the table is final, or re-attach.

4. **`add_extension!` mutates its argument.** It modifies `io.extensions` in place and returns the same object. Call it on a `deepcopy` when the original table must stay unchanged, particularly when comparing two years of the same table.

5. **`by_sector` indexes the consuming product, not the emitting sector.** Entry ``j`` of `by_sector` is the footprint of final demand for product ``j``, computed as ``M_{\cdot j} y_j``. It is not sector ``j``'s own emissions — those are the columns of `F`. Comparing the two is the whole point of the account, so keep them straight.

6. **Footprints of a single-region table cannot show carbon leakage.** With no import block, the consumption-based total necessarily equals the production-based total plus `F_Y`. The interesting divergence between territorial and footprint accounts requires a multi-region table; use `footprint(io, name; by=:region)` on an MRIO (see [MRIO Trade Accounting](@ref io_mrio_page)) or load one from [Downloading Data](@ref io_download_page).

7. **Account names are case-sensitive and exact.** `intensities(io, "co2")` throws even though `"CO2"` exists. The error lists the available names, which is the fastest way to find the right spelling.

---

## API Reference

```@docs
IOExtension
add_extension!
intensities
emission_multipliers
footprint
FootprintResult
RegionalFootprintResult
```

---

## References

- Kitzes, J. (2013). An Introduction to Environmentally-Extended Input-Output Analysis.
  *Resources*, 2(4), 489--503. [DOI](https://doi.org/10.3390/resources2040489)

- Leontief, W. (1970). Environmental Repercussions and the Economic Structure: An Input-Output Approach.
  *The Review of Economics and Statistics*, 52(3), 262--271. [DOI](https://doi.org/10.2307/1926294)

- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions* (2nd ed.).
  Cambridge University Press. ISBN 978-0-521-51713-3. [DOI](https://doi.org/10.1017/CBO9780511626982)

- Stadler, K., Wood, R., Bulavskaya, T., Sodersten, C.-J., Simas, M., Schmidt, S., et al. (2018).
  EXIOBASE 3: Developing a Time Series of Detailed Environmentally Extended Multi-Regional Input-Output Tables.
  *Journal of Industrial Ecology*, 22(3), 502--515. [DOI](https://doi.org/10.1111/jiec.12715)

- Wiedmann, T., & Lenzen, M. (2018). Environmental and Social Footprints of International Trade.
  *Nature Geoscience*, 11(5), 314--321. [DOI](https://doi.org/10.1038/s41561-018-0113-9)
