# [Classical Input-Output Analysis](@id io_classical_page)

Classical input-output analysis answers the question that Leontief (1936) posed: if final demand for one product rises by one unit, how much output must the whole economy produce? The answer is the Leontief inverse, and every quantity on this page — multipliers, linkages, key sectors, decompositions, extraction losses — is a weighted sum of its entries. See [Input-Output Analysis](@ref io_page) for the `IOData` container these functions read.

- **Coefficients and inverses**: technical coefficients ``A``, the Leontief inverse ``L``, allocation coefficients ``B``, and the Ghosh inverse ``G``
- **Multipliers**: output, income, and employment multipliers as Type I (open) or Type II (closed with respect to households)
- **Linkages**: backward and forward linkages, Rasmussen dispersion indices, and the key-sector classification
- **Decomposition**: additive and multiplicative structural decomposition of output change between two tables
- **Extraction**: the output loss from hypothetically removing one or more sectors

```@setup io_classical
using MacroEconometricModels
```

## Quick Start

**Recipe 1: The Leontief inverse**

```@example io_classical
io = load_example(:wiot)
leontief_inverse(io)
```

**Recipe 2: Type I output multipliers**

```@example io_classical
multipliers(io; kind=:output, type=:I)
```

**Recipe 3: Type II multipliers with induced household spending**

```@example io_classical
multipliers(io; kind=:income, type=:II)
```

**Recipe 4: Rasmussen linkages and key sectors**

```@example io_classical
linkages(io)
```

**Recipe 5: Hypothetical extraction of a sector**

```@example io_classical
hypothetical_extraction(io, "Agriculture")
```

**Recipe 6: Structural decomposition between two tables**

```@example io_classical
io_2010 = IOData([150.0 420.0; 200.0 100.0],            # cheaper agricultural inputs
                 reshape([420.0, 2040.0], 2, 1),        # final demand up 20 percent
                 [300.0 1000.0; 340.0 820.0];
                 sectors=io.sectors, fd_cats=io.fd_cats,
                 va_cats=io.va_cats, year=2010)
sda(io, io_2010)
```

---

## Technical Coefficients and the Leontief Inverse

The **technical-coefficients** matrix records what each sector buys per unit of its own output. Substituting it into the row balance turns the accounting identity into a model: gross output is whatever is needed to deliver final demand, including the inputs needed to make the inputs.

```math
A = Z\hat{x}^{-1}, \qquad x = Ax + y \quad \Longrightarrow \quad x = (I - A)^{-1} y = L y
```

where:
- ``A`` is the ``n \times n`` technical-coefficients matrix with ``A_{ij} = Z_{ij}/x_j``, the input of ``i`` per unit of output of ``j``
- ``\hat{x}`` is the diagonal matrix of gross output, so ``\hat{x}^{-1}`` divides each column by that sector's output
- ``y = Y\mathbf{1}`` is the ``n \times 1`` vector of total final demand
- ``L = (I - A)^{-1}`` is the **Leontief inverse**, the total-requirements matrix
- ``L_{ij}`` is the gross output of sector ``i`` required, directly and indirectly, per unit of final demand for sector ``j``

!!! note "Technical Note"
    ``\hat{x}^{-1}`` is formed with a guarded reciprocal that maps a zero output to zero
    rather than to infinity, so sectors that are present in the labels but carry no output
    do not poison the coefficient matrix. The inverse ``(I - A)^{-1}`` exists whenever the
    column sums of ``A`` are below one, which the column balance guarantees for any table
    with strictly positive value added.

```@example io_classical
A = technical_coefficients(io)
L = leontief_inverse(io)
L
```

The diagonal entries exceed one because delivering a unit of final demand requires the sector to produce that unit plus whatever it consumes of its own output along the chain. Agriculture's ``L_{11} = 1.254`` says that one unit of agricultural final demand needs 1.254 units of agricultural gross output; the off-diagonal ``L_{21} = 0.264`` says the same unit also pulls 0.264 units out of manufacturing. Reading down column ``j`` gives the entire production programme triggered by final demand for product ``j``.

`leontief` bundles the coefficients, the inverse, gross output, and a back-reference to the table into one object:

```@example io_classical
m = leontief(io)
report(m)
```

```julia
plot_result(m)
```

```@raw html
<iframe src="../assets/plots/leontief_heatmap.html" width="100%" height="520" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The heatmap reads column-wise: each column is the total-requirements profile of one unit of final demand for that product, with the supplying sector on the vertical axis. Dark cells off the diagonal mark the strong bilateral dependencies that later become linkages and key sectors.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `A` | `Matrix{T}` | ``n \times n`` technical coefficients |
| `L` | `Matrix{T}` | ``n \times n`` Leontief inverse |
| `x` | `Vector{T}` | ``n \times 1`` gross output copied from the table |
| `io` | `IOData{T}` | Back-reference to the source table |

---

## The Ghosh Supply-Driven Model

The Leontief model runs demand backwards through the supply chain. Ghosh (1958) runs value added forwards: given what each sector pays to primary factors, how much output is distributed downstream? The **allocation-coefficients** matrix normalizes the same flow matrix by rows rather than columns.

```math
B = \hat{x}^{-1}Z, \qquad x' = x'B + v' \quad \Longrightarrow \quad x' = v'(I - B)^{-1} = v'G
```

where:
- ``B`` is the ``n \times n`` allocation-coefficients matrix with ``B_{ij} = Z_{ij}/x_i``, the share of ``i``'s output sold to ``j``
- ``v' = \mathbf{1}'V`` is the ``1 \times n`` row vector of total value added by sector
- ``G = (I - B)^{-1}`` is the **Ghosh inverse**, the output-allocation requirements matrix
- ``G_{ij}`` is the output of sector ``j`` induced, directly and indirectly, per unit of primary input into sector ``i``

```@example io_classical
g = ghosh(io)
report(g)
```

The two inverses carry the same information in different coordinates, related by ``G = \hat{x}^{-1} L \hat{x}``. Agriculture's ``G_{12} = 0.660`` is exactly ``L_{12} \cdot x_2 / x_1 = 0.330 \times 2``: because manufacturing is twice agriculture's size, the same technological link looks twice as large when measured as a share of agricultural output. The Ghosh row sums are the forward linkages used below.

!!! warning "Interpret the Ghosh model as an accounting decomposition"
    Read literally, ``x' = v'G`` says that output responds to primary-input supply with fixed
    sales shares and no price response, which is implausible as behaviour. Miller & Blair (2009,
    Ch. 12) recommend treating ``G`` as a descriptive allocation account — which is how the
    forward linkages on this page use it — rather than as a supply-side counterpart of the
    Leontief demand model.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `B` | `Matrix{T}` | ``n \times n`` allocation coefficients |
| `G` | `Matrix{T}` | ``n \times n`` Ghosh inverse |
| `x` | `Vector{T}` | ``n \times 1`` gross output copied from the table |
| `io` | `IOData{T}` | Back-reference to the source table |

`technical_coefficients`, `leontief_inverse`, `allocation_coefficients`, and `ghosh_inverse` return the bare matrices when the bundled object is not needed.

---

## Multipliers

A **multiplier** converts one unit of final demand into the economy-wide total of some quantity — gross output, value added, or jobs. **Type I** multipliers hold household income outside the model. **Type II** multipliers close the model with respect to households, so the wages paid along the supply chain are respent and generate a further round of production.

```math
m^{\text{I}} = L'h, \qquad h_j = \frac{c_j}{x_j}
```

where:
- ``m^{\text{I}}_j`` is the Type I multiplier of sector ``j``
- ``h_j`` is the direct per-unit-output requirement of the chosen quantity
- ``c_j = 1`` for `kind=:output` (so ``m^{\text{I}}`` is the vector of column sums of ``L``)
- ``c_j = \sum_v V_{vj}`` for `kind=:income`, total value added paid by sector ``j``
- ``c_j = \sum_s F_{sj}`` for `kind=:employment`, total jobs in sector ``j`` taken from the `employment` satellite account

```@example io_classical
multipliers(io; kind=:output, type=:I)
```

Agriculture's 1.518 and manufacturing's 1.452 are the column sums of ``L``. The gap is small here because both sectors buy a similar share of intermediates, but the ranking is what matters for policy: an extra unit of final demand placed with agriculture mobilizes 4.5 percent more gross output than the same unit placed with manufacturing.

```@example io_classical
multipliers(io; kind=:employment, type=:I)
```

Agriculture supports 0.0429 thousand jobs per unit of final demand against manufacturing's 0.0323 — a 33 percent difference, far wider than the output ranking, because agriculture is the more labour-intensive sector (0.030 against 0.020 jobs per unit of gross output) *and* has the larger output multiplier.

!!! note "Type I income multipliers equal one by construction"
    With `kind=:income` the coefficient row is *total* value added, ``h' = \mathbf{1}'(I - A)``,
    so ``h'L = \mathbf{1}'(I-A)(I-A)^{-1} = \mathbf{1}'`` exactly. Every unit of final demand
    generates exactly one unit of value added economy-wide — the national accounting identity.
    A vector of ones is therefore a check that the table balances, not a degenerate result.
    Use `type=:II` for the household-income multiplier that varies across sectors.

### Type II: Closing the Model

Closing with respect to households adds a row that turns output into household income and a column that turns household income into consumption demand:

```math
\bar{A} = \begin{bmatrix} A & h^{c} \\ (h^{\text{inc}})' & 0 \end{bmatrix},
\qquad \bar{L} = (I - \bar{A})^{-1}
```

where:
- ``h^{\text{inc}}_j = V_{1j}/x_j`` is compensation of employees per unit of output, taken from the **first** value-added row
- ``h^{c} = y / \sum_i y_i`` is the household consumption column, the shares of total final demand
- ``\bar{L}`` is the ``(n+1) \times (n+1)`` closed Leontief inverse

Type II output multipliers are the column sums of the upper-left ``n \times n`` block of ``\bar{L}``; Type II income multipliers are row ``n+1``; Type II employment multipliers apply the jobs coefficients to that same block.

```@example io_classical
multipliers(io; kind=:output, type=:II)
```

```@example io_classical
multipliers(io; kind=:income, type=:II)
```

Type II output multipliers of 3.55 and 4.09 are roughly 2.3 and 2.8 times their Type I counterparts: with 30 and 50 percent of output paid out as compensation and all of it respent, the induced consumption round dominates. The ranking also flips — manufacturing now leads, because it pays the higher wage share and therefore feeds more income back into the circular flow. The Type II income multipliers say the same thing in levels: one unit of final demand for manufacturing ultimately generates 1.804 units of household income against agriculture's 1.389.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `kind` | `Symbol` | `:output` | `:output`, `:income`, or `:employment` |
| `type` | `Symbol` | `:I` | `:I` for the open model, `:II` for the household-closed model |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `values` | `Vector{Float64}` | Multiplier for each sector |
| `kind` | `Symbol` | The quantity multiplied |
| `type` | `Symbol` | `:I` or `:II` |
| `sectors` | `Vector{String}` | Sector labels |

```julia
plot_result(multipliers(io; kind=:output, type=:I))
```

---

## Linkages and Key Sectors

**Backward linkage** measures how strongly a sector pulls on its suppliers; **forward linkage** measures how strongly it pushes output onto its customers. Rasmussen (1956) normalizes both by the economy-wide average so they can be compared across sectors and tables, and the resulting quadrants deliver the key-sector classification that Hirschman (1958) proposed for development planning.

```math
BL_i = \sum_{k=1}^{n} L_{ki}, \qquad FL_i = \sum_{k=1}^{n} G_{ik}
```

```math
U^{\text{b}}_i = \frac{BL_i}{n^{-1}\sum_{k} BL_k}, \qquad
U^{\text{f}}_i = \frac{FL_i}{n^{-1}\sum_{k} FL_k}
```

where:
- ``BL_i`` is the backward linkage of sector ``i``, the ``i``-th column sum of the Leontief inverse
- ``FL_i`` is the forward linkage of sector ``i``, the ``i``-th row sum of the Ghosh inverse (or of ``L`` with `forward=:leontief`, the Chenery & Watanabe 1958 variant)
- ``U^{\text{b}}_i`` is the **power of dispersion**, stored in the field `Ui`
- ``U^{\text{f}}_i`` is the **sensitivity of dispersion**, stored in the field `Uj`
- ``n`` is the number of sectors, so both indices average to one across the table

```@example io_classical
lk = linkages(io)
report(lk)
```

Agriculture clears both thresholds — ``U^{\text{b}} = 1.022`` and ``U^{\text{f}} = 1.208`` — and is classified as a key sector: it is both an above-average buyer of intermediates and an above-average supplier to the rest of the economy. Manufacturing sits below one on both indices and is classified `:weak`. In a two-sector table one sector is above average whenever the other is below, so the interesting content is the *margin*: agriculture's forward index exceeds average by 21 percent while its backward index exceeds it by only 2 percent, making it a supply-side bottleneck rather than a demand-side engine.

| ``U^{\text{b}}`` | ``U^{\text{f}}`` | `classification` | Reading |
|---|---|---|---|
| ``> 1`` | ``> 1`` | `:key` | Above-average pull and push |
| ``> 1`` | ``\le 1`` | `:backward` | Strong buyer, weak supplier |
| ``\le 1`` | ``> 1`` | `:forward` | Strong supplier, weak buyer |
| ``\le 1`` | ``\le 1`` | `:weak` | Below average on both |

```@example io_classical
key_sectors(io)
```

Switching the forward measure to the Chenery & Watanabe (1958) row sums of ``L`` narrows the gap, because that variant weights each sector by the size of its customers' final demand rather than by its own sales structure:

```@example io_classical
linkages(io; forward=:leontief).Uj
```

`rasmussen(io)` is an alias for `linkages(io)` with the default settings.

```julia
plot_result(lk)
```

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `forward` | `Symbol` | `:ghosh` | `:ghosh` for row sums of ``G``, `:leontief` for row sums of ``L`` |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `backward` | `Vector{Float64}` | Column sums of ``L`` |
| `forward` | `Vector{Float64}` | Row sums of ``G``, or of ``L`` when `forward=:leontief` |
| `Ui` | `Vector{Float64}` | Power of dispersion, ``U^{\text{b}}`` |
| `Uj` | `Vector{Float64}` | Sensitivity of dispersion, ``U^{\text{f}}`` |
| `classification` | `Vector{Symbol}` | `:key`, `:backward`, `:forward`, or `:weak` |
| `sectors` | `Vector{String}` | Sector labels |

---

## Structural Decomposition Analysis

Output changes between two tables for two reasons: technology changes, so each unit of final demand needs a different bundle of inputs, and final demand itself changes. **Structural decomposition analysis** splits the observed change into those two parts. Because the two effects interact, the split depends on which is evaluated at which period's values; Dietzenbacher & Los (1998) show that the average of the two polar orderings is the natural choice, and it is exact.

```math
\Delta x = L^1 y^1 - L^0 y^0
= \underbrace{\tfrac{1}{2}\left(\Delta L\, y^0 + \Delta L\, y^1\right)}_{\text{technology effect}}
+ \underbrace{\tfrac{1}{2}\left(L^1 \Delta y + L^0 \Delta y\right)}_{\text{final-demand effect}}
```

where:
- ``L^0, L^1`` are the Leontief inverses of the base and comparison tables
- ``y^0, y^1`` are total final demand in the two periods
- ``\Delta L = L^1 - L^0`` and ``\Delta y = y^1 - y^0``
- the two braced terms sum to ``\Delta x`` identically, so the reported `residual` is zero up to floating-point error

```@example io_classical
decomp = sda(io, io_2010; method=:additive)
report(decomp)
```

```@example io_classical
decomp.total
```

Agricultural gross output *falls* by 10 units between the two tables even though final demand for every product rose 20 percent. The decomposition explains why: the final-demand effect adds 182.5 units, but the technology effect removes 192.5, because manufacturers cut their agricultural input from 500 to 420 and no longer need the extra farm output. Manufacturing gains 340 units on a much smaller technology drag of 55. Without the decomposition the two forces are invisible in the 990-versus-1000 headline.

### The Multiplicative Form

The multiplicative variant reports the same story as growth factors that multiply to the observed output ratio:

```math
\frac{x^1_i}{x^0_i} = \underbrace{\frac{(L^1 y^0)_i}{x^0_i}}_{\text{technology factor}} \cdot
\underbrace{\frac{x^1_i}{(L^1 y^0)_i}}_{\text{final-demand factor}}
```

```@example io_classical
sda(io, io_2010; method=:multiplicative)
```

Agriculture's factors are 0.825 and 1.200, whose product is the observed 0.99 output ratio: the new technology alone would have cut agricultural output by 17.5 percent, and the 20 percent demand expansion almost exactly offset it. Unlike the additive form, this variant evaluates technology first and demand second rather than averaging the two polar orderings, so the two factors are not symmetric in the two periods.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:additive` | `:additive` for the two-polar difference form, `:multiplicative` for the ratio form |
| `factors` | `Symbol` | `:LY` | Factor split; only the two-factor ``L``/``y`` decomposition is implemented |
| `average` | `Symbol` | `:two_polar` | Averaging scheme; only the two-polar average is implemented |

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `effects` | `Dict{Symbol,Vector{Float64}}` | `:L` technology effect and `:Y` final-demand effect, per sector |
| `total` | `Vector{Float64}` | Output change (additive) or output ratio (multiplicative) |
| `residual` | `Vector{Float64}` | Change left unattributed by the two effects |
| `method` | `Symbol` | `:additive` or `:multiplicative` |

---

## Hypothetical Extraction

The **hypothetical extraction method** asks what the economy would produce if a sector vanished — not just its own output, but everything that existed only to supply it. Setting the sector's row and column of ``A`` to zero severs both its purchases and its sales, and zeroing its final demand removes the deliveries it would have made.

```math
x^{(-k)} = \left(I - A^{(-k)}\right)^{-1} y^{(-k)}, \qquad
\text{loss} = x - x^{(-k)}
```

where:
- ``A^{(-k)}`` is ``A`` with rows and columns in the extracted set ``k`` set to zero
- ``y^{(-k)}`` is total final demand with the extracted entries set to zero
- ``x^{(-k)}`` is gross output in the reduced economy
- `total_loss` is ``\mathbf{1}'\text{loss}``, the economy-wide loss of gross output

```@example io_classical
hypothetical_extraction(io, "Agriculture")
```

```@example io_classical
hypothetical_extraction(io, "Agriculture").sector_loss
```

Removing agriculture costs 1210.5 units of gross output: its own 1000 plus 210.5 of manufacturing output that existed solely to supply it. That is 40.4 percent of the economy's 3000 units of gross output, well above agriculture's own 33.3 percent share — the difference is exactly the indirect dependence that a share-of-output statistic misses. Extracting manufacturing costs 2588.2 units, or 86.3 percent against a direct share of 66.7 percent.

Sectors can be named or indexed, singly or in groups:

```@example io_classical
hypothetical_extraction(io, ["Agriculture", "Manufacturing"])
```

Extracting every sector loses the entire 3000 units of gross output, which is the sanity check on the method.

### Return Values

| Field | Type | Description |
|-------|------|-------------|
| `total_loss` | `Float64` | Economy-wide loss of gross output |
| `sector_loss` | `Vector{Float64}` | Loss of gross output by sector |
| `extracted` | `Vector{Int}` | Indices of the extracted sectors |

---

## Complete Example

This example runs the full classical workflow on the built-in table: build both inverses, rank sectors by their output and job multipliers, classify them by linkage, measure their systemic importance by extraction, and decompose the change against a second-year table.

```@example io_classical
io_09 = load_example(:wiot)
lm = leontief(io_09)
report(lm)
```

```@example io_classical
report(multipliers(io_09; kind=:output, type=:I))
```

```@example io_classical
report(multipliers(io_09; kind=:employment, type=:I))
```

```@example io_classical
report(linkages(io_09))
```

```@example io_classical
report(hypothetical_extraction(io_09, "Manufacturing"))
```

```@example io_classical
io_10 = IOData([150.0 420.0; 200.0 100.0], reshape([420.0, 2040.0], 2, 1),
               [300.0 1000.0; 340.0 820.0];
               sectors=io_09.sectors, fd_cats=io_09.fd_cats,
               va_cats=io_09.va_cats, year=2010)
report(sda(io_09, io_10))
```

```julia
plot_result(lm)
```

```@raw html
<iframe src="../assets/plots/leontief_heatmap.html" width="100%" height="520" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The four diagnostics agree on the economics but rank the sectors differently, which is the point of running all of them. Manufacturing is the larger sector and the bigger loss under extraction (2588.2 against 1210.5), but agriculture has the higher output multiplier (1.518 against 1.452), the higher employment multiplier (0.0429 against 0.0323), and the only key-sector classification. A demand stimulus therefore does more per unit placed with agriculture, while a supply disruption does more damage in manufacturing. The decomposition adds the dynamic layer: agriculture's technology-driven contraction of 192.5 units swamped its 182.5-unit demand gain, so its favourable multipliers did not translate into growth.

---

## Common Pitfalls

1. **The two tables in `sda` must have the same sector ordering.** `sda` differences the Leontief inverses element by element and never checks labels. Tables with the same sectors in a different order produce a silently wrong decomposition. Reorder the rows, columns, and final-demand entries of the comparison table before calling `sda`.

2. **`kind=:income, type=:I` returns ones, and that is correct.** The income coefficient is total value added, whose product with ``L`` is identically one by the column balance. Use `type=:II` for a household-income multiplier that discriminates between sectors.

3. **Type II multipliers depend on which value-added row comes first.** `_closed_leontief` treats row 1 of `va` as compensation of employees. A table whose first value-added category is taxes or operating surplus produces a household-closure that has no economic meaning. Reorder `va` so that compensation is the first row.

4. **Type II treats all final demand as household consumption.** The consumption column is the shares of *total* final demand across sectors. In a table with separate investment, government, and export columns this overstates household spending, and the multipliers are biased upward. Restrict `Y` to the household column before closing the model.

5. **`kind=:employment` requires a satellite account named exactly `"employment"`.** Any other name — `"jobs"`, `"labour"` — throws `ArgumentError` even when the flows are identical. See [Environmental Extensions](@ref io_environmental_page) for `add_extension!`.

6. **Forward linkages change meaning with the `forward` keyword.** The default `:ghosh` normalizes sales by the supplying sector's own output; `:leontief` gives the Chenery & Watanabe (1958) row sums of ``L``. The two indices rank sectors differently — here agriculture's sensitivity index falls from 1.208 to 1.067 — so never compare a Ghosh-based index against a Leontief-based one.

7. **`factors` and `average` are reserved, not dispatched.** `sda` accepts both keywords for forward compatibility but implements only the two-factor ``L``/``y`` split with the two-polar average. Any other value is accepted silently and changes nothing.

8. **Extraction losses are not additive across sectors.** The loss from removing two sectors together is smaller than the sum of the two individual losses, because the shared indirect requirements are counted once. Compare group extractions against group extractions, never against a sum of singletons.

---

## API Reference

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

---

## References

- Chenery, H. B., & Watanabe, T. (1958). International Comparisons of the Structure of Production.
  *Econometrica*, 26(4), 487--521. [DOI](https://doi.org/10.2307/1907514)

- Dietzenbacher, E., & Los, B. (1998). Structural Decomposition Techniques: Sense and Sensitivity.
  *Economic Systems Research*, 10(4), 307--324. [DOI](https://doi.org/10.1080/09535319800000023)

- Ghosh, A. (1958). Input-Output Approach in an Allocation System.
  *Economica*, 25(97), 58--64. [DOI](https://doi.org/10.2307/2550694)

- Hirschman, A. O. (1958). *The Strategy of Economic Development*.
  Yale University Press.

- Leontief, W. W. (1936). Quantitative Input and Output Relations in the Economic System of the United States.
  *The Review of Economics and Statistics*, 18(3), 105--125. [DOI](https://doi.org/10.2307/1927837)

- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions* (2nd ed.).
  Cambridge University Press. ISBN 978-0-521-51713-3. [DOI](https://doi.org/10.1017/CBO9780511626982)

- Rasmussen, P. N. (1956). *Studies in Inter-Sectoral Relations*.
  North-Holland.
