# [Input-Output Analysis API Reference](@id api_io_page)

Complete API documentation for the input-output module in **MacroEconometricModels.jl**. This catalog covers data structures, classical Leontief and Ghosh multipliers, hypothetical extraction, price models, structural decomposition, matrix balancing, network topology, environmental extensions, multi-region trade accounting, nonlinear production networks, and automated database downloaders.

See the [Input-Output Analysis](@ref io_page) hub for an architectural overview, and its child narrative pages:
- [Classical Input-Output Analysis](@ref io_classical_page) — Leontief and Ghosh models, multipliers, linkages, extraction, prices, and network statistics
- [Environmental Extensions](@ref io_environmental_page) — satellite accounts, intensities, and consumption-based footprints
- [MRIO Trade Accounting](@ref io_mrio_page) — vertical specialization (Hummels–Ishii–Yi) and value-added export decompositions (Koopman–Wang–Wei)
- [Baqaee & Farhi (2019) Nonlinear Input-Output](@ref io_baqaee_farhi_page) — nonlinear production networks, Domar weights, and allocative efficiency wedges
- [Downloading IO/MRIO Data](@ref io_download_page) — recipes and downloaders for WIOD, OECD ICIO, EXIOBASE 3, Eora26, and GLORIA

---

## Containers and Metadata

Typed data container for intermediate flows, final demand, value added, gross output, and satellite accounts, with complete metadata.

```@docs
IOData
IOMetaData
```

---

## Inverses and Accounting Systems

Direct requirement and allocation matrices, and the demand-driven Leontief and supply-driven Ghosh inverses.

```@docs
technical_coefficients
leontief_inverse
allocation_coefficients
ghosh_inverse
leontief
ghosh
```

---

## Multipliers, Linkages, and Key Sectors

Output, income, and employment multipliers, backward and forward linkages, Rasmussen dispersion indices, and key sector classification.

```@docs
multipliers
IOMultipliers
linkages
LinkageResult
rasmussen
key_sectors
```

---

## Sectoral Extraction, Prices, and Impact Scenarios

Output losses under hypothetical sectoral extraction, the Leontief cost-push price model, and final-demand impact scenarios.

```@docs
hypothetical_extraction
ExtractionResult
price_model
PriceModelResult
impact
ImpactResult
```

---

## Structural Decomposition and Matrix Balancing

Two-polar additive structural decomposition analysis, and RAS / Generalized RAS (GRAS) biproportional matrix balancing.

```@docs
sda
SDAResult
ras
gras
balance
RASResult
```

---

## Production Network Statistics

Network topology of the inter-industry matrix: Domar HHI, average propagation length (APL), degree distributions, and upstreamness.

```@docs
network_stats
NetworkStatsResult
```

---

## Environmental and Satellite Extensions

Attachment of physical factor and emission accounts, direct intensities, total multiplier matrices, and consumption-based footprint accounting.

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

## Multi-Region Input-Output (MRIO) Trade Accounting

Geographic and sectoral aggregation, inter-country block extraction, bilateral trade flows, vertical specialization, and gross export value-added decompositions.

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

## Production Networks and Nonlinear Shock Propagation

Baqaee & Farhi (2019, 2020) nonlinear general equilibrium model: Domar weights, nested CES production networks, exact nonlinear counterfactuals, second-order propagation, and allocative efficiency wedge decompositions.

```@docs
domar_weights
cost_based_domar
revenue_based_domar
baqaee_farhi
BaqaeeFarhiResult
production_network
ProductionNetwork
bf_equilibrium
BFEquilibrium
bf_shock_curve
BFShockCurve
bf_elasticities
BFElasticities
BFLocal
bf_quadratic
bf_wedge_decomp
BFWedgeDecomp
bf_misallocation
BFMisallocation
bf_wedge_quadratic
```

---

## Public MRIO Downloaders and Parsers

Utilities for fetching, verifying, and parsing global multi-regional input-output databases (OECD ICIO, WIOD, EXIOBASE 3, Eora26, GLORIA).

```@docs
list_io_sources
download_io
download_oecd
download_wiod
download_exiobase3
download_eora26
download_gloria
io_file_digest
parse_io
parse_icio
parse_wiod
```
