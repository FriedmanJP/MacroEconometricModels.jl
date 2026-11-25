# Data Licenses & Provenance

`MacroEconometricModels.jl` is licensed under the **GNU General Public License v3.0 (`GPL-3.0-only`)** (see [LICENSE](LICENSE)). The 13 bundled example datasets in `data/` are distributed as **mere aggregation** alongside the software under their respective public-domain statuses or data licenses, independently of the GPL-3.0-only software license.

---

## Summary Table

| Key | File | Observations × Vars | Source / Citation | License / Redistribution Basis |
| :--- | :--- | :--- | :--- | :--- |
| `:fred_md` | `data/fred_md.toml` | 804 × 114 (Monthly) | McCracken & Ng (2016), *JBES* 34(4), 574–589 (Jan 2026 vintage, public-domain subset) | **Public Domain** (US Federal Government works, 17 U.S.C. § 105: BEA, BLS, Census, DOL, Federal Reserve Board). Commercial/third-party series excluded (see below). |
| `:fred_qd` | `data/fred_qd.toml` | 268 × 226 (Quarterly) | McCracken & Ng (2020), St. Louis Fed WP 2020-005 (Jan 2026 vintage, public-domain subset) | **Public Domain** (US Federal Government works, 17 U.S.C. § 105: BEA, BLS, Census, DOL, Federal Reserve Board, US Treasury, FHFA). Commercial/third-party series excluded (see below). |
| `:pwt` | `data/pwt.toml` | 2,812 × 42 (38 OECD × 74 yr) | Feenstra, Inklaar & Timmer (2015), *AER* 105(10), 3150–3182; Penn World Table v10.01 (Groningen Growth and Development Centre) | **[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)** (attribution & modification notice below). |
| `:ddcg` | `data/ddcg.toml` | 9,384 × 2 (184 countries × 51 yr) | Acemoglu, Naidu, Restrepo & Robinson (2019), *JPE* 127(1), 47–100; Dube, Girardi, Jordà & Taylor (2025) | Published academic replication archive extract (factual country-year GDP per capita and binary democracy indicator, converted to TOML). |
| `:mpdta` | `data/mpdta.toml` | 2,500 × 3 (500 counties × 5 yr) | Callaway & Sant'Anna (2021), *J. Econometrics* 225(2), 200–230 | Distributed in R `did` (GPL-2.0-or-later); derived from US Census QWI and public county minimum-wage records, converted to TOML. |
| `:wiot` | `data/wiot.toml` | 2 sectors (IO table) | Miller & Blair (2009), *Input-Output Analysis*, 2nd ed., Table 2.3 | Synthetic 2×2 textbook illustrative numerical example with illustrative employment and CO₂ extensions. |
| `:nile` | `data/nile.toml` | 100 × 1 (1871–1970) | Cobb (1978), *Biometrika* 65(2), 243–251; Durbin & Koopman (2012), 2nd ed. | **Public Domain** historical hydrological measurements (1871–1970 annual flow of the Nile at Aswan). |
| `:mroz` | `data/mroz.toml` | 753 × 22 (Cross-section) | Mroz (1987), *Econometrica* 55(4), 765–799 | Publicly distributed textbook econometric extract from the 1976 Panel Study of Income Dynamics (PSID, University of Michigan ISR). |
| `:gnp_hamilton` | `data/gnp_hamilton.toml` | 135 × 1 (1951Q2–1984Q4) | Hamilton (1989), *Econometrica* 57(2), 357–384 | **Public Domain** US Bureau of Economic Analysis (BEA) real GNP growth series (17 U.S.C. § 105). |
| `:stackloss` | `data/stackloss.toml` | 21 × 4 (Cross-section) | Brownlee (1965), *Statistical Theory and Methodology in Science and Engineering*, 2nd ed., p. 454 | **Public Domain** published engineering measurement benchmark dataset. |
| `:grunfeld` | `data/grunfeld.toml` | 200 × 3 (10 firms × 20 yr) | Grunfeld (1958), Univ. of Chicago PhD thesis; Zellner (1962), *JASA* 57(298), 348–368 | Classical published corporate financial statistics (1935–1954), widely redistributed econometric benchmark. |
| `:denmark` | `data/denmark.toml` | 55 × 5 (1974Q1–1987Q3) | Johansen & Juselius (1990), *Oxford Bull. Econ. Statist.* 52(2), 169–210 | Historical Danish macroeconomic aggregates; distributed in R `urca` (GPL-2.0-or-later, Pfaff 2008). |
| `:mp_shocks` | `data/mp_shocks.toml` | 240 × 8 (1960Q1–2019Q4) | McKay & Wolf (2023), *Econometrica* 91(5), 1695–1725 (replication data appendix) | Published academic replication archive combining US federal macro series (BEA, FRB) and published academic shock estimates (Romer & Romer 2004 / Wieland & Yang 2020; Gertler & Karadi 2015; Aruoba & Drechsel 2024; Ben Zeev & Khan 2016). |

---

## Detailed Notices

### Penn World Table 10.01 (`:pwt` — `data/pwt.toml`) — CC-BY-4.0 Attribution

- **Dataset**: Penn World Table, version 10.01 (Groningen Growth and Development Centre, University of Groningen).
- **Attribution**: Feenstra, Robert C., Robert Inklaar, and Marcel P. Timmer (2015), *"The Next Generation of the Penn World Table"*, *American Economic Review*, 105(10), 3150–3182. DOI: [10.1257/aer.20130954](https://doi.org/10.1257/aer.20130954).
- **License**: Creative Commons Attribution 4.0 International (**CC-BY-4.0**), [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).
- **Modifications Made**: Subset to 38 OECD member countries across 42 national-accounts variables (1950–2023) and converted from the upstream Stata/Excel release to TOML format for `MacroEconometricModels.jl`.

### FRED-MD (`:fred_md`) & FRED-QD (`:fred_qd`) — Public-Domain Subset & Loading Official CSVs

The official McCracken & Ng FRED-MD and FRED-QD databases distributed by the Federal Reserve Bank of St. Louis include several commercial third-party copyrighted indices alongside US federal government public-domain series. Under the St. Louis Fed FRED® Services Terms of Use, personal/research access does not grant blanket third-party redistribution rights for copyrighted constituent series.

To ensure clean redistribution for both `MacroEconometricModels.jl` and downstream standalone releases, the bundled `data/fred_md.toml` (114 series) and `data/fred_qd.toml` (226 series) contain **only US federal government public-domain series** (BEA, BLS, Census Bureau, DOL, Board of Governors of the Federal Reserve System, US Treasury, and FHFA; 17 U.S.C. § 105). The following commercial or third-party copyrighted series were removed from the bundled files:

- **Removed from `data/fred_md.toml` (12 series)**:
  - S&P Dow Jones Indices: `S&P 500`, `S&P div yield`, `S&P PE ratio`
  - Moody's Investors Service: `AAA`, `BAA`, `AAAFFM`, `BAAFFM`
  - University of Michigan: `UMCSENTx`
  - CBOE Global Markets: `VIXCLSx`
  - The Conference Board / Barnichon Help-Wanted Index: `HWI`, `HWIURATIO`
  - Dow Jones / spliced WTI crude oil: `OILPRICEx`
- **Removed from `data/fred_qd.toml` (19 series)**:
  - S&P Dow Jones Indices: `S&P 500`, `S&P div yield`, `S&P PE ratio`, `SPCS10RSA`, `SPCS20RSA`
  - Moody's Investors Service: `AAA`, `BAA`, `BAA10YM`, `AAAFFM`
  - Freddie Mac: `MORTGAGE30US`, `MORTG10YRx`
  - University of Michigan: `UMCSENTx`
  - CBOE Global Markets: `VIXCLSx`
  - The Conference Board / Barnichon Help-Wanted Index: `HWIx`, `HWIURATIOx`
  - Baker, Bloom & Davis Policy Uncertainty: `USEPUINDXM`
  - Nikkei Inc.: `NIKKEI225`
  - Nasdaq, Inc.: `NASDAQCOM`
  - Dow Jones / spliced WTI crude oil: `OILPRICEx`

#### Loading the Full Official McCracken & Ng Databases Directly

Users who need the complete official FRED-MD or FRED-QD vintage (including equity, credit-spread, sentiment, and housing indices) can download the CSV directly from the Federal Reserve Bank of St. Louis under the FRED® Terms of Use at [https://research.stlouisfed.org/econ/mccracken/fred-databases/](https://research.stlouisfed.org/econ/mccracken/fred-databases/) and load it into a `TimeSeriesData` container:

```julia
using MacroEconometricModels, DelimitedFiles, Downloads

url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
raw, header = readdlm(Downloads.download(url), ',', Any; header=true)
varnames = String.(strip.(vec(header[2:end])))
tcodes   = Int.(raw[1, 2:end])             # row 1 carries FRED-MD transformation codes
dates    = String.(raw[2:end, 1])
mat      = [x isa Real ? Float64(x) : NaN for x in raw[2:end, 2:end]]

fred_full = TimeSeriesData(mat; varnames=varnames, frequency=Monthly, tcode=tcodes)
set_dates!(fred_full, dates)
```
