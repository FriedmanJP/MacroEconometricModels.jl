# [Downloading IO/MRIO Data](@id io_download_page)

The built-in two-sector table is enough to learn the methods; real work needs a real table. The package ships downloaders for the five public multi-regional input-output (MRIO) databases, a SHA-256 integrity registry, and parsers that turn the downloaded files into an [`IOData`](@ref). The design follows the Python `pymrio` toolkit (Stadler 2021): **downloading only fetches**, and converting an archive into a table is a separate, explicit step. See [Input-Output Analysis](@ref io_page) for what to do with the result.

- **Registry**: `list_io_sources` catalogues the five sources, their versions, and their credential requirements
- **Downloaders**: `download_io` dispatches by source symbol; `download_oecd`, `download_wiod`, `download_exiobase3`, `download_eora26`, and `download_gloria` are the per-source entry points
- **Integrity**: every fetched archive is checked against `IO_CHECKSUMS`, with `io_file_digest` to populate it
- **Provenance**: each downloader returns an `IOMetaData` log recording every URL it fetched and when
- **Parsing**: `parse_io` reads CSV and TSV in-core and dispatches ZIP and XLSX to package extensions; `parse_icio` and `parse_wiod` are labeled MRIO recipes for OECD ICIO and WIOD 2013

```@setup io_download
using MacroEconometricModels
```

## Quick Start

**Recipe 1: List the available sources**

```@example io_download
list_io_sources()
```

**Recipe 2: Download a database**

```julia
meta = download_io(:oecd; storage_folder="mrio", version="v2023")
```

**Recipe 3: Fetch with an injected downloader**

```@example io_download
folder = mktempdir()
stub(url, dest; kwargs...) = (write(dest, "archive placeholder"); dest)

meta = download_oecd(folder; version="v2016", years=[2000, 2001],
                     fetch=stub, verify=false)
meta.files
```

**Recipe 4: Digest a downloaded archive**

```@example io_download
io_file_digest(joinpath(folder, "ICIO_v2016_2000.zip"))
```

**Recipe 5: Parse a delimited table**

```@example io_download
path = joinpath(folder, "table.csv")
write(path, "150.0,500.0,350.0\n200.0,100.0,1700.0\n")

parse_io(path; source=:oecd, n_sectors=2, n_fd=1,
         sectors=["Agriculture", "Manufacturing"])
```

---

## The Source Registry

`list_io_sources` prints what the package knows how to fetch. Each entry records the versions available, whether the host demands credentials, and how the URLs are obtained — a fixed table transcribed from the publisher, or an HTML scrape of a release page.

```@example io_download
list_io_sources()
```

| `source` | Database | Versions | Credentials | URL acquisition |
|----------|----------|----------|-------------|-----------------|
| `:oecd` | OECD ICIO (Yamano et al. 2023) | `v2016`, `v2018`, `v2021`, `v2023` | No | Fixed per-version URL table |
| `:wiod` | WIOD 2013 (Timmer et al. 2015) | `2013` | No | Scrape of the release page for `wiot*.xlsx` |
| `:exiobase3` | EXIOBASE 3 (Stadler et al. 2018) | `3.8.2` | No | Scrape of the Zenodo record for `IOT_YYYY_{pxp,ixi}.zip` |
| `:eora26` | EORA26 (Lenzen et al. 2013) | `26` | Yes | Manual only — interactive worldmrio.com login |
| `:gloria` | GLORIA (Lenzen et al. 2017) | `053` | No | Fixed URL set |

The two scraping sources are the fragile ones: they match a regular expression against the live release page, so a redesign upstream breaks them while the fixed-URL sources keep working. The OECD tables are grouped into multi-year blocks from version `v2021` onward, so a `years` filter matches a key such as `"2016-2020"` whenever any requested year falls inside it.

---

## Downloading

`download_io` dispatches on the source symbol and forwards the remaining keywords to the per-source downloader. Every downloader writes into `storage_folder`, skips files that already exist unless `overwrite_existing=true`, and returns the provenance log.

```julia
# Latest OECD ICIO, all year blocks
meta = download_io(:oecd; storage_folder="mrio", version="v2023")

# A single vintage and two specific years
download_oecd("mrio"; version="v2016", years=[2000, 2001])

# EXIOBASE 3, product-by-product, one year
download_exiobase3("mrio"; system="pxp", years=[2010])

# WIOD 2013 national tables and the GLORIA release set
download_wiod("mrio")
download_gloria("mrio")

# EORA26 has no automated path: this call throws and explains why
download_eora26("mrio"; email="you@example.com", password=ENV["MRIO_PASSWORD"])
```

EORA26 is registered as a source but is not downloadable from here. worldmrio.com authenticates through an interactive session that cannot be reproduced headlessly, so `download_eora26` validates the email and then throws an `ErrorException` naming the manual route — fetch the tables from the site and read them with `parse_io`. It fails loudly rather than returning an empty log that reads like success.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `storage_folder` | `AbstractString` | — | Required. Destination directory, created if absent |
| `years` | `AbstractVector`/`Nothing` | `nothing` | Restrict to these years; `nothing` fetches every file the source offers |
| `version` | `AbstractString`/`Nothing` | `nothing` | Source version; OECD only, defaults to `"v2023"` |
| `overwrite_existing` | `Bool` | `false` | Re-fetch files that already exist on disk |
| `email` / `password` | `AbstractString`/`Nothing` | `nothing` | Credentials; EORA26 only |
| `system` | `AbstractString` | `"pxp"` | EXIOBASE only: `"pxp"` product-by-product or `"ixi"` industry-by-industry |
| `verify` | `Bool` | `true` | Check each archive's SHA-256 against `IO_CHECKSUMS`; accepted but unused by `:eora26` |
| `fetch` | `Function` | `fetch_file` | File downloader, injectable |
| `fetch_text` | `Function` | `fetch_text` | HTML fetcher for the scraping sources, injectable |

Downloads go through the `Downloads` standard library, which follows redirects and sends an honest `User-Agent` naming the package and its repository rather than a spoofed browser string.

### Fetching Without the Network

Both `fetch` and `fetch_text` are ordinary function arguments, so an offline substitute can be injected for testing, for reproducible builds, or for working against a local mirror. The replacement for `fetch` takes a URL and a destination and accepts arbitrary keywords; the replacement for `fetch_text` takes a URL and returns the page body.

```@example io_download
scratch = mktempdir()
stub(url, dest; kwargs...) = (write(dest, "archive placeholder"); dest)

# The scraping sources also need the release page
page(url; kwargs...) =
    """<a href="https://www.wiod.org/protected3/data13/AUS/wiot09_row_apr12.xlsx">2009</a>"""

log_wiod = download_wiod(scratch; fetch=stub, fetch_text=page, verify=false)
log_wiod.files
```

The scraper found the one matching link on the mocked page, resolved its local filename, and recorded the pair. Filenames are derived from the URL with any query string stripped, so the Dropbox-hosted GLORIA archives whose URLs end in `?dl=0` land on disk as plain `.zip` files — a `?` is legal in a POSIX path but rejected by Windows.

---

## Integrity Verification

A downloaded archive can be corrupt or substituted, and neither failure announces itself. Each downloader therefore passes every fetched file through a SHA-256 check against `IO_CHECKSUMS`, a registry keyed by exact download URL.

| Registry state | Behaviour |
|----------------|-----------|
| Digest registered and matches | The file is accepted silently |
| Digest registered and differs | `ErrorException` naming the URL with expected and actual digests |
| No digest registered | The file is accepted with a warning that it is unverified |

`IO_CHECKSUMS` ships **empty**, because a real digest requires a maintainer to fetch each archive once and record its hash. Until it is populated, every download warns. That is the honest default: the alternative would be to claim verification that never happened.

```@example io_download
url = "https://www.oecd.org/sti/ind/ICIO2016_2000.zip"
digest = io_file_digest(joinpath(folder, "ICIO_v2016_2000.zip"))

MacroEconometricModels.IO_CHECKSUMS[url] = digest
download_oecd(folder; version="v2016", years=[2000], fetch=stub, verify=true).files
```

```@setup io_download
delete!(MacroEconometricModels.IO_CHECKSUMS, url)
```

Registering the digest turns the warning off and turns any future change in those bytes into a hard error. Pass `verify=false` to skip the check entirely, which is the right choice for a mocked fetcher or a mirror whose contents are known to differ from the publisher's.

---

## The Download Log

Every downloader returns an `IOMetaData`, the provenance record of the fetch. It is the same object carried in `IOData.meta`, so a table built from a download keeps its own audit trail.

```@example io_download
meta.source, meta.version
```

```@example io_download
meta.files
```

`history` carries one timestamped line per file, of the form `2026-08-01T10:36:03.898: downloaded ICIO_v2016_2000.zip from https://www.oecd.org/sti/ind/ICIO2016_2000.zip`. Together the two fields answer the two questions a replication package must answer: which URL each local file came from, and when it was retrieved.

| Field | Type | Description |
|-------|------|-------------|
| `source` | `String` | Database name, e.g. `"OECD ICIO"` |
| `version` | `String` | Version identifier, e.g. `"v2016"` |
| `history` | `Vector{String}` | Timestamped log lines, one per file |
| `files` | `Vector{Pair{String,String}}` | `url => local filename` for every fetched file |

---

## Parsing Files into `IOData`

`parse_io` dispatches on the file extension. Delimited text is parsed in-core; the compressed and Excel formats are handled by package extensions that load only when the optional package is present in the session.

| Extension | Parser | Requires | Keywords |
|-----------|--------|----------|----------|
| `.csv`, `.tsv`, `.txt` | In-core, via `DelimitedFiles` | — | `n_sectors` (required), `n_fd`, `sectors`, `delim` |
| `.zip` | `ZipFile` package extension | `using ZipFile` | `member`, `n_sectors`, `n_fd`, `sectors`, `delim`, `max_uncompressed` |
| `.xlsx`, `.xls` | `XLSX` package extension | `using XLSX` | `sheet`, `n_sectors`, `n_fd`, `sectors` |

All three parsers read the same block layout: the first `n_sectors` columns of the first `n_sectors` rows are the intermediate-flow matrix ``Z``, and the next `n_fd` columns are final demand ``Y``. Gross output follows from the row balance.

```@example io_download
tbl = parse_io(path; source=:oecd, n_sectors=2, n_fd=1,
               sectors=["Agriculture", "Manufacturing"])
report(tbl)
```

```@example io_download
leontief_inverse(tbl)
```

The parsed table reproduces the built-in example exactly, which is the point of the fixture: `x = [1000, 2000]` and the Leontief inverse matches the one on the [Classical Analysis](@ref io_classical_page) page. Note what the parser does *not* do — it never sees a value-added block, so it derives a single value-added row from the column balance, and it constructs the table with `check=false` because a published table trimmed to a ``Z``/``Y`` block rarely balances to machine precision.

The compressed and Excel paths take the same shape once the optional package is loaded:

```julia
using ZipFile, XLSX          # activate the parser extensions

# member selects the file inside the archive; the first member is the default
io_zip = parse_io("mrio/ICIO_v2023_2016-2020.zip"; source=:oecd,
                  member="ICIO2023_2016.csv", n_sectors=45, n_fd=1)

# sheet selects the worksheet by position
io_xlsx = parse_io("mrio/wiot09_row_apr12.xlsx"; source=:wiod,
                   sheet=1, n_sectors=35, n_fd=5)
```

Without the package loaded, `parse_io` raises an actionable error naming the package to install rather than a `MethodError`.

!!! note "Zip-bomb guard"
    The ZIP parser refuses to read a member whose *declared* uncompressed size exceeds
    `max_uncompressed`, one gigabyte by default, before allocating anything. A genuinely
    large MRIO table needs the cap raised explicitly — a deliberate act rather than an
    accidental out-of-memory kill.

---

## Source-Specific Recipes

`parse_io` is deliberately generic: it needs `n_sectors` and does not recover region or final-demand labels. Two recipes wrap the OECD ICIO and WIOD 2013 layouts so a downloaded archive becomes a fully labeled multi-region [`IOData`](@ref) ready for [`export_decomposition`](@ref) and friends.

### OECD ICIO — `parse_icio`

```julia
using ZipFile   # only needed for .zip archives

# Single-year CSV (any OECD ICIO release: 2016 / 2018 / 2021 / 2023 label style)
io = parse_icio("ICIO2023_2018.csv"; year=2018)

# Multi-year zip: year= selects the member (or pass member= explicitly)
io = parse_icio("ICIO_v2023_2016-2020.zip"; year=2018)

# CN1… / MX1… sub-national blocks are aggregated into CHN / MEX by default
# (destination FD columns are merged in the same pass so Y stays region-blocked)
io_raw = parse_icio("ICIO2016_2010.csv"; aggregate_cn_mx=false)
```

The recipe reads the `REGION_SECTOR` row/column index, splits final-demand columns matching `HFCE|NPISH|GGFC|GFCF|INVNT|…`, takes value-added rows (`VA`, `TLS`, `VALU*`, `TAX*`), and drops `OUT`/`TOTAL`/`OUTPUT` margins. Sector labels on the result are the full `REGION_SECTOR` product so
`length(io.regions) · (length(io.x) ÷ length(io.regions)) == length(io.x)`.

### WIOD 2013 — `parse_wiod`

```julia
using XLSX

io = parse_wiod("wiot09_row_apr12.xlsx")          # year inferred from the filename
io = parse_wiod("wiod_folder"; year=2009)         # picks wiot09*.xlsx in the folder
```

WIOT workbooks carry a four-row / four-column header (ISIC code, name, region, c-code). The interindustry block ends at the last column/row whose c-code equals `last_interind_code` (default `"c35"` for the official 35-sector tables). Factor-input rows below that block become `va` categories; the rightmost total column is discarded. Pass `names=:full` or `names=:c_codes` to change the sector token used inside the `REGION_*` product labels.

```@example io_download
# Synthetic ICIO-style CSV (same layout as a real OECD release, tiny)
icio_csv = joinpath(folder, "ICIO_toy_2010.csv")
write(icio_csv, """
,USA_S1,USA_S2,CHN_S1,CHN_S2,USA_HFCE,CHN_HFCE,OUT
USA_S1,10,5,1,0,20,2,38
USA_S2,3,15,0,1,25,1,45
CHN_S1,0,1,12,4,1,30,48
CHN_S2,1,0,3,10,2,20,36
VA,24,24,32,21,0,0,101
OUTPUT,38,45,48,36,48,53,268
""")
icio_mrio = parse_icio(icio_csv; year=2010, check=true)
(length(icio_mrio.regions), length(icio_mrio.x), icio_mrio.regions)
```

```@example io_download
region_block(icio_mrio, "USA", "CHN")
```

---

## Complete Example

This example runs the acquisition pipeline end to end against a local fixture: fetch, register a digest, re-fetch under verification, inspect the provenance log, parse, and analyse.

```@example io_download
work = mktempdir()
fetcher(url, dest; kwargs...) = (write(dest, "150,500,350\n200,100,1700\n"); dest)

fetch_log = download_oecd(work; version="v2016", years=[2005], fetch=fetcher, verify=false)
fetch_log.files
```

```@example io_download
archive = joinpath(work, "ICIO_v2016_2005.zip")
src_url = first(fetch_log.files).first

MacroEconometricModels.IO_CHECKSUMS[src_url] = io_file_digest(archive)
download_oecd(work; version="v2016", years=[2005], fetch=fetcher, verify=true).source
```

```@example io_download
table_path = joinpath(work, "icio_2005.csv")
cp(archive, table_path)

icio = parse_io(table_path; source=:oecd, n_sectors=2, n_fd=1,
                sectors=["Agriculture", "Manufacturing"])
report(icio)
```

```@example io_download
report(multipliers(icio; kind=:output, type=:I))
```

```@setup io_download
delete!(MacroEconometricModels.IO_CHECKSUMS, src_url)
```

The pipeline separates three concerns that are easy to conflate. Fetching produces bytes and a log; verification decides whether those bytes are the ones the publisher shipped; parsing decides how to read them as a table. Only the last step needs to know that the numbers are an input-output table at all, which is why swapping in a different source means writing a parser call, not a new downloader. Once `icio` exists it is an ordinary `IOData`, and the output multipliers of 1.518 and 1.452 are the same ones the built-in example produces.

---

## Common Pitfalls

1. **Downloading does not parse.** `download_io` returns an `IOMetaData`, never an `IOData`. The two steps are deliberately separate, following the `pymrio` convention, because a single MRIO archive contains many tables and only the caller knows which one is wanted.

2. **`:eora26` accepts `verify` but never downloads.** `download_eora26` takes the same `verify`/`fetch` keywords as every other source, so `download_io(:eora26; storage_folder=…, verify=false)` type-checks; it then throws an `ErrorException` because the automated fetch is not implemented. The keyword parity is for uniform forwarding through `download_io`, not a promise that the source works.

3. **`IO_CHECKSUMS` is empty, so every real download warns.** The warning means "not verified", not "verification failed". Silence it by registering the digest with `io_file_digest` after a first trusted fetch, or by passing `verify=false` when the check is not wanted.

4. **`parse_io` records `source` and `year`, so pass `year` when it matters.** The returned `IOData` carries `source` as the string form of the symbol given, and `year` exactly as supplied. `year` is optional and defaults to `nothing`, so a table parsed without it keeps no vintage — give `year=` at parse time rather than rebuilding the table afterwards.

5. **`n_sectors` is required for delimited files.** `_parse_csv_io` has no default, so omitting it is a `MethodError`. The ZIP and XLSX parsers default `n_sectors=0`, which means "use every row", and silently mis-slice a file that carries label rows or a value-added block.

6. **The optional package must be `using`-ed, not merely installed.** Package extensions load when the weak dependency enters the session. `]add ZipFile` alone leaves `parse_io` on the stub method that raises the install instruction; `using ZipFile` is what activates the real parser.

7. **Existing files are skipped, including truncated ones.** `fetch_file` returns early whenever the destination exists and `overwrite_existing` is false, so an interrupted download is never repaired by re-running the same call. Delete the partial file or pass `overwrite_existing=true`.

8. **EORA26 throws instead of fetching.** The downloader validates the email — an empty one is an `ArgumentError` — and then raises an `ErrorException` pointing at the manual download. Wrap the call in `try`/`catch` if a batch script iterates over every registered source.

9. **Scraped sources depend on the publisher's page.** WIOD and EXIOBASE URLs come from a regular expression matched against a live HTML page. A page redesign yields an empty `files` list rather than an error, so check `length(meta.files)` before assuming a download succeeded.

10. **`parse_icio` / `parse_wiod` are layout-specific.** They expect the OECD ICIO `REGION_SECTOR` index and the WIOD 2013 four-row header, respectively. A hand-trimmed numeric block still goes through `parse_io`. For reduced WIOT fixtures, pass `last_interind_code=` (default `"c35"`).

---

## API Reference

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
IOMetaData
```

---

## References

- Lenzen, M., Moran, D., Kanemoto, K., & Geschke, A. (2013). Building Eora: A Global Multi-Region Input-Output Database at High Country and Sector Resolution.
  *Economic Systems Research*, 25(1), 20--49. [DOI](https://doi.org/10.1080/09535314.2013.769938)

- Lenzen, M., Geschke, A., Abd Rahman, M. D., Xiao, Y., Fry, J., Reyes, R., et al. (2017). The Global MRIO Lab -- Charting the World Economy.
  *Economic Systems Research*, 29(2), 158--186. [DOI](https://doi.org/10.1080/09535314.2017.1301887)

- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions* (2nd ed.).
  Cambridge University Press. ISBN 978-0-521-51713-3. [DOI](https://doi.org/10.1017/CBO9780511626982)

- Stadler, K. (2021). Pymrio -- A Python Based Multi-Regional Input-Output Analysis Toolbox.
  *Journal of Open Research Software*, 9(1), 8. [DOI](https://doi.org/10.5334/jors.251)

- Stadler, K., Wood, R., Bulavskaya, T., Sodersten, C.-J., Simas, M., Schmidt, S., et al. (2018).
  EXIOBASE 3: Developing a Time Series of Detailed Environmentally Extended Multi-Regional Input-Output Tables.
  *Journal of Industrial Ecology*, 22(3), 502--515. [DOI](https://doi.org/10.1111/jiec.12715)

- Timmer, M. P., Dietzenbacher, E., Los, B., Stehrer, R., & de Vries, G. J. (2015). An Illustrated User Guide to the World Input-Output Database: The Case of Global Automotive Production.
  *Review of International Economics*, 23(3), 575--605. [DOI](https://doi.org/10.1111/roie.12178)

- Yamano, N., Alsamawi, A., Webb, C., Cimper, A., Zurcher, C., & Chiapin Pechansky, R. (2023). Development of the OECD Inter-Country Input-Output Database 2023.
  *OECD Science, Technology and Industry Working Papers*. [DOI](https://doi.org/10.1787/5a5d0665-en)
