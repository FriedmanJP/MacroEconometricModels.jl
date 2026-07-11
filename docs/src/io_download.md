# Downloading IO/MRIO Data

The module can download several public multi-regional IO (MRIO) databases,
modeled on the Python `pymrio` toolkit. Downloading only *fetches* the files;
converting an archive into an [`IOData`](@ref) is done separately with
[`parse_io`](@ref) (which needs the optional `ZipFile`/`XLSX` packages for
zipped or Excel sources).

## Listing sources

```@example iod
using MacroEconometricModels
list_io_sources()
```

The supported sources are OECD ICIO, WIOD 2013, EXIOBASE 3, EORA26 (login
required), and GLORIA.

## Downloading

`download_io` dispatches by source symbol and returns an [`IOMetaData`](@ref)
log recording every `url => file` it fetched. Per-source wrappers are also
exported.

```julia
# Download the latest OECD ICIO tables (2016–2020 block) into ./mrio
meta = download_io(:oecd; storage_folder="mrio", version="v2023")

# only specific years
download_oecd("mrio"; version="v2016", years=[2000, 2001])

# EXIOBASE 3 (product-by-product), all years on the Zenodo record
download_exiobase3("mrio"; system="pxp")

# EORA26 needs an account
download_eora26("mrio"; email="you@example.com", password="…")
```

Downloads skip files that already exist unless `overwrite_existing=true`. All
source URLs use HTTPS.

## Integrity verification

Each download is checked against a registry of expected SHA-256 digests,
`MacroEconometricModels.IO_CHECKSUMS`. When a URL has a registered digest and the
downloaded bytes do not match, `download_io` throws — the archive is corrupt or
has been substituted. When a URL has *no* registered digest (the default, until
maintainers populate the registry), the file downloads with a warning that its
integrity was not verified. Pass `verify=false` to skip the check entirely.

```julia
# default: verify=true — a mismatch throws, an unregistered URL warns
download_io(:oecd; storage_folder="mrio", version="v2016", years=[2000])

# skip integrity checks
download_io(:oecd; storage_folder="mrio", version="v2016", years=[2000], verify=false)

# compute a freshly downloaded archive's digest to add to the registry
io_file_digest("mrio/ICIO_v2016_2000.zip")
```

## Parsing archives into `IOData`

`parse_io` detects the file type and dispatches. CSV/TSV files parse in-core;
`.zip` and `.xlsx` require the relevant package to be loaded:

```julia
using ZipFile, XLSX        # enable the parser extensions

io = parse_io("mrio/ICIO_v2023_2016-2020.zip"; source=:oecd, member="…csv",
              n_sectors=45, n_fd=1)
```

If the required package is not loaded, `parse_io` raises an actionable error
telling you to `]add ZipFile` or `]add XLSX`.

## API

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
IOMetaData
```
