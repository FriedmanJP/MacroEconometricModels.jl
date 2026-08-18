# Google Colab (precompiled)

**Google Colab only.** This directory is a parallel channel for Colab notebook
users. It is not a replacement for a normal `Pkg.add("MacroEconometricModels")`
install, and it does not change General-registry package behaviour. In-package
`PrecompileTools` (`src/precompile.jl`) still pays first-call latency for every
other platform.

Colab runtimes reset `~/.julia` on every new session. The product here is a
**prebuilt Linux x86_64 depot** (primary) and an optional **PackageCompiler
sysimage** (secondary) attached to the GitHub Release, so the first
`estimate_var` does not wait through a multi-minute cold precompile of
Optim / NonlinearSolve / JuMP / Ipopt / DataFrames.

## Pin

| Field | Value |
|---|---|
| Julia | **1.12.6** (Colab 2026.07 native runtime) |
| OS / arch | **linux-x86_64** only |
| Package | version on the matching GitHub Release tag (`vX.Y.Z`) |

Wrong Julia patch, macOS, Windows, or aarch64: use ordinary `Pkg.add` and let
`PrecompileTools` run. Do not expand this matrix unless Colab itself requires it.

Confirm the live Colab pin before advertising a new release:

```julia
VERSION   # must print 1.12.6
```

If Colab has moved, bump `JULIA_PIN` in `build_depot.sh` and
`.github/workflows/colab-precompiled-env.yml`, refresh `Manifest.toml`, and
rebuild.

## Release assets

On tag `vX.Y.Z` (and on `workflow_dispatch`):

| Asset | Role |
|---|---|
| `mem-colab-depot-vX.Y.Z-julia1.12.6-linux-x86_64.tar.zst` | Isolated depot + `env/Project.toml` + `env/Manifest.toml` |
| `MacroEM-colab-vX.Y.Z-julia1.12.6-linux-x86_64.so` | Best-effort sysimage; **absent is OK** |
| `SHA256SUMS`, `Project.toml`, `Manifest.toml` | Checksums and the env files also nested in the tarball |

Tarball layout after extract:

```text
mem-colab/
  depot/     # JULIA_DEPOT_PATH / DEPOT_PATH entry
  env/       # activate this project
```

These binaries are **not** inside the General-registry package tarball.

## Setup on Colab (depot, primary)

`DEPOT_PATH` is fixed at Julia startup. Setting `ENV["JULIA_DEPOT_PATH"]` in an
already-running kernel is too late — mutate `DEPOT_PATH` instead. A native
Colab Julia kernel should run this **before** any `using Pkg` / package load.

```julia
# 1. Fetch the depot for this package version + Julia pin.
using Downloads
import Pkg
const PKG_VERSION = "0.8.3"
const JULIA_PIN = "1.12.6"
const ASSET = "mem-colab-depot-v$(PKG_VERSION)-julia$(JULIA_PIN)-linux-x86_64.tar.zst"
const URL = "https://github.com/FriedmanJP/MacroEconometricModels.jl/releases/download/v$(PKG_VERSION)/$(ASSET)"
const DEST = "/content/mem-colab"
const TARBALL = "/tmp/$(ASSET)"

if !isdir(joinpath(DEST, "depot"))
    run(`apt-get install -y -qq zstd`)
    Downloads.download(URL, TARBALL)
    mkpath("/content")
    run(`tar -I zstd -xf $(TARBALL) -C /content`)
end

# 2. Point this session at the extracted depot, then activate the shipped env.
empty!(DEPOT_PATH)
push!(DEPOT_PATH, joinpath(DEST, "depot"))
Pkg.activate(joinpath(DEST, "env"))

# 3. Smoke.
using MacroEconometricModels
@time estimate_var(randn(60, 3), 2)
```

A notebook that already contains this cell lives at
[`colab_setup.ipynb`](colab_setup.ipynb).

### Optional: Google Drive cache (session 2+)

After the first successful extract:

```julia
# Mount Drive from a Colab Python cell first:
#   from google.colab import drive
#   drive.mount("/content/drive")
const DRIVE = "/content/drive/MyDrive/mem-colab"
if isdir("/content/drive/MyDrive") && !isdir(DRIVE)
    cp("/content/mem-colab", DRIVE; force=true)
end
```

On the next runtime, copy `DRIVE` back to `/content/mem-colab` instead of
re-downloading the release asset.

### Optional: sysimage (bash / self-installed Julia)

Interactive Colab kernels rarely expose `--sysimage`. Use the `.so` only from a
`%%bash` cell or a Julia you installed yourself, with the **same** 1.12.6 pin:

```bash
julia --sysimage=MacroEM-colab-v0.8.3-julia1.12.6-linux-x86_64.so \
      --project=/content/mem-colab/env -e 'using MacroEconometricModels; estimate_var(randn(60,3), 2)'
```

The sysimage is built with a portable `cpu_target` so typical Colab CPUs do not
SIGILL. If the Release has no `.so`, the depot tarball is still the supported
path.

## What voids the guarantee

`Pkg.update`, adding unpinned packages, or activating a different Manifest
inside the notebook rebuilds / misses pkgimages. Re-download the Release asset
or accept a cold precompile.

## Maintenance policy

| Topic | Policy |
|---|---|
| Supported matrix (v1) | One Julia version (Colab's current pin, 1.12.6) × linux-x86_64 only |
| When to rebuild | Every package release tag that claims Colab assets; when Colab's Julia patch changes; when this Manifest must change for compat or security |
| Who owns it | Release owner ensures the Colab workflow finished (or documents skip) before advertising Colab prebuilts |
| Compatibility promise | Best-effort for that Colab pin only |
| Expanding matrix | Extra OS/arch or extra Julia majors are out of scope unless Colab itself requires them |
| Size budget | Soft cap 1.5 GiB; zstd; no unrelated global packages in the depot |
| Security | Clean CI build only; publish SHA256; no secrets in the depot |
| Deprecation | When Colab moves the pin, rebuild; mark old assets unsupported in the release notes |
| Relationship to PrecompileTools | Package workload stays for all non-Colab users; keep `colab_precompile.jl` coherent with `src/precompile.jl` |
| Failure modes | Depot OK without sysimage. Neither asset → package release still valid; release notes: “Colab prebuilt unavailable” |
| Cost control | Tags / `workflow_dispatch` only — never every PR |
| User voiding the guarantee | `Pkg.update` / unpinned deps → re-download or re-precompile |

## Rebuild locally (maintainers)

CI is the supported builder (`ubuntu-latest`, Julia 1.12.6, isolated
`JULIA_DEPOT_PATH`). A Linux x86_64 machine with that Julia pin can reproduce:

```bash
# isolated depot + pack + best-effort sysimage
OUT_DIR=/tmp/colab-dist environments/colab/build_depot.sh

# depot only
SKIP_SYSIMAGE=1 OUT_DIR=/tmp/colab-dist environments/colab/build_depot.sh
```

`colab_precompile.jl` is the execution file. Extend it in lockstep with
`src/precompile.jl` whenever a Colab demo grows a new high-TTFX entry point.

To refresh the in-repo Manifest on Julia 1.12.6:

```bash
julia --project=environments/colab -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

The in-repo Manifest uses a path dep for local checks. The Manifest **inside
the Release tarball** is rewritten to the GitHub URL so Colab does not see a
CI filesystem path.

## Explicit non-goals

- Non-Colab lab / cluster / workshop distribution
- macOS, Windows, or aarch64 assets
- Committing `~/.julia/compiled` or `.so` into git
- Auto-selecting a sysimage from `Pkg.add`
- GPU / CUDA Colab images (CPU-only v1)
- Replacing or weakening in-package PrecompileTools
