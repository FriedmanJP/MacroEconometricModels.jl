#!/usr/bin/env bash
# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# Build the Colab-only depot tarball (and optionally a PackageCompiler sysimage).
# Google Colab only (#610) — linux-x86_64, one Julia pin. Do not commit the
# output; CI attaches it to the GitHub Release.
#
# Usage:
#   environments/colab/build_depot.sh
#
# Environment:
#   JULIA           julia executable (default: julia)
#   OUT_DIR         output directory (default: ./colab-dist)
#   PKG_VERSION     override version in asset names (default: root Project.toml)
#   JULIA_PIN       expected VERSION string (default: 1.12.6)
#   SKIP_SYSIMAGE   set to 1 to skip PackageCompiler
#   GITHUB_REPO     used when rewriting Manifest repo-url
#                   (default: FriedmanJP/MacroEconometricModels.jl)

set -euo pipefail

JULIA_PIN="${JULIA_PIN:-1.12.6}"
JULIA="${JULIA:-julia}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/mem-colab-dist}"
GITHUB_REPO="${GITHUB_REPO:-FriedmanJP/MacroEconometricModels.jl}"
GITHUB_URL="https://github.com/${GITHUB_REPO}.git"
SIZE_WARN_BYTES=$((1500 * 1024 * 1024))  # 1.5 GiB soft cap
CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)"

if ! command -v zstd >/dev/null 2>&1; then
    echo "error: zstd is required to pack the depot tarball" >&2
    exit 1
fi

JULIA_VER="$("${JULIA}" --startup-file=no -e 'print(VERSION)')"
if [[ "${JULIA_VER}" != "${JULIA_PIN}" ]]; then
    echo "warning: julia reports VERSION=${JULIA_VER}, Colab pin is ${JULIA_PIN}" >&2
    echo "warning: compiled images will not match Colab's native runtime" >&2
fi

if [[ -z "${PKG_VERSION:-}" ]]; then
    PKG_VERSION="$("${JULIA}" --startup-file=no --project="${REPO_ROOT}" -e 'using Pkg; print(Pkg.project().version)')"
fi

REV="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/mem-colab.XXXXXX")"
cleanup() { rm -rf "${STAGE}"; }
trap cleanup EXIT

BUNDLE="${STAGE}/mem-colab"
DEPOT="${BUNDLE}/depot"
ENV_DIR="${BUNDLE}/env"
mkdir -p "${DEPOT}" "${ENV_DIR}" "${OUT_DIR}"

export JULIA_DEPOT_PATH="${DEPOT}"
export JULIA_PKG_PRECOMPILE_AUTO=1

cp "${SCRIPT_DIR}/Project.toml" "${ENV_DIR}/Project.toml"

echo "==> instantiate + add MacroEconometricModels@${REV} into isolated depot"
"${JULIA}" --startup-file=no --project="${ENV_DIR}" -e "
    using Pkg
    Pkg.add(; url=raw\"${REPO_ROOT}\", rev=raw\"${REV}\")
    Pkg.instantiate()
    Pkg.precompile()
"

# Rewrite the local path so a Colab instantiate can fetch from GitHub if the
# depot is incomplete. The packaged depot already contains the tree.
MANIFEST="${ENV_DIR}/Manifest.toml"
if [[ -f "${MANIFEST}" ]]; then
    "${JULIA}" --startup-file=no -e "
        path = raw\"${MANIFEST}\"
        text = read(path, String)
        text = replace(text, raw\"${REPO_ROOT}\" => raw\"${GITHUB_URL}\")
        write(path, text)
    "
fi

echo "==> run colab_precompile.jl"
"${JULIA}" --startup-file=no --project="${ENV_DIR}" "${SCRIPT_DIR}/colab_precompile.jl"

echo "==> smoke estimate_var (warm depot)"
WARM_TIME="$("${JULIA}" --startup-file=no --project="${ENV_DIR}" -e '
    using MacroEconometricModels, Random
    t = @elapsed estimate_var(randn(MersenneTwister(0), 60, 3), 2)
    println(round(t; digits=3))
')"
echo "warm estimate_var: ${WARM_TIME}s"

# Drop caches that are not needed to load the package.
rm -rf "${DEPOT}/logs" "${DEPOT}/compiled/v${JULIA_VER}/PackageCompiler" || true

ASSET_STEM="mem-colab-depot-v${PKG_VERSION}-julia${JULIA_PIN}-linux-x86_64"
TARBALL="${OUT_DIR}/${ASSET_STEM}.tar.zst"
echo "==> pack ${TARBALL}"
tar -I 'zstd -T0' -cf "${TARBALL}" -C "${STAGE}" mem-colab

TARBALL_BYTES="$(wc -c < "${TARBALL}" | tr -d ' ')"
if (( TARBALL_BYTES > SIZE_WARN_BYTES )); then
    echo "warning: depot tarball is ${TARBALL_BYTES} bytes (> 1.5 GiB soft cap)" >&2
fi

SYSIMAGE_NAME="MacroEM-colab-v${PKG_VERSION}-julia${JULIA_PIN}-linux-x86_64.so"
SYSIMAGE_PATH="${OUT_DIR}/${SYSIMAGE_NAME}"
SYSIMAGE_STATUS="skipped"

if [[ "${SKIP_SYSIMAGE:-0}" != "1" ]]; then
    echo "==> PackageCompiler sysimage (best-effort)"
    if "${JULIA}" --startup-file=no --project="${ENV_DIR}" -e "
        using Pkg
        Pkg.add(\"PackageCompiler\")
        using PackageCompiler
        PackageCompiler.create_sysimage(
            [:MacroEconometricModels, :DataFrames];
            sysimage_path=raw\"${SYSIMAGE_PATH}\",
            project=raw\"${ENV_DIR}\",
            precompile_execution_file=raw\"${SCRIPT_DIR}/colab_precompile.jl\",
            cpu_target=raw\"${CPU_TARGET}\",
        )
    "; then
        SYSIMAGE_STATUS="built"
    else
        echo "warning: PackageCompiler failed; publishing depot without sysimage" >&2
        SYSIMAGE_STATUS="failed"
        rm -f "${SYSIMAGE_PATH}"
    fi
fi

SUMS="${OUT_DIR}/SHA256SUMS"
{
    (cd "${OUT_DIR}" && shasum -a 256 "$(basename "${TARBALL}")")
    if [[ -f "${SYSIMAGE_PATH}" ]]; then
        (cd "${OUT_DIR}" && shasum -a 256 "${SYSIMAGE_NAME}")
    fi
    (cd "${ENV_DIR}" && shasum -a 256 Project.toml Manifest.toml)
} > "${SUMS}"

cp "${ENV_DIR}/Project.toml" "${OUT_DIR}/Project.toml"
cp "${ENV_DIR}/Manifest.toml" "${OUT_DIR}/Manifest.toml"

{
    echo "## Colab precompiled assets"
    echo
    echo "- Julia pin: \`${JULIA_PIN}\` (this build: \`${JULIA_VER}\`)"
    echo "- Package version: \`${PKG_VERSION}\`"
    echo "- Git rev: \`${REV}\`"
    echo "- Depot tarball: \`$(basename "${TARBALL}")\` (${TARBALL_BYTES} bytes)"
    echo "- Warm \`estimate_var\`: ${WARM_TIME}s"
    echo "- Sysimage: ${SYSIMAGE_STATUS}"
} | tee "${OUT_DIR}/SUMMARY.md"

echo "==> done: ${OUT_DIR}"
ls -lh "${OUT_DIR}"
