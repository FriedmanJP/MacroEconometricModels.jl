# Vendored plotting assets

## `d3.v7.min.js`

- **Library**: D3.js (Data-Driven Documents)
- **Version**: `7.8.5`
- **Build**: UMD, minified (`d3.min.js`)
- **Upstream URL**: <https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js>
  (identical to <https://cdn.jsdelivr.net/npm/d3@7.8.5/dist/d3.min.js>)
- **Size**: 279,633 bytes
- **SHA-256**: `d6b03aefc9f6c44c7bc78713679c78c295028fa914319119e5cc4b4954855b1c`
- **License**: ISC (Copyright 2010-2023 Mike Bostock). ISC is a permissive,
  GPL-compatible license. This file is redistributed verbatim under its own ISC
  terms; it is **not** relicensed under this package's GPL-3.0-or-later.

## Why vendored?

`PlotOutput` inlines this file into every generated HTML document so that a saved
plot renders **offline** — behind a corporate proxy, on a plane, in archived
supplementary materials, and inside Documenter iframes protected from third-party
CDN outages. This satisfies plotrule **A12** ("A saved plot must render offline.
D3 must be vendored ... not loaded from a CDN `<script src>`").

The blob is read lazily (once, on first plot) via `_d3_source()` in
`src/plotting/types.jl` and cached in a `Ref`. It is inlined once per document
(multi-panel figures share one skeleton), never once per panel.

## Updating

To bump the D3 version: download the matching `d3.min.js`, replace this file,
and update the version / size / SHA-256 above. Verify the new file still defines
`d3` and carries the `// https://d3js.org v<version>` header banner (the inline
self-containment test keys on `d3js.org`).
