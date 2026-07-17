### Summary

The panel regression / panel IV `report()` output renders the covariance type
via raw `string(m.cov_type)`, so it prints the internal symbol (`cluster`,
`twoway`, `driscoll_kraay`) instead of the human-readable label the rest of the
package uses. This is a residual gap from #265 (shared display conventions),
which routed the cross-sectional estimators' `cov_type` through `_label` but did
not reach the panel display bodies.

### Evidence

The `Cov. type` field renders the raw symbol:

```
 Panel Regression — Fixed Effects
 ...
 Cov. type    cluster        <-- should read "Cluster-robust"
```

Surfaced by the #275 display goldens harness (`test/display/goldens/panel_fe.txt`),
which snapshotted `Cov. type cluster`.

### Source sites

- `src/preg/types.jl:417` — `PanelRegModel` show: `"Cov. type"      string(m.cov_type)`
- `src/preg/types.jl:470` — `PanelIVModel` show: `"Cov. type"        string(m.cov_type)`

Correct pattern already used cross-sectionally:

- `src/reg/types.jl:309` — `"Cov. type"    _label(m.cov_type)`

### Fix

1. Route both panel sites through `_label(m.cov_type)` instead of `string(...)`.
2. Add the panel-only cov types to `_LABELS` in `src/core/display.jl` so they
   render as proper names (currently only `:ols`→"OLS" and `:cluster`→"Cluster-robust"
   are present; `:twoway`/`:driscoll_kraay` fall through to the title-case fallback,
   giving "Twoway" / "Driscoll Kraay"):
   - `:twoway => "Two-way"`
   - `:driscoll_kraay => "Driscoll–Kraay"`
3. Update the `panel_fe` golden and any panel display test that asserts the raw
   string.

Cosmetic (the covariance matrix and SEs are correct — only the label string is
raw), hence severity:low.

Found during v0.6.6 / #275 (PR #406).
