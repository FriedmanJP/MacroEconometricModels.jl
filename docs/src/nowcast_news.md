# [News Decomposition](@id nowcast_news_page)

Every data release moves the nowcast, and the question a forecaster is actually asked is not *what is the number* but *why did it change*. The news decomposition (Bańbura & Modugno 2014) answers it by splitting the revision between two data vintages into a contribution from each newly published observation, weighted by how surprising that observation was relative to what the model already expected. Releases that merely confirm the model's forecast contribute nothing, however large the number itself.

The decomposition is a property of the state-space representation, so it requires an estimated [DFM](@ref nowcast_dfm_page). For the shared data layout and the `nowcast()` interface, see [Nowcasting](@ref nowcast_page).

```@setup nc_news
using MacroEconometricModels, Random
Random.seed!(42)
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-59:end, :]
nM, nQ = 4, 1
N = nM + nQ
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN
T_obs = size(Y, 1)
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1)
```

The examples reuse the FRED-MD panel of the [DFM page](@ref nowcast_dfm_page) — four monthly indicators, one quarterly target — with `dfm` estimated on it. The old vintage is constructed by blanking the most recent month of the monthly indicators, so publishing them is the event whose effect is decomposed.

## Quick Start

**Recipe 1: Decompose a revision across three releases**

```@example nc_news
X_old = copy(Y)
X_old[end, 1:3] .= NaN          # INDPRO, UNRATE and CPI not yet published
news = nowcast_news(Y, X_old, dfm, T_obs; target_var=N)
report(news)
```

**Recipe 2: Aggregate impacts into named groups**

```@example nc_news
groups = [1, 1, 2, 2, 2]        # real activity, then nominal
news_grp = nowcast_news(Y, X_old, dfm, T_obs; target_var=N,
                        groups=groups, group_names=["Real", "Nominal"])
[g => round(v, digits=5) for (g, v) in zip(news_grp.group_names, news_grp.group_impacts)]
```

**Recipe 3: Per-release impacts as a named vector**

```@example nc_news
[r => round(v, digits=5) for (r, v) in zip(news.variable_names, news.impact_news)]
```

**Recipe 4: Track the nowcast across a release calendar**

```@example nc_news
base = copy(Y)
base[end, 1:4] .= NaN
varnames = ["INDPRO", "UNRATE", "CPI", "M2"]

map(1:4) do col
    vintage = copy(base)
    vintage[end, col] = Y[end, col]
    step = nowcast_news(vintage, base, dfm, T_obs; target_var=N)
    base[end, col] = Y[end, col]        # carry the release into the next vintage
    (release=varnames[col], nowcast=round(step.new_nowcast, digits=4),
     delta=round(step.new_nowcast - step.old_nowcast, digits=4))
end
```

---

## The News Concept

Let ``\mathcal{J}`` collect the positions ``(t_k, v_k)`` that are missing in the old vintage and observed in the new one. The **news** carried by release ``k`` is the part of its value that the old vintage could not have predicted:

```math
I_k = x_{v_k, t_k}^{\text{new}} - C_{v_k}' \, \hat{z}_{t_k \mid \text{old}}
```

where:
- ``x_{v_k, t_k}^{\text{new}}`` is the newly published (standardized) value
- ``C_{v_k}`` is the observation-equation row for variable ``v_k``
- ``\hat{z}_{t_k \mid \text{old}}`` is the smoothed state at ``t_k`` given the old vintage

Releases do not arrive one at a time in general, and two indicators published together carry overlapping information about the same factors. The decomposition therefore solves for all releases **jointly** rather than applying a scalar Kalman gain to each in turn:

```math
b = \operatorname{Var}(I)^{-1} \operatorname{Cov}(I, F), \qquad
\text{impact}_k = b_k \, I_k \, W_{\text{target}}
```

where:
- ``\operatorname{Var}(I)_{k\ell} = C_{v_k}' \operatorname{Cov}(z_{t_k}, z_{t_\ell}) C_{v_\ell} + R_{v_k v_\ell} \mathbb{1}_{\{t_k = t_\ell\}}`` is the innovation covariance
- ``\operatorname{Cov}(I, F)_k = C_{\text{target}}' \operatorname{Cov}(z_\tau, z_{t_k}) C_{v_k}`` links each release to the target at the nowcast period ``\tau``
- ``\operatorname{Cov}(z_{t}, z_{s})`` comes from the old-vintage smoother, which returns lagged cross-covariances alongside the usual smoothed variances
- ``W_{\text{target}}`` is the target's standard deviation, restoring original units

Solving jointly is what makes the attribution well posed: the weights split shared information across the releases that carry it, so the answer does not depend on the order in which the releases are listed, and two perfectly collinear releases cannot both be credited with the same move.

!!! note "Reading the sign"
    A positive `impact_news[k]` means release ``k`` came in above what the old vintage implied and pushed the nowcast up. The magnitude combines surprise with relevance: a large surprise in a series that loads weakly on the target's factors moves the nowcast less than a small surprise in a series that loads heavily.

---

## Usage

`nowcast_news` takes the new vintage, the old vintage, an estimated `NowcastDFM`, and the period whose nowcast is being decomposed. The two vintages must have identical dimensions; the old one is the more incomplete of the pair.

```@example nc_news
X_old = copy(Y)
X_old[end, 1:3] .= NaN
news = nowcast_news(Y, X_old, dfm, T_obs; target_var=N)
report(news)
```

Publishing the three monthly indicators lifts the current-quarter estimate from 0.0368 to 0.0375. The CPI release accounts for 0.0006 of the 0.0007 revision and industrial production for a further 0.0003, while unemployment came in on the strong side of the model's expectation and subtracts 0.0002. The three impacts sum to the revision to within ``6 \times 10^{-18}``: with the DFM parameters held fixed across vintages, the joint news system reproduces the smoother's own answer exactly, so `impact_reestimation` is numerical noise rather than an unexplained residual.

The **total revision** splits three ways in principle:

```math
\hat{y}^{\text{new}} - \hat{y}^{\text{old}}
  = \underbrace{\textstyle\sum_{k} \text{impact}_k}_{\texttt{impact\_news}}
  + \underbrace{\Delta_{\text{revision}}}_{\texttt{impact\_revision}}
  + \underbrace{\Delta_{\text{re-estimation}}}_{\texttt{impact\_reestimation}}
```

!!! warning "`impact_revision` is always zero"
    The current implementation identifies news only at positions that were missing in the old vintage. A value that was already published and has since been *revised* is not detected as a release, so its effect on the nowcast lands in `impact_reestimation` rather than in `impact_revision`, which is returned as zero unconditionally. Comparing vintages that contain genuine back-revisions therefore gives a residual that is real, not numerical.

### Grouping Releases

Passing `groups` aggregates the per-release impacts into sectors. The vector assigns a group index to each **variable** (not to each release), and `group_names` labels them.

```@example nc_news
groups = [1, 1, 2, 2, 2]
news_grp = nowcast_news(Y, X_old, dfm, T_obs; target_var=N,
                        groups=groups, group_names=["Real", "Nominal"])
[g => round(v, digits=5) for (g, v) in zip(news_grp.group_names, news_grp.group_impacts)]
```

The real-activity block contributes 0.00015 net — industrial production and unemployment nearly cancel — against 0.00057 from the nominal block, which here is CPI alone. Netting within a group is the point of the view: it answers whether the quarter was revised on real or on nominal news, not which individual series moved.

!!! note "Default grouping"
    With `groups` omitted, `group_impacts` has one entry per **variable** in the panel, not one per group, and `group_names` defaults to `"Var1"`, `"Var2"`, … . Variables with no new release get a zero entry. When `groups` is supplied, the length of `group_names` must equal `maximum(groups)`; otherwise it must equal ``N``.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `target_var` | `Int` | `size(X_new, 2)` | Column index of the variable being nowcast |
| `groups` | `Vector{Int}` or `nothing` | `nothing` | Group index per variable; without it, impacts are reported per variable |
| `group_names` | `Vector{String}` or `nothing` | `nothing` | Group labels; defaults to `"Var1"`, `"Var2"`, … |

The third positional argument after the model, `target_period`, selects the period whose nowcast is decomposed and must lie in ``1 \ldots T_{\text{obs}}``. Passing `T_obs` decomposes the current-quarter estimate, which is the usual case.

### NowcastNews Return Values

| Field | Type | Description |
|-------|------|-------------|
| `old_nowcast` | `T` | Nowcast implied by the old vintage, in original units |
| `new_nowcast` | `T` | Nowcast implied by the new vintage |
| `impact_news` | `Vector{T}` | Impact of each new release, one entry per element of ``\mathcal{J}`` |
| `impact_revision` | `T` | Impact of data revisions; always zero in the current implementation |
| `impact_reestimation` | `T` | Residual: total revision minus the news impacts |
| `group_impacts` | `Vector{T}` | News aggregated by group, or per variable when `groups` is omitted |
| `group_names` | `Vector{String}` | Labels matching `group_impacts` |
| `variable_names` | `Vector{String}` | Release identifiers, formatted `"Var{j}_t{t}"` |

---

## Visualization

`plot_result` renders a `NowcastNews` in three views:

```julia
plot_result(news_grp)                    # :releases — one bar per release (default)
plot_result(news_grp; view=:groups)      # stacked bar by group
plot_result(news_grp; view=:individual)  # sorted by absolute impact
```

The `:groups` view stacks each group's net contribution into a single revision bar, so the question "real or nominal?" is answered by which segment dominates:

```@raw html
<iframe src="../assets/plots/nowcast_news_groups.html" width="100%" height="350" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The `:individual` view drops the grouping and ranks releases by absolute impact, which is the view to reach for when the question is which single series moved the number:

```@raw html
<iframe src="../assets/plots/nowcast_news_individual.html" width="100%" height="350" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

All three titles carry the old nowcast, the new nowcast and the revision, so a chart is self-contained when it is lifted into a briefing.

---

## Complete Example

```@example nc_news
# === Step 1: The estimated DFM the decomposition runs against ===
report(dfm)
```

```@example nc_news
# === Step 2: Walk a four-release calendar, one publication at a time ===
base = copy(Y)
base[end, 1:4] .= NaN
varnames = ["INDPRO", "UNRATE", "CPI", "M2"]

map(1:4) do col
    vintage = copy(base)
    vintage[end, col] = Y[end, col]
    step = nowcast_news(vintage, base, dfm, T_obs; target_var=N)
    base[end, col] = Y[end, col]
    (release=varnames[col], old=round(step.old_nowcast, digits=4),
     new=round(step.new_nowcast, digits=4),
     delta=round(step.new_nowcast - step.old_nowcast, digits=4))
end
```

```@example nc_news
# === Step 3: Decompose the same four releases jointly ===
X_old4 = copy(Y)
X_old4[end, 1:4] .= NaN
news4 = nowcast_news(Y, X_old4, dfm, T_obs; target_var=N)
report(news4)
```

**Interpretation.** Released one at a time, the four indicators move the nowcast by +0.0003, ``-``0.0002, +0.0006 and ``-``0.0005, walking it from 0.0373 to 0.0375. Decomposed jointly, the same four releases produce a total revision of 0.0002 with CPI (+0.0006) and M2 (``-``0.0005) as the largest opposing contributions — the sequential deltas and the joint impacts agree here because these releases carry largely distinct information. They need not agree in general: a sequential walk credits whichever series is published first with the information two series share, whereas the joint system splits it, which is why the joint decomposition is the one to quote. The near-cancellation across the four is itself the result — the month's data were collectively uninformative about the quarter, even though individual series surprised in both directions.

---

## Common Pitfalls

1. **The vintages must be the same size.** `X_new` and `X_old` are compared element by element, and a size mismatch throws. Build the old vintage by copying the new one and blanking entries, never by truncating rows.

2. **Only positions that were `NaN` and became observed count as news.** A position observed in both vintages contributes nothing to `impact_news` no matter how much its value changed; a position missing in both is ignored. This is what makes back-revisions invisible to the decomposition — see the warning above.

3. **Parameters are held fixed across vintages.** The function re-runs the Kalman smoother on both vintages with the same estimated DFM; it does not re-estimate. That is the correct experiment for attributing a revision to data, but it means a nowcast that changed because the model was re-fitted is not decomposed by this function.

4. **Only `NowcastDFM` is supported.** The weights come from the state-space representation, so there is no method for `NowcastBVAR` or `NowcastBridge`. To decompose revisions for those, nowcast the same panel with a DFM and decompose that.

5. **Release identifiers are positional.** `variable_names` entries read `"Var{j}_t{t}"` for column ``j`` at row ``t``; the function never sees your column labels. Zip them against your own variable names, as Recipe 3 does, before showing them to anyone.

6. **`groups` indexes variables, not releases.** The vector has one entry per column of the panel even when only a few columns carry new releases, and `maximum(groups)` sets the length of `group_impacts`. Sizing it to the number of releases throws or silently mislabels.

---

## References

- Bańbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29 (1): 133--160. [https://doi.org/10.1002/jae.2306](https://doi.org/10.1002/jae.2306)
- Bańbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2011. "Nowcasting." In *The Oxford Handbook of Economic Forecasting*, 193--224. Oxford: Oxford University Press. [https://doi.org/10.1093/oxfordhb/9780195398649.013.0008](https://doi.org/10.1093/oxfordhb/9780195398649.013.0008)
- Giannone, Domenico, Lucrezia Reichlin, and David Small. 2008. "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics* 55 (4): 665--676. [https://doi.org/10.1016/j.jmoneco.2008.05.010](https://doi.org/10.1016/j.jmoneco.2008.05.010)
