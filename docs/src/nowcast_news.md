# [News Decomposition](@id nowcast_news_page)

When new data releases arrive, the nowcast changes. The **news decomposition** (Banbura and Modugno 2014) attributes this revision to individual data releases, answering a central question in real-time forecasting: *which releases drove the revision?* For the underlying DFM model, see [Nowcasting](@ref).

```@setup nc_news
using MacroEconometricModels, Random
Random.seed!(42)
```

## Quick Start

```@example nc_news
# Standard mixed-frequency data setup (used throughout this page)
fred = load_example(:fred_md)
nc_md = fred[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
Y = to_matrix(apply_tcode(nc_md))
Y = Y[all.(isfinite, eachrow(Y)), :]
Y = Y[end-99:end, :]
nM, nQ = 4, 1
for t in 1:size(Y, 1)
    if mod(t, 3) != 0
        Y[t, end] = NaN
    end
end
Y[end, end] = NaN

# Estimate DFM (required for news decomposition)
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1)
T_obs = size(Y, 1)
N = size(Y, 2)
nothing # hide
```

**Recipe 1: Basic news decomposition**

```@example nc_news
X_old = copy(Y)
X_new = copy(Y)
X_old[end, 1:3] .= NaN   # simulate 3 missing releases in old vintage

news = nowcast_news(X_new, X_old, dfm, T_obs; target_var=N)
report(news)
```

**Recipe 2: News with grouped releases**

```@example nc_news
X_old = copy(Y)
X_new = copy(Y)
X_old[end, 1:3] .= NaN

# Group: 1=real (INDPRO,UNRATE), 2=nominal (CPI,M2,FFR)
groups = [1, 1, 2, 2, 2]
news = nowcast_news(X_new, X_old, dfm, T_obs; target_var=N, groups=groups)
println("Real sector:    ", round(news.group_impacts[1], digits=4))
println("Nominal sector: ", round(news.group_impacts[2], digits=4))
```

**Recipe 3: Visualize news impacts**

```@example nc_news
X_old = copy(Y)
X_new = copy(Y)
X_old[end, 1:3] .= NaN

news = nowcast_news(X_new, X_old, dfm, T_obs; target_var=N)
nothing # hide
```

```julia
plot_result(news)
```

**Recipe 4: Multi-vintage tracking**

```@example nc_news
base = copy(Y)
base[end, 1:3] .= NaN

# Simulate sequential releases
for col in 1:3
    v_new = copy(base)
    v_new[end, col] = Y[end, col]
    news = nowcast_news(v_new, base, dfm, T_obs; target_var=N)
    println("Release col $col: Δ = ", round(news.new_nowcast - news.old_nowcast, digits=4))
    base[end, col] = Y[end, col]   # update base for next iteration
end
```

---

## The News Concept

Central banks and forecasters update their nowcasts as new data releases arrive throughout the quarter. The **news** framework (Banbura and Modugno 2014) decomposes the total nowcast revision into three components: the surprise content of new releases, the effect of data revisions, and a residual from parameter re-estimation.

```math
\hat{y}^{\text{new}} - \hat{y}^{\text{old}} = \underbrace{\sum_{j \in \mathcal{J}} w_j \cdot (x_j^{\text{actual}} - x_j^{\text{forecast}})}_{\text{news}} + \underbrace{\Delta_{\text{revision}}}_{\text{data revisions}} + \underbrace{\Delta_{\text{re-estimation}}}_{\text{parameter updates}}
```

where:
- ``\hat{y}^{\text{new}}`` and ``\hat{y}^{\text{old}}`` are the nowcasts from the new and old data vintages
- ``\mathcal{J}`` is the set of new releases (positions where ``X_{\text{old}}`` is NaN but ``X_{\text{new}}`` is observed)
- ``w_j`` is the Kalman-gain-derived weight linking release ``j`` to the target
- ``x_j^{\text{actual}} - x_j^{\text{forecast}}`` is the **innovation** --- the difference between the actual release and the model's expectation
- ``\Delta_{\text{revision}}`` captures the effect of revised data
- ``\Delta_{\text{re-estimation}}`` is the residual attributable to parameter updating

The weights ``w_j`` are derived from the DFM state-space structure:

```math
w_j = \frac{C_{\text{target}}' \, P_{t|t-1} \, C_j}{C_j' \, P_{t|t-1} \, C_j + R_{jj}}
```

where ``P_{t|t-1}`` is the state forecast covariance from the Kalman smoother, ``C_{\text{target}}`` and ``C_j`` are factor loadings, and ``R_{jj}`` is the observation noise variance for variable ``j``.

!!! note "Interpretation"
    A positive `impact_news[j]` means the actual value of release ``j`` exceeded the model's expectation, contributing to an **upward** revision of the nowcast. The sum of all news impacts plus the re-estimation residual equals the total revision.

---

## Usage

The `nowcast_news` function requires two data vintages and a pre-estimated DFM model. The **old vintage** (`X_old`) contains more NaN values than the **new vintage** (`X_new`), representing the state of the data before and after new releases.

```@example nc_news
X_old = copy(Y)
X_new = copy(Y)
X_old[end, 1:3] .= NaN

news = nowcast_news(X_new, X_old, dfm, T_obs; target_var=N)
report(news)
```

The `report()` function displays a formatted table with old and new nowcasts, total revision, and top contributing releases ranked by absolute impact. Use `plot_result(news)` for a horizontal bar chart of per-release impacts.

```@raw html
<iframe src="../assets/plots/nowcast_news.html" width="100%" height="400" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `target_var` | `Int` | `size(Y,2)` | Target variable column index |
| `groups` | `Vector{Int}` or `nothing` | `nothing` | Group assignment per variable for impact aggregation |

### `NowcastNews` Return Values

| Field | Type | Description |
|-------|------|-------------|
| `old_nowcast` | `T` | Previous nowcast from old data vintage |
| `new_nowcast` | `T` | Updated nowcast from new data vintage |
| `impact_news` | `Vector{T}` | Per-release news impact on the nowcast |
| `impact_revision` | `T` | Impact from data revisions |
| `impact_reestimation` | `T` | Residual impact from parameter re-estimation |
| `group_impacts` | `Vector{T}` | News impacts aggregated by variable group |
| `variable_names` | `Vector{String}` | Release identifiers (format: `"Var{j}_t{t}"`) |

---

## Complete Example

```@example nc_news
# === Step 1: Estimate DFM ===
dfm = nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1, max_iter=100)
report(dfm)

# === Step 2: Simulate sequential data releases ===
base = copy(Y)
base[end, 1:4] .= NaN
varnames = ["INDPRO", "UNRATE", "CPI", "M2"]

for col in 1:4
    v_new = copy(base)
    v_new[end, col] = Y[end, col]
    news = nowcast_news(v_new, base, dfm, T_obs; target_var=N)
    println("Release: $(varnames[col])  ",
            round(news.old_nowcast, digits=4), " → ", round(news.new_nowcast, digits=4),
            "  (Δ = ", round(news.new_nowcast - news.old_nowcast, digits=4), ")")
    base[end, col] = Y[end, col]
end

# === Step 3: Full decomposition ===
X_old = copy(Y)
X_old[end, 1:4] .= NaN
news = nowcast_news(Y, X_old, dfm, T_obs; target_var=N)
report(news)
```

```julia
plot_result(news)
```

**Interpretation.** The sequential release simulation reveals how each monthly indicator contributes to the quarterly nowcast. Industrial production and unemployment typically carry the largest weights because they load heavily on the common factors driving the quarterly target. The bar chart from `plot_result` visualizes which releases had the largest absolute impact, immediately identifying the key drivers of the nowcast revision.

---

## Common Pitfalls

1. **Vintages must have identical dimensions.** `X_new` and `X_old` must be the same size. The old vintage must have strictly more NaN values at the positions representing new releases. Positions where both are NaN or both are observed contribute nothing to the news.

2. **DFM parameters are held fixed.** The function re-runs the Kalman smoother on both vintages using the same model parameters. It does not re-estimate the factor model. Any discrepancy between the sum of news impacts and the total revision is attributed to `impact_reestimation`.

3. **Only DFM models are supported.** The `nowcast_news` function accepts only `NowcastDFM` because the decomposition relies on Kalman gain weights from the state-space representation. BVAR and bridge models do not support news decomposition.

4. **Variable names are auto-generated.** Release identifiers follow the format `"Var{j}_t{t}"` where `j` is the column index and `t` is the row index. Map these to meaningful names using your data matrix's variable ordering.

---

## References

- Banbura, Marta, and Michele Modugno. 2014. "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data."
  *Journal of Applied Econometrics* 29 (1): 133--160. [DOI: 10.1002/jae.2306](https://doi.org/10.1002/jae.2306)

- Banbura, Marta, Domenico Giannone, and Lucrezia Reichlin. 2011. "Nowcasting."
  *Oxford Handbook of Economic Forecasting*, Chapter 7, Oxford University Press. [DOI: 10.1093/oxfordhb/9780195398649.013.0008](https://doi.org/10.1093/oxfordhb/9780195398649.013.0008)
