# [Panel Tests](@id tests_panel_page)

Panel testing follows the order of an empirical project. **Panel unit root tests** ask whether the series are stationary, a prerequisite for panel VARs and factor models. **Panel cointegration tests** then ask whether non-stationary series share a long-run equilibrium. **Panel causality** and **Panel VAR specification tests** validate the dynamic model that follows. For the full test battery and the tables that route a question to a test, see [Hypothesis Tests](@ref tests_page).

- **Levin-Lin-Chu**: Pooled bias-adjusted t-statistic with a common autoregressive root (Levin, Lin & Chu 2002)
- **Im-Pesaran-Shin**: Averaged per-unit ADF t-statistics standardized with finite-sample moments (Im, Pesaran & Shin 2003)
- **Breitung**: Bias-free pooled statistic via forward orthogonal deviations (Breitung 2000)
- **Fisher**: Maddala-Wu / Choi combination of per-unit ADF or Phillips-Perron p-values (Maddala & Wu 1999; Choi 2001)
- **Hadri**: LM stationarity test --- the panel KPSS, with a *stationarity* null (Hadri 2000)
- **PANIC**: Factor decomposition into common and idiosyncratic components (Bai & Ng 2004, 2010)
- **Pesaran CIPS**: Cross-sectionally augmented IPS test robust to a common factor (Pesaran 2007)
- **Moon-Perron**: Factor-adjusted pooled AR(1) with bias correction (Moon & Perron 2004)
- **Panel cointegration**: Pedroni (1999, 2004), Kao (1999), Westerlund (2007), and Fisher-Johansen
- **Dumitrescu-Hurlin**: Heterogeneous-panel Granger non-causality (Dumitrescu & Hurlin 2012)
- **Panel VAR diagnostics**: Hansen J-test, Andrews-Lu MMSC, and MMSC-based lag selection

The first five unit root tests are the EViews *Panel Unit Root Test* dialog and assume cross-sectional independence; the next three are robust to a common factor. The four cointegration tests are the EViews *Panel Cointegration Test* dialog, and `dh_causality_test` is the EViews *Panel Granger Causality* dialog. Single-series unit root and cointegration tests live on [Unit Root & Cointegration](@ref tests_unitroot_page); panel *estimation* lives on [Panel Regression](@ref panel_reg_page) and [Panel VAR](@ref pvar_page).

```@setup test_panel
using MacroEconometricModels, Random, DataFrames
Random.seed!(42)
```

## Quick Start

Every block seeds its own generator, so the numbers reproduce exactly when the code is copied into a fresh session. Matrix-input tests take a `T × N` panel (time in rows, units in columns); cointegration and causality tests take a [`PanelData`](@ref) built with `xtset`.

**Recipe 1: Build a stationary and a non-stationary panel**

```@example test_panel
Random.seed!(42)
X_i1 = cumsum(randn(60, 20); dims=1)   # 20 independent random walks, T = 60
X_i0 = randn(60, 20)                   # the same panel, but I(0)
size(X_i1)
```

**Recipe 2: First-generation battery on the I(1) panel**

```@example test_panel
report(llc_test(X_i1; deterministic=:constant))
```

**Recipe 3: The same test on the I(0) panel**

```@example test_panel
report(ips_test(X_i0; deterministic=:constant))
```

**Recipe 4: Cross-sectionally augmented test for dependent panels**

```@example test_panel
Random.seed!(2024)
f_common = cumsum(randn(100))                    # one I(1) common factor
loadings = 0.5 .+ randn(30)                      # unit-specific loadings
X_csd = f_common * loadings' + randn(100, 30)    # I(1) common + I(0) idiosyncratic

report(pesaran_cips_test(X_csd; lags=1, deterministic=:constant))
```

**Recipe 5: Panel cointegration on a `PanelData`**

```@example test_panel
Random.seed!(2024)
ids = Int[]; yrs = Int[]; yv = Float64[]; xv = Float64[]
for i in 1:20
    x = cumsum(randn(60))                        # I(1) regressor
    e = zeros(60)
    for t in 2:60
        e[t] = 0.3 * e[t-1] + randn()            # stationary equilibrium error
    end
    append!(ids, fill(i, 60)); append!(yrs, 1:60)
    append!(yv, x .+ e); append!(xv, x)
end
pd_coint = xtset(DataFrame(id = ids, t = yrs, y = yv, x1 = xv), :id, :t)

report(kao_test(pd_coint, :y, :x1))
```

---

## First-Generation Panel Unit Root Tests

The **first-generation** tests --- Levin-Lin-Chu, Im-Pesaran-Shin, Breitung, Fisher, and Hadri --- are the baseline diagnostics run before the dependence-robust tests below. Each accepts a `T × N` matrix or a [`PanelData`](@ref) object, from which the first variable of each group is extracted into a balanced matrix.

!!! warning "All five assume cross-sectional independence"
    These tests are valid only when the innovations are independent across units. Macroeconomic panels almost always share global shocks, and the second-generation section below shows how badly the first generation can be misled by them. The `cs_demean=true` keyword subtracts the cross-sectional mean at each period as a crude mitigation, but it alters the null distribution; for genuine dependence use `pesaran_cips_test` instead.

The first four share the **unit-root null** ``H_0``: every panel is non-stationary, and they reject for **very negative** standardized statistics. Hadri **flips the null** to *stationarity* and rejects for **very positive** values. A conclusion never carries across the two families.

### Levin-Lin-Chu

The LLC test of Levin, Lin & Chu (2002) pools per-unit ADF regressions under a **common** autoregressive root. Starting from

```math
\Delta y_{it} = \phi\, y_{i,t-1} + \mathbf{z}_{it}'\boldsymbol{\gamma}_i + \sum_{j=1}^{p_i} \theta_{ij}\,\Delta y_{i,t-j} + u_{it}
```

where:
- ``y_{it}`` is the observation for unit ``i`` at time ``t``
- ``\mathbf{z}_{it}`` collects the deterministic terms
- ``p_i`` is the unit-specific augmentation order

it forms orthogonalized residuals ``\tilde e_{it}`` and ``\tilde v_{i,t-1}`` (normalized by the per-unit short-run standard deviation), estimates the average ratio of long-run to short-run standard deviations ``\bar S_N`` with a Bartlett kernel of bandwidth ``\text{int}(3.21\,T^{1/3})``, and runs the pooled regression ``\tilde e_{it} = \delta\,\tilde v_{i,t-1} + \varepsilon``. The bias-adjusted statistic is

```math
t^*_\delta = \frac{t_\delta - N\,\tilde T\,\bar S_N\,\operatorname{se}(\hat\delta)\,\mu^*_{\tilde T}}{\sigma^*_{\tilde T}} \sim N(0,1)
```

where ``(\mu^*_{\tilde T},\,\sigma^*_{\tilde T})`` are interpolated in ``\tilde T = T - \bar p - 1`` from LLC (2002, Table 2).

```@example test_panel
report(llc_test(X_i0; deterministic=:constant))
```

On the stationary panel the adjusted statistic is ``-30.14``, far below the 1% critical value of ``-2.326``, and the unit-root null is rejected at any conventional level. Compare that with the ``-0.096`` and p-value of 0.462 returned for `X_i1` in Recipe 2: the pooled regression finds no mean reversion in the random walks. The average lag order reported in the specification block records how much augmentation the per-unit information criterion chose, which matters because ``\tilde T`` and hence the adjustment moments depend on it.

### Im-Pesaran-Shin

The IPS ``W_{\bar t}`` test of Im, Pesaran & Shin (2003) allows a **heterogeneous** root ``\phi_i`` by averaging the per-unit ADF t-statistics ``\bar t = N^{-1}\sum_i t_{iT}`` and standardizing with the finite-sample moments from IPS (2003, Table 3):

```math
W_{\bar t} = \frac{\sqrt{N}\left(\bar t - N^{-1}\sum_i \mathrm{E}[t_{iT}]\right)}{\sqrt{N^{-1}\sum_i \mathrm{Var}[t_{iT}]}} \sim N(0,1)
```

The moments are interpolated linearly in ``T`` for each unit's own ``(\text{deterministic}, \text{lag})`` pair, so the lag column matters: `:auto` selects each unit's lag by information criterion and caps it at 8, the last tabulated value.

```@example test_panel
report(ips_test(X_i1; deterministic=:constant))
```

``W_{\bar t} = 0.284`` with a p-value of 0.612 leaves the unit-root null intact for the random-walk panel, matching LLC. The ``\bar t`` of ``-1.464`` reported above it is the raw average of the 20 individual ADF statistics, and reading it against a Dickey-Fuller table would be a mistake: its null distribution is not the single-series one, which is exactly why the finite-sample standardization is needed. On `X_i0` the same test returns ``-29.11`` (Recipe 3) --- the heterogeneous-root formulation costs nothing in power when the roots are in fact common.

### Breitung

Breitung's (2000) pooled ``\lambda`` statistic uses a variance-ratio construction --- forward orthogonal deviations of the differences and detrending of the levels --- that is **bias-free by design**, so no finite-sample moment table is needed and ``\lambda \sim N(0,1)`` under the null. The public name `breitung_panel_test` avoids a collision with the unrelated Breitung-Eickmeier factor break test on [Structural Breaks](@ref tests_breaks_page).

```@example test_panel
report(breitung_panel_test(X_i1; deterministic=:constant))
```

``\lambda = -0.347`` with a p-value of 0.364 agrees with LLC and IPS. Breitung's `lags` keyword defaults to `0`, an AR(1) assumption; with `lags > 0` both ``\Delta y`` and the level are pre-filtered on ``\Delta y_{t-1}, \ldots, \Delta y_{t-p}`` to whiten residual serial correlation. Because the statistic needs no bias-adjustment table, it degrades more gracefully than LLC when ``T`` is small.

### Fisher-type

The Fisher-type test of Maddala & Wu (1999) and Choi (2001) combines the per-unit ADF (or Phillips-Perron) p-values ``p_i``, reusing the same MacKinnon response-surface path as the single-series `adf_test`:

```math
P = -2\sum_{i=1}^{N}\ln p_i \sim \chi^2(2N), \qquad Z = \frac{1}{\sqrt{N}}\sum_{i=1}^{N}\Phi^{-1}(p_i) \sim N(0,1)
```

All four combinations --- Maddala-Wu ``P`` (upper-tailed), Choi inverse-normal ``Z`` (lower-tailed), Choi logit ``L^*`` (lower-tailed), and Choi modified ``P_m`` (upper-tailed) --- are always computed; `combine` selects which one becomes the primary `statistic` and `pvalue`. With ``N = 1`` the test reduces exactly to the single-series `adf_test` p-value.

```@example test_panel
report(fisher_panel_test(X_i1; base=:adf, combine=:mw))
```

``P = 39.32`` against a ``\chi^2(40)`` reference gives a p-value of 0.501, and all four combinations agree. Reading them together is the point: ``P`` and ``P_m`` are upper-tailed while ``Z`` and ``L^*`` are lower-tailed, so a panel with strong evidence against the unit root drives ``P`` and ``P_m`` up while pushing ``Z`` and ``L^*`` down. On `X_i0` the same call returns ``P = 871.08``, and every combination rejects.

### Hadri

The Hadri (2000) LM test is the panel analogue of KPSS, with a **stationarity null**. For each unit it forms partial sums ``S_{it} = \sum_{j\le t}\hat\varepsilon_{ij}`` of the OLS residuals from a regression on the deterministics and ``LM_i = T^{-2}\sum_t S_{it}^2 / \hat\sigma_i^2``, then standardizes:

```math
Z = \frac{\sqrt{N}\,(\overline{LM} - \xi)}{\zeta} \sim N(0,1), \qquad (\xi, \zeta^2) = \begin{cases}(1/6,\ 1/45) & \text{constant}\\(1/15,\ 11/6300) & \text{trend}\end{cases}
```

Because the null is stationarity, the test is **right-tailed**: very positive ``Z`` rejects. `hetero=true` (the default) gives each unit its own ``\hat\sigma_i^2``.

```@example test_panel
report(hadri_test(X_i1; deterministic=:constant))
```

``Z = 75.27`` with a p-value below 0.001 rejects stationarity for the random-walk panel --- the mirror image of the non-rejections that LLC, IPS, Breitung, and Fisher returned on the same data. On `X_i0` Hadri gives ``Z = -0.326`` with a p-value of 0.628 and stationarity survives, again mirroring the other four. That agreement in *opposite directions* is the sanity check to run: when Hadri and the unit-root family point the same way, the panel is telling an inconsistent story and the deterministic specification is usually the culprit.

### Options and return values

| Keyword | Applies to | Default | Description |
|----------|-----------|---------|-------------|
| `deterministic` | all | `:constant` | `:none`/`:constant`/`:trend`; IPS and Hadri accept only `:constant`/`:trend` |
| `lags` | LLC, IPS, Fisher | `:auto` | Common integer lag, or per-unit IC selection |
| `lags` | Breitung | `0` | Prewhitening order ``p`` |
| `max_lags` | LLC, IPS | `nothing` | Cap for `:auto`, default ``\lfloor 12(T/100)^{1/4} \rfloor`` |
| `criterion` | LLC, IPS | `:aic` | `:aic`, `:bic`, or `:hqic` for `:auto` selection |
| `base` | Fisher | `:adf` | Per-unit test: `:adf` or `:pp` |
| `combine` | Fisher | `:mw` | Primary combination: `:mw`, `:choi`, `:logit`, `:pm` |
| `hetero` | Hadri | `true` | Per-unit vs. pooled ``\hat\sigma^2`` |
| `cs_demean` | all | `false` | Subtract the cross-sectional mean each period |

`LLCResult`, `IPSResult`, `BreitungPanelResult`, `FisherPanelResult`, and `HadriResult` all expose `statistic`, `pvalue`, `deterministic`, `nobs` (``T``), and `n_units` (``N``), plus the `StatsAPI` `pvalue`/`nobs` interface and [`refs`](@ref) for citations. `FisherPanelResult` additionally stores every combination (`P`, `Z`, `Lstar`, `Pm` with matching `_pvalue` fields) and the vector of `individual_pvalues`; `IPSResult` stores the per-unit t-statistics; `HadriResult` stores `LM_bar`, `xi`, and `zeta`. LLC needs ``T \geq 10`` and ``N \geq 2``; IPS and Fisher need ``T \geq 20``.

---

## Second-Generation Panel Unit Root Tests

When units share common shocks, the first-generation tests break down. The panel built in Recipe 4 makes the failure concrete: an I(1) common factor drives every unit, but each unit's *idiosyncratic* component is stationary. The first-generation tests see only the composite series and report a unit root everywhere:

```@example test_panel
naive = ips_test(X_csd; deterministic=:constant)
(statistic = round(naive.statistic, digits=3), pvalue = round(naive.pvalue, digits=4))
```

``W_{\bar t} = 4.911`` with a p-value of 1.0 --- a decisive non-rejection of the unit-root null in a panel whose idiosyncratic dynamics are stationary by construction. Three tests correct for this in different ways: PANIC estimates the factor and tests the pieces separately, Pesaran CIPS augments each ADF regression with cross-sectional averages, and Moon-Perron projects the factor space out before pooling.

### PANIC

**PANIC** (Panel Analysis of Nonstationarity in Idiosyncratic and Common components), from Bai & Ng (2004, 2010), decomposes the panel into common factors and idiosyncratic residuals via principal components and tests each part for unit roots separately. The panel follows a factor structure:

```math
X_{it} = \lambda_i' F_t + e_{it}
```

where:
- ``X_{it}`` is the observation for unit ``i`` at time ``t``
- ``F_t`` is the ``r \times 1`` vector of common factors
- ``\lambda_i`` is the ``r \times 1`` vector of unit-specific loadings
- ``e_{it}`` is the idiosyncratic error

Three steps follow. First, ``\hat{F}_t`` and ``\hat{\lambda}_i`` are estimated by PCA on the standardized ``T \times N`` panel. Second, an ADF regression with a constant is run on each estimated factor ``\hat{F}_{j,t}``. Third, ADF regressions without deterministic terms are run on the defactored residuals ``\hat{e}_{it}`` --- the factor extraction has already absorbed any deterministic component --- and the ``N`` p-values are pooled:

```math
P_a = \frac{\sum_{i=1}^N p_i - N/2}{\sqrt{N/12}} \xrightarrow{d} N(0,1)
```

where:
- ``p_i`` is the ADF p-value for unit ``i``'s idiosyncratic residual
- ``N`` is the number of cross-sectional units
- the standardization uses the mean ``1/2`` and variance ``1/12`` of the uniform distribution

Under ``H_0`` --- all idiosyncratic components have unit roots --- the individual p-values are approximately uniform and ``P_a`` is standard normal. Large negative values indicate rejection.

!!! warning "Read the decomposition, not the pooled p-value"
    In simulation on panels satisfying ``H_0`` exactly (I(1) factor, I(1) idiosyncratic components) the pooled ``P_a`` rejects at the nominal 5% level in every replication. The uniformity of ``p_i`` that ``P_a`` assumes does not survive PCA on the levels of the panel, so the pooled statistic is not size-correct. The reliable outputs are `factor_adf_stats` and `factor_adf_pvalues`, which say whether the *common* component is I(1), and the per-unit `individual_stats`. Take the panel-level decision from `pesaran_cips_test`.

```@example test_panel
report(panic_test(X_csd; r=1))
```

The factor ADF statistic of ``-0.583`` with a p-value of 0.873 correctly identifies the common component as I(1), which is the question the first-generation tests could not separate out. That is the decomposition PANIC exists to deliver: the non-stationarity in `X_csd` lives entirely in the common factor, and any cointegration or VAR analysis should be conducted on the defactored data. When `r=:auto`, the number of factors comes from the Bai-Ng (2002) IC2 criterion via `ic_criteria`, with ``r_{\max} = \min(10, \min(T,N)-1)``.

| Keyword | Type | Default | Description |
|----------|------|---------|-------------|
| `r` | `Union{Int,Symbol}` | `:auto` | Number of common factors, or IC2 selection |
| `method` | `Symbol` | `:pooled` | `:pooled` or `:individual` (recorded on the result) |

| Field | Type | Description |
|-------|------|-------------|
| `factor_adf_stats` | `Vector{T}` | ADF statistic for each estimated common factor |
| `factor_adf_pvalues` | `Vector{T}` | ADF p-value for each common factor |
| `pooled_statistic` | `T` | Standardized pooled statistic ``P_a`` |
| `pooled_pvalue` | `T` | Normal p-value for ``P_a`` |
| `individual_stats` | `Vector{T}` | ADF statistic for each unit's idiosyncratic residual |
| `individual_pvalues` | `Vector{T}` | ADF p-value for each unit |
| `n_factors` | `Int` | Number of common factors used |
| `method` | `Symbol` | Pooling method recorded |
| `nobs` | `Int` | Time dimension ``T`` |
| `n_units` | `Int` | Cross-section dimension ``N`` |

### Pesaran CIPS

The **Cross-sectionally Augmented IPS** test of Pesaran (2007) augments each unit's ADF regression with cross-sectional averages, which proxy for a single unobserved common factor. No explicit factor estimation is required. The **CADF** regression for unit ``i`` is

```math
\Delta y_{it} = a_i + b_i y_{i,t-1} + c_i \bar{y}_{t-1} + d_i \Delta\bar{y}_t
  + \sum_{j=1}^{p} \left( \phi_{ij} \Delta y_{i,t-j} + \psi_{ij} \Delta\bar{y}_{t-j} \right) + \varepsilon_{it}
```

where:
- ``\Delta y_{it} = y_{it} - y_{i,t-1}`` is the first difference
- ``b_i`` is the coefficient of interest; ``H_0: b_i = 0`` for all ``i``
- ``\bar{y}_t = N^{-1}\sum_{i=1}^{N} y_{it}`` is the cross-sectional average
- ``a_i`` is a unit-specific intercept, present when `deterministic=:constant`
- ``p`` is the number of augmenting lags, applied to both ``\Delta y_i`` and ``\Delta\bar{y}``

The CIPS statistic averages the individual CADF t-statistics after truncation:

```math
\text{CIPS} = N^{-1} \sum_{i=1}^{N} \tilde{t}_i, \qquad \tilde{t}_i = \max(-6.19, \min(6.19, t_i))
```

The truncation at ``\pm 6.19`` (Pesaran 2007) keeps the average's moments finite when an individual regression produces an extreme t-value. The test is left-tailed.

```@example test_panel
report(pesaran_cips_test(X_csd; lags=1, deterministic=:constant))
```

CIPS of ``-6.138`` sits well below the 1% critical value of ``-2.25``, so the null of a panel unit root is rejected --- the correct answer for a panel whose idiosyncratic components are stationary, and the exact opposite of what the naive IPS test concluded on the same data. Critical values come from the Pesaran (2007) tables, looked up at the nearest tabulated ``N \in \{10, 20, 30, 50, 100\}`` and ``T \in \{20, 30, 50, 70, 100\}``; with ``N = 30`` and ``T = 100`` no rounding is needed here. The `individual_cadf_stats` field stores the un-truncated t-statistics, which is where to look when one unit drives the average.

| Keyword | Type | Default | Description |
|----------|------|---------|-------------|
| `lags` | `Union{Int,Symbol}` | `:auto` | Augmenting lags, or ``\max(1, \lfloor T^{1/3} \rfloor)`` |
| `deterministic` | `Symbol` | `:constant` | `:none`, `:constant`, or `:trend` |

| Field | Type | Description |
|-------|------|-------------|
| `cips_statistic` | `T` | Average of the truncated CADF t-statistics |
| `pvalue` | `T` | Interpolated from the Pesaran (2007) tables |
| `individual_cadf_stats` | `Vector{T}` | Individual CADF t-statistics before truncation |
| `critical_values` | `Dict{Int,T}` | Critical values keyed by `1`, `5`, `10` |
| `lags` | `Int` | Augmenting lags used |
| `deterministic` | `Symbol` | Deterministic specification |
| `nobs` | `Int` | Time dimension ``T`` |
| `n_units` | `Int` | Cross-section dimension ``N`` |

### Moon-Perron

The Moon & Perron (2004) test projects the estimated factor space out of the cross-section and pools unit-level AR(1) regressions on the de-factored data. The projection ``Q_\perp = I_N - \hat{\Lambda}(\hat{\Lambda}'\hat{\Lambda})^{-1}\hat{\Lambda}'`` is applied to each row, and two modified statistics follow:

```math
t_a^* = \frac{\sqrt{N}\, T\, (\hat{\rho}_{\text{pool}} - 1) - B_a}{S_a}, \quad
t_b^* = \frac{\sqrt{N}\, \bar{t} - B_b}{S_b}
```

where:
- ``\hat{\rho}_{\text{pool}}`` is the pooled AR(1) coefficient from the de-factored data
- ``\bar{t}`` is the average of the individual t-statistics
- ``B_a, B_b`` are bias corrections built from the gap between the long-run and short-run variances
- ``S_a, S_b`` are variance corrections built from ``\hat{\omega}_i^4``

Both are ``N(0,1)`` under ``H_0`` and left-tailed. Each unit's long-run variance ``\hat{\omega}_i^2`` uses a Bartlett kernel with a Newey-West bandwidth, and the bias corrections exist to absorb the serial correlation that de-factoring introduces.

!!! warning "The two statistics are not interchangeable"
    In simulation under ``H_0`` the ``t_b^*`` statistic rejects far more often than its nominal size, while ``t_a^*`` almost never rejects even under a strongly stationary alternative. A split verdict is therefore the norm rather than a signal of ambiguous evidence. Use Moon-Perron alongside `pesaran_cips_test` rather than on its own.

```@example test_panel
report(moon_perron_test(X_csd; r=1))
```

``t_a^* = 4.330`` fails to reject while ``t_b^* = -21.084`` rejects below 0.001, which is the split the warning above describes. On this panel the ``t_b^*`` verdict happens to match the truth and the CIPS result, but the disagreement carries no information about the strength of the evidence. `r` behaves exactly as in PANIC: an integer fixes the number of factors, `:auto` selects it with the Bai-Ng IC2 criterion, and the result records `n_factors`, `t_a_statistic`, `t_b_statistic`, `pvalue_a`, `pvalue_b`, `nobs`, and `n_units`.

---

## Panel Unit Root Summary

`panel_unit_root_summary` runs all eight unit root tests --- the five first-generation and the three second-generation --- and returns a `PanelUnitRootSummary`. Displaying the object, with `report(s)` or by leaving it as the last expression, prints the whole battery; any sub-test that errors is recorded in `s.errors` and skipped rather than aborting the run. The individual results stay accessible as fields, which makes a compact side-by-side comparison easy to assemble:

```@example test_panel
s = panel_unit_root_summary(X_csd; r=1)

[(test = "LLC",      statistic = round(s.llc.statistic, digits=3),           pvalue = round(s.llc.pvalue, digits=4)),
 (test = "IPS",      statistic = round(s.ips.statistic, digits=3),           pvalue = round(s.ips.pvalue, digits=4)),
 (test = "Breitung", statistic = round(s.breitung.statistic, digits=3),      pvalue = round(s.breitung.pvalue, digits=4)),
 (test = "Fisher",   statistic = round(s.fisher.statistic, digits=3),        pvalue = round(s.fisher.pvalue, digits=4)),
 (test = "Hadri",    statistic = round(s.hadri.statistic, digits=3),         pvalue = round(s.hadri.pvalue, digits=4)),
 (test = "PANIC",    statistic = round(s.panic.pooled_statistic, digits=3),  pvalue = round(s.panic.pooled_pvalue, digits=4)),
 (test = "CIPS",     statistic = round(s.cips.cips_statistic, digits=3),     pvalue = round(s.cips.pvalue, digits=4))]
```

The table is the whole argument for the second generation in one screen. LLC (``-0.255``), IPS (``4.911``), Breitung (``2.517``), and Fisher (``23.717``) all fail to reject the unit-root null, and Hadri --- with its flipped null --- rejects stationarity at ``312.6``, so the five first-generation tests agree unanimously that `X_csd` is non-stationary. CIPS returns ``-4.104`` with a p-value of 0.001 and reverses that verdict, because the idiosyncratic components really are stationary and only the shared factor is not. The unanimity of the first generation is not evidence of robustness here; it is five tests making the same mistake for the same reason.

The `lags` keyword is passed to CIPS, LLC, IPS, and Fisher; `r` is passed to PANIC and Moon-Perron. Both default to `:auto`, so the summary uses the ``\lfloor T^{1/3} \rfloor`` lag rule and IC2 factor selection unless told otherwise.

---

## Panel Cointegration Tests

Once a panel is established as I(1), the next question is whether the non-stationary series share a long-run equilibrium. All four tests share the null **``H_0``: no cointegration**, except Fisher-Johansen, which tests a sequence of rank hypotheses.

- **Pedroni**: residual-based, seven statistics allowing a *heterogeneous* cointegrating vector (Pedroni 1999, 2004)
- **Kao**: residual-based, five Dickey-Fuller-type statistics under a *homogeneous* vector (Kao 1999)
- **Westerlund**: four error-correction statistics testing the speed-of-adjustment coefficient, with an optional dependence-robust bootstrap (Westerlund 2007; Persyn & Westerlund 2008)
- **Fisher-Johansen**: Maddala-Wu / Choi combination of per-unit Johansen trace and max-eigenvalue p-values (Johansen 1991; Maddala & Wu 1999; Choi 2001)

Pedroni, Kao, and Westerlund take a `PanelData`, a response symbol, and one or more regressor symbols; Fisher-Johansen takes a list of series with no response/regressor split. **All four require a balanced panel** and raise an `ArgumentError` otherwise. The examples reuse `pd_coint` from Recipe 5, in which ``y_{it} = x_{it} + e_{it}`` with ``x_{it}`` a random walk and ``e_{it}`` an AR(1) --- cointegrated with a unit cointegrating vector in every unit.

### Pedroni

`pedroni_test` reports seven statistics: four **within-dimension** (panel) statistics that pool numerators and denominators across units before taking the ratio, and three **between-dimension** (group) statistics that average per-unit ratios. All seven are standardized to ``N(0,1)`` with the Pedroni (1999, Table 2) adjustment moments, which are tabulated for one to six regressors.

!!! warning "The panel-v statistic runs the other way"
    **panel-v is right-tailed**: a large *positive* value rejects no-cointegration. The other six Pedroni statistics, and every Kao and Westerlund statistic, are left-tailed. The `report` output labels each row's tail, and reconstructing a p-value by hand means `ccdf(Normal(), z)` for panel-v against `cdf(Normal(), z)` for the rest.

```@example test_panel
report(pedroni_test(pd_coint, :y, :x1; trend = :constant))
```

Six of the seven statistics reject decisively --- panel-rho at ``-19.39``, panel-t at ``-16.71``, group-t at ``-18.31``, and panel-ADF at ``-83.98`` --- which is the correct verdict for a panel that is cointegrated by construction. The exception is panel-v at ``-0.60`` with a p-value of 0.726, and it is the exception precisely because it is right-tailed: a variance-ratio statistic has low power against this alternative in short panels, and Pedroni (2004) recommends the group-ADF and panel-ADF statistics for ``T`` below about 100. Reading panel-v as a failure to cointegrate would be the sign error the warning above guards against.

The `trend` keyword selects the deterministic case of the cointegrating regression (`:none`, `:constant`, or `:trend`); `lags` sets the Newey-West bandwidth for the residual long-run variances, defaulting to ``\text{round}(4(T/100)^{2/9})``; and `adf_lags` sets the augmentation order of the parametric statistics, defaulting to 2. The result stores `names`, `raw`, `statistics`, `pvalues`, the `mu` and `v` adjustment moments, `bandwidth`, and `adf_lags`.

### Kao

`kao_test` assumes a *homogeneous* cointegrating vector. It within-demeans each unit, fits a single pooled slope, and applies five Dickey-Fuller-type statistics to the pooled residuals. All five are ``N(0,1)`` and left-tailed.

```@example test_panel
report(kao_test(pd_coint, :y, :x1))
```

The pooled residual autoregressive coefficient ``\hat{\rho} = 0.342`` is far from unity, and all five statistics reject: ``\text{DF}_\rho = -51.05``, ``\text{DF}_t = -20.49``, and the ADF form at ``-8.70``. `DFrho` and `DFt` assume strict exogeneity of the regressors, while `DFrho_star`, `DFt_star`, and `ADF` add an endogeneity correction built from the long-run conditional variance ``\hat{\omega}_\nu^2``; the near-identity of the corrected and uncorrected values here says the regressor is in fact close to strictly exogenous, as the design intends. Kao is the right choice when theory implies one common cointegrating vector, and it is more powerful than Pedroni in that case --- but badly misleading when the vector varies by unit.

`lags` sets the pooled ADF lag order and `kernel_lags` the Bartlett bandwidth for the long-run variances; both default to `:auto` with the same ``\text{round}(4(T/100)^{2/9})`` rule.

### Westerlund

`westerlund_test` estimates a per-unit error-correction model for ``\Delta y`` and tests the error-correction coefficient. The two **group-mean** statistics `Gt` and `Ga` average per-unit quantities and have the alternative "at least one unit is cointegrated"; the two **panel** statistics `Pt` and `Pa` pool a common coefficient and have the alternative "the panel as a whole is cointegrated". All four are left-tailed.

```@example test_panel
report(westerlund_test(pd_coint, :y, :x1; trend = :constant, lags = 1, leads = 0))
```

All four statistics reject: ``G_t = -13.47``, ``G_a = -26.42``, ``P_t = -12.88``, and ``P_a = -32.16``. Because the group-mean and panel statistics test different alternatives, agreement between them means the error-correction mechanism is present broadly rather than in a few units --- the informative case is `Gt` rejecting while `Pt` does not, which points to cointegration concentrated in a subset of the panel. `lags` and `leads` set the short-run dynamics of the ECM (defaults 1 and 0) and `lrwindow` the Bartlett window (default 2).

Setting `bootstrap` to a positive integer adds cross-sectional-dependence-robust p-values in `bootstrap_pvalues`, left as `NaN` otherwise; `seed` (default `20240716`) makes them reproducible. Use the bootstrap whenever the panel plausibly shares common shocks, since the asymptotic p-values assume independence.

### Fisher-Johansen

`fisher_johansen_test` runs a per-unit `johansen_test` and combines the trace and max-eigenvalue p-values across units, reporting a combined statistic for each rank hypothesis ``r = 0, 1, \ldots, n-1``. With `combine = :mw` (the default) the combination is Maddala-Wu ``P = -2\sum_i \ln p_i \sim \chi^2(2N)``, upper-tailed; `combine = :choi` uses the inverse-normal ``Z \sim N(0,1)``. With a single unit the combination reduces exactly to that unit's Johansen p-values.

```@example test_panel
Random.seed!(303)
ids_fj = Int[]; yrs_fj = Int[]; av = Float64[]; bv = Float64[]
for i in 1:12
    a = cumsum(randn(60))
    b = a .+ 0.5 .* randn(60)                    # b cointegrated with a, rank 1
    append!(ids_fj, fill(i, 60)); append!(yrs_fj, 1:60)
    append!(av, a); append!(bv, b)
end
pd_fj = xtset(DataFrame(id = ids_fj, t = yrs_fj, a = av, b = bv), :id, :t)

report(fisher_johansen_test(pd_fj, :a, :b; lags = 2))
```

The combined trace statistic for ``r \leq 0`` is 209.99 with a p-value below 0.001, so no cointegration is rejected; at ``r \leq 1`` the statistic falls to 35.53 with a p-value of 0.061, which does not reject at 5%. The estimated `rank` field is therefore 1 --- the first rank hypothesis the combined trace test fails to reject --- which matches the design. The max-eigenvalue column tells the same story, as it should for a two-series system where the two sequences differ only in how they accumulate the eigenvalues. `lags` defaults to 2 and `deterministic` to `:constant`, both passed through to each per-unit Johansen test.

---

## Dumitrescu-Hurlin Panel Causality

The **Dumitrescu-Hurlin (2012)** test asks whether one variable Granger-causes another across a **heterogeneous** panel, allowing the regression coefficients to differ by unit. It is the standard tool for cross-country lead-lag questions --- credit to GDP, oil to inflation --- in which the dynamics plainly vary by country.

For each unit ``i``, ``y_{it}`` is regressed on an intercept, ``p`` lags of ``y_i``, and ``p`` lags of ``x_i``. The individual Wald statistic ``W_i`` tests that all ``p`` coefficients on lagged ``x_i`` are zero. The ``W_i`` are averaged and standardized two ways:

```math
\bar{W} = \frac{1}{N}\sum_{i=1}^{N} W_i, \qquad
\bar{Z} = \sqrt{\frac{N}{2p}}\,(\bar{W} - p) \;\xrightarrow{d}\; N(0,1),
```

```math
\tilde{Z} = \sqrt{N}\,\frac{\bar{W} - \mathbb{E}[W_i]}{\sqrt{\operatorname{Var}[W_i]}} \;\xrightarrow{d}\; N(0,1),
```

where ``\bar{Z}`` is the asymptotic statistic and ``\tilde{Z}`` uses the exact finite-``T`` moments of Dumitrescu-Hurlin (2012, eqs. 26-27):

```math
\mathbb{E}[W_i] = \frac{p\,(T - 2p - 1)}{T - 2p - 3}, \qquad
\operatorname{Var}[W_i] = \frac{2p\,(T - 2p - 1)^2\,(T - p - 3)}{(T - 2p - 3)^2\,(T - 2p - 5)}.
```

- ``H_0``: ``x`` does not Granger-cause ``y`` for **any** unit (homogeneous non-causality)
- ``H_1``: ``x`` Granger-causes ``y`` for **some** units

Both statistics are **right-tailed**: a large ``\bar{W}`` rejects non-causality. `dh_causality_test` reports the ``\chi^2(p)`` Wald form ``W_i = p\,F_i``, whereas R's `plm::pgrangertest` reports the F-based ``F_i = W_i/p``; divide by ``p`` to compare.

```@example test_panel
Random.seed!(88)
ids_dh = Int[]; tt = Int[]; y_dh = Float64[]; x_dh = Float64[]
for i in 1:20
    x = randn(40); y = zeros(40)
    for t in 3:40
        y[t] = 0.4y[t-1] - 0.1y[t-2] + 0.3x[t-1] + 0.2x[t-2] + randn()
    end
    append!(ids_dh, fill(i, 40)); append!(tt, 1:40)
    append!(y_dh, y); append!(x_dh, x)
end
pd_dh = xtset(DataFrame(id = ids_dh, time = tt, y = y_dh, x = x_dh), :id, :time)

report(dh_causality_test(pd_dh, :x, :y; p = 2))
```

The average Wald statistic is ``\bar{W} = 6.88`` against a null expectation of ``\mathbb{E}[W_i] = 2\times33/31 = 2.13`` at ``p = 2`` and an effective sample of 38, so both standardizations reject at any level: ``\bar{Z} = 10.92`` and ``\tilde{Z} = 9.36``. All 20 units were retained and none skipped, which the specification block reports. ``\tilde{Z}`` is the statistic to quote for finite panels; it needs an effective sample ``T_i > 2p + 5`` per unit, because ``\operatorname{Var}[W_i]`` is undefined below that, and units failing the guard are dropped and counted in `n_skipped`. The call errors only when no unit qualifies.

Cross-sectional dependence invalidates the asymptotic normal p-values. Passing `bootstrap` resamples time blocks of the restricted-model residuals jointly across units --- preserving the dependence under the non-causality null --- and returns a robust p-value on ``\bar{Z}``:

```@example test_panel
dh_boot = dh_causality_test(pd_dh, :x, :y; p = 2, bootstrap = 199, seed = 20240816)
(Zbar = round(dh_boot.Zbar, digits=3),
 asymptotic_pvalue = round(dh_boot.Zbar_pvalue, digits=4),
 bootstrap_pvalue = round(dh_boot.bootstrap_pvalue, digits=4))
```

The bootstrap p-value of 0.0 --- none of the 199 replications produced a ``\bar{Z}^*`` at or above the observed 10.915 --- confirms the asymptotic verdict. Blocks of length ``\lceil T_i^{1/3} \rceil`` are drawn circularly and shared across units when the panel is balanced, which is what preserves contemporaneous dependence. When the asymptotic and bootstrap p-values diverge, trust the bootstrap.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `p` | `Int` | `1` | Lag order (``p`` lags of both ``y`` and ``x`` per unit) |
| `bootstrap` | `Int` | `0` | Block-bootstrap replications for a dependence-robust ``\bar{Z}`` p-value |
| `seed` | `Int` | `1234` | RNG seed for the bootstrap, stored on the result |

`DumitrescuHurlinResult` stores `Wbar`, `Zbar`, `Zbar_pvalue`, `Ztilde`, `Ztilde_pvalue`, the per-unit `W_i`, `p`, `N` (units retained), `nobs` (mean effective sample), `n_skipped`, `bootstrap`, `seed`, `bootstrap_pvalue`, and the `cause`/`effect` names.

---

## Panel VAR Specification Tests

After a Panel VAR is estimated by GMM, three diagnostics validate the specification: the Hansen J-test for instrument validity, the Andrews-Lu MMSC criteria for model selection, and MMSC-based lag selection. All three require a GMM-estimated model (`estimate_pvar` with `:onestep` or `:twostep`) and raise an `ArgumentError` on an FE-OLS model.

```@example test_panel
Random.seed!(77)
Z = zeros(100 * 20, 3)
for i in 1:100
    mu = 0.5 * randn(3)
    for t in 2:20
        row = (i - 1) * 20 + t
        Z[row, :] = mu + 0.5 * Z[row - 1, :] + 0.2 * randn(3)   # true PVAR(1)
    end
end
df_pv = DataFrame(Z, ["y1", "y2", "y3"])
df_pv.country = repeat(1:100, inner = 20)
df_pv.year = repeat(1:20, outer = 100)
pd_pvar = xtset(df_pv, :country, :year)

pvar = estimate_pvar(pd_pvar, 2; steps = :twostep, collapse = true, max_lag_endo = 5)
nothing # hide
```

### Hansen J-test

The Hansen (1982) J-test asks whether the overidentifying restrictions hold. In a Panel VAR the instrument set --- lagged levels for first-difference GMM, lagged levels and differences for system GMM --- normally exceeds the number of parameters, and the surplus moment conditions are testable:

```math
J = n \, \bar{g}' \, \hat{W} \, \bar{g} \sim \chi^2(q - k)
```

where:
- ``\bar{g} = n^{-1} \sum_{i=1}^{n} Z_i' e_i`` is the average moment condition over the ``n`` usable units
- ``\hat{W}`` is the inverse of ``n^{-1}\sum_i (Z_i' e_i)(Z_i' e_i)'``
- ``q`` is the number of instruments and ``k`` the number of parameters per equation
- ``q - k`` are the degrees of freedom

The statistic is computed equation by equation and the reported value is the average across the ``m`` equations. Under ``H_0`` --- all moment conditions valid --- it is ``\chi^2(q-k)``; rejection points to invalid instruments or a misspecified model.

```@example test_panel
report(pvar_hansen_j(pvar))
```

``J = 4.53`` on 6 degrees of freedom gives a p-value of 0.605, so the overidentifying restrictions survive: 12 instruments against 6 parameters per equation, and the extra six moment conditions are consistent with the data. Collapsing the instrument matrix and capping the lag depth at 5 is what keeps the count at 12. Without those two keywords the same panel generates 510 instruments for 100 units, `estimate_pvar` warns about instrument proliferation, and the J-statistic degenerates to exactly the number of groups with a p-value pinned near 1 --- a non-rejection that carries no information.

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | `String` | `"Hansen J-test"` |
| `statistic` | `T` | J-statistic, averaged across equations |
| `pvalue` | `T` | P-value from ``\chi^2(q-k)`` |
| `df` | `Int` | Overidentifying restrictions ``q - k`` |
| `n_instruments` | `Int` | Number of instruments ``q`` |
| `n_params` | `Int` | Parameters per equation ``k`` |

### Andrews-Lu MMSC

The Andrews & Lu (2001) **Model and Moment Selection Criteria** extend information criteria to GMM by penalizing the J-statistic with the number of overidentifying restrictions, which makes models with different lag orders *and* different instrument sets comparable:

```math
\text{MMSC-BIC}  = J - (q - k) \ln n, \qquad
\text{MMSC-AIC}  = J - 2(q - k), \qquad
\text{MMSC-HQIC} = J - Q(q - k) \ln \ln n
```

where:
- ``J`` is the Hansen J-statistic
- ``q - k`` is the number of overidentifying restrictions
- ``n`` is the total number of observations
- ``Q`` is the Hannan-Quinn constant, set by the `hq_criterion` keyword and defaulting to 2.1

Lower values are preferred, and the three differ only in how heavily they charge for an extra moment condition.

```@example test_panel
mmsc = pvar_mmsc(pvar)
(bic = round(mmsc.bic, digits=2), aic = round(mmsc.aic, digits=2), hqic = round(mmsc.hqic, digits=2))
```

For the PVAR(2) the three criteria are ``-40.10``, ``-7.47``, and ``-20.75``. The ordering ``\text{BIC} < \text{HQIC} < \text{AIC}`` is mechanical --- ``\ln n > Q \ln\ln n > 2`` for any sample this size --- so the levels only mean something in comparison with another specification, which is what lag selection automates.

### Lag selection

`pvar_lag_selection` estimates Panel VARs for ``p = 1, \ldots, p_{\max}`` and compares them on all three MMSC criteria. Keyword arguments are forwarded to every `estimate_pvar` call, so the instrument settings must be repeated here.

```@example test_panel
sel = pvar_lag_selection(pd_pvar, 4; collapse = true, max_lag_endo = 5)
(best_bic = sel.best_bic, best_aic = sel.best_aic, best_hqic = sel.best_hqic)
```

All three criteria select ``p = 1``, which recovers the true lag order of the generating process; unanimity across the three is the signal that the choice is not an artefact of the penalty. The `table` field holds a ``p_{\max} \times 4`` matrix whose first column is the lag order and whose remaining three columns are the formatted MMSC values, with `"—"` marking a lag order whose estimation failed. The `models` vector holds the fitted `PVARModel` objects, but entries for failed lag orders are left undefined, so index it only for lag orders whose criteria are finite.

---

## Complete Example

A full pre-estimation workflow on a cross-sectionally dependent panel: establish the order of integration with the right family of tests, then estimate and validate a Panel VAR.

```@example test_panel
# Step 1 --- an I(1) common factor with stationary idiosyncratic components
Random.seed!(11)
f_ce = cumsum(randn(80))
lam_ce = 0.5 .+ randn(25)
X_ce = f_ce * lam_ce' + randn(80, 25)

# Step 2 --- what a first-generation test concludes
first_gen = llc_test(X_ce; deterministic=:constant)
(statistic = round(first_gen.statistic, digits=3), pvalue = round(first_gen.pvalue, digits=4))
```

```@example test_panel
# Step 3 --- PANIC separates the common component from the idiosyncratic one
panic_ce = panic_test(X_ce; r=1)
(factor_adf = round(panic_ce.factor_adf_stats[1], digits=3),
 factor_pvalue = round(panic_ce.factor_adf_pvalues[1], digits=4))
```

```@example test_panel
# Step 4 --- the cross-sectionally augmented test decides
report(pesaran_cips_test(X_ce; lags=1, deterministic=:constant))
```

```@example test_panel
# Step 5 --- Panel VAR diagnostics on the GMM specification
j_ce = pvar_hansen_j(pvar)
mmsc_ce = pvar_mmsc(pvar)
(hansen = (statistic = round(j_ce.statistic, digits=3), pvalue = round(j_ce.pvalue, digits=4), df = j_ce.df),
 mmsc = (bic = round(mmsc_ce.bic, digits=2), aic = round(mmsc_ce.aic, digits=2), hqic = round(mmsc_ce.hqic, digits=2)))
```

The sequence is the point. LLC returns 0.132 with a p-value of 0.553 and cannot reject the unit root, because the shared I(1) factor dominates every series and LLC assumes that shared factor away. PANIC locates the non-stationarity: the ADF statistic on the estimated common component is ``-0.759`` with a p-value of 0.827, so the common component is I(1) while the idiosyncratic parts, by construction, are not. Pesaran CIPS, which handles the dependence directly, returns ``-6.012`` against a 1% critical value of ``-2.29`` and rejects the panel unit root. The practical conclusion is to defactor before estimating --- and the Panel VAR diagnostics then confirm the GMM specification, with a Hansen J-test at 4.529 on 6 degrees of freedom (p = 0.605) that leaves the overidentifying restrictions intact.

---

## Common Pitfalls

1. **Carrying a conclusion across Hadri and the unit-root family.** Hadri's null is stationarity; the other four first-generation tests have a unit-root null. A small Hadri p-value and a small LLC p-value point in *opposite* directions. When both reject, the deterministic specification is usually wrong --- try `:trend` for series with a drift.

2. **Applying first-generation tests to macro panels.** Cross-sectional independence fails in virtually every country panel, and the summary table above shows five first-generation tests agreeing on the wrong answer. Unanimity across LLC, IPS, Breitung, and Fisher is not robustness when they share the assumption that fails. Run `pesaran_cips_test` before trusting any of them.

3. **Too few cross-sectional units.** Panel unit root tests rely on ``N \to \infty`` asymptotics. Below ``N = 20`` all eight lose size control and power. LLC and Breitung need ``N \geq 2`` to run at all, but running is not the same as being informative.

4. **Reading the panel-v statistic as left-tailed.** Six of Pedroni's seven statistics reject for very negative values; `panel-v` rejects for very positive ones. A p-value of 0.73 on `panel-v` alongside p-values below 0.001 on the other six, as in the example above, is agreement, not conflict.

5. **Unbalanced panels in the cointegration tests.** Pedroni, Kao, Westerlund, and Fisher-Johansen all require a common ``T`` across units and raise an `ArgumentError` otherwise. Balance the panel first --- by truncating to the common window or by imputing --- and be explicit about which, since the two choices have different implications for the long-run variance estimates.

6. **Instrument proliferation in the Panel VAR J-test.** The number of GMM instruments grows quadratically in ``T``. Once ``q`` approaches the number of groups the weighting matrix becomes singular, the J-statistic collapses to a number near the group count, and the p-value pins near 1 regardless of the model. Use `collapse=true` and `max_lag_endo` to keep ``q`` well below ``N``, and treat a non-rejection with ``q \gtrsim N`` as no evidence at all.

7. **Automatic factor selection swinging the answer.** The `:auto` option for `r` in PANIC and Moon-Perron uses the Bai-Ng IC2 criterion, which is sensitive to the signal-to-noise ratio. Re-run with `r=1`, `r=2`, and `r=3` and check that the conclusion is stable before reporting it.

---

## References

- Andrews, D. W. K., & Lu, B. (2001). Consistent model and moment selection procedures for GMM estimation with application to dynamic panel data models. *Journal of Econometrics*, 101(1), 123-164. [DOI](https://doi.org/10.1016/S0304-4076(00)00077-4)

- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191-221. [DOI](https://doi.org/10.1111/1468-0262.00273)

- Bai, J., & Ng, S. (2004). A PANIC attack on unit roots and cointegration. *Econometrica*, 72(4), 1127-1177. [DOI](https://doi.org/10.1111/j.1468-0262.2004.00528.x)

- Bai, J., & Ng, S. (2010). Panel unit root tests with cross-section dependence: A further investigation. *Econometric Theory*, 26(4), 1088-1114. [DOI](https://doi.org/10.1017/S0266466609990478)

- Breitung, J. (2000). The local power of some unit root tests for panel data. In B. Baltagi (Ed.), *Advances in Econometrics, Vol. 15* (pp. 161-178). JAI Press. [DOI](https://doi.org/10.1016/S0731-9053(00)15006-6)

- Choi, I. (2001). Unit root tests for panel data. *Journal of International Money and Finance*, 20(2), 249-272. [DOI](https://doi.org/10.1016/S0261-5606(00)00048-6)

- Dumitrescu, E.-I., & Hurlin, C. (2012). Testing for Granger non-causality in heterogeneous panels. *Economic Modelling*, 29(4), 1450-1460. [DOI](https://doi.org/10.1016/j.econmod.2012.02.014)

- Hadri, K. (2000). Testing for stationarity in heterogeneous panel data. *Econometrics Journal*, 3(2), 148-161. [DOI](https://doi.org/10.1111/1368-423X.00043)

- Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. *Econometrica*, 50(4), 1029-1054. [DOI](https://doi.org/10.2307/1912775)

- Im, K. S., Pesaran, M. H., & Shin, Y. (2003). Testing for unit roots in heterogeneous panels. *Journal of Econometrics*, 115(1), 53-74. [DOI](https://doi.org/10.1016/S0304-4076(03)00092-7)

- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551-1580. [DOI](https://doi.org/10.2307/2938278)

- Kao, C. (1999). Spurious regression and residual-based tests for cointegration in panel data. *Journal of Econometrics*, 90(1), 1-44. [DOI](https://doi.org/10.1016/S0304-4076(98)00023-2)

- Levin, A., Lin, C.-F., & Chu, C.-S. J. (2002). Unit root tests in panel data: Asymptotic and finite-sample properties. *Journal of Econometrics*, 108(1), 1-24. [DOI](https://doi.org/10.1016/S0304-4076(01)00098-7)

- Maddala, G. S., & Wu, S. (1999). A comparative study of unit root tests with panel data and a new simple test. *Oxford Bulletin of Economics and Statistics*, 61(S1), 631-652. [DOI](https://doi.org/10.1111/1468-0084.0610s1631)

- Moon, H. R., & Perron, B. (2004). Testing for a unit root in panels with dynamic factors. *Journal of Econometrics*, 122(1), 81-126. [DOI](https://doi.org/10.1016/j.jeconom.2003.10.020)

- Pedroni, P. (1999). Critical values for cointegration tests in heterogeneous panels with multiple regressors. *Oxford Bulletin of Economics and Statistics*, 61(S1), 653-670. [DOI](https://doi.org/10.1111/1468-0084.61.s1.14)

- Pedroni, P. (2004). Panel cointegration: Asymptotic and finite sample properties of pooled time series tests with an application to the PPP hypothesis. *Econometric Theory*, 20(3), 597-625. [DOI](https://doi.org/10.1017/S0266466604203073)

- Persyn, D., & Westerlund, J. (2008). Error-correction-based cointegration tests for panel data. *Stata Journal*, 8(2), 232-241. [DOI](https://doi.org/10.1177/1536867X0800800205)

- Pesaran, M. H. (2007). A simple panel unit root test in the presence of cross-section dependence. *Journal of Applied Econometrics*, 22(2), 265-312. [DOI](https://doi.org/10.1002/jae.951)

- Roodman, D. (2009). A note on the theme of too many instruments. *Oxford Bulletin of Economics and Statistics*, 71(1), 135-158. [DOI](https://doi.org/10.1111/j.1468-0084.2008.00542.x)

- Westerlund, J. (2007). Testing for error correction in panel data. *Oxford Bulletin of Economics and Statistics*, 69(6), 709-748. [DOI](https://doi.org/10.1111/j.1468-0084.2007.00477.x)
