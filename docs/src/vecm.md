# [Vector Error Correction Models](@id vecm_page)

**MacroEconometricModels.jl** provides full-featured Vector Error Correction Model (VECM) estimation for cointegrated ``I(1)`` systems. The VECM decomposes multivariate dynamics into long-run equilibrium relationships and short-run adjustment, making it the canonical framework for modeling nonstationary variables that share common stochastic trends.

- **Estimation**: Johansen (1991) reduced-rank MLE with automatic rank selection, and Engle-Granger (1987) two-step for bivariate systems
- **Rank selection**: Trace and maximum eigenvalue tests at user-specified significance levels
- **Deterministic specification**: None, constant, or linear trend in the cointegrating relation
- **VAR conversion**: Automatic conversion to VAR in levels for full structural analysis (Cholesky, sign restrictions, ICA, and all 18 identification methods)
- **Forecasting**: Direct VECM iteration preserving cointegrating relationships, with bootstrap and simulation confidence intervals
- **Granger causality**: Short-run, long-run, and strong (joint) causality decomposition
- **Restriction testing**: Johansen likelihood-ratio tests on ``\alpha`` and ``\beta``, including weak exogeneity
- **TimeSeriesData dispatch**: Pass `TimeSeriesData` objects directly --- variable names propagate automatically

A VECM is the right model only for variables that are individually ``I(1)`` and jointly cointegrated, so pretest first: [Unit Root & Cointegration](@ref tests_unitroot_page) covers the integration-order tests and the Johansen procedure, and [Residual-Based Cointegration Tests](@ref tests_cointegration_page) the Engle-Granger and Phillips-Ouliaris alternatives. If the series turn out to be stationary, estimate a [VAR](@ref var_page) instead; the [Bayesian VAR](@ref bvar_page) page covers shrinkage estimation of the same dynamics.

```@setup vecm
using MacroEconometricModels, Random
Random.seed!(42)
qd = load_example(:fred_qd)
Y = log.(to_matrix(qd[:, ["GDPC1", "PCECC96", "GPDIC1"]]))
Y = Y[all.(isfinite, eachrow(Y)), :]
```

## Quick Start

**Recipe 1: Estimate with automatic rank selection**

```@example vecm
# Automatic rank via Johansen trace test — selects r = 2 here
vecm = estimate_vecm(Y, 2)
report(vecm)
```

**Recipe 2: Explicit rank and deterministic specification**

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1, deterministic=:constant)
report(vecm)
```

**Recipe 3: Impulse responses via VAR conversion**

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)
irfs = irf(vecm, 20; method=:cholesky)
```

```julia
plot_result(irfs)
```

```@raw html
<iframe src="../assets/plots/vecm_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

**Recipe 4: Forecast with bootstrap confidence intervals**

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)
fc = forecast(vecm, 10; ci_method=:bootstrap, reps=50, conf_level=0.95)
report(fc)
```

**Recipe 5: Granger causality decomposition**

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)
g = granger_causality_vecm(vecm, 1, 2)  # GDP -> Consumption
report(g)
```

**Recipe 6: TimeSeriesData dispatch**

```@example vecm
# Construct TimeSeriesData from cleaned log-level data
ts = TimeSeriesData(Y; varnames=["GDPC1", "PCECC96", "GPDIC1"])

# Pass TimeSeriesData directly --- variable names propagate
vecm = estimate_vecm(ts, 2; rank=1)
report(vecm)
```

---

## Model Specification

The VECM reparameterizes a VAR(p) in levels to separate long-run equilibrium relationships from short-run dynamics. Consider a VAR(p) for an ``n``-dimensional ``I(1)`` vector ``y_t``:

```math
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t
```

where:
- ``y_t`` is the ``n \times 1`` vector of endogenous variables at time ``t``
- ``A_i`` are ``n \times n`` coefficient matrices for lag ``i = 1, \ldots, p``
- ``c`` is the ``n \times 1`` intercept vector
- ``u_t \sim N(0, \Sigma)`` are i.i.d. innovations

When the variables are cointegrated, the Granger representation theorem (Engle & Granger 1987) implies the system admits a **Vector Error Correction** representation:

```math
\Delta y_t = \alpha \beta' y_{t-1} + \Gamma_1 \Delta y_{t-1} + \cdots + \Gamma_{p-1} \Delta y_{t-p+1} + \mu + u_t
```

where:
- ``\Pi = \alpha \beta'`` is the ``n \times n`` **long-run matrix** with rank ``r``
- ``\alpha`` is the ``n \times r`` matrix of **adjustment coefficients** (loading matrix)
- ``\beta`` is the ``n \times r`` matrix of **cointegrating vectors**
- ``\Gamma_i = -(A_{i+1} + \cdots + A_p)`` are the ``n \times n`` **short-run dynamics** matrices
- ``\mu`` is the ``n \times 1`` intercept vector
- ``u_t \sim N(0, \Sigma)`` are i.i.d. innovations

### The Cointegrating Relationship

Each column ``\beta_j`` of ``\beta`` defines a stationary linear combination of the ``I(1)`` variables:

```math
z_{j,t} = \beta_j' y_t \sim I(0), \quad j = 1, \ldots, r
```

where:
- ``z_{j,t}`` is the ``j``-th **error correction term** (deviation from long-run equilibrium)
- ``\beta_j`` is the ``j``-th cointegrating vector

The corresponding column ``\alpha_j`` of ``\alpha`` governs the **speed of adjustment**: ``\alpha_{ij}`` measures how quickly variable ``i`` responds to deviations from the ``j``-th equilibrium. The **cointegrating rank** ``r`` determines the number of independent long-run equilibrium relationships. When ``r = 0``, there is no cointegration and the system reduces to a VAR in first differences. When ``r = n``, all variables are stationary in levels.

!!! note "Phillips Normalization"
    The package applies Phillips normalization to ``\beta`` so that the first ``r`` rows form an identity matrix. This ensures unique identification of the cointegrating vectors and makes the ``\alpha`` coefficients directly interpretable as adjustment speeds toward each equilibrium.

---

## Estimation

### Johansen Maximum Likelihood

The Johansen (1991) reduced-rank regression procedure estimates ``\alpha`` and ``\beta`` jointly via maximum likelihood. The algorithm proceeds in four steps:

1. **Concentrate out short-run dynamics** by regressing ``\Delta Y`` and ``Y_{t-1}`` on lagged differences ``Z = [\Delta Y_{t-1}, \ldots, \Delta Y_{t-p+1}, \mu]``
2. **Compute moment matrices** ``S_{00}``, ``S_{11}``, ``S_{01}`` from the concentrated residuals
3. **Solve the generalized eigenvalue problem** ``|\lambda S_{11} - S_{10} S_{00}^{-1} S_{01}| = 0``
4. **Extract** ``\beta`` from the first ``r`` eigenvectors and compute ``\alpha = S_{01} \beta (\beta' S_{11} \beta)^{-1}``

The eigenvalues ``\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n`` correspond to the canonical correlations between ``\Delta Y`` and ``Y_{t-1}`` after removing the short-run dynamics. The cointegrating rank ``r`` equals the number of statistically significant eigenvalues, determined by the trace test:

```math
\text{LR}_{\text{trace}}(r_0) = -T \sum_{i=r_0+1}^{n} \ln(1 - \hat{\lambda}_i)
```

where:
- ``T`` is the effective sample size
- ``\hat{\lambda}_i`` is the ``i``-th largest eigenvalue
- ``r_0`` is the null hypothesis rank

```@example vecm
# Automatic rank selection via Johansen trace test
vecm = estimate_vecm(Y, 2)
report(vecm)
```

```@example vecm
# Explicit rank specification
vecm = estimate_vecm(Y, 2; rank=1, varnames=["GDPC1", "PCECC96", "GPDIC1"])

# Different deterministic specifications
vecm_none = estimate_vecm(Y, 2; rank=1, deterministic=:none)       # No deterministic terms
vecm_const = estimate_vecm(Y, 2; rank=1, deterministic=:constant)  # Constant (default)
vecm_trend = estimate_vecm(Y, 2; rank=1, deterministic=:trend)     # Linear trend

round.([vec(vecm.beta) vec(vecm.alpha)], digits=4)
```

Left to itself the trace test selects ``r = 2`` on this system, so `estimate_vecm(Y, 2)` returns a rank-2 model. The examples below fix ``r = 1`` because a single cointegrating vector is easier to read: it normalizes to ``\log \text{GDP} + 0.372 \log C - 1.025 \log I``, a balanced-growth restriction tying output to the consumption-investment mix. The adjustment coefficients are ``\alpha = (-0.008, -0.016, 0.062)'``: output and consumption both fall when the relation is above equilibrium, and investment --- the fastest-adjusting and most volatile component --- moves in the opposite direction and about four times as hard. Johansen estimates ``\alpha`` and ``\beta`` jointly by maximum likelihood, so the estimates are efficient whatever the true rank.

### Rank Selection

`select_vecm_rank` exposes the rank decision on its own, using either the trace or the maximum-eigenvalue statistic:

```@example vecm
r_trace = select_vecm_rank(Y, 2; criterion=:trace, significance=0.05)
r_max = select_vecm_rank(Y, 2; criterion=:max_eigen)
(r_trace, r_max)
```

Both criteria agree on ``r = 2`` here. The sequence behind that verdict is visible in `johansen_test`: the trace statistics are 154.9, 26.1 and 8.6 against 5% critical values of 34.9, 20.0 and 9.2, so ``r_0 = 0`` and ``r_0 = 1`` are rejected and ``r_0 = 2`` is not. Disagreement between the two criteria is common in small samples and is the signal to inspect both statistics rather than trust either default.

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `criterion` | `Symbol` | `:trace` | Test statistic: `:trace` or `:max_eigen` |
| `significance` | `Real` | `0.05` | Significance level for critical values |
| `deterministic` | `Symbol` | `:constant` | Deterministic specification for Johansen test |

### Engle-Granger Two-Step

For bivariate systems with a single cointegrating relationship (``r = 1``), the Engle-Granger (1987) two-step estimator provides a simpler alternative:

1. **Step 1**: Estimate the cointegrating vector via static OLS regression of ``y_{1,t}`` on ``y_{2,t}, \ldots, y_{n,t}``
2. **Step 2**: Estimate the VECM equations using the OLS residuals as the error correction term

```@example vecm
vecm_eg = estimate_vecm(Y, 2; method=:engle_granger, rank=1)
round.([vec(vecm_eg.beta) vec(vecm_eg.alpha)], digits=4)
```

The two estimators disagree substantially here: Engle-Granger returns ``\beta = (1, -0.848, -0.068)'`` against Johansen's ``(1, 0.372, -1.025)'``, and loads the adjustment almost entirely onto output and investment. The static OLS regression in step 1 is superconsistent for ``\beta`` (Stock 1987), so both are valid estimates of *a* cointegrating vector, but with ``r = 2`` in this system the single Engle-Granger regression is picking one direction out of a two-dimensional cointegrating space --- which one depends on the normalization. Engle-Granger is consistent but inefficient for multivariate systems because it never optimizes the joint likelihood.

!!! warning "Engle-Granger supports rank=1 only"
    The Engle-Granger method estimates a single cointegrating vector via static OLS. For systems with multiple cointegrating relationships, use the Johansen method.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `rank` | `Union{Symbol,Int}` | `:auto` | Cointegrating rank; `:auto` selects via trace test |
| `deterministic` | `Symbol` | `:constant` | Deterministic terms: `:none`, `:constant`, `:trend` |
| `method` | `Symbol` | `:johansen` | Estimation method: `:johansen` or `:engle_granger` |
| `significance` | `Real` | `0.05` | Significance level for automatic rank selection |
| `varnames` | `Vector{String}` | `["y1", ...]` | Variable display names |

### Return Value

`estimate_vecm` returns a `VECMModel{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `Y` | `Matrix{T}` | Original data in levels (``T_{obs} \times n``) |
| `p` | `Int` | Underlying VAR order |
| `rank` | `Int` | Cointegrating rank ``r`` |
| `alpha` | `Matrix{T}` | Adjustment coefficients (``n \times r``) |
| `beta` | `Matrix{T}` | Cointegrating vectors (``n \times r``), Phillips-normalized |
| `Pi` | `Matrix{T}` | Long-run matrix ``\alpha\beta'`` (``n \times n``) |
| `Gamma` | `Vector{Matrix{T}}` | Short-run dynamics matrices ``[\Gamma_1, \ldots, \Gamma_{p-1}]`` |
| `mu` | `Vector{T}` | Intercept vector |
| `U` | `Matrix{T}` | Residuals (``T_{eff} \times n``) |
| `Sigma` | `Matrix{T}` | Residual covariance (``n \times n``) |
| `aic`, `bic`, `hqic` | `T` | Information criteria |
| `loglik` | `T` | Log-likelihood |
| `deterministic` | `Symbol` | Deterministic specification |
| `method` | `Symbol` | Estimation method used |
| `johansen_result` | `JohansenResult{T}` | Johansen test result (if applicable) |
| `varnames` | `Vector{String}` | Variable display names |

---

## VAR Conversion

The `to_var` function converts a VECM back to a VAR in levels, enabling all structural analysis methods. The mapping from VECM to VAR coefficients is:

```math
A_1 = \Pi + I_n + \Gamma_1, \quad A_i = \Gamma_i - \Gamma_{i-1} \text{ for } i = 2, \ldots, p-1, \quad A_p = -\Gamma_{p-1}
```

where:
- ``A_i`` is the ``i``-th VAR coefficient matrix in levels
- ``\Pi = \alpha\beta'`` is the long-run matrix
- ``I_n`` is the ``n \times n`` identity matrix
- ``\Gamma_i`` are the short-run dynamics matrices

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)
var_model = to_var(vecm)
round(is_stationary(var_model).max_modulus, digits=4)
```

The converted VAR has a largest companion eigenvalue modulus of exactly 1.0, so `is_stationary` returns `false` --- as it must. A cointegrated system with ``r < n`` has ``n - r`` unit roots by construction, and the conversion preserves them. That is not a diagnostic failure here; it is the reason the model was specified as a VECM in the first place. The consequence is that the level VAR must not be used for anything requiring stationarity, such as unconditional-moment calculations or the stationarity-filtered bootstrap.

The conversion matters because it makes all 18 identification methods (Cholesky, sign restrictions, ICA, narrative, and the rest) available to VECM models. `irf`, `fevd`, and `historical_decomposition` call `to_var()` internally, so `VECMModel` objects can be passed to them directly. The statistical (non-Gaussian) schemes are documented on the [Statistical Identification](@ref nongaussian_page) hub and its [Non-Gaussian Methods](@ref id_nongaussian_page) child.

---

## Innovation Accounting

All structural analysis functions accept `VECMModel` objects directly. The conversion to VAR in levels is handled automatically via `to_var()`.

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)

# Impulse response functions (Cholesky identification)
irfs = irf(vecm, 20; method=:cholesky)
report(irfs)
```

```@example vecm
# Forecast error variance decomposition
decomp = fevd(vecm, 20)
report(decomp)
```

```@example vecm
# Historical decomposition
T_eff = effective_nobs(to_var(vecm))
hd = historical_decomposition(vecm, T_eff)
verify_decomposition(hd)
```

```julia
plot_result(irfs)
plot_result(decomp)
plot_result(hd)
```

```@raw html
<iframe src="../assets/plots/vecm_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

```@raw html
<iframe src="../assets/plots/vecm_fevd.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Because the VAR is in levels, the impulse responses do not die out: the response of log GDP to its own shock is 0.0103 on impact and still 0.0114 at ``h = 20``, having settled onto a permanent level shift rather than returning to zero. That is the signature of the unit roots the cointegrated system carries. The variance decomposition reflects the same structure --- at ``h = 20`` GDP's forecast error variance is 94.7% own, 0.4% consumption and 4.9% investment, while investment is the most exposed variable at only 28.8% own. The historical decomposition identity holds across all 265 effective observations, which `verify_decomposition` confirms.

!!! warning "Permanent effects are not shock-by-shock"
    With ``r`` cointegrating vectors the long-run impact matrix ``C(1)`` has rank ``n - r``, so the
    space of permanent effects is ``(n-r)``-dimensional. That does **not** mean ``n - r`` of the
    Cholesky shocks are permanent and the rest transitory: an arbitrary rotation spreads the
    permanent component across all ``n`` shocks, which is why every response above levels off at a
    nonzero value. Isolating the permanent shocks requires a long-run identification scheme such as
    Blanchard-Quah --- see [Innovation Accounting](@ref innovation_accounting_page).

---

## Forecasting

VECM forecasting iterates the VECM equations directly in levels, preserving the cointegrating relationships in the forecast path. This approach is preferable to forecasting from the converted VAR because the error correction mechanism operates explicitly during each forecast step, pulling the system toward the long-run equilibrium:

```math
\hat{y}_{T+h} = \hat{y}_{T+h-1} + \alpha\beta'\hat{y}_{T+h-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta\hat{y}_{T+h-i} + \mu
```

where:
- ``\hat{y}_{T+h}`` is the ``h``-step-ahead forecast in levels
- ``\hat{y}_{T+h-1}`` is the previous forecast (or last observed value for ``h = 1``)
- ``\Delta\hat{y}_{T+h-i}`` are lagged forecast differences

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)

# Point forecast
fc = forecast(vecm, 10)
report(fc)
```

```@example vecm
# Bootstrap resamples the residuals; simulation draws from N(0, Σ̂)
fc_boot = forecast(vecm, 10; ci_method=:bootstrap, reps=50, conf_level=0.95)
fc_sim  = forecast(vecm, 10; ci_method=:simulation, reps=50)

# 95% band for log GDP at h = 10 under each scheme
round.([fc_boot.ci_lower[10, 1] fc_boot.ci_upper[10, 1];
        fc_sim.ci_lower[10, 1]  fc_sim.ci_upper[10, 1]], digits=4)
```

```julia
plot_result(fc_boot)
```

```@raw html
<iframe src="../assets/plots/forecast_vecm.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

Iterating in levels keeps the error-correction term active at every step, so the forecast path is pulled toward the long-run relation instead of drifting freely. From a final observation of ``\log \text{GDP} = 10.0869`` the model projects 10.0911 one quarter ahead and 10.1554 at ten, roughly 0.7% cumulative growth. The two interval schemes differ in what they resample: the bootstrap draws from the empirical residuals, simulation from ``N(0, \hat{\Sigma})``. At ``h = 10`` the bootstrap band spans ``[10.071, 10.261]`` and the simulation band ``[10.086, 10.236]``, so the bootstrap is about a quarter wider --- the empirical residuals carry non-Gaussian features that the covariance matrix alone does not reproduce. Both generate `reps` replicate paths and take pointwise quantiles, so both bands are themselves Monte Carlo estimates and shift a little from run to run at `reps=50`. The `differences` field carries the same path in first differences, seeded from the last observed level.

### Keyword Arguments

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `ci_method` | `Symbol` | `:none` | Confidence interval method: `:none`, `:bootstrap`, `:simulation` |
| `reps` | `Int` | `500` | Number of bootstrap or simulation replications |
| `conf_level` | `Real` | `0.95` | Confidence level for intervals |
| `rng` | `AbstractRNG` | `default_rng()` | Random number generator |

### Return Value

`forecast` returns a `VECMForecast{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `levels` | `Matrix{T}` | Forecasts in levels (``h \times n``); also returned by `point_forecast` |
| `differences` | `Matrix{T}` | Forecasts in first differences (``h \times n``) |
| `ci_lower` | `Matrix{T}` | Lower confidence interval bounds in levels (``h \times n``) |
| `ci_upper` | `Matrix{T}` | Upper confidence interval bounds in levels (``h \times n``) |
| `horizon` | `Int` | Forecast horizon |
| `ci_method` | `Symbol` | CI method used |
| `conf_level` | `T` | Confidence level |
| `varnames` | `Vector{String}` | Variable display names |

!!! note "Point forecasts live in `levels`"
    `VECMForecast` stores its point forecast in `levels`, not in a `forecast` field. The generic
    accessor `point_forecast(fc)` returns it, so code written against `AbstractForecastResult`
    works unchanged.

---

## Granger Causality

VECM Granger causality tests (Toda & Phillips 1993) decompose causal channels into **short-run** (through lagged differences ``\Gamma``) and **long-run** (through the error correction term ``\alpha\beta'y_{t-1}``) components. The **strong** test combines both channels in a single joint test.

```@example vecm
vecm = estimate_vecm(Y, 2; rank=1)

# Test: does GDP (var 1) Granger-cause Consumption (var 2)?
g = granger_causality_vecm(vecm, 1, 2)
report(g)
```

The three Wald tests are constructed as follows:

| Test | Null hypothesis | Mechanism |
|------|----------------|-----------|
| **Short-run** | ``\Gamma_i[\text{effect}, \text{cause}] = 0`` for all ``i`` | Causality through lagged differences |
| **Long-run** | ``\alpha[\text{effect}, :] = 0`` | Causality through error correction |
| **Strong** | Joint test of both restrictions | Combined short-run and long-run causality |

Each test reports a Wald ``\chi^2`` statistic, degrees of freedom, and p-value. Testing GDP against consumption gives a short-run statistic of 3.23 on 1 degree of freedom (``p = 0.073``), a long-run statistic of 5.03 (``p = 0.025``), and a joint statistic of 7.19 on 2 degrees of freedom (``p = 0.028``). The reading is specific: past *changes* in GDP add little to predicting consumption growth once the equilibrium term is in the equation, but consumption does error-correct significantly toward the long-run relation. Causality here runs through the equilibrium channel, not the lag structure --- a distinction a standard VAR Granger test, which only sees the ``\Gamma`` block, cannot make. Note that the long-run test restricts the whole ``\alpha`` row of the effect variable, so it is a statement about that variable's adjustment rather than about the named cause.

### Return Value

`granger_causality_vecm` returns a `VECMGrangerResult{T}` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `short_run_stat` | `T` | Wald ``\chi^2`` for short-run test |
| `short_run_pvalue` | `T` | P-value for short-run test |
| `short_run_df` | `Int` | Degrees of freedom for short-run test |
| `long_run_stat` | `T` | Wald ``\chi^2`` for long-run test |
| `long_run_pvalue` | `T` | P-value for long-run test |
| `long_run_df` | `Int` | Degrees of freedom for long-run test |
| `strong_stat` | `T` | Wald ``\chi^2`` for joint test |
| `strong_pvalue` | `T` | P-value for joint test |
| `strong_df` | `Int` | Degrees of freedom for joint test |
| `cause_var` | `Int` | Index of the cause variable |
| `effect_var` | `Int` | Index of the effect variable |

---

## Restriction Testing on the Cointegrating Structure

The central empirical use of a VECM is testing economic hypotheses on the long-run parameters ``\alpha`` and ``\beta`` via Johansen likelihood-ratio tests. Each test re-solves the reduced-rank eigenproblem on transformed product-moment matrices and forms

```math
\mathrm{LR} = T \sum_{i=1}^{r} \ln\!\frac{1-\lambda^{*}_i}{1-\lambda_i} \;\sim\; \chi^2(\mathrm{df}),
```

where ``\lambda_i`` are the unrestricted eigenvalues and ``\lambda^{*}_i`` the restricted ones. The classic testbed is the [`load_example(:denmark)`](@ref) Danish money-demand data (Johansen & Juselius 1990) --- real money `LRM`, real income `LRY`, the bond rate `IBO` and the deposit rate `IDE` --- estimated as a rank-1 VECM on 53 effective quarterly observations. Throughout this section ``n`` is the number of variables (4 here) and ``r`` the cointegrating rank.

```@example vecm
dk = load_example(:denmark)
Ydk = to_matrix(dk[:, ["LRM", "LRY", "IBO", "IDE"]])
mdk = estimate_vecm(Ydk, 2; rank=1, varnames=["LRM", "LRY", "IBO", "IDE"])
round.([vec(mdk.beta) vec(mdk.alpha)], digits=4)
```

The unrestricted long-run relation normalizes to ``\text{LRM} - 0.976\,\text{LRY} + 5.409\,\text{IBO} - 4.162\,\text{IDE}``: a near-unit income elasticity of money demand, with the two interest rates entering with opposite signs and similar magnitudes. Only money adjusts appreciably, at ``\alpha_{\text{LRM}} = -0.28``. Both features are hypotheses the tests below can make precise.

### ``\beta`` restricted to a known space

[`test_beta_restriction(m, H)`](@ref) tests ``\beta = H\varphi`` (the same restriction on every cointegrating vector), where `H` is ``n \times s`` with ``s \ge r``. The degrees of freedom are ``\mathrm{df} = r(n-s)``. Here we test that the bond and deposit rates enter the long-run relation only through their spread (``\beta_{\mathrm{IBO}} = -\beta_{\mathrm{IDE}}``):

```@example vecm
H = Float64[1 0 0; 0 1 0; 0 0 1; 0 0 -1]   # imposes IBO = -IDE
rb = test_beta_restriction(mdk, H)
report(rb)
```

The spread restriction is not rejected: ``\mathrm{LR} = 1.30`` on ``\mathrm{df} = r(n-s) = 1 \times (4-3) = 1``, ``p = 0.254``. Imposing it barely moves the income elasticity (``-0.976`` to ``-0.970``) and replaces the two separate rate coefficients with a single spread coefficient of 6.15. The data are therefore consistent with money demand responding to the bond-deposit spread rather than to the two rates independently, which is the standard opportunity-cost specification.

### ``\alpha`` restricted and weak exogeneity

[`test_alpha_restriction(m, A)`](@ref) tests ``\alpha = A\psi`` (``\mathrm{df} = r(n-a)``, where `A` is ``n \times a``). Its headline special case is [`test_weak_exogeneity(m, vars)`](@ref): the named variables have zero rows in ``\alpha``, so they do not error-correct toward the long-run equilibrium — the central-bank question *"is the policy variable weakly exogenous for the long run?"* With ``m`` = number of tested variables, ``\mathrm{df} = r\cdot m``.

```@example vecm
we = test_weak_exogeneity(mdk, "LRY")   # is real income weakly exogenous?
report(we)
```

Real income is weakly exogenous for the long-run parameters: ``\mathrm{LR} = 0.32`` on 1 degree of freedom, ``p = 0.574``, nowhere near rejection. Income does not adjust to money-demand disequilibrium, so conditioning on it loses no information about ``\beta`` and a single-equation money-demand model is legitimate. The contrast is instructive --- running the same test on `LRM` gives ``\mathrm{LR} = 12.11``, ``p = 0.0005``, decisively rejecting weak exogeneity of money. Money is the variable that does the error-correcting, exactly as its large negative ``\alpha`` suggested.

### Fully known ``\beta`` and joint restrictions

[`test_known_beta(m, b)`](@ref) tests a completely specified cointegrating space ``\beta = b`` (``\mathrm{df} = r(n-r)``), and [`test_joint_restriction(m, H, A)`](@ref) tests ``\beta = H\varphi`` **and** ``\alpha = A\psi`` jointly via the Johansen–Juselius switching algorithm (``\mathrm{df}`` is the sum of the individual restriction counts).

```@example vecm
b = reshape(Float64[1, -1, 0, 0], 4, 1)   # unit income elasticity, no rate terms
rk = test_known_beta(mdk, b)
(lr = rk.lr_stat, df = rk.df, pvalue = rk.pvalue)
```

```@example vecm
# β in span(H) AND only LRM, IBO, IDE error-correct
A = Float64[1 0 0; 0 0 0; 0 1 0; 0 0 1]
rj = test_joint_restriction(mdk, H, A)
(lr = rj.lr_stat, df = rj.df, pvalue = rj.pvalue, converged = rj.converged)
```

Dropping the interest rates entirely is decisively rejected: constant velocity with a unit income elasticity gives ``\mathrm{LR} = 29.35`` on ``\mathrm{df} = r(n-r) = 3``, ``p < 10^{-5}``. Combining the two restrictions that individually survived --- the interest-rate spread in ``\beta`` and weak exogeneity of income in ``\alpha`` --- is not rejected either: the switching algorithm converges to ``\mathrm{LR} = 1.74`` on 2 degrees of freedom, ``p = 0.419``. Jointly imposing them costs almost nothing in fit, which is the case for reporting the restricted system rather than the unrestricted one.

Every test returns a `VECMRestrictionTest{T}` carrying the LR statistic, degrees of freedom, p-value, and a re-estimated `restricted_model::VECMModel` that imposes ``H_0``, so `irf`, `fevd`, and `historical_decomposition` all run on the restricted system:

```@example vecm
irf(rb.restricted_model, 8)   # IRFs from the spread-restricted VECM
nothing # hide
```

!!! note "Non-binding restrictions"
    The non-binding case ``H = I_p`` (``s = p``) returns ``\mathrm{LR} \approx 0`` with ``\mathrm{df} = 0``; likewise `test_known_beta(m, m.beta)` returns ``\mathrm{LR} \approx 0``. These are useful sanity checks.

---

## Complete Example

This example demonstrates the full VECM workflow: cointegration testing, estimation, structural analysis, forecasting, and Granger causality.

```@example vecm
# Step 1: Test for cointegration
joh = johansen_test(Y, 2)
report(joh)

# Step 2: Estimate VECM with rank 1
vecm = estimate_vecm(Y, 2; rank=1)
report(vecm)

# Step 3: Impulse responses (Cholesky identification)
irfs = irf(vecm, 20; method=:cholesky)

# Step 4: Forecast with bootstrap CIs
fc = forecast(vecm, 10; ci_method=:bootstrap, reps=50)
report(fc)

# Step 5: Granger causality --- does GDP Granger-cause Consumption?
g = granger_causality_vecm(vecm, 1, 2)
report(g)

# Step 6: Convert to VAR for FEVD
var_model = to_var(vecm)
decomp = fevd(var_model, 20)
```

```julia
plot_result(irfs)
plot_result(fc)
plot_result(decomp)
```

```@raw html
<iframe src="../assets/plots/vecm_irf.html" width="100%" height="500" frameborder="0" style="border:1px solid #ddd;border-radius:4px;"></iframe>
```

The Johansen test rejects ``r_0 = 0`` and ``r_0 = 1`` but not ``r_0 = 2``, so the system carries two cointegrating relations and one common stochastic trend; the workflow then fixes ``r = 1`` for a single interpretable long-run relation between output, consumption and investment. A cointegrating relationship ``\beta'y_t \sim I(0)`` is what balanced-growth theory predicts for these three series. The adjustment coefficients ``\alpha = (-0.008, -0.016, 0.062)'`` show consumption contracting when the system sits above equilibrium, with investment doing most of the correcting in the opposite direction. The Granger test then separates the two channels: consumption's response to GDP works through error correction (``p = 0.025``) rather than through lagged differences (``p = 0.073``), a decomposition a standard VAR-based Granger test cannot deliver because it never sees the equilibrium term.

---

## Common Pitfalls

1. **Incorrect cointegrating rank**: Specifying ``r`` too high introduces near-unit-root stationary components that contaminate the short-run dynamics. Specifying ``r`` too low discards genuine equilibrium relationships. Always run `johansen_test` first and examine both the trace and maximum eigenvalue statistics before fixing ``r`` manually.

2. **I(2) data passed without differencing**: The VECM framework assumes all variables are ``I(1)``. Passing ``I(2)`` data (e.g., price levels that need double differencing) produces spurious cointegrating vectors. Test each series with `adf_test` or `kpss_test` before estimation and difference any ``I(2)`` variables once to bring them to ``I(1)``.

3. **Too many lags in levels**: The underlying VAR order ``p`` determines the number of lagged differences ``p - 1`` in the VECM. Over-parameterization wastes degrees of freedom and inflates estimation uncertainty, especially in small samples. Use `select_lag_order` on the levels data or compare `aic`/`bic` across candidate orders.

4. **Misinterpreting the Johansen trace test**: The sequential testing procedure starts from ``r_0 = 0`` and increments until the trace statistic falls below the critical value. Rejecting ``r_0 = 0`` but not ``r_0 = 1`` implies exactly one cointegrating vector. The trace test has well-known size distortions in small samples; a small-sample Bartlett correction of the trace statistic mitigates this, or use a more conservative significance level.

5. **Engle-Granger with multiple cointegrating vectors**: The Engle-Granger two-step method estimates only a single cointegrating vector via static OLS. Applying it to a system with ``r > 1`` recovers at most one linear combination and discards the remaining equilibrium relationships. Use the Johansen method for systems with multiple cointegrating vectors.

6. **Forgetting VAR conversion for structural analysis**: `irf`, `fevd`, and `historical_decomposition` dispatch through `to_var()` automatically, so passing a `VECMModel` directly works. However, if you need the VAR coefficient matrices explicitly (e.g., for custom identification schemes), call `to_var(vecm)` and work with the resulting `VARModel`.

---

## References

- Engle, R. F., & Granger, C. W. J. (1987). Co-Integration and Error Correction: Representation, Estimation, and Testing.
  *Econometrica*, 55(2), 251-276. [DOI](https://doi.org/10.2307/1913236)

- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models.
  *Econometrica*, 59(6), 1551-1580. [DOI](https://doi.org/10.2307/2938278)

- Johansen, S., & Juselius, K. (1990). Maximum Likelihood Estimation and Inference on Cointegration --- With Applications to the Demand for Money.
  *Oxford Bulletin of Economics and Statistics*, 52(2), 169-210. [DOI](https://doi.org/10.1111/j.1468-0084.1990.mp52002003.x)

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
  Berlin: Springer. ISBN 978-3-540-40172-8. [DOI](https://doi.org/10.1007/978-3-540-27752-1)

- Osterwald-Lenum, M. (1992). A Note with Quantiles of the Asymptotic Distribution of the Maximum Likelihood Cointegration Rank Test Statistics.
  *Oxford Bulletin of Economics and Statistics*, 54(3), 461-472. [DOI](https://doi.org/10.1111/j.1468-0084.1992.tb00013.x)

- Stock, J. H. (1987). Asymptotic Properties of Least Squares Estimators of Cointegrating Vectors.
  *Econometrica*, 55(5), 1035-1056. [DOI](https://doi.org/10.2307/1911260)

- Toda, H. Y., & Phillips, P. C. B. (1993). Vector Autoregressions and Causality.
  *Econometrica*, 61(6), 1367-1393. [DOI](https://doi.org/10.2307/2951647)
